import torch
import torch.functional as F
import torch.distributed as dist
import numpy as np
from lightllm.common.basemodel.layer_weights.base_layer_weight import BaseLayerWeight
from lightllm.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo

from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from einops import rearrange
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.models.llama.triton_kernel.rmsnorm import rmsnorm_forward
from lightllm.common.basemodel import PostLayerInferTpAndPpl
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.server.router.model_infer.infer_batch import get_pair_groups, get_all_reduce_groups

class LlamaPostLayerInfer(PostLayerInferTpAndPpl):
    """
    """

    def __init__(self, tp_rank, world_size, network_config, mode, pp_rank = 0, pp_size = 1, tp_size=0, post_rank = 0):
        if tp_size == 0:
            tp_size = world_size
        super().__init__(tp_rank, world_size, network_config, mode, pp_rank, pp_size, tp_size, post_rank)
        self.eps_ = network_config["rms_norm_eps"]
        self.vocab_size_ = network_config["vocab_size"]
        self.embed_dim_ = network_config["n_embed"]
        self.gpu_rank_ = tp_rank * pp_size + pp_rank
        self.length_list_ = []   # 广播未完成前，tensor不能删除，否则会引发未定义的行为
        self.shape_list_ = []
        self.input_embeddings_list_ = []
        return
    
    def _norm(self, input, infer_state, layer_weight:LlamaPreAndPostLayerWeight) -> torch.Tensor:
        return rmsnorm_forward(input, layer_weight.final_norm_weight_, eps=self.eps_)
    
    def _slice_get_last_input(self, input_embdings, infer_state: LlamaInferStateInfo):
        if infer_state.is_splitfuse:
            # for SplitFuse
            batch_size = infer_state.batch_size
            last_input = torch.empty((batch_size, self.embed_dim_), device=input_embdings.device, dtype=torch.float16)
            tmp_ = torch.cat([torch.ones(infer_state.decode_req_num, dtype=torch.int32, device="cuda"), infer_state.prefill_b_split_seq_len], dim=0)
            last_index = torch.cumsum(tmp_, dim=0, dtype=torch.long) - 1
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, batch_size
        
        if not infer_state.is_splitfuse and infer_state.is_prefill and not infer_state.return_all_prompt_logprobs:
            batch_size = infer_state.batch_size
            last_input = torch.empty((batch_size, self.embed_dim_), device=input_embdings.device, dtype=torch.float16)
            last_index = torch.cumsum(infer_state.b_seq_len, dim=0, dtype=torch.long) - 1
            last_input[:, :] = input_embdings[last_index, :]
            return last_input, batch_size

        if not infer_state.is_splitfuse and infer_state.is_prefill and infer_state.return_all_prompt_logprobs:
            total_tokens = infer_state.total_token_num
            return input_embdings, total_tokens
        
        if not infer_state.is_splitfuse and not infer_state.is_prefill:
            batch_size = infer_state.batch_size
            return input_embdings[-batch_size:, :], batch_size
        
        assert False, "Error State"


    def soft_max(self, data):
        return torch.softmax(data.permute(1, 0).float(), dim=-1)

    def token_forward(self, input_embdings, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight, return_logics=False):
        last_input, token_num = self._slice_get_last_input(input_embdings, infer_state)
        input_embdings = None
        last_input = self._norm(last_input, infer_state, layer_weight)
        last_input = rearrange(last_input, "batch embed_dim -> embed_dim batch").contiguous().reshape(-1, token_num)
        logic_batch = torch.mm(layer_weight.lm_head_weight_, last_input)

        last_input = None
        if self.tp_size_ == 1:
            gather_data = logic_batch
        else:
            gather_data = torch.empty((self.vocab_size_, token_num), device=logic_batch.device, dtype=torch.float16)
            split_size = self.vocab_size_ // self.tp_size_
            dist.all_gather([gather_data[i * split_size: (i + 1) * split_size, :]
                            for i in range(self.tp_size_)], logic_batch, group=get_all_reduce_groups(self.gpu_rank_), async_op=False)
        logic_batch = None

        if not return_logics:
            prob_out = self.soft_max(gather_data)
            gather_data = None
            return prob_out
        else:
            ans_logics = gather_data.permute(1, 0).float()
            gather_data = None
            return ans_logics
    
    
    def filter_out_complete_broadcast(self):
        self.length_list_ = [(length, req) for length, req in self.length_list_ if not req.is_completed()]
        self.shape_list_ = [(shape, req) for shape, req in self.shape_list_ if not req.is_completed()]
        self.input_embeddings_list_ = [(input_embeddings, req) for input_embeddings, req in self.input_embeddings_list_ if not req.is_completed()]
    
    def pipeline_model_parallel_send_tensor_list(self, input_embdings):
        self.filter_out_complete_broadcast()
        src_rank = self.tp_rank_ * self.pp_size_ + self.pp_rank_
        dst_rank = self.post_rank_
        pair_group = get_pair_groups(src_rank, dst_rank)
        length = torch.tensor(len(input_embdings.shape), dtype=torch.int).cuda(torch.cuda.current_device())
        req1 = torch.distributed.broadcast(length, src=src_rank, group=pair_group, async_op=True)
        self.length_list_.append((length, req1))
        
        shape = torch.tensor(input_embdings.shape, dtype=torch.int).cuda(torch.cuda.current_device())
        req2 = torch.distributed.broadcast(shape, src=src_rank, group=pair_group, async_op=True)
        self.shape_list_.append((shape, req2))
        
        req3 = torch.distributed.broadcast(input_embdings, src=src_rank, group=pair_group, async_op=True)
        self.input_embeddings_list_.append((input_embdings, req3))
        return input_embdings
        
    
    # @mark_cost_time("splitfuse post forward")
    def splitfuse_forward(self, input_embdings, infer_state: SplitFuseInferStateInfo, layer_weight: BaseLayerWeight, return_logics=False):
        
        if self.pp_size_ >= 1 and self.pp_rank_ != self.pp_size_ - 1:
            self.pipeline_model_parallel_send_tensor_list(input_embdings)
            return None
        else:
            return self.token_forward(input_embdings, infer_state, layer_weight, return_logics=return_logics)