import torch
import torch.distributed as dist

from lightllm.common.basemodel.splitfuse_infer_struct import SplitFuseInferStateInfo
from lightllm.models.llama.layer_weights.pre_and_post_layer_weight import LlamaPreAndPostLayerWeight
from lightllm.models.llama.infer_struct import LlamaInferStateInfo
from lightllm.common.basemodel import PreLayerInferTpAndPpl
from lightllm.utils.infer_utils import mark_cost_time
from lightllm.server.router.model_infer.infer_batch import get_pair_groups, get_all_reduce_groups


class LlamaPreLayerInfer(PreLayerInferTpAndPpl):
    """
    """

    def __init__(self, tp_rank, world_size, network_config, mode, pp_rank = 0, pp_size = 1, tp_size=0, pre_rank = None):
        if tp_size == 0:
            tp_size = world_size
        super().__init__(tp_rank, world_size, network_config, mode, pp_rank, pp_size, tp_size, pre_rank)
        tp_vocab_size_ = network_config["vocab_size"] // self.tp_size_
        self.vob_start_id_ = tp_vocab_size_ * self.tp_rank_
        self.vob_end_id_ = tp_vocab_size_ * (self.tp_rank_ + 1)
        self.gpu_rank_ = tp_rank * pp_size + pp_rank
        return

    @mark_cost_time("pre context forward")
    def context_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.tp_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False, group=get_all_reduce_groups(self.gpu_rank_))
        return input_embdings

    def token_forward(self, input_ids, infer_state: LlamaInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        input_mask = torch.logical_or(self.vob_start_id_ > input_ids, input_ids >= self.vob_end_id_)
        tmp_input_ids = (input_ids - self.vob_start_id_)
        tmp_input_ids[input_mask] = 0
        input_embdings = torch.embedding(layer_weight.wte_weight_, tmp_input_ids, padding_idx=-1)
        input_embdings[input_mask] = 0.0
        if self.tp_size_ > 1:
            dist.all_reduce(input_embdings, op=dist.ReduceOp.SUM, async_op=False, group=get_all_reduce_groups(self.gpu_rank_))
        return input_embdings
    
    
    def pipeline_model_parallel_recv_tensor(self):
        src_rank = self.pre_rank_
        dst_rank = self.tp_rank_ * self.pp_size_ + self.pp_rank_
        pair_group = get_pair_groups(src_rank, dst_rank)
        assert pair_group is not None, "wrong: pair group is None"

        length = torch.empty(1, dtype=torch.int).cuda(torch.cuda.current_device())
        torch.distributed.broadcast(length, src=src_rank, group=pair_group)

        shape = torch.empty(length.tolist(), dtype=torch.int).cuda(torch.cuda.current_device())
        torch.distributed.broadcast(shape, src=src_rank, group=pair_group)

        input_embeddings = torch.empty(shape.tolist(), dtype=torch.float16, device=torch.cuda.current_device())
        torch.distributed.broadcast(input_embeddings, src=src_rank, group=pair_group)

        return input_embeddings
    

    # @mark_cost_time("splitfuse forward")
    def splitfuse_forward(self, input_ids, infer_state: SplitFuseInferStateInfo, layer_weight: LlamaPreAndPostLayerWeight):
        if self.pp_size_ is None or self.pp_rank_ == 0:
            return self.token_forward(input_ids, infer_state, layer_weight)
        else:
            return self.pipeline_model_parallel_recv_tensor()
    
