import asyncio
import numpy as np
import rpyc
import torch
import traceback
from datetime import timedelta
from typing import Dict, List, Tuple
from transformers.configuration_utils import PretrainedConfig
from lightllm.models.mixtral.model import MixtralTpPartModel
from lightllm.server.router.model_infer.infer_batch import InferBatch
from rpyc.utils.classic import obtain

from lightllm.models.bloom.model import BloomTpPartModel
from lightllm.models.llama.model import LlamaTpPartModel
from lightllm.models.llama_wquant.model import LlamaTpPartModelWQuant
from lightllm.models.llama_awquant.model import LlamaTpPartModelAWQuant
from lightllm.models.starcoder.model import StarcoderTpPartModel
from lightllm.models.starcoder_wquant.model import StarcoderTpPartModelWQuant
from lightllm.models.qwen.model import QWenTpPartModel
from lightllm.models.qwen_wquant.model import QWenTpPartModelWQuant
from lightllm.models.baichuan7b.model import Baichuan7bTpPartModel
from lightllm.models.baichuan13b.model import Baichuan13bTpPartModel
from lightllm.models.baichuan2_7b.model import Baichuan2_7bTpPartModel
from lightllm.models.baichuan2_13b.model import Baichuan2_13bTpPartModel
from lightllm.models.chatglm2.model import ChatGlm2TpPartModel
from lightllm.models.internlm.model import InternlmTpPartModel
from lightllm.models.internlm_wquant.model import InternlmTpPartModelWQuant
from lightllm.models.yi.model import YiTpPartModel
from lightllm.models.mistral.model import MistralTpPartModel
from lightllm.models.llava.model import LlavaTpPartModel
from lightllm.models.qwen_vl.model import QWenVLTpPartModel
from lightllm.utils.infer_utils import set_random_seed
from lightllm.utils.infer_utils import calculate_time, mark_start, mark_end
from .pre_process import prepare_decode_inputs, prepare_prefill_inputs, splitfuse_prepare_decode_inputs
from .post_process import sample
from .infer_batch import requests_mapping
from .infer_batch import InferReq
from lightllm.server.io_struct import ReqRunStatus
from lightllm.utils.log_utils import init_logger
from .infer_batch import pair_groups, all_reduce_groups
import time
import sys
import json
import pickle


class ModelRpcServer(rpyc.Service):

    def exposed_init_model(self, kvargs):
        import torch
        import torch.distributed as dist
        self.add_token_all_times = 0
        world_size = kvargs["world_size"]
        if world_size != 1:
            kvargs = obtain(kvargs)
            world_size = kvargs["world_size"]

        self.is_multimodal = False
        self.tp_rank = kvargs["rank_id"]
        self.world_size = kvargs["world_size"]
        self.load_way = kvargs["load_way"]
        self.mode = kvargs["mode"]
        self.is_splitfuse_mode = kvargs.get("is_splitfuse_mode", False)
        self.splitfuse_block_size = kvargs.get("splitfuse_block_size", None)
        self.return_all_prompt_logprobs = kvargs.get("return_all_prompt_logprobs", False)
        self.pp_rank = kvargs.get("pp_rank", 0)
        self.pp_size = kvargs.get("pp_size", 1)
        self.tp_size = kvargs.get("tp_size", 0)
        self.pre_and_post_rank = kvargs.get("pre_and_post_rank", [-1, -1])
        self.all_reduce_rank = kvargs.get("all_reduce_rank", [])
        self.gpu_rank = self.tp_rank * self.pp_size + self.pp_rank

        self.cache = {}
        self.logger = init_logger(__name__)

        weight_dir = kvargs["weight_dir"]
        max_total_token_num = kvargs["max_total_token_num"]
        if (self.pp_size > 1):
            gpu_rank = self.gpu_rank
            backend = 'nccl'
            self.logger.info(f"pre {self.pre_and_post_rank[0]} and post {self.pre_and_post_rank[1]}")
            self.logger.info(f"tp_rank: {self.tp_rank}, pp_rank: {self.pp_rank}, gpu_rank: {gpu_rank}, world_size: {world_size}, pre_rank: {self.pre_and_post_rank[0]}, post_rank: {self.pre_and_post_rank[1]}")
            dist.init_process_group(backend, init_method=f'tcp://127.0.0.1:{kvargs["nccl_port"]}', rank=gpu_rank, world_size=world_size)
            
            # 构造tp的组
            for pp_rank in range(self.pp_size):
                gpu_rank_list = [tp_rank * self.pp_size + pp_rank for tp_rank in range(self.tp_size)]
                last_rank = None
                for new_rank in gpu_rank_list:
                    if last_rank is None:
                        all_reduce_groups[new_rank] = torch.distributed.new_group(ranks=gpu_rank_list, backend='nccl')
                    else:
                        all_reduce_groups[new_rank] = all_reduce_groups[last_rank] 
                    last_rank = new_rank
            
            # 构造pp的组
            for tp_rank in range(self.tp_size):
                for pp_rank in range(self.pp_size):
                    new_rank = tp_rank * self.pp_size + pp_rank
                    pre_rank = tp_rank * self.pp_size + pp_rank - 1 if pp_rank > 0 else None
                    next_rank = tp_rank * self.pp_size + pp_rank + 1 if pp_rank < self.pp_size - 1 else None
                    if pre_rank is not None:
                        pair_groups[(new_rank, pre_rank)] = pair_groups[(pre_rank, new_rank)]
                    if next_rank is not None:
                        pair_groups[(new_rank, next_rank)] = torch.distributed.new_group(ranks=[new_rank, next_rank], backend='nccl')
                
            torch.cuda.set_device(gpu_rank)
        else:
            self.logger.info(f"tp_rank: {self.tp_rank},  world_size: {world_size}")
            dist.init_process_group('nccl', init_method=f'tcp://127.0.0.1:{kvargs["nccl_port"]}', rank=self.tp_rank, world_size=world_size)
            torch.cuda.set_device(self.tp_rank)

        model_cfg, _ = PretrainedConfig.get_config_dict(
            weight_dir
        )

        model_kvargs = {
            "tp_rank": self.tp_rank,
            "tp_size": self.tp_size,
            "world_size": self.world_size,
            "weight_dir": weight_dir,
            "max_total_token_num": max_total_token_num,
            "load_way": self.load_way,
            "mode": self.mode,
            "max_req_num": kvargs.get("max_req_num", 1000),
            "max_seq_length": kvargs.get("max_seq_length", 1024 * 5),
            "return_all_prompt_logprobs": self.return_all_prompt_logprobs,
            "pp_rank": self.pp_rank,
            "pp_size": self.pp_size,
            "pre_and_post_rank": self.pre_and_post_rank
        }

        try:
            self.model_type = model_cfg["model_type"]
            if self.model_type == "bloom":
                self.model = BloomTpPartModel(model_kvargs)
            elif self.model_type == "llama":
                if any('int8weight' in mode_ or 'int4weight' in mode_ for mode_ in self.mode):
                    self.model = LlamaTpPartModelWQuant(model_kvargs)
                elif any('int8_activation_weight' in mode_ for mode_ in self.mode):
                    self.model = LlamaTpPartModelAWQuant(model_kvargs)
                else:
                    self.model = LlamaTpPartModel(model_kvargs)
            elif self.model_type == "qwen":
                if "visual" in model_cfg:
                    self.model = QWenVLTpPartModel(model_kvargs)
                    self.is_multimodal = True
                elif any('int8weight' in mode_ or 'int4weight' in mode_ for mode_ in self.mode):
                    self.model = QWenTpPartModelWQuant(model_kvargs)
                else:
                    self.model = QWenTpPartModel(model_kvargs)
            elif self.model_type == "baichuan":
                if model_cfg['hidden_size'] == 4096:
                    if model_cfg['architectures'][0] == 'BaichuanForCausalLM':
                        self.model = Baichuan2_7bTpPartModel(model_kvargs)
                    else:
                        self.model = Baichuan7bTpPartModel(model_kvargs)
                elif model_cfg["hidden_size"] == 5120:
                    if model_cfg['architectures'][0] == 'BaichuanForCausalLM':
                        self.model = Baichuan2_13bTpPartModel(model_kvargs)
                    else:
                        self.model = Baichuan13bTpPartModel(model_kvargs)
                else:
                    raise Exception('can not support baichuan format')
            elif self.model_type == 'gpt_bigcode':
                if any('int8weight' in mode_ or 'int4weight' in mode_ for mode_ in self.mode):
                    self.model = StarcoderTpPartModelWQuant(model_kvargs)
                else:
                    self.model = StarcoderTpPartModel(model_kvargs)
            elif self.model_type == 'chatglm':
                self.model = ChatGlm2TpPartModel(model_kvargs)
            elif self.model_type == 'internlm':
                if any('int8weight' in mode_ or 'int4weight' in mode_ for mode_ in self.mode):
                    self.model = InternlmTpPartModelWQuant(model_kvargs)
                else:
                    self.model = InternlmTpPartModel(model_kvargs)
            elif self.model_type == "Yi":
                self.model = YiTpPartModel(model_kvargs)
            elif self.model_type == "mistral":
                self.model = MistralTpPartModel(model_kvargs)
            elif self.model_type == "mixtral":
                self.model = MixtralTpPartModel(model_kvargs)
            elif self.model_type == "llava":
                self.model = LlavaTpPartModel(model_kvargs)
                self.is_multimodal = True
            else:
                raise Exception(f"can not support {self.model_type} now")
        except Exception as e:
            self.logger.error(f"load model error: {str(e)} {e} {type(e)}")
            import traceback
            traceback.print_exc()
            raise e
        # self.all_add_batch_time = 0
        # self.all_filter_batch_time = 0
        # self.all_remove_batch_time = 0
        # self.all_merge_batch_time = 0
        # self.all_pause_batch_time = 0
        # self.all_after_process_time = 0
        # self.all_sample_time = 0
        set_random_seed(2147483647)
        return
    
    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_add_batch(self, batch_id, reqs, dtype):
        # start_time = time.perf_counter()
        if self.world_size != 1:
            batch_id, reqs, dtype = obtain(batch_id), obtain(reqs), obtain(dtype)
        import torch
        if dtype == "fp16":
            dtype = torch.float16
        else:
            assert False, "error dtype"
        batch_data = InferBatch.init_batch(batch_id, reqs, dtype, torch.cuda.current_device(), self.model.req_manager, self.model.vocab_size)
        self.cache[batch_id] = batch_data

        # 将更新后的状态返回给调用方用于router中请求的状态
        ans = {}
        for req_id in batch_data.request_ids:
            req_obj : InferReq  = requests_mapping[req_id]
            ans[req_id] = (req_obj.req_status, req_obj.cur_kv_len)
        # end_time  = time.perf_counter()
        # self.all_add_batch_time += end_time - start_time
        # print(f"add batch: {self.pp_rank} spend time :{ end_time - start_time}. spend all time: {self.all_add_batch_time} ")
        return pickle.dumps(ans)
    
    @calculate_time(show=False, min_cost_ms=300)
    def exposed_prefill_batch(self, batch_id):
        return self.forward(batch_id, is_prefill=True)

    @calculate_time(show=True, min_cost_ms=200)
    def exposed_decode_batch(self, batch_id):
        # start_time = time.perf_counter()
        # print(f"batch_id type : {type(batch_id)}, req_ids type: {type(req_ids)}")
        # print(f"req_ids: {req_ids}, next_token_ids: {next_token_ids}")
        if self.is_splitfuse_mode:
                # end_time = time.perf_counter()
                # self.add_token_all_times += end_time - start_time
                # print(f"splitfuse add token: {self.pp_rank} spend time :{ end_time - start_time}. spend all time: {self.add_token_all_times} ")
            return self.splitfuse_forward(batch_id)
        else:
            return self.forward(batch_id, is_prefill=False)

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        # start_time = time.perf_counter()
        # if self.world_size != 1:
        batch_id, req_id_list, finished_req_id_list = obtain(batch_id), json.loads(req_id_list), json.loads(finished_req_id_list)
        # print("filter old size:", len(batch.reqs), "new size:", len(req_id_list))
        batch = self.cache.pop(batch_id)
        filter_batch = batch.filter(req_id_list, finished_req_id_list)
        del batch
        self.cache[batch_id] = filter_batch
        # end_time  = time.perf_counter()
        # self.all_filter_batch_time += end_time - start_time
        # print(f"filter batch : {self.pp_rank} spend time :{ end_time - start_time}. spend all time: {self.all_filter_batch_time} ")
        return

    def exposed_pause_reqs(self, batch_id, req_list):
        # start_time = time.perf_counter()
        if self.world_size != 1:
            batch_id, req_list = obtain(batch_id), obtain(req_list)
        batch1 = self.cache.pop(batch_id)
        batch2 = batch1.pause_reqs(req_list)
        self.cache[batch_id] = batch2
        del batch1
        # end_time  = time.perf_counter()
        # self.all_pause_batch_time += end_time - start_time
        # print(f"pause batch : {self.pp_rank} spend time :{ end_time - start_time}. spend all time: {self.all_pause_batch_time} ")
        return

    # @calculate_time(show=True, min_cost_ms=0.1)
    def exposed_merge_batch(self, batch_id1, batch_id2):
        # start_time = time.perf_counter()
        batch1 = self.cache.pop(batch_id1)
        batch2 = self.cache.pop(batch_id2)
        m_batch = InferBatch.merge(batch1, batch2)
        del batch1
        del batch2
        self.cache[batch_id1] = m_batch
        # end_time  = time.perf_counter()
        # self.all_merge_batch_time += end_time - start_time
        # print(f"merge  batch : {self.pp_rank} spend time :{ end_time - start_time}. spend all time: {self.all_merge_batch_time} ")
        return

    # @calculate_time(show=True, min_cost_ms=10)
    def exposed_remove_batch(self, batch_id):
        # start_time = time.perf_counter()
        batch = self.cache.pop(batch_id)
        batch.free_self()
        del batch
        # end_time  = time.perf_counter()
        # self.all_remove_batch_time += end_time - start_time
        # print(f"remove batch : {self.pp_rank} spend time :{ end_time - start_time}. spend all time: {self.all_remove_batch_time} ")
        # torch.cuda.empty_cache()
        return
    
    # 更新前面所有组的统计信息
    def exposed_add_tokens(self, req_ids, next_token_ids):
        # print("exposed_add_tokens")
        # print(f"req_ids: {req_ids}, next_token_ids: {next_token_ids}")
        # print(f"pp_rank: {self.pp_rank}, add_tokens")
        # print(f"req_ids: {len(req_ids)}, next_token_ids: {len(next_token_ids)}")
        decode_reqs, prefill_reqs = [], []
        temp_next_token_ids = []
        for idx, request_id in enumerate(req_ids):
            if request_id in requests_mapping:
                req : InferReq = requests_mapping[request_id]
                if req.cur_kv_len == len(req.input_token_ids) - 1:
                    decode_reqs.append(req)
                elif req.cur_kv_len < len(req.input_token_ids) - 1:
                    prefill_reqs.append(req)
                temp_next_token_ids.append(next_token_ids[idx])
        next_token_ids = temp_next_token_ids
        decode_req_num = len(decode_reqs)
        all_reqs = decode_reqs
        all_reqs.extend(prefill_reqs)
        index = 0
        for req_obj, next_token_id in zip(all_reqs, next_token_ids):
            if index < decode_req_num:
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
            else:
                old_input_token_size = len(req_obj.input_token_ids)
                split_len = min(old_input_token_size - req_obj.cur_kv_len, self.splitfuse_block_size)
                if req_obj.cur_kv_len + split_len == old_input_token_size:
                    # 有输出
                    req_obj.cur_kv_len = old_input_token_size
                    req_obj.input_token_ids.append(next_token_id)
                elif req_obj.cur_kv_len + split_len < old_input_token_size:
                    # 没输出
                    req_obj.cur_kv_len = req_obj.cur_kv_len + split_len
                else:
                    assert False, "error state"
            index += 1
        return

    
    # @calculate_time(show=True, min_cost_ms=150)
    def forward(self, batch_id, is_prefill):
        # special code for return all prompt_logprobs
        if self.return_all_prompt_logprobs and is_prefill:
            return self._prefill_to_return_all_prompt_logprobs(batch_id)
        
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        if is_prefill:
            kwargs, run_reqs, not_run_reqs = prepare_prefill_inputs(batch, self.is_multimodal)
        else:
            kwargs, run_reqs, not_run_reqs = prepare_decode_inputs(batch)
        
        if len(run_reqs) >= 1:
            logits = self.model.forward(**kwargs)
            next_token_ids, next_token_probs = sample(logits, run_reqs)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()

            for req_obj, next_token_id, next_token_logprob in zip(run_reqs, next_token_ids, next_token_logprobs):
                # prefill and decode is same
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                metadata = {
                    'id': int(next_token_id),
                    'logprob': float(next_token_logprob),
                }
                output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, int(next_token_id), metadata) # 状态， cur_kv_len, token_id, metadata

        for req_obj in not_run_reqs:
            output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, None, None) # 状态， cur_kv_len, token_id, metadata

        self.cache[batch.batch_id] = batch
        return output_dict
    
    @torch.no_grad()
    def _prefill_to_return_all_prompt_logprobs(self, batch_id):
        output_dict = {}
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, run_reqs, not_run_reqs = prepare_prefill_inputs(batch)
        
        if len(run_reqs) >= 1:
            prompt_all_logits = self.model.forward(**kwargs)
            input_ids = kwargs["input_ids"]
            b_start_loc = kwargs["b_start_loc"]
            b_seq_len = kwargs["b_seq_len"]            
            last_index = torch.cumsum(b_seq_len, dim=0, dtype=torch.long) - 1
            logits = prompt_all_logits[last_index, :]

            next_token_ids, next_token_probs = sample(logits, run_reqs)
            next_token_ids = next_token_ids.detach().cpu().numpy()
            next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
            
            b_start_loc = b_start_loc.cpu().numpy()
            b_seq_len = b_seq_len.cpu().numpy()
            for req_obj, next_token_id, next_token_logprob, start_loc, seq_len in zip(run_reqs, next_token_ids, next_token_logprobs, b_start_loc, b_seq_len):
                # prefill and decode is same
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                metadata = {
                    'id': int(next_token_id),
                    'logprob': float(next_token_logprob),
                }

                cur_ids: torch.Tensor = input_ids[start_loc : start_loc + seq_len]
                cur_logits = prompt_all_logits[start_loc : start_loc + seq_len]
                cur_logprobs = torch.log_softmax(cur_logits, dim=-1, dtype=torch.float)[0:-1, :]
                cur_logprobs = torch.gather(cur_logprobs, dim=1, index=cur_ids[1:].view(-1, 1)).detach().cpu().numpy()

                cur_ids = cur_ids.cpu().numpy()
                all_prompts = []
                for index in range(len(cur_ids) - 1):
                    tmp_dict = {
                        int(cur_ids[index + 1]) : float(cur_logprobs[index, 0])
                    }
                    all_prompts.append([int(cur_ids[index]), tmp_dict])

                metadata["prompt_logprobs"] = all_prompts
                metadata["prompt_token_ids"] = [int(e) for e in cur_ids]
                output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, int(next_token_id), metadata) # 状态， cur_kv_len, token_id, metadata

        for req_obj in not_run_reqs:
            output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, None, None) # 状态， cur_kv_len, token_id, metadata

        self.cache[batch.batch_id] = batch
        return output_dict

    # @calculate_time(show=True, min_cost_ms=200)
    def splitfuse_forward(self, batch_id):
        # start_time = time.perf_counter()
        output_dict = {}
        # print(f"pp_rank: {self.pp_rank}, batch_id:{batch_id}")
        batch: InferBatch = self.cache.pop(batch_id)
        kwargs, decode_reqs, prefill_reqs = splitfuse_prepare_decode_inputs(batch, self.splitfuse_block_size)
        decode_req_num = len(decode_reqs)
        all_reqs = decode_reqs
        all_reqs.extend(prefill_reqs)
        logits = self.model.splitfuse_forward(**kwargs)
        # print(f"pp_rank: {self.pp_rank}, batch_id:{batch_id} done")
        if self.pp_size != 1 and self.pp_rank == 0:
            # print(f"rank: {self.pp_rank} insert batchId {batch.batch_id}")
            self.cache[batch.batch_id] = batch
            # end_time = time.perf_counter()
            # print(f"splitfuse forward rank: {self.pp_rank} spend time :{ end_time - start_time}, batch_size: {len(batch.request_ids)}")
            return output_dict
        # after_process_start_time = time.perf_counter()
        # sample_start_time = time.perf_counter()
        if self.pp_size != 1 and self.pp_rank != self.pp_size - 1:
            next_token_ids, next_token_probs = torch.full((len(all_reqs),), 13), torch.full((len(all_reqs),), 0.1516)
        else: 
            next_token_ids, next_token_probs = sample(logits, all_reqs)
        # sample_end_time = time.perf_counter()
            
        next_token_ids = next_token_ids.detach().cpu().numpy()
        next_token_logprobs = torch.log(next_token_probs).detach().cpu().numpy()
        
        index = 0
        for req_obj, next_token_id, next_token_logprob in zip(all_reqs, next_token_ids, next_token_logprobs):
            if index < decode_req_num:
                req_obj.cur_kv_len = len(req_obj.input_token_ids)
                req_obj.input_token_ids.append(next_token_id)
                req_obj.out_token_id_count[next_token_id] += 1
                metadata = {
                    'id': int(next_token_id),
                    'logprob': float(next_token_logprob),
                }
                output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, int(next_token_id), metadata) # 状态， cur_kv_len, token_id, metadata
            else:
                old_input_token_size = len(req_obj.input_token_ids)
                split_len = min(old_input_token_size - req_obj.cur_kv_len, self.splitfuse_block_size)
                if req_obj.cur_kv_len + split_len == old_input_token_size:
                    # 有输出
                    req_obj.cur_kv_len = old_input_token_size
                    req_obj.input_token_ids.append(next_token_id)
                    req_obj.out_token_id_count[next_token_id] += 1
                    metadata = {
                        'id': int(next_token_id),
                        'logprob': float(next_token_logprob),
                    }
                    output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, int(next_token_id), metadata)
                elif req_obj.cur_kv_len + split_len < old_input_token_size:
                    # 没输出
                    req_obj.cur_kv_len = req_obj.cur_kv_len + split_len
                    output_dict[req_obj.r_id] = (req_obj.req_status, req_obj.cur_kv_len, None, None)
                else:
                    assert False, "error state"
            index += 1    
        self.cache[batch.batch_id] = batch
        # after_process_end_time = time.perf_counter()
        # pickle_start_time = time.perf_counter()
        if self.pp_size != 1:
            output_dict = pickle.dumps(output_dict)
        pickle_end_time = time.perf_counter()
        # end_time = time.perf_counter()
        # self.all_after_process_time += after_process_end_time - after_process_start_time
        # self.all_sample_time += sample_end_time - sample_start_time
        # print(f"splitfuse forward rank: {self.pp_rank} insert batchId {batch.batch_id}")
        # print(f"splitfuse forward rank: {self.pp_rank} spend time :{ end_time - start_time}, pickle spend time: {pickle_end_time - pickle_start_time}, after process time: {after_process_end_time - after_process_start_time},  batch_size: {len(batch.request_ids)}")
        # print(f"splitfuse forward rank after process time: {self.all_after_process_time}")
        # print(f"splitfuse forward rank sample time: {self.all_sample_time}")
        return output_dict



class ModelRpcClient:
    def __init__(self, model_rpc, world_size, rpc_server_process=None):
        self.model: ModelRpcServer = model_rpc
        self.world_size = world_size
        self.rpc_server_process = rpc_server_process
        self.use_rpc = self.world_size != 1
        if self.use_rpc:
            def async_wrap(f):
                f = rpyc.async_(f)
                async def _func(*args, **kwargs):
                    ans = f(*args, **kwargs)
                    await asyncio.to_thread(ans.wait)
                    # raise if exception
                    return ans.value
                return _func
            self._init_model = async_wrap(self.model.init_model)
            self._add_batch = async_wrap(self.model.add_batch)
            self._prefill_batch = async_wrap(self.model.prefill_batch)
            self._decode_batch = async_wrap(self.model.decode_batch)
            self._pause_reqs = async_wrap(self.model.pause_reqs)
            self._filter_batch = async_wrap(self.model.filter_batch)
            self._merge_batch = async_wrap(self.model.merge_batch)
            self._remove_batch = async_wrap(self.model.remove_batch)
            self._add_tokens = async_wrap(self.model.add_tokens)
        else:
            self._init_model = self.model.exposed_init_model
            self._add_batch = self.model.exposed_add_batch
            self._prefill_batch = self.model.exposed_prefill_batch
            self._decode_batch = self.model.exposed_decode_batch
            self._pause_reqs = self.model.exposed_pause_reqs
            self._filter_batch = self.model.exposed_filter_batch
            self._merge_batch = self.model.exposed_merge_batch
            self._remove_batch = self.model.exposed_remove_batch
            self._add_tokens = self.model.exposed_add_tokens
        self.decode_req_que = None
        self.decode_resp_que = None
        self.init_resp_que = None
        # self.obtain_all_time = 0
        # self.decode_all_time = 0
        return

    async def init_model(self, kvargs):
        ans : rpyc.AsyncResult = self._init_model(kvargs)
        if self.use_rpc:
            await ans
            return
        else:
            return

    async def init_batch(self, batch_id, reqs):
        ans = self._add_batch(batch_id, reqs, "fp16")
        if self.use_rpc:
            return pickle.loads(await ans)
        else:
            return ans
        
    async def pp_init_batch(self, batch, reqs, tp_rank):
        # print(f"pp_init_batch")
        await self.decode_req_que.put(("init_batch", batch, reqs, tp_rank))

    async def prefill_batch(self, batch_id):
        ans = self._prefill_batch(batch_id)
        if self.use_rpc:
            return await ans
        else:
            return ans

    async def pp_decode_batch(self, batch_id, flag = False):
        # print(f"add_req_que: {rank_id}")
        await self.decode_req_que.put(("decode_batch", batch_id, flag))
    
    async def decode_batch(self, batch_id):
        ans = self._decode_batch(batch_id)
        if self.use_rpc:
            return await ans
        else:
            return ans
    def init_que(self):
        self.decode_req_que = asyncio.Queue()
        self.decode_resp_que = asyncio.Queue()
        self.init_resp_que = asyncio.Queue()

    async def decode_batch_loop(self):
        if self.decode_req_que is None or self.decode_resp_que is None:
            self.init_que()
        while True:
            # print(f"start decode_batch_loop, que size: {self.decode_req_que.qsize()}")
            data = await self.decode_req_que.get()
            if data[0] == "decode_batch":
                batch_id, flag = data[1:]
                # start_time = time.perf_counter()
                # print(f"start decode batch, rank _id : {rank_id}")
                ans = self._decode_batch(batch_id)
                true_ans = await ans
                # end_time = time.perf_counter()
                # self.decode_all_time += end_time - start_time
                # print(f"decode batch: {rank_id}, spend time: {end_time - start_time}")
                if flag:
                    # start_time = time.perf_counter()
                    # print(type(true_ans))
                    # if is_ref(true_ans):
                    #     print("is netref")
                    dd = pickle.loads(true_ans)
                    #print(dd)
                    # end_time = time.perf_counter()
                    # self.obtain_all_time += end_time - start_time
                    # self.decode_all_time += end_time - start_time
                    # print(f"decode batch: {rank_id}, obtain spend time: {end_time - start_time}, flag: {flag}, obtain_all_time: {self.obtain_all_time}, decode_all_time: {self.decode_all_time}")
                    await self.decode_resp_que.put(dd)
                else:
                    pass
                    # print(f"decode_all_time: {self.decode_all_time}")
            elif data[0] == "init_batch":
                # print("init_batch")
                batch, reqs, tp_rank = data[1:]
                ans = await self._add_batch(batch.batch_id, reqs, "fp16")
                # print("init_batch done")
                dd = pickle.loads(ans)
                if tp_rank == 0:   
                    await self.init_resp_que.put(dd)
            elif data[0] == "filter_batch":
                batch_id, req_id_list, finished_req_id_list = data[1:]
                await self._filter_batch(batch_id, json.dumps(req_id_list), json.dumps(finished_req_id_list))
            elif data[0] == "pause_reqs":
                batch_id, reqs_list = data[1:]
                await self._pause_reqs(batch_id, reqs_list)
            elif data[0] == "merge_batch":
                batch_id1, batch_id2 = data[1:]
                await self._merge_batch(batch_id1, batch_id2)
            elif data[0] == "remove_batch":
                batch_id = data[-1]
                await self._remove_batch(str(batch_id))
            elif data[0] == "add_tokens":
                req_ids, next_token_ids = data[1:]
                await self._add_tokens(req_ids, next_token_ids)


    async def filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        ans = self._filter_batch(batch_id, json.dumps(req_id_list), json.dumps(finished_req_id_list))
        if self.use_rpc:
            await ans
            return
        else:
            return 

    async def pp_filter_batch(self, batch_id, req_id_list, finished_req_id_list):
        # print(f"pp_filter_batch")
        await self.decode_req_que.put(("filter_batch", batch_id, req_id_list, finished_req_id_list))

    async def pause_reqs(self, batch_id, reqs_list):
        ans = self._pause_reqs(batch_id, reqs_list)
        if self.use_rpc:
            await ans
            return
        else:
            return
        
    async def pp_pause_reqs(self, batch_id, reqs_list):
        print(f"pause_reqs")
        await self.decode_req_que.put(("pause_reqs", batch_id, reqs_list))

    async def merge_batch(self, batch_id1, batch_id2):
        ans = self._merge_batch(batch_id1, batch_id2)
        if self.use_rpc:
            await ans
            return
        else:
            return
    
    async def pp_merge_batch(self, batch_id1, batch_id2):
        # print(f"pp_merge_batch")
        await self.decode_req_que.put(("merge_batch", batch_id1, batch_id2))

    async def remove_batch(self, batch_id):
        ans = self._remove_batch(batch_id)
        if self.use_rpc:
            await ans
            return
        else:
            return
        
    async def pp_remove_batch(self, batch_id):
        # print(f"pp_remove_batch")
        await self.decode_req_que.put(("remove_batch", batch_id))
        
    async def add_tokens(self, req_ids, next_token_ids):
        ans = self._add_tokens(req_ids, next_token_ids)
        if self.use_rpc:
            await ans
            return
        else:
            return
    
    async def pp_add_tokens(self, req_ids, next_token_ids):
        await self.decode_req_que.put(("add_tokens", req_ids, next_token_ids))

def _init_env(port):
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(ModelRpcServer(), port=port, protocol_config={"allow_pickle": True})
    t.start()
    return


async def start_model_process(port, world_size):
    # 单卡时不使用 rpc
    if world_size == 1:
        return ModelRpcClient(ModelRpcServer(), world_size)
    
    import multiprocessing
    proc = multiprocessing.Process(target=_init_env, args=(port,))
    proc.start()
    await asyncio.sleep(2)
    repeat_count = 0
    while repeat_count < 20:
        try:
            con = rpyc.connect("localhost", port, config={"allow_pickle": True})
            break
        except BaseException:
            await asyncio.sleep(1)
        repeat_count += 1
    if repeat_count == 20:
        raise Exception("init rpc env error!")

    assert proc.is_alive()
    return ModelRpcClient(con.root, world_size, rpc_server_process=proc)
