import time
import uuid
import uvloop
import asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
import zmq
import zmq.asyncio
from typing import Dict, List, Optional
from ..sampling_params import SamplingParams
from ..io_struct import Req, NormalReq, SplitFuseReq, Batch
from ..multimodal_params import MultimodalParams
from .model_infer.model_rpc import start_model_process, ModelRpcClient
from .req_queue import ReqQueue
from rpyc.utils.classic import obtain
from lightllm.utils.infer_utils import calculate_time
from ..io_struct import BatchTokenIdOut, AbortReq, ReqRunStatus, FinishStatus
from .stats import Stats
from .pause_strategy import Fcfs, select_paused_reqs
from ..tokenizer import get_tokenizer
from lightllm.utils.log_utils import init_logger
from collections import deque
logger = init_logger(__name__)


class RouterManager:

    def __init__(self, args, router_port, detokenization_port, model_rpc_ports):
        self.args = args
        self.model_weightdir = args.model_dir
        self.world_size = args.tp * args.pp
        self.tp_size = args.tp
        self.pp_size = args.pp
        self.load_way = args.load_way
        self.mode = args.mode
        self.max_total_token_num = args.max_total_token_num
        
        self.pause_strategy = Fcfs()
        self.running_batch_list: Batch = [None] * self.pp_size
        self.wait_to_return = [None] * self.pp_size
        self.decode_carry_message = [(None, None)] * self.pp_size
        self.running_batch: Batch = None
        self.eos_id = args.eos_id
        self.has_wait_tokens = 0
        self.has_wait_tokens_list = [0] * self.pp_size
        self.max_wait_tokens = 10
        self.counter_count = 0
        self.all_times_add_tokens = 0
        self.all_times_handle_finish = 0
        
        self.pp_deque_list = []
        for idx in range(args.pp + 1):
            self.pp_deque_list.append(deque())
        
        context = zmq.asyncio.Context(2)
        self.recv_from_httpserver = context.socket(zmq.PULL)
        self.recv_from_httpserver.bind(f"tcp://127.0.0.1:{router_port}")
        
        self.send_to_detokenization = context.socket(zmq.PUSH)
        self.send_to_detokenization.connect(f"tcp://127.0.0.1:{detokenization_port}")
        self.model_rpc_ports = model_rpc_ports

        self.is_splitfuse_mode = args.splitfuse_mode
        self.splitfuse_block_size = args.splitfuse_block_size
        self.times = 0

        if self.is_splitfuse_mode and len(args.prompt_cache_strs) != 0:
            self.tokenizer = get_tokenizer(self.model_weightdir, args.tokenizer_mode, args.trust_remote_code)

        self.stats_tool = Stats(not args.disable_log_stats, args.log_stats_interval)
        self.stats_tool_list = [Stats(not args.disable_log_stats, args.log_stats_interval) for _ in range(self.pp_size)]
        return

    def compute_pre_and_post_rank(self, tp_rank, pp_rank):
        ans_left, ans_right = -1, -1
        if pp_rank:
            ans_left = tp_rank * self.pp_size + pp_rank - 1
        if pp_rank != self.pp_size - 1:
            ans_right = tp_rank * self.pp_size + pp_rank + 1
        return [ans_left, ans_right]
        
    def compute_all_reduce_rank(self, tp_rank, pp_rank):
        ans_list = []
        for i in range(self.tp_size):
            ans_list.append(i * self.pp_size + pp_rank)
        return ans_list

    async def wait_to_model_ready(self):
        # 初始化模型
        self.model_rpcs: List[ModelRpcClient] = []
        for rank_id in range(self.world_size):
            rpc_model = await start_model_process(port=self.model_rpc_ports[rank_id], world_size=self.world_size)
            self.model_rpcs.append(rpc_model)

        init_model_ret = []
        if self.pp_size == 1:
            for rank_id in range(self.world_size):  # async init model process
                kvargs = {
                    "rank_id" : rank_id,
                    "world_size" : self.world_size,
                    "weight_dir" : self.model_weightdir,
                    "load_way" : self.load_way,
                    "max_total_token_num" : self.max_total_token_num,
                    "mode" : self.mode,
                    "max_req_num" : self.args.running_max_req_size + 8,
                    "max_seq_length" : self.args.max_req_total_len + 8, # 留一点余量
                    "nccl_port" : self.args.nccl_port,
                    "is_splitfuse_mode" : self.is_splitfuse_mode,
                    "splitfuse_block_size" : self.splitfuse_block_size,
                    "return_all_prompt_logprobs" : self.args.return_all_prompt_logprobs
                }
                init_model_ret.append(self.model_rpcs[rank_id].init_model(kvargs))
        else:
            tp_size = self.world_size // self.pp_size
            for pp_rank in range(self.pp_size):
                for tp_rank in range(tp_size):  # async init model process
                    new_rank = tp_rank * self.pp_size + pp_rank
                    kvargs = {
                        "rank_id" : tp_rank,
                        "tp_size" : self.tp_size,
                        "world_size" : self.world_size,
                        "weight_dir" : self.model_weightdir,
                        "load_way" : self.load_way,
                        "max_total_token_num" : self.max_total_token_num,
                        "mode" : self.mode,
                        "max_req_num" : self.args.running_max_req_size + 8,
                        "max_seq_length" : self.args.max_req_total_len + 8, # 留一点余量
                        "nccl_port" : self.args.nccl_port,
                        "is_splitfuse_mode" : self.is_splitfuse_mode,
                        "splitfuse_block_size" : self.splitfuse_block_size,
                        "return_all_prompt_logprobs" : self.args.return_all_prompt_logprobs,
                        "pp_rank": pp_rank,
                        "pp_size": self.pp_size,
                        "pre_and_post_rank": self.compute_pre_and_post_rank(tp_rank, pp_rank),
                        "all_reduce_rank": self.compute_all_reduce_rank(tp_rank, pp_rank)
                    }
                    init_model_ret.append(self.model_rpcs[new_rank].init_model(kvargs))

        await asyncio.gather(*init_model_ret)

        await self._init_prompt_cache()
        
        self.req_queue = ReqQueue(self.args, 
                                  self.prompt_cache_used_tokens, 
                                  self.prompt_cache_req_num)   
        return
    
    async def _init_prompt_cache(self):
        """
        初始化 prompt cache 特性, 这个地方的id 分配要于 httpserver 中的id 分配对齐
        """
        # 初始化 prompt cahce， 然后初始化请求队列
        self.prompt_cache_used_tokens = 0
        self.prompt_cache_req_num = len(self.args.prompt_cache_strs)
        if self.is_splitfuse_mode:
            reqs = []
            id = -1 # id 从 -1， -2， .... 避免和正常的 id 占用
            for prompt_cache_str in self.args.prompt_cache_strs:
                prompt_ids = self.tokenizer.encode(prompt_cache_str)
                req = NormalReq(id, prompt_ids, SamplingParams(stop_sequences=[]))
                self.prompt_cache_used_tokens += len(prompt_ids)
                reqs.append(req)
                id -= 1
            if len(reqs) != 0:
                self.prompt_cache_batch = Batch(uuid.uuid4().hex, reqs)
                await self._prefill_to_init_prompt_cache(self.prompt_cache_batch)
        return

    def add_req(
        self,
        prompt_ids: List[int],
        sampling_params: SamplingParams,
        multimodal_params: MultimodalParams,
        request_id: str,
        prompt_cache_len, 
        prompt_cache_req_id
    ):  
        if self.is_splitfuse_mode:
            req = SplitFuseReq(request_id, prompt_ids, sampling_params, multimodal_params, 
                               prompt_cache_len, prompt_cache_req_id, self.splitfuse_block_size)
        else:
            req = NormalReq(request_id, prompt_ids, sampling_params, multimodal_params, 
                            prompt_cache_len, prompt_cache_req_id)
        self.req_queue.append(req)
        self.send_to_detokenization.send_pyobj(req.to_req_detokenization_state())
        return

    async def abort(self, request_id):
        if self.running_batch is not None:
            for req in self.running_batch.reqs:
                if req.request_id == request_id:
                    req.finish_status = FinishStatus.FINISHED_ABORT
        for req in self.req_queue.waiting_req_list:
            if req.request_id == request_id:
                req.finish_status = FinishStatus.FINISHED_ABORT
        return

    async def loop_for_fwd(self):
        counter_count = 0
        while True:
            await self._step()
            counter_count += 1
            if self.running_batch is not None:
                if counter_count % 50 == 0:
                    total_used_tokens = self.prompt_cache_used_tokens + self.running_batch.batch_used_tokens + self.req_queue.pause_req_used_tokens
                    token_ratio = total_used_tokens / self.max_total_token_num
                    logger.debug(
                        f"current batch size: {len(self.running_batch.reqs)} " 
                        f"paused req num: {len(self.req_queue.pause_req_dict)} "
                        f"token used ratio: {token_ratio} "
                    )
                    pass
                self.stats_tool.print_stats()
                
            if self.running_batch is None:
                await asyncio.sleep(0.01)  # 10ms
                
    async def loop_for_fwd_add_pp(self):
        
        while True:
            for rank_id in range(self.pp_size):
                ans = True
                while ans:
                    ans = await self._pp_step(rank_id)
                    self.counter_count += 1 
                    running_batch = self.running_batch_list[rank_id]
                    if running_batch is not None:
                        if self.counter_count % 50 == 0:
                            total_used_tokens = self.prompt_cache_used_tokens + running_batch.batch_used_tokens + self.req_queue.pause_req_used_tokens
                            token_ratio = total_used_tokens / self.max_total_token_num
                            logger.debug(
                                f"rank_id: {rank_id}"
                                f"current batch size: {len(running_batch.reqs)} " 
                                f"paused req num: {len(self.req_queue.pause_req_dict)} "
                                f"token used ratio: {token_ratio} "
                            )
                            pass
                        self.stats_tool_list[rank_id].print_stats(rank_id)
                        
                    if running_batch is None:
                        await asyncio.sleep(0.01)  # 10ms

    async def _pp_step(self, rank_id):
        """
        事件处理循环
        """
        if self.wait_to_return[rank_id] is not None:
            await asyncio.gather(*self.wait_to_return[rank_id])
            # print(f"rank_id: {rank_id} end")
            self.wait_to_return[rank_id] = None
            # print(f"rank_id: {rank_id} has done")
            if self.world_size != 1:
                last_rank_id = self.tp_size * self.pp_size - 1
                req_to_out_status = await self.model_rpcs[last_rank_id].decode_resp_que.get()
                self.decode_carry_message[rank_id] = (list(req_to_out_status.keys()), [value[2] for value in req_to_out_status.values()])
                # print("get result")
                
                # 使用这个结果更新下前面的结果
                # print(f"req_to_out_statue: {req_to_out_status}")
                # temp_rets = []
                # for pp_rank in range(1):
                #     for tp_rank in range(self.tp_size):
                #         new_rank = tp_rank * self.pp_size + pp_rank
                #         temp_rets.append(self.model_rpcs[new_rank].add_tokens(list(req_to_out_status.keys()), [value[2] for value in req_to_out_status.values()]))
                         
            else:
                req_to_out_status = self.model_rpcs[0].decode_resp_que.get()
            self._update_out_status_to_batch(self.running_batch_list[rank_id], req_to_out_status)
            unfinished_req_ids, finished_req_ids = self.running_batch_list[rank_id].mark_and_get_finished_req_and_preupdate_status(self.eos_id)
            self._send_to_detokenization_proc(self.running_batch_list[rank_id], req_to_out_status)
            self.running_batch_list[rank_id].filter_out_finished_req(unfinished_req_ids, finished_req_ids)
            # start_time = time.perf_counter()
            # await asyncio.gather(*temp_rets) 
            endd_time = time.perf_counter() 
            # print("add token done")
            await self._handle_finish_req(self.running_batch_list[rank_id], unfinished_req_ids, finished_req_ids)
            end_time = time.perf_counter()
            self.all_times_handle_finish += end_time - endd_time
            print(f"all_times_handle_finish:{self.all_times_handle_finish}")
            # print("handle finish req")
            self._pp_filter_runing_batch(rank_id)
            # 一些更进一步的处理
        
        # 删除所有已经 finished 的 req
        # 当前无运行请求时
        if self.running_batch_list[rank_id] is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch_list[rank_id])
            if new_batch is not None:
                self.stats_tool_list[rank_id].count_prompt_tokens(new_batch)
                self.running_batch_list[rank_id] = new_batch
                # print(f"rank_id: {rank_id}")
                await self._prefill_batch(self.running_batch_list[rank_id])
                self._pp_filter_runing_batch(rank_id)
                self.has_wait_tokens_list[rank_id] = 0
                return True
            return False # 是否还需要再调用一次

        
        # 有运行请求，但是已经到了可以调度新的请求合并推理的时机
        if self.has_wait_tokens_list[rank_id] >= self.max_wait_tokens:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch_list[rank_id])
            self.has_wait_tokens_list[rank_id] = 0
            if new_mini_batch is not None:
                self.stats_tool_list[rank_id].count_prompt_tokens(new_mini_batch)
                await self._prefill_batch(new_mini_batch)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch_list[rank_id], new_mini_batch)
                    self.running_batch_list[rank_id].merge(new_mini_batch)
                return True

        # 正常 decode 阶段， 如果可以直接decode就直接decode，否则通过暂停策略暂停一些请求
        # 释放一些管理的 token
        if self._can_decode(self.running_batch_list[rank_id]):
            self.stats_tool_list[rank_id].count_output_tokens(self.running_batch_list[rank_id])
            # print(f"rank_id: {rank_id} prepare decode batch")
            await self._decode_batch(self.running_batch_list[rank_id], rank_id)
            # print(f"rank_id: {rank_id} after decode batch")
            self.has_wait_tokens_list[rank_id] += 1
            return False
        else:
            # pause strategy
            paused_reqs = select_paused_reqs(self.running_batch_list[rank_id], self.pause_strategy, self.req_queue, self.max_total_token_num)
            await self._pause_reqs(self.running_batch_list[rank_id], paused_reqs)
            logger.debug(f"pasued req num: {len(self.req_queue.pause_req_dict)}")
            self.has_wait_tokens_list[rank_id] = 0
            return True
        return

    async def _step(self):
        """
        事件处理循环
        """
        # 删除所有已经 finished 的 req
        # 当前无运行请求时
        if self.running_batch is None:
            new_batch = self.req_queue.generate_new_batch(self.running_batch)
            if new_batch is not None:
                self.stats_tool.count_prompt_tokens(new_batch)
                self.running_batch = new_batch
                await self._prefill_batch(self.running_batch)
                self._filter_runing_batch()
                self.has_wait_tokens = 0
            return

        # 有运行请求，但是已经到了可以调度新的请求合并推理的时机
        if self.has_wait_tokens >= self.max_wait_tokens:
            new_mini_batch = self.req_queue.generate_new_batch(self.running_batch)
            self.has_wait_tokens = 0
            if new_mini_batch is not None:
                self.stats_tool.count_prompt_tokens(new_mini_batch)
                await self._prefill_batch(new_mini_batch)
                if not new_mini_batch.is_clear():
                    await self._merge_batch(self.running_batch, new_mini_batch)
                    self.running_batch.merge(new_mini_batch)
                return

        # 正常 decode 阶段， 如果可以直接decode就直接decode，否则通过暂停策略暂停一些请求
        # 释放一些管理的 token
        if self._can_decode(self.running_batch):
            self.stats_tool.count_output_tokens(self.running_batch)
            await self._decode_batch(self.running_batch)
            self._filter_runing_batch()
            self.has_wait_tokens += 1
            return
        else:
            # pause strategy
            paused_reqs = select_paused_reqs(self.running_batch, self.pause_strategy, self.req_queue, self.max_total_token_num)
            await self._pause_reqs(self.running_batch, paused_reqs)
            logger.debug(f"pasued req num: {len(self.req_queue.pause_req_dict)}")
            self.has_wait_tokens = 0
            return
        return

    async def _init_batch(self, batch: Batch):
        reqs = [r.to_rpc_obj() for r in batch.reqs]
        rets = [self.model_rpcs[tp_rank].init_batch(batch.batch_id, reqs) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_req_status = obtain(ans[0])
        else:
            req_to_req_status = ans[0]
        
        self._update_init_status_to_batch(batch, req_to_req_status)
        return

    async def _prefill_batch(self, batch:Batch):
        await self._init_batch(batch)
        if not self.is_splitfuse_mode:
            # 在 非 splitfuse 模式下，才需要真的执行 prefill 的操作。
            rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
            ans = await asyncio.gather(*rets)
            if self.world_size != 1:
                req_to_out_status = obtain(ans[0])
            else:
                req_to_out_status = ans[0]

            self._update_out_status_to_batch(batch, req_to_out_status)
            unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status(self.eos_id)
            self._send_to_detokenization_proc(batch, req_to_out_status)
            batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
            await self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids)
        return
    
    async def _prefill_to_init_prompt_cache(self, batch:Batch):
        """
        专用于初始化prompt cahce 请求的接口, 只在 splitfuse + prompt cache 模式下调用
        """
        await self._init_batch(batch)
        # 在 splitfuse 模式下，才需要真的执行 prefill 的操作。
        rets = [self.model_rpcs[tp_rank].prefill_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        ans = await asyncio.gather(*rets)
        if self.world_size != 1:
            req_to_out_status = obtain(ans[0])
        else:
            req_to_out_status = ans[0]

        self._update_out_status_to_batch(batch, req_to_out_status)
        return

    async def _decode_batch(self, batch:Batch, rank_id = -1):
        if self.pp_size == 1:
            rets = [self.model_rpcs[tp_rank].decode_batch(batch.batch_id) for tp_rank in range(self.world_size)]
            ans = await asyncio.gather(*rets)
            if self.world_size != 1:
                req_to_out_status = obtain(ans[0])
            else:
                req_to_out_status = ans[0]

            self._update_out_status_to_batch(batch, req_to_out_status)
            unfinished_req_ids, finished_req_ids = batch.mark_and_get_finished_req_and_preupdate_status(self.eos_id)
            self._send_to_detokenization_proc(batch, req_to_out_status)
            batch.filter_out_finished_req(unfinished_req_ids, finished_req_ids)
            await self._handle_finish_req(batch, unfinished_req_ids, finished_req_ids)
            return
        else:
            self.wait_to_return[rank_id] = []
            # print(f"rank_id: {rank_id} start, batch_id: {batch.batch_id}")
            for pp_rank in range(self.pp_size):
                for tp_rank in range(self.tp_size):
                    new_rank = tp_rank * self.pp_size + pp_rank
                    if tp_rank == self.tp_size - 1 and pp_rank == self.pp_size - 1:
                        self.wait_to_return[rank_id].append(asyncio.ensure_future(self.model_rpcs[new_rank].pp_decode_batch(batch.batch_id, None, None, rank_id, True)))
                    elif pp_rank == 0:
                        self.wait_to_return[rank_id].append(asyncio.ensure_future(self.model_rpcs[new_rank].pp_decode_batch(batch.batch_id, self.decode_carry_message[rank_id][0], self.decode_carry_message[rank_id][1], rank_id, False)))
                    else:
                        self.wait_to_return[rank_id].append(asyncio.ensure_future(self.model_rpcs[new_rank].pp_decode_batch(batch.batch_id, None, None, rank_id, False)))
                        
            self.decode_carry_message[rank_id] = (None, None)
            return
                    

    async def _filter_batch(self, batch: Batch, unfinished_req_ids, finished_req_ids: List):
        # print("filter_batch")
        rets = [self.model_rpcs[tp_rank].filter_batch(batch.batch_id, unfinished_req_ids, finished_req_ids) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _merge_batch(self, batch1, batch2):
        rets = [self.model_rpcs[tp_rank].merge_batch(batch1.batch_id, batch2.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _remove_batch(self, batch):
        # print("remove_batch")
        rets = [self.model_rpcs[tp_rank].remove_batch(batch.batch_id) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return
    
    async def _pause_reqs(self, batch: Batch, pasue_reqs):
        pasue_reqs_info = [(r.request_id, r.req_status) for r in pasue_reqs]
        rets = [self.model_rpcs[tp_rank].pause_reqs(batch.batch_id, pasue_reqs_info) for tp_rank in range(self.world_size)]
        await asyncio.gather(*rets)
        return

    async def _handle_finish_req(self, batch: Batch, unfinished_req_ids, finished_req_ids):
        if len(finished_req_ids) != 0:
            #print("fuck finished req")
            if batch.is_clear():
                # print(f"remove_batch")
                await self._remove_batch(batch)
            else:
                # print(f"finished_req_ids: {finished_req_ids}")
                await self._filter_batch(batch, unfinished_req_ids, finished_req_ids)
        return
    
    async def _pp_handle_finish_req(self, batch: Batch, unfinished_req_ids, finished_req_ids):
        if len(finished_req_ids) != 0:
            print("fuck finished req")
            if batch.is_clear():
                await self._remove_batch(batch)
            else:
                await self._filter_batch(batch, unfinished_req_ids, finished_req_ids)
        return

    def start_decode_loop(self, loop):
        for client in self.model_rpcs:
            loop.create_task(client.decode_batch_loop())

    def _filter_runing_batch(self):
        if self.running_batch is not None and self.running_batch.is_clear():
            self.running_batch = None
            return

    def _pp_filter_runing_batch(self, rank_id):
        if self.running_batch_list[rank_id] is not None and self.running_batch_list[rank_id].is_clear():
            self.running_batch_list[rank_id] = None
            return
    
    def _update_init_status_to_batch(self, batch: Batch, req_to_req_status):
        # 更新请求状态
        new_batch_used_tokens = 0
        new_batch_decode_need_tokens = 0 # 只有在 splitfuse 模式下有意义
        for req_id, (req_status, cur_kv_len) in req_to_req_status.items():
            r_obj = batch.id_to_reqs[req_id]
            r_obj.req_status = req_status
            r_obj.cur_kv_len = cur_kv_len
            new_batch_used_tokens += r_obj.get_used_tokens()
            new_batch_decode_need_tokens += r_obj.get_decode_need_tokens()
        
        batch.batch_used_tokens = new_batch_used_tokens
        batch.batch_decode_need_tokens = new_batch_decode_need_tokens
        return
    
    def _update_out_status_to_batch(self, batch: Batch, req_to_out_status):
        new_batch_used_tokens = 0
        new_batch_decode_need_tokens = 0 # 只有在 splitfuse 模式下有意义
        for req_id, (req_status, cur_kv_len, new_token_id, new_gen_metadata) in req_to_out_status.items():
            req : Req = batch.id_to_reqs[req_id]
            req.req_status = req_status
            req.cur_kv_len = cur_kv_len
            if new_token_id is not None:
                req.output_ids.append(new_token_id)
                req.output_metadata_list.append(new_gen_metadata)
            new_batch_used_tokens += req.get_used_tokens()
            new_batch_decode_need_tokens += req.get_decode_need_tokens()
        
        batch.batch_used_tokens = new_batch_used_tokens
        batch.batch_decode_need_tokens = new_batch_decode_need_tokens
        return
        
    def _can_decode(self, batch: Batch):
        total_used_tokens = self.prompt_cache_used_tokens + batch.batch_used_tokens + self.req_queue.pause_req_used_tokens
        remaining_tokens = self.max_total_token_num - total_used_tokens
        return batch.batch_decode_need_tokens <= remaining_tokens
        
    def _send_to_detokenization_proc(self, batch: Batch, req_ans):
        batch_out = BatchTokenIdOut()
        for req_id, (_, _, new_token_id, new_gen_metadata) in req_ans.items():
            req = batch.id_to_reqs[req_id]
            if new_token_id is not None:
                # req.finish_status 传输 value值 不传送对象，可以减少序列化对象的大小。
                batch_out.reqs_infs.append((req_id, new_token_id, new_gen_metadata, req.finish_status.value))
    
        self.send_to_detokenization.send_pyobj(batch_out)
        return

    async def loop_for_netio_req(self):
        while True:
            recv_req = await self.recv_from_httpserver.recv_pyobj()
            if isinstance(recv_req, tuple) and len(recv_req) == 6:
                prompt_ids, sampling_params, multimodal_params, request_id, prompt_cache_len, prompt_cache_req_id = recv_req
                self.add_req(prompt_ids, sampling_params, multimodal_params, request_id, prompt_cache_len, prompt_cache_req_id)
            elif isinstance(recv_req, AbortReq):
                abort_req = recv_req
                request_id = abort_req.req_id
                await self.abort(request_id)
                self.send_to_detokenization.send_pyobj(abort_req)
            else:
                assert False, f"Error Req Inf {recv_req}"

    def clean_up(self):
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.kill()
        for model_rpc in self.model_rpcs:
            model_rpc.rpc_server_process.join()
        return

def start_router_process(args, router_port, detokenization_port, model_rpc_ports, pipe_writer):
    try:
        router = RouterManager(
            args,
            router_port=router_port,
            detokenization_port=detokenization_port,
            model_rpc_ports=model_rpc_ports)
    
        asyncio.run(router.wait_to_model_ready())
    except Exception as e:
        import traceback
        import sys
        etype, evalue, tb = sys.exc_info()
        err_str = '\n'.join(traceback.format_exception(etype, evalue, tb))
        pipe_writer.send(err_str)
        router.clean_up()
        raise

    pipe_writer.send('init ok')
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    if (args.pp == 1):
        loop.create_task(router.loop_for_fwd())
    else:
        loop.create_task(router.loop_for_fwd_add_pp())
        router.start_decode_loop(loop)
    # loop.create_task(router.loop_for_fwd())
    loop.run_until_complete(router.loop_for_netio_req())
    return
