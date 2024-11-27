# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com), TXacs (txacs1993@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Parts of the code here are adapted from https://github.com/microsoft/
# DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/
# pipe/engine.py

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.dataloader import (
    _MultiProcessingDataLoaderIter,
    _SingleProcessDataLoaderIter
)
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from typing import List, Tuple, Union, Optional, Callable

try:
    import apex
    from apex import amp
except ImportError:
    pass
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler

from . import mpu
from .amp import MixedPrecisionConfig
from .printer import (
    SynchronizedWallClockTimer,
    logger,
    set_timer_log_rank,
    see_memory_usage
)

from .pipeline import exec_schedule, PipelineModule
from .mpu import p2p_communication as p2p
from .pipeline import InferenceSchedule, schedule
from .pipeline.exec_schedule import PipeSchedule
from .pipeline.utils import (
    custom_backward,
    deallocate_output_tensor,
    clip_grad_norm_
)
from .recompute import RNGManager
from .recompute import pre_checkpoint as pre_checkpoint_func
from .zero import configure_zero_optimizer
from .engine_utils import (
    mem_status, _zero_grads,
    _recv_tensor_meta, _send_tensor_meta,
    _scale_loss_by_gas, _create_buffer,
    split_half_float_double
)

from ..merak_args import MerakArguments
from ..utils import RepeatingLoader

class CommunicationEngine:

    def __init__(
            self,
            pipeline,
        ):

        self.args = pipeline.args
        self.device = pipeline.device
        self.timers = pipeline.timers
        self.pipeline = pipeline

        self.pipe_recv_buf = None
        self.first_output_send = True
        self.first_gradient_send = True
        # used to disable the pipeline all-reduce when used with 1-bit
        # Adam/1-bit LAMB
        self.pipeline_enable_backward_allreduce = True

        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

        # Set Grid and Communication Groups
        self.grid = self.pipeline.module._grid
        if self.grid.get_global_rank() == 0:
            logger.info(f'CONFIG: micro_batches={self.pipeline.micro_batches} '
                        f'micro_batch_size={self.pipeline.micro_batch_size}')

        self.global_rank = self.grid.get_global_rank()

        assert mpu.get_data_parallel_world_size() == \
            self.grid.data_parallel_size
        assert self.pipeline.tuning_params.total_batch_size == \
                                        self.pipeline.micro_batch_size * \
                                        self.pipeline.micro_batches * \
                                        self.grid.data_parallel_size

        #  Set Stage Inf
        self.stage_id = self.grid.get_stage_id()
        self.prev_stage = self.stage_id - 1
        self.next_stage = self.stage_id + 1
        self.num_stages = self.grid.pipe_parallel_size

        self.is_pipe_parallel = self.grid.pipe_parallel_size > 1
        self.is_data_parallel = self.grid.data_parallel_size > 1
        self.is_model_parallel = self.grid.model_parallel_size > 1

        #intialize peer-2-peer communication and allreduce groups
        if self.is_pipe_parallel:
            next_rank = None
            prev_rank = None
            if 0 <= self.next_stage < self.num_stages:
                next_rank = self.grid.stage_to_global(stage_id=self.next_stage)
            if 0 <= self.prev_stage < self.num_stages:
                prev_rank = self.grid.stage_to_global(stage_id=self.prev_stage)
            mpu.initialize.set_pipeline_model_parallel_next_rank(next_rank)
            mpu.initialize.set_pipeline_model_parallel_prev_rank(prev_rank)


    def _exec_reduce_tied_grads(self):
        # We need to run this first to write to self.averaged_gradients;
        # since this class turns `enable_backward_allreduce` off,
        # `self.overlapping_partition_gradients_reduce_epilogue()` defined in
        # the DeepSpeedEngine never actually runs. I suspect this is because of
        # efficiency problems; get_flat_partition in stage2.py might do
        # something expensive; someone will have to look into that later. But
        # in the meantime, this fixes ZeRO2 + Pipelining enough to run a demo.
        # Further profiling needed to decide if it actually breaks everything.
        # (see https://github.com/EleutherAI/gpt-neox/issues/
        # 62#issuecomment-761471944)
        self.timers('backward_tied_allreduce').start()
        if hasattr(self.pipeline.module, 'tied_comms'):
            self.pipeline.module.allreduce_tied_weight_gradients()
        self.timers('backward_tied_allreduce').stop()

    def _exec_reduce_grads(self):
        # self._force_grad_boundary = self.is_gradient_accumulation_boundary()
        if self.args.sequence_parallel:
            self.allreduce_sequence_parallel_gradients()
        if self.pipeline_enable_backward_allreduce:
            self.timers('backward_allreduce').start()
            self.allreduce_gradients()
            self.timers('backward_allreduce').stop()
        # self._force_grad_boundary = False

    def _exec_send_activations(self, buffer_id: int):
        if self.args.wall_clock_breakdown:
            self.timers('pipe_send_output').start()

        outputs = self.pipeline.pipe_buffers['outputs'][buffer_id]
        self.grad_layer = self.pipeline.grad_layer

        if self.first_output_send:
            self.first_output_send = False
            _send_tensor_meta(outputs, self.device)

        if self.grad_layer is None:
            self.grad_layer = _create_buffer(outputs, self.device)

        # print(outputs.dtype, "==================")
        assert isinstance(outputs, tuple)
        for idx, buffer in enumerate(outputs):
            if not buffer.is_contiguous():
                buffer = buffer.contiguous()
            p2p.send_forward(buffer)
            if self.args.dealloc_pipeoutput:
                deallocate_output_tensor(buffer)

        if self.args.wall_clock_breakdown:
            self.timers('pipe_send_output').stop()

    def _exec_recv_activations(self, buffer_id):

        if self.args.wall_clock_breakdown:
            self.timers('batch_input').start()

        input_stage_list = list(self.pipeline.input_to_stage_dic.keys())
        if (not self.pipeline.is_first_stage() and \
           not self.pipeline.is_last_stage()) or \
           (input_stage_list[-1] + 1 == mpu.get_pipe_parallel_world_size() \
           and not self.pipeline.is_first_stage()):
            extra = []
            # 获取之前stage，在input_to_stage_dic中的输入数
            load_idx = sum([len(self.pipeline.input_to_stage_dic[i])
                            for i in range(self.stage_id)])
            for k, v in self.pipeline.input_to_stage_dic.items():
                # 匹配dic中出现的stage，并防止出现空list的情况
                if k == self.stage_id and len(v):
                    o_input = self.pipeline._next_batch()
                    if len(v) == len(o_input):
                        load_idx = 0
                    start_idx = load_idx
                    end_idx = load_idx + len(v)
                    for idx in range(start_idx, end_idx):
                        x = o_input[idx]
                        assert torch.is_tensor(x)
                        mine = x.clone().detach().to(self.device)
                        mine.requires_grad = mine.is_floating_point()
                        extra.append(mine)
                    self.pipeline.pipe_buffers['extra_inputs'][buffer_id] = tuple(extra)
        
        if self.args.wall_clock_breakdown:
            self.timers('batch_input').stop()
            self.timers('pipe_recv_input').start()

        recvd = None

        # Allocate the buffer if necessary
        if self.pipe_recv_buf is None:
            self.pipe_recv_buf = _recv_tensor_meta(self.device)

        assert isinstance(self.pipe_recv_buf, tuple)
        recvd = [None] * len(self.pipe_recv_buf)
        for idx, buffer in enumerate(self.pipe_recv_buf):
            assert torch.is_tensor(buffer)
            p2p.recv_forward(buffer)
            recvd[idx] = buffer.clone().detach()
            recvd[idx].requires_grad = buffer.requires_grad

        recvd = tuple(recvd)

        self.pipeline.pipe_buffers['inputs'][buffer_id] = recvd

        if self.args.wall_clock_breakdown:
            self.timers('pipe_recv_input').stop()

    def _exec_send_grads(self, buffer_id: int):
        if self.args.wall_clock_breakdown:
            self.timers('pipe_send_grad').start()

        inputs = self.pipeline.pipe_buffers['inputs'][buffer_id]


        if isinstance(inputs, torch.Tensor) and inputs.requires_grad:
            assert inputs.grad is not None
            p2p.send_backward(inputs.grad)

        else:
            for idx, buffer in enumerate(inputs):
                # Skip tensors that will not produce a grad
                if not buffer.is_floating_point():
                    assert buffer.grad is None
                    continue
                if buffer.requires_grad:
                    assert buffer.grad is not None
                    p2p.send_backward(buffer.grad)

        # We can free up the input buffer now
        self.pipeline.pipe_buffers['inputs'][buffer_id] = None

        if self.args.wall_clock_breakdown:
            self.timers('pipe_send_grad').stop()

    def _exec_recv_grads(self, buffer_id):
        if self.args.wall_clock_breakdown:
            self.timers('pipe_recv_grad').start()

        outputs = self.pipeline.pipe_buffers['outputs'][buffer_id]

        # Allocate gradient if necessary
        if self.pipeline.grad_layer is None:
            self.pipeline.grad_layer = _create_buffer(outputs, self.device)
        # print('before', self.global_rank, self.grad_layer)

        if isinstance(self.pipeline.grad_layer, torch.Tensor):
            if self.pipeline.grad_layer.requires_grad:
                p2p.recv_backward(self.pipeline.grad_layer)
        else:
            assert isinstance(outputs, tuple), f'Unexcept type {type(outputs)}'
            for idx, buffer in enumerate(self.pipeline.grad_layer):
                if buffer.requires_grad:
                    p2p.recv_backward(buffer)
        # print('after', self.global_rank, self.grad_layer)
        if self.args.wall_clock_breakdown:
            self.timers('pipe_recv_grad').stop()

    def _exec_send_activations_recv_grads(self, buffer_id: int):

        send_buffer, recv_buffer=buffer_id

        if self.args.wall_clock_breakdown:
            self.timers('pipe_send_output_recv_grad').start()

        activations = self.pipeline.pipe_buffers['outputs'][send_buffer]

        outputs = self.pipeline.pipe_buffers['outputs'][recv_buffer]

        # Allocate gradient if necessary
        # if self.pipeline.grad_layer is None:
        self.pipeline.grad_layer = _create_buffer(outputs, self.device)

        # print('before', self.global_rank, self.grad_layer)
        assert isinstance(outputs, tuple)
        for idx, buffer in enumerate(self.pipeline.grad_layer):
            if not activations[idx].is_contiguous():
                activations = list(activations)
                activations[idx] = activations[idx].contiguous()
            if activations[idx].requires_grad:
                p2p.send_forward_recv_backward(activations[idx], buffer)
            else:
                p2p.send_forward(activations[idx])
            if self.args.dealloc_pipeoutput:
                deallocate_output_tensor(activations[idx])

        # print('after', self.global_rank, self.grad_layer)
        if self.args.wall_clock_breakdown:
            self.timers('pipe_send_output_recv_grad').stop()

    def _exec_send_grads_recv_activations(self, buffer_id: int):

        send_buffer, recv_buffer = buffer_id
        if self.args.wall_clock_breakdown:
            self.timers('batch_input').start()
        input_stage_list = list(self.pipeline.input_to_stage_dic.keys())
        if (not self.pipeline.is_first_stage() and \
           not self.pipeline.is_last_stage()) or \
           (input_stage_list[-1] + 1 == mpu.get_pipe_parallel_world_size() \
           and not self.pipeline.is_first_stage()):
            extra = []
            load_idx = sum([len(self.pipeline.input_to_stage_dic[i])
                            for i in range(self.stage_id)])
            for k, v in self.pipeline.input_to_stage_dic.items():
                if k == self.stage_id and len(v):
                    o_input = self.pipeline._next_batch()
                    if len(v) == len(o_input):
                        load_idx = 0
                    start_idx = load_idx
                    end_idx = load_idx + len(v)
                    for idx in range(start_idx, end_idx):
                        x = o_input[idx]
                        assert torch.is_tensor(x)
                        mine = x.clone().detach().to(self.device)
                        mine.requires_grad = mine.is_floating_point()
                        extra.append(mine)
                    self.pipeline.pipe_buffers['extra_inputs'][recv_buffer] = \
                        tuple(extra)

        if self.args.wall_clock_breakdown:
            self.timers('batch_input').stop()
            self.timers('pipe_send_grad_recv_input').start()

        inputs = self.pipeline.pipe_buffers['inputs'][send_buffer]
        recvd = None

        # Allocate the buffer if necessary
        assert isinstance(self.pipe_recv_buf, tuple)
        recvd = [None] * len(self.pipe_recv_buf)
        for idx, buffer in enumerate(self.pipe_recv_buf):
            if buffer.requires_grad and inputs[idx].is_floating_point():
                p2p.send_backward_recv_forward(inputs[idx].grad, buffer)
            else:
                p2p.recv_forward(buffer)
            recvd[idx] = buffer.clone().detach()
            recvd[idx].requires_grad = buffer.requires_grad

        recvd = tuple(recvd)

        self.pipeline.pipe_buffers['inputs'][send_buffer] = None
        self.pipeline.pipe_buffers['inputs'][recv_buffer] = recvd

        if self.args.wall_clock_breakdown:
            self.timers('pipe_send_grad_recv_input').stop()

    def _exec_recompute_recv_grads(self, buffer_id: int):

        if hasattr(self.pipeline.module, 'act_ckpt_func') \
           and self.pipeline.module.act_ckpt_func is pre_checkpoint_func:
            self.pipeline.module.act_ckpt_func = self.pipeline.checkpoint_func_bak
        if hasattr(self.pipeline.module, 'act_ckpt_interval') \
           and self.pipeline.module.act_ckpt_interval != 0:
            self.pipeline.module.act_ckpt_interval = 0

        inputs = self.pipeline.pipe_buffers['inputs'][buffer_id]
        if isinstance(inputs, tuple):
            inputs = tuple(
                [
                    torch.Size(input.tolist())
                    if input.dim() == 1
                    and (input.data[0] == self.args.per_device_train_batch_size or
                        input.data[0] == -1)
                    and not len(input.data) == 1
                    else input
                    for input in inputs
                ]
            )
        elif isinstance(inputs, dict):
            inputs = {k: v.clone() for k, v in inputs.items()}
        else:
            inputs = inputs.clone()

        extra_inputs = self.pipeline.pipe_buffers['extra_inputs'][buffer_id]
        if extra_inputs is not None:
            _zero_grads(extra_inputs)
            if isinstance(extra_inputs,
                          tuple):
                extra_inputs = tuple(t.clone() for t in extra_inputs)
            else:
                extra_inputs = extra_inputs.clone()
            if not isinstance(inputs, tuple):
                inputs = tuple([inputs])
            inputs = inputs + extra_inputs

        self.pipeline.rng_manager.set_recompute_rng_state(buffer_id)
        # do recomputation
        if isinstance(inputs, dict):
            outputs = self.pipeline.module(**inputs)
        else:
            outputs = self.pipeline.module(inputs)

        outputs: tuple[torch.Tensor] = tuple(
                [
                    output
                    if torch.is_tensor(output)
                    else torch.LongTensor(
                            data=(output,) if isinstance(output, int) else output
                            ).to(self.device)
                    for output in outputs
                ]
            )
        self.pipeline.pipe_buffers['outputs'][buffer_id] = outputs

        if self.args.dealloc_pipeoutput:
            if isinstance(outputs, torch.Tensor):
                deallocate_output_tensor(outputs)
            else:
                for idx, values in enumerate(outputs):
                    deallocate_output_tensor(values)

        self.pipeline.rng_manager.restore_bwd_rng_state(buffer_id)

        if self.args.wall_clock_breakdown:
            self.timers('pipe_recv_grad').start()

        # Allocate gradient if necessary
        if self.pipeline.grad_layer is None:
            self.pipeline.grad_layer = _create_buffer(outputs, self.device)

        assert isinstance(outputs, tuple)
        for idx, buffer in enumerate(self.grad_layer):
            if buffer.requires_grad:
                p2p.recv_backward(buffer)
        if self.args.wall_clock_breakdown:
            self.timers('pipe_recv_grad').stop()

    def _reduce_outputs(
            self,
            outputs: Union[torch.Tensor, Tuple[torch.Tensor]],
            reduce: str = 'avg', 
            reduce_dp: bool = True
        ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        if reduce is None:
            return outputs

        if reduce.lower() == 'avg':
            # first sum over all microbatches
            if torch.is_tensor(outputs[0]):
                reduced = sum(outputs)
            else:
                assert isinstance(outputs, (list, tuple))
                reduced = [torch.zeros_like(o) for o in outputs[0]]
                for idx, out in outputs:
                    reduced[idx] += out

            # Average over the microbatches
            reduced = _scale_loss_by_gas(reduced, self.pipeline.tuning_params.gas)

            # Average over DP groups
            if reduce_dp and self.is_data_parallel:
                if torch.is_tensor(reduced):
                    dist.all_reduce(reduced, \
                                    group=mpu.get_data_parallel_group())
                    reduced /= mpu.get_data_parallel_world_size()
                else:
                    for idx in range(len(reduced)):
                        dist.all_reduce(reduced[idx],
                                        group=mpu.get_data_parallel_group())
                        reduced[idx] /= mpu.get_data_parallel_world_size()

            return reduced
        else:
            raise NotImplementedError(f'reduction type {reduce} not supported.')
        
    def _bcast_pipe_scalar(
            self,
            data: torch.Tensor,
            src_rank: Optional[int] = None,
            dtype: torch.dtype = torch.float32
        ) -> torch.Tensor:
        # Default to last stage (e.g., for broadcasting loss)
        if src_rank is None:
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
        assert src_rank in self.grid.pp_group

        if self.global_rank == src_rank:
            result = data.clone().detach()
        else:
            result = torch.Tensor([0.]).type(dtype).to(self.pipeline.device)

        dist.broadcast(tensor=result,
                       src=src_rank,
                       group=mpu.get_pipe_parallel_group())

        return result
    
    def _aggregate_total_loss(self, total_loss: torch.Tensor) -> torch.Tensor:
        # Scale loss, average among DP ranks, and bcast loss to the rest of my
        # DP group
        if self.pipeline.is_last_stage():
            loss = _scale_loss_by_gas(total_loss, self.pipeline.tuning_params.gas)
            self.dp_group_loss = loss.clone().detach()

            ## Average loss across all data-parallel groups
            agg_loss = self.dp_group_loss.clone().detach()
            #print(f'RANK={self.global_rank} bcast SENDER \
            #src={self.global_rank} group={self.grid.pp_group}', flush=True)
            if self.is_data_parallel:
                dist.all_reduce(agg_loss, group=mpu.get_data_parallel_group())
                agg_loss /= mpu.get_data_parallel_world_size()

            assert self.global_rank in self.grid.pp_group
            losses = torch.Tensor([self.dp_group_loss, agg_loss]).to(self.device)
            dist.broadcast(tensor=losses,
                           src=self.global_rank,
                           group=mpu.get_pipe_parallel_group())

        else:
            # Get loss from last stage
            src_rank = self.grid.stage_to_global(self.num_stages - 1)
            assert src_rank in self.grid.pp_group
            losses = torch.Tensor([0., 0.]).to(self.device)
            dist.broadcast(tensor=losses,
                           src=src_rank,
                           group=self.grid.get_pipe_parallel_group())
            self.dp_group_loss = losses[0].clone().detach()
            agg_loss = losses[1].clone().detach()

        return agg_loss
    
    def allreduce_bucket(self, bucket: int) -> torch.Tensor:
        tensor = self.flatten(bucket)

        tensor_to_allreduce = tensor

        if self.communication_data_type() != tensor.dtype:
            tensor_to_allreduce = tensor.to(self.communication_data_type())

        tensor_to_allreduce.div_(mpu.get_data_parallel_world_size())
        dist.all_reduce(tensor_to_allreduce,
                        group=mpu.get_data_parallel_group())

        if self.communication_data_type() != tensor.dtype and \
            tensor is not tensor_to_allreduce:
            tensor.copy_(tensor_to_allreduce)

        return tensor

    def allreduce_and_copy(self, small_bucket: Tuple[torch.Tensor]):
        allreduced = self.allreduce_bucket(small_bucket)
        for buf, synced in zip(small_bucket,
                               self.unflatten(allreduced, small_bucket)):
            buf.copy_(synced)

    def allreduce_no_retain(self, bucket: torch.Tensor, numel_per_bucket: int):
        small_bucket = []
        numel = 0
        for tensor in bucket:
            small_bucket.append(tensor)
            numel = numel + tensor.numel()
            if numel > numel_per_bucket:
                self.allreduce_and_copy(small_bucket)
                small_bucket = []
                numel = 0
        if len(small_bucket) > 0:
            self.allreduce_and_copy(small_bucket)
    
    def buffered_allreduce_fallback(self, elements_per_buffer: int):
        grads = []
        for param_name, param in self.pipeline.module.named_parameters():
            if param.grad is None:
                # In cases where there is an imbalance of empty grads across
                # ranks we must create empty grads, this will ensure that every
                # rank is reducing the same size. In some cases it may make
                # sense in the future to support the ability to average not
                # w.r.t. world size but with a different value.
                param.grad = torch.zeros(param.size(),
                                         dtype=param.dtype,
                                         device=param.device)
                grads.append(param.grad.data)
            else:
                grad_data = param.grad.data
                grads.append(grad_data)

        split_buckets = split_half_float_double(grads)

        for i, bucket_tuple in enumerate(split_buckets):
            bucket_type, bucket = bucket_tuple
            self.allreduce_no_retain(bucket,
                                     numel_per_bucket=elements_per_buffer)

    def allreduce_gradients(self, bucket_size: int = 500000000):
        # Pass (PP) gas boundary flag to optimizer (required for zero)
        # self.optimizer.is_gradient_accumulation_boundary = \
        # self.is_gradient_accumulation_boundary()
        # if self.pipeline.is_gradient_accumulation_boundary():
        if self.args.zero_stage is not None:
            self.pipeline.optimizer.reduce_gradients()
        else:
            self.buffered_allreduce_fallback(elements_per_buffer=bucket_size)

    def communication_data_type(self) -> torch.dtype:
        if self.args.fp16:
            return torch.float16
        return torch.float32


class NetCalculationEngine:

    def __init__(
            self,
            pipeline,
            loss_fn: Callable
        ):

        self.args = pipeline.args
        self.timers = pipeline.timers
        self.device = pipeline.device
        self.optimizer = pipeline.optimizer

        self.loss_fn = loss_fn
        self.pipeline = pipeline

        self._compute_loss = True
        self.do_train = True

        if hasattr(self.pipeline.module, 'act_ckpt_interval'):
            self.act_ckpt_interval_backup = self.pipeline.module.act_ckpt_interval

        self.grid = self.pipeline.module._grid
        self.loss = torch.tensor(0.0).to(self.device)

        self.logits = []
        self.labels = []
        self.fwd_outputs = []

        if self.args.half_precision_backend == "cuda_amp":
            self.scaler = GradScaler()

        if self.pipeline.is_last_stage():
            if self.loss_fn is not None:
                self.loss_model = self.loss_fn
            elif hasattr(self.pipeline.module, 'loss_fn'):
                self.loss_model = self.pipeline.module.loss_fn
            else:
                raise ValueError("Missing loss function")

        # Initialize pipeline communicators. Just send a 0.
        # if is_even(self.grid.get_stage_id()):
        if self.grid.get_stage_id() % 2 == 0:
            if not self.pipeline.is_last_stage():
                p2p.send_forward(self.loss)
            if not self.pipeline.is_first_stage():
                p2p.recv_forward(self.loss)
        else:
            if not self.pipeline.is_first_stage():
                p2p.recv_forward(self.loss)
            if not self.pipeline.is_last_stage():
                p2p.send_forward(self.loss)

    def _exec_load_micro_batch(self, buffer_id: int):

        if self.args.wall_clock_breakdown:
            self.timers('batch_input').start()

        batch = self.pipeline._next_batch()

        if self.pipeline.is_first_stage():
            if isinstance(batch, dict):
                for k, v in batch.items():
                    batch[k] = v.to(self.device)
                self.pipeline.pipe_buffers['inputs'][buffer_id] = batch
            else:
                loaded = []
                load_idx = len(self.pipeline.input_to_stage_dic[0])
                # if isinstance(batch, dict):
                #     batch = list(batch.values())
                for x in batch[:load_idx]:
                    assert torch.is_tensor(x)
                    mine = x.clone().detach().to(self.device)
                    mine.requires_grad = mine.is_floating_point()
                    loaded.append(mine)
                self.pipeline.pipe_buffers['inputs'][buffer_id] = tuple(loaded)

        if self.pipeline.is_last_stage():
            loaded = []

            # 计算input_to_stage_dic所需的输入数量
            load_idx = sum([len(self.pipeline.input_to_stage_dic[i])
                            for i in self.pipeline.input_to_stage_dic])

            if not self.args.text_generation:
                if isinstance(batch, dict):
                    for k, v in batch.items():
                        batch[k] = v.to(self.device)
                    self.pipeline.pipe_buffers['labels'][buffer_id] = batch
                else:
                    # 如果batch的数据少于load_idx，则是使用了split input，就不需要
                    # 再做划分
                    if load_idx < len(batch):
                        # 去掉last stage不需要的部分
                        batch = batch[load_idx:]
                    for x in batch:
                        assert torch.is_tensor(x)
                        x = x.to(self.device).detach()
                        loaded.append(x)
                    loaded = tuple(loaded)

                    self.pipeline.pipe_buffers['labels'][buffer_id] = loaded
            else:
                self.pipeline.pipe_buffers['labels'][buffer_id] = None


        if self.args.wall_clock_breakdown:
            self.timers('batch_input').stop()

    def _exec_forward_pass(self, buffer_id: int):
        mem_status('BEFORE FWD', reset_max=True)

        if self.timers('backward_microstep').started_ and \
           self.timers('backward').started_ and \
           self.pipeline.is_last_stage():
            self.timers('backward_microstep').stop()
            self.timers('backward').stop()

        inputs = self.pipeline.pipe_buffers['inputs'][buffer_id]
        _zero_grads(inputs)

        if isinstance(inputs, tuple):
            inputs = tuple(
                [
                    torch.Size(input.tolist())
                    if input.dim() == 1
                    and (input.data[0] == self.args.per_device_train_batch_size or
                        input.data[0] == -1)
                    and not len(input.data) == 1
                    else input
                    for input in inputs
                ]
            )
        elif isinstance(inputs, dict):
            inputs = {k: v.clone() for k, v in inputs.items()}
        else:
            inputs = inputs.clone()

        # Zero out the gradients each time we use the tensor because only the
        # data in tensor changes across batches
        extra_inputs = self.pipeline.pipe_buffers['extra_inputs'][buffer_id]
        if extra_inputs is not None:
            _zero_grads(extra_inputs)
            if isinstance(extra_inputs,
                          tuple):
                extra_inputs = tuple(t.clone() for t in extra_inputs)
            else:
                extra_inputs = extra_inputs.clone()
            if not isinstance(inputs, tuple):
                inputs = tuple([inputs])
            inputs = inputs + extra_inputs

        self.timers('forward_microstep').start()
        self.timers('forward').start()

        if isinstance(inputs, dict):
            outputs = self.pipeline.module(**inputs)
        else:
            outputs = self.pipeline.module(inputs)

        self.timers('forward').stop()
        self.timers('forward_microstep').stop()

        # Optionally compute loss on the last device
        if self.pipeline.is_last_stage():
            if self._compute_loss and self.loss_model is not None:
                labels = self.pipeline.pipe_buffers['labels'][buffer_id]
                # print(outputs.shape, labels.shape)
                if not self.args.text_generation:
                    self.loss = self.loss_model(outputs, labels)
                if self.args.return_logits and not self.do_train:
                    self.logits.append(outputs)
                    self.labels.append(labels)
            else:
                # Some models just return loss from forward()
                self.loss = outputs

            if isinstance(self.loss, torch.Tensor):
                self.fwd_outputs.append(self.loss.detach())

                if self.pipeline.total_loss is None:
                    self.pipeline.total_loss = torch.zeros_like(self.loss)
                self.pipeline.total_loss += self.loss.detach()
            else:
                self.fwd_outputs.append([l.detach() for l in self.loss])

                if self.pipeline.total_loss is None:
                    self.pipeline.total_loss = \
                        [torch.zeros_like(l) for l in self.loss]
                for idx, l in enumerate(self.loss):
                    self.pipeline.total_loss[idx] += l.detach()
        else:
            if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
            assert isinstance(outputs, (tuple, list))
            outputs: tuple[torch.Tensor] = tuple(
                [
                    output
                    if torch.is_tensor(output)
                    else torch.LongTensor(
                            data=(output,) if isinstance(output, int) else output
                            ).to(self.device)
                    for output in outputs
                ]
            )
            self.pipeline.pipe_buffers['outputs'][buffer_id] = outputs

    def _exec_restore_recompute_status(self):
        if hasattr(self.pipeline.module, 'act_ckpt_func') and \
            self.pipeline.module.act_ckpt_func is pre_checkpoint_func:
            self.pipeline.module.act_ckpt_func = self.pipeline.checkpoint_func_bak
        if hasattr(self.pipeline.module, 'act_ckpt_interval') \
            and self.pipeline.module.act_ckpt_interval == 0:
            self.pipeline.module.act_ckpt_interval = self.act_ckpt_interval_backup

    def _exec_precheckpoint_forward_pass(self, buffer_id: int):

        mem_status('BEFORE FWD', reset_max=True)

        if hasattr(self.pipeline.module, 'act_ckpt_func') \
            and self.pipeline.module.act_ckpt_func \
                is not pre_checkpoint_func:
            self.pipeline.checkpoint_func_bak = self.pipeline.module.act_ckpt_func
            self.pipeline.module.act_ckpt_func = pre_checkpoint_func

        if isinstance(self.pipeline.pipe_buffers['inputs'][buffer_id], tuple):
            inputs = tuple(t.clone() for t in self.pipeline.pipe_buffers['inputs'][buffer_id])
        else:
            inputs = self.pipeline.pipe_buffers['inputs'][buffer_id].clone()

        assert not self.pipeline.is_last_stage(), \
            'last stage should not pre checkpoint recompute'


        # Zero out the gradients each time we use the tensor because only the
        # data in tensor changes across batches
        if self.pipeline.pipe_buffers['extra_inputs'][buffer_id] is not None:
            if isinstance(self.pipeline.pipe_buffers['extra_inputs'][buffer_id], tuple):
                extra_inputs = tuple(t.clone() \
                                     for t in self.pipeline.pipe_buffers['extra_inputs'][buffer_id])
            else:
                extra_inputs = \
                   self.pipeline.pipe_buffers['extra_inputs'][buffer_id].clone()
            if not isinstance(inputs, tuple):
                inputs = tuple([inputs])
            inputs = inputs + extra_inputs

        _zero_grads(inputs)
    
        self.pipeline.rng_manager.store_fwd_rng_state(buffer_id)

        if isinstance(inputs, dict):
            outputs = self.pipeline.module(**inputs)
        else:
            outputs = self.pipeline.module(inputs)

        if isinstance(outputs, torch.Tensor):
                outputs = [outputs]
        assert isinstance(outputs, (tuple, list))
        outputs: tuple[torch.Tensor] = tuple(
            [
                output
                if torch.is_tensor(output)
                else torch.LongTensor(
                        data=(output,) if isinstance(output, int) else output
                        ).to(self.device)
                for output in outputs
            ]
        )
        self.pipeline.pipe_buffers['outputs'][buffer_id] = outputs

    def _exec_backward_pass(self, buffer_id: int):
        assert self.optimizer is not None, \
            "must provide optimizer during init in order to use backward"

        mem_status('BEFORE BWD', reset_max=True)

        # The last stage just runs backward on the loss using DeepSpeed's
        # typical mechanisms.
        if self.args.wall_clock_breakdown:
            self.timers('backward_microstep').start()
            self.timers('backward').start()
            self.timers('backward_inner_microstep').start()
            self.timers('backward_inner').start()

        if self.pipeline.is_last_stage():
            self.loss = _scale_loss_by_gas(self.loss.float(), self.pipeline.tuning_params.gas)
            # super().backward(self.loss)
            if self.args.half_precision_backend == "apex":
            # AMP requires delaying unscale when inside gradient accumulation boundaries
            # https://nvidia.github.io/apex/advanced.html#gradient-accumulation-across-iterations
                delay_unscale = \
                    not self.pipeline.is_gradient_accumulation_boundary()
                with amp.scale_loss(self.loss,
                                    self.optimizer,
                                    delay_unscale=delay_unscale) as scaled_loss:
                    scaled_loss.backward()
            elif self.args.half_precision_backend == "cuda_amp":
                self.scaler.scale(self.loss).backward()
            elif self.args.fp16 or self.args.zero_stage == 1:
                self.optimizer.backward(self.loss)
            else:
                self.loss.backward()

            self.timers('backward_inner').stop()
            self.timers('backward_inner_microstep').stop()
            mem_status('AFTER BWD')

            self.pipeline.tuning_params.micro_steps += 1

        else:
            outputs = self.pipeline.pipe_buffers['outputs'][buffer_id]

            grad_tensors = self.pipeline.grad_layer

            # This handles either a single tensor or tuple of tensors.
            if isinstance(outputs, tuple):
                out_tensors = tuple([t for t in outputs if t.requires_grad])
                if not len(out_tensors) == len(grad_tensors):
                    grad_tensors = tuple([grad_tensors[t] for t in range(len(outputs))
                                    if outputs[t].requires_grad])
                if self.args.dealloc_pipeoutput:
                    custom_backward(out_tensors, grad_tensors)
                else:
                    torch.autograd.backward(tensors=out_tensors,
                                            grad_tensors=grad_tensors)
            else:
                if self.args.dealloc_pipeoutput:
                    custom_backward(outputs, grad_tensors)
                else:
                    torch.autograd.backward(tensors=(outputs, ),
                                            grad_tensors=(grad_tensors, ))

            # Free up the memory from the output of forward()
            self.pipeline.pipe_buffers['outputs'][buffer_id] = None
            grad_tensors = None

            if self.args.wall_clock_breakdown:
                self.timers('backward_microstep').stop()
                self.timers('backward').stop()
                self.timers('backward_inner').stop()
                self.timers('backward_inner_microstep').stop()

            self.pipeline.tuning_params.micro_steps += 1

            mem_status('AFTER BWD')

    def _exec_optimizer_step(self, lr_kwargs: Optional[dict] = None):
        if self.args.wall_clock_breakdown:
            self.timers('step_microstep').start()
            self.timers('step').start()
        mem_status('BEFORE STEP', reset_max=True)

        if self.args.max_grad_norm > 0.0:
            if not (self.args.half_precision_backend or self.args.fp16):
                self.clip_fp32_gradients()
            elif self.args.half_precision_backend == "apex":
                # AMP's recommended way of doing clipping
                # https://nvidia.github.io/apex/advanced.html#gradient-clipping
                master_params = amp.master_params(self.optimizer)
                clip_grad_norm_(parameters=master_params,
                                max_norm=self.args.max_grad_norm,
                                mpu=mpu)
            elif self.args.half_precision_backend == "cuda_amp":
                if self.scaler._scale is None:
                    self.scaler._lazy_init_scale_growth_tracker(self.device)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                                parameters=self.pipeline.module.parameters(),
                                max_norm=self.args.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
        if not self.args.half_precision_backend == "cuda_amp":
            self.optimizer.step()
        if self.pipeline.update_lr:
            self.pipeline.lr_scheduler.step()

        if (not self.args.fp16 and
            self.args.half_precision_backend not in ["cuda_amp", "apex"]):
            self.zero_grad()
        else:
            self.optimizer.zero_grad()

        mem_status('AFTER STEP')

        if self.args.wall_clock_breakdown:
            self.timers('step_microstep').stop()
            self.timers('step').stop()
            if self.pipeline.tuning_params.global_steps % \
               self.args.logging_steps == 0:
                self.timers.log([
                    'batch_input',
                    'forward_microstep',
                    'backward_microstep',
                    'backward_inner_microstep',
                    'step_microstep'
                ], normalizer=self.args.logging_steps)
            if self.pipeline.tuning_params.global_steps % \
               self.args.logging_steps == 0:
                self.timers.log([
                    'forward',
                    'backward',
                    'backward_inner',
                    'backward_allreduce',
                    'backward_tied_allreduce',
                    'step'
                ], normalizer=self.args.logging_steps)

            if self.args.wall_clock_breakdown and \
            (self.pipeline.tuning_params.global_steps+ 1) \
                % self.args.logging_steps == 0:
                should_log = (mpu.get_data_parallel_rank() == 0 and
                            mpu.get_model_parallel_rank() == 0)
                see_memory_usage('After {} iterations'.format(
                    self.pipeline.tuning_params.global_steps),
                    should_log, ranks=[dist.get_rank()])

    def zero_grad(self):
        """
        Zero parameter grads.
        """
        for param_name, param in self.pipeline.module.named_parameters():
            param.grad = None

    def clip_fp32_gradients(self):
        torch.nn.utils.clip_grad_norm_(
            parameters=self.pipeline.module.parameters(),
            max_norm=self.args.max_grad_norm)

class PipelineEngine:
    """ A training engine hybrid pipeline, data, and model parallel training.
    """
    def __init__(
            self,
            model: nn.Module,
            args: MerakArguments,
            optimizer: Optional[Callable] = None,
            lr_scheduler: Optional[Callable] = None,
            tuning_params: Optional[Callable] = None,
            dataloader: torch.utils.data.DataLoader = None,
            loss_fn: Optional[Callable] = None,
        ):

        self.module = model
        self.args = args
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.tuning_params = tuning_params
        self.loss_fn = loss_fn

        self.micro_batch_size = self.tuning_params.mbs
        self.micro_batches = self.tuning_params.gas

        self.update_lr = True
        self.batch_fn = None
        self.grad_layer = None
        self._force_grad_boundary = False
        self.input_to_stage_dic = model.input_to_stage_dic
        #stores the loss for the entire batch
        self.total_loss = None
        self.checkpoint_func_bak = None

        amp_engine = MixedPrecisionConfig(self.args)
        self.data_iterator = RepeatingLoader(dataloader)

        if self.args.use_cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda', args.local_rank)

        #stores the loss for the current micro batch being processed
        self.data_parallel_group = mpu.get_data_parallel_group()
        self.dp_world_size = mpu.get_data_parallel_world_size()
        pp_size = mpu.get_pipe_parallel_world_size()
        tp_size = mpu.get_model_parallel_world_size()

        # assert not self.elasticity_enabled(), \
        # "Elasticity is not currently supported with pipeline parallelism."
        if not self.args.text_generation:
            self.optimizer = self.optimizer(self.module)
            self.lr_scheduler = self.lr_scheduler(self.optimizer)

        if self.args.fp16 or self.args.half_precision_backend == 'apex':
            self.module, self.optimizer = amp_engine.configure(
                self.optimizer, self.module, self.device
            )

        if isinstance(model, PipelineModule):
            self.module.configure_distributed_model(self.device)
        else:
            self.module.to(self.device)
        if self.args.zero_stage == 1:
            dynamic_loss_args = amp_engine.dynamic_loss_args if args.fp16 else None
            self.optimizer = configure_zero_optimizer(self.optimizer, dynamic_loss_args)
        if not self.args.no_tie_modules:
            self.module.tie_modules()

        self.timers = SynchronizedWallClockTimer()
        self.rng_manager = RNGManager()
        self.comm_engine = CommunicationEngine(self)
        self.netcal_engine = NetCalculationEngine(self, loss_fn)
        self.global_rank = self.comm_engine.global_rank

        # set pipeline scheduler
        TrainScheduleClass = self.scheduler_method()
        self.train_sched = TrainScheduleClass(
            micro_batches=self.micro_batches,
            stages=self.comm_engine.num_stages,
            stage_id=self.comm_engine.stage_id
        )

        model_parameters = filter(lambda p: p.requires_grad,
                                  self.module.parameters())
        num_params = sum([p.numel() for p in model_parameters])
        unique_params = num_params
        # Subtract tied parameters if we don't own them
        if hasattr(self.module, 'tied_comms') and self.module.tied_comms:
            tied_params = 0
            for key, d in self.module.tied_comms.items():
                if self.global_rank != min(d['ranks']):
                    tied_params += sum(p.numel() for p in d['module'].parameters())
            unique_params -= tied_params
        params_tensor = torch.LongTensor(data=[num_params,
                                               unique_params]).to(self.device)
        dist.all_reduce(params_tensor,
                        group=mpu.get_model_parallel_group())

        params_tensor = params_tensor.tolist()
        total_params = params_tensor[0]
        unique_params = params_tensor[1]
        # if self.grid.data_parallel_id == 0:
        if self.comm_engine.grid.data_parallel_id == 0:
            if hasattr(self.module, '_local_start'):
                layers = self.module._local_stop - self.module._local_start
                log_layers = \
                    f'LAYERS={layers} ' \
                    f'[{self.module._local_start}, {self.module._local_stop}) '
            else:
                log_layers = f'LAYERS={self.args.num_layers} ' \
                             f'[0, {self.args.num_layers}) '
            logger.info(
                f'RANK={self.global_rank} '
                f'STAGE={self.comm_engine.stage_id} '
                f'{log_layers}'
                f'STAGE_PARAMS={num_params} ({num_params/1e6:0.3f}M) '
                f'TOTAL_PARAMS={total_params} ({total_params/1e6:0.3f}M) '
                f'UNIQUE_PARAMS={unique_params} ({unique_params/1e6:0.3f}M)')

        # Pipeline buffers
        self.num_pipe_buffers = 0
        self.pipe_buffers = {
            'inputs' : [],   # batch input and received activations
            'labels' : [],   # labels from batch input
            'outputs' : [],  # activations
            'extra_inputs' : [],
            # 'output_tensors' : [], # tensor object to preserve backward graph
        }

        if mpu.get_data_parallel_rank() == 0 and \
            mpu.get_model_parallel_rank() == 0:
            set_timer_log_rank([dist.get_rank()])
        else:
            set_timer_log_rank([])

        self.timers._init()

    def is_gradient_accumulation_boundary(self):
        return self.tuning_params.micro_steps % \
            self.args.gradient_accumulation_steps == 0

    def train_batch(self, data_iter: Optional[torch.utils.data.DataLoader] = None) -> torch.Tensor:
        """Progress the pipeline to train the next batch of data. The engine
        will ingest ``self.train_batch_size()`` total samples collectively
        across all workers.


        An iterator that over training data should be provided as an argument
        unless ``deepspeed.initialize()`` was provided a training set. In that
        event, the training data will automatically be read.


        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be
            pulled from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt
            training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.
            RepeatingLoader` that wraps data loaders to automatically restart
            upon a ``StopIteration``.

        Args:
            data_iter (Iterator, optional): Iterator of training data.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """
        if not torch._C.is_grad_enabled():
            raise RuntimeError(
                f'train_batch() requires gradients enabled. Use eval_batch()'
                f'instead.')

        self.module.train()
        self.do_train = True
        self.total_loss = None
        self._compute_loss = True

        # Do the work
        self.timers('train_batch').start()

        
        self._exec_schedule(self.train_sched)
        self.agg_train_loss = self.comm_engine._aggregate_total_loss(
                                                        self.total_loss)
        self.timers('train_batch').stop()

        if self.tuning_params.global_steps % self.args.logging_steps == 0:
            if self.global_rank == 0:
                elapsed = self.timers('train_batch').elapsed(reset=True)
                # print(self.train_batch_size(), elapsed)
                iter_time = elapsed / self.args.logging_steps
                tput = self.tuning_params.total_batch_size / iter_time
                lr = self.optimizer.param_groups[0]['lr']
                print(f'steps: {self.tuning_params.global_steps} '
                      f'loss: {self.agg_train_loss:0.4f} '
                      f'iter time (s): {iter_time:0.3f} '
                      f'total samples/sec: {tput:0.3f} '
                      f'learning rate: {lr:.4e}')

        if self.args.wall_clock_breakdown \
           and self.tuning_params.global_steps % self.args.logging_steps == 0 \
           and mpu.get_data_parallel_rank()==0:
            self.timers.log([
                'pipe_send_output',
                'pipe_send_grad',
                'pipe_recv_input',
                'pipe_recv_grad', 
                'pipe_send_grad_recv_input',
                'pipe_send_output_recv_grad',
            ], normalizer=self.args.logging_steps)

        # TODO: should return precisely what loss returned and allow others to
        # be queried?
        return self.agg_train_loss

    def eval_batch(
            self,
            batch_fn: Optional[Callable] = None,
            compute_loss: Optional[bool] = True,
            reduce_output: str = 'avg'
        ) -> Tuple[Union[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]:
        """Evaluate the pipeline on a batch of data from ``data_iter``. The
        engine will evaluate ``self.train_batch_size()`` total samples
        collectively across all workers.

        This method is equivalent to:

        .. code-block:: python

            module.eval()
            with torch.no_grad():
                output = module(batch)

        .. warning::
            A total of ``self.gradient_accumulation_steps()`` entries will be
            pulled from ``data_iter`` by each pipeline. There must be sufficient
            data left in ``data_iter`` or else a ``StopIteration`` will halt
            training.

            DeepSpeed provides a convenience class :class:`deepspeed.utils.
            RepeatingLoader` that wraps data loaders to automatically restart
            upon a ``StopIteration``.

        Args:
            data_iter (Iterator): Iterator of data to evaluate.

        Returns:
            The arithmetic mean of the losses computed this batch.
        """

        self.module.eval()
        self.netcal_engine.do_train = False

        if batch_fn:
            self.batch_fn = batch_fn

        eval_output = None
        if self.args.return_logits:
            self.netcal_engine.logits = []
            self.netcal_engine.labels = []

        self._compute_loss = compute_loss

        # Do the work
        sched = InferenceSchedule(micro_batches=self.micro_batches,
                                           stages=self.comm_engine.num_stages,
                                           stage_id=self.comm_engine.stage_id)
        self.timers('eval_batch').start()
        with torch.no_grad():
            self._exec_schedule(sched)

        if self.is_last_stage():
            eval_output = self.comm_engine._reduce_outputs(
                                                self.netcal_engine.fwd_outputs,
                                                reduce=reduce_output)
        self.timers('eval_batch').stop()

        if compute_loss:
            eval_output = self.comm_engine._bcast_pipe_scalar(eval_output)
        else:
            eval_output = 0.

        if self.tuning_params.global_steps % self.args.logging_steps == 0:
            if self.global_rank == 0 and not self.args.text_generation:
                elapsed = self.timers('eval_batch').elapsed(reset=True)
                # print(self.train_batch_size(), elapsed)
                iter_time = elapsed / self.args.logging_steps
                print(f'Evaluation: '
                      f'steps: {self.tuning_params.global_steps} '
                      f'loss: {eval_output.mean().item():0.4f} '
                      f'iter time (s): {iter_time:0.3f} '
                      )

        # Restore the training iterator
        # self.set_dataiterator(train_iterator)
        if self.args.return_logits and self.is_last_stage():
            return (eval_output,
                    self.netcal_engine.logits,
                    self.netcal_engine.labels)
        else:
            return eval_output, None, None

    def _reserve_pipe_buffers(self, num_buffers: int):
        """Ensure that each pipeline buffer has at least ``num_buffers`` slots.

        This method only reserves slots and does not allocate tensors.

        Args:
            num_buffers (int): The number of buffers to reserve.
        """
        if self.num_pipe_buffers >= num_buffers:
            return

        num_added = num_buffers - self.num_pipe_buffers
        for key in self.pipe_buffers:
            self.pipe_buffers[key].extend([None] * num_added)
        self.num_pipe_buffers = num_buffers

    def scheduler_method(self) -> PipeSchedule:
        if self.args.train_schedule == '1f1b':
            TrainScheduleClass = schedule.MergeP2PTrainSchedule
        elif self.args.train_schedule == 'ds_default':
            TrainScheduleClass = schedule.TrainSchedule
        else:
            assert self.micro_batches >= mpu.get_pipe_parallel_world_size(), \
            'please fill pipelines with larger number of microbatches'
            '(gradient_accumulation_steps in training args).'
            if self.args.train_schedule == 'pre_recompute_1f1b':
                TrainScheduleClass = schedule.PreRecomputeTrainSchedule
            elif self.args.train_schedule == 'last_no_recompute_1f1b':
                if self.is_last_stage():
                    self.module.act_ckpt_interval = 0
                TrainScheduleClass = schedule.LastNoRecomputeTrainSchedule
            elif self.args.train_schedule == 'full_critical_path_1f1b':
                assert self.args.activation_checkpoint_ratio is not None, \
                'Please set args activation_checkpoint_ratio'
                assert self.micro_batches >= 4, \
                'number of microbatches (gradient_accumulation_steps in training args)'
                'should be larger than 4.'
                if self.is_last_stage():
                    self.module.act_ckpt_interval = 0
                TrainScheduleClass = schedule.FullCriticalPathTrainSchedule
                if mpu.get_pipe_parallel_world_size() > 2:
                    self.module.act_ckpt_ratio[-2] = \
                        self.module.act_ckpt_ratio[-3]
                # else:
                #     ## full critical path schedule only support stage > 2 for
                #     TrainScheduleClass = schedule.LastNoRecomputeTrainSchedule
            else:
                raise NotImplementedError(f' train schedule {self.train_schedule} not supported.')
        return TrainScheduleClass

    def _exec_schedule(self, pipe_schedule: PipeSchedule):

        # A map of PipeInstruction types to methods. Each method will be 
        # executed with the kwargs provided to the PipeInstruction from the 
        # scheduler.
        _INSTRUCTION_MAP = {
            exec_schedule.OptimizerStep: self.netcal_engine._exec_optimizer_step,
            exec_schedule.ReduceGrads: self.comm_engine._exec_reduce_grads,
            exec_schedule.ReduceTiedGrads: self.comm_engine._exec_reduce_tied_grads,
            exec_schedule.LoadMicroBatch: self.netcal_engine._exec_load_micro_batch,
            exec_schedule.ForwardPass: self.netcal_engine._exec_forward_pass,
            exec_schedule.BackwardPass: self.netcal_engine._exec_backward_pass,
            exec_schedule.SendActivation: self.comm_engine._exec_send_activations,
            exec_schedule.RecvActivation: self.comm_engine._exec_recv_activations,
            exec_schedule.SendGrad: self.comm_engine._exec_send_grads,
            exec_schedule.RecvGrad: self.comm_engine._exec_recv_grads,
            exec_schedule.SendActivationRecvGrad: \
                self.comm_engine._exec_send_activations_recv_grads,
            exec_schedule.SendGradRecvActivation: \
                self.comm_engine._exec_send_grads_recv_activations,
                    
            exec_schedule.PreCheckpointForwardPass: \
                self.netcal_engine._exec_precheckpoint_forward_pass,
            exec_schedule.RecomputeRecvGrad: \
                self.comm_engine._exec_recompute_recv_grads,
            exec_schedule.RestoreRecomputeStatus: \
                self.netcal_engine._exec_restore_recompute_status,
        }
        # Reserve and reset buffers.
        self._reserve_pipe_buffers(pipe_schedule.num_pipe_buffers())
        self.netcal_engine.fwd_outputs = []

        # For each step in the schedule
        for step_cmds in pipe_schedule:
            # print(self.global_rank, step_cmds)
            # For each instruction in the step
            for cmd in step_cmds:
                if type(cmd) not in _INSTRUCTION_MAP:
                    raise RuntimeError(
                        f'{self.__class__.__name__} \
                            does not understand instruction {repr(cmd)}'
                    )

                # Equivalent to: self._exec_forward_pass(buffer_id=0)
                # self._exec_instr = \
                #     MethodType(_INSTRUCTION_MAP[type(cmd)], self)
                # self._exec_instr(**cmd.kwargs)
                _INSTRUCTION_MAP[type(cmd)](**cmd.kwargs)

    def is_first_stage(self) -> bool:
        """True if this process is in the first stage in the pipeline."""
        return self.comm_engine.stage_id == 0

    def is_last_stage(self) -> bool:
        """True if this process is in the last stage in the pipeline."""
        return self.comm_engine.stage_id == self.comm_engine.num_stages - 1
    
    def reset_dataiterator(self, iterator):
        del self.data_iterator
        self.data_iterator = iterator

    def _next_batch(self) -> Tuple[torch.Tensor]:
        batch = None
        if not self.args.text_generation and self.data_iterator is not None:
            if not isinstance(self.data_iterator, \
                (_MultiProcessingDataLoaderIter, _SingleProcessDataLoaderIter)):
                self.data_iterator = iter(self.data_iterator)

            batch = next(self.data_iterator)

        if self.batch_fn:
            batch = self.batch_fn(batch)

        return batch
