# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Yck (eyichenke@gmail.com)
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


# the code here are adapted from https://github.com/microsoft/DeepSpeed/releases/tag/v0.5.8/deepspeed/runtime/zero/stage1.py

import math
import torch
import torch.distributed as dist
import torch.optim as optim

from collections import defaultdict
from typing import Dict, List, Set, Any, Optional, Tuple
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from Merak import print_rank_0
from Merak.merak_args import get_args

from .utils import (get_group_alignment_padding,
                    DynamicLossScaler,
                    LossScaler,
                    get_grad_norm,
                    _range_check,)
from .. import mpu
from ..amp.utils import CheckOverflow
from ..printer import logger, log_dist, see_memory_usage

ZERO_SUPPORTED_OPTIMIZERS = [
    torch.optim.Adam,
    torch.optim.AdamW,
    torch.optim.SGD,
    torch.optim.ASGD,
    torch.optim.Adadelta,
    torch.optim.Adamax,
    
]


def configure_zero_optimizer(optimizer: optim.Optimizer, dynamic_loss_scale_args=None) -> optim.Optimizer:
    args = get_args()

    if type(optimizer) not in ZERO_SUPPORTED_OPTIMIZERS:
        assert (args.zero_allow_untested_optimizer), \
            "You are using an untested ZeRO Optimizer. " \
            "Please set <'zero_allow_untested_optimizer': true> " \
            "in the merak arguments to use it."

    optimizer = ZeroOptimizer_Stage1(
        optimizer,
        dynamic_loss_scale=args.fp16,
        dynamic_loss_args=dynamic_loss_scale_args,
        clip_grad=args.max_grad_norm,
        allgather_size=args.zero_allgather_bucket_size,
        max_elements_per_comm=args.zero_reduce_bucket_size,
        mpu=mpu,
        postscale_gradients=not args.prescale_gradients,
        gradient_predivide_factor=args.gradient_predivide_factor,

        )

    return optimizer


class ZeroOptimizer_Stage1(object):
    """
    ZeroOptimizer_Stage1 designed to 
    reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: 
    Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    This version aligns with stage-1 in the paper above.
    """
    def __init__(self,
                 init_optimizer,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 mpu=None,
                 allgather_size=5e8,
                 clip_grad=0.0,
                 max_elements_per_comm=5e8,
                 postscale_gradients=True,
                 gradient_predivide_factor=1.0,
                 gradient_average=True,
                 communication_data_type=torch.float16):

        self.flatten = _flatten_dense_tensors
        self.unflatten = _unflatten_dense_tensors

        self.optimizer = init_optimizer

        self.mpu = mpu
        self.clip_grad = clip_grad

        self.verbose = verbose
        self.dp_process_group = self.mpu.get_data_parallel_group()

        self.postscale_gradients = postscale_gradients
        self.gradient_predivide_factor = gradient_predivide_factor
        self.gradient_average = gradient_average

        self.allgather_size = allgather_size
        self.is_gradient_accumulation_boundary = True

        self.communication_data_type = communication_data_type

        self.max_elements_per_comm = max_elements_per_comm
        logger.info("max_elements_per_comm={}".format(max_elements_per_comm))

        # param flattened by groups
        self.fp16_groups = []
        self.fp16_groups_flat = []

        # Setup bookkeeping data structures depending on partitioning type

        # parallel_sub_partitioned_fp16_groups[group-idx] -> [comm-ids] 
        # -> [rank-ids]
        self.parallel_sub_partitioned_fp16_groups = []
        # same underlying data as above but viewed as: [groups] -> [rank-ids] 
        # -> [comm-ids]
        self.parallel_comm_sub_partitioned_fp16_groups = []

        # 32-bit sub-partitions of the parallel partitioned parameters
        # that this process will update
        self.local_sub_partitions_of_fp32_groups = []

        # param partition info

        # parameters in each group that will not be updated by this 
        # process directly
        self.params_not_local = []

        # parameters that will be updated by this process directly
        self.params_in_rank_sub_partitions = []

        # parameter offsets for parameters in sub-partitions. Parameter
        # boundaries may not align with sub-partition boundaries
        # so we need to keep track of the offsets
        self.params_in_rank_sub_partitions_offsets = []

        # number of elements per sub-partition in each group
        self.sub_partition_sizes = []

        # number of communication intervals for each group
        self.num_comm_intervals_per_group = []

        local_rank = dist.get_rank(group=self.dp_process_group)

        self.group_paddings = []
        self.partition_count = dist.get_world_size(group=self.dp_process_group)

        self.default_device = self.optimizer.param_groups[0]['params'][0].device

        # max elems per param group
        self.max_elems_per_comm = []

        # loop to deal with groups
        for i, param_group in enumerate(self.optimizer.param_groups):
            # push this group to list before modify
            self.fp16_groups.append(param_group['params'])

            # calculate best max elements per comm based to minimize padding
            self.max_elems_per_comm.append(
                self.best_max_elems_per_comm(
                    num_elements=sum(t.numel() for t in self.fp16_groups[i]),
                    max_elements_per_comm=max_elements_per_comm,
                    dp=dist.get_world_size(group=self.dp_process_group)))

            # flattens all tensors into single 1d tensor aligned with 
            # sub-partition size for later dividing
            # RS: create aligned sub-partitions
            # see_memory_usage('**** \n memory consumption before padding', 
            #                  force=True)
            flat_aligned_params = \
                self.flatten_dense_tensors_sub_partition_aligned(
                tensor_list=self.fp16_groups[i],
                dp=dist.get_world_size(group=self.dp_process_group),
                max_elements_per_comm=self.max_elems_per_comm[i],
                pg=self.dp_process_group)
            self.fp16_groups_flat.append(flat_aligned_params)
            # see_memory_usage('**** \n memory consumption after padding', 
            #                  force=True)
            # exit()
            
            # TODO: I don't think this does anything?
            # set model fp16 weight to slices of flattened buffer
            updated_params = self.unflatten(self.fp16_groups_flat[i],
                                            self.fp16_groups[i])

            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data

            # divide the flat weights into near equal partition equal to the 
            # data parallel degree
            # each process will compute on a different part of the partition
            # RS: split into two layer list -> [comm-id] -> 
            # [sub-partitions per rank]
            comm_partitions, dp_sub_partitions, element_intervals, \
                sub_partition_size, num_comm_intervals = \
                self.get_data_parallel_sub_partitions(
                    tensor=self.fp16_groups_flat[i],
                    max_elements_per_comm=self.max_elems_per_comm[i],
                    world_size=dist.get_world_size(
                        group=self.dp_process_group),
                    dp_process_group=self.dp_process_group
                )
            
            self.parallel_comm_sub_partitioned_fp16_groups.append(
                comm_partitions)  # comm -> rank

            self.parallel_sub_partitioned_fp16_groups.append(
                dp_sub_partitions)  # rank -> comm

            self.sub_partition_sizes.append(sub_partition_size)

            self.num_comm_intervals_per_group.append(num_comm_intervals)

            # a partition of the fp32 master weights that will be updated
            # by this process
            # RS: store/detach/cast our local sub-partitions
            local_sub_partitions = []
            for sub_partition in \
                self.parallel_sub_partitioned_fp16_groups[i][local_rank]:
                fp32_sub_partition = sub_partition.clone().float().detach()
                # fp32_sub_partition = sub_partition.clone().detach()
                fp32_sub_partition.requires_grad = True
                local_sub_partitions.append(fp32_sub_partition)

            self.local_sub_partitions_of_fp32_groups.append(local_sub_partitions)

            # Compute sub_partition paddings
            sub_partition_paddings = get_group_alignment_padding(
                tensor_list=self.fp16_groups[i],
                sub_partition_size=sub_partition_size,
                sub_partition_count=num_comm_intervals * self.partition_count)

            self.group_paddings.append(sub_partition_paddings)

            # modify optimizer of have flat master weight
            param_group['params'] = self.local_sub_partitions_of_fp32_groups[i]

            # RS: divide up the sub-partitions and keep track of offsets for 
            # each param
            params_in_rank_sub_partition, params_in_rank_sub_partitions_offsets, \
                params_not_local = \
                self.get_all_sub_partition_info(
                tensor_list=self.fp16_groups[i],
                all_element_intervals=element_intervals,
                local_rank=local_rank,
                world_size=dist.get_world_size(group=self.dp_process_group)
            )

            self.params_in_rank_sub_partitions.append(params_in_rank_sub_partition)
            
            self.params_not_local.append(params_not_local)

            self.params_in_rank_sub_partitions_offsets.append(
                params_in_rank_sub_partitions_offsets)
            
        # we may have a way of fusing dynamic scale. Do not support for now
        if dynamic_loss_scale:
            if dynamic_loss_args is None:
                self.loss_scaler = DynamicLossScaler()
            else:
                init_scale = dynamic_loss_args['INITIAL_LOSS_SCALE']
                scale_window = dynamic_loss_args['SCALE_WINDOW']
                min_loss_scale = dynamic_loss_args['MIN_LOSS_SCALE']
                self.loss_scaler = DynamicLossScaler(
                    init_scale=init_scale,
                    scale_window=scale_window,
                    min_scale=min_loss_scale
                )

            self.dynamic_loss_scale = True

        else:
            self.dynamic_loss_scale = False
            self.loss_scaler = LossScaler(scale=static_loss_scale)
            self.cur_iter = 0

        self.mpu = mpu
        self.clip_grad = clip_grad

        self.overflow = False
        self.overflow_checker = CheckOverflow(self.fp16_groups,
                                              zero_reduce_scatter=True)

    @staticmethod
    def best_max_elems_per_comm(num_elements, max_elements_per_comm, dp):
        # if we use max-elems-per-comm as is, how many comm intervals 
        # will there be
        max_comm_intervals = math.ceil(num_elements / max_elements_per_comm)
        padding_for_max_comm = (max_elements_per_comm *
                                max_comm_intervals) - num_elements

        # if we use 1 less comm interval how much extra comm padding would 
        # be required
        min_comm_intervals = num_elements // max_elements_per_comm
        if min_comm_intervals == 0:
            log_dist(f'Using default max_elements_per_comm ' \
                     f'{max_elements_per_comm}',ranks=[0])
            return max_elements_per_comm

        padding_for_min_comm = \
            math.ceil(num_elements / (dp * min_comm_intervals))

        # choose padding that uses least amount of overhead
        if padding_for_max_comm > padding_for_min_comm:
            new_max_elements_per_comm = \
                padding_for_min_comm + max_elements_per_comm
            log_dist(
                f'Updating max_elements_per_comm from '
                f'{max_elements_per_comm} -> {new_max_elements_per_comm}.', 
                ranks=[0])
            return new_max_elements_per_comm
        else:
            log_dist(f'Using default max_elements_per_comm '
                     f'{max_elements_per_comm}',
                     ranks=[0])
            return max_elements_per_comm

    @staticmethod
    def get_data_parallel_sub_partitions(tensor,
                                         max_elements_per_comm,
                                         world_size,
                                         dp_process_group=None):
        total_num_elements = tensor.numel()

        # if total elements is less than our max, revert to splitting into 
        # dp partitions
        max_elements_per_comm = min(total_num_elements, max_elements_per_comm)
        sub_partition_size = int(max_elements_per_comm // world_size)

        # Ensure partition alignment was done correctly
        num_sub_partitions = int(total_num_elements // sub_partition_size)
        assert total_num_elements % sub_partition_size == 0, \
            "{} % {} != 0".format(total_num_elements, sub_partition_size)

        # Ensure comm interval alignment was done correctly.
        num_comm_intervals = int(num_sub_partitions // world_size)
        assert num_sub_partitions % world_size == 0, \
            "{} % {} != 0".format(num_sub_partitions, world_size)

        # [comm_id] -> [rank]
        comm_partitions = []
        for _ in range(num_comm_intervals):
            comm_partitions.append([])

        start = 0
        comm_id = 0
        element_intervals = defaultdict(
            list)  # [rank] -> [(start,end), (start,end), ...]
    
        for idx in range(num_sub_partitions):
            rank_id = idx % world_size
            sub_partition = tensor.narrow(0, start, sub_partition_size).detach()
            element_intervals[rank_id].append((start, start + sub_partition_size))
            comm_partitions[comm_id].append(sub_partition)
            start = start + sub_partition_size
            if rank_id == (world_size - 1):
                comm_id += 1

        # [rank] -> [comm_id]
        sub_partitions = []
        for _ in range(world_size):
            sub_partitions.append([])
        for comm_id, partitions in enumerate(comm_partitions):
            for rank_id, partition in enumerate(partitions):
                sub_partitions[rank_id].append(partition)
        return comm_partitions, sub_partitions, element_intervals, \
            sub_partition_size, num_comm_intervals

    @staticmethod
    def get_all_sub_partition_info(tensor_list,
                                   all_element_intervals,
                                   local_rank,
                                   world_size):
        params_not_local = []

        # [rank] -> [comm-id] -> [param/offset]
        params_in_rank_sub_partition = []
        params_in_rank_sub_partitions_offsets = []

        for rank in range(world_size):
            params_in_local_sub_partition = []
            local_sub_partition_offsets = []
            comm_tensor_list = []
            comm_offset_list = []
            current_index = 0
            prev_comm_idx = 0
            for iii, tensor in enumerate(tensor_list):
                tensor_size = tensor.numel()
                results_list = _range_check(current_index,
                                            all_element_intervals[rank],
                                            tensor_size)
                for contained, offset, comm_idx in results_list:
                    if contained:
                        if prev_comm_idx != comm_idx:
                            params_in_local_sub_partition.append(comm_tensor_list)
                            comm_tensor_list = []
                            local_sub_partition_offsets.append(comm_offset_list)
                            comm_offset_list = []
                        comm_tensor_list.append(tensor)
                        comm_offset_list.append(offset)
                        prev_comm_idx = comm_idx
                    elif rank == local_rank:
                        params_not_local.append(tensor)

                current_index = current_index + tensor_size

            #assert len(comm_tensor_list) > 0
            #assert len(comm_offset_list) > 0
            params_in_local_sub_partition.append(comm_tensor_list)
            local_sub_partition_offsets.append(comm_offset_list)

            params_in_rank_sub_partition.append(params_in_local_sub_partition)
            params_in_rank_sub_partitions_offsets.append(local_sub_partition_offsets)

        return params_in_rank_sub_partition, \
            params_in_rank_sub_partitions_offsets, params_not_local
    
    def get_flat_sub_partitions(self,
                                comm_tensor_list,
                                comm_param_offsets,
                                sub_partition_size,
                                dtype,
                                default_device,
                                num_comm_intervals=None,
                                return_partition_params=False):

        partition_params = []
        final_param_offsets = []
        flat_sub_partitions = []
        for tensor_list, param_offsets in zip(comm_tensor_list, 
                                              comm_param_offsets):
            flat_tensor_list = []
            current_size = 0
            my_offsets = []
            my_params = []

            for i, tensor in enumerate(tensor_list):
                if tensor.grad is None:
                    tensor.grad = torch.zeros(tensor.size(),
                                              dtype=tensor.dtype,
                                              device=tensor.device)
                param = tensor
                tensor = tensor.grad
                num_elements = tensor.numel()
                tensor_offset = 0

                # we need to offset to get to the right element
                if i == 0 and param_offsets[i] > 0:
                    tensor_offset = param_offsets[i]
                    num_elements = num_elements - tensor_offset

                # We don't need all elements of the tensor if this tensor is
                # larger than we have space for in our curr sub-partition
                if num_elements > (sub_partition_size - current_size):
                    num_elements = sub_partition_size - current_size

                # we need a narrow view of the tensor based on the tensor offset 
                # and number of elements that we need from this tensor
                if tensor_offset > 0 or num_elements < tensor.numel():
                    flat_tensor_list.append(tensor.contiguous().view(-1).narrow(
                        0,
                        int(tensor_offset),
                        int(num_elements)).to(dtype))
                else:
                    flat_tensor_list.append(tensor.to(dtype))
                my_params.append(param)

                # remember offset into partition and #elems for this tensor
                my_offsets.append((current_size, num_elements))

                current_size = current_size + num_elements

            # this means its the last partition and does not align with the dp 
            # boundary. We need to pad before flattening
            if current_size < sub_partition_size:
                my_offsets.append((None, None))
                my_params.append(None)
                if len(tensor_list) == 0:
                    assert default_device != None
                    flat_tensor_list.append(
                        torch.zeros(int(sub_partition_size - current_size),
                                    dtype=dtype,
                                    device=default_device))
                else:
                    flat_tensor_list.append(
                        torch.zeros(int(sub_partition_size - current_size),
                                    dtype=dtype,
                                    device=tensor_list[0].device))
            partition_params.append(my_params)  #flat_tensor_list)
            final_param_offsets.append(my_offsets)
            assert len(flat_tensor_list) == \
                len(my_offsets), "{} {}".format(len(flat_tensor_list), 
                                                len(my_offsets))
            flat_sub_partitions.append(self.flatten(flat_tensor_list))

        if num_comm_intervals is not None and \
            len(flat_sub_partitions) < num_comm_intervals:
            # logger.info("padding w. sub partitions to ensure uniform 
            # communication")
            device = flat_sub_partitions[0].device
            for _ in range(num_comm_intervals - len(flat_sub_partitions)):
                flat_sub_partitions.append(
                    torch.zeros(int(sub_partition_size),
                                dtype=dtype,
                                device=device))
                partition_params.append([None])
                final_param_offsets.append([(None, None)])

        if return_partition_params:
            assert len(flat_sub_partitions) == len(partition_params)
            assert len(partition_params) == \
                len(final_param_offsets), "{} {}".format(len(partition_params), 
                                                         len(final_param_offsets))
            return flat_sub_partitions, partition_params, final_param_offsets
        return flat_sub_partitions, partition_params

    def zero_grad(self, set_grads_to_None=True):
        """
        Zero FP16 parameter grads.
        """
        # FP32 grad should never exist.
        # For speed, set model fp16 grad to None by default
        for group in self.fp16_groups:
            for p in group:
                if set_grads_to_None:
                    p.grad = None
                else:
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

    def free_grad_in_param_list(self, param_list):
        for p in param_list:
            if isinstance(p, list):
                for _p in p:
                    _p.grad = None
            else:
                p.grad = None

    def flatten_dense_tensors_sub_partition_aligned(self,
                                                    tensor_list,
                                                    dp,
                                                    max_elements_per_comm,
                                                    pg):
        assert max_elements_per_comm >= dp, \
            f"max_elements_per_comm {max_elements_per_comm} < dp {dp}"

        num_elements = sum(t.numel() for t in tensor_list)
        log_dist(
            "Total number of elements in model: {}, \
                max elements per com: {}".format(
                num_elements,
                max_elements_per_comm),
            ranks=[0])

        # Compute aligned partition size based on parameter count
        aligned_param_partition_size = math.ceil(num_elements / dp)

        # Compute aligned partition size based on communication size
        aligned_comm_partition_size = int(max_elements_per_comm // dp)

        if aligned_param_partition_size <= aligned_comm_partition_size:
            sub_partition_count = 1
            sub_partition_size = aligned_param_partition_size
        else:
            sub_partition_count = math.ceil(aligned_param_partition_size /
                                            aligned_comm_partition_size)
            sub_partition_size = aligned_comm_partition_size

        # Compute required padding  for alignment to dp and max_elements_per_comm
        padding = (sub_partition_count * sub_partition_size * dp) - num_elements

        log_dist(
            f"sub_partition_count: {sub_partition_count}, \
                sub_partition_size: {sub_partition_size}, padding: {padding}",
            ranks=[0])
        log_dist(
            f"number of elements with padding: {num_elements} \
                  + {padding} = {num_elements + padding}",
            ranks=[0])

        if padding == 0:
            aligned_tensor_list = tensor_list
        else:
            pad_tensor = torch.zeros(padding,
                                     device=tensor_list[0].device,
                                     dtype=tensor_list[0].dtype)
            aligned_tensor_list = tensor_list + [pad_tensor]
        flat_tensors = self.flatten(aligned_tensor_list)
        return flat_tensors
    
    def reduce_gradients(self, pipeline_parallel=False):

        postscale_gradients = self.postscale_gradients
        gradient_predivide_factor = self.gradient_predivide_factor
        gradient_average = self.gradient_average

        world_size = dist.get_world_size(group=self.dp_process_group)
        local_rank = dist.get_rank(group=self.dp_process_group)


        for i, group in enumerate(self.fp16_groups):
            num_comm_intervals = self.num_comm_intervals_per_group[i]
            all_sub_partitions = []

            for rank in range(world_size):
                # gsp is list of partitions indexed by comm_idx
                grad_sub_partitions, partition_params = \
                    self.get_flat_sub_partitions(
                    comm_tensor_list=self.params_in_rank_sub_partitions[i][rank],
                    comm_param_offsets=self.params_in_rank_sub_partitions_offsets[i]
                    [rank],
                    dtype=self.communication_data_type,
                    default_device=self.default_device,
                    sub_partition_size=self.sub_partition_sizes[i],
                    num_comm_intervals=self.num_comm_intervals_per_group[i])
                all_sub_partitions.append(grad_sub_partitions)

                assert len(grad_sub_partitions) == num_comm_intervals

            local_comm_partitions = []
            for comm_idx in range(num_comm_intervals):
                single_comm_all_partitions = []
                for rank in range(world_size):
                    single_comm_all_partitions.append(
                        all_sub_partitions[rank][comm_idx])
                if postscale_gradients:
                    if gradient_predivide_factor != 1.0:
                        for partition in single_comm_all_partitions:
                            partition.mul_(1. / gradient_predivide_factor)

                    dist.reduce_scatter(output=single_comm_all_partitions[local_rank],
                                        input_list=single_comm_all_partitions,
                                        group=self.dp_process_group)
                    if gradient_average:
                        # Only need to average our local grads in post scaling
                        if gradient_predivide_factor != world_size:
                            single_comm_all_partitions[local_rank].mul_(
                                gradient_predivide_factor / world_size)
                else:
                    for partition in single_comm_all_partitions:
                        partition.div_(world_size)

                    dist.reduce_scatter(output=single_comm_all_partitions[local_rank],
                                        input_list=single_comm_all_partitions,
                                        group=self.dp_process_group)
                
    def step(self, closure=None):
        # First compute norm for all group so we know if there is overflow
        self.overflow = self.overflow_checker.check()

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            self.zero_grad()
            if self.verbose:
                logger.info(
                    f"[deepspeed] fp16 dynamic loss scale overflow! Skipping step. "
                    f"Attempted loss scale: {prev_scale}, reducing to {self.loss_scale}")
            return self.overflow

        norm_groups = []
        local_sub_partitions_grad_groups = []
        partition_id = dist.get_rank(group=self.dp_process_group)

        for i, group in enumerate(self.fp16_groups):

            #RS: update free grads w.r.t. sub partitions
            #free gradients for all the parameters that are not updated by this process
            self.free_grad_in_param_list(self.params_not_local[i])
            # create flat gradient partitions for parameters updated by this process
            local_grad_sub_partitions, partition_params = self.get_flat_sub_partitions(
                comm_tensor_list=self.params_in_rank_sub_partitions[i][partition_id],
                comm_param_offsets=self.params_in_rank_sub_partitions_offsets[i]
                [partition_id],
                sub_partition_size=self.sub_partition_sizes[i],
                dtype=self.local_sub_partitions_of_fp32_groups[i][0].dtype,
                num_comm_intervals=self.num_comm_intervals_per_group[i],
                default_device=self.default_device)

            norm_groups.append(get_grad_norm(local_grad_sub_partitions, partition_params, 
                                             mpu=self.mpu))

            # RS: update all our local params with sub-partition grads
            for idx, sub_partition_param in \
                enumerate(self.local_sub_partitions_of_fp32_groups[i]):
                sub_partition_param.grad = local_grad_sub_partitions[idx]

            # RS: update free grads for sub-partitions
            # release all the gradient since we have already created a necessary copy in 
            # dp_grad_partition
            self.free_grad_in_param_list(
                self.params_in_rank_sub_partitions[i][partition_id])

            local_sub_partitions_grad_groups.append(local_grad_sub_partitions)

        # RS: update unscale/clip with sub partitions

        self.unscale_and_clip_grads(local_sub_partitions_grad_groups, norm_groups)

        self.optimizer.step()

        # RS: clear our sub partition grads
        # get rid of the fp32 gradients. Not needed anymore
        for group in self.local_sub_partitions_of_fp32_groups:
            for idx, sub_partition_param in enumerate(group):
                sub_partition_param.grad = None
            # group.grad = None

        # RS: copy all sub-partition fp32 data to fp16 sub partitions
        # copy fp32 param data to fp16 partitions w.r.t. our local rank
        for fp16_all_sub_partitions, fp32_local_sub_partitions in \
            zip(self.parallel_sub_partitioned_fp16_groups, 
                self.local_sub_partitions_of_fp32_groups):
            for local_sub_partition_param_fp16, local_sub_partition_param_fp32 in \
                zip(fp16_all_sub_partitions[partition_id], fp32_local_sub_partitions):
                local_sub_partition_param_fp16.data.copy_(
                    local_sub_partition_param_fp32.data)

        #RS: all_gather/broadcast sub-partitions in separate comm calls
        #gather the updated weights from everyone
        for fp16_all_sub_partitions in self.parallel_comm_sub_partitioned_fp16_groups:
            for comm_id, sub_partitions in enumerate(fp16_all_sub_partitions):
                dist.all_gather(sub_partitions,
                                sub_partitions[partition_id],
                                group=self.dp_process_group)

        # TODO: we probably don't need this? just to be safe
        for i in range(len(norm_groups)):
            updated_params = self.unflatten(self.fp16_groups_flat[i],
                                            self.fp16_groups[i])
            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data

        return self.overflow

    def unscale_and_clip_grads(self, grad_groups_flat, norm_groups):
        total_norm = 0.0
        for norm in norm_groups:
            total_norm += norm**2.0
        total_norm = math.sqrt(total_norm)

        # compute combined scale factor for this group
        combined_scale = self.loss_scale
        if self.clip_grad > 0.:
            # norm is in fact norm*scale
            clip = ((total_norm / self.loss_scale) + 1e-6) / self.clip_grad
            if clip > 1:
                combined_scale = clip * self.loss_scale

        for grad in grad_groups_flat:
            if isinstance(grad, list):
                sub_partitions = grad
                for g in sub_partitions:
                    g.data.mul_(1. / combined_scale)
            else:
                grad.data.mul_(1. / combined_scale)

    def backward(self, loss, retain_graph=False):
        self.loss_scaler.backward(loss.float(), retain_graph=retain_graph)

    def _update_scale(self, has_overflow=False):
        self.loss_scaler.update_scale(has_overflow)

    # Promote state so it can be retrieved or set via 
    # "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via 
    # "fp16_optimizer_instance.param_groups" 
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    # Promote loss scale so it can be retrieved or set via 
    # "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        return self.loss_scaler.loss_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    def _rigid_state_dict(self):
        """
            Returns a dict that can be loaded for continued training 
            with same DP degree
        """
        """
        Returns a dict containing the current state of this :class:
        `FP16_Optimizer` instance. This dict contains attributes of 
        :class:`FP16_Optimizer`, as well as the state_dict of the 
        contained Pytorch optimizer.
        Example::
            checkpoint = {}
            checkpoint['model'] = model.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            torch.save(checkpoint, "saved.pth")
        """
        state_dict = {}
        state_dict['loss_scaler'] = self.loss_scaler
        state_dict['dynamic_loss_scale'] = self.dynamic_loss_scale
        state_dict['overflow'] = self.overflow
        state_dict['base_optimizer_state'] = self.optimizer.state_dict()
        state_dict[
            'local_sub_partitions_of_fp32_groups'] = \
                self.local_sub_partitions_of_fp32_groups
        return state_dict


    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:
        `FP16_Optimizer` instance. This dict contains attributes of 
        :class:`FP16_Optimizer`, as well as the state_dict of the 
        contained Pytorch optimizer.
        """

        return self._rigid_state_dict()

    def _rigid_load_state_dict(self, state_dict, load_optimizer_states=True):

        self.loss_scaler = state_dict['loss_scaler']
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.overflow = state_dict['overflow']
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict['base_optimizer_state'])

        for curr_group, saved_group in \
            zip(self.local_sub_partitions_of_fp32_groups, 
                state_dict['local_sub_partitions_of_fp32_groups']):
            for curr_param, saved_param in zip(curr_group, saved_group):
                curr_param.data.copy_(saved_param.data)

    def load_state_dict(self,
                        state_dict,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some 
        ``init_optimizer``, whose parameters in turn came from ``model``, 
        it is expected that the user will call ``model.load_state_dict()`` 
        before ``fp16_optimizer_instance.load_state_dict()`` is called.
        Example::
            model = torch.nn.Linear(D_in, D_out).cuda().half()
            optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
            optimizer = FP16_Optimizer(optimizer, static_loss_scale = 128.0)
            ...
            checkpoint = torch.load("saved.pth")
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        """
        self._rigid_load_state_dict(
            state_dict,
            load_optimizer_states)
