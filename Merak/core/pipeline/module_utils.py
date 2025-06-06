# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# TXacs (txacs1993@gmail.com)
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

# Parts of the code here are adapted from https://github.com/NVIDIA/Megatron-LM/blob/806422e5ec35c27b027dbb413b05e27b6590dc56/megatron/model/utils.py
# Parts of the code here are adapted from https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/utils.py


import torch
import torch.distributed as dist
import torch.nn as nn
from math import ceil, floor
from bisect import bisect_left
from typing import List, Union

from .. import mpu
from ..printer import logger

def set_random_seed(seed: int):
    """Set the random seed for common PRNGs used during training: random,
    numpy, and torch.

    Args:
        seed (int): the seed to use
    """
    import numpy
    import random
    from ..recompute.checkpointing import model_parallel_cuda_manual_seed

    if seed is not None and seed > 0:
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        # if torch.cuda.device_count() > 0:
        #     model_parallel_cuda_manual_seed(seed)
        # dsp在tp或sp时，dropout算子也需考虑同一stage的不同rank设备设置不同的种子
        model_parallel_cuda_manual_seed(seed) 
    else:
        raise ValueError('Seed ({}) should be a positive integer.'.format(seed))

def prefix_sum_inc(weights: List[int]) -> List[int]:
    """ Compute an inclusive prefix sum.

    Example:
        >>> prefix_sum_inc([3,4,5])
        [3, 7, 12]
    """
    weights_ = [w for w in weights]
    for x in range(1, len(weights_)):
        weights_[x] += weights_[x - 1]
    return weights_

def partition_uniform(num_items: int, num_parts: int, use_ceil: bool = True) -> List[int]:
    parts = [0] * (num_parts + 1)
    # First check for the trivial edge case
    if num_items <= num_parts:
        for p in range(num_parts + 1):
            parts[p] = min(p, num_items)
        return parts

    if use_ceil:
        chunksize = ceil(num_items / num_parts)
    else:
        chunksize = floor(num_items / num_parts)
    for p in range(num_parts):
        parts[p] = min(chunksize * p, num_items)

    remainder = num_items % num_parts
    for p in range(num_parts):
        for i in range(remainder):
            if p > 1 + i:
                parts[p] += 1

    parts[num_parts] = num_items
    return parts


def _lprobe(weights: List[int], num_parts: int, bottleneck: int) -> Union[List[int], bool]:
    num_items = len(weights)
    total_weight = weights[-1]

    # initialize partitioning
    parts = [0] * (num_parts + 1)
    for p in range(1, num_parts + 1):
        parts[p] = num_items

    bsum = bottleneck  # running sum of target weight for pth partition
    chunksize = num_items // num_parts
    step = chunksize
    for p in range(1, num_parts):
        # Jump to the next bucket
        while (step < num_items) and (weights[step] < bsum):
            step += chunksize

        # Find the end index of partition p
        parts[p] = bisect_left(weights,
                               bsum,
                               lo=step - chunksize,
                               hi=min(step,
                                      num_items))
        # Nothing more to partition, return early
        if parts[p] == num_items:
            # See if the current partition is overweight.
            part_size = weights[-1] - weights[parts[p - 1]]
            return parts, part_size < bottleneck

        # Next partition target
        bsum = weights[parts[p] - 1] + bottleneck

    return parts, bsum >= total_weight


def _rb_partition_balanced(weights: List[int], num_parts: int, eps: float) -> int:
    total_weight = weights[-1]
    lower = total_weight / num_parts  # best case heaviest partition
    upper = total_weight  # worst case heaviest partition

    # Do a binary search for the best partitioning
    while upper > lower + eps:
        mid = lower + ((upper - lower) / 2)
        parts, success = _lprobe(weights, num_parts, mid)
        if success:
            upper = mid
        else:
            lower = mid + eps
    return upper


def partition_balanced(weights: List[int], num_parts: int, eps: float = 1e-3) -> List[int]:
    num_items = len(weights)
    # First check for the trivial edge case
    if num_items <= num_parts:
        return partition_uniform(num_items, num_parts)

    weights_ = prefix_sum_inc(weights)

    # Find the smallest bottleneck (weight of heaviest partition)
    bottleneck = _rb_partition_balanced(weights_, num_parts, eps=eps)

    # Now compute that partitioning
    parts, success = _lprobe(weights_, num_parts, bottleneck)
    assert success

    return parts

def print_trainable_parameters(model: nn.Module):
    """
    Prints the number of trainable parameters in the model.
    """
    dist.barrier()
    trainable_params = 0
    all_param = 0
    if mpu.get_data_parallel_rank() == 0 and mpu.get_model_parallel_rank() == 0:
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        logger.info(
            f"stage: {mpu.get_pipe_parallel_rank()} || trainable params: {trainable_params} \
                || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
        )
