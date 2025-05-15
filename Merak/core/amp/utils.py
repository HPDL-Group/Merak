# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com)
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

import math
import torch
import torch.distributed as dist
from typing import Union, List, Optional, Dict, Any
from types import ModuleType
import logging

from .. import mpu
from ..printer import logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_global_norm(norm_list: List[float]) -> float:
    """
    Compute the total norm from a list of norms.

    Args:
        norm_list (List[float]): A list of norms.

    Returns:
        float: The total norm calculated as the square root of the sum of squares.
    """
    total_norm = 0.0
    for norm in norm_list:
        total_norm += norm ** 2.0
    return math.sqrt(total_norm)

def get_weight_norm(
        parameters: Union[torch.Tensor, List[torch.Tensor]],
        norm_type: int = 2,
    ) -> torch.Tensor:
    """
    Calculate the norm of parameters for a given model.

    Args:
        parameters (Union[torch.Tensor, List[torch.Tensor]]): The parameters to compute norms for.
        norm_type (int, optional): The type of vector norm. Defaults to 2.

    Returns:
        torch.Tensor: The computed norm of the parameters.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    device = parameters[0].device
    norm_type = float(norm_type)
    if norm_type == float('inf'):
        total_norm = max(p.data.abs().max() for p in parameters)
        total_norm_cuda = torch.FloatTensor([float(total_norm)]).to(device)
        # Take max across all GPUs.
        torch.distributed.all_reduce(total_norm_cuda,
                                        op=torch.distributed.ReduceOp.MAX,
                                        group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.
        tensor_mp_rank = mpu.get_model_parallel_rank()
        for p in parameters:
            # Filter to avoid over-counting replicated tensors from tensor
            # model parallelism
            if tensor_mp_rank > 0:
                continue

            param_norm = p.data.float().norm(norm_type)
            total_norm += param_norm**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.FloatTensor([float(total_norm)]).to(device)
        torch.distributed.all_reduce(total_norm_cuda,
                                        op=torch.distributed.ReduceOp.SUM,
                                        group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    if total_norm == float(
            'inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm

class CheckOverflow(object):
    '''Checks for overflow in gradient across parallel process'''
    def __init__(
            self,
            param_groups=None,
            zero_reduce_scatter=False,
        ):
        self.params = [] if param_groups else None
        self.zero_reduce_scatter = zero_reduce_scatter
        self.has_moe_params = False
        self.device = 'cuda'
        if param_groups:
            self.device = param_groups[0][0].device
            for group in param_groups:
                for param in group:
                    self.params.append(param)

    def check_using_norm(self, norm_group, reduce_overflow=True):
        # TODO: I don't think reduce_overflow is needed if mpu is None
        overflow = -1 in norm_group
        overflow_gpu = torch.FloatTensor([overflow]).to(self.device)
        if mpu.get_model_parallel_world_size() > 1:
            torch.distributed.all_reduce(
                overflow_gpu,
                op=torch.distributed.ReduceOp.MAX,
                group=mpu.get_model_parallel_group()
            )
        elif reduce_overflow:
            dist.all_reduce(overflow_gpu, op=torch.distributed.ReduceOp.MAX)
            dist.barrier()
        overflow = overflow_gpu[0].item()
        return bool(overflow)

    def check(self, param_groups=None):
        params = []
        has_moe_params = False
        if param_groups is None:
            params = self.params
            has_moe_params = self.has_moe_params
        else:
            assert param_groups is not None, \
                "self.params and param_groups both cannot be none"

            for group in param_groups:
                for param in group:
                    params.append(param)

        return self.has_overflow(params, has_moe_params=has_moe_params)

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params):
        for i, p in enumerate(params):
            if p.grad is not None and self._has_inf_or_nan(p.grad.data, i):
                return True
        return False

    def has_overflow(self, params, has_moe_params=None):
        if has_moe_params is None:
            has_moe_params = self.has_moe_params
        overflow = self.has_overflow_serial(params)
        # Since each model parallel GPU carries only part of the model,
        # make sure overflow flag is synced across all the model parallel GPUs
        overflow_gpu = torch.ByteTensor([overflow]).to(self.device)
        torch.distributed.all_reduce(
            overflow_gpu,
            op=torch.distributed.ReduceOp.MAX,
            group=mpu.get_model_parallel_group()
        )

        overflow = overflow_gpu[0].item()
        return bool(overflow)

    # `x` is a torch.Tensor
    @staticmethod
    def _has_inf_or_nan(x, i):
        try:
            # if x is half, the .float() incurs an additional deep copy, but
            # it's necessary if Pytorch's .sum() creates a one-element tensor
            # of the same type as x (which is true for some recent version of
            # pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a
            # Python scalar cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum == float('inf') or \
               cpu_sum == -float('inf') or \
               cpu_sum != cpu_sum:
                return True
            return False