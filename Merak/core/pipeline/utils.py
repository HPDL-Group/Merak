# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com)
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

# Parts of the code here are adapted from https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/utils.py

import os
import psutil
import gc
from math import sqrt
from typing import List, Tuple, Union, Optional
from types import ModuleType

import torch
try:
    from torch._six import inf
except ImportError:
    from torch import inf
import torch.distributed as dist
from torch.autograd.variable import Variable

def noop_decorator(func):
    return func

def clip_grad_norm_(
        parameters: Union[torch.Tensor, List[torch.Tensor]],
        max_norm: int,
        norm_type: int = 2,
        mpu: Optional[ModuleType] = None
    ) -> torch.Tensor:
    """Clips gradient norm of an iterable of parameters.

    This has been adapted from Nvidia megatron. We add norm averaging
    to consider MoE params when calculating norm as they will result
    in different norms across different ranks.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
        total_norm_cuda = torch.FloatTensor([float(total_norm)]).cuda()
        # Take max across all GPUs.
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.MAX,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0
        for p in parameters:
            if mpu is not None:
                if (mpu.get_model_parallel_rank()
                        == 0):
                    param_norm = p.grad.data.norm(norm_type)
                    total_norm += param_norm.item()**norm_type
            else:
                param_norm = p.grad.data.float().norm(norm_type)
                total_norm += param_norm.item()**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.FloatTensor([float(total_norm)]).cuda()
        if mpu is not None:
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=mpu.get_model_parallel_group())
        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    # Need to average total_norm across different GPUs due to the presence of
    # moe params
    pg = mpu.get_data_parallel_group()
    scaled_norm = total_norm * 1.0 / float(dist.get_world_size(group=pg))

    scaled_norm_tensor = torch.FloatTensor([float(scaled_norm)]).cuda()
    dist.all_reduce(scaled_norm_tensor, group=pg)
    total_norm = scaled_norm_tensor.item()

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm
    
def custom_backward(
        output: Union[List[torch.Tensor], torch.Tensor],
        grad_output: Union[List[torch.Tensor], torch.Tensor]
    ):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''
    if isinstance(output, list):
        for idx in range(len(output)):
            assert output[idx].numel() == 1, \
                "output should be pseudo-'freed' in schedule, \
                 to optimize memory"
            assert isinstance(output[idx], torch.Tensor), \
                "output == '%s'." % type(output[idx]).__name__
            assert isinstance(grad_output[idx], (torch.Tensor, type(None))), \
                "grad_output == '%s'." % type(grad_output[idx]).__name__

            # Handle scalar output
            if grad_output[idx] is None:
                assert output[idx].numel() == 1, \
                    "implicit grad requires scalar output."
                grad_output[idx] = torch.ones_like(
                    output[idx],
                    memory_format = torch.preserve_format,
                )
        # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
        Variable._execution_engine.run_backward(
            tensors = tuple(output),
            grad_tensors = tuple(grad_output),
            keep_graph = False,
            create_graph = False,
            inputs = tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )
    else:
        assert output.numel() == 1, \
            "output should be pseudo-'freed' in schedule, to optimize memory"
        assert isinstance(output, torch.Tensor), \
            "output == '%s'." % type(output[0]).__name__
        assert isinstance(grad_output, (torch.Tensor, type(None))), \
            "grad_output == '%s'." % type(grad_output).__name__
        # Handle scalar output
        if grad_output is None:
            assert output.numel() == 1, "implicit grad requires scalar output."
            grad_output = torch.ones_like(
                output,
                memory_format = torch.preserve_format,
            )

        # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
        Variable._execution_engine.run_backward(
            tensors = (output, ),
            grad_tensors = (grad_output, ),
            keep_graph = False,
            create_graph = False,
            inputs = tuple(),
            allow_unreachable=True,
            accumulate_grad=True,
        )


def deallocate_output_tensor(out: torch.Tensor):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data'
    field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if out is None:
        return
    assert isinstance(out, torch.Tensor), \
        "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, \
        "counter-productive to free a view of another tensor."
    out.data = torch.empty(
        (1,),
        device = out.device,
        dtype = out.dtype,
    )