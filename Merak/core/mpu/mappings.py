# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

# https://github.com/NVIDIA/Megatron-LM/blob/806422e5ec35c27b027dbb413b05e27b6590dc56/megatron/mpu/mappings.py

import torch
import torch.distributed as dist

from Merak import get_grid
from .initialize import (
    get_model_parallel_group,
    get_model_parallel_world_size,
    get_model_parallel_rank
)
from .utils import split_tensor_along_last_dim, split_tensor_along_channel_dim


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU.
    if get_model_parallel_world_size()==1:
        return input_

    # All-reduce.
    if not input_.is_contiguous():
        input_ = input_.contiguous()
    torch.distributed.all_reduce(input_, group=get_model_parallel_group())

    return input_


def _split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_model_parallel_rank()
    output = input_list[rank].contiguous()

    return output

def _channels_split(input_):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list, size_list = split_tensor_along_channel_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = get_model_parallel_rank()
    output = input_list[rank].contiguous()
    del input_list

    return output, size_list

def _gather(input_):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_,
                                 group=get_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output

def _conv_gather(input_, size_list):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Size and dimension.
    device = input_.device
    channel_dim = 1
    rank = get_model_parallel_rank()

    # tensor_list = [torch.empty_like(input_size) for _ in range(world_size)]
    tensor_list = [torch.empty(size_list[i]).to(device) for i in range(world_size)]

    if dist.get_backend() == 'nccl':
        tensor_list[rank] = input_
        torch.distributed.all_gather(tensor_list, input_,
                                    group=get_model_parallel_group())

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=channel_dim).contiguous()
    else:
        global_rank = get_grid().global_rank
        # send all input_ to model rank 0, finnaly broadcast output
        if rank == 0:
            tensor_list[rank] = input_
            for i in range(world_size-1):
                dist.recv(tensor=tensor_list[i+1], src=global_rank+i+1)
            output = torch.cat(tensor_list, dim=channel_dim).contiguous()
            dist.broadcast(output, src=global_rank, group=get_model_parallel_group())
        else:
            dist.send(tensor=input_, dst=global_rank - rank)
            output = torch.cat(tensor_list, dim=channel_dim).contiguous()
            dist.broadcast(output, src=global_rank-rank, group=get_model_parallel_group())

    return output

def _reduce_scatter(input_):
    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size==1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)
    input_list = list(chunk.contiguous() for chunk in input_list)
    # Note: torch.split does not create contiguous tensors by default.
    rank = get_model_parallel_rank()
    output = input_list[rank]#torch.empty_like(input_list[rank]).contiguous()

    torch.distributed.reduce_scatter(output, input_list,
                                     group=get_model_parallel_group())
    
    return output



SEQUENCE_DIM=1
def set_sequence_dim(dim):
    global SEQUENCE_DIM
    SEQUENCE_DIM = dim

def get_sequence_dim():
    global SEQUENCE_DIM
    return SEQUENCE_DIM


    
def _split_along_first_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along first dimension.
    dim = get_sequence_dim()
    dim_size = input_.size()[dim]
    assert dim_size % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"
    local_dim_size = dim_size // world_size
    rank = get_model_parallel_rank()
    dim_offset = rank * local_dim_size
    output = input_.chunk(world_size, dim=dim)[rank].contiguous()
    # output = input_[dim_offset:dim_offset+local_dim_size].contiguous()

    return output

def _gather_along_first_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    dim = get_sequence_dim()
    dim_size = list(input_.size())
    dim_size[dim] = dim_size[dim] * world_size

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=device)
    torch.distributed._all_gather_base(output, input_.contiguous(),
                                       group=get_model_parallel_group())

    return output

def _reduce_scatter_along_first_dim(input_):
    """Reduce-scatter the input tensor across model parallel group."""
    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_
    dim = get_sequence_dim()
    dim_size = list(input_.size())
    assert dim_size[dim] % world_size == 0, \
        "First dimension of the tensor should be divisible by tensor parallel size"

    dim_size[dim] = dim_size[dim] // world_size

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
    else:
        device = 'cpu'
    output = torch.empty(dim_size, dtype=input_.dtype,
                         device=device)
    torch.distributed._reduce_scatter_base(output, input_.contiguous(), 
                                           group=get_model_parallel_group())
    return output




ASYNC_OP = []

def _async_reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU.
    if get_model_parallel_world_size()==1:
        return input_

    # All-reduce.
    op = torch.distributed.all_reduce(input_,
                                      group=get_model_parallel_group(),
                                      async_op=True)

    return (op, input_)


class _AsyncCopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        op, grad_output = _async_reduce(grad_output)
        global ASYNC_OP
        ASYNC_OP.append(op)
        return grad_output


class _AsyncReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _async_reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        op, input_ = _async_reduce(input_)
        global ASYNC_OP
        ASYNC_OP.append(op)
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        # print('********', *grad_output)
        return grad_output


class _CopyToModelParallelRegion(torch.autograd.Function):
    """Pass the input to the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return input_
    
    @staticmethod
    def forward(ctx, input_):
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _reduce(grad_output)


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-redcue the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)

class _ScatterToConvParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        output, _ = _channels_split(input_)
        return output

    @staticmethod
    def forward(ctx, input_):
        output, channel_list = _channels_split(input_)
        ctx.channel_list = channel_list
        return input_

    @staticmethod
    def backward(ctx, grad_output):
        return _conv_gather(grad_output, ctx.channel_list)

class _ReduceScatterFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split(grad_output)


class _ScatterToSequenceParallelRegion(torch.autograd.Function):
    """Split the input and keep only the corresponding chuck to the rank."""

    @staticmethod
    def symbolic(graph, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _split_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)

class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatinate.""" 

    @staticmethod
    def symbolic(graph, input_, tensor_parallel_output_grad=True):
        return _gather_along_first_dim(input_)
    
    @staticmethod
    def forward(ctx, input_, tensor_parallel_output_grad=True):
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        return _gather_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        # If the computation graph after the gather operation is
        # in the tensor parallel mode, output gradients need to reduce 
        # scattered and whereas if the computation is duplicated, 
        # output gradients need to be scattered.
        if tensor_parallel_output_grad:
            return _reduce_scatter_along_first_dim(grad_output), None
        else:
            return _split_along_first_dim(grad_output), None


class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """Reduce scatter the input from the model parallel region."""

    @staticmethod
    def symbolic(graph, input_):
        return _reduce_scatter_along_first_dim(input_)
    
    @staticmethod
    def forward(ctx, input_):
        return _reduce_scatter_along_first_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather_along_first_dim(grad_output)

# -----------------
# Helper functions.
# -----------------

def copy_to_model_parallel_region(input_):
    return _CopyToModelParallelRegion.apply(input_)


def async_copy_to_model_parallel_region(input_):
    return _AsyncCopyToModelParallelRegion.apply(input_)


def async_reduce_from_model_parallel_region(input_):
    return _AsyncReduceFromModelParallelRegion.apply(input_)


def reduce_from_model_parallel_region(input_):
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_model_parallel_region(input_):
    return _ScatterToModelParallelRegion.apply(input_)

def scatter_to_conv_parallel_region(input_):
    return _ScatterToConvParallelRegion.apply(input_)

def gather_from_model_parallel_region(input_):
    return _GatherFromModelParallelRegion.apply(input_)


def reduce_scatter_to_model_parallel_region(input_):
    return _ReduceScatterFromModelParallelRegion.apply(input_)


def scatter_to_sequence_parallel_region(input_):
    return _ScatterToSequenceParallelRegion.apply(input_)

def gather_from_sequence_parallel_region(input_,
                                         tensor_parallel_output_grad=True):
    return _GatherFromSequenceParallelRegion.apply(input_,
                                                   tensor_parallel_output_grad)


def reduce_scatter_to_sequence_parallel_region(input_):
    return _ReduceScatterToSequenceParallelRegion.apply(input_)