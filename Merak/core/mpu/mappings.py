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

from .initialize import (
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_world_size,
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
)
from .utils import split_tensor_along_channel_dim, split_tensor_along_last_dim


def _reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU.
    if get_model_parallel_world_size() == 1:
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
    if world_size == 1:
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
    if world_size == 1:
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
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_model_parallel_group())

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    return output


def _conv_gather(input_, size_list):
    """Gather tensors and concatinate along the last dimension."""

    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Size and dimension.
    device = input_.device
    channel_dim = 1
    rank = get_model_parallel_rank()

    # tensor_list = [torch.empty_like(input_size) for _ in range(world_size)]
    tensor_list = [torch.empty(size_list[i]).to(device) for i in range(world_size)]

    if dist.get_backend() == "nccl":
        tensor_list[rank] = input_
        torch.distributed.all_gather(
            tensor_list, input_, group=get_model_parallel_group()
        )

        # Note: torch.cat already creates a contiguous tensor.
        output = torch.cat(tensor_list, dim=channel_dim).contiguous()
    else:
        global_rank = dist.get_rank()
        # send all input_ to model rank 0, finnaly broadcast output
        if rank == 0:
            tensor_list[rank] = input_
            for i in range(world_size - 1):
                dist.recv(tensor=tensor_list[i + 1], src=global_rank + i + 1)
            output = torch.cat(tensor_list, dim=channel_dim).contiguous()
            dist.broadcast(output, src=global_rank, group=get_model_parallel_group())
        else:
            dist.send(tensor=input_, dst=global_rank - rank)
            output = torch.cat(tensor_list, dim=channel_dim).contiguous()
            dist.broadcast(
                output, src=global_rank - rank, group=get_model_parallel_group()
            )

    return output


def _reduce_scatter(input_):
    world_size = get_model_parallel_world_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)
    input_list = list(chunk.contiguous() for chunk in input_list)
    # Note: torch.split does not create contiguous tensors by default.
    rank = get_model_parallel_rank()
    output = input_list[rank]  # torch.empty_like(input_list[rank]).contiguous()

    torch.distributed.reduce_scatter(
        output, input_list, group=get_model_parallel_group()
    )

    return output


ASYNC_OP = []


def _async_reduce(input_):
    """All-reduce the the input tensor across model parallel group."""
    # Bypass the function if we are using only 1 GPU.
    if get_model_parallel_world_size() == 1:
        return input_

    # All-reduce.
    op = torch.distributed.all_reduce(
        input_, group=get_model_parallel_group(), async_op=True
    )

    return (op, input_)


def _all_to_all(input_, scatter_dim, gather_dim):
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return input_

    input_list = [
        t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]

    dist.all_to_all(output_list, input_list, group=get_sequence_parallel_group())
    return torch.cat(output_list, dim=gather_dim).contiguous()


def _all_to_all_single(input_, scatter_dim, gather_dim):
    seq_world_size = get_sequence_parallel_world_size()
    if seq_world_size == 1:
        return input_

    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // seq_world_size
    if scatter_dim < 2:
        input_t = input_.reshape(
            [seq_world_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :]
        ).contiguous()
    else:
        input_t = (
            input_.reshape(
                [-1, seq_world_size, inp_shape[scatter_dim]]
                + inp_shape[scatter_dim + 1 :]
            )
            .transpose(0, 1)
            .contiguous()
        )

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=get_sequence_parallel_group())

    if scatter_dim < 2:
        output = output.transpose(0, 1).contiguous()
    return output.reshape(
        inp_shape[:gather_dim]
        + [
            inp_shape[gather_dim] * seq_world_size,
        ]
        + inp_shape[gather_dim + 1 :]
    ).contiguous()


def _all_to_all_comm(t, scatter_dim, gather_dim):
    bs = t.shape[0]
    # using all_to_all_single when batch size is 1
    if bs == 1:
        return _all_to_all_single(
            t,
            scatter_dim,
            gather_dim,
        )
    return _all_to_all(
        t,
        scatter_dim,
        gather_dim,
    )


def _spilt_for_sequence_parallel(input_, dim=-1):
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return input_

    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    rank = get_sequence_parallel_rank()
    output = tensor_list[rank].clone().contiguous()

    return output


def _gather_for_sequence_parallel(input_, dim=-1):
    world_size = get_sequence_parallel_world_size()
    if world_size == 1:
        return input_

    input_ = input_.contiguous()
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]

    dist.all_gather(tensor_list, input_, group=get_sequence_parallel_group())

    output = torch.cat(tensor_list, dim=dim).contiguous()

    return output


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


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, scatter_dim, gather_dim):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim

        return _all_to_all_comm(input_, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, grad_output):
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _all_to_all_comm(grad_output, scatter_dim, gather_dim)

        return (return_grad, None, None)


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split the input and keep only the corresponding chuck to the rank.

    Args:
        input_ (`torch.Tensor`): input matrix.
        dim (int): the dimension to perform split and gather
        process_group (`torch.distributed.ProcessGroup`): the process group used for collective communication

    """

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        return _spilt_for_sequence_parallel(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_scale = 1.0 / get_sequence_parallel_world_size()
        # grad_output = grad_output * grad_scale  ### colossal-ai???

        return (
            _gather_for_sequence_parallel(grad_output, ctx.dim),
            None,
        )


class _GatherForwardSplitBackward(torch.autograd.Function):
    """Gather the input from sequence parallel region and concatenate.

    Args:
        input_: input matrix.
        dim: dimension
    """

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        return _gather_for_sequence_parallel(input_, dim)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_scale = 1.0 / get_sequence_parallel_world_size()
        # grad_output = grad_output * grad_scale

        return (_spilt_for_sequence_parallel(grad_output, ctx.dim), None)


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


SEQUENCE_DIM = 1


def set_sequence_dim(dim):
    global SEQUENCE_DIM
    SEQUENCE_DIM = dim


def all_to_all_sequence_parallel_region(
    input_, scatter_dim=2, gather_dim=1, batch_first=True
):
    global SEQUENCE_DIM
    assert input_.dim() == 3 and SEQUENCE_DIM in (0, 1)

    if SEQUENCE_DIM == 0:
        input_ = input_.transpose(0, 1)  # input_.shape: (B, S, H)
    return _AllToAll.apply(input_, scatter_dim, gather_dim)


def split_for_sequence_parallel_region(input_):
    global SEQUENCE_DIM
    return _SplitForwardGatherBackward.apply(input_, SEQUENCE_DIM)


def gather_for_sequence_parallel_region(input_):
    global SEQUENCE_DIM
    return _GatherForwardSplitBackward.apply(input_, SEQUENCE_DIM)
