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

from typing import Dict, List, Tuple, Union

import torch
import torch.distributed as dist

from Merak import get_logger

from . import mpu
from .mpu.p2p_communication import recv_forward, send_forward

ID_TO_DTYPE = [
    torch.float32,
    torch.float64,
    torch.complex64,
    torch.complex128,
    torch.float16,
    torch.bfloat16,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.bool,
]
DTYPE_TO_ID = {dtype: id_ for id_, dtype in enumerate(ID_TO_DTYPE)}
mem_alloced = 0
mem_cached = 0


def split_half_float_double(
    tensors: torch.Tensor,
) -> List[Tuple[torch.dtype, List[torch.Tensor]]]:
    dtypes = [
        "torch.cuda.HalfTensor",
        "torch.cuda.FloatTensor",
        "torch.cuda.DoubleTensor",
        "torch.HalfTensor",
        "torch.FloatTensor",
        "torch.DoubleTensor",
    ]
    buckets = []
    for i, dtype in enumerate(dtypes):
        bucket = [t for t in tensors if t.type() == dtype]
        if bucket:
            buckets.append((dtype, bucket))
    return buckets


def _send_tensor_meta(buffer: Tuple[torch.Tensor], device: torch.device):
    """Communicate metadata about upcoming p2p transfers.

    Metadata is communicated in this order:
        * type (0: tensor, 1: list)
        * num_tensors if type=list
        foreach tensor in buffer:
            * ndims
            * shape
    """
    assert isinstance(buffer, tuple), f"Could not send meta type {type(buffer)}."
    count_tensor = torch.LongTensor(data=[len(buffer)]).to(device)
    send_forward(count_tensor)
    for idx, tensor in enumerate(buffer):
        assert isinstance(tensor, torch.Tensor)
        send_shape = torch.LongTensor(data=tensor.size()).to(device)
        send_ndims = torch.LongTensor(data=[len(tensor.size())]).to(device)
        send_dtype = torch.LongTensor(data=[DTYPE_TO_ID[tensor.dtype]]).to(device)
        send_req_grad = torch.LongTensor(data=[1 if tensor.requires_grad else 0]).to(
            device
        )

        send_forward(send_dtype)
        send_forward(send_ndims)
        send_forward(send_shape)
        send_forward(send_req_grad)
        # Useful for performance debugging.
        """
        new_bytes = _tensor_bytes(tensor)
        send_bytes += _tensor_bytes(tensor)
        # Useful for performance debugging.
        if self.grid.data_parallel_id == 0:
            print(
                f'STAGE={self.stage_id} pipe-send-volume[{idx}]:'
                f'shape={send_shape} {new_bytes/1024**2:0.2f}MB'
            )
        """

    # Useful for performance debugging.
    """
    if self.grid.data_parallel_id == 0:
        print(f'STAGE={self.stage_id} pipe-send-volume:'
            f'{send_bytes/1024**2:0.2f}MB')
    """


def _recv_tensor_meta(device: torch.device) -> Tuple[torch.Tensor]:
    """Receive metadata about upcoming p2p transfers and return allocated
    buffers.

    Metadata is communicated in this order:
        * type (0: tensor, 1: list)
        * num_tensors if type=list
        foreach tensor in buffer:
            * ndims
            * shape

    Returns:
        Allocated buffer for receiving from send_stage.
    """

    count_tensor = torch.LongTensor(data=[0]).to(device)
    recv_forward(count_tensor)
    num_tensors = count_tensor.item()
    recv_shapes_and_dtypes = []
    recv_req_grads = []
    for idx in range(num_tensors):
        recv_dtype = torch.LongTensor(data=[0]).to(device)
        recv_forward(recv_dtype)
        recv_dtype = ID_TO_DTYPE[recv_dtype.item()]
        recv_ndims = torch.LongTensor(data=[0]).to(device)
        recv_forward(recv_ndims)
        recv_ndims = recv_ndims.item()
        recv_shape = torch.LongTensor([1] * recv_ndims).to(device)
        recv_forward(recv_shape)
        recv_shapes_and_dtypes.append((recv_shape.tolist(), recv_dtype))
        recv_req_grad = torch.LongTensor(data=[0]).to(device)
        recv_forward(recv_req_grad)
        recv_req_grads.append(True if recv_req_grad.item() == 1 else False)

    buffers = _allocate_buffers(
        recv_shapes_and_dtypes, device, num_buffers=1, requires_grad=recv_req_grads
    )[0]
    # Convert to tuples if requested.
    # if recv_type == 2:
    buffers = tuple(buffers)
    return buffers


def _allocate_zeros(shape: List[int], device: torch.device, **kwargs) -> torch.Tensor:
    """Allocate a tensor of zeros on the engine's device.

    Arguments:
        shape: the shape of the tensor to allocate
        fp16 (bool): whether to use FP16. default: defer to self.
                        fp16_enabled()
        kwargs: passed to torch.zeros()

    Returns:
        A tensor from torch.zeros() allocated on self.device.
    """
    # if "dtype" not in kwargs and self.fp16_enabled():
    #     kwargs["dtype"] = torch.half

    return torch.zeros(shape, device=device, **kwargs)


def _allocate_buffer(
    shape: List[int], device: torch.device, num_buffers: int = -1, **kwargs
) -> List[torch.Tensor]:
    buffers = []
    for count in range(num_buffers):
        buffers.append(_allocate_zeros(shape, device, **kwargs))
    return buffers


def _allocate_buffers(
    shapes_and_dtypes: List[Union[List[int], torch.dtype]],
    device: torch.device,
    requires_grad: List[bool] = [False],
    num_buffers: int = -1,
) -> List[torch.Tensor]:
    buffers = []
    for count in range(num_buffers):
        buffer = []
        for i, (shape, dtype) in enumerate(shapes_and_dtypes):
            buffer.append(
                _allocate_zeros(
                    shape, device, dtype=dtype, requires_grad=requires_grad[i]
                )
            )
        buffers.append(buffer)
    return buffers


def _create_buffer(
    tensor: Tuple[torch.Tensor], device: torch.device
) -> List[torch.Tensor]:
    sizes_and_dtype = [[list(t.size()), t.dtype] for t in tensor]
    req_grad = [t.requires_grad for t in tensor]
    buffer = _allocate_buffers(
        sizes_and_dtype, device, num_buffers=1, requires_grad=req_grad
    )[0]
    return buffer


def _scale_loss_by_gas(
    prescaled_loss: Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]],
    gradient_accumulation_steps: int,
) -> Union[torch.Tensor, Tuple[torch.Tensor], List[torch.Tensor]]:
    logger = get_logger("detailed")
    if isinstance(prescaled_loss, torch.Tensor):
        scaled_loss = prescaled_loss / gradient_accumulation_steps
    elif isinstance(prescaled_loss, tuple) or isinstance(prescaled_loss, list):
        scaled_loss = []
        for l in prescaled_loss:
            if isinstance(l, torch.Tensor):
                scaled_loss.append(l / gradient_accumulation_steps)
            else:
                scaled_loss.append(l)
    else:
        scaled_loss = prescaled_loss
        logger.warning(
            f"Merak unable to scale loss because of type: \
              {type(prescaled_loss)}"
        )

    return scaled_loss


def _zero_grads(inputs: Union[torch.Tensor, Dict, Tuple, List]):
    if isinstance(inputs, torch.Tensor):
        if inputs.grad is not None:
            inputs.grad.data.zero_()
    elif isinstance(inputs, dict):
        for t in inputs.values():
            if t.grad is not None:
                t.grad.data.zero_()
    else:
        for t in inputs:
            if t.grad is not None:
                t.grad.data.zero_()


def mem_status(msg: str, print_rank: int = -1, reset_max: bool = False):
    return
    global mem_alloced, mem_cached
    if (
        not self.train_params.global_steps == 0
        or not self.train_params.global_steps == 9
    ):
        # return
        pass
    if mpu.get_data_parallel_rank() != 0:
        return

    if self.global_rank != 0:
        return

    rank = self.global_rank
    if print_rank != -1 and rank != print_rank:
        return

    torch.cuda.synchronize()

    if reset_max:
        torch.cuda.reset_max_memory_cached()
        torch.cuda.reset_max_memory_allocated()

    new_alloced = torch.cuda.memory_allocated()
    new_cached = torch.cuda.memory_cached()

    delta_alloced = new_alloced - mem_alloced
    delta_cached = new_cached - mem_cached

    mem_cached = new_cached
    mem_alloced = new_alloced

    max_alloced = torch.cuda.max_memory_allocated()
    max_cached = torch.cuda.max_memory_cached()

    # convert to GB for printing
    new_alloced /= 1024**3
    new_cached /= 1024**3
    delta_alloced /= 1024**3
    delta_cached /= 1024**3
    max_alloced /= 1024**3
    max_cached /= 1024**3

    print(
        f"RANK={rank} STAGE={self.stage_id} \
              STEP={self.train_params.global_steps} MEMSTATS",
        msg,
        f"current alloc={new_alloced:0.4f}GB \
              (delta={delta_alloced:0.4f}GB max={max_alloced:0.4f}GB) "
        f"current cache={new_cached:0.4f}GB \
              (delta={delta_cached:0.4f}GB max={max_cached:0.4f}GB)",
    )
