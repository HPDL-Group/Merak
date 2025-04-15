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


# Parts of the code here are adapted from PyTorch
# repo: https://github.com/pytorch/pytorch
# https://github.com/NVIDIA/Megatron-LM/blob/806422e5ec35c27b027dbb413b05e27b6590dc56/megatron/mpu/layers.py

import math

import torch
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter

from torch.nn import LayerNorm
from .initialize import get_model_parallel_rank, get_model_parallel_group
from .initialize import get_model_parallel_world_size, get_model_parallel_rank
from .mappings import (
    copy_to_model_parallel_region,
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
    scatter_to_conv_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    get_sequence_dim,
)
from ..recompute.checkpointing import get_rng_tracker
from .utils import divide, channels_uniform
from .utils import split_tensor_along_last_dim
from .utils import VocabUtility
from Merak.merak_args import get_args

def _initialize_affine_weight_gpu(weight, init_method,
                                  partition_dim, stride=1):
    """Initialize affine weight for model parallel on GPU."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride
    
    with get_rng_tracker().fork():
        init_method(weight)


def _initialize_affine_weight_cpu(weight, output_size, input_size,
                                  per_partition_size, partition_dim,
                                  init_method, stride=1,
                                  return_master_weight=False):
    """Initialize affine weight for model parallel.

    Build the master weight on all processes and scatter
    the relevant chunk."""

    weight.model_parallel = True
    weight.partition_dim = partition_dim
    weight.partition_stride = stride

    # Initialize master weight
    master_weight = torch.empty(output_size, input_size,
                                dtype=torch.float,
                                requires_grad=False)
    init_method(master_weight)
    args = get_args()
    master_weight = master_weight.to(dtype=args.params_dtype)

    # Split and copy
    per_partition_per_stride_size = divide(per_partition_size, stride)
    weight_list = torch.split(master_weight, per_partition_per_stride_size,
                              dim=partition_dim)
    rank = get_model_parallel_rank()
    world_size = get_model_parallel_world_size()
    my_weight_list = weight_list[rank::world_size]

    with torch.no_grad():
        torch.cat(my_weight_list, dim=partition_dim, out=weight)
    if return_master_weight:
        return master_weight
    return None


class VocabParallelEmbedding(torch.nn.Module):
    """Embedding parallelized in the vocabulary dimension.

    This is mainly adapted from torch.nn.Embedding and all the default
    values are kept.
    Arguments:
        num_embeddings: vocabulary size.
        embedding_dim: size of hidden state.
        init_method: method to initialize weights.
    """

    def __init__(self, num_embeddings, embedding_dim,
                 init_method=init.xavier_normal_):
        super(VocabParallelEmbedding, self).__init__()
        # Keep the input dimensions.
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # Set the detauls for compatibility.
        self.padding_idx = None
        self.max_norm = None
        self.norm_type = 2.
        self.scale_grad_by_freq = False
        self.sparse = False
        self._weight = None
        self.model_parallel_size = get_model_parallel_world_size()
        # Divide the weight matrix along the vocaburaly dimension.
        (
            self.vocab_start_index,
            self.vocab_end_index
        ) = VocabUtility.vocab_range_from_global_vocab_size(
                self.num_embeddings,
                get_model_parallel_rank(),
                self.model_parallel_size
            )
        self.num_embeddings_per_partition = \
            self.vocab_end_index - self.vocab_start_index

        # Allocate weights and initialize.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                dtype=args.params_dtype))
            _initialize_affine_weight_cpu(
                self.weight, self.num_embeddings, self.embedding_dim,
                self.num_embeddings_per_partition, 0, init_method)
        else:
            self.weight = Parameter(torch.empty(
                self.num_embeddings_per_partition, self.embedding_dim,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=1)

    def forward(self, input_):
        if self.model_parallel_size > 1:
            # Build the mask.
            input_mask = (input_ < self.vocab_start_index) | \
                         (input_ >= self.vocab_end_index)
            # Mask the input.
            masked_input = input_.clone() - self.vocab_start_index
            masked_input[input_mask] = 0
        else:
            masked_input = input_
            # Get the embeddings.
        output_parallel = F.embedding(masked_input, self.weight,
                                      self.padding_idx, self.max_norm,
                                      self.norm_type, self.scale_grad_by_freq,
                                      self.sparse)
        # Mask the output embedding.
        if self.model_parallel_size > 1:
            output_parallel[input_mask, :] = 0.0
        # Reduce across all the model parallel GPUs.
        output = reduce_from_model_parallel_region(output_parallel)
        return output


class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gether on output and make Y avaiable
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where
        bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(ColumnParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        empty_tensor = torch.empty(self.output_size_per_partition,
                                   self.input_size,
                                   dtype=args.params_dtype)
        if args.use_cpu_initialization:
            self.weight = Parameter(empty_tensor)
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(empty_tensor.to(device=torch.cuda.current_device()))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=stride)

        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)


    def forward(self, input_):
        # Set up backprop all-reduce.
        input_parallel = copy_to_model_parallel_region(input_)
        # Matrix multiply.

        bias = self.bias if not self.skip_bias_add else None
        output_parallel = F.linear(input_parallel, self.weight, bias)
        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_size, self.output_size_per_partition, 
            self.bias is not None
        )


class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its first dimension and X along its second dimension as:
               -   -
              | A_1 |
              | .   |
          A = | .   |        X = [X_1, ..., X_p]
              | .   |
              | A_p |
               -   -
    Arguments:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split
                           again.
        init_method: method to initialize weights. Note that bias is always set
                     to zero.
        stride: For the strided linear layers.
        keep_master_weight_for_test: This was added for testing and should be
                                     set to False. It returns the master weights
                                     used for initialization.
        skip_bias_add: This was added to enable performance optimations where
        bias
                       can be fused with other elementwise operations. we skip 
                       adding bias but instead return it.
    """

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(RowParallelLinear, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        empty_tensor = torch.empty(self.output_size,
                                   self.input_size_per_partition,
                                   dtype=args.params_dtype)
        if args.use_cpu_initialization:
            self.weight = Parameter(empty_tensor)
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(empty_tensor.to(torch.cuda.current_device()))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=1, stride=stride)
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_model_parallel_region(input_)
        # Matrix multiply.
        output_parallel = F.linear(input_parallel, self.weight)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_size_per_partition, self.output_size,
            self.bias is not None
        )

class ColParallelConv2d(torch.nn.Module):
    """Conv2d layer with channels parallelism.

    Arguments:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation = 1,
                 groups = 1,
                 bias=True,
                 input_is_parallel=False,
                 **kwargs
        ):
        super(ColParallelConv2d, self).__init__()

        del kwargs
        # Keep input parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        self.world_size = get_model_parallel_world_size()
        self.rank = get_model_parallel_rank()
        self.in_channels_per_partition = channels_uniform(
            in_channels, self.world_size
        )
        self.in_channels_per_partition = self.in_channels_per_partition[self.rank]

        # def conv layer
        self.conv = torch.nn.Conv2d(
            self.in_channels_per_partition,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias
        )
        self.weight = self.conv.weight
        self.bias = self.conv.bias

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_conv_parallel_region(input_)
        # Matrix multiply.
        output_parallel = self.conv(input_parallel)
        # All-reduce across all the partitions.
        output_ = reduce_from_model_parallel_region(output_parallel)
        output = output_
        output_bias = self.bias
        return output, output_bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_channels_per_partition, self.out_channels,
            self.bias is not None
        )




from functools import reduce
import operator
class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name 
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        required_len = reduce(operator.mul, tensor_shape, 1)
        if self.buffer.get((name, dtype), None) is None or \
                self.buffer[(name, dtype)].numel() < required_len:
            if torch.cuda.is_available():
                device = torch.cuda.current_device()
            else:
                device = 'cpu'
            self.buffer[(name, dtype)] = torch.empty(
                required_len,
                dtype=dtype,
                device=device,
                requires_grad=False
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)

_GLOBAL_MEMORY_BUFFER =  GlobalMemoryBuffer()

def get_global_memory_buffer():
    return _GLOBAL_MEMORY_BUFFER

AVOID_EXTRA_ALLGATHER=False

class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """
    Linear layer execution with asynchronous communication and gradient
    accumulation fusion in backprop.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, sequence_parallel):
        if not AVOID_EXTRA_ALLGATHER:
            ctx.save_for_backward(input, weight)
        ctx.use_bias = bias is not None
        ctx.sequence_parallel = sequence_parallel
      
        if sequence_parallel:
            world_size = get_model_parallel_world_size()
            dim = get_sequence_dim()
            dim_size = list(input.size())
            dim_size[dim] = dim_size[dim] * world_size


            all_gather_buffer = get_global_memory_buffer().get_tensor(
                                                            dim_size,
                                                            input.dtype, "mpu"
                                                        )
            torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=get_model_parallel_group())
            total_input = all_gather_buffer
        else:
            total_input = input
        if AVOID_EXTRA_ALLGATHER:
            ctx.save_for_backward(total_input, weight)
        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        # total_input = input
        if ctx.sequence_parallel and not AVOID_EXTRA_ALLGATHER:
            world_size = get_model_parallel_world_size()
            dim = get_sequence_dim()
            dim_size = list(input.size())
            dim_size[dim] = dim_size[dim] * world_size

            all_gather_buffer = get_global_memory_buffer().get_tensor(
                                                            dim_size,
                                                            input.dtype, "mpu"
                                                        )
            handle = torch.distributed._all_gather_base(
                all_gather_buffer,
                input,
                group=get_model_parallel_group(), async_op=True)

            # Delay the start of intput gradient computation shortly
            # (3us) to have gather scheduled first and have GPU resources
            # allocated
            _ = torch.empty(1, device=grad_output.device) + 1
            total_input = all_gather_buffer
        else:
            total_input = input
        grad_input = grad_output.matmul(weight)

        if ctx.sequence_parallel and not AVOID_EXTRA_ALLGATHER:
            handle.wait()
            # # Note: torch.cat already creates a contiguous tensor.
            # total_input = torch.cat(all_gather_buffer).contiguous()

        # Convert the tensor shapes to 2D for execution compatibility
        grad_output = grad_output.view(grad_output.shape[0] *
                                       grad_output.shape[1],
                                       grad_output.shape[2])
        total_input = total_input.view(total_input.shape[0] *
                                       total_input.shape[1],
				       total_input.shape[2])
 
 
        if ctx.sequence_parallel:
            dim_size = list(input.size())
            if AVOID_EXTRA_ALLGATHER:
                dim = get_sequence_dim
                dim_size[dim] = dim_size[dim] // get_model_parallel_world_size()

            sub_grad_input = torch.empty(
                dim_size, dtype=input.dtype,
                device=grad_output.device,
                requires_grad=False
            )
            # reduce_scatter
            handle = torch.distributed._reduce_scatter_base(
                sub_grad_input,
                grad_input,
                group=get_model_parallel_group(),
                async_op=True
            )
            # Delay the start of weight gradient computation shortly
            # (3us) to have reduce scatter scheduled first and have GPU
            # resources allocated
            _ = torch.empty(1, device=grad_output.device) + 1
        


        grad_weight = grad_output.t().matmul(total_input)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        if ctx.sequence_parallel:
            handle.wait()

            return sub_grad_input, grad_weight, grad_bias, None, None, None



        return grad_input, grad_weight, grad_bias, None, None, None



class ColumnSequenceParallel(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True, gather_output=True,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(ColumnSequenceParallel, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size_per_partition,
                                                self.input_size,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.output_size_per_partition, 0, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size_per_partition, self.input_size,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=0, stride=stride)
            
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition, dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            self.bias.model_parallel = True
            self.bias.partition_dim = 0
            self.bias.stride = stride
            # Always initialize bias to zero.
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)


    def forward(self, input_):

        input_parallel = input_

        bias = self.bias if not self.skip_bias_add else None

        output_parallel = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel, self.weight, bias, True)

        if self.gather_output:
            # All-gather across the partitions.
            output = gather_from_model_parallel_region(output_parallel)
        else:
            output = output_parallel 
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
    
    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_size, self.output_size_per_partition,
            self.bias is not None
        )


class RowSequenceParallel(torch.nn.Module):

    def __init__(self, input_size, output_size, bias=True,
                 input_is_parallel=False,
                 init_method=init.xavier_normal_, stride=1,
                 keep_master_weight_for_test=False,
                 skip_bias_add=False):
        super(RowSequenceParallel, self).__init__()

        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        # Divide the weight matrix along the last dimension.
        world_size = get_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        self.skip_bias_add = skip_bias_add

        # Parameters.
        # Note: torch.nn.functional.linear performs XA^T + b and as a result
        # we allocate the transpose.
        # Initialize weight.
        args = get_args()
        if args.use_cpu_initialization:
            self.weight = Parameter(torch.empty(self.output_size,
                                                self.input_size_per_partition,
                                                dtype=args.params_dtype))
            self.master_weight = _initialize_affine_weight_cpu(
                self.weight, self.output_size, self.input_size,
                self.input_size_per_partition, 1, init_method,
                stride=stride, return_master_weight=keep_master_weight_for_test)
        else:
            self.weight = Parameter(torch.empty(
                self.output_size, self.input_size_per_partition,
                device=torch.cuda.current_device(), dtype=args.params_dtype))
            _initialize_affine_weight_gpu(self.weight, init_method,
                                          partition_dim=1, stride=stride)
        if bias:
            if args.use_cpu_initialization:
                self.bias = Parameter(torch.empty(self.output_size,
                                                  dtype=args.params_dtype))
            else:
                self.bias = Parameter(torch.empty(
                    self.output_size, device=torch.cuda.current_device(),
                    dtype=args.params_dtype))
            # Always initialize bias to zero.
            setattr(self.bias, 'sequence_parallel', args.sequence_parallel)
            with torch.no_grad():
                self.bias.zero_()
        else:
            self.register_parameter('bias', None)

    def forward(self, input_):
        # Set up backprop all-reduce.
        if self.input_is_parallel:
            input_parallel = input_
        # Matrix multiply.
        # output_parallel = F.linear(input_parallel, self.weight)

        output_parallel = LinearWithGradAccumulationAndAsyncCommunication.apply(
            input_parallel, self.weight, None, None)
        output_ = reduce_scatter_to_sequence_parallel_region(output_parallel)

        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias
        return output, output_bias

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.input_size_per_partition, self.output_size,
            self.bias is not None
        )

