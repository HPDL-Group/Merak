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


import torch
import torch.distributed as dist

try:
    from torch._six import inf
except ImportError:
    from torch import inf


def get_data_parallel_partitions(self, tensor):
    partitions = []

    dp = dist.get_world_size(group=self.dp_process_group)
    dp_id = dist.get_rank(group=self.dp_process_group)

    total_num_elements = tensor.numel()

    base_size = total_num_elements // dp
    remaining = total_num_elements % dp

    start = 0
    for id in range(dp):
        partition_size = base_size
        if id < remaining:
            partition_size = partition_size + 1
        partitions.append(tensor.narrow(0, start, partition_size))
        start = start + partition_size
    return partitions


def get_alignment_padding(flattened_lean_size, sub_partition_id, sub_partition_size):
    sub_partition_high_limit = (sub_partition_id + 1) * sub_partition_size
    if sub_partition_high_limit <= flattened_lean_size:
        return 0
    else:
        return min(sub_partition_size, sub_partition_high_limit - flattened_lean_size)


def get_group_alignment_padding(tensor_list, sub_partition_size, sub_partition_count):
    group_paddings = []
    flattened_size = sum([tensor.numel() for tensor in tensor_list])
    for i in range(sub_partition_count):
        padding = get_alignment_padding(flattened_size, i, sub_partition_size)
        group_paddings.append(padding)

    return group_paddings


def _single_range_check(current_index, start_index, end_index, tensor_size):
    offset = 0
    if (current_index >= start_index) and (current_index < end_index):
        # Fully inside bounds
        return True, offset
    elif (start_index > current_index) and (
        start_index < (current_index + tensor_size)
    ):
        # Partially contained, compute offset
        offset = start_index - current_index
        return True, offset
    else:
        return False, offset


def _range_check(current_index, element_intervals, tensor_size):
    results = []
    for comm_idx, interval in enumerate(element_intervals):
        start_index, end_index = interval
        contained, offset = _single_range_check(
            current_index, start_index, end_index, tensor_size
        )
        if contained:
            results.append((contained, offset, comm_idx))
    if len(results) == 0:
        return [(False, 0, -1)]
    return results


def get_grad_norm(gradients, parameters, norm_type=2, mpu=None):
    """Get grad norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_
    and added functionality to handle model parallel parameters.
    Note that the gradients are modified in place. Taken from
    Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of
        Tensors or a single Tensor that will have gradients
        normalized max_norm (float or int): max norm of the
        gradients

        norm_type (float or int): type of the used p-norm.
        Can be ``'inf'`` for infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    # my_group = _initialize_parameter_parallel_groups()

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in gradients)
        if torch.cuda.is_available():
            total_norm_cuda = torch.FloatTensor([float(total_norm)]).cuda()
        else:
            total_norm_cuda = torch.FloatTensor([float(total_norm)])
        # # Take max across all GPUs.
        torch.distributed.all_reduce(
            total_norm_cuda,
            op=torch.distributed.ReduceOp.MAX,
            group=mpu.get_data_parallel_group(),
        )

        if (
            mpu.get_model_parallel_group is not None
            and mpu.get_model_parallel_world_size() != 1
        ):
            torch.distributed.all_reduce(
                total_norm_cuda,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_model_parallel_group(),
            )

        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.0
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
        for g, p in zip(gradients, parameters):
            # Pipeline parallelism may replicate parameters.
            # Avoid multi-counting.
            if hasattr(p, "ds_pipe_replicated") and p.ds_pipe_replicated:
                continue

            # Filter to avoid over-counting replicated tensors from tensor
            # model parallelism
            if tensor_mp_rank > 0:
                continue

            # if is_model_parallel_parameter(p) or (tensor_mp_rank  == 0):
            param_norm = g.data.double().norm(norm_type)
            total_norm += param_norm.item() ** norm_type

        # Sum across all model parallel GPUs.
        if torch.cuda.is_available():
            total_norm_cuda = torch.FloatTensor([float(total_norm)]).cuda()
        else:
            total_norm_cuda = torch.FloatTensor([float(total_norm)])

        torch.distributed.all_reduce(
            total_norm_cuda,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_data_parallel_group(),
        )

        if (
            mpu.get_model_parallel_group is not None
            and mpu.get_model_parallel_world_size() != 1
        ):
            torch.distributed.all_reduce(
                total_norm_cuda,
                op=torch.distributed.ReduceOp.SUM,
                group=mpu.get_model_parallel_group(),
            )

        total_norm = total_norm_cuda[0].item() ** (1.0 / norm_type)

    if (
        total_norm == float("inf")
        or total_norm == -float("inf")
        or total_norm != total_norm
    ):
        total_norm = -1

    return total_norm


def bwc_tensor_model_parallel_rank(mpu=None):
    """Backwards-compatible way of querying the tensor model
    parallel rank from an ``mpu`` object.

    *Tensor* model parallelism means that tensors are physically
    split across processes. This contrasts with *pipeline* model
    parallelism, in which the layers are partitioned but tensors
    left intact.

    The API for tensor model parallelism has changed across versions
    and this helper provides a best-effort implementation across
    versions of ``mpu`` objects.  The preferred mechanism is
    ``mpu.get_tensor_model_parallel_rank()``.

    This should "just work" with both Megatron-LM and DeepSpeed's
    pipeline parallelism.

    Args:
        mpu (model parallel unit, optional): The tensor model
        parallel rank.
        If ``mpu=None``, returns 0. Defaults to ``None``.

    Returns:
        int: the rank
    """
    if mpu is None:
        # No model parallelism in easy :)
        return 0

    if hasattr(mpu, "get_tensor_model_parallel_rank"):
        # New Megatron and DeepSpeed convention
        # (post pipeline-parallelism release)
        return mpu.get_tensor_model_parallel_rank()
    elif hasattr(mpu, "get_slice_parallel_rank"):
        # Some DeepSpeed + pipeline parallelism versions
        return mpu.get_slice_parallel_rank()
    else:
        # Deprecated Megatron and DeepSpeed convention
        return mpu.get_model_parallel_rank()


class LossScalerBase:
    """LossScalarBase
    Base class for a loss scaler
    """

    def __init__(self, cur_scale):
        self.cur_scale = cur_scale

    @property
    def loss_scale(self):
        return self.cur_scale

    def scale_gradient(self, module, grad_in, grad_out):
        return tuple(self.loss_scale * g for g in grad_in)

    def update_scale(self, overflow):
        pass

    def backward(self, loss, retain_graph=False):
        scaled_loss = loss * self.loss_scale
        scaled_loss.backward(retain_graph=retain_graph)


class LossScaler(LossScalerBase):
    """
    Class that manages a static loss scale.  This class is intended
    to interact with :class:`FP16_Optimizer`, and should not be
    directly manipulated by the user.

    Use of :class:`LossScaler` is enabled via the ``static_loss_scale``
    argument to:class:`FP16_Optimizer`'s constructor.

    Args:
        scale (float, optional, default=1.0):  The loss scale.
    """

    def __init__(self, scale=1):
        super(LossScaler, self).__init__(scale)

    # `params` is a list / generator of torch.Variable
    def has_overflow(self, params):
        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        return False


class DynamicLossScaler(LossScalerBase):
    """
    Class that manages dynamic loss scaling.  It is recommended
    to use :class:`DynamicLossScaler`indirectly, by supplying
    ``dynamic_loss_scale=True`` to the constructor of:class:
    `FP16_Optimizer`.

    However, it's important to understand how :class:`DynamicLossScaler`
    operates, because the default options can be changed using the the
    ``dynamic_loss_args`` argument to :class:`FP16_Optimizer`'s
    constructor.

    Loss scaling is designed to combat the problem of underflowing
    gradients encountered at long times when training fp16 networks.
    Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.
    If overflowing gradients are encountered, :class:`DynamicLossScaler`
    informs :class:`FP16_Optimizer` that an overflow has occurred.
    :class:`FP16_Optimizer` then skips the update step for this
    particular iteration/minibatch, and :class:`DynamicLossScaler`
    adjusts the loss scale to a lower value. If a certain number of
    iterations occur without overflowing gradients detected,
    :class:`DynamicLossScaler` increases the loss scale once more.
    In this way :class:`DynamicLossScaler` attempts to "ride the edge"
    of always using the highest loss scale possible without incurring
    overflow.

    Args:
        init_scale (float, optional, default=2**32):
        Initial loss scale attempted by :class:`DynamicLossScaler.`

        scale_factor (float, optional, default=2.0):
        Factor used when adjusting the loss scale.
        If an overflow is encountered, the loss scale is readjusted
        to loss scale/``scale_factor``.  If ``scale_window`` consecutive
        iterations take place without an overflow, the loss scale is
        readjusted to loss_scale*``scale_factor``.

        scale_window (int, optional, default=1000):
        Number of consecutive iterations without an overflow to wait
        before increasing the loss scale.
    """

    def __init__(
        self,
        init_scale=2**32,
        scale_factor=2.0,
        scale_window=1000,
        min_scale=1,
        delayed_shift=1,
        consecutive_hysteresis=False,
    ):
        super(DynamicLossScaler, self).__init__(init_scale)
        self.cur_iter = 0
        self.last_overflow_iter = -1
        self.scale_factor = scale_factor
        self.scale_window = scale_window
        self.min_scale = min_scale
        self.delayed_shift = delayed_shift
        self.cur_hysteresis = delayed_shift
        self.consecutive_hysteresis = consecutive_hysteresis

    # `params` is a list / generator of torch.Variable
    def has_overflow_serial(self, params):
        for p in params:
            if p.grad is not None and self._has_inf_or_nan(p.grad.data):
                return True

        return False

    # `x` is a torch.Tensor
    def _has_inf_or_nan(x):
        try:
            # if x is half, the .float() incurs an additional deep copy,
            # but it's necessary if Pytorch's .sum() creates a one-element
            # tensor of the same type as x (which is true for some recent
            # version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a
            # Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum in [float("inf"), -float("inf")] or cpu_sum != cpu_sum:
                return True
            return False

    # `overflow` is boolean indicating whether the gradient overflowed
    def update_scale(self, overflow):
        if overflow:
            # self.cur_scale /= self.scale_factor
            if self.delayed_shift == 1 or self.cur_hysteresis == 1:
                self.cur_scale = max(self.cur_scale / self.scale_factor, self.min_scale)
            else:
                self.cur_hysteresis -= 1
            self.last_overflow_iter = self.cur_iter
        else:
            if self.consecutive_hysteresis:
                self.cur_hysteresis = self.delayed_shift
            if (self.cur_iter - self.last_overflow_iter) % self.scale_window == 0:
                if not self.consecutive_hysteresis:
                    self.cur_hysteresis = self.delayed_shift
                self.cur_scale *= self.scale_factor
        self.cur_iter += 1
