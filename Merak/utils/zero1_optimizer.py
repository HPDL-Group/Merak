'''
Copyright 2019 The Microsoft DeepSpeed Team

This file is adapted from FP16_DeepSpeedZeroOptimizer_Stage1 in deepspeed/runtime/zero
'''
# the code here are adapted from https://github.com/microsoft/DeepSpeed/releases/tag/v0.5.8/deepspeed/runtime/zero/stage1.py

import math
import torch
import torch.distributed as dist
from collections import defaultdict
try:
    from torch._six import inf
except ImportError:
    from torch import inf

from Merak.runtime.utils import CheckOverflow
from Merak.utils import logger, log_dist
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
import gc
from .. import mpu


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
    elif (start_index > current_index) and (start_index < (current_index + tensor_size)):
        # Partially contained, compute offset
        offset = start_index - current_index
        return True, offset
    else:
        return False, offset


def _range_check(current_index, element_intervals, tensor_size):
    results = []
    for comm_idx, interval in enumerate(element_intervals):
        start_index, end_index = interval
        contained, offset = _single_range_check(current_index, start_index, end_index, tensor_size)
        if contained:
            results.append((contained, offset, comm_idx))
    if len(results) == 0:
        return [(False, 0, -1)]
    return results

def move_to_cpu(tensor_list):
    for tensor in tensor_list:
        tensor.data = tensor.data.cpu()

def get_grad_norm(gradients, parameters, norm_type=2, mpu=None):
    """Get grad norm of an iterable of parameters.

    This is adapted from torch.nn.utils.clip_grad.clip_grad_norm_ and
    added functionality to handle model parallel parameters. Note that
    the gradients are modified in place. Taken from Nvidia Megatron.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """

    # my_group = _initialize_parameter_parallel_groups()

    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(g.data.abs().max() for g in gradients)
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])
        # # Take max across all GPUs.
        torch.distributed.all_reduce(total_norm_cuda,
                                        op=torch.distributed.ReduceOp.MAX,
                                        group=mpu.get_data_parallel_group())

        if mpu.get_model_parallel_group is not None and mpu.get_model_parallel_world_size() != 1:
            torch.distributed.all_reduce(total_norm_cuda,
                                         op=torch.distributed.ReduceOp.SUM,
                                         group=mpu.get_model_parallel_group())



        total_norm = total_norm_cuda[0].item()
    else:
        total_norm = 0.
        tensor_mp_rank = bwc_tensor_model_parallel_rank(mpu=mpu)
        for g, p in zip(gradients, parameters):
            # Pipeline parallelism may replicate parameters. Avoid multi-counting.
            if hasattr(p, 'ds_pipe_replicated') and p.ds_pipe_replicated:
                continue

            # Filter to avoid over-counting replicated tensors from tensor
            # model parallelism
            if tensor_mp_rank > 0:
                continue

            # if is_model_parallel_parameter(p) or (tensor_mp_rank  == 0):
            param_norm = g.data.double().norm(norm_type)
            total_norm += param_norm.item()**norm_type

        # Sum across all model parallel GPUs.
        total_norm_cuda = torch.cuda.FloatTensor([float(total_norm)])

        torch.distributed.all_reduce(total_norm_cuda,
                                        op=torch.distributed.ReduceOp.SUM,
                                        group=mpu.get_data_parallel_group())

        if mpu.get_model_parallel_group is not None and mpu.get_model_parallel_world_size() != 1:
            torch.distributed.all_reduce(total_norm_cuda,
                                            op=torch.distributed.ReduceOp.SUM,
                                            group=mpu.get_model_parallel_group())


        total_norm = total_norm_cuda[0].item()**(1. / norm_type)

    if total_norm == float(
            'inf') or total_norm == -float('inf') or total_norm != total_norm:
        total_norm = -1

    return total_norm

def bwc_tensor_model_parallel_rank(mpu=None):
    """Backwards-compatible way of querying the tensor model parallel rank from
    an ``mpu`` object.

    *Tensor* model parallelism means that tensors are physically split across
    processes. This contrasts with *pipeline* model parallelism, in which the
    layers are partitioned but tensors left intact.

    The API for tensor model parallelism has changed across versions and this
    helper provides a best-effort implementation across versions of ``mpu``
    objects.  The preferred mechanism is
    ``mpu.get_tensor_model_parallel_rank()``.

    This should "just work" with both Megatron-LM and DeepSpeed's pipeline
    parallelism.

    Args:
        mpu (model parallel unit, optional): The tensor model parallel rank.
            If ``mpu=None``, returns 0. Defaults to ``None``.

    Returns:
        int: the rank
    """
    if mpu is None:
        # No model parallelism in easy :)
        return 0

    if hasattr(mpu, 'get_tensor_model_parallel_rank'):
        # New Megatron and DeepSpeed convention (post pipeline-parallelism release)
        return mpu.get_tensor_model_parallel_rank()
    elif hasattr(mpu, 'get_slice_parallel_rank'):
        # Some DeepSpeed + pipeline parallelism versions
        return mpu.get_slice_parallel_rank()
    else:
        # Deprecated Megatron and DeepSpeed convention
        return mpu.get_model_parallel_rank()

def bwc_tensor_model_parallel_rank(mpu=None):
    """Backwards-compatible way of querying the tensor model parallel rank from
    an ``mpu`` object.

    *Tensor* model parallelism means that tensors are physically split across
    processes. This contrasts with *pipeline* model parallelism, in which the
    layers are partitioned but tensors left intact.

    The API for tensor model parallelism has changed across versions and this
    helper provides a best-effort implementation across versions of ``mpu``
    objects.  The preferred mechanism is
    ``mpu.get_tensor_model_parallel_rank()``.

    This should "just work" with both Megatron-LM and DeepSpeed's pipeline
    parallelism.

    Args:
        mpu (model parallel unit, optional): The tensor model parallel rank.
            If ``mpu=None``, returns 0. Defaults to ``None``.

    Returns:
        int: the rank
    """
    if mpu is None:
        # No model parallelism in easy :)
        return 0

    if hasattr(mpu, 'get_tensor_model_parallel_rank'):
        # New Megatron and DeepSpeed convention (post pipeline-parallelism release)
        return mpu.get_tensor_model_parallel_rank()
    elif hasattr(mpu, 'get_slice_parallel_rank'):
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
    Class that manages a static loss scale.  This class is intended to interact with
    :class:`FP16_Optimizer`, and should not be directly manipulated by the user.

    Use of :class:`LossScaler` is enabled via the ``static_loss_scale`` argument to
    :class:`FP16_Optimizer`'s constructor.

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
    Class that manages dynamic loss scaling.  It is recommended to use :class:`DynamicLossScaler`
    indirectly, by supplying ``dynamic_loss_scale=True`` to the constructor of
    :class:`FP16_Optimizer`.  However, it's important to understand how :class:`DynamicLossScaler`
    operates, because the default options can be changed using the
    the ``dynamic_loss_args`` argument to :class:`FP16_Optimizer`'s constructor.

    Loss scaling is designed to combat the problem of underflowing gradients encountered at long
    times when training fp16 networks.  Dynamic loss scaling begins by attempting a very high loss
    scale.  Ironically, this may result in OVERflowing gradients.  If overflowing gradients are
    encountered, :class:`DynamicLossScaler` informs :class:`FP16_Optimizer` that an overflow has
    occurred.
    :class:`FP16_Optimizer` then skips the update step for this particular iteration/minibatch,
    and :class:`DynamicLossScaler` adjusts the loss scale to a lower value.
    If a certain number of iterations occur without overflowing gradients detected,
    :class:`DynamicLossScaler` increases the loss scale once more.
    In this way :class:`DynamicLossScaler` attempts to "ride the edge" of
    always using the highest loss scale possible without incurring overflow.

    Args:
        init_scale (float, optional, default=2**32):  Initial loss scale attempted by :class:`DynamicLossScaler.`
        scale_factor (float, optional, default=2.0):  Factor used when adjusting the loss scale. If an overflow is encountered, the loss scale is readjusted to loss scale/``scale_factor``.  If ``scale_window`` consecutive iterations take place without an overflow, the loss scale is readjusted to loss_scale*``scale_factor``.
        scale_window (int, optional, default=1000):  Number of consecutive iterations without an overflow to wait before increasing the loss scale.
    """
    def __init__(self,
                 init_scale=2**32,
                 scale_factor=2.,
                 scale_window=1000,
                 min_scale=1,
                 delayed_shift=1,
                 consecutive_hysteresis=False):
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
            # if x is half, the .float() incurs an additional deep copy, but it's necessary if
            # Pytorch's .sum() creates a one-element tensor of the same type as x
            # (which is true for some recent version of pytorch).
            cpu_sum = float(x.float().sum())
            # More efficient version that can be used if .sum() returns a Python scalar
            # cpu_sum = float(x.sum())
        except RuntimeError as instance:
            # We want to check if inst is actually an overflow exception.
            # RuntimeError could come from a different error.
            # If so, we still want the exception to propagate.
            if "value cannot be converted" not in instance.args[0]:
                raise
            return True
        else:
            if cpu_sum in [float('inf'), -float('inf')] or cpu_sum != cpu_sum:
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

class FP16_DeepSpeedZeroOptimizer_Stage1(object):
    """
    FP16_DeepSpeedZeroOptimizer_Stage1 designed to reduce the memory footprint
    required for training large deep learning models.

    For more details please see ZeRO: Memory Optimization Towards Training A Trillion Parameter Models
    https://arxiv.org/abs/1910.02054

    This version aligns with stage-1 in the paper above.
    """
    def __init__(self,
                 init_optimizer,
                 static_loss_scale=1.0,
                 dynamic_loss_scale=False,
                 dynamic_loss_args=None,
                 verbose=True,
                 dp_process_group=None,
                 partition_size=None,
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

        if dp_process_group is not None and partition_size is not None:
            raise ValueError("Cannot specify both dp_process_group "
                             "and partition size")

        if not torch.cuda.is_available:
            raise SystemError("Cannot use fp16 without CUDA.")
        self.optimizer = init_optimizer

        self.verbose = verbose
        self.dp_process_group = dp_process_group

        self.device = torch.cuda.current_device()

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

        # parallel_sub_partitioned_fp16_groups[group-idx] -> [comm-ids] -> [rank-ids]
        self.parallel_sub_partitioned_fp16_groups = []
        # same underlying data as above but viewed as: [groups] -> [rank-ids] -> [comm-ids]
        self.parallel_comm_sub_partitioned_fp16_groups = []

        # 32-bit sub-partitions of the parallel partitioned parameters
        # that this process will update
        self.local_sub_partitions_of_fp32_groups = []

        # param partition info

        # parameters in each group that will not be updated by this process directly
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

            move_to_cpu(self.fp16_groups[i])

            # flattens all tensors into single 1d tensor aligned with sub-partition size for later dividing
            # RS: create aligned sub-partitions
            self.fp16_groups_flat.append(self.flatten_dense_tensors_sub_partition_aligned(
                tensor_list=self.fp16_groups[i],
                dp=dist.get_world_size(group=self.dp_process_group),
                max_elements_per_comm=self.max_elems_per_comm[i],
                pg=self.dp_process_group).cuda(torch.cuda.current_device()))

            # TODO: I don't think this does anything?
            # set model fp16 weight to slices of flattened buffer
            updated_params = self.unflatten(self.fp16_groups_flat[i],
                                            self.fp16_groups[i])

            for p, q in zip(self.fp16_groups[i], updated_params):
                p.data = q.data

            # divide the flat weights into near equal partition equal to the data parallel degree
            # each process will compute on a different part of the partition
            # RS: split into two layer list -> [comm-id] -> [sub-partitions per rank]
            comm_partitions, dp_sub_partitions, element_intervals, sub_partition_size, num_comm_intervals = \
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

            # a partition of the fp32 master weights that will be updated by this process
            # RS: store/detach/cast our local sub-partitions
            local_sub_partitions = []
            for sub_partition in self.parallel_sub_partitioned_fp16_groups[i][
                    local_rank]:
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
            # self.single_partition_of_fp32_groups[i].requires_grad = True # keep this in case internal optimizer uses it
            param_group['params'] = self.local_sub_partitions_of_fp32_groups[i]

            # RS: divide up the sub-partitions and keep track of offsets for each param
            # partition_size = len(self.fp16_groups_flat[i]) / dist.get_world_size(group=self.dp_process_group)
            params_in_rank_sub_partition, params_in_rank_sub_partitions_offsets, params_not_local = self.get_all_sub_partition_info(
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
                self.loss_scaler = DynamicLossScaler(**dynamic_loss_args)

            self.dynamic_loss_scale = True

        else:
            self.dynamic_loss_scale = False
            self.loss_scaler = LossScaler(scale=static_loss_scale)
            self.cur_iter = 0

        self.mpu = mpu
        self.clip_grad = clip_grad

        self.overflow = False
        self.overflow_checker = CheckOverflow(self.fp16_groups,
                                              mpu=self.mpu,
                                              zero_reduce_scatter=True)

    @staticmethod
    def best_max_elems_per_comm(num_elements, max_elements_per_comm, dp):
        # if we use max-elems-per-comm as is, how many comm intervals will there be
        max_comm_intervals = math.ceil(num_elements / max_elements_per_comm)
        padding_for_max_comm = (max_elements_per_comm *
                                max_comm_intervals) - num_elements

        # if we use 1 less comm interval how much extra comm padding would be required
        min_comm_intervals = num_elements // max_elements_per_comm
        if min_comm_intervals == 0:
            log_dist(f'Using default max_elements_per_comm {max_elements_per_comm}',
                     ranks=[0])
            return max_elements_per_comm

        padding_for_min_comm = math.ceil(num_elements / (dp * min_comm_intervals))

        # choose padding that uses least amount of overhead
        if padding_for_max_comm > padding_for_min_comm:
            new_max_elements_per_comm = padding_for_min_comm + max_elements_per_comm
            log_dist(
                f'Updating max_elements_per_comm from {max_elements_per_comm} -> {new_max_elements_per_comm}',
                ranks=[0])
            return new_max_elements_per_comm
        else:
            log_dist(f'Using default max_elements_per_comm {max_elements_per_comm}',
                     ranks=[0])
            return max_elements_per_comm

    @staticmethod
    def get_data_parallel_sub_partitions(tensor,
                                         max_elements_per_comm,
                                         world_size,
                                         dp_process_group=None):
        total_num_elements = tensor.numel()

        # if total elements is less than our max, revert to splitting into dp partitions
        max_elements_per_comm = min(total_num_elements, max_elements_per_comm)
        sub_partition_size = int(max_elements_per_comm // world_size)

        # Ensure partition alignment was done correctly
        num_sub_partitions = int(total_num_elements // sub_partition_size)
        assert total_num_elements % sub_partition_size == 0, "{} % {} != 0".format(total_num_elements, sub_partition_size)

        # Ensure comm interval alignment was done correctly.
        num_comm_intervals = int(num_sub_partitions // world_size)
        assert num_sub_partitions % world_size == 0, "{} % {} != 0".format(num_sub_partitions, world_size)

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
        return comm_partitions, sub_partitions, element_intervals, sub_partition_size, num_comm_intervals

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

        return params_in_rank_sub_partition, params_in_rank_sub_partitions_offsets, params_not_local

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
        for tensor_list, param_offsets in zip(comm_tensor_list, comm_param_offsets):
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

                #we need to offset to get to the right element
                if i == 0 and param_offsets[i] > 0:
                    tensor_offset = param_offsets[i]
                    num_elements = num_elements - tensor_offset

                # We don't need all elements of the tensor if this tensor is
                # larger than we have space for in our curr sub-partition
                if num_elements > (sub_partition_size - current_size):
                    num_elements = sub_partition_size - current_size

                #we need a narrow view of the tensor based on the tensor offset and number of elements that
                #we need from this tensor
                if tensor_offset > 0 or num_elements < tensor.numel():
                    flat_tensor_list.append(tensor.contiguous().view(-1).narrow(
                        0,
                        int(tensor_offset),
                        int(num_elements)).to(dtype))
                else:
                    flat_tensor_list.append(tensor.to(dtype))
                my_params.append(param)

                #remember offset into partition and #elems for this tensor
                my_offsets.append((current_size, num_elements))

                current_size = current_size + num_elements

            #this means its the last partition and does not align with the dp boundary. We need to pad before flattening
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
                    rank = dist.get_rank()
                    flat_tensor_list.append(
                        torch.zeros(int(sub_partition_size - current_size),
                                    dtype=dtype,
                                    device=tensor_list[0].device))
            # partition_params.append(my_params)  #flat_tensor_list)
            partition_params.append(flat_tensor_list)
            final_param_offsets.append(my_offsets)
            assert len(flat_tensor_list) == len(my_offsets), "{} {}".format(len(flat_tensor_list), len(my_offsets))
            flat_sub_partitions.append(self.flatten(flat_tensor_list))

        if num_comm_intervals is not None and len(
                flat_sub_partitions) < num_comm_intervals:
            # logger.info("padding w. sub partitions to ensure uniform communication")
            device = flat_sub_partitions[0].device
            for _ in range(num_comm_intervals - len(flat_sub_partitions)):
                flat_sub_partitions.append(
                    torch.zeros(int(sub_partition_size),
                                dtype=dtype,
                                device=device))
                final_param_offsets.append([(None, None)])

        if return_partition_params:
            assert len(flat_sub_partitions) == len(partition_params)
            assert len(partition_params) == len(final_param_offsets), "{} {}".format(len(partition_params), len(final_param_offsets))
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
        assert max_elements_per_comm >= dp, f"max_elements_per_comm {max_elements_per_comm} < dp {dp}"

        num_elements = sum(t.numel() for t in tensor_list)
        log_dist(
            "Total number of elements in model: {}, max elements per com: {}".format(
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
            f"sub_partition_count: {sub_partition_count}, sub_partition_size: {sub_partition_size}, padding: {padding}",
            ranks=[0])
        log_dist(
            f"number of elements with padding: {num_elements} + {padding} = {num_elements + padding}",
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
        # pass
        postscale_gradients = self.postscale_gradients
        gradient_predivide_factor = self.gradient_predivide_factor
        gradient_average = self.gradient_average

        world_size = dist.get_world_size(group=self.dp_process_group)
        local_rank = dist.get_rank(group=self.dp_process_group)


        for i, group in enumerate(self.fp16_groups):
            num_comm_intervals = self.num_comm_intervals_per_group[i]
            all_sub_partitions = []
            all_partition_params = []
            for rank in range(world_size):
                # gsp is list of partitions indexed by comm_idx
                grad_sub_partitions, partition_params= self.get_flat_sub_partitions(
                    comm_tensor_list=self.params_in_rank_sub_partitions[i][rank],
                    comm_param_offsets=self.params_in_rank_sub_partitions_offsets[i]
                    [rank],
                    dtype=self.communication_data_type,
                    default_device=self.default_device,
                    sub_partition_size=self.sub_partition_sizes[i],
                    num_comm_intervals=self.num_comm_intervals_per_group[i])
                all_sub_partitions.append(grad_sub_partitions)
                all_partition_params.append(partition_params)

                assert len(grad_sub_partitions) == num_comm_intervals

            local_comm_partitions = []
            for comm_idx in range(num_comm_intervals):
                single_comm_all_partitions = []
                for rank in range(world_size):
                    single_comm_all_partitions.append(all_sub_partitions[rank][comm_idx])
                if postscale_gradients:
                    if gradient_predivide_factor != 1.0:
                        for partition in single_comm_all_partitions:
                            partition.mul_(1. / gradient_predivide_factor)

                    for partition in single_comm_all_partitions:
                        dist.all_reduce(partition, group=self.dp_process_group)
                    if gradient_average:
                        # Only need to average our local grads in post scaling
                        if gradient_predivide_factor != world_size:
                            single_comm_all_partitions[local_rank].mul_(
                                gradient_predivide_factor / world_size)
                else:
                    for partition in single_comm_all_partitions:
                        partition.div_(world_size)

                    for partition in single_comm_all_partitions:
                        dist.all_reduce(partition, group=self.dp_process_group)

                for buf, synced in zip(all_partition_params[local_rank][comm_idx], self.unflatten(single_comm_all_partitions[local_rank] ,all_partition_params[local_rank][comm_idx])):
                        buf.copy_(synced)

    def step(self, closure=None):
        # First compute norm for all group so we know if there is overflow
        self.overflow = self.overflow_checker.check()

        prev_scale = self.loss_scale
        self._update_scale(self.overflow)
        if self.overflow:
            self.zero_grad()
            if self.verbose:
                logger.info(
                    "[deepspeed] fp16 dynamic loss scale overflow! Skipping step. Attempted loss "
                    "scale: {}, reducing to {}".format(prev_scale,
                                                       self.loss_scale))
            return self.overflow

        norm_groups = []
        local_sub_partitions_grad_groups = []
        partition_id = dist.get_rank(group=self.dp_process_group)
        world_size = dist.get_world_size(group=self.dp_process_group)
        for i, group in enumerate(self.fp16_groups):
            #TODO RS: update get grad norm to support sub partitions
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

            norm_groups.append(get_grad_norm(local_grad_sub_partitions, partition_params, mpu=mpu))

            #RS: update all our local params with sub-partition grads
            for idx, sub_partition_param in enumerate(self.local_sub_partitions_of_fp32_groups[i]):
                sub_partition_param.grad = local_grad_sub_partitions[idx]

            #RS: update free grads for sub-partitions
            #release all the gradient since we have already created a necessary copy in dp_grad_partition
            self.free_grad_in_param_list(
                self.params_in_rank_sub_partitions[i][partition_id])

            local_sub_partitions_grad_groups.append(local_grad_sub_partitions)

        #RS: update unscale/clip with sub partitions

        self.unscale_and_clip_grads(local_sub_partitions_grad_groups, norm_groups)

        self.optimizer.step()

        #RS: clear our sub partition grads
        #get rid of the fp32 gradients. Not needed anymore
        for group in self.local_sub_partitions_of_fp32_groups:
            for idx, sub_partition_param in enumerate(group):
                sub_partition_param.grad = None
            # group.grad = None

        #NOTE RS: removed norm_groups outer loop from original code, i don't think it's needed
        #RS: copy all sub-partition fp32 data to fp16 sub partitions
        # copy fp32 param data to fp16 partitions w.r.t. our local rank
        for fp16_all_sub_partitions, fp32_local_sub_partitions in zip(self.parallel_sub_partitioned_fp16_groups, self.local_sub_partitions_of_fp32_groups):
            for local_sub_partition_param_fp16, local_sub_partition_param_fp32 in zip(fp16_all_sub_partitions[partition_id], fp32_local_sub_partitions):
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

    # # Promote state so it can be retrieved or set via "fp16_optimizer_instance.state"
    def _get_state(self):
        return self.optimizer.state

    def _set_state(self, value):
        self.optimizer.state = value

    state = property(_get_state, _set_state)

    # Promote param_groups so it can be retrieved or set via "fp16_optimizer_instance.param_groups"
    # (for example, to adjust the learning rate)
    def _get_param_groups(self):
        return self.optimizer.param_groups

    def _set_param_groups(self, value):
        self.optimizer.param_groups = value

    param_groups = property(_get_param_groups, _set_param_groups)

    # Promote loss scale so it can be retrieved or set via "fp16_optimizer_instance.loss_scale"
    def _get_loss_scale(self):
        return self.loss_scaler.loss_scale

    def _set_loss_scale(self, value):
        self.loss_scaler.cur_scale = value

    loss_scale = property(_get_loss_scale, _set_loss_scale)
    cur_scale = property(_get_loss_scale, _set_loss_scale)

    def _rigid_state_dict(self):
        """
            Returns a dict that can be loaded for continued training with same DP degree
        """
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
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
            'local_sub_partitions_of_fp32_groups'] = self.local_sub_partitions_of_fp32_groups
        return state_dict


    def state_dict(self):
        """
        Returns a dict containing the current state of this :class:`FP16_Optimizer` instance.
        This dict contains attributes of :class:`FP16_Optimizer`, as well as the state_dict
        of the contained Pytorch optimizer.
        """

        return self._rigid_state_dict()

    def _rigid_load_state_dict(self, state_dict, load_optimizer_states=True):

        # I think it should actually be ok to reload the optimizer before the model.
        self.loss_scaler = state_dict['loss_scaler']
        self.dynamic_loss_scale = state_dict['dynamic_loss_scale']
        self.overflow = state_dict['overflow']
        if load_optimizer_states:
            self.optimizer.load_state_dict(state_dict['base_optimizer_state'])

        for curr_group, saved_group in zip(self.local_sub_partitions_of_fp32_groups, state_dict['local_sub_partitions_of_fp32_groups']):
            for curr_param, saved_param in zip(curr_group, saved_group):
                curr_param.data.copy_(saved_param.data)

    def load_state_dict(self,
                        state_dict,
                        load_optimizer_states=True,
                        load_from_fp32_weights=False):
        """
        Loads a state_dict created by an earlier call to state_dict().
        If ``fp16_optimizer_instance`` was constructed from some ``init_optimizer``,
        whose parameters in turn came from ``model``, it is expected that the user
        will call ``model.load_state_dict()`` before
        ``fp16_optimizer_instance.load_state_dict()`` is called.
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
