import os
import time
from collections import OrderedDict
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from Merak import get_logger

from .. import mpu
from .checkpoint import CheckpointSaver, CheckpointLoader
from .redundancy_checker import RedundancyChecker


from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
import pickle
import datetime

# Build communication groups for remote checkpointing
def build_comm_groups_for_remote_checkpointing(k=2):
    """
    Build communication groups following these rules:
    1. Group corresponding local ranks (0-3) across consecutive k nodes
    2. Remainder nodes (n_nodes % k) are added to the last full k-node group
    3. Each local rank (0-3) has independent groups (node0-rank0 with node1-rank0, etc.)

    Args:
        k: Number of consecutive nodes per full group

    Returns:
        Process group for current local rank and global ranks of the process group.
    """
    logger = get_logger("simple")
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    assert world_size % 4 == 0, "World size must be multiple of 4"
    n_nodes = world_size // 4
    node_id = global_rank // 4
    local_rank = global_rank % 4

    n_full_groups = n_nodes // k
    remainder_nodes = n_nodes % k
    
    all_group_global_ranks = []

    # Get all remote checkpointing group ranks.
    for group_idx in range(n_full_groups):
        node_ids = [i for i in range(group_idx * k, (group_idx + 1) * k)]
        if group_idx == n_full_groups - 1 and remainder_nodes > 0:
            for i in range(remainder_nodes):
                node_ids.append(n_full_groups * k + i)
        for i in range(4):
            group_global_ranks = []
            for node_id in node_ids:
                group_global_ranks.append(node_id * 4 + i)
            all_group_global_ranks.append(group_global_ranks)

    # CRITICAL: All processes that belong to this group MUST call new_group
    # Since group_global_ranks is built deterministically, all members will call it
    try:
        for group_global_ranks in all_group_global_ranks:
            cur_comm_group = dist.new_group(ranks=group_global_ranks, backend='gloo', timeout=datetime.timedelta(seconds=60))
            if global_rank in group_global_ranks:
                ret_comm_group = cur_comm_group
                ret_group_global_ranks = group_global_ranks
            logger.info(
            f"[Node {global_rank // 4}] - Local Rank {local_rank} - Build Comm Group with Global Ranks: {group_global_ranks}"
            )
    except Exception as e:
        logger.error(f"Failed to create group for [Node {node_id}] - local_rank={local_rank}, ranks={group_global_ranks}: {e}")
        raise

    return ret_comm_group, ret_group_global_ranks


# Get dictionaries for remote checkpointing.
def get_dicts_for_remote_checkpointing(transmission_time_dict, local_checkpoint, comm_group_for_rcheck, comm_group_global_ranks_for_rcheck):
    '''
        Get partition dictionaries for remote checkpointing.
    '''
    logger = get_logger("simple")

    # split transferred global_ckpt into a list according to profile result
    # vars for remote transmission
    group_for_remote_ckpt = comm_group_for_rcheck
    rank_in_group = torch.distributed.get_rank(group=group_for_remote_ckpt)
    group_size = torch.distributed.get_world_size(group=group_for_remote_ckpt)

    src_of_remote_ckpt = comm_group_global_ranks_for_rcheck[(rank_in_group - 1) % group_size]
    dst_of_remote_ckpt = comm_group_global_ranks_for_rcheck[(rank_in_group + 1) % group_size]


    # record the meta data for each parameter tensor in terms of (name, index, size), done only once thanks to the consistency
    param_meta_data = []
    global_ckpt_size = 0

    # since dict is ordered if python version >= 3.7, no need to record name and shape for weights,
    # while extra states are all empty bytes or None.
    # we can get necessary information from healthy ranks for recovery.
    for k, v in local_checkpoint.items():
        if isinstance(v, torch.Tensor):
            param_meta_data.append((k, v.numel()))
            global_ckpt_size += v.numel()
        else:
            logger.error(f"In the local checkpoint, the value of key {k} is not a tensor.")
            raise 

    # pre-alloc cpu buffer for global_ckpt and remote_ckpt, both on CPU
    remote_ckpt_dict = {}
    remote_ckpt_dict.update({f'rank{src_of_remote_ckpt}': torch.zeros(global_ckpt_size, dtype=torch.float16, device='cpu')})
    cpu_global_ckpt = torch.zeros(global_ckpt_size, dtype=torch.float16, device='cpu', pin_memory=True)
    logger.info(f"checkpoint size: {round(global_ckpt_size*2/1024**2, 3)} MB")

                
    # update splitting information to be used until next profiling results are available
    # -> partitions format: {stage_name : length}
    # -> index_dict format: {stage_name : (start_index, end_index)}
    # -> snapshot_chunks_dict format: {stage_name : [(name, start_offset, end_offset)]}
    def scale_dict_to_target_sum(int_dict, target_sum, alpha=0):
        """ adjust the sum of the input int dict to a given value, take threshold into account. """

        if alpha!=0:
            for k, v in int_dict.items():
                int_dict[k] = max(0, v - alpha)

        current_sum = sum(int_dict.values())
        if current_sum == target_sum:
            return int_dict

        scale_factor = target_sum / current_sum
        scaled_dict = {k:round(v * scale_factor) for k,v in int_dict.items()}
        scaled_sum = sum(scaled_dict.values())
        if scaled_sum != target_sum:
            for k, v in scaled_dict.items():
                if v + target_sum - scaled_sum > 0:
                    scaled_dict[k] += target_sum - scaled_sum
                    break

        return scaled_dict
    
    partitions = scale_dict_to_target_sum(transmission_time_dict.copy(), global_ckpt_size)


    def revert_to_chunk_info_dict(meta_data, length_dict):
        """
            Args:
                meta_data: a list of [(name, numel)] representing the meta_data of each paramter tensor.
                length_dict: a dict of {stage: length} representing the length of checkpoint chunk for each stage.
            Returns:
                index_dict: a dict of {stage_name : (start_index, end_index)} representing the index ranges of the merged tensor for each stage. 
                chunks_dict: a dict of {stage: [(name, start_offset, end_offset)]} representing the set consist of index ranges of tensors constituting a chunk for each stage.

        """
        sum = 0
        id = 0          # id of current tensor
        offset = 0      # offset of current tensor
        index_dict = {}
        chunks_dict = {}
        for stage, length in length_dict.items():
            index_dict[stage] = (sum, sum + length)
            sum += length
            chunks_dict[stage] = []
            curr_length = 0
            while curr_length < length:
                curr_tensor_meta = meta_data[id]    # (name, id, numel)
                if curr_length + curr_tensor_meta[1] - offset <= length:
                    chunks_dict[stage].append((curr_tensor_meta[0], offset, curr_tensor_meta[1]))
                    curr_length += curr_tensor_meta[1] - offset
                    id += 1
                    offset = 0
                else:
                    chunks_dict[stage].append((curr_tensor_meta[0], offset, offset + length - curr_length))
                    offset += length - curr_length
                    break
        assert id == len(meta_data)
        return index_dict, chunks_dict

    index_dict = {}
    snapshot_chunks_dict = {}
    d1,d2 = revert_to_chunk_info_dict(param_meta_data, partitions)
    index_dict.update(d1)
    snapshot_chunks_dict.update(d2)

    # send/recv index_dict to/from remote node
    # -> remote_index_dict format: {stage_name : (start_index, end_index)}
    # p2p-comm for objects is not supported for torch<=2.2.0, thus replaced with broadcast.
    # torch.distributed.all_gather_object(object_list, obj, group=None)
    remote_index_dict = {}
    objects = [index_dict] if rank_in_group == 0 else [None]
    dist.broadcast_object_list(objects, src=dist.get_global_rank(group=group_for_remote_ckpt, group_rank=0), group=group_for_remote_ckpt)
    if rank_in_group == 1:
        remote_index_dict.update(objects[0])

    objects = [index_dict] if rank_in_group == 1 else [None]
    dist.broadcast_object_list(objects, \
        src=dist.get_global_rank(group=group_for_remote_ckpt, group_rank=1), group=group_for_remote_ckpt)
    if rank_in_group == 0:
        remote_index_dict.update(objects[0])

    logger.info(f'Dicts returned for remote checkpointing, param_meta_data:{param_meta_data}, remote_ckpt_dict:{remote_ckpt_dict}, index_dict:{index_dict}, snapshot_chunks_dict:{snapshot_chunks_dict}, remote_index_dict:{remote_index_dict}')

    return param_meta_data, remote_ckpt_dict, cpu_global_ckpt, index_dict, snapshot_chunks_dict, remote_index_dict

def logging_dict(state_dict, pre_fix=""):
    """
    Recursively logs information about a dictionary's contents, including:
    - Gradients (shape and dtype if present)
    - Shapes of tensors/arrays
    - Types of other values

    Handles nested dictionaries, lists, and tuples by recursively processing them.
    Designed for logging PyTorch model states but works with any dictionary-like object.

    Args:
        state_dict (dict): The dictionary to log information about
        pre_fix (str): Optional prefix string for log messages (used internally for recursion)

    Returns:
        None: Only produces log output
    """
    logger = get_logger("simple")

    for key, value in state_dict.items():
        current_prefix = f"{key} {pre_fix}".strip()

        # Handle objects with gradients (PyTorch tensors)
        if hasattr(value, "grad"):
            if value.grad is not None:
                logger.info(
                    f"{current_prefix} has grad {value.grad.shape} with {value.grad.dtype}"
                )
            else:
                logger.info(f"{current_prefix} has grad none but shape {value.shape}")

        # Handle nested dictionaries
        elif isinstance(value, dict):
            logging_dict(value, f"of {current_prefix}")

        # Handle sequences (lists/tuples)
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                item_prefix = f"{idx}th of {current_prefix}"

                if hasattr(item, "grad"):
                    if item.grad is not None:
                        logger.info(
                            f"{item_prefix} has grad {item.grad.shape} with {item.grad.dtype}"
                        )
                    else:
                        logger.info(
                            f"{item_prefix} has grad none but shape {item.shape}"
                        )
                elif isinstance(item, dict):
                    logging_dict(item, f"of {item_prefix}")
                elif hasattr(item, "shape"):
                    logger.info(f"{item_prefix} no grad but shape {item.shape}")
                else:
                    logger.info(f"{item_prefix} no grad no shape {type(item)}")

        # Handle other objects with shape attribute
        elif hasattr(value, "shape"):
            logger.info(f"{current_prefix} no grad but shape {value.shape}")

        # Fallback for other types
        else:
            logger.info(f"{current_prefix} no grad no shape {type(value)}")


def extract_grad_and_const(state_dict, grad_const_dict, exclude_list=None):
    """
    Recursively extracts gradients and constant values from a PyTorch state dictionary.

    Processes nested dictionaries, lists, and tuples to extract:
    - Gradient tensors (when available)
    - Constant tensors (when no gradient exists)
    - Preserves excluded keys without modification

    Args:
        state_dict (dict): The input dictionary containing model parameters/state
        grad_const_dict (dict): Output dictionary to store extracted gradients/constants
        exclude_list (list, optional): List of keys to preserve as-is without processing

    Returns:
        None: Modifies grad_const_dict in-place with extracted values

    Note:
        - All extracted tensors are cloned and detached for safety
        - Handles nested structures recursively
        - Preserves original values for excluded keys
    """
    if exclude_list is None:
        exclude_list = []

    for key, value in state_dict.items():
        if key != 'models':
            grad_const_dict[key] = value
            continue

        # Handle gradient tensors
        if hasattr(value, "grad") and value.grad is not None:
            grad_const_dict[key] = value.grad.data.clone().detach()

        # Handle nested dictionaries
        elif isinstance(value, dict):
            grad_const_dict[key] = {}
            extract_grad_and_const(value, grad_const_dict[key], exclude_list)

        # Handle sequences (lists/tuples)
        elif isinstance(value, (list, tuple)):
            grad_const_dict[key] = []
            for item in value:
                if hasattr(item, "grad") and item.grad is not None:
                    grad_const_dict[key].append(item.grad.data.clone().detach())
                elif isinstance(item, dict):
                    new_dict = {}
                    extract_grad_and_const(item, new_dict, exclude_list)
                    grad_const_dict[key].append(new_dict)
                elif not isinstance(item, torch.Tensor):
                    grad_const_dict[key].append(item.clone().detach())
                else:
                    grad_const_dict[key].append(item)

        # Handle regular tensors
        elif not isinstance(value, torch.Tensor):
            grad_const_dict[key] = value.clone().detach()

        # Preserve other types
        else:
            grad_const_dict[key] = value


def clean_grad(state_dict):
    """
    Recursively cleans gradients from a PyTorch state dictionary in-place.

    Processes nested dictionaries, lists, and tuples to:
    - Remove gradient references from Parameters and Tensors
    - Ensure all tensors are cloned and detached
    - Preserve the original structure of the state_dict

    Args:
        state_dict (dict): The input dictionary containing model parameters/state
                          (modified in-place)

    Returns:
        None: Modifies the input state_dict directly

    Note:
        - Handles both Parameters and regular Tensors
        - Processes nested structures recursively
        - All operations are performed in-place
        - Ensures memory safety by cloning and detaching tensors
    """
    for key, value in state_dict.items():
        # Handle Parameters and Tensors with gradients
        if isinstance(value, torch.nn.parameter.Parameter):
            state_dict[key] = value.clone().detach()

        # Handle nested dictionaries
        elif isinstance(value, dict):
            clean_grad(value)

        # Handle sequences (lists/tuples)
        elif isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                if isinstance(item, torch.nn.parameter.Parameter):
                    value[i] = item.clone().detach()
                elif isinstance(item, dict):
                    clean_grad(item)


class RcharaCheckpointSaver(CheckpointSaver):
    """
    Redundancy-aware Checkpoint Saver with differential saving capability

    Extends base CheckpointSaver to provide advanced checkpointing features including
    differential saving, redundancy analysis, and Zero-Redundancy Optimizer support.
    """

    def __init__(self):
        super().__init__()
        self.logger = get_logger("simple")
        self.ablation_study = False # setting to ablation_study
        self.checkpoint_version = 1
        self.redundancy_checker = None

        # Variables for remote checkpointing
        self.rcheckpoint_comm_group = None
        self.rcheckpoint_group_global_ranks = None

    def init_redundancy_checker(self, zero_state_dict):
        self.redundancy_checker = RedundancyChecker(
            self.model,
            self.optimizer,
            self.args.zero_stage,
            zero_state_dict,
        )

    def _save_state_dict(
            self,
            state_dict: Dict,
            checkpoint_name: str,
            zero_ckpt_name: Optional[str] = None,
            peft_state_dict: Optional[Dict] = None,
            zero_state_dict: Optional[Dict] = None,
            exclude_list: Optional[List] = None,
            **kwargs,
        ):
        """
        Save the state dictionary to the specified path.

        Args:
            state_dict: State dictionary to save.
            checkpoint_name: Path to save the main checkpoint.
            zero_ckpt_name: Path for saving Zero-Redundancy Optimizer states.
            peft_state_dict: LoRA state dictionary to save if applicable.
            zero_state_dict: Zero Optimizer state dictionary.
            **kwargs: Additional keyword arguments.
        """
        if self.iteration > max(self.args.check_steps):
            self.args.use_differential = True

        if self.args.use_differential:
            self.checkpoint_version = self.iteration

        if self.redundancy_checker is None:
            self.init_redundancy_checker(zero_state_dict)

        if self.iteration in self.args.check_steps:
            self.redundancy_checker.add_zero_state(zero_state_dict)

        if exclude_list is None:
            exclude_list = ["rng_state"]

        if mpu.get_data_parallel_rank() == 0:
            if self.iteration > max(self.args.check_steps) and \
                self.checkpoint_version % self.args.differential_interval != 0:
                self.redundancy_checker.get_dict_for_gradient_checkpointing()
                state_dict = self.redundancy_checker.parameter_dict_for_gradient_checkpointing
                # Debugging
                self.logger.debug(f"Keys for parameter_dict_for_gradient_checkpointing:{state_dict.keys()}")
            # else:
            #     if self.iteration in self.args.check_steps:
            #         tmp = state_dict['optimizer'].pop('fp16_groups')

            if self.args.use_differential:
                if self.iteration % self.args.differential_interval == 0:
                    # checkpoint_name = checkpoint_name.replace(
                    #     "iter_", "full_iter_"
                    # )
                    self.logger.info(f"[Rchara] Saving full {checkpoint_name}...")
                    clean_grad(state_dict)
                    # log_dist(f'state_dict:{state_dict}', ranks=[0])
                    self.logger.debug(f"checkpoint_name:{checkpoint_name}, keys of state dict after clean_grad:{state_dict.keys()}")
                    torch.save(
                        state_dict,
                        checkpoint_name,
                        _use_new_zipfile_serialization=False,
                    )
                    # logger.info(f"{path}...")
                    # logging_dict(state_dict=state_dict)
                else:
                    grad_const_dict = {}
                    extract_grad_and_const(state_dict, grad_const_dict, exclude_list)
                    self.logger.info(f"[Rchara] Saving grad {checkpoint_name}...")
                    self.logger.debug(f"checkpoint_name:{checkpoint_name}, keys of state dict after extract_grad_and_const:{state_dict.keys()}")
                    torch.save(
                        grad_const_dict,
                        checkpoint_name,
                        _use_new_zipfile_serialization=False,
                    )
                    # logger.info(f"{path}...")
                    # logging_dict(state_dict=grad_const_dict)
            else:
                self.logger.debug(f"checkpoint_name:{checkpoint_name}, keys of state dict when use_differential is false:{state_dict.keys()}")
                torch.save(
                    state_dict, checkpoint_name, _use_new_zipfile_serialization=False
                )

        # Save Zero Optimizer states if required
        if self.args.zero_stage == 1:
            dist.barrier()
            torch.save(zero_state_dict, zero_ckpt_name)
            self.logger.info(
                f"zero checkpoint saved {zero_ckpt_name}", ranks=[dist.get_rank()]
            )

        dist.barrier()  # Ensure all processes are synchronized

        # Update the latest iteration tracker
        self._save_tracker_file()
        self.logger.info(f"[Rchara] Saved at iteration {self.iteration} to {self.save_path}.", ranks=[0])

        # redundancy check
        if self.iteration in self.args.check_steps:
            self.redundancy_checker.add_model_state(state_dict['model'])
            self.redundancy_checker.add_optimizer_state(state_dict['optimizer'] 
                                                  if zero_state_dict is None else None)
            self.redundancy_check()

        # update checkpoint version
        if self.args.use_differential:
            self.checkpoint_version = self.iteration

    def redundancy_check(self):
        """
        Perform redundancy analysis on model parameters at specified check steps.

        When ablation_study is disabled, runs the standard redundancy checking pipeline.
        When ablation_study is enabled, benchmarks different redundancy checking 
        configurations and algorithms for performance comparison.
        """
        if not self.ablation_study:
            self.redundancy_checker.build_name_value_mapping()
            self.redundancy_checker.get_hash_values()
            self.redundancy_checker.pack_to_tensor()
            self.redundancy_checker.do_redundancy_checking()
            self.redundancy_checker.get_names_of_parameters_requiring_gradient_checkpointing()
            self.logger.info("[Rchara] Redundancy check finished." )

            # Build communication group for remote checkpointing.
            self.rcheckpoint_comm_group, self.rcheckpoint_group_global_ranks = build_comm_groups_for_remote_checkpointing()

        else:
            self.redundancy_checker.build_name_value_mapping()
            self.redundancy_checker.get_hash_values()
            self.redundancy_checker.pack_to_tensor()
            start = time.time()
            self.redundancy_checker.do_redundancy_checking()
            end = time.time()
            self.logger.info(f'Redundancy checking time is {end - start} s', ranks=[0])

            self.redundancy_checker.no_redundant_cal = False
            start = time.time()
            self.redundancy_checker.do_redundancy_checking()
            end = time.time()
            self.logger.info(f'Redundancy checking without no_redundant_cal time is {end - start} s', ranks=[0])
            self.redundancy_checker.no_redundant_cal = True

            self.redundancy_checker.comm_groups_for_redundancy_checking = False
            start = time.time()
            self.redundancy_checker.do_redundancy_checking()
            end = time.time()
            self.logger.info(
                f'Redundancy checking without comm_groups_for_redundancy_checking cal time is {end - start} s',
                ranks=[0]
            )
            self.redundancy_checker.comm_groups_for_redundancy_checking = True

            self.redundancy_checker.redundancy_cache = False
            start = time.time()
            self.redundancy_checker.do_redundancy_checking()
            end = time.time()
            self.logger.info(f'Redundancy checking without redundancy_cache time is {end - start} s', ranks=[0])
            self.redundancy_checker.redundancy_cache = True

            self.redundancy_checker.get_baseline_packked_tensor()
            start = time.time()
            self.redundancy_checker.baseline_compare()
            end = time.time()
            self.logger.info(f'Baseline redundancy checking time is {end - start} s', ranks=[0])     

class RcharaCheckpointLoader(CheckpointLoader):
    pass
