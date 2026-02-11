import argparse
import json
from collections import defaultdict
from hashlib import blake2s

import numpy as np
import torch
import torch.distributed as dist

from Merak import get_logger

from .. import mpu
from ..zero import ZeroOptimizer_Stage1
from .name_value_mapping import NVMapping

class RedundancyChecker:
    def __init__(
        self, model=None, optimizer=None, zero_stage=None, zero_state=None
    ):        
        """
        Check redundancy in model parameters and optimizer states for memory optimization.

        Analyzes parameter redundancy across different states (model, optimizer, zero) 
        to identify opportunities for gradient checkpointing and memory reduction.

        Args:
            optimizer: Model optimizer instance
            model: The model instance being checked for redundancy
            zero_stage: Current stage of Zero Redundancy Optimizer
            zero_state: Zero Optimizer state dictionary if using ZeRO
        """
        self.logger = get_logger("simple")
        self.model = model
        self.optimizer = optimizer
        self.optimizer_state = None
        self.model_state = None
        self.zero_stage = zero_stage
        self.zero_state = zero_state

        # Variables for hash values.
        self.hash_val_for_optimizer = None
        self.hash_val_for_model_state = None
        self.hash_val_for_zero_state = None

        # Variables for index mapping.
        self.data_ranks = []
        self.packed_tensor = []
        self.val_index_dict = {}
        self.optimizer_index_mapping = None
        self.model_state_index_mapping = None
        self.zero_state_index_mapping = None

        # Viriables for redundancy checking.
        self.redundancy_results = {}
        self.redundancy_results_for_optimizer = None
        self.redundancy_results_for_model_state = None
        self.redundancy_results_for_zero_state = None
        self.nvmapping = NVMapping()
        self.comm_groups_for_redundancy_checking = True
        self.no_redundant_cal = False
        self.redundancy_cache = False
        self.k = 2

        # Viriables for gradient checkpointing
        self.parameter_names_for_gradient_checkpointing = None
        self.parameter_dict_for_gradient_checkpointing = None

        # Viriables for name index range mapping of sub tensors.
        self.sub_tensors_name_range_mapping = {}

    def add_zero_state(self, zero_state):
        '''
            Store ZeRO optimizer state during checkpoint saving.

            Args:
                zero_state: ZeRO optimizer states.
        '''
        self.zero_state = zero_state

    def add_optimizer_state(self, optimizer_state):
        '''
            Store optimizer state during checkpoint saving.

            Args:
                optimizer_state: optimizer states.
        '''
        # self.logger.info('Optimizer state has been added to redundancy checker.')
        self.optimizer_state = optimizer_state

    def add_model_state(self, state):
        '''
            Store model state during checkpoint saving. 

            Args:
                state: model states.
        '''
        # self.logger.info('Model state has been added to redundancy checker.')
        self.model_state = state

    def log_dicts(self):
        """
        Logging experts_state_dicts, optimizer_state, and model_state.
        Optimizer state will be logged when exp_dp_rank == 0.
        Model state will be logged when mpu._get_data_parallel_rank() == 0.
        """
        my_rank = dist.get_rank()

        if mpu.get_data_parallel_rank() == 0:
            for k, v in self.model_state.items():
                self.logger.info(f"{k}:{v}\n", ranks=[my_rank])

    def _get_hash_values_for_single_dict(self, original, is_optimizer=False):
        """
        Get hash values for orginal.

        Args:
            original: The dictionary, list, or torch.Tensor that supplys
            the keys and their corresponding values.
        """
        ret = None
        if isinstance(original, (dict, defaultdict)):
            ret = {}
            for key, val in original.items():
                ret[key] = self._get_hash_values_for_single_dict(val, is_optimizer)
        elif isinstance(original, list):
            ret = []
            for val in original:
                ret.append(self._get_hash_values_for_single_dict(val, is_optimizer))
        elif isinstance(original, torch.Tensor):
            val_cpu = original.cpu()
            array = val_cpu.detach().numpy()
            if is_optimizer:
                hash_value = blake2s(
                    array.tobytes() + "optimizer".encode(), digest_size=7
                ).hexdigest()
            else:
                hash_value = blake2s(array.tobytes(), digest_size=7).hexdigest()
            ret = hash_value
        elif isinstance(original, set):
            my_frozenset = frozenset(original)
            json_str = json.dumps(sorted(list(my_frozenset)))
            byte_str = json_str.encode()
            # ret = sha256(byte_str).digest()
            if is_optimizer:
                ret = blake2s(
                    byte_str + "optimizer".encode(), digest_size=7
                ).hexdigest()
            else:
                ret = blake2s(byte_str, digest_size=7).hexdigest()
        elif isinstance(original, tuple):
            ret = []
            for val in original:
                ret.append(self._get_hash_values_for_single_dict(val))
            # return a hash value for a tuple since it only contains numerical values.
            tuple_bytes = "".join(str(byte_str) for byte_str in ret).encode()
            # ret = sha256(tuple_bytes).digest()
            ret = blake2s(tuple_bytes, digest_size=7).hexdigest()
        elif isinstance(original, argparse.Namespace):
            args_dict = vars(original)
            sorted_args = sorted(args_dict.items())
            sorted_args_str = str(dict(sorted_args))
            # ret = sha256(sorted_args_str.encode()).digest()
            ret = blake2s(sorted_args_str.encode(), digest_size=7).hexdigest()
        elif isinstance(original, np.ndarray):
            bytes_representation = original.tobytes()
            # ret = sha256(bytes_representation).digest()
            if is_optimizer:
                ret = blake2s(
                    bytes_representation + "optimizer".encode(), digest_size=7
                ).hexdigest()
            else:
                ret = blake2s(bytes_representation, digest_size=7).hexdigest()
        else:
            # ret = sha256(str(original).encode()).digest()
            if is_optimizer:
                ret = blake2s(
                    str(original).encode() + "optimizer".encode(), digest_size=7
                ).hexdigest()
            else:
                ret = blake2s(str(original).encode(), digest_size=7).hexdigest()
        return ret

    def get_hash_values(self):
        """
        Get hash values for experts state dicts, optimizer states,
        and model states.
        """
        self.hash_val_for_model_statel_state = self._get_hash_values_for_single_dict(
            self.model_state
        )

        # self.logger.info(
        #     f'self.zero_stage:{self.zero_stage}, '
        #     f'type is {type(self.zero_stage)}',
        #     ranks = [dist.get_rank()]
        # )

        if self.zero_stage == 1:
            self.hash_val_for_zero_state = {}
            params_in_partition = self.zero_state["optimizer_state_dict"][
                "params_in_partition"
            ]
            first_offsets = self.zero_state["optimizer_state_dict"]["first_offset"]
            partition_sizes = self.zero_state["optimizer_state_dict"]["partition_size"]

            for i, params in enumerate(params_in_partition):
                sub_params_in_partition, sub_params_name_list = (
                    self.get_sub_tensors_in_partition(
                        params, first_offsets[i], partition_sizes[i]
                    )
                )
                for j, param in enumerate(params):
                    val_cpu = sub_params_in_partition[j].cpu()
                    array = val_cpu.detach().numpy()
                    hash_value = blake2s(
                        array.tobytes() + "optimizer".encode(), digest_size=7
                    ).hexdigest()
                    name = sub_params_name_list[j]
                    self.hash_val_for_zero_state[name] = hash_value
        else:
            self.hash_val_for_optimizer = self._get_hash_values_for_single_dict(
                self.optimizer_state, is_optimizer=True
            )

    def log_hash_dicts(self):
        """
        Logging hash_val_for_experts, hash_val_for_optimizer,
        and hash_val_for_model_statel_state.
        hash_val_for_optimizer will be logged when exp_dp_rank == 0.
        hash_val_for_model_statel_state will be logged
        when mpu._get_data_parallel_rank() == 0.
        """
        my_rank = dist.get_rank()

        if self.zero_stage != 1:
            for k, v in self.hash_val_for_optimizer.items():
                self.logger.info(f"{k}:{v}\n", ranks=[my_rank])

        if mpu.get_data_parallel_rank() == 0:
            # self.logger.info(f"Log model states hash dict, expp_rank:{expp_rank},
            # exp_dp_rank:{exp_dp_rank}, dp_rank:{dp_rank}", ranks=[my_rank])
            for k, v in self.hash_val_for_model_statel_state.items():
                self.logger.info(f"{k}:{v}\n", ranks=[my_rank])

    def _pack_single_to_tensor(self, original):
        """
        Pack values in original into a tensor for later communication
        and comparison, and keep the index mapping in corresponding variables.

        Args:
            original: Dictionary, list, None, or hash values to pack.
        """
        ret = None
        if isinstance(original, (dict, defaultdict)):
            ret = {}
            for key, val in original.items():
                ret[key] = self._pack_single_to_tensor(val)
        elif isinstance(original, list):
            ret = []
            for val in original:
                ret.append(self._pack_single_to_tensor(val))
        elif isinstance(original, tuple):
            ret = []
            for val in original:
                ret.append(self._pack_single_to_tensor(val))
            ret = tuple(ret)
        else:
            if original in self.val_index_dict:
                ret = self.val_index_dict[original]
            else:
                ret = len(self.packed_tensor)
                self.packed_tensor.append(int(original, 16))
                self.val_index_dict[original] = ret
        return ret

    def pack_to_tensor(self):
        """
        pack values for experts state hash dict, optimizer states
        hash dict, and model states hash dict.
        """
        self.packed_tensor = []
        self.val_index_dict = {}

        self.model_state_index_mapping = self._pack_single_to_tensor(
            self.hash_val_for_model_statel_state
        )

        my_rank = dist.get_rank()
        if self.zero_stage == 1:
            self.zero_state_index_mapping = {}
            for k, v in self.hash_val_for_zero_state.items():
                if v in self.val_index_dict:
                    self.zero_state_index_mapping[k] = self.val_index_dict[v]
                    self.logger.info(
                        f"Hash value of {k} already exist in the packed tensor.",
                        ranks=[my_rank],
                    )
                else:
                    index = len(self.packed_tensor)
                    self.packed_tensor.append(int(v, 16))
                    self.val_index_dict[v] = index
                    self.zero_state_index_mapping[k] = index
        else:
            self.optimizer_index_mapping = self._pack_single_to_tensor(
                self.hash_val_for_optimizer
            )

            # self.logger.info(
            #     f'self.zero_state_index_mapping.keys:'
            #     f'{self.zero_state_index_mapping.keys()}',
            #     ranks=[my_rank]
            # )

        self.packed_tensor = torch.tensor(self.packed_tensor, dtype=torch.int64)

    def _get_data_parallel_group_ranks(self):
        """
        Get data parallel groups for AllReduce.
        """
        tensor_model_parallel_size = mpu.get_model_parallel_world_size()
        pipeline_model_parallel_size = mpu.get_pipe_parallel_world_size()
        sequence_parallel_size = mpu.get_sequence_parallel_world_size()

        rank = torch.distributed.get_rank()

        world_size: int = torch.distributed.get_world_size()

        if (
            world_size % (tensor_model_parallel_size * pipeline_model_parallel_size)
            != 0
        ):
            raise RuntimeError(
                f"world_size ({world_size}) is not divisible by tensor_model_parallel_size "
                f"({tensor_model_parallel_size}) x pipeline_model_parallel_size "
                f"({pipeline_model_parallel_size})"
            )

        enable_sequence_parallel = sequence_parallel_size > 1

        data_parallel_size: int = world_size // (
            tensor_model_parallel_size
            * pipeline_model_parallel_size
            * sequence_parallel_size
        )
        sequence_data_parallel_size: int = sequence_parallel_size * data_parallel_size

        num_pipeline_model_parallel_groups: int = (
            world_size // pipeline_model_parallel_size
        )
        num_sequence_data_parallel_groups: int = (
            world_size // sequence_parallel_size // data_parallel_size
        )

        # Build the data-parallel groups.
        for i in range(pipeline_model_parallel_size):
            start_rank = i * num_pipeline_model_parallel_groups
            end_rank = (i + 1) * num_pipeline_model_parallel_groups

            if sequence_parallel_size > 1:
                tp_or_sp_size = sequence_parallel_size
            else:
                tp_or_sp_size = tensor_model_parallel_size

            for j in range(tp_or_sp_size):
                ranks = range(start_rank + j, end_rank, tp_or_sp_size)
                if rank in ranks:
                    self.data_ranks = ranks

        # Build the sequence data parallel groups.
        if enable_sequence_parallel:
            for i in range(num_sequence_data_parallel_groups):
                ranks = range(
                    i * sequence_data_parallel_size,
                    (i + 1) * sequence_data_parallel_size,
                )
                if rank in ranks:
                    self.data_ranks = ranks

    def _convert_index_to_redundancy_results(self, original, redundancy_results):
        """
        Convert index mapping to redundancy results while keeping
        the format of orginal.

        Args:
            original: Dictionary, list, or mapped index to handle.
            redundancy_results: Dictionary, which uses local index
            as keys and lists like [[remote_rank0, remote_index0],...,]
            as values.
        """
        ret = None
        if isinstance(original, (dict, defaultdict)):
            ret = {}
            for key, val in original.items():
                ret[key] = self._convert_index_to_redundancy_results(
                    val, redundancy_results
                )
        elif isinstance(original, list):
            ret = []
            for val in original:
                ret.append(
                    self._convert_index_to_redundancy_results(val, redundancy_results)
                )
        elif isinstance(original, tuple):
            ret = []
            for val in original:
                ret.append(
                    self._convert_index_to_redundancy_results(val, redundancy_results)
                )
            ret = tuple(ret)
        else:
            # original is index
            if original in redundancy_results:
                ret = redundancy_results[original]
            else:
                ret = []
        return ret

    def _store_redundancy_results_to_dicts(self):
        """
        Store redundancy results to corresponding dicts.
        """
        # index mapping dict, keys represents local indexs, values store redundancy
        # results in form [[remote_rank0, remote_index0],...,]
        redundancy_index_mapping = {}
        my_rank = dist.get_rank()
        for remote_rank, results in self.redundancy_results.items():
            for redundancy_result in results:
                remote_index = redundancy_result[
                    redundancy_result.index(remote_rank) + 1
                ]
                local_index = redundancy_result[redundancy_result.index(my_rank) + 1]
                if local_index not in redundancy_index_mapping:
                    # Initialize the redundancy of tensors owned by each rank itself.
                    redundancy_index_mapping[local_index] = [[my_rank, local_index]]
                redundancy_index_mapping[local_index].append(
                    [remote_rank, remote_index]
                )

        self.redundancy_results_for_model_state = (
            self._convert_index_to_redundancy_results(
                self.model_state_index_mapping, redundancy_index_mapping
            )
        )
        my_rank = dist.get_rank()
        if self.zero_stage == 1:
            self.redundancy_results_for_zero_state = {}
            for k, v in self.zero_state_index_mapping.items():
                original_index = v
                if original_index in redundancy_index_mapping:
                    self.redundancy_results_for_zero_state[k] = (
                        redundancy_index_mapping[original_index]
                    )
                else:
                    self.redundancy_results_for_zero_state[k] = []
        else:
            self.redundancy_results_for_optimizer = (
                self._convert_index_to_redundancy_results(
                    self.optimizer_index_mapping, redundancy_index_mapping
                )
            )

            # self.logger.info(
            #     f'self.redundancy_results_for_zero_state.keys:'
            #     f'{self.redundancy_results_for_zero_state.keys()}',
            #     ranks=[my_rank]
            # )

            # self.logger.info(
            #     f'self.redundancy_results_for_zero_state:'
            #     f'{self.redundancy_results_for_zero_state}',
            #     ranks=[my_rank]
            # )

    def get_redundancy_results_for_parameters(
        self,
        recursive_dict,
        redundancy_of_model_parameters,
        needprefix=False,
        prefix="",
    ):
        """
        Get redundancy results for parameters.
        recursive_dict: The dictionary that store redundancy results
        for the experts or the model states.
        redundancy_of_model_parameters: The variable to store results
        for model parameters.
        """
        if isinstance(recursive_dict, (dict, defaultdict)):
            # Name handling: During dictionary traversal, recursively build the
            # full name by appending prefixes to function parameters
            for key, val in recursive_dict.items():
                if needprefix:
                    if prefix == "":
                        cur_prefix = key
                    else:
                        cur_prefix = prefix + "." + key
                else:
                    cur_prefix = key
                # self.logger.info(f'{cur_prefix}', ranks=[0])
                if cur_prefix in redundancy_of_model_parameters.keys():
                    redundancy_of_model_parameters[cur_prefix] = recursive_dict[key]
                else:
                    self.get_redundancy_results_for_parameters(
                        val, redundancy_of_model_parameters, needprefix, cur_prefix
                    )
        elif isinstance(recursive_dict, (list, tuple)):
            for val in recursive_dict:
                self.get_redundancy_results_for_parameters(
                    val, redundancy_of_model_parameters, needprefix, prefix
                )
        else:
            pass

    def get_sub_tensors_in_partition(
        self, params_in_partition, first_offset, partition_size
    ):
        """Extract sub-tensors from parameters for partitioned processing

        Splits model parameters into sub-tensors based on partition boundaries for
        distributed processing. Handles tensor slicing and offset calculations to
        create contiguous views of parameter subsets.

        Args:
            params_in_partition: List of parameters to be partitioned
            first_offset: Starting offset for the first tensor in the partition
            partition_size: Maximum size of the partition in elements

        Returns:
            sub_tensors_list: List of tensor views for the partition
            sub_tensors_name_list: List of corresponding parameter names
        """
        sub_tensors_list = []
        sub_tensors_name_list = []
        current_size = 0

        for i, tensor in enumerate(params_in_partition):
            name = self.nvmapping.get_name_from_optimizer(tensor)

            sub_tensors_name_list.append(name)

            num_elements = tensor.numel()
            tensor_offset = 0

            # we need to offset to get to the right element
            if i == 0 and first_offset > 0:
                tensor_offset = first_offset
                num_elements = num_elements - tensor_offset
                self.logger.debug(f'Rank:{dist.get_rank()}, tensor named {name} has offset {tensor_offset}.')

            # we dont need all elements of the tensor
            if num_elements > (partition_size - current_size):
                self.logger.debug(f'Rank:{dist.get_rank()}, tensor named {name} place {partition_size - current_size} elements, reducing tail elements with size {num_elements - partition_size + current_size}.')
                num_elements = partition_size - current_size
                
            # we need a narrow view of the tensor based on the tensor offset
            # and number of elements that we need from this tensor
            if tensor_offset > 0 or num_elements < tensor.numel():
                self.sub_tensors_name_range_mapping[name] = [
                    int(tensor_offset),
                    int(num_elements),
                ]
                sub_tensors_list.append(
                    tensor.contiguous()
                    .view(-1)
                    .narrow(0, int(tensor_offset), int(num_elements))
                )
            else:
                sub_tensors_list.append(tensor)

            current_size = current_size + num_elements

        return sub_tensors_list, sub_tensors_name_list

    def get_names_of_parameters_requiring_gradient_checkpointing(self):
        """
        Get non redundancy parameters according to failure tolerance
        factor k, redundancy results for model states, and
        redundancy results for optimizer.
        This function return the name of parameters requiring the gradient
        checkpointing during training.
        """
        model_state_redundancy = {}
        optimizer_state_redundancy = {}
        model_state_redundancy_renamed = {}

        for name, _ in self.model.named_parameters():
            model_state_redundancy[name] = name
            model_state_redundancy_renamed[name] = None
            optimizer_state_redundancy[name] = None

        # self.get_redundancy_results_for_parameters(None, model_state_redundancy_renamed)

        if self.redundancy_results_for_model_state is not None:
            # Name handling: The model_state saved by save_checkpoint function contains model
            # parameters under 'module' key, but unlike self.model.named_parameters() which
            # directly gives 'a.b.c' names, the structure is nested like {a: {b: {c: val}}}.
            # During traversal, we need to record the prefix to reconstruct the full parameter
            # names.
            self.get_redundancy_results_for_parameters(
                self.redundancy_results_for_model_state,
                model_state_redundancy_renamed,
                needprefix=True,
            )

        for name, _ in model_state_redundancy.items():
            model_state_redundancy[name] = model_state_redundancy_renamed[
                model_state_redundancy[name]
            ]

        if self.zero_stage == 1:
            for k, v in self.redundancy_results_for_zero_state.items():
                if k in optimizer_state_redundancy:
                    optimizer_state_redundancy[k] = v
        else:
            for i, param_group in enumerate(self.optimizer.state_dict()["fp16_groups"]):
                redundancy_result_of_param_group = (
                    self.redundancy_results_for_optimizer["fp16_groups"][i]
                )

                # index_mapping_value = self.optimizer_index_mapping['optimizer']['fp16_groups'][i]
                for j, value in enumerate(param_group):
                    name = self.nvmapping.get_name_from_optimizer(value)
                    redundancy_results = redundancy_result_of_param_group[j]
                    # mapped_index = index_mapping_value[j]
                    if name in optimizer_state_redundancy:
                        optimizer_state_redundancy[name] = redundancy_results

        parameter_names_for_gradient_checkpointing_cur_iter = []

        # For debugging
        self.logger.debug(f'rank:{dist.get_rank()}, model_state_redundancy:{model_state_redundancy}', ranks=[dist.get_rank()])
        self.logger.debug(f'rank:{dist.get_rank()}, optimizer_state_redundancy:{optimizer_state_redundancy}', ranks=[dist.get_rank()])

        for name, result in optimizer_state_redundancy.items():
            if result is None:
                # parameter_names_for_gradient_checkpointing_cur_iter.append(name)
                continue
            redundancy_result_from_model_state_dicts = model_state_redundancy[name]
            keys = []
            final_redundancy_result_for_parameter = []
            for r in redundancy_result_from_model_state_dicts:
                keys.append(r[0])
            for r in result:
                if r[0] in keys:
                    final_redundancy_result_for_parameter.append(r)

            # Handling redundancy results to node level.
            self.logger.debug(
                f'Before handling to node level: {name}:{len(final_redundancy_result_for_parameter)}',
                ranks=[dist.get_rank()]
            )
            reduced_redundancy_results = set()
            for result in final_redundancy_result_for_parameter:
                if result[0] // 4 not in reduced_redundancy_results:
                    reduced_redundancy_results.add(result[0] // 4)
            final_redundancy_result_for_parameter = reduced_redundancy_results
            self.logger.debug(
                f'After handling to node level: {name}:{len(final_redundancy_result_for_parameter)}',
                ranks=[dist.get_rank()]
            )

            if len(final_redundancy_result_for_parameter) < self.k:
                parameter_names_for_gradient_checkpointing_cur_iter.append(name)

        if self.parameter_names_for_gradient_checkpointing is None:
            self.parameter_names_for_gradient_checkpointing = (
                parameter_names_for_gradient_checkpointing_cur_iter
            )
        else:
            self.parameter_names_for_gradient_checkpointing = [
                name
                for name in self.parameter_names_for_gradient_checkpointing
                if name in parameter_names_for_gradient_checkpointing_cur_iter
            ]

        self.logger.debug(
            f'Rank:{dist.get_rank()}, self.parameter_names_for_gradient_checkpointing:'
            f'{self.parameter_names_for_gradient_checkpointing}',
            ranks=[dist.get_rank()]
        )

    def get_single_dict_for_gradient_checkpointing(
        self,
        recursive_dict,
        parameter_names_for_gradient_checkpointing,
        needprefix=False,
        prefix="",
    ):
        """
        Get parameter dict for gradient checkpointing based on single
        recursive_dict.
        Args:
            recursive_dict: dict to find the parameters.
        """
        if isinstance(recursive_dict, (dict, defaultdict)):
            for key, val in recursive_dict.items():
                if needprefix:
                    if prefix == "":
                        cur_prefix = key
                    else:
                        cur_prefix = prefix + "." + key
                else:
                    cur_prefix = key
                # self.logger.info(f'{cur_prefix}', ranks=[0])
                if cur_prefix in parameter_names_for_gradient_checkpointing:
                    store_tensor = recursive_dict[key]
                    self.parameter_dict_for_gradient_checkpointing[cur_prefix] = (
                        store_tensor
                    )

                    # self.logger.info(
                    #     f'{dist.get_rank()} try to store {cur_prefix}, '
                    #     f'the value is {recursive_dict[key]}, '
                    #     f'the type of value is {type(recursive_dict[key])}',
                    #     ranks=[0]
                    # )
                else:
                    self.get_single_dict_for_gradient_checkpointing(
                        val,
                        parameter_names_for_gradient_checkpointing,
                        needprefix,
                        cur_prefix,
                    )
        elif isinstance(recursive_dict, (list, tuple)):
            for val in recursive_dict:
                self.get_single_dict_for_gradient_checkpointing(
                    val, parameter_names_for_gradient_checkpointing, needprefix, prefix
                )
        else:
            pass

    def get_dict_for_gradient_checkpointing(self):
        """
        Get parameter dict for gradient checkpointing.
        """
        self.parameter_dict_for_gradient_checkpointing = {}
        model_state_redundancy = []

        my_rank = dist.get_rank()

        for name in self.parameter_names_for_gradient_checkpointing:
            model_state_redundancy.append(name)

        if self.model_state is not None:
            self.get_single_dict_for_gradient_checkpointing(
                self.model_state, model_state_redundancy, needprefix=True
            )

        if self.zero_stage == 1:
            self.logger.debug(f'Rank:{my_rank}, self.sub_tensors_name_range_mapping:{self.sub_tensors_name_range_mapping}')
            for name, val in self.sub_tensors_name_range_mapping.items():
                search_name = name

                if search_name in self.parameter_dict_for_gradient_checkpointing:
                    orginal_tensor = self.parameter_dict_for_gradient_checkpointing[
                        search_name
                    ]
                    stored_tensor = (
                        orginal_tensor.contiguous()
                        .view(-1)
                        .narrow(0, int(val[0]), int(val[1]))
                    )
                    stored_tensor_gard = None
                    if orginal_tensor.grad is not None:
                        stored_tensor_gard = (
                            orginal_tensor.grad.contiguous()
                            .view(-1)
                            .narrow(0, int(val[0]), int(val[1]))
                        )
                    stored_tensor.grad = stored_tensor_gard
                    self.parameter_dict_for_gradient_checkpointing[search_name] = (
                        stored_tensor
                    )
                    self.logger.debug(f'Rank:{my_rank}, store tensor {search_name} with size {stored_tensor.shape}, stored_tensor.dtype:{stored_tensor.dtype}')
                else:
                    self.logger.info(
                        f"{search_name} not in self.parameter_dict_for_gradient_checkpointing",
                        ranks=[my_rank],
                    )

    def build_name_value_mapping(self):
        self.nvmapping.empty_cache()
        for name, param in self.model.named_parameters():
            self.nvmapping.initialization(param, name)

        if isinstance(self.optimizer, ZeroOptimizer_Stage1):
            for group in self.optimizer.state_dict()["params_in_partition"]:
                for p in group:
                    name = self.nvmapping.get_name_from_parameters(p)
                    self.nvmapping.optimizer_param_name_mapping(p, name)
        else:
            for group in self.optimizer.state_dict()["fp16_groups"]:
                for p in group:
                    name = self.nvmapping.get_name_from_parameters(p)
                    self.nvmapping.optimizer_param_name_mapping(p, name)

        self.nvmapping.find_name_with_equal_hash_value()

    def get_baseline_packked_tensor(self):
        self.packed_tensor = torch.tensor([], device="cpu")
        # self.packed_tensor = torch.tensor([], device=get_accelerator().device_name())

        self.logger.info(
            f"self.packed_tensor device {self.packed_tensor.device}", ranks=[0]
        )

        for _, parameter in self.model.named_parameters():
            new_tensor = parameter.detach()
            # cpu_tensor = new_tensor.to('cpu')
            # self.logger.info(f'new_tensor device {new_tensor.device}', ranks=[0])
            self.packed_tensor = torch.cat((self.packed_tensor, new_tensor.view(-1)))

    def _redundancy_checking_core(self, group_ranks):
        """
        冗余检查的核心逻辑，被 baseline_compare 和 do_redundancy_checking 共用
        """
        my_rank = dist.get_rank()
        world_size = dist.get_world_size()

        # 获取各rank的tensor大小信息
        packed_tensor_sizes_for_each_rank = torch.empty(
            (world_size, 1), device=torch.cuda.current_device(), dtype=torch.int64
        )
        packed_tensor_sizes_for_each_rank[my_rank] = self.packed_tensor.shape[0]

        for i in range(world_size):
            torch.distributed.broadcast(packed_tensor_sizes_for_each_rank[i], src=i)

        # 处理每个通信组
        for ranks in group_ranks:
            if len(ranks) == 1:
                continue

            n = len(ranks)
            rank_index = ranks.index(my_rank)

            # 计算步骤数
            cal_steps, transfer_steps = self._calculate_steps(n)

            # 邻居rank
            prev_rank = ranks[(rank_index - 1) % n]
            next_rank = ranks[(rank_index + 1) % n]

            # 计算阶段
            tensor_send = self.packed_tensor.clone().to("cuda")

            for i in range(cal_steps):
                recv_rank = ranks[(rank_index + 1 + i) % n]
                tensor_recv = torch.empty(
                    packed_tensor_sizes_for_each_rank[recv_rank],
                    device=torch.cuda.current_device(),
                    dtype=torch.int64,
                )

                # 执行P2P通信
                self._execute_p2p_communication(tensor_send, prev_rank, tensor_recv, next_rank)

                # 计算冗余
                if (not self.redundancy_cache or tensor_recv not in self.redundancy_results):
                    self._calculate_redundancy(recv_rank, tensor_recv, my_rank)

                # 准备下一轮通信
                tensor_send = tensor_recv

            # 释放GPU内存
            tensor_send = None
            tensor_recv = None

            # 传输阶段
            if transfer_steps > 0:
                self._transfer_phase(ranks, rank_index, transfer_steps, n)

        # 释放GPU内存
        packed_tensor_sizes_for_each_rank = None

    def _calculate_steps(self, group_size):
        """计算计算步骤和传输步骤数"""
        if self.no_redundant_cal:
            cal_steps = group_size // 2
            transfer_steps = group_size // 2
            if group_size % 2 == 0:
                transfer_steps -= 1
        else:
            cal_steps = group_size - 1
            transfer_steps = 0
        return cal_steps, transfer_steps

    def _transfer_phase(self, ranks, rank_index, transfer_steps, group_size):
        """传输阶段：交换冗余结果"""
        for i in range(transfer_steps):
            send_to_rank = ranks[(rank_index + transfer_steps - i) % group_size]
            recv_from_rank = ranks[(rank_index - transfer_steps + i) % group_size]

            # 交换tensor形状信息
            send_shape = self._get_redundancy_tensor_shape(send_to_rank)
            recv_shape = torch.empty((2), device=torch.cuda.current_device(), dtype=torch.int64)

            self._execute_p2p_communication(send_shape, send_to_rank, recv_shape, recv_from_rank)

            # 交换冗余结果tensor
            send_tensor = torch.tensor(
                self.redundancy_results[send_to_rank],
                device=torch.cuda.current_device(),
                dtype=torch.int64,
            )
            recv_tensor = torch.empty(
                tuple(recv_shape),
                device=torch.cuda.current_device(),
                dtype=torch.int64,
            )

            self._execute_p2p_communication(send_tensor, send_to_rank, recv_tensor, recv_from_rank)

            # 处理接收到的冗余结果
            if len(recv_tensor) > 0:
                list_recv_tensor = recv_tensor.tolist()
                if list_recv_tensor and list_recv_tensor[0][0] not in self.redundancy_results:
                    self.redundancy_results[list_recv_tensor[0][0]] = list_recv_tensor

            # 释放GPU内存
            send_shape = None
            recv_shape = None
            send_tensor = None
            recv_tensor = None

    def _execute_p2p_communication(self, send_tensor, send_rank, recv_tensor, recv_rank):
        """执行点对点通信"""
        ops = []
        send_op = torch.distributed.P2POp(
            torch.distributed.isend, send_tensor, send_rank
        )
        ops.append(send_op)
        recv_op = torch.distributed.P2POp(
            torch.distributed.irecv, recv_tensor, recv_rank
        )
        ops.append(recv_op)

        if ops:
            reqs = torch.distributed.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

    def _calculate_redundancy(self, target_rank, target_tensor, my_rank):
        """计算与目标rank的冗余"""
        self.redundancy_results[target_rank] = []
        for idx1, packed_tensor in enumerate(self.packed_tensor):
            for idx2, tensor in enumerate(target_tensor):
                if packed_tensor == tensor:
                    self.redundancy_results[target_rank].append(
                        [my_rank, idx1, target_rank, idx2]
                        )

    def _get_redundancy_tensor_shape(self, target_rank):
        """获取冗余结果tensor的形状"""
        if self.redundancy_results.get(target_rank):
            return torch.tensor(
                [len(self.redundancy_results[target_rank]), 4],
                device=torch.cuda.current_device(),
                dtype=torch.int64,
            )
        return torch.tensor(
            [0, 0], device=torch.cuda.current_device(), dtype=torch.int64
        )

    # 重构后的主要方法
    def baseline_compare(self):
        """基准比较方法"""
        self._get_data_parallel_group_ranks()

        # 构建通信组：包含所有rank
        group_ranks = [list(range(dist.get_world_size()))]

        self._redundancy_checking_core(group_ranks)

    def do_redundancy_checking(self):
        """执行冗余检查"""
        self._get_data_parallel_group_ranks()

        self.redundancy_results = {}

        # 构建通信组
        if self.comm_groups_for_redundancy_checking:
            group_ranks = [self.data_ranks]
        else:
            group_ranks = [list(range(dist.get_world_size()))]

        # 扁平化并排序通信组
        flatten_ranks = []
        for ranks in group_ranks:
            for rank in ranks:
                if rank not in flatten_ranks:
                    flatten_ranks.append(rank)
        flatten_ranks.sort()
        group_ranks = [flatten_ranks]

        self._redundancy_checking_core(group_ranks)

        # 存储结果到字典
        self._store_redundancy_results_to_dicts()
