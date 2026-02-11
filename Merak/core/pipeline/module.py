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

# Parts of the code here are adapted from https://github.com/microsoft/DeepSpeed/blob/85acf14c58658964a796c5c901b58123f99fb1df/deepspeed/runtime/pipe/module.py

import collections
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import autocast
from torch.distributed.distributed_c10d import get_global_rank
try:
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor
from transformers import PretrainedConfig

from Merak import get_logger

from .. import mpu
from ..finetuning.lora import (
    LoraConfig,
    _find_and_replace,
    _prepare_lora_config,
    mark_only_lora_as_trainable,
)
from ..fx import add_inputs_to_shards, convert_to_sequential
from ..mpu.layers import VocabParallelEmbedding
from ..recompute.checkpointing import checkpoint as checkpoint_func
from ..recompute.checkpointing import get_rng_tracker
from ..sequence_parallel import get_leaf_modules_for_sp, replace_to_sp_module
from ..tensor_parallel import ModuleRebuild
from ..tensor_parallel.utils import init_method_normal
from . import module_utils
from .layers_partition import LayerPartition

try:
    from contextlib import nullcontext
except ImportError:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext(enter_result=None):
        yield enter_result


class PipelineError(Exception):
    """Errors related to the use of deepspeed.PipelineModule"""


class PipelineModuleBase:
    """Pipeline Module Base Class

    Provides fundamental properties and utilities for pipeline parallel training.
    This class handles the basic initialization and configuration that is common
    across all pipeline module implementations.
    """

    def __init__(self, loss_fn, seed_layers, seed_fn, base_seed, dummy_inputs):
        """Initialize base pipeline module properties."""
        # Model configuration
        self.args = None
        self.config = None
        self.device = None
        self.init_method = None
        self.parts = None
        self.dummy_inputs = dummy_inputs
        self.logger = get_logger("simple")

        # Training configuration
        self.loss_fn = loss_fn
        self.seed_layers = seed_layers
        self.seed_fn = seed_fn
        self.base_seed = base_seed

        # Distributed communication
        self._grid = None
        self.world_group = None
        self.global_rank = None
        self.world_size = None
        self.local_rank = None

        # Parallelism flags
        self.model_parallel = False
        self.sequence_parallel = False
        self.stage_id = None
        self.num_stages = None

        # Model structure
        self.forward_funcs = []
        self.tied_modules = None
        self.tied_weight_attrs = {}
        self.tied_comms = {}
        self.tied_modules_keys = None
        self.tied_stage = None
        self.input_to_stage_dic = None
        self._layer_specs = []
        self._num_layers = 0
        self._local_start = 0
        self._local_stop = None

        # Activation checkpointing
        self.act_ckpt_interval = None
        self.act_ckpt_func = None
        self.act_ckpt_ratio = None
        self.first_checkpointable = False
        self.checkpointable_num_layers = 0
        self.checkpointable_idx = []

        seed_str = self.seed_fn.__name__ if self.seed_fn else "None"
        self.logger.info(
            f"SEED_LAYERS={self.seed_layers} BASE_SEED={self.base_seed} SEED_FN={seed_str}",
            ranks=[0],
        )


class PipelineModuleMixin:
    """Pipeline Module Mixin Class

    Contains reusable methods for pipeline module functionality.
    These methods can be mixed into any pipeline module implementation.
    """

    def _set_bounds(self, start: int = None, stop: int = None):
        """Manually define the range of layers that will be built on this process.

        These boundaries are treated as list slices and so start is inclusive
        and stop is exclusive. The default of None for both results in all
        layers being built locally.

        Args:
            start: Starting layer index (inclusive)
            stop: Ending layer index (exclusive)
        """
        self._local_start = start
        self._local_stop = stop

    def num_pipeline_stages(self) -> int:
        """Returns the number of pipeline stages.

        Returns:
            int: The number of pipeline stages in the current configuration
        """
        return mpu.get_pipe_parallel_world_size()

    def partitions(self) -> List[int]:
        """Returns the partition boundaries of layers for each pipeline stage.

        Returns:
            List[int]: A list where each element represents the start index of a stage's layers
        """
        return self.parts

    def stage_owner(self, layer_idx: int) -> int:
        """Determines which pipeline stage owns a specific layer.

        Args:
            layer_idx: The index of the layer to check ownership for

        Returns:
            int: The stage ID that owns the specified layer

        Raises:
            RuntimeError: If the layer index is not found in any stage's partition
        """
        for stage in range(mpu.get_pipe_parallel_world_size()):
            if self.parts[stage] <= layer_idx < self.parts[stage + 1]:
                return stage
        raise RuntimeError(f"Layer {layer_idx} not owned? Parts: {self.parts}")

    def allreduce_tied_weight_gradients(self):
        """All reduce the gradients of the tied weights between tied stages.

        Performs gradient synchronization for weights that are shared across
        multiple pipeline stages to ensure consistent updates.
        """
        for key, comm in self.tied_comms.items():
            weight = getattr(self.tied_modules[key], comm["weight_attr"])
            dist.all_reduce(weight.grad, group=comm["group"])

    @torch.no_grad
    def _synchronize_tied_weights(self):
        """Broadcast tied weights from the source process to all others.

        Ensures that tied weights are synchronized across all processes
        that share them, typically after weight updates.
        """
        for _, comm in self.tied_comms.items():
            dist.broadcast(
                getattr(comm["module"], comm["weight_attr"]),
                src=min(comm["ranks"]),
                group=comm["group"],
            )


class PipelineModule(PipelineModuleBase, PipelineModuleMixin, nn.Module):
    """A module enabling pipeline parallelism for efficient distributed training.

    This class implements pipeline parallelism by partitioning model layers into
    different stages. Each stage is assigned a subset of the model layers and
    processes inputs sequentially.

    Attributes:
        model (nn.Module): The input model to be parallelized.
        args (MerakArguments): Configuration arguments for training.
        topology (Callable): Defines the process topology for parallelism.
        loss_fn (Callable, optional): Loss function for training.
        seed_layers (bool): Whether to seed layers for reinitialization.
        seed_fn (Callable, optional): Function for seeding layers.
        base_seed (int): Base seed for random number generation.
        communication_grid (Callable, optional): Communication grid for parallelism.
        leaf_modules (list): Additional leaf modules to consider.
        dummy_inputs (Dict[str, torch.Tensor], optional): Dummy inputs for model tracing.
    """

    def __init__(
        self,
        model: nn.Module,
        args: Any,
        loss_fn: Optional[Callable] = None,
        seed_layers: bool = False,
        seed_fn: Optional[Callable] = None,
        base_seed: int = 1234,
        leaf_modules: Tuple = (),
        dummy_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        # Initialize base classes
        PipelineModuleBase.__init__(
            self, loss_fn, seed_layers, seed_fn, base_seed, dummy_inputs
        )
        PipelineModuleMixin.__init__(self)
        nn.Module.__init__(self)

        # Initialize model and configuration
        self._initialize_model_properties(model, args)
        self._initialize_parallel_properties()

        # Build pipeline structure
        self._initialize_layers(model, leaf_modules, checkpoint_func)
        self.configure_distributed_model()

        # Tie modules if applicable
        if not self.args.no_tie_modules and mpu.get_pipe_parallel_world_size() == 1:
            self.tie_modules()

    def _initialize_layers(
        self, model: nn.Module, leaf_modules: list, activation_checkpoint_func: Callable
    ) -> None:
        """Initialize model layers and prepare for partitioning.

        Args:
            model: The base model to partition
            leaf_modules: Additional modules to treat as leaves during tracing
            activation_checkpoint_func: Function for activation checkpointing
        """
        # Calculate total parameters
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        self.total_num_params = sum(p.numel() for p in model_parameters)

        # Handle precision conversion
        model_class = model.__class__

        # First, To enable tensor parallelism, need to set attribute for model
        # in class ModuleRebuild;
        # Second, To enable pipeline parallelism, the model is first converted
        # into a sequential form. Depending on its initial structure, this is
        # done by either using it directly as an nn.Sequential or by tracing
        # it into a single GraphModule.
        rebuild_module = ModuleRebuild(self.args, model, model_class)
        if isinstance(model, nn.Sequential):
            layers, input_to_shard = (list(model), None)
        else:
            # trace module to GraphModule list
            layers, input_to_shard = self._traced_module(model, leaf_modules)

        self.logger.debug(layers, ranks=[0])

        # Store layer specifications
        self._layer_specs = layers if isinstance(layers, list) else list(layers)
        self._num_layers = len(self._layer_specs)

        # For pipeline parallelism, the model is partitioned into stages,
        # with each stage containing a contiguous set of layers.
        self.split_method = LayerPartition(
            self.args, self._layer_specs, self.global_rank
        )
        self._partition_layers(input_to_shard)

        # Configure activation checkpointing
        self._configure_activation_checkpoints(activation_checkpoint_func)

        # Build the final module structure
        tie_dims = self._set_tie_dim(model)
        self._build(tie_dims, input_to_shard)

        # Apply sequence parallelism if enabled
        if self.sequence_parallel:
            replace_to_sp_module(self, model)

        # For tensor parallelism, standard Linear layers are replaced with their
        # RowParallelLinear and ColParallelLinear counterparts, respectively.
        # Additionally, models initialized on a meta device require parameter
        # re-initialization.
        should_reinit = rebuild_module.recover_module(self)
        if should_reinit:
            self.apply(self.init_method)
        rebuild_module.vocab_parallel(emb_dim=tie_dims)

    def _traced_module(
        self, module: nn.Module, leaf_modules: Tuple
    ) -> Tuple[Union[List[torch.fx.GraphModule], Dict[str, int]]]:
        """Convert the model into a traced module for pipeline execution.

        Args:
            module: The model to convert to sequential format
            leaf_modules: Additional leaf modules to consider during tracing

        Returns:
            Tuple containing:
                - Traced model layers as sequential modules
                - Dictionary mapping input names to their shard IDs
        """
        if self.sequence_parallel:
            sp_leaf_modules = get_leaf_modules_for_sp(module)
            leaf_modules = tuple(set(leaf_modules + sp_leaf_modules))

        model, layers, input_to_shard = convert_to_sequential(
            module,
            self.args,
            self.dummy_inputs,
            extra_leaf_modules=leaf_modules,
        )
        del model
        return layers, input_to_shard

    def _initialize_model_properties(self, model: nn.Module, args: Any) -> None:
        """Initialize model properties, including initialization method and device.

        Args:
            model: The neural network model to initialize properties from
            args: Configuration arguments containing device and initialization settings
        """
        self.init_method = (
            model._init_weights
            if hasattr(model, "_init_weights")
            else init_method_normal(args.init_method_std)
        )
        self.args = args
        self.device = args.device
        self.config = model.config if hasattr(model, "config") else None

    def _initialize_parallel_properties(self) -> None:
        """Initialize properties related to distributed parallel execution.

        Sets up world groups, rank information, and parallelism flags based on
        the current distributed configuration.
        """
        self._grid = self.args.get_communication_grid()
        self.world_group = dist.new_group(ranks=range(dist.get_world_size()))
        self.global_rank = dist.get_rank(group=self.world_group)
        self.world_size = dist.get_world_size(group=self.world_group)
        self.local_rank = self.args.local_rank
        assert self.local_rank is not None

        self.model_parallel = mpu.get_model_parallel_world_size() > 1
        self.sequence_parallel = mpu.get_sequence_parallel_world_size() > 1
        if self.sequence_parallel:
            mpu.mappings.set_sequence_dim(self.args.sequence_dim)

        self.stage_id = mpu.get_pipe_parallel_rank()
        self.num_stages = mpu.get_pipe_parallel_world_size()

        self.micro_offset = 0

    def _configure_activation_checkpoints(self, activation_checkpoint_func: Callable):
        """Configure activation checkpoints based on specified parameters.

        Args:
            activation_checkpoint_func: Function to perform activation checkpointing
        """
        self.act_ckpt_interval = self.args.checkpoint_num_layers
        self.act_ckpt_func = activation_checkpoint_func
        self.act_ckpt_ratio = self.args.activation_checkpoint_ratio

        if self.act_ckpt_ratio is not None:
            if len(self.act_ckpt_ratio) == 1:
                first_ratio = 1 - float(self.act_ckpt_ratio[0])
                self.act_ckpt_ratio = [
                    1 - (first_ratio * (self.num_stages - 1) / (self.num_stages - s))
                    for s in range(1, self.num_stages)
                ] + [0]
                self.logger.info(
                    f"Activation checkpoint ratio list: {self.act_ckpt_ratio}",
                    ranks=[0],
                )
                if self.act_ckpt_ratio[self.stage_id] <= 0:
                    self.act_ckpt_interval = 0
            elif len(self.act_ckpt_ratio) < self.num_stages:
                last_ratio = self.act_ckpt_ratio[-1]
                self.act_ckpt_ratio += [last_ratio] * (
                    self.num_stages - len(self.act_ckpt_ratio)
                )

        if (
            self.args.split_method == "nearest_min_deps"
            and self.args.activation_checkpointing
        ):
            param_counts = self.split_method.count_parameters()
            self.act_ckpt_interval = module_utils.partition_balanced(
                weights=param_counts, num_parts=self._num_layers // self.num_stages
            )

    @torch.no_grad
    def _broadcast_model(self):
        """Broadcast the model parameters across distributed processes.

        Ensures all processes have the same initial model parameters by broadcasting
        from the source rank (typically rank 0) to all other processes.
        """
        broadcast_src_rank = get_global_rank(mpu.get_data_parallel_group(), 0)
        for p in self.parameters():
            if isinstance(p, DTensor):
                p.from_local(
                    p.to_local(), device_mesh=p.device_mesh, placements=p.placements
                )
            else:
                dist.broadcast(
                    p, broadcast_src_rank, group=mpu.get_data_parallel_group()
                )

    def configure_distributed_model(self):
        """Configure the model for distributed execution on the specified device.

        Handles precision configuration (FP16, BF16) and ensures all parameters
        are on the correct device with the correct data type.
        """
        if self.args.half_precision_backend == "apex":
            return

        # Determine target data type based on precision settings
        if self.args.fp16 and self.args.half_precision_backend != "cuda_amp":
            should_dtype = torch.half
        elif self.args.bf16 and self.args.half_precision_backend != "cuda_amp":
            should_dtype = torch.bfloat16
        else:
            should_dtype = torch.float

        # Convert model to target data type and device
        self.to(should_dtype)
        if not all([param.dtype == should_dtype for param in self.parameters()]):
            names = [n for n, p in self.named_parameters() if p.dtype != should_dtype]
            raise ValueError(
                f"Current model should be {str(should_dtype)} but the following parameters have "
                f"dtype that is not {str(should_dtype)}: {', '.join(names)}"
            )
        self.to(self.device)

        # Broadcast model parameters if not using APEX
        if not self.args.half_precision_backend == "apex":
            self._broadcast_model()

    def _add_lora_layers(self, model_config: Union[PretrainedConfig, dict]):
        """Add LoRA layers to the model for parameter-efficient fine-tuning.

        Args:
            model_config: Model configuration for LoRA adaptation

        Returns:
            peft_config: The configured LoRA configuration

        Raises:
            AssertionError: If tensor parallelism is enabled (not supported with LoRA)
        """
        assert (
            mpu.get_model_parallel_world_size() == 1
        ), "LoRA adaptation not supported with tensor parallelism."
        peft_config = LoraConfig(**self.args.get_lora_config())
        peft_config = _prepare_lora_config(peft_config, model_config.to_dict())
        _find_and_replace(self, adapter_name=self.args.adapter_name, config=peft_config)
        mark_only_lora_as_trainable(self, peft_config)
        module_utils.print_trainable_parameters(self)
        return peft_config

    def _build(self, tie_dims: set, input_to_shard_dic: Dict[str, int]):
        """Build the pipeline module with specified tie dimensions and input sharding.

        Args:
            tie_dims: Set of weight shapes that should be tied across stages
            input_to_shard_dic: Dictionary mapping input names to their shard IDs
        """
        specs = self._layer_specs

        # Initialize tied module tracking
        self.tied_modules_keys = set(str(s).replace(".", "_") for s in tie_dims)
        self.tied_stage = collections.defaultdict(set)
        self.input_to_stage_dic = collections.defaultdict(list)

        # Map inputs to stages
        if input_to_shard_dic is not None:
            for input_name, shard_id in input_to_shard_dic.items():
                self.input_to_stage_dic[self.stage_owner(shard_id)].append(input_name)

        # Track which stages have which tied weights
        for layer_idx, layer in enumerate(specs):
            for m in layer.modules():
                if hasattr(m, "weight"):
                    try:
                        weight_shape = m.weight.shape
                    except AttributeError:
                        continue
                    if weight_shape in tie_dims:
                        self.tied_stage[str(weight_shape).replace(".", "_")].add(
                            self.stage_owner(layer_idx)
                        )

        # Build layers for current stage
        for local_idx, layer in enumerate(specs[self._local_start : self._local_stop]):
            layer_idx = local_idx + self._local_start

            # Set layer-specific random seed if enabled
            if self.seed_layers:
                if self.seed_fn:
                    self.seed_fn(self.base_seed + layer_idx)
                else:
                    module_utils.set_random_seed(self.base_seed + layer_idx)

            # Handle different layer types
            if isinstance(layer, PipelineModule):
                raise NotImplementedError("RECURSIVE BUILD NOT YET IMPLEMENTED")
            if isinstance(layer, nn.Module):
                name = str(layer_idx)
                inputs_of_this_stage = []

                # Add input shards that belong to this stage
                for input_name in self.input_to_stage_dic[self.stage_id]:
                    if layer_idx < input_to_shard_dic[input_name]:
                        inputs_of_this_stage.append(input_name)
                if inputs_of_this_stage:
                    layer = add_inputs_to_shards(layer, inputs_of_this_stage)

                self.forward_funcs.append(layer)
                self.add_module(name, layer)
            else:
                self.forward_funcs.append(layer)

        # Configure activation checkpointing ranges
        self._setup_activation_checkpointing_ranges()

        # Mark all parameters for model parallelism
        for p in self.parameters():
            p.model_parallel = True

    def _setup_activation_checkpointing_ranges(self):
        """Setup activation checkpointing ranges based on configuration."""
        num_layers = len(self.forward_funcs)
        if (
            self.act_ckpt_ratio is not None
            and float(self.act_ckpt_ratio[self.stage_id]) != 1.0
        ):
            prev_checkpointable = None

            # Determine checkpointable ranges
            for start_idx in range(0, num_layers):
                end_idx = start_idx + 1
                funcs = self.forward_funcs[start_idx:end_idx]

                if self._is_checkpointable(funcs, end_idx, num_layers):
                    if prev_checkpointable:
                        self.checkpointable_idx[-1][1] = end_idx
                    else:
                        if prev_checkpointable is None:
                            self.first_checkpointable = True
                        self.checkpointable_idx.append([start_idx, end_idx])
                    self.checkpointable_num_layers += 1
                    prev_checkpointable = True
                else:
                    if prev_checkpointable is None or prev_checkpointable:
                        if prev_checkpointable is None:
                            self.first_checkpointable = False
                        self.checkpointable_idx.append([start_idx, end_idx])
                    else:
                        self.checkpointable_idx[-1][1] = end_idx
                    prev_checkpointable = False

    def _partition_layers(
        self, input_to_shard_dic: Optional[Dict[str, int]] = None
    ) -> None:
        """Partitions the layers across pipeline stages for distributed training.

        Args:
            input_to_shard_dic: Optional dictionary specifying input sharding

        Returns:
            None
        """
        num_stages = mpu.get_pipe_parallel_world_size()
        stage_id = mpu.get_pipe_parallel_rank()

        self.parts = self.split_method.get_partition(input_to_shard_dic)

        # Print partitioning information on rank 0
        if self.global_rank == 0:
            self.logger.info(self.parts)
            for stage in range(num_stages):
                start = self.parts[stage]
                stop = self.parts[stage + 1]
                self.logger.info(f"stage={stage} layers={stop - start}")
                for idx, layer in enumerate(self._layer_specs[start:stop]):
                    name = str(layer)
                    if isinstance(layer, nn.Module):
                        name = layer.__class__.__name__
                    else:
                        try:
                            name = layer.__name__
                        except AttributeError:
                            pass
                    self.logger.info(f"    {idx+start:2d}: {name}")
            if self.loss_fn:
                try:
                    self.logger.info(f"  loss: {self.loss_fn.__name__}")
                except AttributeError:
                    self.logger.info(f"  loss: {self.loss_fn.__class__.__name__}")

        self._set_bounds(start=self.parts[stage_id], stop=self.parts[stage_id + 1])

    def _set_tie_dim(self, model: nn.Module) -> set:
        """Determine the embedding dimensions to be tied across stages.

        Args:
            model: The model containing embedding layers

        Returns:
            set: Set of unique embedding dimensions that should be tied
        """
        emb_dim = set()

        def add_dim(m):
            emb_dim.add(m.weight.shape)

        # Check for tied embeddings in model config
        if hasattr(model, "config"):
            if model.config.tie_word_embeddings:
                for m in model.modules():
                    try:
                        if (
                            hasattr(m, "get_input_embeddings")
                            and m.get_input_embeddings() is not None
                        ):
                            add_dim(m.get_input_embeddings())
                        if (
                            hasattr(m, "get_output_embeddings")
                            and m.get_output_embeddings() is not None
                        ):
                            add_dim(m.get_output_embeddings())
                    except (AttributeError, NotImplementedError):
                        continue
        elif hasattr(model, "get_input_embeddings"):
            add_dim(model.get_input_embeddings())
        elif hasattr(model, "get_output_embeddings"):
            add_dim(model.get_output_embeddings())

        return emb_dim

    def tie_modules(self) -> None:
        """Create communication structures for tied modules.

        This function identifies and ties together modules across stages that share weights,
        setting up the necessary communication groups for synchronization.
        """

        def get_name(shape):
            """Convert a shape tuple to a string for easy comparison."""
            return str(shape).replace(".", "_")

        # Identify and register tied modules
        self.tied_modules = nn.ModuleDict()
        for m in self.modules():
            if (
                hasattr(m, "weight")
                and m.weight is not None
                and hasattr(m.weight, "shape")
            ):

                name = get_name(m.weight.shape)
                if name in self.tied_modules_keys:
                    if name in self.tied_modules:
                        # Tie to existing module
                        m.weight = self.tied_modules[name].weight
                    else:
                        # Register new tied module
                        self.tied_modules[name] = m
                        self.tied_weight_attrs[name] = "weight"

        # Create communication groups for tied weights
        tied_comms = {}
        if mpu.get_pipe_parallel_rank() == 1:
            return

        for key in self.tied_modules_keys:
            for dp in range(mpu.get_pipe_parallel_world_size()):
                for mp in range(mpu.get_model_parallel_world_size()):
                    tied_ranks = []
                    for s in sorted(self.tied_stage[key]):
                        if mpu.get_model_parallel_world_size() > 1:
                            tied_ranks.append(
                                self._grid.stage_to_global(
                                    stage_id=s, data=dp, model=mp
                                )
                            )
                        else:
                            tied_ranks.append(
                                self._grid.stage_to_global(stage_id=s, data=dp)
                            )

                    group = dist.new_group(ranks=tied_ranks)
                    if self.global_rank in tied_ranks:
                        tied_comms[key] = {
                            "ranks": tied_ranks,
                            "group": group,
                            "weight_attr": self.tied_weight_attrs[key],
                            "module": self.tied_modules[key],
                        }

        self.tied_comms = tied_comms
        # Ensure all processes have the same initial tied weights
        self._synchronize_tied_weights()

    def forward(
        self, forward_input: Union[List[torch.Tensor], Tuple[torch.Tensor]]
    ) -> Tuple[torch.Tensor]:
        """Forward pass through the pipeline stage.

        Args:
            forward_input: Input tensor or tuple of tensors for the current stage

        Returns:
            Tuple[torch.Tensor]: Output tensor or tuple of tensors from the current stage
        """
        # We need to offset the seed by the microbatch ID. Save it in a local
        # var to ensure it is preserved in the closure. Otherwise checkpointed
        # forward funcs will see a different offset.
        self.micro_offset += 1

        def exec_range_func(start: int, end: int):
            """Helper function to be used with checkpoint()
            Adapted from torch.utils.checkpoint:checkpoint_sequential()
            """
            local_micro_offset = self.micro_offset + 1

            def exec_func(*inputs):
                # Single tensor inputs need to be unwrapped
                if len(inputs) == 1:
                    inputs = inputs[0]
                for idx, layer in enumerate(self.forward_funcs[start:end]):
                    curr_layer = idx + self._local_start
                    if self.seed_layers:
                        new_seed = (self.base_seed * local_micro_offset) + curr_layer
                        if self.seed_fn:
                            self.seed_fn(new_seed)
                        else:
                            module_utils.set_random_seed(new_seed)
                    with autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16 if self.args.bf16 else torch.float16,
                        enabled=self.args.half_precision_backend == "cuda_amp",
                    ):
                        if isinstance(inputs, tuple):
                            inputs = layer(*inputs)
                        else:
                            inputs = layer(inputs)
                return inputs

            return exec_func

        # Setup RNG context for model/sequence parallelism
        if self.model_parallel or self.sequence_parallel:
            rng_context = get_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            return self._execute_forward_pass(forward_input, exec_range_func)

    def _execute_forward_pass(self, forward_input, exec_range_func):
        """Execute the forward pass with appropriate activation checkpointing.

        Args:
            forward_input: Input to the forward pass
            exec_range_func: Function to execute a range of layers

        Returns:
            Output from the forward pass
        """
        # No activation checkpointing
        if self.act_ckpt_interval == 0:
            func = exec_range_func(0, len(self.forward_funcs))
            return func(forward_input)

        # Ratio-based activation checkpointing
        if (
            self.act_ckpt_ratio is not None
            and float(self.act_ckpt_ratio[self.stage_id]) != 1.0
        ):
            return self._execute_ratio_based_checkpointing(
                forward_input, exec_range_func
            )

        # Standard activation checkpointing
        return self._execute_standard_checkpointing(forward_input, exec_range_func)

    def _execute_ratio_based_checkpointing(self, forward_input, exec_range_func):
        """Execute forward pass with ratio-based activation checkpointing."""
        ac_num_layers = int(
            len(self.forward_funcs) * float(self.act_ckpt_ratio[self.stage_id])
        )

        # Simple implementation: checkpoint last ac_num_layers, execute first non_ac_layers
        non_ac_layers = len(self.forward_funcs) - ac_num_layers
        x = forward_input
        if not isinstance(x, tuple):
            x = (x,)
        x = exec_range_func(0, non_ac_layers)(*x)
        if not isinstance(x, tuple):
            x = (x,)
        x = self.act_ckpt_func(
            exec_range_func(non_ac_layers, len(self.forward_funcs)), *x
        )

        # Alternative implementation using checkpointable ranges
        next_checkpointable = self.first_checkpointable
        x = forward_input
        for start_idx, end_idx in self.checkpointable_idx:
            if next_checkpointable:
                if not isinstance(x, tuple):
                    x = (x,)
                if ac_num_layers <= 0:
                    x = exec_range_func(start_idx, end_idx)(*x)
                else:
                    layer_num = end_idx - start_idx
                    if ac_num_layers >= layer_num:
                        x = self.act_ckpt_func(exec_range_func(start_idx, end_idx), *x)
                    else:
                        x = self.act_ckpt_func(
                            exec_range_func(start_idx, start_idx + ac_num_layers),
                            *x,
                        )
                        if not isinstance(x, tuple):
                            x = (x,)
                        x = exec_range_func(start_idx + ac_num_layers, end_idx)(*x)
                    ac_num_layers -= layer_num
            else:
                if not isinstance(x, tuple):
                    x = (x,)
                x = exec_range_func(start_idx, end_idx)(*x)
            next_checkpointable = not next_checkpointable

        return x

    def _execute_standard_checkpointing(self, forward_input, exec_range_func):
        """Execute forward pass with standard interval-based activation checkpointing."""
        num_layers = len(self.forward_funcs)
        x = forward_input

        # List-based checkpoint intervals
        if isinstance(self.act_ckpt_interval, list):
            for i in range(len(self.act_ckpt_interval) - 1):
                start_idx = self.act_ckpt_interval[i]
                end_idx = self.act_ckpt_interval[i + 1]
                funcs = self.forward_funcs[start_idx:end_idx]

                if not isinstance(x, tuple):
                    x = (x,)
                if self._is_checkpointable(funcs, end_idx, num_layers):
                    x = self.act_ckpt_func(exec_range_func(start_idx, end_idx), *x)
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        # Fixed-interval checkpointing
        else:
            for start_idx in range(0, num_layers, self.act_ckpt_interval):
                end_idx = min(start_idx + self.act_ckpt_interval, num_layers)
                funcs = self.forward_funcs[start_idx:end_idx]

                if not isinstance(x, tuple):
                    x = (x,)
                if self._is_checkpointable(funcs, end_idx, num_layers):
                    x = self.act_ckpt_func(exec_range_func(start_idx, end_idx), *x)
                else:
                    x = exec_range_func(start_idx, end_idx)(*x)
        return x

    def _is_checkpointable(
        self, funcs: nn.Module, end_idx: int, num_layers: int
    ) -> bool:
        """Check if a set of functions/layers can be checkpointed.

        Args:
            funcs: The functions/layers to check
            end_idx: Ending index of the function range
            num_layers: Total number of layers in this stage

        Returns:
            bool: True if the functions can be checkpointed, False otherwise
        """
        # Don't checkpoint embedding layers
        for f in funcs:
            if isinstance(f, torch.fx.GraphModule):
                for _, m in f.named_modules():
                    if isinstance(m, (torch.nn.Embedding, VocabParallelEmbedding)):
                        return False

        # Check if functions have parameters
        params = [f.parameters() for f in funcs if isinstance(f, torch.nn.Module)]

        # Don't checkpoint final layers of the last stage
        not_final = self.stage_id < self.num_stages - 1 or end_idx != num_layers

        return any(len(list(p)) > 0 for p in params) and not_final
