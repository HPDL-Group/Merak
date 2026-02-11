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

from typing import Callable, List, Optional, Tuple, Type

import torch
import torch.fx
import torch.nn as nn
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    parallelize_module,
)

from Merak import get_logger

from .. import mpu
from ..pipeline.module_utils import set_random_seed
from .mp_attrs import (
    mp_is_setted,
    set_mp_attr,
    set_tp_layer_lists,
    get_col_para_list,
    get_row_para_list,
)
from .mp_layers import ColPara, ConvPara, RowPara
from .transformer_blocks import PipedGPT2Block, PipedGPT2Model
from .utils import (
    init_method_normal,
    reset_module_tensor,
    scaled_init_method_normal,
    tp_overlapping_available,
)


class ModuleRebuild:
    """Rebuild model to support model tensor parallel.

    This class provides functionality to convert a given model into a modelparallelized version.
    It supports various parallelism strategies including tensor model parallelism and pipelining.
    """

    def __init__(self, args, model: List[torch.fx.GraphModule], model_class: Callable):
        """Initialize the ModuleRebuild class.

        Args:
            args: Arguments containing model parallelism settings.
            model: List of torch.fx.GraphModule instances representing the model.
            model_class: The class of the model to be rebuilt.
        """
        self.args = args
        self.model = model if isinstance(model, list) else [model]
        self.model_class = model_class
        self.mp_size = mpu.get_model_parallel_world_size()
        self.init_method = init_method_normal(self.args.init_method_std)
        self.scaled_init_method = scaled_init_method_normal(
            self.args.init_method_std, self.args.num_layers
        )
        self.logger = get_logger("detailed")

        # Set random seed for reproducibility
        set_random_seed(self.args.seed)
        if self.args.tp_overlapping_level > 1:
            self.overlap_initialization()

        if self.mp_size > 1:
            self._set_attr()

    def _set_attr(self):
        if not mp_is_setted():
            from .mp_mapping import get_mp_layer_lists

            mp_layer_lists = get_mp_layer_lists(self.model_class)
            if mp_layer_lists is not None:
                set_tp_layer_lists(**mp_layer_lists)

        assert mp_is_setted(), (
            f"Model {self.model_class.__name__} is not supported by auto parallelism. "
            "Please set parallelism attributes manually using set_tp_layer_lists."
        )

        for model in self.model:
            set_mp_attr(model, self.mp_size)

    def build_linear_parallel_layer(self, module: nn.Linear) -> nn.Module:
        """Build parallel layers for linear (dense) layers.

        This function converts a standard linear layer into a parallelized version based on the module's attributes.
        It supports both row-wise and column-wise parallelism.

        Args:
            layer_name: Name of the layer.
            module: Linear module to be converted.

        Returns:
            nn.Module: The converted parallel linear module.
        """
        _bias = (
            module.bias
            if isinstance(module.bias, bool)
            else isinstance(module.bias, torch.Tensor)
        )
        if not hasattr(module, "mp_attr"):
            return module

        if module.mp_attr.startswith("row"):
            module_args = [module.in_features * self.mp_size, module.out_features]
            return RowPara(
                module_args[0],
                module_args[1],
                self.args,
                self.scaled_init_method,
                bias=_bias,
            )

        if module.mp_attr.startswith("col"):
            module_args = [module.in_features, module.out_features * self.mp_size]
            return ColPara(
                module_args[0],
                module_args[1],
                self.args,
                self.init_method,
                bias=_bias,
            )

    def build_conv2d_parallel_layer(self, module: nn.Conv2d) -> nn.Module:
        """Build parallel layers for 2D convolution layers.

        This function converts a standard 2D convolution layer into a parallelized version based on the module's attributes.

        Args:
            module: Conv2d module to be converted.

        Returns:
            nn.Module: The converted parallel Conv2d module or None if not applicable.
        """
        module_args = module.__dict__
        if not hasattr(module, "mp_attr"):
            return module
        return ConvPara(**module_args)

    def reset_meta_device_tensors(self, model: nn.Module) -> bool:
        """Reset tensors of 'meta' device to CPU and initialize them."""
        should_reinit = False
        for param_name, param in model.named_parameters():
            if str(param.device) == "meta":
                reset_module_tensor(
                    model, param_name, torch.device("cpu"), torch.zeros(param.shape)
                )
                should_reinit = True

        for buffer_name, buffer in model.named_buffers():
            if str(buffer.device) == "meta":
                reset_module_tensor(
                    model, buffer_name, torch.device("cpu"), torch.rand(buffer.shape)
                )

        return should_reinit

    def process_module(
        self,
        model: nn.Module,
        replace_type: Optional[Type[nn.Module]] = None,
        module_func: Optional[Callable] = None,
        get_args: Optional[Callable] = None,
        parallel_layer_names: Optional[list] = None,
    ) -> None:
        """
        Universal module processing function that supports both building parallel layers and replacing modules

        Args:
            model: Model to process
            build_parallel: Whether to build parallel layers
            replace_type: Module type to replace
            module_func: Function to create new modules
            get_args: Function to get module arguments
            parallel_layer_names: List of specific module names to build parallel layers for
        """
        if parallel_layer_names is None:
            parallel_layer_names = ["c_attn", "c_proj", "c_fc"]

        for name, module in model.named_children():

            if isinstance(module, nn.Linear) or name in parallel_layer_names:
                parallel_layer = self.build_linear_parallel_layer(module)
            elif isinstance(module, nn.Conv2d):
                parallel_layer = self.build_conv2d_parallel_layer(module)
            else:
                parallel_layer = None

            if parallel_layer is not None:
                setattr(model, name, parallel_layer)

            if (
                replace_type is not None
                and module_func is not None
                and get_args is not None
                and isinstance(module, replace_type)
                and str(module.weight.shape).replace(".", "_")
                in self.model.tied_modules_keys
            ):
                new_module = module_func(*get_args(module))
                setattr(model, name, new_module)

            if len(list(module.children())) > 0:
                self.process_module(
                    module,
                    replace_type,
                    module_func,
                    get_args,
                    parallel_layer_names,
                )

    def get_parallelize_plan(self):
        parallelize_plan = {}
        col_para_list = get_col_para_list()
        row_para_list = get_row_para_list()
        for n, _ in self.model.named_modules():
            for col in col_para_list:
                if col in n:
                    parallelize_plan[n] = ColwiseParallel()
            for row in row_para_list:
                if row in n:
                    parallelize_plan[n] = RowwiseParallel()
        return parallelize_plan

    def recover_module(self, module: nn.Module) -> bool:
        """Recover the module hierarchy and replace with parallel layers."""
        self.model = module
        # Traverse and build parallel layers
        if self.args.pytorch_tp:
            assert mpu.get_pipe_parallel_world_size() == 1
            assert hasattr(self.args.get_communication_grid(), "tp_mesh")
            parallelize_plan = self.get_parallelize_plan()
            tp_mesh = self.args.get_communication_grid().tp_mesh
            parallelize_module(self.model, tp_mesh, parallelize_plan=parallelize_plan)
        else:
            self.process_module(module)

        # Reset meta device tensors
        should_reinit = self.reset_meta_device_tensors(module)
        return should_reinit

    def vocab_parallel(self, emb_dim: int) -> None:
        """Configure vocabulary parallel settings."""
        if self.args.parallel_vocab and self.mp_size > 1:
            # Replace embedding layers with vocabulary parallel versions
            self.process_module(
                self.model,  # Assuming self.model is a list with one element
                torch.nn.Embedding,
                mpu.VocabParallelEmbedding,
                lambda x: (
                    x.weight.size(0),
                    x.weight.size(1),
                    self.args,
                    self.init_method,
                ),
            )

            # Replace linear layers with standard column parallel versions
            self.process_module(
                self.model,
                torch.nn.Linear,
                mpu.ColumnParallelLinear,
                lambda x: (
                    x.in_features,
                    x.out_features,
                    self.args,
                    x.bias is not None,
                    False,
                    torch.nn.init.xavier_normal_,
                ),
            )

            # Update tied modules keys
            keys_mapping = {
                str(i).replace(".", "_"): f"torch_Size([{i[0]//self.mp_size}, {i[1]}])"
                for i in emb_dim
            }
            self.model.tied_modules_keys = set(keys_mapping.values())
            new_tied_dic = {
                keys_mapping[i]: self.model.tied_stage[i] for i in self.model.tied_stage
            }
            self.model.tied_stage = new_tied_dic

        if self.args.tp_overlapping_level > 1:
            first = True
            for n, m in self.model.named_modules():
                if isinstance(m, PipedGPT2Block):
                    last = m
                    if first:
                        m.is_first_layer = True
                        first = False
            last.is_last_layer = True

    def overlap_initialization(self) -> None:
        """Handle overlapping initialization for tensor parallelism."""
        if not tp_overlapping_available(self.model_class):
            message = (
                f"Overlapping level {self.args.tp_overlapping_level} is not supported for "
                f"model {self.model_class.__name__}. Resetting to level 1."
            )
            self.logger.warning(message)
            self.args.tp_overlapping_level = 1
        else:
            # Ensure certain configurations are set for overlapping to work
            assert (
                not self.model.config.use_cache
            ), "Cache should be disabled for overlapping."
            assert (
                not self.model.config.output_attentions
            ), "Output attentions should be disabled for overlapping."
            assert (
                not self.model.config.output_hidden_states
            ), "Output hidden states should be disabled for overlapping."
            assert (
                not self.model.config.add_cross_attention
            ), "Cross attention should be disabled for overlapping."
            setattr(
                self.model.config,
                "tp_overlapping_level",
                self.args.tp_overlapping_level,
            )
            self.model.transformer = PipedGPT2Model(self.model.config)
