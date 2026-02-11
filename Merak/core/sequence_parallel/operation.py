# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Maintainer: daocaoren360 (ouyangshun@foxmail.com)
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

from typing import Any, Dict, Tuple, Type

from torch import nn

from Merak.utils import init_empty_weights

from .modeling import MODELS_SP_CONFIG


def _get_sub_module_for_replaced(
    raw_model: nn.Module,
) -> Dict[Type[nn.Module], Type[nn.Module]]:
    """
    Get the module replacement mapping for sequence parallelism based on model class.

    This function looks up the model's class name in the sequence parallelism configuration
    to determine which modules should be replaced with their sequence-parallel counterparts.

    Args:
        raw_model: The original PyTorch model instance.

    Returns:
        Dictionary mapping original module classes to their sequence-parallel replacements.
        Returns empty dict if no configuration is found for the model.

    Example:
        >>> model = SomeTransformerModel()
        >>> replacements = _get_sub_module_for_replaced(model)
        >>> # Returns {LlamaDecoderLayer: LlamaDecoderLayerSP}
    """
    models_cls_name = raw_model.__class__.__name__

    for model_type, models_sp_info in MODELS_SP_CONFIG.items():
        if model_type in models_cls_name:
            supported_models = models_sp_info["models"]
            if models_cls_name in supported_models:
                return models_sp_info["sub_module_replacement"]

    return {}


def replace_to_sp_module(pipe_model: nn.Module, raw_model: nn.Module) -> None:
    """
    Replace specified modules in the pipeline model with sequence-parallel versions.

    This function recursively traverses the model hierarchy and replaces modules
    that have sequence-parallel equivalents as defined in the configuration.

    Args:
        pipe_model: The pipeline model where replacements will be applied.
        raw_model: The original model used to determine replacement configuration.

    Raises:
        AssertionError: If the raw model lacks a configuration or if no sequence
                       parallelism configuration is found for the model type.

    Example:
        >>> original_model = SomeTransformerModel()
        >>> pipeline_model = create_pipeline_model(original_model)
        >>> replace_to_sp_module(pipeline_model, original_model)
    """
    raw_model_config = raw_model.config if hasattr(raw_model, "config") else None
    assert raw_model_config, (
        f"Model {raw_model.__class__.__name__} requires a configuration "
        "to enable sequence parallelism"
    )

    sub_module_replacement_dict = _get_sub_module_for_replaced(raw_model)
    assert sub_module_replacement_dict, (
        f"Model {raw_model.__class__.__name__} does not support "
        "sequence parallelism in Merak"
    )

    def _replace_module_recursive(
        model: nn.Module,
        config: Any,
        raw_module_cls: Type[nn.Module],
        sp_module_cls: Type[nn.Module],
    ) -> None:
        """
        Recursively replace modules of raw_module_cls type with sp_module_cls.

        Args:
            model: Current module being processed.
            config: Model configuration for initializing sequence-parallel modules.
            raw_module_cls: Original module class to be replaced.
            sp_module_cls: Sequence-parallel module class replacement.
        """
        for name, module in model.named_children():
            if isinstance(module, raw_module_cls):
                with init_empty_weights():
                    # Initialize with meta device for consistent parameter initialization
                    layer_idx = getattr(module.self_attn, "layer_idx", None)
                    layer_sp = sp_module_cls(config, layer_idx)
                setattr(model, name, layer_sp)

            # Recursively process child modules
            if len(list(module.children())) > 0:
                _replace_module_recursive(module, config, raw_module_cls, sp_module_cls)

    for raw_module_cls, sp_module_cls in sub_module_replacement_dict.items():
        _replace_module_recursive(
            pipe_model, raw_model_config, raw_module_cls, sp_module_cls
        )


def get_leaf_modules_for_sp(raw_model: nn.Module) -> Tuple[Type[nn.Module], ...]:
    """
    Get the leaf module classes that can be replaced for sequence parallelism.

    This function returns the module classes from the original model that have
    sequence-parallel equivalents available for replacement.

    Args:
        raw_model: The original PyTorch model instance.

    Returns:
        Tuple of module classes that can be replaced with sequence-parallel versions.
        Returns empty tuple if no sequence parallelism configuration is available.

    Example:
        >>> model = SomeTransformerModel()
        >>> sp_modules = get_leaf_modules_for_sp(model)
        >>> # Returns (OriginalAttention, OriginalMLP)
    """
    sub_module_replacement_dict = _get_sub_module_for_replaced(raw_model)
    return (
        tuple(sub_module_replacement_dict.keys()) if sub_module_replacement_dict else ()
    )
