# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

__all__ = ["LayerPartition"]

from typing import Any, Dict, List

import torch
import torch.nn as nn

from Merak import get_logger

from .. import mpu
from . import module_utils


class LayerPartition:
    """
    A class used to partition model layers across different stages in a pipeline.

    This class provides various methods to distribute model layers into different
    stages (pipes) based on different partitioning strategies like uniform,
    parameter count, custom, and autopipe.

    Attributes:
        args: MerakArguments containing configuration for partitioning.
        layers: List of model layers (GraphModules) to be partitioned.
        global_rank: Global rank of the current process.
        num_stages: Number of pipeline stages.
        stage_id: ID of the current stage.
    """

    def __init__(
        self,
        args: Any,
        layers: List[torch.fx.GraphModule],
        global_rank: int,
    ):
        """Initialize the LayerPartition instance.

        Args:
            args: MerakArguments containing partitioning configuration.
            layers: List of model layers (GraphModules) to be partitioned.
            global_rank: Global rank of the current process.
        """
        self.args = args
        self.layers = layers
        self.global_rank = global_rank
        self.num_stages = mpu.get_pipe_parallel_world_size()
        self.stage_id = mpu.get_pipe_parallel_rank()
        self.logger = get_logger("normal")

    def count_parameters(self) -> List[int]:
        """Count the number of trainable parameters in each layer.

        Returns:
            List of integers where each integer represents the number of
            trainable parameters in the corresponding layer.
        """
        param_counts = [0] * len(self.layers)
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.Module):
                params = filter(lambda p: p.requires_grad, layer.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
        return param_counts

    def uniform_partition(self) -> List[int]:
        """Partition layers uniformly across stages.

        Returns:
            List of integers representing the number of layers assigned to each stage.
        """
        num_layers = len(self.layers)
        parts = module_utils.partition_uniform(
            num_items=num_layers, num_parts=self.num_stages, use_ceil=True
        )
        return parts

    def floor_uniform_partition(self) -> List[int]:
        """Partition layers uniformly across stages using floor division.

        Returns:
            List of integers representing the number of layers assigned to each stage.
        """
        num_layers = len(self.layers)
        parts = module_utils.partition_uniform(
            num_items=num_layers, num_parts=self.num_stages, use_ceil=False
        )
        return parts

    def parameter_based_partition(self) -> List[int]:
        """Partition layers based on their parameter counts.

        This method distributes layers to balance the total number of parameters across stages.

        Returns:
            List of integers representing the number of layers assigned to each stage.
        """
        param_counts = self.count_parameters()
        parts = module_utils.partition_balanced(
            weights=param_counts, num_parts=self.num_stages
        )
        return parts

    def custom_partition(self) -> List[int]:
        """Partition layers based on a custom pattern defined in Merak arguments.

        Returns:
            List of integers representing the number of layers assigned to each stage.
        """
        parts = [int(i) for i in self.args.custom_partition[0].split(",")]
        return parts

    def autopipe_partition(self, input_to_shard_dict: Dict[str, int]) -> List[int]:
        """Partition layers based on automatic pipeline profiling.

        This method profiles each layer's computational cost (FLOPS) and
        distributes layers to balance computation across stages.

        Args:
            input_to_shard_dict: Dictionary containing input sharding information.

        Returns:
            List of integers representing the number of layers assigned to each stage.
        """
        from .profiler import FlopsProfiler

        # Prepare input for profiling
        input = self.layers[0].dummy_inputs
        if not isinstance(input, dict):
            input = [i.cpu() for i in input]
        dummy_input = input
        extra_inputs = {}
        for k, v in input_to_shard_dict.items():
            if v != 0:
                if v not in extra_inputs:
                    extra_inputs[v] = []
                extra_inputs[v].append(input.pop(k))

        # Profile each layer's FLOPS
        mflops_list = []
        for idx, layer in enumerate(self.layers):
            prof = FlopsProfiler(layer)
            prof.start_profile()
            if idx == 0 and isinstance(input, dict):
                input = layer(**input)
            else:
                if idx in extra_inputs:
                    input.extend(extra_inputs[idx])
                input = layer(*input)
            mflops = prof.get_total_flops() / 10**6  # Convert to million FLOPS
            mflops_list.append(round(mflops))
            prof.end_profile()

        if self.global_rank == 0:
            self.logger.info(f"Autopipe partitioning with mflops: {mflops_list}")

        # Perform autopipe partitioning
        from ...autopipe import pipeline

        parts = pipeline(
            forward_times=mflops_list,
            backward_times=[
                i * 3 for i in mflops_list
            ],  # Assume backward is 3x forward
            num_stages=self.num_stages,
        )
        parts.append(len(self.layers))  # Add the total number of layers
        return parts

    def get_partition(self, input_to_shard_dict: Dict[str, int]) -> List[int]:
        """Get the layer partition based on the specified method.

        Args:
            input_to_shard_dict: Dictionary containing input sharding information.

        Returns:
            List of integers representing the number of layers assigned to each stage.
        """
        if self.global_rank == 0:
            self.logger.info(f"Using partition method: {self.args.partition_method}")

        method = self.args.partition_method.lower()

        if method == "uniform":
            parts = self.uniform_partition()
        elif method == "uniform_floor":
            parts = self.floor_uniform_partition()
        elif method == "parameters":
            parts = self.parameter_based_partition()
        elif method == "custom":
            parts = self.custom_partition()
        elif method == "autopipe":
            parts = self.autopipe_partition(input_to_shard_dict)
        else:
            raise NotImplementedError(
                f"Partition method '{method}' is not implemented."
            )

        return parts
