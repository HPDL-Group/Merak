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

__all__ = ['LayerPartition']

import torch
import torch.nn as nn
from typing import List, Dict

from . import module_utils
from ..printer import logger
from ..mpu.topology import ProcessTopology
from ...merak_args import MerakArguments

class LayerPartition:

    def __init__(
            self,
            args: MerakArguments,
            layers: List[torch.fx.GraphModule],
            topo: ProcessTopology,
            global_rank: int
        ):
        self.args = args
        self.layers = layers
        self._topo = topo
        self.global_rank = global_rank

        self.num_stages = self._topo.get_dim('pipe')
        self.stage_id = self._topo.get_coord(self.global_rank).pipe

    def _count_layer_params(self):
        """Count the trainable parameters in individual layers.

        This routine will only build one layer at a time.

        Returns:
            A list of the number of parameters in each layer.
        """
        param_counts = [0] * len(self.layers)
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.Module):
                params = filter(lambda p: p.requires_grad, layer.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
        return param_counts

    # Each stage gets a simple uniform number of layers.
    def uniform(self) -> List[int]:
        num_layers = len(self.layers)
        parts = module_utils.partition_uniform(num_items=num_layers,
                                                num_parts=self.num_stages, 
                                                use_ceil=True)
        return parts

    def uniform_floor(self) -> List[int]:
        num_layers = len(self.layers)
        parts = module_utils.partition_uniform(num_items=num_layers,
                                                num_parts=self.num_stages, 
                                                use_ceil=False)
        return parts

    def _count_layer_params(self) -> List[int]:
        """Count the trainable parameters in individual layers.

        This routine will only build one layer at a time.

        Returns:
            A list of the number of parameters in each layer.
        """
        param_counts = [0] * len(self.layers)
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, nn.Module):
                params = filter(lambda p: p.requires_grad, layer.parameters())
                param_counts[idx] = sum(p.numel() for p in params)
        return param_counts

    def parameters(self) -> List[int]:
        param_counts = self._count_layer_params()
        # print(param_counts)
        parts = module_utils.partition_balanced(weights=param_counts,
                                                num_parts=self.num_stages)
        return parts

    def custom_layer_idx(self) -> List[int]:
        parts = [int(i) for i in self.args.custom_partition[0].split(",")]
        return parts
        
    def autopipe(self, input_to_shard_dic: Dict[str, int]) -> List[int]:
            
        # experimental partition method
        from .profiler import FlopsProfiler
        mflops_list = []
        input = self.layers[0].dummy_inputs
        if not isinstance(input, dict):
            input = [i.cpu() for i in input]
        self.dummy_input = input 
        extra_inputs = {}
        for k, v in input_to_shard_dic.items():
            if v != 0:
                if v not in extra_inputs:
                    extra_inputs[v] = []
                extra_inputs[v].append(input.pop(k))
        for idx, layer in enumerate(self.layers):
            prof = FlopsProfiler(layer)
            prof.start_profile()
            if idx == 0 and isinstance(input, dict):
                input = layer(**input)
            else:
                if idx in extra_inputs:
                    input.extend(extra_inputs[idx])
                input = layer(*input)
            flops = prof.get_total_flops()
            # if dist.get_rank() == 0:
            #     prof.print_model_profile()
            # print_rank_0(flops)
            mflops_list.append(round(flops / 10.0**6))
            prof.end_profile()
        if self.global_rank == 0:
            logger.info(f'Using experimental autopipe partition, mflops list: {mflops_list}')
        from ...autopipe import pipeline
        #  input is forward_time, backward_time, pipeline_stage
        parts = pipeline(mflops_list, [i*3 for i in mflops_list], self.num_stages)
        parts.append(len(self.layers))

        return parts
    
    def layer_split_by_index(self, input_to_shard_dic: Dict[str, int]) -> List[int]:
        # Each stage gets a simple uniform number of layers.
        if self.global_rank == 0:
            logger.info(f'Partitioning pipeline stages'
                        f' with method {self.args.partition_method}')

        method = self.args.partition_method.lower()

        if method == 'uniform':
            parts = self.uniform()
        elif method == 'uniform_floor':
            parts = self.uniform_floor()
        elif method == 'parameters':
            parts = self.parameters()
        elif method == 'custom':
            parts = self.custom_layer_idx()
        elif method == 'autopipe':
            parts = self.autopipe(input_to_shard_dic)
        else:
            raise NotImplementedError(
                f'Partitioning method {method} \
                    not implemented.')
        
        return parts