# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: Swli (lucasleesw9@gmail.com)
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

# Parts of the code here are adapted from https://github.com/NVIDIA/Megatron-LM/blob/806422e5ec35c27b027dbb413b05e27b6590dc56/megatron/mpu/__init__.py

"""Model parallel utility interface."""

from .cross_entropy import vocab_parallel_cross_entropy
from .initialize import (
    destroy_model_parallel,
    get_data_parallel_group,
    get_data_parallel_rank,
    get_data_parallel_world_size,
    get_model_parallel_group,
    get_model_parallel_rank,
    get_model_parallel_src_rank,
    get_model_parallel_world_size,
    get_pipe_parallel_group,
    get_pipe_parallel_rank,
    get_pipe_parallel_world_size,
    get_pipeline_model_parallel_next_rank,
    get_pipeline_model_parallel_prev_rank,
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_world_size,
    is_pipeline_first_stage,
    is_pipeline_last_stage,
    set_model_parallel_rank,
    set_model_parallel_world_size,
    set_sequence_parallel_group,
    set_topo_grid_communication,
)
from .layers import (
    ColParallelConv2d,
    ColumnParallelLinear,
    LayerNorm,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from .mappings import (
    all_to_all_sequence_parallel_region,
    copy_to_model_parallel_region,
    gather_for_sequence_parallel_region,
    gather_from_model_parallel_region,
    reduce_from_model_parallel_region,
    scatter_to_model_parallel_region,
    split_for_sequence_parallel_region,
)
from .utils import divide, split_tensor_along_last_dim
