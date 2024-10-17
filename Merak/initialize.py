# coding=utf-8
# Copyright (c) 2022, HPDL group, PDL lab, NUDT.  All rights reserved.
#
# Maintainer: TXacs (txacs1993@gmail.com)
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
    import torch_ft
except:
    pass

topo = None
communication_grid = None

def print_rank_0(message: str):
    """If distributed is initialized print only on rank 0."""
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def init(pp: int, tp: int, dp: int, backend: str = 'nccl'):
    """
    Initialized the distributed communication groups, include data parallel, 
    tensor model parallel and pipeline model parallel. Each parallel degree 
    has it own communication group, we can ge the rank or size through mpu API.

    Parameters:
    -   dp (int) -- Parallel degree of data parallelism.
    -   tp (int) -- Parallel degree of tensor model parallelism.
    -   pp (int) -- Parallel degree of pipeline model parallelism.
    """
    compile_config = torch.__config__.show().split(", ")
    if 'USE_NCCL=1' in compile_config or 'USE_NCCL=ON' in compile_config:
        backend = 'nccl'
    elif 'USE_MPI=1' in compile_config or 'USE_MPI=ON' in compile_config:
        backend = 'mpi'
    else:
        raise RuntimeError(f"Distributed package doesn't have NCCL/MPI built in")
    if not dist.is_initialized():
        dist.init_process_group(backend)
    # we init topology and communication grid here
    from .core.mpu.topology import (
        PipeModelDataParallelTopology,
        PipelineParallelGrid)
    global topo
    topo = PipeModelDataParallelTopology(num_pp=pp, num_mp=tp, num_dp=dp)
    global communication_grid
    communication_grid = PipelineParallelGrid(
                        topo,
                        dist.new_group(ranks=range(dist.get_world_size()))
                        )


    # set mpu for transformers model
    from .core.mpu.initialize import (
        set_data_parallel_group,
        set_model_parallel_group,
        set_pipe_parallel_group)

    set_data_parallel_group(communication_grid.get_data_parallel_group())
    set_model_parallel_group(communication_grid.get_slice_parallel_group())
    set_pipe_parallel_group(communication_grid.get_pipe_parallel_group())

    print_rank_0(f'Pipeline Model Parallel Size: {pp} \
                   \nTensor Model Parallel Size: {tp} \
                   \nData Parallel Size: {dp} \n')

def get_topo():
    global topo
    return topo

def get_grid():
    global communication_grid
    return communication_grid