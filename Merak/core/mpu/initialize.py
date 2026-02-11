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

# Parts of the code here are adapted from https://github.com/NVIDIA/Megatron-LM/blob/806422e5ec35c27b027dbb413b05e27b6590dc56/megatron/mpu/initialize.py

"""Model and data parallel groups."""

import os

import torch
import torch.distributed as dist

from .topology import (
    PipeDataSequenceModelParallelTopology,
    PipelineParallelGrid,
    MeshParallelGrid,
)

# Model parallel group that the current rank belongs to.
_MODEL_PARALLEL_GROUP = None
# Data parallel group that the current rank belongs to.
_DATA_PARALLEL_GROUP = None
# Pipeline parallel group that the current rank belongs to.
_PIPE_PARALLEL_GROUP = None
# Sequence parallel group that the current rank belongs to.
_SEQUENCE_PARALLEL_GROUP = None

# These values enable us to change the mpu sizes on the fly.
_MPU_WORLD_SIZE = None
_MPU_RANK = None

# for nccl send & recv p2p communication
_PP_PREV_RANK = None
_PP_NEXT_RANK = None


def get_model_parallel_group():
    """Get the model parallel group the caller rank belongs to."""
    assert _MODEL_PARALLEL_GROUP is not None, "model parallel group is not initialized"
    return _MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get the data parallel group the caller rank belongs to."""
    assert _DATA_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _DATA_PARALLEL_GROUP


def set_model_parallel_world_size(world_size):
    """Set the model parallel size"""
    global _MPU_WORLD_SIZE
    _MPU_WORLD_SIZE = world_size


def get_model_parallel_world_size():
    """Return world size for the model parallel group."""
    global _MPU_WORLD_SIZE
    if _MPU_WORLD_SIZE is not None:
        return _MPU_WORLD_SIZE
    return torch.distributed.get_world_size(group=get_model_parallel_group())


def set_model_parallel_rank(rank):
    """Set model parallel rank."""
    global _MPU_RANK
    _MPU_RANK = rank


def get_model_parallel_rank():
    """Return my rank for the model parallel group."""
    global _MPU_RANK
    if _MPU_RANK is not None:
        return _MPU_RANK
    return torch.distributed.get_rank(group=get_model_parallel_group())


def get_model_parallel_src_rank():
    """Calculate the global rank corresponding to a local rank zeor
    in the model parallel group."""
    global_rank = torch.distributed.get_rank()
    local_world_size = get_model_parallel_world_size()
    return (global_rank // local_world_size) * local_world_size


def get_data_parallel_world_size():
    """Return world size for the data parallel group."""
    return torch.distributed.get_world_size(group=get_data_parallel_group())


def get_data_parallel_rank():
    """Return my rank for the data parallel group."""
    return torch.distributed.get_rank(group=get_data_parallel_group())


# def get_topology():
#     return _MPU_TOPOLOGY


def get_pipe_parallel_group():
    """Get the pipe parallel group the caller rank belongs to."""
    assert _PIPE_PARALLEL_GROUP is not None, "data parallel group is not initialized"
    return _PIPE_PARALLEL_GROUP


def get_pipe_parallel_rank():
    """Return my rank for the pipe parallel group."""
    return torch.distributed.get_rank(group=get_pipe_parallel_group())


def get_pipe_parallel_world_size():
    """Return world size for the pipe parallel group."""
    return torch.distributed.get_world_size(group=get_pipe_parallel_group())


def set_pipeline_model_parallel_prev_rank(prev_rank):
    global _PP_PREV_RANK
    _PP_PREV_RANK = prev_rank


def set_pipeline_model_parallel_next_rank(next_rank):
    global _PP_NEXT_RANK
    _PP_NEXT_RANK = next_rank


def get_pipeline_model_parallel_prev_rank():
    global _PP_PREV_RANK
    return _PP_PREV_RANK


def get_pipeline_model_parallel_next_rank():
    global _PP_NEXT_RANK
    return _PP_NEXT_RANK


def destroy_model_parallel():
    """Set the groups to none."""
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = None
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = None


def set_pipe_parallel_group(group):
    global _PIPE_PARALLEL_GROUP
    _PIPE_PARALLEL_GROUP = group


def set_model_parallel_group(group):
    global _MODEL_PARALLEL_GROUP
    _MODEL_PARALLEL_GROUP = group


def set_data_parallel_group(group):
    global _DATA_PARALLEL_GROUP
    _DATA_PARALLEL_GROUP = group


def is_pipeline_first_stage():
    return get_pipe_parallel_rank() == 0


def is_pipeline_last_stage():
    return get_pipe_parallel_rank() == get_pipe_parallel_world_size() - 1


def get_sequence_parallel_group():
    """Get the sequence parallel group the caller rank belongs to."""
    assert (
        _SEQUENCE_PARALLEL_GROUP is not None
    ), "sequence parallel group is not initialized"
    return _SEQUENCE_PARALLEL_GROUP


def get_sequence_parallel_rank():
    """Return my rank for the sequence parallel group."""
    return torch.distributed.get_rank(group=get_sequence_parallel_group())


def get_sequence_parallel_world_size():
    """Return world size for the sequence parallel group."""
    return torch.distributed.get_world_size(group=get_sequence_parallel_group())


def set_sequence_parallel_group(group):
    global _SEQUENCE_PARALLEL_GROUP
    _SEQUENCE_PARALLEL_GROUP = group


def set_topo_grid_communication():
    # we init topology and communication grid here
    # Get parallelism dim from enviroment
    dp = int(os.environ.get("DP"))
    pp = int(os.environ.get("PP"))
    tp = int(os.environ.get("TP"))
    sp = int(os.environ.get("SP"))

    try:
        # pylint: disable=import-outside-toplevel
        from torch.distributed.device_mesh import init_device_mesh

        device_mesh = init_device_mesh(
            "cuda", (pp, dp, sp, tp), mesh_dim_names=("pp", "dp", "sp", "tp")
        )

        dp_mesh = device_mesh["dp"]
        pp_mesh = device_mesh["pp"]
        tp_mesh = device_mesh["tp"]
        sp_mesh = device_mesh["sp"]
        mesh_dict = {"dp": dp_mesh, "pp": pp_mesh, "tp": tp_mesh, "sp": sp_mesh}
        communication_grid = MeshParallelGrid(mesh_dict=mesh_dict)
    except ImportError:
        topo = PipeDataSequenceModelParallelTopology(
            num_pp=pp, num_dp=dp, num_sp=sp, num_mp=tp
        )
        communication_grid = PipelineParallelGrid(topo)

    set_data_parallel_group(communication_grid.get_data_parallel_group())
    set_model_parallel_group(communication_grid.get_slice_parallel_group())
    set_pipe_parallel_group(communication_grid.get_pipe_parallel_group())
    set_sequence_parallel_group(communication_grid.get_sequence_parallel_group())

    return communication_grid
