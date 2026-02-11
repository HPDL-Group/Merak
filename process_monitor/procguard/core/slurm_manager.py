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

"""
SLURM Manager - Handles SLURM cluster environment and PyTorch distributed training support.

This module provides comprehensive support for SLURM (Simple Linux Utility for Resource
Management) cluster environments, including automatic environment detection, worker ID
generation, and PyTorch distributed training environment variable configuration.

Features:
- Automatic detection of SLURM environment and extraction of relevant information
- Generation of SLURM-compliant worker IDs
- Automatic construction of PyTorch distributed training environment variables
- Support for multi-node, multi-GPU training configurations

Example:
    >>> manager = SLURMManager(gpu_count_per_node=4)
    >>> env_info = manager.detect_environment()
    >>> if env_info.is_slurm:
    ...     pytorch_env = manager.build_pytorch_dist_env()
"""

import os
import socket
import logging
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


@dataclass
class SLURMEnvInfo:
    """
    Data class representing SLURM environment information.

    Contains all relevant environment variables and detected information
    from the SLURM job scheduler.

    Attributes:
        job_id: SLURM job ID
        job_name: Name of the SLURM job
        nodelist: List of nodes allocated to the job
        nnodes: Number of nodes allocated
        ntasks: Total number of tasks in the job
        ntasks_per_node: Number of tasks per node
        cpus_per_task: Number of CPUs allocated per task
        gpu_count: Number of GPUs per node
        local_rank: Local rank of the current process
        rank: Global rank of the current process
        world_size: Total number of processes
        hostname: Hostname of the current node
        is_slurm: Whether running in SLURM environment
    """
    job_id: Optional[str] = None
    job_name: Optional[str] = None
    nodelist: Optional[str] = None
    nnodes: int = 1
    ntasks: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 1
    gpu_count: int = 0
    local_rank: int = 0
    rank: int = 0
    world_size: int = 1
    hostname: str = field(default_factory=socket.gethostname)
    is_slurm: bool = False


@dataclass
class PyTorchDistEnv:
    """
    Data class representing PyTorch distributed training environment variables.

    Contains all environment variables required for PyTorch distributed
    training with either NCCL or gloo backends.

    Attributes:
        local_rank: Local rank of the current process
        rank: Global rank of the current process
        world_size: Total number of processes
        local_world_size: Number of processes per node
        gpu_count: Number of GPUs available
        master_addr: Address of the master process
        master_port: Port for master process communication
        backend: Distributed training backend (nccl, gloo, etc.)
        init_method: Initialization method (env, tcp, file)
    """
    local_rank: int = 0
    rank: int = 0
    world_size: int = 1
    local_world_size: int = 1
    gpu_count: int = 1
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    backend: str = "nccl"
    init_method: str = "env"

    def to_dict(self) -> Dict[str, str]:
        """
        Convert to environment variable dictionary.

        Generates environment variables suitable for PyTorch distributed
        training, including SLURM-specific variables when available.

        Returns:
            Dict[str, str]: Dictionary of environment variables
        """
        env_vars = {
            "LOCAL_RANK": str(self.local_rank),
            "RANK": str(self.rank),
            "WORLD_SIZE": str(self.world_size),
            "LOCAL_WORLD_SIZE": str(self.local_world_size),
            "GPU_COUNT": str(self.gpu_count),
            "MASTER_ADDR": self.master_addr,
            "MASTER_PORT": str(self.master_port),
            "NCCL_SOCKET_IFNAME": os.environ.get("NCCL_SOCKET_IFNAME", "ibs10"),
            "CUDA_VISIBLE_DEVICES": self._get_cuda_visible_devices(),
        }

        if "SLURM_JOB_ID" in os.environ:
            env_vars.update({
                "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", ""),
                "SLURM_JOB_NAME": os.environ.get("SLURM_JOB_NAME", ""),
                "SLURM_NODELIST": os.environ.get("SLURM_NODELIST", ""),
                "SLURM_NNODES": os.environ.get("SLURM_NNODES", ""),
                "SLURM_NTASKS": os.environ.get("SLURM_NTASKS", ""),
                "SLURM_NTASKS_PER_NODE": os.environ.get("SLURM_NTASKS_PER_NODE", ""),
                "SLURM_PROCID": os.environ.get("SLURM_PROCID", ""),
                "SLURM_LOCALID": os.environ.get("SLURM_LOCALID", ""),
                "SLURM_CPUS_PER_TASK": os.environ.get("SLURM_CPUS_PER_TASK", ""),
            })

        return env_vars

    def _get_cuda_visible_devices(self) -> str:
        """
        Generate CUDA_VISIBLE_DEVICES string based on local_rank.

        Returns:
            str: GPU device indices visible to the process
        """
        if self.gpu_count <= 0:
            return ""
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            return cuda_visible
        return str(self.local_rank % self.gpu_count)


class SLURMManager:
    """
    Manager for SLURM cluster environments and PyTorch distributed training.

    This class provides comprehensive utilities for working with SLURM
    job scheduler, including environment detection, worker ID generation,
    and PyTorch distributed training configuration.

    Attributes:
        gpu_count_per_node: Number of GPUs per node
        master_port: Default master port for PyTorch distributed training
        _slurm_env: Cached SLURM environment information
        _lock: Thread lock for thread-safe operations

    Examples:
        >>> manager = SLURMManager(gpu_count_per_node=4, master_port=29500)
        >>> env_info = manager.detect_environment()
        >>> if env_info.is_slurm:
        ...     dist_env = manager.build_pytorch_dist_env()
        ...     env_vars = manager.get_pytorch_env_vars()
    """

    def __init__(self, gpu_count_per_node: int = 1, master_port: int = 29500):
        """
        Initialize the SLURMManager.

        Args:
            gpu_count_per_node: Number of GPUs available per node
            master_port: Default port for PyTorch distributed training master
        """
        self.logger = logging.getLogger("procguard.slurm")
        self.gpu_count_per_node = gpu_count_per_node
        self.master_port = master_port

        self._slurm_env: Optional[SLURMEnvInfo] = None
        self._lock = threading.RLock()

    def detect_environment(self) -> SLURMEnvInfo:
        """
        Detect and return SLURM environment information.

        Checks for SLURM environment variables and extracts relevant
        information about the current job allocation.

        Returns:
            SLURMEnvInfo: Object containing detected environment information

        Note:
            Results are cached after first call for performance.
        """
        with self._lock:
            if self._slurm_env is not None:
                return self._slurm_env

            self._slurm_env = SLURMEnvInfo()

            if "SLURM_JOB_ID" in os.environ:
                self._slurm_env.is_slurm = True
                self._slurm_env.job_id = os.environ.get("SLURM_JOB_ID")
                self._slurm_env.job_name = os.environ.get("SLURM_JOB_NAME")
                self._slurm_env.nodelist = os.environ.get("SLURM_NODELIST")
                self._slurm_env.hostname = os.environ.get("SLURMD_NODENAME", socket.gethostname())

                try:
                    self._slurm_env.nnodes = int(os.environ.get("SLURM_NNODES", 1))
                    self._slurm_env.ntasks = int(os.environ.get("SLURM_NTASKS", 1))
                    self._slurm_env.ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))
                    self._slurm_env.cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to parse SLURM environment variable: {e}")

                try:
                    self._slurm_env.rank = int(os.environ.get("SLURM_PROCID", 0))
                    self._slurm_env.local_rank = int(os.environ.get("SLURM_LOCALID", 0))
                    self._slurm_env.world_size = self._slurm_env.nnodes * self._slurm_env.ntasks_per_node
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Failed to parse SLURM rank info: {e}")
                    self._slurm_env.rank = 0
                    self._slurm_env.local_rank = 0
                    self._slurm_env.world_size = self._slurm_env.nnodes * self._slurm_env.ntasks_per_node

                self._slurm_env.gpu_count = self.gpu_count_per_node

                self.logger.info(
                    f"Detected SLURM environment: job_id={self._slurm_env.job_id}, "
                    f"hostname={self._slurm_env.hostname}, "
                    f"rank={self._slurm_env.rank}/{self._slurm_env.world_size}, "
                    f"local_rank={self._slurm_env.local_rank}, "
                    f"gpu_count={self._slurm_env.gpu_count}"
                )
            else:
                self._slurm_env.hostname = socket.gethostname()
                self._slurm_env.gpu_count = self.gpu_count_per_node
                self._slurm_env.world_size = self.gpu_count_per_node
                self.logger.info(f"Non-SLURM environment, using hostname: {self._slurm_env.hostname}")

            return self._slurm_env

    def generate_worker_id(self, custom_format: Optional[str] = None) -> str:
        """
        Generate a unique worker ID based on the environment.

        Args:
            custom_format: Optional format string with placeholders:
                - {hostname}: Node hostname
                - {local_rank}: Local rank
                - {rank}: Global rank
                - {job_id}: SLURM job ID
                - {node_index}: Node index

        Returns:
            str: Generated worker ID
        """
        env_info = self.detect_environment()

        if custom_format:
            worker_id = custom_format.format(
                hostname=env_info.hostname.replace(".", "_"),
                local_rank=env_info.local_rank,
                rank=env_info.rank,
                job_id=env_info.job_id or "unknown",
                node_index=self._get_node_index(env_info.nodelist),
            )
        elif env_info.is_slurm:
            worker_id = f"{env_info.hostname}-{env_info.local_rank}"
        else:
            worker_id = f"{env_info.hostname}-{env_info.local_rank}"

        return worker_id

    @staticmethod
    def _extract_master_node(nodelist: Optional[str]) -> str:
        """
        Extract the master node from SLURM_NODELIST.

        Args:
            nodelist: SLURM_NODELIST string (may contain ranges)

        Returns:
            str: Hostname of the first/master node
        """
        if not nodelist:
            return ""

        nodes = nodelist.split(",")
        if not nodes:
            return ""

        first_node = nodes[0].strip()

        if "[" in first_node:
            import re
            prefix = first_node.split("[")[0]
            range_match = re.search(r'\[(\d+)-(\d+)\]', first_node)
            if range_match:
                start_num = range_match.group(1)
                num_len = len(start_num)
                return f"{prefix}{start_num.zfill(num_len)}"
            return prefix
        return first_node

    def _get_node_index(self, nodelist: Optional[str]) -> int:
        """
        Extract a node index from the nodelist.

        Args:
            nodelist: SLURM_NODELIST string

        Returns:
            int: Node index (0-999 based on hash)
        """
        if not nodelist:
            return 0

        try:
            if "[" in nodelist:
                nodelist = nodelist.split("[")[0]
            return hash(nodelist) % 1000
        except Exception:
            return 0

    def build_pytorch_dist_env(self, rank: Optional[int] = None) -> PyTorchDistEnv:
        """
        Build PyTorch distributed training environment configuration.

        Args:
            rank: Optional rank override (uses detected rank if not specified)

        Returns:
            PyTorchDistEnv: Environment configuration for PyTorch distributed training
        """
        env_info = self.detect_environment()

        if env_info.is_slurm:
            ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", self.gpu_count_per_node))
            world_size = env_info.nnodes * ntasks_per_node
        else:
            ntasks_per_node = self.gpu_count_per_node
            world_size = self.gpu_count_per_node

        dist_env = PyTorchDistEnv(
            local_rank=env_info.local_rank,
            rank=rank if rank is not None else env_info.rank,
            world_size=world_size,
            local_world_size=ntasks_per_node,
            gpu_count=self.gpu_count_per_node,
            master_port=self.master_port,
        )

        if rank is not None:
            dist_env.rank = rank

        if env_info.is_slurm:
            if env_info.rank == 0:
                dist_env.master_addr = env_info.hostname
            elif env_info.nodelist:
                master_node = self._extract_master_node(env_info.nodelist)
                if master_node:
                    dist_env.master_addr = master_node

        return dist_env

    def get_pytorch_env_vars(self, rank: Optional[int] = None) -> Dict[str, str]:
        """
        Get PyTorch distributed training environment variables.

        Args:
            rank: Optional rank override

        Returns:
            Dict[str, str]: Environment variables for PyTorch distributed training
        """
        dist_env = self.build_pytorch_dist_env(rank)
        return dist_env.to_dict()

    def get_slurm_vars_for_worker(self, worker_id: str) -> Dict[str, str]:
        """
        Get SLURM environment variables to pass to a worker.

        Args:
            worker_id: ID of the worker

        Returns:
            Dict[str, str]: SLURM environment variables for the worker
        """
        env_info = self.detect_environment()

        return {
            "SLURM_JOB_ID": env_info.job_id or "",
            "SLURM_JOB_NAME": env_info.job_name or "",
            "SLURM_NODELIST": env_info.nodelist or "",
            "SLURM_NNODES": str(env_info.nnodes),
            "SLURM_NTASKS": str(env_info.ntasks),
            "SLURM_NTASKS_PER_NODE": str(env_info.ntasks_per_node),
            "SLURM_PROCID": str(env_info.rank),
            "SLURM_LOCALID": str(env_info.local_rank),
            "SLURM_CPUS_PER_TASK": str(env_info.cpus_per_task),
            "SLURM_GPU_COUNT": str(env_info.gpu_count),
            "WORKER_ID": worker_id,
            "IS_SLURM": str(env_info.is_slurm).lower(),
        }

    def configure_nccl_for_slurm(self) -> Dict[str, str]:
        """
        Get NCCL configuration optimized for SLURM environment.

        Returns:
            Dict[str, str]: NCCL environment variables
        """
        env_info = self.detect_environment()

        config = {}

        if env_info.nodelist:
            config["NCCL_SOCKET_IFNAME"] = "ibs10"

        return config

    def get_sbatch_script_content(
        self,
        command: str,
        job_name: str = "procguard_training",
        nodes: int = 1,
        gpus_per_node: int = 1,
        ntasks_per_node: int = 1,
        cpus_per_task: int = 4,
        time_limit: str = "01:00:00",
        **extra_sbatch_options,
    ) -> str:
        """
        Generate SLURM sbatch script content.

        Args:
            command: Command to execute
            job_name: Name of the SLURM job
            nodes: Number of nodes
            gpus_per_node: GPUs per node
            ntasks_per_node: Tasks per node
            cpus_per_task: CPUs per task
            time_limit: Job time limit (HH:MM:SS)
            **extra_sbatch_options: Additional sbatch options

        Returns:
            str: Complete sbatch script content
        """
        sbatch_lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --nodes={nodes}",
            f"#SBATCH --ntasks={nodes * ntasks_per_node}",
            f"#SBATCH --ntasks-per-node={ntasks_per_node}",
            f"#SBATCH --cpus-per-task={cpus_per_task}",
            f"#SBATCH --gpus-per-node={gpus_per_node}",
            f"#SBATCH --time={time_limit}",
        ]

        for key, value in extra_sbatch_options.items():
            key = key.replace("_", "-")
            sbatch_lines.append(f"#SBATCH --{key}={value}")

        sbatch_lines.extend([
            "",
            "# Load required modules",
            "module load cuda/12.1 2>/dev/null || true",
            "",
            "# Set PyTorch distributed training environment variables",
            f"export CUDA_VISIBLE_DEVICES={','.join(str(i) for i in range(gpus_per_node))}",
            "# Print SLURM environment information",
            "echo '=== SLURM Environment ==='",
            "echo 'SLURM_JOB_ID: $SLURM_JOB_ID'",
            "echo 'SLURM_NODELIST: $SLURM_NODELIST'",
            "echo 'SLURM_PROCID: $SLURM_PROCID'",
            "echo 'SLURM_LOCALID: $SLURM_LOCALID'",
            "",
            "# Launch training command",
            f"srun --label --ntasks-per-node={ntasks_per_node} {command}",
        ])

        return "\n".join(sbatch_lines)

    def is_in_slurm(self) -> bool:
        """
        Check if running in a SLURM environment.

        Returns:
            bool: True if SLURM_JOB_ID environment variable is set
        """
        return "SLURM_JOB_ID" in os.environ
