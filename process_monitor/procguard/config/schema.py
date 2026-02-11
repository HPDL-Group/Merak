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
Configuration Schema - Data classes defining the configuration structure for ProcGuard.

This module contains dataclasses that define the configuration schema for
ProcGuard, including worker configuration, monitoring settings, recovery options,
and PyTorch distributed training settings.

Classes:
- WorkerConfig: Configuration for individual worker processes
- MonitoringConfig: Health monitoring and heartbeat settings
- RecoveryConfig: Automatic recovery behavior settings
- CommunicationConfig: Inter-process communication settings
- LoggingConfig: Logging configuration
- StateConfig: State persistence settings
- WebConfig: Web server settings
- SLURMConfig: SLURM cluster integration settings
- PyTorchDistConfig: PyTorch distributed training settings
- ProcGuardConfig: Root configuration container

Example:
    >>> from procguard.config.schema import WorkerConfig, ProcGuardConfig
    >>> worker = WorkerConfig(
    ...     worker_id="trainer_1",
    ...     command="python train.py",
    ...     max_restarts=5
    ... )
    >>> config = ProcGuardConfig(workers=[worker])
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import timedelta


@dataclass
class WorkerConfig:
    """
    Configuration for an individual worker process.

    Defines how a worker process should be started, monitored, and managed
    including command, environment, health checks, and restart behavior.

    Attributes:
        worker_id: Unique identifier for this worker
        command: Command to execute for this worker
        working_dir: Working directory for the worker process
        env: Environment variables to pass to the worker
        pid_file: Path to PID file for tracking the worker
        worker_type: Type of worker (e.g., 'default', 'trainer', 'evaluator')
        max_restarts: Maximum number of restart attempts
        restart_delay: Delay in seconds between restart attempts
        health_check_url: URL for HTTP health check endpoint
    """
    worker_id: str
    command: str
    working_dir: Optional[str] = None
    env: Dict[str, str] = field(default_factory=dict)
    pid_file: Optional[str] = None
    worker_type: str = "default"
    max_restarts: int = 3
    restart_delay: float = 5.0
    health_check_url: Optional[str] = None


@dataclass
class MonitoringConfig:
    """
    Configuration for health monitoring and heartbeat detection.

    Settings for monitoring worker health including check intervals,
    timeout thresholds, and resource usage thresholds.

    Attributes:
        interval: Interval between health checks in seconds
        heartbeat_timeout: Timeout in seconds before marking worker as unresponsive
        zombie_detection: Whether to detect zombie processes
        cpu_threshold: CPU usage threshold (0.0-1.0) for healthy status
        memory_threshold: Memory usage threshold (0.0-1.0) for healthy status
    """
    interval: float = 1.0
    heartbeat_timeout: float = 10.0
    zombie_detection: bool = True
    cpu_threshold: float = 0.1
    memory_threshold: float = 0.1


@dataclass
class RecoveryConfig:
    """
    Configuration for automatic recovery behavior.

    Settings controlling how ProcGuard handles worker failures including
    automatic recovery, task reassignment, and recovery timeouts.

    Attributes:
        enable_auto_recovery: Whether to automatically recover failed workers
        stop_all_on_failure: Whether to stop all workers on any failure
        task_reassignment: Whether to reassign tasks from failed workers
        recovery_timeout: Maximum time in seconds for recovery operations
        max_recovery_attempts: Maximum recovery attempts per worker
    """
    enable_auto_recovery: bool = True
    stop_all_on_failure: bool = True
    task_reassignment: bool = True
    recovery_timeout: float = 60.0
    max_recovery_attempts: int = 3


@dataclass
class CommunicationConfig:
    """
    Configuration for inter-process communication.

    Settings for the communication adapter used between ProcGuard components.

    Attributes:
        adapter_type: Type of communication adapter ('mock', 'zmq', etc.)
        adapter_config: Additional configuration for the adapter
    """
    adapter_type: str = "mock"
    adapter_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoggingConfig:
    """
    Configuration for logging behavior.

    Settings for log output including level, file destination, and format.

    Attributes:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file: Optional path to log file
        format: Log message format string
    """
    level: str = "INFO"
    file: Optional[str] = None
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class StateConfig:
    """
    Configuration for state persistence.

    Settings for saving and loading ProcGuard state including file path
    and auto-save behavior.

    Attributes:
        state_file: Path to state persistence file
        auto_save: Whether to automatically save state changes
        save_interval: Interval in seconds for auto-saving state
    """
    state_file: str = "procguard_state.json"
    auto_save: bool = True
    save_interval: float = 5.0


@dataclass
class WebConfig:
    """
    Configuration for the built-in web server.

    Settings for the ProcGuard web interface server.

    Attributes:
        enabled: Whether to enable the web server
        host: Host address to bind to
        port: Port number to listen on
    """
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 5000


@dataclass
class SLURMConfig:
    """
    Configuration for SLURM cluster integration.

    Settings for running ProcGuard in SLURM cluster environments including
    GPU allocation, worker ID formatting, and sbatch template.

    Attributes:
        enabled: Whether to enable SLURM integration
        gpu_count_per_node: Number of GPUs available per node
        master_port: Port for PyTorch distributed training master
        worker_id_format: Format string for worker ID generation
        auto_detect: Whether to auto-detect SLURM environment
        sbatch_template: Optional custom sbatch script template
    """
    enabled: bool = False
    gpu_count_per_node: int = 1
    master_port: int = 29500
    worker_id_format: str = "{hostname}-{local_rank}"
    auto_detect: bool = True
    sbatch_template: Optional[str] = None


@dataclass
class PyTorchDistConfig:
    """
    Configuration for PyTorch distributed training.

    Settings for PyTorch distributed training environment variables and
    initialization parameters.

    Attributes:
        enabled: Whether to enable distributed training support
        backend: Distributed training backend ('nccl', 'gloo', etc.)
        init_method: Initialization method ('env', 'tcp', 'file')
        master_addr: Address of the master process
        master_port: Port for master process communication
        local_rank_env_var: Environment variable for local rank
        rank_env_var: Environment variable for global rank
        world_size_env_var: Environment variable for world size
    """
    enabled: bool = False
    backend: str = "nccl"
    init_method: str = "env"
    master_addr: Optional[str] = None
    master_port: int = 29500
    local_rank_env_var: str = "LOCAL_RANK"
    rank_env_var: str = "RANK"
    world_size_env_var: str = "WORLD_SIZE"


@dataclass
class ProcGuardConfig:
    """
    Root configuration container for ProcGuard.

    This is the main configuration class that aggregates all other configuration
    classes including workers, monitoring, recovery, and integration settings.

    Attributes:
        workers: List of WorkerConfig instances
        monitoring: MonitoringConfig for health monitoring
        recovery: RecoveryConfig for failure recovery
        communication: CommunicationConfig for IPC
        logging: LoggingConfig for log settings
        state: StateConfig for state persistence
        web: WebConfig for web server
        slurm: SLURMConfig for SLURM integration
        pytorch_dist: PyTorchDistConfig for distributed training

    Example:
        >>> from procguard.config.schema import ProcGuardConfig, WorkerConfig
        >>> config = ProcGuardConfig()
        >>> config.workers = [
        ...     WorkerConfig(worker_id="w1", command="python train.py")
        ... ]
        >>> config.monitoring.interval = 0.5
    """
    workers: List[WorkerConfig] = field(default_factory=list)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    state: StateConfig = field(default_factory=StateConfig)
    web: WebConfig = field(default_factory=WebConfig)
    slurm: SLURMConfig = field(default_factory=SLURMConfig)
    pytorch_dist: PyTorchDistConfig = field(default_factory=PyTorchDistConfig)
