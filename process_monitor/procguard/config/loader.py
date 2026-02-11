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
Configuration Loader - Loads and parses YAML configuration files for ProcGuard.

This module provides the ConfigLoader class that handles loading ProcGuard
configuration from YAML files, parsing them into structured configuration
objects, and validating the configuration.

Example:
    >>> loader = ConfigLoader("procguard.yaml")
    >>> config = loader.load()
    >>> if loader.validate(config):
    ...     print("Configuration is valid")
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from .schema import (
    ProcGuardConfig,
    WorkerConfig,
    MonitoringConfig,
    RecoveryConfig,
    CommunicationConfig,
    LoggingConfig,
    StateConfig,
    WebConfig,
    SLURMConfig,
    PyTorchDistConfig,
)


class ConfigLoader:
    """
    Loads and parses ProcGuard configuration from YAML files.

    This class handles loading configuration from YAML files, parsing each
    section into the appropriate configuration dataclass, and validating
    the resulting configuration.

    Attributes:
        config_path: Path to the YAML configuration file
        logger: Logger instance for this loader

    Examples:
        >>> loader = ConfigLoader("config/procguard.yaml")
        >>> config = loader.load()
        >>> if loader.validate(config):
        ...     print("Configuration loaded successfully")
    """

    def __init__(self, config_path: str):
        """
        Initialize the ConfigLoader.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)

    def load(self) -> ProcGuardConfig:
        """
        Load and parse the configuration file.

        Returns:
            ProcGuardConfig: Parsed configuration object

        Raises:
            FileNotFoundError: If the configuration file does not exist
            yaml.YAMLError: If the file contains invalid YAML
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return self._parse_config(data)

    def _parse_config(self, data: Dict[str, Any]) -> ProcGuardConfig:
        """
        Parse raw configuration data into ProcGuardConfig.

        Args:
            data: Dictionary containing raw configuration data

        Returns:
            ProcGuardConfig: Fully parsed configuration object
        """
        config = ProcGuardConfig()

        if "workers" in data:
            config.workers = [
                self._parse_worker_config(worker_data) for worker_data in data["workers"]
            ]

        if "monitoring" in data:
            config.monitoring = self._parse_monitoring_config(data["monitoring"])

        if "recovery" in data:
            config.recovery = self._parse_recovery_config(data["recovery"])

        if "communication" in data:
            config.communication = self._parse_communication_config(data["communication"])

        if "logging" in data:
            config.logging = self._parse_logging_config(data["logging"])

        if "state" in data:
            config.state = self._parse_state_config(data["state"])

        if "web" in data:
            config.web = self._parse_web_config(data["web"])

        if "slurm" in data:
            config.slurm = self._parse_slurm_config(data["slurm"])

        if "pytorch_dist" in data:
            config.pytorch_dist = self._parse_pytorch_dist_config(data["pytorch_dist"])

        return config

    def _parse_worker_config(self, data: Dict[str, Any]) -> WorkerConfig:
        """
        Parse worker configuration data.

        Args:
            data: Dictionary containing worker configuration

        Returns:
            WorkerConfig: Parsed worker configuration
        """
        return WorkerConfig(
            worker_id=data["worker_id"],
            command=data["command"],
            working_dir=data.get("working_dir"),
            env=data.get("env", {}),
            pid_file=data.get("pid_file"),
            worker_type=data.get("worker_type", "default"),
            max_restarts=data.get("max_restarts", 3),
            restart_delay=data.get("restart_delay", 5.0),
            health_check_url=data.get("health_check_url"),
        )

    def _parse_monitoring_config(self, data: Dict[str, Any]) -> MonitoringConfig:
        """
        Parse monitoring configuration data.

        Args:
            data: Dictionary containing monitoring configuration

        Returns:
            MonitoringConfig: Parsed monitoring configuration
        """
        return MonitoringConfig(
            interval=data.get("interval", 1.0),
            heartbeat_timeout=data.get("heartbeat_timeout", 10.0),
            zombie_detection=data.get("zombie_detection", True),
            cpu_threshold=data.get("cpu_threshold", 0.1),
            memory_threshold=data.get("memory_threshold", 0.1),
        )

    def _parse_recovery_config(self, data: Dict[str, Any]) -> RecoveryConfig:
        """
        Parse recovery configuration data.

        Args:
            data: Dictionary containing recovery configuration

        Returns:
            RecoveryConfig: Parsed recovery configuration
        """
        return RecoveryConfig(
            enable_auto_recovery=data.get("enable_auto_recovery", True),
            stop_all_on_failure=data.get("stop_all_on_failure", True),
            task_reassignment=data.get("task_reassignment", True),
            recovery_timeout=data.get("recovery_timeout", 60.0),
            max_recovery_attempts=data.get("max_recovery_attempts", 3),
        )

    def _parse_communication_config(self, data: Dict[str, Any]) -> CommunicationConfig:
        """
        Parse communication configuration data.

        Args:
            data: Dictionary containing communication configuration

        Returns:
            CommunicationConfig: Parsed communication configuration
        """
        return CommunicationConfig(
            adapter_type=data.get("adapter_type", "mock"),
            adapter_config=data.get("adapter_config", {}),
        )

    def _parse_logging_config(self, data: Dict[str, Any]) -> LoggingConfig:
        """
        Parse logging configuration data.

        Args:
            data: Dictionary containing logging configuration

        Returns:
            LoggingConfig: Parsed logging configuration
        """
        return LoggingConfig(
            level=data.get("level", "INFO"),
            file=data.get("file"),
            format=data.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        )

    def _parse_state_config(self, data: Dict[str, Any]) -> StateConfig:
        """
        Parse state configuration data.

        Args:
            data: Dictionary containing state configuration

        Returns:
            StateConfig: Parsed state configuration
        """
        return StateConfig(
            state_file=data.get("state_file", "procguard_state.json"),
            auto_save=data.get("auto_save", True),
            save_interval=data.get("save_interval", 5.0),
        )

    def _parse_web_config(self, data: Dict[str, Any]) -> WebConfig:
        """
        Parse web server configuration data.

        Args:
            data: Dictionary containing web configuration

        Returns:
            WebConfig: Parsed web configuration
        """
        from .schema import WebConfig

        return WebConfig(
            enabled=data.get("enabled", False),
            host=data.get("host", "0.0.0.0"),
            port=data.get("port", 5000),
        )

    def _parse_slurm_config(self, data: Dict[str, Any]) -> SLURMConfig:
        """
        Parse SLURM configuration data.

        Args:
            data: Dictionary containing SLURM configuration

        Returns:
            SLURMConfig: Parsed SLURM configuration
        """
        return SLURMConfig(
            enabled=data.get("enabled", False),
            gpu_count_per_node=data.get("gpu_count_per_node", 1),
            master_port=data.get("master_port", 29500),
            worker_id_format=data.get("worker_id_format", "{hostname}-{local_rank}"),
            auto_detect=data.get("auto_detect", True),
            sbatch_template=data.get("sbatch_template"),
        )

    def _parse_pytorch_dist_config(self, data: Dict[str, Any]) -> PyTorchDistConfig:
        """
        Parse PyTorch distributed training configuration data.

        Args:
            data: Dictionary containing PyTorch distributed config

        Returns:
            PyTorchDistConfig: Parsed PyTorch distributed configuration
        """
        return PyTorchDistConfig(
            enabled=data.get("enabled", False),
            backend=data.get("backend", "nccl"),
            init_method=data.get("init_method", "env"),
            master_addr=data.get("master_addr"),
            master_port=data.get("master_port", 29500),
            local_rank_env_var=data.get("local_rank_env_var", "LOCAL_RANK"),
            rank_env_var=data.get("rank_env_var", "RANK"),
            world_size_env_var=data.get("world_size_env_var", "WORLD_SIZE"),
        )

    def validate(self, config: ProcGuardConfig) -> bool:
        """
        Validate the configuration.

        Checks for required fields, duplicate worker IDs, and other
        configuration validity issues.

        Args:
            config: ProcGuardConfig to validate

        Returns:
            bool: True if configuration is valid, False otherwise
        """
        if not config.workers:
            self.logger.error("No workers defined in configuration")
            return False

        worker_ids = [w.worker_id for w in config.workers]
        if len(worker_ids) != len(set(worker_ids)):
            self.logger.error("Duplicate worker IDs found")
            return False

        for worker in config.workers:
            if not worker.command:
                self.logger.error(f"Worker {worker.worker_id} has no command defined")
                return False

        return True
