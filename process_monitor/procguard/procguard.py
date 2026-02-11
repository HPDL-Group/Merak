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
ProcGuard - Process Monitoring and Recovery System for Distributed Training.

This module provides the ProcGuard class for managing and monitoring
distributed training processes across a cluster. It supports SLURM
cluster integration, PyTorch distributed training, and automatic
failure recovery.

Classes:
    ProcGuard: Main class for process management and monitoring

Functions:
    serve: Command-line entry point for ProcGuard

Example:
    >>> procguard = ProcGuard("configs/procguard.yaml", slurm_mode=True)
    >>> procguard.load_config()
    >>> procguard.setup_logging()
    >>> procguard.initialize_components()
    >>> procguard.start_workers()
    >>> procguard.run()
"""

import signal
import sys
import logging
import time
import threading
from pathlib import Path

from .config import ConfigLoader
from .core import HealthChecker, ProcessManager, RecoveryOrchestrator, ClusterState, SLURMManager, GroupManager
from .adapters import MockCommunicationAdapter, ZMQCommunicationAdapter
from .utils.logger import setup_logging
from .web import WebMonitor


class ProcGuard:
    """Main ProcGuard class for process monitoring and recovery management.

    ProcGuard provides comprehensive process management capabilities for
    distributed training workloads. It handles worker lifecycle management,
    health monitoring, failure detection, and automatic recovery.

    The class supports multiple deployment modes:
    - Standard mode: Manages workers locally
    - Decoupled mode: Workers register from remote machines
    - SLURM mode: Integration with SLURM cluster scheduler

    Attributes:
        config_path: Path to the configuration YAML file
        decoupled_mode: Whether workers register remotely
        slurm_mode: Whether running in SLURM environment
        gpu_count: Number of GPUs per node
        state_manager: ClusterState instance for worker tracking
        process_manager: ProcessManager for worker control
        health_checker: HealthChecker for monitoring
        recovery_orchestrator: RecoveryOrchestrator for failure handling
        communication_adapter: CommunicationAdapter for inter-process comms
        web_monitor: WebMonitor for web UI
        slurm_manager: SLURMManager for SLURM integration
        group_manager: GroupManager for worker grouping

    Example:
        >>> procguard = ProcGuard("configs/procguard.yaml")
        >>> procguard.load_config()
        >>> procguard.initialize_components()
        >>> procguard.start_workers()
        >>> procguard.run()
    """

    def __init__(self, config_path: str, decoupled_mode: bool = False, slurm_mode: bool = False, gpu_count: int = 1):
        self.config_path = Path(config_path)
        self.logger = logging.getLogger(__name__)
        self.decoupled_mode = decoupled_mode
        self.slurm_mode = slurm_mode
        self.gpu_count = gpu_count

        self._running = False
        self._shutdown_event = threading.Event()

        self.config_loader = ConfigLoader(str(self.config_path))
        self.config = None

        self.state_manager = None
        self.process_manager = None
        self.health_checker = None
        self.recovery_orchestrator = None
        self.communication_adapter = None
        self.web_monitor = None
        self.slurm_manager = None
        self.group_manager = None

        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Configure signal handlers for graceful shutdown.

        Registers handlers for SIGINT, SIGTERM, and SIGHUP signals
        to ensure clean shutdown on user interrupt or system signals.
        """
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle termination signals.

        Args:
            signum: Signal number received
            frame: Current stack frame
        """
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)

    def load_config(self):
        """Load configuration from YAML file.

        Reads the configuration file and validates it against the schema.

        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If config file does not exist
        """
        self.logger.info(f"Loading configuration from {self.config_path}")
        self.config = self.config_loader.load()

        if not self.config_loader.validate(self.config):
            raise ValueError("Invalid configuration")

        self.logger.info("Configuration loaded successfully")

    def setup_logging(self):
        """Configure logging based on loaded configuration.

        Sets up logging level, format, and output file as specified
        in the configuration file.
        """
        logging_config = self.config.logging
        setup_logging(
            level=logging_config.level,
            log_file=logging_config.file,
            log_format=logging_config.format,
        )
        self.logger.info("Logging configured")

    def initialize_components(self):
        """Initialize all ProcGuard components.

        Creates and configures all internal components including:
        - Communication adapter
        - State manager
        - Group manager
        - Web monitor
        - Health checker
        - Process manager
        - Recovery orchestrator
        - SLURM manager
        """
        self.logger.info("Initializing ProcGuard components")

        self._init_communication_adapter()
        self._init_state_manager()
        self._init_group_manager()
        self._init_web_monitor()
        self._init_health_checker()
        self._init_process_manager()
        self._init_recovery_orchestrator()
        self._init_slurm_manager()

        self.logger.info("All components initialized")

    def _init_communication_adapter(self):
        """Initialize the communication adapter.

        Creates a communication adapter based on configuration (ZMQ or Mock).
        The adapter is used for inter-process communication between
        ProcGuard and workers.
        """
        comm_config = self.config.communication
        adapter_type = comm_config.adapter_type

        self.logger.info(f"Initializing communication adapter: {adapter_type}")

        if adapter_type == "zmq":
            self.communication_adapter = ZMQCommunicationAdapter(comm_config.adapter_config)
        else:
            self.communication_adapter = MockCommunicationAdapter(comm_config.adapter_config)

        self.communication_adapter.initialize()

    def _init_state_manager(self):
        """Initialize the cluster state manager.

        Creates a ClusterState instance for tracking worker status,
        registering workers, and persisting state to disk.
        """
        state_config = self.config.state
        self.state_manager = ClusterState(
            state_file=state_config.state_file,
            auto_save=state_config.auto_save,
            decoupled_mode=self.decoupled_mode,
        )

        if not self.decoupled_mode:
            for worker in self.config.workers:
                self.state_manager.register_worker(worker.worker_id)

        self.logger.info("State manager initialized")

    def _init_group_manager(self):
        """Initialize the worker group manager.

        Creates a GroupManager for organizing workers into groups
        for distributed training coordination.
        """
        group_state_file = "procguard_groups.json"
        
        self.group_manager = GroupManager(
            state_file=group_state_file,
            auto_save=False,
        )
        self.logger.info("Group manager initialized (本地缓存已禁用)")

    def _init_process_manager(self):
        """Initialize the process manager.

        Creates a ProcessManager for starting, stopping, and restarting
        worker processes. Registers all configured workers.
        """
        self.process_manager = ProcessManager(
            self.communication_adapter, self.health_checker, self.web_monitor
        )

        if not self.decoupled_mode:
            for worker in self.config.workers:
                self.process_manager.register_worker_config(
                    worker.worker_id,
                    {
                        "command": worker.command,
                        "working_dir": worker.working_dir,
                        "env": worker.env,
                        "restart_delay": worker.restart_delay,
                    },
                )

        self.logger.info("Process manager initialized")

    def _init_health_checker(self):
        """Initialize the health checker.

        Creates a HealthChecker for monitoring worker health, detecting
        zombies, and managing heartbeats. Registers all configured workers.
        """
        monitoring_config = self.config.monitoring

        self.health_checker = HealthChecker(
            worker_configs={},
            check_interval=monitoring_config.interval,
            heartbeat_timeout=monitoring_config.heartbeat_timeout,
            zombie_detection=monitoring_config.zombie_detection,
            cpu_threshold=monitoring_config.cpu_threshold,
            memory_threshold=monitoring_config.memory_threshold,
        )

        if not self.decoupled_mode:
            for worker in self.config.workers:
                self.health_checker.register_worker(worker.worker_id, {})

        self.health_checker.on_failure(self._handle_worker_failure)
        self.health_checker.on_log(self._handle_log_message)

        self.logger.info("Health checker initialized")

    def _init_recovery_orchestrator(self):
        """Initialize the recovery orchestrator.

        Creates a RecoveryOrchestrator for handling worker failures,
        including automatic restart and task reassignment.
        """
        self.recovery_orchestrator = RecoveryOrchestrator(
            state_manager=self.state_manager,
            process_manager=self.process_manager,
            recovery_config={
                "enable_auto_recovery": self.config.recovery.enable_auto_recovery,
                "stop_all_on_failure": self.config.recovery.stop_all_on_failure,
                "task_reassignment": self.config.recovery.task_reassignment,
                "recovery_timeout": self.config.recovery.recovery_timeout,
                "max_recovery_attempts": self.config.recovery.max_recovery_attempts,
            },
        )

        self.recovery_orchestrator.on_recovery_start(self._on_recovery_start)
        self.recovery_orchestrator.on_recovery_complete(self._on_recovery_complete)

        self.logger.info("Recovery orchestrator initialized")

    def _init_slurm_manager(self):
        """Initialize the SLURM manager.

        Creates a SLURMManager for SLURM cluster integration,
        detecting environment, and generating PyTorch distributed
        training configuration.
        """
        slurm_config = getattr(self.config, 'slurm', None)
        pytorch_config = getattr(self.config, 'pytorch_dist', None)

        gpu_count = self.gpu_count
        master_port = 29500

        if slurm_config:
            if slurm_config.gpu_count_per_node > 0:
                gpu_count = slurm_config.gpu_count_per_node
            master_port = slurm_config.master_port

        self.slurm_manager = SLURMManager(
            gpu_count_per_node=gpu_count,
            master_port=master_port,
        )

        if self.slurm_mode or (slurm_config and slurm_config.enabled):
            self.logger.info(f"SLURM mode enabled with {gpu_count} GPUs per node")

            env_info = self.slurm_manager.detect_environment()
            self.logger.info(
                f"SLURM environment: hostname={env_info.hostname}, "
                f"local_rank={env_info.local_rank}, rank={env_info.rank}/{env_info.world_size}"
            )

            if pytorch_config and pytorch_config.enabled:
                dist_env = self.slurm_manager.build_pytorch_dist_env()
                self.logger.info(
                    f"PyTorch distributed config: master_addr={dist_env.master_addr}, "
                    f"master_port={dist_env.master_port}, world_size={dist_env.world_size}"
                )

        self.logger.info("SLURM manager initialized")

    def _init_web_monitor(self):
        """Initialize the web monitoring interface.

        Creates a WebMonitor for real-time web-based monitoring
        and control of workers via a Flask + Socket.IO interface.
        """
        web_config = getattr(self.config, "web", None)
        if web_config and web_config.enabled:
            self.web_monitor = WebMonitor(
                procguard_instance=self, host=web_config.host, port=web_config.port
            )
            self.logger.info(f"Web monitor initialized on {web_config.host}:{web_config.port}")
        else:
            self.logger.info("Web monitor disabled")

    def _handle_worker_failure(self, failed_worker_ids):
        """Handle worker failure events.

        Called when the health checker detects worker failures.
        Marks workers as failed and initiates recovery procedures.

        Args:
            failed_worker_ids: List of worker IDs that have failed
        """
        self.logger.error(f"Worker failure detected: {failed_worker_ids}")

        actual_failures = []
        for worker_id in failed_worker_ids:
            if self.health_checker.is_worker_manually_stopped(worker_id):
                self.logger.info(
                    f"Worker {worker_id} was manually stopped, skipping failure handling"
                )
                continue

            actual_failures.append(worker_id)
            self.state_manager.mark_worker_failed(worker_id)
            self.state_manager.clear_failed_workers()

        if actual_failures:
            self.recovery_orchestrator.handle_failure(actual_failures)

    def _on_recovery_start(self, worker_id: str):
        """Callback when recovery starts for a worker.

        Args:
            worker_id: ID of the worker being recovered
        """
        self.logger.info(f"Recovery started for worker {worker_id}")

    def _on_recovery_complete(self, worker_id: str, success: bool):
        """Callback when recovery completes for a worker.

        Args:
            worker_id: ID of the worker that was recovered
            success: Whether recovery was successful
        """
        if success:
            self.logger.info(f"Recovery completed successfully for worker {worker_id}")
            self.state_manager.clear_failed_workers()
        else:
            self.logger.error(f"Recovery failed for worker {worker_id}")

    def _handle_log_message(self, message: str, log_type: str = "info"):
        """Handle log messages from workers.

        Logs the message and broadcasts alerts to connected web clients.

        Args:
            message: Log message content
            log_type: Log level (info, warning, error, etc.)
        """
        self.logger.info(f"[{log_type.upper()}] {message}")
        if self.web_monitor:
            self.web_monitor.broadcast_alert(log_type, message)

    def start_workers(self):
        """Start all configured workers.

        Launches worker processes based on configuration. In SLURM mode,
        generates SLURM-compliant worker IDs and environment variables.
        In decoupled mode, this is a no-op as workers register remotely.
        """
        if self.decoupled_mode:
            self.logger.info("Decoupled mode: workers will be added when worker launchers register")
            return

        self.logger.info("Starting all workers")

        slurm_config = getattr(self.config, 'slurm', None)
        pytorch_config = getattr(self.config, 'pytorch_dist', None)

        for worker in self.config.workers:
            worker_id = worker.worker_id
            worker_env = dict(worker.env)

            if self.slurm_mode or (slurm_config and slurm_config.enabled):
                if self.slurm_manager:
                    slurm_worker_id = self.slurm_manager.generate_worker_id()
                    worker_id = slurm_worker_id
                    self.logger.info(f"SLURM mode: generated worker_id = {worker_id}")

                    slurm_vars = self.slurm_manager.get_slurm_vars_for_worker(worker_id)
                    worker_env.update(slurm_vars)

            if pytorch_config and pytorch_config.enabled:
                if self.slurm_manager:
                    pytorch_env = self.slurm_manager.get_pytorch_env_vars()
                    worker_env.update(pytorch_env)
                    self.logger.info(f"Added PyTorch distributed environment variables for worker {worker_id}")

            self.logger.info(f"Starting worker: {worker_id}")
            success = self.process_manager.start_worker(
                worker_id,
                custom_env=worker_env if worker_env else None
            )
            if success:
                pid = self.process_manager.get_worker_pid(worker_id)
                self.logger.info(f"Worker {worker_id} started with PID: {pid}")
                self.health_checker.update_worker_pid(worker_id, pid)
                self.state_manager.update_worker_status(
                    worker_id, self.state_manager._workers.get(worker_id, self.state_manager._workers.get(worker.worker_id))
                )
            else:
                self.logger.error(f"Failed to start worker {worker_id}")

        self.logger.info("All workers started")

    def run(self):
        """Run the main ProcGuard event loop.

        Starts the health checker, web monitor (if enabled), and enters
        the main event loop that periodically prints status updates.
        Blocks until shutdown is requested via signal or exception.
        """
        self.logger.info("Starting ProcGuard")

        self._running = True

        try:
            self.health_checker.start()

            if self.web_monitor:
                import threading

                web_thread = threading.Thread(
                    target=self.web_monitor.run, kwargs={"debug": False}, daemon=True
                )
                web_thread.start()
                self.logger.info("Web monitor started")

            self.logger.info("ProcGuard is running")

            while self._running and not self._shutdown_event.is_set():
                self._print_status()
                time.sleep(10)

        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt")
        except Exception as e:
            self.logger.error(f"Error in main loop: {e}")
        finally:
            self.shutdown()

    def _print_status(self):
        """Print current system status to logs.

        Generates and logs a status summary including running workers,
        failed workers, task counts, and recovery statistics.
        """
        summary = self.state_manager.get_state_summary()
        recovery_stats = self.recovery_orchestrator.get_recovery_stats()

        if self.decoupled_mode and self.web_monitor:
            with self.web_monitor._remote_workers_lock:
                remote_count = len(self.web_monitor._remote_workers)
                remote_running = sum(
                    1 for w in self.web_monitor._remote_workers.values() if w.status == "running"
                )
                local_count = summary["total_workers"]
                local_running = summary["running_workers"]
                total_workers = local_count + remote_count
                total_running = local_running + remote_running

                status_msg = (
                    f"Decoupled Mode: {total_running}/{total_workers} workers running "
                    f"({remote_running} remote, {local_running} local), "
                    f"{summary['failed_workers']} failed, "
                    f"{recovery_stats['total_recoveries']} recoveries "
                    f"({recovery_stats['success_rate']:.1%} success)"
                )
        else:
            status_msg = (
                f"Status: {summary['running_workers']}/{summary['total_workers']} workers running, "
                f"{summary['failed_workers']} failed, "
                f"{summary['total_tasks']} tasks, "
                f"{recovery_stats['total_recoveries']} recoveries "
                f"({recovery_stats['success_rate']:.1%} success)"
            )

        self.logger.info(status_msg)

    def shutdown(self):
        """Gracefully shutdown ProcGuard and all components.

        Stops all running workers, health checker, web monitor,
        and communication adapter. Sets the shutdown event to
        break the main event loop.
        """
        if not self._running:
            return

        self.logger.info("Shutting down ProcGuard")
        self._running = False
        self._shutdown_event.set()

        if self.health_checker:
            self.health_checker.stop()

        if self.web_monitor:
            self.web_monitor.shutdown()

        if self.process_manager:
            self.process_manager.shutdown()

        if self.communication_adapter:
            self.communication_adapter.shutdown()

        self.logger.info("ProcGuard shutdown complete")

    def get_status(self):
        """Get current system status.

        Returns a comprehensive status dictionary including:
        - State summary (worker counts, task counts)
        - Recovery statistics
        - Worker statuses
        - PyTorch configuration (if groups exist)
        - Remote workers (in decoupled mode)

        Returns:
            dict: Current system status
        """
        status = {
            "state": self.state_manager.get_state_summary(),
            "recovery": self.recovery_orchestrator.get_recovery_stats(),
            "workers": self.process_manager.get_all_worker_statuses(),
        }

        if self.group_manager:
            status["pytorch_config"] = self._get_pytorch_config_from_groups()

        if self.decoupled_mode and self.web_monitor:
            with self.web_monitor._remote_workers_lock:
                remote_workers = {
                    worker_id: {
                        "status": w.status,
                        "pid": w.pid,
                        "command": w.command,
                        "restart_count": w.restart_count,
                    }
                    for worker_id, w in self.web_monitor._remote_workers.items()
                }
                status["remote_workers"] = remote_workers
                status["decoupled_mode"] = True

        return status

    def _get_pytorch_config_from_groups(self) -> dict:
        """Extract PyTorch configuration from worker groups.

        Searches through groups to find one with valid PyTorch
        distributed training configuration and returns it.

        Returns:
            dict: PyTorch configuration or empty dict if not found
        """
        if not self.group_manager:
            return {}

        config = {}
        groups = self.group_manager.get_all_groups()

        for group_id, group in groups.items():
            group_config = self.group_manager.get_group_config(group_id)

            if group_config and group_config.master_addr and group_config.world_size:
                config = {
                    "master_addr": group_config.master_addr,
                    "master_port": group_config.master_port,
                    "world_size": group_config.world_size,
                    "backend": group_config.backend,
                    "cuda_visible_devices": group_config.cuda_visible_devices,
                    "nccl_socket_ifname": group_config.nccl_socket_ifname,
                }
                self.logger.debug(f"[ProcGuard] 使用分组 {group_id} 的 PyTorch 配置: world_size={config['world_size']}")
                return config

        return config


def serve():
    """ ProcGuard service fromStart command line.

    This is the command-line entry point for ProcGuard. It parses
    command-line arguments, creates a ProcGuard instance, and runs
    the main event loop.

    Command-line arguments:
        --config: Path to configuration file (default: configs/procguard_config.yaml)
        --daemon: Run as daemon process
        --decoupled: Enable decoupled mode for remote workers
        --slurm: Enable SLURM mode for cluster deployment
        --gpu-count: Number of GPUs per node (default: 1)

    Raises:
        SystemExit: On critical errors during startup

    Example:
        $ python -m procguard --config myconfig.yaml --slurm --gpu-count 4
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="ProcGuard - Process Monitoring and Recovery System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/procguard_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    parser.add_argument(
        "--decoupled",
        action="store_true",
        help="Decoupled mode: workers are added when worker launchers register",
    )
    parser.add_argument(
        "--slurm",
        action="store_true",
        help="Enable SLURM mode for cluster deployment",
    )
    parser.add_argument(
        "--gpu-count",
        type=int,
        default=1,
        help="Number of GPUs per node (for SLURM/PyTorch distributed training)",
    )

    args = parser.parse_args()

    try:
        procguard = ProcGuard(
            args.config,
            decoupled_mode=args.decoupled,
            slurm_mode=args.slurm,
            gpu_count=args.gpu_count,
        )
        procguard.load_config()
        procguard.setup_logging()
        procguard.initialize_components()
        procguard.start_workers()
        procguard.run()
    except Exception as e:
        logging.error(f"Failed to start ProcGuard: {e}")
        sys.exit(1)


if __name__ == "__main__":
    serve()
