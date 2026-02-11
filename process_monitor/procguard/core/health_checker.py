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

import time
import threading
import logging
from typing import Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .cluster_state import WorkerStatus, WorkerState


class HealthStatus(Enum):
    """
    Enumeration of possible health check results.

    Attributes:
        HEALTHY: Worker is operating normally
        UNRESPONSIVE: Worker is not responding to heartbeats
        DEAD: Worker process has died
        ZOMBIE: Worker process is a zombie
        UNKNOWN: Health status could not be determined
    """
    HEALTHY = "healthy"
    UNRESPONSIVE = "unresponsive"
    DEAD = "dead"
    ZOMBIE = "zombie"
    UNKNOWN = "unknown"


@dataclass
class WorkerHealthReport:
    """
    Data class representing a health check report for a worker.

    Attributes:
        worker_id: Unique identifier of the worker
        status: Current HealthStatus
        pid: Process ID of the worker
        cpu_percent: CPU usage percentage
        memory_percent: Memory usage percentage
        last_check_time: Timestamp of last health check
        error_message: Error message if health check failed
        is_zombie: Whether the process is a zombie
    """
    worker_id: str
    status: HealthStatus
    pid: Optional[int]
    cpu_percent: float
    memory_percent: float
    last_check_time: str
    error_message: Optional[str] = None
    is_zombie: bool = False


class HealthChecker:
    """
    Monitors the health and status of worker processes.

    This class periodically checks the health of registered workers,
    monitoring their CPU usage, memory usage, heartbeat status,
    and detecting zombie processes. It supports callbacks for
    failure detection and log handling.

    Attributes:
        worker_configs: Configuration for each worker
        check_interval: Interval between health checks in seconds
        heartbeat_timeout: Timeout in seconds before marking worker as unresponsive
        zombie_detection: Whether to detect zombie processes
        cpu_threshold: CPU usage threshold for healthy status
        memory_threshold: Memory usage threshold for healthy status

    Examples:
        >>> checker = HealthChecker(worker_configs={}, check_interval=1.0)
        >>> checker.register_worker("worker_1", {})
        >>> checker.start()
        >>> # Health checks run in background thread
        >>> checker.stop()
    """

    def __init__(
        self,
        worker_configs: Dict[str, any],
        check_interval: float = 1.0,
        heartbeat_timeout: float = 10.0,
        zombie_detection: bool = True,
        cpu_threshold: float = 0.1,
        memory_threshold: float = 0.1,
    ):
        """
        Initialize the HealthChecker with monitoring parameters.

        Args:
            worker_configs: Dictionary mapping worker IDs to their configurations
            check_interval: Time between health checks in seconds
            heartbeat_timeout: Time without heartbeat before marking unresponsive
            zombie_detection: Enable zombie process detection
            cpu_threshold: Maximum CPU percentage for healthy status
            memory_threshold: Maximum memory percentage for healthy status
        """
        self.worker_configs = worker_configs
        self.check_interval = check_interval
        self.heartbeat_timeout = heartbeat_timeout
        self.zombie_detection = zombie_detection
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold

        self.logger = logging.getLogger(__name__)
        self._running = False
        self._check_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()

        self._worker_states: Dict[str, WorkerState] = {}
        self._last_check_times: Dict[str, datetime] = {}
        self._health_reports: Dict[str, WorkerHealthReport] = {}
        self._manually_stopped_workers: Dict[str, bool] = {}

        self._on_failure_callback: Optional[Callable[[List[str]], None]] = None
        self._on_log_callback: Optional[Callable[[str, str], None]] = None

        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available, health checking will be limited")

    def on_failure(self, callback: Callable[[List[str]], None]):
        """
        Register a callback for worker failure detection.

        The callback will be invoked with a list of failed worker IDs
        when health checks detect failures.

        Args:
            callback: Function that takes a list of worker IDs
        """
        self._on_failure_callback = callback

    def on_log(self, callback: Callable[[str, str], None]):
        """
        Register a callback for log messages from workers.

        Args:
            callback: Function that takes (message, log_type) parameters
        """
        self._on_log_callback = callback

    def register_worker(self, worker_id: str, worker_config: dict):
        """
        Register a new worker for health monitoring.

        Initializes the worker's state and health report in the
        monitoring system.

        Args:
            worker_id: Unique identifier of the worker
            worker_config: Configuration dictionary for the worker
        """
        with self._lock:
            self._worker_states[worker_id] = WorkerState(worker_id=worker_id)
            self._last_check_times[worker_id] = datetime.now()
            self._health_reports[worker_id] = WorkerHealthReport(
                worker_id=worker_id,
                status=HealthStatus.UNKNOWN,
                pid=None,
                cpu_percent=0.0,
                memory_percent=0.0,
                last_check_time=datetime.now().isoformat(),
            )
            self.logger.info(f"Registered worker for health checking: {worker_id}")

    def start(self):
        """
        Start the health check monitoring loop.

        Creates and starts a background thread that periodically
        checks the health of all registered workers.
        """
        if self._running:
            self.logger.warning("Health checker already running")
            return

        self._running = True
        self._check_thread = threading.Thread(target=self._run_checks, daemon=True)
        self._check_thread.start()
        self.logger.info("Health checker started")

    def stop(self):
        """
        Stop the health check monitoring loop.

        Sets the running flag to false and waits for the check
        thread to terminate.
        """
        self._running = False
        if self._check_thread:
            self._check_thread.join(timeout=5.0)
        self.logger.info("Health checker stopped")

    def _run_checks(self):
        """
        Internal method that runs the health check loop.

        Continuously checks all workers at the configured interval
        until stop() is called.
        """
        while self._running:
            try:
                self._check_all_workers()
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
            time.sleep(self.check_interval)

    def _check_all_workers(self):
        """
        Check the health of all registered workers.

        Collects failed workers and invokes the failure callback
        if any are detected.
        """
        with self._lock:
            worker_ids = list(self._worker_states.keys())

        failed_workers = []

        for worker_id in worker_ids:
            is_healthy = self._check_worker(worker_id)

            if not is_healthy:
                failed_workers.append(worker_id)

        if failed_workers and self._on_failure_callback:
            self._on_failure_callback(failed_workers)

    def _check_worker(self, worker_id: str) -> bool:
        """
        Perform a health check on a single worker.

        Checks if the worker is responsive based on heartbeat timeout.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            bool: True if worker is healthy, False otherwise
        """
        with self._lock:
            if worker_id not in self._worker_states:
                return True

            worker_state = self._worker_states[worker_id]
            last_check = self._last_check_times.get(worker_id)

            if (
                last_check
                and (datetime.now() - last_check).total_seconds() > self.heartbeat_timeout
            ):
                if worker_state.status == WorkerStatus.RUNNING:
                    self._update_health_report(
                        worker_id, HealthStatus.UNRESPONSIVE, "heartbeat timeout"
                    )
                    return False

            return True

    def _update_health_report(
        self, worker_id: str, status: HealthStatus, error_message: Optional[str] = None
    ):
        """
        Update the health report for a worker.

        Args:
            worker_id: Unique identifier of the worker
            status: New HealthStatus to set
            error_message: Optional error message for failure cases
        """
        with self._lock:
            if worker_id not in self._health_reports:
                return

            self._health_reports[worker_id].status = status
            self._health_reports[worker_id].last_check_time = datetime.now().isoformat()
            self._health_reports[worker_id].error_message = error_message

    def get_health_report(self, worker_id: str) -> Optional[WorkerHealthReport]:
        """
        Get the latest health report for a worker.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            Optional[WorkerHealthReport]: Health report or None if not found
        """
        with self._lock:
            return self._health_reports.get(worker_id)

    def get_all_health_reports(self) -> Dict[str, WorkerHealthReport]:
        """
        Get health reports for all registered workers.

        Returns:
            Dict[str, WorkerHealthReport]: Dictionary mapping worker IDs to reports
        """
        with self._lock:
            return dict(self._health_reports)

    def clear_worker_manually_stopped(self, worker_id: str):
        """
        Clear the manually stopped flag for a worker.

        Args:
            worker_id: Unique identifier of the worker
        """
        with self._lock:
            if worker_id in self._manually_stopped_workers:
                del self._manually_stopped_workers[worker_id]

    def mark_worker_manually_stopped(self, worker_id: str):
        """
        Mark a worker as manually stopped.

        Prevents automatic failure detection for manually stopped workers.

        Args:
            worker_id: Unique identifier of the worker
        """
        with self._lock:
            self._manually_stopped_workers[worker_id] = True

    def is_worker_manually_stopped(self, worker_id: str) -> bool:
        """
        Check if a worker was manually stopped.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            bool: True if worker was manually stopped
        """
        with self._lock:
            return worker_id in self._manually_stopped_workers
