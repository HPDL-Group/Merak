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

import subprocess
import time
import signal
import logging
import threading
import os
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .cluster_state import WorkerStatus, WorkerState


@dataclass
class WorkerProcessInfo:
    """
    Data class representing information about a worker process.

    Attributes:
        worker_id: Unique identifier for the worker
        process: Subprocess Popen object for the worker
        pid: Process ID of the worker
        command: Command being executed by the worker
        working_dir: Working directory for the worker process
        env: Environment variables for the worker
        status: Current WorkerStatus
        start_time: Timestamp when worker started
        restart_count: Number of times worker has been restarted
    """
    worker_id: str
    process: Optional[subprocess.Popen]
    pid: Optional[int]
    command: str
    working_dir: Optional[str]
    env: Dict[str, str]
    status: WorkerStatus
    start_time: Optional[float]
    restart_count: int = 0


class ProcessManager:
    """
    Manages the lifecycle of worker processes.

    This class provides comprehensive process management capabilities including
    starting, stopping, restarting workers, reading process output, and
    handling graceful and forced shutdowns. It supports both psutil-based
    process management (when available) and fallback methods.

    Attributes:
        _worker_processes: Dictionary mapping worker IDs to WorkerProcessInfo
        _worker_configs: Dictionary mapping worker IDs to their configurations
        _worker_logs: Dictionary mapping worker IDs to their output logs
        _lock: Reentrant lock for thread-safe operations

    Examples:
        >>> manager = ProcessManager()
        >>> manager.register_worker_config("worker_1", {"command": "python train.py"})
        >>> manager.start_worker("worker_1")
        True
        >>> manager.stop_worker("worker_1")
        True
    """

    def __init__(self, communication_adapter=None, health_checker=None, web_monitor=None):
        """
        Initialize the ProcessManager.

        Args:
            communication_adapter: Optional communication adapter for sending commands
            health_checker: Optional health checker for monitoring worker health
            web_monitor: Optional web monitor for web-based monitoring
        """
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

        self._worker_processes: Dict[str, WorkerProcessInfo] = {}
        self._worker_configs: Dict[str, dict] = {}
        self._worker_logs: Dict[str, List[str]] = {}
        self._communication_adapter = communication_adapter
        self._health_checker = health_checker
        self._web_monitor = web_monitor

        self._shutdown_event = threading.Event()

        if not PSUTIL_AVAILABLE:
            self.logger.warning("psutil not available, process management will be limited")

    def register_worker_config(self, worker_id: str, config: dict):
        """
        Register configuration for a worker.

        Args:
            worker_id: Unique identifier for the worker
            config: Configuration dictionary containing command, working_dir, env, etc.
        """
        with self._lock:
            self._worker_configs[worker_id] = config
            self.logger.info(f"Registered worker config: {worker_id}")

    def get_worker_config(self, worker_id: str) -> Optional[dict]:
        """
        Get the configuration for a worker.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            Optional[dict]: Configuration dictionary or None if not found
        """
        with self._lock:
            return self._worker_configs.get(worker_id)

    def start_worker(self, worker_id: str, custom_env: Optional[Dict[str, str]] = None) -> bool:
        """
        Start a worker process.

        Launches the worker process according to its registered configuration.
        Sets up environment variables and initializes health tracking.

        Args:
            worker_id: Unique identifier of the worker
            custom_env: Optional custom environment variables to merge

        Returns:
            bool: True if worker started successfully, False otherwise
        """
        with self._lock:
            if worker_id not in self._worker_configs:
                self.logger.error(f"Worker {worker_id} not configured")
                return False

            config = self._worker_configs[worker_id]

            if worker_id in self._worker_processes:
                process_info = self._worker_processes[worker_id]
                if process_info.process and process_info.process.poll() is None:
                    self.logger.warning(f"Worker {worker_id} already running")
                    return False

            try:
                command = config["command"]
                working_dir = config.get("working_dir")
                env = config.get("env", {})

                if custom_env:
                    env.update(custom_env)
                    self.logger.info(f"Using custom environment for worker {worker_id}: {len(custom_env)} variables")

                if self._health_checker:
                    self._health_checker.clear_worker_manually_stopped(worker_id)

                process = self._launch_process(command, working_dir, env)

                if process:
                    self._worker_processes[worker_id] = WorkerProcessInfo(
                        worker_id=worker_id,
                        process=process,
                        pid=process.pid,
                        command=command,
                        working_dir=working_dir,
                        env=env,
                        status=WorkerStatus.RUNNING,
                        start_time=time.time(),
                        restart_count=0,
                    )

                    self.logger.info(f"Started worker {worker_id} with PID {process.pid}")

                    if self._health_checker:
                        self._health_checker.update_worker_pid(worker_id, process.pid)
                        self._health_checker.update_worker_heartbeat(worker_id)

                    if self._web_monitor:
                        self._start_output_reader(worker_id, process)

                    if self._communication_adapter:
                        self._communication_adapter.send_command(worker_id, "start")

                    return True
                else:
                    self.logger.error(f"Failed to start worker {worker_id}")
                    return False

            except Exception as e:
                self.logger.error(f"Error starting worker {worker_id}: {e}")
                return False

    def _launch_process(
        self, command: str, working_dir: Optional[str] = None, env: Optional[Dict[str, str]] = None
    ) -> Optional[subprocess.Popen]:
        """
        Launch a subprocess with the given command and environment.

        Args:
            command: Command string to execute
            working_dir: Working directory for the process
            env: Environment variables for the process

        Returns:
            Optional[subprocess.Popen]: Popen object or None if launch failed
        """
        try:
            import shlex

            args = shlex.split(command)

            self.logger.info(
                f"Launching process: command={command}, working_dir={working_dir}, args={args}"
            )

            process_env = os.environ.copy()
            if env:
                process_env.update(env)

            process = subprocess.Popen(
                args,
                cwd=working_dir,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.PIPE,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None,
            )

            self.logger.info(f"Process launched with PID {process.pid}")

            return process

        except Exception as e:
            self.logger.error(f"Failed to launch process: {e}")
            return None

    def _start_output_reader(self, worker_id: str, process: subprocess.Popen):
        """
        Start reading and logging output from a worker process.

        Creates background threads to read stdout and stderr streams
        and log the output.

        Args:
            worker_id: Unique identifier of the worker
            process: Popen object for the worker process
        """
        with self._lock:
            if worker_id not in self._worker_logs:
                self._worker_logs[worker_id] = []

        def read_output(pipe, stream_type):
            try:
                for line in iter(pipe.readline, b""):
                    if line:
                        message = line.decode("utf-8", errors="replace").strip()
                        timestamp = time.strftime("%H:%M:%S")
                        log_entry = f"[{timestamp}] {message}"

                        with self._lock:
                            if worker_id in self._worker_logs:
                                self._worker_logs[worker_id].append(log_entry)
                                if len(self._worker_logs[worker_id]) > 1000:
                                    self._worker_logs[worker_id] = self._worker_logs[worker_id][
                                        -1000:
                                    ]

                        self.logger.info(f"[{worker_id}][{stream_type}] {message}")
            except Exception as e:
                self.logger.debug(f"Error reading {stream_type} for {worker_id}: {e}")
            finally:
                pipe.close()

        if process.stdout:
            import threading

            threading.Thread(
                target=read_output, args=(process.stdout, "stdout"), daemon=True
            ).start()

        if process.stderr:
            import threading

            threading.Thread(
                target=read_output, args=(process.stderr, "stderr"), daemon=True
            ).start()

    def stop_worker(self, worker_id: str, force: bool = False) -> bool:
        """
        Stop a worker process.

        Gracefully terminates the worker process, or forcefully kills it
        if force is True.

        Args:
            worker_id: Unique identifier of the worker
            force: Whether to forcefully kill the process

        Returns:
            bool: True if worker was stopped successfully
        """
        self.logger.info(f"[Manager] Stopping worker {worker_id}, force={force}")
        self.logger.info(f"[Manager] Available workers: {list(self._worker_processes.keys())}")

        with self._lock:
            if worker_id not in self._worker_processes:
                self.logger.warning(f"[Manager] Worker {worker_id} not found in registered workers")
                return False

            process_info = self._worker_processes[worker_id]

            if not process_info.process:
                self.logger.warning(f"[Manager] Worker {worker_id} has no process")
                return False

            try:
                if self._health_checker:
                    self._health_checker.mark_worker_manually_stopped(worker_id)

                if PSUTIL_AVAILABLE:
                    result = self._stop_worker_with_psutil(worker_id, process_info, force)
                else:
                    result = self._stop_worker_without_psutil(worker_id, process_info, force)

                if result:
                    process_info.status = WorkerStatus.STOPPED
                    process_info.process = None
                    process_info.pid = None

                    if self._health_checker:
                        self._health_checker.update_worker_pid(worker_id, None)

                    if self._communication_adapter:
                        self._communication_adapter.send_command(worker_id, "stop")

                self.logger.info(f"[Manager] Stop worker {worker_id} result: {result}")
                return result
            except Exception as e:
                self.logger.error(f"[Manager] Error stopping worker {worker_id}: {e}")
                import traceback
                self.logger.error(f"[Manager] Traceback: {traceback.format_exc()}")
                return False

    def _stop_worker_with_psutil(
        self, worker_id: str, process_info: WorkerProcessInfo, force: bool
    ) -> bool:
        """
        Stop a worker process using psutil for process management.

        Args:
            worker_id: Unique identifier of the worker
            process_info: WorkerProcessInfo containing process details
            force: Whether to forcefully kill the process

        Returns:
            bool: True if worker was stopped successfully
        """
        try:
            pid = process_info.pid
            if not pid:
                return False

            process = psutil.Process(pid)

            if force:
                self.logger.info(f"Force killing worker {worker_id} (PID {pid})")
                process.kill()
            else:
                self.logger.info(f"Stopping worker {worker_id} (PID {pid}) gracefully")
                process.terminate()

                try:
                    process.wait(timeout=10)
                except psutil.TimeoutExpired:
                    self.logger.warning(f"Worker {worker_id} did not terminate gracefully, killing")
                    process.kill()
                    process.wait(timeout=5)

            return True

        except psutil.NoSuchProcess:
            self.logger.warning(f"Process for worker {worker_id} not found")
            return True
        except Exception as e:
            self.logger.error(f"Error stopping worker {worker_id} with psutil: {e}")
            return False

    def _stop_worker_without_psutil(
        self, worker_id: str, process_info: WorkerProcessInfo, force: bool
    ) -> bool:
        """
        Stop a worker process using standard subprocess methods.

        Fallback method when psutil is not available.

        Args:
            worker_id: Unique identifier of the worker
            process_info: WorkerProcessInfo containing process details
            force: Whether to forcefully kill the process

        Returns:
            bool: True if worker was stopped successfully
        """
        try:
            process = process_info.process
            if not process:
                return False

            if force:
                self.logger.info(f"Force killing worker {worker_id}")
                process.kill()
            else:
                self.logger.info(f"Stopping worker {worker_id} gracefully")
                process.terminate()

                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.logger.warning(f"Worker {worker_id} did not terminate gracefully, killing")
                    process.kill()
                    process.wait(timeout=5)

            return True

        except Exception as e:
            self.logger.error(f"Error stopping worker {worker_id} without psutil: {e}")
            return False

    def stop_all_workers(self, force: bool = False) -> Dict[str, bool]:
        """
        Stop all registered worker processes.

        Args:
            force: Whether to forcefully kill all processes

        Returns:
            Dict[str, bool]: Dictionary mapping worker IDs to stop status
        """
        with self._lock:
            results = {}
            worker_ids = list(self._worker_processes.keys())

            self.logger.info(f"Stopping all workers (force={force})")

            for worker_id in worker_ids:
                if self._health_checker:
                    self._health_checker.mark_worker_manually_stopped(worker_id)

            for worker_id in worker_ids:
                results[worker_id] = self.stop_worker(worker_id, force)

            success_count = sum(1 for success in results.values() if success)
            self.logger.info(f"Stopped {success_count}/{len(results)} workers")

            return results

    def restart_worker(self, worker_id: str, delay: float = 0.0) -> bool:
        """
        Restart a worker process.

        Stops the worker and then starts it again after the specified delay.

        Args:
            worker_id: Unique identifier of the worker
            delay: Delay in seconds before restarting

        Returns:
            bool: True if worker was restarted successfully
        """
        self.logger.info(f"Restarting worker {worker_id} with delay {delay}s")

        if self._health_checker:
            self._health_checker.mark_worker_manually_stopped(worker_id)

        self.stop_worker(worker_id, force=True)

        if delay > 0:
            time.sleep(delay)

        if self._health_checker:
            self._health_checker.clear_worker_manually_stopped(worker_id)

        return self.start_worker(worker_id)

    def restart_healthy_workers(self, exclude_ids: List[str] = None) -> Dict[str, bool]:
        """
        Restart all healthy (running) workers.

        Args:
            exclude_ids: List of worker IDs to exclude from restart

        Returns:
            Dict[str, bool]: Dictionary mapping worker IDs to restart status
        """
        exclude_ids = exclude_ids or []
        results = {}

        with self._lock:
            for worker_id, process_info in self._worker_processes.items():
                if worker_id in exclude_ids:
                    self.logger.info(f"Skipping excluded worker {worker_id}")
                    continue

                if process_info.status == WorkerStatus.RUNNING:
                    config = self._worker_configs.get(worker_id, {})
                    delay = config.get("restart_delay", 5.0)
                    results[worker_id] = self.restart_worker(worker_id, delay)

        success_count = sum(1 for success in results.values() if success)
        self.logger.info(f"Restarted {success_count}/{len(results)} healthy workers")

        return results

    def get_all_worker_info(self) -> Dict[str, WorkerProcessInfo]:
        """
        Get information about all worker processes.

        Returns:
            Dict[str, WorkerProcessInfo]: Dictionary mapping worker IDs to process info
        """
        with self._lock:
            return dict(self._worker_processes)

    def get_worker_info(self, worker_id: str) -> Optional[WorkerProcessInfo]:
        """
        Get information about a specific worker process.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            Optional[WorkerProcessInfo]: WorkerProcessInfo or None if not found
        """
        with self._lock:
            return self._worker_processes.get(worker_id)

    def get_all_worker_statuses(self) -> Dict[str, WorkerStatus]:
        """
        Get the status of all worker processes.

        Returns:
            Dict[str, WorkerStatus]: Dictionary mapping worker IDs to status
        """
        with self._lock:
            return {
                worker_id: self.get_worker_status(worker_id)
                for worker_id in self._worker_processes.keys()
            }

    def is_worker_running(self, worker_id: str) -> bool:
        """
        Check if a worker is currently running.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            bool: True if worker is running
        """
        return self.get_worker_status(worker_id) == WorkerStatus.RUNNING

    def get_running_workers(self) -> List[str]:
        """
        Get list of all running worker IDs.

        Returns:
            List[str]: List of running worker IDs
        """
        with self._lock:
            return [
                worker_id
                for worker_id in self._worker_processes.keys()
                if self.is_worker_running(worker_id)
            ]

    def get_worker_status(self, worker_id: str) -> Optional[WorkerStatus]:
        """
        Get the status of a specific worker.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            Optional[WorkerStatus]: WorkerStatus or None if not found
        """
        with self._lock:
            if worker_id not in self._worker_processes:
                return None

            process_info = self._worker_processes[worker_id]

            if process_info.process and process_info.process.poll() is None:
                return WorkerStatus.RUNNING
            else:
                return process_info.status

    def get_worker_pid(self, worker_id: str) -> Optional[int]:
        """
        Get the process ID of a worker.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            Optional[int]: Process ID or None if not found
        """
        with self._lock:
            if worker_id not in self._worker_processes:
                return None

            return self._worker_processes[worker_id].pid

    def get_worker_logs(self, worker_id: str) -> List[str]:
        """
        Get the logs captured from a worker process.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            List[str]: List of log lines
        """
        with self._lock:
            return list(self._worker_logs.get(worker_id, []))

    def clear_worker_logs(self, worker_id: str):
        """
        Clear the logs for a worker.

        Args:
            worker_id: Unique identifier of the worker
        """
        with self._lock:
            if worker_id in self._worker_logs:
                self._worker_logs[worker_id] = []
            self.logger.info(f"Cleared logs for worker {worker_id}")

    def shutdown(self):
        """
        Shutdown the ProcessManager.

        Stops all worker processes and cleans up resources.
        """
        self.logger.info("Shutting down process manager")
        self._shutdown_event.set()

        with self._lock:
            for worker_id in list(self._worker_processes.keys()):
                self.stop_worker(worker_id, force=True)
