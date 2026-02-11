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

import json
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, asdict


class WorkerStatus(Enum):
    """
    Enumeration of possible worker states.

    Attributes:
        UNKNOWN: Worker state is unknown or not yet determined
        STARTING: Worker process is being launched
        RUNNING: Worker is actively running
        STOPPED: Worker has been stopped
        FAILED: Worker has failed unexpectedly
        ZOMBIE: Worker process is a zombie (stopped but not cleaned up)
    """
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPED = "stopped"
    FAILED = "failed"
    ZOMBIE = "zombie"


@dataclass
class WorkerState:
    """
    Data class representing the current state of a worker.

    Attributes:
        worker_id: Unique identifier for the worker
        pid: Process ID of the worker process
        status: Current WorkerStatus
        start_time: Timestamp when worker started
        last_heartbeat: Timestamp of last heartbeat received
        restart_count: Number of times worker has been restarted
        last_error: Error message if worker failed
        assigned_tasks: List of task IDs assigned to this worker
    """
    worker_id: str
    pid: Optional[int] = None
    status: WorkerStatus = WorkerStatus.UNKNOWN
    start_time: Optional[str] = None
    last_heartbeat: Optional[str] = None
    restart_count: int = 0
    last_error: Optional[str] = None
    assigned_tasks: List[str] = None

    def __post_init__(self):
        if self.assigned_tasks is None:
            self.assigned_tasks = []


@dataclass
class TaskAssignment:
    """
    Data class representing a task assignment to a worker.

    Attributes:
        task_id: Unique identifier for the task
        worker_id: ID of the worker assigned to this task
        command: Command to execute for this task
        status: Current status of the task
        created_at: Timestamp when task was created
    """
    task_id: str
    worker_id: str
    command: str
    status: str = "pending"
    created_at: str = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


class ClusterState:
    """
    Manages the state of the cluster and all registered workers.

    This class provides thread-safe operations for tracking worker status,
    managing task assignments, and persisting cluster state to disk.
    In the current configuration, local cache persistence is disabled
    and all state is managed in memory.

    Attributes:
        state_file: Path to the state persistence file
        auto_save: Whether to automatically save state changes
        decoupled_mode: Whether running in decoupled mode
        _lock: Reentrant lock for thread-safe operations
        _workers: Dictionary of worker_id to WorkerState
        _tasks: Dictionary of task_id to TaskAssignment
        _task_queue: List of pending task IDs
        _failed_workers: List of worker IDs that have failed

    Examples:
        >>> state = ClusterState()
        >>> state.register_worker("worker_1")
        >>> state.update_worker_status("worker_1", WorkerStatus.RUNNING)
        >>> state.get_state_summary()
        {'total_workers': 1, 'running_workers': 1, ...}
    """

    def __init__(
        self,
        state_file: str = "procguard_state.json",
        auto_save: bool = True,
        decoupled_mode: bool = False,
    ):
        """
        Initialize the ClusterState manager.

        Args:
            state_file: Path to the state persistence file
            auto_save: Whether to automatically save state on changes
            decoupled_mode: Whether running in decoupled mode
        """
        self.state_file = Path(state_file)
        self.auto_save = auto_save
        self.decoupled_mode = decoupled_mode
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

        self._workers: Dict[str, WorkerState] = {}
        self._tasks: Dict[str, TaskAssignment] = {}
        self._task_queue: List[str] = []
        self._failed_workers: List[str] = []

        self.logger.info("[ClusterState] Local cache reading disabled")

    def _load_state(self):
        """
        Load cluster state from disk.

        Note: In the current configuration, local cache loading is disabled
        and state is managed entirely in memory.
        """
        self.logger.info("[ClusterState] Local cache reading disabled, skipping procguard_state.json")

    def save_state(self):
        """
        Save the current cluster state to disk.

        This method is called automatically when auto_save is enabled
        and state changes occur.
        """
        if self.auto_save:
            self._save_state()

    def _save_state(self):
        """
        Internal method to persist state to disk.

        Saves worker states as JSON to the configured state file.
        """
        try:
            data = {
                "workers": {worker_id: asdict(state) for worker_id, state in self._workers.items()}
            }

            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved state to {self.state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save state: {e}")

    def register_worker(self, worker_id: str):
        """
        Register a new worker with the cluster.

        Creates a new WorkerState entry for the worker and initializes
        its status to UNKNOWN.

        Args:
            worker_id: Unique identifier for the worker
        """
        with self._lock:
            if worker_id not in self._workers:
                self._workers[worker_id] = WorkerState(worker_id=worker_id)
                self.logger.info(f"Registered worker: {worker_id}")
                self.save_state()

    def update_worker_status(self, worker_id: str, status: WorkerStatus):
        """
        Update the status of a registered worker.

        Args:
            worker_id: Unique identifier of the worker
            status: New WorkerStatus to set
        """
        with self._lock:
            if worker_id in self._workers:
                self._workers[worker_id].status = status
                self.save_state()

    def get_worker_status(self, worker_id: str) -> Optional[WorkerStatus]:
        """
        Get the current status of a worker.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            Optional[WorkerStatus]: Current status or None if worker not found
        """
        with self._lock:
            if worker_id in self._workers:
                return self._workers[worker_id].status
            return None

    def get_running_workers(self) -> List[str]:
        """
        Get list of all workers currently in RUNNING state.

        Returns:
            List[str]: List of worker IDs with RUNNING status
        """
        with self._lock:
            return [
                worker_id
                for worker_id, state in self._workers.items()
                if state.status == WorkerStatus.RUNNING
            ]

    def mark_worker_failed(self, worker_id: str):
        """
        Mark a worker as failed and add to failed workers list.

        Args:
            worker_id: Unique identifier of the failed worker
        """
        with self._lock:
            if worker_id not in self._failed_workers:
                self._failed_workers.append(worker_id)
                self.logger.info(f"Worker marked as failed: {worker_id}")
                self.save_state()

    def clear_failed_workers(self):
        """
        Clear the list of failed workers.

        This is typically called after successful recovery of failed workers.
        """
        with self._lock:
            self._failed_workers.clear()
            self.logger.info("Cleared failed workers")
            self.save_state()

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current cluster state.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - total_workers: Total number of registered workers
                - running_workers: Number of workers in RUNNING state
                - failed_workers: Number of failed workers
                - total_tasks: Total number of tasks
                - queued_tasks: Number of tasks in queue
        """
        with self._lock:
            return {
                "total_workers": len(self._workers),
                "running_workers": len(self.get_running_workers()),
                "failed_workers": len(self._failed_workers),
                "total_tasks": len(self._tasks),
                "queued_tasks": len(self._task_queue),
            }
