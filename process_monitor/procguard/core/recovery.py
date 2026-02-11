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

import logging
import threading
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from .cluster_state import WorkerStatus, ClusterState
from .manager import ProcessManager
from .health_checker import HealthStatus, WorkerHealthReport


class RecoveryStage(Enum):
    """
    Enumeration of recovery process stages.

    Attributes:
        DETECTED: Failure has been detected
        STOPPING: Stopping failed worker and related processes
        REASSIGNING: Reassigning tasks from failed worker
        RESTARTING: Restarting the failed worker
        COMPLETED: Recovery completed successfully
        FAILED: Recovery failed
    """
    DETECTED = "detected"
    STOPPING = "stopping"
    REASSIGNING = "reassigning"
    RESTARTING = "restarting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class RecoveryContext:
    """
    Data class representing the context of a recovery operation.

    Attributes:
        failed_worker_id: ID of the worker that failed
        failed_worker_tasks: List of task IDs that were assigned to failed worker
        target_worker_id: ID of worker assigned to handle recovery
        stage: Current RecoveryStage
        start_time: Timestamp when recovery started
        error_message: Error message if recovery failed
    """
    failed_worker_id: str
    failed_worker_tasks: List[str]
    target_worker_id: Optional[str]
    stage: RecoveryStage
    start_time: float
    error_message: Optional[str] = None


class RecoveryOrchestrator:
    """
    Orchestrates recovery operations for failed workers.

    This class manages the recovery process when workers fail, including
    stopping failed workers, reassigning tasks, and restarting workers.
    It supports automatic and manual recovery modes with configurable
    behavior through recovery_config.

    Attributes:
        state_manager: ClusterState instance for tracking worker states
        process_manager: ProcessManager for controlling worker processes
        recovery_config: Configuration dictionary for recovery behavior
        _active_recoveries: Dictionary of currently active recovery operations
        _recovery_history: List of past recovery operations

    Examples:
        >>> orchestrator = RecoveryOrchestrator(state_manager, process_manager, config)
        >>> orchestrator.handle_failure(["worker_1"])
        {'worker_1': True}
    """

    def __init__(
        self, state_manager: ClusterState, process_manager: ProcessManager, recovery_config: dict
    ):
        """
        Initialize the RecoveryOrchestrator.

        Args:
            state_manager: ClusterState instance for tracking worker states
            process_manager: ProcessManager for controlling worker processes
            recovery_config: Dictionary containing recovery configuration:
                - enable_auto_recovery: Whether to auto-recover failed workers
                - stop_all_on_failure: Whether to stop all workers on failure
                - task_reassignment: Whether to reassign tasks
                - recovery_timeout: Timeout for recovery operations
                - max_recovery_attempts: Maximum recovery attempts per worker
        """
        self.state_manager = state_manager
        self.process_manager = process_manager
        self.recovery_config = recovery_config

        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

        self._enable_auto_recovery = recovery_config.get("enable_auto_recovery", True)
        self._stop_all_on_failure = recovery_config.get("stop_all_on_failure", True)
        self._task_reassignment = recovery_config.get("task_reassignment", True)
        self._recovery_timeout = recovery_config.get("recovery_timeout", 60.0)
        self._max_recovery_attempts = recovery_config.get("max_recovery_attempts", 3)

        self._active_recoveries: Dict[str, RecoveryContext] = {}

        self._on_recovery_start: Optional[Callable[[str], None]] = None
        self._on_recovery_complete: Optional[Callable[[str, bool], None]] = None

        self._recovery_history: List[Dict] = []

    def on_recovery_start(self, callback: Callable[[str], None]):
        """
        Register a callback for recovery start events.

        Args:
            callback: Function that takes worker_id when recovery starts
        """
        self._on_recovery_start = callback

    def on_recovery_complete(self, callback: Callable[[str, bool], None]):
        """
        Register a callback for recovery completion events.

        Args:
            callback: Function that takes (worker_id, success) when recovery completes
        """
        self._on_recovery_complete = callback

    def handle_failure(self, failed_worker_ids: List[str]) -> Dict[str, bool]:
        """
        Handle failures for a list of workers.

        Initiates recovery for each failed worker and returns the
        success status for each recovery operation.

        Args:
            failed_worker_ids: List of worker IDs that have failed

        Returns:
            Dict[str, bool]: Dictionary mapping worker IDs to recovery success
        """
        results = {}

        for failed_worker_id in failed_worker_ids:
            with self._lock:
                if failed_worker_id in self._active_recoveries:
                    self.logger.warning(f"Recovery already in progress for {failed_worker_id}")
                    results[failed_worker_id] = False
                    continue

            recovery_context = RecoveryContext(
                failed_worker_id=failed_worker_id,
                failed_worker_tasks=[],
                target_worker_id=None,
                stage=RecoveryStage.DETECTED,
                start_time=time.time(),
            )

            self._active_recoveries[failed_worker_id] = recovery_context

            if self._on_recovery_start:
                self._on_recovery_start(failed_worker_id)

            success = self._execute_recovery(failed_worker_id)

            results[failed_worker_id] = success

            recovery_context.stage = RecoveryStage.COMPLETED if success else RecoveryStage.FAILED
            self._record_recovery_history(recovery_context)

            del self._active_recoveries[failed_worker_id]

            if self._on_recovery_complete:
                self._on_recovery_complete(failed_worker_id, success)

        return results

    def _execute_recovery(self, failed_worker_id: str) -> bool:
        """
        Execute the recovery procedure for a failed worker.

        Args:
            failed_worker_id: Unique identifier of the failed worker

        Returns:
            bool: True if recovery succeeded, False otherwise
        """
        recovery_context = self._active_recoveries.get(failed_worker_id)
        if not recovery_context:
            return False

        if not self._enable_auto_recovery:
            self.logger.info("Auto-recovery disabled, skipping recovery")
            return True

        try:
            self.logger.info(f"Starting recovery for worker: {failed_worker_id}")

            if self._stop_all_on_failure:
                self.logger.info("Stopping all workers due to failure")
                self.process_manager.stop_all_workers()
                time.sleep(2.0)

            if self._task_reassignment:
                self.logger.info("Reassigning tasks from failed worker")

            self.logger.info(f"Restarting worker: {failed_worker_id}")
            success = self.process_manager.restart_worker(failed_worker_id)

            if success:
                self.logger.info(f"Successfully recovered worker: {failed_worker_id}")
            else:
                self.logger.error(f"Failed to recover worker: {failed_worker_id}")

            return success

        except Exception as e:
            self.logger.error(f"Error during recovery: {e}")
            recovery_context.error_message = str(e)
            return False

    def _record_recovery_history(self, context: RecoveryContext):
        """
        Record a recovery operation in the history.

        Args:
            context: RecoveryContext containing recovery details
        """
        entry = {
            "worker_id": context.failed_worker_id,
            "stage": context.stage.value,
            "duration": time.time() - context.start_time,
            "success": context.stage == RecoveryStage.COMPLETED,
            "error": context.error_message,
        }
        self._recovery_history.append(entry)

        if len(self._recovery_history) > 100:
            self._recovery_history = self._recovery_history[-100:]

    def get_recovery_stats(self) -> Dict[str, any]:
        """
        Get statistics about recovery operations.

        Returns:
            Dict[str, any]: Dictionary containing:
                - total_recoveries: Total number of recovery operations
                - successful_recoveries: Number of successful recoveries
                - failed_recoveries: Number of failed recoveries
                - success_rate: Percentage of successful recoveries
        """
        total = len(self._recovery_history)
        successful = sum(1 for r in self._recovery_history if r["success"])

        return {
            "total_recoveries": total,
            "successful_recoveries": successful,
            "failed_recoveries": total - successful,
            "success_rate": successful / total if total > 0 else 0.0,
        }

    def get_recovery_history(self) -> List[Dict]:
        """
        Get the history of recovery operations.

        Returns:
            List[Dict]: List of recovery history entries (up to 100 most recent)
        """
        return list(self._recovery_history)
