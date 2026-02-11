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

from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
from .base import CommunicationAdapter


class MockCommunicationAdapter(CommunicationAdapter):
    """
    Mock implementation of CommunicationAdapter for testing and development.

    This adapter simulates communication between the manager and workers
    without actual network communication. It stores all commands, heartbeats,
    and worker states in memory for testing and debugging purposes.

    Attributes:
        _worker_states: Dictionary storing the state of each worker
        _command_history: List of all commands sent to workers
        _heartbeat_count: Dictionary tracking heartbeat counts per worker

    Examples:
        >>> adapter = MockCommunicationAdapter({"timeout": 5})
        >>> adapter.send_command("worker_1", "start")
        True
        >>> adapter.get_command_history()
        [{'worker_id': 'worker_1', 'command': 'start', ...}]
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the mock communication adapter.

        Args:
            config: Configuration dictionary (not used in mock adapter)
        """
        super().__init__(config)
        self._worker_states: Dict[str, Dict[str, Any]] = {}
        self._command_history: List[Dict[str, Any]] = []
        self._heartbeat_count: Dict[str, int] = {}

    def send_command(self, worker_id: str, command: str, **kwargs) -> bool:
        """
        Record a command sent to a worker.

        Stores the command in the command history for testing and
        debugging purposes.

        Args:
            worker_id: Unique identifier of the target worker
            command: Command to send to the worker
            **kwargs: Additional command-specific parameters

        Returns:
            bool: Always returns True for mock adapter
        """
        self.logger.debug(f"Mock: Sending command '{command}' to worker {worker_id}")
        self._command_history.append(
            {
                "worker_id": worker_id,
                "command": command,
                "kwargs": kwargs,
                "timestamp": self._get_timestamp(),
            }
        )
        return True

    def send_heartbeat(self, worker_id: str) -> bool:
        """
        Record a heartbeat from a worker.

        Tracks the number of heartbeats received from each worker
        for monitoring purposes.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            bool: Always returns True for mock adapter
        """
        self.logger.debug(f"Mock: Sending heartbeat to worker {worker_id}")
        self._heartbeat_count[worker_id] = self._heartbeat_count.get(worker_id, 0) + 1
        return True

    def receive_status(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve the stored status of a worker.

        Returns the worker state that was previously set, or a default
        running status if no state has been set.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            Optional[Dict]: Dictionary containing worker status information,
                           or default status if not found
        """
        self.logger.debug(f"Mock: Receiving status from worker {worker_id}")
        return self._worker_states.get(
            worker_id,
            {"worker_id": worker_id, "status": "running", "timestamp": self._get_timestamp()},
        )

    def broadcast_command(
        self, command: str, exclude_workers: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Record a broadcast command sent to multiple workers.

        Simulates broadcasting a command to all workers except those
        specified in the exclude list.

        Args:
            command: Command to broadcast to all workers
            exclude_workers: List of worker IDs to exclude from broadcast

        Returns:
            Dict[str, bool]: Dictionary mapping worker IDs to success status
        """
        self.logger.debug(f"Mock: Broadcasting command '{command}'")
        exclude_workers = exclude_workers or []
        results = {}

        for worker_id in self._worker_states.keys():
            if worker_id not in exclude_workers:
                results[worker_id] = self.send_command(worker_id, command)

        return results

    def shutdown(self):
        """
        Clean up the mock adapter state.

        Clears all stored worker states, command history, and heartbeat
        counts to reset the adapter to its initial state.
        """
        self.logger.info("Mock: Shutting down communication adapter")
        self._worker_states.clear()
        self._command_history.clear()
        self._heartbeat_count.clear()

    def set_worker_state(self, worker_id: str, state: Dict[str, Any]):
        """
        Set the state of a worker for testing purposes.

        Allows tests to directly set worker state to simulate
        different scenarios.

        Args:
            worker_id: Unique identifier of the worker
            state: Dictionary containing worker state information
        """
        self._worker_states[worker_id] = state

    def get_command_history(self) -> List[Dict[str, Any]]:
        """
        Retrieve the history of all commands sent.

        Returns:
            List[Dict]: List of command records with worker_id, command,
                       kwargs, and timestamp
        """
        return list(self._command_history)

    def get_heartbeat_count(self, worker_id: str) -> int:
        """
        Get the number of heartbeats received from a worker.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            int: Number of heartbeats received from the worker
        """
        return self._heartbeat_count.get(worker_id, 0)

    def _get_timestamp(self) -> str:
        """
        Generate a timestamp string in ISO format.

        Returns:
            str: Current timestamp in ISO format
        """
        return datetime.now().isoformat()
