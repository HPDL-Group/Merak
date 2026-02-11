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

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging


class CommunicationAdapter(ABC):
    """
    Abstract base class for communication adapters that handle
    communication between the manager and workers.

    This class defines the interface that all communication adapters
    must implement. Communication adapters are responsible for
    sending commands, receiving status updates, and managing
    heartbeat messages between the manager and worker processes.

    Args:
        config: Configuration dictionary containing adapter-specific settings

    Attributes:
        config: The configuration dictionary passed during initialization
        logger: Logger instance for the adapter
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the communication adapter with the given configuration.

        Args:
            config: Dictionary containing adapter configuration options
        """
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def send_command(self, worker_id: str, command: str, **kwargs) -> bool:
        """
        Send a command to a specific worker.

        Args:
            worker_id: Unique identifier of the target worker
            command: Command to send (e.g., 'start', 'stop', 'restart')
            **kwargs: Additional command-specific parameters

        Returns:
            bool: True if command was sent successfully, False otherwise
        """
        pass

    @abstractmethod
    def send_heartbeat(self, worker_id: str) -> bool:
        """
        Send a heartbeat signal to indicate the worker is alive.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            bool: True if heartbeat was sent successfully, False otherwise
        """
        pass

    @abstractmethod
    def receive_status(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Receive status information from a worker.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            Optional[Dict]: Dictionary containing worker status information,
                           or None if status could not be retrieved
        """
        pass

    @abstractmethod
    def broadcast_command(
        self, command: str, exclude_workers: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Broadcast a command to all registered workers.

        Args:
            command: Command to broadcast to all workers
            exclude_workers: List of worker IDs to exclude from broadcast

        Returns:
            Dict[str, bool]: Dictionary mapping worker IDs to their
                            command execution status
        """
        pass

    @abstractmethod
    def shutdown(self):
        """
        Clean up resources and close connections.

        This method should be called when shutting down the adapter
        to ensure proper cleanup of sockets, connections, and other
        resources.
        """
        pass

    def initialize(self):
        """
        Initialize the adapter and prepare for communication.

        Default implementation logs the initialization. Subclasses
        can override this to perform additional setup operations.
        """
        self.logger.info(f"Initializing {self.__class__.__name__} adapter")

    def health_check(self) -> bool:
        """
        Check if the adapter is healthy and operational.

        Returns:
            bool: True if the adapter is healthy, False otherwise
        """
        return True
