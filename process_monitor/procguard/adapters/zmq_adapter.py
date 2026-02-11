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
import hashlib
from datetime import datetime
from .base import CommunicationAdapter


class ZMQCommunicationAdapter(CommunicationAdapter):
    """
    ZMQ-based implementation of CommunicationAdapter for real distributed communication.

    This adapter uses ZeroMQ (ZMQ) sockets to provide real network communication
    between the manager and workers. It supports sending commands, heartbeats,
    and receiving status updates over TCP connections.

    Attributes:
        _context: ZMQ context for managing sockets
        _sockets: Dictionary of active ZMQ sockets per worker
        _host: Host address for ZMQ connections
        _port_base: Base port number for worker connections
        _timeout: Timeout in milliseconds for socket operations

    Examples:
        >>> adapter = ZMQCommunicationAdapter({
        ...     "host": "0.0.0.0",
        ...     "port_base": 5550,
        ...     "timeout": 1000
        ... })
        >>> adapter.initialize()
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the ZMQ communication adapter.

        Args:
            config: Configuration dictionary containing ZMQ settings:
                - host: Host address to bind to (default: "127.0.0.1")
                - port_base: Base port number for worker connections (default: 5550)
                - timeout: Socket timeout in milliseconds (default: 1000)
        """
        super().__init__(config)
        self._context = None
        self._sockets: Dict[str, Any] = {}
        self._host = config.get("host", "127.0.0.1")
        self._port_base = config.get("port_base", 5550)
        self._timeout = config.get("timeout", 1000)

    def initialize(self):
        """
        Initialize the ZMQ context and prepare for connections.

        Attempts to import the ZMQ library and create a context.
        Falls back to mock behavior if ZMQ is not installed.
        """
        try:
            import zmq

            self._context = zmq.Context()
            self.logger.info(f"ZMQ adapter initialized on {self._host}:{self._port_base}")
        except ImportError:
            self.logger.warning("ZMQ not installed, falling back to mock behavior")
            self.logger.warning("Install pyzmq: pip install pyzmq")

    def send_command(self, worker_id: str, command: str, **kwargs) -> bool:
        """
        Send a command to a specific worker via ZMQ.

        Establishes a connection to the worker's ZMQ socket and
        sends the command as a JSON message.

        Args:
            worker_id: Unique identifier of the target worker
            command: Command to send (e.g., 'start', 'stop', 'restart')
            **kwargs: Additional command-specific parameters

        Returns:
            bool: True if command was sent successfully, False otherwise
        """
        try:
            import zmq

            port = self._get_worker_port(worker_id)
            socket = self._get_socket(worker_id, port)

            message = {"command": command, "worker_id": worker_id, "kwargs": kwargs}

            socket.send_json(message)
            self.logger.debug(f"Sent command '{command}' to worker {worker_id} on port {port}")
            return True

        except ImportError:
            self.logger.debug(f"Mock mode: Sending command '{command}' to worker {worker_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send command to worker {worker_id}: {e}")
            return False

    def send_heartbeat(self, worker_id: str) -> bool:
        """
        Send a heartbeat signal to a worker via ZMQ.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            bool: True if heartbeat was sent successfully, False otherwise
        """
        try:
            import zmq

            port = self._get_worker_port(worker_id)
            socket = self._get_socket(worker_id, port)

            message = {"command": "heartbeat", "worker_id": worker_id}

            socket.send_json(message)
            self.logger.debug(f"Sent heartbeat to worker {worker_id}")
            return True

        except ImportError:
            self.logger.debug(f"Mock mode: Sending heartbeat to worker {worker_id}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to send heartbeat to worker {worker_id}: {e}")
            return False

    def receive_status(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """
        Receive status information from a worker via ZMQ.

        Polls the worker's socket for status information with
        the configured timeout.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            Optional[Dict]: Dictionary containing worker status,
                           or None if timeout or error occurred
        """
        try:
            import zmq

            port = self._get_worker_port(worker_id)
            socket = self._get_socket(worker_id, port)

            if socket.poll(timeout=self._timeout):
                status = socket.recv_json()
                self.logger.debug(f"Received status from worker {worker_id}")
                return status
            else:
                self.logger.warning(f"Timeout receiving status from worker {worker_id}")
                return None

        except ImportError:
            self.logger.debug(f"Mock mode: Receiving status from worker {worker_id}")
            return {"worker_id": worker_id, "status": "running", "timestamp": self._get_timestamp()}
        except Exception as e:
            self.logger.error(f"Failed to receive status from worker {worker_id}: {e}")
            return None

    def broadcast_command(
        self, command: str, exclude_workers: Optional[List[str]] = None
    ) -> Dict[str, bool]:
        """
        Broadcast a command to all connected workers.

        Args:
            command: Command to broadcast to all workers
            exclude_workers: List of worker IDs to exclude from broadcast

        Returns:
            Dict[str, bool]: Dictionary mapping worker IDs to their
                            command execution status
        """
        exclude_workers = exclude_workers or []
        results = {}

        for worker_id in self._sockets.keys():
            if worker_id not in exclude_workers:
                results[worker_id] = self.send_command(worker_id, command)

        return results

    def shutdown(self):
        """
        Clean up ZMQ resources and close all connections.

        Closes all worker sockets and terminates the ZMQ context
        to ensure proper cleanup.
        """
        self.logger.info("Shutting down ZMQ communication adapter")

        for socket in self._sockets.values():
            try:
                socket.close()
            except Exception:
                pass

        self._sockets.clear()

        if self._context:
            try:
                self._context.term()
            except Exception:
                pass

    def _get_worker_port(self, worker_id: str) -> int:
        """
        Calculate the ZMQ port for a specific worker.

        Uses a hash of the worker_id to generate a consistent
        port number in the range [port_base, port_base + 100).

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            int: Port number for the worker's ZMQ socket
        """
        hash_val = int(hashlib.md5(worker_id.encode()).hexdigest()[:8], 16)
        return self._port_base + (hash_val % 100)

    def _get_socket(self, worker_id: str, port: int):
        """
        Get or create a ZMQ socket for a worker.

        Establishes a REQ socket connected to the worker's address
        and caches it for reuse.

        Args:
            worker_id: Unique identifier of the worker
            port: Port number for the connection

        Returns:
            ZMQ socket instance for the worker
        """
        import zmq

        if worker_id not in self._sockets:
            socket = self._context.socket(zmq.REQ)
            socket.setsockopt(zmq.LINGER, 0)
            socket.connect(f"tcp://{self._host}:{port}")
            self._sockets[worker_id] = socket

        return self._sockets[worker_id]

    def _get_timestamp(self) -> str:
        """
        Generate a timestamp string in ISO format.

        Returns:
            str: Current timestamp in ISO format
        """
        return datetime.now().isoformat()
