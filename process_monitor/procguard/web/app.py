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
Web Monitor - Flask-based web interface for ProcGuard.

This module provides a web-based monitoring interface for ProcGuard using
Flask for HTTP endpoints and Flask-SocketIO for real-time updates. It
supports managing local and remote workers, PyTorch distributed training
configuration, and worker group management.

Features:
- Real-time worker status monitoring via WebSocket
- Worker control (start, stop, restart)
- PyTorch distributed training configuration
- Worker grouping and management
- Remote worker registration and heartbeat

Example:
    >>> from procguard import ProcGuard
    >>> from procguard.web import WebMonitor
    >>> app = ProcGuard()
    >>> web_monitor = WebMonitor(app, host="0.0.0.0", port=5000)
    >>> web_monitor.run()
"""

import json
import logging
import os
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
from dataclasses import dataclass, field

from procguard.core.ha_manager import (
    AssociationState,
    FailoverTrigger,
    AssociationConfig,
    AssociationStatus,
)

from procguard.core.group_manager import GroupConfig


_pytorch_config_cache_loaded = False


def load_pytorch_config() -> Dict[str, Any]:
    """
    Load PyTorch distributed training configuration.

    Returns cached configuration with basic and advanced settings for
    PyTorch distributed training including master address, port, backend,
    and NCCL configuration options.

    Returns:
        Dict[str, Any]: Configuration dictionary containing basic and advanced settings
    """
    global _pytorch_config_cache_loaded
    if not _pytorch_config_cache_loaded:
        logger = logging.getLogger(__name__)
        logger.debug("[WebAPI] Using default PyTorch config (local cache disabled)")
        _pytorch_config_cache_loaded = True

    default_config = {
        "basic": {
            "master_addr": None,
            "master_port": 29500,
            "world_size": None,
            "node_count": 1,
            "processes_per_node": 1,
            "backend": "nccl",
        },
        "advanced": {
            "cuda_visible_devices": None,
            "nccl_socket_ifname": None,
        },
    }

    return default_config


def save_pytorch_config(config: Dict[str, Any]) -> bool:
    """
    Save PyTorch distributed training configuration.

    Args:
        config: Configuration dictionary to save

    Returns:
        bool: True if configuration was saved successfully, False otherwise
    """
    config_path = get_pytorch_config_path()
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to save PyTorch config: {e}")
        return False


def get_pytorch_env_vars(
    local_rank: int = 0,
    node_rank: int = 0,
    processes_per_node: int = None,
    rank: int = None,
    group_config: GroupConfig = None,
) -> Dict[str, str]:
    """
    Generate environment variables for PyTorch distributed training.

    Constructs the environment variables required for PyTorch distributed
    training based on configuration including master address, port, rank,
    and backend settings.

    Args:
        local_rank: Local rank of the current process
        node_rank: Rank of the current node
        processes_per_node: Number of processes per node
        rank: Global rank of the process
        group_config: Group-level PyTorch configuration (optional)

    Returns:
        Dict[str, str]: Environment variables for PyTorch distributed training
    """
    if group_config:
        config = {
            "basic": {
                "master_addr": group_config.master_addr,
                "master_port": group_config.master_port,
                "world_size": group_config.world_size,
                "backend": group_config.backend,
                "processes_per_node": processes_per_node,
            },
            "advanced": {
                "cuda_visible_devices": group_config.cuda_visible_devices,
                "nccl_socket_ifname": group_config.nccl_socket_ifname,
            },
        }
    else:
        config = load_pytorch_config()

    env_vars = {}

    basic = config.get("basic") or {}

    if basic.get("master_addr"):
        env_vars["MASTER_ADDR"] = str(basic["master_addr"])
        if basic.get("master_port"):
            env_vars["MASTER_PORT"] = str(basic["master_port"])
        if basic.get("world_size") is not None:
            env_vars["WORLD_SIZE"] = str(basic["world_size"])
        env_vars["TORCH_DISTRIBUTED_BACKEND"] = basic.get("backend", "nccl")
        env_vars["LOCAL_RANK"] = str(local_rank)
        env_vars["NODE_RANK"] = str(node_rank)

        if rank is not None:
            env_vars["RANK"] = str(rank)
        else:
            actual_procs = (
                processes_per_node
                if processes_per_node is not None
                else (basic.get("processes_per_node") or 1)
            )
            if basic.get("world_size") and basic.get("world_size") > 0:
                env_vars["RANK"] = str(node_rank * actual_procs + local_rank)

    if config.get("advanced"):
        advanced = config["advanced"]
        if advanced.get("cuda_visible_devices"):
            env_vars["CUDA_VISIBLE_DEVICES"] = str(advanced["cuda_visible_devices"])
        if advanced.get("nccl_socket_ifname"):
            env_vars["NCCL_SOCKET_IFNAME"] = str(advanced["nccl_socket_ifname"])

    return env_vars


def extract_node_name(worker_id: str) -> str:
    """
    Extract node name from worker_id.

    Extracts the node name portion from a worker ID that follows the
    pattern 'nodename-localrank' (e.g., 'gn39-0' -> 'gn39').

    Args:
        worker_id: Worker identifier string

    Returns:
        str: Extracted node name
    """
    if "-" in worker_id:
        return "-".join(worker_id.rsplit("-", 1)[:-1])
    return worker_id


class RemoteWorkerInfo:
    """
    Information about a remote worker process.

    Stores metadata for remote workers including status, PID, command,
    registration time, and heartbeat information.

    Attributes:
        worker_id: Unique identifier for the worker
        command: Command executed by the worker
        working_dir: Working directory for the worker
        status: Current status (running, stopped, etc.)
        pid: Process ID of the worker
        restart_count: Number of times worker has been restarted
        last_heartbeat: Timestamp of last heartbeat received
        pending_command: Command to be sent to the worker
        registered_at: Timestamp when worker was registered
        logs: List of log messages from the worker
        is_failover_worker: Whether this worker was started via HA failover
        failover_info: Details about failover if applicable
        lock: Thread lock for thread-safe operations
    """

    def __init__(self, worker_id: str, command: str, working_dir: Optional[str] = None):
        """
        Initialize RemoteWorkerInfo.

        Args:
            worker_id: Unique identifier for the worker
            command: Command executed by the worker
            working_dir: Optional working directory
        """
        self.worker_id = worker_id
        self.command = command
        self.working_dir = working_dir
        self.status = "stopped"
        self.pid = None
        self.restart_count = 0
        self.last_heartbeat = None
        self.pending_command = None
        self.registered_at = datetime.now()
        self.logs = []
        self.is_failover_worker = False
        self.failover_info = None
        self.lock = threading.RLock()


class WebMonitor:
    """
    Web-based monitoring interface for ProcGuard.

    This class provides a Flask web application with Socket.IO support for
    real-time monitoring and control of ProcGuard workers. It supports
    both local workers managed by ProcGuard and remote workers that register
    via the API.

    Attributes:
        procguard: Reference to the ProcGuard instance
        host: Host address for the web server
        port: Port number for the web server
        app: Flask application instance
        socketio: Socket.IO instance for real-time communication

    Example:
        >>> from procguard import ProcGuard
        >>> from procguard.web import WebMonitor
        >>> app = ProcGuard()
        >>> web = WebMonitor(app, host="0.0.0.0", port=5000)
        >>> web.run()
    """

    def __init__(self, procguard_instance, host: str = "0.0.0.0", port: int = 5000):
        """
        Initialize the WebMonitor.

        Args:
            procguard_instance: ProcGuard instance to monitor
            host: Host address to bind to
            port: Port number to listen on
        """
        self.procguard = procguard_instance
        self.host = host
        self.port = port
        self._startup_time = datetime.now().isoformat()
        self._server_version = str(time.time())

        self.app = Flask(__name__, template_folder="templates", static_folder="static")
        self.app.config["SECRET_KEY"] = "procguard-secret-key"
        self.app.config["SOCKETIO_ASYNC_MODE"] = "threading"
        self.app.config["SOCKETIO_LOGGER"] = False
        self.app.config["ENGINEIO_LOGGER"] = False

        self.socketio = SocketIO(
            self.app,
            cors_allowed_origins="*",
            async_mode="threading",
            logger=False,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25,
        )
        self.logger = logging.getLogger(__name__)

        self._remote_workers: Dict[str, RemoteWorkerInfo] = {}
        self._remote_workers_lock = threading.RLock()
        self._worker_states: Dict[str, str] = {}
        self._worker_states_lock = threading.RLock()
        self._failed_ranks_cache: Dict[str, List[int]] = {}
        self._failed_ranks_cache_time: Dict[str, float] = {}
        self._failed_ranks_cache_ttl: float = 5.0
        self._failed_ranks_fetch_count: Dict[str, int] = {}

        self._group_configs: Dict[str, Dict[str, Any]] = {}
        self._group_configs_lock = threading.RLock()

        self._worker_ranks: Dict[str, int] = {}
        self._worker_ranks_lock = threading.RLock()

        self._heartbeat_check_interval = 5
        self._heartbeat_timeout = 15
        self._heartbeat_check_running = False
        self._heartbeat_check_thread = None

        self._ha_associations: Dict[str, AssociationConfig] = {}
        self._ha_associations_lock = threading.RLock()
        self._ha_status_cache: Dict[str, AssociationStatus] = {}
        self._ha_monitor_threads: Dict[str, threading.Thread] = {}
        self._ha_monitoring_active: Dict[str, bool] = {}

        self._group_state: Dict[str, Dict[str, Any]] = {}
        self._group_state_lock = threading.RLock()

        self._setup_routes()
        self._setup_socketio_events()

        @self.app.after_request
        def after_request(response):
            response.headers.add("Access-Control-Allow-Origin", "*")
            response.headers.add(
                "Access-Control-Allow-Headers", "Content-Type,Authorization"
            )
            response.headers.add(
                "Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS"
            )
            return response

        self._start_heartbeat_check()

    def _start_heartbeat_check(self):
        """
        Start the heartbeat check thread if not already running.
        """
        if (
            self._heartbeat_check_thread is not None
            and self._heartbeat_check_thread.is_alive()
        ):
            self.logger.debug("[WebMonitor] Heartbeat check thread already running")
            return

        self._heartbeat_check_running = True
        self._heartbeat_check_thread = threading.Thread(
            target=self._heartbeat_check_loop, daemon=True, name="HeartbeatCheckThread"
        )
        self._heartbeat_check_thread.start()
        self.logger.info("[WebMonitor] Started heartbeat check thread")

    def _stop_heartbeat_check(self):
        """Stop the heartbeat check thread."""
        self._heartbeat_check_running = False
        if self._heartbeat_check_thread is not None:
            self._heartbeat_check_thread.join(timeout=5)
            self.logger.info("[WebMonitor] Stopped heartbeat check thread")

    def _heartbeat_check_loop(self):
        """Background loop that periodically checks worker heartbeats."""
        while self._heartbeat_check_running:
            try:
                self._check_heartbeat_timeout()
            except Exception as e:
                self.logger.error(f"[WebMonitor] Error in heartbeat check: {e}")

            time.sleep(self._heartbeat_check_interval)

    def _check_heartbeat_timeout(self):
        """
        Check for workers that have missed heartbeats and update their status.

        Workers that are running but have not sent a heartbeat within the
        timeout period are marked as stopped. Workers that are already stopped
        and have timed out are removed.
        """
        with self._remote_workers_lock:
            if not self._remote_workers:
                return

            current_time = datetime.now()
            workers_to_update = []
            workers_to_remove = []

            self.logger.debug(
                f"[WebMonitor] Checking heartbeat timeout for {len(self._remote_workers)} workers"
            )

            for worker_id, worker in self._remote_workers.items():
                status_str = f"status={worker.status}"
                if worker.last_heartbeat:
                    elapsed = (current_time - worker.last_heartbeat).total_seconds()
                    status_str += f", last_heartbeat={elapsed:.1f}s ago"
                    if elapsed > self._heartbeat_timeout:
                        if worker.status == "running":
                            workers_to_update.append((worker_id, elapsed))
                            self.logger.debug(
                                f"[WebMonitor] Worker {worker_id} timeout, will mark as stopped"
                            )
                        elif worker.status == "stopped":
                            workers_to_remove.append((worker_id, elapsed))
                            self.logger.debug(
                                f"[WebMonitor] Worker {worker_id} stopped and timeout, will remove"
                            )
                else:
                    status_str += ", no heartbeat"
                self.logger.debug(f"[WebMonitor] Worker {worker_id}: {status_str}")

            self.logger.info(
                f"[WebMonitor] Heartbeat check: {len(workers_to_update)} to stop, {len(workers_to_remove)} to remove"
            )

            for worker_id, elapsed in workers_to_update:
                worker = self._remote_workers[worker_id]
                old_status = worker.status
                worker.status = "stopped"
                self.logger.warning(
                    f"[WebMonitor] Worker {worker_id} heartbeat timeout ({elapsed:.1f}s), "
                    f"status: {old_status} -> stopped"
                )
                self.broadcast_worker_update(
                    worker_id,
                    {
                        "action": "status_changed",
                        "status": "stopped",
                        "reason": "heartbeat_timeout",
                    },
                )
                self._detect_state_change(worker_id, old_status, "stopped")

            for worker_id, elapsed in workers_to_remove:
                worker = self._remote_workers.pop(worker_id, None)
                if worker:
                    self.logger.info(
                        f"[WebMonitor] Worker {worker_id} stopped and heartbeat timeout ({elapsed:.1f}s), removing"
                    )
                    self.broadcast_worker_update(
                        worker_id,
                        {"action": "unregistered", "reason": "heartbeat_timeout_cleanup"},
                    )
                    
                    if self.procguard and self.procguard.group_manager:
                        group_id = self.procguard.group_manager.get_worker_group(worker_id)
                        if group_id:
                            self.procguard.group_manager.remove_worker_from_group(group_id, worker_id)
                            self.logger.info(f"[WebMonitor] Removed worker {worker_id} from group {group_id} due to timeout")

    def _detect_state_change(self, worker_id: str, old_status: str, new_status: str):
        """Detect worker state changes and cache failed ranks by group.

        When a worker transitions from running to failed/stopped/exited state,
        caches the worker's rank for its group for failover detection.

        Args:
            worker_id: Worker identifier
            old_status: Previous status
            new_status: New status
        """
        if old_status == "running" and new_status in ["stopped", "failed", "exited"]:
            self.logger.info(f"[WebMonitor] Worker {worker_id} 状态变化: {old_status} -> {new_status}")

            if not self.procguard or not getattr(self.procguard, 'group_manager', None):
                return

            try:
                group_info = self.procguard.group_manager.get_group_of_worker(worker_id)
                if not group_info:
                    return

                group = group_info.group_id

                with self._worker_ranks_lock:
                    rank = self._worker_ranks.get(worker_id)

                if rank is not None:
                    current_time = time.time()

                    if group not in self._failed_ranks_cache:
                        self._failed_ranks_cache[group] = []
                        self._failed_ranks_cache_time[group] = current_time
                        self._failed_ranks_fetch_count[group] = 0

                    if rank not in self._failed_ranks_cache[group]:
                        self._failed_ranks_cache[group].append(rank)
                        self._failed_ranks_cache[group] = sorted(self._failed_ranks_cache[group])
                        self._failed_ranks_cache_time[group] = current_time

                    with self._group_state_lock:
                        if group not in self._group_failover_state:
                            self._group_failover_state[group] = {
                                "should_stop": False,
                                "is_recovering": False,
                                "failed_ranks": [],
                                "recover_src_rank": None,
                                "failover_workers": [],
                                "failover_info": {},
                                "acknowledged_workers": set(),
                                "timestamp": None
                            }

                        if not self._group_failover_state[group]["should_stop"]:
                            self._group_failover_state[group]["should_stop"] = True
                            if rank not in self._group_failover_state[group]["failed_ranks"]:
                                self._group_failover_state[group]["failed_ranks"].append(rank)
                                self._group_failover_state[group]["failed_ranks"] = sorted(self._group_failover_state[group]["failed_ranks"])

                            self._group_failover_state[group]["is_recovering"] = True
                            self._group_failover_state[group]["recover_src_rank"] = rank
                            self._group_failover_state[group]["acknowledged_workers"] = set()
                            self._group_failover_state[group]["timestamp"] = time.time()

                            self.logger.info(
                                f"[WebMonitor] Group '{group}' failover triggered: should_stop=True, failed_ranks={self._group_failover_state[group]['failed_ranks']}"
                            )
            except Exception:
                pass

    def _get_all_workers(self) -> Dict[str, Dict[str, Any]]:
        """Get all workers from both local and remote sources."""
        workers = {}

        local_workers = self.procguard.process_manager.get_all_worker_info()
        for worker_id, info in local_workers.items():
            workers[worker_id] = {
                "worker_id": worker_id,
                "status": info.status.value if info.status else "unknown",
                "pid": info.pid,
                "command": info.command,
                "type": "local",
            }

        with self._remote_workers_lock:
            for worker_id, worker in self._remote_workers.items():
                if worker_id not in workers:
                    workers[worker_id] = {
                        "worker_id": worker_id,
                        "status": worker.status,
                        "pid": worker.pid,
                        "command": worker.command,
                        "type": "remote",
                    }

        return workers

    def _get_worker_group(self, worker_id: str) -> Optional[str]:
        """Get the group ID that a worker belongs to."""
        if not self.procguard.group_manager:
            return None

        group = self.procguard.group_manager.get_group_of_worker(worker_id)
        if group:
            return group.group_id
        return None

    def _get_group_workers(self, group_id: str) -> List[str]:
        """Get list of worker IDs in a group."""
        if not self.procguard.group_manager:
            return []

        group = self.procguard.group_manager.get_group(group_id)
        if group:
            return group.workers
        return []

    def _get_group_world_size(self, group_id: str) -> int:
        """Get the world size for a group from its config."""
        if not self.procguard.group_manager:
            return 0

        try:
            config = self.procguard.group_manager.get_group_config(group_id)
            if config:
                return config.world_size or 0
        except Exception:
            pass

        workers = self._get_all_workers()
        count = 0
        for wid in workers:
            if self._get_worker_group(wid) == group_id:
                count += 1
        return count

    def _check_ha_association_health(
        self, config: AssociationConfig
    ) -> Optional[AssociationStatus]:
        """Check the health of an HA association.

        Args:
            config: HA association configuration

        Returns:
            AssociationStatus with current health status, or None if association should be removed
        """
        all_workers = self._get_all_workers()

        active_workers_in_group = []
        standby_workers_in_group = []

        for worker_id, worker in all_workers.items():
            group = self._get_worker_group(worker_id)
            if group == config.active_group_id:
                active_workers_in_group.append(worker_id)
            elif group == config.standby_group_id:
                standby_workers_in_group.append(worker_id)

        if not active_workers_in_group or not standby_workers_in_group:
            self.logger.info(
                f"[HA] Removing association {config.association_id}: active_workers={len(active_workers_in_group)}, standby_workers={len(standby_workers_in_group)}"
            )
            with self._ha_associations_lock:
                if config.association_id in self._ha_associations:
                    del self._ha_associations[config.association_id]
                    self.logger.info(
                        f"[HA] Removed association {config.association_id} (no workers in active or standby group)"
                    )
            return None

        active_running = 0
        active_failed = 0
        standby_available = 0
        standby_running = 0

        for worker_id, worker in all_workers.items():
            group = self._get_worker_group(worker_id)
            if group == config.active_group_id:
                if worker["status"] == "running":
                    active_running += 1
                    self.logger.debug(
                        f"[HA] Worker {worker_id} in active group: running"
                    )
                else:
                    active_failed += 1
                    self.logger.debug(
                        f"[HA] Worker {worker_id} in active group: {worker['status']} (counted as failed)"
                    )
            elif group == config.standby_group_id:
                if worker["status"] == "stopped":
                    standby_available += 1
                    self.logger.debug(
                        f"[HA] Worker {worker_id} in standby group: stopped (available)"
                    )
                elif worker["status"] == "running":
                    standby_running += 1
                    self.logger.debug(
                        f"[HA] Worker {worker_id} in standby group: running"
                    )

        self.logger.info(
            f"[HA] Health check for {config.association_id}: active_running={active_running}, active_failed={active_failed}, standby_available={standby_available}, standby_running={standby_running}"
        )

        failover_needed = active_failed >= config.failover_threshold

        if failover_needed:
            state = AssociationState.FAILING_OVER
        elif active_running >= config.world_size:
            state = AssociationState.ACTIVE
        elif active_running > 0:
            state = AssociationState.DEGRADED
        else:
            state = AssociationState.STANDBY

        if failover_needed:
            health_status = "failover_required"
        elif active_running < config.world_size and standby_available == 0:
            health_status = "no_standby"
        elif active_failed > 0:
            health_status = "degraded"
        else:
            health_status = "healthy"

        status = AssociationStatus(
            association_id=config.association_id,
            state=state,
            active_workers=active_running,
            standby_available=standby_available,
            health_status=health_status,
            last_health_check=datetime.now().isoformat(),
        )

        self._ha_status_cache[config.association_id] = status
        return status

    def _ha_start_worker(
        self, worker_id: str, pytorch_env: Dict[str, str] = None
    ) -> bool:
        """Start a worker, handling both local and remote workers.

        Args:
            worker_id: ID of the worker to start
            pytorch_env: Optional PyTorch environment variables to set for this worker

        Returns:
            True if start was initiated successfully
        """
        with self._remote_workers_lock:
            if worker_id in self._remote_workers:
                self._remote_workers[worker_id].pending_command = "start"
                if pytorch_env:
                    self._remote_workers[worker_id]._failover_pytorch_env = pytorch_env
                return True
            else:
                self.logger.warning(f"[HA] Worker {worker_id} not in _remote_workers")

        if self.procguard.process_manager:
            return self.procguard.process_manager.start_worker(
                worker_id, custom_env=pytorch_env
            )

        self.logger.warning(f"[HA] No process manager available")
        return False

    def _ha_stop_worker_for_failover(self, worker_id: str) -> bool:
        """Stop a worker and mark it as stopped for failover.

        Args:
            worker_id: ID of the worker to stop

        Returns:
            True if stop was initiated successfully
        """
        self.logger.info(f"[HA] Trying to stop worker {worker_id}")

        with self._remote_workers_lock:
            if worker_id in self._remote_workers:
                self._remote_workers[worker_id].pending_command = None
                self._remote_workers[worker_id].status = "stopped"
                self.logger.info(f"[HA] Marked remote worker {worker_id} as stopped")
                return True

        if self.procguard.process_manager:
            self.logger.info(f"[HA] process_manager available, calling stop_worker")
            self.logger.info(
                f"[HA] Registered workers: {list(self.procguard._worker_processes.keys()) if hasattr(self.procguard, '_worker_processes') else 'N/A'}"
            )

            success = self.procguard.process_manager.stop_worker(worker_id, force=False)
            self.logger.info(f"[HA] stop_worker result for {worker_id}: {success}")
            if success:
                self.logger.info(f"[HA] Stopped local worker {worker_id}")
            return success

        self.logger.warning(
            f"[HA] No process manager available to stop worker {worker_id}"
        )
        return False

    def _mark_failover_worker(
        self,
        worker_id: str,
        association_id: str,
        replaced_worker_id: Optional[str] = None,
    ):
        """Mark a worker as a failover replacement.

        Args:
            worker_id: The worker that was started as a failover replacement
            association_id: The HA association ID
            replaced_worker_id: The original failed worker that was replaced
        """
        with self._remote_workers_lock:
            if worker_id in self._remote_workers:
                self._remote_workers[worker_id].is_failover_worker = True
                self._remote_workers[worker_id].failover_info = {
                    "association_id": association_id,
                    "replaced_worker_id": replaced_worker_id,
                    "failover_time": datetime.now().isoformat(),
                }
                self.logger.info(
                    f"[HA] Marked worker {worker_id} as failover worker (replaced {replaced_worker_id})"
                )
            else:
                self.logger.warning(
                    f"[HA] Worker {worker_id} not found when marking as failover"
                )

    def _execute_ha_failover(self, config: AssociationConfig) -> int:
        """Execute failover for an HA association.

        Moves failed workers from active to standby, starts standby workers,
        moves them to active, and updates PyTorch config.

        Args:
            config: HA association configuration

        Returns:
            Number of workers moved
        """
        self.logger.info(f"[HA] Executing failover for {config.association_id}")

        all_workers = self._get_all_workers()
        failed_or_stopped = []
        standby_available = []

        for worker_id, worker in all_workers.items():
            group = self._get_worker_group(worker_id)
            status = worker.get("status", "unknown")

            if group == config.active_group_id:
                if status in ["failed", "exited", "unknown", "stopped"]:
                    failed_or_stopped.append(worker_id)
            elif group == config.standby_group_id:
                if status == "stopped":
                    standby_available.append(worker_id)

        self.logger.info(
            f"[HA] Active group has {len(failed_or_stopped)} failed/stopped workers"
        )
        self.logger.info(
            f"[HA] Standby group has {len(standby_available)} stopped workers available"
        )

        running_in_active = sum(
            1
            for w_id, w in all_workers.items()
            if self._get_worker_group(w_id) == config.active_group_id
            and w.get("status") == "running"
        )

        target_count = max(0, config.world_size - running_in_active)
        self.logger.info(
            f"[HA] Running in active: {running_in_active}, target: {config.world_size}, need to move: {target_count}"
        )

        moved_count = 0

        failed_pytorch_env_map = {}
        for worker_id in failed_or_stopped:
            if self.procguard.group_manager:
                self.logger.info(
                    f"[HA] Stopping worker {worker_id} before moving to standby"
                )
                self._ha_stop_worker_for_failover(worker_id)

                failed_rank = self._worker_ranks.get(worker_id)
                if failed_rank is not None:
                    group = self._get_worker_group(worker_id)
                    group_config = (
                        self.procguard.group_manager.get_group_config(group)
                        if group and self.procguard.group_manager
                        else None
                    )
                    if group_config:
                        local_rank = 0
                        node_rank = 0
                        if "-" in worker_id:
                            try:
                                local_rank = int(worker_id.rsplit("-", 1)[-1])
                            except (ValueError, IndexError):
                                local_rank = 0

                        pytorch_env = {
                            "MASTER_ADDR": group_config.master_addr or "",
                            "MASTER_PORT": str(group_config.master_port or 29500),
                            "WORLD_SIZE": str(group_config.world_size or 0),
                            "LOCAL_RANK": str(local_rank),
                            "NODE_RANK": str(node_rank),
                            "RANK": str(failed_rank),
                            "TORCH_DISTRIBUTED_BACKEND": group_config.backend or "nccl",
                        }
                        failed_pytorch_env_map[worker_id] = pytorch_env
                        self.logger.info(
                            f"[HA] Captured full PyTorch env from failed worker {worker_id}: RANK={failed_rank}, WORLD_SIZE={group_config.world_size}"
                        )

                self.procguard.group_manager.remove_worker_from_group(
                    config.active_group_id, worker_id
                )
                self.procguard.group_manager.add_worker_to_group(
                    config.standby_group_id, worker_id
                )
                self.logger.info(
                    f"[HA] Moved failed/stopped worker {worker_id} from active to standby"
                )
                moved_count += 1

        workers_to_move = min(len(standby_available), target_count)
        workers_started = []

        for i, worker_id in enumerate(standby_available[:workers_to_move]):
            self.logger.info(f"[HA] Processing worker {worker_id} from standby group")

            if self.procguard.group_manager:
                self.procguard.group_manager.remove_worker_from_group(
                    config.standby_group_id, worker_id
                )
                self.procguard.group_manager.add_worker_to_group(
                    config.active_group_id, worker_id
                )
                self.logger.info(f"[HA] Moved worker {worker_id} to active group")

            if i < len(failed_or_stopped):
                failed_worker = failed_or_stopped[i]
                if failed_worker in failed_pytorch_env_map:
                    pytorch_env = failed_pytorch_env_map[failed_worker]
                    self._worker_ranks[worker_id] = int(pytorch_env["RANK"])
                    self.logger.info(
                        f"[HA] Transferred PyTorch env from {failed_worker} to {worker_id}: RANK={pytorch_env['RANK']}, WORLD_SIZE={pytorch_env['WORLD_SIZE']}"
                    )

            self._update_group_pytorch_config(config.active_group_id, config)
            self.logger.info(f"[HA] Updated PyTorch config for active group")

            pytorch_env = None
            if i < len(failed_or_stopped):
                failed_worker = failed_or_stopped[i]
                if failed_worker in failed_pytorch_env_map:
                    pytorch_env = failed_pytorch_env_map[failed_worker]
                    self._worker_ranks[worker_id] = int(pytorch_env["RANK"])
                    self.logger.info(
                        f"[HA] Will transfer PyTorch env from {failed_worker} to {worker_id}: RANK={pytorch_env['RANK']}"
                    )

            if not self._ha_start_worker(worker_id, pytorch_env):
                self.logger.warning(f"[HA] Failed to start worker {worker_id}")
                continue

            self._mark_failover_worker(
                worker_id,
                config.association_id,
                failed_or_stopped[i] if i < len(failed_or_stopped) else None,
            )

            self.logger.info(
                f"[HA] Waiting for worker {worker_id} to start and become healthy..."
            )
            time.sleep(3)

            self.logger.info(f"[HA] Worker {worker_id} started successfully")
            workers_started.append(worker_id)
            moved_count += 1

            time.sleep(1)

        if workers_started:
            self._update_group_pytorch_config(config.active_group_id, config)
            self.logger.info(f"[HA] Updated PyTorch config for active group")
            self.logger.info(
                f"[HA] Started and moved {len(workers_started)} workers to active group"
            )

        if moved_count > 0:
            config.failover_count += 1
            config.last_failover = datetime.now().isoformat()

        self.logger.info(
            f"[HA] Failover complete: moved {moved_count} workers, started {len(workers_started)}"
        )
        return moved_count

    def _update_group_pytorch_config(self, group_id: str, config: AssociationConfig):
        """Update PyTorch configuration for a group.

        Uses the most recent configuration - prefers group config if it exists,
        only uses association config for new groups or when world_size needs update.

        Args:
            group_id: Group ID to update
            config: HA association configuration
        """
        if not self.procguard.group_manager:
            return

        group = self.procguard.group_manager.get_group(group_id)
        if not group:
            self.logger.warning(f"[HA] Group {group_id} not found for config update")
            return

        existing_config = group.config
        world_size = config.world_size if group_id == config.active_group_id else 0

        if existing_config.master_addr is not None:
            master_addr = existing_config.master_addr
            master_port = existing_config.master_port
            backend = existing_config.backend or config.backend or "nccl"
            self.logger.info(
                f"[HA] Using existing group config: master={master_addr}, backend={backend}"
            )
        else:
            master_addr = config.master_addr or "localhost"
            master_port = config.master_port or 29500
            backend = config.backend or "nccl"
            self.logger.info(
                f"[HA] Using association config: master={master_addr}, backend={backend}"
            )

        from procguard.core.group_manager import GroupConfig

        new_config = GroupConfig(
            master_addr=master_addr,
            master_port=master_port,
            world_size=world_size,
            backend=backend,
        )

        self.procguard.group_manager.update_group_config(group_id, new_config)
        self.logger.info(
            f"[HA] Updated PyTorch config for group {group_id}: world_size={world_size}, master={master_addr}:{master_port}"
        )

    def _ha_monitor_loop(self, association_id: str):
        """Background monitoring loop for HA association.

        Args:
            association_id: ID of the association to monitor
        """
        self.logger.info(f"[HA] Started monitoring loop for {association_id}")

        failover_cooldown_until = 0

        while self._ha_monitoring_active.get(association_id, False):
            try:
                config = None
                with self._ha_associations_lock:
                    if association_id not in self._ha_associations:
                        break
                    config = self._ha_associations[association_id]
                    if not config.enabled:
                        continue

                current_time = time.time()
                if current_time < failover_cooldown_until:
                    self.logger.debug(
                        f"[HA] Skipping health check for {association_id} (cooldown until {failover_cooldown_until})"
                    )
                    time.sleep(5.0)
                    continue

                status = self._check_ha_association_health(config)

                if status is None:
                    self.logger.info(
                        f"[HA] Stopping monitoring for {association_id} (association removed)"
                    )
                    break

                if status.health_status == "failover_required" and config.auto_failover:
                    self.logger.info(
                        f"[HA] Auto-triggering failover for {association_id}"
                    )
                    moved = self._execute_ha_failover(config)
                    self.logger.info(
                        f"[HA] Auto failover completed: moved {moved} workers"
                    )

                    if moved > 0:
                        failover_cooldown_until = current_time + 30.0
                        self.logger.info(
                            f"[HA] Cooldown enabled until {failover_cooldown_until} (30 seconds)"
                        )

            except Exception as e:
                self.logger.error(f"[HA] Error in monitoring loop: {e}")
                import traceback

                self.logger.error(f"[HA] Traceback: {traceback.format_exc()}")

            time.sleep(5.0)

        self.logger.info(f"[HA] Stopped monitoring loop for {association_id}")

    def _execute_ha_failover_safe(self, config: AssociationConfig) -> int:
        """Thread-safe version of failover execution.

        Args:
            config: HA association configuration

        Returns:
            Number of workers moved
        """
        try:
            return self._execute_ha_failover(config)
        except Exception as e:
            self.logger.error(f"[HA] Failover execution error: {e}")
            import traceback

            self.logger.error(f"[HA] Traceback: {traceback.format_exc()}")
            return 0

    def _setup_routes(self):
        """Configure Flask routes for the web interface."""

        @self.app.route("/")
        def index():
            return render_template("index.html")

        @self.app.route("/ha_associations")
        def ha_associations():
            return render_template("ha_associations.html")

        @self.app.route("/api/status")
        def get_status():
            """Get overall status of ProcGuard and all workers."""
            status = self.procguard.get_status()

            workers = {}
            for worker_id, worker_status in status["workers"].items():
                workers[worker_id] = (
                    worker_status.value
                    if hasattr(worker_status, "value")
                    else str(worker_status)
                )

            if "remote_workers" in status:
                for worker_id, worker in status["remote_workers"].items():
                    workers[worker_id] = worker["status"]

            status["workers"] = workers

            total = len(workers)
            running = sum(1 for s in workers.values() if s == "running")
            status["summary"] = {
                "total_workers": total,
                "running_workers": running,
                "stopped_workers": total - running,
                "decoupled_mode": status.get("decoupled_mode", False),
            }

            ranks = {}
            worker_to_group_rank = {}

            if self.procguard.group_manager:
                all_groups = self.procguard.group_manager.get_all_groups()
                for group_id, group in all_groups.items():
                    for idx, worker_id in enumerate(group.workers):
                        worker_to_group_rank[worker_id] = idx

            for worker_id in workers.keys():
                rank = None
                local_rank = 0
                if "-" in worker_id:
                    try:
                        local_rank = int(worker_id.rsplit("-", 1)[-1])
                    except ValueError:
                        local_rank = 0

                if worker_id in worker_to_group_rank:
                    rank = worker_to_group_rank[worker_id]
                    self.logger.debug(
                        f"[WebAPI] Worker {worker_id} rank in group: {rank}"
                    )

                with self._worker_ranks_lock:
                    if worker_id in self._worker_ranks:
                        rank = self._worker_ranks[worker_id]

                ranks[worker_id] = {
                    "rank": rank,
                    "local_rank": local_rank,
                    "node_rank": rank,
                }
            status["ranks"] = ranks

            return jsonify(status)

        @self.app.route("/api/server/info")
        def get_server_info():
            """获取服务器信息，用于检测重启"""
            return jsonify({
                "startup_time": self._startup_time,
                "version": self._server_version,
                "timestamp": datetime.now().isoformat()
            })

        @self.app.route("/api/workers")
        def get_workers():
            """Get information about all workers."""
            workers = self.procguard.process_manager.get_all_worker_info()
            result = {}
            for worker_id, info in workers.items():
                result[worker_id] = {
                    "worker_id": worker_id,
                    "status": info.status.value if info.status else "unknown",
                    "pid": info.pid,
                    "command": info.command,
                    "start_time": info.start_time,
                    "restart_count": info.restart_count,
                    "type": "local",
                }

            with self._remote_workers_lock:
                for worker_id, worker in self._remote_workers.items():
                    if worker_id not in result:
                        result[worker_id] = {
                            "worker_id": worker_id,
                            "status": worker.status,
                            "pid": worker.pid,
                            "command": worker.command,
                            "start_time": (
                                worker.registered_at.isoformat()
                                if worker.registered_at
                                else None
                            ),
                            "restart_count": worker.restart_count,
                            "type": "remote",
                            "last_heartbeat": (
                                worker.last_heartbeat.isoformat()
                                if worker.last_heartbeat
                                else None
                            ),
                            "is_failover_worker": worker.is_failover_worker,
                            "failover_info": worker.failover_info,
                        }

            return jsonify(result)

        @self.app.route("/api/health")
        def get_health():
            """Get health reports for all workers."""
            health_reports = self.procguard.health_checker.get_all_health_reports()
            result = {}
            for worker_id, report in health_reports.items():
                result[worker_id] = {
                    "worker_id": worker_id,
                    "status": report.status.value,
                    "cpu_percent": report.cpu_percent,
                    "memory_percent": report.memory_percent,
                    "last_check_time": report.last_check_time,
                    "error_message": report.error_message,
                }
            return jsonify(result)

        @self.app.route("/api/recovery")
        def get_recovery():
            """Get recovery statistics and history."""
            stats = self.procguard.recovery_orchestrator.get_recovery_stats()
            history = self.procguard.recovery_orchestrator.get_recovery_history()
            return jsonify({"stats": stats, "history": history[-10:]})

        @self.app.route("/api/state")
        def get_state():
            """Get cluster state summary."""
            summary = self.procguard.state_manager.get_state_summary()

            with self._remote_workers_lock:
                remote_count = len(self._remote_workers)
                if remote_count > 0:
                    summary["total_workers"] = (
                        summary.get("total_workers", 0) + remote_count
                    )
                    summary["remote_workers"] = remote_count

            return jsonify(summary)

        @self.app.route("/api/workers/<worker_id>/restart", methods=["POST"])
        def restart_worker(worker_id):
            """Restart a worker."""
            with self._remote_workers_lock:
                if worker_id in self._remote_workers:
                    self._remote_workers[worker_id].pending_command = "restart"
                    return jsonify({"success": True, "type": "remote"})

            success = self.procguard.process_manager.restart_worker(worker_id)
            return jsonify({"success": success, "type": "local"})

        @self.app.route("/api/workers/<worker_id>/stop", methods=["POST"])
        def stop_worker(worker_id):
            """Stop a worker."""
            force = request.json.get("force", False) if request.json else False

            with self._remote_workers_lock:
                if worker_id in self._remote_workers:
                    old_status = self._remote_workers[worker_id].status
                    command = "kill" if force else "stop"
                    self._remote_workers[worker_id].pending_command = command
                    self._remote_workers[worker_id].status = "stopped"
                    self._detect_state_change(worker_id, old_status, "stopped")
                    return jsonify({"success": True, "type": "remote"})

            old_status = self.procguard.process_manager.get_worker_status(worker_id)
            success = self.procguard.process_manager.stop_worker(worker_id, force=force)

            if success:
                new_status = "stopped"
                self._detect_state_change(worker_id, old_status, new_status)

            return jsonify({"success": success, "type": "local"})

        @self.app.route("/api/workers/<worker_id>/start", methods=["POST"])
        def start_worker(worker_id):
            """Start a worker."""
            with self._remote_workers_lock:
                if worker_id in self._remote_workers:
                    failover_pytorch_env = getattr(
                        self._remote_workers[worker_id], "_failover_pytorch_env", None
                    )
                    self._remote_workers[worker_id].pending_command = "start"
                    return jsonify({"success": True, "type": "remote"})

            success = self.procguard.process_manager.start_worker(worker_id)
            return jsonify({"success": success, "type": "local"})

        @self.app.route("/api/workers/<worker_id>/logs", methods=["GET"])
        def get_worker_logs(worker_id):
            """Get logs for a worker."""
            with self._remote_workers_lock:
                if worker_id in self._remote_workers:
                    logs = self._remote_workers[worker_id].logs
                    return jsonify({"logs": logs, "type": "remote"})

            logs = self.procguard.process_manager.get_worker_logs(worker_id)
            return jsonify({"logs": logs, "type": "local"})

        @self.app.route("/api/workers/<worker_id>/logs", methods=["DELETE"])
        def clear_worker_logs(worker_id):
            """Clear logs for a worker."""
            with self._remote_workers_lock:
                if worker_id in self._remote_workers:
                    self._remote_workers[worker_id].logs = []
                    return jsonify({"success": True, "type": "remote"})

            self.procguard.process_manager.clear_worker_logs(worker_id)
            return jsonify({"success": True, "type": "local"})

        @self.app.route("/api/workers/<worker_id>/failover-status", methods=["GET"])
        def get_worker_failover_status(worker_id):
            """Get failover status for a specific worker.

            Returns whether the worker is a failover replacement and related info.

            Args:
                worker_id: The worker ID to check

            Returns:
                JSON with is_failover_worker flag and failover_info
            """
            with self._remote_workers_lock:
                if worker_id in self._remote_workers:
                    worker = self._remote_workers[worker_id]
                    return jsonify(
                        {
                            "worker_id": worker_id,
                            "is_failover_worker": worker.is_failover_worker,
                            "failover_info": worker.failover_info,
                        }
                    )

            return (
                jsonify(
                    {
                        "worker_id": worker_id,
                        "is_failover_worker": False,
                        "failover_info": None,
                        "error": "Worker not found in remote workers",
                    }
                ),
                404,
            )

        @self.app.route("/api/workers/<worker_id>/reset-failover-status", methods=["POST"])
        def reset_worker_failover_status(worker_id):
            """Reset failover status for a specific worker.

            This is typically called after a worker has received the failover state
            to clear the is_failover_worker flag.

            Args:
                worker_id: The worker ID to reset

            Returns:
                JSON with success status
            """
            with self._remote_workers_lock:
                if worker_id in self._remote_workers:
                    worker = self._remote_workers[worker_id]
                    worker.is_failover_worker = False
                    worker.failover_info = {}
                    return jsonify(
                        {
                            "success": True,
                            "worker_id": worker_id,
                            "message": "Failover status reset successfully",
                        }
                    )

            return (
                jsonify(
                    {
                        "success": False,
                        "worker_id": worker_id,
                        "error": "Worker not found in remote workers",
                    }
                ),
                404,
            )

        @self.app.route("/api/workers/failover-workers", methods=["GET"])
        def get_failover_workers():
            """Get all workers that were started via failover.

            Returns:
                JSON list of all failover workers with their info
            """
            failover_workers = []

            with self._remote_workers_lock:
                for worker_id, worker in self._remote_workers.items():
                    if worker.is_failover_worker:
                        with self._worker_ranks_lock:
                            rank = self._worker_ranks.get(worker_id)

                        failover_workers.append(
                            {
                                "worker_id": worker_id,
                                "failover_info": worker.failover_info,
                                "status": worker.status,
                                "pid": worker.pid,
                                "rank": rank,
                            }
                        )

            return jsonify(
                {
                    "count": len(failover_workers),
                    "workers": failover_workers,
                }
            )

        @self.app.route("/api/workers/register", methods=["POST"])
        def register_worker():
            """Register a remote worker."""
            data = request.json
            worker_id = data.get("worker_id")
            command = data.get("command")
            working_dir = data.get("working_dir")

            if not worker_id or not command:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "worker_id and command are required",
                        }
                    ),
                    400,
                )

            with self._remote_workers_lock:
                if worker_id in self._remote_workers:
                    self._remote_workers[worker_id].command = command
                    self._remote_workers[worker_id].working_dir = working_dir
                    self._remote_workers[worker_id].status = "stopped"
                    self._remote_workers[worker_id].pid = None
                    self._remote_workers[worker_id].restart_count = 0
                    self._remote_workers[worker_id].logs = []
                    self._remote_workers[worker_id].last_heartbeat = None
                    self._remote_workers[worker_id].registered_at = datetime.now()
                    self.logger.info(f"Updated existing remote worker: {worker_id}")
                else:
                    self._remote_workers[worker_id] = RemoteWorkerInfo(
                        worker_id=worker_id, command=command, working_dir=working_dir
                    )

            self.logger.info(f"Registered remote worker: {worker_id}")

            self.broadcast_worker_update(
                worker_id, {"action": "registered", "command": command, "is_reregistration": worker_id in self._remote_workers}
            )

            return jsonify({"success": True, "worker_id": worker_id})

        @self.app.route("/api/workers/<worker_id>/rank", methods=["POST"])
        def update_worker_rank(worker_id):
            """Update the rank of a worker."""
            data = request.json
            rank = data.get("rank")

            if rank is None:
                return jsonify({"success": False, "error": "rank is required"}), 400

            with self._worker_ranks_lock:
                old_rank = self._worker_ranks.get(worker_id)
                self._worker_ranks[worker_id] = rank
                self.logger.info(
                    f"Updated rank for worker {worker_id}: {old_rank} -> {rank}"
                )

            self.broadcast_worker_update(
                worker_id, {"action": "rank_updated", "rank": rank}
            )

            return jsonify({"success": True, "old_rank": old_rank, "new_rank": rank})

        @self.app.route("/api/workers/<worker_id>/rank", methods=["GET"])
        def get_worker_rank(worker_id):
            """Get the rank of a worker."""
            with self._worker_ranks_lock:
                rank = self._worker_ranks.get(worker_id)

            if rank is not None:
                return jsonify({"success": True, "worker_id": worker_id, "rank": rank})
            else:
                return jsonify({"success": False, "error": "Rank not found for worker"}), 404

        @self.app.route("/api/workers/<worker_id>/heartbeat", methods=["POST"])
        def worker_heartbeat(worker_id):
            data = request.json

            worker_status = None
            worker_pid = None
            worker_restart_count = 0
            worker_source = None

            with self._remote_workers_lock:
                if worker_id in self._remote_workers:
                    worker_source = "remote"
                    worker = self._remote_workers[worker_id]
                    old_status = worker.status
                    new_status = data.get("status", "unknown")
                    worker.status = new_status
                    worker.pid = data.get("pid")
                    worker.restart_count = data.get("restart_count", 0)
                    worker.last_heartbeat = datetime.now()

                    command = worker.pending_command
                    if command:
                        worker.pending_command = None
                        self.logger.info(
                            f"Sending command '{command}' to worker {worker_id}"
                        )

                    if old_status != new_status:
                        self._detect_state_change(worker_id, old_status, new_status)

                    worker_status = worker.status
                    worker_pid = worker.pid
                    worker_restart_count = worker.restart_count

            if worker_source is None and self.procguard.process_manager:
                all_workers = self.procguard.process_manager.get_all_worker_info()
                if worker_id in all_workers:
                    worker_source = "process_manager"
                    worker_status = data.get("status", "running")
                    worker_pid = data.get("pid")
                    self.logger.debug(
                        f"[WebAPI] Worker {worker_id} managed via process_manager"
                    )

            if worker_source is None:
                self.logger.warning(
                    f"[WebAPI] Worker {worker_id} not registered, ignoring heartbeat"
                )
                return (
                    jsonify({"success": False, "error": "Worker not registered"}),
                    404,
                )

            self.broadcast_worker_update(
                worker_id,
                {
                    "status": worker_status,
                    "pid": worker_pid,
                    "restart_count": worker_restart_count,
                },
            )

            pytorch_env = {}

            failover_pytorch_env = None
            if worker_source == "remote" and worker_id in self._remote_workers:
                failover_pytorch_env = getattr(
                    self._remote_workers[worker_id], "_failover_pytorch_env", None
                )

            if failover_pytorch_env:
                pytorch_env = failover_pytorch_env
                self._remote_workers[worker_id]._failover_pytorch_env = None
            else:
                config = load_pytorch_config()
                basic = config.get("basic", {})
                global_processes_per_node = basic.get("processes_per_node", 1)

                local_rank = 0
                if "-" in worker_id:
                    try:
                        local_rank = int(worker_id.split("-")[-1])
                    except ValueError:
                        local_rank = 0

                group_id = None
                group_config = None
                group_workers = []

                if self.procguard.group_manager:
                    worker_group = self.procguard.group_manager.get_group_of_worker(
                        worker_id
                    )
                    if worker_group:
                        group_id = worker_group.group_id
                        group_config_data = (
                            self.procguard.group_manager.get_group_config(group_id)
                        )
                        if group_config_data:
                            group_config = {
                                "name": worker_group.name,
                                "workers": worker_group.workers,
                                "master_addr": group_config_data.master_addr,
                                "master_port": group_config_data.master_port,
                                "world_size": group_config_data.world_size,
                                "backend": group_config_data.backend,
                                "cuda_visible_devices": group_config_data.cuda_visible_devices,
                                "nccl_socket_ifname": group_config_data.nccl_socket_ifname,
                            }
                        group_workers = list(worker_group.workers)
                        self.logger.debug(
                            f"[WebAPI] Worker {worker_id} belongs to group {group_id}"
                        )
                else:
                    self.logger.debug(f"[WebAPI] GroupManager not initialized")

                node_rank = 0
                rank = None

                if group_config:
                    group_name = group_config.get("name", group_id)
                    group_world_size = group_config.get(
                        "world_size", len(group_workers)
                    )
                    master_addr = group_config.get("master_addr")

                    if master_addr and group_world_size:
                        master_node = (
                            extract_node_name(master_addr)
                            if "-" in master_addr
                            else master_addr
                        )
                        current_node = extract_node_name(worker_id)

                        unique_nodes = {}
                        for wid in group_workers:
                            node_name = extract_node_name(wid)
                            if node_name not in unique_nodes:
                                unique_nodes[node_name] = []
                            unique_nodes[node_name].append(wid)

                        sorted_nodes = sorted(unique_nodes.keys())
                        if master_node in sorted_nodes:
                            master_idx = sorted_nodes.index(master_node)
                            if master_idx != 0:
                                sorted_nodes.pop(master_idx)
                                sorted_nodes.insert(0, master_node)

                        if current_node in sorted_nodes:
                            node_rank = sorted_nodes.index(current_node)

                        sorted_workers = []
                        for node in sorted_nodes:
                            sorted_workers.extend(sorted(unique_nodes.get(node, [])))

                        if worker_id in sorted_workers:
                            worker_idx = sorted_workers.index(worker_id)
                            if worker_idx < group_world_size:
                                rank = worker_idx
                                current_node_workers = sorted(
                                    unique_nodes.get(current_node, [])
                                )
                                local_rank = current_node_workers.index(worker_id)

                if rank is not None:
                    pytorch_env = {
                        "MASTER_ADDR": group_config.get("master_addr", ""),
                        "MASTER_PORT": str(group_config.get("master_port", 29500)),
                        "WORLD_SIZE": str(
                            group_config.get("world_size", len(group_workers))
                        ),
                        "LOCAL_RANK": str(local_rank),
                        "NODE_RANK": str(node_rank),
                        "RANK": str(rank),
                        "TORCH_DISTRIBUTED_BACKEND": group_config.get(
                            "backend", "nccl"
                        ),
                    }
                    with self._worker_ranks_lock:
                        old_rank = self._worker_ranks.get(worker_id)
                        if old_rank != rank:
                            self._worker_ranks[worker_id] = rank
                            self.logger.info(
                                f"[WebAPI] Updated rank for {worker_id}: {old_rank} -> {rank}"
                            )

                if not pytorch_env and basic.get("master_addr"):
                    if group_id:
                        self.logger.warning(
                            f"[WebAPI] Worker {worker_id} group {group_id} missing PyTorch config, using global"
                        )
                    else:
                        self.logger.warning(
                            f"[WebAPI] Worker {worker_id} not in any group, using global config"
                        )
                    processes_per_node = global_processes_per_node
                    world_size = basic.get("world_size")
                    master_addr = basic.get("master_addr")

                    if world_size is not None:
                        master_node = (
                            extract_node_name(master_addr)
                            if "-" in master_addr
                            else master_addr
                        )
                        current_node = extract_node_name(worker_id)

                        unique_nodes = {}
                        for wid in list(self._remote_workers.keys()):
                            node_name = extract_node_name(wid)
                            if node_name not in unique_nodes:
                                unique_nodes[node_name] = []
                            unique_nodes[node_name].append(wid)

                        sorted_nodes = sorted(unique_nodes.keys())
                        if master_node in sorted_nodes and master_node != current_node:
                            master_idx = sorted_nodes.index(master_node)
                            if master_idx != 0:
                                sorted_nodes.pop(master_idx)
                                sorted_nodes.insert(0, master_node)

                        if current_node in sorted_nodes:
                            node_rank = sorted_nodes.index(current_node)

                        master_workers = sorted(unique_nodes.get(master_node, []))
                        other_workers = []
                        for node in sorted_nodes:
                            if node != master_node:
                                other_workers.extend(sorted(unique_nodes.get(node, [])))

                        all_active_workers = master_workers + other_workers

                        if worker_id in all_active_workers:
                            worker_idx = all_active_workers.index(worker_id)
                            if worker_idx < world_size:
                                rank = worker_idx

                if not pytorch_env:
                    pytorch_env = get_pytorch_env_vars(
                        local_rank=local_rank, node_rank=node_rank, rank=rank
                    )

            response_data = {
                "success": True,
                "command": command,
                "pytorch_env": pytorch_env,
            }

            if pytorch_env and data.get("config_hash") is not None:
                import hashlib

                pytorch_config = {
                    k: str(v) if v is not None else "" for k, v in pytorch_env.items()
                }
                server_config_str = json.dumps(pytorch_config, sort_keys=True)
                server_hash = int(
                    hashlib.md5(server_config_str.encode()).hexdigest(), 16
                )
                client_hash = data.get("config_hash")
                config_changed = client_hash != server_hash

                worker = self._remote_workers.get(worker_id)
                last_config_hash = (
                    getattr(worker, "_last_config_hash", None) if worker else None
                )

                if config_changed and last_config_hash != server_hash:
                    self.logger.info(f"PyTorch config changed for worker {worker_id}")
                    if worker:
                        worker._last_config_hash = server_hash
                elif not config_changed and last_config_hash != client_hash:
                    if worker:
                        worker._last_config_hash = client_hash

            return jsonify(response_data)

        @self.app.route("/api/workers/<worker_id>/unregister", methods=["POST"])
        def unregister_worker(worker_id):
            """Unregister a remote worker from the monitoring system.

            Removes the specified worker from the remote workers registry and
            broadcasts an update to all connected clients.

            Args:
                worker_id: Unique identifier of the worker to unregister

            Returns:
                dict: Response with success status
                int: HTTP status code (200 if successful, 404 if worker not found)

            Example:
                >>> response = requests.post("/api/workers/worker-1/unregister")
                >>> print(response.json())
                {'success': True}
            """
            with self._remote_workers_lock:
                if worker_id in self._remote_workers:
                    del self._remote_workers[worker_id]
                    self.logger.info(f"Unregistered remote worker: {worker_id}")

                    self.broadcast_worker_update(
                        worker_id, {"action": "unregistered", "status": "unregistered"}
                    )
                    
                    if self.procguard and self.procguard.group_manager:
                        group_id = self.procguard.group_manager.get_worker_group(worker_id)
                        if group_id:
                            self.procguard.group_manager.remove_worker_from_group(group_id, worker_id)
                            self.logger.info(f"[WebAPI] Removed worker {worker_id} from group {group_id} during unregistration")

                    return jsonify({"success": True})

            return jsonify({"success": False, "error": "Worker not found"}), 404

        @self.app.route("/api/workers/<worker_id>/logs", methods=["POST"])
        def worker_log(worker_id):
            """Receive and process log messages from a remote worker.

            Stores log messages from the worker and broadcasts them to
            all connected clients in real-time via Socket.IO.

            Args:
                worker_id: Unique identifier of the worker sending logs

            Returns:
                dict: Response with success status
                int: HTTP status code (200 if successful, 404 if worker not registered)

            Example:
                >>> response = requests.post(
                ...     "/api/workers/worker-1/logs",
                ...     json={"message": "Training step 100 completed", "level": "info"}
                ... )
                >>> print(response.json())
                {'success': True}
            """
            data = request.json
            message = data.get("message", "")
            level = data.get("level", "info")

            with self._remote_workers_lock:
                if worker_id not in self._remote_workers:
                    self.logger.debug(f"Log from unregistered worker: {worker_id}")
                    return (
                        jsonify({"success": False, "error": "Worker not registered"}),
                        404,
                    )

                worker = self._remote_workers[worker_id]
                timestamp = datetime.now().strftime("%H:%M:%S")
                log_entry = f"[{timestamp}] [{level.upper()}] {message}"

                worker.logs.append(log_entry)
                if len(worker.logs) > 1000:
                    worker.logs = worker.logs[-1000:]

            self.socketio.emit(
                "log_message",
                {"worker_id": worker_id, "message": message, "level": level},
            )

            return jsonify({"success": True})

        @self.app.route("/api/workers/<worker_id>/command", methods=["POST"])
        def send_worker_command(worker_id):
            """Send a control command to a remote worker.

            Queues a command (start, stop, restart, kill) for the specified
            worker to execute on its next heartbeat check.

            Args:
                worker_id: Unique identifier of the target worker

            Returns:
                dict: Response with success status
                int: HTTP status code (200 if successful, 400 if no command, 404 if worker not found)

            Example:
                >>> response = requests.post(
                ...     "/api/workers/worker-1/command",
                ...     json={"command": "restart"}
                ... )
                >>> print(response.json())
                {'success': True}
            """
            command = request.json.get("command")

            if not command:
                return jsonify({"success": False, "error": "command is required"}), 400

            with self._remote_workers_lock:
                if worker_id not in self._remote_workers:
                    return (
                        jsonify({"success": False, "error": "Worker not registered"}),
                        404,
                    )

                self._remote_workers[worker_id].pending_command = command
                self.logger.info(f"Queued command '{command}' for worker {worker_id}")

            return jsonify({"success": True})

        @self.app.route("/api/remote-workers")
        def get_remote_workers():
            """Get information about all registered remote workers.

            Returns a dictionary containing detailed information about each
            remote worker including status, PID, command, and registration time.

            Returns:
                dict: Dictionary mapping worker IDs to their information
                    - worker_id: Unique identifier of the worker
                    - command: Command being executed by the worker
                    - working_dir: Working directory for the worker
                    - status: Current status (running, stopped, etc.)
                    - pid: Process ID of the worker
                    - restart_count: Number of times the worker has been restarted
                    - last_heartbeat: ISO timestamp of last heartbeat
                    - registered_at: ISO timestamp when worker registered
                    - logs_count: Number of log entries stored

            Example:
                >>> response = requests.get("/api/remote-workers")
                >>> workers = response.json()
                >>> print(workers)
                {'worker-1': {'worker_id': 'worker-1', 'status': 'running', ...}}
            """
            with self._remote_workers_lock:
                result = {}
                for worker_id, worker in self._remote_workers.items():
                    result[worker_id] = {
                        "worker_id": worker.worker_id,
                        "command": worker.command,
                        "working_dir": worker.working_dir,
                        "status": worker.status,
                        "pid": worker.pid,
                        "restart_count": worker.restart_count,
                        "last_heartbeat": (
                            worker.last_heartbeat.isoformat()
                            if worker.last_heartbeat
                            else None
                        ),
                        "registered_at": worker.registered_at.isoformat(),
                        "logs_count": len(worker.logs),
                    }
            return jsonify(result)

        @self.app.route("/api/remote-workers/<worker_id>/logs", methods=["GET"])
        def get_remote_worker_logs(worker_id):
            """Get log messages for a specific remote worker.

            Retrieves the stored log entries for the specified worker,
            limited to the most recent 1000 entries.

            Args:
                worker_id: Unique identifier of the worker

            Returns:
                dict: Response containing logs list
                    - logs: List of log entries (most recent 1000)

            Example:
                >>> response = requests.get("/api/remote-workers/worker-1/logs")
                >>> print(response.json())
                {'logs': ['[14:30:15] [INFO] Training started', ...]}
            """
            with self._remote_workers_lock:
                if worker_id not in self._remote_workers:
                    return jsonify({"logs": []})

                logs = self._remote_workers[worker_id].logs
            return jsonify({"logs": logs})

        @self.app.route("/api/pytorch/config", methods=["GET"])
        def get_pytorch_config():
            """Get PyTorch distributed training configuration for a group.

            Returns the PyTorch configuration including master address,
            port, world size, and backend settings for a specific group.

            Args:
                group_id: Optional group ID to get config for

            Returns:
                dict: PyTorch configuration settings

            Example:
                >>> response = requests.get("/api/pytorch/config?group_id=group-1")
                >>> print(response.json())
                {'master_addr': 'gn34', 'master_port': 29500, 'world_size': 4, 'backend': 'nccl', ...}
            """
            group_id = request.args.get("group_id")

            if group_id:
                if not self.procguard.group_manager:
                    return jsonify({"error": "Group manager not initialized"}), 503

                config = self.procguard.group_manager.get_group_config(group_id)
                if config:
                    return jsonify(
                        {
                            "master_addr": config.master_addr,
                            "master_port": config.master_port,
                            "world_size": config.world_size,
                            "backend": config.backend,
                            "cuda_visible_devices": config.cuda_visible_devices,
                            "nccl_socket_ifname": config.nccl_socket_ifname,
                        }
                    )
                return jsonify({"error": "Group not found"}), 404

            config = load_pytorch_config()
            return jsonify(config)

        @self.app.route("/api/pytorch/config", methods=["POST"])
        def save_pytorch_config_route():
            """Save PyTorch distributed training configuration.

            Receives and persists PyTorch configuration settings to disk.
            The configuration is used by all workers in distributed training.

            Returns:
                dict: Response with success status

            Example:
                >>> response = requests.post(
                ...     "/api/pytorch/config",
                ...     json={"master_addr": "10.0.0.1", "world_size": 4}
                ... )
                >>> print(response.json())
                {'success': True}
            """
            data = request.json

            if not isinstance(data, dict):
                return jsonify({"success": False, "error": "Invalid data format"}), 400

            if save_pytorch_config(data):
                return jsonify({"success": True})
            else:
                return (
                    jsonify({"success": False, "error": "Failed to save config"}),
                    500,
                )

        @self.app.route("/api/workers/cleanup", methods=["POST"])
        def cleanup_workers():
            """Clean up stale or stopped remote workers.

            Removes workers from the registry based on specified criteria.
            Can remove all stopped workers or force remove all workers.

            Args:
                cleanup_stopped: If True, remove workers with 'stopped' status
                force_remove_all: If True, remove all workers regardless of status

            Returns:
                dict: Response with success status and count of removed workers
                    - success: Boolean indicating success
                    - removed_count: Number of workers removed

            Example:
                >>> response = requests.post(
                ...     "/api/workers/cleanup",
                ...     json={"cleanup_stopped": True}
                ... )
                >>> print(response.json())
                {'success': True, 'removed_count': 3}
            """
            data = request.json or {}
            cleanup_stopped = data.get("cleanup_stopped", True)
            force_remove_all = data.get("force_remove_all", False)

            removed_count = 0
            with self._remote_workers_lock:
                workers_to_remove = []

                for worker_id, worker in self._remote_workers.items():
                    if force_remove_all:
                        workers_to_remove.append(worker_id)
                    elif cleanup_stopped and worker.status == "stopped":
                        workers_to_remove.append(worker_id)

                for worker_id in workers_to_remove:
                    worker = self._remote_workers.pop(worker_id, None)
                    if worker:
                        removed_count += 1
                        self.logger.info(f"[WebAPI] Cleaned up worker: {worker_id}")

                self.logger.info(f"[WebAPI] Cleaned up {removed_count} workers")

            return jsonify({"success": True, "removed_count": removed_count})

        @self.app.route("/api/workers/reset", methods=["POST"])
        def reset_workers():
            """Reset all workers and clear cache.

            Removes all workers from the registry and clears all
            worker-related data. Useful for handling multiple
            worker_launcher restarts.

            Returns:
                dict: Response with success status and count of removed workers
                    - success: Boolean indicating success
                    - removed_count: Number of workers removed
                    - message: Status message

            Example:
                >>> response = requests.post("/api/workers/reset")
                >>> print(response.json())
                {'success': True, 'removed_count': 5, 'message': 'All workers reset'}
            """
            removed_count = 0
            
            with self._remote_workers_lock:
                removed_count = len(self._remote_workers)
                self._remote_workers.clear()
                self.logger.info(f"[WebAPI] Reset all workers: {removed_count} workers removed")
            
            with self._worker_ranks_lock:
                self._worker_ranks.clear()
                self.logger.info(f"[WebAPI] Cleared all worker ranks")
            
            with self._worker_states_lock:
                self._worker_states.clear()
                self.logger.info(f"[WebAPI] Cleared all worker states")
            
            if self.procguard and self.procguard.group_manager:
                all_groups = self.procguard.group_manager.get_all_groups()
                for group_id, group_info in all_groups.items():
                    if group_info.workers:
                        for worker_id in list(group_info.workers):
                            self.procguard.group_manager.remove_worker_from_group(group_id, worker_id)
                self.logger.info(f"[WebAPI] Cleared all workers from groups")

            self.broadcast_alert("info", "All workers have been reset")
            
            return jsonify({
                "success": True, 
                "removed_count": removed_count,
                "message": f"All workers reset ({removed_count} workers removed)"
            })

        @self.app.route("/api/groups", methods=["GET"])
        def get_groups():
            """Get all worker groups and their configuration.

            Returns a list of all worker groups with their members and
            distributed training configuration.

            Returns:
                dict: Response containing groups and summary
                    - success: Boolean indicating success
                    - groups: List of group information dictionaries
                    - summary: Summary statistics for all groups
                int: HTTP status code (503 if group manager not initialized)

            Example:
                >>> response = requests.get("/api/groups")
                >>> print(response.json())
                {'success': True, 'groups': [{'group_id': 'group-1', 'name': 'Training', ...}], ...}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            groups = self.procguard.group_manager.get_all_groups_for_dist_monitor()
            self.logger.debug(f"[WebAPI] 获取分组信息，返回 {len(groups)} 个分组")
            return jsonify(
                {
                    "success": True,
                    "groups": groups,
                    "summary": self.procguard.group_manager.get_state_summary(),
                    "server_info": {
                        "startup_time": self._startup_time,
                        "version": self._server_version
                    }
                }
            )

        @self.app.route("/api/groups/<group_id>", methods=["GET"])
        def get_group(group_id):
            """Get information for a specific worker group.

            Args:
                group_id: Unique identifier of the group

            Returns:
                dict: Response containing group information
                    - success: Boolean indicating success
                    - group: Group information dictionary
                int: HTTP status code (404 if group not found, 503 if group manager not initialized)

            Example:
                >>> response = requests.get("/api/groups/group-1")
                >>> print(response.json())
                {'success': True, 'group': {'group_id': 'group-1', 'name': 'Training', ...}}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            group_info = self.procguard.group_manager.get_group_info_for_dist_monitor(
                group_id
            )
            if group_info:
                return jsonify({"success": True, "group": group_info})
            return jsonify({"success": False, "error": "Group not found"}), 404

        @self.app.route("/api/groups", methods=["POST"])
        def create_group():
            """Create a new worker group.

            Creates a new worker group with the specified ID and name.
            The group can then have workers added to it for distributed training.

            Args:
                group_id: Unique identifier for the new group
                name: Human-readable name for the group

            Returns:
                dict: Response with success status and group ID
                    - success: Boolean indicating success
                    - group_id: ID of the created group
                int: HTTP status code (400 if missing required fields, 409 if group exists)

            Example:
                >>> response = requests.post(
                ...     "/api/groups",
                ...     json={"group_id": "group-1", "name": "Training Group"}
                ... )
                >>> print(response.json())
                {'success': True, 'group_id': 'group-1'}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            data = request.json
            group_id = data.get("group_id")
            name = data.get("name")

            if not group_id or not name:
                return (
                    jsonify(
                        {"success": False, "error": "group_id and name are required"}
                    ),
                    400,
                )

            self.logger.info(f"[WebAPI] 创建分组请求: group_id={group_id}, name={name}")
            success = self.procguard.group_manager.create_group(group_id, name)
            if success:
                self.logger.info(f"[WebAPI] ✓ 分组创建成功: {group_id}")
                return jsonify({"success": True, "group_id": group_id})
            self.logger.warning(f"[WebAPI] ✗ 分组创建失败: {group_id} 已存在")
            return jsonify({"success": False, "error": "Group already exists"}), 409

        @self.app.route("/api/groups/<group_id>", methods=["PUT"])
        def update_group(group_id):
            """Update the name of an existing worker group.

            Args:
                group_id: Unique identifier of the group to update

            Returns:
                dict: Response with success status
                int: HTTP status code (400 if no update data, 404 if group not found)

            Example:
                >>> response = requests.put(
                ...     "/api/groups/group-1",
                ...     json={"name": "Updated Training Group"}
                ... )
                >>> print(response.json())
                {'success': True}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            data = request.json
            name = data.get("name")

            if name:
                success = self.procguard.group_manager.rename_group(group_id, name)
                if success:
                    return jsonify({"success": True})
                return jsonify({"success": False, "error": "Group not found"}), 404

            return jsonify({"success": False, "error": "No update data provided"}), 400

        @self.app.route("/api/groups/<group_id>", methods=["DELETE"])
        def delete_group(group_id):
            """Delete an existing worker group.

            Removes the specified group and all worker associations.
            Workers in the group become ungrouped but continue running.

            Args:
                group_id: Unique identifier of the group to delete

            Returns:
                dict: Response with success status
                int: HTTP status code (404 if group not found)

            Example:
                >>> response = requests.delete("/api/groups/group-1")
                >>> print(response.json())
                {'success': True}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            self.logger.info(f"[WebAPI] 删除分组请求: group_id={group_id}")
            success = self.procguard.group_manager.delete_group(group_id)
            if success:
                self.logger.info(f"[WebAPI] ✓ 分组删除成功: {group_id}")
                return jsonify({"success": True})
            self.logger.warning(f"[WebAPI] ✗ 分组删除失败: {group_id} 不存在")
            return jsonify({"success": False, "error": "Group not found"}), 404

        @self.app.route("/api/groups/<group_id>/workers/<worker_id>", methods=["POST"])
        def add_worker_to_group(group_id, worker_id):
            """Add a worker to a group.

            Registers an existing worker with the specified group.
            If the worker is already in another group, it will be moved.

            Args:
                group_id: Unique identifier of the target group
                worker_id: Unique identifier of the worker to add

            Returns:
                dict: Response with success status
                int: HTTP status code (404 if group not found or worker already in group)

            Example:
                >>> response = requests.post("/api/groups/group-1/workers/worker-1")
                >>> print(response.json())
                {'success': True}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            self.logger.info(
                f"[WebAPI] 添加Worker请求: worker={worker_id} → group={group_id}"
            )
            success = self.procguard.group_manager.add_worker_to_group(
                group_id, worker_id
            )
            if success:
                self.logger.info(f"[WebAPI] ✓ Worker添加成功: {worker_id} → {group_id}")
                return jsonify({"success": True})
            self.logger.warning(f"[WebAPI] ✗ Worker添加失败: {worker_id} → {group_id}")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Group not found or worker already in group",
                    }
                ),
                404,
            )

        @self.app.route(
            "/api/groups/<group_id>/workers/<worker_id>", methods=["DELETE"]
        )
        def remove_worker_from_group(group_id, worker_id):
            """Remove a worker from a group.

            Removes the specified worker from the group but keeps
            the worker running independently.

            Args:
                group_id: Unique identifier of the group
                worker_id: Unique identifier of the worker to remove

            Returns:
                dict: Response with success status
                int: HTTP status code (404 if group not found or worker not in group)

            Example:
                >>> response = requests.delete("/api/groups/group-1/workers/worker-1")
                >>> print(response.json())
                {'success': True}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            self.logger.info(
                f"[WebAPI] 移除Worker请求: worker={worker_id} ← group={group_id}"
            )
            success = self.procguard.group_manager.remove_worker_from_group(
                group_id, worker_id
            )
            if success:
                self.logger.info(f"[WebAPI] ✓ Worker移除成功: {worker_id} ← {group_id}")
                return jsonify({"success": True})
            self.logger.warning(f"[WebAPI] ✗ Worker移除失败: {worker_id} ← {group_id}")
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Group not found or worker not in group",
                    }
                ),
                404,
            )

        @self.app.route("/api/groups/move-worker", methods=["POST"])
        def move_worker():
            """Move a worker from its current group to a target group.

            Relocates a worker from one group to another, updating
            all internal state and group associations.

            Args:
                worker_id: Unique identifier of the worker to move
                target_group_id: Unique identifier of the target group

            Returns:
                dict: Response with success status
                int: HTTP status code (400 if missing required fields, 404 if target group not found)

            Example:
                >>> response = requests.post(
                ...     "/api/groups/move-worker",
                ...     json={"worker_id": "worker-1", "target_group_id": "group-2"}
                ... )
                >>> print(response.json())
                {'success': True}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            data = request.json
            worker_id = data.get("worker_id")
            target_group_id = data.get("target_group_id")

            if not worker_id or not target_group_id:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "worker_id and target_group_id are required",
                        }
                    ),
                    400,
                )

            current_group = self.procguard.group_manager.get_worker_group(worker_id)
            self.logger.info(
                f"[WebAPI] 移动Worker请求: worker={worker_id}, {current_group or '无分组'} → {target_group_id}"
            )
            success = self.procguard.group_manager.move_worker(
                worker_id, target_group_id
            )
            if success:
                self.logger.info(
                    f"[WebAPI] ✓ Worker移动成功: {worker_id}, {current_group or '无分组'} → {target_group_id}"
                )
                return jsonify({"success": True})
            self.logger.warning(
                f"[WebAPI] ✗ Worker移动失败: {worker_id} → {target_group_id}"
            )
            return jsonify({"success": False, "error": "Target group not found"}), 404

        @self.app.route("/api/groups/sync", methods=["POST"])
        def sync_groups():
            """Synchronize group state from frontend.

            Receives complete group configuration from the frontend
            and replaces all local group state with the received data.

            Args:
                groups: Dictionary containing group configurations

            Returns:
                dict: Response with success status
                int: HTTP status code (400 if no data provided, 500 if sync failed)

            Example:
                >>> response = requests.post(
                ...     "/api/groups/sync",
                ...     json={"groups": {"group-1": {"name": "Training", "workers": ["w1", "w2"]}}}
                ... )
                >>> print(response.json())
                {'success': True}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            data = request.json
            if not data:
                return jsonify({"success": False, "error": "No data provided"}), 400

            group_count = len(data.get("groups", {}))
            self.logger.info(f"[WebAPI] 同步分组请求: {group_count} 个分组")
            success = self.procguard.group_manager.sync_from_frontend(data)
            if success:
                self.logger.info(f"[WebAPI] ✓ 分组同步成功: {group_count} 个分组")
                return jsonify({"success": True})
            self.logger.warning(f"[WebAPI] ✗ 分组同步失败")
            return jsonify({"success": False, "error": "Sync failed"}), 500

        @self.app.route("/api/groups/cache", methods=["DELETE"])
        def clear_group_cache():
            """Clear the local group cache.

            Removes all locally cached group data. The next access
            will fetch fresh data from the frontend.

            Returns:
                dict: Response with success status and message
                int: HTTP status code (500 if clear failed)

            Example:
                >>> response = requests.delete("/api/groups/cache")
                >>> print(response.json())
                {'success': True, 'message': 'Cache cleared successfully'}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            self.logger.info("[WebAPI] 清除分组缓存请求")
            success = self.procguard.group_manager.clear_cache()
            if success:
                self.logger.info("[WebAPI] ✓ 分组缓存已清除")
                self.broadcast_group_cache_cleared()
                return jsonify(
                    {"success": True, "message": "Cache cleared successfully"}
                )
            self.logger.error("[WebAPI] ✗ 清除分组缓存失败")
            return jsonify({"success": False, "error": "Failed to clear cache"}), 500

        @self.app.route("/api/groups/validate", methods=["GET"])
        def validate_groups():
            """验证分组数据一致性"""
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            all_workers = self._get_all_workers()
            issues = []

            for group_id, group in self.procguard.group_manager.get_all_groups().items():
                invalid_workers = []
                for worker_id in group.workers:
                    if worker_id not in all_workers:
                        invalid_workers.append(worker_id)

                if invalid_workers:
                    issues.append({
                        'group_id': group_id,
                        'group_name': group.name,
                        'invalid_workers': invalid_workers,
                        'total_workers': len(group.workers),
                        'valid_workers': len(group.workers) - len(invalid_workers)
                    })

            return jsonify({
                "success": True,
                "consistent": len(issues) == 0,
                "issues": issues,
                "total_groups": len(self.procguard.group_manager.get_all_groups())
            })

        @self.app.route("/api/groups/<group_id>/config", methods=["GET"])
        def get_group_config(group_id):
            """Get the PyTorch distributed training configuration for a group.

            Args:
                group_id: Unique identifier of the group

            Returns:
                dict: Response containing group configuration
                    - success: Boolean indicating success
                    - config: GroupConfig as dictionary
                int: HTTP status code (404 if group not found)

            Example:
                >>> response = requests.get("/api/groups/group-1/config")
                >>> print(response.json())
                {'success': True, 'config': {'master_addr': '10.0.0.1', 'world_size': 4, ...}}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            config = self.procguard.group_manager.get_group_config(group_id)
            if config:
                self.logger.debug(f"[WebAPI] 获取分组 {group_id} 配置成功")
                return jsonify({"success": True, "config": asdict(config)})
            self.logger.warning(f"[WebAPI] 分组 {group_id} 不存在，无法获取配置")
            return jsonify({"success": False, "error": "Group not found"}), 404

        @self.app.route("/api/groups/<group_id>/config", methods=["PUT", "POST"])
        def update_group_config(group_id):
            """Update the PyTorch distributed training configuration for a group.

            Modifies the distributed training settings including master address,
            port, world size, backend, and NCCL configuration.
            If the group doesn't exist, it will be created automatically.

            Args:
                group_id: Unique identifier of the group
                master_addr: Address of the master node
                master_port: Port for master node communication
                world_size: Total number of processes
                backend: Distributed training backend (nccl, gloo)
                cuda_visible_devices: GPU devices visible to workers
                nccl_socket_ifname: Network interface for NCCL

            Returns:
                dict: Response with success status
                int: HTTP status code (400 if no data, 500 if update failed)

            Example:
                >>> response = requests.put(
                ...     "/api/groups/group-1/config",
                ...     json={"master_addr": "10.0.0.1", "world_size": 8}
                ... )
                >>> print(response.json())
                {'success': True}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            data = request.json or {}

            from ..core.group_manager import GroupConfig, GroupInfo

            current_config = self.procguard.group_manager.get_group_config(group_id)
            if not current_config:
                self.logger.info(
                    f"[WebAPI] Group {group_id} not found, auto-creating..."
                )
                self.procguard.group_manager.create_group(group_id, group_id)
                current_config = self.procguard.group_manager.get_group_config(group_id)

            self.logger.info(f"[WebAPI] 更新分组 {group_id} 配置: {data}")

            from ..core.group_manager import GroupConfig

            new_config = GroupConfig(
                master_addr=data.get("master_addr", current_config.master_addr),
                master_port=data.get("master_port", current_config.master_port),
                world_size=data.get("world_size", current_config.world_size),
                backend=data.get("backend", current_config.backend),
                cuda_visible_devices=data.get(
                    "cuda_visible_devices", current_config.cuda_visible_devices
                ),
                nccl_socket_ifname=data.get(
                    "nccl_socket_ifname", current_config.nccl_socket_ifname
                ),
            )

            success = self.procguard.group_manager.update_group_config(
                group_id, new_config
            )
            if success:
                self.logger.info(f"[WebAPI] ✓ 分组 {group_id} 配置更新成功")
                return jsonify({"success": True})
            self.logger.error(f"[WebAPI] ✗ 分组 {group_id} 配置更新失败")
            return jsonify({"success": False, "error": "Update failed"}), 500

        @self.app.route("/api/groups/worker/<worker_id>", methods=["GET"])
        def get_worker_group(worker_id):
            """Get the group information for a specific worker.

            Args:
                worker_id: Unique identifier of the worker

            Returns:
                dict: Response containing worker and group information
                    - success: Boolean indicating success
                    - worker_id: ID of the queried worker
                    - group: Group information if worker is in a group, None otherwise

            Example:
                >>> response = requests.get("/api/groups/worker/worker-1")
                >>> print(response.json())
                {'success': True, 'worker_id': 'worker-1', 'group': {'group_id': 'group-1', ...}}
            """
            if not self.procguard.group_manager:
                return (
                    jsonify(
                        {"success": False, "error": "Group manager not initialized"}
                    ),
                    503,
                )

            group = self.procguard.group_manager.get_group_of_worker(worker_id)
            if group:
                return jsonify(
                    {
                        "success": True,
                        "worker_id": worker_id,
                        "group": self.procguard.group_manager.get_group_info_for_dist_monitor(
                            group.group_id
                        ),
                    }
                )
            return jsonify({"success": True, "worker_id": worker_id, "group": None})

        @self.app.route("/api/groups/<group_id>/state", methods=["GET"])
        def get_group_state(group_id):
            """Get the current group state for failover coordination.

            Workers call this to get the current failover state.
            The state includes should_stop, is_recovering, failed_ranks, etc.
            Once all workers in the group have fetched the state, it will be reset.

            Query params:
                worker_id: ID of the worker fetching this state

            Returns:
                dict: Response containing group state
                    - success: Boolean indicating success
                    - group_id: The group ID
                    - should_stop: Whether workers should stop training
                    - is_recovering: Whether the group is in recovery mode
                    - failed_ranks: List of failed ranks
                    - recover_src_rank: Source rank for recovery
                    - failover_workers: List of failover workers
                    - failover_info: Details about the failover
                    - reset: Whether the state was just reset

            Example:
                >>> response = requests.get("/api/groups/group-1/state?worker_id=worker-0")
                >>> print(response.json())
                {'success': True, 'group_id': 'group-1', 'should_stop': True, 'failed_ranks': [1]}
            """
            worker_id = request.args.get("worker_id", "unknown")

            if not hasattr(self, '_group_state_lock') or not self._group_state:
                return jsonify({
                    "success": True,
                    "group_id": group_id,
                    "should_stop": False,
                    "is_recovering": False,
                    "failed_ranks": [],
                    "recover_src_rank": None,
                    "failover_workers": [],
                    "failover_info": {},
                    "reset": False
                })

            with self._group_state_lock:
                if group_id not in self._group_failover_state:
                    return jsonify({
                        "success": True,
                        "group_id": group_id,
                        "should_stop": False,
                        "is_recovering": False,
                        "failed_ranks": [],
                        "recover_src_rank": None,
                        "failover_workers": [],
                        "failover_info": {},
                        "reset": False
                    })

                current_state = self._group_failover_state[group_id]
                current_state["acknowledged_workers"].add(worker_id)

                acknowledged_count = len(current_state["acknowledged_workers"])

                failover_workers = []
                failover_info = {}

                with self._remote_workers_lock:
                    for wid, worker in self._remote_workers.items():
                        if wid in current_state["acknowledged_workers"]:
                            continue
                        if worker.is_failover_worker:
                            failover_workers.append(wid)
                            failover_info[wid] = {
                                "status": worker.status,
                                "pid": worker.pid,
                                "info": worker.failover_info
                            }

                self._group_failover_state[group_id]["failover_workers"] = failover_workers
                self._group_failover_state[group_id]["failover_info"] = failover_info

                world_size = len(current_state.get("acknowledged_workers", [])) + len(failover_workers)
                all_fetched = acknowledged_count >= world_size

                reset = False
                if all_fetched:
                    self._group_failover_state[group_id] = {
                        "should_stop": False,
                        "is_recovering": False,
                        "failed_ranks": [],
                        "recover_src_rank": None,
                        "failover_workers": [],
                        "failover_info": {},
                        "acknowledged_workers": set(),
                        "timestamp": None
                    }
                    reset = True

                return jsonify({
                    "success": True,
                    "group_id": group_id,
                    "should_stop": current_state["should_stop"],
                    "is_recovering": current_state["is_recovering"],
                    "failed_ranks": current_state["failed_ranks"],
                    "recover_src_rank": current_state["recover_src_rank"],
                    "failover_workers": current_state["failover_workers"],
                    "failover_info": current_state["failover_info"],
                    "reset": reset
                })

        @self.app.route("/api/groups/failed-ranks/<group_id>", methods=["GET"])
        def get_group_failed_ranks(group_id):
            """Get cached failed ranks for a specific group.

            Returns failed ranks that were cached when workers in the group
            transitioned from running to failed/stopped/exited state.
            Cache is cleared after all workers in the group have fetched it
            (determined by world_size).

            Args:
                group_id: Unique identifier of the group

            Returns:
                dict: Response containing failed ranks for the group
                    - success: Boolean indicating success
                    - group_id: The group ID
                    - failed_ranks: List of failed ranks (sorted)
                    - cached: Whether the data is from cache
                    - remaining: Number of remaining fetches before cache clears

            Example:
                >>> response = requests.get("/api/groups/failed-ranks/group-1")
                >>> print(response.json())
                {'success': True, 'group_id': 'group-1', 'failed_ranks': [1, 2], 'cached': True, 'remaining': 2}
            """
            current_time = time.time()
            world_size = 1

            with self._worker_ranks_lock:
                failed_ranks = self._failed_ranks_cache.get(group_id, [])
                cache_time = self._failed_ranks_cache_time.get(group_id, 0)
                is_cached = current_time - cache_time < self._failed_ranks_cache_ttl

                if not is_cached:
                    failed_ranks = []
                    return jsonify({
                        "success": True,
                        "group_id": group_id,
                        "failed_ranks": [],
                        "cached": False,
                        "remaining": 0
                    })

                if self.procguard and self.procguard.group_manager:
                    try:
                        group_config = self.procguard.group_manager.get_group_config(group_id)
                        if group_config:
                            world_size = group_config.world_size or 1
                    except Exception:
                        pass

                self._failed_ranks_fetch_count[group_id] = self._failed_ranks_fetch_count.get(group_id, 0) + 1
                fetch_count = self._failed_ranks_fetch_count.get(group_id, 0)
                remaining = max(0, world_size - fetch_count)

                self.logger.debug(f"[WebMonitor] Group '{group_id}' failed ranks fetched: {fetch_count}/{world_size}, remaining: {remaining}")

                if fetch_count >= world_size:
                    self._failed_ranks_cache[group_id] = []
                    self._failed_ranks_cache_time[group_id] = current_time
                    self._failed_ranks_fetch_count[group_id] = 0
                    self.logger.info(f"[WebMonitor] Group '{group_id}' failed ranks cache cleared after {fetch_count} fetches")

                return jsonify({
                    "success": True,
                    "group_id": group_id,
                    "failed_ranks": failed_ranks,
                    "cached": True,
                    "remaining": remaining
                })

        @self.app.route("/api/ha/associations", methods=["GET"])
        def list_ha_associations():
            """List all HA associations.

            Returns:
                dict: List of all HA associations with their configurations

            Example:
                >>> response = requests.get("/api/ha/associations")
                >>> print(response.json())
                {'success': True, 'associations': [...]}
            """
            with self._ha_associations_lock:
                associations = []
                for assoc_id, config in self._ha_associations.items():
                    associations.append(config.to_dict())
                return jsonify({"success": True, "associations": associations})

        @self.app.route("/api/ha/associations", methods=["POST"])
        def create_ha_association():
            """Create a new HA association.

            Request body:
                active_group_id: ID of the active group
                standby_group_id: ID of the standby group
                world_size: Target number of workers in active group
                master_addr: Master address for PyTorch config
                master_port: Master port for PyTorch config
                backend: Distributed training backend
                failover_threshold: Number of failed workers to trigger failover
                auto_failover: Whether to automatically failover

            Returns:
                dict: Created association information

            Example:
                >>> response = requests.post("/api/ha/associations", json={
                ...     "active_group_id": "group-active",
                ...     "standby_group_id": "group-standby",
                ...     "world_size": 4
                ... })
                >>> print(response.json())
                {'success': True, 'association_id': 'assoc_group-active_group-standby'}
            """
            data = request.json

            active_group_id = data.get("active_group_id")
            standby_group_id = data.get("standby_group_id")

            if not active_group_id or not standby_group_id:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "active_group_id and standby_group_id required",
                        }
                    ),
                    400,
                )

            association_id = f"assoc_{active_group_id}_{standby_group_id}"

            with self._ha_associations_lock:
                if association_id in self._ha_associations:
                    return (
                        jsonify(
                            {"success": False, "error": "Association already exists"}
                        ),
                        400,
                    )

                self._ha_associations[association_id] = AssociationConfig(
                    association_id=association_id,
                    active_group_id=active_group_id,
                    standby_group_id=standby_group_id,
                    world_size=data.get("world_size", 4),
                    master_addr=data.get("master_addr"),
                    master_port=data.get("master_port", 29500),
                    backend=data.get("backend", "nccl"),
                    failover_threshold=data.get("failover_threshold", 1),
                    auto_failover=data.get("auto_failover", True),
                )

            self.logger.info(f"Created HA association: {association_id}")

            if data.get("auto_start_monitor", True):
                self.logger.info(f"Auto-starting monitoring for {association_id}")
                thread = threading.Thread(
                    target=self._ha_monitor_loop,
                    args=(association_id,),
                    daemon=True,
                    name=f"HAMonitor-{association_id}",
                )
                thread.start()
                self._ha_monitor_threads[association_id] = thread
                self._ha_monitoring_active[association_id] = True

            return jsonify({"success": True, "association_id": association_id})

        @self.app.route("/api/ha/associations/<association_id>", methods=["DELETE"])
        def delete_ha_association(association_id):
            """Delete an HA association.

            Args:
                association_id: ID of the association to delete

            Returns:
                dict: Deletion result

            Example:
                >>> response = requests.delete("/api/ha/associations/assoc_g1_g2")
                >>> print(response.json())
                {'success': True}
            """
            with self._ha_associations_lock:
                if association_id not in self._ha_associations:
                    return (
                        jsonify({"success": False, "error": "Association not found"}),
                        404,
                    )

                del self._ha_associations[association_id]
                self._ha_status_cache.pop(association_id, None)

            self.logger.info(f"Deleted HA association: {association_id}")
            return jsonify({"success": True})

        @self.app.route("/api/ha/associations/<association_id>/status", methods=["GET"])
        def get_ha_association_status(association_id):
            """Get status of an HA association.

            Args:
                association_id: ID of the association

            Returns:
                dict: Association status including worker counts and health

            Example:
                >>> response = requests.get("/api/ha/associations/assoc_g1_g2/status")
                >>> print(response.json())
                {'success': True, 'status': {...}}
            """
            with self._ha_associations_lock:
                if association_id not in self._ha_associations:
                    return (
                        jsonify({"success": False, "error": "Association not found"}),
                        404,
                    )

                config = self._ha_associations[association_id]

            status = self._check_ha_association_health(config)
            if status is None:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "Association removed (no workers in active or standby group)",
                        }
                    ),
                    404,
                )
            return jsonify({"success": True, "status": status.to_dict()})

        @self.app.route(
            "/api/ha/associations/<association_id>/failover", methods=["POST"]
        )
        def trigger_ha_failover(association_id):
            """Trigger manual failover for an HA association.

            Args:
                association_id: ID of the association

            Returns:
                dict: Failover result
            """
            try:
                with self._ha_associations_lock:
                    if association_id not in self._ha_associations:
                        return (
                            jsonify(
                                {"success": False, "error": "Association not found"}
                            ),
                            404,
                        )

                    config = self._ha_associations[association_id]

                moved_count = self._execute_ha_failover_safe(config)

                return jsonify(
                    {
                        "success": True,
                        "association_id": association_id,
                        "moved_workers": moved_count,
                    }
                )
            except Exception as e:
                self.logger.error(f"[HA] Trigger failover error: {e}")
                import traceback

                self.logger.error(f"[HA] Traceback: {traceback.format_exc()}")
                return jsonify({"success": False, "error": str(e)}), 500

        @self.app.route(
            "/api/ha/associations/<association_id>/monitor/start", methods=["POST"]
        )
        def start_ha_monitoring(association_id):
            """Start monitoring an HA association.

            Args:
                association_id: ID of the association

            Returns:
                dict: Result of starting monitoring
            """
            with self._ha_associations_lock:
                if association_id not in self._ha_associations:
                    return (
                        jsonify({"success": False, "error": "Association not found"}),
                        404,
                    )

                if association_id not in self._ha_monitor_threads:
                    thread = threading.Thread(
                        target=self._ha_monitor_loop,
                        args=(association_id,),
                        daemon=True,
                        name=f"HAMonitor-{association_id}",
                    )
                    thread.start()
                    self._ha_monitor_threads[association_id] = thread
                    self._ha_monitoring_active[association_id] = True

            return jsonify(
                {"success": True, "message": f"Monitoring started for {association_id}"}
            )

        @self.app.route(
            "/api/ha/associations/<association_id>/monitor/stop", methods=["POST"]
        )
        def stop_ha_monitoring(association_id):
            """Stop monitoring an HA association.

            Args:
                association_id: ID of the association

            Returns:
                dict: Result of stopping monitoring
            """
            self._ha_monitoring_active[association_id] = False

            return jsonify(
                {"success": True, "message": f"Monitoring stopped for {association_id}"}
            )

        @self.app.route("/api/pytorch/env", methods=["GET"])
        def get_pytorch_env():
            """Get PyTorch distributed training environment variables for a worker.

            Returns the environment variables based on the worker's group configuration.

            Args:
                worker_id: Worker ID to get group-specific config (required)

            Returns:
                dict: Environment variables for PyTorch distributed training

            Example:
                >>> response = requests.get("/api/pytorch/env?worker_id=worker-1")
                >>> print(response.json())
                {'MASTER_ADDR': 'gn34', 'MASTER_PORT': '29500', 'WORLD_SIZE': '4', ...}
            """
            worker_id = request.args.get("worker_id")

            if not worker_id:
                return jsonify({"error": "worker_id is required"}), 400

            if not self.procguard.group_manager:
                return jsonify({"error": "Group manager not initialized"}), 503

            group = self.procguard.group_manager.get_group_of_worker(worker_id)
            if not group or not group.config:
                return jsonify({"error": "Group or group config not found"}), 404

            rank = self._worker_ranks.get(worker_id)
            local_rank = 0
            if "-" in worker_id:
                try:
                    local_rank = int(worker_id.rsplit("-", 1)[-1])
                except (ValueError, IndexError):
                    local_rank = 0

            node_rank = 0
            all_groups = self.procguard.group_manager.get_all_groups()
            for gid, g in all_groups.items():
                if worker_id in g.workers:
                    sorted_nodes = sorted(
                        set(w.rsplit("-", 1)[0] for w in g.workers if "-" in w)
                    )
                    if sorted_nodes:
                        current_node = (
                            worker_id.rsplit("-", 1)[0]
                            if "-" in worker_id
                            else worker_id
                        )
                        node_rank = sorted_nodes.index(current_node)
                    break

            env_vars = get_pytorch_env_vars(
                local_rank=local_rank,
                node_rank=node_rank,
                rank=rank,
                group_config=group.config,
            )
            self.logger.debug(
                f"[WebAPI] Using group config for worker {worker_id}: {env_vars.get('MASTER_ADDR')}, world_size={env_vars.get('WORLD_SIZE')}"
            )
            return jsonify(env_vars)

    def _setup_socketio_events(self):
        @self.socketio.on("connect")
        def handle_connect():
            """Handle new Socket.IO client connection.

            Logs the connection and sends a confirmation message to the client.
            """
            try:
                self.logger.info("Client connected")
                emit("connected", {"message": "Connected to ProcGuard monitor"})
            except Exception as e:
                self.logger.error(f"[SocketIO] Error in connect handler: {e}")

        @self.socketio.on("disconnect")
        def handle_disconnect():
            """Handle Socket.IO client disconnection.

            Logs the disconnection event for monitoring purposes.
            """
            try:
                self.logger.info("Client disconnected")
            except Exception as e:
                self.logger.error(f"[SocketIO] Error in disconnect handler: {e}")

        @self.socketio.on("subscribe")
        def handle_subscribe(data):
            """Handle client subscription to real-time updates.

            Args:
                data: Subscription request data (may contain filter criteria)
            """
            try:
                self.logger.info(f"Client subscribed to updates: {data}")
                emit("subscribed", {"message": "Subscribed to updates"})
            except Exception as e:
                self.logger.error(f"[SocketIO] Error in subscribe handler: {e}")

    def broadcast_status_update(self, data: Dict[str, Any]):
        self.socketio.emit("status_update", data)

    def broadcast_worker_update(self, worker_id: str, data: Dict[str, Any]):
        self.socketio.emit("worker_update", {"worker_id": worker_id, **data})

    def broadcast_recovery_event(self, data: Dict[str, Any]):
        self.socketio.emit("recovery_event", data)

    def broadcast_alert(self, alert_type: str, message: str):
        self.socketio.emit("alert", {"type": alert_type, "message": message})

    def broadcast_group_cache_cleared(self):
        self.logger.info("[WebAPI] 广播分组缓存清除事件")
        self.socketio.emit(
            "group_cache_cleared", {"message": "Group cache has been cleared"}
        )

    def run(self, debug=False):
        self.logger.info(f"Starting web monitor on {self.host}:{self.port}")
        self.socketio.run(self.app, host=self.host, port=self.port, debug=debug)

    def shutdown(self):
        self.logger.info("Shutting down web monitor")
        self.socketio.stop()
