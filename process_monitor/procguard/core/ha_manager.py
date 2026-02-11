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
HA Manager - High Availability Group Association Manager for ProcGuard.

This module provides functionality for:
- Creating associations between active and standby groups
- Automatic failover when active group workers fail
- Worker migration between groups
- Automatic PyTorch config updates during failover

Features:
- Create/manage group associations
- Monitor worker health in active group
- Automatic failover to standby group
- Worker migration and config sync
- Diagnostic tools for association health

Example:
    >>> from procguard.core.ha_manager import HAGroupManager
    >>> manager = HAGroupManager("http://localhost:5000")
    >>> manager.create_association("active-group", "standby-group", world_size=4)
    >>> manager.start_monitoring()
"""

import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


class AssociationState(Enum):
    """State enumeration for group association."""
    ACTIVE = "active"
    STANDBY = "standby"
    FAILING_OVER = "failover"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class FailoverTrigger(Enum):
    """Trigger conditions for failover."""
    WORKER_STOPPED = "worker_stopped"
    WORKER_FAILED = "worker_failed"
    WORKER_EXITED = "worker_exited"
    WORKER_UNRESPONSIVE = "worker_unresponsive"
    THRESHOLD_EXCEEDED = "threshold_exceeded"


@dataclass
class AssociationConfig:
    """
    Configuration for group association (active/standby pair).

    Attributes:
        association_id: Unique identifier for this association
        active_group_id: ID of the active (primary) group
        standby_group_id: ID of the standby (backup) group
        world_size: Target number of workers in active group
        failover_threshold: Number of failed workers to trigger failover
        auto_failover: Whether to automatically failover
        heartbeat_interval: Seconds between health checks
        migration_delay: Seconds to wait before migrating workers
        master_addr: Master address for PyTorch config
        master_port: Master port for PyTorch config
        backend: Distributed training backend
        enabled: Whether this association is active
        created_at: Creation timestamp
        updated_at: Last update timestamp
    """
    association_id: str
    active_group_id: str
    standby_group_id: str
    world_size: int = 4
    failover_threshold: int = 1
    auto_failover: bool = True
    heartbeat_interval: float = 5.0
    migration_delay: float = 2.0
    master_addr: Optional[str] = None
    master_port: int = 29500
    backend: str = "nccl"
    enabled: bool = True
    created_at: str = ""
    updated_at: str = ""
    failover_count: int = 0
    last_failover: Optional[str] = None

    def __post_init__(self):
        from datetime import datetime
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "association_id": self.association_id,
            "active_group_id": self.active_group_id,
            "standby_group_id": self.standby_group_id,
            "world_size": self.world_size,
            "failover_threshold": self.failover_threshold,
            "auto_failover": self.auto_failover,
            "heartbeat_interval": self.heartbeat_interval,
            "migration_delay": self.migration_delay,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "backend": self.backend,
            "enabled": self.enabled,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "failover_count": self.failover_count,
            "last_failover": self.last_failover,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssociationConfig":
        """Create from dictionary."""
        return cls(
            association_id=data.get("association_id", ""),
            active_group_id=data.get("active_group_id", ""),
            standby_group_id=data.get("standby_group_id", ""),
            world_size=data.get("world_size", 4),
            failover_threshold=data.get("failover_threshold", 1),
            auto_failover=data.get("auto_failover", True),
            heartbeat_interval=data.get("heartbeat_interval", 5.0),
            migration_delay=data.get("migration_delay", 2.0),
            master_addr=data.get("master_addr"),
            master_port=data.get("master_port", 29500),
            backend=data.get("backend", "nccl"),
            enabled=data.get("enabled", True),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            failover_count=data.get("failover_count", 0),
            last_failover=data.get("last_failover"),
        )


@dataclass
class AssociationStatus:
    """
    Current status of a group association.

    Attributes:
        association_id: Unique identifier
        state: Current association state
        active_workers: Number of running workers in active group
        standby_available: Number of stopped workers in standby group
        last_failover: Timestamp of last failover
        failover_count: Total number of failovers
        last_health_check: Timestamp of last health check
        health_status: Overall health status
        events: List of recent events
    """
    association_id: str
    state: AssociationState
    active_workers: int = 0
    standby_available: int = 0
    last_failover: Optional[str] = None
    failover_count: int = 0
    last_health_check: Optional[str] = None
    health_status: str = "healthy"
    events: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "association_id": self.association_id,
            "state": self.state.value if isinstance(self.state, AssociationState) else self.state,
            "active_workers": self.active_workers,
            "standby_available": self.standby_available,
            "last_failover": self.last_failover,
            "failover_count": self.failover_count,
            "last_health_check": self.last_health_check,
            "health_status": self.health_status,
            "events": self.events,
        }


class HAClientBase(ABC):
    """Abstract base class for HA client implementations."""

    @abstractmethod
    def request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Send HTTP request to the API."""
        pass

    @abstractmethod
    def get_workers(self) -> Dict[str, Dict[str, Any]]:
        """Get all workers."""
        pass

    @abstractmethod
    def get_worker_group(self, worker_id: str) -> Optional[str]:
        """Get the group ID that a worker belongs to."""
        pass

    @abstractmethod
    def get_group(self, group_id: str) -> Optional[Dict[str, Any]]:
        """Get group information."""
        pass

    @abstractmethod
    def move_worker_to_group(self, worker_id: str, target_group_id: str) -> bool:
        """Move a worker to a target group."""
        pass

    @abstractmethod
    def start_worker(self, worker_id: str) -> bool:
        """Start a worker."""
        pass

    @abstractmethod
    def save_group_config(self, group_id: str) -> bool:
        """Save group configuration."""
        pass


class HAGroupManager:
    """
    Manages High Availability group associations for ProcGuard.

    This class provides comprehensive failover capabilities including:
    - Creating active/standby group associations
    - Monitoring worker health in the active group
    - Automatic failover when workers fail
    - Migrating workers between groups
    - Updating PyTorch configuration automatically

    Attributes:
        client: Client for API communication
        state_file: Path to JSON file for persisting associations
        auto_save: Whether to automatically save state

    Example:
        >>> from procguard.core.ha_manager import HAGroupManager, HTTPHAClient
        >>> client = HTTPHAClient("http://localhost:5000")
        >>> manager = HAGroupManager(client)
        >>> manager.create_association("active-group", "standby-group", world_size=4)
        >>> manager.start_monitoring()
    """

    def __init__(
        self,
        client: HAClientBase,
        state_file: str = "ha_associations.json",
        auto_save: bool = True,
    ):
        """Initialize the HA Group Manager.

        Args:
            client: Client for API communication
            state_file: Path to JSON file for persisting associations
            auto_save: Whether to automatically save state changes
        """
        self.client = client
        self.state_file = Path(state_file)
        self.auto_save = auto_save
        self.logger = logging.getLogger("HAGroupManager")

        self._lock = threading.RLock()
        self._associations: Dict[str, AssociationConfig] = {}
        self._status_cache: Dict[str, AssociationStatus] = {}

        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

    def create_association(
        self,
        active_group_id: str,
        standby_group_id: str,
        world_size: int = 4,
        master_addr: Optional[str] = None,
        master_port: int = 29500,
        backend: str = "nccl",
        failover_threshold: int = 1,
        auto_failover: bool = True,
    ) -> Optional[str]:
        """Create a new active/standby group association.

        Args:
            active_group_id: ID of the active (primary) group
            standby_group_id: ID of the standby (backup) group
            world_size: Target number of workers in active group
            master_addr: Master address for PyTorch config
            master_port: Master port for PyTorch config
            backend: Distributed training backend
            failover_threshold: Number of failed workers to trigger failover
            auto_failover: Whether to automatically failover

        Returns:
            Association ID if successful, None otherwise
        """
        with self._lock:
            association_id = f"assoc_{active_group_id}_{standby_group_id}"

            if association_id in self._associations:
                self.logger.warning(f"Association {association_id} already exists")
                return None

            config = AssociationConfig(
                association_id=association_id,
                active_group_id=active_group_id,
                standby_group_id=standby_group_id,
                world_size=world_size,
                master_addr=master_addr,
                master_port=master_port,
                backend=backend,
                failover_threshold=failover_threshold,
                auto_failover=auto_failover,
            )

            self._associations[association_id] = config
            self._save_state()

            self.logger.info(
                f"Created association {association_id}: "
                f"{active_group_id} (active) <-> {standby_group_id} (standby)"
            )

            return association_id

    def delete_association(self, association_id: str) -> bool:
        """Delete an association.

        Args:
            association_id: ID of the association to delete

        Returns:
            True if deleted, False if not found
        """
        with self._lock:
            if association_id not in self._associations:
                return False

            del self._associations[association_id]
            self._status_cache.pop(association_id, None)
            self._save_state()

            self.logger.info(f"Deleted association {association_id}")
            return True

    def get_association(self, association_id: str) -> Optional[AssociationConfig]:
        """Get association configuration.

        Args:
            association_id: ID of the association

        Returns:
            AssociationConfig if found, None otherwise
        """
        return self._associations.get(association_id)

    def list_associations(self) -> List[AssociationConfig]:
        """List all associations.

        Returns:
            List of all AssociationConfig objects
        """
        return list(self._associations.values())

    def check_association_health(self, association_id: str) -> Optional[AssociationStatus]:
        """Check the health of an association.

        Args:
            association_id: ID of the association

        Returns:
            AssociationStatus object or None if not found
        """
        config = self._associations.get(association_id)
        if not config:
            return None

        all_workers = self.client.get_workers()

        active_running = 0
        active_failed = []
        standby_available = []

        for worker_id, worker in all_workers.items():
            group = self.client.get_worker_group(worker_id)
            status = worker.get("status", "unknown")

            if group == config.active_group_id:
                if status == "running":
                    active_running += 1
                elif status in ["failed", "exited", "unknown"]:
                    active_failed.append(worker_id)
            elif group == config.standby_group_id:
                if status == "stopped":
                    standby_available.append(worker_id)

        failover_needed = (
            len(active_failed) >= config.failover_threshold
            or (
                active_running < config.world_size
                and len(standby_available) > 0
            )
        )

        if failover_needed and config.auto_failover:
            state = AssociationState.FAILING_OVER
        elif active_running >= config.world_size:
            state = AssociationState.ACTIVE
        elif active_running > 0:
            state = AssociationState.DEGRADED
        else:
            state = AssociationState.STANDBY

        health_status = "healthy"
        if failover_needed:
            health_status = "failover_required"
        elif active_running < config.world_size and not standby_available:
            health_status = "no_standby"
        elif active_failed:
            health_status = "degraded"

        status = AssociationStatus(
            association_id=association_id,
            state=state,
            active_workers=active_running,
            standby_available=len(standby_available),
            health_status=health_status,
            last_health_check=datetime.now().isoformat(),
        )

        self._status_cache[association_id] = status
        return status

    def execute_failover(self, association_id: str) -> bool:
        """Execute failover for an association.

        Moves failed workers from active group to standby,
        starts standby workers, and moves available workers
        from standby to active to maintain world size.

        Args:
            association_id: ID of the association

        Returns:
            True if failover completed, False otherwise
        """
        config = self._associations.get(association_id)
        if not config:
            return False

        self.logger.info(f"Executing failover for {association_id}")

        all_workers = self.client.get_workers()

        failed_or_stopped = []
        standby_available = []

        for worker_id, worker in all_workers.items():
            group = self.client.get_worker_group(worker_id)
            status = worker.get("status", "unknown")

            if group == config.active_group_id:
                if status in ["failed", "exited", "unknown", "stopped"]:
                    failed_or_stopped.append(worker_id)
            elif group == config.standby_group_id:
                if status == "stopped":
                    standby_available.append(worker_id)

        self.logger.info(
            f"Failover {association_id}: "
            f"{len(failed_or_stopped)} failed/stopped in active, "
            f"{len(standby_available)} stopped in standby"
        )

        running_in_active = sum(
            1 for w_id, w in all_workers.items()
            if self.client.get_worker_group(w_id) == config.active_group_id
            and w.get("status") == "running"
        )

        target_count = max(0, config.world_size - running_in_active)
        self.logger.info(
            f"Failover {association_id}: {running_in_active} running, "
            f"target: {config.world_size}, need to move: {target_count}"
        )

        if target_count <= 0:
            self.logger.info(f"Failover {association_id}: no workers need to be moved")
            return True

        if not standby_available:
            self.logger.info(f"Failover {association_id}: no stopped workers in standby")
            return False

        workers_to_move = min(len(standby_available), target_count)
        moved_count = 0

        for worker_id in standby_available[:workers_to_move]:
            self.logger.info(f"Failover {association_id}: starting worker {worker_id}")

            if not self.client.start_worker(worker_id):
                self.logger.warning(f"Failover {association_id}: failed to start {worker_id}")
                continue

            self.logger.info(f"Failover {association_id}: waiting for {worker_id} to start...")
            time.sleep(3)

            self.logger.info(f"Failover {association_id}: moving {worker_id} to active group")

            if self.client.move_worker_to_group(worker_id, config.active_group_id):
                moved_count += 1
                self.logger.info(f"Failover {association_id}: moved {worker_id} to active")
            else:
                self.logger.warning(f"Failover {association_id}: failed to move {worker_id}")

            time.sleep(1)

        if moved_count > 0:
            self.logger.info(f"Failover {association_id}: saving group config")
            self.client.save_group_config(config.active_group_id)
            self.logger.info(f"Failover {association_id}: moved {moved_count} workers from standby to active")

            config.failover_count += 1
            config.last_failover = datetime.now().isoformat()

        return moved_count > 0

    def start_monitoring(self, check_interval: float = 10.0):
        """Start background monitoring for all associations.

        Args:
            check_interval: Seconds between health checks
        """
        if self._monitoring:
            self.logger.warning("Monitoring already started")
            return

        self._monitoring = True

        def _monitor_loop():
            self.logger.info("HA monitoring started")

            while self._monitoring:
                try:
                    for association_id in list(self._associations.keys()):
                        if not self._monitoring:
                            break

                        config = self._associations.get(association_id)
                        if not config or not config.enabled:
                            continue

                        status = self.check_association_health(association_id)

                        if status and status.health_status == "failover_required" and config.auto_failover:
                            self.logger.info(f"Auto-triggering failover for {association_id}")
                            self.execute_failover(association_id)

                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")

                time.sleep(check_interval)

            self.logger.info("HA monitoring stopped")

        self._monitor_thread = threading.Thread(target=_monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
            self._monitor_thread = None

    def get_all_statuses(self) -> Dict[str, AssociationStatus]:
        """Get status for all associations.

        Returns:
            Dictionary mapping association_id to status
        """
        return self._status_cache.copy()

    def _save_state(self):
        """Save association state to disk."""
        if not self.auto_save:
            return

        try:
            data = {"associations": []}
            for assoc in self._associations.values():
                data["associations"].append(assoc.to_dict())

            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Saved {len(self._associations)} associations to {self.state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save association state: {e}")


class HTTPHAClient(HAClientBase):
    """HTTP-based client for HA operations."""

    def __init__(self, base_url: str = "http://localhost:5000", timeout: float = 10.0):
        """Initialize HTTP client.

        Args:
            base_url: Base URL of the ProcGuard manager API server
            timeout: Request timeout in seconds
        """
        import requests

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self.logger = logging.getLogger("HAGroupManager.HTTPClient")

    def request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Send HTTP request to the API."""
        import requests

        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = self._session.get(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "POST":
                response = self._session.post(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "PUT":
                response = self._session.put(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "DELETE":
                response = self._session.delete(url, timeout=self.timeout, **kwargs)
            else:
                self.logger.error(f"Unsupported HTTP method: {method}")
                return None

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            return None

    def get_workers(self) -> Dict[str, Dict[str, Any]]:
        """Get all workers."""
        data = self.request("GET", "/api/workers")
        if not data:
            return {}
        return data

    def get_worker_group(self, worker_id: str) -> Optional[str]:
        """Get the group ID that a worker belongs to."""
        data = self.request("GET", f"/api/groups/worker/{worker_id}")
        if data and data.get("success"):
            group = data.get("group")
            if group:
                return group.get("group_id")
        return None

    def get_group(self, group_id: str) -> Optional[Dict[str, Any]]:
        """Get group information."""
        data = self.request("GET", f"/api/groups/{group_id}")
        if data and data.get("success"):
            return data.get("group")
        return None

    def move_worker_to_group(self, worker_id: str, target_group_id: str) -> bool:
        """Move a worker to a target group."""
        data = self.request(
            "POST",
            "/api/groups/move-worker",
            json={"worker_id": worker_id, "target_group_id": target_group_id}
        )
        if data and data.get("success"):
            return True
        self.logger.error(f"Failed to move worker {worker_id}: {data}")
        return False

    def start_worker(self, worker_id: str) -> bool:
        """Start a worker."""
        data = self.request("POST", f"/api/workers/{worker_id}/start")
        if data and data.get("success"):
            return True
        self.logger.error(f"Failed to start worker {worker_id}: {data}")
        return False

    def save_group_config(self, group_id: str) -> bool:
        """Save group configuration."""
        data = self.request("POST", f"/api/groups/{group_id}/config", json={})
        if data and data.get("success"):
            return True
        self.logger.warning(f"Failed to save group config for {group_id}: {data}")
        return False


def get_ha_manager(
    base_url: str = "http://localhost:5000",
    state_file: str = "ha_associations.json",
    auto_save: bool = True,
) -> HAGroupManager:
    """Create an HAGroupManager with HTTP client.

    Args:
        base_url: Base URL of the ProcGuard manager API server
        state_file: Path to JSON file for persisting associations
        auto_save: Whether to automatically save state changes

    Returns:
        Configured HAGroupManager instance
    """
    client = HTTPHAClient(base_url)
    return HAGroupManager(client, state_file=state_file, auto_save=auto_save)
