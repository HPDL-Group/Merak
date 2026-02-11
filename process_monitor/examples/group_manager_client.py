#!/usr/bin/env python3
"""
ProcGuard Group Management Client

This script provides command-line and programmatic interfaces for managing
ProcGuard worker groups, including querying group information, managing
workers within groups, and configuring PyTorch distributed training
environment variables.

Features:
- List all groups and their configurations
- Get detailed information about a specific group
- View workers within a group
- Move workers between groups
- Configure PyTorch distributed training settings
- Start, stop, and restart workers
- View worker logs

Example:
    $ python group_manager_client.py --manager-url http://localhost:5000 list-groups
    $ python group_manager_client.py --manager-url http://localhost:5000 get-group group-1
    $ python group_manager_client.py --manager-url http://localhost:5000 move-worker worker-1 --to-group group-2
    $ python group_manager_client.py --manager-url http://localhost:5000 set-config group-1 --master-addr 10.0.0.1 --world-size 4
"""

import os
import sys
import json
import argparse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import requests


class WorkerState(Enum):
    """Worker state enumeration matching the server-side states."""
    RUNNING = "running"
    STOPPED = "stopped"
    STARTING = "starting"
    FAILED = "failed"
    EXITED = "exited"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, state_str: str) -> "WorkerState":
        """Convert string to WorkerState enum."""
        state_mapping = {
            "running": cls.RUNNING,
            "stopped": cls.STOPPED,
            "starting": cls.STARTING,
            "failed": cls.FAILED,
            "exited": cls.EXITED,
            "unknown": cls.UNKNOWN,
            "运行中": cls.RUNNING,
            "已停止": cls.STOPPED,
            "启动中": cls.STARTING,
            "已失败": cls.FAILED,
            "已退出": cls.EXITED,
            "未知": cls.UNKNOWN,
        }
        return state_mapping.get(state_str.lower(), cls.UNKNOWN)


@dataclass
class PyTorchConfig:
    """PyTorch distributed training configuration."""
    master_addr: Optional[str] = None
    master_port: int = 29500
    world_size: int = 0
    backend: str = "nccl"
    cuda_visible_devices: Optional[str] = None
    nccl_socket_ifname: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "world_size": self.world_size,
            "backend": self.backend,
        }
        if self.cuda_visible_devices is not None:
            result["cuda_visible_devices"] = self.cuda_visible_devices
        if self.nccl_socket_ifname is not None:
            result["nccl_socket_ifname"] = self.nccl_socket_ifname
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PyTorchConfig":
        """Create from dictionary."""
        return cls(
            master_addr=data.get("master_addr"),
            master_port=data.get("master_port", 29500),
            world_size=data.get("world_size", 0),
            backend=data.get("backend", "nccl"),
            cuda_visible_devices=data.get("cuda_visible_devices"),
            nccl_socket_ifname=data.get("nccl_socket_ifname"),
        )


@dataclass
class GroupInfo:
    """Information about a worker group."""
    group_id: str
    name: str
    workers: List[str]
    worker_count: int
    config: Optional[PyTorchConfig] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupInfo":
        """Create GroupInfo from dictionary."""
        config = None
        if "config" in data and data["config"]:
            config = PyTorchConfig.from_dict(data["config"])

        return cls(
            group_id=data.get("group_id", ""),
            name=data.get("name", ""),
            workers=data.get("workers", []),
            worker_count=data.get("worker_count", 0),
            config=config,
        )


@dataclass
class GroupSummary:
    """Summary statistics for groups."""
    total_groups: int = 0
    total_workers: int = 0
    grouped_workers: int = 0
    ungrouped_workers: int = 0

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GroupSummary":
        """Create GroupSummary from dictionary."""
        return cls(
            total_groups=data.get("total_groups", 0),
            total_workers=data.get("total_workers", 0),
            grouped_workers=data.get("grouped_workers", 0),
            ungrouped_workers=data.get("ungrouped_workers", 0),
        )


@dataclass
class AssociationConfig:
    """Configuration for group association (active/standby pair)."""
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

    def __post_init__(self):
        from datetime import datetime
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

    def to_dict(self) -> Dict[str, Any]:
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
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AssociationConfig":
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
        )


@dataclass
class AssociationStatus:
    """Status of a group association."""
    association_id: str
    state: str
    active_workers: int = 0
    standby_available: int = 0
    last_failover: Optional[str] = None
    failover_count: int = 0
    last_health_check: Optional[str] = None
    health_status: str = "healthy"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "association_id": self.association_id,
            "state": self.state,
            "active_workers": self.active_workers,
            "standby_available": self.standby_available,
            "last_failover": self.last_failover,
            "failover_count": self.failover_count,
            "last_health_check": self.last_health_check,
            "health_status": self.health_status,
        }


class HAClient:
    """Client for managing high-availability group associations."""

    def __init__(self, base_url: str = "http://localhost:5000", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "POST":
                response = self.session.post(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "PUT":
                response = self.session.put(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=self.timeout, **kwargs)
            else:
                print(f"Error: Unsupported HTTP method: {method}")
                return None

            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            print(f"Error: Failed to connect to {url}")
            print(f"       {e}")
            return None
        except requests.exceptions.Timeout as e:
            print(f"Error: Request to {url} timed out")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"Error: HTTP error {e.response.status_code} for {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error: Request failed: {e}")
            return None

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
        data = self._request(
            "POST",
            "/api/ha/associations",
            json={
                "active_group_id": active_group_id,
                "standby_group_id": standby_group_id,
                "world_size": world_size,
                "master_addr": master_addr,
                "master_port": master_port,
                "backend": backend,
                "failover_threshold": failover_threshold,
                "auto_failover": auto_failover,
            }
        )
        if data and data.get("success"):
            print(f"Created association: {data.get('association_id')}")
            return data.get("association_id")
        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to create association: {error}")
        return None

    def delete_association(self, association_id: str) -> bool:
        data = self._request("DELETE", f"/api/ha/associations/{association_id}")
        if data and data.get("success"):
            print(f"Deleted association: {association_id}")
            return True
        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to delete association: {error}")
        return False

    def list_associations(self) -> List[AssociationConfig]:
        data = self._request("GET", "/api/ha/associations")
        if not data or not data.get("success"):
            return []
        associations = []
        for assoc_data in data.get("associations", []):
            associations.append(AssociationConfig.from_dict(assoc_data))
        return associations

    def get_association_status(self, association_id: str) -> Optional[AssociationStatus]:
        data = self._request("GET", f"/api/ha/associations/{association_id}/status")
        if data and data.get("success"):
            status_data = data.get("status", {})
            return AssociationStatus(
                association_id=status_data.get("association_id", association_id),
                state=status_data.get("state", "unknown"),
                active_workers=status_data.get("active_workers", 0),
                standby_available=status_data.get("standby_available", 0),
                last_failover=status_data.get("last_failover"),
                failover_count=status_data.get("failover_count", 0),
                last_health_check=status_data.get("last_health_check"),
                health_status=status_data.get("health_status", "unknown"),
            )
        return None

    def trigger_failover(self, association_id: str) -> bool:
        data = self._request("POST", f"/api/ha/associations/{association_id}/failover")
        if data and data.get("success"):
            print(f"Failover triggered for: {association_id}")
            return True
        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to trigger failover: {error}")
        return False

    def start_monitoring(self, association_id: str) -> bool:
        data = self._request("POST", f"/api/ha/associations/{association_id}/monitor/start")
        if data and data.get("success"):
            print(f"Monitoring started for: {association_id}")
            return True
        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to start monitoring: {error}")
        return False

    def stop_monitoring(self, association_id: str) -> bool:
        data = self._request("POST", f"/api/ha/associations/{association_id}/monitor/stop")
        if data and data.get("success"):
            print(f"Monitoring stopped for: {association_id}")
            return True
        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to stop monitoring: {error}")
        return False

    def print_association_status(self, association_id: str, config: AssociationConfig, status: AssociationStatus):
        state_icons = {
            "active": "🟢",
            "standby": "🟡",
            "failover": "🔄",
            "degraded": "🟠",
            "unhealthy": "🔴",
        }
        icon = state_icons.get(status.state, "⚪")

        health_indicators = {
            "healthy": "✓",
            "failover_required": "⚠ failover needed",
            "no_standby": "⚠ no standby workers",
            "degraded": "△ degraded",
        }
        health = health_indicators.get(status.health_status, "?")

        print(f"\n{icon} Association: {association_id}")
        print(f"   Active Group:   {config.active_group_id}")
        print(f"   Standby Group:  {config.standby_group_id}")
        print(f"   World Size:     {config.world_size}")
        print(f"   State:          {status.state}")
        print(f"   Health:         {health}")
        print(f"   Running:        {status.active_workers}/{config.world_size}")
        print(f"   Standby Avail:  {status.standby_available}")
        print(f"   Auto Failover: {'Yes' if config.auto_failover else 'No'}")
        print(f"   Failover Count: {status.failover_count}")


@dataclass
class WorkerInfo:
    """Information about a worker."""
    worker_id: str
    status: str
    pid: Optional[int] = None
    command: Optional[str] = None
    start_time: Optional[str] = None
    restart_count: int = 0
    worker_type: str = "local"
    last_heartbeat: Optional[str] = None
    rank: Optional[int] = None
    local_rank: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkerInfo":
        """Create WorkerInfo from dictionary."""
        return cls(
            worker_id=data.get("worker_id", ""),
            status=data.get("status", "unknown"),
            pid=data.get("pid"),
            command=data.get("command"),
            start_time=data.get("start_time"),
            restart_count=data.get("restart_count", 0),
            worker_type=data.get("type", "local"),
            last_heartbeat=data.get("last_heartbeat"),
            rank=data.get("rank"),
            local_rank=data.get("local_rank"),
        )

    def is_running(self) -> bool:
        """Check if worker is currently running."""
        return self.status in ["running", "运行中"]

    def is_stopped(self) -> bool:
        """Check if worker is stopped."""
        return self.status in ["stopped", "已停止"]


class GroupManagerClient:
    """Client for interacting with ProcGuard group management API.

    This class provides methods for querying and managing worker groups
    in a ProcGuard cluster.

    Attributes:
        base_url: Base URL of the ProcGuard manager API server
        timeout: Request timeout in seconds

    Example:
        >>> client = GroupManagerClient("http://localhost:5000")
        >>> groups = client.list_groups()
        >>> print(f"Found {len(groups)} groups")
    """

    def __init__(self, base_url: str = "http://localhost:5000", timeout: float = 10.0):
        """Initialize the group management client.

        Args:
            base_url: Base URL of the ProcGuard manager API server
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, method: str, endpoint: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Send HTTP request to the API.

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            **kwargs: Additional arguments passed to requests

        Returns:
            Response data as dictionary, or None if request failed
        """
        url = f"{self.base_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = self.session.get(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "POST":
                response = self.session.post(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "PUT":
                response = self.session.put(url, timeout=self.timeout, **kwargs)
            elif method.upper() == "DELETE":
                response = self.session.delete(url, timeout=self.timeout, **kwargs)
            else:
                print(f"Error: Unsupported HTTP method: {method}")
                return None

            response.raise_for_status()
            return response.json()
        except requests.exceptions.ConnectionError as e:
            print(f"Error: Failed to connect to {url}")
            print(f"       {e}")
            return None
        except requests.exceptions.Timeout as e:
            print(f"Error: Request to {url} timed out")
            return None
        except requests.exceptions.HTTPError as e:
            print(f"Error: HTTP error {e.response.status_code} for {url}")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error: Request failed: {e}")
            return None

    def list_groups(self) -> List[GroupInfo]:
        """List all worker groups.

        Returns:
            List of GroupInfo objects representing all groups

        Example:
            >>> groups = client.list_groups()
            >>> for group in groups:
            ...     print(f"{group.name}: {group.worker_count} workers")
        """
        data = self._request("GET", "/api/groups")
        if not data or not data.get("success"):
            print("Failed to list groups")
            return []

        groups = []
        for group_data in data.get("groups", []):
            groups.append(GroupInfo.from_dict(group_data))

        return groups

    def get_group_summary(self) -> Optional[GroupSummary]:
        """Get summary statistics for all groups.

        Returns:
            GroupSummary object with group statistics, or None if failed
        """
        data = self._request("GET", "/api/groups")
        if not data or not data.get("success"):
            return None

        return GroupSummary.from_dict(data.get("summary", {}))

    def get_group(self, group_id: str) -> Optional[GroupInfo]:
        """Get detailed information about a specific group.

        Args:
            group_id: Unique identifier of the group

        Returns:
            GroupInfo object with group details, or None if not found

        Example:
            >>> group = client.get_group("group-1")
            >>> if group:
            ...     print(f"Group: {group.name}")
            ...     print(f"Workers: {group.workers}")
        """
        data = self._request("GET", f"/api/groups/{group_id}")
        if not data or not data.get("success"):
            print(f"Group '{group_id}' not found")
            return None

        return GroupInfo.from_dict(data.get("group", {}))

    def get_worker_group(self, worker_id: str) -> Optional[str]:
        """Get the group ID that a worker belongs to.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            Group ID string, or None if worker not in any group

        Example:
            >>> group_id = client.get_worker_group("worker-0")
            >>> if group_id:
            ...     print(f"Worker belongs to group: {group_id}")
        """
        data = self._request("GET", f"/api/groups/worker/{worker_id}")
        if not data or not data.get("success"):
            return None

        group_info = data.get("group")
        return group_info.get("group_id") if group_info else None

    def get_group_workers(self, group_id: str) -> Optional[List[str]]:
        """Get list of worker IDs in a specific group.

        Args:
            group_id: Unique identifier of the group

        Returns:
            List of worker IDs, or None if group not found

        Example:
            >>> workers = client.get_group_workers("group-1")
            >>> print(f"Workers in group: {workers}")
        """
        group = self.get_group(group_id)
        if group:
            return group.workers
        return None

    def add_worker_to_group(self, group_id: str, worker_id: str) -> bool:
        """Add a worker to a group.

        If the worker is already in another group, it will be moved.

        Args:
            group_id: Unique identifier of the target group
            worker_id: Unique identifier of the worker to add

        Returns:
            True if successful, False otherwise

        Example:
            >>> success = client.add_worker_to_group("group-1", "worker-0")
            >>> if success:
            ...     print("Worker added successfully")
        """
        data = self._request("POST", f"/api/groups/{group_id}/workers/{worker_id}")
        if data and data.get("success"):
            print(f"Worker '{worker_id}' added to group '{group_id}'")
            return True

        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to add worker: {error}")
        return False

    def remove_worker_from_group(self, group_id: str, worker_id: str) -> bool:
        """Remove a worker from a group.

        The worker will continue running but will no longer be part of the group.

        Args:
            group_id: Unique identifier of the group
            worker_id: Unique identifier of the worker to remove

        Returns:
            True if successful, False otherwise

        Example:
            >>> success = client.remove_worker_from_group("group-1", "worker-0")
            >>> if success:
            ...     print("Worker removed from group")
        """
        data = self._request("DELETE", f"/api/groups/{group_id}/workers/{worker_id}")
        if data and data.get("success"):
            print(f"Worker '{worker_id}' removed from group '{group_id}'")
            return True

        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to remove worker: {error}")
        return False

    def move_worker(self, worker_id: str, target_group_id: str) -> bool:
        """Move a worker from its current group to a target group.

        Args:
            worker_id: Unique identifier of the worker to move
            target_group_id: Unique identifier of the target group

        Returns:
            True if successful, False otherwise

        Example:
            >>> success = client.move_worker("worker-0", "group-2")
            >>> if success:
            ...     print("Worker moved successfully")
        """
        current_group = self.get_worker_group(worker_id)

        data = self._request(
            "POST",
            "/api/groups/move-worker",
            json={"worker_id": worker_id, "target_group_id": target_group_id}
        )
        if data and data.get("success"):
            print(f"Worker '{worker_id}' moved from '{current_group or 'no group'}' to '{target_group_id}'")
            return True

        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to move worker: {error}")
        return False

    def set_group_pytorch_config(
        self,
        group_id: str,
        master_addr: Optional[str] = None,
        master_port: Optional[int] = None,
        world_size: Optional[int] = None,
        backend: Optional[str] = None,
        cuda_visible_devices: Optional[str] = None,
        nccl_socket_ifname: Optional[str] = None,
    ) -> bool:
        """Configure PyTorch distributed training settings for a group.

        Args:
            group_id: Unique identifier of the group
            master_addr: Address of the master node
            master_port: Port for master node communication
            world_size: Total number of processes in distributed training
            backend: Distributed training backend ("nccl", "gloo", etc.)
            cuda_visible_devices: GPU devices visible to workers
            nccl_socket_ifname: Network interface for NCCL

        Returns:
            True if successful, False otherwise

        Example:
            >>> success = client.set_group_pytorch_config(
            ...     "group-1",
            ...     master_addr="10.0.0.1",
            ...     master_port=29500,
            ...     world_size=4,
            ...     backend="nccl"
            ... )
            >>> if success:
            ...     print("PyTorch configuration updated")
        """
        config_data = {}
        if master_addr is not None:
            config_data["master_addr"] = master_addr
        if master_port is not None:
            config_data["master_port"] = master_port
        if world_size is not None:
            config_data["world_size"] = world_size
        if backend is not None:
            config_data["backend"] = backend
        if cuda_visible_devices is not None:
            config_data["cuda_visible_devices"] = cuda_visible_devices
        if nccl_socket_ifname is not None:
            config_data["nccl_socket_ifname"] = nccl_socket_ifname

        if not config_data:
            print("Error: No configuration values provided")
            return False

        data = self._request(
            "PUT",
            f"/api/groups/{group_id}/config",
            json=config_data
        )
        if data and data.get("success"):
            print(f"PyTorch configuration updated for group '{group_id}'")
            return True

        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to update configuration: {error}")
        return False

    def get_group_pytorch_config(self, group_id: str) -> Optional[PyTorchConfig]:
        """Get PyTorch configuration for a group.

        Args:
            group_id: Unique identifier of the group

        Returns:
            PyTorchConfig object, or None if group not found

        Example:
            >>> config = client.get_group_pytorch_config("group-1")
            >>> if config:
            ...     print(f"Master: {config.master_addr}:{config.master_port}")
            ...     print(f"World Size: {config.world_size}")
        """
        group = self.get_group(group_id)
        if group:
            return group.config
        return None

    def list_all_workers(self) -> List[WorkerInfo]:
        """List all workers in the cluster.

        Returns:
            List of WorkerInfo objects representing all workers

        Example:
            >>> workers = client.list_all_workers()
            >>> for worker in workers:
            ...     print(f"{worker.worker_id}: {worker.status}")
        """
        data = self._request("GET", "/api/workers")
        if not data:
            return []

        workers = []
        for worker_id, worker_data in data.items():
            if isinstance(worker_data, dict):
                workers.append(WorkerInfo.from_dict(worker_data))
        return workers

    def get_worker(self, worker_id: str) -> Optional[WorkerInfo]:
        """Get information about a specific worker.

        Args:
            worker_id: Unique identifier of the worker

        Returns:
            WorkerInfo object, or None if worker not found

        Example:
            >>> worker = client.get_worker("worker-0")
            >>> if worker:
            ...     print(f"Worker: {worker.worker_id}")
            ...     print(f"Status: {worker.status}")
            ...     print(f"PID: {worker.pid}")
        """
        all_workers = self.list_all_workers()
        for worker in all_workers:
            if worker.worker_id == worker_id:
                return worker

        data = self._request("GET", f"/api/workers/{worker_id}")
        if data and isinstance(data, dict) and "worker" in data:
            return WorkerInfo.from_dict(data["worker"])

        return None

    def start_worker(self, worker_id: str) -> bool:
        """Start a worker.

        Args:
            worker_id: Unique identifier of the worker to start

        Returns:
            True if successful, False otherwise

        Example:
            >>> success = client.start_worker("worker-0")
            >>> if success:
            ...     print("Worker started successfully")
        """
        data = self._request("POST", f"/api/workers/{worker_id}/start")
        if data and data.get("success"):
            print(f"Worker '{worker_id}' start command sent")
            return True

        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to start worker: {error}")
        return False

    def stop_worker(self, worker_id: str, force: bool = False) -> bool:
        """Stop a worker.

        Args:
            worker_id: Unique identifier of the worker to stop
            force: If True, forcefully kill the worker

        Returns:
            True if successful, False otherwise

        Example:
            >>> success = client.stop_worker("worker-0")
            >>> if success:
            ...     print("Worker stop command sent")
            >>> success = client.stop_worker("worker-1", force=True)
        """
        data = self._request(
            "POST",
            f"/api/workers/{worker_id}/stop",
            json={"force": force}
        )
        if data and data.get("success"):
            print(f"Worker '{worker_id}' stop command sent (force={force})")
            return True

        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to stop worker: {error}")
        return False

    def restart_worker(self, worker_id: str) -> bool:
        """Restart a worker.

        Args:
            worker_id: Unique identifier of the worker to restart

        Returns:
            True if successful, False otherwise

        Example:
            >>> success = client.restart_worker("worker-0")
            >>> if success:
            ...     print("Worker restart command sent")
        """
        data = self._request("POST", f"/api/workers/{worker_id}/restart")
        if data and data.get("success"):
            print(f"Worker '{worker_id}' restart command sent")
            return True

        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to restart worker: {error}")
        return False

    def get_worker_logs(self, worker_id: str, lines: int = 50) -> List[str]:
        """Get logs for a worker.

        Args:
            worker_id: Unique identifier of the worker
            lines: Number of log lines to retrieve (default: 50)

        Returns:
            List of log lines

        Example:
            >>> logs = client.get_worker_logs("worker-0", lines=100)
            >>> for line in logs[-20:]:
            ...     print(line)
        """
        data = self._request("GET", f"/api/workers/{worker_id}/logs")
        if data and "logs" in data:
            all_logs = data["logs"]
            return all_logs[-lines:] if lines > 0 else all_logs
        return []

    def print_worker_info(self, worker: WorkerInfo, verbose: bool = False) -> None:
        """Print formatted worker information.

        Args:
            worker: WorkerInfo object to print
            verbose: If True, print additional details
        """
        status_icon = "🟢" if worker.is_running() else "🔴" if worker.is_stopped() else "🟡"
        print(f"\n{status_icon} Worker: {worker.worker_id}")
        print(f"   Status: {worker.status}")
        print(f"   Type: {worker.worker_type}")

        if worker.pid:
            print(f"   PID: {worker.pid}")
        if worker.rank is not None:
            print(f"   Rank: {worker.rank}")
        if worker.local_rank is not None:
            print(f"   Local Rank: {worker.local_rank}")
        if worker.command:
            cmd_preview = worker.command[:60] + "..." if len(worker.command) > 60 else worker.command
            print(f"   Command: {cmd_preview}")
        if worker.start_time:
            print(f"   Started: {worker.start_time}")
        if worker.restart_count > 0:
            print(f"   Restart Count: {worker.restart_count}")
        if worker.last_heartbeat:
            print(f"   Last Heartbeat: {worker.last_heartbeat}")

    def create_group(self, group_id: str, name: str) -> bool:
        """Create a new worker group.

        Args:
            group_id: Unique identifier for the new group
            name: Human-readable name for the group

        Returns:
            True if successful, False otherwise

        Example:
            >>> success = client.create_group("group-3", "Evaluation Group")
            >>> if success:
            ...     print("Group created successfully")
        """
        data = self._request(
            "POST",
            "/api/groups",
            json={"group_id": group_id, "name": name}
        )
        if data and data.get("success"):
            print(f"Group '{group_id}' created with name '{name}'")
            return True

        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to create group: {error}")
        return False

    def delete_group(self, group_id: str) -> bool:
        """Delete a worker group.

        Workers in the group become ungrouped but continue running.

        Args:
            group_id: Unique identifier of the group to delete

        Returns:
            True if successful, False otherwise

        Example:
            >>> success = client.delete_group("group-3")
            >>> if success:
            ...     print("Group deleted successfully")
        """
        data = self._request("DELETE", f"/api/groups/{group_id}")
        if data and data.get("success"):
            print(f"Group '{group_id}' deleted")
            return True

        error = data.get("error", "Unknown error") if data else "No response"
        print(f"Failed to delete group: {error}")
        return False

    def get_pytorch_env_vars(self, group_id: Optional[str] = None) -> Dict[str, str]:
        """Get environment variables for PyTorch distributed training.

        Args:
            group_id: Optional group ID to get group-specific configuration

        Returns:
            Dictionary of environment variables for PyTorch distributed training

        Example:
            >>> env_vars = client.get_pytorch_env_vars("group-1")
            >>> for key, value in env_vars.items():
            ...     print(f"export {key}={value}")
        """
        endpoint = "/api/pytorch/env"
        if group_id:
            endpoint = f"/api/pytorch/env?group_id={group_id}"

        data = self._request("GET", endpoint)
        if data:
            return {k: str(v) for k, v in data.items() if v is not None}
        return {}

    def print_group_info(self, group: GroupInfo, verbose: bool = False) -> None:
        """Print formatted group information.

        Args:
            group: GroupInfo object to print
            verbose: If True, print additional details
        """
        print(f"\n{'='*60}")
        print(f"Group: {group.name} ({group.group_id})")
        print(f"{'='*60}")
        print(f"  Worker Count: {group.worker_count}")
        print(f"  Workers: {', '.join(group.workers) if group.workers else 'None'}")

        if group.config:
            print(f"\n  PyTorch Configuration:")
            print(f"    Master Address: {group.config.master_addr or 'Not set'}")
            print(f"    Master Port: {group.config.master_port}")
            print(f"    World Size: {group.config.world_size}")
            print(f"    Backend: {group.config.backend}")
            if group.config.cuda_visible_devices:
                print(f"    CUDA Devices: {group.config.cuda_visible_devices}")
            if group.config.nccl_socket_ifname:
                print(f"    NCCL Socket: {group.config.nccl_socket_ifname}")

        if verbose and group.workers:
            print(f"\n  Worker Details:")
            for worker_id in group.workers:
                print(f"    - {worker_id}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="ProcGuard Group Management Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  List all groups:
    %(prog)s list-groups

  Get detailed group info:
    %(prog)s get-group group-1

  Get workers in a group:
    %(prog)s get-workers group-1

  Move worker to another group:
    %(prog)s move-worker worker-1 --to-group group-2

  Set PyTorch configuration:
    %(prog)s set-config group-1 --master-addr 10.0.0.1 --world-size 4

  Create new group:
    %(prog)s create-group group-3 "Evaluation"

  Get PyTorch environment variables:
    %(prog)s get-env --group-id group-1

  List all workers:
    %(prog)s list-workers

  Get worker details:
    %(prog)s get-worker worker-0

  Start a worker:
    %(prog)s start-worker worker-0

  Stop a worker (with force):
    %(prog)s stop-worker worker-0 --force

  Restart a worker:
    %(prog)s restart-worker worker-0

  Get worker logs:
    %(prog)s worker-logs worker-0 --lines 100
        """
    )

    parser.add_argument(
        "--manager-url",
        type=str,
        default=os.environ.get("PROCGuard_URL", "http://localhost:5000"),
        help="ProcGuard manager URL (default: PROCGuard_URL env or http://localhost:5000)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    list_parser = subparsers.add_parser("list-groups", help="List all groups")
    list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    get_parser = subparsers.add_parser("get-group", help="Get group details")
    get_parser.add_argument("group_id", type=str, help="Group ID")

    workers_parser = subparsers.add_parser("get-workers", help="Get workers in a group")
    workers_parser.add_argument("group_id", type=str, help="Group ID")

    move_parser = subparsers.add_parser("move-worker", help="Move worker between groups")
    move_parser.add_argument("worker_id", type=str, help="Worker ID to move")
    move_parser.add_argument("--to-group", required=True, type=str, help="Target group ID")

    add_parser = subparsers.add_parser("add-worker", help="Add worker to group")
    add_parser.add_argument("worker_id", type=str, help="Worker ID to add")
    add_parser.add_argument("group_id", type=str, help="Target group ID")

    remove_parser = subparsers.add_parser("remove-worker", help="Remove worker from group")
    remove_parser.add_argument("worker_id", type=str, help="Worker ID to remove")
    remove_parser.add_argument("group_id", type=str, help="Source group ID")

    config_parser = subparsers.add_parser("set-config", help="Set PyTorch configuration")
    config_parser.add_argument("group_id", type=str, help="Group ID")
    config_parser.add_argument("--master-addr", type=str, help="Master node address")
    config_parser.add_argument("--master-port", type=int, help="Master node port")
    config_parser.add_argument("--world-size", type=int, help="Total world size")
    config_parser.add_argument("--backend", type=str, help="Distributed backend (nccl, gloo)")
    config_parser.add_argument("--cuda-devices", type=str, help="CUDA visible devices")
    config_parser.add_argument("--nccl-socket", type=str, help="NCCL socket interface")

    get_config_parser = subparsers.add_parser("get-config", help="Get PyTorch configuration")
    get_config_parser.add_argument("group_id", type=str, help="Group ID")

    env_parser = subparsers.add_parser("get-env", help="Get PyTorch environment variables")
    env_parser.add_argument("--group-id", type=str, help="Optional group ID")

    create_parser = subparsers.add_parser("create-group", help="Create new group")
    create_parser.add_argument("group_id", type=str, help="Group ID")
    create_parser.add_argument("name", type=str, help="Group name")

    delete_parser = subparsers.add_parser("delete-group", help="Delete a group")
    delete_parser.add_argument("group_id", type=str, help="Group ID to delete")

    list_workers_parser = subparsers.add_parser("list-workers", help="List all workers in the cluster")
    list_workers_parser.add_argument("--status", type=str, help="Filter by status (running, stopped, failed)")
    list_workers_parser.add_argument("--json", action="store_true", help="Output as JSON")

    get_worker_parser = subparsers.add_parser("get-worker", help="Get detailed worker information")
    get_worker_parser.add_argument("worker_id", type=str, help="Worker ID")

    start_worker_parser = subparsers.add_parser("start-worker", help="Start a stopped worker")
    start_worker_parser.add_argument("worker_id", type=str, help="Worker ID to start")

    stop_worker_parser = subparsers.add_parser("stop-worker", help="Stop a running worker")
    stop_worker_parser.add_argument("worker_id", type=str, help="Worker ID to stop")
    stop_worker_parser.add_argument("--force", action="store_true", help="Forcefully kill the worker")

    restart_worker_parser = subparsers.add_parser("restart-worker", help="Restart a worker")
    restart_worker_parser.add_argument("worker_id", type=str, help="Worker ID to restart")

    worker_logs_parser = subparsers.add_parser("worker-logs", help="Get logs for a worker")
    worker_logs_parser.add_argument("worker_id", type=str, help="Worker ID")
    worker_logs_parser.add_argument("--lines", type=int, default=50, help="Number of log lines (default: 50)")

    ha_parser = subparsers.add_parser("ha", help="HA association management commands")
    ha_subparsers = ha_parser.add_subparsers(dest="ha_command", help="HA commands")

    ha_list_parser = ha_subparsers.add_parser("list", help="List all HA associations")
    ha_list_parser.add_argument("--json", action="store_true", help="Output as JSON")

    ha_create_parser = ha_subparsers.add_parser("create", help="Create HA association")
    ha_create_parser.add_argument("active_group", type=str, help="Active group ID")
    ha_create_parser.add_argument("standby_group", type=str, help="Standby group ID")
    ha_create_parser.add_argument("--world-size", type=int, default=4, help="World size (default: 4)")
    ha_create_parser.add_argument("--master-addr", type=str, help="Master address for PyTorch config")
    ha_create_parser.add_argument("--master-port", type=int, default=29500, help="Master port (default: 29500)")
    ha_create_parser.add_argument("--backend", type=str, default="nccl", help="Distributed backend (default: nccl)")
    ha_create_parser.add_argument("--failover-threshold", type=int, default=1, help="Failover threshold (default: 1)")
    ha_create_parser.add_argument("--no-auto-failover", action="store_false", dest="auto_failover", help="Disable auto failover")

    ha_delete_parser = ha_subparsers.add_parser("delete", help="Delete HA association")
    ha_delete_parser.add_argument("association_id", type=str, help="Association ID")

    ha_status_parser = ha_subparsers.add_parser("status", help="Check HA association status")
    ha_status_parser.add_argument("association_id", type=str, help="Association ID")

    ha_failover_parser = ha_subparsers.add_parser("failover", help="Trigger manual failover")
    ha_failover_parser.add_argument("association_id", type=str, help="Association ID")

    ha_monitor_parser = ha_subparsers.add_parser("monitor", help="Control association monitoring")
    ha_monitor_parser.add_argument("association_id", type=str, help="Association ID")
    ha_monitor_parser.add_argument("--start", action="store_true", help="Start monitoring")
    ha_monitor_parser.add_argument("--stop", action="store_true", help="Stop monitoring")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    client = GroupManagerClient(base_url=args.manager_url)

    if args.command == "list-groups":
        groups = client.list_groups()
        if args.json:
            output = {
                "success": True,
                "groups": [
                    {
                        "group_id": g.group_id,
                        "name": g.name,
                        "worker_count": g.worker_count,
                        "workers": g.workers,
                        "config": g.config.to_dict() if g.config else None
                    }
                    for g in groups
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            summary = client.get_group_summary()
            if summary:
                print(f"\nTotal Groups: {summary.total_groups}")
                print(f"Total Workers: {summary.total_workers}")
                print(f"Grouped Workers: {summary.grouped_workers}")
                print(f"Ungrouped Workers: {summary.ungrouped_workers}")

            for group in groups:
                print(f"\n{group.name} ({group.group_id})")
                print(f"  Workers: {group.worker_count}")
                if group.config and group.config.master_addr:
                    print(f"  Master: {group.config.master_addr}:{group.config.master_port}")
                    print(f"  World Size: {group.config.world_size}")

                client.print_group_info(group, verbose=True)

    elif args.command == "get-workers":
        workers = client.get_group_workers(args.group_id)
        if workers is not None:
            print(f"\nWorkers in group '{args.group_id}':")
            for worker_id in workers:
                print(f"  - {worker_id}")
        else:
            print(f"Group '{args.group_id}' not found")
            sys.exit(1)

    elif args.command == "move-worker":
        success = client.move_worker(args.worker_id, args.to_group)
        sys.exit(0 if success else 1)

    elif args.command == "add-worker":
        success = client.add_worker_to_group(args.group_id, args.worker_id)
        sys.exit(0 if success else 1)

    elif args.command == "remove-worker":
        success = client.remove_worker_from_group(args.group_id, args.worker_id)
        sys.exit(0 if success else 1)

    elif args.command == "set-config":
        success = client.set_group_pytorch_config(
            args.group_id,
            master_addr=args.master_addr,
            master_port=args.master_port,
            world_size=args.world_size,
            backend=args.backend,
            cuda_visible_devices=args.cuda_devices,
            nccl_socket_ifname=args.nccl_socket,
        )
        sys.exit(0 if success else 1)

    elif args.command == "get-config":
        config = client.get_group_pytorch_config(args.group_id)
        if config:
            print(json.dumps(config.to_dict(), indent=2))
        else:
            print(json.dumps({"error": "Group not found"}, indent=2))
            sys.exit(1)

    elif args.command == "get-group":
        group = client.get_group(args.group_id)
        if group:
            client.print_group_info(group, verbose=True)
        else:
            print(f"Group '{args.group_id}' not found")
            sys.exit(1)

    elif args.command == "get-env":
        env_vars = client.get_pytorch_env_vars(args.group_id)
        if env_vars:
            for key, value in sorted(env_vars.items()):
                print(f"{key}={value}")
        else:
            print("No environment variables found")

    elif args.command == "create-group":
        success = client.create_group(args.group_id, args.name)
        sys.exit(0 if success else 1)

    elif args.command == "delete-group":
        success = client.delete_group(args.group_id)
        sys.exit(0 if success else 1)

    elif args.command == "list-workers":
        workers = client.list_all_workers()
        if args.status:
            workers = [w for w in workers if w.status.lower() == args.status.lower()]

        if args.json:
            output = {
                "success": True,
                "workers": [
                    {
                        "worker_id": w.worker_id,
                        "status": w.status,
                        "pid": w.pid,
                        "type": w.worker_type,
                        "rank": w.rank,
                        "local_rank": w.local_rank,
                        "restart_count": w.restart_count
                    }
                    for w in workers
                ]
            }
            print(json.dumps(output, indent=2))
        else:
            running = [w for w in workers if w.is_running()]
            stopped = [w for w in workers if w.is_stopped()]
            other = [w for w in workers if not w.is_running() and not w.is_stopped()]

            print(f"\nTotal Workers: {len(workers)}")
            print(f"  Running: {len(running)}")
            print(f"  Stopped: {len(stopped)}")
            print(f"  Other: {len(other)}")

            for worker in workers:
                client.print_worker_info(worker)

    elif args.command == "get-worker":
        worker = client.get_worker(args.worker_id)
        if worker:
            client.print_worker_info(worker, verbose=True)
        else:
            print(f"Worker '{args.worker_id}' not found")
            sys.exit(1)

    elif args.command == "start-worker":
        success = client.start_worker(args.worker_id)
        sys.exit(0 if success else 1)

    elif args.command == "stop-worker":
        success = client.stop_worker(args.worker_id, force=args.force)
        sys.exit(0 if success else 1)

    elif args.command == "restart-worker":
        success = client.restart_worker(args.worker_id)
        sys.exit(0 if success else 1)

    elif args.command == "worker-logs":
        logs = client.get_worker_logs(args.worker_id, lines=args.lines)
        if logs:
            print(f"\n=== Logs for {args.worker_id} (last {len(logs)} lines) ===")
            for line in logs:
                print(line)
        else:
            print(f"No logs available for worker '{args.worker_id}'")

    elif args.command == "ha":
        ha_client = HAClient(base_url=args.manager_url)

        if args.ha_command == "list":
            associations = ha_client.list_associations()
            if args.json:
                output = {
                    "success": True,
                    "associations": [assoc.to_dict() for assoc in associations]
                }
                print(json.dumps(output, indent=2))
            else:
                print(f"\nHA Associations ({len(associations)}):")
                for assoc in associations:
                    print(f"  - {assoc.association_id}: {assoc.active_group_id} (active) <-> {assoc.standby_group_id} (standby)")

        elif args.ha_command == "create":
            assoc_id = ha_client.create_association(
                args.active_group,
                args.standby_group,
                world_size=args.world_size,
                master_addr=args.master_addr,
                master_port=args.master_port,
                backend=args.backend,
                failover_threshold=args.failover_threshold,
                auto_failover=args.auto_failover,
            )
            sys.exit(0 if assoc_id else 1)

        elif args.ha_command == "delete":
            success = ha_client.delete_association(args.association_id)
            sys.exit(0 if success else 1)

        elif args.ha_command == "status":
            config = None
            for assoc in ha_client.list_associations():
                if assoc.association_id == args.association_id:
                    config = assoc
                    break

            if not config:
                print(f"Association '{args.association_id}' not found")
                sys.exit(1)

            status = ha_client.get_association_status(args.association_id)
            if status:
                ha_client.print_association_status(args.association_id, config, status)
            else:
                print(f"Failed to get status for '{args.association_id}'")
                sys.exit(1)

        elif args.ha_command == "failover":
            success = ha_client.trigger_failover(args.association_id)
            sys.exit(0 if success else 1)

        elif args.ha_command == "monitor":
            if args.start:
                success = ha_client.start_monitoring(args.association_id)
            elif args.stop:
                success = ha_client.stop_monitoring(args.association_id)
            else:
                print("Error: --start or --stop required")
                sys.exit(1)
            sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
