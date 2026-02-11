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
Distributed Training Monitor for ProcGuard Cluster Management.

This module provides the DistMonitor class and utility functions for
monitoring and coordinating PyTorch distributed training across a
ProcGuard cluster. It handles worker discovery, rank assignment,
health checking, synchronization barriers, and failure detection.
It also includes High Availability (HA) monitoring for active/standby
group failover management.

Classes:
    WorkerState: Enum representing worker status states
    WorkerMeta: Dataclass containing worker metadata
    GroupInfo: Dataclass containing group configuration
    ClusterMeta: Dataclass containing cluster-wide metadata
    GroupMeta: Dataclass containing group metadata
    AssociationState: Enum representing HA association states
    FailoverTrigger: Enum representing failover trigger conditions
    AssociationConfig: Dataclass containing HA association configuration
    AssociationStatus: Dataclass containing HA association status
    DistMonitor: Main class for distributed training monitoring

Functions:
    get_monitor: Create a DistMonitor with environment-based configuration
    get_local_rank_info: Get rank info from environment variables
    get_local_master_info: Get master info from environment variables

Example:
    >>> monitor = get_monitor()
    >>> with monitor:
    ...     cluster = monitor.get_cluster()
    ...     print(f"World size: {cluster.world_size}")
    ...     monitor.wait_for_all_workers(timeout=300)
"""

import os
import time
import asyncio
import logging
import threading
import requests
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

from procguard.core.ha_manager import (
    AssociationState,
    FailoverTrigger,
    AssociationConfig,
    AssociationStatus,
    HAGroupManager,
    HTTPHAClient,
    get_ha_manager,
)


class WorkerState(Enum):
    运行中 = "running"
    已停止 = "stopped"
    启动中 = "starting"
    已失败 = "failed"
    已退出 = "exited"
    未知 = "unknown"


@dataclass
class WorkerMeta:
    worker_id: str
    rank: Optional[int] = None
    local_rank: Optional[int] = None
    node_rank: Optional[int] = None
    state: WorkerState = WorkerState.未知
    pid: Optional[int] = None
    last_heartbeat: Optional[str] = None


@dataclass
class GroupInfo:
    group_id: str = ""
    name: str = ""
    workers: List[str] = field(default_factory=list)
    master_addr: Optional[str] = None
    master_port: int = 29500
    world_size: int = 0
    backend: str = "nccl"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "name": self.name,
            "workers": self.workers,
            "master_addr": self.master_addr,
            "master_port": self.master_port,
            "world_size": self.world_size,
            "backend": self.backend,
        }


class ClusterMeta:
    world_size: int = 0
    master_addr: Optional[str] = None
    master_port: int = 29500
    backend: str = "nccl"
    workers: Dict[str, WorkerMeta] = field(default_factory=dict)

    def __init__(
        self,
        world_size: int = 0,
        master_addr: Optional[str] = None,
        master_port: int = 29500,
        backend: str = "nccl",
        workers: Optional[Dict[str, WorkerMeta]] = None,
    ):
        self.world_size = world_size
        self.master_addr = master_addr
        self.master_port = master_port
        self.backend = backend
        self.workers = workers if workers is not None else {}


@dataclass
class GroupMeta:
    group_id: str
    name: str
    worker_count: int = 0
    workers: List[str] = field(default_factory=list)
    master_addr: Optional[str] = None
    master_port: int = 29500
    world_size: int = 0
    backend: str = "nccl"


class DistMonitor:
    """Distributed Training Monitor for ProcGuard Cluster Management.

    This class provides comprehensive monitoring and coordination capabilities
    for PyTorch distributed training across a ProcGuard cluster. It handles
    worker discovery, rank assignment, health checking, and synchronization.

    The monitor maintains a cached view of the cluster state and can
    automatically refresh from the ProcGuard manager at configurable intervals.
    It supports both blocking and asynchronous operations for flexibility.

    Attributes:
        manager_url: URL of the ProcGuard manager API server
        worker_id: Optional worker identifier for this instance
        cache_interval: Time in seconds between cache refreshes
        request_timeout: Timeout for API requests in seconds
        enable_auto_refresh: Whether to automatically refresh cluster state

    Example:
        >>> monitor = DistMonitor(manager_url="http://localhost:5000", worker_id="worker-0")
        >>> cluster = monitor.get_cluster()
        >>> print(f"World size: {cluster.world_size}")
        >>> monitor.wait_for_all_workers(timeout=300)
    """

    def __init__(
        self,
        manager_url: Optional[str] = None,
        worker_id: Optional[str] = None,
        cache_interval: float = 5.0,
        request_timeout: float = 5.0,
        enable_auto_refresh: bool = True
    ):
        self._manager_url = manager_url or os.environ.get("PROCGuard_URL", "http://localhost:5000")
        self._worker_id = worker_id or os.environ.get("PROCGuard_WORKER_ID", "")
        self._cache_interval = cache_interval
        self._request_timeout = request_timeout
        self._enable_auto_refresh = enable_auto_refresh
        self._logger = logging.getLogger(__name__)

        self._cluster_meta: Optional[ClusterMeta] = None
        self._last_refresh_time: float = 0
        self._last_pytorch_config: Optional[Dict[str, Any]] = None
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._config_changed_callbacks: List[callable] = []

        self._ha_associations: Dict[str, AssociationConfig] = {}
        self._ha_status_cache: Dict[str, AssociationStatus] = {}
        self._ha_monitoring = False
        self._ha_monitor_thread: Optional[threading.Thread] = None
        self._ha_check_interval: float = 10.0

        if self._enable_auto_refresh:
            self._start_background_refresh()

    def register_config_change_callback(self, callback: callable):
        """Register a callback for PyTorch configuration changes.

        Args:
            callback: Function to call when PyTorch configuration changes.
                      The callback receives (old_config, new_config) as arguments.

        Example:
            >>> def on_config_change(old_cfg, new_cfg):
            ...     print(f"Config changed: {old_cfg} -> {new_cfg}")
            >>> monitor.register_config_change_callback(on_config_change)
        """
        self._config_changed_callbacks.append(callback)

    def _notify_config_changed(self, old_config: Optional[Dict], new_config: Dict):
        """Notify all registered callbacks about configuration changes.

        Args:
            old_config: Previous PyTorch configuration dictionary
            new_config: New PyTorch configuration dictionary
        """
        for callback in self._config_changed_callbacks:
            try:
                callback(old_config, new_config)
            except Exception as e:
                self._logger.warning(f"配置变更回调失败: {e}")

    def _start_background_refresh(self):
        """Start a background thread for automatic cluster state refresh.

        The background thread periodically calls refresh() at intervals
        specified by cache_interval to keep the cluster state up-to-date.
        """
        def _refresh_loop():
            while True:
                try:
                    self.refresh()
                except Exception as e:
                    self._logger.debug(f"后台刷新失败: {e}")
                time.sleep(self._cache_interval)

        thread = threading.Thread(target=_refresh_loop, daemon=True)
        thread.start()
        self._logger.info(f"启动后台刷新，间隔={self._cache_interval}秒")

    def _send_request(self, endpoint: str, method: str = "GET", **kwargs) -> Optional[Dict[str, Any]]:
        """Send an HTTP request to the ProcGuard manager API.

        Args:
            endpoint: API endpoint path (e.g., '/api/status')
            method: HTTP method ('GET' or 'POST')
            **kwargs: Additional arguments passed to requests.get/post

        Returns:
            Optional[Dict]: JSON response as dictionary, or None if request failed

        Example:
            >>> status = monitor._send_request("/api/status")
            >>> workers = monitor._send_request("/api/workers", method="POST")
        """
        url = f"{self._manager_url}{endpoint}"
        try:
            if method.upper() == "GET":
                response = requests.get(url, timeout=self._request_timeout, **kwargs)
            elif method.upper() == "POST":
                response = requests.post(url, timeout=self._request_timeout, **kwargs)
            else:
                return None

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self._logger.warning(f"请求 {url} 失败: {e}")
            return None

    def refresh(self) -> ClusterMeta:
        with self._lock:
            current_time = time.time()
            if current_time - self._last_refresh_time < self._cache_interval and self._cluster_meta is not None:
                return self._cluster_meta

            self._last_refresh_time = current_time

        status_data = self._send_request("/api/status")
        workers_data = self._send_request("/api/workers")

        if not status_data or not workers_data:
            return self._cluster_meta

        cluster_meta = ClusterMeta()
        new_pytorch_config = None

        if "pytorch_config" in status_data:
            pytorch_config = status_data["pytorch_config"]
            cluster_meta.master_addr = pytorch_config.get("master_addr")
            cluster_meta.master_port = pytorch_config.get("master_port", 29500)
            cluster_meta.world_size = pytorch_config.get("world_size", 0)
            cluster_meta.backend = pytorch_config.get("backend", "nccl")
            new_pytorch_config = pytorch_config

            if self._last_pytorch_config != new_pytorch_config:
                old_config = self._last_pytorch_config
                self._last_pytorch_config = new_pytorch_config
                if old_config:
                    self._logger.warning(f"[DistMonitor] ⚠ PyTorch 配置已变更!")
                    self._logger.warning(f"  World Size: {old_config.get('world_size')} -> {new_pytorch_config.get('world_size')}")
                    self._logger.warning(f"  Master: {old_config.get('master_addr')}:{old_config.get('master_port')} -> {new_pytorch_config.get('master_addr')}:{new_pytorch_config.get('master_port')}")
                    self._notify_config_changed(old_config, new_pytorch_config)
                else:
                    self._logger.info(f"[DistMonitor] ✓ PyTorch 配置已加载: world_size={new_pytorch_config.get('world_size')}")

        local_rank_info = {
            "rank": int(os.environ.get("RANK", -1)),
            "local_rank": int(os.environ.get("LOCAL_RANK", -1)),
            "node_rank": int(os.environ.get("NODE_RANK", -1)),
            "world_size": int(os.environ.get("WORLD_SIZE", 0)),
        }

        for worker_id, worker_data in workers_data.items():
            worker_meta = WorkerMeta(
                worker_id=worker_id,
                state=WorkerState(worker_data.get("status", "unknown")),
                pid=worker_data.get("pid"),
                last_heartbeat=worker_data.get("last_heartbeat")
            )

            if worker_id == self._worker_id and local_rank_info["rank"] >= 0:
                worker_meta.rank = local_rank_info["rank"]
                worker_meta.local_rank = local_rank_info["local_rank"]
                worker_meta.node_rank = local_rank_info["node_rank"]
            else:
                rank_info = status_data.get("ranks", {}).get(worker_id)
                if rank_info and rank_info.get("rank") is not None:
                    worker_meta.rank = rank_info.get("rank")
                    worker_meta.local_rank = rank_info.get("local_rank")
                    worker_meta.node_rank = rank_info.get("node_rank")
                else:
                    local_rank = None
                    if '-' in worker_id:
                        try:
                            local_rank = int(worker_id.split('-')[-1])
                        except ValueError:
                            pass
                    worker_meta.local_rank = local_rank

            cluster_meta.workers[worker_id] = worker_meta

        self._logger.debug(f"[Rank分配] 遍历完成，所有运行中worker的rank:")
        running_workers = [w for w in cluster_meta.workers.values() if w.state == WorkerState.运行中]
        for w in running_workers:
            self._logger.debug(f"  - {w.worker_id}: rank={w.rank}, local_rank={w.local_rank}")

        self._logger.debug(f"[Rank分配] 运行中的worker数量: {len(running_workers)}")
        if running_workers:
            existing_ranks = set(w.rank for w in running_workers if w.rank is not None)
            self._logger.debug(f"[Rank分配] 已分配的rank: {sorted(existing_ranks)}")
            missing_ranks = []
            for i in range(len(running_workers)):
                if i not in existing_ranks:
                    missing_ranks.append(i)
            self._logger.debug(f"[Rank分配] 缺少的rank: {missing_ranks}")
            missing_ranks.reverse()

            for worker_meta in running_workers:
                if worker_meta.rank is None and missing_ranks:
                    old_rank = worker_meta.rank
                    new_rank = missing_ranks.pop()
                    worker_meta.rank = new_rank
                    self._logger.debug(f"[Rank分配] Worker '{worker_meta.worker_id}' rank: {old_rank} -> {new_rank}")
                    self._send_request(
                        f"/api/workers/{worker_meta.worker_id}/rank",
                        method="POST",
                        json={"rank": new_rank}
                    )

        running_count = sum(
            1 for w in cluster_meta.workers.values() if w.state == WorkerState.运行中
        )
        cluster_meta.world_size = max(cluster_meta.world_size, running_count)

        with self._lock:
            self._cluster_meta = cluster_meta

        return cluster_meta

    def get_cluster(self) -> ClusterMeta:
        """Get the current cluster metadata.

        Makes direct API call to get real-time data (no cache).
        This ensures accurate status during worker startup and failover.

        Returns:
            ClusterMeta: Current cluster state including workers and configuration

        Example:
            >>> cluster = monitor.get_cluster()
            >>> print(f"Master: {cluster.master_addr}:{cluster.master_port}")
            >>> print(f"World size: {cluster.world_size}")
        """
        with self._lock:
            self._last_refresh_time = 0
        return self.refresh()

    def get_worker_state(self, worker_id: Optional[str] = None) -> WorkerState:
        """Get the current state of a worker.

        Args:
            worker_id: Worker identifier. If None, uses this instance's worker_id.

        Returns:
            WorkerState: Current state of the worker (running, stopped, failed, etc.)

        Example:
            >>> state = monitor.get_worker_state("worker-0")
            >>> if state == WorkerState.运行中:
            ...     print("Worker is running")
        """
        cluster_meta = self.get_cluster()
        target_id = worker_id or self._worker_id

        if target_id and target_id in cluster_meta.workers:
            return cluster_meta.workers[target_id].state
        return WorkerState.未知

    def is_worker_active(self, worker_id: Optional[str] = None) -> bool:
        """Check if a worker is currently running.

        Args:
            worker_id: Worker identifier. If None, uses this instance's worker_id.

        Returns:
            bool: True if worker is in 'running' state

        Example:
            >>> if monitor.is_worker_active("worker-0"):
            ...     print("Worker is active")
        """
        return self.get_worker_state(worker_id) == WorkerState.运行中

    def list_active_workers(self) -> List[WorkerMeta]:
        """Get list of all workers currently in running state.

        Returns:
            List[WorkerMeta]: List of WorkerMeta for active workers, empty if cluster unavailable

        Example:
            >>> active = monitor.list_active_workers()
            >>> for worker in active:
            ...     print(f"{worker.worker_id}: rank={worker.rank}")
        """
        cluster_meta = self.get_cluster()
        if not cluster_meta:
            return []
        return [
            worker for worker in cluster_meta.workers.values()
            if worker.state == WorkerState.运行中
        ]

    def get_rank(self, worker_id: str) -> Optional[int]:
        """Get the rank assigned to a specific worker.

        Args:
            worker_id: Worker identifier

        Returns:
            Optional[int]: Rank number if assigned, None otherwise

        Example:
            >>> rank = monitor.get_rank("worker-0")
            >>> if rank is not None:
            ...     print(f"Worker rank: {rank}")
        """
        cluster_meta = self.get_cluster()
        if not cluster_meta:
            return None
        if worker_id in cluster_meta.workers:
            return cluster_meta.workers[worker_id].rank
        return None

    def get_worker_by_rank(self, rank: int) -> Optional[WorkerMeta]:
        """Find the worker with a specific rank.

        Args:
            rank: Rank number to search for

        Returns:
            Optional[WorkerMeta]: WorkerMeta if found, None otherwise

        Example:
            >>> worker = monitor.get_worker_by_rank(0)
            >>> if worker:
            ...     print(f"Rank 0 worker: {worker.worker_id}")
        """
        cluster_meta = self.get_cluster()
        if not cluster_meta:
            return None
        for worker in cluster_meta.workers.values():
            if worker.rank == rank:
                return worker
        return None

    def get_self_info(self) -> Optional[WorkerMeta]:
        """Get metadata for this worker instance.

        Returns information about the current worker including its rank,
        state, PID, and last heartbeat time.

        Returns:
            Optional[WorkerMeta]: WorkerMeta for this worker, None if not registered

        Example:
            >>> info = monitor.get_self_info()
            >>> if info:
            ...     print(f"My rank: {info.rank}, state: {info.state}")
        """
        if not self._worker_id:
            self._logger.warning(f"WORKER_ID 环境变量未设置")
            return None
        cluster_meta = self.get_cluster()
        if not cluster_meta:
            self._logger.warning(f"无法获取集群信息")
            return None
        return cluster_meta.workers.get(self._worker_id)

    def get_world_size(self) -> int:
        """Get the total number of processes in the distributed training.

        Returns:
            int: Total world size from cluster config, 0 if unavailable

        Example:
            >>> size = monitor.get_world_size()
            >>> print(f"Distributed training with {size} processes")
        """
        cluster_meta = self.get_cluster()
        if not cluster_meta:
            return 0
        return cluster_meta.world_size

    def list_all_ranks(self) -> List[int]:
        """Get list of all assigned ranks in the cluster.

        Returns sorted list of unique rank values currently assigned
        to running workers.

        Returns:
            List[int]: Sorted list of rank values, empty if cluster unavailable

        Example:
            >>> ranks = monitor.list_all_ranks()
            >>> print(f"Active ranks: {ranks}")
        """
        cluster_meta = self.get_cluster()
        if not cluster_meta:
            return []
        ranks = []

        for worker in cluster_meta.workers.values():
            if worker.rank is not None:
                ranks.append(worker.rank)

        if not ranks:
            world_size = cluster_meta.world_size
            if world_size > 0:
                ranks = list(range(world_size))
            else:
                ranks = list(range(len(cluster_meta.workers)))

        return sorted(set(ranks))

    def list_all_workers(self) -> Dict[str, WorkerMeta]:
        """Get metadata for all workers in the cluster.

        Returns:
            Dict[str, WorkerMeta]: Dictionary mapping worker_id to WorkerMeta

        Example:
            >>> workers = monitor.list_all_workers()
            >>> for wid, worker in workers.items():
            ...     print(f"{wid}: state={worker.state}")
        """
        cluster_meta = self.get_cluster()
        if not cluster_meta:
            return {}
        return cluster_meta.workers.copy()

    def wait_for_workers(
        self,
        expected_count: Optional[int] = None,
        timeout: float = 60.0,
        check_interval: float = 1.0
    ) -> bool:
        """Wait for a minimum number of active workers in the same group.

        Blocks until expected_count workers are running in the group,
        or uses the group's world_size if expected_count is not specified.
        The timeout is reached.

        Args:
            expected_count: Minimum number of active workers to wait for.
                          If None, uses the group's world_size from config.
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds

        Returns:
            bool: True if expected count reached, False if timeout

        Example:
            >>> # Wait for all workers in the group (uses group world_size)
            >>> if monitor.wait_for_workers(timeout=120):
            ...     print("All group workers ready")
            >>>
            >>> # Wait for specific number of workers
            >>> if monitor.wait_for_workers(expected_count=2, timeout=60):
            ...     print("Minimum workers ready")
        """
        if expected_count is None:
            group_info = self.get_my_group_info()
            if group_info and group_info.world_size > 0:
                expected_count = group_info.world_size
                self._logger.info(f"[DistMonitor] 使用组 world_size: {expected_count}")
            else:
                self._logger.warning("[DistMonitor] 未指定 expected_count 且组未配置 world_size")
                return False

        start_time = time.time()

        while time.time() - start_time < timeout:
            active_workers = self.list_active_workers()
            group_workers = self._get_group_workers({w.worker_id: w for w in active_workers})
            current_count = len(group_workers)

            if current_count >= expected_count:
                self._logger.info(f"[DistMonitor] ✓ 组内已有 {current_count}/{expected_count} 个活跃 worker，达到预期")
                return True

            self._logger.debug(f"[DistMonitor] 等待组内 worker: {current_count}/{expected_count}")
            time.sleep(check_interval)

        self._logger.warning(f"[DistMonitor] ✗ 等待组内 worker 超时: 期望 {expected_count}，实际 {current_count}")
        return False

    def wait_for_all_workers(
        self,
        timeout: float = 300.0,
        check_interval: float = 1.0
    ) -> bool:
        """Wait for all workers in the cluster to be ready.

        Blocks until the number of active workers equals the configured
        world size, or raises a timeout error.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds

        Returns:
            bool: True if all workers ready

        Raises:
            RuntimeError: If timeout reached before all workers ready

        Example:
            >>> monitor.wait_for_all_workers(timeout=300)
        """
        start_time = time.time()
        self._logger.info("等待所有 Worker 就绪...")

        while time.time() - start_time < timeout:
            cluster_meta = self.get_cluster()
            if not cluster_meta:
                time.sleep(check_interval)
                continue

            world_size = cluster_meta.world_size
            active_workers = self.list_active_workers()
            active_count = len(active_workers)

            if active_count >= world_size and world_size > 0:
                elapsed = time.time() - start_time
                self._logger.info(f"所有 Worker 已就绪 ({active_count}/{world_size})，耗时 {elapsed:.1f}秒")
                return True

            remaining = timeout - (time.time() - start_time)
            self._logger.debug(f"等待 Worker 就绪: {active_count}/{world_size}，剩余 {remaining:.1f}秒")
            time.sleep(check_interval)

        raise RuntimeError(f"等待所有 Worker 超时 ({timeout}秒)")

    def wait_for_rank(self, rank: int, timeout: float = 30.0, check_interval: float = 0.5) -> bool:
        """Wait for a specific rank to become active.

        Blocks until the worker with the given rank is running.

        Args:
            rank: Rank number to wait for
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds

        Returns:
            bool: True if worker with rank became active, False if timeout

        Example:
            >>> if monitor.wait_for_rank(0, timeout=60):
            ...     print("Rank 0 is ready")
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            worker = self.get_worker_by_rank(rank)
            if worker and worker.state == WorkerState.运行中:
                return True
            time.sleep(check_interval)

        return False

    def barrier_sync(self, timeout: float = 300.0, check_interval: float = 0.1):
        """Synchronize at a barrier waiting for all ranks to be present.

        Blocks until all expected ranks (0 to world_size-1) are registered
        and active in the cluster. This provides a distributed barrier
        synchronization point.

        Args:
            timeout: Maximum time to wait in seconds
            check_interval: Time between checks in seconds

        Example:
            >>> monitor.barrier_sync(timeout=300)
            >>> print("All ranks reached barrier")
        """
        self_info = self.get_self_info()
        if not self_info or self_info.rank is None:
            self._logger.warning("无法同步栅栏: 无 rank 信息")
            return

        expected_rank = self_info.rank
        target_world_size = self.get_world_size()

        start_time = time.time()
        while time.time() - start_time < timeout:
            all_ranks = self.list_all_ranks()

            if len(all_ranks) >= target_world_size:
                all_present = all(r in all_ranks for r in range(target_world_size))
                if all_present:
                    active_count = len(self.list_active_workers())
                    if active_count >= target_world_size:
                        return

            time.sleep(check_interval)

        self._logger.warning(f"栅栏同步超时 ({timeout}秒)")

    async def async_get_cluster(self) -> ClusterMeta:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.get_cluster)

    async def async_get_worker_state(self, worker_id: str) -> WorkerState:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.get_worker_state, worker_id)

    async def async_list_active_workers(self) -> List[WorkerMeta]:
        """Async version of list_active_workers."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.list_active_workers)

    def get_env_for_worker(self, worker_id: str) -> Dict[str, str]:
        """Get PyTorch distributed training environment variables for a worker.

        Generates the environment variables needed for PyTorch distributed
        training, including MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, etc.

        Args:
            worker_id: Worker identifier to generate environment for

        Returns:
            Dict[str, str]: Environment variables dictionary for the worker

        Example:
            >>> env = monitor.get_env_for_worker("worker-0")
            >>> os.environ.update(env)
        """
        cluster_meta = self.get_cluster()
        if not cluster_meta:
            self._logger.warning(f"[DistMonitor] 无法获取集群信息，无法为 {worker_id} 生成环境变量")
            return {}
        worker_meta = cluster_meta.workers.get(worker_id)

        env = {}

        if cluster_meta.master_addr:
            env["MASTER_ADDR"] = cluster_meta.master_addr
            env["MASTER_PORT"] = str(cluster_meta.master_port)
            env["WORLD_SIZE"] = str(cluster_meta.world_size)
            env["TORCH_DISTRIBUTED_BACKEND"] = cluster_meta.backend
            self._logger.info(f"[DistMonitor] ✓ 为 {worker_id} 生成 PyTorch 环境变量:")
            self._logger.info(f"  MASTER_ADDR={env['MASTER_ADDR']}")
            self._logger.info(f"  MASTER_PORT={env['MASTER_PORT']}")
            self._logger.info(f"  WORLD_SIZE={env['WORLD_SIZE']}")
            self._logger.info(f"  TORCH_DISTRIBUTED_BACKEND={env['TORCH_DISTRIBUTED_BACKEND']}")
        else:
            self._logger.warning(f"[DistMonitor] 集群配置中缺少 MASTER_ADDR 或 WORLD_SIZE")

        if worker_meta:
            if worker_meta.rank is not None:
                env["RANK"] = str(worker_meta.rank)
            if worker_meta.local_rank is not None:
                env["LOCAL_RANK"] = str(worker_meta.local_rank)
            if worker_meta.node_rank is not None:
                env["NODE_RANK"] = str(worker_meta.node_rank)

        return env

    def is_failover_worker(self) -> bool:
        """Check if this worker is a failover replacement.

        Queries the ProcGuard manager to determine if this worker
        was started via HA failover to replace a failed worker.

        If this is a failover worker, the status will be reset after returning
        to avoid repeatedly detecting the same failover state.

        Returns:
            bool: True if this is a failover worker, False otherwise

        Example:
            >>> if monitor.is_failover_worker():
            ...     print("This worker is a failover replacement")
            ...     failover_info = monitor.get_failover_info()
            ...     print(f"Replaced: {failover_info.get('replaced_worker_id')}")
        """
        if not self._worker_id:
            self._logger.warning("[DistMonitor] No worker ID configured")
            return False

        try:
            response = requests.get(
                f"{self._manager_url}/api/workers/{self._worker_id}/failover-status",
                timeout=self._request_timeout
            )
            if response.status_code == 200:
                data = response.json()
                is_failover = data.get("is_failover_worker", False)
                self._logger.info(f"[DistMonitor] Failover status for {self._worker_id}: {is_failover}")

                if is_failover:
                    self._logger.info(f"[DistMonitor] Resetting failover status for {self._worker_id}")
                    try:
                        reset_response = requests.post(
                            f"{self._manager_url}/api/workers/{self._worker_id}/reset-failover-status",
                            timeout=self._request_timeout
                        )
                        if reset_response.status_code != 200:
                            self._logger.warning(f"[DistMonitor] Failed to reset failover status: {reset_response.status_code}")
                    except requests.exceptions.RequestException as e:
                        self._logger.error(f"[DistMonitor] Error resetting failover status: {e}")

                return is_failover
            else:
                self._logger.warning(f"[DistMonitor] Failed to get failover status: {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            self._logger.error(f"[DistMonitor] Error checking failover status: {e}")
            return False
    
    def is_recovering(self) -> bool:
        """Check if any worker in this group is under failover recovery.

        Queries the ProcGuard manager to check if there are failover workers
        within the same group as this worker.

        Returns:
            bool: True if there are workers under failover recovery in the group, False otherwise

        Example:
            >>> if monitor.is_recovering():
            ...     # 当前组内有 worker 正在进行故障转移恢复
            ...     failover_workers = monitor.get_failover_workers()
            ...     print(f"Failover workers in group: {len(failover_workers)}")
        """
        group_info = self.get_my_group_info()
        if not group_info or not group_info.workers:
            self._logger.warning("[DistMonitor] No group info or no workers in group")
            return False

        try:
            response = requests.get(
                f"{self._manager_url}/api/workers/failover-workers",
                timeout=self._request_timeout
            )
            if response.status_code != 200:
                return False

            data = response.json()
            failover_workers = data.get("workers", [])
            group_worker_ids = set(group_info.workers)

            for worker in failover_workers:
                worker_id = worker.get("worker_id")
                if worker_id and worker_id in group_worker_ids:
                    self._logger.info(f"[DistMonitor] Found failover worker in group: {worker_id}")
                    return True

            return False
        except requests.exceptions.RequestException as e:
            self._logger.error(f"[DistMonitor] Error checking failover workers: {e}")
            return False

    def get_failover_workers(self) -> List[Dict[str, Any]]:
        """Get failover workers within this group.

        Returns a list of failover workers that belong to the same group
        as this worker.

        Returns:
            List[Dict[str, Any]]: List of failover worker info in the group containing:
                - worker_id: The failover worker ID
                - failover_info: Details about the failover
                - status: Current status of the worker
                - pid: Process ID of the worker
                - rank: The PyTorch rank assigned to this worker (if any)

        Example:
            >>> workers = monitor.get_failover_workers()
            >>> for worker in workers:
            ...     print(f"Failover worker: {worker['worker_id']}, rank: {worker['rank']}")
        """
        group_info = self.get_my_group_info()
        if not group_info or not group_info.workers:
            return []

        try:
            response = requests.get(
                f"{self._manager_url}/api/workers/failover-workers",
                timeout=self._request_timeout
            )
            if response.status_code != 200:
                return []

            data = response.json()
            all_failover_workers = data.get("workers", [])
            group_worker_ids = set(group_info.workers)

            return [
                worker for worker in all_failover_workers
                if worker.get("worker_id") in group_worker_ids
            ]
        except requests.exceptions.RequestException as e:
            self._logger.error(f"[DistMonitor] Error getting failover workers: {e}")
            return []

    def get_failover_info(self) -> Optional[Dict[str, Any]]:
        """Get failover information for this worker.

        Returns detailed information about the failover process if this
        worker is a failover replacement.

        Returns:
            Optional[Dict[str, Any]]: Failover info dictionary containing:
                - association_id: The HA association ID
                - replaced_worker_id: The original failed worker
                - failover_time: When the failover occurred
                Returns None if not a failover worker or info unavailable

        Example:
            >>> info = monitor.get_failover_info()
            >>> if info:
            ...     print(f"Replaced worker: {info['replaced_worker_id']}")
            ...     print(f"Failover time: {info['failover_time']}")
        """
        if not self._worker_id:
            self._logger.warning("[DistMonitor] No worker ID configured")
            return None

        try:
            response = requests.get(
                f"{self._manager_url}/api/workers/{self._worker_id}/failover-status",
                timeout=self._request_timeout
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("is_failover_worker"):
                    return data.get("failover_info")
                return None
            return None
        except requests.exceptions.RequestException as e:
            self._logger.error(f"[DistMonitor] Error getting failover info: {e}")
            return None

    def _get_group_workers(self, all_workers: Dict[str, WorkerMeta]) -> Dict[str, WorkerMeta]:
        """Get workers that belong to the same group as this worker.

        If this worker is part of a group, returns only workers in that group.
        Otherwise, returns all workers.

        Args:
            all_workers: Dictionary of all workers

        Returns:
            Dict[str, WorkerMeta]: Workers in the same group as this worker
        """
        if not self._worker_id:
            return all_workers

        response = self._send_request("/api/groups")
        if not response or not response.get("success"):
            return all_workers

        groups_data = response.get("groups", [])
        for group_config in groups_data:
            workers_list = group_config.get("workers", [])
            group_id = group_config.get("group_id", "unknown")
            if self._worker_id in workers_list:
                self._logger.info(f"在分组 '{group_id}' 中找到 {len(workers_list)} 个 workers")
                return {
                    wid: all_workers.get(wid, WorkerMeta(worker_id=wid))
                    for wid in workers_list
                    if wid in all_workers
                }

        return all_workers

    def should_stop(self, force_refresh: bool = False) -> bool:
        """Check if the training process should stop.

        Determines whether any worker in the group has stopped, failed,
        or exited, indicating that the training should terminate.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            bool: True if any worker has stopped/failed/exited

        Example:
            >>> while not monitor.should_stop():
            ...     # Continue training
            ...     time.sleep(1)
        """
        group_info = self.get_my_group_info()
        if not group_info:
            return False

        group_id = group_info.group_id

        if force_refresh:
            with self._lock:
                self._last_refresh_time = 0

        workers = self.list_all_workers()
        if not workers:
            return False

        group_workers = self._get_group_workers(workers)

        for worker in group_workers.values():
            if worker.state in [WorkerState.已停止, WorkerState.已失败, WorkerState.已退出]:
                self._logger.info(f"[DistMonitor] Worker {worker.worker_id} has stopped")
                return True
            elif worker.state == WorkerState.运行中:
                continue
            else:
                pid = worker.pid
                if pid and not self._is_process_running(pid):
                    self._logger.info(f"[DistMonitor] Worker {worker.worker_id} process not running")
                    return True

        return False

    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with the given PID is still running.

        Args:
            pid: Process ID to check

        Returns:
            bool: True if process exists, False otherwise
        """
        try:
            import psutil
            return psutil.pid_exists(pid)
        except Exception:
            return True

    def get_failed_ranks(self, force_refresh: bool = False) -> List[int]:
        """Get list of ranks with failed workers.

        First attempts to get cached failed ranks from the manager API.
        The manager caches failed ranks when workers transition from running
        to failed/stopped/exited state.

        Args:
            force_refresh: If True, bypass manager cache and fetch fresh data

        Returns:
            List[int]: Sorted list of ranks with failed workers, empty if none
        """
        group_info = self.get_my_group_info()
        if not group_info:
            return []

        group_id = group_info.group_id

        if not force_refresh:
            try:
                response = self._send_request(f"/api/groups/failed-ranks/{group_id}")
                if response and response.get("success"):
                    failed_ranks = response.get("failed_ranks", [])
                    is_cached = response.get("cached", False)
                    if is_cached:
                        self._logger.debug(f"[DistMonitor] 从 manager 缓存获取 failed ranks: {failed_ranks}")
                        return failed_ranks
            except Exception as e:
                self._logger.debug(f"[DistMonitor] 获取缓存 failed ranks 失败: {e}")

        workers = self.list_all_workers()
        if not workers:
            return []

        group_workers = self._get_group_workers(workers)
        self._logger.debug(f"[DistMonitor] Group has {len(group_workers)} workers")

        failed_ranks = []

        for worker in group_workers.values():
            if worker.state in [WorkerState.已停止, WorkerState.已失败, WorkerState.已退出]:
                if worker.rank is not None:
                    failed_ranks.append(worker.rank)
                    self._logger.debug(f"[DistMonitor] Worker {worker.worker_id} (rank={worker.rank}) is {worker.state.value}")

        self._logger.debug(f"[DistMonitor] Failed ranks: {sorted(failed_ranks)}")
        return sorted(failed_ranks)

    def get_recover_src_rank(self, force_refresh: bool = True) -> Optional[int]:
        """Get the rank to use as source for recovery.

        Returns the rank cached from manager when failure was detected.
        This rank can be used as a reference for distributed recovery.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data from API.
                          Default is True to ensure accurate rank detection
                          in distributed training scenarios.

        Returns:
            Optional[int]: Source rank from cache, None if not available
        """
        if force_refresh:
            with self._lock:
                self._last_refresh_time = 0

        workers = self.list_all_workers()
        if not workers:
            return None

        group_workers = self._get_group_workers(workers)
        self._logger.debug(f"[DistMonitor] Group has {len(group_workers)} workers")

        running_ranks = []
        for worker in group_workers.values():
            if worker.state == WorkerState.运行中 and worker.rank is not None:
                running_ranks.append(worker.rank)
                self._logger.debug(f"[DistMonitor] Running worker: {worker.worker_id} (rank={worker.rank})")

        if not running_ranks:
            return None

        return min(running_ranks)

    def get_tcp_store_port(self) -> int:
        """Get the current TCPStore port from manager.

        This port is used for distributed process group communication.
        When process group is rebuilt after failure, all workers should
        use the same port from the manager.

        Returns:
            int: Current TCPStore port (default is MASTER_PORT from environment)

        Example:
            >>> port = monitor.get_tcp_store_port()
            >>> print(f"Using TCPStore port: {port}")
        """
        group_info = self.get_my_group_info()
        if not group_info:
            return int(os.environ.get("MASTER_PORT", 29500))

        # Use master_port from GroupInfo
        return int(group_info.master_port)

    def update_tcp_store_port(self, port: int) -> bool:
        """Update the TCPStore port in manager.

        Call this method when rebuilding process group after failure.
        The new port will be used by all workers for distributed communication.

        Args:
            port: New TCPStore port number

        Returns:
            bool: True if update was successful, False otherwise

        Example:
            >>> success = monitor.update_tcp_store_port(29501)
            >>> if success:
            ...     print("Port updated successfully")
        """
        try:
            group_info = self.get_my_group_info()
            if not group_info:
                self._logger.warning("Cannot update TCPStore port: not in any group")
                return False

            group_id = group_info.group_id

            # Update via manager API
            result = self._send_request(
                f"/api/groups/{group_id}/config",
                method="POST",
                json={"master_port": port}
            )

            if result and result.get("success"):
                self._logger.info(f"Updated TCPStore port to {port}")
                return True
            else:
                self._logger.warning(f"Failed to update TCPStore port: {result}")
                return False
        except Exception as e:
            self._logger.error(f"Error updating TCPStore port: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Perform a health check on the monitor and cluster connection.

        Verifies connectivity to the ProcGuard manager and checks
        if this worker is registered and active.

        Returns:
            Dict[str, Any]: Health status dictionary containing:
                - healthy: Overall health status
                - manager_reachable: Whether manager API is accessible
                - worker_registered: Whether this worker is registered
                - worker_active: Whether this worker is running
                - details: Additional diagnostic information

        Example:
            >>> health = monitor.health_check()
            >>> if health["healthy"]:
            ...     print("System is healthy")
        """
        result = {
            "healthy": True,
            "manager_reachable": False,
            "worker_registered": False,
            "worker_active": False,
            "details": {}
        }

        try:
            response = requests.get(
                f"{self._manager_url}/api/status",
                timeout=self._request_timeout
            )
            if response.status_code == 200:
                result["manager_reachable"] = True

                if self._worker_id:
                    workers_data = self._send_request("/api/workers")
                    if workers_data and self._worker_id in workers_data:
                        result["worker_registered"] = True
                        worker_status = workers_data[self._worker_id].get("status")
                        result["worker_active"] = worker_status == "running"
                        result["details"]["worker_status"] = worker_status

                result["details"]["cluster_size"] = response.json().get("total_workers", 0)

        except requests.exceptions.RequestException as e:
            result["healthy"] = False
            result["details"]["error"] = str(e)

        return result

    def get_groups(self, force_refresh: bool = False) -> List[GroupInfo]:
        """Get all worker groups from the cluster.

        Retrieves information about all configured worker groups including
        their members and distributed training configuration.

        Args:
            force_refresh: If True, bypass cache and fetch fresh data

        Returns:
            List[GroupInfo]: List of all worker groups, empty if unavailable

        Example:
            >>> groups = monitor.get_groups()
            >>> for group in groups:
            ...     print(f"{group.name}: {len(group.workers)} workers")
        """
        if force_refresh:
            with self._lock:
                self._last_refresh_time = 0

        response = self._send_request("/api/groups")
        if not response or not response.get("success"):
            self._logger.warning("获取分组信息失败")
            return []

        groups_data = response.get("groups", [])
        groups = []

        for group_data in groups_data:
            group_info = GroupInfo(
                group_id=group_data.get("group_id", ""),
                name=group_data.get("name", ""),
                workers=group_data.get("workers", []),
                master_addr=group_data.get("config", {}).get("master_addr"),
                master_port=group_data.get("config", {}).get("master_port", 29500),
                world_size=group_data.get("config", {}).get("world_size", 0),
                backend=group_data.get("config", {}).get("backend", "nccl"),
            )
            groups.append(group_info)

        self._logger.info(f"[DistMonitor] ✓ 获取到 {len(groups)} 个分组")
        for group in groups:
            self._logger.debug(f"[DistMonitor] 分组 {group.group_id}: {group.name}, workers={group.workers}")
        return groups

    def get_group_of_worker(self, worker_id: Optional[str] = None) -> Optional[GroupInfo]:
        """Get the group that a worker belongs to.

        Args:
            worker_id: Worker identifier. If None, uses this instance's worker_id.

        Returns:
            Optional[GroupInfo]: GroupInfo if worker is in a group, None otherwise

        Example:
            >>> group = monitor.get_group_of_worker("worker-0")
            >>> if group:
            ...     print(f"Worker in group: {group.name}")
        """
        target_id = worker_id or self._worker_id
        if not target_id:
            self._logger.warning("[DistMonitor] 未指定 worker_id")
            return None

        groups = self.get_groups()
        for group in groups:
            if target_id in group.workers:
                self._logger.info(f"[DistMonitor] Worker {target_id} 位于分组 {group.group_id}: {group.name}")
                return group

        self._logger.debug(f"[DistMonitor] 未找到 Worker {target_id} 所属的分组")
        return None

    def get_group_workers(self, group_id: Optional[str] = None) -> Dict[str, WorkerMeta]:
        """Get workers in a specific group or this worker's group.

        Args:
            group_id: Specific group ID. If None, uses this worker's group.

        Returns:
            Dict[str, WorkerMeta]: Dictionary of workers in the group

        Example:
            >>> workers = monitor.get_group_workers("group-1")
            >>> for wid, worker in workers.items():
            ...     print(f"{wid}: rank={worker.rank}")
        """
        target_group = None

        if group_id:
            groups = self.get_groups()
            for g in groups:
                if g.group_id == group_id:
                    target_group = g
                    break
        else:
            target_group = self.get_group_of_worker()

        if not target_group:
            return self.list_all_workers()

        cluster_meta = self.get_cluster()
        if not cluster_meta:
            return {}

        return {
            wid: cluster_meta.workers.get(wid, WorkerMeta(worker_id=wid))
            for wid in target_group.workers
            if wid in cluster_meta.workers
        }

    def get_my_group_info(self) -> Optional[GroupInfo]:
        """Get the group information for this worker instance.

        Returns:
            Optional[GroupInfo]: GroupInfo if this worker is in a group, None otherwise

        Example:
            >>> group = monitor.get_my_group_info()
            >>> if group:
            ...     print(f"My group: {group.name}")
        """
        return self.get_group_of_worker(self._worker_id)

    def sync_groups_from_manager(self) -> bool:
        """Synchronize group information from the ProcGuard manager.

        Fetches the latest group configuration from the manager API.

        Returns:
            bool: True if sync successful, False otherwise

        Example:
            >>> if monitor.sync_groups_from_manager():
            ...     print("Groups synced successfully")
        """
        response = self._send_request("/api/groups")
        if response and response.get("success"):
            group_count = len(response.get("groups", []))
            summary = response.get("summary", {})
            self._logger.info(f"[DistMonitor] ✓ 分组信息同步成功: {group_count} 个分组, 共 {summary.get('total_workers_in_groups', 0)} 个 workers")
            return True
        self._logger.warning("[DistMonitor] ✗ 分组信息同步失败")
        return False

    def __enter__(self):
        """Context manager entry point.

        Returns:
            DistMonitor: Self instance for use in 'with' statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point.

        Args:
            exc_type: Exception type if an exception was raised
            exc_val: Exception value if an exception was raised
            exc_tb: Exception traceback if an exception was raised

        Returns:
            False: Do not suppress exceptions
        """
        self._executor.shutdown(wait=False)
        return False

    def close(self):
        """Close the monitor and release resources.

        Shuts down the background executor and closes connections.
        """
        self._executor.shutdown(wait=True)
        self._logger.info("DistMonitor 关闭完成")


def get_monitor(
    cache_interval: float = 2.0,
    request_timeout: float = 5.0,
    enable_auto_refresh: bool = True
) -> DistMonitor:
    """Create a DistMonitor instance with environment-based configuration.

    Reads PROCGuard_URL and PROCGuard_WORKER_ID from environment variables
    and creates a DistMonitor with the specified settings.

    Args:
        cache_interval: Time between cache refreshes in seconds
        request_timeout: API request timeout in seconds
        enable_auto_refresh: Whether to enable background refresh

    Returns:
        DistMonitor: Configured DistMonitor instance

    Example:
        >>> monitor = get_monitor(cache_interval=5.0)
        >>> with monitor:
        ...     cluster = monitor.get_cluster()
    """
    manager_url = os.environ.get("PROCGuard_URL", "http://localhost:5000")
    worker_id = os.environ.get("PROCGuard_WORKER_ID", "")

    return DistMonitor(
        manager_url=manager_url,
        worker_id=worker_id,
        cache_interval=cache_interval,
        request_timeout=request_timeout,
        enable_auto_refresh=enable_auto_refresh
    )


def get_local_rank_info() -> Dict[str, int]:
    """Get local rank information from environment variables.

    Reads RANK, WORLD_SIZE, LOCAL_RANK, and NODE_RANK from the
    current process environment.

    Returns:
        Dict[str, int]: Dictionary containing:
            - rank: Global rank of this process
            - world_size: Total number of processes
            - local_rank: Local rank on this node
            - node_rank: Rank of this node

    Example:
        >>> info = get_local_rank_info()
        >>> print(f"Rank {info['rank']} of {info['world_size']}")
    """
    return {
        "rank": int(os.environ.get("RANK", 0)),
        "world_size": int(os.environ.get("WORLD_SIZE", 1)),
        "local_rank": int(os.environ.get("LOCAL_RANK", 0)),
        "node_rank": int(os.environ.get("NODE_RANK", 0)),
    }


def get_local_master_info() -> Dict[str, Any]:
    """Get PyTorch distributed training master information from environment.

    Reads MASTER_ADDR, MASTER_PORT, and TORCH_DISTRIBUTED_BACKEND
    from the current process environment.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - master_addr: Address of the master node
            - master_port: Port for master communication
            - backend: Distributed training backend

    Example:
        >>> info = get_local_master_info()
        >>> print(f"Master: {info['master_addr']}:{info['master_port']}")
    """
    return {
        "master_addr": os.environ.get("MASTER_ADDR", "localhost"),
        "master_port": int(os.environ.get("MASTER_PORT", 29500)),
        "backend": os.environ.get("TORCH_DISTRIBUTED_BACKEND", "nccl"),
    }
