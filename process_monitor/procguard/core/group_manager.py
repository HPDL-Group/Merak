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
Group Manager - Manages worker groups for PyTorch distributed training coordination.

This module provides functionality for organizing workers into groups that can
be used for PyTorch distributed training. Each group maintains configuration
for distributed training including master address, port, backend, and NCCL settings.

Features:
- Create and manage worker groups
- Add/remove/move workers between groups
- Configure PyTorch distributed training settings
- Synchronize group state from frontend
- Persist group configuration to disk

Example:
    >>> manager = GroupManager(state_file="groups.json")
    >>> manager.create_group("group_1", "Training Group 1")
    True
    >>> manager.add_worker_to_group("group_1", "worker_1")
    True
    >>> manager.get_group("group_1")
    GroupInfo(group_id='group_1', name='Training Group 1', ...)
"""

import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field


@dataclass
class GroupConfig:
    """
    Configuration settings for a worker group.

    Contains PyTorch distributed training configuration parameters that
    are applied to all workers in the group.

    Attributes:
        master_addr: Address of the master node for distributed training
        master_port: Port for master node communication
        world_size: Total number of processes in the distributed training
        backend: Distributed training backend (nccl, gloo, etc.)
        cuda_visible_devices: GPU devices visible to workers
        nccl_socket_ifname: Network interface for NCCL communication
    """
    master_addr: Optional[str] = None
    master_port: int = 29500
    world_size: int = 0
    backend: str = "nccl"
    cuda_visible_devices: Optional[str] = None
    nccl_socket_ifname: Optional[str] = None


@dataclass
class GroupInfo:
    """
    Information about a worker group.

    Contains group metadata, member workers, and configuration for
    distributed training coordination.

    Attributes:
        group_id: Unique identifier for the group
        name: Human-readable name for the group
        workers: List of worker IDs in this group
        config: GroupConfig with distributed training settings
        created_at: ISO timestamp when group was created
        updated_at: ISO timestamp when group was last updated
    """
    group_id: str
    name: str
    workers: List[str] = field(default_factory=list)
    config: GroupConfig = field(default_factory=GroupConfig)
    created_at: str = ""
    updated_at: str = ""

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        from datetime import datetime
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


class GroupManager:
    """
    Manages worker groups for distributed training coordination.

    This class provides comprehensive group management capabilities including
    creating/deleting groups, managing worker membership, and configuring
    distributed training parameters. Groups can be synchronized from external
    sources like a frontend web interface.

    Attributes:
        state_file: Path to JSON file for persisting group state
        auto_save: Whether to automatically save state changes
        _groups: Dictionary mapping group_id to GroupInfo
        _worker_to_group: Dictionary mapping worker_id to group_id

    Examples:
        >>> manager = GroupManager(state_file="groups.json", auto_save=True)
        >>> manager.create_group("group_1", "Training Group")
        True
        >>> manager.add_worker_to_group("group_1", "worker_1")
        True
        >>> manager.get_state_summary()
        {'total_groups': 1, 'total_workers_in_groups': 1, ...}
    """

    def __init__(
        self,
        state_file: str = "procguard_groups.json",
        auto_save: bool = True,
    ):
        """
        Initialize the GroupManager.

        Args:
            state_file: Path to JSON file for persisting group state
            auto_save: Whether to automatically save state after modifications
        """
        self.state_file = Path(state_file)
        self.auto_save = auto_save
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

        self._groups: Dict[str, GroupInfo] = {}
        self._worker_to_group: Dict[str, str] = {}

        self.logger.info("[GroupManager] Local cache reading disabled, group data will be synchronized from frontend")

    def _load_state(self):
        """Load group state from disk (currently disabled)."""
        self.logger.info("[GroupManager] Local cache reading disabled, skipping procguard_groups.json load")

    def save_state(self):
        """
        Save current group state to disk if auto_save is enabled.

        This method is a wrapper around _save_state that respects the
        auto_save configuration setting.
        """
        if self.auto_save:
            self._save_state()

    def _save_state(self):
        """
        Persist group state to JSON file.

        Saves all group information including workers and configuration
        to the state file with proper formatting.
        """
        try:
            data = {"groups": {}}

            for group_id, group_info in self._groups.items():
                data["groups"][group_id] = {
                    "group_id": group_id,
                    "name": group_info.name,
                    "workers": group_info.workers,
                    "config": {
                        "master_addr": group_info.config.master_addr,
                        "master_port": group_info.config.master_port,
                        "world_size": group_info.config.world_size,
                        "backend": group_info.config.backend,
                        "cuda_visible_devices": group_info.config.cuda_visible_devices,
                        "nccl_socket_ifname": group_info.config.nccl_socket_ifname,
                    },
                    "created_at": group_info.created_at,
                    "updated_at": group_info.updated_at,
                }

            self.state_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.logger.info(f"Saved {len(self._groups)} groups to {self.state_file}")
        except Exception as e:
            self.logger.error(f"Failed to save group state: {e}")

    def create_group(self, group_id: str, name: str) -> bool:
        """
        Create a new worker group.

        Args:
            group_id: Unique identifier for the new group
            name: Human-readable name for the group

        Returns:
            bool: True if group was created successfully, False if group already exists
        """
        with self._lock:
            if group_id in self._groups:
                self.logger.warning(f"[GroupManager] Failed to create group: {group_id} already exists")
                return False

            self._groups[group_id] = GroupInfo(
                group_id=group_id,
                name=name,
            )
            self._worker_to_group.clear()
            for gid, group in self._groups.items():
                for worker_id in group.workers:
                    self._worker_to_group[worker_id] = gid

            self.logger.info(f"[GroupManager] ✓ Successfully created group: {group_id} ({name})")
            self.logger.debug(f"[GroupManager] Current groups: {list(self._groups.keys())}")
            self.save_state()
            return True

    def delete_group(self, group_id: str) -> bool:
        """
        Delete a worker group and remove worker associations.

        Args:
            group_id: ID of the group to delete

        Returns:
            bool: True if group was deleted successfully, False if group doesn't exist
        """
        with self._lock:
            if group_id not in self._groups:
                self.logger.warning(f"[GroupManager] Failed to delete group: {group_id} does not exist")
                return False

            group = self._groups[group_id]
            worker_count = len(group.workers)
            for worker_id in group.workers:
                self._worker_to_group.pop(worker_id, None)

            del self._groups[group_id]
            self.logger.info(f"[GroupManager] ✓ Successfully deleted group: {group_id} (containing {worker_count} workers)")
            self.save_state()
            return True

    def rename_group(self, group_id: str, new_name: str) -> bool:
        """
        Rename an existing group.

        Args:
            group_id: ID of the group to rename
            new_name: New name for the group

        Returns:
            bool: True if renamed successfully, False if group doesn't exist
        """
        with self._lock:
            if group_id not in self._groups:
                self.logger.warning(f"Group {group_id} not found")
                return False

            self._groups[group_id].name = new_name
            from datetime import datetime
            self._groups[group_id].updated_at = datetime.now().isoformat()
            self.logger.info(f"Renamed group {group_id} to {new_name}")
            self.save_state()
            return True

    def add_worker_to_group(self, group_id: str, worker_id: str) -> bool:
        """
        Add a worker to a group.

        If the worker is already in another group, it will be removed
        from that group before being added to the new one.

        Args:
            group_id: ID of the target group
            worker_id: ID of the worker to add

        Returns:
            bool: True if worker was added successfully, False if group doesn't exist
            or worker is already in the group
        """
        with self._lock:
            if group_id not in self._groups:
                self.logger.warning(f"[GroupManager] Failed to add worker: group {group_id} does not exist")
                return False

            group = self._groups[group_id]
            if worker_id in group.workers:
                self.logger.debug(f"[GroupManager] Worker {worker_id} already in group {group_id}, skipping add")
                return False

            old_group_id = self._worker_to_group.get(worker_id)
            if worker_id in self._worker_to_group:
                if old_group_id and old_group_id in self._groups:
                    old_group = self._groups[old_group_id]
                    if worker_id in old_group.workers:
                        old_group.workers.remove(worker_id)
                        self.logger.debug(f"[GroupManager] Removed worker {worker_id} from original group {old_group_id}")

            group.workers.append(worker_id)
            self._worker_to_group[worker_id] = group_id

            from datetime import datetime
            group.updated_at = datetime.now().isoformat()

            self.logger.info(f"[GroupManager] ✓ Successfully added worker {worker_id} to group {group_id}")
            self.logger.debug(f"[GroupManager] Group {group_id} current workers: {group.workers}")
            self.save_state()
            return True

    def remove_worker_from_group(self, group_id: str, worker_id: str) -> bool:
        """
        Remove a worker from a group.

        Args:
            group_id: ID of the group
            worker_id: ID of the worker to remove

        Returns:
            bool: True if worker was removed successfully, False if group doesn't exist
            or worker is not in the group
        """
        with self._lock:
            if group_id not in self._groups:
                self.logger.warning(f"[GroupManager] Failed to remove worker: group {group_id} does not exist")
                return False

            group = self._groups[group_id]
            if worker_id not in group.workers:
                self.logger.debug(f"[GroupManager] Worker {worker_id} not in group {group_id}, skipping remove")
                return False

            group.workers.remove(worker_id)
            self._worker_to_group.pop(worker_id, None)

            from datetime import datetime
            group.updated_at = datetime.now().isoformat()

            self.logger.info(f"[GroupManager] ✓ Successfully removed worker {worker_id} from group {group_id}")
            self.logger.debug(f"[GroupManager] Group {group_id} current workers: {group.workers}")
            self.save_state()
            return True

    def move_worker(self, worker_id: str, target_group_id: str) -> bool:
        """
        Move a worker to a different group.

        Args:
            worker_id: ID of the worker to move
            target_group_id: ID of the target group

        Returns:
            bool: True if worker was moved successfully, False if target group doesn't exist
            or worker is already in the target group
        """
        with self._lock:
            if target_group_id not in self._groups:
                self.logger.warning(f"[GroupManager] Failed to move worker: target group {target_group_id} does not exist")
                return False

            old_group_id = self._worker_to_group.get(worker_id)
            
            if old_group_id == target_group_id:
                self.logger.debug(f"[GroupManager] Worker {worker_id} already in target group {target_group_id}, skipping move")
                return True

            if old_group_id and old_group_id in self._groups:
                old_group = self._groups[old_group_id]
                if worker_id in old_group.workers:
                    old_group.workers.remove(worker_id)
                    self.logger.debug(f"[GroupManager] Removed worker {worker_id} from original group {old_group_id}")

            self._worker_to_group[worker_id] = target_group_id
            target_group = self._groups[target_group_id]
            if worker_id not in target_group.workers:
                target_group.workers.append(worker_id)

            from datetime import datetime
            target_group.updated_at = datetime.now().isoformat()
            if old_group_id and old_group_id in self._groups:
                self._groups[old_group_id].updated_at = datetime.now().isoformat()

            self.logger.info(f"[GroupManager] ✓ Successfully moved worker {worker_id}: {old_group_id or 'no group'} → {target_group_id}")
            self.logger.debug(f"[GroupManager] Group {target_group_id} current workers: {target_group.workers}")
            if old_group_id and old_group_id in self._groups:
                self.logger.debug(f"[GroupManager] Group {old_group_id} current workers: {self._groups[old_group_id].workers}")
            self.save_state()
            return True

    def move_workers(self, worker_ids: List[str], target_group_id: str) -> Dict[str, bool]:
        """
        Move multiple workers to a target group.

        Args:
            worker_ids: List of worker IDs to move
            target_group_id: ID of the target group

        Returns:
            Dict[str, bool]: Dictionary mapping worker IDs to their move success status
        """
        results = {}
        for worker_id in worker_ids:
            results[worker_id] = self.move_worker(worker_id, target_group_id)
        return results

    def update_group_config(self, group_id: str, config: GroupConfig) -> bool:
        """
        Update the configuration for a group.

        Args:
            group_id: ID of the group to update
            config: New GroupConfig to apply

        Returns:
            bool: True if config was updated successfully, False if group doesn't exist
        """
        with self._lock:
            if group_id not in self._groups:
                self.logger.warning(f"[GroupManager] Failed to update config: group {group_id} does not exist")
                return False

            old_config = self._groups[group_id].config
            self._groups[group_id].config = config

            from datetime import datetime
            self._groups[group_id].updated_at = datetime.now().isoformat()

            self.logger.info(f"[GroupManager] ✓ Successfully updated group {group_id} config")
            self.logger.debug(f"[GroupManager] Group {group_id} config changed: master_addr={old_config.master_addr}→{config.master_addr}, master_port={old_config.master_port}→{config.master_port}, world_size={old_config.world_size}→{config.world_size}, backend={old_config.backend}→{config.backend}")
            self.save_state()
            return True

    def get_group(self, group_id: str) -> Optional[GroupInfo]:
        """
        Get group information by ID.

        Args:
            group_id: ID of the group to retrieve

        Returns:
            Optional[GroupInfo]: GroupInfo if found, None otherwise
        """
        with self._lock:
            return self._groups.get(group_id)

    def get_all_groups(self) -> Dict[str, GroupInfo]:
        """
        Get all groups.

        Returns:
            Dict[str, GroupInfo]: Dictionary mapping group_id to GroupInfo
        """
        with self._lock:
            return dict(self._groups)

    def get_worker_group(self, worker_id: str) -> Optional[str]:
        """
        Get the group ID that a worker belongs to.

        Args:
            worker_id: ID of the worker

        Returns:
            Optional[str]: Group ID if worker is in a group, None otherwise
        """
        with self._lock:
            return self._worker_to_group.get(worker_id)

    def get_group_of_worker(self, worker_id: str) -> Optional[GroupInfo]:
        """
        Get the full group information for a worker.

        Args:
            worker_id: ID of the worker

        Returns:
            Optional[GroupInfo]: GroupInfo if worker is in a group, None otherwise
        """
        group_id = self.get_worker_group(worker_id)
        if group_id:
            return self._groups.get(group_id)
        return None

    def get_workers_in_group(self, group_id: str) -> List[str]:
        """
        Get list of workers in a group.

        Args:
            group_id: ID of the group

        Returns:
            List[str]: List of worker IDs in the group, empty list if group doesn't exist
        """
        with self._lock:
            if group_id in self._groups:
                return list(self._groups[group_id].workers)
            return []

    def get_group_config(self, group_id: str) -> Optional[GroupConfig]:
        """
        Get configuration for a group.

        Args:
            group_id: ID of the group

        Returns:
            Optional[GroupConfig]: GroupConfig if group exists, None otherwise
        """
        with self._lock:
            if group_id in self._groups:
                return self._groups[group_id].config
            return None

    def get_group_info_for_dist_monitor(self, group_id: str) -> Optional[Dict[str, Any]]:
        """
        Get group information formatted for distribution monitor display.

        Args:
            group_id: ID of the group

        Returns:
            Optional[Dict[str, Any]]: Formatted group info for display, None if not found
        """
        group = self.get_group(group_id)
        if not group:
            return None

        return {
            "group_id": group.group_id,
            "name": group.name,
            "worker_count": len(group.workers),
            "workers": list(group.workers),
            "config": {
                "master_addr": group.config.master_addr,
                "master_port": group.config.master_port,
                "world_size": group.config.world_size,
                "backend": group.config.backend,
                "cuda_visible_devices": group.config.cuda_visible_devices,
                "nccl_socket_ifname": group.config.nccl_socket_ifname,
            }
        }

    def get_all_groups_for_dist_monitor(self) -> List[Dict[str, Any]]:
        """
        Get all groups formatted for distribution monitor display.

        Returns:
            List[Dict[str, Any]]: List of formatted group info for all groups
        """
        groups = []
        for group_id in self._groups:
            group_info = self.get_group_info_for_dist_monitor(group_id)
            if group_info:
                groups.append(group_info)
        return groups

    def sync_from_frontend(self, groups_data: Dict[str, Any]) -> bool:
        """
        Synchronize group state from frontend data.

        This method merges frontend data with existing groups, creating
        new groups as needed and updating worker memberships.

        Args:
            groups_data: Dictionary containing groups data from frontend

        Returns:
            bool: True if sync completed successfully, False on error
        """
        with self._lock:
            try:
                frontend_groups = groups_data.get("groups", {})
                self.logger.info(f"[GroupManager] Starting to sync frontend group data, {len(frontend_groups)} groups")

                worker_to_groups_count = {}
                worker_to_groups = {}
                for group_id, group_data in frontend_groups.items():
                    for worker_id in group_data.get("workers", []):
                        worker_to_groups_count[worker_id] = worker_to_groups_count.get(worker_id, 0) + 1
                        if worker_id not in worker_to_groups:
                            worker_to_groups[worker_id] = []
                        worker_to_groups[worker_id].append(group_id)

                duplicates = [w for w, c in worker_to_groups_count.items() if c > 1]
                if duplicates:
                    self.logger.warning(f"[GroupManager] Detected {len(duplicates)} workers appearing in multiple groups: {duplicates}")
                    for worker_id in duplicates:
                        groups_list = worker_to_groups[worker_id]
                        self.logger.warning(f"[GroupManager] Worker {worker_id} appears in groups: {groups_list}")
                        keep_group = groups_list[0]
                        remove_groups = groups_list[1:]
                        self.logger.info(f"[GroupManager] Keeping worker {worker_id} in group {keep_group}, removing from {remove_groups}")
                        for group_id in remove_groups:
                            if group_id in frontend_groups and worker_id in frontend_groups[group_id].get("workers", []):
                                frontend_groups[group_id]["workers"].remove(worker_id)
                                self.logger.info(f"[GroupManager] Removed duplicate worker {worker_id} from group {group_id}")

                for group_id, group_data in frontend_groups.items():
                    if group_id not in self._groups:
                        self.create_group(group_id, group_data.get("name", group_id))
                        self.logger.debug(f"[GroupManager] Created new group: {group_id}")

                    group = self._groups[group_id]
                    new_workers = group_data.get("workers", [])

                    current_workers = set(group.workers)
                    new_workers_set = set(new_workers)

                    workers_to_add = new_workers_set - current_workers
                    workers_to_remove = current_workers - new_workers_set

                    self.logger.debug(f"[GroupManager] Group {group_id}: need to add {len(workers_to_add)} workers, need to remove {len(workers_to_remove)} workers")

                    for worker_id in workers_to_remove:
                        if self._worker_to_group.get(worker_id) == group_id:
                            self._worker_to_group.pop(worker_id, None)
                            self.logger.debug(f"[GroupManager] Removed worker from group {group_id}: {worker_id}")

                    group.workers = list(new_workers_set)
                    for worker_id in new_workers_set:
                        old_group = self._worker_to_group.get(worker_id)
                        if old_group and old_group != group_id:
                            self.logger.warning(f"[GroupManager] Worker {worker_id} moved from group {old_group} to {group_id}")
                        self._worker_to_group[worker_id] = group_id

                    name = group_data.get("name")
                    if name and name != group.name:
                        group.name = name
                        self.logger.debug(f"[GroupManager] Updated group {group_id} name to: {name}")

                    config_data = group_data.get("pytorch_config")
                    if config_data:
                        new_config = GroupConfig(
                            master_addr=config_data.get("master_addr"),
                            master_port=config_data.get("master_port", 29500),
                            world_size=config_data.get("world_size", 0),
                            backend=config_data.get("backend", "nccl"),
                            cuda_visible_devices=config_data.get("cuda_visible_devices"),
                            nccl_socket_ifname=config_data.get("nccl_socket_ifname"),
                        )
                        group.config = new_config
                        self.logger.debug(f"[GroupManager] Updated group {group_id} config: master_addr={new_config.master_addr}, world_size={new_config.world_size}")

                    from datetime import datetime
                    group.updated_at = datetime.now().isoformat()

                self.logger.info(f"[GroupManager] ✓ Successfully synced {len(frontend_groups)} groups")
                self.logger.debug(f"[GroupManager] Post-sync state: {self.get_state_summary()}")
                self.save_state()
                return True
            except Exception as e:
                self.logger.error(f"[GroupManager] ✗ Failed to sync frontend data: {e}")
                return False

    def get_state_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current group state.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - total_groups: Number of groups
                - total_workers_in_groups: Total workers across all groups
                - groups: List of group IDs
        """
        with self._lock:
            total_groups = len(self._groups)
            total_workers = sum(len(g.workers) for g in self._groups.values())

            return {
                "total_groups": total_groups,
                "total_workers_in_groups": total_workers,
                "groups": list(self._groups.keys()),
            }

    def clear_all(self):
        """
        Remove all groups and worker associations.

        This clears all group data and worker mappings, then saves the
        empty state to disk if auto_save is enabled.
        """
        with self._lock:
            self._groups.clear()
            self._worker_to_group.clear()
            self.logger.info("[GroupManager] Cleared all group data")
            self.save_state()

    def clear_cache(self) -> bool:
        """
        Clear all group data and remove the cache file.

        Returns:
            bool: True if cache was cleared successfully, False on error
        """
        try:
            with self._lock:
                self._groups.clear()
                self._worker_to_group.clear()
                
                if self.state_file.exists():
                    self.state_file.unlink()
                    self.logger.info(f"[GroupManager] ✓ Cache file cleared: {self.state_file}")
                else:
                    self.logger.debug(f"[GroupManager] Cache file does not exist, no need to clear: {self.state_file}")
                
                return True
        except Exception as e:
            self.logger.error(f"[GroupManager] ✗ Failed to clear cache: {e}")
            return False
