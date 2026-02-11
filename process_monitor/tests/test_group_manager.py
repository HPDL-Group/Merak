#!/usr/bin/env python3
"""
GroupManager 单元测试

测试 procguard.core.group_manager 模块的功能
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from procguard.core.group_manager import GroupManager, GroupConfig


class TestGroupManagerBasic:
    """测试 GroupManager 基本功能"""

    @pytest.fixture
    def gm(self):
        """创建测试用的 GroupManager 实例"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        manager = GroupManager(state_file=state_file, auto_save=False)
        yield manager
        if os.path.exists(state_file):
            os.unlink(state_file)

    def test_initial_state(self, gm):
        """测试初始状态没有分组"""
        assert len(gm.get_all_groups()) == 0

    def test_create_group(self, gm):
        """测试创建分组"""
        gm.create_group("group_1", "测试分组1")
        assert "group_1" in gm.get_all_groups()

        group = gm.get_group("group_1")
        assert group.name == "测试分组1"
        assert len(group.workers) == 0

    def test_create_duplicate_group(self, gm):
        """测试创建重复分组"""
        gm.create_group("group_1", "分组1")
        result = gm.create_group("group_1", "分组1_重复")
        assert result is False
        groups = gm.get_all_groups()
        assert len(groups) == 1

    def test_add_workers_to_group(self, gm):
        """测试添加 worker 到分组"""
        gm.create_group("group_1", "测试分组1")
        gm.add_worker_to_group("group_1", "worker_1")
        gm.add_worker_to_group("group_1", "worker_2")

        group = gm.get_group("group_1")
        assert len(group.workers) == 2
        assert "worker_1" in group.workers
        assert "worker_2" in group.workers

    def test_add_worker_to_nonexistent_group(self, gm):
        """测试向不存在的分组添加 worker"""
        result = gm.add_worker_to_group("nonexistent", "worker_1")
        assert result is False

    def test_add_duplicate_worker(self, gm):
        """测试添加重复的 worker"""
        gm.create_group("group_1", "分组1")
        gm.add_worker_to_group("group_1", "worker_1")
        result = gm.add_worker_to_group("group_1", "worker_1")
        assert result is False

        group = gm.get_group("group_1")
        assert len(group.workers) == 1

    def test_get_worker_group(self, gm):
        """测试获取 worker 所属分组"""
        gm.create_group("group_1", "测试分组1")
        gm.add_worker_to_group("group_1", "worker_1")

        assert gm.get_worker_group("worker_1") == "group_1"
        assert gm.get_worker_group("worker_3") is None

    def test_get_group_of_worker(self, gm):
        """测试获取 worker 所属分组信息"""
        gm.create_group("group_1", "测试分组1")
        gm.add_worker_to_group("group_1", "worker_1")

        group = gm.get_group_of_worker("worker_1")
        assert group is not None
        assert group.name == "测试分组1"

    def test_get_group_of_nonexistent_worker(self, gm):
        """测试获取不存在的 worker 所属分组"""
        group = gm.get_group_of_worker("nonexistent")
        assert group is None

    def test_move_worker(self, gm):
        """测试移动 worker 到不同分组"""
        gm.create_group("group_1", "分组1")
        gm.create_group("group_2", "分组2")
        gm.add_worker_to_group("group_1", "worker_1")

        assert gm.get_worker_group("worker_1") == "group_1"

        gm.move_worker("worker_1", "group_2")
        assert gm.get_worker_group("worker_1") == "group_2"

        group_1 = gm.get_group("group_1")
        group_2 = gm.get_group("group_2")
        assert "worker_1" not in group_1.workers
        assert "worker_1" in group_2.workers

    def test_move_worker_to_same_group(self, gm):
        """测试移动 worker 到同一分组"""
        gm.create_group("group_1", "分组1")
        gm.add_worker_to_group("group_1", "worker_1")

        result = gm.move_worker("worker_1", "group_1")
        assert result is True
        assert gm.get_worker_group("worker_1") == "group_1"

    def test_move_nonexistent_worker_to_existing_group(self, gm):
        """测试移动不存在的 worker 到已存在的分组（会添加 worker）"""
        gm.create_group("group_1", "分组1")
        result = gm.move_worker("nonexistent", "group_1")
        assert result is True
        assert gm.get_worker_group("nonexistent") == "group_1"

    def test_move_worker_to_nonexistent_group(self, gm):
        """测试移动 worker 到不存在的分组"""
        gm.create_group("group_1", "分组1")
        gm.add_worker_to_group("group_1", "worker_1")
        result = gm.move_worker("worker_1", "nonexistent")
        assert result is False

    def test_remove_worker_from_group(self, gm):
        """测试从分组移除 worker"""
        gm.create_group("group_1", "分组1")
        gm.add_worker_to_group("group_1", "worker_1")
        gm.add_worker_to_group("group_1", "worker_2")

        gm.remove_worker_from_group("group_1", "worker_1")
        group = gm.get_group("group_1")
        assert "worker_1" not in group.workers
        assert "worker_2" in group.workers

    def test_remove_nonexistent_worker(self, gm):
        """测试移除不存在的 worker"""
        gm.create_group("group_1", "分组1")
        result = gm.remove_worker_from_group("group_1", "nonexistent")
        assert result is False

    def test_remove_worker_from_nonexistent_group(self, gm):
        """测试从不存在的分组移除 worker"""
        result = gm.remove_worker_from_group("nonexistent", "worker_1")
        assert result is False

    def test_delete_group(self, gm):
        """测试删除分组"""
        gm.create_group("group_1", "分组1")
        assert "group_1" in gm.get_all_groups()

        gm.delete_group("group_1")
        assert "group_1" not in gm.get_all_groups()

    def test_delete_nonexistent_group(self, gm):
        """测试删除不存在的分组"""
        result = gm.delete_group("nonexistent")
        assert result is False

    def test_rename_group(self, gm):
        """测试重命名分组"""
        gm.create_group("group_1", "原名称")
        gm.rename_group("group_1", "新名称")

        group = gm.get_group("group_1")
        assert group.name == "新名称"

    def test_rename_nonexistent_group(self, gm):
        """测试重命名不存在的分组"""
        result = gm.rename_group("nonexistent", "新名称")
        assert result is False


class TestGroupConfig:
    """测试 GroupConfig 功能"""

    @pytest.fixture
    def gm(self):
        """创建测试用的 GroupManager 实例"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        manager = GroupManager(state_file=state_file, auto_save=False)
        yield manager
        if os.path.exists(state_file):
            os.unlink(state_file)

    def test_update_and_get_config(self, gm):
        """测试更新和获取分组配置"""
        gm.create_group("group_1", "分组1")

        config = GroupConfig(
            master_addr="gn39-0",
            master_port=29500,
            world_size=4,
            backend="nccl",
            cuda_visible_devices="0,1",
            nccl_socket_ifname="eth0"
        )
        gm.update_group_config("group_1", config)

        saved_config = gm.get_group_config("group_1")
        assert saved_config.master_addr == "gn39-0"
        assert saved_config.master_port == 29500
        assert saved_config.world_size == 4
        assert saved_config.backend == "nccl"
        assert saved_config.cuda_visible_devices == "0,1"
        assert saved_config.nccl_socket_ifname == "eth0"

    def test_get_config_of_nonexistent_group(self, gm):
        """测试获取不存在的分组配置"""
        config = gm.get_group_config("nonexistent")
        assert config is None

    def test_config_replacement(self, gm):
        """测试配置完全替换"""
        gm.create_group("group_1", "分组1")

        config1 = GroupConfig(master_addr="gn39-0", world_size=4)
        gm.update_group_config("group_1", config1)

        config2 = GroupConfig(master_addr="gn39-1")
        gm.update_group_config("group_1", config2)

        saved_config = gm.get_group_config("group_1")
        assert saved_config.master_addr == "gn39-1"
        assert saved_config.world_size == 0
        assert saved_config.backend == "nccl"


class TestGroupInfoForDistMonitor:
    """测试分组信息格式"""

    @pytest.fixture
    def gm(self):
        """创建测试用的 GroupManager 实例"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        manager = GroupManager(state_file=state_file, auto_save=False)
        yield manager
        if os.path.exists(state_file):
            os.unlink(state_file)

    def test_get_group_info_for_dist_monitor(self, gm):
        """测试获取分组信息"""
        gm.create_group("test_group", "测试分组")
        gm.add_worker_to_group("test_group", "worker_1")
        gm.add_worker_to_group("test_group", "worker_2")

        group_info = gm.get_group_info_for_dist_monitor("test_group")
        assert group_info is not None
        assert group_info["group_id"] == "test_group"
        assert group_info["name"] == "测试分组"
        assert group_info["worker_count"] == 2
        assert len(group_info["workers"]) == 2
        assert "worker_1" in group_info["workers"]
        assert "worker_2" in group_info["workers"]
        assert group_info["config"]["master_addr"] is None
        assert group_info["config"]["world_size"] == 0

    def test_get_group_info_with_config(self, gm):
        """测试获取带配置的分组信息"""
        gm.create_group("test_group", "测试分组")
        gm.add_worker_to_group("test_group", "worker_1")

        config = GroupConfig(master_addr="gn39-0", world_size=4, backend="nccl")
        gm.update_group_config("test_group", config)

        group_info = gm.get_group_info_for_dist_monitor("test_group")
        assert group_info["config"]["master_addr"] == "gn39-0"
        assert group_info["config"]["world_size"] == 4
        assert group_info["config"]["backend"] == "nccl"

    def test_get_all_groups_for_dist_monitor(self, gm):
        """测试获取所有分组信息"""
        gm.create_group("group_1", "分组1")
        gm.add_worker_to_group("group_1", "worker_1")

        gm.create_group("group_2", "分组2")
        gm.add_worker_to_group("group_2", "worker_2")

        all_groups = gm.get_all_groups_for_dist_monitor()
        assert len(all_groups) == 2

        group_ids = [g["group_id"] for g in all_groups]
        assert "group_1" in group_ids
        assert "group_2" in group_ids

    def test_get_group_info_nonexistent(self, gm):
        """测试获取不存在的分组信息"""
        group_info = gm.get_group_info_for_dist_monitor("nonexistent")
        assert group_info is None

    def test_get_all_groups_for_dist_monitor_empty(self, gm):
        """测试空分组列表"""
        all_groups = gm.get_all_groups_for_dist_monitor()
        assert len(all_groups) == 0


class TestSyncFromFrontend:
    """测试从前端同步分组数据"""

    @pytest.fixture
    def gm(self):
        """创建测试用的 GroupManager 实例"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        manager = GroupManager(state_file=state_file, auto_save=False)
        yield manager
        if os.path.exists(state_file):
            os.unlink(state_file)

    def test_sync_from_frontend(self, gm):
        """测试从前端同步分组数据"""
        frontend_data = {
            "groups": {
                "frontend_group_1": {
                    "name": "前端分组1",
                    "workers": ["w1", "w2", "w3"]
                },
                "frontend_group_2": {
                    "name": "前端分组2",
                    "workers": ["w4", "w5"]
                }
            }
        }

        success = gm.sync_from_frontend(frontend_data)
        assert success is True

        assert "frontend_group_1" in gm.get_all_groups()
        assert "frontend_group_2" in gm.get_all_groups()

        group1 = gm.get_group("frontend_group_1")
        assert len(group1.workers) == 3
        assert "w1" in group1.workers
        assert "w2" in group1.workers
        assert "w3" in group1.workers

        group2 = gm.get_group("frontend_group_2")
        assert len(group2.workers) == 2
        assert "w4" in group2.workers
        assert "w5" in group2.workers

    def test_sync_with_config(self, gm):
        """测试同步带配置的分 groups 数据"""
        frontend_data = {
            "groups": {
                "test_group": {
                    "name": "测试分组",
                    "workers": ["w1", "w2"],
                    "config": {
                        "master_addr": "gn39-0",
                        "master_port": 29500,
                        "world_size": 4,
                        "backend": "nccl"
                    }
                }
            }
        }

        gm.sync_from_frontend(frontend_data)

        config = gm.get_group_config("test_group")
        assert config.master_addr == "gn39-0"
        assert config.world_size == 4

    def test_sync_empty_groups(self, gm):
        """测试同步空分组数据"""
        frontend_data = {"groups": {}}
        success = gm.sync_from_frontend(frontend_data)
        assert success is True
        assert len(gm.get_all_groups()) == 0

    def test_sync_updates_existing_group(self, gm):
        """测试同步更新已存在的分组"""
        gm.create_group("test_group", "原分组")
        gm.add_worker_to_group("test_group", "worker_1")

        frontend_data = {
            "groups": {
                "test_group": {
                    "name": "新分组名",
                    "workers": ["worker_2"]
                }
            }
        }

        gm.sync_from_frontend(frontend_data)

        group = gm.get_group("test_group")
        assert group.name == "新分组名"
        assert len(group.workers) == 1
        assert "worker_2" in group.workers
        assert "worker_1" not in group.workers


class TestStateSummary:
    """测试状态摘要"""

    @pytest.fixture
    def gm(self):
        """创建测试用的 GroupManager 实例"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        manager = GroupManager(state_file=state_file, auto_save=False)
        yield manager
        if os.path.exists(state_file):
            os.unlink(state_file)

    def test_state_summary(self, gm):
        """测试状态摘要"""
        gm.create_group("group_1", "分组1")
        gm.add_worker_to_group("group_1", "worker_1")

        gm.create_group("group_2", "分组2")
        gm.add_worker_to_group("group_2", "worker_2")
        gm.add_worker_to_group("group_2", "worker_3")

        summary = gm.get_state_summary()

        assert summary["total_groups"] == 2
        assert summary["total_workers_in_groups"] == 3
        assert "group_1" in summary["groups"]
        assert "group_2" in summary["groups"]

    def test_empty_state_summary(self, gm):
        """测试空状态摘要"""
        summary = gm.get_state_summary()

        assert summary["total_groups"] == 0
        assert summary["total_workers_in_groups"] == 0
        assert summary["groups"] == []


class TestGetAllGroups:
    """测试获取所有分组"""

    @pytest.fixture
    def gm(self):
        """创建测试用的 GroupManager 实例"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        manager = GroupManager(state_file=state_file, auto_save=False)
        yield manager
        if os.path.exists(state_file):
            os.unlink(state_file)

    def test_get_all_groups(self, gm):
        """测试获取所有分组"""
        gm.create_group("group_1", "分组1")
        gm.create_group("group_2", "分组2")
        gm.create_group("group_3", "分组3")

        all_groups = gm.get_all_groups()
        assert len(all_groups) == 3
        assert "group_1" in all_groups
        assert "group_2" in all_groups
        assert "group_3" in all_groups

    def test_get_all_groups_empty(self, gm):
        """测试获取空分组列表"""
        all_groups = gm.get_all_groups()
        assert len(all_groups) == 0
        assert all_groups == {}


class TestWorkerGroupLookup:
    """测试 worker 查找功能"""

    @pytest.fixture
    def gm(self):
        """创建测试用的 GroupManager 实例"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        manager = GroupManager(state_file=state_file, auto_save=False)
        yield manager
        if os.path.exists(state_file):
            os.unlink(state_file)

    def test_get_workers_in_group(self, gm):
        """测试获取分组中的所有 workers"""
        gm.create_group("group_1", "分组1")
        gm.add_worker_to_group("group_1", "worker_1")
        gm.add_worker_to_group("group_1", "worker_2")
        gm.add_worker_to_group("group_1", "worker_3")

        workers = gm.get_workers_in_group("group_1")
        assert set(workers) == {"worker_1", "worker_2", "worker_3"}

    def test_get_workers_in_nonexistent_group(self, gm):
        """测试获取不存在的分组中的 workers"""
        workers = gm.get_workers_in_group("nonexistent")
        assert workers == []

    def test_get_groups_for_workers(self, gm):
        """测试获取多个 worker 的所属分组"""
        gm.create_group("group_1", "分组1")
        gm.create_group("group_2", "分组2")

        gm.add_worker_to_group("group_1", "worker_1")
        gm.add_worker_to_group("group_1", "worker_2")
        gm.add_worker_to_group("group_2", "worker_3")

        assert gm.get_worker_group("worker_1") == "group_1"
        assert gm.get_worker_group("worker_2") == "group_1"
        assert gm.get_worker_group("worker_3") == "group_2"
        assert gm.get_worker_group("worker_4") is None


class TestMoveWorkers:
    """测试批量移动 workers"""

    @pytest.fixture
    def gm(self):
        """创建测试用的 GroupManager 实例"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        manager = GroupManager(state_file=state_file, auto_save=False)
        yield manager
        if os.path.exists(state_file):
            os.unlink(state_file)

    def test_move_multiple_workers(self, gm):
        """测试批量移动 workers"""
        gm.create_group("group_1", "分组1")
        gm.create_group("group_2", "分组2")

        gm.add_worker_to_group("group_1", "worker_1")
        gm.add_worker_to_group("group_1", "worker_2")
        gm.add_worker_to_group("group_1", "worker_3")

        results = gm.move_workers(["worker_1", "worker_2"], "group_2")

        assert results["worker_1"] is True
        assert results["worker_2"] is True
        assert gm.get_worker_group("worker_1") == "group_2"
        assert gm.get_worker_group("worker_2") == "group_2"

    def test_move_empty_workers_list(self, gm):
        """测试移动空的 workers 列表"""
        gm.create_group("group_1", "分组1")
        gm.create_group("group_2", "分组2")

        results = gm.move_workers([], "group_2")
        assert results == {}


class TestClearCache:
    """测试清除缓存功能"""

    @pytest.fixture
    def gm_with_state_file(self):
        """创建有状态文件的 GroupManager 实例"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            state_file = f.name
        manager = GroupManager(state_file=state_file, auto_save=False)
        yield manager, state_file
        if os.path.exists(state_file):
            os.unlink(state_file)

    def test_clear_cache(self, gm_with_state_file):
        """测试清除缓存"""
        gm, state_file = gm_with_state_file

        gm.create_group("group_1", "分组1")
        gm.add_worker_to_group("group_1", "worker_1")

        assert len(gm.get_all_groups()) == 1

        result = gm.clear_cache()
        assert result is True
        assert len(gm.get_all_groups()) == 0

    def test_clear_all(self, gm_with_state_file):
        """测试清除所有分组"""
        gm, state_file = gm_with_state_file

        gm.create_group("group_1", "分组1")
        gm.add_worker_to_group("group_1", "worker_1")

        gm.clear_all()
        assert len(gm.get_all_groups()) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
