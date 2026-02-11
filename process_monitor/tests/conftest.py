"""
Pytest 配置文件和共享 fixtures
"""

import sys
from pathlib import Path
import pytest


def pytest_configure(config):
    """Pytest 配置钩子"""
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_dir():
    """项目根目录"""
    return Path(__file__).parent


@pytest.fixture
def temp_dir(tmp_path):
    """临时目录"""
    return tmp_path


@pytest.fixture
def sample_worker_config():
    """示例工作节点配置"""
    return {"command": 'echo "test"', "working_dir": None, "env": {}, "restart_delay": 0.1}


@pytest.fixture
def sample_recovery_config():
    """示例恢复配置"""
    return {
        "enable_auto_recovery": True,
        "stop_all_on_failure": True,
        "task_reassignment": True,
        "recovery_timeout": 60.0,
        "max_recovery_attempts": 3,
    }
