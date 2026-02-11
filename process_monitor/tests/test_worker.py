"""
Worker Launcher 测试模块

测试 worker_launcher.py 中的核心功能：
- SLURMEnvDetector: SLURM 环境检测
- WorkerLauncher: Worker 启动器核心功能
"""

import unittest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from procguard.worker_launcher import SLURMEnvDetector, WorkerLauncher


class TestSLURMEnvDetector(unittest.TestCase):
    """SLURM 环境检测器测试"""

    def test_detect_non_slurm_environment(self):
        """测试非 SLURM 环境检测"""
        env_backup = os.environ.copy()
        try:
            for key in ["SLURM_JOB_ID", "SLURM_PROCID", "SLURM_LOCALID", "SLURM_NTASKS"]:
                os.environ.pop(key, None)

            result = SLURMEnvDetector.detect()

            self.assertFalse(result["is_slurm"])
            self.assertIsNone(result["job_id"])
            self.assertEqual(result["local_rank"], 0)
            self.assertEqual(result["rank"], 0)
            self.assertEqual(result["world_size"], 1)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    def test_detect_slurm_environment(self):
        """测试 SLURM 环境检测"""
        env_backup = os.environ.copy()
        try:
            os.environ["SLURM_JOB_ID"] = "12345"
            os.environ["SLURM_JOB_NAME"] = "test_job"
            os.environ["SLURM_NODELIST"] = "gnode[01,02]"
            os.environ["SLURMD_NODENAME"] = "gnode01"
            os.environ["SLURM_PROCID"] = "1"
            os.environ["SLURM_LOCALID"] = "0"
            os.environ["SLURM_NTASKS"] = "4"

            result = SLURMEnvDetector.detect()

            self.assertTrue(result["is_slurm"])
            self.assertEqual(result["job_id"], "12345")
            self.assertEqual(result["job_name"], "test_job")
            self.assertEqual(result["rank"], 1)
            self.assertEqual(result["local_rank"], 0)
            self.assertEqual(result["world_size"], 4)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    def test_detect_slurm_environment_invalid_values(self):
        """测试 SLURM 环境检测（无效值）"""
        env_backup = os.environ.copy()
        try:
            os.environ["SLURM_JOB_ID"] = "12345"
            os.environ["SLURM_PROCID"] = "invalid"
            os.environ["SLURM_LOCALID"] = "invalid"
            os.environ["SLURM_NTASKS"] = "invalid"

            result = SLURMEnvDetector.detect()

            self.assertTrue(result["is_slurm"])
            self.assertEqual(result["rank"], 0)
            self.assertEqual(result["local_rank"], 0)
            self.assertEqual(result["world_size"], 1)
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    def test_generate_worker_id_slurm(self):
        """测试 SLURM 模式 worker_id 生成"""
        env_info = {
            "is_slurm": True,
            "hostname": "gnode01",
            "local_rank": 1,
        }

        worker_id = SLURMEnvDetector.generate_worker_id(env_info)

        self.assertEqual(worker_id, "gnode01-1")

    def test_generate_worker_id_non_slurm(self):
        """测试非 SLURM 模式 worker_id 生成"""
        env_info = {
            "is_slurm": False,
            "hostname": "localhost",
            "local_rank": 0,
        }

        worker_id = SLURMEnvDetector.generate_worker_id(env_info)

        self.assertEqual(worker_id, "localhost-0")

    def test_extract_master_node_simple(self):
        """测试提取主节点（简单节点列表）"""
        nodelist = "gnode01,gnode02,gnode03"

        result = SLURMEnvDetector.extract_master_node(nodelist)

        self.assertEqual(result, "gnode01")

    def test_extract_master_node_with_range(self):
        """测试提取主节点（带范围的节点列表）"""
        nodelist = "gnode[01-04]"

        result = SLURMEnvDetector.extract_master_node(nodelist)

        self.assertEqual(result, "gnode01")

    def test_extract_master_node_empty(self):
        """测试提取主节点（空列表）"""
        result = SLURMEnvDetector.extract_master_node("")

        self.assertEqual(result, "")

    def test_extract_master_node_single(self):
        """测试提取主节点（单节点）"""
        nodelist = "gnode01"

        result = SLURMEnvDetector.extract_master_node(nodelist)

        self.assertEqual(result, "gnode01")

    def test_build_pytorch_env_non_slurm(self):
        """测试构建 PyTorch 环境变量（非 SLURM）"""
        env_info = {
            "is_slurm": False,
            "hostname": "testhost",
            "local_rank": 0,
            "rank": 0,
        }

        result = SLURMEnvDetector.build_pytorch_env(env_info, gpu_count=2)

        self.assertEqual(result["LOCAL_RANK"], "0")
        self.assertEqual(result["RANK"], "0")
        self.assertEqual(result["WORLD_SIZE"], "2")
        self.assertEqual(result["MASTER_ADDR"], "testhost")
        self.assertEqual(result["GPU_COUNT"], "2")

    def test_build_pytorch_env_slurm(self):
        """测试构建 PyTorch 环境变量（SLURM）"""
        env_backup = os.environ.copy()
        try:
            for key in ["SLURM_NNODES", "SLURM_NTASKS_PER_NODE"]:
                os.environ.pop(key, None)

            env_info = {
                "is_slurm": True,
                "hostname": "gnode01",
                "local_rank": 1,
                "rank": 3,
                "nodelist": "gnode[01-04]",
            }

            result = SLURMEnvDetector.build_pytorch_env(env_info, gpu_count=2)

            self.assertEqual(result["LOCAL_RANK"], "1")
            self.assertEqual(result["RANK"], "3")
            self.assertEqual(result["WORLD_SIZE"], "2")
            self.assertEqual(result["GPU_COUNT"], "2")
        finally:
            os.environ.clear()
            os.environ.update(env_backup)

    def test_build_pytorch_env_slurm_with_slurm_vars(self):
        """测试构建 PyTorch 环境变量（SLURM 环境变量）"""
        env_backup = os.environ.copy()
        try:
            os.environ["SLURM_NNODES"] = "2"
            os.environ["SLURM_NTASKS_PER_NODE"] = "2"
            os.environ["MASTER_PORT"] = "29501"

            env_info = {
                "is_slurm": True,
                "hostname": "gnode01",
                "local_rank": 0,
                "rank": 0,
                "nodelist": "gnode01,gnode02",
            }

            result = SLURMEnvDetector.build_pytorch_env(env_info, gpu_count=2)

            self.assertEqual(result["WORLD_SIZE"], "4")
            self.assertEqual(result["MASTER_PORT"], "29501")
        finally:
            os.environ.clear()
            os.environ.update(env_backup)


class TestWorkerLauncher(unittest.TestCase):
    """WorkerLauncher 测试"""

    def setUp(self):
        """测试前准备"""
        self.mock_manager_url = "http://localhost:5000"
        self.mock_worker_id = "test-worker-0"
        self.mock_command = "echo 'test'"

    def tearDown(self):
        """测试后清理"""
        pass

    def test_worker_launcher_initialization(self):
        """测试 WorkerLauncher 初始化"""
        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )

        self.assertEqual(launcher.worker_id, self.mock_worker_id)
        self.assertEqual(launcher.manager_url, self.mock_manager_url)
        self.assertEqual(launcher.command, self.mock_command)
        self.assertEqual(launcher.heartbeat_interval, 5.0)
        self.assertEqual(launcher._status, "stopped")

    def test_worker_launcher_initialization_custom_params(self):
        """测试 WorkerLauncher 自定义参数初始化"""
        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
            working_dir="/tmp",
            env={"TEST_VAR": "test_value"},
            heartbeat_interval=10.0,
            slurm_mode=True,
            gpu_count=4,
            auto_start=True,
        )

        self.assertEqual(launcher.working_dir, "/tmp")
        self.assertEqual(launcher.env["TEST_VAR"], "test_value")
        self.assertEqual(launcher.heartbeat_interval, 10.0)
        self.assertTrue(launcher.slurm_mode)
        self.assertEqual(launcher.gpu_count, 4)
        self.assertTrue(launcher.auto_start)

    @patch("procguard.worker_launcher.requests.post")
    def test_register_success(self, mock_post):
        """测试 Worker 注册成功"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response

        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )

        result = launcher.register()

        self.assertTrue(result)
        mock_post.assert_called_once()

    @patch("procguard.worker_launcher.requests.post")
    def test_register_failure(self, mock_post):
        """测试 Worker 注册失败"""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_post.return_value = mock_response

        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )

        result = launcher.register()

        self.assertFalse(result)

    @patch("procguard.worker_launcher.requests.post")
    def test_register_connection_error(self, mock_post):
        """测试 Worker 注册连接错误"""
        import requests

        mock_post.side_effect = requests.RequestException("Connection refused")

        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )

        result = launcher.register()

        self.assertFalse(result)

    @patch("procguard.worker_launcher.requests.post")
    def test_send_heartbeat_success(self, mock_post):
        """测试发送心跳成功"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "command": None,
            "pytorch_env": {
                "RANK": "0",
                "LOCAL_RANK": "0",
                "WORLD_SIZE": "2",
            },
        }
        mock_post.return_value = mock_response

        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )
        launcher._process = Mock()
        launcher._process.pid = 12345

        result = launcher.send_heartbeat()

        self.assertTrue(result)
        self.assertEqual(launcher.env["RANK"], "0")
        self.assertEqual(launcher.env["LOCAL_RANK"], "0")

    @patch("procguard.worker_launcher.requests.post")
    def test_send_heartbeat_with_command(self, mock_post):
        """测试发送心跳并接收命令"""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "success": True,
            "command": "stop",
        }
        mock_post.return_value = mock_response

        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )
        launcher._process = Mock()
        launcher._process.pid = 12345

        with patch.object(launcher, "_handle_command") as mock_handle:
            result = launcher.send_heartbeat()

            self.assertTrue(result)
            mock_handle.assert_called_once_with("stop")

    @patch("procguard.worker_launcher.requests.post")
    def test_send_heartbeat_timeout(self, mock_post):
        """测试发送心跳超时"""
        import requests

        mock_post.side_effect = requests.exceptions.Timeout()

        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )
        launcher._process = Mock()
        launcher._process.pid = 12345

        result = launcher.send_heartbeat()

        self.assertFalse(result)

    @patch("procguard.worker_launcher.requests.post")
    def test_send_heartbeat_connection_error(self, mock_post):
        """测试发送心跳连接错误"""
        import requests

        mock_post.side_effect = requests.exceptions.ConnectionError()

        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )
        launcher._process = Mock()
        launcher._process.pid = 12345

        result = launcher.send_heartbeat()

        self.assertFalse(result)

    def test_handle_command_start(self):
        """测试处理 start 命令"""
        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )

        with patch.object(launcher, "_start_process") as mock_start:
            launcher._handle_command("start")
            mock_start.assert_called_once()

    def test_handle_command_stop(self):
        """测试处理 stop 命令"""
        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )

        with patch.object(launcher, "_stop_process") as mock_stop:
            launcher._handle_command("stop")
            mock_stop.assert_called_once_with(force=False)

    def test_handle_command_kill(self):
        """测试处理 kill 命令"""
        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )

        with patch.object(launcher, "_stop_process") as mock_stop:
            launcher._handle_command("kill")
            mock_stop.assert_called_once_with(force=True)

    def test_handle_command_restart(self):
        """测试处理 restart 命令"""
        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )

        with patch.object(launcher, "_restart_process") as mock_restart:
            launcher._handle_command("restart")
            mock_restart.assert_called_once()

    def test_get_config_hash(self):
        """测试配置哈希计算"""
        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )
        launcher.env = {
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
            "WORLD_SIZE": "4",
            "RANK": "0",
        }

        hash1 = launcher._get_config_hash()
        hash2 = launcher._get_config_hash()

        self.assertEqual(hash1, hash2)
        self.assertIsInstance(hash1, int)

        launcher.env["WORLD_SIZE"] = "8"
        hash3 = launcher._get_config_hash()

        self.assertNotEqual(hash1, hash3)

    def test_get_headers(self):
        """测试请求头生成"""
        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )

        headers = launcher._get_headers()

        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(headers["X-Worker-ID"], self.mock_worker_id)

    def test_shutdown(self):
        """测试关闭"""
        with patch("procguard.worker_launcher.threading.Event") as mock_event_class:
            mock_event = MagicMock()
            mock_event_class.return_value = mock_event

            launcher = WorkerLauncher(
                worker_id=self.mock_worker_id,
                manager_url=self.mock_manager_url,
                command=self.mock_command,
            )
            launcher._running = True

            launcher.shutdown()

            self.assertFalse(launcher._running)
            mock_event.set.assert_called_once()


class TestWorkerLauncherProcessManagement(unittest.TestCase):
    """WorkerLauncher 进程管理测试"""

    def setUp(self):
        """测试前准备"""
        self.mock_manager_url = "http://localhost:5000"
        self.mock_worker_id = "test-worker-0"
        self.mock_command = "echo 'test'"

    @patch("procguard.worker_launcher.subprocess.Popen")
    def test_start_process_success(self, mock_popen):
        """测试启动进程成功"""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )

        result = launcher._start_process()

        self.assertTrue(result)
        self.assertEqual(launcher._status, "running")
        mock_popen.assert_called_once()

    @patch("procguard.worker_launcher.subprocess.Popen")
    def test_start_process_already_running(self, mock_popen):
        """测试启动进程（已运行）"""
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process

        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )
        launcher._process = mock_process
        launcher._status = "running"

        result = launcher._start_process()

        self.assertTrue(result)
        mock_popen.assert_not_called()

    @patch("procguard.worker_launcher.PSUTIL_AVAILABLE", False)
    @patch("procguard.worker_launcher.os")
    def test_stop_process_force_false(self, mock_os):
        """测试停止进程（非强制）"""
        import signal as signal_module

        mock_os.killpg = Mock()
        mock_os.getpgid = Mock(return_value=12345)
        mock_os.setsid = Mock()

        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        launcher._process = mock_process
        launcher._status = "running"

        launcher._stop_process(force=False)

        mock_os.killpg.assert_called()
        self.assertEqual(launcher._status, "stopped")

    @unittest.skipIf(sys.platform == "win32", "信号处理在Windows上不可用")
    @patch("procguard.worker_launcher.PSUTIL_AVAILABLE", False)
    @patch("procguard.worker_launcher.os")
    @patch("procguard.worker_launcher.signal")
    def test_stop_process_force_true(self, mock_signal, mock_os):
        """测试强制停止进程"""
        mock_signal.SIGKILL = 9
        mock_signal.SIGTERM = 15
        mock_os.killpg = Mock()
        mock_os.getpgid = Mock(return_value=12345)
        mock_os.setsid = Mock()

        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )
        mock_process = Mock()
        mock_process.poll.return_value = None
        mock_process.pid = 12345
        launcher._process = mock_process
        launcher._status = "running"

        launcher._stop_process(force=True)

        mock_os.killpg.assert_called()
        self.assertEqual(launcher._status, "stopped")

    def test_stop_process_no_process(self):
        """测试停止进程（无进程）"""
        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )
        launcher._process = None
        launcher._status = "stopped"

        launcher._stop_process(force=False)

        self.assertEqual(launcher._status, "stopped")

    @unittest.skipIf(sys.platform == "win32", "信号处理在Windows上不可用")
    @patch("procguard.worker_launcher.PSUTIL_AVAILABLE", False)
    @patch("procguard.worker_launcher.os")
    @patch("procguard.worker_launcher.signal")
    @patch("procguard.worker_launcher.subprocess.Popen")
    def test_restart_process(self, mock_popen, mock_signal, mock_os):
        """测试重启进程"""
        mock_signal.SIGKILL = 9
        mock_signal.SIGTERM = 15
        mock_os.killpg = Mock()
        mock_os.getpgid = Mock(return_value=12345)
        mock_os.setsid = Mock()

        mock_process = Mock()
        mock_process.poll.return_value = 0
        mock_popen.return_value = mock_process

        launcher = WorkerLauncher(
            worker_id=self.mock_worker_id,
            manager_url=self.mock_manager_url,
            command=self.mock_command,
        )
        launcher._process = mock_process
        launcher._status = "running"
        launcher._restart_count = 0

        result = launcher._restart_process()

        self.assertTrue(result)
        self.assertEqual(launcher._restart_count, 1)


class TestWorkerLauncherEnvVars(unittest.TestCase):
    """WorkerLauncher 环境变量测试"""

    def test_env_update_from_pytorch_config(self):
        """测试从 PyTorch 配置更新环境变量"""
        launcher = WorkerLauncher(
            worker_id="test-worker",
            manager_url="http://localhost:5000",
            command="echo 'test'",
        )

        pytorch_env = {
            "MASTER_ADDR": "gnode01",
            "MASTER_PORT": "29500",
            "WORLD_SIZE": "4",
            "LOCAL_RANK": "0",
            "RANK": "0",
            "TORCH_DISTRIBUTED_BACKEND": "nccl",
        }

        launcher.env.update(pytorch_env)

        self.assertEqual(launcher.env["MASTER_ADDR"], "gnode01")
        self.assertEqual(launcher.env["WORLD_SIZE"], "4")
        self.assertEqual(launcher.env["TORCH_DISTRIBUTED_BACKEND"], "nccl")

    def test_env_merge_custom_env(self):
        """测试合并自定义环境变量"""
        launcher = WorkerLauncher(
            worker_id="test-worker",
            manager_url="http://localhost:5000",
            command="echo 'test'",
            env={"CUSTOM_VAR": "custom_value"},
        )

        self.assertEqual(launcher.env["CUSTOM_VAR"], "custom_value")


if __name__ == "__main__":
    unittest.main()
