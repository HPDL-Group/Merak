import unittest
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from procguard.core import ProcessManager, WorkerStatus


class TestProcessManager(unittest.TestCase):
    def setUp(self):
        self.manager = ProcessManager()
        self.manager.register_worker_config(
            "test_worker",
            {"command": 'echo "test"', "working_dir": None, "env": {}, "restart_delay": 0.1},
        )

    def tearDown(self):
        self.manager.shutdown()

    def test_manager_initialization(self):
        self.assertIsNotNone(self.manager)
        self.assertEqual(len(self.manager.get_all_worker_info()), 0)

    def test_register_worker_config(self):
        self.manager.register_worker_config(
            "test_worker_2",
            {"command": 'echo "test2"', "working_dir": None, "env": {}, "restart_delay": 0.1},
        )
        self.assertIsNotNone(self.manager.get_worker_config("test_worker_2"))

    def test_get_worker_config_not_found(self):
        config = self.manager.get_worker_config("nonexistent_worker")
        self.assertIsNone(config)

    def test_start_stop_worker(self):
        success = self.manager.start_worker("test_worker")
        self.assertTrue(success)

        status = self.manager.get_worker_status("test_worker")
        self.assertEqual(status, WorkerStatus.RUNNING)

        self.manager.stop_worker("test_worker", force=True)
        status = self.manager.get_worker_status("test_worker")
        self.assertEqual(status, WorkerStatus.STOPPED)

    def test_start_unregistered_worker(self):
        success = self.manager.start_worker("unregistered_worker")
        self.assertFalse(success)

    def test_stop_unregistered_worker(self):
        success = self.manager.stop_worker("unregistered_worker")
        self.assertFalse(success)

    def test_get_all_worker_statuses_empty(self):
        statuses = self.manager.get_all_worker_statuses()
        self.assertEqual(len(statuses), 0)

    def test_get_running_workers_empty(self):
        workers = self.manager.get_running_workers()
        self.assertEqual(len(workers), 0)

    def test_restart_worker(self):
        self.manager.start_worker("test_worker")
        time.sleep(0.2)
        result = self.manager.restart_worker("test_worker", delay=0.0)
        self.assertTrue(result)
        status = self.manager.get_worker_status("test_worker")
        self.assertEqual(status, WorkerStatus.RUNNING)


if __name__ == "__main__":
    unittest.main()
