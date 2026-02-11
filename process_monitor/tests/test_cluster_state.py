import unittest
import sys
import tempfile
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from procguard.core import ClusterState, WorkerStatus, WorkerState, TaskAssignment


class TestClusterState(unittest.TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.state_file = os.path.join(self.temp_dir, "test_state.json")
        self.cluster_state = ClusterState(
            state_file=self.state_file, auto_save=False, decoupled_mode=True
        )

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            import shutil

            shutil.rmtree(self.temp_dir)

    def test_cluster_state_initialization(self):
        self.assertIsNotNone(self.cluster_state)
        self.assertEqual(len(self.cluster_state._workers), 0)

    def test_register_worker(self):
        self.cluster_state.register_worker("worker1")
        self.assertIn("worker1", self.cluster_state._workers)
        worker_state = self.cluster_state._workers["worker1"]
        self.assertEqual(worker_state.worker_id, "worker1")
        self.assertEqual(worker_state.status, WorkerStatus.UNKNOWN)

    def test_register_duplicate_worker(self):
        self.cluster_state.register_worker("worker1")
        self.cluster_state.register_worker("worker1")
        self.assertEqual(len(self.cluster_state._workers), 1)

    def test_update_worker_status(self):
        self.cluster_state.register_worker("worker1")
        self.cluster_state.update_worker_status("worker1", WorkerStatus.RUNNING)
        status = self.cluster_state.get_worker_status("worker1")
        self.assertEqual(status, WorkerStatus.RUNNING)

    def test_get_worker_status_not_found(self):
        status = self.cluster_state.get_worker_status("nonexistent")
        self.assertIsNone(status)

    def test_get_running_workers_empty(self):
        workers = self.cluster_state.get_running_workers()
        self.assertEqual(len(workers), 0)

    def test_get_running_workers_with_running_worker(self):
        self.cluster_state.register_worker("worker1")
        self.cluster_state.update_worker_status("worker1", WorkerStatus.RUNNING)
        workers = self.cluster_state.get_running_workers()
        self.assertIn("worker1", workers)

    def test_mark_worker_failed(self):
        self.cluster_state.register_worker("worker1")
        self.cluster_state.mark_worker_failed("worker1")
        self.assertIn("worker1", self.cluster_state._failed_workers)

    def test_clear_failed_workers(self):
        self.cluster_state.register_worker("worker1")
        self.cluster_state.mark_worker_failed("worker1")
        self.cluster_state.clear_failed_workers()
        self.assertEqual(len(self.cluster_state._failed_workers), 0)

    def test_get_state_summary(self):
        summary = self.cluster_state.get_state_summary()
        self.assertIn("total_workers", summary)
        self.assertIn("running_workers", summary)
        self.assertIn("failed_workers", summary)
        self.assertIn("total_tasks", summary)
        self.assertIn("queued_tasks", summary)
        self.assertEqual(summary["total_workers"], 0)

    def test_worker_state_defaults(self):
        self.cluster_state.register_worker("worker1")
        worker_state = self.cluster_state._workers["worker1"]
        self.assertIsNone(worker_state.pid)
        self.assertEqual(worker_state.status, WorkerStatus.UNKNOWN)
        self.assertIsNone(worker_state.start_time)
        self.assertIsNone(worker_state.last_heartbeat)
        self.assertEqual(worker_state.restart_count, 0)
        self.assertIsNone(worker_state.last_error)
        self.assertEqual(worker_state.assigned_tasks, [])


class TestTaskAssignment(unittest.TestCase):
    def test_task_assignment_creation(self):
        task = TaskAssignment(task_id="task1", worker_id="worker1", command="python train.py")
        self.assertEqual(task.task_id, "task1")
        self.assertEqual(task.worker_id, "worker1")
        self.assertEqual(task.command, "python train.py")
        self.assertEqual(task.status, "pending")
        self.assertIsNotNone(task.created_at)


if __name__ == "__main__":
    unittest.main()
