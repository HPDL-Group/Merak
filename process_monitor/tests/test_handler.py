import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from procguard.core import RecoveryOrchestrator, ClusterState, ProcessManager, RecoveryStage


class TestRecoveryOrchestrator(unittest.TestCase):
    def setUp(self):
        self.state_manager = ClusterState(
            state_file=":memory:", auto_save=False, decoupled_mode=True
        )
        self.process_manager = ProcessManager()
        self.recovery_orchestrator = RecoveryOrchestrator(
            state_manager=self.state_manager,
            process_manager=self.process_manager,
            recovery_config={
                "enable_auto_recovery": True,
                "stop_all_on_failure": True,
                "task_reassignment": True,
                "recovery_timeout": 60.0,
                "max_recovery_attempts": 3,
            },
        )

        self.state_manager.register_worker("test_worker")
        self.process_manager.register_worker_config(
            "test_worker",
            {"command": 'echo "test"', "working_dir": None, "env": {}, "restart_delay": 0.1},
        )

    def tearDown(self):
        self.process_manager.shutdown()

    def test_orchestrator_initialization(self):
        self.assertIsNotNone(self.recovery_orchestrator)
        stats = self.recovery_orchestrator.get_recovery_stats()
        self.assertIn("total_recoveries", stats)
        self.assertIn("successful_recoveries", stats)
        self.assertIn("failed_recoveries", stats)
        self.assertIn("success_rate", stats)

    def test_get_recovery_stats(self):
        stats = self.recovery_orchestrator.get_recovery_stats()
        self.assertEqual(stats["total_recoveries"], 0)
        self.assertEqual(stats["successful_recoveries"], 0)
        self.assertEqual(stats["failed_recoveries"], 0)
        self.assertEqual(stats["success_rate"], 0.0)

    def test_handle_failure_unknown_worker(self):
        results = self.recovery_orchestrator.handle_failure(["unknown_worker"])
        self.assertIn("unknown_worker", results)
        self.assertFalse(results["unknown_worker"])

    def test_get_recovery_history(self):
        history = self.recovery_orchestrator.get_recovery_history()
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 0)


if __name__ == "__main__":
    unittest.main()
