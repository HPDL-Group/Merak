import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from procguard.core import HealthChecker, HealthStatus


class TestHealthChecker(unittest.TestCase):
    # pylint: disable=protected-access

    def setUp(self):
        self.health_checker = HealthChecker(
            worker_configs={}, check_interval=0.1, heartbeat_timeout=10.0
        )

    def tearDown(self):
        if self.health_checker._running:
            self.health_checker.stop()

    def test_health_checker_initialization(self):
        self.assertIsNotNone(self.health_checker)
        self.assertFalse(self.health_checker._running)

    def test_register_worker(self):
        self.health_checker.register_worker("test_worker", {})
        report = self.health_checker.get_health_report("test_worker")
        self.assertIsNotNone(report)
        self.assertEqual(report.worker_id, "test_worker")

    def test_get_health_report_not_found(self):
        report = self.health_checker.get_health_report("nonexistent_worker")
        self.assertIsNone(report)

    def test_get_all_health_reports_empty(self):
        reports = self.health_checker.get_all_health_reports()
        self.assertEqual(len(reports), 0)

    def test_start_stop(self):
        self.health_checker.start()
        self.assertTrue(self.health_checker._running)
        self.health_checker.stop()
        self.assertFalse(self.health_checker._running)

    def test_clear_worker_manually_stopped(self):
        self.health_checker.mark_worker_manually_stopped("test_worker")
        self.assertTrue(self.health_checker.is_worker_manually_stopped("test_worker"))
        self.health_checker.clear_worker_manually_stopped("test_worker")
        self.assertFalse(self.health_checker.is_worker_manually_stopped("test_worker"))

    def test_mark_worker_manually_stopped(self):
        self.health_checker.mark_worker_manually_stopped("test_worker")
        self.assertTrue(self.health_checker.is_worker_manually_stopped("test_worker"))


if __name__ == "__main__":
    unittest.main()
