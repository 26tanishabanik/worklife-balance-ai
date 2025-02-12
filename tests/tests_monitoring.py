import unittest
from src.monitoring import Monitoring
import os


class TestMonitoring(unittest.TestCase):
    def setUp(self):
        self.monitoring = Monitoring()

    def test_log_metric(self):
        self.monitoring.log_metric("Test Metric", 10)
        self.assertEqual(len(self.monitoring.metrics), 1)
        self.assertEqual(self.monitoring.metrics[0]["Metric"], "Test Metric")
        self.assertEqual(self.monitoring.metrics[0]["Value"], 10)

    def test_generate_report(self):
        self.monitoring.log_metric("Metric A", 5)
        self.monitoring.generate_report(file_name="test_report.html")
        self.assertTrue(os.path.exists("test_report.html"))
        os.remove("test_report.html")  # Cleanup


if __name__ == "__main__":
    unittest.main()
