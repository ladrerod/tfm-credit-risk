from __future__ import annotations

import unittest

from src.monitoring import build_alert, jensen_shannon, psi


class MonitoringTests(unittest.TestCase):
    def test_identical_distributions_have_no_drift(self) -> None:
        self.assertAlmostEqual(0.0, psi([1, 2, 3], [1, 2, 3]))
        self.assertAlmostEqual(0.0, jensen_shannon([1, 2, 3], [1, 2, 3]))

    def test_alert_thresholds_are_actionable(self) -> None:
        self.assertEqual("critical", build_alert("score", 0.30, warning=0.10, critical=0.25)["severity"])
        self.assertEqual("pending_labels", build_alert("score", None, warning=0.10, critical=0.25)["status"])


if __name__ == "__main__":
    unittest.main()
