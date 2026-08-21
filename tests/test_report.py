from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.pipeline import run_study
from src.reporting import build_report


class ReportTests(unittest.TestCase):
    def test_builds_autonomous_research_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "results.json"
            output = root / "report.html"
            run_study("synthetic", output_path=artifact)
            build_report(artifact, output)
            html = output.read_text(encoding="utf-8")
            self.assertEqual(11, html.count('data-section="'))
            self.assertGreaterEqual(html.count("<svg"), 9)
            self.assertNotIn("<script src=", html)
            self.assertNotIn("<link rel=", html)
            self.assertNotIn("<img", html)
            self.assertIn("break-before: page", html)
            self.assertIn("he construido", html)
            self.assertIn("https://capitalmarkets.fanniemae.com/", html)
            self.assertNotIn("Lectura para decisión", html)


if __name__ == "__main__":
    unittest.main()
