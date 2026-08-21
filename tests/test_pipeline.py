from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.pipeline import _implementation_sha256, run_study


class PipelineTests(unittest.TestCase):
    def test_implementation_identity_ignores_private_runtime_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "model.py").write_text("VALUE = 1\n", encoding="utf-8")
            with patch("src.pipeline.ROOT", root):
                expected = _implementation_sha256()
                for relative in (".private/runtime.py", "data/manifest.json", "outputs/run.json"):
                    target = root / relative
                    target.parent.mkdir(exist_ok=True)
                    target.write_text("{}\n", encoding="utf-8")
                self.assertEqual(expected, _implementation_sha256())

    def test_synthetic_run_produces_aggregate_reproducible_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "study-results.json"
            first = run_study("synthetic", output_path=output)
            second = run_study("synthetic", output_path=output)
            self.assertEqual(first, second)
            self.assertEqual(first, json.loads(output.read_text(encoding="utf-8")))
            self.assertFalse(first["contains_row_data"])
            self.assertEqual(24, first["methodology"]["target_horizon_months"])
            self.assertIn("pd", first)
            self.assertIn("loss_components", first)
            self.assertIn("expected_loss", first)
            self.assertIn("scenarios", first)
            self.assertIn("monitoring", first)
            if first["loss_components"]["decision_grade"]:
                self.assertLessEqual(
                    first["loss_components"]["ead"]["test_metrics"]["portfolio_relative_error"], 0.15
                )
                self.assertLessEqual(
                    first["loss_components"]["lgd"]["test_metrics"]["portfolio_relative_error"], 0.50
                )


if __name__ == "__main__":
    unittest.main()
