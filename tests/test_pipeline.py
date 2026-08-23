from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.data_access import write_csv_zst
from src.integrity import file_sha256, write_json_atomic
from src.pipeline import _implementation_sha256, _private_data, run_study


class PipelineTests(unittest.TestCase):
    def test_data_configuration_uses_only_freddie_inputs(self) -> None:
        config = json.loads((Path(__file__).parents[1] / "configs" / "data.json").read_text(encoding="utf-8"))
        serialized = json.dumps(config).casefold()
        self.assertIn("freddie", serialized)
        self.assertEqual([1], config["quarters"])

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

    def test_full_mode_loader_reads_the_restricted_freddie_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / "freddie-analysis.csv.zst"
            frame = pd.DataFrame({"cohort_year": [2015], "default_24m": [0]})
            write_csv_zst(frame, analysis, level=3)
            manifest = {
                "version": 1,
                "source": "Freddie Mac Single-Family Loan-Level Dataset",
                "population_rows": 2,
                "eligible_rows": 1,
                "files": [
                    {
                        "name": analysis.name,
                        "bytes": analysis.stat().st_size,
                        "rows": 1,
                        "sha256": file_sha256(analysis),
                        "columns": list(frame.columns),
                    }
                ],
            }
            write_json_atomic(root / "manifest.json", manifest)
            config = {
                "analysis_file_env": "FREDDIE_ANALYSIS_FILE",
                "analysis_file": "unused.csv.zst",
                "manifest": "unused.json",
                "chunk_rows": 10,
            }

            with patch.dict("os.environ", {"FREDDIE_ANALYSIS_FILE": str(analysis)}, clear=False):
                actual, identity = _private_data(config, seed=7)

            pd.testing.assert_frame_equal(frame, actual, check_dtype=False)
            self.assertEqual("restricted_freddie_dataset", identity["source"])
            self.assertEqual(2, identity["population_rows"])

    def test_full_mode_preparation_uses_explicit_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            analysis = root / "freddie-analysis.csv.zst"
            raw = root / "raw"
            raw.mkdir()
            frame = pd.DataFrame({"cohort_year": [2015], "default_24m": [0]})

            def prepare_dataset(_raw_root, output, manifest_path, *, seed, **_kwargs):
                write_csv_zst(frame, output, level=3)
                write_json_atomic(
                    manifest_path,
                    {
                        "version": 1,
                        "seed": seed,
                        "files": [
                            {
                                "name": Path(output).name,
                                "bytes": Path(output).stat().st_size,
                                "rows": 1,
                                "sha256": file_sha256(output),
                                "columns": list(frame.columns),
                            }
                        ],
                    },
                )

            config = {
                "analysis_file_env": "FREDDIE_ANALYSIS_FILE",
                "dataset_directory_env": "FREDDIE_DATASET_DIR",
                "analysis_file": analysis.name,
                "manifest": "manifest.json",
                "years": [2015],
                "quarters": [1],
                "maximum_rows_per_quarter": 10,
                "compression_level": 3,
                "macro_cache": "macro",
                "chunk_rows": 10,
            }
            with (
                patch("src.pipeline.ROOT", root),
                patch("src.pipeline.prepare_dataset", side_effect=prepare_dataset),
                patch.dict(
                    "os.environ",
                    {
                        "FREDDIE_ANALYSIS_FILE": str(analysis),
                        "FREDDIE_DATASET_DIR": str(raw),
                    },
                    clear=False,
                ),
            ):
                actual, identity = _private_data(config, seed=7)

            pd.testing.assert_frame_equal(frame, actual, check_dtype=False)
            self.assertEqual(7, identity["seed"])

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
            self.assertIn("ead_observed_tail", first["data_quality"])
            self.assertGreaterEqual(len(first["internal_bank_data_gaps"]), 5)
            if first["loss_components"]["decision_grade"]:
                self.assertLessEqual(
                    first["loss_components"]["ead"]["test_metrics"]["portfolio_relative_error"], 0.15
                )
                self.assertLessEqual(
                    first["loss_components"]["lgd"]["test_metrics"]["portfolio_relative_error"], 0.50
                )


if __name__ == "__main__":
    unittest.main()
