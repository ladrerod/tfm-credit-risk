from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_tfm_figures.py"
FIGURES = {
    "architecture_walk_forward.png",
    "correlation_heatmap.png",
    "drift.png",
    "event_rate.png",
    "expected_loss_cohort.png",
    "interpretability.png",
    "loss_components_error.png",
    "pd_diagnostics.png",
    "scenario_tradeoff.png",
}


def valid_study() -> dict[str, object]:
    return {
        "identity": {"source": "prepared_freddie_dataset", "rows": 8000},
        "backtesting": {
            "available": False,
            "reason": "regenerate compact with temporal lineage columns",
        },
        "data_quality": {
            "cohorts": [
                {"cohort": 2015, "rows": 4000, "events": 40, "event_rate": 0.01},
                {"cohort": 2016, "rows": 4000, "events": 80, "event_rate": 0.02},
            ]
        },
        "pd": {
            "test_metrics": {"n": 2000, "events": 30},
            "test_cohorts": [
                {"cohort_year": 2021, "n": 1000, "events": 10, "roc_auc": 0.76},
                {"cohort_year": 2022, "n": 1000, "events": 20, "roc_auc": 0.74},
            ],
            "test_calibration": [
                {
                    "bin": index,
                    "count": 200,
                    "events": index - 1,
                    "mean_probability": index / 100,
                    "event_rate": (index - 1) / 200,
                }
                for index in range(1, 11)
            ],
        },
        "loss_components": {
            "ead": {
                "selected_name": "hist_gradient_boosting",
                "test_metrics": {
                    "n": 300,
                    "portfolio_relative_error": 0.02,
                    "wape": 0.03,
                },
            },
            "lgd": {
                "selected_name": "hurdle",
                "test_metrics": {
                    "n": 60,
                    "portfolio_relative_error": 0.20,
                    "wape": 1.05,
                },
            },
        },
        "expected_loss": {
            "cohorts": [
                {
                    "cohort_year": 2021,
                    "n": 1000,
                    "exposure_at_default": 100_000_000,
                    "total_expected_loss": 400_000,
                    "expected_loss_rate": 0.004,
                },
                {
                    "cohort_year": 2022,
                    "n": 1000,
                    "exposure_at_default": 120_000_000,
                    "total_expected_loss": 720_000,
                    "expected_loss_rate": 0.006,
                },
            ]
        },
        "monitoring": {
            "feature_drift": [
                {"feature": "original_interest_rate", "psi": 1.2, "jensen_shannon": 0.2},
                {"feature": "original_upb", "psi": 0.18, "jensen_shannon": 0.05},
                {"feature": "original_cltv", "psi": 0.11, "jensen_shannon": 0.03},
            ]
        },
        "scenarios": [
            {
                "policy": policy,
                "macro_scenario": scenario,
                "n": 2000,
                "retained": retained,
                "retention_rate": retention,
                "retained_exposure": exposure,
                "retained_expected_loss": loss,
                "retained_expected_loss_rate": loss / exposure,
            }
            for policy, scenario, retained, retention, exposure, loss in (
                ("base", "observed", 1900, 0.95, 95_000_000, 380_000),
                ("base", "moderate_stress", 1700, 0.85, 85_000_000, 425_000),
                ("base", "severe_stress", 1300, 0.65, 65_000_000, 390_000),
                ("conservative", "observed", 1300, 0.65, 65_000_000, 195_000),
                ("conservative", "moderate_stress", 1050, 0.525, 52_500_000, 183_750),
                ("conservative", "severe_stress", 700, 0.35, 35_000_000, 140_000),
            )
        ],
        "governance": {
            "global_importance": [
                {
                    "feature": "origination_fico",
                    "importance": 0.00042,
                    "standard_deviation": 0.00003,
                },
                {
                    "feature": "original_dti",
                    "importance": 0.00015,
                    "standard_deviation": 0.00002,
                },
            ],
            "representative_sensitivity": [
                {
                    "feature": "origination_fico",
                    "p25": 725,
                    "p75": 788,
                    "probability_change": -0.0133,
                },
                {
                    "feature": "original_dti",
                    "p25": 27,
                    "p75": 42,
                    "probability_change": 0.0065,
                },
            ],
            "associations": {
                "rows": 4000,
                "features": ["original_ltv", "original_cltv", "origination_fico"],
                "pairs": [
                    {
                        "feature_a": "original_ltv",
                        "feature_b": "original_cltv",
                        "pairwise_rows": 3990,
                        "spearman": 0.97,
                    },
                    {
                        "feature_a": "original_ltv",
                        "feature_b": "origination_fico",
                        "pairwise_rows": 3995,
                        "spearman": -0.09,
                    },
                    {
                        "feature_a": "original_cltv",
                        "feature_b": "origination_fico",
                        "pairwise_rows": 3994,
                        "spearman": -0.10,
                    },
                ],
            },
        },
    }


class TfmFigureTests(unittest.TestCase):
    def test_cli_rejects_synthetic_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.json"
            source.write_text(
                json.dumps({"identity": {"source": "generated_in_memory"}}),
                encoding="utf-8",
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output-dir",
                    str(root / "figures"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("generated_in_memory", result.stderr)

    def test_cli_generates_nine_pngs_from_complete_freddie_aggregate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "freddie.json"
            output = root / "figures"
            source.write_text(json.dumps(valid_study()), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output.is_dir(), "el generador no creó el directorio de figuras")
            self.assertEqual({path.name for path in output.iterdir()}, FIGURES)
            for name in FIGURES:
                content = (output / name).read_bytes()
                self.assertGreater(len(content), 1000)
                self.assertEqual(content[:8], b"\x89PNG\r\n\x1a\n")

    def test_report_includes_new_matplotlib_figures(self) -> None:
        tex = (ROOT / "tfm" / "main.tex").read_text(encoding="utf-8")
        for name in {
            "correlation_heatmap.png",
            "interpretability.png",
            "scenario_tradeoff.png",
        }:
            with self.subTest(name=name):
                self.assertIn(f"{{{name}}}", tex)

    def test_cli_rejects_invalid_numeric_evidence_before_writing(self) -> None:
        study = valid_study()
        study["data_quality"]["cohorts"][0]["event_rate"] = "invalid"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "freddie.json"
            output = root / "figures"
            source.write_text(json.dumps(study), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("data_quality.cohorts[0].event_rate", result.stderr)
            self.assertFalse(output.exists())

    def test_cli_rejects_invalid_scenario_before_writing(self) -> None:
        study = valid_study()
        study["scenarios"][0]["retained_expected_loss_rate"] = "invalid"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "freddie.json"
            output = root / "figures"
            source.write_text(json.dumps(study), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("scenarios[0].retained_expected_loss_rate", result.stderr)
            self.assertFalse(output.exists())

    def test_cli_rejects_empty_calibration_bin_before_writing(self) -> None:
        study = valid_study()
        study["pd"]["test_calibration"][0]["count"] = 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "freddie.json"
            output = root / "figures"
            source.write_text(json.dumps(study), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("pd.test_calibration[0].count", result.stderr)
            self.assertFalse(output.exists())

    def test_cli_rejects_out_of_range_correlation_before_writing(self) -> None:
        study = valid_study()
        study["governance"]["associations"]["pairs"][0]["spearman"] = 1.2
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "freddie.json"
            output = root / "figures"
            source.write_text(json.dumps(study), encoding="utf-8")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--input",
                    str(source),
                    "--output-dir",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("governance.associations.pairs[0].spearman", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
