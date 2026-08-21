from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".json", ".lock", ".yml", ".html"}


def delivery_files() -> list[Path]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [ROOT / value for value in completed.stdout.splitlines() if value]


class DeliveryContractTests(unittest.TestCase):
    def test_required_entrypoints_exist(self) -> None:
        required = {
            ROOT / "scripts" / "run_study.py",
            ROOT / "scripts" / "build_report.py",
            ROOT / "results" / "mortgage-credit-risk-study.html",
            ROOT / ".github" / "workflows" / "run-study.yml",
        }
        self.assertEqual([], sorted(str(path.relative_to(ROOT)) for path in required if not path.is_file()))

    def test_repository_contains_only_delivery_files(self) -> None:
        blocked_suffixes = {".md", ".tex", ".pdf", ".parquet", ".csv", ".zst", ".zip"}
        blocked_roots = {"data", "docs", "outputs", "models", ".private"}
        violations: list[str] = []
        for path in delivery_files():
            relative = path.relative_to(ROOT)
            if relative.parts[0] in blocked_roots or (path.is_file() and path.suffix.casefold() in blocked_suffixes):
                violations.append(str(relative))
        self.assertEqual([], violations)

    def test_text_does_not_expose_internal_vocabulary(self) -> None:
        encoded = (
            (112, 104, 97, 115, 101),
            (102, 97, 115, 101),
            (99, 111, 100, 101, 120),
            (97, 103, 101, 110, 116),
            (97, 103, 101, 110, 116, 101),
            (115, 107, 105, 108, 108),
            (105, 97),
            (97, 105),
        )
        words = ["".join(map(chr, values)) for values in encoded]
        pattern = re.compile(r"(?i)\b(?:" + "|".join(map(re.escape, words)) + r")s?\b")
        violations: list[str] = []
        for path in delivery_files():
            if not path.is_file() or path.suffix.casefold() not in TEXT_SUFFIXES:
                continue
            if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                violations.append(str(path.relative_to(ROOT)))
        self.assertEqual([], violations)


if __name__ == "__main__":
    unittest.main()
