from __future__ import annotations

import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".json", ".lock", ".yml", ".md", ".tex"}
WEB_SUFFIXES = {".css", ".htm", ".html", ".js", ".jsx", ".svelte", ".ts", ".tsx", ".vue"}
WEB_REPORT_PATTERN = re.compile(
    r"(?i)(?:\bnicegui\b|\bbuild_report\b|\bsrc\.reporting\b|"
    r"mortgage-credit-risk-study\.html|informe\s+html|html\s+aut[oó]nomo|<!doctype\s+html|<html\b)"
)


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
    def test_tfm_uses_proposed_index(self) -> None:
        path = ROOT / "tfm" / "main.tex"
        self.assertTrue(path.is_file())
        source = path.read_text(encoding="utf-8")
        self.assertEqual(
            [
                "Resumen ejecutivo",
                "Contexto, motivación y objetivos",
                "Fuentes, derechos de uso y privacidad",
                "Metodología y arquitectura",
                "EDA y preparación",
                "Modelo de probabilidad de incumplimiento",
                "Modelización de EAD, LGD y pérdida esperada",
                "Simulación de políticas de riesgo y escenarios macroeconómicos",
                "Interpretabilidad, gobierno y productivización",
                "Resultados, conclusiones y líneas futuras",
                "Bibliografía y anexos",
            ],
            re.findall(r"^\\section\{([^}]+)\}", source, flags=re.MULTILINE),
        )

    def test_required_entrypoints_exist(self) -> None:
        required = {
            ROOT / "README.md",
            ROOT / "scripts" / "run_study.py",
            ROOT / "tfm" / "main.tex",
            ROOT / ".github" / "workflows" / "run-study.yml",
        }
        self.assertEqual([], sorted(str(path.relative_to(ROOT)) for path in required if not path.is_file()))

    def test_repository_contains_only_delivery_files(self) -> None:
        blocked_suffixes = {".md", ".pdf", ".parquet", ".csv", ".zst", ".zip"}
        blocked_roots = {"data", "docs", "outputs", "models", ".private"}
        violations: list[str] = []
        for path in delivery_files():
            relative = path.relative_to(ROOT)
            if relative.parts[0] in blocked_roots or (
                path.is_file()
                and path.suffix.casefold() in blocked_suffixes
                and relative != Path("README.md")
            ):
                violations.append(str(relative))
        self.assertEqual([], violations)

    def test_repository_has_no_web_report_delivery(self) -> None:
        violations: list[str] = []
        for path in delivery_files():
            if not path.is_file() or path == Path(__file__):
                continue
            relative = path.relative_to(ROOT)
            if path.suffix.casefold() in WEB_SUFFIXES or (
                path.suffix.casefold() in TEXT_SUFFIXES
                and WEB_REPORT_PATTERN.search(path.read_text(encoding="utf-8", errors="ignore"))
            ):
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
