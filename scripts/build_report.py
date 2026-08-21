from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the autonomous analytical report.")
    parser.add_argument("--input", default="outputs/study-results.json")
    parser.add_argument("--output", default="results/mortgage-credit-risk-study.html")
    args = parser.parse_args()
    from src.reporting import build_report

    build_report(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
