from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the mortgage credit-risk study.")
    parser.add_argument("--mode", choices=("synthetic", "full"), default="synthetic")
    args = parser.parse_args()
    from src.pipeline import run_study
    from src.reporting import build_report

    artifact = "outputs/study-results.json"
    run_study(args.mode, output_path=artifact)
    build_report(artifact, "results/mortgage-credit-risk-study.html")


if __name__ == "__main__":
    main()
