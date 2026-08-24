from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.freddie_data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare restricted Freddie analysis and monthly panel files.")
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--analysis-path", default="freddie-analysis.csv.zst")
    parser.add_argument("--panel-dir", default="freddie-monthly")
    parser.add_argument("--manifest-path", default=".private/freddie/manifest.json")
    parser.add_argument("--years", nargs="+", type=int, required=True)
    parser.add_argument("--quarters", nargs="+", type=int, default=[1])
    parser.add_argument("--sample-size", type=int, default=50_000)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--compression-level", type=int, default=19)
    args = parser.parse_args()
    print(json.dumps(prepare_dataset(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
