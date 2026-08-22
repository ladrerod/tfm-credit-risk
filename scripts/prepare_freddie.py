from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the restricted Freddie analytical file.")
    parser.add_argument("--config", default="configs/data.json")
    parser.add_argument("--raw-root")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = args.raw_root or os.environ.get(config["dataset_directory_env"])
    if not root:
        parser.error(f"set --raw-root or {config['dataset_directory_env']}")
    output = Path(args.output or config["analysis_file"])
    from src.freddie_data import prepare_dataset

    manifest = prepare_dataset(
        root,
        output,
        output.with_name("manifest.json"),
        years=config["years"],
        quarters=config["quarters"],
        sample_size=config["maximum_rows_per_quarter"],
        seed=json.loads(Path("configs/model.json").read_text(encoding="utf-8"))["seed"],
        compression_level=config["compression_level"],
        macro_sources=config.get("macro_sources"),
        macro_cache=config.get("macro_cache"),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
