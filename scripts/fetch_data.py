from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from src.data_access import fetch_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the private study dataset.")
    parser.add_argument("--config", default="configs/data.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    repo_id = os.environ.get(config["dataset_repository_env"], "")
    manifest, paths = fetch_dataset(
        repo_id,
        config["cache_directory"],
        revision=config["dataset_revision"],
        manifest_name=config["manifest"],
    )
    print(json.dumps({"files": len(paths), "rows": sum(item["rows"] for item in manifest["files"])}))


if __name__ == "__main__":
    main()
