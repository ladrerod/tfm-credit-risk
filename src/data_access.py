from __future__ import annotations

import io
import json
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import zstandard as zstd

from .integrity import file_sha256


HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
PRIVATE_COLUMNS = {
    "loan_identifier",
    "borrower_identifier",
    "borrower_name",
    "seller_name",
    "servicer_name",
}


def load_manifest(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_manifest(payload)
    return payload


def validate_manifest(payload: Mapping[str, Any]) -> None:
    files = payload.get("files")
    if payload.get("version") != 1 or not isinstance(files, list) or not files:
        raise ValueError("manifest must contain a non-empty version 1 file list")
    names: set[str] = set()
    for item in files:
        name = item.get("name")
        columns = item.get("columns")
        if not isinstance(name, str) or not name.endswith(".csv.zst") or name in names:
            raise ValueError("manifest file names must be unique CSV.ZST paths")
        if not isinstance(columns, list) or not columns or any(str(value).casefold() in PRIVATE_COLUMNS for value in columns):
            raise ValueError("manifest contains missing or private columns")
        if not isinstance(item.get("rows"), int) or item["rows"] <= 0:
            raise ValueError("manifest rows must be positive integers")
        if not isinstance(item.get("bytes"), int) or item["bytes"] <= 0:
            raise ValueError("manifest bytes must be positive integers")
        if not isinstance(item.get("sha256"), str) or not HASH_PATTERN.fullmatch(item["sha256"]):
            raise ValueError("manifest hashes must be lowercase SHA-256 values")
        names.add(name)


def verify_file(path: str | Path, item: Mapping[str, Any]) -> None:
    target = Path(path)
    if target.stat().st_size != item["bytes"]:
        raise ValueError(f"byte count mismatch for {item['name']}")
    if file_sha256(target) != item["sha256"]:
        raise ValueError(f"hash mismatch for {item['name']}")


def read_csv_zst(
    path: str | Path,
    *,
    columns: list[str],
    chunksize: int,
    dtypes: Mapping[str, str] | None = None,
) -> Iterator[pd.DataFrame]:
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    with Path(path).open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8", newline="") as text:
                rows = pd.read_csv(text, chunksize=chunksize, dtype=dtypes)
                for frame in rows:
                    if list(frame.columns) != columns:
                        raise ValueError(f"schema mismatch for {Path(path).name}")
                    yield frame


def write_csv_zst_parts(frames: Iterable[pd.DataFrame], path: str | Path, *, level: int = 19) -> None:
    if level <= 0:
        raise ValueError("compression level must be positive")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".tmp", delete=False) as raw:
            temporary = Path(raw.name)
            with zstd.ZstdCompressor(level=level).stream_writer(raw, closefd=False) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    columns: list[str] | None = None
                    wrote_rows = False
                    for frame in frames:
                        if frame.empty:
                            continue
                        if columns is None:
                            columns = list(frame.columns)
                        elif list(frame.columns) != columns:
                            raise ValueError("all compressed CSV parts must have the same schema")
                        frame.to_csv(text, index=False, header=not wrote_rows)
                        wrote_rows = True
                    if not wrote_rows:
                        raise ValueError("compressed CSV requires at least one row")
        temporary.replace(target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def write_csv_zst(frame: pd.DataFrame, path: str | Path, *, level: int = 19) -> None:
    write_csv_zst_parts([frame], path, level=level)
