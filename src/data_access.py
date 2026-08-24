from __future__ import annotations

import io
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd
import zstandard as zstd

PRIVATE_COLUMNS = {
    "loan_identifier",
    "loan_sequence_number",
    "borrower_identifier",
    "borrower_name",
    "seller_name",
    "servicer_name",
}


def read_csv_zst(
    path: str | Path,
    *,
    chunksize: int,
) -> Iterator[pd.DataFrame]:
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    with Path(path).open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8", newline="") as text:
                rows = pd.read_csv(text, chunksize=chunksize)
                for frame in rows:
                    if any(
                        re.sub(r"\W+", "_", str(column).strip().casefold()).strip("_") in PRIVATE_COLUMNS
                        for column in frame.columns
                    ):
                        raise ValueError("prepared file contains private columns")
                    yield frame


def write_csv_zst_parts(
    frames: Iterable[pd.DataFrame], path: str | Path, *, level: int = 19
) -> None:
    if level <= 0:
        raise ValueError("compression level must be positive")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    try:
        with temporary.open("wb") as raw:
            with zstd.ZstdCompressor(level=level).stream_writer(raw) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                    columns: list[str] | None = None
                    wrote_rows = False
                    for frame in frames:
                        if frame.empty:
                            continue
                        normalized = [
                            re.sub(r"\W+", "_", str(column).strip().casefold()).strip("_")
                            for column in frame.columns
                        ]
                        if any(column in PRIVATE_COLUMNS for column in normalized):
                            raise ValueError("prepared file contains private columns")
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
        if temporary.exists():
            temporary.unlink()
