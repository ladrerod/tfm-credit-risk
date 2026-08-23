from __future__ import annotations

import io
import re
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import zstandard as zstd

PRIVATE_COLUMNS = {
    "loan_identifier",
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
