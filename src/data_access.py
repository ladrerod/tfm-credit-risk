from __future__ import annotations

import csv
import io
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import zstandard as zstd

from .integrity import file_sha256

COMPACT_COLUMNS = (
    "cohort_year",
    "cohort_quarter",
    "origination_fico",
    "original_dti",
    "original_cltv",
    "original_interest_rate",
    "number_of_borrowers",
    "pd_label_available_date",
    "default_24m",
)
SOURCE_CUTOFF = pd.Timestamp("2026-03-01")


def read_csv_zst(path: str | Path, *, chunksize: int) -> Iterator[pd.DataFrame]:
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize <= 0:
        raise ValueError("chunksize must be positive")
    with Path(path).open("rb") as compressed:
        with zstd.ZstdDecompressor().stream_reader(compressed) as reader:
            with io.TextIOWrapper(reader, encoding="utf-8", newline="") as text:
                header = next(csv.reader([text.readline()]), [])
                if tuple(header) != COMPACT_COLUMNS:
                    raise ValueError("compact CSV header must match exactly")
                for frame in pd.read_csv(
                    text,
                    header=None,
                    names=COMPACT_COLUMNS,
                    chunksize=chunksize,
                    dtype={"default_24m": "string"},
                ):
                    yield frame


def load_compact(
    path: str | Path,
    expected_sha256: str,
    *,
    years: tuple[int, ...],
    chunksize: int = 100_000,
) -> tuple[pd.DataFrame, str]:
    observed = file_sha256(path)
    if observed != expected_sha256:
        raise ValueError("compact sha256 does not match expected value")
    if not years or any(type(year) is not int or not 2004 <= year <= 2023 for year in years):
        raise ValueError("requested years must be within 2004--2023")

    selected: list[pd.DataFrame] = []
    rows_per_quarter: dict[tuple[int, int], int] = {}
    requested = set(years)
    for chunk in read_csv_zst(path, chunksize=chunksize):
        cohort_year = pd.to_numeric(chunk["cohort_year"], errors="raise")
        if (
            cohort_year.isna().any()
            or not np.isfinite(cohort_year).all()
            or not (cohort_year % 1 == 0).all()
            or not cohort_year.between(2004, 2023).all()
        ):
            raise ValueError("cohort_year is invalid")
        frame = chunk.loc[cohort_year.isin(requested)].copy()
        if frame.empty:
            continue
        frame["cohort_year"] = cohort_year.loc[frame.index].astype(int)
        quarter = pd.to_numeric(frame["cohort_quarter"], errors="raise")
        if (
            quarter.isna().any()
            or not np.isfinite(quarter).all()
            or not (quarter % 1 == 0).all()
            or not quarter.between(1, 4).all()
        ):
            raise ValueError("cohort_quarter is invalid")
        frame["cohort_quarter"] = quarter.astype(int)
        for cohort, count in frame.groupby(["cohort_year", "cohort_quarter"], sort=False).size().items():
            rows_per_quarter[cohort] = rows_per_quarter.get(cohort, 0) + int(count)
            if rows_per_quarter[cohort] > 12_500:
                raise ValueError("compact exceeds the quarterly row cap")

        target = pd.to_numeric(frame["default_24m"], errors="raise")
        if target.isna().any() or not np.isfinite(target).all() or not target.isin([0, 1]).all():
            raise ValueError("default_24m must be binary")
        frame["default_24m"] = target.astype(int)
        available = pd.to_datetime(frame["pd_label_available_date"], errors="raise")
        if available.isna().any() or (available > SOURCE_CUTOFF).any():
            raise ValueError("default_24m is not mature")
        frame["pd_label_available_date"] = available
        for column, minimum, maximum in (
            ("origination_fico", 300, 850),
            ("original_dti", 0, 65),
            ("original_cltv", 0, 200),
            ("original_interest_rate", 0, 1000),
            ("number_of_borrowers", 1, 1000),
        ):
            values = pd.to_numeric(frame[column], errors="raise")
            present = values.notna()
            if not np.isfinite(values[present]).all() or not values[present].between(minimum, maximum).all():
                raise ValueError(f"{column} is outside its permitted range")
            frame[column] = values
        selected.append(frame)
    if not selected:
        raise ValueError("compact contains no requested rows")
    return pd.concat(selected, ignore_index=True), observed
