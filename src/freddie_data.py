from __future__ import annotations

import csv
import hashlib
import re
import zipfile
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from .data_access import write_csv_zst_parts
from .integrity import file_sha256, write_json_atomic
from .macro import fetch_official_macro_features, merge_macro


ORIGINATION_FIELDS = 31
PERFORMANCE_FIELDS = 35
CREDIT_ZERO_BALANCE_CODES = {"02", "03", "09", "15"}
QUARTER_PATTERN = re.compile(r"historical_data_(\d{4})Q([1-4])\.zip$")
ORIGIN_COLUMNS = {
    0: "fico",
    1: "first_payment",
    2: "first_time_home_buyer",
    5: "mi_percentage",
    6: "units",
    7: "occupancy",
    8: "cltv",
    9: "dti",
    10: "original_upb",
    11: "ltv",
    12: "interest_rate",
    15: "amortization",
    16: "state",
    17: "property_type",
    19: "loan_identifier",
    20: "loan_purpose",
    21: "loan_term",
    22: "borrowers",
    24: "high_balance",
}
PERFORMANCE_COLUMNS = {
    0: "loan_identifier",
    1: "period",
    2: "current_upb",
    3: "delinquency",
    6: "defect_settlement",
    8: "zero_balance_code",
    9: "zero_balance_date",
    13: "mi_recoveries",
    14: "net_sale_proceeds",
    15: "non_mi_recoveries",
    16: "expenses",
    21: "actual_loss",
    26: "zero_balance_removal_upb",
    27: "delinquent_accrued_interest",
}


def _entry(archive: zipfile.ZipFile, prefix: str) -> zipfile.ZipInfo:
    matches = [item for item in archive.infolist() if Path(item.filename).name.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"archive must contain exactly one {prefix} file")
    return matches[0]


def _validate_width(row: list[str], expected: int, source: str) -> None:
    if len(row) != expected:
        raise ValueError(f"{source} must contain {expected} fields, found {len(row)}")


def _number(value: object, *, missing: set[str] | None = None) -> float:
    text = str(value).strip()
    if not text or (missing and text in missing):
        return float("nan")
    return float(text)


def _stable_score(seed: int, loan_identifier: str) -> int:
    value = f"{seed}:{loan_identifier}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(value, digest_size=8).digest(), "big")


def _read_origin(archive: zipfile.ZipFile, entry: zipfile.ZipInfo) -> pd.DataFrame:
    with archive.open(entry) as source:
        frame = pd.read_csv(
            source,
            sep="|",
            header=None,
            names=list(range(ORIGINATION_FIELDS)),
            usecols=list(ORIGIN_COLUMNS),
            dtype=str,
            na_filter=False,
        ).rename(columns=ORIGIN_COLUMNS)
    if frame.empty:
        raise ValueError("origination file is empty")
    if frame["loan_identifier"].eq("").any() or frame["loan_identifier"].duplicated().any():
        raise ValueError("origination loan identifiers must be present and unique")
    return frame


def _selected_performance_chunks(
    source: Iterable[bytes], selected_ids: set[str], *, chunksize: int
) -> Iterator[pd.DataFrame]:
    if chunksize <= 0:
        raise ValueError("chunksize must be positive")
    positions = tuple(PERFORMANCE_COLUMNS)
    names = tuple(PERFORMANCE_COLUMNS.values())
    selected = {value.encode("utf-8") for value in selected_ids}
    rows: list[list[str]] = []
    for raw in source:
        line = raw.rstrip(b"\r\n")
        if line.count(b"|") != PERFORMANCE_FIELDS - 1:
            raise ValueError(f"performance row must contain {PERFORMANCE_FIELDS} fields")
        if line.partition(b"|")[0] not in selected:
            continue
        fields = line.split(b"|")
        rows.append([fields[position].decode("utf-8") for position in positions])
        if len(rows) == chunksize:
            yield pd.DataFrame(rows, columns=names)
            rows = []
    if rows:
        yield pd.DataFrame(rows, columns=names)


def _parse_month(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="%Y%m", errors="raise")


def _month_number(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype("int32")
    month = numeric % 100
    if not month.between(1, 12).all():
        raise ValueError("performance periods must use YYYYMM")
    return (numeric // 100) * 12 + month - 1


def _month_timestamp(value: int | None) -> pd.Timestamp | None:
    if value is None:
        return None
    return pd.Timestamp(year=value // 12, month=value % 12 + 1, day=1)


def prepare_quarter(
    zip_path: str | Path,
    *,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    path = Path(zip_path)
    match = QUARTER_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError("Freddie archive name must be historical_data_YYYYQn.zip")
    cohort_year = int(match.group(1))
    with zipfile.ZipFile(path) as archive:
        origin = _entry(archive, "orig_")
        performance = _entry(archive, "perf_")
        for entry, expected, label in (
            (origin, ORIGINATION_FIELDS, "origination row"),
            (performance, PERFORMANCE_FIELDS, "performance row"),
        ):
            with archive.open(entry) as raw:
                row = next(csv.reader((line.decode("utf-8") for line in raw), delimiter="|"), None)
            if row is None:
                raise ValueError(f"{label} file is empty")
            _validate_width(row, expected, label)
        origin_frame = _read_origin(archive, origin)
        origin_frame["_score"] = [
            _stable_score(seed, value) for value in origin_frame["loan_identifier"]
        ]
        selected = origin_frame.nsmallest(min(sample_size, len(origin_frame)), "_score").copy()
        selected_ids = set(selected["loan_identifier"])
        first_payment = dict(zip(selected["loan_identifier"], _parse_month(selected["first_payment"])))
        first_payment_number = {
            loan: value.year * 12 + value.month - 1 for loan, value in first_payment.items()
        }

        maximum_period: dict[str, int] = {}
        terminal_period: dict[str, int] = {}
        event_date: dict[str, int] = {}
        event_ead: dict[str, float] = {}
        loss_rows: dict[str, dict[str, object]] = {}
        performance_cutoff_value = ""

        with archive.open(performance) as source:
            chunks = _selected_performance_chunks(source, selected_ids, chunksize=100_000)
            for chunk in chunks:
                chunk_maximum = chunk["period"].max()
                if chunk_maximum > performance_cutoff_value:
                    performance_cutoff_value = chunk_maximum
                part = chunk.loc[chunk["loan_identifier"].isin(selected_ids)].copy()
                if part.empty:
                    continue
                part["performance_period"] = _month_number(part["period"])
                part["_months"] = (
                    part["performance_period"]
                    - part["loan_identifier"].map(first_payment_number).astype("int32")
                )

                for loan, value in part.groupby("loan_identifier")["performance_period"].max().items():
                    if loan not in maximum_period or value > maximum_period[loan]:
                        maximum_period[loan] = value

                within_horizon = part["_months"].between(0, 23, inclusive="both")
                delinquency = pd.to_numeric(part["delinquency"], errors="coerce").ge(3)
                credit_exit = part["zero_balance_code"].isin(CREDIT_ZERO_BALANCE_CODES)
                first_events = (
                    part.loc[within_horizon & (delinquency | credit_exit)]
                    .sort_values("performance_period")
                    .drop_duplicates("loan_identifier")
                )
                for row in first_events.itertuples(index=False):
                    loan = row.loan_identifier
                    period = row.performance_period
                    if loan in event_date and event_date[loan] <= period:
                        continue
                    balance = _number(row.current_upb)
                    removal = _number(row.zero_balance_removal_upb)
                    event_date[loan] = period
                    event_ead[loan] = (
                        balance
                        if np.isfinite(balance) and balance > 0
                        else removal if np.isfinite(removal) and removal > 0 else float("nan")
                    )

                terminals = part.loc[within_horizon & part["zero_balance_code"].ne("")]
                for loan, value in terminals.groupby("loan_identifier")["performance_period"].min().items():
                    if loan not in terminal_period or value < terminal_period[loan]:
                        terminal_period[loan] = value

                for row in part.loc[part["actual_loss"].ne("")].itertuples(index=False):
                    loss_rows[row.loan_identifier] = {
                        "defect_settlement": row.defect_settlement,
                        "zero_balance_code": row.zero_balance_code,
                        "zero_balance_date": row.zero_balance_date,
                        "mi_recoveries": row.mi_recoveries,
                        "net_sale_proceeds": row.net_sale_proceeds,
                        "non_mi_recoveries": row.non_mi_recoveries,
                        "expenses": row.expenses,
                        "actual_loss": row.actual_loss,
                        "zero_balance_removal_upb": row.zero_balance_removal_upb,
                        "delinquent_accrued_interest": row.delinquent_accrued_interest,
                    }

    if not performance_cutoff_value:
        raise ValueError("performance file is empty")
    performance_cutoff = pd.to_datetime(performance_cutoff_value, format="%Y%m", errors="raise")
    mature_loss_cutoff = performance_cutoff - pd.DateOffset(months=3)
    rows: list[dict[str, object]] = []
    censored = 0
    for origin_row in selected.itertuples(index=False):
        loan = origin_row.loan_identifier
        first = first_payment[loan]
        end_number = maximum_period.get(loan)
        end = _month_timestamp(end_number)
        observed_months = end_number - first_payment_number[loan] if end_number is not None else -1
        defaulted = loan in event_date
        eligible = defaulted or loan in terminal_period or observed_months >= 23
        if not eligible:
            censored += 1
            continue

        exposure = event_ead.get(loan, float("nan"))
        original_upb = _number(origin_row.original_upb)
        loss = loss_rows.get(loan)
        actual_loss = float("nan")
        lgd = float("nan")
        lgd_eligible = False
        zero_balance_date = pd.NaT
        if loss:
            components = [
                loss["mi_recoveries"],
                loss["net_sale_proceeds"],
                loss["non_mi_recoveries"],
                loss["expenses"],
                loss["zero_balance_removal_upb"],
                loss["delinquent_accrued_interest"],
            ]
            complete = all(str(value).strip() for value in components)
            zero_balance_date = pd.to_datetime(loss["zero_balance_date"], format="%Y%m", errors="coerce")
            actual_loss = _number(loss["actual_loss"])
            if complete:
                calculated = sum(_number(value) for value in components)
                if not np.isclose(calculated, actual_loss, atol=0.02, rtol=0):
                    raise ValueError(
                        f"actual loss does not reconcile for a selected {cohort_year}Q{match.group(2)} loan"
                    )
            lgd_eligible = bool(
                defaulted
                and complete
                and not loss["defect_settlement"]
                and loss["zero_balance_code"] in CREDIT_ZERO_BALANCE_CODES
                and pd.notna(zero_balance_date)
                and zero_balance_date <= mature_loss_cutoff
                and np.isfinite(exposure)
                and exposure > 0
            )
            if lgd_eligible:
                lgd = actual_loss / exposure

        mi = _number(origin_row.mi_percentage, missing={"999"})
        rows.append(
            {
                "origination_date": first - pd.DateOffset(months=1),
                "performance_end_date": end,
                "default_event_date": (
                    _month_timestamp(event_date[loan]) if loan in event_date else pd.NaT
                ),
                "zero_balance_date": zero_balance_date,
                "cohort_year": cohort_year,
                "original_interest_rate": _number(origin_row.interest_rate),
                "original_upb": original_upb,
                "original_loan_term": _number(origin_row.loan_term),
                "original_ltv": _number(origin_row.ltv, missing={"999"}),
                "original_cltv": _number(origin_row.cltv, missing={"999"}),
                "number_of_borrowers": _number(origin_row.borrowers, missing={"99"}),
                "original_dti": _number(origin_row.dti, missing={"999"}),
                "origination_fico": _number(origin_row.fico, missing={"9999"}),
                "mortgage_insurance_percentage": mi,
                "first_time_home_buyer": origin_row.first_time_home_buyer,
                "loan_purpose": origin_row.loan_purpose,
                "property_type": origin_row.property_type,
                "number_of_units": origin_row.units,
                "occupancy_status": origin_row.occupancy,
                "property_state": origin_row.state,
                "amortization_type": origin_row.amortization,
                "mortgage_insurance_type": (
                    "unknown" if not np.isfinite(mi) else "insured" if mi > 0 else "none"
                ),
                "high_balance_loan": "Y" if origin_row.high_balance == "Y" else "N",
                "unemployment_3m": float("nan"),
                "unemployment_change_12m": float("nan"),
                "hpi_yoy": float("nan"),
                "default_24m": int(defaulted),
                "ead_ratio": exposure / original_upb if defaulted and original_upb > 0 else float("nan"),
                "lgd": lgd,
                "lgd_eligible": lgd_eligible,
            }
        )

    result = pd.DataFrame(rows).sort_values(
        ["cohort_year", "origination_date", "property_state", "original_upb"],
        kind="stable",
        ignore_index=True,
    )
    metadata = {
        "archive": path.name,
        "population_rows": int(len(origin_frame)),
        "selected_rows": int(len(selected)),
        "eligible_rows": int(len(result)),
        "censored": int(censored),
        "defaults": int(result["default_24m"].sum()) if len(result) else 0,
        "lgd_eligible": int(result["lgd_eligible"].sum()) if len(result) else 0,
        "performance_cutoff": performance_cutoff.strftime("%Y-%m"),
    }
    return result, metadata


def prepare_dataset(
    raw_root: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    *,
    years: list[int],
    quarters: list[int],
    sample_size: int,
    seed: int,
    compression_level: int = 19,
    macro_sources: dict[str, str] | None = None,
    macro_cache: str | Path | None = None,
) -> dict[str, object]:
    if not years or not quarters or any(quarter not in {1, 2, 3, 4} for quarter in quarters):
        raise ValueError("years and calendar quarters are required")
    root = Path(raw_root)
    output = Path(output_path)
    details: list[dict[str, object]] = []
    columns: list[str] = []
    macro = None
    macro_details: list[dict[str, object]] = []
    if macro_sources:
        macro, macro_details = fetch_official_macro_features(
            macro_cache or output.parent / "macro",
            bls_url=macro_sources["unemployment"],
            hpi_url=macro_sources["house_prices"],
            start_year=min(years),
            end_year=max(years),
        )

    def frames():
        nonlocal columns
        for year in years:
            for quarter in quarters:
                archive = root / f"historical_data_{year}" / f"historical_data_{year}Q{quarter}.zip"
                if not archive.is_file():
                    raise FileNotFoundError(f"missing Freddie archive: {archive}")
                source = {
                    "name": str(archive.relative_to(root)).replace("\\", "/"),
                    "bytes": archive.stat().st_size,
                    "sha256": file_sha256(archive),
                }
                frame, metadata = prepare_quarter(archive, sample_size=sample_size, seed=seed)
                if macro is not None:
                    frame = frame.drop(
                        columns=["unemployment_3m", "unemployment_change_12m", "hpi_yoy"]
                    ).rename(columns={"property_state": "state"})
                    frame = merge_macro(frame, macro).rename(columns={"state": "property_state"})
                    frame = frame.drop(columns=["year", "month", "macro_asof_date"])
                if not columns:
                    columns = list(frame.columns)
                details.append({**source, **metadata})
                yield frame

    write_csv_zst_parts(frames(), output, level=compression_level)
    item = {
        "name": output.name,
        "bytes": output.stat().st_size,
        "rows": sum(int(item["eligible_rows"]) for item in details),
        "sha256": file_sha256(output),
        "columns": columns,
    }
    manifest = {
        "version": 1,
        "source": "Freddie Mac Single-Family Loan-Level Dataset",
        "seed": seed,
        "years": years,
        "quarters": quarters,
        "maximum_rows_per_quarter": sample_size,
        "population_rows": sum(int(item["population_rows"]) for item in details),
        "eligible_rows": item["rows"],
        "source_files": details,
        "macro_sources": macro_details,
        "files": [item],
    }
    write_json_atomic(manifest_path, manifest)
    return manifest
