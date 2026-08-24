from __future__ import annotations

import csv
import hashlib
import heapq
import re
import zipfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd

from .data_access import read_csv_zst, write_csv_zst_parts
from .integrity import file_sha256, write_json_atomic


ORIGINATION_WIDTHS = {31, 32}
PERFORMANCE_WIDTHS = {32, 34, 35}
DEFAULT_CODES = {"02", "03", "09", "15"}
ABSORBING_STATES = {"default", "prepay", "other_exit"}
ORIGIN_COLUMNS = {
    0: "fico", 1: "first_payment", 2: "first_time_home_buyer", 3: "maturity_date", 5: "mi_percentage",
    6: "units", 7: "occupancy", 8: "cltv", 9: "dti", 10: "original_upb", 11: "ltv",
    12: "interest_rate", 15: "amortization", 16: "state", 17: "property_type",
    19: "loan_identifier", 20: "loan_purpose", 21: "loan_term", 22: "borrowers",
    24: "high_balance",
}
PERFORMANCE_COLUMNS = {
    0: "loan_identifier", 1: "period", 2: "current_upb", 3: "delinquency",
    4: "loan_age_months", 5: "remaining_months", 7: "modification_flag", 10: "current_interest_rate",
    6: "defect_settlement", 8: "zero_balance_code", 9: "zero_balance_date",
    13: "mi_recoveries", 14: "net_sale_proceeds", 15: "non_mi_recoveries",
    16: "expenses", 21: "actual_loss", 26: "zero_balance_removal_upb",
    27: "delinquent_accrued_interest",
}


def _member(archive: zipfile.ZipFile, kind: str) -> zipfile.ZipInfo:
    def is_match(name: str) -> bool:
        lowered = Path(name).name.casefold()
        if kind == "origin":
            return lowered.startswith("orig_") or (
                lowered.startswith("historical_data") and "_time_" not in lowered
            )
        return lowered.startswith("perf_") or "_time_" in lowered

    matches = [entry for entry in archive.infolist() if is_match(entry.filename)]
    if len(matches) != 1:
        raise ValueError(f"archive must contain exactly one {kind} file")
    return matches[0]


def _number(value: object, *, missing: set[str] | None = None) -> float:
    text = str(value).strip()
    return float("nan") if not text or (missing and text in missing) else float(text)


def _month_number(value: str) -> int:
    if not re.fullmatch(r"\d{6}", value):
        raise ValueError("performance periods must use YYYYMM")
    year, month = int(value[:4]), int(value[4:])
    if not 1 <= month <= 12:
        raise ValueError("performance periods must use YYYYMM")
    return year * 12 + month - 1


def _month_timestamp(value: int) -> pd.Timestamp:
    return pd.Timestamp(year=value // 12, month=value % 12 + 1, day=1)


def _loan_key(seed: int, loan_identifier: str) -> str:
    return hashlib.blake2b(f"{seed}:{loan_identifier}".encode(), digest_size=16).hexdigest()


def _state(delinquency: object, zero_balance_code: object) -> str:
    code, status = str(zero_balance_code).strip(), str(delinquency).strip().upper()
    if code == "01":
        return "prepay"
    if code in DEFAULT_CODES or status == "RA":
        return "default"
    if code in {"16", "96"}:
        return "other_exit"
    if status in {"", "XX"}:
        return "unknown"
    try:
        severity = int(status)
    except ValueError:
        return "unknown"
    return "current" if severity == 0 else "30" if severity == 1 else "60" if severity == 2 else "90_plus"


def _select_origin(
    archive: zipfile.ZipFile,
    entry: zipfile.ZipInfo,
    *,
    sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, int]:
    selected: list[tuple[int, int, list[str]]] = []
    identifiers: set[str] = set()
    with archive.open(entry) as source:
        for row_number, row in enumerate(
            csv.reader((line.decode("utf-8") for line in source), delimiter="|")
        ):
            if len(row) not in ORIGINATION_WIDTHS:
                raise ValueError("origination row must contain 31 or 32 fields")
            loan = row[19]
            if not loan or loan in identifiers:
                raise ValueError("origination loan identifiers must be present and unique")
            identifiers.add(loan)
            score = int.from_bytes(
                hashlib.blake2b(f"{seed}:{loan}".encode(), digest_size=8).digest(), "big"
            )
            values = [
                row[25] if name == "high_balance" and len(row) == 32 else row[position]
                for position, name in ORIGIN_COLUMNS.items()
            ]
            item = (-score, -row_number, values)
            if len(selected) < sample_size:
                heapq.heappush(selected, item)
            elif item[:2] > selected[0][:2]:
                heapq.heapreplace(selected, item)
    if not identifiers:
        raise ValueError("origination file is empty")
    rows = [item[2] for item in sorted(selected, key=lambda item: (-item[0], -item[1]))]
    return pd.DataFrame(rows, columns=ORIGIN_COLUMNS.values()), len(identifiers)


def _selected_performance_loans(
    source: Iterable[bytes], selected_ids: set[str]
) -> Iterator[tuple[str, pd.DataFrame]]:
    selected = {loan.encode("utf-8") for loan in selected_ids}
    closed: set[bytes] = set()
    active: bytes | None = None
    selected_active: bytes | None = None
    previous_period: int | None = None
    rows: list[list[object]] = []
    for raw in source:
        fields = raw.rstrip(b"\r\n").split(b"|")
        if len(fields) not in PERFORMANCE_WIDTHS:
            raise ValueError("performance row must contain 32, 34, or 35 fields")
        loan = fields[0]
        if loan != active:
            if selected_active is not None:
                closed.add(selected_active)
                yield selected_active.decode("utf-8"), pd.DataFrame(
                    rows, columns=[*PERFORMANCE_COLUMNS.values(), "performance_number", "current_state"]
                )
            if loan in closed:
                raise ValueError("performance rows for each selected loan must be contiguous")
            active = loan
            selected_active = loan if loan in selected else None
            previous_period = None
            rows = []
        if selected_active is None:
            continue
        values = [fields[position].decode("utf-8") for position in PERFORMANCE_COLUMNS]
        period = _month_number(values[1])
        if previous_period is not None and period <= previous_period:
            if period == previous_period:
                raise ValueError("performance periods must be unique per loan")
            raise ValueError("performance periods must be strictly increasing per loan")
        previous_period = period
        rows.append([*values, period, _state(values[3], values[9])])
    if selected_active is not None:
        yield selected_active.decode("utf-8"), pd.DataFrame(
            rows, columns=[*PERFORMANCE_COLUMNS.values(), "performance_number", "current_state"]
        )


def _performance_cutoff(source: Iterable[bytes]) -> int:
    cutoff: int | None = None
    for raw in source:
        fields = raw.rstrip(b"\r\n").split(b"|")
        if len(fields) not in PERFORMANCE_WIDTHS:
            raise ValueError("performance row must contain 32, 34, or 35 fields")
        period = _month_number(fields[1].decode("utf-8"))
        cutoff = period if cutoff is None else max(cutoff, period)
    if cutoff is None:
        raise ValueError("performance file is empty")
    return cutoff


def _origin_features(row: pd.Series) -> dict[str, object]:
    mi = _number(row.mi_percentage, missing={"999"})
    return {
        "origination_date": _month_timestamp(_month_number(row.first_payment) - 1),
        "cohort_year": 0,
        "original_interest_rate": _number(row.interest_rate),
        "original_upb": _number(row.original_upb),
        "original_loan_term": _number(row.loan_term),
        "original_ltv": _number(row.ltv, missing={"999"}),
        "original_cltv": _number(row.cltv, missing={"999"}),
        "number_of_borrowers": _number(row.borrowers, missing={"99"}),
        "original_dti": _number(row.dti, missing={"999"}),
        "origination_fico": _number(row.fico, missing={"9999"}),
        "mortgage_insurance_percentage": mi,
        "first_time_home_buyer": row.first_time_home_buyer,
        "loan_purpose": row.loan_purpose,
        "property_type": row.property_type,
        "number_of_units": row.units,
        "occupancy_status": row.occupancy,
        "property_state": row.state,
        "amortization_type": row.amortization,
        "mortgage_insurance_type": "unknown" if not np.isfinite(mi) else "insured" if mi > 0 else "none",
        "high_balance_loan": "Y" if row.high_balance == "Y" else "N",
        "unemployment_3m": float("nan"),
        "unemployment_change_12m": float("nan"),
        "hpi_yoy": float("nan"),
    }


def _loss_amount(loss: dict[str, object]) -> float | None:
    names = ("mi_recoveries", "net_sale_proceeds", "non_mi_recoveries", "expenses", "zero_balance_removal_upb", "delinquent_accrued_interest")
    if str(loss["net_sale_proceeds"]).strip().upper() == "U":
        return None
    if any(not str(loss[name]).strip() for name in names):
        return None
    values = {name: _number(loss[name]) for name in names}
    actual = _number(loss["actual_loss"])
    signed = sum(values.values())
    legacy = (
        values["zero_balance_removal_upb"] + values["delinquent_accrued_interest"]
        - values["mi_recoveries"] - values["net_sale_proceeds"] - values["non_mi_recoveries"]
        - values["expenses"]
    )
    if np.isclose(signed, actual, atol=0.02, rtol=0) or np.isclose(legacy, actual, atol=0.02, rtol=0):
        return actual
    raise ValueError("actual loss does not reconcile for a selected loan")


def _monthly_panel(
    records: pd.DataFrame,
    *,
    loan_key: str,
    first_number: int,
    features: dict[str, object],
) -> pd.DataFrame:
    panel = records.copy()
    panel["loan_key"] = loan_key
    panel["performance_date"] = panel["performance_number"].map(_month_timestamp)
    panel["origination_date"] = features["origination_date"]
    panel["months_since_first_payment"] = panel["performance_number"] - first_number
    panel["current_upb"] = pd.to_numeric(panel["current_upb"], errors="coerce")
    for name, value in features.items():
        if name not in panel:
            panel[name] = value
    panel["current_interest_rate"] = pd.to_numeric(panel["current_interest_rate"], errors="coerce")
    panel["loan_age_months"] = pd.to_numeric(panel["loan_age_months"], errors="coerce")
    panel["remaining_months"] = pd.to_numeric(panel["remaining_months"], errors="coerce")
    panel["monthly_ead_ratio"] = panel["current_upb"] / panel["original_upb"]
    panel["next_state"] = pd.NA
    panel["is_cure"] = False
    panel["is_rollback"] = False
    panel["consecutive_month"] = False
    severity = {"30": 1, "60": 2, "90_plus": 3}
    absorbed = False
    for index in range(len(panel) - 1):
        current, following = panel.iloc[index], panel.iloc[index + 1]
        if absorbed:
            continue
        if current.current_state in ABSORBING_STATES:
            absorbed = True
            continue
        if current.months_since_first_payment < 0:
            continue
        if following.performance_number != current.performance_number + 1:
            continue
        panel.at[index, "consecutive_month"] = True
        panel.at[index, "next_state"] = following.current_state
        if current.current_state in severity and following.current_state == "current":
            panel.at[index, "is_cure"] = True
        elif (
            current.current_state in severity
            and following.current_state in severity
            and severity[following.current_state] < severity[current.current_state]
        ):
            panel.at[index, "is_rollback"] = True
    return panel.drop(columns=["loan_identifier", "period", "performance_number"])


def _loan_result(
    records: pd.DataFrame,
    *,
    loan_key: str,
    first_number: int,
    features: dict[str, object],
    cutoff_date: pd.Timestamp,
) -> dict[str, object] | None:
    maximum = int(records["performance_number"].iloc[-1])
    within_horizon = records.loc[
        records["performance_number"].between(first_number, first_number + 23)
    ]
    events = within_horizon.loc[
        within_horizon["current_state"].isin(["90_plus", "default", "prepay", "other_exit"])
    ]
    first_event = events.iloc[0] if not events.empty else None
    if first_event is not None and str(first_event.current_state) == "other_exit":
        if str(first_event.zero_balance_code).strip() != "01":
            return None
    default_record = (
        first_event
        if first_event is not None and str(first_event.current_state) in {"90_plus", "default"}
        else None
    )
    if first_event is None and maximum - first_number < 23:
        return None

    exposure, ead_label_available_date = float("nan"), pd.NaT
    if default_record is not None:
        exposure = _number(default_record.current_upb)
        if np.isfinite(exposure) and exposure > 0:
            ead_label_available_date = _month_timestamp(int(default_record.performance_number))
        else:
            later_removal = records.loc[
                records["performance_number"].ge(int(default_record.performance_number))
            ].copy()
            later_removal["zero_balance_removal_upb"] = pd.to_numeric(
                later_removal["zero_balance_removal_upb"], errors="coerce"
            )
            later_removal = later_removal.loc[later_removal["zero_balance_removal_upb"].gt(0)]
            if not later_removal.empty:
                exposure = float(later_removal.iloc[0].zero_balance_removal_upb)
                ead_label_available_date = _month_timestamp(
                    int(later_removal.iloc[0].performance_number)
                )
    loss_rows = records.loc[records["actual_loss"].astype(str).str.strip().ne("")]
    loss = loss_rows.iloc[-1] if not loss_rows.empty else None
    has_defect = records["defect_settlement"].astype(str).str.strip().ne("").any()
    lgd, lgd_eligible, zero_balance_date = float("nan"), False, pd.NaT
    lgd_label_available_date = pd.NaT
    if loss is not None:
        zero_balance_date = pd.to_datetime(loss["zero_balance_date"], format="%Y%m", errors="coerce")
        amount = None if has_defect else _loss_amount(loss.to_dict())
        lgd_candidate = bool(
            default_record is not None
            and amount is not None
            and str(loss["zero_balance_code"]).strip() in DEFAULT_CODES
            and pd.notna(zero_balance_date)
            and np.isfinite(exposure)
            and exposure > 0
        )
        if lgd_candidate:
            lgd_label_available_date = max(
                zero_balance_date + pd.DateOffset(months=3),
                _month_timestamp(int(loss.performance_number)),
            )
        lgd_eligible = bool(
            lgd_candidate and lgd_label_available_date <= cutoff_date
        )
        if lgd_eligible:
            lgd = amount / exposure
    original_upb = float(features["original_upb"])
    return {
        **features,
        "loan_key": loan_key,
        "performance_end_date": _month_timestamp(maximum),
        "source_cutoff_date": cutoff_date,
        "pd_label_available_date": _month_timestamp(first_number + 23),
        "default_event_date": (
            _month_timestamp(int(default_record.performance_number))
            if default_record is not None
            else pd.NaT
        ),
        "ead_label_available_date": ead_label_available_date,
        "zero_balance_date": zero_balance_date,
        "lgd_label_available_date": lgd_label_available_date,
        "default_24m": int(default_record is not None),
        "ead_ratio": (
            exposure / original_upb
            if default_record is not None and original_upb > 0
            else float("nan")
        ),
        "lgd": lgd,
        "lgd_eligible": lgd_eligible,
    }


def prepare_quarter(
    zip_path: str | Path,
    panel_path: str | Path,
    *,
    sample_size: int,
    seed: int,
    compression_level: int = 19,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    path = Path(zip_path)
    match = re.search(r"(\d{4}).*?[Qq]([1-4])|[Qq]([1-4]).*?(\d{4})", path.name)
    if not match:
        raise ValueError("Freddie archive name must identify YYYYQn")
    cohort_year, cohort_quarter = (int(match.group(1)), int(match.group(2))) if match.group(1) else (int(match.group(4)), int(match.group(3)))
    results: list[dict[str, object]] = []
    processed_ids: set[str] = set()
    stats = {"censored": 0, "panel_rows": 0}
    with zipfile.ZipFile(path) as archive:
        origin_entry, performance_entry = _member(archive, "origin"), _member(archive, "performance")
        selected, population_rows = _select_origin(
            archive, origin_entry, sample_size=sample_size, seed=seed
        )
        selected_ids = set(selected["loan_identifier"])
        with archive.open(performance_entry) as source:
            cutoff = _performance_cutoff(source)
        cutoff_date = _month_timestamp(cutoff)
        first_numbers = {
            row.loan_identifier: _month_number(row.first_payment)
            for row in selected.itertuples()
        }
        maturity_numbers = {
            row.loan_identifier: (
                _month_number(row.maturity_date)
                if re.fullmatch(r"\d{6}", str(row.maturity_date).strip())
                else None
            )
            for row in selected.itertuples()
        }
        feature_by_loan = {
            row.loan_identifier: _origin_features(row) for _, row in selected.iterrows()
        }
        for features in feature_by_loan.values():
            features["cohort_year"] = cohort_year
        loan_keys = {loan: _loan_key(seed, loan) for loan in selected_ids}

        def panel_frames() -> Iterator[pd.DataFrame]:
            with archive.open(performance_entry) as source:
                for loan, records in _selected_performance_loans(source, selected_ids):
                    processed_ids.add(loan)
                    maturity = maturity_numbers[loan]
                    code_01 = records["zero_balance_code"].astype(str).str.strip().eq("01")
                    # ponytail: public original maturity is the only divider; use current contractual maturity if later available.
                    matured = (
                        code_01
                        if maturity is None
                        else code_01 & records["performance_number"].ge(maturity)
                    )
                    records.loc[matured, "current_state"] = "other_exit"
                    result = _loan_result(
                        records,
                        loan_key=loan_keys[loan],
                        first_number=first_numbers[loan],
                        features=feature_by_loan[loan],
                        cutoff_date=cutoff_date,
                    )
                    if result is None:
                        stats["censored"] += 1
                    else:
                        results.append(result)
                    panel = _monthly_panel(
                        records,
                        loan_key=loan_keys[loan],
                        first_number=first_numbers[loan],
                        features=feature_by_loan[loan],
                    )
                    stats["panel_rows"] += len(panel)
                    yield panel

        write_csv_zst_parts(panel_frames(), panel_path, level=compression_level)

    stats["censored"] += len(selected_ids - processed_ids)
    result = pd.DataFrame(results)
    if not result.empty:
        result = result.sort_values(["cohort_year", "origination_date", "property_state", "original_upb"], kind="stable", ignore_index=True)
    metadata = {
        "archive": path.name, "quarter": f"{cohort_year}Q{cohort_quarter}", "population_rows": population_rows,
        "selected_rows": int(len(selected)), "eligible_rows": int(len(result)), "censored": stats["censored"],
        "defaults": int(result["default_24m"].sum()) if not result.empty else 0,
        "lgd_eligible": int(result["lgd_eligible"].sum()) if not result.empty else 0,
        "performance_cutoff": cutoff_date.strftime("%Y-%m"), "panel_rows": stats["panel_rows"],
    }
    return result, metadata


def _archive_path(root: Path, year: int, quarter: int) -> Path:
    candidates = (
        root / f"historical_data_{year}" / f"historical_data_{year}Q{quarter}.zip",
        root / f"historical_data_{year}Q{quarter}.zip",
        root / f"historical_data_{year}" / f"historical_data1_Q{quarter}{year}.zip",
        root / f"historical_data1_Q{quarter}{year}.zip",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing Freddie archive for {year}Q{quarter}")


def prepare_dataset(
    raw_root: str | Path,
    analysis_path: str | Path,
    panel_dir: str | Path,
    manifest_path: str | Path,
    *,
    years: list[int],
    quarters: list[int],
    sample_size: int,
    seed: int,
    compression_level: int = 19,
    macro: object | None = None,
) -> dict[str, object]:
    if macro is not None:
        raise ValueError("macro enrichment is not supported by the restricted preparer")
    if not years or not quarters or any(quarter not in {1, 2, 3, 4} for quarter in quarters):
        raise ValueError("years and calendar quarters are required")
    root, output, panels = Path(raw_root), Path(analysis_path), Path(panel_dir)
    existing_panels = sorted(path.name for path in panels.glob("*.csv.zst")) if panels.is_dir() else []
    if existing_panels:
        raise ValueError(f"panel directory must be empty before preparation: {existing_panels}")
    details: list[dict[str, object]] = []
    panel_files: list[dict[str, object]] = []
    analysis_rows = 0
    analysis_columns: list[str] = []

    def analysis_frames() -> Iterator[pd.DataFrame]:
        nonlocal analysis_rows, analysis_columns
        for year in years:
            for quarter in quarters:
                archive = _archive_path(root, year, quarter)
                panel_path = panels / f"{year}Q{quarter}.csv.zst"
                frame, metadata = prepare_quarter(
                    archive,
                    panel_path,
                    sample_size=sample_size,
                    seed=seed,
                    compression_level=compression_level,
                )
                analysis_rows += len(frame)
                if not frame.empty and not analysis_columns:
                    analysis_columns = list(frame.columns)
                details.append({"name": str(archive.relative_to(root)).replace("\\", "/"), "bytes": archive.stat().st_size, "sha256": file_sha256(archive), **metadata})
                panel_columns = list(next(read_csv_zst(panel_path, chunksize=1)).columns)
                panel_files.append({"name": str(panel_path.relative_to(output.parent)).replace("\\", "/"), "bytes": panel_path.stat().st_size, "rows": metadata["panel_rows"], "sha256": file_sha256(panel_path), "columns": panel_columns})
                yield frame

    write_csv_zst_parts(analysis_frames(), output, level=compression_level)
    analysis = {"name": output.name, "bytes": output.stat().st_size, "rows": analysis_rows, "sha256": file_sha256(output), "columns": analysis_columns}
    manifest = {
        "version": 1, "source": "Freddie Mac Single-Family Loan-Level Dataset", "seed": seed,
        "years": years, "quarters": quarters, "maximum_rows_per_quarter": sample_size,
        "population_rows": sum(item["population_rows"] for item in details), "eligible_rows": analysis["rows"],
        "source_files": details, "files": [analysis, *panel_files],
    }
    write_json_atomic(manifest_path, manifest)
    return manifest
