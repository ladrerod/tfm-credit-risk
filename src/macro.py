from __future__ import annotations

import calendar
import json
from datetime import date
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

from .integrity import file_sha256


def _quarter_end(year: int, quarter: int) -> date:
    month = quarter * 3
    return date(year, month, calendar.monthrange(year, month)[1])


def build_state_month_features(
    laus: pd.DataFrame,
    hpi: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    if start_year > end_year:
        raise ValueError("invalid macro year range")
    laus = laus.copy()
    laus["date"] = pd.to_datetime(laus["date"])
    if laus.duplicated(["state", "date"]).any() or hpi.duplicated(["state", "year", "quarter"]).any():
        raise ValueError("macro source keys must be unique")
    unemployment = {
        (row.state, row.date.year, row.date.month): float(row.unemployment)
        for row in laus.itertuples()
    }
    prices = {
        (row.state, int(row.year), int(row.quarter)): float(row.hpi)
        for row in hpi.itertuples()
    }
    rows: list[dict[str, object]] = []
    for state in sorted(set(laus["state"]).union(hpi["state"])):
        for origin in pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS"):
            lag_end = origin - pd.DateOffset(months=2)
            recent_dates = [lag_end - pd.DateOffset(months=value) for value in (2, 1, 0)]
            prior_dates = [value - pd.DateOffset(months=12) for value in recent_dates]
            recent = [unemployment.get((state, value.year, value.month)) for value in recent_dates]
            prior = [unemployment.get((state, value.year, value.month)) for value in prior_dates]
            recent_mean = sum(recent) / 3 if all(value is not None for value in recent) else None
            prior_mean = sum(prior) / 3 if all(value is not None for value in prior) else None
            hpi_cutoff = origin - pd.DateOffset(months=2)
            quarter = (hpi_cutoff.month - 1) // 3
            hpi_year = hpi_cutoff.year
            if quarter == 0:
                quarter = 4
                hpi_year -= 1
            current_hpi = prices.get((state, hpi_year, quarter))
            prior_hpi = prices.get((state, hpi_year - 1, quarter))
            bls_asof = (lag_end + pd.offsets.MonthEnd(0) + pd.DateOffset(months=1)).normalize()
            hpi_asof = pd.Timestamp(_quarter_end(hpi_year, quarter)) + pd.DateOffset(months=2)
            rows.append(
                {
                    "state": state,
                    "year": origin.year,
                    "month": origin.month,
                    "unemployment_3m": recent_mean,
                    "unemployment_change_12m": (
                        recent_mean - prior_mean if recent_mean is not None and prior_mean is not None else None
                    ),
                    "hpi_yoy": (
                        current_hpi / prior_hpi - 1
                        if current_hpi is not None and prior_hpi not in {None, 0}
                        else None
                    ),
                    "macro_asof_date": max(bls_asof, hpi_asof),
                }
            )
    result = pd.DataFrame(rows)
    if result.duplicated(["state", "year", "month"]).any():
        raise ValueError("duplicate macro features")
    return result


def build_official_macro_features(
    bls_payload: dict[str, object],
    hpi_master: pd.DataFrame,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    if bls_payload.get("status") != "REQUEST_SUCCEEDED":
        raise ValueError("BLS request did not succeed")
    series = bls_payload.get("Results", {}).get("series", [])
    if len(series) != 1 or series[0].get("seriesID") != "LNS14000000":
        raise ValueError("BLS response must contain national unemployment rate LNS14000000")
    unemployment = pd.DataFrame(
        [
            {
                "date": f"{row['year']}-{row['period'][1:]}-01",
                "unemployment": float(row["value"]),
            }
            for row in series[0]["data"]
            if str(row.get("period", "")).startswith("M") and row["period"] != "M13"
        ]
    )
    required = {
        "hpi_type", "hpi_flavor", "frequency", "level", "place_id", "yr", "period", "index_sa"
    }
    missing = sorted(required.difference(hpi_master.columns))
    if missing:
        raise ValueError(f"FHFA HPI master is missing columns: {missing}")
    selected = hpi_master.loc[
        hpi_master["hpi_type"].eq("traditional")
        & hpi_master["hpi_flavor"].eq("purchase-only")
        & hpi_master["frequency"].eq("quarterly")
        & hpi_master["level"].eq("State"),
        ["place_id", "yr", "period", "index_sa"],
    ].dropna()
    hpi = selected.rename(
        columns={"place_id": "state", "yr": "year", "period": "quarter", "index_sa": "hpi"}
    )
    states = sorted(hpi["state"].unique())
    laus = pd.concat([unemployment.assign(state=state) for state in states], ignore_index=True)
    return build_state_month_features(laus, hpi, start_year, end_year)


def fetch_official_macro_features(
    cache_dir: str | Path,
    *,
    bls_url: str,
    hpi_url: str,
    start_year: int,
    end_year: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    bls_path = cache / f"bls-unemployment-{start_year - 2}-{end_year}.json"
    hpi_path = cache / "fhfa-hpi-master.csv"
    if not bls_path.is_file():
        body = json.dumps(
            {"seriesid": ["LNS14000000"], "startyear": str(start_year - 2), "endyear": str(end_year)}
        ).encode("utf-8")
        request = Request(bls_url, data=body, headers={"Content-Type": "application/json"})
        with urlopen(request, timeout=60) as response:
            payload = response.read()
        temporary = bls_path.with_suffix(".partial")
        temporary.write_bytes(payload)
        temporary.replace(bls_path)
    if not hpi_path.is_file():
        request = Request(hpi_url)
        with urlopen(request, timeout=120) as response:
            payload = response.read()
        temporary = hpi_path.with_suffix(".partial")
        temporary.write_bytes(payload)
        temporary.replace(hpi_path)
    bls_payload = json.loads(bls_path.read_text(encoding="utf-8"))
    features = build_official_macro_features(
        bls_payload,
        pd.read_csv(hpi_path, low_memory=False),
        start_year,
        end_year,
    )
    sources = [
        {"name": bls_path.name, "url": bls_url, "bytes": bls_path.stat().st_size, "sha256": file_sha256(bls_path)},
        {"name": hpi_path.name, "url": hpi_url, "bytes": hpi_path.stat().st_size, "sha256": file_sha256(hpi_path)},
    ]
    return features, sources


def merge_macro(loans: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    result = loans.copy()
    result["origination_date"] = pd.to_datetime(result["origination_date"])
    result["year"] = result["origination_date"].dt.year
    result["month"] = result["origination_date"].dt.month
    result = result.merge(macro, on=["state", "year", "month"], how="left", validate="m:1")
    result["macro_asof_date"] = pd.to_datetime(result["macro_asof_date"])
    future = result["macro_asof_date"].notna() & (result["macro_asof_date"] >= result["origination_date"])
    if future.any():
        raise ValueError("future macro information is not allowed")
    return result
