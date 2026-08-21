from __future__ import annotations

import calendar
from datetime import date

import pandas as pd


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
