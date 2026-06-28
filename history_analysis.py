from __future__ import annotations

"""Fast long-term production-history reduction.

The detailed parser may return thousands of readings from dozens of tests.  A
multi-year performance chart should not render every sample.  This module
reduces each detected test to one stabilized point using the final valid
readings, then adds a small moving-average trend for decline/increase review.
"""

from typing import Iterable

import numpy as np
import pandas as pd


def history_trend_column(feature: str) -> str:
    return f"_history_trend_{feature}"


def _safe_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if frame is None or column not in frame.columns:
        return pd.Series(np.nan, index=getattr(frame, "index", None), dtype="float64")
    positions = [i for i, name in enumerate(frame.columns) if name == column]
    if not positions:
        return pd.Series(np.nan, index=frame.index, dtype="float64")
    candidates = []
    for pos in positions:
        raw = frame.iloc[:, pos]
        if pd.api.types.is_bool_dtype(raw.dtype):
            vals = raw.astype("float64")
        else:
            vals = pd.to_numeric(
                raw.astype(str)
                .str.replace(",", "", regex=False)
                .str.extract(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)", expand=False),
                errors="coerce",
            )
        candidates.append(vals.astype("float64"))
    if len(candidates) == 1:
        return candidates[0]
    return pd.concat(candidates, axis=1).bfill(axis=1).iloc[:, 0].astype("float64")


def _join_unique(values: Iterable[object]) -> str:
    result: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none"} or text in result:
            continue
        result.append(text)
    return ", ".join(result)


def build_production_history(
    frame: pd.DataFrame,
    features: Iterable[str],
    *,
    final_readings: int = 6,
    trend_window: int = 3,
) -> pd.DataFrame:
    """Return one stabilized point per detected test.

    Each feature is calculated independently from its final ``final_readings``
    valid samples.  This avoids a missing value in one signal changing the
    averaging window of another signal.  The point date is the test end time.
    A moving-average trend is added under a private, non-selectable column name.
    """
    if frame is None or frame.empty:
        return pd.DataFrame()

    final_readings = max(1, int(final_readings))
    trend_window = max(2, int(trend_window))
    features = [str(f) for f in features if str(f) in frame.columns]
    if not features:
        return pd.DataFrame()

    data = frame.copy(deep=False)
    if "datetime" not in data.columns:
        return pd.DataFrame()
    data = data.copy()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data = data[data["datetime"].notna()]
    if data.empty:
        return pd.DataFrame()

    if "well" not in data.columns:
        data["well"] = "Unknown"

    has_test_id = "test_id" in data.columns and data["test_id"].astype(str).str.strip().ne("").any()
    if has_test_id:
        group_cols = ["well", "test_id"]
    else:
        fallback = [c for c in ["well", "source", "sheet"] if c in data.columns]
        group_cols = fallback or ["well"]

    rows: list[dict] = []
    for group_key, group in data.groupby(group_cols, dropna=False, sort=False):
        g = group.sort_values("datetime", kind="stable")
        if g.empty:
            continue
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        meta = dict(zip(group_cols, group_key))
        row: dict = {
            "well": str(meta.get("well", g["well"].iloc[0])),
            "datetime": pd.Timestamp(g["datetime"].max()),
            "test_start": pd.Timestamp(g["datetime"].min()),
            "test_end": pd.Timestamp(g["datetime"].max()),
            "test_readings": int(len(g)),
            "_history_source_test_id": str(meta.get("test_id", "")),
            "source": _join_unique(g["source"].dropna()) if "source" in g.columns else "",
            "sheet": _join_unique(g["sheet"].dropna()) if "sheet" in g.columns else "Production history",
            "source_type": "production_history",
            "history_method": f"Average of final {final_readings} valid readings",
        }
        usable = False
        for feature in features:
            valid = _safe_numeric_series(g, feature).dropna()
            value = float(valid.tail(final_readings).mean()) if not valid.empty else np.nan
            row[feature] = value
            usable = usable or pd.notna(value)
        if usable:
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values(["well", "datetime"], kind="stable").reset_index(drop=True)
    out["date"] = out["datetime"].dt.floor("D")
    out["time_text"] = out["datetime"].dt.strftime("%H:%M")

    # A single history test_id per well deliberately keeps all test points on
    # one continuous line.  The original detected test ID is retained privately.
    out["test_id"] = out["well"].astype(str) + "_production_history"
    out["test_sequence"] = out.groupby("well", sort=False).cumcount() + 1

    for feature in features:
        trend_col = history_trend_column(feature)
        out[trend_col] = (
            out.groupby("well", sort=False)[feature]
            .transform(lambda values: pd.to_numeric(values, errors="coerce").rolling(trend_window, min_periods=2).mean())
            .astype("float64")
        )

    return out
