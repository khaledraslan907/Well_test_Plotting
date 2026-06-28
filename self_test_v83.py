from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

import tmu_parser as parser
from history_analysis import build_production_history, history_trend_column


class UploadedBytes(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


assert parser.PARSER_BUILD_ID.startswith("v82-")

# Generic unfamiliar workbook remains supported by the comprehensive parser.
raw = pd.DataFrame(
    {
        "Date and Time": pd.date_range("2026-06-28 08:00", periods=4, freq="30min"),
        "Wellhead Pressure psi": [800, 805, 810, 815],
        "Gross Liquid BBL/D": [1100, 1110, 1120, 1130],
        "New Sensor Index": [1.1, 1.2, 1.3, 1.4],
    }
)
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    raw.to_excel(writer, index=False, sheet_name="Data")
workbook = UploadedBytes(buffer.getvalue(), "generic_test.xlsx")
tables = parser.load_tabular_file(workbook)
assert tables and sum(len(t) for t in tables) == 4

# Production history: three tests, one stabilized point per test.  Every signal
# uses its own final six valid readings, then a three-test moving average.
rows = []
for test_no, start in enumerate(pd.to_datetime(["2024-01-01", "2025-01-01", "2026-01-01"]), start=1):
    for sample in range(10):
        rows.append(
            {
                "well": "WELL-001",
                "test_id": f"TEST-{test_no}",
                "datetime": start + pd.Timedelta(minutes=30 * sample),
                "gross_rate_bpd": 1200 - 100 * (test_no - 1) + sample,
                "oil_rate_stbd": 900 - 80 * (test_no - 1) + sample,
                "source": "generic.xlsx",
                "sheet": f"Test {test_no}",
            }
        )
source = pd.DataFrame(rows)
history = build_production_history(source, ["gross_rate_bpd", "oil_rate_stbd"], final_readings=6, trend_window=3)
assert len(history) == 3
assert history["test_id"].nunique() == 1
expected_first = source[source["test_id"] == "TEST-1"]["gross_rate_bpd"].tail(6).mean()
assert abs(float(history.iloc[0]["gross_rate_bpd"]) - float(expected_first)) < 1e-9
trend_col = history_trend_column("gross_rate_bpd")
assert trend_col in history.columns
assert history[trend_col].notna().sum() == 2

print("Production Test Dashboard v83 self-test: PASS")
