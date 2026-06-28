from __future__ import annotations

import ast
import io
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots

import tmu_parser as parser
from history_analysis import build_production_history

ROOT = Path(__file__).resolve().parent


class UploadedBytes(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


app_source = (ROOT / "app.py").read_text(encoding="utf-8")
tree = ast.parse(app_source)
assert 'APP_UI_BUILD_ID = "v90-multiwell-history-axis-performance-fix-20260628"' in app_source
assert "**axis_tick_settings" not in app_source
assert "automatic_chart_header(selected_wells, selected_tests)" in app_source
assert "scale=x_axis_scale" in app_source
assert "max_total_points: int = 8000" in app_source
assert parser.PARSER_BUILD_ID.startswith("v89-")

# Execute selected pure helpers without importing the Streamlit application.
helper_names = {
    "clean_well_label",
    "well_title_text",
    "automatic_chart_header",
    "_compressed_scale_delta",
    "compressed_axis_tick_kwargs",
    "history_axis_tick_kwargs",
}
helper_nodes = [
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name in helper_names
]
namespace = {"pd": pd, "np": np, "re": re}
exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), str(ROOT / "app.py"), "exec"), namespace)

assert namespace["automatic_chart_header"](["ALASSIL-4"]) == "Well ALASSIL-4"
assert namespace["automatic_chart_header"](["BAHGA-11"]) == "Well BAHGA-11"
assert namespace["automatic_chart_header"](["ALASSIL-4", "BAHGA-11"]) == "Well comparison: ALASSIL-4 vs BAHGA-11"

# Production-history Plotly regression: history ticks must not duplicate the
# explicit showticklabels argument used by report-style subplots.
history_frame = pd.DataFrame({"datetime": pd.date_range("2021-01-01", periods=54, freq="35D")})
history_ticks = namespace["history_axis_tick_kwargs"](history_frame, max_ticks=9)
assert "showticklabels" not in history_ticks
fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
for row in (1, 2):
    merged = dict(history_ticks)
    merged.update(showticklabels=True, showgrid=True)
    fig.update_xaxes(row=row, col=1, **merged)

# Compressed multi-well timeline: the 12-hour choice must alter tick selection,
# labels must be capped, and labels must remain readable across separated years.
rows = []
for dt, x in zip(pd.date_range("2014-12-05 09:00", periods=20, freq="30min"), np.arange(20) * 0.5):
    rows.append((x, dt))
for dt, x in zip(pd.date_range("2020-09-16 22:00", periods=20, freq="30min"), 10.25 + np.arange(20) * 0.5):
    rows.append((x, dt))
compressed = pd.DataFrame(rows, columns=["plot_x", "datetime"])
auto_ticks = namespace["compressed_axis_tick_kwargs"](compressed, scale="Auto readable", max_total_ticks=8)
twelve_hour_ticks = namespace["compressed_axis_tick_kwargs"](compressed, scale="12 hours", max_total_ticks=8)
assert len(auto_ticks["tickvals"]) <= 8
assert len(twelve_hour_ticks["tickvals"]) <= 8
assert auto_ticks["tickvals"] != twelve_hour_ticks["tickvals"]
assert all("2014" in text or "2020" in text for text in twelve_hour_ticks["ticktext"])

# pandas 3 OCR-linking regression remains fixed.
df = pd.DataFrame({
    "datetime": pd.to_datetime(["2026-06-23 15:15:00", "2026-06-23 15:20:00"]),
    "source_type": ["tabular", "ctu_image_ocr_v70"],
    "well": ["WELL-001", "Unknown"],
    "test_id": ["WELL-001_20260623_1515", "Unknown_20260623_1520"],
    "test_sequence": pd.Series([1, 2], dtype="Int64"),
    "link_status": ["source_confirmed", "ocr_manual_link_required"],
    "suggested_well": pd.Series([np.nan, np.nan], dtype="float64"),
    "suggested_test_id": pd.Series([np.nan, np.nan], dtype="float64"),
    "suggested_link_reason": pd.Series([np.nan, np.nan], dtype="float64"),
})
with warnings.catch_warnings():
    warnings.simplefilter("error", FutureWarning)
    linked = parser.auto_link_ocr_rows_by_time_context(df, max_gap_hours=3.0)
assert linked.at[1, "well"] == "WELL-001"

# Generic workbook and production-history behavior remain supported.
raw = pd.DataFrame({
    "Date and Time": pd.date_range("2026-06-28 08:00", periods=4, freq="30min"),
    "Wellhead Pressure psi": [800, 805, 810, 815],
    "Gross Liquid BBL/D": [1100, 1110, 1120, 1130],
    "New Sensor Index": [1.1, 1.2, 1.3, 1.4],
})
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    raw.to_excel(writer, index=False, sheet_name="Data")
tables = parser.load_tabular_file(UploadedBytes(buffer.getvalue(), "generic_test.xlsx"))
assert tables and sum(len(table) for table in tables) == 4

history_rows = []
for test_no, start in enumerate(pd.to_datetime(["2024-01-01", "2025-01-01", "2026-01-01"]), start=1):
    for sample in range(10):
        history_rows.append({
            "well": "WELL-001",
            "test_id": f"TEST-{test_no}",
            "datetime": start + pd.Timedelta(minutes=30 * sample),
            "gross_rate_bpd": 1200 - 100 * (test_no - 1) + sample,
            "source": "generic.xlsx",
            "sheet": f"Test {test_no}",
        })
history = build_production_history(pd.DataFrame(history_rows), ["gross_rate_bpd"])
assert len(history) == 3

print("Production Test Dashboard v90 self-test: PASS")
