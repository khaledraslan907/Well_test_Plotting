from __future__ import annotations

import ast
import re
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

import tmu_parser as parser

ROOT = Path(__file__).resolve().parent
APP = (ROOT / "app.py").read_text(encoding="utf-8")
TREE = ast.parse(APP)

assert 'APP_UI_BUILD_ID = "v92-custom-intervals-drag-signal-order-20260628"' in APP
assert parser.PARSER_BUILD_ID == "v91-ocr-continuity-canonical-pressure-20260628"
assert "streamlit-sortables==0.3.1" in (ROOT / "requirements.txt").read_text(encoding="utf-8")
assert "interval_select_with_custom" in APP
assert "draggable_signal_order" in APP
assert '"Every N readings"' in APP
assert '"First, last + every N tests"' in APP
assert "iter_plot_segments(g_all, feature)" in APP

# Execute pure helper functions without starting Streamlit.
helper_names = {
    "_parse_interval_text",
    "time_aggregation_rule",
    "x_axis_tick_kwargs",
    "_compressed_scale_delta",
    "_plot_scalar_to_float",
    "numeric_feature_series",
    "iter_plot_segments",
}
nodes = []
for node in TREE.body:
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "_PLOT_NUMBER_RE" for t in node.targets
    ):
        nodes.append(node)
    if isinstance(node, ast.FunctionDef) and node.name in helper_names:
        nodes.append(node)
namespace = {
    "pd": pd,
    "np": np,
    "re": re,
    "datetime": datetime,
    "date": date,
    "time": time,
}
module = ast.Module(body=nodes, type_ignores=[])
ast.fix_missing_locations(module)
exec(compile(module, str(ROOT / "app.py"), "exec"), namespace)

parse_interval = namespace["_parse_interval_text"]
assert parse_interval("2 hours")["plotly_dtick"] == 7_200_000
assert parse_interval("90 min")["resample_rule"] == "5400000ms"
assert parse_interval("1.5 days")["timedelta"] == pd.Timedelta(hours=36)
assert parse_interval("2 months")["plotly_dtick"] == "M2"
assert parse_interval("1 year")["plotly_dtick"] == "M12"
assert namespace["time_aggregation_rule"]("Custom: 2 hours") == "7200000ms"
assert namespace["x_axis_tick_kwargs"]("Custom: 2 hours")["dtick"] == 7_200_000
assert namespace["_compressed_scale_delta"]("Custom: 2 hours") == pd.Timedelta(hours=2)

# Rows from other sources with NaN in the selected signal must not break its line.
mixed = pd.DataFrame({
    "datetime": pd.date_range("2026-06-23 13:30", periods=7, freq="30min"),
    "plot_x": range(7),
    "series_segment_id": [0] * 7,
    "pumping_pressure_psi": [1100, np.nan, 1300, np.nan, 1500, np.nan, 1700],
    "gross_rate_bpd": [np.nan, 500, np.nan, 520, np.nan, 540, np.nan],
})
segments = namespace["iter_plot_segments"](mixed, "pumping_pressure_psi")
assert len(segments) == 1
assert segments[0]["pumping_pressure_psi"].tolist() == [1100.0, 1300.0, 1500.0, 1700.0]

# Canonical OCR pressure behavior remains unchanged.
ocr = pd.DataFrame({
    "datetime": pd.to_datetime(["2026-06-23 15:15:32"]),
    "source_type": ["ctu_image_ocr_v70"],
    "ctu_circulation_pressure_psi": [1909.09],
    "ctu_wellhead_pressure_psi": [20.34],
})
normalized = parser.normalize_ctu_ocr_signals(ocr)
assert normalized.at[0, "pumping_pressure_psi"] == 1909.09
assert normalized.at[0, "whp_psi"] == 20.34

print("Production Test Dashboard v92 self-test: PASS")
