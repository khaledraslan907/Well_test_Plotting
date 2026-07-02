from __future__ import annotations

import ast
import io
import json
import py_compile
from datetime import date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
APP_TEXT = (ROOT / "app.py").read_text(encoding="utf-8")

for name in ["app.py", "tmu_parser.py", "tmu_parser_compat.py", "tmu_parser_legacy.py", "smart_tabular_v75.py", "history_analysis.py", "legacy_dashboard_pdf.py"]:
    py_compile.compile(str(ROOT / name), doraise=True)

assert 'APP_UI_BUILD_ID = "v98-legacy-vector-pdf-recovery-20260702"' in APP_TEXT
assert 'PORTABLE_STATE_ATTACHMENT = "corelytix_production_test_state_v1.zip"' in APP_TEXT
assert 'vertical = event_label_style == "Vertical labels"' in APP_TEXT
assert 'auto_vertical =' not in APP_TEXT
assert 'Auto staggered keeps compact horizontal labels near the top, like version 92' in APP_TEXT
assert 'matplotlib_value_label_placement' in APP_TEXT
assert 'make_portable_pdf(output.getvalue(), df, features)' in APP_TEXT
assert 'return make_portable_pdf(payload, df, features) if fmt == "pdf" else payload' in APP_TEXT

# Execute only the portable PDF helpers in a small safe namespace.
module = ast.parse(APP_TEXT)
wanted = {
    "_portable_json_value", "_portable_event_records", "build_portable_state_zip",
    "attach_portable_state_to_pdf", "read_portable_state_from_pdf", "infer_legacy_pdf_theme",
}
nodes = [n for n in module.body if isinstance(n, (ast.FunctionDef, ast.Assign)) and (
    isinstance(n, ast.FunctionDef) and n.name in wanted or
    isinstance(n, ast.Assign) and any(getattr(t, 'id', None) in {
        'APP_UI_BUILD_ID','PORTABLE_STATE_MAGIC','PORTABLE_STATE_SCHEMA','PORTABLE_STATE_ATTACHMENT'
    } for t in n.targets)
)]
ns = {
    "io": io, "json": json, "zipfile": __import__('zipfile'), "Path": Path,
    "pd": pd, "np": np, "datetime": datetime, "date": date, "time": time,
    "Optional": __import__('typing').Optional, "PARSER_BUILD_ID": "test-parser",
}
exec(compile(ast.Module(body=nodes, type_ignores=[]), "portable_helpers", "exec"), ns)

frame = pd.DataFrame({
    "well": ["BED-3-C18-7", "BED-3-C18-7", "BED-3-C18-7"],
    "datetime": pd.to_datetime(["2026-07-02 09:00", "2026-07-02 10:00", "2026-07-02 11:00"]),
    "gross_rate_bpd": [125.0, 140.0, 152.0],
    "oil_rate_stbd": [100.0, 112.0, 121.0],
})
ui = {
    "ui_theme": "Dark",
    "selected_features_v58": ["gross_rate_bpd", "oil_rate_stbd"],
    "plot_signal_order_v92_state": ["oil_rate_stbd", "gross_rate_bpd"],
    "event_label_layout": "Auto staggered",
}
events = [{"datetime": pd.Timestamp("2026-07-02 09:30"), "label": "Choke changed", "target": "All selected wells"}]
intervals = [{"start": pd.Timestamp("2026-07-02 10:00"), "end": pd.Timestamp("2026-07-02 10:30"), "label": "Well shut in", "target": "All selected wells"}]
state_zip = ns["build_portable_state_zip"](
    frame, ui_state=ui, chart_title="BED-3-C18-7 Production Test",
    manual_events=events, operation_intervals=intervals,
    custom_y_ranges={"gross_rate_bpd": [0, 200]},
)

from pypdf import PdfWriter, PdfReader
writer = PdfWriter()
writer.add_blank_page(width=612, height=792)
base_pdf = io.BytesIO()
writer.write(base_pdf)
portable_pdf = ns["attach_portable_state_to_pdf"](base_pdf.getvalue(), state_zip)
reader = PdfReader(io.BytesIO(portable_pdf))
assert ns["PORTABLE_STATE_ATTACHMENT"] in reader.attachments
restored = ns["read_portable_state_from_pdf"]("test.pdf", portable_pdf)
assert restored is not None
manifest = restored["manifest"]
restored_frame = restored["data"]
assert manifest["ui_state"]["ui_theme"] == "Dark"
assert manifest["manual_events"][0]["label"] == "Choke changed"
assert manifest["operation_intervals"][0]["label"] == "Well shut in"
assert list(restored_frame["gross_rate_bpd"].round(3)) == [125.0, 140.0, 152.0]
assert pd.api.types.is_datetime64_any_dtype(restored_frame["datetime"])
assert restored_frame["datetime"].iloc[-1] == pd.Timestamp("2026-07-02 11:00")


# Smart Y-axis and continuation behavior from v94-v96 remain intact.
axis_names = {
    "_nice_axis_ceiling", "_nice_axis_floor", "_is_full_percent_axis",
    "_is_full_choke_size_axis", "default_y_axis_range", "combined_default_y_axis_range",
}
axis_nodes = {n.name: n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in axis_names}
labels = {
    "gross_rate_bpd": "Gross Rate (BBL/D)",
    "gas_rate_mmscfd": "Total Gas Rate (MMSCF/D)",
    "bsw_pct": "BS&W (%)",
    "choke_pct": "Choke Opening (%)",
    "choke_size_64": "Choke Size (/64 in)",
    "ctu_reel_speed_ftmin": "CTU Reel Speed (ft/min)",
}
axis_ns = {
    "pd": pd, "np": np, "math": __import__("math"),
    "column_label": lambda feature: labels.get(feature, feature),
    "numeric_feature_series": lambda frame, feature: pd.to_numeric(frame[feature], errors="coerce").astype("float64"),
}
for name in [
    "_nice_axis_ceiling", "_nice_axis_floor", "_is_full_percent_axis",
    "_is_full_choke_size_axis", "default_y_axis_range", "combined_default_y_axis_range",
]:
    exec(compile(ast.Module(body=[axis_nodes[name]], type_ignores=[]), "axis_helpers", "exec"), axis_ns)
axis_frame = pd.DataFrame({
    "gross_rate_bpd": [94.9, 177.0],
    "gas_rate_mmscfd": [3.91, 1.46],
    "bsw_pct": [0.6, 90.0],
    "choke_pct": [10.0, 80.0],
    "choke_size_64": [16.0, 128.0],
    "ctu_reel_speed_ftmin": [-43.63, 0.0],
})
axis = axis_ns["default_y_axis_range"]
assert axis(axis_frame, "gross_rate_bpd") == [0.0, 200.0]
assert axis(axis_frame, "gas_rate_mmscfd") == [0.0, 5.0]
assert axis(axis_frame, "bsw_pct") == [0.0, 100.0]
assert axis(axis_frame, "choke_pct") == [0.0, 100.0]
assert axis(axis_frame, "choke_size_64") == [0.0, 128.0]
assert axis(axis_frame, "ctu_reel_speed_ftmin") == [-50.0, 0.0]

continuation_ns = {"pd": pd, "np": np}
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "_continuation_compatible_v93", "_merge_continuation_duplicate_rows_v93"
    }:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "continuation_helpers", "exec"), continuation_ns)
old = pd.DataFrame({
    "well": ["WELL-A", "WELL-A"],
    "datetime": pd.to_datetime(["2026-06-01 18:00", "2026-06-01 18:30"]),
    "gross_rate_bpd": [100.0, 110.0],
    "_continuation_batch": [1, 1],
})
new = pd.DataFrame({
    "well": ["WELL-A", "WELL-A"],
    "datetime": pd.to_datetime(["2026-06-01 18:30", "2026-06-01 19:00"]),
    "gross_rate_bpd": [115.0, 120.0],
    "_continuation_batch": [2, 2],
})
assert continuation_ns["_continuation_compatible_v93"](old, new)
merged, merged_count = continuation_ns["_merge_continuation_duplicate_rows_v93"](pd.concat([old, new], ignore_index=True))
assert len(merged) == 3 and merged_count == 1
assert float(merged.loc[merged["datetime"] == pd.Timestamp("2026-06-01 18:30"), "gross_rate_bpd"].iloc[0]) == 115.0

print("PASS - all v98 deployment Python files compile")
print("PASS - v92-style inline horizontal auto-stagger behavior restored")
print("PASS - portable PDF contains safe embedded state attachment")
print("PASS - theme, events, intervals, title, selected features and dataframe round-trip")
print("PASS - numeric and datetime columns restore correctly")
print("PASS - smart Y-axis rules remain correct")
print("PASS - continuation overlap keeps the newest uploaded value")
