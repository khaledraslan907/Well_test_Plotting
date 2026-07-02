from __future__ import annotations

import ast
import py_compile
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
APP = (ROOT / "app.py").read_text(encoding="utf-8")

# 1) Compile every deployment Python file.
for py_file in [
    "app.py", "tmu_parser.py", "tmu_parser_compat.py", "tmu_parser_legacy.py",
    "smart_tabular_v75.py", "history_analysis.py",
]:
    py_compile.compile(str(ROOT / py_file), doraise=True)

# 2) Static regression checks for v94.
assert 'APP_UI_BUILD_ID = "v94-note-clearance-smart-y-axis-20260702"' in APP
assert 'Comments stay in a dedicated band above the chart' in APP
assert '["Auto staggered", "Compact top labels"]' in APP
assert '["Auto staggered", "Vertical labels", "Compact top labels"]' not in APP
assert 'def _remove_value_labels_near_notes' in APP
assert 'def default_y_axis_range' in APP
assert 'return [0.0, 100.0]' in APP
assert 'return [0.0, 128.0]' in APP
assert 'return default_y_axis_range(df, feature)' in APP

# 3) Execute axis helpers with a minimal safe namespace.
module = ast.parse(APP)
axis_names = {
    "_nice_axis_ceiling", "_nice_axis_floor", "_is_full_percent_axis",
    "_is_full_choke_size_axis", "default_y_axis_range",
    "combined_default_y_axis_range",
}
axis_nodes = {node.name: node for node in module.body if isinstance(node, ast.FunctionDef) and node.name in axis_names}
labels = {
    "gross_rate_bpd": "Gross Rate (BBL/D)",
    "gas_rate_mmscfd": "Total Gas Rate (MMSCF/D)",
    "bsw_pct": "BS&W (%)",
    "choke_pct": "Choke Opening (%)",
    "choke_size_64": "Choke Size (/64 in)",
    "ctu_reel_speed_ftmin": "CTU Reel Speed (ft/min)",
}
axis_ns = {
    "pd": pd,
    "np": np,
    "math": __import__("math"),
    "column_label": lambda feature: labels.get(feature, feature),
    "numeric_feature_series": lambda frame, feature: pd.to_numeric(frame[feature], errors="coerce").astype("float64"),
}
for name in [
    "_nice_axis_ceiling", "_nice_axis_floor", "_is_full_percent_axis",
    "_is_full_choke_size_axis", "default_y_axis_range",
    "combined_default_y_axis_range",
]:
    exec(compile(ast.Module(body=[axis_nodes[name]], type_ignores=[]), "<axis-helper>", "exec"), axis_ns)

frame = pd.DataFrame({
    "gross_rate_bpd": [94.9, 130.0, 177.0, 118.0],
    "gas_rate_mmscfd": [3.91, 2.76, 1.51, 1.46],
    "bsw_pct": [0.6, 82.1, 90.0, 88.0],
    "choke_pct": [10.0, 80.0, 65.0, 45.0],
    "choke_size_64": [16.0, 64.0, 96.0, 128.0],
    "ctu_reel_speed_ftmin": [-43.63, -0.01, 0.0, -12.0],
})
axis = axis_ns["default_y_axis_range"]
assert axis(frame, "gross_rate_bpd") == [0.0, 200.0]
assert axis(frame, "gas_rate_mmscfd") == [0.0, 5.0]
assert axis(frame, "bsw_pct") == [0.0, 100.0]
assert axis(frame, "choke_pct") == [0.0, 100.0]
assert axis(frame, "choke_size_64") == [0.0, 128.0]
assert axis(frame, "ctu_reel_speed_ftmin") == [-50.0, 0.0]

# 4) Existing continuation helpers remain valid.
namespace = {"pd": pd, "np": np}
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "_continuation_compatible_v93", "_merge_continuation_duplicate_rows_v93"
    }:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<app-helper>", "exec"), namespace)

old = pd.DataFrame({
    "well": ["WELL-A", "WELL-A"],
    "datetime": pd.to_datetime(["2026-06-01 18:00", "2026-06-01 18:30"]),
    "test_id": ["WELL-A_T1", "WELL-A_T1"],
    "gross_rate_bpd": [100.0, 110.0],
    "_continuation_batch": [1, 1],
})
new = pd.DataFrame({
    "well": ["WELL-A", "WELL-A"],
    "datetime": pd.to_datetime(["2026-06-01 18:30", "2026-06-01 19:00"]),
    "test_id": ["WELL-A_T1", "WELL-A_T1"],
    "gross_rate_bpd": [115.0, 120.0],
    "_continuation_batch": [2, 2],
})
assert namespace["_continuation_compatible_v93"](old, new)
merged, merged_count = namespace["_merge_continuation_duplicate_rows_v93"](pd.concat([old, new], ignore_index=True))
assert len(merged) == 3 and merged_count == 1

# 5) Note collision levels still separate close comments.
note_ns = {"pd": pd, "np": np, "re": __import__("re"), "math": __import__("math"), "html": __import__("html")}
nested = {}
for node in ast.walk(module):
    if isinstance(node, ast.FunctionDef) and node.name in {"_note_x_number", "compact_note_label", "note_event_levels"}:
        nested[node.name] = node
for name in ["_note_x_number", "compact_note_label", "note_event_levels"]:
    exec(compile(ast.Module(body=[nested[name]], type_ignores=[]), "<note-helper>", "exec"), note_ns)
close_events = [
    {"plot_x": 1.00, "label": "Closed the well"},
    {"plot_x": 1.05, "label": "SIWHP 3300 PSI"},
    {"plot_x": 1.10, "label": "SIWHP 3100 PSI"},
]
laid_out = note_ns["note_event_levels"](close_events, x_values=[0.0, 10.0])
assert len({item["level"] for item in laid_out}) >= 2

# 6) Parser regression on representative file types.
sys.path.insert(0, str(ROOT))
import tmu_parser  # noqa: E402
from history_analysis import build_production_history  # noqa: E402

class UploadedBytes:
    def __init__(self, path: Path):
        self.name = path.name
        self._data = path.read_bytes()
    def getvalue(self):
        return self._data
    def read(self, *args):
        return self._data
    def seek(self, *args):
        return 0

samples = [
    Path("/mnt/data/SPR(2).xlsx"),
    Path("/mnt/data/6-B15-42 (Dasco 27) (9-6-2026)(4).xlsx"),
    Path("/mnt/data/WhatsApp Chat - Sitra8-58 clean & test(3).zip"),
]
expected_min_rows = [12, 10, 70]
parser_timings = []
for sample, minimum in zip(samples, expected_min_rows):
    start = time.perf_counter()
    tables = tmu_parser.load_tabular_file(UploadedBytes(sample), parse_images=False, max_ocr_images=0)
    elapsed = time.perf_counter() - start
    rows = sum(len(table) for table in tables)
    assert rows >= minimum, (sample.name, rows)
    parser_timings.append((sample.name, rows, elapsed))

esp_path = Path("/mnt/data/ESP.xlsx")
esp_tables = tmu_parser.load_tabular_file(UploadedBytes(esp_path), parse_images=False, max_ocr_images=0)
esp = pd.concat(esp_tables, ignore_index=True, sort=False)
features = [c for c in ["gross_rate_bpd", "oil_rate_stbd"] if c in esp.columns]
history = build_production_history(esp, features)
assert not history.empty and history.groupby("well").size().max() >= 40

print("v94 self-test passed")
for name, rows, elapsed in parser_timings:
    print(f"- {name}: {rows} rows in {elapsed:.3f} s")
print(f"- ESP production history: {len(esp):,} raw rows -> {len(history):,} averaged test points")
