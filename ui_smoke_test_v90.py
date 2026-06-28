from __future__ import annotations

import ast
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "app.py").read_text(encoding="utf-8")
ast.parse(source)
required = [
    'APP_UI_BUILD_ID = "v90-multiwell-history-axis-performance-fix-20260628"',
    'automatic_chart_header(selected_wells, selected_tests)',
    'scale=x_axis_scale',
    'def merged_xaxis_kwargs',
    'hover_mode = "closest" if len(series_values) > 1',
    'max_total_points: int = 8000',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit("Missing v90 UI fragments: " + ", ".join(missing))
if "**axis_tick_settings" in source:
    raise SystemExit("Unsafe duplicate Plotly axis kwargs remain")
print("Production Test Dashboard v90 UI smoke test: PASS")
