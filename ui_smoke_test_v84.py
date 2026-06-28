from __future__ import annotations

import ast
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "app.py").read_text(encoding="utf-8")
ast.parse(source)

required = [
    'APP_UI_BUILD_ID = "v85-scrollable-note-time-picker-ui-20260628"',
    '"Test detail", "Production history"',
    'build_production_history(filtered, selected_features)',
    'First, last + every 20 tests',
    'average of all valid readings',
    'one connected performance line',
    'Use custom Y-axis ranges',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit("Missing v84 UI/history fragments: " + ", ".join(missing))

for forbidden in [
    "history_trend_column",
    "add_history_trend_plotly",
    "add_history_trend_matplotlib",
    "3-test trend",
    "trend_window=3",
    "final_readings=6",
]:
    if forbidden in source:
        raise SystemExit(f"Obsolete v83 history fragment still present: {forbidden}")

history_source = (root / "history_analysis.py").read_text(encoding="utf-8")
ast.parse(history_source)
assert "Average of all valid readings in each test" in history_source
assert ".mean()" in history_source
assert "rolling(" not in history_source
assert "_history_trend_" not in history_source

print("Production Test Dashboard v84 history / v85 UI compatibility smoke test: PASS")
