from __future__ import annotations

import ast
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "app.py").read_text(encoding="utf-8")
ast.parse(source)

required = [
    'APP_UI_BUILD_ID = "v83-detail-history-fast-ui-20260628"',
    '"Test detail", "Production history"',
    'build_production_history(',
    'final_readings=6',
    'trend_window=3',
    'One point per test',
    '3-test trend',
    'analysis_view == "Production history"',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit("Missing v83 UI/history fragments: " + ", ".join(missing))

history_source = (root / "history_analysis.py").read_text(encoding="utf-8")
ast.parse(history_source)
assert "Average of final" in history_source
assert "rolling(trend_window" in history_source

print("Production Test Dashboard v83 UI smoke test: PASS")
