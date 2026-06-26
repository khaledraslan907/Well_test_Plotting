"""Static checks for the v71 Light/Dark professional TMU interface."""
from __future__ import annotations

import ast
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "app.py").read_text(encoding="utf-8")
ast.parse(source)

required = [
    'APP_UI_BUILD_ID = "v71-simple-light-dark-ui-20260626"',
    '"Light": {',
    '"Dark": {',
    'TMU Production Test Analysis &amp; Visualization',
    'Interactive well-test plotting, engineering diagnostics',
    'class="petro-chart-icon"',
    'st.radio(',
    '["Light", "Dark"]',
    'chart_paper": "#0B1822"',
    'chart_text": "#EAF2F5"',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit("Missing v71 UI fragments: " + ", ".join(missing))

for forbidden in [
    '"Petroleum Dark": {',
    '"Field Report Light": {',
    '"Night Shift": {',
    'TMU Production Test Intelligence',
    '<div class="petro-mark">🛢️</div>',
    'petro-pills',
]:
    if forbidden in source:
        raise SystemExit("Old UI fragment still present: " + forbidden)

if not (root / ".streamlit" / "config.toml").exists():
    raise SystemExit("Missing .streamlit/config.toml")

print("TMU Dashboard v71 Light/Dark UI smoke test: PASS")
