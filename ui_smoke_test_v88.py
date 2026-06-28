from __future__ import annotations

import ast
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / 'app.py').read_text(encoding='utf-8')
ast.parse(source)

required = [
    'APP_UI_BUILD_ID = "v88-history-time-and-zip-media-status-20260628"',
    'Production Test Analysis & Visualization',
    'Test detail',
    'Production history',
    'history_axis_tick_kwargs',
    'Read images inside chat export ZIPs',
    "'image omitted'",
    'well_title_text',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit('Missing v88 UI fragments: ' + ', '.join(missing))

if 'x_axis_scale = "1 year"' in source:
    raise SystemExit('Fixed annual production-history tick scale still present')

print('Production Test Dashboard v88 UI smoke test: PASS')
