"""Static UI checks for v77 segmentation and dark dropdown visibility."""
from __future__ import annotations

import ast
from pathlib import Path

source = Path("app.py").read_text(encoding="utf-8")
ast.parse(source)
required = [
    'APP_UI_BUILD_ID = "v77-complete-test-ocr-dark-dropdown-ui-20260627"',
    'Smart parser detection (recommended)',
    'Custom inactive gap',
    'Keep each well/source as one test',
    'preserve_existing=True',
    '_auto_link_ocr_rows_by_time_context',
    'body > div[data-baseweb="popover"]',
    '[data-baseweb="popover"] [role="option"] span',
    'background-color: {ACTIVE_THEME[\'panel_bg_2\']} !important;',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit("Missing v77 UI fragments: " + ", ".join(missing))
assert Path(".streamlit/config.toml").exists()
print("Production Test Dashboard v77 UI smoke test: PASS")
