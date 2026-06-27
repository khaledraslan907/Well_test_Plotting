"""Static UI checks for the v73 Light/Dark production-test interface."""
from __future__ import annotations
import ast
from pathlib import Path
root = Path(__file__).resolve().parent
source = (root / "app.py").read_text(encoding="utf-8")
ast.parse(source)
required = [
    'APP_UI_BUILD_ID = "v73-ensemble-smart-parser-ui-20260626"',
    'Production Test Analysis &amp; Visualization',
    'Interactive well-test plotting, engineering diagnostics and operational events.',
    '["Light", "Dark"]',
    '[data-testid="stFileUploaderDropzone"]',
    'Engineering Data Checks',
    'Column mapping review / teach new names',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit("Missing v73 UI fragments: " + ", ".join(missing))
if not (root / ".streamlit" / "config.toml").exists():
    raise SystemExit("Missing .streamlit/config.toml")
print("Production Test Dashboard v73 UI smoke test: PASS")
