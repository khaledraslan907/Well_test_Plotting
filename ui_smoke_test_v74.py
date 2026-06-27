"""Static UI/export checks for Production Test Dashboard v74."""
from __future__ import annotations
import ast
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / "app.py").read_text(encoding="utf-8")
ast.parse(source)

required = [
    'APP_UI_BUILD_ID = "v74-safe-numeric-theme-export-20260627"',
    'Production Test Analysis &amp; Visualization',
    'Interactive well-test plotting, engineering diagnostics and operational events.',
    '["Light", "Dark"]',
    'def numeric_feature_series(',
    'Prepare {fmt_label} ({ACTIVE_THEME_NAME})',
    'export_bytes_{export_key}_{theme_key}',
    'facecolor=CHART_PAPER_BG',
    'Exports use the active {ACTIVE_THEME_NAME} theme',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit("Missing v74 UI/export fragments: " + ", ".join(missing))
if 'yrange = float(valid.max() - valid.min())' not in source:
    raise SystemExit("Expected label-range logic not found")
if 'y = numeric_feature_series(g, feature, reset_index=True)' not in source:
    raise SystemExit("Label-range logic is not protected by safe float coercion")
if not (root / ".streamlit" / "config.toml").exists():
    raise SystemExit("Missing .streamlit/config.toml")
print("Production Test Dashboard v74 UI/export smoke test: PASS")
