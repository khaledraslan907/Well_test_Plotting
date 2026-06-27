"""Static UI checks for v75 continuous plotting and responsive parsing."""
from __future__ import annotations
import ast
from pathlib import Path

source = Path('app.py').read_text(encoding='utf-8')
ast.parse(source)
required = [
    'APP_UI_BUILD_ID = "v75-continuous-fast-responsive-ui-20260627"',
    'Production Test Analysis &amp; Visualization',
    'Interactive well-test plotting, engineering diagnostics and operational events.',
    'data.drop(columns=[c for c in ["_upload_order", "_table_order", "_source_row_order"]',
    'A curve is continuous for the complete detected test',
    'Dashed separators mark actual detected test changes only',
    'Keep real spacing for gaps up to (hours)',
    'Prepare {fmt_label} ({ACTIVE_THEME_NAME})',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit('Missing v75 UI fragments: ' + ', '.join(missing))
assert 'changed_report_after_gap' not in source
assert 'if changed_test or changed_report_after_gap or diff_h' not in source
assert Path('.streamlit/config.toml').exists()
print('Production Test Dashboard v75 UI smoke test: PASS')
