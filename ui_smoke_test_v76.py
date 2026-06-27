"""Static UI checks for v76 dark-theme visibility and scroll controls."""
from __future__ import annotations
import ast
from pathlib import Path

source = Path('app.py').read_text(encoding='utf-8')
ast.parse(source)
required = [
    'APP_UI_BUILD_ID = "v76-dark-visibility-scroll-controls-ui-20260627"',
    '--petro-scroll-track:',
    '--petro-scroll-thumb:',
    '--petro-scroll-thumb-hover:',
    '*::-webkit-scrollbar-button:single-button:vertical:decrement',
    '*::-webkit-scrollbar-button:single-button:vertical:increment',
    '[data-testid="stSidebarCollapseButton"]',
    '[data-testid="stExpandSidebarButton"]',
    '[data-testid="stFileChip"]',
    '[data-testid="stFileChipName"]',
    '[data-testid="stNumberInputStepDown"]',
    '[data-testid="stNumberInputStepUp"]',
    '[data-testid="stTabsScrollLeft"]',
    '[data-testid="stTabsScrollRight"]',
    '.js-plotly-plot .modebar-btn path',
    'section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{ overflow-y: scroll !important; }}',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit('Missing v76 dark-theme fragments: ' + ', '.join(missing))
assert Path('.streamlit/config.toml').exists()
print('Production Test Dashboard v76 UI smoke test: PASS')
