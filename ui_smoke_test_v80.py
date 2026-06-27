"""Static checks for the v80 complete dark-control visibility layer."""
from pathlib import Path
import ast

root = Path(__file__).resolve().parent
app = (root / "app.py").read_text(encoding="utf-8")
ast.parse(app)

required = [
    'APP_UI_BUILD_ID = "v80-complete-dark-control-visibility-ui-20260627"',
    '[data-testid="stTooltipIcon"] button',
    '[data-testid="stTooltipHoverTarget"] svg.icon',
    'li[aria-live="polite"]',
    '[data-baseweb="popover"] [data-baseweb="menu"]',
    '[data-testid="stMultiSelect"] span[data-baseweb="tag"]',
    'caret-color: var(--petro-accent-hover)',
    'background: var(--petro-panel-2) !important;',
]
missing = [item for item in required if item not in app]
if missing:
    raise SystemExit("Missing v80 dark-theme fragments: " + ", ".join(missing))

print("Production Test Dashboard v80 dark-control UI smoke test: PASS")
