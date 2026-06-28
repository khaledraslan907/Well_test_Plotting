from pathlib import Path
import ast

root = Path(__file__).resolve().parent
source = (root / "app.py").read_text(encoding="utf-8")
ast.parse(source)
required = [
    'APP_UI_BUILD_ID = "v92-custom-intervals-drag-signal-order-20260628"',
    'from streamlit_sortables import sort_items as _sort_items',
    '"Average readings by time interval"',
    '"X-axis tick scale"',
    'placeholder="e.g. 2 hours"',
    'header="Drag to change plot order"',
    '"Every N readings"',
    '"First, last + every N tests"',
    'default_features = list(dict.fromkeys(default_features))',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit("Missing v92 UI fragments: " + ", ".join(missing))
print("Production Test Dashboard v92 UI smoke test: PASS")
