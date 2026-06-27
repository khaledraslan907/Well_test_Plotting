"""Compatibility smoke test retained from v79 and updated for v80 UI."""
from pathlib import Path
import ast

root = Path(__file__).resolve().parent
app = (root / "app.py").read_text(encoding="utf-8")
parser = (root / "tmu_parser.py").read_text(encoding="utf-8")
ast.parse(app)
ast.parse(parser)
required = [
    'APP_UI_BUILD_ID = "v80-complete-dark-control-visibility-ui-20260627"',
    'PARSER_BUILD_ID = "v79-direct-whatsapp-zip-parser-20260627"',
    '_parse_whatsapp_text_payload',
    '_chat.txt',
    'No usable production-test data was found in the ZIP archive.',
]
missing = [item for item in required if item not in app + "\n" + parser]
if missing:
    raise SystemExit("Missing v79/v80 compatibility fragments: " + ", ".join(missing))
print("Production Test Dashboard v79 parser / v80 UI compatibility smoke test: PASS")
