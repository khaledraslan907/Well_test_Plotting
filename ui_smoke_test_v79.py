"""Static checks for the v79 package and cache-busting build IDs."""
from pathlib import Path
import ast

root = Path(__file__).resolve().parent
app = (root / "app.py").read_text(encoding="utf-8")
parser = (root / "tmu_parser.py").read_text(encoding="utf-8")
ast.parse(app)
ast.parse(parser)
required = [
    'APP_UI_BUILD_ID = "v79-whatsapp-zip-reliability-ui-20260627"',
    'PARSER_BUILD_ID = "v79-direct-whatsapp-zip-parser-20260627"',
    '_parse_whatsapp_text_payload',
    '_chat.txt',
    'No usable production-test data was found in the ZIP archive.',
]
missing = [item for item in required if item not in app + "\n" + parser]
if missing:
    raise SystemExit("Missing v79 fragments: " + ", ".join(missing))
print("Production Test Dashboard v79 UI/package smoke test: PASS")
