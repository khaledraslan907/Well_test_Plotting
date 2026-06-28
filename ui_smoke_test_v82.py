from __future__ import annotations

import ast
from pathlib import Path

source = Path(__file__).with_name("app.py").read_text(encoding="utf-8")
ast.parse(source)

required = [
    '"Share-safe labels"',
    'apply_share_safe_anonymization',
    '"Uploaded file"',
    'step=900',
    '"#FFD166"',
    '"#111827"',
    'Hidden in share-safe mode',
    'production_test_{safe_feature}.png',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit("Missing v82 UI safeguards: " + ", ".join(missing))

print("Production Test Dashboard v82 UI smoke test: PASS")
