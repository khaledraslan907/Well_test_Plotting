"""Runtime checks for v76 Light/Dark UI behavior and parser compatibility."""
from __future__ import annotations
import io
from pathlib import Path
from streamlit.testing.v1 import AppTest
import tmu_parser as parser

assert parser.PARSER_BUILD_ID.startswith('v75-')

# The UI must run and rerun cleanly in both themes.
at = AppTest.from_file('app.py', default_timeout=60)
at.run()
assert not at.exception
assert at.radio and at.radio[0].label == 'Theme'
at.radio[0].set_value('Dark')
at.run()
assert not at.exception

# Confirm a representative workbook still parses after this UI-only update.
class Uploaded(io.BytesIO):
    def __init__(self, path: Path):
        super().__init__(path.read_bytes())
        self.name = path.name

for candidate in [
    Path('/mnt/data/SPR(2).xlsx'),
    Path('/mnt/data/6-B15-42 (Dasco 27) (9-6-2026)(4).xlsx'),
    Path('/mnt/data/12-S8-58 (12-06-2026)(3).xlsx'),
]:
    if candidate.exists():
        tables = parser.load_tabular_file(Uploaded(candidate), parse_images=False)
        assert tables and not tables[0].empty

print('Production Test Dashboard v76 self-test: PASS')
