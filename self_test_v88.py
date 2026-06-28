from __future__ import annotations

import ast
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import tmu_parser as parser
from history_analysis import build_production_history

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path('/mnt/data')


class UploadedBytes(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


assert parser.PARSER_BUILD_ID.startswith('v88-')
app_source = (ROOT / 'app.py').read_text(encoding='utf-8')
ast.parse(app_source)
assert 'history_axis_tick_kwargs' in app_source
assert 'image omitted' in app_source
assert 'apply_share_safe_anonymization(data)' not in app_source

# Extract standalone app helpers without importing Streamlit.
module = ast.parse(app_source)
helper_names = {
    'history_axis_tick_kwargs', 'history_matplotlib_date_format',
    'inspect_chat_zip_media', 'clean_well_label', 'well_title_text',
}
namespace = {
    'pd': pd, 'np': np, 'io': io, 'zipfile': zipfile,
    'Path': Path, 're': __import__('re'),
}
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name in helper_names:
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'app_helpers', 'exec'), namespace)

# Short histories must still show explicit date/time ticks.
short = pd.DataFrame({'datetime': pd.to_datetime([
    '2026-06-12 00:00', '2026-06-13 00:00',
    '2026-06-14 00:00', '2026-06-16 00:00',
])})
short_ticks = namespace['history_axis_tick_kwargs'](short)
assert len(short_ticks['tickvals']) == 4
assert all(short_ticks['ticktext'])

# Multi-year histories use representative month/year ticks and keep endpoints.
long = pd.DataFrame({'datetime': pd.date_range('2021-01-01', periods=54, freq='35D')})
long_ticks = namespace['history_axis_tick_kwargs'](long)
assert 2 <= len(long_ticks['tickvals']) <= 9
assert long_ticks['ticktext'][0]
assert long_ticks['ticktext'][-1]
assert namespace['well_title_text']('Well 1') == 'Well 1'
assert namespace['well_title_text']('B15-42') == 'Well B15-42'

# Generic unfamiliar workbook remains supported.
raw = pd.DataFrame({
    'Date and Time': pd.date_range('2026-06-28 08:00', periods=4, freq='30min'),
    'Wellhead Pressure psi': [800, 805, 810, 815],
    'Gross Liquid BBL/D': [1100, 1110, 1120, 1130],
    'New Sensor Index': [1.1, 1.2, 1.3, 1.4],
})
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
    raw.to_excel(writer, index=False, sheet_name='Data')
tables = parser.load_tabular_file(UploadedBytes(buffer.getvalue(), 'generic_test.xlsx'))
assert tables and sum(len(t) for t in tables) == 4

# Production history remains one arithmetic mean point per test.
rows = []
for test_no, start in enumerate(pd.to_datetime(['2024-01-01', '2025-01-01', '2026-01-01']), start=1):
    for sample in range(10):
        rows.append({
            'well': 'WELL-001',
            'test_id': f'TEST-{test_no}',
            'datetime': start + pd.Timedelta(minutes=30 * sample),
            'gross_rate_bpd': 1200 - 100 * (test_no - 1) + sample,
            'source': 'generic.xlsx',
            'sheet': f'Test {test_no}',
        })
history = build_production_history(pd.DataFrame(rows), ['gross_rate_bpd'])
assert len(history) == 3
assert history['datetime'].notna().all()

# The supplied export contains image placeholders, but no actual image files.
actual_zip = DATA_ROOT / 'WhatsApp Chat - Sitra8-58 clean & test(3).zip'
if actual_zip.exists():
    summary = namespace['inspect_chat_zip_media'](actual_zip.name, actual_zip.read_bytes())
    assert summary['image_files'] == 0
    assert summary['image_omitted_references'] >= 100
    parsed = parser.load_tabular_file(
        UploadedBytes(actual_zip.read_bytes(), actual_zip.name),
        parse_images=True,
        max_ocr_images=1000,
    )
    assert parsed and len(parsed[0]) > 0
    assert not parsed[0]['source_type'].astype(str).str.contains('ocr', case=False, na=False).any()

# A ZIP that physically contains the two supplied images must produce OCR rows.
images = [
    DATA_ROOT / 'WhatsApp Image 2026-06-16 at 14.35.12(1).jpeg',
    DATA_ROOT / 'WhatsApp Image 2026-06-16 at 16.40.11.jpeg',
]
if all(path.exists() for path in images):
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('_chat.txt', '[16/06/2026, 14:35:12] User: images attached\n')
        for path in images:
            zf.writestr(path.name, path.read_bytes())
    summary = namespace['inspect_chat_zip_media']('with_images.zip', zip_buffer.getvalue())
    assert summary['image_files'] == 2
    parsed = parser.load_tabular_file(
        UploadedBytes(zip_buffer.getvalue(), 'with_images.zip'),
        parse_images=True,
        max_ocr_images=100,
    )
    ocr_rows = sum(
        frame['source_type'].astype(str).str.contains('ocr', case=False, na=False).sum()
        for frame in parsed
    )
    assert ocr_rows == 2

print('Production Test Dashboard v88 self-test: PASS')
