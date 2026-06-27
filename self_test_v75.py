"""Regression and performance tests for Production Test Dashboard v75."""
from __future__ import annotations

import ast
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

import smart_tabular_v75 as smart
import tmu_parser as parser

ROOT = Path('/mnt/data')

assert parser.PARSER_BUILD_ID.startswith('v75-')
assert smart.ENGINE_ID.startswith('v73-') or smart.ENGINE_ID.startswith('v75-')
assert smart.table_number('Pressure test 4000 psi') != smart.table_number('4000 psi')
assert pd.isna(smart.table_number('Pressure test 4000 psi'))
assert smart.table_number('4000 psi') == 4000.0
assert smart.infer_field('pumping.p psi').key == 'pumping_pressure_psi'
assert smart.infer_field('Sep.P psi').key == 'sep_p_psi'
assert smart.infer_field('N2 MMSCF/D').key == 'n2_rate_mmscfd'

class Uploaded(io.BytesIO):
    def __init__(self, path: Path):
        super().__init__(path.read_bytes())
        self.name = path.name

results = {}

spr_path = ROOT / 'SPR(2).xlsx'
if spr_path.exists():
    t0 = time.perf_counter()
    spr = parser.load_tabular_file(Uploaded(spr_path), parse_images=False)[0]
    results['spr_seconds'] = time.perf_counter() - t0
    assert len(spr) == 12
    assert {'stroke_length_in', 'stroke_rate_spm', 'peak_load_lbf', 'minimum_load_lbf'}.issubset(spr.columns)
    numeric = parser.available_numeric_columns(spr)
    assert '_upload_order' not in numeric and '_table_order' not in numeric
    assert results['spr_seconds'] < 5.0

b15_path = ROOT / '6-B15-42 (Dasco 27) (9-6-2026)(4).xlsx'
if b15_path.exists():
    t0 = time.perf_counter()
    b15 = parser.load_tabular_file(Uploaded(b15_path), parse_images=False)[0]
    results['b15_seconds'] = time.perf_counter() - t0
    assert len(b15) == 10
    assert {'gross_rate_bpd', 'oil_rate_stbd', 'water_rate_bpd', 'gas_rate_mmscfd'}.issubset(b15.columns)
    numeric = parser.available_numeric_columns(b15)
    assert not any(c.startswith('raw_calcul') or c.startswith('raw_channel') for c in numeric)
    assert results['b15_seconds'] < 12.0

s8_path = ROOT / '12-S8-58 (12-06-2026)(3).xlsx'
if s8_path.exists():
    t0 = time.perf_counter()
    s8 = parser.load_tabular_file(Uploaded(s8_path), parse_images=False)[0]
    results['s8_seconds'] = time.perf_counter() - t0
    assert len(s8) >= 50
    assert pd.Timestamp(s8['datetime'].min()) == pd.Timestamp('2026-06-12 18:00')
    assert {'whp_psi', 'n2_rate_mmscfd', 'sep_p_psi', 'gas_rate_mmscfd', 'gas_formation_mmscfd', 'gross_rate_bpd', 'pumping_pressure_psi'}.issubset(s8.columns)
    assert results['s8_seconds'] < 15.0

# Execute the actual app axis helpers and confirm daily SRP readings stay one curve.
source = Path('app.py').read_text(encoding='utf-8')
tree = ast.parse(source)
needed = {
    'is_aligned_elapsed_mode', 'is_compressed_real_date_mode', 'clean_well_label',
    'compressed_time_mapping', 'add_plot_axis_columns', 'iter_plot_segments',
}
nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in needed]
ns = {'pd': pd, 'np': np, 're': re}
exec(compile(ast.Module(body=nodes, type_ignores=[]), 'axis_helpers', 'exec'), ns)
if spr_path.exists():
    spr = parser.assign_test_ids(parser.load_tabular_file(Uploaded(spr_path), parse_images=False)[0], gap_hours=72)
    plotted = ns['add_plot_axis_columns'](
        spr, 'Compressed real dates - remove empty gaps',
        continuous_gap_hours=2.0, compressed_gap_hours=0.75,
    )
    segments = ns['iter_plot_segments'](plotted)
    assert len(segments) == 1 and len(segments[0]) == 12
    assert plotted.attrs.get('compressed_separators') == []

# Full Streamlit interaction: upload, plot, switch theme, and change plot style.
if spr_path.exists():
    at = AppTest.from_file('app.py', default_timeout=40)
    at.run()
    assert not at.exception
    at.get('file_uploader')[0].upload(
        spr_path.name, spr_path.read_bytes(),
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    )
    t0 = time.perf_counter(); at.run(); results['spr_app_upload_seconds'] = time.perf_counter() - t0
    assert not at.exception and len(at.get('plotly_chart')) == 1
    spec = json.loads(at.get('plotly_chart')[0].proto.spec)
    assert len(spec.get('data', [])) == 4
    assert all('lines' in trace.get('mode', '') for trace in spec['data'])
    assert all(len(trace.get('x', [])) == 12 for trace in spec['data'])
    assert len(spec.get('layout', {}).get('shapes', [])) == 0

    at.radio[0].set_value('Dark')
    t0 = time.perf_counter(); at.run(); results['theme_rerun_seconds'] = time.perf_counter() - t0
    assert not at.exception and results['theme_rerun_seconds'] < 8.0

    for box in at.selectbox:
        if box.label == 'Plot style':
            box.set_value('Overlay actual values')
            break
    t0 = time.perf_counter(); at.run(); results['plot_rerun_seconds'] = time.perf_counter() - t0
    assert not at.exception and results['plot_rerun_seconds'] < 8.0

print('Production Test Dashboard v75 self-test: PASS')
for key, value in results.items():
    print(f'{key}: {value:.3f}')
