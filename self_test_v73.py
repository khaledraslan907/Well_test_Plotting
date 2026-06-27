"""Regression and adaptability tests for Production Test Dashboard v73."""
from __future__ import annotations

import ast
import io
import math
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import xlsxwriter

import smart_tabular_v73 as smart
import tmu_parser as parser
import tmu_parser_legacy as legacy

assert parser.PARSER_BUILD_ID.startswith("v73-")
assert smart._gas_unit("Formation Gas | MM SCF/D") == "mmscfd"
assert smart._gas_unit("Gas Rate | MSCF/D") == "mscfd"
assert smart._gas_unit("Gas Rate | SCF/D") == "scfd"
assert smart.infer_field("SL").key == "stroke_length_in"
assert smart.infer_field("SPM").key == "stroke_rate_spm"
assert smart.infer_field("Peakload").key == "peak_load_lbf"
assert smart.infer_field("Minload").key == "minimum_load_lbf"
assert smart.infer_field("Casing Pressure (bar)").key == "casing_pressure_psi"

# Public facade must not repeat function definitions; the historical file is
# isolated behind tmu_parser_compat.py instead.
tree = ast.parse(Path("tmu_parser.py").read_text(encoding="utf-8"))
seen = set()
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        assert node.name not in seen, f"duplicate public function: {node.name}"
        seen.add(node.name)

# Non-destructive gas checks: supplied conflicting values stay supplied.
gas = pd.DataFrame({
    "datetime": pd.to_datetime(["2026-06-23 13:30"]),
    "well": ["B16-25"],
    "gas_rate_mmscfd": [0.335704],
    "n2_rate_mmscfd": [0.576],
    "gas_formation_mmscfd": [-0.240296],
})
checked = parser._engineering_checks(gas)
assert math.isclose(float(checked.loc[0, "gas_rate_mmscfd"]), 0.335704)
assert math.isclose(float(checked.loc[0, "n2_rate_mmscfd"]), 0.576)
assert math.isclose(float(checked.loc[0, "gas_formation_mmscfd"]), -0.240296)
assert bool(checked.loc[0, "review_required"])

# EXPRO specialist regression: one choke column, no field shift.
expro_text = """
Data & Events
QOil QWat
Well No BAHGA - 9
01/06/2023
10:00:00 128.000 0.000 115.000 49.000 105.204 47.611 260.157 616.482 156.590 0.446 642.915 156.590 0.050 20.256 773.07 0.860 1.070 7.000 150000 0.970 4.000 6.000 723.55 91.701 48.000
"""
expro = legacy.parse_expro_mpfm_text(expro_text, source_name="EXPRO_sample.pdf")
assert len(expro) == 1
row = expro.iloc[0]
assert math.isclose(float(row["oil_rate_stbd"]), 616.482)
assert math.isclose(float(row["water_rate_bpd"]), 156.590)
assert math.isclose(float(row["gross_rate_bpd"]), 773.07)
assert math.isclose(float(row["bsw_pct"]), 20.256)

# Build a previously unseen two-row template in memory. It uses a time-only
# series, a date in the file name, metric units, MM SCF/D, and a new sensor.
buffer = io.BytesIO()
wb = xlsxwriter.Workbook(buffer, {"in_memory": True})
ws = wb.add_worksheet("Future Test")
ws.write("A1", "Well Name: B7-22")
parents = ["Timing", "Pressure", "Gas", "Gas", "Liquid", "Liquid", "Quality", "New Device"]
subs = ["Time", "WHP bar", "Total Gas MM SCF/D", "Formation Gas MM SCF/D", "Oil m3/d", "Water m3/d", "Water Cut fraction", "Foam Stability"]
for c, value in enumerate(parents):
    ws.write(2, c, value)
for c, value in enumerate(subs):
    ws.write(3, c, value)
for r, value in enumerate(["23:00", "23:30", "00:00", "00:30", "01:00"]):
    ws.write(4 + r, 0, value)
    ws.write_number(4 + r, 1, 10 + r)
    ws.write_number(4 + r, 2, 0.8 + r * 0.01)
    ws.write_number(4 + r, 3, 0.8 + r * 0.01)
    ws.write_number(4 + r, 4, 10)
    ws.write_number(4 + r, 5, 20)
    ws.write_number(4 + r, 6, 0.6667)
    ws.write_number(4 + r, 7, 100 + r)
wb.close()
buffer.seek(0)
buffer.name = "B7-22 (20-06-2026) future layout.xlsx"
future = parser.load_tabular_file(buffer)[0]
assert len(future) == 5
assert future["parser_engine"].eq(smart.ENGINE_ID).all()
assert pd.Timestamp(future.iloc[0]["datetime"]) == pd.Timestamp("2026-06-20 23:00")
assert pd.Timestamp(future.iloc[2]["datetime"]) == pd.Timestamp("2026-06-21 00:00")
assert math.isclose(float(future.iloc[0]["gas_rate_mmscfd"]), 0.8)
assert math.isclose(float(future.iloc[0]["whp_psi"]), 145.037738, rel_tol=1e-6)
assert math.isclose(float(future.iloc[0]["bsw_pct"]), 66.67, rel_tol=1e-6)
assert "raw_foam_stability" in future.columns

# Optional field files supplied during development.
class Uploaded(io.BytesIO):
    def __init__(self, path: Path):
        super().__init__(path.read_bytes())
        self.name = path.name

root = Path("/mnt/data")
b3 = root / "B3 C18-7(3).xlsx"
if b3.exists():
    frame = parser.load_tabular_file(Uploaded(b3))[0]
    assert len(frame) == 74
    assert frame["data_quality_note"].fillna("").eq("").all()
    assert math.isclose(float(frame.iloc[0]["gas_formation_mmscfd"]), 0.9638425867184576, abs_tol=1e-12)

srp = root / "SPR.xlsx"
if srp.exists():
    frame = parser.load_tabular_file(Uploaded(srp))[0]
    assert len(frame) == 12
    assert {"stroke_length_in", "stroke_rate_spm", "peak_load_lbf", "minimum_load_lbf"}.issubset(frame.columns)

print("Production Test Dashboard v73 smart parser self-test: PASS")
