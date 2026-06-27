"""Regression tests for Production Test Dashboard v74.

Run from the project folder:
    python self_test_v74.py
"""
from __future__ import annotations

import ast
import io
import math
from pathlib import Path

import pandas as pd
import xlsxwriter

import smart_tabular_v73 as smart
import tmu_parser as parser
import tmu_parser_legacy as legacy

assert parser.PARSER_BUILD_ID.startswith("v74-")
assert smart._gas_unit("Formation Gas | MM SCF/D") == "mmscfd"
assert smart._gas_unit("Gas Rate | MSCF/D") == "mscfd"
assert smart._gas_unit("Gas Rate | SCF/D") == "scfd"
assert smart.infer_field("SL").key == "stroke_length_in"
assert smart.infer_field("SPM").key == "stroke_rate_spm"
assert smart.infer_field("Peakload").key == "peak_load_lbf"
assert smart.infer_field("Minload").key == "minimum_load_lbf"
assert smart.infer_field("Casing Pressure (bar)").key == "casing_pressure_psi"

# Public facade must not repeat function definitions.
tree = ast.parse(Path("tmu_parser.py").read_text(encoding="utf-8"))
seen = set()
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        assert node.name not in seen, f"duplicate public function: {node.name}"
        seen.add(node.name)

# The exact v73 plotting failure: boolean audit fields were being offered as
# numeric curves. pandas keeps pd.to_numeric(bool) as bool and bool max-min
# raises TypeError. v74 excludes these fields from plotting.
flags = pd.DataFrame({
    "datetime": pd.date_range("2026-06-01", periods=3, freq="h"),
    "well": ["A-1"] * 3,
    "gross_rate_bpd": [100.0, 101.0, 102.0],
    "gas_formation_derived": [False, True, False],
    "n2_rate_derived": [False, False, True],
    "total_gas_derived": [False, False, False],
})
plot_columns = parser.available_numeric_columns(flags)
assert plot_columns == ["gross_rate_bpd"], plot_columns

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

# Previously unseen layout: multi-row header, time-only values, midnight
# rollover, metric units and one unknown device channel.
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

class Uploaded(io.BytesIO):
    def __init__(self, path: Path):
        super().__init__(path.read_bytes())
        self.name = path.name

root = Path("/mnt/data")
for srp_name in ("SPR(1).xlsx", "SPR.xlsx"):
    srp = root / srp_name
    if srp.exists():
        frame = parser.load_tabular_file(Uploaded(srp))[0]
        assert len(frame) == 12
        assert {"stroke_length_in", "stroke_rate_spm", "peak_load_lbf", "minimum_load_lbf"}.issubset(frame.columns)
        assert not {"gas_formation_derived", "n2_rate_derived", "total_gas_derived"}.intersection(parser.available_numeric_columns(frame))
        break

b15 = root / "6-B15-42 (Dasco 27) (9-6-2026)(3).xlsx"
if b15.exists():
    frame = parser.load_tabular_file(Uploaded(b15))[0]
    assert len(frame) == 10
    assert {"gross_rate_bpd", "oil_rate_stbd", "water_rate_bpd", "gas_rate_mmscfd"}.issubset(frame.columns)
    assert not {"gas_formation_derived", "n2_rate_derived", "total_gas_derived"}.intersection(parser.available_numeric_columns(frame))

print("Production Test Dashboard v74 self-test: PASS")
