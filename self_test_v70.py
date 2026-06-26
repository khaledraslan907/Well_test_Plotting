"""Sanity and field-sample validation for TMU Dashboard v70.

Run from the project directory:
    python self_test_v70.py

The three CTU image assertions run when 12.jpg, 13.jpg and 14.jpg are found in
/mnt/data or beside this script. They are skipped on a clean deployment where
those private field photos are not present.
"""
from __future__ import annotations

import io
import math
from pathlib import Path

import pandas as pd

import tmu_parser as parser
import tmu_parser_legacy as legacy

assert parser.PARSER_BUILD_ID.startswith("v70-")
assert legacy.CTU_OCR_BUILD_ID_V70.startswith("v70-")

# Gas balance: never plot a negative Formation Gas. If total is less than the
# injected N2 reading, Formation Gas is zero and the source conflict is audited.
gas = pd.DataFrame({
    "datetime": pd.to_datetime(["2026-06-23 13:30", "2026-06-23 17:30", "2026-06-23 19:00"]),
    "well": ["B16-25"] * 3,
    "gas_rate_mmscfd": [0.335704, 0.707169, 0.953062],
    "n2_rate_mmscfd": [0.576, 0.576, 0.576],
    "gas_formation_mmscfd": [-0.240296, 0.131169, 9.37],
})
reconciled = parser._reconcile_gas_balance_v70(gas)
assert reconciled["gas_formation_mmscfd"].ge(0).all()
assert math.isclose(reconciled.loc[0, "gas_formation_mmscfd"], 0.0, abs_tol=1e-9)
assert math.isclose(reconciled.loc[1, "gas_formation_mmscfd"], 0.131169, abs_tol=1e-6)
assert math.isclose(reconciled.loc[2, "gas_formation_mmscfd"], 0.377062, abs_tol=1e-6)
assert reconciled["review_required"].fillna(False).any()

# Decimal OCR candidates must not be diluted by fake /10, /100 alternatives.
normalised = legacy._normalise_ctu_candidate_v70("ctu_reel_speed_ftmin", 0.01, "0.01")
assert normalised == [(0.01, 0.0)]

# Direct image formats are supported by the same OCR parser as ZIP members.
for suffix in {"jpg", "jpeg", "png", "webp"}:
    assert suffix in legacy.IMAGE_SUFFIXES

expected = {
    "12.jpg": [15512, 51, 178.56, 2188.97, 9473.8, 0.01, 0, 400, 0, 81925],
    "13.jpg": [22907, 15, 226.98, 1363.61, 12098.7, 0, 0, 448, 0, 535176],
    "14.jpg": [23891, 13, 200.34, 1362.09, 12098.7, 0, 0, 458, 0, 547799],
}
columns = [
    "ctu_weight_lbf", "ctu_lt_weight_lbf", "ctu_wellhead_pressure_psi",
    "ctu_circulation_pressure_psi", "ctu_reel_depth_ft", "ctu_reel_speed_ftmin",
    "ctu_fluid_rate_bpm", "ctu_n2_rate_scfm", "ctu_fluid_total_bbl", "ctu_n2_total_scf",
]
for filename, values in expected.items():
    candidates = [Path("/mnt/data") / filename, Path(__file__).resolve().parent / filename]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        continue
    uploaded = io.BytesIO(path.read_bytes())
    uploaded.name = filename
    frame = parser.parse_ctu_all_data_screen_image(uploaded, source_name=filename)
    assert not frame.empty
    actual = [float(frame.iloc[0][column]) for column in columns]
    assert all(math.isclose(a, e, abs_tol=1e-6) for a, e in zip(actual, values)), (filename, actual)
    assert bool(frame.iloc[0]["review_required"])

print("TMU Dashboard v70 self-test: PASS")
