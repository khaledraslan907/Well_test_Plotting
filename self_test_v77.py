"""Regression checks for v77 complete-test parsing, OCR and smart segmentation."""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

import tmu_parser as parser

assert parser.PARSER_BUILD_ID.startswith("v77-")

# Sparse daily SRP surveillance remains one connected trend.
srp = pd.DataFrame({
    "source": ["SPR.xlsx"] * 3,
    "sheet": ["SRP"] * 3,
    "well": ["Unknown"] * 3,
    "datetime": pd.to_datetime(["2026-06-14", "2026-06-15", "2026-06-16"]),
    "stroke_length_in": [199, 200, 155],
    "stroke_rate_spm": [4.0, 4.0, 4.0],
    "peak_load_lbf": [15000, 15100, 14900],
    "min_load_lbf": [6000, 6100, 5900],
})
srp = parser.assign_test_ids(srp, gap_hours=12)
assert srp["test_id"].nunique() == 1

# Production readings split after a real inactive gap.
prod = pd.DataFrame({
    "source": ["test.xlsx"] * 4,
    "sheet": ["Test"] * 4,
    "well": ["S8-58"] * 4,
    "datetime": pd.to_datetime([
        "2026-06-12 18:00", "2026-06-13 00:30",
        "2026-06-15 01:30", "2026-06-15 02:00",
    ]),
    "gross_rate_bpd": [100, 110, 120, 130],
})
prod = parser.assign_test_ids(prod, gap_hours=12)
assert prod["test_id"].nunique() == 2

# Debug/OCR metadata never appears as a plotting signal.
meta = pd.DataFrame({
    "datetime": pd.to_datetime(["2026-01-01", "2026-01-02"]),
    "ctu_reel_depth_ft": [1000.0, 1100.0],
    "screen_area_ratio": [0.5, 0.6],
    "ocr_confidence": [0.8, 0.9],
    "ocr_raw__ctu_reel_depth_ft": ["1000", "1100"],
    "ocr_conf__ctu_reel_depth_ft": [0.8, 0.9],
})
features = parser.available_numeric_columns(meta)
assert "ctu_reel_depth_ft" in features
assert not any(c.startswith("screen_") or c.startswith("ocr_") for c in features)


class Uploaded(io.BytesIO):
    def __init__(self, path: Path):
        super().__init__(path.read_bytes())
        self.name = path.name


s8 = Path("/mnt/data/12-S8-58 (12-06-2026)(4).xlsx")
image = Path("/mnt/data/WhatsApp Image 2026-06-16 at 14.35.12.jpeg")
if s8.exists():
    table = parser.load_tabular_file(Uploaded(s8), parse_images=False)[0]
    assert len(table) == 92
    assert pd.Timestamp(table["datetime"].max()) == pd.Timestamp("2026-06-16 16:00:00")
    assert table["test_id"].nunique() == 2

if image.exists():
    ocr = parser.load_tabular_file(Uploaded(image))[0]
    expected = {
        "ctu_weight_lbf": 21024.0,
        "ctu_lt_weight_lbf": -1.0,
        "ctu_wellhead_pressure_psi": 29.16,
        "ctu_circulation_pressure_psi": 693.99,
        "ctu_reel_depth_ft": 10149.6,
        "ctu_reel_speed_ftmin": -0.01,
        "ctu_fluid_rate_bpm": 0.0,
        "ctu_n2_rate_scfm": 0.0,
        "ctu_fluid_total_bbl": 0.0,
        "ctu_n2_total_scf": 0.0,
    }
    for field, value in expected.items():
        assert abs(float(ocr.iloc[0][field]) - value) < 1e-6, (field, ocr.iloc[0][field], value)

if s8.exists() and image.exists():
    table = parser.load_tabular_file(Uploaded(s8), parse_images=False)[0]
    ocr = parser.load_tabular_file(Uploaded(image))[0]
    combined = parser.auto_link_ocr_rows_by_time_context(pd.concat([table, ocr], ignore_index=True, sort=False))
    row = combined[combined["source_type"].astype(str).str.contains("ocr", case=False, na=False)].iloc[0]
    assert row["well"] == "S8-58"
    assert row["test_id"] == "S8-58_20260615_0130"
    assert row["link_status"] == "ocr_auto_linked_by_timestamp"

print("Production Test Dashboard v77 parser self-test: PASS")
