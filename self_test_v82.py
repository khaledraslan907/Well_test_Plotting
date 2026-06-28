from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

import tmu_parser as parser


class UploadedBytes(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


assert parser.PARSER_BUILD_ID.startswith("v82-")

message = """UNIT TMU-01
Date : 28-06-2026
Well name : WELL-001
Time = 10:30
Choke = 25%
W.H.P = 850 PSI
Gross rate = 1200 BBL/D
Oil rate = 900 STB/D
Water rate = 300 BBL/D
"""
parsed = parser.parse_many_tmu_messages(message, source_name="Pasted message")
assert not parsed.empty
assert len(parsed) == 1
assert float(parsed.iloc[0]["whp_psi"]) == 850.0
assert float(parsed.iloc[0]["gross_rate_bpd"]) == 1200.0

# A generic unfamiliar workbook must still be detected without private fixtures.
raw = pd.DataFrame(
    {
        "Date and Time": pd.date_range("2026-06-28 08:00", periods=4, freq="30min"),
        "Wellhead Pressure psi": [800, 805, 810, 815],
        "Gross Liquid BBL/D": [1100, 1110, 1120, 1130],
        "New Sensor Index": [1.1, 1.2, 1.3, 1.4],
    }
)
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
    raw.to_excel(writer, index=False, sheet_name="Data")
workbook = UploadedBytes(buffer.getvalue(), "generic_test.xlsx")
tables = parser.load_tabular_file(workbook)
assert tables and sum(len(t) for t in tables) == 4
assert any("whp_psi" in t.columns for t in tables)

print("Production Test Dashboard v82 self-test: PASS")
