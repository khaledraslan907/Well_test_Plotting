"""Small installation sanity test for TMU Dashboard v66.

Run with: python self_test_v66.py
This does not require user files. It checks the parser import, scientific
notation, well normalization, repeated WhatsApp report splitting and duplicate
coalescing.
"""
from __future__ import annotations

import pandas as pd
import tmu_parser as parser

assert parser.PARSER_BUILD_ID_V66.startswith("v66-")
assert abs(parser.extract_number("9.827E-2 MMSCF/D") - 0.09827) < 1e-9
assert parser.clean_well_name_value("BED_16 C6-9") == "B16C6-9"

text = """*PICO TMU-04*
*Date 22/06/2026*
*Well name* : B16C6-9
*Time* = 7:00
*W.H.P* = 200 PSI
*Gross rate* = 522.9 BBL/D
*gas rate* = Low gas MMSCF/D

*PICO TMU-04*
*Date 22/06/2026*
*Well name* : B16-C6-9
*Time* = 8:00
*W.H.P* = 200 PSI
*Gross rate* = 518.6 BBL/D
*gas rate* = Low gas MMSCF/D
"""
wa = parser.parse_many_tmu_messages(text)
assert len(wa) == 2
assert wa["well"].eq("B16C6-9").all()
assert wa["gas_rate_status"].str.lower().eq("low gas").all()

partial = pd.DataFrame(
    {
        "well": ["B16C6-9", "B16-C6-9"],
        "datetime": ["2026-06-22 07:00", "2026-06-22 07:00"],
        "gross_rate_bpd": [522.9, None],
        "motor_ama_amp": [None, 25.2],
        "source": ["test.xlsx", "device.csv"],
    }
)
merged = parser.merge_duplicate_test_rows_v53(partial)
assert len(merged) == 1
assert float(merged.iloc[0]["gross_rate_bpd"]) == 522.9
assert float(merged.iloc[0]["motor_ama_amp"]) == 25.2

print("TMU Dashboard v66 self-test: PASS")
