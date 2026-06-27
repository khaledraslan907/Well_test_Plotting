"""Regression checks for v79 direct WhatsApp ZIP parsing."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd

import tmu_parser as parser


class Uploaded(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


assert parser.PARSER_BUILD_ID.startswith("v79-")

# Reproduce the failure mode: a WhatsApp ZIP with only _chat.txt plus an
# unsupported voice note.  It must parse the chat and ignore the audio.
chat = """[12/06/2026, 18:03:11] Field Engineer: *PICO TMU-4*
*Date  : 12-06-2026*
*Well name  : S8-58*
*Time = 18:00*
*Choke = 100%*
*W.H.P = 20 PSI*
*Sep. P = 10 PSI*
*T. Gas rate = 0.351 MMSCF/D*
*Gas formation = 0.000 MMSCF/D*
*Gross rate = 57.4 BBL/D*
*Oil rate = 16.5 STB/D*
*Water rate = 40.9 BBL/D*
*BS&W = 71%*
*Salinity = 30 KPPM of NaCl*
*Pumping p = 940 psi*
[12/06/2026, 18:35:00] Field Engineer: *PICO TMU-4*
*Date  : 12-06-2026*
*Well name  : S8-58*
*Time = 18:30*
*Choke = 100%*
*W.H.P = 30 PSI*
*Sep. P = 20 PSI*
*T. Gas rate = 0.532 MMSCF/D*
*Gas formation = 0.000 MMSCF/D*
*Gross rate = 119.7 BBL/D*
*Oil rate = 26.1 STB/D*
*Water rate = 93.6 BBL/D*
*BS&W = 78%*
*Salinity = 38 KPPM of NaCl*
"""
mem = io.BytesIO()
with zipfile.ZipFile(mem, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("_chat.txt", chat.encode("utf-8"))
    zf.writestr("00003066-AUDIO-2026-06-13-12-03-50.opus", b"not-audio-test")

tables = parser.load_tabular_file(Uploaded(mem.getvalue(), "WhatsApp Chat - test.zip"), parse_images=False)
assert len(tables) == 1
assert len(tables[0]) == 2
assert tables[0]["well"].eq("S8-58").all()
assert pd.to_datetime(tables[0]["datetime"]).min() == pd.Timestamp("2026-06-12 18:00")
assert pd.to_numeric(tables[0]["gross_rate_bpd"], errors="coerce").tolist() == [57.4, 119.7]

# Optional validation against the user's field export when it is available in
# the external test environment.  The private ZIP is not bundled in deployment.
field_zip = Path("/mnt/data/WhatsApp Chat - Sitra8-58 clean & test(2).zip")
if field_zip.exists():
    field_tables = parser.load_tabular_file(
        Uploaded(field_zip.read_bytes(), field_zip.name),
        parse_images=False,
        max_ocr_images=1000,
    )
    field = field_tables[0]
    assert len(field) == 74
    assert field["well"].eq("S8-58").all()
    assert pd.to_datetime(field["datetime"]).min() == pd.Timestamp("2026-06-12 00:00")
    assert pd.to_datetime(field["datetime"]).max() == pd.Timestamp("2026-06-16 15:00")
    assert field.duplicated(["well", "datetime"], keep=False).sum() == 0

print("Production Test Dashboard v79 WhatsApp ZIP self-test: PASS")
