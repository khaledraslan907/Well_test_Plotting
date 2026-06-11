"""Quick parser QA utility for the TMU dashboard.

Usage:
    python parser_selftest.py "7-B15-42(10-6-2026).xlsx" "New Microsoft Excel Worksheet.xlsx"

It prints which sheets were accepted, the detected datetime range, well name, and
user-facing column labels. This helps confirm new Excel formats before using the
Streamlit app.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pandas as pd

from tmu_parser import available_numeric_columns, column_label, load_tabular_file


class NamedBytesIO(io.BytesIO):
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        super().__init__(self.path.read_bytes())
        self.name = self.path.name


def describe_file(path: str | os.PathLike[str]) -> None:
    uploaded = NamedBytesIO(path)
    try:
        tables = load_tabular_file(uploaded)
    except Exception as exc:
        print(f"\n❌ {uploaded.name}: parser error: {exc}")
        return

    if not tables:
        print(f"\n⚠️  {uploaded.name}: no usable time-series table detected")
        return

    print(f"\n✅ {uploaded.name}: {len(tables)} table(s) accepted")
    for i, df in enumerate(tables, start=1):
        numeric = available_numeric_columns(df)
        dt = pd.to_datetime(df["datetime"], errors="coerce") if "datetime" in df.columns else pd.Series(dtype="datetime64[ns]")
        wells = ", ".join(sorted({str(w) for w in df.get("well", pd.Series(dtype=str)).dropna().unique()})) or "Unknown"
        print(f"  Table {i}")
        print(f"    rows: {len(df)}")
        print(f"    sheet: {str(df['sheet'].iloc[0]) if 'sheet' in df.columns and len(df) else ''}")
        print(f"    well: {wells}")
        if dt.notna().any():
            print(f"    datetime: {dt.min():%Y-%m-%d %H:%M} → {dt.max():%Y-%m-%d %H:%M}")
        print(f"    detected columns ({len(numeric)}):")
        for col in numeric:
            marker = " [RAW FALLBACK]" if str(col).startswith("raw__") else ""
            print(f"      - {column_label(col)}{marker}")


def main(argv: list[str]) -> int:
    if not argv:
        print("Pass one or more Excel/CSV/TXT/DOCX/PDF paths to test.")
        return 2
    for path in argv:
        describe_file(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
