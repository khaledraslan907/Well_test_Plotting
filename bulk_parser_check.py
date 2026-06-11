"""Bulk parser validation for the TMU dashboard.

Use this before giving the dashboard to users with unknown file formats.
It scans a folder recursively, tries to parse every supported file, and writes a CSV report.

Examples:
    python bulk_parser_check.py "C:/Users/You/Desktop/TMU Files"
    python bulk_parser_check.py "C:/Users/You/Desktop/TMU Files" --out parser_report.csv

Supported extensions: .xlsx, .xls, .csv, .txt, .docx, .pdf
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

from tmu_parser import available_numeric_columns, column_label, load_tabular_file

SUPPORTED_EXTS = {".xlsx", ".xls", ".csv", ".txt", ".docx", ".pdf"}


class NamedBytesIO(io.BytesIO):
    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        super().__init__(self.path.read_bytes())
        self.name = self.path.name


def iter_supported_files(paths: Iterable[str | os.PathLike[str]]):
    for p in paths:
        path = Path(p)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.suffix.lower() in SUPPORTED_EXTS:
                    yield child


def describe_file(path: Path) -> list[dict]:
    uploaded = NamedBytesIO(path)
    rows: list[dict] = []

    try:
        tables = load_tabular_file(uploaded)
    except Exception as exc:
        return [{
            "file": str(path),
            "status": "ERROR",
            "reason": str(exc),
            "table_no": "",
            "sheet": "",
            "rows": 0,
            "well": "",
            "start": "",
            "end": "",
            "numeric_count": 0,
            "detected_columns": "",
            "raw_fallback_columns": "",
        }]

    if not tables:
        return [{
            "file": str(path),
            "status": "NOT_PARSED",
            "reason": "No usable time-series table detected. Need at least date/time plus numeric readings, or recognizable TMU message text.",
            "table_no": "",
            "sheet": "",
            "rows": 0,
            "well": "",
            "start": "",
            "end": "",
            "numeric_count": 0,
            "detected_columns": "",
            "raw_fallback_columns": "",
        }]

    for i, df in enumerate(tables, start=1):
        numeric = available_numeric_columns(df)
        raw_numeric = [c for c in numeric if str(c).startswith("raw__")]
        dt = pd.to_datetime(df["datetime"], errors="coerce") if "datetime" in df.columns else pd.Series(dtype="datetime64[ns]")
        wells = ", ".join(sorted({str(w) for w in df.get("well", pd.Series(dtype=str)).dropna().unique()})) or "Unknown"
        sheet = str(df["sheet"].iloc[0]) if "sheet" in df.columns and len(df) else ""
        rows.append({
            "file": str(path),
            "status": "OK" if not raw_numeric else "OK_WITH_RAW_FALLBACK",
            "reason": "" if not raw_numeric else "Some numeric columns were plotted with raw names because their headers are not in the alias map yet.",
            "table_no": i,
            "sheet": sheet,
            "rows": int(len(df)),
            "well": wells,
            "start": dt.min().strftime("%Y-%m-%d %H:%M") if dt.notna().any() else "",
            "end": dt.max().strftime("%Y-%m-%d %H:%M") if dt.notna().any() else "",
            "numeric_count": len(numeric),
            "detected_columns": "; ".join(column_label(c) for c in numeric),
            "raw_fallback_columns": "; ".join(column_label(c) for c in raw_numeric),
        })
    return rows


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Validate TMU dashboard parser against files/folders.")
    parser.add_argument("paths", nargs="+", help="Files or folders to scan recursively.")
    parser.add_argument("--out", default="parser_validation_report.csv", help="CSV report path.")
    args = parser.parse_args(argv)

    files = list(iter_supported_files(args.paths))
    if not files:
        print("No supported files found. Supported: " + ", ".join(sorted(SUPPORTED_EXTS)))
        return 2

    all_rows: list[dict] = []
    for path in files:
        print(f"Checking: {path}")
        all_rows.extend(describe_file(path))

    report = pd.DataFrame(all_rows)
    report.to_csv(args.out, index=False, encoding="utf-8-sig")

    total = len(files)
    ok = int(report["status"].isin(["OK", "OK_WITH_RAW_FALLBACK"]).sum())
    raw = int((report["status"] == "OK_WITH_RAW_FALLBACK").sum())
    failed = int(report["status"].isin(["NOT_PARSED", "ERROR"]).sum())

    print("\nSummary")
    print(f"  Files checked: {total}")
    print(f"  Parsed tables: {ok}")
    print(f"  Tables with raw fallback columns: {raw}")
    print(f"  Files not parsed / errors: {failed}")
    print(f"  Report saved: {args.out}")

    if raw:
        print("\nAction: review 'raw_fallback_columns' in the CSV. Add those header aliases to best_canonical_name() for cleaner labels.")
    if failed:
        print("\nAction: review NOT_PARSED/ERROR rows. These files may be blank, scanned PDFs/images, summary-only reports, or unsupported layouts.")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
