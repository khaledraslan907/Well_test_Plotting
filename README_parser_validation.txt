TMU Dashboard Parser Validation
===============================

This package includes a stronger parser for mixed TMU / well-test files.

Supported upload types
----------------------
- Excel: .xlsx, .xls
- CSV: .csv
- Text / WhatsApp messages: .txt and pasted text
- Word: .docx
- PDF: .pdf, including EXPRO MPFM Data & Events reports when text is extractable

What the parser does
--------------------
1. Tries multiple interpretations of each sheet/table.
2. Detects DATE, TIME and DATETIME columns from both headers and actual cell values.
3. Combines date + time correctly, including midnight rollover.
4. Accepts date-only production-history files as valid time series at 00:00.
5. Uses a canonical alias map to rename common fields such as WHP, FLP, Sep P, Gas Rate, Oil Rate, Water Rate, BS&W, Salinity, MPFM QOil/QWat/QGas, Pump P, N2 Rate, etc.
6. Rejects non-data text/PDF files that do not contain numeric operational readings with date/time evidence.
7. Keeps unknown numeric fields only as Raw columns when the template is genuinely unknown; known templates avoid confusing Raw: Psig / Raw: Column labels.

Important limitation
--------------------
No parser can guarantee every possible customer format. The parser needs extractable text/table data and at least a date/time or date-only series plus numeric readings. Scanned PDFs/images, protected workbooks, corrupted files, or summary-only documents may still need manual conversion or a new alias rule.

Bulk validation
---------------
Run this before deployment or when receiving a new customer file batch:

python bulk_parser_check.py "C:/path/to/customer/files" --out parser_validation_report.csv

Statuses:
- OK: parsed and column names mapped correctly.
- OK_WITH_RAW_FALLBACK: parsed, but some headers need alias-map additions for cleaner names.
- NOT_PARSED: no usable time-series was found.
- ERROR: file read/library issue.
