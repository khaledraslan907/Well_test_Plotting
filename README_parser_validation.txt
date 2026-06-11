Parser validation workflow
==========================

This parser is robust, but no parser can guarantee perfect parsing for every possible customer file.
Use these checks before deployment:

1) Check one or two files:
   python parser_selftest.py "7-B15-42(10-6-2026).xlsx"

2) Check a full folder recursively:
   python bulk_parser_check.py "C:/path/to/customer/files" --out parser_validation_report.csv

3) Read parser_validation_report.csv:
   - OK = parsed with recognized standard column names.
   - OK_WITH_RAW_FALLBACK = parsed, but at least one header is unknown. The column is still usable/plotable.
   - NOT_PARSED = no usable date/time + numeric time-series was found.
   - ERROR = library/file read problem.

4) For OK_WITH_RAW_FALLBACK:
   Add the new header wording to best_canonical_name() in tmu_parser.py so it receives a clean label.

Supported input types:
   .xlsx, .xls, .csv, .txt, .docx, .pdf, and pasted WhatsApp TMU messages.

Expected limitation:
   Scanned/image-only PDFs, photos, protected/corrupted workbooks, and files without any date/time plus numeric readings cannot be parsed reliably without OCR or manual mapping.
