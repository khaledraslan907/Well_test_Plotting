TMU Dashboard v66 — Unified Robust Ingestion
=============================================

Purpose
-------
This release consolidates the fixes from v43–v65 into one deployment package.
It uses a deterministic ingestion layer for Excel, CSV and pasted WhatsApp
reports, while retaining the mature ZIP/PDF/DOCX/image OCR routines in
`tmu_parser_legacy.py`.

Important deployment rule
-------------------------
Replace the whole repository package. Do not copy only app.py or only
`tmu_parser.py`. The following files must stay together:

- app.py
- tmu_parser.py
- tmu_parser_legacy.py
- requirements.txt
- runtime.txt
- packages.txt
- whatsapp_webhook_fastapi.py

Supported/handled cases
-----------------------
- Standard and irregular TMU Excel reports with multi-row headers.
- Hidden and far-right Pumping Pressure columns, including S8-58 column EG.
- Workbooks with accidental million-row/XFC formatting without loading the
  formatted tail into memory.
- Excel dates, separate dates/times, midnight rollover, 1900-dated time cells,
  isolated wrong dates and floating-second errors.
- Scientific notation such as 9.827E-2.
- Production-test and device-only tables (AMA, Pi, Pd, frequency, Tm, etc.).
- CSV encodings: UTF-8, UTF-8 BOM, UTF-16, Windows-1252 and Latin-1.
- CSV delimiters: comma, semicolon, tab and pipe.
- Dashboard exports with Raw: prefixes and saved source metadata.
- Repeated pasted WhatsApp TMU reports and WhatsApp export text.
- Non-numeric gas statuses such as Low gas, preserved as status/note rather
  than converted to zero.
- Equivalent well-name separators, e.g. B16C6-9, B16-C6-9 and BED_16 C6-9.
- Repeated/incomplete/final reports merged by normalized well + minute;
  populated values are coalesced and blanks never erase measurements.
- Choke source units kept separately for the app's user-selectable conversion.
- PDF/DOCX/ZIP/image extraction retained through the legacy module; image-only
  documents require OCR to be enabled.

Performance and stability
-------------------------
- Python runtime is pinned to 3.11 to avoid Python 3.14/Pandas dtype regressions
  previously seen in Streamlit Cloud.
- Excel XML is read with bounded rows/columns and real-cell detection.
- Streamlit caches parsed files by bytes + parser build ID.
- Changing chart controls does not re-read uploaded workbooks.
- OCR remains disabled by default because it is the most expensive operation.

Local start
-----------
1. Install Python 3.11.
2. Run: pip install -r requirements.txt
3. Run: streamlit run app.py

Streamlit Cloud
---------------
Upload/replace all files, commit, and reboot the app. v66 changes the parser
build ID, so old cached parser results are invalidated automatically.

Scope note
----------
The regression suite covers all supplied Excel/CSV/WhatsApp samples and the
existing ZIP parsing path. A file still needs at least one valid timestamp and
one numeric reading. Image-only PDFs/screenshots require OCR; a document with
no recoverable time-series content is correctly rejected rather than guessed.
