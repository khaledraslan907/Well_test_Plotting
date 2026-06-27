PRODUCTION TEST DASHBOARD v73 — SMART ENSEMBLE PARSER
=====================================================

Why parser regressions happened
-------------------------------
The historical parser was built by appending successive patches.  The old
module contained repeated definitions of core functions such as
load_tabular_file(), canonical_header(), parse_tmu_message() and
_postprocess_table().  In Python, the last definition wins.  A later patch
could therefore unintentionally bypass an earlier correction even when both
pieces of code remained in the file.

v73 architecture
----------------
1. tmu_parser.py
   A small stable public facade.  It contains one definition of every public
   function used by the Streamlit app.

2. tmu_parser_compat.py
   The complete v72 parser retained as a compatibility engine for the proven
   WhatsApp, OCR, PDF and specialist-template paths.

3. smart_tabular_v73.py
   A new independent adaptive parser for Excel and CSV.  It searches multiple
   header rows and table positions, infers date/time from both header meaning
   and cell values, understands petroleum units, and preserves unfamiliar
   numeric channels instead of rejecting a new test layout.

4. Ensemble selection
   Excel/CSV files are interpreted by both engines.  The result is selected by
   timestamp validity, recognized engineering columns, physical plausibility,
   chronological order and review-warning rate.  Distinct valid sheets are
   retained.

New-template behavior
---------------------
- Multi-row and merged headers are combined automatically.
- DateTime, Date + Time, a Date column containing time, and time-only series
  with a date in metadata/file name are supported.
- Midnight rollover is repaired for time-only series.
- Common TMU, MPFM, ESP, CTU and SRP aliases are recognized.
- "MM SCF/D" is explicitly treated as MMSCF/D.
- Unknown numeric columns are kept as raw_<header> and remain plottable.
- Source values are not silently changed to force engineering equations.
- Missing gas components may be derived; supplied conflicting values are kept
  and clearly flagged only when the difference is material.

Deployment
----------
Replace the full repository with this package.  Do not deploy only app.py or
only tmu_parser.py because v73 requires tmu_parser_compat.py and
smart_tabular_v73.py beside it.
