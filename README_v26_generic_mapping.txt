TMU Dashboard v26 - Generic Column Mapping

What changed in v26
-------------------
1) Added a Column mapping review / teach new names panel in Streamlit.
   - Unknown numeric columns appear as Raw columns.
   - User can map them to standard fields from a dropdown.
   - User can save mappings to user_column_aliases.json for future uploads.
   - User can clear saved mappings.

2) Added parser aliases for common ESP/artificial-lift short names:
   - Pi / P intake / intake pressure -> Pi / Intake Pressure (psi)
   - Pd / P discharge / discharge pressure -> Pd / Discharge Pressure (psi)
   - Amp / AMp / Current / Cur. -> Motor Current (A)
   - AMA -> AMA / Motor Current (A)
   - Freq / Hz / Run Freq -> Pump Frequency (Hz)
   - Ti -> Intake Temperature
   - Tm -> Motor Temperature
   - Vx / Vy / Vz -> Vibration X/Y/Z

3) Kept the previous robust table detector:
   - finds table/header location,
   - combines parent headers + unit rows,
   - detects date/time/date-only/time-only formats,
   - keeps unknown numeric columns as selectable raw columns,
   - removes the old "Parser build" caption and the cleaned Excel / filtered CSV buttons.

How the generic mapping works
-----------------------------
The parser first tries automatic detection. If it does not recognize a column name,
it keeps the numeric time-series column as Raw. The user can then teach the app what
that Raw column means from the mapping panel and save it for future uploads.

Run
---
pip install -r requirements.txt
streamlit run app.py

Validation
----------
python bulk_parser_check.py "C:/path/to/customer/files" --out parser_validation_report.csv
