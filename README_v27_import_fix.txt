TMU Dashboard v27 - import-safe generic mapping

Why this version exists
-----------------------
Streamlit Cloud can show a redacted ImportError when app.py imports functions that are not present in the deployed tmu_parser.py. This usually happens when app.py was updated but tmu_parser.py is still an older version in the GitHub repo.

Required deployment structure
-----------------------------
Put these files in the same folder/repo root:

- app.py
- tmu_parser.py
- requirements.txt

Then restart/reboot the Streamlit app. Do not upload only app.py.

What changed in v27
-------------------
1. app.py now imports tmu_parser safely.
2. If the parser file is missing or too old, the app shows a clear error instead of a redacted Streamlit crash.
3. The Column Mapping Review panel still has fallback helpers if an older parser accidentally lacks the new mapping helper functions.
4. Keep using tmu_parser.py from this same package for the full parser and aliases.

Run locally
-----------
pip install -r requirements.txt
streamlit run app.py

Streamlit Cloud
---------------
1. Replace app.py, tmu_parser.py, and requirements.txt in your GitHub repo.
2. Commit and push.
3. In Streamlit Cloud, Manage app > Reboot app.
4. If needed, clear cache from the app menu.
