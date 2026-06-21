v55 — Safe XML datetime fix for pandas 3 / Python 3.14

Root cause fixed
----------------
The safe XML worksheet reader was correctly limiting huge Excel used ranges, but
the generic datetime detector still passed plain four-digit operational values to
pandas. Values such as 1234, 1346, 1423, 1436, 2600, and 2613 were interpreted as
calendar years by pandas 3. When those wide-range timestamps were assigned to a
nanosecond datetime Series, pandas raised OutOfBoundsDatetime or its internal
"Something has gone wrong" AssertionError.

Changes
-------
- Plain numeric measurements are no longer accepted as dates/datetimes.
- Excel serial dates are accepted only in date-labelled columns and only in the
  plausible modern range 20000–80000.
- All parsed dates are limited to years 1900–2100.
- Datetime Series are built once from already-sanitized values; no mixed-unit
  .loc assignment is used.
- Raw XML worksheet frames remain object dtype and date/time columns are rebuilt
  from Python lists.
- Cumulative Excel time values such as 2.083333 are preserved using their
  fractional day, so later rows are not lost.
- Parser cache ID is bumped to v55.
- Cross-file duplicate merging remains unchanged: normalized well + minute,
  keeping the most complete row and filling missing values from repeated files.

Validation
----------
Tested on all 17 uploaded B15-38 Excel files with:
- pandas 2.2.3
- pandas 3.0.3

Both environments produced:
- 17/17 files parsed
- 753 source rows
- 500 unique rows after merging
- 253 repeated rows merged
- 0 remaining duplicate well + minute keys

Deployment
----------
Replace app.py, tmu_parser.py, requirements.txt, packages.txt, and runtime.txt
from this package, commit, reboot the Streamlit app, and use the app's
"Re-parse uploaded files / clear cache" button once.
