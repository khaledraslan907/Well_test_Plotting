# Production Test Dashboard v81
## Simplified test separation and stability hardening

### Interface changes

- Removed the **How should tests be separated?** menu.
- Removed these two modes:
  - Use parser-detected boundaries
  - Keep each well as one continuous test
- Kept one direct control only:
  - **New test after inactive gap (hours)**
- Default remains **12 hours**.
- Removed the long test-segmentation explanation and other nonessential sidebar captions.
- Simplified section names:
  - Processing
  - Units & Choke
  - Timeline & Range
  - Wells & Signals
  - Chart Options
  - Events & Notes
- Simplified common labels such as **Signals to plot** and **Upload CTU/HMI photos**.

### Test separation rule

The rule is now consistent for spreadsheets, WhatsApp reports and OCR images:

- Time gap less than or equal to the selected value: keep readings in the same test.
- Time gap greater than the selected value: start a new test.

Existing parser test IDs are rebuilt using this rule so the interface no longer presents competing segmentation behaviors.

### Stability and performance changes

The generic Streamlit **Oh no** page does not reveal the original cause without the Cloud logs. v81 addresses the main app-side causes that could produce an intermittent failure:

1. Removed the duplicate `st.cache_data` cache for complete parsed DataFrames. Session state is now the single upload cache.
2. Removed deep duplication of the merged dataset.
3. Clears stale workbook data and prepared export bytes when uploads change or are removed.
4. Keeps only the active Light or Dark export in memory.
5. Keeps only one prepared export payload at a time.
6. Limits large browser table previews while leaving source data and exports complete.
7. Reduces only very dense interactive Plotly payloads; full filtered readings remain available to exports.
8. Limits the interactive view to the first 12 selected signals when a user selects a very large number of columns.
9. Adds a safe chart-rendering fallback so one chart error does not terminate the whole Streamlit run.
10. Replaced deprecated `use_container_width` calls with the current `width="stretch"` API.
11. Added Streamlit `maxMessageSize = 200` to match the 200 MB upload setting.

### Files to deploy

Replace the complete repository with this package. Do not mix the v81 `app.py` with older parser files.

Required files:

- `app.py`
- `tmu_parser.py`
- `tmu_parser_compat.py`
- `tmu_parser_legacy.py`
- `smart_tabular_v75.py`
- `requirements.txt`
- `packages.txt`
- `runtime.txt`
- `.streamlit/config.toml`

### Validation

See `TEST_RESULTS_v81.txt` and run:

```bash
python ui_smoke_test_v81.py
python self_test_v81.py
```
