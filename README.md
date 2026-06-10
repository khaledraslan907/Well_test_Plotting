# TMU Production Test Dashboard - Fixed v3

## Main fixes in this version

- Correctly handles multi-row Excel headers such as:
  - row 1: Gas / Oil / Water / Gross Rate
  - row 2: Sep P / Total Gas Rate / Oil Rate / Water Rate / Bs&W / Salinity
  - row 3: units
- Does not confuse:
  - Oil Temp with Oil Rate
  - Gas Sep P with Gas Rate
  - Form/Potential summary rows with time-series rows
  - Final Average row with real timed readings
- Skips event rows without real numeric test readings.
- Explains Detected Data vs Filtered Data inside the app.
- Displays friendly feature names in the UI.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Use

1. Upload Excel/CSV/PDF/Word/TXT files.
2. Or paste WhatsApp TMU reports.
3. Choose well(s).
4. Choose features.
5. Choose time range.
6. Download Excel/CSV/PNG/PDF outputs.


## Fixed v4 additions

- Excel percentage cells are corrected:
  - Choke `1` becomes `100%`
  - Choke `0.75` becomes `75%`
  - BS&W fraction values are also corrected if Excel stores them as fractions.
- Plot value labels are now readable:
  - Auto sparse label mode is the recommended default.
  - Labels are shorter and rounded.
  - Labels alternate above/below points.
  - Y-axis padding is added.
  - PNG/PDF export is wider and taller to avoid overlapping labels.
- Use `All values - use wide export` only when you need every point labeled in the saved PNG/PDF.


## Fixed v5 additions

- Downloaded PNG/PDF charts are now colorful, not black-only.
- Each feature has its own consistent color in separate-panel report mode:
  - Gross: blue
  - Oil: green
  - Water: cyan
  - WHP: purple
  - Separator Pressure: brown
  - BS&W: olive
  - Salinity: red
  - Pumping Pressure: orange
  - Choke: pink
- White background and soft gridlines improve readability in exported images and PDFs.


## Fixed v6 additions

- The exported chart title is now the well name/date instead of generic `Production Test Results vs Time`.
- Example: `Well B3-17 | 28-05-2026 to 29-05-2026`.
- Multiple wells display as `Well Comparison: ...`.
- Plot title, subplot titles, axis labels, tick labels, legend, and value labels are larger and darker.
- Streamlit chart display no longer compresses the plot to the screen width, improving readability on website view.
- PNG/PDF exports are larger for clearer human reading.


## Fixed v7 additions

- Forces a light, high-contrast Streamlit theme so sidebar/title text is readable.
- Removes the single-well legend line because the well name is already centered in the chart title.
- Centers and enlarges the chart title.
- Shows time tick labels on every subplot, not only the bottom axis.
- Keeps chart colors and improves text contrast in website view and exported PNG/PDF.


## Fixed v8 additions

- Added special parser for EXPRO MPFM PDF reports with `Data & Events` pages.
- Handles rows starting after the report header, for example `Time Choke Size WHP Flow Press ...`.
- Skips text/event rows such as `BS&W is ...`, `bypassed the meter`, and `continue testing`.
- Pulls all MPFM numeric columns, including QOil(S), QWat(S), QGas(S), WLR(S), QGross(S), GOR(S), GVF(A), pressure, temperature, DP, salinity, H2S, CO2, and pump frequency.
- Adds optional `Hide zero-flow/bypassed rows` filter.


## Fixed v9 additions

This version was tested against additional historical and vendor-specific templates:

- `Bahga-11 (8-12-2014)` legacy BAPETCO format
- `Client Report - Bahga#9 TMU 2 - Bapetco (03-07-2019)` SLB/TMU style format
- `MAGD C 86-4 (10-09-2020)` older production test template
- `B3-17 (28-5-2026)` PICO-style workbook
- `EXPRO MPFM WELL TEST DATA REPORT (02-06-2023)` PDF Data & Events report

Parser improvements:

- Better multi-row header detection.
- Better handling of merged parent headers.
- Extracts well name from workbook/PDF metadata before using filename.
- Avoids confusing dates in filenames with well names.
- Handles combined `Date & Time` columns and separate `Date` + `Time` columns.
- Supports legacy labels such as FTHP, FLP, FLT, GasP, GasDP, GasT, OilT, LIQ Q, OIL Q, WATER Q, CGR.
- Supports TMU labels such as U/S Press, D/S Press, liquid metering, gas metering, Sep. Press, Sep. DP.
- Supports older production-test layouts with oil/condensate, water reading, gas, GOR, and gross-rate sections.
- Choke percentage stored as `1` is converted to `100%`, while choke size `/64` remains as choke size.
- Summary/form sheets are ignored when a detailed time-series sheet exists.
- EXPRO MPFM PDFs keep only detailed Data & Events rows when the dedicated parser succeeds.

Limitations:

- Unseen company templates may still require adding aliases. The parser is now structured so new templates can be handled by adding header aliases without rewriting the dashboard.


## Fixed v10 additions

- Improved dense-chart readability for long PDF/MPFM datasets.
- New default label mode: `Hourly + min/max - best for reports`.
  - Shows first/last values.
  - Shows hourly readings.
  - Shows minimum and maximum values.
  - Shows zero/bypass points.
- Larger Plotly chart labels, thicker lines, and larger markers.
- Overview PNG/PDF export is taller and less compressed.
- Added two new human-readable exports:
  - `Download multi-page readable PDF`: one full-size chart per selected feature.
  - `Download readable PNGs ZIP`: one high-resolution PNG per selected feature.
- These exports are recommended for human review when the stacked overview has many rows/points.


## Fixed v11 additions

- Fixes the incorrect first readings in B3 C18-7-style templates:
  - Drops `Final Average` rows from the time-series data.
  - Corrects midnight rollover when the TIME resets from 23:30 to 00:00 while DATE is still copied from the previous day.
- Re-imports dashboard-exported CSVs correctly.
- Adds manual graph events/operation notes. Example lines:
  - `2026-06-06 10:30 | Open to flare`
  - `2026-06-06 12:00 | Open to flowline`
  - `13:30 | Start lifting`
- Adds time controls for long tests:
  - Slider / Manual calendar-time / Full range
  - Raw, minute, hourly, daily, monthly, or yearly aggregation
  - X-axis tick scale: auto, minute, hour, day, month, or year
- Event lines and labels are shown on the interactive plot and human-readable PDF/PNG exports.


## Fixed v12 additions

- Fixes PICO-style headers:
  - `Total Gas rate` is detected as `Total Gas Rate (MMSCF/D)`.
  - `formation gas rate` is detected separately as `Formation Gas Rate (MMSCF/D)`.
  - `Pump P` is detected as `Pumping Pressure (psi)`.
- Adds raw numeric fallback columns:
  - If a future company template has a numeric column without an alias, it appears as `Raw: ...` so the user can still select and plot it.
- Adds `Select all numeric columns`.
- Changes label-spacing options:
  - `Every 2 readings` became `Every 4 readings`.
  - `Every 3 readings` became `Every 8 readings`.
- Adds clearer in-app explanations:
  - `Hourly + min/max` = operational labels: hourly points, min, max, zero/bypass, first/last.
  - `Auto sparse` = evenly spaced labels to reduce overlap.
  - `Show internal column names` = debugging mode for parser/code names.
- Adds x-axis display modes:
  - `Real calendar time`
  - `Elapsed time from each test start - remove date gaps`
  - `Reading sequence - remove all time gaps`
- Adds trace grouping:
  - `Well only`
  - `Well + file/sheet test`
- Improves zero-flow removal by prioritizing gross-rate columns.


### v12.1 patch

- Prevents single metadata/reference rows from being treated as header rows.
- Keeps raw fallback columns available, but skips hidden calculation/helper columns.
- Uses recognized operational columns to decide whether a row is a real reading, so event rows are not pulled in just because hidden formulas contain numbers.


## Fixed v13 additions

- Removed the mobile/x-axis tip text above the chart.
- Added easier graph event entry:
  - Select event date.
  - Select event time.
  - Write the operation note.
  - Click `Add event to graph`.
- Added event table in the sidebar with remove/clear buttons.
- Bulk paste event input is still available inside an optional expander.


## Fixed v14 additions

- Fixed `NameError: re is not defined` for Reading Sequence mode and Well + file/sheet test grouping.
- Removed `Minute` x-axis tick scale; tick options now start at `30 minutes`.
- Added clearer tick options: `30 minutes`, `1 hour`, `3 hours`, `6 hours`, `12 hours`, `1 day`, `1 month`, `1 year`.
- Renamed `Aggregate / resample readings` to `Average readings by time interval` and clarified its meaning.
- Fixed vertical-spacing crash when selecting all numeric columns and plotting many rows.
- Made Detected Data and Filtered Data optional collapsed expanders to keep the dashboard cleaner.
- Improved Auto Sparse labels by showing fewer labels with less overlap.
- Added operation intervals/phases:
  - Select interval start date/time.
  - Select interval end date/time.
  - Write label such as `No pumping` or `Start lifting`.
  - The interval is shaded across the graph and included in readable exports.


## Fixed v15 additions

- Moved `Time scale` to section 3 and `Filter and plot` to section 4.
- Manual calendar/time now supports both picker time and typed time such as `09:30`, `0930`, or `9:30 PM`.
- Added custom Y-axis scale per selected graph, allowing ranges such as `0 to 1000`, `50 to 1000`, or `70 to 900`.
- Simplified graph notes:
  - One operation note form.
  - Start date/time always required.
  - End date/time is optional.
  - Without end date/time: vertical event line.
  - With end date/time: shaded interval across the graph.
  - Removed the bulk paste event section.
- Added `Compressed real dates - remove empty gaps` x-axis mode:
  - Removes long empty date gaps between tests.
  - Keeps real date/time tick labels.
  - Adds dotted separators between tests.


## Fixed v16 additions

- Interval notes no longer use shaded background.
- When an end date/time is added, the chart now shows:
  - vertical start line
  - vertical end line
  - operation note centered between the two lines
  - left/right arrows around the interval note
- The same interval style is applied to the interactive Plotly chart and the readable PDF/PNG exports.


## Fixed v17 additions

- Removed `Show internal column names` from the interface.
- Removed `Elapsed time` and `Reading sequence` x-axis modes.
- Removed the confusing `Well only / Well + file/sheet test` selector. Trace grouping is now automatic.
- Compressed x-axis keeps real date/time labels with year and uses dotted separators between test periods.
- Date inputs now allow a broad range from 1900 to 2100 to avoid Streamlit date-range errors.
- Operation notes can now be applied to all selected wells or a specific well.
- This allows different notes at the same time for different wells, such as `Well_1 choke 10%` and `Well_2 choke 50%`.
- Chart title is smaller and clearer to avoid clipping.


## Fixed v18 additions

- Fixes Excel parsing error:
  - `sequence item X: expected str instance, float found`
- The parser now safely converts all multi-row header fragments to strings before joining them.
- This protects files where merged/unit header rows contain numeric values.
