# Production Test Dashboard v75

## Purpose of this release

v75 fixes three related regressions:

1. Long sampling intervals inside one physical test were treated as separate chart segments.
2. Internal `_upload_order` and `_table_order` fields were exposed as plot signals.
3. Wide Excel templates with cached values extending to column XFD caused very slow parsing and an unresponsive Streamlit interface.

## Continuous test plotting

Plot continuity is now controlled by `test_id`, not by the time between adjacent readings.

- A 30-minute TMU series remains continuous.
- A daily SRP series remains continuous.
- A several-day interval inside the same Test ID remains connected.
- A line is split only when the detected Test ID changes.
- Compressed time mode may shorten empty calendar gaps, but it does not break the curve.
- Dashed vertical separators now indicate actual test changes only.

## Removed non-engineering plot fields

The following fields are retained only temporarily for duplicate resolution and are removed before the UI is built:

- `_upload_order`
- `_table_order`
- `_source_row_order`

Generic calculation spillover such as `raw_calcul_7` and `raw_channel_4` is also hidden from the plotting list. Meaningful unfamiliar sensor columns remain available.

## Faster parser

The adaptive Excel reader now:

- Detects workbooks with pathological XFD used ranges.
- Trims cached formula spillover to the actual header/data width.
- Avoids sending ordinary operational text through pandas date parsing.
- Rejects sentences such as `Pressure test 4000 psi` as numeric measurement cells.
- Scores only semantically relevant columns when a header is already understood.
- Uses the adaptive parser first for pathological wide workbooks and avoids running a second full parser when the result is credible.

## Validated files

- `SPR(2).xlsx`: 12 SRP readings, one continuous curve for each SRP signal.
- `6-B15-42 (Dasco 27) (9-6-2026)(4).xlsx`: 10 valid TMU readings.
- `12-S8-58 (12-06-2026)(3).xlsx`: 59 measurement rows from the real test interval, including WHP, N2, separator pressure, total/formation gas, liquid rates and pumping pressure.

## Deployment

Replace the complete repository contents with this package and reboot the Streamlit application. Do not mix the v75 application with an older parser module.
