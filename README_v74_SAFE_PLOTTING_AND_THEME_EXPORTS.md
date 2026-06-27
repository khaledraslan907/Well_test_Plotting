# Production Test Dashboard v74

## Purpose

v74 fixes the chart crash reported in `report_label_indices()` and makes all PNG/PDF report exports follow the selected Light or Dark theme.

## Root cause of the TypeError

The parser correctly kept several audit flags such as:

- `gas_formation_derived`
- `n2_rate_derived`
- `total_gas_derived`

However, they were Boolean columns and were accidentally returned by `available_numeric_columns()`. When "Select all numeric columns" was used, a Boolean flag could be sent to the chart label code. `pandas.to_numeric()` preserves Boolean dtype, and this expression fails:

```python
valid.max() - valid.min()
```

because NumPy does not support Boolean subtraction.

## v74 corrections

1. Boolean and audit/status fields are no longer offered as engineering plot features.
2. Every plotted feature is converted through one defensive `numeric_feature_series()` function.
3. The converter handles Boolean values, nullable types, Decimal/object values, numeric strings with units or commas, and duplicate mapped column names.
4. Plot labels, Y-axis ranges, Plotly traces and all Matplotlib export paths use the same safe float64 series.
5. A stale feature selected in Streamlit session state can no longer crash the chart.

## Theme-aware exports

PNG and PDF exports now use the active UI theme for:

- Figure and plot backgrounds
- Axis, tick, title and legend text
- Grid lines and borders
- Value-label boxes
- Event and interval note boxes

Every `savefig()` and `PdfPages.savefig()` call explicitly supplies the active background color. Export cache keys also contain `light` or `dark`, so a previously prepared Light file is not shown after switching to Dark, and vice versa.

The prepare buttons show the active theme, for example:

- `Prepare single chart PNG (Light)`
- `Prepare single chart PDF (Dark)`

Downloaded filenames also include `_light` or `_dark`.

## Files to deploy

Replace the complete repository contents with this package. Do not copy only `app.py`, because the parser facade and compatibility modules were also updated to exclude Boolean audit columns consistently.

## Validation

Run:

```bash
python self_test_v74.py
python ui_smoke_test_v74.py
```

The package was also tested through Streamlit AppTest using the supplied `SPR(1).xlsx` and `6-B15-42 (Dasco 27) (9-6-2026)(3).xlsx` files, including Select All, Light, Dark, PNG preparation and Dark PDF preparation.
