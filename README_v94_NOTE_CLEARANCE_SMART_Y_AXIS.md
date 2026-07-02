# Production Test Dashboard v94

## Purpose

v94 addresses two chart-readability requirements:

1. Event/interval comments must not overlap data-value labels.
2. Default Y-axis ranges must start at zero for non-negative engineering measurements and end at a readable rounded value above the detected maximum.

## Comment and value-label clearance

- Event comments are always placed horizontally in a dedicated band above the chart.
- The old vertical-comment option was removed because rotated text could extend into the data region.
- Nearby notes continue to use staggered rows and wrapped labels.
- Extra top margin is reserved according to the number of note rows.
- Data-value labels located close to an event or interval guide line are automatically hidden, while the point itself remains visible and its value remains available on hover.
- The same rules are used for interactive Plotly charts and Matplotlib PNG/PDF exports.
- Full note text remains available in the Events & Notes table.

## Smart default Y-axis ranges

Unless the user enables a custom Y-axis range:

- Non-negative measurements start at `0`.
- The upper limit is rounded to a readable engineering value above the maximum measurement.
- Examples:
  - Maximum gas rate `3.91` -> axis `0 to 5`.
  - Maximum gross rate `177` -> axis `0 to 200`.
  - Maximum WHP `1700` -> axis `0 to 2000`.
- BS&W / water cut and percentage choke opening always use `0 to 100%`.
- Choke size in `/64 in` uses `0 to 128`.
- Signals containing real negative readings, such as reverse reel speed, retain a negative lower limit so valid signed data is not hidden.
- User-entered custom ranges continue to override the automatic range.

The behavior is applied consistently to:

- Separate-panel charts
- Overlay charts
- Dual-axis charts
- Production History
- Interactive charts
- PNG and PDF exports

## Deployment

Replace the complete repository contents with this package and reboot the Streamlit application.

Expected startup message:

```text
v94-note-clearance-smart-y-axis-20260702
```
