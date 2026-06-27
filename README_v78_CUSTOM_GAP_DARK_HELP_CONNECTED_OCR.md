# Production Test Dashboard v78

## Purpose

v78 restores the user-controlled inactive-gap workflow, fixes dark-theme help/tooltip visibility, and ensures nearby OCR/image readings are drawn as a continuous line when they belong to the same user-defined test period.

## Test segmentation

Test segmentation means deciding where one production-test period ends and another begins.

The default mode is now **Custom inactive gap (default)**:

- Readings remain in one test while the gap between consecutive readings is less than or equal to the selected number of hours.
- A new test starts only when the gap is larger than the selected value.
- Default value: **12 hours**.
- Two OCR snapshots 2 hours apart therefore share one `test_id` and are connected by a line.

Other optional modes remain available:

- **Use parser-detected boundaries**: preserve test IDs found by the parser.
- **Keep each well as one continuous test**: never split a well by time gap.

## Why two uploaded images were shown as separate points

Unknown OCR images were previously grouped by their individual source filenames. Each image therefore received a different `test_id`, even when the timestamps were close. Plotly correctly treated those IDs as separate segments and displayed two one-point traces.

In explicit Custom Gap mode, unknown readings are now grouped by the selected time-gap rule rather than by filename. The safe parser-level default remains unchanged for unattended ZIP parsing.

## Dark-theme visibility

v78 adds explicit colors for BaseWeb/Streamlit portal content that is mounted outside the normal app container:

- help tooltips and their arrow
- anonymous inner tooltip surfaces used by some Streamlit versions
- dropdown/popover bodies
- option text, hover and selected states
- help icons and tooltip paragraphs

Dark-theme contrast checks:

- normal text on tooltip panel: approximately 13.18:1
- strong text on tooltip panel: approximately 14.83:1
- muted text on tooltip panel: approximately 8.61:1

## Deployment

Replace the complete repository with this package and reboot the Streamlit app. The parser/UI build IDs changed, so cached results from earlier builds are invalidated.
