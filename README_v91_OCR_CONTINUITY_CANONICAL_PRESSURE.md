# Production Test Dashboard v91

## OCR continuity and canonical pressure fix

Build IDs:

- UI: `v91-ocr-continuity-canonical-pressure-20260628`
- Parser: `v91-ocr-continuity-canonical-pressure-20260628`
- CTU OCR: `v91-ocr-continuity-canonical-pressure-20260628`

## Why the OCR charts appeared as scattered points

A WhatsApp ZIP combines several row types in one table:

- WhatsApp production-test messages
- Spreadsheet attachment readings
- CTU/HMI image OCR snapshots

Each row type contains different parameters. For example, a production row can contain Gross Rate but no CTU pressure, while an OCR row can contain CTU pressure but no Gross Rate. Plotly interpreted the intervening blank values as breaks in the line.

v91 removes rows that do not contain the selected signal before it creates that signal's trace. Real zero values are retained. A test is therefore represented by one continuous trace rather than many isolated fragments.

## CTU Circulation Pressure mapping

`CTU Circulation Pressure (psi)` and `Pumping Pressure (psi)` are now treated as the same engineering channel:

- Raw OCR value retained: `ctu_circulation_pressure_psi`
- Canonical plotted/exported value: `pumping_pressure_psi`

Similarly:

- Raw CTU Wellhead Pressure is retained for review.
- The canonical plotted/exported value is `whp_psi`.

The raw aliases are hidden from normal plot selection, preventing duplicate charts for the same physical parameter.

## OCR approval behavior

- OCR rows remain visible in the OCR review table.
- Unapproved OCR rows are excluded from engineering charts.
- Approving or editing an OCR row updates the canonical Pumping Pressure and WHP channels.
- Low-confidence and temporal-outlier flags remain visible for engineering review.

## OCR improvements retained

The CTU template OCR includes focused recovery for:

- Faint leading digits in WHP
- Leading digits in reel depth
- Leading digits in circulation pressure
- Reel-speed signs and missing leading digits

The parser keeps raw OCR text and per-field confidence for auditability.

## Deployment

Replace the complete repository contents with this package. Do not combine the v91 application with an earlier parser. Reboot the Streamlit application and verify the startup log contains:

`v91-ocr-continuity-canonical-pressure-20260628`
