# Production Test Dashboard v98

## Purpose

v98 fixes the legacy dashboard PDF error:

> No usable time-series table was detected.

Older dashboard PDFs exported before v97 contain plotted Matplotlib vector paths, not a normal spreadsheet-style table and not the portable state attachment added in v97. The generic PDF table parser therefore found dates, axis labels and selected value labels, but could not rebuild a timestamped table.

## Legacy vector PDF recovery

When an uploaded PDF is identified as an older Matplotlib production-test dashboard, v98 now reads the PDF drawing commands directly.

It reconstructs:

- Every plotted sample from the vector curve coordinates.
- Date/time values from the repeated X-axis tick labels and compressed timeline scale.
- Engineering values from the Y-axis grid and tick calibration.
- Signal names and plotting order from panel labels and line colors.
- The well name and chart title.
- Light or Dark theme from the page background.
- Vertical point events and their timestamps.
- Operation intervals, start/end timestamps and labels.

No OCR is used for this recovery path.

## Result for the supplied BED-3-C18-7 PDF

- Well: `B3C18-7`
- Recovered rows: `42`
- Time range: `30-Jun-2026 19:00` through `02-Jul-2026 10:00`
- Signals: `9`
  - Total Gas Rate
  - Gross Rate
  - Oil Rate
  - Water Rate
  - BS&W
  - WHP
  - Separator Pressure
  - Choke Opening
  - Salinity
- Point events: `2`
  - `SIWHP 3300 PSI`
  - `SIWHP 3100 PSI`
- Interval event: `Closed the Well`
- Theme: `Dark`

After reopening the PDF, the recovered dataset becomes the current analysis baseline. A later Excel, CSV or portable PDF upload for the same well can continue the curve, with newer overlapping timestamps taking priority.

## Safe activation

The vector recovery activates only when all of the following are present:

- A PDF upload.
- Multiple Matplotlib-style chart panels.
- Real date/time tick labels.
- Recoverable engineering axes and vector curves.

Ordinary vendor reports continue through the existing PDF/table parser.

## Portable PDFs remain preferred

PDFs exported by v97 or later carry the complete CSV and chart state as a safe embedded ZIP attachment. Those PDFs reopen exactly and remain the preferred workflow. v98 vector recovery is the compatibility path for older dashboard PDFs that have no embedded state.

## Deployment

Replace the repository contents with the v98 package and reboot the Streamlit application. Confirm the startup log contains:

`v98-legacy-vector-pdf-recovery-20260702`
