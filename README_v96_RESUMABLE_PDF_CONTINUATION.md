# v96 - Resumable PDF continuation

## Purpose

The earlier **Continue current test with new uploads** option retained data only while the same Streamlit browser session remained alive. Closing the app removed the in-memory DataFrame, chart controls, and event tables.

v96 adds a persistent resume workflow based on the dashboard PDF itself.

## User workflow

1. Build the test chart and add events/intervals normally.
2. In **Engineering Report Exports**, keep **Make PDF resumable - embed data, chart settings, and events** enabled.
3. Download either PDF report.
4. Close the app when required.
5. Reopen the app and upload that saved PDF.
6. The app restores the test data, well, signal order, theme, chart mode, timeline settings, title, events, and intervals.
7. Upload the later spreadsheet, WhatsApp report, ZIP, or image readings. Matching well/test readings are appended; overlapping timestamps keep the latest and most complete values.

## Exact versus legacy recovery

### PDFs exported by v96 or later

The visible report pages remain normal PDF pages. A compressed project attachment is embedded inside the PDF containing:

- canonical source data used by the chart;
- selected wells and signals;
- signal order;
- Light/Dark theme;
- display units;
- X-axis mode and spacing;
- chart mode and markers;
- value-label settings;
- custom Y-axis ranges;
- manual events and operation intervals.

Uploading this PDF provides an exact project restore.

### Older dashboard PDFs

Older PDFs do not contain the original project data. v96 performs best-effort recovery from their vector text, visible point labels, date/time ticks, and event lines. It restores the values that were printed in the report and reconstructs the event/interval table.

A scanned PDF, or an old report that did not print every reading, cannot reproduce hidden/unlabelled source values exactly. In that case, upload the original spreadsheet/chat export together with the PDF.

## Privacy

A resumable PDF contains the underlying project data as an embedded attachment. Treat it as confidential. Disable **Make PDF resumable** when creating a presentation-only PDF for external sharing.

No built-in well or company names are displayed before upload. User-owned names are shown normally after their file is parsed.

## Compatibility and performance

- Project extraction occurs once when the PDF is uploaded and is included in the existing upload cache.
- Changing chart controls does not re-read or re-parse the PDF.
- The embedded project uses compressed JSON and normally adds only a small amount to the PDF size.
- Ordinary vendor/report PDFs that do not match the dashboard signature continue through the normal PDF parser.
- Requires `pypdf` and `PyMuPDF`, now listed in `requirements.txt`.

## Deployment

Replace the complete repository contents with this version and reboot Streamlit. Verify the startup log includes:

`v96-resumable-pdf-project-20260702`
