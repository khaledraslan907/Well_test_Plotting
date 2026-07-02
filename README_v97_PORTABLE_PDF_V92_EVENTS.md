# Production Test Dashboard v97

## Purpose

v97 completes the saved-PDF continuation workflow and restores the representative version-92 event style while retaining the newer parser, continuation, OCR, smart Y-axis, export, Production History, and sidebar-scrolling improvements.

## 1. Reopen an exported PDF as the same analysis

PDFs prepared by v97 contain a safe embedded analysis package. The visible PDF pages remain normal report pages, while the attachment stores the information needed to rebuild the analysis.

When a v97 PDF is uploaded, the application restores:

- The complete canonical test readings used for continuation.
- Light or Dark theme.
- Point events and operation intervals.
- Chart title.
- Selected signals and draggable signal order.
- Selected well(s).
- Pressure and temperature display units.
- Event-label layout.
- Plot style, markers, value-label mode, custom Y-axis ranges, and other key chart controls.
- Choke display settings and saved display-label overrides.

The restored data becomes the continuation baseline. A later Excel, CSV, PDF, WhatsApp export, or supported device file can extend the same well test. Repeated timestamps are merged and the newest uploaded values take priority.

### Important compatibility note

A PDF exported before v97 does not contain the embedded data/event package. The application can make a best-effort Light/Dark theme detection for an older dashboard PDF, but exact events and underlying readings cannot be reconstructed reliably from a chart picture alone. Export the report once from v97 to make subsequent PDF reopening exact.

## 2. Version-92 event presentation

- Point events remain inside the chart at their exact times.
- Dashed vertical event lines remain visible through the plot.
- **Auto staggered** now keeps compact horizontal labels near the top, matching the version-92 presentation.
- Nearby comments are assigned separate height rows using timestamp and estimated text width.
- Vertical event text is used only when **Vertical labels** is explicitly selected.
- Operation intervals retain two dashed boundaries, a bidirectional span arrow, and a compact inline label.
- Full event text remains available in the sidebar tables.

## 3. Numeric-label anti-overlap

Event visibility has priority over non-essential value labels.

- Important first, last, minimum, and maximum values remain prioritized.
- Values close to an event line move to the right of the marker.
- High values inside an operation interval move below the marker to avoid the interval arrow.
- Only a non-essential label located directly on a guide line may be hidden in a dense chart.
- The same placement logic is used in the interactive chart and Matplotlib PNG/PDF exports.

## 4. Stronger Light theme

The Light theme has been strengthened without changing the Dark theme:

- Darker chart and interface text.
- More visible grid lines.
- Stronger panel and control borders.
- More saturated oil, water, gas, and gross-rate curves.
- Clear white plotting surface with a firmer report background.
- Existing high-contrast event colors retained.

## 5. PDF implementation and safety

- Portable state is stored as `corelytix_production_test_state_v1.zip` inside the PDF.
- The embedded table uses CSV plus a JSON manifest; no Python pickle or executable object is loaded.
- Canonical pre-conversion pressure and temperature values are stored to prevent double unit conversion after reopening.
- The PDF pages are not visually changed by adding the attachment.

## Deployment

Replace the complete repository with the v97 package and reboot Streamlit. Confirm the startup log includes:

`v97-portable-pdf-v92-events-20260702`

New dependencies are included in `requirements.txt`:

- `pypdf>=5.9,<7`
- `pypdfium2>=4.30`

## Validation

Run:

```bash
python self_test_v97.py
```

The supplied validation covers compilation, PDF attachment round-trip, theme/events/interval restoration, dataframe datetime/numeric restoration, version-92 event behavior, smart Y-axis rules, and continuation overlap handling.
