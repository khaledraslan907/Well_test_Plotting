# Production Test Dashboard v92

## Custom intervals and draggable signal order

### Custom time intervals

The Test Detail view now places a custom interval field beside these preset lists:

- **Average readings by time interval**
- **X-axis tick scale**

Choose **Custom** and type values such as:

- `2 hours`
- `90 minutes`
- `1.5 days`
- `2 months`
- `1 year`

Invalid or non-positive values produce a visible warning and safely fall back instead of crashing the chart.

The custom X-axis interval works in both real-calendar and compressed-real-date modes.

### Custom value-label frequency

Fixed label intervals were replaced with a single user-controlled value:

- Test Detail: **Every N readings**
- Production History: **First, last + every N tests**

The default remains 20.

### Draggable plot order

Users still select or remove signals with the searchable **Signals to plot** field. Below it, the selected signals appear in a vertical drag list.

Dragging a signal immediately changes:

- chart panel order
- legend/trace order
- PNG/PDF export order
- table/export feature order where selected-feature order is used

The user no longer has to remove and re-add signals merely to rearrange the plots.

The drag component is lightweight and only receives the selected signal names. It does not reparse uploaded files.

### Stability and speed

- Existing parser and OCR behavior remain unchanged from v91.
- File parsing stays cached; changing intervals or signal order reruns chart preparation only.
- Duplicate default signals are removed before rendering.
- If the drag component is unavailable during a partial deployment, the app falls back to the current selected order instead of failing.

## Deployment

Replace the complete repository contents. The package adds:

```text
streamlit-sortables==0.3.1
```

to `requirements.txt`.

Expected startup log:

```text
v92-custom-intervals-drag-signal-order-20260628
```
