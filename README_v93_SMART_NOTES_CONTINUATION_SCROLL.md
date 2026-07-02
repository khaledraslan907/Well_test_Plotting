# Production Test Dashboard v93

## Purpose of this update

v93 addresses three usability problems:

1. Operation comments and interval notes overlapping each other or covering chart values.
2. Having to upload the complete test again when a later file continues the same well test.
3. An ineffective sidebar scrollbar that was difficult to drag or scroll with the mouse wheel.

## 1. Collision-safe comments and notes

The chart now uses a dedicated note band above the plotting area.

- Point notes are arranged by timestamp **and estimated text width**, not timestamp alone.
- Long comments are wrapped to a maximum of three short lines on the chart.
- Nearby comments automatically move to separate rows.
- Interval notes also use the collision layout while preserving parent/child interval levels.
- Note text no longer moves downward into production values or value labels.
- The plot reserves extra top margin only when notes exist.
- Automatic layout is used consistently for interactive Plotly charts and PNG/PDF exports.
- The complete original comment remains in the sidebar note table even when the chart label is shortened.
- Vertical labels are used only when the user explicitly selects **Vertical labels**.

## 2. Continue the current test automatically

A new Processing option is enabled by default:

**Continue current test with new uploads**

Workflow example:

1. Upload a test file ending at 18:00 and select the desired signals/chart options.
2. Later replace or add a file extending the same well test to 08:00.
3. The application appends the new readings to the current analysis automatically.
4. Existing selected signals, signal order, units, labels, chart mode and other controls remain unchanged.
5. Repeated timestamps in the overlap are merged, with values from the newest upload taking priority.
6. Test IDs are rebuilt using the selected inactive-gap rule so the curve stays continuous when appropriate.

The app compares detected well context before appending. A replacement file from a different well/test starts a fresh data view automatically while preserving UI preferences.

Use **Start a new analysis** to clear retained data, uploader files, notes and intervals intentionally.

The retained test data exists only in the current Streamlit browser session. It is not written to a shared server database.

## 3. Sidebar scrollbar

The sidebar now has one real scrolling container instead of competing nested overflow containers.

- Mouse wheel and touchpad scrolling target the correct sidebar surface.
- The scrollbar is wider and has a larger minimum thumb height.
- Dragging the thumb is easier on long control panels.
- Scrolling is contained inside the sidebar and no longer unintentionally moves the main page.
- Extra bottom space allows the final controls to be reached comfortably.

## Performance considerations

- New continuation files reuse already parsed session data.
- Chart setting changes do not reparse uploaded files.
- Only one shallow retained DataFrame is kept for continuation.
- Upload and combined-data caches are invalidated by the v93 UI build ID.
- Duplicate merging runs only when continuation mode is active.

## Deployment

Replace the complete repository contents with this package and reboot the Streamlit application.

Expected startup message:

```text
v93-smart-notes-continuation-sidebar-scroll-20260629
```

The parser build remains the validated v91 parser included in this package.
