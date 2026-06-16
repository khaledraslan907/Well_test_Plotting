TMU Dashboard v48 - Streamlit fixes

Main changes:
1. Added final safety detection for Pumping Pressure columns from Excel headers such as Pumping P, Pumping.P, Pump P, Pump Pressure, Pumping Pressure, and Circulation Pressure.
2. Changed Plot Style back to the two main modes: Separate panels and Overlay actual values.
3. Added an independent option: Add one combined chart with secondary Y-axis. This adds a dual-axis chart above the normal multi-chart report without removing the remaining charts.
4. Improved value labels:
   - New default: Clean readable - recommended.
   - Removes repeated zero-label clutter.
   - Caps labels per chart and prioritizes first/last/min/max/significant changes.
   - Every 20 readings remains available.
   - Export charts use the same label logic as the interactive chart.
5. Improved event labels:
   - More stagger levels.
   - Auto mode switches crowded notes to vertical labels.
   - Point-note table now lets the user edit label text, X shift in pixels, and Y level manually.
   - Plotly annotation dragging remains available for visual adjustment.
6. Speed/readability balance:
   - Markers are automatically off for larger datasets.
   - Parsing cache remains active.
   - OCR remains optional and off by default.

Replace all files in GitHub with this package and reboot the Streamlit app.
