v50 changes:
- Event/operation note dragging is now interactive-only.
- Downloaded PNG/PDF exports no longer use saved X-shift/Y-level manual note positioning.
- Plotly browser camera button is hidden to avoid exporting a dragged on-screen view by mistake; use the app download buttons.
- Point-note table keeps only date/time, target, and label edits.
- Pumping Pressure detection from the S8-58 Excel file was rechecked: the column is EG with header pumping.p and unit psi; parser detects 78 non-empty pumping_pressure_psi readings.
