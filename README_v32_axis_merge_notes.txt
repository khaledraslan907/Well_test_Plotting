TMU Dashboard v32 - axis merge, clean notes, one well curve

Changes:
1. Same well name now appears once only in legend and chart title. It no longer repeats as B15-42 (09-Jun), B15-42 (10-Jun), etc.
2. Compressed real-date timeline now uses a global datetime compression map instead of splitting by file/source. If an Excel test ends at 05:00 and WhatsApp readings continue at 06:00 for the same well, they stay continuous when the gap is within the selected threshold.
3. Added adjustable controls:
   - Treat readings as continuous when gap is <= hours
   - Visual gap shown for separated tests
   - X-axis label density: Sparse / Balanced / Detailed
4. Notes and interval labels now show only the note text, not file/source/well prefixes.
5. Removed subplot titles from every panel. The main title is shown once at the top; the feature name remains on the Y-axis.
6. Added numpy import needed by axis tick calculations.

Recommended settings for same-well multiple uploads:
- X-axis display mode: Compressed real dates - remove empty gaps
- Treat readings as continuous when gap is <= hours: 2
- Visual gap shown for separated tests: 0.75 to 1.5
- X-axis label density: Sparse or Balanced
- Chart screen layout: Wide report view for export, Mobile-friendly for phone viewing

Replace these files together:
- app.py
- tmu_parser.py
- requirements.txt
- runtime.txt

Then reboot Streamlit Cloud.
