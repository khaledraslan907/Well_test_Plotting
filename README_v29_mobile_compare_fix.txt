TMU Dashboard v29 - mobile and multi-well comparison fix

Main changes:
1. Added a new X-axis mode:
   Aligned elapsed time - best for comparing wells
   This starts every selected well/test from 0 hours so one short test is not squeezed into a tiny cluster beside a long test.

2. Improved compressed real-date mode:
   - fewer x-axis labels when there are multiple wells/tests
   - less overlap in downloaded PNG/PDF
   - test separators only appear for real compressed timelines, not aligned elapsed charts

3. Added Chart screen layout:
   - Auto / desktop
   - Mobile-friendly
   - Wide report view
   Mobile-friendly reduces tick/value-label crowding on phone screens.
   Wide report view gives larger panels for desktop/report use.

4. Improved high-resolution exports:
   - larger figure height per selected feature
   - higher PNG DPI
   - clearer x-axis ticks for compressed and aligned modes
   - notes and interval notes remain visible in exports

Recommended use:
- For one well / one test: Real calendar time or Compressed real dates.
- For two wells or several test dates: Aligned elapsed time - best for comparing wells.
- For phones: Chart screen layout = Mobile-friendly.
- For final report export: Chart screen layout = Wide report view, then download PNG/PDF.

Deploy:
Replace app.py, tmu_parser.py, and requirements.txt together, then reboot Streamlit Cloud.
