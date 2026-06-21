TMU Dashboard v58 - Persistent Selections, Unit Conversion, Choke/Test Separation

1. FEATURE SELECTIONS STAY SELECTED
- "Choose features to plot" now uses a stable Streamlit session-state key.
- Changing choke output between Opening (%) and Size (/64 in) does not reset the other selected features.
- When switching to unified choke, a selected raw choke feature maps to choke_unified.
- When switching to separate source units, a selected unified choke maps to the available raw choke fields.
- Pressure and temperature display-unit changes keep the same feature keys and selections.

2. AUTOMATIC CACHE INVALIDATION
- The normal "Re-parse uploaded files / clear cache" button was removed.
- Cache identity now includes parser build, file identity, OCR settings, and pasted-message hash.
- Updating parser build or changing uploaded files automatically creates a new cache key.
- Parsed files and the duplicate-merged dataset remain cached for faster chart interaction.

3. CHOKE ZERO/GAP CORRECTION
- Unified choke is treated as a step setting.
- A zero choke while the well is clearly flowing is treated as a suspicious template/blank zero.
- Missing choke settings are carried only inside the same well + test + source + sheet group.
- Choke values are not filled across separate tests or uploaded reports.
- Interactive and exported choke curves use step-line drawing.

4. TEST SEPARATION
- Curves break at detected test/report boundaries instead of connecting unrelated tests.
- Vertical dotted lines mark test/report boundaries in interactive charts and downloads.
- Boundaries are labelled Test 2, Test 3, etc.; a pure missing-data break is labelled Data gap.
- Time aggregation groups by well + test_id, so separate tests are never averaged together.

5. DISPLAY UNIT CONVERSION
Pressure:
- psi
- bar
- Formula: bar = psi x 0.0689475729

Temperature:
- Keep detected unit
- degC
- degF
- Formula: degF = degC x 9/5 + 32
- Formula: degC = (degF - 32) x 5/9

The parser/source values remain unchanged. Conversion is applied only to the working display/export copy. Axis labels are changed to the selected unit. Existing custom Y-axis ranges are converted when the display unit changes.

6. DEPLOYMENT
- Replace all files in the GitHub app with this package.
- Commit and reboot the Streamlit app.
- No manual cache-clear action is required.
