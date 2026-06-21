TMU Dashboard v57 - Unified Choke Conversion and Faster Reruns

CHOKE CONVERSION
================
The dashboard can now draw one choke line even when source files mix:
- Choke Opening (%)
- Choke Size (/64 in)
- Bare choke values with no written unit

Default calibration requested for this project:
- 100% opening = 128/64 in
- 50% opening = 64/64 in
- 25% opening = 32/64 in

Formula:
- Opening (%) = Choke size (/64) / Full-open size (/64) * 100
- Choke size (/64) = Opening (%) / 100 * Full-open size (/64)

The user can change the full-open calibration in the sidebar. This is important
because the relation is a project/equipment calibration and should not be
assumed to be universal for every choke design.

USER OPTIONS
============
1. Choke unit to plot
   - Opening (%) - combine both source units
   - Size (/64 in) - combine both source units
   - Keep source units separate

2. Full-open choke size (/64 in)
   - Default = 128
   - Adjustable from 1 to 256

3. When a choke value has no unit
   - Auto from surrounding source units
   - Treat as Opening (%)
   - Treat as Size (/64 in)

4. Also show original choke columns
   - Off by default for one clean converted curve
   - Can be enabled for audit/comparison

DATA SAFETY
===========
- Original choke percentage and /64 values are never overwritten.
- The unified curve is a separate derived plotting field.
- If both units exist at the same timestamp, the original value already in the
  selected target unit is used; the other unit only fills missing timestamps.
- A warning appears when both units disagree by more than 2 percentage points.
- A plain value such as "Choke = 64" is preserved as unit-ambiguous until the
  user chooses how to interpret it.

SPEED
=====
- Individual uploaded files remain cached.
- The concatenated and duplicate-merged dataset is now also cached in session.
- Changing choke unit, chart style, labels, filters, or export settings does not
  reparse all workbooks or repeat the duplicate merge.

DEPLOYMENT
==========
Replace all files in the GitHub app with the files from this ZIP, commit,
reboot Streamlit, and press "Re-parse uploaded files / clear cache" once.
