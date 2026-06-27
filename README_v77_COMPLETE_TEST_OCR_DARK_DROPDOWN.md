# Production Test Dashboard v77

## What v77 fixes

### 1. Complete long-test detection
The adaptive parser previously stopped at source row 114 in the S8-58 workbook, even though the same sheet continued to source row 154. The fast-path parser now checks whether its result reaches the workbook tail. If a credible result is incomplete, the compatibility engine is used and the full table is retained.

Validated result for `12-S8-58 (12-06-2026).xlsx`:
- 92 engineering readings
- Start: 12-Jun-2026 18:00
- End: 16-Jun-2026 16:00
- Test 1: 12-Jun 18:00 to 13-Jun 00:30
- Test 2: 15-Jun 01:30 to 16-Jun 16:00

### 2. Smart test segmentation
The app no longer overwrites parser-detected test IDs with a 72-hour UI gap. The default is **Smart parser detection**, which:
- preserves genuine production-test restarts after inactive gaps;
- keeps sparse daily SRP surveillance readings as one connected trend;
- allows an optional custom gap only when the user deliberately overrides detection.

### 3. CTU/HMI OCR corrections
For the uploaded 16-Jun-2026 14:35 image, v77 extracts:
- Weight: 21024 lbf
- Lt Weight: -1 lbf
- WHP: 29.16 psi
- Circulation Pressure: 693.99 psi
- Reel Depth: 10149.6 ft
- Reel Speed: -0.01 ft/min
- Fluid Flow: 0.00 bpm
- N2 Flow: 0 scf/min
- Fluid Total: 0.0 bbl
- N2 Total: 0 scf

The OCR now combines split adaptive-threshold tokens, preserves legitimate negative Lt Weight, and uses field-specific processing for red pressure digits and isolated zero values.

### 4. Automatic OCR context linking
When one uploaded test has readings close to the image timestamp, the OCR row inherits the unique nearby well and test ID. It remains review-required for value approval.

### 5. Clean signal dropdown
OCR/debug metadata such as `screen_area_ratio`, `screen_aspect_ratio`, `ocr_raw__...`, and `ocr_conf__...` is excluded from plotting options. Only engineering channels remain.

### 6. Dark dropdown visibility
Select and multiselect popovers are styled at the body portal level, including their inner menu surfaces, options, hover state, selected state, text and search input. This prevents white dropdown menus with nearly invisible text in Dark mode.

## Deployment
Replace the complete repository with this package, commit, and reboot the Streamlit app. The parser build ID changed, so old cached 59-row results are invalidated automatically.
