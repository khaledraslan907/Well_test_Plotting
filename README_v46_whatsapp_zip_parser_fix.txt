v46 changes:
- Stricter WhatsApp ZIP parser: keeps only real TMU numeric report rows and drops chat comments/rubbish rows.
- Fixed empty rows from messages that mentioned oil/water/N2 but had no actual numeric readings.
- CTU/HMI OCR is stricter: random images are ignored unless enough plausible CTU fields are detected.
- Added OCR value plausibility cleanup, including decimal restoration such as 12096 -> 120.96 when applicable.
- Removed repeated safety/explanation captions from the UI.
- Increased default image OCR limit to 1000 per ZIP.
- Default filters now select one well/test first, not many wells/tests, to avoid crowded empty comparison plots.
- Chart title now resets when a new uploaded dataset is used, so an old title does not stay from the previous file.
