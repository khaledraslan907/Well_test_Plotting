# Production Test Dashboard v84

## Production-history change

Production History now uses one arithmetic-average point for each detected test.
For every selected signal, the app averages all valid readings inside that test,
places the point at the test end date, and connects all test points for the same
well with one continuous chronological line.

The separate dashed moving-average line from v83 has been removed.

## Default labels

The default history label policy is:

- first test point;
- every 20th test point;
- last test point.

For 54 detected tests, labels appear at point indices 0, 20, 40, and 53.
The user can instead select First and last only, Clean readable, or Off.

## Simple history controls

Production History keeps only two chart controls:

1. Optional custom Y-axis minimum and maximum for each selected signal.
2. Value-label density.

All other history behavior is automatic to keep the interface fast and easy.

## Performance

The history reducer processes only the already parsed table and returns one row
per test. On the supplied multi-year workbook:

- 3,672 raw readings;
- 54 detected tests;
- 54 plotted history points;
- approximately 0.11 seconds for history averaging after parsing.

The optional Y-axis preview is calculated only when custom Y-axis ranges are
enabled, avoiding duplicate work during normal interaction.

## Deployment

Replace the complete repository with this package and reboot the Streamlit app.
The parser remains the validated v82/v79-compatible parser; v84 changes the
history calculation and plotting interface only.
