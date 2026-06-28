# Production Test Dashboard v83 — Detail and Production History

## User workflow

The application now has one simple **Analysis view** selector:

1. **Test detail** — plots every reading inside a short production test.
2. **Production history** — plots one stabilized result per detected test across months or years.

No extra history settings are required.

## Production-history engineering method

For each well and detected test period:

- The app sorts readings by time.
- Each selected signal is calculated independently as the average of its final **6 valid readings**.
- The point is placed at the test end date.
- All test points for the same well are connected.
- A dashed **3-test moving-average trend** shows decline or improvement.
- Only the first, latest, minimum, and maximum values are labelled; all other results remain available on hover.

This method avoids mixing thousands of within-test samples into a multi-year chart while preserving a repeatable stabilized-test convention.

## Performance

History reduction is executed only for the selected well(s) and selected signals. It does not globally cache uploaded confidential data.

Validation using the supplied multi-year ESP workbook:

- Raw readings: 3,672
- Detected tests: 54
- History points: 54
- Interactive point reduction: 98.5%
- History calculation after parsing: approximately 0.16 seconds in the validation environment

## Deployment

Replace the complete repository with this package. The new app build ID clears prior UI state. Parser files remain the validated comprehensive v82 parser build.
