# Production Test Dashboard v88

## Production-history date axis

Production History no longer uses a fixed one-year tick interval. That setting could
leave the time axis blank when the selected history covered only days or months.

v88 builds explicit representative ticks from the actual test-average timestamps:

- up to 3 days: date and time
- up to 120 days: full date
- longer histories: month and year
- first and last test dates are always retained

The same adaptive date formatting is used in the interactive chart and PNG/PDF exports.
The chart-title helper also prevents titles such as `Well Well 1`.

## WhatsApp ZIP image status

The `Read images inside chat export ZIPs` switch processes only image files physically
stored inside the ZIP (`JPG`, `JPEG`, `PNG`, or `WEBP`). Text such as `image omitted`
does not contain image pixels and cannot be OCR processed.

v88 inspects every uploaded ZIP and reports one of the following:

- number of included image files and OCR rows extracted
- image files found but OCR could not interpret them
- images available but the ZIP-image option is disabled
- chat contains `image omitted` references but no actual image files

For the supplied `WhatsApp Chat - Sitra8-58 clean & test(3).zip`, inspection found:

- 0 image files
- 1 OPUS audio file
- 146 `image omitted` references
- 4 `document omitted` references
- 74 production-test rows parsed from `_chat.txt`

To OCR those omitted photos, export the WhatsApp chat again and choose **Include media**.

## Deployment

Replace the complete repository with this package, commit, and reboot the Streamlit app.
The v88 parser build ID invalidates previous upload caches.
