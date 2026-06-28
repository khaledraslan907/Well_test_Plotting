# Production Test Dashboard v82

## Purpose

This deployment is prepared for sharing without embedding operational identifiers in the interface or repository package.

## Share-safe labels

`Share-safe labels` is enabled by default. It masks:

- Well identifiers as `Well 1`, `Well 2`, and so on.
- Uploaded filenames as `Uploaded file 1`, `Uploaded file 2`, and so on.
- Worksheet names as `Data table 1`, `Data table 2`, and so on.
- Test-unit/provider names as `Test Unit`.
- Sender, attachment, and image names.
- Matching identifiers inside notes, checks, captions, and test IDs.
- Raw OCR text in share-safe mode.

Engineering values are not changed. Parsing uses the original upload internally, while charts, tables, screenshots, and exports use masked labels.

Turn share-safe labels off only in a private deployment where original operational identifiers are required.

## Event time selector

Start and end time pickers now use explicit 15-minute increments. The dropdown menu and its scrollbar are styled for both Light and Dark themes, including browser-level popover menus.

Typed times remain supported for values outside the 15-minute list.

## Event-note visibility

Event and interval annotations are theme adaptive:

- Light theme: dark annotation colors.
- Dark theme: bright gold, cyan, coral, green, violet, and pink annotation colors.

The same colors are used in interactive charts, PNG exports, and PDF exports.

## Deployment

Upload all files in this directory together:

- `app.py`
- `tmu_parser.py`
- `tmu_parser_compat.py`
- `tmu_parser_legacy.py`
- `smart_tabular_v75.py`
- `requirements.txt`
- `packages.txt`
- `runtime.txt`
- `.streamlit/config.toml`

Do not mix these files with an older deployment.
