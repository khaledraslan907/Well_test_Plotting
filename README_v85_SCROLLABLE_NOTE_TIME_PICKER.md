# Production Test Dashboard v85 — Scrollable Note Time Picker

## Change

The Start time and End time controls inside **Events & Notes** now use a dedicated
scrollable 24-hour selector instead of the browser-dependent native time popup.

- 96 choices per day at 15-minute intervals.
- A visible vertical scrollbar in both Light and Dark themes.
- The current/default minute is preserved even when it is not exactly on a
  15-minute boundary.
- Exact custom times can still be entered in the optional typed-time field.
- The popover height is limited so the list always scrolls instead of extending
  beyond the page.

No parser, production-history, privacy, OCR, or export logic was changed.
