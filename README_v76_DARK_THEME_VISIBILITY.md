# Production Test Dashboard v76 — Dark Theme Visibility

## Problem corrected
The dashboard used application-level dark colors, but several Streamlit/browser controls retained their default light or low-contrast styling. The most visible cases were the sidebar scrollbar/scroll arrows and uploaded-file chips.

## Changes
- High-contrast vertical and horizontal scrollbars for the main page, sidebar, menus and tables.
- Visible scrollbar up/down and left/right buttons where the browser supports native scrollbar buttons.
- The sidebar scrollbar is kept available instead of becoming an almost invisible overlay.
- Explicit styling for sidebar collapse/expand arrows and tab/pagination scroll arrows.
- Dark-compatible uploaded-file chips, file names, sizes, file icons, remove buttons and add-file controls.
- Clear number-input plus/minus controls.
- Clear select, date, time, popover, menu, help and toolbar icons.
- Improved radio, checkbox, toggle and slider visibility.
- Dark-compatible dialogs, toasts, status panels, code blocks and Plotly modebar icons.
- Disabled controls remain readable rather than fading into the panel background.

## Scope
This is a UI-only update. The v75 fast parser and continuous-test plotting behavior are unchanged.

## Deployment
Replace the complete repository with this package and reboot the Streamlit application. The most important changed file is `app.py`.
