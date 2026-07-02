# v95 — Sidebar Scroll Restored to v92 Behavior

## Change

The sidebar scrollbar implementation introduced after v92 was removed. The application now returns to the proven v92 approach:

- Streamlit controls the sidebar height natively.
- `stSidebarContent` is the only explicitly scrollable sidebar surface.
- No fixed `100vh` wrapper or nested `overflow: hidden` rules are used.
- The scrollbar is 13 px wide with a visible thumb and theme-aware track.
- Mouse wheel, touchpad, thumb dragging, and keyboard scrolling act on the controls themselves.
- Extra bottom padding keeps the final controls reachable.

## Preserved functionality

All v94 functionality remains unchanged, including smart Y-axis ranges, comment clearance, continuation uploads, OCR, Production History, exports, Light/Dark themes, and parser behavior.

## Deployment

Replace the complete repository with this package and reboot the Streamlit app. Verify the startup log contains:

`v95-v92-sidebar-scroll-20260702`
