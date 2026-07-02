# Production Test Dashboard v99

## Purpose

v99 fixes the Streamlit exception raised while reopening a recovered dashboard PDF:

`st.session_state.ui_theme cannot be modified after the widget with key ui_theme is instantiated`

## Root cause

The PDF is parsed after the sidebar widgets have already been rendered. Streamlit does not allow code to overwrite the session-state key of an instantiated widget during that same run. The recovered PDF attempted to restore `ui_theme` immediately, so Streamlit rejected it and reported the file as skipped.

The same risk also applied to other saved controls such as continuation mode, test-gap hours, selected wells, selected signals, event layout and custom Y-axis settings.

## Fix

- All recovered widget-backed settings are now placed in a private pending-state container.
- The app requests one clean rerun.
- At the beginning of that rerun, before any widget is instantiated, all recovered settings are applied safely.
- Dataframes and internal recovery signatures continue to be stored normally.
- Both portable v97+ PDFs and older vector dashboard PDFs use the same safe restoration path.
- The legacy Light/Dark detection path also uses staged restoration.
- Starting a new analysis clears any pending restore.

## Scope control

No parser, chart, event-layout, theme-design, continuation, axis, export or data-processing behavior was changed. The v98 vector-PDF recovery remains intact.
