TMU DASHBOARD v71 — LIGHT / DARK UI REFRESH
============================================

What changed
------------
- Theme choices are now only Light and Dark.
- Dark mode uses high-contrast text, inputs, uploaders, menus, alerts, tables, and dark Plotly canvases.
- The old barrel symbol was replaced with a clean production-trend chart icon.
- The title is now: TMU Production Test Analysis & Visualization.
- The header badges/build labels were removed for a simpler professional appearance.
- The page and sidebar backgrounds now use a subtle engineering grid and restrained gradient.
- The underlying parser, OCR, gas-balance logic, plotting workflow, and engineering feature colors were not changed.

Deployment
----------
Replace the repository files with this folder, commit, and reboot the Streamlit app.
Do not mix app.py with an older parser package.

Validation
----------
Run:
    python ui_smoke_test_v71.py
    python self_test_v70.py
