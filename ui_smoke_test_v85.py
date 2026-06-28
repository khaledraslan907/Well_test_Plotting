from pathlib import Path
import ast

source = Path(__file__).with_name("app.py").read_text(encoding="utf-8")
ast.parse(source)
required = [
    'APP_UI_BUILD_ID = "v85-scrollable-note-time-picker-ui-20260628"',
    'def scrollable_time_picker(',
    'step_minutes=15',
    'overflow-y: scroll !important;',
    'scrollbar-gutter: stable !important;',
    'key="note_start_time_picker"',
    'key="note_end_time_picker"',
]
missing = [item for item in required if item not in source]
if missing:
    raise SystemExit("Missing v85 UI fragments: " + ", ".join(missing))

# Native note time inputs must be gone; the separate manual-range inputs may remain.
for marker in [
    'note_start_time_picker = st.time_input(',
    'note_end_time_picker = st.time_input(',
]:
    if marker in source:
        raise SystemExit("Native note time input still present: " + marker)

print("v85 scrollable note time picker smoke test: PASS")
