from __future__ import annotations

import ast
import io
import py_compile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
APP_PATH = ROOT / "app.py"
APP = APP_PATH.read_text(encoding="utf-8")

for name in [
    "app.py", "tmu_parser.py", "tmu_parser_compat.py", "tmu_parser_legacy.py",
    "smart_tabular_v75.py", "history_analysis.py", "legacy_dashboard_pdf.py",
]:
    py_compile.compile(str(ROOT / name), doraise=True)

assert 'APP_UI_BUILD_ID = "v99-safe-session-restore-20260702"' in APP
assert 'PENDING_SESSION_RESTORE_KEY_V99 = "_pending_session_restore_v99"' in APP
assert 'queue_session_state_restore_v99({"ui_theme": _legacy_theme})' in APP
assert 'apply_pending_session_state_restore_v99()' in APP

module = ast.parse(APP)
wanted = {
    "queue_session_state_restore_v99",
    "apply_pending_session_state_restore_v99",
    "apply_portable_state_to_session",
    "apply_legacy_dashboard_state_to_session",
}
nodes = [n for n in module.body if isinstance(n, ast.FunctionDef) and n.name in wanted]

class LockedSessionState(dict):
    """Minimal Streamlit-like state that rejects writes to instantiated widget keys."""
    def __init__(self, *args, locked=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.locked = set(locked)

    def __setitem__(self, key, value):
        if key in self.locked:
            raise RuntimeError(f"widget key is locked: {key}")
        return super().__setitem__(key, value)

class FakeStreamlit:
    def __init__(self, state):
        self.session_state = state

locked_widget_keys = {
    "ui_theme", "continue_current_test_v93", "test_gap_hours_v97",
    "selected_features_v58", "plot_signal_order_v92_state",
    "selected_wells_v97", "analysis_view_v97", "x_axis_mode_v97",
    "event_label_layout", "detail_use_custom_y_scale_v97",
}
state = LockedSessionState({"ui_theme": "Light"}, locked=locked_widget_keys)
fake_st = FakeStreamlit(state)
ns = {
    "st": fake_st,
    "pd": pd,
    "PORTABLE_SESSION_KEYS": [
        "ui_theme", "continue_current_test_v93", "test_gap_hours_v97",
        "selected_features_v58", "plot_signal_order_v92_state",
        "selected_wells_v97", "analysis_view_v97", "x_axis_mode_v97",
        "event_label_layout", "detail_use_custom_y_scale_v97",
    ],
    "PORTABLE_DYNAMIC_PREFIXES": ("ymin_", "ymax_", "label_decimals_", "display_label_", "dual_"),
    "PENDING_SESSION_RESTORE_KEY_V99": "_pending_session_restore_v99",
    "feature_key_text": lambda value: str(value),
}
for node in nodes:
    exec(compile(ast.Module(body=[node], type_ignores=[]), "session_restore_helpers", "exec"), ns)

# Legacy PDF recovery happens after the theme widget exists. It must stage every
# widget-backed value rather than writing any of those keys in the current run.
legacy_state = {
    "theme": "Dark",
    "selected_features": ["gas_rate_mmscfd", "gross_rate_bpd"],
    "well": "B3C18-7",
    "x_axis_mode": "Compressed real dates - remove empty gaps",
    "manual_events": [{"datetime": "2026-06-30T19:00:00", "label": "SIWHP 3300 PSI"}],
    "operation_intervals": [{
        "start": "2026-06-30T20:30:00", "end": "2026-07-01T13:00:00",
        "label": "Closed the Well",
    }],
    "chart_title": "Well B3C18-7",
}
assert ns["apply_legacy_dashboard_state_to_session"](legacy_state, "legacy-signature") is True
pending = state["_pending_session_restore_v99"]
assert pending["ui_theme"] == "Dark"
assert pending["selected_features_v58"] == ["gas_rate_mmscfd", "gross_rate_bpd"]
assert pending["selected_wells_v97"] == ["B3C18-7"]
assert pending["manual_events_table"][0]["label"] == "SIWHP 3300 PSI"
assert pending["operation_intervals_table"][0]["label"] == "Closed the Well"
assert state["ui_theme"] == "Light"  # current instantiated widget was untouched

# Simulate the clean rerun before widgets are instantiated.
state.locked.clear()
ns["apply_pending_session_state_restore_v99"]()
assert state["ui_theme"] == "Dark"
assert state["selected_wells_v97"] == ["B3C18-7"]
assert "_pending_session_restore_v99" not in state

# Portable v97+ PDF restoration must use the same safe staging path for every
# saved control, not only the theme.
state.clear()
state.update({"ui_theme": "Light"})
state.locked = set(locked_widget_keys)
frame = pd.DataFrame({
    "well": ["B3C18-7", "B3C18-7"],
    "datetime": pd.to_datetime(["2026-07-02 10:00", "2026-07-02 11:00"]),
    "gross_rate_bpd": [118.0, 121.0],
})
portable = {
    "manifest": {
        "theme": "Dark",
        "chart_title": "Well B3C18-7",
        "ui_state": {
            "ui_theme": "Dark",
            "continue_current_test_v93": True,
            "test_gap_hours_v97": 12.0,
            "selected_features_v58": ["gross_rate_bpd"],
        },
        "manual_events": [],
        "operation_intervals": [],
        "custom_y_ranges": {"gross_rate_bpd": [0, 200]},
    },
    "data": frame,
}
assert ns["apply_portable_state_to_session"](portable, "portable-signature") is True
pending = state["_pending_session_restore_v99"]
assert pending["ui_theme"] == "Dark"
assert pending["continue_current_test_v93"] is True
assert pending["test_gap_hours_v97"] == 12.0
assert pending["ymin_gross_rate_bpd"] == 0.0
assert pending["ymax_gross_rate_bpd"] == 200.0
assert len(state["continued_test_data_v93"]) == 2
assert state["ui_theme"] == "Light"  # still untouched until the next rerun

print("PASS - all deployment Python files compile")
print("PASS - legacy PDF restoration does not modify instantiated widget keys")
print("PASS - portable PDF restoration stages all saved widget controls")
print("PASS - staged theme, events, intervals, wells and curves apply before widgets")
