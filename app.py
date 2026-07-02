from __future__ import annotations
import io
import json
import re
import zipfile
import traceback
import gc
import hashlib
import html
import math
from pathlib import Path
from datetime import date, datetime, time
from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Optional lightweight drag-and-drop ordering for plotted signals. The package is
# included in requirements.txt, while the fallback keeps the app usable if a
# deployment has not finished installing it yet.
try:
    from streamlit_sortables import sort_items as _sort_items
except Exception:
    _sort_items = None

# Import parser module safely.  Streamlit Cloud often shows a redacted ImportError
# if app.py was updated but tmu_parser.py is still an older file.  This block keeps
# the app running and shows a clear message instead of a redacted crash.
try:
    import tmu_parser as _tmu_parser
except Exception as _parser_import_error:
    st.set_page_config(page_title="Production Test Analysis & Visualization", page_icon="📈", layout="wide")
    st.error("Could not import tmu_parser.py. Make sure app.py and tmu_parser.py are in the same folder and were both updated from the same ZIP package.")
    st.code("""
Required files in the same folder:
- app.py
- tmu_parser.py
- requirements.txt

On Streamlit Cloud: commit/push all three files, then reboot the app.
""".strip())
    st.exception(_parser_import_error)
    st.stop()

# Required parser functions from every supported parser build.
_missing_required = [
    name for name in [
        "apply_fill_method", "available_numeric_columns", "column_label",
        "load_tabular_file", "parse_many_tmu_messages",
    ]
    if not hasattr(_tmu_parser, name)
]
if _missing_required:
    st.set_page_config(page_title="Production Test Analysis & Visualization", page_icon="📈", layout="wide")
    st.error("Your tmu_parser.py is older than app.py. Update tmu_parser.py from the latest package.")
    st.code("Missing parser functions: " + ", ".join(_missing_required))
    st.stop()

apply_fill_method = _tmu_parser.apply_fill_method
available_numeric_columns = _tmu_parser.available_numeric_columns
_parser_column_label = _tmu_parser.column_label
load_tabular_file = _tmu_parser.load_tabular_file
parse_many_tmu_messages = _tmu_parser.parse_many_tmu_messages
parse_whatsapp_plain_or_export_text = getattr(
    _tmu_parser, "parse_whatsapp_plain_or_export_text", parse_many_tmu_messages
)
PARSER_BUILD_ID = getattr(_tmu_parser, 'PARSER_BUILD_ID', 'v70')
assign_test_ids = getattr(_tmu_parser, "assign_test_ids", lambda df, gap_hours=12.0: df)
normalize_ctu_ocr_signals = getattr(_tmu_parser, "normalize_ctu_ocr_signals", lambda df: df)

from history_analysis import build_production_history

DISPLAY_LABEL_FILE = Path(__file__).with_name("user_display_labels.json")

def load_display_label_overrides() -> dict:
    try:
        if DISPLAY_LABEL_FILE.exists():
            data = json.loads(DISPLAY_LABEL_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}

def save_display_label_overrides(overrides: dict) -> None:
    safe = {str(k): str(v).strip() for k, v in (overrides or {}).items() if str(v).strip()}
    DISPLAY_LABEL_FILE.write_text(json.dumps(safe, indent=2, sort_keys=True), encoding="utf-8")

def column_label(col) -> str:
    # Stable internal feature names are kept while only the displayed unit changes.
    # This prevents feature selections from resetting when the user changes units.
    if str(col) == "choke_unified":
        return str(st.session_state.get("choke_unified_label", "Unified Choke"))
    overrides = st.session_state.get("display_label_overrides")
    if overrides is None:
        overrides = load_display_label_overrides()
        st.session_state["display_label_overrides"] = overrides
    base = str(overrides.get(str(col), _parser_column_label(col)))
    unit_labels = st.session_state.get("unit_display_labels", {}) or {}
    return str(unit_labels.get(str(col), base))



PRESSURE_COLUMNS_PSI = {
    "whp_psi", "flp_psi", "flow_press_psi", "sep_p_psi", "pumping_pressure_psi",
    "ctu_wellhead_pressure_psi", "ctu_circulation_pressure_psi", "ct_pressure_psi",
    "ds_press_psi", "us_press_psi", "pump_intake_pressure_psi",
    "pump_discharge_pressure_psi", "pi_psi", "pd_psi",
    "u2_pass_side_pump_pressure_psi",
}

def _replace_unit_in_label(label: str, unit_text: str) -> str:
    txt = re.sub(r"\s*\((?:psi|bar|°C|°F|C|F)\)\s*$", "", str(label), flags=re.I).strip()
    return f"{txt} ({unit_text})"

def apply_display_unit_conversions(df: pd.DataFrame, pressure_unit: str, temperature_unit: str) -> pd.DataFrame:
    """Convert only the working display copy; parser/source values remain cached unchanged."""
    if df is None or df.empty:
        st.session_state["unit_display_labels"] = {}
        return df
    out = df.copy(deep=False)
    labels = {}

    # Canonical pressure fields are stored in psi by the parser.
    for col in [c for c in out.columns if c in PRESSURE_COLUMNS_PSI or str(c).endswith("_psi")]:
        vals = pd.to_numeric(out[col], errors="coerce")
        if pressure_unit == "bar":
            out[col] = vals * 0.0689475729
            labels[col] = _replace_unit_in_label(_parser_column_label(col), "bar")
        else:
            out[col] = vals
            labels[col] = _replace_unit_in_label(_parser_column_label(col), "psi")

    # Temperature field names state their canonical source unit. Keep the same
    # internal feature key so changing °C/°F cannot reset the selected features.
    temp_cols = [c for c in out.columns if str(c).endswith("_c") or str(c).endswith("_f")]
    for col in temp_cols:
        vals = pd.to_numeric(out[col], errors="coerce")
        source_is_c = str(col).endswith("_c")
        if temperature_unit == "°C":
            out[col] = vals if source_is_c else (vals - 32.0) * (5.0 / 9.0)
            labels[col] = _replace_unit_in_label(_parser_column_label(col), "°C")
        elif temperature_unit == "°F":
            out[col] = vals * (9.0 / 5.0) + 32.0 if source_is_c else vals
            labels[col] = _replace_unit_in_label(_parser_column_label(col), "°F")

    st.session_state["unit_display_labels"] = labels
    return out

def canonical_key(s: object) -> str:
    """Stable normalized key for user-taught aliases.

    Local fallback is kept here so the generic mapping panel still works if an
    older parser file is accidentally deployed.
    """
    if hasattr(_tmu_parser, "canonical_key"):
        return _tmu_parser.canonical_key(s)
    txt = "" if s is None else str(s).strip().lower()
    txt = txt.replace("&", " and ")
    txt = re.sub(r"[^a-z0-9]+", "_", txt)
    return re.sub(r"_+", "_", txt).strip("_")


def standard_column_options(include_meta: bool = False) -> dict:
    """Return standard fields available in the mapping dropdown."""
    if hasattr(_tmu_parser, "standard_column_options"):
        return _tmu_parser.standard_column_options(include_meta=include_meta)

    labels = dict(getattr(_tmu_parser, "COLUMN_LABELS", {}))
    # Fallback list for older parser builds.  These are safe internal names used
    # by this app; users can map new customer abbreviations to any of them.
    labels.update({
        "choke_pct": "Choke Opening (%)",
        "choke_size_64": "Choke Size (/64 in)",
        "choke_ambiguous": "Choke (unit not stated)",
        "choke_unified": "Unified Choke",
        "whp_psi": "WHP (psi)",
        "flp_psi": "FLP (psi)",
        "flow_press_psi": "Flow Pressure (psi)",
        "flow_temp_c": "Flow Temp (°C)",
        "sep_p_psi": "Separator Pressure (psi)",
        "pumping_pressure_psi": "Pumping Pressure (psi)",
        "gas_rate_mmscfd": "Total Gas Rate (MMSCF/D)",
        "gas_formation_mmscfd": "Formation Gas Rate (MMSCF/D)",
        "gross_rate_bpd": "Gross Rate (BBL/D)",
        "oil_rate_stbd": "Oil Rate (STB/D)",
        "water_rate_bpd": "Water Rate (BBL/D)",
        "bsw_pct": "BS&W (%)",
        "wlr_s_pct": "WLR (%)",
        "salinity_kppm": "Salinity (K ppm NaCl)",
        "gor_scf_bbl": "GOR (scf/bbl)",
        "h2s_ppm": "H2S (ppm)",
        "co2_mole_pct": "CO2 (mole %)",
        "oil_api": "Oil Gravity (API)",
        "gas_sg": "Gas Specific Gravity",
        "pump_freq_hz": "Pump Frequency (Hz)",
        "motor_current_amp": "Motor Current (A)",
        "ama_current_amp": "AMA / Motor Current (A)",
        "pi_psi": "Pi / Intake Pressure (psi)",
        "pd_psi": "Pd / Discharge Pressure (psi)",
        "ti_f": "Ti / Intake Temperature (°F)",
        "tm_f": "Tm / Motor Temperature (°F)",
        "vx": "Vibration X",
        "vy": "Vibration Y",
        "vz": "Vibration Z",
    })
    if include_meta:
        labels.update({"well": "Well", "date": "Date", "time": "Time", "datetime": "Date/Time"})
    return dict(sorted(labels.items(), key=lambda kv: kv[1].lower()))


def _to_numeric_for_mapping(series: pd.Series) -> pd.Series:
    """Convert mapped raw numeric columns safely."""
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.extract(r"([-+]?\d+(?:\.\d+)?)", expand=False),
        errors="coerce",
    )


def apply_user_column_mappings(df: pd.DataFrame, mapping: Optional[dict] = None) -> pd.DataFrame:
    """Apply user-taught aliases; fallback for older parser builds."""
    if hasattr(_tmu_parser, "apply_user_column_mappings"):
        return _tmu_parser.apply_user_column_mappings(df, mapping)

    if not mapping or df is None or df.empty:
        return df
    out = df.copy()
    drop_cols = []
    for col in list(out.columns):
        target = mapping.get(str(col)) or mapping.get(canonical_key(col))
        if not target or target == "__keep__":
            continue
        if target == "__drop__":
            drop_cols.append(col)
            continue
        if col == target:
            continue
        vals = _to_numeric_for_mapping(out[col])
        if target in out.columns:
            out[target] = pd.to_numeric(out[target], errors="coerce").combine_first(vals)
        else:
            out[target] = vals
        drop_cols.append(col)
    if drop_cols:
        out = out.drop(columns=[c for c in drop_cols if c in out.columns], errors="ignore")
    return out


FEATURE_COLORS = {
    # Production rates
    "gross_rate_bpd": "#607D8B",          # steel grey / total liquid
    "qgross_s_bpd": "#607D8B",
    "oil_rate_stbd": "#2E7D32",          # oil green
    "qoil_s_stbd": "#2E7D32",
    "qoil_a_bpd": "#43A047",
    "water_rate_bpd": "#1976D2",         # water blue
    "qwat_s_bpd": "#1976D2",
    "qwat_a_bpd": "#42A5F5",
    "gas_rate_mmscfd": "#00ACC1",        # gas cyan
    "qgas_s_mmscfd": "#00ACC1",
    "qgas_a_mmcfd": "#26C6DA",
    "gas_formation_mmscfd": "#00897B",   # formation gas teal
    "n2_rate_mmscfd": "#7E57C2",
    "n2_rate_scfm": "#7E57C2",           # nitrogen violet

    # Surface / pumping pressures
    "whp_psi": "#C62828",                # wellhead red
    "flp_psi": "#F9A825",                # flowline amber
    "flow_press_psi": "#EF6C00",
    "sep_p_psi": "#F57C00",              # separator orange
    "pumping_pressure_psi": "#E65100",   # pumping deep orange
    "pump_intake_pressure_psi": "#D84315",
    "pump_discharge_pressure_psi": "#8E0000",
    "pi_psi": "#D84315",
    "pd_psi": "#8E0000",
    "ct_pressure_psi": "#AD1457",
    "mpfm_press_psig": "#8E24AA",
    "dp_mbar": "#5E35B1",

    # Fluid quality / properties
    "bsw_pct": "#8E44AD",
    "wlr_s_pct": "#7B1FA2",
    "salinity_kppm": "#8D6E63",
    "oil_api": "#6D4C41",
    "gas_sg": "#546E7A",
    "gor_s_scf_stb": "#EC407A",
    "gvf_a_pct": "#7CB342",
    "h2s_ppm": "#B71C1C",
    "co2_mole_pct": "#455A64",

    # Choke / temperatures / device telemetry
    "choke_pct": "#C49A44",
    "choke_size_64": "#C49A44",
    "choke_ambiguous": "#C49A44",
    "choke_unified": "#C49A44",
    "flow_temp_c": "#FBC02D",
    "mpfm_temp_f": "#FFB300",
    "pump_freq_hz": "#3F6E8A",
    "motor_current_amp": "#AD1457",
    "stroke_length_in": "#00897B",
    "stroke_rate_spm": "#3949AB",
    "peak_load_lbf": "#C62828",
    "minimum_load_lbf": "#EF6C00",
    "ama_current_amp": "#AD1457",
    "tm_f": "#F57F17",
    "ti_f": "#F9A825",
    "vx": "#5C6BC0",
    "vy": "#26A69A",
    "vz": "#AB47BC",

    # CT / cumulative values
    "oil_cum_bbl": "#558B2F",
    "water_cum_bbl": "#1565C0",
    "motor_ama_amp": "#AD1457",
    "motor_temp_f": "#F57F17",
    "motor_temp_c": "#F57F17",
    "ctu_wellhead_pressure_psi": "#C62828",
    "ctu_circulation_pressure_psi": "#E65100",
    "ctu_fluid_rate_bpm": "#1976D2",
    "ctu_n2_rate_scfm": "#7E57C2",
    "ctu_fluid_total_bbl": "#1565C0",
    "ctu_n2_total_scf": "#5E35B1",
    "ctu_reel_depth_ft": "#00897B",
    "ct_depth_m": "#00897B",
    "ct_running_speed_ftmin": "#5C6BC0",
    "ct_pipe_weight_lbf": "#607D8B",
}

WELL_COLORS = [
    "#2F6D8A", "#C62828", "#2E7D32", "#7E57C2", "#F57C00",
    "#00ACC1", "#8D6E63", "#AD1457", "#7CB342", "#546E7A",
    "#C49A44", "#00897B",
]

LIGHT_FEATURE_COLOR_OVERRIDES = {
    "gross_rate_bpd": "#365D70",
    "qgross_s_bpd": "#365D70",
    "gas_rate_mmscfd": "#007F91",
    "qgas_s_mmscfd": "#007F91",
    "qgas_a_mmcfd": "#008FA3",
    "water_rate_bpd": "#075DAE",
    "qwat_s_bpd": "#075DAE",
    "oil_rate_stbd": "#176B29",
    "qoil_s_stbd": "#176B29",
}

def feature_color(feature_name: str, fallback_index: int = 0) -> str:
    if globals().get("ACTIVE_THEME_NAME") == "Light":
        if feature_name in LIGHT_FEATURE_COLOR_OVERRIDES:
            return LIGHT_FEATURE_COLOR_OVERRIDES[feature_name]
    return FEATURE_COLORS.get(feature_name, WELL_COLORS[fallback_index % len(WELL_COLORS)])

def well_color(index: int) -> str:
    return WELL_COLORS[index % len(WELL_COLORS)]

def is_aligned_elapsed_mode(x_axis_mode: str) -> bool:
    return str(x_axis_mode or "").startswith("Aligned elapsed")

def is_compressed_real_date_mode(x_axis_mode: str) -> bool:
    return str(x_axis_mode or "").startswith("Compressed real dates")

def chart_title_from_data(df, custom_title: str = "") -> str:
    """Default chart title: well name(s) only.

    Dates are intentionally not included because field reports usually need the
    well name in the main title and the date/time on the x-axis.  The user can
    override the title from the sidebar.
    """
    custom_title = str(custom_title or "").strip()
    if custom_title:
        return custom_title

    if "well" not in df.columns or df.empty:
        return "Well Production Test"

    wells = [str(w).strip() for w in df["well"].dropna().astype(str).unique()]
    wells = [w for w in wells if w and w.lower() != "unknown"]

    if len(wells) == 1:
        return well_title_text(wells[0])
    if len(wells) > 1:
        shown = " vs ".join(wells[:5])
        suffix = "" if len(wells) <= 5 else f" +{len(wells) - 5} more"
        return shown + suffix
    return "Well Production Test"


st.set_page_config(
    page_title="Production Test Analysis & Visualization",
    page_icon="📈",
    layout="wide",
)

APP_UI_BUILD_ID = "v99-safe-session-restore-20260702"
print(f"Starting Production Test Dashboard: {APP_UI_BUILD_ID} | parser={PARSER_BUILD_ID}")

PORTABLE_STATE_MAGIC = "CORELYTIX_PRODUCTION_TEST_ANALYSIS"
PORTABLE_STATE_SCHEMA = 1
PORTABLE_STATE_ATTACHMENT = "corelytix_production_test_state_v1.zip"

# Streamlit does not allow a widget-backed session key to be changed after that
# widget has been instantiated in the current run. PDF state is discovered only
# after the upload widget has rendered, so recovered controls are staged here and
# applied at the very start of the next rerun, before any widget is created.
PENDING_SESSION_RESTORE_KEY_V99 = "_pending_session_restore_v99"


def queue_session_state_restore_v99(values: dict) -> None:
    """Stage recovered control state without touching live widget keys."""
    if not isinstance(values, dict) or not values:
        return
    current = st.session_state.get(PENDING_SESSION_RESTORE_KEY_V99)
    merged = dict(current) if isinstance(current, dict) else {}
    merged.update({str(key): value for key, value in values.items()})
    st.session_state[PENDING_SESSION_RESTORE_KEY_V99] = merged


def apply_pending_session_state_restore_v99() -> None:
    """Apply staged state before any Streamlit widget is instantiated."""
    pending = st.session_state.pop(PENDING_SESSION_RESTORE_KEY_V99, None)
    if not isinstance(pending, dict):
        return
    for key, value in pending.items():
        st.session_state[str(key)] = value

# These controls are safe to restore before their widgets are created. Dynamic
# data-dependent selections are reconciled against the newly restored dataframe
# before Streamlit renders the corresponding widget.
PORTABLE_SESSION_KEYS = [
    "ui_theme", "continue_current_test_v93", "test_gap_hours_v97",
    "pressure_display_unit_v58", "temperature_display_unit_v58",
    "analysis_view_v97", "time_filter_mode_v97",
    "time_aggregation_interval_preset", "time_aggregation_interval_custom_value",
    "x_axis_tick_interval_preset", "x_axis_tick_interval_custom_value",
    "x_axis_mode_v97", "continuous_gap_hours_v97", "compressed_gap_hours_v97",
    "selected_wells_v97", "selected_features_v58", "plot_signal_order_v92_state",
    "history_use_custom_y_scale", "detail_use_custom_y_scale_v97",
    "fill_method_v97", "hide_zero_flow_rows_v97", "plot_mode_v97",
    "show_points_v97", "value_label_mode_v97", "detail_value_label_step",
    "history_value_label_mode", "history_value_label_step",
    "label_decimals_default_v97", "event_label_layout",
    "auto_hide_crowded_notes_v97", "max_visible_notes_per_chart_v97",
    "enable_drag_annotations_v97",
    "choke_plot_mode_v97", "choke_full_open_64_v97",
    "ambiguous_choke_unit_v97", "treat_zero_choke_as_missing_v97",
    "display_label_overrides", "choke_unified_label",
]
PORTABLE_DYNAMIC_PREFIXES = ("ymin_", "ymax_", "label_decimals_", "display_label_", "dual_")


def _portable_json_value(value):
    """Convert Streamlit/pandas values into safe JSON-compatible objects."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date, time)):
        return value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _portable_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_portable_json_value(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _portable_event_records(records, datetime_fields):
    cleaned = []
    for record in records or []:
        if not isinstance(record, dict):
            continue
        item = {str(k): _portable_json_value(v) for k, v in record.items()}
        for field in datetime_fields:
            if field in item and item[field]:
                try:
                    item[field] = pd.Timestamp(item[field]).isoformat()
                except Exception:
                    pass
        cleaned.append(item)
    return cleaned


def build_portable_state_zip(dataframe: pd.DataFrame, *, ui_state: dict, chart_title: str,
                             manual_events: list, operation_intervals: list,
                             custom_y_ranges: Optional[dict] = None) -> bytes:
    """Create a safe ZIP payload embedded in exported PDFs.

    CSV is intentionally used instead of pickle so an uploaded PDF never causes
    executable Python objects to be deserialized. The manifest records dtypes and
    datetime columns for a faithful dataframe reconstruction.
    """
    frame = dataframe.copy() if dataframe is not None else pd.DataFrame()
    datetime_columns = []
    dtype_map = {}
    export_frame = frame.copy()
    for col in export_frame.columns:
        series = export_frame[col]
        dtype_map[str(col)] = str(series.dtype)
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            datetime_columns.append(str(col))
            export_frame[col] = pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        elif pd.api.types.is_timedelta64_dtype(series.dtype):
            export_frame[col] = series.astype(str)
        elif pd.api.types.is_object_dtype(series.dtype):
            # Preserve date/time objects and mixed report metadata predictably.
            export_frame[col] = series.map(_portable_json_value)

    manifest = {
        "magic": PORTABLE_STATE_MAGIC,
        "schema": PORTABLE_STATE_SCHEMA,
        "app_build": APP_UI_BUILD_ID,
        "parser_build": PARSER_BUILD_ID,
        "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "chart_title": str(chart_title or ""),
        "ui_state": _portable_json_value(ui_state or {}),
        "manual_events": _portable_event_records(manual_events, ["datetime"]),
        "operation_intervals": _portable_event_records(operation_intervals, ["start", "end"]),
        "custom_y_ranges": _portable_json_value(custom_y_ranges or {}),
        "data": {
            "rows": int(len(export_frame)),
            "columns": [str(c) for c in export_frame.columns],
            "dtypes": dtype_map,
            "datetime_columns": datetime_columns,
            "file": "analysis_data.csv",
        },
    }

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, separators=(",", ":")))
        zf.writestr("analysis_data.csv", export_frame.to_csv(index=False, lineterminator="\n"))
    return output.getvalue()


def attach_portable_state_to_pdf(pdf_bytes: bytes, state_zip_bytes: bytes) -> bytes:
    """Embed the complete recoverable analysis state without changing PDF pages."""
    if not pdf_bytes or not state_zip_bytes:
        return pdf_bytes
    try:
        from pypdf import PdfReader, PdfWriter
        reader = PdfReader(io.BytesIO(pdf_bytes))
        writer = PdfWriter()
        writer.clone_document_from_reader(reader)
        writer.add_attachment(PORTABLE_STATE_ATTACHMENT, state_zip_bytes)
        metadata = {
            "/CorelytixPortableState": "1",
            "/CorelytixStateSchema": str(PORTABLE_STATE_SCHEMA),
            "/CorelytixAppBuild": APP_UI_BUILD_ID,
        }
        try:
            existing = dict(reader.metadata or {})
            existing.update(metadata)
            writer.add_metadata({str(k): str(v) for k, v in existing.items() if v is not None})
        except Exception:
            writer.add_metadata(metadata)
        output = io.BytesIO()
        writer.write(output)
        return output.getvalue()
    except Exception:
        # Never block a normal PDF download because an attachment library is
        # unavailable; the PDF remains readable, only reopening state is absent.
        return pdf_bytes


def read_portable_state_from_pdf(file_name: str, pdf_bytes: bytes) -> Optional[dict]:
    """Read and validate an embedded portable analysis package from a PDF."""
    if Path(str(file_name)).suffix.lower() != ".pdf" or not pdf_bytes:
        return None
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        attachment_values = reader.attachments.get(PORTABLE_STATE_ATTACHMENT, [])
        if isinstance(attachment_values, (bytes, bytearray)):
            attachment_values = [bytes(attachment_values)]
        for payload_bytes in attachment_values or []:
            with zipfile.ZipFile(io.BytesIO(payload_bytes)) as zf:
                manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
                if manifest.get("magic") != PORTABLE_STATE_MAGIC:
                    continue
                if int(manifest.get("schema", 0) or 0) != PORTABLE_STATE_SCHEMA:
                    continue
                data_info = manifest.get("data", {}) or {}
                csv_name = str(data_info.get("file") or "analysis_data.csv")
                frame = pd.read_csv(io.BytesIO(zf.read(csv_name)), low_memory=False)
                expected_columns = [str(c) for c in data_info.get("columns", [])]
                if expected_columns and len(expected_columns) == len(frame.columns):
                    frame.columns = expected_columns
                for col in data_info.get("datetime_columns", []) or []:
                    if col in frame.columns:
                        frame[col] = pd.to_datetime(frame[col], errors="coerce")
                dtype_map = data_info.get("dtypes", {}) or {}
                for col, dtype_name in dtype_map.items():
                    if col not in frame.columns or col in set(data_info.get("datetime_columns", []) or []):
                        continue
                    dtype_text = str(dtype_name).lower()
                    try:
                        if any(token in dtype_text for token in ["int", "float", "double", "decimal"]):
                            frame[col] = pd.to_numeric(frame[col], errors="coerce")
                        elif "bool" in dtype_text:
                            frame[col] = frame[col].astype(str).str.lower().map({"true": True, "false": False})
                    except Exception:
                        pass
                return {"manifest": manifest, "data": frame}
    except Exception:
        return None
    return None


def infer_legacy_pdf_theme(pdf_bytes: bytes) -> Optional[str]:
    """Best-effort Light/Dark detection for older PDFs without embedded state."""
    try:
        import pypdfium2 as pdfium
        from PIL import ImageStat
        doc = pdfium.PdfDocument(pdf_bytes)
        if len(doc) == 0:
            return None
        bitmap = doc[0].render(scale=0.35)
        image = bitmap.to_pil().convert("L")
        width, height = image.size
        crop = image.crop((int(width * 0.08), int(height * 0.08), int(width * 0.92), int(height * 0.88)))
        luminance = float(ImageStat.Stat(crop).mean[0])
        return "Dark" if luminance < 118 else "Light"
    except Exception:
        return None


def apply_portable_state_to_session(portable: dict, signature: str) -> bool:
    """Stage recovered PDF controls and request one clean pre-widget rerun."""
    if not portable or st.session_state.get("_portable_state_applied_v97") == signature:
        return False
    manifest = portable.get("manifest", {}) or {}
    ui_state = manifest.get("ui_state", {}) or {}
    pending_state = {}
    for key, value in ui_state.items():
        if key in PORTABLE_SESSION_KEYS or str(key).startswith(PORTABLE_DYNAMIC_PREFIXES):
            pending_state[str(key)] = value

    theme = str(ui_state.get("ui_theme") or manifest.get("theme") or "")
    if theme in {"Light", "Dark"}:
        pending_state["ui_theme"] = theme

    events = []
    for item in manifest.get("manual_events", []) or []:
        if not isinstance(item, dict):
            continue
        restored = dict(item)
        if restored.get("datetime"):
            restored["datetime"] = pd.Timestamp(restored["datetime"])
        events.append(restored)
    intervals = []
    for item in manifest.get("operation_intervals", []) or []:
        if not isinstance(item, dict):
            continue
        restored = dict(item)
        if restored.get("start"):
            restored["start"] = pd.Timestamp(restored["start"])
        if restored.get("end"):
            restored["end"] = pd.Timestamp(restored["end"])
        intervals.append(restored)
    pending_state["manual_events_table"] = events
    pending_state["operation_intervals_table"] = intervals
    pending_state["portable_chart_title_v97"] = str(manifest.get("chart_title") or "")

    custom_ranges = manifest.get("custom_y_ranges", {}) or {}
    if custom_ranges:
        pending_state["detail_use_custom_y_scale_v97"] = True
        pending_state["history_use_custom_y_scale"] = True
        for feature, limits in custom_ranges.items():
            if isinstance(limits, (list, tuple)) and len(limits) == 2:
                try:
                    pending_state[f"ymin_{feature_key_text(feature)}"] = float(limits[0])
                    pending_state[f"ymax_{feature_key_text(feature)}"] = float(limits[1])
                except Exception:
                    pass

    queue_session_state_restore_v99(pending_state)

    frame = portable.get("data")
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        st.session_state["continued_test_data_v93"] = frame.copy(deep=False)
        st.session_state["portable_pdf_data_v97"] = frame.copy(deep=False)
        st.session_state["portable_pdf_signature_v97"] = signature

    st.session_state["_portable_state_applied_v97"] = signature
    restored_theme = pending_state.get("ui_theme", st.session_state.get("ui_theme", "Light"))
    st.session_state["_portable_state_notice_v97"] = (
        f"Restored {len(frame):,} readings, {len(events)} point event(s), "
        f"{len(intervals)} interval event(s), and the saved {restored_theme} theme."
        if isinstance(frame, pd.DataFrame) else
        f"Restored {len(events)} point event(s), {len(intervals)} interval event(s), and the saved theme."
    )
    return True


def apply_legacy_dashboard_state_to_session(state: dict, signature: str) -> bool:
    """Stage recovered pre-v97 controls and request one clean pre-widget rerun."""
    if not state or st.session_state.get("_legacy_dashboard_state_applied_v98") == signature:
        return False

    pending_state = {}
    theme = str(state.get("theme") or "")
    if theme in {"Light", "Dark"}:
        pending_state["ui_theme"] = theme

    features = [str(x) for x in (state.get("selected_features") or []) if str(x)]
    if features:
        pending_state["selected_features_v58"] = features
        pending_state["plot_signal_order_v92_state"] = features

    well = str(state.get("well") or "").strip()
    if well:
        pending_state["selected_wells_v97"] = [well]

    pending_state["analysis_view_v97"] = "Test detail"
    x_mode = str(state.get("x_axis_mode") or "")
    if x_mode in {"Real calendar time", "Compressed real dates - remove empty gaps"}:
        pending_state["x_axis_mode_v97"] = x_mode
    pending_state["event_label_layout"] = "Auto staggered"

    events = []
    for item in state.get("manual_events", []) or []:
        if not isinstance(item, dict):
            continue
        restored = dict(item)
        if restored.get("datetime"):
            restored["datetime"] = pd.Timestamp(restored["datetime"])
        events.append(restored)
    intervals = []
    for item in state.get("operation_intervals", []) or []:
        if not isinstance(item, dict):
            continue
        restored = dict(item)
        if restored.get("start"):
            restored["start"] = pd.Timestamp(restored["start"])
        if restored.get("end"):
            restored["end"] = pd.Timestamp(restored["end"])
        intervals.append(restored)
    pending_state["manual_events_table"] = events
    pending_state["operation_intervals_table"] = intervals

    title = str(state.get("chart_title") or "").strip()
    if title:
        pending_state["portable_chart_title_v97"] = title
        st.session_state["portable_pdf_signature_v97"] = signature

    queue_session_state_restore_v99(pending_state)
    st.session_state["_legacy_dashboard_state_applied_v98"] = signature
    st.session_state["_portable_state_notice_v97"] = (
        f"Recovered the older dashboard PDF from vector chart data: {len(features)} signal(s), "
        f"{len(events)} point event(s), {len(intervals)} interval event(s), and the {theme or 'saved'} theme."
    )
    return True


# Consume recovered widget values before theme calculation and before any widget
# is instantiated. This is the only safe point to update widget-backed keys.
apply_pending_session_state_restore_v99()


UI_THEME_PRESETS = {
    "Light": {
        "color_scheme": "light",
        "app_bg": "#E7EEF3",
        "app_bg_2": "#DCE7EE",
        "sidebar_bg": "#F5F8FA",
        "panel_bg": "#FFFFFF",
        "panel_bg_2": "#EAF1F5",
        "input_bg": "#FFFFFF",
        "border": "#A9BFCC",
        "border_strong": "#7898A9",
        "accent": "#075F7A",
        "accent_hover": "#087C9E",
        "accent_soft": "#DCEEF3",
        "gold": "#A97B32",
        "gold_soft": "#C69A52",
        "text": "#061B26",
        "text_strong": "#03141D",
        "text_muted": "#3D5967",
        "success": "#287A57",
        "warning": "#A86100",
        "danger": "#B73E38",
        "grid": "rgba(15, 98, 123, 0.035)",
        "glow": "rgba(47, 141, 168, 0.14)",
        "shadow": "rgba(22, 46, 60, 0.10)",
        "control_bg": "#DCE8EE",
        "control_hover": "#C9DDE7",
        "control_icon": "#214454",
        "disabled_bg": "#E8EEF2",
        "disabled_text": "#6A7E89",
        "scroll_track": "#E7EEF2",
        "scroll_thumb": "#6D8D9C",
        "scroll_thumb_hover": "#0F627B",
        "chart_paper": "#EEF3F7",
        "chart_plot": "#FFFFFF",
        "chart_text": "#041923",
        "chart_grid": "#AFC1CD",
        "chart_grid_soft": "#D2DEE6",
        "chart_legend": "rgba(255,255,255,0.96)",
    },
    "Dark": {
        "color_scheme": "dark",
        "app_bg": "#08141D",
        "app_bg_2": "#0C1C27",
        "sidebar_bg": "#091720",
        "panel_bg": "#10232F",
        "panel_bg_2": "#142A37",
        "input_bg": "#0D202B",
        "border": "#365361",
        "border_strong": "#587483",
        "accent": "#55B8D0",
        "accent_hover": "#78CCE0",
        "accent_soft": "#173B48",
        "gold": "#D1AA63",
        "gold_soft": "#E2C27E",
        "text": "#EAF3F7",
        "text_strong": "#FFFFFF",
        "text_muted": "#B7C8D0",
        "success": "#55C58A",
        "warning": "#F2B75D",
        "danger": "#F2766E",
        "grid": "rgba(139, 194, 214, 0.035)",
        "glow": "rgba(61, 166, 198, 0.13)",
        "shadow": "rgba(0, 0, 0, 0.34)",
        "control_bg": "#17313E",
        "control_hover": "#214655",
        "control_icon": "#E9F6FA",
        "disabled_bg": "#152A35",
        "disabled_text": "#A4B7C0",
        "scroll_track": "#061119",
        "scroll_thumb": "#5F8798",
        "scroll_thumb_hover": "#78CCE0",
        "chart_paper": "#0C1B25",
        "chart_plot": "#10232F",
        "chart_text": "#EAF2F5",
        "chart_grid": "#3A5360",
        "chart_grid_soft": "#263E4B",
        "chart_legend": "rgba(11,24,34,0.96)",
    },
}

if st.session_state.get("ui_theme") not in UI_THEME_PRESETS:
    st.session_state["ui_theme"] = "Light"
# Clear obsolete privacy state left by older deployments. Privacy is handled by
# generic built-in placeholders only; user-uploaded identifiers remain visible.
st.session_state.pop("share_safe_mode", None)
st.session_state.pop("share_safe_replacements", None)
ACTIVE_THEME_NAME = st.session_state.get("ui_theme", "Light")
ACTIVE_THEME = UI_THEME_PRESETS.get(ACTIVE_THEME_NAME, UI_THEME_PRESETS["Light"])
CHART_PAPER_BG = ACTIVE_THEME["chart_paper"]
CHART_PLOT_BG = ACTIVE_THEME["chart_plot"]
CHART_TEXT = ACTIVE_THEME["chart_text"]
CHART_GRID = ACTIVE_THEME["chart_grid"]
CHART_GRID_SOFT = ACTIVE_THEME["chart_grid_soft"]
CHART_LEGEND_BG = ACTIVE_THEME["chart_legend"]


def _clear_heavy_session_state_v83(*, include_uploads: bool = False) -> None:
    """Release large cached DataFrames and export bytes without resetting user controls."""
    exact_keys = {
        "combined_data_key_v58", "combined_data_bundle_v58",
        "combined_data_key_v83", "combined_data_bundle_v83",
    }
    if include_uploads:
        exact_keys.update({
            "upload_parse_key_v58", "upload_parse_bundle_v58",
            "upload_parse_key_v83", "upload_parse_bundle_v83",
        })
    for key in list(st.session_state.keys()):
        if key in exact_keys or key.startswith("export_bytes_") or key.startswith("export_error_"):
            st.session_state.pop(key, None)


# Old deployments kept parsed tables in both Streamlit's global cache and session
# state. Clear the obsolete session keys once so a long-lived browser session does
# not carry multiple full workbook copies after upgrading.
if not st.session_state.get("_v83_state_migrated", False):
    _clear_heavy_session_state_v83(include_uploads=True)
    st.session_state["_v83_state_migrated"] = True

# Keep export memory limited to the active theme. Prepared files can be large, and
# retaining Light and Dark copies at the same time can exhaust small cloud workers.
_active_theme_slug_v83 = re.sub(r"[^a-z0-9]+", "_", ACTIVE_THEME_NAME.lower()).strip("_")
for _state_key in list(st.session_state.keys()):
    if _state_key.startswith("export_bytes_") and not _state_key.endswith("_" + _active_theme_slug_v83):
        st.session_state.pop(_state_key, None)
    elif _state_key.startswith("export_error_") and not _state_key.endswith("_" + _active_theme_slug_v83):
        st.session_state.pop(_state_key, None)

# Export annotations and report notes must follow the active Light/Dark theme.
EXPORT_LABEL_BG = CHART_PAPER_BG
EXPORT_LABEL_EDGE = CHART_GRID

_PLOT_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?")

def _plot_scalar_to_float(value) -> float:
    """Convert one plotting value to a real float without leaving bool/object dtypes.

    pandas intentionally keeps Boolean series as dtype bool when pd.to_numeric()
    is used. Subtracting bool min/max raises TypeError. Field sheets can also
    contain Decimal objects, numeric strings with units, commas, or duplicated
    mapped columns. This converter always returns float64-compatible values.
    """
    if value is None:
        return np.nan
    if isinstance(value, (pd.Timestamp, datetime, date, time)):
        return np.nan
    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass
    if isinstance(value, (bool, np.bool_)):
        return float(bool(value))
    try:
        number = float(value)
        return number if np.isfinite(number) else np.nan
    except (TypeError, ValueError, OverflowError):
        pass
    text = str(value).strip().replace("−", "-").replace("–", "-")
    match = _PLOT_NUMBER_RE.search(text)
    if not match:
        return np.nan
    try:
        number = float(match.group(0).replace(",", ""))
    except (TypeError, ValueError, OverflowError):
        return np.nan
    return number if np.isfinite(number) else np.nan

def numeric_feature_series(frame: pd.DataFrame, feature: str, *, reset_index: bool = False) -> pd.Series:
    """Return one safe float64 series, including when a mapped header is duplicated."""
    if frame is None or feature not in frame.columns:
        result = pd.Series(np.nan, index=getattr(frame, "index", None), dtype="float64", name=feature)
        return result.reset_index(drop=True) if reset_index else result

    positions = [i for i, col in enumerate(frame.columns) if col == feature]
    if not positions:
        result = pd.Series(np.nan, index=frame.index, dtype="float64", name=feature)
    elif len(positions) == 1:
        result = frame.iloc[:, positions[0]].map(_plot_scalar_to_float).astype("float64")
        result.name = feature
    else:
        # Mapping can create the same canonical name more than once. Combine the
        # duplicate candidates left-to-right instead of returning a DataFrame.
        candidates = pd.concat(
            [frame.iloc[:, pos].map(_plot_scalar_to_float).astype("float64") for pos in positions],
            axis=1,
        )
        result = candidates.bfill(axis=1).iloc[:, 0].astype("float64")
        result.name = feature

    result = result.replace([np.inf, -np.inf], np.nan)
    return result.reset_index(drop=True) if reset_index else result



def _nice_axis_ceiling(value: float) -> float:
    """Round a positive number up to a readable engineering axis limit."""
    try:
        value = float(value)
    except Exception:
        return 1.0
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    base = 10.0 ** exponent
    fraction = value / base
    for step in (1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
        if fraction <= step + 1e-12:
            return float(step * base)
    return float(10.0 * base)


def _nice_axis_floor(value: float) -> float:
    """Round a negative number down to a readable engineering axis limit."""
    try:
        value = float(value)
    except Exception:
        return -1.0
    if not np.isfinite(value) or value >= 0:
        return 0.0
    return -_nice_axis_ceiling(abs(value))


def _is_full_percent_axis(feature: str) -> bool:
    label = column_label(feature).lower()
    return (
        feature in {"bsw_pct", "wlr_s_pct", "water_cut_pct", "wc_pct", "choke_pct"}
        or "water cut" in label
        or "bs&w" in label
        or ("choke" in label and "%" in label)
    )


def _is_full_choke_size_axis(feature: str) -> bool:
    label = column_label(feature).lower()
    return (
        feature == "choke_size_64"
        or ("choke" in label and ("/64" in label or "128" in label))
    )


def default_y_axis_range(frame: pd.DataFrame, feature: str) -> list[float] | None:
    """Return a clear default Y range for one engineering signal.

    Non-negative measurements start at zero and end at a rounded value above
    the detected maximum.  Signed measurements keep their negative portion so
    physically meaningful values such as reverse reel speed are not hidden.
    """
    if frame is None or feature not in getattr(frame, "columns", []):
        return None
    if _is_full_percent_axis(feature):
        return [0.0, 100.0]
    if _is_full_choke_size_axis(feature):
        return [0.0, 128.0]

    vals = numeric_feature_series(frame, feature).dropna()
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return None
    ymin = float(vals.min())
    ymax = float(vals.max())

    if ymax <= 0:
        return [_nice_axis_floor(ymin * 1.08), 0.0]
    upper = _nice_axis_ceiling(ymax * 1.08)
    if ymin < 0:
        lower = _nice_axis_floor(ymin * 1.08)
    else:
        lower = 0.0
    if upper <= lower:
        upper = lower + max(abs(lower) * 0.1, 1.0)
    return [float(lower), float(upper)]


def combined_default_y_axis_range(frame: pd.DataFrame, features: list[str]) -> list[float] | None:
    """Return one readable range for a combined axis containing several signals."""
    features = [f for f in features if f in getattr(frame, "columns", [])]
    if not features:
        return None
    if len(features) == 1:
        return default_y_axis_range(frame, features[0])
    pieces = [numeric_feature_series(frame, f).dropna() for f in features]
    pieces = [s for s in pieces if not s.empty]
    if not pieces:
        return None
    vals = pd.concat(pieces, ignore_index=True)
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        return None
    ymin = float(vals.min())
    ymax = float(vals.max())
    lower = _nice_axis_floor(ymin * 1.08) if ymin < 0 else 0.0
    upper = _nice_axis_ceiling(max(ymax, 0.0) * 1.08) if ymax > 0 else 0.0
    if upper <= lower:
        upper = lower + 1.0
    return [float(lower), float(upper)]



def optimize_interactive_plot_frame(
    frame: pd.DataFrame,
    features: list[str],
    *,
    max_total_points: int = 8000,
    max_points_per_group: int = 2000,
) -> tuple[pd.DataFrame, bool]:
    """Reduce only the browser chart payload while preserving full source data.

    Large uploaded files can overwhelm the browser even when parsing succeeds.
    The interactive view keeps evenly spaced points plus each selected signal's
    extrema. Exports and data tables continue to use the complete filtered data.
    """
    if frame is None or frame.empty or len(frame) <= max_total_points:
        return frame, False

    group_cols = [c for c in ["series_label", "test_id"] if c in frame.columns]
    groups = list(frame.groupby(group_cols, dropna=False, sort=False)) if group_cols else [(None, frame)]
    if not groups:
        return frame, False

    group_budget = max(250, min(max_points_per_group, max_total_points // max(len(groups), 1)))
    pieces = []
    reduced = False
    for _, group in groups:
        g = group.sort_values("datetime", kind="stable") if "datetime" in group.columns else group
        n = len(g)
        if n <= group_budget:
            pieces.append(g)
            continue
        reduced = True
        chosen = set(np.linspace(0, n - 1, group_budget).round().astype(int).tolist())
        for feature in features:
            if feature not in g.columns:
                continue
            vals = numeric_feature_series(g, feature, reset_index=True)
            valid = vals.dropna()
            if valid.empty:
                continue
            chosen.add(int(valid.idxmin()))
            chosen.add(int(valid.idxmax()))
        pieces.append(g.iloc[sorted(i for i in chosen if 0 <= i < n)])

    if not pieces:
        return frame, False
    out = pd.concat(pieces, ignore_index=False, sort=False, copy=False)
    sort_cols = [c for c in ["series_label", "test_id", "datetime"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="stable")
    return out, reduced


def limited_dataframe_preview(
    frame: pd.DataFrame,
    *,
    max_rows: int = 500,
    max_cols: int = 48,
) -> tuple[pd.DataFrame, int, int]:
    """Return a browser-safe preview and the omitted row/column counts."""
    if frame is None or frame.empty:
        return frame, 0, 0
    shown = frame.iloc[:max_rows, :max_cols].copy(deep=False)
    return shown, max(0, len(frame) - len(shown)), max(0, frame.shape[1] - shown.shape[1])


MIN_DATE_ALLOWED = pd.Timestamp("1900-01-01").date()
MAX_DATE_ALLOWED = pd.Timestamp("2100-12-31").date()

# User-taught column aliases are stored beside app.py. This makes the app learn
# new field abbreviations without editing Python code every time.
USER_ALIAS_FILE = Path(__file__).with_name("user_column_aliases.json")


def load_saved_aliases() -> dict:
    try:
        if USER_ALIAS_FILE.exists():
            data = json.loads(USER_ALIAS_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def save_aliases(alias_map: dict) -> None:
    safe = {str(k): str(v) for k, v in alias_map.items() if v and v != "__keep__"}
    USER_ALIAS_FILE.write_text(json.dumps(safe, indent=2, sort_keys=True), encoding="utf-8")


def alias_display_name(key: str, labels: dict) -> str:
    if key == "__keep__":
        return "Keep as detected"
    if key == "__drop__":
        return "Ignore / hide this column"
    return labels.get(key, key)


def editable_column_mapping_panel(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Interactive review step that lets users teach new column names.

    The parser still auto-detects columns first.  This panel is the safety net for
    any new field abbreviation, such as Pi, Pd, AMp, Freq, Ti, Tm, Vx, etc.
    """
    saved_aliases = load_saved_aliases()
    labels = standard_column_options(include_meta=False)
    options = ["__keep__", "__drop__"] + list(labels.keys())

    # Apply already-saved aliases immediately.
    df_after_saved = apply_user_column_mappings(df, saved_aliases)

    numeric = available_numeric_columns(df_after_saved)
    raw_cols = [c for c in numeric if str(c).startswith("raw__")]

    with st.expander("Column mapping review / teach new names", expanded=bool(raw_cols)):
        st.caption(
            "Use this when a new customer template has abbreviations the parser has not seen before. "
            "Map Raw columns to standard fields, then save the aliases so future uploads are detected automatically."
        )

        if saved_aliases:
            st.success(f"Loaded {len(saved_aliases)} saved column alias(es).")

        show_all = st.checkbox(
            "Show all numeric columns, not only Raw unknown columns",
            value=False,
            help="Normally you only need to map Raw columns. Use this to override any auto-detected column.",
        )

        cols_to_review = numeric if show_all else raw_cols
        runtime_aliases = {}

        if not cols_to_review:
            st.info("No unknown Raw numeric columns were found in this upload. The parser recognized the detected plot columns.")
        else:
            st.write("Map detected upload columns to standard app fields:")
            for col in cols_to_review:
                current_saved = saved_aliases.get(str(col)) or saved_aliases.get(canonical_key(col)) or "__keep__"
                if current_saved not in options:
                    current_saved = "__keep__"
                selected = st.selectbox(
                    f"{column_label(col)}  →",
                    options,
                    index=options.index(current_saved),
                    format_func=lambda x, labels=labels: alias_display_name(x, labels),
                    key=f"map_col_{canonical_key(col)}",
                )
                if selected != "__keep__":
                    # Store both the current parsed column name and its normalized alias key.
                    runtime_aliases[str(col)] = selected
                    runtime_aliases[canonical_key(col)] = selected

        applied_aliases = {**saved_aliases, **runtime_aliases}

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Save mappings for future uploads", disabled=not applied_aliases):
                save_aliases(applied_aliases)
                st.success("Saved. These aliases will be applied automatically on the next upload.")
        with c2:
            if st.button("Clear saved mappings", disabled=not USER_ALIAS_FILE.exists()):
                try:
                    USER_ALIAS_FILE.unlink(missing_ok=True)
                except Exception:
                    pass
                st.success("Saved aliases cleared. Refresh or upload again to reload clean detection.")
        with c3:
            st.caption(f"Alias file: {USER_ALIAS_FILE.name}")

        if applied_aliases:
            preview_rows = []
            for k, v in sorted(applied_aliases.items()):
                if k.startswith("raw__") or k in [canonical_key(c) for c in cols_to_review]:
                    preview_rows.append({"Alias/header key": k, "Mapped to": alias_display_name(v, labels)})
            if preview_rows:
                st.dataframe(pd.DataFrame(preview_rows), width="stretch", height=160)

    df_final = apply_user_column_mappings(df_after_saved, runtime_aliases)
    return df_final, {**saved_aliases, **runtime_aliases}


def parse_time_value(typed_text, fallback_time):
    """Accept either a typed time such as 09:30 / 9:30 PM / 0930, or a time_input value."""
    text = str(typed_text or "").strip()
    if text:
        # Handles 09:30, 09.30, 0930, 9pm, 9:30 pm.
        cleaned = text.lower().replace(".", ":")
        try:
            parsed = pd.to_datetime(cleaned, errors="raise")
            return parsed.time().replace(second=0, microsecond=0)
        except Exception:
            pass

        m = re.match(r"^\s*(\d{1,2})(\d{2})\s*(am|pm)?\s*$", cleaned)
        if m:
            hh = int(m.group(1))
            mm = int(m.group(2))
            ampm = m.group(3)
            if ampm == "pm" and hh < 12:
                hh += 12
            if ampm == "am" and hh == 12:
                hh = 0
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return pd.Timestamp(year=1900, month=1, day=1, hour=hh, minute=mm).time()

        m = re.match(r"^\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*$", cleaned)
        if m:
            hh = int(m.group(1))
            mm = int(m.group(2) or 0)
            ampm = m.group(3)
            if ampm == "pm" and hh < 12:
                hh += 12
            if ampm == "am" and hh == 12:
                hh = 0
            if 0 <= hh <= 23 and 0 <= mm <= 59:
                return pd.Timestamp(year=1900, month=1, day=1, hour=hh, minute=mm).time()

        st.warning(f"Could not read typed time: {typed_text}. Using the picker time instead.")

    return fallback_time.replace(second=0, microsecond=0) if fallback_time else None


def combine_date_and_time(date_value, picker_time, typed_time_text=""):
    parsed_time = parse_time_value(typed_time_text, picker_time)
    if date_value is None or parsed_time is None:
        return None
    return pd.Timestamp.combine(date_value, parsed_time)


def _time_picker_options(default_time=None, step_minutes: int = 15):
    """Return a full-day list of time values for a reliably scrollable picker.

    Streamlit's native ``time_input`` may be rendered by the browser as a popup
    without a visible scrollbar.  A normal BaseWeb selectbox gives us a real
    scrollable list that can be styled consistently in Light and Dark themes.
    """
    step_minutes = max(1, int(step_minutes or 15))
    minute_values = set(range(0, 24 * 60, step_minutes))

    if default_time is not None:
        try:
            minute_values.add(int(default_time.hour) * 60 + int(default_time.minute))
        except Exception:
            pass

    return [
        pd.Timestamp(year=2000, month=1, day=1, hour=minute // 60, minute=minute % 60).time()
        for minute in sorted(minute_values)
    ]


def scrollable_time_picker(label, default_time=None, *, key: str, step_minutes: int = 15):
    """A 24-hour time picker with a visible vertical scrollbar."""
    options = _time_picker_options(default_time, step_minutes=step_minutes)
    normalized_default = None
    if default_time is not None:
        try:
            normalized_default = default_time.replace(second=0, microsecond=0)
        except Exception:
            normalized_default = default_time

    try:
        default_index = options.index(normalized_default) if normalized_default is not None else 0
    except ValueError:
        default_index = 0

    return st.selectbox(
        label,
        options,
        index=default_index,
        format_func=lambda value: value.strftime("%H:%M"),
        key=key,
        help=(
            "Scroll through the full 24-hour list in 15-minute steps. "
            "For an exact minute not shown, use the optional typed-time field below."
        ),
    )


def feature_key_text(feature_name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(feature_name))


def _compressed_scale_delta(scale):
    scale = str(scale or "")
    if scale.startswith("Custom:"):
        parsed = _parse_interval_text(scale.split(":", 1)[1].strip())
        return parsed.get("timedelta") if parsed else None
    return {
        "30 minutes": pd.Timedelta(minutes=30),
        "1 hour": pd.Timedelta(hours=1),
        "3 hours": pd.Timedelta(hours=3),
        "6 hours": pd.Timedelta(hours=6),
        "12 hours": pd.Timedelta(hours=12),
        "1 day": pd.Timedelta(days=1),
        "1 month": pd.Timedelta(days=30),
        "1 year": pd.Timedelta(days=365),
    }.get(str(scale or ""))


def compressed_axis_tick_kwargs(df, scale="Auto readable", max_ticks_per_series=4, max_total_ticks=22):
    """Build readable ticks for a compressed real-date timeline.

    The selected tick scale changes which real dates are considered. Labels are
    then pruned by their visual compressed positions to prevent overlap when
    wells have widely separated calendar periods.
    """
    if df.empty or "plot_x" not in df.columns or "datetime" not in df.columns:
        return {}

    pts = df[["plot_x", "datetime"]].dropna().copy()
    if pts.empty:
        return {}
    pts["plot_x"] = pd.to_numeric(pts["plot_x"], errors="coerce")
    pts["datetime"] = pd.to_datetime(pts["datetime"], errors="coerce")
    pts = pts.dropna().sort_values(["datetime", "plot_x"], kind="stable")
    pts = pts.drop_duplicates(subset=["plot_x"], keep="first").reset_index(drop=True)
    if pts.empty:
        return {}

    max_total_ticks = max(3, int(max_total_ticks or 10))
    delta = _compressed_scale_delta(scale)
    n = len(pts)

    if delta is None:
        if n <= max_total_ticks:
            candidate_idxs = list(range(n))
        else:
            candidate_idxs = sorted(set(np.linspace(0, n - 1, max_total_ticks).round().astype(int).tolist()))
    else:
        candidate_idxs = [0]
        last_dt = pd.Timestamp(pts.loc[0, "datetime"])
        for i in range(1, max(n - 1, 1)):
            dt = pd.Timestamp(pts.loc[i, "datetime"])
            if dt - last_dt >= delta:
                candidate_idxs.append(i)
                last_dt = dt
        if n > 1:
            candidate_idxs.append(n - 1)
        candidate_idxs = sorted(set(candidate_idxs))
        if len(candidate_idxs) > max_total_ticks:
            pick = np.linspace(0, len(candidate_idxs) - 1, max_total_ticks).round().astype(int)
            candidate_idxs = sorted(set(candidate_idxs[int(i)] for i in pick))

    x_min = float(pts["plot_x"].min())
    x_max = float(pts["plot_x"].max())
    x_span = max(x_max - x_min, 0.0)
    min_visual_gap = x_span / max(max_total_ticks - 1, 1) * 0.62 if x_span > 0 else 0.0
    pruned = []
    for idx in candidate_idxs:
        x = float(pts.loc[idx, "plot_x"])
        if not pruned or x - float(pts.loc[pruned[-1], "plot_x"]) >= min_visual_gap:
            pruned.append(idx)
    if n > 1 and (not pruned or pruned[-1] != n - 1):
        if len(pruned) > 1 and x_max - float(pts.loc[pruned[-1], "plot_x"]) < min_visual_gap * 0.65:
            pruned[-1] = n - 1
        else:
            pruned.append(n - 1)
    candidate_idxs = sorted(set(pruned or [0]))

    span = pd.Timestamp(pts["datetime"].max()) - pd.Timestamp(pts["datetime"].min())
    selected_dts = [pd.Timestamp(pts.loc[i, "datetime"]) for i in candidate_idxs]
    duplicate_days = len({dt.date() for dt in selected_dts}) < len(selected_dts)
    if delta is not None and delta < pd.Timedelta(days=1):
        fmt = "%d-%b-%Y<br>%H:%M"
    elif str(scale) == "1 day":
        fmt = "%d-%b-%Y"
    elif str(scale) in {"1 month", "1 year"}:
        fmt = "%b-%Y"
    elif duplicate_days:
        fmt = "%d-%b-%Y<br>%H:%M"
    elif span <= pd.Timedelta(days=730):
        fmt = "%d-%b-%Y"
    else:
        fmt = "%b-%Y"

    tickvals = [float(pts.loc[i, "plot_x"]) for i in candidate_idxs]
    ticktext = [dt.strftime(fmt.replace("<br>", "\n")).replace("\n", "<br>") for dt in selected_dts]
    angle = -35 if len(tickvals) >= 6 or "<br>" in fmt else 0
    return {"tickmode": "array", "tickvals": tickvals, "ticktext": ticktext, "tickangle": angle}


def elapsed_axis_tick_kwargs(df, max_ticks=10):
    """Readable ticks for aligned elapsed-hour comparison charts."""
    if df.empty or "plot_x" not in df.columns:
        return {}
    vals = pd.to_numeric(df["plot_x"], errors="coerce").dropna()
    if vals.empty:
        return {}
    xmax = float(vals.max())
    if xmax <= 0:
        return {"tickmode": "array", "tickvals": [0], "ticktext": ["0 h"]}
    if xmax <= 8:
        step = 1
    elif xmax <= 24:
        step = 2
    elif xmax <= 72:
        step = 6
    elif xmax <= 240:
        step = 24
    else:
        step = max(24, round(xmax / max_ticks))
    tickvals = list(np.arange(0, xmax + step, step))[: max_ticks + 2]
    ticktext = [f"{int(v)} h" if abs(v - int(v)) < 1e-9 else f"{v:.1f} h" for v in tickvals]
    return {"tickmode": "array", "tickvals": tickvals, "ticktext": ticktext, "tickangle": 0}

def compressed_separator_positions(df):
    """Positions where long empty gaps were compressed."""
    seps = []
    try:
        for item in df.attrs.get("compressed_separators", []) or []:
            seps.append(float(item.get("x")))
    except Exception:
        pass
    return seps


def detected_test_separator_positions(df):
    """Return visual boundaries between separate tests/plot segments."""
    if df is None or df.empty or "plot_x" not in df.columns:
        return []
    group_col = "series_group_key" if "series_group_key" in df.columns else ("well" if "well" in df.columns else None)
    groups = df.groupby(group_col, dropna=False) if group_col else [("All", df)]
    found = []
    for _, g in groups:
        sort_col = "datetime" if "datetime" in g.columns else "plot_x"
        g = g.sort_values(sort_col).copy()
        if len(g) < 2:
            continue
        prev = g.iloc[0]
        test_no = 1
        for i in range(1, len(g)):
            cur = g.iloc[i]
            changed_test = ("test_id" in g.columns and str(cur.get("test_id", "")) != str(prev.get("test_id", "")))
            changed_segment = ("series_segment_id" in g.columns and cur.get("series_segment_id") != prev.get("series_segment_id"))
            source_cols = [c for c in ["source", "sheet"] if c in g.columns]
            changed_source = bool(source_cols) and any(str(cur.get(c, "")) != str(prev.get(c, "")) for c in source_cols)
            if changed_test or changed_segment:
                x0, x1 = prev.get("plot_x"), cur.get("plot_x")
                try:
                    x_sep = x0 + (x1 - x0) / 2
                except Exception:
                    try:
                        x_sep = pd.Timestamp(x0) + (pd.Timestamp(x1) - pd.Timestamp(x0)) / 2
                    except Exception:
                        x_sep = x1
                # Keep the boundary visual only.  Do not add Test 1/Test 2
                # labels, because report numbering can be misleading when an
                # incomplete and final workbook belong to one physical test.
                found.append({"x": x_sep})
            prev = cur
    dedup, seen = [], set()
    for item in found:
        key = str(item.get("x"))
        if key not in seen:
            seen.add(key)
            dedup.append(item)
    return dedup

def chart_separator_positions(df):
    # Dashed separators mark actual detected test changes only. Long intervals
    # between readings inside one test remain a continuous curve.
    return list(detected_test_separator_positions(df))



st.markdown(
    f"""
    <style>
    :root {{
        color-scheme: {ACTIVE_THEME['color_scheme']};
        --petro-bg: {ACTIVE_THEME['app_bg']};
        --petro-bg-2: {ACTIVE_THEME['app_bg_2']};
        --petro-sidebar: {ACTIVE_THEME['sidebar_bg']};
        --petro-panel: {ACTIVE_THEME['panel_bg']};
        --petro-panel-2: {ACTIVE_THEME['panel_bg_2']};
        --petro-input: {ACTIVE_THEME['input_bg']};
        --petro-border: {ACTIVE_THEME['border']};
        --petro-border-strong: {ACTIVE_THEME['border_strong']};
        --petro-accent: {ACTIVE_THEME['accent']};
        --petro-accent-hover: {ACTIVE_THEME['accent_hover']};
        --petro-accent-soft: {ACTIVE_THEME['accent_soft']};
        --petro-gold: {ACTIVE_THEME['gold']};
        --petro-gold-soft: {ACTIVE_THEME['gold_soft']};
        --petro-text: {ACTIVE_THEME['text']};
        --petro-text-strong: {ACTIVE_THEME['text_strong']};
        --petro-muted: {ACTIVE_THEME['text_muted']};
        --petro-success: {ACTIVE_THEME['success']};
        --petro-warning: {ACTIVE_THEME['warning']};
        --petro-danger: {ACTIVE_THEME['danger']};
        --petro-grid: {ACTIVE_THEME['grid']};
        --petro-glow: {ACTIVE_THEME['glow']};
        --petro-shadow: {ACTIVE_THEME['shadow']};
        --petro-control-bg: {ACTIVE_THEME['control_bg']};
        --petro-control-hover: {ACTIVE_THEME['control_hover']};
        --petro-control-icon: {ACTIVE_THEME['control_icon']};
        --petro-disabled-bg: {ACTIVE_THEME['disabled_bg']};
        --petro-disabled-text: {ACTIVE_THEME['disabled_text']};
        --petro-scroll-track: {ACTIVE_THEME['scroll_track']};
        --petro-scroll-thumb: {ACTIVE_THEME['scroll_thumb']};
        --petro-scroll-thumb-hover: {ACTIVE_THEME['scroll_thumb_hover']};
        --petro-chart-text: {ACTIVE_THEME['chart_text']};
        --petro-chart-paper: {ACTIVE_THEME['chart_paper']};
    }}

    html, body, [class*="css"] {{
        font-family: Inter, Aptos, "Segoe UI", Roboto, Arial, sans-serif;
    }}


    /* High-contrast scrollbars and scroll arrows for both the main page and sidebar. */
    html, body, .stApp, .stApp *, section[data-testid="stSidebar"] * {{
        scrollbar-color: var(--petro-scroll-thumb) var(--petro-scroll-track);
        scrollbar-width: auto;
    }}
    html {{ overflow-y: scroll; scrollbar-gutter: stable; }}
    [data-testid="stAppViewContainer"],
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"] {{ scrollbar-gutter: stable; }}

    /* Sidebar scrolling restored to the proven v92 behavior.
       Let Streamlit keep its native sidebar height and make its own content
       container the scrolling surface.  Avoid fixed 100vh/nested overflow
       rules because they can create a visible thumb that does not move the
       controls in some browsers. */
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {{
        overflow-x: hidden !important;
        overflow-y: scroll !important;
        scrollbar-gutter: stable !important;
        overscroll-behavior-y: auto;
    }}
    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {{
        padding-bottom: 3rem !important;
    }}
    *::-webkit-scrollbar {{ width: 13px; height: 13px; }}
    *::-webkit-scrollbar-track {{
        background: var(--petro-scroll-track);
        border-radius: 999px;
    }}
    *::-webkit-scrollbar-thumb {{
        min-height: 42px;
        background: var(--petro-scroll-thumb);
        border: 3px solid var(--petro-scroll-track);
        border-radius: 999px;
    }}
    *::-webkit-scrollbar-thumb:hover,
    *::-webkit-scrollbar-thumb:active {{ background: var(--petro-scroll-thumb-hover); }}
    *::-webkit-scrollbar-corner {{ background: var(--petro-scroll-track); }}
    *::-webkit-scrollbar-button:single-button {{
        display: block;
        width: 13px;
        height: 13px;
        background-color: var(--petro-control-bg);
        border: 1px solid var(--petro-border);
        background-repeat: no-repeat;
    }}
    *::-webkit-scrollbar-button:single-button:hover {{ background-color: var(--petro-control-hover); }}
    *::-webkit-scrollbar-button:single-button:vertical:decrement {{
        background-image:
            linear-gradient(135deg, transparent 50%, var(--petro-control-icon) 50%),
            linear-gradient(225deg, transparent 50%, var(--petro-control-icon) 50%);
        background-size: 5px 5px, 5px 5px;
        background-position: 2px 5px, 6px 5px;
    }}
    *::-webkit-scrollbar-button:single-button:vertical:increment {{
        background-image:
            linear-gradient(45deg, transparent 50%, var(--petro-control-icon) 50%),
            linear-gradient(315deg, transparent 50%, var(--petro-control-icon) 50%);
        background-size: 5px 5px, 5px 5px;
        background-position: 2px 3px, 6px 3px;
    }}
    *::-webkit-scrollbar-button:single-button:horizontal:decrement {{
        background-image:
            linear-gradient(45deg, transparent 50%, var(--petro-control-icon) 50%),
            linear-gradient(135deg, transparent 50%, var(--petro-control-icon) 50%);
        background-size: 5px 5px, 5px 5px;
        background-position: 5px 2px, 5px 6px;
    }}
    *::-webkit-scrollbar-button:single-button:horizontal:increment {{
        background-image:
            linear-gradient(225deg, transparent 50%, var(--petro-control-icon) 50%),
            linear-gradient(315deg, transparent 50%, var(--petro-control-icon) 50%);
        background-size: 5px 5px, 5px 5px;
        background-position: 3px 2px, 3px 6px;
    }}

    .stApp {{
        color: var(--petro-text) !important;
        background-color: var(--petro-bg) !important;
        background-image:
            radial-gradient(circle at 92% -4%, var(--petro-glow), transparent 31rem),
            linear-gradient(var(--petro-grid) 1px, transparent 1px),
            linear-gradient(90deg, var(--petro-grid) 1px, transparent 1px),
            linear-gradient(155deg, var(--petro-bg) 0%, var(--petro-bg-2) 100%) !important;
        background-size: auto, 42px 42px, 42px 42px, auto !important;
        background-attachment: fixed !important;
    }}

    header[data-testid="stHeader"] {{
        background: color-mix(in srgb, var(--petro-bg) 92%, transparent) !important;
        border-bottom: 1px solid color-mix(in srgb, var(--petro-border) 75%, transparent) !important;
        backdrop-filter: blur(12px);
    }}
    header[data-testid="stHeader"] button,
    [data-testid="stToolbar"] button,
    [data-testid="stHeaderActionElements"] button {{
        color: var(--petro-text) !important;
    }}
    header[data-testid="stHeader"] button svg,
    [data-testid="stToolbar"] button svg,
    [data-testid="stHeaderActionElements"] button svg {{
        color: var(--petro-text) !important;
        fill: var(--petro-text) !important;
    }}


    /* Sidebar open/close arrows, tab scroll arrows and icon-only controls. */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="stTabsScrollLeft"],
    [data-testid="stTabsScrollRight"],
    [data-testid="stPaginationPrev"],
    [data-testid="stPaginationNext"] {{
        color: var(--petro-control-icon) !important;
        background: var(--petro-control-bg) !important;
        border: 1px solid var(--petro-border-strong) !important;
        border-radius: 8px !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebarCollapseButton"]:hover,
    [data-testid="stExpandSidebarButton"]:hover,
    [data-testid="stTabsScrollLeft"]:hover,
    [data-testid="stTabsScrollRight"]:hover,
    [data-testid="stPaginationPrev"]:hover,
    [data-testid="stPaginationNext"]:hover {{
        color: var(--petro-text-strong) !important;
        background: var(--petro-control-hover) !important;
    }}
    [data-testid="stSidebarCollapseButton"] svg,
    [data-testid="stExpandSidebarButton"] svg,
    [data-testid="stTabsScrollLeft"] svg,
    [data-testid="stTabsScrollRight"] svg,
    [data-testid="stPaginationPrev"] svg,
    [data-testid="stPaginationNext"] svg {{
        color: currentColor !important;
        fill: currentColor !important;
        stroke: currentColor !important;
        opacity: 1 !important;
    }}
    [data-testid="stIconMaterial"],
    [data-testid="stIconEmoji"] {{ color: inherit !important; opacity: 1 !important; }}

    .block-container {{
        padding-top: 1.1rem;
        padding-left: clamp(1rem, 2.4vw, 2.8rem);
        padding-right: clamp(1rem, 2.4vw, 2.8rem);
        padding-bottom: 3rem;
        max-width: 100%;
    }}

    section[data-testid="stSidebar"] {{
        background: color-mix(in srgb, var(--petro-sidebar) 97%, transparent) !important;
        border-right: 1px solid var(--petro-border) !important;
        box-shadow: 8px 0 28px var(--petro-shadow) !important;
    }}
    section[data-testid="stSidebar"] > div {{ padding-top: .7rem; }}
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {{ color: var(--petro-text) !important; }}
    section[data-testid="stSidebar"] .stCaption {{ color: var(--petro-muted) !important; }}

    h1, h2, h3, h4, h5, h6, p, label,
    .stMarkdown, .stCaption, .stText, [data-testid="stWidgetLabel"] {{
        color: var(--petro-text) !important;
    }}
    h1, h2, h3, h4, h5, h6 {{ letter-spacing: .005em; }}
    .stCaption, small {{ color: var(--petro-muted) !important; }}
    a {{ color: var(--petro-accent-hover) !important; }}

    .petro-hero {{
        position: relative;
        overflow: hidden;
        border: 1px solid var(--petro-border);
        border-left: 5px solid var(--petro-accent);
        border-radius: 16px;
        padding: 1.25rem 1.45rem;
        margin: 0 0 1rem;
        background: linear-gradient(135deg, var(--petro-panel) 0%, var(--petro-panel-2) 100%);
        box-shadow: 0 8px 24px color-mix(in srgb, var(--petro-shadow) 70%, transparent);
    }}
    .petro-hero::after {{ display: none; }}
    .petro-hero-row {{
        display: flex;
        gap: 1rem;
        align-items: center;
        position: relative;
        z-index: 2;
    }}
    .petro-mark {{
        width: 3.45rem;
        height: 3.45rem;
        flex: 0 0 3.45rem;
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: var(--petro-text-strong);
        background: linear-gradient(145deg, var(--petro-accent), color-mix(in srgb, var(--petro-accent) 64%, #07131b));
        border: 1px solid color-mix(in srgb, var(--petro-accent) 78%, var(--petro-border));
        box-shadow: 0 9px 22px color-mix(in srgb, var(--petro-accent) 24%, transparent);
    }}
    .petro-chart-icon {{ width: 2.15rem; height: 2.15rem; overflow: visible; }}
    .petro-chart-icon .axis {{ stroke: color-mix(in srgb, white 76%, var(--petro-gold-soft)); stroke-width: 2; fill: none; stroke-linecap: round; }}
    .petro-chart-icon .trend {{ stroke: #FFFFFF; stroke-width: 3.2; fill: none; stroke-linecap: round; stroke-linejoin: round; }}
    .petro-chart-icon .point {{ fill: var(--petro-gold-soft); stroke: #FFFFFF; stroke-width: 1.1; }}
    .petro-title {{
        margin: 0;
        font-family: Aptos, "Segoe UI", Arial, sans-serif;
        font-size: clamp(1.55rem, 2.55vw, 2.25rem);
        line-height: 1.12;
        font-weight: 760;
        letter-spacing: -.018em;
        color: var(--petro-text-strong) !important;
        text-shadow: none !important;
        -webkit-text-stroke: 0 transparent !important;
        filter: none !important;
        font-synthesis: none;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }}
    .petro-subtitle {{
        margin-top: .45rem;
        max-width: 64rem;
        color: var(--petro-muted);
        font-size: .98rem;
        line-height: 1.55;
    }}

    .petro-section-title {{
        margin: 1.05rem 0 .65rem;
        padding: .68rem .85rem;
        border: 1px solid color-mix(in srgb, var(--petro-border) 82%, transparent);
        border-left: 4px solid var(--petro-gold);
        border-radius: 10px;
        background: color-mix(in srgb, var(--petro-panel) 92%, transparent);
        color: var(--petro-text-strong);
        font-size: 1.03rem;
        font-weight: 780;
        letter-spacing: .01em;
        box-shadow: 0 5px 16px color-mix(in srgb, var(--petro-shadow) 62%, transparent);
    }}

    div[data-testid="stMetric"] {{
        min-height: 112px;
        padding: .92rem 1rem .82rem;
        border-radius: 13px;
        border: 1px solid var(--petro-border);
        border-top: 3px solid var(--petro-accent);
        background: linear-gradient(145deg, var(--petro-panel), var(--petro-panel-2));
        box-shadow: 0 8px 22px var(--petro-shadow);
    }}
    div[data-testid="stMetricValue"] {{
        font-size: clamp(1.42rem, 2.1vw, 1.95rem) !important;
        font-weight: 820 !important;
        color: var(--petro-text-strong) !important;
    }}
    div[data-testid="stMetricLabel"] {{
        font-size: .9rem !important;
        font-weight: 700 !important;
        color: var(--petro-muted) !important;
    }}

    [data-testid="stExpander"],
    [data-testid="stExpander"] details {{
        border: 1px solid var(--petro-border) !important;
        border-radius: 12px !important;
        background: var(--petro-panel) !important;
        overflow: hidden;
        box-shadow: 0 6px 18px color-mix(in srgb, var(--petro-shadow) 58%, transparent);
    }}
    [data-testid="stExpander"] details summary {{
        border-left: 3px solid var(--petro-gold);
        background: var(--petro-panel-2) !important;
        color: var(--petro-text-strong) !important;
        font-weight: 730;
    }}
    [data-testid="stExpander"] details summary:hover {{
        background: color-mix(in srgb, var(--petro-panel-2) 88%, var(--petro-accent-soft)) !important;
    }}
    [data-testid="stExpander"] details summary *,
    [data-testid="stExpander"] details summary p {{ color: var(--petro-text-strong) !important; }}
    [data-testid="stExpander"] details summary svg {{ fill: var(--petro-text-strong) !important; color: var(--petro-text-strong) !important; }}

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div,
    div[data-baseweb="base-input"] > div,
    div[data-baseweb="textarea"] > div,
    textarea, input {{
        background-color: var(--petro-input) !important;
        color: var(--petro-text-strong) !important;
        border-color: var(--petro-border) !important;
        border-radius: 8px !important;
    }}
    input::placeholder, textarea::placeholder {{ color: var(--petro-muted) !important; opacity: .82 !important; }}
    div[data-baseweb="select"] *,
    div[data-baseweb="input"] *,
    div[data-baseweb="textarea"] *,
    textarea, input {{ color: var(--petro-text-strong) !important; }}
    div[data-baseweb="select"] svg,
    div[data-baseweb="input"] svg {{ fill: var(--petro-muted) !important; }}

    div[data-baseweb="popover"],
    div[role="listbox"], ul[role="listbox"],
    div[data-baseweb="menu"] {{
        background: var(--petro-panel-2) !important;
        color: var(--petro-text) !important;
        border-color: var(--petro-border) !important;
        box-shadow: 0 12px 30px var(--petro-shadow) !important;
    }}
    li[role="option"], div[role="option"] {{ color: var(--petro-text) !important; }}
    li[role="option"]:hover, div[role="option"]:hover,
    li[aria-selected="true"], div[aria-selected="true"] {{
        background: color-mix(in srgb, var(--petro-accent) 22%, var(--petro-panel-2)) !important;
        color: var(--petro-text-strong) !important;
    }}

    /* BaseWeb mounts select/multiselect menus in a body-level portal. Some
       Streamlit builds apply an inline white surface to the inner menu, so use
       explicit theme colors on every portal layer and option descendant. */
    body > div[data-baseweb="popover"],
    body > div[data-baseweb="popover"] > div,
    [data-baseweb="popover"] [data-baseweb="menu"],
    [data-baseweb="popover"] [role="listbox"],
    [data-baseweb="popover"] ul[role="listbox"] {{
        background-color: {ACTIVE_THEME['panel_bg_2']} !important;
        color: {ACTIVE_THEME['text']} !important;
        border-color: {ACTIVE_THEME['border_strong']} !important;
    }}
    [data-baseweb="popover"] [role="option"],
    [data-baseweb="popover"] [role="option"] > div,
    [data-baseweb="popover"] [role="option"] span,
    [data-baseweb="popover"] li,
    [data-baseweb="popover"] li * {{
        color: {ACTIVE_THEME['text']} !important;
        opacity: 1 !important;
    }}
    [data-baseweb="popover"] [role="option"]:not([aria-selected="true"]) {{
        background-color: {ACTIVE_THEME['panel_bg_2']} !important;
    }}
    [data-baseweb="popover"] [role="option"]:hover,
    [data-baseweb="popover"] [role="option"][aria-selected="true"] {{
        background-color: {ACTIVE_THEME['control_hover']} !important;
        color: {ACTIVE_THEME['text_strong']} !important;
    }}
    [data-baseweb="popover"] [role="option"]:hover *,
    [data-baseweb="popover"] [role="option"][aria-selected="true"] * {{
        color: {ACTIVE_THEME['text_strong']} !important;
    }}
    [data-baseweb="popover"] input {{
        background-color: {ACTIVE_THEME['input_bg']} !important;
        color: {ACTIVE_THEME['text_strong']} !important;
    }}


    /* Help/tooltip popovers are also mounted in a body portal. Streamlit and
       BaseWeb may put an inline white background on an inner tooltip node, so
       style every tooltip layer explicitly instead of styling only the icon. */
    body [role="tooltip"],
    body [data-baseweb="tooltip"],
    body [data-testid="stTooltipContent"],
    body [data-testid*="Tooltip" i][role="tooltip"],
    body > div[data-baseweb="popover"] [role="tooltip"],
    body > div[data-baseweb="popover"] [data-baseweb="tooltip"] {{
        background-color: {ACTIVE_THEME['panel_bg_2']} !important;
        background: {ACTIVE_THEME['panel_bg_2']} !important;
        color: {ACTIVE_THEME['text_strong']} !important;
        border: 1px solid {ACTIVE_THEME['border_strong']} !important;
        box-shadow: 0 10px 28px {ACTIVE_THEME['shadow']} !important;
        opacity: 1 !important;
    }}
    body [role="tooltip"] *,
    body [data-baseweb="tooltip"] *,
    body [data-testid="stTooltipContent"] *,
    body > div[data-baseweb="popover"] [role="tooltip"] * {{
        color: {ACTIVE_THEME['text_strong']} !important;
        opacity: 1 !important;
    }}
    body [role="tooltip"] p,
    body [data-baseweb="tooltip"] p,
    body [data-testid="stTooltipContent"] p {{
        color: {ACTIVE_THEME['text_strong']} !important;
        line-height: 1.45 !important;
    }}
    body [data-baseweb="tooltip"] [data-popper-arrow],
    body [role="tooltip"] [data-popper-arrow],
    body [data-baseweb="tooltip"] [data-baseweb="arrow"] {{
        background-color: {ACTIVE_THEME['panel_bg_2']} !important;
        color: {ACTIVE_THEME['panel_bg_2']} !important;
        fill: {ACTIVE_THEME['panel_bg_2']} !important;
    }}
    body [data-baseweb="tooltip"] svg,
    body [role="tooltip"] svg {{
        color: {ACTIVE_THEME['panel_bg_2']} !important;
        fill: {ACTIVE_THEME['panel_bg_2']} !important;
    }}

    /* Fallback for Streamlit releases where the help card has no tooltip role
       and is only an anonymous nested div inside the BaseWeb popover portal. */
    body > div[data-baseweb="popover"] > div > div,
    body > div[data-baseweb="popover"] > div > div > div {{
        background-color: {ACTIVE_THEME['panel_bg_2']} !important;
        color: {ACTIVE_THEME['text_strong']} !important;
        border-color: {ACTIVE_THEME['border_strong']} !important;
    }}
    body > div[data-baseweb="popover"] > div > div p,
    body > div[data-baseweb="popover"] > div > div span,
    body > div[data-baseweb="popover"] > div > div label {{
        color: {ACTIVE_THEME['text_strong']} !important;
        opacity: 1 !important;
    }}
    body > div[data-baseweb="popover"] [role="listbox"],
    body > div[data-baseweb="popover"] [data-baseweb="menu"],
    body > div[data-baseweb="popover"] ul {{
        max-height: min(22rem, 58vh) !important;
        overflow-y: scroll !important;
        overscroll-behavior: contain !important;
        scrollbar-gutter: stable !important;
        scrollbar-width: auto !important;
        scrollbar-color: {ACTIVE_THEME['scroll_thumb']} {ACTIVE_THEME['scroll_track']} !important;
    }}
    body > div[data-baseweb="popover"] [role="listbox"]::-webkit-scrollbar,
    body > div[data-baseweb="popover"] [data-baseweb="menu"]::-webkit-scrollbar,
    body > div[data-baseweb="popover"] ul::-webkit-scrollbar {{ width: 12px !important; }}
    body > div[data-baseweb="popover"] [role="listbox"]::-webkit-scrollbar-track,
    body > div[data-baseweb="popover"] [data-baseweb="menu"]::-webkit-scrollbar-track,
    body > div[data-baseweb="popover"] ul::-webkit-scrollbar-track {{ background: {ACTIVE_THEME['scroll_track']} !important; }}
    body > div[data-baseweb="popover"] [role="listbox"]::-webkit-scrollbar-thumb,
    body > div[data-baseweb="popover"] [data-baseweb="menu"]::-webkit-scrollbar-thumb,
    body > div[data-baseweb="popover"] ul::-webkit-scrollbar-thumb {{
        min-height: 36px !important;
        background: {ACTIVE_THEME['scroll_thumb']} !important;
        border: 2px solid {ACTIVE_THEME['scroll_track']} !important;
        border-radius: 999px !important;
    }}
    span[data-baseweb="tag"] {{
        background: color-mix(in srgb, var(--petro-accent) 38%, var(--petro-panel-2)) !important;
        color: var(--petro-text-strong) !important;
    }}
    input:focus, textarea:focus {{
        border-color: var(--petro-accent) !important;
        box-shadow: 0 0 0 2px color-mix(in srgb, var(--petro-accent) 26%, transparent) !important;
    }}

    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label,
    [data-testid="stToggle"] label {{ color: var(--petro-text) !important; }}
    [data-testid="stRadio"] [role="radiogroup"] {{
        gap: .35rem;
        padding: .28rem;
        border: 1px solid var(--petro-border);
        border-radius: 10px;
        background: var(--petro-input);
    }}
    [data-testid="stRadio"] [role="radiogroup"] > label {{
        padding: .24rem .45rem;
        border-radius: 7px;
    }}

    [data-testid="stFileUploader"],
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] > div {{
        background: var(--petro-input) !important;
        color: var(--petro-text) !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        border: 1.5px dashed color-mix(in srgb, var(--petro-accent) 76%, var(--petro-border)) !important;
        border-radius: 11px !important;
    }}
    [data-testid="stFileUploaderDropzone"] *,
    [data-testid="stFileUploaderDropzoneInstructions"] *,
    [data-testid="stFileUploader"] small {{ color: var(--petro-text) !important; }}
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploaderDropzoneInstructions"] small {{ color: var(--petro-muted) !important; opacity: 1 !important; }}
    [data-testid="stFileUploaderDropzone"] svg {{ color: var(--petro-accent) !important; fill: var(--petro-accent) !important; }}
    [data-testid="stFileUploaderDropzone"] button,
    [data-testid="stFileUploaderDropzone"] button:disabled {{
        background: var(--petro-panel-2) !important;
        color: var(--petro-text-strong) !important;
        border: 1px solid var(--petro-border-strong) !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }}
    [data-testid="stFileUploaderDropzone"] button *,
    [data-testid="stFileUploaderDropzone"] button:disabled * {{ color: var(--petro-text-strong) !important; opacity: 1 !important; }}
    button:disabled {{ opacity: .82 !important; }}

    .stButton > button, .stDownloadButton > button {{
        min-height: 2.55rem;
        border-radius: 9px !important;
        border: 1px solid color-mix(in srgb, var(--petro-accent) 76%, white 8%) !important;
        background: linear-gradient(145deg, var(--petro-accent), color-mix(in srgb, var(--petro-accent) 72%, #07131b)) !important;
        color: #FFFFFF !important;
        font-weight: 730 !important;
        box-shadow: 0 7px 18px color-mix(in srgb, var(--petro-accent) 20%, transparent);
        transition: transform .14s ease, filter .14s ease;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        transform: translateY(-1px);
        filter: brightness(1.08);
    }}
    .stButton > button p, .stDownloadButton > button p {{ color: #FFFFFF !important; }}

    div[data-testid="stDataFrame"], div[data-testid="stTable"] {{
        border-radius: 11px;
        overflow: hidden;
        border: 1px solid var(--petro-border);
        background: var(--petro-panel);
    }}
    div[data-testid="stDataFrame"] * {{ color: var(--petro-text) !important; }}

    div[data-testid="stAlert"] {{
        border-radius: 10px;
        border: 1px solid var(--petro-border);
        background: color-mix(in srgb, var(--petro-panel) 91%, var(--petro-accent-soft)) !important;
        color: var(--petro-text) !important;
    }}
    div[data-testid="stAlert"] * {{ color: var(--petro-text) !important; }}

    [data-testid="stTabs"] button {{ color: var(--petro-muted) !important; font-weight: 680; }}
    [data-testid="stTabs"] button[aria-selected="true"] {{
        color: var(--petro-text-strong) !important;
        border-bottom-color: var(--petro-accent) !important;
    }}
    [data-testid="stTabs"] [data-baseweb="tab-border"] {{ background: var(--petro-border) !important; }}


    [data-testid="stProgress"] > div > div {{ background-color: var(--petro-accent) !important; }}
    hr {{ border-color: var(--petro-border) !important; }}

    /* Uploaded-file chips were rendered with a light background in Dark mode. */
    [data-testid="stFileChips"] {{ gap: .45rem !important; }}
    [data-testid="stFileChip"] {{
        background: var(--petro-panel-2) !important;
        color: var(--petro-text) !important;
        border: 1px solid var(--petro-border-strong) !important;
        box-shadow: 0 3px 10px color-mix(in srgb, var(--petro-shadow) 64%, transparent) !important;
    }}
    [data-testid="stFileChipName"] {{ color: var(--petro-text-strong) !important; opacity: 1 !important; }}
    [data-testid="stFileChip"] small,
    [data-testid="stFileChip"] div {{ color: var(--petro-muted) !important; opacity: 1 !important; }}
    [data-testid="stFileChip"] > div:first-child {{
        background: var(--petro-control-bg) !important;
        color: var(--petro-accent-hover) !important;
        border: 1px solid var(--petro-border) !important;
    }}
    [data-testid="stFileChipDeleteBtn"] button,
    [data-testid="stFileChips"] > button,
    [data-testid="stFileChips"] button[aria-label*="upload" i] {{
        background: var(--petro-control-bg) !important;
        color: var(--petro-control-icon) !important;
        border: 1px solid var(--petro-border-strong) !important;
        opacity: 1 !important;
    }}
    [data-testid="stFileChipDeleteBtn"] button:hover,
    [data-testid="stFileChips"] > button:hover,
    [data-testid="stFileChips"] button[aria-label*="upload" i]:hover {{
        background: var(--petro-control-hover) !important;
        color: var(--petro-text-strong) !important;
    }}
    [data-testid="stFileChipDeleteBtn"] svg,
    [data-testid="stFileChips"] button svg {{ color: currentColor !important; fill: currentColor !important; }}

    /* Number-input +/- controls, input adornments and clear buttons. */
    [data-testid="stNumberInputStepDown"],
    [data-testid="stNumberInputStepUp"],
    [data-testid="stTimeInputClearButton"] {{
        background: var(--petro-control-bg) !important;
        color: var(--petro-control-icon) !important;
        border-left: 1px solid var(--petro-border) !important;
        opacity: 1 !important;
    }}
    [data-testid="stNumberInputStepDown"]:hover:not(:disabled),
    [data-testid="stNumberInputStepUp"]:hover:not(:disabled),
    [data-testid="stTimeInputClearButton"]:hover:not(:disabled) {{
        background: var(--petro-control-hover) !important;
        color: var(--petro-text-strong) !important;
    }}
    [data-testid="stNumberInputStepDown"] svg,
    [data-testid="stNumberInputStepUp"] svg,
    [data-testid="stTimeInputClearButton"] svg {{ color: currentColor !important; fill: currentColor !important; }}

    /* Select, date/time, help, popover and toolbar icons. */
    [data-testid="stSelectbox"] svg,
    [data-testid="stMultiSelect"] svg,
    [data-testid="stDateInput"] svg,
    [data-testid="stDateTimeInput"] svg,
    [data-testid="stTimeInput"] svg,
    [data-testid="stPopoverButton"] svg,
    [data-testid="stMenuButton"] svg,
    [data-testid="stElementToolbarButton"] svg,
    [data-testid="stMainMenuButton"] svg {{
        color: var(--petro-control-icon) !important;
        fill: currentColor !important;
        stroke: currentColor !important;
        opacity: 1 !important;
    }}
    [data-testid="stPopoverButton"] button,
    [data-testid="stMenuButtonButton"],
    [data-testid="stElementToolbarButton"],
    [data-testid="stMainMenuButton"] {{
        color: var(--petro-control-icon) !important;
        opacity: 1 !important;
    }}
    button[aria-label*="help" i],
    button[aria-label*="tooltip" i],
    button[title*="help" i] {{ color: var(--petro-control-icon) !important; opacity: 1 !important; }}

    /* Radio, checkbox, toggle and slider states stay visible on dark surfaces. */
    [data-testid="stRadio"] [role="radio"],
    [data-testid="stCheckbox"] [role="checkbox"],
    [data-testid="stToggle"] [role="switch"] {{
        border-color: var(--petro-border-strong) !important;
        opacity: 1 !important;
    }}
    [data-testid="stRadio"] [role="radio"][aria-checked="true"],
    [data-testid="stCheckbox"] [role="checkbox"][aria-checked="true"],
    [data-testid="stToggle"] [role="switch"][aria-checked="true"] {{
        background: var(--petro-accent) !important;
        border-color: var(--petro-accent-hover) !important;
    }}
    [data-testid="stSlider"] [role="slider"] {{
        background: var(--petro-accent) !important;
        border: 2px solid var(--petro-text-strong) !important;
        box-shadow: 0 0 0 2px color-mix(in srgb, var(--petro-accent) 28%, transparent) !important;
    }}
    [data-testid="stSliderThumbValue"],
    [data-testid="stSliderTickBar"] {{ color: var(--petro-text) !important; }}

    /* Tables, menus, dialogs, status cards and code blocks. */
    [data-testid="stPopoverBody"],
    [data-testid="stDialog"],
    [data-testid="stToast"],
    [data-testid="stStatusWidget"],
    [data-testid="stMainMenuPopover"] {{
        background: var(--petro-panel) !important;
        color: var(--petro-text) !important;
        border-color: var(--petro-border) !important;
    }}
    [data-testid="stPopoverBody"] *,
    [data-testid="stDialog"] *,
    [data-testid="stToast"] *,
    [data-testid="stStatusWidget"] *,
    [data-testid="stMainMenuPopover"] * {{ color: var(--petro-text) !important; }}
    [data-testid="stMarkdownPre"],
    [data-testid="stCode"], pre, code {{
        background: var(--petro-input) !important;
        color: var(--petro-text-strong) !important;
        border-color: var(--petro-border) !important;
    }}
    [data-testid="stSpinnerIcon"] {{ color: var(--petro-accent-hover) !important; }}

    /* Plotly modebar follows the selected app theme. */
    .js-plotly-plot .modebar {{
        background: color-mix(in srgb, var(--petro-chart-paper) 92%, transparent) !important;
        border: 1px solid var(--petro-border) !important;
        border-radius: 8px !important;
    }}
    .js-plotly-plot .modebar-btn {{ opacity: .82 !important; }}
    .js-plotly-plot .modebar-btn:hover {{ opacity: 1 !important; }}
    .js-plotly-plot .modebar-btn path {{ fill: var(--petro-chart-text) !important; }}

    /* v80: final high-contrast layer for Streamlit 1.58+ controls.
       Newer Streamlit releases render tooltip icons and select menus through
       BaseWeb portals whose generated classes change between releases.  These
       selectors use stable data-testid / ARIA attributes instead. */

    /* Help/question-mark icon: the icon is stroke-based, so changing only the
       button text color does not make it visible. */
    [data-testid="stTooltipIcon"],
    [data-testid="stTooltipHoverTarget"] {{
        color: var(--petro-control-icon) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}
    [data-testid="stTooltipIcon"] button {{
        width: 1.08rem !important;
        height: 1.08rem !important;
        min-width: 1.08rem !important;
        min-height: 1.08rem !important;
        padding: 0 !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        color: var(--petro-control-icon) !important;
        background: color-mix(in srgb, var(--petro-control-bg) 88%, transparent) !important;
        border: 1px solid color-mix(in srgb, var(--petro-control-icon) 48%, var(--petro-border-strong)) !important;
        border-radius: 999px !important;
        opacity: 1 !important;
        visibility: visible !important;
        cursor: help !important;
    }}
    [data-testid="stTooltipIcon"] button:hover,
    [data-testid="stTooltipIcon"] button:focus-visible {{
        color: var(--petro-text-strong) !important;
        background: var(--petro-control-hover) !important;
        border-color: var(--petro-accent-hover) !important;
        box-shadow: 0 0 0 2px color-mix(in srgb, var(--petro-accent) 28%, transparent) !important;
    }}
    [data-testid="stTooltipIcon"] button svg.icon,
    [data-testid="stTooltipIcon"] svg,
    [data-testid="stTooltipHoverTarget"] svg.icon,
    [data-testid="stTooltipHoverTarget"] svg {{
        width: .82rem !important;
        height: .82rem !important;
        color: currentColor !important;
        stroke: currentColor !important;
        stroke-width: 1.9 !important;
        fill: none !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}

    /* Select and multiselect controls, including selected well tags, clear
       icons, search cursor and open arrow. */
    [data-testid="stSelectbox"] div[data-baseweb="select"],
    [data-testid="stMultiSelect"] div[data-baseweb="select"] {{
        color: var(--petro-text-strong) !important;
        opacity: 1 !important;
    }}
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div {{
        background: var(--petro-input) !important;
        background-color: var(--petro-input) !important;
        color: var(--petro-text-strong) !important;
        border-color: var(--petro-border-strong) !important;
        box-shadow: none !important;
        opacity: 1 !important;
    }}
    [data-testid="stSelectbox"] div[data-baseweb="select"]:focus-within > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"]:focus-within > div {{
        border-color: var(--petro-accent-hover) !important;
        box-shadow: 0 0 0 2px color-mix(in srgb, var(--petro-accent) 24%, transparent) !important;
    }}
    [data-testid="stSelectbox"] input,
    [data-testid="stMultiSelect"] input {{
        color: var(--petro-text-strong) !important;
        caret-color: var(--petro-accent-hover) !important;
        background: transparent !important;
        opacity: 1 !important;
    }}
    [data-testid="stSelectbox"] input::placeholder,
    [data-testid="stMultiSelect"] input::placeholder {{
        color: var(--petro-muted) !important;
        opacity: 1 !important;
    }}
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] {{
        background: color-mix(in srgb, var(--petro-accent) 52%, var(--petro-panel-2)) !important;
        color: var(--petro-text-strong) !important;
        border: 1px solid color-mix(in srgb, var(--petro-accent-hover) 50%, var(--petro-border-strong)) !important;
        opacity: 1 !important;
    }}
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] *,
    [data-testid="stMultiSelect"] span[data-baseweb="tag"] svg {{
        color: var(--petro-text-strong) !important;
        fill: currentColor !important;
        stroke: currentColor !important;
        opacity: 1 !important;
    }}
    [data-testid="stSelectbox"] svg,
    [data-testid="stMultiSelect"] svg {{
        color: var(--petro-control-icon) !important;
        fill: currentColor !important;
        stroke: currentColor !important;
        opacity: 1 !important;
    }}

    /* BaseWeb menu portals.  The no-results row is a li[aria-live="polite"]
       and did not inherit the earlier menu rules, leaving a white rectangle. */
    body [data-baseweb="popover"],
    body [data-baseweb="popover"] > div,
    body [data-baseweb="popover"] > div > div,
    body [data-baseweb="popover"] > div > div > div,
    body [data-baseweb="popover"] [data-baseweb="menu"],
    body [data-baseweb="popover"] [role="listbox"],
    body [data-baseweb="popover"] ul[role="listbox"] {{
        background: var(--petro-panel-2) !important;
        background-color: var(--petro-panel-2) !important;
        color: var(--petro-text-strong) !important;
        border-color: var(--petro-border-strong) !important;
        opacity: 1 !important;
    }}
    body [data-baseweb="popover"] [data-baseweb="menu"] {{
        border: 1px solid var(--petro-border-strong) !important;
        border-radius: 9px !important;
        overflow: hidden !important;
        box-shadow: 0 14px 34px var(--petro-shadow) !important;
    }}
    body [data-baseweb="popover"] [role="listbox"],
    body [data-baseweb="popover"] ul[role="listbox"] {{
        max-height: min(22rem, 58vh) !important;
        overflow-y: scroll !important;
        overscroll-behavior: contain !important;
        scrollbar-gutter: stable !important;
    }}
    body [data-baseweb="popover"] li[aria-live="polite"],
    body [data-baseweb="popover"] [aria-live="polite"][aria-atomic="true"],
    body [data-baseweb="menu"] li[aria-live="polite"] {{
        min-height: 4.5rem !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        margin: 0 !important;
        padding: .9rem !important;
        background: var(--petro-panel-2) !important;
        background-color: var(--petro-panel-2) !important;
        color: var(--petro-muted) !important;
        border: 0 !important;
        opacity: 1 !important;
        cursor: default !important;
    }}
    body [data-baseweb="popover"] li[aria-live="polite"] *,
    body [data-baseweb="popover"] [aria-live="polite"][aria-atomic="true"] * {{
        color: var(--petro-muted) !important;
        opacity: 1 !important;
    }}
    body [data-baseweb="popover"] [role="option"],
    body [data-baseweb="popover"] li[role="option"] {{
        background: var(--petro-panel-2) !important;
        color: var(--petro-text) !important;
        border-bottom: 1px solid color-mix(in srgb, var(--petro-border) 62%, transparent) !important;
        opacity: 1 !important;
    }}
    body [data-baseweb="popover"] [role="option"] *,
    body [data-baseweb="popover"] li[role="option"] * {{
        color: inherit !important;
        opacity: 1 !important;
    }}
    body [data-baseweb="popover"] [role="option"]:hover,
    body [data-baseweb="popover"] [role="option"][aria-selected="true"],
    body [data-baseweb="popover"] li[role="option"]:hover,
    body [data-baseweb="popover"] li[role="option"][aria-selected="true"] {{
        background: var(--petro-control-hover) !important;
        color: var(--petro-text-strong) !important;
    }}

    /* Other icon-only controls use either fill or stroke depending on the
       Streamlit release.  Set both without dimming disabled-but-readable UI. */
    [data-testid="stWidgetLabel"] svg,
    [data-testid="stSelectbox"] button svg,
    [data-testid="stMultiSelect"] button svg,
    [data-testid="stDateInput"] button svg,
    [data-testid="stTimeInput"] button svg,
    [data-testid="stNumberInput"] button svg,
    [data-testid="stPopoverButton"] button svg,
    [data-testid="stMenuButton"] button svg {{
        color: var(--petro-control-icon) !important;
        fill: currentColor !important;
        stroke: currentColor !important;
        opacity: 1 !important;
        visibility: visible !important;
    }}

    /* Disabled widgets remain readable instead of fading into dark panels. */
    button:disabled,
    input:disabled,
    textarea:disabled,
    [aria-disabled="true"] {{
        opacity: 1 !important;
        color: var(--petro-disabled-text) !important;
    }}
    input:disabled,
    textarea:disabled,
    [aria-disabled="true"]:not(button) {{ background: var(--petro-disabled-bg) !important; }}

    @media (max-width: 900px) {{
        .block-container {{ padding-left: .8rem; padding-right: .8rem; }}
        .petro-hero {{ padding: 1rem; border-radius: 13px; }}
        .petro-mark {{ width: 3rem; height: 3rem; flex-basis: 3rem; }}
        .petro-chart-icon {{ width: 1.9rem; height: 1.9rem; }}
        div[data-testid="stMetric"] {{ min-height: 96px; padding: .72rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="petro-hero">
      <div class="petro-hero-row">
        <div class="petro-mark" aria-hidden="true">
          <svg class="petro-chart-icon" viewBox="0 0 48 48" role="img" aria-label="Production trend chart">
            <path class="axis" d="M8 8v32h32" />
            <path class="trend" d="M11 34l8-9 7 4 11-15" />
            <circle class="point" cx="11" cy="34" r="2.6" />
            <circle class="point" cx="19" cy="25" r="2.6" />
            <circle class="point" cx="26" cy="29" r="2.6" />
            <circle class="point" cx="37" cy="14" r="2.6" />
          </svg>
        </div>
        <div>
          <div class="petro-title">Production Test Analysis &amp; Visualization</div>
          <div class="petro-subtitle">Interactive well-test plotting, engineering diagnostics and operational events.</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

def render_section_title(title: str, subtitle: str = "") -> None:
    extra = f'<div style="font-size:.82rem;font-weight:500;color:var(--petro-muted);margin-top:.18rem">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="petro-section-title">{title}{extra}</div>', unsafe_allow_html=True)


with st.sidebar:
    st.markdown("### 📊 Analysis Controls")
    st.caption("Upload data, choose wells and signals, then adjust the chart only when needed.")
    st.radio(
        "Theme",
        ["Light", "Dark"],
        horizontal=True,
        key="ui_theme",
    )

with st.sidebar.expander("1. Data Sources", expanded=True):
    _uploader_generation_v93 = int(st.session_state.get("uploader_generation_v93", 0) or 0)
    uploaded_files = st.file_uploader(
        "Upload test files, reports, device exports, or chat export ZIPs",
        type=["xlsx", "xls", "csv", "txt", "docx", "pdf", "zip", "jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Upload normal test files or a chat export ZIP. Directly uploaded images are OCR-processed automatically; the OCR switch controls images inside ZIP files.",
        key=f"general_data_uploader_v93_{_uploader_generation_v93}",
    )
    uploaded_ocr_images = st.file_uploader(
        "Upload CTU/HMI photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Use this dedicated uploader for field photos like the CTU ALL DATA screens. The display is rectified, OCR values are extracted, and every field remains editable in the OCR Review before plotting.",
        key=f"direct_ctu_image_uploader_v93_{_uploader_generation_v93}",
    )
    # Combine and deduplicate general files and dedicated OCR images.
    _combined_uploads = list(uploaded_files or []) + list(uploaded_ocr_images or [])
    _seen_uploads = set()
    uploaded_files = []
    for _upload in _combined_uploads:
        _identity = (str(getattr(_upload, "name", "")), int(getattr(_upload, "size", 0) or 0), str(getattr(_upload, "file_id", "") or ""))
        if _identity in _seen_uploads:
            continue
        _seen_uploads.add(_identity)
        uploaded_files.append(_upload)
    direct_image_preview_map = {}
    for _upload in uploaded_files:
        if Path(str(getattr(_upload, "name", ""))).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            try:
                direct_image_preview_map[str(_upload.name)] = _upload.getvalue()
            except Exception:
                pass

    st.caption("The interface uses generic examples before upload. Names from your own uploaded data are shown normally after parsing.")
    _restore_notice_v97 = st.session_state.pop("_portable_state_notice_v97", None)
    if _restore_notice_v97:
        st.success(str(_restore_notice_v97))

    whatsapp_text = st.text_area(
        "Paste field-test messages",
        height=180,
        placeholder="""TEST UNIT-01
Date :06-06-2026
Well name : WELL-001
Time @ 10:30
Choke = 100%
W.H.P =60 PSI
Sep. P = 50 PSI
Gas rate =1.386 MMSCF/D
Gross rate = 264 BBL/D
Oil rate= 0 STB/D
Water rate = 264 BBL/D
BS&W = 100 %
Salinity = 35K PPM of NaCl
Pumping P= 849 Psi""",
    )

with st.sidebar.expander("2. Processing", expanded=False):
    continue_current_test = st.checkbox(
        "Continue current test with new uploads",
        value=True,
        key="continue_current_test_v93",
        help=(
            "When a later file belongs to the same well/test, the app appends it to the current analysis, "
            "removes overlapping duplicate timestamps, and keeps your selected signals and chart settings."
        ),
    )
    if st.button("Start a new analysis", key="start_new_analysis_v93", use_container_width=True):
        for _key in (
            "continued_test_data_v93", "continued_upload_signatures_v93", "continued_batch_v93",
            "upload_parse_key_v83", "upload_parse_bundle_v83", "combined_data_key_v83", "combined_data_bundle_v83",
            "manual_events_table", "operation_intervals_table",
            "portable_pdf_data_v97", "portable_pdf_signature_v97",
            "portable_chart_title_v97", "_portable_state_applied_v97",
            "_portable_title_applied_token_v97", "_legacy_pdf_theme_applied_v97",
            "_legacy_dashboard_state_applied_v98", PENDING_SESSION_RESTORE_KEY_V99,
        ):
            st.session_state.pop(_key, None)
        st.session_state["uploader_generation_v93"] = _uploader_generation_v93 + 1
        _clear_heavy_session_state_v83(include_uploads=True)
        gc.collect()
        st.rerun()

    test_gap_hours = st.number_input(
        "New test after inactive gap (hours)",
        min_value=1.0,
        max_value=8760.0,
        value=12.0,
        step=1.0,
        key="test_gap_hours_v97",
        help="Readings stay connected until the time gap is larger than this value.",
    )
    enable_ctu_ocr = st.checkbox(
        "Read images inside chat export ZIPs",
        value=False,
        help=("Direct image uploads are always read. For ZIP OCR, the JPG/PNG/WebP files must be physically "
              "included in the archive; text saying 'image omitted' cannot be OCR-read."),
    )
    if enable_ctu_ocr:
        max_ocr_images = st.number_input(
            "Maximum ZIP images to read (0 = all)",
            min_value=0,
            max_value=5000,
            value=1000,
            step=50,
        )
    else:
        max_ocr_images = 1000


def sanitize_share_text(value) -> str:
    """Return user-entered text unchanged; uploaded identifiers are user-owned data."""
    return "" if value is None else str(value)


def apply_share_safe_anonymization(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Deprecated compatibility shim: uploaded user data must remain unchanged."""
    return df, {}


def inspect_chat_zip_media(file_name: str, file_bytes: bytes) -> dict:
    """Inspect a ZIP without extracting it and report whether OCR media exists."""
    summary = {
        "file": str(file_name),
        "image_files": 0,
        "audio_files": 0,
        "data_attachments": 0,
        "image_omitted_references": 0,
        "document_omitted_references": 0,
        "ocr_rows": 0,
        "invalid_zip": False,
    }
    if Path(str(file_name)).suffix.lower() != ".zip":
        return summary
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/") and not n.startswith("__MACOSX/")]
            image_exts = {".jpg", ".jpeg", ".png", ".webp"}
            audio_exts = {".opus", ".ogg", ".mp3", ".m4a", ".wav", ".aac"}
            data_exts = {".xlsx", ".xls", ".xlsm", ".csv", ".tsv", ".pdf", ".docx"}
            summary["image_files"] = sum(Path(n).suffix.lower() in image_exts for n in names)
            summary["audio_files"] = sum(Path(n).suffix.lower() in audio_exts for n in names)
            summary["data_attachments"] = sum(Path(n).suffix.lower() in data_exts for n in names)
            chat_parts = []
            for n in names:
                if Path(n).suffix.lower() != ".txt":
                    continue
                try:
                    payload = zf.read(n)
                except Exception:
                    continue
                decoded = ""
                for enc in ("utf-8-sig", "utf-16", "utf-8", "cp1256", "latin-1"):
                    try:
                        decoded = payload.decode(enc)
                        break
                    except Exception:
                        continue
                if decoded:
                    chat_parts.append(decoded)
            chat_text = "\n".join(chat_parts)
            summary["image_omitted_references"] = len(re.findall(r"(?i)(?:image|photo)\s+omitted|<media omitted>", chat_text))
            summary["document_omitted_references"] = len(re.findall(r"(?i)document\s+omitted", chat_text))
    except Exception:
        summary["invalid_zip"] = True
    return summary


def load_uploaded_file_once(file_name: str, file_bytes: bytes, parse_images: bool, max_ocr_images: int, parser_build_id: str):
    class CachedUploadedFile(io.BytesIO):
        def __init__(self, data: bytes, name: str):
            super().__init__(data)
            self.name = name
    return load_tabular_file(
        CachedUploadedFile(file_bytes, file_name),
        parse_images=bool(parse_images),
        max_ocr_images=int(max_ocr_images),
    )

frames = []
errors = []
zip_media_summaries = []


def _uploaded_file_identity_v58(uploaded_file):
    """Fast stable identity without hashing the full workbook on every rerun."""
    name = str(getattr(uploaded_file, "name", "uploaded"))
    size = int(getattr(uploaded_file, "size", 0) or 0)
    file_id = str(getattr(uploaded_file, "file_id", "") or "")
    if file_id:
        return (name, size, file_id)
    # Fallback for older Streamlit builds: hash only the first/last 64 KB.
    try:
        view = uploaded_file.getbuffer()
        head = bytes(view[: min(len(view), 65536)])
        tail = bytes(view[max(0, len(view) - 65536):])
        digest = hashlib.sha1(head + tail + str(len(view)).encode()).hexdigest()
    except Exception:
        digest = f"{name}:{size}"
    return (name, size, digest)


upload_key = None
if uploaded_files:
    upload_key = (
        APP_UI_BUILD_ID,
        PARSER_BUILD_ID,
        bool(enable_ctu_ocr),
        int(max_ocr_images),
        tuple(_uploaded_file_identity_v58(f) for f in uploaded_files),
    )
    cached_bundle = st.session_state.get("upload_parse_bundle_v83")
    cached_key = st.session_state.get("upload_parse_key_v83")

    if cached_bundle is not None and cached_key == upload_key:
        frames.extend(cached_bundle.get("frames", []))
        errors.extend(list(cached_bundle.get("errors", [])))
        zip_media_summaries.extend(list(cached_bundle.get("zip_media_summaries", [])))
    else:
        # A new upload invalidates merged data and prepared reports immediately.
        # This prevents stale large objects from accumulating across file changes.
        _clear_heavy_session_state_v83(include_uploads=False)
        parsed_frames = []
        parsed_errors = []
        parsed_zip_media_summaries = []
        upload_progress = st.progress(0.0, text="Reading uploaded files...") if len(uploaded_files) > 1 else None
        for upload_order, f in enumerate(uploaded_files):
            try:
                if upload_progress is not None:
                    upload_progress.progress(
                        upload_order / max(len(uploaded_files), 1),
                        text=f"Reading {upload_order + 1}/{len(uploaded_files)}: {f.name}",
                    )
                file_bytes = bytes(f.getbuffer()) if hasattr(f, "getbuffer") else f.getvalue()
                _suffix = Path(str(f.name)).suffix.lower()
                _zip_media_summary = inspect_chat_zip_media(f.name, file_bytes) if _suffix == ".zip" else None
                _is_direct_image = _suffix in {".jpg", ".jpeg", ".png", ".webp"}
                _parse_images_for_file = bool(enable_ctu_ocr) or _is_direct_image

                # PDFs exported by v97+ carry a safe embedded ZIP with the full
                # dataframe and chart state. Restore it before ordinary PDF table
                # parsing so reopening is exact instead of OCR/visual guesswork.
                _portable_pdf = read_portable_state_from_pdf(f.name, file_bytes) if _suffix == ".pdf" else None
                if _portable_pdf is not None:
                    _portable_signature = hashlib.sha1(file_bytes).hexdigest()
                    if apply_portable_state_to_session(_portable_pdf, _portable_signature):
                        st.rerun()
                    _portable_frame = _portable_pdf.get("data")
                    parsed_tables = [_portable_frame.copy(deep=False)] if isinstance(_portable_frame, pd.DataFrame) and not _portable_frame.empty else []
                else:
                    # Older dashboard PDFs did not contain recoverable events/data.
                    # Their visible background can still restore Light/Dark without
                    # affecting ordinary vendor reports unless the dashboard title
                    # is present in the PDF text.
                    if _suffix == ".pdf":
                        try:
                            from pypdf import PdfReader
                            _legacy_reader = PdfReader(io.BytesIO(file_bytes))
                            _legacy_text = "\n".join((page.extract_text() or "") for page in _legacy_reader.pages[:2])
                            _legacy_signature = hashlib.sha1(file_bytes).hexdigest()
                            if ("Production Test" in _legacy_text or "Well Production Test" in _legacy_text) and st.session_state.get("_legacy_pdf_theme_applied_v97") != _legacy_signature:
                                _legacy_theme = infer_legacy_pdf_theme(file_bytes)
                                if _legacy_theme in UI_THEME_PRESETS:
                                    queue_session_state_restore_v99({"ui_theme": _legacy_theme})
                                    st.session_state["_legacy_pdf_theme_applied_v97"] = _legacy_signature
                                    st.session_state["_portable_state_notice_v97"] = (
                                        f"Detected the {_legacy_theme} theme from an older dashboard PDF. "
                                        "The app will also attempt vector recovery of its readings and events."
                                    )
                                    st.rerun()
                        except Exception:
                            pass
                    parsed_tables = load_uploaded_file_once(
                        f.name, file_bytes, _parse_images_for_file, int(max_ocr_images), PARSER_BUILD_ID
                    )
                    if _suffix == ".pdf" and parsed_tables:
                        _legacy_state = None
                        for _legacy_table in parsed_tables:
                            if isinstance(_legacy_table, pd.DataFrame):
                                _legacy_state = (_legacy_table.attrs or {}).get("legacy_dashboard_state")
                                if _legacy_state:
                                    break
                        if _legacy_state:
                            _legacy_signature = hashlib.sha1(file_bytes).hexdigest()
                            if apply_legacy_dashboard_state_to_session(_legacy_state, _legacy_signature):
                                st.rerun()
                if _zip_media_summary is not None:
                    _zip_media_summary["ocr_rows"] = int(sum(
                        table.get("source_type", pd.Series(dtype=str)).astype(str).str.contains("ocr", case=False, na=False).sum()
                        for table in (parsed_tables or []) if table is not None and not table.empty
                    ))
                    parsed_zip_media_summaries.append(_zip_media_summary)
                if parsed_tables:
                    for table_order, table in enumerate(parsed_tables):
                        if table is None or table.empty:
                            continue
                        table = table.copy(deep=False)
                        table["_upload_order"] = int(upload_order)
                        table["_table_order"] = int(table_order)
                        parsed_frames.append(table)
                else:
                    parsed_errors.append(
                        f"{f.name}: no usable time-series table detected. "
                        "The file may be blank or may not contain date/time plus engineering readings."
                    )
            except Exception as e:
                if locals().get("_zip_media_summary") is not None and _zip_media_summary not in parsed_zip_media_summaries:
                    parsed_zip_media_summaries.append(_zip_media_summary)
                parsed_errors.append(f"{f.name}: {e}")
                with st.expander(f"Technical details: {f.name}", expanded=False):
                    st.code(traceback.format_exc())
            finally:
                try:
                    del file_bytes
                except Exception:
                    pass
                try:
                    del parsed_tables
                except Exception:
                    pass
        if upload_progress is not None:
            upload_progress.progress(1.0, text="Uploaded files parsed")

        st.session_state["upload_parse_key_v83"] = upload_key
        st.session_state["upload_parse_bundle_v83"] = {
            "frames": parsed_frames,
            "errors": list(parsed_errors),
            "zip_media_summaries": list(parsed_zip_media_summaries),
        }
        frames.extend(parsed_frames)
        errors.extend(parsed_errors)
        zip_media_summaries.extend(parsed_zip_media_summaries)
        gc.collect()
else:
    # Removing all uploads should also release the previous workbook data and
    # prepared export files instead of leaving them in the browser session.
    if st.session_state.get("upload_parse_key_v83") is not None:
        _clear_heavy_session_state_v83(include_uploads=True)
        gc.collect()

if whatsapp_text.strip():
    try:
        msg_df = parse_whatsapp_plain_or_export_text(whatsapp_text, source_name="Pasted_Message_Text")
        if not msg_df.empty:
            frames.append(msg_df)
        else:
            errors.append("WhatsApp text: no recognizable production-test report detected")
    except Exception as e:
        errors.append(f"WhatsApp text: {e}")

for _zip_status in zip_media_summaries:
    _zip_name = _zip_status.get("file", "Chat export ZIP")
    _image_files = int(_zip_status.get("image_files", 0) or 0)
    _omitted_refs = int(_zip_status.get("image_omitted_references", 0) or 0)
    _ocr_rows = int(_zip_status.get("ocr_rows", 0) or 0)
    if _zip_status.get("invalid_zip"):
        st.warning(f"{_zip_name}: the ZIP could not be inspected for media files.")
    elif enable_ctu_ocr and _image_files == 0 and _omitted_refs > 0:
        st.warning(
            f"{_zip_name}: ZIP image reading is enabled, but the archive contains no JPG/PNG/WebP files. "
            f"The chat contains {_omitted_refs:,} 'image omitted' reference(s), which means it was exported without media. "
            "Export the WhatsApp chat again and choose Include media, then upload the new ZIP."
        )
    elif enable_ctu_ocr and _image_files > 0:
        if _ocr_rows > 0:
            st.success(f"{_zip_name}: found {_image_files:,} image file(s) and extracted {_ocr_rows:,} OCR row(s).")
        else:
            st.warning(
                f"{_zip_name}: found {_image_files:,} image file(s), but none produced a usable CTU/HMI OCR row. "
                "Open OCR Review to inspect supported screens or upload the photos directly."
            )
    elif not enable_ctu_ocr and _image_files > 0:
        st.info(
            f"{_zip_name}: contains {_image_files:,} image file(s). Enable 'Read images inside chat export ZIPs' to process them."
        )

if errors:
    st.warning("Some files/messages were skipped or could not be parsed:\n\n" + "\n".join(f"- {e}" for e in errors))

def _continuation_compatible_v93(previous: pd.DataFrame, incoming: pd.DataFrame) -> bool:
    """Return True when a new upload can safely extend the current analysis."""
    if previous is None or incoming is None or previous.empty or incoming.empty:
        return False

    def _known_wells(frame):
        if "well" not in frame.columns:
            return set()
        values = frame["well"].dropna().astype(str).str.strip()
        return {
            value.casefold() for value in values
            if value and value.casefold() not in {"unknown", "nan", "none", "unlinked"}
        }

    previous_wells = _known_wells(previous)
    incoming_wells = _known_wells(incoming)
    if previous_wells & incoming_wells:
        return True

    # A direct/ZIP OCR continuation can initially have an Unknown well.  Keep it
    # with the current data only when its timestamp is close enough to be linked
    # to the existing test context.
    incoming_is_ocr = False
    if "source_type" in incoming.columns:
        incoming_is_ocr = incoming["source_type"].astype(str).str.contains("ocr", case=False, na=False).any()
    if incoming_is_ocr and "datetime" in previous.columns and "datetime" in incoming.columns:
        prev_dt = pd.to_datetime(previous["datetime"], errors="coerce").dropna()
        inc_dt = pd.to_datetime(incoming["datetime"], errors="coerce").dropna()
        if not prev_dt.empty and not inc_dt.empty:
            nearest_gap = min(abs(inc_dt.min() - prev_dt.max()), abs(inc_dt.max() - prev_dt.min()))
            if nearest_gap <= pd.Timedelta(hours=36):
                return True
    return False


_saved_continuation_v93 = st.session_state.get("continued_test_data_v93")
_using_saved_continuation_v93 = False

if not frames:
    if (
        continue_current_test
        and isinstance(_saved_continuation_v93, pd.DataFrame)
        and not _saved_continuation_v93.empty
    ):
        # Removing/replacing the uploader must not erase a test that the user is
        # continuing in the same browser session.  The saved table is already
        # parsed, so chart-option changes remain fast and require no re-reading.
        data = _saved_continuation_v93.copy(deep=False)
        rows_merged = 0
        _using_saved_continuation_v93 = True
        st.info("Current test retained. Upload the next file to continue the same analysis, or choose Start a new analysis.")
    else:
        st.info("Start by uploading field files or pasting one or more field-test messages in Data Sources.")
        st.stop()

if not _using_saved_continuation_v93:
    # Keep one merged result for fast control changes. The session state stores
    # the same DataFrame object instead of a deep duplicate, reducing peak memory.
    _whatsapp_key = hashlib.sha1(whatsapp_text.encode("utf-8", errors="ignore")).hexdigest() if whatsapp_text.strip() else ""
    _combined_key = (APP_UI_BUILD_ID, PARSER_BUILD_ID, upload_key, _whatsapp_key)
    _combined_cached = st.session_state.get("combined_data_bundle_v83")
    _combined_cached_key = st.session_state.get("combined_data_key_v83")

    if _combined_cached is not None and _combined_cached_key == _combined_key:
        current_upload_data_v93 = _combined_cached["data"].copy(deep=False)
        rows_merged = int(_combined_cached.get("rows_merged", 0))
    else:
        current_upload_data_v93 = pd.concat(frames, ignore_index=True, sort=False, copy=False)
        rows_before_dedup = len(current_upload_data_v93)
        try:
            if hasattr(_tmu_parser, "merge_duplicate_test_rows_v53"):
                current_upload_data_v93 = _tmu_parser.merge_duplicate_test_rows_v53(current_upload_data_v93)
        except Exception as dedup_error:
            errors.append(f"Duplicate-row merge was skipped: {dedup_error}")
        rows_merged = max(0, rows_before_dedup - len(current_upload_data_v93))
        st.session_state["combined_data_key_v83"] = _combined_key
        st.session_state["combined_data_bundle_v83"] = {
            "data": current_upload_data_v93.copy(deep=False),
            "rows_merged": int(rows_merged),
        }

    if continue_current_test:
        signature_text = repr((APP_UI_BUILD_ID, PARSER_BUILD_ID, upload_key, _whatsapp_key))
        continuation_signature = hashlib.sha1(signature_text.encode("utf-8", errors="ignore")).hexdigest()
        seen_signatures = set(st.session_state.get("continued_upload_signatures_v93", []))
        previous = st.session_state.get("continued_test_data_v93")

        if continuation_signature not in seen_signatures:
            batch_no = int(st.session_state.get("continued_batch_v93", 0) or 0) + 1
            incoming = current_upload_data_v93.copy()
            incoming["_continuation_batch"] = batch_no
            if isinstance(previous, pd.DataFrame) and not previous.empty and _continuation_compatible_v93(previous, incoming):
                previous = previous.copy()
                if "_continuation_batch" not in previous.columns:
                    previous["_continuation_batch"] = max(0, batch_no - 1)
                data = pd.concat([previous, incoming], ignore_index=True, sort=False, copy=False)
                previous_rows = len(previous)
            else:
                data = incoming
                previous_rows = 0
                if isinstance(previous, pd.DataFrame) and not previous.empty:
                    # Different well/context uploaded after replacing the prior
                    # file: start fresh automatically while keeping UI choices.
                    seen_signatures.clear()
                    batch_no = 1
                    data["_continuation_batch"] = batch_no
                    st.info("A different well/test context was detected, so the data view started a new analysis automatically.")

            seen_signatures.add(continuation_signature)
            st.session_state["continued_upload_signatures_v93"] = list(seen_signatures)[-100:]
            st.session_state["continued_batch_v93"] = batch_no
            st.session_state["continued_test_data_v93"] = data.copy(deep=False)
            if previous_rows:
                st.success(
                    f"Continued the current analysis: loaded the new file(s) while retaining {previous_rows:,} earlier row(s). "
                    "Overlapping timestamps will be merged automatically."
                )
        else:
            if isinstance(previous, pd.DataFrame) and not previous.empty:
                data = previous.copy(deep=False)
            else:
                data = current_upload_data_v93.copy(deep=False)
    else:
        # Turning continuation off returns to the normal uploader-only behavior.
        data = current_upload_data_v93.copy(deep=False)
        st.session_state.pop("continued_test_data_v93", None)
        st.session_state.pop("continued_upload_signatures_v93", None)
        st.session_state.pop("continued_batch_v93", None)


if rows_merged:
    st.caption(f"Merged {rows_merged:,} repeated row(s) with the same well and date/time, keeping the most complete values.")

# Internal ingestion-order fields are used only while resolving duplicate rows.
# They are not engineering measurements and must never appear in plots/tables.
data.drop(columns=[c for c in ["_upload_order", "_table_order", "_source_row_order"] if c in data.columns], inplace=True, errors="ignore")


def _auto_link_ocr_rows_by_time_context(df: pd.DataFrame, max_gap_hours: float = 3.0) -> pd.DataFrame:
    """Link an OCR image only when one nearby test context is unambiguous.

    Directly uploaded photos do not contain the well name, but their WhatsApp
    filename contains an accurate timestamp. When spreadsheet/text readings
    from exactly one test exist within the short time window, inherit that
    well/test ID. Values still remain review-required until the engineer
    approves the OCR fields.
    """
    parser_linker = getattr(_tmu_parser, "auto_link_ocr_rows_by_time_context", None)
    if parser_linker is not None:
        # pandas 3 uses strict assignment rules for Arrow/numeric columns. ZIP
        # concatenation can infer all-null OCR suggestion columns as non-text,
        # so normalize them before calling the parser-level linker.
        safe_df = df.copy()
        for _col in (
            "well", "test_id", "source_type", "link_status",
            "suggested_well", "suggested_test_id", "suggested_link_reason",
        ):
            if _col not in safe_df.columns:
                safe_df[_col] = pd.Series([None] * len(safe_df), index=safe_df.index, dtype="object")
            else:
                safe_df[_col] = safe_df[_col].astype("object")
        try:
            return parser_linker(safe_df, max_gap_hours=max_gap_hours)
        except (TypeError, ValueError) as link_error:
            # Never terminate the full app because optional OCR context linking
            # failed. The OCR rows remain available for manual review/linking.
            safe_df.attrs["ocr_context_link_warning"] = str(link_error)
            return safe_df
    if df is None or df.empty or "datetime" not in df.columns or "source_type" not in df.columns:
        return df
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    ocr_mask = out["source_type"].astype(str).str.contains("ocr", case=False, na=False)
    if not ocr_mask.any():
        return out
    well_text = out.get("well", pd.Series("Unknown", index=out.index)).fillna("Unknown").astype(str).str.strip()
    test_text = out.get("test_id", pd.Series("", index=out.index)).fillna("").astype(str).str.strip()
    anchors = out[
        (~ocr_mask)
        & out["datetime"].notna()
        & well_text.ne("")
        & ~well_text.str.casefold().eq("unknown")
        & test_text.ne("")
    ].copy()
    if anchors.empty:
        return out

    for idx in out.index[ocr_mask & out["datetime"].notna()]:
        current_well = str(out.at[idx, "well"] if "well" in out.columns else "Unknown").strip()
        current_link = str(out.at[idx, "link_status"] if "link_status" in out.columns else "").strip()
        if current_well and current_well.casefold() != "unknown" and current_link not in {"", "ocr_manual_link_required"}:
            continue
        deltas = (anchors["datetime"] - out.at[idx, "datetime"]).abs()
        nearby = anchors.loc[deltas <= pd.Timedelta(hours=float(max_gap_hours))].copy()
        if nearby.empty:
            continue
        contexts = nearby[[c for c in ["well", "test_id", "test_sequence"] if c in nearby.columns]].drop_duplicates()
        if "test_id" not in contexts.columns or contexts["test_id"].astype(str).nunique() != 1:
            continue
        nearest_idx = deltas.loc[nearby.index].idxmin()
        anchor = anchors.loc[nearest_idx]
        for col in ["well", "test_id", "test_sequence"]:
            if col in anchor.index and pd.notna(anchor[col]):
                out.at[idx, col] = anchor[col]
        out.at[idx, "link_status"] = "ocr_auto_linked_by_timestamp"
        out.at[idx, "suggested_well"] = anchor.get("well", "")
        out.at[idx, "suggested_test_id"] = anchor.get("test_id", "")
        gap_minutes = float(deltas.loc[nearest_idx].total_seconds() / 60.0)
        out.at[idx, "suggested_link_reason"] = f"Unique nearby test reading ({gap_minutes:.1f} min)"
    return out


data = _auto_link_ocr_rows_by_time_context(data, max_gap_hours=3.0)
_ocr_link_warning = str(data.attrs.pop("ocr_context_link_warning", "") or "").strip()
if _ocr_link_warning:
    st.warning(
        "Image OCR rows were loaded, but automatic well/test linking was skipped safely. "
        "You can still review and link the OCR rows manually."
    )

# Parser-level quality review. Impossible physical values are excluded from
# plotted canonical columns but retained in Rejected Values for audit.
_quality_mask = pd.Series(False, index=data.index)
if "review_required" in data.columns:
    _quality_mask |= data["review_required"].fillna(False).astype(bool)
if "data_quality_note" in data.columns:
    _quality_mask |= data["data_quality_note"].fillna("").astype(str).str.strip().ne("")
if _quality_mask.any():
    with st.expander(f"⚠️ Engineering Data Checks ({int(_quality_mask.sum()):,} row(s))", expanded=False):
        st.caption(
            "These are automatic engineering consistency checks—not a statement that the whole file is bad. "
            "A row appears here only when a value is outside a physical range, a balance does not close, "
            "or the parser corrected/withheld a source value. Original values remain available in Rejected Values for audit."
        )
        _quality_columns = [
            "source", "sheet", "source_row", "well", "datetime",
            "data_quality_note", "rejected_values", "note",
        ]
        _quality_columns = [column for column in _quality_columns if column in data.columns]
        _quality_data = data.loc[_quality_mask, _quality_columns].copy()
        _quality_download = _quality_data.copy(deep=False)
        _quality_data, _quality_omitted_rows, _quality_omitted_cols = limited_dataframe_preview(
            _quality_data, max_rows=1000, max_cols=24
        )
        _quality_rename = {
            "data_quality_note": "Engineering Check",
            "rejected_values": "Original / Withheld Values",
            "source_row": "Source Row",
            "datetime": "Date / Time",
        }
        _quality_data = _quality_data.rename(columns=_quality_rename)
        _quality_download = _quality_download.rename(columns=_quality_rename)
        st.dataframe(_quality_data, width="stretch", hide_index=True)
        if _quality_omitted_rows or _quality_omitted_cols:
            st.caption("Preview limited for stability; the CSV contains all engineering checks.")
        st.download_button(
            "Download engineering checks CSV",
            data=_quality_download.to_csv(index=False).encode("utf-8-sig"),
            file_name="production_test_engineering_checks.csv",
            mime="text/csv",
            key="download_data_quality_review_v69",
        )

# Raw choke source columns are preserved. A user-selectable unified curve is
# created later, after the safe parsing/mapping steps, so changing the display
# unit never changes the original uploaded values.

def _merge_continuation_duplicate_rows_v93(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Merge overlapping continuation files after test IDs are rebuilt.

    A later export often repeats the final rows of the earlier export.  Newer
    upload batches take priority, while missing cells are filled from the older
    copy and notes/audit text are preserved.  The operation runs only when the
    continuation feature is active, so normal uploads keep the existing fast
    path.
    """
    if df is None or df.empty or "_continuation_batch" not in df.columns or "datetime" not in df.columns:
        return df, 0
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    keys = [c for c in ("well", "datetime", "test_id") if c in out.columns]
    if len(keys) < 2:
        return out, 0
    out["_continuation_batch"] = pd.to_numeric(out["_continuation_batch"], errors="coerce").fillna(0)
    out = out.sort_values(["_continuation_batch"], ascending=False, kind="stable")
    text_merge_cols = {"note", "data_quality_note", "rejected_values", "suggested_link_reason"}
    rows = []
    for _, group in out.groupby(keys, dropna=False, sort=False):
        row = group.iloc[0].copy()
        for col in out.columns:
            values = group[col].dropna()
            if col in text_merge_cols:
                clean = [str(v).strip() for v in values if str(v).strip() and str(v).strip().casefold() != "nan"]
                row[col] = "; ".join(dict.fromkeys(clean))
            elif len(values):
                # Group is newest-first, therefore the first real value wins.
                row[col] = values.iloc[0]
        rows.append(row)
    merged = pd.DataFrame(rows).reset_index(drop=True)
    return merged, max(0, len(out) - len(merged))


# One clear rule only: a new test starts after the user-selected inactive gap.
# Existing parser IDs are rebuilt so spreadsheet, message and OCR readings use
# the same rule and nearby image readings remain connected.
try:
    data = assign_test_ids(
        data,
        gap_hours=float(test_gap_hours),
        preserve_existing=False,
        group_unknown_by_source=False,
    )
except TypeError:
    data = assign_test_ids(data, gap_hours=float(test_gap_hours))

_continuation_rows_merged_v93 = 0
if continue_current_test and "_continuation_batch" in data.columns:
    data, _continuation_rows_merged_v93 = _merge_continuation_duplicate_rows_v93(data)
    if _continuation_rows_merged_v93:
        st.caption(
            f"Merged {_continuation_rows_merged_v93:,} overlapping row(s) from the continued test, keeping the newest values."
        )

if "datetime" in data.columns:
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
if "date" in data.columns:
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

# Learn/apply user mappings before feature lists and plots are built.
data, active_column_aliases = editable_column_mapping_panel(data)

# v52 safety: make sure any Pumping Pressure column rescued from hidden/far-right
# Excel columns remains canonical, numeric, and available for plotting after user
# mappings/cache reruns.
try:
    if hasattr(_tmu_parser, "ensure_pumping_pressure_column_v48"):
        data = _tmu_parser.ensure_pumping_pressure_column_v48(data)
    if "pumping_pressure_psi" in data.columns:
        data["pumping_pressure_psi"] = pd.to_numeric(data["pumping_pressure_psi"], errors="coerce")
    data = normalize_ctu_ocr_signals(data)
except Exception:
    pass

# Save the merged/re-segmented table for the next continuation upload.  This is
# one shallow session copy, not a second parser cache, so chart interactions stay
# responsive and the selected signals/options remain untouched.
if continue_current_test:
    data.drop(columns=["_continuation_batch"], inplace=True, errors="ignore")
    st.session_state["continued_test_data_v93"] = data.copy(deep=False)
else:
    data.drop(columns=["_continuation_batch"], inplace=True, errors="ignore")

ocr_mask = data.get("source_type", pd.Series([], dtype=str)).astype(str).str.contains("ocr", case=False, na=False) if "source_type" in data.columns else pd.Series([False] * len(data), index=data.index)
if ocr_mask.any():
    with st.expander("CTU / HMI OCR Review & Approval", expanded=True):
        st.markdown(
            "Review OCR values before engineering use. Low-confidence fields remain editable, "
            "and OCR rows are excluded from confirmed status until you approve them."
        )

        ocr_numeric_cols = [
            "ctu_weight_lbf", "ctu_lt_weight_lbf", "ctu_wellhead_pressure_psi",
            "ctu_circulation_pressure_psi", "ctu_reel_depth_ft", "ctu_reel_speed_ftmin",
            "ctu_fluid_rate_bpm", "ctu_n2_rate_scfm", "ctu_fluid_total_bbl", "ctu_n2_total_scf",
        ]
        ocr_meta_cols = [
            "image_file", "datetime", "well", "test_id", "link_status",
            "ocr_fields_found", "ocr_confidence", "screen_rectified",
            "screen_detection_score", "ocr_low_confidence_fields", "caption_text",
        ]
        ocr_cols = [c for c in ocr_meta_cols + ocr_numeric_cols if c in data.columns]
        review_df = data.loc[ocr_mask, ocr_cols].copy()
        review_df.insert(0, "row_id", review_df.index.astype(int))
        _existing_approval = data.loc[ocr_mask, "ocr_approved"].fillna(False).astype(bool) if "ocr_approved" in data.columns else pd.Series(False, index=review_df.index)
        review_df.insert(1, "Approve OCR", _existing_approval.reindex(review_df.index).fillna(False).to_numpy())

        # Preview directly uploaded field photos. ZIP-contained images still appear
        # by filename and can be reviewed after extraction outside the application.
        preview_names = [
            str(name) for name in data.loc[ocr_mask, "_private_image_file"].dropna().astype(str).unique()
            if "_private_image_file" in data.columns and str(name) in direct_image_preview_map
        ] if "_private_image_file" in data.columns else [
            str(name) for name in review_df.get("image_file", pd.Series(dtype=str)).dropna().astype(str).unique()
            if str(name) in direct_image_preview_map
        ]
        if preview_names:
            def _preview_display_name(private_name):
                return str(private_name)
            selected_preview = st.selectbox(
                "Image preview", preview_names, format_func=_preview_display_name, key="ocr_image_preview_v70"
            )
            st.image(
                direct_image_preview_map[selected_preview],
                caption=f"OCR source: {_preview_display_name(selected_preview)}",
                width="stretch",
            )

        confirmed_test_options = []
        if "test_id" in data.columns:
            confirmed_test_options = sorted([
                str(t) for t in data.loc[~ocr_mask, "test_id"].dropna().astype(str).unique()
                if not str(t).startswith("Unlinked")
            ])
        well_options = ["Unknown"]
        if "well" in data.columns:
            well_options += sorted([
                str(w) for w in data.loc[~ocr_mask, "well"].dropna().astype(str).unique()
                if str(w).strip() and str(w).lower() != "unknown"
            ])

        column_config = {
            "row_id": st.column_config.NumberColumn("Row", disabled=True),
            "Approve OCR": st.column_config.CheckboxColumn(
                "Approve OCR", help="Approve only after checking the image and all extracted fields."
            ),
            "ocr_confidence": st.column_config.ProgressColumn(
                "Overall confidence", min_value=0.0, max_value=1.0, format="%.0f%%"
            ),
        }
        if well_options:
            column_config["well"] = st.column_config.SelectboxColumn("Well", options=well_options)
        if confirmed_test_options:
            column_config["test_id"] = st.column_config.SelectboxColumn(
                "Test ID", options=["Unlinked_OCR_or_Unknown_Well"] + confirmed_test_options
            )
        _ocr_review_labels = {
            "ctu_circulation_pressure_psi": "Pumping Pressure (image OCR)",
            "ctu_wellhead_pressure_psi": "WHP (image OCR)",
        }
        for col in ocr_numeric_cols:
            if col in review_df.columns:
                column_config[col] = st.column_config.NumberColumn(
                    _ocr_review_labels.get(col, column_label(col)), format="%.3f"
                )

        editable_columns = {"Approve OCR", "datetime", "well", "test_id", *ocr_numeric_cols}
        edited_review = st.data_editor(
            review_df,
            width="stretch",
            height=min(520, 150 + 42 * max(1, len(review_df))),
            key="ctu_ocr_review_editor_v70",
            column_config=column_config,
            disabled=[c for c in review_df.columns if c not in editable_columns],
        )

        for _, erow in edited_review.iterrows():
            rid = int(erow.get("row_id"))
            if rid not in data.index:
                continue
            for col in ["datetime", "well", "test_id", *ocr_numeric_cols]:
                if col in edited_review.columns and col in data.columns:
                    data.at[rid, col] = erow.get(col)
            approved = bool(erow.get("Approve OCR", False))
            if "ocr_approved" not in data.columns:
                data["ocr_approved"] = False
            data.at[rid, "ocr_approved"] = approved
            # The CTU screen's Circulation Pressure is the same engineering
            # signal as Pumping Pressure; Wellhead Pressure is WHP. Keep the raw
            # OCR fields for audit but update the canonical plotted channels.
            if "ctu_circulation_pressure_psi" in data.columns:
                data.at[rid, "pumping_pressure_psi"] = pd.to_numeric(
                    pd.Series([data.at[rid, "ctu_circulation_pressure_psi"]]), errors="coerce"
                ).iloc[0]
            if "ctu_wellhead_pressure_psi" in data.columns:
                data.at[rid, "whp_psi"] = pd.to_numeric(
                    pd.Series([data.at[rid, "ctu_wellhead_pressure_psi"]]), errors="coerce"
                ).iloc[0]
            well_ok = str(data.at[rid, "well"]).strip().lower() not in ["", "unknown", "nan"] if "well" in data.columns else False
            if approved:
                data.at[rid, "link_status"] = "ocr_manually_verified"
                data.at[rid, "review_required"] = not well_ok
                data.at[rid, "ocr_status"] = "manually_verified"
                if "data_quality_note" in data.columns:
                    _existing_note = str(data.at[rid, "data_quality_note"] or "").strip(" ;")
                    data.at[rid, "data_quality_note"] = (
                        f"{_existing_note}; OCR values manually verified" if _existing_note
                        else "OCR values manually verified"
                    )
        data = normalize_ctu_ocr_signals(data)

        unapproved = ocr_mask & data.get("review_required", pd.Series(True, index=data.index)).fillna(True).astype(bool)
        if unapproved.any():
            st.info(
                f"{int(unapproved.sum())} OCR row(s) still require review. They remain visible for editing "
                "but are excluded from charts until approved."
            )

        # Preserve manual OCR corrections/approvals when the next continuation
        # file is added in the same session.
        if continue_current_test:
            st.session_state["continued_test_data_v93"] = data.copy(deep=False)



def parse_manual_events(text, reference_start=None):
    events = []
    if not text or not str(text).strip():
        return events

    ref_date = pd.Timestamp(reference_start).date() if reference_start is not None and pd.notna(reference_start) else None

    for line in str(text).splitlines():
        line = line.strip()
        if not line or "|" not in line:
            continue
        when_txt, label = [p.strip() for p in line.split("|", 1)]
        if not label:
            continue

        dt = pd.to_datetime(when_txt, errors="coerce", dayfirst=True)
        if pd.isna(dt):
            # Accept HH:MM only. For multi-day tests, user should include the date.
            m = pd.Series([when_txt]).str.extract(r"(\d{1,2})[:.](\d{2})").iloc[0]
            if ref_date is not None and not m.isna().any():
                dt = pd.Timestamp(ref_date) + pd.Timedelta(hours=int(m.iloc[0]), minutes=int(m.iloc[1]))
        if pd.notna(dt):
            events.append({"datetime": pd.Timestamp(dt), "label": label})
    return events



def _parse_interval_text(value, *, default_unit="minutes"):
    """Parse user-friendly intervals such as 2 hours, 90 min, 1.5 days, 2 months.

    Returns a dictionary with a pandas resampling rule, a Plotly tick value, and
    a Timedelta approximation used by compressed timelines. Invalid or
    non-positive values return None instead of raising an application error.
    """
    text = str(value or "").strip().lower()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)

    # Accept compact forms such as 2h, 30m, 1d, 2w, 3M, and plain numbers.
    compact = re.fullmatch(r"([0-9]*\.?[0-9]+)\s*([a-z]+)?", text)
    if not compact:
        return None
    amount = float(compact.group(1))
    unit = (compact.group(2) or default_unit).lower()
    if not np.isfinite(amount) or amount <= 0:
        return None

    aliases = {
        "s": "second", "sec": "second", "secs": "second", "second": "second", "seconds": "second",
        "m": "minute", "min": "minute", "mins": "minute", "minute": "minute", "minutes": "minute",
        "h": "hour", "hr": "hour", "hrs": "hour", "hour": "hour", "hours": "hour",
        "d": "day", "day": "day", "days": "day",
        "w": "week", "wk": "week", "wks": "week", "week": "week", "weeks": "week",
        "mo": "month", "mos": "month", "month": "month", "months": "month",
        "y": "year", "yr": "year", "yrs": "year", "year": "year", "years": "year",
    }
    unit = aliases.get(unit)
    if unit is None:
        return None

    if unit in {"month", "year"}:
        rounded = int(round(amount))
        if rounded < 1 or abs(amount - rounded) > 1e-9:
            return None
        months = rounded if unit == "month" else rounded * 12
        return {
            "label": f"{rounded} {unit}{'' if rounded == 1 else 's'}",
            "resample_rule": f"{months}MS",
            "plotly_dtick": f"M{months}",
            "timedelta": pd.Timedelta(days=30.4375 * months),
        }

    seconds_per_unit = {
        "second": 1.0,
        "minute": 60.0,
        "hour": 3600.0,
        "day": 86400.0,
        "week": 7.0 * 86400.0,
    }
    total_seconds = amount * seconds_per_unit[unit]
    if total_seconds < 0.001:
        return None
    total_ms = int(round(total_seconds * 1000.0))
    # Pandas accepts milliseconds and avoids fractional rule strings.
    return {
        "label": f"{amount:g} {unit}{'' if amount == 1 else 's'}",
        "resample_rule": f"{total_ms}ms",
        "plotly_dtick": total_ms,
        "timedelta": pd.Timedelta(milliseconds=total_ms),
    }


def _custom_interval_choice(preset, custom_text):
    if str(preset) != "Custom":
        return str(preset)
    parsed = _parse_interval_text(custom_text)
    return f"Custom: {parsed['label']}" if parsed else "Custom: invalid"


def interval_select_with_custom(label, options, *, index=0, key, placeholder="e.g. 2 hours", help_text=None):
    """Render a compact preset dropdown and a custom interval box beside it."""
    c1, c2 = st.columns([1.35, 1.0], gap="small")
    with c1:
        preset = st.selectbox(
            label,
            [*options, "Custom"],
            index=index,
            key=f"{key}_preset",
            help=help_text,
        )
    with c2:
        custom_text = st.text_input(
            "Custom",
            value=str(st.session_state.get(f"{key}_custom_value", "")),
            placeholder=placeholder,
            key=f"{key}_custom_value",
            disabled=preset != "Custom",
            help="Type a positive interval such as 2 hours, 90 minutes, 1.5 days, 2 months, or 1 year.",
        )
    resolved = _custom_interval_choice(preset, custom_text)
    if preset == "Custom" and resolved == "Custom: invalid":
        st.warning(f"Enter a valid {label.lower()} such as 2 hours or 30 minutes.")
    return resolved


def _unique_signal_labels(features):
    """Create unique readable labels for the drag-order component."""
    counts = {}
    labels = []
    mapping = {}
    for feature in features:
        base = str(column_label(feature))
        counts[base] = counts.get(base, 0) + 1
        label = base if counts[base] == 1 else f"{base} · {counts[base]}"
        labels.append(label)
        mapping[label] = feature
    return labels, mapping


def draggable_signal_order(selected, *, key):
    """Return selected signals in user-controlled drag order.

    Selection remains a normal searchable multiselect. Reordering is handled by
    a small drag list so users do not have to remove and re-add signals merely
    to change panel order.
    """
    selected = list(dict.fromkeys([x for x in selected if x]))
    if len(selected) <= 1:
        return selected

    state_key = f"{key}_state"
    previous = [x for x in st.session_state.get(state_key, []) if x in selected]
    ordered_seed = previous + [x for x in selected if x not in previous]
    labels, label_to_feature = _unique_signal_labels(ordered_seed)

    if _sort_items is None:
        st.caption("Signal order follows the selected list until the drag-order component is installed.")
        st.session_state[state_key] = ordered_seed
        return ordered_seed

    dark = ACTIVE_THEME_NAME == "Dark"
    bg = "#0D2430" if dark else "#F4F8FB"
    item_bg = "#173B4B" if dark else "#FFFFFF"
    text = "#F2F7FA" if dark else "#102A43"
    border = "#5D8DA3" if dark else "#9CB7C7"
    custom_style = f"""
    .sortable-component {{ background: transparent; padding: 0; }}
    .sortable-container {{ background: {bg}; border: 1px solid {border}; border-radius: 9px; padding: 5px; }}
    .sortable-container-header {{ color: {text}; font-weight: 700; background: transparent; padding: 3px 5px 7px 5px; }}
    .sortable-container-body {{ background: transparent; }}
    .sortable-item, .sortable-item:hover {{
        background: {item_bg}; color: {text}; border: 1px solid {border};
        border-radius: 7px; margin: 4px 0; padding: 8px 10px; cursor: grab;
        font-weight: 600; line-height: 1.25;
    }}
    """
    set_signature = hashlib.sha1("|".join(labels).encode("utf-8")).hexdigest()[:10]
    try:
        sorted_labels = _sort_items(
            labels,
            header="Drag to change plot order",
            direction="vertical",
            custom_style=custom_style,
            key=f"{key}_{set_signature}_{ACTIVE_THEME_NAME}",
        )
        ordered = [label_to_feature[x] for x in sorted_labels if x in label_to_feature]
        ordered += [x for x in selected if x not in ordered]
    except Exception:
        ordered = ordered_seed
    st.session_state[state_key] = ordered
    return ordered

def time_aggregation_rule(choice):
    choice = str(choice or "")
    if choice.startswith("Custom:"):
        parsed = _parse_interval_text(choice.split(":", 1)[1].strip())
        return parsed.get("resample_rule") if parsed else None
    return {
        "Raw data": None,
        "5 minutes": "5min",
        "15 minutes": "15min",
        "30 minutes": "30min",
        "1 hour": "1h",
        "6 hours": "6h",
        "1 day": "1D",
        "1 month": "MS",
        "1 year": "YS",
    }.get(choice)


def aggregate_time_data(df, agg_choice):
    rule = time_aggregation_rule(agg_choice)
    if rule is None or df.empty or "datetime" not in df.columns or not df["datetime"].notna().any():
        return df

    numeric = available_numeric_columns(df)
    if not numeric:
        return df

    pieces = []
    # Never average two separate tests into one resampling bin. Keep test_id in
    # the grouping whenever available, then preserve source/report context.
    agg_group_cols = [c for c in ["well", "test_id"] if c in df.columns]
    if not agg_group_cols:
        agg_group_cols = ["well"] if "well" in df.columns else []
    grouped = df.dropna(subset=["datetime"]).groupby(agg_group_cols, dropna=False) if agg_group_cols else [("All", df.dropna(subset=["datetime"]))]
    for group_key, g in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        group_meta = dict(zip(agg_group_cols, group_key))
        g = g.sort_values("datetime").set_index("datetime")
        res = g[numeric].resample(rule).mean().dropna(how="all")
        if res.empty:
            continue
        res["well"] = group_meta.get("well", "All")
        if "test_id" in agg_group_cols:
            res["test_id"] = group_meta.get("test_id", "")
        res["source"] = ", ".join(sorted(set(map(str, g.get("source", pd.Series(dtype=str)).dropna().unique())))) if "source" in g else ""
        res["sheet"] = "Aggregated"
        res = res.reset_index()
        res["date"] = res["datetime"].dt.floor("D")
        res["time_text"] = res["datetime"].dt.strftime("%H:%M")
        pieces.append(res)

    return pd.concat(pieces, ignore_index=True, sort=False) if pieces else df


def x_axis_tick_kwargs(scale):
    scale = str(scale or "")
    if scale.startswith("Custom:"):
        parsed = _parse_interval_text(scale.split(":", 1)[1].strip())
        if parsed:
            delta = parsed["timedelta"]
            if delta >= pd.Timedelta(days=365):
                tickformat = "%Y"
            elif delta >= pd.Timedelta(days=28):
                tickformat = "%b-%Y"
            elif delta >= pd.Timedelta(days=1):
                tickformat = "%d-%b-%Y"
            else:
                tickformat = "%d-%b-%Y<br>%H:%M"
            return {"tickformat": tickformat, "dtick": parsed["plotly_dtick"]}
        return {"tickformat": "%d-%b-%Y<br>%H:%M", "nticks": 10}
    if scale == "30 minutes":
        return {"tickformat": "%d-%b-%Y<br>%H:%M", "dtick": 30 * 60 * 1000}
    if scale == "1 hour":
        return {"tickformat": "%d-%b-%Y<br>%H:%M", "dtick": 60 * 60 * 1000}
    if scale == "3 hours":
        return {"tickformat": "%d-%b-%Y<br>%H:%M", "dtick": 3 * 60 * 60 * 1000}
    if scale == "6 hours":
        return {"tickformat": "%d-%b-%Y<br>%H:%M", "dtick": 6 * 60 * 60 * 1000}
    if scale == "12 hours":
        return {"tickformat": "%d-%b-%Y<br>%H:%M", "dtick": 12 * 60 * 60 * 1000}
    if scale == "1 day":
        return {"tickformat": "%d-%b-%Y", "dtick": 24 * 60 * 60 * 1000}
    if scale == "1 month":
        return {"tickformat": "%b-%Y", "dtick": "M1"}
    if scale == "1 year":
        return {"tickformat": "%Y", "dtick": "M12"}
    return {"tickformat": "%d-%b-%Y<br>%H:%M", "nticks": 10}


def history_axis_tick_kwargs(df: pd.DataFrame, max_ticks: int = 9) -> dict:
    """Always show readable dates for production-history points.

    A fixed one-year dtick can produce no visible labels when the selected
    history spans only days or months. Explicit ticks guarantee that the first
    and last test dates, plus representative dates in between, are shown.
    """
    if df is None or df.empty or "datetime" not in df.columns:
        return {}
    dts = pd.to_datetime(df["datetime"], errors="coerce").dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if dts.empty:
        return {}
    max_ticks = max(2, int(max_ticks or 9))
    if len(dts) <= max_ticks:
        idxs = list(range(len(dts)))
    else:
        idxs = sorted(set(np.linspace(0, len(dts) - 1, max_ticks).round().astype(int).tolist()))
        idxs = sorted(set([0, len(dts) - 1] + idxs))

    span = pd.Timestamp(dts.iloc[-1]) - pd.Timestamp(dts.iloc[0])
    if span <= pd.Timedelta(days=3):
        fmt = "%d-%b-%Y<br>%H:%M"
    elif span <= pd.Timedelta(days=120):
        fmt = "%d-%b-%Y"
    else:
        fmt = "%b-%Y"

    tickvals = [pd.Timestamp(dts.iloc[i]).to_pydatetime() for i in idxs]
    ticktext = [pd.Timestamp(dts.iloc[i]).strftime(fmt.replace("<br>", "\n")).replace("\n", "<br>") for i in idxs]
    return {
        "type": "date",
        "tickmode": "array",
        "tickvals": tickvals,
        "ticktext": ticktext,
        "tickangle": -25 if len(tickvals) >= 7 else 0,
    }


def history_matplotlib_date_format(df: pd.DataFrame) -> str:
    """Match exported PNG/PDF date labels to the adaptive history axis."""
    if df is None or df.empty or "datetime" not in df.columns:
        return "%d-%b-%Y"
    dts = pd.to_datetime(df["datetime"], errors="coerce").dropna()
    if dts.empty:
        return "%d-%b-%Y"
    span = dts.max() - dts.min()
    if span <= pd.Timedelta(days=3):
        return "%d-%b-%Y\n%H:%M"
    if span <= pd.Timedelta(days=120):
        return "%d-%b-%Y"
    return "%b-%Y"


def short_test_label(value, max_len=34):
    s = str(value or "").strip()
    if not s:
        return ""
    s = s.replace("\\", "/").split("/")[-1]
    for ext in [".xlsx", ".xls", ".csv", ".pdf", ".docx", ".txt"]:
        if s.lower().endswith(ext):
            s = s[: -len(ext)]
    s = re.sub(r"\s+", " ", s).strip()
    return s[:max_len]


def clean_well_label(value):
    """Return well name only, without source/date/test suffixes."""
    s = str(value or "Unknown").strip()
    s = re.sub(r"\s+", " ", s)
    return s or "Unknown"


def well_title_text(value) -> str:
    """Return a readable chart title without duplicating the word Well."""
    label = clean_well_label(value)
    if re.match(r"(?i)^well(?:\s|[-_:]|$)", label):
        return label
    return f"Well {label}"


def automatic_chart_header(selected_wells, selected_tests=None) -> str:
    """Build a title that follows the current well/test selection."""
    tests = [str(x).strip() for x in (selected_tests or []) if str(x).strip()]
    wells = [clean_well_label(x) for x in (selected_wells or []) if str(x).strip()]
    if tests:
        if len(tests) == 1:
            return tests[0]
        shown = " vs ".join(tests[:3])
        return shown + (f" +{len(tests) - 3} more" if len(tests) > 3 else "")
    if wells:
        if len(wells) == 1:
            return well_title_text(wells[0])
        shown = " vs ".join(wells[:4])
        return "Well comparison: " + shown + (f" +{len(wells) - 4} more" if len(wells) > 4 else "")
    return "Well Production Test"


def compressed_time_mapping(datetimes: pd.Series, continuous_gap_hours: float = 2.0, compressed_gap_hours: float = 0.75):
    """Map real datetimes to a compressed numeric x-axis.

    Small gaps are kept with their real duration.  Long empty gaps are replaced
    by a short visual gap.  This keeps continuous WhatsApp readings after an
    Excel test on the same timeline while avoiding huge empty spaces between
    separate tests/days.
    """
    dts = pd.to_datetime(datetimes, errors="coerce").dropna().drop_duplicates().sort_values().reset_index(drop=True)
    if dts.empty:
        return {}, []

    continuous_gap_hours = max(float(continuous_gap_hours or 0), 0.0)
    compressed_gap_hours = max(float(compressed_gap_hours or 0.25), 0.10)

    mapping = {pd.Timestamp(dts.iloc[0]): 0.0}
    separators = []
    current_x = 0.0
    prev_dt = pd.Timestamp(dts.iloc[0])

    for dt_raw in dts.iloc[1:]:
        dt = pd.Timestamp(dt_raw)
        diff_h = max((dt - prev_dt).total_seconds() / 3600.0, 0.0)
        if diff_h > continuous_gap_hours:
            # Compress empty calendar time only. A long sampling interval does
            # not split a physical test and does not create a dashed boundary.
            current_x += compressed_gap_hours
        else:
            current_x += diff_h
        mapping[dt] = float(current_x)
        prev_dt = dt

    return mapping, separators


def add_plot_axis_columns(df, x_axis_mode, trace_grouping="Auto", continuous_gap_hours=2.0, compressed_gap_hours=0.75):
    if df.empty:
        return df
    out = df.copy()

    # Legend/curve grouping is by clean well name only. This prevents the same
    # well from appearing several times with date suffixes.
    if "well" in out.columns:
        out["series_group_key"] = out["well"].apply(clean_well_label)
    else:
        out["series_group_key"] = "All"
    out["series_label"] = out["series_group_key"]
    out["series_segment_id"] = 0
    out["plot_x"] = None
    out.attrs["compressed_separators"] = []

    if "datetime" in out.columns and out["datetime"].notna().any():
        for _, idx in out.groupby("series_group_key", dropna=False).groups.items():
            g0 = out.loc[idx].copy().sort_values("datetime")
            seg_id = 0
            prev_test = None
            for row_idx, dt in zip(g0.index, pd.to_datetime(g0["datetime"], errors="coerce")):
                current_test = str(g0.loc[row_idx, "test_id"]).strip() if "test_id" in g0.columns else ""
                if pd.isna(dt):
                    out.loc[row_idx, "series_segment_id"] = seg_id
                    continue
                # A curve is continuous for the complete detected test, regardless
                # of whether samples are 30 minutes, 24 hours, or several days
                # apart. Split only when test_id explicitly changes.
                changed_test = bool(current_test and prev_test and current_test != prev_test)
                if changed_test:
                    seg_id += 1
                out.loc[row_idx, "series_segment_id"] = seg_id
                prev_test = current_test or prev_test

    if x_axis_mode == "Real calendar time":
        if "datetime" in out.columns and out["datetime"].notna().any():
            out["plot_x"] = out["datetime"]
        else:
            out["plot_x"] = range(1, len(out) + 1)
        return out

    if is_aligned_elapsed_mode(x_axis_mode):
        # Each well starts at 0 elapsed hours. This is best for comparing two
        # different wells. Same-well uploads remain one curve per well.
        group_col = "series_group_key"
        for _, idx in out.groupby(group_col, dropna=False).groups.items():
            g = out.loc[idx].copy()
            if "datetime" in g.columns and g["datetime"].notna().any():
                g = g.sort_values("datetime")
                elapsed = (g["datetime"] - g["datetime"].min()).dt.total_seconds() / 3600.0
                out.loc[g.index, "plot_x"] = elapsed.astype(float)
            else:
                out.loc[g.index, "plot_x"] = pd.Series(range(0, len(g)), index=g.index, dtype=float)
        return out

    # Compressed real-date timeline. The mapping is global and only shortens
    # empty calendar gaps; it never splits readings that share one Test ID.
    if "datetime" in out.columns and out["datetime"].notna().any():
        mapping, separators = compressed_time_mapping(out["datetime"], continuous_gap_hours, compressed_gap_hours)
        out["plot_x"] = pd.to_datetime(out["datetime"], errors="coerce").map(lambda d: mapping.get(pd.Timestamp(d), np.nan) if pd.notna(d) else np.nan)
        out.attrs["compressed_separators"] = separators
    else:
        out["plot_x"] = range(1, len(out) + 1)
    return out


def x_axis_title_from_mode(x_axis_mode):
    if is_aligned_elapsed_mode(x_axis_mode):
        return "Elapsed test time from each selected test start (hours)"
    if is_compressed_real_date_mode(x_axis_mode):
        return "Compressed real-date timeline — empty gaps removed"
    return "Time"


def iter_plot_segments(g, feature=None):
    """Return continuous segments containing actual values for one signal.

    Combined WhatsApp ZIP data interleaves production rows, workbook rows and
    OCR rows. A NaN from another source must not break the line for the selected
    signal. Rows without that feature are removed before Plotly/Matplotlib sees
    the trace; real zero readings are preserved.
    """
    if g is None or g.empty:
        return []
    sort_col = "plot_x" if "plot_x" in g.columns else ("datetime" if "datetime" in g.columns else None)
    source_segments = (
        [seg for _, seg in g.groupby("series_segment_id", dropna=False, sort=True)]
        if "series_segment_id" in g.columns else [g]
    )
    segments = []
    for seg in source_segments:
        seg2 = seg.sort_values(sort_col, kind="stable") if sort_col else seg
        if feature is not None and feature in seg2.columns:
            valid = numeric_feature_series(seg2, feature).notna()
            seg2 = seg2.loc[valid]
        seg2 = seg2.reset_index(drop=True)
        if not seg2.empty:
            segments.append(seg2)
    return segments


def interval_levels(intervals):
    """Assign visual rows for interval notes.

    Longer/parent intervals are placed on the top row. Shorter intervals inside
    them are placed underneath, so a main operation from 12:00-18:00 can sit
    above two child steps from 12:00-15:00 and 15:00-18:00.
    """
    if not intervals:
        return []
    prepared = []
    for interval in intervals:
        x0 = interval.get("x0", 0)
        x1 = interval.get("x1", x0)
        try:
            duration = abs(float(x1) - float(x0))
        except Exception:
            try:
                duration = abs((x1 - x0).total_seconds())
            except Exception:
                duration = 0
        item = dict(interval)
        item["_duration"] = duration
        prepared.append(item)

    # Long intervals first, then early start. This gives parent intervals the top row.
    decorated = []
    active_ends = []
    for interval in sorted(prepared, key=lambda x: (-x.get("_duration", 0), x.get("x0", 0), x.get("x1", 0))):
        x0 = interval.get("x0", 0)
        x1 = interval.get("x1", x0)
        level = 0
        while level < len(active_ends) and x0 < active_ends[level]:
            level += 1
        if level == len(active_ends):
            active_ends.append(x1)
        else:
            active_ends[level] = max(active_ends[level], x1)
        interval["level"] = level
        decorated.append(interval)

    # Draw from top to bottom, left to right.
    decorated = sorted(decorated, key=lambda x: (x.get("level", 0), x.get("x0", 0), -x.get("_duration", 0)))
    for item in decorated:
        item.pop("_duration", None)
    return decorated


def note_target_matches_series(target, series_label):
    target = str(target or "All selected wells")
    series_label = str(series_label or "")
    if target == "All selected wells":
        return True
    return series_label == target or series_label.startswith(f"{target} ") or series_label.startswith(f"{target}(")


def convert_events_for_plot(manual_events, df, x_axis_mode):
    if not manual_events or df.empty or "series_label" not in df.columns:
        return []
    converted = []

    def _event_common(e):
        return {
            "x_shift_px": float(e.get("x_shift_px", 0) or 0),
            "y_level": str(e.get("y_level", "Auto") or "Auto"),
        }

    if x_axis_mode == "Real calendar time":
        for e in manual_events:
            target = e.get("target", "All selected wells")
            label = e["label"]
            item = {"plot_x": e["datetime"], "label": label, "target": target}
            item.update(_event_common(e))
            converted.append(item)
        return converted
    if "datetime" not in df.columns:
        return []
    for event in manual_events:
        event_dt = pd.Timestamp(event["datetime"])
        target = event.get("target", "All selected wells")
        for series_label, g in df.dropna(subset=["datetime"]).groupby("series_label", dropna=False):
            if not note_target_matches_series(target, series_label):
                continue
            g = g.sort_values("datetime").reset_index(drop=True)
            if g.empty or not (g["datetime"].min() <= event_dt <= g["datetime"].max()):
                continue
            nearest_i = int((g["datetime"] - event_dt).abs().idxmin())
            px = g.loc[nearest_i, "plot_x"] if "plot_x" in g.columns else nearest_i + 1
            label = event["label"]
            item = {"plot_x": px, "label": label, "target": target}
            item.update(_event_common(event))
            converted.append(item)
    return converted

def convert_intervals_for_plot(operation_intervals, df, x_axis_mode):
    if not operation_intervals or df.empty or "series_label" not in df.columns:
        return []
    converted = []
    if x_axis_mode == "Real calendar time":
        for i in operation_intervals:
            target = i.get("target", "All selected wells")
            label = i["label"]
            converted.append({"x0": pd.Timestamp(i["start"]), "x1": pd.Timestamp(i["end"]), "label": label, "target": target})
        return converted
    if "datetime" not in df.columns:
        return []
    multiple_series = df["series_label"].nunique() > 1
    for interval in operation_intervals:
        start_dt = pd.Timestamp(interval["start"])
        end_dt = pd.Timestamp(interval["end"])
        target = interval.get("target", "All selected wells")
        for series_label, g in df.dropna(subset=["datetime"]).groupby("series_label", dropna=False):
            if not note_target_matches_series(target, series_label):
                continue
            g = g.sort_values("datetime").reset_index(drop=True)
            if g.empty:
                continue
            overlap_start = max(start_dt, g["datetime"].min())
            overlap_end = min(end_dt, g["datetime"].max())
            if overlap_end <= overlap_start:
                continue
            i0 = int((g["datetime"] - overlap_start).abs().idxmin())
            i1 = int((g["datetime"] - overlap_end).abs().idxmin())
            x0 = g.loc[i0, "plot_x"] if "plot_x" in g.columns else i0 + 1
            x1 = g.loc[i1, "plot_x"] if "plot_x" in g.columns else i1 + 1
            if x1 <= x0:
                x1 = x0 + 1
            label = interval["label"]
            converted.append({"x0": x0, "x1": x1, "label": label, "target": target})
    return converted


# ---------------------------------------------------------------------------
# Display-unit conversion (v58)
# ---------------------------------------------------------------------------
units_sidebar_section = st.sidebar.expander("3. Units & Choke", expanded=False)
with units_sidebar_section:
    pressure_display_unit = st.selectbox(
        "Pressure unit", ["psi", "bar"], index=0, key="pressure_display_unit_v58",
        help="Changes plot/table display only. Uploaded source values remain unchanged.",
    )
    temperature_display_unit = st.selectbox(
        "Temperature unit", ["Keep detected unit", "°C", "°F"], index=0, key="temperature_display_unit_v58",
        help="Converts all detected temperature features consistently for display and export.",
    )

# Keep previously entered custom axis ranges physically consistent when units change.
_prev_pressure_unit = st.session_state.get("_prev_pressure_display_unit_v58", pressure_display_unit)
if _prev_pressure_unit != pressure_display_unit:
    _pressure_factor = 0.0689475729 if (_prev_pressure_unit == "psi" and pressure_display_unit == "bar") else (1.0 / 0.0689475729)
    for _pc in [c for c in data.columns if c in PRESSURE_COLUMNS_PSI or str(c).endswith("_psi")]:
        for _prefix in ["ymin_", "ymax_"]:
            _k = _prefix + feature_key_text(_pc)
            if _k in st.session_state:
                try:
                    st.session_state[_k] = float(st.session_state[_k]) * _pressure_factor
                except Exception:
                    pass
st.session_state["_prev_pressure_display_unit_v58"] = pressure_display_unit

_prev_temp_unit = st.session_state.get("_prev_temperature_display_unit_v58", temperature_display_unit)
if _prev_temp_unit != temperature_display_unit and _prev_temp_unit in {"°C", "°F"} and temperature_display_unit in {"°C", "°F"}:
    for _tc in [c for c in data.columns if str(c).endswith("_c") or str(c).endswith("_f")]:
        for _prefix in ["ymin_", "ymax_"]:
            _k = _prefix + feature_key_text(_tc)
            if _k in st.session_state:
                try:
                    _v = float(st.session_state[_k])
                    st.session_state[_k] = (_v * 9.0 / 5.0 + 32.0) if temperature_display_unit == "°F" else ((_v - 32.0) * 5.0 / 9.0)
                except Exception:
                    pass
st.session_state["_prev_temperature_display_unit_v58"] = temperature_display_unit

# Keep one canonical pre-display-unit view for portable PDF reopening. Exported
# PDF state must not store bar/°F display values under psi/°C canonical column
# names, otherwise a later import would convert the same values twice.
canonical_data_for_portable_v97 = data.copy(deep=False)
data = apply_display_unit_conversions(data, pressure_display_unit, temperature_display_unit)

# ---------------------------------------------------------------------------
# Unified choke conversion for plotting (v58)
# ---------------------------------------------------------------------------
# This is a configurable calibration, not a universal choke law.  The default
# requested by the user is 100% opening = 128/64 in.  The original percentage
# and /64 values stay in the dataframe for audit/review.
_has_choke_pct = "choke_pct" in data.columns and pd.to_numeric(data["choke_pct"], errors="coerce").notna().any()
_has_choke_size = "choke_size_64" in data.columns and pd.to_numeric(data["choke_size_64"], errors="coerce").notna().any()
_has_choke_ambiguous = "choke_ambiguous" in data.columns and pd.to_numeric(data["choke_ambiguous"], errors="coerce").notna().any()
choke_plot_mode = "Keep source units separate"
choke_full_open_64 = 128.0
show_raw_choke_features = False
ambiguous_choke_unit = "Auto from surrounding source units"
treat_zero_choke_as_missing = True

if _has_choke_pct or _has_choke_size or _has_choke_ambiguous:
    with units_sidebar_section:
        st.markdown("**Choke**")
        choke_plot_mode = st.selectbox(
            "Choke unit to plot",
            [
                "Opening (%) - combine both source units",
                "Size (/64 in) - combine both source units",
                "Keep source units separate",
            ],
            index=0,
            help=(
                "Creates one choke curve. Percentage rows and /64-inch rows are converted "
                "to the selected unit. Original uploaded columns remain unchanged."
            ),
            key="choke_plot_mode_v97",
        )
        choke_full_open_64 = st.number_input(
            "Full-open choke size (/64 in)",
            min_value=1.0,
            max_value=256.0,
            value=128.0,
            step=1.0,
            help="Calibration used for conversion. Default: 100% = 128/64 in; therefore 50% = 64/64 in.",
            key="choke_full_open_64_v97",
        )
        if _has_choke_ambiguous:
            ambiguous_choke_unit = st.selectbox(
                "When a choke value has no unit",
                [
                    "Auto from surrounding source units",
                    "Treat as Opening (%)",
                    "Treat as Size (/64 in)",
                ],
                index=0,
                help=(
                    "A plain entry such as 'Choke = 64' is impossible to identify from the number alone. "
                    "Choose its meaning here, or let Auto use explicit values in the same source file."
                ),
                key="ambiguous_choke_unit_v97",
            )
        treat_zero_choke_as_missing = st.checkbox(
            "Treat zero choke as blank/template value",
            value=True,
            key="treat_zero_choke_as_missing_v97",
            help=(
                "Recommended for TMU reports where unused choke cells are stored as 0. "
                "Turn this off only when 0 truly means the choke was fully closed."
            ),
        )

    _pct = pd.to_numeric(data.get("choke_pct", pd.Series(index=data.index, dtype=float)), errors="coerce")
    _size = pd.to_numeric(data.get("choke_size_64", pd.Series(index=data.index, dtype=float)), errors="coerce")
    _amb = pd.to_numeric(data.get("choke_ambiguous", pd.Series(index=data.index, dtype=float)), errors="coerce")

    # Interpret only unit-less entries. Explicit % and /64 values are never
    # reclassified. Fractional percentage entries such as 0.5 become 50%.
    _amb_as_pct = pd.Series(np.nan, index=data.index, dtype=float)
    _amb_as_size = pd.Series(np.nan, index=data.index, dtype=float)
    _auto_ambiguous_fallback_groups = 0
    if _amb.notna().any():
        if ambiguous_choke_unit.startswith("Treat as Opening"):
            _amb_as_pct = _amb.where(_amb > 1.0, _amb * 100.0)
        elif ambiguous_choke_unit.startswith("Treat as Size"):
            _amb_as_size = _amb.where(_amb >= 1.0, _amb * 100.0)
        else:
            group_cols = [c for c in ["source", "sheet"] if c in data.columns]
            if group_cols:
                grouped_indexes = data.groupby(group_cols, dropna=False, sort=False).groups.values()
            else:
                grouped_indexes = [data.index]
            for _idx in grouped_indexes:
                _idx = pd.Index(_idx)
                _a = _amb.loc[_idx]
                if not _a.notna().any():
                    continue
                _group_has_pct = _pct.loc[_idx].notna().any()
                _group_has_size = _size.loc[_idx].notna().any()
                if _group_has_size and not _group_has_pct:
                    _amb_as_size.loc[_idx] = _a.where(_a >= 1.0, _a * 100.0)
                elif _group_has_pct and not _group_has_size:
                    _amb_as_pct.loc[_idx] = _a.where(_a > 1.0, _a * 100.0)
                else:
                    # No decisive unit evidence (or both units occur): keep the
                    # historical field convention and treat bare <=100 values as %.
                    _amb_as_pct.loc[_idx] = _a.where(_a > 1.0, _a * 100.0)
                    _auto_ambiguous_fallback_groups += 1

    # Build the effective source series using plain float64 assignment instead
    # of combine_first.  This avoids nullable/Arrow dtype surprises on newer
    # pandas versions and guarantees that missing values never become numeric 0.
    def _as_float64_series(values):
        return pd.Series(pd.to_numeric(values, errors="coerce"), index=data.index, dtype="float64")

    _pct = _as_float64_series(_pct)
    _size = _as_float64_series(_size)
    _amb_as_pct = _as_float64_series(_amb_as_pct)
    _amb_as_size = _as_float64_series(_amb_as_size)

    _pct_effective = _pct.copy()
    _pct_fill_mask = _pct_effective.isna() & _amb_as_pct.notna()
    _pct_effective.loc[_pct_fill_mask] = _amb_as_pct.loc[_pct_fill_mask]

    _size_effective = _size.copy()
    _size_fill_mask = _size_effective.isna() & _amb_as_size.notna()
    _size_effective.loc[_size_fill_mask] = _amb_as_size.loc[_size_fill_mask]

    _converted_pct = (_size_effective / float(choke_full_open_64)) * 100.0
    _converted_size = (_pct_effective / 100.0) * float(choke_full_open_64)

    # Reject only impossible conversions above the user-defined full-open size;
    # keep the original source columns untouched so the user can review them.
    _oversize_mask = _size_effective.notna() & (_size_effective > float(choke_full_open_64))
    _converted_pct = _converted_pct.mask(_oversize_mask)

    # When both units exist at the same timestamp, the value already expressed
    # in the requested target unit wins. The converted other unit fills gaps.
    _conflict_mask = _pct_effective.notna() & _size_effective.notna()
    _pct_difference = (_pct_effective - ((_size_effective / float(choke_full_open_64)) * 100.0)).abs()
    _conflict_count = int((_conflict_mask & (_pct_difference > 2.0)).sum())

    if choke_plot_mode.startswith("Opening"):
        _target = _pct_effective.copy()
        _fill_mask = _target.isna() & _converted_pct.notna()
        _target.loc[_fill_mask] = _converted_pct.loc[_fill_mask]
        _target = _target.where((_target >= 0) & (_target <= 100))
        st.session_state["choke_unified_label"] = "Choke Opening (%)"
    elif choke_plot_mode.startswith("Size"):
        _target = _size_effective.copy()
        _fill_mask = _target.isna() & _converted_size.notna()
        _target.loc[_fill_mask] = _converted_size.loc[_fill_mask]
        _target = _target.where((_target >= 0) & (_target <= float(choke_full_open_64)))
        st.session_state["choke_unified_label"] = "Choke Size (/64 in)"
    else:
        _target = None
        data.drop(columns=["choke_unified"], inplace=True, errors="ignore")
        st.session_state["choke_unified_label"] = "Unified Choke"

    # TMU workbooks commonly store blank formula/template choke cells as zero.
    # By default, remove every zero before filling.  This is intentionally not
    # conditioned on flow columns, because some rows carry choke but no rate or
    # pressure values.  Users can turn the option off for a genuine shut-in.
    _choke_group_cols = [c for c in ["well", "source", "sheet"] if c in data.columns]
    if _target is not None:
        _target = pd.Series(_target, index=data.index, dtype="float64")
        if treat_zero_choke_as_missing:
            _target = _target.mask(_target.eq(0))
        if _choke_group_cols:
            _tmp = data[_choke_group_cols].copy()
            _tmp["__choke_target"] = _target
            _target = _tmp.groupby(_choke_group_cols, dropna=False, sort=False)["__choke_target"].transform(
                lambda x: pd.Series(x, dtype="float64").ffill().bfill()
            )
        if treat_zero_choke_as_missing:
            _target = pd.Series(_target, index=data.index, dtype="float64").mask(lambda x: x.eq(0))
        data["choke_unified"] = pd.Series(_target, index=data.index, dtype="float64")

    # Apply the same zero-as-blank rule to optional raw choke plots.  This keeps
    # an old saved raw-column selection from reintroducing false zero steps.
    if treat_zero_choke_as_missing:
        for _raw_choke_col in ["choke_pct", "choke_size_64", "choke_ambiguous"]:
            if _raw_choke_col not in data.columns:
                continue
            _raw_vals = pd.Series(pd.to_numeric(data[_raw_choke_col], errors="coerce"), index=data.index, dtype="float64").mask(lambda x: x.eq(0))
            if _choke_group_cols:
                _tmp = data[_choke_group_cols].copy()
                _tmp["__raw_choke"] = _raw_vals
                _raw_vals = _tmp.groupby(_choke_group_cols, dropna=False, sort=False)["__raw_choke"].transform(
                    lambda x: pd.Series(x, dtype="float64").ffill().bfill()
                )
            data[_raw_choke_col] = pd.Series(_raw_vals, index=data.index, dtype="float64").mask(lambda x: x.eq(0))


numeric_cols = available_numeric_columns(data)
show_raw_choke_features = choke_plot_mode.startswith("Keep")
if not show_raw_choke_features and "choke_unified" in numeric_cols:
    numeric_cols = [c for c in numeric_cols if c not in {"choke_pct", "choke_size_64", "choke_ambiguous"}]
if not numeric_cols:
    st.error("No numeric plotting columns were detected. Check the file headers or paste format.")
    st.stop()

# Small parser QA panel so users can immediately see whether uploaded columns were
# recognized correctly instead of seeing generic names such as Raw: Column / Raw: Psig.
with st.expander("Detected columns from uploaded files", expanded=False):
    detected_rows = []
    grouped = data.groupby([c for c in ["source", "sheet"] if c in data.columns], dropna=False)
    for group_key, g in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        meta = dict(zip([c for c in ["source", "sheet"] if c in data.columns], group_key))
        g_numeric = available_numeric_columns(g)
        dt_min = pd.to_datetime(g["datetime"], errors="coerce").min() if "datetime" in g.columns else pd.NaT
        dt_max = pd.to_datetime(g["datetime"], errors="coerce").max() if "datetime" in g.columns else pd.NaT
        detected_rows.append({
            "Source": meta.get("source", ""),
            "Sheet": meta.get("sheet", ""),
            "Rows": int(len(g)),
            "Start": dt_min.strftime("%Y-%m-%d %H:%M") if pd.notna(dt_min) else "",
            "End": dt_max.strftime("%Y-%m-%d %H:%M") if pd.notna(dt_max) else "",
            "Detected columns": ", ".join(column_label(c) for c in g_numeric),
            "Raw fallback columns": ", ".join(column_label(c) for c in g_numeric if str(c).startswith("raw__")),
        })
    st.dataframe(pd.DataFrame(detected_rows), width="stretch", height=180)
    if any(str(c).startswith("raw__") for c in numeric_cols):
        st.info(
            "Raw fallback columns mean the parser found a numeric time-series column but did not know its header alias yet. "
            "Use the Column mapping review panel above to map it once and save the alias for future uploads."
        )

# Sidebar filters
with st.sidebar.expander("4. Analysis View", expanded=True):
    analysis_view = st.radio(
        "Choose analysis view",
        ["Test detail", "Production history"],
        horizontal=True,
        key="analysis_view_v97",
        help=(
            "Test detail shows every reading inside a short test. Production history shows one stabilized "
            "value per test across months or years."
        ),
    )

    if analysis_view == "Production history":
        # Fixed engineering defaults keep this mode simple and fast:
        # one arithmetic-average point per detected test, connected chronologically.
        time_filter_mode = "All data"
        time_aggregation = "Raw data"
        x_axis_scale = "Auto history dates"
        x_axis_mode = "Real calendar time"
        continuous_gap_hours = 2.0
        compressed_gap_hours = 0.75
        x_axis_label_density = "Balanced"
        chart_view_mode = "Auto / desktop"
        trace_grouping = "Auto"
        st.caption("One point per test · average of all valid readings · one connected performance line")
    else:
        time_filter_mode = st.selectbox(
            "Time range control",
            ["Slider", "Manual calendar/time"],
            index=0,
            key="time_filter_mode_v97",
            help="Use Manual calendar/time for long tests where a slider is difficult.",
        )
        time_aggregation = interval_select_with_custom(
            "Average readings by time interval",
            ["Raw data", "5 minutes", "15 minutes", "30 minutes", "1 hour", "6 hours", "1 day", "1 month", "1 year"],
            index=0,
            key="time_aggregation_interval",
            placeholder="e.g. 2 hours",
            help_text="Use Raw data for normal tests. Choose or type an interval only when the chart is very dense.",
        )
        x_axis_scale = interval_select_with_custom(
            "X-axis tick scale",
            ["Auto readable", "30 minutes", "1 hour", "3 hours", "6 hours", "12 hours", "1 day", "1 month", "1 year"],
            index=0,
            key="x_axis_tick_interval",
            placeholder="e.g. 2 hours",
            help_text="Choose a preset or select Custom and type the spacing required between X-axis labels.",
        )
        x_axis_mode = st.selectbox(
            "X-axis display mode",
            ["Real calendar time", "Compressed real dates - remove empty gaps"],
            index=0,
            key="x_axis_mode_v97",
        )
        if is_compressed_real_date_mode(x_axis_mode):
            continuous_gap_hours = st.number_input(
                "Keep real spacing for gaps up to (hours)",
                min_value=0.0,
                max_value=24.0,
                value=2.0,
                step=0.5,
                key="continuous_gap_hours_v97",
            )
            compressed_gap_hours = st.number_input(
                "Visual gap shown after long gaps (hours)",
                min_value=0.1,
                max_value=12.0,
                value=0.75,
                step=0.25,
                key="compressed_gap_hours_v97",
            )
        else:
            continuous_gap_hours = 2.0
            compressed_gap_hours = 0.75
        x_axis_label_density = "Balanced"
        chart_view_mode = "Auto / desktop"
        trace_grouping = "Auto"


with st.sidebar.expander("5. Wells & Signals", expanded=True):

    # Prefer recent wells/tests with actual numeric readings, not alphabetical chat history.
    def _has_any_plot_numeric(_df):
        mask = pd.Series([False] * len(_df), index=_df.index)
        preferred = [
            "gross_rate_bpd", "oil_rate_stbd", "water_rate_bpd", "gas_rate_mmscfd",
            "whp_psi", "sep_p_psi", "pumping_pressure_psi", "bsw_pct",
            "ctu_reel_depth_ft",
        ]
        for _c in preferred:
            if _c in _df.columns:
                mask |= pd.to_numeric(_df[_c], errors="coerce").notna()
        return mask

    if "well" in data.columns:
        well_df = data.copy()
        useful_mask_for_wells = _has_any_plot_numeric(well_df)
        well_df = well_df[useful_mask_for_wells] if useful_mask_for_wells.any() else well_df
        well_df = well_df[well_df["well"].astype(str).str.strip().ne("") & well_df["well"].astype(str).str.lower().ne("unknown")]
        if "datetime" in well_df.columns and well_df["datetime"].notna().any():
            all_wells = well_df.groupby(well_df["well"].astype(str))["datetime"].max().sort_values(ascending=False).index.tolist()
        else:
            all_wells = sorted(well_df["well"].dropna().astype(str).unique())
    else:
        all_wells = []
    _saved_wells_v97 = st.session_state.get("selected_wells_v97", [])
    if isinstance(_saved_wells_v97, (list, tuple)):
        st.session_state["selected_wells_v97"] = [w for w in _saved_wells_v97 if w in all_wells]
    if not st.session_state.get("selected_wells_v97") and all_wells:
        st.session_state["selected_wells_v97"] = all_wells[:1]
    selected_wells = st.multiselect(
        "Choose wells", all_wells, default=all_wells[:1] if all_wells else [], key="selected_wells_v97"
    )

    # Test/period filtering removed from the sidebar.
    # Tests are still detected internally and shown in the data preview/export,
    # but the main workflow now filters by well and time range only.
    selected_tests = []
    all_tests = []

    select_all_features = st.checkbox(
        "Select all signals",
        value=False,
        key="select_all_features_v97",
        help="Shows every detected numeric column in the plot list, including raw fallback columns from unseen templates.",
    )

    default_features = [
        c for c in [
            "gross_rate_bpd", "qgross_s_bpd", "oil_rate_stbd", "qoil_s_stbd",
            "water_rate_bpd", "qwat_s_bpd", "gas_rate_mmscfd", "gas_formation_mmscfd",
            "pumping_pressure_psi", "whp_psi",
            "ctu_reel_depth_ft", "ctu_reel_speed_ftmin", "ctu_n2_rate_scfm",
            "bsw_pct", "wlr_s_pct", "whp_psi", "choke_unified",
            "flow_press_psi", "sep_p_psi", "salinity_kppm",
        ] if c in numeric_cols
    ] or numeric_cols[: min(6, len(numeric_cols))]
    default_features = list(dict.fromkeys(default_features))

    feature_state_key = "selected_features_v58"
    desired_default_features = numeric_cols if select_all_features else default_features
    select_all_prev_key = "select_all_features_prev_v58"
    if feature_state_key not in st.session_state:
        st.session_state[feature_state_key] = list(desired_default_features)
    elif select_all_features and not st.session_state.get(select_all_prev_key, False):
        st.session_state[feature_state_key] = list(numeric_cols)
    else:
        previous = list(st.session_state.get(feature_state_key, []))
        # Keep every still-valid selection. If the user previously selected a raw
        # choke feature and then enables unified choke, preserve intent by using
        # the stable choke_unified feature instead of resetting all defaults.
        had_choke = any(x in {"choke_pct", "choke_size_64", "choke_ambiguous", "choke_unified"} for x in previous)
        reconciled = [x for x in previous if x in numeric_cols]
        if had_choke and "choke_unified" in numeric_cols and "choke_unified" not in reconciled:
            reconciled.append("choke_unified")
        elif had_choke and "choke_unified" not in numeric_cols:
            for _raw_choke in ["choke_pct", "choke_size_64", "choke_ambiguous"]:
                if _raw_choke in numeric_cols and _raw_choke not in reconciled:
                    reconciled.append(_raw_choke)
        st.session_state[feature_state_key] = reconciled
    selected_feature_set = st.multiselect(
        "Signals to plot",
        numeric_cols,
        format_func=column_label,
        key=feature_state_key,
        help="Select or remove signals here, then drag the selected signals below to control graph order.",
    )
    selected_features = draggable_signal_order(selected_feature_set, key="plot_signal_order_v92")
    st.session_state[select_all_prev_key] = bool(select_all_features)


    auto_chart_header = automatic_chart_header(selected_wells, selected_tests)

    data_title_signature = "_".join([str(x) for x in sorted(data.get("source", pd.Series(dtype=str)).dropna().astype(str).unique())[:3]])[:80]
    chart_header_key = f"chart_header_{hashlib.sha1(data_title_signature.encode('utf-8')).hexdigest()[:10]}_{len(data)}"
    chart_header_selection_key = chart_header_key + "_selection"
    selection_signature = json.dumps(
        {"view": analysis_view, "wells": list(selected_wells), "tests": list(selected_tests)},
        sort_keys=True,
        ensure_ascii=False,
    )
    _portable_chart_title_v97 = str(st.session_state.get("portable_chart_title_v97", "") or "").strip()
    _portable_title_token_v97 = f"{chart_header_key}|{st.session_state.get('portable_pdf_signature_v97', '')}"
    if _portable_chart_title_v97 and st.session_state.get("_portable_title_applied_token_v97") != _portable_title_token_v97:
        st.session_state[chart_header_key] = _portable_chart_title_v97
        st.session_state[chart_header_selection_key] = selection_signature
        st.session_state["_portable_title_applied_token_v97"] = _portable_title_token_v97
    elif st.session_state.get(chart_header_selection_key) != selection_signature:
        st.session_state[chart_header_key] = auto_chart_header
        st.session_state[chart_header_selection_key] = selection_signature
    elif chart_header_key in st.session_state:
        old_header = str(st.session_state.get(chart_header_key, ""))
        st.session_state[chart_header_key] = re.sub(r"(?i)^well\s+well\s+", "Well ", old_header).strip()
    custom_chart_title = st.text_input(
        "Chart header / title",
        value=auto_chart_header,
        help="The title updates automatically when selected wells change. You can still edit it for the current selection.",
        key=chart_header_key,
    )

with st.sidebar.expander("6. Chart Options", expanded=False):
    if analysis_view == "Production history":
        # Keep only the two controls that materially improve a long-term history:
        # optional Y-axis limits and sparse value labels.
        custom_y_ranges = {}
        fill_method = "No fill"
        hide_zero_flow_rows = False
        plot_mode = "Separate panels like report"
        dual_axis_charts = []
        show_points = True

        with st.expander("Y-axis scale", expanded=False):
            use_custom_y_scale = st.checkbox(
                "Use custom Y-axis ranges",
                value=False,
                key="history_use_custom_y_scale",
            )
            if use_custom_y_scale and selected_features:
                # Calculate the small history preview only when the user opens
                # this optional control; normal history plotting avoids duplicate work.
                history_source_for_scale = data
                if selected_wells and "well" in history_source_for_scale.columns:
                    history_source_for_scale = history_source_for_scale[
                        history_source_for_scale["well"].astype(str).isin(selected_wells)
                    ]
                history_scale_preview = build_production_history(history_source_for_scale, selected_features)
                for feature in selected_features:
                    vals = (
                        numeric_feature_series(history_scale_preview, feature).dropna()
                        if feature in history_scale_preview.columns
                        else pd.Series(dtype="float64")
                    )
                    auto_range = default_y_axis_range(history_scale_preview, feature) or [0.0, 1.0]
                    default_min, default_max = float(auto_range[0]), float(auto_range[1])
                    st.markdown(f"**{column_label(feature)}**")
                    cy1, cy2 = st.columns(2)
                    with cy1:
                        y_min = st.number_input(
                            "Min",
                            value=float(round(default_min, 3)),
                            key=f"history_ymin_{feature_key_text(feature)}",
                        )
                    with cy2:
                        y_max = st.number_input(
                            "Max",
                            value=float(round(default_max, 3)),
                            key=f"history_ymax_{feature_key_text(feature)}",
                        )
                    if y_max > y_min:
                        custom_y_ranges[feature] = [float(y_min), float(y_max)]
                    else:
                        st.warning(f"Max must be greater than Min for {column_label(feature)}")

        _vl1, _vl2 = st.columns([1.45, 0.85], gap="small")
        with _vl1:
            value_label_mode = st.selectbox(
                "Value labels",
                [
                    "First, last + every N tests",
                    "First and last only",
                    "Clean readable - recommended",
                    "Off",
                ],
                index=0,
                help="Choose a simple label rule; N can be typed beside the list.",
                key="history_value_label_mode",
            )
        with _vl2:
            custom_value_label_step = int(st.number_input(
                "Every N tests",
                min_value=1,
                max_value=500,
                value=20,
                step=1,
                key="history_value_label_step",
                disabled=value_label_mode != "First, last + every N tests",
            ))
        label_decimals_default = "Auto"
        label_decimals_by_feature = {}
        note_color_theme = "Theme adaptive"
        show_internal_names = False
        st.caption("Each marker is the average of all valid readings in one test; markers are connected in date order.")
    else:
        custom_y_ranges = {}
        with st.expander("Y-axis scale per graph", expanded=False):
            use_custom_y_scale = st.checkbox(
                "Use custom Y-axis ranges",
                value=False,
                help="Set min/max for each selected graph, e.g. Gross Rate from 0 to 1000.",
                key="detail_use_custom_y_scale_v97",
            )

            if use_custom_y_scale and selected_features:
                for feature in selected_features:
                    vals = numeric_feature_series(data, feature).dropna() if feature in data.columns else pd.Series(dtype="float64")
                    auto_range = default_y_axis_range(data, feature) or [0.0, 1.0]
                    default_min, default_max = float(auto_range[0]), float(auto_range[1])

                    st.markdown(f"**{column_label(feature)}**")
                    cy1, cy2 = st.columns(2)
                    with cy1:
                        y_min = st.number_input(
                            "Min",
                            value=float(round(default_min, 3)),
                            key=f"ymin_{feature_key_text(feature)}",
                        )
                    with cy2:
                        y_max = st.number_input(
                            "Max",
                            value=float(round(default_max, 3)),
                            key=f"ymax_{feature_key_text(feature)}",
                        )

                    if y_max > y_min:
                        custom_y_ranges[feature] = [float(y_min), float(y_max)]
                    else:
                        st.warning(f"Max must be greater than Min for {column_label(feature)}")

        fill_method = st.selectbox(
            "Handle missing values",
            ["No fill", "Linear interpolation by row"],
            index=0,
            key="fill_method_v97",
            help="This only affects the plotted/filtered copy, not the originally detected data.",
        )

        hide_zero_flow_rows = st.checkbox(
            "Hide zero-flow/bypassed rows",
            value=False,
            key="hide_zero_flow_rows_v97",
            help="Useful for multiphase-meter reports during bypass periods where oil, water, gas, and gross are all zero.",
        )

        plot_mode = st.selectbox(
            "Plot style",
            ["Separate panels like report", "Overlay actual values"],
            index=0,
            key="plot_mode_v97",
            help="Use separate panels for normal reports. Use overlay to compare actual values on one axis.",
        )

        # Optional combined dual-axis charts.  These do not replace the normal
        # multi-panel report; they add extra comparison charts above it.
        dual_axis_charts = []
        if len(numeric_cols) >= 2 and selected_features:
            with st.expander("Combined charts with secondary Y-axis", expanded=False):
                n_dual_axis_charts = st.number_input(
                    "Number of combined secondary-axis charts",
                    min_value=0,
                    max_value=3,
                    value=0,
                    step=1,
                    help="Use 0 for no combined charts. Use 1-3 when you want several custom overlays.",
                )
                dual_defaults = selected_features if selected_features else numeric_cols[:2]
                for chart_i in range(int(n_dual_axis_charts)):
                    st.markdown(f"**Combined chart {chart_i + 1}**")
                    default_left = [dual_defaults[min(chart_i * 2, len(dual_defaults) - 1)]] if dual_defaults else [numeric_cols[0]]
                    default_right_seed = dual_defaults[min(chart_i * 2 + 1, len(dual_defaults) - 1)] if len(dual_defaults) > 1 else numeric_cols[min(1, len(numeric_cols) - 1)]
                    left_features_i = st.multiselect(
                        f"Chart {chart_i + 1} - left Y-axis feature(s)",
                        numeric_cols,
                        default=[f for f in default_left if f in numeric_cols],
                        format_func=column_label,
                        key=f"dual_left_features_{chart_i}",
                    )
                    right_features_i = st.multiselect(
                        f"Chart {chart_i + 1} - right Y-axis feature(s)",
                        numeric_cols,
                        default=[default_right_seed] if default_right_seed in numeric_cols else [],
                        format_func=column_label,
                        key=f"dual_right_features_{chart_i}",
                    )
                    chart_title_i = st.text_input(
                        f"Chart {chart_i + 1} title suffix",
                        value="",
                        placeholder="Optional, e.g. Rates vs Pumping Pressure",
                        key=f"dual_title_suffix_{chart_i}",
                    ).strip()
                    if left_features_i and right_features_i:
                        dual_axis_charts.append({
                            "left": left_features_i,
                            "right": right_features_i,
                            "title": chart_title_i or f"Combined chart {chart_i + 1}",
                        })

        # Keep markers off automatically on large datasets for speed/readability, but allow the user to turn them on.
        estimated_points_for_speed = int(len(data)) if "data" in globals() else 0
        default_markers = estimated_points_for_speed <= 350
        show_points = st.checkbox("Show markers", value=default_markers, key="show_points_v97")

        _vl1, _vl2 = st.columns([1.45, 0.85], gap="small")
        with _vl1:
            value_label_mode = st.selectbox(
                "Value labels on chart",
                [
                    "Clean readable - recommended",
                    "Every N readings",
                    "Hourly + min/max",
                    "All values - use wide export",
                    "First and last only",
                    "Off",
                ],
                index=0,
                help=(
                    "Clean readable keeps labels to important/non-crowded points. "
                    "Choose Every N readings and type the required spacing beside it."
                ),
                key="value_label_mode_v97",
            )
        with _vl2:
            custom_value_label_step = int(st.number_input(
                "Every N readings",
                min_value=1,
                max_value=10000,
                value=20,
                step=1,
                key="detail_value_label_step",
                disabled=value_label_mode != "Every N readings",
            ))

        label_decimals_default = st.selectbox(
            "Default number format on labels",
            ["Auto", "0 decimals", "1 decimal", "2 decimals"],
            index=0,
            key="label_decimals_default_v97",
        )
        label_decimals_by_feature = {}
        if selected_features:
            with st.expander("Number format per graph", expanded=False):
                for feature in selected_features:
                    label_decimals_by_feature[feature] = st.selectbox(
                        column_label(feature),
                        ["Use default", "Auto", "0 decimals", "1 decimal", "2 decimals"],
                        index=0,
                        key=f"label_decimals_{feature_key_text(feature)}",
                    )

        with st.expander("Rename column labels for view/export", expanded=False):
            current_overrides = dict(st.session_state.get("display_label_overrides", {}))
            edited_overrides = dict(current_overrides)
            for feature in selected_features:
                parser_label = _parser_column_label(feature)
                edited_overrides[str(feature)] = st.text_input(
                    parser_label,
                    value=str(current_overrides.get(str(feature), parser_label)),
                    key=f"display_label_{feature_key_text(feature)}",
                ).strip()
            c_save, c_clear = st.columns(2)
            with c_save:
                if st.button("Save label names"):
                    save_display_label_overrides(edited_overrides)
                    st.session_state["display_label_overrides"] = edited_overrides
                    st.success("Saved label names for future uploads.")
            with c_clear:
                if st.button("Reset saved label names"):
                    try:
                        DISPLAY_LABEL_FILE.unlink(missing_ok=True)
                    except Exception:
                        pass
                    st.session_state["display_label_overrides"] = {}
                    st.success("Saved label names cleared.")

        note_color_theme = "Theme adaptive"

        show_internal_names = False
with st.sidebar.expander("7. Events & Notes", expanded=False):

    auto_hide_crowded_notes = st.checkbox(
        "Auto hide some notes when too crowded",
        value=False,
        key="auto_hide_crowded_notes_v97",
        help=(
            "When enabled, the app keeps the most important/representative notes visible and hides extra notes "
            "only on the chart/export. The full note list remains saved in the sidebar."
        ),
    )

    max_visible_notes_per_chart = st.slider(
        "Maximum visible notes per chart",
        min_value=3,
        max_value=20,
        value=8,
        step=1,
        key="max_visible_notes_per_chart_v97",
        disabled=not auto_hide_crowded_notes,
        help="Used only when auto-hide is enabled.",
    )

    event_label_style = st.selectbox(
        "Event label layout",
        ["Auto staggered", "Vertical labels", "Compact top labels"],
        index=0,
        help=(
            "Notes stay inside the plot and remain tied to their exact times. "
            "Auto staggered keeps compact horizontal labels near the top, like version 92, and moves nearby labels to separate rows."
        ),
        key="event_label_layout",
    )
    enable_drag_annotations = st.checkbox(
        "Allow dragging event labels on the interactive chart",
        value=False,
        key="enable_drag_annotations_v97",
        help="Mouse drag is for on-screen adjustment only. Downloaded PNG/PDF charts use the clean automatic note layout and do not save dragged positions.",
    )


    if "manual_events_table" not in st.session_state:
        st.session_state.manual_events_table = []
    if "operation_intervals_table" not in st.session_state:
        st.session_state.operation_intervals_table = []

    default_event_dt = None
    if "datetime" in data.columns and data["datetime"].notna().any():
        default_event_dt = data["datetime"].dropna().min().to_pydatetime()

    note_label_input = st.text_input(
        "Operation note",
        placeholder="Shut-in pressure = 2000 psi, choke changed, start lifting...",
        key="operation_note_input",
    )

    note_target_options = ["All selected wells"] + (selected_wells if selected_wells else all_wells)
    note_target = st.selectbox(
        "Apply note to",
        note_target_options,
        index=0,
        help="Use this when different wells have different notes at the same time.",
        key="operation_note_target",
    )

    st.markdown("**Start**")
    n1, n2 = st.columns(2)
    with n1:
        note_start_date = st.date_input(
            "Start date",
            value=default_event_dt.date() if default_event_dt else None,
            min_value=MIN_DATE_ALLOWED,
            max_value=MAX_DATE_ALLOWED,
            key="note_start_date_input",
        )
    with n2:
        note_start_time_picker = scrollable_time_picker(
            "Start time",
            default_event_dt.time().replace(second=0, microsecond=0) if default_event_dt else None,
            key="note_start_time_picker",
            step_minutes=15,
        )

    note_start_time_text = st.text_input(
        "Or type start time",
        placeholder="09:30, 21:30, 0930, 9:30 PM",
        key="note_start_time_text",
    )

    add_end_time = st.checkbox(
        "Add end date/time to cover an interval",
        value=False,
        help="OFF = vertical event line. ON = shaded interval from start to end.",
    )

    note_end_date = None
    note_end_time_picker = None
    note_end_time_text = ""

    if add_end_time:
        st.markdown("**End**")
        e1, e2 = st.columns(2)
        with e1:
            note_end_date = st.date_input(
                "End date",
                value=default_event_dt.date() if default_event_dt else None,
                min_value=MIN_DATE_ALLOWED,
                max_value=MAX_DATE_ALLOWED,
                key="note_end_date_input",
            )
        with e2:
            note_end_time_picker = scrollable_time_picker(
                "End time",
                default_event_dt.time().replace(second=0, microsecond=0) if default_event_dt else None,
                key="note_end_time_picker",
                step_minutes=15,
            )
        note_end_time_text = st.text_input(
            "Or type end time",
            placeholder="10:30, 22:00, 1030, 10 PM",
            key="note_end_time_text",
        )

    if st.button("Add note to graph"):
        start_dt_note = combine_date_and_time(note_start_date, note_start_time_picker, note_start_time_text)
        if not start_dt_note or not note_label_input.strip():
            st.warning("Select start date/time and write a note first.")
        elif add_end_time:
            end_dt_note = combine_date_and_time(note_end_date, note_end_time_picker, note_end_time_text)
            if not end_dt_note:
                st.warning("Select end date/time or turn off interval mode.")
            elif end_dt_note <= start_dt_note:
                st.warning("End date/time must be after start date/time.")
            else:
                st.session_state.operation_intervals_table.append(
                    {"start": start_dt_note, "end": end_dt_note, "label": sanitize_share_text(note_label_input.strip()), "target": note_target}
                )
                st.success(f"Added interval: {start_dt_note:%Y-%m-%d %H:%M} to {end_dt_note:%Y-%m-%d %H:%M} | {sanitize_share_text(note_label_input.strip())}")
        else:
            st.session_state.manual_events_table.append(
                {
                    "datetime": start_dt_note,
                    "label": sanitize_share_text(note_label_input.strip()),
                    "target": note_target,
                }
            )
            st.success(f"Added event: {start_dt_note:%Y-%m-%d %H:%M} | {sanitize_share_text(note_label_input.strip())}")

    if st.session_state.manual_events_table:
        st.caption("Current point notes")
        events_df_sidebar = pd.DataFrame(st.session_state.manual_events_table)
        events_df_sidebar["datetime"] = pd.to_datetime(events_df_sidebar["datetime"])
        events_df_sidebar = events_df_sidebar.sort_values("datetime").reset_index(drop=True)
        # Note dragging is intentionally interactive-only. Do not keep manual
        # x/y note position columns because exported charts should use the clean
        # automatic layout, independent from any on-screen drag movement.
        events_df_sidebar = events_df_sidebar.drop(columns=["x_shift_px", "y_level"], errors="ignore")
        display_events = events_df_sidebar.copy()
        display_events["datetime"] = display_events["datetime"].dt.strftime("%Y-%m-%d %H:%M")
        display_events.insert(0, "No.", range(1, len(display_events) + 1))
        edited_events = st.data_editor(
            display_events,
            width="stretch",
            height=185,
            key="point_notes_editor_v50",
            column_config={
                "No.": st.column_config.NumberColumn("No.", disabled=True),
                "datetime": st.column_config.TextColumn("Date/time", disabled=True),
                "target": st.column_config.TextColumn("Target", disabled=True),
                "label": st.column_config.TextColumn("Label"),
            },
            disabled=["No.", "datetime", "target"],
        )
        # Save label edits immediately without changing the original datetime.
        # Position edits are not stored because mouse dragging is on-screen only.
        if len(edited_events) == len(events_df_sidebar):
            for _i in range(len(events_df_sidebar)):
                if "label" in edited_events.columns:
                    events_df_sidebar.at[_i, "label"] = edited_events.at[_i, "label"]
            st.session_state.manual_events_table = events_df_sidebar.to_dict("records")

        event_options = {
            f"{i + 1}) {row['datetime']:%Y-%m-%d %H:%M} | {row.get('target', 'All selected wells')} | {row['label']}": i
            for i, row in events_df_sidebar.iterrows()
        }
        selected_event_labels = st.multiselect(
            "Select point notes to remove",
            list(event_options.keys()),
            key="selected_point_notes_to_remove",
        )
        ce1, ce2 = st.columns(2)
        with ce1:
            if st.button("Remove selected point notes"):
                remove_idxs = {event_options[x] for x in selected_event_labels}
                if remove_idxs:
                    events_df_sidebar = events_df_sidebar.drop(index=list(remove_idxs)).reset_index(drop=True)
                    st.session_state.manual_events_table = events_df_sidebar.to_dict("records")
                    st.rerun()
        with ce2:
            if st.button("Remove all point notes"):
                st.session_state.manual_events_table = []
                st.rerun()

    if st.session_state.operation_intervals_table:
        st.caption("Current interval notes")
        intervals_df_sidebar = pd.DataFrame(st.session_state.operation_intervals_table)
        intervals_df_sidebar["start"] = pd.to_datetime(intervals_df_sidebar["start"])
        intervals_df_sidebar["end"] = pd.to_datetime(intervals_df_sidebar["end"])
        intervals_df_sidebar = intervals_df_sidebar.sort_values("start").reset_index(drop=True)
        display_intervals = intervals_df_sidebar.copy()
        display_intervals["start"] = display_intervals["start"].dt.strftime("%Y-%m-%d %H:%M")
        display_intervals["end"] = display_intervals["end"].dt.strftime("%Y-%m-%d %H:%M")
        display_intervals.insert(0, "No.", range(1, len(display_intervals) + 1))
        st.dataframe(display_intervals, width="stretch", height=150)

        interval_options = {
            f"{i + 1}) {row['start']:%Y-%m-%d %H:%M} → {row['end']:%Y-%m-%d %H:%M} | {row.get('target', 'All selected wells')} | {row['label']}": i
            for i, row in intervals_df_sidebar.iterrows()
        }
        selected_interval_labels = st.multiselect(
            "Select interval notes to remove",
            list(interval_options.keys()),
            key="selected_interval_notes_to_remove",
        )
        ci1, ci2 = st.columns(2)
        with ci1:
            if st.button("Remove selected interval notes"):
                remove_idxs = {interval_options[x] for x in selected_interval_labels}
                if remove_idxs:
                    intervals_df_sidebar = intervals_df_sidebar.drop(index=list(remove_idxs)).reset_index(drop=True)
                    st.session_state.operation_intervals_table = intervals_df_sidebar.to_dict("records")
                    st.rerun()
        with ci2:
            if st.button("Remove all interval notes"):
                st.session_state.operation_intervals_table = []
                st.rerun()

filtered = data.copy()
if "source_type" in filtered.columns:
    _plot_ocr_mask = filtered["source_type"].astype(str).str.contains("ocr", case=False, na=False)
    _plot_ocr_approved = filtered.get("ocr_approved", pd.Series(False, index=filtered.index)).fillna(False).astype(bool)
    filtered = filtered.loc[~_plot_ocr_mask | _plot_ocr_approved].copy()
if selected_wells:
    filtered = filtered[filtered["well"].astype(str).isin(selected_wells)]
if "test_id" in filtered.columns and selected_tests:
    filtered = filtered[filtered["test_id"].astype(str).isin(selected_tests)]

if analysis_view == "Production history":
    # Convert thousands of within-test readings into one average point per test.
    # This is substantially faster to render and is the correct view for multi-year performance.
    filtered = build_production_history(filtered, selected_features)

if hide_zero_flow_rows:
    # Prefer gross-rate columns when available, because some bypass periods still
    # carry constant pressure/choke/salinity values while the real production is zero.
    gross_like_cols = [c for c in ["gross_rate_bpd", "qgross_s_bpd"] if c in filtered.columns]
    if gross_like_cols:
        filtered = filtered[filtered[gross_like_cols].abs().sum(axis=1) > 0]
    else:
        flow_cols_for_zero_filter = [
            c for c in [
                "gross_rate_bpd", "qgross_s_bpd", "oil_rate_stbd", "qoil_s_stbd",
                "water_rate_bpd", "qwat_s_bpd", "gas_rate_mmscfd", "qgas_s_mmscfd",
                "gas_formation_mmscfd",
            ]
            if c in filtered.columns
        ]
        if flow_cols_for_zero_filter:
            filtered = filtered[filtered[flow_cols_for_zero_filter].abs().sum(axis=1) > 0]

# Time filter
start_dt = None
end_dt = None
if "datetime" in filtered.columns and filtered["datetime"].notna().any():
    min_dt = filtered["datetime"].min().to_pydatetime()
    max_dt = filtered["datetime"].max().to_pydatetime()
    start_dt, end_dt = min_dt, max_dt

    if min_dt < max_dt and time_filter_mode == "Slider":
        start_dt, end_dt = st.slider(
            "Time range",
            min_value=min_dt,
            max_value=max_dt,
            value=(min_dt, max_dt),
            format="DD/MM/YYYY HH:mm",
        )
    elif min_dt < max_dt and time_filter_mode == "Manual calendar/time":
        st.markdown("#### Manual time range")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            sd = st.date_input("Start date", min_dt.date(), min_value=MIN_DATE_ALLOWED, max_value=MAX_DATE_ALLOWED)
        with m2:
            stime_picker = st.time_input("Start time", min_dt.time().replace(second=0, microsecond=0))
            stime_text = st.text_input("Or type start time", placeholder="08:30, 0830, 8:30 PM")
        with m3:
            ed = st.date_input("End date", max_dt.date(), min_value=MIN_DATE_ALLOWED, max_value=MAX_DATE_ALLOWED)
        with m4:
            etime_picker = st.time_input("End time", max_dt.time().replace(second=0, microsecond=0))
            etime_text = st.text_input("Or type end time", placeholder="21:00, 2100, 9 PM")
        start_dt = combine_date_and_time(sd, stime_picker, stime_text)
        end_dt = combine_date_and_time(ed, etime_picker, etime_text)

    if start_dt is not None and end_dt is not None:
        filtered = filtered[
            (filtered["datetime"] >= pd.Timestamp(start_dt))
            & (filtered["datetime"] <= pd.Timestamp(end_dt))
        ]

# Optional aggregation/resampling applies only to the detailed within-test view.
if analysis_view == "Test detail":
    filtered = aggregate_time_data(filtered, time_aggregation)

if selected_features and analysis_view == "Test detail":
    filtered = apply_fill_method(filtered, selected_features, fill_method)

manual_events = []

# Add events created from the easy date/time + note UI.
for e in st.session_state.get("manual_events_table", []):
    try:
        manual_events.append({
            "datetime": pd.Timestamp(e["datetime"]),
            "label": sanitize_share_text(e["label"]),
            "target": sanitize_share_text(e.get("target", "All selected wells")),
        })
    except Exception:
        pass

# Prepare real-time / elapsed-time / sequence x-axis.
filtered = add_plot_axis_columns(filtered, x_axis_mode, trace_grouping, continuous_gap_hours, compressed_gap_hours)
plot_events = convert_events_for_plot(manual_events, filtered, x_axis_mode)

operation_intervals = []
for i in st.session_state.get("operation_intervals_table", []):
    try:
        operation_intervals.append(
            {"start": pd.Timestamp(i["start"]), "end": pd.Timestamp(i["end"]), "label": sanitize_share_text(i.get("label", "")), "target": sanitize_share_text(i.get("target", "All selected wells"))}
        )
    except Exception:
        pass

plot_intervals = convert_intervals_for_plot(operation_intervals, filtered, x_axis_mode)

# Engineering snapshot KPIs
render_section_title("Engineering Snapshot")
quality_count = int(_quality_mask.sum()) if "_quality_mask" in globals() else 0
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Detected readings", f"{len(data):,}")
c2.metric("Plotted tests" if analysis_view == "Production history" else "Active readings", f"{len(filtered):,}")
c3.metric("Wells", f"{data['well'].nunique() if 'well' in data.columns else 0:,}")
c4.metric("Test periods", f"{data['test_id'].nunique() if 'test_id' in data.columns else 0:,}")
c5.metric("Signals", f"{len(numeric_cols):,}")
c6.metric("Engineering checks", f"{quality_count:,}")

with st.expander("Detected data preview", expanded=False):
    preview_cols = ["source", "sheet", "source_type", "well", "test_id", "link_status", "datetime", "time_text"] + numeric_cols
    preview_cols = [c for c in preview_cols if c in data.columns]
    display_detected = data[preview_cols].copy(deep=False)
    display_detected, _preview_omitted_rows, _preview_omitted_cols = limited_dataframe_preview(
        display_detected, max_rows=500, max_cols=48
    )
    if not show_internal_names:
        display_detected = display_detected.rename(columns={c: column_label(c) for c in display_detected.columns})
    st.dataframe(display_detected, width="stretch", height=260)
    if _preview_omitted_rows or _preview_omitted_cols:
        st.caption(
            f"Preview limited to {len(display_detected):,} rows and {display_detected.shape[1]:,} columns to keep the app responsive."
        )

if selected_features and not filtered.empty:
    render_section_title("Production History" if analysis_view == "Production history" else "Production Test Visualization")

    # Protect the browser from very wide selections and very dense tests. The
    # complete filtered data remains available to exports; only the interactive
    # Plotly payload is reduced.
    max_interactive_features = 12
    interactive_features = list(selected_features[:max_interactive_features])
    interactive_filtered, interactive_was_reduced = optimize_interactive_plot_frame(
        filtered, interactive_features
    )
    if len(selected_features) > max_interactive_features:
        st.info(
            f"Interactive view shows the first {max_interactive_features} selected signals for stability. "
            "Prepared exports can still include all selected signals."
        )
    if interactive_was_reduced:
        st.info("Interactive chart optimized for speed; prepared exports still use all filtered readings.")
    if analysis_view == "Production history":
        st.caption("Each marker = average of all valid readings in one test · one line connects tests in chronological order")

    series_count_for_hint = interactive_filtered["series_label"].dropna().astype(str).nunique() if "series_label" in interactive_filtered.columns else 1
    if analysis_view == "Production history":
        axis_tick_settings = history_axis_tick_kwargs(filtered, max_ticks=9)
    elif x_axis_mode == "Real calendar time":
        axis_tick_settings = x_axis_tick_kwargs(x_axis_scale)
    elif is_aligned_elapsed_mode(x_axis_mode):
        axis_tick_settings = elapsed_axis_tick_kwargs(filtered, max_ticks=8 if chart_view_mode == "Mobile-friendly" else 12)
    elif is_compressed_real_date_mode(x_axis_mode):
        density_ticks = {"Sparse": 7, "Balanced": 11, "Detailed": 18}.get(x_axis_label_density, 11)
        if chart_view_mode == "Mobile-friendly":
            density_ticks = min(density_ticks, 7)
        elif chart_view_mode == "Wide report view":
            density_ticks = max(density_ticks, 14)
        if series_count_for_hint > 1:
            density_ticks = min(density_ticks, 8)
        axis_tick_settings = compressed_axis_tick_kwargs(
            filtered,
            scale=x_axis_scale,
            max_ticks_per_series=3,
            max_total_ticks=density_ticks,
        )
    else:
        axis_tick_settings = {}
    x_axis_title = x_axis_title_from_mode(x_axis_mode)

    NOTE_COLOR_PALETTES = {
        "Light": ["#A16207", "#007C91", "#C2410C", "#1D4ED8", "#15803D", "#7C3AED"],
        "Dark": ["#FFD166", "#66D9EF", "#FF8A72", "#8FE388", "#C9A7FF", "#F9A8D4"],
    }

    def note_palette():
        # Notes stay dark on a light report and switch to luminous colors on a
        # dark report so interval arrows, text, and borders remain readable.
        return NOTE_COLOR_PALETTES["Dark" if ACTIVE_THEME_NAME == "Dark" else "Light"]

    def note_color(idx):
        palette = note_palette()
        return palette[int(idx) % len(palette)]

    def adaptive_note_font_sizes(total_note_count, mobile=False):
        base_interval = 13 if not mobile else 11
        base_event = 13 if not mobile else 11
        if total_note_count >= 14:
            base_interval -= 3
            base_event -= 3
        elif total_note_count >= 9:
            base_interval -= 2
            base_event -= 2
        elif total_note_count >= 5:
            base_interval -= 1
            base_event -= 1
        return max(base_interval, 8), max(base_event, 8)

    def total_note_count():
        return len(plot_intervals or []) + len(plot_events or [])

    def _note_x_number(value):
        """Convert numeric/datetime plot positions to one collision-layout scale."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return np.nan
        try:
            ts = pd.Timestamp(value)
            if not pd.isna(ts) and not isinstance(value, (int, float, np.integer, np.floating)):
                return float(ts.value) / 1_000_000_000.0
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return np.nan

    def compact_note_label(label, *, max_chars=72, line_chars=26):
        """Return a safe, short multi-line label for charts.

        Full text remains in the sidebar table.  Chart labels are wrapped and,
        only when exceptionally long, shortened so they cannot cover a large
        part of the plot.
        """
        text = re.sub(r"\s+", " ", str(label or "")).strip()
        if len(text) > max_chars:
            text = text[: max(1, max_chars - 1)].rstrip() + "…"
        words = text.split()
        lines, current = [], ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= line_chars or not current:
                current = candidate
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return "<br>".join(html.escape(line) for line in lines[:3])

    def note_event_levels(events, x_values=None, max_levels=10):
        """Place note labels in non-overlapping rows using estimated text width.

        The earlier implementation used only the distance between event times.
        Long comments could therefore overlap even when their timestamps were
        sufficiently separated.  This version reserves an x-interval based on
        each label's length and chooses the first free row.
        """
        if not events:
            return []

        domain = []
        for event in events:
            n = _note_x_number(event.get("plot_x"))
            if pd.notna(n):
                domain.append(float(n))
        if x_values is not None:
            try:
                for value in x_values:
                    n = _note_x_number(value)
                    if pd.notna(n):
                        domain.append(float(n))
            except Exception:
                pass
        span = max(domain) - min(domain) if len(domain) >= 2 else 1.0
        if not np.isfinite(span) or span <= 0:
            span = 1.0

        occupied = [[] for _ in range(max_levels)]
        sortable = []
        for i, event in enumerate(events):
            sx = _note_x_number(event.get("plot_x"))
            if pd.isna(sx):
                sx = float(i)
            item = dict(event)
            label_len = max(6, min(72, len(re.sub(r"\s+", " ", str(item.get("label", ""))).strip())))
            # Approximate the horizontal chart space occupied by the text box.
            half_width = span * min(0.18, max(0.025, 0.012 + label_len * 0.0019))
            item["_left"] = float(sx) - half_width
            item["_right"] = float(sx) + half_width
            item["label_html"] = compact_note_label(item.get("label", ""))
            sortable.append((float(sx), i, item))
        sortable.sort(key=lambda entry: entry[0])

        decorated = []
        pad = span * 0.008
        for _, _, event in sortable:
            manual_level = str(event.get("y_level", "Auto") or "Auto")
            if manual_level != "Auto":
                try:
                    level = max(0, min(max_levels - 1, int(float(manual_level))))
                except Exception:
                    level = 0
            else:
                level = 0
                while level < max_levels:
                    overlaps = any(
                        not (event["_right"] + pad < left or event["_left"] - pad > right)
                        for left, right in occupied[level]
                    )
                    if not overlaps:
                        break
                    level += 1
                if level >= max_levels:
                    # Use the least crowded row rather than stacking everything
                    # on the final row.
                    level = min(range(max_levels), key=lambda idx: len(occupied[idx]))
            occupied[level].append((event["_left"], event["_right"]))
            event["level"] = level
            event.pop("_left", None)
            event.pop("_right", None)
            decorated.append(event)
        return decorated

    def _evenly_limit_items(items, max_items):
        if not items or not auto_hide_crowded_notes:
            return items
        max_items = max(0, int(max_items or 0))
        if max_items <= 0:
            return []
        if len(items) <= max_items:
            return items
        idxs = sorted(set(np.linspace(0, len(items) - 1, max_items).round().astype(int).tolist()))
        return [items[i] for i in idxs]

    def visible_intervals_for_notes():
        intervals = interval_levels(plot_intervals)
        if auto_hide_crowded_notes:
            # Long parent intervals are already first from interval_levels(), so this keeps main events first.
            intervals = intervals[: int(max_visible_notes_per_chart)]
        if not intervals:
            return []

        # Reuse the text-width collision engine on interval midpoints.  Preserve
        # nesting levels, but move adjacent long labels to another row when the
        # text boxes would collide.
        pseudo_events = []
        for idx, interval in enumerate(intervals):
            x0, x1 = interval.get("x0"), interval.get("x1")
            try:
                x_mid = x0 + (x1 - x0) / 2
            except Exception:
                x_mid = x0
            pseudo_events.append({
                "plot_x": x_mid,
                "label": interval.get("label", ""),
                "_interval_index": idx,
            })
        laid_out = note_event_levels(
            pseudo_events,
            x_values=(filtered["plot_x"] if "plot_x" in filtered.columns else None),
            max_levels=8,
        )
        auto_levels = {int(item.get("_interval_index", 0)): int(item.get("level", 0)) for item in laid_out}
        for idx, interval in enumerate(intervals):
            interval["level"] = max(int(interval.get("level", 0) or 0), auto_levels.get(idx, 0))
            interval["label_html"] = compact_note_label(interval.get("label", ""))
        return intervals

    def visible_events_for_notes(x_values=None):
        events = note_event_levels(plot_events, x_values=x_values, max_levels=10)
        if not auto_hide_crowded_notes:
            return events
        remaining = int(max_visible_notes_per_chart) - len(visible_intervals_for_notes())
        return _evenly_limit_items(events, max(0, remaining))

    def add_compressed_test_separators_to_plotly(fig, features):
        separators = chart_separator_positions(filtered)
        for sep_i, item in enumerate(separators):
            x_sep = item.get("x")
            if x_sep is None:
                continue
            for r in range(1, len(features) + 1):
                try:
                    fig.add_vline(
                        x=x_sep, line_width=2.4, line_dash="dash",
                        line_color="#334155", opacity=0.88, row=r, col=1,
                    )
                except Exception:
                    pass
        return fig

    def _inline_note_bg():
        return "rgba(255,255,255,0.97)" if ACTIVE_THEME_NAME == "Light" else "rgba(8,20,29,0.94)"

    def _inline_interval_y(level: int) -> float:
        return max(0.70, 0.965 - 0.085 * min(int(level or 0), 3))

    def _inline_event_y(level: int, vertical: bool, has_intervals: bool) -> float:
        if vertical:
            base = 0.84 if has_intervals else 0.92
            return max(0.22, base - 0.095 * min(int(level or 0), 6))
        base = 0.84 if has_intervals else 0.95
        return max(0.30, base - 0.075 * min(int(level or 0), 7))

    def add_operation_intervals_to_plotly(fig, features):
        """Draw interval boundaries and labels inside the plot, as in v92.

        The interval label remains visually attached to its start/end times. Two
        inward arrows form a clear <-> span while leaving the data region usable.
        """
        fig = add_compressed_test_separators_to_plotly(fig, features)
        intervals = visible_intervals_for_notes()
        if not intervals:
            return fig

        interval_font_size, _ = adaptive_note_font_sizes(
            total_note_count(), mobile=(chart_view_mode == "Mobile-friendly")
        )
        for idx, interval in enumerate(intervals):
            x0, x1 = interval["x0"], interval["x1"]
            level = int(interval.get("level", 0) or 0)
            note_col = note_color(idx)
            y_row = _inline_interval_y(level)
            label = str(interval.get("label_html") or compact_note_label(interval.get("label", "")))

            for x_val in (x0, x1):
                try:
                    fig.add_shape(
                        type="line", x0=x_val, x1=x_val, y0=0, y1=1,
                        xref="x", yref="paper",
                        line=dict(color=note_col, width=2.0, dash="dash"),
                        opacity=0.90, layer="above",
                    )
                except Exception:
                    pass

            try:
                x_mid = x0 + (x1 - x0) / 2
            except Exception:
                x_mid = x0

            # Two arrows, midpoint to each endpoint, create a true bidirectional span.
            for x_tip in (x0, x1):
                try:
                    fig.add_annotation(
                        x=x_tip, y=y_row, ax=x_mid, ay=y_row,
                        xref="x", yref="paper", axref="x", ayref="paper",
                        text="", showarrow=True, arrowhead=2, arrowsize=0.9,
                        arrowwidth=1.8, arrowcolor=note_col, opacity=0.95,
                    )
                except Exception:
                    pass
            try:
                fig.add_annotation(
                    x=x_mid, y=min(0.995, y_row + 0.018),
                    xref="x", yref="paper",
                    text=f"<b>{label}</b>", showarrow=False,
                    xanchor="center", yanchor="bottom",
                    bgcolor=_inline_note_bg(), bordercolor=note_col,
                    borderwidth=1.5, borderpad=3,
                    font=dict(size=interval_font_size, color=note_col),
                )
            except Exception:
                pass
        return fig

    def add_manual_events_to_plotly(fig, features):
        """Draw point events inside the plot and stagger them away from data labels."""
        fig = add_operation_intervals_to_plotly(fig, features)
        events = visible_events_for_notes(
            x_values=(filtered["plot_x"] if "plot_x" in filtered.columns else None)
        )
        if not events:
            return fig

        _, event_font_size = adaptive_note_font_sizes(
            total_note_count(), mobile=(chart_view_mode == "Mobile-friendly")
        )
        has_intervals = bool(visible_intervals_for_notes())
        vertical = event_label_style == "Vertical labels"

        for idx, event in enumerate(events):
            x = event["plot_x"]
            level = int(event.get("level", 0) or 0)
            note_col = note_color(idx + len(plot_intervals or []))
            label = str(event.get("label_html") or compact_note_label(event.get("label", "")))
            try:
                fig.add_shape(
                    type="line", x0=x, x1=x, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color=note_col, width=1.8, dash="dash"),
                    opacity=0.82, layer="above",
                )
            except Exception:
                pass

            y_note = _inline_event_y(level, vertical=vertical, has_intervals=has_intervals)
            text_angle = -90 if vertical else 0
            x_anchor = "right" if vertical else "center"
            y_anchor = "top" if vertical else "bottom"
            if event_label_style == "Compact top labels" and not vertical:
                y_note = max(0.35, y_note - 0.02)
            try:
                fig.add_annotation(
                    x=x, y=y_note, xref="x", yref="paper",
                    text=f"<b>{label}</b>", showarrow=False,
                    xanchor=x_anchor, yanchor=y_anchor, textangle=text_angle,
                    font=dict(size=event_font_size, color=note_col),
                    bgcolor=_inline_note_bg(), bordercolor=note_col,
                    borderwidth=1.4, borderpad=2, opacity=0.98,
                )
            except Exception:
                pass
        return fig

    def x_values(df):
        if x_axis_mode == "Real calendar time" and "datetime" in df.columns and df["datetime"].notna().any():
            return df["datetime"]
        if "plot_x" in df.columns and df["plot_x"].notna().any():
            return df["plot_x"]
        if "datetime" in df.columns and df["datetime"].notna().any():
            return df["datetime"]
        if "time_text" in df.columns:
            return df["time_text"]
        return df.index

    def max_points_per_trace(df):
        if "series_label" in df.columns:
            return int(df.groupby("series_label").size().max())
        if "well" in df.columns:
            return int(df.groupby("well").size().max())
        return len(df)

    def label_indices(n, mode):
        if mode == "Off" or n <= 0:
            return set()
        if mode == "All values - use wide export":
            return set(range(n))
        if mode in {"Every N readings", "First, last + every N tests"}:
            step = max(1, int(custom_value_label_step or 20))
            return set([0, n - 1] + list(range(0, n, step)))
        if mode == "First and last only":
            return {0, n - 1}

        # Clean/auto sparse: use a small number of evenly spaced labels.
        if n <= 10:
            step = 1
        elif n <= 30:
            step = 4
        elif n <= 80:
            step = 8
        elif n <= 180:
            step = 15
        else:
            step = max(20, round(n / 16))
        return set(list(range(0, n, step)) + [n - 1])

    def format_plot_value(feature, value):
        v = _plot_scalar_to_float(value)
        if pd.isna(v):
            return ""

        fmt_choice = label_decimals_by_feature.get(feature, "Use default")
        if fmt_choice == "Use default":
            fmt_choice = label_decimals_default

        if fmt_choice == "0 decimals":
            return f"{v:.0f}"
        if fmt_choice == "1 decimal":
            return f"{v:.1f}"
        if fmt_choice == "2 decimals":
            return f"{v:.2f}"

        # Auto compact format to prevent label overlap.
        if feature in ["bsw_pct", "co2_mole_pct"]:
            txt = f"{v:.1f}"
        elif feature in ["salinity_kppm", "choke_pct", "choke_size_64", "choke_ambiguous", "choke_unified", "whp_psi", "sep_p_psi", "pumping_pressure_psi"]:
            txt = f"{v:.0f}"
        elif abs(v) >= 100:
            txt = f"{v:.0f}"
        elif abs(v) >= 10:
            txt = f"{v:.1f}"
        else:
            txt = f"{v:.2f}"

        # Remove trailing zeroes for compactness.
        if "." in txt:
            txt = txt.rstrip("0").rstrip(".")
        return txt

    def report_label_indices(g, feature):
        """Readable labels for dense field reports.

        v49 policy: keep labels sparse enough to read.  It keeps first/last,
        min/max, a small number of evenly spaced points, and only the strongest
        local peaks/troughs.  Repeated zero labels are suppressed unless the
        series is very small.
        """
        g = g.reset_index(drop=True)
        n = len(g)
        if n == 0 or feature not in g.columns:
            return set()

        y = numeric_feature_series(g, feature, reset_index=True)
        valid = y.dropna()
        if valid.empty:
            return set()

        important = {0, n - 1, int(valid.idxmin()), int(valid.idxmax())}
        idxs = set(important)

        yrange = float(valid.max() - valid.min()) if valid.notna().any() else 0.0
        eps_zero = max(abs(float(valid.max())) * 1e-9, 1e-9)

        # Add only strong local peaks/troughs.  This avoids the dense blue-number
        # clutter seen when every small zig-zag is labeled.
        try:
            threshold = max(yrange * 0.22, abs(float(valid.mean())) * 0.12, 1.0)
            candidates = []
            for i in range(1, n - 1):
                if pd.isna(y.iloc[i - 1]) or pd.isna(y.iloc[i]) or pd.isna(y.iloc[i + 1]):
                    continue
                if abs(float(y.iloc[i])) <= eps_zero and n > 20:
                    continue
                is_peak = y.iloc[i] > y.iloc[i - 1] and y.iloc[i] > y.iloc[i + 1]
                is_trough = y.iloc[i] < y.iloc[i - 1] and y.iloc[i] < y.iloc[i + 1]
                amp = max(abs(float(y.iloc[i] - y.iloc[i - 1])), abs(float(y.iloc[i] - y.iloc[i + 1])))
                if (is_peak or is_trough) and amp >= threshold:
                    candidates.append((amp, i))
            # Keep only strongest candidates.
            candidates = sorted(candidates, reverse=True)[:6]
            idxs.update(i for _, i in candidates)
        except Exception:
            pass

        if value_label_mode == "Hourly + min/max" and "datetime" in g.columns and g["datetime"].notna().any():
            dt = pd.to_datetime(g["datetime"], errors="coerce")
            hourly = list(dt.reset_index(drop=True)[(dt.dt.minute == 0) & dt.notna()].index)
            if len(hourly) < 3:
                hourly = list(dt.reset_index(drop=True)[(dt.dt.minute.isin([0, 30])) & dt.notna()].index)
            idxs.update(hourly)
            max_labels = 16 if chart_view_mode == "Mobile-friendly" else 22
        else:
            # Clean readable: fewer labels, wider spacing.
            max_labels = 8 if chart_view_mode == "Mobile-friendly" else 12
            if n <= 18:
                spacing = 4
            elif n <= 60:
                spacing = 10
            elif n <= 160:
                spacing = 20
            else:
                spacing = max(25, int(round(n / 8)))
            idxs.update(range(0, n, spacing))

        # Avoid repeated zero labels in long charts.
        if n > 20:
            idxs = {i for i in idxs if i in important or (0 <= i < n and pd.notna(y.iloc[i]) and abs(float(y.iloc[i])) > eps_zero)}

        # Enforce minimum horizontal separation; important extrema are always kept.
        ordered = sorted(i for i in idxs if 0 <= i < n)
        protected = {i for i in important if 0 <= i < n}
        min_gap_idx = max(1, int(round(n / max(max_labels, 1))))
        kept = []
        for i in ordered:
            if i in protected:
                kept.append(i)
                continue
            if len(kept) >= max_labels:
                break
            if all(abs(i - j) >= min_gap_idx for j in kept if j not in protected):
                kept.append(i)
        idxs = set(kept) | protected

        # Hard cap.  Prefer first/last/min/max, then spread the remaining labels.
        if len(idxs) > max_labels:
            remaining = [i for i in sorted(idxs) if i not in protected]
            keep_n = max(0, max_labels - len(protected))
            if keep_n and remaining:
                pick = sorted(set(np.linspace(0, len(remaining) - 1, keep_n).round().astype(int).tolist()))
                idxs = protected | {remaining[i] for i in pick}
            else:
                idxs = protected

        return {i for i in idxs if 0 <= i < n}

    def _note_guard_positions():
        """Numeric X positions where event/interval guide lines cross the data area."""
        positions = []
        for event in plot_events or []:
            n = _note_x_number(event.get("plot_x"))
            if pd.notna(n):
                positions.append(float(n))
        for interval in plot_intervals or []:
            for key in ("x0", "x1"):
                n = _note_x_number(interval.get(key))
                if pd.notna(n):
                    positions.append(float(n))
        return positions

    def _value_label_context(g, feature):
        gx = x_values(g.reset_index(drop=True))
        xnums = pd.Series([_note_x_number(v) for v in gx], dtype="float64")
        yvals = numeric_feature_series(g, feature, reset_index=True)
        valid_x = xnums.dropna()
        valid_y = yvals.dropna()
        span_x = float(valid_x.max() - valid_x.min()) if not valid_x.empty else 1.0
        if not np.isfinite(span_x) or span_x <= 0:
            span_x = 1.0
        span_y = float(valid_y.max() - valid_y.min()) if not valid_y.empty else 1.0
        if not np.isfinite(span_y) or span_y <= 0:
            span_y = max(abs(float(valid_y.iloc[0])) if not valid_y.empty else 1.0, 1.0)
        return xnums, yvals, span_x, span_y

    def _important_value_label_indices(g, feature):
        n = len(g)
        if n <= 0:
            return set()
        y = numeric_feature_series(g, feature, reset_index=True)
        valid = y.dropna()
        important = {0, n - 1}
        if not valid.empty:
            important.update({int(valid.idxmin()), int(valid.idxmax())})
        return {i for i in important if 0 <= i < n}

    def _remove_value_labels_near_notes(g, idxs):
        """Suppress only non-essential labels that sit directly on note guide lines.

        First/last/min/max values remain available and are repositioned by the
        interactive text-position logic. This keeps exports readable without
        losing the most important engineering values.
        """
        idxs = set(idxs or set())
        guards = _note_guard_positions()
        if not idxs or not guards or g is None or g.empty:
            return idxs
        try:
            xnums, _, span_x, _ = _value_label_context(g, g.columns[-1] if len(g.columns) else "")
            important = set()
            # Preserve first/last by default; caller-specific extrema are normally
            # already included and are retained unless the chart is extremely dense.
            if len(g):
                important.update({0, len(g) - 1})
            exact_clearance = max(span_x * 0.006, 1e-9)
            cleaned = set()
            for i in idxs:
                if i < 0 or i >= len(xnums) or pd.isna(xnums.iloc[i]):
                    continue
                nearest = min(abs(float(xnums.iloc[i]) - guard) for guard in guards)
                if i in important or nearest > exact_clearance or len(idxs) <= 8:
                    cleaned.add(i)
            return cleaned
        except Exception:
            return idxs

    def build_text_and_positions(g, feature):
        if analysis_view == "Production history":
            if value_label_mode == "Clean readable - recommended":
                idxs = report_label_indices(g.reset_index(drop=True), feature)
            else:
                idxs = label_indices(len(g), value_label_mode)
        elif value_label_mode in ["Clean readable - recommended", "Hourly + min/max"]:
            idxs = report_label_indices(g.reset_index(drop=True), feature)
        else:
            idxs = label_indices(len(g), value_label_mode)

        values = numeric_feature_series(g, feature, reset_index=True)
        xnums, yvals, span_x, span_y = _value_label_context(g, feature)
        guards = _note_guard_positions()
        important = _important_value_label_indices(g, feature)
        clearance = max(span_x * 0.030, 1e-9)

        interval_bounds = []
        for interval in plot_intervals or []:
            x0 = _note_x_number(interval.get("x0"))
            x1 = _note_x_number(interval.get("x1"))
            if pd.notna(x0) and pd.notna(x1):
                interval_bounds.append((min(float(x0), float(x1)), max(float(x0), float(x1))))

        text, pos = [], []
        position_cycle = ["top center", "bottom center", "middle right", "middle left"]
        valid_y = yvals.dropna()
        ymin = float(valid_y.min()) if not valid_y.empty else 0.0

        for i, v in enumerate(values):
            show = i in idxs
            default_pos = position_cycle[i % len(position_cycle)]
            chosen_pos = default_pos
            if show and i < len(xnums) and pd.notna(xnums.iloc[i]):
                xv = float(xnums.iloc[i])
                nearest = min((abs(xv - guard) for guard in guards), default=float("inf"))
                near_guide = nearest <= clearance
                y_norm = 0.5
                if i < len(yvals) and pd.notna(yvals.iloc[i]):
                    y_norm = (float(yvals.iloc[i]) - ymin) / span_y
                inside_interval = any(lo - clearance <= xv <= hi + clearance for lo, hi in interval_bounds)

                # Keep first/last/min/max labels, but move them away from note boxes.
                if near_guide:
                    chosen_pos = "top right" if y_norm < 0.82 else "bottom right"
                    # In very dense cases, suppress non-essential labels exactly on a guide line.
                    if i not in important and nearest <= clearance * 0.22 and len(idxs) > 8:
                        show = False
                elif inside_interval and y_norm >= 0.78:
                    # The interval arrow sits near the top of the plot. Put high values below their marker.
                    chosen_pos = "bottom center"
                elif y_norm >= 0.92:
                    chosen_pos = "bottom center"
                elif y_norm <= 0.08:
                    chosen_pos = "top center"

            text.append(format_plot_value(feature, v) if show else "")
            pos.append(chosen_pos)
        return text, pos

    def padded_range(df, feature):
        if feature in custom_y_ranges:
            return custom_y_ranges[feature]
        return default_y_axis_range(df, feature)


    def build_figure(df, features, mode):
        series_values = sorted(df["series_label"].dropna().astype(str).unique()) if "series_label" in df.columns else (
            sorted(df["well"].dropna().astype(str).unique()) if "well" in df.columns else ["All"]
        )
        line_mode = "lines+markers" if show_points else "lines"
        hover_mode = "closest" if len(series_values) > 1 or len(df) > 3000 else "x unified"
        plot_uirevision = hashlib.sha1(
            f"{analysis_view}|{x_axis_mode}|{x_axis_scale}|{selection_signature}|{mode}".encode("utf-8")
        ).hexdigest()[:12]

        def merged_xaxis_kwargs(**overrides):
            merged = dict(axis_tick_settings or {})
            merged.update(overrides)
            return merged

        if mode == "Overlay two features with secondary Y-axis":
            dual_features = [f for f in features if f in df.columns][:2]
            if len(dual_features) < 2:
                return go.Figure()

            left_feature, right_feature = dual_features[0], dual_features[1]
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            for f_idx, feature in enumerate([left_feature, right_feature]):
                secondary = (f_idx == 1)
                for series_idx, series_label in enumerate(series_values):
                    g_all = df[df["series_label"].astype(str) == series_label].copy() if "series_label" in df.columns else df.copy()
                    if g_all.empty or feature not in g_all.columns:
                        continue
                    color = feature_color(feature, series_idx + f_idx)
                    first_segment = True
                    for g in iter_plot_segments(g_all, feature):
                        if g.empty:
                            continue
                        text, textposition = build_text_and_positions(g, feature)
                        trace_name = f"{series_label} - {column_label(feature)}" if len(series_values) > 1 else column_label(feature)
                        fig.add_trace(
                            go.Scatter(
                                x=x_values(g),
                                y=numeric_feature_series(g, feature),
                                mode=line_mode + ("+text" if value_label_mode != "Off" else ""),
                                text=text,
                                textposition=textposition,
                                textfont=dict(size=10 if chart_view_mode == "Mobile-friendly" else 12, color=color, family="Segoe UI, Arial, sans-serif"),
                                cliponaxis=False,
                                name=trace_name,
                                legendgroup=trace_name,
                                showlegend=first_segment,
                                line=dict(color=color, width=3.0, shape=("hv" if feature in {"choke_unified", "choke_pct", "choke_size_64", "choke_ambiguous"} else "linear")),
                                marker=dict(color=color, size=6 if chart_view_mode == "Mobile-friendly" else 8),
                                connectgaps=True,
                            ),
                            secondary_y=secondary,
                        )
                        first_segment = False

            fig.update_layout(
                height=850 if chart_view_mode != "Mobile-friendly" else 680,
                width=None,
                title=dict(text=chart_title_from_data(df, custom_chart_title), font=dict(size=28, color=CHART_TEXT, family="Segoe UI Semibold, Arial, sans-serif")),
                xaxis_title=x_axis_title,
                hovermode=hover_mode,
                uirevision=plot_uirevision,
                margin=dict(l=85, r=90, t=185, b=80),
                plot_bgcolor=CHART_PLOT_BG,
                paper_bgcolor=CHART_PAPER_BG,
                font=dict(color=CHART_TEXT, size=15),
                legend=dict(font=dict(size=15, color=CHART_TEXT), bgcolor=CHART_LEGEND_BG, bordercolor=CHART_GRID, borderwidth=1),
                title_x=0.5,
                title_xanchor="center",
            )
            fig.update_xaxes(**merged_xaxis_kwargs(showgrid=True, gridcolor=CHART_GRID, zeroline=False, tickfont=dict(size=14, color=CHART_TEXT)))
            fig.update_yaxes(
                title_text=column_label(left_feature),
                secondary_y=False,
                showgrid=True,
                gridcolor=CHART_GRID_SOFT,
                zeroline=False,
                range=custom_y_ranges.get(left_feature) or default_y_axis_range(df, left_feature),
            )
            fig.update_yaxes(
                title_text=column_label(right_feature),
                secondary_y=True,
                showgrid=False,
                zeroline=False,
                range=custom_y_ranges.get(right_feature) or default_y_axis_range(df, right_feature),
            )
            fig = add_manual_events_to_plotly(fig, [left_feature])
            return fig

        if mode == "Separate panels like report":
            show_chart_legend = len(series_values) > 1
            rows_count = len(features)
            if rows_count <= 1:
                vertical_gap = 0.03
            else:
                vertical_gap = min(0.045, 0.85 / max(rows_count - 1, 1))

            fig = make_subplots(
                rows=rows_count,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=vertical_gap,
            )

            for row_idx, feature in enumerate(features, start=1):
                feature_data_for_range = []

                for series_label in series_values:
                    g_all = df[df["series_label"].astype(str) == series_label].copy()
                    if g_all.empty or feature not in g_all.columns:
                        continue

                    feature_data_for_range.append(g_all[[feature]])
                    series_idx = series_values.index(series_label)
                    color = well_color(series_idx) if len(series_values) > 1 else feature_color(feature, series_idx)
                    first_segment = True
                    for g in iter_plot_segments(g_all, feature):
                        if g.empty:
                            continue
                        text, textposition = build_text_and_positions(g, feature)
                        fig.add_trace(
                            go.Scatter(
                                x=x_values(g),
                                y=numeric_feature_series(g, feature),
                                mode=line_mode + ("+text" if value_label_mode != "Off" else ""),
                                text=text,
                                textposition=textposition,
                                textfont=dict(size=10 if chart_view_mode == "Mobile-friendly" else 12, color=color, family="Segoe UI, Arial, sans-serif"),
                                cliponaxis=False,
                                name=f"{series_label}",
                                legendgroup=str(series_label),
                                showlegend=(show_chart_legend and row_idx == 1 and first_segment),
                                line=dict(color=color, width=3.0, shape=("hv" if feature in {"choke_unified", "choke_pct", "choke_size_64", "choke_ambiguous"} else "linear")),
                                marker=dict(color=color, size=6 if chart_view_mode == "Mobile-friendly" else 8),
                                connectgaps=True,
                            ),
                            row=row_idx,
                            col=1,
                        )
                        first_segment = False

                y_range = padded_range(pd.concat(feature_data_for_range), feature) if feature_data_for_range else None
                fig.update_yaxes(
                    title_text=column_label(feature),
                    row=row_idx,
                    col=1,
                    automargin=True,
                    range=y_range,
                )

            n_points = max_points_per_trace(df)
            mobile_view = chart_view_mode == "Mobile-friendly"
            wide_view = chart_view_mode == "Wide report view"
            panel_height = 330 if mobile_view else (440 if wide_view else 380)
            layout_kwargs = dict(
                height=max(620, panel_height * len(features)),
                title=dict(text=chart_title_from_data(df, custom_chart_title), font=dict(size=26 if mobile_view else 30, color=CHART_TEXT, family="Segoe UI Semibold, Arial, sans-serif")),
                hovermode=hover_mode,
                uirevision=plot_uirevision,
                margin=dict(l=85, r=50, t=185, b=80),
                uniformtext_minsize=8,
                uniformtext_mode="hide",
                plot_bgcolor=CHART_PLOT_BG,
                paper_bgcolor=CHART_PAPER_BG,
                font=dict(color=CHART_TEXT, size=15),
                legend=dict(
                    font=dict(size=17, color=CHART_TEXT),
                    bgcolor=CHART_LEGEND_BG,
                    bordercolor=CHART_GRID,
                    borderwidth=1.4,
                ),
                showlegend=show_chart_legend,
                title_x=0.5,
                title_xanchor="center",
            )
            if mobile_view and show_chart_legend:
                layout_kwargs["legend"] = dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0,
                    font=dict(size=12, color=CHART_TEXT),
                    bgcolor=CHART_LEGEND_BG,
                    bordercolor=CHART_GRID,
                    borderwidth=1.4,
                )
            fig.update_layout(**layout_kwargs)
            # Show readable time ticks on EVERY subplot, not only the bottom one.
            for r in range(1, len(features) + 1):
                fig.update_xaxes(
                    row=r,
                    col=1,
                    **merged_xaxis_kwargs(
                        showgrid=True,
                        gridcolor=CHART_GRID,
                        zeroline=False,
                        showticklabels=True,
                        tickfont=dict(size=11 if chart_view_mode == "Mobile-friendly" else 15, color=CHART_TEXT),
                        title_text=x_axis_title if r == len(features) else "",
                        title_font=dict(size=16 if chart_view_mode == "Mobile-friendly" else 20, color=CHART_TEXT),
                        automargin=True,
                    ),
                )
                fig.update_yaxes(
                    row=r,
                    col=1,
                    showgrid=True,
                    gridcolor=CHART_GRID_SOFT,
                    zeroline=False,
                    title_font=dict(size=14 if chart_view_mode == "Mobile-friendly" else 17, color=CHART_TEXT),
                    tickfont=dict(size=11 if chart_view_mode == "Mobile-friendly" else 14, color=CHART_TEXT),
                    automargin=True,
                )

            # Subplot titles are stored as annotations.
            for annotation in fig.layout.annotations:
                annotation.font = dict(size=21, color=CHART_TEXT)
            fig = add_manual_events_to_plotly(fig, features)
            return fig

        fig = go.Figure()
        for feature in features:
            for series_label in series_values:
                g_all = df[df["series_label"].astype(str) == series_label].copy()
                if g_all.empty or feature not in g_all.columns:
                    continue

                plot_name = f"{series_label} - {column_label(feature)}"
                y_title = "Actual values"
                series_idx = series_values.index(series_label)
                color = feature_color(feature, series_idx)
                first_segment = True
                for g in iter_plot_segments(g_all, feature):
                    y = g[feature].astype(float)
                    fig.add_trace(
                        go.Scatter(
                            x=x_values(g),
                            y=y,
                            mode=line_mode,
                            name=plot_name,
                            legendgroup=plot_name,
                            showlegend=first_segment,
                            line=dict(color=color, width=3.0, shape=("hv" if feature in {"choke_unified", "choke_pct", "choke_size_64", "choke_ambiguous"} else "linear")),
                            marker=dict(color=color, size=8),
                        )
                    )
                    first_segment = False

        fig.update_layout(
            height=850,
            width=None,
            title=dict(text=chart_title_from_data(df, custom_chart_title), font=dict(size=30, color=CHART_TEXT, family="Segoe UI Semibold, Arial, sans-serif")),
            yaxis_title=y_title,
            xaxis_title=x_axis_title,
            hovermode=hover_mode,
                uirevision=plot_uirevision,
            margin=dict(l=85, r=50, t=185, b=80),
            plot_bgcolor=CHART_PLOT_BG,
            paper_bgcolor=CHART_PAPER_BG,
            font=dict(color=CHART_TEXT, size=15),
            legend=dict(
                font=dict(size=17, color=CHART_TEXT),
                bgcolor=CHART_LEGEND_BG,
                bordercolor=CHART_GRID,
                borderwidth=1.4,
            ),
            title_x=0.5,
            title_xanchor="center",
        )
        fig.update_xaxes(**merged_xaxis_kwargs(
            showgrid=True,
            gridcolor=CHART_GRID,
            zeroline=False,
            title_font=dict(size=20, color=CHART_TEXT),
            tickfont=dict(size=15, color=CHART_TEXT),
        ))
        fig.update_yaxes(
            showgrid=True,
            gridcolor=CHART_GRID_SOFT,
            zeroline=False,
            title_font=dict(size=20, color=CHART_TEXT),
            tickfont=dict(size=15, color=CHART_TEXT),
            range=combined_default_y_axis_range(df, list(features)),
        )
        return add_manual_events_to_plotly(fig, features)

    def build_dual_axis_multi_figure(df, left_features, right_features, chart_name=""):
        """Build one combined chart with multiple left/right Y-axis features."""
        left_features = [f for f in left_features if f in df.columns]
        right_features = [f for f in right_features if f in df.columns]
        if not left_features or not right_features:
            return go.Figure()

        series_values = sorted(df["series_label"].dropna().astype(str).unique()) if "series_label" in df.columns else (
            sorted(df["well"].dropna().astype(str).unique()) if "well" in df.columns else ["All"]
        )
        hover_mode = "closest" if len(series_values) > 1 or len(df) > 3000 else "x unified"
        plot_uirevision = hashlib.sha1(
            f"{analysis_view}|{x_axis_mode}|{x_axis_scale}|{selection_signature}|dual|{chart_name}".encode("utf-8")
        ).hexdigest()[:12]

        def merged_xaxis_kwargs(**overrides):
            merged = dict(axis_tick_settings or {})
            merged.update(overrides)
            return merged

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        line_mode = "lines+markers" if show_points else "lines"
        plot_items = [(f, False) for f in left_features] + [(f, True) for f in right_features]

        for f_idx, (feature, secondary) in enumerate(plot_items):
            for series_idx, series_label in enumerate(series_values):
                g_all = df[df["series_label"].astype(str) == series_label].copy() if "series_label" in df.columns else df.copy()
                if g_all.empty or feature not in g_all.columns:
                    continue
                color = feature_color(feature, series_idx + f_idx)
                first_segment = True
                for g in iter_plot_segments(g_all, feature):
                    if g.empty:
                        continue
                    text, textposition = build_text_and_positions(g, feature)
                    trace_name = f"{series_label} - {column_label(feature)}" if len(series_values) > 1 else column_label(feature)
                    fig.add_trace(
                        go.Scatter(
                            x=x_values(g),
                            y=numeric_feature_series(g, feature),
                            mode=line_mode + ("+text" if value_label_mode != "Off" else ""),
                            text=text,
                            textposition=textposition,
                            textfont=dict(size=9 if chart_view_mode == "Mobile-friendly" else 11, color=color, family="Segoe UI, Arial, sans-serif"),
                            cliponaxis=False,
                            name=trace_name,
                            legendgroup=trace_name,
                            showlegend=first_segment,
                            line=dict(color=color, width=2.8, shape=("hv" if feature in {"choke_unified", "choke_pct", "choke_size_64", "choke_ambiguous"} else "linear")),
                            marker=dict(color=color, size=5 if chart_view_mode == "Mobile-friendly" else 7),
                        ),
                        secondary_y=secondary,
                    )
                    first_segment = False

        suffix = f" - {chart_name}" if chart_name else ""
        fig.update_layout(
            height=820 if chart_view_mode != "Mobile-friendly" else 640,
            width=None,
            title=dict(text=chart_title_from_data(df, custom_chart_title) + suffix, font=dict(size=26, color=CHART_TEXT, family="Segoe UI Semibold, Arial, sans-serif")),
            xaxis_title=x_axis_title,
            hovermode=hover_mode,
            uirevision=plot_uirevision,
            margin=dict(l=85, r=95, t=185, b=80),
            plot_bgcolor=CHART_PLOT_BG,
            paper_bgcolor=CHART_PAPER_BG,
            font=dict(color=CHART_TEXT, size=15),
            legend=dict(font=dict(size=13, color=CHART_TEXT), bgcolor=CHART_LEGEND_BG, bordercolor=CHART_GRID, borderwidth=1),
            title_x=0.5,
            title_xanchor="center",
        )
        fig.update_xaxes(**merged_xaxis_kwargs(showgrid=True, gridcolor=CHART_GRID, zeroline=False, tickfont=dict(size=13, color=CHART_TEXT)))
        fig.update_yaxes(
            title_text=" / ".join(column_label(f) for f in left_features[:3]),
            secondary_y=False,
            showgrid=True,
            gridcolor=CHART_GRID_SOFT,
            zeroline=False,
            range=combined_default_y_axis_range(df, left_features),
        )
        fig.update_yaxes(
            title_text=" / ".join(column_label(f) for f in right_features[:3]),
            secondary_y=True,
            showgrid=False,
            zeroline=False,
            range=combined_default_y_axis_range(df, right_features),
        )
        return add_manual_events_to_plotly(fig, [left_features[0]])

    plotly_config_common = {
        "responsive": True,
        "displaylogo": False,
        "editable": bool(enable_drag_annotations),
        "edits": {"annotationPosition": bool(enable_drag_annotations), "shapePosition": bool(enable_drag_annotations)},
        # Hide Plotly's browser camera button because it captures the current
        # dragged on-screen view. Use the app download buttons instead; they
        # generate clean export charts using automatic note layout.
        "modeBarButtonsToRemove": ["toImage"],
        "scrollZoom": True,
    }

    def render_plotly_safely(builder, *, key: str, fallback_feature: str | None = None):
        """Keep one chart error from terminating the full Streamlit session."""
        try:
            figure = builder()
            st.plotly_chart(figure, width="stretch", config=plotly_config_common, key=key)
            return figure
        except Exception:
            st.error("This chart could not be rendered with the current settings. A safe simplified view is shown below.")
            with st.expander("Chart technical details", expanded=False):
                st.code(traceback.format_exc())
            fallback = go.Figure()
            try:
                feature = fallback_feature or (interactive_features[0] if interactive_features else None)
                if feature and feature in interactive_filtered.columns:
                    fallback_df = interactive_filtered.head(3000)
                    fallback.add_trace(
                        go.Scatter(
                            x=x_values(fallback_df),
                            y=numeric_feature_series(fallback_df, feature),
                            mode="lines+markers",
                            name=column_label(feature),
                            line=dict(color=feature_color(feature, 0), width=2.4),
                            marker=dict(size=5),
                        )
                    )
                    fallback.update_layout(
                        title=chart_title_from_data(fallback_df, custom_chart_title),
                        xaxis_title=x_axis_title,
                        yaxis_title=column_label(feature),
                        paper_bgcolor=CHART_PAPER_BG,
                        plot_bgcolor=CHART_PLOT_BG,
                        font=dict(color=CHART_TEXT),
                    )
                    st.plotly_chart(
                        fallback, width="stretch",
                        config={"responsive": True, "displaylogo": False},
                        key=key + "_fallback",
                    )
            except Exception:
                pass
            return fallback

    if dual_axis_charts:
        for cfg_i, cfg in enumerate(dual_axis_charts, start=1):
            st.markdown(f"### Combined secondary Y-axis chart {cfg_i}")
            render_plotly_safely(
                lambda cfg=cfg: build_dual_axis_multi_figure(
                    interactive_filtered, cfg.get("left", []), cfg.get("right", []), cfg.get("title", "")
                ),
                key=f"dual_axis_chart_{cfg_i}",
                fallback_feature=(cfg.get("left", []) or interactive_features or [None])[0],
            )

    fig = render_plotly_safely(
        lambda: build_figure(interactive_filtered, interactive_features, plot_mode),
        key="main_production_test_chart",
        fallback_feature=interactive_features[0] if interactive_features else None,
    )

    with st.expander("Filtered data used by current plot", expanded=False):
        filtered_cols = ["source", "sheet", "well", "datetime", "time_text"] + selected_features
        filtered_cols = [c for c in filtered_cols if c in filtered.columns]
        display_filtered = filtered[filtered_cols].copy(deep=False)
        display_filtered, _filtered_omitted_rows, _filtered_omitted_cols = limited_dataframe_preview(
            display_filtered, max_rows=1000, max_cols=48
        )
        if not show_internal_names:
            display_filtered = display_filtered.rename(columns={c: column_label(c) for c in display_filtered.columns})
        st.dataframe(display_filtered, width="stretch", height=280)
        if _filtered_omitted_rows or _filtered_omitted_cols:
            st.caption("Preview limited for browser stability; chart exports still use all filtered readings.")

    render_section_title("Engineering Report Exports")
    st.caption(f"Downloads use the active {ACTIVE_THEME_NAME} theme.")

    def chart_label_indices_for_export(g, feature):
        """Use the same readable label logic for exports as the interactive chart."""
        g2 = g.reset_index(drop=True)
        n = len(g2)
        if analysis_view == "Production history":
            if value_label_mode == "Clean readable - recommended":
                return report_label_indices(g2, feature)
            return label_indices(n, value_label_mode)
        if value_label_mode in ["Clean readable - recommended", "Hourly + min/max"]:
            return report_label_indices(g2, feature)
        idxs = label_indices(n, value_label_mode)
        # Even when user asks every 8/20, keep exports readable on very dense curves.
        max_labels = 26 if chart_view_mode != "Mobile-friendly" else 18
        if value_label_mode != "All values - use wide export" and len(idxs) > max_labels:
            keep = sorted(idxs)
            chosen = [keep[i] for i in sorted(set(np.linspace(0, len(keep) - 1, max_labels).round().astype(int).tolist()))]
            idxs = set(chosen)
            if n:
                idxs.update({0, n - 1})
        idxs = _remove_value_labels_near_notes(g2, idxs)
        return {i for i in idxs if 0 <= i < n}

    def matplotlib_value_label_placement(g, feature, index):
        """Return an export label offset that respects inline event geometry."""
        default = (0, 11 if int(index) % 2 == 0 else -16, "center")
        try:
            g2 = g.reset_index(drop=True)
            xnums, yvals, span_x, span_y = _value_label_context(g2, feature)
            if index < 0 or index >= len(xnums) or pd.isna(xnums.iloc[index]):
                return default
            xv = float(xnums.iloc[index])
            guards = _note_guard_positions()
            nearest = min((abs(xv - guard) for guard in guards), default=float("inf"))
            clearance = max(span_x * 0.030, 1e-9)
            valid_y = yvals.dropna()
            ymin = float(valid_y.min()) if not valid_y.empty else 0.0
            y_norm = 0.5
            if index < len(yvals) and pd.notna(yvals.iloc[index]):
                y_norm = (float(yvals.iloc[index]) - ymin) / span_y

            interval_bounds = []
            for interval in plot_intervals or []:
                x0 = _note_x_number(interval.get("x0"))
                x1 = _note_x_number(interval.get("x1"))
                if pd.notna(x0) and pd.notna(x1):
                    interval_bounds.append((min(float(x0), float(x1)), max(float(x0), float(x1))))
            inside_interval = any(lo - clearance <= xv <= hi + clearance for lo, hi in interval_bounds)

            if nearest <= clearance:
                return (14, -16 if y_norm >= 0.82 else 12, "left")
            if inside_interval and y_norm >= 0.76:
                return (0, -17, "center")
            if y_norm >= 0.92:
                return (0, -17, "center")
            if y_norm <= 0.08:
                return (0, 12, "center")
            return default
        except Exception:
            return default

    def apply_matplotlib_petro_style(fig_obj, ax_obj=None):
        """Apply the active engineering theme to PNG/PDF exports."""
        fig_obj.patch.set_facecolor(CHART_PAPER_BG)
        if ax_obj is None:
            axes = list(getattr(fig_obj, "axes", []))
        elif isinstance(ax_obj, (list, tuple, np.ndarray)):
            axes = list(np.asarray(ax_obj).ravel())
        else:
            axes = [ax_obj]
        for axis in axes:
            try:
                axis.set_facecolor(CHART_PLOT_BG)
                axis.grid(True, color=CHART_GRID, linewidth=0.75, alpha=0.75)
                axis.tick_params(axis="both", colors=CHART_TEXT, labelcolor=CHART_TEXT)
                axis.xaxis.label.set_color(CHART_TEXT)
                axis.yaxis.label.set_color(CHART_TEXT)
                axis.title.set_color(CHART_TEXT)
                for spine in axis.spines.values():
                    spine.set_color(CHART_GRID)
                legend = axis.get_legend()
                if legend is not None:
                    legend.get_frame().set_facecolor(CHART_PAPER_BG)
                    legend.get_frame().set_edgecolor(CHART_GRID)
                    for legend_text in legend.get_texts():
                        legend_text.set_color(CHART_TEXT)
            except Exception:
                pass
        for figure_text in getattr(fig_obj, "texts", []):
            try:
                figure_text.set_color(CHART_TEXT)
            except Exception:
                pass

    def portable_ui_state_snapshot(features):
        """Capture only chart-related controls needed to reopen the same analysis."""
        snapshot = {}
        for key in PORTABLE_SESSION_KEYS:
            if key in st.session_state:
                snapshot[key] = _portable_json_value(st.session_state.get(key))
        for key in list(st.session_state.keys()):
            if str(key).startswith(PORTABLE_DYNAMIC_PREFIXES):
                snapshot[str(key)] = _portable_json_value(st.session_state.get(key))
        snapshot.update({
            "ui_theme": ACTIVE_THEME_NAME,
            "selected_features_v58": list(features or []),
            "plot_signal_order_v92_state": list(features or []),
            "selected_wells_v97": list(selected_wells or []),
            "analysis_view_v97": analysis_view,
            "event_label_layout": event_label_style,
            "pressure_display_unit_v58": pressure_display_unit,
            "temperature_display_unit_v58": temperature_display_unit,
        })
        return snapshot

    def make_portable_pdf(pdf_bytes, df, features):
        portable_frame = canonical_data_for_portable_v97.copy(deep=False)
        if selected_wells and "well" in portable_frame.columns:
            portable_frame = portable_frame[portable_frame["well"].astype(str).isin(selected_wells)].copy(deep=False)
        state_zip = build_portable_state_zip(
            portable_frame,
            ui_state=portable_ui_state_snapshot(features),
            chart_title=chart_title_from_data(df, custom_chart_title),
            manual_events=list(st.session_state.get("manual_events_table", []) or []),
            operation_intervals=list(st.session_state.get("operation_intervals_table", []) or []),
            custom_y_ranges=custom_y_ranges,
        )
        return attach_portable_state_to_pdf(pdf_bytes, state_zip)

    def human_readable_pdf_bytes(df, features):
        """Create a multi-page PDF: one large chart per feature, suitable for human reading/printing."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.backends.backend_pdf import PdfPages

        output = io.BytesIO()
        series_values = sorted(df["series_label"].dropna().astype(str).unique()) if "series_label" in df.columns else (
            sorted(df["well"].dropna().astype(str).unique()) if "well" in df.columns else ["All"]
        )

        with PdfPages(output) as pdf:
            for feature in features:
                fig_m, ax = plt.subplots(figsize=(18.5, 11.0), dpi=180)
                apply_matplotlib_petro_style(fig_m, ax)
                title = f"{chart_title_from_data(df, custom_chart_title)}\n{column_label(feature)}"
                ax.set_title(title, fontsize=22, fontweight="bold", pad=18)

                for wi, series_label in enumerate(series_values):
                    g_all = df[df["series_label"].astype(str) == series_label].copy()
                    if g_all.empty or feature not in g_all.columns:
                        continue
                    color = feature_color(feature, wi)
                    first_segment = True
                    for g in iter_plot_segments(g_all, feature):
                        x = g["plot_x"] if "plot_x" in g.columns and g["plot_x"].notna().any() else (
                            pd.to_datetime(g["datetime"], errors="coerce") if "datetime" in g.columns and g["datetime"].notna().any() else pd.Series(range(len(g)))
                        )
                        y = numeric_feature_series(g, feature)
                        ax.plot(
                            x,
                            y,
                            marker="o",
                            markersize=4.6,
                            linewidth=2.2,
                            color=color,
                            drawstyle="steps-post" if feature in {"choke_unified", "choke_pct", "choke_size_64", "choke_ambiguous"} else "default",
                            label=series_label if (len(series_values) > 1 and first_segment) else None,
                        )

                        idxs = chart_label_indices_for_export(g, feature)
                        if value_label_mode in ["Hourly + min/max", "Clean readable - recommended"]:
                            max_lbl = 16 if chart_view_mode == "Mobile-friendly" else 24
                            if len(idxs) > max_lbl:
                                keep = sorted(idxs)
                                step = max(1, len(keep) // max_lbl)
                                idxs = set(keep[::step])
                                idxs.update({0, len(g) - 1})
                        for i in sorted(idxs):
                            if i >= len(g) or pd.isna(y.iloc[i]):
                                continue
                            _label_dx, _label_dy, _label_ha = matplotlib_value_label_placement(g, feature, i)
                            ax.annotate(
                                format_plot_value(feature, y.iloc[i]),
                                (x.iloc[i], y.iloc[i]),
                                textcoords="offset points",
                                xytext=(_label_dx, _label_dy),
                                ha=_label_ha,
                                fontsize=9.5,
                                color=color,
                                fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.12", fc=EXPORT_LABEL_BG, ec=EXPORT_LABEL_EDGE, alpha=0.65),
                            )
                        first_segment = False

                y_limits = custom_y_ranges.get(feature) or default_y_axis_range(df, feature)
                if y_limits:
                    ax.set_ylim(y_limits[0], y_limits[1])

                ax.set_ylabel(column_label(feature), fontsize=15, fontweight="bold")
                ax.set_xlabel(x_axis_title, fontsize=15, fontweight="bold")

                try:
                    x_for_notes = ax.lines[0].get_xdata() if ax.lines else None
                except Exception:
                    x_for_notes = None
                _apply_matplotlib_notes(ax, x_values=x_for_notes)
                ax.grid(True, which="major", alpha=0.28)
                ax.tick_params(axis="both", labelsize=12)

                if x_axis_mode == "Real calendar time" and "datetime" in df.columns and df["datetime"].notna().any():
                    ax.xaxis.set_major_formatter(mdates.DateFormatter(history_matplotlib_date_format(df) if analysis_view == "Production history" else "%d-%b\n%H:%M"))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3 if analysis_view == "Production history" else 8, maxticks=9 if analysis_view == "Production history" else 16))
                    fig_m.autofmt_xdate(rotation=0)
                elif is_aligned_elapsed_mode(x_axis_mode):
                    tick_settings = elapsed_axis_tick_kwargs(df)
                    if tick_settings:
                        ax.set_xticks(tick_settings.get("tickvals", []))
                        ax.set_xticklabels(tick_settings.get("ticktext", []), rotation=0)
                elif is_compressed_real_date_mode(x_axis_mode):
                    density_ticks = {"Sparse": 7, "Balanced": 11, "Detailed": 18}.get(x_axis_label_density, 11)
                    if chart_view_mode == "Mobile-friendly":
                        density_ticks = min(density_ticks, 7)
                    elif chart_view_mode == "Wide report view":
                        density_ticks = max(density_ticks, 14)
                    tick_settings = compressed_axis_tick_kwargs(df, max_ticks_per_series=3, max_total_ticks=density_ticks)
                    if tick_settings:
                        ax.set_xticks(tick_settings.get("tickvals", []))
                        ax.set_xticklabels([str(t).replace("<br>", "\n") for t in tick_settings.get("ticktext", [])], rotation=30, ha="right")

                if len(series_values) > 1:
                    ax.legend(fontsize=12, loc="best")

                apply_matplotlib_petro_style(fig_m, ax)
                fig_m.tight_layout(rect=[0.02, 0.02, 0.98, _matplotlib_note_top_limit()])
                pdf.savefig(fig_m, facecolor=CHART_PAPER_BG, edgecolor=CHART_PAPER_BG)
                plt.close(fig_m)

        output.seek(0)
        return make_portable_pdf(output.getvalue(), df, features)

    def human_readable_png_zip_bytes(df, features):
        """Create a ZIP containing one large PNG per selected feature."""
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        zip_buffer = io.BytesIO()
        series_values = sorted(df["series_label"].dropna().astype(str).unique()) if "series_label" in df.columns else (
            sorted(df["well"].dropna().astype(str).unique()) if "well" in df.columns else ["All"]
        )

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for feature in features:
                fig_m, ax = plt.subplots(figsize=(17.5, 9.8), dpi=190)
                apply_matplotlib_petro_style(fig_m, ax)
                ax.set_title(f"{chart_title_from_data(df, custom_chart_title)}\n{column_label(feature)}", fontsize=24, fontweight="bold", pad=18)

                for wi, series_label in enumerate(series_values):
                    g = df[df["series_label"].astype(str) == series_label].sort_values(
                        "plot_x" if "plot_x" in df.columns else ("datetime" if "datetime" in df.columns else df.index.name)
                    ).reset_index(drop=True)
                    if g.empty or feature not in g.columns:
                        continue

                    x = g["plot_x"] if "plot_x" in g.columns and g["plot_x"].notna().any() else (
                        pd.to_datetime(g["datetime"], errors="coerce") if "datetime" in g.columns and g["datetime"].notna().any() else pd.Series(range(len(g)))
                    )
                    y = numeric_feature_series(g, feature)
                    color = feature_color(feature, wi)

                    ax.plot(x, y, marker="o", markersize=5, linewidth=2.6, color=color,
                            drawstyle="steps-post" if feature in {"choke_unified", "choke_pct", "choke_size_64", "choke_ambiguous"} else "default",
                            label=series_label if len(series_values) > 1 else None)

                    idxs = chart_label_indices_for_export(g, feature)
                    for i in sorted(idxs):
                        if i >= len(g) or pd.isna(y.iloc[i]):
                            continue
                        _label_dx, _label_dy, _label_ha = matplotlib_value_label_placement(g, feature, i)
                        ax.annotate(
                            format_plot_value(feature, y.iloc[i]),
                            (x.iloc[i], y.iloc[i]),
                            textcoords="offset points",
                            xytext=(_label_dx, _label_dy),
                            ha=_label_ha,
                            fontsize=11,
                            color=color,
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.12", fc=EXPORT_LABEL_BG, ec=EXPORT_LABEL_EDGE, alpha=0.7),
                        )

                y_limits = custom_y_ranges.get(feature) or default_y_axis_range(df, feature)
                if y_limits:
                    ax.set_ylim(y_limits[0], y_limits[1])

                ax.set_ylabel(column_label(feature), fontsize=16, fontweight="bold")
                ax.set_xlabel(x_axis_title, fontsize=16, fontweight="bold")

                try:
                    x_for_notes = ax.lines[0].get_xdata() if ax.lines else None
                except Exception:
                    x_for_notes = None
                _apply_matplotlib_notes(ax, x_values=x_for_notes)
                ax.grid(True, which="major", alpha=0.28)
                ax.tick_params(axis="both", labelsize=13)

                if x_axis_mode == "Real calendar time" and "datetime" in df.columns and df["datetime"].notna().any():
                    ax.xaxis.set_major_formatter(mdates.DateFormatter(history_matplotlib_date_format(df) if analysis_view == "Production history" else "%d-%b\n%H:%M"))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=3 if analysis_view == "Production history" else 8, maxticks=9 if analysis_view == "Production history" else 16))
                    fig_m.autofmt_xdate(rotation=0)
                elif is_aligned_elapsed_mode(x_axis_mode):
                    tick_settings = elapsed_axis_tick_kwargs(df)
                    if tick_settings:
                        ax.set_xticks(tick_settings.get("tickvals", []))
                        ax.set_xticklabels(tick_settings.get("ticktext", []), rotation=0)
                elif is_compressed_real_date_mode(x_axis_mode):
                    density_ticks = {"Sparse": 7, "Balanced": 11, "Detailed": 18}.get(x_axis_label_density, 11)
                    if chart_view_mode == "Mobile-friendly":
                        density_ticks = min(density_ticks, 7)
                    elif chart_view_mode == "Wide report view":
                        density_ticks = max(density_ticks, 14)
                    tick_settings = compressed_axis_tick_kwargs(df, max_ticks_per_series=3, max_total_ticks=density_ticks)
                    if tick_settings:
                        ax.set_xticks(tick_settings.get("tickvals", []))
                        ax.set_xticklabels([str(t).replace("<br>", "\n") for t in tick_settings.get("ticktext", [])], rotation=30, ha="right")

                if len(series_values) > 1:
                    ax.legend(fontsize=12, loc="best")

                apply_matplotlib_petro_style(fig_m, ax)
                fig_m.tight_layout(rect=[0.02, 0.02, 0.98, _matplotlib_note_top_limit()])

                png_buffer = io.BytesIO()
                fig_m.savefig(png_buffer, format="png", dpi=190, facecolor=CHART_PAPER_BG, edgecolor=CHART_PAPER_BG)
                plt.close(fig_m)
                png_buffer.seek(0)

                safe_feature = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in column_label(feature))[:80]
                zf.writestr(f"{safe_feature}.png", png_buffer.getvalue())

        zip_buffer.seek(0)
        return zip_buffer.getvalue()


    def plotly_static_image_bytes(fig_obj, fmt, width, height, scale=1):
        """Return Plotly static-image bytes, or (None, error_message) if Kaleido/Chrome is unavailable."""
        try:
            kwargs = {"format": fmt, "width": width, "height": height}
            if fmt == "png":
                kwargs["scale"] = scale
            return fig_obj.to_image(**kwargs), None
        except Exception as e:
            return None, str(e)


    def _matplotlib_x_values(g):
        if x_axis_mode == "Real calendar time" and "datetime" in g.columns and g["datetime"].notna().any():
            return pd.to_datetime(g["datetime"], errors="coerce")
        if "plot_x" in g.columns and g["plot_x"].notna().any():
            return pd.to_numeric(g["plot_x"], errors="coerce")
        return pd.Series(range(1, len(g) + 1), index=g.index)

    def _apply_matplotlib_x_axis(ax, df_for_ticks):
        import matplotlib.dates as mdates
        if x_axis_mode == "Real calendar time" and "datetime" in df_for_ticks.columns and df_for_ticks["datetime"].notna().any():
            dt_all = pd.to_datetime(df_for_ticks["datetime"], errors="coerce").dropna()
            if analysis_view == "Production history":
                fmt = history_matplotlib_date_format(df_for_ticks)
                locator = mdates.AutoDateLocator(minticks=3, maxticks=9)
            else:
                fmt = "%d-%b-%Y\n%H:%M" if (not dt_all.empty and (dt_all.dt.year.nunique() > 1 or (dt_all.max() - dt_all.min()).days >= 330)) else "%d-%b\n%H:%M"
                locator = mdates.AutoDateLocator(minticks=6, maxticks=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
            ax.xaxis.set_major_locator(locator)
        elif is_aligned_elapsed_mode(x_axis_mode):
            tick_settings = elapsed_axis_tick_kwargs(df_for_ticks, max_ticks=8 if chart_view_mode == "Mobile-friendly" else 12)
            if tick_settings:
                ax.set_xticks(tick_settings.get("tickvals", []))
                ax.set_xticklabels(tick_settings.get("ticktext", []), rotation=0)
        elif is_compressed_real_date_mode(x_axis_mode):
            density_ticks = {"Sparse": 7, "Balanced": 11, "Detailed": 18}.get(x_axis_label_density, 11)
            if chart_view_mode == "Mobile-friendly":
                density_ticks = min(density_ticks, 7)
            elif chart_view_mode == "Wide report view":
                density_ticks = max(density_ticks, 14)
            tick_settings = compressed_axis_tick_kwargs(
                df_for_ticks,
                max_ticks_per_series=3,
                max_total_ticks=density_ticks,
            )
            if tick_settings:
                ax.set_xticks(tick_settings.get("tickvals", []))
                ax.set_xticklabels([str(t).replace("<br>", "\n") for t in tick_settings.get("ticktext", [])], rotation=30, ha="right")

    def _draw_all_test_separators_matplotlib(ax, df_for_sep):
        for _sep in chart_separator_positions(df_for_sep):
            try:
                ax.axvline(_sep.get("x"), color="#334155", linestyle="--", linewidth=2.0, alpha=0.88, zorder=8)
            except Exception:
                pass

    def _matplotlib_event_levels(events, x_values=None, max_levels=4):
        return note_event_levels(events, x_values=x_values, max_levels=max_levels)

    def _matplotlib_note_top_limit():
        # Notes are drawn inside the axes, so no large external note band is needed.
        return 0.94

    def _apply_matplotlib_notes(ax, x_values=None):
        """Draw v92-style inline notes with stronger contrast and staggered placement."""
        _draw_all_test_separators_matplotlib(ax, filtered)
        note_count = total_note_count()
        interval_font_size, event_font_size = adaptive_note_font_sizes(
            note_count, mobile=(chart_view_mode == "Mobile-friendly")
        )
        intervals = visible_intervals_for_notes()
        events = visible_events_for_notes(x_values=x_values)

        for idx, interval in enumerate(intervals):
            x0, x1 = interval["x0"], interval["x1"]
            level = int(interval.get("level", 0) or 0)
            note_col = note_color(idx)
            ax.axvline(x0, color=note_col, linestyle="--", linewidth=1.8, alpha=0.92, zorder=10)
            ax.axvline(x1, color=note_col, linestyle="--", linewidth=1.8, alpha=0.92, zorder=10)
            try:
                x_mid = x0 + (x1 - x0) / 2
            except Exception:
                x_mid = x0
            y_frac = max(0.70, 0.965 - 0.085 * min(level, 3))
            try:
                ax.annotate(
                    "", xy=(x1, y_frac), xytext=(x0, y_frac),
                    xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="<->", color=note_col, lw=1.8),
                    annotation_clip=False, zorder=13,
                )
            except Exception:
                pass
            ax.text(
                x_mid, min(0.992, y_frac + 0.018),
                str(interval.get("label_html") or compact_note_label(interval.get("label", ""))).replace("<br>", "\n"),
                transform=ax.get_xaxis_transform(), fontsize=interval_font_size,
                fontweight="bold", ha="center", va="bottom", color=note_col,
                bbox=dict(boxstyle="round,pad=0.22", fc=EXPORT_LABEL_BG, ec=note_col, alpha=0.98),
                clip_on=False, zorder=14,
            )

        vertical = event_label_style == "Vertical labels"
        base_frac = 0.84 if intervals else 0.92
        for idx, event in enumerate(events):
            level = int(event.get("level", 0) or 0)
            note_col = note_color(idx + len(plot_intervals or []))
            y_frac = max(0.22, base_frac - 0.095 * min(level, 6)) if vertical else max(0.30, base_frac - 0.075 * min(level, 7))
            ax.axvline(event["plot_x"], color=note_col, linestyle="--", linewidth=1.6, alpha=0.86, zorder=10)
            rotation = 90 if vertical else 0
            ha = "right" if vertical else "center"
            va = "top" if vertical else "bottom"
            try:
                x_shift_points = float(event.get("x_shift_px", 0) or 0) * 0.5
            except Exception:
                x_shift_points = 0
            ax.annotate(
                str(event.get("label_html") or compact_note_label(event.get("label", ""))).replace("<br>", "\n"),
                xy=(event["plot_x"], y_frac), xycoords=("data", "axes fraction"),
                xytext=(x_shift_points, 0), textcoords="offset points",
                rotation=rotation, va=va, ha=ha, fontsize=event_font_size,
                fontweight="bold", color=note_col,
                bbox=dict(boxstyle="round,pad=0.16", fc=EXPORT_LABEL_BG, ec=note_col, alpha=0.96),
                clip_on=False, annotation_clip=False, zorder=14,
            )

    def matplotlib_overview_export_bytes(df, features, fmt="png"):
        """High-resolution single report chart export with labels and notes."""
        import matplotlib.pyplot as plt

        if df.empty or not features:
            raise ValueError("No filtered data/features available for export.")

        series_values = sorted(df["series_label"].dropna().astype(str).unique()) if "series_label" in df.columns else (
            sorted(df["well"].dropna().astype(str).unique()) if "well" in df.columns else ["All"]
        )

        n_features = max(1, len(features))
        n_points = max_points_per_trace(df)
        if chart_view_mode == "Mobile-friendly":
            width_in = max(15.0, min(24.0, n_points * (0.26 if len(series_values) > 1 else 0.22)))
            height_in = max(8.0, 5.8 * n_features)
            title_fs = 22
        elif chart_view_mode == "Wide report view":
            width_in = max(24.0, min(46.0, n_points * (0.56 if len(series_values) > 1 else 0.42)))
            height_in = max(8.0, 6.2 * n_features)
            title_fs = 28
        else:
            width_in = max(20.0, min(40.0, n_points * (0.48 if len(series_values) > 1 else 0.34)))
            height_in = max(8.0, 5.6 * n_features)
            title_fs = 25
        fig_m, axes = plt.subplots(n_features, 1, figsize=(width_in, height_in), dpi=240, squeeze=False)
        apply_matplotlib_petro_style(fig_m, axes)
        axes = axes.flatten()
        fig_m.suptitle(chart_title_from_data(df, custom_chart_title), fontsize=title_fs, fontweight="bold", y=0.995)

        group_col = "series_label" if "series_label" in df.columns else "well"
        for ax, feature in zip(axes, features):
            for wi, series_label in enumerate(series_values):
                g_all = df[df[group_col].astype(str) == series_label].copy()
                if g_all.empty or feature not in g_all.columns:
                    continue

                color = well_color(wi) if len(series_values) > 1 else feature_color(feature, wi)
                first_segment = True
                for g in iter_plot_segments(g_all, feature):
                    if g.empty:
                        continue
                    y = numeric_feature_series(g, feature)
                    if y.notna().sum() == 0:
                        continue
                    x = _matplotlib_x_values(g)

                    # Plot each detected test/data segment separately.  This is
                    # essential for PDF/PNG exports: matplotlib otherwise joins
                    # the last point of one test to the first point of the next.
                    ax.plot(
                        x,
                        y,
                        marker="o" if show_points else None,
                        markersize=4.8,
                        linewidth=2.4,
                        color=color,
                        drawstyle="steps-post" if feature in {"choke_unified", "choke_pct", "choke_size_64", "choke_ambiguous"} else "default",
                        label=series_label if (len(series_values) > 1 and first_segment) else None,
                    )

                    idxs = chart_label_indices_for_export(g, feature)
                    for i in sorted(idxs):
                        if i >= len(g) or pd.isna(y.iloc[i]):
                            continue
                        _label_dx, _label_dy, _label_ha = matplotlib_value_label_placement(g, feature, i)
                        ax.annotate(
                            format_plot_value(feature, y.iloc[i]),
                            (x.iloc[i], y.iloc[i]),
                            textcoords="offset points",
                            xytext=(_label_dx, _label_dy),
                            ha=_label_ha,
                            fontsize=10.5,
                            color=color,
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.10", fc=EXPORT_LABEL_BG, ec=EXPORT_LABEL_EDGE, alpha=0.72),
                        )
                    first_segment = False

            y_limits = custom_y_ranges.get(feature) or default_y_axis_range(df, feature)
            if y_limits:
                ax.set_ylim(y_limits[0], y_limits[1])

            try:
                x_for_notes = ax.lines[0].get_xdata() if ax.lines else None
            except Exception:
                x_for_notes = None
            _apply_matplotlib_notes(ax, x_values=x_for_notes)
            ax.set_ylabel(column_label(feature), fontsize=14, fontweight="bold")
            ax.grid(True, which="major", alpha=0.28)
            ax.tick_params(axis="both", labelsize=11)
            _apply_matplotlib_x_axis(ax, df)
            if len(series_values) > 1:
                ax.legend(fontsize=11, loc="best")

        axes[-1].set_xlabel(x_axis_title_from_mode(x_axis_mode), fontsize=14, fontweight="bold")
        apply_matplotlib_petro_style(fig_m, axes)
        fig_m.tight_layout(rect=[0.02, 0.02, 0.98, max(_matplotlib_note_top_limit(), 0.90)])

        output = io.BytesIO()
        if fmt == "pdf":
            fig_m.savefig(output, format="pdf", bbox_inches="tight", facecolor=CHART_PAPER_BG, edgecolor=CHART_PAPER_BG)
        else:
            fig_m.savefig(output, format="png", dpi=320, bbox_inches="tight", facecolor=CHART_PAPER_BG, edgecolor=CHART_PAPER_BG)
        plt.close(fig_m)
        output.seek(0)
        payload = output.getvalue()
        return make_portable_pdf(payload, df, features) if fmt == "pdf" else payload

    def human_readable_multi_png_bytes(df, features):
        """Create one PNG byte stream per selected feature for phone-friendly separate downloads."""
        import matplotlib.pyplot as plt
        if df.empty or not features:
            raise ValueError("No filtered data/features available for export.")

        series_values = sorted(df["series_label"].dropna().astype(str).unique()) if "series_label" in df.columns else (
            sorted(df["well"].dropna().astype(str).unique()) if "well" in df.columns else ["All"]
        )
        outputs = {}

        for feature in features:
            width_in = 18.0 if chart_view_mode == "Mobile-friendly" else 20.0
            height_in = 9.5 if chart_view_mode == "Mobile-friendly" else 10.5
            fig_m, ax = plt.subplots(figsize=(width_in, height_in), dpi=220)
            apply_matplotlib_petro_style(fig_m, ax)
            ax.set_title(f"{chart_title_from_data(df, custom_chart_title)}\n{column_label(feature)}", fontsize=22, fontweight="bold", pad=18)

            for wi, series_label in enumerate(series_values):
                g_all = df[df["series_label"].astype(str) == series_label].copy() if "series_label" in df.columns else df.copy()
                if g_all.empty or feature not in g_all.columns:
                    continue
                color = well_color(wi) if len(series_values) > 1 else feature_color(feature, wi)
                first_segment = True

                for g in iter_plot_segments(g_all, feature):
                    x = _matplotlib_x_values(g)
                    y = numeric_feature_series(g, feature)
                    ax.plot(
                        x,
                        y,
                        marker="o" if show_points else None,
                        markersize=4.4,
                        linewidth=2.4,
                        color=color,
                        drawstyle="steps-post" if feature in {"choke_unified", "choke_pct", "choke_size_64", "choke_ambiguous"} else "default",
                        label=series_label if (len(series_values) > 1 and first_segment) else None,
                    )

                    idxs = chart_label_indices_for_export(g, feature)
                    if value_label_mode in ["Hourly + min/max", "Clean readable - recommended"]:
                        max_lbl = 14 if chart_view_mode == "Mobile-friendly" else 22
                        if len(idxs) > max_lbl:
                            keep = sorted(idxs)
                            step = max(1, len(keep) // max_lbl)
                            idxs = set(keep[::step])
                            idxs.update({0, len(g) - 1})
                    for i in sorted(idxs):
                        if i >= len(g) or pd.isna(y.iloc[i]):
                            continue
                        _label_dx, _label_dy, _label_ha = matplotlib_value_label_placement(g, feature, i)
                        ax.annotate(
                            format_plot_value(feature, y.iloc[i]),
                            (x.iloc[i], y.iloc[i]),
                            textcoords="offset points",
                            xytext=(_label_dx, _label_dy),
                            ha=_label_ha,
                            fontsize=9.5,
                            color=color,
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.10", fc=EXPORT_LABEL_BG, ec=EXPORT_LABEL_EDGE, alpha=0.70),
                        )
                    first_segment = False

            y_limits = custom_y_ranges.get(feature) or default_y_axis_range(df, feature)
            if y_limits:
                ax.set_ylim(y_limits[0], y_limits[1])

            ax.set_ylabel(column_label(feature), fontsize=15, fontweight="bold")
            ax.set_xlabel(x_axis_title_from_mode(x_axis_mode), fontsize=14, fontweight="bold")
            try:
                x_for_notes = ax.lines[0].get_xdata() if ax.lines else None
            except Exception:
                x_for_notes = None
            _apply_matplotlib_notes(ax, x_values=x_for_notes)
            ax.grid(True, which="major", alpha=0.28)
            ax.tick_params(axis="both", labelsize=11)
            _apply_matplotlib_x_axis(ax, df)
            if len(series_values) > 1:
                ax.legend(fontsize=11, loc="best")

            apply_matplotlib_petro_style(fig_m, ax)
            fig_m.tight_layout(rect=[0.02, 0.02, 0.98, _matplotlib_note_top_limit()])
            output = io.BytesIO()
            fig_m.savefig(output, format="png", dpi=260, bbox_inches="tight", facecolor=CHART_PAPER_BG, edgecolor=CHART_PAPER_BG)
            plt.close(fig_m)
            output.seek(0)

            safe_feature = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in column_label(feature))[:80]
            outputs[f"production_test_{safe_feature}.png"] = output.getvalue()

        return outputs


    def _prepare_export(export_key, fmt_label, make_bytes_func, file_name, mime):
        # Theme is part of every key. A Light export can therefore never remain
        # visible or downloadable after the user switches to Dark, or vice versa.
        theme_key = re.sub(r"[^a-z0-9]+", "_", ACTIVE_THEME_NAME.lower()).strip("_")
        bkey = f"export_bytes_{export_key}_{theme_key}"
        ekey = f"export_error_{export_key}_{theme_key}"
        st.session_state.setdefault(bkey, None)
        st.session_state.setdefault(ekey, "")

        if st.button(f"Prepare {fmt_label} ({ACTIVE_THEME_NAME})", key=f"prepare_{export_key}_{theme_key}"):
            for _k in list(st.session_state.keys()):
                if _k.startswith("export_bytes_") and _k != bkey:
                    st.session_state.pop(_k, None)
                elif _k.startswith("export_error_") and _k != ekey:
                    st.session_state.pop(_k, None)
            gc.collect()
            st.session_state[bkey] = None
            st.session_state[ekey] = ""
            try:
                with st.spinner(f"Preparing {fmt_label}..."):
                    st.session_state[bkey] = make_bytes_func()
            except Exception as e:
                st.session_state[ekey] = str(e)

        if st.session_state.get(ekey):
            st.error(f"{fmt_label} export failed.")
            st.caption(st.session_state[ekey])

        prepared = st.session_state.get(bkey)
        if prepared:
            if isinstance(prepared, dict):
                st.caption("Download each selected chart as a separate PNG. This works on phones without ZIP files.")
                for i, (fname, data_bytes) in enumerate(prepared.items(), start=1):
                    st.download_button(
                        f"Download {fname}",
                        data=data_bytes,
                        file_name=fname,
                        mime="image/png",
                        key=f"download_{export_key}_{theme_key}_{i}",
                    )
            else:
                st.download_button(
                    f"Download {fmt_label}",
                    data=prepared,
                    file_name=file_name,
                    mime=mime,
                    key=f"download_{export_key}_{theme_key}",
                )

    dl1, dl2 = st.columns(2)
    dl3, dl4 = st.columns(2)

    with dl1:
        _prepare_export(
            "single_png",
            "single chart PNG",
            lambda: matplotlib_overview_export_bytes(filtered, selected_features, fmt="png"),
            f"production_test_single_chart_{ACTIVE_THEME_NAME.lower()}.png",
            "image/png",
        )

    with dl2:
        _prepare_export(
            "single_pdf",
            "single chart PDF",
            lambda: matplotlib_overview_export_bytes(filtered, selected_features, fmt="pdf"),
            f"production_test_single_chart_{ACTIVE_THEME_NAME.lower()}.pdf",
            "application/pdf",
        )

    with dl3:
        _prepare_export(
            "multi_png",
            "separate charts PNGs",
            lambda: human_readable_multi_png_bytes(filtered, selected_features),
            f"production_test_separate_charts_{ACTIVE_THEME_NAME.lower()}.png",
            "image/png",
        )

    with dl4:
        _prepare_export(
            "multi_pdf",
            "multi-charts PDF",
            lambda: human_readable_pdf_bytes(filtered, selected_features),
            f"production_test_multi_charts_{ACTIVE_THEME_NAME.lower()}.pdf",
            "application/pdf",
        )
else:
    st.warning("Choose at least one feature to plot, and make sure the filters leave some rows.")
