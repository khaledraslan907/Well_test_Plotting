from __future__ import annotations
import io
import json
import re
import zipfile
import traceback
import gc
import hashlib
from pathlib import Path
from datetime import date, datetime, time
from typing import Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

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

def feature_color(feature_name: str, fallback_index: int = 0) -> str:
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
        return f"Well {wells[0]}"
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

APP_UI_BUILD_ID = "v75-continuous-fast-responsive-ui-20260627"

UI_THEME_PRESETS = {
    "Light": {
        "color_scheme": "light",
        "app_bg": "#F4F7FA",
        "app_bg_2": "#EAF0F5",
        "sidebar_bg": "#F8FAFC",
        "panel_bg": "#FFFFFF",
        "panel_bg_2": "#F5F8FA",
        "input_bg": "#FFFFFF",
        "border": "#CDD9E2",
        "border_strong": "#AFC1CE",
        "accent": "#0F627B",
        "accent_hover": "#147D9A",
        "accent_soft": "#DCEEF3",
        "gold": "#A97B32",
        "gold_soft": "#C69A52",
        "text": "#10222E",
        "text_strong": "#081923",
        "text_muted": "#5C7180",
        "success": "#287A57",
        "warning": "#A86100",
        "danger": "#B73E38",
        "grid": "rgba(15, 98, 123, 0.035)",
        "glow": "rgba(47, 141, 168, 0.14)",
        "shadow": "rgba(22, 46, 60, 0.10)",
        "chart_paper": "#FFFFFF",
        "chart_plot": "#F8FAFC",
        "chart_text": "#152631",
        "chart_grid": "#D9E3EA",
        "chart_grid_soft": "#ECF1F4",
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
ACTIVE_THEME_NAME = st.session_state.get("ui_theme", "Light")
ACTIVE_THEME = UI_THEME_PRESETS.get(ACTIVE_THEME_NAME, UI_THEME_PRESETS["Light"])
CHART_PAPER_BG = ACTIVE_THEME["chart_paper"]
CHART_PLOT_BG = ACTIVE_THEME["chart_plot"]
CHART_TEXT = ACTIVE_THEME["chart_text"]
CHART_GRID = ACTIVE_THEME["chart_grid"]
CHART_GRID_SOFT = ACTIVE_THEME["chart_grid_soft"]
CHART_LEGEND_BG = ACTIVE_THEME["chart_legend"]

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

MIN_DATE_ALLOWED = pd.Timestamp("1900-01-01").date()
MAX_DATE_ALLOWED = pd.Timestamp("2100-12-31").date()

# User-taught column aliases are stored beside app.py. This makes the app learn
# new company abbreviations without editing Python code every time.
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
    any new company abbreviation, such as Pi, Pd, AMp, Freq, Ti, Tm, Vx, etc.
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
                st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, height=160)

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


def feature_key_text(feature_name):
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(feature_name))


def compressed_axis_tick_kwargs(df, max_ticks_per_series=4, max_total_ticks=22):
    """Build readable x-axis labels for the global compressed timeline."""
    if df.empty or "plot_x" not in df.columns or "datetime" not in df.columns:
        return {}

    pts = df[["plot_x", "datetime"]].dropna().copy()
    if pts.empty:
        return {}
    pts["plot_x"] = pd.to_numeric(pts["plot_x"], errors="coerce")
    pts["datetime"] = pd.to_datetime(pts["datetime"], errors="coerce")
    pts = pts.dropna().drop_duplicates(subset=["plot_x"]).sort_values("plot_x").reset_index(drop=True)
    if pts.empty:
        return {}

    max_total_ticks = max(3, int(max_total_ticks or 10))
    n = len(pts)
    if n <= max_total_ticks:
        idxs = list(range(n))
    else:
        idxs = sorted(set(np.linspace(0, n - 1, max_total_ticks).round().astype(int).tolist()))

    # Include year only when the displayed data spans more than one year.
    # This keeps normal short tests clean, but prevents 2014 vs 2026 comparisons
    # from looking like the same month/day.
    years = pts["datetime"].dt.year.dropna().unique()
    show_year = len(years) > 1 or (pts["datetime"].max() - pts["datetime"].min()).days >= 330

    tickvals = []
    ticktext = []
    for i in idxs:
        dt = pd.Timestamp(pts.loc[i, "datetime"])
        tickvals.append(float(pts.loc[i, "plot_x"]))
        if show_year:
            ticktext.append(dt.strftime("%d-%b-%Y<br>%H:%M"))
        else:
            ticktext.append(dt.strftime("%d-%b<br>%H:%M"))

    return {"tickmode": "array", "tickvals": tickvals, "ticktext": ticktext, "tickangle": 0}


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
    }}

    html, body, [class*="css"] {{
        font-family: Inter, Aptos, "Segoe UI", Roboto, Arial, sans-serif;
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
    st.caption("Configure data ingestion, engineering units, plots, events, and reports.")
    st.radio(
        "Theme",
        ["Light", "Dark"],
        horizontal=True,
        key="ui_theme",
        help="Light is optimized for office review and printed reports. Dark is optimized for screen-based engineering analysis.",
    )

with st.sidebar.expander("1. Data Sources", expanded=True):
    st.caption("Bring field data from spreadsheets, reports, exported chats, device files, or images.")
    uploaded_files = st.file_uploader(
        "Upload test files, reports, device exports, or WhatsApp ZIPs",
        type=["xlsx", "xls", "csv", "txt", "docx", "pdf", "zip", "jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Upload normal test files or a WhatsApp exported ZIP. Directly uploaded images are OCR-processed automatically; the OCR switch controls images inside ZIP files.",
        key="general_data_uploader_v70",
    )
    uploaded_ocr_images = st.file_uploader(
        "Upload CTU/HMI screen photos directly",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        help="Use this dedicated uploader for field photos like the CTU ALL DATA screens. The display is rectified, OCR values are extracted, and every field remains editable in the OCR Review before plotting.",
        key="direct_ctu_image_uploader_v70",
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

    whatsapp_text = st.text_area(
        "Or paste one or many TMU WhatsApp messages",
        height=180,
        placeholder="""PICO TMU-02
Date :06-06-2026
Well name : B3C18-7
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

with st.sidebar.expander("2. Ingestion & Processing", expanded=False):
    st.caption("Control test segmentation, OCR workload, and robust parsing behavior.")
    keep_same_well_one_test = st.checkbox(
        "Keep same well as one test regardless of time gap",
        value=False,
        help="Use this when a long test has large inactive periods but still belongs to the same well test.",
    )
    test_gap_hours = st.number_input(
        "Start a new test for the same well if gap exceeds (hours)",
        min_value=1.0,
        max_value=8760.0,
        value=72.0,
        step=1.0,
        disabled=keep_same_well_one_test,
        help="Same well continues the same test until this inactive gap is exceeded. Use a large value or the checkbox above for long tests.",
    )
    effective_test_gap_hours = 1_000_000.0 if keep_same_well_one_test else float(test_gap_hours)
    enable_ctu_ocr = st.checkbox(
        "Process images contained inside WhatsApp ZIPs",
        value=False,
        help="Directly uploaded CTU/HMI images are always OCR-processed. Turn this on only when a ZIP also contains screen images; keeping it off makes large ZIP uploads faster.",
    )
    max_ocr_images = st.number_input(
        "CTU/HMI image OCR limit per ZIP (0 = no limit)",
        min_value=0,
        max_value=5000,
        value=1000,
        step=50,
        help="Use the OCR checkbox above to skip images completely. Set this to 0 only when you want to OCR every image in the ZIP. Excel/text/PDF attachments are always parsed normally.",
    )

@st.cache_data(show_spinner=False, ttl=3600, max_entries=24)
def cached_load_uploaded_file(file_name: str, file_bytes: bytes, parse_images: bool, max_ocr_images: int, parser_build_id: str):
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
        PARSER_BUILD_ID,
        bool(enable_ctu_ocr),
        int(max_ocr_images),
        tuple(_uploaded_file_identity_v58(f) for f in uploaded_files),
    )
    cached_bundle = st.session_state.get("upload_parse_bundle_v58")
    cached_key = st.session_state.get("upload_parse_key_v58")

    if cached_bundle is not None and cached_key == upload_key:
        # UI changes now reuse the already parsed DataFrames directly. This
        # avoids per-file cache deserialization and repeated workbook handling.
        frames.extend(cached_bundle.get("frames", []))
        errors.extend(list(cached_bundle.get("errors", [])))
    else:
        parsed_frames = []
        parsed_errors = []
        upload_progress = st.progress(0.0, text="Reading uploaded files...") if len(uploaded_files) > 1 else None
        for upload_order, f in enumerate(uploaded_files):
            try:
                if upload_progress is not None:
                    upload_progress.progress(
                        upload_order / max(len(uploaded_files), 1),
                        text=f"Reading {upload_order + 1}/{len(uploaded_files)}: {f.name}",
                    )
                file_bytes = f.getvalue()
                _suffix = Path(str(f.name)).suffix.lower()
                _is_direct_image = _suffix in {".jpg", ".jpeg", ".png", ".webp"}
                _parse_images_for_file = bool(enable_ctu_ocr) or _is_direct_image
                parsed_tables = cached_load_uploaded_file(
                    f.name, file_bytes, _parse_images_for_file, int(max_ocr_images), PARSER_BUILD_ID
                )
                if parsed_tables:
                    for table_order, table in enumerate(parsed_tables):
                        if table is None or table.empty:
                            continue
                        table = table.copy()
                        table["_upload_order"] = int(upload_order)
                        table["_table_order"] = int(table_order)
                        parsed_frames.append(table)
                else:
                    parsed_errors.append(f"{f.name}: no usable time-series table detected. The file may be blank, or it has no date/time plus numeric readings after all parsers were tried.")
            except Exception as e:
                parsed_errors.append(f"{f.name}: {e}")
                with st.expander(f"Technical error details for {f.name}", expanded=False):
                    st.code(traceback.format_exc())
            finally:
                gc.collect()
        if upload_progress is not None:
            upload_progress.progress(1.0, text="Uploaded files parsed")

        st.session_state["upload_parse_key_v58"] = upload_key
        st.session_state["upload_parse_bundle_v58"] = {
            "frames": parsed_frames,
            "errors": list(parsed_errors),
        }
        frames.extend(parsed_frames)
        errors.extend(parsed_errors)

if whatsapp_text.strip():
    try:
        msg_df = parse_whatsapp_plain_or_export_text(whatsapp_text, source_name="Pasted_WhatsApp_Text")
        if not msg_df.empty:
            frames.append(msg_df)
        else:
            errors.append("WhatsApp text: no recognizable TMU report detected")
    except Exception as e:
        errors.append(f"WhatsApp text: {e}")

if errors:
    st.warning("Some files/messages were skipped or could not be parsed:\n\n" + "\n".join(f"- {e}" for e in errors))

if not frames:
    st.info("Start by uploading field files or pasting one or more WhatsApp TMU reports in the Data Sources panel.")
    st.stop()

# Concatenation and duplicate merging are cached in session state as well as
# individual file parsing. This keeps sidebar/chart interaction fast after a
# large multi-file upload instead of rebuilding the same merged dataset on every
# Streamlit rerun.
_whatsapp_key = hashlib.sha1(whatsapp_text.encode("utf-8", errors="ignore")).hexdigest() if whatsapp_text.strip() else ""
_combined_key = (PARSER_BUILD_ID, upload_key, _whatsapp_key)
_combined_cached = st.session_state.get("combined_data_bundle_v58")
_combined_cached_key = st.session_state.get("combined_data_key_v58")

if _combined_cached is not None and _combined_cached_key == _combined_key:
    data = _combined_cached["data"].copy(deep=False)
    rows_merged = int(_combined_cached.get("rows_merged", 0))
else:
    data = pd.concat(frames, ignore_index=True, sort=False)
    # Merge repeated/incomplete reports before assigning tests. The same well
    # and minute timestamp becomes one row: the most complete upload wins and
    # missing fields are filled from the other copy.
    rows_before_dedup = len(data)
    try:
        if hasattr(_tmu_parser, "merge_duplicate_test_rows_v53"):
            data = _tmu_parser.merge_duplicate_test_rows_v53(data)
    except Exception as dedup_error:
        errors.append(f"Duplicate-row merge was skipped: {dedup_error}")
    rows_merged = max(0, rows_before_dedup - len(data))
    st.session_state["combined_data_key_v58"] = _combined_key
    st.session_state["combined_data_bundle_v58"] = {
        "data": data.copy(deep=True),
        "rows_merged": int(rows_merged),
    }

if rows_merged:
    st.caption(f"Merged {rows_merged:,} repeated row(s) with the same well and date/time, keeping the most complete values.")

# Internal ingestion-order fields are used only while resolving duplicate rows.
# They are not engineering measurements and must never appear in plots/tables.
data.drop(columns=[c for c in ["_upload_order", "_table_order", "_source_row_order"] if c in data.columns], inplace=True, errors="ignore")

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
        _quality_data = _quality_data.rename(columns={
            "data_quality_note": "Engineering Check",
            "rejected_values": "Original / Withheld Values",
            "source_row": "Source Row",
            "datetime": "Date / Time",
        })
        st.dataframe(_quality_data, use_container_width=True, hide_index=True)
        st.download_button(
            "Download engineering checks CSV",
            data=_quality_data.to_csv(index=False).encode("utf-8-sig"),
            file_name="production_test_engineering_checks.csv",
            mime="text/csv",
            key="download_data_quality_review_v69",
        )

# Raw choke source columns are preserved. A user-selectable unified curve is
# created later, after the safe parsing/mapping steps, so changing the display
# unit never changes the original uploaded values.

# Test segmentation: same well continues until the selected gap is exceeded;
# a different well is always a separate test stream.
data = assign_test_ids(data, gap_hours=float(effective_test_gap_hours))

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
except Exception:
    pass

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
        review_df.insert(1, "Approve OCR", False)

        # Preview directly uploaded field photos. ZIP-contained images still appear
        # by filename and can be reviewed after extraction outside the application.
        preview_names = [
            str(name) for name in review_df.get("image_file", pd.Series(dtype=str)).dropna().astype(str).unique()
            if str(name) in direct_image_preview_map
        ]
        if preview_names:
            selected_preview = st.selectbox(
                "Image preview", preview_names, key="ocr_image_preview_v70"
            )
            st.image(
                direct_image_preview_map[selected_preview],
                caption=f"OCR source: {selected_preview}",
                use_container_width=True,
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
        for col in ocr_numeric_cols:
            if col in review_df.columns:
                column_config[col] = st.column_config.NumberColumn(
                    column_label(col), format="%.3f"
                )

        editable_columns = {"Approve OCR", "datetime", "well", "test_id", *ocr_numeric_cols}
        edited_review = st.data_editor(
            review_df,
            use_container_width=True,
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

        unapproved = ocr_mask & data.get("review_required", pd.Series(True, index=data.index)).fillna(True).astype(bool)
        if unapproved.any():
            st.info(
                f"{int(unapproved.sum())} OCR row(s) still require review. They remain visible for editing "
                "but are clearly flagged in Engineering Data Checks."
            )



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


def time_aggregation_rule(choice):
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
    # well from appearing several times as B15-42 (09-Jun), B15-42 (10-Jun), etc.
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


def iter_plot_segments(g):
    if g is None or g.empty:
        return []
    sort_col = "plot_x" if "plot_x" in g.columns else ("datetime" if "datetime" in g.columns else None)
    if "series_segment_id" in g.columns:
        segments = []
        for _, seg in g.groupby("series_segment_id", dropna=False, sort=True):
            seg2 = seg.sort_values(sort_col).reset_index(drop=True) if sort_col else seg.reset_index(drop=True)
            if not seg2.empty:
                segments.append(seg2)
        return segments
    return [g.sort_values(sort_col).reset_index(drop=True) if sort_col else g.reset_index(drop=True)]


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
units_sidebar_section = st.sidebar.expander("3. Engineering Units & Choke", expanded=False)
with units_sidebar_section:
    st.caption("Choose reporting units and interpret choke opening versus choke size.")
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
        )
        choke_full_open_64 = st.number_input(
            "Full-open choke size (/64 in)",
            min_value=1.0,
            max_value=256.0,
            value=128.0,
            step=1.0,
            help="Calibration used for conversion. Default: 100% = 128/64 in; therefore 50% = 64/64 in.",
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
            )
        treat_zero_choke_as_missing = st.checkbox(
            "Treat zero choke as blank/template value",
            value=True,
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
    st.dataframe(pd.DataFrame(detected_rows), use_container_width=True, height=180)
    if any(str(c).startswith("raw__") for c in numeric_cols):
        st.info(
            "Raw fallback columns mean the parser found a numeric time-series column but did not know its header alias yet. "
            "Use the Column mapping review panel above to map it once and save the alias for future uploads."
        )

# Sidebar filters
with st.sidebar.expander("4. Timeline & Test Segmentation", expanded=False):
    st.caption("Configure calendar time, elapsed time, compressed gaps, and reporting ranges.")
    time_filter_mode = st.selectbox(
        "Time range control",
        ["Slider", "Manual calendar/time"],
        index=0,
        help="Use Manual calendar/time for long tests where a slider is difficult.",
    )
    time_aggregation = st.selectbox(
        "Average readings by time interval",
        ["Raw data", "5 minutes", "15 minutes", "30 minutes", "1 hour", "6 hours", "1 day", "1 month", "1 year"],
        index=0,
        help=(
            "This reduces dense data. Example: 1 hour means all readings inside each hour are averaged "
            "into one plotted point. Raw data keeps every original reading."
        ),
    )
    x_axis_scale = st.selectbox(
        "X-axis tick scale",
        ["Auto readable", "30 minutes", "1 hour", "3 hours", "6 hours", "12 hours", "1 day", "1 month", "1 year"],
        index=0,
        help="Controls x-axis tick spacing. Start from 30 minutes to avoid unreadable dense time labels.",
    )

    x_axis_mode = st.selectbox(
        "X-axis display mode",
        [
            "Real calendar time",
            "Compressed real dates - remove empty gaps",
        ],
        index=0,
        help=(
            "Real calendar time keeps true dates and gaps. "
            "Compressed real dates removes long empty gaps between test periods."
        ),
    )
    if is_compressed_real_date_mode(x_axis_mode):
        continuous_gap_hours = st.number_input(
            "Keep real spacing for gaps up to (hours)",
            min_value=0.0,
            max_value=24.0,
            value=2.0,
            step=0.5,
            help="Longer empty periods are visually compressed, but all readings with the same Test ID remain connected as one curve.",
        )
        compressed_gap_hours = st.number_input(
            "Visual gap shown for separated tests (hours)",
            min_value=0.1,
            max_value=12.0,
            value=0.75,
            step=0.25,
            help="Controls how much empty space remains after long gaps are compressed.",
        )
    else:
        continuous_gap_hours = 2.0
        compressed_gap_hours = 0.75

    x_axis_label_density = st.selectbox(
        "X-axis label density",
        ["Sparse", "Balanced", "Detailed"],
        index=1,
        help="Use Sparse for phone view or many uploads; Detailed for final wide reports.",
    )

    chart_view_mode = st.selectbox(
        "Chart screen layout",
        ["Auto / desktop", "Mobile-friendly", "Wide report view"],
        index=0,
        help=(
            "Mobile-friendly reduces tick and value-label crowding on phones. "
            "Wide report view gives larger panels on desktop and exports."
        ),
    )
    trace_grouping = "Auto"


with st.sidebar.expander("5. Well & Signal Selection", expanded=True):
    st.caption("Select the wells, test streams, and measured parameters to review.")

    # Prefer recent wells/tests with actual numeric readings, not alphabetical chat history.
    def _has_any_plot_numeric(_df):
        mask = pd.Series([False] * len(_df), index=_df.index)
        preferred = [
            "gross_rate_bpd", "oil_rate_stbd", "water_rate_bpd", "gas_rate_mmscfd",
            "whp_psi", "sep_p_psi", "pumping_pressure_psi", "bsw_pct",
            "ctu_wellhead_pressure_psi", "ctu_circulation_pressure_psi", "ctu_reel_depth_ft",
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
    selected_wells = st.multiselect("Choose wells", all_wells, default=all_wells[:1] if all_wells else [])

    # Test/period filtering removed from the sidebar.
    # Tests are still detected internally and shown in the data preview/export,
    # but the main workflow now filters by well and time range only.
    selected_tests = []
    all_tests = []

    select_all_features = st.checkbox(
        "Select all numeric columns",
        value=False,
        help="Shows every detected numeric column in the plot list, including raw fallback columns from unseen templates.",
    )

    default_features = [
        c for c in [
            "gross_rate_bpd", "qgross_s_bpd", "oil_rate_stbd", "qoil_s_stbd",
            "water_rate_bpd", "qwat_s_bpd", "gas_rate_mmscfd", "gas_formation_mmscfd",
            "pumping_pressure_psi", "ctu_circulation_pressure_psi", "ctu_wellhead_pressure_psi",
            "ctu_reel_depth_ft", "ctu_reel_speed_ftmin", "ctu_n2_rate_scfm",
            "bsw_pct", "wlr_s_pct", "whp_psi", "choke_unified",
            "flow_press_psi", "sep_p_psi", "salinity_kppm",
        ] if c in numeric_cols
    ] or numeric_cols[: min(6, len(numeric_cols))]

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
    selected_features = st.multiselect(
        "Choose features to plot",
        numeric_cols,
        format_func=column_label,
        key=feature_state_key,
    )
    st.session_state[select_all_prev_key] = bool(select_all_features)


    if selected_tests:
        if len(selected_tests) == 1:
            auto_chart_header = selected_tests[0]
        else:
            auto_chart_header = " vs ".join(selected_tests[:3]) + (f" +{len(selected_tests) - 3} more" if len(selected_tests) > 3 else "")
    elif selected_wells:
        if len(selected_wells) == 1:
            auto_chart_header = f"Well {selected_wells[0]}"
        else:
            auto_chart_header = " vs ".join(selected_wells[:5]) + (f" +{len(selected_wells) - 5} more" if len(selected_wells) > 5 else "")
    else:
        auto_chart_header = "Well Production Test"

    data_title_signature = "_".join([str(x) for x in sorted(data.get("source", pd.Series(dtype=str)).dropna().astype(str).unique())[:3]])[:80]
    custom_chart_title = st.text_input(
        "Chart header / title",
        value=auto_chart_header,
        help="Edit this header if you want a different title. The default uses selected well/test only.",
        key=f"chart_header_{abs(hash(data_title_signature))}_{len(data)}",
    )

with st.sidebar.expander("6. Visualization Studio", expanded=False):
    st.caption("Semantic colors are fixed by engineering meaning: oil green, water blue, gas cyan, pressure red/orange, choke gold, and quality properties purple/brown.")
    custom_y_ranges = {}
    with st.expander("Y-axis scale per graph", expanded=False):
        use_custom_y_scale = st.checkbox(
            "Use custom Y-axis ranges",
            value=False,
            help="Set min/max for each selected graph, e.g. Gross Rate from 0 to 1000.",
        )

        if use_custom_y_scale and selected_features:
            for feature in selected_features:
                vals = numeric_feature_series(data, feature).dropna() if feature in data.columns else pd.Series(dtype="float64")
                default_min = float(vals.min()) if not vals.empty else 0.0
                default_max = float(vals.max()) if not vals.empty else 1.0
                if default_min == default_max:
                    default_max = default_min + 1.0

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
        help="This only affects the plotted/filtered copy, not the originally detected data.",
    )

    hide_zero_flow_rows = st.checkbox(
        "Hide zero-flow/bypassed rows",
        value=False,
        help="Useful for EXPRO MPFM reports during bypass periods where oil, water, gas, and gross are all zero.",
    )

    plot_mode = st.selectbox(
        "Plot style",
        ["Separate panels like report", "Overlay actual values"],
        index=0,
        help="Use separate panels for normal reports. Use overlay to compare actual values on one axis.",
    )

    # Optional combined dual-axis charts.  These do not replace the normal
    # multi-panel report; they add extra comparison charts above it.
    dual_axis_charts = []
    if len(numeric_cols) >= 2 and selected_features:
        with st.expander("Combined charts with secondary Y-axis", expanded=False):
            st.caption(
                "Create one, two, or three combined charts. Each combined chart can have one or more "
                "features on the left Y-axis and one or more features on the right Y-axis. "
                "The normal selected-feature report remains below."
            )
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
    show_points = st.checkbox("Show markers", value=default_markers)

    value_label_mode = st.selectbox(
        "Value labels on chart",
        [
            "Clean readable - recommended",
            "Every 20 readings",
            "Every 8 readings",
            "Hourly + min/max",
            "All values - use wide export",
            "First and last only",
            "Off",
        ],
        index=0,
        help=(
            "Clean readable keeps labels to important/non-crowded points. "
            "Every 20 readings is usually best for long field-test reports. "
            "Use All values only with a wide export."
        ),
    )

    st.caption(
        "For dense charts, avoid All values. Clean readable and Every 20 readings are designed to keep values legible."
    )

    label_decimals_default = st.selectbox(
        "Default number format on labels",
        ["Auto", "0 decimals", "1 decimal", "2 decimals"],
        index=0,
    )
    label_decimals_by_feature = {}
    if selected_features:
        with st.expander("Number format per graph", expanded=False):
            st.caption("Use this when one curve needs 0 decimals and another needs 1 or 2 decimals.")
            for feature in selected_features:
                label_decimals_by_feature[feature] = st.selectbox(
                    column_label(feature),
                    ["Use default", "Auto", "0 decimals", "1 decimal", "2 decimals"],
                    index=0,
                    key=f"label_decimals_{feature_key_text(feature)}",
                )

    with st.expander("Rename column labels for view/export", expanded=False):
        st.caption("Rename how columns appear in charts and tables without changing the parser field names.")
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

    note_color_theme = "High contrast"

    show_internal_names = False

with st.sidebar.expander("7. Operations & Events", expanded=False):
    st.caption("Add field events, operational intervals, and report annotations.")

    auto_hide_crowded_notes = st.checkbox(
        "Auto hide some notes when too crowded",
        value=False,
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
        disabled=not auto_hide_crowded_notes,
        help="Used only when auto-hide is enabled.",
    )

    event_label_style = st.selectbox(
        "Event label layout",
        ["Auto staggered", "Vertical labels", "Compact top labels"],
        index=0,
        help="Use Vertical labels or Compact top labels when many notes are close together.",
    )
    enable_drag_annotations = st.checkbox(
        "Allow dragging event labels on the interactive chart",
        value=False,
        help="Mouse drag is for on-screen adjustment only. Downloaded PNG/PDF charts use the clean automatic note layout and do not save dragged positions.",
    )

    st.caption("Select a start date/time and write a note. Add an optional end date/time only when the note should cover a period.")

    if "manual_events_table" not in st.session_state:
        st.session_state.manual_events_table = []
    if "operation_intervals_table" not in st.session_state:
        st.session_state.operation_intervals_table = []

    default_event_dt = None
    if "datetime" in data.columns and data["datetime"].notna().any():
        default_event_dt = data["datetime"].dropna().min().to_pydatetime()

    note_label_input = st.text_input(
        "Operation note",
        placeholder="SIWHP = 2000 psi, Well_1 choke 10%, Well_2 choke 50%, Start lifting...",
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
        note_start_time_picker = st.time_input(
            "Start time",
            value=default_event_dt.time().replace(second=0, microsecond=0) if default_event_dt else None,
            key="note_start_time_picker",
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
            note_end_time_picker = st.time_input(
                "End time",
                value=default_event_dt.time().replace(second=0, microsecond=0) if default_event_dt else None,
                key="note_end_time_picker",
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
                    {"start": start_dt_note, "end": end_dt_note, "label": note_label_input.strip(), "target": note_target}
                )
                st.success(f"Added interval: {start_dt_note:%Y-%m-%d %H:%M} to {end_dt_note:%Y-%m-%d %H:%M} | {note_label_input.strip()}")
        else:
            st.session_state.manual_events_table.append(
                {
                    "datetime": start_dt_note,
                    "label": note_label_input.strip(),
                    "target": note_target,
                }
            )
            st.success(f"Added event: {start_dt_note:%Y-%m-%d %H:%M} | {note_label_input.strip()}")

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
            use_container_width=True,
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
        st.dataframe(display_intervals, use_container_width=True, height=150)

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
if selected_wells:
    filtered = filtered[filtered["well"].astype(str).isin(selected_wells)]
if "test_id" in filtered.columns and selected_tests:
    filtered = filtered[filtered["test_id"].astype(str).isin(selected_tests)]

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

# Optional aggregation/resampling for long tests.
filtered = aggregate_time_data(filtered, time_aggregation)

if selected_features:
    filtered = apply_fill_method(filtered, selected_features, fill_method)

manual_events = []

# Add events created from the easy date/time + note UI.
for e in st.session_state.get("manual_events_table", []):
    try:
        manual_events.append({
            "datetime": pd.Timestamp(e["datetime"]),
            "label": str(e["label"]),
            "target": str(e.get("target", "All selected wells")),
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
            {"start": pd.Timestamp(i["start"]), "end": pd.Timestamp(i["end"]), "label": str(i.get("label", "")), "target": str(i.get("target", "All selected wells"))}
        )
    except Exception:
        pass

plot_intervals = convert_intervals_for_plot(operation_intervals, filtered, x_axis_mode)

# Engineering snapshot KPIs
render_section_title(
    "Engineering Snapshot",
    "Live summary of the parsed dataset and the rows currently feeding the visualization.",
)
quality_count = int(_quality_mask.sum()) if "_quality_mask" in globals() else 0
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Detected readings", f"{len(data):,}")
c2.metric("Active readings", f"{len(filtered):,}")
c3.metric("Wells", f"{data['well'].nunique() if 'well' in data.columns else 0:,}")
c4.metric("Test periods", f"{data['test_id'].nunique() if 'test_id' in data.columns else 0:,}")
c5.metric("Signals", f"{len(numeric_cols):,}")
c6.metric("Engineering checks", f"{quality_count:,}")

with st.expander("Detected data preview", expanded=False):
    st.caption("Detected data = cleaned rows pulled from uploads before your well/time/feature filters.")
    preview_cols = ["source", "sheet", "source_type", "well", "test_id", "link_status", "datetime", "time_text"] + numeric_cols
    preview_cols = [c for c in preview_cols if c in data.columns]
    display_detected = data[preview_cols].copy()
    if not show_internal_names:
        display_detected = display_detected.rename(columns={c: column_label(c) for c in display_detected.columns})
    st.dataframe(display_detected, use_container_width=True, height=260)

if selected_features and not filtered.empty:
    render_section_title(
        "Production Test Visualization",
        "Interactive engineering charts using the selected wells, test periods, signals, units, and event annotations.",
    )
    series_count_for_hint = filtered["series_label"].dropna().astype(str).nunique() if "series_label" in filtered.columns else 1
    if x_axis_mode == "Real calendar time":
        axis_tick_settings = x_axis_tick_kwargs(x_axis_scale)
    elif is_aligned_elapsed_mode(x_axis_mode):
        axis_tick_settings = elapsed_axis_tick_kwargs(filtered, max_ticks=8 if chart_view_mode == "Mobile-friendly" else 12)
    elif is_compressed_real_date_mode(x_axis_mode):
        density_ticks = {"Sparse": 7, "Balanced": 11, "Detailed": 18}.get(x_axis_label_density, 11)
        if chart_view_mode == "Mobile-friendly":
            density_ticks = min(density_ticks, 7)
        elif chart_view_mode == "Wide report view":
            density_ticks = max(density_ticks, 14)
        axis_tick_settings = compressed_axis_tick_kwargs(
            filtered,
            max_ticks_per_series=3,
            max_total_ticks=density_ticks,
        )
    else:
        axis_tick_settings = {}
    x_axis_title = x_axis_title_from_mode(x_axis_mode)

    NOTE_COLOR_PALETTES = {
        "Automatic multi-color": [
            "#92400e", "#1d4ed8", "#15803d", "#7c3aed", "#dc2626",
            "#0f766e", "#c2410c", "#be185d", "#0369a1", "#4d7c0f",
        ],
        "High contrast": [
            "#000000", "#d97706", "#2563eb", "#16a34a", "#dc2626",
            "#9333ea", "#0891b2", "#db2777", "#65a30d", "#ea580c",
        ],
        "Oilfield earth tones": [
            "#92400e", "#78350f", "#a16207", "#854d0e", "#7f1d1d",
            "#166534", "#365314", "#475569", "#713f12", "#431407",
        ],
        "Blue / green": [
            "#1d4ed8", "#0369a1", "#0f766e", "#15803d", "#4d7c0f",
            "#0e7490", "#1e40af", "#065f46", "#2563eb", "#059669",
        ],
        "Monochrome dark": ["#111827"],
    }

    def note_palette():
        return NOTE_COLOR_PALETTES.get(note_color_theme, NOTE_COLOR_PALETTES["Automatic multi-color"])

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

    def note_event_levels(events, x_values=None, max_levels=8):
        """Assign staggered rows for point-event labels so close labels do not overlap.

        Users can override the automatic row using the point-note table Y level.
        """
        if not events:
            return []
        try:
            xs = []
            for e in events:
                try:
                    xs.append(float(e.get("plot_x", 0)))
                except Exception:
                    pass
            if x_values is not None:
                try:
                    xs += [float(v) for v in x_values if pd.notna(v)]
                except Exception:
                    pass
            span = (max(xs) - min(xs)) if len(xs) >= 2 else 1.0
            min_gap = max(span * 0.070, 0.8)
        except Exception:
            min_gap = 1.0

        placed_until = [-1e18] * max_levels
        decorated = []
        sortable = []
        for i, event in enumerate(events):
            x = event.get("plot_x", 0)
            try:
                sx = float(x)
            except Exception:
                sx = float(i)
            sortable.append((sx, i, dict(event)))
        sortable.sort(key=lambda t: t[0])

        for sx, _, event in sortable:
            manual_level = str(event.get("y_level", "Auto") or "Auto")
            if manual_level != "Auto":
                try:
                    level = max(0, min(max_levels - 1, int(float(manual_level))))
                except Exception:
                    level = 0
            else:
                level = 0
                while level < max_levels and sx - placed_until[level] < min_gap:
                    level += 1
                if level >= max_levels:
                    level = max_levels - 1
            placed_until[level] = sx
            event["level"] = level
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
        if not auto_hide_crowded_notes:
            return intervals
        # Long parent intervals are already first from interval_levels(), so this keeps main events first.
        return intervals[: int(max_visible_notes_per_chart)]

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

    def add_operation_intervals_to_plotly(fig, features):
        fig = add_compressed_test_separators_to_plotly(fig, features)

        if not plot_intervals:
            return fig

        interval_font_size, _ = adaptive_note_font_sizes(total_note_count(), mobile=(chart_view_mode == "Mobile-friendly"))
        for idx, interval in enumerate(visible_intervals_for_notes()):
            x0 = interval["x0"]
            x1 = interval["x1"]
            label = interval["label"]
            level = int(interval.get("level", 0) or 0)
            note_col = note_color(idx)

            # v51: use whole-figure shapes instead of repeated labels in each subplot.
            # One draggable event/interval line spans all charts, so moving it in the
            # interactive Plotly view affects the whole multi-chart figure visually.
            for x_val in [x0, x1]:
                try:
                    fig.add_shape(
                        type="line",
                        x0=x_val,
                        x1=x_val,
                        y0=0,
                        y1=1,
                        xref="x",
                        yref="paper",
                        line=dict(color=note_col, width=2.2, dash="dash"),
                        opacity=0.90,
                    )
                except Exception:
                    pass

            try:
                x_mid = x0 + (x1 - x0) / 2
            except Exception:
                x_mid = x0

            try:
                y_row = max(1.035, 1.155 - 0.055 * min(level, 4))
                fig.add_annotation(
                    x=x_mid,
                    y=y_row,
                    xref="x",
                    yref="paper",
                    text=f"<b>{label}</b>",
                    showarrow=False,
                    xanchor="center",
                    yanchor="bottom",
                    bgcolor=CHART_LEGEND_BG,
                    bordercolor=note_col,
                    borderwidth=1.4,
                    font=dict(size=interval_font_size, color=note_col),
                )
            except Exception:
                pass
        return fig

    def add_manual_events_to_plotly(fig, features):
        fig = add_operation_intervals_to_plotly(fig, features)

        if not plot_events:
            return fig
        _, event_font_size = adaptive_note_font_sizes(total_note_count(), mobile=(chart_view_mode == "Mobile-friendly"))
        decorated_events = visible_events_for_notes(x_values=(filtered["plot_x"] if "plot_x" in filtered.columns else None))
        for idx, event in enumerate(decorated_events):
            x = event["plot_x"]
            label = event["label"]
            level = int(event.get("level", 0) or 0)
            note_col = note_color(idx + len(plot_intervals or []))
            try:
                fig.add_shape(
                    type="line",
                    x0=x,
                    x1=x,
                    y0=0,
                    y1=1,
                    xref="x",
                    yref="paper",
                    line=dict(color=note_col, width=2, dash="dash"),
                    opacity=0.75,
                )
            except Exception:
                pass
            try:
                y_note = max(1.015, 1.115 - 0.050 * min(level, 8))
                text_angle = 0
                x_anchor = "center"
                if event_label_style == "Vertical labels" or (event_label_style == "Auto staggered" and total_note_count() >= 3):
                    text_angle = -90
                    y_note = max(1.005, 1.105 - 0.045 * min(level, 8))
                    x_anchor = "right"
                elif event_label_style == "Compact top labels":
                    y_note = max(1.015, 1.115 - 0.045 * min(level, 8))
                    x_anchor = "center"
                fig.add_annotation(
                    x=x,
                    y=y_note,
                    xref="x",
                    yref="paper",
                    text=f"<b>{label}</b>",
                    showarrow=False,
                    xanchor=x_anchor,
                    yanchor="bottom",
                    textangle=text_angle,
                    font=dict(size=event_font_size, color=note_col),
                    bgcolor=CHART_LEGEND_BG,
                    bordercolor=note_col,
                    borderwidth=1.4,
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
        if mode == "Every 8 readings":
            return set(list(range(0, n, 8)) + [n - 1])
        if mode == "Every 20 readings":
            return set(list(range(0, n, 20)) + [n - 1])
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

    def build_text_and_positions(g, feature):
        if value_label_mode in ["Clean readable - recommended", "Hourly + min/max"]:
            idxs = report_label_indices(g.reset_index(drop=True), feature)
        else:
            idxs = label_indices(len(g), value_label_mode)

        values = numeric_feature_series(g, feature, reset_index=True)
        text = []
        pos = []
        position_cycle = ["top center", "bottom center", "middle right", "middle left"]
        for i, v in enumerate(values):
            text.append(format_plot_value(feature, v) if i in idxs else "")
            pos.append(position_cycle[i % len(position_cycle)])
        return text, pos

    def padded_range(df, feature):
        if feature in custom_y_ranges:
            return custom_y_ranges[feature]

        vals = numeric_feature_series(df, feature).dropna()
        if vals.empty:
            return None

        ymin = float(vals.min())
        ymax = float(vals.max())

        if ymin == ymax:
            pad = max(abs(ymin) * 0.08, 1.0)
        else:
            pad = (ymax - ymin) * 0.22

        return [ymin - pad, ymax + pad]

    def build_figure(df, features, mode):
        series_values = sorted(df["series_label"].dropna().astype(str).unique()) if "series_label" in df.columns else (
            sorted(df["well"].dropna().astype(str).unique()) if "well" in df.columns else ["All"]
        )
        line_mode = "lines+markers" if show_points else "lines"

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
                    for g in iter_plot_segments(g_all):
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
                            ),
                            secondary_y=secondary,
                        )
                        first_segment = False

            fig.update_layout(
                height=850 if chart_view_mode != "Mobile-friendly" else 680,
                width=None if chart_view_mode == "Mobile-friendly" else 1700,
                title=dict(text=chart_title_from_data(df, custom_chart_title), font=dict(size=28, color=CHART_TEXT, family="Segoe UI Semibold, Arial, sans-serif")),
                xaxis_title=x_axis_title,
                hovermode="x unified",
                margin=dict(l=85, r=90, t=185, b=80),
                plot_bgcolor=CHART_PLOT_BG,
                paper_bgcolor=CHART_PAPER_BG,
                font=dict(color=CHART_TEXT, size=15),
                legend=dict(font=dict(size=15, color=CHART_TEXT), bgcolor=CHART_LEGEND_BG, bordercolor=CHART_GRID, borderwidth=1),
                title_x=0.5,
                title_xanchor="center",
            )
            fig.update_xaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, tickfont=dict(size=14, color=CHART_TEXT), **axis_tick_settings)
            fig.update_yaxes(
                title_text=column_label(left_feature),
                secondary_y=False,
                showgrid=True,
                gridcolor=CHART_GRID_SOFT,
                zeroline=False,
                range=custom_y_ranges.get(left_feature),
            )
            fig.update_yaxes(
                title_text=column_label(right_feature),
                secondary_y=True,
                showgrid=False,
                zeroline=False,
                range=custom_y_ranges.get(right_feature),
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
                    for g in iter_plot_segments(g_all):
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
                hovermode="x unified",
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
            if not mobile_view:
                layout_kwargs["width"] = max(1400, min(3200 if wide_view else 2600, n_points * (52 if wide_view else 42)))
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
                    showgrid=True,
                    gridcolor=CHART_GRID,
                    zeroline=False,
                    showticklabels=True,
                    tickfont=dict(size=11 if chart_view_mode == "Mobile-friendly" else 15, color=CHART_TEXT),
                    title_text=x_axis_title if r == len(features) else "",
                    title_font=dict(size=16 if chart_view_mode == "Mobile-friendly" else 20, color=CHART_TEXT),
                    automargin=True,
                    **axis_tick_settings,
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
                for g in iter_plot_segments(g_all):
                    y = g[feature].astype(float)
                    fig.add_trace(
                        go.Scatter(
                            x=x_values(g),
                            y=y,
                            mode=line_mode,
                            name=plot_name,
                            legendgroup=plot_name,
                            showlegend=first_segment,
                            line=dict(color=color, width=2.5, shape=("hv" if feature in {"choke_unified", "choke_pct", "choke_size_64", "choke_ambiguous"} else "linear")),
                            marker=dict(color=color, size=7),
                        )
                    )
                    first_segment = False

        fig.update_layout(
            height=850,
            width=1700,
            title=dict(text=chart_title_from_data(df, custom_chart_title), font=dict(size=30, color=CHART_TEXT, family="Segoe UI Semibold, Arial, sans-serif")),
            yaxis_title=y_title,
            xaxis_title=x_axis_title,
            hovermode="x unified",
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
        fig.update_xaxes(
            showgrid=True,
            gridcolor=CHART_GRID,
            zeroline=False,
            title_font=dict(size=20, color=CHART_TEXT),
            tickfont=dict(size=15, color=CHART_TEXT),
            **axis_tick_settings,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor=CHART_GRID_SOFT,
            zeroline=False,
            title_font=dict(size=20, color=CHART_TEXT),
            tickfont=dict(size=15, color=CHART_TEXT),
        )
        if manual_events:
            for event in manual_events:
                fig.add_vline(x=event["datetime"], line_dash="dash", line_color=CHART_TEXT, line_width=2, opacity=0.75)
                fig.add_annotation(
                    x=event["datetime"],
                    y=1,
                    xref="x",
                    yref="paper",
                    text=event["label"],
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=13, color=CHART_TEXT),
                    bgcolor=CHART_LEGEND_BG,
                    bordercolor=CHART_TEXT,
                    borderwidth=1.4,
                )
        return fig

    def build_dual_axis_multi_figure(df, left_features, right_features, chart_name=""):
        """Build one combined chart with multiple left/right Y-axis features."""
        left_features = [f for f in left_features if f in df.columns]
        right_features = [f for f in right_features if f in df.columns]
        if not left_features or not right_features:
            return go.Figure()

        series_values = sorted(df["series_label"].dropna().astype(str).unique()) if "series_label" in df.columns else (
            sorted(df["well"].dropna().astype(str).unique()) if "well" in df.columns else ["All"]
        )
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
                for g in iter_plot_segments(g_all):
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
            width=None if chart_view_mode == "Mobile-friendly" else 1700,
            title=dict(text=chart_title_from_data(df, custom_chart_title) + suffix, font=dict(size=26, color=CHART_TEXT, family="Segoe UI Semibold, Arial, sans-serif")),
            xaxis_title=x_axis_title,
            hovermode="x unified",
            margin=dict(l=85, r=95, t=185, b=80),
            plot_bgcolor=CHART_PLOT_BG,
            paper_bgcolor=CHART_PAPER_BG,
            font=dict(color=CHART_TEXT, size=15),
            legend=dict(font=dict(size=13, color=CHART_TEXT), bgcolor=CHART_LEGEND_BG, bordercolor=CHART_GRID, borderwidth=1),
            title_x=0.5,
            title_xanchor="center",
        )
        fig.update_xaxes(showgrid=True, gridcolor=CHART_GRID, zeroline=False, tickfont=dict(size=13, color=CHART_TEXT), **axis_tick_settings)
        fig.update_yaxes(
            title_text=" / ".join(column_label(f) for f in left_features[:3]),
            secondary_y=False,
            showgrid=True,
            gridcolor=CHART_GRID_SOFT,
            zeroline=False,
        )
        fig.update_yaxes(
            title_text=" / ".join(column_label(f) for f in right_features[:3]),
            secondary_y=True,
            showgrid=False,
            zeroline=False,
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

    if dual_axis_charts:
        for cfg_i, cfg in enumerate(dual_axis_charts, start=1):
            st.markdown(f"### Combined secondary Y-axis chart {cfg_i}")
            dual_fig = build_dual_axis_multi_figure(filtered, cfg.get("left", []), cfg.get("right", []), cfg.get("title", ""))
            st.plotly_chart(dual_fig, use_container_width=True, config=plotly_config_common)

    fig = build_figure(filtered, selected_features, plot_mode)
    st.plotly_chart(fig, use_container_width=True, config=plotly_config_common)

    with st.expander("Filtered data used by current plot", expanded=False):
        st.caption("Filtered data = only the rows currently feeding the chart after your sidebar selections.")
        filtered_cols = ["source", "sheet", "well", "datetime", "time_text"] + selected_features
        filtered_cols = [c for c in filtered_cols if c in filtered.columns]
        display_filtered = filtered[filtered_cols].copy()
        if not show_internal_names:
            display_filtered = display_filtered.rename(columns={c: column_label(c) for c in display_filtered.columns})
        st.dataframe(display_filtered, use_container_width=True, height=280)

    render_section_title("Engineering Report Exports", "Prepare publication-ready PNG, PDF, and filtered-data outputs using the active chart configuration.")
    st.caption(f"Exports use the active {ACTIVE_THEME_NAME} theme. Prepare again after changing theme or chart settings; cached files are separated by theme.")

    def chart_label_indices_for_export(g, feature):
        """Use the same readable label logic for exports as the interactive chart."""
        g2 = g.reset_index(drop=True)
        n = len(g2)
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
        return {i for i in idxs if 0 <= i < n}

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
                    for g in iter_plot_segments(g_all):
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
                            ax.annotate(
                                format_plot_value(feature, y.iloc[i]),
                                (x.iloc[i], y.iloc[i]),
                                textcoords="offset points",
                                xytext=(0, 10 if i % 2 == 0 else -15),
                                ha="center",
                                fontsize=9.5,
                                color=color,
                                fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.12", fc=EXPORT_LABEL_BG, ec=EXPORT_LABEL_EDGE, alpha=0.65),
                            )
                        first_segment = False

                if feature in custom_y_ranges:
                    ax.set_ylim(custom_y_ranges[feature][0], custom_y_ranges[feature][1])
                else:
                    vals = numeric_feature_series(df, feature).dropna()
                    if not vals.empty:
                        ymin = float(vals.min())
                        ymax = float(vals.max())
                        pad = max((ymax - ymin) * 0.18, max(abs(ymax), 1) * 0.03, 0.5)
                        ax.set_ylim(ymin - pad, ymax + pad)

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
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=16))
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
                fig_m.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
                pdf.savefig(fig_m, facecolor=CHART_PAPER_BG, edgecolor=CHART_PAPER_BG)
                plt.close(fig_m)

        output.seek(0)
        return output.getvalue()

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
                        ax.annotate(
                            format_plot_value(feature, y.iloc[i]),
                            (x.iloc[i], y.iloc[i]),
                            textcoords="offset points",
                            xytext=(0, 11 if i % 2 == 0 else -16),
                            ha="center",
                            fontsize=11,
                            color=color,
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.12", fc=EXPORT_LABEL_BG, ec=EXPORT_LABEL_EDGE, alpha=0.7),
                        )

                if feature in custom_y_ranges:
                    ax.set_ylim(custom_y_ranges[feature][0], custom_y_ranges[feature][1])
                else:
                    vals = numeric_feature_series(df, feature).dropna()
                    if not vals.empty:
                        ymin = float(vals.min())
                        ymax = float(vals.max())
                        pad = max((ymax - ymin) * 0.18, max(abs(ymax), 1) * 0.03, 0.5)
                        ax.set_ylim(ymin - pad, ymax + pad)

                ax.set_ylabel(column_label(feature), fontsize=16, fontweight="bold")
                ax.set_xlabel(x_axis_title, fontsize=16, fontweight="bold")

                if plot_intervals:
                    ymin_i, ymax_i = ax.get_ylim()
                    y_span = ymax_i - ymin_i if ymax_i != ymin_i else 1.0
                    y_note = ymax_i - 0.04 * y_span
                    for interval in visible_intervals_for_notes():
                        x0 = interval["x0"]
                        x1 = interval["x1"]
                        ax.axvline(x0, color="#92400e", linestyle="--", linewidth=1.8, alpha=0.90)
                        ax.axvline(x1, color="#92400e", linestyle="--", linewidth=1.8, alpha=0.90)
                        try:
                            x_mid = x0 + (x1 - x0) / 2
                        except Exception:
                            x_mid = x0
                        try:
                            ax.annotate(
                                "",
                                xy=(x1, y_note),
                                xytext=(x0, y_note),
                                arrowprops=dict(arrowstyle="<->", color="#92400e", lw=1.8),
                            )
                        except Exception:
                            pass
                        ax.text(
                            x_mid,
                            y_note,
                            interval["label"],
                            fontsize=12,
                            fontweight="bold",
                            ha="center",
                            va="center",
                            color=CHART_TEXT,
                            bbox=dict(boxstyle="round,pad=0.22", fc=EXPORT_LABEL_BG, ec="#C98B3C", alpha=0.95),
                        )

                if plot_events:
                    for event in plot_events:
                        ax.axvline(event["plot_x"], color=CHART_TEXT, linestyle="--", linewidth=1.8, alpha=0.75)
                        ax.text(
                            event["plot_x"],
                            0.98,
                            event["label"],
                            transform=ax.get_xaxis_transform(),
                            rotation=90,
                            va="top",
                            ha="right",
                            fontsize=11,
                            color=CHART_TEXT,
                            bbox=dict(boxstyle="round,pad=0.15", fc=EXPORT_LABEL_BG, ec=CHART_GRID, alpha=0.75),
                        )
                ax.grid(True, which="major", alpha=0.28)
                ax.tick_params(axis="both", labelsize=13)

                if x_axis_mode == "Real calendar time" and "datetime" in df.columns and df["datetime"].notna().any():
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=16))
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
                fig_m.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])

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
            fmt = "%d-%b-%Y\n%H:%M" if (not dt_all.empty and (dt_all.dt.year.nunique() > 1 or (dt_all.max() - dt_all.min()).days >= 330)) else "%d-%b\n%H:%M"
            ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
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

    def _apply_matplotlib_notes(ax, x_values=None):
        """Apply interval and point notes with staggered rows to avoid overlap in exports."""
        _draw_all_test_separators_matplotlib(ax, filtered)
        # Interval notes: parent/long intervals on top row, child intervals below.
        note_count = total_note_count()
        interval_font_size, event_font_size = adaptive_note_font_sizes(note_count, mobile=(chart_view_mode == "Mobile-friendly"))
        if plot_intervals:
            for idx, interval in enumerate(visible_intervals_for_notes()):
                x0 = interval["x0"]
                x1 = interval["x1"]
                level = int(interval.get("level", 0))
                note_col = note_color(idx)
                ax.axvline(x0, color=note_col, linestyle="--", linewidth=1.8, alpha=0.90)
                ax.axvline(x1, color=note_col, linestyle="--", linewidth=1.8, alpha=0.90)
                try:
                    x_mid = x0 + (x1 - x0) / 2
                except Exception:
                    x_mid = x0
                y_frac = max(0.68, 0.96 - 0.10 * min(level, 3))
                try:
                    ax.annotate(
                        "",
                        xy=(x1, y_frac),
                        xytext=(x0, y_frac),
                        xycoords=("data", "axes fraction"),
                        textcoords=("data", "axes fraction"),
                        arrowprops=dict(arrowstyle="<->", color=note_col, lw=1.7),
                    )
                except Exception:
                    pass
                ax.text(
                    x_mid,
                    min(0.985, y_frac + 0.016),
                    interval["label"],
                    transform=ax.get_xaxis_transform(),
                    fontsize=interval_font_size,
                    fontweight="bold",
                    ha="center",
                    va="bottom",
                    color=note_col,
                    bbox=dict(boxstyle="round,pad=0.22", fc=EXPORT_LABEL_BG, ec=note_col, alpha=0.96),
                    clip_on=False,
                )

        # Point notes: stagger close labels on multiple rows.
        if plot_events:
            event_rows = visible_events_for_notes(x_values=x_values)
            base_frac = 0.80 if plot_intervals else 0.98
            for idx, event in enumerate(event_rows):
                level = int(event.get("level", 0))
                note_col = note_color(idx + len(plot_intervals or []))
                y_frac = max(0.12, base_frac - 0.060 * min(level, 12))
                ax.axvline(event["plot_x"], color=note_col, linestyle="--", linewidth=1.5, alpha=0.78)
                rotation = 90 if (event_label_style == "Vertical labels" or (event_label_style == "Auto staggered" and total_note_count() >= 3)) else 0
                ha = "right" if rotation else "center"
                try:
                    x_shift_points = float(event.get("x_shift_px", 0) or 0) * 0.5
                except Exception:
                    x_shift_points = 0
                ax.annotate(
                    event["label"],
                    xy=(event["plot_x"], y_frac),
                    xycoords=("data", "axes fraction"),
                    xytext=(x_shift_points, 0),
                    textcoords="offset points",
                    rotation=rotation,
                    va="top",
                    ha=ha,
                    fontsize=event_font_size,
                    color=note_col,
                    bbox=dict(boxstyle="round,pad=0.15", fc=EXPORT_LABEL_BG, ec=note_col, alpha=0.82),
                    clip_on=False,
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
                for g in iter_plot_segments(g_all):
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
                        ax.annotate(
                            format_plot_value(feature, y.iloc[i]),
                            (x.iloc[i], y.iloc[i]),
                            textcoords="offset points",
                            xytext=(0, 11 if i % 2 == 0 else -16),
                            ha="center",
                            fontsize=10.5,
                            color=color,
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.10", fc=EXPORT_LABEL_BG, ec=EXPORT_LABEL_EDGE, alpha=0.72),
                        )
                    first_segment = False

            if feature in custom_y_ranges:
                ax.set_ylim(custom_y_ranges[feature][0], custom_y_ranges[feature][1])
            else:
                vals = numeric_feature_series(df, feature).dropna()
                if not vals.empty:
                    ymin = float(vals.min())
                    ymax = float(vals.max())
                    pad = max((ymax - ymin) * 0.22, max(abs(ymax), 1) * 0.03, 0.5)
                    ax.set_ylim(ymin - pad, ymax + pad)

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
        fig_m.tight_layout(rect=[0.02, 0.02, 0.98, 0.975])

        output = io.BytesIO()
        if fmt == "pdf":
            fig_m.savefig(output, format="pdf", bbox_inches="tight", facecolor=CHART_PAPER_BG, edgecolor=CHART_PAPER_BG)
        else:
            fig_m.savefig(output, format="png", dpi=320, bbox_inches="tight", facecolor=CHART_PAPER_BG, edgecolor=CHART_PAPER_BG)
        plt.close(fig_m)
        output.seek(0)
        return output.getvalue()

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

                for g in iter_plot_segments(g_all):
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
                        ax.annotate(
                            format_plot_value(feature, y.iloc[i]),
                            (x.iloc[i], y.iloc[i]),
                            textcoords="offset points",
                            xytext=(0, 10 if i % 2 == 0 else -14),
                            ha="center",
                            fontsize=9.5,
                            color=color,
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.10", fc=EXPORT_LABEL_BG, ec=EXPORT_LABEL_EDGE, alpha=0.70),
                        )
                    first_segment = False

            vals = numeric_feature_series(df, feature).dropna()
            if not vals.empty:
                ymin = float(vals.min())
                ymax = float(vals.max())
                pad = max((ymax - ymin) * 0.22, max(abs(ymax), 1) * 0.03, 0.5)
                ax.set_ylim(ymin - pad, ymax + pad)

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
            fig_m.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
            output = io.BytesIO()
            fig_m.savefig(output, format="png", dpi=260, bbox_inches="tight", facecolor=CHART_PAPER_BG, edgecolor=CHART_PAPER_BG)
            plt.close(fig_m)
            output.seek(0)

            safe_feature = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in column_label(feature))[:80]
            outputs[f"tmu_{safe_feature}.png"] = output.getvalue()

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
