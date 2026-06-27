from __future__ import annotations

"""Adaptive, schema-tolerant tabular parser for production-test data.

This module is intentionally independent of the historical parser.  It is used
as a second parsing engine and is selected only when it produces a more
credible interpretation.  The design is source-first:

* detect tables rather than assuming one fixed header row;
* infer date/time from header meaning and column values;
* recognize common petroleum/production-test aliases and units;
* preserve unknown numeric channels instead of rejecting a new template;
* never silently overwrite source values to satisfy an engineering equation.
"""

import csv
import io
import math
import re
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ENGINE_ID = "v75-adaptive-fast-semantic-table-engine-20260627"

_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

META_FIELDS = {"datetime", "date", "time", "well", "note"}

# Labels are deliberately broader than the current UI.  Unknown numeric columns
# are also retained, so a new customer's sensor does not make a sheet unusable.
FIELD_LABELS: Dict[str, str] = {
    "choke_pct": "Choke Opening (%)",
    "choke_size_64": "Choke Size (/64 in)",
    "whp_psi": "WHP (psi)",
    "flp_psi": "FLP (psi)",
    "flow_press_psi": "Flow Pressure (psi)",
    "sep_p_psi": "Separator Pressure (psi)",
    "pumping_pressure_psi": "Pumping Pressure (psi)",
    "ct_pressure_psi": "CT Pressure (psi)",
    "casing_pressure_psi": "Casing Pressure (psi)",
    "tubing_pressure_psi": "Tubing Pressure (psi)",
    "annulus_pressure_psi": "Annulus Pressure (psi)",
    "line_pressure_psi": "Line Pressure (psi)",
    "us_press_psi": "Upstream Pressure (psi)",
    "ds_press_psi": "Downstream Pressure (psi)",
    "mpfm_press_psig": "MPFM Pressure (psig)",
    "dp_mbar": "Differential Pressure (mbar)",
    "gas_rate_mmscfd": "Total Gas Rate (MMSCF/D)",
    "gas_formation_mmscfd": "Formation Gas Rate (MMSCF/D)",
    "n2_rate_mmscfd": "N₂ Rate (MMSCF/D)",
    "n2_rate_scfm": "N₂ Rate (scf/min)",
    "oil_rate_stbd": "Oil Rate (STB/D)",
    "water_rate_bpd": "Water Rate (BBL/D)",
    "gross_rate_bpd": "Gross Rate (BBL/D)",
    "qoil_s_stbd": "QOil(S) (STB/D)",
    "qwat_s_bpd": "QWat(S) (BBL/D)",
    "qgas_s_mmscfd": "QGas(S) (MMSCF/D)",
    "qoil_a_bpd": "QOil(A) (BBL/D)",
    "qwat_a_bpd": "QWat(A) (BBL/D)",
    "qgas_a_mmcfd": "QGas(A) (MMCF/D)",
    "qgross_s_bpd": "QGross(S) (BBL/D)",
    "bsw_pct": "BS&W (%)",
    "wlr_s_pct": "WLR / Water Cut (%)",
    "gvf_a_pct": "GVF(A) (%)",
    "salinity_kppm": "Salinity (K ppm NaCl)",
    "oil_api": "Oil Gravity (API)",
    "oil_sg": "Oil Specific Gravity",
    "water_sg": "Water Specific Gravity",
    "gas_sg": "Gas Specific Gravity",
    "water_ph": "Water pH",
    "h2s_ppm": "H₂S (ppm)",
    "co2_mole_pct": "CO₂ (mole %)",
    "gor_scf_bbl": "GOR (scf/bbl)",
    "gor_s_scf_stb": "GOR(S) (scf/STB)",
    "flow_temp_c": "Flow Temperature (°C)",
    "flow_temp_f": "Flow Temperature (°F)",
    "sep_temp_c": "Separator Temperature (°C)",
    "sep_temp_f": "Separator Temperature (°F)",
    "gas_temp_c": "Gas Temperature (°C)",
    "gas_temp_f": "Gas Temperature (°F)",
    "oil_temp_c": "Oil Temperature (°C)",
    "oil_temp_f": "Oil Temperature (°F)",
    "pump_freq_hz": "Pump Frequency (Hz)",
    "drive_freq_hz": "Drive Frequency (Hz)",
    "motor_current_amp": "Motor Current (A)",
    "ama_current_amp": "AMA / Motor Current (A)",
    "pump_intake_pressure_psi": "Pi / Intake Pressure (psi)",
    "pump_discharge_pressure_psi": "Pd / Discharge Pressure (psi)",
    "intake_temp_c": "Intake Temperature (°C)",
    "intake_temp_f": "Intake Temperature (°F)",
    "motor_temp_c": "Motor Temperature (°C)",
    "motor_temp_f": "Motor Temperature (°F)",
    "motor_load_pct": "Motor Load (%)",
    "vibration_x": "Vibration X",
    "vibration_y": "Vibration Y",
    "vibration_z": "Vibration Z",
    "stroke_length_in": "Stroke Length (in)",
    "stroke_rate_spm": "Stroke Rate (SPM)",
    "peak_load_lbf": "Peak Load (lbf)",
    "minimum_load_lbf": "Minimum Load (lbf)",
    "ctu_weight_lbf": "CTU Weight (lbf)",
    "ctu_light_weight_lbf": "CTU Light Weight (lbf)",
    "ctu_wellhead_pressure_psi": "CTU Wellhead Pressure (psi)",
    "ctu_circulation_pressure_psi": "CTU Circulation Pressure (psi)",
    "ctu_reel_depth_ft": "CTU Reel Depth (ft)",
    "ctu_reel_speed_ftmin": "CTU Reel Speed (ft/min)",
    "ctu_fluid_rate_bpm": "CTU Fluid Rate (bpm)",
    "ctu_fluid_total_bbl": "CTU Fluid Total (bbl)",
    "ctu_n2_total_scf": "CTU N₂ Total (scf)",
}


def safe_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "nat", "none"} else text


def normalize(value: object) -> str:
    text = safe_text(value).lower()
    text = text.replace("₂", "2").replace("°", " deg ").replace("&", " and ")
    text = text.replace("\u200e", "").replace("\u200f", "").replace("\ufeff", "")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9%/+.-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def compact(value: object) -> str:
    return re.sub(r"[^a-z0-9%]+", "", normalize(value))


def slug(value: object) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", normalize(value)).strip("_")
    return out[:72] or "channel"


_NUMBER = re.compile(r"[-+]?(?:\d+(?:[,.]\d+)*|\.\d+)(?:[eE][-+]?\d+)?")


def number(value: object) -> float:
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
        return val if math.isfinite(val) else np.nan
    text = safe_text(value).replace("−", "-").replace("–", "-")
    if not text or normalize(text) in {"na", "n/a", "nil", "none", "-", "--"}:
        return np.nan
    match = _NUMBER.search(text)
    if not match:
        return np.nan
    try:
        val = float(match.group(0).replace(",", ""))
    except Exception:
        return np.nan
    return val if math.isfinite(val) else np.nan


def table_number(value: object) -> float:
    """Strict numeric parser for spreadsheet measurement cells.

    Numeric values and values followed by units are accepted. Operational text
    such as ``Pressure test 4000 psi`` is rejected instead of becoming 4000.
    """
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
        return val if math.isfinite(val) else np.nan
    text = safe_text(value).replace("−", "-").replace("–", "-").strip()
    if not text or text.startswith("#") or normalize(text) in {"na", "n/a", "nil", "none", "-", "--"}:
        return np.nan
    match = _NUMBER.search(text)
    if not match:
        return np.nan
    before = text[:match.start()].strip()
    after = text[match.end():].strip()
    # Only inequality signs may precede a measurement. Text before the number
    # means this is an operational sentence, not a numeric cell.
    if before and before not in {"<", ">", "<=", ">=", "~", "≈"}:
        return np.nan
    # Suffix may contain units/symbols, but no second numeric token or sentence.
    if _NUMBER.search(after):
        return np.nan
    if after and not re.fullmatch(r"[%A-Za-z°²³/_ .-]{0,32}", after):
        return np.nan
    try:
        val = float(match.group(0).replace(",", ""))
    except Exception:
        return np.nan
    return val if math.isfinite(val) else np.nan


def excel_datetime(value: object) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, date):
        ts = pd.Timestamp(datetime.combine(value, time.min))
    elif isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
        if not math.isfinite(val) or not 20000 <= val <= 80000:
            return pd.NaT
        ts = pd.Timestamp("1899-12-30") + pd.to_timedelta(val, unit="D")
    else:
        text = safe_text(value)
        if not text:
            return pd.NaT
        if re.fullmatch(r"\d+(?:\.\d+)?", text):
            val = float(text)
            if not 20000 <= val <= 80000:
                return pd.NaT
            ts = pd.Timestamp("1899-12-30") + pd.to_timedelta(val, unit="D")
        else:
            # Do not send ordinary operational text, formulas, or units through
            # pandas' general date parser. Wide field sheets contain thousands
            # of strings such as "CASING OPEN" and "#REF!"; probing all of them
            # is extremely slow and can make Streamlit appear frozen.
            if not re.search(
                r"(?:\b\d{1,4}[\-/]\d{1,2}[\-/]\d{1,4}\b|"
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b)",
                text, flags=re.I,
            ):
                return pd.NaT
            ts = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(ts):
        return pd.NaT
    ts = pd.Timestamp(ts)
    return ts if 1900 <= ts.year <= 2100 else pd.NaT


def time_fraction(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        return (value.hour * 3600 + value.minute * 60 + value.second) / 86400.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
        if not math.isfinite(val):
            return np.nan
        if 0 <= val < 2:
            return val % 1.0
        return np.nan
    text = safe_text(value).lower()
    if not text:
        return np.nan
    m = re.search(r"\b(\d{1,2})[:.](\d{2})(?::(\d{2}))?\s*(am|pm)?\b", text)
    if not m:
        return np.nan
    hour, minute, second = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    ap = m.group(4)
    if ap:
        hour %= 12
        if ap == "pm":
            hour += 12
    if hour > 23 or minute > 59 or second > 59:
        return np.nan
    return (hour * 3600 + minute * 60 + second) / 86400.0


def date_only(value: object) -> Optional[date]:
    ts = excel_datetime(value)
    return None if pd.isna(ts) else ts.date()


def combine_dt(date_value: object, time_value: object) -> pd.Timestamp:
    d = date_only(date_value)
    tf = time_fraction(time_value)
    if d is None or not math.isfinite(tf):
        return pd.NaT
    return pd.Timestamp(datetime.combine(d, time.min) + timedelta(seconds=round(tf * 86400) % 86400))


@dataclass(frozen=True)
class FieldGuess:
    key: Optional[str]
    confidence: float = 0.0
    unit: str = ""


def _temperature_key(base: str, header: str) -> str:
    h = normalize(header)
    if re.search(r"(?:deg\s*f|fahrenheit|\bf\b)", h):
        return base + "_f"
    return base + "_c"


def infer_field(header: object) -> FieldGuess:
    raw = safe_text(header)
    h = normalize(raw)
    # Treat field punctuation as spacing for labels such as W.H.P, Sep.P and
    # pumping.p while leaving slash-based units available.
    h = re.sub(r"[._]+", " ", h)
    h = re.sub(r"\s+", " ", h).strip()
    c = compact(raw)
    if not h:
        return FieldGuess(None)

    # Metadata and timing.
    if c in {"datetime", "dateandtime", "date/time", "timestamp", "timeanddate"} or "timestamp" in h:
        return FieldGuess("datetime", 1.0)
    if c in {"date", "testdate", "reportdate", "day"} or re.fullmatch(r"date(?:ddmmyy|ddmmyyyy|mmddyyyy)?", c):
        return FieldGuess("date", 0.98)
    if c in {"time", "testtime", "hhmm", "hhmmss"} or re.fullmatch(r"time(?:hhmmss?)?", c):
        return FieldGuess("time", 0.98)
    if re.search(r"\bwell(?: name| no| number| id)?\b", h) or c in {"well", "wellname", "wellno", "wellnumber"}:
        return FieldGuess("well", 0.98)
    if re.search(r"\b(?:comment|comments|event|events|remark|remarks|note|description)\b", h):
        return FieldGuess("note", 0.9)

    # SRP.
    if re.fullmatch(r"(?:sl|stroke length)(?: in| inch| inches)?", h):
        return FieldGuess("stroke_length_in", 0.99, "in")
    if re.fullmatch(r"(?:spm|stroke rate|strokes per minute)", h):
        return FieldGuess("stroke_rate_spm", 0.99, "spm")
    if re.search(r"\b(?:peak|max(?:imum)?)\s*load\b", h):
        return FieldGuess("peak_load_lbf", 0.98, "lbf")
    if re.search(r"\b(?:min|minimum)\s*load\b", h):
        return FieldGuess("minimum_load_lbf", 0.98, "lbf")

    # CTU/HMI.
    if re.search(r"\blt\s*weight\b|\blight\s*weight\b", h):
        return FieldGuess("ctu_light_weight_lbf", 0.95, "lbf")
    if re.search(r"\b(?:ctu\s*)?weight\b", h) and "load" not in h:
        return FieldGuess("ctu_weight_lbf", 0.85, "lbf")
    if "circulation pressure" in h:
        return FieldGuess("ctu_circulation_pressure_psi", 0.99, _pressure_unit(h))
    if "reel depth" in h:
        return FieldGuess("ctu_reel_depth_ft", 0.99, "ft")
    if "reel speed" in h:
        return FieldGuess("ctu_reel_speed_ftmin", 0.99, "ft/min")
    if re.search(r"\bfluid\s*(?:flow|rate)\b", h):
        return FieldGuess("ctu_fluid_rate_bpm", 0.95, "bpm")
    if "fluid total" in h:
        return FieldGuess("ctu_fluid_total_bbl", 0.95, "bbl")
    if re.search(r"\bn2\s*total\b|\bnitrogen\s*total\b", h):
        return FieldGuess("ctu_n2_total_scf", 0.95, "scf")

    # Choke.
    if "choke" in h or re.search(r"\bchk\b", h):
        if "%" in raw or "percent" in h or "pct" in h or "opening" in h:
            return FieldGuess("choke_pct", 0.98, "%")
        if "/64" in raw or "64" in h or "size" in h:
            return FieldGuess("choke_size_64", 0.96, "/64")
        return FieldGuess("choke_size_64", 0.7, "")

    # Pressures, most specific first.
    if re.search(r"\bpumping?\s*(?:p|press|pressure)\b|\bpump\s*pressure\b", h):
        return FieldGuess("pumping_pressure_psi", 0.99, _pressure_unit(h))
    if re.search(r"\bcasing\s*(?:p|press|pressure)\b", h):
        return FieldGuess("casing_pressure_psi", 0.97, _pressure_unit(h))
    if re.search(r"\btubing\s*(?:p|press|pressure)\b", h):
        return FieldGuess("tubing_pressure_psi", 0.97, _pressure_unit(h))
    if re.search(r"\b(?:annulus|annular|casing annulus)\s*(?:p|press|pressure)\b", h):
        return FieldGuess("annulus_pressure_psi", 0.97, _pressure_unit(h))
    if re.search(r"\bline\s*(?:p|press|pressure)\b", h):
        return FieldGuess("line_pressure_psi", 0.88, _pressure_unit(h))
    if re.search(r"\b(?:well\s*head|wellhead|whp|w\s*h\s*p|fthp)\b", h):
        if "ctu" in h:
            return FieldGuess("ctu_wellhead_pressure_psi", 0.99, _pressure_unit(h))
        return FieldGuess("whp_psi", 0.99, _pressure_unit(h))
    if re.search(r"\b(?:flp|flowline pressure|flow line pressure)\b", h):
        return FieldGuess("flp_psi", 0.98, _pressure_unit(h))
    if re.search(r"\b(?:flow press|flow pressure|downstream pressure)\b", h):
        return FieldGuess("flow_press_psi", 0.92, _pressure_unit(h))
    if re.search(r"\b(?:separator|sep)\s*(?:p|press|pressure)\b", h):
        return FieldGuess("sep_p_psi", 0.99, _pressure_unit(h))
    if re.search(r"\b(?:upstream|u/s|us)\s*(?:p|press|pressure)?\b", h):
        return FieldGuess("us_press_psi", 0.9, _pressure_unit(h))
    if re.search(r"\b(?:downstream|d/s|ds)\s*(?:p|press|pressure)?\b", h):
        return FieldGuess("ds_press_psi", 0.9, _pressure_unit(h))
    if re.search(r"\bmpfm\s*(?:p|press|pressure)\b", h) or ("press" in h and "psig" in h):
        return FieldGuess("mpfm_press_psig", 0.85, _pressure_unit(h))
    if re.search(r"\bct\s*(?:p|press|pressure)\b", h):
        return FieldGuess("ct_pressure_psi", 0.9, _pressure_unit(h))
    if re.search(r"\b(?:dp|differential pressure)\b", h):
        return FieldGuess("dp_mbar", 0.85, _dp_unit(h))

    # Explicit MPFM standard/actual rates.
    if re.search(r"q\s*oil\s*\(?s\)?|oil\s*standard", h):
        return FieldGuess("qoil_s_stbd", 0.99, _liquid_unit(h))
    if re.search(r"q\s*(?:wat|water)\s*\(?s\)?|water\s*standard", h):
        return FieldGuess("qwat_s_bpd", 0.99, _liquid_unit(h))
    if re.search(r"q\s*gas\s*\(?s\)?|gas\s*standard", h):
        return FieldGuess("qgas_s_mmscfd", 0.99, _gas_unit(h))
    if re.search(r"q\s*oil\s*\(?a\)?|oil\s*actual", h):
        return FieldGuess("qoil_a_bpd", 0.98, _liquid_unit(h))
    if re.search(r"q\s*(?:wat|water)\s*\(?a\)?|water\s*actual", h):
        return FieldGuess("qwat_a_bpd", 0.98, _liquid_unit(h))
    if re.search(r"q\s*gas\s*\(?a\)?|gas\s*actual", h):
        return FieldGuess("qgas_a_mmcfd", 0.98, _gas_unit(h))
    if re.search(r"q\s*gross\s*\(?s\)?", h):
        return FieldGuess("qgross_s_bpd", 0.99, _liquid_unit(h))

    # Main production rates.
    if re.search(r"\b(?:formation gas|gas formation|native gas)\b", h):
        return FieldGuess("gas_formation_mmscfd", 0.99, _gas_unit(h))
    if re.search(r"\b(?:n2|nitrogen)\b", h) and (re.search(r"\b(?:rate|flow|standard)\b", h) or _gas_unit(h) in {"mmscfd", "mscfd", "scfd", "scfm"}):
        unit = _gas_unit(h)
        return FieldGuess("n2_rate_scfm" if unit == "scfm" else "n2_rate_mmscfd", 0.98, unit)
    if re.search(r"\b(?:total gas|t gas|gas rate|qgas|gas q)\b", h) and "gor" not in h:
        return FieldGuess("gas_rate_mmscfd", 0.96, _gas_unit(h))
    if re.search(r"\b(?:oil rate|oil q|qoil|oil production|oil or cond)\b", h) or (re.search(r"\boil\b", h) and re.search(r"(?:bbl|stb|m3|/d|day)", h)):
        return FieldGuess("oil_rate_stbd", 0.97, _liquid_unit(h))
    if re.search(r"\b(?:water rate|water q|qwat|qwater|water production)\b", h) or (re.search(r"\bwater\b", h) and re.search(r"(?:bbl|stb|m3|/d|day)", h)):
        return FieldGuess("water_rate_bpd", 0.97, _liquid_unit(h))
    if re.search(r"\b(?:gross rate|gross liquid|total liquid|liquid rate|qgross)\b", h):
        return FieldGuess("gross_rate_bpd", 0.97, _liquid_unit(h))

    # Fluid properties.
    if re.search(r"\b(?:bsw|bs and w|water cut|wc)\b", h):
        return FieldGuess("bsw_pct", 0.98, "%")
    if re.search(r"\b(?:wlr|water liquid ratio)\b", h):
        return FieldGuess("wlr_s_pct", 0.98, "%")
    if re.search(r"\bgvf\b", h):
        return FieldGuess("gvf_a_pct", 0.97, "%")
    if "salinity" in h:
        return FieldGuess("salinity_kppm", 0.98, _salinity_unit(h))
    if re.search(r"\b(?:oil api|api gravity|deg api)\b", h) or c == "api":
        return FieldGuess("oil_api", 0.96, "api")
    if re.search(r"\boil\s*(?:sg|specific gravity)\b", h):
        return FieldGuess("oil_sg", 0.96, "sg")
    if re.search(r"\bwater\s*(?:sg|specific gravity)\b", h):
        return FieldGuess("water_sg", 0.96, "sg")
    if re.search(r"\bgas\s*(?:sg|specific gravity)\b", h):
        return FieldGuess("gas_sg", 0.96, "sg")
    if re.search(r"\b(?:water\s*)?ph\b", h):
        return FieldGuess("water_ph", 0.95, "ph")
    if re.search(r"\bh2s\b", h):
        return FieldGuess("h2s_ppm", 0.98, "ppm")
    if re.search(r"\bco2\b", h):
        return FieldGuess("co2_mole_pct", 0.98, "%")
    if re.search(r"\bgor\b", h):
        return FieldGuess("gor_s_scf_stb" if "(s" in h or "standard" in h else "gor_scf_bbl", 0.95, "scf/bbl")

    # Temperatures.
    if re.search(r"\bflow\s*(?:temp|temperature)|\bflt\b", h):
        key = _temperature_key("flow_temp", h)
        return FieldGuess(key, 0.96, "f" if key.endswith("_f") else "c")
    if re.search(r"\b(?:separator|sep)\s*(?:temp|temperature)\b", h):
        key = _temperature_key("sep_temp", h)
        return FieldGuess(key, 0.96, "f" if key.endswith("_f") else "c")
    if re.search(r"\bgas\s*(?:temp|temperature)\b", h):
        key = _temperature_key("gas_temp", h)
        return FieldGuess(key, 0.96, "f" if key.endswith("_f") else "c")
    if re.search(r"\boil\s*(?:temp|temperature)\b", h):
        key = _temperature_key("oil_temp", h)
        return FieldGuess(key, 0.96, "f" if key.endswith("_f") else "c")

    # ESP/device telemetry.
    if re.search(r"\b(?:run|pump|drive)\s*(?:freq|frequency)\b", h) or c in {"freq", "frequency", "hz"}:
        return FieldGuess("pump_freq_hz", 0.94, "hz")
    if re.search(r"\b(?:ama|motor current|current|amps?)\b", h):
        return FieldGuess("ama_current_amp" if "ama" in h else "motor_current_amp", 0.9, "amp")
    if re.search(r"\b(?:pi|intake pressure|pump intake)\b", h):
        return FieldGuess("pump_intake_pressure_psi", 0.92, _pressure_unit(h))
    if re.search(r"\b(?:pd|discharge pressure|pump discharge)\b", h):
        return FieldGuess("pump_discharge_pressure_psi", 0.92, _pressure_unit(h))
    if re.search(r"\b(?:ti|intake temp|intake temperature)\b", h):
        key = _temperature_key("intake_temp", h)
        return FieldGuess(key, 0.9, "f" if key.endswith("_f") else "c")
    if re.search(r"\b(?:tm|motor temp|motor temperature)\b", h):
        key = _temperature_key("motor_temp", h)
        return FieldGuess(key, 0.9, "f" if key.endswith("_f") else "c")
    if re.search(r"\bmotor load\b", h):
        return FieldGuess("motor_load_pct", 0.94, "%")
    if re.search(r"\b(?:vx|vibration x)\b", h):
        return FieldGuess("vibration_x", 0.92)
    if re.search(r"\b(?:vy|vibration y)\b", h):
        return FieldGuess("vibration_y", 0.92)
    if re.search(r"\b(?:vz|vibration z)\b", h):
        return FieldGuess("vibration_z", 0.92)

    return FieldGuess(None, 0.0, "")


def _pressure_unit(h: str) -> str:
    h = normalize(h)
    if "mpa" in h:
        return "mpa"
    if "kpa" in h:
        return "kpa"
    if re.search(r"\bbar\b", h):
        return "bar"
    return "psi"


def _dp_unit(h: str) -> str:
    h = normalize(h)
    if "inh2o" in compact(h) or "in h2o" in h:
        return "inh2o"
    if "pa" in h and "kpa" not in h and "mpa" not in h:
        return "pa"
    return "mbar"


def _gas_unit(h: str) -> str:
    h0 = normalize(h)
    c = compact(h0)
    # MM SCF/D and MM-SCF/D are MMSCF/D, not SCF/D.
    if "scfm" in c or re.search(r"scf\s*/?\s*min", h0):
        return "scfm"
    if any(token in c for token in ("mmscfd", "mmscfday", "mmscfd", "mmcfd", "mmcfday")):
        return "mmscfd"
    if re.search(r"\bmm\s*scf\s*/?\s*d", h0) or re.search(r"\bmm\s*cf\s*/?\s*d", h0):
        return "mmscfd"
    if "mscfd" in c or re.search(r"\bmscf\s*/?\s*d", h0):
        return "mscfd"
    if "scfd" in c or re.search(r"\bscf\s*/?\s*d", h0):
        return "scfd"
    return "mmscfd"


def _liquid_unit(h: str) -> str:
    h = normalize(h)
    if re.search(r"\b(?:m3|m 3|cubic meter)s?\s*/?\s*d", h):
        return "m3d"
    return "bpd"


def _salinity_unit(h: str) -> str:
    h = normalize(h)
    return "kppm" if "kppm" in compact(h) or "k ppm" in h else "ppm"


def convert_series(values: pd.Series, guess: FieldGuess) -> pd.Series:
    out = values.map(table_number).astype("float64")
    unit = guess.unit
    key = guess.key or ""
    if key.endswith("_psi"):
        if unit == "bar":
            out *= 14.5037738
        elif unit == "kpa":
            out *= 0.145037738
        elif unit == "mpa":
            out *= 145.037738
    elif key == "dp_mbar":
        if unit == "pa":
            out /= 100.0
        elif unit == "inh2o":
            out *= 2.490889
    elif key in {"gas_rate_mmscfd", "gas_formation_mmscfd", "n2_rate_mmscfd", "qgas_s_mmscfd", "qgas_a_mmcfd"}:
        if unit == "mscfd":
            out /= 1000.0
        elif unit == "scfd":
            out /= 1_000_000.0
        elif unit == "scfm":
            out *= 1440.0 / 1_000_000.0
    elif key in {"oil_rate_stbd", "water_rate_bpd", "gross_rate_bpd", "qoil_s_stbd", "qwat_s_bpd", "qoil_a_bpd", "qwat_a_bpd", "qgross_s_bpd"}:
        if unit == "m3d":
            out *= 6.28981077
    elif key == "salinity_kppm" and unit == "ppm":
        out /= 1000.0
    elif key in {"bsw_pct", "wlr_s_pct", "gvf_a_pct", "choke_pct", "motor_load_pct", "co2_mole_pct"}:
        finite = out.dropna()
        if len(finite) >= 3 and finite.abs().quantile(0.95) <= 1.05 and finite.abs().median() <= 1.0:
            out *= 100.0
    return out


# ---- XLSX sparse reader ---------------------------------------------------

def _col_number(ref: str) -> int:
    m = re.match(r"([A-Z]+)", ref.upper())
    if not m:
        return 0
    n = 0
    for char in m.group(1):
        n = n * 26 + ord(char) - 64
    return n


def _shared_strings(zf: zipfile.ZipFile) -> List[str]:
    if "xl/sharedStrings.xml" not in zf.namelist():
        return []
    root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
    return ["".join(node.text or "" for node in si.iter() if node.tag.endswith("}t"))
            for si in root.findall(f"{{{_XLSX_NS}}}si")]


def _sheet_paths(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    wb = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    targets = {r.attrib.get("Id"): r.attrib.get("Target", "")
               for r in rels.findall(f"{{{_PKG_REL_NS}}}Relationship")}
    result = []
    for sh in wb.findall(f".//{{{_XLSX_NS}}}sheet"):
        rid = sh.attrib.get(f"{{{_REL_NS}}}id")
        target = targets.get(rid, "")
        if not target:
            continue
        if target.startswith("/"):
            target = target.lstrip("/")
        elif not target.startswith("xl/"):
            target = "xl/" + target.lstrip("/")
        target = re.sub(r"(?:^|/)\.\./", "", target)
        result.append((sh.attrib.get("name", "Sheet"), target))
    return result


def _cell_value(cell: ET.Element, shared: Sequence[str]) -> object:
    typ = cell.attrib.get("t")
    if typ == "inlineStr":
        return "".join(n.text or "" for n in cell.iter() if n.tag.endswith("}t"))
    vnode = cell.find(f"{{{_XLSX_NS}}}v")
    if vnode is None:
        return None
    raw = vnode.text or ""
    if typ == "s":
        try:
            return shared[int(raw)]
        except Exception:
            return raw
    if typ in {"str", "e"}:
        return raw
    if typ == "b":
        return raw == "1"
    try:
        return float(raw)
    except Exception:
        return raw


def read_xlsx_sheets(data: bytes, max_rows: int = 50000, max_cols: int = 768,
                     max_cells: int = 900000) -> List[Tuple[str, pd.DataFrame]]:
    result: List[Tuple[str, pd.DataFrame]] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        shared = _shared_strings(zf)
        for sheet_name, path in _sheet_paths(zf):
            rows: Dict[int, Dict[int, object]] = {}
            count = 0
            with zf.open(path) as handle:
                for _, elem in ET.iterparse(handle, events=("end",)):
                    if not elem.tag.endswith("}c"):
                        continue
                    ref = elem.attrib.get("r", "")
                    m = re.match(r"[A-Z]+(\d+)", ref.upper())
                    if not m:
                        elem.clear(); continue
                    r = int(m.group(1)); c = _col_number(ref)
                    if r <= max_rows and 0 < c <= max_cols:
                        val = _cell_value(elem, shared)
                        if safe_text(val):
                            rows.setdefault(r, {})[c] = val
                            count += 1
                    elem.clear()
                    if count >= max_cells:
                        break
            if not rows:
                continue
            row_nums = sorted(rows)

            # Some field workbooks have a declared XFD used range and cached
            # formula/error values across thousands of columns, even though the
            # actual engineering table ends near column EG. Building and scoring
            # all of those columns made one small workbook take more than a
            # minute. Determine the useful table width from header/metadata text
            # and populated-column density, then ignore formula spillover.
            scan_row_limit = (row_nums[0] + 120) if row_nums else 120
            text_header_cols = []
            populated_counts: Dict[int, int] = {}
            for r, cols in rows.items():
                for c, value in cols.items():
                    populated_counts[c] = populated_counts.get(c, 0) + 1
                    txt = safe_text(value)
                    if r <= scan_row_limit and txt and not txt.startswith("#"):
                        # Header/metadata text normally contains letters. Numeric
                        # cached formula values do not establish a table width.
                        if re.search(r"[A-Za-z]", txt):
                            text_header_cols.append(c)

            if text_header_cols:
                header_width = max(text_header_cols)
                width = min(max_cols, max(32, header_width + 3))
            else:
                dense_cols = [c for c, n in populated_counts.items() if n >= 2]
                width = min(max_cols, max(dense_cols) if dense_cols else max(max(cols) for cols in rows.values()))

            trimmed_rows = {
                r: {c: value for c, value in cols.items() if c <= width}
                for r, cols in rows.items()
            }
            # Drop rows that only contained ignored far-right spillover.
            trimmed_rows = {r: cols for r, cols in trimmed_rows.items() if cols}
            if not trimmed_rows:
                continue
            row_nums = sorted(trimmed_rows)
            raw = pd.DataFrame(
                [[trimmed_rows[r].get(c) for c in range(1, width + 1)] for r in row_nums],
                index=row_nums, dtype=object,
            )
            raw.attrs["source_row_numbers"] = row_nums
            raw.attrs["trimmed_source_width"] = width
            result.append((sheet_name, raw))
    return result


@dataclass
class Candidate:
    start: int
    height: int
    headers: List[str]
    guesses: List[FieldGuess]
    dt_mode: Tuple[str, int, Optional[int]]
    score: float


def _combined_headers(raw: pd.DataFrame, start: int, height: int) -> List[str]:
    block = [raw.iloc[start + i].tolist() for i in range(height)]
    width = raw.shape[1]
    top_ffill: List[str] = []
    current = ""
    first_row_values = [safe_text(v) for v in block[0]]
    first_nonempty = [v for v in first_row_values if v]
    # Forward-fill only a genuine grouped header row. A single metadata cell
    # such as reference conditions must not be copied across 140 columns.
    use_parent_ffill = len(first_nonempty) >= 2
    for c in range(width):
        val = first_row_values[c]
        if val:
            current = val
        top_ffill.append(current if use_parent_ffill else val)
    headers = []
    for c in range(width):
        parts: List[str] = []
        seen = set()
        for r, row in enumerate(block):
            txt = safe_text(row[c])
            if r == 0 and not txt:
                txt = top_ffill[c]
            n = normalize(txt)
            if txt and n not in seen:
                seen.add(n); parts.append(txt)
        headers.append(" | ".join(parts))
    return headers


def _datetime_profile(series: pd.Series) -> Tuple[float, float, float]:
    sample = series.dropna().head(80)
    if sample.empty:
        return 0.0, 0.0, 0.0
    dt = sample.map(excel_datetime).notna().mean()
    tf = sample.map(time_fraction).notna().mean()
    date_ratio = sample.map(lambda x: date_only(x) is not None).mean()
    # Numeric times such as 0.5 should not be counted as full datetimes.
    full = 0.0
    for x in sample:
        ts = excel_datetime(x)
        if pd.notna(ts):
            if isinstance(x, (datetime, pd.Timestamp)) or re.search(r"\d{1,2}[:.]\d{2}", safe_text(x)):
                full += 1
            elif isinstance(x, (int, float, np.number)) and float(x) >= 20000 and float(x) % 1:
                full += 1
    return full / len(sample), date_ratio, tf


def _numeric_ratio(series: pd.Series) -> float:
    sample = series.dropna().head(100)
    return 0.0 if sample.empty else sample.map(table_number).notna().mean()


def _detect_dt_mode(raw: pd.DataFrame, data_start: int, guesses: List[FieldGuess]) -> Tuple[Tuple[str, int, Optional[int]], float]:
    sample = raw.iloc[data_start:min(len(raw), data_start + 100)]
    keys = [g.key for g in guesses]
    if "datetime" in keys:
        idx = keys.index("datetime")
        full, d, t = _datetime_profile(sample.iloc[:, idx])
        return (("datetime", idx, None), max(full, d) * 20 + 10)
    if "date" in keys and "time" in keys:
        di, ti = keys.index("date"), keys.index("time")
        _, dr, _ = _datetime_profile(sample.iloc[:, di])
        _, _, tr = _datetime_profile(sample.iloc[:, ti])
        return (("date_time", di, ti), (dr + tr) * 12 + 12)
    if "date" in keys:
        idx = keys.index("date")
        full, dr, tr = _datetime_profile(sample.iloc[:, idx])
        return (("date", idx, None), max(full, dr) * 18 + tr * 4 + 6)
    if "time" in keys:
        idx = keys.index("time")
        _, _, tr = _datetime_profile(sample.iloc[:, idx])
        return (("time", idx, None), tr * 18 + 4)

    # Content-driven fallback for unfamiliar headers.
    profiles = [_datetime_profile(sample.iloc[:, i]) for i in range(sample.shape[1])]
    full_candidates = [(p[0], i) for i, p in enumerate(profiles) if p[0] >= 0.45]
    if full_candidates:
        ratio, idx = max(full_candidates)
        return (("datetime", idx, None), ratio * 18)
    dates = [(p[1], i) for i, p in enumerate(profiles) if p[1] >= 0.55]
    times = [(p[2], i) for i, p in enumerate(profiles) if p[2] >= 0.55]
    if dates and times:
        dr, di = max(dates); tr, ti = max(times)
        if di != ti:
            return (("date_time", di, ti), (dr + tr) * 8)
    if times:
        tr, ti = max(times)
        return (("time", ti, None), tr * 10)
    return (("none", -1, None), -20.0)


def find_candidates(raw: pd.DataFrame) -> List[Candidate]:
    if raw.empty:
        return []
    scan = min(len(raw), 160)
    candidates: List[Candidate] = []

    # Do not brute-force every row in wide petroleum workbooks. Candidate header
    # blocks almost always occur near the top or around an explicit Date/Time
    # marker. Include a small window above each timing row so merged parent
    # headers are still detected.
    probable_starts = set(range(min(scan, 12))) if raw.shape[1] <= 48 else set()
    for r in range(scan):
        row_text = " | ".join(normalize(v) for v in raw.iloc[r].tolist() if safe_text(v))
        if re.search(r"\b(?:date|time|timestamp|hh:mm|dd/mm|d/mm)\b", row_text):
            probable_starts.update(range(max(0, r - 3), min(scan, r + 2)))

    infer_cache: Dict[str, FieldGuess] = {}
    for start in sorted(probable_starts):
        for height in (1, 2, 3, 4):
            if start + height >= len(raw):
                continue
            headers = _combined_headers(raw, start, height)
            guesses = []
            for header in headers:
                key = safe_text(header)
                if key not in infer_cache:
                    infer_cache[key] = infer_field(header)
                guesses.append(infer_cache[key])
            known = [g for g in guesses if g.key and g.key not in META_FIELDS]
            guessed_keys = {g.key for g in guesses if g.key}
            header_joined = " | ".join(normalize(h) for h in headers if safe_text(h))
            has_timing_hint = bool(guessed_keys & {"datetime", "date", "time"}) or bool(re.search(r"\b(?:date|time|timestamp|hh:mm)\b", header_joined))
            if not has_timing_hint and not (raw.shape[1] <= 48 and start < 12 and height == 1):
                continue
            dt_mode, dt_score = _detect_dt_mode(raw, start + height, guesses)
            if dt_mode[0] == "none":
                continue
            sample = raw.iloc[start + height:min(len(raw), start + height + 50)]
            # Profile only columns that have a header or data. This avoids
            # thousands of regex conversions on cached blank/formula columns.
            active_indices = [
                i for i, h in enumerate(headers)
                if safe_text(h) or sample.iloc[:, i].notna().any()
            ]
            known_indices = [
                i for i, guess in enumerate(guesses)
                if guess.key and guess.key not in META_FIELDS
            ]
            # When semantic headers are already recognized, profile only those
            # columns for candidate scoring. Scanning every calculation/helper
            # column on every candidate was the main cost in wide TMU sheets.
            profile_indices = known_indices if known_indices else active_indices
            num_cols = sum(_numeric_ratio(sample.iloc[:, i]) >= 0.45 for i in profile_indices)
            if not known and num_cols < 2:
                continue
            unit_bonus = sum(bool(re.search(r"psi|bar|mmscf|scf|bbl|stb|ppm|%|hz|lbf|spm|/64", normalize(headers[i]))) for i in active_indices)
            duplicate_penalty = len([g.key for g in known]) - len(set(g.key for g in known))
            score = dt_score + len(set(g.key for g in known)) * 5 + min(num_cols, 20) * 1.3 + min(unit_bonus, 12) - duplicate_penalty * 2
            candidates.append(Candidate(start, height, headers, guesses, dt_mode, score))
    candidates.sort(key=lambda c: c.score, reverse=True)
    selected: List[Candidate] = []
    for c in candidates:
        if any(abs(c.start - x.start) <= 3 for x in selected):
            continue
        selected.append(c)
        if len(selected) >= 6:
            break
    return selected


def _date_from_text(text: str) -> Optional[date]:
    for m in re.finditer(r"(?<!\d)(\d{1,2})[\-/](\d{1,2})[\-/](20\d{2}|\d{2})(?!\d)", text):
        d, mo, y = map(int, m.groups())
        if y < 100:
            y += 2000
        try:
            return date(y, mo, d)
        except Exception:
            pass
    return None


def _default_date(raw: pd.DataFrame, source_name: str, sheet_name: str) -> Optional[date]:
    for r in range(min(len(raw), 50)):
        joined = " | ".join(safe_text(v) for v in raw.iloc[r].tolist())
        d = _date_from_text(joined)
        if d:
            return d
    return _date_from_text(f"{source_name} {sheet_name}")


def _clean_well(value: object) -> str:
    text = safe_text(value).upper().strip()
    text = re.sub(r"\b(?:WELL|WELL NAME|WELL NO|NO)\b\s*[:#-]?\s*", "", text)
    text = re.sub(r"[^A-Z0-9-]+", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "Unknown"


def infer_well(raw: pd.DataFrame, source_name: str, sheet_name: str) -> str:
    for r in range(min(len(raw), 60)):
        vals = raw.iloc[r].tolist()
        for c, value in enumerate(vals):
            h = normalize(value)
            if re.fullmatch(r"well(?: name| no| number| id)?", h):
                for j in range(c + 1, min(len(vals), c + 4)):
                    candidate = _clean_well(vals[j])
                    if candidate != "Unknown":
                        return candidate
            m = re.search(r"\bwell(?: name| no| number)?\s*[:#-]\s*([a-z0-9][a-z0-9 -]{1,30})", safe_text(value), flags=re.I)
            if m:
                return _clean_well(m.group(1))
    # File/sheet patterns: require at least one digit to avoid words like DATA.
    text = f"{sheet_name} {Path(source_name).stem}"
    patterns = [
        r"\b([A-Z]{1,6}\s*[- ]?\s*[A-Z]?\d+(?:\s*[- ]\s*\d+)*)\b",
        r"\b(\d+\s*[- ]\s*[A-Z]?\d+(?:\s*[- ]\s*\d+)*)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text.upper()):
            candidate = _clean_well(m.group(1))
            if candidate not in {"UNKNOWN", "SHEET1", "DATA"} and re.search(r"\d", candidate):
                return candidate
    return "Unknown"


def _build_datetimes(raw_data: pd.DataFrame, mode: Tuple[str, int, Optional[int]], default_date: Optional[date]) -> pd.Series:
    kind, a, b = mode
    if kind == "datetime":
        out = raw_data.iloc[:, a].map(excel_datetime)
    elif kind == "date_time":
        out = pd.Series([combine_dt(d, t) for d, t in zip(raw_data.iloc[:, a], raw_data.iloc[:, int(b)])], index=raw_data.index)
    elif kind == "date":
        out = raw_data.iloc[:, a].map(excel_datetime)
    elif kind == "time":
        if default_date is None:
            return pd.Series(pd.NaT, index=raw_data.index, dtype="datetime64[ns]")
        current = default_date
        last_tf = np.nan
        values = []
        for value in raw_data.iloc[:, a]:
            tf = time_fraction(value)
            if not math.isfinite(tf):
                values.append(pd.NaT); continue
            if math.isfinite(last_tf) and tf + 0.25 < last_tf:
                current = current + timedelta(days=1)
            values.append(pd.Timestamp(datetime.combine(current, time.min) + timedelta(seconds=round(tf * 86400) % 86400)))
            last_tf = tf
        out = pd.Series(values, index=raw_data.index)
    else:
        out = pd.Series(pd.NaT, index=raw_data.index, dtype="datetime64[ns]")
    return pd.to_datetime(out, errors="coerce")


def parse_candidate(raw: pd.DataFrame, candidate: Candidate, source_name: str, sheet_name: str) -> pd.DataFrame:
    data = raw.iloc[candidate.start + candidate.height:].copy()
    if data.empty:
        return pd.DataFrame()
    default_date = _default_date(raw, source_name, sheet_name)
    datetimes = _build_datetimes(data, candidate.dt_mode, default_date)

    # Use recognized engineering channels to decide which rows are readings.
    # This prevents cached calculation cells in far-right template columns from
    # turning operational text/event rows into false measurement points.
    numeric_profiles = [_numeric_ratio(data.iloc[:, i]) for i in range(data.shape[1])]
    dt_positions = {candidate.dt_mode[1], candidate.dt_mode[2]}
    primary_measurement_fields = {
        "choke_pct", "choke_size_64", "whp_psi", "flp_psi", "flow_press_psi",
        "sep_p_psi", "pumping_pressure_psi", "ct_pressure_psi", "casing_pressure_psi",
        "tubing_pressure_psi", "annulus_pressure_psi", "line_pressure_psi",
        "gas_rate_mmscfd", "gas_formation_mmscfd", "n2_rate_mmscfd", "n2_rate_scfm",
        "oil_rate_stbd", "water_rate_bpd", "gross_rate_bpd",
        "qoil_s_stbd", "qwat_s_bpd", "qgas_s_mmscfd", "qoil_a_bpd",
        "qwat_a_bpd", "qgas_a_mmcfd", "qgross_s_bpd",
        "pump_freq_hz", "drive_freq_hz", "motor_current_amp", "ama_current_amp",
        "pump_intake_pressure_psi", "pump_discharge_pressure_psi",
        "stroke_length_in", "stroke_rate_spm", "peak_load_lbf", "minimum_load_lbf",
        "ctu_weight_lbf", "ctu_light_weight_lbf", "ctu_wellhead_pressure_psi",
        "ctu_circulation_pressure_psi", "ctu_reel_depth_ft", "ctu_reel_speed_ftmin",
        "ctu_fluid_rate_bpm", "ctu_fluid_total_bbl", "ctu_n2_total_scf",
    }
    known_measurement_cols = [
        i for i, (guess, ratio) in enumerate(zip(candidate.guesses, numeric_profiles))
        if i not in dt_positions and guess.key in primary_measurement_fields and ratio >= 0.10
    ]
    measurement_cols = known_measurement_cols or [
        i for i, ratio in enumerate(numeric_profiles) if ratio >= 0.25 and i not in dt_positions
    ]
    valid_measurements = pd.Series(False, index=data.index)
    for i in measurement_cols:
        valid_measurements |= data.iloc[:, i].map(table_number).notna()
    valid = datetimes.notna() & valid_measurements
    if valid.sum() < 2:
        return pd.DataFrame()
    # Keep the full valid span; isolated event rows without readings remain notes elsewhere.
    positions = np.flatnonzero(valid.to_numpy())
    lo, hi = int(positions.min()), int(positions.max())
    data = data.iloc[lo:hi + 1]
    datetimes = datetimes.iloc[lo:hi + 1]
    valid = valid.iloc[lo:hi + 1]
    data = data.loc[valid]
    datetimes = datetimes.loc[valid]

    out = pd.DataFrame(index=data.index)
    out["datetime"] = datetimes
    keys_used: Dict[str, Tuple[int, float]] = {}
    unknown_headers: List[str] = []
    dt_indices = {candidate.dt_mode[1], candidate.dt_mode[2]}

    for i, (header, guess) in enumerate(zip(candidate.headers, candidate.guesses)):
        if i in dt_indices or guess.key in {"datetime", "date", "time"}:
            continue
        if guess.key == "well":
            out["well"] = data.iloc[:, i].map(_clean_well)
            continue
        if guess.key == "note":
            out["note"] = data.iloc[:, i].map(safe_text)
            continue
        ratio = _numeric_ratio(data.iloc[:, i])
        if ratio < 0.25:
            continue
        if guess.key:
            prev = keys_used.get(guess.key)
            if prev and prev[1] >= guess.confidence:
                raw_key = f"raw_{slug(header)}"
                out[raw_key] = data.iloc[:, i].map(table_number)
                unknown_headers.append(header)
                continue
            if prev:
                old_i = prev[0]
                old_header = candidate.headers[old_i]
                out[f"raw_{slug(old_header)}"] = out[guess.key]
                unknown_headers.append(old_header)
            out[guess.key] = convert_series(data.iloc[:, i], guess)
            keys_used[guess.key] = (i, guess.confidence)
        else:
            key = f"raw_{slug(header)}"
            suffix = 2
            while key in out.columns:
                key = f"raw_{slug(header)}_{suffix}"; suffix += 1
            out[key] = data.iloc[:, i].map(table_number)
            unknown_headers.append(header)

    source_rows = raw.attrs.get("source_row_numbers")
    if source_rows:
        out["source_row"] = [source_rows[int(pos)] if int(pos) < len(source_rows) else int(pos) + 1 for pos in out.index]
    else:
        out["source_row"] = [int(i) + 1 for i in out.index]
    if "well" not in out or out["well"].replace("Unknown", np.nan).notna().sum() == 0:
        out["well"] = infer_well(raw, source_name, sheet_name)
    else:
        fallback = infer_well(raw, source_name, sheet_name)
        out["well"] = out["well"].replace({"": np.nan, "Unknown": np.nan}).fillna(fallback)
    if "note" not in out:
        out["note"] = ""
    out["source"] = source_name
    out["sheet"] = sheet_name
    out["source_type"] = "tabular"
    out["test_unit"] = sheet_name
    out["parser_engine"] = ENGINE_ID
    out["parse_confidence"] = min(1.0, max(0.35, candidate.score / 100.0))
    out["unmapped_columns"] = "; ".join(dict.fromkeys(safe_text(h) for h in unknown_headers if safe_text(h)))
    out["date"] = out["datetime"].dt.date
    out["time_text"] = out["datetime"].dt.strftime("%H:%M")
    return out.reset_index(drop=True)


def parse_raw_sheet(raw: pd.DataFrame, source_name: str, sheet_name: str) -> List[pd.DataFrame]:
    tables = []
    for candidate in find_candidates(raw):
        table = parse_candidate(raw, candidate, source_name, sheet_name)
        if not table.empty:
            tables.append(table)
    # Remove overlapping interpretations. Keep the richest/highest-confidence one.
    tables.sort(key=lambda d: (len(d), len([c for c in d.columns if c not in {"source", "sheet", "well", "date", "time_text", "datetime", "note", "test_unit", "source_type", "parser_engine"}]), float(d["parse_confidence"].iloc[0])), reverse=True)
    selected: List[pd.DataFrame] = []
    for table in tables:
        start, end = table["datetime"].min(), table["datetime"].max()
        if any(t["sheet"].iloc[0] == table["sheet"].iloc[0] and
               max(start, t["datetime"].min()) <= min(end, t["datetime"].max()) for t in selected):
            continue
        selected.append(table)
    return selected


def parse_xlsx(data: bytes, name: str) -> List[pd.DataFrame]:
    tables: List[pd.DataFrame] = []
    for sheet_name, raw in read_xlsx_sheets(data):
        tables.extend(parse_raw_sheet(raw, name, sheet_name))
    return tables


def _decode(data: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            if "\x00" in text and not enc.startswith("utf-16"):
                continue
            return text.replace("\x00", "")
        except Exception:
            pass
    return data.decode("latin-1", errors="replace")


def parse_delimited(data: bytes, name: str) -> List[pd.DataFrame]:
    text = _decode(data)
    sample = "\n".join(x for x in text.splitlines()[:50] if x.strip())
    try:
        sep = csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        sep = max((",", ";", "\t", "|"), key=sample.count)
    raw = pd.read_csv(io.StringIO(text), sep=sep, header=None, dtype=object, engine="python", on_bad_lines="skip")
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    raw.index = range(1, len(raw) + 1)
    return parse_raw_sheet(raw, name, "CSV")


def parse_file(data: bytes, name: str) -> List[pd.DataFrame]:
    suffix = Path(name).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return parse_xlsx(data, name)
    if suffix in {".csv", ".tsv"}:
        return parse_delimited(data, name)
    return []


def interpretation_score(df: pd.DataFrame) -> float:
    if df is None or df.empty or "datetime" not in df:
        return -1e9
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    valid_dt = dt.notna().mean()
    if valid_dt < 0.5:
        return -1e6
    numeric_cols = []
    for c in df.columns:
        if c in {"source", "sheet", "well", "date", "time", "time_text", "datetime", "note", "test_unit", "source_type", "parser_engine", "unmapped_columns", "data_quality_note", "rejected_values", "review_required", "link_status", "test_id"}:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().sum() >= 2:
            numeric_cols.append(c)
    known = [c for c in numeric_cols if c in FIELD_LABELS]
    raw = [c for c in numeric_cols if c.startswith("raw_")]
    score = valid_dt * 35 + min(len(df), 100) * 0.25 + len(known) * 4 + min(len(raw), 8) * 1.5
    if dt.is_monotonic_increasing:
        score += 8
    years = dt.dropna().dt.year
    if not years.empty and years.between(1990, 2100).all():
        score += 5
    elif not years.empty:
        score -= 80
    # Physical plausibility is used only to choose between interpretations,
    # never to delete source data.
    for c in ("bsw_pct", "wlr_s_pct", "gvf_a_pct", "choke_pct"):
        if c in df:
            vals = pd.to_numeric(df[c], errors="coerce")
            score -= ((vals < -0.1) | (vals > 100.5)).mean() * 40
    if all(c in df for c in ("oil_rate_stbd", "water_rate_bpd", "gross_rate_bpd")):
        oil = pd.to_numeric(df["oil_rate_stbd"], errors="coerce")
        wat = pd.to_numeric(df["water_rate_bpd"], errors="coerce")
        gross = pd.to_numeric(df["gross_rate_bpd"], errors="coerce")
        comparable = oil.notna() & wat.notna() & gross.notna() & (gross.abs() > 1)
        if comparable.any():
            rel = (gross - oil - wat).abs() / gross.abs().clip(lower=1)
            score -= (rel[comparable] > 0.25).mean() * 20
    if "review_required" in df:
        score -= df["review_required"].fillna(False).astype(bool).mean() * 15
    if "parse_confidence" in df:
        score += pd.to_numeric(df["parse_confidence"], errors="coerce").fillna(0).mean() * 5
    return float(score)
