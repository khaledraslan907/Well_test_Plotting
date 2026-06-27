from __future__ import annotations

"""Unified robust ingestion layer for the TMU dashboard.

This module intentionally keeps the mature OCR/PDF/WhatsApp ZIP logic from the
previous parser in ``tmu_parser_legacy.py`` and replaces the fragile tabular
loading path with one deterministic pipeline for Excel, CSV and pasted reports.

Design goals:
- Never trust an Excel declared used range.
- Never infer a date from ordinary process values.
- Accept production-test and device-only time series.
- Preserve unknown numeric channels instead of rejecting a file.
- Normalize equivalent well names and merge repeated/incomplete reports safely.
- Keep source values auditable while presenting stable canonical feature names.
"""

import csv
import io
import math
import re
import zipfile
import warnings
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

import tmu_parser_legacy as legacy

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning, module=r"tmu_parser_legacy")

PARSER_BUILD_ID_V66 = "v66-unified-robust-ingestion-20260622"
PARSER_BUILD_ID_V67 = "v67-cumulative-excel-time-sequence-fix-20260623"
PARSER_BUILD_ID = PARSER_BUILD_ID_V67

# ---------------------------------------------------------------------------
# Stable schema and labels
# ---------------------------------------------------------------------------

COLUMN_LABELS: Dict[str, str] = dict(getattr(legacy, "COLUMN_LABELS", {}))
COLUMN_LABELS.update(
    {
        "gas_rate_status": "Gas Rate Status",
        "source_priority": "Source Priority",
        "source_row": "Source Row",
        "source_group": "Source Group",
        "data_quality_note": "Engineering Check",
        "stroke_length_in": "Stroke Length (in)",
        "stroke_rate_spm": "Stroke Rate (SPM)",
        "peak_load_lbf": "Peak Load (lbf)",
        "minimum_load_lbf": "Minimum Load (lbf)",
    }
)

BASE_NON_PLOT_COLS = {
    "source", "sheet", "well", "date", "time", "time_text", "datetime",
    "note", "test_unit", "test_id", "test_sequence", "source_type",
    "link_status", "review_required", "message_index", "source_priority",
    "source_row", "source_group", "data_quality_note", "_well_key",
    "_minute_key", "_source_order", "_row_completeness",
}

CANONICAL_NUMERIC_FIELDS = {
    key for key in COLUMN_LABELS
    if key not in BASE_NON_PLOT_COLS and key not in {"gas_rate_status"}
}

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

class UploadedBytes(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def _uploaded_bytes(uploaded_file) -> tuple[bytes, str]:
    name = str(getattr(uploaded_file, "name", "uploaded_file"))
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    else:
        data = bytes(uploaded_file)
    if not isinstance(data, (bytes, bytearray)):
        data = bytes(data)
    return bytes(data), name


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


def normalize_text(value: object) -> str:
    text = safe_text(value).lower()
    text = text.replace("\u200e", "").replace("\u200f", "").replace("\ufeff", "")
    text = text.replace("&", " and ").replace("°", " deg ")
    text = re.sub(r"[*_`~]+", "", text)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"[^a-z0-9%/+.-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: object) -> str:
    text = normalize_text(value)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text[:80] or "column"


def append_note(existing: object, addition: object) -> str:
    parts: List[str] = []
    for raw in (existing, addition):
        text = safe_text(raw)
        if not text:
            continue
        for item in re.split(r"\s*;\s*", text):
            item = item.strip()
            if item and item not in parts:
                parts.append(item)
    return "; ".join(parts)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return normalize_text(value) in {
            "", "n/a", "na", "nil", "none", "not available", "#n/a", "#ref!",
            "#div/0!", "#value!", "-", "--",
        }
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


_NUMBER_RE = re.compile(
    r"(?<![A-Za-z0-9])[-+]?(?:\d+(?:[,.]\d+)*|\.\d+)(?:[eE][-+]?\d+)?"
)


def extract_number(value: object) -> float:
    """Parse normal, comma-formatted and scientific-notation values safely."""
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else np.nan
    text = safe_text(value)
    if not text or normalize_text(text) in {
        "n/a", "na", "nil", "none", "not available", "low gas", "trace gas",
    }:
        return np.nan
    match = _NUMBER_RE.search(text.replace("−", "-").replace("–", "-"))
    if not match:
        return np.nan
    token = match.group(0).replace(",", "")
    try:
        number = float(token)
    except ValueError:
        return np.nan
    return number if math.isfinite(number) else np.nan



def extract_tabular_number(value: object) -> float:
    """Strict numeric parser for spreadsheet cells.

    Accepts numeric values and strings such as ``200 PSI`` or ``9.8E-2`` but
    rejects operational notes like ``Pressure test 3000 psi``.
    """
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else np.nan
    text = safe_text(value)
    if not text or normalize_text(text) in {"n/a", "na", "nil", "none", "not available"}:
        return np.nan
    match = _NUMBER_RE.search(text.replace("−", "-").replace("–", "-"))
    if not match:
        return np.nan
    remainder = (text[:match.start()] + " " + text[match.end():]).lower()
    remainder = remainder.replace("°", " ")
    remainder = re.sub(r"[()\[\]{}\'\".,;:+*/_-]+", " ", remainder)
    words = [w for w in re.findall(r"[a-z]+", remainder) if w]
    allowed = {
        "psi", "psig", "psia", "bar", "kpa", "mpa", "bbl", "bpd", "stb",
        "d", "day", "mmscf", "mmscfd", "mscf", "scf", "ppm", "kppm",
        "nacl", "mole", "mol", "hz", "amp", "amps", "a", "c", "f",
        "deg", "in", "inch", "inches", "api", "percent", "pct", "mm",
        "cm", "m", "ft", "min", "hr", "hours", "factor", "air",
    }
    if any(word not in allowed for word in words):
        return np.nan
    try:
        number = float(match.group(0).replace(",", ""))
    except ValueError:
        return np.nan
    return number if math.isfinite(number) else np.nan


def clean_tabular_numeric_series(series: pd.Series) -> pd.Series:
    return series.map(extract_tabular_number).astype("float64")

def clean_numeric_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return series.map(extract_number).astype("float64")


def _safe_datetime_scalar(value: object, *, dayfirst: bool = True) -> pd.Timestamp:
    if value is None or _is_missing(value):
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        ts = value
    elif isinstance(value, datetime):
        ts = pd.Timestamp(value)
    elif isinstance(value, date):
        ts = pd.Timestamp(datetime.combine(value, time.min))
    elif isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        # Excel serial dates only. Ordinary process readings are excluded.
        if not math.isfinite(number) or not (20000 <= number <= 80000):
            return pd.NaT
        ts = pd.Timestamp("1899-12-30") + pd.to_timedelta(number, unit="D")
    else:
        text = safe_text(value)
        if not text:
            return pd.NaT
        # Reject bare numeric strings as dates unless they look like Excel serials.
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", text):
            try:
                number = float(text)
            except ValueError:
                return pd.NaT
            if not 20000 <= number <= 80000:
                return pd.NaT
            ts = pd.Timestamp("1899-12-30") + pd.to_timedelta(number, unit="D")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ts = pd.to_datetime(text, errors="coerce", dayfirst=dayfirst)
    if pd.isna(ts):
        return pd.NaT
    ts = pd.Timestamp(ts)
    if ts.year < 1900 or ts.year > 2100:
        return pd.NaT
    return ts


def _time_fraction(value: object) -> float:
    if value is None or _is_missing(value):
        return np.nan
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        return (value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6) / 86400.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return np.nan
        return number % 1.0
    text = safe_text(value)
    if not text:
        return np.nan
    # Excel sometimes exports 1900-01-01 00:30 as the time cell.
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.notna(parsed):
        parsed = pd.Timestamp(parsed)
        return (parsed.hour * 3600 + parsed.minute * 60 + parsed.second) / 86400.0
    match = re.search(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\s*(am|pm)?\b", text, flags=re.I)
    if not match:
        return np.nan
    hour, minute = int(match.group(1)), int(match.group(2))
    second = int(match.group(3) or 0)
    ampm = (match.group(4) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59 or second > 59:
        return np.nan
    return (hour * 3600 + minute * 60 + second) / 86400.0


def _date_only(value: object) -> Optional[date]:
    ts = _safe_datetime_scalar(value)
    return ts.date() if pd.notna(ts) else None


def _repair_date_time_sequence(date_values: Sequence[object], time_values: Sequence[object]) -> pd.Series:
    """Combine Excel date/time columns and repair common isolated date errors.

    Rules are intentionally conservative:
    - An isolated date different from equal previous/next dates is treated as a typo.
    - A midnight rollover increments the day when time drops by more than six hours
      and the date cell did not advance.
    - Duplicate timestamps remain valid; process readings are never parsed as dates.
    """
    dates: List[Optional[date]] = [_date_only(v) for v in date_values]
    fractions: List[float] = [_time_fraction(v) for v in time_values]

    # Correct isolated date typos such as 22-Jun 18:30 surrounded by 21-Jun rows.
    for i in range(1, len(dates) - 1):
        if dates[i - 1] is not None and dates[i + 1] is not None and dates[i - 1] == dates[i + 1]:
            if dates[i] is not None and dates[i] != dates[i - 1]:
                dates[i] = dates[i - 1]

    result: List[pd.Timestamp] = []
    previous: Optional[pd.Timestamp] = None
    for d, frac in zip(dates, fractions):
        if d is None or not math.isfinite(frac):
            result.append(pd.NaT)
            continue
        seconds = int(round(frac * 86400)) % 86400
        current = pd.Timestamp(datetime.combine(d, time.min) + timedelta(seconds=seconds))
        if previous is not None:
            previous_tod = (previous.hour * 3600 + previous.minute * 60 + previous.second) / 86400.0
            current_tod = seconds / 86400.0
            if current.date() <= previous.date() and previous_tod - current_tod > 0.25:
                while current <= previous:
                    current += pd.Timedelta(days=1)
            elif current < previous - pd.Timedelta(hours=18):
                while current < previous - pd.Timedelta(hours=18):
                    current += pd.Timedelta(days=1)
        result.append(current)
        if pd.notna(current):
            previous = current
    return pd.Series(result, dtype="datetime64[ns]")


def parse_datetime_series(series: pd.Series) -> pd.Series:
    values = [_safe_datetime_scalar(v) for v in series]
    return pd.Series(values, index=series.index, dtype="datetime64[ns]")


def combine_date_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
    combined = _repair_date_time_sequence(date_series.tolist(), time_series.tolist())
    combined.index = date_series.index
    return combined



def _date_from_name(value: object) -> Optional[date]:
    text = safe_text(value)
    patterns = [
        r"(?<!\d)(20\d{2})[-_](\d{1,2})[-_](\d{1,2})(?!\d)",
        r"(?<!\d)(\d{1,2})[-_/](\d{1,2})[-_/](20\d{2})(?!\d)",
    ]
    match = re.search(patterns[0], text)
    if match:
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            pass
    match = re.search(patterns[1], text)
    if match:
        try:
            return date(int(match.group(3)), int(match.group(2)), int(match.group(1)))
        except ValueError:
            pass
    return None


def _repair_datetime_ordered(values: Sequence[object]) -> pd.Series:
    parsed = [_safe_datetime_scalar(value) for value in values]
    # Isolated wrong date surrounded by a consistent sequence.
    for i in range(1, len(parsed) - 1):
        prev, cur, nxt = parsed[i - 1], parsed[i], parsed[i + 1]
        if pd.isna(prev) or pd.isna(cur) or pd.isna(nxt):
            continue
        prev, cur, nxt = pd.Timestamp(prev), pd.Timestamp(cur), pd.Timestamp(nxt)
        expected_step = nxt - prev
        # Typical device/test cadence: next is 30-120 minutes after previous,
        # while the current row has the correct clock time but wrong day.
        if pd.Timedelta(0) < expected_step <= pd.Timedelta(hours=4):
            midpoint = (prev + expected_step / 2).round("s")
            if abs(cur - midpoint) > pd.Timedelta(hours=12):
                parsed[i] = midpoint.normalize() + pd.Timedelta(
                    hours=cur.hour, minutes=cur.minute, seconds=cur.second
                )
    return pd.Series(parsed, dtype="datetime64[ns]")


def _align_dates_to_source_name(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "datetime" not in df.columns:
        return df
    out = df.copy()
    group_cols = [c for c in ["source", "sheet"] if c in out.columns]
    groups = out.groupby(group_cols, dropna=False, sort=False) if group_cols else [(None, out)]
    for _, group in groups:
        source_text = " ".join(safe_text(group[c].iloc[0]) for c in group_cols)
        normalized_source = normalize_text(source_text)
        expected = _date_from_name(source_text)
        if expected is None:
            continue
        valid = pd.to_datetime(group["datetime"], errors="coerce")
        valid_nonnull = valid.dropna()
        if valid_nonnull.empty:
            continue

        unique_dates = sorted(set(valid_nonnull.dt.date))
        hours = valid_nonnull.dt.hour + valid_nonnull.dt.minute / 60.0
        has_evening = bool((hours >= 12).any())
        has_morning = bool((hours < 12).any())
        date_span = (max(unique_dates) - min(unique_dates)).days if unique_dates else 0

        # Repair old dashboard exports where an overnight test was shifted into
        # dates such as 22/23 although the source report name says 21-Jun. When
        # a short group contains both evening and after-midnight readings, the
        # clock time establishes the intended two-day sequence reliably.
        if has_evening and has_morning and len(unique_dates) <= 3 and date_span <= 3:
            repaired = []
            for ts in valid:
                if pd.isna(ts):
                    repaired.append(pd.NaT)
                    continue
                ts = pd.Timestamp(ts).round("s")
                target_date = expected if ts.hour >= 12 else expected + timedelta(days=1)
                repaired.append(pd.Timestamp(datetime.combine(target_date, ts.time())))
            out.loc[group.index, "datetime"] = repaired
            continue

        actual_start = valid_nonnull.min().date()
        difference = (expected - actual_start).days
        if difference and abs(difference) <= 7 and "export" not in normalized_source:
            out.loc[group.index, "datetime"] = valid + pd.Timedelta(days=difference)
    return out


# ---------------------------------------------------------------------------
# Well normalization and source priority
# ---------------------------------------------------------------------------

WELL_TOKEN_RE = re.compile(
    r"\b(?:BED[_ -]?)?(?:[A-Z]\d{0,2}|S\d+|C\d+|B\d+)(?:[ _-]*[A-Z]?\d+){1,3}\b",
    flags=re.I,
)


def _normalize_bed_device_alias(value: object) -> str:
    """Normalize device-export aliases such as BED_16 C6-9 -> B16C6-9.

    The rule is deliberately limited to BED + number + an alphabetic well tail
    so a genuine name such as BED-15-33 is not silently changed.
    """
    text = safe_text(value).upper().replace("*", "").strip()
    match = re.fullmatch(r"BED[ _-]?(\d{1,2})[ _-]*([A-Z]\d+(?:[ _-]*\d+)*)", text)
    if match:
        tail = re.sub(r"[ _-]+", "-", match.group(2)).strip("-")
        return f"B{match.group(1)}{tail}"
    return text


def _well_key(value: object) -> str:
    text = _normalize_bed_device_alias(value)
    return re.sub(r"[^A-Z0-9]", "", text)


def clean_well_name_value(value: object) -> str:
    text = safe_text(value).upper().replace("*", "")
    text = re.sub(r"\bWELL(?:\s+NAME)?\b\s*[:=-]?", "", text, flags=re.I).strip()
    text = _normalize_bed_device_alias(text)
    text = re.sub(r"\s+", "", text)
    text = text.replace("_", "-")
    text = re.sub(r"-{2,}", "-", text).strip("-")
    # Canonicalize compound well names while preserving ordinary B15-40/S8-58.
    compound = re.fullmatch(r"([A-Z]\d{1,2})-([A-Z]\d+)-(\d+)", text)
    if compound:
        text = f"{compound.group(1)}{compound.group(2)}-{compound.group(3)}"
    return text if _well_key(text) else "Unknown"


def normalize_well_name(value: object) -> str:
    return clean_well_name_value(value)


def guess_well_from_name(value: object) -> str:
    text = safe_text(value).upper()
    # Device exports often use field-style names such as "BED_16 C6-9" for
    # well B16C6-9. Preserve the B prefix so device and TMU files merge.
    bed_match = re.search(r"\bBED[ _-]?(\d{1,2})[ _-]*([A-Z]\d+(?:[ _-]*\d+)?)", text)
    if bed_match:
        tail = re.sub(r"[ _-]+", "-", bed_match.group(2)).strip("-")
        return clean_well_name_value(f"B{bed_match.group(1)}{tail}")
    # Prefer explicit WELL label.
    match = re.search(r"WELL(?:\s+NAME)?\s*[:=-]?\s*([A-Z0-9 _-]{3,30})", text, flags=re.I)
    if match:
        candidate = re.split(r"\b(?:DATE|TIME|FIELD|TEST|REPORT)\b", match.group(1), maxsplit=1)[0]
        candidate = clean_well_name_value(candidate)
        if candidate != "Unknown":
            return candidate
    # File/sheet names: B15-40, B16C6-9, S8-58, B3C18-7, etc.
    candidates = re.findall(r"\b[A-Z]\d{0,2}(?:[ _-]*[A-Z]?\d+){1,3}\b", text)
    candidates = [clean_well_name_value(c) for c in candidates]
    candidates = [c for c in candidates if c != "Unknown" and not re.fullmatch(r"TMU-?\d+", c)]
    if candidates:
        # Prefer candidates containing a separator or more than one letter-number group.
        return max(candidates, key=lambda c: (len(_well_key(c)), len(c)))
    return "Unknown"


def _source_priority(name: str) -> int:
    norm = normalize_text(name)
    score = 0
    if re.search(r"\b(final|complete|completed|final report)\b", norm):
        score += 40
    if re.search(r"\b(clean|corrected|updated|latest)\b", norm):
        score += 20
    if re.search(r"\b(copy|old|draft|partial|incomplete)\b", norm):
        score -= 10
    return score


# ---------------------------------------------------------------------------
# Header mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeaderInfo:
    canonical: Optional[str]
    unit: str = ""
    raw_label: str = ""


def _header_contains(header: str, *patterns: str) -> bool:
    return any(re.search(pattern, header, flags=re.I) for pattern in patterns)


def canonical_header(header: object) -> HeaderInfo:
    raw = safe_text(header)
    h = normalize_text(raw)
    h = re.sub(r"^raw\s*[: -]\s*", "", h)
    if not h:
        return HeaderInfo(None, raw_label=raw)

    # Metadata and date/time.
    if re.fullmatch(r"datetime|date time|timestamp", h):
        return HeaderInfo("datetime", raw_label=raw)
    if _header_contains(h, r"^date(?:\s|$)", r"test date", r"current test date", r"d/mm", r"dd/mm") and "time" not in h:
        return HeaderInfo("date", raw_label=raw)
    if _header_contains(h, r"^time(?:\s|$)", r"hh:mm", r"clock time") and "date" not in h:
        return HeaderInfo("time", raw_label=raw)
    if _header_contains(h, r"^well$", r"well name"):
        return HeaderInfo("well", raw_label=raw)
    if h == "source":
        return HeaderInfo("source_meta", raw_label=raw)
    if h == "sheet":
        return HeaderInfo("sheet_meta", raw_label=raw)
    if h in {"source type", "source_type"}:
        return HeaderInfo("source_type_meta", raw_label=raw)
    if h in {"note", "notes"}:
        return HeaderInfo("note_meta", raw_label=raw)
    if h in {"gas status", "gas rate status"}:
        return HeaderInfo("gas_rate_status_meta", raw_label=raw)

    # Choke. Explicit unit wins.
    if "choke" in h:
        if "%" in h or "percent" in h or "opening" in h:
            return HeaderInfo("choke_pct", "%", raw)
        if re.search(r"(?:/\s*64|64th|in\s*/\s*64)", h):
            return HeaderInfo("choke_size_64", "/64", raw)
        return HeaderInfo("choke_ambiguous", "", raw)

    # Sucker-rod-pump surveillance fields. Exact aliases are used to avoid
    # confusing short headers with unrelated production-test measurements.
    if re.fullmatch(r"(?:sl|stroke length)(?: in| inch| inches)?", h):
        return HeaderInfo("stroke_length_in", "in", raw)
    if re.fullmatch(r"(?:spm|stroke rate|strokes per minute)", h):
        return HeaderInfo("stroke_rate_spm", "spm", raw)
    if re.fullmatch(r"(?:peak ?load|max(?:imum)? ?load)(?: lbf| lb)?", h):
        return HeaderInfo("peak_load_lbf", "lbf", raw)
    if re.fullmatch(r"(?:min ?load|minimum ?load)(?: lbf| lb)?", h):
        return HeaderInfo("minimum_load_lbf", "lbf", raw)

    # Device channels before generic pressure/temperature rules.
    if re.fullmatch(r"ama(?: amp)?", h) or _header_contains(h, r"motor current", r"current amp"):
        return HeaderInfo("motor_ama_amp", "A", raw)
    if re.fullmatch(r"pi(?: psi)?", h) or _header_contains(h, r"pump intake", r"intake pressure"):
        return HeaderInfo("pump_intake_pressure_psi", _pressure_unit(h), raw)
    if re.fullmatch(r"pd(?: psi)?", h) or _header_contains(h, r"pump discharge", r"discharge pressure"):
        return HeaderInfo("pump_discharge_pressure_psi", _pressure_unit(h), raw)
    if _header_contains(h, r"run freq", r"pump freq", r"pump frequency", r"drive freq"):
        return HeaderInfo("pump_freq_hz", "Hz", raw)
    if re.fullmatch(r"tm(?: f| c)?", h) or _header_contains(h, r"motor temp"):
        if _is_fahrenheit_header(h):
            return HeaderInfo("motor_temp_f", "F", raw)
        if _is_celsius_header(h):
            return HeaderInfo("motor_temp_c", "C", raw)
        return HeaderInfo("motor_temp_f", "infer", raw)

    # Pressures.
    if _header_contains(h, r"pumping[ ._-]*p", r"pump[ ._-]*p(?:ressure)?", r"pumping pressure", r"circulation pressure"):
        return HeaderInfo("pumping_pressure_psi", _pressure_unit(h), raw)
    if _header_contains(h, r"\bwhp\b", r"w h p", r"wellhead pressure"):
        return HeaderInfo("whp_psi", _pressure_unit(h), raw)
    if _header_contains(h, r"\bflp\b", r"flowing line pressure", r"flow line pressure"):
        return HeaderInfo("flp_psi", _pressure_unit(h), raw)
    if _header_contains(h, r"sep(?:arator)?\s*[ ._-]*p", r"separator pressure"):
        return HeaderInfo("sep_p_psi", _pressure_unit(h), raw)
    if _header_contains(h, r"upstream pressure", r"\bus press"):
        return HeaderInfo("us_press_psi", _pressure_unit(h), raw)
    if _header_contains(h, r"downstream pressure", r"\bds press"):
        return HeaderInfo("ds_press_psi", _pressure_unit(h), raw)
    if _header_contains(h, r"ct pressure", r"coiled tubing pressure"):
        return HeaderInfo("ct_pressure_psi", _pressure_unit(h), raw)

    # Rates and production measurements.
    if _header_contains(h, r"formation gas"):
        return HeaderInfo("gas_formation_mmscfd", _gas_rate_unit(h), raw)
    if _header_contains(h, r"total gas rate", r"qgas", r"gas rate") and "status" not in h:
        return HeaderInfo("gas_rate_mmscfd", _gas_rate_unit(h), raw)
    if _header_contains(h, r"gross rate", r"gross liquid", r"qgross"):
        return HeaderInfo("gross_rate_bpd", _liquid_rate_unit(h), raw)
    if _header_contains(h, r"oil rate", r"cond rate", r"condensate rate", r"qoil"):
        return HeaderInfo("oil_rate_stbd", _liquid_rate_unit(h), raw)
    if _header_contains(h, r"water rate", r"qwat"):
        return HeaderInfo("water_rate_bpd", _liquid_rate_unit(h), raw)
    if _header_contains(h, r"n2 rate", r"nitrogen rate"):
        return HeaderInfo("n2_rate_scfm", "scfm", raw)
    if _header_contains(h, r"bsw", r"bs and w", r"water cut", r"watercut", r"\bwc\b"):
        return HeaderInfo("bsw_pct", "%", raw)
    if "salinity" in h:
        return HeaderInfo("salinity_kppm", "ppm" if "kppm" not in h and "k ppm" not in h else "kppm", raw)

    # Gas/oil properties.
    if _header_contains(h, r"gas specific gravity", r"gas sp gr", r"sp gr", r"air =1"):
        return HeaderInfo("gas_sg", "", raw)
    if _header_contains(h, r"orifice"):
        return HeaderInfo("orifice_size_in", "in", raw)
    if _header_contains(h, r"\bgor\b"):
        return HeaderInfo("gor_scf_bbl", "scf/bbl", raw)
    if _header_contains(h, r"\bh2s\b"):
        return HeaderInfo("h2s_ppm", "ppm", raw)
    if _header_contains(h, r"\bco2\b"):
        return HeaderInfo("co2_mole_pct", "%", raw)
    if _header_contains(h, r"oil gravity", r"api gravity", r"deg api"):
        return HeaderInfo("oil_api", "API", raw)
    if _header_contains(h, r"oil k f", r"oil kf"):
        return HeaderInfo("oil_kf", "", raw)
    if _header_contains(h, r"oil meter increment", r"meter increment"):
        return HeaderInfo("oil_meter_increment_bbl", "bbl", raw)
    if _header_contains(h, r"oil cmf", r"\bcmf\b"):
        return HeaderInfo("oil_cmf", "", raw)
    if _header_contains(h, r"oil cum"):
        return HeaderInfo("oil_cum_bbl", "bbl", raw)
    if _header_contains(h, r"water cum", r"wat cum"):
        return HeaderInfo("water_cum_bbl", "bbl", raw)

    # Temperature rules with context.
    if _header_contains(h, r"\bflt\b", r"flow line temp"):
        return HeaderInfo("flt_f" if _is_fahrenheit_header(h) else "flt_c", "F" if _is_fahrenheit_header(h) else "C", raw)
    if _header_contains(h, r"gas temp", r"gas t"):
        return HeaderInfo("gas_temp_f" if _is_fahrenheit_header(h) else "gas_temp_c", "F" if _is_fahrenheit_header(h) else "C", raw)
    if _header_contains(h, r"oil temp"):
        return HeaderInfo("oil_temp_f" if _is_fahrenheit_header(h) else "oil_temp_c", "F" if _is_fahrenheit_header(h) else "C", raw)
    if _header_contains(h, r"separator temp", r"sep temp"):
        return HeaderInfo("sep_temp_f" if _is_fahrenheit_header(h) else "sep_temp_c", "F" if _is_fahrenheit_header(h) else "C", raw)
    if _header_contains(h, r"wellhead temp"):
        return HeaderInfo("wellhead_temp_f" if _is_fahrenheit_header(h) else "wellhead_temp_c", "F" if _is_fahrenheit_header(h) else "C", raw)

    # CTU fields.
    if _header_contains(h, r"ct depth", r"coiled tubing depth"):
        return HeaderInfo("ct_depth_m", "m", raw)
    if _header_contains(h, r"ct running speed", r"running speed"):
        return HeaderInfo("ct_running_speed_ftmin", "ft/min", raw)
    if _header_contains(h, r"ct pipe weight", r"pipe weight"):
        return HeaderInfo("ct_pipe_weight_lbf", "lbf", raw)

    return HeaderInfo(None, raw_label=raw)


def _pressure_unit(h: str) -> str:
    if "bar" in h:
        return "bar"
    if "mpa" in h:
        return "mpa"
    if "kpa" in h:
        return "kpa"
    return "psi"


def _gas_rate_unit(h: str) -> str:
    # Spreadsheet headers frequently insert spaces inside the unit, e.g.
    # ``MM SCF/D``. Compacting the unit prevents a valid MMSCF/D value from
    # being divided by one million as though it were plain SCF/D.
    compact = re.sub(r"[^a-z0-9/]", "", normalize_text(h))
    if "mmscf/d" in compact or "mmscfd" in compact:
        return "mmscfd"
    if "mscf/d" in compact or "mscfd" in compact:
        return "mscfd"
    if "scf/d" in compact or "scfd" in compact:
        return "scfd"
    return "mmscfd"


def _liquid_rate_unit(h: str) -> str:
    if "m3" in h or "m 3" in h:
        return "m3d"
    return "bpd"


def _is_fahrenheit_header(h: str) -> bool:
    return bool(re.search(r"(?:deg\s*f|\(f\)|\bf\b|fahrenheit)", h))


def _is_celsius_header(h: str) -> bool:
    return bool(re.search(r"(?:deg\s*c|\(c\)|\bc\b|celsius)", h))


def _convert_numeric(values: pd.Series, info: HeaderInfo) -> pd.Series:
    out = clean_tabular_numeric_series(values)
    key, unit = info.canonical, info.unit
    if key is None:
        return out
    if key.endswith("_psi"):
        if unit == "bar":
            out *= 14.5037738
        elif unit == "kpa":
            out *= 0.145037738
        elif unit == "mpa":
            out *= 145.037738
    elif key in {"gas_rate_mmscfd", "gas_formation_mmscfd"}:
        if unit == "scfd":
            out /= 1_000_000.0
        elif unit == "mscfd":
            out /= 1_000.0
    elif key in {"gross_rate_bpd", "oil_rate_stbd", "water_rate_bpd"} and unit == "m3d":
        out *= 6.28981077
    elif key == "salinity_kppm":
        if unit == "ppm":
            out /= 1000.0
        else:
            # Some files say K ppm but store 225000 ppm.
            out = out.where(out.abs() <= 1000, out / 1000.0)
    elif key == "choke_pct":
        out = out.where((out <= 1.0) | (out > 1.0), out)
        out = out.where(~((out > 0) & (out <= 1.0)), out * 100.0)
    elif key == "choke_size_64":
        # Some reports mix inch notation (1 = one inch = 64/64) with explicit
        # /64 values (34, 38, 42) in the same column. Convert row-by-row.
        inch_mask = out.notna() & (out.abs() > 0) & (out.abs() <= 2.0)
        out = out.where(~inch_mask, out * 64.0)
    return out


# ---------------------------------------------------------------------------
# XLSX direct bounded reader
# ---------------------------------------------------------------------------

_XLSX_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PKG_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _xlsx_col_number(ref: str) -> int:
    letters = re.match(r"([A-Z]+)", ref.upper())
    if not letters:
        return 0
    number = 0
    for char in letters.group(1):
        number = number * 26 + ord(char) - 64
    return number


def _read_shared_strings(zf: zipfile.ZipFile) -> List[str]:
    path = "xl/sharedStrings.xml"
    if path not in zf.namelist():
        return []
    strings: List[str] = []
    root = ET.fromstring(zf.read(path))
    for si in root.findall(f"{{{_XLSX_NS}}}si"):
        strings.append("".join(node.text or "" for node in si.iter() if node.tag.endswith("}t")))
    return strings


def _workbook_sheet_paths(zf: zipfile.ZipFile) -> List[Tuple[str, str]]:
    workbook = ET.fromstring(zf.read("xl/workbook.xml"))
    rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
    targets = {
        rel.attrib.get("Id"): rel.attrib.get("Target", "")
        for rel in rels.findall(f"{{{_PKG_REL_NS}}}Relationship")
    }
    output: List[Tuple[str, str]] = []
    for sheet in workbook.findall(f".//{{{_XLSX_NS}}}sheet"):
        name = sheet.attrib.get("name", "Sheet")
        rid = sheet.attrib.get(f"{{{_REL_NS}}}id")
        target = targets.get(rid, "")
        if not target:
            continue
        if target.startswith("/"):
            target = target.lstrip("/")
        elif not target.startswith("xl/"):
            target = "xl/" + target.lstrip("/")
        target = re.sub(r"(?:^|/)\.\./", "", target)
        output.append((name, target))
    return output


def _cell_value(cell: ET.Element, shared: Sequence[str]) -> object:
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter() if node.tag.endswith("}t"))
    value_node = cell.find(f"{{{_XLSX_NS}}}v")
    if value_node is None:
        return None
    raw = value_node.text or ""
    if cell_type == "s":
        try:
            return shared[int(raw)]
        except Exception:
            return raw
    if cell_type in {"str", "e"}:
        return raw
    if cell_type == "b":
        return raw == "1"
    try:
        return float(raw)
    except ValueError:
        return raw


def _peek_sheet_sparse(
    zf: zipfile.ZipFile,
    sheet_path: str,
    shared: Sequence[str],
    *,
    max_row: int = 20,
    max_cols: int = 80,
) -> pd.DataFrame:
    rows: Dict[int, Dict[int, object]] = {}
    with zf.open(sheet_path) as handle:
        for _, elem in ET.iterparse(handle, events=("end",)):
            if not elem.tag.endswith("}c"):
                continue
            ref = elem.attrib.get("r", "")
            match = re.match(r"[A-Z]+(\d+)", ref.upper())
            if not match:
                elem.clear(); continue
            row_num = int(match.group(1))
            if row_num > max_row:
                elem.clear()
                break
            col_num = _xlsx_col_number(ref)
            if 0 < col_num <= max_cols:
                value = _cell_value(elem, shared)
                if not _is_missing(value):
                    rows.setdefault(row_num, {})[col_num] = value
            elem.clear()
    if not rows:
        return pd.DataFrame(dtype=object)
    row_numbers = sorted(rows)
    width = max(max(row) for row in rows.values())
    return pd.DataFrame([[rows[r].get(c) for c in range(1, width + 1)] for r in row_numbers], index=row_numbers, dtype=object)


def _xlsx_simple_table_signature(zf: zipfile.ZipFile, sheets: Sequence[Tuple[str, str]], shared: Sequence[str]) -> bool:
    if len(sheets) != 1:
        return False
    _, path = sheets[0]
    peek = _peek_sheet_sparse(zf, path, shared)
    if peek.empty:
        return False
    for row_position, row in enumerate(peek.itertuples(index=False, name=None)):
        # A genuinely simple export starts with its table header. Standard TMU
        # reports have title/client rows first and must stay on the mature path.
        if row_position >= 3:
            break
        labels = [normalize_text(v) for v in row if safe_text(v)]
        joined = " | ".join(labels)
        has_dt = any(label in {"datetime", "date time", "timestamp"} for label in labels)
        has_date = any(label == "date" for label in labels)
        has_date_time = has_date and any(label == "time" for label in labels)
        recognized = sum(canonical_header(label).canonical not in {None, "date", "time", "datetime", "well"} for label in labels)
        if (has_dt or has_date_time or has_date) and recognized >= 1:
            return True
        # Device abbreviations in a one-row header.
        if (has_dt or has_date_time or has_date) and re.search(r"\b(?:ama|pi|pd|run freq|tm|sl|spm|peak ?load|min ?load)\b", joined):
            return True
    return False


def _read_sheet_sparse(
    zf: zipfile.ZipFile,
    sheet_path: str,
    shared: Sequence[str],
    *,
    max_rows: int = 20000,
    max_cols: int = 512,
    max_nonempty_cells: int = 500000,
) -> pd.DataFrame:
    rows: Dict[int, Dict[int, object]] = {}
    nonempty = 0
    with zf.open(sheet_path) as handle:
        for event, elem in ET.iterparse(handle, events=("end",)):
            if not elem.tag.endswith("}c"):
                continue
            ref = elem.attrib.get("r", "")
            match = re.match(r"[A-Z]+(\d+)", ref.upper())
            if not match:
                elem.clear()
                continue
            row_num = int(match.group(1))
            col_num = _xlsx_col_number(ref)
            if row_num > max_rows or col_num <= 0 or col_num > max_cols:
                elem.clear()
                continue
            value = _cell_value(elem, shared)
            if not _is_missing(value):
                rows.setdefault(row_num, {})[col_num] = value
                nonempty += 1
                if nonempty >= max_nonempty_cells:
                    elem.clear()
                    break
            elem.clear()
    if not rows:
        return pd.DataFrame(dtype=object)
    row_numbers = sorted(rows)
    max_col = max(max(values) for values in rows.values())
    matrix = [[rows[r].get(c) for c in range(1, max_col + 1)] for r in row_numbers]
    frame = pd.DataFrame(matrix, index=row_numbers, dtype=object)
    frame.attrs["source_row_numbers"] = row_numbers
    return frame


# ---------------------------------------------------------------------------
# Table detection and standardization
# ---------------------------------------------------------------------------

@dataclass
class HeaderCandidate:
    start_pos: int
    height: int
    headers: List[str]
    infos: List[HeaderInfo]
    score: float


def _combined_headers(raw: pd.DataFrame, start: int, height: int) -> List[str]:
    rows = [raw.iloc[start + offset].tolist() for offset in range(height)]
    width = raw.shape[1]
    parent_values: List[str] = [""] * width
    current_parent = ""
    for col in range(width):
        top = safe_text(rows[0][col])
        if top:
            current_parent = top
        parent_values[col] = current_parent

    headers: List[str] = []
    for col in range(width):
        pieces: List[str] = []
        normalized_seen: set[str] = set()
        for row_index, row in enumerate(rows):
            text = safe_text(row[col])
            if row_index == 0 and not text:
                text = parent_values[col]
            norm = normalize_text(text)
            if text and norm not in normalized_seen:
                normalized_seen.add(norm)
                pieces.append(text)
        headers.append(" | ".join(pieces))
    return headers


def _combine_scalar_date_time(date_value: object, time_value: object) -> pd.Timestamp:
    d = _date_only(date_value)
    fraction = _time_fraction(time_value)
    if d is None or not math.isfinite(fraction):
        return pd.NaT
    seconds = int(round(fraction * 86400)) % 86400
    return pd.Timestamp(datetime.combine(d, time.min) + timedelta(seconds=seconds))


def _header_candidate_score(raw: pd.DataFrame, start: int, height: int) -> HeaderCandidate:
    headers = _combined_headers(raw, start, height)
    infos = [canonical_header(header) for header in headers]
    canonical = [info.canonical for info in infos]
    has_datetime = "datetime" in canonical or "date" in canonical or ("date" in canonical and "time" in canonical)
    recognized_indices = [
        idx for idx, key in enumerate(canonical)
        if key and key not in {"date", "time", "datetime", "well", "source_meta", "sheet_meta", "source_type_meta", "note_meta", "gas_rate_status_meta"}
    ]
    unique_recognized = len({canonical[idx] for idx in recognized_indices})
    # Reject obvious non-headers before examining data rows.
    if not has_datetime or unique_recognized == 0:
        return HeaderCandidate(start, height, headers, infos, -100.0)

    sample = raw.iloc[start + height : min(len(raw), start + height + 50)]
    datetime_count = 0
    numeric_count = 0
    date_idx = canonical.index("date") if "date" in canonical else None
    time_idx = canonical.index("time") if "time" in canonical else None
    dt_idx = canonical.index("datetime") if "datetime" in canonical else None
    for row_tuple in sample.itertuples(index=False, name=None):
        if dt_idx is not None:
            dt = _safe_datetime_scalar(row_tuple[dt_idx])
        elif date_idx is not None and time_idx is not None:
            dt = _combine_scalar_date_time(row_tuple[date_idx], row_tuple[time_idx])
        elif date_idx is not None:
            dt = _safe_datetime_scalar(row_tuple[date_idx])
        else:
            dt = pd.NaT
        if pd.isna(dt):
            continue
        datetime_count += 1
        if any(pd.notna(extract_number(row_tuple[col])) for col in recognized_indices):
            numeric_count += 1
    unit_tokens = sum(bool(re.search(r"(?:psi|psig|mmscf|bbl|stb|ppm|%|deg|hh:mm|/64|scf)", normalize_text(h))) for h in headers)
    score = 20 + unique_recognized * 4 + min(datetime_count, 20) + min(numeric_count, 20) * 2 + min(unit_tokens, 12) * 0.5
    # Prefer an explicit Date + Time header over a broader parent/header block
    # that happens to contain only Date. Date-only remains valid for compact
    # surveillance exports such as SRP daily monitoring files.
    if "date" in canonical and "time" in canonical:
        score += 24
    elif "date" in canonical and "datetime" not in canonical:
        score -= 24
    return HeaderCandidate(start, height, headers, infos, score)

def _find_header_candidates(raw: pd.DataFrame) -> List[HeaderCandidate]:
    if raw is None or raw.empty:
        return []
    scan_rows = min(len(raw), 120)
    candidates: List[HeaderCandidate] = []
    for start in range(scan_rows):
        for height in (1, 2, 3):
            if start + height > len(raw):
                continue
            candidate = _header_candidate_score(raw, start, height)
            if candidate.score >= 16:
                candidates.append(candidate)
    candidates.sort(key=lambda c: c.score, reverse=True)
    selected: List[HeaderCandidate] = []
    for candidate in candidates:
        if any(abs(candidate.start_pos - previous.start_pos) <= 3 for previous in selected):
            continue
        selected.append(candidate)
        if len(selected) >= 4:
            break
    return selected


def _well_from_raw(raw: pd.DataFrame, source_name: str, sheet_name: str) -> str:
    for row_pos in range(min(len(raw), 40)):
        row = raw.iloc[row_pos].tolist()
        for col, value in enumerate(row):
            norm = normalize_text(value)
            if norm in {"well", "well name"} or norm.startswith("well "):
                for next_col in range(col + 1, min(len(row), col + 5)):
                    candidate = clean_well_name_value(row[next_col])
                    if candidate != "Unknown":
                        return candidate
            match = re.search(r"\bwell(?: name)?\s*[:=-]\s*([A-Z0-9 _-]+)", safe_text(value), flags=re.I)
            if match:
                candidate = clean_well_name_value(match.group(1))
                if candidate != "Unknown":
                    return candidate
    for text in (sheet_name, source_name):
        candidate = guess_well_from_name(text)
        if candidate != "Unknown":
            return candidate
    return "Unknown"


def _generic_numeric_columns(raw_table: pd.DataFrame, headers: Sequence[str], occupied: set[int]) -> Dict[int, str]:
    result: Dict[int, str] = {}
    for col in range(raw_table.shape[1]):
        if col in occupied:
            continue
        values = clean_tabular_numeric_series(raw_table.iloc[:, col])
        if values.notna().sum() < 2:
            continue
        label = safe_text(headers[col]) or f"Column {col + 1}"
        norm_label = normalize_text(label)
        if not norm_label or re.search(r"^(?:unnamed|column \d+|source|sheet|source type|test id|link status|time text|index)$", norm_label):
            continue
        result[col] = "raw__" + slugify(label)
    return result


def _parse_candidate(
    raw: pd.DataFrame,
    candidate: HeaderCandidate,
    *,
    source_name: str,
    sheet_name: str,
    default_well: str,
) -> pd.DataFrame:
    data = raw.iloc[candidate.start_pos + candidate.height :].copy().reset_index(drop=False)
    if data.empty:
        return pd.DataFrame()
    source_rows = data["index"].tolist()
    data = data.drop(columns=["index"])
    infos = candidate.infos
    canonical = [info.canonical for info in infos]

    if "datetime" in canonical:
        idx = canonical.index("datetime")
        dt = _repair_datetime_ordered(data.iloc[:, idx].tolist())
        dt.index = data.index
    elif "date" in canonical and "time" in canonical:
        date_idx, time_idx = canonical.index("date"), canonical.index("time")
        dt = combine_date_time(data.iloc[:, date_idx], data.iloc[:, time_idx])
    else:
        return pd.DataFrame()

    output = pd.DataFrame(index=data.index)
    output["datetime"] = dt
    output["source"] = source_name
    output["sheet"] = sheet_name
    if "source_meta" in canonical:
        meta_idx = canonical.index("source_meta")
        meta_values = data.iloc[:, meta_idx].map(safe_text)
        output["source"] = meta_values.where(meta_values.ne(""), source_name)
    if "sheet_meta" in canonical:
        meta_idx = canonical.index("sheet_meta")
        meta_values = data.iloc[:, meta_idx].map(safe_text)
        output["sheet"] = meta_values.where(meta_values.ne(""), sheet_name)
    if "source_type_meta" in canonical:
        meta_idx = canonical.index("source_type_meta")
        output["source_type"] = data.iloc[:, meta_idx].map(safe_text)
    if "note_meta" in canonical:
        meta_idx = canonical.index("note_meta")
        output["note"] = data.iloc[:, meta_idx].map(safe_text)
    if "gas_rate_status_meta" in canonical:
        meta_idx = canonical.index("gas_rate_status_meta")
        output["gas_rate_status"] = data.iloc[:, meta_idx].map(safe_text)
    output["source_row"] = source_rows
    output["source_priority"] = output["source"].map(_source_priority)
    output["source_group"] = output["source"].astype(str) + "::" + output["sheet"].astype(str)

    if "well" in canonical:
        well_idx = canonical.index("well")
        output["well"] = data.iloc[:, well_idx].map(clean_well_name_value)
        output["well"] = output["well"].where(output["well"].ne("Unknown"), default_well)
    else:
        output["well"] = default_well

    occupied: set[int] = {
        idx for idx, key in enumerate(canonical) if key in {"date", "time", "datetime", "well", "source_meta", "sheet_meta", "source_type_meta", "note_meta", "gas_rate_status_meta"}
    }
    note_parts: List[pd.Series] = []
    for col, info in enumerate(infos):
        key = info.canonical
        if key in {None, "date", "time", "datetime", "well", "source_meta", "sheet_meta", "source_type_meta", "note_meta", "gas_rate_status_meta"}:
            continue
        occupied.add(col)
        converted = _convert_numeric(data.iloc[:, col], info)
        if key in output.columns:
            output[key] = pd.to_numeric(output[key], errors="coerce").combine_first(converted)
        else:
            output[key] = converted

    # Preserve unknown numeric channels for simple device/export tables. Complex
    # TMU sheets contain many hidden formula/helper columns that should not be
    # exposed as measurements.
    recognized_count = len({key for key in canonical if key and key not in {"date", "time", "datetime", "well"}})
    if recognized_count <= 10:
        for col, raw_key in _generic_numeric_columns(data, candidate.headers, occupied).items():
            output[raw_key] = clean_tabular_numeric_series(data.iloc[:, col])
            COLUMN_LABELS.setdefault(raw_key, safe_text(candidate.headers[col]) or raw_key)

    # Capture operational text from non-mapped columns only on rows that also have data.
    for col in range(data.shape[1]):
        if col in occupied or canonical[col] in {"date", "time", "datetime", "well"}:
            continue
        text_series = data.iloc[:, col].map(safe_text)
        text_series = text_series.where(~text_series.str.fullmatch(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", na=False), "")
        if text_series.str.len().gt(3).any():
            note_parts.append(text_series)
    if note_parts:
        notes = output.get("note", pd.Series("", index=output.index, dtype=object)).fillna("").astype(object)
        for part in note_parts:
            notes = pd.Series([append_note(a, b) for a, b in zip(notes, part)], index=notes.index)
        output["note"] = notes
    elif "note" not in output.columns:
        output["note"] = ""

    numeric_cols = [
        col for col in output.columns
        if col not in BASE_NON_PLOT_COLS and pd.to_numeric(output[col], errors="coerce").notna().any()
    ]
    useful = output["datetime"].notna()
    if numeric_cols:
        useful &= output[numeric_cols].apply(lambda row: pd.to_numeric(row, errors="coerce").notna().any(), axis=1)
    else:
        useful &= False
    output = output.loc[useful].copy()
    if output.empty:
        return output

    # Infer Tm unit only when no unit was given.
    if "motor_temp_f" in output.columns:
        vals = pd.to_numeric(output["motor_temp_f"], errors="coerce")
        if vals.notna().any() and vals.median() < 180 and any(info.canonical == "motor_temp_f" and info.unit == "infer" for info in infos):
            output["motor_temp_c"] = vals
            output.drop(columns=["motor_temp_f"], inplace=True)

    output["date"] = output["datetime"].dt.date
    output["time_text"] = output["datetime"].dt.strftime("%H:%M")
    output["test_unit"] = sheet_name
    return _postprocess_table(output)


def _postprocess_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "datetime" not in out.columns:
        return pd.DataFrame()
    repaired = _repair_datetime_ordered(out["datetime"].tolist())
    repaired.index = out.index
    out["datetime"] = repaired
    out = _align_dates_to_source_name(out)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.round("s")
    out = out.loc[out["datetime"].notna()].copy()
    if out.empty:
        return out
    if "well" not in out.columns:
        out["well"] = "Unknown"
    out["well"] = out["well"].map(clean_well_name_value)
    for col in list(out.columns):
        if col in BASE_NON_PLOT_COLS or col == "gas_rate_status":
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().any():
            out[col] = numeric.astype("float64")
    # Normalize common template storage conventions after every parser path.
    if "choke_pct" in out.columns:
        vals = pd.to_numeric(out["choke_pct"], errors="coerce")
        out["choke_pct"] = vals.where(~((vals > 0) & (vals <= 1.0)), vals * 100.0)
    if "choke_size_64" in out.columns:
        vals = pd.to_numeric(out["choke_size_64"], errors="coerce")
        inch_mask = vals.notna() & (vals.abs() > 0) & (vals.abs() <= 2.0)
        out["choke_size_64"] = vals.where(~inch_mask, vals * 64.0)
    if "salinity_kppm" in out.columns:
        vals = pd.to_numeric(out["salinity_kppm"], errors="coerce")
        out["salinity_kppm"] = vals.where(vals.abs() <= 1000, vals / 1000.0)

    if "pumping_pressure_psi" in out.columns and "pump_intake_pressure_psi" in out.columns:
        pump = pd.to_numeric(out["pumping_pressure_psi"], errors="coerce")
        intake = pd.to_numeric(out["pump_intake_pressure_psi"], errors="coerce")
        comparable = pump.notna() & intake.notna()
        source_text = out.get("source", pd.Series("", index=out.index)).astype(str)
        compatibility = source_text.str.contains(r"legacy|compatible|temporary", case=False, regex=True, na=False)
        equal = comparable & np.isclose(pump, intake, rtol=1e-9, atol=1e-9) & compatibility
        out.loc[equal, "pumping_pressure_psi"] = np.nan
        if pd.to_numeric(out["pumping_pressure_psi"], errors="coerce").notna().sum() == 0:
            out.drop(columns=["pumping_pressure_psi"], inplace=True)

    for col, default in {
        "source": "Unknown source", "sheet": "Data", "note": "", "test_unit": "Data",
        "source_type": "tabular", "link_status": "source_confirmed",
    }.items():
        if col not in out.columns:
            out[col] = default
    # Always regenerate display date/time from the repaired canonical datetime.
    # Source date/time cells may contain the exact defects that this pipeline fixes
    # (wrong overnight date, 1900-dated time, or floating 17:29:59 values).
    out["date"] = out["datetime"].dt.date
    out["time_text"] = out["datetime"].dt.strftime("%H:%M")
    return out.sort_values(["well", "datetime", "source", "sheet"], kind="stable").reset_index(drop=True)


def _parse_xlsx(data: bytes, name: str) -> List[pd.DataFrame]:
    tables: List[pd.DataFrame] = []
    diagnostics: List[str] = []

    # Decide by workbook structure, not file size. Standard TMU reports are
    # handled by the mature fast parser; one-sheet simple DateTime/device tables
    # use the deterministic v67 parser. This avoids scanning enormous formatted
    # tails while still accepting new device/export templates.
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as _probe:
            _shared = _read_shared_strings(_probe)
            _sheets = _workbook_sheet_paths(_probe)
            simple_table = _xlsx_simple_table_signature(_probe, _sheets, _shared)
        if not simple_table:
            legacy_tables = legacy.load_tabular_file(
                UploadedBytes(data, name), parse_images=False, max_ocr_images=0
            )
            legacy_tables = [_postprocess_table(table) for table in legacy_tables]
            legacy_tables = [table for table in legacy_tables if is_valid_timeseries(table)]
            if legacy_tables:
                return _deduplicate_table_interpretations(legacy_tables)
    except Exception as exc:
        diagnostics.append(f"fast legacy path: {exc}")

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        shared = _read_shared_strings(zf)
        for sheet_name, sheet_path in _workbook_sheet_paths(zf):
            try:
                raw = _read_sheet_sparse(zf, sheet_path, shared)
            except Exception as exc:
                diagnostics.append(f"{sheet_name}: {exc}")
                continue
            if raw.empty:
                continue
            default_well = _well_from_raw(raw, name, sheet_name)
            candidates = _find_header_candidates(raw)
            for candidate in candidates:
                try:
                    table = _parse_candidate(
                        raw, candidate, source_name=name, sheet_name=sheet_name,
                        default_well=default_well,
                    )
                except Exception as exc:
                    diagnostics.append(f"{sheet_name} row {candidate.start_pos + 1}: {exc}")
                    continue
                if is_valid_timeseries(table):
                    tables.append(table)
                    # Most sheets contain one primary data table. Do not produce
                    # overlapping duplicate interpretations from nearby headers.
                    break
    tables = _deduplicate_table_interpretations(tables)
    if tables:
        non_helper = [
            table for table in tables
            if not re.search(r"(?:^|\b)(form|cover|summary|cmsf|shrinkage)(?:\b|$)",
                             safe_text(table.get("sheet", pd.Series([""])).iloc[0]), flags=re.I)
        ]
        if non_helper:
            tables = non_helper
        # Prefer the richest interpretation when a workbook contains helper
        # summaries plus the real time-series sheet.
        tables.sort(key=lambda table: (len(table), len(available_numeric_columns(table))), reverse=True)
        return tables
    # Fallback preserves PDF/image/rare workbook behavior from prior releases.
    try:
        fallback = legacy.load_tabular_file(UploadedBytes(data, name), parse_images=False, max_ocr_images=0)
        fallback = [_postprocess_table(table) for table in fallback]
        fallback = [table for table in fallback if is_valid_timeseries(table)]
        if fallback:
            return fallback
    except Exception as exc:
        diagnostics.append(f"legacy fallback: {exc}")
    detail = "; ".join(diagnostics[:8])
    raise RuntimeError(
        f"no usable time-series table detected. The file was read safely, but no row set contained "
        f"a valid date/time plus numeric readings. {detail}".strip()
    )


# ---------------------------------------------------------------------------
# CSV and generic delimited files
# ---------------------------------------------------------------------------


def _decode_text_bytes(data: bytes) -> tuple[str, str]:
    # UTF-16 must be tried before Latin-1 because Latin-1 accepts every byte.
    encodings = ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1252", "latin-1")
    for encoding in encodings:
        try:
            text = data.decode(encoding, errors="strict")
            if "\x00" in text and not encoding.startswith("utf-16"):
                continue
            return text.replace("\x00", ""), encoding
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace").replace("\x00", ""), "latin-1"


def _detect_delimiter(text: str) -> str:
    sample = "\n".join(line for line in text.splitlines()[:30] if line.strip())
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except Exception:
        counts = {sep: sample.count(sep) for sep in [",", ";", "\t", "|"]}
        return max(counts, key=counts.get) if counts and max(counts.values()) else ","


def _parse_delimited(data: bytes, name: str) -> List[pd.DataFrame]:
    text, _encoding = _decode_text_bytes(data)
    if not text.strip():
        raise RuntimeError("file is blank")
    sep = _detect_delimiter(text)
    try:
        raw = pd.read_csv(io.StringIO(text), sep=sep, header=None, dtype=object, engine="python")
    except Exception as exc:
        raise RuntimeError(f"could not read delimited text: {exc}") from exc
    raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if raw.empty:
        raise RuntimeError("file is blank")
    raw.index = range(1, len(raw) + 1)
    default_well = _well_from_raw(raw, name, "CSV")
    tables: List[pd.DataFrame] = []
    for candidate in _find_header_candidates(raw):
        table = _parse_candidate(raw, candidate, source_name=name, sheet_name="CSV", default_well=default_well)
        if is_valid_timeseries(table):
            tables.append(table)
            break
    if tables:
        return tables
    # Pasted WhatsApp exported as text/CSV can still be parsed by the report parser.
    wa = parse_whatsapp_plain_or_export_text(text, source_name=name)
    if is_valid_timeseries(wa):
        return [wa]
    raise RuntimeError(
        "no usable time-series table detected. Check that the first data header contains DateTime "
        "or Date + Time and at least one numeric measurement column."
    )


# ---------------------------------------------------------------------------
# WhatsApp parsing
# ---------------------------------------------------------------------------


def _clean_whatsapp_text(text: object) -> str:
    value = str(text or "")
    value = value.replace("\u200e", "").replace("\u200f", "").replace("\ufeff", "")
    value = value.replace("\xa0", " ")
    value = re.sub(r"[*_`~]+", "", value)
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def split_messages(text: str) -> List[str]:
    cleaned = _clean_whatsapp_text(text)
    if not cleaned:
        return []
    header = re.compile(r"(?im)^\s*(?:PICO\s*TMU|TMU)\s*[- ]?\s*\d+\b[^\n]*")
    starts = [match.start() for match in header.finditer(cleaned)]
    if not starts:
        starts = [match.start() for match in re.finditer(r"(?im)^\s*date\s*[:=@-]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", cleaned)]
    if len(starts) <= 1:
        return [cleaned]
    return [cleaned[start : (starts[index + 1] if index + 1 < len(starts) else len(cleaned))].strip() for index, start in enumerate(starts)]


def _value_after_label(text: str, labels: Sequence[str]) -> str:
    for label in labels:
        match = re.search(rf"(?im)^\s*{label}\s*(?:[:=@-]\s*)?(.+?)\s*$", text)
        if match:
            return match.group(1).strip()
    return ""


def parse_tmu_message(message: str, source_name: str = "WhatsApp_Text") -> Dict[str, object]:
    text = _clean_whatsapp_text(message)
    date_text = _value_after_label(text, [r"date"])
    time_text = _value_after_label(text, [r"time"])
    date_value = _safe_datetime_scalar(date_text)
    fraction = _time_fraction(time_text)
    dt = pd.NaT
    if pd.notna(date_value) and math.isfinite(fraction):
        dt = date_value.normalize() + pd.to_timedelta(fraction, unit="D")

    well = clean_well_name_value(_value_after_label(text, [r"well\s*name", r"well"]))
    if well == "Unknown":
        well = guess_well_from_name(text)

    row: Dict[str, object] = {
        "source": source_name,
        "sheet": "WhatsApp",
        "source_type": "pasted_whatsapp_text",
        "well": well,
        "datetime": dt,
        "note": "",
        "test_unit": "WhatsApp",
        "source_priority": _source_priority(source_name),
        "source_group": source_name,
    }


    mappings = [
        ("choke_pct", [r"choke"], "%"),
        ("whp_psi", [r"w\.?\s*h\.?\s*p\.?", r"whp", r"wellhead pressure"], "psi"),
        ("sep_p_psi", [r"sep\.?\s*p\.?", r"separator pressure"], "psi"),
        ("gross_rate_bpd", [r"gross rate"], "bpd"),
        ("oil_rate_stbd", [r"oil rate", r"cond rate"], "bpd"),
        ("water_rate_bpd", [r"water rate"], "bpd"),
        ("bsw_pct", [r"bs\s*&?\s*w", r"bsw", r"water cut"], "%"),
        ("salinity_kppm", [r"salinity"], "kppm"),
        ("co2_mole_pct", [r"co2"], "%"),
        ("h2s_ppm", [r"h2s"], "ppm"),
        ("pumping_pressure_psi", [r"pumping pressure", r"pump p"], "psi"),
    ]
    for key, labels, unit in mappings:
        raw = _value_after_label(text, labels)
        value = extract_number(raw)
        if pd.notna(value):
            if key == "choke_pct" and 0 < value <= 1:
                value *= 100.0
            if key == "salinity_kppm" and "ppm" in normalize_text(raw) and "kppm" not in normalize_text(raw) and "k ppm" not in normalize_text(raw):
                value /= 1000.0
            row[key] = value

    gas_raw = _value_after_label(text, [r"gas rate", r"total gas rate"])
    gas_value = extract_number(gas_raw)
    if pd.notna(gas_value):
        row["gas_rate_mmscfd"] = gas_value
    else:
        norm = normalize_text(gas_raw)
        status = None
        if "low gas" in norm:
            status = "Low gas"
        elif re.search(r"\b(no|zero|nil) gas\b", norm):
            status = "No gas"
        elif "trace gas" in norm:
            status = "Trace gas"
        elif norm:
            status = safe_text(gas_raw)
        if status:
            row["gas_rate_status"] = status
            row["note"] = append_note(row.get("note"), f"Gas rate: {status}")
    if re.search(r"(?im)^\s*production test\s*$", text):
        row["note"] = append_note(row.get("note"), "Production test")
    return row


def parse_many_tmu_messages(text: str, source_name: str = "WhatsApp_Text") -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for index, chunk in enumerate(split_messages(text), start=1):
        row = parse_tmu_message(chunk, source_name=source_name)
        numeric = sum(pd.notna(extract_number(row.get(key))) for key in CANONICAL_NUMERIC_FIELDS if key in row)
        if pd.notna(row.get("datetime")) and numeric >= 1:
            row["message_index"] = index
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return _postprocess_table(pd.DataFrame(rows))


def parse_whatsapp_plain_or_export_text(text: str, source_name: str = "WhatsApp_Text") -> pd.DataFrame:
    # Legacy export parser handles WhatsApp timestamp prefixes and attachment captions.
    try:
        exported = legacy.parse_whatsapp_export_text(text, source_name=source_name)
        if exported is not None and not exported.empty:
            return _postprocess_table(exported)
    except Exception:
        pass
    return parse_many_tmu_messages(text, source_name=source_name)


# ---------------------------------------------------------------------------
# Public loading API
# ---------------------------------------------------------------------------


def is_valid_timeseries(df: pd.DataFrame) -> bool:
    if df is None or df.empty or "datetime" not in df.columns:
        return False
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    if dt.notna().sum() == 0:
        return False
    numeric_cols = available_numeric_columns(df)
    return bool(numeric_cols and any(pd.to_numeric(df[col], errors="coerce").notna().any() for col in numeric_cols))


def _deduplicate_table_interpretations(tables: Sequence[pd.DataFrame]) -> List[pd.DataFrame]:
    unique: List[pd.DataFrame] = []
    signatures = set()
    for table in tables:
        if not is_valid_timeseries(table):
            continue
        dt = pd.to_datetime(table["datetime"], errors="coerce")
        numeric = tuple(sorted(available_numeric_columns(table)))
        signature = (
            _well_key(table["well"].iloc[0]) if "well" in table.columns and len(table) else "",
            len(table), str(dt.min()), str(dt.max()), numeric,
        )
        if signature not in signatures:
            signatures.add(signature)
            unique.append(table)
    return unique


def load_tabular_file(uploaded_file, parse_images: bool = True, max_ocr_images: int = 1000) -> List[pd.DataFrame]:
    data, name = _uploaded_bytes(uploaded_file)
    suffix = Path(name).suffix.lower()
    if suffix in {".xlsx", ".xlsm"}:
        return _parse_xlsx(data, name)
    if suffix == ".xls":
        try:
            book = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=object, engine="xlrd")
            tables: List[pd.DataFrame] = []
            for sheet_name, raw in book.items():
                raw = raw.dropna(axis=0, how="all").dropna(axis=1, how="all")
                default_well = _well_from_raw(raw, name, sheet_name)
                for candidate in _find_header_candidates(raw):
                    table = _parse_candidate(raw, candidate, source_name=name, sheet_name=sheet_name, default_well=default_well)
                    if is_valid_timeseries(table):
                        tables.append(table)
                        break
            if tables:
                return _deduplicate_table_interpretations(tables)
        except Exception:
            pass
    if suffix in {".csv", ".tsv"}:
        return _parse_delimited(data, name)
    if suffix in {".txt", ".log"}:
        text, _ = _decode_text_bytes(data)
        table = parse_whatsapp_plain_or_export_text(text, source_name=name)
        if is_valid_timeseries(table):
            return [table]
    # ZIP, PDF, DOCX and images retain mature extraction/OCR support.
    fallback = legacy.load_tabular_file(
        UploadedBytes(data, name), parse_images=parse_images, max_ocr_images=max_ocr_images
    )
    normalized = [
        table.copy() if (hasattr(legacy, "is_usable_ocr_table") and legacy.is_usable_ocr_table(table))
        else _postprocess_table(table)
        for table in fallback
    ]
    valid_timeseries = [table for table in normalized if is_valid_timeseries(table)]
    valid_ocr = [
        table for table in normalized
        if not is_valid_timeseries(table)
        and hasattr(legacy, "is_usable_ocr_table")
        and legacy.is_usable_ocr_table(table)
    ]
    if valid_timeseries or valid_ocr:
        return _deduplicate_table_interpretations(valid_timeseries) + valid_ocr
    raise RuntimeError(
        "no usable time-series table detected. The file may be blank, or it has no valid date/time plus numeric readings."
    )


# ---------------------------------------------------------------------------
# Duplicate merging and test segmentation
# ---------------------------------------------------------------------------


def _row_completeness(row: pd.Series) -> int:
    count = 0
    for col, value in row.items():
        if col in BASE_NON_PLOT_COLS:
            continue
        if not _is_missing(value):
            count += 1
    return count


def merge_duplicate_test_rows_v53(df: pd.DataFrame) -> pd.DataFrame:
    """Merge repeated/incomplete reports by normalized well + minute.

    Values are coalesced column-by-column. A more complete row wins conflicts;
    final/clean files and later upload order break ties. Blank cells never erase
    an existing measurement.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    out["datetime"] = parse_datetime_series(out.get("datetime", pd.Series(index=out.index)))
    out["well"] = out.get("well", pd.Series("Unknown", index=out.index)).map(clean_well_name_value)
    out["_well_key"] = out["well"].map(_well_key)
    out["_minute_key"] = out["datetime"].dt.round("min")
    out["_source_order"] = np.arange(len(out), dtype=int)
    if "source_priority" in out.columns:
        out["source_priority"] = pd.to_numeric(out["source_priority"], errors="coerce").fillna(0)
    else:
        out["source_priority"] = 0.0
    if "source" in out.columns:
        out["source_priority"] += out["source"].map(_source_priority)
    out["_row_completeness"] = out.apply(_row_completeness, axis=1)

    key_valid = out["_well_key"].ne("") & out["_minute_key"].notna()
    valid = out.loc[key_valid].copy()
    invalid = out.loc[~key_valid].copy()
    merged_rows: List[Dict[str, object]] = []

    for _, group in valid.groupby(["_well_key", "_minute_key"], sort=False, dropna=False):
        ranked = group.sort_values(
            ["_row_completeness", "source_priority", "_source_order"],
            ascending=[False, False, False], kind="stable",
        )
        base = ranked.iloc[0].to_dict()
        for _, row in ranked.iloc[1:].iterrows():
            for col in out.columns:
                if col in {"_well_key", "_minute_key", "_source_order", "_row_completeness"}:
                    continue
                if col == "note":
                    base[col] = append_note(base.get(col), row.get(col))
                elif _is_missing(base.get(col)) and not _is_missing(row.get(col)):
                    base[col] = row.get(col)
        base["datetime"] = pd.Timestamp(base["_minute_key"])
        merged_rows.append(base)

    merged = pd.DataFrame(merged_rows)
    if not invalid.empty:
        merged = pd.concat([merged, invalid], ignore_index=True, sort=False)
    merged.drop(columns=["_well_key", "_minute_key", "_source_order", "_row_completeness"], inplace=True, errors="ignore")
    return _postprocess_table(merged)


def assign_test_ids(df: pd.DataFrame, gap_hours: float = 12.0) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["well"] = out.get("well", pd.Series("Unknown", index=out.index)).map(clean_well_name_value)
    out["datetime"] = parse_datetime_series(out.get("datetime", pd.Series(index=out.index)))
    assignments: Dict[object, Tuple[str, float]] = {}
    for well, group in out.sort_values(["well", "datetime"], kind="stable").groupby("well", dropna=False):
        sequence = 0
        current_id = ""
        last_dt = pd.NaT
        for index, row in group.iterrows():
            dt = row.get("datetime")
            if pd.isna(dt):
                sequence += 1
                assignments[index] = (f"{well}_NoTime_{sequence}", float(sequence))
                continue
            dt = pd.Timestamp(dt)
            if not current_id or pd.isna(last_dt) or dt - pd.Timestamp(last_dt) > pd.Timedelta(hours=float(gap_hours)):
                sequence += 1
                current_id = f"{well}_{dt:%Y%m%d_%H%M}"
            assignments[index] = (current_id, float(sequence))
            last_dt = dt
    out["test_id"] = ""
    out["test_sequence"] = np.nan
    for index, (test_id, sequence) in assignments.items():
        out.at[index, "test_id"] = test_id
        out.at[index, "test_sequence"] = sequence
    return out


# ---------------------------------------------------------------------------
# UI compatibility functions
# ---------------------------------------------------------------------------


def available_numeric_columns(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    audit_flags = {
        "gas_formation_derived", "n2_rate_derived", "total_gas_derived",
        "review_required", "ocr_approved", "is_event", "is_duplicate",
    }
    columns: List[str] = []
    seen = set()
    for col in df.columns:
        if col in seen or col in BASE_NON_PLOT_COLS or col in audit_flags or str(col).startswith("_"):
            continue
        seen.add(col)
        positions = [i for i, name in enumerate(df.columns) if name == col]
        usable = False
        for pos in positions:
            series = df.iloc[:, pos]
            if pd.api.types.is_bool_dtype(series.dtype):
                continue
            values = pd.to_numeric(series, errors="coerce")
            if values.notna().any():
                usable = True
                break
        if usable:
            columns.append(col)
    return columns


def column_label(col: object) -> str:
    key = str(col)
    if key in COLUMN_LABELS:
        return COLUMN_LABELS[key]
    if key.startswith("raw__"):
        return key[5:].replace("_", " ").title()
    try:
        return legacy.column_label(col)
    except Exception:
        return key.replace("_", " ").title()


def apply_fill_method(df: pd.DataFrame, features: Iterable[str], method: str) -> pd.DataFrame:
    return legacy.apply_fill_method(df, features, method)


def apply_user_column_mappings(df: pd.DataFrame, mappings: Mapping[str, str]) -> pd.DataFrame:
    try:
        return legacy.apply_user_column_mappings(df, mappings)
    except Exception:
        return df


# Re-export mature OCR/linking helpers used by older integrations.
for _name in [
    "suggest_links_for_ocr_rows", "approve_suggested_ocr_links",
    "parse_ctu_all_data_screen_image", "parse_whatsapp_export_text",
    "parse_whatsapp_export_messages", "parse_expro_mpfm_text",
]:
    if hasattr(legacy, _name) and _name not in globals():
        globals()[_name] = getattr(legacy, _name)


# ---------------------------------------------------------------------------
# v67 - preserve source calendar dates; filenames are metadata, not timestamps
# ---------------------------------------------------------------------------
def _align_dates_to_source_name(df: pd.DataFrame) -> pd.DataFrame:
    """Keep parsed calendar dates unchanged.

    Earlier releases tried to force dates to the date embedded in a filename.
    That is unsafe for multi-day tests and dashboard exports (whose filename
    date is the export/download date).  Date/time repair is now performed from
    the actual Date and Time columns before this stage.
    """
    return df.copy() if df is not None else df

# =============================================================================
# v68 - corpus-driven robust ingestion, validation and engine reconciliation
# =============================================================================
# This layer is intentionally appended so it overrides older implementations at
# runtime while preserving the mature OCR/PDF/WhatsApp-ZIP functions above.
PARSER_BUILD_ID_V68 = "v68-corpus-driven-robust-ingestion-20260623"
PARSER_BUILD_ID = PARSER_BUILD_ID_V68

COLUMN_LABELS.update({
    "n2_rate_mmscfd": "N₂ Rate (MMSCF/D)",
    "gas_chart_reading": "Gas Chart Reading",
    "dp1_psi": "DP1 (psi)",
    "dp2_psi": "DP2 (psi)",
    "parse_confidence": "Parse Confidence",
    "rejected_values": "Rejected Values",
})
BASE_NON_PLOT_COLS.update({"parse_confidence", "rejected_values", "parser_engine"})
CANONICAL_NUMERIC_FIELDS.update({
    "n2_rate_mmscfd", "gas_chart_reading", "dp1_psi", "dp2_psi",
    "stroke_length_in", "stroke_rate_spm", "peak_load_lbf", "minimum_load_lbf",
})

# Keep references to pre-v68 implementations for fallbacks.
_extract_number_before_v68 = extract_number
_extract_tabular_number_before_v68 = extract_tabular_number
_canonical_header_before_v68 = canonical_header
_convert_numeric_before_v68 = _convert_numeric
_postprocess_before_v68 = _postprocess_table
_parse_candidate_before_v68 = _parse_candidate
_parse_xlsx_before_v68 = _parse_xlsx
_parse_delimited_before_v68 = _parse_delimited
_split_messages_before_v68 = split_messages
_parse_tmu_message_before_v68 = parse_tmu_message
_parse_many_tmu_messages_before_v68 = parse_many_tmu_messages
_parse_whatsapp_before_v68 = parse_whatsapp_plain_or_export_text
_merge_duplicates_before_v68 = merge_duplicate_test_rows_v53


# ---------------------------------------------------------------------------
# Locale-aware numeric parsing
# ---------------------------------------------------------------------------
_NUMERIC_CANDIDATE_V68 = re.compile(
    r"(?<![A-Za-z0-9])\(?[−–—+\-]?\s*(?:\d[\d\s,.'’]*|\.\d+)(?:[eE][+\-]?\d+)?\)?"
)


def _normalize_numeric_token_v68(token: object) -> Optional[float]:
    text = safe_text(token)
    if not text:
        return None
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    negative_parentheses = text.strip().startswith("(") and text.strip().endswith(")")
    text = text.strip().strip("() ")
    text = re.sub(r"[\s'’]", "", text)
    if not text:
        return None

    # Separate exponent before normalizing decimal/thousands punctuation.
    exponent = ""
    exp_match = re.search(r"([eE][+\-]?\d+)$", text)
    if exp_match:
        exponent = exp_match.group(1)
        text = text[: exp_match.start()]

    comma_count = text.count(",")
    dot_count = text.count(".")
    if comma_count and dot_count:
        # Whichever punctuation appears last is the decimal separator.
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif comma_count:
        parts = text.split(",")
        if comma_count == 1:
            left, right = parts
            left_digits = left.lstrip("+-")
            # 0,445 and 12,5 are decimal; 1,234 is normally a thousands group.
            decimal_comma = (
                len(right) != 3
                or left_digits in {"0", ""}
                or (len(left_digits) > 3 and len(right) <= 3)
            )
            text = left + ("." if decimal_comma else "") + right
        elif all(len(part) == 3 for part in parts[1:]):
            text = "".join(parts)
        else:
            text = "".join(parts[:-1]) + "." + parts[-1]
    elif dot_count > 1:
        parts = text.split(".")
        if all(len(part) == 3 for part in parts[1:]):
            text = "".join(parts)
        else:
            text = "".join(parts[:-1]) + "." + parts[-1]

    text += exponent
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    if negative_parentheses:
        value = -abs(value)
    return value if math.isfinite(value) else None


def extract_number(value: object) -> float:
    """Parse engineering numbers including scientific and locale notation.

    Examples accepted: ``9.827E-2``, ``0,445``, ``1,234.5``, ``1.234,5`` and
    ``(12.5)``. Text statuses such as ``Low gas`` remain missing.
    """
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else np.nan
    text = safe_text(value)
    if not text or normalize_text(text) in {
        "n/a", "na", "nil", "none", "not available", "low gas", "trace gas",
        "no reading", "not measured",
    }:
        return np.nan
    match = _NUMERIC_CANDIDATE_V68.search(text)
    if not match:
        return np.nan
    number = _normalize_numeric_token_v68(match.group(0))
    return np.nan if number is None else float(number)


def extract_tabular_number(value: object) -> float:
    """Strict numeric parser that rejects operational prose in measurement cells."""
    if value is None or isinstance(value, bool):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else np.nan
    text = safe_text(value)
    if not text or normalize_text(text) in {
        "n/a", "na", "nil", "none", "not available", "low gas", "trace gas",
        "no reading", "not measured",
    }:
        return np.nan
    match = _NUMERIC_CANDIDATE_V68.search(text)
    if not match:
        return np.nan
    remainder = (text[: match.start()] + " " + text[match.end() :]).lower()
    remainder = remainder.replace("°", " ")
    remainder = re.sub(r"[()\[\]{}'\".,;:+*/_\-]+", " ", remainder)
    words = [word for word in re.findall(r"[a-z]+", remainder) if word]
    allowed = {
        "psi", "psig", "psia", "bar", "kpa", "mpa", "bbl", "bpd", "stb",
        "d", "day", "mmscf", "mmscfd", "mscf", "mscfd", "scf", "scfd",
        "scfm", "ppm", "kppm", "nacl", "mole", "mol", "hz", "amp", "amps",
        "a", "c", "f", "deg", "in", "inch", "inches", "api", "percent",
        "pct", "mm", "cm", "m", "ft", "min", "hr", "hours", "factor",
        "air", "kg", "lbf", "lb", "rpm", "v", "volt", "volts",
    }
    if any(word not in allowed for word in words):
        return np.nan
    number = _normalize_numeric_token_v68(match.group(0))
    return np.nan if number is None else float(number)


def clean_tabular_numeric_series(series: pd.Series) -> pd.Series:
    return series.map(extract_tabular_number).astype("float64")


def clean_numeric_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype="float64")
    return series.map(extract_number).astype("float64")


# ---------------------------------------------------------------------------
# Header aliases and unit conversions
# ---------------------------------------------------------------------------
def canonical_header(header: object) -> HeaderInfo:
    raw = safe_text(header)
    h = normalize_text(raw)
    h = re.sub(r"^(?:raw|source|original|detected)\s*[: -]\s*", "", h)
    if not h:
        return HeaderInfo(None, raw_label=raw)
    if h in {"time text", "timetext", "display time"}:
        return HeaderInfo("time", raw_label=raw)

    # Fields that need more context than the older generic aliases.
    if re.search(r"\bn\s*2\b|\bnitrogen\b", h) and re.search(r"rate|mmscf|scfm|scf", h):
        if "scfm" in h or re.search(r"scf\s*/\s*(?:min|m)", h):
            return HeaderInfo("n2_rate_scfm", "scfm", raw)
        return HeaderInfo("n2_rate_mmscfd", _gas_rate_unit(h), raw)
    if re.search(r"\b(?:dp|delta p)\s*1\b|differential pressure\s*1", h):
        return HeaderInfo("dp1_psi", _pressure_unit(h), raw)
    if re.search(r"\b(?:dp|delta p)\s*2\b|differential pressure\s*2", h):
        return HeaderInfo("dp2_psi", _pressure_unit(h), raw)
    if re.search(r"chart\s*(?:reading|rdg)|orifice chart", h):
        return HeaderInfo("gas_chart_reading", "", raw)
    if re.search(r"total\s*(?:liquid|fluid)\s*rate", h):
        return HeaderInfo("gross_rate_bpd", _liquid_rate_unit(h), raw)
    if re.fullmatch(r"frequency(?: hz)?", h):
        return HeaderInfo("pump_freq_hz", "Hz", raw)

    return _canonical_header_before_v68(raw)


def _convert_numeric(values: pd.Series, info: HeaderInfo) -> pd.Series:
    if info.canonical in {"n2_rate_mmscfd"}:
        out = clean_tabular_numeric_series(values)
        if info.unit == "scfd":
            out /= 1_000_000.0
        elif info.unit == "mscfd":
            out /= 1_000.0
        return out
    if info.canonical in {"dp1_psi", "dp2_psi"}:
        out = clean_tabular_numeric_series(values)
        if info.unit == "bar":
            out *= 14.5037738
        elif info.unit == "kpa":
            out *= 0.145037738
        elif info.unit == "mpa":
            out *= 145.037738
        return out
    return _convert_numeric_before_v68(values, info)


# ---------------------------------------------------------------------------
# Date/time parsing: explicit dates are authoritative; note rows never roll time
# ---------------------------------------------------------------------------
def _time_fraction(value: object) -> float:
    if value is None or _is_missing(value):
        return np.nan
    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()
    if isinstance(value, datetime):
        value = value.time()
    if isinstance(value, time):
        seconds = value.hour * 3600 + value.minute * 60 + value.second + value.microsecond / 1e6
        return seconds / 86400.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        if not math.isfinite(number):
            return np.nan
        fraction = number % 1.0
        if fraction > 0.999999 or fraction < 0.000001:
            fraction = 0.0
        seconds = int(round(fraction * 86400)) % 86400
        # Snap Excel floating artefacts such as 17:29:59.999 to a minute.
        minute_seconds = int(round(seconds / 60.0)) * 60
        if abs(seconds - minute_seconds) <= 2:
            seconds = minute_seconds % 86400
        return seconds / 86400.0

    text = safe_text(value)
    if not text:
        return np.nan
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.notna(parsed):
        parsed = pd.Timestamp(parsed)
        return (parsed.hour * 3600 + parsed.minute * 60 + parsed.second) / 86400.0
    match = re.search(r"\b(\d{1,2})[:.](\d{2})(?::(\d{2}))?\s*(am|pm)?\b", text, flags=re.I)
    if not match:
        return np.nan
    hour, minute = int(match.group(1)), int(match.group(2))
    second = int(match.group(3) or 0)
    ampm = (match.group(4) or "").lower()
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59 or second > 59:
        return np.nan
    return (hour * 3600 + minute * 60 + second) / 86400.0


def _date_only_with_base_v68(value: object, date_base: str = "1899-12-30") -> Optional[date]:
    if value is None or _is_missing(value):
        return None
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number) and 0 <= number <= 80000:
            try:
                return (pd.Timestamp(date_base) + pd.to_timedelta(number, unit="D")).date()
            except Exception:
                return None
    return _date_only(value)


def _xlsx_date_base_v68(zf: zipfile.ZipFile) -> str:
    try:
        root = ET.fromstring(zf.read("xl/workbook.xml"))
        workbook_pr = root.find(f"{{{_XLSX_NS}}}workbookPr")
        if workbook_pr is not None and str(workbook_pr.attrib.get("date1904", "0")).lower() in {"1", "true"}:
            return "1904-01-01"
    except Exception:
        pass
    return "1899-12-30"


def _repair_date_time_sequence_v68(
    date_values: Sequence[object],
    time_values: Sequence[object],
    measurement_mask: Optional[Sequence[bool]] = None,
    source_date_hint: Optional[date] = None,
    date_base: str = "1899-12-30",
) -> pd.Series:
    dates: List[Optional[date]] = [_date_only_with_base_v68(value, date_base) for value in date_values]
    fractions: List[float] = [_time_fraction(value) for value in time_values]
    is_measurement = list(measurement_mask) if measurement_mask is not None else [True] * len(dates)
    if len(is_measurement) != len(dates):
        is_measurement = [True] * len(dates)

    # Repair only isolated date typos surrounded by equal dates on measurement rows.
    for pos in range(1, len(dates) - 1):
        if not is_measurement[pos]:
            continue
        prev_d, cur_d, next_d = dates[pos - 1], dates[pos], dates[pos + 1]
        if prev_d is not None and next_d is not None and prev_d == next_d and cur_d not in {None, prev_d}:
            dates[pos] = prev_d

    # Supply dates only when missing. Never replace a valid explicit date with filename metadata.
    next_explicit: List[Optional[date]] = [None] * len(dates)
    following: Optional[date] = None
    for pos in range(len(dates) - 1, -1, -1):
        if dates[pos] is not None:
            following = dates[pos]
        next_explicit[pos] = following

    output: List[pd.Timestamp] = []
    inferred_date: Optional[date] = source_date_hint
    previous_clock: Optional[float] = None
    previous_explicit: Optional[date] = None

    for pos, (explicit_date, fraction) in enumerate(zip(dates, fractions)):
        if not math.isfinite(fraction):
            output.append(pd.NaT)
            continue
        clock_seconds = int(round(fraction * 86400)) % 86400
        clock_fraction = clock_seconds / 86400.0

        if explicit_date is not None:
            base_date = explicit_date
            if previous_explicit is not None and explicit_date > previous_explicit:
                inferred_date = explicit_date
            elif inferred_date is None:
                inferred_date = explicit_date
            elif previous_explicit == explicit_date and inferred_date > explicit_date:
                # The Date column is being copied down after an inferred midnight.
                # Keep the inferred day until the explicit Date finally advances.
                base_date = inferred_date
        else:
            base_date = inferred_date or next_explicit[pos] or source_date_hint
            if base_date is None:
                output.append(pd.NaT)
                continue

        if is_measurement[pos]:
            # Infer midnight only when the Date cell is missing or failed to advance.
            if previous_clock is not None and previous_clock - clock_fraction > 0.25:
                same_or_missing_date = explicit_date is None or previous_explicit is None or explicit_date <= previous_explicit
                if same_or_missing_date:
                    candidate = (inferred_date or base_date) + timedelta(days=1)
                    if explicit_date is None or candidate > explicit_date:
                        base_date = candidate
                        inferred_date = candidate
            if explicit_date is not None and (previous_explicit is None or explicit_date >= previous_explicit):
                # Explicit calendar advance wins over any inferred rollover.
                if previous_explicit is None or explicit_date > previous_explicit:
                    base_date = explicit_date
                    inferred_date = explicit_date
            previous_clock = clock_fraction
            if explicit_date is not None:
                previous_explicit = explicit_date
            elif inferred_date is None:
                inferred_date = base_date
        # Note/event rows are combined for audit but never update rollover state.
        current = pd.Timestamp(datetime.combine(base_date, time.min) + timedelta(seconds=clock_seconds))
        output.append(current)

    return pd.Series(output, dtype="datetime64[ns]")


def combine_date_time(
    date_series: pd.Series,
    time_series: pd.Series,
    measurement_mask: Optional[Sequence[bool]] = None,
    source_date_hint: Optional[date] = None,
    date_base: str = "1899-12-30",
) -> pd.Series:
    combined = _repair_date_time_sequence_v68(
        date_series.tolist(), time_series.tolist(), measurement_mask=measurement_mask,
        source_date_hint=source_date_hint, date_base=date_base,
    )
    combined.index = date_series.index
    return combined


# ---------------------------------------------------------------------------
# Quality checks and physically impossible-value rejection
# ---------------------------------------------------------------------------
_NONNEGATIVE_FIELDS_V68 = {
    "gas_rate_mmscfd", "gas_formation_mmscfd", "n2_rate_mmscfd", "n2_rate_scfm",
    "gross_rate_bpd", "oil_rate_stbd", "water_rate_bpd", "oil_cum_bbl",
    "water_cum_bbl", "gor_scf_bbl", "h2s_ppm", "co2_mole_pct", "salinity_kppm",
    "motor_ama_amp", "pump_freq_hz", "ct_depth_m", "ct_running_speed_ftmin",
    "orifice_size_in", "oil_meter_increment_bbl",
}
_RANGE_RULES_V68: Dict[str, Tuple[float, float]] = {
    "bsw_pct": (0.0, 100.0),
    "choke_pct": (0.0, 100.0),
    "co2_mole_pct": (0.0, 100.0),
    "gas_sg": (0.05, 5.0),
    "oil_api": (-10.0, 100.0),
    "pump_freq_hz": (0.0, 200.0),
    "motor_ama_amp": (0.0, 5000.0),
    "salinity_kppm": (0.0, 500.0),
    "orifice_size_in": (0.0, 20.0),
    "choke_size_64": (0.0, 512.0),
    "motor_temp_f": (-100.0, 1200.0),
    "motor_temp_c": (-100.0, 650.0),
    "gas_temp_f": (-100.0, 1000.0),
    "gas_temp_c": (-100.0, 550.0),
    "oil_temp_f": (-100.0, 1000.0),
    "oil_temp_c": (-100.0, 550.0),
}


def _append_quality_note_v68(out: pd.DataFrame, mask: pd.Series, message: str) -> None:
    if not mask.any():
        return
    current = out.loc[mask, "data_quality_note"].fillna("").astype(str)
    out.loc[mask, "data_quality_note"] = [append_note(value, message) for value in current]
    out.loc[mask, "review_required"] = True


def _apply_sanity_rules_v68(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "data_quality_note" not in out.columns:
        out["data_quality_note"] = ""
    if "review_required" not in out.columns:
        out["review_required"] = False
    if "rejected_values" not in out.columns:
        out["rejected_values"] = ""

    for field in sorted(_NONNEGATIVE_FIELDS_V68):
        if field not in out.columns:
            continue
        values = pd.to_numeric(out[field], errors="coerce")
        tiny_negative = values.notna() & (values < 0) & (values.abs() <= 1e-6)
        if tiny_negative.any():
            out.loc[tiny_negative, field] = 0.0
            values = pd.to_numeric(out[field], errors="coerce")
        invalid = values.notna() & (values < -1e-6)
        if invalid.any():
            for idx in out.index[invalid]:
                out.at[idx, "rejected_values"] = append_note(
                    out.at[idx, "rejected_values"], f"{field}={values.loc[idx]:g}"
                )
            out.loc[invalid, field] = np.nan
            _append_quality_note_v68(out, invalid, f"Rejected negative {column_label(field)}")

    for field, (minimum, maximum) in _RANGE_RULES_V68.items():
        if field not in out.columns:
            continue
        values = pd.to_numeric(out[field], errors="coerce")
        invalid = values.notna() & ((values < minimum) | (values > maximum))
        if invalid.any():
            for idx in out.index[invalid]:
                out.at[idx, "rejected_values"] = append_note(
                    out.at[idx, "rejected_values"], f"{field}={values.loc[idx]:g}"
                )
            out.loc[invalid, field] = np.nan
            _append_quality_note_v68(
                out, invalid,
                f"Rejected {column_label(field)} outside {minimum:g}–{maximum:g}",
            )

    # Consistency checks are warnings only; source values remain untouched.
    if {"gross_rate_bpd", "oil_rate_stbd", "water_rate_bpd"}.issubset(out.columns):
        gross = pd.to_numeric(out["gross_rate_bpd"], errors="coerce")
        oil = pd.to_numeric(out["oil_rate_stbd"], errors="coerce")
        water = pd.to_numeric(out["water_rate_bpd"], errors="coerce")
        valid = gross.notna() & oil.notna() & water.notna()
        tolerance = np.maximum(10.0, 0.15 * np.maximum(gross.abs(), 1.0))
        mismatch = valid & ((gross - (oil + water)).abs() > tolerance)
        _append_quality_note_v68(out, mismatch, "Gross rate differs materially from oil + water")
        oil_gt_gross = gross.notna() & oil.notna() & (oil > gross + np.maximum(5.0, gross.abs() * 0.05))
        _append_quality_note_v68(out, oil_gt_gross, "Oil rate exceeds gross rate")

    if {"gas_rate_mmscfd", "gas_formation_mmscfd"}.issubset(out.columns):
        total = pd.to_numeric(out["gas_rate_mmscfd"], errors="coerce")
        formation = pd.to_numeric(out["gas_formation_mmscfd"], errors="coerce")
        bad = total.notna() & formation.notna() & (formation > total + 1e-9)
        _append_quality_note_v68(out, bad, "Formation gas exceeds total gas")

    return out


def _canonical_numeric_columns_v68(df: pd.DataFrame) -> List[str]:
    return [
        col for col in df.columns
        if col not in BASE_NON_PLOT_COLS
        and not str(col).startswith("raw__")
        and col != "gas_rate_status"
        and pd.to_numeric(df[col], errors="coerce").notna().any()
    ]


def _table_quality_v68(df: pd.DataFrame) -> float:
    if df is None or df.empty or "datetime" not in df.columns:
        return -1e9
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    valid_dt = int(dt.notna().sum())
    if valid_dt == 0:
        return -1e9
    numeric_cols = _canonical_numeric_columns_v68(df)
    if not numeric_cols:
        return -1e9
    numeric_counts = df[numeric_cols].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
    coverage = float((numeric_counts > 0).mean())
    median_fields = float(numeric_counts.median()) if len(numeric_counts) else 0.0
    duplicate_ratio = float(dt.duplicated().mean())
    score = valid_dt + len(numeric_cols) * 8 + coverage * 50 + min(median_fields, 12) * 5
    score -= duplicate_ratio * 40
    sheet_name = safe_text(df.get("sheet", pd.Series([""])).iloc[0]) if len(df) else ""
    if re.search(r"(?:^|\b)(form|cover|summary|cmsf|shrinkage)(?:\b|$)", sheet_name, flags=re.I):
        score -= 120
    if "note" in df.columns:
        errors = df["note"].astype(str).str.contains(r"#REF!|#DIV/0!|#VALUE!|#N/A", regex=True, na=False)
        score -= float(errors.mean()) * 40
    return float(score)


# ---------------------------------------------------------------------------
# Deterministic table parser with measurement-row-aware time repair
# ---------------------------------------------------------------------------
def _parse_candidate(
    raw: pd.DataFrame,
    candidate: HeaderCandidate,
    *,
    source_name: str,
    sheet_name: str,
    default_well: str,
) -> pd.DataFrame:
    data = raw.iloc[candidate.start_pos + candidate.height :].copy().reset_index(drop=False)
    if data.empty:
        return pd.DataFrame()
    source_rows = data["index"].tolist()
    data = data.drop(columns=["index"])
    infos = candidate.infos
    canonical = [info.canonical for info in infos]

    output = pd.DataFrame(index=data.index)
    output["source"] = source_name
    output["sheet"] = sheet_name
    output["parser_engine"] = "v68_sparse_xml"
    output["source_row"] = source_rows
    output["source_priority"] = _source_priority(source_name)
    output["source_group"] = source_name + "::" + sheet_name

    if "source_meta" in canonical:
        idx = canonical.index("source_meta")
        values = data.iloc[:, idx].map(safe_text)
        output["source"] = values.where(values.ne(""), source_name)
    if "sheet_meta" in canonical:
        idx = canonical.index("sheet_meta")
        values = data.iloc[:, idx].map(safe_text)
        output["sheet"] = values.where(values.ne(""), sheet_name)
    if "source_type_meta" in canonical:
        output["source_type"] = data.iloc[:, canonical.index("source_type_meta")].map(safe_text)
    if "note_meta" in canonical:
        output["note"] = data.iloc[:, canonical.index("note_meta")].map(safe_text)
    if "gas_rate_status_meta" in canonical:
        output["gas_rate_status"] = data.iloc[:, canonical.index("gas_rate_status_meta")].map(safe_text)

    if "well" in canonical:
        well_values = data.iloc[:, canonical.index("well")].map(clean_well_name_value)
        output["well"] = well_values.where(well_values.ne("Unknown"), default_well)
    else:
        output["well"] = default_well

    occupied = {
        idx for idx, key in enumerate(canonical)
        if key in {"date", "time", "datetime", "well", "source_meta", "sheet_meta", "source_type_meta", "note_meta", "gas_rate_status_meta"}
    }
    recognized_fields: List[str] = []
    for col, info in enumerate(infos):
        key = info.canonical
        if key in {None, "date", "time", "datetime", "well", "source_meta", "sheet_meta", "source_type_meta", "note_meta", "gas_rate_status_meta"}:
            continue
        occupied.add(col)
        recognized_fields.append(key)
        converted = _convert_numeric(data.iloc[:, col], info)
        if key in output.columns:
            output[key] = pd.to_numeric(output[key], errors="coerce").combine_first(converted)
        else:
            output[key] = converted

    canonical_row_counts = pd.Series(0, index=data.index, dtype=int)
    for key in sorted(set(recognized_fields)):
        if key in output.columns:
            canonical_row_counts += pd.to_numeric(output[key], errors="coerce").notna().astype(int)

    # A lone calculated zero (commonly Gross Rate before the test starts) is not
    # enough to create a measurement row. Accept rows with two measurements, a
    # strong pressure/device/gas anchor, or a non-zero production rate.
    strong_fields = {
        "whp_psi", "flp_psi", "sep_p_psi", "pumping_pressure_psi",
        "pump_intake_pressure_psi", "pump_discharge_pressure_psi",
        "motor_ama_amp", "pump_freq_hz", "motor_temp_f", "motor_temp_c",
        "gas_rate_mmscfd", "gas_formation_mmscfd", "n2_rate_mmscfd",
        "n2_rate_scfm", "ct_pressure_psi", "ct_depth_m",
    }
    strong_present = pd.Series(False, index=data.index, dtype=bool)
    for key in strong_fields:
        if key in output.columns:
            strong_present |= pd.to_numeric(output[key], errors="coerce").notna()
    nonzero_rate = pd.Series(False, index=data.index, dtype=bool)
    for key in {"gross_rate_bpd", "oil_rate_stbd", "water_rate_bpd"}:
        if key in output.columns:
            values = pd.to_numeric(output[key], errors="coerce")
            nonzero_rate |= values.notna() & values.abs().gt(1e-12)
    measurement_mask = canonical_row_counts.ge(2) | strong_present | nonzero_rate

    # Date/time is parsed only after identifying real measurement rows, so note
    # timestamps cannot trigger midnight rollover.
    source_date_hint = _date_from_name(source_name)
    date_base = str(raw.attrs.get("excel_date_base", "1899-12-30"))
    if "datetime" in canonical:
        idx = canonical.index("datetime")
        dt_values = [_safe_datetime_scalar(value) for value in data.iloc[:, idx]]
        dt = pd.Series(dt_values, index=data.index, dtype="datetime64[ns]")
    elif "date" in canonical and "time" in canonical:
        date_idx, time_idx = canonical.index("date"), canonical.index("time")
        dt = combine_date_time(
            data.iloc[:, date_idx], data.iloc[:, time_idx],
            measurement_mask=measurement_mask.tolist(), source_date_hint=source_date_hint,
            date_base=date_base,
        )
    elif "date" in canonical:
        date_idx = canonical.index("date")
        dt = pd.Series(
            [_safe_datetime_scalar(value) for value in data.iloc[:, date_idx]],
            index=data.index, dtype="datetime64[ns]",
        )
    elif "time" in canonical and source_date_hint is not None:
        time_idx = canonical.index("time")
        date_values = pd.Series([source_date_hint] * len(data), index=data.index)
        dt = combine_date_time(
            date_values, data.iloc[:, time_idx], measurement_mask=measurement_mask.tolist(),
            source_date_hint=source_date_hint, date_base=date_base,
        )
    else:
        return pd.DataFrame()
    output["datetime"] = dt

    # Preserve unknown channels only for compact/simple exports.
    unique_recognized = len(set(recognized_fields))
    if unique_recognized <= 12:
        for col, raw_key in _generic_numeric_columns(data, candidate.headers, occupied).items():
            output[raw_key] = clean_tabular_numeric_series(data.iloc[:, col])
            COLUMN_LABELS.setdefault(raw_key, safe_text(candidate.headers[col]) or raw_key)

    # Notes are collected only from non-mapped text cells on measurement rows;
    # formula errors and repeated headers are ignored.
    notes = output.get("note", pd.Series("", index=output.index, dtype=object)).fillna("").astype(object)
    for col in range(data.shape[1]):
        if col in occupied or canonical[col] in {"date", "time", "datetime", "well"}:
            continue
        values = data.iloc[:, col].map(safe_text)
        values = values.where(~values.str.contains(r"#REF!|#DIV/0!|#VALUE!|#N/A", regex=True, na=False), "")
        values = values.where(~values.str.fullmatch(r"[-+]?\d+(?:[.,]\d+)?(?:[eE][-+]?\d+)?", na=False), "")
        values = values.where(measurement_mask, "")
        if values.str.len().gt(3).any():
            notes = pd.Series([append_note(a, b) for a, b in zip(notes, values)], index=notes.index)
    output["note"] = notes

    # A row is useful only when it has a timestamp and a recognized engineering
    # measurement. Generic hidden/formula columns alone cannot create rows.
    useful = output["datetime"].notna() & measurement_mask
    output = output.loc[useful].copy()
    if output.empty:
        return output

    if "motor_temp_f" in output.columns:
        values = pd.to_numeric(output["motor_temp_f"], errors="coerce")
        infer_unit = any(info.canonical == "motor_temp_f" and info.unit == "infer" for info in infos)
        if infer_unit and values.notna().any() and values.median() < 180:
            output["motor_temp_c"] = values
            output.drop(columns=["motor_temp_f"], inplace=True)

    output["date"] = output["datetime"].dt.date
    output["time_text"] = output["datetime"].dt.strftime("%H:%M")
    output["test_unit"] = sheet_name
    return _postprocess_table(output)


def _postprocess_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "datetime" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.round("s")
    out = out.loc[out["datetime"].notna()].copy()
    if out.empty:
        return out
    if "well" not in out.columns:
        out["well"] = "Unknown"
    out["well"] = out["well"].map(clean_well_name_value)

    for col in list(out.columns):
        if col in BASE_NON_PLOT_COLS or col == "gas_rate_status":
            continue
        numeric = pd.to_numeric(out[col], errors="coerce")
        if numeric.notna().any():
            out[col] = numeric.astype("float64")

    if "choke_pct" in out.columns:
        values = pd.to_numeric(out["choke_pct"], errors="coerce")
        out["choke_pct"] = values.where(~((values > 0) & (values <= 1.0)), values * 100.0)
    if "choke_size_64" in out.columns:
        values = pd.to_numeric(out["choke_size_64"], errors="coerce")
        inch = values.notna() & (values > 0) & (values <= 2.0)
        out["choke_size_64"] = values.where(~inch, values * 64.0)
    if "salinity_kppm" in out.columns:
        values = pd.to_numeric(out["salinity_kppm"], errors="coerce")
        out["salinity_kppm"] = values.where(values.abs() <= 1000, values / 1000.0)

    # Remove the known temporary compatibility copy of intake pressure.
    if "pumping_pressure_psi" in out.columns and "pump_intake_pressure_psi" in out.columns:
        pump = pd.to_numeric(out["pumping_pressure_psi"], errors="coerce")
        intake = pd.to_numeric(out["pump_intake_pressure_psi"], errors="coerce")
        source = out.get("source", pd.Series("", index=out.index)).astype(str)
        compatibility = source.str.contains(r"legacy|compatible|temporary", case=False, regex=True, na=False)
        copied = pump.notna() & intake.notna() & np.isclose(pump, intake, rtol=1e-9, atol=1e-9) & compatibility
        out.loc[copied, "pumping_pressure_psi"] = np.nan
        if pd.to_numeric(out["pumping_pressure_psi"], errors="coerce").notna().sum() == 0:
            out.drop(columns=["pumping_pressure_psi"], inplace=True)

    for col, default in {
        "source": "Unknown source", "sheet": "Data", "note": "", "test_unit": "Data",
        "source_type": "tabular", "link_status": "source_confirmed", "parser_engine": "v68",
    }.items():
        if col not in out.columns:
            out[col] = default
    out = _apply_sanity_rules_v68(out)
    out["date"] = out["datetime"].dt.date
    out["time_text"] = out["datetime"].dt.strftime("%H:%M")
    numeric_cols = _canonical_numeric_columns_v68(out)
    if numeric_cols:
        counts = out[numeric_cols].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1)
        out["parse_confidence"] = np.minimum(1.0, 0.45 + counts / max(2.0, min(10.0, len(numeric_cols))))
    else:
        out["parse_confidence"] = 0.0
    return out.sort_values(["well", "datetime", "source", "sheet"], kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Multi-engine Excel parsing and conservative reconciliation
# ---------------------------------------------------------------------------
def _merge_engine_complements_v68(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    if primary is None or primary.empty:
        return secondary
    if secondary is None or secondary.empty:
        return primary
    left = primary.copy()
    right = secondary.copy()
    left["_key"] = pd.to_datetime(left["datetime"], errors="coerce").dt.round("min")
    right["_key"] = pd.to_datetime(right["datetime"], errors="coerce").dt.round("min")
    overlap = set(left["_key"].dropna()) & set(right["_key"].dropna())
    denominator = max(1, min(left["_key"].notna().sum(), right["_key"].notna().sum()))
    if len(overlap) / denominator < 0.60:
        return primary
    right_by_key = right.sort_values("_key").drop_duplicates("_key", keep="last").set_index("_key")
    for idx in left.index:
        key = left.at[idx, "_key"]
        if pd.isna(key) or key not in right_by_key.index:
            continue
        source_row = right_by_key.loc[key]
        if isinstance(source_row, pd.DataFrame):
            source_row = source_row.iloc[-1]
        for col in right.columns:
            if col == "_key" or col.startswith("_"):
                continue
            if col not in left.columns:
                left[col] = np.nan if col not in BASE_NON_PLOT_COLS else ""
            if col == "note":
                left.at[idx, col] = append_note(left.at[idx, col], source_row.get(col))
            elif _is_missing(left.at[idx, col]) and not _is_missing(source_row.get(col)):
                left.at[idx, col] = source_row.get(col)
    return _postprocess_table(left.drop(columns=["_key"], errors="ignore"))


def _choose_engine_table_v68(candidates: Sequence[pd.DataFrame]) -> Optional[pd.DataFrame]:
    valid = [table for table in candidates if is_valid_timeseries(table)]
    if not valid:
        return None
    ranked = sorted(valid, key=lambda table: (_table_quality_v68(table), len(table), len(_canonical_numeric_columns_v68(table))), reverse=True)
    primary = ranked[0]
    for secondary in ranked[1:]:
        primary = _merge_engine_complements_v68(primary, secondary)
    return primary


def _parse_xlsx(data: bytes, name: str) -> List[pd.DataFrame]:
    deterministic: List[pd.DataFrame] = []
    legacy_tables: List[pd.DataFrame] = []
    diagnostics: List[str] = []
    simple_table = False

    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            shared = _read_shared_strings(zf)
            sheet_paths = _workbook_sheet_paths(zf)
            workbook_date_base = _xlsx_date_base_v68(zf)
            simple_table = _xlsx_simple_table_signature(zf, sheet_paths, shared)
            for sheet_name, sheet_path in sheet_paths:
                try:
                    raw = _read_sheet_sparse(
                        zf, sheet_path, shared, max_rows=50000, max_cols=1024,
                        max_nonempty_cells=1_000_000,
                    )
                except Exception as exc:
                    diagnostics.append(f"{sheet_name} sparse read: {exc}")
                    continue
                if raw.empty:
                    continue
                raw.attrs["excel_date_base"] = workbook_date_base
                default_well = _well_from_raw(raw, name, sheet_name)
                parsed_for_sheet: List[pd.DataFrame] = []
                for candidate in _find_header_candidates(raw):
                    try:
                        table = _parse_candidate(
                            raw, candidate, source_name=name, sheet_name=sheet_name,
                            default_well=default_well,
                        )
                    except Exception as exc:
                        diagnostics.append(f"{sheet_name} header row {candidate.start_pos + 1}: {exc}")
                        continue
                    if is_valid_timeseries(table):
                        parsed_for_sheet.append(table)
                chosen = _choose_engine_table_v68(parsed_for_sheet)
                if chosen is not None:
                    deterministic.append(chosen)
    except Exception as exc:
        diagnostics.append(f"xlsx sparse engine: {exc}")

    # The legacy engine remains valuable for rare templates, old .xlsx quirks,
    # PDF-derived tables and OCR-linked fields. It is now a secondary engine,
    # never the sole decision maker when the deterministic parser succeeds.
    deterministic_good = any(
        len(table) >= 3
        and len(_canonical_numeric_columns_v68(table)) >= 2
        and _table_quality_v68(table) >= 80
        for table in deterministic
    )
    if not deterministic_good:
        try:
            raw_legacy = legacy.load_tabular_file(UploadedBytes(data, name), parse_images=False, max_ocr_images=0)
            for table in raw_legacy:
                normalized = _postprocess_table(table)
                if is_valid_timeseries(normalized):
                    normalized["parser_engine"] = "legacy"
                    legacy_tables.append(normalized)
        except Exception as exc:
            diagnostics.append(f"legacy engine: {exc}")

    # Group by well/sheet and reconcile the best interpretations.
    all_tables = deterministic + legacy_tables
    if not all_tables:
        detail = "; ".join(diagnostics[:8])
        raise RuntimeError(
            "no usable time-series table detected. The workbook was read safely, but no table "
            f"contained a valid date/time plus engineering readings. {detail}".strip()
        )

    grouped: Dict[Tuple[str, str], List[pd.DataFrame]] = {}
    for table in all_tables:
        well = _well_key(table["well"].iloc[0]) if "well" in table.columns and len(table) else ""
        sheet = normalize_text(table["sheet"].iloc[0]) if "sheet" in table.columns and len(table) else "data"
        grouped.setdefault((well, sheet), []).append(table)

    selected: List[pd.DataFrame] = []
    for (_well, sheet_key), candidates in grouped.items():
        chosen = _choose_engine_table_v68(candidates)
        if chosen is None:
            continue
        if re.search(r"(?:^|\b)(form|cover|summary|cmsf|shrinkage)(?:\b|$)", sheet_key, flags=re.I):
            if _table_quality_v68(chosen) < 60:
                continue
        selected.append(chosen)

    selected = _deduplicate_table_interpretations(selected)
    selected.sort(key=lambda table: (_table_quality_v68(table), len(table)), reverse=True)
    return selected


# ---------------------------------------------------------------------------
# Robust delimited data, including decimal-comma and dashboard re-imports
# ---------------------------------------------------------------------------
def _parse_delimited(data: bytes, name: str) -> List[pd.DataFrame]:
    text, _encoding = _decode_text_bytes(data)
    if not text.strip():
        raise RuntimeError("file is blank")
    separator = _detect_delimiter(text)
    try:
        raw = pd.read_csv(
            io.StringIO(text), sep=separator, header=None, dtype=object,
            engine="python", keep_default_na=False, na_values=[], on_bad_lines="skip",
        )
    except Exception as exc:
        raise RuntimeError(f"could not read delimited text: {exc}") from exc
    raw = raw.replace(r"^\s*$", np.nan, regex=True).dropna(axis=0, how="all").dropna(axis=1, how="all")
    if raw.empty:
        wa = parse_whatsapp_plain_or_export_text(text, source_name=name)
        if is_valid_timeseries(wa):
            return [wa]
        raise RuntimeError("file is blank")
    raw.index = range(1, len(raw) + 1)
    default_well = _well_from_raw(raw, name, "CSV")
    candidates: List[pd.DataFrame] = []
    for header in _find_header_candidates(raw):
        try:
            table = _parse_candidate(raw, header, source_name=name, sheet_name="CSV", default_well=default_well)
        except Exception:
            continue
        if is_valid_timeseries(table):
            candidates.append(table)
    chosen = _choose_engine_table_v68(candidates)
    if chosen is not None:
        return [chosen]
    wa = parse_whatsapp_plain_or_export_text(text, source_name=name)
    if is_valid_timeseries(wa):
        return [wa]
    raise RuntimeError(
        "no usable time-series table detected. A DateTime column, or Date + Time columns, "
        "and at least one recognized numeric measurement are required."
    )


# ---------------------------------------------------------------------------
# WhatsApp parsing with inline labels and explicit choke/gas semantics
# ---------------------------------------------------------------------------
_WHATSAPP_LABELS_V68: List[Tuple[str, str]] = [
    ("total_gas", r"total\s+gas\s+rate"),
    ("formation_gas", r"formation\s+gas(?:\s+rate)?"),
    ("gross", r"gross\s+rate"),
    ("oil", r"(?:oil|cond(?:ensate)?)\s+rate"),
    ("water", r"water\s+rate"),
    ("water_cum", r"water\s+cum(?:ulative)?\.?"),
    ("oil_cum", r"oil\s+cum(?:ulative)?\.?"),
    ("gas", r"gas\s+rate"),
    ("pumping", r"(?:pumping\s*p(?:ressure)?|pump\s*p(?:ressure)?)"),
    ("sep_p", r"(?:sep\.?\s*p\.?|separator\s+pressure)"),
    ("whp", r"(?:w\.?\s*h\.?\s*p\.?|whp|wellhead\s+pressure)"),
    ("choke", r"choke"),
    ("bsw", r"(?:bs\s*&?\s*w|bsw|water\s*cut)"),
    ("salinity", r"salinity"),
    ("h2s", r"h\s*2\s*s"),
    ("co2", r"co\s*2"),
    ("well", r"well\s*name|well"),
    ("date", r"date"),
    ("time", r"time"),
    ("note", r"note"),
]


def _clean_whatsapp_text_v68(text: object) -> str:
    value = str(text or "")
    value = value.replace("\u200e", "").replace("\u200f", "").replace("\ufeff", "")
    value = value.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"[*_`~]+", "", value)
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def _extract_whatsapp_fields_v68(text: str) -> Dict[str, str]:
    cleaned = _clean_whatsapp_text_v68(text)
    occurrences: List[Tuple[int, int, str]] = []
    for key, pattern in _WHATSAPP_LABELS_V68:
        regex = re.compile(rf"(?im)(?<![A-Za-z0-9])(?:{pattern})(?![A-Za-z0-9])\s*(?:[:=@-]\s*)?")
        for match in regex.finditer(cleaned):
            occurrences.append((match.start(), match.end(), key))
    occurrences.sort(key=lambda item: (item[0], -(item[1] - item[0])))

    # Remove nested/overlapping short matches (e.g. gas rate inside total gas rate).
    filtered: List[Tuple[int, int, str]] = []
    for item in occurrences:
        if filtered and item[0] < filtered[-1][1]:
            continue
        filtered.append(item)

    fields: Dict[str, str] = {}
    for pos, (_start, end, key) in enumerate(filtered):
        next_start = filtered[pos + 1][0] if pos + 1 < len(filtered) else len(cleaned)
        value = cleaned[end:next_start]
        value = re.sub(r"^[\s:=@-]+", "", value)
        value = re.sub(r"[\n;]+$", "", value).strip()
        # Avoid swallowing the next report header or generic production-test marker.
        value = re.split(r"(?im)\n\s*(?:PICO\s*TMU|TMU)\s*[- ]?\d+", value, maxsplit=1)[0].strip()
        value = re.split(r"(?im)\n\s*production\s+test\s*$", value, maxsplit=1)[0].strip()
        if value and key not in fields:
            fields[key] = value
    return fields


def split_messages(text: str) -> List[str]:
    cleaned = _clean_whatsapp_text_v68(text)
    if not cleaned:
        return []
    header = re.compile(r"(?im)^\s*(?:PICO\s*TMU|TMU)\s*[- ]?\s*\d+\b[^\n]*")
    starts = [match.start() for match in header.finditer(cleaned)]
    if len(starts) <= 1:
        # Multiple reports without repeated TMU header: split on repeated Date labels.
        date_starts = [
            match.start() for match in re.finditer(
                r"(?im)^\s*date\s*(?:[:=@-]\s*)?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", cleaned
            )
        ]
        if len(date_starts) > 1:
            starts = date_starts
    if len(starts) <= 1:
        return [cleaned]
    return [
        cleaned[start : (starts[pos + 1] if pos + 1 < len(starts) else len(cleaned))].strip()
        for pos, start in enumerate(starts)
    ]


def _parse_choke_text_v68(raw: object, row: Dict[str, object]) -> None:
    text = safe_text(raw)
    value = extract_number(text)
    if pd.isna(value):
        return
    normalized = normalize_text(text)
    fraction_match = re.search(r"([-+]?\d+(?:[.,]\d+)?)\s*/\s*64", text)
    if fraction_match:
        parsed = _normalize_numeric_token_v68(fraction_match.group(1))
        if parsed is not None:
            row["choke_size_64"] = float(parsed)
        return
    if "%" in text or "percent" in normalized or "opening" in normalized:
        row["choke_pct"] = float(value)
        return
    if '"' in text or "inch" in normalized or re.search(r"\bin\b", normalized):
        row["choke_size_64"] = float(value) * 64.0 if abs(value) <= 2 else float(value)
        return
    if 0 < value <= 1:
        row["choke_pct"] = float(value) * 100.0
    elif value > 2:
        row["choke_size_64"] = float(value)
    else:
        row["choke_ambiguous"] = float(value)


def parse_tmu_message(message: str, source_name: str = "WhatsApp_Text") -> Dict[str, object]:
    text = _clean_whatsapp_text_v68(message)
    fields = _extract_whatsapp_fields_v68(text)
    date_value = _safe_datetime_scalar(fields.get("date", ""))
    fraction = _time_fraction(fields.get("time", ""))
    dt = pd.NaT
    if pd.notna(date_value) and math.isfinite(fraction):
        dt = pd.Timestamp(date_value).normalize() + pd.to_timedelta(fraction, unit="D")

    well = clean_well_name_value(fields.get("well", ""))
    if well == "Unknown":
        well = guess_well_from_name(text)
    unit_match = re.search(r"(?im)^\s*((?:PICO\s*)?TMU\s*[- ]?\s*\d+)\b", text)
    test_unit = re.sub(r"\s+", " ", unit_match.group(1)).strip() if unit_match else "WhatsApp"

    row: Dict[str, object] = {
        "source": source_name,
        "sheet": "WhatsApp",
        "source_type": "pasted_whatsapp_text",
        "parser_engine": "v68_whatsapp",
        "well": well,
        "datetime": dt,
        "note": "",
        "test_unit": test_unit,
        "source_priority": _source_priority(source_name),
        "source_group": source_name,
    }

    _parse_choke_text_v68(fields.get("choke", ""), row)
    numeric_map = {
        "whp": "whp_psi",
        "sep_p": "sep_p_psi",
        "gross": "gross_rate_bpd",
        "oil": "oil_rate_stbd",
        "water": "water_rate_bpd",
        "water_cum": "water_cum_bbl",
        "oil_cum": "oil_cum_bbl",
        "bsw": "bsw_pct",
        "salinity": "salinity_kppm",
        "h2s": "h2s_ppm",
        "co2": "co2_mole_pct",
        "pumping": "pumping_pressure_psi",
    }
    for raw_key, canonical_key in numeric_map.items():
        raw_value = fields.get(raw_key, "")
        number = extract_number(raw_value)
        if pd.isna(number):
            continue
        if canonical_key == "salinity_kppm":
            norm = normalize_text(raw_value)
            if "ppm" in norm and "kppm" not in norm and "k ppm" not in norm:
                number /= 1000.0
        row[canonical_key] = float(number)

    total_raw = fields.get("total_gas") or fields.get("gas", "")
    total_value = extract_number(total_raw)
    if pd.notna(total_value):
        row["gas_rate_mmscfd"] = float(total_value)
    else:
        norm = normalize_text(total_raw)
        status = None
        if "low gas" in norm:
            status = "Low gas"
        elif "trace gas" in norm:
            status = "Trace gas"
        elif re.search(r"\b(?:no|zero|nil)\s+gas\b", norm):
            status = "No gas"
        elif norm:
            status = safe_text(total_raw)
        if status:
            row["gas_rate_status"] = status
            row["note"] = append_note(row.get("note"), f"Gas rate: {status}")

    formation_value = extract_number(fields.get("formation_gas", ""))
    if pd.notna(formation_value):
        row["gas_formation_mmscfd"] = float(formation_value)
    if pd.notna(total_value) and pd.notna(formation_value):
        derived_n2 = float(total_value) - float(formation_value)
        if derived_n2 >= -1e-9:
            row["n2_rate_mmscfd"] = max(0.0, derived_n2)

    note = safe_text(fields.get("note", ""))
    if note:
        row["note"] = append_note(row.get("note"), note)
    if re.search(r"(?im)^\s*production\s+test\s*$", text):
        row["note"] = append_note(row.get("note"), "Production test")
    return row


def parse_many_tmu_messages(text: str, source_name: str = "WhatsApp_Text") -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for index, chunk in enumerate(split_messages(text), start=1):
        row = parse_tmu_message(chunk, source_name=source_name)
        numeric_count = sum(
            pd.notna(extract_number(row.get(key)))
            for key in CANONICAL_NUMERIC_FIELDS if key in row
        )
        if pd.notna(row.get("datetime")) and numeric_count >= 1:
            row["message_index"] = index
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return _postprocess_table(pd.DataFrame(rows))


def parse_whatsapp_plain_or_export_text(text: str, source_name: str = "WhatsApp_Text") -> pd.DataFrame:
    direct = parse_many_tmu_messages(text, source_name=source_name)
    exported = pd.DataFrame()
    try:
        exported = legacy.parse_whatsapp_export_text(text, source_name=source_name)
        exported = _postprocess_table(exported) if exported is not None and not exported.empty else pd.DataFrame()
    except Exception:
        exported = pd.DataFrame()
    chosen = _choose_engine_table_v68([table for table in [direct, exported] if table is not None and not table.empty])
    return chosen if chosen is not None else pd.DataFrame()


# ---------------------------------------------------------------------------
# Duplicate merge with conflict audit
# ---------------------------------------------------------------------------
def merge_duplicate_test_rows_v53(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    out["datetime"] = pd.to_datetime(out.get("datetime"), errors="coerce").dt.round("min")
    out["well"] = out.get("well", pd.Series("Unknown", index=out.index)).map(clean_well_name_value)
    out["_well_key"] = out["well"].map(_well_key)
    out["_minute_key"] = out["datetime"]
    out["_source_order"] = np.arange(len(out), dtype=int)
    out["source_priority"] = pd.to_numeric(out.get("source_priority", pd.Series(0, index=out.index)), errors="coerce").fillna(0)
    if "source" in out.columns:
        out["source_priority"] += out["source"].map(_source_priority)
    out["_row_completeness"] = out.apply(_row_completeness, axis=1)

    merged_rows: List[Dict[str, object]] = []
    valid = out.loc[out["_well_key"].ne("") & out["_minute_key"].notna()].copy()
    invalid = out.loc[~(out["_well_key"].ne("") & out["_minute_key"].notna())].copy()
    for _, group in valid.groupby(["_well_key", "_minute_key"], sort=False, dropna=False):
        ranked = group.sort_values(
            ["_row_completeness", "source_priority", "_source_order"],
            ascending=[False, False, False], kind="stable",
        )
        base = ranked.iloc[0].to_dict()
        conflict_notes: List[str] = []
        for _, row in ranked.iloc[1:].iterrows():
            for col in out.columns:
                if col.startswith("_"):
                    continue
                incoming = row.get(col)
                existing = base.get(col)
                if col in {"note", "data_quality_note", "rejected_values"}:
                    base[col] = append_note(existing, incoming)
                elif _is_missing(existing) and not _is_missing(incoming):
                    base[col] = incoming
                elif (
                    col not in BASE_NON_PLOT_COLS and col != "gas_rate_status"
                    and not _is_missing(existing) and not _is_missing(incoming)
                ):
                    a, b = extract_number(existing), extract_number(incoming)
                    if pd.notna(a) and pd.notna(b):
                        tolerance = max(1e-9, 0.02 * max(abs(a), abs(b), 1.0))
                        if abs(a - b) > tolerance:
                            conflict_notes.append(f"Conflicting {column_label(col)} values: {a:g} vs {b:g}")
        if conflict_notes:
            base["data_quality_note"] = append_note(base.get("data_quality_note"), "; ".join(conflict_notes[:6]))
            base["review_required"] = True
        base["datetime"] = pd.Timestamp(base["_minute_key"])
        merged_rows.append(base)

    merged = pd.DataFrame(merged_rows)
    if not invalid.empty:
        merged = pd.concat([merged, invalid], ignore_index=True, sort=False)
    merged.drop(columns=["_well_key", "_minute_key", "_source_order", "_row_completeness"], inplace=True, errors="ignore")
    return _postprocess_table(merged)


# v68 performance override: shortlist likely header rows before expensive scoring.
def _find_header_candidates(raw: pd.DataFrame) -> List[HeaderCandidate]:
    if raw is None or raw.empty:
        return []
    scan_rows = min(len(raw), 160)
    likely_starts: set[int] = set()
    row_tokens: List[str] = []
    for pos in range(scan_rows):
        values = [normalize_text(value) for value in raw.iloc[pos].tolist() if safe_text(value)]
        joined = " | ".join(values)
        row_tokens.append(joined)
        has_date = bool(re.search(r"(?:^|\| |\b)(?:date|d/mm|dd/mm|timestamp|datetime)(?:\b| |\|)", joined))
        has_time = bool(re.search(r"(?:^|\| |\b)(?:time|hh:mm|timestamp|datetime)(?:\b| |\|)", joined))
        if has_date and has_time:
            likely_starts.update({max(0, pos - 2), max(0, pos - 1), pos})

    # Date and Time can be on adjacent header rows.
    for pos in range(max(0, scan_rows - 1)):
        window = " | ".join(row_tokens[pos : min(scan_rows, pos + 3)])
        if re.search(r"\b(?:date|d/mm|dd/mm|timestamp|datetime)\b", window) and re.search(r"\b(?:time|hh:mm|timestamp|datetime)\b", window):
            likely_starts.add(pos)

    if not likely_starts:
        # Small fallback shortlist, not a full 120 x 3 brute-force scan.
        likely_starts.update(range(min(scan_rows, 25)))

    candidates: List[HeaderCandidate] = []
    for start in sorted(likely_starts):
        for height in (1, 2, 3, 4, 5, 6):
            if start + height > len(raw):
                continue
            candidate = _header_candidate_score(raw, start, height)
            if candidate.score >= 16:
                candidates.append(candidate)
    candidates.sort(key=lambda item: item.score, reverse=True)
    selected: List[HeaderCandidate] = []
    for candidate in candidates:
        if any(abs(candidate.start_pos - previous.start_pos) <= 3 for previous in selected):
            continue
        selected.append(candidate)
        if len(selected) >= 6:
            break
    return selected

# v68 canonical unknown-well handling. Unknown OCR/message rows must never merge
# with each other merely because they share the literal text "UNKNOWN".
_clean_well_name_value_before_v68 = clean_well_name_value


def clean_well_name_value(value: object) -> str:
    normalized = normalize_text(value)
    if normalized in {
        "", "unknown", "unk", "n/a", "na", "none", "not known",
        "unlinked", "unknown well", "well unknown",
    }:
        return "Unknown"
    result = _clean_well_name_value_before_v68(value)
    return "Unknown" if normalize_text(result) in {"unknown", "unk", "n/a", "na"} else result


def normalize_well_name(value: object) -> str:
    return clean_well_name_value(value)


def _well_key(value: object) -> str:
    cleaned = clean_well_name_value(value)
    if cleaned == "Unknown":
        return ""
    return re.sub(r"[^A-Z0-9]", "", _normalize_bed_device_alias(cleaned).upper())

# Extend physical checks for OCR totals/rates. Values are kept auditable in
# rejected_values and the row remains visible in the review panel.
_NONNEGATIVE_FIELDS_V68.update({
    "ctu_fluid_rate_bpm", "ctu_n2_rate_scfm", "ctu_fluid_total_bbl",
    "ctu_n2_total_scf", "ctu_reel_depth_ft",
})

# Wrap post-processing once more so OCR confidence is not overstated by the
# number of fields detected.
_postprocess_table_v68_core = _postprocess_table


def _postprocess_table(df: pd.DataFrame) -> pd.DataFrame:
    out = _postprocess_table_v68_core(df)
    if out is None or out.empty:
        return out
    if "source_type" in out.columns:
        ocr_mask = out["source_type"].astype(str).str.contains("ocr", case=False, na=False)
        if ocr_mask.any() and "ocr_confidence" in out.columns:
            ocr_conf = pd.to_numeric(out.loc[ocr_mask, "ocr_confidence"], errors="coerce").clip(0, 1)
            out.loc[ocr_mask, "parse_confidence"] = ocr_conf
    return out

# Correct standalone implementation (avoids recursive dependency between
# clean_well_name_value and _well_key).
def clean_well_name_value(value: object) -> str:
    text = safe_text(value).upper().replace("*", "")
    text = re.sub(r"\bWELL(?:\s+NAME)?\b\s*[:=-]?", "", text, flags=re.I).strip()
    if normalize_text(text) in {
        "", "unknown", "unk", "n/a", "na", "none", "not known",
        "unlinked", "unknown well", "well unknown",
    }:
        return "Unknown"
    text = _normalize_bed_device_alias(text)
    text = re.sub(r"\s+", "", text).replace("_", "-")
    text = re.sub(r"-{2,}", "-", text).strip("-")
    compound = re.fullmatch(r"([A-Z]\d{1,2})-([A-Z]\d+)-(\d+)", text)
    if compound:
        text = f"{compound.group(1)}{compound.group(2)}-{compound.group(3)}"
    if not re.search(r"[A-Z0-9]", text) or normalize_text(text) in {"unknown", "unk", "na"}:
        return "Unknown"
    return text


def normalize_well_name(value: object) -> str:
    return clean_well_name_value(value)


def _well_key(value: object) -> str:
    cleaned = clean_well_name_value(value)
    if cleaned == "Unknown":
        return ""
    return re.sub(r"[^A-Z0-9]", "", _normalize_bed_device_alias(cleaned).upper())

# Metadata that must never appear as selectable engineering curves.
BASE_NON_PLOT_COLS.update({
    "chat_datetime", "ocr_fields_found", "ocr_confidence", "ocr_status",
    "ocr_template", "image_file", "attachment_name", "source_member",
    "chat_sender", "caption_text", "suggested_well", "suggested_test_id",
    "suggested_link_reason", "row_id", "image_index", "source_member_index",
})

# =============================================================================
# v70 finalization: OCR audit schema + conservative gas-balance reconciliation
# =============================================================================
PARSER_BUILD_ID_V72 = "v72-ui-contrast-srp-expro-gas-unit-fix-20260626"
PARSER_BUILD_ID = PARSER_BUILD_ID_V72

COLUMN_LABELS.update({
    "gas_balance_status": "Gas Balance Status",
    "gas_formation_derived": "Formation Gas Derived",
    "n2_rate_derived": "N2 Rate Derived",
    "total_gas_derived": "Total Gas Derived",
    "ocr_low_confidence_fields": "OCR Low-Confidence Fields",
    "screen_rectified": "Screen Rectified",
    "screen_detection_score": "Screen Detection Confidence",
})

BASE_NON_PLOT_COLS.update({
    "gas_balance_status", "gas_formation_derived", "n2_rate_derived",
    "total_gas_derived", "ocr_low_confidence_fields", "ocr_build_id",
    "screen_rectified", "screen_detection_score", "screen_detection_method",
    "screen_area_ratio", "screen_aspect_ratio",
})


def _record_reconciled_source_v70(out: pd.DataFrame, index, field: str, old_value, message: str) -> None:
    if "rejected_values" not in out.columns:
        out["rejected_values"] = ""
    if "data_quality_note" not in out.columns:
        out["data_quality_note"] = ""
    if "review_required" not in out.columns:
        out["review_required"] = False
    if pd.notna(old_value):
        out.at[index, "rejected_values"] = append_note(
            out.at[index, "rejected_values"], f"source_{field}={float(old_value):g}"
        )
    out.at[index, "data_quality_note"] = append_note(
        out.at[index, "data_quality_note"], message
    )
    out.at[index, "review_required"] = True


def _reconcile_gas_balance_v70(df: pd.DataFrame) -> pd.DataFrame:
    """Reconcile Total Gas, Formation Gas and N2 without producing negatives.

    Engineering identity:
        Total Gas = Formation Gas + Injected N2

    Rules are conservative:
    - tiny floating negatives become zero;
    - a negative Formation Gas from spreadsheet subtraction becomes zero when
      Total Gas does not exceed N2;
    - missing one member is derived only when the other two are valid;
    - conflicting populated values are retained for audit, then replaced by the
      physically consistent derived value and flagged for review.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    for column in (
        "gas_rate_mmscfd", "gas_formation_mmscfd", "n2_rate_mmscfd",
        "gas_balance_status", "gas_formation_derived", "n2_rate_derived",
        "total_gas_derived",
    ):
        if column not in out.columns:
            if column.endswith("_derived"):
                out[column] = False
            elif column == "gas_balance_status":
                out[column] = ""
            else:
                out[column] = np.nan

    total = pd.to_numeric(out["gas_rate_mmscfd"], errors="coerce")
    formation = pd.to_numeric(out["gas_formation_mmscfd"], errors="coerce")
    n2 = pd.to_numeric(out["n2_rate_mmscfd"], errors="coerce")

    for idx in out.index:
        t = total.loc[idx]
        f = formation.loc[idx]
        n = n2.loc[idx]
        status_parts: list[str] = []

        # Treat minute spreadsheet round-off as exact zero.
        if pd.notna(f) and -1e-6 <= f < 0:
            out.at[idx, "gas_formation_mmscfd"] = 0.0
            f = 0.0
            status_parts.append("tiny negative formation gas normalized to zero")

        # Physically impossible negative source formula.
        if pd.notna(f) and f < -1e-6:
            old = f
            if pd.notna(t) and pd.notna(n):
                corrected = max(float(t) - float(n), 0.0)
            else:
                corrected = 0.0
            out.at[idx, "gas_formation_mmscfd"] = corrected
            out.at[idx, "gas_formation_derived"] = True
            f = corrected
            _record_reconciled_source_v70(
                out, idx, "gas_formation_mmscfd", old,
                "Corrected negative Formation Gas using max(Total Gas - N2, 0)",
            )
            status_parts.append("negative formation gas corrected")

        # Derive one missing member from the other two.
        if pd.isna(f) and pd.notna(t) and pd.notna(n):
            f = max(float(t) - float(n), 0.0)
            out.at[idx, "gas_formation_mmscfd"] = f
            out.at[idx, "gas_formation_derived"] = True
            status_parts.append("formation gas derived")
        if pd.isna(n) and pd.notna(t) and pd.notna(f):
            n = max(float(t) - float(f), 0.0)
            out.at[idx, "n2_rate_mmscfd"] = n
            out.at[idx, "n2_rate_derived"] = True
            status_parts.append("N2 rate derived")
        if pd.isna(t) and pd.notna(f) and pd.notna(n):
            t = max(float(f) + float(n), 0.0)
            out.at[idx, "gas_rate_mmscfd"] = t
            out.at[idx, "total_gas_derived"] = True
            status_parts.append("total gas derived")

        # Re-read after derivation.
        t = pd.to_numeric(pd.Series([out.at[idx, "gas_rate_mmscfd"]]), errors="coerce").iloc[0]
        f = pd.to_numeric(pd.Series([out.at[idx, "gas_formation_mmscfd"]]), errors="coerce").iloc[0]
        n = pd.to_numeric(pd.Series([out.at[idx, "n2_rate_mmscfd"]]), errors="coerce").iloc[0]

        if pd.notna(t) and pd.notna(f) and pd.notna(n):
            expected_formation = max(float(t) - float(n), 0.0)
            tolerance = max(0.005, 0.03 * max(abs(float(t)), 0.1))
            imbalance = abs(float(t) - (float(f) + float(n)))
            if imbalance > tolerance:
                old = f
                out.at[idx, "gas_formation_mmscfd"] = expected_formation
                out.at[idx, "gas_formation_derived"] = True
                _record_reconciled_source_v70(
                    out, idx, "gas_formation_mmscfd", old,
                    "Gas balance conflict corrected so Total Gas = Formation Gas + N2",
                )
                status_parts.append("gas balance reconciled")
            else:
                status_parts.append("gas balance checked")

        out.at[idx, "gas_balance_status"] = "; ".join(dict.fromkeys(status_parts))

    return out


_postprocess_table_v70_base = _postprocess_table


def _postprocess_table(df: pd.DataFrame) -> pd.DataFrame:
    out = _postprocess_table_v70_base(df)
    if out is None or out.empty:
        return out
    out = _reconcile_gas_balance_v70(out)
    # OCR audit helper columns are metadata, never engineering curves.
    for column in list(out.columns):
        if str(column).startswith("ocr_raw__") or str(column).startswith("ocr_conf__") or str(column).startswith("ocr_status__"):
            BASE_NON_PLOT_COLS.add(str(column))
    return out


# Refresh re-export after the legacy v70 OCR override.
if hasattr(legacy, "parse_ctu_all_data_screen_image"):
    parse_ctu_all_data_screen_image = legacy.parse_ctu_all_data_screen_image
