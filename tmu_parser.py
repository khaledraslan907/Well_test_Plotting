from __future__ import annotations

"""Stable adaptive parser facade.

Compatibility capabilities remain available through ``tmu_parser_compat`` while a
clean adaptive engine independently interprets Excel/CSV tables.  The facade
scores both interpretations and selects the more credible result.  This avoids
regressions caused by repeatedly redefining the same parser functions in one
large file.
"""

import io
import json
import math
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

import tmu_parser_compat as compat
import smart_tabular_v75 as smart

PARSER_BUILD_ID = "v91-ocr-continuity-canonical-pressure-20260628"

COLUMN_LABELS: Dict[str, str] = dict(getattr(compat, "COLUMN_LABELS", {}))
COLUMN_LABELS.update({
    "pumping_pressure_psi": "Pumping Pressure (psi)",
    "whp_psi": "WHP (psi)",
    "ctu_circulation_pressure_psi": "Pumping Pressure (image OCR)",
    "ctu_wellhead_pressure_psi": "WHP (image OCR)",
})

# Public helper aliases used by the Streamlit app.
def canonical_key(value: object) -> str:
    text = smart.normalize(value).replace(" and ", " ")
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", text)).strip("_")


def standard_column_options(include_meta: bool = False) -> Dict[str, str]:
    labels = dict(COLUMN_LABELS)
    if include_meta:
        labels.update({"well": "Well Name", "date": "Date", "time": "Time", "datetime": "Date & Time", "note": "Note / Event"})
    return dict(sorted(labels.items(), key=lambda kv: kv[1].lower()))


def apply_user_column_mappings(df: pd.DataFrame, mappings: Optional[Mapping[str, str]] = None) -> pd.DataFrame:
    fn = getattr(compat, "apply_user_column_mappings", None)
    if fn is not None:
        return fn(df, mappings or {})
    if df is None or df.empty or not mappings:
        return df
    out = df.copy()
    for col in list(out.columns):
        target = mappings.get(str(col)) or mappings.get(canonical_key(col))
        if not target or target == "__keep__":
            continue
        if target == "__drop__":
            out.drop(columns=[col], inplace=True, errors="ignore")
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        if target in out:
            out[target] = pd.to_numeric(out[target], errors="coerce").combine_first(vals)
        else:
            out[target] = vals
        if col != target:
            out.drop(columns=[col], inplace=True, errors="ignore")
    return out


def ensure_pumping_pressure_column_v48(df: pd.DataFrame) -> pd.DataFrame:
    fn = getattr(compat, "ensure_pumping_pressure_column_v48", None)
    return fn(df) if fn is not None else df


def column_label(col: object) -> str:
    return COLUMN_LABELS.get(str(col), getattr(compat, "column_label")(col))


def _bytes_and_name(uploaded_file) -> tuple[bytes, str]:
    name = str(getattr(uploaded_file, "name", "uploaded_file"))
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    else:
        data = bytes(uploaded_file)
    return bytes(data), name


class UploadedBytes(io.BytesIO):
    def __init__(self, data: bytes, name: str):
        super().__init__(data)
        self.name = name


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    exclude = {
        "source", "sheet", "well", "date", "time", "time_text", "datetime",
        "note", "test_unit", "test_id", "test_sequence", "source_type",
        "link_status", "review_required", "message_index", "source_priority",
        "source_row", "source_group", "data_quality_note", "rejected_values",
        "parser_engine", "unmapped_columns", "parse_confidence",
        "gas_balance_status", "gas_rate_status", "ocr_status", "ocr_template",
        "image_file", "attachment_name", "source_member", "chat_sender",
        "caption_text", "suggested_well", "suggested_test_id",
        "suggested_link_reason", "test_start", "test_end",
        # Raw OCR aliases are retained in the review/audit table.  They are
        # merged into the canonical WHP/Pumping Pressure channels and should
        # not appear as duplicate sparse plotting signals.
        "ctu_circulation_pressure_psi", "ctu_wellhead_pressure_psi",
    }
    # Boolean/audit flags are not engineering curves.  pandas keeps bool dtype
    # after pd.to_numeric(), and subtracting bool min/max raises TypeError in
    # chart labelling.  Keep them in the audit table but never offer them as
    # plot features.
    exclude |= {
        "gas_formation_derived", "n2_rate_derived", "total_gas_derived",
        "ocr_approved", "is_event", "is_duplicate", "is_interpolated",
    }

    result = []
    seen = set()
    for c in df.columns:
        ctext = str(c)
        if (
            c in seen
            or c in exclude
            or ctext.startswith("source_")
            or ctext.startswith("_")
            or ctext.startswith("screen_")
            or ctext.startswith("ocr_raw__")
            or ctext.startswith("ocr_conf__")
            or ctext.startswith("ocr_status__")
            or ctext in {"ocr_fields_found", "ocr_confidence"}
        ):
            continue
        # Hide generic template calculation helpers from the plotting list.
        # Meaningful unfamiliar headers (for example raw_sand_probe_count) stay
        # available, while raw_calcul_7/raw_channel_4 do not slow the UI.
        if ctext.startswith("raw_") and re.match(
            r"raw_(?:channel(?:_|$)|calcul(?:_|$)|factor(?:_|$)|psia(?:_|$)|h2o(?:_|$)|deg_[cf](?:_|$)|in(?:_|$)|fb(?:_|$)|ftf(?:_|$)|fg(?:_|$)|fpv(?:_|$)|y2(?:_|$)|0(?:_|$))",
            ctext, flags=re.I,
        ):
            continue
        seen.add(c)
        positions = [i for i, name in enumerate(df.columns) if name == c]
        candidates = []
        for pos in positions:
            series = df.iloc[:, pos]
            if pd.api.types.is_bool_dtype(series.dtype):
                continue
            converted = pd.to_numeric(series, errors="coerce")
            # Force numeric extension/object results to real floats so Decimal,
            # nullable and mixed spreadsheet columns behave consistently.
            try:
                converted = converted.astype("float64")
            except (TypeError, ValueError):
                converted = converted.map(lambda v: float(v) if pd.notna(v) else np.nan)
            candidates.append(converted)
        if candidates and pd.concat(candidates, axis=1).notna().any(axis=1).sum() > 0:
            result.append(c)
    return result


def available_numeric_columns(df: pd.DataFrame) -> List[str]:
    return _numeric_columns(df)


def apply_fill_method(df: pd.DataFrame, columns: Sequence[str], method: str) -> pd.DataFrame:
    out = df.copy()
    selected = [c for c in columns if c in out.columns]
    if not selected or method in {None, "None", "No fill", "Do not fill"}:
        return out
    if method in {"Forward fill", "ffill"}:
        out[selected] = out[selected].ffill()
    elif method in {"Backward fill", "bfill"}:
        out[selected] = out[selected].bfill()
    elif method in {"Linear interpolation", "Linear interpolation by row", "interpolate"}:
        out[selected] = out[selected].apply(pd.to_numeric, errors="coerce").interpolate(limit_direction="both")
    elif method in {"Zero", "Fill zero"}:
        out[selected] = out[selected].fillna(0)
    return out



def normalize_ctu_ocr_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Merge CTU screen terminology into the standard engineering channels.

    The HMI label ``Circulation Pressure`` is the operation's pumping pressure,
    not a separate physical signal.  Likewise, CTU ``Wellhead Pressure`` is WHP.
    Raw OCR columns remain available for review, while charts use one continuous
    canonical signal across spreadsheets, messages and images.
    """
    if df is None or df.empty:
        return df
    out = df.copy()
    alias_pairs = [
        ("ctu_circulation_pressure_psi", "pumping_pressure_psi"),
        ("ctu_wellhead_pressure_psi", "whp_psi"),
    ]
    for raw_col, canonical_col in alias_pairs:
        if raw_col not in out.columns:
            continue
        raw = pd.to_numeric(out[raw_col], errors="coerce")
        current = pd.to_numeric(
            out.get(canonical_col, pd.Series(np.nan, index=out.index)),
            errors="coerce",
        )
        out[canonical_col] = current.combine_first(raw).astype("float64")
    if "source_type" in out.columns:
        ocr_mask = out["source_type"].astype(str).str.contains("ocr", case=False, na=False)
        if "ocr_approved" not in out.columns:
            out["ocr_approved"] = False
        out["ocr_approved"] = out["ocr_approved"].astype("boolean").fillna(False)
        out.loc[~ocr_mask, "ocr_approved"] = True
    return out


def flag_ocr_temporal_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Flag isolated OCR spikes without changing the displayed source value.

    A point is flagged only when both neighbouring OCR readings agree with each
    other and the middle value is far away.  Genuine operating steps therefore
    remain valid.  The user can still correct/approve flagged fields in the OCR
    review table.
    """
    if df is None or df.empty or "source_type" not in df.columns or "datetime" not in df.columns:
        return df
    out = df.copy()
    ocr_mask = out["source_type"].astype(str).str.contains("ocr", case=False, na=False)
    if int(ocr_mask.sum()) < 3:
        return out
    if "ocr_low_confidence_fields" not in out.columns:
        out["ocr_low_confidence_fields"] = ""
    if "data_quality_note" not in out.columns:
        out["data_quality_note"] = ""
    fields = {
        "ctu_weight_lbf": (5000.0, 0.45),
        "ctu_lt_weight_lbf": (500.0, 0.75),
        "ctu_wellhead_pressure_psi": (150.0, 0.75),
        "ctu_circulation_pressure_psi": (500.0, 0.70),
        "ctu_reel_depth_ft": (1500.0, 0.35),
        "ctu_reel_speed_ftmin": (50.0, 1.50),
        "ctu_fluid_rate_bpm": (2.0, 2.00),
        "ctu_n2_rate_scfm": (150.0, 1.00),
    }
    group_cols = [c for c in ("well", "test_id") if c in out.columns]
    group_items = list(out.loc[ocr_mask].groupby(group_cols, dropna=False, sort=False)) if group_cols else [(None, out.loc[ocr_mask])]
    for _, group in group_items:
        g = group.sort_values("datetime", kind="stable")
        for field, (abs_tol, rel_tol) in fields.items():
            if field not in g.columns:
                continue
            vals = pd.to_numeric(g[field], errors="coerce")
            valid_idx = list(vals.dropna().index)
            for pos in range(1, len(valid_idx) - 1):
                i0, i1, i2 = valid_idx[pos - 1], valid_idx[pos], valid_idx[pos + 1]
                prev_v, cur_v, next_v = float(vals.loc[i0]), float(vals.loc[i1]), float(vals.loc[i2])
                neighbour_scale = max(abs(prev_v), abs(next_v), 1.0)
                neighbours_agree = abs(prev_v - next_v) <= max(abs_tol, rel_tol * neighbour_scale)
                middle_far = min(abs(cur_v - prev_v), abs(cur_v - next_v)) > max(abs_tol * 1.5, rel_tol * 1.5 * neighbour_scale)
                if not (neighbours_agree and middle_far):
                    continue
                status_col = f"ocr_status__{field}"
                out.at[i1, status_col] = "temporal_outlier_review_required"
                label = column_label(field)
                old = str(out.at[i1, "ocr_low_confidence_fields"] or "").strip(" ;")
                if label not in old:
                    out.at[i1, "ocr_low_confidence_fields"] = f"{old}; {label}".strip(" ;")
                note = str(out.at[i1, "data_quality_note"] or "").strip(" ;")
                msg = f"OCR temporal outlier requires review: {label}"
                if msg not in note:
                    out.at[i1, "data_quality_note"] = f"{note}; {msg}".strip(" ;")
                if "review_required" in out.columns:
                    out.at[i1, "review_required"] = True
    # Cumulative totals should not materially decrease.
    for field in ("ctu_fluid_total_bbl", "ctu_n2_total_scf"):
        if field not in out.columns:
            continue
        for _, group in group_items:
            g = group.sort_values("datetime", kind="stable")
            vals = pd.to_numeric(g[field], errors="coerce")
            previous = None
            for idx, value in vals.items():
                if pd.isna(value):
                    continue
                value = float(value)
                if previous is not None and value < previous * 0.80:
                    out.at[idx, f"ocr_status__{field}"] = "counter_decrease_review_required"
                    label = column_label(field)
                    old = str(out.at[idx, "ocr_low_confidence_fields"] or "").strip(" ;")
                    if label not in old:
                        out.at[idx, "ocr_low_confidence_fields"] = f"{old}; {label}".strip(" ;")
                    if "review_required" in out.columns:
                        out.at[idx, "review_required"] = True
                previous = max(previous, value) if previous is not None else value
    return out

def _safe_postprocess(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    if "datetime" in out:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce").dt.round("s")
        out = out[out["datetime"].notna()].copy()
    if out.empty:
        return out
    if "source" not in out:
        out["source"] = "Unknown source"
    if "sheet" not in out:
        out["sheet"] = "Data"
    if "well" not in out:
        out["well"] = "Unknown"
    if "note" not in out:
        out["note"] = ""
    if "test_unit" not in out:
        out["test_unit"] = out["sheet"]
    if "source_type" not in out:
        out["source_type"] = "tabular"
    if "link_status" not in out:
        out["link_status"] = "source_confirmed"
    if "parser_engine" not in out:
        out["parser_engine"] = "v72-compatible"
    if "parse_confidence" not in out:
        out["parse_confidence"] = 0.85
    out["date"] = out["datetime"].dt.date
    out["time_text"] = out["datetime"].dt.strftime("%H:%M")
    out = normalize_ctu_ocr_signals(out)
    out = _engineering_checks(out)
    return out.sort_values(["well", "datetime", "source", "sheet"], kind="stable").reset_index(drop=True)


def _append(existing: object, addition: str) -> str:
    old = "" if existing is None or (isinstance(existing, float) and math.isnan(existing)) else str(existing).strip()
    if not old:
        return addition
    if addition in old:
        return old
    return old + "; " + addition


def _engineering_checks(df: pd.DataFrame) -> pd.DataFrame:
    """Non-destructive engineering checks with material tolerances.

    Source values are not changed merely to force an equation to balance.  A
    missing gas component may be derived, but all supplied components are kept.
    """
    out = df.copy()
    if "data_quality_note" not in out:
        out["data_quality_note"] = ""
    if "rejected_values" not in out:
        out["rejected_values"] = ""
    if "review_required" not in out:
        out["review_required"] = False
    out["review_required"] = out["review_required"].astype("boolean").fillna(False).astype(bool)

    for col in ("bsw_pct", "wlr_s_pct", "gvf_a_pct", "choke_pct"):
        if col not in out:
            continue
        vals = pd.to_numeric(out[col], errors="coerce")
        bad = vals.notna() & ((vals < 0) | (vals > 100))
        for i in out.index[bad]:
            out.at[i, "data_quality_note"] = _append(out.at[i, "data_quality_note"], f"{column_label(col)} is outside 0–100")
            out.at[i, "review_required"] = True

    # Production-rate balance: tolerance prevents normal rounding/noise from
    # creating hundreds of warnings.
    if all(c in out for c in ("oil_rate_stbd", "water_rate_bpd", "gross_rate_bpd")):
        oil = pd.to_numeric(out["oil_rate_stbd"], errors="coerce")
        wat = pd.to_numeric(out["water_rate_bpd"], errors="coerce")
        gross = pd.to_numeric(out["gross_rate_bpd"], errors="coerce")
        tol = np.maximum(5.0, 0.08 * gross.abs())
        bad = oil.notna() & wat.notna() & gross.notna() & ((gross - oil - wat).abs() > tol)
        for i in out.index[bad]:
            out.at[i, "data_quality_note"] = _append(out.at[i, "data_quality_note"], "Gross rate differs materially from oil + water")
            out.at[i, "review_required"] = True

    for col in ("gas_rate_mmscfd", "gas_formation_mmscfd", "n2_rate_mmscfd"):
        if col not in out:
            out[col] = np.nan
        else:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "gas_formation_derived" not in out:
        out["gas_formation_derived"] = False
    if "n2_rate_derived" not in out:
        out["n2_rate_derived"] = False
    if "total_gas_derived" not in out:
        out["total_gas_derived"] = False
    if "gas_balance_status" not in out:
        out["gas_balance_status"] = ""

    total = out["gas_rate_mmscfd"]
    form = out["gas_formation_mmscfd"]
    n2 = out["n2_rate_mmscfd"]
    missing_form = total.notna() & n2.notna() & form.isna()
    out.loc[missing_form, "gas_formation_mmscfd"] = (total - n2).clip(lower=0)
    out.loc[missing_form, "gas_formation_derived"] = True
    missing_n2 = total.notna() & form.notna() & n2.isna()
    out.loc[missing_n2, "n2_rate_mmscfd"] = (total - form).clip(lower=0)
    out.loc[missing_n2, "n2_rate_derived"] = True
    missing_total = total.isna() & form.notna() & n2.notna()
    out.loc[missing_total, "gas_rate_mmscfd"] = form + n2
    out.loc[missing_total, "total_gas_derived"] = True

    total = pd.to_numeric(out["gas_rate_mmscfd"], errors="coerce")
    form = pd.to_numeric(out["gas_formation_mmscfd"], errors="coerce")
    n2 = pd.to_numeric(out["n2_rate_mmscfd"], errors="coerce")
    supplied = total.notna() & form.notna() & n2.notna()
    delta = (total - form - n2).abs()
    tol = np.maximum(0.005, 0.05 * total.abs())
    conflict = supplied & (delta > tol)
    for i in out.index[conflict]:
        out.at[i, "data_quality_note"] = _append(out.at[i, "data_quality_note"], "Gas components do not materially balance: Total Gas ≠ Formation Gas + N₂")
        out.at[i, "review_required"] = True
        out.at[i, "gas_balance_status"] = "source conflict — values preserved"
    balanced = supplied & ~conflict
    out.loc[balanced, "gas_balance_status"] = "source values balanced"

    negative_form = form.notna() & (form < 0)
    for i in out.index[negative_form]:
        out.at[i, "data_quality_note"] = _append(out.at[i, "data_quality_note"], "Formation Gas is negative")
        out.at[i, "review_required"] = True

    return out


def _group_key(df: pd.DataFrame) -> tuple:
    sheet = str(df["sheet"].iloc[0]) if "sheet" in df and len(df) else "Data"
    well = str(df["well"].iloc[0]) if "well" in df and len(df) else "Unknown"
    start = pd.to_datetime(df.get("datetime"), errors="coerce").min() if "datetime" in df else pd.NaT
    end = pd.to_datetime(df.get("datetime"), errors="coerce").max() if "datetime" in df else pd.NaT
    return sheet, well, start, end


def _choose_ensemble(compat_tables: List[pd.DataFrame], smart_tables: List[pd.DataFrame]) -> List[pd.DataFrame]:
    all_tables = [("compat", _safe_postprocess(t)) for t in compat_tables] + [("smart", _safe_postprocess(t)) for t in smart_tables]
    all_tables = [(engine, t) for engine, t in all_tables if t is not None and not t.empty]
    if not all_tables:
        return []
    # Compare interpretations that cover the same sheet/time span; keep distinct
    # tables from different sheets.
    groups: List[List[tuple[str, pd.DataFrame]]] = []
    for item in all_tables:
        _, table = item
        s, w, start, end = _group_key(table)
        placed = False
        for group in groups:
            _, g = group[0]
            gs, gw, gstart, gend = _group_key(g)
            overlap = pd.notna(start) and pd.notna(end) and pd.notna(gstart) and pd.notna(gend) and max(start, gstart) <= min(end, gend)
            same_shape = abs(len(table) - len(g)) <= max(2, int(0.05 * max(len(table), len(g))))
            if s == gs and (w == gw or "Unknown" in {w, gw}) and (overlap or same_shape):
                group.append(item); placed = True; break
        if not placed:
            groups.append([item])
    selected = []
    for group in groups:
        # A slightly cleaner interpretation must not win by silently truncating
        # the end of a long field test. Score quality and coverage together.
        max_rows = max(len(table) for _, table in group)
        spans = []
        for _, table in group:
            dt = pd.to_datetime(table.get("datetime"), errors="coerce") if "datetime" in table else pd.Series(dtype="datetime64[ns]")
            span_h = 0.0
            if not dt.empty and dt.notna().any():
                span_h = max(0.0, float((dt.max() - dt.min()).total_seconds() / 3600.0))
            spans.append(span_h)
        max_span = max(spans or [0.0])

        ranked = []
        for (engine, table), span_h in zip(group, spans):
            quality = float(smart.interpretation_score(table))
            row_coverage = len(table) / max(max_rows, 1)
            span_coverage = span_h / max(max_span, 1.0) if max_span > 0 else 1.0
            # Coverage bonuses are modest: they break close quality ties, but
            # cannot rescue a genuinely poor interpretation.
            ensemble_score = quality + 10.0 * row_coverage + 4.0 * span_coverage
            ranked.append((ensemble_score, quality, len(table), span_h, engine, table))
        ranked.sort(key=lambda item: (item[0], item[1], item[2], item[3]), reverse=True)
        selected.append(ranked[0][-1])
    return selected


def _xlsx_is_pathologically_wide(data: bytes) -> bool:
    """Cheaply detect Excel used-range/formula spillover without opening pandas."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            worksheet_infos = [i for i in zf.infolist() if i.filename.startswith("xl/worksheets/sheet") and i.filename.endswith(".xml")]
            if any(i.file_size >= 1_500_000 for i in worksheet_infos):
                return True
            for info in worksheet_infos:
                head = zf.read(info.filename)[:4096].decode("utf-8", errors="ignore")
                m = re.search(r'<dimension[^>]+ref="[A-Z]+\d+:([A-Z]+)\d+"', head)
                if m:
                    col = 0
                    for ch in m.group(1):
                        col = col * 26 + (ord(ch.upper()) - 64)
                    if col > 512:
                        return True
    except Exception:
        pass
    return False


def _xlsx_last_declared_row(data: bytes) -> int:
    """Return the largest worksheet row declared in the XLSX XML."""
    last_row = 0
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for info in zf.infolist():
                if not (info.filename.startswith("xl/worksheets/sheet") and info.filename.endswith(".xml")):
                    continue
                xml = zf.read(info.filename)
                for match in re.finditer(rb"<row\s[^>]*\br=\"(\d+)\"", xml):
                    last_row = max(last_row, int(match.group(1)))
    except Exception:
        return 0
    return last_row


def _smart_tables_reach_workbook_tail(tables: List[pd.DataFrame], data: bytes) -> bool:
    """Guard the fast path against a credible but prematurely truncated table."""
    last_declared = _xlsx_last_declared_row(data)
    if last_declared <= 0 or not tables:
        return True
    parsed_rows = []
    for table in tables:
        if "source_row" in table.columns:
            values = pd.to_numeric(table["source_row"], errors="coerce")
            if values.notna().any():
                parsed_rows.append(int(values.max()))
    if not parsed_rows:
        return True
    parsed_last = max(parsed_rows)
    # Allow headers, averages and a short footer. A larger unparsed tail is a
    # strong sign that another test block exists below the first one.
    allowance = max(8, int(0.08 * last_declared))
    return parsed_last >= last_declared - allowance


def _credible_smart_tables(tables: List[pd.DataFrame]) -> bool:
    if not tables:
        return False
    for table in tables:
        if table is None or table.empty or smart.interpretation_score(table) < 70.0:
            return False
        known = [c for c in table.columns if c in smart.FIELD_LABELS and pd.to_numeric(table[c], errors="coerce").notna().sum() >= 2]
        if len(known) < 2:
            return False
    return True




_WHATSAPP_CHAT_BASENAMES = {
    "_chat.txt", "chat.txt", "whatsapp chat.txt", "whatsapp_chat.txt",
}


def _decode_text_payload(data: bytes) -> str:
    """Decode WhatsApp/text exports without silently dropping the whole file."""
    if not data:
        return ""
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "utf-16-le", "utf-16-be", "cp1256", "latin-1"):
        try:
            text = data.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
        # Reject an obviously wrong UTF-16 decode of an ASCII/UTF-8 file.
        if text and text.count("\x00") <= max(2, len(text) // 100):
            return text.replace("\ufeff", "")
    return data.decode("utf-8", errors="replace").replace("\ufeff", "")


def _looks_like_whatsapp_export_text(text: str, member_name: str = "") -> bool:
    base = Path(member_name).name.lower().strip()
    if base in _WHATSAPP_CHAT_BASENAMES or "whatsapp chat" in base:
        return True
    sample = (text or "")[:12000]
    # Android bracket format: [12/06/2026, 09:05:04] Sender: body
    if re.search(r"(?m)^\s*[\u200e\u200f]?\[\d{1,2}[/-]\d{1,2}[/-]\d{2,4},\s*\d{1,2}:\d{2}", sample):
        return True
    # iOS/plain format: 12/06/2026, 09:05 - Sender: body
    if re.search(r"(?m)^\s*[\u200e\u200f]?\d{1,2}[/-]\d{1,2}[/-]\d{2,4},\s*\d{1,2}:\d{2}.*?\s-\s", sample):
        return True
    return False


def _parse_whatsapp_text_payload(data: bytes, source_name: str, member_name: str) -> pd.DataFrame:
    """Parse a WhatsApp export directly, independent of the generic TXT loader.

    ZIP parsing previously sent ``_chat.txt`` through the recursive generic file
    loader and swallowed any exception.  That made text-only WhatsApp exports
    appear empty even though the chat contained valid timestamped production-test reports.  This
    routine deliberately tries the mature export parser and the robust block
    parser, then keeps the strongest non-empty interpretation.
    """
    text = _decode_text_payload(data)
    if not text.strip() or not _looks_like_whatsapp_export_text(text, member_name):
        return pd.DataFrame()

    candidates: List[pd.DataFrame] = []
    parsers = [
        getattr(compat, "parse_whatsapp_plain_or_export_text", None),
        getattr(compat, "parse_many_tmu_messages", None),
    ]
    legacy = getattr(compat, "legacy", None)
    if legacy is not None:
        parsers.extend([
            getattr(legacy, "parse_whatsapp_export_text", None),
            getattr(legacy, "parse_whatsapp_plain_or_export_text", None),
        ])

    for parser in parsers:
        if parser is None:
            continue
        try:
            frame = parser(text, source_name=source_name)
        except TypeError:
            try:
                frame = parser(text, source_name)
            except Exception:
                continue
        except Exception:
            continue
        if frame is None or frame.empty:
            continue
        frame = _safe_postprocess(frame)
        if frame.empty:
            continue
        frame["attachment_name"] = Path(member_name).name
        frame["source_member"] = member_name
        frame["source_type"] = frame.get("source_type", "whatsapp_export_text")
        candidates.append(frame)

    if not candidates:
        return pd.DataFrame()

    # Prefer the interpretation with the most unique, timestamped engineering
    # readings.  Duplicate quoted/edited messages should not win merely because
    # they inflate row count.
    ranked = []
    for frame in candidates:
        candidate_dedupe_cols = [
            c for c in ("well", "datetime", "gross_rate_bpd", "oil_rate_stbd",
                        "water_rate_bpd", "whp_psi", "pumping_pressure_psi")
            if c in frame.columns
        ]
        unique_rows = (
            frame.drop_duplicates(subset=candidate_dedupe_cols, keep="last")
            if candidate_dedupe_cols else frame
        )
        numeric_fields = sum(
            pd.to_numeric(unique_rows[c], errors="coerce").notna().sum()
            for c in _numeric_columns(unique_rows)
        )
        ranked.append((len(unique_rows), numeric_fields, frame))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    result = ranked[0][2].copy()
    dedupe_cols = [c for c in ("well", "datetime", "gross_rate_bpd", "oil_rate_stbd", "water_rate_bpd", "whp_psi", "pumping_pressure_psi") if c in result.columns]
    if dedupe_cols:
        result = result.drop_duplicates(subset=dedupe_cols, keep="last")
    return assign_test_ids(result.reset_index(drop=True))


def _zip_member_is_chat_text(member_name: str, payload: bytes) -> bool:
    if Path(member_name).suffix.lower() != ".txt":
        return False
    return _looks_like_whatsapp_export_text(_decode_text_payload(payload), member_name)


def load_tabular_file(uploaded_file, parse_images: bool = True, max_ocr_images: int = 1000) -> List[pd.DataFrame]:
    data, name = _bytes_and_name(uploaded_file)
    suffix = Path(name).suffix.lower()

    # Handle WhatsApp ZIP exports explicitly.  A valid export may contain only
    # ``_chat.txt`` and an unsupported audio file, so the chat must be parsed
    # directly rather than delegated to the generic recursive TXT path.
    if suffix == ".zip":
        tables: List[pd.DataFrame] = []
        diagnostics: List[str] = []
        chat_members_found = 0
        chat_members_parsed = 0
        image_count = 0
        try:
            zf = zipfile.ZipFile(io.BytesIO(data))
        except (zipfile.BadZipFile, OSError) as exc:
            raise RuntimeError(f"The uploaded ZIP archive is invalid or damaged: {exc}") from exc

        with zf:
            members = [
                m for m in zf.namelist()
                if not m.endswith("/") and not m.startswith("__MACOSX/")
            ]

            # Pass 1: parse exported chat files first.  This guarantees that a
            # text-only WhatsApp export still produces production-test rows.
            chat_member_names = set()
            for member in members:
                member_name = Path(member).name
                if Path(member_name).suffix.lower() != ".txt":
                    continue
                try:
                    payload = zf.read(member)
                except Exception as exc:
                    diagnostics.append(f"{member_name}: could not be read ({exc})")
                    continue
                if not _zip_member_is_chat_text(member_name, payload):
                    continue
                chat_members_found += 1
                chat_member_names.add(member)
                try:
                    frame = _parse_whatsapp_text_payload(
                        payload,
                        source_name=f"{name}:{member_name}",
                        member_name=member,
                    )
                except Exception as exc:
                    diagnostics.append(f"{member_name}: WhatsApp text parser failed ({exc})")
                    continue
                if frame is not None and not frame.empty:
                    chat_members_parsed += 1
                    tables.append(frame)
                else:
                    diagnostics.append(f"{member_name}: no complete timestamped production-test readings detected")

            # Pass 2: parse tabular/PDF/image attachments. Audio, video and
            # stickers are intentionally ignored and do not make the ZIP fail.
            # The mature compatibility image path is retained because it has
            # already been validated on full WhatsApp media exports.
            supported = {".xlsx", ".xls", ".xlsm", ".csv", ".tsv", ".txt", ".docx", ".pdf", ".jpg", ".jpeg", ".png", ".webp"}
            image_exts = {".jpg", ".jpeg", ".png", ".webp"}
            for member in members:
                if member in chat_member_names:
                    continue
                member_name = Path(member).name
                ext = Path(member_name).suffix.lower()
                if ext not in supported:
                    continue
                if ext in image_exts:
                    if not parse_images:
                        continue
                    image_count += 1
                    if max_ocr_images > 0 and image_count > int(max_ocr_images):
                        continue
                try:
                    payload = zf.read(member)
                    sub = load_tabular_file(
                        UploadedBytes(payload, member_name),
                        parse_images=parse_images,
                        max_ocr_images=max_ocr_images,
                    )
                    for table in sub:
                        if table is None or table.empty:
                            continue
                        table = table.copy()
                        table["attachment_name"] = member_name
                        table["source_member"] = member
                        tables.append(table)
                except Exception as exc:
                    diagnostics.append(f"{member_name}: {exc}")

        if tables:
            merged = pd.concat(tables, ignore_index=True, sort=False)
            merged = _safe_postprocess(merged)
            merged = assign_test_ids(merged)
            merged = flag_ocr_temporal_outliers(merged)
            merged = normalize_ctu_ocr_signals(merged)
            return [merged]

        detail = ""
        if chat_members_found:
            detail = f" Found {chat_members_found} WhatsApp chat text file(s), but none produced complete timestamped readings."
        elif members:
            detail = " The archive did not contain a recognizable _chat.txt/WhatsApp chat export or a supported data attachment."
        if diagnostics:
            detail += " Details: " + " | ".join(diagnostics[:5])
        raise RuntimeError("No usable production-test data was found in the ZIP archive." + detail)

    compat_tables: List[pd.DataFrame] = []
    smart_tables: List[pd.DataFrame] = []
    compat_error = None
    smart_error = None

    # Wide Excel workbooks often contain cached formulas to column XFD. The
    # adaptive XML reader trims that spillover and is dramatically faster, so
    # use it first. Normal workbooks continue through the mature compatibility
    # parser. Only run the second engine when the first interpretation is weak.
    smart_first = suffix in {".xlsx", ".xlsm"} and _xlsx_is_pathologically_wide(data)
    if smart_first:
        try:
            smart_tables = smart.parse_file(data, name)
        except Exception as exc:
            smart_error = exc
        if _credible_smart_tables(smart_tables) and _smart_tables_reach_workbook_tail(smart_tables, data):
            return [assign_test_ids(_safe_postprocess(t)) for t in smart_tables]

    try:
        compat_tables = compat.load_tabular_file(UploadedBytes(data, name), parse_images=parse_images, max_ocr_images=max_ocr_images)
    except Exception as exc:
        compat_error = exc

    if suffix in {".xlsx", ".xlsm", ".csv", ".tsv"}:
        compat_ready = [_safe_postprocess(t) for t in compat_tables if t is not None and not t.empty]
        # Engineering warnings are not parser-confidence failures. A legitimate
        # field test may contain balance conflicts, so do not rescan the entire
        # workbook merely because review_required is true.
        high_confidence = bool(compat_ready) and all(
            smart.interpretation_score(t) >= 70.0
            and len([c for c in t.columns if c in COLUMN_LABELS and pd.to_numeric(t[c], errors="coerce").notna().sum() >= 2]) >= 2
            for t in compat_ready
        )
        if high_confidence:
            return [assign_test_ids(t) for t in compat_ready]
        if not smart_tables:
            try:
                smart_tables = smart.parse_file(data, name)
            except Exception as exc:
                smart_error = exc

    selected = _choose_ensemble(compat_tables, smart_tables)
    if selected:
        return [assign_test_ids(_safe_postprocess(t)) for t in selected]

    detail = "; ".join(x for x in [f"compatible engine: {compat_error}" if compat_error else "", f"adaptive engine: {smart_error}" if smart_error else ""] if x)
    raise RuntimeError(
        "No usable time-series table was detected. The parser searched multiple header rows, "
        "date/time layouts, units and numeric channels, but could not identify at least two "
        f"timestamped readings. {detail}".strip()
    )


# Internal compatibility hooks retained for existing regression scripts.
canonical_header = getattr(compat, "canonical_header", smart.infer_field)
_gas_rate_unit = getattr(compat, "_gas_rate_unit", smart._gas_unit)
_reconcile_gas_balance_v70 = getattr(compat, "_reconcile_gas_balance_v70", _engineering_checks)

# Text, WhatsApp, OCR and specialist PDF parsers remain mature compatibility
# adapters. Their outputs still pass through the same non-destructive checks.
def parse_many_tmu_messages(text: str, source_name: str = "Pasted WhatsApp") -> pd.DataFrame:
    return assign_test_ids(_safe_postprocess(compat.parse_many_tmu_messages(text, source_name=source_name)))


def parse_whatsapp_plain_or_export_text(text: str, source_name: str = "WhatsApp") -> pd.DataFrame:
    fn = getattr(compat, "parse_whatsapp_plain_or_export_text", compat.parse_many_tmu_messages)
    return assign_test_ids(_safe_postprocess(fn(text, source_name=source_name)))


def _is_srp_trend_group(group: pd.DataFrame) -> bool:
    """Return True for sparse SRP surveillance/trend datasets."""
    srp_cols = {
        "stroke_length_in", "stroke_rate_spm", "spm", "peak_load_lbf",
        "min_load_lbf", "minimum_load_lbf", "polished_rod_load_lbf",
    }
    production_cols = {
        "gross_rate_bpd", "oil_rate_stbd", "water_rate_bpd",
        "gas_rate_mmscfd", "gas_formation_mmscfd", "sep_p_psi",
        "pumping_pressure_psi", "n2_rate_mmscfd",
    }
    srp_hits = sum(
        pd.to_numeric(group[c], errors="coerce").notna().sum()
        for c in srp_cols if c in group.columns
    )
    production_hits = sum(
        pd.to_numeric(group[c], errors="coerce").notna().sum()
        for c in production_cols if c in group.columns
    )
    return srp_hits >= max(3, len(group)) and production_hits == 0


def assign_test_ids(
    df: pd.DataFrame,
    gap_hours: float = 12.0,
    *,
    preserve_existing: bool = False,
    group_unknown_by_source: bool = True,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "well" not in out:
        out["well"] = "Unknown"
    if "datetime" not in out:
        out["datetime"] = pd.NaT
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    source_key = out.get("source", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
    sheet_key = out.get("sheet", pd.Series([""] * len(out), index=out.index)).fillna("").astype(str)
    well_key = out["well"].fillna("Unknown").astype(str).str.strip().replace("", "Unknown")
    if group_unknown_by_source:
        # Parser-level default: unrelated files with no well context remain
        # separated so an arbitrary ZIP cannot merge independent images.
        out["__segment_key"] = np.where(
            well_key.str.casefold().eq("unknown"),
            "Unknown|" + source_key + "|" + sheet_key,
            well_key,
        )
    else:
        # Explicit UI custom-gap override: all unknown readings uploaded in the
        # current analysis are allowed to share the same time-based segment.
        # This lets two OCR snapshots within the selected gap form one line.
        out["__segment_key"] = well_key
    out = out.sort_values(["__segment_key", "datetime", "source"], na_position="last", kind="stable").reset_index(drop=True)

    ids = pd.Series(index=out.index, dtype="object")
    sequences = pd.Series(index=out.index, dtype="Int64")
    for _, idxs in out.groupby("__segment_key", sort=False, dropna=False).groups.items():
        idxs = list(idxs)
        group = out.loc[idxs]
        effective_gap = max(float(gap_hours), 24.0 * 14.0) if _is_srp_trend_group(group) else float(gap_hours)
        last = pd.NaT
        seq = 0
        current = ""
        for i in idxs:
            row = out.loc[i]
            dt = row.get("datetime")
            existing = str(row.get("test_id") or "").strip()
            if preserve_existing and existing and existing.lower() not in {"nan", "none"}:
                current = existing
                try:
                    seq = int(row.get("test_sequence")) if pd.notna(row.get("test_sequence")) else max(seq, 1)
                except Exception:
                    seq = max(seq, 1)
                ids.at[i] = current
                sequences.at[i] = seq
                if pd.notna(dt):
                    last = pd.Timestamp(dt)
                continue

            new = (
                not current
                or pd.isna(dt)
                or pd.isna(last)
                or pd.Timestamp(dt) - pd.Timestamp(last) > pd.Timedelta(hours=effective_gap)
                or pd.Timestamp(dt) < pd.Timestamp(last)
            )
            if new:
                seq += 1
                display_well = str(row.get("well") or "Unknown").strip() or "Unknown"
                current = (
                    f"{display_well}_{pd.Timestamp(dt).strftime('%Y%m%d_%H%M')}"
                    if pd.notna(dt)
                    else f"{display_well}_T{seq:02d}"
                )
            ids.at[i] = current
            sequences.at[i] = seq
            if pd.notna(dt):
                last = pd.Timestamp(dt)

    out["test_id"] = ids
    out["test_sequence"] = sequences.astype("Int64")
    return out.drop(columns=["__segment_key"], errors="ignore")


_OCR_LINK_TEXT_COLUMNS = (
    "well", "test_id", "source_type", "link_status",
    "suggested_well", "suggested_test_id", "suggested_link_reason",
)


def _ensure_assignable_ocr_link_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare mixed OCR-link metadata for pandas 3 / Arrow-backed frames.

    pandas 3 no longer silently upcasts a numeric or Arrow column when a string
    is assigned. ZIP ingestion can create all-null suggestion columns that are
    inferred as float/Arrow dtypes, so they must be made explicitly object-like
    before row-level context linking.
    """
    out = df.copy()
    for col in _OCR_LINK_TEXT_COLUMNS:
        if col not in out.columns:
            out[col] = pd.Series([None] * len(out), index=out.index, dtype="object")
        else:
            out[col] = out[col].astype("object")
    if "test_sequence" not in out.columns:
        out["test_sequence"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    else:
        out["test_sequence"] = pd.to_numeric(out["test_sequence"], errors="coerce").astype("Int64")
    return out


def _plain_context_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def auto_link_ocr_rows_by_time_context(df: pd.DataFrame, max_gap_hours: float = 3.0) -> pd.DataFrame:
    """Safely inherit well/test context for timestamped OCR rows.

    A link is made only when all nearby non-OCR readings belong to one unique
    test ID. OCR measurements remain review-required; only context is inherited.
    Text/status columns are explicitly object dtype so pandas 3 cannot fail when
    ZIP OCR rows are linked to a spreadsheet or WhatsApp text context.
    """
    if df is None or df.empty or "datetime" not in df.columns or "source_type" not in df.columns:
        return df
    out = _ensure_assignable_ocr_link_columns(df)
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    ocr_mask = out["source_type"].astype(str).str.contains("ocr", case=False, na=False)
    if not ocr_mask.any():
        return out
    well_text = out["well"].fillna("Unknown").astype(str).str.strip()
    test_text = out["test_id"].fillna("").astype(str).str.strip()
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
        deltas = (anchors["datetime"] - out.at[idx, "datetime"]).abs()
        nearby = anchors.loc[deltas <= pd.Timedelta(hours=float(max_gap_hours))]
        if nearby.empty or nearby["test_id"].astype(str).nunique() != 1:
            continue
        nearest_idx = deltas.loc[nearby.index].idxmin()
        anchor = anchors.loc[nearest_idx]
        anchor_well = _plain_context_text(anchor.get("well", ""))
        anchor_test_id = _plain_context_text(anchor.get("test_id", ""))
        if anchor_well:
            out.at[idx, "well"] = anchor_well
            out.at[idx, "suggested_well"] = anchor_well
        if anchor_test_id:
            out.at[idx, "test_id"] = anchor_test_id
            out.at[idx, "suggested_test_id"] = anchor_test_id
        seq = pd.to_numeric(pd.Series([anchor.get("test_sequence", pd.NA)]), errors="coerce").iloc[0]
        if pd.notna(seq):
            out.at[idx, "test_sequence"] = int(seq)
        out.at[idx, "link_status"] = "ocr_auto_linked_by_timestamp"
        gap_minutes = float(deltas.loc[nearest_idx].total_seconds() / 60.0)
        out.at[idx, "suggested_link_reason"] = f"Unique nearby test reading ({gap_minutes:.1f} min)"
    return out


def merge_duplicate_test_rows_v53(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "datetime" not in out:
        return out
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    keys = [c for c in ("well", "datetime", "test_id") if c in out]
    if not keys:
        return out
    rows = []
    for _, group in out.groupby(keys, dropna=False, sort=False):
        row = group.iloc[0].copy()
        for c in out.columns:
            values = group[c].dropna()
            if c in {"note", "data_quality_note", "rejected_values"}:
                unique = [str(v).strip() for v in values if str(v).strip()]
                row[c] = "; ".join(dict.fromkeys(unique))
            elif len(values):
                row[c] = values.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows).reset_index(drop=True)
