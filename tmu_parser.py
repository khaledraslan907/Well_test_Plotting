from __future__ import annotations

"""Stable v75 parser facade.

Historical capabilities remain available through ``tmu_parser_compat`` while a
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
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import tmu_parser_compat as compat
import smart_tabular_v75 as smart

PARSER_BUILD_ID = "v75-fast-continuous-smart-parser-20260627"

COLUMN_LABELS: Dict[str, str] = dict(getattr(compat, "COLUMN_LABELS", {}))
COLUMN_LABELS.update(smart.FIELD_LABELS)

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
        if c in seen or c in exclude or ctext.startswith("source_") or ctext.startswith("_"):
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
        ranked = sorted(group, key=lambda x: smart.interpretation_score(x[1]), reverse=True)
        selected.append(ranked[0][1])
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


def load_tabular_file(uploaded_file, parse_images: bool = True, max_ocr_images: int = 1000) -> List[pd.DataFrame]:
    data, name = _bytes_and_name(uploaded_file)
    suffix = Path(name).suffix.lower()

    # ZIP is recursively handled by this facade so every spreadsheet attachment
    # benefits from the adaptive engine.
    if suffix == ".zip":
        tables: List[pd.DataFrame] = []
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            for member in zf.namelist():
                if member.endswith("/") or member.startswith("__MACOSX/"):
                    continue
                member_name = Path(member).name
                ext = Path(member_name).suffix.lower()
                if ext not in {".xlsx", ".xls", ".xlsm", ".csv", ".tsv", ".txt", ".docx", ".pdf", ".jpg", ".jpeg", ".png", ".webp"}:
                    continue
                if ext in {".jpg", ".jpeg", ".png", ".webp"} and not parse_images:
                    continue
                try:
                    sub = load_tabular_file(UploadedBytes(zf.read(member), member_name), parse_images=parse_images, max_ocr_images=max_ocr_images)
                    for t in sub:
                        t = t.copy(); t["attachment_name"] = member_name; t["source_member"] = member
                        tables.append(t)
                except Exception:
                    continue
        if tables:
            merged = pd.concat(tables, ignore_index=True, sort=False)
            merged = assign_test_ids(merged)
            return [merged]
        raise RuntimeError("No usable production-test data was found in the ZIP archive.")

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
        if _credible_smart_tables(smart_tables):
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


def assign_test_ids(df: pd.DataFrame, gap_hours: float = 12.0) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    if "well" not in out:
        out["well"] = "Unknown"
    if "datetime" not in out:
        out["datetime"] = pd.NaT
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.sort_values(["well", "datetime", "source"], na_position="last", kind="stable").reset_index(drop=True)
    ids = []
    sequence = []
    state: Dict[str, tuple[pd.Timestamp, int, str]] = {}
    for _, row in out.iterrows():
        well = str(row.get("well") or "Unknown")
        dt = row.get("datetime")
        last, seq, current = state.get(well, (pd.NaT, 0, ""))
        new = not current or pd.isna(dt) or pd.isna(last) or dt - last > pd.Timedelta(hours=gap_hours) or dt < last
        if new:
            seq += 1
            current = f"{well}_{pd.Timestamp(dt).strftime('%Y%m%d_%H%M')}" if pd.notna(dt) else f"{well}_T{seq:02d}"
        ids.append(current); sequence.append(seq)
        if pd.notna(dt):
            last = dt
        state[well] = (last, seq, current)
    out["test_id"] = ids
    out["test_sequence"] = sequence
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
