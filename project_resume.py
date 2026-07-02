from __future__ import annotations

"""Resumable production-test PDF projects.

New dashboard PDFs carry a compressed project snapshot as an embedded attachment.
Older dashboard PDFs can be recovered from their visible vector text, value labels,
and event markers on a best-effort basis.
"""

import io
import json
import math
import re
import zipfile
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd

PROJECT_ATTACHMENT_NAME = "production_test_project_v1.zip"
PROJECT_SCHEMA_VERSION = 1

FEATURE_LABEL_TO_KEY = {
    "total gas rate (mmscf/d)": "gas_rate_mmscfd",
    "gas rate (mmscf/d)": "gas_rate_mmscfd",
    "gross rate (bbl/d)": "gross_rate_bpd",
    "oil rate (stb/d)": "oil_rate_stbd",
    "water rate (bbl/d)": "water_rate_bpd",
    "bs&w (%)": "bsw_pct",
    "water cut (%)": "bsw_pct",
    "whp (psi)": "whp_psi",
    "separator pressure (psi)": "sep_p_psi",
    "pumping pressure (psi)": "pumping_pressure_psi",
    "ctu circulation pressure (psi)": "pumping_pressure_psi",
    "choke opening (%)": "choke_pct",
    "choke size (/64 in)": "choke_size_64",
    "salinity (k ppm nacl)": "salinity_kppm",
}


def _normalize_label(value: object) -> str:
    text = "" if value is None else str(value)
    text = text.replace("\u2082", "2").replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text.strip().lower())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime, date, time)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _dataframe_schema(df: pd.DataFrame) -> Dict[str, Any]:
    datetime_columns = []
    numeric_columns = []
    boolean_columns = []
    for col in df.columns:
        series = df[col]
        key = str(col)
        if pd.api.types.is_datetime64_any_dtype(series.dtype) or key in {"datetime", "date", "test_start", "test_end"}:
            datetime_columns.append(key)
        elif pd.api.types.is_bool_dtype(series.dtype):
            boolean_columns.append(key)
        elif pd.api.types.is_numeric_dtype(series.dtype):
            numeric_columns.append(key)
    return {
        "datetime_columns": datetime_columns,
        "numeric_columns": numeric_columns,
        "boolean_columns": boolean_columns,
    }


def build_project_bundle(dataframe: pd.DataFrame, state: Dict[str, Any]) -> bytes:
    """Return a compressed project package suitable for embedding in a PDF."""
    df = dataframe.copy() if dataframe is not None else pd.DataFrame()
    data_json = df.to_json(
        orient="split",
        date_format="iso",
        date_unit="ms",
        default_handler=str,
        force_ascii=False,
    ).encode("utf-8")
    safe_state = _json_safe(dict(state or {}))
    safe_state.setdefault("schema_version", PROJECT_SCHEMA_VERSION)
    safe_state["data_schema"] = _dataframe_schema(df)
    state_json = json.dumps(safe_state, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=7) as zf:
        zf.writestr("state.json", state_json)
        zf.writestr("data.json", data_json)
    return output.getvalue()


def _restore_dataframe(data_bytes: bytes, state: Dict[str, Any]) -> pd.DataFrame:
    payload = json.loads(data_bytes.decode("utf-8"))
    columns = payload.get("columns", [])
    values = payload.get("data", [])
    index = payload.get("index")
    df = pd.DataFrame(values, columns=columns)
    if isinstance(index, list) and len(index) == len(df):
        df.index = index

    schema = state.get("data_schema", {}) if isinstance(state, dict) else {}
    for col in schema.get("datetime_columns", []):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in schema.get("numeric_columns", []):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in schema.get("boolean_columns", []):
        if col in df.columns:
            text = df[col].astype(str).str.strip().str.lower()
            df[col] = text.map({"true": True, "1": True, "false": False, "0": False}).astype("boolean")
    return df


def embed_project_bundle(pdf_bytes: bytes, bundle_bytes: bytes) -> bytes:
    """Embed a project bundle in an existing PDF without changing its pages."""
    try:
        from pypdf import PdfReader, PdfWriter
    except Exception as exc:  # pragma: no cover - deployment dependency guard
        raise RuntimeError("pypdf is required to create resumable PDF reports") from exc

    reader = PdfReader(io.BytesIO(pdf_bytes))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    metadata = {}
    try:
        for key, value in (reader.metadata or {}).items():
            if key and value is not None:
                metadata[str(key)] = str(value)
    except Exception:
        pass
    metadata.update({
        "/ProductionTestProject": "1",
        "/ProductionTestProjectSchema": str(PROJECT_SCHEMA_VERSION),
    })
    writer.add_metadata(metadata)
    writer.add_attachment(PROJECT_ATTACHMENT_NAME, bundle_bytes)
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def _iter_pdf_attachments(reader) -> Iterable[tuple[str, bytes]]:
    attachments = getattr(reader, "attachments", None)
    if isinstance(attachments, dict):
        for name, value in attachments.items():
            items = value if isinstance(value, list) else [value]
            for item in items:
                if isinstance(item, (bytes, bytearray)):
                    yield str(name), bytes(item)
    # Compatibility fallback for older pypdf versions.
    try:
        names = reader.trailer["/Root"].get("/Names")
        embedded = names.get("/EmbeddedFiles") if names else None
        arr = embedded.get("/Names") if embedded else None
        if arr:
            for i in range(0, len(arr), 2):
                name = str(arr[i])
                spec = arr[i + 1].get_object()
                ef = spec.get("/EF", {}).get("/F")
                if ef is not None:
                    yield name, bytes(ef.get_object().get_data())
    except Exception:
        pass


def extract_embedded_project(pdf_bytes: bytes) -> Optional[Dict[str, Any]]:
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception:
        return None

    seen = set()
    for name, payload in _iter_pdf_attachments(reader):
        sig = (name, len(payload))
        if sig in seen:
            continue
        seen.add(sig)
        if Path(name).name != PROJECT_ATTACHMENT_NAME and not name.lower().endswith("production_test_project_v1.zip"):
            continue
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                state = json.loads(zf.read("state.json").decode("utf-8"))
                df = _restore_dataframe(zf.read("data.json"), state)
            return {
                "data": df,
                "state": state,
                "source_kind": "embedded_project",
                "exact": True,
            }
        except Exception:
            continue
    return None


def _feature_key(label: str) -> Optional[str]:
    normalized = _normalize_label(label)
    if normalized in FEATURE_LABEL_TO_KEY:
        return FEATURE_LABEL_TO_KEY[normalized]
    # Small tolerance for labels split or slightly changed by older exports.
    if "gross" in normalized and "rate" in normalized:
        return "gross_rate_bpd"
    if "oil" in normalized and "rate" in normalized:
        return "oil_rate_stbd"
    if "water" in normalized and "rate" in normalized:
        return "water_rate_bpd"
    if "gas" in normalized and "rate" in normalized:
        return "gas_rate_mmscfd"
    if "bs&w" in normalized or "water cut" in normalized:
        return "bsw_pct"
    if normalized.startswith("whp"):
        return "whp_psi"
    if "separator" in normalized and "pressure" in normalized:
        return "sep_p_psi"
    if "pumping" in normalized and "pressure" in normalized:
        return "pumping_pressure_psi"
    if "choke" in normalized and "%" in normalized:
        return "choke_pct"
    if "choke" in normalized and "64" in normalized:
        return "choke_size_64"
    if "salinity" in normalized:
        return "salinity_kppm"
    return None


def _round_color(color: Any) -> Optional[tuple[float, float, float]]:
    if color is None or len(color) < 3:
        return None
    return tuple(round(float(v), 2) for v in color[:3])


def _color_is_visible_note(color: Optional[tuple[float, float, float]]) -> bool:
    if not color:
        return False
    hi, lo = max(color), min(color)
    return hi >= 0.55 and (hi - lo) >= 0.18


def _round_timestamp(ts: pd.Timestamp, minutes: int) -> pd.Timestamp:
    minutes = max(1, int(minutes or 30))
    epoch = pd.Timestamp("1970-01-01")
    total = (pd.Timestamp(ts) - epoch).total_seconds() / 60.0
    return epoch + pd.Timedelta(minutes=round(total / minutes) * minutes)


def _infer_round_minutes(ticks: list[tuple[float, pd.Timestamp]]) -> int:
    diffs = []
    for (_, a), (_, b) in zip(ticks, ticks[1:]):
        minutes = int(round((b - a).total_seconds() / 60.0))
        if 0 < minutes <= 12 * 60:
            diffs.append(minutes)
    gcd_value = 0
    for value in diffs:
        gcd_value = math.gcd(gcd_value, value)
    return max(5, min(gcd_value or 30, 60))


def _x_time_mapper(ticks: list[tuple[float, pd.Timestamp]]):
    ticks = sorted(ticks, key=lambda item: item[0])
    x_values = np.array([x for x, _ in ticks], dtype=float)
    ratios = []
    for (x1, t1), (x2, t2) in zip(ticks, ticks[1:]):
        hours = (t2 - t1).total_seconds() / 3600.0
        if 0 < hours <= 6 and x2 > x1:
            ratios.append((x2 - x1) / hours)
    scale = float(np.median(ratios)) if ratios else max((x_values[-1] - x_values[0]) / 24.0, 1.0)
    rounding_minutes = _infer_round_minutes(ticks)

    def map_x(x: float) -> pd.Timestamp:
        x = float(x)
        nearest = int(np.argmin(np.abs(x_values - x)))
        if abs(x_values[nearest] - x) <= max(2.0, scale * 0.12):
            return ticks[nearest][1]
        pos = int(np.searchsorted(x_values, x))
        if pos <= 0:
            result = ticks[0][1] + pd.Timedelta(hours=(x - x_values[0]) / scale)
        elif pos >= len(ticks):
            result = ticks[-1][1] + pd.Timedelta(hours=(x - x_values[-1]) / scale)
        else:
            lx, lt = ticks[pos - 1]
            rx, rt = ticks[pos]
            actual_hours = (rt - lt).total_seconds() / 3600.0
            visual_hours = (rx - lx) / scale
            if actual_hours <= 6 or abs(visual_hours - actual_hours) <= max(0.5, actual_hours * 0.25):
                fraction = (x - lx) / max(rx - lx, 1e-9)
                result = lt + (rt - lt) * fraction
            else:
                left_hours = (x - lx) / scale
                right_hours = (rx - x) / scale
                if left_hours <= 4 and (left_hours <= right_hours or right_hours > 4):
                    result = lt + pd.Timedelta(hours=left_hours)
                elif right_hours <= 4:
                    result = rt - pd.Timedelta(hours=right_hours)
                else:
                    fraction = (x - lx) / max(rx - lx, 1e-9)
                    result = lt + (rt - lt) * fraction
        return _round_timestamp(pd.Timestamp(result), rounding_minutes)

    return map_x


def _legacy_dashboard_pdf(pdf_bytes: bytes, source_name: str) -> Optional[Dict[str, Any]]:
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None

    row_values: Dict[pd.Timestamp, Dict[str, Any]] = {}
    panel_series: Dict[str, list[tuple[float, float, pd.Timestamp]]] = {}
    feature_order: list[str] = []
    recovered_events: list[dict] = []
    recovered_intervals: list[dict] = []
    well_name = "Unknown"
    first_mapper = None
    theme = "Light"
    x_axis_mode = "Real calendar time"
    dashboard_signature = False

    for page_no, page in enumerate(doc):
        full_text = page.get_text("text") or ""
        if "Compressed real-date timeline" in full_text or "Compressed real dates" in full_text:
            x_axis_mode = "Compressed real dates - remove empty gaps"
            dashboard_signature = True
        well_matches = list(re.finditer(r"\bWell[ \t]+([A-Za-z0-9][A-Za-z0-9_.\-/ ]{0,60})", full_text))
        for well_match in reversed(well_matches):
            candidate = well_match.group(1).splitlines()[0].strip()
            if candidate and not candidate.upper().startswith(("SIWHP", "WHP", "CLOSED")):
                well_name = re.sub(r"\s+", " ", candidate)
                break

        blocks_raw = page.get_text("blocks")
        blocks = [
            {"x0": b[0], "y0": b[1], "x1": b[2], "y1": b[3], "text": str(b[4]).strip(), "block": int(b[5])}
            for b in blocks_raw
        ]
        words = [
            {"x0": w[0], "y0": w[1], "x1": w[2], "y1": w[3], "text": str(w[4]).strip(), "block": int(w[5])}
            for w in page.get_text("words")
        ]
        feature_blocks = []
        for block in blocks:
            key = _feature_key(block["text"])
            if key and block["x0"] < max(100.0, page.rect.width * 0.12):
                feature_blocks.append({**block, "feature": key})
        feature_blocks.sort(key=lambda item: (item["y0"] + item["y1"]) / 2.0)
        if len(feature_blocks) >= 2 and re.search(r"\bWell[ \t]+[A-Za-z0-9]", full_text):
            dashboard_signature = True
        if not feature_blocks:
            continue

        try:
            pix = page.get_pixmap(matrix=fitz.Matrix(0.08, 0.08), alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8)
            if arr.size and float(arr.mean()) < 150:
                theme = "Dark"
        except Exception:
            pass

        centers = np.array([(b["y0"] + b["y1"]) / 2.0 for b in feature_blocks], dtype=float)

        def nearest_panel(y_center: float) -> int:
            return int(np.argmin(np.abs(centers - float(y_center))))

        page_panel_mappers = {}
        for panel_index, feature_block in enumerate(feature_blocks):
            feature = feature_block["feature"]
            if feature not in feature_order:
                feature_order.append(feature)
            panel_words = [
                w for w in words
                if nearest_panel((w["y0"] + w["y1"]) / 2.0) == panel_index
            ]
            panel_blocks = [
                b for b in blocks
                if nearest_panel((b["y0"] + b["y1"]) / 2.0) == panel_index
            ]
            date_words = sorted(
                [w for w in panel_words if re.fullmatch(r"\d{2}-[A-Za-z]{3}-\d{4}", w["text"])],
                key=lambda w: w["block"],
            )
            time_words = sorted(
                [w for w in panel_words if re.fullmatch(r"\d{1,2}:\d{2}", w["text"])],
                key=lambda w: w["block"],
            )
            ticks = []
            for date_word in date_words:
                candidates = [
                    t for t in time_words
                    if t["block"] > date_word["block"] and t["block"] - date_word["block"] <= 2
                ]
                if not candidates:
                    continue
                time_word = min(candidates, key=lambda item: item["block"])
                parsed = pd.to_datetime(
                    f"{date_word['text']} {time_word['text']}",
                    format="%d-%b-%Y %H:%M",
                    errors="coerce",
                )
                if pd.notna(parsed):
                    # The right edge of a rotated time label aligns closely with
                    # the actual chart tick position in matplotlib PDF output.
                    ticks.append((float(time_word["x1"]), pd.Timestamp(parsed)))
            if len(ticks) < 2:
                continue
            mapper = _x_time_mapper(ticks)
            page_panel_mappers[panel_index] = mapper
            if first_mapper is None:
                first_mapper = mapper

            date_y = min(w["y0"] for w in date_words)
            text_event_blocks = {
                b["block"] for b in panel_blocks
                if re.search(r"[A-Za-z]", b["text"]) and re.search(r"\d", b["text"])
            }
            numeric_words = []
            for word in panel_words:
                if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", word["text"]):
                    continue
                if word["x0"] <= max(70.0, page.rect.width * 0.05):
                    continue
                if word["y0"] >= date_y - 1:
                    continue
                if word["block"] in text_event_blocks:
                    continue
                numeric_words.append(word)
            numeric_words.sort(key=lambda w: (w["x0"] + w["x1"]) / 2.0)

            for word in numeric_words:
                x_center = (word["x0"] + word["x1"]) / 2.0
                timestamp = mapper(x_center)
                value = pd.to_numeric(pd.Series([word["text"]]), errors="coerce").iloc[0]
                if pd.isna(value):
                    continue
                panel_series.setdefault(feature, []).append((x_center, float(value), pd.Timestamp(timestamp)))
                row_values.setdefault(timestamp, {})[feature] = float(value)

        # Recover events and intervals from the first panel only; they are
        # repeated on every panel in legacy multi-chart exports.
        if page_no == 0 and first_mapper is not None:
            first_center = centers[0]
            first_blocks = [
                b for b in blocks
                if nearest_panel((b["y0"] + b["y1"]) / 2.0) == 0
            ]
            label_blocks = [
                b for b in first_blocks
                if b["text"] and re.search(r"[A-Za-z]", b["text"])
                and _feature_key(b["text"]) is None
                and not re.search(r"\d{2}-[A-Za-z]{3}-\d{4}", b["text"])
                and not re.fullmatch(r"\d{1,2}:\d{2}", b["text"])
                and not b["text"].lower().startswith("well ")
            ]

            vertical_lines = []
            for drawing in page.get_drawings():
                dashes = str(drawing.get("dashes") or "")
                if not dashes or dashes == "[] 0":
                    continue
                color = _round_color(drawing.get("color"))
                width = float(drawing.get("width") or 0.0)
                for item in drawing.get("items", []):
                    if not item or item[0] != "l":
                        continue
                    p1, p2 = item[1], item[2]
                    if abs(float(p1.x) - float(p2.x)) > 0.8:
                        continue
                    if abs(float(p1.y) - float(p2.y)) < page.rect.height * 0.04:
                        continue
                    if nearest_panel((float(p1.y) + float(p2.y)) / 2.0) != 0:
                        continue
                    vertical_lines.append({
                        "x": (float(p1.x) + float(p2.x)) / 2.0,
                        "color": color,
                        "width": width,
                    })

            visible_lines = [line for line in vertical_lines if _color_is_visible_note(line["color"])]
            grouped: Dict[tuple, list[float]] = {}
            for line in visible_lines:
                grouped.setdefault((line["color"], round(line["width"], 1)), []).append(line["x"])
            grouped = {key: sorted(set(round(x, 2) for x in xs)) for key, xs in grouped.items()}

            used_labels = set()
            for key, xs in grouped.items():
                if len(xs) < 2:
                    continue
                for left, right in zip(xs, xs[1:]):
                    label_candidates = []
                    for block in label_blocks:
                        x_mid = (block["x0"] + block["x1"]) / 2.0
                        if left <= x_mid <= right and block["y0"] > 45 and block["y0"] < first_center:
                            label_candidates.append(block)
                    if not label_candidates:
                        continue
                    label_block = min(label_candidates, key=lambda b: abs(((b["x0"] + b["x1"]) / 2.0) - (left + right) / 2.0))
                    label = re.sub(r"\s+", " ", label_block["text"]).strip()
                    if len(label) < 3 or label_block["block"] in used_labels:
                        continue
                    recovered_intervals.append({
                        "start": first_mapper(left),
                        "end": first_mapper(right),
                        "label": label,
                        "target": well_name,
                    })
                    used_labels.add(label_block["block"])

            interval_block_ids = used_labels
            for block in label_blocks:
                if block["block"] in interval_block_ids:
                    continue
                label = re.sub(r"\s+", " ", block["text"]).strip()
                if len(label) < 3:
                    continue
                x_mid = (block["x0"] + block["x1"]) / 2.0
                nearest_line = None
                if visible_lines:
                    nearest_line = min(visible_lines, key=lambda line: abs(line["x"] - x_mid))
                if nearest_line is None or abs(nearest_line["x"] - x_mid) > max(25.0, page.rect.width * 0.025):
                    continue
                recovered_events.append({
                    "datetime": first_mapper(nearest_line["x"]),
                    "label": label,
                    "target": well_name,
                })

    # Legacy dashboard exports generally label the same readings in every panel.
    # Align panels by visible point order when counts match. This is more reliable
    # than independently interpolating compressed-date x coordinates, especially
    # around long shut-in gaps.
    usable_series = {
        key: sorted(values, key=lambda item: item[0])
        for key, values in panel_series.items()
        if values
    }
    preferred = ["gas_rate_mmscfd", "gross_rate_bpd", "oil_rate_stbd", "water_rate_bpd"]
    master_key = next(
        (key for key in preferred if key in usable_series and len(usable_series[key]) == max(map(len, usable_series.values()))),
        max(usable_series, key=lambda key: len(usable_series[key])) if usable_series else None,
    )
    if master_key is not None:
        master_times = [item[2] for item in usable_series[master_key]]
        rebuilt: Dict[pd.Timestamp, Dict[str, Any]] = {}
        for feature, values in usable_series.items():
            if len(values) == len(master_times):
                for timestamp, (_, value, _) in zip(master_times, values):
                    rebuilt.setdefault(pd.Timestamp(timestamp), {})[feature] = float(value)
            else:
                for _, value, timestamp in values:
                    rebuilt.setdefault(pd.Timestamp(timestamp), {})[feature] = float(value)
        row_values = rebuilt

    if not dashboard_signature or not row_values or len(feature_order) < 1:
        return None

    rows = []
    for timestamp in sorted(row_values):
        row = {
            "datetime": pd.Timestamp(timestamp),
            "date": pd.Timestamp(timestamp).normalize(),
            "time_text": pd.Timestamp(timestamp).strftime("%H:%M"),
            "well": well_name,
            "source": source_name,
            "sheet": "Recovered dashboard PDF",
            "source_type": "dashboard_pdf_resume",
            "pdf_resume_recovered": True,
        }
        row.update(row_values[timestamp])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("datetime").reset_index(drop=True)

    # Remove repeated event/interval entries and reject zero-length intervals.
    event_seen = set()
    events = []
    for event in recovered_events:
        key = (pd.Timestamp(event["datetime"]).isoformat(), event["label"], event.get("target", ""))
        if key not in event_seen:
            event_seen.add(key)
            events.append(event)
    interval_seen = set()
    intervals = []
    for interval in recovered_intervals:
        start = pd.Timestamp(interval["start"])
        end = pd.Timestamp(interval["end"])
        if end <= start:
            continue
        key = (start.isoformat(), end.isoformat(), interval["label"], interval.get("target", ""))
        if key not in interval_seen:
            interval_seen.add(key)
            intervals.append(interval)

    state = {
        "schema_version": PROJECT_SCHEMA_VERSION,
        "source_kind": "legacy_dashboard_pdf",
        "exact_restore": False,
        "chart": {
            "theme": theme,
            "analysis_view": "Test detail",
            "selected_wells": [well_name] if well_name and well_name != "Unknown" else [],
            "selected_features": feature_order,
            "signal_order": feature_order,
            "pressure_display_unit": "psi",
            "temperature_display_unit": "Keep detected unit",
            "time_aggregation": "Raw data",
            "x_axis_scale": "Auto readable",
            "x_axis_mode": x_axis_mode,
            "continuous_gap_hours": 2.0,
            "compressed_gap_hours": 0.75,
            "plot_mode": "Separate panels like report",
            "show_points": True,
            "value_label_mode": "Clean readable - recommended",
            "value_label_step": 20,
            "chart_title": f"Well {well_name}" if well_name and well_name != "Unknown" else "Production Test",
        },
        "events": events,
        "intervals": intervals,
        "restore_note": (
            "Recovered from visible values and event markers in an older dashboard PDF. "
            "New PDFs exported by this version carry an exact embedded project snapshot."
        ),
    }
    return {
        "data": df,
        "state": state,
        "source_kind": "legacy_dashboard_pdf",
        "exact": False,
    }


def extract_project_from_pdf(pdf_bytes: bytes, source_name: str = "dashboard_report.pdf") -> Optional[Dict[str, Any]]:
    embedded = extract_embedded_project(pdf_bytes)
    if embedded is not None:
        return embedded
    return _legacy_dashboard_pdf(pdf_bytes, source_name)
