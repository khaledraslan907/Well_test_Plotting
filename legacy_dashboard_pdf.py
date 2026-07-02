from __future__ import annotations

"""Recover data and chart state from older Matplotlib dashboard PDFs.

Dashboard PDFs exported before v97 did not embed the portable CSV/state package.
They are nevertheless vector PDFs: plotted curves, grid lines, tick labels, and
operation annotations are stored as PDF drawing commands.  This module reads
those vectors directly and reconstructs the underlying time series without OCR.

The recovery is deliberately conservative. It activates only for Matplotlib
PDFs containing multiple production-test axes with real date/time tick labels.
Ordinary vendor PDFs continue through the normal parser.
"""

from dataclasses import dataclass
from datetime import datetime
import io
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

RECOVERY_BUILD_ID = "v98-legacy-vector-pdf-recovery-20260702"


@dataclass(frozen=True)
class AxisBox:
    x0: float
    x1: float
    top: float
    bottom: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.bottom - self.top


# Normalized dashboard labels -> canonical parser fields.
_LABEL_ALIASES = {
    "totalgasratemmscfd": "gas_rate_mmscfd",
    "gasratemmscfd": "gas_rate_mmscfd",
    "formationgasratemmscfd": "gas_formation_mmscfd",
    "grossratebbld": "gross_rate_bpd",
    "grossratebp d": "gross_rate_bpd",
    "oilratestbd": "oil_rate_stbd",
    "oilratebbld": "oil_rate_stbd",
    "waterratebbld": "water_rate_bpd",
    "waterratebpd": "water_rate_bpd",
    "bsw": "bsw_pct",
    "bswpercent": "bsw_pct",
    "watercut": "bsw_pct",
    "whppsi": "whp_psi",
    "wellheadpressurepsi": "whp_psi",
    "flowlinepressurepsi": "flp_psi",
    "flppsi": "flp_psi",
    "separatorpressurepsi": "sep_p_psi",
    "pumpingpressurepsi": "pumping_pressure_psi",
    "chokeopening": "choke_pct",
    "chokeopeningpercent": "choke_pct",
    "chokesize64in": "choke_size_64",
    "salinitykppmnacl": "salinity_kppm",
    "salinitykppm": "salinity_kppm",
    "oilgravityapi": "oil_api",
    "gasspecificgravity": "gas_sg",
    "h2sppm": "h2s_ppm",
    "co2mole": "co2_mole_pct",
    "gor scfbbl": "gor_scf_bbl",
}

# Stroke colors used by the dashboard exports. Label matching remains primary,
# while this map provides a reliable fallback when a Type-3 font extracts poorly.
_COLOR_FEATURES = {
    (0.000, 0.675, 0.757): "gas_rate_mmscfd",
    (0.376, 0.490, 0.545): "gross_rate_bpd",
    (0.180, 0.490, 0.196): "oil_rate_stbd",
    (0.098, 0.463, 0.824): "water_rate_bpd",
    (0.557, 0.267, 0.678): "bsw_pct",
    (0.776, 0.157, 0.157): "whp_psi",
    (0.961, 0.486, 0.000): "sep_p_psi",
    (0.769, 0.604, 0.267): "choke_pct",
    (0.553, 0.431, 0.388): "salinity_kppm",
}

_DATE_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{4}$")
_TIME_RE = re.compile(r"^\d{1,2}:\d{2}$")
_NUMBER_RE = re.compile(r"^[-+]?(?:\d+(?:\.\d+)?|\.\d+)$")


def _norm(value: Any) -> str:
    text = str(value or "").lower().replace("&", "and")
    text = text.replace("%", "percent")
    return re.sub(r"[^a-z0-9]+", "", text)


def _color_tuple(value: Any) -> Optional[tuple[float, float, float]]:
    if not isinstance(value, (tuple, list)) or len(value) < 3:
        return None
    try:
        return tuple(round(float(value[i]), 3) for i in range(3))
    except Exception:
        return None


def _color_distance(a: Optional[tuple[float, float, float]], b: tuple[float, float, float]) -> float:
    if a is None:
        return 99.0
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def _feature_from_color(color: Any) -> Optional[str]:
    c = _color_tuple(color)
    if c is None:
        return None
    feature, distance = None, 99.0
    for ref, candidate in _COLOR_FEATURES.items():
        d = _color_distance(c, ref)
        if d < distance:
            feature, distance = candidate, d
    return feature if distance <= 0.08 else None


def _canonical_feature(label: str, color: Any = None) -> Optional[str]:
    key = _norm(label)
    if key in _LABEL_ALIASES:
        return _LABEL_ALIASES[key]

    # Flexible matching for harmless font/extraction variations.
    rules = [
        (("totalgasrate", "gasrate"), "gas_rate_mmscfd"),
        (("grossrate",), "gross_rate_bpd"),
        (("oilrate",), "oil_rate_stbd"),
        (("waterrate",), "water_rate_bpd"),
        (("bsw", "watercut"), "bsw_pct"),
        (("separatorpressure",), "sep_p_psi"),
        (("pumpingpressure",), "pumping_pressure_psi"),
        (("wellheadpressure", "whppsi"), "whp_psi"),
        (("flowlinepressure", "flppsi"), "flp_psi"),
        (("chokeopening",), "choke_pct"),
        (("chokesize",), "choke_size_64"),
        (("salinity",), "salinity_kppm"),
        (("oilgravity",), "oil_api"),
        (("gasspecificgravity",), "gas_sg"),
    ]
    for alternatives, feature in rules:
        if any(token in key for token in alternatives):
            return feature
    return _feature_from_color(color)


def _detect_axes(page) -> list[AxisBox]:
    axes: list[AxisBox] = []
    for rect in getattr(page, "rects", []) or []:
        try:
            width = float(rect.get("width", 0) or 0)
            height = float(rect.get("height", 0) or 0)
            x0, x1 = float(rect["x0"]), float(rect["x1"])
            top, bottom = float(rect["top"]), float(rect["bottom"])
        except Exception:
            continue
        if width < float(page.width) * 0.58 or height < 90:
            continue
        if width > float(page.width) * 0.98 and height > float(page.height) * 0.90:
            continue
        if top < 15 or bottom > float(page.height) - 5:
            continue
        axes.append(AxisBox(x0=x0, x1=x1, top=top, bottom=bottom))

    # Remove nearly identical duplicates while retaining top-to-bottom order.
    unique: list[AxisBox] = []
    for axis in sorted(axes, key=lambda a: (a.top, a.x0)):
        if any(abs(axis.top - old.top) < 2 and abs(axis.bottom - old.bottom) < 2 for old in unique):
            continue
        unique.append(axis)
    return unique


def _axis_label(page, axis: AxisBox) -> str:
    chars = []
    for ch in getattr(page, "chars", []) or []:
        try:
            x0, x1 = float(ch["x0"]), float(ch["x1"])
            top, bottom = float(ch["top"]), float(ch["bottom"])
        except Exception:
            continue
        if bool(ch.get("upright", True)):
            continue
        if x1 > axis.x0 - 2 or bottom < axis.top - 2 or top > axis.bottom + 2:
            continue
        text = str(ch.get("text", ""))
        if not text.strip():
            continue
        chars.append((round((x0 + x1) / 2, 1), top, text))
    if not chars:
        return ""

    # Select the vertical character column containing the most letters.
    groups: dict[float, list[tuple[float, str]]] = {}
    for xmid, top, text in chars:
        groups.setdefault(xmid, []).append((top, text))
    selected = max(groups.values(), key=lambda items: sum(c.isalnum() for _, c in items))
    # Matplotlib rotates the label bottom-to-top, so reverse visual top order.
    return "".join(text for _, text in sorted(selected, key=lambda item: item[0], reverse=True)).strip()


def _horizontal_grid_positions(page, axis: AxisBox) -> list[float]:
    ys: list[float] = []
    for line in getattr(page, "lines", []) or []:
        try:
            x0, x1 = float(line["x0"]), float(line["x1"])
            top, bottom = float(line["top"]), float(line["bottom"])
            width = float(line.get("linewidth", 0) or 0)
        except Exception:
            continue
        if abs(top - bottom) > 0.8:
            continue
        if x1 - x0 < axis.width * 0.78:
            continue
        if top < axis.top - 1 or top > axis.bottom + 1:
            continue
        if width > 1.2:
            continue
        ys.append((top + bottom) / 2)
    return sorted(set(round(y, 4) for y in ys))


def _vertical_grid_positions(page, axis: AxisBox) -> list[float]:
    xs: list[float] = []
    for line in getattr(page, "lines", []) or []:
        try:
            x0, x1 = float(line["x0"]), float(line["x1"])
            top, bottom = float(line["top"]), float(line["bottom"])
            width = float(line.get("linewidth", 0) or 0)
        except Exception:
            continue
        if abs(x0 - x1) > 0.8:
            continue
        if bottom - top < axis.height * 0.78:
            continue
        x = (x0 + x1) / 2
        if x <= axis.x0 + 1 or x >= axis.x1 - 1:
            continue
        if top > axis.top + 2 or bottom < axis.bottom - 2:
            continue
        if width > 1.2:
            continue
        xs.append(x)
    return sorted(set(round(x, 4) for x in xs))


def _numeric_tick_words(words: Iterable[dict], axis: AxisBox) -> list[tuple[float, float]]:
    ticks: list[tuple[float, float]] = []
    for word in words:
        text = str(word.get("text", "")).strip().replace(",", "")
        if not _NUMBER_RE.fullmatch(text):
            continue
        if not bool(word.get("upright", True)):
            continue
        try:
            x1 = float(word["x1"])
            y = (float(word["top"]) + float(word["bottom"])) / 2
        except Exception:
            continue
        if x1 > axis.x0 - 1:
            continue
        if y < axis.top - 8 or y > axis.bottom + 8:
            continue
        try:
            ticks.append((y, float(text)))
        except ValueError:
            continue
    return sorted(ticks, key=lambda item: item[0])


def _fit_y_scale(page, words: list[dict], axis: AxisBox) -> Optional[tuple[float, float]]:
    grid = _horizontal_grid_positions(page, axis)
    ticks = _numeric_tick_words(words, axis)
    if len(grid) < 2 or len(ticks) < 2:
        return None

    pairs: list[tuple[float, float]] = []
    used: set[int] = set()
    for word_y, value in ticks:
        idx = min(range(len(grid)), key=lambda i: abs(grid[i] - word_y))
        if idx in used:
            continue
        # Font baseline offsets are small relative to normal grid spacing.
        spacing = np.median(np.diff(grid)) if len(grid) > 2 else axis.height
        if abs(grid[idx] - word_y) > max(14.0, float(spacing) * 0.42):
            continue
        used.add(idx)
        pairs.append((grid[idx], value))
    if len(pairs) < 2:
        n = min(len(grid), len(ticks))
        pairs = [(grid[i], ticks[i][1]) for i in range(n)]
    if len(pairs) < 2:
        return None

    ys = np.array([p[0] for p in pairs], dtype=float)
    vals = np.array([p[1] for p in pairs], dtype=float)
    slope, intercept = np.polyfit(ys, vals, 1)
    if not np.isfinite(slope) or abs(slope) < 1e-12:
        return None
    return float(slope), float(intercept)


def _extract_datetime_anchors(page, axis: AxisBox, datetime_sequence: list[pd.Timestamp]) -> list[tuple[float, pd.Timestamp]]:
    grid_x = _vertical_grid_positions(page, axis)
    n = min(len(grid_x), len(datetime_sequence))
    if n < 2:
        return []
    # Matplotlib emits tick labels and grid lines in the same left-to-right order.
    return [(float(grid_x[i]), pd.Timestamp(datetime_sequence[i])) for i in range(n)]


def _datetime_tick_sequence(pdf_bytes: bytes) -> list[pd.Timestamp]:
    """Read the first repeated x-axis date/time sequence with pypdf.

    Type-3 rotated fonts are often fragmented by pdfplumber, while pypdf keeps
    each rendered date and time together in extraction order.
    """
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join((page.extract_text() or "") for page in reader.pages[:2])
    except Exception:
        return []
    pairs = re.findall(r"(\d{1,2}-[A-Za-z]{3}-\d{4})\s*(\d{1,2}:\d{2})", text)
    sequence: list[pd.Timestamp] = []
    for date_text, time_text in pairs:
        dt = pd.to_datetime(f"{date_text} {time_text}", format="%d-%b-%Y %H:%M", errors="coerce")
        if pd.isna(dt):
            continue
        stamp = pd.Timestamp(dt)
        # The same tick sequence is repeated under every panel. Stop when the
        # first tick appears again after at least two distinct ticks.
        if len(sequence) >= 2 and stamp == sequence[0]:
            break
        sequence.append(stamp)
    return sequence


def _normal_pixels_per_hour(anchors: list[tuple[float, pd.Timestamp]]) -> Optional[float]:
    candidates: list[float] = []
    for (x0, dt0), (x1, dt1) in zip(anchors, anchors[1:]):
        hours = (dt1 - dt0).total_seconds() / 3600.0
        dx = x1 - x0
        if 0.05 <= hours <= 8.0 and dx > 1:
            candidates.append(dx / hours)
    if not candidates:
        return None
    result = float(np.median(candidates))
    return result if np.isfinite(result) and result > 0 else None


def _x_to_datetime(x: float, anchors: list[tuple[float, pd.Timestamp]], px_per_hour: float) -> pd.Timestamp:
    anchor_x, anchor_dt = min(anchors, key=lambda item: abs(item[0] - x))
    dt = anchor_dt + pd.Timedelta(hours=(float(x) - float(anchor_x)) / float(px_per_hour))
    # PDF coordinates retain sub-second precision artifacts; dashboard readings
    # are minute based, so normalize to the nearest minute.
    return pd.Timestamp(dt).round("min")


def _object_points(obj: dict) -> list[tuple[float, float]]:
    points = obj.get("pts") or []
    result: list[tuple[float, float]] = []
    for point in points:
        try:
            result.append((float(point[0]), float(point[1])))
        except Exception:
            continue
    if result:
        return result
    try:
        return [(float(obj["x0"]), float(obj["top"])), (float(obj["x1"]), float(obj["bottom"]))]
    except Exception:
        return []


def _signal_objects(page, axis: AxisBox) -> tuple[list[dict], Any]:
    candidates: list[dict] = []
    for obj in list(getattr(page, "curves", []) or []) + list(getattr(page, "lines", []) or []):
        points = _object_points(obj)
        if len(points) < 2:
            continue
        try:
            linewidth = float(obj.get("linewidth", 0) or 0)
        except Exception:
            linewidth = 0.0
        if linewidth < 1.9:
            continue
        inside = [(x, y) for x, y in points if axis.x0 - 2 <= x <= axis.x1 + 2 and axis.top - 2 <= y <= axis.bottom + 2]
        if len(inside) < 2:
            continue
        xs = [p[0] for p in inside]
        if max(xs) - min(xs) < 3.0:
            continue
        item = dict(obj)
        item["_points"] = inside
        item["_linewidth"] = linewidth
        candidates.append(item)
    if not candidates:
        return [], None

    # The plotted signal is the thickest, longest color group. Operation arrows
    # are thinner and therefore cannot win this selection.
    max_width = max(c["_linewidth"] for c in candidates)
    candidates = [c for c in candidates if c["_linewidth"] >= max_width - 0.18]
    groups: dict[Any, list[dict]] = {}
    for item in candidates:
        groups.setdefault(_color_tuple(item.get("stroking_color")), []).append(item)

    def score(items: list[dict]) -> float:
        xs = [x for item in items for x, _ in item["_points"]]
        return len(set(round(x, 3) for x in xs)) * 10.0 + (max(xs) - min(xs) if xs else 0.0)

    color, objects = max(groups.items(), key=lambda kv: score(kv[1]))
    return sorted(objects, key=lambda obj: min(x for x, _ in obj["_points"])), color


def _dedupe_step_points(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not points:
        return []
    # For a steps-post path, the same X appears first at the preceding value and
    # then at the new value. Keeping the last point recovers the original sample.
    ordered: dict[float, tuple[float, float]] = {}
    for x, y in points:
        ordered[round(x, 5)] = (float(x), float(y))
    return [ordered[key] for key in sorted(ordered)]


def _clean_vertical_label(raw: str) -> str:
    text = re.sub(r"\s+", "", str(raw or ""))
    m = re.fullmatch(r"(?i)(SIWHP)(\d+(?:\.\d+)?)(PSI)", text)
    if m:
        return f"{m.group(1).upper()} {m.group(2)} {m.group(3).upper()}"
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    return text.strip()


def _vertical_text_near(page, axis: AxisBox, x: float) -> str:
    chars: list[tuple[float, str]] = []
    for ch in getattr(page, "chars", []) or []:
        if bool(ch.get("upright", True)):
            continue
        try:
            xmid = (float(ch["x0"]) + float(ch["x1"])) / 2
            top, bottom = float(ch["top"]), float(ch["bottom"])
        except Exception:
            continue
        if abs(xmid - x) > 16:
            continue
        if bottom < axis.top + 10 or top > axis.bottom - 10:
            continue
        text = str(ch.get("text", ""))
        if text.strip():
            chars.append((top, text))
    if not chars:
        return ""
    return _clean_vertical_label("".join(text for _, text in sorted(chars, key=lambda item: item[0], reverse=True)))


def _event_and_interval_state(page, words: list[dict], axis: AxisBox,
                              anchors: list[tuple[float, pd.Timestamp]], px_per_hour: float) -> tuple[list[dict], list[dict]]:
    verticals: list[dict] = []
    for line in getattr(page, "lines", []) or []:
        try:
            x0, x1 = float(line["x0"]), float(line["x1"])
            top, bottom = float(line["top"]), float(line["bottom"])
            width = float(line.get("linewidth", 0) or 0)
        except Exception:
            continue
        if abs(x0 - x1) > 0.8 or bottom - top < axis.height * 0.80:
            continue
        if top > axis.top + 2 or bottom < axis.bottom - 2:
            continue
        if not (1.25 <= width <= 1.95):
            continue
        verticals.append({"x": (x0 + x1) / 2, "color": _color_tuple(line.get("stroking_color")), "width": width})

    gold = (1.000, 0.820, 0.400)
    interval_bounds = sorted(v["x"] for v in verticals if _color_distance(v["color"], gold) < 0.08)
    events: list[dict] = []
    for item in verticals:
        if _color_distance(item["color"], gold) < 0.08:
            continue
        label = _vertical_text_near(page, axis, item["x"])
        if not label or len(re.sub(r"[^A-Za-z0-9]", "", label)) < 3:
            continue
        events.append({
            "datetime": _x_to_datetime(item["x"], anchors, px_per_hour),
            "label": label,
            "target": "All selected wells",
            "x_shift_px": 0.0,
            "y_level": "Auto",
        })

    intervals: list[dict] = []
    # Locate the bidirectional interval arrow and its horizontal label.
    for curve in getattr(page, "curves", []) or []:
        try:
            color = _color_tuple(curve.get("stroking_color"))
            width = float(curve.get("linewidth", 0) or 0)
            x0, x1 = float(curve["x0"]), float(curve["x1"])
            top, bottom = float(curve["top"]), float(curve["bottom"])
        except Exception:
            continue
        if _color_distance(color, gold) >= 0.08 or not (1.45 <= width <= 1.90):
            continue
        if x1 - x0 < 20 or abs(top - bottom) > 3:
            continue
        if top < axis.top or top > axis.top + 40:
            continue
        if len(interval_bounds) < 2:
            continue
        left = min(interval_bounds, key=lambda x: abs(x - x0))
        right = min(interval_bounds, key=lambda x: abs(x - x1))
        if right <= left:
            continue
        label_words = []
        for word in words:
            if not bool(word.get("upright", True)):
                continue
            try:
                wx0, wx1 = float(word["x0"]), float(word["x1"])
                wtop, wbottom = float(word["top"]), float(word["bottom"])
            except Exception:
                continue
            if wx1 < left or wx0 > right:
                continue
            if wbottom < axis.top or wtop > axis.top + 35:
                continue
            text = str(word.get("text", "")).strip()
            if re.search(r"[A-Za-z]", text):
                label_words.append((wx0, text))
        label = " ".join(text for _, text in sorted(label_words)).strip() or "Operation interval"
        intervals.append({
            "start": _x_to_datetime(left, anchors, px_per_hour),
            "end": _x_to_datetime(right, anchors, px_per_hour),
            "label": label,
            "target": "All selected wells",
        })

    # Remove repeated labels/lines that can arise from compound PDF paths.
    event_seen, event_unique = set(), []
    for event in events:
        key = (pd.Timestamp(event["datetime"]), event["label"])
        if key not in event_seen:
            event_seen.add(key)
            event_unique.append(event)
    interval_seen, interval_unique = set(), []
    for interval in intervals:
        key = (pd.Timestamp(interval["start"]), pd.Timestamp(interval["end"]), interval["label"])
        if key not in interval_seen:
            interval_seen.add(key)
            interval_unique.append(interval)
    return event_unique, interval_unique


def _detect_theme(page) -> str:
    for rect in getattr(page, "rects", []) or []:
        try:
            if float(rect.get("width", 0)) < float(page.width) * 0.95:
                continue
            if float(rect.get("height", 0)) < float(page.height) * 0.90:
                continue
        except Exception:
            continue
        color = _color_tuple(rect.get("non_stroking_color"))
        if color is not None:
            luminance = 0.2126 * color[0] + 0.7152 * color[1] + 0.0722 * color[2]
            return "Dark" if luminance < 0.45 else "Light"
    return "Light"


def _well_and_title(text: str, file_name: str) -> tuple[str, str]:
    matches = re.findall(r"(?im)\bWell\s+([A-Za-z0-9][A-Za-z0-9._/\-]*)", text or "")
    # Plot annotations such as "Closed the Well" can be followed by a numeric
    # value in extraction order. Prefer the title-like token containing letters.
    candidates = [m.strip() for m in matches if re.search(r"[A-Za-z]", m) and m.lower() not in {"the", "closed"}]
    if candidates:
        well = max(candidates, key=lambda value: (bool(re.search(r"[-_/]", value)), len(value)))
        return well, f"Well {well}"
    stem = Path(str(file_name or "Legacy dashboard PDF")).stem
    stem = re.sub(r"(?i)\btest\s+resu?lts?\b", "", stem).strip(" _-()")
    return stem or "Unknown", f"Well {stem}" if stem else "Production Test"


def recover_legacy_dashboard_pdf(pdf_bytes: bytes, file_name: str = "uploaded.pdf") -> Optional[dict]:
    """Return recovered dataframe/state, or ``None`` for a non-dashboard PDF."""
    if not pdf_bytes or Path(str(file_name)).suffix.lower() != ".pdf":
        return None
    try:
        import pdfplumber
    except Exception:
        return None

    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if not pdf.pages:
                return None
            page = pdf.pages[0]
            metadata = dict(pdf.metadata or {})
            creator = " ".join(str(metadata.get(k, "")) for k in ("Creator", "Producer")).lower()
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            full_text = page.extract_text() or ""
            datetime_sequence = _datetime_tick_sequence(pdf_bytes)
            axes = _detect_axes(page)
            # Activation guard: do not treat ordinary reports as dashboard plots.
            if len(axes) < 2 or ("matplotlib" not in creator and "compressed real-date timeline" not in full_text.lower()):
                return None

            rows_by_dt: dict[pd.Timestamp, dict[str, Any]] = {}
            selected_features: list[str] = []
            axis_debug: list[dict[str, Any]] = []
            first_axis_state = None

            for i, axis in enumerate(axes):
                next_top = axes[i + 1].top if i + 1 < len(axes) else float(page.height)
                label = _axis_label(page, axis)
                y_scale = _fit_y_scale(page, words, axis)
                anchors = _extract_datetime_anchors(page, axis, datetime_sequence)
                px_per_hour = _normal_pixels_per_hour(anchors)
                objects, color = _signal_objects(page, axis)
                feature = _canonical_feature(label, color)
                if not feature or y_scale is None or len(anchors) < 2 or not px_per_hour or not objects:
                    axis_debug.append({"label": label, "feature": feature, "objects": len(objects), "anchors": len(anchors)})
                    continue

                slope, intercept = y_scale
                recovered_points = 0
                for obj in objects:
                    points = _dedupe_step_points(obj.get("_points", []))
                    for x, y in points:
                        dt = _x_to_datetime(x, anchors, px_per_hour)
                        value = round(float(slope * y + intercept), 4)
                        if abs(value) < 5e-5:
                            value = 0.0
                        row = rows_by_dt.setdefault(dt, {"datetime": dt})
                        # Prefer the first vector value at a timestamp; duplicate
                        # path objects are normally identical marker/segment data.
                        row.setdefault(feature, value)
                        recovered_points += 1
                if recovered_points >= 2 and feature not in selected_features:
                    selected_features.append(feature)
                if first_axis_state is None:
                    first_axis_state = (axis, anchors, px_per_hour)
                axis_debug.append({"label": label, "feature": feature, "points": recovered_points, "anchors": len(anchors)})

            if len(rows_by_dt) < 2 or not selected_features:
                return None

            well, chart_title = _well_and_title(full_text, file_name)
            records = [rows_by_dt[key] for key in sorted(rows_by_dt)]
            frame = pd.DataFrame(records).sort_values("datetime", kind="stable").reset_index(drop=True)
            frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
            frame["date"] = frame["datetime"].dt.floor("D")
            frame["time"] = frame["datetime"].dt.time
            frame["time_text"] = frame["datetime"].dt.strftime("%H:%M")
            frame["well"] = well
            frame["source"] = str(file_name)
            frame["sheet"] = "Legacy dashboard PDF vector recovery"
            frame["source_type"] = "legacy_dashboard_pdf_vector"
            frame["recovery_note"] = "Recovered from Matplotlib vector paths; no OCR used."

            events: list[dict] = []
            intervals: list[dict] = []
            if first_axis_state is not None:
                events, intervals = _event_and_interval_state(page, words, *first_axis_state)

            return {
                "data": frame,
                "theme": _detect_theme(page),
                "manual_events": events,
                "operation_intervals": intervals,
                "selected_features": selected_features,
                "chart_title": chart_title,
                "well": well,
                "x_axis_mode": "Compressed real dates - remove empty gaps" if "compressed real-date" in full_text.lower() else "Real calendar time",
                "recovery_build_id": RECOVERY_BUILD_ID,
                "axis_debug": axis_debug,
            }
    except Exception:
        return None
