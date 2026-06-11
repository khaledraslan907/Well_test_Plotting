
import io
import re
import zipfile
import traceback

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from tmu_parser import (
    apply_fill_method,
    available_numeric_columns,
    column_label,
    load_tabular_file,
    parse_many_tmu_messages,
)


FEATURE_COLORS = {
    "gross_rate_bpd": "#1f77b4",
    "oil_rate_stbd": "#2ca02c",
    "water_rate_bpd": "#17becf",
    "whp_psi": "#9467bd",
    "sep_p_psi": "#8c564b",
    "flp_psi": "#7f7f7f",
    "bsw_pct": "#bcbd22",
    "salinity_kppm": "#d62728",
    "pumping_pressure_psi": "#ff7f0e",
    "gas_rate_mmscfd": "#1f9ed4",
    "gas_formation_mmscfd": "#00b5ad",
    "choke_pct": "#e377c2",
    "n2_rate_scfm": "#6f42c1",
    "h2s_ppm": "#8c1c13",
    "co2_mole_pct": "#2f4f4f",
    "water_cum_bbl": "#4daf4a",
    "oil_api": "#a65628",
    "gas_sg": "#636363",
    "ct_pressure_psi": "#ff1493",
    "ct_depth_m": "#20b2aa",
    "ct_running_speed_ftmin": "#6495ed",
    "ct_pipe_weight_lbf": "#708090",
    "flow_press_psi": "#636efa",
    "flow_temp_c": "#ef553b",
    "mpfm_press_psig": "#ab63fa",
    "mpfm_temp_f": "#ffa15a",
    "dp_mbar": "#19d3f3",
    "qoil_s_stbd": "#2ca02c",
    "qwat_s_bpd": "#17becf",
    "qgas_s_mmscfd": "#1f9ed4",
    "qoil_a_bpd": "#00cc96",
    "qwat_a_bpd": "#00b5ad",
    "qgas_a_mmcfd": "#1ca9c9",
    "wlr_s_pct": "#bcbd22",
    "qgross_s_bpd": "#1f77b4",
    "gor_s_scf_stb": "#ff6692",
    "gvf_a_pct": "#b6e880",
    "choke_size_64": "#e377c2",
}

WELL_COLORS = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#e377c2", "#bcbd22", "#7f7f7f",
]

def feature_color(feature_name: str, fallback_index: int = 0) -> str:
    return FEATURE_COLORS.get(feature_name, WELL_COLORS[fallback_index % len(WELL_COLORS)])

def well_color(index: int) -> str:
    return WELL_COLORS[index % len(WELL_COLORS)]

def chart_title_from_data(df) -> str:
    if "well" not in df.columns or df.empty:
        return "Well Production Test"

    wells = [str(w) for w in df["well"].dropna().astype(str).unique()]
    wells = [w for w in wells if w and w.lower() != "unknown"]

    if len(wells) == 1:
        title = f"Well {wells[0]}"
    elif len(wells) > 1:
        shown = " vs ".join(wells[:4])
        suffix = "" if len(wells) <= 4 else f" +{len(wells) - 4} more"
        title = f"Well Comparison: {shown}{suffix}"
    else:
        title = "Well Production Test"

    if "datetime" in df.columns and df["datetime"].notna().any():
        start_dt = df["datetime"].min()
        end_dt = df["datetime"].max()
        if pd.notna(start_dt) and pd.notna(end_dt):
            if start_dt.date() == end_dt.date():
                title += f" | {start_dt:%d-%b-%Y}"
            elif start_dt.year == end_dt.year:
                title += f" | {start_dt:%d-%b} to {end_dt:%d-%b-%Y}"
            else:
                title += f" | {start_dt:%d-%b-%Y} to {end_dt:%d-%b-%Y}"
    return title


st.set_page_config(
    page_title="TMU Production Test Dashboard",
    page_icon="📈",
    layout="wide",
)

MIN_DATE_ALLOWED = pd.Timestamp("1900-01-01").date()
MAX_DATE_ALLOWED = pd.Timestamp("2100-12-31").date()


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


def compressed_axis_tick_kwargs(df, max_ticks_per_series=6):
    """Build readable real-date tick labels for compressed timelines."""
    if df.empty or "plot_x" not in df.columns or "datetime" not in df.columns:
        return {}

    tickvals = []
    ticktext = []
    group_col = "series_label" if "series_label" in df.columns else "well"

    for _, g in df.dropna(subset=["plot_x", "datetime"]).groupby(group_col, dropna=False):
        g = g.sort_values("plot_x").reset_index(drop=True)
        if g.empty:
            continue

        n = len(g)
        if n <= max_ticks_per_series:
            idxs = list(range(n))
        else:
            idxs = sorted(set([round(i * (n - 1) / (max_ticks_per_series - 1)) for i in range(max_ticks_per_series)]))

        for i in idxs:
            dt = pd.Timestamp(g.loc[i, "datetime"])
            tickvals.append(g.loc[i, "plot_x"])
            ticktext.append(dt.strftime("%d-%b-%Y<br>%H:%M"))

    return {"tickmode": "array", "tickvals": tickvals, "ticktext": ticktext}


def compressed_separator_positions(df):
    """Positions between compressed tests to visually separate them."""
    if df.empty or "plot_x" not in df.columns or "series_label" not in df.columns:
        return []

    ranges = []
    for label, g in df.dropna(subset=["plot_x"]).groupby("series_label", dropna=False):
        ranges.append((str(label), float(g["plot_x"].min()), float(g["plot_x"].max())))

    ranges = sorted(ranges, key=lambda x: x[1])
    separators = []
    for i in range(1, len(ranges)):
        prev_max = ranges[i - 1][2]
        next_min = ranges[i][1]
        if next_min > prev_max:
            separators.append((prev_max + next_min) / 2)

    return separators



st.markdown(
    """
    <style>
    /* Force readable UI text even when browser/app is in dark mode */
    .stApp {
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
        max-width: 100%;
        background-color: #ffffff !important;
        color: #111827 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #f8fafc !important;
        color: #111827 !important;
    }
    section[data-testid="stSidebar"] * {
        color: #111827 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 800 !important;
        letter-spacing: 0.01em;
    }
    h1 {
        font-size: 2.4rem !important;
    }
    h2 {
        font-size: 1.65rem !important;
    }
    h3 {
        font-size: 1.35rem !important;
    }
    p, label, span, div, .stMarkdown, .stCaption {
        color: #111827 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 850 !important;
        color: #0f172a !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 1.05rem !important;
        font-weight: 750 !important;
        color: #334155 !important;
    }
    div[data-baseweb="select"] *, div[data-baseweb="input"] *, textarea {
        color: #111827 !important;
    }
    button {
        font-weight: 700 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("TMU Production Test Dashboard")
st.caption(
    "Upload Excel / CSV / Word / PDF files, paste WhatsApp reports, clean the data, compare wells, "
    "and export clear production-test plots."
)

with st.sidebar:
    st.header("1) Upload data")
    uploaded_files = st.file_uploader(
        "Upload one or many files",
        type=["xlsx", "xls", "csv", "txt", "docx", "pdf"],
        accept_multiple_files=True,
    )

    st.header("2) Paste WhatsApp report")
    whatsapp_text = st.text_area(
        "Paste one or many TMU WhatsApp messages",
        height=260,
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

frames = []
errors = []

if uploaded_files:
    for f in uploaded_files:
        try:
            parsed_tables = load_tabular_file(f)
            if parsed_tables:
                frames.extend(parsed_tables)
            else:
                errors.append(f"{f.name}: no usable time-series table detected. The file may be blank, or it has no date/time plus numeric readings after all Excel fallback parsers were tried.")
        except Exception as e:
            errors.append(f"{f.name}: {e}")
            with st.expander(f"Technical error details for {f.name}", expanded=False):
                st.code(traceback.format_exc())

if whatsapp_text.strip():
    try:
        msg_df = parse_many_tmu_messages(whatsapp_text, source_name="Pasted_WhatsApp_Text")
        if not msg_df.empty:
            frames.append(msg_df)
        else:
            errors.append("WhatsApp text: no recognizable TMU report detected")
    except Exception as e:
        errors.append(f"WhatsApp text: {e}")

if errors:
    st.warning("Some files/messages were skipped or could not be parsed:\n\n" + "\n".join(f"- {e}" for e in errors))

if not frames:
    st.info("Upload files or paste a WhatsApp TMU message to start.")
    st.stop()

data = pd.concat(frames, ignore_index=True, sort=False)

if "datetime" in data.columns:
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
if "date" in data.columns:
    data["date"] = pd.to_datetime(data["date"], errors="coerce")


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
    for well, g in df.dropna(subset=["datetime"]).groupby("well", dropna=False):
        g = g.sort_values("datetime").set_index("datetime")
        res = g[numeric].resample(rule).mean().dropna(how="all")
        if res.empty:
            continue
        res["well"] = well
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


def add_plot_axis_columns(df, x_axis_mode, trace_grouping="Auto"):
    if df.empty:
        return df
    out = df.copy()

    def make_group_key(row):
        well = str(row.get("well", "Unknown"))
        source = short_test_label(row.get("source", ""))
        sheet = short_test_label(row.get("sheet", ""))
        test = sheet if sheet and sheet not in ["CSV", "Aggregated"] else source
        return f"{well}__{test}" if test else well

    out["series_group_key"] = out.apply(make_group_key, axis=1)

    group_meta = []
    for key, g in out.groupby("series_group_key", dropna=False):
        well = str(g["well"].dropna().iloc[0]) if "well" in g.columns and g["well"].notna().any() else str(key)
        if "datetime" in g.columns and g["datetime"].notna().any():
            date_label = pd.Timestamp(g["datetime"].min()).strftime("%d-%b-%Y")
        else:
            date_label = ""
        group_meta.append((key, well, date_label))

    well_counts = {}
    for _, well, _ in group_meta:
        well_counts[well] = well_counts.get(well, 0) + 1

    label_map = {key: (f"{well} ({date_label})" if well_counts.get(well, 0) > 1 and date_label else well)
                 for key, well, date_label in group_meta}

    out["series_label"] = out["series_group_key"].map(label_map)
    out["plot_x"] = None

    group_col = "series_group_key" if x_axis_mode.startswith("Compressed") else "series_label"
    groups = []
    for label, idx in out.groupby(group_col, dropna=False).groups.items():
        g = out.loc[idx]
        start_time = g["datetime"].min() if "datetime" in g.columns and g["datetime"].notna().any() else pd.Timestamp.max
        groups.append((str(label), idx, start_time))
    groups = sorted(groups, key=lambda item: item[2])

    current_offset = 0.0
    for label, idx, _ in groups:
        g = out.loc[idx].copy()
        sort_col = "datetime" if "datetime" in g.columns and g["datetime"].notna().any() else None
        if sort_col:
            g = g.sort_values(sort_col)

        if x_axis_mode == "Real calendar time":
            out.loc[g.index, "plot_x"] = g["datetime"] if sort_col else range(1, len(g) + 1)
        else:
            if sort_col:
                elapsed = (g["datetime"] - g["datetime"].min()).dt.total_seconds() / 3600.0
                out.loc[g.index, "plot_x"] = current_offset + elapsed
                span = float(elapsed.max()) if len(elapsed) else 0.0
                current_offset += max(span, max(len(g) - 1, 1) * 0.05) + max(span * 0.08, 1.0)
            else:
                seq = pd.Series(range(1, len(g) + 1), index=g.index, dtype=float)
                out.loc[g.index, "plot_x"] = current_offset + seq
                current_offset += float(len(g)) + 1.0
    return out


def x_axis_title_from_mode(x_axis_mode):
    if x_axis_mode.startswith("Compressed"):
        return "Compressed real-date timeline — empty gaps removed"
    return "Time"


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
    if x_axis_mode == "Real calendar time":
        for e in manual_events:
            target = e.get("target", "All selected wells")
            label = e["label"] if target == "All selected wells" else f"{target}: {e['label']}"
            converted.append({"plot_x": e["datetime"], "label": label, "target": target})
        return converted
    if "datetime" not in df.columns:
        return []
    multiple_series = df["series_label"].nunique() > 1
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
            label = f"{series_label}: {event['label']}" if multiple_series and target == "All selected wells" else event["label"]
            if target != "All selected wells":
                label = f"{target}: {event['label']}"
            converted.append({"plot_x": px, "label": label, "target": target})
    return converted


def convert_intervals_for_plot(operation_intervals, df, x_axis_mode):
    if not operation_intervals or df.empty or "series_label" not in df.columns:
        return []
    converted = []
    if x_axis_mode == "Real calendar time":
        for i in operation_intervals:
            target = i.get("target", "All selected wells")
            label = i["label"] if target == "All selected wells" else f"{target}: {i['label']}"
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
            label = f"{series_label}: {interval['label']}" if multiple_series and target == "All selected wells" else interval["label"]
            if target != "All selected wells":
                label = f"{target}: {interval['label']}"
            converted.append({"x0": x0, "x1": x1, "label": label, "target": target})
    return converted


numeric_cols = available_numeric_columns(data)
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
            "The column is still plottable, but add its header text to best_canonical_name() if you want a standard label."
        )

# Sidebar filters
with st.sidebar:
    st.header("3) Time scale")
    time_filter_mode = st.selectbox(
        "Time range control",
        ["Slider", "Manual calendar/time", "Full range"],
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
        ["Real calendar time", "Compressed real dates - remove empty gaps"],
        index=0,
        help=(
            "Real calendar time keeps true dates and gaps. "
            "Compressed real dates removes long empty gaps between tests, keeps real date/time labels, "
            "and adds dotted separators between test periods."
        ),
    )
    trace_grouping = "Auto"


    st.header("4) Filter and plot")

    all_wells = sorted([w for w in data["well"].dropna().astype(str).unique()]) if "well" in data.columns else []
    selected_wells = st.multiselect("Choose wells", all_wells, default=all_wells[:5] if all_wells else [])

    select_all_features = st.checkbox(
        "Select all numeric columns",
        value=False,
        help="Shows every detected numeric column in the plot list, including raw fallback columns from unseen templates.",
    )

    default_features = [
        c for c in [
            "gross_rate_bpd", "qgross_s_bpd", "oil_rate_stbd", "qoil_s_stbd",
            "water_rate_bpd", "qwat_s_bpd", "gas_rate_mmscfd", "gas_formation_mmscfd",
            "pumping_pressure_psi", "bsw_pct", "wlr_s_pct", "whp_psi",
            "flow_press_psi", "sep_p_psi", "salinity_kppm",
        ] if c in numeric_cols
    ] or numeric_cols[: min(6, len(numeric_cols))]

    selected_features = st.multiselect(
        "Choose features to plot",
        numeric_cols,
        default=numeric_cols if select_all_features else default_features,
        format_func=column_label,
    )

    custom_y_ranges = {}
    with st.expander("Y-axis scale per graph", expanded=False):
        use_custom_y_scale = st.checkbox(
            "Use custom Y-axis ranges",
            value=False,
            help="Set min/max for each selected graph, e.g. Gross Rate from 0 to 1000.",
        )

        if use_custom_y_scale and selected_features:
            for feature in selected_features:
                vals = pd.to_numeric(data[feature], errors="coerce").dropna() if feature in data.columns else pd.Series(dtype=float)
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
        ["No fill", "Forward fill", "Forward + backward fill", "Linear interpolation by row", "Time interpolation"],
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
        ["Separate panels like report", "Overlay normalized", "Overlay actual values"],
        index=0,
    )

    show_points = st.checkbox("Show markers", value=True)

    value_label_mode = st.selectbox(
        "Value labels on chart",
        [
            "Hourly + min/max - best for reports",
            "Auto sparse - recommended",
            "All values - use wide export",
            "Every 4 readings",
            "Every 8 readings",
            "First and last only",
            "Off",
        ],
        index=0,
        help=(
            "Dense plots cannot show every number clearly. "
            "Hourly + min/max gives a human-readable report view. "
            "Use All values only with single-feature PNG/PDF export."
        ),
    )

    st.caption(
        "Label modes: Hourly + min/max labels operationally important points; "
        "Auto sparse labels evenly spaced points to reduce overlap."
    )

    label_decimals = st.selectbox(
        "Number format on labels",
        ["Auto", "0 decimals", "1 decimal", "2 decimals"],
        index=0,
    )

    show_internal_names = False

    st.header("5) Graph events")

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
                {"datetime": start_dt_note, "label": note_label_input.strip(), "target": note_target}
            )
            st.success(f"Added event: {start_dt_note:%Y-%m-%d %H:%M} | {note_label_input.strip()}")

    if st.session_state.manual_events_table:
        st.caption("Current point notes")
        events_df_sidebar = pd.DataFrame(st.session_state.manual_events_table)
        events_df_sidebar["datetime"] = pd.to_datetime(events_df_sidebar["datetime"])
        events_df_sidebar = events_df_sidebar.sort_values("datetime").reset_index(drop=True)
        st.dataframe(
            events_df_sidebar.assign(datetime=events_df_sidebar["datetime"].dt.strftime("%Y-%m-%d %H:%M")),
            use_container_width=True,
            height=140,
        )

        remove_event_idx = st.number_input(
            "Remove point note number",
            min_value=1,
            max_value=len(st.session_state.manual_events_table),
            value=1,
            step=1,
        )

        ce1, ce2 = st.columns(2)
        with ce1:
            if st.button("Remove point note"):
                events_df_sidebar = events_df_sidebar.drop(index=remove_event_idx - 1).reset_index(drop=True)
                st.session_state.manual_events_table = events_df_sidebar.to_dict("records")
                st.rerun()
        with ce2:
            if st.button("Clear point notes"):
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
        st.dataframe(display_intervals, use_container_width=True, height=140)

        remove_interval_idx = st.number_input(
            "Remove interval note number",
            min_value=1,
            max_value=len(st.session_state.operation_intervals_table),
            value=1,
            step=1,
        )

        ci1, ci2 = st.columns(2)
        with ci1:
            if st.button("Remove interval note"):
                intervals_df_sidebar = intervals_df_sidebar.drop(index=remove_interval_idx - 1).reset_index(drop=True)
                st.session_state.operation_intervals_table = intervals_df_sidebar.to_dict("records")
                st.rerun()
        with ci2:
            if st.button("Clear interval notes"):
                st.session_state.operation_intervals_table = []
                st.rerun()


filtered = data.copy()
if selected_wells:
    filtered = filtered[filtered["well"].astype(str).isin(selected_wells)]

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
        manual_events.append({"datetime": pd.Timestamp(e["datetime"]), "label": str(e["label"]), "target": str(e.get("target", "All selected wells"))})
    except Exception:
        pass

# Prepare real-time / elapsed-time / sequence x-axis.
filtered = add_plot_axis_columns(filtered, x_axis_mode, trace_grouping)
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

# Header KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Detected rows", f"{len(data):,}")
c2.metric("Filtered rows", f"{len(filtered):,}")
c3.metric("Wells", f"{data['well'].nunique() if 'well' in data.columns else 0:,}")
c4.metric("Numeric features", f"{len(numeric_cols):,}")

with st.expander("Detected data preview", expanded=False):
    st.caption("Detected data = cleaned rows pulled from uploads before your well/time/feature filters.")
    preview_cols = ["source", "sheet", "well", "datetime", "time_text"] + numeric_cols
    preview_cols = [c for c in preview_cols if c in data.columns]
    display_detected = data[preview_cols].copy()
    if not show_internal_names:
        display_detected = display_detected.rename(columns={c: column_label(c) for c in display_detected.columns})
    st.dataframe(display_detected, use_container_width=True, height=260)

if selected_features and not filtered.empty:
    st.subheader("Interactive plot")
    if x_axis_mode == "Real calendar time":
        axis_tick_settings = x_axis_tick_kwargs(x_axis_scale)
    elif x_axis_mode.startswith("Compressed"):
        axis_tick_settings = compressed_axis_tick_kwargs(filtered)
    else:
        axis_tick_settings = {}
    x_axis_title = x_axis_title_from_mode(x_axis_mode)

    def add_compressed_test_separators_to_plotly(fig, features):
        if not x_axis_mode.startswith("Compressed"):
            return fig

        for x_sep in compressed_separator_positions(filtered):
            for r in range(1, len(features) + 1):
                try:
                    fig.add_vline(
                        x=x_sep,
                        line_width=1.8,
                        line_dash="dot",
                        line_color="#64748b",
                        opacity=0.75,
                        row=r,
                        col=1,
                    )
                except Exception:
                    pass
        return fig

    def add_operation_intervals_to_plotly(fig, features):
        fig = add_compressed_test_separators_to_plotly(fig, features)

        if not plot_intervals:
            return fig

        for interval in plot_intervals:
            x0 = interval["x0"]
            x1 = interval["x1"]
            label = interval["label"]

            # Draw only the interval start and end lines. No shaded background.
            for r in range(1, len(features) + 1):
                for x_val in [x0, x1]:
                    try:
                        fig.add_vline(
                            x=x_val,
                            line_width=2.2,
                            line_dash="dash",
                            line_color="#92400e",
                            opacity=0.90,
                            row=r,
                            col=1,
                        )
                    except Exception:
                        pass

            try:
                x_mid = x0 + (x1 - x0) / 2
            except Exception:
                x_mid = x0

            # Center the note between start/end and show left/right arrows.
            try:
                fig.add_annotation(
                    x=x_mid,
                    y=1.055,
                    xref="x",
                    yref="paper",
                    text=f"←  {label}  →",
                    showarrow=False,
                    bgcolor="rgba(255,255,255,0.96)",
                    bordercolor="#92400e",
                    borderwidth=1,
                    font=dict(size=16, color="#111827"),
                )
            except Exception:
                pass
        return fig

    def add_manual_events_to_plotly(fig, features):
        fig = add_operation_intervals_to_plotly(fig, features)

        if not plot_events:
            return fig
        for event in plot_events:
            x = event["plot_x"]
            label = event["label"]
            for r in range(1, len(features) + 1):
                try:
                    fig.add_vline(
                        x=x,
                        line_width=2,
                        line_dash="dash",
                        line_color="#111827",
                        opacity=0.75,
                        row=r,
                        col=1,
                    )
                    fig.add_annotation(
                        x=x,
                        y=1,
                        xref=f"x{r if r > 1 else ''}",
                        yref=f"y{r if r > 1 else ''} domain",
                        text=label,
                        showarrow=False,
                        xanchor="left",
                        yanchor="top",
                        font=dict(size=13, color="#111827"),
                        bgcolor="rgba(255,255,255,0.85)",
                        bordercolor="#111827",
                        borderwidth=1,
                    )
                except Exception:
                    pass
        return fig

    def x_values(df):
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
        if mode == "Off":
            return set()

        if n <= 0:
            return set()

        if mode == "All values - use wide export":
            return set(range(n))

        if mode == "Every 4 readings":
            return set(list(range(0, n, 4)) + [n - 1])

        if mode == "Every 8 readings":
            return set(list(range(0, n, 8)) + [n - 1])

        if mode == "First and last only":
            return {0, n - 1}

        # Auto sparse: evenly spaced labels with fewer collisions.
        if n <= 14:
            step = 1
        elif n <= 35:
            step = 3
        elif n <= 80:
            step = 5
        elif n <= 160:
            step = 8
        else:
            step = max(10, round(n / 22))

        return set(list(range(0, n, step)) + [n - 1])

    def format_plot_value(feature, value):
        if pd.isna(value):
            return ""

        v = float(value)

        if label_decimals == "0 decimals":
            return f"{v:.0f}"
        if label_decimals == "1 decimal":
            return f"{v:.1f}"
        if label_decimals == "2 decimals":
            return f"{v:.2f}"

        # Auto compact format to prevent label overlap.
        if feature in ["bsw_pct", "co2_mole_pct"]:
            txt = f"{v:.1f}"
        elif feature in ["salinity_kppm", "choke_pct", "whp_psi", "sep_p_psi", "pumping_pressure_psi"]:
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
        """Meaningful labels for dense field reports: first/last, hourly points, min/max, and zero/bypass values."""
        n = len(g)
        idxs = {0, n - 1} if n else set()

        if n == 0 or feature not in g.columns:
            return idxs

        y = pd.to_numeric(g[feature], errors="coerce")
        if y.notna().any():
            idxs.add(int(y.idxmin() if isinstance(y.index, pd.RangeIndex) else y.reset_index(drop=True).idxmin()))
            idxs.add(int(y.idxmax() if isinstance(y.index, pd.RangeIndex) else y.reset_index(drop=True).idxmax()))
            zero_positions = list(y.reset_index(drop=True)[y.reset_index(drop=True).abs() < 1e-12].index)
            idxs.update(zero_positions[:10])

        if "datetime" in g.columns and g["datetime"].notna().any():
            dt = pd.to_datetime(g["datetime"], errors="coerce")
            # Prefer full-hour labels. If no full hours exist, use half-hour labels.
            hourly = list(dt.reset_index(drop=True)[(dt.dt.minute == 0) & dt.notna()].index)
            if len(hourly) < 3:
                hourly = list(dt.reset_index(drop=True)[(dt.dt.minute.isin([0, 30])) & dt.notna()].index)
            idxs.update(hourly)
        else:
            idxs.update(label_indices(n, "Auto sparse - recommended"))

        # Safety limit: if the selected period is very long, keep labels readable.
        if len(idxs) > 45:
            idxs = set(sorted(idxs)[::max(1, len(idxs) // 45)])
            idxs.update({0, n - 1})

        return {i for i in idxs if 0 <= i < n}

    def build_text_and_positions(g, feature):
        if value_label_mode == "Hourly + min/max - best for reports":
            idxs = report_label_indices(g.reset_index(drop=True), feature)
        else:
            idxs = label_indices(len(g), value_label_mode)

        text = []
        pos = []
        for i, v in enumerate(g[feature]):
            text.append(format_plot_value(feature, v) if i in idxs else "")
            # Alternate labels above/below points to reduce collisions.
            pos.append("top center" if i % 2 == 0 else "bottom center")
        return text, pos

    def padded_range(df, feature):
        if feature in custom_y_ranges:
            return custom_y_ranges[feature]

        vals = pd.to_numeric(df[feature], errors="coerce").dropna()
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
                subplot_titles=[f"{i + 1}) {column_label(f)}" for i, f in enumerate(features)],
            )

            for row_idx, feature in enumerate(features, start=1):
                feature_data_for_range = []

                for series_label in series_values:
                    g = df[df["series_label"].astype(str) == series_label].sort_values(
                        "plot_x" if "plot_x" in df.columns else ("datetime" if "datetime" in df.columns else df.index.name)
                    )
                    if g.empty or feature not in g.columns:
                        continue

                    text, textposition = build_text_and_positions(g, feature)
                    feature_data_for_range.append(g[[feature]])

                    series_idx = series_values.index(series_label)
                    color = well_color(series_idx) if len(series_values) > 1 else feature_color(feature, series_idx)
                    fig.add_trace(
                        go.Scatter(
                            x=x_values(g),
                            y=g[feature],
                            mode=line_mode + ("+text" if value_label_mode != "Off" else ""),
                            text=text,
                            textposition=textposition,
                            textfont=dict(size=16, color=color, family="Arial, sans-serif"),
                            cliponaxis=False,
                            name=f"{series_label}",
                            legendgroup=str(series_label),
                            showlegend=(show_chart_legend and row_idx == 1),
                            line=dict(color=color, width=3.0),
                            marker=dict(color=color, size=8),
                        ),
                        row=row_idx,
                        col=1,
                    )

                y_range = padded_range(pd.concat(feature_data_for_range), feature) if feature_data_for_range else None
                fig.update_yaxes(
                    title_text=column_label(feature),
                    row=row_idx,
                    col=1,
                    automargin=True,
                    range=y_range,
                )

            n_points = max_points_per_trace(df)
            fig.update_layout(
                height=max(700, 360 * len(features)),
                width=max(1400, min(2600, n_points * 42)),
                title=dict(text=chart_title_from_data(df), font=dict(size=30, color="#0f172a", family="Arial Black, Arial, sans-serif")),
                hovermode="x unified",
                margin=dict(l=85, r=50, t=115, b=80),
                uniformtext_minsize=8,
                uniformtext_mode="hide",
                plot_bgcolor="white",
                paper_bgcolor="white",
                font=dict(color="#111827", size=15),
                legend=dict(
                    font=dict(size=17, color="#111827"),
                    bgcolor="rgba(255,255,255,0.90)",
                    bordercolor="#e5e7eb",
                    borderwidth=1,
                ),
                showlegend=show_chart_legend,
                title_x=0.5,
                title_xanchor="center",
            )
            # Show readable time ticks on EVERY subplot, not only the bottom one.
            for r in range(1, len(features) + 1):
                fig.update_xaxes(
                    row=r,
                    col=1,
                    showgrid=True,
                    gridcolor="#dddddd",
                    zeroline=False,
                    showticklabels=True,
                    tickfont=dict(size=16, color="#111827"),
                    title_text=x_axis_title if r == len(features) else "",
                    title_font=dict(size=20, color="#111827"),
                    automargin=True,
                    **axis_tick_settings,
                )
                fig.update_yaxes(
                    row=r,
                    col=1,
                    showgrid=True,
                    gridcolor="#eeeeee",
                    zeroline=False,
                    title_font=dict(size=17, color="#111827"),
                    tickfont=dict(size=14, color="#111827"),
                    automargin=True,
                )

            # Subplot titles are stored as annotations.
            for annotation in fig.layout.annotations:
                annotation.font = dict(size=21, color="#111827")
            fig = add_manual_events_to_plotly(fig, features)
            return fig

        fig = go.Figure()
        for feature in features:
            for series_label in series_values:
                g = df[df["series_label"].astype(str) == series_label].copy()
                if g.empty or feature not in g.columns:
                    continue

                y = g[feature].astype(float)
                plot_name = f"{series_label} - {column_label(feature)}"

                if mode == "Overlay normalized":
                    if y.notna().sum() and y.max() != y.min():
                        y = (y - y.min()) / (y.max() - y.min()) * 100
                    else:
                        y = y * 0
                    y_title = "Normalized scale, 0–100"
                else:
                    y_title = "Actual values"

                series_idx = series_values.index(series_label)
                color = well_color(series_idx) if mode != "Overlay actual values" else feature_color(feature, series_idx)
                fig.add_trace(
                    go.Scatter(
                        x=x_values(g),
                        y=y,
                        mode=line_mode,
                        name=plot_name,
                        line=dict(color=color, width=2.5),
                        marker=dict(color=color, size=7),
                    )
                )

        fig.update_layout(
            height=850,
            width=1700,
            title=dict(text=chart_title_from_data(df), font=dict(size=30, color="#0f172a", family="Arial Black, Arial, sans-serif")),
            yaxis_title=y_title,
            xaxis_title=x_axis_title,
            hovermode="x unified",
            margin=dict(l=85, r=50, t=115, b=80),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#111827", size=15),
            legend=dict(
                font=dict(size=17, color="#111827"),
                bgcolor="rgba(255,255,255,0.90)",
                bordercolor="#e5e7eb",
                borderwidth=1,
            ),
            title_x=0.5,
            title_xanchor="center",
        )
        fig.update_xaxes(
            showgrid=True,
            gridcolor="#dddddd",
            zeroline=False,
            title_font=dict(size=20, color="#111827"),
            tickfont=dict(size=15, color="#374151"),
            **axis_tick_settings,
        )
        fig.update_yaxes(
            showgrid=True,
            gridcolor="#eeeeee",
            zeroline=False,
            title_font=dict(size=20, color="#111827"),
            tickfont=dict(size=15, color="#374151"),
        )
        if manual_events:
            for event in manual_events:
                fig.add_vline(x=event["datetime"], line_dash="dash", line_color="#111827", line_width=2, opacity=0.75)
                fig.add_annotation(
                    x=event["datetime"],
                    y=1,
                    xref="x",
                    yref="paper",
                    text=event["label"],
                    showarrow=False,
                    xanchor="left",
                    yanchor="top",
                    font=dict(size=13, color="#111827"),
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="#111827",
                    borderwidth=1,
                )
        return fig

    fig = build_figure(filtered, selected_features, plot_mode)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Filtered data used by current plot", expanded=False):
        st.caption("Filtered data = only the rows currently feeding the chart after your sidebar selections.")
        filtered_cols = ["source", "sheet", "well", "datetime", "time_text"] + selected_features
        filtered_cols = [c for c in filtered_cols if c in filtered.columns]
        display_filtered = filtered[filtered_cols].copy()
        if not show_internal_names:
            display_filtered = display_filtered.rename(columns={c: column_label(c) for c in display_filtered.columns})
        st.dataframe(display_filtered, use_container_width=True, height=280)

    st.subheader("Chart downloads")

    def chart_label_indices_for_export(g, feature):
        """Label points that matter for a printed chart."""
        g2 = g.reset_index(drop=True)
        n = len(g2)
        idxs = {0, n - 1} if n else set()

        y = pd.to_numeric(g2[feature], errors="coerce")
        if y.notna().any():
            idxs.add(int(y.idxmin()))
            idxs.add(int(y.idxmax()))
            zero_positions = list(y[y.abs() < 1e-12].index)
            idxs.update(zero_positions[:12])

        if "datetime" in g2.columns and g2["datetime"].notna().any():
            dt = pd.to_datetime(g2["datetime"], errors="coerce")
            hourly = list(dt[(dt.dt.minute == 0) & dt.notna()].index)
            if len(hourly) < 4:
                hourly = list(dt[(dt.dt.minute.isin([0, 30])) & dt.notna()].index)
            idxs.update(hourly)
        else:
            idxs.update(label_indices(n, "Auto sparse - recommended"))

        if len(idxs) > 55:
            keep = sorted(idxs)
            step = max(1, len(keep) // 55)
            idxs = set(keep[::step])
            idxs.update({0, n - 1})

        return {i for i in idxs if 0 <= i < n}

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
                fig_m, ax = plt.subplots(figsize=(16.5, 9.3), dpi=160)
                title = f"{chart_title_from_data(df)}\\n{column_label(feature)}"
                ax.set_title(title, fontsize=22, fontweight="bold", pad=18)

                for wi, series_label in enumerate(series_values):
                    g = df[df["series_label"].astype(str) == series_label].sort_values(
                        "plot_x" if "plot_x" in df.columns else ("datetime" if "datetime" in df.columns else df.index.name)
                    ).reset_index(drop=True)
                    if g.empty or feature not in g.columns:
                        continue

                    x = g["plot_x"] if "plot_x" in g.columns and g["plot_x"].notna().any() else (
                        pd.to_datetime(g["datetime"], errors="coerce") if "datetime" in g.columns and g["datetime"].notna().any() else pd.Series(range(len(g)))
                    )
                    y = pd.to_numeric(g[feature], errors="coerce")
                    color = feature_color(feature, wi)

                    ax.plot(
                        x,
                        y,
                        marker="o",
                        markersize=4.8,
                        linewidth=2.4,
                        color=color,
                        label=series_label if len(series_values) > 1 else None,
                    )

                    idxs = chart_label_indices_for_export(g, feature)
                    for i in sorted(idxs):
                        if i >= len(g) or pd.isna(y.iloc[i]):
                            continue
                        ax.annotate(
                            format_plot_value(feature, y.iloc[i]),
                            (x.iloc[i], y.iloc[i]),
                            textcoords="offset points",
                            xytext=(0, 10 if i % 2 == 0 else -15),
                            ha="center",
                            fontsize=10.5,
                            color=color,
                            fontweight="bold",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
                        )

                if feature in custom_y_ranges:
                    ax.set_ylim(custom_y_ranges[feature][0], custom_y_ranges[feature][1])
                else:
                    vals = pd.to_numeric(df[feature], errors="coerce").dropna()
                    if not vals.empty:
                        ymin = float(vals.min())
                        ymax = float(vals.max())
                        pad = max((ymax - ymin) * 0.18, max(abs(ymax), 1) * 0.03, 0.5)
                        ax.set_ylim(ymin - pad, ymax + pad)

                ax.set_ylabel(column_label(feature), fontsize=15, fontweight="bold")
                ax.set_xlabel(x_axis_title, fontsize=15, fontweight="bold")

                if plot_intervals:
                    ymin_i, ymax_i = ax.get_ylim()
                    y_span = ymax_i - ymin_i if ymax_i != ymin_i else 1.0
                    y_note = ymax_i - 0.04 * y_span
                    for interval in plot_intervals:
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
                            fontsize=11,
                            fontweight="bold",
                            ha="center",
                            va="center",
                            color="#111827",
                            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#92400e", alpha=0.95),
                        )

                if plot_events:
                    for event in plot_events:
                        ax.axvline(event["plot_x"], color="#111827", linestyle="--", linewidth=1.6, alpha=0.75)
                        ax.text(
                            event["plot_x"],
                            0.98,
                            event["label"],
                            transform=ax.get_xaxis_transform(),
                            rotation=90,
                            va="top",
                            ha="right",
                            fontsize=10,
                            color="#111827",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#111827", alpha=0.75),
                        )
                ax.grid(True, which="major", alpha=0.28)
                ax.tick_params(axis="both", labelsize=12)

                if x_axis_mode == "Real calendar time" and "datetime" in df.columns and df["datetime"].notna().any():
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=16))
                    fig_m.autofmt_xdate(rotation=0)
                elif x_axis_mode.startswith("Compressed"):
                    tick_settings = compressed_axis_tick_kwargs(df)
                    if tick_settings:
                        ax.set_xticks(tick_settings.get("tickvals", []))
                        ax.set_xticklabels([str(t).replace("<br>", "\n") for t in tick_settings.get("ticktext", [])], rotation=0)
                    for x_sep in compressed_separator_positions(df):
                        ax.axvline(x_sep, color="#64748b", linestyle=":", linewidth=1.5, alpha=0.75)

                if len(series_values) > 1:
                    ax.legend(fontsize=12, loc="best")

                fig_m.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])
                pdf.savefig(fig_m)
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
                ax.set_title(f"{chart_title_from_data(df)}\\n{column_label(feature)}", fontsize=24, fontweight="bold", pad=18)

                for wi, series_label in enumerate(series_values):
                    g = df[df["series_label"].astype(str) == series_label].sort_values(
                        "plot_x" if "plot_x" in df.columns else ("datetime" if "datetime" in df.columns else df.index.name)
                    ).reset_index(drop=True)
                    if g.empty or feature not in g.columns:
                        continue

                    x = g["plot_x"] if "plot_x" in g.columns and g["plot_x"].notna().any() else (
                        pd.to_datetime(g["datetime"], errors="coerce") if "datetime" in g.columns and g["datetime"].notna().any() else pd.Series(range(len(g)))
                    )
                    y = pd.to_numeric(g[feature], errors="coerce")
                    color = feature_color(feature, wi)

                    ax.plot(x, y, marker="o", markersize=5, linewidth=2.6, color=color, label=series_label if len(series_values) > 1 else None)

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
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.7),
                        )

                if feature in custom_y_ranges:
                    ax.set_ylim(custom_y_ranges[feature][0], custom_y_ranges[feature][1])
                else:
                    vals = pd.to_numeric(df[feature], errors="coerce").dropna()
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
                    for interval in plot_intervals:
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
                            color="#111827",
                            bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="#92400e", alpha=0.95),
                        )

                if plot_events:
                    for event in plot_events:
                        ax.axvline(event["plot_x"], color="#111827", linestyle="--", linewidth=1.8, alpha=0.75)
                        ax.text(
                            event["plot_x"],
                            0.98,
                            event["label"],
                            transform=ax.get_xaxis_transform(),
                            rotation=90,
                            va="top",
                            ha="right",
                            fontsize=11,
                            color="#111827",
                            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="#111827", alpha=0.75),
                        )
                ax.grid(True, which="major", alpha=0.28)
                ax.tick_params(axis="both", labelsize=13)

                if x_axis_mode == "Real calendar time" and "datetime" in df.columns and df["datetime"].notna().any():
                    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
                    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=8, maxticks=16))
                    fig_m.autofmt_xdate(rotation=0)
                elif x_axis_mode.startswith("Compressed"):
                    tick_settings = compressed_axis_tick_kwargs(df)
                    if tick_settings:
                        ax.set_xticks(tick_settings.get("tickvals", []))
                        ax.set_xticklabels([str(t).replace("<br>", "\n") for t in tick_settings.get("ticktext", [])], rotation=0)
                    for x_sep in compressed_separator_positions(df):
                        ax.axvline(x_sep, color="#64748b", linestyle=":", linewidth=1.5, alpha=0.75)

                if len(series_values) > 1:
                    ax.legend(fontsize=12, loc="best")

                fig_m.tight_layout(rect=[0.02, 0.02, 0.98, 0.94])

                png_buffer = io.BytesIO()
                fig_m.savefig(png_buffer, format="png", dpi=190)
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


    def matplotlib_overview_export_bytes(df, features, fmt="png"):
        """Kaleido-free fallback for PNG/PDF export.

        It creates a static report-style chart directly with Matplotlib, so the
        dashboard can still export PNG/PDF on machines where Plotly Kaleido or
        Chrome is not installed.
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        if df.empty or not features:
            raise ValueError("No filtered data/features available for export.")

        series_values = sorted(df["series_label"].dropna().astype(str).unique()) if "series_label" in df.columns else (
            sorted(df["well"].dropna().astype(str).unique()) if "well" in df.columns else ["All"]
        )

        n_features = max(1, len(features))
        n_points = max_points_per_trace(df)
        width_in = max(16.0, min(30.0, n_points * 0.20))
        height_in = max(7.0, 3.9 * n_features)
        fig_m, axes = plt.subplots(n_features, 1, figsize=(width_in, height_in), dpi=170, squeeze=False)
        axes = axes.flatten()
        fig_m.suptitle(chart_title_from_data(df), fontsize=20, fontweight="bold", y=0.995)

        for ax, feature in zip(axes, features):
            for wi, series_label in enumerate(series_values):
                group_col = "series_label" if "series_label" in df.columns else "well"
                g = df[df[group_col].astype(str) == series_label].copy()
                if g.empty or feature not in g.columns:
                    continue

                sort_col = "plot_x" if "plot_x" in g.columns else ("datetime" if "datetime" in g.columns else None)
                if sort_col:
                    g = g.sort_values(sort_col)
                else:
                    g = g.reset_index(drop=True)

                y = pd.to_numeric(g[feature], errors="coerce")
                if y.notna().sum() == 0:
                    continue

                if x_axis_mode == "Real calendar time" and "datetime" in g.columns and g["datetime"].notna().any():
                    x = pd.to_datetime(g["datetime"], errors="coerce")
                elif "plot_x" in g.columns and g["plot_x"].notna().any():
                    x = pd.to_numeric(g["plot_x"], errors="coerce")
                else:
                    x = pd.Series(range(1, len(g) + 1), index=g.index)

                color = feature_color(feature, wi)
                ax.plot(x, y, marker="o" if show_points else None, markersize=4, linewidth=2.1, color=color, label=series_label if len(series_values) > 1 else None)

            if feature in custom_y_ranges:
                ax.set_ylim(custom_y_ranges[feature][0], custom_y_ranges[feature][1])
            ax.set_ylabel(column_label(feature), fontsize=12, fontweight="bold")
            ax.grid(True, which="major", alpha=0.28)
            ax.tick_params(axis="both", labelsize=10)

            if x_axis_mode == "Real calendar time" and "datetime" in df.columns and df["datetime"].notna().any():
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%d-%b\n%H:%M"))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=12))
            elif x_axis_mode.startswith("Compressed"):
                tick_settings = compressed_axis_tick_kwargs(df)
                if tick_settings:
                    ax.set_xticks(tick_settings.get("tickvals", []))
                    ax.set_xticklabels([str(t).replace("<br>", "\n") for t in tick_settings.get("ticktext", [])], rotation=0)
                for x_sep in compressed_separator_positions(df):
                    ax.axvline(x_sep, color="#64748b", linestyle=":", linewidth=1.3, alpha=0.75)

            if len(series_values) > 1:
                ax.legend(fontsize=9, loc="best")

        axes[-1].set_xlabel(x_axis_title_from_mode(x_axis_mode), fontsize=12, fontweight="bold")
        fig_m.tight_layout(rect=[0.02, 0.02, 0.98, 0.975])

        output = io.BytesIO()
        if fmt == "pdf":
            fig_m.savefig(output, format="pdf", bbox_inches="tight")
        else:
            fig_m.savefig(output, format="png", dpi=190, bbox_inches="tight")
        plt.close(fig_m)
        output.seek(0)
        return output.getvalue()

    # Keep only chart export buttons in the main view. The cleaned Excel and filtered CSV buttons were removed.
    d3, d4 = st.columns(2)

    # Wide export prevents label overlap when there are many readings.
    n_export_points = max_points_per_trace(filtered)
    if value_label_mode == "All values - use wide export":
        export_width = max(2200, min(4200, n_export_points * 55))
    else:
        export_width = max(2000, min(3600, n_export_points * 42))

    export_height = max(1000, 390 * len(selected_features))

    png_bytes, png_error = plotly_static_image_bytes(
        fig, "png", width=export_width, height=export_height, scale=2
    )
    png_label = "Download readable chart PNG"
    if png_bytes is None:
        try:
            png_bytes = matplotlib_overview_export_bytes(filtered, selected_features, fmt="png")
            png_label = "Download readable chart PNG"
            with d3:
                st.caption("Plotly/Kaleido PNG export was unavailable, so a Matplotlib PNG fallback was used.")
        except Exception as fallback_error:
            with d3:
                st.error("PNG export failed.")
                st.caption(f"Plotly/Kaleido error: {png_error}")
                st.caption(f"Matplotlib fallback error: {fallback_error}")

    if png_bytes is not None:
        with d3:
            st.download_button(
                png_label,
                data=png_bytes,
                file_name="tmu_chart_readable.png",
                mime="image/png",
            )

    pdf_bytes, pdf_error = plotly_static_image_bytes(
        fig, "pdf", width=export_width, height=export_height
    )
    pdf_label = "Download readable chart PDF"
    if pdf_bytes is None:
        try:
            pdf_bytes = matplotlib_overview_export_bytes(filtered, selected_features, fmt="pdf")
            pdf_label = "Download readable chart PDF"
            with d4:
                st.caption("Plotly/Kaleido PDF export was unavailable, so a Matplotlib PDF fallback was used.")
        except Exception as fallback_error:
            with d4:
                st.error("PDF export failed.")
                st.caption(f"Plotly/Kaleido error: {pdf_error}")
                st.caption(f"Matplotlib fallback error: {fallback_error}")

    if pdf_bytes is not None:
        with d4:
            st.download_button(
                pdf_label,
                data=pdf_bytes,
                file_name="tmu_chart_readable.pdf",
                mime="application/pdf",
            )

    st.markdown("#### Human-readable report exports")
    st.caption(
        "Use these when the overview PNG/PDF is still too crowded. "
        "They create one large chart per selected feature, so text and values are readable."
    )

    hr1, hr2 = st.columns(2)
    try:
        with hr1:
            st.download_button(
                "Download multi-page readable PDF",
                data=human_readable_pdf_bytes(filtered, selected_features),
                file_name="tmu_human_readable_report.pdf",
                mime="application/pdf",
            )
    except Exception as e:
        with hr1:
            st.caption(f"Human-readable PDF export error: {e}")

    try:
        with hr2:
            st.download_button(
                "Download readable PNGs ZIP",
                data=human_readable_png_zip_bytes(filtered, selected_features),
                file_name="tmu_readable_png_charts.zip",
                mime="application/zip",
            )
    except Exception as e:
        with hr2:
            st.caption(f"Readable PNG ZIP export error: {e}")
else:
    st.warning("Choose at least one feature to plot, and make sure the filters leave some rows.")