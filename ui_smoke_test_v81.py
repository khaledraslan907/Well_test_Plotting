"""Static validation for v81 simplified controls and stability protections."""
from pathlib import Path
import ast

root = Path(__file__).resolve().parent
source = (root / "app.py").read_text(encoding="utf-8")
ast.parse(source)

required = [
    'APP_UI_BUILD_ID = "v81-simple-controls-stability-ui-20260627"',
    '"New test after inactive gap (hours)"',
    'with st.sidebar.expander("2. Processing"',
    'with st.sidebar.expander("4. Timeline & Range"',
    'with st.sidebar.expander("5. Wells & Signals"',
    'with st.sidebar.expander("6. Chart Options"',
    'def _clear_heavy_session_state_v81',
    'def optimize_interactive_plot_frame',
    'def limited_dataframe_preview',
    'def render_plotly_safely',
    'width="stretch"',
]
missing = [fragment for fragment in required if fragment not in source]
if missing:
    raise SystemExit("Missing v81 fragments: " + ", ".join(missing))

for forbidden in [
    "How should tests be separated?",
    "Use parser-detected boundaries",
    "Keep each well as one continuous test",
    "Test segmentation controls when consecutive readings",
    "@st.cache_data",
    "use_container_width",
]:
    if forbidden in source:
        raise SystemExit(f"Obsolete v81 fragment still present: {forbidden}")

config = (root / ".streamlit" / "config.toml").read_text(encoding="utf-8")
if "maxMessageSize = 200" not in config:
    raise SystemExit("Missing Streamlit maxMessageSize protection")

print("Production Test Dashboard v81 UI/stability smoke test: PASS")
