from __future__ import annotations

import io
import py_compile
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

for name in [
    "app.py", "tmu_parser.py", "tmu_parser_compat.py", "tmu_parser_legacy.py",
    "smart_tabular_v75.py", "history_analysis.py", "legacy_dashboard_pdf.py",
]:
    py_compile.compile(str(ROOT / name), doraise=True)

APP = (ROOT / "app.py").read_text(encoding="utf-8")
PARSER = (ROOT / "tmu_parser.py").read_text(encoding="utf-8")
assert 'APP_UI_BUILD_ID = "v98-legacy-vector-pdf-recovery-20260702"' in APP
assert 'PARSER_BUILD_ID = "v98-legacy-vector-pdf-recovery-20260702"' in PARSER
assert "apply_legacy_dashboard_state_to_session" in APP
assert "legacy_dashboard_state" in PARSER

# Build a small pre-v97-style vector dashboard PDF. No embedded attachment is
# used, so recovery must come from paths/ticks/annotations only.
import matplotlib.pyplot as plt

x = np.array([0.0, 0.5, 1.0, 1.5, 2.25, 2.75, 3.25, 3.75])
gas = np.array([3.9, 3.3, 2.9, 2.75, 2.6, 2.4, 2.2, 2.0])
gross = np.array([95.0, 142.0, 151.5, 130.5, 126.2, 134.6, 143.1, 126.3])
labels = [
    "30-Jun-2026\n19:00", "30-Jun-2026\n20:00", "30-Jun-2026\n20:30",
    "01-Jul-2026\n13:00", "01-Jul-2026\n13:30", "01-Jul-2026\n14:30",
]
ticks = [0.0, 1.0, 1.5, 2.25, 2.75, 3.75]

fig, axes = plt.subplots(2, 1, figsize=(16, 9), squeeze=False)
axes = axes.flatten()
fig.patch.set_facecolor("#0C1B25")
fig.suptitle("Well TEST-1", color="#EAF2F7", fontweight="bold")
for ax, y, ylabel, color, ylim in [
    (axes[0], gas, "Total Gas Rate (MMSCF/D)", "#00ACC1", (0, 5)),
    (axes[1], gross, "Gross Rate (BBL/D)", "#607D8B", (0, 200)),
]:
    ax.set_facecolor("#102331")
    ax.plot(x[:4], y[:4], marker="o", linewidth=2.4, color=color)
    ax.plot(x[4:], y[4:], marker="o", linewidth=2.4, color=color)
    ax.set_ylabel(ylabel, color="#EAF2F7", fontweight="bold")
    ax.set_ylim(*ylim)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=30, ha="right", color="#EAF2F7")
    ax.tick_params(colors="#EAF2F7")
    ax.grid(True, color="#3A5360", linewidth=0.75)
    for spine in ax.spines.values():
        spine.set_color("#3A5360")
    ax.axvline(0.0, color="#66D9EF", linestyle="--", linewidth=1.5)
    ax.text(0.0, 0.52, "SIWHP 3300 PSI", rotation=90, transform=ax.get_xaxis_transform(),
            color="#66D9EF", ha="right", va="center", fontweight="bold")
    ax.axvline(2.25, color="#FF8A72", linestyle="--", linewidth=1.5)
    ax.text(2.25, 0.52, "SIWHP 3100 PSI", rotation=90, transform=ax.get_xaxis_transform(),
            color="#FF8A72", ha="right", va="center", fontweight="bold")
    ax.axvline(1.5, color="#FFD166", linestyle="--", linewidth=1.8)
    ax.axvline(2.25, color="#FFD166", linestyle="--", linewidth=1.8)
    ax.annotate("", xy=(2.22, 0.96), xytext=(1.53, 0.96),
                xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                arrowprops=dict(arrowstyle="<->", color="#FFD166", lw=1.7))
    ax.text(1.875, 0.975, "Closed the Well", transform=ax.get_xaxis_transform(),
            color="#FFD166", ha="center", va="bottom", fontweight="bold")
axes[-1].set_xlabel("Compressed real-date timeline - empty gaps removed", color="#EAF2F7")
fig.tight_layout(rect=[0.02, 0.03, 0.98, 0.96])

pdf_buffer = io.BytesIO()
fig.savefig(pdf_buffer, format="pdf", facecolor=fig.get_facecolor(), edgecolor=fig.get_facecolor())
plt.close(fig)

from legacy_dashboard_pdf import recover_legacy_dashboard_pdf
recovered = recover_legacy_dashboard_pdf(pdf_buffer.getvalue(), "synthetic_legacy.pdf")
assert recovered is not None
assert recovered["theme"] == "Dark"
assert recovered["well"] == "TEST-1"
assert recovered["selected_features"] == ["gas_rate_mmscfd", "gross_rate_bpd"]
assert len(recovered["data"]) == 8
assert recovered["data"]["datetime"].min() == pd.Timestamp("2026-06-30 19:00")
assert recovered["data"]["datetime"].max() == pd.Timestamp("2026-07-01 14:30")
assert np.allclose(recovered["data"]["gas_rate_mmscfd"], gas, atol=0.002)
assert np.allclose(recovered["data"]["gross_rate_bpd"], gross, atol=0.05)
assert {e["label"] for e in recovered["manual_events"]} == {"SIWHP 3300 PSI", "SIWHP 3100 PSI"}
assert len(recovered["operation_intervals"]) == 1
assert recovered["operation_intervals"][0]["label"] == "Closed the Well"

# Facade regression: the normal parser must no longer raise the old
# "No usable time-series table" error for the same legacy vector PDF.
import tmu_parser
class Uploaded(io.BytesIO):
    def __init__(self, payload: bytes, name: str):
        super().__init__(payload)
        self.name = name

tables = tmu_parser.load_tabular_file(Uploaded(pdf_buffer.getvalue(), "synthetic_legacy.pdf"))
assert len(tables) == 1 and len(tables[0]) == 8
assert tables[0].attrs.get("legacy_dashboard_state", {}).get("theme") == "Dark"

print("PASS - v98 legacy vector PDF recovery")
