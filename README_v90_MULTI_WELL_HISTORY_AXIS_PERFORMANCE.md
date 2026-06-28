# Production Test Dashboard v90

## Multi-well, production-history, and performance corrections

### 1. Production History Plotly crash

The production-history report view passed `showticklabels` twice to Plotly:

- once explicitly in the subplot call;
- once inside the generated history tick dictionary.

Plotly rejected the duplicated keyword. v90 removes the duplicated setting and merges all axis options before calling `update_xaxes`, so the same error cannot return through another tick mode.

### 2. Chart title follows selected wells

The previous chart-title text input reused one Streamlit state key for every well selection. A title created while one well was selected could therefore remain after switching to another well.

v90 resets the automatic title only when the analysis view or selected wells/tests change:

- one well: `Well <name>`;
- multiple wells: `Well comparison: <well 1> vs <well 2>`;
- manual title edits remain while other chart settings change.

### 3. X-axis scale now works in compressed-date mode

The old compressed timeline ignored `X-axis tick scale`; it always generated ticks from a fixed density rule. v90 uses the selected interval (30 minutes, 1 hour, 12 hours, 1 day, etc.) to choose candidate real dates, then removes labels that would overlap after long calendar gaps are compressed.

Additional protections:

- a maximum of eight visible labels for multi-well comparisons;
- angled labels when full date/time text is needed;
- first and last periods retained;
- Plotly `uirevision` changes when the selected scale, wells, view, or display mode changes, forcing the browser to apply the new axis instead of retaining an older zoom/tick state.

### 4. Faster and more stable interactive charts

- Browser payload target reduced from 12,000 to 8,000 points.
- Per-series interactive target reduced from 3,000 to 2,000 points.
- Complete source data remains available to tables and exports.
- Multi-well and dense charts use `closest` hover instead of the heavier unified hover mode.
- Fixed 1,700–3,200 px interactive figure widths were removed; charts now resize to the browser container.
- Tick generation is capped and linear-time.
- Existing session-local upload parsing cache remains unchanged, so changing wells, labels, themes, or axes does not reparse uploaded files.

### Deployment

Replace the complete repository with the v90 package and reboot the Streamlit app. The startup log should show:

```text
v90-multiwell-history-axis-performance-fix-20260628
```

The parser remains the validated v89 parser because this release changes plotting and interface state, not ingestion logic.
