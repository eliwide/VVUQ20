# VVUQ Validation Plotter — Code Audit

**Date**: 2026-02-27
**Reviewed against**: `C:\CLD\Quality-Assurance\code-review-checklist.md`

---

## Bugs Found & Fixed

| # | Severity | File(s) | Issue | Fix |
|---|----------|---------|-------|-----|
| 1 | HIGH | `export.py` | Single-chart "Export PNG" renders white E markers invisible on white background — `_apply_light_theme` never touched scatter `PathCollection` colours | Save/restore scatter face+edge colours; `_apply_light_theme` now detects near-white (R,G,B > 0.9) scatter markers and converts to near-black |
| 2 | HIGH | `gui_config_panel.py` | `bias_included` checkbox always returns `True`/`False`, never `None` — bias subtitle shown on every render with no off-state | Replaced checkbox with 3-state `QComboBox`: "(none)" / "Includes additional bias" / "Does not include additional bias" mapping to `None`/`True`/`False` |
| 3 | MEDIUM | `chart_dominant_source.py` | "Percentage of total uncertainty" used linear fraction (`u_i / sum`) instead of V&V 20 variance fraction (`u_i^2 / U_val^2`) — physically misleading | Switched to variance fraction; updated chart subtitle to show `u_i^2 / U_val^2` |
| 4 | MEDIUM | `csv_parser.py` | `_smart_split_line` split on raw commas without CSV quoting support — condition labels containing commas silently corrupt data | Replaced raw split with `csv.reader()` for proper quoting/escaping support; `_detect_delimiter()` still does tab/semicolon/comma priority |
| 5 | MEDIUM | `chart_stacked_band.py` | Bias subtitle at `y=1.0, va='top'` in axes transform extends downward into chart data area, overlapping bars | Replaced `ax.set_title` with `ax.text` title hierarchy: category at `y=1.16`, bias at `y=1.09`, TC labels at `y=1.02`; `fig.subplots_adjust(top=0.84)` after `tight_layout` leaves room |
| 6 | LOW | `chart_heatmap.py`, `chart_dominant_source.py` | `sensors.index(pt.sensor)` is O(N) per point — O(N^2 M) total for N sensors, M conditions | Pre-built `sensor_idx = {s: i for i, s in enumerate(sensors)}` dict for O(1) lookups |
| 7 | LOW | `csv_parser.py` | `import math` inside `_locale_float()` body — re-imported on every call | Moved to module-level import |

---

## Checklist Walk-Through

### 1. State & Repeated Invocation
- All chart renderers call `fig.clf()` at top — no stale state accumulation
- `_load_from_folder` clears ALL paths before re-populating (GPT #3 fix from prior session)
- Category combo rebuilt with `blockSignals(True)` and selection preservation
- `_dataset` replaced atomically on reload; cleared to `None` on failure

### 2. User Workflow Tracing
- Load > Generate > Load again > Generate: Works (dataset replaced)
- Cancel file dialog: Returns empty string, guarded
- Browse individual file after folder load: Works, updates single entry, triggers `_try_load`
- Load example then browse one replacement file: Paths are independent, works correctly
- `_try_load` early-return disables Generate/Export buttons (GPT #7 fix)

### 3. Boundary Conditions
- Empty CSV: `ValueError` with clear message
- Single row: "at least header + one data row" error
- Single sensor, single condition: `n_total=1`, renders correctly
- `condition_code()` guards negative index
- `_render_condition_key()` guards `n == 0`
- `n_conditions == 0`: `n_total = 0`, early-return with "No valid data" message

### 4. Data Integrity
- Cross-validation enforces all 5 CSVs share identical headers and row keys
- `has_data` flag is `True` only when ALL five values are non-None
- Partial load failure clears dataset and disables buttons
- Duplicate CSV rows warned and skipped (GPT #1 fix from prior session)

### 5. GUI-Specific (PySide6 / Qt)
- Category combo: `blockSignals(True)` during rebuild, selection restored
- No output tables in this tool (charts are matplotlib)
- Signals connected once in `_connect_signals()` — no duplicate connections
- All `clicked`/`toggled`/`triggered` signals use `lambda *_:` wrappers to absorb args
- Bias combo mapped to `{0: None, 1: True, 2: False}` — clean 3-state semantics

### 6. Arithmetic & Units
- Division by zero guarded: margin ratio (`e_abs < 1e-12`), heatmap ratio (`u_val > 0`), dominant source (`u_val_sq > 0`)
- `abs()` on all uncertainty values in stacked band
- k-factor correctly applied: `k = 1.0 if already_expanded else k_factor`
- Dominant source: variance fraction `u_i^2 / U_val^2` consistent with RSS decomposition
- Histogram uses `|E| - U_val` consistent with V&V 20 `|E| <= U_val` criterion

### 7. Integration Points
- All call sites for ALL 6 renderers pass `bias_included` (gui_chart_tabs.py — both live and batch)
- `get_config()` returns `None`/`True`/`False` for bias — all renderers handle all three states
- Export save/restore now covers: figure, axes, titles, labels, spines, ticks, grid, text, **scatter**, legend
- Font sizes standardised: titles at 9pt (stacked band 10pt), axis labels 8pt, legends 6pt

### 8. Locale & International Input
- `_locale_float()` used exclusively — no bare `float()` on CSV/user data
- `_smart_split_line()` uses `csv.reader` with auto-detected delimiter
- Tab > semicolon > comma priority for European CSV compatibility
- Quoted fields (commas inside field values) now handled correctly

### 10. NaN / Inf Propagation
- `_locale_float()` rejects `inf`/`nan` via `math.isfinite()`
- `np.isfinite()` guards in every chart renderer before computation
- `has_data` flag as primary entry-point guard
- No formatting of raw float values in reports/HTML (charts only)

---

## Not Applicable

- **Section 9 (Project File Round-Trip)**: No save/load project feature in v1.0
- **Section 5 (QTableWidget)**: No data tables in this tool

---

## Remaining Design Notes

These are not bugs but noted for future consideration:

1. **No progress indication for batch export** — GUI freezes during 24-chart export at 600 DPI. A `QProgressDialog` or `QApplication.processEvents()` between figures would improve UX.

2. **No keyboard shortcuts** — Consider `Ctrl+G` (Generate), `Ctrl+E` (Export Current), `Ctrl+Shift+E` (Export All), `Ctrl+L` (Load Folder).

3. **Example data writer** uses `','.join()` instead of `csv.writer` — acceptable for controlled example data with no commas in labels, but fragile if copied for other purposes.

4. **Scatter chart `color_by` parameter** exposed but not connected to any GUI control — always defaults to `"sensor"`. Could add a toggle for `"condition"` colouring.

5. **Stacked band uses `fig.subplots_adjust(top=0.84)` after `tight_layout`** — this is the standard matplotlib pattern for making room above axes ([tight_layout guide](https://matplotlib.org/stable/users/explain/axes/tight_layout_guide.html)), but a future upgrade to `constrained_layout` would handle title space automatically without manual `subplots_adjust`. Constrained layout is the [recommended approach](https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html) for new matplotlib figures.
