"""
Stacked bar chart for the VVUQ Validation Plotter.

The centerpiece chart: decomposes the comparison error E into
stacked uncertainty contributions (u_num, u_input, u_d) using
vertical bars, showing how each source consumes validation margin.

Bar structure for each point i (under-prediction concern):
    Bar top:           E_i
    u_num band:        E_i  →  E_i − u_num_i
    u_input band:      E_i − u_num_i  →  E_i − u_num_i − u_input_i
    u_d band:          …  →  E_i − U_val_i

For "both" concern, bars are drawn symmetrically above AND below E.

Features:
- TC group separators (dashed lines + labels)
- Condition codes shown under first TC group only (adaptive thinning)
- Clean condition key table below the chart
- Gap markers for missing data
- Prominent E = 0 line and optional validation requirement line
- Legend placed outside the chart area
"""

import warnings

import numpy as np
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from .constants import (
    BOEING_PALETTE, condition_code, DEFAULT_NO_DATA_LABEL,
    DARK_COLORS, EXPORT_TEXT_COLOR,
)
from .data_model import CategoryData


def render_stacked_band(
    fig: Figure,
    category_data: CategoryData,
    *,
    k_factor: float = 1.0,
    already_expanded: bool = True,
    concern_direction: str = "both",
    no_data_label: str = DEFAULT_NO_DATA_LABEL,
    show_val_requirement: bool = False,
    val_requirement_value: float = 0.0,
    units: str = "",
    for_export: bool = False,
    bias_included: bool = None,
) -> None:
    """Render a stacked bar chart on *fig*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to draw on (will be cleared).
    category_data : CategoryData
        Data for one category.
    k_factor : float
        Coverage factor multiplier. Only applied when
        *already_expanded* is ``False``.
    already_expanded : bool
        If ``True`` (default), uncertainty values are used as-is.
    concern_direction : str
        ``"under"``, ``"over"``, or ``"both"``.
    no_data_label : str
        Text shown in gap markers.
    show_val_requirement : bool
        Whether to draw a validation requirement line.
    val_requirement_value : float
        Y value for the validation requirement line.
    units : str
        Unit label for the y-axis.
    for_export : bool
        If ``True``, use light-theme colours for separator shading.
    bias_included : bool or None
        If not None, a subtitle is added noting whether CFD includes
        additional bias (True) or does not (False).
    """
    fig.clf()
    pal = BOEING_PALETTE

    # Multiplier: apply k only if values are NOT already expanded
    k = 1.0 if already_expanded else k_factor

    sensors = category_data.sensors
    conditions = category_data.conditions
    points = category_data.points
    n_conditions = len(conditions)
    n_sensors = len(sensors)
    n_total = n_sensors * n_conditions

    if n_total == 0:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, 'No valid data points',
                transform=ax.transAxes, ha='center', va='center')
        return

    # ── Build flat data arrays ─────────────────────────────────────
    x_positions = np.arange(n_total)
    e_vals = np.full(n_total, np.nan)
    u_num_vals = np.full(n_total, np.nan)
    u_input_vals = np.full(n_total, np.nan)
    u_d_vals = np.full(n_total, np.nan)
    has_data_arr = np.zeros(n_total, dtype=bool)

    for i, pt in enumerate(points):
        if pt.has_data:
            # Guard: np.isfinite on all values (checklist §10)
            vals = (pt.E, pt.u_num, pt.u_input, pt.u_d)
            if all(np.isfinite(v) for v in vals):
                has_data_arr[i] = True
                e_vals[i] = pt.E
                # abs() guards: uncertainties are magnitudes (always ≥ 0)
                u_num_vals[i] = abs(pt.u_num) * k
                u_input_vals[i] = abs(pt.u_input) * k
                u_d_vals[i] = abs(pt.u_d) * k

    # Safe values for bar plotting (NaN → 0, so no-data bars are invisible)
    e_safe = np.nan_to_num(e_vals, nan=0.0)
    u_num_h = np.nan_to_num(u_num_vals, nan=0.0)
    u_input_h = np.nan_to_num(u_input_vals, nan=0.0)
    u_d_h = np.nan_to_num(u_d_vals, nan=0.0)

    # ── Create axes with space for key table ───────────────────────
    n_key_rows = (n_conditions + 3) // 4  # 4 columns in key table
    key_height_ratio = max(0.22, min(0.42, n_key_rows * 0.09))
    gs = GridSpec(
        2, 1,
        height_ratios=[1.0 - key_height_ratio, key_height_ratio],
        hspace=0.30,
        figure=fig,
    )
    ax = fig.add_subplot(gs[0])
    ax_key = fig.add_subplot(gs[1])

    # ── TC group separators & labels (no alternating background) ────
    for s_idx, sensor in enumerate(sensors):
        x_start = s_idx * n_conditions - 0.5
        x_end = (s_idx + 1) * n_conditions - 0.5

        # Vertical dashed line at group boundary (skip first)
        if s_idx > 0:
            ax.axvline(x_start, color='#888888', linewidth=0.8,
                       linestyle='--', zorder=1)

        # TC label centered at top of each group
        x_center = (x_start + x_end) / 2
        ax.text(
            x_center, 1.02, sensor,
            transform=ax.get_xaxis_transform(),
            ha='center', va='bottom',
            fontsize=6, fontweight='bold',
            color=EXPORT_TEXT_COLOR if for_export else DARK_COLORS['fg'],
        )

    # ── Draw stacked bars ──────────────────────────────────────────
    # Scale bar width based on number of conditions to avoid overly
    # wide bars when there are many conditions per sensor group.
    bar_width = min(0.7, max(0.25, 5.0 / max(n_conditions, 1)))

    if concern_direction == "under":
        # Bands peel downward from E
        ax.bar(x_positions, u_num_h,
               bottom=e_safe - u_num_h,
               width=bar_width, color=pal['band_num'], alpha=0.85,
               edgecolor=pal['band_num_edge'], linewidth=0.3,
               label=r'$u_{num}$', zorder=3)
        ax.bar(x_positions, u_input_h,
               bottom=e_safe - u_num_h - u_input_h,
               width=bar_width, color=pal['band_input'], alpha=0.85,
               edgecolor=pal['band_input_edge'], linewidth=0.3,
               label=r'$u_{input}$', zorder=3)
        ax.bar(x_positions, u_d_h,
               bottom=e_safe - u_num_h - u_input_h - u_d_h,
               width=bar_width, color=pal['band_data'], alpha=0.85,
               edgecolor=pal['band_data_edge'], linewidth=0.3,
               label=r'$u_d$', zorder=3)

    elif concern_direction == "over":
        # Bands stack upward from E
        ax.bar(x_positions, u_num_h,
               bottom=e_safe,
               width=bar_width, color=pal['band_num'], alpha=0.85,
               edgecolor=pal['band_num_edge'], linewidth=0.3,
               label=r'$u_{num}$', zorder=3)
        ax.bar(x_positions, u_input_h,
               bottom=e_safe + u_num_h,
               width=bar_width, color=pal['band_input'], alpha=0.85,
               edgecolor=pal['band_input_edge'], linewidth=0.3,
               label=r'$u_{input}$', zorder=3)
        ax.bar(x_positions, u_d_h,
               bottom=e_safe + u_num_h + u_input_h,
               width=bar_width, color=pal['band_data'], alpha=0.85,
               edgecolor=pal['band_data_edge'], linewidth=0.3,
               label=r'$u_d$', zorder=3)

    else:
        # "both": symmetric bands above AND below E
        # ── Downward bars (with labels for legend) ──
        ax.bar(x_positions, u_num_h,
               bottom=e_safe - u_num_h,
               width=bar_width, color=pal['band_num'], alpha=0.85,
               edgecolor=pal['band_num_edge'], linewidth=0.3,
               label=r'$u_{num}$', zorder=3)
        ax.bar(x_positions, u_input_h,
               bottom=e_safe - u_num_h - u_input_h,
               width=bar_width, color=pal['band_input'], alpha=0.85,
               edgecolor=pal['band_input_edge'], linewidth=0.3,
               label=r'$u_{input}$', zorder=3)
        ax.bar(x_positions, u_d_h,
               bottom=e_safe - u_num_h - u_input_h - u_d_h,
               width=bar_width, color=pal['band_data'], alpha=0.85,
               edgecolor=pal['band_data_edge'], linewidth=0.3,
               label=r'$u_d$', zorder=3)

        # ── Upward bars (no labels — suppress legend duplicates) ──
        ax.bar(x_positions, u_num_h,
               bottom=e_safe,
               width=bar_width, color=pal['band_num'], alpha=0.85,
               edgecolor=pal['band_num_edge'], linewidth=0.3,
               zorder=3)
        ax.bar(x_positions, u_input_h,
               bottom=e_safe + u_num_h,
               width=bar_width, color=pal['band_input'], alpha=0.85,
               edgecolor=pal['band_input_edge'], linewidth=0.3,
               zorder=3)
        ax.bar(x_positions, u_d_h,
               bottom=e_safe + u_num_h + u_input_h,
               width=bar_width, color=pal['band_data'], alpha=0.85,
               edgecolor=pal['band_data_edge'], linewidth=0.3,
               zorder=3)

    # ── E markers (horizontal cap at E for each valid bar) ─────────
    # Use white on dark GUI, near-black on export — clearly distinct
    # from the blue u_num band colour.
    e_marker_color = '#1a1a1a' if for_export else '#ffffff'
    valid_x = x_positions[has_data_arr]
    valid_e = e_vals[has_data_arr]
    if len(valid_x) > 0:
        ax.scatter(
            valid_x, valid_e,
            marker='_', s=25, color=e_marker_color,
            linewidths=1.2, zorder=5,
            label='E = S \u2212 D',
        )

    # ── Gap markers ────────────────────────────────────────────────
    # Solid constant gray (alpha=1.0) so gaps are clearly distinct.
    gap_indices = np.where(~has_data_arr)[0]
    for gi in gap_indices:
        ax.axvspan(
            gi - 0.4, gi + 0.4,
            color=pal['gap_fill'], alpha=1.0, zorder=2,
        )

    # Place gap label once per contiguous gap group
    if len(gap_indices) > 0:
        gap_groups = []
        current_group = [gap_indices[0]]
        for idx in gap_indices[1:]:
            if idx == current_group[-1] + 1:
                current_group.append(idx)
            else:
                gap_groups.append(current_group)
                current_group = [idx]
        gap_groups.append(current_group)

        for group in gap_groups:
            x_mid = (group[0] + group[-1]) / 2
            ax.text(
                x_mid, 0, no_data_label,
                ha='center', va='center',
                fontsize=4.5, color=pal['gap_text'],
                rotation=90, style='italic',
                zorder=6,
                bbox=dict(
                    boxstyle='round,pad=0.2',
                    facecolor=pal['gap_fill'],
                    edgecolor='none',
                    alpha=0.7,
                ),
            )

    # ── E = 0 reference line ───────────────────────────────────────
    ax.axhline(
        0, color=pal['zero_line'], linewidth=1.2, linestyle='-',
        zorder=4, label='E = 0',
    )

    # ── Optional validation requirement line ───────────────────────
    if show_val_requirement:
        ax.axhline(
            val_requirement_value,
            color=pal['val_req_line'], linewidth=1.0, linestyle='--',
            zorder=4, label=f'Requirement = {val_requirement_value}',
        )

    # ── X-axis: condition codes under FIRST TC group only ──────────
    # All sensor groups share the same conditions, so labelling every
    # group repeats identical codes and crowds the axis.  Instead,
    # show codes only under the first group; subsequent groups get
    # minor ticks but no labels.  Adaptive thinning keeps the first
    # group readable when there are many conditions.
    tick_positions = list(range(n_total))  # tick mark at every bar

    # Build labels: first TC group gets codes, rest blank
    tick_labels_list = [''] * n_total

    # Decide which codes to show in the first group
    if n_conditions <= 6:
        show_set = set(range(n_conditions))          # all
    elif n_conditions <= 14:
        # Every 2nd, always including first and last
        show_set = set(range(0, n_conditions, 2))
        show_set.add(n_conditions - 1)
    else:
        # Every 3rd, always including first and last
        step = max(3, n_conditions // 5)
        show_set = set(range(0, n_conditions, step))
        show_set.add(n_conditions - 1)

    for i in sorted(show_set):
        tick_labels_list[i] = condition_code(i)

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels_list, fontsize=5.5)
    ax.set_xlim(-0.5, n_total - 0.5)

    # ── Axis labels and title ──────────────────────────────────────
    # Y-axis: single static statement regardless of concern direction
    unit_str = f" ({units})" if units else ""
    ax.set_ylabel(f"Comparison Error, E{unit_str}", fontsize=8)
    ax.set_xlabel("Condition Code", fontsize=7)

    # Title hierarchy using ax.text (export pipeline handles ax.texts):
    #   y ≈ 1.16  Category title (bold, 10pt)
    #   y ≈ 1.09  Bias note (bold, 9pt) — only if bias set
    #   y ≈ 1.02  TC sensor labels (bold, 6pt)
    #   y = 1.00  Axes top edge
    text_color = EXPORT_TEXT_COLOR if for_export else DARK_COLORS['fg']
    ax.text(
        0.5, 1.16,
        category_data.category,
        transform=ax.transAxes,
        ha='center', va='bottom',
        fontsize=10, fontweight='bold',
        color=text_color,
    )
    if bias_included is not None:
        if bias_included:
            bias_note = "(CFD prediction includes additional bias)"
        else:
            bias_note = "(CFD prediction does not include additional bias)"
        ax.text(
            0.5, 1.09,
            bias_note,
            transform=ax.transAxes,
            ha='center', va='bottom',
            fontsize=9, fontweight='bold',
            color=text_color,
        )

    # ── Legend outside chart area ──────────────────────────────────
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles, labels,
            loc='upper left',
            bbox_to_anchor=(1.01, 1.0),
            fontsize=6,
            framealpha=0.9,
            edgecolor='#999999' if for_export else DARK_COLORS['border'],
        )

    # ── Grid ───────────────────────────────────────────────────────
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)
    ax.tick_params(axis='both', which='both', length=3)

    # ── Condition key table (below chart) ──────────────────────────
    ax_key.axis('off')
    _render_condition_key(ax_key, conditions, for_export)

    # ── Layout ─────────────────────────────────────────────────────
    # tight_layout positions the GridSpec (chart + key).
    # subplots_adjust(top) then pushes everything down to leave
    # room for the title / bias / TC-label text above the axes.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            fig.tight_layout()
        except ValueError:
            pass
    fig.subplots_adjust(top=0.84)


# ── Helper: clean condition key ───────────────────────────────────

def _render_condition_key(ax, conditions, for_export):
    """Render a clean condition code → label key using text.

    Text entries arranged in up to 4 columns, replacing the
    matplotlib table widget for a cleaner look.
    """
    n = len(conditions)
    if n == 0:
        return

    text_color = EXPORT_TEXT_COLOR if for_export else DARK_COLORS['fg']

    # Title — positioned at very top of key area
    ax.text(
        0.5, 1.0, 'Condition Key',
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=6.5, fontweight='bold',
        color=text_color,
    )

    # Build entries: "A = Condition 1", "B = Condition 2", …
    entries = []
    for i in range(n):
        code = condition_code(i)
        entries.append(f"{code} = {conditions[i]}")

    # Arrange in columns (max 4), reading down then across
    n_cols = min(4, n)
    n_rows = (n + n_cols - 1) // n_cols
    col_width = 1.0 / n_cols
    # Entries start well below title to avoid overlap
    entry_top = 0.74
    entry_bottom = 0.02
    row_height = (entry_top - entry_bottom) / max(n_rows, 1)

    for i, entry in enumerate(entries):
        col = i // n_rows
        row = i % n_rows
        x = col * col_width + 0.02
        y = entry_top - row * row_height
        ax.text(
            x, y, entry,
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=5.5,
            color=text_color,
        )
