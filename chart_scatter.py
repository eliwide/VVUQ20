"""
E vs U_val scatter plot for the VVUQ Validation Plotter.

Classic V&V 20 summary: |E| on x-axis, U_val on y-axis, with a
1:1 reference line.  Points below the line fail validation
(|E| > U_val).  Coloured by sensor.
"""

import numpy as np
from matplotlib.figure import Figure

from .constants import BOEING_PALETTE
from .data_model import CategoryData


def render_scatter(
    fig: Figure,
    category_data: CategoryData,
    *,
    color_by: str = "sensor",
    units: str = "",
    for_export: bool = False,
    bias_included: bool = None,
) -> None:
    """Render an |E| vs U_val scatter plot on *fig*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to draw on (will be cleared).
    category_data : CategoryData
        Data for one category.
    color_by : str
        ``"sensor"`` or ``"condition"``.
    units : str
        Unit label for the axes.
    for_export : bool
        If ``True``, use light-theme colours.
    bias_included : bool or None
        If not None, a note is appended to the title.
    """
    fig.clf()
    pal = BOEING_PALETTE
    cycle = pal['scatter_cycle']

    sensors = category_data.sensors
    conditions = category_data.conditions
    points = category_data.points

    ax = fig.add_subplot(111)

    # ── Collect data by grouping key ─────────────────────────────────
    if color_by == "condition":
        groups = conditions
        get_group = lambda pt: pt.condition
    else:
        groups = sensors
        get_group = lambda pt: pt.sensor

    group_data = {g: {'e_abs': [], 'u_val': []} for g in groups}

    for pt in points:
        if not pt.has_data:
            continue
        e_abs = abs(pt.E)
        u_val = pt.U_val
        # Guard: np.isfinite (checklist §10)
        if not (np.isfinite(e_abs) and np.isfinite(u_val)):
            continue
        g = get_group(pt)
        group_data[g]['e_abs'].append(e_abs)
        group_data[g]['u_val'].append(u_val)

    # ── 1:1 reference line ───────────────────────────────────────────
    # Find data range for the reference line
    all_e = []
    all_u = []
    for gd in group_data.values():
        all_e.extend(gd['e_abs'])
        all_u.extend(gd['u_val'])

    if not all_e:
        ax.text(0.5, 0.5, 'No valid data points',
                transform=ax.transAxes, ha='center', va='center')
        return

    max_val = max(max(all_e), max(all_u)) * 1.1
    ax.plot(
        [0, max_val], [0, max_val],
        color='#888888', linewidth=1.0, linestyle='--',
        zorder=2, label='|E| = U_val (1:1)',
    )

    # ── Scatter by group ─────────────────────────────────────────────
    for g_idx, group_name in enumerate(groups):
        gd = group_data[group_name]
        if not gd['e_abs']:
            continue
        color = cycle[g_idx % len(cycle)]
        ax.scatter(
            gd['e_abs'], gd['u_val'],
            c=color, s=20, alpha=0.8, edgecolors='white',
            linewidths=0.4, zorder=3,
            label=group_name,
        )

    # ── Shading for validation regions ───────────────────────────────
    # Above the 1:1 line = validated region (U_val > |E|)
    ax.fill_between(
        [0, max_val], [0, max_val], [max_val, max_val],
        color=pal['pass_green'], alpha=0.08, zorder=0,
    )
    # Below the 1:1 line = failed region
    ax.fill_between(
        [0, max_val], [0, 0], [0, max_val],
        color=pal['fail_red'], alpha=0.08, zorder=0,
    )

    # ── Labels ───────────────────────────────────────────────────────
    unit_str = f" ({units})" if units else ""
    ax.set_xlabel(f"|E| — Comparison Error Magnitude{unit_str}", fontsize=8)
    ax.set_ylabel(f"U_val — Validation Uncertainty{unit_str}", fontsize=8)
    title = f"Validation Scatter — {category_data.category}"
    if bias_included is True:
        title += "\n(CFD prediction includes additional bias)"
    elif bias_included is False:
        title += "\n(CFD prediction does not include additional bias)"
    ax.set_title(title, fontsize=10, fontweight='bold')

    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(linewidth=0.4, alpha=0.5)

    # ── Legend ────────────────────────────────────────────────────────
    # Limit legend entries if there are many groups
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 12:
        # Show first 10 + "..."
        handles = handles[:11]
        labels = labels[:11]
        labels[-1] = f"... ({len(groups) - 10} more)"

    ax.legend(
        handles, labels,
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        fontsize=5.5,
        framealpha=0.9,
        markerscale=1.5,
    )

    fig.tight_layout(pad=1.5)
