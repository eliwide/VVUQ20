"""
Margin ratio trend chart for the VVUQ Validation Plotter.

Plots U_val / |E| for each sensor across conditions.  Values above
1.0 mean validation uncertainty covers the comparison error (validated);
below 1.0 means it does not.
"""

import numpy as np
from matplotlib.figure import Figure

from .constants import BOEING_PALETTE
from .data_model import CategoryData


# Cap ratio to avoid inf dominating the plot (when |E| → 0)
_RATIO_CAP = 20.0


def render_margin_trend(
    fig: Figure,
    category_data: CategoryData,
    *,
    units: str = "",
    for_export: bool = False,
    bias_included: bool = None,
) -> None:
    """Render a margin ratio trend chart on *fig*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to draw on (will be cleared).
    category_data : CategoryData
        Data for one category.
    units : str
        Unit label (for subtitle context).
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
    n_conditions = len(conditions)

    ax = fig.add_subplot(111)

    any_data = False

    for s_idx, sensor in enumerate(sensors):
        ratios = []
        x_vals = []

        for c_idx in range(n_conditions):
            # Find the point for this sensor/condition
            pt = None
            for p in category_data.points:
                if p.sensor == sensor and p.condition_index == c_idx:
                    pt = p
                    break

            if pt is None or not pt.has_data:
                ratios.append(np.nan)
                x_vals.append(c_idx)
                continue

            e_abs = abs(pt.E)
            u_val = pt.U_val

            # Guard: np.isfinite (checklist §10)
            if not (np.isfinite(e_abs) and np.isfinite(u_val)):
                ratios.append(np.nan)
                x_vals.append(c_idx)
                continue

            # Guard: division by zero (checklist §6)
            if e_abs < 1e-12:
                ratios.append(_RATIO_CAP)
            else:
                ratio = u_val / e_abs
                ratios.append(min(ratio, _RATIO_CAP))

            x_vals.append(c_idx)

        ratios = np.array(ratios)
        x_vals = np.array(x_vals)

        # Only plot if there's at least one finite value
        if np.any(np.isfinite(ratios)):
            any_data = True
            masked = np.ma.masked_where(~np.isfinite(ratios), ratios)
            color = cycle[s_idx % len(cycle)]
            ax.plot(
                x_vals, masked,
                color=color, linewidth=1.0, marker='o', markersize=3,
                markerfacecolor=color, markeredgecolor='white',
                markeredgewidth=0.3,
                label=sensor, zorder=3,
            )

    if not any_data:
        ax.text(0.5, 0.5, 'No valid data points',
                transform=ax.transAxes, ha='center', va='center')
        return

    # ── Reference line at ratio = 1.0 ────────────────────────────────
    ax.axhline(
        1.0, color=pal['val_req_line'], linewidth=1.2, linestyle='--',
        zorder=4, label='U_val / |E| = 1.0',
    )

    # ── Adaptive y-axis (GPT #6) ─────────────────────────────────────
    # Use data-driven top limit with small headroom, instead of
    # forcing to _RATIO_CAP + 1 which wastes most of the y-axis.
    ylim = ax.get_ylim()
    y_top = max(ylim[1] * 1.1, 2.0)  # at least show up to 2.0
    ax.set_ylim(bottom=0, top=y_top)

    # ── Shading ──────────────────────────────────────────────────────
    ax.fill_between(
        [-0.5, n_conditions - 0.5], 1.0, y_top,
        color=pal['pass_green'], alpha=0.06, zorder=0,
    )
    ax.fill_between(
        [-0.5, n_conditions - 0.5], 0, 1.0,
        color=pal['fail_red'], alpha=0.06, zorder=0,
    )

    # ── Labels ───────────────────────────────────────────────────────
    ax.set_xticks(range(n_conditions))
    ax.set_xticklabels(conditions, fontsize=5.5, rotation=45, ha='right')
    ax.set_xlim(-0.5, n_conditions - 0.5)
    ax.set_ylim(bottom=0)

    ax.set_xlabel("Condition", fontsize=8)
    ax.set_ylabel("U_val / |E| (Margin Ratio)", fontsize=8)
    title = (f"Margin Ratio Trend — {category_data.category}\n"
             f"(above 1.0 = validated)")
    if bias_included is True:
        title += "\n(CFD prediction includes additional bias)"
    elif bias_included is False:
        title += "\n(CFD prediction does not include additional bias)"
    ax.set_title(title, fontsize=10, fontweight='bold')

    ax.grid(linewidth=0.4, alpha=0.5)

    # Legend outside
    handles, labels = ax.get_legend_handles_labels()
    if len(handles) > 12:
        handles = handles[:11]
        labels = labels[:11]
        labels[-1] = f"... ({len(sensors) - 10} more)"
    ax.legend(
        handles, labels,
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        fontsize=5.5,
        framealpha=0.9,
    )

    fig.tight_layout(pad=1.5)
