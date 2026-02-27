"""
Dominant uncertainty source map for the VVUQ Validation Plotter.

Sensor × Condition grid coloured by which uncertainty source has
the largest contribution at each point:
  - Blue:   u_num dominant
  - Orange: u_input dominant
  - Green:  u_d dominant
  - Gray:   No data
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .constants import BOEING_PALETTE
from .data_model import CategoryData


# Source codes
_SRC_NO_DATA = 0
_SRC_NUM = 1
_SRC_INPUT = 2
_SRC_DATA = 3


def render_dominant_source(
    fig: Figure,
    category_data: CategoryData,
    *,
    for_export: bool = False,
    bias_included: bool = None,
) -> None:
    """Render a dominant uncertainty source heatmap on *fig*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to draw on (will be cleared).
    category_data : CategoryData
        Data for one category.
    for_export : bool
        If ``True``, use light-theme text colours.
    bias_included : bool or None
        If not None, a note is appended to the title.
    """
    fig.clf()
    pal = BOEING_PALETTE

    sensors = category_data.sensors
    conditions = category_data.conditions
    n_sensors = len(sensors)
    n_conditions = len(conditions)

    # ── Build source matrix ──────────────────────────────────────────
    source_grid = np.full((n_sensors, n_conditions), _SRC_NO_DATA, dtype=int)
    pct_text = np.full((n_sensors, n_conditions), '', dtype=object)

    # Pre-build sensor index for O(1) lookup (checklist §3)
    sensor_idx = {s: i for i, s in enumerate(sensors)}

    for pt in category_data.points:
        s_idx = sensor_idx[pt.sensor]
        c_idx = pt.condition_index

        if not pt.has_data:
            continue

        u_num = abs(pt.u_num)
        u_input = abs(pt.u_input)
        u_d = abs(pt.u_d)

        # Guard: np.isfinite (checklist §10)
        if not all(np.isfinite(v) for v in (u_num, u_input, u_d)):
            continue

        # Determine dominant source by magnitude
        vals = {'num': u_num, 'input': u_input, 'data': u_d}
        dominant = max(vals, key=vals.get)

        if dominant == 'num':
            source_grid[s_idx, c_idx] = _SRC_NUM
        elif dominant == 'input':
            source_grid[s_idx, c_idx] = _SRC_INPUT
        else:
            source_grid[s_idx, c_idx] = _SRC_DATA

        # Variance fraction (u_i² / U_val²) — standard V&V 20 presentation.
        # RSS composition: u_i² / U_val² gives each source's contribution
        # to the combined validation uncertainty.
        u_val_sq = (pt.U_val ** 2 if (pt.U_val is not None
                    and np.isfinite(pt.U_val) and pt.U_val > 0) else 0.0)
        if u_val_sq > 0:
            pct = 100.0 * vals[dominant] ** 2 / u_val_sq
            pct_text[s_idx, c_idx] = f"{pct:.0f}%"
        else:
            pct_text[s_idx, c_idx] = "N/A"

    # ── Colour map ───────────────────────────────────────────────────
    cmap = ListedColormap([
        pal['no_data_gray'],   # 0 = no data
        pal['band_num'],       # 1 = u_num
        pal['band_input'],     # 2 = u_input
        pal['band_data'],      # 3 = u_d
    ])

    ax = fig.add_subplot(111)
    im = ax.imshow(
        source_grid, cmap=cmap, vmin=0, vmax=3,
        aspect='auto', origin='upper',
    )

    # ── Cell annotations ─────────────────────────────────────────────
    for s_idx in range(n_sensors):
        for c_idx in range(n_conditions):
            text = pct_text[s_idx, c_idx]
            if not text:
                text = "N/A"
            src = source_grid[s_idx, c_idx]
            tc = '#ffffff' if src != _SRC_NO_DATA else '#333333'
            ax.text(
                c_idx, s_idx, text,
                ha='center', va='center',
                fontsize=5.5, color=tc, fontweight='bold',
            )

    # ── Axis labels ──────────────────────────────────────────────────
    ax.set_xticks(range(n_conditions))
    ax.set_xticklabels(conditions, fontsize=5.5, rotation=45, ha='right')
    ax.set_yticks(range(n_sensors))
    ax.set_yticklabels(sensors, fontsize=6)

    ax.set_xlabel("Condition", fontsize=8)
    ax.set_ylabel("Sensor", fontsize=8)
    title = (f"Dominant Uncertainty Source — {category_data.category}\n"
             r"(cell value = $u_i^2 / U_{val}^2$ variance fraction)")
    if bias_included is True:
        title += "\n(CFD prediction includes additional bias)"
    elif bias_included is False:
        title += "\n(CFD prediction does not include additional bias)"
    ax.set_title(title, fontsize=10, fontweight='bold')

    # ── Legend ────────────────────────────────────────────────────────
    legend_elements = [
        Patch(facecolor=pal['band_num'], label=r'$u_{num}$ dominant'),
        Patch(facecolor=pal['band_input'], label=r'$u_{input}$ dominant'),
        Patch(facecolor=pal['band_data'], label=r'$u_d$ dominant'),
        Patch(facecolor=pal['no_data_gray'], label='No Data'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        fontsize=6,
        framealpha=0.9,
    )

    fig.tight_layout(pad=1.5)
