"""
Validation status heatmap for the VVUQ Validation Plotter.

Sensor × Condition grid coloured by validation outcome:
  - Green: |E| ≤ U_val (validated)
  - Red:   |E| > U_val (failed)
  - Gray:  No data
"""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap

from .constants import BOEING_PALETTE
from .data_model import CategoryData


# Status codes for the colour map
_STATUS_NO_DATA = 0
_STATUS_PASS = 1
_STATUS_FAIL = 2


def render_heatmap(
    fig: Figure,
    category_data: CategoryData,
    *,
    for_export: bool = False,
    bias_included: bool = None,
) -> None:
    """Render a pass/fail validation heatmap on *fig*.

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
    points = category_data.points
    n_sensors = len(sensors)
    n_conditions = len(conditions)

    # ── Build status matrix ──────────────────────────────────────────
    status = np.full((n_sensors, n_conditions), _STATUS_NO_DATA, dtype=int)
    ratio_text = np.full((n_sensors, n_conditions), '', dtype=object)

    # Pre-build sensor index for O(1) lookup (checklist §3)
    sensor_idx = {s: i for i, s in enumerate(sensors)}

    for pt in points:
        s_idx = sensor_idx[pt.sensor]
        c_idx = pt.condition_index

        if not pt.has_data:
            continue

        e_abs = abs(pt.E)
        u_val = pt.U_val

        # Guard: np.isfinite check (checklist §10)
        if not (np.isfinite(e_abs) and np.isfinite(u_val)):
            continue

        if e_abs <= u_val:
            status[s_idx, c_idx] = _STATUS_PASS
        else:
            status[s_idx, c_idx] = _STATUS_FAIL

        # Ratio text for cell annotation
        if u_val > 0 and np.isfinite(e_abs / u_val):
            ratio_text[s_idx, c_idx] = f"{e_abs / u_val:.2f}"
        elif u_val == 0 and e_abs == 0:
            ratio_text[s_idx, c_idx] = "0"
        else:
            ratio_text[s_idx, c_idx] = "N/A"

    # ── Colour map: 0=gray, 1=green, 2=red ──────────────────────────
    cmap = ListedColormap([
        pal['no_data_gray'],
        pal['pass_green'],
        pal['fail_red'],
    ])

    ax = fig.add_subplot(111)
    im = ax.imshow(
        status, cmap=cmap, vmin=0, vmax=2,
        aspect='auto', origin='upper',
    )

    # ── Cell annotations ─────────────────────────────────────────────
    text_color_light = '#ffffff'
    text_color_dark = '#333333'

    for s_idx in range(n_sensors):
        for c_idx in range(n_conditions):
            cell_status = status[s_idx, c_idx]
            text = ratio_text[s_idx, c_idx]
            if not text:
                text = "N/A"

            # Choose text colour for contrast
            if cell_status == _STATUS_NO_DATA:
                tc = text_color_dark
            else:
                tc = text_color_light

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
    title = (f"Validation Status — {category_data.category}\n"
             f"(cell value = |E| / U_val; green \u2264 1.0 = validated)")
    if bias_included is True:
        title += "\n(CFD prediction includes additional bias)"
    elif bias_included is False:
        title += "\n(CFD prediction does not include additional bias)"
    ax.set_title(title, fontsize=10, fontweight='bold')

    # ── Legend ────────────────────────────────────────────────────────
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=pal['pass_green'], label='Validated (|E| ≤ U_val)'),
        Patch(facecolor=pal['fail_red'], label='Not Validated (|E| > U_val)'),
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
