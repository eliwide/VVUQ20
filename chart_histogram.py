"""
|E| − U_val histogram for the VVUQ Validation Plotter.

Shows the distribution of validation margin (|E| − U_val) across all
points within a category.  Points where |E| − U_val > 0 have positive
margin (the absolute comparison error exceeds validation uncertainty),
indicating potential validation failure.

Uses |E| (absolute value) for consistency with ASME V&V 20 validation
methodology, where the validation comparison is |E| ≤ U_val.
"""

import numpy as np
from matplotlib.figure import Figure

from .constants import (
    BOEING_PALETTE, DARK_COLORS,
    EXPORT_TEXT_COLOR, EXPORT_BG_COLOR,
)
from .data_model import CategoryData


def render_histogram(
    fig: Figure,
    category_data: CategoryData,
    *,
    units: str = "",
    for_export: bool = False,
    bias_included: bool = None,
) -> None:
    """Render an |E| − U_val histogram on *fig*.

    Uses |E| (absolute comparison error) for consistency with
    ASME V&V 20 validation methodology (|E| ≤ U_val).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to draw on (will be cleared).
    category_data : CategoryData
        Data for one category.
    units : str
        Unit label for the x-axis.
    for_export : bool
        If ``True``, use light-theme colours.
    bias_included : bool or None
        If not None, a note is appended to the title.
    """
    fig.clf()
    pal = BOEING_PALETTE

    # ── Collect |E| - U_val for all valid points ─────────────────────
    # V&V 20 consistency: validation is |E| ≤ U_val, so margin = |E| - U_val
    margins = []
    for pt in category_data.points:
        if not pt.has_data:
            continue
        # Guard: np.isfinite (checklist §10)
        if not (np.isfinite(pt.E) and np.isfinite(pt.U_val)):
            continue
        margins.append(abs(pt.E) - pt.U_val)

    ax = fig.add_subplot(111)

    if not margins:
        ax.text(0.5, 0.5, 'No valid data points',
                transform=ax.transAxes, ha='center', va='center')
        return

    margins = np.array(margins)
    n_total = len(margins)

    # ── Histogram ────────────────────────────────────────────────────
    # Auto-bin with Sturges' rule, capped at 50
    n_bins = min(50, max(10, int(np.ceil(np.log2(n_total) + 1))))

    # Colour bars by whether they represent pass (negative) or fail (positive)
    n, bins, patches = ax.hist(
        margins, bins=n_bins, edgecolor='white', linewidth=0.5,
        zorder=3, alpha=0.85,
    )

    # Colour each bar based on its bin centre
    for patch, left_edge, right_edge in zip(patches, bins[:-1], bins[1:]):
        centre = (left_edge + right_edge) / 2
        if centre <= 0:
            patch.set_facecolor(pal['pass_green'])
        else:
            patch.set_facecolor(pal['fail_red'])

    # ── Zero line ────────────────────────────────────────────────────
    ax.axvline(
        0, color=pal['zero_line'], linewidth=1.5, linestyle='-',
        zorder=4, label='|E| − U_val = 0',
    )

    # ── Statistics annotation ────────────────────────────────────────
    n_pass = int(np.sum(margins <= 0))
    n_fail = n_total - n_pass
    pct_pass = 100.0 * n_pass / n_total if n_total > 0 else 0

    stats_text = (
        f"Total points: {n_total}\n"
        f"Validated (|E| − U_val ≤ 0): {n_pass} ({pct_pass:.1f}%)\n"
        f"Not validated: {n_fail} ({100 - pct_pass:.1f}%)\n"
        f"Mean margin: {np.mean(margins):.2f}\n"
        f"Std dev: {np.std(margins):.2f}"
    )

    text_color = EXPORT_TEXT_COLOR if for_export else DARK_COLORS['fg']
    box_color = EXPORT_BG_COLOR if for_export else DARK_COLORS['bg_widget']
    ax.text(
        0.98, 0.95, stats_text,
        transform=ax.transAxes, ha='right', va='top',
        fontsize=6.5, family='monospace',
        color=text_color,
        bbox=dict(
            boxstyle='round,pad=0.4',
            facecolor=box_color,
            edgecolor='#999999',
            alpha=0.9,
        ),
    )

    # ── Labels ───────────────────────────────────────────────────────
    unit_str = f" ({units})" if units else ""
    ax.set_xlabel(f"|E| − U_val{unit_str}", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    title = f"Validation Margin Distribution — {category_data.category}"
    if bias_included is True:
        title += "\n(CFD prediction includes additional bias)"
    elif bias_included is False:
        title += "\n(CFD prediction does not include additional bias)"
    ax.set_title(title, fontsize=10, fontweight='bold')

    ax.legend(fontsize=6, framealpha=0.9)
    ax.grid(axis='y', linewidth=0.4, alpha=0.5)

    fig.tight_layout(pad=1.5)
