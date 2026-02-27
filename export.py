"""
Export utilities for the VVUQ Validation Plotter.

Handles PNG export with automatic light-theme switching (dark GUI
theme → white-background export), clipboard copy, and batch export.
Uses try/finally to guarantee theme restoration.

State save/restore is comprehensive: every visual property that
``_apply_light_theme`` touches is captured and restored exactly,
including grid lines, all text objects, and tick mark colours.
"""

import io
import os
from matplotlib.figure import Figure

from .constants import (
    EXPORT_DPI, EXPORT_WIDTH_INCHES, CLIPBOARD_DPI, REPORT_DPI,
    PLOT_STYLE_DARK, PLOT_STYLE_LIGHT, DARK_COLORS,
)


def _save_figure_state(fig: Figure) -> dict:
    """Save current figure/axes colours for later restoration.

    Returns a dict containing **all** mutable visual properties
    that ``_apply_light_theme`` will modify, so that
    ``_restore_figure_state`` can undo every change.
    """
    state = {
        'fig_facecolor': fig.get_facecolor(),
        'axes_states': [],
    }
    for ax in fig.get_axes():
        ax_state = {
            'facecolor': ax.get_facecolor(),
            'title_color': ax.title.get_color(),
            'xlabel_color': ax.xaxis.label.get_color(),
            'ylabel_color': ax.yaxis.label.get_color(),
            'spine_colors': {
                spine: ax.spines[spine].get_edgecolor()
                for spine in ax.spines
            },
            # Tick label colours
            'tick_label_colors_x': [
                t.get_color() for t in ax.get_xticklabels()
            ],
            'tick_label_colors_y': [
                t.get_color() for t in ax.get_yticklabels()
            ],
            # Tick mark line colours (save from first major tick)
            'xtick_mark_color': None,
            'ytick_mark_color': None,
            # Grid line colours
            'grid_colors_x': [
                line.get_color() for line in ax.get_xgridlines()
            ],
            'grid_colors_y': [
                line.get_color() for line in ax.get_ygridlines()
            ],
            # ALL text object colours (annotations, key entries, etc.)
            'text_colors': [t.get_color() for t in ax.texts],
        }

        # Tick mark colours from first major tick on each axis
        xticks = ax.xaxis.get_major_ticks()
        if xticks:
            ax_state['xtick_mark_color'] = xticks[0].tick1line.get_color()
        yticks = ax.yaxis.get_major_ticks()
        if yticks:
            ax_state['ytick_mark_color'] = yticks[0].tick1line.get_color()

        # Scatter / PathCollection face colours
        ax_state['collection_facecolors'] = [
            coll.get_facecolor().copy() for coll in ax.collections
        ]
        ax_state['collection_edgecolors'] = [
            coll.get_edgecolor().copy() for coll in ax.collections
        ]

        # Legend
        legend = ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            ax_state['legend_facecolor'] = frame.get_facecolor()
            ax_state['legend_edgecolor'] = frame.get_edgecolor()
            ax_state['legend_text_colors'] = [
                t.get_color() for t in legend.get_texts()
            ]
        state['axes_states'].append(ax_state)
    return state


def _apply_light_theme(fig: Figure) -> None:
    """Apply light (white background) theme to figure for export."""
    light = PLOT_STYLE_LIGHT
    fig.set_facecolor(light['figure.facecolor'])

    # Known dark-theme foreground colours to convert
    _dark_fg_set = frozenset((
        DARK_COLORS['fg'], DARK_COLORS['fg_dim'],
        DARK_COLORS['fg_bright'], '#cdd6f4',
    ))

    for ax in fig.get_axes():
        ax.set_facecolor(light['axes.facecolor'])
        ax.title.set_color(light['text.color'])
        ax.xaxis.label.set_color(light['axes.labelcolor'])
        ax.yaxis.label.set_color(light['axes.labelcolor'])

        for spine in ax.spines.values():
            spine.set_edgecolor(light['axes.edgecolor'])

        ax.tick_params(
            axis='x',
            colors=light['xtick.color'],
            labelcolor=light['xtick.color'],
        )
        ax.tick_params(
            axis='y',
            colors=light['ytick.color'],
            labelcolor=light['ytick.color'],
        )

        # Legend
        legend = ax.get_legend()
        if legend is not None:
            frame = legend.get_frame()
            frame.set_facecolor(light['legend.facecolor'])
            frame.set_edgecolor(light['legend.edgecolor'])
            for text in legend.get_texts():
                text.set_color(light['text.color'])

        # Grid lines
        for line in ax.get_xgridlines() + ax.get_ygridlines():
            line.set_color(light['grid.color'])

        # Scatter / PathCollection colours — convert white/near-white
        # markers to dark so they're visible on white export background.
        for coll in ax.collections:
            fc = coll.get_facecolor()
            if fc.size > 0:
                # Check if any face colour is white/near-white (R,G,B all > 0.9)
                new_fc = fc.copy()
                for row_i in range(new_fc.shape[0]):
                    r, g, b = new_fc[row_i, :3]
                    if r > 0.9 and g > 0.9 and b > 0.9:
                        new_fc[row_i, :3] = [0.1, 0.1, 0.1]  # near-black
                coll.set_facecolor(new_fc)

            ec = coll.get_edgecolor()
            if ec.size > 0:
                new_ec = ec.copy()
                for row_i in range(new_ec.shape[0]):
                    r, g, b = new_ec[row_i, :3]
                    if r > 0.9 and g > 0.9 and b > 0.9:
                        new_ec[row_i, :3] = [0.1, 0.1, 0.1]
                coll.set_edgecolor(new_ec)

        # Text objects on axes (annotations, table text, etc.)
        # Only convert dark-theme foreground colours to light.
        for text in ax.texts:
            current = text.get_color()
            if current in _dark_fg_set:
                text.set_color(light['text.color'])


def _restore_figure_state(fig: Figure, state: dict) -> None:
    """Restore saved figure/axes colours after export.

    Restores every property captured by ``_save_figure_state``,
    including grid lines, tick marks, and all text objects.
    """
    fig.set_facecolor(state['fig_facecolor'])

    for ax, ax_state in zip(fig.get_axes(), state['axes_states']):
        ax.set_facecolor(ax_state['facecolor'])
        ax.title.set_color(ax_state['title_color'])
        ax.xaxis.label.set_color(ax_state['xlabel_color'])
        ax.yaxis.label.set_color(ax_state['ylabel_color'])

        for spine_name, color in ax_state['spine_colors'].items():
            ax.spines[spine_name].set_edgecolor(color)

        # Restore tick mark colours first (tick_params sets both
        # mark and label colours, so we re-set labels after)
        if ax_state.get('xtick_mark_color') is not None:
            ax.tick_params(axis='x', colors=ax_state['xtick_mark_color'])
        if ax_state.get('ytick_mark_color') is not None:
            ax.tick_params(axis='y', colors=ax_state['ytick_mark_color'])

        # Restore tick label colours (must come AFTER tick_params)
        for label, color in zip(
            ax.get_xticklabels(), ax_state['tick_label_colors_x']
        ):
            label.set_color(color)
        for label, color in zip(
            ax.get_yticklabels(), ax_state['tick_label_colors_y']
        ):
            label.set_color(color)

        # Restore grid line colours
        for line, color in zip(
            ax.get_xgridlines(), ax_state.get('grid_colors_x', [])
        ):
            line.set_color(color)
        for line, color in zip(
            ax.get_ygridlines(), ax_state.get('grid_colors_y', [])
        ):
            line.set_color(color)

        # Restore ALL text object colours
        for text, color in zip(
            ax.texts, ax_state.get('text_colors', [])
        ):
            text.set_color(color)

        # Restore scatter / PathCollection colours
        for coll, fc in zip(
            ax.collections, ax_state.get('collection_facecolors', [])
        ):
            coll.set_facecolor(fc)
        for coll, ec in zip(
            ax.collections, ax_state.get('collection_edgecolors', [])
        ):
            coll.set_edgecolor(ec)

        # Legend
        legend = ax.get_legend()
        if legend is not None and 'legend_facecolor' in ax_state:
            frame = legend.get_frame()
            frame.set_facecolor(ax_state['legend_facecolor'])
            frame.set_edgecolor(ax_state['legend_edgecolor'])
            for text, color in zip(
                legend.get_texts(), ax_state['legend_text_colors']
            ):
                text.set_color(color)


def export_png(
    fig: Figure,
    filepath: str,
    *,
    dpi: int = EXPORT_DPI,
    width_inches: float = EXPORT_WIDTH_INCHES,
) -> None:
    """Export figure as PNG with light theme.

    Theme is switched to light (white background) for the export
    and restored afterwards.  Uses ``try/finally`` to guarantee
    restoration even on error (checklist §7).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filepath : str
        Output file path (should end with ``.png``).
    dpi : int
        Export resolution (default 600).
    width_inches : float
        Figure width in inches (default 6.0).
    """
    state = _save_figure_state(fig)
    try:
        # Adjust figure size for export
        current_h = fig.get_figheight()
        current_w = fig.get_figwidth()
        scale = width_inches / current_w if current_w > 0 else 1.0
        fig.set_size_inches(width_inches, current_h * scale)

        _apply_light_theme(fig)
        fig.savefig(
            filepath,
            dpi=dpi,
            bbox_inches='tight',
            facecolor=fig.get_facecolor(),
            edgecolor='none',
            pad_inches=0.1,
        )
    finally:
        # Restore original size and theme
        fig.set_size_inches(current_w, current_h)
        _restore_figure_state(fig, state)


def copy_to_clipboard(fig: Figure, dpi: int = CLIPBOARD_DPI) -> bool:
    """Copy figure to system clipboard as PNG image.

    Returns ``True`` on success, ``False`` if clipboard is unavailable.
    """
    try:
        from PySide6.QtWidgets import QApplication
        from PySide6.QtGui import QImage
        from PySide6.QtCore import QBuffer, QIODevice

        buf = io.BytesIO()
        state = _save_figure_state(fig)
        try:
            _apply_light_theme(fig)
            fig.savefig(
                buf, format='png', dpi=dpi,
                bbox_inches='tight',
                facecolor=fig.get_facecolor(),
                edgecolor='none',
            )
        finally:
            _restore_figure_state(fig, state)

        buf.seek(0)
        img = QImage()
        img.loadFromData(buf.read())

        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setImage(img)
            return True
        return False
    except ImportError:
        return False


def export_all_charts(
    figures: dict,
    output_dir: str,
    *,
    dpi: int = EXPORT_DPI,
    width_inches: float = EXPORT_WIDTH_INCHES,
) -> list:
    """Export multiple figures as PNGs to *output_dir*.

    Parameters
    ----------
    figures : dict
        ``{filename_stem: Figure}``
    output_dir : str
        Directory to write PNG files into.
    dpi, width_inches : int, float
        Export settings.

    Returns
    -------
    list of str
        Paths of exported files.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for name, fig in figures.items():
        # Sanitise filename
        safe_name = "".join(
            c if c.isalnum() or c in '-_ ' else '_'
            for c in name
        ).strip().replace(' ', '_')
        filepath = os.path.join(output_dir, f"{safe_name}.png")
        export_png(fig, filepath, dpi=dpi, width_inches=width_inches)
        paths.append(filepath)
    return paths
