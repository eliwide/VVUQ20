"""
Chart tabs widget (right side) for the VVUQ Validation Plotter.

Six tabs, each hosting a matplotlib FigureCanvas with a navigation
toolbar and export buttons.
"""

import os

from PySide6.QtWidgets import (
    QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QComboBox, QLabel,
)
from PySide6.QtCore import Qt

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)

from .constants import (
    DARK_COLORS, PLOT_STYLE_DARK, EXPORT_DPI, EXPORT_WIDTH_INCHES,
)
from .theme import apply_plot_style
from .data_model import CategoryData, ValidationDataSet
from .export import export_png, copy_to_clipboard, export_all_charts

from .chart_stacked_band import render_stacked_band
from .chart_heatmap import render_heatmap
from .chart_scatter import render_scatter
from .chart_histogram import render_histogram
from .chart_margin_trend import render_margin_trend
from .chart_dominant_source import render_dominant_source


class _ChartTab(QWidget):
    """Single chart tab with figure canvas, toolbar, and export buttons."""

    def __init__(self, figsize=(6, 4), parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # ── Toolbar row ──────────────────────────────────────────────
        toolbar_row = QHBoxLayout()
        toolbar_row.setSpacing(4)

        self._fig = Figure(figsize=figsize)
        self._fig.set_facecolor(DARK_COLORS['bg_alt'])
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)

        toolbar_row.addWidget(self._toolbar)
        toolbar_row.addStretch()

        # Copy buttons
        self._btn_copy = QPushButton("Copy to Clipboard")
        self._btn_copy.setFixedHeight(28)
        self._btn_copy.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        self._btn_copy.clicked.connect(lambda *_: self._on_copy())
        toolbar_row.addWidget(self._btn_copy)

        self._btn_export = QPushButton("Export PNG...")
        self._btn_export.setFixedHeight(28)
        self._btn_export.setStyleSheet("font-size: 11px; padding: 2px 8px;")
        self._btn_export.clicked.connect(lambda *_: self._on_export())
        toolbar_row.addWidget(self._btn_export)

        layout.addLayout(toolbar_row)

        # ── Canvas ───────────────────────────────────────────────────
        layout.addWidget(self._canvas, 1)

    @property
    def fig(self) -> Figure:
        return self._fig

    @property
    def canvas(self) -> FigureCanvas:
        return self._canvas

    def refresh(self):
        """Redraw the canvas after figure changes."""
        self._canvas.draw_idle()

    def _on_copy(self):
        success = copy_to_clipboard(self._fig)
        if success:
            self.window().statusBar().showMessage(
                "Chart copied to clipboard", 3000
            )
        else:
            QMessageBox.warning(self, "Copy Failed",
                                "Could not copy chart to clipboard.")

    def _on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Chart as PNG",
            "", "PNG Files (*.png);;All Files (*)",
        )
        if path:
            if not path.lower().endswith('.png'):
                path += '.png'
            try:
                export_png(self._fig, path)
                self.window().statusBar().showMessage(
                    f"Exported to {os.path.basename(path)}", 3000
                )
            except Exception as exc:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export: {exc}"
                )


class ChartTabsWidget(QTabWidget):
    """Tabbed container for all 6 chart types with matplotlib canvases."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self._dataset = None
        self._config = {}

        # Create the 6 tabs
        self._tab_band = _ChartTab(figsize=(8, 5))
        self._tab_heatmap = _ChartTab(figsize=(7, 5))
        self._tab_scatter = _ChartTab(figsize=(6, 5))
        self._tab_histogram = _ChartTab(figsize=(6, 4))
        self._tab_margin = _ChartTab(figsize=(7, 5))
        self._tab_dominant = _ChartTab(figsize=(7, 5))

        self.addTab(self._tab_band, "Stacked Band")
        self.addTab(self._tab_heatmap, "Validation Heatmap")
        self.addTab(self._tab_scatter, "E vs U_val Scatter")
        self.addTab(self._tab_histogram, "|E| − U_val Histogram")
        self.addTab(self._tab_margin, "Margin Ratio Trend")
        self.addTab(self._tab_dominant, "Dominant Source")

        # Apply dark plot style
        apply_plot_style(PLOT_STYLE_DARK)

    def update_all_charts(
        self,
        dataset: ValidationDataSet,
        config: dict,
        selected_category: str = "",
    ) -> None:
        """Re-render all charts with current data and config.

        Parameters
        ----------
        dataset : ValidationDataSet
        config : dict
            From ``ConfigPanel.get_config()``.
        selected_category : str
            Category name to display. If empty, uses first category.
        """
        self._dataset = dataset
        self._config = config

        if not dataset or not dataset.categories:
            return

        # Find selected category
        cat_data = None
        for cat in dataset.categories:
            if cat.category == selected_category:
                cat_data = cat
                break
        if cat_data is None:
            cat_data = dataset.categories[0]

        # Ensure dark style is applied for GUI rendering
        apply_plot_style(PLOT_STYLE_DARK)

        self._update_stacked_band(cat_data, config)
        self._update_heatmap(cat_data, config)
        self._update_scatter(cat_data, config)
        self._update_histogram(cat_data, config)
        self._update_margin_trend(cat_data, config)
        self._update_dominant_source(cat_data, config)

    def _update_stacked_band(self, cat_data: CategoryData, config: dict):
        fig = self._tab_band.fig
        render_stacked_band(
            fig, cat_data,
            k_factor=config.get('k_factor', 1.0),
            already_expanded=config.get('already_expanded', True),
            concern_direction=config.get('concern_direction', 'both'),
            no_data_label=config.get('no_data_label', 'No Test Data Available'),
            show_val_requirement=config.get('show_val_requirement', False),
            val_requirement_value=config.get('val_requirement_value', 0.0),
            units=config.get('units', ''),
            for_export=False,
            bias_included=config.get('bias_included'),
        )
        self._tab_band.refresh()

    def _update_heatmap(self, cat_data: CategoryData, config: dict):
        fig = self._tab_heatmap.fig
        render_heatmap(
            fig, cat_data,
            for_export=False,
            bias_included=config.get('bias_included'),
        )
        self._tab_heatmap.refresh()

    def _update_scatter(self, cat_data: CategoryData, config: dict):
        fig = self._tab_scatter.fig
        render_scatter(
            fig, cat_data,
            units=config.get('units', ''),
            for_export=False,
            bias_included=config.get('bias_included'),
        )
        self._tab_scatter.refresh()

    def _update_histogram(self, cat_data: CategoryData, config: dict):
        fig = self._tab_histogram.fig
        render_histogram(
            fig, cat_data,
            units=config.get('units', ''),
            for_export=False,
            bias_included=config.get('bias_included'),
        )
        self._tab_histogram.refresh()

    def _update_margin_trend(self, cat_data: CategoryData, config: dict):
        fig = self._tab_margin.fig
        render_margin_trend(
            fig, cat_data,
            units=config.get('units', ''),
            for_export=False,
            bias_included=config.get('bias_included'),
        )
        self._tab_margin.refresh()

    def _update_dominant_source(self, cat_data: CategoryData, config: dict):
        fig = self._tab_dominant.fig
        render_dominant_source(
            fig, cat_data,
            for_export=False,
            bias_included=config.get('bias_included'),
        )
        self._tab_dominant.refresh()

    def get_all_figures(self, dataset: ValidationDataSet, config: dict) -> dict:
        """Generate figures for ALL categories and chart types for batch export.

        Returns dict of ``{filename_stem: Figure}``.
        """
        figures = {}
        bias = config.get('bias_included')

        for cat_data in dataset.categories:
            safe_cat = cat_data.category.replace(' ', '_')

            # Stacked band
            fig_band = Figure(figsize=(8, 5))
            render_stacked_band(
                fig_band, cat_data,
                k_factor=config.get('k_factor', 1.0),
                already_expanded=config.get('already_expanded', True),
                concern_direction=config.get('concern_direction', 'both'),
                no_data_label=config.get('no_data_label',
                                         'No Test Data Available'),
                show_val_requirement=config.get('show_val_requirement', False),
                val_requirement_value=config.get('val_requirement_value', 0.0),
                units=config.get('units', ''),
                for_export=True,
                bias_included=bias,
            )
            figures[f"{safe_cat}_stacked_band"] = fig_band

            # Heatmap
            fig_hm = Figure(figsize=(7, 5))
            render_heatmap(fig_hm, cat_data,
                           for_export=True, bias_included=bias)
            figures[f"{safe_cat}_validation_heatmap"] = fig_hm

            # Scatter
            fig_sc = Figure(figsize=(6, 5))
            render_scatter(
                fig_sc, cat_data,
                units=config.get('units', ''),
                for_export=True, bias_included=bias,
            )
            figures[f"{safe_cat}_scatter"] = fig_sc

            # Histogram
            fig_hist = Figure(figsize=(6, 4))
            render_histogram(
                fig_hist, cat_data,
                units=config.get('units', ''),
                for_export=True, bias_included=bias,
            )
            figures[f"{safe_cat}_histogram"] = fig_hist

            # Margin trend
            fig_mt = Figure(figsize=(7, 5))
            render_margin_trend(
                fig_mt, cat_data,
                units=config.get('units', ''),
                for_export=True, bias_included=bias,
            )
            figures[f"{safe_cat}_margin_trend"] = fig_mt

            # Dominant source
            fig_ds = Figure(figsize=(7, 5))
            render_dominant_source(fig_ds, cat_data,
                                   for_export=True, bias_included=bias)
            figures[f"{safe_cat}_dominant_source"] = fig_ds

        return figures
