"""
Main window for the VVUQ Validation Plotter.

Hosts the ConfigPanel (left) and ChartTabsWidget (right) in a
horizontal splitter, with a menu bar and status bar.
"""

import os

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QSplitter, QScrollArea,
    QFileDialog, QMessageBox, QStatusBar, QMenuBar,
)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt

from . import APP_NAME, APP_VERSION
from .constants import DARK_COLORS
from .data_model import ValidationDataSet
from .export import export_all_charts
from .gui_config_panel import ConfigPanel
from .gui_chart_tabs import ChartTabsWidget


class PlotterMainWindow(QMainWindow):
    """Main window for the VVUQ Validation Plotter."""

    def __init__(self):
        super().__init__()
        self._dataset = None

        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1200, 800)

        self._setup_ui()
        self._setup_menu()
        self._connect_signals()

        self.statusBar().showMessage("Ready — load CSV files to begin")

    # ── UI setup ─────────────────────────────────────────────────────

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)
        main_layout.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: config in scroll area
        self._config_panel = ConfigPanel()
        scroll = QScrollArea()
        scroll.setWidget(self._config_panel)
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(320)
        scroll.setMaximumWidth(500)

        # Right panel: chart tabs
        self._chart_tabs = ChartTabsWidget()

        splitter.addWidget(scroll)
        splitter.addWidget(self._chart_tabs)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 820])

        main_layout.addWidget(splitter)

    def _setup_menu(self):
        menubar = self.menuBar()

        # ── File menu ────────────────────────────────────────────────
        file_menu = menubar.addMenu("File")

        act_load_folder = QAction("Load Folder...", self)
        act_load_folder.triggered.connect(
            lambda *_: self._config_panel._load_from_folder()
        )
        file_menu.addAction(act_load_folder)

        file_menu.addSeparator()

        act_export_current = QAction("Export Current Chart...", self)
        act_export_current.triggered.connect(
            lambda *_: self._export_current()
        )
        file_menu.addAction(act_export_current)

        act_export_all = QAction("Export All Charts...", self)
        act_export_all.triggered.connect(lambda *_: self._export_all())
        file_menu.addAction(act_export_all)

        file_menu.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # ── Examples menu ────────────────────────────────────────────
        examples_menu = menubar.addMenu("Examples")

        act_load_example = QAction("Load Example Dataset", self)
        act_load_example.triggered.connect(
            lambda *_: self._config_panel._load_example()
        )
        examples_menu.addAction(act_load_example)

        # ── Help menu ────────────────────────────────────────────────
        help_menu = menubar.addMenu("Help")

        act_about = QAction("About", self)
        act_about.triggered.connect(lambda *_: self._show_about())
        help_menu.addAction(act_about)

    def _connect_signals(self):
        self._config_panel.files_loaded.connect(self._on_files_loaded)
        # Use lambda wrappers so the bool argument from clicked(bool)
        # is safely absorbed (same pattern as config_changed fix).
        self._config_panel.generate_button.clicked.connect(
            lambda *_: self._on_generate()
        )
        self._config_panel.export_all_button.clicked.connect(
            lambda *_: self._export_all()
        )
        self._config_panel.category_combo.currentTextChanged.connect(
            self._on_category_changed
        )

    # ── Slots ────────────────────────────────────────────────────────

    def _on_files_loaded(self, dataset):
        """Slot: called when CSV files are successfully loaded.

        GPT #4: if dataset is None (load failed), clear stale state.
        """
        self._dataset = dataset
        if dataset is None:
            self.statusBar().showMessage("Data load cleared")
            return
        n_cats = len(dataset.categories)
        n_conds = len(dataset.all_conditions)
        total_sensors = sum(len(c.sensors) for c in dataset.categories)
        self.statusBar().showMessage(
            f"Loaded: {n_cats} categories, {n_conds} conditions, "
            f"{total_sensors} sensors"
        )
        # Auto-generate charts with new data (clears stale charts)
        self._on_generate()

    def _on_generate(self):
        """Slot: Generate All Charts button clicked."""
        dataset = self._config_panel.get_dataset()
        if dataset is None:
            QMessageBox.warning(
                self, "No Data",
                "Please load CSV files before generating charts.",
            )
            return

        config = self._config_panel.get_config()
        selected = config.get('selected_category', '')

        self.statusBar().showMessage("Generating charts...")
        try:
            self._chart_tabs.update_all_charts(dataset, config, selected)
            self.statusBar().showMessage("Charts generated", 5000)
        except Exception as exc:
            QMessageBox.critical(
                self, "Chart Generation Error",
                f"An error occurred while generating charts:\n\n{exc}",
            )
            self.statusBar().showMessage("Chart generation failed")

    def _on_category_changed(self, category_text):
        """Slot: category dropdown changed — re-render if data is loaded."""
        if self._dataset is None:
            return
        config = self._config_panel.get_config()
        try:
            self._chart_tabs.update_all_charts(
                self._dataset, config, category_text
            )
        except Exception as exc:
            # During data loading transitions the combo may briefly hold
            # stale text.  Log but do not pop up a dialog.
            import sys
            print(f"[VVUQ] Category change render warning: {exc}",
                  file=sys.stderr)

    def _export_current(self):
        """Export the currently visible chart tab as PNG."""
        current_tab = self._chart_tabs.currentWidget()
        if current_tab is None or not hasattr(current_tab, 'fig'):
            QMessageBox.warning(
                self, "Nothing to Export",
                "No chart is currently displayed.",
            )
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Chart as PNG",
            "", "PNG Files (*.png);;All Files (*)",
        )
        if path:
            if not path.lower().endswith('.png'):
                path += '.png'
            try:
                from .export import export_png
                export_png(current_tab.fig, path)
                self.statusBar().showMessage(
                    f"Exported to {os.path.basename(path)}", 5000
                )
            except Exception as exc:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export: {exc}"
                )

    def _export_all(self):
        """Export all charts for all categories to a folder."""
        dataset = self._config_panel.get_dataset()
        if dataset is None:
            QMessageBox.warning(
                self, "No Data",
                "Please load CSV files and generate charts first.",
            )
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder for All Charts"
        )
        if not folder:
            return

        config = self._config_panel.get_config()
        self.statusBar().showMessage("Exporting all charts...")

        try:
            figures = self._chart_tabs.get_all_figures(dataset, config)
            paths = export_all_charts(figures, folder)
            self.statusBar().showMessage(
                f"Exported {len(paths)} charts to {os.path.basename(folder)}",
                5000,
            )
            QMessageBox.information(
                self, "Export Complete",
                f"Successfully exported {len(paths)} charts to:\n\n"
                f"{folder}",
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export charts:\n\n{exc}",
            )

    def _show_about(self):
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<h3>{APP_NAME} v{APP_VERSION}</h3>"
            f"<p>Publication-quality plotting tool for ASME V&V 20 "
            f"validation analysis.</p>"
            f"<p>Generates stacked band charts, validation heatmaps, "
            f"scatter plots, histograms, margin ratio trends, and "
            f"dominant uncertainty source maps.</p>"
            f"<p>Boeing-branded colour palette for internal reports "
            f"and regulatory submittals.</p>",
        )
