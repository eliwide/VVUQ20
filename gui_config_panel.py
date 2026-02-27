"""
Configuration panel (left side) for the VVUQ Validation Plotter.

File inputs, coverage factor, units, concern direction, validation
requirement, category selection, and the Generate button.
"""

import os

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QLabel, QPushButton, QLineEdit, QComboBox, QCheckBox,
    QRadioButton, QButtonGroup, QDoubleSpinBox, QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Signal

from .constants import (
    DARK_COLORS, UNIT_OPTIONS, DEFAULT_NO_DATA_LABEL,
    DEFAULT_K_FACTOR, DEFAULT_UNITS, DEFAULT_CONCERN_DIRECTION,
)
from .csv_parser import load_validation_csvs
from .data_model import ValidationDataSet


class ConfigPanel(QWidget):
    """Left-side configuration panel with file inputs and plot options."""

    # Signals
    config_changed = Signal()
    files_loaded = Signal(object)  # emits ValidationDataSet

    # File type keys in display order
    _FILE_TYPES = [
        ('E', 'E (Comparison Error)'),
        ('u_num', 'u_num (Numerical Unc.)'),
        ('u_input', 'u_input (Input Unc.)'),
        ('u_d', 'u_d (Experimental Unc.)'),
        ('U_val', 'U_val (Validation Unc.)'),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._file_paths = {key: '' for key, _ in self._FILE_TYPES}
        self._dataset = None
        self._setup_ui()
        self._connect_signals()

    # ── UI setup ─────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # ── Group 1: Data Files ──────────────────────────────────────
        grp_files = QGroupBox("Data Files")
        files_layout = QVBoxLayout(grp_files)
        files_layout.setSpacing(4)

        self._file_edits = {}
        self._file_buttons = {}

        for key, label in self._FILE_TYPES:
            row = QHBoxLayout()
            row.setSpacing(4)
            lbl = QLabel(label)
            lbl.setFixedWidth(160)
            lbl.setStyleSheet(f"font-size: 11px;")
            edit = QLineEdit()
            edit.setReadOnly(True)
            edit.setPlaceholderText("No file selected")
            edit.setStyleSheet("font-size: 11px;")
            btn = QPushButton("Browse...")
            btn.setFixedWidth(70)
            btn.setStyleSheet("font-size: 11px;")
            row.addWidget(lbl)
            row.addWidget(edit, 1)
            row.addWidget(btn)
            files_layout.addLayout(row)
            self._file_edits[key] = edit
            self._file_buttons[key] = btn

        # Load All from Folder button
        self._btn_load_folder = QPushButton("Load All from Folder...")
        self._btn_load_folder.setToolTip(
            "Select a folder containing E.csv, u_num.csv, u_input.csv, "
            "u_d.csv, and U_val.csv"
        )
        files_layout.addWidget(self._btn_load_folder)

        # Status label
        self._lbl_file_status = QLabel("")
        self._lbl_file_status.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; font-size: 11px;"
        )
        self._lbl_file_status.setWordWrap(True)
        files_layout.addWidget(self._lbl_file_status)

        layout.addWidget(grp_files)

        # ── Group 2: Coverage Factor ─────────────────────────────────
        grp_coverage = QGroupBox("Coverage Factor")
        cov_layout = QVBoxLayout(grp_coverage)
        cov_layout.setSpacing(4)

        self._chk_expanded = QCheckBox("Values already in expanded terms")
        self._chk_expanded.setChecked(True)
        cov_layout.addWidget(self._chk_expanded)

        k_row = QHBoxLayout()
        self._lbl_k = QLabel("k =")
        self._spn_k = QDoubleSpinBox()
        self._spn_k.setRange(1.0, 4.0)
        self._spn_k.setSingleStep(0.1)
        self._spn_k.setDecimals(1)
        self._spn_k.setValue(DEFAULT_K_FACTOR)
        self._spn_k.setVisible(False)
        self._lbl_k.setVisible(False)
        k_row.addWidget(self._lbl_k)
        k_row.addWidget(self._spn_k)
        k_row.addStretch()
        cov_layout.addLayout(k_row)

        layout.addWidget(grp_coverage)

        # ── Group 3: Units & Labels ──────────────────────────────────
        grp_units = QGroupBox("Units & Labels")
        units_layout = QFormLayout(grp_units)
        units_layout.setSpacing(4)

        self._cmb_units = QComboBox()
        self._cmb_units.addItems(UNIT_OPTIONS)
        self._cmb_units.setCurrentText(DEFAULT_UNITS)
        units_layout.addRow("Units:", self._cmb_units)

        self._edt_no_data = QLineEdit(DEFAULT_NO_DATA_LABEL)
        units_layout.addRow("No-data label:", self._edt_no_data)

        layout.addWidget(grp_units)

        # ── Group 4: Concern Direction ───────────────────────────────
        grp_direction = QGroupBox("Concern Direction")
        dir_layout = QVBoxLayout(grp_direction)
        dir_layout.setSpacing(2)

        self._btn_group_direction = QButtonGroup(self)
        self._rb_under = QRadioButton("Under-prediction")
        self._rb_over = QRadioButton("Over-prediction")
        self._rb_both = QRadioButton("Both")
        self._rb_both.setChecked(True)

        self._btn_group_direction.addButton(self._rb_under, 0)
        self._btn_group_direction.addButton(self._rb_over, 1)
        self._btn_group_direction.addButton(self._rb_both, 2)

        dir_layout.addWidget(self._rb_under)
        dir_layout.addWidget(self._rb_over)
        dir_layout.addWidget(self._rb_both)

        layout.addWidget(grp_direction)

        # ── Group 4b: Bias Inclusion ───────────────────────────────
        grp_bias = QGroupBox("Bias Annotation")
        bias_layout = QFormLayout(grp_bias)
        bias_layout.setSpacing(4)

        self._cmb_bias = QComboBox()
        self._cmb_bias.addItems([
            "(none)",
            "Includes additional bias",
            "Does not include additional bias",
        ])
        self._cmb_bias.setCurrentIndex(0)
        self._cmb_bias.setToolTip(
            "Adds a bias note to chart titles.\n"
            "Select '(none)' to hide the note."
        )
        bias_layout.addRow("Title note:", self._cmb_bias)

        layout.addWidget(grp_bias)

        # ── Group 5: Validation Requirement ──────────────────────────
        grp_val = QGroupBox("Validation Requirement")
        val_layout = QVBoxLayout(grp_val)
        val_layout.setSpacing(4)

        self._chk_val_req = QCheckBox("Show requirement line")
        self._chk_val_req.setChecked(False)
        val_layout.addWidget(self._chk_val_req)

        val_row = QHBoxLayout()
        self._lbl_val = QLabel("Value:")
        self._spn_val = QDoubleSpinBox()
        self._spn_val.setRange(-9999.0, 9999.0)
        self._spn_val.setSingleStep(0.1)
        self._spn_val.setDecimals(4)
        self._spn_val.setValue(0.0)
        self._spn_val.setVisible(False)
        self._lbl_val.setVisible(False)
        val_row.addWidget(self._lbl_val)
        val_row.addWidget(self._spn_val)
        val_row.addStretch()
        val_layout.addLayout(val_row)

        layout.addWidget(grp_val)

        # ── Group 6: Category Selection ──────────────────────────────
        grp_cat = QGroupBox("Category")
        cat_layout = QVBoxLayout(grp_cat)
        cat_layout.setSpacing(4)

        self._cmb_category = QComboBox()
        self._cmb_category.addItem("(load data first)")
        self._cmb_category.setEnabled(False)
        cat_layout.addWidget(self._cmb_category)

        layout.addWidget(grp_cat)

        # ── Actions ──────────────────────────────────────────────────
        c = DARK_COLORS
        self._btn_generate = QPushButton("Generate All Charts")
        self._btn_generate.setStyleSheet(
            f"QPushButton {{ background-color: {c['accent']}; "
            f"color: {c['bg']}; font-weight: bold; "
            f"font-size: 14px; padding: 10px; }}"
            f"QPushButton:hover {{ background-color: {c['accent_hover']}; }}"
            f"QPushButton:disabled {{ background-color: {c['bg']}; "
            f"color: {c['fg_dim']}; }}"
        )
        self._btn_generate.setEnabled(False)
        layout.addWidget(self._btn_generate)

        self._btn_export_all = QPushButton("Export All Charts (All Categories)...")
        self._btn_export_all.setStyleSheet(
            f"QPushButton {{ background-color: {c['surface0']}; "
            f"font-size: 12px; padding: 8px; }}"
            f"QPushButton:hover {{ background-color: {c['overlay0']}; }}"
            f"QPushButton:disabled {{ background-color: {c['bg']}; "
            f"color: {c['fg_dim']}; }}"
        )
        self._btn_export_all.setEnabled(False)
        layout.addWidget(self._btn_export_all)

        self._btn_example = QPushButton("Load Example Data")
        layout.addWidget(self._btn_example)

        layout.addStretch()

    # ── Signal connections ───────────────────────────────────────────

    def _connect_signals(self):
        # Browse buttons
        for key, _ in self._FILE_TYPES:
            btn = self._file_buttons[key]
            # Use default argument to capture key in closure
            btn.clicked.connect(lambda checked=False, k=key: self._browse_file(k))

        self._btn_load_folder.clicked.connect(self._load_from_folder)
        self._btn_example.clicked.connect(self._load_example)

        # Coverage factor toggle
        self._chk_expanded.toggled.connect(self._on_expanded_toggled)

        # Validation requirement toggle
        self._chk_val_req.toggled.connect(self._on_val_req_toggled)

        # Config change signals — use lambdas to absorb the argument
        # that each signal passes (bool, int, float, str) since
        # config_changed is Signal() with no arguments.
        self._cmb_units.currentIndexChanged.connect(
            lambda *_: self.config_changed.emit()
        )
        self._edt_no_data.textChanged.connect(
            lambda *_: self.config_changed.emit()
        )
        self._btn_group_direction.idToggled.connect(
            lambda *_: self.config_changed.emit()
        )
        self._chk_expanded.toggled.connect(
            lambda *_: self.config_changed.emit()
        )
        self._spn_k.valueChanged.connect(
            lambda *_: self.config_changed.emit()
        )
        self._chk_val_req.toggled.connect(
            lambda *_: self.config_changed.emit()
        )
        self._spn_val.valueChanged.connect(
            lambda *_: self.config_changed.emit()
        )
        self._cmb_bias.currentIndexChanged.connect(
            lambda *_: self.config_changed.emit()
        )

    # ── Slot implementations ─────────────────────────────────────────

    def _on_expanded_toggled(self, checked):
        self._lbl_k.setVisible(not checked)
        self._spn_k.setVisible(not checked)

    def _on_val_req_toggled(self, checked):
        self._lbl_val.setVisible(checked)
        self._spn_val.setVisible(checked)

    def _browse_file(self, file_key):
        path, _ = QFileDialog.getOpenFileName(
            self, f"Select {file_key} CSV File",
            "", "CSV Files (*.csv);;All Files (*)",
        )
        if path:
            self._file_paths[file_key] = path
            self._file_edits[file_key].setText(os.path.basename(path))
            self._file_edits[file_key].setToolTip(path)
            self._try_load()

    def _load_from_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Folder Containing CSV Files",
        )
        if not folder:
            return

        # GPT #3 fix: clear ALL stale paths first to prevent mixing
        # old and new files if the new folder is incomplete.
        for key in self._file_paths:
            self._file_paths[key] = ''
            self._file_edits[key].setText('')
            self._file_edits[key].setToolTip('')

        expected = {
            'E': 'E.csv',
            'u_num': 'u_num.csv',
            'u_input': 'u_input.csv',
            'u_d': 'u_d.csv',
            'U_val': 'U_val.csv',
        }
        missing = []
        for key, filename in expected.items():
            path = os.path.join(folder, filename)
            if os.path.isfile(path):
                self._file_paths[key] = path
                self._file_edits[key].setText(filename)
                self._file_edits[key].setToolTip(path)
            else:
                missing.append(filename)

        if missing:
            # Disable buttons since we cleared all paths (GPT #7)
            self._btn_generate.setEnabled(False)
            self._btn_export_all.setEnabled(False)
            self._dataset = None
            QMessageBox.warning(
                self, "Missing Files",
                f"The following expected files were not found in "
                f"'{os.path.basename(folder)}':\n\n"
                + "\n".join(f"  - {f}" for f in missing)
                + "\n\nPlease browse for these files individually.",
            )
        else:
            self._try_load()

    def _load_example(self):
        """Generate and load example data."""
        from .example_data import generate_example_csvs
        import tempfile

        example_dir = os.path.join(
            tempfile.gettempdir(), 'vvuq_plotter_example'
        )
        paths = generate_example_csvs(example_dir)

        for key in self._file_paths:
            if key in paths:
                self._file_paths[key] = paths[key]
                self._file_edits[key].setText(os.path.basename(paths[key]))
                self._file_edits[key].setToolTip(paths[key])

        self._try_load()

    def _try_load(self):
        """Attempt to load and validate all five CSVs."""
        # Check all files are specified
        missing = [
            label for key, label in self._FILE_TYPES
            if not self._file_paths[key]
        ]
        if missing:
            self._lbl_file_status.setText(
                f"Still need: {', '.join(missing)}"
            )
            self._lbl_file_status.setStyleSheet(
                f"color: {DARK_COLORS['yellow']}; font-size: 11px;"
            )
            # GPT #7: disable action buttons when files are missing
            self._btn_generate.setEnabled(False)
            self._btn_export_all.setEnabled(False)
            return

        try:
            dataset = load_validation_csvs(
                self._file_paths['E'],
                self._file_paths['u_num'],
                self._file_paths['u_input'],
                self._file_paths['u_d'],
                self._file_paths['U_val'],
            )
        except (ValueError, FileNotFoundError, OSError) as exc:
            self._lbl_file_status.setText(f"Error: {exc}")
            self._lbl_file_status.setStyleSheet(
                f"color: {DARK_COLORS['red']}; font-size: 11px;"
            )
            self._btn_generate.setEnabled(False)
            self._btn_export_all.setEnabled(False)
            self._dataset = None
            QMessageBox.critical(self, "Data Load Error", str(exc))
            return

        self._dataset = dataset

        # Update category combo — preserve selection (checklist §5)
        prev_text = self._cmb_category.currentText()
        self._cmb_category.blockSignals(True)
        self._cmb_category.clear()
        cat_names = [c.category for c in dataset.categories]
        self._cmb_category.addItems(cat_names)
        # Restore previous selection if still valid
        idx = self._cmb_category.findText(prev_text)
        if idx >= 0:
            self._cmb_category.setCurrentIndex(idx)
        self._cmb_category.blockSignals(False)
        self._cmb_category.setEnabled(True)

        # Update status
        n_cats = len(dataset.categories)
        n_conds = len(dataset.all_conditions)
        total_sensors = sum(len(c.sensors) for c in dataset.categories)
        self._lbl_file_status.setText(
            f"Loaded: {n_cats} categories, {n_conds} conditions, "
            f"{total_sensors} sensors"
        )
        self._lbl_file_status.setStyleSheet(
            f"color: {DARK_COLORS['green']}; font-size: 11px;"
        )

        self._btn_generate.setEnabled(True)
        self._btn_export_all.setEnabled(True)
        self.files_loaded.emit(dataset)

    # ── Public API ───────────────────────────────────────────────────

    def get_config(self) -> dict:
        """Return current configuration as a dict for chart renderers."""
        direction_map = {0: "under", 1: "over", 2: "both"}
        checked_id = self._btn_group_direction.checkedId()

        return {
            'k_factor': self._spn_k.value(),
            'already_expanded': self._chk_expanded.isChecked(),
            'concern_direction': direction_map.get(checked_id, "both"),
            'no_data_label': self._edt_no_data.text() or DEFAULT_NO_DATA_LABEL,
            'show_val_requirement': self._chk_val_req.isChecked(),
            'val_requirement_value': self._spn_val.value(),
            'units': self._cmb_units.currentText(),
            'selected_category': self._cmb_category.currentText(),
            'bias_included': {0: None, 1: True, 2: False}.get(
                self._cmb_bias.currentIndex()
            ),
        }

    def get_dataset(self) -> ValidationDataSet:
        """Return the currently loaded dataset, or ``None``."""
        return self._dataset

    @property
    def generate_button(self) -> QPushButton:
        """Access to the Generate button for external signal connection."""
        return self._btn_generate

    @property
    def export_all_button(self) -> QPushButton:
        """Access to the Export All button for external signal connection."""
        return self._btn_export_all

    @property
    def category_combo(self) -> QComboBox:
        """Access to the category combo for external signal connection."""
        return self._cmb_category
