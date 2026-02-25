#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GCI Calculator v1.5.0
Grid Convergence Index Tool per Celik et al. (JFE 2008) & Roache (1998)

Standalone PySide6 application for computing Grid Convergence Index (GCI)
from 2, 3, or N-grid refinement studies.  Computes Richardson extrapolation,
observed order of accuracy, asymptotic range checks, and converts GCI to
a standard uncertainty u_num for use in ASME V&V 20 uncertainty budgets.

Standards References:
    - ASME V&V 20-2009 (R2021) Section 5.1 — Numerical Uncertainty
    - Celik et al. (2008) "Procedure for Estimation and Reporting of
      Uncertainty Due to Discretization in CFD Applications"
      J. Fluids Eng. 130(7), 078001
    - Roache, P.J. (1998) "Verification and Validation in Computational
      Science and Engineering" Hermosa Publishers
    - Richardson, L.F. (1911) "The Approximate Arithmetical Solution by
      Finite Differences of Physical Problems"
    - ITTC (2024) "Uncertainty Analysis in CFD Verification and Validation"
      Procedure 7.5-03-01-01

Copyright (c) 2026. All rights reserved.
"""

import sys
import os
import json
import io
import base64
import textwrap
import datetime
import tempfile
import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, List, Tuple
from html import escape as _html_esc

import numpy as np

# Force matplotlib to use PySide6
os.environ["QT_API"] = "pyside6"
if "MPLCONFIGDIR" not in os.environ:
    try:
        mpl_cache_dir = os.path.join(tempfile.gettempdir(), "gci_calculator_mplconfig")
        os.makedirs(mpl_cache_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_cache_dir
    except OSError:
        pass

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QGroupBox, QLabel, QPushButton, QLineEdit,
    QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QRadioButton, QButtonGroup, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QScrollArea, QFrame, QMessageBox,
    QAbstractItemView, QFileDialog, QStatusBar, QStackedWidget,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
# ═══════════════════════════════════════════════════════════════════════════
# SECTION: SHARED VVUQ REPORT COMPONENTS
# Keep shared data (constants, terms, renderer) consistent across all four
# VVUQ tools (Uncertainty Aggregator, GCI Calculator, Iterative Uncertainty,
# Statistical Analyzer).
# Import line and data definitions below must match across all four files.
# Last synchronized: 2026-02-23
# ═══════════════════════════════════════════════════════════════════════════

DECISION_CONSEQUENCE_LEVELS = ("Low", "Medium", "High")
DEFAULT_DECISION_CONSEQUENCE = "Medium"

VVUQ_TERMS: list = [
    ("Verification",
     "Checks whether the numerical model is solved correctly for the implemented equations."),
    ("Validation",
     "Checks whether the model predicts reality with acceptable uncertainty for the intended use."),
    ("Aleatory uncertainty",
     "Irreducible variability in the system or environment (e.g., natural scatter)."),
    ("Epistemic uncertainty",
     "Reducible uncertainty caused by limited knowledge, sparse data, or model assumptions."),
    ("Standard uncertainty (1-sigma)",
     "The uncertainty value carried into combination calculations before coverage expansion."),
    ("Expanded uncertainty",
     "Standard uncertainty multiplied by k-factor to meet a target coverage/confidence."),
    ("Model-form uncertainty",
     "Uncertainty from missing or imperfect physics, closures, or assumptions in the model."),
    ("Carry-Over value",
     "Single uncertainty value selected for transfer into the aggregator workflow."),
]

_CHECK_LABELS = {
    "inputs_documented": "Inputs and assumptions are documented",
    "method_selected": "Method choice is justified and traceable",
    "units_consistent": "Units are consistent across all sources",
    "data_quality": "Data quantity/quality checks passed",
    "diagnostics_pass": "Diagnostics passed (fit, stationarity, convergence, etc.)",
    "independent_review": "Independent technical review completed",
    "conservative_bound": "Conservative bound or margin policy applied",
    "validation_plan": "Validation closure plan is defined",
}

_REQUIRED_BY_LEVEL = {
    "Low": (
        "inputs_documented",
        "method_selected",
        "units_consistent",
    ),
    "Medium": (
        "inputs_documented",
        "method_selected",
        "units_consistent",
        "data_quality",
        "diagnostics_pass",
    ),
    "High": (
        "inputs_documented",
        "method_selected",
        "units_consistent",
        "data_quality",
        "diagnostics_pass",
        "independent_review",
        "conservative_bound",
        "validation_plan",
    ),
}


class CredibilityEvaluation:
    """Result of deterministic credibility checklist evaluation."""
    __slots__ = ("consequence", "passed", "required_checks", "missing_labels")

    def __init__(self, consequence, passed, required_checks, missing_labels):
        self.consequence = consequence
        self.passed = passed
        self.required_checks = required_checks
        self.missing_labels = missing_labels


def normalize_decision_consequence(value: str) -> str:
    """Normalize user text to one of Low/Medium/High."""
    txt = (value or "").strip().lower()
    if txt.startswith("h"):
        return "High"
    if txt.startswith("l"):
        return "Low"
    return "Medium"


def evaluate_credibility(consequence, evidence):
    """Evaluate deterministic credibility checklist for a decision level."""
    level = normalize_decision_consequence(consequence)
    required = _REQUIRED_BY_LEVEL[level]
    required_checks = []
    missing = []
    for key in required:
        ok = bool(evidence.get(key, False))
        label = _CHECK_LABELS.get(key, key)
        required_checks.append((label, ok))
        if not ok:
            missing.append(label)
    return CredibilityEvaluation(
        consequence=level,
        passed=(len(missing) == 0),
        required_checks=required_checks,
        missing_labels=missing,
    )


def render_credibility_html(consequence, evidence, section_id="section-credibility"):
    """Render credibility section HTML with pass/fail banner and checklist."""
    ev = evaluate_credibility(consequence, evidence)
    if ev.passed:
        verdict_cls = "pass"
        verdict_text = f"Minimum evidence is met for {ev.consequence.lower()}-consequence use."
    else:
        verdict_cls = "fail"
        verdict_text = f"Minimum evidence is NOT met for {ev.consequence.lower()}-consequence use."
    items = []
    for label, ok in ev.required_checks:
        mark = "PASS" if ok else "MISSING"
        items.append(f"<li><strong>{mark}</strong> - {_html_esc(label)}</li>")
    missing_html = ""
    if ev.missing_labels:
        missing_lines = "".join(f"<li>{_html_esc(lbl)}</li>" for lbl in ev.missing_labels)
        missing_html = f"<h3>Required Before Release</h3><ul>{missing_lines}</ul>"
    return (
        f'<div class="section" id="{_html_esc(section_id)}">'
        "<h2>Credibility Framing</h2>"
        f"<p><strong>Decision consequence:</strong> {_html_esc(ev.consequence)}</p>"
        f'<div class="verdict {verdict_cls}">{_html_esc(verdict_text)}</div>'
        "<h3>Checklist</h3>"
        f"<ul>{''.join(items)}</ul>"
        f"{missing_html}"
        "</div>"
    )


def render_decision_card_html(title, use_value, use_distribution, use_combination,
                              stop_checks, notes=""):
    """Render novice-first final decision card."""
    checks = list(stop_checks)
    stop_list = "".join(f"<li>{_html_esc(x)}</li>" for x in checks)
    notes_html = f"<p><strong>Why:</strong> {_html_esc(notes)}</p>" if notes else ""
    return (
        '<div class="section">'
        f"<h2>{_html_esc(title)}</h2>"
        '<div class="highlight">'
        f"<p><strong>Use this value:</strong> {_html_esc(use_value)}</p>"
        f"<p><strong>Use this distribution:</strong> {_html_esc(use_distribution)}</p>"
        f"<p><strong>Set Aggregator analysis mode:</strong> {_html_esc(use_combination)}</p>"
        "<p><strong>Do not proceed if any check fails:</strong></p>"
        f"<ul>{stop_list}</ul>"
        f"{notes_html}"
        "</div></div>"
    )


def render_vvuq_glossary_html(section_id="section-vvuq-glossary"):
    """Render fixed VVUQ terminology table for report consistency."""
    rows = "".join(
        f"<tr><td>{_html_esc(term)}</td><td>{_html_esc(defn)}</td></tr>"
        for term, defn in VVUQ_TERMS
    )
    return (
        f'<div class="section" id="{_html_esc(section_id)}">'
        "<h2>VVUQ Terminology Panel</h2>"
        "<table><tr><th>Term</th><th>Definition</th></tr>"
        f"{rows}</table></div>"
    )


def render_conformity_template_html(metric_name, metric_value, consequence,
                                    section_id="section-conformity"):
    """Render optional conformity-assessment wording template."""
    level = normalize_decision_consequence(consequence)
    if level == "High":
        gb = "Recommended guard-band: 10% of requirement margin"
    elif level == "Medium":
        gb = "Recommended guard-band: 5% of requirement margin"
    else:
        gb = "Recommended guard-band: analyst judgment (document rationale)"
    return (
        f'<div class="section" id="{_html_esc(section_id)}">'
        "<h2>Conformity Assessment Template (Optional)</h2>"
        "<p>This section is a wording template. Fill in requirement limits "
        "and acceptance logic before release.</p>"
        '<div class="findings-block">'
        f"<p><strong>Assessed metric:</strong> {_html_esc(metric_name)} = {_html_esc(metric_value)}</p>"
        "<p><strong>Requirement limit:</strong> [INSERT LIMIT AND DIRECTION]</p>"
        "<p><strong>Acceptance rule:</strong> Accept only if measured/predicted value "
        "including uncertainty remains inside requirement after guard-banding.</p>"
        f"<p><strong>Decision consequence:</strong> {_html_esc(level)}. {_html_esc(gb)}.</p>"
        "<p><strong>Guard-band statement:</strong> [INSERT APPLIED GUARD-BAND METHOD]</p>"
        "</div></div>"
    )

# ═══════════════════════════════════════════════════════════════════════════
# END: SHARED VVUQ REPORT COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════

# =============================================================================
# CONSTANTS & THEME — matches VVUQ Uncertainty Aggregator
# =============================================================================

APP_VERSION = "1.5.0"
APP_NAME = "GCI Calculator"
APP_DATE = "2026-02-24"
__version__ = APP_VERSION

DARK_COLORS = {
    'bg': '#1e1e2e',
    'bg_alt': '#252536',
    'surface0': '#313244',
    'bg_widget': '#2a2a3c',
    'bg_input': '#333348',
    'fg': '#cdd6f4',
    'fg_dim': '#9399b2',
    'fg_bright': '#ffffff',
    'accent': '#89b4fa',
    'accent_hover': '#74c7ec',
    'green': '#a6e3a1',
    'yellow': '#f9e2af',
    'red': '#f38ba8',
    'orange': '#fab387',
    'border': '#45475a',
    'overlay0': '#6c7086',
    'selection': '#45475a',
    'link': '#89dceb',
}

PLOT_STYLE = {
    'figure.facecolor': DARK_COLORS['bg_alt'],
    'axes.facecolor': DARK_COLORS['bg_widget'],
    'axes.edgecolor': DARK_COLORS['border'],
    'axes.labelcolor': DARK_COLORS['fg'],
    'text.color': DARK_COLORS['fg'],
    'xtick.color': DARK_COLORS['fg_dim'],
    'ytick.color': DARK_COLORS['fg_dim'],
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 6.5,
    'grid.color': DARK_COLORS['border'],
    'legend.facecolor': DARK_COLORS['bg_widget'],
    'legend.edgecolor': DARK_COLORS['border'],
}
plt.rcParams.update(PLOT_STYLE)

# Light theme for HTML report export (print-ready)
REPORT_PLOT_STYLE = {
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#333333',
    'axes.labelcolor': '#1a1a2e',
    'text.color': '#1a1a2e',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 6.5,
    'grid.color': '#cccccc',
    'legend.facecolor': '#f5f5f5',
    'legend.edgecolor': '#999999',
}

# Greyscale theme for formal print submissions
PRINT_GREYSCALE_STYLE = {
    'figure.facecolor': '#ffffff',
    'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#000000',
    'axes.labelcolor': '#000000',
    'text.color': '#000000',
    'xtick.color': '#000000',
    'ytick.color': '#000000',
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'axes.labelsize': 8,
    'axes.titlesize': 9,
    'legend.fontsize': 6.5,
    'grid.color': '#aaaaaa',
    'legend.facecolor': '#ffffff',
    'legend.edgecolor': '#000000',
}

# Named style profiles for export
CHART_STYLE_PROFILES = {
    "Interactive Dark": PLOT_STYLE,
    "Report Light": REPORT_PLOT_STYLE,
    "Print Greyscale": PRINT_GREYSCALE_STYLE,
}

# Figure size presets for export (width, height in inches)
FIGURE_SIZE_PRESETS = {
    "Half-page (3.5×2.8 in)": (3.5, 2.8),
    "Full-page (7.0×4.5 in)": (7.0, 4.5),
    "Appendix landscape (10.0×6.0 in)": (10.0, 6.0),
    "Default (6×4 in)": (6.0, 4.0),
}

FONT_FAMILIES = ["Segoe UI", "DejaVu Sans", "Liberation Sans", "Noto Sans",
                  "Ubuntu", "Helvetica", "Arial", "sans-serif"]

UNIT_PRESETS = {
    "Temperature": ["K", "\u00b0C", "\u00b0F", "R"],
    "Pressure":    ["Pa", "kPa", "psia", "psig", "bar", "atm"],
    "Velocity":    ["m/s", "ft/s", "ft/min"],
    "Mass Flow":   ["kg/s", "lb/s", "lb/min"],
    "Force":       ["N", "kN", "lbf"],
    "Length":      ["m", "mm", "in", "ft"],
    "Other":       [],
}

# Scheme order presets for the theoretical-p combo box
THEORETICAL_ORDER_PRESETS = [
    ("Second-order (most CFD codes)", 2.0),
    ("First-order upwind", 1.0),
    ("Third-order (MUSCL, WENO-3)", 3.0),
    ("Fourth-order (spectral, high-order DG)", 4.0),
    ("Custom", None),
]


def get_dark_stylesheet():
    """Generate the dark mode stylesheet (matches VVUQ Uncertainty Aggregator)."""
    c = DARK_COLORS
    return f"""
    QMainWindow, QWidget {{
        background-color: {c['bg']};
        color: {c['fg']};
        font-size: 13px;
    }}
    QTabWidget::pane {{
        border: 1px solid {c['border']};
        background-color: {c['bg']};
    }}
    QTabBar::tab {{
        background-color: {c['bg_alt']};
        color: {c['fg_dim']};
        padding: 8px 16px;
        margin-right: 2px;
        border: 1px solid {c['border']};
        border-bottom: none;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }}
    QTabBar::tab:selected {{
        background-color: {c['bg_widget']};
        color: {c['accent']};
        border-bottom: 2px solid {c['accent']};
    }}
    QTabBar::tab:hover {{
        background-color: {c['bg_widget']};
        color: {c['fg']};
    }}
    QGroupBox {{
        border: 1px solid {c['border']};
        border-radius: 6px;
        margin-top: 12px;
        padding-top: 16px;
        font-weight: bold;
        color: {c['accent']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 12px;
        padding: 0 6px;
    }}
    QPushButton {{
        background-color: {c['bg_widget']};
        color: {c['fg']};
        border: 1px solid {c['border']};
        border-radius: 4px;
        padding: 6px 16px;
        min-height: 24px;
    }}
    QPushButton:hover {{
        background-color: {c['selection']};
        border-color: {c['accent']};
    }}
    QPushButton:pressed {{
        background-color: {c['accent']};
        color: {c['bg']};
    }}
    QPushButton:disabled {{
        color: {c['fg_dim']};
        background-color: {c['bg']};
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {c['bg_input']};
        color: {c['fg']};
        border: 1px solid {c['border']};
        border-radius: 4px;
        padding: 4px 8px;
        min-height: 22px;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {c['accent']};
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c['bg_widget']};
        color: {c['fg']};
        border: 1px solid {c['border']};
        selection-background-color: {c['selection']};
    }}
    QTableWidget {{
        background-color: {c['bg_widget']};
        alternate-background-color: {c['bg_alt']};
        color: {c['fg']};
        gridline-color: {c['border']};
        border: 1px solid {c['border']};
        border-radius: 4px;
    }}
    QTableWidget::item {{
        padding: 4px;
    }}
    QTableWidget::item:selected {{
        background-color: {c['selection']};
    }}
    QHeaderView::section {{
        background-color: {c['bg_alt']};
        color: {c['fg']};
        padding: 4px 8px;
        border: 1px solid {c['border']};
        font-weight: bold;
    }}
    QTextEdit, QPlainTextEdit {{
        background-color: {c['bg_input']};
        color: {c['fg']};
        border: 1px solid {c['border']};
        border-radius: 4px;
    }}
    QScrollBar:vertical {{
        background-color: {c['bg']};
        width: 12px;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background-color: {c['border']};
        border-radius: 4px;
        min-height: 20px;
    }}
    QScrollBar::handle:vertical:hover {{
        background-color: {c['fg_dim']};
    }}
    QScrollBar:horizontal {{
        background-color: {c['bg']};
        height: 12px;
        border: none;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {c['border']};
        border-radius: 4px;
        min-width: 20px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{
        height: 0; width: 0;
    }}
    QProgressBar {{
        background-color: {c['bg_input']};
        border: 1px solid {c['border']};
        border-radius: 4px;
        text-align: center;
        color: {c['fg']};
    }}
    QProgressBar::chunk {{
        background-color: {c['accent']};
        border-radius: 3px;
    }}
    QStatusBar {{
        background-color: {c['bg_alt']};
        color: {c['fg_dim']};
        border-top: 1px solid {c['border']};
    }}
    QMenuBar {{
        background-color: {c['bg_alt']};
        color: {c['fg']};
    }}
    QMenuBar::item:selected {{
        background-color: {c['selection']};
    }}
    QMenu {{
        background-color: {c['bg_widget']};
        color: {c['fg']};
        border: 1px solid {c['border']};
    }}
    QMenu::item:selected {{
        background-color: {c['selection']};
    }}
    QToolTip {{
        background-color: {c['bg_widget']};
        color: {c['fg']};
        border: 1px solid {c['accent']};
        padding: 6px;
        border-radius: 4px;
    }}
    QCheckBox {{
        color: {c['fg']};
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 16px; height: 16px;
        border: 1px solid {c['border']};
        border-radius: 3px;
        background-color: {c['bg_input']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {c['accent']};
        border-color: {c['accent']};
    }}
    QRadioButton {{
        color: {c['fg']};
        spacing: 8px;
    }}
    QRadioButton::indicator {{
        width: 16px; height: 16px;
        border: 1px solid {c['border']};
        border-radius: 8px;
        background-color: {c['bg_input']};
    }}
    QRadioButton::indicator:checked {{
        background-color: {c['accent']};
        border-color: {c['accent']};
    }}
    QSplitter::handle {{
        background-color: {c['border']};
    }}
    QLabel {{
        color: {c['fg']};
    }}
    """


# =============================================================================
# GUIDANCE PANEL — reusable (matches VVUQ Uncertainty Aggregator)
# =============================================================================

class GuidancePanel(QFrame):
    """Reusable panel for color-coded guidance messages (green/yellow/red)."""

    SEVERITY_CONFIG = {
        'green': {
            'border_color': DARK_COLORS['green'],
            'bg_color': '#1a2e1a',
            'icon': '\u2714',
            'label': 'OK',
        },
        'yellow': {
            'border_color': DARK_COLORS['yellow'],
            'bg_color': '#2e2a1a',
            'icon': '\u26A0',
            'label': 'CAUTION',
        },
        'red': {
            'border_color': DARK_COLORS['red'],
            'bg_color': '#2e1a1a',
            'icon': '\u2716',
            'label': 'WARNING',
        },
    }

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 8, 10, 8)
        self._layout.setSpacing(4)

        header = QHBoxLayout()
        header.setSpacing(6)
        self._icon_label = QLabel()
        self._icon_label.setFixedWidth(20)
        font = self._icon_label.font()
        font.setPointSize(12)
        self._icon_label.setFont(font)
        header.addWidget(self._icon_label)

        self._title_label = QLabel(self._title)
        tf = self._title_label.font()
        tf.setBold(True)
        self._title_label.setFont(tf)
        header.addWidget(self._title_label)
        header.addStretch()
        self._layout.addLayout(header)

        self._message_label = QLabel()
        self._message_label.setWordWrap(True)
        self._message_label.setTextFormat(Qt.PlainText)
        self._message_label.setStyleSheet(f"color: {DARK_COLORS['fg']};")
        self._layout.addWidget(self._message_label)

        self._apply_severity('green')

    def _apply_severity(self, severity: str):
        cfg = self.SEVERITY_CONFIG.get(severity, self.SEVERITY_CONFIG['green'])
        self._icon_label.setText(cfg['icon'])
        self._icon_label.setStyleSheet(f"color: {cfg['border_color']};")
        self.setStyleSheet(
            f"GuidancePanel {{"
            f"  background-color: {cfg['bg_color']};"
            f"  border-left: 4px solid {cfg['border_color']};"
            f"  border-top: none; border-right: none; border-bottom: none;"
            f"  border-radius: 4px;"
            f"}}"
        )

    def set_guidance(self, message: str, severity: str = 'green'):
        self._severity = severity
        self._message_label.setText(message)
        self._apply_severity(severity)

    def set_title(self, title: str):
        self._title = title
        self._title_label.setText(title)

    def clear(self):
        self._message_label.setText("")
        self._apply_severity('green')


# =============================================================================
# HELPER: copy figure to clipboard
# =============================================================================

def copy_figure_to_clipboard(fig):
    """Copy a matplotlib Figure to the system clipboard as an image."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    buf.seek(0)
    from PySide6.QtGui import QImage
    image = QImage.fromData(buf.read())
    QApplication.clipboard().setImage(image)


def copy_report_quality_figure(fig):
    """Copy a matplotlib Figure to clipboard at 300 DPI with light report theme."""
    orig_props = []
    for ax in fig.get_axes():
        orig_props.append({
            'facecolor': ax.get_facecolor(),
            'title_color': ax.title.get_color() if ax.title else None,
            'xlabel_color': ax.xaxis.label.get_color(),
            'ylabel_color': ax.yaxis.label.get_color(),
            'spine_colors': {s: ax.spines[s].get_edgecolor() for s in ax.spines},
        })
        ax.set_facecolor(REPORT_PLOT_STYLE['axes.facecolor'])
        ax.title.set_color(REPORT_PLOT_STYLE['text.color'])
        ax.xaxis.label.set_color(REPORT_PLOT_STYLE['axes.labelcolor'])
        ax.yaxis.label.set_color(REPORT_PLOT_STYLE['axes.labelcolor'])
        ax.tick_params(axis='x', colors=REPORT_PLOT_STYLE['xtick.color'])
        ax.tick_params(axis='y', colors=REPORT_PLOT_STYLE['ytick.color'])
        for spine in ax.spines.values():
            spine.set_edgecolor(REPORT_PLOT_STYLE['axes.edgecolor'])
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor(REPORT_PLOT_STYLE['legend.facecolor'])
            leg.get_frame().set_edgecolor(REPORT_PLOT_STYLE['legend.edgecolor'])
            for text in leg.get_texts():
                text.set_color(REPORT_PLOT_STYLE['text.color'])

    orig_fig_fc = fig.get_facecolor()
    fig.set_facecolor('#ffffff')

    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        from PySide6.QtGui import QImage
        image = QImage.fromData(buf.read())
        QApplication.clipboard().setImage(image)
    except Exception as exc:
        QMessageBox.warning(None, "Clipboard Error",
                            f"Could not copy figure to clipboard:\n\n{exc}")
    finally:
        # Restore original dark-theme properties (always runs)
        fig.set_facecolor(orig_fig_fc)
        for ax, props in zip(fig.get_axes(), orig_props):
            ax.set_facecolor(props['facecolor'])
            if props['title_color']:
                ax.title.set_color(props['title_color'])
            ax.xaxis.label.set_color(props['xlabel_color'])
            ax.yaxis.label.set_color(props['ylabel_color'])
            ax.tick_params(axis='x', colors=PLOT_STYLE['xtick.color'])
            ax.tick_params(axis='y', colors=PLOT_STYLE['ytick.color'])
            for s_name, s_color in props['spine_colors'].items():
                ax.spines[s_name].set_edgecolor(s_color)
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor(PLOT_STYLE['legend.facecolor'])
                leg.get_frame().set_edgecolor(PLOT_STYLE['legend.edgecolor'])
                for text in leg.get_texts():
                    text.set_color(PLOT_STYLE['text.color'])


def export_figure_package(fig, base_path: str, metadata: Optional[dict] = None,
                          figure_id: str = "", analysis_id: str = "",
                          settings_hash: str = "", data_hash: str = "",
                          units: str = "", method_context: str = ""):
    """Export a matplotlib Figure in publication-quality formats.

    Generates PNG (300+600 DPI), SVG, PDF, and a JSON metadata sidecar.
    """
    import json
    from datetime import datetime, timezone

    for dpi_val in (300, 600):
        fig.savefig(f"{base_path}_{dpi_val}dpi.png", format="png", dpi=dpi_val,
                    bbox_inches="tight", facecolor="white", edgecolor="none")

    fig.savefig(f"{base_path}.svg", format="svg",
                bbox_inches="tight", facecolor="white", edgecolor="none")
    fig.savefig(f"{base_path}.pdf", format="pdf",
                bbox_inches="tight", facecolor="white", edgecolor="none")

    meta = {
        "tool_name": APP_NAME,
        "tool_version": APP_VERSION,
        "figure_id": figure_id or os.path.basename(base_path),
        "analysis_id": analysis_id,
        "profile": "report_light",
        "settings_hash": settings_hash,
        "data_hash": data_hash,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "units": units,
        "method_context": method_context,
        "formats": ["png@300dpi", "png@600dpi", "svg", "pdf"],
    }
    if metadata:
        meta.update(metadata)

    with open(f"{base_path}_meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)


def _export_figure_package_dialog(fig, parent, **metadata_kwargs):
    """Open a file dialog and export a figure package.

    Auto-populates metadata from the parent widget context when available.
    Caller-supplied ``metadata_kwargs`` override auto-collected values.
    """
    filepath, _ = QFileDialog.getSaveFileName(
        parent, "Export Figure Package",
        os.path.expanduser("~"),
        "Figure Base Name (*)",
    )
    if not filepath:
        return
    base = os.path.splitext(filepath)[0]
    meta_kw: dict = {'method_context': 'GCI Celik et al. (2008)'}
    # Walk up to main window for project context
    w = parent
    main_win = None
    while w is not None:
        if hasattr(w, '_project_path') and w._project_path:
            meta_kw['analysis_id'] = os.path.basename(w._project_path)
            main_win = w
            break
        if hasattr(w, 'windowTitle') and callable(w.windowTitle):
            title = w.windowTitle()
            if 'GCI' in title:
                meta_kw['analysis_id'] = title
                main_win = w
                break
        w = w.parent() if hasattr(w, 'parent') and callable(w.parent) else None
    # Walk up to find the GCICalculatorTab for units, settings, and data hash
    tab_w = parent
    gci_tab = None
    while tab_w is not None:
        if hasattr(tab_w, '_quantity_units') and hasattr(tab_w, '_read_table_data'):
            gci_tab = tab_w
            break
        tab_w = (tab_w.parent()
                 if hasattr(tab_w, 'parent') and callable(tab_w.parent)
                 else None)
    if gci_tab is not None:
        if gci_tab._quantity_units:
            meta_kw['units'] = gci_tab._quantity_units[0]
        # settings_hash: GCI analysis settings fingerprint
        try:
            import hashlib as _hl
            s_dict = {
                'n_grids': (gci_tab._cmb_n_grids.currentData()
                            if hasattr(gci_tab, '_cmb_n_grids') else 0),
                'dimension': (gci_tab._cmb_dim.currentData()
                              if hasattr(gci_tab, '_cmb_dim') else 3),
                'n_quantities': getattr(gci_tab, '_n_quantities', 1),
            }
            s_json = json.dumps(s_dict, sort_keys=True)
            meta_kw['settings_hash'] = _hl.sha256(
                s_json.encode()).hexdigest()[:16]
        except Exception:
            pass
        # data_hash: fingerprint of grid data (cell counts + solutions)
        try:
            import hashlib as _hl2
            cell_counts, solutions = gci_tab._read_table_data()
            d_str = json.dumps({
                'cell_counts': cell_counts,
                'solutions': solutions,
            }, sort_keys=True)
            meta_kw['data_hash'] = _hl2.sha256(
                d_str.encode()).hexdigest()[:16]
        except Exception:
            pass
    # Caller overrides take precedence
    meta_kw.update(metadata_kwargs)
    try:
        export_figure_package(fig, base, **meta_kw)
    except Exception as exc:
        QMessageBox.critical(
            parent, "Export Error",
            f"Could not export figure package:\n\n{exc}"
        )
        return
    # Status bar feedback (matching VVUQ pattern)
    sb_w = main_win or parent
    while sb_w is not None:
        if hasattr(sb_w, 'statusBar') and callable(sb_w.statusBar):
            sb_w.statusBar().showMessage(
                f"Figure package exported to {base}_*.{{png,svg,pdf,json}}",
                8000)
            break
        sb_w = (sb_w.parent()
                if hasattr(sb_w, 'parent') and callable(sb_w.parent)
                else None)


def style_table(table, column_widths=None, stretch_col=None):
    """
    Apply consistent styling to a QTableWidget so that:
      - All columns are user-resizable (Interactive mode)
      - Horizontal scroll bar appears when columns exceed available width
      - Word wrap is enabled so long text grows the row height
      - Optional explicit column widths dict {col_index: width_px}
      - Optional stretch_col: gives that column a generous default width
        but keeps it Interactive (user-resizable) -- does NOT lock it
    """
    header = table.horizontalHeader()
    # Make ALL columns user-draggable / resizable
    header.setSectionResizeMode(QHeaderView.Interactive)
    header.setMinimumSectionSize(50)
    header.setStretchLastSection(True)
    # Allow horizontal scrolling
    table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    # Word-wrap so long names grow row height instead of clipping
    table.setWordWrap(True)
    # If caller specified widths, apply them
    if column_widths:
        for col, w in column_widths.items():
            table.setColumnWidth(col, w)
    # Give stretch_col a generous default width (still user-resizable)
    if stretch_col is not None and stretch_col < table.columnCount():
        table.setColumnWidth(stretch_col, 300)


# =============================================================================
# GCI MATH ENGINE
# =============================================================================

def compute_refinement_ratio(n_cells_fine: float, n_cells_coarse: float,
                             dim: int = 3) -> float:
    """
    Compute effective grid refinement ratio from cell counts.

    r = (N_fine / N_coarse) ^ (1/dim)

    For non-integer refinement ratios (unstructured meshes), this gives
    the effective ratio based on representative cell size.

    Parameters
    ----------
    n_cells_fine : float
        Number of cells in the finer grid.
    n_cells_coarse : float
        Number of cells in the coarser grid.
    dim : int
        Spatial dimension (2 or 3).

    Returns
    -------
    r : float
        Effective refinement ratio (> 1.0 means coarser has larger cells).

    Ref: Celik et al. (2008) Eq. (3): h = [1/N * sum(dV_i)]^(1/3)
    """
    if n_cells_fine <= 0 or n_cells_coarse <= 0:
        return float('nan')
    # r = h_coarse / h_fine = (N_fine / N_coarse)^(1/dim) > 1.0
    return (n_cells_fine / n_cells_coarse) ** (1.0 / dim)


def _solve_observed_order(e21: float, e32: float, r21: float,
                          r32: float) -> float:
    """
    Iteratively solve for observed order of accuracy p.

    For constant refinement ratio (r21 == r32 = r):
        p = ln(e32/e21) / ln(r)

    For non-constant refinement ratio, solve the transcendental equation:
        p = (1/ln(r21)) * |ln|e32/e21| + ln((r21^p - s) / (r32^p - s))|
    where s = sign(e32/e21).

    Uses fixed-point iteration per Celik et al. (2008) Eq. (5).

    Returns
    -------
    p : float
        Observed order of accuracy.  Returns NaN if solution fails.
    """
    if abs(e21) < 1e-30 or abs(e32) < 1e-30:
        return float('nan')

    ratio = e32 / e21
    s = 1.0 if ratio > 0 else -1.0

    # Constant refinement ratio — direct formula
    if abs(r21 - r32) < 1e-10:
        if ratio <= 0:
            # Oscillatory convergence — use absolute values
            return abs(np.log(abs(ratio)) / np.log(r21))
        return np.log(ratio) / np.log(r21)

    # Non-constant refinement ratio — fixed-point iteration
    ln_r21 = np.log(r21)

    # Initial guess from constant-ratio formula
    if ratio > 0:
        p = np.log(ratio) / ln_r21
    else:
        p = np.log(abs(ratio)) / ln_r21

    p = max(p, 0.1)  # floor at 0.1 — allows first-order upwind on coarse meshes

    for _iter in range(100):
        p_old = p
        try:
            rp21 = _safe_rp(r21, p)
            rp32 = _safe_rp(r32, p)
            if not np.isfinite(rp21) or not np.isfinite(rp32):
                return float('nan')
            q = np.log(abs((rp21 - s) / (rp32 - s)))
            p = (1.0 / ln_r21) * abs(np.log(abs(ratio)) + q)
        except (ValueError, ZeroDivisionError, OverflowError):
            return float('nan')
        if abs(p - p_old) < 1e-10:
            break
    else:
        return float('nan')  # failed to converge

    return p


@dataclass
class GCIResult:
    """Results from a GCI calculation for a single quantity of interest."""
    # Input data
    grid_solutions: List[float] = field(default_factory=list)
    grid_cells: List[float] = field(default_factory=list)
    grid_spacings: List[float] = field(default_factory=list)
    dim: int = 3
    safety_factor: float = 1.25
    theoretical_order: float = 2.0

    # Computed results
    refinement_ratios: List[float] = field(default_factory=list)
    observed_order: float = float('nan')
    order_is_assumed: bool = False       # True for 2-grid (order assumed, not computed)
    richardson_extrapolation: float = float('nan')
    convergence_type: str = "unknown"   # monotonic, oscillatory, divergent
    convergence_ratio: float = float('nan')  # R = e21/e32

    # GCI values
    gci_fine: float = float('nan')      # GCI on the finest grid pair
    gci_coarse: float = float('nan')    # GCI on the coarser grid pair
    asymptotic_ratio: float = float('nan')  # should be ~1.0

    # Errors
    e21_abs: float = float('nan')       # |f2 - f1|
    e32_abs: float = float('nan')       # |f3 - f2|
    e21_rel: float = float('nan')       # |(f2-f1)/f1|  (relative)
    e32_rel: float = float('nan')       # |(f3-f2)/f2|

    # Derived uncertainty (fine grid — standard GCI)
    u_num: float = float('nan')         # GCI converted to standard uncertainty
    u_num_pct: float = float('nan')    # u_num as % of fine grid solution

    # Per-grid uncertainty (from Richardson extrapolation)
    per_grid_error: List[float] = field(default_factory=list)   # |f_i - f_exact|
    per_grid_u_num: List[float] = field(default_factory=list)   # u_num for each grid
    per_grid_u_pct: List[float] = field(default_factory=list)   # u_num as % of |f_i|
    target_grid_idx: int = 0            # 0-based index of production grid

    # Method used
    n_grids: int = 0
    method: str = ""                    # "3-grid RE", "2-grid assumed-p", etc.
    is_valid: bool = False
    warnings: List[str] = field(default_factory=list)
    notes: str = ""

    # Auto-selection (when primary triplet is divergent and N > 3)
    auto_selected_triplet: Optional[List[int]] = None  # 0-based grid indices used
    auto_selection_reason: str = ""
    auto_selection_candidates: List[dict] = field(default_factory=list)  # all evaluated triplets

    # Effective grid-independence (relative threshold)
    effectively_grid_independent: bool = False

    # Directional u_num (one-sided validation support)
    direction_of_concern: str = "both"      # "both", "underprediction", "overprediction"
    discretization_bias_sign: str = ""      # "high", "low", "neutral", "" (production vs grid-converged)
    u_num_directional: float = float('nan') # u_num adjusted for direction of concern
    directional_note: str = ""              # explanation when directional differs from standard



def _safe_rp(r: float, p: float) -> float:
    """Compute r**p with overflow protection.

    Returns float('inf') if the result overflows or is not finite.
    """
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            val = r ** p
    except (OverflowError, ValueError):
        return float('inf')
    if not np.isfinite(val):
        return float('inf')
    return val


def _find_best_triplet(solutions: List[float], cell_counts: List[float],
                       dim: int = 3, target_idx: Optional[int] = None,
                       ) -> Optional[dict]:
    """Find the best grid triplet for GCI from N grids.

    When the primary (finest 3) triplet shows divergent behaviour due to
    low refinement ratios, this helper evaluates ALL C(N,3) combinations
    and returns the one with monotonic convergence and the best refinement
    ratios.

    Parameters
    ----------
    solutions : list of float
        Solution values on each grid (finest first).
    cell_counts : list of float
        Cell counts per grid (finest first).
    dim : int
        Spatial dimension for refinement ratio computation.
    target_idx : int or None
        0-based index of the production grid.  Triplets containing this
        grid receive a small scoring bonus.

    Returns
    -------
    dict or None
        Best triplet info, or None if no valid triplet exists.
        Keys: indices, solutions, cell_counts, ratios, R, score, reason,
              all_candidates (list of ALL evaluated triplets incl. rejected)
    """
    from itertools import combinations
    n = len(solutions)
    if n < 3:
        return None

    candidates = []       # (score, info_dict) — valid triplets only
    all_evaluated = []    # ALL triplets (valid + rejected) for traceability

    for combo in combinations(range(n), 3):
        # combo is already sorted ascending — indices are 0-based (finest first)
        i, j, k = combo
        cc = [cell_counts[i], cell_counts[j], cell_counts[k]]
        ss = [solutions[i], solutions[j], solutions[k]]

        # Refinement ratios
        r12 = compute_refinement_ratio(cc[0], cc[1], dim)
        r23 = compute_refinement_ratio(cc[1], cc[2], dim)

        # Convergence ratio
        e21 = ss[1] - ss[0]
        e32 = ss[2] - ss[1]

        if abs(e32) < 1e-30:
            all_evaluated.append({
                'indices': list(combo),
                'R': float('nan'),
                'convergence_type': 'indeterminate',
                'min_r': min(r12, r23),
                'score': None,
                'selected': False,
            })
            continue  # can't compute R

        R = e21 / e32

        if 0 <= R < 1:
            conv_type = "monotonic"
        elif -1 < R < 0:
            conv_type = "oscillatory"
        else:
            # Divergent — record and skip
            all_evaluated.append({
                'indices': list(combo),
                'R': R,
                'convergence_type': 'divergent',
                'min_r': min(r12, r23),
                'score': None,
                'selected': False,
            })
            continue

        # Score: prefer monotonic, then best min refinement ratio
        type_bonus = 1000.0 if conv_type == "monotonic" else 0.0
        r_min = min(r12, r23)
        target_bonus = 0.5 if (target_idx is not None and target_idx in combo) else 0.0
        finest_bonus = 0.1 if 0 in combo else 0.0
        score = type_bonus + r_min + target_bonus + finest_bonus

        info = {
            'indices': list(combo),
            'solutions': ss,
            'cell_counts': cc,
            'ratios': [r12, r23],
            'R': R,
            'convergence_type': conv_type,
            'score': score,
            'reason': (
                f"Auto-selected Grids {i+1}, {j+1}, {k+1} "
                f"({conv_type} convergence, R = {R:.4f}, "
                f"r = {r12:.3f}, {r23:.3f})"
            ),
        }
        candidates.append((score, info))

        all_evaluated.append({
            'indices': list(combo),
            'R': R,
            'convergence_type': conv_type,
            'min_r': r_min,
            'score': score,
            'selected': False,
        })

    if not candidates:
        return None

    # Return highest-scoring candidate
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0][1]

    # Mark the selected triplet in all_evaluated
    for entry in all_evaluated:
        if entry['indices'] == best['indices']:
            entry['selected'] = True
            break

    best['all_candidates'] = all_evaluated
    return best


def compute_gci(solutions: List[float], cell_counts: List[float],
                dim: int = 3, safety_factor: Optional[float] = None,
                theoretical_order: float = 2.0,
                assumed_order: Optional[float] = None,
                reference_scale: Optional[float] = None,
                direction_of_concern: str = "both",
                grid_independent_tol: float = 0.001,
                target_grid_idx: int = 0) -> GCIResult:
    """
    Compute Grid Convergence Index for 2, 3, or N grids.

    Parameters
    ----------
    solutions : list of float
        Solution values on each grid, ORDERED from finest to coarsest.
        For N grids, solutions[0] = finest, solutions[-1] = coarsest.
    cell_counts : list of float
        Number of cells on each grid (same order as solutions).
    dim : int
        Spatial dimension (2 or 3) for refinement ratio computation.
    safety_factor : float or None
        GCI safety factor Fs.  If None:
        - 3+ grids with good convergence: Fs = 1.25 (Roache recommended)
        - 2 grids or poor convergence: Fs = 3.0 (conservative)
    theoretical_order : float
        Theoretical/formal order of the numerical method (default 2.0
        for second-order schemes, which covers most CFD).
    assumed_order : float or None
        For 2-grid studies, the assumed order of accuracy.
        If None, uses theoretical_order.
    reference_scale : float or None
        Optional reference value for normalizing relative errors.
        Use this when solution values are near zero (e.g., pressure
        differences, residuals) to avoid division by near-zero.
        If None (default), uses the fine-grid solution abs(f1).
    direction_of_concern : str
        Validation concern direction: "both" (default, two-sided),
        "underprediction", or "overprediction".  When the discretization
        bias is conservative for the direction of concern (e.g.,
        production grid overpredicts and concern is underprediction),
        u_num_directional is set to 0.
    grid_independent_tol : float
        Relative tolerance for effective grid-independence detection.
        If the spread across the 3 finest grids is less than this
        fraction of the mean solution, the grids are considered
        effectively independent.  Default 0.001 (0.1%).

    Returns
    -------
    GCIResult
        Complete results including GCI, order p, Richardson extrapolation,
        convergence type, asymptotic check, and u_num.

    References
    ----------
    Celik et al. (2008), J. Fluids Eng. 130(7), 078001 — Eqs. 1-12.
    Roache (1998) — Safety factor recommendations.
    ASME V&V 20-2009 §5.1 — Numerical uncertainty for validation.

    Notes
    -----
    - For 3 grids: uses the finest three to compute observed order via
      Richardson extrapolation (the standard Celik et al. procedure).
    - For 4+ grids: primary GCI uses the finest three grids.  If that
      triplet is divergent, auto-triplet reselection evaluates all C(N,3)
      combinations and picks the best monotonically-converging triplet.
      Additional grids also provide extra pairs for cross-checking.
    - For 2 grids: Richardson extrapolation is NOT possible (underdetermined).
      Uses an assumed order (typically the theoretical order) with the
      conservative Fs = 3.0 safety factor.
    - Convergence types:
      * Monotonic (0 < R < 1): ideal; all grids converge toward same answer
      * Oscillatory (-1 < R < 0): solution oscillates but amplitude decreases
      * Monotonic divergence (R > 1): grids diverge — GCI not meaningful
      * Oscillatory divergence (R < -1): grids diverge with oscillation
    """
    if len(solutions) != len(cell_counts):
        raise ValueError(
            f"solutions ({len(solutions)}) and cell_counts "
            f"({len(cell_counts)}) must have the same length.")

    # Validate dimension
    if dim not in (1, 2, 3):
        res = GCIResult()
        res.n_grids = len(solutions)
        res.dim = dim
        res.grid_solutions = list(solutions)
        res.grid_cells = list(cell_counts)
        res.warnings.append(f"dim must be 1, 2, or 3, got {dim}.")
        res.notes = "Invalid dimension parameter."
        return res

    n_grids = len(solutions)

    # Auto-sort: enforce finest-first (descending cell count) ordering.
    # The GUI already does this, but direct API callers may not.
    # We must also remap target_grid_idx so it tracks the same grid.
    indexed = list(enumerate(zip(cell_counts, solutions)))  # [(orig_idx, (cells, sol)), ...]
    indexed.sort(key=lambda x: -x[1][0])
    sort_map = {orig: new for new, (orig, _) in enumerate(indexed)}
    cell_counts = [item[1][0] for item in indexed]
    solutions = [item[1][1] for item in indexed]
    if target_grid_idx is not None:
        target_grid_idx = sort_map.get(target_grid_idx, target_grid_idx)
    else:
        target_grid_idx = 0  # Default to finest grid (index 0 after sort)

    # Clamp target_grid_idx to valid range
    if target_grid_idx < 0 or target_grid_idx >= n_grids:
        target_grid_idx = 0

    res = GCIResult()
    res.n_grids = n_grids
    res.dim = dim
    res.theoretical_order = theoretical_order
    res.grid_solutions = list(solutions)
    res.grid_cells = list(cell_counts)
    res.target_grid_idx = target_grid_idx

    if n_grids < 2:
        res.warnings.append("Need at least 2 grids for GCI calculation.")
        return res

    # Guard: non-finite inputs
    if any(not np.isfinite(s) for s in solutions):
        res.warnings.append(
            "Non-finite solution value (NaN or Inf) detected. "
            "Cannot compute GCI.")
        res.notes = "Input contains NaN or Inf — check data extraction."
        return res
    if any(c <= 0 or not np.isfinite(c) for c in cell_counts):
        res.warnings.append(
            "Non-positive or non-finite cell count detected. "
            "Cannot compute GCI.")
        res.notes = "Cell counts must be positive finite numbers."
        return res

    # ------------------------------------------------------------------
    # 1. Compute representative spacings and refinement ratios
    # ------------------------------------------------------------------
    # h_i = (1/N_i)^(1/dim) — representative cell size (normalized)
    spacings = [(1.0 / max(n, 1)) ** (1.0 / dim) for n in cell_counts]
    res.grid_spacings = spacings

    # Refinement ratios between consecutive grid pairs
    ratios = []
    for i in range(n_grids - 1):
        r = compute_refinement_ratio(cell_counts[i], cell_counts[i + 1], dim)
        ratios.append(r)
    res.refinement_ratios = ratios

    # Validate refinement ratios
    for i, r in enumerate(ratios):
        if r < 1.05:
            res.warnings.append(
                f"Refinement ratio r_{i+1}{i+2} = {r:.3f} is very close to 1. "
                f"This pair is too similar for a stable GCI estimate. "
                f"Celik et al. recommend r > 1.3 for reliable studies."
            )
        elif r < 1.30:
            res.warnings.append(
                f"Refinement ratio r_{i+1}{i+2} = {r:.3f} is below the "
                f"recommended 1.3 threshold. GCI may be sensitive to noise; "
                f"interpret with caution."
            )

    # ------------------------------------------------------------------
    # 2. Handle based on number of grids
    # ------------------------------------------------------------------

    if n_grids == 2:
        # ========== 2-GRID STUDY ==========
        res.method = "2-grid (assumed order)"
        f1, f2 = solutions[0], solutions[1]
        r21 = ratios[0]
        p = assumed_order if assumed_order is not None else theoretical_order

        res.observed_order = p
        res.order_is_assumed = True
        res.e21_abs = abs(f2 - f1)
        ref = reference_scale if reference_scale is not None else None
        norm_2g = ref if ref is not None else abs(f1)
        res.e21_rel = abs((f2 - f1) / norm_2g) if norm_2g > 1e-30 else float('nan')

        # Warn about near-zero solutions
        if reference_scale is None and abs(f1) > 0 and abs(f1) < 1e-10 * abs(f2):
            res.warnings.append(
                f"Fine-grid solution ({f1:.4e}) is near zero. "
                f"Consider providing a reference scale value."
            )

        # Richardson extrapolation with assumed order
        rp21 = _safe_rp(r21, p)
        if abs(rp21 - 1.0) > 1e-30 and np.isfinite(rp21):
            res.richardson_extrapolation = f1 + (f1 - f2) / (rp21 - 1.0)
        else:
            res.richardson_extrapolation = f1

        # Safety factor: Fs = 3.0 for 2-grid (Roache recommendation)
        fs = safety_factor if safety_factor is not None else 3.0
        res.safety_factor = fs

        # GCI_fine = Fs * |e_rel| / (r^p - 1)
        if abs(rp21 - 1.0) > 1e-30 and np.isfinite(rp21) and norm_2g > 1e-30:
            ea21 = abs((f2 - f1) / norm_2g)
            res.gci_fine = fs * ea21 / (rp21 - 1.0)
        else:
            res.gci_fine = float('nan')

        res.convergence_type = "assumed (2-grid)"
        res.is_valid = not np.isnan(res.gci_fine)

        res.notes = (
            f"2-grid study: Richardson extrapolation uses assumed order "
            f"p = {p:.1f} (theoretical). Safety factor Fs = {fs:.2f} "
            f"(conservative, per Roache 1998). For more reliable results, "
            f"use 3+ grids."
        )

    else:
        # ========== 3+ GRID STUDY ==========
        # Use the finest 3 grids for the primary GCI calculation
        f1, f2, f3 = solutions[0], solutions[1], solutions[2]
        r21, r32 = ratios[0], ratios[1]

        res.method = f"{n_grids}-grid Richardson extrapolation"

        # Differences (signed, for convergence type detection)
        e21 = f2 - f1      # signed difference
        e32 = f3 - f2      # signed difference

        res.e21_abs = abs(e21)
        res.e32_abs = abs(e32)

        # Use reference_scale for normalization if provided (near-zero solutions)
        ref = reference_scale if reference_scale is not None else None
        norm1 = ref if ref is not None else abs(f1)
        norm2 = ref if ref is not None else abs(f2)
        res.e21_rel = abs(e21 / norm1) if norm1 > 1e-30 else float('nan')
        res.e32_rel = abs(e32 / norm2) if norm2 > 1e-30 else float('nan')

        # Warn about near-zero solutions if no reference_scale provided
        if reference_scale is None:
            max_sol = max(abs(f1), abs(f2), abs(f3))
            if max_sol > 1e-30 and abs(f1) < 1e-10 * max_sol:
                res.warnings.append(
                    f"Fine-grid solution ({f1:.4e}) is near zero relative to "
                    f"other grids. Relative errors may be unreliable. "
                    f"Consider providing a reference scale value."
                )

        # ----- Effective grid-independence check (relative threshold) -----
        _spread = max(f1, f2, f3) - min(f1, f2, f3)
        _ref_gi = reference_scale if reference_scale else abs(
            (f1 + f2 + f3) / 3.0)
        _rel_spread = (_spread / _ref_gi) if _ref_gi > 1e-30 else float('inf')

        if _rel_spread < grid_independent_tol and _spread > 0:
            res.convergence_type = "grid-independent"
            res.effectively_grid_independent = True
            res.convergence_ratio = 0.0
            res.observed_order = float('inf')
            res.richardson_extrapolation = f1  # finest grid is best estimate
            res.gci_fine = 0.0
            res.gci_coarse = 0.0
            res.asymptotic_ratio = 1.0
            res.is_valid = True
            res.u_num = _spread / 2.0  # conservative: half the spread
            if abs(f1) > 1e-30:
                res.u_num_pct = (res.u_num / abs(f1)) * 100.0
            else:
                res.u_num_pct = 0.0
            fs = safety_factor if safety_factor is not None else 1.25
            res.safety_factor = fs
            res.notes = (
                f"Effectively grid-independent: solutions differ by "
                f"{_spread:.4g} ({_rel_spread*100:.3f}% of mean), which is "
                f"below the {grid_independent_tol*100:.1f}% threshold. "
                f"u_num estimated conservatively as half the solution "
                f"spread ({res.u_num:.4g})."
            )

        # ----- Convergence type (standard checks) -----
        elif abs(e32) < 1e-30 and abs(e21) < 1e-30:
            # True grid-independent — ALL three solutions match
            res.convergence_type = "grid-independent"
            res.convergence_ratio = 0.0
            res.observed_order = float('inf')
            res.richardson_extrapolation = f1
            res.gci_fine = 0.0
            res.gci_coarse = 0.0
            res.asymptotic_ratio = 1.0
            res.is_valid = True
            res.notes = "Solutions are grid-independent (no further refinement needed)."
            fs = safety_factor if safety_factor is not None else 1.25
            res.safety_factor = fs

        elif abs(e32) < 1e-30:
            # e32 ~ 0 but e21 != 0: medium/coarse converged but fine differs.
            # This is NOT grid-independence — it indicates the coarse grids
            # have not resolved the solution that the fine grid captures.
            res.convergence_type = "divergent"
            res.convergence_ratio = float('inf')
            res.is_valid = False
            res.observed_order = float('nan')  # e32 ~ 0 → cannot estimate
            res.warnings.append(
                "Medium and coarse grids give identical solutions, but fine "
                "grid differs. This suggests the coarse grids have not "
                "resolved the solution. Check mesh quality and solver "
                "convergence."
            )
            res.notes = (
                "DIVERGENT: Medium and coarse grids agree but fine grid "
                "differs. The coarse grids have likely not resolved the "
                "flow feature captured by the fine grid."
            )
            return res

        else:
            R = e21 / e32
            res.convergence_ratio = R

            if 0 <= R < 1:
                # R=0 means f1==f2 (finest pair converged) with e32!=0
                res.convergence_type = "monotonic"
            elif -1 < R < 0:
                res.convergence_type = "oscillatory"
            else:
                # R >= 1 (monotonic divergence) or R <= -1 (oscillatory divergence)
                res.convergence_type = "divergent"

            if res.convergence_type == "divergent":
                # ----- DIVERGENT — try auto-selecting a better triplet -----
                res.observed_order = _solve_observed_order(e21, e32, r21, r32)

                if n_grids > 3:
                    best = _find_best_triplet(
                        solutions, cell_counts, dim,
                        target_idx=res.target_grid_idx)
                    if best is not None:
                        # Re-run GCI on the auto-selected triplet
                        bi = best['indices']
                        auto_res = compute_gci(
                            solutions=best['solutions'],
                            cell_counts=best['cell_counts'],
                            dim=dim,
                            safety_factor=safety_factor,
                            theoretical_order=theoretical_order,
                            reference_scale=reference_scale,
                            direction_of_concern=direction_of_concern,
                            grid_independent_tol=grid_independent_tol,
                        )
                        if auto_res.is_valid:
                            # Carry over results from auto-selected triplet
                            auto_res.grid_solutions = list(solutions)
                            auto_res.grid_cells = list(cell_counts)
                            auto_res.grid_spacings = res.grid_spacings
                            auto_res.refinement_ratios = res.refinement_ratios
                            auto_res.n_grids = n_grids
                            auto_res.target_grid_idx = target_grid_idx
                            auto_res.auto_selected_triplet = bi
                            auto_res.auto_selection_reason = best['reason']
                            auto_res.auto_selection_candidates = best.get(
                                'all_candidates', [])
                            auto_res.method = (
                                f"{n_grids}-grid (auto-selected "
                                f"Grids {bi[0]+1},{bi[1]+1},{bi[2]+1})"
                            )
                            primary_note = (
                                f"Primary triplet (Grids 1-2-3) showed "
                                f"divergent behaviour (R = {R:.4f}). "
                            )
                            auto_res.notes = primary_note + best['reason'] + \
                                "\n\n" + auto_res.notes
                            auto_res.warnings = res.warnings + auto_res.warnings
                            auto_res.warnings.append(
                                f"Primary triplet divergent (R = {R:.4f}); "
                                f"auto-selected Grids {bi[0]+1}, {bi[1]+1}, "
                                f"{bi[2]+1} for GCI computation."
                            )
                            # Recompute per-grid u_num for ALL original grids
                            f_ext_auto = auto_res.richardson_extrapolation
                            if not np.isnan(f_ext_auto):
                                auto_res.per_grid_error = []
                                auto_res.per_grid_u_num = []
                                auto_res.per_grid_u_pct = []
                                for gi in range(n_grids):
                                    err = abs(solutions[gi] - f_ext_auto)
                                    auto_res.per_grid_error.append(err)
                                    auto_res.per_grid_u_num.append(err)
                                    if abs(solutions[gi]) > 1e-30:
                                        auto_res.per_grid_u_pct.append(
                                            (err / abs(solutions[gi])) * 100.0)
                                    else:
                                        auto_res.per_grid_u_pct.append(0.0)
                            else:
                                # Oscillatory auto-selected: RE not available.
                                # Use oscillation-based u_num for all grids.
                                auto_res.per_grid_error = [auto_res.u_num] * n_grids
                                auto_res.per_grid_u_num = [auto_res.u_num] * n_grids
                                auto_res.per_grid_u_pct = [
                                    (auto_res.u_num / abs(solutions[gi]) * 100.0
                                     if abs(solutions[gi]) > 1e-30 else 0.0)
                                    for gi in range(n_grids)
                                ]
                            # Recompute directional u_num with correct target
                            tgt_a = target_grid_idx
                            if (0 <= tgt_a < len(solutions)
                                    and not np.isnan(f_ext_auto)):
                                bias_a = solutions[tgt_a] - f_ext_auto
                                if abs(bias_a) < 1e-30:
                                    auto_res.discretization_bias_sign = "neutral"
                                elif bias_a > 0:
                                    auto_res.discretization_bias_sign = "high"
                                else:
                                    auto_res.discretization_bias_sign = "low"
                                std_u_a = auto_res.per_grid_u_num[tgt_a]
                                auto_res.direction_of_concern = direction_of_concern
                                if (direction_of_concern == "both"
                                        or auto_res.discretization_bias_sign == "neutral"):
                                    auto_res.u_num_directional = std_u_a
                                    auto_res.directional_note = ""
                                elif (direction_of_concern == "underprediction"
                                      and auto_res.discretization_bias_sign == "high"):
                                    auto_res.u_num_directional = 0.0
                                    auto_res.directional_note = (
                                        "Production grid overpredicts relative to "
                                        "grid-converged solution. Discretization bias "
                                        "is conservative for underprediction — "
                                        "u_num contribution is zero."
                                    )
                                elif (direction_of_concern == "overprediction"
                                      and auto_res.discretization_bias_sign == "low"):
                                    auto_res.u_num_directional = 0.0
                                    auto_res.directional_note = (
                                        "Production grid underpredicts relative to "
                                        "grid-converged solution. Discretization bias "
                                        "is conservative for overprediction — "
                                        "u_num contribution is zero."
                                    )
                                else:
                                    # Adverse bias — full u_num applies
                                    auto_res.u_num_directional = std_u_a
                                    auto_res.directional_note = (
                                        f"Production grid {auto_res.discretization_bias_sign}-predicts "
                                        f"relative to grid-converged solution. Discretization bias "
                                        f"is NOT conservative for {direction_of_concern} — "
                                        f"full u_num applies."
                                    )
                            else:
                                # Oscillatory auto-selected triplet: Richardson
                                # extrapolation is NaN, so directional bias
                                # cannot be determined.  Use full u_num.
                                auto_res.direction_of_concern = direction_of_concern
                                if 0 <= tgt_a < n_grids:
                                    auto_res.u_num_directional = (
                                        auto_res.per_grid_u_num[tgt_a]
                                        if tgt_a < len(auto_res.per_grid_u_num)
                                        else auto_res.u_num
                                    )
                                else:
                                    auto_res.u_num_directional = auto_res.u_num
                                auto_res.directional_note = (
                                    "Oscillatory convergence in auto-selected "
                                    "triplet — Richardson extrapolation is "
                                    "unavailable, so directional bias cannot "
                                    "be assessed. Full u_num applies."
                                )
                            return auto_res

                # No auto-selection possible — original divergent result
                res.is_valid = False
                if R <= -1:
                    res.warnings.append(
                        f"Oscillatory divergence (R = {R:.4f}): oscillations "
                        f"are growing with grid refinement. GCI is not "
                        f"applicable."
                    )
                else:
                    res.warnings.append(
                        f"Convergence ratio R = {R:.4f} indicates divergence. "
                        f"GCI is not applicable. Check mesh quality, solver "
                        f"settings, and whether the grids are in the "
                        f"asymptotic range."
                    )
                res.notes = (
                    "DIVERGENT: The solution does not converge with grid "
                    "refinement. GCI cannot be computed reliably. "
                    "Possible causes: insufficient grid quality, "
                    "non-asymptotic range, inadequate solver convergence, "
                    "or a modeling issue."
                )
                return res

            elif res.convergence_type == "oscillatory":
                # ----- OSCILLATORY — use bounding approach -----
                res.observed_order = _solve_observed_order(e21, e32, r21, r32)
                fs = safety_factor if safety_factor is not None else 3.0
                res.safety_factor = fs

                # For oscillatory: use max oscillation as uncertainty
                # Celik et al.: use the range of the oscillation
                # Note: u_num = GCI_fine * |f1| / Fs = osc_range / 2
                # (Fs cancels out), so the Fs value does not affect u_num.
                osc_range = abs(max(solutions[:3]) - min(solutions[:3]))
                osc_norm = ref if ref is not None else abs(f1)
                if osc_norm > 1e-30:
                    res.gci_fine = fs * (osc_range / osc_norm) / 2.0
                else:
                    # Near-zero normaliser — GCI% is meaningless but u_num
                    # (absolute) is still valid from osc_range / 2.
                    res.warnings.append(
                        "Fine-grid solution is near zero — GCI_fine cannot "
                        "be expressed as a relative value. The absolute "
                        "u_num is still computed from the oscillation range."
                    )

                # Richardson extrapolation not reliable for oscillatory
                res.richardson_extrapolation = float('nan')
                res.is_valid = True
                res.warnings.append(
                    f"Oscillatory convergence (R = {R:.4f}). "
                    f"Richardson extrapolation is unreliable. "
                    f"Using oscillation range with Fs = {fs:.2f} for GCI."
                )
                res.notes = (
                    f"Oscillatory convergence detected. The solution "
                    f"oscillates between grid levels. GCI is computed from "
                    f"the oscillation range with a conservative safety "
                    f"factor Fs = {fs:.2f}."
                )

            else:
                # ----- MONOTONIC CONVERGENCE — standard procedure -----
                # Solve for observed order p
                p = _solve_observed_order(e21, e32, r21, r32)
                res.observed_order = p

                if np.isnan(p) or p <= 0:
                    res.warnings.append(
                        f"Could not determine valid observed order of "
                        f"accuracy. Using theoretical order p = "
                        f"{theoretical_order:.1f} as fallback."
                    )
                    p = theoretical_order
                    res.observed_order = p

                # Check if p is physically reasonable
                if p > 2.0 * theoretical_order + 1.0:
                    res.warnings.append(
                        f"Observed order p = {p:.2f} exceeds "
                        f"2 x theoretical ({2*theoretical_order:.1f}). "
                        f"This may indicate the grids are not in the "
                        f"asymptotic range, or there is error cancellation."
                    )

                # Safety factor
                fs = safety_factor if safety_factor is not None else 1.25
                res.safety_factor = fs

                # Richardson extrapolation (Celik Eq. 6)
                rp21 = _safe_rp(r21, p)
                rp32 = _safe_rp(r32, p)
                if np.isfinite(rp21) and abs(rp21 - 1.0) > 1e-30:
                    res.richardson_extrapolation = (
                        f1 + (f1 - f2) / (rp21 - 1.0)
                    )
                else:
                    res.richardson_extrapolation = f1
                    if not np.isfinite(rp21):
                        res.warnings.append(
                            f"r^p overflow (r={r21:.4f}, p={p:.2f}). "
                            f"GCI values may be unreliable."
                        )

                # GCI_fine (Celik Eq. 9)
                gci_norm1 = ref if ref is not None else abs(f1)
                ea21 = abs((f2 - f1) / gci_norm1) if gci_norm1 > 1e-30 else 0.0
                if np.isfinite(rp21) and abs(rp21 - 1.0) > 1e-30:
                    res.gci_fine = fs * ea21 / (rp21 - 1.0)
                else:
                    res.gci_fine = float('nan')

                # GCI_coarse (Celik Eq. 10)
                gci_norm2 = ref if ref is not None else abs(f2)
                ea32 = abs((f3 - f2) / gci_norm2) if gci_norm2 > 1e-30 else 0.0
                if np.isfinite(rp32) and abs(rp32 - 1.0) > 1e-30:
                    res.gci_coarse = fs * ea32 / (rp32 - 1.0)
                else:
                    res.gci_coarse = 0.0

                # Asymptotic ratio (Celik Eq. 11) — should be ~1.0
                if abs(res.gci_fine) > 1e-30 and np.isfinite(rp21):
                    res.asymptotic_ratio = (
                        res.gci_coarse / (rp21 * res.gci_fine)
                    )
                else:
                    res.asymptotic_ratio = float('nan')

                res.is_valid = True
                res.notes = (
                    f"Monotonic convergence. Observed order p = {p:.4f} "
                    f"(theoretical = {theoretical_order:.1f}). "
                    f"Safety factor Fs = {fs:.2f}."
                )

        # ----- Additional grid pairs (for N > 3) -----
        if n_grids > 3:
            extra_info = []
            for i in range(1, n_grids - 2):
                fa, fb, fc = solutions[i], solutions[i+1], solutions[i+2]
                ea = fb - fa
                eb = fc - fb
                if abs(eb) > 1e-30:
                    R_extra = ea / eb
                    p_extra = _solve_observed_order(ea, eb, ratios[i], ratios[i+1])
                    extra_info.append(
                        f"  Grids {i+1}-{i+2}-{i+3}: R = {R_extra:.4f}, "
                        f"p = {p_extra:.4f}"
                    )
            if extra_info:
                res.notes += (
                    "\n\nAdditional grid triplets:\n" +
                    "\n".join(extra_info)
                )

    # ------------------------------------------------------------------
    # 3. Convert GCI to standard uncertainty u_num (fine grid)
    # ------------------------------------------------------------------
    # u_num = |f1 - f_RE| (the Richardson extrapolation error on the fine grid).
    # Derivation: GCI_fine = Fs * |f2-f1| / (norm * (r^p - 1))
    #   When norm = |f1| (default): u_num = GCI_fine * |f1| / Fs = |f2-f1|/(r^p-1) = |f1 - f_RE|
    # Using |f1 - f_RE| directly is always correct, even when a custom
    # reference_scale is set (the norm cancels differently in the GCI path).
    # Skip if already set by effectively grid-independent check
    if not res.effectively_grid_independent:
        f_ext_for_unum = res.richardson_extrapolation
        if not np.isnan(f_ext_for_unum):
            res.u_num = abs(solutions[0] - f_ext_for_unum)
            if abs(solutions[0]) > 1e-30:
                res.u_num_pct = (res.u_num / abs(solutions[0])) * 100.0
            else:
                res.u_num_pct = 0.0
    if not res.effectively_grid_independent and res.convergence_type == "oscillatory":
        # Oscillatory: u_num = half the oscillation range (independent of ref_scale).
        # GCI_fine was normalized by ref, but u_num must be absolute.
        osc_range = abs(max(solutions[:min(n_grids, 3)])
                        - min(solutions[:min(n_grids, 3)]))
        res.u_num = osc_range / 2.0
        if abs(solutions[0]) > 1e-30:
            res.u_num_pct = (res.u_num / abs(solutions[0])) * 100.0
        else:
            res.u_num_pct = 0.0

    # ------------------------------------------------------------------
    # 4. Per-grid uncertainty estimates (for production grid selection)
    # ------------------------------------------------------------------
    # Once we know f_exact from Richardson extrapolation, the estimated
    # discretization error on ANY grid i is simply |f_i - f_exact|.
    # This is the key feature for "my production grid is not the finest."
    f_ext = res.richardson_extrapolation
    if res.effectively_grid_independent:
        # Effectively grid-independent: use spread/2 for all grids
        for i in range(n_grids):
            res.per_grid_error.append(res.u_num)
            res.per_grid_u_num.append(res.u_num)
            if abs(solutions[i]) > 1e-30:
                res.per_grid_u_pct.append(
                    (res.u_num / abs(solutions[i])) * 100.0)
            else:
                res.per_grid_u_pct.append(0.0)
    elif not np.isnan(f_ext) and res.is_valid:
        # Standard: per-grid u_num from Richardson extrapolation
        for i in range(n_grids):
            err_abs = abs(solutions[i] - f_ext)
            res.per_grid_error.append(err_abs)
            u_i = err_abs
            res.per_grid_u_num.append(u_i)
            if abs(solutions[i]) > 1e-30:
                res.per_grid_u_pct.append((u_i / abs(solutions[i])) * 100.0)
            else:
                res.per_grid_u_pct.append(0.0)
    else:
        # Fallback: only have fine-grid GCI estimate
        for i in range(n_grids):
            if i == 0 and not np.isnan(res.u_num):
                res.per_grid_error.append(res.u_num)
                res.per_grid_u_num.append(res.u_num)
                res.per_grid_u_pct.append(res.u_num_pct)
            else:
                res.per_grid_error.append(float('nan'))
                res.per_grid_u_num.append(float('nan'))
                res.per_grid_u_pct.append(float('nan'))

    # ------------------------------------------------------------------
    # 5. Directional u_num (one-sided validation support)
    # ------------------------------------------------------------------
    # When Richardson extrapolation is available, determine whether the
    # production grid over- or under-predicts relative to the grid-
    # converged solution.  If the bias is conservative for the stated
    # direction of concern, u_num_directional = 0.
    res.direction_of_concern = direction_of_concern
    tgt = res.target_grid_idx
    f_ext_dir = res.richardson_extrapolation

    if (res.is_valid and not np.isnan(f_ext_dir)
            and 0 <= tgt < len(solutions)):
        bias = solutions[tgt] - f_ext_dir
        if abs(bias) < 1e-30:
            res.discretization_bias_sign = "neutral"
        elif bias > 0:
            res.discretization_bias_sign = "high"   # production overpredicts
        else:
            res.discretization_bias_sign = "low"    # production underpredicts

        # Standard u_num for the production grid
        std_u = (res.per_grid_u_num[tgt]
                 if tgt < len(res.per_grid_u_num) else res.u_num)
        if np.isnan(std_u):
            std_u = res.u_num

        if direction_of_concern == "both" or res.discretization_bias_sign == "neutral":
            res.u_num_directional = std_u
            res.directional_note = ""
        elif (direction_of_concern == "underprediction"
              and res.discretization_bias_sign == "high"):
            # Production overpredicts → conservative for underprediction
            res.u_num_directional = 0.0
            res.directional_note = (
                "Production grid overpredicts relative to the grid-converged "
                "solution. Discretization bias is conservative for "
                "underprediction — u_num contribution is zero in the "
                "direction of concern."
            )
        elif (direction_of_concern == "overprediction"
              and res.discretization_bias_sign == "low"):
            # Production underpredicts → conservative for overprediction
            res.u_num_directional = 0.0
            res.directional_note = (
                "Production grid underpredicts relative to the grid-converged "
                "solution. Discretization bias is conservative for "
                "overprediction — u_num contribution is zero in the "
                "direction of concern."
            )
        else:
            # Bias is adverse for the direction of concern
            res.u_num_directional = std_u
            res.directional_note = (
                f"Production grid {res.discretization_bias_sign}-predicts "
                f"relative to grid-converged solution. Discretization bias "
                f"is NOT conservative for {direction_of_concern} — "
                f"full u_num applies."
            )
    else:
        # Can't determine direction — use standard u_num
        res.u_num_directional = res.u_num
        res.directional_note = ""

    return res


def compute_multi_quantity_gci(
    quantities: List[dict],
    dim: int = 3,
    safety_factor: Optional[float] = None,
    theoretical_order: float = 2.0,
) -> List[GCIResult]:
    """
    Compute GCI for multiple quantities of interest simultaneously.

    Each dict in `quantities` must have keys:
        'name': str — descriptive name
        'solutions': list of float — values on each grid (finest first)
        'cell_counts': list of float — cell counts per grid

    Returns a list of GCIResult, one per quantity.
    """
    results = []
    for q in quantities:
        res = compute_gci(
            solutions=q['solutions'],
            cell_counts=q['cell_counts'],
            dim=dim,
            safety_factor=safety_factor,
            theoretical_order=theoretical_order,
        )
        res.notes = f"Quantity: {q.get('name', 'unnamed')}\n" + res.notes
        results.append(res)
    return results


# =============================================================================
# ALTERNATIVE u_num METHODS — Factor of Safety and Least Squares Root
# =============================================================================

def compute_fs_uncertainty(
    solutions: List[float],
    cell_counts: List[float],
    dim: int = 3,
    theoretical_order: float = 2.0,
    reference_scale: Optional[float] = None,
) -> dict:
    """Factor of Safety (FS) variant (after Xing & Stern 2010).

    Uses a variable safety factor that depends on the ratio of the
    observed order to the theoretical order (P = p_obs / p_th) and
    a correction factor CF:

        CF = r21^p / (r21^p - 1)
        delta_RE = (f1 - f2) / (r21^p - 1)    [RE error estimate]

        if |1 - CF| < 1/FS1:
            FS = FS1 * |1 - CF| + FS0
        else:
            FS = FS2 * |1 - CF| + FS0

    with FS0 = 1.6, FS1 = 2.45, FS2 = 14.8 (Xing & Stern 2010).
    u_num = FS * |delta_RE| / 2

    Note: This implementation follows the FS framework of Xing & Stern
    (2010) but uses the tool's own convergence-type routing (including
    oscillatory handling with Fs=3.0) and a /2 divisor to convert the
    FS bound to a 1-sigma estimate. Results are labeled as an "FS
    variant" to distinguish from the exact Xing & Stern procedure.

    Requires 3+ grids for observed order estimation.

    Returns dict with keys:
        method, fs_value, u_num, u_num_pct, delta_RE, correction_factor,
        P_ratio, observed_order, convergence_type, is_valid, note
    """
    n = len(solutions)
    _invalid_fs = {
        "method": "FS variant (after Xing & Stern 2010)",
        "is_valid": False,
        "u_num": float('nan'),
        "u_num_pct": float('nan'),
    }
    if len(solutions) != len(cell_counts):
        return {**_invalid_fs,
                "note": (f"solutions ({len(solutions)}) and cell_counts "
                         f"({len(cell_counts)}) must have the same length.")}
    if n < 3:
        return {**_invalid_fs, "note": "FS method requires at least 3 grids."}
    if dim not in (1, 2, 3):
        return {**_invalid_fs, "note": f"dim must be 1, 2, or 3, got {dim}."}
    if any(c <= 0 or not np.isfinite(c) for c in cell_counts):
        return {**_invalid_fs,
                "note": "Cell counts must be positive finite numbers."}
    if any(not np.isfinite(s) for s in solutions):
        return {**_invalid_fs,
                "note": "Solution values must be finite numbers."}

    # Sort finest to coarsest (most cells first)
    paired = sorted(zip(cell_counts, solutions), key=lambda x: -x[0])
    cc = [c for c, _ in paired]
    ss = [s for _, s in paired]

    f1, f2, f3 = ss[0], ss[1], ss[2]
    h1 = cc[0] ** (-1.0 / dim)
    h2 = cc[1] ** (-1.0 / dim)
    h3 = cc[2] ** (-1.0 / dim)
    r21 = h2 / h1
    r32 = h3 / h2

    e21 = f2 - f1
    e32 = f3 - f2

    result = {
        "method": "FS variant (after Xing & Stern 2010)",
        "is_valid": False,
        "u_num": float('nan'),
        "u_num_pct": float('nan'),
        "delta_RE": float('nan'),
        "correction_factor": float('nan'),
        "P_ratio": float('nan'),
        "observed_order": float('nan'),
        "fs_value": float('nan'),
        "convergence_type": "unknown",
        "note": "",
    }

    if abs(e32) < 1e-30 and abs(e21) < 1e-30:
        result["convergence_type"] = "grid-independent"
        result["u_num"] = 0.0
        result["u_num_pct"] = 0.0
        result["is_valid"] = True
        result["note"] = "Grid-independent solution."
        return result

    if abs(e32) < 1e-30:
        result["convergence_type"] = "divergent"
        result["note"] = "e32 ~ 0 but e21 != 0; cannot estimate order."
        return result

    R = e21 / e32
    if R < 0:
        # Oscillatory or divergent
        ct = "oscillatory" if -1 < R < 0 else "divergent"
        result["convergence_type"] = ct
        if ct == "oscillatory":
            # FS method for oscillatory: uses max pairwise difference
            # with Fs=3.0 (differs from GCI, which uses the full range
            # of all three solutions; both are conservative estimates).
            osc_range = max(abs(f1 - f2), abs(f2 - f3))
            result["u_num"] = 3.0 * osc_range / 2.0
            if abs(f1) > 1e-30:
                result["u_num_pct"] = 100 * result["u_num"] / abs(f1)
            result["is_valid"] = True
            result["fs_value"] = 3.0
            result["note"] = (
                "Oscillatory convergence; FS method uses Fs=3.0 "
                "on oscillation range."
            )
        else:
            result["note"] = "Divergent; FS method not applicable."
        return result

    if R >= 1:
        result["convergence_type"] = "divergent"
        result["note"] = "R >= 1; divergent, FS method not applicable."
        return result

    # Monotonic convergence: compute observed order
    result["convergence_type"] = "monotonic"

    p_obs = _solve_observed_order(e21, e32, r21, r32)
    if np.isnan(p_obs) or p_obs > 1e6:
        p_obs = theoretical_order  # fallback to theoretical
    result["observed_order"] = p_obs

    # P ratio (distance from asymptotic range)
    P = p_obs / theoretical_order
    result["P_ratio"] = P

    # Correction factor
    rp21 = _safe_rp(r21, p_obs)
    if np.isfinite(rp21) and abs(rp21 - 1.0) > 1e-30:
        CF = rp21 / (rp21 - 1.0)
    else:
        CF = 1.0
    result["correction_factor"] = CF

    # Richardson extrapolation error estimate
    if np.isfinite(rp21) and abs(rp21 - 1.0) > 1e-30:
        delta_RE = e21 / (rp21 - 1.0)
    else:
        delta_RE = float('nan')
    result["delta_RE"] = delta_RE

    # Variable safety factor
    FS0 = 1.6
    FS1 = 2.45
    FS2 = 14.8
    abs_1_minus_CF = abs(1.0 - CF)

    if abs_1_minus_CF < 1.0 / FS1:
        fs_val = FS1 * abs_1_minus_CF + FS0
    else:
        fs_val = FS2 * abs_1_minus_CF + FS0

    result["fs_value"] = fs_val

    # Uncertainty estimate
    u_num = fs_val * abs(delta_RE) / 2.0
    result["u_num"] = u_num
    pct_denom = reference_scale if reference_scale is not None else abs(f1)
    if pct_denom > 1e-30:
        result["u_num_pct"] = 100 * u_num / pct_denom
    else:
        result["u_num_pct"] = 0.0
    result["is_valid"] = True
    result["note"] = (
        f"P = p/p_th = {P:.3f}, CF = {CF:.4f}, "
        f"FS = {fs_val:.3f}"
    )
    return result


def compute_lsr_uncertainty(
    solutions: List[float],
    cell_counts: List[float],
    dim: int = 3,
    theoretical_order: float = 2.0,
    reference_scale: Optional[float] = None,
) -> dict:
    """LSR variant with AICc model selection (after Eca & Hoekstra 2014).

    Fits four power-law models to solution data from 4+ grids:

        Model 1: phi = phi_0 + alpha * h^p           (p estimated)
        Model 2: phi = phi_0 + alpha * h^p_th        (p = theoretical)
        Model 3: phi = phi_0 + alpha * h + beta * h^2 (two-term)
        Model 4: phi = phi_0 + alpha * h              (first order)

    Selects the best fit by AICc (corrected Akaike Information Criterion).
    Combines the extrapolation error with the model standard deviation
    via RSS (root-sum-square).

    Note: This implementation enhances the original Eca & Hoekstra (2014)
    approach in two ways: (1) model selection uses AICc instead of minimum
    standard deviation, which penalizes model complexity and prevents
    overfitting; and (2) the final uncertainty combines extrapolation
    error and model std via RSS rather than the paper's additive form.
    These are improvements that make the method more robust for small
    numbers of grids.

    Requires 4+ grids.

    Returns dict with keys:
        method, u_num, u_num_pct, phi_0, best_model, model_results,
        is_valid, note
    """
    from scipy.optimize import curve_fit

    n = len(solutions)
    _invalid_lsr = {
        "method": "LSR variant with AICc (after Eca & Hoekstra 2014)",
        "is_valid": False,
        "u_num": float('nan'),
        "u_num_pct": float('nan'),
    }
    if len(solutions) != len(cell_counts):
        return {**_invalid_lsr,
                "note": (f"solutions ({len(solutions)}) and cell_counts "
                         f"({len(cell_counts)}) must have the same length.")}
    if n < 4:
        return {**_invalid_lsr,
                "note": f"LSR method requires 4+ grids (got {n})."}
    if dim not in (1, 2, 3):
        return {**_invalid_lsr, "note": f"dim must be 1, 2, or 3, got {dim}."}
    if any(c <= 0 or not np.isfinite(c) for c in cell_counts):
        return {**_invalid_lsr,
                "note": "Cell counts must be positive finite numbers."}
    if any(not np.isfinite(s) for s in solutions):
        return {**_invalid_lsr,
                "note": "Solution values must be finite numbers."}

    # Sort finest to coarsest
    paired = sorted(zip(cell_counts, solutions), key=lambda x: -x[0])
    cc = [c for c, _ in paired]
    ss = [s for _, s in paired]

    h = np.array([c ** (-1.0 / dim) for c in cc])
    phi = np.array(ss)

    result = {
        "method": "LSR variant with AICc (after Eca & Hoekstra 2014)",
        "is_valid": False,
        "u_num": float('nan'),
        "u_num_pct": float('nan'),
        "phi_0": float('nan'),
        "best_model": None,
        "model_results": {},
        "note": "",
    }

    models: dict = {}

    # --- Model 1: phi = phi_0 + alpha * h^p (p free) ---
    try:
        def model1(hh, phi0, alpha, p):
            return phi0 + alpha * hh**p

        p0 = [phi[0], (phi[-1] - phi[0]) / h[-1]**2, 2.0]
        popt, _ = curve_fit(model1, h, phi, p0=p0, maxfev=5000)
        phi0_1, alpha_1, p_1 = popt
        if p_1 < 0:
            raise ValueError("Negative order — divergent model, rejected")
        if not all(np.isfinite(popt)):
            raise ValueError("Non-finite parameters from curve_fit")
        resid_1 = phi - model1(h, *popt)
        std_1 = np.sqrt(np.sum(resid_1**2) / max(len(resid_1) - len(popt), 1))
        models["M1"] = {
            "phi_0": phi0_1, "alpha": alpha_1, "p": p_1,
            "std": std_1, "label": f"phi_0 + alpha*h^p (p={p_1:.2f})",
        }
    except Exception:
        pass

    # --- Model 2: phi = phi_0 + alpha * h^p_th (p fixed) ---
    try:
        def model2(hh, phi0, alpha):
            return phi0 + alpha * hh**theoretical_order

        p0 = [phi[0], (phi[-1] - phi[0]) / h[-1]**theoretical_order]
        popt, _ = curve_fit(model2, h, phi, p0=p0, maxfev=5000)
        if not all(np.isfinite(popt)):
            raise ValueError("Non-finite parameters from curve_fit")
        phi0_2, alpha_2 = popt
        resid_2 = phi - model2(h, *popt)
        std_2 = np.sqrt(np.sum(resid_2**2) / max(len(resid_2) - len(popt), 1))
        models["M2"] = {
            "phi_0": phi0_2, "alpha": alpha_2,
            "p": theoretical_order,
            "std": std_2,
            "label": f"phi_0 + alpha*h^{theoretical_order:.0f} (fixed)",
        }
    except Exception:
        pass

    # --- Model 3: phi = phi_0 + alpha * h + beta * h^2 (two-term) ---
    try:
        def model3(hh, phi0, alpha, beta):
            return phi0 + alpha * hh + beta * hh**2

        p0 = [phi[0], 0.0, 0.0]
        popt, _ = curve_fit(model3, h, phi, p0=p0, maxfev=5000)
        if not all(np.isfinite(popt)):
            raise ValueError("Non-finite parameters from curve_fit")
        phi0_3, alpha_3, beta_3 = popt
        resid_3 = phi - model3(h, *popt)
        std_3 = np.sqrt(np.sum(resid_3**2) / max(len(resid_3) - len(popt), 1))
        models["M3"] = {
            "phi_0": phi0_3, "alpha": alpha_3, "beta": beta_3,
            "std": std_3,
            "label": "phi_0 + alpha*h + beta*h^2 (two-term)",
        }
    except Exception:
        pass

    # --- Model 4: phi = phi_0 + alpha * h (first order) ---
    try:
        def model4(hh, phi0, alpha):
            return phi0 + alpha * hh

        p0 = [phi[0], (phi[-1] - phi[0]) / h[-1]]
        popt, _ = curve_fit(model4, h, phi, p0=p0, maxfev=5000)
        if not all(np.isfinite(popt)):
            raise ValueError("Non-finite parameters from curve_fit")
        phi0_4, alpha_4 = popt
        resid_4 = phi - model4(h, *popt)
        std_4 = np.sqrt(np.sum(resid_4**2) / max(len(resid_4) - len(popt), 1))
        models["M4"] = {
            "phi_0": phi0_4, "alpha": alpha_4,
            "std": std_4,
            "label": "phi_0 + alpha*h (first order)",
        }
    except Exception:
        pass

    if not models:
        result["note"] = "All model fits failed."
        return result

    # Select best model using AICc (corrected Akaike Information Criterion)
    # to penalise model complexity and avoid overfitting.
    n_data = len(phi)
    for key, m in models.items():
        k = {"M1": 3, "M2": 2, "M3": 3, "M4": 2}.get(key, 2)
        rss = m["std"]**2 * max(n_data - k, 1)  # reconstruct RSS
        if rss > 0 and n_data > k + 1:
            aic = n_data * np.log(rss / n_data) + 2 * k
            # AICc correction for small samples
            aic += 2 * k * (k + 1) / max(n_data - k - 1, 1)
        else:
            aic = float('inf')
        m["aic"] = aic

    best_key = min(models, key=lambda k: models[k]["aic"])
    best = models[best_key]
    phi_0 = best["phi_0"]

    result["model_results"] = models
    result["best_model"] = best_key
    result["phi_0"] = phi_0

    # Error estimate: |phi_0 - phi_finest|
    delta = abs(phi_0 - phi[0])

    # Safety factor based on observed vs theoretical order
    # If Model 1 was fit, use its p; otherwise use theoretical
    p_obs = models["M1"]["p"] if "M1" in models else theoretical_order
    P = p_obs / theoretical_order
    if 0.5 <= P <= 2.0:
        fs_lsr = 1.25
    elif 0.1 <= P < 0.5 or 2.0 < P <= 5.0:
        fs_lsr = 1.6
    else:
        fs_lsr = 3.0

    u_num = fs_lsr * delta / 2.0
    # Also consider the model standard deviation
    # LSR uncertainty = sqrt( (fs * delta/2)^2 + std^2 )
    u_num_combined = np.sqrt(u_num**2 + best["std"]**2)

    result["u_num"] = u_num_combined
    pct_denom = reference_scale if reference_scale is not None else abs(phi[0])
    if pct_denom > 1e-30:
        result["u_num_pct"] = 100 * u_num_combined / pct_denom
    else:
        result["u_num_pct"] = 0.0
    result["is_valid"] = True
    result["note"] = (
        f"Best fit: {best['label']} (std={best['std']:.4g}). "
        f"phi_0 = {phi_0:.6g}, Fs_LSR = {fs_lsr:.2f}"
    )
    return result


# =============================================================================
# SPATIAL / FIELD GCI — Data Structures, Engine, Parsers
# =============================================================================

@dataclass
class SpatialGCISummary:
    """Aggregated results from a point-by-point spatial GCI analysis."""
    n_points_total: int = 0
    n_points_valid: int = 0
    n_monotonic: int = 0
    n_oscillatory: int = 0
    n_divergent: int = 0
    n_grid_independent: int = 0

    u_num_mean: float = 0.0
    u_num_median: float = 0.0
    u_num_p95: float = 0.0       # 95th percentile — recommended for V&V 20
    u_num_max: float = 0.0
    u_num_rms: float = 0.0
    u_num_std: float = 0.0

    p_mean: float = 0.0          # observed order statistics
    p_median: float = 0.0

    point_results: List[GCIResult] = field(default_factory=list)
    point_coords: object = None   # np.ndarray (N, 2 or 3)
    point_u_num: object = None    # np.ndarray (N,)
    point_convergence_type: List[str] = field(default_factory=list)

    recommended_u_num: float = 0.0  # = u_num_p95
    quantity_name: str = ""
    quantity_unit: str = ""


def compute_spatial_gci(
    solutions_per_grid: List[np.ndarray],
    cell_counts: List[float],
    coordinates: np.ndarray,
    dim: int = 3,
    safety_factor: Optional[float] = None,
    theoretical_order: float = 2.0,
    quantity_name: str = "",
    quantity_unit: str = "",
    include_oscillatory: bool = True,
    min_solution_threshold: float = 0.0,
) -> SpatialGCISummary:
    """Compute point-by-point GCI over a spatial field.

    Args:
        solutions_per_grid: List of arrays [fine_vals, med_vals, coarse_vals, ...]
            at common (interpolated) points.  Each array has shape (N,).
        cell_counts: Cell counts for each grid level.
        coordinates: (N, d) array of spatial coordinates.
        dim: Spatial dimension (2 or 3).
        safety_factor: GCI safety factor (None = auto).
        theoretical_order: Theoretical scheme order.
        quantity_name: Name of the field variable.
        quantity_unit: Unit string for the field variable.
        include_oscillatory: Include oscillatory points in statistics.
        min_solution_threshold: Skip points where |solution| < threshold.

    Returns:
        SpatialGCISummary with per-point results and distribution statistics.
    """
    n_grids = len(solutions_per_grid)
    n_points = len(solutions_per_grid[0])

    summary = SpatialGCISummary(
        n_points_total=n_points,
        quantity_name=quantity_name,
        quantity_unit=quantity_unit,
        point_coords=coordinates,
    )

    point_u = np.full(n_points, np.nan)
    point_conv = []
    point_results = []
    valid_p = []

    for i in range(n_points):
        sols = [solutions_per_grid[g][i] for g in range(n_grids)]

        # Skip near-zero solutions if threshold is set
        if min_solution_threshold > 0 and abs(sols[0]) < min_solution_threshold:
            point_conv.append("skipped")
            point_results.append(None)
            continue

        res = compute_gci(
            solutions=sols,
            cell_counts=list(cell_counts),
            dim=dim,
            safety_factor=safety_factor,
            theoretical_order=theoretical_order,
        )
        point_results.append(res)
        point_conv.append(res.convergence_type)

        if res.convergence_type == "monotonic":
            summary.n_monotonic += 1
            point_u[i] = res.u_num
            if not np.isnan(res.observed_order) and res.observed_order < 1e6:
                valid_p.append(res.observed_order)
        elif res.convergence_type == "oscillatory":
            summary.n_oscillatory += 1
            if include_oscillatory and res.u_num > 0:
                point_u[i] = res.u_num
                if not np.isnan(res.observed_order) and res.observed_order < 1e6:
                    valid_p.append(res.observed_order)
        elif res.convergence_type == "divergent":
            summary.n_divergent += 1
            # Leave point_u[i] as NaN (excluded from stats)
        elif res.convergence_type == "grid-independent":
            summary.n_grid_independent += 1
            point_u[i] = 0.0  # Valid with zero numerical uncertainty

    summary.point_results = point_results
    summary.point_u_num = point_u
    summary.point_convergence_type = point_conv

    # Compute distribution statistics on valid (non-NaN) points
    valid_mask = ~np.isnan(point_u)
    summary.n_points_valid = int(np.sum(valid_mask))

    if summary.n_points_valid > 0:
        u_valid = point_u[valid_mask]
        summary.u_num_mean = float(np.mean(u_valid))
        summary.u_num_median = float(np.median(u_valid))
        summary.u_num_p95 = float(np.percentile(u_valid, 95))
        summary.u_num_max = float(np.max(u_valid))
        summary.u_num_rms = float(np.sqrt(np.mean(u_valid ** 2)))
        summary.u_num_std = float(np.std(u_valid))
        summary.recommended_u_num = summary.u_num_p95

    if valid_p:
        summary.p_mean = float(np.mean(valid_p))
        summary.p_median = float(np.median(valid_p))

    return summary


# --- Interpolation --------------------------------------------------------

def interpolate_field_idw(
    source_coords: np.ndarray,
    source_values: np.ndarray,
    target_coords: np.ndarray,
    k: int = 8,
    power: float = 2.0,
) -> np.ndarray:
    """Inverse-distance-weighted interpolation using cKDTree.

    Args:
        source_coords: (M, d) coordinates of known points.
        source_values: (M,) known field values.
        target_coords: (N, d) coordinates where we need values.
        k: Number of nearest neighbors to use.
        power: IDW exponent (2.0 = standard quadratic weighting).

    Returns:
        (N,) interpolated values at target locations.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(source_coords)
    k_actual = min(k, len(source_coords))
    distances, indices = tree.query(target_coords, k=k_actual)

    # Handle case where k=1 (distances/indices are 1-D)
    if k_actual == 1:
        return source_values[indices]

    # Replace zero distances with tiny value to avoid division by zero
    distances = np.maximum(distances, 1e-30)

    weights = 1.0 / distances ** power
    weight_sums = np.sum(weights, axis=1, keepdims=True)
    weights_norm = weights / weight_sums

    # Gather neighbor values
    neighbor_vals = source_values[indices]  # (N, k)
    result = np.sum(weights_norm * neighbor_vals, axis=1)
    return result


def farthest_point_sampling(
    coords: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    """Select *n_samples* points from *coords* that are approximately
    equally spaced using the greedy farthest-point (maximin) algorithm.

    This works in 3-D Euclidean space, so it naturally handles curved
    surfaces (e.g. a wing leading edge) because nearby points in 3-D
    remain close even when the surface folds back on itself.

    Args:
        coords: (M, d) coordinate array of all available points.
        n_samples: Desired number of sample points (clamped to M).

    Returns:
        (n_samples,) integer index array into *coords*.
    """
    n_total = len(coords)
    n_samples = min(n_samples, n_total)
    if n_samples <= 0:
        return np.array([], dtype=int)

    # Start with the point closest to the centroid
    centroid = coords.mean(axis=0)
    dists_to_centroid = np.linalg.norm(coords - centroid, axis=1)
    first_idx = int(np.argmin(dists_to_centroid))

    selected = np.empty(n_samples, dtype=int)
    selected[0] = first_idx

    # min_dist[i] = min distance from point i to any selected point
    min_dist = np.linalg.norm(coords - coords[first_idx], axis=1)

    for s in range(1, n_samples):
        # Pick the point farthest from all already-selected points
        next_idx = int(np.argmax(min_dist))
        selected[s] = next_idx
        # Update minimum distances
        new_dists = np.linalg.norm(coords - coords[next_idx], axis=1)
        np.minimum(min_dist, new_dists, out=min_dist)

    return selected


# --- File Parsers ----------------------------------------------------------

def parse_csv_field(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a CSV with columns: x, y, [z], value.

    Supports header row (auto-detected) or no header.
    Returns (coords, values) where coords is (N, d) and values is (N,).
    """
    # Guard against very large files that could stall the UI
    file_size = os.path.getsize(filepath)
    if file_size > 100 * 1024 * 1024:  # 100 MB
        warnings.warn(
            f"File is very large ({file_size / (1024*1024):.0f} MB). "
            f"Consider subsampling for spatial analysis.",
            stacklevel=2,
        )

    lines = []
    line_numbers = []  # actual 1-based file line number for each entry
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for file_lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            lines.append(line)
            line_numbers.append(file_lineno)

    if not lines:
        raise ValueError(f"No data found in {filepath}")

    # Detect header: if first line contains non-numeric tokens
    first_tokens = lines[0].replace(',', '\t').split('\t')
    has_header = False
    for t in first_tokens:
        t = t.strip()
        if t:
            try:
                float(t)
            except ValueError:
                has_header = True
                break

    data_start = 1 if has_header else 0
    rows = []
    row_file_lines = []  # actual file line number for each data row
    bad_token_rows = []  # (file_line, token_value) for error reporting
    for i, line in enumerate(lines[data_start:]):
        file_ln = line_numbers[data_start + i]
        tokens = line.replace(',', '\t').split('\t')
        vals = []
        for t in tokens:
            t = t.strip()
            if t:
                try:
                    vals.append(float(t))
                except ValueError:
                    vals.append(float('nan'))  # preserve column alignment
                    bad_token_rows.append((file_ln, t))
        if vals:
            rows.append(vals)
            row_file_lines.append(file_ln)

    if not rows:
        raise ValueError(f"No numeric data rows in {filepath}")

    if bad_token_rows:
        examples = bad_token_rows[:5]
        detail = "; ".join(f"line {r}: '{v}'" for r, v in examples)
        if len(bad_token_rows) > 5:
            detail += f" ... and {len(bad_token_rows) - 5} more"
        raise ValueError(
            f"Non-numeric values found in '{os.path.basename(filepath)}' — "
            f"{len(bad_token_rows)} bad token(s): {detail}. "
            f"Please clean the CSV so every data cell is a number."
        )

    # Verify consistent column count across all rows (use file line numbers)
    expected_cols = len(rows[0])
    ragged = [(row_file_lines[i], len(r))
              for i, r in enumerate(rows) if len(r) != expected_cols]
    if ragged:
        bad_lines = [ln for ln, _ in ragged[:5]]
        raise ValueError(
            f"Inconsistent column count in '{os.path.basename(filepath)}': "
            f"line {row_file_lines[0]} has {expected_cols} columns but line(s) "
            f"{bad_lines} differ. Check for missing/extra values."
        )

    n_cols = expected_cols
    if n_cols < 3:
        raise ValueError(
            f"Need at least 3 columns (x, y, value), got {n_cols} in "
            f"'{os.path.basename(filepath)}'.")
    if n_cols > 4:
        raise ValueError(
            f"Single-field CSV expects 3 columns (x, y, value) for 2D or "
            f"4 columns (x, y, z, value) for 3D, but got {n_cols} columns "
            f"in '{os.path.basename(filepath)}'. If the file contains "
            f"multiple grid solutions, use the pre-interpolated CSV loader "
            f"instead.")

    data = np.array(rows)
    if n_cols == 3:
        # 2D: x, y, value
        coords = data[:, :2]
        values = data[:, 2]
    else:
        # 3D: x, y, z, value
        coords = data[:, :3]
        values = data[:, 3]

    return coords, values


def parse_fluent_prof(filepath: str, field_name: str = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a Fluent .prof surface data file.

    Handles multiple surface definitions in a single file by combining
    all surfaces.  E.g. if the .prof contains upper-lipskin and
    lower-lipskin sections, all points are merged into one dataset.

    Fluent .prof format (single or multi-surface):
        ((surface-name point N)
         (x coord1 coord2 ... coordN)
         (y coord1 coord2 ... coordN)
         (z coord1 coord2 ... coordN)   <-- if 3D
         (field-name val1 val2 ... valN)
         ...
        )

    Returns (coords, values).
    """
    import re

    with open(filepath, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    # ------------------------------------------------------------------
    # Try multi-profile parsing first: ((name point|line|mesh|radial|axial ...))
    # Each top-level section starts with "((" and contains "(key v1 v2 ...)"
    # sub-blocks for coordinates and fields. We use balanced-parentheses
    # parsing because nested blocks are common in Fluent profile files.
    # ------------------------------------------------------------------
    # Fluent standard profile sections are typically:
    #   ((name point|line|mesh|radial|axial n[ m])
    # Some legacy exports omit the explicit profile type; treat those as point.
    header_pat = re.compile(
        r'\(\(([^()\s]+)\s+(?:(point|line|mesh|radial|axial)\s+)?'
        r'([0-9]+(?:\s+[0-9]+)?)\)',
        re.IGNORECASE,
    )

    sections: list = []
    for m in header_pat.finditer(content):
        # m.start() is the position of the outer '(' of '(('
        # Walk forward from after the header to find the matching ')'
        depth = 1  # we consumed the outer '(' at m.start()
        pos = m.end()
        while pos < len(content) and depth > 0:
            ch = content[pos]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            pos += 1
        # body is everything between the header close and the section close
        body = content[m.end():pos - 1]
        surf_name = m.group(1).strip().strip('"').lower()
        profile_type = (m.group(2) or "point").lower()
        sections.append((surf_name, profile_type, m.group(3), body))

    coord_keys = {
        'x', 'y', 'z', 'x-coordinate', 'y-coordinate', 'z-coordinate',
        'point', 'r', 'radius', 'axis', 'time', 'angle',
    }
    requested_field = field_name.lower().strip() if field_name else None

    if sections:
        # --- Multi-profile path ---
        all_coords: List[np.ndarray] = []
        all_vals: List[np.ndarray] = []
        detected_field: Optional[str] = None

        block_pat = re.compile(r'\((\S+)([\s\d.eE+\-]+)\)', re.DOTALL)

        for _surf_name, _profile_type, _n_def, body in sections:
            blocks = block_pat.findall(body)
            local: dict = {}
            for key, vals_str in blocks:
                key_norm = key.strip().strip('"').lower()
                vals = [float(v) for v in vals_str.strip().split()]
                local[key_norm] = np.array(vals, dtype=float)

            # Coordinate extraction by documented Fluent profile types.
            x = local.get('x', local.get('x-coordinate'))
            y = local.get('y', local.get('y-coordinate'))
            z = local.get('z', local.get('z-coordinate'))
            r = local.get('r', local.get('radius'))
            axis = local.get('axis')

            coords = None
            if x is not None and y is not None:
                if z is not None and len(z) == len(x):
                    coords = np.column_stack([x, y, z])
                else:
                    coords = np.column_stack([x, y])
            elif r is not None:
                coords = np.column_stack([r])
            elif axis is not None:
                coords = np.column_stack([axis])

            if coords is None or coords.shape[0] == 0:
                continue

            n_pts = coords.shape[0]

            # Extract field value.
            if requested_field and requested_field in local and len(local[requested_field]) == n_pts:
                vals = local[requested_field]
                all_vals.append(vals)
                if detected_field is None:
                    detected_field = requested_field
            else:
                remaining = [(k, v) for k, v in local.items()
                             if k not in coord_keys and len(v) == n_pts]
                if remaining:
                    if detected_field is None:
                        detected_field = remaining[0][0]
                    if detected_field in local:
                        all_vals.append(local[detected_field])
                    else:
                        all_vals.append(remaining[0][1])
                else:
                    continue

            all_coords.append(coords)

        if not all_coords:
            raise ValueError(
                f"No valid surface data found in {filepath}")

        if not all_vals:
            raise ValueError(
                f"No field data found in {filepath}. "
                f"Profiles: {[s[0] for s in sections]}")

        max_dim = max(c.shape[1] for c in all_coords)
        coords_padded = []
        for c in all_coords:
            if c.shape[1] < max_dim:
                c = np.pad(c, ((0, 0), (0, max_dim - c.shape[1])),
                           mode='constant', constant_values=0.0)
            coords_padded.append(c)
        coords = np.vstack(coords_padded)

        values = np.concatenate(all_vals)
        return coords, values

    # ------------------------------------------------------------------
    # Fallback: flat parsing for single-section files without header
    # ------------------------------------------------------------------
    pattern = r'\((\S+)\s+([\d.eE+\-\s]+)\)'
    matches = re.findall(pattern, content)

    if not matches:
        raise ValueError(
            f"No data sections found in Fluent .prof file: {filepath}")

    data_dict: dict = {}
    for name, vals_str in matches:
        vals = [float(v) for v in vals_str.strip().split()]
        data_dict[name.strip().strip('"').lower()] = np.array(vals, dtype=float)

    # Find coordinate fields
    x = data_dict.get('x', data_dict.get('x-coordinate', None))
    y = data_dict.get('y', data_dict.get('y-coordinate', None))
    z = data_dict.get('z', data_dict.get('z-coordinate', None))
    r = data_dict.get('r', data_dict.get('radius', None))
    axis = data_dict.get('axis', None)

    if x is not None and y is not None:
        if z is not None and len(z) == len(x):
            coords = np.column_stack([x, y, z])
        else:
            coords = np.column_stack([x, y])
    elif r is not None:
        coords = np.column_stack([r])
    elif axis is not None:
        coords = np.column_stack([axis])
    else:
        raise ValueError(
            f"Could not find coordinate fields (x/y[/z], r, or axis) in {filepath}")

    # Find the field of interest
    if requested_field and requested_field in data_dict:
        values = data_dict[requested_field]
    else:
        remaining = [(k, v) for k, v in data_dict.items()
                     if k not in coord_keys and len(v) == len(coords)]
        if not remaining:
            raise ValueError(
                f"No field data found in {filepath}. "
                f"Available sections: {list(data_dict.keys())}")
        values = remaining[0][1]

    return coords, values


def parse_pre_interpolated_csv(filepath: str
                                ) -> Tuple[np.ndarray, List[np.ndarray],
                                           List[str]]:
    """Parse a pre-interpolated CSV: x, y, [z], f_grid1, f_grid2, f_grid3, ...

    All grids already share the same point locations.
    Returns (coords, solutions_per_grid, grid_col_names).
    """
    lines_raw = []
    line_numbers_raw = []  # actual 1-based file line number for each entry
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        for file_lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            lines_raw.append(line)
            line_numbers_raw.append(file_lineno)

    if not lines_raw:
        raise ValueError(f"No data found in {filepath}")

    # Detect header
    first_tokens = lines_raw[0].replace(',', '\t').split('\t')
    has_header = False
    col_names = []
    for t in first_tokens:
        t = t.strip()
        if t:
            try:
                float(t)
            except ValueError:
                has_header = True
                break

    if has_header:
        col_names = [t.strip() for t in first_tokens if t.strip()]
        data_start = 1
    else:
        data_start = 0

    rows = []
    row_file_lines = []  # actual file line number for each data row
    bad_token_rows = []  # (file_line, token_value) for error reporting
    for i, line in enumerate(lines_raw[data_start:]):
        file_ln = line_numbers_raw[data_start + i]
        tokens = line.replace(',', '\t').split('\t')
        vals = []
        for t in tokens:
            t = t.strip()
            if t:
                try:
                    vals.append(float(t))
                except ValueError:
                    vals.append(float('nan'))  # preserve column alignment
                    bad_token_rows.append((file_ln, t))
        if vals:
            rows.append(vals)
            row_file_lines.append(file_ln)

    if not rows:
        raise ValueError(f"No numeric data rows in {filepath}")

    if bad_token_rows:
        examples = bad_token_rows[:5]
        detail = "; ".join(f"line {r}: '{v}'" for r, v in examples)
        if len(bad_token_rows) > 5:
            detail += f" ... and {len(bad_token_rows) - 5} more"
        raise ValueError(
            f"Non-numeric values found in '{os.path.basename(filepath)}' — "
            f"{len(bad_token_rows)} bad token(s): {detail}. "
            f"Please clean the CSV so every data cell is a number."
        )

    # Verify consistent column count across all rows (use file line numbers)
    expected_cols = len(rows[0])
    ragged = [(row_file_lines[i], len(r))
              for i, r in enumerate(rows) if len(r) != expected_cols]
    if ragged:
        bad_lines = [ln for ln, _ in ragged[:5]]
        raise ValueError(
            f"Inconsistent column count in '{os.path.basename(filepath)}': "
            f"line {row_file_lines[0]} has {expected_cols} columns but line(s) "
            f"{bad_lines} differ. Check for missing/extra values."
        )

    n_cols = expected_cols
    data = np.array(rows)

    # Determine coord vs field columns using headers if available
    # Default: 2D coords (x, y) + grid fields
    n_coord_cols = 2
    if col_names:
        # Use header names to identify coordinate columns
        coord_keywords = {'x', 'y', 'z', 'x_coord', 'y_coord', 'z_coord',
                          'x-coordinate', 'y-coordinate', 'z-coordinate'}
        n_coord_cols = 0
        for name in col_names:
            if name.lower().strip() in coord_keywords:
                n_coord_cols += 1
            else:
                break  # stop at first non-coordinate column
        if n_coord_cols < 2:
            n_coord_cols = 2  # minimum 2 coordinate columns
    elif n_cols == 5:
        # No headers, 5 columns: could be 2D (x,y + 3 grids) or
        # 3D (x,y,z + 2 grids).  Default to 2D (more common case).
        n_coord_cols = 2
        warnings.warn(
            f"Headerless CSV with 5 columns: assumed 2D "
            f"(2 coordinate columns + 3 grid solutions). If this "
            f"is actually 3D data (x, y, z + 2 grids), add a header "
            f"row with coordinate names (x, y, z) to disambiguate.",
            stacklevel=2,
        )
    elif n_cols >= 6:
        # No headers and 6+ columns: could be 2D with (n_cols-2) grids
        # or 3D with (n_cols-3) grids.
        # Heuristic: if the 3rd column (index 2) has a range/spread
        # similar to the first two columns (coordinates), treat it as a
        # z-coordinate (3D). If it looks more like a field value (range
        # similar to columns 3+), keep 2D. This resolves the ambiguity
        # for headerless x,y,z,g1,g2,g3 files.
        col2_range = data[:, 2].max() - data[:, 2].min() if len(data) > 1 else 0
        # Compare col 2 range to the avg range of the known-coordinate cols
        coord_ranges = []
        for ci in range(2):
            cr = data[:, ci].max() - data[:, ci].min() if len(data) > 1 else 0
            coord_ranges.append(cr)
        avg_coord_range = np.mean(coord_ranges) if coord_ranges else 0
        # Compare col 2 range to the avg range of candidate field cols (col 3+)
        field_ranges = []
        for ci in range(3, min(n_cols, 6)):
            fr = data[:, ci].max() - data[:, ci].min() if len(data) > 1 else 0
            field_ranges.append(fr)
        avg_field_range = np.mean(field_ranges) if field_ranges else 0
        # If col 2 range is closer to coordinate range than field range → 3D
        if (avg_coord_range > 0 and avg_field_range > 0
                and abs(col2_range - avg_coord_range)
                < abs(col2_range - avg_field_range)):
            n_coord_cols = 3
        else:
            n_coord_cols = 2  # default: safer 2D assumption
        # Heuristic is inherently fragile — warn the user so they can verify
        dim_label = "3D" if n_coord_cols == 3 else "2D"
        n_grids_guess = n_cols - n_coord_cols
        warnings.warn(
            f"Headerless CSV with {n_cols} columns: auto-detected as "
            f"{dim_label} ({n_coord_cols} coordinate columns + "
            f"{n_grids_guess} grid solutions) using a range-based "
            f"heuristic. If this is wrong, add a header row with "
            f"coordinate names (x, y, z) to disambiguate.",
            stacklevel=2,
        )

    coords = data[:, :n_coord_cols]
    n_grids = n_cols - n_coord_cols
    if n_grids < 2:
        raise ValueError(
            f"Need at least 2 grid solution columns, got {n_grids} in {filepath}")

    solutions = [data[:, n_coord_cols + g] for g in range(n_grids)]
    grid_names = (col_names[n_coord_cols:n_coord_cols + n_grids]
                  if col_names and len(col_names) >= n_coord_cols + n_grids
                  else [f"Grid {g+1}" for g in range(n_grids)])

    return coords, solutions, grid_names


# =============================================================================
# MAIN GUI — GCI Calculator Tab
# =============================================================================

class GCICalculatorTab(QWidget):
    """Main GCI calculation interface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: List[GCIResult] = []
        self._setup_ui()

    def _setup_ui(self):
        splitter = QSplitter(Qt.Horizontal, self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(splitter)

        # ---- LEFT PANEL: Input ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)
        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        splitter.addWidget(left_scroll)

        # -- Grid study setup --
        grp_setup = QGroupBox("Grid Study Setup")
        setup_form = QFormLayout(grp_setup)
        setup_form.setSpacing(6)

        self._cmb_n_grids = QComboBox()
        for n in range(2, 7):
            label = f"{n} grids"
            if n == 2:
                label += " (minimum — assumes order p)"
            elif n == 3:
                label += " (standard — computes order p)"
            else:
                label += " (extended — cross-checking)"
            self._cmb_n_grids.addItem(label, n)
        self._cmb_n_grids.setCurrentIndex(1)  # default to 3 grids
        self._cmb_n_grids.setToolTip(
            "Number of grid levels in your refinement study.\n\n"
            "2 grids: Minimum. Cannot compute observed order of accuracy.\n"
            "  Uses assumed order (theoretical) with conservative Fs = 3.0.\n"
            "  [Not recommended for publication or certification]\n\n"
            "3 grids: Standard procedure per Celik et al. (2008).\n"
            "  Computes observed order p via Richardson extrapolation.\n"
            "  Uses Fs = 1.25 safety factor. [Recommended]\n\n"
            "4+ grids: Extended study. Uses finest 3 grids for primary\n"
            "  GCI, additional grids for cross-checking and confidence."
        )
        setup_form.addRow("Number of grids:", self._cmb_n_grids)

        self._cmb_dim = QComboBox()
        self._cmb_dim.addItem("3D", 3)
        self._cmb_dim.addItem("2D", 2)
        self._cmb_dim.addItem("1D (temporal)", 1)
        self._cmb_dim.setToolTip(
            "Spatial dimension of your CFD problem.\n"
            "Affects how cell count converts to representative spacing:\n"
            "  h = (1/N)^(1/dim)\n"
            "\n"
            "Use 1D for temporal convergence studies where N = number\n"
            "of time steps (or 1/dt). This gives r = N_fine/N_coarse.\n"
            "Use 2D for 2D simulations, 3D for 3D."
        )
        setup_form.addRow("Dimensions:", self._cmb_dim)

        self._cmb_order_preset = QComboBox()
        for label, val in THEORETICAL_ORDER_PRESETS:
            self._cmb_order_preset.addItem(label, val)
        self._cmb_order_preset.setCurrentIndex(0)  # "Second-order" default
        self._cmb_order_preset.setToolTip(
            "Select the formal order of accuracy of your numerical scheme.\n"
            "This sets the theoretical order p used as fallback and for\n"
            "sanity-checking the observed order.\n\n"
            "Select 'Custom' to enter any value in the spinbox below."
        )
        setup_form.addRow("Scheme order:", self._cmb_order_preset)

        self._spn_theoretical_p = QDoubleSpinBox()
        self._spn_theoretical_p.setRange(1.0, 4.0)
        self._spn_theoretical_p.setValue(2.0)
        self._spn_theoretical_p.setSingleStep(0.5)
        self._spn_theoretical_p.setDecimals(1)
        self._spn_theoretical_p.setToolTip(
            "Theoretical (formal) order of your numerical scheme.\n\n"
            "  1.0 — First-order upwind\n"
            "  2.0 — Second-order (most CFD codes) [Default]\n"
            "  3.0 — Third-order (MUSCL, WENO-3)\n"
            "  4.0 — Fourth-order (spectral, high-order DG)\n\n"
            "Driven by the scheme order preset above.\n"
            "Edit directly only when 'Custom' is selected."
        )
        setup_form.addRow("Theoretical order p:", self._spn_theoretical_p)

        self._spn_safety = QDoubleSpinBox()
        self._spn_safety.setRange(0.0, 5.0)  # 0.0 = "Auto"
        self._spn_safety.setValue(0.0)
        self._spn_safety.setSingleStep(0.25)
        self._spn_safety.setDecimals(2)
        self._spn_safety.setSpecialValueText("Auto")
        self._spn_safety.setToolTip(
            "Safety factor Fs for GCI computation.\n\n"
            "Auto (recommended):\n"
            "  3-grid monotonic: Fs = 1.25 (Roache, 1998)\n"
            "  2-grid or oscillatory: Fs = 3.0 (conservative)\n\n"
            "Manual: set your own value (1.0 - 5.0).\n"
            "Fs = 1.0 is NOT recommended (no safety margin)."
        )
        setup_form.addRow("Safety factor Fs:", self._spn_safety)

        self._spn_ref_scale = QDoubleSpinBox()
        self._spn_ref_scale.setRange(0.0, 1e15)
        self._spn_ref_scale.setValue(0.0)
        self._spn_ref_scale.setDecimals(6)
        self._spn_ref_scale.setSpecialValueText("Auto (use f\u2081)")
        self._spn_ref_scale.setMinimum(0.0)
        self._spn_ref_scale.setToolTip(
            "Reference scale for normalizing relative errors.\n\n"
            "Auto (default): uses abs(f\u2081) — the fine-grid solution.\n\n"
            "Set a value when your solution is near zero (e.g.\n"
            "pressure differences, residuals) to avoid dividing\n"
            "by near-zero. Use a characteristic value of the\n"
            "quantity (e.g., freestream pressure, max temperature)."
        )
        self._grp_ref_scale = QGroupBox("Advanced: Reference Scale Configuration")
        self._grp_ref_scale.setCheckable(True)
        self._grp_ref_scale.setChecked(False)
        _ref_scale_layout = QFormLayout(self._grp_ref_scale)
        _ref_scale_layout.addRow("Reference scale:", self._spn_ref_scale)
        setup_form.addRow(self._grp_ref_scale)

        self._cmb_production = QComboBox()
        self._cmb_production.setToolTip(
            "Which grid is your PRODUCTION mesh?\n\n"
            "This is the grid you actually use for your analysis runs.\n"
            "It does NOT have to be the finest grid.\n\n"
            "The tool always uses the finest 3 grids to compute the\n"
            "observed order and Richardson extrapolation. But it will\n"
            "report u_num specifically for your production grid.\n\n"
            "Example: You run 5 grids for the study, but Grid 3 is\n"
            "the one you use in production. Set this to 'Grid 3' and\n"
            "the tool will tell you u_num for that grid."
        )
        setup_form.addRow("Production grid:", self._cmb_production)

        self._cmb_direction = QComboBox()
        self._cmb_direction.addItem("Both (two-sided)", "both")
        self._cmb_direction.addItem("Underprediction only", "underprediction")
        self._cmb_direction.addItem("Overprediction only", "overprediction")
        self._cmb_direction.setToolTip(
            "Direction of validation concern.\n\n"
            "If the discretization bias is conservative for your\n"
            "concern direction, u_num (directional) will be zero\n"
            "because mesh refinement cannot make the prediction\n"
            "worse in the direction you care about.\n\n"
            "Example: If finer grids predict cooler temperatures\n"
            "and you only care about underprediction, the production\n"
            "grid is biased warm (conservative) — u_num = 0."
        )
        setup_form.addRow("Direction of concern:", self._cmb_direction)

        left_layout.addWidget(grp_setup)

        # -- Grid data table --
        grp_data = QGroupBox("Grid Data (finest first, coarsest last)")
        data_layout = QVBoxLayout(grp_data)

        self._grid_table = QTableWidget()
        self._grid_table.setAlternatingRowColors(True)
        self._grid_table.setMinimumHeight(200)
        data_layout.addWidget(self._grid_table)

        # Quantity management
        qty_row = QHBoxLayout()
        self._btn_add_qty = QPushButton("+ Add Quantity")
        self._btn_add_qty.setToolTip(
            "Add another quantity of interest (e.g., a second\n"
            "thermocouple location, a different output variable).\n"
            "Each quantity gets its own GCI calculation."
        )
        self._btn_remove_qty = QPushButton("- Remove Last")
        self._btn_paste = QPushButton("Paste from Clipboard")
        self._btn_paste.setToolTip(
            "Paste grid data from Excel/spreadsheet.\n"
            "Format: rows = grids (finest first), columns = cell count + quantities."
        )
        qty_row.addWidget(self._btn_add_qty)
        qty_row.addWidget(self._btn_remove_qty)
        qty_row.addWidget(self._btn_paste)
        qty_row.addStretch()
        data_layout.addLayout(qty_row)

        left_layout.addWidget(grp_data)

        # -- Quantity properties (unit labels) --
        self._grp_qty_config = QGroupBox("Quantity Properties (units)")
        self._grp_qty_config.setToolTip(
            "Set the unit label for each quantity of interest.\n"
            "This is purely cosmetic \u2014 units appear in column headers,\n"
            "results text, carry-over box, Celik Table, and plots.\n"
            "No unit conversion is performed."
        )
        self._qty_config_layout = QFormLayout(self._grp_qty_config)
        self._qty_config_layout.setSpacing(4)
        left_layout.addWidget(self._grp_qty_config)

        # -- Compute button --
        self._btn_compute = QPushButton("Compute GCI")
        self._btn_compute.setStyleSheet(
            f"QPushButton {{ background-color: {DARK_COLORS['accent']}; "
            f"color: {DARK_COLORS['bg']}; font-weight: bold; "
            f"font-size: 14px; padding: 10px; }}"
            f"QPushButton:hover {{ background-color: {DARK_COLORS['accent_hover']}; }}"
        )
        left_layout.addWidget(self._btn_compute)

        # -- Guidance panels --
        self._guidance_convergence = GuidancePanel("Convergence Assessment")
        left_layout.addWidget(self._guidance_convergence)

        self._guidance_order = GuidancePanel("Order of Accuracy")
        left_layout.addWidget(self._guidance_order)

        self._guidance_asymptotic = GuidancePanel("Asymptotic Range Check")
        left_layout.addWidget(self._guidance_asymptotic)

        # -- Conservative bounding panel (shown only for divergent + small spread) --
        self._guidance_bounding = GuidancePanel("Conservative Bounding Estimate")
        self._guidance_bounding.setVisible(False)
        left_layout.addWidget(self._guidance_bounding)

        # -- Bounding threshold configuration --
        self._bounding_config_frame = QGroupBox("Advanced: Bounding Threshold Settings")
        self._bounding_config_frame.setCheckable(True)
        self._bounding_config_frame.setChecked(False)
        bnd_layout = QVBoxLayout(self._bounding_config_frame)
        bnd_layout.setContentsMargins(8, 6, 8, 6)
        bnd_layout.setSpacing(4)

        bnd_mode_row = QHBoxLayout()
        bnd_mode_row.setSpacing(6)
        bnd_mode_lbl = QLabel("Mode:")
        bnd_mode_lbl.setStyleSheet(f"color: {DARK_COLORS['fg_dim']};")
        bnd_mode_row.addWidget(bnd_mode_lbl)
        self._cmb_bnd_mode = QComboBox()
        self._cmb_bnd_mode.addItem("Auto-detect from unit", "auto")
        self._cmb_bnd_mode.addItem("Percentage of mean", "percentage")
        self._cmb_bnd_mode.addItem("Absolute value", "absolute")
        self._cmb_bnd_mode.setToolTip(
            "Auto-detect: uses absolute threshold for temperature units "
            "(°F, K, °C) and percentage for everything else (Pa, m/s, N, etc.)."
        )
        bnd_mode_row.addWidget(self._cmb_bnd_mode)
        bnd_mode_row.addStretch()
        bnd_layout.addLayout(bnd_mode_row)

        bnd_thresh_row = QHBoxLayout()
        bnd_thresh_row.setSpacing(6)
        bnd_pct_lbl = QLabel("% threshold:")
        bnd_pct_lbl.setStyleSheet(f"color: {DARK_COLORS['fg_dim']};")
        bnd_thresh_row.addWidget(bnd_pct_lbl)
        self._spn_bnd_pct = QDoubleSpinBox()
        self._spn_bnd_pct.setRange(0.1, 50.0)
        self._spn_bnd_pct.setValue(2.0)
        self._spn_bnd_pct.setSuffix(" %")
        self._spn_bnd_pct.setDecimals(1)
        self._spn_bnd_pct.setToolTip(
            "Maximum spread (as % of |mean|) to consider bounding. "
            "Default 2%. Increase if divergence is marginal."
        )
        self._spn_bnd_pct.setFixedWidth(90)
        bnd_thresh_row.addWidget(self._spn_bnd_pct)

        bnd_abs_lbl = QLabel("Abs threshold:")
        bnd_abs_lbl.setStyleSheet(f"color: {DARK_COLORS['fg_dim']};")
        bnd_thresh_row.addWidget(bnd_abs_lbl)
        self._spn_bnd_abs = QDoubleSpinBox()
        self._spn_bnd_abs.setRange(0.01, 1000.0)
        self._spn_bnd_abs.setValue(1.0)
        self._spn_bnd_abs.setDecimals(2)
        self._spn_bnd_abs.setToolTip(
            "Maximum absolute spread to consider bounding. "
            "Default 1.0 (e.g. 1°F for temperature). "
            "Increase if your quantity has a larger natural spread."
        )
        self._spn_bnd_abs.setFixedWidth(90)
        bnd_thresh_row.addWidget(self._spn_bnd_abs)
        bnd_thresh_row.addStretch()
        bnd_layout.addLayout(bnd_thresh_row)

        self._bounding_config_frame.setVisible(False)
        left_layout.addWidget(self._bounding_config_frame)

        left_layout.addStretch()

        # ---- RIGHT PANEL: Results (sub-tabs) ----
        self._right_tabs = QTabWidget()
        splitter.addWidget(self._right_tabs)

        # Sub-tab 1: Results text
        results_widget = QWidget()
        results_lay = QVBoxLayout(results_widget)
        results_lay.setContentsMargins(6, 6, 6, 6)
        # Carry-value copy button row
        carry_row = QHBoxLayout()
        self._btn_copy_carry = QPushButton("\U0001f4cb Copy Carry Value")
        self._btn_copy_carry.setToolTip(
            "Copy u_num carry-over value to clipboard, "
            "ready to paste into the Uncertainty Aggregator.")
        self._btn_copy_carry.setEnabled(False)
        self._btn_copy_carry.clicked.connect(self._copy_carry_to_clipboard)
        carry_row.addWidget(self._btn_copy_carry)
        carry_row.addStretch()
        results_lay.addLayout(carry_row)
        self._results_text = QPlainTextEdit()
        self._results_text.setReadOnly(True)
        ff = QFont("Consolas", 9)
        ff.setStyleHint(QFont.StyleHint.Monospace)
        self._results_text.setFont(ff)
        results_lay.addWidget(self._results_text)
        self._right_tabs.addTab(results_widget, "Results")

        # Sub-tab 2: Summary table
        table_widget = QWidget()
        table_lay = QVBoxLayout(table_widget)
        table_lay.setContentsMargins(6, 6, 6, 6)
        self._results_table = QTableWidget()
        self._results_table.setAlternatingRowColors(True)
        table_lay.addWidget(self._results_table)
        self._right_tabs.addTab(table_widget, "Summary Table")

        # Sub-tab 3: Convergence plot
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(6, 6, 6, 6)
        self._fig = Figure(figsize=(6, 4))
        self._canvas = FigureCanvas(self._fig)
        toolbar = NavigationToolbar(self._canvas, self)
        btn_copy = QPushButton("Copy to Clipboard")
        btn_copy.clicked.connect(lambda: copy_figure_to_clipboard(self._fig))
        btn_rq = QPushButton("Copy Report-Quality")
        btn_rq.setToolTip("Copy at 300 DPI with light report theme.")
        btn_rq.clicked.connect(lambda: copy_report_quality_figure(self._fig))
        btn_export = QPushButton("Export Figure Package...")
        btn_export.setToolTip("Export PNG (300+600 DPI), SVG, PDF, and JSON.")
        btn_export.clicked.connect(lambda: _export_figure_package_dialog(self._fig, self))
        self._btn_summary_card = QPushButton("Export Summary Card...")
        self._btn_summary_card.setToolTip(
            "Export a compact PNG summary card with convergence plot\n"
            "and key GCI metrics — ready for PowerPoint or reports.")
        self._btn_summary_card.setEnabled(False)
        self._btn_summary_card.clicked.connect(self._export_summary_card)
        tb_row = QHBoxLayout()
        tb_row.addWidget(toolbar)
        tb_row.addWidget(btn_copy)
        tb_row.addWidget(btn_rq)
        tb_row.addWidget(btn_export)
        tb_row.addWidget(self._btn_summary_card)
        plot_layout.addLayout(tb_row)
        plot_layout.addWidget(self._canvas)
        self._right_tabs.addTab(plot_widget, "Convergence Plot")

        # Sub-tab 4: Report statements
        report_widget = QWidget()
        report_lay = QVBoxLayout(report_widget)
        report_lay.setContentsMargins(6, 6, 6, 6)
        stmt_toolbar = QHBoxLayout()
        self._cmb_audience = QComboBox()
        self._cmb_audience.addItems([
            "Internal Engineering",
            "External Technical Report",
            "Regulatory Submission",
        ])
        self._cmb_audience.setCurrentIndex(1)  # default: External
        self._cmb_audience.setToolTip(
            "Statement audience mode:\n"
            "  • Internal Engineering — terse, numeric-heavy\n"
            "  • External Technical Report — balanced narrative + numbers\n"
            "  • Regulatory Submission — full formal language, explicit references"
        )
        self._cmb_audience.currentIndexChanged.connect(
            self._generate_report_statements
        )
        stmt_toolbar.addWidget(QLabel("Audience:"))
        stmt_toolbar.addWidget(self._cmb_audience)
        stmt_toolbar.addStretch()
        btn_copy_report = QPushButton("Copy Statements to Clipboard")
        btn_copy_report.clicked.connect(self._copy_report_to_clipboard)
        stmt_toolbar.addWidget(btn_copy_report)
        report_lay.addLayout(stmt_toolbar)
        self._report_text = QPlainTextEdit()
        self._report_text.setReadOnly(True)
        ff = QFont("Consolas", 9)
        ff.setStyleHint(QFont.StyleHint.Monospace)
        self._report_text.setFont(ff)
        report_lay.addWidget(self._report_text)
        self._right_tabs.addTab(report_widget, "Report Statements")

        # Sub-tab 5: Carry-Over Summary
        carry_widget = QWidget()
        carry_lay = QVBoxLayout(carry_widget)
        carry_lay.setContentsMargins(6, 6, 6, 6)

        carry_header = QLabel(
            "<b>Carry-Over Summary</b> — Values to transfer to "
            "the Uncertainty Aggregator")
        carry_header.setWordWrap(True)
        carry_header.setStyleSheet(
            f"color: {DARK_COLORS['accent']}; font-size: 13px; "
            f"padding: 4px 0;")
        carry_lay.addWidget(carry_header)

        self._carry_table = QTableWidget()
        self._carry_table.setColumnCount(8)
        self._carry_table.setHorizontalHeaderLabels([
            "Quantity", "u_num", "Unit", "Distribution",
            "DOF", "Sigma Basis", "Convergence", "Status"
        ])
        self._carry_table.setAlternatingRowColors(True)
        style_table(self._carry_table)
        carry_lay.addWidget(self._carry_table)

        self._carry_warnings = QPlainTextEdit()
        self._carry_warnings.setReadOnly(True)
        self._carry_warnings.setMaximumHeight(100)
        self._carry_warnings.setPlaceholderText(
            "Warnings and convergence notes will appear here after computation.")
        ff = QFont("Consolas", 9)
        ff.setStyleHint(QFont.StyleHint.Monospace)
        self._carry_warnings.setFont(ff)
        carry_lay.addWidget(self._carry_warnings)

        carry_btn_row = QHBoxLayout()
        self._btn_copy_all_carry = QPushButton(
            "\U0001f4cb Copy All Carry Values")
        self._btn_copy_all_carry.setEnabled(False)
        self._btn_copy_all_carry.clicked.connect(
            self._copy_all_carry_values)
        carry_btn_row.addWidget(self._btn_copy_all_carry)
        carry_btn_row.addStretch()
        carry_lay.addLayout(carry_btn_row)

        self._right_tabs.addTab(carry_widget, "Carry-Over Summary")

        # Set splitter proportions
        splitter.setSizes([450, 600])

        # ---- Connections ----
        self._cmb_n_grids.currentIndexChanged.connect(self._rebuild_table)
        self._btn_add_qty.clicked.connect(self._add_quantity)
        self._btn_remove_qty.clicked.connect(self._remove_quantity)
        self._btn_paste.clicked.connect(self._paste_data)
        self._btn_compute.clicked.connect(self._compute)
        self._cmb_order_preset.currentIndexChanged.connect(
            self._on_order_preset_changed)
        self._spn_theoretical_p.valueChanged.connect(
            self._on_theoretical_p_manual_change)

        # Mark results stale when inputs change
        self._cmb_n_grids.currentIndexChanged.connect(self._mark_results_stale)
        self._cmb_dim.currentIndexChanged.connect(self._mark_results_stale)
        self._spn_theoretical_p.valueChanged.connect(self._mark_results_stale)
        self._spn_safety.valueChanged.connect(self._mark_results_stale)
        self._spn_ref_scale.valueChanged.connect(self._mark_results_stale)
        self._grid_table.cellChanged.connect(self._mark_results_stale)
        self._cmb_production.currentIndexChanged.connect(self._mark_results_stale)
        self._cmb_direction.currentIndexChanged.connect(self._mark_results_stale)

        # Internal state
        self._results_stale = False
        self._n_quantities = 1
        self._quantity_names = ["Temperature"]
        self._quantity_units = ["K"]
        self._rebuild_table()
        self._rebuild_qty_config()

    # ------------------------------------------------------------------
    # QUANTITY UNIT CONFIGURATION
    # ------------------------------------------------------------------
    # SCHEME ORDER PRESET
    # ------------------------------------------------------------------

    def _on_order_preset_changed(self, index):
        """Set theoretical order spinbox from preset combo selection."""
        val = self._cmb_order_preset.currentData()
        if val is not None:
            self._spn_theoretical_p.blockSignals(True)
            self._spn_theoretical_p.setValue(val)
            self._spn_theoretical_p.blockSignals(False)
            self._mark_results_stale()

    def _on_theoretical_p_manual_change(self, value):
        """Switch preset combo to 'Custom' if spinbox edited to non-preset."""
        preset_val = self._cmb_order_preset.currentData()
        if preset_val is not None and abs(preset_val - value) > 0.01:
            # Find the Custom entry (data == None)
            for i in range(self._cmb_order_preset.count()):
                if self._cmb_order_preset.itemData(i) is None:
                    self._cmb_order_preset.blockSignals(True)
                    self._cmb_order_preset.setCurrentIndex(i)
                    self._cmb_order_preset.blockSignals(False)
                    break

    # ------------------------------------------------------------------
    # QUANTITY UNIT CONFIGURATION
    # ------------------------------------------------------------------

    def _rebuild_qty_config(self):
        """Rebuild the quantity properties panel (category + unit combos)."""
        # Clear old widgets
        while self._qty_config_layout.rowCount() > 0:
            self._qty_config_layout.removeRow(0)

        self._qty_cat_combos = []
        self._qty_unit_combos = []

        for i in range(self._n_quantities):
            name = (self._quantity_names[i]
                    if i < len(self._quantity_names)
                    else f"Quantity {i+1}")
            cur_unit = (self._quantity_units[i]
                        if i < len(self._quantity_units) else "")

            row_widget = QWidget()
            row_lay = QHBoxLayout(row_widget)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.setSpacing(6)

            cat_combo = QComboBox()
            cat_combo.setMinimumWidth(100)
            for cat in UNIT_PRESETS:
                cat_combo.addItem(cat)
            row_lay.addWidget(cat_combo)

            unit_combo = QComboBox()
            unit_combo.setEditable(True)
            unit_combo.setMinimumWidth(80)
            unit_combo.setToolTip(
                "Select a preset or type a custom unit string.\n"
                "Examples: BTU/hr, kg/m\u00b3, MW, etc."
            )
            row_lay.addWidget(unit_combo)

            self._qty_cat_combos.append(cat_combo)
            self._qty_unit_combos.append(unit_combo)
            self._qty_config_layout.addRow(f"{name}:", row_widget)

            # Detect the best-fit category for the current unit
            best_cat = "Other"
            for cat, units in UNIT_PRESETS.items():
                if cur_unit in units:
                    best_cat = cat
                    break
            cat_idx = cat_combo.findText(best_cat)
            if cat_idx >= 0:
                cat_combo.setCurrentIndex(cat_idx)
            self._populate_unit_combo(unit_combo, best_cat, cur_unit)

            # Connect category change to repopulate unit combo
            idx_capture = i  # capture loop variable
            cat_combo.currentTextChanged.connect(
                lambda text, idx=idx_capture: self._on_category_changed(idx, text)
            )
            unit_combo.currentTextChanged.connect(
                lambda text, idx=idx_capture: self._on_unit_changed(idx, text)
            )

    def _populate_unit_combo(self, combo, category, current_unit=""):
        """Populate a unit combo box from the given category."""
        combo.blockSignals(True)
        combo.clear()
        presets = UNIT_PRESETS.get(category, [])
        for u in presets:
            combo.addItem(u)
        # If current unit is not in the preset list, add it
        if current_unit and combo.findText(current_unit) < 0:
            combo.addItem(current_unit)
        # Select current unit
        idx = combo.findText(current_unit)
        if idx >= 0:
            combo.setCurrentIndex(idx)
        elif current_unit:
            combo.setEditText(current_unit)
        combo.blockSignals(False)

    def _on_category_changed(self, qty_idx, category):
        """Handle category combo change — repopulate unit combo."""
        if qty_idx < len(self._qty_unit_combos):
            combo = self._qty_unit_combos[qty_idx]
            presets = UNIT_PRESETS.get(category, [])
            first_unit = presets[0] if presets else ""
            self._populate_unit_combo(combo, category, first_unit)
            # Update the stored unit
            if qty_idx < len(self._quantity_units):
                self._quantity_units[qty_idx] = first_unit
            self._update_table_headers()

    def _on_unit_changed(self, qty_idx, unit_text):
        """Handle unit combo text change — store the selected unit and update headers."""
        if qty_idx < len(self._quantity_units):
            self._quantity_units[qty_idx] = unit_text
            self._update_table_headers()

    def _qty_label(self, idx):
        """Return 'Blade Temp (K)' or 'Blade Temp' if no unit."""
        name = (self._quantity_names[idx]
                if idx < len(self._quantity_names)
                else f"Quantity {idx+1}")
        unit = self._qty_unit(idx)
        return f"{name} ({unit})" if unit else name

    def _qty_unit(self, idx):
        """Return unit string or '' for a given quantity index."""
        if idx < len(self._quantity_units):
            return self._quantity_units[idx]
        return ""

    def _update_table_headers(self):
        """Update column headers with current quantity names and units.

        Called when unit or name changes so the table headers stay in sync
        without rebuilding the entire table (which would clear data).
        """
        headers = ["Cell Count"]
        for i, name in enumerate(self._quantity_names):
            unit = self._qty_unit(i)
            hdr = f"Solution: {name} ({unit})" if unit else f"Solution: {name}"
            headers.append(hdr)
        self._grid_table.setHorizontalHeaderLabels(headers)
        # Use Interactive mode to avoid layout thrashing when headers change
        header = self._grid_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Interactive)
        header.setDefaultSectionSize(160)
        header.setStretchLastSection(True)
        self._grid_table.setWordWrap(True)

    # ------------------------------------------------------------------
    # TABLE MANAGEMENT
    # ------------------------------------------------------------------

    def _rebuild_table(self):
        """Rebuild the grid data table when grid count changes."""
        n_grids = self._cmb_n_grids.currentData()
        n_cols = 1 + self._n_quantities  # cell count + N quantities

        self._grid_table.blockSignals(True)
        try:
            self._grid_table.setRowCount(n_grids)
            self._grid_table.setColumnCount(n_cols)

            # Set headers via shared method
            self._update_table_headers()

            # Row headers
            for i in range(n_grids):
                if i == 0:
                    label = f"Grid {i+1} (finest)"
                elif i == n_grids - 1:
                    label = f"Grid {i+1} (coarsest)"
                else:
                    label = f"Grid {i+1}"
                self._grid_table.setVerticalHeaderItem(
                    i, QTableWidgetItem(label)
                )

            # Set column widths — user-resizable
            style_table(self._grid_table)
        finally:
            self._grid_table.blockSignals(False)

        # Rebuild production grid combo
        prev_idx = self._cmb_production.currentIndex()
        self._cmb_production.blockSignals(True)
        self._cmb_production.clear()
        self._cmb_production.addItem("Grid 1 (finest)", 0)
        for i in range(1, n_grids):
            tag = " (coarsest)" if i == n_grids - 1 else ""
            self._cmb_production.addItem(f"Grid {i+1}{tag}", i)
        # Restore previous selection if valid, otherwise default to finest
        if 0 <= prev_idx < n_grids:
            self._cmb_production.setCurrentIndex(prev_idx)
        else:
            self._cmb_production.setCurrentIndex(0)
        self._cmb_production.blockSignals(False)

    def _add_quantity(self):
        """Add a quantity of interest column."""
        if self._n_quantities >= 10:
            QMessageBox.information(self, "Limit", "Maximum 10 quantities.")
            return
        self._n_quantities += 1
        self._quantity_names.append(f"Quantity {self._n_quantities}")
        self._quantity_units.append("")
        self._rebuild_table()
        self._rebuild_qty_config()

    def _remove_quantity(self):
        """Remove the last quantity of interest column."""
        if self._n_quantities <= 1:
            return
        self._n_quantities -= 1
        self._quantity_names.pop()
        self._quantity_units.pop()
        self._rebuild_table()
        self._rebuild_qty_config()

    def _paste_data(self):
        """Paste grid data from clipboard."""
        text = QApplication.clipboard().text()
        if not text or not text.strip():
            QMessageBox.information(self, "Clipboard Empty",
                                   "No data found on clipboard.")
            return

        rows = []
        for line in text.strip().split('\n'):
            tokens = line.replace(',', '\t').split('\t')
            vals = []
            for t in tokens:
                t = t.strip()
                if not t:
                    continue
                try:
                    vals.append(float(t))
                except ValueError:
                    pass
            if vals:
                rows.append(vals)

        if not rows:
            QMessageBox.warning(self, "Parse Error",
                                "No numeric data found in clipboard.")
            return

        # Determine grid count and quantities from data shape
        n_grids = len(rows)
        row_lengths = [len(r) for r in rows]
        n_cols = min(row_lengths)
        if max(row_lengths) != n_cols:
            QMessageBox.warning(
                self, "Ragged Data",
                f"Rows have different column counts "
                f"({min(row_lengths)}–{max(row_lengths)}). "
                f"Using the minimum ({n_cols} columns). "
                f"Extra values on longer rows will be ignored.\n\n"
                f"Check that each grid row has the same number of "
                f"quantities."
            )

        if n_cols < 2:
            QMessageBox.warning(self, "Parse Error",
                                "Need at least 2 columns (cell count + solution).")
            return

        # Update UI
        n_qty = n_cols - 1
        self._n_quantities = n_qty
        self._quantity_names = [f"Quantity {i+1}" for i in range(n_qty)]
        self._quantity_units = [""] * n_qty

        # Set grid count combo
        for idx in range(self._cmb_n_grids.count()):
            if self._cmb_n_grids.itemData(idx) == n_grids:
                self._cmb_n_grids.setCurrentIndex(idx)
                break
        else:
            # Add custom grid count if not in combo
            self._cmb_n_grids.addItem(f"{n_grids} grids (pasted)", n_grids)
            self._cmb_n_grids.setCurrentIndex(self._cmb_n_grids.count() - 1)

        self._rebuild_table()
        self._rebuild_qty_config()

        # Fill table
        for i, row in enumerate(rows):
            for j in range(min(n_cols, len(row))):
                item = QTableWidgetItem(f"{row[j]:.6g}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._grid_table.setItem(i, j, item)

        QMessageBox.information(
            self, "Data Pasted",
            f"Pasted {n_grids} grids x {n_qty} quantities."
        )

    # ------------------------------------------------------------------
    # COMPUTATION
    # ------------------------------------------------------------------

    def _read_table_data(self):
        """Read grid data from table. Returns (cell_counts, solution_lists)."""
        n_grids = self._cmb_n_grids.currentData()
        n_qty = self._n_quantities

        cell_counts = []
        solutions = [[] for _ in range(n_qty)]

        for i in range(n_grids):
            # Cell count
            item = self._grid_table.item(i, 0)
            if item is None or not item.text().strip():
                return None, None
            try:
                val = float(item.text())
                if not np.isfinite(val):
                    QMessageBox.warning(
                        self, "Invalid Input",
                        f"Grid {i+1} cell count is not a finite number "
                        f"(got {item.text()}).\nPlease enter a positive integer.")
                    return None, None
                cell_counts.append(val)
            except ValueError:
                QMessageBox.warning(
                    self, "Invalid Input",
                    f"Grid {i+1} cell count '{item.text()}' is not a valid number.")
                return None, None

            # Solutions
            for q in range(n_qty):
                item = self._grid_table.item(i, 1 + q)
                if item is None or not item.text().strip():
                    return None, None
                try:
                    sol_val = float(item.text())
                    if not np.isfinite(sol_val):
                        q_label = (self._quantity_names[q]
                                   if q < len(self._quantity_names)
                                   else f"Quantity {q+1}")
                        QMessageBox.warning(
                            self, "Invalid Input",
                            f"Grid {i+1}, {q_label}: solution value is not "
                            f"finite (got {item.text()}).\n"
                            f"Please enter a valid number.")
                        return None, None
                    solutions[q].append(sol_val)
                except ValueError:
                    q_label = (self._quantity_names[q]
                               if q < len(self._quantity_names)
                               else f"Quantity {q+1}")
                    QMessageBox.warning(
                        self, "Invalid Input",
                        f"Grid {i+1}, {q_label}: '{item.text()}' is not "
                        f"a valid number.")
                    return None, None

        return cell_counts, solutions

    def _mark_results_stale(self):
        """Visually indicate that displayed results are out of date."""
        if not self._results:
            return
        if not self._results_stale:
            self._results_stale = True
            self._btn_compute.setText("⟳ Recompute GCI")
            self._btn_compute.setToolTip(
                "Input data or settings have changed since the last computation."
            )

    def _compute(self):
        """Run the GCI computation."""
        cell_counts, solutions_list = self._read_table_data()
        if cell_counts is None:
            QMessageBox.warning(
                self, "Incomplete Data",
                "Please fill in all grid data cells before computing.\n"
                "Every cell needs a numeric value."
            )
            return

        # Validate ordering: cell counts should decrease (finest first)
        if not all(cell_counts[i] >= cell_counts[i+1]
                   for i in range(len(cell_counts) - 1)):
            # Ask user if they want to auto-sort
            reply = QMessageBox.question(
                self, "Grid Ordering",
                "Cell counts should be in descending order (finest first, "
                "coarsest last). The data appears to be in a different order.\n\n"
                "Would you like to auto-sort the data?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                # Sort all data by cell count descending
                indices = sorted(range(len(cell_counts)),
                                key=lambda i: cell_counts[i], reverse=True)
                cell_counts = [cell_counts[i] for i in indices]
                for q in range(len(solutions_list)):
                    solutions_list[q] = [solutions_list[q][i] for i in indices]
                # Write sorted data back to the table UI
                for row, idx in enumerate(indices):
                    self._grid_table.item(row, 0).setText(
                        f"{cell_counts[row]:.6g}")
                    for q in range(len(solutions_list)):
                        self._grid_table.item(row, 1 + q).setText(
                            f"{solutions_list[q][row]:.6g}")
            else:
                return

        # Validate: all cell counts must be positive
        for i, cc in enumerate(cell_counts):
            if cc <= 0:
                QMessageBox.warning(
                    self, "Invalid Cell Count",
                    f"Grid {i+1} has a non-positive cell count ({cc:.6g}).\n"
                    f"All cell counts must be greater than zero."
                )
                return

        # Validate: check for identical solutions across all grids
        for q in range(len(solutions_list)):
            unique_vals = set(solutions_list[q])
            if len(unique_vals) == 1:
                q_label = self._qty_label(q)
                QMessageBox.warning(
                    self, "Identical Solutions",
                    f"All {len(solutions_list[q])} grid solutions for "
                    f"{q_label} are identical ({solutions_list[q][0]:.6g}).\n\n"
                    f"Grid convergence analysis requires different solution "
                    f"values on different grids. Check that:\n"
                    f"  - You extracted the correct quantity from each grid\n"
                    f"  - The solver converged to different answers on each grid\n"
                    f"  - You are not reading the same file multiple times"
                )
                return

        n_grids = len(cell_counts)
        dim = self._cmb_dim.currentData()
        theoretical_p = self._spn_theoretical_p.value()
        fs_val = self._spn_safety.value()
        fs = fs_val if fs_val > 0 else None  # 0 = Auto
        ref_scale_val = self._spn_ref_scale.value()
        ref_scale = ref_scale_val if ref_scale_val > 0 else None  # 0 = Auto

        target_idx = self._cmb_production.currentData()
        if target_idx is None:
            target_idx = 0

        direction = self._cmb_direction.currentData()
        if direction is None:
            direction = "both"

        self._btn_compute.setEnabled(False)
        try:
            self._results = []
            self._fs_results = []
            self._lsr_results = []
            for q in range(len(solutions_list)):
                res = compute_gci(
                    solutions=solutions_list[q],
                    cell_counts=cell_counts,
                    dim=dim,
                    safety_factor=fs,
                    theoretical_order=theoretical_p,
                    reference_scale=ref_scale,
                    direction_of_concern=direction,
                    target_grid_idx=target_idx,
                )
                self._results.append(res)

                # Auto-compute FS method (3+ grids)
                if n_grids >= 3:
                    fs_res = compute_fs_uncertainty(
                        solutions=solutions_list[q],
                        cell_counts=cell_counts,
                        dim=dim,
                        theoretical_order=theoretical_p,
                        reference_scale=ref_scale,
                    )
                    self._fs_results.append(fs_res)
                else:
                    self._fs_results.append(None)

                # Auto-compute LSR method (4+ grids)
                if n_grids >= 4:
                    lsr_res = compute_lsr_uncertainty(
                        solutions=solutions_list[q],
                        cell_counts=cell_counts,
                        dim=dim,
                        theoretical_order=theoretical_p,
                        reference_scale=ref_scale,
                    )
                    self._lsr_results.append(lsr_res)
                else:
                    self._lsr_results.append(None)

            self._display_results()
            self._update_guidance()
            self._update_plot()
            self._generate_report_statements()
            self._update_carry_over_summary()
            self._btn_copy_carry.setEnabled(bool(self._results))
            self._btn_summary_card.setEnabled(bool(self._results))
            self._results_stale = False
            self._btn_compute.setText("Compute GCI")
            self._btn_compute.setToolTip("")
        except Exception as exc:
            QMessageBox.critical(
                self, "Computation Error",
                f"GCI computation failed:\n\n{exc}"
            )
        finally:
            self._btn_compute.setEnabled(True)

    # ------------------------------------------------------------------
    # DISPLAY
    # ------------------------------------------------------------------

    def _display_results(self):
        """Populate results text and table."""
        if not self._results:
            return

        lines = []
        lines.append("=" * 65)
        lines.append("  GRID CONVERGENCE INDEX (GCI) RESULTS")
        lines.append("=" * 65)
        lines.append("")

        for q_idx, res in enumerate(self._results):
            q_label = self._qty_label(q_idx)
            q_unit = self._qty_unit(q_idx)
            unit_sfx = f" {q_unit}" if q_unit else ""

            if len(self._results) > 1:
                lines.append(f"--- {q_label} ---")
                lines.append("")

            lines.append(f"  Method:              {res.method}")
            lines.append(f"  Number of grids:     {res.n_grids}")
            lines.append(f"  Dimensions:          {res.dim}D")
            lines.append(f"  Theoretical order:   {res.theoretical_order:.1f}")
            lines.append(f"  Safety factor Fs:    {res.safety_factor:.2f}")
            lines.append("")

            # Grid details
            lines.append("  Grid Details:")
            for i in range(res.n_grids):
                sol = res.grid_solutions[i] if i < len(res.grid_solutions) else 0
                cells = res.grid_cells[i] if i < len(res.grid_cells) else 0
                h = res.grid_spacings[i] if i < len(res.grid_spacings) else 0
                tag = " (finest)" if i == 0 else (" (coarsest)" if i == res.n_grids - 1 else "")
                lines.append(
                    f"    Grid {i+1}{tag}: "
                    f"N = {cells:,.0f}, h = {h:.6f}, f = {sol:.6g}{unit_sfx}"
                )
            lines.append("")

            # Refinement ratios
            if res.refinement_ratios:
                lines.append("  Refinement Ratios:")
                for i, r in enumerate(res.refinement_ratios):
                    lines.append(
                        f"    r_{i+1}{i+2} = {r:.4f}"
                    )
                lines.append("")

            # Errors
            if not np.isnan(res.e21_abs):
                lines.append("  Discretization Errors:")
                lines.append(f"    |e21| = |f2 - f1|     = {res.e21_abs:.6g}{unit_sfx}")
                if not np.isnan(res.e21_rel):
                    lines.append(f"    |e21|/|f1| (relative) = {res.e21_rel:.6g} "
                                f"({res.e21_rel*100:.4f}%)")
                if not np.isnan(res.e32_abs):
                    lines.append(f"    |e32| = |f3 - f2|     = {res.e32_abs:.6g}{unit_sfx}")
                if not np.isnan(res.e32_rel):
                    lines.append(f"    |e32|/|f2| (relative) = {res.e32_rel:.6g} "
                                f"({res.e32_rel*100:.4f}%)")
                lines.append("")

            # Convergence
            lines.append(f"  Convergence type:    {res.convergence_type}")
            if not np.isnan(res.convergence_ratio):
                lines.append(f"  Convergence ratio R: {res.convergence_ratio:.6f}")
            lines.append("")

            # Observed order
            if not np.isnan(res.observed_order) and res.observed_order < 1e6:
                order_label = ("Assumed order p:" if res.order_is_assumed
                               else "Observed order p:")
                lines.append(
                    f"  {order_label:21s}{res.observed_order:.4f}"
                )
                # Order diagnostic
                if (not res.order_is_assumed
                        and res.theoretical_order > 0
                        and res.observed_order > 0):
                    p_obs = res.observed_order
                    p_th = res.theoretical_order
                    lo = 0.5 * p_th
                    hi = 2.0 * p_th
                    if lo <= p_obs <= hi:
                        lines.append(
                            f"  Order diagnostic:    \u2713 Within "
                            f"acceptable range [{lo:.1f}, {hi:.1f}]"
                        )
                    else:
                        lines.append(
                            f"  Order diagnostic:    \u2717 Outside "
                            f"range [{lo:.1f}, {hi:.1f}] — "
                            f"use with caution"
                        )
            lines.append("")

            # Richardson extrapolation
            if not np.isnan(res.richardson_extrapolation):
                lines.append(
                    f"  Richardson extrap:   {res.richardson_extrapolation:.6g}{unit_sfx}"
                )
                lines.append(
                    f"  Fine grid solution:  {res.grid_solutions[0]:.6g}{unit_sfx}"
                )
                lines.append(
                    f"  RE correction:       "
                    f"{res.richardson_extrapolation - res.grid_solutions[0]:+.6g}{unit_sfx}"
                )
                lines.append("")

            # GCI
            if not np.isnan(res.gci_fine):
                lines.append("  GCI Results:")
                lines.append(
                    f"    GCI_fine   = {res.gci_fine:.6g}  "
                    f"({res.gci_fine*100:.4f}%)"
                )
                if not np.isnan(res.gci_coarse):
                    lines.append(
                        f"    GCI_coarse = {res.gci_coarse:.6g}  "
                        f"({res.gci_coarse*100:.4f}%)"
                    )
                if not np.isnan(res.asymptotic_ratio):
                    lines.append(
                        f"    Asymptotic ratio = {res.asymptotic_ratio:.4f}  "
                        f"(should be ~1.0)"
                    )
                lines.append("")

            # u_num — fine grid (standard GCI)
            if not np.isnan(res.u_num):
                lines.append("  Numerical Uncertainty \u2014 Fine Grid (standard GCI):")
                lines.append(
                    f"    u_num (1\u03c3) = {res.u_num:.6g}{unit_sfx}"
                )
                lines.append(
                    f"    u_num / |f1| = {res.u_num_pct:.4f}%"
                )
                lines.append(
                    f"    Expanded (k=2): \u00b1{2*res.u_num:.6g}{unit_sfx}  "
                    f"({2*res.u_num_pct:.4f}%)"
                )
                lines.append("")

            # Per-grid uncertainty table
            if res.per_grid_u_num and any(
                    not np.isnan(u) for u in res.per_grid_u_num):
                tgt = res.target_grid_idx
                lines.append("  Per-Grid Numerical Uncertainty:")
                lines.append(
                    f"    {'Grid':<12s} {'N cells':>12s} "
                    f"{'f_i':>12s} {'|f_i-f_RE|':>12s} "
                    f"{'u_num':>12s} {'u_num%':>8s}"
                )
                lines.append("    " + "-" * 72)
                for i in range(res.n_grids):
                    if i >= len(res.per_grid_u_num):
                        break
                    u_i = res.per_grid_u_num[i]
                    err_i = res.per_grid_error[i]
                    pct_i = res.per_grid_u_pct[i]
                    cells = res.grid_cells[i] if i < len(res.grid_cells) else 0
                    sol = res.grid_solutions[i] if i < len(res.grid_solutions) else 0
                    marker = " \u25c0 PRODUCTION" if i == tgt else ""
                    if not np.isnan(u_i):
                        lines.append(
                            f"    Grid {i+1:<7d} {cells:>12,.0f} "
                            f"{sol:>12.6g} {err_i:>12.6g} "
                            f"{u_i:>12.6g} {pct_i:>7.3f}%{marker}"
                        )
                    else:
                        lines.append(
                            f"    Grid {i+1:<7d} {cells:>12,.0f} "
                            f"{sol:>12.6g} {'N/A':>12s} "
                            f"{'N/A':>12s} {'N/A':>8s}{marker}"
                        )
                lines.append("")

                # Production grid summary
                if 0 <= tgt < len(res.per_grid_u_num):
                    u_prod = res.per_grid_u_num[tgt]
                    if not np.isnan(u_prod):
                        pct_prod = res.per_grid_u_pct[tgt]
                        f_prod = res.grid_solutions[tgt]
                        lines.append(
                            f"  \u2605 PRODUCTION GRID (Grid {tgt+1}) "
                            f"Numerical Uncertainty:"
                        )
                        lines.append(
                            f"    u_num (1\u03c3) = {u_prod:.6g}{unit_sfx}  "
                            f"({pct_prod:.4f}% of f = {f_prod:.6g}{unit_sfx})"
                        )
                        lines.append(
                            f"    Expanded (k=2): \u00b1{2*u_prod:.6g}{unit_sfx}  "
                            f"({2*pct_prod:.4f}%)"
                        )
                        if tgt > 0 and np.isfinite(res.u_num) and res.u_num > 0:
                            ratio = u_prod / res.u_num
                            lines.append(
                                f"    Ratio to fine-grid u_num: "
                                f"{ratio:.2f}x  ({ratio*100 - 100:+.1f}% larger)"
                            )
                        lines.append("")

            # Warnings
            if res.warnings:
                lines.append("  Warnings:")
                for w in res.warnings:
                    lines.append(f"    \u26A0 {w}")
                lines.append("")

            # Notes
            if res.notes:
                lines.append("  Notes:")
                for note_line in res.notes.split('\n'):
                    lines.append(f"    {note_line}")
                lines.append("")

            # Triplet comparison table (auto-selection traceability)
            if res.auto_selection_candidates:
                lines.append("  Triplet Evaluation Table (auto-selection):")
                lines.append(
                    f"    {'Grids':<12s} {'R':>8s} {'Type':<14s} "
                    f"{'min(r)':>8s} {'Score':>10s} {'Status':<10s}"
                )
                lines.append("    " + "\u2500" * 66)
                for cand in res.auto_selection_candidates:
                    idx = cand['indices']
                    grids_str = f"{idx[0]+1}-{idx[1]+1}-{idx[2]+1}"
                    R_val = cand['R']
                    R_str = f"{R_val:.4f}" if np.isfinite(R_val) else "N/A"
                    ctype = cand['convergence_type']
                    min_r = cand.get('min_r', 0.0)
                    score = cand.get('score')
                    score_str = f"{score:.2f}" if score is not None else "rejected"
                    if cand.get('selected'):
                        status = "\u2190 SELECTED"
                    elif score is not None:
                        status = "valid"
                    else:
                        status = "rejected"
                    lines.append(
                        f"    {grids_str:<12s} {R_str:>8s} "
                        f"{ctype:<14s} {min_r:>8.3f} "
                        f"{score_str:>10s} {status:<10s}"
                    )
                lines.append("")

            # ============================================================
            # CARRY-OVER BOX — the value to feed into the Uncertainty
            # Aggregator.  Made impossible to miss.
            # ============================================================
            tgt = res.target_grid_idx
            carry_u = None
            carry_pct = None
            carry_grid_label = ""

            if (res.is_valid and res.per_grid_u_num and
                    0 <= tgt < len(res.per_grid_u_num) and
                    not np.isnan(res.per_grid_u_num[tgt])):
                carry_u = res.per_grid_u_num[tgt]
                carry_pct = res.per_grid_u_pct[tgt]
                carry_grid_label = (
                    f"Grid {tgt+1}"
                    + (" (finest)" if tgt == 0 else " (production)")
                )
            elif res.is_valid and not np.isnan(res.u_num):
                carry_u = res.u_num
                carry_pct = res.u_num_pct
                carry_grid_label = "Grid 1 (finest)"

            if carry_u is not None:
                # Auto-selection note
                if res.auto_selected_triplet is not None:
                    lines.append(
                        f"  \u26A0  {res.auto_selection_reason}")
                    lines.append("")

                # Effective grid-independence note
                if res.effectively_grid_independent:
                    lines.append(
                        "  \u2713  Grid-independent: solutions differ by "
                        f"< {res.u_num*2:.4g} — u_num = {res.u_num:.4g}")
                    lines.append("")

                lines.append("  " + "\u2550" * 61)
                lines.append(
                    "  \u2551  \u279C  CARRY THIS VALUE TO THE UNCERTAINTY "
                    "AGGREGATOR:          \u2551"
                )
                lines.append("  " + "\u2550" * 61)
                lines.append(
                    f"  \u2551                                                "
                    f"             \u2551"
                )
                val_line = f"u_num (standard)     = {carry_u:.6g}{unit_sfx}"
                pct_line = f"({carry_pct:.4f}%)"
                combined = f"{val_line}   {pct_line}"
                lines.append(
                    f"  \u2551    {combined:<57s}\u2551"
                )

                # Directional u_num line
                dir_u = res.u_num_directional
                if not np.isnan(dir_u):
                    dir_label = res.direction_of_concern
                    if dir_label != "both":
                        if dir_u == 0.0:
                            dir_line = (
                                f"u_num (directional)  = 0.0000"
                                f"{unit_sfx}   \u2190 CONSERVATIVE BIAS"
                            )
                        else:
                            dir_pct = ((dir_u / abs(res.grid_solutions[tgt]))
                                       * 100.0
                                       if abs(res.grid_solutions[tgt]) > 1e-30
                                       else 0.0) if q_idx == 0 else carry_pct
                            dir_line = (
                                f"u_num (directional)  = {dir_u:.6g}"
                                f"{unit_sfx}   ({dir_pct:.4f}%, adverse)"
                            )
                        lines.append(
                            f"  \u2551    {dir_line:<57s}\u2551"
                        )
                        dir_info = f"Direction: {dir_label}"
                        lines.append(
                            f"  \u2551    {dir_info:<57s}\u2551"
                        )
                        if res.directional_note:
                            # Wrap note to fit box (multi-line)
                            import textwrap
                            wrapped = textwrap.wrap(
                                res.directional_note, width=55)
                            for wline in wrapped:
                                lines.append(
                                    f"  \u2551    {wline:<57s}\u2551"
                                )

                lines.append(
                    f"  \u2551                                                "
                    f"             \u2551"
                )
                grid_info = f"Source: {carry_grid_label}  |  Fs = {res.safety_factor:.2f}"
                lines.append(
                    f"  \u2551    {grid_info:<57s}\u2551"
                )
                aggr_info = (
                    "Enter as: Sigma Value = above  |  Basis = "
                    "Confirmed 1\u03c3  |  DOF = \u221e"
                )
                lines.append(
                    f"  \u2551    {aggr_info:<57s}\u2551"
                )
                lines.append("  " + "\u2550" * 61)
                lines.append("")

            lines.append("-" * 65)
            lines.append("")

        # ---- Celik Table 1 (standard reporting format) ----
        for q_idx, res in enumerate(self._results):
            q_label_celik = self._qty_label(q_idx)
            if not res.is_valid and np.isnan(res.gci_fine):
                continue

            lines.append(f"  Celik et al. (2008) Table 1 \u2014 {q_label_celik}")
            lines.append("  " + "-" * 50)
            # Grid solutions row
            hdr = "  {:30s}".format("")
            for i in range(min(res.n_grids, 3)):
                hdr += f"  {'f'+str(i+1):>10s}"
            lines.append(hdr)
            lines.append("  " + "-" * 50)

            row_n = "  {:30s}".format("N (cells)")
            for i in range(min(res.n_grids, 3)):
                row_n += f"  {res.grid_cells[i]:>10,.0f}"
            lines.append(row_n)

            if res.refinement_ratios:
                row_r = "  {:30s}".format("r_21")
                row_r += f"  {res.refinement_ratios[0]:>10.4f}"
                lines.append(row_r)
                if len(res.refinement_ratios) > 1:
                    row_r32 = "  {:30s}".format("r_32")
                    row_r32 += f"  {'':>10s}  {res.refinement_ratios[1]:>10.4f}"
                    lines.append(row_r32)

            row_f = "  {:30s}".format("phi (solution)")
            for i in range(min(res.n_grids, 3)):
                row_f += f"  {res.grid_solutions[i]:>10.6g}"
            lines.append(row_f)

            if not np.isnan(res.observed_order) and res.observed_order < 1e6:
                p_label = ("p (assumed order)" if res.order_is_assumed
                           else "p (observed order)")
                row_p = "  {:30s}  {:>10.4f}".format(
                    p_label, res.observed_order)
                lines.append(row_p)

            if not np.isnan(res.richardson_extrapolation):
                row_re = "  {:30s}  {:>10.6g}".format(
                    "phi_ext (RE)", res.richardson_extrapolation)
                lines.append(row_re)

            if not np.isnan(res.e21_rel):
                row_ea = "  {:30s}  {:>10.4f}%".format(
                    "e_a^21 (approx. rel. error)",
                    res.e21_rel * 100)
                lines.append(row_ea)

            if not np.isnan(res.richardson_extrapolation):
                e_ext = abs(
                    (res.richardson_extrapolation - res.grid_solutions[0])
                    / res.richardson_extrapolation
                ) if abs(res.richardson_extrapolation) > 1e-30 else 0.0
                row_eext = "  {:30s}  {:>10.4f}%".format(
                    "e_ext^21 (extrapolated rel.)",
                    e_ext * 100)
                lines.append(row_eext)

            if not np.isnan(res.gci_fine):
                row_gci = "  {:30s}  {:>10.4f}%".format(
                    "GCI_fine^21", res.gci_fine * 100)
                lines.append(row_gci)

            lines.append("  " + "-" * 50)
            lines.append("")

        # ---- Reviewer Checklist ----
        if self._results:
            res0 = self._results[0]
            lines.append("  Grid Convergence Review Checklist:")
            lines.append("  " + "-" * 50)

            # Number of grids
            if res0.n_grids >= 3:
                lines.append(
                    f"  [\u2714 PASS] Grids: {res0.n_grids} "
                    f"(>= 3 for observed order)")
            elif res0.n_grids == 2:
                lines.append(
                    f"  [\u26A0 NOTE] Grids: 2 "
                    f"(assumed order — consider 3+ grids)")
            else:
                lines.append(
                    f"  [\u2716 FAIL] Grids: {res0.n_grids} (insufficient)")

            # Refinement ratio
            if res0.refinement_ratios:
                r_min = min(res0.refinement_ratios)
                if r_min >= 1.3:
                    lines.append(
                        f"  [\u2714 PASS] Refinement ratio: "
                        f"r_min = {r_min:.3f} (>= 1.3)")
                elif r_min >= 1.1:
                    lines.append(
                        f"  [\u26A0 NOTE] Refinement ratio: "
                        f"r_min = {r_min:.3f} (< 1.3 — marginal)")
                else:
                    lines.append(
                        f"  [\u2716 FAIL] Refinement ratio: "
                        f"r_min = {r_min:.3f} (< 1.1 — too low)")

            # Convergence type
            if res0.convergence_type == "monotonic":
                lines.append(
                    f"  [\u2714 PASS] Convergence: monotonic")
            elif res0.convergence_type == "oscillatory":
                lines.append(
                    f"  [\u26A0 NOTE] Convergence: oscillatory "
                    f"(Fs = 3.0 applied)")
            elif res0.convergence_type == "divergent":
                lines.append(
                    f"  [\u2716 FAIL] Convergence: DIVERGENT — "
                    f"GCI invalid")
            else:
                lines.append(
                    f"  [\u2714 PASS] Convergence: {res0.convergence_type}")

            # Observed / assumed order
            p = res0.observed_order
            p_th = res0.theoretical_order
            if res0.order_is_assumed:
                lines.append(
                    f"  [\u26A0 NOTE] Assumed order: "
                    f"p = {p_th:.1f} (theoretical, not computed — "
                    f"2-grid study)")
            elif not np.isnan(p) and p < 1e6:
                ratio_p = abs(p - p_th) / max(p_th, 0.1)
                if ratio_p < 0.3:
                    lines.append(
                        f"  [\u2714 PASS] Observed order: "
                        f"p = {p:.3f} (theoretical = {p_th:.1f}, "
                        f"ratio = {p/p_th:.2f})")
                elif ratio_p < 0.5:
                    lines.append(
                        f"  [\u26A0 NOTE] Observed order: "
                        f"p = {p:.3f} (theoretical = {p_th:.1f}, "
                        f"ratio = {p/p_th:.2f} — moderate deviation)")
                else:
                    lines.append(
                        f"  [\u2716 WARN] Observed order: "
                        f"p = {p:.3f} (theoretical = {p_th:.1f}, "
                        f"ratio = {p/p_th:.2f} — significant deviation)")
            elif res0.method.startswith("2-grid"):
                lines.append(
                    f"  [\u26A0 NOTE] Assumed order: "
                    f"p = {p_th:.1f} (2-grid, not computed)")

            # Asymptotic range
            ar = res0.asymptotic_ratio
            if not np.isnan(ar):
                if 0.95 <= ar <= 1.05:
                    lines.append(
                        f"  [\u2714 PASS] Asymptotic ratio: "
                        f"{ar:.4f} (within [0.95, 1.05])")
                elif 0.8 <= ar <= 1.2:
                    lines.append(
                        f"  [\u26A0 NOTE] Asymptotic ratio: "
                        f"{ar:.4f} (within [0.8, 1.2] — acceptable)")
                else:
                    lines.append(
                        f"  [\u2716 FAIL] Asymptotic ratio: "
                        f"{ar:.4f} (outside [0.8, 1.2])")
            elif res0.convergence_type not in ("oscillatory", "divergent"):
                lines.append(
                    f"  [\u26A0 NOTE] Asymptotic ratio: not computed")

            # GCI magnitude
            if not np.isnan(res0.gci_fine):
                gci_pct = res0.gci_fine * 100
                if gci_pct < 2.0:
                    lines.append(
                        f"  [\u2714 PASS] GCI_fine: "
                        f"{gci_pct:.3f}% (< 2% — excellent)")
                elif gci_pct < 5.0:
                    lines.append(
                        f"  [\u2714 PASS] GCI_fine: "
                        f"{gci_pct:.3f}% (< 5% — acceptable)")
                else:
                    lines.append(
                        f"  [\u26A0 NOTE] GCI_fine: "
                        f"{gci_pct:.3f}% (>= 5% — consider finer grids)")

            lines.append(
                "  [INFO] Iterative convergence: verify "
                "independently (residuals << discretization error)")
            lines.append(
                "  [INFO] Solver settings: confirm identical "
                "across all grid levels")
            lines.append("  " + "-" * 50)
            lines.append("")

        # ---- Alternative Method Comparison ----
        has_alt = (hasattr(self, '_fs_results') and
                   any(r is not None for r in self._fs_results))
        has_lsr = (hasattr(self, '_lsr_results') and
                   any(r is not None for r in self._lsr_results))

        if has_alt or has_lsr:
            lines.append("=" * 65)
            lines.append("  METHOD COMPARISON")
            lines.append("=" * 65)
            lines.append("")

            # Build comparison table — always use FINE-GRID u_num for
            # fair comparison (FS and LSR always report fine-grid values).
            hdr = f"  {'Quantity':<20s} {'GCI':>12s}"
            if has_alt:
                hdr += f" {'FS':>12s}"
            if has_lsr:
                hdr += f" {'LSR':>12s}"
            lines.append("  u_num values — fine grid, 1-sigma:")
            lines.append(hdr)
            lines.append("  " + "-" * (22 + 14 * (1 + int(has_alt)
                         + int(has_lsr))))

            for q_idx, res in enumerate(self._results):
                q_label = self._qty_label(q_idx)
                # GCI u_num — always fine grid (index 0) for comparison
                u_gci = res.u_num  # fine-grid u_num

                row = f"  {q_label:<20s} {u_gci:>12.4g}"

                if has_alt:
                    fs_r = (self._fs_results[q_idx]
                            if q_idx < len(self._fs_results) else None)
                    if fs_r and fs_r.get("is_valid"):
                        row += f" {fs_r['u_num']:>12.4g}"
                    else:
                        row += f" {'N/A':>12s}"

                if has_lsr:
                    lsr_r = (self._lsr_results[q_idx]
                             if q_idx < len(self._lsr_results) else None)
                    if lsr_r and lsr_r.get("is_valid"):
                        row += f" {lsr_r['u_num']:>12.4g}"
                    else:
                        row += f" {'N/A':>12s}"

                lines.append(row)

            lines.append("")

            # Per-quantity detail for FS
            for q_idx in range(len(self._results)):
                q_label = self._qty_label(q_idx)
                q_unit = self._qty_unit(q_idx)
                unit_sfx = f" {q_unit}" if q_unit else ""

                if has_alt and q_idx < len(self._fs_results):
                    fs_r = self._fs_results[q_idx]
                    if fs_r and fs_r.get("is_valid"):
                        lines.append(
                            f"  FS variant (Xing & Stern 2010) \u2014 {q_label}:")
                        lines.append(
                            f"    u_num = {fs_r['u_num']:.4g}{unit_sfx} "
                            f"({fs_r['u_num_pct']:.2f}%)")
                        lines.append(f"    {fs_r.get('note', '')}")
                        lines.append("")

                if has_lsr and q_idx < len(self._lsr_results):
                    lsr_r = self._lsr_results[q_idx]
                    if lsr_r and lsr_r.get("is_valid"):
                        lines.append(
                            f"  LSR variant with AICc (Eca & Hoekstra 2014) \u2014 {q_label}:")
                        lines.append(
                            f"    u_num = {lsr_r['u_num']:.4g}{unit_sfx} "
                            f"({lsr_r['u_num_pct']:.2f}%)")
                        lines.append(f"    {lsr_r.get('note', '')}")
                        lines.append(
                            f"    phi_0 (extrapolated) = "
                            f"{lsr_r.get('phi_0', float('nan')):.6g}"
                            f"{unit_sfx}")
                        lines.append("")

            lines.append("-" * 65)
            lines.append("")

        # References
        lines.append("References:")
        lines.append("  Celik et al. (2008) J. Fluids Eng. 130(7), 078001")
        lines.append("  Roache (1998) Hermosa Publishers")
        lines.append("  ASME V&V 20-2009 (R2021) Section 5.1")
        if has_alt:
            lines.append(
                "  Xing, T. and Stern, F. (2010) ASME J. Fluids Eng. "
                "132(6), 061403")
        if has_lsr:
            lines.append(
                "  Eca, L. and Hoekstra, M. (2014) J. Comp. Physics "
                "262, 104-130")

        self._results_text.setPlainText("\n".join(lines))

        # ---- Results summary table ----
        n_qty = len(self._results)

        # Determine if any result uses a non-finest production grid
        has_prod = any(r.target_grid_idx > 0 for r in self._results)

        if has_prod:
            cols = [
                "Quantity", "Grids", "Type", "p (obs)", "f\u2081 (fine)",
                "f_RE", "GCI_fine", "GCI%", "u_num (fine)", "Prod Grid",
                "f_prod", "u_num (prod)", "Fs", "Asymptotic"
            ]
        else:
            cols = [
                "Quantity", "Grids", "Type", "p (obs)", "f\u2081 (fine)",
                "f_RE", "GCI_fine", "GCI%", "u_num", "Fs",
                "Asymptotic"
            ]
        self._results_table.setColumnCount(len(cols))
        self._results_table.setHorizontalHeaderLabels(cols)
        self._results_table.setRowCount(n_qty)

        for i, res in enumerate(self._results):
            q_name = (self._quantity_names[i]
                      if i < len(self._quantity_names)
                      else f"Q{i+1}")

            tgt = res.target_grid_idx
            # Production grid u_num
            u_prod_val = float('nan')
            f_prod_val = float('nan')
            if (res.per_grid_u_num and 0 <= tgt < len(res.per_grid_u_num)):
                u_prod_val = res.per_grid_u_num[tgt]
                f_prod_val = (res.grid_solutions[tgt]
                              if tgt < len(res.grid_solutions)
                              else float('nan'))

            if has_prod:
                row_data = [
                    q_name,
                    str(res.n_grids),
                    res.convergence_type,
                    f"{res.observed_order:.3f}" if (
                        not np.isnan(res.observed_order) and
                        res.observed_order < 1e6) else "N/A",
                    f"{res.grid_solutions[0]:.6g}" if res.grid_solutions else "\u2014",
                    f"{res.richardson_extrapolation:.6g}" if not np.isnan(
                        res.richardson_extrapolation) else "N/A",
                    f"{res.gci_fine:.6g}" if not np.isnan(res.gci_fine) else "N/A",
                    f"{res.gci_fine*100:.3f}%" if not np.isnan(res.gci_fine) else "N/A",
                    f"{res.u_num:.6g}" if not np.isnan(res.u_num) else "\u2014",
                    f"Grid {tgt+1}",
                    f"{f_prod_val:.6g}" if not np.isnan(f_prod_val) else "\u2014",
                    f"{u_prod_val:.6g}" if not np.isnan(u_prod_val) else "\u2014",
                    f"{res.safety_factor:.2f}",
                    f"{res.asymptotic_ratio:.3f}" if not np.isnan(
                        res.asymptotic_ratio) else "N/A",
                ]
            else:
                row_data = [
                    q_name,
                    str(res.n_grids),
                    res.convergence_type,
                    f"{res.observed_order:.3f}" if (
                        not np.isnan(res.observed_order) and
                        res.observed_order < 1e6) else "N/A",
                    f"{res.grid_solutions[0]:.6g}" if res.grid_solutions else "\u2014",
                    f"{res.richardson_extrapolation:.6g}" if not np.isnan(
                        res.richardson_extrapolation) else "N/A",
                    f"{res.gci_fine:.6g}" if not np.isnan(res.gci_fine) else "N/A",
                    f"{res.gci_fine*100:.3f}%" if not np.isnan(res.gci_fine) else "N/A",
                    f"{res.u_num:.6g}" if not np.isnan(res.u_num) else "\u2014",
                    f"{res.safety_factor:.2f}",
                    f"{res.asymptotic_ratio:.3f}" if not np.isnan(
                        res.asymptotic_ratio) else "N/A",
                ]

            for j, val in enumerate(row_data):
                item = QTableWidgetItem(val)
                item.setTextAlignment(Qt.AlignCenter)
                # Color-code convergence type
                if j == 2:
                    if val == "monotonic":
                        item.setForeground(QColor(DARK_COLORS['green']))
                    elif val == "oscillatory":
                        item.setForeground(QColor(DARK_COLORS['yellow']))
                    elif val == "divergent":
                        item.setForeground(QColor(DARK_COLORS['red']))
                # Color-code asymptotic ratio (last column in both layouts)
                ar_col = len(cols) - 1
                if j == ar_col and not np.isnan(res.asymptotic_ratio):
                    if 0.95 <= res.asymptotic_ratio <= 1.05:
                        item.setForeground(QColor(DARK_COLORS['green']))
                    elif 0.8 <= res.asymptotic_ratio <= 1.2:
                        item.setForeground(QColor(DARK_COLORS['yellow']))
                    else:
                        item.setForeground(QColor(DARK_COLORS['red']))
                # Highlight production grid column in accent color
                if has_prod and j in (9, 10, 11):
                    item.setForeground(QColor(DARK_COLORS['accent']))
                self._results_table.setItem(i, j, item)

        style_table(self._results_table)

    def _update_guidance(self):
        """Update the three guidance panels."""
        if not self._results:
            return

        # Hide bounding panel by default — only shown for divergent + small spread
        self._guidance_bounding.setVisible(False)
        self._bounding_config_frame.setVisible(False)

        # Use the first quantity for guidance (representative)
        res = self._results[0]

        # 1. Convergence assessment
        if res.convergence_type == "monotonic":
            self._guidance_convergence.set_guidance(
                f"Monotonic convergence detected (R = "
                f"{res.convergence_ratio:.4f}). The solution is converging "
                f"toward a single value with grid refinement. This is the "
                f"ideal case for Richardson extrapolation and GCI.\n\n"
                f"DO THIS NOW: Use the carry-over box u_num value for the Aggregator.",
                'green'
            )
        elif res.convergence_type == "oscillatory":
            self._guidance_convergence.set_guidance(
                f"Oscillatory convergence detected (R = "
                f"{res.convergence_ratio:.4f}). The solution oscillates "
                f"between grid levels. Richardson extrapolation is unreliable. "
                f"GCI is computed from the oscillation range with a "
                f"conservative safety factor. Consider: (a) checking solver "
                f"residual convergence on each grid level, "
                f"(b) using more grid levels to better resolve the trend.\n\n"
                f"DO THIS NOW: You may carry the shown u_num as a conservative "
                f"estimate only after reviewing warnings.",
                'yellow'
            )
        elif res.convergence_type == "divergent":
            # Compute spread metrics for diagnostic severity
            solutions = res.grid_solutions
            spread = max(solutions) - min(solutions)
            mean_val = np.mean(solutions)
            spread_pct = ((spread / abs(mean_val) * 100.0)
                          if abs(mean_val) > 1e-30 else float('inf'))

            if spread_pct < 2.0:
                severity = "SMALL"
                spread_advice = (
                    f"The grid solutions differ by {spread_pct:.1f}% "
                    f"(spread = {spread:.4g}). This is marginal "
                    f"divergence.\n"
                    f"  -> A conservative bounding estimate "
                    f"(u_num = spread/2 = {spread/2:.4g}) MAY be "
                    f"acceptable with engineering justification.\n"
                    f"  -> Adding a finer grid may resolve the "
                    f"convergence direction."
                )
            elif spread_pct < 10.0:
                severity = "MODERATE"
                spread_advice = (
                    f"The grid solutions differ by {spread_pct:.1f}% "
                    f"(spread = {spread:.4g}). This suggests a "
                    f"genuine convergence problem.\n"
                    f"  -> Do NOT use bounding without investigating "
                    f"root causes below."
                )
            else:
                severity = "LARGE"
                spread_advice = (
                    f"The grid solutions differ by {spread_pct:.1f}% "
                    f"(spread = {spread:.4g}). This indicates a "
                    f"fundamental problem.\n"
                    f"  -> DO NOT proceed until resolved."
                )

            self._guidance_convergence.set_guidance(
                f"DIVERGENT — the solution gets worse with grid "
                f"refinement (R = {res.convergence_ratio:.4f}). "
                f"GCI is NOT valid.\n\n"
                f"DIAGNOSTIC: Solution spread is {severity} "
                f"({spread_pct:.1f}%)\n"
                f"{spread_advice}\n\n"
                f"COMMON ROOT CAUSES (check in order):\n"
                f"1. Insufficient iterative convergence — residuals "
                f"must drop 3+ orders on EVERY grid. Check that "
                f"monitor points have flatlined, not just residuals.\n"
                f"2. y+ mismatch — if using wall functions, verify "
                f"y+ is in the valid range (30-300 for standard, "
                f"<1 for enhanced wall treatment) on ALL grids.\n"
                f"3. First-order boundary gradients — some solvers "
                f"default to first-order at boundaries. Check Fluent: "
                f"'Alternative Formulation'; CFX: 'High Resolution' "
                f"at walls.\n"
                f"4. Mesh quality degradation — check that coarser "
                f"meshes don't have degraded quality (skewness < 0.9, "
                f"aspect ratio < 100).\n"
                f"5. Extraction point location — move away from "
                f"singularities, trailing edges, or re-entrant "
                f"corners.\n\n"
                f"IF YOU MUST PROCEED (document ALL of the following):\n"
                f"  - State that formal grid convergence was not "
                f"achieved.\n"
                f"  - Report the grid solutions and spread.\n"
                f"  - If spread is small, assign u_num = spread/2 as "
                f"a bounding estimate with Uniform distribution.\n"
                f"  - Flag as epistemic uncertainty requiring "
                f"additional investigation.",
                'red'
            )
            # --- Conservative bounding estimate for small-spread divergence ---
            self._update_bounding_estimate(res)
        elif res.convergence_type == "grid-independent":
            self._guidance_convergence.set_guidance(
                "Grid-independent solution — the finest grids give identical "
                "results. Numerical uncertainty is effectively zero.\n\n"
                "DO THIS NOW: Carry u_num = 0 only if this is confirmed for "
                "the production grid quantity.",
                'green'
            )
        else:
            self._guidance_convergence.set_guidance(
                "Convergence type could not be determined. Check input data.",
                'yellow'
            )

        # 2. Observed order
        p = res.observed_order
        p_th = res.theoretical_order
        if np.isnan(p) or p > 1e6:
            self._guidance_order.set_guidance(
                "Observed order of accuracy could not be computed. "
                "Using theoretical order as fallback.",
                'yellow'
            )
        elif abs(p - p_th) / max(p_th, 0.1) < 0.3:
            self._guidance_order.set_guidance(
                f"Observed order p = {p:.3f} is consistent with the "
                f"theoretical order ({p_th:.1f}). The grids are in the "
                f"asymptotic range (the grids are fine enough that error "
                f"decreases at the expected theoretical rate). This is ideal.",
                'green'
            )
        elif p > 2 * p_th:
            self._guidance_order.set_guidance(
                f"Observed order p = {p:.3f} is much higher than the "
                f"theoretical order ({p_th:.1f}). This may indicate error "
                f"cancellation, superconvergence, or that the grids are not "
                f"in the asymptotic range. Consider using finer grids or "
                f"verifying solver convergence at each grid level.",
                'yellow'
            )
        elif p < 0.5 * p_th:
            self._guidance_order.set_guidance(
                f"Observed order p = {p:.3f} is significantly lower than "
                f"the theoretical order ({p_th:.1f}). This suggests the "
                f"grids are too coarse — not in the asymptotic range "
                f"(the grids are not fine enough for error to decrease "
                f"at the expected theoretical rate). Other causes include "
                f"boundary layer issues, singularities, or solver settings "
                f"degrading convergence.",
                'red'
            )
        else:
            self._guidance_order.set_guidance(
                f"Observed order p = {p:.3f} differs moderately from the "
                f"theoretical order ({p_th:.1f}). This is acceptable but "
                f"suggests the grids may not be fully in the asymptotic "
                f"range. The GCI remains valid.",
                'yellow'
            )

        # 3. Asymptotic range
        ar = res.asymptotic_ratio
        if np.isnan(ar):
            if res.convergence_type in ("oscillatory", "divergent"):
                self._guidance_asymptotic.set_guidance(
                    "Asymptotic range check is not applicable for "
                    f"{res.convergence_type} convergence.",
                    'yellow'
                )
            else:
                self._guidance_asymptotic.set_guidance(
                    "Asymptotic range check could not be computed.",
                    'yellow'
                )
        elif 0.95 <= ar <= 1.05:
            self._guidance_asymptotic.set_guidance(
                f"Asymptotic ratio = {ar:.4f} (target: 1.0). Excellent — "
                f"the grids are well within the asymptotic range. "
                f"GCI_fine is a reliable estimate of discretization "
                f"uncertainty.",
                'green'
            )
        elif 0.8 <= ar <= 1.2:
            self._guidance_asymptotic.set_guidance(
                f"Asymptotic ratio = {ar:.4f} (target: 1.0). Acceptable — "
                f"the grids are approximately in the asymptotic range. "
                f"GCI_fine is a reasonable estimate.",
                'yellow'
            )
        else:
            self._guidance_asymptotic.set_guidance(
                f"Asymptotic ratio = {ar:.4f} (target: 1.0). The grids "
                f"are NOT in the asymptotic range. GCI may be unreliable. "
                f"Consider: (a) adding a finer grid, (b) increasing "
                f"refinement ratios, (c) improving solver convergence.",
                'red'
            )

    # ------------------------------------------------------------------
    # CONSERVATIVE BOUNDING (divergent with small spread)
    # ------------------------------------------------------------------

    _TEMPERATURE_UNITS = {
        "°F", "°C", "K", "F", "C", "degF", "degC", "degR", "R",
        "°R", "Kelvin", "Celsius", "Fahrenheit", "Rankine",
    }

    def _is_temperature_unit(self, unit: str) -> bool:
        """Return True if *unit* looks like a temperature unit."""
        return unit.strip() in self._TEMPERATURE_UNITS

    def _update_bounding_estimate(self, res: 'GCIResult'):
        """Show conservative bounding panel when divergent spread is small.

        Two threshold modes:
        - **Percentage** (default for pressure, velocity, force, etc.):
          threshold = user-chosen % of |mean|.
        - **Absolute** (default for temperature): threshold = user-chosen
          absolute value (default 1 °F).

        The mode is auto-detected from the quantity unit unless the user
        overrides it.
        """
        solutions = res.grid_solutions
        if len(solutions) < 2:
            self._guidance_bounding.setVisible(False)
            self._bounding_config_frame.setVisible(False)
            return

        sol_arr = np.array(solutions, dtype=float)
        spread = float(np.max(sol_arr) - np.min(sol_arr))
        mean_val = float(np.mean(sol_arr))

        # Determine unit for the first quantity
        unit = ""
        if hasattr(self, '_quantity_units') and self._quantity_units:
            unit = self._quantity_units[0]
        unit_sfx = f" {unit}" if unit else ""

        # Determine threshold mode
        mode_selection = self._cmb_bnd_mode.currentData()
        if mode_selection == "auto":
            use_absolute = self._is_temperature_unit(unit)
        elif mode_selection == "absolute":
            use_absolute = True
        else:
            use_absolute = False

        # Evaluate threshold
        if use_absolute:
            threshold = self._spn_bnd_abs.value()
            below_threshold = spread < threshold
            spread_desc = f"{spread:.4g}{unit_sfx} (absolute spread)"
            thresh_desc = f"{threshold:.4g}{unit_sfx}"
        else:
            if abs(mean_val) < 1e-30:
                # Mean is effectively zero — percentage is meaningless
                self._guidance_bounding.setVisible(False)
                self._bounding_config_frame.setVisible(True)
                return
            spread_pct = spread / abs(mean_val) * 100.0
            threshold = self._spn_bnd_pct.value()
            below_threshold = spread_pct < threshold
            spread_desc = f"{spread_pct:.2f}% of |mean| (spread = {spread:.4g}{unit_sfx})"
            thresh_desc = f"{threshold:.1f}%"

        # Always show the config frame so the user can adjust
        self._bounding_config_frame.setVisible(True)

        if below_threshold:
            u_bound = spread / 2.0
            self._guidance_bounding.set_guidance(
                f"OPTIONAL CONSERVATIVE BOUND (use engineering judgment):\n"
                f"The grid solutions differ by only {spread_desc} "
                f"(threshold: {thresh_desc}).\n"
                f"A conservative bounding uncertainty can be assigned as:\n"
                f"  u_num_bound = spread / 2 = {u_bound:.4g}{unit_sfx}\n"
                f"This treats the full range as a ±bound with Uniform "
                f"distribution.\n\n"
                f"Enter in the VVUQ Uncertainty Aggregator as:\n"
                f"  Sigma = {u_bound:.4g} | Basis = Bounding (min/max) | "
                f"Dist = Uniform\n\n"
                f"⚠ This is NOT a GCI result. Document as 'bounding "
                f"estimate from grid spread' and note that formal grid "
                f"convergence was not achieved.\n\n"
                f"DO THIS NOW: Use this only if your program allows "
                f"engineering bounding when formal GCI fails.",
                'green'
            )
            self._guidance_bounding.setVisible(True)
        else:
            self._guidance_bounding.set_guidance(
                f"Spread is {spread_desc} — exceeds the threshold "
                f"({thresh_desc}). Conservative bounding is NOT recommended "
                f"at this spread level. Follow the recommended next steps "
                f"above to achieve grid convergence before assigning "
                f"a numerical uncertainty.",
                'yellow'
            )
            self._guidance_bounding.setVisible(True)

    def _update_plot(self):
        """Draw convergence plot (top) and log-log error plot (bottom)."""
        self._fig.clear()

        if not self._results:
            self._canvas.draw()
            return

        res = self._results[0]  # plot first quantity
        if res.n_grids < 2:
            self._canvas.draw()
            return

        spacings = np.array(res.grid_spacings)
        solutions = np.array(res.grid_solutions)
        tgt = res.target_grid_idx

        # Decide layout: 2 subplots if we have RE, otherwise 1
        has_re = not np.isnan(res.richardson_extrapolation)
        if has_re and res.n_grids >= 3:
            ax = self._fig.add_subplot(211)
            ax2 = self._fig.add_subplot(212)
        else:
            ax = self._fig.add_subplot(111)
            ax2 = None

        # ---- TOP PLOT: Solution vs h ----

        # Plot grid solutions
        ax.plot(spacings, solutions, 'o-', color=DARK_COLORS['accent'],
                markersize=8, linewidth=2, label='Grid solutions',
                zorder=5)

        # Highlight production grid with a larger diamond marker
        if 0 <= tgt < len(spacings):
            ax.plot(spacings[tgt], solutions[tgt], 'D',
                    color=DARK_COLORS['orange'], markersize=14,
                    markeredgecolor=DARK_COLORS['fg_bright'],
                    markeredgewidth=1.5, zorder=7,
                    label=f'Production grid (Grid {tgt+1})')

        # Richardson extrapolation (at h=0) and RE convergence curve
        if has_re:
            f_ext = res.richardson_extrapolation
            ax.axhline(y=f_ext,
                       color=DARK_COLORS['green'], linestyle='--',
                       linewidth=1.5, alpha=0.8,
                       label=f'Richardson extrap. = {f_ext:.6g}')
            ax.plot(0, f_ext, '*',
                    color=DARK_COLORS['green'], markersize=14, zorder=6)

            # RE convergence curve: f(h) = f_ext + C·h^p
            p_obs = res.observed_order
            if (not np.isnan(p_obs) and 0 < p_obs < 100
                    and spacings[0] > 1e-30
                    and abs(solutions[0] - f_ext) > 1e-30):
                C_coeff = (solutions[0] - f_ext) / _safe_rp(spacings[0], p_obs)
                if np.isfinite(C_coeff):
                    h_curve = np.linspace(0, spacings[-1] * 1.05, 200)
                    f_curve = f_ext + C_coeff * h_curve ** p_obs
                    ax.plot(h_curve, f_curve, '-',
                            color=DARK_COLORS['green'], linewidth=1.2,
                            alpha=0.4, zorder=2,
                            label=f'RE trend (p = {p_obs:.2f})')

        # u_num error band on fine grid (light blue) — k=2 expanded
        if not np.isnan(res.u_num) and len(solutions) > 0:
            f1 = solutions[0]
            band_fine = 2 * res.u_num  # expanded (k=2), same basis as production
            ax.fill_between(
                [0, spacings[0]],
                [f1 - band_fine, f1 - band_fine],
                [f1 + band_fine, f1 + band_fine],
                alpha=0.15, color=DARK_COLORS['accent'],
                label=f'Fine u_num \u00b12\u03c3 (\u00b1{band_fine:.4g})'
            )

        # GCI error band on PRODUCTION grid (orange) — if not finest
        if (tgt > 0 and 0 <= tgt < len(spacings) and
                res.per_grid_u_num and
                tgt < len(res.per_grid_u_num) and
                not np.isnan(res.per_grid_u_num[tgt])):
            f_prod = solutions[tgt]
            u_prod = res.per_grid_u_num[tgt]
            band_half = 2 * u_prod
            ax.fill_between(
                [spacings[tgt] * 0.85, spacings[tgt] * 1.15],
                [f_prod - band_half, f_prod - band_half],
                [f_prod + band_half, f_prod + band_half],
                alpha=0.25, color=DARK_COLORS['orange'],
                label=f'Prod. u_num \u00b12\u03c3 (\u00b1{band_half:.4g})',
                zorder=3
            )

        # Label each grid point
        for i, (h, f) in enumerate(zip(spacings, solutions)):
            if i == tgt:
                tag = f"Grid {i+1} \u2605"
                color = DARK_COLORS['orange']
            else:
                tag = f"Grid {i+1}"
                color = DARK_COLORS['fg_dim']
            ax.annotate(tag, (h, f), textcoords="offset points",
                       xytext=(8, 8), fontsize=7, color=color,
                       fontweight='bold' if i == tgt else 'normal')

        ax.set_xlabel("Representative spacing h (normalized)")
        unit_plot = self._qty_unit(0)
        ax.set_ylabel(f"Solution ({unit_plot})" if unit_plot else "Solution value")
        title = "Grid Convergence Study"
        if not np.isnan(res.observed_order) and res.observed_order < 1e6:
            title += f"  (p = {res.observed_order:.2f})"
        ax.set_title(title)
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=-0.001)

        # ---- BOTTOM PLOT: Log-log error vs h (slope = observed order) ----
        if ax2 is not None and has_re:
            f_re = res.richardson_extrapolation
            errors = np.array([abs(s - f_re) for s in solutions])

            # Filter out zero or near-zero errors for log plot
            valid = errors > 1e-30
            if np.sum(valid) >= 2:
                h_valid = spacings[valid]
                e_valid = errors[valid]

                ax2.loglog(h_valid, e_valid, 'o-',
                          color=DARK_COLORS['accent'],
                          markersize=8, linewidth=2,
                          label='|f_i - f_RE|', zorder=5)

                # Highlight production grid
                if 0 <= tgt < len(spacings) and valid[tgt]:
                    ax2.loglog(spacings[tgt], errors[tgt], 'D',
                              color=DARK_COLORS['orange'],
                              markersize=12,
                              markeredgecolor=DARK_COLORS['fg_bright'],
                              markeredgewidth=1.5, zorder=7)

                # Reference slope line at observed order
                p_obs = res.observed_order
                if not np.isnan(p_obs) and p_obs < 1e6 and p_obs > 0 and h_valid[0] > 1e-30:
                    h_ref = np.array([h_valid.min() * 0.7,
                                     h_valid.max() * 1.3])
                    # Anchor at the fine-grid point
                    e_ref = e_valid[0] * (h_ref / h_valid[0]) ** p_obs
                    ax2.loglog(h_ref, e_ref, '--',
                              color=DARK_COLORS['yellow'],
                              linewidth=1.5, alpha=0.8,
                              label=f'Slope = p = {p_obs:.2f}')

                # Label points
                for i in range(len(spacings)):
                    if valid[i]:
                        ax2.annotate(
                            f"G{i+1}", (spacings[i], errors[i]),
                            textcoords="offset points",
                            xytext=(6, 6), fontsize=7,
                            color=DARK_COLORS['fg_dim'])

                ax2.set_xlabel("log(h)")
                ax2.set_ylabel("log(|f - f_RE|)")
                ax2.set_title(
                    "Log-Log Convergence (slope = observed order p)")
                ax2.legend(loc='best', fontsize=7)
                ax2.grid(True, alpha=0.3, which='both')

        # ---- ORDER DIAGNOSTIC GAUGE (inset on top subplot) ----
        p_obs = res.observed_order
        p_th = res.theoretical_order
        if (has_re and not res.order_is_assumed
                and not np.isnan(p_obs) and 0 < p_obs < 100
                and p_th > 0):
            ax_ins = ax.inset_axes([0.62, 0.02, 0.36, 0.08])
            ax_ins.set_xlim(0, max(2.5 * p_th, p_obs * 1.3))
            ax_ins.set_ylim(0, 1)

            # Green zone: 0.5p_th to 2p_th
            ax_ins.axvspan(0.5 * p_th, 2.0 * p_th,
                           color=DARK_COLORS['green'], alpha=0.3)
            # Yellow zones
            ax_ins.axvspan(0, 0.5 * p_th,
                           color=DARK_COLORS['yellow'], alpha=0.15)
            ax_ins.axvspan(2.0 * p_th, ax_ins.get_xlim()[1],
                           color=DARK_COLORS['yellow'], alpha=0.15)

            # Marker for observed order
            in_range = 0.5 * p_th <= p_obs <= 2.0 * p_th
            marker_color = DARK_COLORS['green'] if in_range else DARK_COLORS['red']
            ax_ins.axvline(p_obs, color=marker_color, linewidth=2.5)
            ax_ins.axvline(p_th, color=DARK_COLORS['fg_dim'],
                           linewidth=1, linestyle=':')

            status = "\u2713" if in_range else "\u2717"
            ax_ins.set_title(
                f"p_obs = {p_obs:.2f}  (p_th = {p_th:.1f})  {status}",
                fontsize=6.5, color=marker_color, pad=2)
            ax_ins.set_yticks([])
            ax_ins.tick_params(axis='x', labelsize=5.5)
            ax_ins.set_facecolor(DARK_COLORS['bg_alt'])
            for spine in ax_ins.spines.values():
                spine.set_color(DARK_COLORS['border'])
                spine.set_linewidth(0.5)

        self._fig.tight_layout()
        self._canvas.draw()

    # ------------------------------------------------------------------
    # REPORT STATEMENTS
    # ------------------------------------------------------------------

    def _copy_report_to_clipboard(self):
        """Copy report statements to system clipboard."""
        try:
            txt = self._report_text.toPlainText()
            if txt.strip():
                QApplication.clipboard().setText(txt)
        except Exception as exc:
            QMessageBox.warning(self, "Clipboard Error",
                                f"Could not copy to clipboard:\n\n{exc}")

    # ------------------------------------------------------------------
    # CARRY-OVER SUMMARY SUBTAB
    # ------------------------------------------------------------------

    def _update_carry_over_summary(self):
        """Populate the Carry-Over Summary subtab table after computation."""
        if not self._results:
            self._carry_table.setRowCount(0)
            self._carry_warnings.clear()
            self._btn_copy_all_carry.setEnabled(False)
            return

        n_qty = len(self._results)
        self._carry_table.setRowCount(n_qty)
        warnings_lines = []

        for q_idx, res in enumerate(self._results):
            q_label = self._qty_label(q_idx)
            q_unit = self._qty_unit(q_idx)
            unit_sfx = f" {q_unit}" if q_unit else ""
            tgt = res.target_grid_idx

            # Determine carry value (same logic as carry-over box)
            carry_u = None
            if (res.is_valid and res.per_grid_u_num
                    and 0 <= tgt < len(res.per_grid_u_num)
                    and not np.isnan(res.per_grid_u_num[tgt])):
                carry_u = res.per_grid_u_num[tgt]
            elif res.is_valid and not np.isnan(res.u_num):
                carry_u = res.u_num

            conv_type = res.convergence_type or "unknown"

            if carry_u is not None and conv_type != "divergent":
                u_str = f"{carry_u:.6g}{unit_sfx}"
                dist = "Normal"
                dof = "\u221e"
                basis = "Confirmed 1\u03c3"
                status = "READY"
                status_color = DARK_COLORS['green']
            elif conv_type == "divergent":
                # Check for bounding estimate
                sols = res.grid_solutions
                spread = max(sols) - min(sols)
                mean_v = np.mean(sols)
                spread_pct = ((spread / abs(mean_v) * 100.0)
                              if abs(mean_v) > 1e-30 else float('inf'))
                if spread_pct < 2.0:
                    u_str = f"{spread/2:.6g}{unit_sfx} (bounding)"
                    dist = "Uniform"
                    status = "BOUNDING"
                    status_color = DARK_COLORS['yellow']
                else:
                    u_str = "N/A"
                    dist = "N/A"
                    status = "BLOCKED"
                    status_color = DARK_COLORS['red']
                dof = "\u221e"
                basis = "Bounding estimate"
                warnings_lines.append(
                    f"{q_label}: DIVERGENT convergence "
                    f"(R = {res.convergence_ratio:.4f}). "
                    f"Spread = {spread:.4g}{unit_sfx} "
                    f"({spread_pct:.1f}%).")
            else:
                u_str = "N/A"
                dist = "N/A"
                dof = "N/A"
                basis = "N/A"
                status = "INVALID"
                status_color = DARK_COLORS['red']
                warnings_lines.append(
                    f"{q_label}: No valid u_num available.")

            if conv_type == "oscillatory":
                status = "CAUTION"
                status_color = DARK_COLORS['yellow']
                warnings_lines.append(
                    f"{q_label}: Oscillatory convergence. "
                    f"u_num is a conservative estimate.")

            items_data = [
                (q_label, None),
                (u_str, None),
                (q_unit, None),
                (dist, None),
                (dof, None),
                (basis, None),
                (conv_type.capitalize(), None),
                (status, status_color),
            ]
            for col, (text, color) in enumerate(items_data):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if color:
                    item.setForeground(QColor(color))
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                self._carry_table.setItem(q_idx, col, item)

        self._carry_warnings.setPlainText("\n".join(warnings_lines))
        self._btn_copy_all_carry.setEnabled(True)

    def _copy_all_carry_values(self):
        """Copy all carry-over values to clipboard as tab-separated text."""
        if not self._results:
            return
        try:
            lines = ["Quantity\tu_num\tUnit\tDistribution\tDOF\tSigma Basis\t"
                     "Convergence\tStatus"]
            for row in range(self._carry_table.rowCount()):
                row_data = []
                for col in range(self._carry_table.columnCount()):
                    item = self._carry_table.item(row, col)
                    row_data.append(item.text() if item else "")
                lines.append("\t".join(row_data))
            QApplication.clipboard().setText("\n".join(lines))
        except Exception as exc:
            QMessageBox.warning(self, "Clipboard Error",
                                f"Could not copy:\n\n{exc}")

    def _copy_carry_to_clipboard(self):
        """Copy u_num carry-over value to clipboard."""
        if not self._results:
            return
        try:
            res = self._results[0]
            tgt = res.target_grid_idx
            q_unit = self._qty_unit(0)
            unit_sfx = f" {q_unit}" if q_unit else ""

            # Determine carry value
            carry_u = res.u_num
            if (res.per_grid_u_num and 0 <= tgt < len(res.per_grid_u_num)
                    and not np.isnan(res.per_grid_u_num[tgt])):
                carry_u = res.per_grid_u_num[tgt]
            if np.isnan(carry_u):
                return

            cells = res.grid_cells[tgt] if tgt < len(res.grid_cells) else 0
            grid_label = f"Grid {tgt+1} ({cells:,.0f} cells)"

            parts = [
                f"u_num = {carry_u:.6g}{unit_sfx}",
                "Basis: Confirmed 1\u03c3",
                "DOF: \u221e",
                f"Source: {grid_label}  |  Fs = {res.safety_factor:.2f}",
            ]

            # Directional info
            dir_u = res.u_num_directional
            if not np.isnan(dir_u) and res.direction_of_concern != "both":
                if dir_u == 0.0:
                    parts.append(
                        f"Direction: {res.direction_of_concern} "
                        f"\u2192 CONSERVATIVE (u_num = 0.0)")
                else:
                    parts.append(
                        f"Direction: {res.direction_of_concern} "
                        f"\u2192 adverse bias (u_num = {dir_u:.6g}{unit_sfx})")

            parts.append(
                "Method: GCI per Celik et al. (2008), ASME V&V 20 \u00a75.1")

            clipboard_text = "\n".join(parts)
            QApplication.clipboard().setText(clipboard_text)

            # Brief visual feedback
            orig_text = self._btn_copy_carry.text()
            self._btn_copy_carry.setText("\u2713 Copied!")
            QTimer.singleShot(
                1500, lambda: self._btn_copy_carry.setText(orig_text))

        except Exception as exc:
            QMessageBox.warning(
                self, "Clipboard Error",
                f"Could not copy carry value:\n\n{exc}")

    def _export_summary_card(self):
        """Export a compact PNG summary card with convergence plot + metrics.

        Creates a single-image card suitable for PowerPoint or report appendices.
        Left panel (60%): Convergence plot with RE curve and error bands.
        Right panel (40%): Key GCI metrics as formatted text.
        """
        if not self._results:
            return

        res = self._results[0]
        q_unit = self._qty_unit(0)
        unit_sfx = f" {q_unit}" if q_unit else ""

        # ---- Ask for save path ----
        default_name = "GCI_Summary_Card.png"
        path, filt = QFileDialog.getSaveFileName(
            self, "Export GCI Summary Card", default_name,
            "PNG Image (*.png);;PDF Document (*.pdf);;All Files (*)")
        if not path:
            return

        # ---- Use a light report theme for print quality ----
        CARD_BG = '#ffffff'
        CARD_FG = '#1a1a2e'
        CARD_DIM = '#555566'
        CARD_ACCENT = '#2563eb'
        CARD_GREEN = '#16a34a'
        CARD_ORANGE = '#ea580c'
        CARD_YELLOW = '#ca8a04'
        CARD_RED = '#dc2626'
        CARD_BORDER = '#d1d5db'

        try:
            fig = Figure(figsize=(10, 5), facecolor=CARD_BG)
            gs = GridSpec(1, 2, width_ratios=[6, 4], figure=fig,
                         left=0.07, right=0.97, top=0.90, bottom=0.12,
                         wspace=0.05)

            # ---- LEFT PANEL: Convergence plot ----
            ax = fig.add_subplot(gs[0, 0])
            ax.set_facecolor('#f8fafc')

            spacings = np.array(res.grid_spacings)
            solutions = np.array(res.grid_solutions)
            tgt = res.target_grid_idx
            has_re = not np.isnan(res.richardson_extrapolation)

            # Grid solutions
            ax.plot(spacings, solutions, 'o-', color=CARD_ACCENT,
                    markersize=7, linewidth=1.8, label='Grid solutions',
                    zorder=5)

            # Production grid marker
            if 0 <= tgt < len(spacings):
                ax.plot(spacings[tgt], solutions[tgt], 'D',
                        color=CARD_ORANGE, markersize=11,
                        markeredgecolor=CARD_FG, markeredgewidth=1.2,
                        zorder=7, label=f'Production (Grid {tgt+1})')

            # Richardson extrapolation
            if has_re:
                f_ext = res.richardson_extrapolation
                ax.axhline(y=f_ext, color=CARD_GREEN, linestyle='--',
                           linewidth=1.3, alpha=0.7,
                           label=f'RE = {f_ext:.6g}')
                ax.plot(0, f_ext, '*', color=CARD_GREEN,
                        markersize=12, zorder=6)

                # RE convergence curve
                p_obs = res.observed_order
                if (not np.isnan(p_obs) and 0 < p_obs < 100
                        and spacings[0] > 1e-30
                        and abs(solutions[0] - f_ext) > 1e-30):
                    C_coeff = ((solutions[0] - f_ext)
                               / _safe_rp(spacings[0], p_obs))
                    if np.isfinite(C_coeff):
                        h_curve = np.linspace(0, spacings[-1] * 1.05, 200)
                        f_curve = f_ext + C_coeff * h_curve ** p_obs
                        ax.plot(h_curve, f_curve, '-',
                                color=CARD_GREEN, linewidth=1.0,
                                alpha=0.35, zorder=2,
                                label=f'RE trend (p={p_obs:.2f})')

            # Fine-grid u_num band
            if not np.isnan(res.u_num) and len(solutions) > 0:
                f1 = solutions[0]
                band = 2 * res.u_num
                ax.fill_between(
                    [0, spacings[0]],
                    [f1 - band, f1 - band],
                    [f1 + band, f1 + band],
                    alpha=0.12, color=CARD_ACCENT,
                    label=f'Fine u_num \u00b12\u03c3')

            # Production-grid u_num band
            if (tgt > 0 and 0 <= tgt < len(spacings) and
                    res.per_grid_u_num and
                    tgt < len(res.per_grid_u_num) and
                    not np.isnan(res.per_grid_u_num[tgt])):
                f_prod = solutions[tgt]
                u_prod = res.per_grid_u_num[tgt]
                bh = 2 * u_prod
                ax.fill_between(
                    [spacings[tgt] * 0.85, spacings[tgt] * 1.15],
                    [f_prod - bh, f_prod - bh],
                    [f_prod + bh, f_prod + bh],
                    alpha=0.2, color=CARD_ORANGE, zorder=3)

            # Grid labels
            for i, (h, f) in enumerate(zip(spacings, solutions)):
                c = CARD_ORANGE if i == tgt else CARD_DIM
                tag = f"G{i+1}\u2605" if i == tgt else f"G{i+1}"
                ax.annotate(tag, (h, f), textcoords="offset points",
                            xytext=(7, 7), fontsize=6.5, color=c,
                            fontweight='bold' if i == tgt else 'normal')

            ax.set_xlabel("Representative spacing h", fontsize=8,
                          color=CARD_FG)
            ax.set_ylabel(f"Solution{' (' + q_unit + ')' if q_unit else ''}",
                          fontsize=8, color=CARD_FG)
            ax.set_title("Grid Convergence Study", fontsize=10,
                         fontweight='bold', color=CARD_FG)
            ax.legend(loc='best', fontsize=6, framealpha=0.8)
            ax.grid(True, alpha=0.2)
            ax.set_xlim(left=-0.001)
            ax.tick_params(labelsize=7, colors=CARD_FG)
            for spine in ax.spines.values():
                spine.set_color(CARD_BORDER)

            # ---- RIGHT PANEL: Metrics text ----
            ax_txt = fig.add_subplot(gs[0, 1])
            ax_txt.axis('off')

            # Build metrics lines
            lines = []
            lines.append("GCI Summary")
            lines.append("\u2500" * 32)
            lines.append("")
            lines.append(f"Grids analyzed:      {res.n_grids}")

            # Production grid info
            cells = res.grid_cells[tgt] if tgt < len(res.grid_cells) else 0
            if cells > 0:
                lines.append(
                    f"Production grid:     Grid {tgt+1} ({cells:,.0f} cells)")
            else:
                lines.append(f"Production grid:     Grid {tgt+1}")

            # Convergence type
            conv = res.convergence_type
            if res.effectively_grid_independent:
                conv = "grid-independent"
            lines.append(f"Convergence:         {conv}")

            # Auto-selected triplet
            if res.auto_selected_triplet is not None:
                trip_str = "-".join(str(g+1) for g in res.auto_selected_triplet)
                lines.append(f"Selected triplet:    Grids {trip_str}")

            lines.append("")

            # Order
            if not np.isnan(res.observed_order) and res.observed_order < 1e6:
                lines.append(f"Observed order:      {res.observed_order:.2f}")
            elif res.order_is_assumed:
                lines.append(
                    f"Assumed order:       {res.theoretical_order:.1f}")
            lines.append(f"Theoretical order:   {res.theoretical_order:.1f}")

            # Order diagnostic
            if (not res.order_is_assumed and res.theoretical_order > 0
                    and not np.isnan(res.observed_order)
                    and 0 < res.observed_order < 100):
                p_o = res.observed_order
                p_t = res.theoretical_order
                lo, hi = 0.5 * p_t, 2.0 * p_t
                if lo <= p_o <= hi:
                    lines.append(
                        f"Order diagnostic:    \u2713 [{lo:.1f}, {hi:.1f}]")
                else:
                    lines.append(
                        f"Order diagnostic:    \u2717 [{lo:.1f}, {hi:.1f}]")

            lines.append("")

            # u_num values
            carry_u = res.u_num
            if (res.per_grid_u_num and 0 <= tgt < len(res.per_grid_u_num)
                    and not np.isnan(res.per_grid_u_num[tgt])):
                carry_u = res.per_grid_u_num[tgt]

            lines.append(f"u_num (std):         {carry_u:.4g}{unit_sfx}")

            dir_u = res.u_num_directional
            if not np.isnan(dir_u) and res.direction_of_concern != "both":
                lines.append(
                    f"u_num (directional): {dir_u:.4g}{unit_sfx}")
                lines.append(
                    f"Direction:           {res.direction_of_concern}")
                if dir_u == 0.0:
                    lines.append(
                        f"Bias:                Conservative \u2713")
                else:
                    lines.append(
                        f"Bias:                Adverse")

            lines.append("")
            lines.append(f"Safety factor:       {res.safety_factor:.2f}")
            lines.append(f"Method:              GCI (Celik 2008)")
            lines.append(f"Standard:            ASME V&V 20 \u00a75.1")

            # Richardson extrapolation
            if has_re:
                lines.append("")
                lines.append(
                    f"f_exact (RE):        {res.richardson_extrapolation:.6g}"
                    f"{unit_sfx}")

            metrics_text = "\n".join(lines)

            ax_txt.text(0.05, 0.95, metrics_text,
                        transform=ax_txt.transAxes,
                        verticalalignment='top',
                        fontfamily='monospace', fontsize=7.5,
                        color=CARD_FG, linespacing=1.4)

            # ---- Title banner ----
            fig.text(0.5, 0.96, "GCI Analysis Summary Card",
                     ha='center', va='top', fontsize=12,
                     fontweight='bold', color=CARD_FG)

            # ---- Footer with timestamp ----
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            fig.text(0.97, 0.02,
                     f"Generated {now}  |  Celik et al. (2008), "
                     f"ASME V&V 20-2009",
                     ha='right', va='bottom', fontsize=5.5,
                     color=CARD_DIM, style='italic')

            # ---- Save ----
            dpi = 300
            if path.lower().endswith('.pdf'):
                fig.savefig(path, dpi=dpi, facecolor=CARD_BG,
                            bbox_inches='tight')
            else:
                fig.savefig(path, dpi=dpi, facecolor=CARD_BG,
                            bbox_inches='tight', pad_inches=0.15)
            plt.close(fig)

            QMessageBox.information(
                self, "Summary Card Exported",
                f"Summary card saved to:\n\n{path}")

        except Exception as exc:
            QMessageBox.warning(
                self, "Export Error",
                f"Could not export summary card:\n\n{exc}")

    def _generate_report_statements(self):
        """Generate copy-pasteable regulatory paragraphs for V&V reports.

        Called at the end of _display_results().  Produces five sections:
        1. Numerical Uncertainty Statement (per quantity)
        2. Production Grid Statement (if production != finest)
        3. Multi-Quantity Summary (if > 1 quantity)
        4. Limitations & Caveats
        5. Standards Compliance
        """
        if not self._results:
            self._report_text.clear()
            return

        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("  REPORT STATEMENTS FOR V&V DOCUMENTATION")
        lines.append("  (Copy-paste into regulatory submission or V&V report)")
        lines.append("=" * 70)
        lines.append("")

        # ---- Section 1: Numerical Uncertainty Statement ----
        lines.append("1. NUMERICAL UNCERTAINTY STATEMENT")
        lines.append("-" * 50)
        lines.append("")

        for q_idx, res in enumerate(self._results):
            q_label = self._qty_label(q_idx)
            q_unit = self._qty_unit(q_idx)
            unit_sfx = f" {q_unit}" if q_unit else ""
            tgt = res.target_grid_idx

            if len(self._results) > 1:
                lines.append(f"  [{q_label}]")

            if res.convergence_type == "monotonic":
                # Full statement for monotonic convergence
                f_fine = res.grid_solutions[0]
                n_fine = res.grid_cells[0]
                n_coarse = res.grid_cells[-1]
                r_ratios = ", ".join(
                    f"{r:.2f}" for r in res.refinement_ratios)

                u_val = res.u_num
                u_pct = res.u_num_pct
                # Use production grid values if applicable
                if (res.per_grid_u_num and
                        0 <= tgt < len(res.per_grid_u_num) and
                        not np.isnan(res.per_grid_u_num[tgt])):
                    u_val = res.per_grid_u_num[tgt]
                    u_pct = res.per_grid_u_pct[tgt]

                lines.append(
                    f"  A grid convergence study was performed using "
                    f"{res.n_grids} systematically refined grids "
                    f"({n_fine:,.0f} to {n_coarse:,.0f} cells, "
                    f"refinement ratio(s): {r_ratios}). "
                    f"Monotonic convergence was observed with a "
                    f"convergence ratio R = {res.convergence_ratio:.4f}. "
                    f"The {'assumed' if res.order_is_assumed else 'observed'}"
                    f" order of accuracy "
                    f"p = {res.observed_order:.2f} "
                    f"(theoretical: {res.theoretical_order:.1f}). "
                    f"Using the Grid Convergence Index (GCI) method "
                    f"with a safety factor Fs = {res.safety_factor:.2f}, "
                    f"the numerical uncertainty for {q_label} is "
                    f"u_num = {u_val:.4g}{unit_sfx} "
                    f"({u_pct:.2f}% of the solution). "
                    f"The expanded uncertainty (k=2) is "
                    f"+/-{2*u_val:.4g}{unit_sfx} "
                    f"({2*u_pct:.2f}%)."
                )
                if not np.isnan(res.asymptotic_ratio):
                    lines.append(
                        f"  The asymptotic ratio of "
                        f"{res.asymptotic_ratio:.4f} "
                        + ("confirms the grids are in the asymptotic "
                           "range."
                           if 0.95 <= res.asymptotic_ratio <= 1.05
                           else "indicates the grids are approximately "
                           "in the asymptotic range.")
                    )

            elif res.convergence_type == "oscillatory":
                # Oscillatory convergence statement
                u_val = res.u_num
                u_pct = res.u_num_pct
                if (res.per_grid_u_num and
                        0 <= tgt < len(res.per_grid_u_num) and
                        not np.isnan(res.per_grid_u_num[tgt])):
                    u_val = res.per_grid_u_num[tgt]
                    u_pct = res.per_grid_u_pct[tgt]

                lines.append(
                    f"  A grid convergence study was performed using "
                    f"{res.n_grids} grids. Oscillatory convergence "
                    f"was observed (R = {res.convergence_ratio:.4f}), "
                    f"indicating the solution alternates between "
                    f"grid levels. Richardson extrapolation is not "
                    f"reliable in this regime. The numerical "
                    f"uncertainty was estimated from the oscillation "
                    f"range with a conservative safety factor "
                    f"Fs = {res.safety_factor:.2f}. "
                    f"The numerical uncertainty for {q_label} is "
                    f"u_num = {u_val:.4g}{unit_sfx} "
                    f"({u_pct:.2f}%). "
                    f"The expanded uncertainty (k=2) is "
                    f"+/-{2*u_val:.4g}{unit_sfx} "
                    f"({2*u_pct:.2f}%)."
                )

            elif res.convergence_type == "divergent":
                sols = res.grid_solutions
                spread = max(sols) - min(sols)
                mean_v = np.mean(sols)
                spread_pct_r = ((spread / abs(mean_v) * 100.0)
                                if abs(mean_v) > 1e-30 else float('inf'))
                lines.append(
                    f"  A grid convergence study was performed using "
                    f"{res.n_grids} grids. DIVERGENT behavior was "
                    f"observed (R = {res.convergence_ratio:.4f}) — the "
                    f"solution does not converge monotonically with "
                    f"grid refinement. The Grid Convergence Index "
                    f"(GCI) method does not yield a valid numerical "
                    f"uncertainty estimate for {q_label} under these "
                    f"conditions."
                )
                lines.append("")
                lines.append(
                    f"  The solution spread across grids is "
                    f"{spread:.4g}{unit_sfx} ({spread_pct_r:.1f}% "
                    f"of the mean)."
                )
                if spread_pct_r < 2.0:
                    lines.append(
                        f"  Given the small spread, a conservative "
                        f"bounding uncertainty of u_num = "
                        f"{spread/2:.4g}{unit_sfx} (half the spread, "
                        f"treated as a bounding estimate) MAY be "
                        f"justified with appropriate engineering "
                        f"documentation. This is NOT a GCI-derived "
                        f"value. Use Uniform distribution."
                    )
                else:
                    lines.append(
                        f"  Additional grid refinement, investigation "
                        f"of solver settings, or an alternative "
                        f"uncertainty estimation approach is required "
                        f"before a numerical uncertainty can be "
                        f"assigned."
                    )

            elif res.convergence_type == "grid-independent":
                lines.append(
                    f"  A grid convergence study was performed using "
                    f"{res.n_grids} grids. The solution for {q_label} "
                    f"is grid-independent — all grids produce "
                    f"identical results within numerical precision. "
                    f"The numerical uncertainty is effectively zero."
                )

            elif res.method.startswith("2-grid"):
                u_val = res.u_num
                u_pct = res.u_num_pct
                lines.append(
                    f"  A 2-grid convergence study was performed for "
                    f"{q_label} using an assumed order of accuracy "
                    f"p = {res.theoretical_order:.1f}. "
                    f"The numerical uncertainty is "
                    f"u_num = {u_val:.4g}{unit_sfx} "
                    f"({u_pct:.2f}%). "
                    f"NOTE: A 2-grid study cannot determine the "
                    f"observed order or verify asymptotic convergence. "
                    f"A 3-grid (or more) study is recommended for "
                    f"certification-level analysis."
                )

            lines.append("")

        # ---- Production Grid Statement ----
        sect = 1  # running section counter (section 1 was above)
        has_prod = any(r.target_grid_idx > 0 for r in self._results)
        if has_prod:
            sect += 1
            lines.append(f"{sect}. PRODUCTION GRID STATEMENT")
            lines.append("-" * 50)
            lines.append("")
            for q_idx, res in enumerate(self._results):
                tgt = res.target_grid_idx
                if tgt <= 0:
                    continue
                q_label = self._qty_label(q_idx)
                q_unit = self._qty_unit(q_idx)
                unit_sfx = f" {q_unit}" if q_unit else ""
                if (res.per_grid_u_num and
                        0 <= tgt < len(res.per_grid_u_num) and
                        not np.isnan(res.per_grid_u_num[tgt])):
                    u_prod = res.per_grid_u_num[tgt]
                    u_prod_pct = res.per_grid_u_pct[tgt]
                    f_prod = res.grid_solutions[tgt]
                    lines.append(
                        f"  The production simulation uses Grid {tgt+1} "
                        f"({res.grid_cells[tgt]:,.0f} cells), which is "
                        f"coarser than the finest grid in the "
                        f"convergence study. For {q_label}, the "
                        f"production-grid numerical uncertainty is "
                        f"u_num = {u_prod:.4g}{unit_sfx} "
                        f"({u_prod_pct:.2f}% of "
                        f"f = {f_prod:.6g}{unit_sfx}), which is "
                        f"{u_prod/res.u_num:.1f}x the fine-grid "
                        f"uncertainty."
                        if np.isfinite(res.u_num) and res.u_num > 0 else
                        f"u_num = {u_prod:.4g}{unit_sfx} "
                        f"({u_prod_pct:.2f}%)."
                    )
                    lines.append("")
            lines.append("")

        # ---- Multi-Quantity Summary ----
        if len(self._results) > 1:
            sect += 1
            lines.append(f"{sect}. MULTI-QUANTITY SUMMARY")
            lines.append("-" * 50)
            lines.append("")
            lines.append("  Quantity                | Conv. Type   | u_num        | u_num %")
            lines.append("  " + "-" * 66)
            dominant_u = 0.0
            dominant_q = ""
            for q_idx, res in enumerate(self._results):
                q_label = self._qty_label(q_idx)
                tgt = res.target_grid_idx
                u_val = res.u_num
                u_pct = res.u_num_pct
                if (res.per_grid_u_num and
                        0 <= tgt < len(res.per_grid_u_num) and
                        not np.isnan(res.per_grid_u_num[tgt])):
                    u_val = res.per_grid_u_num[tgt]
                    u_pct = res.per_grid_u_pct[tgt]
                lines.append(
                    f"  {q_label:<24s}| "
                    f"{res.convergence_type:<13s}| "
                    f"{u_val:<13.4g}| "
                    f"{u_pct:.2f}%"
                )
                if u_pct > dominant_u:
                    dominant_u = u_pct
                    dominant_q = q_label
            lines.append("")
            if dominant_q:
                lines.append(
                    f"  Dominant numerical uncertainty: {dominant_q} "
                    f"({dominant_u:.2f}%)"
                )
            lines.append("")

        # ---- Limitations & Caveats ----
        sect += 1
        lines.append(f"{sect}. LIMITATIONS & CAVEATS")
        lines.append("-" * 50)
        lines.append("")

        caveats: List[str] = []
        for q_idx, res in enumerate(self._results):
            q_label = self._qty_label(q_idx)
            if res.refinement_ratios:
                r_min = min(res.refinement_ratios)
                if r_min < 1.3:
                    caveats.append(
                        f"  - Low refinement ratio (r_min = {r_min:.3f} "
                        f"< 1.3 recommended) for {q_label}. "
                        f"This may reduce the reliability of the "
                        f"observed order estimate."
                    )
            if (not np.isnan(res.observed_order) and
                    res.observed_order < 1e6):
                ratio_p = abs(res.observed_order - res.theoretical_order
                              ) / max(res.theoretical_order, 0.1)
                if ratio_p > 0.5:
                    caveats.append(
                        f"  - Observed order p = {res.observed_order:.2f}"
                        f" deviates significantly from theoretical "
                        f"({res.theoretical_order:.1f}) for {q_label}."
                    )
            if res.convergence_type == "oscillatory":
                caveats.append(
                    f"  - Oscillatory convergence for {q_label}: "
                    f"Richardson extrapolation is unreliable; "
                    f"u_num is based on oscillation range with "
                    f"Fs = 3.0."
                )
            if res.convergence_type == "divergent":
                caveats.append(
                    f"  - DIVERGENT behavior for {q_label}: no valid "
                    f"numerical uncertainty can be assigned."
                )
            if res.method.startswith("2-grid"):
                caveats.append(
                    f"  - 2-grid study for {q_label}: assumed order "
                    f"p = {res.theoretical_order:.1f}. A 3+ grid "
                    f"study is recommended for certification."
                )
            if res.warnings:
                for w in res.warnings:
                    if "near-zero" in w.lower():
                        caveats.append(
                            f"  - Near-zero solution warning for "
                            f"{q_label}: relative uncertainty may be "
                            f"artificially inflated."
                        )

        if caveats:
            for c in caveats:
                lines.append(c)
        else:
            lines.append("  No significant limitations identified.")
        lines.append("")

        # ---- Alternative Methods Note ----
        has_alt = (hasattr(self, '_fs_results') and
                   any(r is not None for r in self._fs_results))
        has_lsr = (hasattr(self, '_lsr_results') and
                   any(r is not None for r in self._lsr_results))

        if has_alt or has_lsr:
            sect += 1
            lines.append(
                f"{sect}. ALTERNATIVE METHOD COMPARISON")
            lines.append("-" * 50)
            lines.append("")
            alt_parts: List[str] = []
            for q_idx, res in enumerate(self._results):
                q_label = self._qty_label(q_idx)
                methods_str = []
                if has_alt and q_idx < len(self._fs_results):
                    fs_r = self._fs_results[q_idx]
                    if fs_r and fs_r.get("is_valid"):
                        methods_str.append(
                            f"FS variant (Xing & Stern 2010): "
                            f"u_num = {fs_r['u_num']:.4g} "
                            f"({fs_r['u_num_pct']:.2f}%)")
                if has_lsr and q_idx < len(self._lsr_results):
                    lsr_r = self._lsr_results[q_idx]
                    if lsr_r and lsr_r.get("is_valid"):
                        methods_str.append(
                            f"LSR variant with AICc (Eca & Hoekstra 2014): "
                            f"u_num = {lsr_r['u_num']:.4g} "
                            f"({lsr_r['u_num_pct']:.2f}%)")
                if methods_str:
                    alt_parts.append(
                        f"  For {q_label}, alternative methods yielded: "
                        + "; ".join(methods_str) + "."
                    )
            if alt_parts:
                for part in alt_parts:
                    lines.append(part)
                lines.append("")
                lines.append(
                    "  These alternative estimates provide independent "
                    "cross-checks of the GCI-based numerical "
                    "uncertainty. Consistency between methods "
                    "increases confidence in the uncertainty estimate."
                )
            lines.append("")

        # ---- UQ Mapping Assumption ----
        sect += 1
        lines.append(f"{sect}. UQ MAPPING ASSUMPTION")
        lines.append("-" * 50)
        lines.append("")
        lines.append(
            "  The numerical uncertainty is defined as:")
        lines.append(
            "      u_num = delta = |f1 - f_RE|")
        lines.append(
            "  where f1 is the fine-grid solution and f_RE is the "
            "Richardson extrapolation estimate.")
        lines.append("")
        lines.append(
            "  This quantity is treated as a 1-sigma equivalent "
            "estimate for aggregation purposes (e.g., in the "
            "V&V 20 RSS uncertainty budget). This is a modeling "
            "assumption, not a statistically derived quantity. "
            "The GCI error band (Fs x u_num) is a conservative "
            "engineering estimate of the discretization error, "
            "not a formal confidence interval.")
        lines.append("")
        lines.append(
            "  The 1-sigma interpretation follows Roache (1998) and "
            "is standard practice in ASME V&V 20 Section 5.1. "
            "DOF is set to infinity because the estimate is "
            "deterministic (model-based), not sampled from a "
            "statistical population.")
        lines.append("")

        # ---- Standards Compliance ----
        sect += 1
        lines.append(f"{sect}. STANDARDS COMPLIANCE")
        lines.append("-" * 50)
        lines.append("")
        lines.append(
            "  Method:     Grid Convergence Index (GCI) per Celik "
            "et al. (2008)"
        )
        lines.append(
            "  Procedure:  ASME V&V 20-2009 (R2021) Section 5.1"
        )
        lines.append(
            "  Basis:      Richardson Extrapolation with observed "
            "or assumed order"
        )
        lines.append(
            "  Coverage:   u_num = |f1 - f_RE| provides a standard "
            "(1-sigma) uncertainty estimate;"
        )
        lines.append(
            "              the expanded uncertainty (k=2) provides "
            "conservative coverage."
        )
        lines.append(
            "  References: Celik et al. (2008) J. Fluids Eng. "
            "130(7), 078001;"
        )
        lines.append(
            "              Roache (1998) Verification and Validation "
            "in Computational"
        )
        lines.append(
            "              Science and Engineering, Hermosa Publishers."
        )
        if has_alt:
            lines.append(
                "              Xing & Stern (2010) ASME J. Fluids "
                "Eng. 132(6), 061403"
            )
        if has_lsr:
            lines.append(
                "              Eca & Hoekstra (2014) J. Comp. Physics "
                "262, 104-130"
            )
        lines.append("")

        # ---- Formal References ----
        sect += 1
        lines.append(f"{sect}. REFERENCES")
        lines.append("-" * 50)
        lines.append("")
        ref_num = 1
        lines.append(
            f"  [{ref_num}] Celik, I.B., Ghia, U., Roache, P.J., "
            "Freitas, C.J., Coleman, H.,"
        )
        lines.append(
            "      and Raad, P.E. (2008). \"Procedure for Estimation "
            "and Reporting"
        )
        lines.append(
            "      of Uncertainty Due to Discretization in CFD "
            "Applications.\""
        )
        lines.append(
            "      J. Fluids Eng., 130(7), 078001."
        )
        lines.append("")
        ref_num += 1
        lines.append(
            f"  [{ref_num}] Roache, P.J. (1998). Verification and "
            "Validation in Computational"
        )
        lines.append(
            "      Science and Engineering. Hermosa Publishers, "
            "Albuquerque, NM."
        )
        lines.append("")
        ref_num += 1
        lines.append(
            f"  [{ref_num}] ASME V&V 20-2009 (R2021). Standard for "
            "Verification and Validation"
        )
        lines.append(
            "      in Computational Fluid Dynamics and Heat Transfer."
        )
        lines.append("")
        ref_num += 1
        lines.append(
            f"  [{ref_num}] Richardson, L.F. (1911). \"The Approximate "
            "Arithmetical Solution"
        )
        lines.append(
            "      by Finite Differences of Physical Problems "
            "Involving Differential"
        )
        lines.append(
            "      Equations.\" Phil. Trans. R. Soc. A, 210, 307-357."
        )
        if has_alt:
            lines.append("")
            ref_num += 1
            lines.append(
                f"  [{ref_num}] Xing, T. and Stern, F. (2010). "
                "\"Factors of Safety for Richardson"
            )
            lines.append(
                "      Extrapolation.\" ASME J. Fluids Eng., "
                "132(6), 061403."
            )
        if has_lsr:
            lines.append("")
            ref_num += 1
            lines.append(
                f"  [{ref_num}] Eca, L. and Hoekstra, M. (2014). "
                "\"A Procedure for the Estimation"
            )
            lines.append(
                "      of the Numerical Uncertainty of CFD Calculations "
                "Based on Grid"
            )
            lines.append(
                "      Refinement Studies.\" J. Comp. Physics, "
                "262, 104-130."
            )
        lines.append("")
        lines.append("=" * 70)

        self._report_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # EXPORT
    # ------------------------------------------------------------------

    def get_results_text(self) -> str:
        return self._results_text.toPlainText()

    def get_results(self) -> List[GCIResult]:
        return list(self._results)

    # ------------------------------------------------------------------
    # STATE SERIALIZATION (save / load)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Serialize all input state to a dict for JSON saving."""
        n_grids = self._cmb_n_grids.currentData()
        state = {
            "version": APP_VERSION,
            "n_grids": n_grids,
            "dim_index": self._cmb_dim.currentIndex(),
            "order_preset_index": self._cmb_order_preset.currentIndex(),
            "theoretical_order": self._spn_theoretical_p.value(),
            "safety_factor": self._spn_safety.value(),
            "reference_scale": self._spn_ref_scale.value(),
            "production_grid": self._cmb_production.currentIndex(),
            "direction_of_concern": self._cmb_direction.currentIndex(),
            "n_quantities": self._n_quantities,
            "quantity_names": list(self._quantity_names),
            "quantity_units": list(self._quantity_units),
            "grid_data": [],
        }

        # Read table data
        for i in range(self._grid_table.rowCount()):
            row_data = []
            for j in range(self._grid_table.columnCount()):
                item = self._grid_table.item(i, j)
                if item and item.text().strip():
                    try:
                        row_data.append(float(item.text()))
                    except ValueError:
                        row_data.append(item.text())
                else:
                    row_data.append(None)
            state["grid_data"].append(row_data)

        return state

    def set_state(self, state: dict):
        """Restore input state from a dict (loaded from JSON)."""
        # Set number of grids
        n_grids = state.get("n_grids", 3)
        found = False
        for idx in range(self._cmb_n_grids.count()):
            if self._cmb_n_grids.itemData(idx) == n_grids:
                self._cmb_n_grids.setCurrentIndex(idx)
                found = True
                break
        if not found and n_grids > 0:
            # Custom grid count (> 6): add it to the combo
            self._cmb_n_grids.addItem(f"{n_grids} grids (loaded)", n_grids)
            self._cmb_n_grids.setCurrentIndex(
                self._cmb_n_grids.count() - 1)

        # Set dimension
        self._cmb_dim.setCurrentIndex(state.get("dim_index", 0))

        # Set theoretical order preset and safety factor
        preset_idx = state.get("order_preset_index", 0)
        if 0 <= preset_idx < self._cmb_order_preset.count():
            self._cmb_order_preset.setCurrentIndex(preset_idx)
        self._spn_theoretical_p.setValue(state.get("theoretical_order", 2.0))
        self._spn_safety.setValue(state.get("safety_factor", 0.0))
        self._spn_ref_scale.setValue(state.get("reference_scale", 0.0))

        # Set quantities
        self._n_quantities = state.get("n_quantities", 1)
        self._quantity_names = list(state.get("quantity_names", ["Temperature"]))
        self._quantity_units = list(state.get(
            "quantity_units", [""] * self._n_quantities))
        # Ensure units list matches quantity count
        while len(self._quantity_units) < self._n_quantities:
            self._quantity_units.append("")
        self._rebuild_table()
        self._rebuild_qty_config()

        # Set production grid
        prod_idx = state.get("production_grid", 0)
        if 0 <= prod_idx < self._cmb_production.count():
            self._cmb_production.setCurrentIndex(prod_idx)

        # Set direction of concern
        dir_idx = state.get("direction_of_concern", 0)
        if 0 <= dir_idx < self._cmb_direction.count():
            self._cmb_direction.setCurrentIndex(dir_idx)

        # Fill table data
        grid_data = state.get("grid_data", [])
        for i, row in enumerate(grid_data):
            if i >= self._grid_table.rowCount():
                break
            for j, val in enumerate(row):
                if j >= self._grid_table.columnCount():
                    break
                if val is not None:
                    text = f"{val:.6g}" if isinstance(val, (int, float)) else str(val)
                    item = QTableWidgetItem(text)
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    self._grid_table.setItem(i, j, item)

    def clear_all(self):
        """Reset the calculator tab to default state."""
        self._cmb_order_preset.setCurrentIndex(0)  # Second-order
        self._spn_theoretical_p.setValue(2.0)
        self._spn_safety.setValue(0.0)
        self._spn_ref_scale.setValue(0.0)
        self._cmb_dim.setCurrentIndex(0)
        self._cmb_n_grids.setCurrentIndex(1)  # 3 grids
        self._cmb_production.setCurrentIndex(0)
        self._cmb_direction.setCurrentIndex(0)  # "Both (two-sided)"
        self._n_quantities = 1
        self._quantity_names = ["Temperature"]
        self._quantity_units = ["K"]
        self._rebuild_table()
        self._rebuild_qty_config()
        self._results = []
        self._fs_results = []
        self._lsr_results = []
        self._results_text.clear()
        self._results_table.setRowCount(0)
        self._fig.clear()
        self._canvas.draw()
        self._guidance_convergence.set_guidance("", 'green')
        self._guidance_order.set_guidance("", 'green')
        self._guidance_asymptotic.set_guidance("", 'green')
        self._report_text.clear()
        self._btn_copy_carry.setEnabled(False)
        self._btn_summary_card.setEnabled(False)
        self._carry_table.setRowCount(0)
        self._carry_warnings.clear()
        self._btn_copy_all_carry.setEnabled(False)


# =============================================================================
# SPATIAL GCI TAB
# =============================================================================

class SpatialGCITab(QWidget):
    """Spatial / field GCI analysis — point-by-point GCI over a surface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._summary: Optional[SpatialGCISummary] = None
        self._setup_ui()

    def _setup_ui(self):
        splitter = QSplitter(Qt.Horizontal, self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(splitter)

        # ---- LEFT PANEL: Input ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)
        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        splitter.addWidget(left_scroll)

        # -- Spatial study setup --
        grp_setup = QGroupBox("Spatial Study Setup")
        setup_form = QFormLayout(grp_setup)
        setup_form.setSpacing(6)

        self._cmb_mode = QComboBox()
        self._cmb_mode.addItem("Pre-interpolated (single CSV)", "pre_interp")
        self._cmb_mode.addItem("Separate CSV files per grid", "sep_csv")
        self._cmb_mode.addItem("Separate Fluent .prof files", "sep_prof")
        self._cmb_mode.setToolTip(
            "Pre-interpolated: One CSV with columns x, y, [z], f_grid1, f_grid2, ...\n"
            "   All grids share the same point locations.\n\n"
            "Separate files: One file per grid. Points are interpolated\n"
            "   onto the finest grid using IDW (inverse distance weighting)."
        )
        setup_form.addRow("Data mode:", self._cmb_mode)

        self._cmb_dim_spatial = QComboBox()
        self._cmb_dim_spatial.addItem("3D", 3)
        self._cmb_dim_spatial.addItem("2D", 2)
        setup_form.addRow("Dimensions:", self._cmb_dim_spatial)

        self._cmb_order_preset_spatial = QComboBox()
        for label, val in THEORETICAL_ORDER_PRESETS:
            self._cmb_order_preset_spatial.addItem(label, val)
        self._cmb_order_preset_spatial.setCurrentIndex(0)
        self._cmb_order_preset_spatial.setToolTip(
            "Select the formal order of accuracy of your numerical scheme.\n"
            "Select 'Custom' to enter any value in the spinbox below."
        )
        setup_form.addRow("Scheme order:", self._cmb_order_preset_spatial)

        self._spn_p_theo = QDoubleSpinBox()
        self._spn_p_theo.setRange(1.0, 4.0)
        self._spn_p_theo.setValue(2.0)
        self._spn_p_theo.setSingleStep(0.5)
        self._spn_p_theo.setDecimals(1)
        setup_form.addRow("Theoretical order p:", self._spn_p_theo)

        self._spn_fs_spatial = QDoubleSpinBox()
        self._spn_fs_spatial.setRange(0.0, 5.0)
        self._spn_fs_spatial.setValue(0.0)
        self._spn_fs_spatial.setSingleStep(0.25)
        self._spn_fs_spatial.setDecimals(2)
        self._spn_fs_spatial.setSpecialValueText("Auto")
        self._spn_fs_spatial.setMinimum(0.0)
        setup_form.addRow("Safety factor Fs:", self._spn_fs_spatial)

        self._edt_qty_name = QLineEdit("Temperature")
        setup_form.addRow("Quantity name:", self._edt_qty_name)

        self._cmb_unit_cat = QComboBox()
        for cat in UNIT_PRESETS:
            self._cmb_unit_cat.addItem(cat)
        self._cmb_unit_val = QComboBox()
        self._cmb_unit_val.setEditable(True)
        self._cmb_unit_val.setMinimumWidth(80)
        unit_row = QHBoxLayout()
        unit_row.addWidget(self._cmb_unit_cat)
        unit_row.addWidget(self._cmb_unit_val)
        unit_widget = QWidget()
        unit_widget.setLayout(unit_row)
        setup_form.addRow("Unit:", unit_widget)
        self._cmb_unit_cat.setCurrentIndex(0)
        self._populate_spatial_unit_combo("Temperature")
        self._cmb_unit_cat.currentTextChanged.connect(
            self._populate_spatial_unit_combo)

        left_layout.addWidget(grp_setup)

        # -- Grid Data (stacked widget for different modes) --
        grp_grid = QGroupBox("Grid Data")
        grid_layout = QVBoxLayout(grp_grid)

        self._stack = QStackedWidget()
        grid_layout.addWidget(self._stack)

        # --- Page 0: Pre-interpolated CSV ---
        page_pre = QWidget()
        pre_lay = QVBoxLayout(page_pre)
        pre_lay.setContentsMargins(0, 0, 0, 0)

        pre_top = QHBoxLayout()
        self._btn_load_pre = QPushButton("Load Data...")
        self._btn_load_pre.clicked.connect(self._load_pre_interpolated)
        self._lbl_pre_file = QLabel("No file loaded")
        self._lbl_pre_file.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; font-style: italic;")
        pre_top.addWidget(self._btn_load_pre)
        pre_top.addWidget(self._lbl_pre_file, 1)
        pre_lay.addLayout(pre_top)

        self._lbl_pre_info = QLabel("")
        pre_lay.addWidget(self._lbl_pre_info)

        pre_lay.addWidget(QLabel("Cell counts per grid (finest first):"))
        self._tbl_pre_cells = QTableWidget(0, 2)
        self._tbl_pre_cells.setHorizontalHeaderLabels(["Grid", "Cell Count"])
        style_table(self._tbl_pre_cells)
        pre_lay.addWidget(self._tbl_pre_cells)

        self._stack.addWidget(page_pre)

        # --- Page 1: Separate CSV files ---
        page_sep = QWidget()
        sep_lay = QVBoxLayout(page_sep)
        sep_lay.setContentsMargins(0, 0, 0, 0)

        sep_lay.addWidget(QLabel(
            "Load one CSV per grid (columns: x, y, [z], value).\n"
            "Points will be interpolated to the finest grid."))

        self._tbl_sep_files = QTableWidget(3, 3)
        self._tbl_sep_files.setHorizontalHeaderLabels(
            ["File Path", "Cell Count", "Points"])
        style_table(self._tbl_sep_files, stretch_col=0)
        for i in range(3):
            self._tbl_sep_files.setVerticalHeaderItem(
                i, QTableWidgetItem(f"Grid {i+1}"))
        sep_lay.addWidget(self._tbl_sep_files)

        sep_btns = QHBoxLayout()
        self._btn_browse_sep = QPushButton("Browse Grid Files...")
        self._btn_browse_sep.clicked.connect(self._browse_sep_files)
        self._btn_add_grid_sep = QPushButton("+ Add Grid")
        self._btn_add_grid_sep.clicked.connect(self._add_sep_grid)
        self._btn_rem_grid_sep = QPushButton("- Remove Grid")
        self._btn_rem_grid_sep.clicked.connect(self._rem_sep_grid)
        sep_btns.addWidget(self._btn_browse_sep)
        sep_btns.addWidget(self._btn_add_grid_sep)
        sep_btns.addWidget(self._btn_rem_grid_sep)
        sep_btns.addStretch()
        sep_lay.addLayout(sep_btns)

        idw_row = QHBoxLayout()
        idw_row.addWidget(QLabel("IDW neighbors k:"))
        self._spn_idw_k = QSpinBox()
        self._spn_idw_k.setRange(1, 32)
        self._spn_idw_k.setValue(8)
        idw_row.addWidget(self._spn_idw_k)
        idw_row.addWidget(QLabel("  IDW power:"))
        self._spn_idw_power = QDoubleSpinBox()
        self._spn_idw_power.setRange(0.5, 6.0)
        self._spn_idw_power.setValue(2.0)
        self._spn_idw_power.setSingleStep(0.5)
        idw_row.addWidget(self._spn_idw_power)
        idw_row.addStretch()
        sep_lay.addLayout(idw_row)

        self._stack.addWidget(page_sep)

        # --- Page 2: Separate Fluent .prof files ---
        page_prof = QWidget()
        prof_lay = QVBoxLayout(page_prof)
        prof_lay.setContentsMargins(0, 0, 0, 0)
        prof_lay.addWidget(QLabel(
            "Load one Fluent .prof file per grid.\n"
            "Points will be interpolated to the finest grid."))

        self._tbl_prof_files = QTableWidget(3, 3)
        self._tbl_prof_files.setHorizontalHeaderLabels(
            ["File Path", "Cell Count", "Points"])
        style_table(self._tbl_prof_files, stretch_col=0)
        for i in range(3):
            self._tbl_prof_files.setVerticalHeaderItem(
                i, QTableWidgetItem(f"Grid {i+1}"))
        prof_lay.addWidget(self._tbl_prof_files)

        prof_btns = QHBoxLayout()
        self._btn_browse_prof = QPushButton("Browse .prof Files...")
        self._btn_browse_prof.clicked.connect(self._browse_prof_files)
        self._btn_add_grid_prof = QPushButton("+ Add Grid")
        self._btn_add_grid_prof.clicked.connect(self._add_prof_grid)
        self._btn_rem_grid_prof = QPushButton("- Remove Grid")
        self._btn_rem_grid_prof.clicked.connect(self._rem_prof_grid)
        prof_btns.addWidget(self._btn_browse_prof)
        prof_btns.addWidget(self._btn_add_grid_prof)
        prof_btns.addWidget(self._btn_rem_grid_prof)
        prof_btns.addStretch()
        prof_lay.addLayout(prof_btns)

        idw_row2 = QHBoxLayout()
        idw_row2.addWidget(QLabel("IDW neighbors k:"))
        self._spn_idw_k_prof = QSpinBox()
        self._spn_idw_k_prof.setRange(1, 32)
        self._spn_idw_k_prof.setValue(8)
        idw_row2.addWidget(self._spn_idw_k_prof)
        idw_row2.addWidget(QLabel("  IDW power:"))
        self._spn_idw_power_prof = QDoubleSpinBox()
        self._spn_idw_power_prof.setRange(0.5, 6.0)
        self._spn_idw_power_prof.setValue(2.0)
        self._spn_idw_power_prof.setSingleStep(0.5)
        idw_row2.addWidget(self._spn_idw_power_prof)
        idw_row2.addStretch()
        prof_lay.addLayout(idw_row2)

        self._stack.addWidget(page_prof)

        left_layout.addWidget(grp_grid)

        # -- Point sampling --
        grp_sampling = QGroupBox("Point Sampling")
        sampling_form = QFormLayout(grp_sampling)
        sampling_form.setSpacing(6)

        self._cmb_sampling = QComboBox()
        self._cmb_sampling.addItem("Use all points", "all")
        self._cmb_sampling.addItem("Subsample (equal spacing)", "subsample")
        self._cmb_sampling.setToolTip(
            "Choose whether to use all imported points or to\n"
            "subsample a smaller set with uniform spatial coverage.\n\n"
            "Subsampling uses farthest-point sampling (FPS) which\n"
            "selects points that are approximately equally spaced\n"
            "on the surface, including curved surfaces."
        )
        sampling_form.addRow("Mode:", self._cmb_sampling)

        self._spn_n_samples = QSpinBox()
        self._spn_n_samples.setRange(10, 100000)
        self._spn_n_samples.setValue(500)
        self._spn_n_samples.setSingleStep(50)
        self._spn_n_samples.setToolTip(
            "Number of sample points to select from the finest grid.\n\n"
            "Points are selected using farthest-point sampling (FPS)\n"
            "to ensure uniform spatial coverage. The algorithm works\n"
            "in 3D Euclidean space, so it naturally handles curved\n"
            "surfaces (e.g. wing leading edges)."
        )
        self._lbl_n_samples = QLabel("Number of points:")
        sampling_form.addRow(self._lbl_n_samples, self._spn_n_samples)

        self._lbl_sampling_info = QLabel(
            "Points are selected from the finest grid using\n"
            "farthest-point sampling for uniform coverage."
        )
        self._lbl_sampling_info.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; font-style: italic;"
        )
        sampling_form.addRow(self._lbl_sampling_info)

        # Toggle visibility of subsample controls
        self._spn_n_samples.setVisible(False)
        self._lbl_n_samples.setVisible(False)
        self._lbl_sampling_info.setVisible(False)
        self._cmb_sampling.currentIndexChanged.connect(
            self._on_sampling_mode_changed)

        left_layout.addWidget(grp_sampling)

        # -- Analysis options --
        grp_opts = QGroupBox("Analysis Options")
        opts_form = QFormLayout(grp_opts)
        self._chk_incl_osc = QCheckBox("Include oscillatory points in statistics")
        self._chk_incl_osc.setChecked(True)
        opts_form.addRow(self._chk_incl_osc)
        self._spn_min_thresh = QDoubleSpinBox()
        self._spn_min_thresh.setRange(0.0, 1e12)
        self._spn_min_thresh.setValue(0.0)
        self._spn_min_thresh.setDecimals(6)
        self._spn_min_thresh.setSpecialValueText("Off")
        opts_form.addRow("Min |solution| threshold:", self._spn_min_thresh)
        left_layout.addWidget(grp_opts)

        # -- Compute button --
        self._btn_compute_spatial = QPushButton("Compute Spatial GCI")
        self._btn_compute_spatial.setStyleSheet(
            f"QPushButton {{ background-color: {DARK_COLORS['accent']}; "
            f"color: {DARK_COLORS['bg']}; font-weight: bold; "
            f"font-size: 14px; padding: 10px; }}"
            f"QPushButton:hover {{ background-color: {DARK_COLORS['accent_hover']}; }}"
        )
        left_layout.addWidget(self._btn_compute_spatial)

        # -- Status / guidance --
        self._guidance_import = GuidancePanel("Import Status")
        left_layout.addWidget(self._guidance_import)

        self._guidance_spatial = GuidancePanel("Spatial Summary")
        left_layout.addWidget(self._guidance_spatial)

        left_layout.addStretch()

        # ---- RIGHT PANEL: Results (sub-tabs) ----
        self._right_tabs = QTabWidget()
        splitter.addWidget(self._right_tabs)

        # Sub-tab 1: Results text
        results_widget = QWidget()
        results_lay = QVBoxLayout(results_widget)
        results_lay.setContentsMargins(6, 6, 6, 6)
        self._results_text = QPlainTextEdit()
        self._results_text.setReadOnly(True)
        ff = QFont("Consolas", 9)
        ff.setStyleHint(QFont.StyleHint.Monospace)
        self._results_text.setFont(ff)
        results_lay.addWidget(self._results_text)
        self._right_tabs.addTab(results_widget, "Results")

        # Sub-tab 2: Statistics table
        stats_widget = QWidget()
        stats_lay = QVBoxLayout(stats_widget)
        stats_lay.setContentsMargins(6, 6, 6, 6)
        self._stats_table = QTableWidget()
        self._stats_table.setAlternatingRowColors(True)
        stats_lay.addWidget(self._stats_table)
        self._right_tabs.addTab(stats_widget, "Statistics")

        # Sub-tab 3: Plots (nested tab widget)
        plots_widget = QWidget()
        plots_outer_lay = QVBoxLayout(plots_widget)
        plots_outer_lay.setContentsMargins(6, 6, 6, 6)
        self._plot_tabs = QTabWidget()

        # Helper: create a plot tab with toolbar + copy button
        def _make_plot_tab(fig, label):
            canvas = FigureCanvas(fig)
            toolbar = NavigationToolbar(canvas, self)
            btn_copy = QPushButton("Copy to Clipboard")
            btn_copy.clicked.connect(lambda _f=fig: copy_figure_to_clipboard(_f))
            btn_rq = QPushButton("Copy Report-Quality")
            btn_rq.setToolTip("Copy at 300 DPI with light report theme.")
            btn_rq.clicked.connect(lambda _f=fig: copy_report_quality_figure(_f))
            btn_export = QPushButton("Export Figure Package...")
            btn_export.setToolTip("Export PNG (300+600 DPI), SVG, PDF, and JSON.")
            btn_export.clicked.connect(lambda _f=fig: _export_figure_package_dialog(_f, self))
            widget = QWidget()
            lay = QVBoxLayout(widget)
            tb_row = QHBoxLayout()
            tb_row.addWidget(toolbar)
            tb_row.addWidget(btn_copy)
            tb_row.addWidget(btn_rq)
            tb_row.addWidget(btn_export)
            lay.addLayout(tb_row)
            lay.addWidget(canvas)
            self._plot_tabs.addTab(widget, label)
            return canvas

        # Histogram tab
        self._fig_hist = Figure(figsize=(6, 4))
        self._canvas_hist = _make_plot_tab(self._fig_hist, "Histogram")

        # CDF tab
        self._fig_cdf = Figure(figsize=(6, 4))
        self._canvas_cdf = _make_plot_tab(self._fig_cdf, "CDF")

        # Convergence map tab
        self._fig_conv = Figure(figsize=(6, 4))
        self._canvas_conv = _make_plot_tab(self._fig_conv, "Convergence Map")

        # Spatial u_num map tab
        self._fig_umap = Figure(figsize=(6, 4))
        self._canvas_umap = _make_plot_tab(self._fig_umap, "Spatial u_num Map")

        plots_outer_lay.addWidget(self._plot_tabs)
        self._right_tabs.addTab(plots_widget, "Plots")

        # Sub-tab 4: Report statements
        report_widget = QWidget()
        report_lay = QVBoxLayout(report_widget)
        report_lay.setContentsMargins(6, 6, 6, 6)
        btn_row = QHBoxLayout()
        btn_copy_report = QPushButton("Copy Statements to Clipboard")
        btn_copy_report.clicked.connect(self._copy_spatial_report)
        btn_export_report = QPushButton("Export Statements to File...")
        btn_export_report.clicked.connect(self._export_spatial_report)
        btn_row.addWidget(btn_copy_report)
        btn_row.addWidget(btn_export_report)
        report_lay.addLayout(btn_row)
        self._report_text = QPlainTextEdit()
        self._report_text.setReadOnly(True)
        ff = QFont("Consolas", 9)
        ff.setStyleHint(QFont.StyleHint.Monospace)
        self._report_text.setFont(ff)
        report_lay.addWidget(self._report_text)
        self._right_tabs.addTab(report_widget, "Report Statements")

        # Sub-tab 5: 3D Point Cloud Viewer
        self._setup_3d_viewer()

        # Sub-tab 6: Carry-Over Summary
        spatial_carry_widget = QWidget()
        spatial_carry_lay = QVBoxLayout(spatial_carry_widget)
        spatial_carry_lay.setContentsMargins(6, 6, 6, 6)

        sc_header = QLabel(
            "<b>Carry-Over Summary</b> — Recommended u_num for "
            "the Uncertainty Aggregator")
        sc_header.setWordWrap(True)
        sc_header.setStyleSheet(
            f"color: {DARK_COLORS['accent']}; font-size: 13px; "
            f"padding: 4px 0;")
        spatial_carry_lay.addWidget(sc_header)

        self._spatial_carry_table = QTableWidget()
        self._spatial_carry_table.setColumnCount(8)
        self._spatial_carry_table.setHorizontalHeaderLabels([
            "Statistic", "u_num", "Unit", "Distribution",
            "DOF", "Sigma Basis", "Points", "Status"
        ])
        self._spatial_carry_table.setAlternatingRowColors(True)
        style_table(self._spatial_carry_table)
        spatial_carry_lay.addWidget(self._spatial_carry_table)

        self._spatial_carry_info = QPlainTextEdit()
        self._spatial_carry_info.setReadOnly(True)
        self._spatial_carry_info.setMaximumHeight(120)
        self._spatial_carry_info.setPlaceholderText(
            "Convergence breakdown and recommendations appear here.")
        ff = QFont("Consolas", 9)
        ff.setStyleHint(QFont.StyleHint.Monospace)
        self._spatial_carry_info.setFont(ff)
        spatial_carry_lay.addWidget(self._spatial_carry_info)

        sc_btn_row = QHBoxLayout()
        self._btn_copy_spatial_carry = QPushButton(
            "\U0001f4cb Copy Carry Value")
        self._btn_copy_spatial_carry.setEnabled(False)
        self._btn_copy_spatial_carry.clicked.connect(
            self._copy_spatial_carry_value)
        sc_btn_row.addWidget(self._btn_copy_spatial_carry)
        sc_btn_row.addStretch()
        spatial_carry_lay.addLayout(sc_btn_row)

        self._right_tabs.addTab(spatial_carry_widget, "Carry-Over Summary")

        # Splitter proportions
        splitter.setSizes([450, 600])

        # ---- Connections ----
        self._cmb_mode.currentIndexChanged.connect(
            lambda idx: self._stack.setCurrentIndex(idx))
        self._btn_compute_spatial.clicked.connect(self._compute_spatial)

        # Mark results stale when settings change
        self._cmb_dim_spatial.currentIndexChanged.connect(
            self._mark_spatial_stale)
        self._cmb_order_preset_spatial.currentIndexChanged.connect(
            self._on_order_preset_spatial_changed)
        self._spn_p_theo.valueChanged.connect(
            self._on_p_theo_manual_change)
        self._spn_p_theo.valueChanged.connect(self._mark_spatial_stale)
        self._spn_fs_spatial.valueChanged.connect(self._mark_spatial_stale)
        self._chk_incl_osc.stateChanged.connect(self._mark_spatial_stale)
        self._spn_min_thresh.valueChanged.connect(self._mark_spatial_stale)

        # Internal data storage
        self._spatial_stale = False
        self._pre_coords = None
        self._pre_solutions = None
        self._pre_grid_names = None
        self._sep_data = {}    # grid_idx -> (coords, values, filepath)
        self._prof_data = {}   # grid_idx -> (coords, values, filepath)

    # ------------------------------------------------------------------
    # SCHEME ORDER PRESET (Spatial)
    # ------------------------------------------------------------------

    def _on_order_preset_spatial_changed(self, index):
        val = self._cmb_order_preset_spatial.currentData()
        if val is not None:
            self._spn_p_theo.blockSignals(True)
            self._spn_p_theo.setValue(val)
            self._spn_p_theo.blockSignals(False)
            self._mark_spatial_stale()

    def _on_p_theo_manual_change(self, value):
        preset_val = self._cmb_order_preset_spatial.currentData()
        if preset_val is not None and abs(preset_val - value) > 0.01:
            for i in range(self._cmb_order_preset_spatial.count()):
                if self._cmb_order_preset_spatial.itemData(i) is None:
                    self._cmb_order_preset_spatial.blockSignals(True)
                    self._cmb_order_preset_spatial.setCurrentIndex(i)
                    self._cmb_order_preset_spatial.blockSignals(False)
                    break

    # ------------------------------------------------------------------
    # UNIT COMBO HELPER
    # ------------------------------------------------------------------

    def _populate_spatial_unit_combo(self, category):
        """Populate the spatial tab unit combo based on category."""
        self._cmb_unit_val.blockSignals(True)
        self._cmb_unit_val.clear()
        presets = UNIT_PRESETS.get(category, [])
        for u in presets:
            self._cmb_unit_val.addItem(u)
        if presets:
            self._cmb_unit_val.setCurrentIndex(0)
        self._cmb_unit_val.blockSignals(False)

    # ------------------------------------------------------------------
    # POINT SAMPLING
    # ------------------------------------------------------------------

    def _on_sampling_mode_changed(self, idx):
        """Show/hide subsample controls based on sampling mode."""
        is_sub = self._cmb_sampling.currentData() == "subsample"
        self._spn_n_samples.setVisible(is_sub)
        self._lbl_n_samples.setVisible(is_sub)
        self._lbl_sampling_info.setVisible(is_sub)

    def _apply_subsampling(self, coords, solutions_per_grid):
        """Apply farthest-point sampling if enabled.

        Args:
            coords: (N, d) coordinates
            solutions_per_grid: list of (N,) arrays, one per grid

        Returns:
            (coords_sub, solutions_sub) — subsampled or original
        """
        if self._cmb_sampling.currentData() != "subsample":
            return coords, solutions_per_grid

        n_requested = self._spn_n_samples.value()
        if n_requested >= len(coords):
            return coords, solutions_per_grid

        indices = farthest_point_sampling(coords, n_requested)
        coords_sub = coords[indices]
        solutions_sub = [sol[indices] for sol in solutions_per_grid]
        return coords_sub, solutions_sub

    # ------------------------------------------------------------------
    # FILE LOADING — Pre-interpolated CSV
    # ------------------------------------------------------------------

    def _load_pre_interpolated(self):
        """Load a pre-interpolated CSV file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Pre-Interpolated CSV", "",
            "CSV files (*.csv);;Text files (*.txt);;All files (*.*)")
        if not path:
            return

        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                coords, solutions, names = parse_pre_interpolated_csv(path)
            if caught:
                warn_msgs = "\n".join(str(w.message) for w in caught)
                QMessageBox.warning(self, "CSV Parse Warning", warn_msgs)
        except Exception as exc:
            QMessageBox.critical(self, "Parse Error", str(exc))
            self._guidance_import.set_guidance(
                f"Failed to load: {exc}", 'red')
            return

        self._pre_coords = coords
        self._pre_solutions = solutions
        self._pre_grid_names = names
        n_grids = len(solutions)
        n_points = len(solutions[0])
        n_dim = coords.shape[1]

        self._lbl_pre_file.setText(os.path.basename(path))
        self._lbl_pre_file.setStyleSheet(
            f"color: {DARK_COLORS['green']}; font-weight: bold;")
        self._lbl_pre_info.setText(
            f"Loaded {n_points:,} points, {n_grids} grids, {n_dim}D coordinates")

        # Populate cell count table
        self._tbl_pre_cells.setRowCount(n_grids)
        for i in range(n_grids):
            name_item = QTableWidgetItem(
                names[i] if i < len(names) else f"Grid {i+1}")
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self._tbl_pre_cells.setItem(i, 0, name_item)
            cell_item = QTableWidgetItem("")
            cell_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._tbl_pre_cells.setItem(i, 1, cell_item)

        self._guidance_import.set_guidance(
            f"Loaded {n_points:,} points x {n_grids} grids from CSV. "
            f"Enter cell counts below, then click Compute.",
            'green')

    # ------------------------------------------------------------------
    # FILE LOADING — Separate CSV files
    # ------------------------------------------------------------------

    def _browse_sep_files(self):
        """Browse and load separate CSV files for each grid."""
        n_grids = self._tbl_sep_files.rowCount()
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select CSV Files (finest first)", "",
            "CSV files (*.csv);;Text files (*.txt);;All files (*.*)")
        if not paths:
            return

        # Adjust table row count if needed
        if len(paths) > n_grids:
            self._tbl_sep_files.setRowCount(len(paths))
            for i in range(n_grids, len(paths)):
                self._tbl_sep_files.setVerticalHeaderItem(
                    i, QTableWidgetItem(f"Grid {i+1}"))

        for i, path in enumerate(paths):
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    coords, values = parse_csv_field(path)
                if caught:
                    warn_msgs = "\n".join(str(w.message) for w in caught)
                    QMessageBox.warning(
                        self, f"CSV Parse Warning (Grid {i+1})", warn_msgs)
                self._sep_data[i] = (coords, values, path)
                self._tbl_sep_files.setItem(
                    i, 0, QTableWidgetItem(os.path.basename(path)))
                self._tbl_sep_files.setItem(
                    i, 2, QTableWidgetItem(f"{len(values):,}"))
            except Exception as exc:
                QMessageBox.warning(
                    self, f"Error Loading Grid {i+1}", str(exc))

        self._guidance_import.set_guidance(
            f"Loaded {len(paths)} grid files. Enter cell counts, then Compute.",
            'green')

    def _add_sep_grid(self):
        n = self._tbl_sep_files.rowCount()
        if n >= 6:
            return
        self._tbl_sep_files.setRowCount(n + 1)
        self._tbl_sep_files.setVerticalHeaderItem(
            n, QTableWidgetItem(f"Grid {n+1}"))

    def _rem_sep_grid(self):
        n = self._tbl_sep_files.rowCount()
        if n <= 2:
            return
        self._tbl_sep_files.setRowCount(n - 1)
        self._sep_data.pop(n - 1, None)

    # ------------------------------------------------------------------
    # FILE LOADING — Separate Fluent .prof files
    # ------------------------------------------------------------------

    def _browse_prof_files(self):
        """Browse and load Fluent .prof files for each grid."""
        n_grids = self._tbl_prof_files.rowCount()
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select .prof Files (finest first)", "",
            "Fluent profile files (*.prof);;All files (*.*)")
        if not paths:
            return

        if len(paths) > n_grids:
            self._tbl_prof_files.setRowCount(len(paths))
            for i in range(n_grids, len(paths)):
                self._tbl_prof_files.setVerticalHeaderItem(
                    i, QTableWidgetItem(f"Grid {i+1}"))

        for i, path in enumerate(paths):
            try:
                coords, values = parse_fluent_prof(path)
                self._prof_data[i] = (coords, values, path)
                self._tbl_prof_files.setItem(
                    i, 0, QTableWidgetItem(os.path.basename(path)))
                self._tbl_prof_files.setItem(
                    i, 2, QTableWidgetItem(f"{len(values):,}"))
            except Exception as exc:
                QMessageBox.warning(
                    self, f"Error Loading Grid {i+1}", str(exc))

        self._guidance_import.set_guidance(
            f"Loaded {len(paths)} .prof files. Enter cell counts, then Compute.",
            'green')

    def _add_prof_grid(self):
        n = self._tbl_prof_files.rowCount()
        if n >= 6:
            return
        self._tbl_prof_files.setRowCount(n + 1)
        self._tbl_prof_files.setVerticalHeaderItem(
            n, QTableWidgetItem(f"Grid {n+1}"))

    def _rem_prof_grid(self):
        n = self._tbl_prof_files.rowCount()
        if n <= 2:
            return
        self._tbl_prof_files.setRowCount(n - 1)
        self._prof_data.pop(n - 1, None)

    # ------------------------------------------------------------------
    # COMPUTE
    # ------------------------------------------------------------------

    def _mark_spatial_stale(self):
        """Visually indicate that displayed spatial results are out of date."""
        if not hasattr(self, '_summary') or self._summary is None:
            return
        if not self._spatial_stale:
            self._spatial_stale = True
            self._btn_compute_spatial.setText("⟳ Recompute Spatial GCI")
            self._btn_compute_spatial.setToolTip(
                "Settings have changed since the last computation."
            )

    def _compute_spatial(self):
        """Run the spatial GCI analysis."""
        mode = self._cmb_mode.currentData()
        dim = self._cmb_dim_spatial.currentData()
        p_theo = self._spn_p_theo.value()
        fs_val = self._spn_fs_spatial.value()
        fs = None if fs_val == 0.0 else fs_val
        qty_name = self._edt_qty_name.text().strip() or "Field"
        qty_unit = self._cmb_unit_val.currentText().strip()
        include_osc = self._chk_incl_osc.isChecked()
        min_thresh = self._spn_min_thresh.value()

        self._btn_compute_spatial.setEnabled(False)
        self._guidance_spatial.set_guidance("Computing spatial GCI…", 'yellow')
        QApplication.processEvents()

        try:
            if mode == "pre_interp":
                result = self._compute_pre_interp(
                    dim, fs, p_theo, qty_name, qty_unit,
                    include_osc, min_thresh)
            elif mode == "sep_csv":
                result = self._compute_separate(
                    self._sep_data, self._tbl_sep_files,
                    dim, fs, p_theo, qty_name, qty_unit,
                    include_osc, min_thresh,
                    idw_k=self._spn_idw_k.value(),
                    idw_power=self._spn_idw_power.value())
            elif mode == "sep_prof":
                result = self._compute_separate(
                    self._prof_data, self._tbl_prof_files,
                    dim, fs, p_theo, qty_name, qty_unit,
                    include_osc, min_thresh,
                    idw_k=self._spn_idw_k_prof.value(),
                    idw_power=self._spn_idw_power_prof.value())
            else:
                return

            self._summary = result
            self._display_spatial_results()
            self._update_spatial_plots()
            self._update_3d_viewer()
            self._generate_spatial_report_statements()
            self._update_spatial_carry_over()
            self._spatial_stale = False
            self._btn_compute_spatial.setText("Compute Spatial GCI")
            self._btn_compute_spatial.setToolTip("")

        except Exception as exc:
            QMessageBox.critical(self, "Computation Error", str(exc))
            self._guidance_spatial.set_guidance(
                f"Error: {exc}", 'red')
        finally:
            self._btn_compute_spatial.setEnabled(True)

    def _compute_pre_interp(self, dim, fs, p_theo, qty_name, qty_unit,
                             include_osc, min_thresh):
        """Compute spatial GCI from pre-interpolated data."""
        if self._pre_solutions is None:
            raise ValueError("No pre-interpolated CSV loaded.")

        n_grids = len(self._pre_solutions)
        cell_counts = []
        for i in range(n_grids):
            item = self._tbl_pre_cells.item(i, 1)
            if item is None or not item.text().strip():
                raise ValueError(f"Missing cell count for Grid {i+1}")
            try:
                cell_counts.append(float(item.text().replace(',', '')))
            except ValueError:
                raise ValueError(
                    f"Invalid cell count for Grid {i+1}: '{item.text()}'")

        # Validate grid ordering (finest first, descending cell counts)
        solutions = self._pre_solutions
        coords = self._pre_coords
        if not all(cell_counts[i] >= cell_counts[i+1]
                   for i in range(len(cell_counts) - 1)):
            reply = QMessageBox.question(
                self, "Grid Ordering",
                "Cell counts should be in descending order (finest first, "
                "coarsest last). The data appears to be in a different "
                "order.\n\nWould you like to auto-sort the data?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                idx = sorted(range(len(cell_counts)),
                             key=lambda i: cell_counts[i], reverse=True)
                cell_counts = [cell_counts[i] for i in idx]
                solutions = [solutions[i] for i in idx]
                # Write sorted cell counts back to the table UI
                for row in range(len(cell_counts)):
                    self._tbl_pre_cells.item(row, 1).setText(
                        f"{cell_counts[row]:.6g}")
            else:
                raise ValueError("Computation cancelled — fix grid ordering.")

        # Validate: all cell counts must be positive
        for i, cc in enumerate(cell_counts):
            if cc <= 0:
                raise ValueError(
                    f"Grid {i+1} has a non-positive cell count ({cc:.6g}). "
                    f"All cell counts must be greater than zero.")

        # Apply point subsampling if enabled
        coords, solutions = self._apply_subsampling(coords, solutions)

        return compute_spatial_gci(
            solutions_per_grid=solutions,
            cell_counts=cell_counts,
            coordinates=coords,
            dim=dim,
            safety_factor=fs,
            theoretical_order=p_theo,
            quantity_name=qty_name,
            quantity_unit=qty_unit,
            include_oscillatory=include_osc,
            min_solution_threshold=min_thresh,
        )

    def _compute_separate(self, data_dict, table, dim, fs, p_theo,
                           qty_name, qty_unit, include_osc, min_thresh,
                           idw_k=8, idw_power=2.0):
        """Compute spatial GCI from separate grid files (CSV or .prof)."""
        n_grids = table.rowCount()

        # Validate data loaded for all grids
        loaded_grids = []
        cell_counts = []
        for i in range(n_grids):
            if i not in data_dict:
                raise ValueError(f"No file loaded for Grid {i+1}")
            cell_item = table.item(i, 1)
            if cell_item is None or not cell_item.text().strip():
                raise ValueError(f"Missing cell count for Grid {i+1}")
            try:
                cell_counts.append(float(cell_item.text().replace(',', '')))
            except ValueError:
                raise ValueError(
                    f"Invalid cell count for Grid {i+1}: "
                    f"'{cell_item.text()}'")
            loaded_grids.append(data_dict[i])

        # Validate grid ordering (finest first, descending cell counts)
        if not all(cell_counts[i] >= cell_counts[i+1]
                   for i in range(len(cell_counts) - 1)):
            reply = QMessageBox.question(
                self, "Grid Ordering",
                "Cell counts should be in descending order (finest first, "
                "coarsest last). The data appears to be in a different "
                "order.\n\nWould you like to auto-sort the data?",
                QMessageBox.Yes | QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                idx = sorted(range(len(cell_counts)),
                             key=lambda i: cell_counts[i], reverse=True)
                cell_counts = [cell_counts[i] for i in idx]
                loaded_grids = [loaded_grids[i] for i in idx]
                # Write sorted cell counts back to the table UI
                for row in range(len(cell_counts)):
                    table.item(row, 1).setText(f"{cell_counts[row]:.6g}")
            else:
                raise ValueError("Computation cancelled — fix grid ordering.")

        # Validate: all cell counts must be positive
        for i, cc in enumerate(cell_counts):
            if cc <= 0:
                raise ValueError(
                    f"Grid {i+1} has a non-positive cell count ({cc:.6g}). "
                    f"All cell counts must be greater than zero.")

        # Interpolate all grids onto the finest grid's coordinates
        finest_coords = loaded_grids[0][0]
        solutions_at_common = [loaded_grids[0][1]]  # finest = no interp

        k = idw_k
        power = idw_power

        for g in range(1, n_grids):
            src_coords, src_values, _ = loaded_grids[g]
            interp_vals = interpolate_field_idw(
                src_coords, src_values, finest_coords,
                k=k, power=power)
            solutions_at_common.append(interp_vals)

        # Apply point subsampling if enabled
        finest_coords, solutions_at_common = self._apply_subsampling(
            finest_coords, solutions_at_common)

        return compute_spatial_gci(
            solutions_per_grid=solutions_at_common,
            cell_counts=cell_counts,
            coordinates=finest_coords,
            dim=dim,
            safety_factor=fs,
            theoretical_order=p_theo,
            quantity_name=qty_name,
            quantity_unit=qty_unit,
            include_oscillatory=include_osc,
            min_solution_threshold=min_thresh,
        )

    # ------------------------------------------------------------------
    # DISPLAY RESULTS
    # ------------------------------------------------------------------

    def _display_spatial_results(self):
        """Populate results text, stats table, and guidance panels."""
        s = self._summary
        if s is None:
            return

        unit_sfx = f" {s.quantity_unit}" if s.quantity_unit else ""
        q_label = (f"{s.quantity_name} ({s.quantity_unit})"
                   if s.quantity_unit else s.quantity_name)

        lines = []
        lines.append("=" * 65)
        lines.append("  SPATIAL GCI ANALYSIS RESULTS")
        lines.append("=" * 65)
        lines.append("")
        lines.append(f"  Quantity:      {q_label}")
        lines.append(f"  Total points:  {s.n_points_total:,}")
        lines.append(f"  Valid points:  {s.n_points_valid:,}")
        lines.append("")

        # Convergence breakdown
        lines.append("  Convergence Breakdown:")
        lines.append(f"    Monotonic:   {s.n_monotonic:>6,} "
                     f"({100*s.n_monotonic/max(s.n_points_total,1):.1f}%)")
        lines.append(f"    Oscillatory: {s.n_oscillatory:>6,} "
                     f"({100*s.n_oscillatory/max(s.n_points_total,1):.1f}%)")
        lines.append(f"    Divergent:   {s.n_divergent:>6,} "
                     f"({100*s.n_divergent/max(s.n_points_total,1):.1f}%)")
        if s.n_grid_independent > 0:
            lines.append(f"    Grid-indep.: {s.n_grid_independent:>6,} "
                         f"({100*s.n_grid_independent/max(s.n_points_total,1):.1f}%)")
        skipped = (s.n_points_total - s.n_monotonic - s.n_oscillatory
                   - s.n_divergent - s.n_grid_independent)
        if skipped > 0:
            lines.append(f"    Skipped:     {skipped:>6,}")
        lines.append("")

        # Divergent point spatial reporting (when fraction > 10%)
        div_frac = s.n_divergent / max(s.n_points_total, 1)
        if s.n_divergent > 0 and div_frac > 0.10:
            div_mask = np.array(
                [ct == "divergent" for ct in s.point_convergence_type])
            if s.point_coords is not None and np.any(div_mask):
                div_coords = s.point_coords[div_mask]
                lines.append(
                    f"  \u26A0 WARNING: {div_frac*100:.0f}% of points are "
                    f"divergent ({s.n_divergent:,} points)")
                # Bounding box of divergent points
                dim_labels = ["x", "y", "z"]
                for d in range(div_coords.shape[1]):
                    lbl = dim_labels[d] if d < 3 else f"dim{d}"
                    lo, hi = div_coords[:, d].min(), div_coords[:, d].max()
                    lines.append(
                        f"    Divergent {lbl}-range: "
                        f"[{lo:.4g}, {hi:.4g}]")
                # Mean |R| of divergent points
                div_R_vals = []
                for i, ct in enumerate(s.point_convergence_type):
                    if ct == "divergent" and i < len(s.point_results):
                        r_val = s.point_results[i].convergence_ratio
                        if not np.isnan(r_val) and not np.isinf(r_val):
                            div_R_vals.append(abs(r_val))
                if div_R_vals:
                    lines.append(
                        f"    Mean |R| of divergent points: "
                        f"{np.mean(div_R_vals):.3f}")
                lines.append(
                    "    Check mesh quality in the divergent region.")
                if div_frac > 0.25:
                    lines.append("")
                    lines.append(
                        "    CRITICAL: More than 25% of points are "
                        "divergent. This indicates a systemic issue "
                        "with the grid refinement strategy. Common "
                        "causes: non-conformal mesh interfaces, "
                        "different mesh topology between grids, or "
                        "interpolation artifacts in data transfer.")
                elif div_frac > 0.10:
                    lines.append("")
                    lines.append(
                        "    NOTE: Divergent points may be concentrated "
                        "near geometry features (leading/trailing "
                        "edges, junctions) where mesh refinement is "
                        "non-uniform. Use the 3D Point Cloud viewer "
                        "to visualize the spatial distribution.")
                lines.append("")

        if s.n_points_valid > 0:
            # Distribution statistics
            lines.append("  u_num Distribution Statistics:")
            lines.append(f"    Mean:          {s.u_num_mean:.6g}{unit_sfx}")
            lines.append(f"    Median:        {s.u_num_median:.6g}{unit_sfx}")
            lines.append(
                f"    95th pctile:   {s.u_num_p95:.6g}{unit_sfx}  "
                f"\u25c0 RECOMMENDED")
            lines.append(f"    Maximum:       {s.u_num_max:.6g}{unit_sfx}")
            lines.append(f"    RMS:           {s.u_num_rms:.6g}{unit_sfx}")
            lines.append(f"    Std Dev:       {s.u_num_std:.6g}{unit_sfx}")
            lines.append("")

            if s.p_mean > 0:
                lines.append("  Observed Order Statistics:")
                lines.append(f"    p mean:   {s.p_mean:.3f}")
                lines.append(f"    p median: {s.p_median:.3f}")
                lines.append("")

            # Carry-over box
            lines.append("  " + "\u2550" * 61)
            lines.append(
                "  \u2551  \u279C  RECOMMENDED u_num FOR V&V 20 "
                "UNCERTAINTY BUDGET:    \u2551")
            lines.append("  " + "\u2550" * 61)
            lines.append(
                f"  \u2551                                                "
                f"             \u2551")
            val_line = f"u_num (95th pctile) = {s.u_num_p95:.6g}{unit_sfx}"
            lines.append(f"  \u2551    {val_line:<57s}\u2551")
            lines.append(
                f"  \u2551                                                "
                f"             \u2551")
            basis_line = (
                f"Based on {s.n_points_valid:,} valid points  |  "
                f"Basis = Confirmed 1\u03c3  |  DOF = \u221e")
            lines.append(f"  \u2551    {basis_line:<57s}\u2551")
            rationale = (
                "Rationale: 95th pctile is conservative but not "
                "dominated by outliers")
            lines.append(f"  \u2551    {rationale:<57s}\u2551")
            lines.append("  " + "\u2550" * 61)
            lines.append("")
        else:
            lines.append("  \u26A0 No valid points for statistics.")
            lines.append("  All points are divergent or skipped.")
            lines.append("")

        lines.append("References:")
        lines.append("  Celik et al. (2008) J. Fluids Eng. 130(7), 078001")
        lines.append(
            "  Eca & Hoekstra (2014a) J. Comp. Physics 262, 104-130 (LSR)")
        lines.append(
            "  Eca & Hoekstra (2014b) Int. J. Numer. Meth. Fluids 75 "
            "(spatial GCI)")
        lines.append("  Roache (1998) Hermosa Publishers, Ch. 5")
        lines.append("  ASME V&V 20-2009 (R2021)")

        self._results_text.setPlainText("\n".join(lines))

        # ---- Stats table ----
        stats = [
            ("Mean", f"{s.u_num_mean:.6g}{unit_sfx}"),
            ("Median", f"{s.u_num_median:.6g}{unit_sfx}"),
            ("95th Percentile", f"{s.u_num_p95:.6g}{unit_sfx}"),
            ("Maximum", f"{s.u_num_max:.6g}{unit_sfx}"),
            ("RMS", f"{s.u_num_rms:.6g}{unit_sfx}"),
            ("Std Dev", f"{s.u_num_std:.6g}{unit_sfx}"),
        ]
        self._stats_table.setColumnCount(2)
        self._stats_table.setHorizontalHeaderLabels(["Statistic", "Value"])
        self._stats_table.setRowCount(len(stats))
        for i, (label, val) in enumerate(stats):
            lbl_item = QTableWidgetItem(label)
            val_item = QTableWidgetItem(val)
            val_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            # Highlight the 95th percentile row
            if i == 2:
                lbl_item.setForeground(QColor(DARK_COLORS['accent']))
                val_item.setForeground(QColor(DARK_COLORS['accent']))
                f = lbl_item.font()
                f.setBold(True)
                lbl_item.setFont(f)
                val_item.setFont(f)
            self._stats_table.setItem(i, 0, lbl_item)
            self._stats_table.setItem(i, 1, val_item)
        style_table(self._stats_table)

        # ---- Guidance panels ----
        if s.n_points_valid > 0:
            pct_mono = 100 * s.n_monotonic / max(s.n_points_total, 1)
            if pct_mono >= 80:
                color = 'green'
                msg = (f"{s.n_points_valid:,} valid points analyzed. "
                       f"{pct_mono:.0f}% monotonic convergence. "
                       f"Recommended u_num = {s.u_num_p95:.6g}{unit_sfx} "
                       f"(95th percentile).")
            elif pct_mono >= 50:
                color = 'yellow'
                msg = (f"{s.n_points_valid:,} valid points. "
                       f"Only {pct_mono:.0f}% monotonic. "
                       f"Consider checking mesh topology. "
                       f"u_num (95th pctile) = {s.u_num_p95:.6g}{unit_sfx}.")
            else:
                color = 'red'
                msg = (f"Only {pct_mono:.0f}% monotonic convergence. "
                       f"Results may be unreliable. "
                       f"Check mesh quality and solver convergence.")
            self._guidance_spatial.set_guidance(msg, color)
        else:
            self._guidance_spatial.set_guidance(
                "No valid points. Cannot compute statistics.", 'red')

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------

    def _update_spatial_plots(self):
        """Draw all four spatial GCI plots."""
        s = self._summary
        if s is None:
            return

        self._plot_histogram(s)
        self._plot_cdf(s)
        self._plot_convergence_map(s)
        self._plot_unum_map(s)

    def _plot_histogram(self, s):
        """Plot 1: u_num distribution histogram."""
        self._fig_hist.clear()
        ax = self._fig_hist.add_subplot(111)

        valid = s.point_u_num[~np.isnan(s.point_u_num)]
        if len(valid) == 0:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center',
                    transform=ax.transAxes, color=DARK_COLORS['fg_dim'])
            self._canvas_hist.draw()
            return

        n_bins = min(50, max(10, len(valid) // 10))
        ax.hist(valid, bins=n_bins, color=DARK_COLORS['accent'],
                alpha=0.7, edgecolor=DARK_COLORS['border'])

        # Vertical lines for key statistics
        ax.axvline(s.u_num_mean, color=DARK_COLORS['green'],
                   linestyle='-', linewidth=2, alpha=0.8,
                   label=f'Mean = {s.u_num_mean:.4g}')
        ax.axvline(s.u_num_median, color=DARK_COLORS['yellow'],
                   linestyle='--', linewidth=2, alpha=0.8,
                   label=f'Median = {s.u_num_median:.4g}')
        ax.axvline(s.u_num_p95, color=DARK_COLORS['red'],
                   linestyle='-', linewidth=2.5, alpha=0.9,
                   label=f'95th pctile = {s.u_num_p95:.4g}')
        ax.axvline(s.u_num_max, color=DARK_COLORS['orange'],
                   linestyle=':', linewidth=1.5, alpha=0.7,
                   label=f'Max = {s.u_num_max:.4g}')

        unit_sfx = f" ({s.quantity_unit})" if s.quantity_unit else ""
        ax.set_xlabel(f"u_num{unit_sfx}")
        ax.set_ylabel("Count")
        ax.set_title(f"Numerical Uncertainty Distribution \u2014 {s.quantity_name}")
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)

        self._fig_hist.tight_layout()
        self._canvas_hist.draw()

    def _plot_cdf(self, s):
        """Plot 2: Cumulative distribution function of u_num."""
        self._fig_cdf.clear()
        ax = self._fig_cdf.add_subplot(111)

        valid = s.point_u_num[~np.isnan(s.point_u_num)]
        if len(valid) == 0:
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center',
                    transform=ax.transAxes, color=DARK_COLORS['fg_dim'])
            self._canvas_cdf.draw()
            return

        sorted_u = np.sort(valid)
        cdf = np.arange(1, len(sorted_u) + 1) / len(sorted_u)

        ax.plot(sorted_u, cdf * 100, '-', color=DARK_COLORS['accent'],
                linewidth=2, label='CDF')

        # 95th percentile marker
        ax.axhline(95, color=DARK_COLORS['red'], linestyle='--',
                   linewidth=1.5, alpha=0.7)
        ax.axvline(s.u_num_p95, color=DARK_COLORS['red'],
                   linestyle='--', linewidth=1.5, alpha=0.7)
        ax.plot(s.u_num_p95, 95, 'o', color=DARK_COLORS['red'],
                markersize=10, zorder=5,
                label=f'95th pctile = {s.u_num_p95:.4g}')

        unit_sfx = f" ({s.quantity_unit})" if s.quantity_unit else ""
        ax.set_xlabel(f"u_num{unit_sfx}")
        ax.set_ylabel("Cumulative % of points")
        ax.set_title(f"CDF of Numerical Uncertainty \u2014 {s.quantity_name}")
        ax.legend(loc='best', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 102)

        self._fig_cdf.tight_layout()
        self._canvas_cdf.draw()

    def _plot_convergence_map(self, s):
        """Plot 3: Scatter colored by convergence type."""
        self._fig_conv.clear()
        ax = self._fig_conv.add_subplot(111)

        if s.point_coords is None or len(s.point_convergence_type) == 0:
            self._canvas_conv.draw()
            return

        coords = s.point_coords
        conv_types = s.point_convergence_type

        color_map = {
            'monotonic': DARK_COLORS['green'],
            'oscillatory': DARK_COLORS['yellow'],
            'divergent': DARK_COLORS['red'],
            'grid-independent': DARK_COLORS['accent'],
            'skipped': DARK_COLORS['fg_dim'],
        }

        for ctype, color in color_map.items():
            mask = [c == ctype for c in conv_types]
            if not any(mask):
                continue
            idx = np.array(mask)
            count = np.sum(idx)
            ax.scatter(coords[idx, 0], coords[idx, 1],
                      c=color, s=4, alpha=0.6,
                      label=f'{ctype} ({count:,})')

        # Highlight divergent cluster with convex hull boundary
        div_mask = np.array([c == 'divergent' for c in conv_types])
        n_div = np.sum(div_mask)
        if n_div >= 5:
            try:
                from scipy.spatial import ConvexHull
                div_pts = coords[div_mask, :2]  # use x,y
                hull = ConvexHull(div_pts)
                hull_verts = np.append(hull.vertices, hull.vertices[0])
                ax.plot(div_pts[hull_verts, 0], div_pts[hull_verts, 1],
                        '--', color=DARK_COLORS['red'], linewidth=1.5,
                        alpha=0.6, label=f'Divergent region ({n_div:,} pts)')
            except Exception:
                pass  # ConvexHull can fail for degenerate cases

        # Divergent count annotation
        if n_div > 0:
            ax.annotate(
                f"{n_div:,} divergent",
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=8, color=DARK_COLORS['red'], alpha=0.9)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Convergence Type Map \u2014 {s.quantity_name}")
        ax.legend(loc='best', fontsize=7, markerscale=4)
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')

        self._fig_conv.tight_layout()
        self._canvas_conv.draw()

    def _plot_unum_map(self, s):
        """Plot 4: Scatter colored by u_num magnitude."""
        self._fig_umap.clear()
        ax = self._fig_umap.add_subplot(111)

        if s.point_coords is None or s.point_u_num is None:
            self._canvas_umap.draw()
            return

        valid = ~np.isnan(s.point_u_num)
        if not np.any(valid):
            ax.text(0.5, 0.5, "No valid data", ha='center', va='center',
                    transform=ax.transAxes, color=DARK_COLORS['fg_dim'])
            self._canvas_umap.draw()
            return

        coords = s.point_coords[valid]
        u_vals = s.point_u_num[valid]

        sc = ax.scatter(coords[:, 0], coords[:, 1],
                       c=u_vals, s=4, alpha=0.7,
                       cmap='hot_r', vmin=0)
        cbar = self._fig_umap.colorbar(sc, ax=ax, shrink=0.8)
        unit_sfx = f" ({s.quantity_unit})" if s.quantity_unit else ""
        cbar.set_label(f"u_num{unit_sfx}", fontsize=8)

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Spatial u_num Map \u2014 {s.quantity_name}")
        ax.grid(True, alpha=0.2)
        ax.set_aspect('equal', adjustable='datalim')

        self._fig_umap.tight_layout()
        self._canvas_umap.draw()

    # ------------------------------------------------------------------
    # REPORT STATEMENTS (Spatial)
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # 3D Point Cloud Viewer (PyVista)
    # ------------------------------------------------------------------
    def _setup_3d_viewer(self):
        """Add a 3D point cloud viewer sub-tab (requires pyvista)."""
        self._pv_available = False
        self._pv_plotter = None

        viewer_widget = QWidget()
        viewer_lay = QVBoxLayout(viewer_widget)
        viewer_lay.setContentsMargins(6, 6, 6, 6)

        try:
            import pyvista as pv
            from pyvistaqt import QtInteractor
            self._pv_available = True

            self._pv_plotter = QtInteractor(viewer_widget)
            self._pv_plotter.set_background(DARK_COLORS['bg'])
            viewer_lay.addWidget(self._pv_plotter)

            # Color-by selector
            ctrl_row = QHBoxLayout()
            ctrl_row.addWidget(QLabel("Color by:"))
            self._cmb_3d_color = QComboBox()
            self._cmb_3d_color.addItems([
                "Convergence Type", "u_num Magnitude",
                "Observed Order (p)", "GCI (%)"
            ])
            self._cmb_3d_color.currentIndexChanged.connect(
                lambda: self._update_3d_viewer())
            ctrl_row.addWidget(self._cmb_3d_color)
            ctrl_row.addStretch()
            viewer_lay.insertLayout(0, ctrl_row)

        except ImportError:
            # Fallback: show install instructions
            lbl = QLabel(
                "<p style='font-size:14px; color:" + DARK_COLORS['fg_dim'] + ";'>"
                "<b>3D Point Cloud Viewer</b><br><br>"
                "This feature requires <code>pyvista</code> and "
                "<code>pyvistaqt</code>.<br>"
                "Install with:<br><br>"
                "<code>pip install pyvista pyvistaqt</code>"
                "<br><br>"
                "After installing, restart the application to enable "
                "the interactive 3D point cloud viewer.</p>"
            )
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setWordWrap(True)
            viewer_lay.addWidget(lbl)

        self._right_tabs.addTab(viewer_widget, "3D Point Cloud")

    def _update_3d_viewer(self):
        """Populate the 3D viewer with spatial GCI results."""
        if not self._pv_available or self._pv_plotter is None:
            return
        if self._summary is None:
            return

        import pyvista as pv

        self._pv_plotter.clear()

        summary = self._summary
        coords = summary.point_coords  # numpy array (N, 2 or 3)
        if coords is None or len(coords) == 0:
            return

        n_pts = len(coords)
        # Ensure 3D coords
        if coords.shape[1] == 2:
            coords_3d = np.column_stack([coords, np.zeros(n_pts)])
        else:
            coords_3d = coords

        # Build point cloud
        cloud = pv.PolyData(coords_3d.astype(np.float64))

        # Get color mode
        color_mode = self._cmb_3d_color.currentText()

        # Color map by convergence type
        CONV_COLORS = {
            "monotonic": 0,
            "oscillatory": 1,
            "grid_independent": 2,
            "divergent": 3,
        }
        CONV_CMAP_NAMES = {
            0: "Monotonic",
            1: "Oscillatory",
            2: "Grid-Independent",
            3: "Divergent",
        }

        if color_mode == "Convergence Type":
            conv_types = summary.point_convergence_type
            if conv_types is not None and len(conv_types) == n_pts:
                scalars = np.array([
                    CONV_COLORS.get(str(ct).lower(), 3) for ct in conv_types
                ], dtype=float)
                cloud["convergence"] = scalars
                self._pv_plotter.add_mesh(
                    cloud, scalars="convergence",
                    cmap=["#a6e3a1", "#89b4fa", "#f9e2af", "#f38ba8"],
                    clim=[0, 3],
                    point_size=6,
                    render_points_as_spheres=True,
                    scalar_bar_args={
                        "title": "Convergence Type",
                        "n_labels": 4,
                        "color": DARK_COLORS['fg'],
                    },
                )
            else:
                self._pv_plotter.add_mesh(
                    cloud, color="#89b4fa",
                    point_size=6,
                    render_points_as_spheres=True,
                )

        elif color_mode == "u_num Magnitude":
            u_num = summary.point_u_num
            if u_num is not None and len(u_num) == n_pts:
                cloud["u_num"] = np.array(u_num, dtype=float)
                self._pv_plotter.add_mesh(
                    cloud, scalars="u_num",
                    cmap="coolwarm",
                    point_size=6,
                    render_points_as_spheres=True,
                    scalar_bar_args={
                        "title": "u_num",
                        "color": DARK_COLORS['fg'],
                    },
                )
            else:
                self._pv_plotter.add_mesh(
                    cloud, color="#89b4fa",
                    point_size=6,
                    render_points_as_spheres=True,
                )

        elif color_mode == "Observed Order (p)":
            p_vals = [r.observed_order for r in summary.point_results] \
                if summary.point_results else None
            if p_vals is not None and len(p_vals) == n_pts:
                cloud["p_obs"] = np.array(p_vals, dtype=float)
                self._pv_plotter.add_mesh(
                    cloud, scalars="p_obs",
                    cmap="viridis",
                    point_size=6,
                    render_points_as_spheres=True,
                    scalar_bar_args={
                        "title": "Observed Order (p)",
                        "color": DARK_COLORS['fg'],
                    },
                )
            else:
                self._pv_plotter.add_mesh(
                    cloud, color="#89b4fa",
                    point_size=6,
                    render_points_as_spheres=True,
                )

        elif color_mode == "GCI (%)":
            gci = [r.gci_fine for r in summary.point_results] \
                if summary.point_results else None
            if gci is not None and len(gci) == n_pts:
                cloud["GCI"] = np.array(gci, dtype=float) * 100.0
                self._pv_plotter.add_mesh(
                    cloud, scalars="GCI",
                    cmap="hot_r",
                    point_size=6,
                    render_points_as_spheres=True,
                    scalar_bar_args={
                        "title": "GCI (%)",
                        "color": DARK_COLORS['fg'],
                    },
                )
            else:
                self._pv_plotter.add_mesh(
                    cloud, color="#89b4fa",
                    point_size=6,
                    render_points_as_spheres=True,
                )

        # Add axis labels
        self._pv_plotter.show_axes()
        self._pv_plotter.reset_camera()

    # ------------------------------------------------------------------
    # CARRY-OVER SUMMARY (Spatial)
    # ------------------------------------------------------------------

    def _update_spatial_carry_over(self):
        """Populate the Spatial Carry-Over Summary subtab."""
        s = self._summary
        if s is None or s.n_points_valid == 0:
            self._spatial_carry_table.setRowCount(0)
            self._spatial_carry_info.clear()
            self._btn_copy_spatial_carry.setEnabled(False)
            return

        unit_sfx = f" {s.quantity_unit}" if s.quantity_unit else ""
        total = max(s.n_points_total, 1)
        mono_pct = 100 * s.n_monotonic / total
        div_pct = 100 * s.n_divergent / total

        # Populate table with recommended value and alternatives
        rows = [
            ("95th Percentile (recommended)",
             f"{s.u_num_p95:.6g}{unit_sfx}",
             "RECOMMENDED", DARK_COLORS['green']),
            ("Mean",
             f"{s.u_num_mean:.6g}{unit_sfx}",
             "Alternative", None),
            ("Maximum (conservative)",
             f"{s.u_num_max:.6g}{unit_sfx}",
             "Conservative", None),
        ]
        self._spatial_carry_table.setRowCount(len(rows))
        for i, (stat, val, status, color) in enumerate(rows):
            items = [
                (stat, None),
                (val, None),
                (s.quantity_unit or "", None),
                ("Normal", None),
                ("\u221e", None),
                ("Confirmed 1\u03c3", None),
                (f"{s.n_points_valid:,}", None),
                (status, color),
            ]
            for col, (text, clr) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if clr:
                    item.setForeground(QColor(clr))
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                if i == 0:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                self._spatial_carry_table.setItem(i, col, item)

        # Info text
        info_lines = [
            f"Quantity: {s.quantity_name}",
            f"Total points: {s.n_points_total:,}  |  "
            f"Valid: {s.n_points_valid:,}",
            f"Monotonic: {mono_pct:.1f}%  |  "
            f"Divergent: {div_pct:.1f}%",
            "",
            f"Enter in Aggregator as:",
            f"  Source type: Numerical (u_num)",
            f"  Sigma Value: {s.u_num_p95:.6g}{unit_sfx}",
            f"  Sigma Basis: Confirmed 1\u03c3",
            f"  DOF: \u221e",
            f"  Distribution: Normal",
        ]
        if div_pct > 10:
            info_lines.append("")
            info_lines.append(
                f"\u26A0 {div_pct:.0f}% of points are divergent. "
                f"Review spatial distribution before carry-over.")
        self._spatial_carry_info.setPlainText("\n".join(info_lines))
        self._btn_copy_spatial_carry.setEnabled(True)

    def _copy_spatial_carry_value(self):
        """Copy recommended spatial carry value to clipboard."""
        s = self._summary
        if s is None or s.n_points_valid == 0:
            return
        try:
            unit_sfx = f" {s.quantity_unit}" if s.quantity_unit else ""
            text = (
                f"u_num = {s.u_num_p95:.6g}{unit_sfx}\n"
                f"Basis: Confirmed 1\u03c3\n"
                f"DOF: \u221e\n"
                f"Distribution: Normal\n"
                f"Source: Spatial GCI 95th percentile "
                f"({s.n_points_valid:,} points)")
            QApplication.clipboard().setText(text)
        except Exception as exc:
            QMessageBox.warning(self, "Clipboard Error",
                                f"Could not copy:\n\n{exc}")

    def _copy_spatial_report(self):
        """Copy spatial report statements to clipboard."""
        try:
            txt = self._report_text.toPlainText()
            if txt.strip():
                QApplication.clipboard().setText(txt)
        except Exception as exc:
            QMessageBox.warning(self, "Clipboard Error",
                                f"Could not copy to clipboard:\n\n{exc}")

    def _export_spatial_report(self):
        """Export spatial report statements to a text file."""
        txt = self._report_text.toPlainText()
        if not txt.strip():
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Spatial Report Statements",
            "", "Text Files (*.txt);;All Files (*)"
        )
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(txt)
            except Exception as exc:
                QMessageBox.critical(
                    self, "Export Error",
                    f"Could not write file:\n\n{exc}"
                )

    def _generate_spatial_report_statements(self):
        """Generate copy-pasteable regulatory paragraphs for spatial GCI.

        Called at the end of _compute_spatial().  Produces three sections:
        1. Spatial Numerical Uncertainty Statement
        2. Limitations & Caveats
        3. Standards Compliance
        """
        s = self._summary
        if s is None:
            self._report_text.clear()
            return

        unit_sfx = f" {s.quantity_unit}" if s.quantity_unit else ""
        q_label = s.quantity_name or "the field variable"
        total = max(s.n_points_total, 1)
        mono_pct = 100 * s.n_monotonic / total
        osc_pct = 100 * s.n_oscillatory / total
        div_pct = 100 * s.n_divergent / total
        gi_pct = 100 * s.n_grid_independent / total

        # Detect subsampling
        is_subsampled = (self._cmb_sampling.currentData() == "subsample")
        n_samples_used = self._spn_n_samples.value() if is_subsampled else 0

        lines: List[str] = []
        lines.append("=" * 70)
        lines.append("  SPATIAL REPORT STATEMENTS FOR V&V DOCUMENTATION")
        lines.append("  (Copy-paste into regulatory submission or V&V report)")
        lines.append("=" * 70)
        lines.append("")

        # ---- Section 1: Spatial Numerical Uncertainty Statement ----
        lines.append("1. SPATIAL NUMERICAL UNCERTAINTY STATEMENT")
        lines.append("-" * 50)
        lines.append("")

        subsamp_note = ""
        if is_subsampled:
            subsamp_note = (
                f" The analysis used farthest-point sampling (FPS) "
                f"to select {n_samples_used:,} representative "
                f"points from the full field for computational "
                f"efficiency."
            )

        # Build convergence breakdown text
        conv_parts = []
        if mono_pct > 0:
            conv_parts.append(f"{mono_pct:.1f}% monotonic convergence")
        if osc_pct > 0:
            conv_parts.append(f"{osc_pct:.1f}% oscillatory convergence")
        if div_pct > 0:
            conv_parts.append(f"{div_pct:.1f}% divergent behavior")
        if gi_pct > 0:
            conv_parts.append(f"{gi_pct:.1f}% grid-independent (u_num = 0)")
        conv_text = ", ".join(conv_parts) if conv_parts else "no classified points"

        # H4 guard: gate numeric statement on n_points_valid > 0
        if s.n_points_valid == 0:
            lines.append(
                f"  A point-by-point grid convergence study was performed "
                f"for {q_label} over {s.n_points_total:,} spatial "
                f"locations.{subsamp_note} "
                f"Of these, {conv_text}."
            )
            lines.append("")
            lines.append(
                f"  WARNING: No defensible spatial numerical uncertainty "
                f"could be established. All field points are divergent "
                f"or skipped. Analyst action is required before using "
                f"any numeric values from this analysis in a V&V report."
            )
        else:
            lines.append(
                f"  A point-by-point grid convergence study was performed "
                f"for {q_label} over {s.n_points_total:,} spatial "
                f"locations.{subsamp_note} "
                f"Of these, {conv_text}. "
                f"The 95th percentile numerical uncertainty across the "
                f"field is u_num(95%) = {s.u_num_p95:.4g}{unit_sfx}, "
                f"which is the recommended value for the V&V 20 "
                f"uncertainty budget. "
                f"Summary statistics: mean u_num = {s.u_num_mean:.4g}"
                f"{unit_sfx}, median = {s.u_num_median:.4g}{unit_sfx}, "
                f"max = {s.u_num_max:.4g}{unit_sfx}."
            )
        lines.append("")

        # ---- Section 2: Limitations & Caveats ----
        lines.append("2. LIMITATIONS & CAVEATS")
        lines.append("-" * 50)
        lines.append("")

        caveats: List[str] = []
        if s.n_points_valid == 0:
            caveats.append(
                "  - CRITICAL: No valid field points produced a "
                "defensible numerical uncertainty. All points are "
                "divergent, skipped, or otherwise invalid. This "
                "spatial analysis cannot be used for V&V 20 "
                "uncertainty budgeting without corrective action."
            )
        if div_pct > 20:
            caveats.append(
                f"  - HIGH divergent fraction ({div_pct:.1f}%). "
                f"A significant portion of the field does not "
                f"exhibit grid convergence. The 95th percentile "
                f"u_num may underestimate the true spatial "
                f"numerical uncertainty. Investigate mesh quality "
                f"and solver settings in divergent regions."
            )
        elif div_pct > 10:
            caveats.append(
                f"  - Elevated divergent fraction ({div_pct:.1f}%). "
                f"Some regions of the field do not converge with "
                f"grid refinement."
            )

        if is_subsampled:
            caveats.append(
                f"  - Farthest-point subsampling was used "
                f"({n_samples_used:,} of {s.n_points_total:,} "
                f"points). Statistics are approximate; localized "
                f"features may be underrepresented. Increase "
                f"sample count or use all points for final "
                f"certification."
            )

        mode = self._cmb_mode.currentData()
        if mode in ("sep_csv", "sep_prof"):
            caveats.append(
                "  - Separate-files mode: field values were "
                "interpolated to common point locations using "
                "inverse-distance weighting (IDW). Interpolation "
                "error is not included in the uncertainty estimate."
            )

        if caveats:
            for c in caveats:
                lines.append(c)
        else:
            lines.append("  No significant limitations identified.")
        lines.append("")

        # ---- Section 3: Standards Compliance ----
        lines.append("3. STANDARDS COMPLIANCE")
        lines.append("-" * 50)
        lines.append("")
        lines.append(
            "  Method:     Point-by-point Grid Convergence Index "
            "(GCI)"
        )
        lines.append(
            "  Procedure:  ASME V&V 20-2009 (R2021) Section 5.1, "
            "extended to spatial fields"
        )
        lines.append(
            "  Statistic:  95th percentile u_num recommended for "
            "the V&V 20 uncertainty budget"
        )
        lines.append(
            "  Basis:      Richardson Extrapolation with observed "
            "or assumed order at each point"
        )
        lines.append(
            "  References: Celik et al. (2008) J. Fluids Eng. "
            "130(7), 078001;"
        )
        lines.append(
            "              Roache (1998) Verification and Validation "
            "in Computational"
        )
        lines.append(
            "              Science and Engineering, Hermosa Publishers."
        )
        lines.append("")
        lines.append("=" * 70)

        self._report_text.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # STATE SERIALIZATION (save / load)
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Serialize spatial tab settings (not the data itself)."""
        return {
            "mode": self._cmb_mode.currentData(),
            "dim": self._cmb_dim_spatial.currentData(),
            "order_preset_index": self._cmb_order_preset_spatial.currentIndex(),
            "theoretical_order": self._spn_p_theo.value(),
            "safety_factor": self._spn_fs_spatial.value(),
            "quantity_name": self._edt_qty_name.text(),
            "quantity_unit": self._cmb_unit_val.currentText(),
            "include_oscillatory": self._chk_incl_osc.isChecked(),
            "min_threshold": self._spn_min_thresh.value(),
            "sampling_mode": self._cmb_sampling.currentData(),
            "n_samples": self._spn_n_samples.value(),
        }

    def set_state(self, state: dict):
        """Restore spatial tab settings from a dict."""
        mode = state.get("mode", "pre_interp")
        for i in range(self._cmb_mode.count()):
            if self._cmb_mode.itemData(i) == mode:
                self._cmb_mode.setCurrentIndex(i)
                break

        dim = state.get("dim", 3)
        idx_dim = 0 if dim == 3 else 1
        self._cmb_dim_spatial.setCurrentIndex(idx_dim)

        preset_idx = state.get("order_preset_index", 0)
        if 0 <= preset_idx < self._cmb_order_preset_spatial.count():
            self._cmb_order_preset_spatial.setCurrentIndex(preset_idx)
        self._spn_p_theo.setValue(state.get("theoretical_order", 2.0))
        self._spn_fs_spatial.setValue(state.get("safety_factor", 0.0))
        self._edt_qty_name.setText(state.get("quantity_name", "Temperature"))
        unit = state.get("quantity_unit", "K")
        self._cmb_unit_val.setEditText(unit)
        self._chk_incl_osc.setChecked(
            state.get("include_oscillatory", True))
        self._spn_min_thresh.setValue(state.get("min_threshold", 0.0))
        sampling_mode = state.get("sampling_mode", "all")
        for i in range(self._cmb_sampling.count()):
            if self._cmb_sampling.itemData(i) == sampling_mode:
                self._cmb_sampling.setCurrentIndex(i)
                break
        self._spn_n_samples.setValue(state.get("n_samples", 500))

    def clear_all(self):
        """Reset the spatial tab to defaults."""
        self._cmb_mode.setCurrentIndex(0)
        self._cmb_dim_spatial.setCurrentIndex(0)
        self._cmb_order_preset_spatial.setCurrentIndex(0)
        self._spn_p_theo.setValue(2.0)
        self._spn_fs_spatial.setValue(0.0)
        self._edt_qty_name.setText("Temperature")
        self._cmb_unit_cat.setCurrentIndex(0)
        self._populate_spatial_unit_combo("Temperature")
        self._chk_incl_osc.setChecked(True)
        self._spn_min_thresh.setValue(0.0)
        self._cmb_sampling.setCurrentIndex(0)  # "Use all points"
        self._spn_n_samples.setValue(500)
        self._summary = None
        self._pre_coords = None
        self._pre_solutions = None
        self._pre_grid_names = None
        self._sep_data.clear()
        self._prof_data.clear()
        self._results_text.clear()
        self._stats_table.setRowCount(0)
        self._lbl_pre_file.setText("No file loaded")
        self._lbl_pre_file.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; font-style: italic;")
        self._lbl_pre_info.setText("")
        self._tbl_pre_cells.setRowCount(0)
        for fig in (self._fig_hist, self._fig_cdf,
                    self._fig_conv, self._fig_umap):
            fig.clear()
        for canvas in (self._canvas_hist, self._canvas_cdf,
                       self._canvas_conv, self._canvas_umap):
            canvas.draw()
        self._guidance_import.set_guidance("", 'green')
        self._guidance_spatial.set_guidance("", 'green')
        self._report_text.clear()
        self._spatial_carry_table.setRowCount(0)
        self._spatial_carry_info.clear()
        self._btn_copy_spatial_carry.setEnabled(False)


# =============================================================================
# REFERENCE TAB
# =============================================================================

class GCIReferenceTab(QWidget):
    """Built-in reference documentation for GCI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        browser = QTextEdit()
        browser.setReadOnly(True)
        browser.setHtml(self._build_html())
        layout.addWidget(browser)

    def _build_html(self):
        c = DARK_COLORS
        return f"""
        <style>
            body {{
                background-color: {c['bg']};
                color: {c['fg']};
                font-family: {', '.join(FONT_FAMILIES)};
                font-size: 13px;
                padding: 16px;
            }}
            h1 {{ color: {c['accent']}; border-bottom: 2px solid {c['accent']};
                  padding-bottom: 8px; }}
            h2 {{ color: {c['accent']}; margin-top: 24px; }}
            h3 {{ color: {c['fg_bright']}; }}
            .formula {{ background-color: {c['bg_input']};
                       border: 1px solid {c['border']}; border-radius: 4px;
                       padding: 12px; margin: 8px 0; font-family: monospace;
                       color: {c['orange']}; }}
            .note {{ background-color: #1a2e1a; border-left: 4px solid {c['green']};
                    padding: 10px; margin: 8px 0; border-radius: 4px; }}
            .warn {{ background-color: #2e2a1a; border-left: 4px solid {c['yellow']};
                    padding: 10px; margin: 8px 0; border-radius: 4px; }}
            .danger {{ background-color: #2e1a1a; border-left: 4px solid {c['red']};
                      padding: 10px; margin: 8px 0; border-radius: 4px; }}
            table {{ border-collapse: collapse; margin: 12px 0; width: 100%; }}
            th {{ background-color: {c['bg_alt']}; color: {c['fg']};
                 padding: 8px; border: 1px solid {c['border']}; text-align: left; }}
            td {{ padding: 8px; border: 1px solid {c['border']}; }}
            code {{ color: {c['orange']}; background-color: {c['bg_input']};
                   padding: 2px 4px; border-radius: 3px; }}
            a {{ color: {c['link']}; }}
        </style>

        <h1>Grid Convergence Index (GCI) Reference</h1>

        <h2>What Is GCI?</h2>
        <p>The Grid Convergence Index is a standardized method for estimating
        the <strong>numerical uncertainty</strong> in a CFD solution due to
        the finite grid (mesh) used. In plain terms: it tells you how much
        your answer might change if you made the mesh infinitely fine.</p>

        <p>GCI is the accepted method for quantifying u_num (numerical
        uncertainty) in the ASME V&amp;V 20 framework.</p>

        <h2>The Basic Idea</h2>
        <p>Run your CFD case on multiple grids of different fineness.
        If the answer changes less and less as the grid gets finer,
        you can mathematically estimate what the answer would be on
        an infinitely fine grid — and how far your current grid is
        from that answer.</p>

        <h2>How Many Grids Do I Need?</h2>
        <table>
        <tr><th>Grids</th><th>What You Get</th><th>Recommendation</th></tr>
        <tr><td><strong>2</strong></td>
            <td>Can compute GCI but must <em>assume</em> the order of
            accuracy (typically p=2 for second-order schemes).
            Uses a conservative safety factor Fs = 3.0.</td>
            <td>Minimum viable. Not recommended for certification or
            publication.</td></tr>
        <tr><td><strong>3</strong></td>
            <td>Can <em>compute</em> the observed order of accuracy.
            Uses Fs = 1.25 (less conservative because you have more
            information). Can check the asymptotic range.</td>
            <td><strong>Standard procedure.</strong> This is what Celik
            et al. (2008) and most journals require.</td></tr>
        <tr><td><strong>4+</strong></td>
            <td>Primary GCI from the finest 3 grids. Additional grids
            provide cross-checks (compute GCI on different triplets
            to verify consistency).</td>
            <td>Best practice for high-confidence applications.
            Extra grids increase confidence but the GCI value itself
            comes from the finest 3.</td></tr>
        </table>

        <h2>Is Richardson Extrapolation Required?</h2>
        <p><strong>For 3+ grids: Yes</strong> — Richardson extrapolation is
        the mathematical foundation of GCI. It estimates the "exact"
        (zero-spacing) solution by extrapolating the observed convergence
        trend to h = 0.</p>
        <p><strong>For 2 grids: Limited</strong> — with only 2 grids, the
        observed order cannot be computed. The tool assumes the order
        equals the theoretical order of your numerical scheme and uses
        this assumed order to compute Richardson extrapolation and GCI,
        with a larger safety factor (Fs = 3.0) to compensate for the
        added uncertainty of not knowing the true order.</p>
        <p><strong>For oscillatory convergence: No</strong> — when the
        solution oscillates between grid levels, Richardson extrapolation
        is unreliable. Instead, the GCI is computed from the oscillation
        range with a conservative safety factor.</p>

        <h2>The Three-Grid Procedure</h2>
        <p><em>Per Celik et al. (2008) and Roache (1998):</em></p>

        <h3>Step 1: Define the grids</h3>
        <p>Label grids 1 (finest), 2 (medium), 3 (coarsest).
        Compute representative cell size:</p>
        <div class="formula">h = [ (1/N) &times; &Sigma;(&Delta;V<sub>i</sub>) ]<sup>1/3</sup>
        &asymp; (1/N)<sup>1/dim</sup></div>

        <h3>Step 2: Compute refinement ratios</h3>
        <div class="formula">r<sub>21</sub> = h<sub>2</sub> / h<sub>1</sub>
        = (N<sub>1</sub> / N<sub>2</sub>)<sup>1/dim</sup></div>
        <div class="note">Celik et al. recommend r &gt; 1.3 for reliable results.</div>

        <h3>Step 3: Determine convergence type</h3>
        <div class="formula">R = (f<sub>2</sub> - f<sub>1</sub>) /
        (f<sub>3</sub> - f<sub>2</sub>)</div>
        <table>
        <tr><th>R value</th><th>Type</th><th>Meaning</th></tr>
        <tr><td>0 &le; R &lt; 1</td><td style="color: {c['green']};">
            <strong>Monotonic</strong></td>
            <td>Ideal. Solutions converge steadily (R=0 means finest
            pair is grid-independent).</td></tr>
        <tr><td>&minus;1 &lt; R &lt; 0</td><td style="color: {c['yellow']};">
            <strong>Oscillatory</strong></td>
            <td>Solution oscillates. RE unreliable.</td></tr>
        <tr><td>R &ge; 1 or R &le; &minus;1</td><td style="color: {c['red']};">
            <strong>Divergent</strong></td>
            <td>Solution gets worse with refinement. R &ge; 1 is monotonic
            divergence; R &le; &minus;1 is oscillatory divergence (growing
            oscillations). GCI invalid.</td></tr>
        </table>

        <h3>Step 4: Compute observed order of accuracy</h3>
        <p>For constant refinement ratio (r<sub>21</sub> = r<sub>32</sub> = r):</p>
        <div class="formula">p = ln(e<sub>32</sub> / e<sub>21</sub>) / ln(r)</div>
        <p>For non-constant ratios, solve iteratively (Celik Eq. 5).</p>

        <h3>Step 5: Richardson extrapolation</h3>
        <div class="formula">f<sub>exact</sub> = f<sub>1</sub> +
        (f<sub>1</sub> - f<sub>2</sub>) / (r<sub>21</sub><sup>p</sup> - 1)</div>

        <h3>Step 6: Compute GCI</h3>
        <div class="formula">GCI<sub>fine</sub> = F<sub>s</sub> &times;
        |e<sub>a</sub><sup>21</sup>| / (r<sub>21</sub><sup>p</sup> - 1)
        <br><br>where e<sub>a</sub><sup>21</sup> = |(f<sub>2</sub> - f<sub>1</sub>)
        / f<sub>1</sub>|</div>

        <h3>Step 7: Asymptotic range check</h3>
        <div class="formula">Asymptotic ratio = GCI<sub>coarse</sub> /
        (r<sub>21</sub><sup>p</sup> &times; GCI<sub>fine</sub>)
        &asymp; 1.0</div>
        <div class="note">If asymptotic ratio is between 0.95 and 1.05,
        the grids are in the asymptotic range and GCI is reliable.</div>

        <h2>Safety Factor Recommendations</h2>
        <table>
        <tr><th>Scenario</th><th>F<sub>s</sub></th><th>Why</th></tr>
        <tr><td>3-grid, monotonic convergence</td><td><strong>1.25</strong></td>
            <td>Standard; observed p provides enough info</td></tr>
        <tr><td>2-grid study</td><td><strong>3.0</strong></td>
            <td>Conservative; order p is assumed, not observed</td></tr>
        <tr><td>Oscillatory convergence</td><td><strong>3.0</strong></td>
            <td>Conservative; RE is unreliable</td></tr>
        <tr><td>1st-order schemes</td><td><strong>3.0</strong></td>
            <td>First-order is very sensitive to grid</td></tr>
        </table>

        <h2>Converting GCI to u_num (for V&amp;V 20)</h2>
        <p>GCI_fine represents a conservative error band on the fine-grid solution.
        To convert to a standard uncertainty (1&sigma;) for use in the V&amp;V 20
        RSS combination:</p>
        <div class="formula">u_num = GCI_fine &times; |f<sub>1</sub>| /
        F<sub>s</sub></div>
        <p>This strips out the safety factor (which acts like a coverage
        factor k) to recover the underlying 1&sigma; estimate. You can then
        enter u_num directly into your V&amp;V 20 uncertainty budget.</p>

        <h2>References</h2>
        <ul>
        <li>Celik, I.B. et al. (2008) "Procedure for Estimation and Reporting
        of Uncertainty Due to Discretization in CFD Applications"
        <em>J. Fluids Eng.</em> 130(7), 078001</li>
        <li>Roache, P.J. (1998) <em>Verification and Validation in
        Computational Science and Engineering</em>, Hermosa Publishers</li>
        <li>Richardson, L.F. (1911) "The Approximate Arithmetical Solution
        by Finite Differences of Physical Problems"
        <em>Phil. Trans. R. Soc. A</em> 210, 307-357</li>
        <li>ASME V&amp;V 20-2009 (R2021) Section 5.1</li>
        <li>ITTC (2024) Procedure 7.5-03-01-01</li>
        </ul>
        """


# =============================================================================
# HTML REPORT GENERATOR
# =============================================================================

_esc = _html_esc  # Alias: use stdlib html.escape for all HTML escaping.


# =============================================================================
# MESH STUDY PLANNER TAB
# =============================================================================

# Study checklist items — grouped by category
STUDY_CHECKLIST_ITEMS = {
    "Mesh Generation": [
        "Same meshing strategy (topology) on all grids",
        "Consistent boundary layer resolution approach",
        "Mesh quality checked on all grids (skewness, aspect ratio, orthogonality)",
        "y+ values in valid range for wall treatment on all grids",
        "No unintended mesh features (e.g., refinement zones in wrong place)",
    ],
    "Solver Settings": [
        "Identical solver settings on all grids (scheme, turbulence model, BCs)",
        "Same convergence criteria on all grids",
        "Iterative convergence achieved on all grids (residuals 3+ orders)",
        "Monitor points have flatlined on all grids",
        "Same time-step (transient) or pseudo-time settings on all grids",
    ],
    "Data Extraction": [
        "Same quantity extraction location(s) on all grids",
        "Extraction point away from singularities and sharp gradients",
        "Solution values recorded at the same physical conditions",
        "Cell counts recorded accurately for each grid",
        "Units consistent across all grids",
    ],
    "Study Design": [
        "At least 3 grids planned (2-grid is minimum viable only)",
        "Refinement ratios r > 1.3 between consecutive grids",
        "Systematic (not random) refinement strategy",
        "Production grid identified",
        "Budget allows for all planned grids (time, storage, licenses)",
    ],
}


def _plan_mesh_sizes(production_cells, dim, r, n_grids, strategy):
    """Compute target mesh sizes for a grid convergence study.

    Returns list of (grid_label, cell_count, is_production) tuples,
    ordered finest-first.
    """
    cells = []
    if strategy == "refine":
        # Production is coarsest; refine upward
        for i in range(n_grids):
            level = n_grids - 1 - i  # finest first
            n = production_cells * (r ** (dim * level))
            cells.append(n)
        prod_idx = n_grids - 1  # last (coarsest)
    elif strategy == "coarsen":
        # Production is finest; coarsen downward
        for i in range(n_grids):
            n = production_cells / (r ** (dim * i))
            cells.append(n)
        prod_idx = 0  # first (finest)
    else:
        # Production is middle grid
        mid = n_grids // 2
        for i in range(n_grids):
            offset = mid - i  # positive = finer, negative = coarser
            n = production_cells * (r ** (dim * offset))
            cells.append(n)
        prod_idx = mid

    result = []
    for i, n in enumerate(cells):
        if i == 0:
            label = f"Grid {i+1} (finest)"
        elif i == n_grids - 1:
            label = f"Grid {i+1} (coarsest)"
        else:
            label = f"Grid {i+1}"
        result.append((label, round(n), i == prod_idx))
    return result


class MeshStudyPlannerTab(QWidget):
    """Pre-analysis planning tab for GCI studies."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        splitter = QSplitter(Qt.Horizontal, self)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.addWidget(splitter)

        # ---- LEFT PANEL: Inputs ----
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(8)
        left_scroll = QScrollArea()
        left_scroll.setWidget(left)
        left_scroll.setWidgetResizable(True)
        splitter.addWidget(left_scroll)

        # -- Section A: Target Mesh Calculator --
        grp_target = QGroupBox("Target Mesh Calculator")
        target_form = QFormLayout(grp_target)
        target_form.setSpacing(6)

        self._spn_prod_cells = QDoubleSpinBox()
        self._spn_prod_cells.setRange(1, 1e12)
        self._spn_prod_cells.setDecimals(0)
        self._spn_prod_cells.setValue(1000000)
        self._spn_prod_cells.setGroupSeparatorShown(True)
        self._spn_prod_cells.setToolTip(
            "Total cell count of your production (baseline) mesh.")
        target_form.addRow("Production mesh cells:", self._spn_prod_cells)

        self._cmb_plan_dim = QComboBox()
        self._cmb_plan_dim.addItem("3D", 3)
        self._cmb_plan_dim.addItem("2D", 2)
        target_form.addRow("Dimensions:", self._cmb_plan_dim)

        self._spn_target_r = QDoubleSpinBox()
        self._spn_target_r.setRange(1.1, 3.0)
        self._spn_target_r.setValue(1.5)
        self._spn_target_r.setSingleStep(0.1)
        self._spn_target_r.setDecimals(2)
        self._spn_target_r.setToolTip(
            "Desired refinement ratio between consecutive grids.\n"
            "Celik et al. recommend r > 1.3.\n"
            "r = 1.5 is a good default; r = 2.0 is ideal but expensive.")
        target_form.addRow("Target refinement ratio:", self._spn_target_r)

        self._cmb_study_size = QComboBox()
        self._cmb_study_size.addItems([
            "3-grid study (standard)",
            "4-grid study (cross-check)",
            "5-grid study (best practice)",
        ])
        target_form.addRow("Study type:", self._cmb_study_size)

        self._cmb_strategy = QComboBox()
        self._cmb_strategy.addItem(
            "Refine from production (production = coarsest)", "refine")
        self._cmb_strategy.addItem(
            "Coarsen from production (production = finest)", "coarsen")
        self._cmb_strategy.addItem(
            "Production is middle grid", "middle")
        self._cmb_strategy.setToolTip(
            "How to build the study grids relative to production:\n\n"
            "Refine: Build finer grids above your production mesh.\n"
            "  Your production mesh is the coarsest in the study.\n\n"
            "Coarsen: Build coarser grids below your production mesh.\n"
            "  Your production mesh is the finest in the study.\n\n"
            "Middle: Build grids above and below your production mesh.\n"
            "  Your production mesh is in the middle of the study.")
        target_form.addRow("Strategy:", self._cmb_strategy)

        self._btn_compute_targets = QPushButton("Compute Mesh Targets")
        self._btn_compute_targets.setStyleSheet(
            f"QPushButton {{ background-color: {DARK_COLORS['accent']}; "
            f"color: {DARK_COLORS['bg']}; font-weight: bold; "
            f"font-size: 14px; padding: 10px; }}"
            f"QPushButton:hover {{ background-color: "
            f"{DARK_COLORS['accent_hover']}; }}")
        self._btn_compute_targets.clicked.connect(self._compute_targets)
        target_form.addRow(self._btn_compute_targets)

        left_layout.addWidget(grp_target)

        # -- Section B: Refinement Ratio Checker --
        grp_ratio = QGroupBox("Refinement Ratio Checker")
        ratio_lay = QVBoxLayout(grp_ratio)
        ratio_lay.setSpacing(4)

        ratio_lay.addWidget(QLabel(
            "Enter your actual cell counts to check ratios:"))
        self._tbl_ratio_check = QTableWidget(3, 2)
        self._tbl_ratio_check.setHorizontalHeaderLabels(
            ["Grid", "Cell Count"])
        self._tbl_ratio_check.setMaximumHeight(150)
        for i in range(3):
            lbl = QTableWidgetItem(f"Grid {i+1}")
            lbl.setFlags(lbl.flags() & ~Qt.ItemIsEditable)
            self._tbl_ratio_check.setItem(i, 0, lbl)
        style_table(self._tbl_ratio_check)
        ratio_lay.addWidget(self._tbl_ratio_check)

        ratio_btn_row = QHBoxLayout()
        self._btn_add_ratio_grid = QPushButton("+ Grid")
        self._btn_add_ratio_grid.clicked.connect(self._add_ratio_grid)
        self._btn_rem_ratio_grid = QPushButton("- Grid")
        self._btn_rem_ratio_grid.clicked.connect(self._rem_ratio_grid)
        self._btn_check_ratios = QPushButton("Check Ratios")
        self._btn_check_ratios.clicked.connect(self._check_ratios)
        ratio_btn_row.addWidget(self._btn_add_ratio_grid)
        ratio_btn_row.addWidget(self._btn_rem_ratio_grid)
        ratio_btn_row.addWidget(self._btn_check_ratios)
        ratio_btn_row.addStretch()
        ratio_lay.addLayout(ratio_btn_row)

        left_layout.addWidget(grp_ratio)

        # -- Section C: Resource Estimator --
        grp_resource = QGroupBox("Resource Estimator")
        res_form = QFormLayout(grp_resource)
        res_form.setSpacing(6)

        self._spn_baseline_time = QDoubleSpinBox()
        self._spn_baseline_time.setRange(0.001, 10000)
        self._spn_baseline_time.setValue(1.0)
        self._spn_baseline_time.setDecimals(2)
        self._spn_baseline_time.setSuffix(" hours")
        self._spn_baseline_time.setToolTip(
            "Wall-clock time for a single production run.")
        res_form.addRow("Baseline solve time:", self._spn_baseline_time)

        self._cmb_scaling = QComboBox()
        self._cmb_scaling.addItem("Linear (N)", "linear")
        self._cmb_scaling.addItem("N^1.5 (typical implicit CFD)", "n15")
        self._cmb_scaling.addItem("N^2 (dense linear algebra)", "n2")
        self._cmb_scaling.setCurrentIndex(1)  # N^1.5 default
        self._cmb_scaling.setToolTip(
            "How solve time scales with cell count:\n"
            "  Linear: direct proportionality\n"
            "  N^1.5: typical for implicit pressure-based CFD solvers\n"
            "  N^2: dense direct solvers")
        res_form.addRow("Scaling model:", self._cmb_scaling)

        self._btn_estimate = QPushButton("Estimate Resources")
        self._btn_estimate.clicked.connect(self._estimate_resources)
        res_form.addRow(self._btn_estimate)

        left_layout.addWidget(grp_resource)

        left_layout.addStretch()

        # ---- RIGHT PANEL: Results subtabs ----
        self._right_tabs = QTabWidget()
        splitter.addWidget(self._right_tabs)

        # Subtab 1: Mesh Targets
        targets_widget = QWidget()
        targets_lay = QVBoxLayout(targets_widget)
        targets_lay.setContentsMargins(6, 6, 6, 6)

        self._targets_table = QTableWidget()
        self._targets_table.setColumnCount(5)
        self._targets_table.setHorizontalHeaderLabels([
            "Grid", "Cell Count", "Ref. Ratio", "r > 1.3?", "Role"
        ])
        self._targets_table.setAlternatingRowColors(True)
        style_table(self._targets_table)
        targets_lay.addWidget(self._targets_table)

        self._guidance_targets = GuidancePanel("Target Assessment")
        targets_lay.addWidget(self._guidance_targets)
        self._right_tabs.addTab(targets_widget, "Mesh Targets")

        # Subtab 2: Ratio Check
        ratio_widget = QWidget()
        ratio_r_lay = QVBoxLayout(ratio_widget)
        ratio_r_lay.setContentsMargins(6, 6, 6, 6)

        self._ratio_results_table = QTableWidget()
        self._ratio_results_table.setColumnCount(4)
        self._ratio_results_table.setHorizontalHeaderLabels([
            "Grid Pair", "Refinement Ratio", "Assessment", "Suggested Count"
        ])
        self._ratio_results_table.setAlternatingRowColors(True)
        style_table(self._ratio_results_table)
        ratio_r_lay.addWidget(self._ratio_results_table)

        self._guidance_ratios = GuidancePanel("Ratio Assessment")
        ratio_r_lay.addWidget(self._guidance_ratios)
        self._right_tabs.addTab(ratio_widget, "Ratio Check")

        # Subtab 3: Resource Estimator
        resource_widget = QWidget()
        resource_lay = QVBoxLayout(resource_widget)
        resource_lay.setContentsMargins(6, 6, 6, 6)

        self._resource_table = QTableWidget()
        self._resource_table.setColumnCount(5)
        self._resource_table.setHorizontalHeaderLabels([
            "Grid", "Cells", "Est. Time", "Multiplier", "Est. Memory"
        ])
        self._resource_table.setAlternatingRowColors(True)
        style_table(self._resource_table)
        resource_lay.addWidget(self._resource_table)

        self._resource_summary = QLabel("")
        self._resource_summary.setWordWrap(True)
        self._resource_summary.setStyleSheet(
            f"color: {DARK_COLORS['accent']}; font-size: 13px; "
            f"padding: 8px; border: 1px solid {DARK_COLORS['border']}; "
            f"border-radius: 4px;")
        resource_lay.addWidget(self._resource_summary)
        self._right_tabs.addTab(resource_widget, "Resources")

        # Subtab 4: Study Checklist
        checklist_widget = QWidget()
        checklist_lay = QVBoxLayout(checklist_widget)
        checklist_lay.setContentsMargins(6, 6, 6, 6)

        cl_header = QLabel(
            "<b>Pre-Flight Checklist</b> — Verify before running "
            "your grid convergence study")
        cl_header.setWordWrap(True)
        cl_header.setStyleSheet(
            f"color: {DARK_COLORS['accent']}; padding: 4px 0;")
        checklist_lay.addWidget(cl_header)

        cl_scroll = QScrollArea()
        cl_scroll.setWidgetResizable(True)
        cl_inner = QWidget()
        cl_inner_lay = QVBoxLayout(cl_inner)
        cl_inner_lay.setSpacing(4)

        self._checklist_boxes = {}
        for category, items in STUDY_CHECKLIST_ITEMS.items():
            grp = QGroupBox(category)
            grp_lay = QVBoxLayout(grp)
            grp_lay.setSpacing(2)
            for item_text in items:
                cb = QCheckBox(item_text)
                cb.setStyleSheet(f"color: {DARK_COLORS['fg']};")
                grp_lay.addWidget(cb)
                self._checklist_boxes[item_text] = cb
            cl_inner_lay.addWidget(grp)

        cl_inner_lay.addStretch()
        cl_scroll.setWidget(cl_inner)
        checklist_lay.addWidget(cl_scroll)

        cl_btn_row = QHBoxLayout()
        btn_check_all = QPushButton("Check All")
        btn_check_all.clicked.connect(self._check_all_items)
        btn_uncheck_all = QPushButton("Uncheck All")
        btn_uncheck_all.clicked.connect(self._uncheck_all_items)
        self._lbl_checklist_status = QLabel("")
        cl_btn_row.addWidget(btn_check_all)
        cl_btn_row.addWidget(btn_uncheck_all)
        cl_btn_row.addWidget(self._lbl_checklist_status)
        cl_btn_row.addStretch()
        checklist_lay.addLayout(cl_btn_row)

        self._right_tabs.addTab(checklist_widget, "Study Checklist")

        # Subtab 5: Quick Reference
        ref_widget = QWidget()
        ref_lay = QVBoxLayout(ref_widget)
        ref_lay.setContentsMargins(6, 6, 6, 6)

        self._ref_text = QTextEdit()
        self._ref_text.setReadOnly(True)
        self._ref_text.setHtml(self._build_quick_ref_html())
        ref_lay.addWidget(self._ref_text)

        self._right_tabs.addTab(ref_widget, "Quick Reference")

        # Splitter proportions
        splitter.setSizes([400, 650])

    # ------------------------------------------------------------------
    # MESH TARGET CALCULATOR
    # ------------------------------------------------------------------

    def _compute_targets(self):
        """Compute target mesh sizes and display results."""
        prod_cells = int(self._spn_prod_cells.value())
        dim = self._cmb_plan_dim.currentData()
        r = self._spn_target_r.value()
        n_grids = self._cmb_study_size.currentIndex() + 3  # 3, 4, or 5
        strategy = self._cmb_strategy.currentData()

        mesh_plan = _plan_mesh_sizes(prod_cells, dim, r, n_grids, strategy)

        self._targets_table.setRowCount(len(mesh_plan))
        all_ok = True
        for i, (label, cells, is_prod) in enumerate(mesh_plan):
            # Compute ratio to next coarser grid
            if i < len(mesh_plan) - 1:
                next_cells = mesh_plan[i + 1][1]
                if next_cells > 0:
                    ratio = compute_refinement_ratio(cells, next_cells, dim)
                    ratio_str = f"{ratio:.3f}"
                    ok = ratio >= 1.3
                    check_str = "\u2713 Yes" if ok else "\u2717 No"
                    if not ok:
                        all_ok = False
                else:
                    ratio_str = "N/A"
                    check_str = "\u2717"
                    all_ok = False
            else:
                ratio_str = "\u2014"
                check_str = "\u2014"

            role = "PRODUCTION" if is_prod else "Study grid"

            items = [
                (label, None),
                (f"{cells:,}", None),
                (ratio_str, None),
                (check_str,
                 DARK_COLORS['green'] if "\u2713" in check_str
                 else (DARK_COLORS['red'] if "\u2717" in check_str
                       else None)),
                (role,
                 DARK_COLORS['accent'] if is_prod else None),
            ]
            for col, (text, color) in enumerate(items):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if color:
                    item.setForeground(QColor(color))
                if is_prod:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                self._targets_table.setItem(i, col, item)

        # Summary guidance
        total_cells = sum(m[1] for m in mesh_plan)
        finest_cells = mesh_plan[0][1]
        if all_ok:
            self._guidance_targets.set_guidance(
                f"All refinement ratios meet the r > 1.3 criterion. "
                f"Total cells across all grids: {total_cells:,}. "
                f"Finest grid: {finest_cells:,} cells.\n\n"
                f"This study is ready to proceed.",
                'green')
        else:
            self._guidance_targets.set_guidance(
                f"Some refinement ratios are below 1.3. Consider "
                f"increasing the target ratio or adjusting the "
                f"production mesh cell count.\n\n"
                f"Total cells: {total_cells:,}. "
                f"Finest: {finest_cells:,} cells.",
                'yellow')

        # Store for resource estimator
        self._last_mesh_plan = mesh_plan

    # ------------------------------------------------------------------
    # RATIO CHECKER
    # ------------------------------------------------------------------

    def _add_ratio_grid(self):
        n = self._tbl_ratio_check.rowCount()
        if n >= 6:
            return
        self._tbl_ratio_check.setRowCount(n + 1)
        lbl = QTableWidgetItem(f"Grid {n+1}")
        lbl.setFlags(lbl.flags() & ~Qt.ItemIsEditable)
        self._tbl_ratio_check.setItem(n, 0, lbl)

    def _rem_ratio_grid(self):
        n = self._tbl_ratio_check.rowCount()
        if n <= 2:
            return
        self._tbl_ratio_check.setRowCount(n - 1)

    def _check_ratios(self):
        """Read user-entered cell counts and compute refinement ratios."""
        dim = self._cmb_plan_dim.currentData()
        counts = []
        for i in range(self._tbl_ratio_check.rowCount()):
            item = self._tbl_ratio_check.item(i, 1)
            if item and item.text().strip():
                try:
                    val = float(item.text().replace(",", ""))
                    if val > 0 and np.isfinite(val):
                        counts.append((i, val))
                except ValueError:
                    pass

        if len(counts) < 2:
            self._guidance_ratios.set_guidance(
                "Enter at least 2 cell counts to check ratios.", 'yellow')
            return

        # Sort by cell count descending (finest first)
        counts.sort(key=lambda x: x[1], reverse=True)

        n_pairs = len(counts) - 1
        self._ratio_results_table.setRowCount(n_pairs)
        all_ok = True
        target_r = self._spn_target_r.value()

        for j in range(n_pairs):
            i_fine, n_fine = counts[j]
            i_coarse, n_coarse = counts[j + 1]
            r = compute_refinement_ratio(n_fine, n_coarse, dim)

            pair_label = f"Grid {i_fine+1} \u2192 Grid {i_coarse+1}"
            r_str = f"{r:.4f}"

            if 1.3 <= r <= 2.0:
                assess = "\u2713 Ideal"
                color = DARK_COLORS['green']
            elif 1.1 <= r < 1.3:
                assess = "\u26A0 Too close"
                color = DARK_COLORS['yellow']
                all_ok = False
            elif 2.0 < r <= 3.0:
                assess = "\u26A0 Large gap"
                color = DARK_COLORS['yellow']
            elif r > 3.0:
                assess = "\u2717 Too large"
                color = DARK_COLORS['red']
                all_ok = False
            else:
                assess = "\u2717 Too small"
                color = DARK_COLORS['red']
                all_ok = False

            # Suggested optimal count for the coarser grid
            suggested = n_fine / (target_r ** dim)
            sugg_str = f"{suggested:,.0f}"

            for col, (text, clr) in enumerate([
                (pair_label, None),
                (r_str, None),
                (assess, color),
                (sugg_str, DARK_COLORS['fg_dim']),
            ]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if clr:
                    item.setForeground(QColor(clr))
                self._ratio_results_table.setItem(j, col, item)

        if all_ok:
            self._guidance_ratios.set_guidance(
                "All refinement ratios are in the ideal range "
                "(1.3 \u2264 r \u2264 2.0).", 'green')
        else:
            self._guidance_ratios.set_guidance(
                "Some ratios are outside the ideal range. See the "
                "'Suggested Count' column for recommended cell counts.",
                'yellow')

    # ------------------------------------------------------------------
    # RESOURCE ESTIMATOR
    # ------------------------------------------------------------------

    def _estimate_resources(self):
        """Estimate computational resources for the mesh study."""
        if not hasattr(self, '_last_mesh_plan') or not self._last_mesh_plan:
            # Auto-compute targets first
            self._compute_targets()
        if not hasattr(self, '_last_mesh_plan') or not self._last_mesh_plan:
            return

        baseline_time = self._spn_baseline_time.value()
        scaling = self._cmb_scaling.currentData()
        plan = self._last_mesh_plan
        prod_cells = int(self._spn_prod_cells.value())

        self._resource_table.setRowCount(len(plan))
        total_time = 0.0

        for i, (label, cells, is_prod) in enumerate(plan):
            ratio = cells / max(prod_cells, 1)
            if scaling == "linear":
                time_mult = ratio
            elif scaling == "n15":
                time_mult = ratio ** 1.5
            else:  # n2
                time_mult = ratio ** 2.0

            est_time = baseline_time * time_mult
            total_time += est_time

            # Memory estimate: ~1 KB per cell (rough CFD estimate)
            mem_gb = cells * 1e-6  # ~1 KB/cell = 1e-6 GB/cell
            if mem_gb < 1.0:
                mem_str = f"{mem_gb*1000:.0f} MB"
            else:
                mem_str = f"{mem_gb:.1f} GB"

            if est_time < 1.0:
                time_str = f"{est_time*60:.0f} min"
            else:
                time_str = f"{est_time:.1f} hours"

            for col, (text, color) in enumerate([
                (label, DARK_COLORS['accent'] if is_prod else None),
                (f"{cells:,}", None),
                (time_str, None),
                (f"{time_mult:.2f}x", None),
                (mem_str, None),
            ]):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if color:
                    item.setForeground(QColor(color))
                if is_prod:
                    f = item.font()
                    f.setBold(True)
                    item.setFont(f)
                self._resource_table.setItem(i, col, item)

        if total_time < 1.0:
            total_str = f"{total_time*60:.0f} minutes"
        else:
            total_str = f"{total_time:.1f} hours"
        cost_mult = total_time / max(baseline_time, 0.001)

        self._resource_summary.setText(
            f"<b>Total estimated study time:</b> {total_str} "
            f"({cost_mult:.1f}x your single production run)<br>"
            f"<b>Scaling model:</b> "
            f"{self._cmb_scaling.currentText()}<br><br>"
            f"<i>Note: Estimates assume similar convergence rates "
            f"across all grids. Actual times may vary due to solver "
            f"behavior, parallel efficiency, and I/O.</i>")

    # ------------------------------------------------------------------
    # CHECKLIST
    # ------------------------------------------------------------------

    def _check_all_items(self):
        for cb in self._checklist_boxes.values():
            cb.setChecked(True)
        self._update_checklist_status()

    def _uncheck_all_items(self):
        for cb in self._checklist_boxes.values():
            cb.setChecked(False)
        self._update_checklist_status()

    def _update_checklist_status(self):
        total = len(self._checklist_boxes)
        checked = sum(1 for cb in self._checklist_boxes.values()
                      if cb.isChecked())
        if checked == total:
            self._lbl_checklist_status.setText(
                f"\u2713 All {total} items checked")
            self._lbl_checklist_status.setStyleSheet(
                f"color: {DARK_COLORS['green']}; font-weight: bold;")
        else:
            self._lbl_checklist_status.setText(
                f"{checked}/{total} items checked")
            self._lbl_checklist_status.setStyleSheet(
                f"color: {DARK_COLORS['fg_dim']};")

    # ------------------------------------------------------------------
    # QUICK REFERENCE CARD
    # ------------------------------------------------------------------

    def _build_quick_ref_html(self):
        c = DARK_COLORS
        return f"""
        <html>
        <head><style>
            body {{ background-color: {c['bg']}; color: {c['fg']};
                   font-family: {','.join(FONT_FAMILIES)}; font-size: 12px;
                   padding: 12px; }}
            h2 {{ color: {c['accent']}; margin: 16px 0 8px 0; }}
            h3 {{ color: {c['fg']}; margin: 12px 0 6px 0; }}
            table {{ border-collapse: collapse; width: 100%;
                     margin: 8px 0; }}
            th, td {{ border: 1px solid {c['border']}; padding: 6px 10px;
                      text-align: left; }}
            th {{ background-color: {c['bg_alt']}; color: {c['accent']};
                  font-weight: bold; }}
            td {{ background-color: {c['bg_widget']}; }}
            .green {{ color: {c['green']}; }}
            .yellow {{ color: {c['yellow']}; }}
            .red {{ color: {c['red']}; }}
            ul {{ margin: 4px 0 8px 20px; }}
            li {{ margin: 2px 0; }}
            .formula {{ background-color: {c['bg_input']};
                        padding: 8px; border-radius: 4px;
                        font-family: Consolas, monospace;
                        margin: 4px 0; }}
        </style></head>
        <body>

        <h2>GCI Study Quick Reference</h2>

        <h3>When to Use 2, 3, or 4+ Grids</h3>
        <table>
        <tr><th>Grids</th><th>What You Get</th><th>Fs</th>
            <th>Recommendation</th></tr>
        <tr><td>2</td>
            <td>GCI with <i>assumed</i> order</td>
            <td>3.0</td>
            <td class="yellow">Minimum viable — not for publication</td></tr>
        <tr><td>3</td>
            <td>GCI with <i>computed</i> order + FS method</td>
            <td>1.25</td>
            <td class="green">Standard procedure (recommended)</td></tr>
        <tr><td>4+</td>
            <td>Cross-checking + LSR method</td>
            <td>1.25</td>
            <td class="green">Best practice for critical work</td></tr>
        </table>

        <h3>Refinement Ratio Guidelines</h3>
        <table>
        <tr><th>Ratio</th><th>Assessment</th></tr>
        <tr><td>r &lt; 1.1</td>
            <td class="red">Too small — grids too similar, noisy GCI</td></tr>
        <tr><td>1.1 &le; r &lt; 1.3</td>
            <td class="yellow">Marginal — may work but not reliable</td></tr>
        <tr><td>1.3 &le; r &le; 2.0</td>
            <td class="green">Ideal range (Celik et al. recommend r &gt; 1.3)
            </td></tr>
        <tr><td>r &gt; 2.0</td>
            <td class="yellow">Large gap — acceptable if consistent</td></tr>
        </table>

        <h3>Common Pitfalls</h3>
        <ul>
        <li>Only running 2 grids instead of 3</li>
        <li>Confusing cell count with mesh spacing (h)</li>
        <li>Not checking iterative convergence on every grid</li>
        <li>Using non-systematic refinement (random cell counts)</li>
        <li>Extracting data at singularities or sharp gradients</li>
        <li>Different solver settings between grids</li>
        <li>y+ out of valid range on some grids</li>
        </ul>

        <h3>Key Formulas</h3>
        <div class="formula">
        <b>Representative spacing:</b> h = (1/N)<sup>1/dim</sup><br><br>
        <b>Refinement ratio:</b> r = h_coarse / h_fine
            = (N_fine / N_coarse)<sup>1/dim</sup><br><br>
        <b>Observed order:</b> p = ln((f3-f2)/(f2-f1)) / ln(r)<br><br>
        <b>Richardson extrapolation:</b>
            f_exact = f1 + (f1-f2) / (r<sup>p</sup> - 1)<br><br>
        <b>GCI:</b> GCI = Fs &middot; |e| / (r<sup>p</sup> - 1)<br><br>
        <b>u_num = GCI &middot; |f1| / Fs = |f1 - f_exact|</b>
        </div>

        <h3>Standards References</h3>
        <ul>
        <li>Celik et al. (2008) J. Fluids Eng. 130(7), 078001</li>
        <li>Roache (1998) Verification and Validation in CFD</li>
        <li>ASME V&amp;V 20-2009 (R2021) Section 5.1</li>
        <li>ITTC 7.5-03-01-01 (2024) CFD Uncertainty Analysis</li>
        </ul>

        </body></html>
        """

    # ------------------------------------------------------------------
    # STATE SERIALIZATION
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Serialize planner state for project save."""
        checklist = {}
        for text, cb in self._checklist_boxes.items():
            checklist[text] = cb.isChecked()
        return {
            "production_cells": self._spn_prod_cells.value(),
            "dim": self._cmb_plan_dim.currentData(),
            "target_r": self._spn_target_r.value(),
            "study_size_index": self._cmb_study_size.currentIndex(),
            "strategy_index": self._cmb_strategy.currentIndex(),
            "baseline_time": self._spn_baseline_time.value(),
            "scaling_index": self._cmb_scaling.currentIndex(),
            "checklist": checklist,
        }

    def set_state(self, state: dict):
        """Restore planner state from project load."""
        self._spn_prod_cells.setValue(
            state.get("production_cells", 1000000))
        dim = state.get("dim", 3)
        self._cmb_plan_dim.setCurrentIndex(0 if dim == 3 else 1)
        self._spn_target_r.setValue(state.get("target_r", 1.5))
        idx = state.get("study_size_index", 0)
        if 0 <= idx < self._cmb_study_size.count():
            self._cmb_study_size.setCurrentIndex(idx)
        idx = state.get("strategy_index", 0)
        if 0 <= idx < self._cmb_strategy.count():
            self._cmb_strategy.setCurrentIndex(idx)
        self._spn_baseline_time.setValue(
            state.get("baseline_time", 1.0))
        idx = state.get("scaling_index", 1)
        if 0 <= idx < self._cmb_scaling.count():
            self._cmb_scaling.setCurrentIndex(idx)

        checklist = state.get("checklist", {})
        for text, cb in self._checklist_boxes.items():
            cb.setChecked(checklist.get(text, False))
        self._update_checklist_status()

    def clear_all(self):
        """Reset planner to defaults."""
        self._spn_prod_cells.setValue(1000000)
        self._cmb_plan_dim.setCurrentIndex(0)
        self._spn_target_r.setValue(1.5)
        self._cmb_study_size.setCurrentIndex(0)
        self._cmb_strategy.setCurrentIndex(0)
        self._spn_baseline_time.setValue(1.0)
        self._cmb_scaling.setCurrentIndex(1)
        self._targets_table.setRowCount(0)
        self._ratio_results_table.setRowCount(0)
        self._resource_table.setRowCount(0)
        self._resource_summary.clear()
        self._guidance_targets.set_guidance("", 'green')
        self._guidance_ratios.set_guidance("", 'green')
        for cb in self._checklist_boxes.values():
            cb.setChecked(False)
        self._update_checklist_status()


class GCIReportGenerator:
    """Generate a self-contained HTML report for GCI Calculator results.

    Follows the same architecture as HTMLReportGenerator in
    vv20_validation_tool.py: modular section builders, base64-embedded
    figures, professional light-theme CSS for print.
    """

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def generate_report(
        self,
        results: List[GCIResult],
        quantity_names: List[str],
        quantity_units: List[str],
        fs_results: List[Optional[dict]],
        lsr_results: List[Optional[dict]],
        convergence_fig=None,
        settings: Optional[dict] = None,
    ) -> str:
        """Build and return a complete self-contained HTML report."""
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if settings is None:
            settings = {}
        consequence = normalize_decision_consequence(
            settings.get("decision_consequence", "Medium")
        )

        sections: List[str] = []

        # 1. Header
        sections.append(self._build_header(now_str))

        # 2. Build TOC
        has_methods = (any(r is not None for r in fs_results) or
                       any(r is not None for r in lsr_results))
        toc = [
            ("section-config", "1. Input Configuration"),
            ("section-results", "2. Results Per Quantity"),
            ("section-plot", "3. Convergence Plot"),
        ]
        n = 3
        if has_methods:
            n += 1
            toc.append(("section-methods", f"{n}. Method Comparison"))
        n += 1; toc.append(("section-report", f"{n}. Report Statements"))
        n += 1; toc.append(("section-checklist", f"{n}. Reviewer Checklist"))
        n += 1; toc.append(("section-caveats", f"{n}. Limitations &amp; Caveats"))
        n += 1; toc.append(("section-uqmap", f"{n}. UQ Mapping Assumption"))
        n += 1; toc.append(("section-credibility", f"{n}. Credibility Framing"))
        n += 1; toc.append(("section-vvuq-glossary", f"{n}. VVUQ Terminology Panel"))
        n += 1; toc.append(("section-conformity", f"{n}. Conformity Assessment Template"))
        n += 1; toc.append(("section-refs", f"{n}. References"))
        sections.append(self._build_toc(toc))

        # 3+. Content sections
        sections.append(self._build_config_section(settings, results,
                                                   quantity_names, quantity_units))
        sections.append(self._build_results_section(results, quantity_names,
                                                    quantity_units))
        sections.append(self._build_plot_section(convergence_fig))
        if has_methods:
            sections.append(self._build_method_comparison(
                results, fs_results, lsr_results, quantity_names, quantity_units))
        sections.append(self._build_report_statements(
            results, quantity_names, quantity_units))
        sections.append(self._build_decision_card(results, quantity_names, quantity_units))
        sections.append(self._build_checklist(results, quantity_names))
        sections.append(self._build_caveats(results, quantity_names))
        sections.append(self._build_uq_mapping_section())
        sections.append(self._build_credibility_section(results, settings, consequence))
        sections.append(render_vvuq_glossary_html())
        sections.append(self._build_conformity_section(results, quantity_names, consequence))
        sections.append(self._build_references(fs_results, lsr_results))
        sections.append(self._build_footer(now_str))

        return self._wrap_html("\n".join(sections))

    def save_report(self, html_content: str, filepath: str) -> None:
        """Write the HTML report string to *filepath*."""
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(html_content)

    # -----------------------------------------------------------------
    # Image embedding helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _figure_to_base64(fig) -> str:
        """Render a matplotlib Figure to a PNG data-URI with light theme.

        Temporarily applies REPORT_PLOT_STYLE for print-ready appearance,
        then restores original dark-theme properties.
        """
        # Save original properties
        orig_props = []
        for ax in fig.get_axes():
            orig_props.append({
                'facecolor': ax.get_facecolor(),
                'title_color': ax.title.get_color() if ax.title else None,
                'xlabel_color': ax.xaxis.label.get_color(),
                'ylabel_color': ax.yaxis.label.get_color(),
                'spine_colors': {s: ax.spines[s].get_edgecolor() for s in ax.spines},
            })
            ax.set_facecolor(REPORT_PLOT_STYLE['axes.facecolor'])
            ax.title.set_color(REPORT_PLOT_STYLE['text.color'])
            ax.xaxis.label.set_color(REPORT_PLOT_STYLE['axes.labelcolor'])
            ax.yaxis.label.set_color(REPORT_PLOT_STYLE['axes.labelcolor'])
            ax.tick_params(axis='x', colors=REPORT_PLOT_STYLE['xtick.color'])
            ax.tick_params(axis='y', colors=REPORT_PLOT_STYLE['ytick.color'])
            for spine in ax.spines.values():
                spine.set_edgecolor(REPORT_PLOT_STYLE['axes.edgecolor'])
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor(REPORT_PLOT_STYLE['legend.facecolor'])
                leg.get_frame().set_edgecolor(REPORT_PLOT_STYLE['legend.edgecolor'])
                for text in leg.get_texts():
                    text.set_color(REPORT_PLOT_STYLE['text.color'])

        orig_fig_fc = fig.get_facecolor()
        fig.set_facecolor('#ffffff')

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        buf.close()

        # Restore original dark-theme properties
        fig.set_facecolor(orig_fig_fc)
        for ax, props in zip(fig.get_axes(), orig_props):
            ax.set_facecolor(props['facecolor'])
            if props['title_color']:
                ax.title.set_color(props['title_color'])
            ax.xaxis.label.set_color(props['xlabel_color'])
            ax.yaxis.label.set_color(props['ylabel_color'])
            ax.tick_params(axis='x', colors=PLOT_STYLE['xtick.color'])
            ax.tick_params(axis='y', colors=PLOT_STYLE['ytick.color'])
            for s_name, s_color in props['spine_colors'].items():
                ax.spines[s_name].set_edgecolor(s_color)
            leg = ax.get_legend()
            if leg:
                leg.get_frame().set_facecolor(PLOT_STYLE['legend.facecolor'])
                leg.get_frame().set_edgecolor(PLOT_STYLE['legend.edgecolor'])
                for text in leg.get_texts():
                    text.set_color(PLOT_STYLE['text.color'])

        return f"data:image/png;base64,{encoded}"

    @staticmethod
    def _embed_figure(fig, alt: str = "figure") -> str:
        """Return an <img> tag for the figure, or empty string."""
        if fig is None:
            return ""
        uri = GCIReportGenerator._figure_to_base64(fig)
        safe_alt = _esc(alt)
        return (
            f'<div class="figure-container">'
            f'<img src="{uri}" alt="{safe_alt}" '
            f'style="max-width:100%; height:auto;" />'
            f'</div>\n'
        )

    # -----------------------------------------------------------------
    # Section builders
    # -----------------------------------------------------------------
    def _build_header(self, now_str: str) -> str:
        return textwrap.dedent(f"""\
        <div class="header-block">
            <h1 class="report-title">GCI Calculator v{_esc(APP_VERSION)} &mdash; Grid Convergence Report</h1>
            <div class="report-meta">
                Generated: {_esc(now_str)}<br/>
                Tool: GCI Calculator v{_esc(APP_VERSION)}<br/>
                Standards: ASME V&amp;V 20-2009 (R2021) &bull; Celik et al. (2008)
            </div>
        </div>
        <hr class="header-rule"/>
        """)

    def _build_toc(self, entries: List[Tuple[str, str]]) -> str:
        items = "\n".join(
            f'        <li><a href="#{anchor}">{label}</a></li>'
            for anchor, label in entries
        )
        return textwrap.dedent(f"""\
        <div class="toc">
            <h2>Table of Contents</h2>
            <ol>
        {items}
            </ol>
        </div>
        """)

    def _build_config_section(self, settings: dict, results: List[GCIResult],
                              qty_names: List[str], qty_units: List[str]) -> str:
        n_grids = settings.get("n_grids", results[0].n_grids if results else 0)
        dim = settings.get("dim", "3D")
        fs = settings.get("safety_factor", 0.0)
        fs_str = f"{fs:.2f}" if fs > 0 else "Auto"
        p_th = settings.get("theoretical_order", 2.0)
        ref = settings.get("reference_scale", 0.0)
        ref_str = f"{ref:.4g}" if ref > 0 else "Auto (|f1|)"
        prod = settings.get("production_grid", 0)
        prod_str = f"Grid {prod + 1}" if prod > 0 else "Grid 1 (finest)"
        program = settings.get("program", "—")
        analyst = settings.get("analyst", "—")
        date = settings.get("date", "—")
        consequence = normalize_decision_consequence(
            settings.get("decision_consequence", "Medium")
        )

        qty_rows = ""
        for i, (name, unit) in enumerate(zip(qty_names, qty_units)):
            unit_str = f" [{_esc(unit)}]" if unit else ""
            qty_rows += f"<tr><td>Quantity {i+1}</td><td>{_esc(name)}{unit_str}</td></tr>\n"

        return textwrap.dedent(f"""\
        <div class="section" id="section-config">
            <h2>1. Input Configuration</h2>
            <table>
                <tr><th>Parameter</th><th>Value</th></tr>
                <tr><td>Number of Grids</td><td>{n_grids}</td></tr>
                <tr><td>Dimension</td><td>{_esc(str(dim))}</td></tr>
                <tr><td>Theoretical Order (p<sub>th</sub>)</td><td>{p_th:.1f}</td></tr>
                <tr><td>Safety Factor (F<sub>s</sub>)</td><td>{fs_str}</td></tr>
                <tr><td>Reference Scale</td><td>{ref_str}</td></tr>
                <tr><td>Production Grid</td><td>{prod_str}</td></tr>
                <tr><td>Program / Project</td><td>{_esc(program)}</td></tr>
                <tr><td>Analyst</td><td>{_esc(analyst)}</td></tr>
                <tr><td>Date</td><td>{_esc(date)}</td></tr>
                <tr><td>Decision Consequence</td><td>{_esc(consequence)}</td></tr>
                {qty_rows}
            </table>
        </div>
        """)

    def _build_results_section(self, results: List[GCIResult],
                               qty_names: List[str],
                               qty_units: List[str]) -> str:
        parts = ['<div class="section" id="section-results">',
                 '<h2>2. Results Per Quantity</h2>']

        for q_idx, res in enumerate(results):
            q_name = qty_names[q_idx] if q_idx < len(qty_names) else f"Qty {q_idx+1}"
            q_unit = qty_units[q_idx] if q_idx < len(qty_units) else ""
            unit_sfx = f" [{_esc(q_unit)}]" if q_unit else ""

            # Convergence type badge
            ct = res.convergence_type
            badge_class = ("pass" if ct == "monotonic" else
                           "neutral" if ct in ("oscillatory", "grid-independent") else
                           "fail")
            parts.append(f'<h3>{_esc(q_name)}{unit_sfx}</h3>')
            parts.append(
                f'<div class="verdict {badge_class}">'
                f'Convergence: <strong>{_esc(ct)}</strong> '
                f'(R = {res.convergence_ratio:.4f}, '
                f'Method: {_esc(res.method)})</div>')

            # Celik Table 1
            parts.append('<table>')
            parts.append('<tr><th>Parameter</th><th>Value</th></tr>')

            # Cell counts
            if res.grid_cells:
                for i, nc in enumerate(res.grid_cells[:min(res.n_grids, 6)]):
                    parts.append(
                        f'<tr><td>N<sub>{i+1}</sub> (cells)</td>'
                        f'<td>{nc:,.0f}</td></tr>')

            # Refinement ratios
            if res.refinement_ratios:
                for i, r in enumerate(res.refinement_ratios):
                    parts.append(
                        f'<tr><td>r<sub>{i+1}{i+2}</sub></td>'
                        f'<td>{r:.4f}</td></tr>')

            # Solutions
            if res.grid_solutions:
                for i, f in enumerate(res.grid_solutions[:min(res.n_grids, 6)]):
                    parts.append(
                        f'<tr><td>&phi;<sub>{i+1}</sub> (solution)</td>'
                        f'<td>{f:.6g}</td></tr>')

            # Observed order
            if not np.isnan(res.observed_order) and res.observed_order < 1e6:
                p_label = "p (assumed order)" if res.order_is_assumed else "p (observed order)"
                parts.append(
                    f'<tr><td>{_esc(p_label)}</td>'
                    f'<td>{res.observed_order:.4f}</td></tr>')

            # Richardson extrapolation
            if not np.isnan(res.richardson_extrapolation):
                parts.append(
                    f'<tr><td>&phi;<sub>ext</sub> (RE)</td>'
                    f'<td>{res.richardson_extrapolation:.6g}</td></tr>')

            # Relative errors
            if not np.isnan(res.e21_rel):
                parts.append(
                    f'<tr><td>e<sub>a</sub><sup>21</sup> (approx. rel. error)</td>'
                    f'<td>{res.e21_rel * 100:.4f}%</td></tr>')

            if not np.isnan(res.richardson_extrapolation):
                e_ext = abs(
                    (res.richardson_extrapolation - res.grid_solutions[0])
                    / res.richardson_extrapolation
                ) if abs(res.richardson_extrapolation) > 1e-30 else 0.0
                parts.append(
                    f'<tr><td>e<sub>ext</sub><sup>21</sup> (extrapolated rel.)</td>'
                    f'<td>{e_ext * 100:.4f}%</td></tr>')

            # GCI
            if not np.isnan(res.gci_fine):
                parts.append(
                    f'<tr><td>GCI<sub>fine</sub><sup>21</sup></td>'
                    f'<td>{res.gci_fine * 100:.4f}%</td></tr>')

            # u_num
            if not np.isnan(res.u_num):
                parts.append(
                    f'<tr class="total-row"><td>u<sub>num</sub> (1&sigma;)</td>'
                    f'<td>{res.u_num:.4g} ({res.u_num_pct:.2f}%)</td></tr>')

            # Safety factor and asymptotic ratio
            parts.append(
                f'<tr><td>F<sub>s</sub></td>'
                f'<td>{res.safety_factor:.2f}</td></tr>')
            if not np.isnan(res.asymptotic_ratio):
                parts.append(
                    f'<tr><td>Asymptotic Ratio</td>'
                    f'<td>{res.asymptotic_ratio:.4f}</td></tr>')

            parts.append('</table>')

            # Production grid note
            if (res.target_grid_idx > 0 and res.per_grid_u_num and
                    res.target_grid_idx < len(res.per_grid_u_num) and
                    not np.isnan(res.per_grid_u_num[res.target_grid_idx])):
                tgt = res.target_grid_idx
                parts.append(
                    f'<div class="highlight">'
                    f'<strong>Production Grid {tgt+1}:</strong> '
                    f'u<sub>num</sub> = {res.per_grid_u_num[tgt]:.4g} '
                    f'({res.per_grid_u_pct[tgt]:.2f}%)</div>')

        parts.append('</div>')
        return "\n".join(parts)

    def _build_plot_section(self, fig) -> str:
        if fig is None:
            return ""
        return textwrap.dedent(f"""\
        <div class="section" id="section-plot">
            <h2>3. Convergence Plot</h2>
            {self._embed_figure(fig, "Grid Convergence Plot")}
        </div>
        """)

    def _build_method_comparison(self, results: List[GCIResult],
                                 fs_results: List[Optional[dict]],
                                 lsr_results: List[Optional[dict]],
                                 qty_names: List[str],
                                 qty_units: List[str]) -> str:
        has_fs = any(r is not None and r.get("is_valid") for r in fs_results)
        has_lsr = any(r is not None and r.get("is_valid") for r in lsr_results)

        parts = ['<div class="section" id="section-methods">',
                 '<h2>Method Comparison</h2>',
                 '<p>u<sub>num</sub> values &mdash; fine grid, 1&sigma;:</p>',
                 '<table>']

        # Header row
        hdr = '<tr><th>Quantity</th><th>GCI</th>'
        if has_fs:
            hdr += '<th>FS variant</th>'
        if has_lsr:
            hdr += '<th>LSR variant</th>'
        hdr += '</tr>'
        parts.append(hdr)

        for q_idx, res in enumerate(results):
            q_name = qty_names[q_idx] if q_idx < len(qty_names) else f"Qty {q_idx+1}"
            u_gci = res.u_num

            row = f'<tr><td>{_esc(q_name)}</td><td>{u_gci:.4g}</td>'

            if has_fs:
                fs_r = fs_results[q_idx] if q_idx < len(fs_results) else None
                if fs_r and fs_r.get("is_valid"):
                    row += f'<td>{fs_r["u_num"]:.4g}</td>'
                else:
                    row += '<td>N/A</td>'

            if has_lsr:
                lsr_r = lsr_results[q_idx] if q_idx < len(lsr_results) else None
                if lsr_r and lsr_r.get("is_valid"):
                    row += f'<td>{lsr_r["u_num"]:.4g}</td>'
                else:
                    row += '<td>N/A</td>'

            row += '</tr>'
            parts.append(row)

        parts.append('</table>')

        # Per-method detail
        for q_idx, res in enumerate(results):
            q_name = qty_names[q_idx] if q_idx < len(qty_names) else f"Qty {q_idx+1}"
            q_unit = qty_units[q_idx] if q_idx < len(qty_units) else ""
            unit_sfx = f" {_esc(q_unit)}" if q_unit else ""

            if has_fs and q_idx < len(fs_results):
                fs_r = fs_results[q_idx]
                if fs_r and fs_r.get("is_valid"):
                    parts.append(
                        f'<p><strong>FS variant (Xing &amp; Stern 2010) &mdash; '
                        f'{_esc(q_name)}:</strong> '
                        f'u<sub>num</sub> = {fs_r["u_num"]:.4g}{unit_sfx} '
                        f'({fs_r["u_num_pct"]:.2f}%)<br/>'
                        f'<em>{_esc(fs_r.get("note", ""))}</em></p>')

            if has_lsr and q_idx < len(lsr_results):
                lsr_r = lsr_results[q_idx]
                if lsr_r and lsr_r.get("is_valid"):
                    parts.append(
                        f'<p><strong>LSR variant with AICc (Eca &amp; Hoekstra 2014) '
                        f'&mdash; {_esc(q_name)}:</strong> '
                        f'u<sub>num</sub> = {lsr_r["u_num"]:.4g}{unit_sfx} '
                        f'({lsr_r["u_num_pct"]:.2f}%)<br/>'
                        f'<em>{_esc(lsr_r.get("note", ""))}</em></p>')

        parts.append('</div>')
        return "\n".join(parts)

    def _build_report_statements(self, results: List[GCIResult],
                                 qty_names: List[str],
                                 qty_units: List[str]) -> str:
        """Render the certification-ready report statement paragraphs."""
        parts = ['<div class="section" id="section-report">',
                 '<h2>Report Statements</h2>',
                 '<p><em>Ready-to-paste text for V&amp;V reports and '
                 'certification documentation:</em></p>']

        for q_idx, res in enumerate(results):
            q_name = qty_names[q_idx] if q_idx < len(qty_names) else f"Qty {q_idx+1}"
            q_unit = qty_units[q_idx] if q_idx < len(qty_units) else ""
            unit_sfx = f" {_esc(q_unit)}" if q_unit else ""
            u_val = res.u_num
            u_pct = res.u_num_pct

            # Use production grid values if applicable
            tgt = res.target_grid_idx
            if (res.per_grid_u_num and 0 <= tgt < len(res.per_grid_u_num)
                    and not np.isnan(res.per_grid_u_num[tgt])):
                u_val = res.per_grid_u_num[tgt]
                u_pct = res.per_grid_u_pct[tgt]

            parts.append(f'<h3>{_esc(q_name)}</h3>')

            if res.convergence_type == "monotonic":
                n_fine = res.grid_cells[0] if res.grid_cells else 0
                n_coarse = res.grid_cells[-1] if res.grid_cells else 0
                r_ratios = ", ".join(f"{r:.3f}" for r in res.refinement_ratios)
                parts.append(
                    f'<div class="findings-block">'
                    f'A grid convergence study was performed using '
                    f'{res.n_grids} systematically refined grids '
                    f'({n_fine:,.0f} to {n_coarse:,.0f} cells, '
                    f'refinement ratio(s): {r_ratios}). '
                    f'Monotonic convergence was observed with a '
                    f'convergence ratio R = {res.convergence_ratio:.4f}. '
                    f'The {"assumed" if res.order_is_assumed else "observed"}'
                    f' order of accuracy '
                    f'p = {res.observed_order:.2f} '
                    f'(theoretical: {res.theoretical_order:.1f}). '
                    f'Using the Grid Convergence Index (GCI) method '
                    f'with a safety factor Fs = {res.safety_factor:.2f}, '
                    f'the numerical uncertainty for {_esc(q_name)} is '
                    f'u_num = {u_val:.4g}{unit_sfx} '
                    f'({u_pct:.2f}% of the solution). '
                    f'The expanded uncertainty (k=2) is '
                    f'&plusmn;{2*u_val:.4g}{unit_sfx} '
                    f'({2*u_pct:.2f}%).'
                    f'</div>')

            elif res.convergence_type == "oscillatory":
                parts.append(
                    f'<div class="findings-block">'
                    f'A grid convergence study was performed using '
                    f'{res.n_grids} grids. Oscillatory convergence '
                    f'was observed (R = {res.convergence_ratio:.4f}). '
                    f'Richardson extrapolation is not reliable in this regime. '
                    f'The numerical uncertainty was estimated from the '
                    f'oscillation range with Fs = {res.safety_factor:.2f}. '
                    f'u_num = {u_val:.4g}{unit_sfx} ({u_pct:.2f}%). '
                    f'The expanded uncertainty (k=2) is '
                    f'&plusmn;{2*u_val:.4g}{unit_sfx} ({2*u_pct:.2f}%).'
                    f'</div>')

            elif res.convergence_type == "divergent":
                parts.append(
                    f'<div class="verdict fail">'
                    f'DIVERGENT behavior observed. The Grid Convergence '
                    f'Index does not yield a valid numerical uncertainty '
                    f'estimate for {_esc(q_name)}. Additional grid refinement '
                    f'or investigation is required.'
                    f'</div>')

            elif res.method.startswith("2-grid"):
                parts.append(
                    f'<div class="findings-block">'
                    f'A 2-grid convergence study was performed for '
                    f'{_esc(q_name)} using an assumed order of accuracy '
                    f'p = {res.theoretical_order:.1f}. '
                    f'u_num = {u_val:.4g}{unit_sfx} ({u_pct:.2f}%). '
                    f'NOTE: A 2-grid study cannot determine the observed '
                    f'order or verify asymptotic convergence. A 3+ grid '
                    f'study is recommended for certification.'
                    f'</div>')

            else:
                parts.append(
                    f'<div class="findings-block">'
                    f'Grid convergence type: {_esc(res.convergence_type)}. '
                    f'u_num = {u_val:.4g}{unit_sfx} ({u_pct:.2f}%).'
                    f'</div>')

        parts.append('</div>')
        return "\n".join(parts)

    def _build_decision_card(self, results: List[GCIResult],
                             qty_names: List[str],
                             qty_units: List[str]) -> str:
        """Render novice-first carry-over decision card."""
        if not results:
            return render_decision_card_html(
                title="Decision Card (Carry to Aggregator)",
                use_value="No results available",
                use_distribution="No results available",
                use_combination="No results available",
                stop_checks=["Run a valid GCI analysis first"],
                notes="No computable quantities found.",
            )

        valid_results = [(i, r) for i, r in enumerate(results) if r.is_valid and np.isfinite(r.u_num)]
        if not valid_results:
            return render_decision_card_html(
                title="Decision Card (Carry to Aggregator)",
                use_value="No valid u_num available (all quantities divergent or invalid)",
                use_distribution="N/A",
                use_combination="N/A",
                stop_checks=[
                    "STOP: All quantities have divergent or invalid convergence — resolve before carry-over",
                ],
                notes="Re-run with refined grids or check solver convergence before proceeding.",
            )

        best_vi, best_res = max(valid_results, key=lambda x: x[1].u_num)
        best_name = qty_names[best_vi] if best_vi < len(qty_names) else f"Quantity {best_vi+1}"
        best_unit = qty_units[best_vi] if best_vi < len(qty_units) else ""
        unit_sfx = f" {best_unit}" if best_unit else ""

        return render_decision_card_html(
            title="Decision Card (Carry to Aggregator)",
            use_value=(
                "Use u_num (1 sigma) from Section 2 per quantity. "
                f"Largest current value: {best_name} = {best_res.u_num:.4g}{unit_sfx}."
            ),
            use_distribution="Normal",
            use_combination="RSS",
            stop_checks=[
                "Any quantity with divergent convergence behavior",
                "Any quantity missing valid u_num estimate",
                "Any quantity with unresolved unit mismatch",
                "Any quantity with unresolved oscillatory/divergent warning",
            ],
            notes="For production-grid studies, use the production-grid carry value shown in Section 2.",
        )

    def _build_credibility_section(self, results: List[GCIResult],
                                   settings: dict,
                                   consequence: str) -> str:
        """Deterministic credibility framing section."""
        program = str(settings.get("program", "")).strip()
        analyst = str(settings.get("analyst", "")).strip()
        units = [u for u in settings.get("quantity_units", []) if u]
        if not units:
            units = [u for u in settings.get("qty_units", []) if u]
        if not units:
            # Fallback: infer from caller quantity units if not stored in settings.
            units = []

        evidence = {
            'inputs_documented': bool(program) and bool(analyst),
            'method_selected': True,
            'units_consistent': len(set(units)) <= 1 if units else True,
            'data_quality': len(results) > 0 and all(r.n_grids >= 3 for r in results),
            'diagnostics_pass': len(results) > 0 and all(r.convergence_type != "divergent" for r in results),
            'independent_review': str(settings.get("independent_review", "")).strip().lower() in ("1", "true", "yes", "y"),
            'conservative_bound': len(results) > 0 and all(r.safety_factor >= 1.25 for r in results),
            'validation_plan': len(results) > 0 and all(r.is_valid for r in results),
        }
        return render_credibility_html(consequence, evidence)

    def _build_conformity_section(self, results: List[GCIResult],
                                  qty_names: List[str],
                                  consequence: str) -> str:
        """Optional conformity wording template section."""
        if not results:
            return render_conformity_template_html(
                metric_name="Numerical uncertainty u_num",
                metric_value="No valid result",
                consequence=consequence,
            )
        best_idx = int(np.nanargmax([r.u_num if np.isfinite(r.u_num) else -1.0 for r in results]))
        best_name = qty_names[best_idx] if best_idx < len(qty_names) else f"Quantity {best_idx+1}"
        best_val = results[best_idx].u_num
        return render_conformity_template_html(
            metric_name=f"u_num (1 sigma) - {best_name}",
            metric_value=f"{best_val:.6g}",
            consequence=consequence,
        )

    def _build_checklist(self, results: List[GCIResult],
                         qty_names: List[str]) -> str:
        if not results:
            return ""
        res0 = results[0]
        parts = ['<div class="section" id="section-checklist">',
                 '<h2>Reviewer Checklist</h2>']

        # Number of grids
        if res0.n_grids >= 3:
            parts.append(
                f'<div class="verdict pass">'
                f'<strong>PASS</strong> &mdash; Grids: {res0.n_grids} '
                f'(&ge; 3 for observed order)</div>')
        elif res0.n_grids == 2:
            parts.append(
                f'<div class="verdict neutral">'
                f'<strong>NOTE</strong> &mdash; Grids: 2 '
                f'(assumed order &mdash; consider 3+ grids)</div>')

        # Refinement ratio
        if res0.refinement_ratios:
            r_min = min(res0.refinement_ratios)
            if r_min >= 1.3:
                parts.append(
                    f'<div class="verdict pass">'
                    f'<strong>PASS</strong> &mdash; Refinement ratio: '
                    f'r_min = {r_min:.3f} (&ge; 1.3)</div>')
            else:
                parts.append(
                    f'<div class="verdict neutral">'
                    f'<strong>NOTE</strong> &mdash; Refinement ratio: '
                    f'r_min = {r_min:.3f} (&lt; 1.3 recommended)</div>')

        # Convergence type
        ct = res0.convergence_type
        if ct == "monotonic":
            parts.append(
                '<div class="verdict pass">'
                '<strong>PASS</strong> &mdash; Convergence: monotonic</div>')
        elif ct == "oscillatory":
            parts.append(
                '<div class="verdict neutral">'
                '<strong>NOTE</strong> &mdash; Convergence: oscillatory '
                '(Fs = 3.0 applied)</div>')
        elif ct == "divergent":
            parts.append(
                '<div class="verdict fail">'
                '<strong>FAIL</strong> &mdash; Convergence: DIVERGENT &mdash; '
                'GCI invalid</div>')

        # Observed / assumed order
        p = res0.observed_order
        p_th = res0.theoretical_order
        if res0.order_is_assumed:
            parts.append(
                f'<div class="verdict neutral">'
                f'<strong>NOTE</strong> &mdash; Assumed order: '
                f'p = {p_th:.1f} (theoretical, not computed)</div>')
        elif not np.isnan(p) and p < 1e6:
            ratio_p = abs(p - p_th) / max(p_th, 0.1)
            if ratio_p < 0.3:
                parts.append(
                    f'<div class="verdict pass">'
                    f'<strong>PASS</strong> &mdash; Observed order: '
                    f'p = {p:.3f} (theoretical = {p_th:.1f}, '
                    f'ratio = {p/p_th:.2f})</div>')
            elif ratio_p < 0.5:
                parts.append(
                    f'<div class="verdict neutral">'
                    f'<strong>NOTE</strong> &mdash; Observed order: '
                    f'p = {p:.3f} (ratio = {p/p_th:.2f} &mdash; '
                    f'moderate deviation)</div>')
            else:
                parts.append(
                    f'<div class="verdict fail">'
                    f'<strong>WARN</strong> &mdash; Observed order: '
                    f'p = {p:.3f} (ratio = {p/p_th:.2f} &mdash; '
                    f'significant deviation)</div>')

        # Asymptotic range
        ar = res0.asymptotic_ratio
        if not np.isnan(ar):
            if 0.95 <= ar <= 1.05:
                parts.append(
                    f'<div class="verdict pass">'
                    f'<strong>PASS</strong> &mdash; Asymptotic ratio: '
                    f'{ar:.4f} (within [0.95, 1.05])</div>')
            elif 0.8 <= ar <= 1.2:
                parts.append(
                    f'<div class="verdict neutral">'
                    f'<strong>NOTE</strong> &mdash; Asymptotic ratio: '
                    f'{ar:.4f} (within [0.8, 1.2] &mdash; acceptable)</div>')
            else:
                parts.append(
                    f'<div class="verdict fail">'
                    f'<strong>FAIL</strong> &mdash; Asymptotic ratio: '
                    f'{ar:.4f} (outside [0.8, 1.2])</div>')

        # GCI magnitude
        if not np.isnan(res0.gci_fine):
            gci_pct = res0.gci_fine * 100
            if gci_pct < 2.0:
                parts.append(
                    f'<div class="verdict pass">'
                    f'<strong>PASS</strong> &mdash; GCI_fine: '
                    f'{gci_pct:.3f}% (&lt; 2% &mdash; excellent)</div>')
            elif gci_pct < 5.0:
                parts.append(
                    f'<div class="verdict pass">'
                    f'<strong>PASS</strong> &mdash; GCI_fine: '
                    f'{gci_pct:.3f}% (&lt; 5% &mdash; acceptable)</div>')
            else:
                parts.append(
                    f'<div class="verdict neutral">'
                    f'<strong>NOTE</strong> &mdash; GCI_fine: '
                    f'{gci_pct:.3f}% (&ge; 5% &mdash; consider finer grids)</div>')

        parts.append('</div>')
        return "\n".join(parts)

    def _build_caveats(self, results: List[GCIResult],
                       qty_names: List[str]) -> str:
        parts = ['<div class="section" id="section-caveats">',
                 '<h2>Limitations &amp; Caveats</h2>', '<ul>']

        any_caveats = False
        for q_idx, res in enumerate(results):
            q_name = qty_names[q_idx] if q_idx < len(qty_names) else f"Qty {q_idx+1}"
            if res.refinement_ratios:
                r_min = min(res.refinement_ratios)
                if r_min < 1.3:
                    parts.append(
                        f'<li>Low refinement ratio (r_min = {r_min:.3f} '
                        f'&lt; 1.3 recommended) for {_esc(q_name)}.</li>')
                    any_caveats = True
            if (not np.isnan(res.observed_order) and res.observed_order < 1e6):
                ratio_p = abs(res.observed_order - res.theoretical_order
                              ) / max(res.theoretical_order, 0.1)
                if ratio_p > 0.5:
                    parts.append(
                        f'<li>Observed order p = {res.observed_order:.2f} '
                        f'deviates significantly from theoretical '
                        f'({res.theoretical_order:.1f}) for {_esc(q_name)}.</li>')
                    any_caveats = True
            if res.convergence_type == "oscillatory":
                parts.append(
                    f'<li>Oscillatory convergence for {_esc(q_name)}: '
                    f'Richardson extrapolation is unreliable.</li>')
                any_caveats = True
            if res.convergence_type == "divergent":
                parts.append(
                    f'<li><strong>DIVERGENT</strong> behavior for '
                    f'{_esc(q_name)}: no valid numerical uncertainty.</li>')
                any_caveats = True
            if res.method.startswith("2-grid"):
                parts.append(
                    f'<li>2-grid study for {_esc(q_name)}: assumed order '
                    f'p = {res.theoretical_order:.1f}. A 3+ grid study is '
                    f'recommended for certification.</li>')
                any_caveats = True

        if not any_caveats:
            parts.append('<li>No significant limitations identified.</li>')

        parts.extend(['</ul>', '</div>'])
        return "\n".join(parts)

    def _build_uq_mapping_section(self) -> str:
        return textwrap.dedent("""\
        <div class="section" id="section-uqmap">
            <h2>UQ Mapping Assumption</h2>
            <div class="highlight">
                <p>The numerical uncertainty is defined as:</p>
                <p style="text-align:center; font-size:13pt; font-weight:600;">
                    u<sub>num</sub> = &delta; = |f<sub>1</sub> &minus; f<sub>RE</sub>|
                </p>
                <p>This quantity is treated as a <strong>1-sigma equivalent estimate</strong>
                for aggregation purposes (e.g., in the V&amp;V 20 RSS uncertainty budget).
                This is a <strong>modeling assumption</strong>, not a statistically derived
                quantity. The GCI error band (F<sub>s</sub> &times; u<sub>num</sub>) is a
                conservative engineering estimate of the discretization error, not a formal
                confidence interval.</p>
                <p>The 1-sigma interpretation follows Roache (1998) and is standard practice
                in ASME V&amp;V 20 Section 5.1. DOF is set to &infin; because the estimate
                is deterministic (model-based), not sampled from a statistical population.</p>
            </div>
        </div>
        """)

    def _build_references(self, fs_results: List[Optional[dict]],
                          lsr_results: List[Optional[dict]]) -> str:
        has_fs = any(r is not None for r in fs_results)
        has_lsr = any(r is not None for r in lsr_results)

        refs = [
            "ASME V&amp;V 20-2009 (R2021) &mdash; Standard for Verification and "
            "Validation in Computational Fluid Dynamics and Heat Transfer.",
            "Celik, I.B., Ghia, U., Roache, P.J., et al. (2008) &ldquo;Procedure "
            "for Estimation and Reporting of Uncertainty Due to Discretization in "
            "CFD Applications,&rdquo; <em>J. Fluids Eng.</em> 130(7), 078001.",
            "Roache, P.J. (1998) <em>Verification and Validation in Computational "
            "Science and Engineering,</em> Hermosa Publishers.",
            "Richardson, L.F. (1911) &ldquo;The Approximate Arithmetical Solution by "
            "Finite Differences of Physical Problems,&rdquo; <em>Phil. Trans. Royal "
            "Soc. London A</em> 210, 307&ndash;357.",
        ]

        if has_fs:
            refs.append(
                "Xing, T. and Stern, F. (2010) &ldquo;Factors of Safety for "
                "Richardson Extrapolation,&rdquo; <em>ASME J. Fluids Eng.</em> "
                "132(6), 061403.")

        if has_lsr:
            refs.append(
                "E&ccedil;a, L. and Hoekstra, M. (2014) &ldquo;A Procedure for "
                "the Estimation of the Numerical Uncertainty of CFD Calculations "
                "Based on Grid Refinement Studies,&rdquo; <em>J. Comp. Physics</em> "
                "262, 104&ndash;130.")

        refs.append(
            "ITTC (2024) &ldquo;Uncertainty Analysis in CFD Verification and "
            "Validation,&rdquo; Procedure 7.5-03-01-01.")

        items = "\n".join(f"<li>{r}</li>" for r in refs)
        return textwrap.dedent(f"""\
        <div class="section" id="section-refs">
            <h2>References</h2>
            <ol>
        {items}
            </ol>
        </div>
        """)

    def _build_footer(self, now_str: str) -> str:
        return textwrap.dedent(f"""\
        <hr class="footer-rule"/>
        <div class="footer-block">
            <p>Report generated by <strong>GCI Calculator</strong>
               v{_esc(APP_VERSION)} on {_esc(now_str)}.</p>
            <p class="footer-notice">Grid Convergence Index per Celik et al.
               (2008) and ASME V&amp;V 20-2009 (R2021).</p>
        </div>
        """)

    # -----------------------------------------------------------------
    # Full HTML wrapper with embedded CSS
    # -----------------------------------------------------------------
    def _wrap_html(self, body_content: str) -> str:
        css = self._get_css()
        return (
            '<!DOCTYPE html>\n'
            '<html lang="en">\n'
            '<head>\n'
            '    <meta charset="UTF-8"/>\n'
            '    <meta name="viewport" content="width=device-width, '
            'initial-scale=1.0"/>\n'
            f'    <title>GCI Calculator v{_esc(APP_VERSION)} '
            f'— Grid Convergence Report</title>\n'
            '    <style>\n'
            f'{css}\n'
            '    </style>\n'
            '</head>\n'
            '<body>\n'
            f'{body_content}\n'
            '</body>\n'
            '</html>\n'
        )

    @staticmethod
    def _get_css() -> str:
        """Professional light-theme CSS for print-ready reports."""
        return textwrap.dedent("""\
        /* ---- Reset & Base ---- */
        *, *::before, *::after { box-sizing: border-box; }
        body {
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            font-size: 11pt; line-height: 1.5;
            color: #1a1a2e; background: #ffffff;
            margin: 0; padding: 20px 40px;
        }
        /* ---- Header ---- */
        .header-block { text-align: center; margin-bottom: 10px; }
        .report-title {
            font-size: 18pt; color: #1a365d; margin: 18px 0 4px 0;
        }
        .report-meta {
            font-size: 10pt; color: #666; margin-bottom: 6px;
        }
        .header-rule, .footer-rule {
            border: none; border-top: 2px solid #1a365d; margin: 16px 0;
        }
        /* ---- Table of Contents ---- */
        .toc {
            background: #f0f4fa; border: 1px solid #c8d6e5;
            border-radius: 4px; padding: 12px 24px;
            margin: 12px 0 20px 0; max-width: 500px;
        }
        .toc h2 { font-size: 13pt; margin: 0 0 8px 0; color: #1a365d; }
        .toc ol { margin: 0; padding-left: 20px; }
        .toc li { margin: 3px 0; font-size: 10.5pt; }
        .toc a { color: #2563eb; text-decoration: none; }
        .toc a:hover { text-decoration: underline; }
        /* ---- Sections ---- */
        .section { margin: 24px 0; }
        .section h2 {
            font-size: 15pt; color: #1a365d;
            border-bottom: 2px solid #2563eb;
            padding-bottom: 4px; margin-bottom: 12px;
        }
        .section h3 {
            font-size: 12pt; color: #1e3a5f; margin: 14px 0 6px 0;
        }
        /* ---- Tables ---- */
        table {
            border-collapse: collapse; width: 100%;
            margin: 8px 0 14px 0; font-size: 10pt;
        }
        th, td {
            border: 1px solid #b0bec5;
            padding: 5px 10px; text-align: left; vertical-align: top;
        }
        th {
            background: #1a365d; color: #fff;
            font-weight: 600; white-space: nowrap;
        }
        tbody tr:nth-child(even) { background: #f5f7fa; }
        tbody tr:hover { background: #e8edf5; }
        .total-row td {
            background: #1a365d !important;
            color: #fff; font-weight: 700;
        }
        /* ---- Verdicts ---- */
        .verdict {
            padding: 10px 14px; border-radius: 4px;
            margin: 10px 0; font-size: 10.5pt;
        }
        .verdict.pass {
            background: #d4edda; border-left: 5px solid #28a745;
            color: #155724;
        }
        .verdict.fail {
            background: #f8d7da; border-left: 5px solid #dc3545;
            color: #721c24;
        }
        .verdict.neutral {
            background: #fff3cd; border-left: 5px solid #ffc107;
            color: #856404;
        }
        .highlight {
            background: #fff3cd; border-left: 5px solid #ffc107;
            padding: 8px 12px; margin: 10px 0; font-size: 10.5pt;
        }
        /* ---- Figures ---- */
        .figure-container {
            text-align: center; margin: 14px 0;
            page-break-inside: avoid;
        }
        .figure-container img {
            border: 1px solid #dee2e6; border-radius: 3px;
        }
        /* ---- Report text blocks ---- */
        .findings-block {
            background: #f8f9fa; border: 1px solid #dee2e6;
            border-radius: 4px; padding: 12px 16px;
            font-size: 10.5pt; line-height: 1.6;
            margin: 8px 0;
        }
        /* ---- Footer ---- */
        .footer-block {
            text-align: center; font-size: 9.5pt; color: #666;
            margin-top: 10px;
        }
        .footer-notice {
            font-size: 8pt; color: #999;
            max-width: 700px; margin: 4px auto 0 auto;
        }
        /* ---- Print ---- */
        @media print {
            body { padding: 10px 20px; font-size: 10pt; }
            .toc { page-break-after: always; }
            .figure-container { page-break-inside: avoid; }
            a { color: #000; text-decoration: none; }
            .verdict.pass { border-left: 3px solid #28a745; }
            .verdict.fail { border-left: 3px solid #dc3545; }
            .verdict.neutral { border-left: 3px solid #ffc107; }
        }
        """)


# =============================================================================
# MAIN WINDOW
# =============================================================================

class GCIMainWindow(QMainWindow):
    """Main window for the GCI Calculator application."""

    def __init__(self):
        super().__init__()
        self._project_path = None   # Current .gci file path
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1100, 750)
        self._setup_ui()

    def _setup_ui(self):
        # Central widget: project info bar + tab widget
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.setCentralWidget(central)

        # --- Project info / notes bar (collapsible, collapsed by default) ---
        self._project_bar = QFrame()
        self._project_bar.setStyleSheet(
            f"QFrame {{ background-color: {DARK_COLORS['bg_alt']}; "
            f"border-bottom: 1px solid {DARK_COLORS['border']}; }}"
        )
        bar_layout = QVBoxLayout(self._project_bar)
        bar_layout.setContentsMargins(12, 6, 12, 6)
        bar_layout.setSpacing(4)

        bar_top = QHBoxLayout()
        bar_top.setSpacing(8)
        self._btn_toggle_info = QPushButton("\u25b6 Project Info")
        self._btn_toggle_info.setFlat(True)
        self._btn_toggle_info.setStyleSheet(
            f"QPushButton {{ color: {DARK_COLORS['accent']}; font-weight: bold; "
            f"font-size: 12px; text-align: left; border: none; padding: 2px; }}"
            f"QPushButton:hover {{ color: {DARK_COLORS['accent_hover']}; }}"
        )
        self._btn_toggle_info.setCursor(Qt.PointingHandCursor)
        self._btn_toggle_info.clicked.connect(self._toggle_project_info)
        bar_top.addWidget(self._btn_toggle_info)
        bar_top.addStretch()

        self._lbl_project_name = QLabel("Untitled")
        self._lbl_project_name.setStyleSheet(
            f"color: {DARK_COLORS['fg']}; font-size: 12px; font-weight: bold;"
        )
        bar_top.addWidget(self._lbl_project_name)
        bar_layout.addLayout(bar_top)

        # Collapsible detail area
        self._project_detail_frame = QFrame()
        from PySide6.QtWidgets import QGridLayout as _QGL
        detail_layout = _QGL(self._project_detail_frame)
        detail_layout.setContentsMargins(4, 4, 4, 4)
        detail_layout.setSpacing(6)

        lbl_style = f"color: {DARK_COLORS['fg_dim']}; font-size: 11px;"
        val_style = (
            f"background-color: {DARK_COLORS['bg_input']}; "
            f"color: {DARK_COLORS['fg']}; border: 1px solid {DARK_COLORS['border']}; "
            f"border-radius: 3px; padding: 3px 6px; font-size: 11px;"
        )

        def _mk_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(lbl_style)
            return lbl

        detail_layout.addWidget(_mk_label("Program / Project:"), 0, 0)
        self._edit_program = QLineEdit()
        self._edit_program.setStyleSheet(val_style)
        self._edit_program.setPlaceholderText("e.g., XYZ CFD Grid Study")
        detail_layout.addWidget(self._edit_program, 0, 1)

        detail_layout.addWidget(_mk_label("Analyst:"), 0, 2)
        self._edit_analyst = QLineEdit()
        self._edit_analyst.setStyleSheet(val_style)
        self._edit_analyst.setPlaceholderText("e.g., J. Smith")
        detail_layout.addWidget(self._edit_analyst, 0, 3)

        detail_layout.addWidget(_mk_label("Date:"), 0, 4)
        self._edit_date = QLineEdit()
        self._edit_date.setStyleSheet(val_style)
        self._edit_date.setText(datetime.datetime.now().strftime("%Y-%m-%d"))
        detail_layout.addWidget(self._edit_date, 0, 5)

        detail_layout.addWidget(_mk_label("Notes:"), 1, 0)
        self._edit_notes = QTextEdit()
        self._edit_notes.setStyleSheet(f"QTextEdit {{ {val_style} }}")
        self._edit_notes.setPlaceholderText(
            "Free-form notes: grid strategy, key assumptions, etc."
        )
        self._edit_notes.setMaximumHeight(80)
        detail_layout.addWidget(self._edit_notes, 1, 1, 1, 5)

        detail_layout.addWidget(_mk_label("Decision Consequence:"), 2, 0)
        self._cmb_consequence = QComboBox()
        self._cmb_consequence.addItems(["Low", "Medium", "High"])
        self._cmb_consequence.setCurrentText("Medium")
        self._cmb_consequence.setStyleSheet(val_style)
        detail_layout.addWidget(self._cmb_consequence, 2, 1)

        detail_layout.setColumnStretch(1, 3)
        detail_layout.setColumnStretch(3, 2)
        detail_layout.setColumnStretch(5, 2)

        bar_layout.addWidget(self._project_detail_frame)
        self._project_detail_frame.setVisible(False)
        self._project_info_visible = False

        central_layout.addWidget(self._project_bar)

        # --- Tab widget ---
        self._tabs = QTabWidget()
        central_layout.addWidget(self._tabs)

        self._tab_calc = GCICalculatorTab()
        self._tab_spatial = SpatialGCITab()
        self._tab_planner = MeshStudyPlannerTab()
        self._tab_ref = GCIReferenceTab()

        self._tabs.addTab(self._tab_calc, "\U0001F4CA GCI Calculator")
        self._tabs.addTab(self._tab_spatial, "\U0001F30D Spatial GCI")
        self._tabs.addTab(self._tab_planner, "\U0001F4D0 Study Planner")
        self._tabs.addTab(self._tab_ref, "\U0001F4D6 Reference")

        # Menu bar
        mb = self.menuBar()

        file_menu = mb.addMenu("File")
        file_menu.addAction("New Study", self._new_study, "Ctrl+N")
        file_menu.addSeparator()
        file_menu.addAction("Open Study...", self._open_study, "Ctrl+O")
        file_menu.addAction("Save Study", self._save_study, "Ctrl+S")
        file_menu.addAction("Save Study As...", self._save_study_as,
                           "Ctrl+Shift+S")
        file_menu.addSeparator()
        file_menu.addAction("Export Results to Clipboard",
                           self._export_clipboard, "Ctrl+Shift+C")
        file_menu.addAction("Export Results to File...",
                           self._export_file, "Ctrl+E")
        file_menu.addAction("Export HTML Report...",
                           self._export_html, "Ctrl+H")
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close, "Alt+F4")

        examples_menu = mb.addMenu("Examples")
        examples_menu.addAction(
            "Turbine Blade — 4-Grid Monotonic (2 quantities)",
            self._load_example_turbine)
        examples_menu.addAction(
            "Pipe Flow — 3-Grid Oscillatory",
            self._load_example_oscillatory)
        examples_menu.addAction(
            "Heat Exchanger — 3-Grid Clean 2nd-Order",
            self._load_example_clean)
        examples_menu.addAction(
            "Nozzle Flow — 3-Grid Divergent",
            self._load_example_divergent)
        examples_menu.addSeparator()
        examples_menu.addAction(
            "Combustor Liner — 2-Grid (Assumed Order)",
            self._load_example_two_grid)
        examples_menu.addSeparator()
        examples_menu.addAction(
            "Exhaust Duct \u2014 5-Grid (Production = Grid 3)",
            self._load_example_five_grid)
        examples_menu.addSeparator()
        examples_menu.addAction(
            "Flat Plate Heat Transfer \u2014 Spatial (3-Grid)",
            self._load_example_spatial)

        help_menu = mb.addMenu("Help")
        help_menu.addAction("About", self._show_about)

        # Status bar
        self.statusBar().showMessage(
            f"{APP_NAME} v{APP_VERSION} ({APP_DATE})"
        )

    # ------------------------------------------------------------------
    # PROJECT INFO BAR HELPERS
    # ------------------------------------------------------------------

    def _toggle_project_info(self):
        self._project_info_visible = not self._project_info_visible
        self._project_detail_frame.setVisible(self._project_info_visible)
        arrow = "\u25bc" if self._project_info_visible else "\u25b6"
        self._btn_toggle_info.setText(f"{arrow} Project Info")

    def get_project_metadata(self) -> dict:
        return {
            'program': self._edit_program.text().strip(),
            'analyst': self._edit_analyst.text().strip(),
            'date': self._edit_date.text().strip(),
            'notes': self._edit_notes.toPlainText().strip(),
            'decision_consequence': self._cmb_consequence.currentText().strip(),
        }

    def set_project_metadata(self, meta: dict):
        self._edit_program.setText(meta.get('program', ''))
        self._edit_analyst.setText(meta.get('analyst', ''))
        self._edit_date.setText(meta.get('date', ''))
        self._edit_notes.setPlainText(meta.get('notes', ''))
        dc = normalize_decision_consequence(meta.get('decision_consequence', 'Medium'))
        idx = self._cmb_consequence.findText(dc)
        if idx >= 0:
            self._cmb_consequence.setCurrentIndex(idx)

    # ------------------------------------------------------------------
    # MENU ACTIONS
    # ------------------------------------------------------------------

    def _export_clipboard(self):
        try:
            text = self._tab_calc.get_results_text()
            if not text.strip():
                QMessageBox.information(self, "No Results",
                                       "Run a GCI computation first.")
                return
            QApplication.clipboard().setText(text)
            self.statusBar().showMessage("Results copied to clipboard.", 5000)
        except Exception as exc:
            QMessageBox.warning(self, "Clipboard Error",
                                f"Could not copy to clipboard:\n\n{exc}")

    def _export_file(self):
        text = self._tab_calc.get_results_text()
        if not text.strip():
            QMessageBox.information(self, "No Results",
                                   "Run a GCI computation first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "GCI_Results.txt",
            "Text files (*.txt);;All files (*.*)"
        )
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(text)
                self.statusBar().showMessage(f"Exported to {path}", 5000)
            except Exception as exc:
                QMessageBox.critical(
                    self, "Export Error",
                    f"Could not write file:\n\n{exc}"
                )

    def _export_html(self):
        """Export a self-contained HTML report with embedded plots."""
        tab = self._tab_calc
        if not hasattr(tab, '_results') or not tab._results:
            QMessageBox.information(self, "No Results",
                                   "Run a GCI computation first.")
            return

        try:
            results = list(tab._results)
            qty_names = list(tab._quantity_names)
            qty_units = list(tab._quantity_units)
            fs_results = (list(tab._fs_results)
                          if hasattr(tab, '_fs_results') else [])
            lsr_results = (list(tab._lsr_results)
                           if hasattr(tab, '_lsr_results') else [])

            settings = {
                "dim": tab._cmb_dim.currentText(),
                "safety_factor": tab._spn_safety.value(),
                "theoretical_order": tab._spn_theoretical_p.value(),
                "reference_scale": tab._spn_ref_scale.value(),
                "n_grids": tab._cmb_n_grids.currentData(),
                "production_grid": tab._cmb_production.currentIndex(),
                "quantity_units": qty_units,
            }
            settings.update(self.get_project_metadata())

            convergence_fig = tab._fig if hasattr(tab, '_fig') else None

            generator = GCIReportGenerator()
            html_content = generator.generate_report(
                results=results,
                quantity_names=qty_names,
                quantity_units=qty_units,
                fs_results=fs_results,
                lsr_results=lsr_results,
                convergence_fig=convergence_fig,
                settings=settings,
            )

            default_name = "GCI_Report.html"
            if self._project_path:
                base = os.path.splitext(
                    os.path.basename(self._project_path))[0]
                default_name = f"{base}_Report.html"

            filepath, _ = QFileDialog.getSaveFileName(
                self, "Export HTML Report", default_name,
                "HTML Files (*.html);;All Files (*.*)"
            )
            if not filepath:
                return

            generator.save_report(html_content, filepath)
            self.statusBar().showMessage(
                f"HTML report exported to {filepath}", 8000)

        except Exception as exc:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export HTML report:\n{exc}"
            )

    # ------------------------------------------------------------------
    # PROJECT SAVE / LOAD
    # ------------------------------------------------------------------

    def _update_title(self):
        """Update window title with project name."""
        if self._project_path:
            name = os.path.splitext(os.path.basename(self._project_path))[0]
            self.setWindowTitle(
                f"{APP_NAME} v{APP_VERSION} — {name}"
            )
        else:
            self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")

    def _new_study(self):
        """Start a fresh GCI study."""
        reply = QMessageBox.question(
            self, "New Study",
            "Start a new GCI study? Unsaved data will be lost.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            self._tab_calc.clear_all()
            self._tab_spatial.clear_all()
            self._tab_planner.clear_all()
            self.set_project_metadata({})
            self._project_path = None
            self._update_title()
            self.statusBar().showMessage("New study started.", 5000)

    def _open_study(self):
        """Load a GCI study from a .gci JSON file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Open GCI Study", "",
            "GCI Study files (*.gci);;JSON files (*.json);;All files (*.*)"
        )
        if not path:
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to read file:\n{exc}"
            )
            return

        try:
            self._tab_calc.set_state(state)
            if "spatial" in state:
                self._tab_spatial.set_state(state["spatial"])
            if "planner" in state:
                self._tab_planner.set_state(state["planner"])
            if "project_metadata" in state:
                self.set_project_metadata(state["project_metadata"])
        except Exception as exc:
            QMessageBox.critical(
                self, "Load Error",
                f"Project file appears corrupt or incompatible:\n\n{exc}"
            )
            return
        self._project_path = path
        self._update_title()
        self.statusBar().showMessage(f"Loaded: {path}", 7000)

    def _save_study(self):
        """Save to current file path, or prompt if none."""
        if not self._project_path:
            self._save_study_as()
            return
        self._do_save(self._project_path)

    def _save_study_as(self):
        """Prompt for file path and save."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save GCI Study", "GCI_Study.gci",
            "GCI Study files (*.gci);;JSON files (*.json);;All files (*.*)"
        )
        if path:
            self._do_save(path)

    def _do_save(self, path: str):
        """Perform the actual save to file."""
        state = self._tab_calc.get_state()
        state["spatial"] = self._tab_spatial.get_state()
        state["planner"] = self._tab_planner.get_state()
        state["project_metadata"] = self.get_project_metadata()
        state["saved_at"] = datetime.datetime.now().isoformat()

        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            self._project_path = path
            self._update_title()
            self.statusBar().showMessage(f"Saved: {path}", 7000)
        except Exception as exc:
            QMessageBox.critical(
                self, "Save Error",
                f"Failed to save file:\n{exc}"
            )

    # ------------------------------------------------------------------
    # EXAMPLE DATA LOADERS
    # ------------------------------------------------------------------

    def _populate_example(self, n_grids, dim_idx, qty_names, data, msg,
                          production_grid=0, qty_units=None):
        """Helper to populate the calculator tab with example data.

        Args:
            n_grids: Number of grids (2-6).
            dim_idx: Dimension combo index (0 = 3D, 1 = 2D).
            qty_names: List of quantity-of-interest names.
            data: List of rows [cells, qty1, qty2, ...].
            msg: Status bar message.
            production_grid: 0-based index of the production grid
                (default 0 = finest grid).
            qty_units: List of unit strings per quantity (optional).
        """
        tab = self._tab_calc

        # Set number of grids
        found = False
        for idx in range(tab._cmb_n_grids.count()):
            if tab._cmb_n_grids.itemData(idx) == n_grids:
                tab._cmb_n_grids.setCurrentIndex(idx)
                found = True
                break
        if not found and n_grids > 0:
            tab._cmb_n_grids.addItem(f"{n_grids} grids", n_grids)
            tab._cmb_n_grids.setCurrentIndex(
                tab._cmb_n_grids.count() - 1)

        # Set dimension
        tab._cmb_dim.setCurrentIndex(dim_idx)

        # Set quantities
        tab._n_quantities = len(qty_names)
        tab._quantity_names = list(qty_names)
        if qty_units:
            tab._quantity_units = list(qty_units)
        else:
            tab._quantity_units = [""] * len(qty_names)
        tab._rebuild_table()
        tab._rebuild_qty_config()

        # Set production grid
        if 0 <= production_grid < tab._cmb_production.count():
            tab._cmb_production.setCurrentIndex(production_grid)

        # Fill table
        for i, row in enumerate(data):
            for j, val in enumerate(row):
                item = QTableWidgetItem(f"{val:.6g}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                tab._grid_table.setItem(i, j, item)

        self.statusBar().showMessage(msg, 7000)

    def _load_example_turbine(self):
        """Example 1: Turbine blade 4-grid monotonic convergence, 2 quantities.

        A realistic aerospace thermal CFD example.  Four grids with roughly
        constant refinement ratio (~1.44).  Two quantities of interest:
        blade surface temperature (K) and static pressure (psi).
        The temperature converges monotonically with observed order ~3.
        Expected: monotonic, Fs = 1.25, asymptotic ratio near 1.0.
        """
        self._populate_example(
            n_grids=4,
            dim_idx=0,  # 3D
            qty_names=["Blade Temp", "Pressure"],
            data=[
                # [cells,      temp,      pressure]
                [2_400_000,   847.32,    14.523],   # finest
                [  800_000,   849.15,    14.498],   # medium
                [  267_000,   854.78,    14.441],   # coarse
                [   89_000,   868.91,    14.329],   # very coarse
            ],
            msg="Example 1 loaded: Turbine blade 4-grid study (monotonic, 2 quantities).",
            qty_units=["K", "psi"],
        )

    def _load_example_oscillatory(self):
        """Example 2: Pipe flow 3-grid oscillatory convergence.

        Internal flow in a pipe with SIMPLE-type pressure-velocity coupling.
        The pressure drop oscillates between grid levels — a common occurrence
        with collocated grids and Rhie-Chow interpolation.
        Expected: oscillatory, Fs = 3.0, Richardson extrapolation = N/A.
        """
        self._populate_example(
            n_grids=3,
            dim_idx=0,  # 3D
            qty_names=["Pressure Drop"],
            data=[
                # [cells,      dP]
                [  500_000,   1247.8],     # finest — overshoots
                [  150_000,   1252.3],     # medium — undershoots
                [   45_000,   1239.6],     # coarse — overshoots again
            ],
            msg="Example 2 loaded: Pipe flow 3-grid study (oscillatory convergence).",
            qty_units=["Pa"],
        )

    def _load_example_clean(self):
        """Example 3: Heat exchanger 3-grid clean 2nd-order convergence.

        A textbook-clean case with constant refinement ratio r = 2.0
        and exact 2nd-order convergence.  The data follows f(h) = f_exact + C*h^2
        with f_exact = 423.15 K (outlet temperature).
        Expected: monotonic, p ≈ 2.0, asymptotic ratio ≈ 1.0.
        """
        # f(h) = 423.15 + 850 * h^2
        # h = (1/N)^(1/3)
        # N = 1,000,000 -> h = 0.01   -> f = 423.15 + 0.085 = 423.235
        # N =   125,000 -> h = 0.02   -> f = 423.15 + 0.340 = 423.490
        # N =    15,625 -> h = 0.04   -> f = 423.15 + 1.360 = 424.510
        self._populate_example(
            n_grids=3,
            dim_idx=0,  # 3D
            qty_names=["Outlet Temp"],
            data=[
                # [cells,       T_out]
                [1_000_000,   423.235],    # finest
                [  125_000,   423.490],    # medium
                [   15_625,   424.510],    # coarse
            ],
            msg="Example 3 loaded: Heat exchanger 3-grid study (clean 2nd-order, p \u2248 2.0).",
            qty_units=["K"],
        )

    def _load_example_divergent(self):
        """Example 4: Nozzle flow 3-grid divergent convergence.

        A case where the grids are too coarse to be in the asymptotic range.
        The coarser grids happen to give answers closer to experiment due to
        error cancellation, but the fine grid reveals the true (larger)
        discretization error.  R > 1 → divergent.
        Expected: divergent, GCI invalid, warnings issued.
        """
        self._populate_example(
            n_grids=3,
            dim_idx=0,  # 3D
            qty_names=["Thrust"],
            data=[
                # [cells,      thrust]
                [  800_000,   4521.3],     # finest — worst (highest error)
                [  200_000,   4518.7],     # medium
                [   50_000,   4517.1],     # coarse — looks best (error cancel)
            ],
            msg="Example 4 loaded: Nozzle flow 3-grid study (DIVERGENT \u2014 GCI invalid).",
            qty_units=["N"],
        )

    def _load_example_two_grid(self):
        """Example 5: Combustor liner 2-grid study with assumed order.

        A common real-world scenario: you only have two grids (perhaps a
        production mesh and one refinement).  The tool uses the assumed
        theoretical order (p = 2) and the conservative safety factor
        Fs = 3.0.  Demonstrates how a 2-grid GCI is still useful but
        carries a wider uncertainty band than a 3-grid study.
        Expected: 2-grid assumed order, Fs = 3.0, valid result.
        """
        self._populate_example(
            n_grids=2,
            dim_idx=0,  # 3D
            qty_names=["Wall Temp"],
            data=[
                # [cells,      T_wall]
                [3_200_000,   1184.6],     # fine (production mesh refined)
                [  800_000,   1191.2],     # coarse (production mesh)
            ],
            msg="Example 5 loaded: Combustor liner 2-grid study (assumed order, Fs = 3.0).",
            qty_units=["K"],
        )

    def _load_example_five_grid(self):
        """Example 6: Exhaust duct 5-grid study with middle production grid.

        A realistic scenario: you ran your production CFD on a 500K-cell mesh
        (Grid 3), then created two finer grids and two coarser grids for the
        GCI study.  The production mesh is NOT the finest — it sits in the
        middle.  This demonstrates how the tool computes u_num for any grid,
        not just the finest.

        5 grids with roughly constant refinement ratio (~1.5 in 3D):
          Grid 1: 3,375,000 cells  (finest — GCI study only)
          Grid 2: 1,000,000 cells  (finer verification mesh)
          Grid 3:   500,000 cells  ★ PRODUCTION MESH
          Grid 4:   148,000 cells  (coarser check)
          Grid 5:    44,000 cells  (coarsest)

        Quantity: Average exit temperature (K).
        Expected: monotonic, production grid u_num larger than fine-grid u_num.
        """
        self._populate_example(
            n_grids=5,
            dim_idx=0,  # 3D
            qty_names=["Exit Temp"],
            data=[
                # [cells,       T_exit]
                [3_375_000,   712.48],    # Grid 1 (finest)
                [1_000_000,   713.21],    # Grid 2
                [  500_000,   714.55],    # Grid 3 — PRODUCTION
                [  148_000,   718.32],    # Grid 4
                [   44_000,   727.91],    # Grid 5 (coarsest)
            ],
            msg="Example 6 loaded: Exhaust duct 5-grid study "
                "(production mesh = Grid 3).",
            production_grid=2,  # 0-based → Grid 3
            qty_units=["K"],
        )

    def _load_example_spatial(self):
        """Example 7: Synthetic flat plate heat transfer — spatial GCI.

        Generates ~200 points on a rectangular plate with a sinusoidal
        temperature field plus grid-dependent noise. Demonstrates the
        full spatial GCI workflow with pre-interpolated data.
        """
        # Generate synthetic data
        rng = np.random.default_rng(42)
        nx, ny = 14, 14
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        xx, yy = np.meshgrid(x, y)
        xf = xx.ravel()
        yf = yy.ravel()
        n_pts = len(xf)

        # Base temperature field: T(x,y) = 350 + 120*sin(pi*x)*cos(pi*y/2)
        T_exact = 350.0 + 120.0 * np.sin(np.pi * xf) * np.cos(
            np.pi * yf / 2.0)

        # Grid-dependent noise (decreases with refinement)
        # Fine: 2.4M cells, noise sigma ~0.15 K
        # Medium: 800K cells, noise sigma ~0.45 K
        # Coarse: 267K cells, noise sigma ~1.2 K
        noise_fine = rng.normal(0, 0.15, n_pts)
        noise_med = rng.normal(0, 0.45, n_pts)
        noise_coarse = rng.normal(0, 1.2, n_pts)

        # Add systematic bias that scales with h^2
        h_fine = (1.0 / 2_400_000) ** (1.0 / 3.0)
        h_med = (1.0 / 800_000) ** (1.0 / 3.0)
        h_coarse = (1.0 / 267_000) ** (1.0 / 3.0)
        C_bias = 8000.0  # bias coefficient

        T_fine = T_exact + C_bias * h_fine ** 2 + noise_fine
        T_med = T_exact + C_bias * h_med ** 2 + noise_med
        T_coarse = T_exact + C_bias * h_coarse ** 2 + noise_coarse

        coords = np.column_stack([xf, yf])

        # Switch to spatial tab and populate
        tab = self._tab_spatial
        tab._cmb_mode.setCurrentIndex(0)  # pre-interpolated
        tab._cmb_dim_spatial.setCurrentIndex(0)  # 3D
        tab._edt_qty_name.setText("Surface Temperature")
        tab._cmb_unit_cat.setCurrentIndex(0)  # Temperature
        tab._populate_spatial_unit_combo("Temperature")
        tab._cmb_unit_val.setCurrentIndex(0)  # K

        # Store the data directly
        tab._pre_coords = coords
        tab._pre_solutions = [T_fine, T_med, T_coarse]
        tab._pre_grid_names = ["Fine (2.4M)", "Medium (800K)", "Coarse (267K)"]

        tab._lbl_pre_file.setText("Synthetic flat plate data")
        tab._lbl_pre_file.setStyleSheet(
            f"color: {DARK_COLORS['green']}; font-weight: bold;")
        tab._lbl_pre_info.setText(
            f"Generated {n_pts} points, 3 grids, 2D coordinates")

        # Populate cell count table
        cell_counts = [2_400_000, 800_000, 267_000]
        tab._tbl_pre_cells.setRowCount(3)
        for i in range(3):
            name_item = QTableWidgetItem(tab._pre_grid_names[i])
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            tab._tbl_pre_cells.setItem(i, 0, name_item)
            cell_item = QTableWidgetItem(f"{cell_counts[i]}")
            cell_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            tab._tbl_pre_cells.setItem(i, 1, cell_item)

        tab._guidance_import.set_guidance(
            f"Loaded {n_pts} synthetic points x 3 grids. "
            f"Click 'Compute Spatial GCI' to analyze.",
            'green')

        # Switch to spatial tab
        for i in range(self._tabs.count()):
            if self._tabs.widget(i) is self._tab_spatial:
                self._tabs.setCurrentIndex(i)
                break

        self.statusBar().showMessage(
            "Example 7 loaded: Flat plate heat transfer \u2014 spatial GCI "
            "(synthetic, 3 grids, ~200 points).", 7000)

    def _show_about(self):
        QMessageBox.about(
            self, f"About {APP_NAME}",
            f"<h2>{APP_NAME} v{APP_VERSION}</h2>"
            f"<p>Grid Convergence Index Calculator</p>"
            f"<p>Computes GCI per Celik et al. (2008) and Roache (1998) "
            f"for 2, 3, or N-grid refinement studies.</p>"
            f"<p>Converts GCI to standard uncertainty u_num for use in "
            f"ASME V&V 20 uncertainty budgets.</p>"
            f"<p><b>Features:</b><br>"
            f"\u2022 Single-point GCI (2\u20136 grids, N quantities)<br>"
            f"\u2022 Spatial/Field GCI (point-by-point, distributions)<br>"
            f"\u2022 Unit labels for quantities<br>"
            f"\u2022 Save/Load .gci project files<br>"
            f"\u2022 Factor of Safety &amp; LSR alternative methods</p>"
            f"<p><b>Standards:</b><br>"
            f"ASME V&V 20-2009 (R2021) Section 5.1<br>"
            f"Celik et al. (2008) JFE 130(7)<br>"
            f"Xing &amp; Stern (2010) JFE 132(6)<br>"
            f"Eca &amp; Hoekstra (2014a) JCP 262 (LSR)<br>"
            f"Eca &amp; Hoekstra (2014b) IJNMF 75 (spatial)<br>"
            f"Roache (1998) Hermosa Publishers</p>"
        )


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    import traceback as _tb

    def _global_exception_handler(exc_type, exc_value, exc_traceback):
        """Show unhandled exceptions in a dialog instead of crashing silently."""
        error_text = "".join(
            _tb.format_exception(exc_type, exc_value, exc_traceback))
        try:
            QMessageBox.critical(
                None, f"{APP_NAME} v{APP_VERSION} — Unexpected Error",
                f"An unexpected error occurred:\n\n"
                f"{exc_value}\n\n"
                f"Please report this issue with the details below:\n\n"
                f"{error_text[:1500]}"
            )
        except Exception:
            pass  # If GUI is dead, don't compound the problem

    sys.excepthook = _global_exception_handler

    app = QApplication(sys.argv)
    app.setStyleSheet(get_dark_stylesheet())
    app.setStyle("Fusion")
    window = GCIMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
