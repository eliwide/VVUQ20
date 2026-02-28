#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistical Analyzer v1.4
Data Distribution Analysis Tool for ASME V&V 20 Uncertainty Budgets

Standalone PySide6 application for statistical analysis of experimental
and simulation data. Fits distributions, computes summary statistics,
and produces carry-over values for the VVUQ Uncertainty Aggregator.

Standards References:
    - JCGM 100:2008 (GUM) — Guide to the Expression of Uncertainty
    - JCGM 101:2008 (GUM Supplement 1) — Monte Carlo Methods
    - ASME PTC 19.1-2018 — Test Uncertainty
    - ASME V&V 20-2009 (R2021)

Copyright (c) 2026. All rights reserved.
"""

import sys
import os
import json
import io
import base64
import datetime
import textwrap
import subprocess
import importlib
import tempfile
import copy
import warnings
from html import escape as _html_esc
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, List, Dict, Tuple, Any

APP_VERSION = "1.4.0"
APP_NAME = "Statistical Analyzer"
APP_DATE = "2026-02-23"

REQUIRED_PACKAGES = {
    'PySide6': 'PySide6',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
}

def check_and_install_dependencies():
    """Check for required packages and offer to install missing ones."""
    missing = []
    for import_name, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(import_name)
        except ImportError:
            missing.append((import_name, pip_name))
    if missing:
        print(f"\n{'='*60}")
        print(f"  {APP_NAME} v{APP_VERSION} - Dependency Check")
        print(f"{'='*60}")
        print(f"\nThe following required packages are missing:\n")
        for imp, pip in missing:
            print(f"  - {pip}")
        print(f"\nTo install all dependencies, run:")
        print(f"  pip install {' '.join(p for _, p in missing)}")
        resp = input("\nAttempt automatic installation now? [y/N]: ").strip().lower()
        if resp == 'y':
            for imp, pip in missing:
                print(f"Installing {pip}...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip])
            print("\nAll dependencies installed. Restarting...")
            subprocess.Popen([sys.executable] + sys.argv)
            sys.exit(0)
        else:
            print("\nPlease install dependencies manually and restart.")
            sys.exit(1)

if __name__ == '__main__':
    check_and_install_dependencies()

import numpy as np
from scipy import stats as sp_stats
from scipy.stats import (
    norm, t as t_dist, lognorm, uniform, triang,
    weibull_min, gamma, beta,
    shapiro, kstest, anderson, skew, kurtosis, goodness_of_fit,
)
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

os.environ["QT_API"] = "pyside6"
if "MPLCONFIGDIR" not in os.environ:
    try:
        mpl_cache_dir = os.path.join(tempfile.gettempdir(), "statistical_analyzer_mplconfig")
        os.makedirs(mpl_cache_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_cache_dir
    except OSError:
        pass

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout, QGroupBox, QLabel, QPushButton, QLineEdit,
    QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter, QScrollArea, QFrame, QMessageBox,
    QAbstractItemView, QFileDialog, QStatusBar, QStackedWidget, QMenu,
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont, QColor, QFontDatabase, QAction

import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

# =============================================================================
# CONSTANTS & THEME
# =============================================================================

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
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 6.5,
    'grid.color': DARK_COLORS['border'],
    'legend.facecolor': DARK_COLORS['bg_widget'],
    'legend.edgecolor': DARK_COLORS['border'],
}
plt.rcParams.update(PLOT_STYLE)

REPORT_PLOT_STYLE = {
    'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff',
    'axes.edgecolor': '#333333', 'axes.labelcolor': '#1a1a2e',
    'text.color': '#1a1a2e', 'xtick.color': '#333333', 'ytick.color': '#333333',
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.labelsize': 8, 'axes.titlesize': 9, 'legend.fontsize': 6.5,
    'grid.color': '#cccccc', 'legend.facecolor': '#f5f5f5',
    'legend.edgecolor': '#999999',
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
    "Dimensionless": ["—"],
    "Other":       [],
}

DISTRIBUTIONS = [
    ("Normal",      norm),
    ("Log-Normal",  lognorm),
    ("Uniform",     uniform),
    ("Triangular",  triang),
    ("Weibull",     weibull_min),
    ("Gamma",       gamma),
    ("Student-t",   t_dist),
    ("Beta",        beta),
]

# Distribution labels accepted by the VV20 uncertainty aggregator.
AGGREGATOR_DIST_MAP = {
    "Normal": "Normal",
    "Log-Normal": "Lognormal",
    "Uniform": "Uniform",
    "Triangular": "Triangular",
    "Weibull": "Weibull",
}

def get_dark_stylesheet():
    c = DARK_COLORS
    return f"""
    QMainWindow, QWidget {{
        background-color: {c['bg']}; color: {c['fg']}; font-size: 13px;
    }}
    QTabWidget::pane {{
        border: 1px solid {c['border']}; background-color: {c['bg']};
    }}
    QTabBar::tab {{
        background-color: {c['bg_alt']}; color: {c['fg_dim']};
        padding: 8px 16px; margin-right: 2px;
        border: 1px solid {c['border']}; border-bottom: none;
        border-top-left-radius: 4px; border-top-right-radius: 4px;
    }}
    QTabBar::tab:selected {{
        background-color: {c['bg_widget']}; color: {c['accent']};
        border-bottom: 2px solid {c['accent']};
    }}
    QTabBar::tab:hover {{
        background-color: {c['bg_widget']}; color: {c['fg']};
    }}
    QGroupBox {{
        border: 1px solid {c['border']}; border-radius: 6px;
        margin-top: 12px; padding-top: 16px;
        font-weight: bold; color: {c['accent']};
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; left: 12px; padding: 0 6px;
    }}
    QPushButton {{
        background-color: {c['bg_widget']}; color: {c['fg']};
        border: 1px solid {c['border']}; border-radius: 4px;
        padding: 6px 16px; min-height: 24px;
    }}
    QPushButton:hover {{
        background-color: {c['selection']}; border-color: {c['accent']};
    }}
    QPushButton:pressed {{
        background-color: {c['accent']}; color: {c['bg']};
    }}
    QPushButton:disabled {{
        color: {c['fg_dim']}; background-color: {c['bg']};
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background-color: {c['bg_input']}; color: {c['fg']};
        border: 1px solid {c['border']}; border-radius: 4px;
        padding: 4px 8px; min-height: 22px;
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {c['accent']};
    }}
    QComboBox::drop-down {{ border: none; width: 24px; }}
    QComboBox QAbstractItemView {{
        background-color: {c['bg_widget']}; color: {c['fg']};
        border: 1px solid {c['border']}; selection-background-color: {c['selection']};
    }}
    QTableWidget {{
        background-color: {c['bg_widget']}; alternate-background-color: {c['bg_alt']};
        color: {c['fg']}; gridline-color: {c['border']};
        border: 1px solid {c['border']}; border-radius: 4px;
    }}
    QTableWidget::item {{ padding: 4px; }}
    QTableWidget::item:selected {{ background-color: {c['selection']}; }}
    QHeaderView::section {{
        background-color: {c['bg_alt']}; color: {c['fg']};
        padding: 4px 8px; border: 1px solid {c['border']}; font-weight: bold;
    }}
    QTextEdit, QPlainTextEdit {{
        background-color: {c['bg_input']}; color: {c['fg']};
        border: 1px solid {c['border']}; border-radius: 4px;
    }}
    QScrollBar:vertical {{
        background-color: {c['bg']}; width: 12px; border: none;
    }}
    QScrollBar::handle:vertical {{
        background-color: {c['border']}; border-radius: 4px; min-height: 20px;
    }}
    QScrollBar::handle:vertical:hover {{ background-color: {c['fg_dim']}; }}
    QScrollBar:horizontal {{
        background-color: {c['bg']}; height: 12px; border: none;
    }}
    QScrollBar::handle:horizontal {{
        background-color: {c['border']}; border-radius: 4px; min-width: 20px;
    }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QStatusBar {{
        background-color: {c['bg_alt']}; color: {c['fg_dim']};
        border-top: 1px solid {c['border']};
    }}
    QMenuBar {{
        background-color: {c['bg_alt']}; color: {c['fg']};
    }}
    QMenuBar::item:selected {{ background-color: {c['selection']}; }}
    QMenu {{
        background-color: {c['bg_widget']}; color: {c['fg']};
        border: 1px solid {c['border']};
    }}
    QMenu::item:selected {{ background-color: {c['selection']}; }}
    QToolTip {{
        background-color: {c['bg_widget']}; color: {c['fg']};
        border: 1px solid {c['accent']}; padding: 6px; border-radius: 4px;
    }}
    QCheckBox {{ color: {c['fg']}; spacing: 8px; }}
    QCheckBox::indicator {{
        width: 16px; height: 16px;
        border: 1px solid {c['border']}; border-radius: 3px;
        background-color: {c['bg_input']};
    }}
    QCheckBox::indicator:checked {{
        background-color: {c['accent']}; border-color: {c['accent']};
    }}
    QSplitter::handle {{ background-color: {c['border']}; }}
    QLabel {{ color: {c['fg']}; }}
    """


# =============================================================================
# GUIDANCE PANEL
# =============================================================================

class GuidancePanel(QFrame):
    SEVERITY_CONFIG = {
        'green':  {'border_color': DARK_COLORS['green'],  'bg_color': '#1a2e1a', 'icon': '\u2714', 'label': 'OK'},
        'yellow': {'border_color': DARK_COLORS['yellow'], 'bg_color': '#2e2a1a', 'icon': '\u26A0', 'label': 'CAUTION'},
        'red':    {'border_color': DARK_COLORS['red'],    'bg_color': '#2e1a1a', 'icon': '\u2716', 'label': 'WARNING'},
    }

    def __init__(self, title="", parent=None):
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
        f = self._icon_label.font()
        f.setPointSize(12)
        self._icon_label.setFont(f)
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

    def _apply_severity(self, severity):
        cfg = self.SEVERITY_CONFIG.get(severity, self.SEVERITY_CONFIG['green'])
        self._icon_label.setText(cfg['icon'])
        self._icon_label.setStyleSheet(f"color: {cfg['border_color']};")
        self.setStyleSheet(
            f"GuidancePanel {{ background-color: {cfg['bg_color']}; "
            f"border-left: 4px solid {cfg['border_color']}; "
            f"border-top: none; border-right: none; border-bottom: none; "
            f"border-radius: 4px; }}")

    def set_guidance(self, message, severity='green'):
        self._message_label.setText(message)
        self._apply_severity(severity)

    def clear(self):
        self._message_label.setText("")
        self._apply_severity('green')


# =============================================================================
# HELPER: figure export
# =============================================================================

def _figure_to_base64(fig) -> str:
    orig_props = []
    for ax in fig.get_axes():
        orig_props.append({
            'facecolor': ax.get_facecolor(),
            'title_color': ax.title.get_color(),
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
    orig_fig_fc = fig.get_facecolor()
    fig.set_facecolor('#ffffff')
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode("ascii")
        buf.close()
    finally:
        fig.set_facecolor(orig_fig_fc)
        for ax, props in zip(fig.get_axes(), orig_props):
            ax.set_facecolor(props['facecolor'])
            if props['title_color']:
                ax.title.set_color(props['title_color'])
            ax.xaxis.label.set_color(props['xlabel_color'])
            ax.yaxis.label.set_color(props['ylabel_color'])
            # NOTE: Tick colors are restored from the global PLOT_STYLE rather
            # than per-axis originals because tick_params() does not expose a
            # getter. This is acceptable as long as all axes share the same
            # tick color from the dark theme stylesheet.
            ax.tick_params(axis='x', colors=PLOT_STYLE['xtick.color'])
            ax.tick_params(axis='y', colors=PLOT_STYLE['ytick.color'])
            for s_name, s_color in props['spine_colors'].items():
                ax.spines[s_name].set_edgecolor(s_color)
    return encoded


def export_figure_package(fig, base_path, metadata=None):
    from datetime import datetime, timezone
    orig_props = []
    for ax in fig.get_axes():
        legends = ax.get_legend()
        legend_text_colors = []
        legend_title_color = None
        legend_frame_fc = None
        legend_frame_ec = None
        if legends is not None:
            legend_text_colors = [t.get_color() for t in legends.get_texts()]
            if legends.get_title() is not None:
                legend_title_color = legends.get_title().get_color()
            frame = legends.get_frame()
            legend_frame_fc = frame.get_facecolor()
            legend_frame_ec = frame.get_edgecolor()
        orig_props.append({
            'facecolor': ax.get_facecolor(),
            'title_color': ax.title.get_color(),
            'xlabel_color': ax.xaxis.label.get_color(),
            'ylabel_color': ax.yaxis.label.get_color(),
            'spine_colors': {s: ax.spines[s].get_edgecolor() for s in ax.spines},
            'legend_text_colors': legend_text_colors,
            'legend_title_color': legend_title_color,
            'legend_frame_fc': legend_frame_fc,
            'legend_frame_ec': legend_frame_ec,
        })
        ax.set_facecolor(REPORT_PLOT_STYLE['axes.facecolor'])
        ax.title.set_color(REPORT_PLOT_STYLE['text.color'])
        ax.xaxis.label.set_color(REPORT_PLOT_STYLE['axes.labelcolor'])
        ax.yaxis.label.set_color(REPORT_PLOT_STYLE['axes.labelcolor'])
        ax.tick_params(axis='x', colors=REPORT_PLOT_STYLE['xtick.color'])
        ax.tick_params(axis='y', colors=REPORT_PLOT_STYLE['ytick.color'])
        for spine in ax.spines.values():
            spine.set_edgecolor(REPORT_PLOT_STYLE['axes.edgecolor'])
        if legends is not None:
            for t in legends.get_texts():
                t.set_color(REPORT_PLOT_STYLE['text.color'])
            if legends.get_title() is not None:
                legends.get_title().set_color(REPORT_PLOT_STYLE['text.color'])
            legends.get_frame().set_facecolor(REPORT_PLOT_STYLE['legend.facecolor'])
            legends.get_frame().set_edgecolor(REPORT_PLOT_STYLE['legend.edgecolor'])

    orig_fig_fc = fig.get_facecolor()
    fig.set_facecolor('#ffffff')
    try:
        for dpi_val in (300, 600):
            fig.savefig(f"{base_path}_{dpi_val}dpi.png", format="png", dpi=dpi_val,
                        bbox_inches="tight", facecolor="white", edgecolor="none")
        fig.savefig(f"{base_path}.svg", format="svg",
                    bbox_inches="tight", facecolor="white", edgecolor="none")
        fig.savefig(f"{base_path}.pdf", format="pdf",
                    bbox_inches="tight", facecolor="white", edgecolor="none")
    finally:
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
            legends = ax.get_legend()
            if legends is not None:
                for txt, col in zip(legends.get_texts(), props['legend_text_colors']):
                    txt.set_color(col)
                if legends.get_title() is not None and props['legend_title_color'] is not None:
                    legends.get_title().set_color(props['legend_title_color'])
                frame = legends.get_frame()
                if props['legend_frame_fc'] is not None:
                    frame.set_facecolor(props['legend_frame_fc'])
                if props['legend_frame_ec'] is not None:
                    frame.set_edgecolor(props['legend_frame_ec'])

    meta = {
        "tool_name": APP_NAME, "tool_version": APP_VERSION,
        "figure_id": os.path.basename(base_path),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "formats": ["png@300dpi", "png@600dpi", "svg", "pdf"],
    }
    if metadata:
        meta.update(metadata)
    with open(f"{base_path}_meta.json", "w", encoding="utf-8") as fp:
        json.dump(meta, fp, indent=2)


# =============================================================================
# ENGINE: Dataclasses
# =============================================================================

@dataclass
class ColumnData:
    """A single data column with metadata."""
    name: str = "Variable"
    unit: str = ""
    unit_category: str = "Other"
    values: List[float] = field(default_factory=list)
    mirror: bool = False
    mirror_center: float = 0.0

    def valid_values(self) -> np.ndarray:
        arr = np.array(self.values, dtype=float)
        return arr[np.isfinite(arr)]

    def analysis_values(self) -> np.ndarray:
        """Return values for analysis, applying mirroring if enabled.

        When mirror is True, each data point v is reflected about
        mirror_center to produce (2 * mirror_center - v).  The
        reflected points are appended so the sample size doubles.
        This is useful when only one side of a symmetric phenomenon
        is measured (e.g. thermal gradient on one side of a symmetric
        geometry) and the opposite side is assumed identical.
        """
        vals = self.valid_values()
        if not self.mirror or len(vals) == 0:
            return vals
        reflected = 2.0 * self.mirror_center - vals
        return np.sort(np.concatenate([vals, reflected]))


@dataclass
class DistributionFitResult:
    """Result of fitting a single distribution."""
    name: str = ""
    params: Tuple = ()
    param_str: str = ""
    aic: float = np.inf
    bic: float = np.inf
    gof_statistic: float = np.inf
    gof_pvalue: float = 0.0
    gof_method: str = "bootstrap_ad"  # bootstrap_ad, ks_screening, ks_fallback
    passed_gof: bool = False
    ks_statistic: float = 0.0
    ks_pvalue: float = 0.0
    ad_statistic: float = np.inf
    ad_critical_5pct: float = 0.0
    passed_ks: bool = False
    passed_ad: bool = False
    gof_fallback: bool = False
    beta_clipped: bool = False
    rank: int = 0
    delta_aic: float = 0.0
    akaike_weight: float = 0.0


def _format_dist_params(name: str, params: tuple) -> str:
    """Format distribution parameters for display."""
    if name == "Normal":
        return f"\u03bc={params[0]:.4g}, \u03c3={params[1]:.4g}"
    elif name == "Log-Normal":
        return f"s={params[0]:.4g}, loc={params[1]:.4g}, scale={params[2]:.4g}"
    elif name == "Uniform":
        return f"a={params[0]:.4g}, b={params[0]+params[1]:.4g}"
    elif name == "Triangular":
        return f"c={params[0]:.4g}, loc={params[1]:.4g}, scale={params[2]:.4g}"
    elif name == "Weibull":
        return f"k={params[0]:.4g}, loc={params[1]:.4g}, \u03bb={params[2]:.4g}"
    elif name == "Gamma":
        return f"\u03b1={params[0]:.4g}, loc={params[1]:.4g}, \u03b2={params[2]:.4g}"
    elif name == "Student-t":
        return f"\u03bd={params[0]:.1f}, loc={params[1]:.4g}, scale={params[2]:.4g}"
    elif name == "Beta":
        if len(params) >= 2:
            return f"\u03b1={params[0]:.4g}, \u03b2={params[1]:.4g}"
        return ", ".join(f"{p:.4g}" for p in params)
    return ", ".join(f"{p:.4g}" for p in params)


@dataclass
class ColumnStatistics:
    """Full statistics for one data column."""
    variable: str = ""
    unit: str = ""
    n: int = 0
    mean: float = 0.0
    median: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    skewness: float = 0.0
    kurtosis_val: float = 0.0
    ci95_low: float = 0.0
    ci95_high: float = 0.0
    dof: int = 0
    std_uncertainty_mean: float = 0.0
    std_uncertainty_pop: float = 0.0
    n_eff: float = 0.0
    autocorr_rho1: float = 0.0
    autocorr_tau: float = 1.0
    autocorr_warning: bool = False
    normality_shapiro_p: float = 0.0
    is_normal: bool = True
    best_fit: Optional[DistributionFitResult] = None
    all_fits: List[DistributionFitResult] = field(default_factory=list)
    candidate_set: List[str] = field(default_factory=list)
    candidate_set_note: str = ""
    best_fit_note: str = ""
    carry_distribution: str = "Normal"
    carry_method: str = "RSS"
    carry_note: str = ""
    sparse_replicate_mode: bool = False
    mirror_applied: bool = False
    n_raw: int = 0
    recommendation: str = ""


# =============================================================================
# ENGINE: Core statistical functions
# =============================================================================

def _estimate_effective_sample_size(values: np.ndarray) -> Tuple[float, float, float]:
    """Estimate effective sample size for autocorrelated data.

    Returns:
        (n_eff, rho_1, tau_int)
    where tau_int is the integrated autocorrelation time estimate.
    """
    n = len(values)
    if n < 3:
        return float(max(n, 1)), 0.0, 1.0

    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float(max(n, 1)), 0.0, 1.0

    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if denom <= 1e-15:
        return float(n), 0.0, 1.0

    max_lag = min(n - 1, max(1, int(np.sqrt(n))))
    rho_vals: List[float] = []
    for lag in range(1, max_lag + 1):
        num = float(np.dot(x[:-lag], x[lag:]))
        rho = num / denom
        if not np.isfinite(rho):
            break
        # Practical truncation: stop after correlation turns negative.
        if lag > 1 and rho < 0:
            break
        rho_vals.append(rho)

    if not rho_vals:
        return float(n), 0.0, 1.0

    tau_int = 1.0
    for lag, rho in enumerate(rho_vals, start=1):
        # Bartlett taper reduces overestimation for finite windows.
        tau_int += 2.0 * max(0.0, rho) * (1.0 - lag / n)
    tau_int = max(1.0, tau_int)
    n_eff = float(n) / tau_int
    n_eff = min(float(n), max(2.0, n_eff))
    return n_eff, float(rho_vals[0]), float(tau_int)


def _build_candidate_set(fits: List[DistributionFitResult]) -> Tuple[List[str], str]:
    """Build AIC-close candidate set and guidance note."""
    if not fits:
        return [], ""

    candidates = [f for f in fits if f.delta_aic <= 2.0]
    if len(candidates) <= 1:
        return [fits[0].name], "Single clear best-fit distribution by AICc."

    summary = ", ".join(
        f"{f.name} ({100.0 * f.akaike_weight:.1f}%)" for f in candidates
    )
    note = (
        "Multiple distributions are statistically close (delta AICc <= 2). "
        "Use Monte Carlo model-mixture carry-over instead of forcing one shape: "
        f"{summary}."
    )
    return [f.name for f in candidates], note


def _resolve_gof_mc_samples(n: int, requested: Optional[int]) -> int:
    """Resolve bootstrap GOF sample count with runtime guardrails.

    Rules:
    - Explicit request always wins (clamped at >= 0).
    - Sparse replicate regime (N <= 8): skip bootstrap GOF (0 samples).
    - Otherwise use adaptive defaults to keep GUI latency manageable.
    """
    if requested is not None:
        return max(0, int(requested))
    if n <= 8:
        return 0
    if n <= 30:
        return 49
    if n <= 200:
        return 39
    if n <= 1000:
        return 29
    return 19


def compute_column_statistics(col: ColumnData,
                              _n_mc_gof: Optional[int] = None) -> ColumnStatistics:
    """Compute full statistics for a single data column."""
    vals = col.analysis_values()  # applies mirroring if enabled
    n_raw = len(col.valid_values())
    n = len(vals)
    if n < 2:
        return ColumnStatistics(variable=col.name, unit=col.unit, n=n,
                                recommendation="Insufficient data (N < 2).")

    mean_val = float(np.mean(vals))
    median_val = float(np.median(vals))
    std_val = float(np.std(vals, ddof=1))

    # Zero-variance check: all data values identical
    if std_val == 0.0:
        return ColumnStatistics(
            variable=col.name, unit=col.unit, n=n,
            mean=mean_val, median=median_val, std=0.0,
            min_val=mean_val, max_val=mean_val,
            skewness=0.0, kurtosis_val=0.0,
            ci95_low=mean_val, ci95_high=mean_val,
            dof=max(1, n - 1),
            std_uncertainty_mean=0.0, std_uncertainty_pop=0.0,
            n_eff=float(n), autocorr_rho1=0.0, autocorr_tau=1.0,
            normality_shapiro_p=1.0, is_normal=True,
            carry_distribution="Normal", carry_method="RSS",
            recommendation=(
                "All data values are identical (zero variance). "
                "Statistical tests and distribution fitting are not meaningful. "
                "The carry-over uncertainty is zero."
            ),
        )

    min_v = float(np.min(vals))
    max_v = float(np.max(vals))
    skew_val = float(skew(vals, bias=False)) if n >= 3 else 0.0
    kurt_val = float(kurtosis(vals, bias=False)) if n >= 3 else 0.0
    n_eff, rho1, tau_int = _estimate_effective_sample_size(vals)
    dof = max(1, int(round(n_eff - 1)))
    autocorr_warning = n_eff < (0.8 * n)

    # 95% CI for the mean
    se = std_val / np.sqrt(max(n_eff, 1.0))
    t_crit = t_dist.ppf(0.975, dof)
    ci_low = mean_val - t_crit * se
    ci_high = mean_val + t_crit * se

    # Standard uncertainty
    u_mean = se  # uncertainty of the mean
    u_pop = std_val  # population uncertainty

    # Normality test (Shapiro-Wilk, valid for 3 <= n <= 5000)
    shapiro_p = 1.0
    is_norm = True
    if 3 <= n <= 5000:
        try:
            _, shapiro_p = shapiro(vals)
            is_norm = shapiro_p > 0.05
        except Exception:
            pass

    # Fit distributions
    all_fits = fit_distributions(vals, _n_mc_gof=_n_mc_gof)
    best, best_note = _select_best_fit(all_fits)
    candidate_set, candidate_note = _build_candidate_set(all_fits)
    sparse_replicate_mode = (n <= 8)
    candidate_ambiguous = len(candidate_set) > 1

    no_robust_fit = not (best is not None and best.passed_gof)

    carry_dist, carry_method, carry_note = _recommend_carry_distribution(
        best_fit=best, is_normal=is_norm, n=n,
        skew_val=skew_val, min_val=min_v, max_val=max_v,
        candidate_ambiguous=candidate_ambiguous,
        sparse_replicate_mode=sparse_replicate_mode,
        no_robust_fit=no_robust_fit,
    )

    # Outlier detection (1.5 IQR rule)
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    n_outliers = int(np.sum((vals < q1 - 1.5 * iqr) | (vals > q3 + 1.5 * iqr)))

    # Build recommendation
    rec = _build_recommendation(
        n=n, skew_val=skew_val, kurt_val=kurt_val,
        min_val=min_v, max_val=max_v, is_normal=is_norm,
        best_fit=best, n_outliers=n_outliers,
        best_fit_note=best_note, carry_dist=carry_dist,
        carry_method=carry_method, carry_note=carry_note,
        n_eff=n_eff, rho1=rho1, autocorr_warning=autocorr_warning,
        candidate_note=candidate_note,
        sparse_replicate_mode=sparse_replicate_mode,
        no_robust_fit=no_robust_fit,
    )

    # Append warnings for GOF fallback and Beta clipping (M-01, M-07)
    n_gof_screen = sum(1 for f in all_fits if f.gof_method == "ks_screening")
    if n_gof_screen > 0:
        rec += (
            f" • Runtime guardrail: bootstrap GOF was skipped for {n_gof_screen} "
            "distribution(s), and KS screening was used instead (marked with *). "
            "This is expected for sparse-replicate data and keeps analysis responsive."
        )
    n_gof_fb = sum(1 for f in all_fits if f.gof_method == "ks_fallback")
    if n_gof_fb > 0:
        rec += (
            f" \u2022 \u26a0 Bootstrap GOF failed for {n_gof_fb} distribution(s); "
            "KS p-value was used as fallback (marked with \u2020 in the table). "
            "KS p-values are less reliable than bootstrap GOF for fitted parameters."
        )
    n_beta_clip = sum(1 for f in all_fits if f.beta_clipped)
    if n_beta_clip > 0:
        rec += (
            " \u2022 \u26a0 Beta distribution was fitted with boundary values "
            "clipped to [1e-6, 1-1e-6] to avoid log(0). "
            "If your data sits exactly at 0 or 1, the Beta fit may be approximate."
        )

    # Prepend mirror note to recommendation and carry_note when active
    _mirror_on = col.mirror and n_raw > 0 and n > n_raw
    if _mirror_on:
        _mirror_prefix = (
            f"DATA MIRRORED about {col.mirror_center:g} "
            f"(original N={n_raw}, mirrored N={n}). "
        )
        rec = _mirror_prefix + rec
        carry_note = _mirror_prefix + carry_note

    return ColumnStatistics(
        variable=col.name, unit=col.unit, n=n,
        mean=mean_val, median=median_val, std=std_val,
        min_val=min_v, max_val=max_v,
        skewness=skew_val, kurtosis_val=kurt_val,
        ci95_low=ci_low, ci95_high=ci_high,
        dof=dof,
        std_uncertainty_mean=u_mean,
        std_uncertainty_pop=u_pop,
        n_eff=n_eff,
        autocorr_rho1=rho1,
        autocorr_tau=tau_int,
        autocorr_warning=autocorr_warning,
        normality_shapiro_p=shapiro_p,
        is_normal=is_norm,
        best_fit=best,
        all_fits=all_fits,
        candidate_set=candidate_set,
        candidate_set_note=candidate_note,
        best_fit_note=best_note,
        carry_distribution=carry_dist,
        carry_method=carry_method,
        carry_note=carry_note,
        sparse_replicate_mode=sparse_replicate_mode,
        mirror_applied=_mirror_on,
        n_raw=n_raw,
        recommendation=rec,
    )


def fit_distributions(values: np.ndarray,
                      _n_mc_gof: Optional[int] = None) -> List[DistributionFitResult]:
    """Fit all candidate distributions and rank by AICc.

    Parameters
    ----------
    _n_mc_gof : Optional[int]
        Number of Monte Carlo resamples for bootstrap GOF.
        If None, an adaptive default is used.
        Set to 0 to skip bootstrap GOF and use KS screening.
    """
    n = len(values)
    if n < 5:
        return []
    n_mc_gof = _resolve_gof_mc_samples(n, _n_mc_gof)

    results = []
    for dist_name, dist_obj in DISTRIBUTIONS:
        try:
            fit_values = values
            # Special handling for distributions requiring positive data
            if dist_name in ("Log-Normal", "Weibull", "Gamma") and np.any(values <= 0):
                continue
            _beta_clipped = False
            if dist_name == "Beta":
                vmin, vmax = np.min(values), np.max(values)
                # Only consider beta for true proportions [0,1].
                # This avoids false positives for arbitrary positive variables.
                if n < 20:
                    continue
                if vmin >= -1e-9 and vmax <= 1.0 + 1e-9:
                    fit_values = np.clip(values, 1e-6, 1.0 - 1e-6)
                    _beta_clipped = True
                else:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    params = dist_obj.fit(fit_values, floc=0, fscale=1)
                    ll = np.sum(dist_obj.logpdf(fit_values, *params))
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    params = dist_obj.fit(values)
                    ll = np.sum(dist_obj.logpdf(values, *params))

            k = len(params)
            aic_standard = 2 * k - 2 * ll
            # AICc: corrected AIC for small-sample bias
            if n - k - 1 > 0:
                aic = aic_standard + (2 * k * (k + 1)) / (n - k - 1)
            else:
                aic = aic_standard  # fall back to standard AIC
            bic = k * np.log(n) - 2 * ll

            if np.isnan(aic) or np.isinf(aic):
                continue

            # KS test (diagnostic only; parameters are estimated from same data)
            ks_stat, ks_p = kstest(fit_values, dist_obj.cdf, args=params)

            gof_stat = np.inf
            gof_p = 0.0
            gof_method = "bootstrap_ad"
            _gof_fallback = False
            if n_mc_gof <= 0:
                # Intentional runtime guardrail path (not a failure).
                gof_stat = float(ks_stat)
                gof_p = float(ks_p)
                gof_method = "ks_screening"
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        gof = goodness_of_fit(
                            dist=dist_obj,
                            data=fit_values,
                            statistic='ad',
                            n_mc_samples=n_mc_gof,
                            random_state=0,
                        )
                    gof_stat = float(gof.statistic)
                    gof_p = float(gof.pvalue)
                except Exception:
                    # Fallback to heuristic KS p-value if bootstrap GOF fails.
                    gof_stat = float(ks_stat)
                    gof_p = float(ks_p)
                    gof_method = "ks_fallback"
                    _gof_fallback = True

            # Anderson-Darling test (only for select distributions)
            ad_stat = np.inf
            ad_crit_5 = 0.0
            passed_ad = False
            try:
                if dist_name == "Normal":
                    try:
                        ad_result = anderson(values, dist='norm', method='interpolate')
                    except TypeError:
                        ad_result = anderson(values, dist='norm')
                    ad_stat = ad_result.statistic
                    # 5% significance is index 2
                    ad_crit_5 = ad_result.critical_values[2]
                    passed_ad = ad_stat < ad_crit_5
                elif dist_name == "Log-Normal" and np.all(values > 0):
                    try:
                        ad_result = anderson(np.log(values), dist='norm', method='interpolate')
                    except TypeError:
                        ad_result = anderson(np.log(values), dist='norm')
                    ad_stat = ad_result.statistic
                    ad_crit_5 = ad_result.critical_values[2]
                    passed_ad = ad_stat < ad_crit_5
            except Exception:
                pass

            results.append(DistributionFitResult(
                name=dist_name, params=params,
                param_str=_format_dist_params(dist_name, params),
                aic=aic, bic=bic,
                gof_statistic=gof_stat, gof_pvalue=gof_p,
                gof_method=gof_method,
                passed_gof=(gof_p > 0.05),
                ks_statistic=ks_stat, ks_pvalue=ks_p,
                ad_statistic=ad_stat, ad_critical_5pct=ad_crit_5,
                passed_ks=(ks_p > 0.05), passed_ad=passed_ad,
                gof_fallback=_gof_fallback,
                beta_clipped=_beta_clipped,
            ))
        except Exception:
            continue

    results.sort(key=lambda r: r.aic)
    if results:
        min_aic = results[0].aic
        weights_raw = np.array([np.exp(-0.5 * (r.aic - min_aic)) for r in results])
        w_sum = float(np.sum(weights_raw))
        if w_sum <= 0:
            weights = np.ones(len(results)) / len(results)
        else:
            weights = weights_raw / w_sum
        for idx, r in enumerate(results):
            r.delta_aic = float(r.aic - min_aic)
            r.akaike_weight = float(weights[idx])
    for i, r in enumerate(results):
        r.rank = i + 1
    return results


def _select_best_fit(
    fits: List[DistributionFitResult],
) -> Tuple[Optional[DistributionFitResult], str]:
    """Choose the best fit with bootstrap GOF preference over pure AICc rank."""
    if not fits:
        return None, "No candidate distribution could be fitted."
    bootstrap_pass = [
        f for f in fits
        if f.passed_gof and f.gof_method == "bootstrap_ad"
    ]
    if bootstrap_pass:
        best = min(bootstrap_pass, key=lambda f: f.aic)
        if best.rank == 1:
            note = "Rank 1 fit also passed bootstrap goodness-of-fit."
        else:
            note = (
                f"Selected rank {best.rank} because rank 1 failed bootstrap "
                "goodness-of-fit."
            )
        return best, note
    gof_pass = [f for f in fits if f.passed_gof]
    if gof_pass:
        best = min(gof_pass, key=lambda f: f.aic)
        if best.gof_method == "ks_screening":
            return best, (
                "Bootstrap GOF was skipped by runtime guardrail; selected "
                "lowest-AICc fit that passed KS screening."
            )
        return best, (
            "Bootstrap GOF unavailable for top-ranked fits; selected "
            "lowest-AICc fit that passed KS fallback screening."
        )
    return fits[0], (
        "No fit passed GOF screening; using lowest-AICc fit as a fallback."
    )


def _recommend_carry_distribution(
    best_fit: Optional[DistributionFitResult], is_normal: bool, n: int,
    skew_val: float, min_val: float, max_val: float,
    candidate_ambiguous: bool = False,
    sparse_replicate_mode: bool = False,
    no_robust_fit: bool = False,
) -> Tuple[str, str, str]:
    """Map analysis output to a concrete Aggregator carry-over choice."""
    if sparse_replicate_mode:
        return (
            "Custom/Empirical (Bootstrap)",
            "Monte Carlo",
            "Sparse replicate mode (N <= 8): preserve discrete replicates rather than force a parametric shape.",
        )
    if candidate_ambiguous:
        return (
            "Custom/Empirical (Bootstrap)",
            "Monte Carlo",
            "Multiple close distribution candidates: propagate model-form uncertainty with AICc-weighted mixture sampling.",
        )

    if no_robust_fit:
        return (
            "Custom/Empirical (Bootstrap)",
            "Monte Carlo",
            "No candidate distribution passed GOF. Use empirical bootstrap carry-over (do not force a parametric shape).",
        )

    if best_fit:
        fit_name = best_fit.name
        if fit_name in AGGREGATOR_DIST_MAP:
            mapped = AGGREGATOR_DIST_MAP[fit_name]
            method = "Monte Carlo" if mapped in ("Lognormal", "Weibull") else "RSS"
            return mapped, method, f"Mapped from best-fit '{fit_name}'."
        if fit_name == "Student-t":
            fitted_df = best_fit.params[0] if best_fit.params else 5.0
            if fitted_df <= 7:
                mapped = "Student-t (df=5)"
            else:
                mapped = "Student-t (df=10)"
            note = f"Student-t mapped to nearest Aggregator option (fitted df={fitted_df:.1f})."
            if fitted_df < 3:
                note += (" WARNING: Very heavy tails (fitted df < 3). Consider "
                         "using Custom/Empirical (Bootstrap) with Monte Carlo "
                         "for more conservative tail coverage.")
            return mapped, "RSS", note
        if fit_name == "Gamma":
            if min_val > 0 and skew_val > 0.5:
                return (
                    "Lognormal", "Monte Carlo",
                    "Gamma is not native in the Aggregator; Lognormal is a positive-skew surrogate."
                )
            return (
                "Custom/Empirical (Bootstrap)", "Monte Carlo",
                "Gamma is not native in the Aggregator; bootstrap avoids forcing the wrong shape."
            )
        if fit_name == "Beta":
            return (
                "Custom/Empirical (Bootstrap)", "Monte Carlo",
                "Beta is not native in the Aggregator; bootstrap keeps bounded-shape behavior."
            )

    if is_normal:
        return "Normal", "RSS", "Normal fallback from Shapiro-Wilk."
    if min_val > 0 and skew_val > 0.5:
        return "Lognormal", "Monte Carlo", "Positive-skew fallback for non-normal data."
    if n < 30:
        return "Student-t (df=5)", "RSS", "Small-sample conservative fallback."
    return (
        "Custom/Empirical (Bootstrap)", "Monte Carlo",
        "Non-normal fallback when no reliable parametric fit is available."
    )


def _build_recommendation(
    n, skew_val, kurt_val, min_val, max_val, is_normal, best_fit,
    n_outliers=0, best_fit_note="", carry_dist="Normal",
    carry_method="RSS", carry_note="", n_eff=0.0, rho1=0.0,
    autocorr_warning=False, candidate_note="", sparse_replicate_mode=False,
    no_robust_fit=False,
):
    """Build a human-readable recommendation string."""

    # --- SHORT-CIRCUIT: sparse replicate mode ---
    if sparse_replicate_mode:
        parts = [
            f"Sparse replicate mode (N={n}, 8 or fewer samples). "
            "Distribution fitting is not reliable at this sample size.",
            "ACTION REQUIRED: Use 'Custom/Empirical (Bootstrap)' with "
            "'Monte Carlo' in the Aggregator.",
            "This preserves your actual data values rather than forcing "
            "a potentially wrong shape.",
        ]
        if n_outliers > 0:
            parts.append(f"\u26a0 {n_outliers} outlier(s) detected (1.5\u00d7IQR rule) "
                          "\u2014 investigate before excluding.")
        if autocorr_warning:
            parts.append("Data are correlated; confidence is reduced.")
        return " \u2022 ".join(parts)

    # --- SHORT-CIRCUIT: no robust fit (all GOF failed) ---
    if no_robust_fit:
        parts = [
            f"ALERT: None of the candidate distributions passed goodness-of-fit "
            f"testing (all p-values < 0.05). Your data does not match any "
            f"standard statistical shape.",
            "ACTION REQUIRED: Use 'Custom/Empirical (Bootstrap)' with "
            "'Monte Carlo' in the Aggregator. Do NOT force a parametric distribution.",
            f"[Sample size: N={n}]",
        ]
        # Only append critical quality notes
        if n_outliers > 0:
            parts.append(f"\u26a0 {n_outliers} outlier(s) detected (1.5\u00d7IQR rule) "
                          "\u2014 investigate before excluding.")
        if autocorr_warning:
            parts.append(
                f"Data are correlated (N_eff={n_eff:.1f}, lag-1 rho={rho1:.3f}); "
                "confidence is reduced."
            )
        return " \u2022 ".join(parts)

    # --- NORMAL PATH: at least one distribution passed GOF ---
    parts = []
    if n < 10:
        parts.append(f"Insufficient data (N={n}) per GUM \u00a7G.3 \u2014 "
                      "consider collecting more samples.")
    elif n < 30:
        parts.append(f"Marginal sample size (N={n}) \u2014 "
                      "results have higher uncertainty.")
    else:
        parts.append(f"Adequate sample size (N={n}).")
    if n > 50000:
        parts.append(
            f"\u26a0 Very large dataset (N={n:,}). Bootstrap GOF and "
            "distribution fitting may be slow. Consider subsampling if "
            "computation time is excessive."
        )

    if n_eff > 0 and n_eff < n:
        parts.append(
            f"Autocorrelation-adjusted effective sample size is N_eff={n_eff:.1f} "
            f"(lag-1 rho={rho1:.3f})."
        )
    if autocorr_warning:
        parts.append("Data are correlated; confidence is reduced.")

    if n_outliers > 0:
        parts.append(f"\u26a0 {n_outliers} outlier(s) detected (1.5\u00d7IQR rule) "
                      "\u2014 investigate before excluding.")
    if abs(kurt_val) > 3:
        parts.append("Heavy tails detected \u2014 Student-t or bootstrap is safer.")
    if abs(skew_val) > 1:
        parts.append(
            "Significant skewness detected. If the variable is positive-only, "
            "prefer Log-Normal/Weibull over Normal."
        )
    if not is_normal:
        parts.append(
            "Data is NOT normally distributed (Shapiro-Wilk p < 0.05). "
            "Do not force Normal just to simplify the setup."
        )
    else:
        parts.append("Data appears normally distributed (Shapiro-Wilk p \u2265 0.05).")

    if best_fit:
        parts.append(
            f"Best-fit distribution: {best_fit.name} "
            f"(AICc={best_fit.aic:.1f}, bootstrap GOF p={best_fit.gof_pvalue:.3f})."
        )
        if best_fit_note:
            parts.append(best_fit_note)
    if candidate_note:
        parts.append(candidate_note)

    if min_val >= 0 and max_val <= 1:
        parts.append(
            "Variable is bounded in [0,1]. Keep bounded behavior in the final model."
        )

    parts.append(
        f"Carry-over decision: use distribution '{carry_dist}' with sigma = this "
        f"column's sample standard deviation. Recommended Aggregator analysis mode "
        f"for this variable: '{carry_method}'."
    )
    parts.append(
        "Aggregator mode is global (set once). If any variable recommends Monte Carlo, "
        "run the Aggregator in Monte Carlo mode."
    )
    if carry_note:
        parts.append(f"Carry-over rationale: {carry_note}")

    return " \u2022 ".join(parts)


# =============================================================================
# TAB 1: DATA INPUT
# =============================================================================

class DataInputTab(QWidget):
    """CSV import, clipboard paste, or manual entry of data columns."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: List[ColumnData] = [ColumnData(name="Variable 1")]
        self._results_exist_fn = None  # Set by main window to check if results exist
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # --- Left: controls ---
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)

        # Import group
        grp_import = QGroupBox("Import Data")
        gi_lay = QVBoxLayout(grp_import)
        self._btn_load_csv = QPushButton("Load CSV...")
        self._btn_load_csv.setToolTip(
            "Import a CSV file with one or more columns of numeric data.\n"
            "First row is treated as header (column names).\n"
            "Non-numeric values and blanks are treated as missing data.")
        self._btn_load_csv.clicked.connect(self._load_csv)
        gi_lay.addWidget(self._btn_load_csv)
        self._btn_paste = QPushButton("Paste from Clipboard")
        self._btn_paste.setToolTip(
            "Paste tab-separated data from Excel or a spreadsheet.\n"
            "Select a range in Excel, copy (Ctrl+C), then click here.")
        self._btn_paste.clicked.connect(self._paste_clipboard)
        gi_lay.addWidget(self._btn_paste)
        self._btn_example = QPushButton("Load Example Dataset \u25bc")
        example_menu = QMenu(self._btn_example)
        example_menu.addAction("Example 1: CFD Validation (N=60)", self._load_example_dataset)
        example_menu.addAction("Example 2: Small Sample (N=6)", self._load_example_small_sample)
        example_menu.addAction("Example 3: Autocorrelated Data (AR(1))", self._load_example_autocorrelated)
        self._btn_example.setMenu(example_menu)
        self._btn_example.setToolTip(
            "Load a built-in example dataset to explore\n"
            "the tool's features without importing your own data.")
        gi_lay.addWidget(self._btn_example)
        self._btn_clear = QPushButton("Clear All Data")
        self._btn_clear.setToolTip("Remove all data from the table and reset columns.")
        self._btn_clear.clicked.connect(self._clear_data)
        gi_lay.addWidget(self._btn_clear)
        left_lay.addWidget(grp_import)

        # Column management
        grp_col = QGroupBox("Column Management")
        gc_lay = QVBoxLayout(grp_col)
        col_row = QHBoxLayout()
        self._btn_add_col = QPushButton("+ Add Column")
        self._btn_add_col.setToolTip("Add another data variable (column) to the analysis.")
        self._btn_add_col.clicked.connect(self._add_column)
        col_row.addWidget(self._btn_add_col)
        self._btn_del_col = QPushButton("- Remove Column")
        self._btn_del_col.setToolTip("Remove the currently selected column and its data.")
        self._btn_del_col.clicked.connect(self._remove_column)
        col_row.addWidget(self._btn_del_col)
        gc_lay.addLayout(col_row)

        # Active column selector
        form = QFormLayout()
        self._cmb_active_col = QComboBox()
        self._cmb_active_col.setToolTip(
            "Select which column to edit (name, unit) or view in the table.\n"
            "Each column is treated as an independent variable for analysis.")
        self._cmb_active_col.currentIndexChanged.connect(self._on_active_col_changed)
        form.addRow("Active Column:", self._cmb_active_col)
        gc_lay.addLayout(form)

        # Column properties
        self._edit_col_name = QLineEdit("Variable 1")
        self._edit_col_name.setToolTip(
            "Descriptive name for this variable (e.g., 'TC-01 Reading').\n"
            "This label appears in tables, charts, carry-over output, and reports.")
        self._edit_col_name.textChanged.connect(self._on_col_name_changed)
        form2 = QFormLayout()
        form2.addRow("Name:", self._edit_col_name)

        self._cmb_unit_cat = QComboBox()
        self._cmb_unit_cat.setToolTip(
            "Select a unit category to populate the Unit dropdown.\n"
            "Categories: Temperature, Pressure, Velocity, Length, etc.")
        self._cmb_unit_cat.addItems(list(UNIT_PRESETS.keys()))
        self._cmb_unit_cat.currentTextChanged.connect(self._on_unit_cat_changed)
        form2.addRow("Unit Category:", self._cmb_unit_cat)

        self._cmb_unit = QComboBox()
        self._cmb_unit.setEditable(True)
        self._cmb_unit.setToolTip(
            "Select a preset unit or type a custom unit string.\n"
            "Examples: \u00b0F, K, Pa, psia, m/s, BTU/hr")
        self._cmb_unit.currentTextChanged.connect(self._on_unit_changed)
        form2.addRow("Unit:", self._cmb_unit)

        gc_lay.addLayout(form2)

        # Data mirroring controls
        grp_mirror = QGroupBox("Advanced: Data Mirroring")
        grp_mirror.setCheckable(True)
        grp_mirror.setChecked(False)
        grp_mirror.setToolTip(
            "Mirror data about a center value to create a symmetric dataset.\n"
            "Use this when you measured only one side of a symmetric phenomenon\n"
            "(e.g., thermal gradient on one side of a symmetric geometry)\n"
            "and assume the opposite side behaves identically.\n\n"
            "Example: [1.2, 0.8, 1.5] mirrored about 0\n"
            "becomes [-1.5, -1.2, -0.8, 0.8, 1.2, 1.5].\n"
            "The sample size doubles, giving the correct symmetric statistics.")
        gm_lay = QVBoxLayout(grp_mirror)
        self._chk_mirror = QCheckBox("Mirror about center value")
        self._chk_mirror.setToolTip(
            "When checked, each data point is reflected about the center\n"
            "value during analysis. The original data in the table is\n"
            "not modified — mirroring is applied only when computing statistics.")
        self._chk_mirror.stateChanged.connect(self._on_mirror_changed)
        gm_lay.addWidget(self._chk_mirror)
        mirror_form = QFormLayout()
        self._spn_mirror_center = QDoubleSpinBox()
        self._spn_mirror_center.setRange(-1e12, 1e12)
        self._spn_mirror_center.setDecimals(6)
        self._spn_mirror_center.setValue(0.0)
        self._spn_mirror_center.setToolTip(
            "The center point about which data is reflected.\n"
            "Most common value is 0 (e.g., for deviations or deltas).\n"
            "For temperature offsets, this might be a reference temperature.")
        self._spn_mirror_center.valueChanged.connect(self._on_mirror_center_changed)
        mirror_form.addRow("Center:", self._spn_mirror_center)
        gm_lay.addLayout(mirror_form)
        self._lbl_mirror_preview = QLabel("")
        self._lbl_mirror_preview.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; font-size: 10px;")
        self._lbl_mirror_preview.setWordWrap(True)
        gm_lay.addWidget(self._lbl_mirror_preview)
        gc_lay.addWidget(grp_mirror)

        left_lay.addWidget(grp_col)

        # Row management
        grp_rows = QGroupBox("Row Management")
        gr_lay = QHBoxLayout(grp_rows)
        self._spn_rows = QSpinBox()
        self._spn_rows.setRange(1, 100000)
        self._spn_rows.setValue(20)
        self._spn_rows.setToolTip(
            "Number of data rows to show in the table.\n"
            "Increase for larger datasets; decrease for manual entry.")
        gr_lay.addWidget(QLabel("Rows:"))
        gr_lay.addWidget(self._spn_rows)
        self._btn_set_rows = QPushButton("Set Rows")
        self._btn_set_rows.setToolTip("Apply the row count to the data table.")
        self._btn_set_rows.clicked.connect(self._set_rows)
        gr_lay.addWidget(self._btn_set_rows)
        left_lay.addWidget(grp_rows)

        # Guidance
        self._guidance = GuidancePanel("Data Input")
        self._guidance.set_guidance(
            "Load a CSV file, paste tab-separated data from the clipboard, "
            "or type values directly into the table. Use column management "
            "to add/remove variables and set units.",
            'green')
        left_lay.addWidget(self._guidance)
        left_lay.addStretch()

        splitter.addWidget(left)

        # --- Right: data table ---
        right = QWidget()
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(4, 4, 4, 4)

        self._data_table = QTableWidget()
        self._data_table.setAlternatingRowColors(True)
        self._data_table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self._data_table.cellChanged.connect(self._on_cell_changed)
        self._suppress_cell_validation = False
        right_lay.addWidget(self._data_table)

        # Data summary
        self._lbl_summary = QLabel("No data loaded.")
        self._lbl_summary.setStyleSheet(f"color: {DARK_COLORS['fg_dim']}; font-size: 11px;")
        right_lay.addWidget(self._lbl_summary)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 3)

        # Initialize table
        self._rebuild_table()
        self._on_unit_cat_changed(self._cmb_unit_cat.currentText())

    def _on_cell_changed(self, row, col):
        """Validate manual cell entry; flash red and show message if invalid."""
        if self._suppress_cell_validation:
            return
        item = self._data_table.item(row, col)
        if item is None:
            return
        text = item.text().strip()
        if not text:
            # Even clearing a cell is a data modification
            if self._results_exist_fn and self._results_exist_fn():
                self._guidance.set_guidance(
                    "Data modified \u2014 re-run analysis for updated results.",
                    'yellow')
            return
        try:
            float(text)
            # Valid numeric edit — check for stale results
            if self._results_exist_fn and self._results_exist_fn():
                self._guidance.set_guidance(
                    "Data modified \u2014 re-run analysis for updated results.",
                    'yellow')
        except ValueError:
            # Flash cell red to indicate rejected value
            original_bg = item.background()
            item.setBackground(QColor(DARK_COLORS['red']))
            self._guidance.set_guidance(
                f"Rejected non-numeric value \"{text}\" in row {row + 1}, "
                f"column {col + 1}. Only numeric values are accepted.",
                'yellow')
            # Clear the invalid text
            self._data_table.blockSignals(True)
            item.setText("")
            self._data_table.blockSignals(False)
            # Restore background after 600ms
            QTimer.singleShot(600, lambda it=item, bg=original_bg: it.setBackground(bg))

    def _rebuild_table(self):
        self._suppress_cell_validation = True
        n_cols = len(self._columns)
        self._data_table.setColumnCount(n_cols)
        headers = []
        for col in self._columns:
            h = f"{col.name} ({col.unit})" if col.unit else col.name
            headers.append(h)
        self._data_table.setHorizontalHeaderLabels(headers)
        self._data_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        # Populate data
        max_rows = max(len(c.values) for c in self._columns) if self._columns else 20
        max_rows = max(max_rows, 20)
        self._data_table.setRowCount(max_rows)
        for ci, col in enumerate(self._columns):
            for ri, val in enumerate(col.values):
                if np.isfinite(val):
                    item = QTableWidgetItem(f"{val}")
                else:
                    item = QTableWidgetItem("")
                self._data_table.setItem(ri, ci, item)
        self._suppress_cell_validation = False
        self._update_combo()
        self._update_summary()

    def _update_combo(self):
        saved_idx = self._cmb_active_col.currentIndex()
        self._cmb_active_col.blockSignals(True)
        self._cmb_active_col.clear()
        for i, col in enumerate(self._columns):
            self._cmb_active_col.addItem(f"{i+1}: {col.name}")
        if 0 <= saved_idx < self._cmb_active_col.count():
            self._cmb_active_col.setCurrentIndex(saved_idx)
        self._cmb_active_col.blockSignals(False)

    def _update_summary(self):
        total_vals = 0
        for col in self._columns:
            total_vals += len(col.valid_values())
        self._lbl_summary.setText(
            f"{len(self._columns)} column(s), {total_vals} valid value(s) total.")

    def _sync_table_to_columns(self):
        """Read current table contents back into column data."""
        for ci, col in enumerate(self._columns):
            vals = []
            last_finite_idx = -1
            for ri in range(self._data_table.rowCount()):
                item = self._data_table.item(ri, ci)
                if item and item.text().strip():
                    try:
                        value = float(item.text().strip())
                    except ValueError:
                        value = float('nan')
                else:
                    value = float('nan')
                vals.append(value)
                if np.isfinite(value):
                    last_finite_idx = ri
            col.values = vals[:last_finite_idx + 1] if last_finite_idx >= 0 else []
        self._update_summary()

    def _add_column(self):
        idx = len(self._columns) + 1
        self._columns.append(ColumnData(name=f"Variable {idx}"))
        self._rebuild_table()

    def _remove_column(self):
        if len(self._columns) <= 1:
            QMessageBox.information(self, "Info", "Cannot remove the last column.")
            return
        idx = self._cmb_active_col.currentIndex()
        if 0 <= idx < len(self._columns):
            self._sync_table_to_columns()
            self._columns.pop(idx)
            self._rebuild_table()

    def _on_active_col_changed(self, idx):
        if 0 <= idx < len(self._columns):
            col = self._columns[idx]
            self._edit_col_name.blockSignals(True)
            self._edit_col_name.setText(col.name)
            self._edit_col_name.blockSignals(False)
            self._chk_mirror.blockSignals(True)
            self._chk_mirror.setChecked(col.mirror)
            self._chk_mirror.blockSignals(False)
            self._spn_mirror_center.blockSignals(True)
            self._spn_mirror_center.setValue(col.mirror_center)
            self._spn_mirror_center.blockSignals(False)
            # Sync unit category and unit combos to selected column
            self._cmb_unit_cat.blockSignals(True)
            self._cmb_unit_cat.setCurrentText(col.unit_category)
            self._cmb_unit_cat.blockSignals(False)
            self._cmb_unit.blockSignals(True)
            self._cmb_unit.clear()
            cat_units = UNIT_PRESETS.get(col.unit_category, [])
            self._cmb_unit.addItems(cat_units)
            self._cmb_unit.setCurrentText(col.unit)
            self._cmb_unit.blockSignals(False)
            self._update_mirror_preview()

    def _on_col_name_changed(self, text):
        idx = self._cmb_active_col.currentIndex()
        if 0 <= idx < len(self._columns):
            self._columns[idx].name = text
            h = f"{text} ({self._columns[idx].unit})" if self._columns[idx].unit else text
            self._data_table.setHorizontalHeaderItem(
                idx, QTableWidgetItem(h))
            self._update_combo()

    def _on_unit_cat_changed(self, cat):
        self._cmb_unit.blockSignals(True)
        self._cmb_unit.clear()
        units = UNIT_PRESETS.get(cat, [])
        self._cmb_unit.addItems(units)
        self._cmb_unit.blockSignals(False)
        idx = self._cmb_active_col.currentIndex()
        if 0 <= idx < len(self._columns):
            self._columns[idx].unit_category = cat
            current_unit = self._columns[idx].unit
            if current_unit and current_unit in units:
                # Preserve existing unit if it is valid in the new category
                self._cmb_unit.setCurrentText(current_unit)
            elif units:
                self._columns[idx].unit = units[0]
                self._cmb_unit.setCurrentText(units[0])

    def _on_unit_changed(self, unit_text):
        idx = self._cmb_active_col.currentIndex()
        if 0 <= idx < len(self._columns):
            self._columns[idx].unit = unit_text
            name = self._columns[idx].name
            h = f"{name} ({unit_text})" if unit_text else name
            self._data_table.setHorizontalHeaderItem(
                idx, QTableWidgetItem(h))

    def _on_mirror_changed(self, state):
        idx = self._cmb_active_col.currentIndex()
        if 0 <= idx < len(self._columns):
            self._columns[idx].mirror = bool(state)
            self._update_mirror_preview()

    def _on_mirror_center_changed(self, val):
        idx = self._cmb_active_col.currentIndex()
        if 0 <= idx < len(self._columns):
            self._columns[idx].mirror_center = val
            self._update_mirror_preview()

    def _update_mirror_preview(self):
        idx = self._cmb_active_col.currentIndex()
        if idx < 0 or idx >= len(self._columns):
            self._lbl_mirror_preview.setText("")
            return
        col = self._columns[idx]
        if not col.mirror:
            self._lbl_mirror_preview.setText("Mirroring is off.")
            return
        vals = col.valid_values()
        n_raw = len(vals)
        if n_raw == 0:
            self._lbl_mirror_preview.setText("No data to mirror.")
            return
        self._lbl_mirror_preview.setText(
            f"Original: {n_raw} points \u2192 Mirrored: {2 * n_raw} points "
            f"(about center = {col.mirror_center:g})")

    def _set_rows(self):
        n = self._spn_rows.value()
        # Check if reducing rows would truncate existing data
        current_rows = self._data_table.rowCount()
        if n < current_rows:
            max_data_row = -1
            for ci in range(self._data_table.columnCount()):
                for ri in range(n, current_rows):
                    item = self._data_table.item(ri, ci)
                    if item and item.text().strip():
                        max_data_row = max(max_data_row, ri)
            if max_data_row >= n:
                reply = QMessageBox.question(
                    self, "Confirm Row Reduction",
                    f"Reducing rows from {current_rows} to {n} will discard "
                    f"data in rows {n + 1}\u2013{max_data_row + 1}. Continue?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply != QMessageBox.Yes:
                    self._spn_rows.blockSignals(True)
                    self._spn_rows.setValue(current_rows)
                    self._spn_rows.blockSignals(False)
                    return
        self._data_table.setRowCount(n)

    @staticmethod
    def _parse_tabular_text(text: str) -> Tuple[List[str], List[List[float]], bool]:
        """Parse CSV/TSV block and return headers, data rows, and header flag.

        NOTE: Header detection uses a heuristic — if any cell in the first row
        cannot be parsed as a float, the row is treated as a header. This means
        purely numeric column names (e.g. "1", "2.5") will be misidentified as
        data. This is a reasonable trade-off for the common case.
        """
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return [], [], False

        delim = '\t' if '\t' in lines[0] else ','
        first_parts = [p.strip().strip('"').strip("'") for p in lines[0].split(delim)]
        has_header = False
        for p in first_parts:
            try:
                float(p)
            except ValueError:
                has_header = True
                break

        start_idx = 1 if has_header else 0
        headers = first_parts if has_header else []
        data_rows: List[List[float]] = []
        for line in lines[start_idx:]:
            row: List[float] = []
            for cell in line.split(delim):
                v = cell.strip().strip('"').strip("'")
                try:
                    row.append(float(v))
                except ValueError:
                    row.append(float('nan'))
            data_rows.append(row)

        if not data_rows:
            return headers, [], has_header

        n_cols = max(len(r) for r in data_rows)
        for r in data_rows:
            while len(r) < n_cols:
                r.append(float('nan'))
        if not headers:
            headers = [f"Variable {i+1}" for i in range(n_cols)]
        return headers, data_rows, has_header

    def _replace_dataset(self, headers: List[str], data_rows: List[List[float]]):
        """Replace all existing data with a new table."""
        n_cols = max(len(r) for r in data_rows)
        self._columns = []
        for ci in range(n_cols):
            name = headers[ci] if ci < len(headers) else f"Variable {ci+1}"
            vals = [r[ci] for r in data_rows]
            self._columns.append(ColumnData(name=name, values=vals))
        self._rebuild_table()
        # Sync mirror UI controls to the new active column
        self._on_active_col_changed(self._cmb_active_col.currentIndex())

    def _table_is_effectively_empty(self) -> bool:
        self._sync_table_to_columns()
        for c in self._columns:
            if len(c.valid_values()) > 0:
                return False
        return True

    def _paste_into_existing_columns(
        self, headers: List[str], data_rows: List[List[float]], has_header: bool
    ) -> Tuple[int, int]:
        """Paste rows into the selected cell anchor instead of forcing col 1."""
        self._sync_table_to_columns()

        anchor_row = self._data_table.currentRow()
        anchor_col = self._data_table.currentColumn()
        if anchor_row < 0:
            anchor_row = 0
        if anchor_col < 0:
            anchor_col = 0

        n_rows = len(data_rows)
        n_cols = len(data_rows[0]) if data_rows else 0
        need_cols = anchor_col + n_cols
        need_rows = anchor_row + n_rows

        while len(self._columns) < need_cols:
            idx = len(self._columns) + 1
            self._columns.append(ColumnData(name=f"Variable {idx}"))

        for col in self._columns:
            if len(col.values) < need_rows:
                col.values.extend([float('nan')] * (need_rows - len(col.values)))

        for r_idx, row in enumerate(data_rows):
            for c_idx, val in enumerate(row):
                tgt_row = anchor_row + r_idx
                tgt_col = anchor_col + c_idx
                self._columns[tgt_col].values[tgt_row] = val

        if has_header:
            for c_idx, name in enumerate(headers):
                tgt_col = anchor_col + c_idx
                if tgt_col >= len(self._columns):
                    break
                existing = self._columns[tgt_col].name.strip()
                if not existing or existing.startswith("Variable "):
                    self._columns[tgt_col].name = name

        self._rebuild_table()
        self._data_table.setCurrentCell(anchor_row, anchor_col)
        return n_rows, n_cols

    def _load_csv(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load CSV", os.path.expanduser("~"),
            "CSV Files (*.csv);;All Files (*)")
        if not filepath:
            return
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                text = f.read()
            if not text.strip():
                QMessageBox.warning(self, "Empty File", "The file is empty.")
                return
            headers, data_rows, _ = self._parse_tabular_text(text)
            if not data_rows:
                QMessageBox.warning(self, "No Data", "No numeric data found.")
                return
            n_cols = max(len(r) for r in data_rows)
            self._replace_dataset(headers, data_rows)
            self._guidance.set_guidance(
                f"Loaded {len(data_rows)} rows \u00d7 {n_cols} columns "
                f"from {os.path.basename(filepath)}.", 'green')
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Error loading CSV:\n\n{e}")

    def _paste_clipboard(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text or not text.strip():
            QMessageBox.information(self, "Empty Clipboard",
                                    "No text data on the clipboard.")
            return
        headers, data_rows, has_header = self._parse_tabular_text(text)
        if not data_rows:
            QMessageBox.warning(self, "No Data", "No numeric data found on clipboard.")
            return
        n_cols = max(len(r) for r in data_rows)

        # Count NaN values introduced from non-numeric cells
        nan_count = sum(
            1 for row in data_rows for val in row if np.isnan(val)
        )

        if has_header and self._table_is_effectively_empty():
            self._replace_dataset(headers, data_rows)
            msg = f"Pasted {len(data_rows)} rows \u00d7 {n_cols} columns as a new dataset."
            if nan_count > 0:
                self._guidance.set_guidance(
                    f"{msg} Warning: {nan_count} non-numeric value(s) were "
                    f"converted to empty cells.", 'yellow')
            else:
                self._guidance.set_guidance(msg, 'green')
            return

        n_rows, n_cols = self._paste_into_existing_columns(headers, data_rows, has_header)
        msg = f"Pasted {n_rows} rows \u00d7 {n_cols} columns at selected cell anchor."
        if nan_count > 0:
            self._guidance.set_guidance(
                f"{msg} Warning: {nan_count} non-numeric value(s) were "
                f"converted to empty cells.", 'yellow')
        else:
            self._guidance.set_guidance(msg, 'green')

    def _load_example_dataset(self):
        """Populate a reproducible CFD-style example for quick validation."""
        rng = np.random.default_rng(42)
        n = 60

        wall_temp_k = rng.normal(loc=820.0, scale=7.5, size=n)
        pressure_drop_pa = rng.lognormal(mean=np.log(1450.0), sigma=0.09, size=n)
        cooling_eff = np.clip(rng.beta(10.0, 3.0, size=n), 0.0, 1.0)

        self._columns = [
            ColumnData(
                name="Wall Temperature", unit="K", unit_category="Temperature",
                values=[float(v) for v in wall_temp_k],
            ),
            ColumnData(
                name="Pressure Drop", unit="Pa", unit_category="Pressure",
                values=[float(v) for v in pressure_drop_pa],
            ),
            ColumnData(
                name="Cooling Effectiveness", unit="—", unit_category="Dimensionless",
                values=[float(v) for v in cooling_eff],
            ),
        ]
        self._rebuild_table()
        self._guidance.set_guidance(
            "Loaded built-in example dataset (N=60): one near-normal variable, "
            "one right-skewed positive variable, and one bounded [0,1] variable.",
            'green'
        )

    def _load_example_small_sample(self):
        """Populate a small-sample example (N=6) to trigger sparse replicate mode."""
        self._columns = [
            ColumnData(
                name="Thermocouple Temperature", unit="\u00b0F", unit_category="Temperature",
                values=[451.2, 448.7, 453.1, 449.5, 450.8, 452.3],
            ),
        ]
        self._rebuild_table()
        self._guidance.set_guidance(
            "Loaded small-sample example (N=6): thermocouple readings in \u00b0F. "
            "With fewer than ~20 replicates the tool activates sparse replicate mode "
            "and recommends bootstrap intervals.",
            'green'
        )

    def _load_example_autocorrelated(self):
        """Populate an AR(1) autocorrelated dataset (N=200, rho\u22480.7)."""
        rng = np.random.default_rng(42)
        n = 200
        mu = 450.0       # mean temperature, \u00b0F
        sigma = 3.0       # marginal standard deviation
        phi = 0.7         # AR(1) coefficient  (lag-1 autocorrelation)

        # Generate AR(1) process: x_t = mu + phi*(x_{t-1} - mu) + innovation
        innovation_std = sigma * np.sqrt(1.0 - phi ** 2)
        x = np.empty(n)
        x[0] = mu + rng.normal(0.0, sigma)          # stationary start
        for t in range(1, n):
            x[t] = mu + phi * (x[t - 1] - mu) + rng.normal(0.0, innovation_std)

        self._columns = [
            ColumnData(
                name="Exhaust Temperature", unit="\u00b0F", unit_category="Temperature",
                values=[float(v) for v in x],
            ),
        ]
        self._rebuild_table()
        self._guidance.set_guidance(
            "Loaded autocorrelated example (N=200, AR(1) \u03c1\u22480.7): exhaust temperature "
            "readings in \u00b0F. The effective sample size N_eff should be much less "
            "than 200 due to serial correlation.",
            'green'
        )

    def _clear_data(self):
        self._columns = [ColumnData(name="Variable 1")]
        self._rebuild_table()
        self._guidance.set_guidance("Data cleared.", 'yellow')

    def get_columns(self) -> List[ColumnData]:
        """Return columns with current table data synced."""
        self._sync_table_to_columns()
        return self._columns

    def get_state(self) -> dict:
        self._sync_table_to_columns()
        return {
            'columns': [
                {'name': c.name, 'unit': c.unit, 'unit_category': c.unit_category,
                 'values': [v if np.isfinite(v) else None for v in c.values],
                 'mirror': c.mirror, 'mirror_center': c.mirror_center}
                for c in self._columns
            ]
        }

    def set_state(self, state: dict):
        cols_data = state.get('columns', [])
        self._columns = []
        for cd in cols_data:
            vals = [v if v is not None else float('nan') for v in cd.get('values', [])]
            self._columns.append(ColumnData(
                name=cd.get('name', 'Variable'),
                unit=cd.get('unit', ''),
                unit_category=cd.get('unit_category', 'Other'),
                values=vals,
                mirror=cd.get('mirror', False),
                mirror_center=cd.get('mirror_center', 0.0),
            ))
        if not self._columns:
            self._columns = [ColumnData(name="Variable 1")]
        self._rebuild_table()


# =============================================================================
# TAB 2: STATISTICS
# =============================================================================

class StatisticsTab(QWidget):
    """Summary statistics, distribution fitting, goodness-of-fit."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results: List[ColumnStatistics] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        self._btn_compute = QPushButton("Compute Statistics")
        self._btn_compute.setToolTip(
            "Run full statistical analysis on all data columns:\n"
            "  \u2022 Summary statistics (mean, \u03c3, skewness, kurtosis)\n"
            "  \u2022 Normality testing (Shapiro-Wilk)\n"
            "  \u2022 Autocorrelation-adjusted effective sample size\n"
            "  \u2022 Distribution fitting (8 candidates, AICc + bootstrap GOF)\n"
            "  \u2022 Carry-over recommendation for the Aggregator")
        self._btn_compute.setStyleSheet(
            f"QPushButton {{ background-color: {DARK_COLORS['accent']}; "
            f"color: {DARK_COLORS['bg']}; font-weight: bold; padding: 8px 24px; }}"
            f"QPushButton:hover {{ background-color: {DARK_COLORS['accent_hover']}; }}")
        top.addWidget(self._btn_compute)
        self._cmb_var = QComboBox()
        self._cmb_var.setToolTip(
            "Switch between variables to view their individual\n"
            "statistics and distribution fitting results.")
        self._cmb_var.currentIndexChanged.connect(self._on_var_changed)
        top.addWidget(QLabel("Display:"))
        top.addWidget(self._cmb_var)
        top.addStretch()
        layout.addLayout(top)

        # Splitter: summary table (top) + distribution table (bottom)
        splitter = QSplitter(Qt.Vertical)

        # Summary statistics table
        self._summary_table = QTableWidget()
        self._summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._summary_table.setToolTip(
            "Summary statistics for the selected variable.\n"
            "DOF uses autocorrelation-adjusted N_eff \u2212 1.\n"
            "95% CI is computed via Student-t with N_eff-based DOF.")
        self._summary_table.setColumnCount(2)
        self._summary_table.setHorizontalHeaderLabels(["Statistic", "Value"])
        self._summary_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self._summary_table.setAlternatingRowColors(True)
        splitter.addWidget(self._summary_table)

        # Distribution fitting table
        grp_dist = QGroupBox("Distribution Fitting Results")
        grp_dist.setToolTip(
            "Each row is a candidate distribution ranked by AICc.\n"
            "Bootstrap GOF p > 0.05 means the fit is adequate.\n"
            "For sparse/fast mode, KS screening is marked as KS*.\n"
            "If bootstrap fails unexpectedly, KS fallback is marked KS†.\n"
            "AICc Weight shows relative likelihood. Delta AICc \u2264 2\n"
            "means the candidate is statistically competitive with the best.\n"
            "[JCGM 100:2008 \u00a74.3, JCGM 101:2008]")
        gd_lay = QVBoxLayout(grp_dist)
        self._dist_table = QTableWidget()
        self._dist_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._dist_table.setColumnCount(13)
        self._dist_table.setHorizontalHeaderLabels([
            "Rank", "Distribution", "Parameters", "AICc", "Delta AICc",
            "AICc Weight", "BIC",
            "Bootstrap GOF p-value", "GOF Pass?", "GOF Stat",
            "KS p (diag)", "AD Stat", "AD Pass?"])
        self._dist_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        self._dist_table.setAlternatingRowColors(True)
        gd_lay.addWidget(self._dist_table)
        splitter.addWidget(grp_dist)

        layout.addWidget(splitter)

        # Guidance / recommendation
        self._guidance = GuidancePanel("Recommendation")
        self._guidance.set_guidance("Click 'Compute Statistics' after entering data.", 'green')
        layout.addWidget(self._guidance)

    def compute(self, columns: List[ColumnData]):
        """Compute statistics for all columns."""
        self._results = []
        self._cmb_var.blockSignals(True)
        self._cmb_var.clear()
        for col in columns:
            vals = col.valid_values()
            if len(vals) < 2:
                continue
            result = compute_column_statistics(col)
            self._results.append(result)
            self._cmb_var.addItem(f"{col.name} ({col.unit})" if col.unit else col.name)
        self._cmb_var.blockSignals(False)
        if self._results:
            self._display_result(0)
        else:
            self._guidance.set_guidance(
                "No columns with sufficient data (N \u2265 2).", 'red')

    def _on_var_changed(self, idx):
        if 0 <= idx < len(self._results):
            self._display_result(idx)

    def _display_result(self, idx):
        r = self._results[idx]
        rows = [
            ("Variable", r.variable),
            ("Unit", r.unit or "—"),
            ("Sample Size (N)", f"{r.n} (mirrored from {r.n_raw})" if r.mirror_applied else str(r.n)),
            ("Data Mirrored?", "Yes" if r.mirror_applied else "No"),
            ("Effective Sample Size (N_eff)", f"{r.n_eff:.2f}"),
            ("Degrees of Freedom", str(r.dof)),
            ("Lag-1 Autocorrelation", f"{r.autocorr_rho1:.4f}"),
            ("Autocorrelation Time", f"{r.autocorr_tau:.3f}"),
            ("Mean", f"{r.mean:.6g}"),
            ("Median", f"{r.median:.6g}"),
            ("Std Deviation (\u03c3)", f"{r.std:.6g}"),
            ("Minimum", f"{r.min_val:.6g}"),
            ("Maximum", f"{r.max_val:.6g}"),
            ("Range", f"{r.max_val - r.min_val:.6g}"),
            ("Skewness", f"{r.skewness:.4f}"),
            ("Excess Kurtosis", f"{r.kurtosis_val:.4f}"),
            ("95% CI for Mean (low)", f"{r.ci95_low:.6g}"),
            ("95% CI for Mean (high)", f"{r.ci95_high:.6g}"),
            ("Std Uncertainty of Mean (\u03c3/\u221aN_eff)", f"{r.std_uncertainty_mean:.6g}"),
            ("Std Uncertainty (population \u03c3)", f"{r.std_uncertainty_pop:.6g}"),
            ("Shapiro-Wilk p-value", f"{r.normality_shapiro_p:.4f}"),
            ("Normally Distributed?", "Yes" if r.is_normal else "No"),
            ("Best-Fit Distribution", r.best_fit.name if r.best_fit else "—"),
            ("Best-Fit GOF p-value",
             f"{r.best_fit.gof_pvalue:.4f}" if r.best_fit else "—"),
            ("Best-Fit GOF Method", (
                "Bootstrap AD" if (r.best_fit and r.best_fit.gof_method == "bootstrap_ad")
                else "KS screening (fast mode)" if (r.best_fit and r.best_fit.gof_method == "ks_screening")
                else "KS fallback (bootstrap failed)" if (r.best_fit and r.best_fit.gof_method == "ks_fallback")
                else r.best_fit.gof_method if r.best_fit
                else "\u2014"
            )),
            ("Best-Fit GOF Pass?",
             "Yes" if (r.best_fit and r.best_fit.passed_gof) else "No"),
            ("Best-Fit Selection Note", r.best_fit_note or "—"),
            ("Candidate Set", ", ".join(r.candidate_set) if r.candidate_set else "—"),
            ("Candidate Set Note", r.candidate_set_note or "—"),
            ("Carry to Aggregator (Distribution)", r.carry_distribution),
            ("Carry to Aggregator (Method)", r.carry_method),
            ("Carry-Over Note", r.carry_note or "—"),
            ("Sparse Replicate Mode", "Yes" if r.sparse_replicate_mode else "No"),
        ]
        self._summary_table.setRowCount(len(rows))
        for i, (stat, val) in enumerate(rows):
            self._summary_table.setItem(i, 0, QTableWidgetItem(stat))
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._summary_table.setItem(i, 1, item)

        # Distribution table
        fits = r.all_fits
        self._dist_table.setRowCount(len(fits))
        for i, f in enumerate(fits):
            self._dist_table.setItem(i, 0, QTableWidgetItem(str(f.rank)))
            dist_label = f.name
            if f.beta_clipped:
                dist_label += " (clipped)"
            self._dist_table.setItem(i, 1, QTableWidgetItem(dist_label))
            self._dist_table.setItem(i, 2, QTableWidgetItem(f.param_str))
            self._dist_table.setItem(i, 3, QTableWidgetItem(f"{f.aic:.1f}"))
            self._dist_table.setItem(i, 4, QTableWidgetItem(f"{f.delta_aic:.2f}"))
            self._dist_table.setItem(i, 5, QTableWidgetItem(f"{f.akaike_weight:.3f}"))
            self._dist_table.setItem(i, 6, QTableWidgetItem(f"{f.bic:.1f}"))
            gof_txt = f"{f.gof_pvalue:.4f}"
            if f.gof_method == "ks_screening":
                gof_txt += " (KS*)"
            elif f.gof_fallback:
                gof_txt += " (KS\u2020)"
            self._dist_table.setItem(i, 7, QTableWidgetItem(gof_txt))
            gof_item = QTableWidgetItem("\u2714 Yes" if f.passed_gof else "\u2716 No")
            gof_item.setForeground(
                QColor(DARK_COLORS['green'] if f.passed_gof else DARK_COLORS['red']))
            self._dist_table.setItem(i, 8, gof_item)
            gof_stat_txt = f"{f.gof_statistic:.4f}" if np.isfinite(f.gof_statistic) else "\u2014"
            self._dist_table.setItem(i, 9, QTableWidgetItem(gof_stat_txt))
            self._dist_table.setItem(i, 10, QTableWidgetItem(f"{f.ks_pvalue:.4f}"))
            # Anderson-Darling
            if np.isfinite(f.ad_statistic):
                self._dist_table.setItem(i, 11, QTableWidgetItem(
                    f"{f.ad_statistic:.3f} (crit={f.ad_critical_5pct:.3f})"))
                ad_item = QTableWidgetItem("\u2714 Yes" if f.passed_ad else "\u2716 No")
                ad_item.setForeground(
                    QColor(DARK_COLORS['green'] if f.passed_ad else DARK_COLORS['red']))
                self._dist_table.setItem(i, 12, ad_item)
            else:
                self._dist_table.setItem(i, 11, QTableWidgetItem("\u2014"))
                self._dist_table.setItem(i, 12, QTableWidgetItem("\u2014"))

        # Recommendation — force RED if no robust fit available
        no_robust = not (r.best_fit and r.best_fit.passed_gof)
        if no_robust or r.sparse_replicate_mode:
            sev = 'red'
        elif r.std == 0.0:
            sev = 'yellow'
        elif r.n >= 30 and r.is_normal:
            sev = 'green'
        elif r.n >= 10:
            sev = 'yellow'
        else:
            sev = 'red'
        self._guidance.set_guidance(r.recommendation, sev)

    def clear(self):
        """Reset statistics tab to initial state."""
        self._results = []
        self._cmb_var.blockSignals(True)
        self._cmb_var.clear()
        self._cmb_var.blockSignals(False)
        self._summary_table.setRowCount(0)
        self._dist_table.setRowCount(0)
        self._guidance.set_guidance(
            "Click 'Compute Statistics' after entering data.", 'green')

    def get_results(self) -> List[ColumnStatistics]:
        return self._results


# =============================================================================
# TAB 3: CHARTS
# =============================================================================

class ChartsTab(QWidget):
    """Histogram, QQ plot, CDF, box plot for each variable."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: List[ColumnData] = []
        self._results: List[ColumnStatistics] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        self._cmb_var = QComboBox()
        self._cmb_var.setToolTip("Select which variable to plot.")
        self._cmb_var.currentIndexChanged.connect(self._update_charts)
        top.addWidget(QLabel("Variable:"))
        top.addWidget(self._cmb_var)

        self._cmb_chart = QComboBox()
        self._cmb_chart.setToolTip(
            "Histogram: frequency distribution with best-fit overlay.\n"
            "QQ Plot: quantile-quantile plot vs. Normal \u2014 points on\n"
            "  the diagonal mean the data matches a Normal distribution.\n"
            "CDF: empirical vs. fitted cumulative distribution function.\n"
            "Box Plot: median, IQR, and outlier visualization.")
        self._cmb_chart.addItems(["Histogram", "QQ Plot", "CDF", "Box Plot"])
        self._cmb_chart.currentIndexChanged.connect(self._update_charts)
        top.addWidget(QLabel("Chart:"))
        top.addWidget(self._cmb_chart)

        self._btn_export = QPushButton("Export Figure Package...")
        self._btn_export.setToolTip(
            "Export PNG (300+600 DPI), SVG, PDF, and JSON\n"
            "to a folder for reports and presentations.")
        self._btn_export.clicked.connect(self._export_figure)
        top.addWidget(self._btn_export)
        top.addStretch()
        layout.addLayout(top)

        # Canvas
        self._fig = Figure(figsize=(8, 5))
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

    def update_data(self, columns: List[ColumnData], results: List[ColumnStatistics]):
        self._columns = columns
        self._results = results
        self._cmb_var.blockSignals(True)
        self._cmb_var.clear()
        for r in results:
            lbl = f"{r.variable} ({r.unit})" if r.unit else r.variable
            self._cmb_var.addItem(lbl)
        self._cmb_var.blockSignals(False)
        if results:
            self._update_charts()

    def _get_current_data(self):
        idx = self._cmb_var.currentIndex()
        if idx < 0 or idx >= len(self._results):
            return None, None
        r = self._results[idx]
        # Match column by index position in the results list rather than by
        # name, which avoids ambiguity when duplicate column names exist.
        if idx < len(self._columns):
            return self._columns[idx].analysis_values(), r
        return None, None

    def _update_charts(self):
        vals, result = self._get_current_data()
        if vals is None or len(vals) < 2:
            return
        chart_type = self._cmb_chart.currentText()
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        unit_str = f" ({result.unit})" if result.unit else ""

        if chart_type == "Histogram":
            self._plot_histogram(ax, vals, result, unit_str)
        elif chart_type == "QQ Plot":
            self._plot_qq(ax, vals, result, unit_str)
        elif chart_type == "CDF":
            self._plot_cdf(ax, vals, result, unit_str)
        elif chart_type == "Box Plot":
            self._plot_box(ax, vals, result, unit_str)

        self._fig.tight_layout()
        self._canvas.draw()

    def _plot_histogram(self, ax, vals, result, unit_str):
        n_bins = min(max(int(np.sqrt(len(vals))), 5), 50)
        ax.hist(vals, bins=n_bins, density=True, alpha=0.7,
                color=DARK_COLORS['accent'], edgecolor=DARK_COLORS['border'],
                label='Data')
        # Overlay best-fit distribution
        if result.best_fit and result.best_fit.name != "Beta":
            dist_obj = dict(DISTRIBUTIONS).get(result.best_fit.name)
            if dist_obj:
                x = np.linspace(np.min(vals), np.max(vals), 200)
                pdf = dist_obj.pdf(x, *result.best_fit.params)
                ax.plot(x, pdf, color=DARK_COLORS['orange'], lw=2,
                        label=f'Best fit: {result.best_fit.name}')
        # Mean + sigma lines
        ax.axvline(result.mean, color=DARK_COLORS['green'], ls='--', lw=1.5,
                   label=f'Mean = {result.mean:.4g}')
        ax.axvline(result.mean + result.std, color=DARK_COLORS['yellow'],
                   ls=':', lw=1, label=f'\u00b1\u03c3 = {result.std:.4g}')
        ax.axvline(result.mean - result.std, color=DARK_COLORS['yellow'],
                   ls=':', lw=1)
        ax.set_xlabel(f"{result.variable}{unit_str}")
        ax.set_ylabel("Probability Density")
        ax.set_title(f"Histogram \u2014 {result.variable}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_qq(self, ax, vals, result, unit_str):
        sorted_vals = np.sort(vals)
        n = len(sorted_vals)
        theoretical_q = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        ax.scatter(theoretical_q, sorted_vals, s=12, alpha=0.7,
                   color=DARK_COLORS['accent'], edgecolors='none')
        # Reference line
        q25, q75 = np.percentile(sorted_vals, [25, 75])
        t25, t75 = norm.ppf(0.25), norm.ppf(0.75)
        if t75 - t25 > 0:
            slope = (q75 - q25) / (t75 - t25)
            intercept = q25 - slope * t25
            x_line = np.array([theoretical_q[0], theoretical_q[-1]])
            ax.plot(x_line, slope * x_line + intercept,
                    color=DARK_COLORS['red'], lw=1.5, ls='--', label='Reference line')
        ax.set_xlabel("Theoretical Quantiles (Normal)")
        ax.set_ylabel(f"Sample Quantiles{unit_str}")
        ax.set_title(f"Q-Q Plot \u2014 {result.variable}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_cdf(self, ax, vals, result, unit_str):
        sorted_vals = np.sort(vals)
        n = len(sorted_vals)
        ecdf = np.arange(1, n + 1) / n
        ax.step(sorted_vals, ecdf, where='post', color=DARK_COLORS['accent'],
                lw=1.5, label='Empirical CDF')
        # Overlay best-fit distribution CDF (fall back to Normal if unavailable)
        x = np.linspace(sorted_vals[0], sorted_vals[-1], 200)
        if result.best_fit:
            dist_obj = dict(DISTRIBUTIONS).get(result.best_fit.name)
            if dist_obj:
                ax.plot(x, dist_obj.cdf(x, *result.best_fit.params),
                        color=DARK_COLORS['orange'], lw=1.5, ls='--',
                        label=f'Best fit: {result.best_fit.name}')
            else:
                ax.plot(x, norm.cdf(x, loc=result.mean, scale=result.std),
                        color=DARK_COLORS['orange'], lw=1.5, ls='--',
                        label='Normal CDF')
        else:
            ax.plot(x, norm.cdf(x, loc=result.mean, scale=result.std),
                    color=DARK_COLORS['orange'], lw=1.5, ls='--',
                    label='Normal CDF')
        ax.set_xlabel(f"{result.variable}{unit_str}")
        ax.set_ylabel("Cumulative Probability")
        ax.set_title(f"CDF \u2014 {result.variable}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    def _plot_box(self, ax, vals, result, unit_str):
        bp = ax.boxplot(vals, vert=True, patch_artist=True,
                        boxprops=dict(facecolor=DARK_COLORS['bg_input'],
                                      edgecolor=DARK_COLORS['accent']),
                        medianprops=dict(color=DARK_COLORS['orange'], lw=2),
                        whiskerprops=dict(color=DARK_COLORS['fg_dim']),
                        capprops=dict(color=DARK_COLORS['fg_dim']),
                        flierprops=dict(markeredgecolor=DARK_COLORS['red'],
                                        marker='o', markersize=4))
        ax.set_ylabel(f"{result.variable}{unit_str}")
        ax.set_title(f"Box Plot \u2014 {result.variable}")
        ax.set_xticklabels([result.variable])
        ax.grid(True, alpha=0.3, axis='y')

    def _export_figure(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Figure Package", os.path.expanduser("~"),
            "Figure Base Name (*)")
        if not filepath:
            return
        base = os.path.splitext(filepath)[0]
        try:
            export_figure_package(self._fig, base, metadata={
                'chart_type': self._cmb_chart.currentText(),
                'variable': self._cmb_var.currentText(),
            })
            QMessageBox.information(self, "Exported",
                                    f"Figure package exported to:\n{base}_*.png/svg/pdf")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
        self._canvas.draw()

    def clear(self):
        """Reset charts tab to initial state."""
        self._columns = []
        self._results = []
        self._cmb_var.blockSignals(True)
        self._cmb_var.clear()
        self._cmb_var.blockSignals(False)
        self._fig.clear()
        self._canvas.draw()

    def get_figure(self) -> Figure:
        return self._fig


# =============================================================================
# TAB 4: CARRY-OVER SUMMARY
# =============================================================================

class CarryOverTab(QWidget):
    """Shows what to enter into the VVUQ Uncertainty Aggregator."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._override_widgets: Dict[int, Tuple[QComboBox, QComboBox, QLineEdit]] = {}
        self._carry_results: List[ColumnStatistics] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        lbl = QLabel(
            "This table shows the values to carry-over into the VVUQ "
            "Uncertainty Aggregator for each variable. Copy individual cells "
            "or the entire table to the clipboard.")
        lbl.setWordWrap(True)
        lbl.setStyleSheet(f"color: {DARK_COLORS['fg']}; font-size: 12px; margin-bottom: 8px;")
        layout.addWidget(lbl)

        self._table = QTableWidget()
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setToolTip(
            "Each row is one variable's carry-over package for the Aggregator.\n"
            "\u03c3 = population standard deviation (1-sigma).\n"
            "Override columns let you change the auto-selected\n"
            "distribution and recommended analysis mode \u2014 "
            "always document your rationale.\n"
            "Aggregator analysis mode is global (set once per run);\n"
            "if any row requires Monte Carlo, run the Aggregator in Monte Carlo mode.\n"
            "'Final' columns show what will actually be used.")
        self._table.setColumnCount(13)
        self._table.setHorizontalHeaderLabels([
            "Source Name",
            "\u03c3 (Std Uncertainty)",
            "Best Fit (This Tool)",
            "Auto Dist (Aggregator)",
            "Auto Method",
            "Override Dist",
            "Override Method",
            "Override Rationale",
            "Final Dist (Use This)",
            "Final Method (Use This)",
            "DOF",
            "Sample Size",
            "Notes",
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

        btn_row = QHBoxLayout()
        self._btn_copy = QPushButton("Copy Table to Clipboard")
        self._btn_copy.setToolTip(
            "Copy the carry-over table as tab-separated text.\n"
            "Paste directly into Excel or the VVUQ Aggregator.")
        self._btn_copy.clicked.connect(self._copy_to_clipboard)
        btn_row.addWidget(self._btn_copy)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._guidance = GuidancePanel("Carry-Over Guidance")
        self._guidance.set_guidance(
            "After computing statistics, this table is auto-populated with "
            "the values you should enter into the VVUQ Uncertainty Aggregator "
            "as uncertainty sources. Use overrides only when justified, and "
            "always provide a rationale.", 'green')
        layout.addWidget(self._guidance)

    def update_results(self, results: List[ColumnStatistics]):
        prev = self._capture_override_state()
        self._carry_results = results
        self._override_widgets = {}
        self._table.setRowCount(len(results))
        for i, r in enumerate(results):
            self._table.setItem(i, 0, QTableWidgetItem(r.variable))
            self._table.setItem(i, 1, QTableWidgetItem(f"{r.std_uncertainty_pop:.6g}"))
            best_fit_name = r.best_fit.name if r.best_fit else "\u2014"
            if r.best_fit and not r.best_fit.passed_gof:
                best_fit_name += " (FAILED GOF)"
            self._table.setItem(i, 2, QTableWidgetItem(best_fit_name))
            self._table.setItem(i, 3, QTableWidgetItem(r.carry_distribution))
            self._table.setItem(i, 4, QTableWidgetItem(r.carry_method))

            cmb_dist = QComboBox()
            cmb_dist.addItems([
                "(Auto)",
                "Normal",
                "Lognormal",
                "Uniform",
                "Triangular",
                "Weibull",
                "Student-t (df=5)",
                "Student-t (df=10)",
                "Custom/Empirical (Bootstrap)",
            ])
            cmb_dist.currentTextChanged.connect(self._refresh_final_decisions)
            self._table.setCellWidget(i, 5, cmb_dist)

            cmb_method = QComboBox()
            cmb_method.addItems([
                "(Auto)",
                "RSS",
                "Monte Carlo",
                "RSS (Conservative max)",
            ])
            cmb_method.currentTextChanged.connect(self._refresh_final_decisions)
            self._table.setCellWidget(i, 6, cmb_method)

            edit_reason = QLineEdit()
            edit_reason.setPlaceholderText("Required if override is used.")
            edit_reason.textChanged.connect(self._refresh_final_decisions)
            self._table.setCellWidget(i, 7, edit_reason)
            self._override_widgets[i] = (cmb_dist, cmb_method, edit_reason)

            # Final values get populated by _refresh_final_decisions.
            self._table.setItem(i, 8, QTableWidgetItem(r.carry_distribution))
            self._table.setItem(i, 9, QTableWidgetItem(r.carry_method))
            self._table.setItem(i, 10, QTableWidgetItem(str(r.dof)))
            n_text = f"{r.n} (mirrored)" if r.mirror_applied else str(r.n)
            self._table.setItem(i, 11, QTableWidgetItem(n_text))

            notes = []
            no_robust = not (r.best_fit and r.best_fit.passed_gof)
            if no_robust and not r.sparse_replicate_mode:
                notes.append("ALL GOF FAILED - use Custom/Empirical")
            if r.mirror_applied:
                notes.insert(0, f"MIRRORED (raw N={r.n_raw})")
            if r.n < 10:
                notes.append("Low sample size")
            if r.autocorr_warning:
                notes.append(f"Correlated data (N_eff={r.n_eff:.1f})")
            if not r.is_normal:
                notes.append("Non-normal")
            if r.candidate_set_note:
                notes.append("Multiple plausible fits")
            if r.best_fit_note:
                notes.append(r.best_fit_note)
            if r.carry_note:
                notes.append(r.carry_note)
            if r.sparse_replicate_mode:
                notes.append("Sparse replicate mode")
            note_item = QTableWidgetItem("; ".join(notes) if notes else "OK")
            note_item.setData(Qt.UserRole, note_item.text())
            self._table.setItem(i, 12, note_item)

            if r.variable in prev:
                prev_dist, prev_method, prev_reason = prev[r.variable]
                if prev_dist in [cmb_dist.itemText(j) for j in range(cmb_dist.count())]:
                    cmb_dist.setCurrentText(prev_dist)
                if prev_method in [cmb_method.itemText(j) for j in range(cmb_method.count())]:
                    cmb_method.setCurrentText(prev_method)
                edit_reason.setText(prev_reason)

        self._refresh_final_decisions()

    def _capture_override_state(self) -> Dict[str, Tuple[str, str, str]]:
        """Capture override state keyed by variable name (not row index)."""
        state: Dict[str, Tuple[str, str, str]] = {}
        for row, widgets in self._override_widgets.items():
            name_item = self._table.item(row, 0)
            if name_item is None:
                continue
            var_name = name_item.text().strip()
            if not var_name:
                continue
            dist_cmb, method_cmb, reason_edit = widgets
            state[var_name] = (
                dist_cmb.currentText().strip(),
                method_cmb.currentText().strip(),
                reason_edit.text().strip(),
            )
        return state

    def _refresh_final_decisions(self):
        override_count = 0
        missing_reason = 0
        for row in range(self._table.rowCount()):
            auto_dist_item = self._table.item(row, 3)
            auto_method_item = self._table.item(row, 4)
            auto_dist = auto_dist_item.text() if auto_dist_item else "Normal"
            auto_method = auto_method_item.text() if auto_method_item else "RSS"
            widgets = self._override_widgets.get(row)
            if not widgets:
                continue
            dist_cmb, method_cmb, reason_edit = widgets
            dist_override = dist_cmb.currentText().strip()
            method_override = method_cmb.currentText().strip()
            rationale = reason_edit.text().strip()

            use_dist_override = dist_override and dist_override != "(Auto)"
            use_method_override = method_override and method_override != "(Auto)"
            is_override = use_dist_override or use_method_override
            if is_override:
                override_count += 1

            final_dist = dist_override if use_dist_override else auto_dist
            final_method = method_override if use_method_override else auto_method
            self._table.setItem(row, 8, QTableWidgetItem(final_dist))
            self._table.setItem(row, 9, QTableWidgetItem(final_method))

            note_item = self._table.item(row, 12)
            base_note = ""
            if note_item is not None:
                base_note = str(note_item.data(Qt.UserRole) or "")
            notes = []
            if base_note and base_note != "OK":
                notes.append(base_note)

            if is_override:
                if rationale:
                    notes.append("Override applied with rationale.")
                    reason_edit.setStyleSheet("")
                else:
                    notes.append("Override selected but rationale is missing.")
                    reason_edit.setStyleSheet(
                        f"QLineEdit {{ border: 1px solid {DARK_COLORS['red']}; }}"
                    )
                    missing_reason += 1
            else:
                reason_edit.setStyleSheet("")

            final_note = "; ".join(notes) if notes else "OK"
            if note_item is None:
                note_item = QTableWidgetItem(final_note)
                self._table.setItem(row, 12, note_item)
            else:
                note_item.setText(final_note)

        # Check for GOF failures across all results
        gof_fail_count = 0
        results = getattr(self, '_carry_results', [])
        for r in results:
            no_robust = not (r.best_fit and r.best_fit.passed_gof)
            if no_robust or r.sparse_replicate_mode:
                gof_fail_count += 1

        if missing_reason > 0:
            self._guidance.set_guidance(
                f"{missing_reason} override row(s) are missing rationale. "
                "Add rationale before finalizing the carry-over export.",
                'red',
            )
        elif gof_fail_count > 0:
            self._guidance.set_guidance(
                f"\u26a0 {gof_fail_count} variable(s) have no robust parametric fit "
                "(all GOF tests failed or sparse replicate mode). Use "
                "'Custom/Empirical (Bootstrap)' with 'Monte Carlo' for those "
                "variables in the Aggregator.",
                'red',
            )
        elif override_count > 0:
            self._guidance.set_guidance(
                f"{override_count} override row(s) active. Final Dist/Method "
                "columns are the values to carry into the Aggregator.",
                'yellow',
            )
        else:
            self._guidance.set_guidance(
                "No overrides active. Final Dist/Method columns are identical "
                "to the auto recommendation and are ready for the Aggregator.",
                'green',
            )

    def get_effective_results(self, results: List[ColumnStatistics]) -> List[ColumnStatistics]:
        """Return result objects with carry-over overrides applied."""
        effective: List[ColumnStatistics] = []
        for i, r in enumerate(results):
            rr = copy.deepcopy(r)
            widgets = self._override_widgets.get(i)
            rationale = ""
            if widgets:
                dist_cmb, method_cmb, reason_edit = widgets
                dist_override = dist_cmb.currentText().strip()
                method_override = method_cmb.currentText().strip()
                rationale = reason_edit.text().strip()
                if dist_override and dist_override != "(Auto)":
                    rr.carry_distribution = dist_override
                if method_override and method_override != "(Auto)":
                    rr.carry_method = method_override
            if rationale:
                base = rr.carry_note.strip()
                rr.carry_note = (
                    f"{base} Override rationale: {rationale}".strip()
                    if base else f"Override rationale: {rationale}"
                )
            effective.append(rr)
        return effective

    def _copy_to_clipboard(self):
        rows = []
        headers = []
        for c in range(self._table.columnCount()):
            headers.append(self._table.horizontalHeaderItem(c).text())
        rows.append("\t".join(headers))
        for r in range(self._table.rowCount()):
            row = []
            for c in range(self._table.columnCount()):
                widget = self._table.cellWidget(r, c)
                if isinstance(widget, QComboBox):
                    row.append(widget.currentText())
                elif isinstance(widget, QLineEdit):
                    row.append(widget.text())
                else:
                    item = self._table.item(r, c)
                    row.append(item.text() if item else "")
            rows.append("\t".join(row))
        text = "\n".join(rows)
        try:
            QApplication.clipboard().setText(text)
            self._guidance.set_guidance("Table copied to clipboard!", 'green')
        except Exception as e:
            QMessageBox.warning(self, "Clipboard Error",
                                f"Could not copy to clipboard:\n{e}")

    def clear(self):
        """Reset carry-over tab to initial state."""
        self._carry_results = []
        self._override_widgets = {}
        self._table.setRowCount(0)
        self._guidance.set_guidance(
            "After computing statistics, this table is auto-populated with "
            "the values you should enter into the VVUQ Uncertainty Aggregator "
            "as uncertainty sources. Use overrides only when justified, and "
            "always provide a rationale.", 'green')


# =============================================================================
# TAB 5: REFERENCE
# =============================================================================

class ReferenceTab(QWidget):
    """Distribution guide, sample size guidance, GUM references."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        tabs = QTabWidget()
        tabs.addTab(self._build_dist_guide(), "Distribution Guide")
        tabs.addTab(self._build_sample_guide(), "Sample Size Guidance")
        tabs.addTab(self._build_gum_ref(), "GUM References")
        tabs.addTab(self._build_glossary(), "Glossary")
        layout.addWidget(tabs)

    def _make_text(self, html):
        te = QTextEdit()
        te.setReadOnly(True)
        te.setHtml(html)
        return te

    def _build_dist_guide(self):
        c = DARK_COLORS
        html = f"""<html><body style="background-color:{c['bg']};color:{c['fg']};
                    font-family:Segoe UI,sans-serif;font-size:12px;padding:12px;">
        <h2 style="color:{c['accent']};">Distribution Selection Guide</h2>
        <table border="1" cellpadding="6" cellspacing="0"
               style="border-color:{c['border']};width:100%;">
        <tr style="background-color:{c['bg_alt']};">
            <th>Distribution</th><th>When to Use</th><th>Key Parameters</th>
        </tr>
        <tr><td><b>Normal</b></td>
            <td>Default for most measurement uncertainties, instrument errors,
                repeated readings. Symmetric bell curve.</td>
            <td>\u03bc (mean), \u03c3 (std dev)</td></tr>
        <tr><td><b>Log-Normal</b></td>
            <td>Positive-only data with right skew. Common for flow rates,
                concentrations, material properties.</td>
            <td>\u03bc, \u03c3 (of log-transformed data)</td></tr>
        <tr><td><b>Uniform</b></td>
            <td>All values equally likely within bounds. Digitization error,
                rounding, manufacturer tolerance bands.</td>
            <td>a (lower), b (upper)</td></tr>
        <tr><td><b>Triangular</b></td>
            <td>Best estimate with bounds. When you know min, max, and most
                likely value but have limited data.</td>
            <td>a (min), c (mode), b (max)</td></tr>
        <tr><td><b>Weibull</b></td>
            <td>Failure/lifetime data, wind speeds, material strength.
                Flexible shape for positive data.</td>
            <td>k (shape), \u03bb (scale)</td></tr>
        <tr><td><b>Gamma</b></td>
            <td>Wait times, rainfall, positive skewed data.
                Sum of exponential processes.</td>
            <td>\u03b1 (shape), \u03b2 (rate)</td></tr>
        <tr><td><b>Student-t</b></td>
            <td>Small samples (N < 30) where population \u03c3 is unknown.
                Heavier tails than Normal — more conservative.</td>
            <td>\u03bd (degrees of freedom)</td></tr>
        <tr><td><b>Beta</b></td>
            <td>Data bounded on [0,1]. Proportions, probabilities,
                percentages.</td>
            <td>\u03b1, \u03b2 (shape parameters)</td></tr>
        </table>
        </body></html>"""
        return self._make_text(html)

    def _build_sample_guide(self):
        c = DARK_COLORS
        html = f"""<html><body style="background-color:{c['bg']};color:{c['fg']};
                    font-family:Segoe UI,sans-serif;font-size:12px;padding:12px;">
        <h2 style="color:{c['accent']};">Sample Size Guidance</h2>
        <table border="1" cellpadding="6" cellspacing="0"
               style="border-color:{c['border']};width:100%;">
        <tr style="background-color:{c['bg_alt']};">
            <th>Sample Size</th><th>Assessment</th><th>Action</th>
        </tr>
        <tr><td style="color:{c['red']};">N &lt; 10</td>
            <td>Insufficient per GUM \u00a7G.3</td>
            <td>Collect more samples. Use Student-t with low DOF.
                Results have high uncertainty.</td></tr>
        <tr><td style="color:{c['yellow']};">10 \u2264 N &lt; 30</td>
            <td>Marginal</td>
            <td>Usable but distribution fitting has limited reliability.
                Prefer Student-t over Normal.</td></tr>
        <tr><td style="color:{c['green']};">N \u2265 30</td>
            <td>Adequate</td>
            <td>Standard analysis applicable. Distribution fitting
                reliable. CLT applies for mean estimation.</td></tr>
        <tr><td style="color:{c['green']};">N \u2265 100</td>
            <td>Good</td>
            <td>High confidence in distribution shape. Anderson-Darling
                and bootstrap GOF checks are more stable.</td></tr>
        </table>
        <h3 style="color:{c['accent']};">Key Formulas</h3>
        <ul>
        <li><b>Standard uncertainty of the mean:</b> u = \u03c3 / \u221aN</li>
        <li><b>Degrees of freedom:</b> \u03bd = N<sub>eff</sub> \u2212 1 (N<sub>eff</sub> accounts for autocorrelation)</li>
        <li><b>95% CI for the mean:</b> x\u0304 \u00b1 t<sub>0.975,\u03bd</sub> \u00b7 u</li>
        <li><b>Coverage factor (95%):</b> k \u2248 t<sub>0.975,\u03bd</sub>
            (\u2248 2.0 for large N, higher for small N)</li>
        </ul>
        </body></html>"""
        return self._make_text(html)

    def _build_gum_ref(self):
        c = DARK_COLORS
        html = f"""<html><body style="background-color:{c['bg']};color:{c['fg']};
                    font-family:Segoe UI,sans-serif;font-size:12px;padding:12px;">
        <h2 style="color:{c['accent']};">GUM & Standards References</h2>
        <h3>JCGM 100:2008 (GUM)</h3>
        <ul>
        <li>\u00a74.2 — Type A evaluation: statistical analysis of repeated observations</li>
        <li>\u00a74.3 — Type B evaluation: other means (manufacturer specs, handbooks)</li>
        <li>\u00a7G.3 — Guidance on sample size: N \u2265 10 recommended</li>
        <li>\u00a7G.4 — Degrees of freedom and Welch-Satterthwaite</li>
        </ul>
        <h3>JCGM 101:2008 (GUM Supplement 1)</h3>
        <ul>
        <li>Monte Carlo propagation of distributions</li>
        <li>Adaptive MC for convergence</li>
        </ul>
        <h3>ASME PTC 19.1-2018</h3>
        <ul>
        <li>Test uncertainty analysis procedures</li>
        <li>Systematic and random components</li>
        </ul>
        <h3>ASME V&V 20-2009 (R2021)</h3>
        <ul>
        <li>\u00a75 — Numerical uncertainty (u<sub>num</sub>)</li>
        <li>\u00a76 — Input uncertainty (u<sub>input</sub>)</li>
        <li>\u00a77 — Experimental uncertainty (u<sub>D</sub>)</li>
        <li>\u00a78 — Validation comparison</li>
        </ul>
        </body></html>"""
        return self._make_text(html)

    def _build_glossary(self):
        c = DARK_COLORS
        html = f"""<html><body style="background-color:{c['bg']};color:{c['fg']};
                    font-family:Segoe UI,sans-serif;font-size:12px;padding:12px;">
        <h2 style="color:{c['accent']};">Glossary</h2>
        <table border="1" cellpadding="6" cellspacing="0"
               style="border-color:{c['border']};width:100%;">
        <tr style="background-color:{c['bg_alt']};"><th>Term</th><th>Definition</th></tr>
        <tr><td>\u03c3</td><td>Standard deviation (sample, with Bessel correction N\u22121)</td></tr>
        <tr><td>DOF (\u03bd)</td><td>Degrees of freedom = N<sub>eff</sub> \u2212 1 (where N<sub>eff</sub> accounts for autocorrelation)</td></tr>
        <tr><td>AICc</td><td>Corrected Akaike Information Criterion &mdash; a score measuring how well a distribution fits the data; lower is better. The correction prevents overfitting with small samples.</td></tr>
        <tr><td>BIC</td><td>Bayesian Information Criterion — penalizes complexity more</td></tr>
        <tr><td>Bootstrap GOF</td><td>Parametric bootstrap goodness-of-fit p-value</td></tr>
        <tr><td>KS test</td><td>Kolmogorov-Smirnov diagnostic (not primary pass/fail)</td></tr>
        <tr><td>Shapiro-Wilk</td><td>Normality test (p &gt; 0.05 \u2192 normal)</td></tr>
        <tr><td>Skewness</td><td>Measure of asymmetry (0 = symmetric)</td></tr>
        <tr><td>Kurtosis</td><td>Measure of tail weight (0 = Normal-like)</td></tr>
        <tr><td>u<sub>mean</sub></td><td>Standard uncertainty of the mean = \u03c3/\u221aN<sub>eff</sub>, where N<sub>eff</sub> is the autocorrelation-adjusted effective sample size</td></tr>
        <tr><td>Coverage factor k</td><td>Multiplier for expanded uncertainty U = k \u00b7 u</td></tr>
        </table>
        </body></html>"""
        return self._make_text(html)


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html_report(results: List[ColumnStatistics], project_meta: dict,
                         chart_fig: Optional[Figure] = None) -> str:
    """Generate a full HTML report for statistical carry-over decisions."""
    from html import escape as _esc_html

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    consequence = normalize_decision_consequence(
        project_meta.get('decision_consequence', 'Medium')
    )

    def _meta_truthy(key: str) -> bool:
        raw = project_meta.get(key, False)
        if isinstance(raw, bool):
            return raw
        txt = str(raw).strip().lower()
        return txt in ("1", "true", "yes", "y", "checked")

    # Embed chart if available
    chart_html = ""
    if chart_fig is not None:
        try:
            b64 = _figure_to_base64(chart_fig)
            chart_html = f'<img src="data:image/png;base64,{b64}" style="max-width:100%;"/>'
        except Exception:
            chart_html = "<p><i>Chart embedding failed.</i></p>"

    rows_html = ""
    for r in results:
        var_name = _esc_html(r.variable)
        unit_name = _esc_html(r.unit) if r.unit else "—"
        dist_name = _esc_html(r.best_fit.name) if r.best_fit else "—"
        gof_p = f"{r.best_fit.gof_pvalue:.4f}" if r.best_fit else "—"
        norm_str = "Yes" if r.is_normal else "No"
        n_display = f"{r.n} (mirrored from {r.n_raw})" if r.mirror_applied else str(r.n)
        rows_html += f"""
        <tr>
            <td>{var_name}</td><td>{unit_name}</td><td>{n_display}</td>
            <td>{r.n_eff:.2f}</td>
            <td>{r.mean:.6g}</td><td>{r.median:.6g}</td><td>{r.std:.6g}</td>
            <td>{r.min_val:.6g}</td><td>{r.max_val:.6g}</td>
            <td>{r.skewness:.3f}</td><td>{r.kurtosis_val:.3f}</td>
            <td>{r.autocorr_rho1:.3f}</td><td>{norm_str}</td>
            <td>{dist_name}</td><td>{gof_p}</td>
            <td>{r.std_uncertainty_pop:.6g}</td><td>{r.dof}</td>
        </tr>"""

    carry_rows = ""
    for r in results:
        var_name = _esc_html(r.variable)
        dist_name = _esc_html(r.best_fit.name) if r.best_fit else "Normal"
        notes = []
        if r.n < 10:
            notes.append("Low N")
        if r.autocorr_warning:
            notes.append(f"Correlated data (N_eff={r.n_eff:.1f})")
        if not r.is_normal:
            notes.append("Non-normal")
        if r.candidate_set_note:
            notes.append("Model mixture recommended")
        if r.carry_note:
            notes.append(r.carry_note)
        notes_text = _esc_html('; '.join(notes) if notes else 'OK')
        carry_rows += f"""
        <tr>
            <td>{var_name}</td><td>{r.std_uncertainty_pop:.6g}</td>
            <td>{dist_name}</td><td>{_esc_html(r.carry_distribution)}</td>
            <td>{_esc_html(r.carry_method)}</td><td>{r.dof}</td><td>{r.n}</td>
            <td>{notes_text}</td>
        </tr>"""

    rec_html = ""
    for r in results:
        # Use dark, high-contrast colors readable on the white report background
        color = '#2d8a4e' if r.n >= 30 and r.is_normal else (
            '#b8860b' if r.n >= 10 else '#c0392b')
        rec_html += (
            f'<p style="color:{color};"><b>{_esc_html(r.variable)}:</b> '
            f'{_esc_html(r.recommendation)}</p>'
        )

    program_raw = str(project_meta.get('program', '—'))
    analyst_raw = str(project_meta.get('analyst', '—'))
    date_raw = str(project_meta.get('date', '—'))
    notes_raw = str(project_meta.get('notes', '—'))
    program = _esc_html(program_raw)
    analyst = _esc_html(analyst_raw)
    date = _esc_html(date_raw)
    notes = _esc_html(notes_raw)

    units = {r.unit for r in results if r.unit}
    diagnostics_ok = all(
        (not r.autocorr_warning) and (r.best_fit is None or r.best_fit.passed_gof or r.is_normal)
        for r in results
    )
    evidence = {
        'inputs_documented': bool(program_raw.strip()) and bool(analyst_raw.strip()),
        'method_selected': True,
        'units_consistent': len(units) <= 1,
        'data_quality': all(r.n >= 10 for r in results),
        'diagnostics_pass': diagnostics_ok,
        'independent_review': _meta_truthy('independent_review'),
        'conservative_bound': any(r.carry_method == "Monte Carlo" for r in results),
        'validation_plan': _meta_truthy('validation_plan'),
    }

    primary = max(results, key=lambda x: x.std_uncertainty_pop) if results else None
    if primary is None:
        decision_card_html = render_decision_card_html(
            title="Decision Card",
            use_value="No result available",
            use_distribution="No result available",
            use_combination="No result available",
            stop_checks=["Run analysis first"],
            notes="No computed rows are available.",
        )
    else:
        decision_card_html = render_decision_card_html(
            title="Decision Card (Carry to Aggregator)",
            use_value=(
                "Use per-variable sigma from Section 7. "
                f"Largest current carry value: {primary.variable} = "
                f"{primary.std_uncertainty_pop:.6g}."
            ),
            use_distribution=(
                "Use per-variable recommended distribution in Section 7 "
                f"(dominant variable default: {primary.carry_distribution})."
            ),
            use_combination=(
                "Aggregator mode is analysis-level: if any Section 7 row "
                "shows Monte Carlo, run the Aggregator in Monte Carlo mode; "
                f"otherwise use RSS (current dominant row: {primary.carry_method})."
            ),
            stop_checks=[
                "Any row with N < 10",
                "Any row with correlated-data warning",
                "Any row where diagnostics indicate non-robust fit",
                "Any override without written rationale",
            ],
            notes="This card is guidance; Section 7 values remain the authoritative transfer table.",
        )

    credibility_html = render_credibility_html(consequence, evidence)
    glossary_html = render_vvuq_glossary_html()
    conformity_html = render_conformity_template_html(
        metric_name="Carry-over uncertainty per source",
        metric_value="See Section 7 table",
        consequence=consequence,
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Statistical Analysis Report</title>
<style>
    body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #fff; color: #1a1a2e; }}
    h1 {{ color: #1a1a2e; border-bottom: 2px solid #333; padding-bottom: 8px; }}
    h2 {{ color: #2a2a5e; margin-top: 24px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0; font-size: 11px; }}
    th {{ background-color: #e8e8f0; padding: 6px 8px; border: 1px solid #ccc; text-align: left; }}
    td {{ padding: 5px 8px; border: 1px solid #ccc; }}
    tr:nth-child(even) {{ background-color: #f5f5fa; }}
    .meta-table td {{ border: none; padding: 3px 12px 3px 0; }}
    .section {{ margin-bottom: 20px; }}
    .verdict {{
        padding: 10px 12px; border-radius: 4px; margin: 10px 0;
        border-left: 5px solid #6c757d; background: #eef1f5; color: #1a1a2e;
    }}
    .verdict.pass {{ border-left-color: #2e7d32; background: #eaf6ec; }}
    .verdict.fail {{ border-left-color: #b71c1c; background: #fdeaea; }}
    .highlight {{
        background: #fff3cd; border-left: 5px solid #ffc107;
        padding: 10px 12px; margin: 10px 0;
    }}
    .findings-block {{
        background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 4px; padding: 10px 12px;
    }}
    .checklist {{ list-style: none; padding: 0; }}
    .checklist li::before {{ content: '\u2610 '; }}
    @media print {{
        body {{ margin: 12mm; }}
        h2 {{ page-break-after: avoid; }}
        table {{ page-break-inside: avoid; }}
    }}
</style>
</head><body>

<h1>Statistical Analysis Report</h1>

<!-- Section 1: Project Info -->
<div class="section">
<h2>1. Project Information</h2>
<table class="meta-table">
<tr><td><b>Program / Project:</b></td><td>{program}</td></tr>
<tr><td><b>Analyst:</b></td><td>{analyst}</td></tr>
<tr><td><b>Date:</b></td><td>{date}</td></tr>
<tr><td><b>Generated:</b></td><td>{now}</td></tr>
<tr><td><b>Tool:</b></td><td>{APP_NAME} v{APP_VERSION}</td></tr>
<tr><td><b>Decision Consequence:</b></td><td>{consequence}</td></tr>
<tr><td><b>Notes:</b></td><td>{notes}</td></tr>
</table>
</div>

<!-- Section 2: Executive Summary -->
<div class="section">
<h2>2. Executive Summary</h2>
<p>This report presents statistical analysis of {len(results)} variable(s).
Distribution fitting was performed using 8 candidate distributions with
AICc-based ranking and bootstrap goodness-of-fit p-values.</p>
</div>

<!-- Section 3: Data Summary -->
<div class="section">
<h2>3. Summary Statistics</h2>
<table>
<tr><th>Variable</th><th>Unit</th><th>N</th><th>N_eff</th><th>Mean</th><th>Median</th>
<th>\u03c3</th><th>Min</th><th>Max</th><th>Skew</th><th>Kurt</th>
<th>Lag-1 \u03c1</th>
<th>Normal?</th><th>Best Fit</th><th>GOF p</th><th>u (\u03c3)</th><th>DOF</th></tr>
{rows_html}
</table>
</div>

<!-- Section 4: Charts -->
<div class="section">
<h2>4. Charts</h2>
{chart_html if chart_html else '<p><i>No charts embedded.</i></p>'}
</div>

<!-- Section 5: Distribution Fitting -->
<div class="section">
<h2>5. Distribution Fitting Details</h2>
<p>Distributions fitted via Maximum Likelihood Estimation (MLE).
Ranked by corrected Akaike Information Criterion (AICc). Final pass/fail decisions
use parametric-bootstrap goodness-of-fit p-values at \u03b1 = 0.05.</p>
</div>

<!-- Section 6: Recommendations -->
<div class="section">
<h2>6. Recommendations</h2>
{rec_html}
</div>

<!-- Section 7: Carry-Over to VVUQ Aggregator -->
<div class="section">
<h2>7. Carry-Over Summary (for VVUQ Aggregator)</h2>
<table>
<tr><th>Source Name</th><th>\u03c3 (Std Uncertainty)</th><th>Best Fit (Tool)</th>
<th>Distribution for Aggregator</th><th>Recommended Aggregator Analysis Mode</th><th>DOF</th><th>N</th><th>Notes</th></tr>
{carry_rows}
</table>
</div>

<!-- Section 8: Methodology -->
<div class="section">
<h2>8. Methodology</h2>
<ul>
<li>Type A evaluation per JCGM 100:2008 \u00a74.2</li>
<li>Distribution fitting: MLE with 8 candidate distributions</li>
<li>Goodness-of-fit: AICc, BIC, bootstrap GOF p-value (\u03b1=0.05)</li>
<li>KS p-values are retained only as diagnostics</li>
<li>Normality: Shapiro-Wilk test (\u03b1=0.05)</li>
<li>Standard uncertainty: population \u03c3 (with Bessel correction)</li>
<li>Autocorrelation adjustment: N_eff used for uncertainty-of-mean and DOF guidance</li>
<li>Candidate-set carry mode: delta AICc <= 2 triggers model-mixture recommendation</li>
<li>Sparse replicate mode: N <= 8 triggers empirical/discrete carry mode</li>
</ul>
</div>

<!-- Section 9: Assumptions & Limitations -->
<div class="section">
<h2>9. Assumptions & Limitations</h2>
<ul>
<li>Rows are assumed to be from a consistent process for each variable</li>
<li>Distribution fitting assumes sufficient sample size for MLE convergence</li>
<li>Beta fitting is only attempted for true bounded proportion data in [0,1]</li>
<li>Autocorrelation correction uses a practical finite-lag estimate</li>
</ul>
</div>

{decision_card_html}

{credibility_html}

{glossary_html}

{conformity_html}

<!-- Reviewer Checklist -->
<div class="section">
<h2>Reviewer Checklist</h2>
<ul class="checklist">
<li>Sample sizes adequate for intended analysis?</li>
<li>Autocorrelation reviewed (N_eff compared to N)?</li>
<li>Data source and collection method documented?</li>
<li>Distribution selection justified?</li>
<li>Outliers investigated?</li>
<li>Carry-over values entered into VVUQ Aggregator?</li>
<li>Units consistent with other uncertainty sources?</li>
</ul>
</div>

</body></html>"""
    return html


# =============================================================================
# MAIN WINDOW
# =============================================================================

class StatisticalAnalyzerWindow(QMainWindow):
    """Main application window for the Statistical Analyzer."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1100, 750)
        self.setStyleSheet(get_dark_stylesheet())

        # Application font
        font = QFont()
        for family in FONT_FAMILIES:
            if QFontDatabase.hasFamily(family):
                font.setFamily(family)
                break
        font.setPointSize(10)
        QApplication.instance().setFont(font)

        # Project state
        self._project_path: str = ""
        self._project_name: str = "Untitled"
        self._unsaved_changes: bool = False

        # Central widget
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.setCentralWidget(central)

        # Project info bar (collapsed by default)
        self._project_bar = QFrame()
        self._project_bar.setStyleSheet(
            f"QFrame {{ background-color: {DARK_COLORS['bg_alt']}; "
            f"border-bottom: 1px solid {DARK_COLORS['border']}; }}")
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
            f"QPushButton:hover {{ color: {DARK_COLORS['accent_hover']}; }}")
        self._btn_toggle_info.setCursor(Qt.PointingHandCursor)
        self._btn_toggle_info.clicked.connect(self._toggle_project_info)
        bar_top.addWidget(self._btn_toggle_info)
        bar_top.addStretch()
        self._lbl_project_name = QLabel("Untitled")
        self._lbl_project_name.setStyleSheet(
            f"color: {DARK_COLORS['fg']}; font-size: 12px; font-weight: bold;")
        bar_top.addWidget(self._lbl_project_name)
        bar_layout.addLayout(bar_top)

        # Collapsible detail area
        self._project_detail_frame = QFrame()
        detail_layout = QGridLayout(self._project_detail_frame)
        detail_layout.setContentsMargins(4, 4, 4, 4)
        detail_layout.setSpacing(6)
        lbl_style = f"color: {DARK_COLORS['fg_dim']}; font-size: 11px;"
        val_style = (
            f"background-color: {DARK_COLORS['bg_input']}; "
            f"color: {DARK_COLORS['fg']}; border: 1px solid {DARK_COLORS['border']}; "
            f"border-radius: 3px; padding: 3px 6px; font-size: 11px;")

        detail_layout.addWidget(self._make_label("Program / Project:", lbl_style), 0, 0)
        self._edit_program = QLineEdit()
        self._edit_program.setStyleSheet(val_style)
        self._edit_program.setPlaceholderText("e.g., XYZ Flight Test Campaign")
        self._edit_program.setToolTip("Name of the program, project, or test campaign.")
        self._edit_program.textChanged.connect(self._mark_unsaved)
        detail_layout.addWidget(self._edit_program, 0, 1)

        detail_layout.addWidget(self._make_label("Analyst:", lbl_style), 0, 2)
        self._edit_analyst = QLineEdit()
        self._edit_analyst.setStyleSheet(val_style)
        self._edit_analyst.setPlaceholderText("e.g., J. Smith")
        self._edit_analyst.setToolTip("Name of the analyst performing this assessment.")
        self._edit_analyst.textChanged.connect(self._mark_unsaved)
        detail_layout.addWidget(self._edit_analyst, 0, 3)

        detail_layout.addWidget(self._make_label("Date:", lbl_style), 0, 4)
        self._edit_date = QLineEdit()
        self._edit_date.setStyleSheet(val_style)
        self._edit_date.setText(datetime.datetime.now().strftime("%Y-%m-%d"))
        self._edit_date.setToolTip("Date of this analysis (auto-filled, editable).")
        self._edit_date.textChanged.connect(self._mark_unsaved)
        detail_layout.addWidget(self._edit_date, 0, 5)

        detail_layout.addWidget(self._make_label("Notes:", lbl_style), 1, 0)
        self._edit_notes = QTextEdit()
        self._edit_notes.setStyleSheet(f"QTextEdit {{ {val_style} }}")
        self._edit_notes.setPlaceholderText(
            "Free-form notes: describe the data source, test conditions, etc.")
        self._edit_notes.setMaximumHeight(80)
        self._edit_notes.textChanged.connect(self._mark_unsaved)
        detail_layout.addWidget(self._edit_notes, 1, 1, 1, 5)

        detail_layout.addWidget(self._make_label("Decision Consequence:", lbl_style), 2, 0)
        self._cmb_consequence = QComboBox()
        self._cmb_consequence.setToolTip(
            "Decision consequence level per ASME V&V 20 \u00a76:\n"
            "  \u2022 Low \u2014 minimal safety or cost impact\n"
            "  \u2022 Medium \u2014 moderate impact (default)\n"
            "  \u2022 High \u2014 safety-critical or high-cost decisions\n"
            "Higher consequence requires stricter evidence thresholds.")
        self._cmb_consequence.addItems(["Low", "Medium", "High"])
        self._cmb_consequence.setCurrentText("Medium")
        self._cmb_consequence.setStyleSheet(val_style)
        self._cmb_consequence.currentTextChanged.connect(self._mark_unsaved)
        detail_layout.addWidget(self._cmb_consequence, 2, 1)

        detail_layout.setColumnStretch(1, 3)
        detail_layout.setColumnStretch(3, 2)
        detail_layout.setColumnStretch(5, 2)

        bar_layout.addWidget(self._project_detail_frame)
        self._project_detail_frame.setVisible(False)
        self._project_info_visible = False
        central_layout.addWidget(self._project_bar)

        # Tabs
        self._tabs = QTabWidget()
        central_layout.addWidget(self._tabs)

        self._tab_data = DataInputTab()
        self._tab_stats = StatisticsTab()
        self._tab_charts = ChartsTab()
        self._tab_carry = CarryOverTab()
        self._tab_ref = ReferenceTab()

        # Let DataInputTab know when analysis results exist so it can
        # show a stale-analysis warning after data edits
        self._tab_data._results_exist_fn = lambda: bool(self._tab_stats.get_results())

        self._tabs.addTab(self._tab_data, "\U0001f4e5 Data Input")
        self._tabs.addTab(self._tab_stats, "\U0001f4ca Statistics")
        self._tabs.addTab(self._tab_charts, "\U0001f4c8 Charts")
        self._tabs.addTab(self._tab_carry, "\U0001f4cb Carry-Over Summary")
        self._tabs.addTab(self._tab_ref, "\U0001f4d6 Reference")

        # Wire compute button
        self._tab_stats._btn_compute.clicked.connect(self._run_analysis)

        # Menu bar
        self._build_menu_bar()

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        version_label = QLabel(f"{APP_NAME} v{APP_VERSION} ({APP_DATE})")
        version_label.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; font-size: 11px; padding-right: 8px;")
        self._status_bar.addPermanentWidget(version_label)
        self._status_bar.showMessage("Ready.", 5000)

    # ---- helpers ----

    @staticmethod
    def _make_label(text, style):
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        return lbl

    def _toggle_project_info(self):
        self._project_info_visible = not self._project_info_visible
        self._project_detail_frame.setVisible(self._project_info_visible)
        arrow = "\u25bc" if self._project_info_visible else "\u25b6"
        self._btn_toggle_info.setText(f"{arrow} Project Info")

    def _mark_unsaved(self):
        self._unsaved_changes = True

    def _check_unsaved_changes(self) -> bool:
        """Check for unsaved changes; return True to proceed, False to cancel."""
        if not self._unsaved_changes:
            return True
        reply = QMessageBox.question(
            self, "Unsaved Changes",
            "You have unsaved changes. Do you want to save before continuing?",
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
            QMessageBox.Save,
        )
        if reply == QMessageBox.Save:
            self._save_project()
            return True
        elif reply == QMessageBox.Discard:
            return True
        else:  # Cancel
            return False

    def closeEvent(self, event):
        """Prompt to save before closing the application."""
        if self._check_unsaved_changes():
            event.accept()
        else:
            event.ignore()

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

    # ---- analysis ----

    def _run_analysis(self):
        columns = self._tab_data.get_columns()
        valid_cols = [c for c in columns if len(c.valid_values()) >= 2]
        if not valid_cols:
            QMessageBox.warning(self, "No Data",
                                "No columns with at least 2 valid values.")
            return
        self._tab_stats.compute(valid_cols)
        results = self._tab_stats.get_results()
        self._tab_charts.update_data(valid_cols, results)
        self._tab_carry.update_results(results)
        self._tabs.setCurrentWidget(self._tab_stats)
        self._status_bar.showMessage(
            f"Analysis complete: {len(results)} variable(s) processed.", 5000)

    # ---- menu ----

    def _build_menu_bar(self):
        mb = self.menuBar()
        file_menu = mb.addMenu("File")

        act_new = QAction("New Project", self)
        act_new.setShortcut("Ctrl+N")
        act_new.triggered.connect(self._new_project)
        file_menu.addAction(act_new)

        act_open = QAction("Open Project...", self)
        act_open.setShortcut("Ctrl+O")
        act_open.triggered.connect(self._open_project)
        file_menu.addAction(act_open)

        act_save = QAction("Save Project", self)
        act_save.setShortcut("Ctrl+S")
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save Project As...", self)
        act_save_as.setShortcut("Ctrl+Shift+S")
        act_save_as.triggered.connect(self._save_project_as)
        file_menu.addAction(act_save_as)

        file_menu.addSeparator()

        act_report = QAction("Export HTML Report...", self)
        act_report.setShortcut("Ctrl+H")
        act_report.triggered.connect(self._export_report)
        file_menu.addAction(act_report)

        act_sparse = QAction("Export Sparse Replicates CSV...", self)
        act_sparse.triggered.connect(self._export_sparse_replicates)
        file_menu.addAction(act_sparse)

        file_menu.addSeparator()

        act_exit = QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        analysis_menu = mb.addMenu("Analysis")
        act_recompute = QAction("Recompute Statistics", self)
        act_recompute.setShortcut("Ctrl+R")
        act_recompute.triggered.connect(self._run_analysis)
        analysis_menu.addAction(act_recompute)

        examples_menu = mb.addMenu("Examples")
        act_example = QAction("Example 1: CFD Validation (N=60)", self)
        act_example.triggered.connect(self._load_example_data)
        examples_menu.addAction(act_example)
        act_example2 = QAction("Example 2: Small Sample (N=6)", self)
        act_example2.triggered.connect(self._load_example_small_sample)
        examples_menu.addAction(act_example2)
        act_example3 = QAction("Example 3: Autocorrelated Data (AR(1))", self)
        act_example3.triggered.connect(self._load_example_autocorrelated)
        examples_menu.addAction(act_example3)

        help_menu = mb.addMenu("Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    # ---- save / load ----

    def _new_project(self):
        if not self._check_unsaved_changes():
            return
        self._tab_data._clear_data()
        self._tab_stats.clear()
        self._tab_charts.clear()
        self._tab_carry.clear()
        self._project_path = ""
        self._project_name = "Untitled"
        self._lbl_project_name.setText("Untitled")
        self._edit_program.clear()
        self._edit_analyst.clear()
        self._edit_date.setText(datetime.datetime.now().strftime("%Y-%m-%d"))
        self._edit_notes.clear()
        self._cmb_consequence.setCurrentText("Medium")
        self._unsaved_changes = False
        self._status_bar.showMessage("New project created.", 3000)

    def _load_example_data(self):
        if not self._check_unsaved_changes():
            return
        self._tab_data._load_example_dataset()
        self._tabs.setCurrentWidget(self._tab_data)
        self._status_bar.showMessage(
            "Built-in example loaded. Press Ctrl+R to compute statistics.", 5000
        )

    def _load_example_small_sample(self):
        if not self._check_unsaved_changes():
            return
        self._tab_data._load_example_small_sample()
        self._tabs.setCurrentWidget(self._tab_data)
        self._status_bar.showMessage(
            "Small-sample example loaded (N=6). Press Ctrl+R to compute statistics.", 5000
        )

    def _load_example_autocorrelated(self):
        if not self._check_unsaved_changes():
            return
        self._tab_data._load_example_autocorrelated()
        self._tabs.setCurrentWidget(self._tab_data)
        self._status_bar.showMessage(
            "Autocorrelated example loaded (N=200, AR(1)). Press Ctrl+R to compute statistics.", 5000
        )

    def _save_project(self):
        if self._project_path:
            self._do_save(self._project_path)
        else:
            self._save_project_as()

    def _save_project_as(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Project", os.path.expanduser("~"),
            "Statistical Analyzer Project (*.sta);;All Files (*)")
        if filepath:
            if not filepath.endswith('.sta'):
                filepath += '.sta'
            self._do_save(filepath)

    def _do_save(self, filepath):
        try:
            state = {
                'tool': APP_NAME,
                'version': APP_VERSION,
                'saved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'project_metadata': self.get_project_metadata(),
                'data': self._tab_data.get_state(),
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            self._project_path = filepath
            self._project_name = os.path.splitext(os.path.basename(filepath))[0]
            self._lbl_project_name.setText(self._project_name)
            self._unsaved_changes = False
            self._status_bar.showMessage(f"Saved: {filepath}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save:\n{e}")

    def _open_project(self):
        if not self._check_unsaved_changes():
            return
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Project", os.path.expanduser("~"),
            "Statistical Analyzer Project (*.sta);;All Files (*)")
        if not filepath:
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self.set_project_metadata(state.get('project_metadata', {}))
            self._tab_data.set_state(state.get('data', {}))
            self._project_path = filepath
            self._project_name = os.path.splitext(os.path.basename(filepath))[0]
            self._lbl_project_name.setText(self._project_name)
            self._unsaved_changes = False
            self._status_bar.showMessage(f"Opened: {filepath}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Open Error", f"Could not open:\n{e}")

    # ---- report ----

    def _export_report(self):
        results = self._tab_stats.get_results()
        if not results:
            QMessageBox.information(self, "No Results",
                                    "Run analysis first before exporting a report.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export HTML Report", os.path.expanduser("~"),
            "HTML Files (*.html);;All Files (*)")
        if not filepath:
            return
        try:
            # By design, the embedded chart is a snapshot of whatever the user
            # last viewed on the Charts tab — it is not regenerated here.
            chart_fig = self._tab_charts.get_figure()
            effective_results = self._tab_carry.get_effective_results(results)
            html = generate_html_report(
                effective_results, self.get_project_metadata(), chart_fig)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            self._status_bar.showMessage(f"Report exported: {filepath}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Could not export:\n{e}")

    def _export_sparse_replicates(self):
        """Export discrete replicate realizations for sparse-replicate carry mode."""
        columns = self._tab_data.get_columns()
        if not columns:
            QMessageBox.information(self, "No Data", "No data available to export.")
            return

        rows: List[Tuple[str, int, float, float]] = []
        for col in columns:
            # Use mirrored (analysis) values when mirror is enabled
            vals = col.analysis_values() if col.mirror else col.valid_values()
            if len(vals) == 0:
                continue
            weight = 1.0 / len(vals)
            for idx, v in enumerate(vals, start=1):
                rows.append((col.name, idx, float(v), weight))

        if not rows:
            QMessageBox.information(self, "No Data", "No valid numeric values found.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Sparse Replicates",
            os.path.expanduser("~/sparse_replicates.csv"),
            "CSV Files (*.csv);;All Files (*)")
        if not filepath:
            return
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write("variable,replicate_id,value,weight\n")
                for var, rid, val, w in rows:
                    safe_var = str(var).replace('"', '""')
                    f.write(f"\"{safe_var}\",{rid},{val:.12g},{w:.12g}\n")
            self._status_bar.showMessage(
                f"Sparse replicate export complete: {filepath}", 5000
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Could not export:\n{e}")

    def _show_about(self):
        QMessageBox.about(self, "About",
            f"<h3>{APP_NAME} v{APP_VERSION}</h3>"
            f"<p>Date: {APP_DATE}</p>"
            f"<p>Data distribution analysis tool for ASME V&V 20 "
            f"uncertainty budgets.</p>"
            f"<p>Fits 8 distributions (Normal, Log-Normal, Uniform, "
            f"Triangular, Weibull, Gamma, Student-t, Beta) using MLE "
            f"with AICc/BIC ranking and bootstrap goodness-of-fit testing.</p>"
            f"<p>Standards: JCGM 100:2008, JCGM 101:2008, "
            f"ASME PTC 19.1-2018, ASME V&V 20-2009 (R2021)</p>")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = StatisticalAnalyzerWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
