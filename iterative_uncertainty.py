#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterative Uncertainty Calculator v1.4
CFD Iterative Convergence Uncertainty Tool per ITTC 7.5-03-01-01

Standalone PySide6 application for computing iterative convergence
uncertainty from Fluent .out files. Supports both ITTC half-range
and sigma-based methods with time-weighted statistics.

Standards References:
    - ITTC (2024) "Uncertainty Analysis in CFD Verification and Validation"
      Procedure 7.5-03-01-01
    - ASME V&V 20-2009 (R2021) Section 5
    - JCGM 100:2008 (GUM)

Fluent Output Formats Supported:
    - Modern Report Definition format (columnar with quoted headers)
    - Legacy XY-plot format (Scheme/Lisp parenthesized)

Copyright (c) 2026. All rights reserved.
"""

import sys
import os
import re
import json
import io
import csv
import glob
import base64
import datetime
import textwrap
from html import escape as _html_esc
import subprocess
import importlib
import tempfile
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Iterable, Optional, List, Dict, Tuple, Any

APP_VERSION = "1.4.0"
APP_NAME = "Iterative Uncertainty Calculator"
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
from scipy.stats import norm, shapiro
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
        mpl_cache_dir = os.path.join(tempfile.gettempdir(), "iterative_uncertainty_mplconfig")
        os.makedirs(mpl_cache_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_cache_dir
    except OSError:
        pass

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout, QGroupBox, QLabel, QPushButton, QLineEdit,
    QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QTableWidget, QTableWidgetItem, QTreeWidget, QTreeWidgetItem,
    QHeaderView, QSplitter, QScrollArea, QFrame, QMessageBox,
    QAbstractItemView, QFileDialog, QStatusBar, QProgressBar, QMenu,
)
from PySide6.QtCore import Qt
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

# Time/iteration column auto-detection tokens
TIME_TOKENS = [
    "flow-time", "flow time", "flow_time", "time-step", "time step",
    "time_step", "physical-time", "physical time", "physical_time",
]
ITERATION_TOKENS = [
    "iteration", "iter", "step", "time-step-count",
]

# Unit auto-detection and conversion
UNIT_CONVERSION = {
    'K_to_F': lambda t: (t - 273.15) * 9.0 / 5.0 + 32.0,
    'Pa_to_psia': lambda p: p / 6894.757,
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
    QProgressBar {{
        background-color: {c['bg_input']}; border: 1px solid {c['border']};
        border-radius: 4px; text-align: center; color: {c['fg']};
    }}
    QProgressBar::chunk {{
        background-color: {c['accent']}; border-radius: 3px;
    }}
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
    QTreeWidget {{
        background-color: {c['bg_widget']}; color: {c['fg']};
        border: 1px solid {c['border']}; border-radius: 4px;
        alternate-background-color: {c['bg_alt']};
    }}
    QTreeWidget::item:selected {{ background-color: {c['selection']}; }}
    QSplitter::handle {{ background-color: {c['border']}; }}
    QLabel {{ color: {c['fg']}; }}
    """


# =============================================================================
# GUIDANCE PANEL
# =============================================================================

class GuidancePanel(QFrame):
    SEVERITY_CONFIG = {
        'green':  {'border_color': DARK_COLORS['green'],  'bg_color': '#1a2e1a', 'icon': '\u2714'},
        'yellow': {'border_color': DARK_COLORS['yellow'], 'bg_color': '#2e2a1a', 'icon': '\u26A0'},
        'red':    {'border_color': DARK_COLORS['red'],    'bg_color': '#2e1a1a', 'icon': '\u2716'},
    }

    def __init__(self, title="", parent=None):
        super().__init__(parent)
        self._title = title
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
        self._title_label = QLabel(title)
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
# ENGINE: Fluent .out Parser
# =============================================================================

@dataclass
class FluentData:
    """Parsed Fluent .out file data."""
    filepath: str = ""
    case_name: str = ""
    file_base: str = ""
    headers: List[str] = field(default_factory=list)
    time_column: str = ""
    time_col_idx: int = -1
    is_transient: bool = False
    data: Any = None  # np.ndarray (rows x cols)
    units: Dict[str, str] = field(default_factory=dict)
    format_type: str = ""  # "report_definition" or "xy_plot"
    warnings: List[str] = field(default_factory=list)


def detect_fluent_format(filepath: str) -> str:
    """Detect whether a .out file is Report Definition or XY-plot format."""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = []
            for i, line in enumerate(f):
                lines.append(line)
                if i > 20:
                    break
        for line in lines:
            stripped = line.strip().lower()
            # XY-plot markers from Fluent user guide:
            # (title ...), (labels ...), ((xy/key/label ...)
            if stripped.startswith("((xy/key/label"):
                return "xy_plot"
            if stripped.startswith("(title ") or stripped.startswith("(labels "):
                return "xy_plot"
            if "xy/key/label" in stripped:
                return "xy_plot"
        # Check for quoted header pattern (Report Definition)
        for line in lines:
            if re.search(r'"[^"]*"', line):
                return "report_definition"
        # Default to report_definition
        return "report_definition"
    except Exception:
        return "report_definition"


def parse_fluent_report_definition(filepath: str) -> FluentData:
    """Parse modern Fluent Report Definition .out format.

    Format:
        Line 1-2: metadata (ignored)
        Line 3: quoted headers: "col1" "col2" "col3"
        Line 4+: numeric data (space or tab separated)
    """
    result = FluentData(filepath=filepath, format_type="report_definition")
    result.file_base = os.path.splitext(os.path.basename(filepath))[0]
    result.case_name = os.path.basename(os.path.dirname(filepath))

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            all_lines = f.readlines()
    except Exception as e:
        result.warnings.append(f"Read error: {e}")
        return result

    if len(all_lines) < 3:
        result.warnings.append("File has fewer than 3 lines.")
        return result

    # Find header line:
    # prefer a line with multiple quoted fields (report columns), not the title.
    header_line_idx = -1
    fallback_header_idx = -1
    for i, line in enumerate(all_lines):
        quoted = re.findall(r'"([^"]*)"', line)
        if not quoted:
            continue
        if fallback_header_idx < 0:
            fallback_header_idx = i
        if len(quoted) >= 2:
            header_line_idx = i
            break

    if header_line_idx < 0 and fallback_header_idx >= 0:
        header_line_idx = fallback_header_idx

    if header_line_idx < 0:
        # Try to parse as pure numeric with no headers
        result.warnings.append("No quoted headers found; attempting headerless parse.")
        header_line_idx = -1
        data_start = 0
        headers = []
    else:
        raw_headers = re.findall(r'"([^"]*)"', all_lines[header_line_idx])
        # Handle duplicate headers
        seen = {}
        headers = []
        for h in raw_headers:
            if h in seen:
                seen[h] += 1
                headers.append(f"{h}_{seen[h]}")
            else:
                seen[h] = 0
                headers.append(h)
        data_start = header_line_idx + 1

    # Parse numeric data
    rows = []
    for line in all_lines[data_start:]:
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('('):
            continue
        parts = re.split(r'[\s,]+', line)
        row = []
        for p in parts:
            try:
                row.append(float(p))
            except ValueError:
                continue
        if row:
            rows.append(row)

    if not rows:
        result.warnings.append("No numeric data found.")
        return result

    # Pad rows to equal length
    max_cols = max(len(r) for r in rows)
    for r in rows:
        while len(r) < max_cols:
            r.append(float('nan'))

    if not headers:
        headers = [f"Variable_{i+1}" for i in range(max_cols)]
    elif len(headers) < max_cols:
        for i in range(len(headers), max_cols):
            headers.append(f"Variable_{i+1}")
    elif len(headers) > max_cols:
        headers = headers[:max_cols]

    result.headers = headers
    result.data = np.array(rows, dtype=float)

    # Detect time/iteration column
    _detect_time_column(result)
    # Detect units
    _detect_units(result)

    return result


def parse_fluent_xy_plot(filepath: str) -> FluentData:
    """Parse legacy Fluent XY-plot .out format.

    Format: Scheme/Lisp parenthesized data
        ((xy/key/label "variable name")
         row_idx  value
         ...
        )
    """
    result = FluentData(filepath=filepath, format_type="xy_plot")
    result.file_base = os.path.splitext(os.path.basename(filepath))[0]
    result.case_name = os.path.basename(os.path.dirname(filepath))

    try:
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        content = content.replace('\r\n', '\n').replace('\r', '\n')
    except Exception as e:
        result.warnings.append(f"Read error: {e}")
        return result

    # Extract variable names from labels
    labels = re.findall(r'xy/key/label\s+"([^"]*)"', content)
    if not labels:
        labels = re.findall(r'"([^"]*)"', content)

    # Extract numeric data blocks
    # Robust numeric pattern: handles negative numbers and scientific notation
    # e.g., 1.23, -1.23, 1.23e-05, -1.23E+10, .5, -.5
    _num = r'-?\.?\d+\.?\d*(?:[eE][+\-]?\d+)?'
    # Pattern: sequences of "number number" lines between parens
    blocks = re.findall(r'\(\s*\n?((?:\s*' + _num + r'\s+' + _num + r'\s*\n?)+)\s*\)', content)

    if not blocks:
        # Try simpler pattern
        numbers = re.findall(r'(' + _num + r')\s+(' + _num + r')', content)
        if numbers:
            try:
                col1 = [float(n[0]) for n in numbers]
                col2 = [float(n[1]) for n in numbers]
                result.headers = ["Index", labels[0] if labels else "Value"]
                result.data = np.column_stack([col1, col2])
            except (ValueError, IndexError):
                result.warnings.append("Could not parse XY data.")
                return result
        else:
            result.warnings.append("No numeric data blocks found.")
            return result
    else:
        # Parse each block as a variable
        all_cols = {}
        for i, block in enumerate(blocks):
            pairs = re.findall(r'(' + _num + r')\s+(' + _num + r')', block)
            indices = [float(p[0]) for p in pairs]
            values = [float(p[1]) for p in pairs]
            name = labels[i] if i < len(labels) else f"Variable_{i+1}"
            all_cols[name] = (indices, values)

        if not all_cols:
            result.warnings.append("No data extracted from XY blocks.")
            return result

        # Use the first block's indices as the index column
        first_name = list(all_cols.keys())[0]
        indices = all_cols[first_name][0]
        n_rows = len(indices)

        # Warn when XY-plot blocks have different row counts
        row_counts = {name: len(vals) for name, (idx, vals) in all_cols.items()}
        unique_counts = set(row_counts.values())
        if len(unique_counts) > 1:
            counts_str = ", ".join(f"{name}: {cnt}" for name, cnt in row_counts.items())
            result.warnings.append(
                f"XY-plot blocks have different row counts ({counts_str}). "
                f"Data will be padded/truncated to {n_rows} rows.")

        headers = ["Index"]
        data_cols = [indices]
        for name, (idx, vals) in all_cols.items():
            headers.append(name)
            # Pad if needed
            if len(vals) < n_rows:
                vals = vals + [float('nan')] * (n_rows - len(vals))
            elif len(vals) > n_rows:
                vals = vals[:n_rows]
            data_cols.append(vals)

        result.headers = headers
        result.data = np.column_stack(data_cols)

    _detect_time_column(result)
    _detect_units(result)
    return result


def parse_fluent_out(filepath: str) -> FluentData:
    """Auto-detect format and parse a Fluent .out file."""
    fmt = detect_fluent_format(filepath)
    if fmt == "xy_plot":
        return parse_fluent_xy_plot(filepath)
    else:
        return parse_fluent_report_definition(filepath)


def _detect_time_column(result: FluentData):
    """Detect which column is time/iteration."""
    for i, h in enumerate(result.headers):
        h_lower = h.lower().strip()
        for token in TIME_TOKENS:
            if token in h_lower:
                result.time_column = h
                result.time_col_idx = i
                result.is_transient = True
                return
    for i, h in enumerate(result.headers):
        h_lower = h.lower().strip()
        for token in ITERATION_TOKENS:
            if token in h_lower:
                result.time_column = h
                result.time_col_idx = i
                result.is_transient = False
                return
    # Default: first column
    if result.headers:
        result.time_column = result.headers[0]
        result.time_col_idx = 0
        result.is_transient = False


def _detect_units(result: FluentData):
    """Auto-detect units from header keywords first, then median-value heuristic.

    Priority order:
      1. Header name contains a known keyword (temperature, pressure, etc.)
      2. Median-value heuristic (>2000 → Pa, >200 → K)
      3. Default to empty string (user must set manually)

    Note:
      Velocity-like variables are intentionally not auto-assigned to "m/s".
      Fluent projects commonly mix SI and English units, and velocity labels
      alone are not reliable enough to infer units safely.
    """
    _TEMP_KEYWORDS = {"temperature", "temp", "static-temperature",
                      "total-temperature", "wall-temp", "t-wall"}
    _PRESS_KEYWORDS = {"pressure", "press", "static-pressure",
                       "total-pressure", "p-static", "p-total",
                       "dynamic-pressure", "gauge-pressure"}
    if result.data is None or result.data.size == 0:
        return
    for i, h in enumerate(result.headers):
        if i == result.time_col_idx:
            continue
        h_lower = h.strip().lower()
        # Keyword-based detection (highest confidence)
        if any(kw in h_lower for kw in _PRESS_KEYWORDS):
            result.units[h] = "Pa"
            continue
        if any(kw in h_lower for kw in _TEMP_KEYWORDS):
            result.units[h] = "K"
            continue
        # Do not auto-tag velocity units from names alone.
        # Leave blank unless user sets units explicitly.
        _VELOCITY_KEYWORDS = {"velocity", "mach", "speed", "vel-mag",
                              "x-velocity", "y-velocity", "z-velocity"}
        if any(vkw in h_lower for vkw in _VELOCITY_KEYWORDS):
            result.units[h] = ""
            continue
        # Fallback: median-value heuristic with physical range validation
        col = result.data[:, i]
        valid = col[np.isfinite(col)]
        if len(valid) == 0:
            continue
        median_val = np.median(np.abs(valid))
        if median_val > 2000 and 1e2 < median_val < 1e8:
            result.units[h] = "Pa"
        elif median_val > 200 and 100 < median_val < 5000:
            result.units[h] = "K"
        else:
            result.units[h] = ""


# =============================================================================
# ENGINE: Statistics
# =============================================================================

@dataclass
class IterativeStats:
    """Statistics for one variable from one file."""
    variable: str = ""
    source_file: str = ""
    source_case: str = ""
    unit: str = ""
    converted_unit: str = ""
    n_samples: int = 0
    final_value: float = 0.0
    mean: float = 0.0
    median: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    sigma: float = 0.0
    three_sigma: float = 0.0
    p95: float = 0.0
    corrected_3sigma: float = 0.0
    difference: float = 0.0
    cov: float = 0.0
    half_range: float = 0.0
    # Time-weighted versions
    tw_mean: float = 0.0
    tw_sigma: float = 0.0
    tw_cov: float = 0.0
    # Autocorrelation-aware sample size
    n_eff: float = 0.0
    autocorr_rho1: float = 0.0
    autocorr_tau: float = 1.0
    autocorr_warning: bool = False
    # Stationarity diagnostics
    trend_slope: float = 0.0
    trend_pvalue: float = 1.0
    drift_ratio: float = 0.0
    allan_ratio: float = 1.0
    stationarity_pass: bool = True
    stationarity_note: str = ""


def compute_time_weighted_stats(values: np.ndarray,
                                 times: np.ndarray) -> Tuple[float, float]:
    """Compute time-weighted mean and std dev for non-uniform dt."""
    def _safe_mean_std(arr: np.ndarray) -> Tuple[float, float]:
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            return 0.0, 0.0
        if len(arr) == 1:
            return float(arr[0]), 0.0
        return float(np.mean(arr)), float(np.std(arr, ddof=1))

    if len(values) < 2 or len(times) < 2:
        return _safe_mean_std(values)

    valid_mask = np.isfinite(values) & np.isfinite(times)
    if np.count_nonzero(valid_mask) < 2:
        return _safe_mean_std(values)
    values = values[valid_mask]
    times = times[valid_mask]

    # Ensure strictly increasing time for weight computation.
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    dt = np.diff(times)
    if len(dt) == 0 or np.any(dt <= 0):
        return float(np.mean(values)), float(np.std(values, ddof=1))

    # Trapezoidal weights
    weights = np.zeros(len(values))
    weights[0] = dt[0] / 2.0
    weights[-1] = dt[-1] / 2.0
    for i in range(1, len(values) - 1):
        weights[i] = (dt[i - 1] + dt[i]) / 2.0
    total_w = np.sum(weights)
    if total_w <= 0:
        return float(np.mean(values)), float(np.std(values, ddof=1))
    tw_mean = float(np.sum(weights * values) / total_w)
    # Reliability weights correction for weighted variance
    V1 = np.sum(weights)
    V2 = np.sum(weights**2)
    denom = V1 - V2 / V1
    if denom <= 0:
        tw_var = float(np.sum(weights * (values - tw_mean) ** 2) / total_w)
    else:
        tw_var = float(np.sum(weights * (values - tw_mean) ** 2) / denom)
    tw_sigma = float(np.sqrt(tw_var))
    return tw_mean, tw_sigma


def estimate_effective_sample_size(values: np.ndarray) -> Tuple[float, float, float]:
    """Estimate effective sample size for serially correlated traces."""
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
        if lag > 1 and rho < 0:
            break
        rho_vals.append(rho)

    if not rho_vals:
        return float(n), 0.0, 1.0

    tau_int = 1.0
    for lag, rho in enumerate(rho_vals, start=1):
        tau_int += 2.0 * max(0.0, rho) * (1.0 - lag / n)
    tau_int = max(1.0, tau_int)
    n_eff = float(n) / tau_int
    n_eff = min(float(n), max(2.0, n_eff))
    return float(n_eff), float(rho_vals[0]), float(tau_int)


def _allan_deviation(values: np.ndarray, m: int) -> float:
    """Simple Allan deviation estimate for averaging factor m."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2 * m or m <= 0:
        return float('nan')
    k = n // m
    if k < 2:
        return float('nan')
    means = x[:k * m].reshape(k, m).mean(axis=1)
    diffs = np.diff(means)
    if len(diffs) == 0:
        return float('nan')
    return float(np.sqrt(0.5 * np.mean(diffs ** 2)))


def stationarity_diagnostics(values: np.ndarray,
                             times: Optional[np.ndarray] = None,
                             trend_alpha: float = 0.05,
                             drift_limit: float = 0.5,
                             allan_limit: float = 1.5) -> Dict[str, float]:
    """Stationarity checks: trend, drift split, and Allan ratio.

    Args:
        trend_alpha: p-value threshold for linear-regression trend test.
        drift_limit: max allowed drift_ratio (|mean_B - mean_A| / sigma).
        allan_limit: max allowed Allan deviation ratio (tau=4 / tau=1).
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return {
            'trend_slope': 0.0,
            'trend_pvalue': 1.0,
            'drift_ratio': 0.0,
            'allan_ratio': 1.0,
            'pass': False,
            'note': "Stationarity not demonstrated (N < 8).",
        }

    if times is not None and len(times) == n:
        tvals = np.asarray(times, dtype=float)
        tvals = tvals[np.isfinite(tvals)]
        if len(tvals) != n:
            tvals = np.arange(n, dtype=float)
    else:
        tvals = np.arange(n, dtype=float)

    try:
        lr = sp_stats.linregress(tvals, x)
        slope = float(lr.slope)
        pval = float(lr.pvalue) if np.isfinite(lr.pvalue) else 1.0
    except Exception:
        slope, pval = 0.0, 1.0

    half = n // 2
    mu_a = float(np.mean(x[:half]))
    mu_b = float(np.mean(x[half:]))
    sigma = float(np.std(x, ddof=1))
    drift_ratio = abs(mu_b - mu_a) / max(sigma, 1e-12)

    adev_1 = _allan_deviation(x, 1)
    adev_4 = _allan_deviation(x, 4)
    if np.isfinite(adev_1) and np.isfinite(adev_4) and adev_1 > 0:
        allan_ratio = float(adev_4 / adev_1)
    else:
        allan_ratio = 1.0

    passed = (pval >= trend_alpha) and (drift_ratio <= drift_limit) and (allan_ratio <= allan_limit)
    if passed:
        note = "Stationarity checks passed."
    else:
        fail_bits = []
        if pval < trend_alpha:
            fail_bits.append("trend detected")
        if drift_ratio > drift_limit:
            fail_bits.append("mean drift between halves")
        if allan_ratio > allan_limit:
            fail_bits.append("low-frequency drift indicator")
        note = "Stationarity gate failed: " + ", ".join(fail_bits) + "."

    return {
        'trend_slope': slope,
        'trend_pvalue': pval,
        'drift_ratio': float(drift_ratio),
        'allan_ratio': float(allan_ratio),
        'pass': bool(passed),
        'note': note,
    }


def compute_iterative_stats(values: np.ndarray, times: Optional[np.ndarray] = None,
                             variable: str = "", unit: str = "",
                             source_file: str = "", source_case: str = "",
                             convert_units: bool = True,
                             trend_alpha: float = 0.05,
                             drift_limit: float = 0.5,
                             allan_limit: float = 1.5) -> IterativeStats:
    """Compute all iterative convergence statistics for one variable."""
    result = IterativeStats(
        variable=variable,
        source_file=source_file,
        source_case=source_case,
        unit=unit,
    )
    n = len(values)
    if n < 2:
        return result

    # Unit conversion
    converted = values.copy()
    conv_unit = unit
    if convert_units and unit:
        if unit == "K":
            converted = UNIT_CONVERSION['K_to_F'](values)
            conv_unit = "\u00b0F"
        elif unit == "Pa":
            converted = UNIT_CONVERSION['Pa_to_psia'](values)
            conv_unit = "psia"
    result.converted_unit = conv_unit

    result.n_samples = n
    result.final_value = float(converted[-1])
    result.mean = float(np.mean(converted))
    result.median = float(np.median(converted))
    result.min_val = float(np.min(converted))
    result.max_val = float(np.max(converted))
    result.sigma = float(np.std(converted, ddof=1))
    result.three_sigma = 3.0 * result.sigma
    result.p95 = float(np.percentile(converted, 95))
    result.corrected_3sigma = min(result.max_val, result.mean + result.three_sigma)
    result.difference = result.corrected_3sigma - result.final_value
    result.cov = result.sigma / abs(result.mean) if abs(result.mean) > 1e-15 else 0.0
    result.half_range = 0.5 * (result.max_val - result.min_val)

    # Time-weighted stats
    if times is not None and len(times) == n:
        tw_m, tw_s = compute_time_weighted_stats(converted, times)
        result.tw_mean = tw_m
        result.tw_sigma = tw_s
        result.tw_cov = tw_s / abs(tw_m) if abs(tw_m) > 1e-15 else 0.0
    else:
        result.tw_mean = result.mean
        result.tw_sigma = result.sigma
        result.tw_cov = result.cov

    n_eff, rho1, tau_int = estimate_effective_sample_size(converted)
    result.n_eff = n_eff
    result.autocorr_rho1 = rho1
    result.autocorr_tau = tau_int
    result.autocorr_warning = n_eff < (0.8 * n)

    diag = stationarity_diagnostics(converted, times=times,
                                     trend_alpha=trend_alpha,
                                     drift_limit=drift_limit,
                                     allan_limit=allan_limit)
    result.trend_slope = float(diag['trend_slope'])
    result.trend_pvalue = float(diag['trend_pvalue'])
    result.drift_ratio = float(diag['drift_ratio'])
    result.allan_ratio = float(diag['allan_ratio'])
    result.stationarity_pass = bool(diag['pass'])
    result.stationarity_note = str(diag['note'])

    return result


def derive_carry_over_value(
    s: IterativeStats,
    method: str,
    allow_stationarity_override: bool = False,
    override_reason: str = "",
) -> Dict[str, str]:
    """Return a concrete carry-over recommendation for the Aggregator."""
    method = method.strip()
    sigma_pref = s.tw_sigma if np.isfinite(s.tw_sigma) and s.tw_sigma > 0 else s.sigma

    if not s.stationarity_pass:
        if not allow_stationarity_override:
            return {
                "u_carry": "INVALID",
                "sigma_basis": "N/A",
                "distribution": "N/A",
                "combine_method": "STOP",
                "formula": "Stationarity gate failed",
                "note": (
                    "Do not carry this value: stationarity checks failed. "
                    "Do this now: increase Last N or move the analysis window "
                    "to a steady region, then recompute. Enable override only "
                    "with documented rationale."
                ),
            }
        if not override_reason.strip():
            return {
                "u_carry": "INVALID",
                "sigma_basis": "N/A",
                "distribution": "N/A",
                "combine_method": "STOP",
                "formula": "Stationarity gate failed",
                "note": (
                    "Override selected but rationale is missing. "
                    "Provide rationale to proceed. Without rationale, do not carry."
                ),
            }

    auto_note_suffix = ""
    if s.autocorr_warning:
        auto_note_suffix = (
            f" Data are correlated (N_eff={s.n_eff:.1f}, rho1={s.autocorr_rho1:.3f}); "
            "confidence is reduced."
        )

    if method == "Sigma-based":
        out = {
            "u_carry": f"{sigma_pref:.6g}",
            "sigma_basis": "Confirmed 1\u03c3",
            "distribution": "Normal",
            "combine_method": "RSS",
            "formula": "sigma (time-weighted for transient data)",
            "note": "Use this when residual variation is stable and near-normal.",
        }
        if auto_note_suffix:
            out["note"] += auto_note_suffix
        if allow_stationarity_override and override_reason.strip():
            out["note"] += f" Stationarity override rationale: {override_reason.strip()}"
        return out
    if method == "ITTC Half-Range":
        out = {
            "u_carry": f"{s.half_range:.6g}",
            "sigma_basis": "Bounding (min/max)",
            "distribution": "Uniform",
            "combine_method": "RSS",
            "formula": "0.5 x (max - min)",
            "note": "Conservative interval-style estimate from ITTC guidance.",
        }
        if auto_note_suffix:
            out["note"] += auto_note_suffix
        if allow_stationarity_override and override_reason.strip():
            out["note"] += f" Stationarity override rationale: {override_reason.strip()}"
        return out

    # "Both" mode: guide novice users to a single conservative value.
    if s.half_range >= sigma_pref:
        out = {
            "u_carry": f"{s.half_range:.6g}",
            "sigma_basis": "Bounding (min/max)",
            "distribution": "Uniform",
            "combine_method": "RSS (Conservative max)",
            "formula": "max(sigma, ITTC) = ITTC",
            "note": "Half-range dominates; use ITTC value as the carry-over.",
        }
        if auto_note_suffix:
            out["note"] += auto_note_suffix
        if allow_stationarity_override and override_reason.strip():
            out["note"] += f" Stationarity override rationale: {override_reason.strip()}"
        return out
    out = {
        "u_carry": f"{sigma_pref:.6g}",
        "sigma_basis": "Confirmed 1\u03c3",
        "distribution": "Normal",
        "combine_method": "RSS (Conservative max)",
        "formula": "max(sigma, ITTC) = sigma",
        "note": "Sigma dominates; use sigma as the carry-over.",
    }
    if auto_note_suffix:
        out["note"] += auto_note_suffix
    if allow_stationarity_override and override_reason.strip():
        out["note"] += f" Stationarity override rationale: {override_reason.strip()}"
    return out


def scan_out_files(root_folder: str) -> Dict[str, List[str]]:
    """Recursively scan for .out files, grouped by subdirectory (case)."""
    cases = {}
    root = Path(root_folder)
    for out_file in sorted(root.rglob("*.out")):
        # Case name = relative path of parent from root
        rel_parent = out_file.parent.relative_to(root)
        case_name = str(rel_parent) if str(rel_parent) != "." else root.name
        if case_name not in cases:
            cases[case_name] = []
        cases[case_name].append(str(out_file))
    return cases


def write_stats_csv(stats_list: List[IterativeStats], filepath: str):
    """Write per-case statistics CSV."""
    if not stats_list:
        return
    headers = [
        "Variable", "Unit", "N", "Final Value", "Mean", "Median",
        "Min", "Max", "Sigma", "3*Sigma", "P95",
        "Corrected 3-Sigma", "Difference", "CoV",
        "Half-Range (ITTC)", "TW Mean", "TW Sigma", "TW CoV",
        "N_eff", "rho1", "tau_int", "Trend p", "Drift Ratio",
        "Allan Ratio", "Stationarity Pass", "Stationarity Note",
    ]
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for s in stats_list:
            row = [
                s.variable, s.converted_unit or s.unit, str(s.n_samples),
                f"{s.final_value:.8g}", f"{s.mean:.8g}", f"{s.median:.8g}",
                f"{s.min_val:.8g}", f"{s.max_val:.8g}", f"{s.sigma:.8g}",
                f"{s.three_sigma:.8g}", f"{s.p95:.8g}",
                f"{s.corrected_3sigma:.8g}", f"{s.difference:.8g}",
                f"{s.cov:.8g}", f"{s.half_range:.8g}",
                f"{s.tw_mean:.8g}", f"{s.tw_sigma:.8g}", f"{s.tw_cov:.8g}",
                f"{s.n_eff:.8g}", f"{s.autocorr_rho1:.8g}", f"{s.autocorr_tau:.8g}",
                f"{s.trend_pvalue:.8g}", f"{s.drift_ratio:.8g}",
                f"{s.allan_ratio:.8g}", "1" if s.stationarity_pass else "0",
                s.stationarity_note,
            ]
            writer.writerow(row)


# =============================================================================
# TAB 1: DATA IMPORT
# =============================================================================

class DataImportTab(QWidget):
    """Folder selection, recursive .out scanning, file listing."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._root_folder: str = ""
        self._cases: Dict[str, List[str]] = {}
        self._parsed: Dict[str, List[FluentData]] = {}  # case -> list of FluentData
        self._setup_ui()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)

        # Left: controls
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)

        grp = QGroupBox("Folder Selection")
        gl = QVBoxLayout(grp)
        row = QHBoxLayout()
        self._edit_folder = QLineEdit()
        self._edit_folder.setPlaceholderText("Select root folder containing .out files...")
        self._edit_folder.setReadOnly(True)
        self._edit_folder.setToolTip(
            "Root folder path. The scanner looks for Fluent .out files\n"
            "in this folder and all subfolders (one subfolder = one case).")
        row.addWidget(self._edit_folder)
        self._btn_browse = QPushButton("Browse...")
        self._btn_browse.setToolTip("Open a file dialog to select the root folder.")
        self._btn_browse.clicked.connect(self._browse_folder)
        row.addWidget(self._btn_browse)
        gl.addLayout(row)

        self._btn_scan = QPushButton("Scan for .out Files")
        self._btn_scan.setToolTip(
            "Recursively scan the root folder for Fluent .out files.\n"
            "Subfolders are treated as separate cases.\n"
            "Supports report-definition, xy-plot, and custom CSV formats.")
        self._btn_scan.setStyleSheet(
            f"QPushButton {{ background-color: {DARK_COLORS['accent']}; "
            f"color: {DARK_COLORS['bg']}; font-weight: bold; padding: 8px 24px; }}"
            f"QPushButton:hover {{ background-color: {DARK_COLORS['accent_hover']}; }}")
        self._btn_scan.clicked.connect(self._scan_files)
        gl.addWidget(self._btn_scan)
        self._btn_example = QPushButton("Load Built-in Example \u25bc")
        example_menu = QMenu(self._btn_example)
        example_menu.addAction(
            "Example 1: Steady Thermal (Stationary)", self._load_example_dataset)
        example_menu.addAction(
            "Example 2: Transient Drift (Non-Stationary)", self._load_example_nonstationary)
        self._btn_example.setMenu(example_menu)
        self._btn_example.setToolTip(
            "Load a built-in example dataset to explore the tool\u2019s\n"
            "features without importing your own data.")
        gl.addWidget(self._btn_example)
        ll.addWidget(grp)

        # Options
        grp_opts = QGroupBox("Options")
        go_lay = QVBoxLayout(grp_opts)
        self._chk_convert = QCheckBox("Apply unit conversion (K\u2192\u00b0F, Pa\u2192psia)")
        self._chk_convert.setChecked(True)
        self._chk_convert.setToolTip(
            "When checked, auto-detected temperature (K) and pressure (Pa)\n"
            "columns are converted to \u00b0F and psia for display and statistics.\n"
            "Carry-over values are reported in the converted units.")
        go_lay.addWidget(self._chk_convert)
        ll.addWidget(grp_opts)

        self._guidance = GuidancePanel("Data Import")
        self._guidance.set_guidance(
            "Select a root folder containing Fluent .out files. "
            "Each subdirectory is treated as a separate case/condition. "
            "Both Report Definition and legacy XY-plot formats are auto-detected. "
            "You can also load built-in synthetic cases for a quick dry run.",
            'green')
        ll.addWidget(self._guidance)
        ll.addStretch()
        splitter.addWidget(left)

        # Right: file tree
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Case / File", "Format", "Rows", "Columns"])
        self._tree.setAlternatingRowColors(True)
        self._tree.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        rl.addWidget(self._tree)

        self._lbl_summary = QLabel("No files scanned.")
        self._lbl_summary.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; font-size: 11px;")
        rl.addWidget(self._lbl_summary)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 2)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Root Folder", os.path.expanduser("~"))
        if folder:
            self._root_folder = folder
            self._edit_folder.setText(folder)

    def _scan_files(self):
        if not self._root_folder:
            QMessageBox.information(self, "No Folder", "Select a folder first.")
            return
        self._cases = scan_out_files(self._root_folder)
        if not self._cases:
            self._guidance.set_guidance(
                "No .out files found in the selected folder.", 'red')
            return

        # Parse all files
        self._parsed = {}
        self._tree.clear()
        total_files = 0
        total_rows = 0
        for case_name, file_list in sorted(self._cases.items()):
            case_item = QTreeWidgetItem([case_name, "", "", ""])
            case_item.setExpanded(True)
            self._parsed[case_name] = []
            for fp in file_list:
                fd = parse_fluent_out(fp)
                self._parsed[case_name].append(fd)
                n_rows = fd.data.shape[0] if fd.data is not None else 0
                n_cols = fd.data.shape[1] if fd.data is not None else 0
                total_rows += n_rows
                total_files += 1
                file_item = QTreeWidgetItem([
                    os.path.basename(fp), fd.format_type,
                    str(n_rows), str(n_cols)])
                if fd.warnings:
                    file_item.setForeground(0, QColor(DARK_COLORS['yellow']))
                    file_item.setToolTip(0, "\n".join(fd.warnings))
                case_item.addChild(file_item)
            self._tree.addTopLevelItem(case_item)

        self._lbl_summary.setText(
            f"{len(self._cases)} case(s), {total_files} file(s), "
            f"{total_rows} total rows.")
        self._guidance.set_guidance(
            f"Scanned {total_files} .out files across {len(self._cases)} cases.",
            'green')

    def _load_example_dataset(self):
        """Create synthetic transient cases for quick workflow validation."""
        rng = np.random.default_rng(7)
        self._parsed = {}
        self._cases = {}
        self._tree.clear()

        case_specs = {
            "example_takeoff": (840.0, 14200.0),
            "example_cruise": (805.0, 10800.0),
            "example_idle": (770.0, 7600.0),
        }

        n_rows = 450
        total_rows = 0
        for case_name, (temp_base, flux_base) in case_specs.items():
            dt = rng.uniform(0.2, 1.5, n_rows)
            times = np.cumsum(dt)
            temp = (
                temp_base
                + 3.5 * np.exp(-times / 55.0) * np.sin(0.42 * times)
                + rng.normal(0.0, 0.12, n_rows)
            )
            flux = (
                flux_base
                + 120.0 * np.exp(-times / 70.0) * np.cos(0.35 * times + 0.5)
                + rng.normal(0.0, 9.0, n_rows)
            )

            headers = [
                "flow-time",
                "Area-Weighted Average of Static Temperature",
                "Area-Weighted Average of Wall Heat Flux",
            ]
            data = np.column_stack([times, temp, flux])
            fd = FluentData(
                filepath=f"[example]/{case_name}/surface_report.out",
                case_name=case_name,
                file_base="surface_report",
                headers=headers,
                time_column="flow-time",
                time_col_idx=0,
                is_transient=True,
                data=data,
                units={
                    headers[1]: "K",
                    headers[2]: "",
                },
                format_type="synthetic_example",
                warnings=[],
            )
            self._parsed[case_name] = [fd]
            self._cases[case_name] = [fd.filepath]
            total_rows += n_rows

            case_item = QTreeWidgetItem([case_name, "", "", ""])
            case_item.setExpanded(True)
            file_item = QTreeWidgetItem([
                os.path.basename(fd.filepath),
                fd.format_type,
                str(fd.data.shape[0]),
                str(fd.data.shape[1]),
            ])
            case_item.addChild(file_item)
            self._tree.addTopLevelItem(case_item)

        self._lbl_summary.setText(
            f"{len(self._parsed)} case(s), {len(self._parsed)} file(s), "
            f"{total_rows} total rows."
        )
        self._guidance.set_guidance(
            "Loaded built-in transient example cases. Use Analysis -> Compute "
            "to generate carry-over uncertainty values for the Aggregator.",
            'green'
        )

    def _load_example_nonstationary(self):
        """Create a synthetic non-stationary case with a clear upward trend.

        Purpose: demonstrate what non-converged iteration data looks like
        and how the stationarity gate correctly flags it as INVALID.
        Temperature drifts upward ~20 degF over 500 iterations with +-2 degF noise.
        """
        rng = np.random.default_rng(42)
        self._parsed = {}
        self._cases = {}
        self._tree.clear()

        case_name = "example_nonstationary"
        n_rows = 500

        # Iteration indices 1..500
        iterations = np.arange(1, n_rows + 1, dtype=float)

        # Temperature: starts ~350 degF, drifts upward ~20 degF over run
        # Linear trend: 20 degF over 500 iterations = 0.04 degF/iteration
        trend = 350.0 + (20.0 / n_rows) * iterations
        noise = rng.normal(0.0, 2.0, n_rows)
        temp = trend + noise

        headers = [
            "iteration",
            "Area-Weighted Average of Static Temperature",
        ]
        data = np.column_stack([iterations, temp])
        fd = FluentData(
            filepath=f"[example]/{case_name}/surface_report.out",
            case_name=case_name,
            file_base="surface_report",
            headers=headers,
            time_column="iteration",
            time_col_idx=0,
            is_transient=False,
            data=data,
            units={
                headers[1]: "degF",
            },
            format_type="synthetic_example",
            warnings=[],
        )
        self._parsed[case_name] = [fd]
        self._cases[case_name] = [fd.filepath]

        case_item = QTreeWidgetItem([case_name, "", "", ""])
        case_item.setExpanded(True)
        file_item = QTreeWidgetItem([
            os.path.basename(fd.filepath),
            fd.format_type,
            str(fd.data.shape[0]),
            str(fd.data.shape[1]),
        ])
        case_item.addChild(file_item)
        self._tree.addTopLevelItem(case_item)

        self._lbl_summary.setText(
            f"1 case(s), 1 file(s), {n_rows} total rows."
        )
        self._guidance.set_guidance(
            "Loaded non-stationary example (trending temperature). "
            "Use Analysis -> Compute to see the stationarity gate fire.",
            'green'
        )

    def get_parsed_data(self) -> Dict[str, List[FluentData]]:
        return self._parsed

    def is_convert_enabled(self) -> bool:
        return self._chk_convert.isChecked()

    def get_root_folder(self) -> str:
        return self._root_folder


# =============================================================================
# TAB 2: ANALYSIS
# =============================================================================

class AnalysisTab(QWidget):
    """Per-file statistics, variable selection, last-N configuration."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._all_stats: Dict[str, List[IterativeStats]] = {}  # case -> stats list
        self._current_case: str = ""
        self._current_file_idx: int = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        # Controls row
        top = QHBoxLayout()
        top.addWidget(QLabel("Case:"))
        self._cmb_case = QComboBox()
        self._cmb_case.setToolTip(
            "Select a case (subfolder) to analyze.\n"
            "Each case may contain one or more .out files.")
        self._cmb_case.currentTextChanged.connect(self._on_case_changed)
        top.addWidget(self._cmb_case)

        top.addWidget(QLabel("File:"))
        self._cmb_file = QComboBox()
        self._cmb_file.setToolTip(
            "Select which .out file within the case to analyze.\n"
            "Each file may contain multiple variables (columns).")
        self._cmb_file.currentIndexChanged.connect(self._on_file_changed)
        top.addWidget(self._cmb_file)

        top.addWidget(QLabel("Last N:"))
        self._spn_last_n = QSpinBox()
        self._spn_last_n.setRange(10, 100000)
        self._spn_last_n.setValue(1000)
        self._spn_last_n.setToolTip(
            "Number of last iterations/time-steps to use for statistics.\n"
            "Default: 1000 iterations for steady, 60s window for transient.")
        top.addWidget(self._spn_last_n)

        top.addWidget(QLabel("Window:"))
        self._cmb_window = QComboBox()
        self._cmb_window.addItems(["By Iterations", "By Time (seconds)"])
        self._cmb_window.setToolTip(
            "By Iterations: use last N rows.\n"
            "By Time: use last N seconds of flow-time (transient only).")
        top.addWidget(self._cmb_window)

        top.addWidget(QLabel("Method:"))
        self._cmb_method = QComboBox()
        self._cmb_method.addItems(["Sigma-based", "ITTC Half-Range", "Both"])
        self._cmb_method.setToolTip(
            "Sigma-based: \u03c3 from last N samples\n"
            "ITTC: U_I = 0.5 \u00d7 (max \u2212 min)\n"
            "Both: compute both and auto-select the more conservative carry-over value")
        self._cmb_method.currentTextChanged.connect(self._on_method_changed)
        top.addWidget(self._cmb_method)

        self._chk_stationarity_override = QCheckBox("Allow stationarity override")
        self._chk_stationarity_override.setToolTip(
            "If enabled, failed stationarity traces can be carried only when "
            "a rationale is documented."
        )
        self._chk_stationarity_override.toggled.connect(self._on_method_changed)
        top.addWidget(self._chk_stationarity_override)

        self._btn_compute = QPushButton("Compute")
        self._btn_compute.setToolTip(
            "Compute iterative convergence statistics for the selected file:\n"
            "  \u2022 Mean, \u03c3, min, max from the last N samples\n"
            "  \u2022 ITTC half-range uncertainty\n"
            "  \u2022 Stationarity diagnostics (trend, drift, Allan)\n"
            "  \u2022 Autocorrelation-adjusted effective sample size\n"
            "  \u2022 Carry-over values for the Aggregator")
        self._btn_compute.setStyleSheet(
            f"QPushButton {{ background-color: {DARK_COLORS['accent']}; "
            f"color: {DARK_COLORS['bg']}; font-weight: bold; }}"
            f"QPushButton:hover {{ background-color: {DARK_COLORS['accent_hover']}; }}")
        top.addWidget(self._btn_compute)
        top.addStretch()
        layout.addLayout(top)

        # Stationarity threshold controls
        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Stationarity thresholds \u2014"))
        thresh_row.addWidget(QLabel("Trend \u03b1:"))
        self._spn_trend_alpha = QDoubleSpinBox()
        self._spn_trend_alpha.setRange(0.001, 0.20)
        self._spn_trend_alpha.setSingleStep(0.01)
        self._spn_trend_alpha.setDecimals(3)
        self._spn_trend_alpha.setValue(0.05)
        self._spn_trend_alpha.setToolTip(
            "p-value threshold for the linear-regression trend test.\n"
            "Lower values are more lenient (allow more trend).\n"
            "Default: 0.05 (standard 95% confidence level).")
        thresh_row.addWidget(self._spn_trend_alpha)
        thresh_row.addWidget(QLabel("Drift limit:"))
        self._spn_drift_limit = QDoubleSpinBox()
        self._spn_drift_limit.setRange(0.1, 5.0)
        self._spn_drift_limit.setSingleStep(0.1)
        self._spn_drift_limit.setDecimals(2)
        self._spn_drift_limit.setValue(0.50)
        self._spn_drift_limit.setToolTip(
            "Maximum allowed drift ratio: |mean(2nd half) - mean(1st half)| / sigma.\n"
            "Higher values are more lenient.\n"
            "Default: 0.5 (half a standard deviation of drift).")
        thresh_row.addWidget(self._spn_drift_limit)
        thresh_row.addWidget(QLabel("Allan limit:"))
        self._spn_allan_limit = QDoubleSpinBox()
        self._spn_allan_limit.setRange(1.0, 5.0)
        self._spn_allan_limit.setSingleStep(0.1)
        self._spn_allan_limit.setDecimals(2)
        self._spn_allan_limit.setValue(1.50)
        self._spn_allan_limit.setToolTip(
            "Maximum allowed Allan deviation ratio (tau=4 / tau=1).\n"
            "Values > 1 indicate low-frequency drift.\n"
            "Default: 1.5.")
        thresh_row.addWidget(self._spn_allan_limit)
        thresh_row.addStretch()
        layout.addLayout(thresh_row)

        # Results table
        self._stats_table = QTableWidget()
        self._stats_table.setAlternatingRowColors(True)
        layout.addWidget(self._stats_table)

        grp_carry = QGroupBox("Carry-Over to Aggregator")
        grp_carry.setToolTip(
            "Values to transfer into the VVUQ Uncertainty Aggregator.\n"
            "Each row is one variable\u2019s iterative convergence uncertainty.")
        carry_lay = QVBoxLayout(grp_carry)
        self._carry_table = QTableWidget()
        self._carry_table.setAlternatingRowColors(True)
        self._carry_table.setColumnCount(8)
        self._carry_table.setHorizontalHeaderLabels([
            "Variable", "Unit", "Carry u", "Sigma Basis",
            "DOF", "Distribution for Aggregator", "Aggregator Analysis Mode",
            "How Derived",
        ])
        self._carry_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        carry_lay.addWidget(self._carry_table)

        override_row = QHBoxLayout()
        override_row.addWidget(QLabel("Override Rationale:"))
        self._edit_override_reason = QLineEdit()
        self._edit_override_reason.setToolTip(
            "Document why you are overriding a failed stationarity gate.\n"
            "Required when the override checkbox is enabled and the\n"
            "stationarity test has failed for any variable.")
        self._edit_override_reason.setPlaceholderText(
            "Required if stationarity override is enabled and any gate fails."
        )
        self._edit_override_reason.textChanged.connect(self._on_method_changed)
        override_row.addWidget(self._edit_override_reason)
        carry_lay.addLayout(override_row)

        layout.addWidget(grp_carry)

        # Guidance
        self._guidance = GuidancePanel("Analysis")
        self._guidance.set_guidance(
            "Select a case and file, set the window size (Last N), "
            "choose the method, and click Compute. Then use the carry-over "
            "table to transfer a single value into the Aggregator. Stationarity "
            "failures are blocked unless override is documented.", 'green')
        layout.addWidget(self._guidance)

    def populate_cases(self, parsed: Dict[str, List[FluentData]]):
        self._cmb_case.blockSignals(True)
        self._cmb_case.clear()
        for case_name in sorted(parsed.keys()):
            self._cmb_case.addItem(case_name)
        self._cmb_case.blockSignals(False)
        if parsed:
            self._on_case_changed(self._cmb_case.currentText())

    def _on_case_changed(self, case_name):
        self._current_case = case_name

    def _on_file_changed(self, idx):
        self._current_file_idx = idx

    def _on_method_changed(self, _method_text):
        case = self._cmb_case.currentText()
        if case in self._all_stats:
            self._update_carry_table(self._all_stats[case])

    def update_file_list(self, parsed: Dict[str, List[FluentData]], case_name: str):
        self._cmb_file.blockSignals(True)
        self._cmb_file.clear()
        if case_name in parsed:
            for fd in parsed[case_name]:
                self._cmb_file.addItem(os.path.basename(fd.filepath))
        self._cmb_file.blockSignals(False)

    def compute(self, parsed: Dict[str, List[FluentData]], convert: bool):
        """Compute stats for the selected case/file."""
        case = self._cmb_case.currentText()
        file_idx = self._cmb_file.currentIndex()
        if case not in parsed or file_idx < 0:
            return []

        fd = parsed[case][file_idx]
        if fd.data is None or fd.data.size == 0:
            self._guidance.set_guidance("No data in selected file.", 'red')
            return []

        last_n = self._spn_last_n.value()
        data = fd.data
        n_rows = data.shape[0]

        if last_n > 0.9 * n_rows:
            self._guidance.set_guidance(
                f"Warning: Last-N ({last_n}) exceeds 90% of available data "
                f"rows ({n_rows}). Statistics may not reflect converged behaviour.",
                'yellow')

        # Time-based windowing for transient cases
        window_mode = self._cmb_window.currentText()
        if window_mode == "By Time (seconds)" and fd.is_transient and fd.time_col_idx >= 0:
            time_col = data[:, fd.time_col_idx]
            # Validate monotonicity before searchsorted
            if not np.all(np.diff(time_col) >= 0):
                self._guidance.set_guidance(
                    "Warning: time column is not monotonically increasing; sorting data.",
                    'yellow')
                sort_order = np.argsort(time_col)
                data = data[sort_order]
                time_col = data[:, fd.time_col_idx]
            t_end = time_col[-1]
            t_start = t_end - last_n  # last_n is seconds in this mode
            start = int(np.searchsorted(time_col, t_start))
            start = max(0, start)
        else:
            start = max(0, n_rows - last_n)
        windowed = data[start:]

        times = None
        if fd.is_transient and fd.time_col_idx >= 0:
            times = windowed[:, fd.time_col_idx]

        stats_list = []
        for i, header in enumerate(fd.headers):
            if i == fd.time_col_idx:
                continue
            vals = windowed[:, i]
            valid_mask = np.isfinite(vals)
            valid = vals[valid_mask]
            if len(valid) < 2:
                continue
            unit = fd.units.get(header, "")
            times_var = None
            if times is not None and len(times) == len(vals):
                times_var = times[valid_mask]
            s = compute_iterative_stats(
                valid, times=times_var, variable=header,
                unit=unit, source_file=fd.file_base,
                source_case=case, convert_units=convert,
                trend_alpha=self._spn_trend_alpha.value(),
                drift_limit=self._spn_drift_limit.value(),
                allan_limit=self._spn_allan_limit.value())
            stats_list.append(s)

        # Display in table
        self._display_stats(stats_list)
        self._update_carry_table(stats_list)
        self._all_stats[case] = stats_list

        n_used = windowed.shape[0]
        n_stationarity_fail = sum(1 for s in stats_list if not s.stationarity_pass)
        n_corr_warn = sum(1 for s in stats_list if s.autocorr_warning)
        if n_stationarity_fail > 0:
            self._guidance.set_guidance(
                f"Computed {len(stats_list)} variables using last {n_used}/{n_rows} rows. "
                f"{n_stationarity_fail} variable(s) failed stationarity gate and are "
                "blocked from carry-over unless override rationale is provided.\n\n"
                "DO THIS NOW: adjust Last N/window to a steady region and recompute.",
                'red'
            )
        elif n_corr_warn > 0:
            self._guidance.set_guidance(
                f"Computed {len(stats_list)} variables using last {n_used}/{n_rows} rows. "
                f"{n_corr_warn} variable(s) show serial correlation; N_eff-adjusted DOF is applied.\n\n"
                "DO THIS NOW: use reported N_eff/DOF in the Aggregator (do not replace with N-1).",
                'yellow'
            )
        else:
            self._guidance.set_guidance(
                f"Computed statistics for {len(stats_list)} variables "
                f"using last {n_used} of {n_rows} rows. "
                "Carry-over values are ready in the carry-over table.\n\n"
                "DO THIS NOW: copy each row to the Aggregator exactly as shown.",
                'green'
            )
        return stats_list

    def _display_stats(self, stats_list: List[IterativeStats]):
        headers = [
            "Variable", "Unit", "N", "Final", "Mean", "Median",
            "Min", "Max", "\u03c3", "3\u03c3", "P95",
            "Corr. 3\u03c3", "Diff", "CoV",
            "ITTC U\u1d62", "TW Mean", "TW \u03c3", "TW CoV",
            "N_eff", "rho1", "Trend p", "Drift/sigma", "Allan ratio", "Stationary?",
        ]
        self._stats_table.setColumnCount(len(headers))
        self._stats_table.setHorizontalHeaderLabels(headers)
        self._stats_table.setRowCount(len(stats_list))
        self._stats_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)

        for i, s in enumerate(stats_list):
            display_unit = s.converted_unit or s.unit
            self._stats_table.setItem(i, 0, QTableWidgetItem(s.variable))
            self._stats_table.setItem(i, 1, QTableWidgetItem(display_unit))
            self._stats_table.setItem(i, 2, QTableWidgetItem(str(s.n_samples)))
            for j, val in enumerate([
                s.final_value, s.mean, s.median,
                s.min_val, s.max_val, s.sigma, s.three_sigma, s.p95,
                s.corrected_3sigma, s.difference, s.cov,
                s.half_range, s.tw_mean, s.tw_sigma, s.tw_cov,
                s.n_eff, s.autocorr_rho1, s.trend_pvalue,
                s.drift_ratio, s.allan_ratio,
                1.0 if s.stationarity_pass else 0.0,
            ], start=3):
                if j == 23:
                    item = QTableWidgetItem("PASS" if s.stationarity_pass else "FAIL")
                else:
                    item = QTableWidgetItem(f"{val:.6g}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                # Color-code CoV column
                if j in (13, 17):  # CoV and TW CoV columns
                    if val < 0.001:
                        item.setForeground(QColor(DARK_COLORS['green']))
                    elif val < 0.01:
                        item.setForeground(QColor(DARK_COLORS['yellow']))
                    else:
                        item.setForeground(QColor(DARK_COLORS['red']))
                if j == 23:
                    item.setForeground(
                        QColor(DARK_COLORS['green'] if s.stationarity_pass else DARK_COLORS['red'])
                    )
                self._stats_table.setItem(i, j, item)

    def _update_carry_table(self, stats_list: List[IterativeStats]):
        method = self._cmb_method.currentText()
        allow_override = self._chk_stationarity_override.isChecked()
        override_reason = self._edit_override_reason.text().strip()
        self._carry_table.setRowCount(len(stats_list))
        for i, s in enumerate(stats_list):
            carry = derive_carry_over_value(
                s,
                method,
                allow_stationarity_override=allow_override,
                override_reason=override_reason,
            )
            unit = s.converted_unit or s.unit
            self._carry_table.setItem(i, 0, QTableWidgetItem(s.variable))
            self._carry_table.setItem(i, 1, QTableWidgetItem(unit))
            self._carry_table.setItem(i, 2, QTableWidgetItem(carry["u_carry"]))
            self._carry_table.setItem(i, 3, QTableWidgetItem(carry["sigma_basis"]))
            self._carry_table.setItem(i, 4, QTableWidgetItem(str(max(1, int(round(s.n_eff - 1))))))
            self._carry_table.setItem(i, 5, QTableWidgetItem(carry["distribution"]))
            self._carry_table.setItem(i, 6, QTableWidgetItem(carry["combine_method"]))
            self._carry_table.setItem(i, 7, QTableWidgetItem(carry["formula"]))
            if carry["u_carry"] == "INVALID":
                for col in range(8):
                    item = self._carry_table.item(i, col)
                    if item is not None:
                        item.setForeground(QColor(DARK_COLORS['red']))

    def get_all_stats(self) -> Dict[str, List[IterativeStats]]:
        return self._all_stats

    def get_last_n(self) -> int:
        return self._spn_last_n.value()

    def get_method(self) -> str:
        return self._cmb_method.currentText()

    def get_stationarity_override_enabled(self) -> bool:
        return self._chk_stationarity_override.isChecked()

    def get_stationarity_override_reason(self) -> str:
        return self._edit_override_reason.text().strip()


# =============================================================================
# TAB 3: CHARTS
# =============================================================================

class ChartsTab(QWidget):
    """Time series, zoomed, histogram, QQ plots."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        self._cmb_var = QComboBox()
        self._cmb_var.setToolTip("Select which variable to plot.")
        self._cmb_var.currentIndexChanged.connect(self._update_chart)
        top.addWidget(QLabel("Variable:"))
        top.addWidget(self._cmb_var)

        self._cmb_chart = QComboBox()
        self._cmb_chart.setToolTip(
            "Full Time Series: all iterations/time-steps.\n"
            "Zoomed (Last N): only the window used for statistics.\n"
            "Peak Variable (Annotated): highlights max, min, mean, \u03c3 bands.\n"
            "Histogram: frequency distribution of the windowed data.\n"
            "QQ Plot: quantile-quantile plot vs. Normal distribution.")
        self._cmb_chart.addItems([
            "Full Time Series", "Zoomed (Last N)",
            "Peak Variable (Annotated)", "Histogram", "QQ Plot",
        ])
        self._cmb_chart.currentIndexChanged.connect(self._update_chart)
        top.addWidget(QLabel("Chart:"))
        top.addWidget(self._cmb_chart)

        self._btn_save = QPushButton("Save Plot...")
        self._btn_save.setToolTip("Save the current chart as a PNG or SVG file.")
        self._btn_save.clicked.connect(self._save_plot)
        top.addWidget(self._btn_save)

        self._btn_export = QPushButton("Export Figure Package...")
        self._btn_export.setToolTip(
            "Export PNG (300+600 DPI), SVG, PDF, and JSON\n"
            "to a folder for reports and presentations.")
        self._btn_export.clicked.connect(self._export_figure)
        top.addWidget(self._btn_export)
        top.addStretch()
        layout.addLayout(top)

        self._fig = Figure(figsize=(9, 5))
        self._canvas = FigureCanvas(self._fig)
        self._toolbar = NavigationToolbar(self._canvas, self)
        layout.addWidget(self._toolbar)
        layout.addWidget(self._canvas)

        self._fluent_data: Optional[FluentData] = None
        self._stats_list: List[IterativeStats] = []
        self._last_n: int = 1000

    def update_data(self, fd: FluentData, stats: List[IterativeStats], last_n: int):
        self._fluent_data = fd
        self._stats_list = stats
        self._last_n = last_n
        self._cmb_var.blockSignals(True)
        self._cmb_var.clear()
        for s in stats:
            lbl = f"{s.variable} ({s.converted_unit})" if s.converted_unit else s.variable
            self._cmb_var.addItem(lbl)
        self._cmb_var.blockSignals(False)
        if stats:
            self._update_chart()

    def _get_var_data(self):
        idx = self._cmb_var.currentIndex()
        if idx < 0 or not self._fluent_data or self._fluent_data.data is None:
            return None, None, None
        s = self._stats_list[idx]
        fd = self._fluent_data
        # Find column index
        var_idx = -1
        for i, h in enumerate(fd.headers):
            if h == s.variable:
                var_idx = i
                break
        if var_idx < 0:
            return None, None, None
        vals = fd.data[:, var_idx]
        times = fd.data[:, fd.time_col_idx] if fd.time_col_idx >= 0 else np.arange(len(vals))
        return vals, times, s

    def _update_chart(self):
        vals, times, stats = self._get_var_data()
        if vals is None:
            return
        chart_type = self._cmb_chart.currentText()
        self._fig.clear()
        ax = self._fig.add_subplot(111)

        convert = stats.converted_unit and stats.converted_unit != stats.unit
        if convert:
            if stats.unit == "K":
                display_vals = UNIT_CONVERSION['K_to_F'](vals)
            elif stats.unit == "Pa":
                display_vals = UNIT_CONVERSION['Pa_to_psia'](vals)
            else:
                display_vals = vals
        else:
            display_vals = vals

        unit_str = f" ({stats.converted_unit})" if stats.converted_unit else ""
        time_label = self._fluent_data.time_column if self._fluent_data else "Index"

        if chart_type == "Full Time Series":
            ax.plot(times, display_vals, color=DARK_COLORS['accent'],
                    lw=0.7, alpha=0.8)
            ax.axhline(stats.mean, color=DARK_COLORS['green'], ls='--', lw=1,
                       label=f'Mean = {stats.mean:.4g}')
            ax.axhline(stats.corrected_3sigma, color=DARK_COLORS['red'],
                       ls=':', lw=1, label=f'Corr. 3\u03c3 = {stats.corrected_3sigma:.4g}')
            ax.set_xlabel(time_label)
            ax.set_ylabel(f"{stats.variable}{unit_str}")
            ax.set_title(f"Time Series \u2014 {stats.variable}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        elif chart_type == "Zoomed (Last N)":
            start = max(0, len(display_vals) - self._last_n)
            ax.plot(times[start:], display_vals[start:],
                    color=DARK_COLORS['accent'], lw=0.8)
            ax.axhline(stats.mean, color=DARK_COLORS['green'], ls='--', lw=1,
                       label=f'Mean = {stats.mean:.4g}')
            ax.axhline(stats.corrected_3sigma, color=DARK_COLORS['red'],
                       ls=':', lw=1, label=f'Corr. 3\u03c3 = {stats.corrected_3sigma:.4g}')
            ax.set_xlabel(time_label)
            ax.set_ylabel(f"{stats.variable}{unit_str}")
            ax.set_title(f"Zoomed (Last {self._last_n}) \u2014 {stats.variable}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        elif chart_type == "Peak Variable (Annotated)":
            # Show full series with key annotations
            ax.plot(times, display_vals, color=DARK_COLORS['accent'],
                    lw=0.6, alpha=0.7, label='Data')
            # Mean line
            ax.axhline(stats.mean, color=DARK_COLORS['green'], ls='--', lw=1.5,
                       label=f'Mean = {stats.mean:.4g}')
            # ±σ band
            ax.axhspan(stats.mean - stats.sigma, stats.mean + stats.sigma,
                       alpha=0.1, color=DARK_COLORS['green'], label='\u00b1\u03c3 band')
            # Corrected 3σ line with annotation
            ax.axhline(stats.corrected_3sigma, color=DARK_COLORS['red'],
                       ls=':', lw=2, label=f'Corr. 3\u03c3 = {stats.corrected_3sigma:.4g}')
            # Final value marker
            ax.plot(times[-1], stats.final_value, 'o', color=DARK_COLORS['orange'],
                    markersize=8, zorder=5, label=f'Final = {stats.final_value:.4g}')
            # Difference annotation arrow
            if abs(stats.difference) > 0:
                mid_x = times[-1]
                ax.annotate(
                    f'\u0394 = {stats.difference:.4g}',
                    xy=(mid_x, stats.final_value),
                    xytext=(mid_x - (times[-1] - times[0]) * 0.15,
                            (stats.corrected_3sigma + stats.final_value) / 2),
                    fontsize=8, color=DARK_COLORS['orange'],
                    arrowprops=dict(arrowstyle='->', color=DARK_COLORS['orange'],
                                    lw=1.5),
                    ha='center')
            # ITTC half-range band
            ax.axhspan(stats.mean - stats.half_range, stats.mean + stats.half_range,
                       alpha=0.05, color=DARK_COLORS['yellow'],
                       label=f'ITTC U\u1d62 = {stats.half_range:.4g}')
            ax.set_xlabel(time_label)
            ax.set_ylabel(f"{stats.variable}{unit_str}")
            ax.set_title(f"Peak Variable Analysis \u2014 {stats.variable}")
            ax.legend(fontsize=6.5, loc='upper left')
            ax.grid(True, alpha=0.3)

        elif chart_type == "Histogram":
            start = max(0, len(display_vals) - self._last_n)
            window = display_vals[start:]
            valid = window[np.isfinite(window)]
            n_bins = min(max(int(np.sqrt(len(valid))), 5), 50)
            ax.hist(valid, bins=n_bins, density=True, alpha=0.7,
                    color=DARK_COLORS['accent'], edgecolor=DARK_COLORS['border'])
            ax.axvline(stats.mean, color=DARK_COLORS['green'], ls='--', lw=1.5,
                       label=f'Mean = {stats.mean:.4g}')
            ax.set_xlabel(f"{stats.variable}{unit_str}")
            ax.set_ylabel("Density")
            ax.set_title(f"Histogram \u2014 {stats.variable}")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        elif chart_type == "QQ Plot":
            start = max(0, len(display_vals) - self._last_n)
            window = display_vals[start:]
            valid = np.sort(window[np.isfinite(window)])
            n = len(valid)
            if n > 1:
                theoretical = norm.ppf((np.arange(1, n + 1) - 0.5) / n)
                ax.scatter(theoretical, valid, s=8, alpha=0.7,
                           color=DARK_COLORS['accent'], edgecolors='none')
                q25, q75 = np.percentile(valid, [25, 75])
                t25, t75 = norm.ppf(0.25), norm.ppf(0.75)
                if t75 - t25 > 0:
                    slope = (q75 - q25) / (t75 - t25)
                    intercept = q25 - slope * t25
                    x_line = np.array([theoretical[0], theoretical[-1]])
                    ax.plot(x_line, slope * x_line + intercept,
                            color=DARK_COLORS['red'], ls='--', lw=1.5)
            ax.set_xlabel("Theoretical Quantiles")
            ax.set_ylabel(f"Sample Quantiles{unit_str}")
            ax.set_title(f"QQ Plot \u2014 {stats.variable}")
            ax.grid(True, alpha=0.3)

        self._fig.tight_layout()
        self._canvas.draw()

    def _save_plot(self):
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        var_name = self._cmb_var.currentText().replace(" ", "_").replace("/", "_")
        var_name = re.sub(r'[^\w\-.]', '_', var_name)
        chart_type = self._cmb_chart.currentText().replace(" ", "_")
        chart_type = re.sub(r'[^\w\-.]', '_', chart_type)
        # Zoomed suffix per spec
        suffix = "_zoomed" if "Zoomed" in self._cmb_chart.currentText() else ""
        default_name = f"{var_name}_{chart_type}{suffix}_{now}.png"
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Plot", os.path.join(os.path.expanduser("~"), default_name),
            "PNG (*.png);;SVG (*.svg);;PDF (*.pdf)")
        if filepath:
            orig_props = []
            for ax in self._fig.get_axes():
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
            orig_fc = self._fig.get_facecolor()
            self._fig.set_facecolor('#ffffff')
            ext = os.path.splitext(filepath)[1].lower()
            dpi_val = 600 if ext == ".png" else 300
            try:
                self._fig.savefig(
                    filepath,
                    dpi=dpi_val,
                    bbox_inches='tight',
                    facecolor='white',
                    edgecolor='none',
                )
            finally:
                self._fig.set_facecolor(orig_fc)
                for ax, props in zip(self._fig.get_axes(), orig_props):
                    ax.set_facecolor(props['facecolor'])
                    if props['title_color']:
                        ax.title.set_color(props['title_color'])
                    ax.xaxis.label.set_color(props['xlabel_color'])
                    ax.yaxis.label.set_color(props['ylabel_color'])
                    ax.tick_params(axis='x', colors=PLOT_STYLE['xtick.color'])
                    ax.tick_params(axis='y', colors=PLOT_STYLE['ytick.color'])
                    for s_name, s_color in props['spine_colors'].items():
                        ax.spines[s_name].set_edgecolor(s_color)
            # Also save metadata sidecar
            meta = {
                "tool": APP_NAME, "version": APP_VERSION,
                "variable": self._cmb_var.currentText(),
                "chart_type": self._cmb_chart.currentText(),
                "generated": now,
            }
            meta_path = os.path.splitext(filepath)[0] + "_meta.json"
            try:
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass  # Non-critical

    def _export_figure(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Figure Package", os.path.expanduser("~"),
            "Figure Base Name (*)")
        if not filepath:
            return
        base = os.path.splitext(filepath)[0]
        try:
            export_figure_package(self._fig, base)
            QMessageBox.information(self, "Exported",
                                    f"Figure package exported to:\n{base}_*.png/svg/pdf")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))
        self._canvas.draw()

    def get_figure(self) -> Figure:
        return self._fig


# =============================================================================
# TAB 4: BATCH
# =============================================================================

class BatchTab(QWidget):
    """Cross-case batch processing, variable matching, combined CSV."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        self._btn_batch = QPushButton("Run Batch Processing")
        self._btn_batch.setToolTip(
            "Process all cases at once. Computes per-case statistics\n"
            "and cross-case comparison for each variable.\n"
            "Also writes per-case CSV summaries to each case folder.")
        self._btn_batch.setStyleSheet(
            f"QPushButton {{ background-color: {DARK_COLORS['accent']}; "
            f"color: {DARK_COLORS['bg']}; font-weight: bold; padding: 8px 24px; }}"
            f"QPushButton:hover {{ background-color: {DARK_COLORS['accent_hover']}; }}")
        top.addWidget(self._btn_batch)

        top.addWidget(QLabel("Cross-case method:"))
        self._cmb_cross = QComboBox()
        self._cmb_cross.setToolTip(
            "How to combine uncertainties across multiple cases:\n"
            "  Pooled Std Dev: pool all samples, compute single \u03c3.\n"
            "  RMS of Per-Case \u03c3: RMS-combine each case\u2019s \u03c3.\n"
            "  Both: compute both and show side-by-side.")
        self._cmb_cross.addItems(["Pooled Std Dev", "RMS of Per-Case \u03c3", "Both"])
        top.addWidget(self._cmb_cross)

        top.addWidget(QLabel("Variable keying:"))
        self._cmb_keying = QComboBox()
        self._cmb_keying.addItem(
            "Merge by Variable Name",
            "merge_by_name",
        )
        self._cmb_keying.addItem(
            "Separate by File + Variable",
            "separate_by_file",
        )
        self._cmb_keying.setToolTip(
            "Merge by Variable Name: combine same variable labels across files.\n"
            "Separate by File + Variable: keep each file-specific variable separate "
            "to avoid hidden overwrites when labels repeat."
        )
        top.addWidget(self._cmb_keying)

        self._btn_export_csv = QPushButton("Export Combined CSV...")
        self._btn_export_csv.setToolTip(
            "Export all batch results to a single CSV file\n"
            "with one row per case \u00d7 variable combination.")
        self._btn_export_csv.clicked.connect(self._export_csv)
        self._btn_export_csv.setEnabled(False)
        top.addWidget(self._btn_export_csv)
        top.addStretch()
        layout.addLayout(top)

        # Splitter: table on top, cross-case chart on bottom
        splitter = QSplitter(Qt.Vertical)

        self._batch_table = QTableWidget()
        self._batch_table.setAlternatingRowColors(True)
        splitter.addWidget(self._batch_table)

        # Cross-case comparison chart
        chart_widget = QWidget()
        chart_lay = QVBoxLayout(chart_widget)
        chart_lay.setContentsMargins(0, 0, 0, 0)
        chart_top = QHBoxLayout()
        self._cmb_batch_var = QComboBox()
        self._cmb_batch_var.currentIndexChanged.connect(self._update_batch_chart)
        chart_top.addWidget(QLabel("Cross-Case Variable:"))
        chart_top.addWidget(self._cmb_batch_var)
        self._cmb_batch_chart = QComboBox()
        self._cmb_batch_chart.addItems(["Box Plot", "Bar Chart (Final Values)",
                                         "Bar Chart (\u03c3 Comparison)"])
        self._cmb_batch_chart.currentIndexChanged.connect(self._update_batch_chart)
        chart_top.addWidget(self._cmb_batch_chart)
        chart_top.addStretch()
        chart_lay.addLayout(chart_top)
        self._batch_fig = Figure(figsize=(8, 4))
        self._batch_canvas = FigureCanvas(self._batch_fig)
        chart_lay.addWidget(self._batch_canvas)
        splitter.addWidget(chart_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

        self._guidance = GuidancePanel("Batch Processing")
        self._guidance.set_guidance(
            "Process all cases and all files at once. Cross-case statistics "
            "use the selected variable keying policy.", 'green')
        layout.addWidget(self._guidance)

        self._batch_results: List[Dict] = []
        self._cross_case_data: Dict[str, Dict[str, IterativeStats]] = {}
        self._cross_case_samples: Dict[str, Dict[str, np.ndarray]] = {}

    def run_batch(self, parsed: Dict[str, List[FluentData]],
                  last_n: int, convert: bool,
                  trend_alpha: float = 0.05,
                  drift_limit: float = 0.5,
                  allan_limit: float = 1.5):
        """Process all cases, compute per-case and cross-case stats."""
        key_policy = self._cmb_keying.currentData() or "merge_by_name"
        all_case_stats = {}
        all_case_samples: Dict[str, Dict[str, np.ndarray]] = {}
        csv_warnings: List[str] = []
        for case_name, fd_list in sorted(parsed.items()):
            case_stats = []
            case_sample_lists: Dict[str, List[np.ndarray]] = {}
            for fd in fd_list:
                if fd.data is None or fd.data.size == 0:
                    continue
                n_rows = fd.data.shape[0]
                start = max(0, n_rows - last_n)
                windowed = fd.data[start:]
                times = None
                if fd.is_transient and fd.time_col_idx >= 0:
                    times = windowed[:, fd.time_col_idx]
                for i, header in enumerate(fd.headers):
                    if i == fd.time_col_idx:
                        continue
                    vals = windowed[:, i]
                    valid_mask = np.isfinite(vals)
                    valid = vals[valid_mask]
                    if len(valid) < 2:
                        continue
                    unit = fd.units.get(header, "")
                    times_var = None
                    if times is not None and len(times) == len(vals):
                        times_var = times[valid_mask]
                    s = compute_iterative_stats(
                        valid, times=times_var, variable=header,
                        unit=unit, source_file=fd.file_base,
                        source_case=case_name, convert_units=convert,
                        trend_alpha=trend_alpha,
                        drift_limit=drift_limit,
                        allan_limit=allan_limit)
                    case_stats.append(s)

                    # Store actual samples for cross-case box plots.
                    display_valid = valid.copy()
                    if convert and unit == "K":
                        display_valid = UNIT_CONVERSION['K_to_F'](display_valid)
                    elif convert and unit == "Pa":
                        display_valid = UNIT_CONVERSION['Pa_to_psia'](display_valid)
                    cross_key = (
                        header if key_policy == "merge_by_name"
                        else f"{fd.file_base} :: {header}"
                    )
                    if cross_key not in case_sample_lists:
                        case_sample_lists[cross_key] = []
                    case_sample_lists[cross_key].append(display_valid)
            all_case_stats[case_name] = case_stats
            all_case_samples[case_name] = {
                var_name: np.concatenate(samples)
                for var_name, samples in case_sample_lists.items() if samples
            }
            if not case_stats:
                csv_warnings.append(
                    f"{case_name}: no valid data columns found — "
                    "check that files contain numeric data with at least 2 rows."
                )
            # Write per-case CSV
            if case_stats:
                case_folder = os.path.dirname(fd_list[0].filepath) if fd_list else ""
                if case_folder:
                    csv_path = os.path.join(case_folder, f"stats_{case_name}.csv")
                    try:
                        write_stats_csv(case_stats, csv_path)
                    except Exception as exc:
                        csv_warnings.append(f"{case_name}: {exc}")

        # Cross-case analysis: explicit keying policy to prevent silent overwrite.
        var_map: Dict[str, Dict[str, IterativeStats]] = {}
        var_samples: Dict[str, Dict[str, np.ndarray]] = {}
        collision_count = 0
        for case_name, stats_list in all_case_stats.items():
            for s in stats_list:
                cross_key = (
                    s.variable if key_policy == "merge_by_name"
                    else f"{s.source_file} :: {s.variable}"
                )
                case_key = (
                    case_name if key_policy == "merge_by_name"
                    else f"{case_name} :: {s.source_file}"
                )
                if cross_key not in var_map:
                    var_map[cross_key] = {}
                if case_key in var_map[cross_key]:
                    collision_count += 1
                    # Merge policy for collisions in "merge_by_name":
                    # keep the more conservative sigma.
                    if s.sigma >= var_map[cross_key][case_key].sigma:
                        var_map[cross_key][case_key] = s
                else:
                    var_map[cross_key][case_key] = s
                if (
                    case_name in all_case_samples
                    and cross_key in all_case_samples[case_name]
                ):
                    if cross_key not in var_samples:
                        var_samples[cross_key] = {}
                    var_samples[cross_key][case_key] = all_case_samples[case_name][cross_key]

        # Build batch results table
        self._batch_results = []
        rows = []
        cross_method = self._cmb_cross.currentText()
        for cross_key, case_dict in sorted(var_map.items()):
            finals = [s.final_value for s in case_dict.values()]
            sigmas = [s.sigma for s in case_dict.values()]
            half_ranges = [s.half_range for s in case_dict.values()]
            n_cases = len(case_dict)
            pooled_sigma = float(np.std(finals, ddof=1)) if n_cases > 1 else (
                sigmas[0] if sigmas else 0)
            # NOTE: This computes RMS (root-mean-square, divides by N), not RSS (root-sum-of-squares)
            rss_sigma = float(np.sqrt(np.sum(np.array(sigmas) ** 2) / n_cases))
            if cross_method == "Pooled Std Dev":
                selected_sigma = pooled_sigma
                selected_method = "Pooled Std Dev"
            elif cross_method == "RMS of Per-Case σ":
                selected_sigma = rss_sigma
                selected_method = "RMS of Per-Case σ"
            else:
                selected_sigma = max(pooled_sigma, rss_sigma)
                selected_method = "Both (conservative max)"
            first_stats = list(case_dict.values())[0]
            first_unit = first_stats.converted_unit or first_stats.unit
            rows.append({
                'variable_key': cross_key,
                'base_variable': first_stats.variable,
                'key_policy': key_policy,
                'unit': first_unit,
                'n_cases': n_cases,
                'mean_final': float(np.mean(finals)),
                'pooled_sigma': pooled_sigma,
                'rss_sigma': rss_sigma,
                'selected_sigma': selected_sigma,
                'selected_method': selected_method,
                'mean_half_range': float(np.mean(half_ranges)),
                'cases': list(case_dict.keys()),
            })
            self._batch_results.append(rows[-1])

        # Display
        headers = [
            "Cross-Case Key", "Base Variable", "Unit", "Cases", "Mean Final",
            "Pooled \u03c3", "RSS \u03c3", "Carry u (Selected)",
            "Selected Method", "Mean ITTC U_I",
        ]
        self._batch_table.setColumnCount(len(headers))
        self._batch_table.setHorizontalHeaderLabels(headers)
        self._batch_table.setRowCount(len(rows))
        self._batch_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents)
        for i, r in enumerate(rows):
            self._batch_table.setItem(i, 0, QTableWidgetItem(r['variable_key']))
            self._batch_table.setItem(i, 1, QTableWidgetItem(r['base_variable']))
            self._batch_table.setItem(i, 2, QTableWidgetItem(r['unit']))
            self._batch_table.setItem(i, 3, QTableWidgetItem(str(r['n_cases'])))
            for j, key in enumerate([
                'mean_final', 'pooled_sigma', 'rss_sigma', 'selected_sigma'
            ], start=4):
                item = QTableWidgetItem(f"{r[key]:.6g}")
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self._batch_table.setItem(i, j, item)
            self._batch_table.setItem(i, 8, QTableWidgetItem(r['selected_method']))
            item_hr = QTableWidgetItem(f"{r['mean_half_range']:.6g}")
            item_hr.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._batch_table.setItem(i, 9, item_hr)

        self._btn_export_csv.setEnabled(True)

        # Populate cross-case chart variable selector
        self._cross_case_data = var_map  # {cross_key: {case_key: IterativeStats}}
        self._cross_case_samples = var_samples  # {cross_key: {case_key: np.ndarray}}
        self._cmb_batch_var.blockSignals(True)
        self._cmb_batch_var.clear()
        for vn in sorted(var_map.keys()):
            self._cmb_batch_var.addItem(vn)
        self._cmb_batch_var.blockSignals(False)
        if var_map:
            self._update_batch_chart()

        if csv_warnings:
            self._guidance.set_guidance(
                f"Batch complete with {len(csv_warnings)} CSV export warning(s). "
                "Results are shown, but check write permissions/output paths.",
                'yellow'
            )
        else:
            policy_label = (
                "merge by variable name" if key_policy == "merge_by_name"
                else "separate by file + variable"
            )
            collision_msg = (
                f" {collision_count} duplicate key collision(s) were merged conservatively."
                if collision_count > 0 else ""
            )
            self._guidance.set_guidance(
                f"Batch complete: {len(rows)} variables across "
                f"{len(all_case_stats)} cases. Selected cross-case method: "
                f"{cross_method}. Keying policy: {policy_label}.{collision_msg}",
                'yellow' if collision_count > 0 else 'green')
        return all_case_stats

    def _export_csv(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Combined CSV", os.path.expanduser("~"),
            "CSV Files (*.csv);;All Files (*)")
        if not filepath:
            return
        try:
            with open(filepath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["Cross Key", "Base Variable", "Key Policy",
                                 "Unit", "N Cases", "Mean Final",
                                 "Pooled Sigma", "RSS Sigma", "Carry Selected",
                                 "Selected Method", "Mean ITTC U_I", "Cases"])
                for r in self._batch_results:
                    cases_str = "; ".join(r['cases'])
                    writer.writerow([
                        r['variable_key'], r['base_variable'],
                        r['key_policy'], r['unit'], r['n_cases'],
                        f"{r['mean_final']:.8g}", f"{r['pooled_sigma']:.8g}",
                        f"{r['rss_sigma']:.8g}", f"{r['selected_sigma']:.8g}",
                        r['selected_method'], f"{r['mean_half_range']:.8g}",
                        cases_str,
                    ])
            self._guidance.set_guidance(f"CSV exported: {filepath}", 'green')
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    @staticmethod
    def _truncate_case_name(name: str, max_len: int = 30) -> str:
        """Smart truncation that preserves prefix and suffix."""
        if len(name) <= max_len:
            return name
        return name[:12] + "..." + name[-12:]

    def _update_batch_chart(self):
        var_name = self._cmb_batch_var.currentText()
        if not var_name or var_name not in self._cross_case_data:
            return
        case_dict = self._cross_case_data[var_name]
        chart_type = self._cmb_batch_chart.currentText()
        cases = sorted(case_dict.keys())
        self._batch_fig.clear()
        ax = self._batch_fig.add_subplot(111)

        if chart_type == "Box Plot":
            # Collect actual sample windows per case for box plot.
            data_lists = []
            labels = []
            fallback_cases = []
            sample_map = self._cross_case_samples.get(var_name, {})
            for c in cases:
                vals = sample_map.get(c)
                if vals is None or len(vals) < 2:
                    s = case_dict[c]
                    vals = np.array([s.final_value, s.mean, s.mean + s.sigma,
                                     s.mean - s.sigma, s.min_val, s.max_val])
                    fallback_cases.append(c)
                data_lists.append(vals)
                labels.append(self._truncate_case_name(c))
            bp = ax.boxplot(data_lists, labels=labels, patch_artist=True,
                            boxprops=dict(facecolor=DARK_COLORS['bg_input'],
                                          edgecolor=DARK_COLORS['accent']),
                            medianprops=dict(color=DARK_COLORS['orange'], lw=2),
                            whiskerprops=dict(color=DARK_COLORS['fg_dim']),
                            capprops=dict(color=DARK_COLORS['fg_dim']))
            ax.set_ylabel(f"{var_name}")
            ax.set_title(f"Cross-Case Box Plot \u2014 {var_name}")
            ax.tick_params(axis='x', rotation=30)
            if fallback_cases:
                fallback_label = ", ".join(self._truncate_case_name(c) for c in fallback_cases[:3])
                if len(fallback_cases) > 3:
                    fallback_label += ", ..."
                ax.text(
                    0.01, 0.02,
                    f"Fallback synthetic samples used for: {fallback_label}",
                    transform=ax.transAxes,
                    fontsize=7,
                    color=DARK_COLORS['yellow'],
                    va='bottom',
                    ha='left',
                )

        elif "Final Values" in chart_type:
            finals = [case_dict[c].final_value for c in cases]
            sigmas = [case_dict[c].sigma for c in cases]
            x = np.arange(len(cases))
            ax.bar(x, finals, yerr=sigmas, color=DARK_COLORS['accent'],
                   edgecolor=DARK_COLORS['border'], alpha=0.8, capsize=4,
                   error_kw={'color': DARK_COLORS['orange'], 'lw': 1.5})
            ax.set_xticks(x)
            ax.set_xticklabels([self._truncate_case_name(c) for c in cases], rotation=30, ha='right')
            ax.set_ylabel(f"{var_name}")
            ax.set_title(f"Final Values \u00b1 \u03c3 \u2014 {var_name}")

        elif "\u03c3 Comparison" in chart_type:
            sigmas = [case_dict[c].sigma for c in cases]
            half_ranges = [case_dict[c].half_range for c in cases]
            x = np.arange(len(cases))
            w = 0.35
            ax.bar(x - w / 2, sigmas, w, label='\u03c3',
                   color=DARK_COLORS['accent'], edgecolor=DARK_COLORS['border'])
            ax.bar(x + w / 2, half_ranges, w, label='ITTC U\u1d62',
                   color=DARK_COLORS['orange'], edgecolor=DARK_COLORS['border'])
            ax.set_xticks(x)
            ax.set_xticklabels([self._truncate_case_name(c) for c in cases], rotation=30, ha='right')
            ax.set_ylabel("Uncertainty")
            ax.set_title(f"\u03c3 vs ITTC U\u1d62 \u2014 {var_name}")
            ax.legend(fontsize=7)

        ax.grid(True, alpha=0.3, axis='y')
        self._batch_fig.tight_layout()
        self._batch_canvas.draw()

    def get_batch_results(self) -> List[Dict]:
        return self._batch_results


# =============================================================================
# TAB 5: REPORT
# =============================================================================

class ReportTab(QWidget):
    """HTML report generation with live preview."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        grp = QGroupBox("Report Options")
        gl = QVBoxLayout(grp)
        self._chk_batch_only = QCheckBox("Batch-only mode (exclude per-file details)")
        self._chk_batch_only.setToolTip(
            "When checked, the report only includes batch-level\n"
            "cross-case summaries. Omits per-file detailed tables.")
        gl.addWidget(self._chk_batch_only)
        self._chk_embed_charts = QCheckBox("Embed charts in report (base64 PNG)")
        self._chk_embed_charts.setChecked(True)
        self._chk_embed_charts.setToolTip(
            "When checked, charts are embedded as base64 images\n"
            "in the HTML file (self-contained, larger file size).\n"
            "When unchecked, charts are referenced as external files.")
        gl.addWidget(self._chk_embed_charts)
        top.addWidget(grp)

        btn_col = QVBoxLayout()
        self._btn_preview = QPushButton("Generate Preview")
        self._btn_preview.setToolTip(
            "Render the HTML report in the preview pane below.\n"
            "Requires at least one case to be computed first.")
        self._btn_preview.setStyleSheet(
            f"QPushButton {{ background-color: {DARK_COLORS['bg_widget']}; "
            f"color: {DARK_COLORS['accent']}; font-weight: bold; padding: 8px 20px; }}")
        btn_col.addWidget(self._btn_preview)

        self._btn_export = QPushButton("Export HTML Report...")
        self._btn_export.setToolTip(
            "Save the report as a standalone HTML file.\n"
            "Open in any browser or print to PDF.")
        self._btn_export.setStyleSheet(
            f"QPushButton {{ background-color: {DARK_COLORS['accent']}; "
            f"color: {DARK_COLORS['bg']}; font-weight: bold; padding: 8px 20px; }}"
            f"QPushButton:hover {{ background-color: {DARK_COLORS['accent_hover']}; }}")
        btn_col.addWidget(self._btn_export)
        top.addLayout(btn_col)
        top.addStretch()
        layout.addLayout(top)

        # HTML preview pane
        self._preview = QTextEdit()
        self._preview.setReadOnly(True)
        self._preview.setPlaceholderText(
            "Click 'Generate Preview' after running analysis to see "
            "the HTML report preview here.")
        layout.addWidget(self._preview)

    def is_batch_only(self) -> bool:
        return self._chk_batch_only.isChecked()

    def is_embed_charts(self) -> bool:
        return self._chk_embed_charts.isChecked()

    def show_preview(self, html: str):
        """Display HTML report preview in the text pane."""
        self._preview.setHtml(html)


# =============================================================================
# TAB 6: REFERENCE
# =============================================================================

class ITUReferenceTab(QWidget):
    """ITTC, ASME V&V 20, CoV method, Fluent format guide."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        tabs = QTabWidget()
        tabs.addTab(self._build_ittc(), "ITTC Method")
        tabs.addTab(self._build_sigma(), "Sigma Method")
        tabs.addTab(self._build_fluent_guide(), "Fluent Format Guide")
        tabs.addTab(self._build_glossary(), "Glossary")
        layout.addWidget(tabs)

    def _make_text(self, html):
        te = QTextEdit()
        te.setReadOnly(True)
        te.setHtml(html)
        return te

    def _build_ittc(self):
        c = DARK_COLORS
        html = f"""<html><body style="background-color:{c['bg']};color:{c['fg']};
                    font-family:Segoe UI,sans-serif;font-size:12px;padding:12px;">
        <h2 style="color:{c['accent']};">ITTC 7.5-03-01-01 Iterative Uncertainty</h2>
        <h3>Half-Range Method</h3>
        <p>The ITTC recommends evaluating iterative convergence uncertainty as:</p>
        <p style="font-size:14px;color:{c['green']};">
            <b>U<sub>I</sub> = 0.5 \u00d7 (S<sub>max</sub> \u2212 S<sub>min</sub>)</b></p>
        <p>where S<sub>max</sub> and S<sub>min</sub> are the maximum and minimum
        values of the solution variable over the last N iterations/time-steps.</p>
        <h3>When to Use</h3>
        <ul>
        <li>Standard in marine/naval CFD verification</li>
        <li>Conservative (uses full range, not just \u03c3)</li>
        <li>Applicable when oscillations are present</li>
        <li>Does not assume a specific distribution</li>
        </ul>
        </body></html>"""
        return self._make_text(html)

    def _build_sigma(self):
        c = DARK_COLORS
        html = f"""<html><body style="background-color:{c['bg']};color:{c['fg']};
                    font-family:Segoe UI,sans-serif;font-size:12px;padding:12px;">
        <h2 style="color:{c['accent']};">Sigma-Based Method</h2>
        <h3>Standard Deviation Approach</h3>
        <p>Compute the standard deviation \u03c3 of the solution over the last N
        iterations/time-steps. The corrected 3\u03c3 value is:</p>
        <p style="font-size:14px;color:{c['green']};">
            <b>Corrected 3\u03c3 = min(S<sub>max</sub>, \u03bc + 3\u03c3)</b></p>
        <p>The iterative uncertainty is the difference between the corrected
        3\u03c3 and the final solution value.</p>
        <h3>Coefficient of Variation (CoV)</h3>
        <p>CoV = \u03c3 / |\u03bc| provides a normalized measure of iterative scatter.
        Values below 0.001 (0.1%) typically indicate good convergence.</p>
        <h3>Time-Weighted Statistics</h3>
        <p>For transient simulations with non-uniform \u0394t, time-weighted
        statistics use trapezoidal weights:</p>
        <p><b>w<sub>i</sub> = (\u0394t<sub>i-1</sub> + \u0394t<sub>i</sub>) / 2</b></p>
        </body></html>"""
        return self._make_text(html)

    def _build_fluent_guide(self):
        c = DARK_COLORS
        html = f"""<html><body style="background-color:{c['bg']};color:{c['fg']};
                    font-family:Segoe UI,sans-serif;font-size:12px;padding:12px;">
        <h2 style="color:{c['accent']};">Fluent .out File Formats</h2>
        <h3>Modern: Report Definition Format</h3>
        <p>Created by Report Definitions in Fluent. Structure:</p>
        <pre style="background:{c['bg_input']};padding:8px;border-radius:4px;font-size:11px;">
Line 1: (metadata — ignored)
Line 2: (metadata — ignored)
Line 3: "flow-time" "var-1-name" "var-2-name"
Line 4+: 0.001  340.5  101325.0
          0.002  341.2  101320.0
          ...</pre>
        <h3>Legacy: XY-Plot Format</h3>
        <p>Created by XY-plot file export. Scheme/Lisp parenthesized:</p>
        <pre style="background:{c['bg_input']};padding:8px;border-radius:4px;font-size:11px;">
((xy/key/label "Temperature")
 0  340.5
 1  341.2
 ...)
((xy/key/label "Pressure")
 0  101325.0
 1  101320.0
 ...)</pre>
        <h3>Auto-Detection</h3>
        <p>The tool auto-detects format by looking for parenthesized Scheme
        syntax (XY-plot) vs quoted headers (Report Definition).</p>
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
        <tr><td>U<sub>I</sub></td><td>Iterative convergence uncertainty (ITTC half-range)</td></tr>
        <tr><td>\u03c3</td><td>Standard deviation of solution over window</td></tr>
        <tr><td>CoV</td><td>Coefficient of Variation = \u03c3 / |\u03bc|</td></tr>
        <tr><td>Last N</td><td>Window of last N iterations/time-steps for analysis</td></tr>
        <tr><td>Corrected 3\u03c3</td><td>min(max, mean + 3\u03c3) — bounded estimate</td></tr>
        <tr><td>Half-Range</td><td>0.5 \u00d7 (max \u2212 min) — ITTC method</td></tr>
        <tr><td>Time-Weighted</td><td>Statistics weighted by \u0394t for non-uniform time steps</td></tr>
        </table>
        </body></html>"""
        return self._make_text(html)


# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_itu_html_report(all_stats: Dict[str, List[IterativeStats]],
                              batch_results: List[Dict],
                              project_meta: dict,
                              method: str = "Both",
                              chart_fig: Optional[Figure] = None,
                              batch_only: bool = False,
                              allow_stationarity_override: bool = False,
                              stationarity_override_reason: str = "") -> str:
    """Generate iterative uncertainty HTML report."""
    from html import escape as _esc_html

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    consequence = normalize_decision_consequence(
        project_meta.get('decision_consequence', 'Medium')
    )
    chart_html = ""
    if chart_fig:
        try:
            b64 = _figure_to_base64(chart_fig)
            chart_html = f'<img src="data:image/png;base64,{b64}" style="max-width:100%;"/>'
        except Exception:
            chart_html = "<p><i>Chart embedding failed.</i></p>"

    # Per-case stats tables
    case_tables = ""
    if not batch_only:
        for case_name, stats_list in sorted(all_stats.items()):
            if not stats_list:
                continue
            rows = ""
            for s in stats_list:
                u = s.converted_unit or s.unit
                station_txt = "PASS" if s.stationarity_pass else "FAIL"
                rows += f"""<tr>
                    <td>{_esc_html(s.variable)}</td><td>{_esc_html(u)}</td><td>{s.n_samples}</td>
                    <td>{s.n_eff:.2f}</td><td>{s.autocorr_rho1:.3f}</td>
                    <td>{s.final_value:.6g}</td><td>{s.mean:.6g}</td>
                    <td>{s.sigma:.6g}</td><td>{s.corrected_3sigma:.6g}</td>
                    <td>{s.difference:.6g}</td><td>{s.half_range:.6g}</td>
                    <td>{s.cov:.4g}</td><td>{station_txt}</td></tr>"""
            case_tables += f"""
            <h3>{case_name}</h3>
            <table><tr><th>Variable</th><th>Unit</th><th>N</th>
            <th>N_eff</th><th>rho1</th>
            <th>Final</th><th>Mean</th><th>\u03c3</th><th>Corr. 3\u03c3</th>
            <th>Diff</th><th>ITTC U<sub>I</sub></th><th>CoV</th><th>Stationary?</th></tr>
            {rows}</table>"""

    # Batch table
    batch_html = ""
    if batch_results:
        brows = ""
        for r in batch_results:
            brows += (
                f"<tr><td>{_esc_html(str(r['variable_key']))}</td><td>{_esc_html(str(r['base_variable']))}</td>"
                f"<td>{_esc_html(str(r['unit']))}</td><td>{r['n_cases']}</td>"
                f"<td>{r['mean_final']:.6g}</td><td>{r['pooled_sigma']:.6g}</td>"
                f"<td>{r['rss_sigma']:.6g}</td><td>{r['selected_sigma']:.6g}</td>"
                f"<td>{_esc_html(str(r['selected_method']))}</td><td>{r['mean_half_range']:.6g}</td>"
                f"<td>{_esc_html(str(r.get('key_policy', 'merge_by_name')))}</td></tr>"
            )
        batch_html = (
            "<table><tr><th>Cross-Case Key</th><th>Base Variable</th><th>Unit</th>"
            "<th>Cases</th><th>Mean Final</th><th>Pooled \u03c3</th>"
            "<th>RSS \u03c3</th><th>Carry u (Selected)</th>"
            "<th>Selected Method</th><th>Mean ITTC U<sub>I</sub></th>"
            "<th>Key Policy</th></tr>"
            f"{brows}</table>"
        )

    carry_rows = ""
    for case_name, stats_list in sorted(all_stats.items()):
        for s in stats_list:
            carry = derive_carry_over_value(
                s,
                method,
                allow_stationarity_override=allow_stationarity_override,
                override_reason=stationarity_override_reason,
            )
            unit = s.converted_unit or s.unit
            carry_rows += (
                f"<tr><td>{_esc_html(case_name)}</td><td>{_esc_html(s.variable)}</td><td>{_esc_html(unit)}</td>"
                f"<td>{_esc_html(str(carry['u_carry']))}</td>"
                f"<td>{_esc_html(str(carry['sigma_basis']))}</td>"
                f"<td>{max(1, int(round(s.n_eff - 1)))}</td>"
                f"<td>{_esc_html(str(carry['distribution']))}</td>"
                f"<td>{_esc_html(str(carry['combine_method']))}</td>"
                f"<td>{_esc_html(str(carry['formula']))}</td></tr>"
            )
    carry_html = (
        "<table><tr><th>Case</th><th>Variable</th><th>Unit</th>"
        "<th>Carry u</th><th>Sigma Basis</th><th>DOF</th>"
        "<th>Distribution for Aggregator</th>"
        "<th>Aggregator Analysis Mode</th><th>How Derived</th></tr>"
        f"{carry_rows}</table>"
        if carry_rows else "<p><i>No carry-over values available.</i></p>"
    )

    program_raw = str(project_meta.get('program', '\u2014'))
    analyst_raw = str(project_meta.get('analyst', '\u2014'))
    date_raw = str(project_meta.get('date', '\u2014'))
    notes_raw = str(project_meta.get('notes', '\u2014'))
    program = _esc_html(program_raw)
    analyst = _esc_html(analyst_raw)
    date = _esc_html(date_raw)
    notes = _esc_html(notes_raw)
    key_policy_summary_raw = (
        batch_results[0].get('key_policy', 'merge_by_name')
        if batch_results else "merge_by_name"
    )
    if key_policy_summary_raw == "separate_by_file":
        key_policy_summary = "separate_by_file (file + variable)"
    else:
        key_policy_summary = "merge_by_name (variable only)"

    all_rows = [s for stats in all_stats.values() for s in stats]
    has_stats = len(all_rows) > 0
    stationarity_ok = has_stats and all(s.stationarity_pass for s in all_rows)
    diagnostics_ok = has_stats and all((not s.autocorr_warning) and s.stationarity_pass for s in all_rows)
    evidence = {
        'inputs_documented': bool(program_raw.strip()) and bool(analyst_raw.strip()),
        'method_selected': True,
        'units_consistent': len({(s.converted_unit or s.unit) for s in all_rows if (s.converted_unit or s.unit)}) <= 1,
        'data_quality': has_stats and all(s.n_samples >= 10 for s in all_rows),
        'diagnostics_pass': diagnostics_ok,
        'independent_review': False,
        'conservative_bound': method in ("ITTC Half-Range", "Both"),
        'validation_plan': stationarity_ok,
    }

    decision_card_html = render_decision_card_html(
        title="Decision Card (Carry to Aggregator)",
        use_value="Use Carry u and Sigma Basis values from Section 5.",
        use_distribution="Use Distribution for Aggregator from Section 5.",
        use_combination=(
            "Set Aggregator analysis mode once using Section 5 recommendation "
            "(RSS unless a program rule requires Monte Carlo)."
        ),
        stop_checks=[
            "Any row with stationarity FAIL",
            "Any row with carry value marked INVALID",
            "Any override without documented rationale",
            "Any unit mismatch with Aggregator setup",
        ],
        notes="Stationarity failures are blocked by default to prevent misuse.",
    )
    credibility_html = render_credibility_html(consequence, evidence)
    glossary_html = render_vvuq_glossary_html()
    conformity_html = render_conformity_template_html(
        metric_name="Iterative uncertainty carry-over value",
        metric_value="See Section 5",
        consequence=consequence,
    )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Iterative Uncertainty Report</title>
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

<h1>Iterative Uncertainty Report</h1>

<h2>1. Project Information</h2>
<table class="meta-table">
<tr><td><b>Program / Project:</b></td><td>{program}</td></tr>
<tr><td><b>Analyst:</b></td><td>{analyst}</td></tr>
<tr><td><b>Date:</b></td><td>{date}</td></tr>
<tr><td><b>Generated:</b></td><td>{now}</td></tr>
<tr><td><b>Tool:</b></td><td>{APP_NAME} v{APP_VERSION}</td></tr>
<tr><td><b>Method:</b></td><td>{method}</td></tr>
<tr><td><b>Decision Consequence:</b></td><td>{consequence}</td></tr>
<tr><td><b>Notes:</b></td><td>{notes}</td></tr>
</table>

<h2>2. Executive Summary</h2>
<p>Iterative convergence uncertainty analysis of {len(all_stats)} case(s) using
{method} method. Statistics computed from the last N iterations/time-steps.
Carry-over values for the Aggregator are listed in Section 5 as final values to transfer.</p>

<h2>3. Per-Case Statistics</h2>
{case_tables if case_tables else '<p><i>Batch-only mode selected.</i></p>'}

<h2>4. Cross-Case Batch Results</h2>
{batch_html if batch_html else '<p><i>No batch results.</i></p>'}

<h2>5. Carry-Over Summary for Aggregator</h2>
{carry_html}

<h2>6. Charts</h2>
{chart_html if chart_html else '<p><i>No charts embedded.</i></p>'}

<h2>7. Method Comparison</h2>
<table>
<tr><th>Method</th><th>Formula</th><th>When to Use</th></tr>
<tr><td>ITTC Half-Range</td><td>U<sub>I</sub> = 0.5 \u00d7 (max \u2212 min)</td>
    <td>Conservative; standard in naval CFD</td></tr>
<tr><td>Sigma-Based</td><td>\u03c3 from last N; Corr. 3\u03c3 = min(max, \u03bc+3\u03c3)</td>
    <td>Statistical; assumes approximate normality</td></tr>
</table>

<h2>8. Unit Conversion Notes</h2>
<ul>
<li>Temperature: K \u2192 \u00b0F: T<sub>\u00b0F</sub> = (T<sub>K</sub> \u2212 273.15) \u00d7 9/5 + 32</li>
<li>Pressure: Pa \u2192 psia: P<sub>psia</sub> = P<sub>Pa</sub> / 6894.757</li>
<li>Conversion applied automatically when detected; disable in Import tab options</li>
</ul>

<h2>9. Methodology</h2>
<ul>
<li>ITTC 7.5-03-01-01 \u2014 Iterative convergence uncertainty</li>
<li>ASME V&V 20-2009 (R2021) Section 5</li>
<li>Time-weighted statistics for non-uniform \u0394t (trapezoidal weights)</li>
<li>Auto-detection of Fluent .out format (Report Definition vs XY-plot)</li>
<li>Autocorrelation-aware effective sample size (N_eff) for DOF guidance</li>
<li>Stationarity gate: trend test + drift split + Allan ratio indicator</li>
<li>Cross-case key policy: {key_policy_summary}</li>
</ul>

<h3>How Stationarity Is Checked</h3>
<p>Before trusting any statistics from your iteration data, the tool checks whether the data is &ldquo;stationary&rdquo;
(i.e., the average value is not drifting over time). Three independent checks are performed:</p>
<ol>
<li><b>Trend Test:</b> Fits a straight line through your data. If the slope is statistically significant
(p-value &lt; 0.05), your data has a trend and is NOT stationary. <i>Think of it as asking: &ldquo;Is the value
consistently going up or down?&rdquo;</i></li>
<li><b>Drift Ratio:</b> Splits your data in half and compares the average of the first half to the second half.
If the difference exceeds 0.5&times; the standard deviation, there is drift. <i>Think of it as asking:
&ldquo;Did the average shift between the first half and second half?&rdquo;</i></li>
<li><b>Allan Deviation Ratio:</b> Compares short-term noise to long-term noise. If long-term noise is much
larger (ratio &gt; 1.5), there is low-frequency drift the other tests might miss. <i>Think of it as asking:
&ldquo;Is there a slow wobble hiding under the fast noise?&rdquo;</i></li>
</ol>
<p>If ANY of these checks fails, the carry-over value is flagged as invalid. You must either extend the analysis
window (use more iterations from the converged region) or provide a documented rationale for overriding.</p>

<h2>10. Assumptions & Limitations</h2>
<ul>
<li>Stationarity is explicitly checked; failed traces are blocked by default</li>
<li>Window size (Last N) should capture representative oscillations</li>
<li>Unit detection is heuristic (median-value with physical range validation; velocity variables are excluded)</li>
<li>XY-plot parser may fail on non-standard Scheme formatting</li>
</ul>

{decision_card_html}

{credibility_html}

{glossary_html}

{conformity_html}

<h2>Reviewer Checklist</h2>
<ul class="checklist">
<li>Convergence achieved before analysis window?</li>
<li>Window size adequate to capture oscillation cycles?</li>
<li>Stationarity checks passed (or override rationale documented)?</li>
<li>Unit conversions verified against Fluent setup?</li>
<li>Iterative uncertainty is small relative to discretization error?</li>
<li>Values carried to VVUQ Aggregator as u_num contribution?</li>
</ul>

</body></html>"""
    return html


# =============================================================================
# MAIN WINDOW
# =============================================================================

class IterativeUncertaintyWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(get_dark_stylesheet())

        font = QFont()
        for family in FONT_FAMILIES:
            if QFontDatabase.hasFamily(family):
                font.setFamily(family)
                break
        font.setPointSize(10)
        QApplication.instance().setFont(font)

        self._project_path: str = ""
        self._project_name: str = "Untitled"
        self._unsaved_changes: bool = False

        central = QWidget()
        cl = QVBoxLayout(central)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(0)
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

        self._project_detail_frame = QFrame()
        dl = QGridLayout(self._project_detail_frame)
        dl.setContentsMargins(4, 4, 4, 4)
        dl.setSpacing(6)
        lbl_style = f"color: {DARK_COLORS['fg_dim']}; font-size: 11px;"
        val_style = (
            f"background-color: {DARK_COLORS['bg_input']}; "
            f"color: {DARK_COLORS['fg']}; border: 1px solid {DARK_COLORS['border']}; "
            f"border-radius: 3px; padding: 3px 6px; font-size: 11px;")

        dl.addWidget(self._make_label("Program / Project:", lbl_style), 0, 0)
        self._edit_program = QLineEdit()
        self._edit_program.setStyleSheet(val_style)
        self._edit_program.setPlaceholderText("e.g., Turbine CFD Campaign")
        self._edit_program.setToolTip("Name of the program, project, or test campaign.")
        self._edit_program.textChanged.connect(self._mark_unsaved)
        dl.addWidget(self._edit_program, 0, 1)

        dl.addWidget(self._make_label("Analyst:", lbl_style), 0, 2)
        self._edit_analyst = QLineEdit()
        self._edit_analyst.setStyleSheet(val_style)
        self._edit_analyst.setToolTip("Name of the analyst performing this assessment.")
        self._edit_analyst.textChanged.connect(self._mark_unsaved)
        dl.addWidget(self._edit_analyst, 0, 3)

        dl.addWidget(self._make_label("Date:", lbl_style), 0, 4)
        self._edit_date = QLineEdit()
        self._edit_date.setStyleSheet(val_style)
        self._edit_date.setToolTip("Date of this analysis (auto-filled, editable).")
        self._edit_date.setText(datetime.datetime.now().strftime("%Y-%m-%d"))
        self._edit_date.textChanged.connect(self._mark_unsaved)
        dl.addWidget(self._edit_date, 0, 5)

        dl.addWidget(self._make_label("Notes:", lbl_style), 1, 0)
        self._edit_notes = QTextEdit()
        self._edit_notes.setStyleSheet(f"QTextEdit {{ {val_style} }}")
        self._edit_notes.setPlaceholderText("Analysis notes...")
        self._edit_notes.setMaximumHeight(80)
        self._edit_notes.textChanged.connect(self._mark_unsaved)
        dl.addWidget(self._edit_notes, 1, 1, 1, 5)

        dl.addWidget(self._make_label("Decision Consequence:", lbl_style), 2, 0)
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
        dl.addWidget(self._cmb_consequence, 2, 1)

        dl.setColumnStretch(1, 3)
        dl.setColumnStretch(3, 2)
        dl.setColumnStretch(5, 2)

        bar_layout.addWidget(self._project_detail_frame)
        self._project_detail_frame.setVisible(False)
        self._project_info_visible = False
        cl.addWidget(self._project_bar)

        # Tabs
        self._tabs = QTabWidget()
        cl.addWidget(self._tabs)

        self._tab_import = DataImportTab()
        self._tab_analysis = AnalysisTab()
        self._tab_charts = ChartsTab()
        self._tab_batch = BatchTab()
        self._tab_report = ReportTab()
        self._tab_ref = ITUReferenceTab()

        self._tabs.addTab(self._tab_import, "\U0001f4c2 Data Import")
        self._tabs.addTab(self._tab_analysis, "\U0001f4ca Analysis")
        self._tabs.addTab(self._tab_charts, "\U0001f4c8 Charts")
        self._tabs.addTab(self._tab_batch, "\U0001f504 Batch")
        self._tabs.addTab(self._tab_report, "\U0001f4cb Report")
        self._tabs.addTab(self._tab_ref, "\U0001f4d6 Reference")

        # Wire signals
        self._tab_analysis._btn_compute.clicked.connect(self._run_analysis)
        self._tab_analysis._cmb_case.currentTextChanged.connect(
            lambda c: self._tab_analysis.update_file_list(
                self._tab_import.get_parsed_data(), c))
        self._tab_batch._btn_batch.clicked.connect(self._run_batch)
        self._tab_report._btn_preview.clicked.connect(self._preview_report)
        self._tab_report._btn_export.clicked.connect(self._export_report)

        # Menu
        self._build_menu_bar()

        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        v_label = QLabel(f"{APP_NAME} v{APP_VERSION} ({APP_DATE})")
        v_label.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; font-size: 11px; padding-right: 8px;")
        self._status_bar.addPermanentWidget(v_label)
        self._status_bar.showMessage("Ready.", 5000)

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

    def _run_analysis(self):
        parsed = self._tab_import.get_parsed_data()
        if not parsed:
            QMessageBox.information(self, "No Data", "Scan for .out files first.")
            return
        convert = self._tab_import.is_convert_enabled()
        self._tab_analysis.populate_cases(parsed)
        stats = self._tab_analysis.compute(parsed, convert)
        if stats:
            case = self._tab_analysis._cmb_case.currentText()
            file_idx = self._tab_analysis._cmb_file.currentIndex()
            if case in parsed and 0 <= file_idx < len(parsed[case]):
                fd = parsed[case][file_idx]
                self._tab_charts.update_data(
                    fd, stats, self._tab_analysis.get_last_n())
        self._tabs.setCurrentWidget(self._tab_analysis)
        self._status_bar.showMessage(
            "Analysis complete. Carry-over values are ready in the Analysis tab.", 5000
        )

    def _run_batch(self):
        parsed = self._tab_import.get_parsed_data()
        if not parsed:
            QMessageBox.information(self, "No Data", "Scan for .out files first.")
            return
        convert = self._tab_import.is_convert_enabled()
        last_n = self._tab_analysis.get_last_n()
        all_case_stats = self._tab_batch.run_batch(
            parsed, last_n, convert,
            trend_alpha=self._tab_analysis._spn_trend_alpha.value(),
            drift_limit=self._tab_analysis._spn_drift_limit.value(),
            allan_limit=self._tab_analysis._spn_allan_limit.value())
        if all_case_stats:
            self._tab_analysis._all_stats = all_case_stats
        self._status_bar.showMessage("Batch processing complete.", 5000)

    def _generate_report_html(self) -> Optional[str]:
        """Generate report HTML, returning None if no data."""
        all_stats = self._tab_analysis.get_all_stats()
        batch_results = self._tab_batch.get_batch_results()
        if not all_stats and not batch_results:
            return None
        chart_fig = None
        if self._tab_report.is_embed_charts():
            chart_fig = self._tab_charts.get_figure()
        return generate_itu_html_report(
            all_stats, batch_results, self.get_project_metadata(),
            method=self._tab_analysis.get_method(),
            chart_fig=chart_fig,
            batch_only=self._tab_report.is_batch_only(),
            allow_stationarity_override=self._tab_analysis.get_stationarity_override_enabled(),
            stationarity_override_reason=self._tab_analysis.get_stationarity_override_reason(),
        )

    def _preview_report(self):
        html = self._generate_report_html()
        if html is None:
            QMessageBox.information(self, "No Results",
                                    "Run analysis or batch first.")
            return
        self._tab_report.show_preview(html)
        self._status_bar.showMessage("Report preview generated.", 3000)

    def _export_report(self):
        html = self._generate_report_html()
        if html is None:
            QMessageBox.information(self, "No Results",
                                    "Run analysis or batch first.")
            return
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export HTML Report", os.path.expanduser("~"),
            "HTML Files (*.html);;All Files (*)")
        if not filepath:
            return
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html)
            self._status_bar.showMessage(f"Report exported: {filepath}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

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

        file_menu.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.setShortcut("Ctrl+Q")
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        analysis_menu = mb.addMenu("Analysis")
        act_recompute = QAction("Recompute Analysis", self)
        act_recompute.setShortcut("Ctrl+R")
        act_recompute.triggered.connect(self._run_analysis)
        analysis_menu.addAction(act_recompute)

        act_batch = QAction("Run Batch Processing", self)
        act_batch.setShortcut("Ctrl+M")
        act_batch.triggered.connect(self._run_batch)
        analysis_menu.addAction(act_batch)

        examples_menu = mb.addMenu("Examples")
        act_example = QAction("Load Built-in Example Cases", self)
        act_example.triggered.connect(self._load_example_data)
        examples_menu.addAction(act_example)

        act_example_ns = QAction("Example 4: Non-Stationary (Trending)", self)
        act_example_ns.triggered.connect(self._load_example_nonstationary_data)
        examples_menu.addAction(act_example_ns)

        help_menu = mb.addMenu("Help")
        act_about = QAction("About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

    def _new_project(self):
        self._project_path = ""
        self._project_name = "Untitled"
        self._lbl_project_name.setText("Untitled")
        self._edit_program.clear()
        self._edit_analyst.clear()
        self._edit_date.setText(datetime.datetime.now().strftime("%Y-%m-%d"))
        self._edit_notes.clear()
        self._cmb_consequence.setCurrentText("Medium")
        self._unsaved_changes = False

    def _load_example_data(self):
        self._tab_import._load_example_dataset()
        parsed = self._tab_import.get_parsed_data()
        if parsed:
            self._tab_analysis.populate_cases(parsed)
            first_case = next(iter(sorted(parsed.keys())))
            self._tab_analysis.update_file_list(parsed, first_case)
            self._tab_analysis._cmb_case.setCurrentText(first_case)
        self._tabs.setCurrentWidget(self._tab_import)
        self._status_bar.showMessage(
            "Built-in examples loaded. Press Ctrl+R to compute.", 5000
        )

    def _load_example_nonstationary_data(self):
        self._tab_import._load_example_nonstationary()
        parsed = self._tab_import.get_parsed_data()
        if parsed:
            self._tab_analysis.populate_cases(parsed)
            first_case = next(iter(sorted(parsed.keys())))
            self._tab_analysis.update_file_list(parsed, first_case)
            self._tab_analysis._cmb_case.setCurrentText(first_case)
        self._tabs.setCurrentWidget(self._tab_import)
        self._status_bar.showMessage(
            "Non-stationary example loaded. Press Ctrl+R to compute and "
            "observe the stationarity gate.", 5000
        )

    def _save_project(self):
        if self._project_path:
            self._do_save(self._project_path)
        else:
            self._save_project_as()

    def _save_project_as(self):
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Project", os.path.expanduser("~"),
            "Iterative Uncertainty Project (*.itu);;All Files (*)")
        if filepath:
            if not filepath.endswith('.itu'):
                filepath += '.itu'
            self._do_save(filepath)

    def _do_save(self, filepath):
        try:
            state = {
                'tool': APP_NAME,
                'version': APP_VERSION,
                'saved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'project_metadata': self.get_project_metadata(),
                'root_folder': self._tab_import.get_root_folder(),
                'last_n': self._tab_analysis.get_last_n(),
                'method': self._tab_analysis.get_method(),
                'convert_units': self._tab_import.is_convert_enabled(),
            }
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            self._project_path = filepath
            self._project_name = os.path.splitext(os.path.basename(filepath))[0]
            self._lbl_project_name.setText(self._project_name)
            self._unsaved_changes = False
            self._status_bar.showMessage(f"Saved: {filepath}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _open_project(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Open Project", os.path.expanduser("~"),
            "Iterative Uncertainty Project (*.itu);;All Files (*)")
        if not filepath:
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
            self.set_project_metadata(state.get('project_metadata', {}))
            root = state.get('root_folder', '')
            if root and os.path.isdir(root):
                self._tab_import._root_folder = root
                self._tab_import._edit_folder.setText(root)
            self._tab_analysis._spn_last_n.setValue(state.get('last_n', 1000))
            method = state.get('method', 'Both')
            idx = self._tab_analysis._cmb_method.findText(method)
            if idx >= 0:
                self._tab_analysis._cmb_method.setCurrentIndex(idx)
            self._project_path = filepath
            self._project_name = os.path.splitext(os.path.basename(filepath))[0]
            self._lbl_project_name.setText(self._project_name)
            self._unsaved_changes = False
            self._status_bar.showMessage(f"Opened: {filepath}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Open Error", str(e))

    def _show_about(self):
        QMessageBox.about(self, "About",
            f"<h3>{APP_NAME} v{APP_VERSION}</h3>"
            f"<p>Date: {APP_DATE}</p>"
            f"<p>CFD iterative convergence uncertainty tool.</p>"
            f"<p>Supports Fluent .out files (Report Definition + XY-plot).</p>"
            f"<p>Methods: ITTC half-range, sigma-based, time-weighted.</p>"
            f"<p>Standards: ITTC 7.5-03-01-01, ASME V&V 20, JCGM 100</p>")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = IterativeUncertaintyWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
