#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VVUQ Uncertainty Aggregator v1.2
CFD Validation Uncertainty Tool per ASME V&V 20 Framework

Single-file PySide6 GUI application for computing CFD model validation
uncertainty per the ASME V&V 20 framework with RSS and Monte Carlo methods.

Standards References:
    - ASME V&V 20-2009 (R2021)
    - JCGM 100:2008 (GUM)
    - JCGM 101:2008 (GUM Supplement 1)
    - ASME PTC 19.1-2018
    - AIAA G-077-1998

Copyright (c) 2026. All rights reserved.
"""

# =============================================================================
# SECTION 0: DEPENDENCY CHECK & INSTALLATION HELPER
# =============================================================================
import sys
import subprocess
import importlib

APP_VERSION = "1.2.0"
APP_NAME = "VVUQ Uncertainty Aggregator"
APP_DATE = "2026-02-23"

REQUIRED_PACKAGES = {
    'PySide6': 'PySide6',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'matplotlib': 'matplotlib',
    'openpyxl': 'openpyxl',
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
        print(f"\nOr install from requirements.txt:")
        print(f"  pip install -r requirements.txt")

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

    # Generate requirements.txt if it doesn't exist
    import os
    req_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
    if not os.path.exists(req_path):
        try:
            with open(req_path, 'w') as f:
                f.write("# VVUQ Uncertainty Aggregator v1.2 - Dependencies\n")
                f.write("# Install with: pip install -r requirements.txt\n")
                f.write("PySide6>=6.5.0\n")
                f.write("numpy>=1.24.0\n")
                f.write("scipy>=1.10.0\n")
                f.write("matplotlib>=3.7.0\n")
                f.write("openpyxl>=3.1.0\n")
        except OSError:
            pass  # Read-only installation — skip requirements.txt generation

if __name__ == '__main__':
    check_and_install_dependencies()

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
import os
import json
import io
import copy
import itertools
import threading
import datetime
import base64
import textwrap
import warnings
import traceback
import tempfile
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
from scipy import stats
from scipy.stats import (
    norm, t as t_dist, nct, chi2, binom, uniform, triang,
    shapiro, kstest, anderson, skew, kurtosis,
    lognorm, logistic, laplace, weibull_min, expon, goodness_of_fit
)
# ═══════════════════════════════════════════════════════════════════════════
# SECTION: SHARED VVUQ REPORT COMPONENTS
# Keep shared data (constants, terms, renderer) consistent across all four
# VVUQ tools (Uncertainty Aggregator, GCI Calculator, Iterative Uncertainty,
# Statistical Analyzer).
# Import line and data definitions below must match across all four files.
# Last synchronized: 2026-02-23
# ═══════════════════════════════════════════════════════════════════════════
from html import escape as _html_esc

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
        f"<p><strong>Use this combination method:</strong> {_html_esc(use_combination)}</p>"
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

# Force matplotlib to use PySide6 (must be set BEFORE importing matplotlib.backends)
os.environ["QT_API"] = "pyside6"
if "MPLCONFIGDIR" not in os.environ:
    try:
        mpl_cache_dir = os.path.join(tempfile.gettempdir(), "vv20_validation_tool_mplconfig")
        os.makedirs(mpl_cache_dir, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = mpl_cache_dir
    except OSError:
        pass

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QFormLayout, QGroupBox, QLabel, QPushButton, QLineEdit,
    QTextEdit, QPlainTextEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QRadioButton, QButtonGroup, QTableWidget, QTableWidgetItem,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QSplitter, QScrollArea,
    QFileDialog, QMessageBox, QProgressBar, QStatusBar, QMenuBar, QMenu,
    QToolBar, QDialog, QDialogButtonBox, QFrame, QSizePolicy, QToolTip,
    QAbstractItemView, QStackedWidget
)
from PySide6.QtCore import (
    Qt, QTimer, Signal, Slot, QThread, QSize, QSettings
)
from PySide6.QtGui import (
    QFont, QColor, QPalette, QAction, QIcon, QKeySequence, QFontDatabase
)

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

warnings.filterwarnings('ignore', category=RuntimeWarning)

# =============================================================================
# SECTION 2: CONSTANTS & CONFIGURATION
# =============================================================================

# Proprietary / Export Control Header (edit these for your organization)
COMPANY_NAME = "[Company Name]"
TEAM_NAME = "[Team / Department Name]"
PROPRIETARY_NOTICE = (
    f"PROPRIETARY NOTICE: This document contains proprietary and confidential "
    f"information of {COMPANY_NAME}. It is furnished under agreement and shall "
    f"not be duplicated, used, or disclosed in whole or in part for any purpose "
    f"without the prior written consent of {COMPANY_NAME}."
)
EXPORT_CONTROL_NOTICE = (
    "EXPORT CONTROL: This document may contain technical data subject to "
    "export control under the International Traffic in Arms Regulations (ITAR) "
    "(22 CFR Parts 120-130) or the Export Administration Regulations (EAR) "
    "(15 CFR Parts 730-774). Transfer of this data to a foreign person, "
    "whether in the United States or abroad, without proper authorization "
    "from the U.S. Government may be a violation of federal law."
)

# Units available in the tool
UNIT_OPTIONS = ["°F", "°C", "psig", "psia", "lb/min", "lb/s"]
DEFAULT_UNIT = "°F"

# V&V 20 uncertainty categories
UNCERTAINTY_CATEGORIES = ["Numerical (u_num)", "Input/BC (u_input)", "Experimental (u_D)"]
CATEGORY_KEYS = ["u_num", "u_input", "u_D"]

# Distribution options
DISTRIBUTION_NAMES = [
    "Normal",
    "Uniform",
    "Triangular",
    "Lognormal",
    "Lognormal (σ=0.5)",
    "Logistic",
    "Weibull",
    "Laplace",
    "Student-t (df=5)",
    "Student-t (df=10)",
    "Exponential",
    "Custom/Empirical (Bootstrap)",
]

# Sigma basis options
SIGMA_BASIS_OPTIONS = [
    "Confirmed 1σ",
    "Assumed 1σ (unverified)",
    "2σ (95%)",
    "3σ (99.7%)",
    "Bounding (min/max)",
]

# Input type options
INPUT_TYPE_OPTIONS = [
    "Tabular Data",
    "Sigma Value Only",
    "Tolerance/Expanded Value",
    "RSS of Sub-Components",
    "CFD Sensitivity Run",
]

# Coverage / confidence presets
COVERAGE_OPTIONS = [0.90, 0.95, 0.99]
CONFIDENCE_OPTIONS = [0.90, 0.95, 0.99]

# K-factor method options
K_METHOD_VV20 = "ASME V&V 20 Default (k=2)"
K_METHOD_WS = "GUM Welch-Satterthwaite (computed k)"
K_METHOD_TOLERANCE = "One-Sided Tolerance Factor (non-central t)"
K_METHOD_MANUAL = "Manual k Entry"

# Font preference list (cross-platform safe)
FONT_FAMILIES = ["Segoe UI", "DejaVu Sans", "Liberation Sans", "Noto Sans",
                  "Ubuntu", "Helvetica", "Arial", "sans-serif"]

# Dark mode color palette
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

# Matplotlib dark theme for plots
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

# Named style profiles for export dialogs
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


# =============================================================================
# SECTION 3: AUDIT LOG
# =============================================================================
class AuditLog:
    """Timestamped audit trail of all user actions and computations."""

    def __init__(self):
        self.entries: List[Dict[str, str]] = []
        self.log("SESSION_START", f"{APP_NAME} v{APP_VERSION} started")

    def log(self, action_type: str, description: str, details: str = ""):
        entry = {
            'timestamp': datetime.datetime.now().isoformat(),
            'action': action_type,
            'description': description,
            'details': details,
        }
        self.entries.append(entry)

    def log_override(self, field: str, recommended: str, chosen: str, reason: str = ""):
        self.log("USER_OVERRIDE",
                 f"Override on '{field}': recommended='{recommended}', chosen='{chosen}'",
                 reason)

    def log_assumption(self, assumption: str, source: str = ""):
        self.log("ASSUMPTION", assumption, f"Source: {source}")

    def log_computation(self, calc_type: str, details: str):
        self.log("COMPUTATION", calc_type, details)

    def log_data_load(self, source: str, details: str):
        self.log("DATA_LOAD", f"Data loaded from {source}", details)

    def log_warning(self, warning: str):
        self.log("WARNING", warning)

    def export_text(self) -> str:
        lines = [
            f"{'='*70}",
            f"  {APP_NAME} v{APP_VERSION} — Audit Log",
            f"  Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"{'='*70}\n",
        ]
        for e in self.entries:
            ts = e['timestamp'][:19].replace('T', ' ')
            lines.append(f"[{ts}] [{e['action']}] {e['description']}")
            if e['details']:
                for dl in e['details'].split('\n'):
                    lines.append(f"    {dl}")
        return '\n'.join(lines)

    def to_dict(self) -> List[Dict]:
        return self.entries

    def from_dict(self, data: List[Dict]):
        self.entries = data


# Global audit log instance
audit_log = AuditLog()


# =============================================================================
# SECTION 4: STATISTICAL UTILITY FUNCTIONS
# =============================================================================

def one_sided_tolerance_k(n, coverage=0.95, confidence=0.95):
    """
    One-sided normal tolerance factor using non-central t-distribution.

    Ref: Krishnamoorthy & Mathew (2009), Statistical Tolerance Regions.
    Ref: JCGM 100:2008 (GUM), Annex G.

    Parameters:
        n: sample size
        coverage: proportion of population to cover (e.g. 0.95)
        confidence: confidence level that coverage is achieved (e.g. 0.95)

    Returns:
        k: one-sided tolerance factor
    """
    if n <= 1:
        return float('inf')
    z_p = norm.ppf(coverage)
    delta = z_p * np.sqrt(n)
    k = nct.ppf(confidence, df=n - 1, nc=delta) / np.sqrt(n)
    return float(k)


def two_sided_tolerance_k(n, coverage=0.95, confidence=0.95):
    """
    Two-sided normal tolerance factor (Howe / Wald-Wolfowitz approximation).

    Ref: JCGM 100:2008 (GUM), Annex G.
    """
    if n <= 1:
        return float('inf')
    z_p = norm.ppf((1 + coverage) / 2)
    chi2_val = chi2.ppf(1 - confidence, df=n - 1)
    if chi2_val <= 0:
        return float('inf')
    k = z_p * np.sqrt((n - 1) * (1 + 1.0 / n) / chi2_val)
    return float(k)


def welch_satterthwaite(sigmas, dofs):
    """
    Compute effective degrees of freedom via Welch-Satterthwaite formula.

    Ref: JCGM 100:2008 (GUM), Annex G, Section G.4.1, Equation G.2b.

    Parameters:
        sigmas: list of standard uncertainties
        dofs: list of degrees of freedom (use np.inf for Type B / supplier)

    Returns:
        nu_eff: effective degrees of freedom
    """
    sigmas = np.array(sigmas, dtype=float)
    dofs = np.array(dofs, dtype=float)

    u_c = np.sqrt(np.sum(sigmas ** 2))
    if u_c == 0:
        return float('inf')

    numerator = u_c ** 4
    denominator = 0.0
    for s, v in zip(sigmas, dofs):
        if np.isinf(v):
            continue  # Type B sources drop out
        if v == 0:
            warnings.warn(
                f"Source with sigma={s:.4g} has DOF=0 (no data supports "
                f"this estimate). Treating as Type B (infinite DOF)."
            )
            continue
        denominator += s ** 4 / v

    if denominator == 0:
        return float('inf')

    nu_eff = numerator / denominator
    return float(nu_eff)


def coverage_factor_from_dof(nu_eff, coverage=0.95, two_sided=False):
    """
    Compute coverage factor k from effective DOF using Student's t-distribution.

    Ref: JCGM 100:2008 (GUM), Annex G, Table G.2.
    """
    if np.isinf(nu_eff) or nu_eff > 1e6:
        if two_sided:
            return norm.ppf((1 + coverage) / 2)
        else:
            return norm.ppf(coverage)

    nu_eff = max(nu_eff, 1.0)
    if two_sided:
        return float(t_dist.ppf((1 + coverage) / 2, df=nu_eff))
    else:
        return float(t_dist.ppf(coverage, df=nu_eff))


def min_n_distribution_free(coverage=0.95, confidence=0.95, r=1):
    """
    Minimum sample size for distribution-free tolerance bound using r-th
    order statistic.

    Ref: ASME PTC 19.1-2018, non-parametric tolerance intervals.
    """
    q = 1 - coverage
    for n in range(r, 10000):
        prob = 1 - binom.cdf(r - 1, n, q)
        if prob >= confidence:
            return n
    return None


def sigma_from_basis(value, basis, distribution="Normal"):
    """
    Convert a user-entered value to a standard uncertainty (1σ) based on
    the stated sigma basis, accounting for the source's distribution type.

    For 2σ (95%) and 3σ (99.7%) bases, the divisor is the distribution's
    percentile factor rather than the Normal assumption of 2.0 / 3.0,
    so that the resulting 1σ matches the distribution's actual variance.

    Ref: ASME PTC 19.1-2018, Type B evaluation.
    Ref: JCGM 100:2008 (GUM), Section 4.3.
    """
    if basis == "Confirmed 1σ" or basis == "Assumed 1σ (unverified)":
        return value
    elif basis == "2σ (95%)":
        # Divisor = z such that CDF(z*σ) - CDF(-z*σ) ≈ 95% for the distribution
        divisor = _percentile_divisor(distribution, 0.95)
        return value / divisor
    elif basis == "3σ (99.7%)":
        divisor = _percentile_divisor(distribution, 0.997)
        return value / divisor
    elif basis == "Bounding (min/max)":
        if distribution == "Uniform":
            return value / np.sqrt(3)
        elif distribution == "Triangular":
            return value / np.sqrt(6)
        else:
            return value / 3.0  # assume normal 3σ
    return value


def _percentile_divisor(distribution: str, coverage: float) -> float:
    """
    Return the number of standard deviations that span the given symmetric
    central coverage interval for the specified distribution type.

    For Normal, this is simply the inverse-normal (e.g. 1.96 for 95%).
    For other distributions, the factor differs.
    """
    from scipy.stats import norm as _norm, t as _t
    alpha = (1 - coverage) / 2.0

    if distribution in ("Normal", "Lognormal", "Lognormal (σ=0.5)"):
        return float(_norm.ppf(1 - alpha))
    elif distribution == "Student-t (df=5)":
        # 95th percentile of t(5) / std(t(5)); std = sqrt(5/3)
        t_val = float(_t.ppf(1 - alpha, 5))
        t_std = np.sqrt(5.0 / 3.0)
        return t_val / t_std
    elif distribution == "Student-t (df=10)":
        t_val = float(_t.ppf(1 - alpha, 10))
        t_std = np.sqrt(10.0 / 8.0)
        return t_val / t_std
    elif distribution == "Uniform":
        # Uniform: half-width = sqrt(3)*σ, so P(|X|<k*σ) = min(k/sqrt(3), 1)
        # k = sqrt(3) * coverage (but capped at sqrt(3))
        return min(np.sqrt(3) * coverage, np.sqrt(3))
    elif distribution == "Triangular":
        # Symmetric triangular on [-a, a] with σ = a/√6.
        # CDF is quadratic, not linear.  The central coverage p corresponds
        # to k standard deviations where k = √6 · (1 − √(1 − p)).
        # Derivation: P(|X| < k·σ) = p ⟹ F(k·σ) - F(-k·σ) = p
        #   with F(x) = 0.5 + x/a − x²/(2a²), solving gives
        #   k·σ = a·(1 − √(1−p)), hence k = (a/σ)·(1−√(1−p)) = √6·(1−√(1−p)).
        return min(np.sqrt(6) * (1.0 - np.sqrt(1.0 - coverage)), np.sqrt(6))
    else:
        # Default: normal assumption
        return float(_norm.ppf(1 - alpha))


def compute_descriptive_stats(data):
    """Compute comprehensive descriptive statistics for a dataset."""
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) == 0:
        return {}

    n = len(data)
    result = {
        'n': n,
        'mean': float(np.mean(data)),
        'std': float(np.std(data, ddof=1)) if n > 1 else 0.0,
        'se': float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'range': float(np.max(data) - np.min(data)),
        'median': float(np.median(data)),
        'p5': float(np.percentile(data, 5)),
        'p95': float(np.percentile(data, 95)),
    }

    if n >= 3:
        result['skewness'] = float(skew(data, bias=False))
        result['kurtosis'] = float(kurtosis(data, bias=False))  # excess kurtosis
    else:
        result['skewness'] = 0.0
        result['kurtosis'] = 0.0

    if 3 <= n <= 5000:
        try:
            stat_sw, p_sw = shapiro(data)
            result['shapiro_stat'] = float(stat_sw)
            result['shapiro_p'] = float(p_sw)
        except Exception:
            result['shapiro_stat'] = None
            result['shapiro_p'] = None
    else:
        result['shapiro_stat'] = None
        result['shapiro_p'] = None

    return result


def compute_multivariate_validation(cd: 'ComparisonData') -> dict:
    """Compute covariance-aware supplemental validation metrics.

    Returns a dictionary with:
      - computed (bool)
      - score (normalized Mahalanobis mean-bias scalar)
      - t2 (Hotelling-style statistic approximation)
      - pvalue (chi-square approximation on t2)
      - n_locations, n_conditions
      - note
    """
    out = {
        'computed': False,
        'score': 0.0,
        't2': 0.0,
        'pvalue': 1.0,
        'n_locations': 0,
        'n_conditions': 0,
        'note': "",
    }
    if cd is None or cd.data is None or cd.data.size == 0 or cd.data.ndim != 2:
        out['note'] = "Multivariate metric not computed: no matrix data."
        return out

    X = np.asarray(cd.data, dtype=float)
    # Keep only conditions with complete location vectors.
    valid_cols = np.all(np.isfinite(X), axis=0)
    X = X[:, valid_cols]

    m, n = X.shape if X.ndim == 2 else (0, 0)
    out['n_locations'] = int(m)
    out['n_conditions'] = int(n)
    if m < 2 or n < 3:
        out['note'] = (
            "Multivariate metric requires at least 2 locations and 3 complete conditions."
        )
        return out

    try:
        mean_vec = np.mean(X, axis=1)
        S = np.cov(X, bias=False)
        if S.ndim == 0:
            S = np.array([[float(S)]], dtype=float)
        S = np.asarray(S, dtype=float)
        # Light Tikhonov regularization for near-singular covariance.
        tr = float(np.trace(S)) if S.size > 0 else 0.0
        reg = max(1e-12, 1e-10 * tr / max(m, 1))
        S_reg = S + reg * np.eye(m)
        S_inv = np.linalg.pinv(S_reg)
        d2 = float(mean_vec.T @ S_inv @ mean_vec)
        d2 = max(d2, 0.0)
        score = float(np.sqrt(d2) / max(np.sqrt(m), 1.0))
        t2 = float(n * d2)
        pvalue = float(1.0 - chi2.cdf(t2, df=m))

        out.update({
            'computed': True,
            'score': score,
            't2': t2,
            'pvalue': pvalue,
            'note': (
                "Multivariate supplemental metric computed on complete-condition matrix "
                "(covariance-aware mean-bias distance)."
            ),
        })
    except Exception as exc:
        out['note'] = f"Multivariate metric failed: {exc}"
    return out


def assess_distribution(stats_dict):
    """
    Provide automated distribution assessment based on descriptive statistics.

    Returns a tuple of (message, severity) where severity is 'green', 'yellow', or 'red'.
    """
    if not stats_dict or stats_dict.get('n', 0) < 3:
        return "Insufficient data for distribution assessment.", "yellow"

    sk = abs(stats_dict.get('skewness', 0))
    ku = stats_dict.get('kurtosis', 0)
    p_sw = stats_dict.get('shapiro_p', None)

    messages = []
    severity = 'green'

    # Normality check
    if p_sw is not None and sk < 0.5 and abs(ku) < 1.0 and p_sw > 0.05:
        messages.append(
            "Data is consistent with a normal distribution. Normal k-factors "
            "are appropriate. [GUM §4.3]"
        )
    elif sk < 0.5 and ku < -0.5:
        messages.append(
            "Data is symmetric but platykurtic (flatter than normal, e.g., "
            "uniform-like). Normal k-factors are conservative — they overestimate "
            "the tails. This is acceptable for safety-critical applications. [GUM §4.3]"
        )
        severity = 'yellow'
    elif sk < 0.5 and ku > 1.0:
        messages.append(
            "Data is symmetric but leptokurtic (heavier tails than normal). "
            "Normal k-factors may be NON-CONSERVATIVE. Consider distribution-free "
            "methods or Monte Carlo. [GUM §4.3, JCGM 101:2008]"
        )
        severity = 'red'
    elif sk > 1.0:
        messages.append(
            "Data is significantly skewed. Normal k-factors may not be "
            "appropriate. Consider fitting an asymmetric distribution or "
            "using Monte Carlo. [GUM §4.3, JCGM 101:2008]"
        )
        severity = 'red'
    else:
        messages.append(
            "Data shows moderate deviation from normality. Review the QQ plot "
            "and consider whether normal k-factors are appropriate. [GUM §4.3]"
        )
        severity = 'yellow'

    return '\n'.join(messages), severity


def assess_sample_size(n):
    """
    Provide sample size guidance per V&V 20 and PTC 19.1.

    Returns (message, severity).
    """
    if n < 20:
        k_approx = one_sided_tolerance_k(max(n, 2), 0.95, 0.95)
        msg = (
            f"SMALL SAMPLE (n={n}). k-factor penalty is significant "
            f"(k ≈ {k_approx:.2f} for 95/95 vs k→1.645 at large n). "
            f"Consider pooling additional locations if justified. "
            f"Distribution-free 95/95 bounds require n ≥ 59. "
            f"To reduce k below 2.2, you need n ≥ 32. "
            f"[V&V 20 §6, PTC 19.1 §7]"
        )
        return msg, 'red'
    elif n < 60:
        k_approx = one_sided_tolerance_k(n, 0.95, 0.95)
        n_needed = min_n_distribution_free(0.95, 0.95)
        msg = (
            f"MODERATE SAMPLE (n={n}). k-factor penalty is moderate "
            f"(k ≈ {k_approx:.2f}). Distribution-free bounds require "
            f"n ≥ {n_needed}. [V&V 20 §6, PTC 19.1 §7]"
        )
        return msg, 'yellow'
    else:
        k_approx = one_sided_tolerance_k(n, 0.95, 0.95)
        msg = (
            f"ADEQUATE SAMPLE (n={n}). k-factor penalty is small "
            f"(k ≈ {k_approx:.2f}). Distribution-free 95/95 bounds are "
            f"available. [V&V 20 §6, PTC 19.1 §7]"
        )
        return msg, 'green'


def _resolve_gof_mc_samples(n: int, requested: Optional[int]) -> int:
    """Resolve bootstrap GOF sample count with runtime guardrails."""
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


def fit_distributions(data, _n_mc_gof: Optional[int] = None):
    """
    Fit candidate distributions to data and rank by goodness of fit.

    Returns list of dicts sorted by best fit, each containing:
        name, params, gof_p, ks_p, AICc, recommendation
    """
    data = np.array(data, dtype=float)
    data = data[~np.isnan(data)]
    if len(data) < 5:
        return []
    n_mc_gof = _resolve_gof_mc_samples(len(data), _n_mc_gof)

    results = []
    candidates = {
        'Normal': {'dist': stats.norm, 'args': ()},
        'Uniform': {'dist': stats.uniform, 'args': ()},
        'Logistic': {'dist': stats.logistic, 'args': ()},
        'Laplace': {'dist': stats.laplace, 'args': ()},
        'Lognormal': {'dist': stats.lognorm, 'args': ()},
        'Weibull': {'dist': stats.weibull_min, 'args': ()},
        'Exponential': {'dist': stats.expon, 'args': ()},
    }

    for name, info in candidates.items():
        try:
            dist = info['dist']
            # Skip distributions that require positive data
            if name in ('Lognormal', 'Weibull', 'Exponential'):
                if np.any(data <= 0):
                    # Shift data for fitting
                    shift = abs(np.min(data)) + 1.0
                    fit_data = data + shift
                else:
                    fit_data = data
                    shift = 0
            else:
                fit_data = data
                shift = 0

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                params = dist.fit(fit_data)
            # KS test is retained as a diagnostic only.
            ks_stat, ks_p = kstest(fit_data, dist.cdf, args=params)

            gof_stat = float('inf')
            gof_p = 0.0
            gof_method = "bootstrap_ad"
            if n_mc_gof <= 0:
                gof_stat = float(ks_stat)
                gof_p = float(ks_p)
                gof_method = "ks_screening"
            else:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        gof = goodness_of_fit(
                            dist=dist,
                            data=fit_data,
                            statistic='ad',
                            n_mc_samples=n_mc_gof,
                            random_state=0,
                        )
                    gof_stat = float(gof.statistic)
                    gof_p = float(gof.pvalue)
                except Exception:
                    # Fallback for environments/distributions where bootstrap GOF fails.
                    gof_stat = float(ks_stat)
                    gof_p = float(ks_p)
                    gof_method = "ks_fallback"

            # AIC/BIC information criteria (preferred ranking)
            n = len(fit_data)
            k_params = len(params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                log_lik = float(np.sum(dist.logpdf(fit_data, *params)))
            aic = 2.0 * k_params - 2.0 * log_lik
            # Corrected AIC (AICc) for small samples
            if n - k_params - 1 > 0:
                aicc = aic + (2.0 * k_params * (k_params + 1)) / (n - k_params - 1)
            else:
                aicc = aic
            bic = k_params * np.log(n) - 2.0 * log_lik

            results.append({
                'name': name,
                'params': params,
                'shift': shift,
                'gof_stat': float(gof_stat),
                'gof_p': float(gof_p),
                'gof_method': gof_method,
                'passed_gof': bool(gof_p > 0.05),
                'ks_stat': float(ks_stat),
                'ks_p': float(ks_p),
                'aic': float(aic),
                'aicc': float(aicc),
                'bic': float(bic),
                'n_params': k_params,
            })
        except Exception:
            continue

    # Also test Triangular
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            params = stats.triang.fit(data)
        ks_stat, ks_p = kstest(data, stats.triang.cdf, args=params)
        gof_stat = float('inf')
        gof_p = 0.0
        gof_method = "bootstrap_ad"
        if n_mc_gof <= 0:
            gof_stat = float(ks_stat)
            gof_p = float(ks_p)
            gof_method = "ks_screening"
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    gof = goodness_of_fit(
                        dist=stats.triang,
                        data=data,
                        statistic='ad',
                        n_mc_samples=n_mc_gof,
                        random_state=0,
                    )
                gof_stat = float(gof.statistic)
                gof_p = float(gof.pvalue)
            except Exception:
                gof_stat = float(ks_stat)
                gof_p = float(ks_p)
                gof_method = "ks_fallback"
        n = len(data)
        k_params = len(params)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            log_lik = float(np.sum(stats.triang.logpdf(data, *params)))
        aic = 2.0 * k_params - 2.0 * log_lik
        if n - k_params - 1 > 0:
            aicc = aic + (2.0 * k_params * (k_params + 1)) / (n - k_params - 1)
        else:
            aicc = aic
        bic = k_params * np.log(n) - 2.0 * log_lik
        results.append({
            'name': 'Triangular',
            'params': params,
            'shift': 0,
            'gof_stat': float(gof_stat),
            'gof_p': float(gof_p),
            'gof_method': gof_method,
            'passed_gof': bool(gof_p > 0.05),
            'ks_stat': float(ks_stat),
            'ks_p': float(ks_p),
            'aic': float(aic),
            'aicc': float(aicc),
            'bic': float(bic),
            'n_params': k_params,
        })
    except Exception:
        pass

    # Primary sort: GOF pass first, then AICc, then GOF p-value.
    results.sort(
        key=lambda x: (
            0 if x.get('passed_gof') else 1,
            x.get('aicc', float('inf')),
            -x.get('gof_p', 0.0),
            -x.get('ks_p', 0.0),
        )
    )

    # Add recommendations
    # AICc = Corrected Akaike Information Criterion: a score measuring
    # how well a distribution fits the data; lower is better.  The
    # correction prevents overfitting with small samples.
    for i, r in enumerate(results):
        aicc_str = f"AICc={r.get('aicc', 0):.1f}"
        gof_str = f"GOF p={r.get('gof_p', 0.0):.3f}"
        diag_str = f"KS(diag)={r['ks_p']:.3f}"
        gof_method = r.get('gof_method', 'bootstrap_ad')
        method_note = (
            "Bootstrap GOF passed at α=0.05."
            if gof_method == "bootstrap_ad"
            else "KS screening used (runtime guardrail mode)."
            if gof_method == "ks_screening"
            else "Bootstrap GOF unavailable; KS fallback used."
        )
        if i == 0:
            if r.get('passed_gof'):
                r['recommendation'] = (
                    f"Best fit: {r['name']} ({aicc_str}, {gof_str}; {diag_str}). "
                    f"{method_note} [GUM §4.3, JCGM 101]"
                )
            else:
                r['recommendation'] = (
                    f"Best fit by AICc fallback: {r['name']} "
                    f"({aicc_str}, {gof_str}; {diag_str}). "
                    "No candidate passed GOF screening."
                )
        elif r.get('passed_gof'):
            r['recommendation'] = (
                f"Acceptable: {r['name']} ({aicc_str}, {gof_str}; {diag_str}). "
                f"{method_note}"
            )
        else:
            r['recommendation'] = (
                f"Poor fit: {r['name']} ({aicc_str}, {gof_str}; {diag_str}). "
                "GOF screening below α=0.05."
            )

    return results


def generate_bifurcated_normal(sigma_upper, sigma_lower, mean, n_trials,
                               rng=None):
    """Generate samples from a bifurcated (split) Gaussian.

    Uses σ⁺ for samples above the mean and σ⁻ for samples below.
    This produces a continuous, potentially asymmetric distribution that
    honours the different uncertainties in each direction.

    References: Barlow, R. (2004) "Asymmetric Statistical Errors".
    """
    if rng is None:
        rng = np.random.default_rng()
    # Draw from standard normal and rescale each half
    z = rng.standard_normal(n_trials)
    samples = np.where(z >= 0,
                       mean + z * sigma_upper,
                       mean + z * sigma_lower)
    return samples


def generate_mc_samples(distribution, sigma, mean, n_trials, raw_data=None,
                        rng=None, sigma_upper=None, sigma_lower=None):
    """
    Generate Monte Carlo samples from a specified distribution.

    Parameters
    ----------
    rng : numpy.random.Generator, optional
        Thread-safe RNG instance.  Falls back to the legacy global RNG
        if not provided (for backward compatibility in non-threaded use).
    sigma_upper, sigma_lower : float, optional
        When both are provided (asymmetric mode), uses a bifurcated Gaussian
        instead of the symmetric distribution for Normal sources.

    Ref: JCGM 101:2008 (GUM Supplement 1).
    """
    # Use provided Generator or fall back to legacy API
    if rng is None:
        rng = np.random.default_rng()
    # Asymmetric (bifurcated) Gaussian for Normal sources
    if (sigma_upper is not None and sigma_lower is not None
            and distribution == "Normal"):
        return generate_bifurcated_normal(
            sigma_upper, sigma_lower, mean, n_trials, rng=rng)
    if distribution == "Normal":
        return rng.normal(mean, sigma, n_trials)
    elif distribution == "Uniform":
        a = sigma * np.sqrt(3)
        return rng.uniform(mean - a, mean + a, n_trials)
    elif distribution == "Triangular":
        a = sigma * np.sqrt(6)
        if a == 0:
            return np.full(n_trials, mean)
        return rng.triangular(mean - a, mean, mean + a, n_trials)
    elif distribution == "Lognormal":
        if sigma > 0 and abs(mean) > sigma * 0.01:
            abs_mean = abs(mean)
            underlying_sigma = np.sqrt(np.log(1 + (sigma / abs_mean) ** 2))
            underlying_mu = np.log(abs_mean) - underlying_sigma ** 2 / 2
            samples = rng.lognormal(underlying_mu, underlying_sigma, n_trials)
            if mean < 0:
                samples = -samples  # mirror for negative mean
            return samples
        return rng.normal(mean, max(sigma, 1e-15), n_trials)
    elif distribution == "Lognormal (σ=0.5)":
        # Moderately skewed lognormal: cap underlying_sigma at 0.5 but
        # still honour the user's sigma for the variance.
        if sigma > 0 and abs(mean) > sigma * 0.01:
            abs_mean = abs(mean)
            user_sigma_ln = np.sqrt(np.log(1 + (sigma / abs_mean) ** 2))
            underlying_sigma = min(user_sigma_ln, 0.5)
            underlying_mu = np.log(abs_mean) - underlying_sigma ** 2 / 2
            raw = rng.lognormal(underlying_mu, underlying_sigma, n_trials)
            raw_std = float(np.std(raw, ddof=0))
            if raw_std > 1e-15:
                samples = mean + (raw - np.mean(raw)) * (sigma / raw_std)
            else:
                samples = raw
            if mean < 0:
                samples = 2 * mean - samples  # mirror
            return samples
        return rng.normal(mean, max(sigma, 1e-15), n_trials)
    elif distribution == "Logistic":
        scale = sigma * np.sqrt(3) / np.pi
        return rng.logistic(mean, scale, n_trials)
    elif distribution == "Laplace":
        scale = sigma / np.sqrt(2)
        return rng.laplace(mean, scale, n_trials)
    elif distribution == "Student-t (df=5)":
        samples = rng.standard_t(5, n_trials)
        t_std = np.sqrt(5.0 / 3.0)
        return mean + sigma * samples / t_std
    elif distribution == "Student-t (df=10)":
        samples = rng.standard_t(10, n_trials)
        t_std = np.sqrt(10.0 / 8.0)
        return mean + sigma * samples / t_std
    elif distribution == "Exponential":
        return rng.exponential(sigma, n_trials) + (mean - sigma)
    elif distribution == "Weibull":
        # Use shape=2 (Rayleigh-like) by default, scale to match sigma
        # Var(Weibull k=2) = scale^2 * (1 - pi/4), so scale = sigma / sqrt(1 - pi/4)
        shape = 2.0
        scale = sigma / np.sqrt(1.0 - np.pi / 4.0)
        samples = rng.weibull(shape, n_trials) * scale
        # Use theoretical mean for recentering (avoids sample-mean correlation)
        from scipy.special import gamma as _gamma_fn
        theoretical_mean = scale * _gamma_fn(1.0 + 1.0 / shape)
        return samples - theoretical_mean + mean
    elif distribution == "Custom/Empirical (Bootstrap)":
        if raw_data is not None and len(raw_data) > 0:
            centered = raw_data - np.mean(raw_data) + mean
            return rng.choice(centered, size=n_trials, replace=True)
        return rng.normal(mean, sigma, n_trials)
    else:
        return rng.normal(mean, sigma, n_trials)


def generate_lhs_samples(distribution, sigma, mean, n_trials, raw_data=None,
                         rng=None, sigma_upper=None, sigma_lower=None):
    """
    Generate Latin Hypercube samples from a specified distribution.

    Uses stratified sampling of the [0, 1] probability space followed by
    the inverse-CDF (percent-point function) transform.  Produces the same
    marginal distribution as ``generate_mc_samples`` but with stratified
    coverage that often stabilizes percentile estimates faster than random MC.

    Parameters
    ----------
    distribution : str
        Distribution name (same strings as ``generate_mc_samples``).
    sigma : float
        Standard uncertainty (standard deviation) for the source.
    mean : float
        Central value of the distribution.
    n_trials : int
        Number of samples to draw.
    raw_data : array-like, optional
        Raw data for Custom/Empirical Bootstrap resampling.
    rng : numpy.random.Generator, optional
        Thread-safe RNG.  Falls back to a fresh Generator if not provided.
    sigma_upper, sigma_lower : float, optional
        Asymmetric σ⁺/σ⁻ — uses bifurcated Gaussian via inverse CDF.

    Returns
    -------
    samples : numpy.ndarray of shape (n_trials,)

    Ref
    ---
    McKay, Beckman & Conover (1979);
    JCGM 101:2008 (Monte Carlo framework);
    ASME V&V 20, Section 4.4 — Monte Carlo propagation methods.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Core LHS: stratify [0, 1] into n_trials equal intervals, draw one
    # uniform random deviate per interval, then shuffle.
    # ------------------------------------------------------------------
    n = n_trials
    intervals = (np.arange(n, dtype=np.float64) + rng.uniform(0, 1, n)) / n
    rng.shuffle(intervals)
    # Clip to avoid ±inf at the boundaries of the ppf
    u = np.clip(intervals, 0.5 / n, 1.0 - 0.5 / n)

    # ------------------------------------------------------------------
    # Inverse-CDF transform for each distribution type.
    # ------------------------------------------------------------------
    if sigma == 0:
        return np.full(n, mean)

    # Asymmetric bifurcated Gaussian (LHS version)
    if (sigma_upper is not None and sigma_lower is not None
            and distribution == "Normal"):
        z = norm.ppf(u)
        return np.where(z >= 0,
                        mean + z * sigma_upper,
                        mean + z * sigma_lower)

    if distribution == "Normal":
        return norm.ppf(u, loc=mean, scale=sigma)

    elif distribution == "Uniform":
        a = sigma * np.sqrt(3)
        return uniform.ppf(u, loc=mean - a, scale=2 * a)

    elif distribution == "Triangular":
        a = sigma * np.sqrt(6)
        if a == 0:
            return np.full(n, mean)
        # scipy triang: c = (mode - loc) / scale.  Symmetric ⇒ c = 0.5
        return triang.ppf(u, c=0.5, loc=mean - a, scale=2 * a)

    elif distribution == "Lognormal":
        if sigma > 0 and abs(mean) > sigma * 0.01:
            abs_mean = abs(mean)
            underlying_sigma = np.sqrt(np.log(1 + (sigma / abs_mean) ** 2))
            underlying_mu = np.log(abs_mean) - underlying_sigma ** 2 / 2
            samples = lognorm.ppf(u, s=underlying_sigma, scale=np.exp(underlying_mu))
            if mean < 0:
                samples = -samples
            return samples
        return norm.ppf(u, loc=mean, scale=max(sigma, 1e-15))

    elif distribution == "Lognormal (σ=0.5)":
        if sigma > 0 and abs(mean) > sigma * 0.01:
            abs_mean = abs(mean)
            user_sigma_ln = np.sqrt(np.log(1 + (sigma / abs_mean) ** 2))
            underlying_sigma = min(user_sigma_ln, 0.5)
            underlying_mu = np.log(abs_mean) - underlying_sigma ** 2 / 2
            raw = lognorm.ppf(u, s=underlying_sigma, scale=np.exp(underlying_mu))
            raw_std = float(np.std(raw, ddof=0))
            if raw_std > 1e-15:
                samples = mean + (raw - np.mean(raw)) * (sigma / raw_std)
            else:
                samples = raw
            if mean < 0:
                samples = 2 * mean - samples
            return samples
        return norm.ppf(u, loc=mean, scale=max(sigma, 1e-15))

    elif distribution == "Logistic":
        scale = sigma * np.sqrt(3) / np.pi
        return logistic.ppf(u, loc=mean, scale=scale)

    elif distribution == "Laplace":
        scale = sigma / np.sqrt(2)
        return laplace.ppf(u, loc=mean, scale=scale)

    elif distribution == "Student-t (df=5)":
        t_std = np.sqrt(5.0 / 3.0)
        raw = t_dist.ppf(u, df=5)
        return mean + sigma * raw / t_std

    elif distribution == "Student-t (df=10)":
        t_std = np.sqrt(10.0 / 8.0)
        raw = t_dist.ppf(u, df=10)
        return mean + sigma * raw / t_std

    elif distribution == "Exponential":
        # Exponential with mean = sigma, shifted so distribution mean = mean
        return expon.ppf(u, loc=mean - sigma, scale=sigma)

    elif distribution == "Weibull":
        shape = 2.0
        scale = sigma / np.sqrt(1.0 - np.pi / 4.0)
        raw = weibull_min.ppf(u, c=shape, scale=scale)
        from scipy.special import gamma as _gamma_fn
        theoretical_mean = scale * _gamma_fn(1.0 + 1.0 / shape)
        return raw - theoretical_mean + mean

    elif distribution == "Custom/Empirical (Bootstrap)":
        # Stratified resampling from empirical CDF
        if raw_data is not None and len(raw_data) > 0:
            centered = np.sort(raw_data - np.mean(raw_data) + mean)
            # Build empirical CDF: F(x_i) = i / (N+1)  (Hazen plotting position)
            n_data = len(centered)
            ecdf_x = centered
            ecdf_y = (np.arange(1, n_data + 1)) / (n_data + 1.0)
            # Interpolate the quantile function at the stratified u values
            return np.interp(u, ecdf_y, ecdf_x)
        return norm.ppf(u, loc=mean, scale=sigma)

    else:
        return norm.ppf(u, loc=mean, scale=sigma)


# K-factor lookup table generator for reference tab
def generate_k_factor_table():
    """Generate precomputed k-factor tables for common n values."""
    sample_sizes = [5, 7, 10, 15, 20, 30, 50, 60, 100, 150, 200, 500, 1000]
    coverages = [0.90, 0.95, 0.99]
    confidences = [0.90, 0.95, 0.99]

    tables = {}
    for sided in ['one-sided', 'two-sided']:
        tables[sided] = {}
        for cov in coverages:
            for conf in confidences:
                key = f"p={int(cov*100)}%, γ={int(conf*100)}%"
                rows = []
                for n in sample_sizes:
                    if sided == 'one-sided':
                        k = one_sided_tolerance_k(n, cov, conf)
                    else:
                        k = two_sided_tolerance_k(n, cov, conf)
                    rows.append((n, round(k, 4)))
                # Add infinity
                if sided == 'one-sided':
                    k_inf = norm.ppf(cov)
                else:
                    k_inf = norm.ppf((1 + cov) / 2)
                rows.append(('∞', round(k_inf, 4)))
                tables[sided][key] = rows
    return tables


# =============================================================================
# SECTION 5: DATA MODEL CLASSES
# =============================================================================

@dataclass
class UncertaintySource:
    """Represents a single uncertainty source."""
    name: str = "New Source"
    category: str = "Numerical (u_num)"  # maps to V&V 20 categories
    input_type: str = "Sigma Value Only"
    distribution: str = "Normal"
    sigma_basis: str = "Confirmed 1σ"
    sigma_value: float = 0.0
    raw_sigma_value: float = 0.0  # before basis conversion
    mean_value: float = 0.0
    sample_size: int = 0
    dof: float = float('inf')  # degrees of freedom
    is_supplier: bool = False  # if True, dof = inf
    unit: str = "°F"
    tabular_data: List[float] = field(default_factory=list)
    tolerance_value: float = 0.0
    tolerance_k: float = 2.0
    rss_value: float = 0.0
    rss_n_components: int = 1
    sensitivity_deltas: List[float] = field(default_factory=list)
    notes: str = ""
    is_centered_on_zero: bool = True
    enabled: bool = True
    correlation_group: str = ""       # group label for correlated sources
    correlation_coefficient: float = 0.0  # pairwise ρ with group reference
    # Epistemic/aleatoric classification (Section 6.2)
    uncertainty_class: str = "aleatoric"    # aleatoric, epistemic, mixed
    representation: str = "distribution"    # distribution, interval, scenario_set, model_ensemble
    basis_type: str = "measured"            # measured, spec_limit, expert_judgment, standard_reference, assumed
    reducibility: str = "low"               # high, medium, low
    evidence_note: str = ""                 # concise traceability text
    # Interval bounds for epistemic interval representation (double-loop MC)
    interval_lower: float = 0.0            # lower bound when representation="interval"
    interval_upper: float = 0.0            # upper bound when representation="interval"
    # Asymmetric uncertainty (GUM §4.3.8, Barlow 2004)
    asymmetric: bool = False                # Enable asymmetric mode
    sigma_upper: float = 0.0               # σ⁺ (positive direction)
    sigma_lower: float = 0.0               # σ⁻ (negative direction)
    one_sided: bool = False                 # Only one direction tested
    one_sided_direction: str = "upper"      # "upper" or "lower"
    mirror_assumed: bool = True             # Assume σ_missing = σ_observed

    def get_standard_uncertainty(self) -> float:
        """Return the 1σ standard uncertainty for this source.

        When *asymmetric* is True, returns the effective σ per GUM §4.3.8:
            σ_eff = √((σ⁺² + σ⁻²) / 2)
        If *one_sided* and *mirror_assumed*, the missing direction is set
        equal to the observed direction.
        """
        if self.asymmetric:
            sp = self.sigma_upper
            sm = self.sigma_lower
            if self.one_sided and self.mirror_assumed:
                if self.one_sided_direction == "upper":
                    sm = sp
                else:
                    sp = sm
            return float(np.sqrt((sp ** 2 + sm ** 2) / 2.0))
        if self.input_type == "Tabular Data" and len(self.tabular_data) > 1:
            return float(np.std(self.tabular_data, ddof=1))
        elif self.input_type == "Sigma Value Only":
            return sigma_from_basis(self.raw_sigma_value, self.sigma_basis, self.distribution)
        elif self.input_type == "Tolerance/Expanded Value":
            if self.tolerance_k > 0:
                return self.tolerance_value / self.tolerance_k
            return self.tolerance_value
        elif self.input_type == "RSS of Sub-Components":
            return sigma_from_basis(self.rss_value, self.sigma_basis, self.distribution)
        elif self.input_type == "CFD Sensitivity Run":
            if len(self.sensitivity_deltas) > 1:
                return float(np.std(self.sensitivity_deltas, ddof=1))
            elif len(self.sensitivity_deltas) == 1:
                return abs(self.sensitivity_deltas[0])
            return 0.0
        return self.sigma_value

    def get_sigma_upper(self) -> float:
        """Return σ⁺ (positive direction uncertainty)."""
        if not self.asymmetric:
            return self.get_standard_uncertainty()
        sp = self.sigma_upper
        if self.one_sided and self.mirror_assumed and self.one_sided_direction == "lower":
            sp = self.sigma_lower
        return sp

    def get_sigma_lower(self) -> float:
        """Return σ⁻ (negative direction uncertainty)."""
        if not self.asymmetric:
            return self.get_standard_uncertainty()
        sm = self.sigma_lower
        if self.one_sided and self.mirror_assumed and self.one_sided_direction == "upper":
            sm = self.sigma_upper
        return sm

    def get_dof(self) -> float:
        """Return degrees of freedom for this source."""
        if self.is_supplier:
            return float('inf')
        if self.input_type == "Tabular Data" and len(self.tabular_data) > 1:
            return float(len(self.tabular_data) - 1)
        if self.sample_size > 1:
            return float(self.sample_size - 1)
        return self.dof

    def get_category_key(self) -> str:
        """Return the V&V 20 category key."""
        for i, cat in enumerate(UNCERTAINTY_CATEGORIES):
            if self.category == cat:
                return CATEGORY_KEYS[i]
        return "u_num"

    def to_dict(self) -> dict:
        d = asdict(self)
        d['dof'] = self.dof if not np.isinf(self.dof) else "inf"
        return d

    @staticmethod
    def from_dict(d: dict) -> 'UncertaintySource':
        if d.get('dof') == "inf":
            d['dof'] = float('inf')
        return UncertaintySource(**{k: v for k, v in d.items()
                                     if k in UncertaintySource.__dataclass_fields__})


@dataclass
class ComparisonData:
    """Holds the CFD-to-flight-test comparison error data."""
    data: np.ndarray = field(default_factory=lambda: np.array([]))
    sensor_names: List[str] = field(default_factory=list)
    condition_names: List[str] = field(default_factory=list)
    unit: str = "°F"
    is_pooled: bool = True

    def flat_data(self) -> np.ndarray:
        """Return all data as a flat 1D array."""
        if self.data.size == 0:
            return np.array([])
        flat = self.data.flatten()
        return flat[~np.isnan(flat)]

    def get_stats(self) -> dict:
        flat = self.flat_data()
        if flat.size == 0:
            return {}
        return compute_descriptive_stats(flat)

    def per_location_stats(self) -> List[dict]:
        """Compute per-sensor-location statistics."""
        if self.data.ndim != 2 or self.data.shape[0] == 0:
            return []
        results = []
        for i in range(self.data.shape[0]):
            row = self.data[i, :]
            row = row[~np.isnan(row)]
            if len(row) > 0:
                name = self.sensor_names[i] if i < len(self.sensor_names) else f"Location {i+1}"
                results.append({
                    'name': name,
                    'mean': float(np.mean(row)),
                    'std': float(np.std(row, ddof=1)) if len(row) > 1 else 0.0,
                    'n': len(row),
                    'orig_row': i,
                })
        return results

    def to_dict(self) -> dict:
        return {
            'data': self.data.tolist() if self.data.size > 0 else [],
            'sensor_names': self.sensor_names,
            'condition_names': self.condition_names,
            'unit': self.unit,
            'is_pooled': self.is_pooled,
        }

    @staticmethod
    def from_dict(d: dict) -> 'ComparisonData':
        cd = ComparisonData()
        cd.data = np.array(d.get('data', []), dtype=float)
        cd.sensor_names = d.get('sensor_names', [])
        cd.condition_names = d.get('condition_names', [])
        cd.unit = d.get('unit', '°F')
        cd.is_pooled = d.get('is_pooled', True)
        return cd


@dataclass
class AnalysisSettings:
    """All analysis configuration settings."""
    coverage: float = 0.95
    confidence: float = 0.95
    one_sided: bool = True
    k_method: str = K_METHOD_VV20
    manual_k: float = 2.0
    mc_n_trials: int = 100000
    mc_seed: Optional[int] = None
    mc_bootstrap: bool = True
    mc_sampling_method: str = "Monte Carlo (Random)"
    bound_type: str = "Both (for comparison)"
    mc_comparison_sampling: str = "Bootstrap from raw data"
    global_unit: str = "°F"
    validation_mode: str = "Standard scalar (V&V 20)"
    # Double-loop Monte Carlo settings (Oberkampf & Roy 2010)
    mc_mode: str = "Single-Loop"               # "Single-Loop", "Double-Loop (Corners)", "Double-Loop (Full)"
    mc_n_outer: int = 200                       # outer loop epistemic samples (Full mode)
    mc_n_inner: int = 10000                     # inner loop aleatory samples per realization
    mc_mixed_treatment: str = "Treat as epistemic"  # "Treat as epistemic" or "Treat as aleatory"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> 'AnalysisSettings':
        return AnalysisSettings(**{k: v for k, v in d.items()
                                    if k in AnalysisSettings.__dataclass_fields__})


@dataclass
class RSSResults:
    """Stores RSS analysis results."""
    u_num: float = 0.0
    u_input: float = 0.0
    u_D: float = 0.0
    u_val: float = 0.0
    nu_eff: float = float('inf')
    k_factor: float = 2.0
    k_method_used: str = ""
    U_val: float = 0.0
    E_mean: float = 0.0
    s_E: float = 0.0
    n_data: int = 0
    lower_bound_uval: float = 0.0
    upper_bound_uval: float = 0.0
    lower_bound_sE: float = 0.0
    upper_bound_sE: float = 0.0
    u_model: float = 0.0
    model_form_pct: float = 0.0
    bias_explained: Optional[bool] = None
    source_contributions: List[dict] = field(default_factory=list)
    computed: bool = False
    has_correlations: bool = False
    correlation_groups: List[str] = field(default_factory=list)
    # Effective pairwise correlation detail: list of (group_name, names_list, C_matrix)
    correlation_matrices: List[tuple] = field(default_factory=list)
    # Uncertainty class split
    u_aleatoric: float = 0.0     # RSS of aleatoric-class source sigmas
    u_epistemic: float = 0.0     # RSS of epistemic-class source sigmas
    u_mixed: float = 0.0         # RSS of mixed-class source sigmas
    pct_epistemic: float = 0.0   # U_E² / u_val² × 100
    # Optional multivariate supplemental validation metric
    multivariate_enabled: bool = False
    multivariate_computed: bool = False
    multivariate_score: float = 0.0
    multivariate_t2: float = 0.0
    multivariate_pvalue: float = 1.0
    multivariate_n_locations: int = 0
    multivariate_n_conditions: int = 0
    multivariate_note: str = ""


@dataclass
class MCResults:
    """Stores Monte Carlo analysis results."""
    n_trials: int = 0
    combined_mean: float = 0.0
    combined_std: float = 0.0
    pct_5: float = 0.0
    pct_95: float = 0.0
    lower_bound: float = 0.0
    upper_bound: float = 0.0
    bootstrap_ci_low: float = 0.0
    bootstrap_ci_high: float = 0.0
    # Coverage-matched percentiles (computed from settings, not hardcoded)
    _lower_pct: float = 5.0     # lower percentile used (e.g. 2.5 for 95% two-sided)
    _upper_pct: float = 95.0    # upper percentile used (e.g. 97.5 for 95% two-sided)
    _coverage: float = 0.95
    _one_sided: bool = True
    sampling_method: str = "Monte Carlo (Random)"
    samples: np.ndarray = field(default_factory=lambda: np.array([]))
    notes: List[str] = field(default_factory=list)
    computed: bool = False

    def to_dict(self):
        d = {}
        _skip = {'samples', '_p5_boots', '_p95_boots'}
        for k, v in self.__dict__.items():
            if k in _skip:
                continue  # Don't save large arrays (samples, bootstrap)
            d[k] = v
        return d


@dataclass
class DoubleLoopMCResults:
    """Stores double-loop Monte Carlo results (epistemic/aleatory separation).

    The outer loop varies epistemic parameters; the inner loop propagates
    aleatory uncertainty.  The result is a family of CDFs whose envelope
    forms a probability box (p-box).

    References:
        Oberkampf & Roy (2010), Verification and Validation in Scientific Computing
        Ferson et al. (2003), Constructing probability boxes and Dempster-Shafer structures
    """
    mode: str = "corners"                  # "corners" or "full"
    n_outer: int = 0                       # number of epistemic realizations
    n_inner: int = 0                       # aleatory trials per realization
    n_epistemic_sources: int = 0           # count of epistemic sources in outer loop
    n_aleatory_sources: int = 0            # count of aleatory sources in inner loop

    # P-box envelope (bounding CDFs on a shared x-grid)
    pbox_x: np.ndarray = field(default_factory=lambda: np.array([]))
    pbox_cdf_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    pbox_cdf_upper: np.ndarray = field(default_factory=lambda: np.array([]))

    # Summary statistics across the family of inner-loop CDFs
    lower_bound_min: float = 0.0           # worst-case (most negative) lower bound
    lower_bound_max: float = 0.0           # best-case lower bound
    upper_bound_min: float = 0.0           # best-case upper bound
    upper_bound_max: float = 0.0           # worst-case (most positive) upper bound
    mean_of_means: float = 0.0             # mean of inner-loop means
    std_of_means: float = 0.0              # std of inner-loop means (epistemic spread)

    # Validation metric
    validation_fraction: float = 0.0       # fraction of realizations bracketing zero

    # Coverage settings used
    _coverage: float = 0.95
    _one_sided: bool = True
    _lower_pct: float = 5.0
    _upper_pct: float = 95.0

    # Per-realization summary arrays (each length n_outer)
    realization_lower_bounds: np.ndarray = field(default_factory=lambda: np.array([]))
    realization_upper_bounds: np.ndarray = field(default_factory=lambda: np.array([]))
    realization_means: np.ndarray = field(default_factory=lambda: np.array([]))

    # Epistemic source names and corner values for reporting
    epistemic_source_names: List[str] = field(default_factory=list)

    notes: List[str] = field(default_factory=list)
    computed: bool = False

    def to_dict(self):
        d = {}
        _skip = {'pbox_x', 'pbox_cdf_lower', 'pbox_cdf_upper',
                 'realization_lower_bounds', 'realization_upper_bounds',
                 'realization_means'}
        for k, v in self.__dict__.items():
            if k in _skip:
                continue
            if isinstance(v, np.ndarray):
                continue  # skip any remaining arrays
            d[k] = v
        return d


# =============================================================================
# SECTION 5b: TABLE HELPER
# =============================================================================

def style_table(table, column_widths=None, stretch_col=None):
    """
    Apply consistent styling to a QTableWidget so that:
      - All columns are user-resizable (Interactive mode)
      - Horizontal scroll bar appears when columns exceed available width
      - Word wrap is enabled so long text grows the row height
      - Optional explicit column widths dict {col_index: width_px}
      - Optional stretch_col: gives that column a generous default width
        but keeps it Interactive (user-resizable) — does NOT lock it
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


def copy_figure_to_clipboard(fig):
    """Copy a matplotlib Figure to the system clipboard as an image."""
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        from PySide6.QtGui import QImage
        image = QImage.fromData(buf.read())
        QApplication.clipboard().setImage(image)
    except Exception as exc:
        QMessageBox.warning(
            None, "Clipboard Error",
            f"Could not copy figure to clipboard:\n\n{exc}"
        )


def export_figure_package(fig, base_path: str, metadata: Optional[dict] = None,
                          figure_id: str = "", analysis_id: str = "",
                          settings_hash: str = "", data_hash: str = "",
                          units: str = "", method_context: str = ""):
    """Export a matplotlib Figure in publication-quality formats.

    Generates:
        {base_path}_300dpi.png  — 300 DPI raster
        {base_path}_600dpi.png  — 600 DPI raster
        {base_path}.svg         — Scalable Vector Graphics
        {base_path}.pdf         — PDF vector
        {base_path}_meta.json   — Metadata sidecar (10-key regulatory spec)

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    base_path : str
        Path without extension (e.g., "C:/output/gci_convergence").
    metadata : dict, optional
        Extra metadata entries to include in the JSON sidecar.
    figure_id : str, optional
        Figure identifier (e.g., "rss_pdf", "convergence_plot").
    analysis_id : str, optional
        Project name or unique analysis identifier.
    settings_hash : str, optional
        SHA-256 hash of the settings used to generate this figure.
    data_hash : str, optional
        SHA-256 hash of the input data.
    units : str, optional
        Unit string for the plotted quantity.
    method_context : str, optional
        Method identifier (e.g., "RSS ASME V&V 20", "GCI Celik 2008").
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


def copy_report_quality_figure(fig):
    """Copy a matplotlib Figure to clipboard at 300 DPI with light report theme."""
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

    # Render at 300 DPI and copy to clipboard
    try:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        buf.seek(0)
        from PySide6.QtGui import QImage
        image = QImage.fromData(buf.read())
        QApplication.clipboard().setImage(image)
    except Exception as exc:
        QMessageBox.warning(
            None, "Clipboard Error",
            f"Could not copy report-quality figure to clipboard:\n\n{exc}"
        )
    finally:
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


def _find_main_window(widget):
    """Walk up the widget hierarchy to find the MainWindow ancestor."""
    w = widget
    while w is not None:
        if hasattr(w, '_project_name') and hasattr(w, '_tab_settings'):
            return w
        w = w.parent() if hasattr(w, 'parent') and callable(w.parent) else None
    return None


def _export_figure_package_dialog(fig, parent, **metadata_kwargs):
    """Open a file dialog and export a figure package.

    Any keyword arguments are forwarded to export_figure_package() as
    metadata fields (analysis_id, settings_hash, method_context, etc.).
    If *parent* has a ``_get_export_metadata()`` method, its return dict
    is used as defaults (overridden by explicit kwargs).
    """
    filepath, _ = QFileDialog.getSaveFileName(
        parent, "Export Figure Package",
        os.path.expanduser("~"),
        "Figure Base Name (*)",
    )
    if not filepath:
        return
    # Remove any extension the user may have typed
    base = os.path.splitext(filepath)[0]
    # Collect metadata from application context
    meta = {}
    mw = _find_main_window(parent)
    if mw is not None:
        meta['analysis_id'] = getattr(mw, '_project_name', '')
        try:
            settings = mw._tab_settings.get_settings()
            import hashlib as _hashlib
            s_json = json.dumps({
                'coverage': settings.coverage,
                'confidence': settings.confidence,
                'one_sided': settings.one_sided,
                'k_method': settings.k_method,
                'bound_type': getattr(settings, 'bound_type', ''),
            }, sort_keys=True)
            meta['settings_hash'] = _hashlib.sha256(
                s_json.encode()).hexdigest()[:16]
            meta['units'] = settings.global_unit
        except Exception:
            pass
        # data_hash: fingerprint of comparison data + source sigmas
        try:
            import hashlib as _hashlib2
            comp = mw._tab_comparison.get_comparison_data()
            sources = mw._tab_sources.get_sources()
            d_parts = []
            if comp.data.size > 0:
                d_parts.append(comp.data.tobytes())
            for s in sources:
                if s.enabled:
                    d_parts.append(f"{s.name}:{s.get_standard_uncertainty()}".encode())
            if d_parts:
                h = _hashlib2.sha256(b"".join(d_parts))
                meta['data_hash'] = h.hexdigest()[:16]
        except Exception:
            pass
    meta.update(metadata_kwargs)
    try:
        export_figure_package(fig, base, **meta)
    except Exception as exc:
        QMessageBox.critical(
            parent, "Export Error",
            f"Could not export figure package:\n\n{exc}"
        )
        return
    if hasattr(parent, '_status_bar'):
        parent._status_bar.showMessage(
            f"Figure package exported to {base}_*.{{png,svg,pdf,json}}", 8000
        )


def make_plot_toolbar_with_copy(canvas, fig, parent, method_context=""):
    """
    Create a toolbar row containing the matplotlib NavigationToolbar,
    'Copy to Clipboard', 'Copy Report-Quality', and 'Export Figure Package'
    buttons. Returns a QWidget containing all controls.

    Parameters
    ----------
    method_context : str
        Auto-populated into the figure metadata sidecar (e.g.,
        "RSS ASME V&V 20", "Monte Carlo JCGM 101").
    """
    toolbar_widget = QWidget(parent)
    toolbar_layout = QHBoxLayout(toolbar_widget)
    toolbar_layout.setContentsMargins(0, 0, 0, 0)
    toolbar_layout.setSpacing(4)

    nav_toolbar = NavigationToolbar(canvas, parent)
    toolbar_layout.addWidget(nav_toolbar)

    btn_style = (
        f"QPushButton {{ background-color: {DARK_COLORS['bg_widget']}; "
        f"color: {DARK_COLORS['fg']}; border: 1px solid {DARK_COLORS['border']}; "
        f"border-radius: 3px; padding: 4px 8px; font-size: 11px; }}"
        f"QPushButton:hover {{ border-color: {DARK_COLORS['accent']}; "
        f"color: {DARK_COLORS['accent']}; }}"
    )

    btn_copy = QPushButton("Copy to Clipboard")
    btn_copy.setToolTip(
        "Copy this chart as a draft image (150 DPI) to the clipboard.\n"
        "Paste directly into PowerPoint, Word, or email."
    )
    btn_copy.setMaximumWidth(140)
    btn_copy.setStyleSheet(btn_style)
    btn_copy.clicked.connect(lambda: copy_figure_to_clipboard(fig))
    toolbar_layout.addWidget(btn_copy)

    btn_rq = QPushButton("Copy Report-Quality")
    btn_rq.setToolTip(
        "Copy this chart at 300 DPI with light report theme.\n"
        "Suitable for formal report insertion."
    )
    btn_rq.setMaximumWidth(155)
    btn_rq.setStyleSheet(btn_style)
    btn_rq.clicked.connect(lambda: copy_report_quality_figure(fig))
    toolbar_layout.addWidget(btn_rq)

    btn_export = QPushButton("Export Figure Package...")
    btn_export.setToolTip(
        "Export PNG (300+600 DPI), SVG, PDF, and JSON metadata sidecar.\n"
        "Produces a complete regulatory-quality figure archive."
    )
    btn_export.setMaximumWidth(170)
    btn_export.setStyleSheet(btn_style)
    _mc = method_context  # capture for lambda
    btn_export.clicked.connect(
        lambda: _export_figure_package_dialog(fig, parent, method_context=_mc)
    )
    toolbar_layout.addWidget(btn_export)

    return toolbar_widget


# =============================================================================
# SECTION 6: DARK MODE STYLESHEET
# =============================================================================

def get_dark_stylesheet():
    """Generate a comprehensive dark mode stylesheet for PySide6."""
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
    QTableWidget::item:alternate {{
        background-color: {c['bg_alt']};
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
        width: 16px;
        height: 16px;
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
        width: 16px;
        height: 16px;
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
    QFrame[frameShape="4"] {{ /* HLine */
        color: {c['border']};
    }}
    QLabel {{
        color: {c['fg']};
    }}
    """


# =============================================================================
# SECTION 7: GUIDANCE WIDGET + TAB 1 (COMPARISON DATA TAB)
# =============================================================================

class GuidancePanel(QFrame):
    """
    Reusable panel for displaying color-coded guidance messages.

    Severity levels:
        'green'  - Acceptable / nominal condition
        'yellow' - Caution / review recommended
        'red'    - Warning / action required

    The panel uses a colored left border and icon to convey severity,
    with background tinting that harmonizes with the dark theme palette.
    """

    SEVERITY_CONFIG = {
        'green': {
            'border_color': DARK_COLORS['green'],
            'bg_color': '#1a2e1a',
            'icon': '\u2714',      # checkmark
            'label': 'OK',
        },
        'yellow': {
            'border_color': DARK_COLORS['yellow'],
            'bg_color': '#2e2a1a',
            'icon': '\u26A0',      # warning triangle
            'label': 'CAUTION',
        },
        'red': {
            'border_color': DARK_COLORS['red'],
            'bg_color': '#2e1a1a',
            'icon': '\u2716',      # cross mark
            'label': 'WARNING',
        },
    }

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self._severity = 'green'
        self._setup_ui()

    def _setup_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(10, 8, 10, 8)
        self._layout.setSpacing(4)

        # Header row: icon + title
        header_layout = QHBoxLayout()
        header_layout.setSpacing(6)
        self._icon_label = QLabel()
        self._icon_label.setFixedWidth(20)
        font = self._icon_label.font()
        font.setPointSize(12)
        self._icon_label.setFont(font)
        header_layout.addWidget(self._icon_label)

        self._title_label = QLabel(self._title)
        title_font = self._title_label.font()
        title_font.setBold(True)
        self._title_label.setFont(title_font)
        header_layout.addWidget(self._title_label)
        header_layout.addStretch()
        self._layout.addLayout(header_layout)

        # Message body
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
        """Update the guidance panel with a new message and severity level."""
        self._severity = severity
        self._message_label.setText(message)
        self._apply_severity(severity)

    def set_title(self, title: str):
        self._title = title
        self._title_label.setText(title)

    def clear(self):
        self._message_label.setText("")
        self._apply_severity('green')


class ComparisonDataTab(QWidget):
    """
    Tab 1: Comparison Data Tab.

    Allows users to load or enter CFD-to-flight-test comparison errors
    (E = S - D) and provides automated statistical analysis, distribution
    assessment, and visualization.

    References:
        - ASME V&V 20-2009 Section 2.4 (comparison error definition)
        - JCGM 100:2008 (GUM) Section 4.2 (Type A evaluation)
        - ASME PTC 19.1-2018 Section 7 (sample statistics)
    """

    data_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._comp_data = ComparisonData()
        self._stats = {}
        self._setup_ui()
        self._connect_signals()

    # -----------------------------------------------------------------
    # UI CONSTRUCTION
    # -----------------------------------------------------------------
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)

        # ---- LEFT PANEL (scrollable) ----
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setMinimumWidth(500)
        left_widget = QWidget()
        self._left_layout = QVBoxLayout(left_widget)
        self._left_layout.setSpacing(8)
        self._left_layout.setContentsMargins(6, 6, 6, 6)

        self._build_data_input_section()
        self._build_metadata_section()
        self._build_statistics_section()
        self._build_guidance_section()
        self._build_per_location_section()

        self._left_layout.addStretch()
        left_scroll.setWidget(left_widget)

        # ---- RIGHT PANEL (plots) ----
        right_widget = QWidget()
        self._right_layout = QVBoxLayout(right_widget)
        self._right_layout.setContentsMargins(4, 4, 4, 4)
        self._right_layout.setSpacing(4)
        self._build_plot_section()

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([650, 550])
        main_layout.addWidget(splitter)

    # -- Data Input Section --
    def _build_data_input_section(self):
        grp = QGroupBox("Data Input  (E = S \u2212 D)")
        grp.setToolTip(
            "Enter comparison errors E = S - D where S is the CFD simulation "
            "result and D is the flight-test (experimental) datum.\n"
            "[ASME V&V 20-2009 \u00a72.4]"
        )
        layout = QVBoxLayout(grp)

        # Buttons row
        btn_layout = QHBoxLayout()
        self._btn_import = QPushButton("Import CSV / Excel")
        self._btn_import.setToolTip(
            "Import transposed data: rows = sensor IDs (col 1 = sensor name),\n"
            "columns = flight-test conditions (row 1 = condition labels).\n"
            "Supported formats: .csv, .xlsx, .xls"
        )
        self._btn_paste = QPushButton("Paste from Clipboard")
        self._btn_paste.setToolTip(
            "Paste tab-delimited or comma-delimited data copied from a\n"
            "spreadsheet. Same transposed format as Import."
        )
        self._btn_clear = QPushButton("Clear Data")
        self._btn_clear.setToolTip("Remove all loaded comparison data.")

        btn_layout.addWidget(self._btn_import)
        btn_layout.addWidget(self._btn_paste)
        btn_layout.addWidget(self._btn_clear)
        layout.addLayout(btn_layout)

        # Table
        self._data_table = QTableWidget()
        self._data_table.setMinimumHeight(180)
        self._data_table.setAlternatingRowColors(True)
        self._data_table.setToolTip(
            "Editable table of comparison errors E = S - D.\n"
            "Rows represent sensor locations; columns represent flight-test conditions.\n"
            "[ASME V&V 20 \u00a72.4, PTC 19.1 \u00a77]"
        )
        style_table(self._data_table, stretch_col=0)
        layout.addWidget(self._data_table)

        self._left_layout.addWidget(grp)

    # -- Metadata Section --
    def _build_metadata_section(self):
        grp = QGroupBox("Metadata")
        form = QFormLayout(grp)
        form.setSpacing(6)

        self._lbl_n_conditions = QLabel("0")
        self._lbl_n_conditions.setToolTip("Number of flight-test operating conditions (columns).")
        form.addRow("Flight Conditions:", self._lbl_n_conditions)

        self._lbl_n_locations = QLabel("0")
        self._lbl_n_locations.setToolTip("Number of sensor measurement locations (rows).")
        form.addRow("Sensor Locations:", self._lbl_n_locations)

        self._lbl_n_total = QLabel("0")
        self._lbl_n_total.setToolTip(
            "Total sample count n used in statistical calculations.\n"
            "[ASME PTC 19.1 \u00a77]"
        )
        form.addRow("Total Sample Count (n):", self._lbl_n_total)

        self._chk_pooled = QCheckBox("Treat as pooled data")
        self._chk_pooled.setChecked(True)
        self._chk_pooled.setToolTip(
            "When checked, all sensor locations are pooled into a single\n"
            "sample for computing E\u0304 and s_E. This assumes the comparison\n"
            "error distribution is stationary across locations.\n"
            "[ASME V&V 20 \u00a72.4, GUM \u00a74.2]"
        )
        form.addRow(self._chk_pooled)

        note = QLabel(
            "Note: Pooling assumes the comparison error distribution is "
            "the same across all sensor locations. Verify using the "
            "per-location breakdown below."
        )
        note.setWordWrap(True)
        note.setStyleSheet(f"color: {DARK_COLORS['fg_dim']}; font-size: 11px;")
        form.addRow(note)

        self._cmb_unit = QComboBox()
        self._cmb_unit.addItems(UNIT_OPTIONS)
        self._cmb_unit.setCurrentText(DEFAULT_UNIT)
        self._cmb_unit.setToolTip(
            "Engineering unit for the comparison error values.\n"
            "Used for axis labels and report generation."
        )
        form.addRow("Unit:", self._cmb_unit)

        self._left_layout.addWidget(grp)

    # -- Statistics Section --
    def _build_statistics_section(self):
        grp = QGroupBox("Descriptive Statistics  (auto-updated)")
        grp.setToolTip(
            "Summary statistics computed from the pooled (or per-location) "
            "comparison error data E = S - D.\n"
            "[GUM \u00a74.2, ASME PTC 19.1 \u00a77]"
        )
        form = QFormLayout(grp)
        form.setSpacing(4)

        self._stat_labels = {}
        stat_fields = [
            ("mean", "Mean (\u0112)"),
            ("std", "Std Dev (s_E)"),
            ("se", "Standard Error (SE)"),
            ("min", "Min"),
            ("max", "Max"),
            ("range", "Range"),
            ("median", "Median"),
            ("p5", "P5 (5th percentile)"),
            ("p95", "P95 (95th percentile)"),
            ("skewness", "Skewness"),
            ("kurtosis", "Excess Kurtosis"),
            ("shapiro_p", "Shapiro-Wilk p-value"),
        ]
        for key, label in stat_fields:
            lbl = QLabel("\u2014")
            lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self._stat_labels[key] = lbl
            form.addRow(f"{label}:", lbl)

        # Shapiro-Wilk flag
        self._lbl_normality_flag = QLabel("")
        self._lbl_normality_flag.setWordWrap(True)
        form.addRow(self._lbl_normality_flag)

        self._left_layout.addWidget(grp)

    # -- Guidance Section --
    def _build_guidance_section(self):
        self._dist_guidance = GuidancePanel("Distribution Guidance")
        self._dist_guidance.setToolTip(
            "Automated assessment of whether the comparison error data is\n"
            "consistent with a normal distribution and whether normal\n"
            "k-factors are appropriate.\n"
            "[GUM \u00a74.3, JCGM 101:2008]"
        )
        self._left_layout.addWidget(self._dist_guidance)

        self._sample_guidance = GuidancePanel("Sample Size Guidance")
        self._sample_guidance.setToolTip(
            "Assessment of whether the sample size is adequate for\n"
            "reliable tolerance-bound estimation.\n"
            "[ASME V&V 20 \u00a76, PTC 19.1 \u00a77]"
        )
        self._left_layout.addWidget(self._sample_guidance)

    # -- Per-Location Breakdown (collapsible) --
    def _build_per_location_section(self):
        self._grp_locations = QGroupBox("Per-Location Breakdown")
        self._grp_locations.setCheckable(True)
        self._grp_locations.setChecked(False)
        self._grp_locations.setToolTip(
            "Statistics for each individual sensor location.\n"
            "Locations whose mean deviates more than 2\u03c3 from the\n"
            "overall mean are flagged as potential outlier locations.\n"
            "[ASME V&V 20 \u00a72.4]"
        )
        self._loc_layout = QVBoxLayout(self._grp_locations)
        self._loc_table = QTableWidget()
        self._loc_table.setColumnCount(4)
        self._loc_table.setHorizontalHeaderLabels(["Location", "Mean", "Std Dev", "Flag"])
        style_table(self._loc_table,
                    column_widths={0: 120, 1: 90, 2: 90, 3: 100},
                    stretch_col=0)
        self._loc_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._loc_layout.addWidget(self._loc_table)

        self._left_layout.addWidget(self._grp_locations)

    # -- Plot Section --
    def _build_plot_section(self):
        plt.rcParams.update(PLOT_STYLE)

        self._fig = Figure(figsize=(5.2, 7), dpi=100)
        self._fig.set_facecolor(PLOT_STYLE['figure.facecolor'])

        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumWidth(400)
        self._canvas.setMinimumHeight(500)

        # Toolbar with Copy to Clipboard button
        toolbar_row = make_plot_toolbar_with_copy(
            self._canvas, self._fig, self,
            method_context="Comparison Data (ASME V&V 20)")

        # Wrap canvas in a scroll area for horizontal scrolling
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        plot_container = QWidget()
        plot_lay = QVBoxLayout(plot_container)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        plot_lay.addWidget(self._canvas)
        plot_scroll.setWidget(plot_container)

        self._right_layout.addWidget(toolbar_row)
        self._right_layout.addWidget(plot_scroll)

    # -----------------------------------------------------------------
    # SIGNAL CONNECTIONS
    # -----------------------------------------------------------------
    def _connect_signals(self):
        self._btn_import.clicked.connect(self._on_import)
        self._btn_paste.clicked.connect(self._on_paste)
        self._btn_clear.clicked.connect(self._on_clear)
        self._data_table.cellChanged.connect(self._on_table_edited)
        self._chk_pooled.stateChanged.connect(self._on_pooled_changed)
        self._cmb_unit.currentTextChanged.connect(self._on_unit_changed)

    # -----------------------------------------------------------------
    # DATA LOADING
    # -----------------------------------------------------------------
    def _on_import(self):
        """Import comparison data from CSV or Excel file (transposed format)."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Comparison Error Data",
            "",
            "Data Files (*.csv *.xlsx *.xls);;CSV (*.csv);;Excel (*.xlsx *.xls);;All (*.*)",
        )
        if not path:
            return

        try:
            if path.lower().endswith('.csv'):
                self._load_csv(path)
            else:
                self._load_excel(path)
            audit_log.log_data_load(
                os.path.basename(path),
                f"Loaded {self._comp_data.data.shape[0]} locations x "
                f"{self._comp_data.data.shape[1]} conditions "
                f"({self._comp_data.flat_data().size} total values)",
            )
            self._populate_table()
            self.update_stats()
            self.update_plots()
            self.data_changed.emit()
        except Exception as exc:
            QMessageBox.critical(
                self, "Import Error",
                f"Failed to load data from file:\n{exc}\n\n{traceback.format_exc()}",
            )
            audit_log.log_warning(f"Import failed: {exc}")

    def _load_csv(self, path: str):
        """Parse a CSV file in transposed format."""
        import csv
        with open(path, 'r', newline='', encoding='utf-8-sig') as f:
            reader = csv.reader(f)
            rows = list(reader)
        self._parse_rows(rows, os.path.basename(path))

    def _load_excel(self, path: str):
        """Parse an Excel file in transposed format."""
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        ws = wb.active
        rows = []
        for row in ws.iter_rows(values_only=True):
            rows.append([str(c) if c is not None else '' for c in row])
        wb.close()
        self._parse_rows(rows, os.path.basename(path))

    def _parse_rows(self, rows: list, source_name: str = ""):
        """
        Parse a list of row-lists from CSV/Excel/clipboard into ComparisonData.

        Expected transposed format:
            Row 0:   [ "", cond_1, cond_2, ... ]
            Row 1:   [ sensor_name_1, val, val, ... ]
            Row 2:   [ sensor_name_2, val, val, ... ]
            ...
        """
        if len(rows) < 2:
            raise ValueError("Data must have at least a header row and one data row.")

        header = rows[0]
        # Condition names from columns 1..N of header row
        condition_names = [str(h).strip() for h in header[1:] if str(h).strip()]
        if not condition_names:
            # Assume no header; treat all as data
            condition_names = [f"Cond {j+1}" for j in range(len(header))]
            data_rows = rows
            sensor_names = [f"Loc {i+1}" for i in range(len(data_rows))]
            col_start = 0
        else:
            data_rows = rows[1:]
            sensor_names = []
            col_start = 1
            for r in data_rows:
                sensor_names.append(str(r[0]).strip() if r else f"Loc {len(sensor_names)+1}")

        n_sensors = len(data_rows)
        n_conds = len(condition_names)
        data_array = np.full((n_sensors, n_conds), np.nan)

        for i, row in enumerate(data_rows):
            for j in range(n_conds):
                idx = col_start + j
                if idx < len(row):
                    val_str = str(row[idx]).strip()
                    if val_str:
                        try:
                            data_array[i, j] = float(val_str)
                        except ValueError:
                            pass  # leave as NaN

        self._comp_data = ComparisonData(
            data=data_array,
            sensor_names=sensor_names,
            condition_names=condition_names,
            unit=self._cmb_unit.currentText(),
            is_pooled=self._chk_pooled.isChecked(),
        )

    def _on_paste(self):
        """Paste comparison data from the system clipboard."""
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text or not text.strip():
            QMessageBox.information(self, "Paste", "Clipboard is empty.")
            return

        try:
            lines = text.strip().split('\n')
            rows = []
            for line in lines:
                # Try tab first, then comma
                if '\t' in line:
                    rows.append(line.split('\t'))
                else:
                    rows.append(line.split(','))
            self._parse_rows(rows, "clipboard")
            audit_log.log_data_load(
                "clipboard",
                f"Pasted {self._comp_data.data.shape[0]} locations x "
                f"{self._comp_data.data.shape[1]} conditions "
                f"({self._comp_data.flat_data().size} total values)",
            )
            self._populate_table()
            self.update_stats()
            self.update_plots()
            self.data_changed.emit()
        except Exception as exc:
            QMessageBox.critical(
                self, "Paste Error",
                f"Failed to parse clipboard data:\n{exc}",
            )
            audit_log.log_warning(f"Paste failed: {exc}")

    def _on_clear(self):
        """Clear all comparison data."""
        self._comp_data = ComparisonData()
        self._data_table.blockSignals(True)
        self._data_table.setRowCount(0)
        self._data_table.setColumnCount(0)
        self._data_table.blockSignals(False)
        self._stats = {}
        self._update_metadata_labels()
        self._clear_stat_labels()
        self._dist_guidance.clear()
        self._sample_guidance.clear()
        self._loc_table.setRowCount(0)
        self._clear_plots()
        audit_log.log("DATA_CLEAR", "Comparison data cleared.")
        self.data_changed.emit()

    # -----------------------------------------------------------------
    # TABLE MANAGEMENT
    # -----------------------------------------------------------------
    def _populate_table(self):
        """Fill the QTableWidget from the current ComparisonData."""
        self._data_table.blockSignals(True)
        cd = self._comp_data
        if cd.data.size == 0:
            self._data_table.setRowCount(0)
            self._data_table.setColumnCount(0)
            self._data_table.blockSignals(False)
            return

        n_rows, n_cols = cd.data.shape
        self._data_table.setRowCount(n_rows)
        self._data_table.setColumnCount(n_cols)

        # Column headers = condition names
        col_headers = cd.condition_names if cd.condition_names else [
            f"Cond {j+1}" for j in range(n_cols)
        ]
        self._data_table.setHorizontalHeaderLabels(col_headers[:n_cols])

        # Row headers = sensor names
        row_headers = cd.sensor_names if cd.sensor_names else [
            f"Loc {i+1}" for i in range(n_rows)
        ]
        self._data_table.setVerticalHeaderLabels(row_headers[:n_rows])

        for i in range(n_rows):
            for j in range(n_cols):
                val = cd.data[i, j]
                if np.isnan(val):
                    item = QTableWidgetItem("")
                else:
                    item = QTableWidgetItem(f"{val:.6g}")
                self._data_table.setItem(i, j, item)

        self._data_table.blockSignals(False)
        self._update_metadata_labels()

    def _on_table_edited(self, row, col):
        """Handle user edits to the data table."""
        item = self._data_table.item(row, col)
        if item is None:
            return
        text = item.text().strip()
        if text == '':
            if row < self._comp_data.data.shape[0] and col < self._comp_data.data.shape[1]:
                self._comp_data.data[row, col] = np.nan
        else:
            try:
                val = float(text)
                if row < self._comp_data.data.shape[0] and col < self._comp_data.data.shape[1]:
                    self._comp_data.data[row, col] = val
            except ValueError:
                # Revert the cell to the current data-model value so the
                # display never shows invalid text like "abc".
                self._data_table.blockSignals(True)
                if row < self._comp_data.data.shape[0] and col < self._comp_data.data.shape[1]:
                    cur = self._comp_data.data[row, col]
                    item.setText("" if np.isnan(cur) else f"{cur:g}")
                else:
                    item.setText("")
                self._data_table.blockSignals(False)
                return  # No data change — skip update cycle

        self.update_stats()
        self.update_plots()
        self.data_changed.emit()

    def _update_metadata_labels(self):
        cd = self._comp_data
        if cd.data.size == 0:
            self._lbl_n_conditions.setText("0")
            self._lbl_n_locations.setText("0")
            self._lbl_n_total.setText("0")
            return
        n_rows, n_cols = cd.data.shape
        n_total = cd.flat_data().size
        self._lbl_n_conditions.setText(str(n_cols))
        self._lbl_n_locations.setText(str(n_rows))
        self._lbl_n_total.setText(str(n_total))

    def _on_pooled_changed(self, state):
        self._comp_data.is_pooled = bool(state)
        self.update_stats()
        self.update_plots()
        self.data_changed.emit()

    def _on_unit_changed(self, unit_text):
        self._comp_data.unit = unit_text
        self.update_plots()
        self.data_changed.emit()

    # -----------------------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------------------
    def update_stats(self):
        """Recalculate all descriptive statistics and update the UI."""
        flat = self._comp_data.flat_data()
        if flat.size == 0:
            self._stats = {}
            self._clear_stat_labels()
            self._dist_guidance.clear()
            self._sample_guidance.clear()
            self._loc_table.setRowCount(0)
            return

        self._stats = compute_descriptive_stats(flat)
        self._update_stat_labels()
        self._update_guidance()
        self._update_per_location()
        self._update_metadata_labels()

    def _update_stat_labels(self):
        s = self._stats
        if not s:
            self._clear_stat_labels()
            return

        fmt_map = {
            'mean': f"{s.get('mean', 0):.6g}",
            'std': f"{s.get('std', 0):.6g}",
            'se': f"{s.get('se', 0):.6g}",
            'min': f"{s.get('min', 0):.6g}",
            'max': f"{s.get('max', 0):.6g}",
            'range': f"{s.get('range', 0):.6g}",
            'median': f"{s.get('median', 0):.6g}",
            'p5': f"{s.get('p5', 0):.6g}",
            'p95': f"{s.get('p95', 0):.6g}",
            'skewness': f"{s.get('skewness', 0):.4f}",
            'kurtosis': f"{s.get('kurtosis', 0):.4f}",
        }
        p_sw = s.get('shapiro_p', None)
        if p_sw is not None:
            fmt_map['shapiro_p'] = f"{p_sw:.4f}"
        else:
            fmt_map['shapiro_p'] = "N/A (n < 3 or n > 5000)"

        for key, txt in fmt_map.items():
            if key in self._stat_labels:
                self._stat_labels[key].setText(txt)

        # Shapiro flag
        if p_sw is not None and p_sw < 0.05:
            self._lbl_normality_flag.setText(
                f"\u26A0 Shapiro-Wilk p = {p_sw:.4f} < 0.05: "
                f"Normality assumption may not hold. Review QQ plot and "
                f"consider distribution-free or Monte Carlo methods. "
                f"[GUM \u00a74.3, JCGM 101:2008]"
            )
            self._lbl_normality_flag.setStyleSheet(
                f"color: {DARK_COLORS['red']}; font-weight: bold;"
            )
        elif p_sw is not None:
            self._lbl_normality_flag.setText(
                f"Shapiro-Wilk p = {p_sw:.4f} \u2265 0.05: "
                f"Cannot reject normality at the 5% significance level."
            )
            self._lbl_normality_flag.setStyleSheet(
                f"color: {DARK_COLORS['green']};"
            )
        else:
            self._lbl_normality_flag.setText("")

    def _clear_stat_labels(self):
        for lbl in self._stat_labels.values():
            lbl.setText("\u2014")
        self._lbl_normality_flag.setText("")

    # -----------------------------------------------------------------
    # GUIDANCE
    # -----------------------------------------------------------------
    def _update_guidance(self):
        if not self._stats:
            self._dist_guidance.clear()
            self._sample_guidance.clear()
            return

        # Distribution guidance
        dist_msg, dist_sev = assess_distribution(self._stats)
        self._dist_guidance.set_guidance(dist_msg, dist_sev)

        # Sample size guidance
        n = self._stats.get('n', 0)
        if n > 0:
            ss_msg, ss_sev = assess_sample_size(n)
            self._sample_guidance.set_guidance(ss_msg, ss_sev)
        else:
            self._sample_guidance.clear()

    # -----------------------------------------------------------------
    # PER-LOCATION BREAKDOWN
    # -----------------------------------------------------------------
    def _update_per_location(self):
        loc_stats = self._comp_data.per_location_stats()
        self._loc_table.setRowCount(len(loc_stats))
        if not loc_stats:
            return

        overall_mean = self._stats.get('mean', 0.0)
        overall_std = self._stats.get('std', 1.0)
        if overall_std == 0:
            overall_std = 1.0

        for i, ls in enumerate(loc_stats):
            name_item = QTableWidgetItem(ls['name'])
            mean_item = QTableWidgetItem(f"{ls['mean']:.6g}")
            std_item = QTableWidgetItem(f"{ls['std']:.6g}")

            deviation = abs(ls['mean'] - overall_mean)
            if deviation > 2.0 * overall_std:
                flag_item = QTableWidgetItem(
                    f"\u26A0 |mean - \u0112| = {deviation:.4g} > 2\u03c3 = {2*overall_std:.4g}"
                )
                flag_item.setForeground(QColor(DARK_COLORS['red']))
                warn_key = f"loc_outlier_{ls['name']}"
                if not hasattr(self, '_logged_warnings'):
                    self._logged_warnings = set()
                if warn_key not in self._logged_warnings:
                    self._logged_warnings.add(warn_key)
                    audit_log.log_warning(
                        f"Location '{ls['name']}' mean ({ls['mean']:.4g}) deviates "
                        f"> 2\u03c3 from overall mean ({overall_mean:.4g}). "
                        f"Pooling assumption may be violated."
                    )
            else:
                flag_item = QTableWidgetItem("OK")
                flag_item.setForeground(QColor(DARK_COLORS['green']))

            self._loc_table.setItem(i, 0, name_item)
            self._loc_table.setItem(i, 1, mean_item)
            self._loc_table.setItem(i, 2, std_item)
            self._loc_table.setItem(i, 3, flag_item)

    # -----------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------
    def update_plots(self):
        """Refresh all matplotlib plots with the current data."""
        self._fig.clear()
        flat = self._comp_data.flat_data()
        unit = self._comp_data.unit

        if flat.size < 2:
            ax = self._fig.add_subplot(111)
            ax.text(
                0.5, 0.5, "No data loaded.\nImport or paste comparison errors.",
                ha='center', va='center', fontsize=11,
                color=DARK_COLORS['fg_dim'],
                transform=ax.transAxes,
            )
            ax.set_facecolor(PLOT_STYLE['axes.facecolor'])
            self._canvas.draw_idle()
            return

        loc_stats = self._comp_data.per_location_stats()
        has_multiple_locations = len(loc_stats) > 1

        if has_multiple_locations:
            gs = self._fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
            ax_hist = self._fig.add_subplot(gs[0, 0])
            ax_qq = self._fig.add_subplot(gs[0, 1])
            ax_box = self._fig.add_subplot(gs[1, :])
        else:
            gs = self._fig.add_gridspec(1, 2, wspace=0.30)
            ax_hist = self._fig.add_subplot(gs[0, 0])
            ax_qq = self._fig.add_subplot(gs[0, 1])
            ax_box = None

        # -- Histogram with normal PDF overlay --
        n_bins = max(int(np.sqrt(len(flat))), 5)
        n_bins = min(n_bins, 50)
        ax_hist.hist(
            flat, bins=n_bins, density=True, alpha=0.7,
            color=DARK_COLORS['accent'], edgecolor=DARK_COLORS['border'],
            label='Data',
        )
        mu, sigma = float(np.mean(flat)), float(np.std(flat, ddof=1))
        if sigma > 0:
            x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 200)
            pdf_vals = norm.pdf(x_range, mu, sigma)
            ax_hist.plot(
                x_range, pdf_vals, color=DARK_COLORS['red'],
                linewidth=2, label=f'N({mu:.3g}, {sigma:.3g}\u00b2)',
            )
        ax_hist.set_xlabel(f"E = S \u2212 D  [{unit}]")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title("Histogram of Comparison Errors", fontsize=9)
        ax_hist.legend(fontsize=6.5, loc='upper right')
        ax_hist.grid(True, alpha=0.3)

        # -- QQ plot --
        sorted_data = np.sort(flat)
        n = len(sorted_data)
        theoretical_quantiles = norm.ppf(
            (np.arange(1, n + 1) - 0.375) / (n + 0.25)
        )
        ax_qq.scatter(
            theoretical_quantiles, sorted_data,
            s=18, color=DARK_COLORS['accent'], alpha=0.8,
            edgecolors='none', label='Data',
        )
        # Reference line
        q25, q75 = np.percentile(flat, [25, 75])
        z25, z75 = norm.ppf(0.25), norm.ppf(0.75)
        if z75 != z25:
            slope = (q75 - q25) / (z75 - z25)
            intercept = q25 - slope * z25
            line_x = np.array([theoretical_quantiles[0], theoretical_quantiles[-1]])
            ax_qq.plot(
                line_x, intercept + slope * line_x,
                color=DARK_COLORS['red'], linewidth=1.5,
                linestyle='--', label='Reference line',
            )
        ax_qq.set_xlabel("Theoretical Quantiles (Normal)")
        ax_qq.set_ylabel(f"Sample Quantiles  [{unit}]")
        ax_qq.set_title("Normal QQ Plot", fontsize=9)
        ax_qq.legend(fontsize=6.5, loc='upper left')
        ax_qq.grid(True, alpha=0.3)

        # -- Box plot of locations (if multiple) --
        if ax_box is not None and has_multiple_locations:
            box_data = []
            box_labels = []
            for ls in loc_stats:
                row_idx = ls.get('orig_row', 0)
                if row_idx < self._comp_data.data.shape[0]:
                    row = self._comp_data.data[row_idx, :]
                    row = row[~np.isnan(row)]
                    if row.size > 0:
                        box_data.append(row)
                        box_labels.append(ls['name'])

            if box_data:
                bp = ax_box.boxplot(
                    box_data, tick_labels=box_labels, patch_artist=True,
                    boxprops=dict(facecolor=DARK_COLORS['bg_input'],
                                  edgecolor=DARK_COLORS['accent']),
                    whiskerprops=dict(color=DARK_COLORS['fg_dim']),
                    capprops=dict(color=DARK_COLORS['fg_dim']),
                    medianprops=dict(color=DARK_COLORS['yellow'], linewidth=2),
                    flierprops=dict(marker='o', markerfacecolor=DARK_COLORS['red'],
                                    markersize=4, alpha=0.7),
                )
                ax_box.axhline(
                    y=mu, color=DARK_COLORS['green'], linewidth=1,
                    linestyle='--', alpha=0.7, label=f'Overall \u0112 = {mu:.3g}',
                )
                ax_box.set_xlabel("Sensor Location")
                ax_box.set_ylabel(f"E = S \u2212 D  [{unit}]")
                ax_box.set_title("Comparison Error by Location", fontsize=9)
                ax_box.legend(fontsize=6.5)
                ax_box.grid(True, axis='y', alpha=0.3)

                # Rotate labels if many locations
                if len(box_labels) > 6:
                    ax_box.tick_params(axis='x', rotation=45)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._fig.tight_layout()
            except Exception:
                pass
        self._canvas.draw_idle()

    def _clear_plots(self):
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(
            0.5, 0.5, "No data loaded.\nImport or paste comparison errors.",
            ha='center', va='center', fontsize=11,
            color=DARK_COLORS['fg_dim'],
            transform=ax.transAxes,
        )
        ax.set_facecolor(PLOT_STYLE['axes.facecolor'])
        self._canvas.draw_idle()

    # -----------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------
    def set_comparison_data(self, comp_data: ComparisonData):
        """
        Receive comparison data from an external source (e.g., session load).

        Parameters:
            comp_data: A ComparisonData instance containing the error matrix.
        """
        self._comp_data = comp_data
        self._chk_pooled.setChecked(comp_data.is_pooled)
        idx = self._cmb_unit.findText(comp_data.unit)
        if idx >= 0:
            self._cmb_unit.setCurrentIndex(idx)
        self._populate_table()
        self.update_stats()
        self.update_plots()
        audit_log.log("DATA_SET", "Comparison data set programmatically.",
                       f"Shape: {comp_data.data.shape}, n={comp_data.flat_data().size}")

    def get_comparison_data(self) -> ComparisonData:
        """
        Provide the current comparison data to the analysis engine.

        Returns:
            ComparisonData instance with the current error matrix, metadata,
            and pooling flag.
        """
        self._comp_data.is_pooled = self._chk_pooled.isChecked()
        self._comp_data.unit = self._cmb_unit.currentText()
        return self._comp_data


# =============================================================================
# SECTION 8: TAB 2 (UNCERTAINTY SOURCES TAB)
# =============================================================================

class _MiniDistributionCanvas(FigureCanvas):
    """
    Small embedded matplotlib canvas for showing distribution fit previews.

    Displays a histogram of the source data overlaid with the top two fitted
    probability density functions (PDFs) from fit_distributions().
    """

    def __init__(self, parent=None, width=3.6, height=2.2):
        with plt.rc_context(PLOT_STYLE):
            self._fig = Figure(figsize=(width, height), dpi=90)
        super().__init__(self._fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(int(height * 90))
        self.setMaximumHeight(int(height * 90) + 20)

    def plot_fit(self, data, fit_results):
        """
        Draw histogram of *data* and overlay the top two fitted PDFs.

        Parameters:
            data: 1-D array of raw sample values.
            fit_results: list returned by fit_distributions(), already sorted
                         best-fit-first.
        """
        self._fig.clear()
        if data is None or len(data) < 3 or not fit_results:
            ax = self._fig.add_subplot(111)
            ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center',
                    fontsize=10, color=DARK_COLORS['fg_dim'],
                    transform=ax.transAxes)
            ax.set_facecolor(PLOT_STYLE['axes.facecolor'])
            self.draw_idle()
            return

        ax = self._fig.add_subplot(111)
        data = np.asarray(data, dtype=float)

        # Histogram
        n_bins = min(max(int(np.sqrt(len(data))), 8), 40)
        ax.hist(data, bins=n_bins, density=True, alpha=0.45,
                color=DARK_COLORS['accent'], edgecolor=DARK_COLORS['border'],
                label='Data')

        # Overlay top-2 fitted PDFs
        x_range = np.linspace(data.min() - 0.15 * np.ptp(data),
                              data.max() + 0.15 * np.ptp(data), 300)
        colors_pdf = [DARK_COLORS['green'], DARK_COLORS['orange']]
        dist_map = {
            'Normal': stats.norm, 'Uniform': stats.uniform,
            'Logistic': stats.logistic, 'Laplace': stats.laplace,
            'Lognormal': stats.lognorm, 'Weibull': stats.weibull_min,
            'Exponential': stats.expon, 'Triangular': stats.triang,
        }

        for idx, res in enumerate(fit_results[:2]):
            dist_cls = dist_map.get(res['name'])
            if dist_cls is None:
                continue
            try:
                shift = res.get('shift', 0)
                if shift:
                    pdf_vals = dist_cls.pdf(x_range + shift, *res['params'])
                else:
                    pdf_vals = dist_cls.pdf(x_range, *res['params'])
                lbl = f"{res['name']} (GOF p={res.get('gof_p', 0.0):.3f})"
                ax.plot(x_range, pdf_vals, linewidth=1.6,
                        color=colors_pdf[idx], label=lbl)
            except Exception:
                pass

        ax.set_title("Distribution Fit Preview", fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=7)
        ax.set_facecolor(PLOT_STYLE['axes.facecolor'])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._fig.tight_layout(pad=0.8)
            except Exception:
                pass
        self.draw_idle()

    def clear_plot(self):
        self._fig.clear()
        self.draw_idle()


# ---- Input-type-specific panels ----

class _TabularDataPanel(QWidget):
    """Panel shown when Input Type is 'Tabular Data'."""

    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: List[float] = []
        self._fit_results: List[dict] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Import buttons
        btn_row = QHBoxLayout()
        self._btn_csv = QPushButton("Import CSV")
        self._btn_csv.setToolTip(
            "Import a single-column CSV file of raw measurements.\n"
            "One value per line. Header row is optional."
        )
        self._btn_paste = QPushButton("Paste from Clipboard")
        self._btn_paste.setToolTip(
            "Paste newline- or comma-separated values from a spreadsheet."
        )
        btn_row.addWidget(self._btn_csv)
        btn_row.addWidget(self._btn_paste)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Data table
        self._table = QTableWidget()
        self._table.setColumnCount(1)
        self._table.setHorizontalHeaderLabels(["Value"])
        style_table(self._table, stretch_col=0)
        self._table.setMinimumHeight(120)
        self._table.setMaximumHeight(200)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

        # Stats summary
        stats_grp = QGroupBox("Auto-Computed Statistics")
        stats_form = QFormLayout(stats_grp)
        stats_form.setSpacing(3)
        self._lbl_mean = QLabel("\u2014")
        self._lbl_sigma = QLabel("\u2014")
        self._lbl_n = QLabel("\u2014")
        self._lbl_dof = QLabel("\u2014")
        stats_form.addRow("Mean:", self._lbl_mean)
        stats_form.addRow("\u03c3 (1\u03c3):", self._lbl_sigma)
        stats_form.addRow("n:", self._lbl_n)
        stats_form.addRow("DOF:", self._lbl_dof)
        layout.addWidget(stats_grp)

        # Centered-on-zero checkbox
        self._chk_centered = QCheckBox("Is this data centered on zero?")
        self._chk_centered.setChecked(True)
        self._chk_centered.setToolTip(
            "Check if the data represents deviations from a reference.\n"
            "If unchecked, the mean will be used as a systematic offset."
        )
        layout.addWidget(self._chk_centered)

        # Fit button and guidance
        fit_row = QHBoxLayout()
        self._btn_fit = QPushButton("Auto-Fit Distribution")
        self._btn_fit.setToolTip(
            "Fit candidate distributions to the data and rank by\n"
            "GOF p-value: bootstrap AD by default, with KS screening\n"
            "used in sparse/fast mode and KS fallback if bootstrap fails.\n"
            "[GUM \u00a74.3, JCGM 101]"
        )
        fit_row.addWidget(self._btn_fit)
        fit_row.addStretch()
        layout.addLayout(fit_row)

        self._fit_guidance = GuidancePanel("Distribution Fit Result")
        self._fit_guidance.setVisible(False)
        layout.addWidget(self._fit_guidance)

        # Mini plot
        self._mini_canvas = _MiniDistributionCanvas(self)
        self._mini_canvas.setVisible(False)
        layout.addWidget(self._mini_canvas)

        # Small-sample warning
        self._small_sample_warn = GuidancePanel("Sample Size")
        self._small_sample_warn.setVisible(False)
        layout.addWidget(self._small_sample_warn)

        # Connect
        self._btn_csv.clicked.connect(self._import_csv)
        self._btn_paste.clicked.connect(self._paste_clipboard)
        self._btn_fit.clicked.connect(self._auto_fit)
        self._chk_centered.toggled.connect(lambda: self.value_changed.emit())

    # -- data helpers --
    def _import_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Tabular Data", "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )
        if not path:
            return
        try:
            raw = np.genfromtxt(path, delimiter=',', skip_header=0)
            raw = raw.flatten()
            raw = raw[~np.isnan(raw)]
            if raw.size == 0:
                # Try with header skip
                raw = np.genfromtxt(path, delimiter=',', skip_header=1)
                raw = raw.flatten()
                raw = raw[~np.isnan(raw)]
            self._set_data(raw.tolist())
            audit_log.log_data_load(path, f"n={len(self._data)} values")
        except Exception as exc:
            QMessageBox.warning(self, "Import Error", f"Failed to read file:\n{exc}")

    def _paste_clipboard(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text or not text.strip():
            QMessageBox.information(self, "Clipboard Empty",
                                   "No data found on clipboard.")
            return
        values = []
        for token in text.replace(',', '\n').replace('\t', '\n').split('\n'):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                pass
        if not values:
            QMessageBox.warning(self, "Parse Error",
                                "No numeric values found in clipboard text.")
            return
        self._set_data(values)
        audit_log.log_data_load("clipboard", f"n={len(values)} values pasted")

    def _set_data(self, values: List[float]):
        self._data = values
        self._populate_table()
        self._update_stats()
        self._check_sample_size()
        self.value_changed.emit()

    def _populate_table(self):
        self._table.setRowCount(len(self._data))
        for i, v in enumerate(self._data):
            item = QTableWidgetItem(f"{v:.6g}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 0, item)

    def _update_stats(self):
        if not self._data or len(self._data) < 1:
            self._lbl_mean.setText("\u2014")
            self._lbl_sigma.setText("\u2014")
            self._lbl_n.setText("0")
            self._lbl_dof.setText("\u2014")
            return
        arr = np.array(self._data, dtype=float)
        n = len(arr)
        self._lbl_n.setText(str(n))
        self._lbl_mean.setText(f"{np.mean(arr):.6g}")
        if n > 1:
            s = float(np.std(arr, ddof=1))
            self._lbl_sigma.setText(f"{s:.6g}")
            self._lbl_dof.setText(str(n - 1))
        else:
            self._lbl_sigma.setText("\u2014")
            self._lbl_dof.setText("\u2014")

    def _check_sample_size(self):
        n = len(self._data)
        if n > 0 and n < 10:
            self._small_sample_warn.set_guidance(
                f"Small sample (n={n}) \u2014 distribution shape cannot be "
                f"reliably determined. Consider increasing sample size if "
                f"possible. [GUM \u00a74.3]",
                'yellow'
            )
            self._small_sample_warn.setVisible(True)
        else:
            self._small_sample_warn.setVisible(False)

    def _auto_fit(self):
        if not self._data or len(self._data) < 5:
            QMessageBox.information(
                self, "Insufficient Data",
                "At least 5 data points are needed for distribution fitting."
            )
            return
        self._fit_results = fit_distributions(self._data)
        if self._fit_results:
            best = self._fit_results[0]
            self._fit_guidance.set_guidance(best['recommendation'],
                                           'green' if best.get('passed_gof') else 'yellow')
            self._fit_guidance.setVisible(True)
            self._mini_canvas.plot_fit(self._data, self._fit_results)
            self._mini_canvas.setVisible(True)
            audit_log.log_computation(
                "AUTO_FIT",
                f"Best fit: {best['name']} (GOF p={best.get('gof_p', 0.0):.4f})"
            )
        else:
            self._fit_guidance.set_guidance(
                "Could not fit any distribution to the data.", 'yellow')
            self._fit_guidance.setVisible(True)
            self._mini_canvas.setVisible(False)
        self.value_changed.emit()

    # -- public --
    def get_data(self) -> List[float]:
        return list(self._data)

    def set_data(self, values: List[float]):
        self._set_data(values)

    def is_centered(self) -> bool:
        return self._chk_centered.isChecked()

    def set_centered(self, val: bool):
        self._chk_centered.setChecked(val)

    def get_fit_results(self) -> List[dict]:
        return self._fit_results

    def get_recommended_distribution(self) -> str:
        """Return the name of the best-fit distribution, or empty string."""
        if self._fit_results and self._fit_results[0].get('passed_gof'):
            return self._fit_results[0]['name']
        return ""


class _SigmaValuePanel(QWidget):
    """Panel shown when Input Type is 'Sigma Value Only'."""

    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Sigma value
        self._spn_sigma = QDoubleSpinBox()
        self._spn_sigma.setRange(0.0, 1e9)
        self._spn_sigma.setDecimals(6)
        self._spn_sigma.setSingleStep(0.01)
        self._spn_sigma.setToolTip(
            "Enter the standard uncertainty value as stated by the source.\n"
            "Select the Sigma Basis below to specify how the value was defined.\n"
            "[PTC 19.1 Type B evaluation]"
        )
        layout.addRow("\u03c3 value:", self._spn_sigma)

        # Sigma basis
        self._cmb_basis = QComboBox()
        self._cmb_basis.addItems(SIGMA_BASIS_OPTIONS)
        self._cmb_basis.setToolTip(
            "How was the \u03c3 value defined by the data source?\n"
            "For supplier specifications, verify whether the stated\n"
            "tolerance is 1\u03c3, 2\u03c3, or a bounding value.\n"
            "[PTC 19.1 Type B]"
        )
        layout.addRow("Sigma Basis:", self._cmb_basis)

        # Sample size
        size_row = QHBoxLayout()
        self._spn_sample = QSpinBox()
        self._spn_sample.setRange(0, 1_000_000)
        self._spn_sample.setToolTip(
            "Number of samples used to determine this \u03c3 value.\n"
            "Set to 0 or check 'Supplier/Reference' if unknown."
        )
        size_row.addWidget(self._spn_sample)
        self._chk_supplier = QCheckBox("Supplier/Reference (DOF = \u221e)")
        self._chk_supplier.setToolTip(
            "Check if this uncertainty comes from a supplier specification,\n"
            "calibration certificate, or reference standard. This sets\n"
            "DOF = \u221e (Type B evaluation). [GUM \u00a74.3, PTC 19.1 \u00a74]"
        )
        size_row.addWidget(self._chk_supplier)
        layout.addRow("Sample Size:", size_row)

        # Converted 1-sigma label
        self._lbl_converted = QLabel("\u2014")
        self._lbl_converted.setToolTip(
            "The equivalent 1\u03c3 standard uncertainty after applying\n"
            "the sigma basis conversion."
        )
        self._lbl_converted.setStyleSheet(
            f"font-weight: bold; color: {DARK_COLORS['accent']};"
        )
        layout.addRow("Converted 1\u03c3:", self._lbl_converted)

        # Sigma basis warning
        self._basis_warn = GuidancePanel("Sigma Basis")
        self._basis_warn.setVisible(False)
        layout.addRow(self._basis_warn)

        # ---- Asymmetric uncertainty section ----
        self._chk_asymmetric = QCheckBox("Asymmetric \u03c3\u207a / \u03c3\u207b")
        self._chk_asymmetric.setToolTip(
            "Enable when sensitivity results differ in the positive\n"
            "and negative directions. Uses effective \u03c3 per GUM \u00a74.3.8:\n"
            "\u03c3_eff = \u221a((\u03c3\u207a\u00b2 + \u03c3\u207b\u00b2) / 2)\n\n"
            "MC propagation uses a Split Gaussian (different spread\n"
            "above/below the mean) per Barlow (2004) for more\n"
            "rigorous asymmetric treatment."
        )
        layout.addRow(self._chk_asymmetric)

        # Asymmetric detail frame (hidden by default)
        self._asym_frame = QFrame()
        self._asym_frame.setVisible(False)
        asym_lay = QFormLayout(self._asym_frame)
        asym_lay.setContentsMargins(8, 4, 0, 4)
        asym_lay.setSpacing(4)

        self._spn_sigma_upper = QDoubleSpinBox()
        self._spn_sigma_upper.setRange(0.0, 1e9)
        self._spn_sigma_upper.setDecimals(6)
        self._spn_sigma_upper.setSingleStep(0.01)
        self._spn_sigma_upper.setToolTip("Uncertainty in the positive direction (\u03c3\u207a)")
        asym_lay.addRow("\u03c3\u207a (upper):", self._spn_sigma_upper)

        self._spn_sigma_lower = QDoubleSpinBox()
        self._spn_sigma_lower.setRange(0.0, 1e9)
        self._spn_sigma_lower.setDecimals(6)
        self._spn_sigma_lower.setSingleStep(0.01)
        self._spn_sigma_lower.setToolTip("Uncertainty in the negative direction (\u03c3\u207b)")
        asym_lay.addRow("\u03c3\u207b (lower):", self._spn_sigma_lower)

        self._chk_one_sided = QCheckBox("One-sided only")
        self._chk_one_sided.setToolTip(
            "Check if only one perturbation direction was tested.\n"
            "Mirror assumption: \u03c3_missing = \u03c3_observed."
        )
        asym_lay.addRow(self._chk_one_sided)

        self._cmb_one_sided_dir = QComboBox()
        self._cmb_one_sided_dir.addItems(["upper", "lower"])
        self._cmb_one_sided_dir.setToolTip("Which direction was actually tested?")
        self._cmb_one_sided_dir.setEnabled(False)
        asym_lay.addRow("Tested direction:", self._cmb_one_sided_dir)

        self._lbl_eff_sigma = QLabel("\u2014")
        self._lbl_eff_sigma.setStyleSheet(
            f"font-weight: bold; color: {DARK_COLORS['accent']};"
        )
        self._lbl_eff_sigma.setToolTip(
            "Effective \u03c3 = \u221a((\u03c3\u207a\u00b2 + \u03c3\u207b\u00b2) / 2)"
        )
        asym_lay.addRow("Effective \u03c3:", self._lbl_eff_sigma)

        layout.addRow(self._asym_frame)

        # Connections
        self._spn_sigma.valueChanged.connect(self._on_changed)
        self._cmb_basis.currentIndexChanged.connect(self._on_changed)
        self._spn_sample.valueChanged.connect(self._on_changed)
        self._chk_supplier.toggled.connect(self._on_supplier_toggled)
        self._chk_asymmetric.toggled.connect(self._on_asymmetric_toggled)
        self._spn_sigma_upper.valueChanged.connect(self._on_asym_changed)
        self._spn_sigma_lower.valueChanged.connect(self._on_asym_changed)
        self._chk_one_sided.toggled.connect(self._on_one_sided_toggled)
        self._cmb_one_sided_dir.currentIndexChanged.connect(self._on_asym_changed)

    def _on_supplier_toggled(self, checked):
        self._spn_sample.setEnabled(not checked)
        if checked:
            self._spn_sample.setValue(0)
        self._on_changed()

    def _on_asymmetric_toggled(self, checked):
        self._asym_frame.setVisible(checked)
        # When asymmetric is ON, hide the symmetric σ row
        self._spn_sigma.setVisible(not checked)
        if not checked:
            self._on_changed()
        else:
            self._on_asym_changed()

    def _on_one_sided_toggled(self, checked):
        self._cmb_one_sided_dir.setEnabled(checked)
        if checked:
            # Disable the non-tested direction spinner
            direction = self._cmb_one_sided_dir.currentText()
            self._spn_sigma_upper.setEnabled(direction == "upper")
            self._spn_sigma_lower.setEnabled(direction == "lower")
        else:
            self._spn_sigma_upper.setEnabled(True)
            self._spn_sigma_lower.setEnabled(True)
        self._on_asym_changed()

    def _on_asym_changed(self):
        sp = self._spn_sigma_upper.value()
        sm = self._spn_sigma_lower.value()
        if self._chk_one_sided.isChecked():
            direction = self._cmb_one_sided_dir.currentText()
            if direction == "upper":
                sm = sp  # mirror
                self._spn_sigma_upper.setEnabled(True)
                self._spn_sigma_lower.setEnabled(False)
            else:
                sp = sm  # mirror
                self._spn_sigma_upper.setEnabled(False)
                self._spn_sigma_lower.setEnabled(True)
        eff = float(np.sqrt((sp ** 2 + sm ** 2) / 2.0))
        self._lbl_eff_sigma.setText(f"{eff:.6g}")
        self.value_changed.emit()

    def _on_changed(self):
        raw = self._spn_sigma.value()
        basis = self._cmb_basis.currentText()
        converted = sigma_from_basis(raw, basis)
        self._lbl_converted.setText(f"{converted:.6g}")

        # Basis warning
        if basis == "Assumed 1\u03c3 (unverified)":
            self._basis_warn.set_guidance(
                "Sigma basis is unverified. Confirm with supplier "
                "documentation. [PTC 19.1 \u00a74]",
                'yellow'
            )
            self._basis_warn.setVisible(True)
        else:
            self._basis_warn.setVisible(False)

        self.value_changed.emit()

    # -- public --
    def get_sigma(self) -> float:
        return self._spn_sigma.value()

    def set_sigma(self, val: float):
        self._spn_sigma.setValue(val)

    def get_basis(self) -> str:
        return self._cmb_basis.currentText()

    def set_basis(self, val: str):
        idx = self._cmb_basis.findText(val)
        if idx >= 0:
            self._cmb_basis.setCurrentIndex(idx)

    def get_sample_size(self) -> int:
        return self._spn_sample.value()

    def set_sample_size(self, val: int):
        self._spn_sample.setValue(val)

    def is_supplier(self) -> bool:
        return self._chk_supplier.isChecked()

    def set_supplier(self, val: bool):
        self._chk_supplier.setChecked(val)

    def get_converted_sigma(self) -> float:
        return sigma_from_basis(self._spn_sigma.value(),
                                self._cmb_basis.currentText())

    # -- asymmetric public API --
    def is_asymmetric(self) -> bool:
        return self._chk_asymmetric.isChecked()

    def set_asymmetric(self, val: bool):
        self._chk_asymmetric.setChecked(val)

    def get_sigma_upper(self) -> float:
        return self._spn_sigma_upper.value()

    def set_sigma_upper(self, val: float):
        self._spn_sigma_upper.setValue(val)

    def get_sigma_lower(self) -> float:
        return self._spn_sigma_lower.value()

    def set_sigma_lower(self, val: float):
        self._spn_sigma_lower.setValue(val)

    def is_one_sided(self) -> bool:
        return self._chk_one_sided.isChecked()

    def set_one_sided(self, val: bool):
        self._chk_one_sided.setChecked(val)

    def get_one_sided_direction(self) -> str:
        return self._cmb_one_sided_dir.currentText()

    def set_one_sided_direction(self, val: str):
        idx = self._cmb_one_sided_dir.findText(val)
        if idx >= 0:
            self._cmb_one_sided_dir.setCurrentIndex(idx)


class _TolerancePanel(QWidget):
    """Panel shown when Input Type is 'Tolerance/Expanded Value'."""

    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Tolerance value
        self._spn_tol = QDoubleSpinBox()
        self._spn_tol.setRange(0.0, 1e9)
        self._spn_tol.setDecimals(6)
        self._spn_tol.setSingleStep(0.01)
        self._spn_tol.setToolTip(
            "Enter the expanded uncertainty or tolerance band value.\n"
            "This will be divided by k to obtain the 1\u03c3 standard uncertainty."
        )
        layout.addRow("Tolerance value:", self._spn_tol)

        # k value
        k_row = QHBoxLayout()
        self._spn_k = QDoubleSpinBox()
        self._spn_k.setRange(0.001, 100.0)
        self._spn_k.setDecimals(4)
        self._spn_k.setValue(2.0)
        self._spn_k.setSingleStep(0.1)
        self._spn_k.setToolTip(
            "Coverage factor k used to expand from standard uncertainty.\n"
            "Common values: k=2 (95% normal), k=3 (99.7% normal).\n"
            "[GUM \u00a76.3]"
        )
        k_row.addWidget(self._spn_k)

        self._chk_unknown_k = QCheckBox("I don't know k")
        self._chk_unknown_k.setToolTip(
            "If k is unknown, provide the sample size n and the tool\n"
            "will compute k from the Student-t distribution. [GUM Annex G]"
        )
        k_row.addWidget(self._chk_unknown_k)
        layout.addRow("k used to expand:", k_row)

        # n for computing k (hidden by default)
        self._lbl_n_for_k = QLabel("n (for computing k):")
        self._spn_n_for_k = QSpinBox()
        self._spn_n_for_k.setRange(2, 1_000_000)
        self._spn_n_for_k.setValue(30)
        self._spn_n_for_k.setToolTip(
            "Sample size used to determine k via Student-t.\n"
            "k = t_{n-1, 0.975} for 95% two-sided coverage. [GUM Annex G]"
        )
        self._lbl_n_for_k.setVisible(False)
        self._spn_n_for_k.setVisible(False)
        layout.addRow(self._lbl_n_for_k, self._spn_n_for_k)

        # Back-calculated sigma
        self._lbl_sigma = QLabel("\u2014")
        self._lbl_sigma.setStyleSheet(
            f"font-weight: bold; color: {DARK_COLORS['accent']};"
        )
        self._lbl_sigma.setToolTip(
            "Back-calculated standard uncertainty: \u03c3 = tolerance / k"
        )
        layout.addRow("Back-calculated \u03c3:", self._lbl_sigma)

        # Connections
        self._spn_tol.valueChanged.connect(self._on_changed)
        self._spn_k.valueChanged.connect(self._on_changed)
        self._spn_n_for_k.valueChanged.connect(self._on_k_from_n)
        self._chk_unknown_k.toggled.connect(self._on_unknown_k_toggled)

    def _on_unknown_k_toggled(self, checked):
        self._spn_k.setEnabled(not checked)
        self._lbl_n_for_k.setVisible(checked)
        self._spn_n_for_k.setVisible(checked)
        if checked:
            self._on_k_from_n()
        else:
            self._on_changed()

    def _on_k_from_n(self):
        n = self._spn_n_for_k.value()
        if n >= 2:
            k = float(t_dist.ppf(0.975, df=n - 1))
            self._spn_k.setValue(round(k, 4))
        self._on_changed()

    def _on_changed(self):
        tol = self._spn_tol.value()
        k = self._spn_k.value()
        if k > 0:
            sigma = tol / k
            self._lbl_sigma.setText(f"{sigma:.6g}")
        else:
            self._lbl_sigma.setText("\u2014")
        self.value_changed.emit()

    # -- public --
    def get_tolerance(self) -> float:
        return self._spn_tol.value()

    def set_tolerance(self, val: float):
        self._spn_tol.setValue(val)

    def get_k(self) -> float:
        return self._spn_k.value()

    def set_k(self, val: float):
        self._spn_k.setValue(val)

    def get_back_calculated_sigma(self) -> float:
        k = self._spn_k.value()
        return self._spn_tol.value() / k if k > 0 else 0.0


class _RSSPanel(QWidget):
    """Panel shown when Input Type is 'RSS of Sub-Components'."""

    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # RSS value
        self._spn_rss = QDoubleSpinBox()
        self._spn_rss.setRange(0.0, 1e9)
        self._spn_rss.setDecimals(6)
        self._spn_rss.setSingleStep(0.01)
        self._spn_rss.setToolTip(
            "Enter the pre-computed RSS (root-sum-square) of sub-component\n"
            "uncertainties. The sigma basis will be applied to this value."
        )
        layout.addRow("RSS value:", self._spn_rss)

        # Sigma basis
        self._cmb_basis = QComboBox()
        self._cmb_basis.addItems(SIGMA_BASIS_OPTIONS)
        self._cmb_basis.setToolTip("Sigma basis of the RSS value.")
        layout.addRow("Sigma Basis:", self._cmb_basis)

        # Number of sub-components
        self._spn_n_comp = QSpinBox()
        self._spn_n_comp.setRange(1, 1000)
        self._spn_n_comp.setValue(1)
        self._spn_n_comp.setToolTip(
            "Number of sub-components combined in the RSS.\n"
            "Used for documentation and optional DOF estimation."
        )
        layout.addRow("Number of sub-components:", self._spn_n_comp)

        # Effective DOF
        self._spn_dof = QDoubleSpinBox()
        self._spn_dof.setRange(0.0, 1e9)
        self._spn_dof.setDecimals(2)
        self._spn_dof.setSpecialValueText("\u221e (not specified)")
        self._spn_dof.setValue(0.0)
        self._spn_dof.setToolTip(
            "Effective degrees of freedom for the RSS combination,\n"
            "if known (e.g., from a Welch-Satterthwaite computation).\n"
            "Leave at 0 / \u221e if unknown. [GUM Annex G]"
        )
        layout.addRow("Effective DOF (optional):", self._spn_dof)

        # Converted 1-sigma
        self._lbl_converted = QLabel("\u2014")
        self._lbl_converted.setStyleSheet(
            f"font-weight: bold; color: {DARK_COLORS['accent']};"
        )
        layout.addRow("Converted 1\u03c3:", self._lbl_converted)

        # Connections
        self._spn_rss.valueChanged.connect(self._on_changed)
        self._cmb_basis.currentIndexChanged.connect(self._on_changed)
        self._spn_n_comp.valueChanged.connect(lambda: self.value_changed.emit())
        self._spn_dof.valueChanged.connect(lambda: self.value_changed.emit())

    def _on_changed(self):
        raw = self._spn_rss.value()
        basis = self._cmb_basis.currentText()
        converted = sigma_from_basis(raw, basis)
        self._lbl_converted.setText(f"{converted:.6g}")
        self.value_changed.emit()

    # -- public --
    def get_rss(self) -> float:
        return self._spn_rss.value()

    def set_rss(self, val: float):
        self._spn_rss.setValue(val)

    def get_basis(self) -> str:
        return self._cmb_basis.currentText()

    def set_basis(self, val: str):
        idx = self._cmb_basis.findText(val)
        if idx >= 0:
            self._cmb_basis.setCurrentIndex(idx)

    def get_n_components(self) -> int:
        return self._spn_n_comp.value()

    def set_n_components(self, val: int):
        self._spn_n_comp.setValue(val)

    def get_effective_dof(self) -> float:
        val = self._spn_dof.value()
        return float('inf') if val == 0.0 else val

    def set_effective_dof(self, val: float):
        self._spn_dof.setValue(0.0 if np.isinf(val) else val)


class _CFDSensitivityPanel(QWidget):
    """Panel shown when Input Type is 'CFD Sensitivity Run'."""

    value_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._deltas: List[float] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Paste button
        btn_row = QHBoxLayout()
        self._btn_paste = QPushButton("Paste Deltas")
        self._btn_paste.setToolTip(
            "Paste delta values (sensitivity differences) from clipboard.\n"
            "One value per line or comma-separated."
        )
        btn_row.addWidget(self._btn_paste)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Delta table
        self._table = QTableWidget()
        self._table.setColumnCount(1)
        self._table.setHorizontalHeaderLabels(["Delta"])
        style_table(self._table, stretch_col=0)
        self._table.setMinimumHeight(100)
        self._table.setMaximumHeight(180)
        self._table.setAlternatingRowColors(True)
        layout.addWidget(self._table)

        # Stats
        form = QFormLayout()
        form.setSpacing(3)
        self._lbl_sigma = QLabel("\u2014")
        self._lbl_sigma.setStyleSheet(
            f"font-weight: bold; color: {DARK_COLORS['accent']};"
        )
        self._lbl_n_locs = QLabel("\u2014")
        form.addRow("\u03c3 from variation:", self._lbl_sigma)
        form.addRow("Number of sensor locations:", self._lbl_n_locs)
        layout.addLayout(form)

        # Connections
        self._btn_paste.clicked.connect(self._paste_deltas)

    def _paste_deltas(self):
        clipboard = QApplication.clipboard()
        text = clipboard.text()
        if not text or not text.strip():
            QMessageBox.information(self, "Clipboard Empty",
                                   "No data found on clipboard.")
            return
        values = []
        for token in text.replace(',', '\n').replace('\t', '\n').split('\n'):
            token = token.strip()
            if not token:
                continue
            try:
                values.append(float(token))
            except ValueError:
                pass
        if not values:
            QMessageBox.warning(self, "Parse Error",
                                "No numeric delta values found in clipboard.")
            return
        self._deltas = values
        self._populate_table()
        self._update_stats()
        audit_log.log_data_load("clipboard",
                                f"CFD sensitivity deltas: n={len(values)}")
        self.value_changed.emit()

    def _populate_table(self):
        self._table.setRowCount(len(self._deltas))
        for i, v in enumerate(self._deltas):
            item = QTableWidgetItem(f"{v:.6g}")
            item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self._table.setItem(i, 0, item)

    def _update_stats(self):
        if len(self._deltas) > 1:
            arr = np.array(self._deltas, dtype=float)
            self._lbl_sigma.setText(f"{np.std(arr, ddof=1):.6g}")
            self._lbl_n_locs.setText(str(len(self._deltas)))
        elif len(self._deltas) == 1:
            self._lbl_sigma.setText(f"{abs(self._deltas[0]):.6g}")
            self._lbl_n_locs.setText("1")
        else:
            self._lbl_sigma.setText("\u2014")
            self._lbl_n_locs.setText("\u2014")

    # -- public --
    def get_deltas(self) -> List[float]:
        return list(self._deltas)

    def set_deltas(self, values: List[float]):
        self._deltas = list(values)
        self._populate_table()
        self._update_stats()


# ---- Main Uncertainty Sources Tab ----

class UncertaintySourcesTab(QWidget):
    """
    Tab 2: Uncertainty Sources Tab.

    Core input tab where users define each uncertainty source for the V&V 20
    uncertainty budget. Supports multiple input types (tabular data, sigma
    value, tolerance, RSS, CFD sensitivity) and provides per-source guidance,
    warnings, and distribution fitting.

    References:
        - ASME V&V 20-2009 Section 3 (uncertainty classification)
        - JCGM 100:2008 (GUM) Section 4 (evaluation of uncertainty)
        - ASME PTC 19.1-2018 Section 4 (Type B evaluation)
        - JCGM 101:2008 (GUM Supplement 1) (Monte Carlo)
    """

    sources_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._sources: List[UncertaintySource] = []
        self._current_index: int = -1
        self._updating_ui = False  # guard for programmatic updates
        self._setup_ui()
        self._connect_signals()

    # =================================================================
    # UI CONSTRUCTION
    # =================================================================
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(5)

        # ---- LEFT PANEL: Source List ----
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)

        list_label = QLabel("Uncertainty Sources")
        list_label.setStyleSheet(
            f"font-weight: bold; font-size: 14px; color: {DARK_COLORS['accent']};"
        )
        left_layout.addWidget(list_label)

        # Source table
        self._source_table = QTableWidget()
        self._source_table.setColumnCount(8)
        self._source_table.setHorizontalHeaderLabels(
            ["Name", "Category", "\u03c3 (1\u03c3)", "Unit", "DOF",
             "% Contrib.", "Class", "Reducibility"]
        )
        self._source_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._source_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._source_table.setAlternatingRowColors(True)
        style_table(self._source_table,
                    column_widths={0: 150, 1: 120, 2: 75, 3: 55, 4: 55,
                                   5: 80, 6: 80, 7: 90},
                    stretch_col=0)
        self._source_table.setMinimumWidth(420)
        left_layout.addWidget(self._source_table)

        # Buttons row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)
        self._btn_add = QPushButton("Add Source")
        self._btn_add.setToolTip("Add a new uncertainty source to the budget.")
        self._btn_remove = QPushButton("Remove Source")
        self._btn_remove.setToolTip("Remove the selected uncertainty source.")
        self._btn_dup = QPushButton("Duplicate Source")
        self._btn_dup.setToolTip("Create a copy of the selected source.")
        btn_layout.addWidget(self._btn_add)
        btn_layout.addWidget(self._btn_remove)
        btn_layout.addWidget(self._btn_dup)
        left_layout.addLayout(btn_layout)

        move_layout = QHBoxLayout()
        move_layout.setSpacing(4)
        self._btn_up = QPushButton("Move Up")
        self._btn_up.setToolTip("Move the selected source up in the list.")
        self._btn_down = QPushButton("Move Down")
        self._btn_down.setToolTip("Move the selected source down in the list.")
        move_layout.addWidget(self._btn_up)
        move_layout.addWidget(self._btn_down)
        move_layout.addStretch()
        left_layout.addLayout(move_layout)

        splitter.addWidget(left_widget)

        # ---- RIGHT PANEL: Detail Editor (scrollable) ----
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setMinimumWidth(440)
        right_widget = QWidget()
        self._right_layout = QVBoxLayout(right_widget)
        self._right_layout.setContentsMargins(8, 8, 8, 8)
        self._right_layout.setSpacing(8)

        self._build_common_fields()
        self._build_input_type_panels()
        self._build_source_warnings()

        self._right_layout.addStretch()
        right_scroll.setWidget(right_widget)
        splitter.addWidget(right_scroll)

        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([550, 650])
        main_layout.addWidget(splitter)

    # -- Common Fields --
    def _build_common_fields(self):
        grp = QGroupBox("Source Properties")
        form = QFormLayout(grp)
        form.setSpacing(6)

        # Name
        self._edt_name = QLineEdit()
        self._edt_name.setPlaceholderText("e.g., Grid Convergence Index")
        self._edt_name.setToolTip("Descriptive name for this uncertainty source.")
        form.addRow("Name:", self._edt_name)

        # Category
        self._cmb_category = QComboBox()
        self._cmb_category.addItems(UNCERTAINTY_CATEGORIES)
        self._cmb_category.setToolTip(
            "V&V 20 uncertainty category:\n"
            "  \u2022 Numerical (u_num): grid, iteration, discretization\n"
            "  \u2022 Input/BC (u_input): boundary conditions, material properties\n"
            "  \u2022 Experimental (u_D): instrumentation, data acquisition\n"
            "[ASME V&V 20 \u00a73]"
        )
        form.addRow("Category:", self._cmb_category)

        # Input Type
        self._cmb_input_type = QComboBox()
        self._cmb_input_type.addItems(INPUT_TYPE_OPTIONS)
        self._cmb_input_type.setToolTip(
            "Select how you will provide the uncertainty data:\n"
            "  \u2022 Tabular Data: raw measurements, auto-compute \u03c3\n"
            "  \u2022 Sigma Value Only: enter \u03c3 directly\n"
            "  \u2022 Tolerance/Expanded: enter expanded U, back-compute \u03c3\n"
            "  \u2022 RSS of Sub-Components: pre-combined RSS\n"
            "  \u2022 CFD Sensitivity Run: delta values from sensitivity study"
        )
        form.addRow("Input Type:", self._cmb_input_type)

        # Distribution
        self._cmb_distribution = QComboBox()
        self._cmb_distribution.addItems(DISTRIBUTION_NAMES)
        self._cmb_distribution.setToolTip(
            "Assumed probability distribution for this source.\n"
            "Use 'Auto-Fit Distribution' in the Tabular Data panel\n"
            "for data-driven selection. [GUM \u00a74.3]"
        )
        form.addRow("Distribution:", self._cmb_distribution)

        # Unit
        self._cmb_unit = QComboBox()
        self._cmb_unit.addItems(UNIT_OPTIONS)
        self._cmb_unit.setCurrentText(DEFAULT_UNIT)
        self._cmb_unit.setToolTip("Engineering unit for this uncertainty source.")
        form.addRow("Unit:", self._cmb_unit)

        # Correlation Group
        self._edt_corr_group = QLineEdit()
        self._edt_corr_group.setPlaceholderText("Leave blank = independent")
        self._edt_corr_group.setToolTip(
            "Correlation group label. Sources sharing the same non-empty\n"
            "group label AND category are treated as correlated in both RSS\n"
            "and Monte Carlo computations. Cross-category correlation terms\n"
            "are intentionally not applied. Leave blank for independent sources.\n"
            "[ASME V&V 20-2009 §4.3.3]"
        )
        form.addRow("Corr. Group:", self._edt_corr_group)

        # Correlation Coefficient
        self._spn_corr_coeff = QDoubleSpinBox()
        self._spn_corr_coeff.setRange(-1.0, 1.0)
        self._spn_corr_coeff.setSingleStep(0.05)
        self._spn_corr_coeff.setDecimals(2)
        self._spn_corr_coeff.setValue(0.0)
        self._spn_corr_coeff.setToolTip(
            "Pairwise correlation coefficient ρ with the group reference source.\n"
            "ρ = 1.0 = fully correlated, ρ = 0.0 = independent (uncorrelated),\n"
            "ρ = −1.0 = fully anti-correlated.\n"
            "The reference source (first alphabetically in each group) always\n"
            "has ρ = 1.0 enforced automatically. Other sources specify their\n"
            "correlation with the reference. Pairwise: ρ(a,b) = ρ_a × ρ_b.\n"
            "Range: −1.0 to +1.0. [ASME V&V 20-2009 §4.3.3]"
        )
        form.addRow("Corr. ρ:", self._spn_corr_coeff)

        # Reference source hint (shown when this source is the group reference)
        self._lbl_corr_ref_hint = QLabel("")
        self._lbl_corr_ref_hint.setStyleSheet(
            f"color: {DARK_COLORS['yellow']}; font-size: 11px; "
            f"padding: 2px 0;"
        )
        self._lbl_corr_ref_hint.setWordWrap(True)
        self._lbl_corr_ref_hint.hide()
        form.addRow("", self._lbl_corr_ref_hint)

        # Notes
        self._edt_notes = QLineEdit()
        self._edt_notes.setPlaceholderText("Optional notes or reference...")
        self._edt_notes.setToolTip(
            "Free-text notes, source references, or calibration cert ID."
        )
        form.addRow("Notes:", self._edt_notes)

        # Enabled
        self._chk_enabled = QCheckBox("Enabled")
        self._chk_enabled.setChecked(True)
        self._chk_enabled.setToolTip(
            "Uncheck to exclude this source from the uncertainty budget\n"
            "without deleting it. Useful for sensitivity studies."
        )
        form.addRow(self._chk_enabled)

        self._right_layout.addWidget(grp)

        # --- Classification group (epistemic/aleatoric, Section 6.2) ---
        class_grp = QGroupBox("Uncertainty Classification")
        class_form = QFormLayout(class_grp)
        class_form.setSpacing(6)

        self._cmb_uclass = QComboBox()
        self._cmb_uclass.addItems(["aleatoric", "epistemic", "mixed"])
        self._cmb_uclass.setToolTip(
            "Uncertainty nature:\n"
            "  • aleatoric — inherent randomness (irreducible)\n"
            "  • epistemic — lack of knowledge (reducible)\n"
            "  • mixed — both components present\n\n"
            "Practical rule:\n"
            "  • Sensor-based BCs (e.g., inlet mass flow) are often MIXED\n"
            "    unless random and bias parts are separated.\n"
            "  • Grid/discretization uncertainty is usually epistemic or mixed.\n"
            "    If over-refined/asymptotic, reducibility may be low."
        )
        class_form.addRow("Class:", self._cmb_uclass)

        self._cmb_representation = QComboBox()
        self._cmb_representation.addItems([
            "distribution", "interval", "scenario_set", "model_ensemble"
        ])
        self._cmb_representation.setToolTip(
            "How this uncertainty is represented:\n"
            "  • distribution — probabilistic (PDF/CDF)\n"
            "  • interval — bounded range [low, high]\n"
            "  • scenario_set — discrete scenarios\n"
            "  • model_ensemble — spread of model runs"
        )
        class_form.addRow("Representation:", self._cmb_representation)

        self._cmb_basis_type = QComboBox()
        self._cmb_basis_type.addItems([
            "measured", "spec_limit", "expert_judgment",
            "standard_reference", "assumed"
        ])
        self._cmb_basis_type.setToolTip(
            "Evidential basis for this uncertainty value.\n"
            "Type A/Type B describe HOW value was obtained, not whether\n"
            "it is aleatoric or epistemic."
        )
        class_form.addRow("Basis:", self._cmb_basis_type)

        self._cmb_reducibility = QComboBox()
        self._cmb_reducibility.addItems(["low", "medium", "high"])
        self._cmb_reducibility.setToolTip(
            "Potential for reducing this uncertainty:\n"
            "  • low — irreducible or near-irreducible\n"
            "  • medium — reducible with moderate effort\n"
            "  • high — readily reducible with additional data/testing\n\n"
            "If class='epistemic' but reducibility='low', add evidence note\n"
            "explaining why it is currently not reducible in your program."
        )
        class_form.addRow("Reducibility:", self._cmb_reducibility)

        self._edt_evidence = QLineEdit()
        self._edt_evidence.setPlaceholderText("Traceability reference...")
        self._edt_evidence.setToolTip(
            "Concise traceability note (e.g., calibration cert #, "
            "test report ID, standard section)."
        )
        class_form.addRow("Evidence:", self._edt_evidence)

        self._right_layout.addWidget(class_grp)

    # -- Input-Type-Specific Panels --
    def _build_input_type_panels(self):
        grp = QGroupBox("Input Data")
        layout = QVBoxLayout(grp)
        layout.setContentsMargins(8, 12, 8, 8)

        self._stack = QStackedWidget()

        # Index 0: Tabular Data
        self._panel_tabular = _TabularDataPanel()
        self._stack.addWidget(self._panel_tabular)

        # Index 1: Sigma Value Only
        self._panel_sigma = _SigmaValuePanel()
        self._stack.addWidget(self._panel_sigma)

        # Index 2: Tolerance / Expanded Value
        self._panel_tolerance = _TolerancePanel()
        self._stack.addWidget(self._panel_tolerance)

        # Index 3: RSS of Sub-Components
        self._panel_rss = _RSSPanel()
        self._stack.addWidget(self._panel_rss)

        # Index 4: CFD Sensitivity Run
        self._panel_cfd = _CFDSensitivityPanel()
        self._stack.addWidget(self._panel_cfd)

        layout.addWidget(self._stack)
        self._right_layout.addWidget(grp)

        # Map input type text to stack index
        self._input_type_index = {
            "Tabular Data": 0,
            "Sigma Value Only": 1,
            "Tolerance/Expanded Value": 2,
            "RSS of Sub-Components": 3,
            "CFD Sensitivity Run": 4,
        }

    # -- Warnings --
    def _build_source_warnings(self):
        self._warn_general = GuidancePanel("Source Warnings")
        self._warn_general.setVisible(False)
        self._right_layout.addWidget(self._warn_general)

    # =================================================================
    # SIGNAL CONNECTIONS
    # =================================================================
    def _connect_signals(self):
        # Source table selection
        self._source_table.currentCellChanged.connect(self._on_source_selected)
        self._source_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self._source_table.customContextMenuRequested.connect(
            self._on_source_table_context_menu
        )

        # Buttons
        self._btn_add.clicked.connect(self._add_source)
        self._btn_remove.clicked.connect(self._remove_source)
        self._btn_dup.clicked.connect(self._duplicate_source)
        self._btn_up.clicked.connect(self._move_up)
        self._btn_down.clicked.connect(self._move_down)

        # Common field changes
        self._edt_name.textChanged.connect(self._on_field_changed)
        self._cmb_category.currentIndexChanged.connect(self._on_field_changed)
        self._cmb_input_type.currentIndexChanged.connect(self._on_input_type_changed)
        self._cmb_distribution.currentIndexChanged.connect(self._on_field_changed)
        self._cmb_unit.currentIndexChanged.connect(self._on_field_changed)
        self._edt_notes.textChanged.connect(self._on_field_changed)
        self._chk_enabled.toggled.connect(self._on_field_changed)

        # Input-type panel changes
        self._panel_tabular.value_changed.connect(self._on_panel_changed)
        self._panel_sigma.value_changed.connect(self._on_panel_changed)
        self._panel_tolerance.value_changed.connect(self._on_panel_changed)
        self._panel_rss.value_changed.connect(self._on_panel_changed)
        self._panel_cfd.value_changed.connect(self._on_panel_changed)

    # =================================================================
    # SOURCE LIST OPERATIONS
    # =================================================================
    def _add_source(self):
        src = UncertaintySource(name=f"Source {len(self._sources) + 1}")
        self._sources.append(src)
        self._refresh_table()
        self._source_table.selectRow(len(self._sources) - 1)
        audit_log.log("ADD_SOURCE", f"Added source: {src.name}")
        self.sources_changed.emit()

    def _remove_source(self):
        idx = self._current_index
        if idx < 0 or idx >= len(self._sources):
            return
        name = self._sources[idx].name
        reply = QMessageBox.question(
            self, "Remove Source",
            f"Remove uncertainty source \"{name}\"?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self._sources.pop(idx)
        self._current_index = -1
        self._refresh_table()
        if self._sources:
            new_idx = min(idx, len(self._sources) - 1)
            self._source_table.selectRow(new_idx)
        audit_log.log("REMOVE_SOURCE", f"Removed source: {name}")
        self.sources_changed.emit()

    def _duplicate_source(self):
        idx = self._current_index
        if idx < 0 or idx >= len(self._sources):
            return
        dup = copy.deepcopy(self._sources[idx])
        dup.name = f"{dup.name} (copy)"
        self._sources.insert(idx + 1, dup)
        self._refresh_table()
        self._source_table.selectRow(idx + 1)
        audit_log.log("DUPLICATE_SOURCE",
                      f"Duplicated source: {self._sources[idx].name}")
        self.sources_changed.emit()

    def _move_up(self):
        idx = self._current_index
        if idx <= 0 or idx >= len(self._sources):
            return
        self._sources[idx], self._sources[idx - 1] = (
            self._sources[idx - 1], self._sources[idx]
        )
        self._refresh_table()
        self._source_table.selectRow(idx - 1)
        self.sources_changed.emit()

    def _move_down(self):
        idx = self._current_index
        if idx < 0 or idx >= len(self._sources) - 1:
            return
        self._sources[idx], self._sources[idx + 1] = (
            self._sources[idx + 1], self._sources[idx]
        )
        self._refresh_table()
        self._source_table.selectRow(idx + 1)
        self.sources_changed.emit()

    # =================================================================
    # CONTEXT MENU
    # =================================================================
    def _on_source_table_context_menu(self, pos):
        """Right-click context menu for sort/filter by class or reducibility."""
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)

        # Sort actions
        sort_menu = menu.addMenu("Sort by")
        sort_menu.addAction("Name (A\u2192Z)").triggered.connect(
            lambda: self._sort_sources_by('name'))
        sort_menu.addAction("\u03c3 (descending)").triggered.connect(
            lambda: self._sort_sources_by('sigma'))
        sort_menu.addAction("Class").triggered.connect(
            lambda: self._sort_sources_by('class'))
        sort_menu.addAction("Reducibility").triggered.connect(
            lambda: self._sort_sources_by('reducibility'))

        # Filter (highlight) actions
        filter_menu = menu.addMenu("Highlight class")
        filter_menu.addAction("All (clear highlight)").triggered.connect(
            lambda: self._highlight_by_class(None))
        filter_menu.addAction("Aleatoric").triggered.connect(
            lambda: self._highlight_by_class('aleatoric'))
        filter_menu.addAction("Epistemic").triggered.connect(
            lambda: self._highlight_by_class('epistemic'))
        filter_menu.addAction("Mixed").triggered.connect(
            lambda: self._highlight_by_class('mixed'))

        menu.exec(self._source_table.viewport().mapToGlobal(pos))

    def _sort_sources_by(self, key: str):
        """Sort the source list by the given key and refresh."""
        if key == 'name':
            self._sources.sort(key=lambda s: s.name.lower())
        elif key == 'sigma':
            self._sources.sort(key=lambda s: s.get_standard_uncertainty(),
                               reverse=True)
        elif key == 'class':
            order = {'aleatoric': 0, 'mixed': 1, 'epistemic': 2}
            self._sources.sort(
                key=lambda s: order.get(
                    getattr(s, 'uncertainty_class', 'aleatoric'), 3
                )
            )
        elif key == 'reducibility':
            order = {'low': 0, 'medium': 1, 'high': 2}
            self._sources.sort(
                key=lambda s: order.get(
                    getattr(s, 'reducibility', 'low'), 3
                )
            )
        self._refresh_table()
        self.sources_changed.emit()

    def _highlight_by_class(self, cls: Optional[str]):
        """Highlight rows matching the given class; None clears all highlights."""
        for i in range(self._source_table.rowCount()):
            if i >= len(self._sources):
                break
            src = self._sources[i]
            match = (cls is None or
                     getattr(src, 'uncertainty_class', 'aleatoric') == cls)
            for col in range(self._source_table.columnCount()):
                item = self._source_table.item(i, col)
                if item is None:
                    continue
                if match or cls is None:
                    # Reset to default
                    if src.enabled:
                        item.setForeground(QColor(DARK_COLORS['fg']))
                    else:
                        item.setForeground(QColor(DARK_COLORS['fg_dim']))
                else:
                    # Dim non-matching rows
                    item.setForeground(QColor(DARK_COLORS['fg_dim']))

    # =================================================================
    # TABLE REFRESH
    # =================================================================
    def _refresh_table(self):
        """Rebuild the source list table from self._sources."""
        self._source_table.setRowCount(len(self._sources))
        for i, src in enumerate(self._sources):
            sigma = src.get_standard_uncertainty()
            dof = src.get_dof()
            dof_str = "\u221e" if np.isinf(dof) else f"{dof:.1f}"

            u_class = getattr(src, 'uncertainty_class', 'aleatoric')
            reducibility = getattr(src, 'reducibility', 'low')
            items = [
                src.name,
                src.category.split(" (")[0],  # short form
                f"{sigma:.4g}",
                src.unit,
                dof_str,
                "\u2014",  # placeholder for % contribution
                u_class.capitalize() if u_class else "",
                reducibility.capitalize() if reducibility else "",
            ]
            for col, text in enumerate(items):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if col in (2, 4, 5):
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if not src.enabled:
                    item.setForeground(QColor(DARK_COLORS['fg_dim']))
                # Highlight unit column if mismatched with first enabled source
                if col == 3 and src.enabled:
                    ref_unit = next(
                        (s.unit for s in self._sources if s.enabled), None
                    )
                    if ref_unit and src.unit != ref_unit:
                        item.setBackground(QColor(DARK_COLORS['yellow']))
                        item.setForeground(QColor("#1e1e2e"))
                        item.setToolTip(
                            f"Unit mismatch: this source uses '{src.unit}' "
                            f"but other sources use '{ref_unit}'."
                        )
                self._source_table.setItem(i, col, item)

    # =================================================================
    # SELECTION / EDITOR SYNC
    # =================================================================
    def _on_source_selected(self, row, col, prev_row, prev_col):
        if row == self._current_index:
            return
        self._current_index = row
        self._load_source_to_editor(row)

    def _load_source_to_editor(self, idx):
        """Populate the right-panel editor from the source at *idx*."""
        if idx < 0 or idx >= len(self._sources):
            return
        self._updating_ui = True
        try:
            src = self._sources[idx]

            # Common fields
            self._edt_name.setText(src.name)
            self._cmb_category.setCurrentText(src.category)
            self._cmb_input_type.setCurrentText(src.input_type)
            self._cmb_distribution.setCurrentText(src.distribution)
            self._cmb_unit.setCurrentText(src.unit)
            self._edt_corr_group.setText(src.correlation_group)
            self._spn_corr_coeff.setValue(src.correlation_coefficient)

            # Update reference-source hint
            grp = src.correlation_group.strip()
            if grp:
                # Find alphabetically-first enabled source in the same group
                group_names = sorted(
                    (s.name for s in self._sources
                     if s.enabled and s.correlation_group.strip() == grp),
                    key=str.lower,
                )
                if group_names and src.name == group_names[0]:
                    self._lbl_corr_ref_hint.setText(
                        "\u26a0 This source is the group reference "
                        "(1st alphabetically). Its \u03c1 will be "
                        "set to 1.0 during computation."
                    )
                    self._lbl_corr_ref_hint.show()
                else:
                    self._lbl_corr_ref_hint.hide()
            else:
                self._lbl_corr_ref_hint.hide()

            self._edt_notes.setText(src.notes)
            self._chk_enabled.setChecked(src.enabled)

            # Classification fields
            self._cmb_uclass.setCurrentText(src.uncertainty_class)
            self._cmb_representation.setCurrentText(src.representation)
            self._cmb_basis_type.setCurrentText(src.basis_type)
            self._cmb_reducibility.setCurrentText(src.reducibility)
            self._edt_evidence.setText(src.evidence_note)

            # Switch stacked widget
            stack_idx = self._input_type_index.get(src.input_type, 1)
            self._stack.setCurrentIndex(stack_idx)

            # Populate the specific panel
            if src.input_type == "Tabular Data":
                self._panel_tabular.set_data(src.tabular_data)
                self._panel_tabular.set_centered(src.is_centered_on_zero)
            elif src.input_type == "Sigma Value Only":
                self._panel_sigma.set_sigma(src.raw_sigma_value)
                self._panel_sigma.set_basis(src.sigma_basis)
                self._panel_sigma.set_sample_size(src.sample_size)
                self._panel_sigma.set_supplier(src.is_supplier)
                self._panel_sigma.set_asymmetric(src.asymmetric)
                self._panel_sigma.set_sigma_upper(src.sigma_upper)
                self._panel_sigma.set_sigma_lower(src.sigma_lower)
                self._panel_sigma.set_one_sided(src.one_sided)
                self._panel_sigma.set_one_sided_direction(src.one_sided_direction)
            elif src.input_type == "Tolerance/Expanded Value":
                self._panel_tolerance.set_tolerance(src.tolerance_value)
                self._panel_tolerance.set_k(src.tolerance_k)
            elif src.input_type == "RSS of Sub-Components":
                self._panel_rss.set_rss(src.rss_value)
                self._panel_rss.set_basis(src.sigma_basis)
                self._panel_rss.set_n_components(src.rss_n_components)
                self._panel_rss.set_effective_dof(src.dof)
            elif src.input_type == "CFD Sensitivity Run":
                self._panel_cfd.set_deltas(src.sensitivity_deltas)

            # Update warnings
            self._update_source_warnings(src)
        finally:
            self._updating_ui = False

    def _save_editor_to_source(self):
        """Write all editor fields back to the current UncertaintySource."""
        idx = self._current_index
        if idx < 0 or idx >= len(self._sources):
            return
        src = self._sources[idx]

        # Common fields
        src.name = self._edt_name.text()
        src.category = self._cmb_category.currentText()
        src.input_type = self._cmb_input_type.currentText()
        src.distribution = self._cmb_distribution.currentText()
        src.unit = self._cmb_unit.currentText()
        src.correlation_group = self._edt_corr_group.text().strip()
        src.correlation_coefficient = self._spn_corr_coeff.value()
        src.notes = self._edt_notes.text()
        src.enabled = self._chk_enabled.isChecked()

        # Classification fields
        src.uncertainty_class = self._cmb_uclass.currentText()
        src.representation = self._cmb_representation.currentText()
        src.basis_type = self._cmb_basis_type.currentText()
        src.reducibility = self._cmb_reducibility.currentText()
        src.evidence_note = self._edt_evidence.text()

        # Input-type-specific
        if src.input_type == "Tabular Data":
            src.tabular_data = self._panel_tabular.get_data()
            src.is_centered_on_zero = self._panel_tabular.is_centered()
            if src.tabular_data and len(src.tabular_data) > 1:
                arr = np.array(src.tabular_data, dtype=float)
                src.sigma_value = float(np.std(arr, ddof=1))
                src.mean_value = float(np.mean(arr))
                src.sample_size = len(src.tabular_data)
                src.dof = float(len(src.tabular_data) - 1)
            else:
                # No data or single point — reset stale fields
                src.sigma_value = 0.0
                src.mean_value = 0.0
                src.sample_size = 0
                src.dof = float('inf')
                src.is_supplier = False
            # Check if user accepted recommended distribution
            rec = self._panel_tabular.get_recommended_distribution()
            if rec and src.distribution != rec:
                # User overrode the recommendation -- log it
                audit_log.log_override(
                    f"Distribution for '{src.name}'",
                    rec, src.distribution,
                    "User selected a different distribution than auto-fit."
                )

        elif src.input_type == "Sigma Value Only":
            src.raw_sigma_value = self._panel_sigma.get_sigma()
            src.sigma_basis = self._panel_sigma.get_basis()
            src.sigma_value = self._panel_sigma.get_converted_sigma()
            src.sample_size = self._panel_sigma.get_sample_size()
            src.is_supplier = self._panel_sigma.is_supplier()
            # Asymmetric fields
            src.asymmetric = self._panel_sigma.is_asymmetric()
            src.sigma_upper = self._panel_sigma.get_sigma_upper()
            src.sigma_lower = self._panel_sigma.get_sigma_lower()
            src.one_sided = self._panel_sigma.is_one_sided()
            src.one_sided_direction = self._panel_sigma.get_one_sided_direction()
            src.mirror_assumed = True  # always true in current UI
            if src.asymmetric and src.one_sided:
                src.evidence_note = (
                    src.evidence_note or
                    f"One-sided ({src.one_sided_direction}); "
                    f"mirror assumption applied."
                )
            if src.is_supplier:
                src.dof = float('inf')
            elif src.sample_size > 1:
                src.dof = float(src.sample_size - 1)
            else:
                src.dof = float('inf')  # Type B — infinite DOF

        elif src.input_type == "Tolerance/Expanded Value":
            src.tolerance_value = self._panel_tolerance.get_tolerance()
            src.tolerance_k = self._panel_tolerance.get_k()
            src.sigma_value = self._panel_tolerance.get_back_calculated_sigma()
            src.dof = float('inf')  # Tolerance specs are Type B

        elif src.input_type == "RSS of Sub-Components":
            src.rss_value = self._panel_rss.get_rss()
            src.sigma_basis = self._panel_rss.get_basis()
            src.rss_n_components = self._panel_rss.get_n_components()
            src.sigma_value = sigma_from_basis(
                src.rss_value, src.sigma_basis, src.distribution
            )
            eff_dof = self._panel_rss.get_effective_dof()
            src.dof = eff_dof

        elif src.input_type == "CFD Sensitivity Run":
            src.sensitivity_deltas = self._panel_cfd.get_deltas()
            if len(src.sensitivity_deltas) > 1:
                arr = np.array(src.sensitivity_deltas, dtype=float)
                src.sigma_value = float(np.std(arr, ddof=1))
                src.sample_size = len(src.sensitivity_deltas)
                src.dof = float(len(src.sensitivity_deltas) - 1)
            elif len(src.sensitivity_deltas) == 1:
                src.sigma_value = abs(src.sensitivity_deltas[0])
                src.sample_size = 1
                src.dof = float('inf')  # Single perturbation — Type B
            else:
                # No deltas — reset stale fields
                src.sigma_value = 0.0
                src.sample_size = 0
                src.dof = float('inf')

    # =================================================================
    # FIELD CHANGE HANDLERS
    # =================================================================
    def _on_field_changed(self):
        """Called when any common field changes."""
        if self._updating_ui:
            return
        self._save_editor_to_source()
        self._refresh_table()
        if self._current_index >= 0 and self._current_index < len(self._sources):
            self._update_source_warnings(self._sources[self._current_index])
        self.sources_changed.emit()

    def _on_input_type_changed(self):
        """Called when the Input Type combo changes."""
        if self._updating_ui:
            return
        new_type = self._cmb_input_type.currentText()
        stack_idx = self._input_type_index.get(new_type, 1)
        self._stack.setCurrentIndex(stack_idx)
        self._save_editor_to_source()
        self._refresh_table()
        if self._current_index >= 0 and self._current_index < len(self._sources):
            self._update_source_warnings(self._sources[self._current_index])
        self.sources_changed.emit()

    def _on_panel_changed(self):
        """Called when any input-type-specific panel value changes."""
        if self._updating_ui:
            return
        self._save_editor_to_source()
        self._refresh_table()
        if self._current_index >= 0 and self._current_index < len(self._sources):
            self._update_source_warnings(self._sources[self._current_index])
        self.sources_changed.emit()

    # =================================================================
    # WARNINGS
    # =================================================================
    def _update_source_warnings(self, src: UncertaintySource):
        """Evaluate and display per-source warnings in the detail panel."""
        warnings_list = []
        severity = 'green'

        # Small sample warning for tabular data
        if src.input_type == "Tabular Data" and 0 < len(src.tabular_data) < 10:
            warnings_list.append(
                f"Small sample (n={len(src.tabular_data)}) \u2014 distribution "
                f"shape cannot be reliably determined. [GUM \u00a74.3]"
            )
            severity = 'yellow'

        # Unverified sigma basis
        if src.sigma_basis == "Assumed 1\u03c3 (unverified)":
            warnings_list.append(
                "Sigma basis is unverified. Confirm with supplier "
                "documentation. [PTC 19.1 \u00a74]"
            )
            if severity != 'red':
                severity = 'yellow'

        # Large sigma flag (placeholder — actual combined will be computed
        # in Tab 3; store a flag for later)
        sigma = src.get_standard_uncertainty()
        if sigma == 0.0 and src.enabled:
            warnings_list.append(
                "Standard uncertainty is zero. Verify that data has been "
                "entered correctly."
            )
            if severity != 'red':
                severity = 'yellow'

        # Classification guardrails (prevent common misuse in early-stage teams)
        uclass = (getattr(src, 'uncertainty_class', 'aleatoric') or 'aleatoric').lower()
        reducibility = (getattr(src, 'reducibility', 'low') or 'low').lower()
        basis_type = (getattr(src, 'basis_type', 'measured') or 'measured').lower()
        cat_key = src.get_category_key()
        name_l = (src.name or "").lower()
        evidence_l = (src.evidence_note or "").lower()

        if uclass == "epistemic" and reducibility == "low":
            warnings_list.append(
                "Class/reducibility mismatch: epistemic uncertainty is usually "
                "reducible. If this is a true measurement floor, classify as "
                "'mixed' or 'aleatoric' and document evidence."
            )
            if severity != 'red':
                severity = 'yellow'

        if uclass == "epistemic" and basis_type in (
            "measured", "spec_limit", "standard_reference"
        ):
            warnings_list.append(
                "Measured/spec-based uncertainty often contains both random "
                "(aleatoric) and bias (epistemic) parts. Consider 'mixed' "
                "unless those parts were separated explicitly."
            )
            if severity != 'red':
                severity = 'yellow'

        if cat_key == "u_num" and uclass == "aleatoric":
            warnings_list.append(
                "Numerical uncertainty is usually epistemic or mixed in V&V "
                "practice. Use purely aleatoric classification only when an "
                "irreducible numerical floor has been demonstrated."
            )
            if severity != 'red':
                severity = 'yellow'

        if ("mass flow" in name_l or "massflow" in name_l or "mdot" in name_l) and uclass == "epistemic":
            warnings_list.append(
                "Inlet mass-flow sensor uncertainty is often mixed "
                "(repeatability/noise + calibration bias). Consider class "
                "'mixed' unless you have isolated one component."
            )
            if severity != 'red':
                severity = 'yellow'

        if cat_key == "u_num" and (
            "over-refin" in name_l or "over-refin" in evidence_l
            or "grid-independent" in evidence_l or "asymptotic" in evidence_l
        ) and uclass == "epistemic":
            warnings_list.append(
                "If the mesh study already shows asymptotic/grid-independent "
                "behavior, residual numerical uncertainty may be low-reducibility "
                "mixed/aleatoric-like. Document rationale and consider 'mixed'."
            )
            if severity != 'red':
                severity = 'yellow'

        # Disabled source note
        if not src.enabled:
            warnings_list.append(
                "This source is disabled and will not be included in the "
                "uncertainty budget."
            )

        if warnings_list:
            self._warn_general.set_guidance('\n'.join(warnings_list), severity)
            self._warn_general.setVisible(True)
        else:
            self._warn_general.setVisible(False)

    # =================================================================
    # PUBLIC API
    # =================================================================
    def get_sources(self) -> List[UncertaintySource]:
        """
        Return all enabled uncertainty sources.

        Returns:
            List of UncertaintySource instances where enabled is True.
        """
        return [s for s in self._sources if s.enabled]

    def get_all_sources(self) -> List[UncertaintySource]:
        """
        Return all uncertainty sources including disabled ones.

        Returns:
            List of all UncertaintySource instances.
        """
        return list(self._sources)

    def set_sources(self, sources: List[UncertaintySource]):
        """
        Load a list of sources (e.g., from a saved project file).

        Parameters:
            sources: List of UncertaintySource instances to populate the tab.
        """
        self._sources = [copy.deepcopy(s) for s in sources]
        self._current_index = -1
        self._refresh_table()
        if self._sources:
            self._source_table.selectRow(0)
        audit_log.log("SOURCES_LOADED",
                      f"Loaded {len(self._sources)} uncertainty sources.")
        self.sources_changed.emit()


# =============================================================================
# SECTION 9: TAB 3 — ANALYSIS SETTINGS TAB
# =============================================================================
class AnalysisSettingsTab(QWidget):
    """
    Tab 3: Analysis Settings Tab.

    Configuration interface for coverage probability, confidence level,
    k-factor method, Monte Carlo parameters, and bound type selection.

    All fields emit ``settings_changed`` whenever the user modifies a value
    so that downstream tabs can react immediately.

    References:
        - ASME V&V 20-2009 Section 6 (coverage factor)
        - JCGM 100:2008 (GUM) Annex G (Welch-Satterthwaite)
        - JCGM 101:2008 (Monte Carlo supplement)
        - ASME PTC 19.1-2018 Section 7 (statistical methods)
    """

    settings_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    # -----------------------------------------------------------------
    # UI CONSTRUCTION
    # -----------------------------------------------------------------
    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        self._main_layout = QVBoxLayout(scroll_widget)
        self._main_layout.setSpacing(10)
        self._main_layout.setContentsMargins(8, 8, 8, 8)

        self._build_coverage_confidence_section()
        self._build_k_method_section()
        self._build_mc_settings_section()
        self._build_bound_type_section()
        self._build_comparison_sampling_section()

        self._main_layout.addStretch()
        scroll.setWidget(scroll_widget)
        outer.addWidget(scroll)

    # =================================================================
    # SECTION 1: COVERAGE AND CONFIDENCE
    # =================================================================
    def _build_coverage_confidence_section(self):
        grp = QGroupBox("Coverage and Confidence")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        form = QFormLayout()
        form.setSpacing(6)

        # -- Coverage probability --
        self._cmb_coverage = QComboBox()
        for val in COVERAGE_OPTIONS:
            self._cmb_coverage.addItem(f"{int(val * 100)}%", val)
        self._cmb_coverage.setCurrentIndex(
            COVERAGE_OPTIONS.index(0.95) if 0.95 in COVERAGE_OPTIONS else 0
        )
        self._cmb_coverage.setToolTip(
            "The fraction of the population that the bound is intended to "
            "contain. 95% coverage means the bound captures 95% of "
            "predictions. [V&V 20 / GUM]"
        )
        form.addRow("Coverage probability p:", self._cmb_coverage)

        # -- Confidence level --
        self._cmb_confidence = QComboBox()
        for val in CONFIDENCE_OPTIONS:
            self._cmb_confidence.addItem(f"{int(val * 100)}%", val)
        self._cmb_confidence.setCurrentIndex(
            CONFIDENCE_OPTIONS.index(0.95) if 0.95 in CONFIDENCE_OPTIONS else 0
        )
        self._cmb_confidence.setToolTip(
            "How confident you are that the bound achieves the stated "
            "coverage. Higher confidence requires larger k, especially "
            "for small samples. [GUM Annex G]"
        )
        form.addRow("Confidence level \u03b3:", self._cmb_confidence)

        layout.addLayout(form)

        # -- One-sided vs Two-sided --
        sided_layout = QHBoxLayout()
        sided_layout.setSpacing(12)
        sided_label = QLabel("Bound type:")
        sided_layout.addWidget(sided_label)

        self._bg_sided = QButtonGroup(self)
        self._rb_one_sided = QRadioButton("One-sided")
        self._rb_one_sided.setToolTip(
            "Use when you have a directional concern (e.g., only "
            "underprediction matters). [Recommended for safety-critical "
            "thermal validation]"
        )
        self._rb_two_sided = QRadioButton("Two-sided")
        self._rb_two_sided.setToolTip(
            "Use when both over- and under-prediction are equally "
            "concerning. Coverage split between both tails."
        )
        self._rb_one_sided.setChecked(True)
        self._bg_sided.addButton(self._rb_one_sided, 0)
        self._bg_sided.addButton(self._rb_two_sided, 1)
        sided_layout.addWidget(self._rb_one_sided)
        sided_layout.addWidget(self._rb_two_sided)
        sided_layout.addStretch()
        layout.addLayout(sided_layout)

        # -- Collapsible guidance --
        self._guidance_cc_btn = QPushButton("\u25b6  Which should I pick?")
        self._guidance_cc_btn.setFlat(True)
        self._guidance_cc_btn.setCursor(Qt.PointingHandCursor)
        self._guidance_cc_btn.setStyleSheet(
            f"color: {DARK_COLORS['link']}; text-align: left; "
            f"padding: 2px 0px; font-weight: bold;"
        )
        self._guidance_cc_btn.clicked.connect(self._toggle_cc_guidance)
        layout.addWidget(self._guidance_cc_btn)

        self._guidance_cc_panel = GuidancePanel("Coverage / Confidence Guidance")
        self._guidance_cc_panel.set_guidance(
            "\u2022 Is your concern one-directional (e.g., only underprediction "
            "matters for structural margins)?  \u2192  Use one-sided.\n"
            "\u2022 Are both tails equally important?  \u2192  Use two-sided.\n"
            "\u2022 For aerospace thermal certification, 95/95 one-sided is "
            "the most common requirement. [V&V 20 \u00a76, PTC 19.1 \u00a77]",
            'green'
        )
        self._guidance_cc_panel.setVisible(False)
        layout.addWidget(self._guidance_cc_panel)

        self._main_layout.addWidget(grp)

    def _toggle_cc_guidance(self):
        visible = not self._guidance_cc_panel.isVisible()
        self._guidance_cc_panel.setVisible(visible)
        arrow = "\u25bc" if visible else "\u25b6"
        self._guidance_cc_btn.setText(f"{arrow}  Which should I pick?")

    # =================================================================
    # SECTION 2: COVERAGE FACTOR (k) METHOD
    # =================================================================
    def _build_k_method_section(self):
        grp = QGroupBox("Coverage Factor (k) Method")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        self._bg_k_method = QButtonGroup(self)

        # --- Option 1: V&V 20 Default ---
        self._rb_k_vv20 = QRadioButton("ASME V&V 20 Default:  k = 2")
        self._rb_k_vv20.setToolTip(
            "Uses a fixed k = 2, which corresponds approximately to 95% "
            "coverage for a Gaussian distribution with large degrees of "
            "freedom (\u03bd \u2192 \u221e). Appropriate when \u03bd_eff \u2265 30. [V&V 20 \u00a76]"
        )
        self._rb_k_vv20.setChecked(True)
        self._bg_k_method.addButton(self._rb_k_vv20, 0)
        layout.addWidget(self._rb_k_vv20)

        desc_vv20 = QLabel(
            "  Fixed k = 2.  Valid when effective DOF \u03bd_eff \u2265 30."
        )
        desc_vv20.setStyleSheet(f"color: {DARK_COLORS['fg_dim']}; margin-left: 22px;")
        desc_vv20.setWordWrap(True)
        layout.addWidget(desc_vv20)

        # --- Option 2: GUM Welch-Satterthwaite ---
        self._rb_k_ws = QRadioButton("GUM Welch\u2013Satterthwaite (computed k)")
        self._rb_k_ws.setToolTip(
            "Computes effective degrees of freedom via the "
            "Welch\u2013Satterthwaite equation, then derives k from the "
            "t-distribution. Essential for small samples. [GUM Annex G]"
        )
        self._bg_k_method.addButton(self._rb_k_ws, 1)
        layout.addWidget(self._rb_k_ws)

        ws_detail = QHBoxLayout()
        ws_detail.setContentsMargins(22, 0, 0, 0)
        self._lbl_nu_eff = QLabel("\u03bd_eff = \u2014")
        self._lbl_nu_eff.setStyleSheet(f"color: {DARK_COLORS['fg_dim']};")
        ws_detail.addWidget(self._lbl_nu_eff)
        self._lbl_k_ws = QLabel("k = \u2014")
        self._lbl_k_ws.setStyleSheet(f"color: {DARK_COLORS['fg_dim']};")
        ws_detail.addWidget(self._lbl_k_ws)
        ws_detail.addStretch()
        layout.addLayout(ws_detail)

        # --- Option 3: One-Sided Tolerance Factor ---
        self._rb_k_tol = QRadioButton(
            "One-Sided Tolerance Factor (non-central t)"
        )
        self._rb_k_tol.setToolTip(
            "Uses the non-central t-distribution to determine a one-sided "
            "tolerance factor. This accounts for both the desired coverage "
            "AND the extra uncertainty from having a limited sample size "
            "(the fewer data points you have, the larger k must be to "
            "maintain the same confidence). [ISO 16269-6, ASME PTC 19.1]"
        )
        self._bg_k_method.addButton(self._rb_k_tol, 2)
        layout.addWidget(self._rb_k_tol)

        desc_tol = QLabel(
            "  Derives k from non-central t for exact coverage/confidence."
        )
        desc_tol.setStyleSheet(f"color: {DARK_COLORS['fg_dim']}; margin-left: 22px;")
        desc_tol.setWordWrap(True)
        layout.addWidget(desc_tol)

        # --- Option 4: Manual k ---
        self._rb_k_manual = QRadioButton("Manual k Entry")
        self._rb_k_manual.setToolTip(
            "Specify your own coverage factor. Use only when a specific k "
            "is contractually mandated or agreed with the certifying authority."
        )
        self._bg_k_method.addButton(self._rb_k_manual, 3)
        layout.addWidget(self._rb_k_manual)

        manual_row = QHBoxLayout()
        manual_row.setContentsMargins(22, 0, 0, 0)
        manual_row.setSpacing(6)
        lbl_manual = QLabel("k =")
        lbl_manual.setStyleSheet(f"color: {DARK_COLORS['fg_dim']};")
        manual_row.addWidget(lbl_manual)
        self._spn_manual_k = QDoubleSpinBox()
        self._spn_manual_k.setRange(0.5, 10.0)
        self._spn_manual_k.setSingleStep(0.1)
        self._spn_manual_k.setDecimals(2)
        self._spn_manual_k.setValue(2.0)
        self._spn_manual_k.setToolTip(
            "Enter a custom coverage factor k. Typical range is 1.5 to 3.0."
        )
        self._spn_manual_k.setFixedWidth(90)
        self._spn_manual_k.setEnabled(False)
        manual_row.addWidget(self._spn_manual_k)
        manual_row.addStretch()
        layout.addLayout(manual_row)

        # Enable/disable manual spin box based on selection
        self._bg_k_method.idToggled.connect(self._on_k_method_toggled)

        # -- k-method guidance --
        self._guidance_k = GuidancePanel("k-Factor Guidance")
        self._guidance_k.set_guidance(
            "For certification applications, the coverage/confidence level "
            "should be agreed upon with the certifying authority before "
            "analysis. Using k=2 without verifying effective DOF may be "
            "non-conservative for small samples. [V&V 20 \u00a76]",
            'yellow'
        )
        layout.addWidget(self._guidance_k)

        self._guidance_k_cert_btn = QPushButton(
            "\u25b6  WHY THIS MATTERS FOR CERTIFICATION"
        )
        self._guidance_k_cert_btn.setFlat(True)
        self._guidance_k_cert_btn.setCursor(Qt.PointingHandCursor)
        self._guidance_k_cert_btn.setStyleSheet(
            f"color: {DARK_COLORS['link']}; text-align: left; "
            f"padding: 2px 0px; font-weight: bold;"
        )
        self._guidance_k_cert_btn.clicked.connect(self._toggle_k_cert_guidance)
        layout.addWidget(self._guidance_k_cert_btn)

        self._guidance_k_cert_panel = GuidancePanel("Certification Note")
        self._guidance_k_cert_panel.set_guidance(
            "If \u03bd_eff < 30, the V&V 20 default k=2 may underestimate the "
            "true coverage. The GUM Welch\u2013Satterthwaite or tolerance "
            "factor methods account for this by computing k from the "
            "t-distribution with the actual effective degrees of freedom.",
            'yellow'
        )
        self._guidance_k_cert_panel.setVisible(False)
        layout.addWidget(self._guidance_k_cert_panel)

        self._main_layout.addWidget(grp)

    def _on_k_method_toggled(self, btn_id: int, checked: bool):
        if not checked:
            return
        # Enable manual k spin box only when manual is selected
        self._spn_manual_k.setEnabled(btn_id == 3)
        self._emit_settings_changed()

    def _toggle_k_cert_guidance(self):
        visible = not self._guidance_k_cert_panel.isVisible()
        self._guidance_k_cert_panel.setVisible(visible)
        arrow = "\u25bc" if visible else "\u25b6"
        self._guidance_k_cert_btn.setText(
            f"{arrow}  WHY THIS MATTERS FOR CERTIFICATION"
        )

    # =================================================================
    # SECTION 3: MONTE CARLO SETTINGS
    # =================================================================
    def _build_mc_settings_section(self):
        grp = QGroupBox("Monte Carlo Settings")
        form = QFormLayout(grp)
        form.setSpacing(6)

        # -- Sampling method --
        self._cmb_mc_sampling = QComboBox()
        self._cmb_mc_sampling.addItem("Monte Carlo (Random)",
                                       "Monte Carlo (Random)")
        self._cmb_mc_sampling.addItem("Latin Hypercube (LHS)",
                                       "Latin Hypercube (LHS)")
        self._cmb_mc_sampling.setToolTip(
            "Monte Carlo (Random): standard pseudo-random sampling.\n"
            "Latin Hypercube (LHS): stratified sampling that divides each\n"
            "source's probability space into N equal intervals and draws\n"
            "exactly one sample per interval.  Produces the same result\n"
            "distributions and often stabilizes percentile estimates faster\n"
            "than random MC for the same N.\n"
            "[McKay et al. 1979; JCGM 101:2008 MC framework; ASME V&V 20 §4.4]"
        )
        form.addRow("Sampling method:", self._cmb_mc_sampling)

        # -- Number of trials --
        self._cmb_mc_trials = QComboBox()
        mc_trial_options = [10000, 50000, 100000, 500000, 1000000]
        for n in mc_trial_options:
            self._cmb_mc_trials.addItem(f"{n:,}", n)
        # Default to 100,000
        self._cmb_mc_trials.setCurrentIndex(
            mc_trial_options.index(100000) if 100000 in mc_trial_options else 2
        )
        self._cmb_mc_trials.setToolTip(
            "More trials = more stable results. 100,000 is adequate for "
            "most applications. Increase for tighter convergence of tail "
            "probabilities. [JCGM 101:2008]"
        )
        form.addRow("Number of trials:", self._cmb_mc_trials)

        # -- Random seed --
        self._spn_seed = QSpinBox()
        self._spn_seed.setRange(0, 999999999)
        self._spn_seed.setValue(0)
        self._spn_seed.setSpecialValueText("None (random)")
        self._spn_seed.setToolTip(
            "Set a fixed seed for reproducible results. "
            "0 = no fixed seed (truly random each run)."
        )
        form.addRow("Random seed:", self._spn_seed)

        # -- Bootstrap confidence --
        self._chk_bootstrap = QCheckBox("Enable bootstrap confidence intervals")
        self._chk_bootstrap.setChecked(True)
        self._chk_bootstrap.setToolTip(
            "When enabled, the Monte Carlo engine uses bootstrapping to "
            "estimate confidence intervals on the validation metric. This "
            "provides a non-parametric estimate of sampling uncertainty "
            "on the bound itself. [Efron & Tibshirani, 1993]"
        )
        form.addRow("", self._chk_bootstrap)

        self._main_layout.addWidget(grp)

    # =================================================================
    # SECTION 4: BOUND TYPE SELECTION
    # =================================================================
    def _build_bound_type_section(self):
        grp = QGroupBox("Bound Type Selection")
        layout = QVBoxLayout(grp)
        layout.setSpacing(6)

        self._bg_bound = QButtonGroup(self)

        self._rb_bound_uval = QRadioButton(
            "Known uncertainties only (u_val)"
        )
        self._rb_bound_uval.setToolTip(
            "Constructs the validation bound using only the RSS of known "
            "uncertainty sources (u_num, u_input, u_D). Does not use the "
            "observed comparison-error scatter s_E directly in the bound."
        )
        self._bg_bound.addButton(self._rb_bound_uval, 0)
        layout.addWidget(self._rb_bound_uval)

        desc_uval = QLabel(
            "  Bound = |E\u0305| + k \u00b7 u_val.  Uses only catalogued "
            "uncertainty sources."
        )
        desc_uval.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; margin-left: 22px;"
        )
        desc_uval.setWordWrap(True)
        layout.addWidget(desc_uval)

        self._rb_bound_sE = QRadioButton(
            "Total observed scatter (s_E)"
        )
        self._rb_bound_sE.setToolTip(
            "Constructs the validation bound using the sample standard "
            "deviation of comparison errors (s_E). Captures unmodelled "
            "effects but may overcount if s_E includes known sources."
        )
        self._bg_bound.addButton(self._rb_bound_sE, 1)
        layout.addWidget(self._rb_bound_sE)

        desc_sE = QLabel(
            "  Bound = |E\u0305| + k \u00b7 s_E.  Uses empirical scatter "
            "of comparison errors."
        )
        desc_sE.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; margin-left: 22px;"
        )
        desc_sE.setWordWrap(True)
        layout.addWidget(desc_sE)

        self._rb_bound_both = QRadioButton(
            "Both (for comparison)  [Recommended]"
        )
        self._rb_bound_both.setToolTip(
            "Computes both bounds and presents them side-by-side. This is "
            "recommended so you can compare whether the catalogued "
            "uncertainties fully explain the observed scatter."
        )
        self._rb_bound_both.setChecked(True)
        self._bg_bound.addButton(self._rb_bound_both, 2)
        layout.addWidget(self._rb_bound_both)

        desc_both = QLabel(
            "  Computes both bounds and displays them side-by-side for "
            "comparison.  [Recommended]"
        )
        desc_both.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; margin-left: 22px;"
        )
        desc_both.setWordWrap(True)
        layout.addWidget(desc_both)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Validation metric mode:"))
        self._cmb_validation_mode = QComboBox()
        self._cmb_validation_mode.addItem(
            "Standard scalar (V&V 20)",
            "Standard scalar (V&V 20)",
        )
        self._cmb_validation_mode.addItem(
            "Multivariate supplement (covariance-aware)",
            "Multivariate supplement (covariance-aware)",
        )
        self._cmb_validation_mode.setToolTip(
            "Standard scalar keeps the classic V&V 20 metric.\n"
            "Multivariate supplement additionally reports a covariance-aware\n"
            "global metric for multi-location datasets."
        )
        mode_row.addWidget(self._cmb_validation_mode)
        mode_row.addStretch()
        layout.addLayout(mode_row)

        self._main_layout.addWidget(grp)

    # =================================================================
    # SECTION 5: (Removed — comparison-data sampling is not applicable;
    # V&V 20 MC propagates uncertainty sources only, using E_mean as a
    # constant offset.  The mc_comparison_sampling dataclass field is
    # retained for backward compatibility with saved JSON projects.)
    # =================================================================
    def _build_comparison_sampling_section(self):
        pass  # intentionally empty — section removed

    # -----------------------------------------------------------------
    # SIGNAL CONNECTIONS
    # -----------------------------------------------------------------
    def _connect_signals(self):
        """Wire every user-modifiable widget to emit settings_changed."""
        # Coverage / Confidence
        self._cmb_coverage.currentIndexChanged.connect(
            self._emit_settings_changed
        )
        self._cmb_confidence.currentIndexChanged.connect(
            self._emit_settings_changed
        )
        self._bg_sided.idToggled.connect(
            lambda *_: self._emit_settings_changed()
        )

        # k-method (already connected via _on_k_method_toggled which calls
        # _emit_settings_changed, but connect group signal too for safety)
        self._spn_manual_k.valueChanged.connect(self._emit_settings_changed)

        # Monte Carlo
        self._cmb_mc_sampling.currentIndexChanged.connect(
            self._emit_settings_changed
        )
        self._cmb_mc_trials.currentIndexChanged.connect(
            self._emit_settings_changed
        )
        self._spn_seed.valueChanged.connect(self._emit_settings_changed)
        self._chk_bootstrap.toggled.connect(self._emit_settings_changed)

        # Bound type
        self._bg_bound.idToggled.connect(
            lambda *_: self._emit_settings_changed()
        )
        self._cmb_validation_mode.currentIndexChanged.connect(
            self._emit_settings_changed
        )

    def _emit_settings_changed(self, *_args):
        """Common slot that logs the change and emits the signal."""
        if getattr(self, '_loading', False):
            return  # Suppress during programmatic load
        audit_log.log("SETTINGS_CHANGED", "Analysis settings modified.")
        self.settings_changed.emit()

    # =================================================================
    # PUBLIC API
    # =================================================================
    def get_settings(self) -> AnalysisSettings:
        """
        Read the current widget state into an AnalysisSettings dataclass.

        Returns:
            AnalysisSettings with all current user selections.
        """
        # Coverage / Confidence
        coverage = self._cmb_coverage.currentData()
        confidence = self._cmb_confidence.currentData()
        one_sided = self._rb_one_sided.isChecked()

        # k-method
        k_id = self._bg_k_method.checkedId()
        k_map = {
            0: K_METHOD_VV20,
            1: K_METHOD_WS,
            2: K_METHOD_TOLERANCE,
            3: K_METHOD_MANUAL,
        }
        k_method = k_map.get(k_id, K_METHOD_VV20)
        manual_k = self._spn_manual_k.value()

        # Monte Carlo
        mc_sampling_method = self._cmb_mc_sampling.currentData()
        mc_n_trials = self._cmb_mc_trials.currentData()
        seed_val = self._spn_seed.value()
        mc_seed = seed_val if seed_val != 0 else None
        mc_bootstrap = self._chk_bootstrap.isChecked()

        # Bound type
        bound_id = self._bg_bound.checkedId()
        bound_map = {
            0: "Known uncertainties only (u_val)",
            1: "Total observed scatter (s_E)",
            2: "Both (for comparison)",
        }
        bound_type = bound_map.get(bound_id, "Both (for comparison)")
        validation_mode = self._cmb_validation_mode.currentData()

        return AnalysisSettings(
            coverage=coverage,
            confidence=confidence,
            one_sided=one_sided,
            k_method=k_method,
            manual_k=manual_k,
            mc_n_trials=mc_n_trials,
            mc_seed=mc_seed,
            mc_bootstrap=mc_bootstrap,
            mc_sampling_method=mc_sampling_method,
            bound_type=bound_type,
            validation_mode=validation_mode,
        )

    def set_settings(self, settings: AnalysisSettings):
        """
        Restore widget state from an AnalysisSettings instance (e.g., when
        loading a saved project file).

        Parameters:
            settings: AnalysisSettings to apply to the UI.
        """
        # Block signals while bulk-loading to avoid spurious emissions
        self._loading = True
        self.blockSignals(True)
        try:
            # Coverage
            for idx in range(self._cmb_coverage.count()):
                if self._cmb_coverage.itemData(idx) == settings.coverage:
                    self._cmb_coverage.setCurrentIndex(idx)
                    break

            # Confidence
            for idx in range(self._cmb_confidence.count()):
                if self._cmb_confidence.itemData(idx) == settings.confidence:
                    self._cmb_confidence.setCurrentIndex(idx)
                    break

            # One-sided / two-sided
            if settings.one_sided:
                self._rb_one_sided.setChecked(True)
            else:
                self._rb_two_sided.setChecked(True)

            # k-method
            k_reverse = {
                K_METHOD_VV20: 0,
                K_METHOD_WS: 1,
                K_METHOD_TOLERANCE: 2,
                K_METHOD_MANUAL: 3,
            }
            btn_id = k_reverse.get(settings.k_method, 0)
            btn = self._bg_k_method.button(btn_id)
            if btn:
                btn.setChecked(True)
            self._spn_manual_k.setValue(settings.manual_k)
            self._spn_manual_k.setEnabled(btn_id == 3)

            # MC sampling method (backward-compat: default to Random)
            mc_method = getattr(settings, 'mc_sampling_method',
                                "Monte Carlo (Random)")
            for idx in range(self._cmb_mc_sampling.count()):
                if self._cmb_mc_sampling.itemData(idx) == mc_method:
                    self._cmb_mc_sampling.setCurrentIndex(idx)
                    break

            # Monte Carlo trials
            for idx in range(self._cmb_mc_trials.count()):
                if self._cmb_mc_trials.itemData(idx) == settings.mc_n_trials:
                    self._cmb_mc_trials.setCurrentIndex(idx)
                    break

            # Seed
            self._spn_seed.setValue(settings.mc_seed if settings.mc_seed else 0)

            # Bootstrap
            self._chk_bootstrap.setChecked(settings.mc_bootstrap)

            # Bound type
            bound_reverse = {
                "Known uncertainties only (u_val)": 0,
                "Total observed scatter (s_E)": 1,
                "Both (for comparison)": 2,
            }
            bound_id = bound_reverse.get(settings.bound_type, 2)
            bound_btn = self._bg_bound.button(bound_id)
            if bound_btn:
                bound_btn.setChecked(True)

            val_mode = getattr(settings, 'validation_mode', "Standard scalar (V&V 20)")
            for idx in range(self._cmb_validation_mode.count()):
                if self._cmb_validation_mode.itemData(idx) == val_mode:
                    self._cmb_validation_mode.setCurrentIndex(idx)
                    break
        finally:
            self.blockSignals(False)
            self._loading = False

        audit_log.log(
            "SETTINGS_LOADED",
            "Analysis settings restored from project file."
        )
        self.settings_changed.emit()

    def update_k_display(self, nu_eff: float, k_val: float):
        """
        Update the live readout labels under the Welch-Satterthwaite
        radio button. Called by the RSS computation engine when results
        are available.

        Parameters:
            nu_eff: Effective degrees of freedom from the W-S equation.
            k_val:  Computed coverage factor k from t-distribution.
        """
        if nu_eff == float('inf'):
            nu_str = "\u221e"
        else:
            nu_str = f"{nu_eff:.1f}"
        self._lbl_nu_eff.setText(f"\u03bd_eff = {nu_str}")
        self._lbl_k_ws.setText(f"k = {k_val:.3f}")

        # Update guidance if nu_eff is low and V&V 20 default is selected
        if self._rb_k_vv20.isChecked() and nu_eff < 30:
            self._guidance_k.set_guidance(
                f"\u03bd_eff = {nu_str} < 30 \u2014 the V&V 20 default k=2 "
                f"may be non-conservative. Consider switching to "
                f"Welch\u2013Satterthwaite (k = {k_val:.3f}) or the "
                f"tolerance factor method. [V&V 20 \u00a76, GUM Annex G]",
                'red'
            )
        elif self._rb_k_vv20.isChecked():
            self._guidance_k.set_guidance(
                f"\u03bd_eff = {nu_str} \u2265 30 \u2014 the V&V 20 default "
                f"k=2 is appropriate for this sample size. [V&V 20 \u00a76]",
                'green'
            )


# =============================================================================
# SECTION 10: TAB 4 — RSS RESULTS TAB
# =============================================================================
class RSSResultsTab(QWidget):
    """
    Tab 4: RSS Results Tab.

    Displays the RSS-based uncertainty combination results including:
      - Uncertainty budget table organized by V&V 20 category
      - Combined results with validation assessment
      - Variance contribution pie chart
      - Category magnitude bar chart
      - Normal PDF with expanded bounds overlay

    References:
        - ASME V&V 20-2009 Section 5 (RSS combination)
        - ASME V&V 20-2009 Section 6 (validation metrics)
        - JCGM 100:2008 (GUM) Section 5 (combined uncertainty)
        - JCGM 100:2008 (GUM) Annex G (Welch-Satterthwaite)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._results = RSSResults()
        self._budget_data: List[dict] = []
        self._setup_ui()

    # -----------------------------------------------------------------
    # UI CONSTRUCTION
    # -----------------------------------------------------------------
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self._splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self._splitter)

        # -- Left panel (scrollable text results) --
        left_container = QWidget()
        self._left_layout = QVBoxLayout(left_container)
        self._left_layout.setContentsMargins(8, 8, 4, 8)
        self._left_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_container)
        scroll.setMinimumWidth(560)
        self._splitter.addWidget(scroll)

        self._build_budget_table()
        self._build_results_display()
        self._build_guidance_panels()
        self._left_layout.addStretch()

        # -- Right panel (plots) --
        right_container = QWidget()
        self._right_layout = QVBoxLayout(right_container)
        self._right_layout.setContentsMargins(4, 8, 8, 8)
        self._splitter.addWidget(right_container)

        self._build_plot_section()

        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setSizes([650, 550])

    # -- Budget Table --
    def _build_budget_table(self):
        grp = QGroupBox("Uncertainty Budget Table")
        grp.setToolTip(
            "Itemized uncertainty budget showing each source's contribution\n"
            "to the combined validation uncertainty, organized by V&V 20\n"
            "category. [ASME V&V 20 Table 1, GUM Table H.1]"
        )
        lay = QVBoxLayout(grp)

        self._budget_table = QTableWidget()
        self._budget_table.setColumnCount(10)
        self._budget_table.setHorizontalHeaderLabels([
            "Source", "Category", "\u03c3 [unit]", "\u03c3\u00b2 [unit\u00b2]",
            "\u03bd (DOF)", "% of u_val\u00b2", "Distribution", "Data Basis",
            "Class", "Reducibility"
        ])
        style_table(self._budget_table,
                    column_widths={0: 160, 1: 110, 2: 80, 3: 90,
                                   4: 65, 5: 90, 6: 100, 7: 100,
                                   8: 80, 9: 85},
                    stretch_col=0)
        self._budget_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._budget_table.setAlternatingRowColors(True)
        self._budget_table.verticalHeader().setVisible(False)
        lay.addWidget(self._budget_table)

        self._left_layout.addWidget(grp)

    # -- Combined Results Display --
    def _build_results_display(self):
        grp = QGroupBox("Combined RSS Results")
        grp.setToolTip(
            "Full RSS analysis output: combined uncertainty, expanded\n"
            "uncertainty, validation assessment, model form estimate,\n"
            "and prediction bounds. [ASME V&V 20 \u00a75-6]"
        )
        lay = QVBoxLayout(grp)

        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMinimumHeight(400)
        self._results_text.setStyleSheet(
            f"QTextEdit {{"
            f"  background-color: {DARK_COLORS['bg_input']};"
            f"  color: {DARK_COLORS['fg']};"
            f"  font-family: 'Consolas', 'Courier New', monospace;"
            f"  font-size: 10pt;"
            f"  border: 1px solid {DARK_COLORS['border']};"
            f"  border-radius: 4px;"
            f"  padding: 8px;"
            f"}}"
        )
        lay.addWidget(self._results_text)

        self._left_layout.addWidget(grp)

    # -- Guidance Panels --
    def _build_guidance_panels(self):
        self._guidance_dominant = GuidancePanel("Dominant Source Check")
        self._guidance_dominant.setToolTip(
            "Flags when a single uncertainty source dominates the\n"
            "combined uncertainty (>80% of variance). This may indicate\n"
            "resources should focus on reducing that source.\n"
            "[GUM Section 5.1.3]"
        )
        self._left_layout.addWidget(self._guidance_dominant)

        self._guidance_dof = GuidancePanel("Degrees of Freedom Check")
        self._guidance_dof.setToolTip(
            "Warns when the effective degrees of freedom are low,\n"
            "meaning the coverage factor may be unreliable.\n"
            "[GUM Annex G, V&V 20 \u00a76]"
        )
        self._left_layout.addWidget(self._guidance_dof)

        self._guidance_model = GuidancePanel("Model Form Assessment")
        self._guidance_model.setToolTip(
            "Assesses whether the observed comparison scatter exceeds\n"
            "the known uncertainty sources, indicating unaccounted\n"
            "model form uncertainty. [V&V 20 \u00a76.3]"
        )
        self._left_layout.addWidget(self._guidance_model)

        self._guidance_bias = GuidancePanel("Validation Assessment")
        self._guidance_bias.setToolTip(
            "Core V&V 20 validation metric: compares the absolute\n"
            "mean comparison error to the expanded uncertainty.\n"
            "[ASME V&V 20 Eq. (1)]"
        )
        self._left_layout.addWidget(self._guidance_bias)

        self._guidance_unit_mismatch = GuidancePanel("Unit Consistency Check")
        self._guidance_unit_mismatch.setVisible(False)
        self._left_layout.addWidget(self._guidance_unit_mismatch)

    # -- Plot Section --
    def _build_plot_section(self):
        plt.rcParams.update(PLOT_STYLE)

        self._fig = Figure(figsize=(5.2, 8), dpi=100)
        self._fig.set_facecolor(PLOT_STYLE['figure.facecolor'])

        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumWidth(400)
        self._canvas.setMinimumHeight(550)

        toolbar_row = make_plot_toolbar_with_copy(
            self._canvas, self._fig, self,
            method_context="RSS ASME V&V 20")

        # Wrap canvas in a scroll area for horizontal scrolling
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        plot_container = QWidget()
        plot_lay = QVBoxLayout(plot_container)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        plot_lay.addWidget(self._canvas)
        plot_scroll.setWidget(plot_container)

        self._right_layout.addWidget(toolbar_row)
        self._right_layout.addWidget(plot_scroll)

    # -----------------------------------------------------------------
    # PUBLIC API
    # -----------------------------------------------------------------
    def compute_and_display(self, sources: List['UncertaintySource'],
                            comp_data: 'ComparisonData',
                            settings: 'AnalysisSettings'):
        """
        Main computation method.  Performs the full RSS uncertainty
        combination, validation assessment, and updates all displays.

        Parameters:
            sources:   list of UncertaintySource objects
            comp_data: ComparisonData with the comparison error array
            settings:  AnalysisSettings (coverage, confidence, k-method, etc.)
        """
        audit_log.log_computation("RSS_START", "Beginning RSS analysis")

        unit = settings.global_unit

        # ----------------------------------------------------------
        # 1. Extract sigma and categorize
        # ----------------------------------------------------------
        cat_sigmas = {"u_num": [], "u_input": [], "u_D": []}
        cat_dofs = {"u_num": [], "u_input": [], "u_D": []}
        all_sigmas = []
        all_dofs = []
        contributions = []

        for src in sources:
            sigma = src.get_standard_uncertainty()
            dof = src.get_dof()
            cat_key = src.get_category_key()

            cat_sigmas[cat_key].append(sigma)
            cat_dofs[cat_key].append(dof)
            all_sigmas.append(sigma)
            all_dofs.append(dof)

            contrib_entry = {
                'name': src.name,
                'category': src.category,
                'category_key': cat_key,
                'sigma': sigma,
                'sigma_sq': sigma ** 2,
                'dof': dof,
                'distribution': src.distribution,
                'data_basis': src.sigma_basis,
                'uncertainty_class': src.uncertainty_class,
                'basis_type': src.basis_type,
                'reducibility': src.reducibility,
            }
            if src.asymmetric:
                contrib_entry['asymmetric'] = True
                contrib_entry['sigma_upper'] = src.get_sigma_upper()
                contrib_entry['sigma_lower'] = src.get_sigma_lower()
                contrib_entry['one_sided'] = src.one_sided
                contrib_entry['mirror_assumed'] = src.mirror_assumed
            contributions.append(contrib_entry)

        audit_log.log_computation(
            "RSS_SOURCES",
            f"Processed {len(sources)} sources: "
            + ", ".join(f"{s.name}(\u03c3={s.get_standard_uncertainty():.4g})" for s in sources)
        )

        # ----------------------------------------------------------
        # 2. Compute category-level RSS (correlation-aware)
        # ----------------------------------------------------------
        # Build per-category lists of (sigma, correlation_group, rho, name)
        cat_sources: dict[str, list] = {"u_num": [], "u_input": [], "u_D": []}
        for src in sources:
            cat_key = src.get_category_key()
            sigma = src.get_standard_uncertainty()
            cat_sources[cat_key].append(
                (sigma, src.correlation_group.strip(), src.correlation_coefficient, src.name)
            )

        def _correlated_rss(entries: list) -> tuple[float, bool, list]:
            """Compute RSS with correlation support.

            Returns (combined_sigma, has_correlations, corr_matrices).
            entries: list of (sigma, group, rho, name).
            corr_matrices: list of (group_name, source_names, C_submatrix).
            When a non-empty correlation group exists, build the full
            correlation matrix C and compute sqrt(σᵀ · C · σ).
            Falls back to independent sqrt(Σσ²) when all groups are empty.
            """
            if not entries:
                return 0.0, False, []
            sigmas = np.array([e[0] for e in entries])
            groups = [e[1] for e in entries]
            rhos = [e[2] for e in entries]

            has_corr = any(g != "" for g in groups)
            if not has_corr:
                # Standard independent quadrature
                return float(np.sqrt(np.sum(sigmas ** 2))), False, []

            # Build correlation matrix
            n = len(entries)
            C = np.eye(n)
            # Group sources by correlation group
            group_indices: dict[str, list[int]] = {}
            for i, g in enumerate(groups):
                if g:
                    group_indices.setdefault(g, []).append(i)

            # Within each group, assign pairwise correlations
            # Reference source = first alphabetically (ρ_ref = 1.0 implicitly)
            # Other sources have ρ_i with reference, and ρ_ij ≈ ρ_i · ρ_j
            for g_name, indices in group_indices.items():
                if len(indices) < 2:
                    continue
                # Sort alphabetically by source name for deterministic reference
                indices.sort(key=lambda i: entries[i][3].lower())
                group_rhos = [rhos[i] for i in indices]
                # Reference source (first alphabetically) has ρ = 1.0 implicitly
                group_rhos[0] = 1.0
                ref_name = entries[indices[0]][3]
                audit_log.log_computation(
                    "CORR_REF",
                    f"Group '{g_name}': reference source = '{ref_name}' "
                    f"(first alphabetically, ρ = 1.0)"
                )
                # Warn about rho=0.0 in a group (effectively independent)
                for k_idx, rho_k in zip(indices[1:], group_rhos[1:]):
                    if rho_k == 0.0:
                        src_name = entries[k_idx][3]
                        audit_log.log_assumption(
                            f"Source '{src_name}' is in correlation group '{g_name}' "
                            f"but has ρ = 0.0 — effectively independent of group members."
                        )
                # Build pairwise correlations via transitivity
                for a_pos, a_idx in enumerate(indices):
                    for b_pos, b_idx in enumerate(indices):
                        if a_idx == b_idx:
                            continue
                        # ρ(a,b) = ρ_a * ρ_b for transitivity
                        rho_a = group_rhos[a_pos]
                        rho_b = group_rhos[b_pos]
                        C[a_idx, b_idx] = rho_a * rho_b

            # Collect per-group sub-matrices for reporting
            corr_detail: list[tuple] = []
            for g_name, indices in group_indices.items():
                if len(indices) < 2:
                    continue
                names = [entries[i][3] for i in indices]
                sub = np.eye(len(indices))
                for ap, ai in enumerate(indices):
                    for bp, bi in enumerate(indices):
                        if ai != bi:
                            sub[ap, bp] = C[ai, bi]
                corr_detail.append((g_name, names, sub))

            # Ensure positive semi-definiteness
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < -1e-10):
                audit_log.log_computation(
                    "RSS_CORR_WARNING",
                    f"Correlation matrix has negative eigenvalues "
                    f"(min={float(eigvals.min()):.4g}). Falling back "
                    f"to independent RSS."
                )
                return float(np.sqrt(np.sum(sigmas ** 2))), False, []

            # σᵀ · C · σ
            var = float(sigmas @ C @ sigmas)
            return float(np.sqrt(max(var, 0.0))), True, corr_detail

        u_num, corr_num, cm_num = _correlated_rss(cat_sources["u_num"])
        u_input, corr_inp, cm_inp = _correlated_rss(cat_sources["u_input"])
        u_D, corr_D, cm_D = _correlated_rss(cat_sources["u_D"])
        has_any_correlation = corr_num or corr_inp or corr_D
        all_corr_matrices = cm_num + cm_inp + cm_D

        # ----------------------------------------------------------
        # 3. Combined standard uncertainty
        # ----------------------------------------------------------
        u_val = np.sqrt(u_num ** 2 + u_input ** 2 + u_D ** 2)

        corr_label = ("with correlation" if has_any_correlation
                       else "independent (no correlations)")
        audit_log.log_computation(
            "RSS_COMBINED",
            f"u_num={u_num:.6g}, u_input={u_input:.6g}, u_D={u_D:.6g}, "
            f"u_val={u_val:.6g} [{unit}] ({corr_label})"
        )
        if has_any_correlation:
            corr_groups = set(
                src.correlation_group.strip() for src in sources
                if src.correlation_group.strip()
            )
            audit_log.log_assumption(
                f"Correlation matrix applied for groups: "
                f"{', '.join(sorted(corr_groups))}. "
                f"Cross-category correlations assumed zero."
            )
        else:
            audit_log.log_assumption(
                "All uncertainty sources treated as independent — "
                "no correlation groups defined."
            )

        # ----------------------------------------------------------
        # 4. Effective DOF via Welch-Satterthwaite
        # ----------------------------------------------------------
        if len(all_sigmas) > 0 and u_val > 0:
            nu_eff = welch_satterthwaite(all_sigmas, all_dofs)
        else:
            nu_eff = float('inf')

        nu_eff_str = "\u221e" if np.isinf(nu_eff) else f"{nu_eff:.1f}"
        audit_log.log_computation(
            "RSS_DOF",
            f"Welch-Satterthwaite \u03bd_eff = {nu_eff_str}"
        )

        # ----------------------------------------------------------
        # 5. Determine coverage factor k
        # ----------------------------------------------------------
        k_method_used = settings.k_method
        two_sided = not settings.one_sided

        if settings.k_method == K_METHOD_VV20:
            k_factor = 2.0
        elif settings.k_method == K_METHOD_WS:
            k_factor = coverage_factor_from_dof(
                nu_eff, coverage=settings.coverage, two_sided=two_sided
            )
        elif settings.k_method == K_METHOD_TOLERANCE:
            n_equiv = int(nu_eff + 1) if not np.isinf(nu_eff) else 10000
            if settings.one_sided:
                k_factor = one_sided_tolerance_k(
                    n_equiv, coverage=settings.coverage,
                    confidence=settings.confidence
                )
            else:
                k_factor = two_sided_tolerance_k(
                    n_equiv, coverage=settings.coverage,
                    confidence=settings.confidence
                )
        elif settings.k_method == K_METHOD_MANUAL:
            k_factor = settings.manual_k
            if k_factor <= 0:
                audit_log.log_warning(
                    "MANUAL_K_INVALID",
                    f"Manual k = {k_factor:.4f} is not valid (must be > 0). "
                    f"Falling back to k = 2.0."
                )
                k_factor = 2.0
            elif k_factor < 1.0:
                audit_log.log_warning(
                    "MANUAL_K_LOW",
                    f"Manual k = {k_factor:.4f} is unusually low. "
                    f"Typical values range from 1.5 to 3.0. A low k-factor "
                    f"may produce a non-conservative expanded uncertainty."
                )
        else:
            k_factor = 2.0

        audit_log.log_computation(
            "RSS_K_FACTOR",
            f"k = {k_factor:.4f} (method: {k_method_used})"
        )

        # ----------------------------------------------------------
        # 6. Expanded uncertainty
        # ----------------------------------------------------------
        U_val = k_factor * u_val

        # ----------------------------------------------------------
        # 7. Comparison error statistics
        # ----------------------------------------------------------
        flat = comp_data.flat_data()
        if flat.size > 0:
            E_mean = float(np.mean(flat))
            s_E = float(np.std(flat, ddof=1)) if flat.size > 1 else 0.0
            n_data = int(flat.size)
            p5 = float(np.percentile(flat, 5))
            p95 = float(np.percentile(flat, 95))
        else:
            E_mean = 0.0
            s_E = 0.0
            n_data = 0
            p5 = 0.0
            p95 = 0.0

        validation_mode = getattr(settings, 'validation_mode', "Standard scalar (V&V 20)")
        mv_enabled = "multivariate" in validation_mode.lower()
        if mv_enabled:
            mv = compute_multivariate_validation(comp_data)
            audit_log.log_computation(
                "RSS_MULTIVARIATE",
                (
                    f"enabled={mv_enabled}, computed={mv.get('computed', False)}, "
                    f"score={mv.get('score', 0.0):.4g}, p={mv.get('pvalue', 1.0):.4g}, "
                    f"locations={mv.get('n_locations', 0)}, "
                    f"conditions={mv.get('n_conditions', 0)}"
                ),
            )
        else:
            mv = {
                'computed': False,
                'score': 0.0,
                't2': 0.0,
                'pvalue': 1.0,
                'n_locations': 0,
                'n_conditions': 0,
                'note': "Multivariate supplement disabled.",
            }

        # ----------------------------------------------------------
        # 8. Validation assessment (V&V 20 Eq. 1)
        # ----------------------------------------------------------
        if n_data > 0:
            bias_explained = abs(E_mean) <= U_val
            audit_log.log_computation(
                "RSS_VALIDATION",
                f"|E_mean| = {abs(E_mean):.6g} vs U_val = {U_val:.6g} => "
                f"Bias {'IS' if bias_explained else 'IS NOT'} explained"
            )
        else:
            bias_explained = None  # Cannot determine without comparison data
            audit_log.log_computation(
                "RSS_VALIDATION_SKIPPED",
                "No comparison data loaded — validation verdict "
                "cannot be determined."
            )

        # ----------------------------------------------------------
        # 9. Model form uncertainty estimate
        # ----------------------------------------------------------
        if s_E > u_val and u_val > 0:
            u_model = np.sqrt(s_E ** 2 - u_val ** 2)
            model_form_pct = ((s_E ** 2 - u_val ** 2) / s_E ** 2) * 100.0
        else:
            u_model = 0.0
            model_form_pct = 0.0

        # ----------------------------------------------------------
        # 10. Prediction bounds (gated by bound_type setting)
        # ----------------------------------------------------------
        lower_bound_uval = E_mean - k_factor * u_val
        upper_bound_uval = E_mean + k_factor * u_val
        lower_bound_sE = E_mean - k_factor * s_E
        upper_bound_sE = E_mean + k_factor * s_E

        bound_type = getattr(settings, 'bound_type', 'Both (for comparison)')
        bt_lower = bound_type.lower()
        if 'both' not in bt_lower:
            if 'u_val' in bt_lower and 's_e' not in bt_lower:
                # "Known uncertainties only (u_val)" — suppress s_E bounds
                lower_bound_sE = float('nan')
                upper_bound_sE = float('nan')
            elif 's_e' in bt_lower and 'u_val' not in bt_lower:
                # "Total observed scatter (s_E)" — suppress u_val bounds
                lower_bound_uval = float('nan')
                upper_bound_uval = float('nan')
        # else: "Both (for comparison)" — keep both

        audit_log.log_computation(
            "RSS_BOUNDS",
            f"Bound mode: {bound_type}. "
            f"u_val bounds: [{lower_bound_uval:.6g}, {upper_bound_uval:.6g}], "
            f"s_E bounds: [{lower_bound_sE:.6g}, {upper_bound_sE:.6g}] [{unit}]"
        )

        # ----------------------------------------------------------
        # Compute uncertainty class split (U_A, U_E, U_mixed)
        # ----------------------------------------------------------
        class_sigmas = {'aleatoric': [], 'epistemic': [], 'mixed': []}
        for c in contributions:
            cls = c.get('uncertainty_class', 'aleatoric')
            if cls not in ('aleatoric', 'epistemic', 'mixed'):
                cls = 'aleatoric'
            class_sigmas[cls].append(c['sigma'])
        u_aleatoric = float(np.sqrt(sum(s**2 for s in class_sigmas['aleatoric'])))
        u_epistemic = float(np.sqrt(sum(s**2 for s in class_sigmas['epistemic'])))
        u_mixed_val = float(np.sqrt(sum(s**2 for s in class_sigmas['mixed'])))
        u_val_sq_for_pct = u_val ** 2 if u_val > 0 else 1.0
        pct_epistemic = (u_epistemic ** 2 / u_val_sq_for_pct * 100.0) if u_val > 0 else 0.0

        # ----------------------------------------------------------
        # Store results
        # ----------------------------------------------------------
        self._results = RSSResults(
            u_num=float(u_num),
            u_input=float(u_input),
            u_D=float(u_D),
            u_val=float(u_val),
            nu_eff=float(nu_eff),
            k_factor=float(k_factor),
            k_method_used=k_method_used,
            U_val=float(U_val),
            E_mean=float(E_mean),
            s_E=float(s_E),
            n_data=n_data,
            lower_bound_uval=float(lower_bound_uval),
            upper_bound_uval=float(upper_bound_uval),
            lower_bound_sE=float(lower_bound_sE),
            upper_bound_sE=float(upper_bound_sE),
            u_model=float(u_model),
            model_form_pct=float(model_form_pct),
            bias_explained=bias_explained,
            source_contributions=contributions,
            computed=True,
            has_correlations=has_any_correlation,
            correlation_groups=sorted(set(
                s.correlation_group.strip() for s in sources
                if s.correlation_group.strip()
            )),
            correlation_matrices=all_corr_matrices,
            u_aleatoric=u_aleatoric,
            u_epistemic=u_epistemic,
            u_mixed=u_mixed_val,
            pct_epistemic=pct_epistemic,
            multivariate_enabled=mv_enabled,
            multivariate_computed=bool(mv.get('computed', False)),
            multivariate_score=float(mv.get('score', 0.0)),
            multivariate_t2=float(mv.get('t2', 0.0)),
            multivariate_pvalue=float(mv.get('pvalue', 1.0)),
            multivariate_n_locations=int(mv.get('n_locations', 0)),
            multivariate_n_conditions=int(mv.get('n_conditions', 0)),
            multivariate_note=str(mv.get('note', "")),
        )

        # Build budget table data
        self._budget_data = self._build_budget_data(contributions, u_val, unit)

        # ----------------------------------------------------------
        # Update all displays
        # ----------------------------------------------------------
        self._update_budget_table(unit)
        self._update_results_text(unit, settings, p5, p95)
        self._update_guidance_panels(contributions, u_val, nu_eff, s_E,
                                     bias_explained, E_mean, U_val, unit)
        self._update_plots(contributions, u_val, u_num, u_input, u_D,
                           E_mean, s_E, k_factor, p5, p95, unit, settings)

        audit_log.log_computation("RSS_COMPLETE", "RSS analysis completed")

    def get_results(self) -> RSSResults:
        """Return the most recently computed RSS results."""
        return self._results

    def clear_results(self):
        """Reset RSS results and clear all displays."""
        self._results = RSSResults()
        self._budget_data = []
        self._budget_table.setRowCount(0)
        self._results_text.clear()
        self._fig.clear()
        self._canvas.draw_idle()
        for gp in [self._guidance_dominant, self._guidance_dof,
                    self._guidance_model, self._guidance_bias]:
            gp.set_guidance("", "green")
        self._guidance_unit_mismatch.setVisible(False)

    def show_unit_mismatch(self, mismatched: list, global_unit: str):
        """Show a prominent unit mismatch warning on the RSS tab."""
        names = ", ".join(mismatched[:5])
        if len(mismatched) > 5:
            names += f", ... (+{len(mismatched) - 5} more)"
        self._guidance_unit_mismatch.set_guidance(
            f"UNIT MISMATCH: {len(mismatched)} source(s) use different "
            f"units than the global setting '{global_unit}':\n"
            f"{names}\n\n"
            f"The RSS combination assumes all sigma values are in the "
            f"same unit. Mixing units (e.g., °F and K) will produce "
            f"incorrect results. Fix the mismatched sources on Tab 2 "
            f"or change the global unit on Tab 3.",
            'yellow'
        )
        self._guidance_unit_mismatch.setVisible(True)

    def hide_unit_mismatch(self):
        """Hide the unit mismatch warning."""
        self._guidance_unit_mismatch.setVisible(False)

    def get_budget_table_data(self) -> list:
        """Return the budget table data for export."""
        return list(self._budget_data)

    # -----------------------------------------------------------------
    # BUDGET TABLE CONSTRUCTION
    # -----------------------------------------------------------------
    def _build_budget_data(self, contributions: List[dict],
                           u_val: float, unit: str) -> List[dict]:
        """Build the budget data list with subtotals and grand total."""
        budget = []
        u_val_sq = u_val ** 2 if u_val > 0 else 1.0

        # Group by category key
        by_cat = {"u_num": [], "u_input": [], "u_D": []}
        for c in contributions:
            by_cat[c['category_key']].append(c)

        cat_labels = {
            "u_num": "Numerical (u_num)",
            "u_input": "Input/BC (u_input)",
            "u_D": "Experimental (u_D)",
        }

        for cat_key in CATEGORY_KEYS:
            items = by_cat[cat_key]
            if not items:
                continue

            for item in items:
                pct = (item['sigma_sq'] / u_val_sq * 100.0) if u_val_sq > 0 else 0.0
                entry = {
                    'name': item['name'],
                    'category': item['category'],
                    'sigma': item['sigma'],
                    'sigma_sq': item['sigma_sq'],
                    'dof': item['dof'],
                    'pct': pct,
                    'distribution': item['distribution'],
                    'data_basis': item['data_basis'],
                    'uncertainty_class': item.get('uncertainty_class', 'aleatoric'),
                    'basis_type': item.get('basis_type', 'measured'),
                    'reducibility': item.get('reducibility', 'low'),
                    'is_subtotal': False,
                    'is_total': False,
                }
                if item.get('asymmetric'):
                    entry['asymmetric'] = True
                    entry['sigma_upper'] = item['sigma_upper']
                    entry['sigma_lower'] = item['sigma_lower']
                budget.append(entry)

            # Subtotal row
            sub_sigma_sq = sum(it['sigma_sq'] for it in items)
            sub_sigma = np.sqrt(sub_sigma_sq)
            sub_pct = (sub_sigma_sq / u_val_sq * 100.0) if u_val_sq > 0 else 0.0

            # Welch-Satterthwaite for subtotal DOF
            sub_sigmas = [it['sigma'] for it in items]
            sub_dofs = [it['dof'] for it in items]
            if len(sub_sigmas) > 0 and sub_sigma > 0:
                sub_nu = welch_satterthwaite(sub_sigmas, sub_dofs)
            else:
                sub_nu = float('inf')

            budget.append({
                'name': f"  Subtotal: {cat_labels[cat_key]}",
                'category': '',
                'sigma': sub_sigma,
                'sigma_sq': sub_sigma_sq,
                'dof': sub_nu,
                'pct': sub_pct,
                'distribution': '',
                'data_basis': '',
                'is_subtotal': True,
                'is_total': False,
            })

        # Grand total row
        total_pct = 100.0 if u_val > 0 else 0.0
        budget.append({
            'name': "  TOTAL: u_val (combined)",
            'category': '',
            'sigma': u_val,
            'sigma_sq': u_val ** 2,
            'dof': self._results.nu_eff,
            'pct': total_pct,
            'distribution': '',
            'data_basis': '',
            'is_subtotal': False,
            'is_total': True,
        })

        return budget

    def _update_budget_table(self, unit: str):
        """Populate the QTableWidget from the budget data."""
        self._budget_table.setRowCount(0)
        self._budget_table.setRowCount(len(self._budget_data))

        # Update column headers with unit
        has_corr = (self._results is not None
                    and getattr(self._results, 'has_correlations', False))
        pct_header = "% of u_val\u00b2 \u2020" if has_corr else "% of u_val\u00b2"
        self._budget_table.setHorizontalHeaderLabels([
            "Source", "Category",
            f"\u03c3 (Sigma) [{unit}]", f"\u03c3\u00b2 (Variance) [{unit}\u00b2]",
            "DOF (\u03bd)", pct_header, "Distribution", "Data Basis",
            "Class", "Reducibility"
        ])
        if has_corr:
            h5 = self._budget_table.horizontalHeaderItem(5)
            if h5:
                h5.setToolTip(
                    "\u2020 With correlated sources, percentages do not sum\n"
                    "to 100% because cross-correlation terms also\n"
                    "contribute to u_val\u00b2."
                )

        for row, item in enumerate(self._budget_data):
            is_special = item['is_subtotal'] or item['is_total']

            # Source name
            name_item = QTableWidgetItem(item['name'])
            if is_special:
                font = name_item.font()
                font.setBold(True)
                name_item.setFont(font)
            self._budget_table.setItem(row, 0, name_item)

            # Category
            self._budget_table.setItem(row, 1, QTableWidgetItem(item['category']))

            # Sigma (with asymmetric indicator)
            if item.get('asymmetric'):
                sp = item.get('sigma_upper', 0)
                sm = item.get('sigma_lower', 0)
                sigma_text = f"{item['sigma']:.4g} (\u03c3\u207a={sp:.3g} / \u03c3\u207b={sm:.3g})"
            else:
                sigma_text = f"{item['sigma']:.4g}"
            sigma_item = QTableWidgetItem(sigma_text)
            sigma_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if item.get('asymmetric'):
                sigma_item.setToolTip(
                    f"\u03c3\u207a (upper) = {item.get('sigma_upper', 0):.6g}\n"
                    f"\u03c3\u207b (lower) = {item.get('sigma_lower', 0):.6g}\n"
                    f"Effective \u03c3 = \u221a((\u03c3\u207a\u00b2 + \u03c3\u207b\u00b2) / 2)"
                    f" = {item['sigma']:.6g}\n"
                    f"[GUM \u00a74.3.8]"
                )
            if is_special:
                font = sigma_item.font()
                font.setBold(True)
                sigma_item.setFont(font)
            self._budget_table.setItem(row, 2, sigma_item)

            # Sigma squared
            sq_item = QTableWidgetItem(f"{item['sigma_sq']:.4g}")
            sq_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            if is_special:
                font = sq_item.font()
                font.setBold(True)
                sq_item.setFont(font)
            self._budget_table.setItem(row, 3, sq_item)

            # DOF
            dof_val = item['dof']
            if np.isinf(dof_val):
                dof_str = "\u221e"
            else:
                dof_str = f"{dof_val:.1f}"
            dof_item = QTableWidgetItem(dof_str)
            dof_item.setTextAlignment(Qt.AlignCenter)
            self._budget_table.setItem(row, 4, dof_item)

            # Percent of u_val^2
            pct = item['pct']
            pct_item = QTableWidgetItem(f"{pct:.1f}%")
            pct_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)

            # Color coding for dominance
            if not is_special:
                if pct > 80.0:
                    pct_item.setBackground(QColor(DARK_COLORS['red']))
                    pct_item.setForeground(QColor("#ffffff"))
                elif pct > 50.0:
                    pct_item.setBackground(QColor(DARK_COLORS['yellow']))
                    pct_item.setForeground(QColor("#1e1e2e"))
            if is_special:
                font = pct_item.font()
                font.setBold(True)
                pct_item.setFont(font)
            self._budget_table.setItem(row, 5, pct_item)

            # Distribution
            self._budget_table.setItem(row, 6, QTableWidgetItem(item['distribution']))

            # Data basis
            self._budget_table.setItem(row, 7, QTableWidgetItem(item['data_basis']))

            # Class and Reducibility
            self._budget_table.setItem(row, 8, QTableWidgetItem(
                item.get('uncertainty_class', '')))
            self._budget_table.setItem(row, 9, QTableWidgetItem(
                item.get('reducibility', '')))

            # Style subtotal/total rows
            if item['is_subtotal']:
                for col in range(10):
                    cell = self._budget_table.item(row, col)
                    if cell:
                        cell.setBackground(QColor(DARK_COLORS['bg_alt']))
            elif item['is_total']:
                for col in range(10):
                    cell = self._budget_table.item(row, col)
                    if cell:
                        cell.setBackground(QColor(DARK_COLORS['accent']))
                        cell.setForeground(QColor("#1e1e2e"))

        self._budget_table.resizeRowsToContents()

    # -----------------------------------------------------------------
    # RESULTS TEXT
    # -----------------------------------------------------------------
    def _update_results_text(self, unit: str, settings: 'AnalysisSettings',
                             p5: float, p95: float):
        """Build and display the formatted results text."""
        r = self._results
        nu_str = "\u221e" if np.isinf(r.nu_eff) else f"{r.nu_eff:.1f}"
        side_str = "One-sided" if settings.one_sided else "Two-sided"
        cov_pct = settings.coverage * 100.0
        conf_pct = settings.confidence * 100.0

        lines = []
        lines.append("=" * 60)
        lines.append("  RSS UNCERTAINTY COMBINATION RESULTS")
        lines.append("=" * 60)
        lines.append("")
        lines.append(f"Combined Standard Uncertainty: u_val = {r.u_val:.4f} [{unit}]")
        lines.append(f"Effective DOF (Welch-Satterthwaite): \u03bd_eff = {nu_str}")
        lines.append(f"Coverage Factor: k = {r.k_factor:.4f}  [method: {r.k_method_used}]")
        lines.append(f"Expanded Uncertainty: U_val = {r.U_val:.4f} [{unit}]")
        bound_type = getattr(settings, 'bound_type', 'Both (for comparison)')
        lines.append(f"Bound mode: {bound_type}")
        if r.has_correlations:
            lines.append(
                f"Source correlation: Applied (groups: "
                f"{', '.join(r.correlation_groups)})"
            )
            # Effective pairwise correlation matrices
            for grp_name, names, C_mat in r.correlation_matrices:
                lines.append("")
                lines.append(f"  Group \"{grp_name}\" effective \u03c1 matrix:")
                # Header row
                max_name = max(len(n) for n in names)
                pad = max(max_name, 8)
                hdr = " " * (pad + 4) + "  ".join(
                    f"{n:>{pad}}" for n in names
                )
                lines.append(f"  {hdr}")
                # Data rows
                for row_idx, row_name in enumerate(names):
                    vals = "  ".join(
                        f"{C_mat[row_idx, col_idx]:>{pad}.2f}"
                        for col_idx in range(len(names))
                    )
                    lines.append(f"    {row_name:>{pad}}  {vals}")
        else:
            lines.append("Source correlation: Independent (no correlations)")
        lines.append("")

        # Uncertainty class split
        lines.append(
            f"Uncertainty class split: U_A={r.u_aleatoric:.4f}, "
            f"U_E={r.u_epistemic:.4f}, U_mixed={r.u_mixed:.4f} [{unit}]"
        )
        lines.append(
            f"  Epistemic fraction: {r.pct_epistemic:.1f}% of total variance"
        )
        lines.append(
            "  Note: class labels guide prioritization/reporting; the RSS and "
            "MC combination math still includes all enabled sources."
        )
        lines.append(
            "  ASME V&V 20 compliance: The RSS combination of all sources "
            "(regardless of epistemic/aleatory class) follows ASME V&V 20-2009 "
            "(R2021) Section 9 and the GUM (JCGM 100:2008) framework. V&V 20 "
            "converts all uncertainty sources to standard uncertainties (1-sigma) "
            "before RSS combination."
        )
        if r.pct_epistemic > 50:
            lines.append(
                "  Caveat: Epistemic sources dominate this budget (>50%). The RSS "
                "assumption of random cancellation may be non-conservative for "
                "systematic/knowledge-gap uncertainties. For stricter separation, "
                "frameworks such as Oberkampf & Roy (2010) recommend double-loop "
                "Monte Carlo producing probability boxes (p-boxes). This tool "
                "follows V&V 20's pragmatic engineering approach."
            )

        # Dominant drivers by class
        u_val_sq_for_drv = r.u_val ** 2 if r.u_val > 0 else 1.0
        _dom_a_name = _dom_e_name = ""
        _dom_a_pct = _dom_e_pct = 0.0
        _dom_e_red = ""
        for c in r.source_contributions:
            cpct = (c['sigma'] ** 2 / u_val_sq_for_drv * 100.0) if r.u_val > 0 else 0.0
            cls = c.get('uncertainty_class', 'aleatoric')
            if cls == 'aleatoric' and cpct > _dom_a_pct:
                _dom_a_pct = cpct
                _dom_a_name = c['name']
            elif cls == 'epistemic' and cpct > _dom_e_pct:
                _dom_e_pct = cpct
                _dom_e_name = c['name']
                _dom_e_red = c.get('reducibility', '')
        if _dom_a_name:
            lines.append(
                f"  Dominant aleatoric driver: {_dom_a_name} ({_dom_a_pct:.1f}%)"
            )
        if _dom_e_name:
            red_tag = f", reducibility: {_dom_e_red}" if _dom_e_red else ""
            lines.append(
                f"  Dominant epistemic gap: {_dom_e_name} "
                f"({_dom_e_pct:.1f}%{red_tag})"
            )

        if r.pct_epistemic > 50.0:
            lines.append(
                "  \u26a0 Epistemic uncertainty dominates \u2014 consider "
                "knowledge-reduction actions before making compliance claims."
            )
        lines.append("")

        lines.append("-" * 60)
        lines.append("  Comparison Error Statistics")
        lines.append("-" * 60)
        lines.append(f"  Mean: \u0112 = {r.E_mean:+.4f} [{unit}]")
        lines.append(f"  Std Dev: s_E = {r.s_E:.4f} [{unit}]")
        lines.append(f"  Sample Size: n = {r.n_data}")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  Validation Assessment  [V&V 20 Eq. (1)]")
        lines.append("-" * 60)
        if r.bias_explained is None:
            lines.append(
                "  Validation assessment not available (no comparison data)."
            )
            lines.append(
                "  Load experimental comparison data on Tab 1 to enable "
                "the V&V 20 validation verdict."
            )
        else:
            bias_result = ("Bias IS explained by known uncertainties"
                           if r.bias_explained else
                           "Bias IS NOT explained by known uncertainties")
            lines.append(
                f"  |\u0112| vs U_val: {abs(r.E_mean):.4f} vs {r.U_val:.4f} "
                f"\u2192 [{bias_result}]"
            )
        lines.append("")

        if getattr(r, 'multivariate_enabled', False):
            lines.append("-" * 60)
            lines.append("  Multivariate Supplemental Validation (covariance-aware)")
            lines.append("-" * 60)
            if getattr(r, 'multivariate_computed', False):
                mv_pass = getattr(r, 'multivariate_pvalue', 1.0) >= 0.05
                mv_verdict = (
                    "Consistent with zero-bias hypothesis (p >= 0.05)"
                    if mv_pass else
                    "Potential structured multivariate bias (p < 0.05)"
                )
                lines.append(
                    f"  Normalized multivariate score: {r.multivariate_score:.4f}"
                )
                lines.append(
                    f"  Approx. Hotelling-style T\u00b2: {r.multivariate_t2:.4f}"
                )
                lines.append(
                    f"  Chi-square p-value: {r.multivariate_pvalue:.4f} "
                    f"\u2192 [{mv_verdict}]"
                )
                lines.append(
                    f"  Coverage of dataset: {r.multivariate_n_locations} locations "
                    f"x {r.multivariate_n_conditions} conditions"
                )
                if r.multivariate_note:
                    lines.append(f"  Note: {r.multivariate_note}")
            else:
                lines.append("  Multivariate supplement enabled but not computable.")
                if r.multivariate_note:
                    lines.append(f"  Note: {r.multivariate_note}")
            lines.append("")

        lines.append("-" * 60)
        lines.append("  Estimated Model Form Uncertainty")
        lines.append("-" * 60)
        if r.s_E > r.u_val and r.u_val > 0:
            lines.append(
                f"  u_model = \u221a(s_E\u00b2 - u_val\u00b2) = "
                f"\u221a({r.s_E:.4f}\u00b2 - {r.u_val:.4f}\u00b2) = "
                f"{r.u_model:.4f} [{unit}]"
            )
            lines.append(
                f"  Model form accounts for {r.model_form_pct:.1f}% "
                f"of total observed variance"
            )
        elif r.u_val > 0:
            lines.append(
                f"  s_E ({r.s_E:.4f}) \u2264 u_val ({r.u_val:.4f}) "
                f"\u2014 no additional model form uncertainty detected"
            )
            lines.append(
                f"  (Negative model form variance: known uncertainties "
                f"may be overestimated)"
            )
        else:
            lines.append("  No uncertainty sources defined.")
        lines.append("")

        lines.append("-" * 60)
        lines.append("  Prediction Bounds")
        lines.append("-" * 60)
        lines.append(
            f"  {side_str} {cov_pct:.0f}% coverage, "
            f"{conf_pct:.0f}% confidence"
        )
        lines.append("")
        if not np.isnan(r.lower_bound_uval):
            lines.append("  Using u_val only:")
            lines.append(
                f"    Lower bound = \u0112 - k \u00d7 u_val = "
                f"{r.E_mean:+.4f} - {r.k_factor:.4f} \u00d7 {r.u_val:.4f} = "
                f"{r.lower_bound_uval:+.4f} [{unit}]"
            )
            lines.append(
                f"    Upper bound = \u0112 + k \u00d7 u_val = "
                f"{r.E_mean:+.4f} + {r.k_factor:.4f} \u00d7 {r.u_val:.4f} = "
                f"{r.upper_bound_uval:+.4f} [{unit}]"
            )
            lines.append("")
        if not np.isnan(r.lower_bound_sE):
            lines.append("  Using s_E (includes model form):")
            lines.append(
                f"    Lower bound = \u0112 - k \u00d7 s_E = "
                f"{r.E_mean:+.4f} - {r.k_factor:.4f} \u00d7 {r.s_E:.4f} = "
                f"{r.lower_bound_sE:+.4f} [{unit}]"
            )
            lines.append(
                f"    Upper bound = \u0112 + k \u00d7 s_E = "
                f"{r.E_mean:+.4f} + {r.k_factor:.4f} \u00d7 {r.s_E:.4f} = "
                f"{r.upper_bound_sE:+.4f} [{unit}]"
            )
            lines.append("")
        lines.append("")
        lines.append("  Empirical percentiles:")
        lines.append(f"    5th percentile  = {p5:+.4f} [{unit}]")
        lines.append(f"    95th percentile = {p95:+.4f} [{unit}]")
        lines.append("")
        lines.append("=" * 60)

        self._results_text.setPlainText("\n".join(lines))

    # -----------------------------------------------------------------
    # GUIDANCE PANELS
    # -----------------------------------------------------------------
    def _update_guidance_panels(self, contributions: List[dict],
                                u_val: float, nu_eff: float,
                                s_E: float, bias_explained: Optional[bool],
                                E_mean: float, U_val: float, unit: str):
        """Update all guidance panels based on computation results."""
        u_val_sq = u_val ** 2 if u_val > 0 else 1.0

        # -- Dominant source check --
        dominant_name = None
        dominant_pct = 0.0
        any_over_50 = False
        for c in contributions:
            pct = (c['sigma_sq'] / u_val_sq * 100.0) if u_val_sq > 0 else 0.0
            if pct > dominant_pct:
                dominant_pct = pct
                dominant_name = c['name']
            if pct > 50.0:
                any_over_50 = True

        if dominant_pct > 80.0:
            self._guidance_dominant.set_guidance(
                f"DOMINANT SOURCE: \"{dominant_name}\" contributes "
                f"{dominant_pct:.1f}% of the total variance. "
                f"This single source controls the combined uncertainty. "
                f"Priority should be given to reducing this source. "
                f"[GUM Section 5.1.3]",
                'red'
            )
        elif any_over_50:
            self._guidance_dominant.set_guidance(
                f"Source \"{dominant_name}\" contributes {dominant_pct:.1f}% "
                f"of the total variance. While not fully dominant, it is "
                f"the primary contributor. [GUM Section 5.1.3]",
                'yellow'
            )
        else:
            self._guidance_dominant.set_guidance(
                "No single source dominates the uncertainty budget. "
                "Variance is reasonably distributed. [GUM Section 5.1.3]",
                'green'
            )

        # -- DOF check --
        if np.isinf(nu_eff):
            self._guidance_dof.set_guidance(
                "\u03bd_eff = \u221e \u2014 all sources are Type B (infinite DOF). "
                "Coverage factor is reliable. [GUM Annex G]",
                'green'
            )
        elif nu_eff < 5:
            self._guidance_dof.set_guidance(
                f"\u03bd_eff = {nu_eff:.1f} < 5 \u2014 very low effective "
                f"degrees of freedom. The coverage factor k is highly "
                f"sensitive to this value and may be unreliable. Consider "
                f"obtaining more data or using a tolerance-factor approach. "
                f"[GUM Annex G, V&V 20 \u00a76]",
                'red'
            )
        elif nu_eff < 30:
            self._guidance_dof.set_guidance(
                f"\u03bd_eff = {nu_eff:.1f} < 30 \u2014 moderate DOF. "
                f"The Student-t correction increases k above 2.0. "
                f"Consider using the Welch-Satterthwaite k-method "
                f"rather than the V&V 20 default k=2. "
                f"[GUM Annex G, V&V 20 \u00a76]",
                'yellow'
            )
        else:
            self._guidance_dof.set_guidance(
                f"\u03bd_eff = {nu_eff:.1f} \u2265 30 \u2014 sufficient "
                f"degrees of freedom. Coverage factor k is reliable. "
                f"[GUM Annex G]",
                'green'
            )

        # -- Model form assessment --
        if u_val > 0 and s_E > 0:
            if u_val > s_E:
                self._guidance_model.set_guidance(
                    f"u_val ({u_val:.4f}) > s_E ({s_E:.4f}) \u2014 negative "
                    f"model form variance. The known uncertainty sources "
                    f"exceed the observed scatter. This may indicate "
                    f"overestimated source uncertainties or fortuitous "
                    f"error cancellation. Review source estimates. "
                    f"[V&V 20 \u00a76.3]",
                    'yellow'
                )
            elif self._results.model_form_pct > 50:
                self._guidance_model.set_guidance(
                    f"Model form uncertainty u_model = {self._results.u_model:.4f} [{unit}] "
                    f"accounts for {self._results.model_form_pct:.1f}% of "
                    f"observed variance. Significant unaccounted physics or "
                    f"modeling errors may be present. [V&V 20 \u00a76.3]",
                    'red'
                )
            else:
                self._guidance_model.set_guidance(
                    f"Model form uncertainty u_model = {self._results.u_model:.4f} [{unit}] "
                    f"accounts for {self._results.model_form_pct:.1f}% of "
                    f"observed variance. Known sources explain most of the "
                    f"observed scatter. [V&V 20 \u00a76.3]",
                    'green'
                )
        else:
            self._guidance_model.set_guidance(
                "Insufficient data for model form assessment.", 'yellow'
            )

        # -- Validation assessment --
        n_data = getattr(self._results, 'n_data', 0) if self._results else 0
        mv_enabled = bool(getattr(self._results, 'multivariate_enabled', False)) if self._results else False
        mv_computed = bool(getattr(self._results, 'multivariate_computed', False)) if self._results else False
        mv_p = float(getattr(self._results, 'multivariate_pvalue', 1.0)) if self._results else 1.0
        mv_warn = mv_enabled and mv_computed and mv_p < 0.05

        if n_data == 0:
            self._guidance_bias.set_guidance(
                "\u26a0 No comparison data loaded. The validation assessment "
                "requires experimental data (E = S \u2212 D) to compare against "
                "the expanded uncertainty. Import comparison data on Tab 1 "
                "before interpreting these results.",
                'yellow'
            )
        elif bias_explained is None:
            # bias_explained is None when n_data == 0 — already handled above
            # but guard explicitly in case this path is reached differently
            self._guidance_bias.set_guidance(
                "\u26a0 Validation verdict cannot be determined — no "
                "comparison data available.",
                'yellow'
            )
        elif bias_explained:
            if mv_warn:
                self._guidance_bias.set_guidance(
                    f"|\u0112| = {abs(E_mean):.4f} \u2264 U_val = {U_val:.4f} [{unit}] "
                    f"\u2014 Scalar V&V 20 validation passes, but the covariance-aware "
                    f"multivariate supplement reports p = {mv_p:.4f} (< 0.05). "
                    f"This suggests residual structured bias across locations/conditions. "
                    f"Review model-form assumptions before closing validation.",
                    'yellow'
                )
            else:
                self._guidance_bias.set_guidance(
                    f"|\u0112| = {abs(E_mean):.4f} \u2264 U_val = {U_val:.4f} [{unit}] "
                    f"\u2014 The mean comparison error is within the expanded "
                    f"uncertainty. The model is validated at the specified "
                    f"coverage level for this comparison metric. "
                    f"[ASME V&V 20 \u00a76, Eq. (1)]",
                    'green'
                )
        else:
            self._guidance_bias.set_guidance(
                f"RESULT: NOT VALIDATED \u2014 the comparison error exceeds "
                f"the known uncertainty.\n"
                f"|Ē| = {abs(E_mean):.4f} > U_val = {U_val:.4f} [{unit}]\n\n"
                f"This means either:\n"
                f"1. The CFD model has a deficiency that produces bias "
                f"beyond what uncertainties explain\n"
                f"2. One or more uncertainty sources were underestimated\n"
                f"3. Both\n\n"
                f"RECOMMENDED NEXT STEPS:\n"
                f"\u2022 Review each uncertainty source \u2014 is anything "
                f"missing or underestimated?\n"
                f"\u2022 Check if the dominant source can be reduced "
                f"(epistemic sources are reducible)\n"
                f"\u2022 Consider running additional validation points "
                f"to confirm the trend\n"
                f"\u2022 If bias is consistent, consider a bias correction "
                f"approach\n"
                f"\u2022 Document the gap and flag for model improvement\n"
                f"[ASME V&V 20 \u00a76, Eq. (1)]",
                'red'
            )

    # -----------------------------------------------------------------
    # PLOTS
    # -----------------------------------------------------------------
    def _update_plots(self, contributions: List[dict], u_val: float,
                      u_num: float, u_input: float, u_D: float,
                      E_mean: float, s_E: float, k_factor: float,
                      p5: float, p95: float, unit: str,
                      settings: 'AnalysisSettings'):
        """Update all three subplots."""
        self._fig.clear()

        # ---- Subplot 1: Pie chart of variance contributions ----
        ax1 = self._fig.add_subplot(3, 1, 1)
        self._plot_variance_pie(ax1, contributions, u_val, unit)

        # ---- Subplot 2: Bar chart of category magnitudes ----
        ax2 = self._fig.add_subplot(3, 1, 2)
        self._plot_category_bars(ax2, u_num, u_input, u_D, u_val, unit)

        # ---- Subplot 3: Normal PDF with bounds ----
        ax3 = self._fig.add_subplot(3, 1, 3)
        self._plot_normal_pdf(ax3, E_mean, u_val, s_E, k_factor,
                              p5, p95, unit, settings)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._fig.tight_layout(pad=2.0)
            except Exception:
                pass
        self._canvas.draw()

    def _plot_variance_pie(self, ax, contributions: List[dict],
                           u_val: float, unit: str):
        """Plot 1: Pie chart of individual source variance contributions."""
        u_val_sq = u_val ** 2 if u_val > 0 else 1.0

        # Collect source labels and values
        labels = []
        sizes = []
        for c in contributions:
            pct = (c['sigma_sq'] / u_val_sq * 100.0) if u_val_sq > 0 else 0.0
            labels.append(f"{c['name']}\n({pct:.1f}%)")
            sizes.append(c['sigma_sq'])

        if not sizes or sum(sizes) == 0:
            ax.text(0.5, 0.5, "No uncertainty sources defined",
                    ha='center', va='center', color=DARK_COLORS['fg_dim'],
                    transform=ax.transAxes)
            ax.set_title("Variance Contributions", color=DARK_COLORS['fg'])
            return

        # Color palette
        cat_color_map = {
            "u_num": "#89b4fa",
            "u_input": "#a6e3a1",
            "u_D": "#fab387",
        }
        colors = []
        for c in contributions:
            base = cat_color_map.get(c['category_key'], "#cdd6f4")
            colors.append(base)

        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors, autopct='',
            startangle=140, pctdistance=0.85,
            textprops={'color': DARK_COLORS['fg'], 'fontsize': 6.5}
        )
        for t in texts:
            t.set_fontsize(6.5)

        ax.set_title("Variance Contributions (\u03c3\u00b2\u1d62 / u_val\u00b2)",
                      color=DARK_COLORS['fg'], fontsize=9, fontweight='bold')

    def _plot_category_bars(self, ax, u_num: float, u_input: float,
                            u_D: float, u_val: float, unit: str):
        """Plot 2: Bar chart comparing u_num, u_input, u_D, and u_val."""
        categories = ["u_num", "u_input", "u_D", "u_val"]
        values = [u_num, u_input, u_D, u_val]
        bar_colors = ["#89b4fa", "#a6e3a1", "#fab387", DARK_COLORS['accent']]

        bars = ax.bar(categories, values, color=bar_colors, edgecolor=DARK_COLORS['border'],
                      linewidth=0.8, width=0.6)

        # Add value labels on bars
        for bar, val in zip(bars, values):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + max(values) * 0.02,
                        f"{val:.4g}",
                        ha='center', va='bottom',
                        color=DARK_COLORS['fg'], fontsize=7)

        ax.set_ylabel(f"Standard Uncertainty [{unit}]",
                       color=DARK_COLORS['fg'], fontsize=8)
        ax.set_title("Category Uncertainty Magnitudes",
                      color=DARK_COLORS['fg'], fontsize=9, fontweight='bold')
        ax.tick_params(axis='x', colors=DARK_COLORS['fg_dim'])
        ax.tick_params(axis='y', colors=DARK_COLORS['fg_dim'])
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=0.3)

    def _plot_normal_pdf(self, ax, E_mean: float, u_val: float,
                         s_E: float, k_factor: float,
                         p5: float, p95: float, unit: str,
                         settings: 'AnalysisSettings'):
        """Plot 3: Normal PDF with mean, expanded bounds, and empirical percentiles."""
        # Use the larger of u_val and s_E for plot range
        sigma_plot = max(u_val, s_E, 0.001)
        x_min = E_mean - 4.5 * sigma_plot
        x_max = E_mean + 4.5 * sigma_plot
        x = np.linspace(x_min, x_max, 500)

        # PDF based on u_val
        if u_val > 0:
            pdf_uval = norm.pdf(x, loc=E_mean, scale=u_val)
            ax.plot(x, pdf_uval, color="#89b4fa", linewidth=1.5,
                    label=f"N(\u0112, u_val) [\u03c3={u_val:.3g}]")
            ax.fill_between(x, pdf_uval, alpha=0.15, color="#89b4fa")

        # PDF based on s_E
        if s_E > 0 and abs(s_E - u_val) > 1e-10:
            pdf_sE = norm.pdf(x, loc=E_mean, scale=s_E)
            ax.plot(x, pdf_sE, color="#a6e3a1", linewidth=1.5,
                    linestyle='--',
                    label=f"N(\u0112, s_E) [\u03c3={s_E:.3g}]")

        # Mean line
        ax.axvline(E_mean, color=DARK_COLORS['fg'], linewidth=1.0,
                    linestyle='-', alpha=0.7, label=f"\u0112 = {E_mean:+.3g}")

        # u_val bounds (gated by bound_type setting)
        bound_type = getattr(settings, 'bound_type', 'Both (for comparison)')
        bt_lower = bound_type.lower()
        if 'both' in bt_lower:
            show_uval = True
            show_sE = True
        else:
            show_uval = 'u_val' in bt_lower
            show_sE = 's_e' in bt_lower

        if show_uval:
            lb_uval = E_mean - k_factor * u_val
            ub_uval = E_mean + k_factor * u_val
            ax.axvline(lb_uval, color="#89b4fa", linewidth=1.2, linestyle=':',
                        alpha=0.9, label=f"\u0112 \u00b1 k\u00b7u_val = [{lb_uval:+.3g}, {ub_uval:+.3g}]")
            ax.axvline(ub_uval, color="#89b4fa", linewidth=1.2, linestyle=':',
                        alpha=0.9)

        # s_E bounds (gated by bound_type setting)
        if show_sE and s_E > 0:
            lb_sE = E_mean - k_factor * s_E
            ub_sE = E_mean + k_factor * s_E
            ax.axvline(lb_sE, color="#a6e3a1", linewidth=1.2, linestyle='--',
                        alpha=0.7,
                        label=f"\u0112 \u00b1 k\u00b7s_E = [{lb_sE:+.3g}, {ub_sE:+.3g}]")
            ax.axvline(ub_sE, color="#a6e3a1", linewidth=1.2, linestyle='--',
                        alpha=0.7)

        # Empirical percentiles
        if p5 != 0.0 or p95 != 0.0:
            ax.axvline(p5, color=DARK_COLORS['orange'], linewidth=1.0,
                        linestyle='-.', alpha=0.8,
                        label=f"P5 = {p5:+.3g}")
            ax.axvline(p95, color=DARK_COLORS['red'], linewidth=1.0,
                        linestyle='-.', alpha=0.8,
                        label=f"P95 = {p95:+.3g}")

        ax.set_xlabel(f"Comparison Error [{unit}]",
                       color=DARK_COLORS['fg'], fontsize=8)
        ax.set_ylabel("Probability Density",
                       color=DARK_COLORS['fg'], fontsize=8)
        ax.set_title("Normal PDF with Expanded Bounds",
                      color=DARK_COLORS['fg'], fontsize=9, fontweight='bold')
        ax.legend(fontsize=6.5, loc='upper right',
                  facecolor=DARK_COLORS['bg_widget'],
                  edgecolor=DARK_COLORS['border'],
                  labelcolor=DARK_COLORS['fg'])
        ax.tick_params(axis='x', colors=DARK_COLORS['fg_dim'])
        ax.tick_params(axis='y', colors=DARK_COLORS['fg_dim'])
        ax.set_axisbelow(True)
        ax.grid(alpha=0.3)


# =============================================================================
# SECTION 11: TAB 5 — MONTE CARLO RESULTS TAB
# =============================================================================

class _MCWorkerThread(QThread):
    """
    Background worker that executes the Monte Carlo uncertainty propagation.

    For each trial the worker draws independent samples from every enabled
    uncertainty source (using ``generate_mc_samples``) and sums them to form
    the combined-error sample.  The comparison-error mean (Ē) is added as
    a constant offset so the MC distribution is centred on the observed bias,
    consistent with the V&V 20 validation assessment (|Ē| vs k·u_val).

    Percentile bounds are computed at the coverage probability specified in
    AnalysisSettings (not hard-coded to P5/P95).  Optional bootstrap CIs
    quantify sampling uncertainty on those percentiles.

    Uses a thread-local ``numpy.random.Generator`` for reproducibility and
    thread safety (no global ``np.random.seed``).

    Signals
    -------
    progress_updated : Signal(int)
        Emits the current percentage of completion (0-100).
    finished_result : Signal(object)
        Emits the completed :class:`MCResults` when the run finishes.
    error_occurred : Signal(str)
        Emits a descriptive error string if the run fails.
    """

    progress_updated = Signal(int)
    finished_result = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, sources, comp_data, settings, parent=None):
        """
        Parameters
        ----------
        sources : list[UncertaintySource]
            Enabled uncertainty sources to propagate.
        comp_data : ComparisonData
            The comparison-error dataset.
        settings : AnalysisSettings
            Analysis configuration (n_trials, seed, bootstrap flag, etc.).
        """
        super().__init__(parent)
        self._sources = sources
        self._comp_data = comp_data
        self._settings = settings
        self._abort_event = threading.Event()

    # -- public helpers --------------------------------------------------
    def abort(self):
        """Request early termination (checked between chunks)."""
        self._abort_event.set()

    # -- main execution --------------------------------------------------
    def run(self):  # noqa: D401 – Qt override
        try:
            self._execute()
        except Exception as exc:
            self.error_occurred.emit(f"{type(exc).__name__}: {exc}")

    def _execute(self):
        settings = self._settings
        n_trials = settings.mc_n_trials
        # Thread-safe RNG — isolated from global numpy state
        if settings.mc_seed is not None:
            rng = np.random.default_rng(int(settings.mc_seed))
        else:
            rng = np.random.default_rng()

        # Select sampling function based on user setting
        use_lhs = getattr(settings, 'mc_sampling_method', '') == "Latin Hypercube (LHS)"
        _sample_fn = generate_lhs_samples if use_lhs else generate_mc_samples

        # ----------------------------------------------------------
        # 1. Draw samples from each uncertainty source and sum them.
        #    Correlation-aware: sources sharing a non-empty correlation_group
        #    within the same V&V category and using Normal distributions are
        #    sampled jointly via multivariate normal + Cholesky decomposition.
        # ----------------------------------------------------------
        combined = np.zeros(n_trials, dtype=np.float64)
        n_sources = len(self._sources)

        # Identify correlation groups (category-scoped to match RSS assumptions)
        corr_groups: dict[tuple[str, str], list[int]] = {}
        group_to_categories: dict[str, set[str]] = {}
        for idx, src in enumerate(self._sources):
            grp = src.correlation_group.strip()
            if grp:
                cat_key = src.get_category_key()
                corr_groups.setdefault((cat_key, grp), []).append(idx)
                group_to_categories.setdefault(grp, set()).add(cat_key)

        # Track which source indices are handled via multivariate sampling
        handled_mv: set[int] = set()

        # User-facing Monte Carlo notes (shown in Tab 5 / report)
        mc_notes: List[str] = []
        _mc_note_set = set()

        def _add_mc_note(message: str):
            if message and message not in _mc_note_set:
                _mc_note_set.add(message)
                mc_notes.append(message)

        cat_labels = {
            "u_num": "Numerical (u_num)",
            "u_input": "Input/BC (u_input)",
            "u_D": "Experimental (u_D)",
        }
        for grp_name, cat_set in sorted(group_to_categories.items()):
            if len(cat_set) > 1:
                cat_list = ", ".join(
                    cat_labels.get(c, c) for c in sorted(cat_set)
                )
                _add_mc_note(
                    f"Correlation group '{grp_name}' appears in multiple "
                    f"categories ({cat_list}); cross-category correlation was "
                    "not applied to stay consistent with RSS assumptions."
                )
                audit_log.log_warning(
                    "MC_CORR_CROSS_CATEGORY",
                    f"Group '{grp_name}' spans categories ({cat_list}); "
                    "cross-category correlation terms were not applied."
                )

        # Process correlated groups first (multivariate normal)
        for (cat_key, grp_name), indices in corr_groups.items():
            if len(indices) < 2:
                continue  # single-source group → treat as independent
            # Sort alphabetically by source name for deterministic reference
            indices.sort(key=lambda i: self._sources[i].name.lower())
            grp_sources = [self._sources[i] for i in indices]
            cat_label = cat_labels.get(cat_key, cat_key)
            # Only use multivariate sampling for all-Normal groups
            all_normal = all(s.distribution == "Normal" for s in grp_sources)
            if not all_normal:
                _add_mc_note(
                    f"Correlation group '{grp_name}' ({cat_label}) includes "
                    "non-Normal sources; Monte Carlo treated that group as independent."
                )
                audit_log.log_warning(
                    "MC_CORR_FALLBACK_NONNORMAL",
                    f"Group '{grp_name}' ({cat_label}) contains non-Normal "
                    "distributions; "
                    "falling back to independent sampling."
                )
                continue  # mixed distributions → independent sampling

            sigmas = np.array([s.get_standard_uncertainty() for s in grp_sources])
            means = np.array([
                0.0 if s.is_centered_on_zero else s.mean_value
                for s in grp_sources
            ])
            rhos = [s.correlation_coefficient for s in grp_sources]
            # Reference source (first alphabetically) has ρ = 1.0 implicitly
            rhos[0] = 1.0

            # Warn about rho=0.0 in a group (effectively independent)
            for mc_idx, (mc_src, mc_rho) in enumerate(
                    zip(grp_sources[1:], rhos[1:])):
                if mc_rho == 0.0:
                    audit_log.log_assumption(
                        f"MC: Source '{mc_src.name}' is in correlation group "
                        f"'{grp_name}' ({cat_label}) but has ρ = 0.0 — "
                        "effectively independent."
                    )

            # Build correlation matrix for this group
            n_g = len(indices)
            C = np.eye(n_g)
            for a in range(n_g):
                for b in range(n_g):
                    if a == b:
                        continue
                    rho_a = rhos[a]
                    rho_b = rhos[b]
                    C[a, b] = rho_a * rho_b

            # Validate positive semi-definiteness
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < -1e-10):
                # Fall back to independent for this group
                _add_mc_note(
                    f"Correlation group '{grp_name}' ({cat_label}) produced a non-PSD "
                    "correlation matrix; Monte Carlo used independent sampling."
                )
                audit_log.log_warning(
                    "MC_CORR_FALLBACK_PSD",
                    f"Group '{grp_name}' ({cat_label}) correlation matrix is not positive "
                    "semi-definite; falling back to independent sampling."
                )
                continue

            # Build covariance matrix: Σ = diag(σ) · C · diag(σ)
            cov = np.outer(sigmas, sigmas) * C

            # Cholesky decomposition for sampling
            try:
                # Ensure PSD by clamping small negative eigenvalues
                cov = (cov + cov.T) / 2.0
                L = np.linalg.cholesky(cov)
            except np.linalg.LinAlgError:
                _add_mc_note(
                    f"Correlation group '{grp_name}' ({cat_label}) covariance factorization "
                    "failed; Monte Carlo used independent sampling."
                )
                audit_log.log_warning(
                    "MC_CORR_FALLBACK_CHOLESKY",
                    f"Group '{grp_name}' ({cat_label}) covariance Cholesky failed; "
                    "falling back to independent sampling."
                )
                continue  # not PSD → fall back to independent

            # Generate multivariate normal samples
            z = rng.standard_normal((n_trials, n_g))
            mv_samples = z @ L.T + means[np.newaxis, :]

            # Sum all correlated source contributions
            for col_idx in range(n_g):
                combined += mv_samples[:, col_idx]
                handled_mv.add(indices[col_idx])

        # Process remaining independent sources
        processed = 0
        for idx, src in enumerate(self._sources):
            if self._abort_event.is_set():
                return
            if idx in handled_mv:
                processed += 1
                pct = int(50 * processed / max(n_sources, 1))
                self.progress_updated.emit(pct)
                continue
            sigma = src.get_standard_uncertainty()
            mean = 0.0 if src.is_centered_on_zero else src.mean_value
            raw_data = (
                np.array(src.tabular_data, dtype=float)
                if src.input_type == "Tabular Data" and len(src.tabular_data) > 1
                else None
            )
            # Asymmetric bifurcated Gaussian support
            kw_asym = {}
            if src.asymmetric:
                kw_asym['sigma_upper'] = src.get_sigma_upper()
                kw_asym['sigma_lower'] = src.get_sigma_lower()
            samples = _sample_fn(
                src.distribution, sigma, mean, n_trials, raw_data=raw_data,
                rng=rng, **kw_asym,
            )
            combined += samples
            processed += 1
            pct = int(50 * processed / max(n_sources, 1))
            self.progress_updated.emit(pct)

        if use_lhs and handled_mv:
            _add_mc_note(
                "LHS stratification was applied to independent sources. "
                "Correlated Normal groups used multivariate Normal sampling."
            )

        # ----------------------------------------------------------
        # 2. Offset by comparison-error mean (E_mean).
        # ----------------------------------------------------------
        # NOTE: The MC propagation combines uncertainty *sources* only
        # (matching the RSS u_val computation).  The comparison error
        # mean E_mean is added as a constant offset so that the MC
        # distribution is centered on the observed bias, consistent
        # with V&V 20 validation assessment (|E| vs k*u_val).
        # We do NOT add the comparison-error scatter (s_E) as
        # additional variance — that would double-count it.
        flat = self._comp_data.flat_data()
        if flat.size > 0:
            E_mean = float(np.mean(flat))
            combined += E_mean

        self.progress_updated.emit(60)

        # ----------------------------------------------------------
        # 3. Compute summary statistics.
        # ----------------------------------------------------------
        if self._abort_event.is_set():
            return

        mc = MCResults()
        mc.n_trials = n_trials
        mc.sampling_method = getattr(settings, 'mc_sampling_method',
                                     "Monte Carlo (Random)")
        mc.combined_mean = float(np.mean(combined))
        mc.combined_std = float(np.std(combined, ddof=1))

        # Compute coverage-matched percentiles from settings
        coverage = settings.coverage
        one_sided = settings.one_sided
        mc._coverage = coverage
        mc._one_sided = one_sided

        if one_sided:
            # One-sided: lower = 1-coverage, upper = coverage
            lower_pct = (1.0 - coverage) * 100.0
            upper_pct = coverage * 100.0
        else:
            # Two-sided: symmetric central interval
            alpha = (1.0 - coverage) / 2.0
            lower_pct = alpha * 100.0
            upper_pct = (1.0 - alpha) * 100.0

        mc._lower_pct = lower_pct
        mc._upper_pct = upper_pct
        mc.pct_5 = float(np.percentile(combined, lower_pct))
        mc.pct_95 = float(np.percentile(combined, upper_pct))
        mc.lower_bound = mc.pct_5
        mc.upper_bound = mc.pct_95
        mc.samples = combined
        mc.notes = mc_notes

        self.progress_updated.emit(75)

        # ----------------------------------------------------------
        # 4. Optional bootstrap confidence intervals on percentiles.
        #    1000 resamples gives stable 95% CIs on tail percentiles
        #    (JCGM 101:2008 §7.9 recommends M >= 10^4 for adaptive;
        #    1000 is a practical compromise for interactive use).
        # ----------------------------------------------------------
        if settings.mc_bootstrap:
            if self._abort_event.is_set():
                return
            n_bootstrap = 1000
            p5_boots = np.empty(n_bootstrap)
            p95_boots = np.empty(n_bootstrap)
            for b in range(n_bootstrap):
                if self._abort_event.is_set():
                    return
                boot = rng.choice(combined, size=n_trials, replace=True)
                p5_boots[b] = np.percentile(boot, lower_pct)
                p95_boots[b] = np.percentile(boot, upper_pct)
                if b % 50 == 0:  # update progress every 50 resamples
                    pct = 75 + int(25 * (b + 1) / n_bootstrap)
                    self.progress_updated.emit(min(pct, 99))

            mc.bootstrap_ci_low = float(np.percentile(p5_boots, 2.5))
            mc.bootstrap_ci_high = float(np.percentile(p95_boots, 97.5))
            # Store additional bootstrap arrays as extra attributes so the
            # tab can display per-percentile confidence intervals.
            mc._p5_boots = p5_boots
            mc._p95_boots = p95_boots

        mc.computed = True
        self.progress_updated.emit(100)
        self.finished_result.emit(mc)


# =====================================================================
# Double-Loop Monte Carlo Worker Thread
# (Oberkampf & Roy 2010 — epistemic/aleatory separation → p-box)
# =====================================================================

class _DoubleLoopMCWorkerThread(QThread):
    """Background worker for double-loop (nested) Monte Carlo propagation.

    Outer loop samples epistemic parameters; inner loop propagates aleatory
    uncertainty.  The result is a family of CDFs whose envelope forms a
    probability box (p-box).

    Two sub-modes:
      * **corners** — enumerate 2^n worst-case corners of the epistemic space
      * **full** — stochastically sample the epistemic space (N_outer draws)

    References:
        Oberkampf & Roy (2010), Chapter 5
        Ferson et al. (2003), Probability boxes
        SAND2007-0939
    """

    progress_updated = Signal(int)
    finished_result = Signal(object)
    error_occurred = Signal(str)

    def __init__(self, sources, comp_data, settings, parent=None):
        super().__init__(parent)
        self._sources = sources
        self._comp_data = comp_data
        self._settings = settings
        self._abort_event = threading.Event()

    def abort(self):
        self._abort_event.set()

    def run(self):
        try:
            self._execute()
        except Exception as exc:
            self.error_occurred.emit(f"Double-loop MC error: {exc}")
            traceback.print_exc()

    # -----------------------------------------------------------------
    def _execute(self):
        settings = self._settings
        rng = np.random.default_rng(
            settings.mc_seed if settings.mc_seed else None
        )

        # ---- Step 1: classify sources into epistemic vs aleatory ----
        epistemic_srcs = []
        aleatory_srcs = []
        for src in self._sources:
            if src.get_standard_uncertainty() <= 0 and src.representation != "interval":
                continue  # skip zero-sigma non-interval sources
            cls = src.uncertainty_class
            if cls == "mixed":
                if settings.mc_mixed_treatment == "Treat as aleatory":
                    cls = "aleatoric"
                else:
                    cls = "epistemic"
            if cls == "epistemic":
                epistemic_srcs.append(src)
            else:
                aleatory_srcs.append(src)

        n_epi = len(epistemic_srcs)
        n_ale = len(aleatory_srcs)
        notes: List[str] = []

        # ---- Degenerate case: no epistemic sources ----
        if n_epi == 0:
            notes.append(
                "No epistemic sources found after classification. "
                "Double-loop degenerates to a single inner-loop CDF (no p-box). "
                "Consider classifying sources or using Single-Loop mode."
            )

        # ---- Step 2: determine outer-loop realizations ----
        is_corners = settings.mc_mode == "Double-Loop (Corners)"
        n_inner = settings.mc_n_inner

        if is_corners:
            if n_epi > 15:
                self.error_occurred.emit(
                    f"Corners mode requires 2^N_epistemic evaluations. "
                    f"With {n_epi} epistemic sources that is "
                    f"{2**n_epi:,} corners — too many. "
                    f"Use 'Double-Loop (Full)' mode instead, or reduce "
                    f"the number of epistemic sources to ≤15."
                )
                return

            # Build corner values for each epistemic source
            corner_pairs = []  # list of (low, high) per epistemic source
            for src in epistemic_srcs:
                if src.representation == "interval":
                    lo = src.interval_lower
                    hi = src.interval_upper
                    if lo > hi:
                        lo, hi = hi, lo
                        notes.append(
                            f"Interval bounds for '{src.name}' were inverted; "
                            f"auto-swapped to [{lo}, {hi}]."
                        )
                    if lo == hi:
                        # Degenerate interval — treat as single point
                        corner_pairs.append((lo, lo))
                        notes.append(
                            f"Interval for '{src.name}' has zero width "
                            f"(lower = upper = {lo}). Using as fixed value."
                        )
                    else:
                        corner_pairs.append((lo, hi))
                else:
                    # Distribution-represented epistemic: use ±2σ as corners
                    sigma = src.get_standard_uncertainty()
                    mean = src.mean_value if not src.is_centered_on_zero else 0.0
                    corner_pairs.append((mean - 2.0 * sigma, mean + 2.0 * sigma))

            if n_epi > 0:
                corner_combos = list(itertools.product(*corner_pairs))
            else:
                corner_combos = [()]  # single "empty corner"
            n_outer = len(corner_combos)
        else:
            # Full stochastic mode
            n_outer = settings.mc_n_outer
            corner_combos = None  # not used

        # Choose sampling function
        if settings.mc_sampling_method == "Latin Hypercube (LHS)":
            _sample_fn = generate_lhs_samples
        else:
            _sample_fn = generate_mc_samples

        # ---- Step 3: comparison-error offset ----
        E_mean = float(np.nanmean(self._comp_data.data)) if (
            self._comp_data.data is not None and len(self._comp_data.data) > 0
        ) else 0.0

        # ---- Step 4: coverage-matched percentile levels ----
        coverage = settings.coverage
        if settings.one_sided:
            lower_pct = (1.0 - coverage) * 100.0
            upper_pct = coverage * 100.0
        else:
            alpha = 1.0 - coverage
            lower_pct = (alpha / 2.0) * 100.0
            upper_pct = (1.0 - alpha / 2.0) * 100.0

        # ---- Step 5: outer loop ----
        real_lower_bounds = np.zeros(n_outer)
        real_upper_bounds = np.zeros(n_outer)
        real_means = np.zeros(n_outer)
        # Store sorted inner-loop samples for p-box construction
        # (keep only quantile summaries to limit memory)
        n_pbox_pts = 500
        pbox_quantile_levels = np.linspace(0.0, 1.0, n_pbox_pts)
        # Each row = quantiles for one realization
        quantile_matrix = np.zeros((n_outer, n_pbox_pts))

        for i_outer in range(n_outer):
            if self._abort_event.is_set():
                return

            # --- Determine epistemic offsets for this realization ---
            epi_offsets = np.zeros(n_epi)
            if is_corners and n_epi > 0:
                # Use the corner combination
                combo = corner_combos[i_outer]
                epi_offsets = np.array(combo, dtype=float)
            elif n_epi > 0:
                # Stochastic: draw one sample per epistemic source
                for j, src in enumerate(epistemic_srcs):
                    if src.representation == "interval":
                        lo = src.interval_lower
                        hi = src.interval_upper
                        if lo > hi:
                            lo, hi = hi, lo
                        epi_offsets[j] = rng.uniform(lo, hi)
                    else:
                        sigma = src.get_standard_uncertainty()
                        mean = src.mean_value if not src.is_centered_on_zero else 0.0
                        sample = _sample_fn(
                            src.distribution, sigma, mean, 1,
                            rng=rng,
                        )
                        epi_offsets[j] = float(sample[0])

            total_epi_offset = float(np.sum(epi_offsets))

            # --- Inner loop: sample aleatory sources ---
            combined = np.zeros(n_inner)
            for src in aleatory_srcs:
                sigma = src.get_standard_uncertainty()
                if sigma <= 0:
                    continue
                mean = src.mean_value if not src.is_centered_on_zero else 0.0
                su = src.get_sigma_upper() if src.asymmetric else None
                sl = src.get_sigma_lower() if src.asymmetric else None
                samples = _sample_fn(
                    src.distribution, sigma, mean, n_inner,
                    rng=rng,
                    sigma_upper=su,
                    sigma_lower=sl,
                )
                combined += samples

            # Add fixed epistemic offset + comparison-error mean
            combined += total_epi_offset + E_mean

            # --- Inner-loop statistics ---
            real_means[i_outer] = float(np.mean(combined))
            lb = float(np.percentile(combined, lower_pct))
            ub = float(np.percentile(combined, upper_pct))
            real_lower_bounds[i_outer] = lb
            real_upper_bounds[i_outer] = ub

            # Store quantiles for p-box envelope construction
            quantile_matrix[i_outer, :] = np.quantile(
                combined, pbox_quantile_levels
            )

            # Progress
            pct = int((i_outer + 1) / n_outer * 95)  # reserve 5% for postproc
            self.progress_updated.emit(pct)

        # ---- Step 6: p-box construction ----
        # At each quantile level, the p-box envelope is the min/max across
        # all realizations.  We report the envelope as lower/upper CDF
        # bounds on a shared x-grid.
        pbox_x_lower = np.min(quantile_matrix, axis=0)  # most-left CDF
        pbox_x_upper = np.max(quantile_matrix, axis=0)  # most-right CDF

        # ---- Step 7: validation fraction ----
        brackets_zero = (real_lower_bounds <= 0.0) & (real_upper_bounds >= 0.0)
        val_frac = float(np.mean(brackets_zero))

        # ---- Step 8: assemble results ----
        dl = DoubleLoopMCResults(
            mode="corners" if is_corners else "full",
            n_outer=n_outer,
            n_inner=n_inner,
            n_epistemic_sources=n_epi,
            n_aleatory_sources=n_ale,
            pbox_x=np.concatenate([pbox_x_lower, pbox_x_upper[::-1]]),
            pbox_cdf_lower=pbox_x_lower,
            pbox_cdf_upper=pbox_x_upper,
            lower_bound_min=float(np.min(real_lower_bounds)) if n_outer > 0 else 0.0,
            lower_bound_max=float(np.max(real_lower_bounds)) if n_outer > 0 else 0.0,
            upper_bound_min=float(np.min(real_upper_bounds)) if n_outer > 0 else 0.0,
            upper_bound_max=float(np.max(real_upper_bounds)) if n_outer > 0 else 0.0,
            mean_of_means=float(np.mean(real_means)) if n_outer > 0 else 0.0,
            std_of_means=float(np.std(real_means, ddof=1)) if n_outer > 1 else 0.0,
            validation_fraction=val_frac,
            _coverage=coverage,
            _one_sided=settings.one_sided,
            _lower_pct=lower_pct,
            _upper_pct=upper_pct,
            realization_lower_bounds=real_lower_bounds,
            realization_upper_bounds=real_upper_bounds,
            realization_means=real_means,
            epistemic_source_names=[s.name for s in epistemic_srcs],
            notes=notes,
            computed=True,
        )

        self.progress_updated.emit(100)
        self.finished_result.emit(dl)


class MonteCarloResultsTab(QWidget):
    """
    Tab 5: Monte Carlo Uncertainty Propagation Results.

    Runs a full Monte Carlo propagation of all enabled uncertainty sources,
    sums them trial-by-trial, and presents:
      - Numerical summary (mean, std, percentiles, prediction bounds)
      - Bootstrap confidence intervals on the percentile estimates
      - Interpretation guidance comparing MC vs RSS bounds
      - Histogram with RSS normal overlay
      - CDF with coverage percentile markers
      - Convergence plot (running percentile vs trial count)

    References:
        - JCGM 101:2008 (GUM Supplement 1)
        - ASME V&V 20-2009 Section 5.3.2 (Monte Carlo alternative)
    """

    mc_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mc_results = MCResults()
        self._rss_results: Optional[RSSResults] = None
        self._worker: Optional[_MCWorkerThread] = None
        self._setup_ui()

    # =================================================================
    # PUBLIC API
    # =================================================================
    def run_mc(self, sources, comp_data, settings, rss_results=None):
        """
        Start (or restart) the Monte Carlo analysis.

        Parameters
        ----------
        sources : list[UncertaintySource]
            Only enabled sources are propagated.
        comp_data : ComparisonData
            Raw comparison-error data.
        settings : AnalysisSettings
            MC configuration (n_trials, seed, bootstrap, sampling method).
        rss_results : RSSResults, optional
            If provided, the interpretation panel will compare MC vs RSS bounds.
        """
        if self._worker is not None and self._worker.isRunning():
            self._worker.abort()
            self._worker.wait(2000)

        self._rss_results = rss_results
        self._display_unit = settings.global_unit  # Store unit for plot/text labels
        self._status_label.setText("Running...")
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._btn_run.setEnabled(False)
        self._results_text.clear()

        # Deep-copy inputs so the worker thread is isolated from GUI edits
        enabled = [copy.deepcopy(s) for s in sources if s.enabled]
        self._worker = _MCWorkerThread(
            enabled, copy.deepcopy(comp_data), copy.deepcopy(settings),
            parent=self,
        )
        self._worker.progress_updated.connect(self._on_progress)
        self._worker.finished_result.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    def get_results(self) -> MCResults:
        """Return the most recent MC results (may be empty if not yet run)."""
        return self._mc_results

    def clear_results(self):
        """Reset MC results and clear all displays."""
        self._mc_results = MCResults()
        self._rss_results = None
        self._results_text.clear()
        self._status_label.setText("Not yet run")
        self._progress_bar.setVisible(False)
        self._fig.clear()
        self._canvas.draw_idle()
        if hasattr(self, '_guidance_mc_convergence'):
            self._guidance_mc_convergence.set_guidance("", "green")
        if hasattr(self, '_guidance_mc_vs_rss'):
            self._guidance_mc_vs_rss.set_guidance("", "green")

    # =================================================================
    # UI CONSTRUCTION
    # =================================================================
    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self._splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(self._splitter)

        # -- Left panel (scrollable controls + text results) ----------
        left_container = QWidget()
        self._left_layout = QVBoxLayout(left_container)
        self._left_layout.setContentsMargins(8, 8, 4, 8)
        self._left_layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(left_container)
        scroll.setMinimumWidth(560)
        self._splitter.addWidget(scroll)

        self._build_controls()
        self._build_results_display()
        self._build_guidance_panels()
        self._left_layout.addStretch()

        # -- Right panel (plots) -------------------------------------
        right_container = QWidget()
        self._right_layout = QVBoxLayout(right_container)
        self._right_layout.setContentsMargins(4, 8, 8, 8)
        self._splitter.addWidget(right_container)

        self._build_plot_section()
        self._splitter.setStretchFactor(0, 3)
        self._splitter.setStretchFactor(1, 2)
        self._splitter.setSizes([650, 550])

    # -- Controls ----------------------------------------------------
    def _build_controls(self):
        grp = QGroupBox("Monte Carlo Execution")
        lay = QVBoxLayout(grp)
        lay.setSpacing(8)

        # Run button – prominent accent-colored
        self._btn_run = QPushButton("Run Monte Carlo")
        self._btn_run.setMinimumHeight(36)
        self._btn_run.setStyleSheet(
            f"QPushButton {{"
            f"  background-color: {DARK_COLORS['accent']};"
            f"  color: {DARK_COLORS['bg']};"
            f"  font-weight: bold; font-size: 14px;"
            f"  border-radius: 4px; padding: 6px 16px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: {DARK_COLORS['accent_hover']};"
            f"}}"
            f"QPushButton:disabled {{"
            f"  background-color: {DARK_COLORS['bg_input']};"
            f"  color: {DARK_COLORS['fg_dim']};"
            f"}}"
        )
        self._btn_run.setToolTip(
            "Launch the Monte Carlo uncertainty propagation.\n"
            "Each trial draws independently from every source's\n"
            "distribution and sums to form the combined error.\n"
            "[JCGM 101:2008, ASME V&V 20 \u00a75.3.2]"
        )
        # NOTE: The button click is NOT auto-connected here; the main
        # application window connects it to call ``run_mc`` with the
        # correct data.  However, we also provide a local fallback
        # that emits nothing — the parent is expected to wire this.
        lay.addWidget(self._btn_run)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet(
            f"QProgressBar {{"
            f"  background-color: {DARK_COLORS['bg_input']};"
            f"  border: 1px solid {DARK_COLORS['border']};"
            f"  border-radius: 3px; text-align: center;"
            f"  color: {DARK_COLORS['fg']};"
            f"}}"
            f"QProgressBar::chunk {{"
            f"  background-color: {DARK_COLORS['accent']};"
            f"  border-radius: 3px;"
            f"}}"
        )
        lay.addWidget(self._progress_bar)

        # Status label
        self._status_label = QLabel("Ready")
        self._status_label.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; font-size: 12px;"
        )
        lay.addWidget(self._status_label)

        self._left_layout.addWidget(grp)

    # -- Results text display ----------------------------------------
    def _build_results_display(self):
        grp = QGroupBox("Monte Carlo Results")
        grp.setToolTip(
            "Numerical summary of the Monte Carlo combined-error\n"
            "distribution. Percentiles are computed directly from\n"
            "the empirical sample. [JCGM 101:2008 \u00a77]"
        )
        lay = QVBoxLayout(grp)

        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setMinimumHeight(260)
        self._results_text.setStyleSheet(
            f"QTextEdit {{"
            f"  background-color: {DARK_COLORS['bg_input']};"
            f"  color: {DARK_COLORS['fg']};"
            f"  font-family: 'Consolas', 'Courier New', monospace;"
            f"  font-size: 12px;"
            f"  border: 1px solid {DARK_COLORS['border']};"
            f"  padding: 8px;"
            f"}}"
        )
        self._results_text.setPlaceholderText(
            "Click 'Run Monte Carlo' to execute the analysis..."
        )
        lay.addWidget(self._results_text)
        self._left_layout.addWidget(grp)

    # -- Guidance panels (interpretation) ----------------------------
    def _build_guidance_panels(self):
        self._guidance_mc_convergence = GuidancePanel(
            "MC Convergence Check", parent=self,
        )
        self._left_layout.addWidget(self._guidance_mc_convergence)
        self._guidance_mc_vs_rss = GuidancePanel(
            "MC vs RSS Comparison", parent=self,
        )
        self._left_layout.addWidget(self._guidance_mc_vs_rss)

    # -- Plot section ------------------------------------------------
    def _build_plot_section(self):
        plt.rcParams.update(PLOT_STYLE)

        self._fig = Figure(figsize=(5.2, 8), dpi=100)
        self._fig.set_facecolor(PLOT_STYLE['figure.facecolor'])
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setMinimumWidth(400)
        self._canvas.setMinimumHeight(550)

        toolbar_row = make_plot_toolbar_with_copy(
            self._canvas, self._fig, self,
            method_context="Monte Carlo JCGM 101:2008")

        # Wrap canvas in a scroll area for horizontal scrolling
        plot_scroll = QScrollArea()
        plot_scroll.setWidgetResizable(True)
        plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        plot_container = QWidget()
        plot_lay = QVBoxLayout(plot_container)
        plot_lay.setContentsMargins(0, 0, 0, 0)
        plot_lay.addWidget(self._canvas)
        plot_scroll.setWidget(plot_container)

        self._right_layout.addWidget(toolbar_row)
        self._right_layout.addWidget(plot_scroll)

    # =================================================================
    # SLOTS — worker communication
    # =================================================================
    @Slot(int)
    def _on_progress(self, value: int):
        self._progress_bar.setValue(value)

    @Slot(object)
    def _on_finished(self, mc_results: MCResults):
        self._mc_results = mc_results
        self._progress_bar.setValue(100)
        self._progress_bar.setVisible(False)
        self._btn_run.setEnabled(True)
        method_abbr = ("LHS" if getattr(mc_results, 'sampling_method', '')
                       == "Latin Hypercube (LHS)" else "MC")
        self._status_label.setText(
            f"Complete — {method_abbr} ({mc_results.n_trials:,} trials)"
        )
        self._populate_results_text(mc_results)
        self._check_convergence(mc_results)
        self._update_guidance(mc_results)
        self._update_plots(mc_results)
        self.mc_finished.emit()

    @Slot(str)
    def _on_error(self, msg: str):
        self._progress_bar.setVisible(False)
        self._btn_run.setEnabled(True)
        self._status_label.setText("Error")
        self._results_text.setPlainText(f"Monte Carlo run failed:\n{msg}")

    # =================================================================
    # RESULTS FORMATTING
    # =================================================================
    def _populate_results_text(self, mc: MCResults):
        """Fill the text panel with a formatted summary."""
        unit_str = getattr(self, '_display_unit', 'unit')

        lines = []
        method_name = getattr(mc, 'sampling_method', 'Monte Carlo (Random)')
        lines.append(
            f"{method_name} Results (N = {mc.n_trials:,} trials):\n"
        )
        lo_pct = mc._lower_pct
        hi_pct = mc._upper_pct
        cov = mc._coverage
        sided = "one-sided" if mc._one_sided else "two-sided"
        cov_label = f"{cov*100:.0f}% {sided}"

        lines.append("Combined Error Distribution:")
        lines.append(f"  Mean           = {mc.combined_mean:+.4f} [{unit_str}]")
        lines.append(f"  Std Dev        = {mc.combined_std:.4f} [{unit_str}]")
        lines.append(f"  P{lo_pct:.4g}       = {mc.pct_5:+.4f} [{unit_str}]")
        lines.append(f"  P{hi_pct:.4g}      = {mc.pct_95:+.4f} [{unit_str}]")
        lines.append("")
        lines.append(f"Prediction Bounds ({cov_label}):")
        lines.append(
            f"  Lower bound (P{lo_pct:.4g})  = {mc.lower_bound:+.4f} [{unit_str}]"
        )
        lines.append(
            f"  Upper bound (P{hi_pct:.4g}) = {mc.upper_bound:+.4f} [{unit_str}]"
        )

        # Bootstrap confidence intervals on percentile estimates
        if hasattr(mc, '_p5_boots') and hasattr(mc, '_p95_boots'):
            p5_boots = mc._p5_boots
            p95_boots = mc._p95_boots
            p5_mean = float(np.mean(p5_boots))
            p5_std = float(np.std(p5_boots, ddof=1))
            p5_ci_lo = float(np.percentile(p5_boots, 2.5))
            p5_ci_hi = float(np.percentile(p5_boots, 97.5))
            p95_mean = float(np.mean(p95_boots))
            p95_std = float(np.std(p95_boots, ddof=1))
            p95_ci_lo = float(np.percentile(p95_boots, 2.5))
            p95_ci_hi = float(np.percentile(p95_boots, 97.5))

            lines.append("")
            lines.append("Bootstrap Confidence on Percentiles (1000 resamples):")
            lines.append(
                f"  P{lo_pct:.4g}:  {p5_mean:+.4f} \u00b1 {p5_std:.4f}"
                f"  (95% CI: [{p5_ci_lo:+.4f}, {p5_ci_hi:+.4f}])"
            )
            lines.append(
                f"  P{hi_pct:.4g}: {p95_mean:+.4f} \u00b1 {p95_std:.4f}"
                f"  (95% CI: [{p95_ci_lo:+.4f}, {p95_ci_hi:+.4f}])"
            )

        if getattr(mc, 'notes', None):
            lines.append("")
            lines.append("Important Notes:")
            for note in mc.notes:
                lines.append(f"  - {note}")

        self._results_text.setPlainText("\n".join(lines))

    # =================================================================
    # GUIDANCE / INTERPRETATION
    # =================================================================
    def _check_convergence(self, mc: MCResults):
        """
        Estimate the sampling uncertainty of the MC percentile estimates
        and warn if convergence is insufficient.

        Uses the bootstrap CIs when available; otherwise falls back to an
        analytical approximation for the standard error of a percentile
        (SE = sqrt(q(1-q) / (n * f(x_q)^2))).

        Ref: JCGM 101:2008 §7.9 — adaptive MC procedure.
        """
        if mc.samples.size == 0:
            self._guidance_mc_convergence.set_guidance(
                "No MC samples available.", "green")
            return

        n = mc.n_trials
        samples = mc.samples

        # Coverage-matched quantile fractions
        lo_q = getattr(mc, '_lower_pct', 5.0) / 100.0   # e.g. 0.05
        hi_q = getattr(mc, '_upper_pct', 95.0) / 100.0   # e.g. 0.95
        lo_label = f"P{mc._lower_pct:.4g}" if hasattr(mc, '_lower_pct') else "P5"
        hi_label = f"P{mc._upper_pct:.4g}" if hasattr(mc, '_upper_pct') else "P95"

        # Estimate SE of the percentiles via analytical formula
        # SE(P_q) ≈ sqrt(q*(1-q) / n) / f(x_q)  where f is the kernel density
        # at the percentile.  We use a simple bandwidth estimator.
        from scipy.stats import gaussian_kde
        try:
            kde = gaussian_kde(samples)
            f_p5 = float(kde.evaluate([mc.pct_5])[0])
            f_p95 = float(kde.evaluate([mc.pct_95])[0])
        except Exception:
            f_p5 = f_p95 = 0.0

        se_p5 = np.sqrt(lo_q * (1 - lo_q) / n) / max(f_p5, 1e-15)
        se_p95 = np.sqrt(hi_q * (1 - hi_q) / n) / max(f_p95, 1e-15)

        # Relative SE as fraction of the interval width
        interval_width = abs(mc.pct_95 - mc.pct_5)
        if interval_width > 1e-15:
            rel_se_p5 = se_p5 / interval_width
            rel_se_p95 = se_p95 / interval_width
            max_rel_se = max(rel_se_p5, rel_se_p95)
        else:
            max_rel_se = 0.0

        # Threshold: 1% of interval width ≈ 2 significant digits
        if max_rel_se > 0.02:
            severity = "red"
            msg = (
                f"\u26a0 MC percentile estimates may not be converged.\n"
                f"SE({lo_label}) \u2248 {se_p5:.4g}, SE({hi_label}) \u2248 {se_p95:.4g}  "
                f"(max relative SE = {max_rel_se:.1%} of interval width).\n\n"
                f"Consider increasing the number of MC trials for stable "
                f"results. [JCGM 101:2008 \u00a77.9]"
            )
        elif max_rel_se > 0.01:
            severity = "yellow"
            msg = (
                f"MC convergence is marginal.\n"
                f"SE({lo_label}) \u2248 {se_p5:.4g}, SE({hi_label}) \u2248 {se_p95:.4g}  "
                f"(max relative SE = {max_rel_se:.1%} of interval width).\n\n"
                f"Results are usable but increasing trials would improve "
                f"precision. [JCGM 101:2008 \u00a77.9]"
            )
        else:
            severity = "green"
            msg = (
                f"MC percentile estimates are well converged.\n"
                f"SE({lo_label}) \u2248 {se_p5:.4g}, SE({hi_label}) \u2248 {se_p95:.4g}  "
                f"(max relative SE = {max_rel_se:.1%} of interval width).\n"
                f"[JCGM 101:2008 \u00a77.9]"
            )

        self._guidance_mc_convergence.set_guidance(msg, severity)

    def _update_guidance(self, mc: MCResults):
        """
        Compare MC bounds against RSS bounds and set guidance severity.

        Both intervals are computed at the *same* coverage probability so the
        comparison is apples-to-apples.  The coverage comes from the RSS
        settings (e.g. 95% one-sided → P95, or 95% two-sided → P2.5/P97.5).

        Lighter-tailed (platykurtic / uniform) sources tend to produce
        a tighter MC bound, while heavier-tailed or skewed sources
        widen it relative to the RSS normal assumption.
        """
        rss = self._rss_results
        if rss is None or not rss.computed:
            self._guidance_mc_vs_rss.set_guidance(
                "No RSS results available for comparison.  Run the RSS "
                "analysis first to enable the MC-vs-RSS interpretation.",
                "green",
            )
            return

        # Determine the coverage probability used by the MC run
        # so we can extract the matching percentiles for comparison.
        coverage = getattr(mc, '_coverage', 0.95)
        one_sided = getattr(mc, '_one_sided', True)

        samples = mc.samples
        if samples.size == 0:
            self._guidance_mc_vs_rss.set_guidance(
                "No MC samples available.", "green")
            return

        mc_center = mc.combined_mean

        if one_sided:
            # One-sided: RSS upper bound = E_mean + k*u_val
            # Compare with MC upper percentile at same coverage
            rss_upper = rss.E_mean + rss.k_factor * rss.u_val
            mc_upper = float(np.percentile(samples, coverage * 100))
            rss_half = rss_upper - rss.E_mean
            mc_half = mc_upper - mc_center
            cov_label = f"{coverage*100:.0f}% one-sided"
        else:
            # Two-sided: RSS = E_mean ± k*u_val
            # Compare with MC symmetric percentile at same coverage
            alpha = (1 - coverage) / 2.0
            mc_lo = float(np.percentile(samples, alpha * 100))
            mc_hi = float(np.percentile(samples, (1 - alpha) * 100))
            rss_half = rss.k_factor * rss.u_val
            mc_half = (mc_hi - mc_lo) / 2.0
            cov_label = f"{coverage*100:.0f}% two-sided"

        if mc_half < 1e-12 and rss_half < 1e-12:
            self._guidance_mc_vs_rss.set_guidance(
                "Both MC and RSS bounds are effectively zero.",
                "green",
            )
            return

        ratio = mc_half / max(abs(rss_half), 1e-12)

        if ratio < 0.95:
            severity = "green"
            msg = (
                f"Monte Carlo {cov_label} half-width ({mc_half:.4g}) is "
                f"tighter than the RSS bound ({rss_half:.4g}).\n"
                f"Ratio MC/RSS = {ratio:.3f}.\n\n"
                "This is typical when the dominant uncertainty source has "
                "lighter tails than a normal distribution (platykurtic or "
                "uniform).  The RSS normal assumption is conservative here."
            )
        elif ratio > 1.05:
            severity = "yellow"
            msg = (
                f"Monte Carlo {cov_label} half-width ({mc_half:.4g}) is "
                f"wider than the RSS bound ({rss_half:.4g}).\n"
                f"Ratio MC/RSS = {ratio:.3f}.\n\n"
                "This may indicate heavier tails (leptokurtic), skewness, "
                "or non-linear interactions among sources.  Consider "
                "reviewing the distribution assumptions for dominant "
                "uncertainty contributors."
            )
        else:
            severity = "green"
            msg = (
                f"Monte Carlo {cov_label} half-width ({mc_half:.4g}) is "
                f"consistent with the RSS bound ({rss_half:.4g}).\n"
                f"Ratio MC/RSS = {ratio:.3f}.\n\n"
                "The RSS normal-assumption adequately represents the "
                "combined uncertainty for this set of sources."
            )

        self._guidance_mc_vs_rss.set_guidance(msg, severity)

    # =================================================================
    # PLOT UPDATES
    # =================================================================
    def _update_plots(self, mc: MCResults):
        """Redraw all three subplots with the latest MC results."""
        self._fig.clear()
        samples = mc.samples
        if samples.size == 0:
            self._canvas.draw_idle()
            return

        with plt.rc_context(PLOT_STYLE):
            axes = self._fig.subplots(3, 1)
            self._plot_histogram(axes[0], mc)
            self._plot_cdf(axes[1], mc)
            self._plot_convergence(axes[2], mc)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    self._fig.tight_layout(pad=2.0)
                except Exception:
                    pass

        self._canvas.draw_idle()

    # -- Subplot 1: Histogram with RSS overlay -----------------------
    def _plot_histogram(self, ax, mc: MCResults):
        """
        Histogram of the combined MC distribution with 5th / 95th
        percentile vertical lines and an RSS normal PDF overlay for
        visual comparison.
        """
        samples = mc.samples
        unit_str = getattr(self, '_display_unit', 'unit')

        # Histogram
        n_bins = min(200, max(50, int(np.sqrt(mc.n_trials))))
        ax.hist(
            samples, bins=n_bins, density=True,
            color=DARK_COLORS['accent'], edgecolor=DARK_COLORS['border'],
            alpha=0.7, linewidth=0.4, label="MC histogram",
        )

        # RSS normal overlay (if available)
        rss = self._rss_results
        if rss is not None and rss.computed and rss.u_val > 0:
            x_range = np.linspace(
                mc.combined_mean - 4.5 * mc.combined_std,
                mc.combined_mean + 4.5 * mc.combined_std,
                500,
            )
            # RSS uses E_mean and u_val as the normal parameters
            rss_pdf = norm.pdf(x_range, loc=rss.E_mean, scale=rss.u_val)
            ax.plot(
                x_range, rss_pdf, color=DARK_COLORS['red'],
                linewidth=1.5, linestyle='--',
                label=f"RSS Normal (\u03c3=u_val={rss.u_val:.3g})",
            )

        # Percentile lines — use coverage-matched labels
        lo_pct = getattr(mc, '_lower_pct', 5.0)
        hi_pct = getattr(mc, '_upper_pct', 95.0)
        ax.axvline(
            mc.pct_5, color=DARK_COLORS['orange'], linewidth=1.2,
            linestyle='-.', alpha=0.9,
            label=f"P{lo_pct:.4g} = {mc.pct_5:+.4g}",
        )
        ax.axvline(
            mc.pct_95, color=DARK_COLORS['red'], linewidth=1.2,
            linestyle='-.', alpha=0.9,
            label=f"P{hi_pct:.4g} = {mc.pct_95:+.4g}",
        )
        # Mean line
        ax.axvline(
            mc.combined_mean, color=DARK_COLORS['fg'], linewidth=1.0,
            linestyle='-', alpha=0.7,
            label=f"Mean = {mc.combined_mean:+.4g}",
        )

        ax.set_xlabel(f"Combined Error [{unit_str}]",
                       color=DARK_COLORS['fg'], fontsize=8)
        ax.set_ylabel("Probability Density",
                       color=DARK_COLORS['fg'], fontsize=8)
        ax.set_title("MC Combined Error Distribution",
                      color=DARK_COLORS['fg'], fontsize=9, fontweight='bold')
        ax.legend(fontsize=6.5, loc='upper right',
                  facecolor=DARK_COLORS['bg_widget'],
                  edgecolor=DARK_COLORS['border'],
                  labelcolor=DARK_COLORS['fg'])
        ax.tick_params(axis='x', colors=DARK_COLORS['fg_dim'])
        ax.tick_params(axis='y', colors=DARK_COLORS['fg_dim'])
        ax.set_axisbelow(True)
        ax.grid(alpha=0.3)

    # -- Subplot 2: CDF with percentile markers ----------------------
    def _plot_cdf(self, ax, mc: MCResults):
        """
        Empirical CDF of the MC combined-error distribution with
        horizontal lines at the 5th and 95th coverage percentiles.
        """
        samples = mc.samples
        unit_str = getattr(self, '_display_unit', 'unit')

        sorted_samples = np.sort(samples)
        cdf_y = np.arange(1, len(sorted_samples) + 1) / len(sorted_samples)

        ax.plot(
            sorted_samples, cdf_y,
            color=DARK_COLORS['accent'], linewidth=1.2,
            label="Empirical CDF",
        )

        # Horizontal lines at coverage-matched percentile levels
        lo_frac = getattr(mc, '_lower_pct', 5.0) / 100.0
        hi_frac = getattr(mc, '_upper_pct', 95.0) / 100.0
        lo_cdf_label = f"{mc._lower_pct:.4g}%" if hasattr(mc, '_lower_pct') else "5%"
        hi_cdf_label = f"{mc._upper_pct:.4g}%" if hasattr(mc, '_upper_pct') else "95%"
        ax.axhline(
            lo_frac, color=DARK_COLORS['orange'], linewidth=1.0,
            linestyle=':', alpha=0.8, label=f"{lo_cdf_label} level",
        )
        ax.axhline(
            hi_frac, color=DARK_COLORS['red'], linewidth=1.0,
            linestyle=':', alpha=0.8, label=f"{hi_cdf_label} level",
        )

        # Vertical drop lines from percentile values
        ax.axvline(
            mc.pct_5, color=DARK_COLORS['orange'], linewidth=0.8,
            linestyle='-.', alpha=0.6,
        )
        ax.axvline(
            mc.pct_95, color=DARK_COLORS['red'], linewidth=0.8,
            linestyle='-.', alpha=0.6,
        )

        ax.set_xlabel(f"Combined Error [{unit_str}]",
                       color=DARK_COLORS['fg'], fontsize=8)
        ax.set_ylabel("Cumulative Probability",
                       color=DARK_COLORS['fg'], fontsize=8)
        ax.set_title("MC Cumulative Distribution Function",
                      color=DARK_COLORS['fg'], fontsize=9, fontweight='bold')
        ax.legend(fontsize=6.5, loc='lower right',
                  facecolor=DARK_COLORS['bg_widget'],
                  edgecolor=DARK_COLORS['border'],
                  labelcolor=DARK_COLORS['fg'])
        ax.set_ylim(-0.02, 1.02)
        ax.tick_params(axis='x', colors=DARK_COLORS['fg_dim'])
        ax.tick_params(axis='y', colors=DARK_COLORS['fg_dim'])
        ax.set_axisbelow(True)
        ax.grid(alpha=0.3)

    # -- Subplot 3: Convergence plot ---------------------------------
    def _plot_convergence(self, ax, mc: MCResults):
        """
        Running estimate of the 5th and 95th percentiles as a function
        of the number of trials, demonstrating the stability (or lack
        thereof) of the Monte Carlo estimate.
        """
        samples = mc.samples
        n = len(samples)
        if n < 100:
            ax.text(
                0.5, 0.5, "Insufficient trials for convergence plot",
                ha='center', va='center', color=DARK_COLORS['fg_dim'],
                fontsize=9, transform=ax.transAxes,
            )
            return

        # Subsample evaluation points (log-spaced for readability)
        eval_points = np.unique(np.geomspace(100, n, num=200).astype(int))
        p5_running = np.empty(len(eval_points))
        p95_running = np.empty(len(eval_points))

        lo_pct = getattr(mc, '_lower_pct', 5.0)
        hi_pct = getattr(mc, '_upper_pct', 95.0)
        for i, k in enumerate(eval_points):
            p5_running[i] = np.percentile(samples[:k], lo_pct)
            p95_running[i] = np.percentile(samples[:k], hi_pct)

        ax.plot(
            eval_points, p5_running,
            color=DARK_COLORS['orange'], linewidth=1.2,
            label=f"Running P{lo_pct:.4g}",
        )
        ax.plot(
            eval_points, p95_running,
            color=DARK_COLORS['red'], linewidth=1.2,
            label=f"Running P{hi_pct:.4g}",
        )

        # Final converged values as horizontal reference
        ax.axhline(
            mc.pct_5, color=DARK_COLORS['orange'], linewidth=0.8,
            linestyle=':', alpha=0.5,
        )
        ax.axhline(
            mc.pct_95, color=DARK_COLORS['red'], linewidth=0.8,
            linestyle=':', alpha=0.5,
        )

        ax.set_xlabel("Number of Trials",
                       color=DARK_COLORS['fg'], fontsize=8)
        ax.set_ylabel("Percentile Estimate",
                       color=DARK_COLORS['fg'], fontsize=8)
        ax.set_title("MC Convergence (Percentile vs Trials)",
                      color=DARK_COLORS['fg'], fontsize=9, fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=6.5, loc='upper right',
                  facecolor=DARK_COLORS['bg_widget'],
                  edgecolor=DARK_COLORS['border'],
                  labelcolor=DARK_COLORS['fg'])
        ax.tick_params(axis='x', colors=DARK_COLORS['fg_dim'])
        ax.tick_params(axis='y', colors=DARK_COLORS['fg_dim'])
        ax.set_axisbelow(True)
        ax.grid(alpha=0.3)


# =============================================================================
# SECTION 12: TAB 6 — COMPARISON ROLL-UP TAB
# =============================================================================

class ComparisonRollUpTab(QWidget):
    """
    Tab 6: Comparison Roll-Up — Final Summary & Side-by-Side Comparison.

    Presents a consolidated view of RSS and Monte Carlo validation results
    side-by-side, auto-generates key findings and assumptions lists, and
    provides export / save functionality.

    Sections:
      1. Side-by-side comparison table (RSS u_val, RSS s_E, MC, Empirical)
      2. Key findings (auto-generated narrative)
      3. Assumptions & engineering judgments (from audit_log)
      4. Compare Projects (load a second .json project for comparison)
      5. Export / Save controls

    References:
        - ASME V&V 20-2009 Section 2.5 (Validation Assessment)
        - AIAA G-077-1998 (Reporting of V&V Results)
    """

    export_requested = Signal()
    save_requested = Signal()

    # ------------------------------------------------------------------
    # Row labels for the comparison table
    # ------------------------------------------------------------------
    _ROW_LABELS = [
        "Combined \u03c3 or equivalent",
        "k-factor used",
        "Expanded uncertainty (U_val = k\u00d7\u03c3)",
        "Lower bound (underprediction)",
        "Upper bound (overprediction)",
        "Mean comparison error (\u0112)",
        "|\u0112| \u2264 U_val ?  (Validated?)",
        "Includes model form error?",
        "Distribution assumption",
        "Reference standard",
    ]

    _N_ROWS = len(_ROW_LABELS)
    _BASE_COLUMNS = ["Quantity", "RSS (u_val)", "RSS (s_E)",
                      "Monte Carlo", "Empirical"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rss: Optional[RSSResults] = None
        self._mc: Optional[MCResults] = None
        self._comp: Optional[ComparisonData] = None
        self._settings: Optional[AnalysisSettings] = None
        self._sources: list = []
        self._comparison_rss: Optional[RSSResults] = None
        self._comparison_mc: Optional[MCResults] = None
        self._comparison_label: str = ""
        self._setup_ui()

    # =================================================================
    # PUBLIC API
    # =================================================================
    def update_rollup(
        self,
        rss_results: RSSResults,
        mc_results: MCResults,
        comp_data: ComparisonData,
        settings: AnalysisSettings,
        sources: list,
    ):
        """
        Populate every section of the roll-up tab with current results.

        Parameters
        ----------
        rss_results : RSSResults
            RSS validation-uncertainty results.
        mc_results : MCResults
            Monte Carlo propagation results.
        comp_data : ComparisonData
            Raw comparison-error data.
        settings : AnalysisSettings
            Current analysis settings.
        sources : list[UncertaintySource]
            All uncertainty sources (enabled + disabled).
        """
        self._rss = rss_results
        self._mc = mc_results
        self._comp = comp_data
        self._settings = settings
        self._sources = sources

        self._populate_table()
        self._generate_findings()
        self._generate_assumptions()

    def get_findings_text(self) -> str:
        """Return the auto-generated key-findings narrative."""
        return self._findings_edit.toPlainText()

    def get_assumptions_text(self) -> str:
        """Return the auto-generated assumptions list."""
        return self._assumptions_edit.toPlainText()

    def get_rollup_table_data(self) -> list:
        """
        Return the comparison table contents as a list of lists (for export).

        Each inner list is one row: [row_label, rss_uval, rss_sE, mc, empirical].
        If a comparison project is loaded, additional columns are appended.
        """
        data = []
        n_cols = self._table.columnCount()
        for r in range(self._table.rowCount()):
            row = []
            for c in range(n_cols):
                item = self._table.item(r, c)
                row.append(item.text() if item else "")
            data.append(row)
        return data

    def get_rollup_header_labels(self) -> list:
        """Return the current column header labels from the rollup table."""
        headers = []
        for c in range(self._table.columnCount()):
            item = self._table.horizontalHeaderItem(c)
            headers.append(item.text() if item else f"Col {c}")
        return headers

    # =================================================================
    # UI CONSTRUCTION
    # =================================================================
    def _setup_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(scroll)

        container = QWidget()
        self._layout = QVBoxLayout(container)
        self._layout.setContentsMargins(12, 12, 12, 12)
        self._layout.setSpacing(14)
        scroll.setWidget(container)

        # -- Section 1: Comparison table ---------------------------------
        self._build_table_section()

        # -- Section 2: Key findings -------------------------------------
        self._build_findings_section()

        # -- Section 3: Assumptions & engineering judgments ---------------
        self._build_assumptions_section()

        # -- Section 4: Compare projects ---------------------------------
        self._build_compare_section()

        # -- Section 5: Export / Save ------------------------------------
        self._build_export_section()

        self._layout.addStretch(1)

    # -----------------------------------------------------------------
    # Section 1 — Side-by-Side Comparison Table
    # -----------------------------------------------------------------
    def _build_table_section(self):
        group = QGroupBox("Side-by-Side Comparison")
        group.setStyleSheet(
            f"QGroupBox {{ font-weight: bold; color: {DARK_COLORS['accent']}; }}"
        )
        vbox = QVBoxLayout(group)

        self._table = QTableWidget(self._N_ROWS, len(self._BASE_COLUMNS))
        self._table.setHorizontalHeaderLabels(self._BASE_COLUMNS)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setAlternatingRowColors(True)
        style_table(self._table,
                    column_widths={0: 210, 1: 120, 2: 120, 3: 120, 4: 110},
                    stretch_col=0)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setMinimumHeight(300)

        # Pre-fill row labels
        for r, label in enumerate(self._ROW_LABELS):
            item = QTableWidgetItem(label)
            item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self._table.setItem(r, 0, item)

        vbox.addWidget(self._table)
        self._layout.addWidget(group)

    # -----------------------------------------------------------------
    # Section 2 — Key Findings
    # -----------------------------------------------------------------
    def _build_findings_section(self):
        group = QGroupBox("Key Findings (Auto-Generated)")
        group.setStyleSheet(
            f"QGroupBox {{ font-weight: bold; color: {DARK_COLORS['accent']}; }}"
        )
        vbox = QVBoxLayout(group)

        # Audience mode toolbar
        audience_toolbar = QHBoxLayout()
        self._cmb_audience = QComboBox()
        self._cmb_audience.addItems([
            "Internal Engineering",
            "External Technical Report",
            "Regulatory Submission",
        ])
        self._cmb_audience.setCurrentIndex(0)  # default: Internal
        self._cmb_audience.setToolTip(
            "Statement audience mode:\n"
            "  \u2022 Internal Engineering \u2014 terse, numeric-heavy (current format)\n"
            "  \u2022 External Technical Report \u2014 adds method references and scope context\n"
            "  \u2022 Regulatory Submission \u2014 adds full assumptions, conditional language, standards citations"
        )
        self._cmb_audience.currentIndexChanged.connect(self._generate_findings)
        audience_toolbar.addWidget(QLabel("Audience:"))
        audience_toolbar.addWidget(self._cmb_audience)
        audience_toolbar.addStretch()
        vbox.addLayout(audience_toolbar)

        self._findings_edit = QTextEdit()
        self._findings_edit.setReadOnly(False)  # user may append notes
        self._findings_edit.setMinimumHeight(180)
        self._findings_edit.setPlaceholderText(
            "Run both RSS and Monte Carlo analyses to auto-generate findings."
        )
        vbox.addWidget(self._findings_edit)
        self._layout.addWidget(group)

    # -----------------------------------------------------------------
    # Section 3 — Assumptions & Engineering Judgments
    # -----------------------------------------------------------------
    def _build_assumptions_section(self):
        group = QGroupBox("Assumptions & Engineering Judgments")
        group.setStyleSheet(
            f"QGroupBox {{ font-weight: bold; color: {DARK_COLORS['accent']}; }}"
        )
        vbox = QVBoxLayout(group)

        self._assumptions_edit = QTextEdit()
        self._assumptions_edit.setReadOnly(False)
        self._assumptions_edit.setMinimumHeight(160)
        self._assumptions_edit.setPlaceholderText(
            "Assumptions will be populated from the audit log when analyses complete."
        )
        vbox.addWidget(self._assumptions_edit)
        self._layout.addWidget(group)

    # -----------------------------------------------------------------
    # Section 4 — Compare Projects
    # -----------------------------------------------------------------
    def _build_compare_section(self):
        group = QGroupBox("Compare Projects")
        group.setStyleSheet(
            f"QGroupBox {{ font-weight: bold; color: {DARK_COLORS['accent']}; }}"
        )
        hbox = QHBoxLayout(group)

        self._btn_load_comparison = QPushButton("Load Comparison Project")
        self._btn_load_comparison.setMinimumWidth(180)
        self._btn_load_comparison.clicked.connect(self.load_comparison_project)
        hbox.addWidget(self._btn_load_comparison)

        self._comparison_info_label = QLabel("No comparison project loaded.")
        self._comparison_info_label.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']};"
        )
        hbox.addWidget(self._comparison_info_label, 1)

        self._btn_clear_comparison = QPushButton("Clear Comparison")
        self._btn_clear_comparison.setMinimumWidth(140)
        self._btn_clear_comparison.setEnabled(False)
        self._btn_clear_comparison.clicked.connect(self._clear_comparison)
        hbox.addWidget(self._btn_clear_comparison)

        self._layout.addWidget(group)

    # -----------------------------------------------------------------
    # Section 5 — Export / Save
    # -----------------------------------------------------------------
    def _build_export_section(self):
        group = QGroupBox("Export / Save")
        group.setStyleSheet(
            f"QGroupBox {{ font-weight: bold; color: {DARK_COLORS['accent']}; }}"
        )
        hbox = QHBoxLayout(group)

        btn_clipboard = QPushButton("Export to Clipboard")
        btn_clipboard.setMinimumWidth(160)
        btn_clipboard.clicked.connect(self._export_to_clipboard)
        hbox.addWidget(btn_clipboard)

        btn_html = QPushButton("Export Full Report (HTML)")
        btn_html.setMinimumWidth(190)
        btn_html.clicked.connect(lambda: self.export_requested.emit())
        hbox.addWidget(btn_html)

        btn_save = QPushButton("Save Project")
        btn_save.setMinimumWidth(130)
        btn_save.clicked.connect(lambda: self.save_requested.emit())
        hbox.addWidget(btn_save)

        hbox.addStretch(1)
        self._layout.addWidget(group)

    # =================================================================
    # TABLE POPULATION
    # =================================================================
    def _populate_table(self):
        """Fill columns 1-4 from the current RSS / MC / empirical results."""
        rss = self._rss
        mc = self._mc
        comp = self._comp
        settings = self._settings

        if comp is not None and comp.flat_data().size > 0:
            stats = comp.get_stats()
        else:
            stats = {}

        # ---------- column helpers ------------------------------------
        def _set(row: int, col: int, text: str, color: str = ""):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            if color:
                item.setForeground(QColor(color))
            self._table.setItem(row, col, item)

        # Row indices:
        #  0  Combined σ or equivalent
        #  1  k-factor used
        #  2  Expanded uncertainty (U_val = k×σ)
        #  3  Lower bound (underprediction)
        #  4  Upper bound (overprediction)
        #  5  Mean comparison error (Ē)
        #  6  |Ē| ≤ U_val ?  (Validated?)
        #  7  Includes model form error?
        #  8  Distribution assumption
        #  9  Reference standard

        # ---------- RSS (u_val) column  — col 1 -----------------------
        if rss is not None and rss.computed:
            _set(0, 1, f"{rss.u_val:.4f}")
            _set(1, 1, f"{rss.k_factor:.3f}")
            _set(2, 1, f"{rss.U_val:.4f}")
            _set(3, 1, f"{rss.lower_bound_uval:.4f}"
                 if not np.isnan(rss.lower_bound_uval) else "\u2014")
            _set(4, 1, f"{rss.upper_bound_uval:.4f}"
                 if not np.isnan(rss.upper_bound_uval) else "\u2014")
            _set(5, 1, f"{rss.E_mean:.4f}")
            # Validation verdict
            if rss.U_val > 0:
                ratio = abs(rss.E_mean) / rss.U_val
                if ratio <= 1.0:
                    _set(6, 1, f"YES ({ratio:.3f} \u2264 1.0)",
                         DARK_COLORS.get('green', '#66bb6a'))
                else:
                    _set(6, 1, f"NO ({ratio:.3f} > 1.0)",
                         DARK_COLORS.get('red', '#ef5350'))
            else:
                _set(6, 1, "\u2014")
            _set(7, 1, "No")
            _set(8, 1, "Normal")
            _set(9, 1, "ASME V&V 20-2009")
        else:
            for r in range(self._N_ROWS):
                _set(r, 1, "\u2014")

        # ---------- RSS (s_E) column  — col 2 -------------------------
        if rss is not None and rss.computed:
            U_sE = rss.k_factor * rss.s_E
            _set(0, 2, f"{rss.s_E:.4f}")
            _set(1, 2, f"{rss.k_factor:.3f}")
            _set(2, 2, f"{U_sE:.4f}")
            _set(3, 2, f"{rss.lower_bound_sE:.4f}"
                 if not np.isnan(rss.lower_bound_sE) else "\u2014")
            _set(4, 2, f"{rss.upper_bound_sE:.4f}"
                 if not np.isnan(rss.upper_bound_sE) else "\u2014")
            _set(5, 2, f"{rss.E_mean:.4f}")
            if U_sE > 0:
                ratio_sE = abs(rss.E_mean) / U_sE
                if ratio_sE <= 1.0:
                    _set(6, 2, f"YES ({ratio_sE:.3f} \u2264 1.0)",
                         DARK_COLORS.get('green', '#66bb6a'))
                else:
                    _set(6, 2, f"NO ({ratio_sE:.3f} > 1.0)",
                         DARK_COLORS.get('red', '#ef5350'))
            else:
                _set(6, 2, "\u2014")
            _set(7, 2, "Yes")
            _set(8, 2, "Normal")
            _set(9, 2, "ASME V&V 20-2009")
        else:
            for r in range(self._N_ROWS):
                _set(r, 2, "\u2014")

        # ---------- Monte Carlo column  — col 3 -----------------------
        if mc is not None and mc.computed:
            mc_half = (mc.pct_95 - mc.pct_5) / 2.0
            lo_lbl = f"P{mc._lower_pct:.4g}" if hasattr(mc, '_lower_pct') else "P5"
            hi_lbl = f"P{mc._upper_pct:.4g}" if hasattr(mc, '_upper_pct') else "P95"
            _set(0, 3, f"{mc.combined_std:.4f}")
            _set(1, 3, "N/A (distribution-free)")
            _set(2, 3, f"\u00b1{mc_half:.4f} ({lo_lbl}\u2013{hi_lbl})")
            _set(3, 3, f"{mc.lower_bound:.4f}")
            _set(4, 3, f"{mc.upper_bound:.4f}")
            _set(5, 3, f"{mc.combined_mean:.4f}")
            # MC validation: is Ē within the MC prediction interval?
            E_mean = mc.combined_mean
            if mc.lower_bound <= 0 <= mc.upper_bound:
                _set(6, 3, f"YES (\u0112 within {lo_lbl}\u2013{hi_lbl})",
                     DARK_COLORS.get('green', '#66bb6a'))
            else:
                _set(6, 3, f"NO (\u0112 outside {lo_lbl}\u2013{hi_lbl})",
                     DARK_COLORS.get('red', '#ef5350'))
            _set(7, 3, "Depends on sources")
            _set(8, 3, "Actual (sampled)")
            _set(9, 3, "JCGM 101:2008")
        else:
            for r in range(self._N_ROWS):
                _set(r, 3, "\u2014")

        # ---------- Empirical column  — col 4 -------------------------
        if stats:
            emp_std = stats.get('std', 0.0)
            emp_mean = stats.get('mean', 0.0)
            n = stats.get('n', 0)
            flat = comp.flat_data()
            _set(0, 4, f"{emp_std:.4f}")
            _set(1, 4, "N/A (empirical)")
            _set(2, 4, "N/A")
            if flat.size > 0:
                _set(3, 4, f"{float(np.min(flat)):.4f}")
                _set(4, 4, f"{float(np.max(flat)):.4f}")
            else:
                _set(3, 4, "\u2014")
                _set(4, 4, "\u2014")
            _set(5, 4, f"{emp_mean:.4f}")
            _set(6, 4, "N/A")
            _set(7, 4, "Yes (all sources)")
            _set(8, 4, "None (raw data)")
            _set(9, 4, f"n = {n} data points")
        else:
            for r in range(self._N_ROWS):
                _set(r, 4, "\u2014")

        self._table.resizeColumnsToContents()

    # =================================================================
    # KEY FINDINGS AUTO-GENERATION
    # =================================================================
    def _generate_findings(self):
        """
        Build a narrative summary paragraph from the current results.

        Addresses:
          - Mean bias and significance
          - Validation assessment (|E_mean| vs U_val)
          - Underprediction bounds from each method
          - Dominant uncertainty source
          - Data quality concerns
          - Certification-relevant language
        """
        rss = self._rss
        mc = self._mc
        comp = self._comp
        settings = self._settings
        sources = self._sources

        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  KEY FINDINGS — Auto-Generated Summary")
        lines.append("=" * 60)
        lines.append("")

        # --- Mean bias --------------------------------------------------
        if rss is not None and rss.computed:
            if rss.n_data == 0:
                lines.append(
                    "1. MEAN BIAS: Not available — no comparison data loaded."
                )
            else:
                bias = rss.E_mean
                direction = "overprediction" if bias > 0 else "underprediction"
                lines.append(
                    f"1. MEAN BIAS: The mean comparison error is "
                    f"\u0112 = {bias:+.4f} ({direction})."
                )
                if rss.U_val > 0:
                    ratio = abs(bias) / rss.U_val
                    if ratio < 1.0:
                        lines.append(
                            f"   |\u0112| / U_val = {ratio:.3f} < 1.0 \u2014 "
                            f"the model IS validated at the {settings.coverage*100:.0f}% / "
                            f"{settings.confidence*100:.0f}% level."
                        )
                    else:
                        lines.append(
                            f"   |\u0112| / U_val = {ratio:.3f} \u2265 1.0 \u2014 "
                            f"the model is NOT validated at the {settings.coverage*100:.0f}% / "
                            f"{settings.confidence*100:.0f}% level."
                        )
            lines.append("")

        # --- Underprediction bounds from each method --------------------
        lines.append("2. UNDERPREDICTION BOUNDS:")
        if rss is not None and rss.computed:
            if not np.isnan(rss.lower_bound_uval):
                lines.append(
                    f"   RSS (u_val basis):  {rss.lower_bound_uval:+.4f}"
                )
            if not np.isnan(rss.lower_bound_sE):
                lines.append(
                    f"   RSS (s_E basis):    {rss.lower_bound_sE:+.4f}"
                )
        if mc is not None and mc.computed:
            lines.append(
                f"   Monte Carlo:        {mc.lower_bound:+.4f}"
            )
        if comp is not None and comp.flat_data().size > 0:
            lines.append(
                f"   Empirical min:      {float(np.min(comp.flat_data())):+.4f}"
            )
        lines.append("")

        # --- Dominant uncertainty source --------------------------------
        if sources:
            enabled = [s for s in sources if s.enabled]
            if enabled:
                dominant = max(enabled, key=lambda s: s.get_standard_uncertainty())
                dom_sigma = dominant.get_standard_uncertainty()
                lines.append(
                    f"3. DOMINANT SOURCE: \"{dominant.name}\" "
                    f"(\u03c3 = {dom_sigma:.4f}, category: {dominant.category})."
                )
                lines.append("")

        # --- Data quality concerns --------------------------------------
        concern_lines: list[str] = []
        if comp is not None:
            flat = comp.flat_data()
            if flat.size < 10:
                concern_lines.append(
                    f"   - Small data set (n = {flat.size}); "
                    "statistical confidence may be limited."
                )
            if flat.size > 2:
                from scipy import stats as sp_stats
                _, shapiro_p = sp_stats.shapiro(flat[:5000])
                if shapiro_p < 0.05:
                    concern_lines.append(
                        f"   - Shapiro-Wilk p = {shapiro_p:.4f} \u2014 data "
                        "may not be normally distributed; MC bounds are more "
                        "appropriate."
                    )
        if concern_lines:
            lines.append("4. DATA QUALITY CONCERNS:")
            lines.extend(concern_lines)
            lines.append("")
        else:
            lines.append("4. DATA QUALITY CONCERNS: None identified.")
            lines.append("")

        # --- Expanded Certification Statement ----------------------------
        lines.append("=" * 60)
        lines.append("  VALIDATION CERTIFICATION STATEMENT")
        lines.append("=" * 60)
        lines.append("")

        sided_label = "one-sided" if settings.one_sided else "two-sided"
        cov_pct = f"{settings.coverage*100:.0f}%"
        conf_pct = f"{settings.confidence*100:.0f}%"
        unit = settings.global_unit

        # --- 5a. RSS Validation Assessment (ASME V&V 20) ---
        lines.append("5a. RSS VALIDATION ASSESSMENT (ASME V&V 20):")
        if rss is not None and rss.computed and rss.n_data == 0:
            lines.append("    VALIDATION VERDICT: Not available — no comparison data loaded.")
            lines.append("    Load experimental comparison data (E = S − D) on Tab 1")
            lines.append("    to enable the ASME V&V 20 validation assessment.")
        elif rss is not None and rss.computed and rss.U_val > 0:
            ratio = abs(rss.E_mean) / rss.U_val
            bound_type = getattr(settings, 'bound_type', 'Both (for comparison)')
            corr_info = ("Independent" if not rss.has_correlations
                         else f"Correlated (groups: {', '.join(rss.correlation_groups)})")
            lines.append(f"    Method:           Root-Sum-Square (RSS) per ASME V&V 20-2009 (R2021)")
            lines.append(f"    Coverage:         {cov_pct} / {conf_pct} ({sided_label})")
            lines.append(f"    k-factor:         {rss.k_factor:.4f}  (method: {rss.k_method_used})")
            lines.append(f"    Bound basis:      {bound_type}")
            lines.append(f"    Correlation:      {corr_info}")
            lines.append(f"    Sample size:      n_data = {rss.n_data}")
            if rss.n_data >= 30:
                suff_label = "adequate"
            elif rss.n_data >= 10:
                suff_label = "marginal"
            else:
                suff_label = "insufficient per GUM \u00a7G.3"
            lines.append(f"    Data sufficiency: {suff_label}  (n = {rss.n_data})")
            lines.append(f"    u_val:            {rss.u_val:.4f} [{unit}]  (combined standard uncertainty)")
            lines.append(f"    U_val:            {rss.U_val:.4f} [{unit}]  (expanded uncertainty = k \u00d7 u_val)")
            lines.append(f"    |\u0112|:              {abs(rss.E_mean):.4f} [{unit}]  (mean comparison error magnitude)")
            lines.append(f"    |\u0112| / U_val:      {ratio:.4f}")
            lines.append(f"    Class split:      U_A = {rss.u_aleatoric:.4f}, U_E = {rss.u_epistemic:.4f} (epistemic: {rss.pct_epistemic:.1f}%)")
            if not np.isnan(rss.lower_bound_uval):
                lines.append(f"    Prediction band (u_val):  [{rss.lower_bound_uval:+.4f}, {rss.upper_bound_uval:+.4f}] [{unit}]")
            if not np.isnan(rss.lower_bound_sE):
                lines.append(f"    Prediction band (s_E):    [{rss.lower_bound_sE:+.4f}, {rss.upper_bound_sE:+.4f}] [{unit}]")
            if ratio <= 1.0:
                lines.append(f"")
                lines.append(f"    \u2713 VALIDATED: The mean comparison error (|\u0112| = {abs(rss.E_mean):.4f}) is")
                lines.append(f"      within the expanded validation uncertainty (U_val = {rss.U_val:.4f}).")
                lines.append(f"      The model is validated at {cov_pct} coverage / {conf_pct} confidence")
                lines.append(f"      for the tested conditions per ASME V&V 20 \u00a76, Eq. (1).")
            else:
                lines.append(f"")
                lines.append(f"    \u2717 NOT VALIDATED: The mean comparison error (|\u0112| = {abs(rss.E_mean):.4f})")
                lines.append(f"      exceeds the expanded validation uncertainty (U_val = {rss.U_val:.4f}).")
                lines.append(f"      A statistically significant model bias exists that is not")
                lines.append(f"      explained by the identified uncertainty sources.")
                lines.append(f"      Recommendation: investigate systematic errors in boundary")
                lines.append(f"      conditions, grid convergence, or turbulence modelling, or")
                lines.append(f"      identify additional uncertainty sources.")

            # --- 8b. Conditional finding caveats ---
            _enabled_sources = [s for s in (sources or []) if s.enabled]
            assumed_sources = [s for s in _enabled_sources
                               if getattr(s, 'basis_type', '') == 'assumed']
            high_red_sources = [s for s in _enabled_sources
                                if getattr(s, 'reducibility', '') == 'high']
            if assumed_sources or high_red_sources or rss.pct_epistemic > 50.0:
                lines.append("")
                lines.append("    --- Conditional Finding Caveats ---")
                if assumed_sources:
                    lines.append(
                        f"    Note: {len(assumed_sources)} source(s) rely on assumed "
                        "uncertainty estimates."
                    )
                if high_red_sources:
                    lines.append(
                        f"    Note: {len(high_red_sources)} source(s) have high "
                        "reducibility \u2014 additional testing could narrow the "
                        "uncertainty budget."
                    )
                # --- 8c. Epistemic dominance caveat ---
                if rss.pct_epistemic > 50.0:
                    lines.append("")
                    lines.append(
                        "    \u26a0 CONDITIONAL FINDING: Epistemic uncertainty contributes "
                        f">{rss.pct_epistemic:.0f}% of the"
                    )
                    lines.append(
                        "      total variance. This validation conclusion should be interpreted with"
                    )
                    lines.append(
                        "      caution. Reducing epistemic sources would strengthen the assessment."
                    )
        else:
            lines.append("    RSS analysis not available or U_val = 0.")
        lines.append("")

        # --- 5b. Monte Carlo Validation Assessment (JCGM 101) ---
        lines.append("5b. MONTE CARLO VALIDATION ASSESSMENT (JCGM 101:2008):")
        if mc is not None and mc.computed:
            mc_half = (mc.pct_95 - mc.pct_5) / 2.0
            mc_lo_lbl = f"P{mc._lower_pct:.4g}" if hasattr(mc, '_lower_pct') else "P5"
            mc_hi_lbl = f"P{mc._upper_pct:.4g}" if hasattr(mc, '_upper_pct') else "P95"
            mc_cov_pct = f"{mc._coverage*100:.0f}%" if hasattr(mc, '_coverage') else cov_pct
            mc_sided = "one-sided" if getattr(mc, '_one_sided', True) else "two-sided"
            mc_interval_label = f"{mc_cov_pct} {mc_sided}"
            mc_method_name = getattr(mc, 'sampling_method', 'Monte Carlo (Random)')
            if "LHS" in mc_method_name:
                method_line = ("Monte Carlo propagation (Latin Hypercube "
                               "stratified sampling; McKay et al. 1979; "
                               "JCGM 101 framework)")
            else:
                method_line = "Monte Carlo propagation (random sampling) per JCGM 101:2008"
            lines.append(f"    Method:           {method_line}")
            lines.append(f"    Trials:           {mc.n_trials:,}")
            lines.append(f"    Coverage:         {mc_interval_label}")
            lines.append(f"    Combined \u03c3:       {mc.combined_std:.4f} [{unit}]")
            lines.append(f"    {mc_lo_lbl} (lower bound): {mc.pct_5:+.4f} [{unit}]")
            lines.append(f"    {mc_hi_lbl} (upper bound): {mc.pct_95:+.4f} [{unit}]")
            lines.append(f"    Interval half-width: \u00b1{mc_half:.4f} [{unit}]  ({mc_interval_label})")
            lines.append(f"    \u0112 (MC mean):      {mc.combined_mean:+.4f} [{unit}]")
            if hasattr(mc, '_p5_boots') and hasattr(mc, '_p95_boots'):
                p5_ci = (float(np.percentile(mc._p5_boots, 2.5)),
                         float(np.percentile(mc._p5_boots, 97.5)))
                p95_ci = (float(np.percentile(mc._p95_boots, 2.5)),
                          float(np.percentile(mc._p95_boots, 97.5)))
                lines.append(f"    Bootstrap 95% CI on {mc_lo_lbl}:  [{p5_ci[0]:+.4f}, {p5_ci[1]:+.4f}] [{unit}]")
                lines.append(f"    Bootstrap 95% CI on {mc_hi_lbl}: [{p95_ci[0]:+.4f}, {p95_ci[1]:+.4f}] [{unit}]")
            lines.append(f"")
            # MC validation: the prediction interval should contain zero
            # (zero = no prediction error) if the model is validated
            if mc.pct_5 <= 0 <= mc.pct_95:
                lines.append(f"    \u2713 VALIDATED: Zero (no prediction error) falls within the")
                lines.append(f"      Monte Carlo {mc_interval_label} prediction interval [{mc.pct_5:+.4f}, {mc.pct_95:+.4f}].")
                lines.append(f"      The model prediction uncertainty, propagated from actual source")
                lines.append(f"      distributions without assuming normality, encompasses the")
                lines.append(f"      observed comparison errors.")
            else:
                lines.append(f"    \u2717 NOT VALIDATED: Zero falls outside the Monte Carlo")
                lines.append(f"      {mc_interval_label} prediction interval [{mc.pct_5:+.4f}, {mc.pct_95:+.4f}].")
                lines.append(f"      The model exhibits a systematic bias that exceeds the")
                lines.append(f"      propagated uncertainty from the identified sources.")
        else:
            lines.append("    Monte Carlo analysis not available.")
        lines.append("")

        # --- 5c. MC vs RSS Agreement ---
        lines.append("5c. MC vs RSS AGREEMENT:")
        if (rss is not None and rss.computed and rss.U_val > 0
                and mc is not None and mc.computed):
            mc_half = (mc.pct_95 - mc.pct_5) / 2.0
            rss_half = rss.k_factor * rss.u_val
            if rss_half > 1e-12:
                mc_rss_ratio = mc_half / rss_half
                mc_cov_label = mc_interval_label if (mc is not None and mc.computed and hasattr(mc, '_coverage')) else f"{cov_pct} {sided_label}"
                lines.append(f"    MC {mc_cov_label} half-width / RSS half-width = {mc_rss_ratio:.3f}")
                if abs(mc_rss_ratio - 1.0) < 0.05:
                    lines.append(f"    The RSS normal assumption adequately represents the combined")
                    lines.append(f"    uncertainty.  Both methods produce consistent bounds.")
                elif mc_rss_ratio > 1.05:
                    lines.append(f"    The MC interval is wider than RSS, indicating the source")
                    lines.append(f"    distributions have heavier tails or skewness than Normal.")
                    lines.append(f"    The MC bounds are recommended for conservative reporting.")
                else:
                    lines.append(f"    The MC interval is tighter than RSS, indicating the source")
                    lines.append(f"    distributions have lighter tails than Normal (e.g., Uniform).")
                    lines.append(f"    The RSS bounds are conservative.")
        else:
            lines.append("    Both RSS and MC results required for comparison.")
        lines.append("")

        # --- 5d. Recommended Accuracy Statement for Report ---
        lines.append("5d. RECOMMENDED CFD ACCURACY STATEMENT:")
        lines.append("    (Copy/adapt the following for the VVUQ report conclusion)")
        lines.append("")
        if rss is not None and rss.computed and rss.n_data == 0:
            lines.append('    "The uncertainty budget has been quantified per ASME V&V 20-2009 (R2021)')
            lines.append(f'     using {len([s for s in (sources or []) if s.enabled])} identified uncertainty sources.')
            lines.append(f'     Expanded uncertainty: U_val = \u00b1{rss.U_val:.4f} [{unit}]')
            lines.append(f'       ({cov_pct} coverage / {conf_pct} confidence, {sided_label})')
            lines.append(f'     RESULT: Validation verdict not available \u2014 no comparison data loaded.')
            lines.append(f'     Load comparison data (E = S \u2212 D) to complete the validation assessment."')
        elif rss is not None and rss.computed and rss.U_val > 0:
            ratio = abs(rss.E_mean) / rss.U_val
            validated = ratio <= 1.0
            lines.append(f'    "The CFD model has been assessed per ASME V&V 20-2009 (R2021)')
            lines.append(f'     using {len([s for s in (sources or []) if s.enabled])} identified uncertainty sources.')
            if mc is not None and mc.computed:
                mc_lo_s = f"P{mc._lower_pct:.4g}" if hasattr(mc, '_lower_pct') else "P5"
                mc_hi_s = f"P{mc._upper_pct:.4g}" if hasattr(mc, '_upper_pct') else "P95"
                mc_cov_s = f"{mc._coverage*100:.0f}%" if hasattr(mc, '_coverage') else cov_pct
                mc_side_s = "one-sided" if getattr(mc, '_one_sided', True) else "two-sided"
                lines.append(f'     RSS expanded uncertainty:  U_val = \u00b1{rss.U_val:.4f} [{unit}]')
                lines.append(f'       ({cov_pct} coverage / {conf_pct} confidence, {sided_label})')
                lines.append(f'     MC prediction interval:   [{mc.pct_5:+.4f}, {mc.pct_95:+.4f}] [{unit}]')
                lines.append(f'       ({mc.n_trials:,} trials, {mc_cov_s} {mc_side_s}, JCGM 101:2008)')
                if hasattr(mc, '_p5_boots') and hasattr(mc, '_p95_boots'):
                    lines.append(f'     MC bootstrap 95% CI:      {mc_lo_s} \u2208 [{float(np.percentile(mc._p5_boots, 2.5)):+.4f}, {float(np.percentile(mc._p5_boots, 97.5)):+.4f}]')
                    lines.append(f'                                {mc_hi_s} \u2208 [{float(np.percentile(mc._p95_boots, 2.5)):+.4f}, {float(np.percentile(mc._p95_boots, 97.5)):+.4f}]')
            else:
                lines.append(f'     Expanded uncertainty: U_val = \u00b1{rss.U_val:.4f} [{unit}]')
                lines.append(f'       ({cov_pct} coverage / {conf_pct} confidence, {sided_label})')
            lines.append(f'     Mean comparison error: \u0112 = {rss.E_mean:+.4f} [{unit}]')
            lines.append(f'     Validation ratio:     |\u0112| / U_val = {ratio:.4f}')
            if validated:
                lines.append(f'     RESULT: The model IS VALIDATED at the stated coverage level."')
            else:
                lines.append(f'     RESULT: The model is NOT VALIDATED at the stated coverage level."')
        else:
            lines.append("    Insufficient data to generate a certification statement.")
        lines.append("")

        # --- Audience-mode additions ---
        audience_idx = (self._cmb_audience.currentIndex()
                        if hasattr(self, '_cmb_audience') else 0)
        # 0 = Internal Engineering (terse — already done above)
        # 1 = External Technical Report
        # 2 = Regulatory Submission

        if audience_idx >= 1 and rss is not None and rss.computed:
            # External: add method references and scope context
            lines.append("=" * 60)
            lines.append("  SCOPE & METHOD CONTEXT")
            lines.append("=" * 60)
            lines.append("")
            n_enabled = len([s for s in (sources or []) if s.enabled])
            lines.append(f"  Validation methodology: ASME V&V 20-2009 (R2021) §4–§6")
            lines.append(f"  Uncertainty combination: Root-Sum-Square (RSS) with Welch-Satterthwaite DOF")
            if mc is not None and mc.computed:
                lines.append(f"  Independent check: Monte Carlo propagation (JCGM 101:2008)")
            lines.append(f"  Number of uncertainty sources: {n_enabled}")
            lines.append(f"  Number of comparison data points: {rss.n_data}")
            if rss.n_data >= 30:
                _ds_label = "adequate"
            elif rss.n_data >= 10:
                _ds_label = "marginal"
            else:
                _ds_label = "insufficient per GUM §G.3"
            lines.append(f"  Data sufficiency: {_ds_label}  (n = {rss.n_data})")
            if rss.has_correlations:
                lines.append(f"  Correlation groups: {', '.join(rss.correlation_groups)}")
                lines.append(f"  Correlation model: Pairwise transitivity ρ(a,b) = ρ_a × ρ_b")
            else:
                lines.append(f"  Source correlations: All sources treated as independent")
            bound_type = getattr(settings, 'bound_type', 'Both (for comparison)')
            lines.append(f"  Bound type: {bound_type}")
            lines.append(f"  Epistemic fraction: {rss.pct_epistemic:.1f}% of total variance")
            lines.append("")

        if audience_idx >= 2 and rss is not None and rss.computed:
            # Regulatory: add full assumptions section and conditional language
            lines.append("=" * 60)
            lines.append("  ASSUMPTIONS & LIMITATIONS")
            lines.append("=" * 60)
            lines.append("")
            lines.append("  The following assumptions apply to this validation assessment:")
            lines.append("  1. Uncertainty sources are assumed to be independently distributed")
            if rss.has_correlations:
                lines.append("     except where explicit correlation groups have been defined.")
            else:
                lines.append("     (no correlation groups defined).")
            lines.append("  2. The RSS method assumes Gaussian combination per GUM §5.1;")
            lines.append("     departures are checked via Monte Carlo comparison (JCGM 101:2008).")
            lines.append("  3. Coverage factors are derived from Student's t-distribution")
            lines.append("     with effective degrees of freedom per Welch-Satterthwaite formula.")
            lines.append("  4. Validation is limited to the tested operating conditions and")
            lines.append("     geometry; extrapolation requires additional assessment.")
            lines.append("")

            # Source basis classification
            _enabled = [s for s in (sources or []) if s.enabled]
            assumed_srcs = [s for s in _enabled
                           if getattr(s, 'basis_type', '') == 'assumed']
            expert_srcs = [s for s in _enabled
                           if getattr(s, 'basis_type', '') == 'expert_judgment']
            high_red = [s for s in _enabled
                        if getattr(s, 'reducibility', '') == 'high']
            if assumed_srcs or expert_srcs or high_red:
                lines.append("  Source Classification Notes:")
                if assumed_srcs:
                    names = ", ".join(s.name for s in assumed_srcs)
                    lines.append(
                        f"    - Assumed uncertainty estimates ({len(assumed_srcs)}): {names}"
                    )
                if expert_srcs:
                    names = ", ".join(s.name for s in expert_srcs)
                    lines.append(
                        f"    - Expert judgment basis ({len(expert_srcs)}): {names}"
                    )
                if high_red:
                    names = ", ".join(s.name for s in high_red)
                    lines.append(
                        f"    - High reducibility ({len(high_red)}): {names}"
                    )
                lines.append("")

            lines.append("  Standards Citations:")
            lines.append("    - ASME V&V 20-2009 (R2021): Standard for Verification and")
            lines.append("      Validation in Computational Fluid Dynamics and Heat Transfer")
            lines.append("    - JCGM 100:2008 (GUM): Evaluation of measurement data — Guide to")
            lines.append("      the expression of uncertainty in measurement")
            lines.append("    - JCGM 101:2008: Evaluation of measurement data — Supplement 1 to")
            lines.append('      the "Guide to the expression of uncertainty in measurement" —')
            lines.append("      Propagation of distributions using a Monte Carlo method")
            lines.append("")

        self._findings_edit.setPlainText("\n".join(lines))

    # =================================================================
    # ASSUMPTIONS AUTO-GENERATION
    # =================================================================
    def _generate_assumptions(self):
        """
        Pull ASSUMPTION and USER_OVERRIDE entries from the global audit_log
        and present them as a structured list.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("  ASSUMPTIONS & ENGINEERING JUDGMENTS")
        lines.append("=" * 60)
        lines.append("")

        # --- Pooling assumption -----------------------------------------
        if self._comp is not None:
            if self._comp.is_pooled:
                lines.append(
                    "\u2022 Pooling: Comparison data POOLED across all "
                    "sensor locations / conditions."
                )
            else:
                lines.append(
                    "\u2022 Pooling: Comparison data treated PER-LOCATION "
                    "(not pooled)."
                )
        lines.append("")

        # --- Distribution choices ---------------------------------------
        lines.append("\u2022 Distribution Choices:")
        for src in self._sources:
            if src.enabled:
                lines.append(
                    f"    - {src.name}: {src.distribution} "
                    f"(input type: {src.input_type})"
                )
        lines.append("")

        # --- k-factor method --------------------------------------------
        if self._settings is not None:
            lines.append(
                f"\u2022 k-factor method: {self._settings.k_method}"
            )
            lines.append(
                f"    Coverage = {self._settings.coverage*100:.0f}%, "
                f"Confidence = {self._settings.confidence*100:.0f}%, "
                f"One-sided = {self._settings.one_sided}"
            )
            lines.append("")

        # --- Audit-log overrides ----------------------------------------
        override_entries = [
            e for e in audit_log.entries
            if e['action'] in ('ASSUMPTION', 'USER_OVERRIDE')
        ]
        if override_entries:
            lines.append("\u2022 Logged Overrides & Assumptions:")
            for entry in override_entries:
                ts = entry['timestamp'][:19].replace('T', ' ')
                lines.append(f"    [{ts}] {entry['description']}")
                if entry['details']:
                    lines.append(f"       {entry['details']}")
            lines.append("")
        else:
            lines.append(
                "\u2022 No user overrides or manual assumptions recorded "
                "in the audit log."
            )
            lines.append("")

        self._assumptions_edit.setPlainText("\n".join(lines))

    # =================================================================
    # COMPARE PROJECTS
    # =================================================================
    def load_comparison_project(self):
        """
        Open a file dialog to load a second .json project file.

        Extracts RSSResults and MCResults from the saved project and adds
        comparison columns to the table.
        """
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Comparison Project",
            "",
            "VV20 Project Files (*.json);;All Files (*)",
        )
        if not path:
            return

        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                proj = json.load(f)
        except Exception as exc:
            QMessageBox.warning(
                self,
                "Load Error",
                f"Could not load project file:\n{exc}",
            )
            return

        # --- Extract results from the loaded project --------------------
        rss_dict = proj.get('rss_results', {})
        mc_dict = proj.get('mc_results', {})
        project_name = proj.get('project_name', os.path.basename(path))

        comp_rss = RSSResults()
        for k, v in rss_dict.items():
            if hasattr(comp_rss, k):
                setattr(comp_rss, k, v)
        comp_rss.computed = bool(rss_dict)

        comp_mc = MCResults()
        for k, v in mc_dict.items():
            if k == 'samples':
                continue
            if hasattr(comp_mc, k):
                setattr(comp_mc, k, v)
        comp_mc.computed = bool(mc_dict)

        self._comparison_rss = comp_rss
        self._comparison_mc = comp_mc
        self._comparison_label = project_name

        self._add_comparison_columns()
        self._comparison_info_label.setText(
            f"Comparison: {project_name}"
        )
        self._comparison_info_label.setStyleSheet(
            f"color: {DARK_COLORS['green']}; font-weight: bold;"
        )
        self._btn_clear_comparison.setEnabled(True)

        audit_log.log(
            "COMPARE_PROJECT",
            f"Loaded comparison project: {project_name}",
            path,
        )

    def _add_comparison_columns(self):
        """
        Append two extra columns for the comparison project's RSS and MC
        results, plus a 'Delta' column showing differences.
        """
        comp_rss = self._comparison_rss
        comp_mc = self._comparison_mc
        label = self._comparison_label

        # Add three columns: Comp RSS, Comp MC, Delta
        base_cols = len(self._BASE_COLUMNS)
        new_col_count = base_cols + 3
        self._table.setColumnCount(new_col_count)
        self._table.setHorizontalHeaderLabels(
            self._BASE_COLUMNS + [
                f"{label} RSS",
                f"{label} MC",
                "Delta (RSS)",
            ]
        )
        for c in range(base_cols, new_col_count):
            self._table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.Stretch
            )

        def _set(row: int, col: int, text: str, color: str = ""):
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            if color:
                item.setForeground(QColor(color))
            self._table.setItem(row, col, item)

        # --- Comparison RSS column (col = base_cols) --------------------
        col_rss = base_cols
        if comp_rss is not None and comp_rss.computed:
            _set(0, col_rss, f"{comp_rss.u_val:.4f}")
            _set(1, col_rss, f"{comp_rss.k_factor:.3f}")
            _set(2, col_rss, f"{comp_rss.U_val:.4f}")
            _set(3, col_rss, f"{comp_rss.lower_bound_uval:.4f}")
            _set(4, col_rss, f"{comp_rss.upper_bound_uval:.4f}")
            _set(5, col_rss, f"{comp_rss.E_mean:.4f}")
            if comp_rss.U_val > 0:
                r_cmp = abs(comp_rss.E_mean) / comp_rss.U_val
                if r_cmp <= 1.0:
                    _set(6, col_rss, f"YES ({r_cmp:.3f})",
                         DARK_COLORS.get('green', '#66bb6a'))
                else:
                    _set(6, col_rss, f"NO ({r_cmp:.3f})",
                         DARK_COLORS.get('red', '#ef5350'))
            else:
                _set(6, col_rss, "\u2014")
            _set(7, col_rss, "No")
            _set(8, col_rss, "Normal")
            _set(9, col_rss, "ASME V&V 20-2009")
        else:
            for r in range(self._N_ROWS):
                _set(r, col_rss, "\u2014")

        # --- Comparison MC column (col = base_cols + 1) -----------------
        col_mc = base_cols + 1
        if comp_mc is not None and comp_mc.computed:
            mc_half = (comp_mc.pct_95 - comp_mc.pct_5) / 2.0
            lo_lbl_c = f"P{comp_mc._lower_pct:.4g}" if hasattr(comp_mc, '_lower_pct') else "P5"
            hi_lbl_c = f"P{comp_mc._upper_pct:.4g}" if hasattr(comp_mc, '_upper_pct') else "P95"
            _set(0, col_mc, f"{comp_mc.combined_std:.4f}")
            _set(1, col_mc, "N/A (distribution-free)")
            _set(2, col_mc, f"\u00b1{mc_half:.4f} ({lo_lbl_c}\u2013{hi_lbl_c})")
            _set(3, col_mc, f"{comp_mc.lower_bound:.4f}")
            _set(4, col_mc, f"{comp_mc.upper_bound:.4f}")
            _set(5, col_mc, f"{comp_mc.combined_mean:.4f}")
            if comp_mc.lower_bound <= 0 <= comp_mc.upper_bound:
                _set(6, col_mc, "YES",
                     DARK_COLORS.get('green', '#66bb6a'))
            else:
                _set(6, col_mc, "NO",
                     DARK_COLORS.get('red', '#ef5350'))
            _set(7, col_mc, "Depends")
            _set(8, col_mc, "Actual")
            _set(9, col_mc, "JCGM 101:2008")
        else:
            for r in range(self._N_ROWS):
                _set(r, col_mc, "\u2014")

        # --- Delta column (col = base_cols + 2) -------------------------
        col_delta = base_cols + 2
        if (
            self._rss is not None and self._rss.computed
            and comp_rss is not None and comp_rss.computed
        ):
            # Row 0: combined sigma delta
            delta_sigma = self._rss.u_val - comp_rss.u_val
            _set(0, col_delta, f"{delta_sigma:+.4f}",
                 DARK_COLORS['green'] if abs(delta_sigma) < 0.01
                 else DARK_COLORS['yellow'])
            # Row 1: k-factor delta
            delta_k = self._rss.k_factor - comp_rss.k_factor
            _set(1, col_delta, f"{delta_k:+.3f}",
                 DARK_COLORS['green'] if abs(delta_k) < 0.01
                 else DARK_COLORS['yellow'])
            # Row 2: U_val delta
            delta_U = self._rss.U_val - comp_rss.U_val
            _set(2, col_delta, f"{delta_U:+.4f}",
                 DARK_COLORS['green'] if abs(delta_U) < 0.01
                 else DARK_COLORS['yellow'])
            # Row 3: lower bound delta (guard NaN from bound_type gating)
            if (not np.isnan(self._rss.lower_bound_uval)
                    and not np.isnan(comp_rss.lower_bound_uval)):
                delta_lb = self._rss.lower_bound_uval - comp_rss.lower_bound_uval
                _set(3, col_delta, f"{delta_lb:+.4f}",
                     DARK_COLORS['green'] if abs(delta_lb) < 0.01
                     else DARK_COLORS['yellow'])
            else:
                _set(3, col_delta, "\u2014")
            # Row 4: upper bound delta (guard NaN from bound_type gating)
            if (not np.isnan(self._rss.upper_bound_uval)
                    and not np.isnan(comp_rss.upper_bound_uval)):
                delta_ub = self._rss.upper_bound_uval - comp_rss.upper_bound_uval
                _set(4, col_delta, f"{delta_ub:+.4f}",
                     DARK_COLORS['green'] if abs(delta_ub) < 0.01
                     else DARK_COLORS['yellow'])
            else:
                _set(4, col_delta, "\u2014")
            # Row 5: mean error delta
            delta_E = self._rss.E_mean - comp_rss.E_mean
            _set(5, col_delta, f"{delta_E:+.4f}",
                 DARK_COLORS['green'] if abs(delta_E) < 0.01
                 else DARK_COLORS['yellow'])
            # Rows 6-9: qualitative — no numeric delta
            for r in range(6, self._N_ROWS):
                _set(r, col_delta, "\u2014")
        else:
            for r in range(self._N_ROWS):
                _set(r, col_delta, "\u2014")

        self._table.resizeColumnsToContents()

    def clear_results(self):
        """Full reset: clear table data, findings, assumptions, and comparison."""
        # Restore the fixed row structure (don't destroy rows — _populate_table
        # relies on rowCount == _N_ROWS for setItem to work).
        self._clear_comparison()
        self._table.setRowCount(self._N_ROWS)
        self._table.setColumnCount(len(self._BASE_COLUMNS))
        self._table.setHorizontalHeaderLabels(self._BASE_COLUMNS)
        # Re-write bold row labels in column 0
        for r, label in enumerate(self._ROW_LABELS):
            item = QTableWidgetItem(label)
            item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self._table.setItem(r, 0, item)
        # Clear data columns (1..N)
        for r in range(self._N_ROWS):
            for c in range(1, len(self._BASE_COLUMNS)):
                self._table.setItem(r, c, QTableWidgetItem(""))
        # Clear text sections
        self._findings_edit.clear()
        self._assumptions_edit.clear()

    def _clear_comparison(self):
        """Remove the comparison columns and reset state."""
        self._comparison_rss = None
        self._comparison_mc = None
        self._comparison_label = ""

        self._table.setColumnCount(len(self._BASE_COLUMNS))
        self._table.setHorizontalHeaderLabels(self._BASE_COLUMNS)
        for c in range(1, len(self._BASE_COLUMNS)):
            self._table.horizontalHeader().setSectionResizeMode(
                c, QHeaderView.Stretch
            )
        self._comparison_info_label.setText("No comparison project loaded.")
        self._comparison_info_label.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']};"
        )
        self._btn_clear_comparison.setEnabled(False)

    # =================================================================
    # EXPORT HELPERS
    # =================================================================
    def _export_to_clipboard(self):
        """Copy the comparison table as tab-separated text to clipboard."""
        n_rows = self._table.rowCount()
        n_cols = self._table.columnCount()

        # Header row
        headers = []
        for c in range(n_cols):
            h = self._table.horizontalHeaderItem(c)
            headers.append(h.text() if h else "")
        lines = ["\t".join(headers)]

        # Data rows
        for r in range(n_rows):
            row_data = []
            for c in range(n_cols):
                item = self._table.item(r, c)
                row_data.append(item.text() if item else "")
            lines.append("\t".join(row_data))

        # Append findings and assumptions
        lines.append("")
        lines.append(self._findings_edit.toPlainText())
        lines.append("")
        lines.append(self._assumptions_edit.toPlainText())

        clipboard = QApplication.clipboard()
        clipboard.setText("\n".join(lines))

        QMessageBox.information(
            self,
            "Exported",
            "Roll-up table and findings copied to clipboard.",
        )


# =============================================================================
# SECTION 13: TAB 7 (REFERENCE TAB)
# =============================================================================

class ReferenceTab(QWidget):
    """
    Built-in documentation / reference tab with sub-tabs covering:
      1. V&V 20 Framework Overview
      2. k-Factor Reference Tables
      3. Welch-Satterthwaite Explained
      4. Distribution Guide
      5. Distribution-Free (Non-Parametric) Bounds
      6. Monte Carlo Method
      7. Glossary
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------
    @staticmethod
    def _html_style():
        """Return a CSS block that works with the dark theme."""
        c = DARK_COLORS
        return f"""
        <style>
            body {{
                background-color: {c['bg']};
                color: {c['fg']};
                font-family: 'Segoe UI', Calibri, Arial, sans-serif;
                font-size: 13px;
                line-height: 1.5;
                margin: 12px;
            }}
            h1 {{
                color: {c['accent']};
                font-size: 20px;
                border-bottom: 2px solid {c['accent']};
                padding-bottom: 4px;
                margin-top: 8px;
            }}
            h2 {{
                color: {c['accent_hover']};
                font-size: 16px;
                margin-top: 18px;
            }}
            h3 {{
                color: {c['fg_bright']};
                font-size: 14px;
                margin-top: 14px;
            }}
            p, li {{
                color: {c['fg']};
            }}
            code {{
                background-color: {c['bg_input']};
                color: {c['orange']};
                padding: 1px 5px;
                border-radius: 3px;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
            }}
            pre {{
                background-color: {c['bg_input']};
                color: {c['fg']};
                padding: 10px;
                border-radius: 6px;
                border: 1px solid {c['border']};
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                overflow-x: auto;
            }}
            table {{
                border-collapse: collapse;
                margin: 10px 0;
                width: auto;
            }}
            th {{
                background-color: {c['bg_widget']};
                color: {c['accent']};
                padding: 6px 12px;
                border: 1px solid {c['border']};
                text-align: center;
            }}
            td {{
                background-color: {c['bg_alt']};
                color: {c['fg']};
                padding: 5px 12px;
                border: 1px solid {c['border']};
                text-align: center;
            }}
            .cite {{
                color: {c['fg_dim']};
                font-style: italic;
                font-size: 11px;
                margin-top: 6px;
            }}
            .formula {{
                background-color: {c['bg_input']};
                color: {c['yellow']};
                padding: 8px 12px;
                border-left: 3px solid {c['accent']};
                margin: 10px 0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 13px;
            }}
            .note {{
                background-color: {c['bg_widget']};
                border-left: 3px solid {c['yellow']};
                padding: 8px 12px;
                margin: 10px 0;
                color: {c['fg']};
            }}
            .warn {{
                background-color: {c['bg_widget']};
                border-left: 3px solid {c['red']};
                padding: 8px 12px;
                margin: 10px 0;
                color: {c['fg']};
            }}
            ul {{ margin-left: 16px; }}
            a {{ color: {c['link']}; }}
        </style>
        """

    def _make_text_browser(self, html_content):
        """Create a read-only QTextEdit inside a QScrollArea."""
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setHtml(self._html_style() + html_content)
        text_edit.setStyleSheet(
            f"QTextEdit {{ background-color: {DARK_COLORS['bg']}; "
            f"border: none; color: {DARK_COLORS['fg']}; }}"
        )

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(text_edit)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background-color: {DARK_COLORS['bg']}; }}"
        )
        return scroll

    # -----------------------------------------------------------------
    # Main UI
    # -----------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        tabs = QTabWidget()
        tabs.setTabPosition(QTabWidget.TabPosition.North)

        tabs.addTab(self._build_vv20_overview_tab(), "V&&V 20 Overview")
        tabs.addTab(self._build_k_factor_tab(), "k-Factor Tables")
        tabs.addTab(self._build_welch_satterthwaite_tab(), "Welch-Satterthwaite")
        tabs.addTab(self._build_distribution_guide_tab(), "Distribution Guide")
        tabs.addTab(self._build_uncertainty_classification_tab(),
                     "Uncertainty Classification")
        tabs.addTab(self._build_distribution_free_tab(), "Distribution-Free Bounds")
        tabs.addTab(self._build_monte_carlo_tab(), "Monte Carlo Method")
        tabs.addTab(self._build_glossary_tab(), "Glossary")

        layout.addWidget(tabs)

    # =================================================================
    # Sub-tab 1: V&V 20 Framework Overview
    # =================================================================
    def _build_vv20_overview_tab(self):
        html = """
        <h1>ASME V&amp;V 20 Framework Overview</h1>

        <h2>Validation Comparison Error</h2>
        <p>The fundamental quantity in ASME V&amp;V 20 is the <b>comparison error</b>,
        defined as the difference between the simulation result (<i>S</i>) and the
        experimental data (<i>D</i>):</p>

        <div class="formula">E &nbsp;=&nbsp; S &minus; D</div>

        <p>This seemingly simple equation is the foundation of the entire validation
        framework. <i>E</i> captures the total discrepancy between what the model
        predicts and what was measured.</p>

        <h3>Text-Based Diagram</h3>
<pre>
   Simulation (S)         Experiment (D)
        |                       |
        +--- Numerical Unc.     +--- Measurement Unc.
        |    (u_num)            |    (u_D)
        +--- Input Unc.         |
             (u_input)          |
                                |
        S ==================== D
              |--- E ---|
              comparison error

    Validation Assessment:
    ----------------------
    |E|  vs  U_val = k * u_val

    where  u_val = sqrt( u_num^2 + u_input^2 + u_D^2 )
</pre>

        <h2>Validation Uncertainty</h2>
        <p>The <b>validation uncertainty</b> combines all sources via root-sum-square
        (RSS):</p>

        <div class="formula">u_val = &radic;( u_num&sup2; + u_input&sup2; + u_D&sup2; )</div>

        <p>Where:</p>
        <ul>
            <li><b>u_num</b> &mdash; Numerical uncertainty (discretisation, iteration,
                round-off, etc.)</li>
            <li><b>u_input</b> &mdash; Input / parameter uncertainty (boundary conditions,
                material properties, etc.)</li>
            <li><b>u_D</b> &mdash; Experimental / data uncertainty (measurement chain,
                calibration, repeatability, etc.)</li>
        </ul>

        <h2>Expanded Validation Uncertainty</h2>
        <p>The expanded validation uncertainty at a chosen coverage level is:</p>
        <div class="formula">U_val = k &middot; u_val</div>
        <p>where <i>k</i> is the coverage factor determined from the effective degrees
        of freedom (via Welch-Satterthwaite) and the desired coverage probability.</p>

        <h2>Validation Assessment</h2>
        <p>The comparison error is compared against the expanded validation uncertainty:</p>

        <table>
            <tr>
                <th>Condition</th>
                <th>Interpretation</th>
            </tr>
            <tr>
                <td><code>|E| &le; U_val</code></td>
                <td><b>Validation is achieved</b> at the stated coverage level.
                The model agrees with the experiment within the combined uncertainty.</td>
            </tr>
            <tr>
                <td><code>|E| &gt; U_val</code></td>
                <td><b>Validation is NOT achieved.</b> The discrepancy exceeds what
                uncertainty alone can explain. Model deficiency is indicated.</td>
            </tr>
        </table>

        <h2>When |E| Exceeds U_val</h2>
        <p>If the comparison error exceeds the validation uncertainty, V&amp;V 20
        recommends the following actions:</p>
        <ol>
            <li><b>Examine the model</b> &mdash; Identify potential sources of model
                deficiency (missing physics, incorrect assumptions, etc.).</li>
            <li><b>Review uncertainty sources</b> &mdash; Ensure all significant
                uncertainty sources have been identified and properly quantified.</li>
            <li><b>Refine the experiment</b> &mdash; Consider whether the experimental
                setup or measurement procedure contributes to the discrepancy.</li>
            <li><b>Improve the model</b> &mdash; Use the sign and magnitude of E as
                diagnostic guidance for model improvement.</li>
            <li><b>Document the model deficiency</b> &mdash; Report the model
                deficiency for the specific validation case; do not simply increase
                uncertainty to force |E| &le; U_val.</li>
        </ol>

        <div class="warn">
            <b>Warning:</b> It is not appropriate to artificially inflate uncertainty
            estimates to make the validation assessment pass. The purpose of V&amp;V 20
            is to provide an honest assessment of model accuracy.
        </div>

        <p class="cite">Ref: ASME V&amp;V 20-2009 (R2021), Sections 2, 3, and 9.<br>
        Ref: AIAA G-077-1998, Section 5.</p>
        """
        return self._make_text_browser(html)

    # =================================================================
    # Sub-tab 2: k-Factor Reference Tables
    # =================================================================
    def _build_k_factor_tab(self):
        container = QWidget()
        outer_layout = QVBoxLayout(container)
        outer_layout.setContentsMargins(6, 6, 6, 6)

        # --- Interactive calculator ---
        calc_group = QGroupBox("Interactive k-Factor Calculator")
        calc_layout = QHBoxLayout(calc_group)

        calc_layout.addWidget(QLabel("Sample size n:"))
        self._kref_n_spin = QSpinBox()
        self._kref_n_spin.setRange(2, 100000)
        self._kref_n_spin.setValue(30)
        calc_layout.addWidget(self._kref_n_spin)

        calc_layout.addWidget(QLabel("Coverage %:"))
        self._kref_cov_combo = QComboBox()
        self._kref_cov_combo.addItems(["90", "95", "99"])
        self._kref_cov_combo.setCurrentText("95")
        calc_layout.addWidget(self._kref_cov_combo)

        calc_layout.addWidget(QLabel("Confidence %:"))
        self._kref_conf_combo = QComboBox()
        self._kref_conf_combo.addItems(["90", "95", "99"])
        self._kref_conf_combo.setCurrentText("95")
        calc_layout.addWidget(self._kref_conf_combo)

        calc_btn = QPushButton("Compute k")
        calc_btn.clicked.connect(self._compute_interactive_k)
        calc_layout.addWidget(calc_btn)

        self._kref_result_label = QLabel("k = ...")
        self._kref_result_label.setStyleSheet(
            f"color: {DARK_COLORS['yellow']}; font-weight: bold; font-size: 14px;"
        )
        calc_layout.addWidget(self._kref_result_label)
        calc_layout.addStretch()

        outer_layout.addWidget(calc_group)

        # --- Precomputed tables ---
        tables_data = generate_k_factor_table()

        table_tabs = QTabWidget()
        for sided_label in ['one-sided', 'two-sided']:
            sided_widget = QWidget()
            sided_layout = QVBoxLayout(sided_widget)
            sided_layout.setContentsMargins(4, 4, 4, 4)

            for combo_key, rows in tables_data[sided_label].items():
                grp = QGroupBox(f"{sided_label.title()} \u2014 {combo_key}")
                grp_layout = QVBoxLayout(grp)

                tw = QTableWidget()
                tw.setRowCount(len(rows))
                tw.setColumnCount(2)
                tw.setHorizontalHeaderLabels(["n", "k"])
                style_table(tw, column_widths={0: 80, 1: 100}, stretch_col=1)
                tw.verticalHeader().setVisible(False)
                tw.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
                tw.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

                for r_idx, (n_val, k_val) in enumerate(rows):
                    n_item = QTableWidgetItem(str(n_val))
                    n_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    tw.setItem(r_idx, 0, n_item)

                    k_item = QTableWidgetItem(f"{k_val:.4f}")
                    k_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    tw.setItem(r_idx, 1, k_item)

                tw.resizeColumnsToContents()
                tw.setMaximumHeight(min(38 * (len(rows) + 1), 550))
                grp_layout.addWidget(tw)
                sided_layout.addWidget(grp)

            sided_layout.addStretch()

            side_scroll = QScrollArea()
            side_scroll.setWidgetResizable(True)
            side_scroll.setWidget(sided_widget)

            display_name = "One-Sided" if sided_label == "one-sided" else "Two-Sided"
            table_tabs.addTab(side_scroll, display_name)

        outer_layout.addWidget(table_tabs)

        # Citation
        cite = QLabel(
            "One-sided factors from non-central t-distribution.  "
            "Two-sided from chi-squared approximation.  [GUM Annex G]"
        )
        cite.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; font-style: italic; font-size: 11px;"
        )
        cite.setWordWrap(True)
        outer_layout.addWidget(cite)

        # Wrap in scroll
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(container)
        scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background-color: {DARK_COLORS['bg']}; }}"
        )
        return scroll

    @Slot()
    def _compute_interactive_k(self):
        n = self._kref_n_spin.value()
        cov = int(self._kref_cov_combo.currentText()) / 100.0
        conf = int(self._kref_conf_combo.currentText()) / 100.0

        k_one = one_sided_tolerance_k(n, cov, conf)
        k_two = two_sided_tolerance_k(n, cov, conf)

        self._kref_result_label.setText(
            f"One-sided k = {k_one:.4f}   |   Two-sided k = {k_two:.4f}"
        )

    # =================================================================
    # Sub-tab 3: Welch-Satterthwaite Explained
    # =================================================================
    def _build_welch_satterthwaite_tab(self):
        html = """
        <h1>Welch-Satterthwaite Equation</h1>

        <h2>Formula</h2>
        <p>When combining multiple uncertainty sources that may each have different
        degrees of freedom, the <b>effective degrees of freedom</b>
        (&nu;<sub>eff</sub>) of the combined standard uncertainty is estimated
        using the Welch-Satterthwaite equation:</p>

        <div class="formula">
        &nu;<sub>eff</sub> &nbsp;=&nbsp;
        u<sub>c</sub><sup>4</sup> &nbsp;/&nbsp;
        &Sigma;<sub>i</sub> ( u<sub>i</sub><sup>4</sup> / &nu;<sub>i</sub> )
        </div>

        <p>where:</p>
        <ul>
            <li><b>u<sub>c</sub></b> = combined standard uncertainty =
                &radic;(&Sigma; u<sub>i</sub>&sup2;)</li>
            <li><b>u<sub>i</sub></b> = standard uncertainty of source <i>i</i></li>
            <li><b>&nu;<sub>i</sub></b> = degrees of freedom of source <i>i</i></li>
        </ul>

        <h2>Physical Meaning</h2>
        <p>&nu;<sub>eff</sub> tells you <b>how well you know the combined
        uncertainty itself</b>. A small &nu;<sub>eff</sub> means you have limited
        information about the true uncertainty (few samples), so you need a larger
        coverage factor <i>k</i> to maintain the same confidence. A large
        &nu;<sub>eff</sub> (say &gt; 30) means you know the uncertainty quite well,
        and <i>k</i> approaches the Normal-distribution value.</p>

        <table>
            <tr><th>&nu;<sub>eff</sub></th><th>Interpretation</th><th>k (95%, one-sided)</th></tr>
            <tr><td>5</td><td>Very limited knowledge</td><td>2.015</td></tr>
            <tr><td>10</td><td>Moderate knowledge</td><td>1.812</td></tr>
            <tr><td>30</td><td>Good knowledge</td><td>1.697</td></tr>
            <tr><td>100</td><td>Excellent knowledge</td><td>1.660</td></tr>
            <tr><td>&infin;</td><td>Perfect knowledge</td><td>1.645</td></tr>
        </table>

        <h2>How Supplier / Type B Sources Are Handled</h2>
        <p>Type B sources (e.g. manufacturer specifications, engineering judgement)
        are typically assigned <b>&nu; = &infin;</b>, meaning the stated uncertainty
        is considered exact (no additional uncertainty about the uncertainty).
        In the Welch-Satterthwaite sum, these terms have
        u<sub>i</sub><sup>4</sup>/&infin; = 0 and simply drop out of the
        denominator. This means Type B sources <b>do not reduce</b>
        &nu;<sub>eff</sub>.</p>

        <div class="note">
            <b>Note:</b> If <i>all</i> sources have &nu; = &infin; (all Type B),
            then &nu;<sub>eff</sub> = &infin; and the coverage factor is simply the
            Normal-distribution quantile.
        </div>

        <h2>Worked Example</h2>
        <p>Suppose we have three uncertainty sources:</p>
        <table>
            <tr><th>Source</th><th>u<sub>i</sub></th><th>&nu;<sub>i</sub></th></tr>
            <tr><td>Grid convergence (Type A, 3 grids)</td><td>2.5 &deg;F</td><td>2</td></tr>
            <tr><td>TC calibration (Type B, supplier)</td><td>1.0 &deg;F</td><td>&infin;</td></tr>
            <tr><td>Repeatability (Type A, 10 runs)</td><td>1.5 &deg;F</td><td>9</td></tr>
        </table>

        <p><b>Step 1:</b> Combined uncertainty</p>
        <div class="formula">u<sub>c</sub> = &radic;(2.5&sup2; + 1.0&sup2; + 1.5&sup2;)
        = &radic;(6.25 + 1.00 + 2.25) = &radic;9.50 = 3.082 &deg;F</div>

        <p><b>Step 2:</b> Welch-Satterthwaite denominator (supplier term drops out)</p>
        <div class="formula">
        &Sigma; u<sub>i</sub><sup>4</sup>/&nu;<sub>i</sub>
        = 2.5<sup>4</sup>/2 + 1.5<sup>4</sup>/9
        = 39.0625/2 + 5.0625/9
        = 19.531 + 0.563 = 20.094
        </div>

        <p><b>Step 3:</b> Effective degrees of freedom</p>
        <div class="formula">&nu;<sub>eff</sub> = 3.082<sup>4</sup> / 20.094
        = 90.25 / 20.094 &approx; 4.49</div>

        <p><b>Step 4:</b> Coverage factor for 95% one-sided, &nu;<sub>eff</sub> &approx; 4.5</p>
        <div class="formula">k &approx; t<sub>0.95, 4.5</sub> &approx; 2.05</div>

        <p><b>Result:</b></p>
        <div class="formula">U<sub>val</sub> = k &times; u<sub>c</sub>
        = 2.05 &times; 3.082 = 6.32 &deg;F</div>

        <p class="cite">Ref: JCGM 100:2008 (GUM), Annex G, Section G.4.1,
        Equation G.2b.</p>
        """
        return self._make_text_browser(html)

    # =================================================================
    # Sub-tab 4: Distribution Guide
    # =================================================================
    def _build_distribution_guide_tab(self):
        html = """
        <h1>Distribution Guide</h1>

        <h2>Common Distributions and Their Properties</h2>
        <p>The choice of assumed distribution for an uncertainty source affects the
        coverage factor. Below are reference properties and one-sided 95%
        tolerance/coverage factors for common distributions.</p>

        <table>
            <tr>
                <th>Distribution</th>
                <th>k (95% one-sided)</th>
                <th>Shape</th>
                <th>Typical Use</th>
            </tr>
            <tr>
                <td>Normal (Gaussian)</td>
                <td>1.645</td>
                <td>Symmetric, bell-shaped</td>
                <td>Default for Type A data with many samples; central limit theorem applies</td>
            </tr>
            <tr>
                <td>Uniform (Rectangular)</td>
                <td>1.559</td>
                <td>Flat, bounded</td>
                <td>Type B with known bounds but no information about shape
                    (e.g. digitisation, manufacturer tolerance band)</td>
            </tr>
            <tr>
                <td>Triangular</td>
                <td>1.675</td>
                <td>Symmetric triangle</td>
                <td>Type B when values near the centre are more likely than at the
                    bounds (e.g. calibration data with known midpoint)</td>
            </tr>
            <tr>
                <td>Logistic</td>
                <td>1.831</td>
                <td>Symmetric, heavier tails than Normal</td>
                <td>Growth / threshold phenomena; slightly heavier tails</td>
            </tr>
            <tr>
                <td>Laplace (Double Exponential)</td>
                <td>1.862</td>
                <td>Symmetric, heavy tails</td>
                <td>Data with more outliers than a Normal; robustness studies</td>
            </tr>
            <tr>
                <td>Student-t (df = 5)</td>
                <td>&mdash;</td>
                <td>Symmetric, heavy tails</td>
                <td>Small-sample Type A data; captures higher tail probability</td>
            </tr>
            <tr>
                <td>Student-t (df = 10)</td>
                <td>&mdash;</td>
                <td>Symmetric, moderate tails</td>
                <td>Moderate-sample Type A data; tails still heavier than Normal</td>
            </tr>
            <tr>
                <td>Lognormal</td>
                <td>&mdash;</td>
                <td>Right-skewed</td>
                <td>Quantities that are always positive and span orders of magnitude
                    (e.g. fatigue life, surface roughness)</td>
            </tr>
            <tr>
                <td>Weibull</td>
                <td>&mdash;</td>
                <td>Flexible shape</td>
                <td>Reliability / failure data; shape parameter controls tail
                    behaviour</td>
            </tr>
            <tr>
                <td>Exponential</td>
                <td>&mdash;</td>
                <td>One-sided, right-skewed</td>
                <td>Time-to-event / memoryless processes; always &ge; 0</td>
            </tr>
        </table>

        <h2>When the Normal Assumption Is Conservative</h2>
        <p>The Normal distribution is <b>conservative</b> (over-estimates the tail
        probability) when the true distribution has <b>lighter tails</b> than Normal.
        This includes:</p>
        <ul>
            <li>Uniform (bounded &mdash; no tails at all)</li>
            <li>Triangular (bounded with tapering density)</li>
            <li>Beta distributions with shape parameters &gt; 1</li>
        </ul>

        <h2>When the Normal Assumption Is Non-Conservative</h2>
        <p>The Normal distribution is <b>non-conservative</b> (under-estimates the
        tail probability) when the true distribution has <b>heavier tails</b>:</p>
        <ul>
            <li>Student-t with small df (especially df &lt; 10)</li>
            <li>Logistic</li>
            <li>Laplace</li>
            <li>Lognormal (extreme right tail)</li>
            <li>Contaminated or mixture distributions</li>
        </ul>

        <h2>Decision Guidance</h2>
        <div class="note">
        <b>Rule of thumb:</b>
        <ul>
            <li>If you have enough data, use a normality test (Shapiro-Wilk, Anderson-Darling)
                and choose accordingly.</li>
            <li>If you have very few data points (&lt; 10), consider using a Student-t
                with n-1 degrees of freedom rather than Normal.</li>
            <li>If you only have bounds (e.g. &plusmn; tolerance), use Uniform unless you
                have reason to believe values cluster at the centre (use Triangular).</li>
            <li>If the quantity is strictly positive and right-skewed, consider
                Lognormal or Weibull.</li>
            <li>When in doubt, the Normal assumption is a reasonable default for RSS,
                but always note the assumption in your documentation.</li>
        </ul>
        </div>

        <p class="cite">Ref: JCGM 100:2008 (GUM), Section 4.3 and 4.4.<br>
        Ref: ASME PTC 19.1-2018, Annex C.</p>
        """
        return self._make_text_browser(html)

    # =================================================================
    # Sub-tab 5: Uncertainty Classification Guide
    # =================================================================
    def _build_uncertainty_classification_tab(self):
        c = DARK_COLORS
        html = f"""
        <h1>Aleatory vs Epistemic Uncertainty</h1>

        <h2>Definitions</h2>
        <table>
        <tr>
            <th style="text-align:left; width:120px;">Class</th>
            <th style="text-align:left;">Description</th>
            <th style="text-align:left; width:140px;">Also Known As</th>
            <th style="text-align:left; width:100px;">Reducible?</th>
        </tr>
        <tr>
            <td style="text-align:left; color:{c['green']}; font-weight:bold;">
                Aleatory</td>
            <td style="text-align:left;">Inherent randomness in the physical
                process. Cannot be reduced by collecting more data or
                improving models.</td>
            <td style="text-align:left;">Stochastic variability,
                irreducible floor</td>
            <td style="text-align:left; color:{c['red']};">No</td>
        </tr>
        <tr>
            <td style="text-align:left; color:{c['yellow']}; font-weight:bold;">
                Epistemic</td>
            <td style="text-align:left;">Due to lack of knowledge. Can be
                reduced with better data, finer grids, or improved models.
            </td>
            <td style="text-align:left;">Knowledge/model uncertainty,
                reducible bias</td>
            <td style="text-align:left; color:{c['green']};">Yes</td>
        </tr>
        </table>

        <div class="note">
        <strong>Key distinction:</strong> Aleatory uncertainty reflects
        what we <em>cannot</em> know (nature is random); epistemic reflects
        what we <em>do not yet</em> know (knowledge gaps we can close).
        </div>
        <div class="note">
        <strong>Important:</strong> GUM Type A / Type B labels describe how
        uncertainty is quantified (data vs non-statistical information). They
        are not the same as aleatory/epistemic classes.
        </div>

        <h2>Practical Examples for CFD Validation</h2>
        <table>
        <tr>
            <th style="text-align:left;">Uncertainty Source</th>
            <th>Class</th>
            <th style="text-align:left;">Rationale</th>
            <th>Reducible?</th>
        </tr>
        <tr>
            <td style="text-align:left;">Iterative convergence scatter</td>
            <td style="color:{c['green']};">Aleatory</td>
            <td style="text-align:left;">Inherent solver noise at a given
                residual tolerance</td>
            <td>Low</td>
        </tr>
        <tr>
            <td style="text-align:left;">Discretization error (GCI u_num)</td>
            <td style="color:{c['orange']};">Mixed</td>
            <td style="text-align:left;">Usually epistemic while refining;
                near asymptotic/grid-independent behavior the residual acts
                like a low-reducibility floor</td>
            <td>Medium</td>
        </tr>
        <tr>
            <td style="text-align:left;">Thermocouple measurement error</td>
            <td style="color:{c['orange']};">Mixed</td>
            <td style="text-align:left;">Repeatability/noise is aleatory;
                calibration and installation bias are epistemic</td>
            <td>Medium</td>
        </tr>
        <tr>
            <td style="text-align:left;">Inlet mass flow uncertainty</td>
            <td style="color:{c['orange']};">Mixed</td>
            <td style="text-align:left;">Often a combination of sensor noise
                (aleatory) and calibration/setup bias (epistemic). Split if
                possible; otherwise classify mixed.</td>
            <td>Medium</td>
        </tr>
        <tr>
            <td style="text-align:left;">Over-refined/asymptotic mesh residual</td>
            <td style="color:{c['orange']};">Mixed</td>
            <td style="text-align:left;">Model/discretization origin is
                epistemic, but once further refinement gives negligible
                improvement the remaining floor has low reducibility</td>
            <td>Low</td>
        </tr>
        <tr>
            <td style="text-align:left;">Turbulence model form error</td>
            <td style="color:{c['yellow']};">Epistemic</td>
            <td style="text-align:left;">Model limitation; use LES/DNS or
                model-form uncertainty methods</td>
            <td>High</td>
        </tr>
        <tr>
            <td style="text-align:left;">Material property uncertainty</td>
            <td style="color:{c['orange']};">Mixed</td>
            <td style="text-align:left;">Published tolerance bands (aleatory)
                plus limited characterization (epistemic)</td>
            <td>Medium</td>
        </tr>
        <tr>
            <td style="text-align:left;">Geometry tolerance</td>
            <td style="color:{c['yellow']};">Epistemic</td>
            <td style="text-align:left;">Manufacturing variation; reducible
                with tighter specs or as-built measurement</td>
            <td>Medium</td>
        </tr>
        <tr>
            <td style="text-align:left;">Boundary condition sensitivity</td>
            <td style="color:{c['orange']};">Mixed</td>
            <td style="text-align:left;">Both measurement uncertainty
                (epistemic) and physical variability (aleatory)</td>
            <td>Medium</td>
        </tr>
        </table>

        <div class="note">
        <strong>Decision rule for common edge cases:</strong><br>
        1) <em>Inlet mass flow from sensors</em>: if random repeatability/noise
        dominates, treat mostly aleatoric; if calibration/setup bias dominates,
        treat mostly epistemic; if both are present and not separated, use mixed.<br>
        2) <em>Over-refined mesh</em>: if additional refinement no longer reduces
        the numerical term, keep the source as mixed with low reducibility and
        document the asymptotic evidence (do not force it to purely epistemic).
        </div>

        <h2>Uncertainty Combination Diagram</h2>
        <p>The V&amp;V 20 validation uncertainty combines multiple sources
        via root-sum-square (RSS):</p>
        <pre>
  Numerical (usually epistemic, sometimes mixed near asymptotic floor)
  ├── u_num  (discretization / GCI)  ──┐
  └── u_iter (iterative convergence)   │
                                       ├──► u_val = sqrt(u_num² + u_input² + u_D²)
  Input (mixed)                        │
  ├── u_input (boundary conditions)  ──┤       ├── U_E  (epistemic component)
  └── u_param (material properties)    │       ├── U_A  (aleatory component)
                                       │       └── Class split for interpretation
  Experimental (aleatory)              │
  ├── u_D   (measurement data)      ──┘
  └── u_cal (calibration)
        </pre>
        <div class="note">
        <strong>Classification split:</strong> When building the uncertainty
        budget in the Sources tab, marking each source as primarily
        "aleatory" or "epistemic" enables class-split analysis. The
        RSS Results tab shows the total broken down by class. This supports
        prioritization; it does not remove any enabled source from the total
        u_val combination.
        </div>

        <h2>One-Sided (Asymmetric) Uncertainty</h2>
        <p>Sometimes a sensitivity study yields different magnitudes in the
        positive and negative directions. For example, a +10&deg;F inlet
        temperature perturbation causes +5 PSI change in exit pressure,
        while &minus;10&deg;F causes only &minus;3 PSI change.</p>

        <table>
        <tr>
            <th style="text-align:left;">Situation</th>
            <th style="text-align:left;">Recommended Approach</th>
        </tr>
        <tr>
            <td style="text-align:left;">Both directions tested,
                different magnitudes</td>
            <td style="text-align:left;">Use asymmetric &sigma;&#8314;
                / &sigma;&#8315; fields.
                Effective &sigma; = &radic;((&sigma;&#8314;&sup2; +
                &sigma;&#8315;&sup2;) / 2) per GUM &sect;4.3.8</td>
        </tr>
        <tr>
            <td style="text-align:left;">Only one direction tested</td>
            <td style="text-align:left;">Mirror assumption: assume
                &sigma;_missing = &sigma;_observed. Flag with note in
                evidence. Conservative but standard practice.</td>
        </tr>
        <tr>
            <td style="text-align:left;">Physical bound exists
                (e.g., temperature &gt; 0 K)</td>
            <td style="text-align:left;">Use bounded distribution (e.g.,
                truncated normal, log-normal). Note physical justification
                in evidence.</td>
        </tr>
        </table>

        <div class="cite">References: JCGM 100:2008 (GUM) &sect;4.3.8;
        Barlow, R. (2004) "Asymmetric Statistical Errors";
        ASME V&amp;V 20-2009 &sect;4.3; AIAA G-077-1998 &sect;4.2</div>

        <h2>Why It Matters</h2>
        <table>
        <tr>
            <th style="text-align:left;">Budget Dominance</th>
            <th style="text-align:left;">Interpretation</th>
            <th style="text-align:left;">Action</th>
        </tr>
        <tr>
            <td style="text-align:left; color:{c['yellow']};">
                Epistemic-dominant</td>
            <td style="text-align:left;">The analysis can be improved.
                Knowledge gaps are the main contributor.</td>
            <td style="text-align:left;">Refine grid, improve BCs, use
                better turbulence model, collect more data</td>
        </tr>
        <tr>
            <td style="text-align:left; color:{c['green']};">
                Aleatory-dominant</td>
            <td style="text-align:left;">The analysis is near the
                irreducible floor. Physical randomness dominates.</td>
            <td style="text-align:left;">Report result with confidence
                interval; no further model improvement will help</td>
        </tr>
        <tr>
            <td style="text-align:left; color:{c['orange']};">
                Balanced</td>
            <td style="text-align:left;">Both classes contribute
                meaningfully.</td>
            <td style="text-align:left;">Prioritize reducing the largest
                individual source regardless of class</td>
        </tr>
        </table>
        """
        return self._make_text_browser(html)

    # =================================================================
    # Sub-tab 6: Distribution-Free (Non-Parametric) Bounds
    # =================================================================
    def _build_distribution_free_tab(self):
        # Build the minimum-n table dynamically
        coverages = [0.90, 0.95, 0.99]
        confidences = [0.90, 0.95, 0.99]
        orders = [1, 2, 3]

        table_rows = ""
        for cov in coverages:
            for conf in confidences:
                for r in orders:
                    n_min = min_n_distribution_free(cov, conf, r)
                    n_str = str(n_min) if n_min is not None else "&gt;10000"
                    table_rows += (
                        f"<tr>"
                        f"<td>{int(cov*100)}%</td>"
                        f"<td>{int(conf*100)}%</td>"
                        f"<td>{r}</td>"
                        f"<td><b>{n_str}</b></td>"
                        f"</tr>\n"
                    )

        html = f"""
        <h1>Distribution-Free (Non-Parametric) Tolerance Bounds</h1>

        <h2>Overview</h2>
        <p>When the underlying distribution of the data is <b>unknown</b> or clearly
        <b>non-Normal</b>, parametric tolerance intervals (which assume a specific
        distribution) may be invalid. Distribution-free methods use only the
        <b>order statistics</b> (sorted data values) to construct bounds that are
        valid for <i>any</i> continuous distribution.</p>

        <h2>Minimum Sample Size Requirements</h2>
        <p>The table below shows the minimum sample size <b>n</b> required so that
        the <i>r</i>-th order statistic from each tail provides at least the stated
        coverage at the stated confidence level.</p>

        <table>
            <tr>
                <th>Coverage</th>
                <th>Confidence</th>
                <th>Order <i>r</i></th>
                <th>Min <i>n</i></th>
            </tr>
            {table_rows}
        </table>

        <div class="note">
            <b>Interpretation of order <i>r</i>:</b>
            <ul>
                <li><i>r</i> = 1 : use the sample minimum/maximum as the bound</li>
                <li><i>r</i> = 2 : use the 2nd-smallest / 2nd-largest value</li>
                <li><i>r</i> = 3 : use the 3rd-smallest / 3rd-largest value</li>
            </ul>
            Higher <i>r</i> gives a more robust bound but requires more data.
        </div>

        <h2>When To Use Distribution-Free Bounds</h2>
        <ul>
            <li>The data clearly fails normality tests (Shapiro-Wilk p &lt; 0.05,
                Anderson-Darling rejects, etc.)</li>
            <li>The data is multi-modal, heavily skewed, or has outliers</li>
            <li>You have enough data to meet the minimum sample-size requirement above</li>
            <li>A fully conservative, assumption-free bound is desired</li>
        </ul>

        <h2>When Distribution-Free Bounds Cannot Be Used</h2>
        <ul>
            <li>Sample size is too small (does not meet the minimum <i>n</i>)</li>
            <li>You need a tighter (less conservative) interval &mdash; parametric
                methods will generally give narrower bounds when their assumptions
                hold</li>
        </ul>

        <h2>How Order Statistics Provide Bounds</h2>
        <p>Given <i>n</i> observations sorted as X<sub>(1)</sub> &le; X<sub>(2)</sub>
        &le; &hellip; &le; X<sub>(n)</sub>, the probability that the interval
        [X<sub>(r)</sub>, X<sub>(n-r+1)</sub>] contains at least fraction <i>p</i>
        of the population is computed using the Binomial distribution, without any
        assumption about the shape of the population distribution.</p>

        <p>For a one-sided bound, we use X<sub>(n)</sub> (the sample maximum) or
        X<sub>(1)</sub> (the sample minimum) and require:</p>

        <div class="formula">P(X<sub>(n-r+1)</sub> &ge; x<sub>p</sub>) &ge; &gamma;</div>

        <p>where <i>p</i> is the coverage and &gamma; is the confidence.</p>

        <p class="cite">Ref: ASME PTC 19.1-2018, non-parametric tolerance intervals.<br>
        Ref: Hahn &amp; Meeker (1991), <i>Statistical Intervals</i>, Chapter 5.</p>
        """
        return self._make_text_browser(html)

    # =================================================================
    # Sub-tab 6: Monte Carlo Method
    # =================================================================
    def _build_monte_carlo_tab(self):
        html = """
        <h1>Monte Carlo Uncertainty Propagation</h1>

        <h2>How It Works (Plain Language)</h2>
        <p>Instead of using an algebraic formula (RSS) to combine uncertainties,
        Monte Carlo simulation takes a direct, brute-force approach:</p>
        <ol>
            <li><b>Define</b> a probability distribution for each uncertainty source
                (Normal, Uniform, etc.) based on its stated standard uncertainty
                and distribution type.</li>
            <li><b>Draw</b> a large number of random samples (e.g. 100,000) from
                each source&rsquo;s distribution simultaneously.</li>
            <li><b>Combine</b> the drawn values for each trial using the model
                equation (typically RSS: square, sum, square root).</li>
            <li><b>Analyse</b> the resulting distribution of combined values to
                extract the desired percentile (e.g. 95th percentile) as the
                expanded uncertainty.</li>
        </ol>

        <h2>Mathematical Formulation</h2>
        <p>For <i>M</i> Monte Carlo trials and <i>N</i> uncertainty sources:</p>
        <div class="formula">
        For each trial j = 1 &hellip; M:<br>
        &nbsp;&nbsp; Draw x<sub>i,j</sub> ~ F<sub>i</sub>(&mu;<sub>i</sub>,
            &sigma;<sub>i</sub>) &nbsp; for i = 1 &hellip; N<br>
        &nbsp;&nbsp; Compute u<sub>c,j</sub> = &radic;( &Sigma;<sub>i</sub>
            x<sub>i,j</sub>&sup2; )
        </div>

        <p>The expanded uncertainty at coverage <i>p</i> is then:</p>
        <div class="formula">
        U<sub>val</sub>(p) = Percentile( {u<sub>c,1</sub>, &hellip;,
            u<sub>c,M</sub>}, &nbsp; p &times; 100 )
        </div>

        <h2>Why Monte Carlo Is Equivalent To (But More General Than) RSS</h2>
        <p>When all sources are <b>Normal</b> and the combination model is
        <b>linear</b>, the Monte Carlo result converges to the same answer as the
        analytical RSS formula. However, Monte Carlo is <b>more general</b>
        because it:</p>
        <ul>
            <li>Handles <b>non-Normal</b> distributions (Uniform, Triangular,
                Lognormal, etc.) without approximation</li>
            <li>Captures <b>non-linear</b> effects in the combination model</li>
            <li>Naturally produces the <b>full probability distribution</b> of the
                result, not just a point estimate and interval</li>
            <li>Does not require the Welch-Satterthwaite approximation for degrees
                of freedom &mdash; tail behaviour emerges directly from simulation</li>
        </ul>

        <div class="note">
            <b>Practical implication:</b> If your RSS and Monte Carlo results agree
            closely, the Normal/linear assumptions are justified. If they differ
            significantly, Monte Carlo is the more trustworthy result.
        </div>

        <h2>Bootstrap Confidence on the Percentile</h2>
        <p>A single Monte Carlo run gives one estimate of U<sub>val</sub>. To
        quantify the <b>sampling uncertainty of the Monte Carlo estimate itself</b>,
        a bootstrap procedure is used:</p>
        <ol>
            <li>From the M combined-uncertainty samples, draw B bootstrap resamples
                (each of size M, with replacement).</li>
            <li>For each resample, compute the p-th percentile.</li>
            <li>The spread of these B percentile estimates gives a confidence interval
                on U<sub>val</sub>.</li>
        </ol>
        <p>Typical choices: M = 100,000 trials, B = 1,000 bootstrap resamples.
        The 2.5th and 97.5th percentiles of the bootstrap distribution give a
        95% confidence interval on U<sub>val</sub>.</p>

        <h2>Convergence</h2>
        <p>The Monte Carlo estimate converges at a rate proportional to
        1/&radic;M. Doubling precision requires 4&times; more trials.
        For engineering uncertainty analysis, M = 50,000&ndash;200,000 is usually
        sufficient for 3-4 significant figures on the 95th percentile.</p>

        <p class="cite">Ref: JCGM 101:2008 (GUM Supplement 1), Sections 5, 6,
        and 7.<br>
        Ref: ASME V&amp;V 20-2009, Section 7.3 (Monte Carlo alternative).</p>
        """
        return self._make_text_browser(html)

    # =================================================================
    # Sub-tab 7: Glossary
    # =================================================================
    def _build_glossary_tab(self):
        html = """
        <h1>Glossary of Key Terms</h1>

        <table>
            <tr>
                <th style="text-align:left; width:220px;">Term</th>
                <th style="text-align:left;">Definition</th>
                <th style="text-align:left; width:180px;">Source Standard</th>
            </tr>
            <tr>
                <td><b>Coverage (Probability)</b></td>
                <td>The proportion of the population distribution that a tolerance
                    interval or expanded uncertainty is intended to contain. For
                    example, 95% coverage means the interval is expected to contain
                    at least 95% of the population values.</td>
                <td>JCGM 100:2008, 2.3.5;<br>ASME V&amp;V 20, Sec. 2</td>
            </tr>
            <tr>
                <td><b>Confidence (Level)</b></td>
                <td>The probability that the tolerance interval or expanded
                    uncertainty actually achieves the stated coverage. It reflects
                    how certain we are about our uncertainty estimate given finite
                    sample sizes. Higher confidence requires a larger coverage
                    factor <i>k</i>.</td>
                <td>JCGM 100:2008, 6.2.2;<br>ASME PTC 19.1-2018</td>
            </tr>
            <tr>
                <td><b>Coverage Factor (k)</b></td>
                <td>A multiplier applied to the combined standard uncertainty to
                    obtain an expanded uncertainty at a specified coverage and
                    confidence level. For a Normal distribution with large
                    degrees of freedom: k &approx; 1.645 (95% one-sided) or
                    k &approx; 1.960 (95% two-sided).</td>
                <td>JCGM 100:2008, 2.3.6;<br>GUM Annex G</td>
            </tr>
            <tr>
                <td><b>Tolerance Interval</b></td>
                <td>A statistical interval computed from sample data that is
                    expected to contain at least a specified proportion (coverage)
                    of the population with a stated confidence level. Unlike a
                    confidence interval (which bounds a parameter), a tolerance
                    interval bounds a <i>proportion of the population</i>.</td>
                <td>ASME PTC 19.1-2018;<br>ISO 16269-6</td>
            </tr>
            <tr>
                <td><b>Tolerance Limit</b></td>
                <td>The endpoint of a one-sided tolerance interval. A one-sided
                    upper tolerance limit, for example, is a value below which
                    at least the stated coverage proportion of the population
                    falls, with the stated confidence.</td>
                <td>ASME PTC 19.1-2018;<br>Krishnamoorthy &amp; Mathew (2009)</td>
            </tr>
            <tr>
                <td><b>Standard Uncertainty (u)</b></td>
                <td>The uncertainty of a measurement or quantity expressed as a
                    standard deviation (1&sigma;). It represents one standard
                    deviation of the probability distribution assigned to the
                    measurand. All uncertainties in V&amp;V 20 are first reduced to
                    standard uncertainties before combination.</td>
                <td>JCGM 100:2008, 2.3.1;<br>ASME V&amp;V 20, Sec. 3</td>
            </tr>
            <tr>
                <td><b>Expanded Uncertainty (U)</b></td>
                <td>The product of the combined standard uncertainty and the
                    coverage factor: U = k &middot; u<sub>c</sub>. It defines an
                    interval about the result that is expected to encompass a
                    large fraction (coverage) of the distribution of values that
                    could reasonably be attributed to the measurand.</td>
                <td>JCGM 100:2008, 2.3.5;<br>ASME V&amp;V 20, Sec. 9</td>
            </tr>
            <tr>
                <td><b>Type A Evaluation</b></td>
                <td>Evaluation of uncertainty by <b>statistical analysis</b> of a
                    series of observations. The standard uncertainty is the
                    experimental standard deviation of the mean or, for tolerance
                    intervals, the sample standard deviation. Degrees of freedom
                    are typically n &minus; 1.</td>
                <td>JCGM 100:2008, 2.3.2;<br>ASME V&amp;V 20, Sec. 3</td>
            </tr>
            <tr>
                <td><b>Type B Evaluation</b></td>
                <td>Evaluation of uncertainty by means <b>other than statistical
                    analysis</b> of repeated observations. Sources include
                    manufacturer specifications, calibration certificates,
                    engineering judgement, handbook data, or prior experience.
                    Degrees of freedom are typically set to &infin; (uncertainty
                    is taken as exact).</td>
                <td>JCGM 100:2008, 2.3.3;<br>ASME V&amp;V 20, Sec. 3</td>
            </tr>
            <tr>
                <td><b>Effective Degrees of Freedom (&nu;<sub>eff</sub>)</b></td>
                <td>The degrees of freedom of the combined standard uncertainty,
                    estimated via the Welch-Satterthwaite equation. It determines
                    the appropriate Student-t distribution from which the coverage
                    factor is taken. Low &nu;<sub>eff</sub> yields a larger k;
                    &nu;<sub>eff</sub> &rarr; &infin; yields the Normal k.</td>
                <td>JCGM 100:2008, Annex G;<br>ASME V&amp;V 20, Sec. 9</td>
            </tr>
        </table>

        <h2>Additional Related Terms</h2>
        <table>
            <tr>
                <th style="text-align:left; width:220px;">Term</th>
                <th style="text-align:left;">Definition</th>
            </tr>
            <tr>
                <td><b>RSS (Root Sum of Squares)</b></td>
                <td>The method of combining independent standard uncertainties by
                    taking the square root of the sum of their squares:
                    u<sub>c</sub> = &radic;(&Sigma; u<sub>i</sub>&sup2;). Valid
                    when sources are uncorrelated.</td>
            </tr>
            <tr>
                <td><b>Comparison Error (E)</b></td>
                <td>The difference between the simulation result and the experimental
                    datum: E = S &minus; D. The central quantity in V&amp;V 20
                    validation assessment.</td>
            </tr>
            <tr>
                <td><b>Validation Uncertainty (u<sub>val</sub>)</b></td>
                <td>The combined standard uncertainty of the validation comparison,
                    incorporating numerical, input, and experimental uncertainties.</td>
            </tr>
            <tr>
                <td><b>Model Deficiency</b></td>
                <td>The portion of the comparison error that cannot be explained by
                    the combined uncertainty. Indicated when |E| &gt; U<sub>val</sub>.</td>
            </tr>
            <tr>
                <td><b>GUM</b></td>
                <td>Guide to the Expression of Uncertainty in Measurement
                    (JCGM 100:2008). The foundational international standard for
                    uncertainty quantification.</td>
            </tr>
            <tr>
                <td><b>GUM Supplement 1</b></td>
                <td>JCGM 101:2008. Extends the GUM framework to Monte Carlo
                    propagation of distributions.</td>
            </tr>
        </table>

        <p class="cite">
        Ref: JCGM 100:2008 (GUM).<br>
        Ref: ASME V&amp;V 20-2009 (R2021).<br>
        Ref: ASME PTC 19.1-2018.<br>
        Ref: ISO 16269-6:2014.<br>
        Ref: Krishnamoorthy &amp; Mathew (2009), <i>Statistical Tolerance
        Regions</i>.
        </p>
        """
        return self._make_text_browser(html)


# =============================================================================
# SECTION 14: HTML REPORT GENERATOR
# =============================================================================
def _esc(text: str) -> str:
    """Escape a plain-text string for safe embedding in HTML."""
    if not isinstance(text, str):
        text = str(text)
    return (
        text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


class HTMLReportGenerator:
    """
    Generates a self-contained HTML report with embedded images for the
    VVUQ Uncertainty Aggregator analysis results.

    The report is designed for professional distribution and printing, with
    all images base64-encoded inline so the HTML file has no external
    dependencies.
    """

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------
    def generate_report(
        self,
        rss_results: RSSResults,
        mc_results: Optional[MCResults],
        comp_data: ComparisonData,
        settings: AnalysisSettings,
        sources: List[UncertaintySource],
        budget_table: list,
        rollup_table: list,
        findings_text: str,
        assumptions_text: str,
        figures: Dict[str, Figure],
        audit_entries: list,
        rollup_headers: list = None,
        project_metadata: Optional[dict] = None,
    ) -> str:
        """
        Build and return a complete self-contained HTML report string.

        Parameters
        ----------
        rss_results : RSSResults
            The RSS analysis results dataclass.
        mc_results : MCResults or None
            Monte Carlo results (may be None if MC was not run).
        comp_data : ComparisonData
            The comparison (CFD vs. test) data.
        settings : AnalysisSettings
            All analysis configuration settings.
        sources : list of UncertaintySource
            The uncertainty sources used in the analysis.
        budget_table : list
            Budget table rows (list of dicts) from
            ``RSSResultsTab.get_budget_table_data()``.
        rollup_table : list
            Roll-up table rows (list of lists) from
            ``ComparisonRollUpTab.get_rollup_table_data()``.
        findings_text : str
            Key-findings narrative written by the analyst.
        assumptions_text : str
            Assumptions and engineering judgments text.
        figures : dict mapping str -> matplotlib Figure
            Matplotlib figures to embed.  Expected keys (all optional):
            ``'comparison_plots'``, ``'rss_plots'``, ``'mc_plots'``.
        audit_entries : list
            List of audit-log entry dicts from ``audit_log.to_dict()``.
        project_metadata : dict, optional
            Program/analyst/date/notes and decision consequence metadata
            from the project info panel.

        Returns
        -------
        str
            A complete HTML document as a string.
        """
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        unit = settings.global_unit
        project_meta = project_metadata or {}
        consequence = normalize_decision_consequence(
            project_meta.get("decision_consequence", "Medium")
        )

        sections_html: List[str] = []

        # ---- Header ---------------------------------------------------
        sections_html.append(self._build_header(now_str))

        # ---- Table of Contents ----------------------------------------
        toc_entries = []
        n = 0
        n += 1; toc_entries.append(("section-config", f"{n}. Analysis Configuration"))
        n += 1; toc_entries.append(("section-compdata", f"{n}. Comparison Data Summary"))
        n += 1; toc_entries.append(("section-budget", f"{n}. Uncertainty Budget"))
        n += 1; toc_entries.append(("section-rss", f"{n}. RSS Results"))
        if mc_results is not None and mc_results.computed:
            n += 1
            toc_entries.append(("section-mc", f"{n}. Monte Carlo Results"))
        n += 1; toc_entries.append(("section-rollup", f"{n}. Comparison Roll-Up"))
        n += 1; toc_entries.append(("section-assumptions", f"{n}. Assumptions &amp; Engineering Judgments"))
        n += 1; toc_entries.append(("section-audit", f"{n}. Audit Trail"))
        n += 1; toc_entries.append(("section-decision-card", f"{n}. Decision Card"))
        n += 1; toc_entries.append(("section-credibility", f"{n}. Credibility Framing"))
        n += 1; toc_entries.append(("section-vvuq-glossary", f"{n}. VVUQ Terminology Panel"))
        n += 1; toc_entries.append(("section-conformity", f"{n}. Conformity Assessment Template"))
        sections_html.append(self._build_toc(toc_entries))

        # ---- 1. Analysis Configuration --------------------------------
        sections_html.append(self._build_config_section(
            settings, project_meta=project_meta
        ))

        # ---- 2. Comparison Data Summary --------------------------------
        sections_html.append(self._build_comp_data_section(
            comp_data, unit, figures.get("comparison_plots")))

        # ---- 3. Uncertainty Budget ------------------------------------
        sections_html.append(self._build_budget_section(
            budget_table, sources, unit, rss_results))

        # ---- 4. RSS Results -------------------------------------------
        if rss_results is not None and rss_results.computed:
            sections_html.append(self._build_rss_section(
                rss_results, settings, unit, figures.get("rss_plots")))

        # ---- 5. Monte Carlo Results (optional) ------------------------
        if mc_results is not None and mc_results.computed:
            sections_html.append(self._build_mc_section(
                mc_results, rss_results, settings, unit,
                figures.get("mc_plots")))

        # ---- 6. Comparison Roll-Up ------------------------------------
        sections_html.append(self._build_rollup_section(
            rollup_table, findings_text,
            header_labels=rollup_headers))

        # ---- 7. Assumptions & Engineering Judgments --------------------
        sections_html.append(self._build_assumptions_section(
            assumptions_text, audit_entries))

        # ---- 8. Audit Trail -------------------------------------------
        sections_html.append(self._build_audit_section(audit_entries))

        # ---- 9+. Decision guidance + credibility framing --------------
        sections_html.append(self._build_decision_card_section(
            rss_results, mc_results, settings, unit
        ))
        sections_html.append(self._build_credibility_section(
            rss_results, settings, sources, project_meta, consequence
        ))
        sections_html.append(render_vvuq_glossary_html())
        sections_html.append(self._build_conformity_section(
            rss_results, consequence
        ))

        # ---- Footer ---------------------------------------------------
        sections_html.append(self._build_footer(now_str))

        body_content = "\n".join(sections_html)

        return self._wrap_html(body_content)

    def save_report(self, html_content: str, filepath: str) -> None:
        """Write the HTML report string to *filepath*."""
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(html_content)

    # -----------------------------------------------------------------
    # Image embedding helper
    # -----------------------------------------------------------------
    @staticmethod
    def _figure_to_base64(fig: Figure) -> str:
        """Render a matplotlib *Figure* to a PNG data-URI with light theme.

        Temporarily applies REPORT_PLOT_STYLE to all axes for
        print-ready appearance, then restores original properties.
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
            # Apply light theme
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
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight",
                        facecolor="white", edgecolor="none")
            buf.seek(0)
            encoded = base64.b64encode(buf.read()).decode("ascii")
            buf.close()
        finally:
            # Restore original properties (always runs, even on savefig error)
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
    def _embed_figure(fig: Optional[Figure], alt: str = "figure") -> str:
        """Return an ``<img>`` tag for the figure, or an empty string."""
        if fig is None:
            return ""
        uri = HTMLReportGenerator._figure_to_base64(fig)
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
            <div class="company-name">{_esc(COMPANY_NAME)}</div>
            <div class="team-name">{_esc(TEAM_NAME)}</div>

            <div class="notice-box proprietary-notice">
                <strong>PROPRIETARY NOTICE</strong><br/>
                {_esc(PROPRIETARY_NOTICE)}
            </div>

            <div class="notice-box export-notice">
                <strong>EXPORT CONTROL NOTICE</strong><br/>
                {_esc(EXPORT_CONTROL_NOTICE)}
            </div>

            <h1 class="report-title">{_esc(APP_NAME)} v{_esc(APP_VERSION)} &mdash; Analysis Report</h1>
            <div class="report-meta">
                Generated: {_esc(now_str)}<br/>
                Version: {_esc(APP_VERSION)} ({_esc(APP_DATE)})
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

    # -- 1. Configuration -----------------------------------------------
    def _build_config_section(self, s: AnalysisSettings,
                              project_meta: Optional[dict] = None) -> str:
        project_meta = project_meta or {}
        sided_label = "One-Sided" if s.one_sided else "Two-Sided"
        consequence = normalize_decision_consequence(
            project_meta.get("decision_consequence", "Medium")
        )
        rows = [
            ("Program / Project", _esc(project_meta.get("program", "—"))),
            ("Analyst", _esc(project_meta.get("analyst", "—"))),
            ("Date", _esc(project_meta.get("date", "—"))),
            ("Decision consequence", _esc(consequence)),
            ("Coverage probability", f"{s.coverage*100:.1f}%"),
            ("Confidence level", f"{s.confidence*100:.1f}%"),
            ("Interval type", sided_label),
            ("k-factor method", _esc(s.k_method)),
            ("Validation metric mode", _esc(getattr(s, 'validation_mode', "Standard scalar (V&V 20)"))),
            ("Bound type", _esc(s.bound_type)),
            ("Units", _esc(s.global_unit)),
            ("MC sampling method", _esc(getattr(s, 'mc_sampling_method',
                                               'Monte Carlo (Random)'))),
            ("MC trials", f"{s.mc_n_trials:,}"),
            ("MC seed", str(s.mc_seed) if s.mc_seed is not None else "Random"),
            ("MC bootstrap resampling",
             "Enabled" if s.mc_bootstrap else "Disabled"),
            ("MC comparison data", "E\u0305 constant offset (V&V 20)"),
        ]
        if s.k_method == K_METHOD_MANUAL:
            rows.insert(4, ("Manual k value", f"{s.manual_k:.4f}"))

        body_rows = "\n".join(
            f"            <tr><td class='label-cell'>{lbl}</td>"
            f"<td>{val}</td></tr>"
            for lbl, val in rows
        )
        return textwrap.dedent(f"""\
        <div class="section" id="section-config">
            <h2>1. Analysis Configuration</h2>
            <table class="config-table">
                <tbody>
        {body_rows}
                </tbody>
            </table>
        </div>
        """)

    # -- 2. Comparison Data Summary ------------------------------------
    def _build_comp_data_section(self, cd: ComparisonData, unit: str,
                                  fig: Optional[Figure]) -> str:
        flat = cd.flat_data()
        n_total = flat.size
        n_locs = cd.data.shape[0] if cd.data.ndim == 2 else 0
        n_conds = cd.data.shape[1] if cd.data.ndim == 2 else 0

        stats_dict = cd.get_stats()
        stats_rows = ""
        if stats_dict:
            stat_keys = [
                ("n", "N"),
                ("mean", "Mean"),
                ("std", "Std Dev (s)"),
                ("min", "Min"),
                ("max", "Max"),
                ("median", "Median"),
                ("skewness", "Skewness"),
                ("kurtosis", "Excess Kurtosis"),
            ]
            for key, label in stat_keys:
                val = stats_dict.get(key, "")
                if isinstance(val, float):
                    val = f"{val:.6g}"
                stats_rows += (
                    f"            <tr><td class='label-cell'>{label}</td>"
                    f"<td>{val} {_esc(unit) if key != 'n' else ''}</td></tr>\n"
                )

        img_html = self._embed_figure(fig, "Comparison data plots")

        return textwrap.dedent(f"""\
        <div class="section" id="section-compdata">
            <h2>2. Comparison Data Summary</h2>
            <p>Data dimensions: <strong>{n_locs}</strong> locations &times;
               <strong>{n_conds}</strong> conditions = <strong>{n_total}</strong>
               total data points.</p>
            <table class="stats-table">
                <thead>
                    <tr><th>Statistic</th><th>Value</th></tr>
                </thead>
                <tbody>
        {stats_rows}
                </tbody>
            </table>
            {img_html}
        </div>
        """)

    # -- 3. Uncertainty Budget ------------------------------------------
    def _build_budget_section(self, budget_table: list,
                               sources: List[UncertaintySource],
                               unit: str,
                               rss: Optional[RSSResults] = None) -> str:
        # Build HTML table from budget_table (list of dicts)
        _has_corr = rss is not None and getattr(rss, 'has_correlations', False)
        _pct_th = ("% Contribution<sup>&dagger;</sup>"
                   if _has_corr else "% Contribution")
        header = (
            "<tr>"
            "<th>Source</th>"
            "<th>Category</th>"
            "<th>&sigma; (1&sigma;)</th>"
            "<th>&sigma;&sup2;</th>"
            "<th>DOF (&nu;)</th>"
            f"<th>{_pct_th}</th>"
            "<th>Distribution</th>"
            "<th>Data Basis</th>"
            "<th>Class</th>"
            "<th>Reducibility</th>"
            "</tr>"
        )

        body_rows = []
        dominant_name = ""
        dominant_pct = 0.0

        for row in budget_table:
            is_sub = row.get("is_subtotal", False)
            is_tot = row.get("is_total", False)
            name = _esc(str(row.get("name", "")))
            category = _esc(str(row.get("category", "")))
            sigma = row.get("sigma", 0.0)
            sigma_sq = row.get("sigma_sq", 0.0)
            dof_val = row.get("dof", "")
            pct = row.get("pct", 0.0)
            dist = _esc(str(row.get("distribution", "")))
            basis = _esc(str(row.get("data_basis", "")))
            u_class = _esc(str(row.get("uncertainty_class", "")))
            reducibility = _esc(str(row.get("reducibility", "")))

            # Track dominant source (skip subtotals/totals)
            if not is_sub and not is_tot and pct > dominant_pct:
                dominant_pct = pct
                dominant_name = row.get("name", "")

            # Format DOF
            if isinstance(dof_val, float):
                dof_str = "&infin;" if np.isinf(dof_val) else f"{dof_val:.1f}"
            else:
                dof_str = _esc(str(dof_val))

            row_class = ""
            if is_tot:
                row_class = ' class="total-row"'
            elif is_sub:
                row_class = ' class="subtotal-row"'

            # Asymmetric annotation on sigma
            if row.get('asymmetric'):
                sp = row.get('sigma_upper', 0)
                sm = row.get('sigma_lower', 0)
                sigma_html = (
                    f"{sigma:.6g}<br>"
                    f"<small>&sigma;&sup1; = {sp:.4g} / "
                    f"&sigma;&sub; = {sm:.4g}</small>"
                )
            else:
                sigma_html = f"{sigma:.6g}"

            body_rows.append(
                f"            <tr{row_class}>"
                f"<td>{name}</td>"
                f"<td>{category}</td>"
                f"<td>{sigma_html}</td>"
                f"<td>{sigma_sq:.6g}</td>"
                f"<td>{dof_str}</td>"
                f"<td>{pct:.2f}%</td>"
                f"<td>{dist}</td>"
                f"<td>{basis}</td>"
                f"<td>{u_class}</td>"
                f"<td>{reducibility}</td>"
                f"</tr>"
            )

        all_body = "\n".join(body_rows)

        # Category subtotals narrative
        cat_summaries = []
        for row in budget_table:
            if row.get("is_subtotal", False):
                cat_summaries.append(
                    f"<li><strong>{_esc(row['name'].strip())}</strong>: "
                    f"&sigma; = {row['sigma']:.6g} {_esc(unit)}, "
                    f"{row['pct']:.1f}% of total variance</li>"
                )

        subtotal_html = ""
        if cat_summaries:
            subtotal_html = (
                "<h3>Category Subtotals</h3>\n<ul>\n"
                + "\n".join(cat_summaries)
                + "\n</ul>"
            )

        dominant_html = ""
        if dominant_name:
            dominant_html = (
                f'<p class="highlight">Dominant uncertainty source: '
                f'<strong>{_esc(dominant_name)}</strong> '
                f'({dominant_pct:.1f}% of total variance).</p>'
            )

        # --- Uncertainty class-split summary ---
        # Use engine-computed values (correlation-aware) when available;
        # fall back to independent recomputation only if rss is unavailable.
        if rss is not None and rss.computed:
            u_a = rss.u_aleatoric
            u_e = rss.u_epistemic
            u_m = rss.u_mixed
            pct_epi = rss.pct_epistemic
        else:
            class_sigmas_fb: Dict[str, List[float]] = {
                'aleatoric': [], 'epistemic': [], 'mixed': []
            }
            for row in budget_table:
                if row.get("is_subtotal") or row.get("is_total"):
                    continue
                cls = row.get("uncertainty_class", "aleatoric")
                if cls not in ('aleatoric', 'epistemic', 'mixed'):
                    cls = 'aleatoric'
                class_sigmas_fb[cls].append(row.get("sigma", 0.0))
            u_a = float(np.sqrt(sum(s ** 2 for s in class_sigmas_fb['aleatoric'])))
            u_e = float(np.sqrt(sum(s ** 2 for s in class_sigmas_fb['epistemic'])))
            u_m = float(np.sqrt(sum(s ** 2 for s in class_sigmas_fb['mixed'])))
            u_total_fb = float(np.sqrt(u_a ** 2 + u_e ** 2 + u_m ** 2))
            pct_epi = (u_e ** 2 / (u_total_fb ** 2) * 100.0) if u_total_fb > 0 else 0.0
        u_total_class = float(np.sqrt(u_a ** 2 + u_e ** 2 + u_m ** 2))

        class_split_html = (
            f'<p><strong>Uncertainty class split:</strong> '
            f'U<sub>A</sub>&nbsp;=&nbsp;{u_a:.4g}&nbsp;{_esc(unit)}, '
            f'U<sub>E</sub>&nbsp;=&nbsp;{u_e:.4g}&nbsp;{_esc(unit)}, '
            f'U<sub>total</sub>&nbsp;=&nbsp;{u_total_class:.4g}&nbsp;{_esc(unit)}. '
            f'Epistemic fraction: {pct_epi:.1f}% of total variance.</p>'
        )

        # Dominant drivers by class
        driver_html_parts = []
        for cls_key, cls_label in [('aleatoric', 'aleatoric'),
                                    ('epistemic', 'epistemic')]:
            best_name = ""
            best_pct = 0.0
            best_red = ""
            for row in budget_table:
                if row.get("is_subtotal") or row.get("is_total"):
                    continue
                if row.get("uncertainty_class", "aleatoric") == cls_key:
                    rpct = row.get("pct", 0.0)
                    if rpct > best_pct:
                        best_pct = rpct
                        best_name = row.get("name", "")
                        best_red = row.get("reducibility", "")
            if best_name:
                extra = (f", reducibility: {_esc(best_red)}"
                         if cls_key == "epistemic" and best_red else "")
                driver_html_parts.append(
                    f"Dominant {cls_label} driver: "
                    f"<strong>{_esc(best_name)}</strong> ({best_pct:.1f}%{extra})"
                )

        if driver_html_parts:
            class_split_html += "<p>" + ". ".join(driver_html_parts) + ".</p>"

        # Epistemic dominance warning
        if pct_epi > 50.0:
            class_split_html += (
                '<p class="highlight">&#9888; Epistemic uncertainty dominates '
                f'({pct_epi:.1f}% of total variance) &mdash; consider '
                'knowledge-reduction actions before making compliance claims.</p>'
            )

        return textwrap.dedent(f"""\
        <div class="section" id="section-budget">
            <h2>3. Uncertainty Budget</h2>
            <div class="table-wrapper">
            <table class="budget-table">
                <thead>
                    {header}
                </thead>
                <tbody>
        {all_body}
                </tbody>
            </table>
            </div>
            {('<p style="font-size:0.85em;color:#666;">'
              '<sup>&dagger;</sup> With correlated sources, individual '
              'percentages do not sum to 100% because cross-correlation '
              'terms also contribute to u<sub>val</sub>&sup2;.</p>'
              ) if _has_corr else ''}
            {subtotal_html}
            {dominant_html}
            {class_split_html}
        </div>
        """)

    # -- 4. RSS Results -------------------------------------------------
    def _build_rss_section(self, r: RSSResults, s: AnalysisSettings,
                            unit: str, fig: Optional[Figure]) -> str:
        nu_str = "&infin;" if np.isinf(r.nu_eff) else f"{r.nu_eff:.2f}"
        sided_label = "one-sided" if s.one_sided else "two-sided"

        results_rows = [
            ("u<sub>num</sub>", f"{r.u_num:.6g} {_esc(unit)}"),
            ("u<sub>input</sub>", f"{r.u_input:.6g} {_esc(unit)}"),
            ("u<sub>D</sub>", f"{r.u_D:.6g} {_esc(unit)}"),
            ("u<sub>val</sub> (combined standard uncertainty)",
             f"{r.u_val:.6g} {_esc(unit)}"),
            ("Effective DOF (&nu;<sub>eff</sub>)", nu_str),
            ("k-factor", f"{r.k_factor:.4f}"),
            ("k-factor method", _esc(r.k_method_used)),
            (f"U<sub>val</sub> ({s.coverage*100:.0f}%/{s.confidence*100:.0f}% "
             f"{sided_label})",
             f"{r.U_val:.6g} {_esc(unit)}"),
            ("Mean comparison error (E&#772;)", f"{r.E_mean:.6g} {_esc(unit)}"),
            ("Comparison scatter (s<sub>E</sub>)",
             f"{r.s_E:.6g} {_esc(unit)}"),
            ("Number of data points", f"{r.n_data}"),
            ("Data sufficiency",
             "adequate" if r.n_data >= 30
             else ("marginal" if r.n_data >= 10
                   else "insufficient per GUM &sect;G.3")),
        ]

        if getattr(r, "multivariate_enabled", False):
            results_rows.append(
                ("Multivariate supplement", "Enabled (covariance-aware)")
            )
            if getattr(r, "multivariate_computed", False):
                mv_txt = (
                    f"score={r.multivariate_score:.4g}, "
                    f"T<sup>2</sup>={r.multivariate_t2:.4g}, "
                    f"p={r.multivariate_pvalue:.4g}, "
                    f"n={r.multivariate_n_locations}&times;{r.multivariate_n_conditions}"
                )
                results_rows.append(("Multivariate summary", mv_txt))
            else:
                results_rows.append(("Multivariate summary", _esc(r.multivariate_note or "Not computable")))

        # Correlation disclosure
        corr_html_extra = ""
        if r.has_correlations:
            results_rows.append(
                ("Source correlation",
                 f"Applied (groups: {', '.join(r.correlation_groups)})")
            )
            # Build HTML correlation matrix tables
            for grp_name, names, C_mat in r.correlation_matrices:
                th_cells = "".join(
                    f"<th>{_esc(n)}</th>" for n in names
                )
                rows_html = ""
                for ri, rn in enumerate(names):
                    td_cells = "".join(
                        f"<td>{C_mat[ri, ci]:.2f}</td>"
                        for ci in range(len(names))
                    )
                    rows_html += (
                        f"<tr><th>{_esc(rn)}</th>{td_cells}</tr>\n"
                    )
                corr_html_extra += (
                    f'<h4>Group &ldquo;{_esc(grp_name)}&rdquo; '
                    f'effective &rho; matrix</h4>\n'
                    f'<table class="stats-table"><thead><tr>'
                    f'<th></th>{th_cells}</tr></thead>\n'
                    f'<tbody>{rows_html}</tbody></table>\n'
                )
        else:
            results_rows.append(
                ("Source correlation", "Independent (no correlations)")
            )

        if (not np.isnan(r.lower_bound_uval)
                and (r.lower_bound_uval != 0.0 or r.upper_bound_uval != 0.0)):
            results_rows.append(
                ("U<sub>val</sub> bounds",
                 f"[{r.lower_bound_uval:.6g}, {r.upper_bound_uval:.6g}] "
                 f"{_esc(unit)}")
            )
        if (not np.isnan(r.lower_bound_sE)
                and (r.lower_bound_sE != 0.0 or r.upper_bound_sE != 0.0)):
            results_rows.append(
                ("s<sub>E</sub> bounds",
                 f"[{r.lower_bound_sE:.6g}, {r.upper_bound_sE:.6g}] "
                 f"{_esc(unit)}")
            )

        body = "\n".join(
            f"            <tr><td class='label-cell'>{lbl}</td>"
            f"<td>{val}</td></tr>"
            for lbl, val in results_rows
        )

        # Validation assessment
        if r.u_val > 0:
            ratio = abs(r.E_mean) / r.U_val if r.U_val > 0 else float('inf')
            if ratio <= 1.0:
                verdict = (
                    f'<p class="verdict pass">|E&#772;| / U<sub>val</sub> = '
                    f'{ratio:.3f} &le; 1.0 &mdash; <strong>Validation '
                    f'requirement is SATISFIED.</strong> The model-form '
                    f'uncertainty is within the expanded validation '
                    f'uncertainty.</p>'
                )
            else:
                verdict = (
                    f'<p class="verdict fail">|E&#772;| / U<sub>val</sub> = '
                    f'{ratio:.3f} &gt; 1.0 &mdash; <strong>Validation '
                    f'requirement is NOT satisfied.</strong> Significant '
                    f'unaccounted model-form error exists.</p>'
                )
        else:
            verdict = (
                '<p class="verdict neutral">RSS results not computed or '
                'u<sub>val</sub> = 0.</p>'
            )

        mv_verdict = ""
        if getattr(r, "multivariate_enabled", False):
            if getattr(r, "multivariate_computed", False):
                mv_pass = r.multivariate_pvalue >= 0.05
                cls = "pass" if mv_pass else "fail"
                txt = (
                    "multivariate residual structure is consistent with expected noise."
                    if mv_pass else
                    "multivariate residual structure suggests unresolved systematic pattern."
                )
                mv_verdict = (
                    f'<p class="verdict {cls}">Multivariate supplement: '
                    f'p = {r.multivariate_pvalue:.4f} '
                    f'&mdash; <strong>{txt}</strong></p>'
                )
            else:
                mv_verdict = (
                    '<p class="verdict neutral">Multivariate supplement enabled '
                    'but not computable for the current dataset.</p>'
                )
            if r.multivariate_note:
                mv_verdict += f"<p><em>{_esc(r.multivariate_note)}</em></p>"

        img_html = self._embed_figure(fig, "RSS result plots")

        return textwrap.dedent(f"""\
        <div class="section" id="section-rss">
            <h2>4. RSS Results</h2>
            <table class="results-table">
                <thead>
                    <tr><th>Quantity</th><th>Value</th></tr>
                </thead>
                <tbody>
        {body}
                </tbody>
            </table>
            {corr_html_extra}
            {verdict}
            {mv_verdict}
            {img_html}
        </div>
        """)

    # -- 5. Monte Carlo Results -----------------------------------------
    def _build_mc_section(self, mc: MCResults, rss: RSSResults,
                           s: AnalysisSettings, unit: str,
                           fig: Optional[Figure]) -> str:
        sided_label = "one-sided" if s.one_sided else "two-sided"

        mc_rows = [
            ("Sampling method",
             _esc(getattr(mc, 'sampling_method', 'Monte Carlo (Random)'))),
            ("Number of trials", f"{mc.n_trials:,}"),
            ("Combined mean", f"{mc.combined_mean:.6g} {_esc(unit)}"),
            ("Combined std dev", f"{mc.combined_std:.6g} {_esc(unit)}"),
            (f"P{mc._lower_pct:.4g}" if hasattr(mc, '_lower_pct') else "P5",
             f"{mc.pct_5:.6g} {_esc(unit)}"),
            (f"P{mc._upper_pct:.4g}" if hasattr(mc, '_upper_pct') else "P95",
             f"{mc.pct_95:.6g} {_esc(unit)}"),
            (f"Lower bound ({s.coverage*100:.0f}%/{s.confidence*100:.0f}% "
             f"{sided_label})",
             f"{mc.lower_bound:.6g} {_esc(unit)}"),
            (f"Upper bound ({s.coverage*100:.0f}%/{s.confidence*100:.0f}% "
             f"{sided_label})",
             f"{mc.upper_bound:.6g} {_esc(unit)}"),
        ]

        lo_h = f"P{mc._lower_pct:.4g}" if hasattr(mc, '_lower_pct') else "P5"
        hi_h = f"P{mc._upper_pct:.4g}" if hasattr(mc, '_upper_pct') else "P95"
        if mc.bootstrap_ci_low != 0 or mc.bootstrap_ci_high != 0:
            mc_rows.append(
                (f"Bootstrap 95% CI envelope (lower on {lo_h}, upper on {hi_h})",
                 f"[{mc.bootstrap_ci_low:.6g}, {mc.bootstrap_ci_high:.6g}] "
                 f"{_esc(unit)}")
            )

        notes_html = ""
        mc_notes = getattr(mc, 'notes', None)
        if mc_notes:
            note_items = "\n".join(
                f"<li>{_esc(str(n))}</li>" for n in mc_notes
            )
            notes_html = (
                "<h3>Monte Carlo Notes</h3>"
                f"<ul>{note_items}</ul>"
            )

        body = "\n".join(
            f"            <tr><td class='label-cell'>{lbl}</td>"
            f"<td>{val}</td></tr>"
            for lbl, val in mc_rows
        )

        # Comparison to RSS
        if rss.computed and rss.U_val > 0:
            mc_bound = max(abs(mc.lower_bound), abs(mc.upper_bound))
            diff = mc_bound - rss.U_val
            pct_diff = diff / rss.U_val * 100.0 if rss.U_val != 0 else 0
            comparison_html = (
                f'<h3>Comparison to RSS</h3>'
                f'<p>MC bound magnitude: <strong>{mc_bound:.6g} '
                f'{_esc(unit)}</strong><br/>'
                f'RSS U<sub>val</sub>: <strong>{rss.U_val:.6g} '
                f'{_esc(unit)}</strong><br/>'
                f'Difference: <strong>{diff:+.6g} {_esc(unit)}</strong> '
                f'({pct_diff:+.2f}%)</p>'
            )
        else:
            comparison_html = ""

        img_html = self._embed_figure(fig, "Monte Carlo result plots")

        return textwrap.dedent(f"""\
        <div class="section page-break-before" id="section-mc">
            <h2>5. Monte Carlo Results</h2>
            <table class="results-table">
                <thead>
                    <tr><th>Quantity</th><th>Value</th></tr>
                </thead>
                <tbody>
        {body}
                </tbody>
            </table>
            {comparison_html}
            {notes_html}
            {img_html}
        </div>
        """)

    # -- 6. Comparison Roll-Up ------------------------------------------
    def _build_rollup_section(self, rollup_table: list,
                               findings_text: str,
                               header_labels: list = None) -> str:
        # rollup_table is a list of lists (rows) from the QTableWidget
        if not rollup_table:
            table_html = "<p><em>No roll-up data available.</em></p>"
        else:
            # Column headers — use provided labels or infer from data width
            if header_labels is None or len(header_labels) == 0:
                n_cols = len(rollup_table[0]) if rollup_table else 5
                header_labels = ["Quantity", "RSS (u_val)", "RSS (s_E)",
                                 "Monte Carlo", "Empirical"]
                # Pad if comparison columns added more
                while len(header_labels) < n_cols:
                    header_labels.append(f"Col {len(header_labels)+1}")
            header_cells = "".join(
                f"<th>{_esc(h)}</th>" for h in header_labels
            )
            header_row = f"            <tr>{header_cells}</tr>"

            rows_html = []
            for i, row in enumerate(rollup_table):
                tag = "td"
                cells = "".join(
                    f"<{tag}>{_esc(str(c))}</{tag}>" for c in row
                )
                row_class = ' class="alt-row"' if i % 2 == 1 else ""
                rows_html.append(f"            <tr{row_class}>{cells}</tr>")
            all_rows = "\n".join(rows_html)
            table_html = textwrap.dedent(f"""\
            <div class="table-wrapper">
            <table class="rollup-table">
                <thead>
        {header_row}
                </thead>
                <tbody>
        {all_rows}
                </tbody>
            </table>
            </div>""")

        findings_html = ""
        if findings_text and findings_text.strip():
            findings_html = (
                "<h3>Key Findings</h3>\n"
                f"<div class='findings-block'>{_esc(findings_text)}</div>"
            )

        return textwrap.dedent(f"""\
        <div class="section page-break-before" id="section-rollup">
            <h2>6. Comparison Roll-Up</h2>
            {table_html}
            {findings_html}
        </div>
        """)

    # -- 7. Assumptions & Engineering Judgments --------------------------
    def _build_assumptions_section(self, assumptions_text: str,
                                    audit_entries: list) -> str:
        # Auto-populate from audit log ASSUMPTION entries
        auto_items = []
        for entry in audit_entries:
            if entry.get("action") == "ASSUMPTION":
                desc = _esc(entry.get("description", ""))
                details = _esc(entry.get("details", ""))
                line = desc
                if details:
                    line += f" &mdash; {details}"
                auto_items.append(f"<li>{line}</li>")

        override_items = []
        for entry in audit_entries:
            if entry.get("action") == "USER_OVERRIDE":
                desc = _esc(entry.get("description", ""))
                details = _esc(entry.get("details", ""))
                line = desc
                if details:
                    line += f" &mdash; {details}"
                override_items.append(f"<li>{line}</li>")

        auto_html = ""
        if auto_items:
            auto_html = (
                "<h3>Logged Assumptions</h3>\n<ul>\n"
                + "\n".join(auto_items) + "\n</ul>"
            )

        override_html = ""
        if override_items:
            override_html = (
                "<h3>User Overrides</h3>\n<ul>\n"
                + "\n".join(override_items) + "\n</ul>"
            )

        user_text_html = ""
        if assumptions_text and assumptions_text.strip():
            user_text_html = (
                "<h3>Analyst Notes</h3>\n"
                f"<div class='assumptions-block'>"
                f"{_esc(assumptions_text)}</div>"
            )

        return textwrap.dedent(f"""\
        <div class="section" id="section-assumptions">
            <h2>7. Assumptions &amp; Engineering Judgments</h2>
            {auto_html}
            {override_html}
            {user_text_html}
        </div>
        """)

    # -- 8. Audit Trail -------------------------------------------------
    def _build_audit_section(self, audit_entries: list) -> str:
        if not audit_entries:
            return textwrap.dedent("""\
            <div class="section page-break-before" id="section-audit">
                <h2>8. Audit Trail</h2>
                <p><em>No audit entries recorded.</em></p>
            </div>
            """)

        rows = []
        for i, entry in enumerate(audit_entries):
            ts = _esc(entry.get("timestamp", "")[:19].replace("T", " "))
            action = _esc(entry.get("action", ""))
            desc = _esc(entry.get("description", ""))
            details = _esc(entry.get("details", ""))
            row_class = ' class="alt-row"' if i % 2 == 1 else ""
            rows.append(
                f"            <tr{row_class}>"
                f"<td class='ts-cell'>{ts}</td>"
                f"<td class='action-cell'>{action}</td>"
                f"<td>{desc}</td>"
                f"<td class='details-cell'>{details}</td>"
                f"</tr>"
            )
        all_rows = "\n".join(rows)

        return textwrap.dedent(f"""\
        <div class="section page-break-before" id="section-audit">
            <h2>8. Audit Trail</h2>
            <div class="table-wrapper">
            <table class="audit-table">
                <thead>
                    <tr>
                        <th>Timestamp</th>
                        <th>Action</th>
                        <th>Description</th>
                        <th>Details</th>
                    </tr>
                </thead>
                <tbody>
        {all_rows}
                </tbody>
            </table>
            </div>
        </div>
        """)

    def _build_decision_card_section(
        self,
        rss: Optional[RSSResults],
        mc: Optional[MCResults],
        settings: AnalysisSettings,
        unit: str,
    ) -> str:
        """Render novice-first final decision card."""
        if not rss or not rss.computed:
            card = render_decision_card_html(
                title="Decision Card (Final Validation Summary)",
                use_value="No valid result. Run RSS first.",
                use_distribution="N/A",
                use_combination="N/A",
                stop_checks=["RSS analysis not computed"],
                notes="No carry decision can be made yet.",
            )
            return card.replace(
                '<div class="section">',
                '<div class="section" id="section-decision-card">',
                1,
            )

        use_value = f"U_val = {rss.U_val:.6g} {unit}"
        if mc is not None and mc.computed:
            use_value += (
                f"; MC interval = [{mc.lower_bound:.6g}, {mc.upper_bound:.6g}] {unit}"
            )
        use_distribution = (
            "Normal (RSS basis) plus sampled source distributions in Monte Carlo"
            if mc is not None and mc.computed else
            "Normal (RSS basis)"
        )
        use_combination = (
            "RSS for formal scalar check; Monte Carlo as distribution-shape cross-check"
            if mc is not None and mc.computed else
            "RSS"
        )

        stop_checks = [
            "No comparison data loaded (n = 0)",
            "Scalar validation fails: |Ebar| > U_val",
            "Effective DOF below 5 (k-factor instability)",
            "Any unresolved unit mismatch between source and global unit",
        ]
        if getattr(rss, "multivariate_enabled", False):
            stop_checks.append(
                "Multivariate supplement fails (p < 0.05) without documented justification"
            )

        card = render_decision_card_html(
            title="Decision Card (Final Validation Summary)",
            use_value=use_value,
            use_distribution=use_distribution,
            use_combination=use_combination,
            stop_checks=stop_checks,
            notes=(
                f"Configured for {settings.coverage*100:.0f}% coverage and "
                f"{settings.confidence*100:.0f}% confidence."
            ),
        )
        return card.replace(
            '<div class="section">',
            '<div class="section" id="section-decision-card">',
            1,
        )

    def _build_credibility_section(
        self,
        rss: Optional[RSSResults],
        settings: AnalysisSettings,
        sources: List[UncertaintySource],
        project_meta: dict,
        consequence: str,
    ) -> str:
        """Render deterministic credibility checklist section."""
        enabled_sources = [
            s for s in sources
            if getattr(s, "enabled", True) and s.get_standard_uncertainty() > 0
        ]
        unit_ok = all(
            (not getattr(s, "unit", "").strip())
            or s.unit == settings.global_unit
            for s in enabled_sources
        )
        diagnostics_pass = bool(rss and rss.computed)
        if rss and rss.computed:
            diagnostics_pass = (
                (rss.nu_eff >= 5 or np.isinf(rss.nu_eff))
                and (rss.bias_explained is not False)
            )
            if (
                getattr(rss, "multivariate_enabled", False)
                and getattr(rss, "multivariate_computed", False)
                and getattr(rss, "multivariate_pvalue", 1.0) < 0.05
            ):
                diagnostics_pass = False

        evidence = {
            'inputs_documented': bool(project_meta.get("program", "").strip())
            and bool(project_meta.get("analyst", "").strip()),
            'method_selected': True,
            'units_consistent': unit_ok,
            'data_quality': bool(rss and rss.n_data >= 10),
            'diagnostics_pass': diagnostics_pass,
            'independent_review': str(
                project_meta.get("independent_review", "")
            ).strip().lower() in ("1", "true", "yes", "y"),
            'conservative_bound': (
                settings.coverage >= 0.95
                and (settings.one_sided or "s_E" in str(settings.bound_type))
            ),
            'validation_plan': bool(rss and rss.n_data > 0),
        }
        return render_credibility_html(consequence, evidence)

    def _build_conformity_section(
        self,
        rss: Optional[RSSResults],
        consequence: str,
    ) -> str:
        """Render optional conformity wording template section."""
        if rss is None or not rss.computed or rss.U_val <= 0:
            metric_value = "Not available (run RSS first)"
        else:
            ratio = abs(rss.E_mean) / rss.U_val if rss.U_val > 0 else float("inf")
            metric_value = f"{ratio:.4f}"
        return render_conformity_template_html(
            metric_name="Validation ratio |Ebar| / U_val",
            metric_value=metric_value,
            consequence=consequence,
        )

    # -- Footer ---------------------------------------------------------
    def _build_footer(self, now_str: str) -> str:
        return textwrap.dedent(f"""\
        <hr class="footer-rule"/>
        <div class="footer-block">
            <p>Report generated by <strong>{_esc(APP_NAME)}</strong>
               v{_esc(APP_VERSION)} on {_esc(now_str)}.</p>
            <p class="footer-notice">{_esc(PROPRIETARY_NOTICE)}</p>
        </div>
        """)

    # -----------------------------------------------------------------
    # Full HTML wrapper with embedded CSS
    # -----------------------------------------------------------------
    def _wrap_html(self, body_content: str) -> str:
        css = self._get_css()
        return textwrap.dedent(f"""\
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8"/>
            <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
            <title>{_esc(APP_NAME)} v{_esc(APP_VERSION)} &mdash; Analysis Report</title>
            <style>
        {css}
            </style>
        </head>
        <body>
        {body_content}
        </body>
        </html>
        """)

    # -----------------------------------------------------------------
    # CSS stylesheet (light/printable theme)
    # -----------------------------------------------------------------
    @staticmethod
    def _get_css() -> str:
        return textwrap.dedent("""\
        /* ---- Reset & Base ---- */
        *, *::before, *::after { box-sizing: border-box; }
        body {
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #1a1a2e;
            background: #ffffff;
            margin: 0;
            padding: 20px 40px;
        }

        /* ---- Header ---- */
        .header-block { text-align: center; margin-bottom: 10px; }
        .company-name {
            font-size: 16pt; font-weight: 700; color: #1a1a2e;
            letter-spacing: 1px; margin-bottom: 2px;
        }
        .team-name {
            font-size: 11pt; color: #555; margin-bottom: 14px;
        }
        .notice-box {
            border: 2px solid #c0392b;
            background: #fdf2f2;
            color: #6b1a1a;
            padding: 10px 16px;
            margin: 8px auto;
            max-width: 760px;
            font-size: 9pt;
            line-height: 1.4;
            text-align: left;
        }
        .export-notice {
            border-color: #d4a017;
            background: #fefbe8;
            color: #5c4a00;
        }
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
            background: #f0f4fa;
            border: 1px solid #c8d6e5;
            border-radius: 4px;
            padding: 12px 24px;
            margin: 12px 0 20px 0;
            max-width: 500px;
        }
        .toc h2 { font-size: 13pt; margin: 0 0 8px 0; color: #1a365d; }
        .toc ol { margin: 0; padding-left: 20px; }
        .toc li { margin: 3px 0; font-size: 10.5pt; }
        .toc a { color: #2563eb; text-decoration: none; }
        .toc a:hover { text-decoration: underline; }

        /* ---- Sections ---- */
        .section {
            margin: 24px 0;
        }
        .section h2 {
            font-size: 15pt; color: #1a365d;
            border-bottom: 2px solid #2563eb;
            padding-bottom: 4px; margin-bottom: 12px;
        }
        .section h3 {
            font-size: 12pt; color: #1e3a5f; margin: 14px 0 6px 0;
        }

        /* ---- Tables (general) ---- */
        .table-wrapper { overflow-x: auto; }
        table {
            border-collapse: collapse; width: 100%;
            margin: 8px 0 14px 0; font-size: 10pt;
        }
        th, td {
            border: 1px solid #b0bec5;
            padding: 5px 10px; text-align: left;
            vertical-align: top;
        }
        th {
            background: #1a365d; color: #fff;
            font-weight: 600; white-space: nowrap;
        }
        tbody tr:nth-child(even) { background: #f5f7fa; }
        tbody tr:hover { background: #e8edf5; }
        .label-cell { font-weight: 600; white-space: nowrap; width: 260px; }

        /* Budget table specifics */
        .subtotal-row td {
            background: #e2e8f0 !important;
            font-weight: 700; font-style: italic;
        }
        .total-row td {
            background: #1a365d !important;
            color: #fff; font-weight: 700;
        }

        /* Audit table */
        .audit-table .ts-cell { white-space: nowrap; width: 150px; font-family: monospace; font-size: 9pt; }
        .audit-table .action-cell { width: 130px; font-weight: 600; }
        .audit-table .details-cell { font-size: 9pt; color: #555; }

        /* ---- Verdicts / highlights ---- */
        .verdict {
            padding: 10px 14px; border-radius: 4px;
            margin: 10px 0; font-size: 10.5pt;
        }
        .verdict.pass { background: #d4edda; border-left: 5px solid #28a745; color: #155724; }
        .verdict.fail { background: #f8d7da; border-left: 5px solid #dc3545; color: #721c24; }
        .verdict.neutral { background: #e2e3e5; border-left: 5px solid #6c757d; color: #383d41; }
        .highlight {
            background: #fff3cd; border-left: 5px solid #ffc107;
            padding: 8px 12px; margin: 10px 0; font-size: 10.5pt;
        }

        /* ---- Misc ---- */
        .figure-container {
            text-align: center; margin: 14px 0;
            page-break-inside: avoid;
        }
        .figure-container img {
            border: 1px solid #dee2e6; border-radius: 3px;
        }
        .findings-block, .assumptions-block {
            background: #f8f9fa; border: 1px solid #dee2e6;
            border-radius: 4px; padding: 12px 16px;
            white-space: pre-wrap; font-size: 10.5pt;
            line-height: 1.6;
        }
        .footer-block {
            text-align: center; font-size: 9.5pt; color: #666;
            margin-top: 10px;
        }
        .footer-notice {
            font-size: 8pt; color: #999; max-width: 700px;
            margin: 4px auto 0 auto;
        }

        /* ---- Print ---- */
        @media print {
            body { padding: 10px 20px; font-size: 10pt; }
            .page-break-before { page-break-before: always; }
            .toc { page-break-after: always; }
            .notice-box { border-width: 1px; }
            table { font-size: 9pt; }
            .figure-container { page-break-inside: avoid; }
            a { color: #000; text-decoration: none; }
            .verdict.pass { border-left: 3px solid #28a745; }
            .verdict.fail { border-left: 3px solid #dc3545; }
        }
        """)


# =============================================================================
# SECTION 15: PROJECT SAVE/LOAD SYSTEM
# =============================================================================

class ProjectManager:
    """Handles saving and loading complete project state to/from disk."""

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Sanitize a project name for safe filesystem use."""
        # Replace characters not safe for file/folder names
        keepchars = (' ', '_', '-', '.')
        sanitized = ''.join(c if (c.isalnum() or c in keepchars) else '_' for c in name)
        # Collapse multiple underscores/spaces and strip
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        sanitized = sanitized.strip(' _.')
        if not sanitized:
            sanitized = "Untitled_Project"
        return sanitized

    @staticmethod
    def save_project(
        folder_path: str,
        project_name: str,
        comp_data: 'ComparisonData',
        sources: List['UncertaintySource'],
        settings: 'AnalysisSettings',
        rss_results: 'RSSResults',
        mc_results: 'MCResults',
        audit_log_instance: 'AuditLog',
        html_report_content: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Save the complete project state to a dated folder.

        Parameters
        ----------
        folder_path : str
            Parent directory where the project folder will be created.
        project_name : str
            Human-readable project name (will be sanitized).
        comp_data : ComparisonData
            Comparison error data.
        sources : list of UncertaintySource
            All uncertainty sources.
        settings : AnalysisSettings
            Current analysis settings.
        rss_results : RSSResults
            RSS analysis results.
        mc_results : MCResults
            Monte Carlo analysis results.
        audit_log_instance : AuditLog
            The audit log to save.
        html_report_content : str, optional
            If provided, the HTML report string is saved alongside.

        Returns
        -------
        str
            Path to the created project folder, or an error message
            prefixed with "ERROR: ".
        """
        try:
            safe_name = ProjectManager._sanitize_name(project_name)
            # If folder_path already contains a JSON with this project name,
            # re-use it directly (re-save). Otherwise create a dated subfolder.
            existing_json = os.path.join(folder_path, f"{safe_name}.json")
            if os.path.isfile(existing_json):
                project_dir = folder_path
            else:
                date_str = datetime.datetime.now().strftime("%Y-%m-%d")
                folder_name = f"{safe_name}_{date_str}"
                project_dir = os.path.join(folder_path, folder_name)
            os.makedirs(project_dir, exist_ok=True)

            # ---- Build the master JSON payload ----
            payload: Dict[str, Any] = {
                'app_version': APP_VERSION,
                'save_date': datetime.datetime.now().isoformat(),
                'project_name': project_name,
                'comparison_data': comp_data.to_dict(),
                'sources': [s.to_dict() for s in sources],
                'settings': settings.to_dict(),
                'project_metadata': kwargs.get('project_metadata', {}),
            }

            # RSS results — serialize only JSON-safe scalar fields
            def _sanitize_for_json(obj):
                """Recursively convert inf/nan floats to string representations."""
                if isinstance(obj, float):
                    if np.isnan(obj):
                        return "nan"
                    if np.isinf(obj):
                        return "-inf" if obj < 0 else "inf"
                    return obj
                if isinstance(obj, (np.floating,)):
                    return _sanitize_for_json(float(obj))
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, np.bool_):
                    return bool(obj)
                if isinstance(obj, dict):
                    return {k: _sanitize_for_json(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_sanitize_for_json(v) for v in obj]
                return obj

            if rss_results and rss_results.computed:
                rss_dict: Dict[str, Any] = {}
                for key, val in rss_results.__dict__.items():
                    rss_dict[key] = _sanitize_for_json(val)
                payload['rss_results'] = rss_dict

            # MC results — use its own to_dict (already skips samples array)
            # Sanitize for JSON-safe inf/nan handling
            if mc_results and mc_results.computed:
                payload['mc_results'] = _sanitize_for_json(mc_results.to_dict())

            # Audit log entries
            payload['audit_log'] = audit_log_instance.to_dict()

            # ---- Write JSON file (with numpy-safe encoder) ----
            class _NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, (np.floating,)):
                        if np.isnan(obj):
                            return "nan"
                        if np.isinf(obj):
                            return "-inf" if obj < 0 else "inf"
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.bool_):
                        return bool(obj)
                    return super().default(obj)

            json_path = os.path.join(project_dir, f"{safe_name}.json")
            with open(json_path, 'w', encoding='utf-8') as fh:
                json.dump(payload, fh, indent=2, ensure_ascii=False,
                          cls=_NumpyEncoder)

            # ---- Write audit log text file ----
            audit_txt_path = os.path.join(project_dir, f"{safe_name}_AuditLog.txt")
            with open(audit_txt_path, 'w', encoding='utf-8') as fh:
                fh.write(audit_log_instance.export_text())

            # ---- Optionally write HTML report ----
            if html_report_content:
                report_path = os.path.join(project_dir, f"{safe_name}_Report.html")
                with open(report_path, 'w', encoding='utf-8') as fh:
                    fh.write(html_report_content)

            return project_dir

        except Exception as exc:
            return f"ERROR: Failed to save project — {exc}"

    @staticmethod
    def load_project(json_path: str) -> dict:
        """
        Load a previously saved project from its JSON file.

        Parameters
        ----------
        json_path : str
            Full path to the project .json file.

        Returns
        -------
        dict
            Dictionary with keys:
                'comparison_data'  : ComparisonData instance
                'sources'          : list of UncertaintySource
                'settings'         : AnalysisSettings instance
                'audit_entries'    : list of dict (raw audit log entries)
                'app_version'      : str
                'save_date'        : str
            On failure, returns a dict with a single key 'error' containing
            the error message string.
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as fh:
                payload = json.load(fh)

            comp_data = ComparisonData.from_dict(payload.get('comparison_data', {}))

            sources = [
                UncertaintySource.from_dict(sd)
                for sd in payload.get('sources', [])
            ]

            settings = AnalysisSettings.from_dict(payload.get('settings', {}))

            audit_entries = payload.get('audit_log', [])

            return {
                'comparison_data': comp_data,
                'sources': sources,
                'settings': settings,
                'audit_entries': audit_entries,
                'app_version': payload.get('app_version', 'unknown'),
                'save_date': payload.get('save_date', 'unknown'),
                'project_metadata': payload.get('project_metadata', {}),
            }

        except Exception as exc:
            return {'error': f"Failed to load project — {exc}"}

    @staticmethod
    def get_save_folder_dialog(parent=None) -> str:
        """Open a folder-selection dialog for choosing a save location.

        Returns
        -------
        str
            Selected folder path, or empty string if cancelled.
        """
        folder = QFileDialog.getExistingDirectory(
            parent,
            "Select Folder to Save Project",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks,
        )
        return folder if folder else ""

    @staticmethod
    def get_load_file_dialog(parent=None) -> str:
        """Open a file-selection dialog for picking a .json project file.

        Returns
        -------
        str
            Selected file path, or empty string if cancelled.
        """
        path, _ = QFileDialog.getOpenFileName(
            parent,
            "Open VVUQ Project File",
            os.path.expanduser("~"),
            "VVUQ Project Files (*.json);;All Files (*)",
        )
        return path if path else ""


# =============================================================================
# SECTION 16: SYNTHETIC DEMO DATA
# =============================================================================

def generate_synthetic_demo() -> Tuple['ComparisonData', List['UncertaintySource'], 'AnalysisSettings']:
    """
    Generate a complete synthetic demonstration dataset.

    Returns a tuple of (ComparisonData, list of UncertaintySource,
    AnalysisSettings) populated with realistic fictional data suitable
    for exercising the full RSS and Monte Carlo analysis pipeline.

    Returns
    -------
    tuple
        (ComparisonData, List[UncertaintySource], AnalysisSettings)
    """

    # ---- Comparison Data ----
    n_locations = 8
    n_conditions = 5
    sensor_names = [f"TC-{i+1:02d}" for i in range(n_locations)]
    condition_names = [f"FC-{j+1:03d}" for j in range(n_conditions)]

    rng = np.random.default_rng(42)
    base_errors = rng.normal(loc=5.0, scale=6.0, size=(n_locations, n_conditions))

    # Per-location offsets to introduce systematic biases
    location_offsets = np.array([-2, -1, 0, 1, 2, -1, 0, 1], dtype=float)
    for i in range(n_locations):
        base_errors[i, :] += location_offsets[i]

    comp_data = ComparisonData(
        data=base_errors,
        sensor_names=sensor_names,
        condition_names=condition_names,
        unit="°F",
        is_pooled=True,
    )

    # ---- Uncertainty Sources ----
    sources: List[UncertaintySource] = []

    # 1. Grid Convergence — Numerical
    sources.append(UncertaintySource(
        name="Grid Convergence",
        category="Numerical (u_num)",
        input_type="Sigma Value Only",
        distribution="Normal",
        sigma_basis="Confirmed 1σ",
        sigma_value=1.5,
        raw_sigma_value=1.5,
        mean_value=0.0,
        sample_size=120,
        dof=119.0,
        is_supplier=False,
        unit="°F",
        notes="",
        enabled=True,
    ))

    # 2. Time Step Sensitivity — Numerical (supplier)
    sources.append(UncertaintySource(
        name="Time Step Sensitivity",
        category="Numerical (u_num)",
        input_type="Sigma Value Only",
        distribution="Normal",
        sigma_basis="Confirmed 1σ",
        sigma_value=0.8,
        raw_sigma_value=0.8,
        mean_value=0.0,
        sample_size=0,
        dof=float('inf'),
        is_supplier=True,
        unit="°F",
        notes="",
        enabled=True,
    ))

    # 3. Inlet BC Temperature — Input/BC
    sources.append(UncertaintySource(
        name="Inlet BC Temperature",
        category="Input/BC (u_input)",
        input_type="Sigma Value Only",
        distribution="Normal",
        sigma_basis="Confirmed 1σ",
        sigma_value=2.0,
        raw_sigma_value=2.0,
        mean_value=0.0,
        sample_size=5,
        dof=4.0,
        is_supplier=False,
        unit="°F",
        notes="",
        enabled=True,
    ))

    # 4. Thermocouple Accuracy — Experimental (supplier, 2-sigma spec)
    sources.append(UncertaintySource(
        name="Thermocouple Accuracy",
        category="Experimental (u_D)",
        input_type="Sigma Value Only",
        distribution="Normal",
        sigma_basis="2σ (95%)",
        sigma_value=1.8,
        raw_sigma_value=1.8,
        mean_value=0.0,
        sample_size=0,
        dof=float('inf'),
        is_supplier=True,
        unit="°F",
        notes="Manufacturer spec sheet",
        enabled=True,
    ))

    # 5. Data Acquisition Noise — Experimental
    sources.append(UncertaintySource(
        name="Data Acquisition Noise",
        category="Experimental (u_D)",
        input_type="Sigma Value Only",
        distribution="Normal",
        sigma_basis="Confirmed 1σ",
        sigma_value=0.5,
        raw_sigma_value=0.5,
        mean_value=0.0,
        sample_size=1000,
        dof=999.0,
        is_supplier=False,
        unit="°F",
        notes="",
        enabled=True,
    ))

    # ---- Analysis Settings ----
    settings = AnalysisSettings(
        coverage=0.95,
        confidence=0.95,
        one_sided=True,
        k_method=K_METHOD_VV20,
        global_unit="°F",
        mc_n_trials=100000,
        mc_bootstrap=True,
        bound_type="Both (for comparison)",
    )

    return comp_data, sources, settings


def generate_advanced_demo() -> Tuple['ComparisonData', List['UncertaintySource'], 'AnalysisSettings']:
    """
    Generate an advanced demonstration dataset exercising non-Normal
    distributions, asymmetric uncertainties, and correlated sources.

    The comparison data is constructed with a deliberate systematic bias
    of ~15 degF (CFD hotter than test) so the validation metric will show
    a **NOT VALIDATED** result, illustrating what failure looks like.

    Returns
    -------
    tuple
        (ComparisonData, List[UncertaintySource], AnalysisSettings)
    """

    # ---- Comparison Data (6 locations x 4 conditions) ----
    n_locations = 6
    n_conditions = 4

    sensor_names = [
        "LE-Stag",      # leading-edge stagnation point
        "Suc-25%",      # suction side 25% chord
        "Suc-50%",      # suction side 50% chord
        "Suc-75%",      # suction side 75% chord
        "Prs-50%",      # pressure side 50% chord
        "TE-Slot",      # trailing-edge cooling slot
    ]
    condition_names = [
        "Cruise-Lo",    # cruise, low-power
        "Cruise-Hi",    # cruise, high-power
        "Climb",        # max-climb
        "Takeoff",      # takeoff / max-power
    ]

    # Build a repeatable comparison-error matrix with ~15 degF mean bias.
    # Values represent E = T_CFD - T_test (positive = CFD hotter).
    rng = np.random.default_rng(2024)

    # The *errors* vary around +15 degF with realistic scatter.
    base_bias = 15.0  # systematic CFD over-prediction
    location_scatter = np.array([1.0, -0.5, 0.2, -0.8, 0.6, -0.3])
    condition_scatter = np.array([-1.0, 0.5, 1.2, -0.7])

    errors = np.empty((n_locations, n_conditions), dtype=float)
    for i in range(n_locations):
        for j in range(n_conditions):
            errors[i, j] = (base_bias
                            + location_scatter[i]
                            + condition_scatter[j]
                            + rng.normal(0.0, 2.0))

    comp_data = ComparisonData(
        data=errors,
        sensor_names=sensor_names,
        condition_names=condition_names,
        unit="\u00b0F",
        is_pooled=True,
    )

    # ---- Uncertainty Sources ----
    sources: List[UncertaintySource] = []

    # 1. Grid convergence -- standard Normal numerical uncertainty
    sources.append(UncertaintySource(
        name="Grid convergence u_num",
        category="Numerical (u_num)",
        input_type="Sigma Value Only",
        distribution="Normal",
        sigma_basis="Confirmed 1\u03c3",
        sigma_value=2.5,
        raw_sigma_value=2.5,
        mean_value=0.0,
        sample_size=60,
        dof=59.0,
        is_supplier=False,
        unit="\u00b0F",
        notes="Richardson extrapolation on 3-grid sequence",
        enabled=True,
    ))

    # 2. Inlet temperature uncertainty -- asymmetric (upper > lower)
    sources.append(UncertaintySource(
        name="Inlet temperature uncertainty",
        category="Input/BC (u_input)",
        input_type="Sigma Value Only",
        distribution="Normal",
        sigma_basis="Confirmed 1\u03c3",
        sigma_value=0.0,           # not used when asymmetric=True
        raw_sigma_value=0.0,
        mean_value=0.0,
        sample_size=30,
        dof=29.0,
        is_supplier=False,
        unit="\u00b0F",
        notes="Thermocouple rake; asymmetric due to radiation correction",
        enabled=True,
        asymmetric=True,
        sigma_upper=3.0,
        sigma_lower=1.5,
    ))

    # 3a. Thermocouple bias A -- correlated with 3b
    sources.append(UncertaintySource(
        name="Thermocouple bias A",
        category="Experimental (u_D)",
        input_type="Sigma Value Only",
        distribution="Normal",
        sigma_basis="Confirmed 1\u03c3",
        sigma_value=1.2,
        raw_sigma_value=1.2,
        mean_value=0.0,
        sample_size=50,
        dof=49.0,
        is_supplier=False,
        unit="\u00b0F",
        notes="Common-lot TC, correlated with TC bias B",
        enabled=True,
        correlation_group="TC_bias",
        correlation_coefficient=0.8,
    ))

    # 3b. Thermocouple bias B -- correlated with 3a
    sources.append(UncertaintySource(
        name="Thermocouple bias B",
        category="Experimental (u_D)",
        input_type="Sigma Value Only",
        distribution="Normal",
        sigma_basis="Confirmed 1\u03c3",
        sigma_value=1.4,
        raw_sigma_value=1.4,
        mean_value=0.0,
        sample_size=50,
        dof=49.0,
        is_supplier=False,
        unit="\u00b0F",
        notes="Common-lot TC, correlated with TC bias A",
        enabled=True,
        correlation_group="TC_bias",
        correlation_coefficient=0.8,
    ))

    # 4. Positioning uncertainty -- Uniform distribution
    sources.append(UncertaintySource(
        name="Positioning uncertainty",
        category="Experimental (u_D)",
        input_type="Sigma Value Only",
        distribution="Uniform",
        sigma_basis="Confirmed 1\u03c3",
        sigma_value=1.0,
        raw_sigma_value=1.0,
        mean_value=0.0,
        sample_size=0,
        dof=float('inf'),
        is_supplier=True,
        unit="\u00b0F",
        notes="TC bead location on airfoil surface; +/-0.020 in temperature effect",
        enabled=True,
    ))

    # 5. Material property scatter -- Lognormal distribution
    sources.append(UncertaintySource(
        name="Material property scatter",
        category="Input/BC (u_input)",
        input_type="Sigma Value Only",
        distribution="Lognormal",
        sigma_basis="Confirmed 1\u03c3",
        sigma_value=2.0,
        raw_sigma_value=2.0,
        mean_value=0.0,
        sample_size=15,
        dof=14.0,
        is_supplier=False,
        unit="\u00b0F",
        notes="TBC thermal conductivity lot-to-lot variation",
        enabled=True,
    ))

    # ---- Analysis Settings ----
    settings = AnalysisSettings(
        coverage=0.95,
        confidence=0.95,
        one_sided=True,
        k_method=K_METHOD_VV20,
        global_unit="\u00b0F",
        mc_n_trials=100000,
        mc_bootstrap=True,
        bound_type="Both (for comparison)",
    )

    return comp_data, sources, settings


# =============================================================================
# SECTION 17: MAIN APPLICATION WINDOW + ENTRY POINT
# =============================================================================

class MainWindow(QMainWindow):
    """
    Main application window for the VVUQ Uncertainty Aggregator.

    Hosts all seven analysis tabs, the menu bar, status bar, and orchestrates
    data flow and computation triggers between tabs.
    """

    def __init__(self):
        super().__init__()

        # ---- Window setup ----
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1200, 800)
        self.setStyleSheet(get_dark_stylesheet())

        # ---- Application font ----
        font = QFont()
        for family in FONT_FAMILIES:
            if QFontDatabase.hasFamily(family):
                font.setFamily(family)
                break
        font.setPointSize(10)
        QApplication.instance().setFont(font)

        # ---- Project state ----
        self._project_folder: str = ""
        self._project_name: str = "Untitled"
        self._unsaved_changes: bool = False

        # ---- Central widget: project info bar + tab widget ----
        central = QWidget()
        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.setCentralWidget(central)

        # Project info / notes bar (collapsible)
        self._project_bar = QFrame()
        self._project_bar.setStyleSheet(
            f"QFrame {{ background-color: {DARK_COLORS['bg_alt']}; "
            f"border-bottom: 1px solid {DARK_COLORS['border']}; }}"
        )
        bar_layout = QVBoxLayout(self._project_bar)
        bar_layout.setContentsMargins(12, 6, 12, 6)
        bar_layout.setSpacing(4)

        # Toggle button row
        bar_top = QHBoxLayout()
        bar_top.setSpacing(8)
        self._btn_toggle_info = QPushButton("▶ Project Info")
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

        # Project name display (always visible)
        self._lbl_project_name = QLabel("Untitled")
        self._lbl_project_name.setStyleSheet(
            f"color: {DARK_COLORS['fg']}; font-size: 12px; font-weight: bold;"
        )
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
            f"border-radius: 3px; padding: 3px 6px; font-size: 11px;"
        )

        detail_layout.addWidget(
            self._make_label("Program / Project:", lbl_style), 0, 0)
        self._edit_program = QLineEdit()
        self._edit_program.setStyleSheet(val_style)
        self._edit_program.setPlaceholderText("e.g., XYZ Flight Test Campaign")
        self._edit_program.textChanged.connect(self._mark_unsaved)
        detail_layout.addWidget(self._edit_program, 0, 1)

        detail_layout.addWidget(
            self._make_label("Analyst:", lbl_style), 0, 2)
        self._edit_analyst = QLineEdit()
        self._edit_analyst.setStyleSheet(val_style)
        self._edit_analyst.setPlaceholderText("e.g., J. Smith")
        self._edit_analyst.textChanged.connect(self._mark_unsaved)
        detail_layout.addWidget(self._edit_analyst, 0, 3)

        detail_layout.addWidget(
            self._make_label("Date:", lbl_style), 0, 4)
        self._edit_date = QLineEdit()
        self._edit_date.setStyleSheet(val_style)
        self._edit_date.setText(datetime.datetime.now().strftime("%Y-%m-%d"))
        self._edit_date.textChanged.connect(self._mark_unsaved)
        detail_layout.addWidget(self._edit_date, 0, 5)

        detail_layout.addWidget(
            self._make_label("Decision Consequence:", lbl_style), 1, 0)
        self._cmb_consequence = QComboBox()
        self._cmb_consequence.addItems(["Low", "Medium", "High"])
        self._cmb_consequence.setCurrentText("Medium")
        self._cmb_consequence.setStyleSheet(val_style)
        self._cmb_consequence.currentTextChanged.connect(self._mark_unsaved)
        detail_layout.addWidget(self._cmb_consequence, 1, 1)

        detail_layout.addWidget(
            self._make_label("Notes:", lbl_style), 2, 0)
        self._edit_notes = QTextEdit()
        self._edit_notes.setStyleSheet(
            f"QTextEdit {{ {val_style} }}"
        )
        self._edit_notes.setPlaceholderText(
            "Free-form notes: describe the analysis objective, flight conditions, "
            "model version, key assumptions, reviewer comments, etc."
        )
        self._edit_notes.setMaximumHeight(80)
        self._edit_notes.textChanged.connect(self._mark_unsaved)
        detail_layout.addWidget(self._edit_notes, 2, 1, 1, 5)

        # Column stretch: fields get more space than labels
        detail_layout.setColumnStretch(1, 3)
        detail_layout.setColumnStretch(3, 2)
        detail_layout.setColumnStretch(5, 2)

        bar_layout.addWidget(self._project_detail_frame)
        self._project_detail_frame.setVisible(False)
        self._project_info_visible = False

        central_layout.addWidget(self._project_bar)

        self._tabs = QTabWidget()
        central_layout.addWidget(self._tabs)

        self._tab_comparison = ComparisonDataTab()
        self._tab_sources = UncertaintySourcesTab()
        self._tab_settings = AnalysisSettingsTab()
        self._tab_rss = RSSResultsTab()
        self._tab_mc = MonteCarloResultsTab()
        self._tab_rollup = ComparisonRollUpTab()
        self._tab_reference = ReferenceTab()

        self._tabs.addTab(self._tab_comparison, "\U0001f4ca Comparison Data")
        self._tabs.addTab(self._tab_sources, "\U0001f4cb Uncertainty Sources")
        self._tabs.addTab(self._tab_settings, "\u2699\ufe0f Analysis Settings")
        self._tabs.addTab(self._tab_rss, "\U0001f4c8 Results \u2014 RSS")
        self._tabs.addTab(self._tab_mc, "\U0001f3b2 Results \u2014 Monte Carlo")
        self._tabs.addTab(self._tab_rollup, "\U0001f4d1 Comparison Roll-Up")
        self._tabs.addTab(self._tab_reference, "\U0001f4d6 Reference")

        # ---- Menu bar ----
        self._build_menu_bar()

        # ---- Status bar ----
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._version_label = QLabel(
            f"{APP_NAME} v{APP_VERSION} ({APP_DATE})"
        )
        self._version_label.setStyleSheet(
            f"color: {DARK_COLORS['fg_dim']}; font-size: 11px; padding-right: 8px;"
        )
        self._status_bar.addPermanentWidget(self._version_label)
        self._status_bar.showMessage("Ready.", 5000)

        # ---- Signal wiring ----
        self._wire_signals()

        audit_log.log("UI_INIT", "Main window initialized.")

    # =================================================================
    # PROJECT INFO BAR HELPERS
    # =================================================================
    @staticmethod
    def _make_label(text, style):
        lbl = QLabel(text)
        lbl.setStyleSheet(style)
        return lbl

    def _toggle_project_info(self):
        self._project_info_visible = not self._project_info_visible
        self._project_detail_frame.setVisible(self._project_info_visible)
        arrow = "▼" if self._project_info_visible else "▶"
        self._btn_toggle_info.setText(f"{arrow} Project Info")

    def _mark_unsaved(self):
        self._unsaved_changes = True

    def get_project_metadata(self) -> dict:
        """Return the project-level metadata as a dict."""
        return {
            'program': self._edit_program.text().strip(),
            'analyst': self._edit_analyst.text().strip(),
            'date': self._edit_date.text().strip(),
            'notes': self._edit_notes.toPlainText().strip(),
            'decision_consequence': self._cmb_consequence.currentText().strip(),
        }

    def set_project_metadata(self, meta: dict):
        """Restore project-level metadata from a dict."""
        self._edit_program.setText(meta.get('program', ''))
        self._edit_analyst.setText(meta.get('analyst', ''))
        self._edit_date.setText(meta.get('date', ''))
        self._edit_notes.setPlainText(meta.get('notes', ''))
        dc = normalize_decision_consequence(meta.get('decision_consequence', 'Medium'))
        idx = self._cmb_consequence.findText(dc)
        if idx >= 0:
            self._cmb_consequence.setCurrentIndex(idx)

    # =================================================================
    # MENU BAR
    # =================================================================
    def _build_menu_bar(self):
        menu_bar = self.menuBar()

        # ---- File menu ----
        file_menu = menu_bar.addMenu("&File")

        act_new = QAction("&New Project", self)
        act_new.setShortcut(QKeySequence("Ctrl+N"))
        act_new.triggered.connect(self._new_project)
        file_menu.addAction(act_new)

        act_open = QAction("&Open Project...", self)
        act_open.setShortcut(QKeySequence("Ctrl+O"))
        act_open.triggered.connect(self._open_project)
        file_menu.addAction(act_open)

        act_save = QAction("&Save Project", self)
        act_save.setShortcut(QKeySequence("Ctrl+S"))
        act_save.triggered.connect(self._save_project)
        file_menu.addAction(act_save)

        act_save_as = QAction("Save Project &As...", self)
        act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        act_save_as.triggered.connect(self._save_project_as)
        file_menu.addAction(act_save_as)

        file_menu.addSeparator()

        act_export = QAction("&Export HTML Report...", self)
        act_export.setShortcut(QKeySequence("Ctrl+H"))
        act_export.triggered.connect(self._export_html_report)
        file_menu.addAction(act_export)

        act_manifest = QAction("Export &Reproducibility Manifest...", self)
        act_manifest.triggered.connect(self._export_reproducibility_manifest)
        file_menu.addAction(act_manifest)

        file_menu.addSeparator()

        act_exit = QAction("E&xit", self)
        act_exit.setShortcut(QKeySequence("Alt+F4"))
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        # ---- Analysis menu ----
        analysis_menu = menu_bar.addMenu("&Analysis")

        act_rss = QAction("Compute &RSS", self)
        act_rss.setShortcut(QKeySequence("Ctrl+R"))
        act_rss.triggered.connect(self._auto_compute_rss)
        analysis_menu.addAction(act_rss)

        act_mc = QAction("Run &Monte Carlo", self)
        act_mc.setShortcut(QKeySequence("Ctrl+M"))
        act_mc.triggered.connect(self._run_monte_carlo)
        analysis_menu.addAction(act_mc)

        act_all = QAction("Compute &All", self)
        act_all.setShortcut(QKeySequence("Ctrl+Shift+A"))
        act_all.triggered.connect(self._compute_all)
        analysis_menu.addAction(act_all)

        # ---- Tools menu ----
        tools_menu = menu_bar.addMenu("&Tools")

        act_example = QAction("&Load Example Data", self)
        act_example.triggered.connect(self._load_example_data)
        tools_menu.addAction(act_example)

        act_adv_example = QAction("Load &Advanced Example", self)
        act_adv_example.triggered.connect(self._load_advanced_example_data)
        tools_menu.addAction(act_adv_example)

        act_clear = QAction("&Clear All Data", self)
        act_clear.triggered.connect(self._clear_all)
        tools_menu.addAction(act_clear)

        # ---- Help menu ----
        help_menu = menu_bar.addMenu("&Help")

        act_about = QAction("&About", self)
        act_about.triggered.connect(self._show_about)
        help_menu.addAction(act_about)

        act_ref = QAction("Show &Reference Tab", self)
        act_ref.triggered.connect(lambda: self._tabs.setCurrentWidget(self._tab_reference))
        help_menu.addAction(act_ref)

    # =================================================================
    # SIGNAL WIRING
    # =================================================================
    def _wire_signals(self):
        """Connect tab signals to main window slots with debouncing."""
        # Debounced auto-compute on data/source/settings changes
        self._tab_comparison.data_changed.connect(
            lambda: QTimer.singleShot(100, self._auto_compute_rss)
        )
        self._tab_sources.sources_changed.connect(
            lambda: QTimer.singleShot(100, self._auto_compute_rss)
        )
        self._tab_settings.settings_changed.connect(
            lambda: QTimer.singleShot(100, self._auto_compute_rss)
        )

        # MC finished
        self._tab_mc.mc_finished.connect(self._on_mc_finished)

        # MC tab "Run Monte Carlo" button → main window handler
        self._tab_mc._btn_run.clicked.connect(self._run_monte_carlo)

        # Roll-up export/save
        self._tab_rollup.export_requested.connect(self._export_html_report)
        self._tab_rollup.save_requested.connect(self._save_project)

    # =================================================================
    # COMPUTATION SLOTS
    # =================================================================
    def _auto_compute_rss(self):
        """Gather inputs, validate, compute RSS, and update downstream tabs."""
        try:
            sources = self._tab_sources.get_sources()
            comp_data = self._tab_comparison.get_comparison_data()
            settings = self._tab_settings.get_settings()

            # Sync unit from comparison data tab into settings
            settings.global_unit = comp_data.unit

            # Validate: need at least one source with positive uncertainty
            valid_sources = [
                s for s in sources if s.get_standard_uncertainty() > 0
            ]
            if not valid_sources:
                self._tab_rss.clear_results()
                self._tab_rollup.clear_results()
                self._status_bar.showMessage(
                    "No valid uncertainty sources — results cleared.", 5000
                )
                return

            # Unit consistency check
            mismatched = [
                s.name for s in valid_sources
                if s.unit and s.unit != settings.global_unit
            ]
            if mismatched:
                audit_log.log_computation(
                    "UNIT_MISMATCH",
                    f"Sources with unit mismatch vs global '{settings.global_unit}': "
                    f"{', '.join(mismatched)}"
                )
                self._status_bar.showMessage(
                    f"Warning: {len(mismatched)} source(s) have different "
                    f"units than global setting ({settings.global_unit}).",
                    10000,
                )
                self._tab_rss.show_unit_mismatch(mismatched, settings.global_unit)
            else:
                self._tab_rss.hide_unit_mismatch()

            # Compute RSS
            self._tab_rss.compute_and_display(sources, comp_data, settings)
            rss = self._tab_rss.get_results()

            # Update k-display on settings tab
            self._tab_settings.update_k_display(rss.nu_eff, rss.k_factor)

            # Update roll-up tab
            mc = self._tab_mc.get_results()
            self._tab_rollup.update_rollup(
                rss, mc, comp_data, settings,
                self._tab_sources.get_all_sources()
            )

            self._unsaved_changes = True
            self._status_bar.showMessage(
                f"RSS computed.  u_val = {rss.u_val:.4f},  k = {rss.k_factor:.4f}",
                10000,
            )

        except Exception as exc:
            self._status_bar.showMessage(f"RSS error: {exc}", 8000)
            audit_log.log_warning(f"RSS computation failed: {exc}")
            traceback.print_exc()

    def _run_monte_carlo(self):
        """Launch Monte Carlo analysis in a background thread."""
        try:
            sources = self._tab_sources.get_sources()
            comp_data = self._tab_comparison.get_comparison_data()
            settings = self._tab_settings.get_settings()
            rss_results = self._tab_rss.get_results()

            valid_sources = [
                s for s in sources if s.get_standard_uncertainty() > 0
            ]
            if not valid_sources:
                self._status_bar.showMessage(
                    "No valid uncertainty sources for Monte Carlo.", 5000
                )
                return

            self._tab_mc.run_mc(sources, comp_data, settings, rss_results)
            self._status_bar.showMessage("Monte Carlo running...", 0)

            # Switch to MC tab so user can watch progress
            self._tabs.setCurrentWidget(self._tab_mc)

        except Exception as exc:
            self._status_bar.showMessage(f"MC launch error: {exc}", 8000)
            audit_log.log_warning(f"Monte Carlo launch failed: {exc}")
            traceback.print_exc()

    def _compute_all(self):
        """Run both RSS and Monte Carlo in sequence."""
        self._auto_compute_rss()
        self._run_monte_carlo()

    def _on_mc_finished(self):
        """Handle Monte Carlo completion: update roll-up and status."""
        try:
            mc = self._tab_mc.get_results()
            rss = self._tab_rss.get_results()
            comp_data = self._tab_comparison.get_comparison_data()
            settings = self._tab_settings.get_settings()
            all_sources = self._tab_sources.get_all_sources()

            self._tab_rollup.update_rollup(rss, mc, comp_data, settings, all_sources)
            self._unsaved_changes = True
            self._status_bar.showMessage("Monte Carlo complete.", 10000)

        except Exception as exc:
            self._status_bar.showMessage(f"MC post-processing error: {exc}", 8000)
            audit_log.log_warning(f"MC post-processing failed: {exc}")
            traceback.print_exc()

    # =================================================================
    # PROJECT MANAGEMENT SLOTS
    # =================================================================
    def _new_project(self):
        """Reset all tabs to a blank state after confirming with the user."""
        if self._unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Discard and start a new project?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        # Reset each tab
        self._tab_comparison.set_comparison_data(ComparisonData())
        self._tab_sources.set_sources([])
        self._tab_settings.set_settings(AnalysisSettings())
        self._tab_rss.clear_results()
        self._tab_mc.clear_results()
        self._tab_rollup.clear_results()

        # Reset project state
        self._project_folder = ""
        self._project_name = "Untitled"
        self._unsaved_changes = False
        self._lbl_project_name.setText("Untitled")
        self.set_project_metadata({})

        # Reset audit log
        audit_log.entries.clear()
        audit_log.log("SESSION_START", f"{APP_NAME} v{APP_VERSION} — New project")

        self._status_bar.showMessage("New project started.", 5000)
        self._tabs.setCurrentIndex(0)

    def _open_project(self):
        """Load a saved project from a JSON file."""
        json_path = ProjectManager.get_load_file_dialog(self)
        if not json_path:
            return

        result = ProjectManager.load_project(json_path)
        if 'error' in result:
            QMessageBox.critical(self, "Load Error", result['error'])
            return

        try:
            # Clear stale results before populating from loaded project
            self._tab_rss.clear_results()
            self._tab_mc.clear_results()
            self._tab_rollup.clear_results()

            # Block signals on sources/settings tabs to prevent triple
            # redundant _auto_compute_rss during bulk load.  The single
            # QTimer.singleShot(200, ...) below handles the recompute.
            self._tab_sources.blockSignals(True)
            self._tab_settings.blockSignals(True)
            try:
                self._tab_comparison.set_comparison_data(result['comparison_data'])
                self._tab_sources.set_sources(result['sources'])
                self._tab_settings.set_settings(result['settings'])
            finally:
                self._tab_sources.blockSignals(False)
                self._tab_settings.blockSignals(False)

            # Restore project metadata
            if result.get('project_metadata'):
                self.set_project_metadata(result['project_metadata'])

            # Restore audit log entries
            if result.get('audit_entries'):
                audit_log.from_dict(result['audit_entries'])

            self._project_folder = os.path.dirname(json_path)
            self._project_name = os.path.splitext(
                os.path.basename(json_path)
            )[0]
            self._lbl_project_name.setText(self._project_name)
            self._unsaved_changes = False

            audit_log.log("PROJECT_LOADED",
                          f"Loaded project from {json_path}",
                          f"Saved with v{result.get('app_version', '?')} "
                          f"on {result.get('save_date', '?')}")

            # Trigger auto-compute so results tabs are populated
            QTimer.singleShot(200, self._auto_compute_rss)

            self._status_bar.showMessage(
                f"Project loaded from {json_path}", 8000
            )
            self._tabs.setCurrentIndex(0)

        except Exception as exc:
            QMessageBox.critical(
                self, "Load Error",
                f"Failed to populate tabs from project file:\n{exc}"
            )
            audit_log.log_warning(f"Project load tab population failed: {exc}")
            traceback.print_exc()

    def _save_project(self):
        """Save the project to the current folder, prompting if needed."""
        if not self._project_folder:
            self._save_project_as()
            return

        self._do_save(self._project_folder)

    def _save_project_as(self):
        """Prompt the user for a save location, then save."""
        folder = ProjectManager.get_save_folder_dialog(self)
        if not folder:
            return

        # Ask for a project name
        name, ok = self._ask_project_name()
        if not ok:
            return

        # Save old values in case _do_save fails
        old_name = self._project_name
        old_folder = self._project_folder
        old_label = self._lbl_project_name.text()

        self._project_name = name
        self._project_folder = folder
        self._lbl_project_name.setText(name)
        self._do_save(folder)

        # If save failed, restore old values
        if self._unsaved_changes:
            self._project_name = old_name
            self._project_folder = old_folder
            self._lbl_project_name.setText(old_label)

    def _ask_project_name(self):
        """Prompt the user for a project name using an input dialog."""
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(
            self,
            "Project Name",
            "Enter a name for this project:",
            QLineEdit.Normal,
            self._project_name,
        )
        return name.strip() if name else "Untitled", ok

    def _do_save(self, folder: str):
        """Execute the actual project save."""
        try:
            # Generate HTML report content
            html_content = self._generate_html_report_content()

            rss = self._tab_rss.get_results()
            mc = self._tab_mc.get_results()
            comp_data = self._tab_comparison.get_comparison_data()
            settings = self._tab_settings.get_settings()
            all_sources = self._tab_sources.get_all_sources()

            result = ProjectManager.save_project(
                folder_path=folder,
                project_name=self._project_name,
                comp_data=comp_data,
                sources=all_sources,
                settings=settings,
                rss_results=rss,
                mc_results=mc,
                audit_log_instance=audit_log,
                html_report_content=html_content,
                project_metadata=self.get_project_metadata(),
            )

            if result.startswith("ERROR:"):
                QMessageBox.critical(self, "Save Error", result)
                return

            # Update project folder to the actual created directory (not parent)
            self._project_folder = result
            self._unsaved_changes = False
            audit_log.log("PROJECT_SAVED", f"Project saved to {result}")
            self._status_bar.showMessage(
                f"Project saved to {result}", 8000
            )

        except Exception as exc:
            QMessageBox.critical(
                self, "Save Error", f"Failed to save project:\n{exc}"
            )
            audit_log.log_warning(f"Project save failed: {exc}")
            traceback.print_exc()

    # =================================================================
    # HTML REPORT EXPORT
    # =================================================================
    def _generate_html_report_content(self) -> str:
        """Build the HTML report string from current state."""
        generator = HTMLReportGenerator()

        rss = self._tab_rss.get_results()
        mc = self._tab_mc.get_results()
        comp_data = self._tab_comparison.get_comparison_data()
        settings = self._tab_settings.get_settings()
        all_sources = self._tab_sources.get_all_sources()
        budget_table = self._tab_rss.get_budget_table_data()
        rollup_table = self._tab_rollup.get_rollup_table_data()
        rollup_headers = self._tab_rollup.get_rollup_header_labels()
        findings = self._tab_rollup.get_findings_text()
        assumptions = self._tab_rollup.get_assumptions_text()

        # Gather matplotlib figures from tabs
        figures: Dict[str, Figure] = {}
        try:
            if hasattr(self._tab_comparison, '_fig') and self._tab_comparison._fig is not None:
                figures['comparison_plots'] = self._tab_comparison._fig
        except Exception:
            pass
        try:
            if hasattr(self._tab_rss, '_fig') and self._tab_rss._fig is not None:
                figures['rss_plots'] = self._tab_rss._fig
        except Exception:
            pass
        try:
            if hasattr(self._tab_mc, '_fig') and self._tab_mc._fig is not None:
                figures['mc_plots'] = self._tab_mc._fig
        except Exception:
            pass

        # Use None for MC if not computed
        mc_for_report = mc if (mc and mc.computed) else None

        html = generator.generate_report(
            rss_results=rss,
            mc_results=mc_for_report,
            comp_data=comp_data,
            settings=settings,
            sources=all_sources,
            budget_table=budget_table,
            rollup_table=rollup_table,
            findings_text=findings,
            assumptions_text=assumptions,
            figures=figures,
            audit_entries=audit_log.to_dict(),
            rollup_headers=rollup_headers,
            project_metadata=self.get_project_metadata(),
        )
        return html

    def _export_html_report(self):
        """Export the HTML report to a user-chosen location."""
        try:
            html_content = self._generate_html_report_content()

            # Determine default directory
            default_dir = (
                self._project_folder
                if self._project_folder
                else os.path.expanduser("~")
            )
            default_name = os.path.join(
                default_dir,
                f"{ProjectManager._sanitize_name(self._project_name)}_Report.html",
            )

            filepath, _ = QFileDialog.getSaveFileName(
                self,
                "Export HTML Report",
                default_name,
                "HTML Files (*.html);;All Files (*)",
            )
            if not filepath:
                return

            generator = HTMLReportGenerator()
            generator.save_report(html_content, filepath)

            audit_log.log("REPORT_EXPORTED", f"HTML report exported to {filepath}")
            self._status_bar.showMessage(
                f"HTML report exported to {filepath}", 8000
            )

        except Exception as exc:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export HTML report:\n{exc}"
            )
            audit_log.log_warning(f"HTML report export failed: {exc}")
            traceback.print_exc()

    def _export_reproducibility_manifest(self):
        """Export a reproducibility manifest JSON for the current analysis."""
        import hashlib
        import platform as _platform
        import json as _json

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Reproducibility Manifest",
            os.path.join(
                self._project_folder or os.path.expanduser("~"),
                "reproducibility_manifest.json"
            ),
            "JSON Files (*.json);;All Files (*)"
        )
        if not filepath:
            return

        try:
            settings = self._tab_settings.get_settings()
            sources = self._tab_sources.get_sources()
            rss = self._tab_rss.get_results()
            mc = self._tab_mc.get_results()

            # Compute settings hash
            settings_str = _json.dumps(settings.to_dict(), sort_keys=True)
            settings_hash = hashlib.sha256(settings_str.encode()).hexdigest()

            manifest = {
                "tool_name": APP_NAME,
                "tool_version": APP_VERSION,
                "python_version": sys.version,
                "platform": _platform.platform(),
                "os": _platform.system(),
                "library_versions": {
                    "numpy": np.__version__,
                    "scipy": getattr(__import__('scipy'), '__version__', 'unknown'),
                    "matplotlib": getattr(__import__('matplotlib'), '__version__', 'unknown'),
                    "PySide6": getattr(__import__('PySide6'), '__version__', 'unknown'),
                },
                "analysis_date": datetime.datetime.now().isoformat(),
                "project_name": self._project_name,
                "settings_hash": settings_hash,
                "n_sources": len(sources),
                "n_enabled_sources": len([s for s in sources if s.enabled]),
                "n_data_points": rss.n_data if rss and rss.computed else 0,
                "rss_computed": rss.computed if rss else False,
                "mc_computed": mc.computed if mc else False,
                "mc_n_trials": mc.n_trials if mc and mc.computed else 0,
                "random_seed": getattr(settings, 'mc_seed', None),
                "method_used": "RSS + MC" if (rss and rss.computed and mc and mc.computed) else (
                    "RSS" if rss and rss.computed else "None"
                ),
            }

            with open(filepath, 'w', encoding='utf-8') as fp:
                _json.dump(manifest, fp, indent=2, default=str)

            self._status_bar.showMessage(
                f"Reproducibility manifest exported to {filepath}", 8000
            )
        except Exception as exc:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export manifest:\n{exc}"
            )

    # =================================================================
    # TOOLS MENU SLOTS
    # =================================================================
    def _load_example_data(self):
        """Load synthetic demo data into all tabs."""
        try:
            comp_data, sources, settings = generate_synthetic_demo()

            # Clear stale results before loading new data
            self._tab_rss.clear_results()
            self._tab_mc.clear_results()
            self._tab_rollup.clear_results()

            self._tab_comparison.set_comparison_data(comp_data)
            self._tab_sources.set_sources(sources)
            self._tab_settings.set_settings(settings)

            self._project_name = "Synthetic_Demo"
            self._project_folder = ""
            self._unsaved_changes = True
            self._lbl_project_name.setText("Synthetic_Demo")

            audit_log.log("DEMO_LOADED",
                          "Synthetic demo data loaded for testing.",
                          f"{len(sources)} sources, "
                          f"{comp_data.flat_data().size} comparison points")

            # Trigger auto-compute
            QTimer.singleShot(200, self._auto_compute_rss)

            self._status_bar.showMessage(
                "Example data loaded. This is synthetic demo data for testing.",
                10000,
            )
            self._tabs.setCurrentIndex(0)

        except Exception as exc:
            QMessageBox.critical(
                self, "Demo Data Error",
                f"Failed to load example data:\n{exc}"
            )
            audit_log.log_warning(f"Demo data load failed: {exc}")
            traceback.print_exc()

    def _load_advanced_example_data(self):
        """Load advanced demo data (NOT VALIDATED scenario) into all tabs."""
        try:
            comp_data, sources, settings = generate_advanced_demo()

            # Clear stale results before loading new data
            self._tab_rss.clear_results()
            self._tab_mc.clear_results()
            self._tab_rollup.clear_results()

            self._tab_comparison.set_comparison_data(comp_data)
            self._tab_sources.set_sources(sources)
            self._tab_settings.set_settings(settings)

            self._project_name = "Advanced_Demo_NOT_VALIDATED"
            self._project_folder = ""
            self._unsaved_changes = True
            self._lbl_project_name.setText("Advanced_Demo_NOT_VALIDATED")

            audit_log.log("ADV_DEMO_LOADED",
                          "Advanced demo data loaded (NOT VALIDATED scenario).",
                          f"{len(sources)} sources, "
                          f"{comp_data.flat_data().size} comparison points")

            # Trigger auto-compute
            QTimer.singleShot(200, self._auto_compute_rss)

            self._status_bar.showMessage(
                "Advanced example loaded. Demonstrates asymmetric, correlated, "
                "and non-Normal sources with a NOT VALIDATED result.",
                10000,
            )
            self._tabs.setCurrentIndex(0)

        except Exception as exc:
            QMessageBox.critical(
                self, "Advanced Demo Error",
                f"Failed to load advanced example data:\n{exc}"
            )
            audit_log.log_warning(f"Advanced demo data load failed: {exc}")
            traceback.print_exc()

    def _clear_all(self):
        """Clear all data after user confirmation."""
        reply = QMessageBox.question(
            self,
            "Clear All Data",
            "This will clear all data, sources, and results.\nAre you sure?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        self._tab_comparison.set_comparison_data(ComparisonData())
        self._tab_sources.set_sources([])
        self._tab_settings.set_settings(AnalysisSettings())
        self._tab_rss.clear_results()
        self._tab_mc.clear_results()
        self._tab_rollup.clear_results()

        self._project_folder = ""
        self._project_name = "Untitled"
        self._lbl_project_name.setText("Untitled")
        self._unsaved_changes = False

        audit_log.entries.clear()
        audit_log.log("SESSION_START", f"{APP_NAME} v{APP_VERSION} — Data cleared")

        self._status_bar.showMessage("All data cleared.", 5000)
        self._tabs.setCurrentIndex(0)

    # =================================================================
    # HELP MENU SLOTS
    # =================================================================
    def _show_about(self):
        """Display the About dialog."""
        QMessageBox.about(
            self,
            f"About {APP_NAME}",
            f"<h2>{APP_NAME}</h2>"
            f"<p><b>Version:</b> {APP_VERSION}<br>"
            f"<b>Date:</b> {APP_DATE}<br>"
            f"<b>Python:</b> {sys.version.split()[0]}</p>"
            f"<p>CFD Validation Uncertainty Tool implementing the "
            f"ASME V&amp;V 20 framework with RSS and Monte Carlo methods.</p>"
            f"<p><b>Standards:</b></p>"
            f"<ul>"
            f"<li>ASME V&amp;V 20-2009 (R2021)</li>"
            f"<li>JCGM 100:2008 (GUM)</li>"
            f"<li>JCGM 101:2008 (GUM Supplement 1)</li>"
            f"<li>ASME PTC 19.1-2018</li>"
            f"<li>AIAA G-077-1998</li>"
            f"</ul>"
        )

    # =================================================================
    # WINDOW CLOSE EVENT
    # =================================================================
    def closeEvent(self, event):
        """Prompt to save unsaved changes before closing."""
        if self._unsaved_changes:
            reply = QMessageBox.question(
                self,
                "Unsaved Changes",
                "You have unsaved changes. Do you want to save before exiting?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )
            if reply == QMessageBox.Save:
                self._save_project()
                # Only close if save actually succeeded (unsaved_changes cleared)
                if self._unsaved_changes:
                    event.ignore()
                    return
                event.accept()
            elif reply == QMessageBox.Discard:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")  # Most consistent cross-platform style

    # Set font
    font = QFont()
    for family in FONT_FAMILIES:
        if QFontDatabase.hasFamily(family):
            font.setFamily(family)
            break
    font.setPointSize(10)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())
