"""
Theme and stylesheet for the VVUQ Validation Plotter.

Provides the dark Catppuccin GUI stylesheet (identical to other VVUQ tools)
and helper functions for switching between dark (GUI) and light (export)
matplotlib themes.
"""

from .constants import DARK_COLORS, PLOT_STYLE_DARK, PLOT_STYLE_LIGHT


def get_dark_stylesheet() -> str:
    """Generate the dark mode stylesheet (matches other VVUQ tools)."""
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


def apply_plot_style(style_dict: dict) -> None:
    """Apply a style dictionary to matplotlib rcParams.

    Parameters
    ----------
    style_dict : dict
        One of ``PLOT_STYLE_DARK`` or ``PLOT_STYLE_LIGHT``.
    """
    import matplotlib as mpl
    for key, value in style_dict.items():
        mpl.rcParams[key] = value
