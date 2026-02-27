"""
Entry point for the VVUQ Validation Plotter.

Usage:
    python -m vvuq_plotter
"""

import sys
import os
import traceback


def _check_dependencies():
    """Verify required packages are installed."""
    missing = []
    try:
        import PySide6  # noqa: F401
    except ImportError:
        missing.append("PySide6")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        missing.append("matplotlib")
    try:
        import numpy  # noqa: F401
    except ImportError:
        missing.append("numpy")

    if missing:
        print(
            f"Missing required packages: {', '.join(missing)}\n"
            f"Install with: pip install {' '.join(missing)}",
            file=sys.stderr,
        )
        sys.exit(1)


def _exception_hook(exc_type, exc_value, exc_tb):
    """Global exception handler to prevent silent crashes."""
    msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print(f"Unhandled exception:\n{msg}", file=sys.stderr)

    # Try to show a dialog if Qt is available
    try:
        from PySide6.QtWidgets import QMessageBox, QApplication
        app = QApplication.instance()
        if app is not None:
            QMessageBox.critical(
                None, "Unhandled Error",
                f"An unexpected error occurred:\n\n"
                f"{exc_type.__name__}: {exc_value}\n\n"
                f"See console for full traceback.",
            )
    except Exception:
        pass


def main():
    """Launch the VVUQ Validation Plotter GUI."""
    _check_dependencies()

    # Set exception hook before anything else
    sys.excepthook = _exception_hook

    # Configure matplotlib backend before importing Qt widgets
    os.environ.setdefault("QT_API", "pyside6")
    import matplotlib
    matplotlib.use('QtAgg')

    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont, QFontDatabase

    from .constants import FONT_FAMILIES
    from .theme import get_dark_stylesheet
    from .gui_main import PlotterMainWindow

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Set font — same pattern as other VVUQ tools
    font = QFont()
    for family in FONT_FAMILIES:
        if QFontDatabase.hasFamily(family):
            font.setFamily(family)
            break
    font.setPointSize(10)
    app.setFont(font)

    # Apply dark stylesheet
    app.setStyleSheet(get_dark_stylesheet())

    window = PlotterMainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
