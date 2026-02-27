"""
Constants for the VVUQ Validation Plotter.

Centralises all color palettes, font families, unit options,
named column indices, and the Boeing brand-derived plot palette.
"""

# ── Named column indices (avoid magic numbers — checklist §3) ────────────
COL_CATEGORY = 0
COL_VARIABLE = 1
COL_SENSOR = 2
COL_DATA_START = 3

# ── Font family fallback chain (matches other VVUQ tools) ───────────────
FONT_FAMILIES = [
    "Segoe UI", "DejaVu Sans", "Liberation Sans", "Noto Sans",
    "Ubuntu", "Helvetica", "Arial", "sans-serif",
]

# ── Dark Catppuccin-inspired GUI colour palette ──────────────────────────
DARK_COLORS = {
    'bg':           '#1e1e2e',
    'bg_alt':       '#252536',
    'surface0':     '#313244',
    'bg_widget':    '#2a2a3c',
    'bg_input':     '#333348',
    'fg':           '#cdd6f4',
    'fg_dim':       '#9399b2',
    'fg_bright':    '#ffffff',
    'accent':       '#89b4fa',
    'accent_hover': '#74c7ec',
    'green':        '#a6e3a1',
    'yellow':       '#f9e2af',
    'red':          '#f38ba8',
    'orange':       '#fab387',
    'border':       '#45475a',
    'overlay0':     '#6c7086',
    'selection':    '#45475a',
    'link':         '#89dceb',
}

# ── Boeing brand-derived plot palette ────────────────────────────────────
BOEING_PALETTE = {
    # Primary colours
    'primary':        '#0033A1',   # Boeing Blue — E line
    'primary_light':  '#3366CC',   # Lighter Boeing blue
    'primary_dark':   '#002070',   # Darker Boeing blue

    # Uncertainty band fills (VVUQ-20 formula order: u_num, u_input, u_d)
    'band_num':       '#4472C4',   # u_num — blue family
    'band_input':     '#ED7D31',   # u_input — orange family
    'band_data':      '#70AD47',   # u_d — green family

    # Band edge lines (darker tints of the fills)
    'band_num_edge':    '#2F5597',
    'band_input_edge':  '#C55A11',
    'band_data_edge':   '#548235',

    # Reference lines
    'zero_line':      '#333333',   # E = 0
    'val_req_line':   '#C00000',   # Validation requirement

    # Gap / no-data markers
    'gap_fill':       '#D9D9D9',
    'gap_text':       '#666666',

    # Validation status
    'pass_green':     '#70AD47',
    'fail_red':       '#C00000',
    'no_data_gray':   '#BFBFBF',

    # Alternating TC-group background shading (light theme export)
    'separator_light': ['#FFFFFF', '#E8ECF0'],
    # Alternating TC-group background shading (dark theme GUI)
    'separator_dark':  ['#1e1e2e', '#383852'],

    # Scatter / multi-series colour cycle (10 distinct colours)
    'scatter_cycle': [
        '#0033A1', '#ED7D31', '#70AD47', '#FFC000', '#5B9BD5',
        '#C00000', '#7030A0', '#00B050', '#BF8F00', '#404040',
    ],
}

# ── Unit selection options ───────────────────────────────────────────────
UNIT_OPTIONS = [
    "°F",
    "°C",
    "K",
    "°R",
    "ft/min",
    "ft/s",
    "m/s",
    "psi",
    "psig",
    "Pa",
    "kPa",
    "%",
]

# ── Default configuration values ─────────────────────────────────────────
DEFAULT_NO_DATA_LABEL = "No Test Data Available"
DEFAULT_K_FACTOR = 2.0
DEFAULT_UNITS = "°F"
DEFAULT_CONCERN_DIRECTION = "both"   # "under", "over", "both"

# ── Export / light-theme text colours (for for_export branches) ──────────
EXPORT_TEXT_COLOR = '#333333'
EXPORT_BG_COLOR = '#ffffff'
EXPORT_BOX_COLOR = '#ffffff'
EXPORT_EDGE_COLOR = '#cccccc'

# ── Export settings ──────────────────────────────────────────────────────
EXPORT_DPI = 600
EXPORT_WIDTH_INCHES = 6.0
CLIPBOARD_DPI = 150
REPORT_DPI = 300

# ── Matplotlib dark-theme style dict (GUI preview) ──────────────────────
PLOT_STYLE_DARK = {
    'figure.facecolor':  DARK_COLORS['bg_alt'],
    'axes.facecolor':    DARK_COLORS['bg_widget'],
    'axes.edgecolor':    DARK_COLORS['border'],
    'axes.labelcolor':   DARK_COLORS['fg'],
    'text.color':        DARK_COLORS['fg'],
    'xtick.color':       DARK_COLORS['fg_dim'],
    'ytick.color':       DARK_COLORS['fg_dim'],
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'axes.labelsize':    8,
    'axes.titlesize':    9,
    'legend.fontsize':   6.5,
    'grid.color':        DARK_COLORS['border'],
    'legend.facecolor':  DARK_COLORS['bg_widget'],
    'legend.edgecolor':  DARK_COLORS['border'],
}

# ── Matplotlib light-theme style dict (export / report) ─────────────────
PLOT_STYLE_LIGHT = {
    'figure.facecolor':  '#ffffff',
    'axes.facecolor':    '#ffffff',
    'axes.edgecolor':    '#333333',
    'axes.labelcolor':   '#1a1a2e',
    'text.color':        '#1a1a2e',
    'xtick.color':       '#333333',
    'ytick.color':       '#333333',
    'xtick.labelsize':   7,
    'ytick.labelsize':   7,
    'axes.labelsize':    8,
    'axes.titlesize':    9,
    'legend.fontsize':   6.5,
    'grid.color':        '#cccccc',
    'legend.facecolor':  '#ffffff',
    'legend.edgecolor':  '#999999',
}


# ── Alphabet coding for x-axis condition labels ─────────────────────────
def condition_code(index: int) -> str:
    """Convert 0-based index to A, B, …, Z, AA, AB, …, AZ, BA, …

    Examples
    --------
    >>> condition_code(0)
    'A'
    >>> condition_code(25)
    'Z'
    >>> condition_code(26)
    'AA'
    >>> condition_code(27)
    'AB'
    """
    if index < 0:
        raise ValueError(f"condition_code requires non-negative index, got {index}")
    letters = ""
    while True:
        letters = chr(ord('A') + index % 26) + letters
        index = index // 26 - 1
        if index < 0:
            break
    return letters
