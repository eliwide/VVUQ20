"""
CSV parser for the VVUQ Validation Plotter.

Loads five identically-structured CSV files (E, u_num, u_input, u_d,
U_val) and assembles them into a ``ValidationDataSet``.  Handles:

- Auto-detected delimiters (tab → semicolon → comma)
- European locale decimal-comma parsing
- UTF-8 BOM markers
- Blank cells (mapped to ``None``)
- Cross-file structure validation
- Mixed-variable-per-category error detection
"""

import csv
import math
import os
import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from .constants import COL_CATEGORY, COL_VARIABLE, COL_SENSOR, COL_DATA_START
from .data_model import CategoryData, PointData, ValidationDataSet


# ── Locale-safe float parsing ────────────────────────────────────────────

def _locale_float(text: str) -> float:
    """Parse a numeric string that may use comma as decimal separator.

    Handles:
    - Standard period decimals: ``"3.14"``
    - European comma decimals: ``"3,14"``
    - Whitespace / thousand separators stripped

    Raises ``ValueError`` for genuinely non-numeric strings.

    Note: never use bare ``float()`` on user/CSV data — always use
    this function (checklist §8).
    """
    s = text.strip()
    if not s:
        raise ValueError("empty string")
    # Strip common thousand separators (space, thin space, period in "1.234,56")
    # Heuristic: if both '.' and ',' are present, the last one is the decimal
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            # European: "1.234,56"  →  "1234.56"
            s = s.replace('.', '').replace(',', '.')
        else:
            # US: "1,234.56"  →  "1234.56"
            s = s.replace(',', '')
    elif ',' in s:
        # Comma-only: assume decimal comma  →  "3,14" → "3.14"
        s = s.replace(',', '.')
    result = float(s)
    # Reject inf/nan — these are not valid measurement data (checklist §10)
    if not math.isfinite(result):
        raise ValueError(f"non-finite value: {text.strip()!r}")
    return result


# ── Delimiter auto-detection ─────────────────────────────────────────────

def _detect_delimiter(sample_line: str) -> str:
    """Detect CSV delimiter from a sample line.

    Priority: tab → semicolon → comma.
    European CSVs use semicolons as field delimiters with comma decimals.
    """
    if '\t' in sample_line:
        return '\t'
    if ';' in sample_line:
        return ';'
    return ','


def _smart_split_line(line: str) -> List[str]:
    """Split a data line using auto-detected delimiter with csv.reader.

    Handles quoted fields correctly — condition labels like
    ``"Flight Test XX, Condition .102"`` won't be split on the comma.
    Priority: tab → semicolon → comma.
    European CSVs use semicolons as field delimiters with comma decimals.
    """
    delimiter = _detect_delimiter(line)
    # csv.reader handles quoting, escaping, and multi-field values.
    rows = list(csv.reader([line], delimiter=delimiter))
    if rows:
        return [t.strip() for t in rows[0]]
    return []


# ── Single-CSV parser ────────────────────────────────────────────────────

def _parse_single_csv(
    filepath: str,
) -> Tuple[List[str], List[Tuple[str, str, str]], Dict[Tuple[str, str, str], Dict[int, Optional[float]]]]:
    """Parse one CSV file into condition labels, row keys, and values.

    Returns
    -------
    conditions : list of str
        Condition labels from the header row (cols 3+).
    row_keys : list of (category, variable, sensor)
        Ordered list of row identifiers.
    values : dict
        ``{(category, variable, sensor): {condition_index: float_or_None}}``
    """
    # Guard against very large files
    file_size = os.path.getsize(filepath)
    if file_size > 100 * 1024 * 1024:
        warnings.warn(
            f"File is very large ({file_size / (1024 * 1024):.0f} MB). "
            f"Consider subsampling for analysis.",
            stacklevel=2,
        )

    # Read lines, skip blanks and comments
    raw_lines: List[str] = []
    with open(filepath, 'r', encoding='utf-8-sig') as fh:
        for line in fh:
            stripped = line.rstrip('\n\r')
            if stripped.strip() == '' or stripped.strip().startswith('#'):
                continue
            raw_lines.append(stripped)

    if len(raw_lines) < 2:
        raise ValueError(
            f"CSV file '{os.path.basename(filepath)}' must have at least "
            f"a header row and one data row."
        )

    # ── Header row (row 0) — condition labels in cols 3+ ──
    header_tokens = _smart_split_line(raw_lines[0])
    if len(header_tokens) <= COL_DATA_START:
        raise ValueError(
            f"CSV header in '{os.path.basename(filepath)}' has only "
            f"{len(header_tokens)} columns — expected at least "
            f"{COL_DATA_START + 1} (category, variable, sensor, + "
            f"at least one condition)."
        )
    conditions = header_tokens[COL_DATA_START:]
    n_conditions = len(conditions)

    # ── Data rows (row 1+) ──
    row_keys: List[Tuple[str, str, str]] = []
    values: Dict[Tuple[str, str, str], Dict[int, Optional[float]]] = {}
    bad_tokens: List[str] = []

    for line_idx, raw_line in enumerate(raw_lines[1:], start=2):
        tokens = _smart_split_line(raw_line)

        # Require at least the 3 key columns
        if len(tokens) < COL_DATA_START:
            warnings.warn(
                f"Line {line_idx} in '{os.path.basename(filepath)}' has only "
                f"{len(tokens)} columns — skipping.",
                stacklevel=2,
            )
            continue

        category = tokens[COL_CATEGORY].strip()
        variable = tokens[COL_VARIABLE].strip()
        sensor = tokens[COL_SENSOR].strip()

        if not category or not variable or not sensor:
            warnings.warn(
                f"Line {line_idx} in '{os.path.basename(filepath)}' has a "
                f"blank category, variable, or sensor — skipping.",
                stacklevel=2,
            )
            continue

        key = (category, variable, sensor)
        is_duplicate = key in values
        if is_duplicate:
            warnings.warn(
                f"Duplicate row key {key} at line {line_idx} in "
                f"'{os.path.basename(filepath)}' — skipping duplicate.",
                stacklevel=2,
            )
            continue

        row_values: Dict[int, Optional[float]] = {}
        data_tokens = tokens[COL_DATA_START:]

        for cond_idx in range(n_conditions):
            if cond_idx < len(data_tokens):
                cell = data_tokens[cond_idx].strip()
            else:
                cell = ""

            if not cell:
                # Blank cell → no data
                row_values[cond_idx] = None
            else:
                try:
                    row_values[cond_idx] = _locale_float(cell)
                except ValueError:
                    bad_tokens.append(
                        f"line {line_idx} col {COL_DATA_START + cond_idx + 1}: "
                        f"'{cell}'"
                    )
                    row_values[cond_idx] = None

        row_keys.append(key)
        values[key] = row_values

    if bad_tokens:
        examples = bad_tokens[:10]
        detail = "; ".join(examples)
        if len(bad_tokens) > 10:
            detail += f" ... and {len(bad_tokens) - 10} more"
        warnings.warn(
            f"Non-numeric values in '{os.path.basename(filepath)}': "
            f"{detail}. These cells were treated as missing data.",
            stacklevel=2,
        )

    if not row_keys:
        raise ValueError(
            f"No valid data rows found in '{os.path.basename(filepath)}'."
        )

    return conditions, row_keys, values


# ── Cross-file validation and assembly ───────────────────────────────────

def load_validation_csvs(
    e_path: str,
    u_num_path: str,
    u_input_path: str,
    u_d_path: str,
    u_val_path: str,
) -> ValidationDataSet:
    """Load and cross-validate five CSV files into a ``ValidationDataSet``.

    Parameters
    ----------
    e_path, u_num_path, u_input_path, u_d_path, u_val_path : str
        File paths to the five CSV files.

    Returns
    -------
    ValidationDataSet

    Raises
    ------
    ValueError
        If files have inconsistent structure, mixed variables per
        category, or other data-quality issues.
    """
    file_map = {
        'E': e_path,
        'u_num': u_num_path,
        'u_input': u_input_path,
        'u_d': u_d_path,
        'U_val': u_val_path,
    }

    # Parse each file
    parsed = {}
    for name, path in file_map.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"CSV file for '{name}' not found: {path}"
            )
        parsed[name] = _parse_single_csv(path)

    # ── Cross-validate structure ──
    ref_name = 'E'
    ref_conditions, ref_keys, _ = parsed[ref_name]

    for name in ('u_num', 'u_input', 'u_d', 'U_val'):
        conds, keys, _ = parsed[name]

        # Check condition labels match
        if conds != ref_conditions:
            raise ValueError(
                f"Condition labels in '{os.path.basename(file_map[name])}' "
                f"do not match '{os.path.basename(file_map[ref_name])}'. "
                f"All five CSV files must have identical header rows.\n"
                f"  E has {len(ref_conditions)} conditions: "
                f"{ref_conditions[:5]}{'...' if len(ref_conditions) > 5 else ''}\n"
                f"  {name} has {len(conds)} conditions: "
                f"{conds[:5]}{'...' if len(conds) > 5 else ''}"
            )

        # Check row keys match
        if keys != ref_keys:
            # Find the first mismatch for a clear error
            ref_set = set(ref_keys)
            other_set = set(keys)
            only_in_ref = ref_set - other_set
            only_in_other = other_set - ref_set
            details = []
            if only_in_ref:
                examples = list(only_in_ref)[:3]
                details.append(f"rows in E but not in {name}: {examples}")
            if only_in_other:
                examples = list(only_in_other)[:3]
                details.append(f"rows in {name} but not in E: {examples}")
            if not details:
                details.append("same rows but in different order")
            raise ValueError(
                f"Row keys in '{os.path.basename(file_map[name])}' do not "
                f"match '{os.path.basename(file_map[ref_name])}'. "
                f"All five CSV files must have identical row structure. "
                f"{'; '.join(details)}"
            )

    # ── Assemble into data model ──
    conditions = ref_conditions
    row_keys = ref_keys
    n_conditions = len(conditions)

    # Group row keys by category, preserving order
    category_order: List[str] = []
    category_rows: Dict[str, List[Tuple[str, str, str]]] = {}
    for key in row_keys:
        cat = key[0]
        if cat not in category_rows:
            category_order.append(cat)
            category_rows[cat] = []
        category_rows[cat].append(key)

    # Build CategoryData for each category
    categories: List[CategoryData] = []

    for cat in category_order:
        keys_in_cat = category_rows[cat]

        # ── Validate: single variable per category ──
        variables_in_cat = list(OrderedDict.fromkeys(k[1] for k in keys_in_cat))
        if len(variables_in_cat) > 1:
            raise ValueError(
                f"Category '{cat}' contains multiple variables: "
                f"{variables_in_cat}. Each category must have a single "
                f"variable type. Please split into separate categories "
                f"(e.g., '{cat} - {variables_in_cat[0]}' and "
                f"'{cat} - {variables_in_cat[1]}')."
            )
        variable = variables_in_cat[0]

        # Collect unique sensors in order
        sensors = list(OrderedDict.fromkeys(k[2] for k in keys_in_cat))

        # Build point list: ordered by sensor (primary), condition (secondary)
        points: List[PointData] = []
        for key in keys_in_cat:
            sensor = key[2]
            e_vals = parsed['E'][2].get(key, {})
            u_num_vals = parsed['u_num'][2].get(key, {})
            u_input_vals = parsed['u_input'][2].get(key, {})
            u_d_vals = parsed['u_d'][2].get(key, {})
            u_val_vals = parsed['U_val'][2].get(key, {})

            for cond_idx in range(n_conditions):
                e_val = e_vals.get(cond_idx)
                u_num_val = u_num_vals.get(cond_idx)
                u_input_val = u_input_vals.get(cond_idx)
                u_d_val = u_d_vals.get(cond_idx)
                u_val_val = u_val_vals.get(cond_idx)

                has_data = all(
                    v is not None
                    for v in (e_val, u_num_val, u_input_val, u_d_val, u_val_val)
                )

                points.append(PointData(
                    sensor=sensor,
                    condition=conditions[cond_idx],
                    condition_index=cond_idx,
                    E=e_val,
                    u_num=u_num_val,
                    u_input=u_input_val,
                    u_d=u_d_val,
                    U_val=u_val_val,
                    has_data=has_data,
                ))

        categories.append(CategoryData(
            category=cat,
            variable=variable,
            points=points,
            sensors=sensors,
            conditions=conditions,
        ))

    return ValidationDataSet(
        categories=categories,
        all_conditions=conditions,
        source_files=file_map,
    )
