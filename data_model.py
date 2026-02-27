"""
Data model for the VVUQ Validation Plotter.

Immutable dataclasses representing parsed validation data from five
CSV files.  The data model is constructed once by ``csv_parser`` and
never mutated — chart renderers receive it read-only.

Missing data is modelled as ``None`` (not ``NaN``).  The ``has_data``
flag on ``PointData`` is ``True`` only when all five uncertainty
values are present, providing a single guard for chart renderers.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict


@dataclass(frozen=True)
class PointData:
    """A single measurement point (one sensor, one condition).

    Parameters
    ----------
    sensor : str
        Sensor / thermocouple label, e.g. ``"TC-01"``.
    condition : str
        Condition label from the CSV header row.
    condition_index : int
        0-based column position of this condition in the CSV.
    E : float or None
        Comparison error ``S − D``.  ``None`` if the CSV cell was blank.
    u_num : float or None
        Numerical uncertainty (expanded unless coverage-factor checkbox
        is unchecked in the GUI).
    u_input : float or None
        Input / boundary-condition uncertainty (expanded).
    u_d : float or None
        Experimental / data uncertainty (expanded).
    U_val : float or None
        Validation uncertainty (expanded).
    has_data : bool
        ``True`` only when **all five** values above are not ``None``.
    """
    sensor: str
    condition: str
    condition_index: int
    E: Optional[float]
    u_num: Optional[float]
    u_input: Optional[float]
    u_d: Optional[float]
    U_val: Optional[float]
    has_data: bool


@dataclass(frozen=True)
class CategoryData:
    """All data for one category (produces one set of charts).

    The ``variable`` field is guaranteed to be uniform within the
    category — ``csv_parser`` raises an error if mixed variables
    are found.

    Parameters
    ----------
    category : str
        Category label, e.g. ``"Exhaust Nozzle Temperature"``.
    variable : str
        Variable type, e.g. ``"Temperature"``, ``"Pressure"``.
    points : list of PointData
        Flat list of all points, ordered by sensor (primary) then
        condition (secondary) — matching CSV row order within the
        category.
    sensors : list of str
        Unique sensor labels in CSV row order.
    conditions : list of str
        Condition labels in CSV column order.
    """
    category: str
    variable: str
    points: List[PointData]
    sensors: List[str]
    conditions: List[str]


@dataclass(frozen=True)
class ValidationDataSet:
    """Complete parsed dataset assembled from five CSV files.

    Parameters
    ----------
    categories : list of CategoryData
        One entry per unique category string found in the CSVs.
    all_conditions : list of str
        Global condition labels from the CSV header row.
    source_files : dict
        Mapping of file type to file path, e.g.
        ``{"E": "C:/data/E.csv", "u_num": "C:/data/u_num.csv", …}``.
    """
    categories: List[CategoryData]
    all_conditions: List[str]
    source_files: Dict[str, str]
