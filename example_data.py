"""
Example data generator for the VVUQ Validation Plotter.

Creates five synthetic CSV files with realistic validation data for
testing and demonstration.  The example covers four categories, two
variable types, 10-12 conditions, and 10 sensors per category — with
~10% of cells intentionally blank to exercise gap-handling logic.

Temperature values include a +12 deg F bias.
Pressure values include a +0.015 psig bias.
"""

import os
import random
import math


def generate_example_csvs(output_dir: str) -> dict:
    """Generate five example CSV files in *output_dir*.

    Returns
    -------
    dict
        ``{"E": path, "u_num": path, "u_input": path,
          "u_d": path, "U_val": path}``
    """
    os.makedirs(output_dir, exist_ok=True)

    # Reproducible randomness
    rng = random.Random(42)

    # ── Category definitions ─────────────────────────────────────────
    categories = [
        {
            'name': 'Exhaust Nozzle Temperature',
            'variable': 'Temperature',
            'sensors': [f'TC-{i:02d}' for i in range(1, 11)],
            'n_conditions': 12,
            'cond_prefix': 'Flight Test',
            # E baseline: over-prediction by ~12 F (bias) ± spread
            'e_mean': 12.0,
            'e_std': 6.0,
            # Uncertainty magnitudes (expanded, k=2)
            'u_num_range': (2.0, 8.0),
            'u_input_range': (1.0, 5.0),
            'u_d_range': (3.0, 10.0),
        },
        {
            'name': 'Exhaust Nozzle Pressure',
            'variable': 'Pressure',
            'sensors': [f'PT-{i:02d}' for i in range(1, 11)],
            'n_conditions': 12,
            'cond_prefix': 'Flight Test',
            # E baseline: over-prediction by ~0.015 psig (bias) ± spread
            'e_mean': 0.015,
            'e_std': 0.012,
            'u_num_range': (0.003, 0.010),
            'u_input_range': (0.002, 0.008),
            'u_d_range': (0.005, 0.015),
        },
        {
            'name': 'Combustor Liner Temperature',
            'variable': 'Temperature',
            'sensors': [f'TC-{i:02d}' for i in range(11, 21)],
            'n_conditions': 12,
            'cond_prefix': 'Ground Test',
            'e_mean': 14.0,
            'e_std': 8.0,
            'u_num_range': (3.0, 10.0),
            'u_input_range': (2.0, 7.0),
            'u_d_range': (4.0, 12.0),
        },
        {
            'name': 'Turbine Inlet Temperature',
            'variable': 'Temperature',
            'sensors': [f'TC-{i:02d}' for i in range(21, 31)],
            'n_conditions': 10,
            'cond_prefix': 'Condition',
            'e_mean': 10.0,
            'e_std': 7.0,
            'u_num_range': (2.5, 9.0),
            'u_input_range': (1.5, 6.0),
            'u_d_range': (3.5, 11.0),
        },
    ]

    # ── Build the global condition list ──────────────────────────────
    # Use the maximum number of conditions across all categories.
    # Categories with fewer conditions will have blanks in the extra cols.
    max_conditions = max(c['n_conditions'] for c in categories)
    condition_labels = []
    for i in range(1, max_conditions + 1):
        condition_labels.append(f"Condition {i}")

    # ── Generate data matrices ───────────────────────────────────────
    # Structure: {file_type: {(cat, var, sensor): [val_or_None, ...]}}
    data = {ft: {} for ft in ('E', 'u_num', 'u_input', 'u_d', 'U_val')}

    # Track which rows exist (for consistent ordering)
    row_keys = []

    # ── Pre-select "dispositioned" sensors per category ────────────
    # Gaps should be ALL conditions for entire TCs (not random cells),
    # simulating a dispositioned thermocouple/sensor.  Pick 1-2
    # sensors per category to be completely blank.
    blank_sensors = {}  # cat_name -> set of blank sensor names
    for cat_def in categories:
        cat_sensors = cat_def['sensors']
        n_blank = rng.randint(1, 2)
        chosen = set(rng.sample(cat_sensors, min(n_blank, len(cat_sensors))))
        blank_sensors[cat_def['name']] = chosen

    for cat_def in categories:
        cat_name = cat_def['name']
        variable = cat_def['variable']
        n_conds = cat_def['n_conditions']

        for sensor in cat_def['sensors']:
            key = (cat_name, variable, sensor)
            row_keys.append(key)

            # If this sensor is dispositioned, ALL conditions are blank
            sensor_is_blank = sensor in blank_sensors[cat_name]

            e_row = []
            u_num_row = []
            u_input_row = []
            u_d_row = []
            u_val_row = []

            for cond_idx in range(max_conditions):
                # Conditions beyond this category's count → blank
                if cond_idx >= n_conds:
                    e_row.append(None)
                    u_num_row.append(None)
                    u_input_row.append(None)
                    u_d_row.append(None)
                    u_val_row.append(None)
                    continue

                # Dispositioned sensor → all conditions blank
                if sensor_is_blank:
                    e_row.append(None)
                    u_num_row.append(None)
                    u_input_row.append(None)
                    u_d_row.append(None)
                    u_val_row.append(None)
                    continue

                # Generate E (comparison error)
                e_val = rng.gauss(cat_def['e_mean'], cat_def['e_std'])

                # Add some sensor-specific systematic variation
                sensor_num = int(sensor.split('-')[1])
                e_val += (sensor_num % 5 - 2) * cat_def['e_std'] * 0.3

                # Add condition-specific variation
                e_val += (cond_idx - n_conds / 2) * cat_def['e_std'] * 0.1

                # Deliberately make some points cross the validation boundary
                if rng.random() < 0.15:
                    # Push E close to zero or negative (under-prediction)
                    e_val = rng.gauss(0.0, cat_def['e_std'] * 0.5)

                # Generate uncertainties
                u_num_val = rng.uniform(*cat_def['u_num_range'])
                u_input_val = rng.uniform(*cat_def['u_input_range'])
                u_d_val = rng.uniform(*cat_def['u_d_range'])

                # U_val = sqrt(u_num^2 + u_input^2 + u_d^2)
                u_val_val = math.sqrt(
                    u_num_val ** 2 + u_input_val ** 2 + u_d_val ** 2
                )

                e_row.append(round(e_val, 4))
                u_num_row.append(round(u_num_val, 4))
                u_input_row.append(round(u_input_val, 4))
                u_d_row.append(round(u_d_val, 4))
                u_val_row.append(round(u_val_val, 4))

            data['E'][key] = e_row
            data['u_num'][key] = u_num_row
            data['u_input'][key] = u_input_row
            data['u_d'][key] = u_d_row
            data['U_val'][key] = u_val_row

    # ── Write CSV files ──────────────────────────────────────────────
    file_names = {
        'E': 'E.csv',
        'u_num': 'u_num.csv',
        'u_input': 'u_input.csv',
        'u_d': 'u_d.csv',
        'U_val': 'U_val.csv',
    }

    result_paths = {}

    for file_type, filename in file_names.items():
        filepath = os.path.join(output_dir, filename)
        result_paths[file_type] = filepath

        with open(filepath, 'w', encoding='utf-8', newline='') as fh:
            # Header row
            header_parts = ['Category', 'Variable', 'Sensor']
            header_parts.extend(condition_labels)
            fh.write(','.join(header_parts) + '\n')

            # Data rows
            for key in row_keys:
                values = data[file_type][key]
                row_parts = [key[0], key[1], key[2]]
                for val in values:
                    if val is None:
                        row_parts.append('')
                    else:
                        row_parts.append(str(val))
                fh.write(','.join(row_parts) + '\n')

    return result_paths


if __name__ == '__main__':
    # Quick test: generate to a temporary directory and print summary
    import tempfile
    out_dir = os.path.join(tempfile.gettempdir(), 'vvuq_plotter_example')
    paths = generate_example_csvs(out_dir)
    for name, path in paths.items():
        size = os.path.getsize(path)
        print(f"  {name}: {path} ({size:,} bytes)")
