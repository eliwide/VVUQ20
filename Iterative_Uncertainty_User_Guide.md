# Iterative Uncertainty Calculator v1.4.0 — User Guide & Technical Reference

**CFD Iterative Convergence Uncertainty Tool per ITTC 7.5-03-01-01 and ASME V&V 20 Section 5**

*Written for engineers — not statisticians.*

---

## Revision Update (v1.4.0)

- Added **autocorrelation-aware effective sample size** (`N_eff`) for DOF guidance.
- Added a **stationarity gate** (trend + drift + Allan-style indicator):
  - failed traces are blocked from carry-over by default,
  - optional override requires documented rationale.
- Added **Decision Consequence** in project metadata (`Low`, `Medium`, `High`).
- HTML report now includes:
  - Decision card,
  - Credibility framing checklist,
  - VVUQ glossary panel,
  - Conformity assessment template.

### Quick Example (New Guardrail Workflow)

1. Import Fluent `.out` files and set `Last N`.
2. Click **Compute**.
3. If any row shows stationarity `FAIL`, do **not** carry values to the Aggregator.
4. Fix setup/windowing and recompute; use override only with written rationale.
5. Copy Section 5 carry-over values from the HTML report decision workflow.

### 60-Second Operator Decision Flow

1. **If Stationary = PASS:** carry values exactly as shown in the carry-over table.
2. **If Stationary = FAIL:** do not carry; expand/shift `Last N` to a stable window and recompute.
3. **If you must override a FAIL:** enter a written rationale first, then proceed under program approval.
4. **If correlation warning appears:** use reported `N_eff`-based DOF in the Aggregator (not raw `N-1`).

---

## Table of Contents

1. [What Is This Tool and Why Do I Need It?](#1-what-is-this-tool-and-why-do-i-need-it)
2. [What Is Iterative Convergence Uncertainty?](#2-what-is-iterative-convergence-uncertainty)
3. [Getting Started — Application Layout](#3-getting-started--application-layout)
4. [Step-by-Step: Analyzing Fluent .out Files](#4-step-by-step-analyzing-fluent-out-files)
5. [Understanding the Results](#5-understanding-the-results)
6. [The Two Methods: ITTC vs Sigma](#6-the-two-methods-ittc-vs-sigma)
7. [Time-Weighted Statistics](#7-time-weighted-statistics)
8. [Unit Conversion](#8-unit-conversion)
9. [Charts and Visualization](#9-charts-and-visualization)
10. [Batch Cross-Case Processing](#10-batch-cross-case-processing)
11. [Saving and Loading Projects](#11-saving-and-loading-projects)
12. [HTML Report Generation](#12-html-report-generation)
13. [Figure Export](#13-figure-export)
14. [Fluent Output Formats Explained](#14-fluent-output-formats-explained)
15. [Feeding Results into the VVUQ Aggregator](#15-feeding-results-into-the-vvuq-aggregator)
16. [Project Info Panel](#16-project-info-panel)
17. [Reading the Guidance Panels](#17-reading-the-guidance-panels)
18. [Key Formulas Reference](#18-key-formulas-reference)
19. [Standards References](#19-standards-references)
20. [Glossary](#20-glossary)
21. [Frequently Asked Questions](#21-frequently-asked-questions)

---

## 1. What Is This Tool and Why Do I Need It?

If you run CFD simulations in ANSYS Fluent, you need to know how much of your answer is real physics and how much is just the solver still wandering around looking for convergence. The Iterative Uncertainty Calculator answers that question.

**The problem:** Every CFD simulation iterates toward a solution. If you stop too early, or if the residuals plateau at a level where the monitored quantities are still oscillating, the answer you extract has **iterative convergence error** baked in. But how much?

**Why it matters:** In the ASME V&V 20 framework for CFD validation uncertainty, iterative convergence uncertainty (u_iter) is a component of the numerical uncertainty u_num. If your monitored blade temperature is oscillating by 2 degrees F and you pretend it is converged, you are hiding 2 degrees of uncertainty from your validation budget. That can be the difference between "validated" and "not validated."

**What this tool does:** You point it at a folder of Fluent `.out` files (report definition exports or XY-plot exports), and the tool tells you:

> *"Over the last 1000 iterations, your outlet temperature oscillated with a half-range of 0.85 deg F (ITTC method) and a sigma of 0.28 deg F (sigma method). Your iterative convergence uncertainty is u_iter = 0.85 deg F."*

That u_iter number goes directly into your V&V 20 uncertainty budget as part of u_num.

> **Common Mistakes to Avoid**
>
> Iterative uncertainty analysis looks simple -- you just look at the tail end of your convergence history. But there are several ways to get a misleading answer:
>
> - **Selecting a convergence window that is too short.** If your "Last N" window only covers 50 iterations but the solution has a low-frequency oscillation with a period of 200 iterations, you will miss the full amplitude and underestimate u_iter. Use a window that captures at least several full oscillation cycles.
> - **Not checking stationarity before trusting the result.** The tool has a stationarity gate for a reason. If your signal is still drifting (trending up or down) within the analysis window, the mean and sigma are not meaningful. A "FAIL" on the stationarity check means you need to run more iterations or shift your window.
> - **Ignoring autocorrelation effects.** Consecutive CFD iterations are not independent samples. If the autocorrelation is high, your effective sample size (N_eff) is much smaller than the raw iteration count, which means your degrees of freedom are lower than you think. Use the N_eff-based DOF when carrying values to the Aggregator.
> - **Using the tool on a solution that has not converged at all.** This tool quantifies the residual oscillation in a nearly converged solution. If your residuals are still dropping and monitored quantities are still changing significantly, the solution is not ready for this analysis. Get the simulation closer to convergence first.
> - **Confusing iteration uncertainty with discretization uncertainty.** u_iter (from this tool) and u_num from a grid study (GCI) measure different things. Iteration uncertainty is about the solver not finishing; discretization uncertainty is about the mesh not being fine enough. You need both in your V&V 20 budget -- they are separate line items.

---

## 2. What Is Iterative Convergence Uncertainty?

### The Basic Idea

CFD solvers work by iterating. Each iteration (or time step in transient simulations) updates the solution. Ideally, the solution converges to a single steady value. In practice, monitored quantities -- temperatures, pressures, flow rates -- often oscillate around a mean, even when residuals look flat.

Those oscillations represent **iterative convergence uncertainty**. The solution you extract at the final iteration is one sample from an oscillating signal, and there is no guarantee it is the "true" converged answer.

### Where u_iter Fits in the V&V 20 Budget

The numerical uncertainty u_num in ASME V&V 20 has several potential components:

| Component | Source | How to Quantify |
|-----------|--------|-----------------|
| **Grid convergence** | Finite mesh resolution | GCI study (see GCI Calculator) |
| **Time step sensitivity** | Finite temporal resolution | Time step refinement study |
| **Iterative convergence** | Solver not fully converged | **This tool** |
| **Round-off error** | Finite precision arithmetic | Usually negligible |

Iterative convergence is often the easiest to quantify because you already have the data -- your Fluent `.out` files contain the iteration history of every monitored quantity. You just need a systematic way to extract the uncertainty from that data.

### What "Converged" Really Means

A simulation is iteratively converged when the solution variables no longer change meaningfully from one iteration to the next. Residuals dropping 3 orders of magnitude is a necessary condition, but it is not sufficient. You should also check that:

- Monitored quantities (report definitions) have reached a statistically stationary state
- The oscillation amplitude is small relative to the mean value (low CoV)
- The oscillations are random noise, not systematic drift

This tool helps you check all three.

---

## 3. Getting Started -- Application Layout

### How to Run

```
python iterative_uncertainty.py
```

The application automatically checks for required dependencies (PySide6, NumPy, SciPy, Matplotlib) and offers to install any that are missing.

### Main Window

The application uses a dark **Catppuccin Mocha** theme -- dark backgrounds with soft blue accents. All exported charts and HTML reports switch to a light/print-friendly theme automatically.

At the top of the window is a collapsible **Project Info** bar (collapsed by default -- click the arrow to expand). See [Section 16](#16-project-info-panel) for details.

Below the project bar, the application has **six tabs**:

| Tab | Icon | Purpose |
|-----|------|---------|
| **Data Import** | Folder | Select root folder, scan for .out files, configure unit conversion |
| **Analysis** | Chart | Per-file statistics, Last N windowing, method selector, and carry-over table for the Aggregator |
| **Charts** | Graph | Time series, zoomed last-N, peak variable (annotated), histogram, QQ plot |
| **Batch** | Cycle | Cross-case processing, combined CSV export, box plots, bar charts |
| **Report** | Clipboard | HTML report generation, batch-only mode, embedded charts |
| **Reference** | Book | ITTC method, sigma method, Fluent format guide, glossary sub-tabs |

### Menu Bar

**File Menu:**

| Menu Item | Shortcut | Action |
|-----------|----------|--------|
| New Project | Ctrl+N | Clear all data and start fresh |
| Open Project... | Ctrl+O | Load a previously saved .itu project file |
| Save Project | Ctrl+S | Save the current project (overwrites if previously saved) |
| Save Project As... | Ctrl+Shift+S | Save to a new .itu project file |
| Export HTML Report... | Ctrl+H | Export the current report directly from the menu |
| Exit | Ctrl+Q | Close the application |

**Analysis Menu:**

| Menu Item | Shortcut | Action |
|-----------|----------|--------|
| Recompute Analysis | Ctrl+R | Run analysis for the selected case/file |
| Run Batch Processing | Ctrl+M | Run cross-case processing |

**Examples Menu:**

| Menu Item | Action |
|-----------|--------|
| Load Built-in Example Cases | Loads 3 synthetic transient CFD cases so you can test the full workflow instantly |

**Help Menu:**

| Menu Item | Action |
|-----------|--------|
| About | Version info, supported standards, and capabilities |

The status bar at the bottom displays the application name, version, and build date on the right side, with transient status messages (e.g., "Analysis complete," "Batch processing complete") on the left.

---

## 4. Step-by-Step: Analyzing Fluent .out Files

### Step 1: Import Data

1. Go to the **Data Import** tab
2. Either:
- Click **Browse...** and select the root folder containing your Fluent `.out` files, then click **Scan for .out Files**
- Or click **Load Built-in Example Cases** to run a full dry-run without external files

The tool recursively scans the folder. Each subdirectory is treated as a separate **case** (test condition, operating point, etc.). Files at the root level are grouped under the root folder name.

**Example folder structure:**

```
turbine_thermal_study/
    case_takeoff/
        blade_temp-rfile.out       (Report Definition, 5000 rows, 4 columns)
        shroud_pressure-rfile.out  (Report Definition, 5000 rows, 2 columns)
    case_cruise/
        blade_temp-rfile.out
        shroud_pressure-rfile.out
    case_idle/
        blade_temp-rfile.out
```

The tool finds 3 cases with their respective `.out` files. The right-hand tree view shows every file organized by case, with the detected format (`report_definition` or `xy_plot`), row count, and column count. Files with parsing warnings are highlighted in yellow -- hover over the file name to see the warning text.

**Options:** The "Apply unit conversion (K -> deg F, Pa -> psia)" checkbox controls whether automatic unit conversion is applied. It is checked by default. See [Section 8](#8-unit-conversion).

### Step 2: Configure Analysis Settings

Switch to the **Analysis** tab. Set the following controls:

| Control | What to Enter | Guidance |
|---------|---------------|----------|
| **Case** dropdown | Select the case folder to analyze | Each subdirectory from Step 1 appears here |
| **File** dropdown | Select the specific .out file | If a case has multiple .out files, choose the one containing your variable of interest |
| **Last N** spinner | Number of last iterations or seconds (default: 1000) | Should capture several oscillation cycles. For steady cases, 500-2000 is typical. |
| **Window** dropdown | "By Iterations" or "By Time (seconds)" | By Iterations uses the last N data rows. By Time uses the last N seconds of flow-time (transient only). |
| **Method** dropdown | "Sigma-based," "ITTC Half-Range," or "Both" | "Both" computes and displays both methods side by side. Recommended. |

### Step 3: Click Compute

Click the blue **Compute** button. The tool:

1. Reads the selected `.out` file
2. Windows the data to the last N rows (or last N seconds of flow-time)
3. Optionally applies unit conversion (K to deg F, Pa to psia)
4. Computes all statistics for every variable in the file (excluding the time/iteration column)
5. Displays results in a color-coded table
6. Auto-populates the **Carry-Over to Aggregator** table with one recommended value per variable
7. Writes a per-case CSV file (`stats_<casename>.csv`) to the case folder

### Step 4: Review the Results Table

The results table shows one row per variable with 18 columns of statistics. See [Section 5](#5-understanding-the-results) for a detailed breakdown.

**Quick health check -- the CoV columns are color-coded:**

| Color | CoV Range | Interpretation |
|-------|-----------|----------------|
| **Green** | < 0.001 (0.1%) | Good convergence -- iterative scatter is small |
| **Yellow** | 0.001 to 0.01 (0.1% to 1%) | Acceptable -- worth documenting but probably not dominant |
| **Red** | > 0.01 (> 1%) | Poor convergence -- iterative uncertainty may be a significant contributor |

### Step 5: Explore Charts

Switch to the **Charts** tab. Select a variable from the dropdown and a chart type. The Full Time Series chart lets you verify that the analysis window falls within the stationary portion of the run. See [Section 9](#9-charts-and-visualization).

### Step 6: Save Your Project

Use **File > Save Project** (Ctrl+S) to save your settings to a `.itu` file. This records the root folder path, last-N setting, method selection, unit conversion preference, and project metadata. See [Section 11](#11-saving-and-loading-projects).

---

## 5. Understanding the Results

The Analysis tab displays a comprehensive statistics table for every variable in the selected file. Here is what each column means:

### Per-Variable Statistics (IterativeStats)

| Column | Symbol | Formula / Meaning |
|--------|--------|------------------|
| **Variable** | -- | The variable name from the Fluent .out file header (e.g., "Area-Weighted Average of Static Temperature") |
| **Unit** | -- | Detected or converted unit (e.g., deg F, psia). Blank if no unit was detected. |
| **N** | n_samples | Number of data points in the analysis window |
| **Final** | final_value | The very last value in the window -- what you would get if you stopped the solver right now |
| **Mean** | mean | Arithmetic mean of all values in the window |
| **Median** | median | 50th percentile -- robust to outliers |
| **Min** | min_val | Minimum value in the window |
| **Max** | max_val | Maximum value in the window |
| **sigma** | sigma | Standard deviation (with Bessel's correction, ddof=1) |
| **3*sigma** | three_sigma | 3 x sigma -- a 99.7% coverage interval if the distribution were normal |
| **P95** | p95 | 95th percentile of the values in the window |
| **Corr. 3sigma** | corrected_3sigma | min(max_val, mean + 3*sigma) -- the capped version that never exceeds the observed maximum |
| **Diff** | difference | corrected_3sigma - final_value -- how far the worst-case (corrected) bound is from your current answer |
| **CoV** | cov | sigma / abs(mean) -- the coefficient of variation, normalized measure of iterative scatter |
| **ITTC U_I** | half_range | 0.5 x (max - min) -- the ITTC half-range uncertainty |
| **TW Mean** | tw_mean | Time-weighted mean (see [Section 7](#7-time-weighted-statistics)) |
| **TW sigma** | tw_sigma | Time-weighted standard deviation |
| **TW CoV** | tw_cov | Time-weighted coefficient of variation |

### What Should You Look For?

1. **Is CoV < 0.001?** If yes, iterative uncertainty is negligible for most practical purposes. Move on.

2. **Is ITTC U_I small relative to your other uncertainties?** If your grid convergence uncertainty from the GCI Calculator is 5 deg F and your iterative half-range is 0.2 deg F, the iterative contribution is negligible. If they are the same order of magnitude, iterative uncertainty matters and must be carried forward.

3. **Is the Final value close to the Mean?** If the final value is near the edge of the distribution (close to min or max), your answer is sensitive to exactly when you stopped the solver. The mean is a more robust estimate of the converged value.

4. **Do the time-weighted and unweighted statistics agree?** If they differ significantly, you have non-uniform time steps and the time-weighted values are more appropriate for transient cases.

### Example

For a gas turbine blade cooling study, the Analysis tab might show:

| Variable | Unit | N | Final | Mean | sigma | CoV | ITTC U_I |
|----------|------|---|-------|------|-------|-----|----------|
| Blade Surface Temp | deg F | 1000 | 1652.3 | 1652.1 | 0.31 | 0.000188 | 0.85 |
| Coolant Exit Pressure | psia | 1000 | 147.22 | 147.24 | 0.018 | 0.000122 | 0.042 |

Both CoV values are green (< 0.001), indicating excellent convergence. The ITTC U_I for temperature is 0.85 deg F -- small compared to a typical GCI-based u_num of 5-10 deg F for this type of simulation.

---

## 6. The Two Methods: ITTC vs Sigma

### ITTC Half-Range Method

The International Towing Tank Conference (ITTC) Procedure 7.5-03-01-01 recommends a simple, conservative approach:

```
U_I = 0.5 x (S_max - S_min)
```

where S_max and S_min are the maximum and minimum values of the monitored quantity over the analysis window (last N iterations or last N seconds).

**How it works:** The half-range is half the total oscillation band. It represents the worst-case deviation from the midpoint. No distributional assumptions are required.

**When to use it:**
- Standard requirement in naval/marine CFD verification
- When the data is not approximately normal (multimodal, skewed, bounded)
- When you want a conservative estimate that does not depend on statistical assumptions
- When the oscillation pattern is irregular or has occasional large excursions

**Conservatism:** The ITTC method is generally more conservative than the sigma method because it responds to the single largest excursion, not the typical scatter. A single outlier spike will significantly inflate U_I.

### Sigma-Based Method

The sigma-based method computes the sample standard deviation and derives a corrected upper bound:

```
sigma = std(S, ddof=1)                        (sample standard deviation)
three_sigma = 3.0 x sigma                     (three-sigma bound)
corrected_3sigma = min(S_max, mean + 3*sigma)  (capped at observed maximum)
difference = corrected_3sigma - S_final        (iterative uncertainty)
```

**How it works:** For normally distributed data, 99.7% of values fall within mean +/- 3*sigma. The "corrected" 3-sigma caps this at the actual observed maximum -- it never exceeds what the data actually shows. The iterative uncertainty ("Diff") is the gap between this upper bound and the final value.

**When to use it:**
- When the data is approximately normally distributed (verify with the histogram and QQ plot)
- When you want a statistically grounded estimate that is less sensitive to outliers
- When you need the CoV as a normalized convergence quality metric

### Side-by-Side Comparison

| Property | ITTC Half-Range | Sigma-Based |
|----------|----------------|-------------|
| **Formula** | U_I = 0.5 x (max - min) | corrected_3sigma - final |
| **Assumptions** | None (distribution-free) | Approximate normality / stationarity |
| **Conservatism** | More conservative | Less conservative |
| **Sensitivity to outliers** | High (driven by extremes) | Moderate (sigma is more robust) |
| **Sensitivity to sample size** | Low | Higher (sigma needs N > 30 for stability) |
| **Standard reference** | ITTC 7.5-03-01-01 | ASME V&V 20, JCGM 100 |
| **Normalized metric** | None built-in | CoV = sigma / abs(mean) |

**Recommendation:** Select **"Both"** in the Method dropdown to compute and display both methods simultaneously. For the V&V 20 budget, use whichever method your program or certifying authority requires. If no specific method is mandated, the ITTC half-range is the safer choice.

---

## 7. Time-Weighted Statistics

### Why Time-Weighting Matters

In transient simulations with non-uniform time steps (adaptive time stepping, variable CFL), each row of data covers a different duration of physical time. A naive arithmetic mean treats a tiny 0.001-second time step the same as a large 0.1-second time step. The longer time step should carry more weight because the solution spent more physical time at that value.

**Time-weighted statistics** correct for this by weighting each data point proportionally to the time interval it represents.

### The Trapezoidal Weighting Scheme

The tool uses trapezoidal quadrature weights. For a series of N data points at times t_1, t_2, ..., t_N:

```
w_1     = dt_1 / 2                              (first point: half the first interval)
w_i     = (dt_{i-1} + dt_i) / 2                 (interior points: average of neighbors)
w_N     = dt_{N-1} / 2                           (last point: half the last interval)
```

where dt_i = t_{i+1} - t_i.

### Time-Weighted Formulas

**Time-weighted mean:**

```
TW_mean = sum(w_i x S_i) / sum(w_i)
```

**Time-weighted standard deviation:**

```
TW_sigma = sqrt( sum(w_i x (S_i - TW_mean)^2) / sum(w_i) )
```

**Time-weighted CoV:**

```
TW_CoV = TW_sigma / abs(TW_mean)
```

### When to Use Which

| Simulation Type | Time Steps | Use TW Stats? |
|----------------|------------|---------------|
| Steady-state (iteration-based) | All equal (1 per iteration) | No difference -- TW equals unweighted |
| Transient, fixed dt | All equal | No difference |
| Transient, adaptive dt | Non-uniform | **Yes** -- TW stats are more physically meaningful |

The tool automatically detects whether the file has a time column (by looking for headers containing "flow-time," "flow time," "physical-time," etc.). If a time column is detected and flagged as transient, time-weighted statistics are computed automatically and shown in the TW Mean, TW sigma, and TW CoV columns.

### Practical Example

Suppose you are running a transient conjugate heat transfer simulation in Fluent. For the first 5 seconds you use dt = 0.01s (500 steps for stabilization). Then you switch to dt = 0.1s for the remaining 55 seconds (550 steps for production). If you analyze the last 1,050 data points (60 seconds of physical time), the first 500 points cover 5 seconds while the last 550 cover 55 seconds. Without time-weighting, those 500 stabilization points (47% of the rows) get 47% of the influence but represent only 8% of the physical time. Time-weighting correctly gives 92% of the weight to the production period.

---

## 8. Unit Conversion

### Why Convert?

Fluent typically outputs temperatures in Kelvin and pressures in Pascals (SI units). Engineering reports and certification packages in many industries use degrees Fahrenheit and psia. Converting within the tool ensures your uncertainty values are in the same units as your report -- no post-hoc conversion needed.

### Temperature: Kelvin to Fahrenheit

```
T_degF = (T_K - 273.15) x 9/5 + 32
```

| Input (K) | Output (deg F) | Typical CFD Context |
|-----------|----------------|---------------------|
| 273.15 | 32.0 | Freezing point of water |
| 373.15 | 212.0 | Boiling point of water |
| 723.0 | 841.7 | Gas turbine blade surface |
| 1200.0 | 1700.3 | Combustor liner |

### Pressure: Pascals to psia

```
P_psia = P_Pa / 6894.757
```

| Input (Pa) | Output (psia) | Typical CFD Context |
|------------|---------------|---------------------|
| 101,325 | 14.696 | Standard atmosphere |
| 689,476 | 100.0 | Turbine stage inlet |
| 6,894,757 | 1000.0 | High-pressure compressor |

### Auto-Detection Logic

The tool uses this sequence (excluding the time/iteration column):

1. If header includes pressure keywords, assume `Pa`
2. If header includes temperature keywords, assume `K`
3. If header includes velocity or Mach keywords, leave unit blank (no auto-conversion)
4. Otherwise, apply median-absolute-value heuristic:

| Median abs(value) | Assumed Unit | Conversion Applied |
|-------------------|--------------|-------------------|
| > 2000 | Pa (Pascals) | Pa to psia |
| 200 to 2000 | K (Kelvin) | K to deg F |
| < 200 | No unit detected | None |

This heuristic works well for typical thermal-hydraulic CFD outputs. It will **not** work correctly for:
- Temperatures already in Celsius or Fahrenheit
- Cryogenic temperatures below 200 K
- Very low pressures (below 2000 Pa, e.g., vacuum systems)
- Non-thermal/non-pressure variables that happen to have large magnitudes

In those cases, **uncheck the unit conversion toggle** in the Data Import tab.

### Important: All Statistics Are in Converted Units

When unit conversion is enabled, all statistics -- mean, sigma, CoV, half-range, final value, everything -- are computed on the **converted values**. Your ITTC U_I is directly in deg F or psia. No further conversion is needed before entering it into the Uncertainty Aggregator.

---

## 9. Charts and Visualization

The **Charts** tab provides five visualization types for the currently selected variable. Use the **Variable** dropdown to pick which variable to plot and the **Chart** dropdown to select the chart type.

### Chart Types

**1. Full Time Series**

Plots the entire iteration history of the selected variable (all rows in the .out file). Shows the mean as a green dashed line and the corrected 3-sigma as a red dotted line. Use this to confirm the simulation reached a statistically stationary state before your analysis window begins.

**2. Zoomed (Last N)**

Same as Full Time Series but cropped to show only the analysis window (last N rows). This is the data the statistics are actually computed from. Look for:
- Random scatter around a stable mean (good)
- Systematic drift (bad -- the solution has not converged)
- Large periodic oscillations (note the period and ensure your window covers multiple cycles)

**3. Peak Variable (Annotated)**

The most informative chart. Shows the full time series with:
- Mean line (green dashed)
- Plus/minus sigma band (green shaded region)
- Corrected 3-sigma line (red dotted, thick)
- Final value marker (orange dot at the last data point)
- Difference annotation arrow (orange, showing the gap between final value and corrected 3-sigma)
- ITTC half-range band (yellow shaded region around the mean)

This chart is designed for inclusion in V&V reports. It visually communicates every uncertainty measure at a glance.

**4. Histogram**

Distribution of values within the analysis window. The number of bins is auto-selected (square root of N, clamped between 5 and 50). A vertical line marks the mean. Use this to check whether the iterative scatter is approximately normal (bell-shaped). Skewed or multi-modal histograms suggest the solver has not fully converged, or the physics are inherently oscillatory.

**5. QQ Plot**

Quantile-quantile plot comparing the sample distribution to a theoretical normal distribution. Points that fall on the diagonal reference line indicate normality. Systematic deviations reveal:
- S-shaped curve: heavy tails (outliers more frequent than expected)
- Concave or convex departure: skewed data
- Steps or plateaus: discrete data or quantized output

If the QQ plot looks good (points follow the line), the sigma-based method is well-justified. If it deviates substantially, the ITTC half-range method makes fewer assumptions and may be more appropriate.

### Interactive Navigation

All charts include the standard Matplotlib toolbar:
- **Pan** -- click and drag to pan
- **Zoom** -- rubber-band zoom to a region of interest
- **Home** -- reset to the original view
- **Save** -- save the current view via the toolbar's built-in save button

### Save Plot

The **Save Plot...** button exports the current chart to PNG, SVG, or PDF in **report-light style** (white background, dark labels). PNG is exported at 600 DPI by default; vector formats (SVG/PDF) preserve line/text scalability. The filename is auto-generated from the variable name and chart type with a timestamp suffix. A JSON metadata sidecar file is also saved alongside the image.

### Export Figure Package

The **Export Figure Package...** button generates a publication-quality set of files. See [Section 13](#13-figure-export) for the full list of output files and formats.

---

## 10. Batch Cross-Case Processing

### What Is Cross-Case Processing?

When you have multiple cases (e.g., different operating conditions, mesh variants, or design iterations), the Batch tab processes all of them at once and computes cross-case statistics by matching variables by name across cases.

**Example:** You have three cases -- `case_takeoff`, `case_cruise`, `case_idle` -- each with a `blade_temp-rfile.out` containing a variable named "Area-Weighted Average of Static Temperature." The batch processor:

1. Computes per-case statistics for each file in each case (using the current Last N setting)
2. Matches variables by name across cases
3. Computes cross-case pooled statistics for each matched variable
4. Selects a single cross-case carry-over value based on your chosen method
5. Writes per-case CSVs and displays the combined results

### How to Run Batch Processing

1. Import and scan your data in the Data Import tab (all cases will be included)
2. Set the **Last N** value in the Analysis tab (this window size applies globally)
3. Switch to the **Batch** tab
4. Select the cross-case method: **Pooled Std Dev**, **RSS of Per-Case sigma**, or **Both**
5. Click **Run Batch Processing**

### Cross-Case Statistics

For each variable that appears in multiple cases, the batch processor collects the final values and per-case sigmas, then computes:

| Statistic | Formula | Meaning |
|-----------|---------|---------|
| **Mean Final** | mean of final values across cases | The average converged answer across all operating conditions |
| **Pooled sigma** | std dev of the final values across cases (ddof=1) | How much the final answers vary between cases. Captures case-to-case variability. |
| **RSS sigma** | sqrt( sum(sigma_i^2) / N_cases ) | Root-sum-square of the per-case iterative sigmas. Captures the average per-case iterative scatter. |
| **Carry u (Selected)** | Depends on method selection | The single value to transfer when you want one cross-case uncertainty number |
| **Selected Method** | Method label | Confirms whether Pooled, RSS, or conservative Both(max) was used |
| **Mean ITTC U_I** | mean of per-case half-range values | Average ITTC half-range across cases |

### Variable Keying Policy (Important)

The Batch tab now includes a **Variable keying** selector:

| Keying Option | Behavior | When to Use |
|---------------|----------|-------------|
| **Merge by Variable Name** | Treats same variable labels as one cross-case group | Use when names are guaranteed unique and consistent across files |
| **Separate by File + Variable** | Uses `file_base :: variable` as the cross-case key | Use when multiple files can contain identical variable names |

If duplicates are found while using merge mode, collisions are merged conservatively (higher sigma retained) and the guidance panel reports it.

### Pooled vs RSS -- Which to Use

| Method | What It Captures | Best For |
|--------|-----------------|----------|
| **Pooled Std Dev** | Case-to-case variation + iterative scatter | Characterizing total repeatability across your test matrix |
| **RSS of Per-Case sigma** | Pure iterative scatter averaged across cases | Isolating u_iter for the V&V 20 budget |

For the V&V 20 uncertainty budget, the RSS method is usually more appropriate because it isolates the iterative convergence component from the physical variation between operating conditions.

### Batch Results Table

The table shows one row per cross-case key with columns for **Cross-Case Key**, **Base Variable**, Unit, Cases, Mean Final, Pooled sigma, RSS sigma, **Carry u (Selected)**, **Selected Method**, and Mean ITTC U_I.

### Cross-Case Charts

Below the table, a chart area provides three visualization types:

| Chart | What It Shows |
|-------|--------------|
| **Box Plot** | Distribution summary per case for the selected variable using the actual analysis-window samples |
| **Bar Chart (Final Values)** | Final values with +/- sigma error bars, one bar per case |
| **Bar Chart (sigma Comparison)** | Side-by-side bars comparing sigma and ITTC U_I per case |

Use the **Cross-Case Variable** dropdown to select which variable to plot.

If a case has no stored raw sample window for the selected key, the box plot shows a visible fallback note inside the chart.

### Per-Case CSV Output

When batch processing completes, the tool automatically writes a CSV file named `stats_<casename>.csv` into each case's folder. This file contains all 18 statistics columns for every variable in that case:

```
case_takeoff/stats_case_takeoff.csv
case_cruise/stats_case_cruise.csv
case_idle/stats_case_idle.csv
```

### Combined CSV Export

Click **Export Combined CSV...** to save the cross-case summary table to a single CSV file. Columns include Cross Key, Base Variable, Key Policy, Unit, N Cases, Mean Final, Pooled Sigma, RSS Sigma, Carry Selected, Selected Method, Mean ITTC U_I, Cases.

---

## 11. Saving and Loading Projects

### Project Files (.itu)

The tool saves project state to `.itu` files (JSON format, human-readable). A project file records:

| Saved Field | Description |
|-------------|-------------|
| Tool name and version | For compatibility checking |
| Timestamp (UTC) | When the file was saved |
| Project metadata | Program/Project, Analyst, Date, Notes |
| Root folder path | The folder that was scanned for .out files |
| Last N setting | The iteration/time window size |
| Method selection | Sigma-based, ITTC Half-Range, or Both |
| Unit conversion preference | Whether K-to-deg-F and Pa-to-psia conversion is enabled |

**What is NOT saved:** The actual parsed data and computed statistics are not stored in the project file. After loading a project, you need to re-scan the folder (**Scan for .out Files**) and re-run the analysis (**Compute**). This keeps the project file lightweight and ensures results always reflect the current data on disk.

### Saving

- **File > Save Project** (Ctrl+S) -- saves to the current file path, or prompts for a path if this is a new project
- **File > Save Project As...** (Ctrl+Shift+S) -- always prompts for a new file path

The `.itu` extension is added automatically if you do not type it.

### Loading

- **File > Open Project...** (Ctrl+O) -- opens a file dialog filtered to `.itu` files
- If the saved root folder still exists on disk, the folder path is populated in the Data Import tab
- Last N, method, and unit conversion settings are restored
- Project metadata (Program, Analyst, Date, Notes) is restored

After loading, click **Scan for .out Files** in the Data Import tab, then **Compute** in the Analysis tab to regenerate results.

### New Project

**File > New Project** (Ctrl+N) clears all settings and metadata, resets the date to today, and returns the application to its initial state.

---

## 12. HTML Report Generation

### Overview

The **Report** tab generates a self-contained HTML report documenting your iterative uncertainty analysis. The report is designed for inclusion in V&V documentation packages and can be opened in any web browser.

### Report Options

| Option | Default | Effect |
|--------|---------|--------|
| **Batch-only mode** | Unchecked | When checked, excludes per-file detail tables and only includes the cross-case batch summary. Useful when you have many cases and only care about the aggregate. |
| **Embed charts** | Checked | When checked, embeds the current chart as a base64-encoded PNG image directly in the HTML. The report is then fully self-contained -- no external image files needed. |

### Report Sections

The generated HTML report includes:

1. **Project Information** -- Program/Project, Analyst, Date, Notes, tool version, generation timestamp
2. **Per-Case Statistics** (unless batch-only) -- A table for each case showing Variable, Unit, N, Final, Mean, sigma, Corrected 3-sigma, Diff, ITTC U_I, and CoV
3. **Batch Cross-Case Summary** -- If batch processing has been run, shows the cross-case statistics table
4. **Embedded Chart** -- If chart embedding is enabled, the current Charts tab figure
5. **Method Summary** -- Table comparing the ITTC half-range and sigma-based formulas
6. **Unit Conversion Notes** -- K-to-deg-F and Pa-to-psia conversion formulas and detection logic
7. **Methodology** -- Standards references (ITTC 7.5-03-01-01, ASME V&V 20, JCGM 100)
8. **Assumptions and Limitations** -- Stationarity assumption, window size, heuristic unit detection, XY-plot parser limitations
9. **Reviewer Checklist** -- Five manual verification items for the reviewer

### Reviewer Checklist Items

The report includes a checklist:

1. Convergence achieved before analysis window?
2. Window size adequate to capture oscillation cycles?
3. Unit conversions verified against Fluent setup?
4. Iterative uncertainty is small relative to discretization error?
5. Values carried to VVUQ Aggregator as u_num contribution?

### Generating and Exporting

1. Run your analysis (Analysis tab) and/or batch processing (Batch tab) first
2. Switch to the Report tab
3. Click **Generate Preview** to see the report in the preview pane
4. Click **Export HTML Report...** to save the report to an HTML file

The exported HTML file is fully self-contained -- all styles are inline, all images are base64-encoded. It does not depend on external files, stylesheets, or network access. It uses a light theme (white background, dark text) for print readability.

---

## 13. Figure Export

### Publication-Quality Export Package

The **Export Figure Package...** button on the Charts tab generates a complete set of publication-ready figure files:

| File | Format | Description |
|------|--------|-------------|
| `<base>_300dpi.png` | PNG, 300 DPI | Standard print quality |
| `<base>_600dpi.png` | PNG, 600 DPI | High-resolution for poster or large-format print |
| `<base>.svg` | SVG (vector) | Scalable vector -- ideal for journal submissions and presentations |
| `<base>.pdf` | PDF (vector) | Vector PDF -- ideal for LaTeX inclusion |
| `<base>_meta.json` | JSON | Metadata sidecar for traceability |

All raster exports use a white background with dark text and axes, regardless of the dark theme used in the application. This ensures figures look correct on white paper or in light-themed documents.

### JSON Metadata Sidecar

Every figure export includes a JSON sidecar file containing:

```json
{
  "tool_name": "Iterative Uncertainty Calculator",
  "tool_version": "1.3.0",
  "figure_id": "blade_temp_Full_Time_Series",
  "generated_utc": "2026-02-22T14:30:00+00:00",
  "formats": ["png@300dpi", "png@600dpi", "svg", "pdf"]
}
```

This provides traceability -- you can always determine which tool, version, and variable produced a specific figure. Useful for audit trails in regulated environments.

### Single-File Save

The simpler **Save Plot...** button exports just the current chart as a single file (PNG, SVG, or PDF) in report-light style. PNG uses 600 DPI. The filename is auto-generated with the pattern:

```
VariableName_ChartType_zoomed_YYYYMMDD_HHMMSS.png
```

The `_zoomed` suffix is appended when the chart type is "Zoomed (Last N)". A JSON metadata sidecar is also saved alongside the image.

---

## 14. Fluent Output Formats Explained

The tool supports two Fluent `.out` file formats and auto-detects which one to use.

### Modern: Report Definition Format

This is the format created by **Report Definitions** in Fluent (Solution > Report Definitions). This is the most common format in Fluent 2020+.

**Structure:**

```
"Blade Temperature Report"
Column 1: description text
"flow-time" "Area-Weighted Average of Static Temperature" "Mass-Weighted Average of Total Pressure"
0.001  723.15  689476.0
0.002  723.22  689471.0
0.003  723.18  689478.0
...
```

**Line-by-line breakdown (typical):**
- Metadata lines at the top are ignored
- First line containing quoted strings is treated as the header line
- Numeric lines after the header are parsed as data rows (space- or comma-separated)

**Header handling:**
- The parser finds the first line containing quoted strings and treats it as the header line
- Duplicate header names are automatically disambiguated with `_1`, `_2` suffixes
- Headers containing tokens like "flow-time," "iteration," or "time-step" are identified as the time/iteration column and excluded from statistical analysis

### Legacy: XY-Plot Format

This is the older format created by Fluent's XY-plot file export command. It uses a Scheme/Lisp parenthesized syntax:

```
((xy/key/label "Static Temperature")
 0  723.15
 1  723.22
 2  723.18
 ...)
((xy/key/label "Total Pressure")
 0  689476.0
 1  689471.0
 2  689478.0
 ...)
```

Each block starts with `((xy/key/label "Variable Name")`, followed by index-value pairs, and ends with a closing parenthesis. Multiple blocks can appear in a single file, one per variable.

### Auto-Detection

The parser reads the first 20 lines of the file and applies these rules:

1. If any line starts with `(` or `((`, the file is classified as **XY-plot format**
2. If any line contains quoted strings `"..."`, the file is classified as **Report Definition format**
3. If neither pattern is found, it defaults to Report Definition and attempts a numeric-only parse

In the Data Import tree view, the "Format" column shows `report_definition` or `xy_plot` for each file.

### Practical Tips

- **Prefer Report Definition format** -- it is more robust, carries variable names in the header, includes a time column, and supports multiple variables in a single file
- **XY-plot files** work fine but typically contain fewer variables per file and use a simple row index instead of physical time
- **Multiple .out files per case** are handled naturally -- the tool parses all files found in each case folder and lets you select which one to analyze
- **Set up Report Definitions for all quantities of interest** in your Fluent case (outlet temperature, wall heat flux, drag coefficient, mass flow rate, etc.) and write them to `.out` files every iteration. This is the data the tool needs.

---

## 15. Feeding Results into the VVUQ Aggregator

The iterative uncertainty values computed by this tool are one component of the total uncertainty budget in an ASME V&V 20 analysis. Here is exactly how to transfer them to the VVUQ Uncertainty Aggregator.

### Which Value to Use

Use the **Carry-Over to Aggregator** table in the Analysis tab. It already chooses a single value per variable based on your selected method and labels the distribution plus recommended Aggregator analysis mode (set once per run) for `vv20_validation_tool.py`.

| Method Setting | Carry u | Sigma Basis | Distribution for Aggregator | Aggregator Analysis Mode (set once) |
|----------------|---------|-------------|-------------------------|--------------------|
| Sigma-based | sigma (time-weighted sigma when available) | Confirmed 1σ | Normal | RSS |
| ITTC Half-Range | 0.5 x (max - min) | Bounding (min/max) | Uniform | RSS |
| Both | max(sigma, ITTC) (conservative auto-pick) | Confirmed 1σ or Bounding (min/max) | Normal or Uniform (based on dominant value) | RSS (carry value selected conservatively) |

For novice users, **Both** is usually the safest default because it prevents accidental underestimation.

### Step-by-Step Transfer

1. In the Aggregator, go to the **Uncertainty Sources** tab
2. Click **"Add Source"**
3. For each variable row in the carry-over table, fill in:

| Field | Value |
|-------|-------|
| **Name** | Use the variable name from the carry-over table |
| **Category** | Numerical (u_num) |
| **Distribution** | Copy **Distribution for Aggregator** |
| **Input Type** | Sigma Value Only |
| **Sigma Value** | Copy **Carry u** |
| **Sigma Basis** | Copy **Sigma Basis** from the carry-over table |
| **DOF** | Copy DOF from the carry-over table |
| **Aggregator Analysis Mode** | Set once in Aggregator Analysis Settings. Use **RSS** for values from this table. If another source requires Monte Carlo, run the Aggregator in Monte Carlo mode globally. |

4. Verify the source is enabled (checked). The RSS results will automatically include it.

### Why Does Sigma Basis Vary by Method?

- **Sigma-based:** The carry value is a true sample standard deviation computed from the iteration data. It is not an assumed or estimated value -- it comes directly from the data. Therefore, "Confirmed 1σ" is the correct basis.
- **ITTC Half-Range:** The carry value is a bounding half-range (half the peak-to-peak span), not a standard deviation. The Aggregator expects "Bounding (min/max)" so it can correctly convert to a standard uncertainty (dividing by sqrt(3) for a Uniform distribution).
- **Both:** The basis follows whichever method dominates -- "Confirmed 1σ" when sigma is larger, "Bounding (min/max)" when the ITTC half-range is larger.

### Why Not Always DOF = Infinity?

If your sample window is modest (for example N = 100 to 300), using DOF = N_eff - 1 is more traceable and less assumption-heavy. When autocorrelation is negligible, N_eff ≈ N, so this reduces to N - 1. Reserve Infinity for cases where your quality process explicitly allows it.

### What About ITTC Half-Range?

The tool maps ITTC half-range to **Uniform** distribution with **Bounding (min/max)** sigma basis. This tells the Aggregator that the carry value is a bounding interval, not a standard deviation, so the correct conversion factor (1/sqrt(3)) is applied automatically.

### Should u_iter Be u_num or u_input?

In most cases, iterative convergence uncertainty belongs under **u_num** (numerical uncertainty). It is a numerical artifact -- the solver has not fully converged to machine precision.

**Exception:** If you are running a transient simulation where the inherent solution variability is physical (e.g., turbulent fluctuations in a time-averaged quantity), the uncertainty from the time-averaging window might be classified as u_input. When in doubt, classify as u_num.

---

## 16. Project Info Panel

### Purpose

The Project Info panel provides traceability metadata for your analysis. It is designed for regulated environments where every analysis must be traceable to a project, analyst, and date.

### Location and Behavior

The panel is a collapsible bar at the very top of the main window, **collapsed by default**. Click the arrow button ("Project Info") to expand it. The arrow changes from a right-pointing triangle (collapsed) to a downward-pointing triangle (expanded).

### Fields

| Field | Purpose | Example |
|-------|---------|---------|
| **Program / Project** | Project or program name | "Gas Turbine Blade Cooling CFD Campaign" |
| **Analyst** | Name of the person performing the analysis | "J. Smith" |
| **Date** | Date of the analysis (auto-populated with today) | "2026-02-22" |
| **Notes** | Free-text field for boundary conditions, mesh strategy, solver settings, or any other context | "SST k-omega, y+ < 1, second-order upwind, CFL = 5, 5000 iterations, adaptive dt for transient phase" |

### Persistence

All four fields are:
- Saved with the `.itu` project file
- Restored when a project file is loaded
- Included in the HTML report under the "Project Information" section

When you create a new project (Ctrl+N), the metadata is cleared and the date is reset to today.

---

## 17. Reading the Guidance Panels

Each tab includes a **guidance panel** -- a colored bar with a traffic-light severity indicator that provides contextual feedback as you work.

### Severity Levels

| Color | Icon | Meaning |
|-------|------|---------|
| **Green** (left border) | Checkmark | Everything looks good. The panel shows helpful tips or confirms success. |
| **Yellow** (left border) | Warning triangle | Something needs attention. Not blocking, but worth checking. |
| **Red** (left border) | X mark | A problem was detected. Action is needed before proceeding. |

### Tab-Specific Messages

| Tab | Example Message | Severity | What to Do |
|-----|----------------|----------|------------|
| **Data Import** | "Scanned 7 .out files across 3 cases." | Green | Nothing -- data loaded successfully |
| **Data Import** | "No .out files found in the selected folder." | Red | Check your folder path; verify .out files exist |
| **Analysis** | "Computed statistics for 4 variables using last 1000 of 5000 rows." | Green | Review results; 1000 samples is sufficient |
| **Analysis** | "No data in selected file." | Red | File could not be parsed; check format |
| **Batch** | "Batch complete: 5 variables across 3 cases." | Green | Review the batch table and cross-case charts |

The guidance panels are designed to be glanceable. If they are green, you are on track. If they turn yellow or red, read the message -- it tells you what to do.

---

## 18. Key Formulas Reference

### ITTC Iterative Uncertainty (Half-Range)

```
U_I = 0.5 x (S_max - S_min)
```

per ITTC 7.5-03-01-01. S_max and S_min are the maximum and minimum values over the analysis window.

### Standard Deviation (sigma)

```
sigma = sqrt( (1 / (N-1)) x sum( (S_i - mean)^2 ) )
```

Bessel's correction (N-1) is used for an unbiased sample estimate.

### Corrected 3-Sigma

```
corrected_3sigma = min(S_max, mean + 3 x sigma)
```

Prevents the 3-sigma bound from exceeding the observed maximum.

### Difference (Sigma-Method Iterative Uncertainty)

```
difference = corrected_3sigma - final_value
```

### Coefficient of Variation (CoV)

```
CoV = sigma / abs(mean)
```

Dimensionless. Values below 0.001 indicate excellent iterative convergence.

### Time-Weighted Mean

```
w_1     = dt_1 / 2
w_i     = (dt_{i-1} + dt_i) / 2     for i = 2, ..., N-1
w_N     = dt_{N-1} / 2

TW_mean = sum(w_i x S_i) / sum(w_i)
```

### Time-Weighted Standard Deviation

```
TW_sigma = sqrt( sum(w_i x (S_i - TW_mean)^2) / sum(w_i) )
```

### Unit Conversions

```
T_degF = (T_K - 273.15) x 9/5 + 32       (Kelvin to Fahrenheit)
P_psia = P_Pa / 6894.757                   (Pascals to psia)
```

### 95th Percentile (P95)

```
P95 = value below which 95% of the data falls
```

Computed using numpy's percentile function. Not a formula per se -- it is the 95th order statistic of the sorted data.

### Cross-Case Pooled Standard Deviation

```
pooled_sigma = std(final_1, final_2, ..., final_k, ddof=1)
```

where k is the number of cases.

### Cross-Case RSS of Per-Case Sigma

```
RSS_sigma = sqrt( (sigma_1^2 + sigma_2^2 + ... + sigma_k^2) / k )
```

where sigma_i is the iterative standard deviation for case i.

---

## 19. Standards References

| Standard | Full Title | How This Tool Uses It |
|----------|-----------|----------------------|
| **ITTC 7.5-03-01-01** (2024) | Uncertainty Analysis in CFD Verification and Validation, Methodology and Procedures | Half-range method for iterative convergence uncertainty: U_I = 0.5 x (max - min). The primary reference for the ITTC method. |
| **ASME V&V 20-2009 (R2021)** | Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer, Section 5 | Framework: iterative convergence uncertainty as a component of u_num. Defines where u_iter sits in the uncertainty budget. |
| **JCGM 100:2008 (GUM)** | Guide to the Expression of Uncertainty in Measurement | Statistical foundations: sample standard deviation, Type A evaluation from repeated observations, coverage factors, degrees of freedom. |
| **ASME PTC 19.1-2018** | Test Uncertainty | General uncertainty analysis methodology, sigma-basis conversions, bounding-to-sigma conversion factors. |
| **Celik et al. (2008)** | Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications, *J. Fluids Eng.* 130(7) | Context: distinguishes discretization (grid) uncertainty from iterative convergence uncertainty. The GCI Calculator handles discretization; this tool handles iterative convergence. |

---

## 20. Glossary

| Term | Plain English Definition |
|------|--------------------------|
| **Iterative convergence** | The process of a CFD solver approaching a steady answer through repeated iterations |
| **u_iter** | Iterative convergence uncertainty -- the uncertainty in the solution value due to imperfect convergence |
| **u_num** | Total numerical uncertainty -- includes grid convergence, time step sensitivity, iterative convergence, and round-off |
| **U_I** | ITTC iterative uncertainty -- specifically the half-range method: 0.5 x (max - min) |
| **Half-range** | Half the distance between the maximum and minimum values in the analysis window |
| **Sigma** | Standard deviation -- a measure of how spread out values are around the mean |
| **3-sigma** | Three times the standard deviation. For normal data, 99.7% falls within +/- 3-sigma of the mean |
| **Corrected 3-sigma** | min(max, mean + 3*sigma) -- caps the 3-sigma bound at the observed maximum to avoid exceeding real data |
| **CoV** | Coefficient of Variation = sigma / abs(mean). Dimensionless measure of relative scatter. CoV < 0.001 indicates good convergence |
| **P95** | The 95th percentile -- the value below which 95% of the data falls |
| **Last N** | The number of last iterations (or seconds of flow-time) used to compute statistics |
| **Analysis window** | The subset of iteration data used for statistics (the last N rows or last N seconds) |
| **Time-weighted** | Statistics computed with trapezoidal weights that account for non-uniform time step sizes |
| **Trapezoidal weights** | Quadrature weights derived from the midpoints of adjacent time intervals |
| **Flow-time** | The physical simulation time in a transient CFD run (seconds), as opposed to the iteration count |
| **Final value** | The last data point in the analysis window -- the value you would read off the last iteration |
| **Transient** | A simulation where the solution evolves over physical time, as opposed to steady-state |
| **Report Definition** | A Fluent feature that writes monitored quantities to a .out file every N iterations. The modern, preferred format. |
| **XY-plot** | A legacy Fluent export format using Scheme/Lisp parenthesized syntax with ((xy/key/label ...) ...) blocks |
| **Scheme/Lisp** | The scripting language used internally by Fluent; the XY-plot format uses its parenthesized syntax |
| **Pooled std dev** | The standard deviation of final values across multiple cases -- measures case-to-case variation |
| **RSS** | Root Sum of Squares -- a method for combining independent uncertainties: sqrt(sum of squares) |
| **Stationary state** | A signal whose statistical properties (mean, variance) do not change over time |
| **QQ plot** | Quantile-Quantile plot -- a graphical method for checking whether data follows a normal distribution |
| **Histogram** | A bar chart showing the frequency distribution of values in the analysis window |
| **Batch processing** | Analyzing all cases and files in a single operation, then computing cross-case statistics |
| **Cross-case** | Comparing or combining statistics across multiple cases (operating conditions, test points) |
| **JSON sidecar** | A metadata file (JSON format) exported alongside figures for audit trail traceability |
| **.itu file** | The project file format for this tool (JSON-based, extension .itu). Stores settings and metadata. |
| **.out file** | Fluent output file containing iteration history data for monitored quantities |
| **Catppuccin Mocha** | The dark color theme used by the application -- soft pastels on a dark blue-gray background |
| **V&V 20** | ASME V&V 20-2009 (R2021) -- the standard for verification and validation of CFD and heat transfer |
| **ITTC** | International Towing Tank Conference -- publishes naval/marine CFD verification procedures |
| **GUM** | Guide to the Expression of Uncertainty in Measurement (JCGM 100:2008) |
| **DOF** | Degrees of Freedom -- a measure of how much data supports an uncertainty estimate. Higher is better. |
| **Coverage factor (k)** | A multiplier that converts 1-sigma uncertainty to expanded uncertainty at a chosen confidence level (k=2 for ~95%) |
| **ddof=1** | "Delta degrees of freedom" = 1, meaning Bessel's correction is applied (N-1 denominator). Standard for sample statistics. |
| **Bessel's correction** | Using N-1 instead of N in the standard deviation denominator to correct for bias when estimating from a sample |

---

## 21. Frequently Asked Questions

### Q: What file formats does the tool support?

**A:** Two Fluent `.out` file formats: the modern Report Definition format (columnar data with quoted headers) and the legacy XY-plot format (Scheme/Lisp parenthesized). The tool auto-detects which format each file uses by examining the first 20 lines. You do not need to specify the format manually.

### Q: How many iterations do I need in my .out file?

**A:** Enough to establish a statistically stationary state. As a rule of thumb, your total iteration history should be at least 2x your analysis window (Last N). If Last N = 1000, you want at least 2,000 iterations total -- the first 1,000 to reach steady state and the last 1,000 for analysis. More is always better. If you are not sure, start with Last N = 1000 and look at the Full Time Series chart to verify the window falls in the stationary region.

### Q: Should I use the ITTC method or the sigma method?

**A:** If your program or certifying authority specifies a method, use that one. Otherwise, choose **Both** and use the tool's carry-over table output. In Both mode, the tool auto-selects the conservative value `max(sigma, ITTC)` so newer engineers do not have to make a subjective call.

### Q: I do not have Fluent files yet. Can I still validate the workflow?

**A:** Yes. Use **Data Import > Load Built-in Example Cases** (or **Examples > Load Built-in Example Cases**). This loads synthetic transient cases so you can test Analysis, Charts, Batch, and Report without external data.

### Q: The CoV is red (> 0.01). What should I do?

**A:** A red CoV means the iterative scatter is more than 1% of the mean value. Common causes and fixes:

1. **Insufficient iterations** -- the solver has not fully converged. Run more iterations.
2. **CFL too high** -- aggressive time-stepping causes oscillations. Reduce the CFL or Courant number.
3. **Solver settings** -- under-relaxation factors may be too aggressive. Tighten them.
4. **Physical oscillations** -- for some problems (unsteady flow past a bluff body, combustion instabilities), the oscillation is real physics, not numerical. Run a transient solver and time-average appropriately.
5. **Wrong window** -- you may be including the convergence ramp. Check the Full Time Series chart and increase Last N to exclude the startup transient.

### Q: Why is my velocity or Mach variable unit blank?

**A:** This is intentional. The tool does not auto-assign velocity units because CFD projects often mix SI and English unit systems. A blank unit is safer than a wrong conversion. If you need conversions for a specific variable, keep conversion enabled for temperature/pressure and handle velocity conversion explicitly in your post-processing workflow.

### Q: My simulation has physical oscillations (vortex shedding). Is that iterative uncertainty?

**A:** No. If you are running a transient simulation and the oscillations represent real physics (vortex shedding, combustion oscillations, etc.), those are the correct answer, not iterative error. Iterative uncertainty applies to the convergence of the solver within each time step, not to the physical time variation. For transient simulations, ensure your inner-iteration convergence is tight within each time step. The scatter in the time-averaged quantity over many shedding cycles is input uncertainty, not iterative uncertainty.

### Q: What does "corrected 3-sigma" mean? Why not just use raw 3-sigma?

**A:** The raw 3-sigma bound (mean + 3*sigma) can exceed the observed maximum value. For a bounded oscillation, this is unphysical -- the solution never actually reached that high. The correction caps the 3-sigma bound at the observed maximum: corrected_3sigma = min(max, mean + 3*sigma). This prevents overly conservative estimates driven by the distributional assumption rather than the actual data.

### Q: ITTC U_I is much larger than sigma. Which one should I trust?

**A:** Both are valid -- they answer slightly different questions. If your monitor has occasional spikes (e.g., pressure correction overshoots), the half-range captures those spikes while sigma is relatively robust to them. Check the histogram. If it looks approximately normal with no outliers, sigma is the better characterization. If there are clearly visible spikes or a heavy tail, ITTC U_I is more conservative and may be more appropriate. Report both and let your reviewer decide.

### Q: I have 50 .out files across 10 cases. Do I have to analyze each one individually?

**A:** No. Use the Batch tab. Click "Run Batch Processing" to analyze all files in all cases simultaneously. The batch process computes per-variable statistics for every file, writes per-case CSVs, and performs cross-case analysis to combine the results.

### Q: The time-weighted statistics differ significantly from the unweighted ones. Which should I use?

**A:** If your simulation has non-uniform time steps (adaptive time stepping), the time-weighted statistics are more physically meaningful. The unweighted statistics treat every row equally, which biases the result toward phases with many small time steps. Use the TW columns (TW Mean, TW sigma, TW CoV) for non-uniform transient data.

### Q: How do I know if my analysis window is large enough?

**A:** Two tests:

1. **Visual check:** Look at the Zoomed (Last N) chart. If the data shows a stable oscillation pattern with no drift, the window is adequate. If you see the mean trending up or down, the window includes startup transient data -- increase it to capture only the stationary portion.

2. **Sensitivity check:** Change Last N by +/- 50% and re-compute. If the sigma and mean do not change significantly, the window is robust. If they shift, the data is not stationary in the current window.

Also verify the window contains at least 5-10 complete oscillation cycles if the data has a periodic pattern.

### Q: Can I analyze files from solvers other than Fluent?

**A:** Not directly. The parser is designed for Fluent's two `.out` file formats. However, if you can format your data as a simple columnar file with one quoted-header line (mimicking Report Definition export), the tool can parse it. Metadata lines above the header are ignored, and numeric rows below the header can be space- or comma-separated.

### Q: My .out file has duplicate column headers. Does the tool handle that?

**A:** Yes. If two columns have the same header name (e.g., two "Temperature" columns from different monitors), the parser appends a numeric suffix to duplicates: "Temperature" and "Temperature_1". The original column order is preserved.

### Q: What happens if my .out file has NaN or missing values?

**A:** The tool handles NaN values gracefully. During parsing, non-numeric tokens are ignored and short rows are padded as needed, which can introduce NaNs in sparse columns. During statistics computation, NaN values are filtered out -- only finite values are used. The N column in the results table shows the number of valid (non-NaN) data points actually analyzed.

### Q: Can I compare results across different window sizes?

**A:** Yes. Change the Last N value and click Compute again. The results table updates immediately. Compare sigma, CoV, and ITTC U_I at different window sizes to check for sensitivity. If the values stabilize as you increase the window, your analysis is robust. If they keep changing, the solution may not be stationary in the window you are analyzing.

### Q: My Fluent case uses adaptive time-stepping. Does the tool handle that?

**A:** Yes -- this is exactly what time-weighted statistics are designed for. As long as the `.out` file includes a flow-time column (which Report Definition files from transient Fluent runs always do), the tool detects the non-uniform time steps and computes time-weighted mean, sigma, and CoV. Use the TW columns for your uncertainty estimate.

### Q: Where do the per-case CSV files get saved?

**A:** In the same folder as the `.out` files for that case. The filename is `stats_<casename>.csv`. For example, if your case folder is `C:\CFD\turbine_study\case_takeoff\`, the CSV is saved as `C:\CFD\turbine_study\case_takeoff\stats_case_takeoff.csv`.

### Q: How do I open the exported HTML report?

**A:** Double-click the `.html` file -- it opens in your default web browser. The report is fully self-contained (all styles inline, all images base64-encoded), so it works offline and can be emailed, archived, or printed directly from the browser.

### Q: What is the difference between "By Iterations" and "By Time (seconds)" windowing?

**A:** "By Iterations" takes the last N rows of data regardless of what time they represent. "By Time (seconds)" looks at the flow-time column and takes all rows within the last N seconds of physical time.

**Example:** Your transient run has 100 small-dt steps (0.001s each, covering 0.1 seconds) followed by 50 large-dt steps (0.1s each, covering 5 seconds). "By Iterations: 50" gives you the last 50 rows (5 seconds of physics). "By Time: 1" gives you the last 1 second of physics, which spans a different number of rows depending on the dt distribution.

The time-based option is essential for transient cases with variable time-step sizes where you want to analyze a consistent physical time window rather than a fixed number of rows.

### Q: Can I run the tool from the command line without the GUI?

**A:** The current version (v1.4.0) is a GUI application only. However, the core engine functions are defined at the module level and can be imported into a Python script for headless batch processing:

```python
from iterative_uncertainty import parse_fluent_out, compute_iterative_stats, scan_out_files

fd = parse_fluent_out("path/to/temperature.out")
vals = fd.data[:, 1]  # second column (first data variable)
stats = compute_iterative_stats(vals, variable="Temperature", unit="K")
print(f"ITTC U_I = {stats.half_range:.4f}")
print(f"Sigma    = {stats.sigma:.4f}")
print(f"CoV      = {stats.cov:.6f}")
```

### Q: Is the .itu project file human-readable?

**A:** Yes. It is a standard JSON file with indented formatting. You can open it in any text editor to inspect or manually edit the saved settings. The file contains only configuration (root folder path, Last N, method, project metadata) -- no computed results.

---

*Iterative Uncertainty Calculator v1.4.0 -- Built for engineers who need defensible iterative convergence numbers, not statistics PhDs.*
*Standards: ITTC 7.5-03-01-01, ASME V&V 20-2009 (R2021), JCGM 100:2008*
