# Statistical Analyzer v1.4.0 — User Guide & Technical Reference

**Data Distribution Analysis Tool for ASME V&V 20 Uncertainty Budgets**

*Written for engineers — not statisticians.*

---

## Revision Update (v1.4.0)

- Clipboard paste now anchors to the **selected cell/column** (not forced to column 1).
- Added autocorrelation-aware `N_eff` and lag-1 correlation reporting.
- Added **candidate-set logic** for near-tied distributions (delta AICc <= 2) with model-mixture guidance.
- Added **sparse replicate mode** (`N <= 8`) with empirical/bootstrap carry recommendation and CSV export.
- Added **automatic GOF runtime guardrails**:
  - sparse datasets use KS screening automatically (no manual tuning needed),
  - larger datasets use adaptive bootstrap sample counts to avoid long UI stalls.
- Added **Decision Consequence** in project metadata and expanded HTML report sections:
  - Decision card,
  - Credibility framing,
  - VVUQ glossary panel,
  - Conformity template.

### Quick Example (Non-Normal Data, No Statistics Expert Needed)

1. Paste data into the selected output column.
2. Run **Compute Statistics**.
3. If the tool flags candidate ambiguity or sparse replicate mode, follow the table recommendation:
   - Distribution: `Custom/Empirical (Bootstrap)`
   - Method: `Monte Carlo`
4. Copy the carry row directly to the Aggregator without forcing a manual Log-Normal/Weibull decision.

---

## Table of Contents

1. [What Is This Tool and Why Do I Need It?](#1-what-is-this-tool-and-why-do-i-need-it)
2. [The Role of Statistics in V&V 20](#2-the-role-of-statistics-in-vv-20)
3. [Getting Started — Application Layout](#3-getting-started--application-layout)
4. [Step-by-Step: Analyzing Your Data](#4-step-by-step-analyzing-your-data)
5. [Understanding the Results](#5-understanding-the-results)
6. [Distribution Fitting Explained](#6-distribution-fitting-explained)
7. [Goodness-of-Fit Tests](#7-goodness-of-fit-tests)
8. [Charts and Visualization](#8-charts-and-visualization)
9. [Carry-Over Summary — What Goes Into the VVUQ Aggregator](#9-carry-over-summary--what-goes-into-the-vvuq-aggregator)
10. [Saving and Loading Projects](#10-saving-and-loading-projects)
11. [HTML Report Generation](#11-html-report-generation)
12. [Figure Export](#12-figure-export)
13. [When Do I Need This Tool?](#13-when-do-i-need-this-tool)
14. [Project Info Panel](#14-project-info-panel)
15. [Reading the Guidance Panels](#15-reading-the-guidance-panels)
16. [Sample Size Guidance](#16-sample-size-guidance)
17. [Key Formulas Reference](#17-key-formulas-reference)
18. [Standards References](#18-standards-references)
19. [Glossary](#19-glossary)
20. [Frequently Asked Questions](#20-frequently-asked-questions)

---

## 1. What Is This Tool and Why Do I Need It?

If you have experimental measurements or repeated simulation outputs, you need to know how much scatter is in your data and what that scatter means for your uncertainty budget. The Statistical Analyzer answers that question.

**The problem:** In CFD validation, you compare simulation to experiment. But experimental data has scatter — instrument noise, repeatability variation, environmental drift. Before you can say "my CFD matches the data," you need to quantify how much uncertainty is embedded in the data itself. Similarly, if you run the same CFD case with slightly perturbed input parameters, the outputs will vary, and you need to characterize that variation statistically.

**Why it matters:** In the ASME V&V 20 framework, experimental uncertainty (u_D) and input uncertainty (u_input) are built from standard deviations, degrees of freedom, and distribution assumptions. If you get the statistics wrong — use the wrong distribution, undercount the DOF, or ignore skewness — your entire uncertainty budget is flawed.

**What this tool does:** You feed in your data (from a CSV, the clipboard, or by typing it in), and the tool tells you:

> *"Your data has a standard deviation of X, best fits a Y distribution, and the standard uncertainty to carry into the VVUQ Aggregator is Z with DOF = W."*

That standard uncertainty, distribution recommendation, and DOF go directly into your V&V 20 uncertainty budget via the VVUQ Uncertainty Aggregator.

**The problem it solves:** Engineers often guess at distribution shapes or assume everything is Normal without checking. This tool removes the guesswork. It fits 8 candidate distributions using Maximum Likelihood Estimation, ranks them by AICc (small-sample corrected AIC), and applies an automatic GOF policy (bootstrap when appropriate, KS screening when data is sparse) so the analysis stays both defensible and practical.

> **Common Mistakes to Avoid**
>
> The Statistical Analyzer automates a lot of the statistics, but you still need to pay attention. These are the mistakes that trip up new users the most:
>
> - **Forcing a distribution when none fits (the GOF test fails).** If the goodness-of-fit test fails for a distribution, that distribution does not describe your data well. Do not pick it anyway just because it is familiar. Use the "Custom/Empirical (Bootstrap)" option instead, or collect more data.
> - **Using too few data points and trusting the fit.** With 5 or 6 data points, any distribution can appear to "fit" because there is not enough data to tell distributions apart. The tool will flag sparse replicate mode when N is very small. Take the warning seriously -- carry an empirical/bootstrap value rather than a parametric fit.
> - **Ignoring the "ALL GOF FAILED" warning.** When every candidate distribution fails the goodness-of-fit test, it means your data does not match any standard shape. This is not a minor footnote -- it means a parametric assumption could badly misrepresent your uncertainty. Switch to Monte Carlo with an empirical distribution in the Aggregator.
> - **Assuming Normal distribution without testing.** "It is probably Normal" is not a defensible basis for a certification analysis. Run the tool and check. Many real-world datasets (especially material properties and manufacturing tolerances) are skewed, and a Normal assumption can underestimate tail behavior.
> - **Not understanding what the carry-over values mean.** The carry-over table gives you a sigma, a DOF, and a recommended distribution to enter into the Aggregator. The sigma is a standard uncertainty (1-sigma), not an expanded value. The DOF tells the Aggregator how much confidence to place in that sigma. If you enter the sigma but ignore the DOF or distribution recommendation, the downstream analysis will be wrong.

---

## 2. The Role of Statistics in V&V 20

The ASME V&V 20 standard (Section 7) requires that experimental uncertainties be evaluated using **Type A** methods (statistical analysis of repeated observations, per JCGM 100:2008 Section 4.2) or **Type B** methods (engineering judgment, manufacturer specs, handbooks — per Section 4.3).

This tool handles **Type A evaluation**: you have N repeated measurements or simulation outputs, and you need to extract:

1. **Standard deviation** (the population scatter)
2. **Standard uncertainty of the mean** (how well you know the average)
3. **Degrees of freedom** (how much confidence to place in the standard deviation estimate)
4. **Distribution shape** (is it Normal? Skewed? Heavy-tailed?)

### Why Distribution Shape Matters

The V&V 20 validation comparison (Section 8) uses an RSS combination of uncertainties. That RSS assumes each source contributes a roughly symmetric, well-characterized uncertainty. If your data is heavily skewed or has fat tails, using a Normal assumption underestimates the extreme-percentile behavior. The tool warns you when this happens and recommends a better distribution.

### Type A vs. Type B — Which Do I Use?

| Source | Evaluation Type | Use This Tool? |
|--------|----------------|----------------|
| Repeated thermocouple readings | Type A | **Yes** — load the readings and compute |
| Manufacturer's stated accuracy (e.g., +/- 0.5 K) | Type B | **No** — enter directly into the Aggregator as a Uniform or Normal distribution |
| CFD outputs from perturbed input parameters | Type A | **Yes** — load the outputs and compute |
| Engineering judgment ("I think it's about 2%") | Type B | **No** — enter directly into the Aggregator |

**Bottom line:** If you have data points, use this tool to analyze them. If you have a specification or judgment call, enter it directly into the Aggregator.

---

## 3. Getting Started — Application Layout

### How to Run

```
python statistical_analyzer.py
```

The application requires Python 3 with PySide6, NumPy, SciPy, and Matplotlib. On first launch, if any packages are missing, the tool will offer to install them automatically.

### Main Window

At the top of the window is a collapsible **Project Info** bar (click the arrow to expand). This lets you record project metadata for traceability. It is collapsed by default to maximize screen space.

Below the project bar, the application has **five tabs**:

| Tab | Icon | Purpose |
|-----|------|---------|
| **Data Input** | 📥 | CSV import, clipboard paste, editable table for manual entry |
| **Statistics** | 📊 | Summary statistics, distribution fitting results, recommendation engine |
| **Charts** | 📈 | Histogram with best-fit overlay, QQ plot, CDF, box plot |
| **Carry-Over Summary** | 📋 | Table showing exactly what to enter into the VVUQ Aggregator |
| **Reference** | 📖 | Distribution guide, sample size guidance, GUM references, glossary |

### Theme

The application uses a **dark Catppuccin Mocha** theme throughout — dark backgrounds with high-contrast text and color-coded indicators (green/yellow/red) for guidance panels. Charts use the same dark palette for on-screen display and automatically switch to a light-on-white palette when exporting for publication or reports.

### Menu Bar

**File Menu:**

| Menu Item | Shortcut | Action |
|-----------|----------|--------|
| New Project | Ctrl+N | Clear all data and start fresh |
| Open Project... | Ctrl+O | Load a previously saved .sta project file |
| Save Project | Ctrl+S | Save to the current .sta file (or Save As if no file yet) |
| Save Project As... | Ctrl+Shift+S | Save to a new .sta project file |
| Export HTML Report... | Ctrl+H | Generate a self-contained HTML report |
| Exit | Ctrl+Q | Close the application |

**Analysis Menu:**

| Menu Item | Shortcut | Action |
|-----------|----------|--------|
| Recompute Statistics | Ctrl+R | Run statistics on the current data |

**Examples Menu:**

| Menu Item | Action |
|-----------|--------|
| Load Built-in Example Dataset | Loads a ready-to-run CFD-style dataset (normal, skewed-positive, bounded [0,1]) |

**Help Menu:**

| Menu Item | Action |
|-----------|--------|
| About | Version info, tool description, standards list |

The window title shows the project name. The status bar at the bottom displays the tool name, version, build date, and transient status messages (e.g., "Analysis complete: 3 variable(s) processed").

### Unit Selection

Each data column can have an associated unit. The tool provides preset unit categories:

| Category | Available Units |
|----------|----------------|
| **Temperature** | K, °C, °F, R |
| **Pressure** | Pa, kPa, psia, psig, bar, atm |
| **Velocity** | m/s, ft/s, ft/min |
| **Mass Flow** | kg/s, lb/s, lb/min |
| **Force** | N, kN, lbf |
| **Length** | m, mm, in, ft |
| **Dimensionless** | — |
| **Other** | (custom — type your own) |

Units are cosmetic labels — they appear in column headers, results tables, chart axes, and reports. The tool does **not** perform unit conversion. If your data is in degrees Celsius and you label it Kelvin, the numbers will not be converted. Make sure your data values match the selected unit.

---

## 4. Step-by-Step: Analyzing Your Data

### Step 1: Enter Your Data

Navigate to the **Data Input** tab. You have three ways to get data in:

**Option A — Load a CSV file:**
1. Click **"Load CSV..."**
2. Select your file. The tool accepts comma-delimited and tab-delimited files.
3. Header detection is automatic: if the first row contains non-numeric text, it is treated as column headers. Otherwise, columns are named "Variable 1," "Variable 2," etc.

**Option B — Paste from clipboard:**
1. Copy your data in Excel, Google Sheets, or any spreadsheet (select the cells, Ctrl+C).
2. Click **"Paste from Clipboard."**
3. Tab-separated data is detected automatically. Headers are detected the same way as CSV import.
4. Paste starts at the currently selected table cell (row + column anchor). This lets you fill column 2, 3, etc. without overwriting column 1.

**Option C — Load built-in example data:**
1. Click **"Load Example Dataset"** in the Data Input tab, or use **Examples > Load Built-in Example Dataset**.
2. The tool loads 3 CFD-style variables:
- Wall Temperature (near-normal)
- Pressure Drop (right-skewed positive)
- Cooling Effectiveness (bounded in [0,1])
3. Press **Ctrl+R** to run the full workflow immediately.

**Option D — Type directly:**
1. Click into the table cells and type values.
2. Use the **Row Management** group to set the number of rows if you need more than the default 20.

**Example CSV for a thermocouple calibration study:**

```csv
TC-01 (K), TC-02 (K), TC-03 (K)
305.2, 310.8, 298.4
305.5, 311.1, 298.7
304.9, 310.6, 298.2
305.3, 310.9, 298.5
305.1, 310.7, 298.3
305.4, 311.0, 298.6
305.0, 310.5, 298.1
305.6, 311.2, 298.8
305.2, 310.8, 298.4
305.3, 310.9, 298.5
```

### Step 2: Set Column Properties

For each column, you can set:
- **Name:** A descriptive label (e.g., "TC-01 Reading" or "Exit Temperature")
- **Unit Category:** Select from the preset categories (Temperature, Pressure, etc.)
- **Unit:** Choose from the category's presets or type a custom unit

Use the **Active Column** dropdown to switch between columns. The column name and unit appear in the table header as "Name (Unit)" — for example, "TC-01 Reading (K)."

### Step 3: Add or Remove Columns

- Click **"+ Add Column"** to add a new variable column
- Click **"- Remove Column"** to remove the currently selected column (you cannot remove the last column)

### Step 4: Compute Statistics

Switch to the **Statistics** tab and click the blue **"Compute Statistics"** button.

The tool computes full statistics for every column that has at least 2 valid values. Results appear in two tables:

1. **Summary Statistics** — the key numbers for each variable
2. **Distribution Fitting Results** — all 8 candidate distributions ranked by AICc

A **Recommendation** guidance panel at the bottom provides a plain-English assessment.

### Step 5: Review the Charts

Switch to the **Charts** tab. Select your variable from the dropdown and cycle through the four chart types (Histogram, QQ Plot, CDF, Box Plot) to visually confirm the statistical results.

### Step 6: Read the Carry-Over Summary

Switch to the **Carry-Over Summary** tab. This table is auto-populated after analysis and shows **exactly what to enter** into the VVUQ Uncertainty Aggregator for each variable.

### Step 7: Enter Values into the VVUQ Aggregator

For each row in the Carry-Over Summary:
1. Open the VVUQ Uncertainty Aggregator
2. Add a new uncertainty source
3. Set the category to **"Experimental (u_D)"** or **"Input/BC (u_input)"** as appropriate
4. Set the input type to **"Sigma Value Only"**
5. Enter the sigma value from the Carry-Over table
6. Set sigma basis to **"Confirmed 1σ"**
7. Enter the DOF from the Carry-Over table
8. Note the distribution recommendation for your records

### Step 8: Save Your Project

Use **File > Save Project** (Ctrl+S) to save all data, settings, and metadata to a `.sta` file. See [Section 10](#10-saving-and-loading-projects) for details.

---

## 5. Understanding the Results

The Statistics tab shows a comprehensive breakdown for each variable. Use the **Display** dropdown to switch between variables when you have multiple columns. Here is what each row in the Summary Statistics table means.

### Summary Statistics Table

| Statistic | What It Means | Example |
|-----------|--------------|---------|
| **Variable** | The column name you assigned | "TC-01 Reading" |
| **Unit** | The unit label | "K" |
| **Sample Size (N)** | Number of valid (non-blank, non-NaN) data points | 50 |
| **Degrees of Freedom** | N_eff - 1, where N_eff is the autocorrelation-adjusted effective sample size (see [Section 9.2](#92-autocorrelation-adjustment)). Used for Student-t coverage factors and Welch-Satterthwaite in the Aggregator. When data has no autocorrelation, N_eff = N and this reduces to N - 1. | 49 |
| **Mean** | Arithmetic average of all values | 305.25 |
| **Median** | Middle value when sorted. Robust to outliers. | 305.20 |
| **Std Deviation (sigma)** | Sample standard deviation with Bessel correction (ddof=1). This is the population scatter. | 0.21 |
| **Minimum** | Smallest observed value | 304.9 |
| **Maximum** | Largest observed value | 305.6 |
| **Range** | Max minus Min | 0.7 |
| **Skewness** | Measure of asymmetry. Zero means symmetric. Positive means right tail is longer. | 0.15 |
| **Excess Kurtosis** | Measure of tail weight relative to Normal. Zero means Normal-like tails. Positive means heavier tails (more outlier-prone). | -0.42 |
| **95% CI for Mean (low / high)** | 95% confidence interval for the population mean, computed using the Student-t distribution: mean +/- t(0.975, DOF) * sigma/sqrt(N_eff). When autocorrelation is negligible, N_eff = N. | [305.19, 305.31] |
| **Std Uncertainty of Mean (sigma/sqrt(N_eff))** | The standard error of the mean using the autocorrelation-adjusted effective sample size. This is how precisely you know the average value. | 0.030 |
| **Std Uncertainty (population sigma)** | The sample standard deviation. This is the spread of individual values. | 0.21 |
| **Shapiro-Wilk p-value** | Normality test. p > 0.05 means the data is consistent with a Normal distribution. p < 0.05 means it is not. | 0.72 |
| **Normally Distributed?** | Yes (p >= 0.05) or No (p < 0.05) | Yes |
| **Best-Fit Distribution** | The distribution with the lowest AICc among all 8 candidates | Normal |

### Which Uncertainty to Use?

This is the most common question. The answer depends on what your quantity of interest is:

| Your Quantity of Interest | Use This Uncertainty | Why |
|--------------------------|---------------------|-----|
| **The mean value** (e.g., "the average temperature at this location is 305.2 K") | sigma / sqrt(N) — the standard uncertainty of the mean | You are interested in how well the average is known, not the spread of individual readings |
| **Individual values** (e.g., "any single reading could be this far off") | sigma — the population standard deviation | You are characterizing the spread itself |
| **Monte Carlo input** (e.g., "perturbed CFD input parameter") | sigma — the population standard deviation | MC sampling needs the full distribution width |

The Carry-Over Summary tab uses the **population sigma** by default, because this is the most conservative choice and the one most commonly needed for V&V 20 budgets. If you need the standard uncertainty of the mean instead, divide the reported sigma by sqrt(N).

---

## 6. Distribution Fitting Explained

The tool fits **8 candidate distributions** to your data using **Maximum Likelihood Estimation (MLE)** and ranks them by AICc.

### The 8 Distributions

| Distribution | When to Use It | Key Parameters | Data Constraint |
|-------------|---------------|----------------|-----------------|
| **Normal** | Default for most measurement uncertainties. Symmetric bell curve. Instrument errors, repeated readings, well-behaved quantities. | mu (mean), sigma (std dev) | None |
| **Log-Normal** | Positive-only data with right skew. Flow rates, concentrations, material properties, fatigue life. | s, loc, scale | Data must be > 0 |
| **Uniform** | All values equally likely within bounds. Digitization error, rounding, manufacturer tolerance bands. | a (lower bound), b (upper bound) | None |
| **Triangular** | You know the min, max, and most likely value. Limited data with engineering judgment about bounds and mode. | a (min), c (mode), b (max) | None |
| **Weibull** | Failure and lifetime data, wind speeds, material strength. Flexible shape handles various skewness patterns. | k (shape), lambda (scale) | Data must be > 0 |
| **Gamma** | Wait times, rainfall, positive skewed data. Sum of exponential random variables. | alpha (shape), beta (rate) | Data must be > 0 |
| **Student-t** | Small samples (N < 30) where population sigma is unknown. Heavier tails than Normal. | nu (DOF), loc, scale | None |
| **Beta** | Data bounded on [0, 1]. Proportions, probabilities, percentages, efficiency ratios. | alpha, beta (shape parameters) | Data must already be in [0,1] |

### How MLE Works (Plain English)

For each candidate distribution, MLE finds the parameter values that make your observed data the "most likely" outcome. Imagine you are adjusting the shape of a probability curve until it best matches the histogram of your data. The parameter values at the best match are the MLE estimates.

### Why 8 Distributions?

Because not all engineering data is Normal. Pressure losses in piping systems tend to follow Log-Normal distributions. Turbine blade failure data is typically Weibull. Rounding errors from digitized instruments are Uniform. By fitting all 8, the tool lets you see which distribution actually describes your data, rather than assuming Normal and hoping for the best.

### Distributions That Require Positive Data

Log-Normal, Weibull, and Gamma require all data values to be strictly positive (> 0). If your dataset contains zero or negative values, these three distributions are automatically skipped and will not appear in the ranking table.

### Beta Distribution Constraint

Beta distribution is only attempted when the original data is truly bounded in [0, 1] (for example probabilities or efficiency). Arbitrary positive variables are **not** min-max scaled into Beta, because that can create misleading fits.

### Minimum Data Requirement

Distribution fitting requires at least **N = 5** data points for MLE to produce meaningful parameter estimates. With fewer than 5 points, the distribution fitting table will be empty. Basic statistics (mean, std, min, max) are still computed for N >= 2.

---

## 7. Goodness-of-Fit Tests

After fitting each distribution, the tool evaluates how well it actually matches the data using multiple tests.

### Corrected Akaike Information Criterion (AICc)

AICc measures the relative quality of a statistical model — lower is better. It balances goodness of fit against model complexity (number of parameters) and adds a small-sample correction. A model that fits the data well but has fewer parameters is preferred over one that fits slightly better but requires more parameters.

```
AIC = 2k - 2 ln(L)
AICc = AIC + [2k(k+1)] / (N-k-1)
```

Where k = number of parameters, L = maximum likelihood, and N = sample size. The distribution with the lowest AICc is ranked #1.

**Rule of thumb for comparing:**
- Delta AICc < 2 between two distributions: essentially tied — either is fine
- Delta AICc 2-10: moderate evidence favoring the lower one
- Delta AICc > 10: strong evidence favoring the lower one

### Bayesian Information Criterion (BIC)

BIC is similar to AICc but penalizes model complexity more heavily, especially for large sample sizes:

```
BIC = k ln(N) - 2 ln(L)
```

BIC tends to prefer simpler models than AICc. Both are reported so you can compare. When AICc and BIC disagree, the simpler distribution (per BIC) is often the safer choice for uncertainty analysis — unless you have a physical reason to expect a more complex shape.

### GOF Policy (Automatic Guardrails)

The tool chooses the GOF method automatically so junior analysts do not need to tune statistical settings.

- **Bootstrap AD mode (default for adequate N):** primary pass/fail mode.
- **KS screening mode (`KS*`):** used automatically for sparse replicate cases to keep runtime practical.
- **KS fallback mode (`KS†`):** used only when bootstrap fails for numerical reasons.

Pass/fail rule is consistent across modes:

- **GOF p-value > 0.05:** distribution passes
- **GOF p-value <= 0.05:** distribution fails

If no candidate passes, the tool falls back to the lowest-AICc model and clearly marks that fallback.

### KS Diagnostic (Context)

KS values are always shown for context. In `KS*` and `KS†` rows, KS is the active GOF method.

### Anderson-Darling (AD) Test

The AD test gives more weight to distribution tails. It is available for **Normal** and **Log-Normal** distributions only (for Log-Normal, the test is applied to the log-transformed data).

- **AD statistic < critical value (5%):** Pass (green checkmark)
- **AD statistic >= critical value (5%):** Fail (red X)

The critical value at the 5% significance level is shown alongside the test statistic in the table (e.g., "0.342 (crit=0.752)"). This is especially useful when tail behavior drives certification decisions.

For distributions other than Normal and Log-Normal, the AD columns show a dash, indicating the test was not performed.

### Shapiro-Wilk Normality Test

This is a dedicated normality test run on the raw data (independent of any fitted distribution). It is valid for sample sizes between 3 and 5000.

- **p > 0.05:** Data is consistent with normality
- **p < 0.05:** Data deviates significantly from normal

The result appears in the Summary Statistics table as "Shapiro-Wilk p-value" and "Normally Distributed? Yes/No."

### Reading the Distribution Fitting Table

The table includes these key GOF columns:

| Column | Meaning |
|--------|---------|
| **Rank** | Overall ranking by AICc (1 = best) |
| **Distribution** | Name of the candidate distribution |
| **Parameters** | MLE-estimated parameters in human-readable form |
| **AICc** | Corrected Akaike Information Criterion (lower = better) |
| **BIC** | Bayesian Information Criterion (lower = better) |
| **GOF p-value** | Primary pass/fail p-value (bootstrap AD, KS*, or KS† depending on mode) |
| **GOF Pass?** | Green check (p > 0.05) or red X (p <= 0.05) |
| **GOF Stat** | GOF statistic from the active mode |
| **KS p (diag)** | KS diagnostic p-value (not primary pass/fail) |
| **AD Stat** | Anderson-Darling statistic with critical value (Normal/Log-Normal only) |
| **AD Pass?** | Green check or red X (Normal/Log-Normal only) |

**Best practice:** Look at the lowest-rank distribution that passes GOF. If multiple distributions have similar AICc values (within ~2 units), they are statistically close — pick the one with the strongest physical meaning for the variable.

---

## 8. Charts and Visualization

The Charts tab provides four visualization types. Use the **Variable** dropdown to select which column to plot, and the **Chart** dropdown to switch between chart types.

### Histogram with Best-Fit Overlay

Shows the probability density histogram of your data with the best-fit distribution curve overlaid in orange. Also displays vertical dashed lines at the mean (green) and +/- one sigma (yellow).

**What a good histogram looks like:** The fitted curve follows the histogram bars closely. The data looks roughly bell-shaped (or follows the shape of the recommended distribution).

**What a bad histogram looks like:** Large gaps between the curve and the bars, or a shape that clearly does not match any standard distribution (e.g., bimodal with two peaks — which would suggest your data is actually two populations mixed together).

The number of histogram bins is computed automatically as sqrt(N), clamped between 5 and 50.

### QQ (Quantile-Quantile) Plot

Compares your data quantiles against theoretical Normal quantiles. Points falling on the reference line indicate Normal distribution. Deviations reveal:

- **S-shaped curve:** Heavy tails (more extreme values than Normal predicts) — consider Student-t
- **Concave or convex arc:** Skewness (asymmetry) — consider Log-Normal or Weibull
- **Points peeling away at the ends:** Outliers or tail behavior different from Normal
- **Discrete steps:** Possible rounding or quantization in the data

The reference line is fitted through the interquartile range of the data (Q1 to Q3), which is robust to outliers at the extremes.

### CDF (Cumulative Distribution Function)

Shows the empirical CDF (step function in blue) against a fitted Normal CDF (dashed orange curve). The vertical gap between them is what the KS test measures.

**Practical use:** If you need to read off percentile values (e.g., "95% of readings are below X"), this is the chart to use. Follow the horizontal line at 0.95 on the y-axis to where it meets the step function, then read down to the x-axis.

### Box Plot

Shows the median (orange line), interquartile range (box), whiskers (1.5 x IQR), and outliers (red dots beyond the whiskers).

**Practical use:** Quick visual check for outliers and data symmetry. If the median is not centered in the box, the data is skewed. Red dots beyond the whiskers are flagged outliers — investigate them.

### All Charts Use the Dark Theme On-Screen

On-screen, charts use the dark Catppuccin Mocha palette for consistency with the application. When exported (see [Section 12](#12-figure-export)), they automatically switch to a white background with dark text suitable for publication and print.

### Matplotlib Toolbar

Below each chart, a standard Matplotlib navigation toolbar provides zoom, pan, home (reset view), and save-to-file functions.

---

## 9. Carry-Over Summary — What Goes Into the VVUQ Aggregator

The **Carry-Over Summary** tab is the bridge between this tool and the VVUQ Uncertainty Aggregator. After running the analysis, this table is auto-populated with one row per variable.

### Table Columns

| Column | What It Shows | Example |
|--------|-------------|---------|
| **Source Name** | The variable name from your data column | "TC-01 Reading" |
| **sigma (Std Uncertainty)** | The population standard deviation (sigma, with Bessel correction) | 0.213 |
| **Best Fit (This Tool)** | The best-fit distribution from the analyzer | "Log-Normal" |
| **Auto Dist / Auto Method** | Tool-recommended carry-over mapping for the Aggregator (`Auto Method` is the recommended Aggregator analysis mode) | "Lognormal" / "Monte Carlo" |
| **Override Dist / Override Method** | Optional analyst override (if needed) | "Normal" / "RSS" |
| **Override Rationale** | Required text when override is used | "Program requirement to force Normal model." |
| **Final Dist / Final Method** | Final values to carry into the Aggregator (auto or override) | "Normal" / "RSS" |
| **DOF** | Degrees of freedom = N_eff - 1 (autocorrelation-adjusted) | 49 |
| **Sample Size** | Number of valid data points (N) | 50 |
| **Notes** | Why the carry-over mapping was chosen | "Gamma not native in the Aggregator; bootstrap recommended" |

### How to Enter These Values in the Aggregator

For each row in the Carry-Over Summary, open the VVUQ Uncertainty Aggregator and set:

| Aggregator Field | Enter This |
|-----------------|------------|
| **Source Name** | Copy the Source Name column |
| **Category** | "Experimental (u_D)" for test data, "Input/BC (u_input)" for perturbed-parameter data |
| **Input Type** | "Sigma Value Only" |
| **Sigma Value** | Copy the sigma column value |
| **Sigma Basis** | "Confirmed 1σ" |
| **DOF** | Copy the DOF column value |
| **Distribution** | Copy **Final Dist (Use This)** exactly |
| **Aggregator Analysis Mode** | Aggregator propagation mode is analysis-level (set once). If any row shows **Final Method = Monte Carlo**, run the Aggregator in Monte Carlo mode. If all rows show RSS, use RSS. |

### Copy to Clipboard

Click **"Copy Table to Clipboard"** to copy the entire table as tab-separated text. You can paste this directly into Excel, a text document, or your notes while filling out the Aggregator.

### When the Notes Column Says "Low sample size"

This means N < 10. The GUM (Section G.3) recommends at least 10 observations for a reliable Type A evaluation. With fewer than 10, the standard deviation estimate has high uncertainty itself, and distribution fitting is unreliable. Consider using a Student-t distribution with low DOF for more conservative coverage.

### When the Notes Column Says "Non-normal"

This means the Shapiro-Wilk test rejected the Normal distribution hypothesis (p < 0.05). Check **Final Dist / Final Method** and consider:
- If the tool provides an Aggregator distribution mapping, use that mapping directly.
- If the tool says to use **Custom/Empirical (Bootstrap)**, run the Aggregator with Monte Carlo and bootstrap enabled.
- Investigate the root cause (outliers, mixed populations, drift) before final sign-off.

### When the Notes Column Explains a Mapping Decision

This appears when the best-fit distribution is not directly available in the Aggregator naming/options (for example Gamma or Beta). The note tells you the exact fallback path so a new engineer can proceed without guessing.

### If No Distribution Fits (Simple Rule)

> **This is one of the most important sections in this guide.** Read this even if you skip everything else.

Sometimes none of the 8 candidate distributions will pass goodness-of-fit testing. The tool will show every row in the GOF Pass column as **"No"** in red, and the recommendation panel will turn **red**.

**What this means in plain language:** Your data does not match any of the standard statistical shapes (Normal bell curve, skewed Lognormal, etc.). This is not a failure of the tool or your data — it just means your data has a shape that does not fit a textbook pattern.

**What to do (three simple steps):**

1. Set **Final Dist** = `Custom/Empirical (Bootstrap)`
2. Set **Final Method** = `Monte Carlo`
3. Keep sigma and DOF from this tool and document:
   `"No candidate distribution passed GOF; empirical bootstrap carry-over used."`

**Why this is safe:** Instead of forcing your data into a wrong shape (which would give a wrong uncertainty number), the bootstrap method uses your actual data values directly. The Aggregator will resample from your real data during Monte Carlo propagation, which is more honest than pretending the data is Normal when it is not.

**What NOT to do:**
- Do NOT override to "Normal" just because it is simpler. If the GOF test rejected Normal, forcing it will give an unreliable uncertainty estimate.
- Do NOT re-run the analysis with fewer distributions hoping one will pass. The GOF test is protecting you.
- Do NOT ignore the red warning and carry a parametric distribution anyway.

**When does this happen most often?**
- Multi-modal data (e.g., two clusters of values from different operating conditions)
- Data with hard physical bounds (e.g., efficiency between 0 and 1 but clustered near 0.95)
- Very small samples (N < 10) where no distribution can be confirmed
- Data with strong outliers that skew the fit

**For your supervisor/reviewer:** The "Custom/Empirical (Bootstrap)" path is the most conservative and defensible option per JCGM 101:2008. It avoids parametric assumptions entirely and is appropriate when data does not conform to standard distributions.

---

## 10. Saving and Loading Projects

### Project Files (.sta)

The Statistical Analyzer saves projects as `.sta` files — plain JSON files containing all input data, column metadata, and project information.

**What is saved:**
- Tool name and version
- UTC timestamp
- All project metadata (program, analyst, date, notes)
- All data columns (names, units, unit categories, values)

**What is NOT saved:**
- Computed results (re-compute after loading by clicking "Compute Statistics")
- Chart state (charts are regenerated from data)

This design ensures that results always reflect the current version of the tool's algorithms. If a bug is fixed or an algorithm is improved, reopening an old project and recomputing gives you the benefit of the latest code.

### Saving

| Action | How |
|--------|-----|
| Save (overwrite current file) | File > Save Project (Ctrl+S) |
| Save to a new file | File > Save Project As... (Ctrl+Shift+S) |

If you have not saved before, Save redirects to Save As. The `.sta` extension is added automatically if you omit it.

### Loading

1. File > Open Project... (Ctrl+O)
2. Select a `.sta` file
3. The data and metadata are loaded into the Data Input tab
4. Click **"Compute Statistics"** on the Statistics tab to regenerate results

### Starting Fresh

File > New Project (Ctrl+N) clears all data, resets metadata fields, and sets the date to today.

### File Naming Convention

A good naming convention for traceability:

```
ThermalCal_TC01_2026-02-22.sta
InletPressure_Phase3_50pts.sta
MaterialYieldStrength_Coupon_Test.sta
```

---

## 11. HTML Report Generation

The tool generates a **self-contained HTML report** suitable for archival, review, or inclusion in a V&V report package.

### How to Generate

1. Run the analysis first (click "Compute Statistics" on the Statistics tab)
2. File > Export HTML Report...
3. Choose a save location and filename
4. The report is written as a single `.html` file that opens in any web browser

### Report Sections

The HTML report contains 10 sections:

| Section | Contents |
|---------|----------|
| **1. Project Information** | Program, analyst, date, generation timestamp, tool version, notes |
| **2. Executive Summary** | Number of variables analyzed, methodology overview |
| **3. Summary Statistics** | Full table for all variables: N, mean, median, sigma, min, max, skewness, kurtosis, normality, best fit, GOF p-value, uncertainty, DOF |
| **4. Charts** | Embedded chart image (the currently displayed chart from the Charts tab, rendered with white background) |
| **5. Distribution Fitting Details** | MLE methodology description |
| **6. Recommendations** | Color-coded recommendation text for each variable |
| **7. Carry-Over Summary** | Table formatted for direct entry into the VVUQ Aggregator |
| **8. Methodology** | Standards citations, Type A evaluation method, distribution fitting approach |
| **9. Assumptions & Limitations** | i.i.d. assumption, MLE convergence requirements, Beta bounded-data constraint |
| **10. Reviewer Checklist** | Checkbox list for independent review (sample sizes adequate? data source documented? distribution justified? outliers investigated? carry-over values entered? units consistent?) |

### Embedded Charts

The currently displayed chart from the Charts tab is embedded directly in the HTML as a base64-encoded PNG image at 300 DPI. This means the report is fully self-contained — no external image files needed. The chart is automatically rendered with a white background for print readability, regardless of the dark theme displayed on-screen.

### Tips for Good Reports

- Fill in the Project Info panel before exporting (program name, analyst, date, notes)
- Select the most informative chart (usually the Histogram for the primary variable) before exporting
- Run the analysis on all variables before exporting — the report captures all computed results at the time of export

---

## 12. Figure Export

The Charts tab includes an **"Export Figure Package..."** button that generates publication-quality figures in multiple formats simultaneously.

### What Gets Exported

When you click "Export Figure Package..." and choose a base filename (e.g., `temperature_histogram`), the tool creates:

| File | Format | Purpose |
|------|--------|---------|
| `temperature_histogram_300dpi.png` | PNG at 300 DPI | Standard publication quality |
| `temperature_histogram_600dpi.png` | PNG at 600 DPI | High-resolution printing |
| `temperature_histogram.svg` | Scalable Vector Graphics | Infinitely scalable, editable in Illustrator/Inkscape |
| `temperature_histogram.pdf` | PDF vector format | Direct inclusion in LaTeX documents |
| `temperature_histogram_meta.json` | JSON sidecar | Metadata for traceability and audit trails |

### Automatic Light Theme for Export

All exported figures use a **white background with dark text** regardless of the on-screen dark theme. This ensures figures are publication-ready without manual color adjustment. After export, the on-screen chart reverts to the dark theme.

### JSON Sidecar Metadata

The `_meta.json` file contains traceability information:

```json
{
  "tool_name": "Statistical Analyzer",
  "tool_version": "1.3.0",
  "figure_id": "temperature_histogram",
  "generated_utc": "2026-02-22T14:30:00+00:00",
  "formats": ["png@300dpi", "png@600dpi", "svg", "pdf"],
  "chart_type": "Histogram",
  "variable": "TC-01 Reading (K)"
}
```

This metadata makes it possible to trace any figure in a report back to the exact tool version and analysis that produced it — essential for certification and regulatory audit trails.

---

## 13. When Do I Need This Tool?

Use the Statistical Analyzer when you have **measured data or repeated outputs** that need to be characterized for an uncertainty budget. Here are the most common scenarios:

### Scenario 1: Thermocouple Repeatability

You measured the same temperature 30 times with a thermocouple during a steady-state test period. Load the readings, compute statistics, and carry the sigma and DOF into the Aggregator as experimental uncertainty (u_D).

**What you will see:** sigma around 0.2 K, Normal distribution (typical for well-behaved thermocouples), DOF = 29. The recommendation will say "Adequate sample size."

### Scenario 2: CFD Input Parameter Sensitivity

You ran your CFD case 20 times with slightly different inlet total pressure values sampled from the manufacturer's calibration tolerance. Load the output temperatures, compute statistics, and carry the sigma and DOF into the Aggregator as input uncertainty (u_input).

**What you will see:** The output scatter may be Normal or slightly skewed depending on the nonlinearity of the CFD model. The sigma represents how much the output changes due to the input perturbation.

### Scenario 3: Wind Tunnel Test Data

You have 50 pressure tap readings from a wind tunnel calibration. The readings show right-skewed scatter due to turbulence-driven pressure spikes. Load the data — the tool will identify the skewness, recommend Log-Normal or Weibull, and give you the correct sigma for the Aggregator.

**What you will see:** Positive skewness (> 1.0), Shapiro-Wilk rejection (p < 0.05), Log-Normal or Weibull ranked #1. The recommendation will flag the skewness and non-normality.

### Scenario 4: Material Property Scatter

You have 15 tensile strength measurements from coupon tests. The data may follow a Weibull distribution (common for material failure data). The tool fits Weibull, reports the shape and scale parameters, and flags if the sample size is marginal (N < 30).

**What you will see:** sigma around 5-10% of the mean (typical for material properties), possibly Weibull or Log-Normal best fit. The recommendation will say "Marginal sample size (N=15)."

### Scenario 5: Digitization Uncertainty

You measured a value with a digital instrument that has 0.1 K resolution. You have 100 readings. The tool may show a distribution with structure related to the digital resolution. Since you have actual repeated readings, the tool computes the actual sigma from your data, which may differ from the theoretical Uniform value of resolution / sqrt(12).

### When NOT to Use This Tool

- **Manufacturer specifications without data:** Enter directly into the Aggregator as Type B (Uniform or Normal)
- **Engineering estimates:** Enter directly into the Aggregator as Type B
- **Grid convergence studies:** Use the GCI Calculator instead
- **Single data point:** You need at least 2 values (and realistically 10+) for any meaningful statistical analysis
- **Time-series data with autocorrelation:** See the FAQ for guidance on handling correlated data

---

## 14. Project Info Panel

The Project Info panel sits at the top of the main window, collapsed by default. Click the arrow button to expand it.

### Fields

| Field | Purpose | Example |
|-------|---------|---------|
| **Program / Project** | Project name for traceability | "XYZ Flight Test Campaign" |
| **Analyst** | Who performed the analysis | "J. Smith" |
| **Date** | Analysis date (auto-filled with today's date) | "2026-02-22" |
| **Notes** | Free-form text area for data source, test conditions, instrumentation, assumptions | "Thermocouples calibrated per ASTM E230. Data from Run 14, steady-state period 10:30-10:45. Ambient temp 22 C." |

These fields are:
- Saved with the `.sta` project file
- Included in HTML reports (Section 1: Project Information)
- Printed in the report header for traceability

### Why Bother?

V&V 20 requires traceability. Six months from now, when a reviewer asks "where did this u_D = 0.21 K value come from?", you need to trace it back to a specific dataset, analyst, and analysis date. The Project Info panel makes this easy. Fill it in before you save or export — it takes 30 seconds and saves hours of forensic work later.

---

## 15. Reading the Guidance Panels

The tool includes color-coded guidance panels that provide real-time feedback. These appear on the Data Input tab, the Statistics tab (as the Recommendation panel), and the Carry-Over Summary tab.

### Color Coding

| Color | Icon | Meaning | Action |
|-------|------|---------|--------|
| **Green** | Checkmark | OK — no issues detected | Proceed normally |
| **Yellow** | Warning triangle | Caution — something to be aware of | Review the message, may need documentation or action |
| **Red** | X mark | Warning — a significant issue detected | Investigate before proceeding |

### Recommendation Engine Logic

After computing statistics, the Recommendation panel provides a multi-part assessment separated by bullet points. Each part addresses a specific aspect of data quality:

**Sample size assessment:**

| Condition | Severity | Message |
|-----------|----------|---------|
| N < 10 | Red | "Insufficient data (N=7) per GUM Section G.3 — consider collecting more samples." |
| 10 <= N < 30 | Yellow | "Marginal sample size (N=18) — results have higher uncertainty." |
| N >= 30 | Green | "Adequate sample size (N=50)." |

**Outlier detection:**

| Condition | Severity | Message |
|-----------|----------|---------|
| Outliers found | Yellow | "2 outlier(s) detected (1.5xIQR rule) — investigate before excluding." |

**Tail behavior:**

| Condition | Severity | Message |
|-----------|----------|---------|
| Excess kurtosis > 3 | Yellow | "Heavy tails detected — consider Student-t distribution." |

**Skewness:**

| Condition | Severity | Message |
|-----------|----------|---------|
| abs(skewness) > 1 | Yellow | "Significant skewness — consider Log-Normal or Weibull." |

**Normality:**

| Condition | Severity | Message |
|-----------|----------|---------|
| Shapiro-Wilk p >= 0.05 | Green | "Data appears normally distributed (Shapiro-Wilk p >= 0.05)." |
| Shapiro-Wilk p < 0.05 | Yellow/Red | "Data is NOT normally distributed (Shapiro-Wilk p < 0.05)." |

**Best fit:**

Always reported: "Best-fit distribution: Normal (AICc=125.3, GOF p=0.842)."

The overall guidance panel severity is determined by the worst condition:
- **Green:** N >= 30 AND data is Normal
- **Yellow:** N >= 10 but either N < 30 or data is non-Normal
- **Red:** N < 10

---

## 16. Sample Size Guidance

How many data points do you need? It depends on what you are trying to do.

### Sample Size Assessment Table

| Sample Size | Assessment | What You Can Reliably Do |
|-------------|-----------|--------------------------|
| **N < 5** | Insufficient | Distribution fitting is disabled (MLE needs at least 5 points). Only basic statistics (mean, std, min, max) are computed. Shapiro-Wilk requires N >= 3. |
| **N = 5-9** | Insufficient (per GUM) | Basic statistics work. Distribution fitting runs but results are unreliable — the standard deviation estimate itself is very uncertain. DOF is low (4-8), so coverage factors are large (k = 2.3 to 2.8 for 95%). |
| **N = 10-29** | Marginal | Usable for standard analysis. The standard deviation estimate is moderately reliable. Shapiro-Wilk normality test has limited power. Prefer Student-t over Normal for coverage intervals. CLT begins to apply for mean estimation. |
| **N = 30-99** | Adequate | Standard analysis fully applicable. Central Limit Theorem applies. Distribution fitting is reasonably reliable. Normality tests can reliably distinguish normal from non-normal data. |
| **N >= 100** | Good | High confidence in distribution shape. Bootstrap GOF and AD results are usually stable enough to separate similar distributions (e.g., Normal vs. Student-t). |

If serial correlation is present, replace N with N_eff for uncertainty-of-mean and DOF calculations. In that case, the effective statistical strength can be much lower than the raw sample count shown here.

### Impact on Coverage Factor k

The coverage factor k for a 95% confidence interval depends on DOF. Here is the practical impact:

| DOF (= N_eff - 1) | k for 95% two-sided | Penalty vs. large-sample k |
|----------------|---------------------|---------------------------|
| 4 | 2.776 | +42% |
| 9 | 2.262 | +15% |
| 19 | 2.093 | +7% |
| 29 | 2.045 | +4% |
| 49 | 2.009 | +2% |
| infinity | 1.960 | baseline |

**What this means in practice:** With only 5 data points (DOF = 4), your 95% confidence interval is 42% wider than it would be with infinite data. That is the statistical penalty for small samples. This is not pessimism — it is an honest reflection of how much you do not know about the true standard deviation.

### Recommendation for CFD Validation Work

- **Minimum:** N = 10 (per GUM Section G.3 recommendation)
- **Target:** N = 30 (Central Limit Theorem threshold; DOF = 29, so coverage factor is about 2.05, close to the large-sample value of 1.96)
- **Best practice for certification:** N = 50+ (reliable distribution fitting, powerful goodness-of-fit tests)

### What Happens with Very Small Samples?

With N = 3, for example:
- The standard deviation is very noisy (it could easily be off by a factor of 2)
- The DOF = 2, so the 95% coverage factor is t(0.975, 2) = 4.30 — more than double the large-sample value
- The Shapiro-Wilk test has almost no power (it will pass almost anything)
- Distribution fitting is disabled

If you truly cannot collect more data, acknowledge the limitation in your report and use a Student-t distribution with the low DOF. The Aggregator's Welch-Satterthwaite formula will propagate the low DOF correctly, producing appropriately wider combined uncertainty intervals.

### Key Formulas for Sample Size Guidance

- Standard uncertainty of the mean: u = sigma / sqrt(N_eff)
- Degrees of freedom: DOF = N_eff - 1
- 95% CI for the mean: x_bar +/- t(0.975, DOF) * u
- Coverage factor (95%): k = t(0.975, DOF), which approaches 2.0 for large DOF (equivalently, large N_eff)

---

## 17. Key Formulas Reference

### Mean

```
x_bar = (1/N) * SUM(x_i)
```

### Sample Standard Deviation (Bessel-corrected)

```
sigma = sqrt( (1/(N-1)) * SUM((x_i - x_bar)^2) )
```

The N-1 denominator (Bessel correction) gives an unbiased estimate of the population standard deviation. This is what the tool reports as "Std Deviation."

### Standard Uncertainty of the Mean

```
u_mean = sigma / sqrt(N_eff)
```

This is how precisely you know the population mean based on effective sample size N_eff. Decreases as you collect more effectively independent data.

### Degrees of Freedom

```
DOF = N_eff - 1
```

For the tool's autocorrelation-adjusted workflow. When autocorrelation is negligible, N_eff = N and this reduces to DOF = N - 1.

### 95% Confidence Interval for the Mean

```
CI_95 = x_bar +/- t(0.975, DOF) * u_mean
```

Where t(0.975, DOF) is the Student-t critical value at the 97.5th percentile with DOF degrees of freedom.

### Coverage Factor (95%, two-sided)

```
k_95 = t(0.975, DOF)
```

Approaches 1.96 as DOF approaches infinity. For DOF = 29, k_95 = 2.045. For DOF = 9, k_95 = 2.262.

### Skewness (Fisher, unbiased)

```
skew = [N / ((N-1)(N-2))] * SUM( ((x_i - x_bar) / sigma)^3 )
```

Zero for perfectly symmetric data. Positive means right tail is longer. Negative means left tail is longer. The tool flags abs(skew) > 1 as "significant skewness."

### Excess Kurtosis (Fisher, unbiased)

```
kurt = [(N(N+1)) / ((N-1)(N-2)(N-3))] * SUM( ((x_i - x_bar) / sigma)^4 ) - [3(N-1)^2 / ((N-2)(N-3))]
```

Zero for Normal distribution. Positive means heavier tails than Normal (more extreme values). The tool flags kurtosis > 3 as "heavy tails detected."

### AICc (Corrected Akaike Information Criterion)

```
AIC = 2k - 2 ln(L)
AICc = AIC + [2k(k+1)] / (N-k-1)
```

Where k = number of distribution parameters, L = maximum likelihood value, and N = sample size. Lower is better.

### BIC (Bayesian Information Criterion)

```
BIC = k * ln(N) - 2 ln(L)
```

Where k = number of parameters, N = sample size. Penalizes complexity more than AICc for larger samples.

### KS Test Statistic (Diagnostic Only)

```
D = max |F_empirical(x) - F_theoretical(x)|
```

The maximum absolute difference between the empirical and theoretical CDFs. Smaller D means better fit. In this tool, bootstrap GOF is primary when available; KS is promoted to active screening in sparse/guardrail mode.

### IQR Outlier Detection

```
IQR = Q3 - Q1
Lower fence = Q1 - 1.5 * IQR
Upper fence = Q3 + 1.5 * IQR
```

Any data point outside [Lower fence, Upper fence] is flagged as a potential outlier.

---

## 18. Standards References

### JCGM 100:2008 (GUM) — Guide to the Expression of Uncertainty in Measurement

The foundational international uncertainty standard. Key sections relevant to this tool:

| Section | Topic |
|---------|-------|
| 4.2 | Type A evaluation of standard uncertainty — statistical analysis of repeated observations |
| 4.3 | Type B evaluation — other means (manufacturer specs, handbooks) |
| G.3 | Guidance on sample size — recommends N >= 10 |
| G.4 | Degrees of freedom and the Welch-Satterthwaite formula |

### JCGM 101:2008 (GUM Supplement 1) — Monte Carlo Methods

Covers propagation of distributions through measurement models using Monte Carlo simulation. Relevant when you need to propagate non-Normal distributions.

### ASME PTC 19.1-2018 — Test Uncertainty

The ASME standard for test uncertainty analysis. Covers systematic and random components, pre-test and post-test analyses, sample size requirements.

### ASME V&V 20-2009 (R2021) — Standard for Verification and Validation in CFD and Heat Transfer

The primary CFD validation uncertainty standard. Key sections:

| Section | Topic |
|---------|-------|
| 5 | Numerical uncertainty (u_num) — addressed by the GCI Calculator |
| 6 | Input uncertainty (u_input) — may use this tool for perturbed-parameter studies |
| 7 | Experimental uncertainty (u_D) — primary use case for this tool |
| 8 | Validation comparison — combining all uncertainties via RSS |

### Additional References

| Reference | Relevance |
|-----------|-----------|
| Akaike (1974), "A new look at the statistical model identification" | Base AIC model-selection framework |
| Hurvich & Tsai (1989), "Regression and time series model selection in small samples" | AICc small-sample correction used for ranking |
| Schwarz (1978), "Estimating the dimension of a model" | BIC for model comparison |
| Shapiro & Wilk (1965), "An analysis of variance test for normality" | Normality test used by the tool |

---

## 19. Glossary

| Term | Plain English Definition |
|------|--------------------------|
| **sigma** | Standard deviation (sample, with Bessel correction using N-1). The basic measure of data scatter. |
| **DOF** | Degrees of freedom = N - 1. Quantifies how much information is in the standard deviation estimate. |
| **AICc** | Corrected Akaike Information Criterion — balances fit vs complexity with small-sample correction. Lower AICc = better. |
| **BIC** | Bayesian Information Criterion — like AICc but penalizes complexity more heavily for large N. |
| **Bootstrap GOF** | Parametric bootstrap goodness-of-fit p-value (primary when bootstrap mode is active). |
| **KS test** | Kolmogorov-Smirnov diagnostic goodness-of-fit metric (secondary context). |
| **AD test** | Anderson-Darling goodness-of-fit test, tail-sensitive. Available for Normal and Log-Normal only. |
| **Shapiro-Wilk** | A normality test. p > 0.05 means the data is consistent with Normal. p < 0.05 means it is not. Valid for N between 3 and 5000. |
| **MLE** | Maximum Likelihood Estimation — the method used to find the best-fit parameters for each distribution. |
| **Skewness** | Measure of asymmetry. Zero = symmetric, positive = right-skewed, negative = left-skewed. |
| **Kurtosis** | Measure of tail weight relative to Normal. Zero = Normal-like, positive = heavier tails, negative = lighter tails. |
| **u_mean** | Standard uncertainty of the mean = sigma / sqrt(N). How well the average is known. |
| **u_pop** | Population standard uncertainty = sigma. The spread of individual values. |
| **Coverage factor k** | Multiplier for expanded uncertainty. U = k * u. For 95% coverage with large N, k is approximately 2.0. |
| **Expanded uncertainty U** | U = k * u. A wider interval providing higher coverage (typically 95%). |
| **IQR** | Interquartile range = Q3 - Q1 (75th percentile minus 25th percentile). A robust measure of spread. |
| **Outlier** | A data point beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR. Should be investigated, not automatically removed. |
| **CDF** | Cumulative Distribution Function — the probability of observing a value less than or equal to x. |
| **PDF** | Probability Density Function — the shape of the distribution curve. Area under the curve between two values gives probability. |
| **QQ Plot** | Quantile-Quantile plot — compares data quantiles to theoretical Normal quantiles. Points on the diagonal = Normal data. |
| **Type A evaluation** | Statistical analysis of repeated observations (GUM Section 4.2). What this tool does. |
| **Type B evaluation** | Evaluation by other means — engineering judgment, specs, handbooks (GUM Section 4.3). Does not need this tool. |
| **i.i.d.** | Independent and identically distributed — the assumption that each data point is drawn independently from the same population. |
| **Bessel correction** | Dividing by N-1 instead of N when computing sample standard deviation, correcting for bias in finite samples. |
| **RSS** | Root Sum of Squares — the method V&V 20 uses to combine independent uncertainty sources: u_total = sqrt(u1^2 + u2^2 + ...). |
| **Carry-Over** | The values (sigma, DOF, distribution) that you transfer from this tool to the VVUQ Uncertainty Aggregator. |
| **Project File (.sta)** | A JSON file containing all input data and settings for a Statistical Analyzer project, for saving and reloading. |
| **Catppuccin Mocha** | The dark color theme used by the application interface — dark background, high-contrast text, blue accents. |
| **u_D** | Experimental (data) uncertainty — from sensors, DAQ, test conditions. One of the three V&V 20 uncertainty categories. |
| **u_input** | Input/boundary condition uncertainty — from BCs, material properties, geometry. One of the three V&V 20 categories. |
| **u_num** | Numerical uncertainty — from grid convergence, time step, solver convergence. Produced by the GCI Calculator, not this tool. |
| **V&V 20** | ASME V&V 20-2009 — the standard for verification and validation of CFD simulations. Defines the uncertainty framework this toolset implements. |
| **Welch-Satterthwaite** | A formula for computing effective degrees of freedom when combining multiple uncertainty sources with different DOF values. Used by the Aggregator. |
| **CLT** | Central Limit Theorem — the principle that the distribution of means of independent samples approaches Normal as N increases, regardless of the underlying distribution shape. |

---

## 20. Frequently Asked Questions

### Q: I loaded my CSV but the column names are wrong. How do I fix them?

**A:** Select the column using the **Active Column** dropdown in the Column Management group on the Data Input tab, then type the correct name in the **Name** field. The table header, chart labels, and all results will update automatically. If the CSV had no header row (all numeric first row), the tool assigns default names ("Variable 1," "Variable 2," etc.) and you can rename them manually.

### Q: My data has some blank cells or non-numeric values. What happens?

**A:** The tool ignores blanks and non-numeric entries. When loading CSV data, any value that cannot be parsed as a floating-point number is treated as NaN (not a number) and excluded from all calculations. The **Sample Size (N)** reported in the results reflects only the valid numeric values. Check the data summary line below the table ("X column(s), Y valid value(s) total") to verify.

### Q: The Shapiro-Wilk test says my data is not Normal. Should I be worried?

**A:** Not necessarily, but pay attention. There are three common situations:

1. **Large N (100+):** The Shapiro-Wilk test becomes very sensitive with large samples. Even tiny, practically irrelevant deviations from normality will trigger a rejection. Check the QQ plot — if the points are close to the line with minor deviations at the tails, the Normal assumption is probably fine for engineering purposes.

2. **Truly non-Normal data:** If the histogram shows obvious skewness, bimodality, or heavy tails, the rejection is meaningful. Use the recommended distribution.

3. **Small N (< 20):** The test has low power and may fail to detect non-normality. A "pass" does not guarantee normality — it just means the test could not detect a problem.

**Bottom line:** Look at the histogram and QQ plot. If the data looks roughly bell-shaped to your eye, Normal is probably fine for RSS aggregation, even if Shapiro-Wilk rejects.

### Q: Which sigma should I enter into the VVUQ Aggregator — sigma/sqrt(N_eff) or sigma?

**A:** It depends on what uncertainty source you are characterizing:

- **u_D (experimental uncertainty) for comparison to a single CFD value:** Usually sigma / sqrt(N_eff) — because you are comparing the CFD output to the best estimate of the measured mean, and the uncertainty of that mean should reflect autocorrelation-adjusted sample size.
- **u_input (input parameter uncertainty) for Monte Carlo propagation:** Usually sigma (population) — because each Monte Carlo sample represents one possible realization from the full distribution, not an average.
- **Conservatively, if unsure:** Use sigma (population). It is larger and more conservative.

The Carry-Over Summary tab reports the population sigma by default.

### Q: The tool ranked Log-Normal as #1 but Normal as #2 and their AICc values are very close. Which should I use?

**A:** When AICc values differ by less than about 2, the models are statistically indistinguishable. In that case, use engineering judgment:

- Is the data physically bounded below by zero (e.g., flow rate, temperature in Kelvin)? Lean toward Log-Normal.
- Is there no physical reason for skewness? Lean toward Normal — it is simpler and more widely understood.
- For RSS aggregation in V&V 20, the difference between Normal and slightly-log-Normal will usually be negligible.

### Q: AICc says Log-Normal is rank 1, but GOF does not pass. Which do I trust?

**A:** Trust the GOF gate for pass/fail. The tool now prefers the lowest-AICc model that passes GOF (bootstrap when available, KS screening in sparse/guardrail mode). If no candidate passes, it falls back to the lowest-AICc model and flags that fallback so you can use a conservative Monte Carlo/empirical carry-over path in the Aggregator.

### Q: Distribution fitting failed for some distributions. Why?

**A:** Several distributions have data requirements:

- **Log-Normal, Weibull, Gamma:** Require all values > 0. If you have any zero or negative values, these are skipped.
- **Beta:** Only attempted for true bounded proportion data in [0,1], and requires enough samples (N >= 20).
- **Any distribution:** If MLE optimization fails to converge (very rare), that distribution is silently skipped.

Skipped distributions simply do not appear in the ranking table.

### Q: I have fewer than 5 data points. Why is the distribution fitting table empty?

**A:** Distribution fitting requires at least N = 5 data points for MLE to produce meaningful results. With fewer than 5 points, there are not enough observations to reliably estimate distribution parameters. Basic statistics (mean, std, min, max) are still computed for N >= 2.

Recommendation: Collect more data if possible. If not, use a Student-t distribution with DOF = N_eff - 1 for your uncertainty assessment (enter this manually in the Aggregator). If autocorrelation is negligible, this reduces to N - 1.

### Q: Can I analyze multiple variables at once?

**A:** Yes. The tool supports multiple data columns. Each column is analyzed independently when you click "Compute Statistics." Use the **Display** dropdown on the Statistics tab or the **Variable** dropdown on the Charts tab to switch between variables. The Carry-Over Summary shows all variables in a single table.

To add more columns, use **"+ Add Column"** on the Data Input tab, or load a multi-column CSV file.

### Q: How do I handle outliers?

**A:** The tool detects outliers using the 1.5 x IQR rule and reports them in the Recommendation panel. However, it does **not** automatically exclude them. This is intentional — outlier exclusion is an engineering judgment that requires investigation.

Before excluding any outlier:
1. Check if it is a data entry error (typo, instrument glitch)
2. Check if it represents a real physical event (turbulence spike, transient)
3. Check if the data collection process was compromised during that measurement
4. Document your justification for inclusion or exclusion

The GUM (Section 4.4.1) emphasizes that data should not be discarded without a clear reason. If outliers are genuine (not errors), consider using a Student-t distribution — its heavier tails naturally accommodate occasional large deviations.

To manually remove an outlier, delete the value from the table on the Data Input tab and re-run the analysis.

### Q: The HTML report only shows one chart. Can I include all four chart types?

**A:** The report embeds whichever chart is currently displayed on the Charts tab at the time of export. To include a specific chart, select it before exporting the report. If you need all four chart types in your report, export figure packages separately (see [Section 12](#12-figure-export)) and insert them manually into your document.

### Q: What is the .sta file format? Can I edit it manually?

**A:** The `.sta` file is plain JSON. You can open it in any text editor. The structure is:

```json
{
  "tool": "Statistical Analyzer",
  "version": "1.2.0",
  "saved_utc": "2026-02-22T14:30:00",
  "project_metadata": {
    "program": "XYZ Campaign",
    "analyst": "J. Smith",
    "date": "2026-02-22",
    "notes": "Calibration data from Run 14"
  },
  "data": {
    "columns": [
      {
        "name": "TC-01 Reading",
        "unit": "K",
        "unit_category": "Temperature",
        "values": [305.2, 305.5, 304.9, 305.3, 305.1]
      }
    ]
  }
}
```

You can edit values, add columns, or modify metadata directly in the JSON. After editing, open the file in the tool (File > Open Project) and re-run the analysis. NaN values are stored as `null` in the JSON.

### Q: Does this tool perform unit conversions?

**A:** No. Unit labels are cosmetic — they appear in headers, results, charts, and reports, but no mathematical conversion is performed. If your data is in degrees Fahrenheit and you label it degrees Celsius, the numbers will not be converted. Make sure your data values match the selected unit before analysis.

### Q: How does this tool relate to the GCI Calculator and the VVUQ Uncertainty Aggregator?

**A:** The three tools form a complete V&V 20 workflow:

| Tool | Purpose | Output |
|------|---------|--------|
| **GCI Calculator** | Grid convergence analysis | u_num (numerical uncertainty) |
| **Statistical Analyzer** (this tool) | Data distribution analysis | sigma, DOF, distribution (experimental or input uncertainty) |
| **VVUQ Uncertainty Aggregator** | Combine all uncertainty sources via RSS and Monte Carlo | Total validation uncertainty u_val |

The GCI Calculator produces u_num. The Statistical Analyzer produces u_D (experimental) and/or u_input. Both feed into the Aggregator, which combines them to compute the total validation uncertainty and the E/u_val validation metric.

### Q: My kurtosis value is very high (> 6). What does that mean?

**A:** High excess kurtosis means your data has much heavier tails than a Normal distribution — there are more extreme values than Normal would predict. Common causes:

- **Mixed populations:** Your data comes from two or more overlapping processes (e.g., steady-state readings mixed with transient spikes)
- **Outliers:** A few extreme values are inflating the kurtosis
- **Physical process with heavy tails:** Some phenomena (turbulence intensity, combustion instabilities) naturally produce heavy-tailed distributions

The tool recommends Student-t distribution when excess kurtosis exceeds 3. Student-t has a shape parameter (nu, degrees of freedom) that controls tail heaviness, making it a flexible choice for heavy-tailed data.

### Q: Can I use this tool for time-series data?

**A:** The tool assumes data are **independent and identically distributed (i.i.d.)**. Time-series data often violates this assumption because consecutive measurements may be correlated (autocorrelation). If you have time-series data:

1. **Check for autocorrelation first.** If consecutive readings are correlated, the effective sample size is smaller than N, and the standard deviation of the mean is underestimated.
2. **Subsample to remove correlation.** If your thermocouple samples at 10 Hz but readings decorrelate after 0.5 seconds, use every 5th reading.
3. **Use steady-state segments only.** Do not include startup transients or drift periods.

If you have confirmed that your time-series data is effectively i.i.d. (no significant autocorrelation), this tool works perfectly.

### Q: Can I use this tool for non-CFD applications?

**A:** Absolutely. The statistics, distribution fitting, and uncertainty characterization are completely general. Any dataset of repeated measurements or observations can be analyzed — mechanical testing, chemical analysis, environmental monitoring, manufacturing quality control. The tool happens to be designed for V&V 20 workflows, but the mathematics does not care about the application domain.

### Q: I loaded an old .sta project file and the results look different. What happened?

**A:** The project file only stores raw data and metadata — not computed results. When you reopen a project, results are recomputed from scratch using the current version of the tool's algorithms. If the fitting algorithms or statistical methods were updated between versions, results may differ slightly. This is by design — it ensures you always get results from the latest, most-correct algorithms.

### Q: The carry-over table shows Auto and Final columns. Which one do I use?

**A:** Use the **Final Dist (Use This)** column for each variable.  
For **Final Method (Use This)**, treat it as a recommended Aggregator analysis mode. Because the Aggregator mode is global (set once), use this rule: if any row says Monte Carlo, run the Aggregator in Monte Carlo mode; if all rows say RSS, use RSS. Auto columns are tool recommendations. If you apply an override, add a rationale and the Final columns reflect that override.

### Q: Can I compare multiple datasets side by side?

**A:** Yes — load each dataset as a separate column. For example, if you want to compare thermocouple readings from two different instruments, put instrument A's readings in column 1 and instrument B's in column 2. Each gets its own independent statistical analysis, and both appear in the Carry-Over Summary. Use the Charts tab variable selector to switch between them.

### Q: Clipboard paste keeps going to the first column. How do I paste into another column?

**A:** Click the exact start cell first (for example row 1, column 2), then click **Paste from Clipboard**. The tool pastes relative to that selected cell now. If the table is empty and your clipboard includes headers, paste still creates a new full dataset (expected behavior).

### Q: What if my data has negative values but I think Log-Normal should fit?

**A:** The Log-Normal distribution is only defined for strictly positive values. If your data has negative values, the tool automatically skips Log-Normal. Consider whether a shift is appropriate — for example, if your data represents "delta T from baseline" and all values could be made positive by adding a constant offset, you might redefine the variable. Alternatively, the data may genuinely not be Log-Normal, and Normal or Student-t may be appropriate.

---

*Statistical Analyzer v1.4.0 — Built for engineers who need to characterize their data before building an uncertainty budget.*

*Standards: JCGM 100:2008 (GUM), JCGM 101:2008, ASME V&V 20-2009 (R2021), ASME PTC 19.1-2018*
