# CFD Validation Uncertainty Tool — Specification for Claude Code

## Purpose

Build a single-file PySide6 GUI application for computing CFD model validation uncertainty per the ASME V&V 20 framework. The tool must support both RSS (analytical) and Monte Carlo uncertainty combination methods, accommodate flexible input types, and produce a final comparison roll-up. The target users are aerospace thermal/CFD engineers performing model validation against flight test data.

---

## 1. Standards & References

The tool should include a Help/About section and tooltips throughout that reference these sources. Every calculation option should indicate which standard it traces to.

| Standard | Full Title | What It Covers in This Tool |
|---|---|---|
| **ASME V&V 20-2009 (R2016)** | Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer | Overall framework: E = S − D, u_val = √(u_num² + u_input² + u_D²), validation comparison, k=2 default for 95% coverage |
| **JCGM 100:2008 (GUM)** | Evaluation of Measurement Data — Guide to the Expression of Uncertainty in Measurement | RSS of standard uncertainties, Annex G: Welch-Satterthwaite formula for effective degrees of freedom (Section G.4.1), coverage factor from t-distribution (Table G.2) |
| **JCGM 101:2008 (GUM Supplement 1)** | Propagation of Distributions Using a Monte Carlo Method | Monte Carlo as a valid alternative to GUM analytical propagation. Referenced by V&V 20 as an accepted method. |
| **ASME PTC 19.1-2018** | Test Uncertainty | Companion standard to V&V 20 for experimental uncertainty analysis. Type A / Type B classification of uncertainties. |
| **AIAA G-077-1998** | Guide for Verification and Validation of Computational Fluid Dynamics Simulations | Supplementary guidance on CFD V&V methodology |

### Key Formulas to Implement (with traceability)

**Combined standard uncertainty (V&V 20 Section 6):**
```
u_val = √(u_num² + u_input² + u_D²)
```

**Expanded uncertainty (V&V 20 / GUM Section 6.2):**
```
U_val = k × u_val
```
Where k is the coverage factor (default k=2 per V&V 20; or computed from t-distribution per GUM Annex G).

**Welch-Satterthwaite effective degrees of freedom (GUM Annex G, Section G.4.1, Equation G.2b):**
```
ν_eff = (u_c(y))⁴ / Σ(uᵢ(y)⁴ / νᵢ)
```
Where:
- u_c(y) = combined standard uncertainty = √(Σuᵢ²)
- uᵢ = standard uncertainty of source i  
- νᵢ = degrees of freedom for source i (= nᵢ − 1 for Type A; = ∞ for Type B / supplier specs)
- Sources with νᵢ = ∞ drop out of the denominator sum

Then k = t_p(ν_eff) from the Student's t-distribution for the desired coverage probability p.

**One-sided tolerance factor for normal distribution (for k-factor lookup tables):**
```
k = nct.ppf(γ, df=n-1, nc=z_p × √n) / √n
```
Where γ = confidence level, z_p = normal quantile for coverage p, n = sample size. Use scipy.stats.nct.

**Monte Carlo combination (GUM Supplement 1 / JCGM 101:2008):**
For each trial i = 1 to N_trials:
```
E_combined,i = Σ (random sample from each source's distribution)
```
Sort all E_combined values. The p-th percentile is the bound. No k-factor needed.

---

## 2. Application Architecture

### 2.1 Technology

- **Single Python file** using PySide6
- Use `scipy` for statistical functions (nct, norm, chi2, binom, t distributions)  
- Use `numpy` for numerical computation and Monte Carlo sampling
- Use `matplotlib` embedded in PySide6 for all plotting (use `FigureCanvasQTAgg`)
- Target Python 3.10+

### 2.2 Overall Layout

Use a **QTabWidget** as the main container with the following tabs:

1. **Comparison Data** — Load/enter CFD-to-flight-test comparison errors (E = S − D)
2. **Uncertainty Sources** — Define all uncertainty sources with categories and input types
3. **Analysis Settings** — Configure coverage, confidence, k-factor method, Monte Carlo parameters
4. **Results — RSS** — RSS-based uncertainty combination and bounds
5. **Results — Monte Carlo** — Monte Carlo uncertainty propagation and bounds
6. **Comparison Roll-Up** — Side-by-side comparison of both methods with final summary
7. **Reference** — Built-in documentation of formulas, k-factor tables, standards references

---

## 3. Tab Specifications

### 3.1 Tab 1: Comparison Data

This tab handles the primary CFD-to-flight-test comparison errors.

**Input Section:**
- **Import button**: Load CSV/Excel file with delta temperatures. Expected format:
  - Rows = flight test conditions
  - Columns = thermocouple/sensor locations
  - Cell values = E = T_CFD − T_FlightTest [°F or °C, user selects units]
- **Manual entry table**: Editable QTableWidget as fallback
- **Metadata fields:**
  - Number of flight test conditions (auto-detected from data, editable)
  - Number of sensor locations (auto-detected from data, editable)
  - Total sample count n (auto-calculated: conditions × locations, shown prominently)
  - A checkbox: "Treat as pooled data" with a note: *"Pooling assumes all locations are drawn from the same model-error population. Verify that no single location systematically dominates the tails. [Engineering judgment required]"*

**Computed Statistics (auto-update on data change):**
- Mean (Ē)
- Standard deviation (s_E)
- Standard error of the mean (SE = s_E / √n)
- Min, Max, Range
- Empirical 5th and 95th percentiles
- Skewness and Kurtosis (excess kurtosis, so normal = 0)
- Shapiro-Wilk normality test p-value (flag if p < 0.05: "Data may not be normally distributed")

**Per-Location Breakdown (collapsible section):**
- Show mean and std dev for each sensor location across all flight conditions
- Flag any location whose mean is more than 2σ from the overall mean as a potential outlier
- This helps the user decide whether pooling is appropriate

**Distribution Guidance Panel:**
Based on the computed statistics, show an automated assessment:
- If |skewness| < 0.5 AND |excess kurtosis| < 1.0 AND Shapiro-Wilk p > 0.05: **"Data is consistent with a normal distribution. Normal k-factors are appropriate."**
- If |skewness| < 0.5 AND excess kurtosis < −0.5: **"Data is symmetric but platykurtic (flatter than normal, e.g., uniform-like). Normal k-factors are conservative — they overestimate the tails. This is acceptable for safety-critical applications."**
- If |skewness| < 0.5 AND excess kurtosis > 1.0: **"Data is symmetric but leptokurtic (heavier tails than normal). Normal k-factors may be NON-CONSERVATIVE. Consider distribution-free methods or Monte Carlo."**
- If |skewness| > 1.0: **"Data is significantly skewed. Normal k-factors may not be appropriate. Consider fitting an asymmetric distribution or using Monte Carlo."**

**Sample Size Guidance Panel:**
- n < 20: **"SMALL SAMPLE. k-factor penalty is significant (k > 2.4 for 95/95). Consider pooling additional locations if justified. Distribution-free 95/95 bounds require n ≥ 59."** [Background: light red]
- 20 ≤ n < 60: **"MODERATE SAMPLE. k-factor penalty is moderate (k ≈ 2.0–2.4). Distribution-free bounds may not be available at 95/95."** [Background: light yellow]
- n ≥ 60: **"ADEQUATE SAMPLE. k-factor penalty is small (k < 2.0). Distribution-free 95/95 bounds are available using the minimum observation."** [Background: light green]

**Plotting (embedded matplotlib):**
- Histogram of all comparison errors with normal PDF overlay
- QQ plot (normal probability plot)
- Box plot showing all locations side-by-side (if multiple locations)

---

### 3.2 Tab 2: Uncertainty Sources

This is the core input tab. The user defines each uncertainty source with its properties.

**Source List:**
- A QTableWidget or QTreeWidget where each row is one uncertainty source
- Buttons: Add Source, Remove Source, Duplicate Source, Move Up/Down
- Each source has the following fields (in a detail panel below the list, or in an expandable row):

**Fields per source:**

| Field | Type | Description |
|---|---|---|
| Name | Text | User-defined name (e.g., "Iteration Jitter", "Sensor Error") |
| Category | Dropdown | **Numerical (u_num)**, **Input/BC (u_input)**, **Experimental (u_D)** — these map to V&V 20 categories. Tooltip: "Per ASME V&V 20: Numerical = discretization, iteration, spatial extraction. Input/BC = boundary condition uncertainties propagated through the model. Experimental = measurement/sensor uncertainties." |
| Input Type | Dropdown | See below |
| Distribution | Dropdown | Normal, Uniform, Triangular, Custom/Empirical — availability depends on Input Type |
| Sigma Basis | Dropdown | Only shown for sigma-type inputs: "Confirmed 1σ", "Assumed 1σ (unverified)", "2σ (95%)", "3σ (99.7%)", "Bounding (min/max)" — Tooltip: "For supplier specifications, verify whether the stated tolerance is 1σ, 2σ, or a bounding value. This significantly affects the analysis. [PTC 19.1 Type B evaluation]" |
| Degrees of Freedom | Auto/Manual | Auto-calculated from data or entered manually. For supplier specs, default = ∞ (Type B). Tooltip: "ν = n−1 for data computed from n samples (Type A). ν = ∞ for supplier specs or well-established reference values (Type B). [GUM Section 4.3]" |

**Input Types (dropdown determines what fields appear):**

1. **Tabular Data** — User loads or pastes a table of values (like CFD-to-FT, iteration jitter, or spatial extraction data)
   - Load CSV/paste button
   - Auto-compute: mean, σ, n, DOF, empirical percentiles
   - Show distribution diagnostics (same as Tab 1)
   - Option: "Is this data centered on zero?" (for jitter/spatial type data vs. bias-containing data)

2. **Sigma Value Only** — User enters σ directly
   - Fields: σ value, Sigma Basis dropdown (1σ, 2σ, 3σ, bounding)
   - If "Bounding" selected: convert to σ using σ = range/(2×√3) for uniform, or σ = range/6 for normal 3σ
   - Fields: sample size n (or "Supplier/Reference" checkbox which sets ν = ∞)

3. **Tolerance/Expanded Value** — User enters an already-expanded tolerance value
   - Fields: tolerance value, what k was used to expand it
   - Auto-compute: σ = tolerance / k
   - Or: user says "I don't know what k was used" → tool asks for n and computes k, then back-calculates σ

4. **RSS of Sub-Components** — User enters a pre-combined RSS value
   - Fields: RSS value, whether it's a σ or expanded value, how many sub-components, effective DOF if known
   - Tooltip: "If this is an RSS of a 3σ assessment, enter the value and select '3σ' as the basis."

5. **CFD Sensitivity Run** — User enters before/after delta from a single perturbation study
   - Fields: number of sensor locations, delta values at each location (or summary σ and n)
   - This handles the inlet temp effect, flow rate effect type sources
   - Auto-compute σ from the sensor-to-sensor variation

**Validation checks per source:**
- If σ is very large relative to other sources (>80% of variance), flag as dominant and note that its DOF will control the effective DOF of the combined result
- If n < 10 and input type is tabular, warn: "Small sample — distribution shape cannot be reliably determined"

---

### 3.3 Tab 3: Analysis Settings

**Coverage and Confidence:**
- Coverage probability p: dropdown [90%, 95%, 99%] — default 95%
  - Tooltip: "The fraction of the population that the bound is intended to contain. 95% coverage means the bound captures 95% of predictions. [V&V 20 / GUM]"
- Confidence level γ: dropdown [90%, 95%, 99%] — default 95%
  - Tooltip: "How confident you are that the bound achieves the stated coverage. Higher confidence requires larger k, especially for small samples. [GUM Annex G]"
- One-sided vs Two-sided: radio buttons — default One-sided
  - Tooltip for One-sided: "Use when you have a directional concern (e.g., only underprediction matters). The entire coverage budget is allocated to one tail. [Recommended for safety-critical thermal validation]"
  - Tooltip for Two-sided: "Use when both over- and under-prediction are equally concerning. Coverage is split between both tails (e.g., 95% coverage = 2.5% in each tail)."

**Coverage Factor (k) Method:**
Radio buttons with descriptions:

1. **ASME V&V 20 Default: k = 2**
   - Description: "Uses k = 2 for approximately 95% coverage assuming a normal distribution with large degrees of freedom. This is the default in ASME V&V 20-2009 and is appropriate when ν_eff > 30. [V&V 20 Section 6]"

2. **GUM Welch-Satterthwaite: Computed k from effective DOF**
   - Description: "Computes effective degrees of freedom from all sources using the Welch-Satterthwaite formula (GUM Annex G, Section G.4.1, Eq. G.2b), then looks up k from the Student's t-distribution for the desired coverage and confidence. Accounts for small-sample penalties. [JCGM 100:2008]"
   - Show the computed ν_eff and resulting k as a live readout

3. **One-Sided Tolerance Factor: k from non-central t**
   - Description: "Computes the one-sided tolerance factor that provides the stated coverage with the stated confidence for a normal distribution, accounting for uncertainty in both mean and standard deviation. Uses the non-central t-distribution. Appropriate when the concern is one-directional (e.g., underprediction only). [Krishnamoorthy & Mathew, Statistical Tolerance Regions]"
   - Note: this is the most conservative and most rigorous for one-sided bounds

4. **Manual k Entry**
   - Text field for user-specified k
   - Description: "Enter a custom coverage factor. Document the justification."

**Monte Carlo Settings:**
- Number of trials: [10000, 50000, 100000, 500000, 1000000] — default 100,000
  - Tooltip: "More trials = more stable results, especially for extreme percentiles. 100,000 is adequate for most applications. [JCGM 101:2008]"
- Random seed: optional integer for reproducibility
- Bootstrap confidence on MC percentile: checkbox (default on)
  - If checked, repeat the MC analysis 100 times with resampled inputs and report the spread of the percentile estimate
  - Tooltip: "Shows how stable the Monte Carlo percentile estimate is given the uncertainty in the input distributions. Particularly useful when dominant sources have small sample sizes."

**Bound Type Selection:**
Radio buttons:

1. **Known uncertainties only (u_val)**
   - Description: "Computes the expanded validation uncertainty from the defined uncertainty sources. Does NOT include model form error. Appropriate for assessing whether the observed comparison error is explained by known uncertainties. [V&V 20 primary intent]"

2. **Total observed scatter (s_E)**
   - Description: "Uses the total standard deviation of the comparison errors (which includes model form error). Provides a prediction bound for future comparisons. More conservative but captures the full observed variability."

3. **Both (for comparison)**
   - Description: "Computes both and displays side-by-side. [Recommended]"

---

### 3.4 Tab 4: Results — RSS

**Uncertainty Budget Table:**
A table showing all sources, organized by V&V 20 category:

| Source | Category | σ [unit] | σ² [unit²] | ν (DOF) | % of u_val² | Distribution | Data Basis |
|---|---|---|---|---|---|---|---|

With subtotals for each category (u_num, u_input, u_D) and a grand total (u_val).

**Combined Results Panel:**

```
Combined Standard Uncertainty: u_val = X.XX °F
Effective DOF (Welch-Satterthwaite): ν_eff = XX
Coverage Factor: k = X.XX [method: <selected method>]
Expanded Uncertainty: U_val = X.XX °F

Comparison Error Statistics:
  Mean: Ē = +X.XX °F
  Std Dev: s_E = X.XX °F
  Sample Size: n = XXX

Validation Assessment:
  |Ē| vs U_val: X.XX vs X.XX → [Bias IS / IS NOT explained by known uncertainties]
  
Estimated Model Form Uncertainty:
  u_model = √(s_E² - u_val²) = X.XX °F  [if s_E > u_val]
  Model form accounts for XX% of total observed variance

Prediction Bounds:
  [One-sided / Two-sided] [Coverage]% coverage, [Confidence]% confidence

  Using u_val only:
    Lower bound = Ē - k × u_val = X.XX °F
    Upper bound = Ē + k × u_val = X.XX °F
    
  Using s_E (includes model form):
    Lower bound = Ē - k × s_E = X.XX °F
    Upper bound = Ē + k × s_E = X.XX °F
    
  Empirical percentiles (from raw comparison data):
    5th percentile = X.XX °F
    95th percentile = X.XX °F
```

**RSS Plots:**
- Pie chart of variance contributions (σ²ᵢ / u_val²)
- Bar chart comparing u_num, u_input, u_D magnitudes
- Normal PDF showing the mean, the expanded bounds (both u_val and s_E based), and the empirical percentiles marked for comparison

---

### 3.5 Tab 5: Results — Monte Carlo

**Monte Carlo Execution:**
- "Run Monte Carlo" button (with progress bar)
- For each trial:
  - For each uncertainty source: draw a random sample from its assigned distribution
    - Normal(0, σ): for sources with normal distribution assumption
    - Uniform(−a, +a): for sources with uniform distribution (a = σ × √3)
    - Triangular(−a, 0, +a): for sources with triangular assumption
    - Empirical resampling (bootstrap): for sources with tabular data — randomly select one value from the data
  - The comparison error source (Tab 1 data) can be sampled as:
    - Bootstrap from raw data (recommended for non-normal data)
    - Normal(Ē, s_E) — parametric assumption
    - User selects which approach
  - Sum all samples to get one combined error value
  - Record the combined value

**Results Panel:**

```
Monte Carlo Results (N = 100,000 trials):

Combined Error Distribution:
  Mean = X.XX °F
  Std Dev = X.XX °F
  5th Percentile = X.XX °F
  95th Percentile = X.XX °F
  
Prediction Bounds:
  Lower bound (5th percentile) = X.XX °F
  Upper bound (95th percentile) = X.XX °F

Bootstrap Confidence on Percentile (if enabled):
  5th percentile: X.XX °F ± X.XX °F (95% CI: [X.XX, X.XX])
```

**Interpretation Guidance:**
- If the MC 5th percentile is less negative than the RSS bound: "The Monte Carlo bound is tighter (less conservative) than the RSS parametric bound. This is typical when the dominant source has lighter tails than normal (platykurtic/uniform-like)."
- If the MC 5th percentile is more negative than the RSS bound: "The Monte Carlo bound is wider (more conservative) than the RSS parametric bound. This may indicate heavier tails or skewness in the dominant source that the normal assumption underestimates."

**MC Plots:**
- Histogram of combined MC distribution with vertical lines at 5th/95th percentiles
- Overlay of the RSS-based normal PDF for visual comparison
- Cumulative distribution function (CDF) — useful for reading off arbitrary percentiles

---

### 3.6 Tab 6: Comparison Roll-Up

This is the final summary tab. It should be print-ready / exportable.

**Side-by-Side Comparison Table:**

| Quantity | RSS (u_val) | RSS (s_E) | Monte Carlo | Empirical |
|---|---|---|---|---|
| Combined σ or equivalent | | | | |
| k-factor used | | | N/A | N/A |
| Lower bound (underprediction) | | | | |
| Upper bound (overprediction) | | | | |
| Mean comparison error | | | | |
| Includes model form error? | No | Yes | Depends on input | Yes |
| Distribution assumption | Normal | Normal | Actual | None |
| Reference standard | V&V 20 / GUM | V&V 20 / GUM | JCGM 101:2008 | — |

**Key Findings Section (auto-generated text):**
Auto-generate a summary paragraph based on the results:
- State the mean bias and whether it's significant
- State the validation assessment (|Ē| vs U_val)
- State the underprediction bound from each method
- Identify the dominant uncertainty source
- Note any data quality concerns (non-normality, small samples, outlier locations)

**Export Button:**
- Export the full roll-up table and findings to a text file or clipboard
- Include the full uncertainty budget table, analysis settings, and all computed values

---

### 3.7 Tab 7: Reference

**Built-in documentation with sub-tabs or collapsible sections:**

1. **V&V 20 Framework Overview**
   - E = S − D diagram
   - u_val = √(u_num² + u_input² + u_D²)
   - Validation assessment: |E| vs U_val
   - What to do when bias exceeds uncertainty

2. **k-Factor Reference Tables**
   - Precomputed tables for one-sided and two-sided, 90/95/99% coverage, 90/95/99% confidence
   - Sample sizes: 5, 7, 10, 15, 20, 30, 50, 60, 100, 150, 200, 500, 1000, ∞
   - Interactive: user can enter n and get k
   - Citation: "One-sided factors computed from non-central t-distribution. Two-sided factors from chi-squared approximation (Wald-Wolfowitz). [GUM Annex G]"

3. **Welch-Satterthwaite Explained**
   - Formula with worked example
   - What ν_eff means physically
   - How supplier specs (ν = ∞) are handled
   - Citation: "JCGM 100:2008 (GUM), Annex G, Section G.4.1, Equation G.2b"

4. **Distribution Guide**
   - Table of common distributions with their effective k for 95% one-sided coverage (known σ):
     - Normal: k = 1.645
     - Uniform: k = 1.559
     - Triangular: k = 1.675
     - Logistic: k = 1.831
   - When normal k is conservative vs non-conservative
   - Decision flowchart from the k-factor guide

5. **Distribution-Free (Non-Parametric) Bounds**
   - Minimum sample size requirements table
   - When they can and cannot be used
   - How order statistics provide bounds without distribution assumptions

6. **Monte Carlo Method**
   - How it works (plain language + mathematical)
   - Why it's equivalent to but more general than RSS
   - LHS vs basic MC — when it matters
   - Citation: "JCGM 101:2008 (GUM Supplement 1)"

7. **Glossary**
   - Coverage, Confidence, Coverage Factor, Tolerance Interval, Tolerance Limit, Standard Uncertainty, Expanded Uncertainty, Type A, Type B, Effective Degrees of Freedom
   - Each term with its precise definition and the standard it comes from

---

## 4. Computational Details

### 4.1 RSS Computation Workflow

```python
# 1. For each source, extract standard uncertainty (σ)
#    Apply sigma basis conversion if needed:
#    - "1σ": use as-is
#    - "2σ": divide by 2
#    - "3σ": divide by 3
#    - "Bounding": divide by √3 (uniform) or 3 (normal 3σ) — depends on distribution selection

# 2. Categorize into u_num, u_input, u_D
u_num = sqrt(sum(σ² for sources in "Numerical"))
u_input = sqrt(sum(σ² for sources in "Input/BC"))
u_D = sqrt(sum(σ² for sources in "Experimental"))

# 3. Combined validation uncertainty
u_val = sqrt(u_num² + u_input² + u_D²)

# 4. Effective degrees of freedom (Welch-Satterthwaite)
# ν_eff = u_val⁴ / Σ(σᵢ⁴ / νᵢ)  [omit sources with νᵢ = ∞]
numerator = u_val ** 4
denominator = sum(σ_i**4 / ν_i for each source where ν_i != inf)
ν_eff = numerator / denominator

# 5. Coverage factor k — depends on user selection:
#    V&V 20 default: k = 2
#    GUM W-S: k = t_p(ν_eff) from scipy.stats.t.ppf for two-sided
#             or from scipy.stats.nct for one-sided tolerance
#    Manual: user-entered value

# 6. Expanded uncertainty
U_val = k * u_val

# 7. Bounds
lower_bound = E_mean - U_val  # using u_val
upper_bound = E_mean + U_val
# OR
lower_bound_sE = E_mean - k * s_E  # using total scatter
upper_bound_sE = E_mean + k * s_E
```

### 4.2 Monte Carlo Computation Workflow

```python
import numpy as np

N = 100_000  # user-configurable

results = np.zeros(N)

for each source:
    if distribution == "Normal":
        samples = np.random.normal(mean, sigma, N)
    elif distribution == "Uniform":
        a = sigma * np.sqrt(3)
        samples = np.random.uniform(mean - a, mean + a, N)
    elif distribution == "Triangular":
        a = sigma * np.sqrt(6)
        samples = np.random.triangular(mean - a, mean, mean + a, N)
    elif distribution == "Empirical":
        samples = np.random.choice(raw_data, size=N, replace=True)
    
    results += samples

# Results
pct_5 = np.percentile(results, 5)
pct_95 = np.percentile(results, 95)

# Bootstrap confidence (if enabled)
bootstrap_pcts = []
for b in range(100):
    # Resample from each source's data (if tabular) or regenerate
    boot_results = run_mc_once(N)
    bootstrap_pcts.append(np.percentile(boot_results, 5))
boot_ci = np.percentile(bootstrap_pcts, [2.5, 97.5])
```

### 4.3 One-Sided Tolerance Factor Computation

```python
from scipy.stats import nct, norm

def one_sided_tolerance_k(n, coverage=0.95, confidence=0.95):
    """
    One-sided normal tolerance factor.
    Uses non-central t-distribution.
    Ref: Krishnamoorthy & Mathew (2009)
    """
    z_p = norm.ppf(coverage)
    delta = z_p * np.sqrt(n)
    k = nct.ppf(confidence, df=n-1, nc=delta) / np.sqrt(n)
    return k
```

### 4.4 Two-Sided Tolerance Factor Computation

```python
from scipy.stats import chi2, norm

def two_sided_tolerance_k(n, coverage=0.95, confidence=0.95):
    """
    Two-sided normal tolerance factor.
    Approximate (Howe / Wald-Wolfowitz).
    """
    z_p = norm.ppf((1 + coverage) / 2)
    chi2_val = chi2.ppf(1 - confidence, df=n-1)
    k = z_p * np.sqrt((n - 1) * (1 + 1/n) / chi2_val)
    return k
```

### 4.5 Distribution-Free Minimum Sample Size

```python
from scipy.stats import binom

def min_n_distribution_free(coverage=0.95, confidence=0.95, r=1):
    """
    Minimum n for the r-th smallest observation to be a
    one-sided lower tolerance bound.
    """
    q = 1 - coverage  # probability below the bound (e.g., 0.05)
    for n in range(r, 10000):
        # P(at least r of n observations fall below the q-quantile)
        prob = 1 - binom.cdf(r - 1, n, q)
        if prob >= confidence:
            return n
    return None
```

---

## 5. GUI Behavior & UX Notes

### 5.1 Live Updates
- As data or sources change, recompute RSS results automatically (they're fast)
- Monte Carlo requires explicit "Run" button click (computationally heavier)
- Show a status bar at the bottom with current state: "Ready", "Computing MC...", "RSS up to date"

### 5.2 Tooltips Everywhere
- Every field, button, and computed value should have a tooltip explaining what it is, why it matters, and where it comes from (which standard/section)
- Tooltips should be concise but informative — aimed at an engineer, not a statistician

### 5.3 Color Coding
- Red backgrounds for warnings (small sample, non-normal dominant source, outlier locations)
- Yellow for moderate cautions
- Green for "looks good"
- Use sparingly — don't overwhelm

### 5.4 Validation Checks
- Before computing, verify: at least one uncertainty source exists, comparison data is loaded (or user has confirmed they want u_val only without comparison data), no negative σ values, no negative DOF
- If comparison data has exactly 0 rows or 0 columns, grey out the s_E bound option and the empirical percentile column in the roll-up

### 5.5 Units
- Global unit selector: °F or °C — displayed in all tables and results
- All computations are unit-agnostic (just numbers), the unit is for display/documentation only

### 5.6 File I/O
- Save/Load project: serialize all inputs to JSON for reproducibility
- Export results: text report with all tables and settings

---

## 6. Important Edge Cases

1. **No comparison data loaded (u_val only):** Valid use case — user just wants to propagate defined uncertainties. Disable comparison-dependent outputs (Ē, s_E bound, empirical percentiles, validation assessment).

2. **Single source dominates (>90% of variance):** Show a prominent note: "Source '[name]' contributes >90% of the combined uncertainty. The effective DOF and combined bound are effectively determined by this source alone."

3. **All sources are Type B (ν = ∞):** Welch-Satterthwaite denominator is 0. Set ν_eff = ∞ and use the z-value (no t-distribution correction needed).

4. **Very small ν_eff (< 5):** Display warning: "Effective DOF is very low. The t-distribution correction significantly inflates k. Consider acquiring more data for the dominant uncertainty source."

5. **Negative model form uncertainty:** If u_val > s_E, then √(s_E² - u_val²) is imaginary. This means the known uncertainties already exceed the observed scatter — possible overestimation of some sources. Display: "Known uncertainties exceed observed scatter. Model form uncertainty cannot be estimated — known uncertainty sources may be overestimated."

6. **Monte Carlo with empirical data and small n:** If bootstrapping from n < 20 data points, the MC results will be granular. Warn the user and suggest fitting a distribution instead of resampling.

---

## 7. Example Default Data

Pre-load the tool with the worked example from this conversation for testing:

**Comparison Data (E = S − D):** 
Mean = +10.18°F, σ = 7.76°F, n = 152, empirical 5th pct = −0.24°F, empirical 95th pct = +26.99°F
(Since we don't have the raw 152 data points, generate synthetic data that matches these statistics for demo purposes)

**Uncertainty Sources:**
| Name | Category | σ | n | Type |
|---|---|---|---|---|
| Iteration Jitter | Numerical | 1.968 | 15453 | Tabular/Sigma |
| Spatial Extraction | Numerical | 0.410 | 3285 | Tabular/Sigma |
| Inlet Temperature Effect | Input/BC | 1.500 | ∞ | Sigma (estimated, 1σ) |
| Flow Rate Calculation | Input/BC | 0.680 | 9 | Sigma (Type A) |
| Flow Rate Sensor | Input/BC | 1.070 | 9 | Sigma (Type A) |
| Sensor Accuracy | Experimental | 2.120 | ∞ | Sigma (supplier, 1σ) |

---

## 8. What Success Looks Like

When complete, the tool should:

1. Accept flexible inputs (tabular data, sigma values, tolerance values, RSS roll-ups, sensitivity runs) for any number of uncertainty sources
2. Correctly categorize sources per V&V 20 and compute u_num, u_input, u_D
3. Compute the RSS combination with proper k-factor options (V&V 20 k=2, GUM W-S effective DOF, one-sided tolerance, manual)
4. Run Monte Carlo with proper distribution sampling and report percentile bounds
5. Present a clear side-by-side comparison of RSS vs Monte Carlo results
6. Guide the user with automated assessments of sample size adequacy, distribution normality, dominant source identification, and appropriate k-factor selection
7. Trace every computation to its standards reference
8. Be usable by an aerospace thermal engineer who is not a statistician

---

## 9. Dependencies

```
PySide6
numpy
scipy
matplotlib
openpyxl  # for Excel file import
```

---

## 10. File Structure

Single file: `vv20_validation_tool.py`

Suggested code organization within the file:
1. Imports and constants
2. Statistical utility functions (k-factors, W-S, MC sampling, distribution tests)
3. Data model classes (UncertaintySource, ComparisonData, AnalysisSettings, Results)
4. Individual tab widget classes
5. Main application window class
6. Entry point (`if __name__ == "__main__"`)

Target: ~2000–3500 lines for a single well-structured file.
