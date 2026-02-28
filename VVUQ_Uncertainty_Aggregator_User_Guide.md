# VVUQ Uncertainty Aggregator v1.2.0 — User Guide & Technical Reference

**CFD Validation Uncertainty Tool per ASME V&V 20 Framework**

*Written for engineers — not statisticians.*

---

## Revision Update (v1.2.0)

- Added **Validation metric mode** in Analysis Settings:
  - `Standard scalar (V&V 20)` (default)
  - `Multivariate supplement (covariance-aware)` for mapped/location-rich comparisons
- Added **Decision Consequence** in the Project Info bar (`Low`, `Medium`, `High`) for risk-based reporting.
- Expanded HTML report with:
  - Decision card (plain-language final actions),
  - Credibility framing checklist,
  - Fixed VVUQ glossary panel,
  - Conformity assessment wording template.
- RSS results text and report now show multivariate supplemental metrics when enabled.
- Distribution auto-fit now uses **automatic GOF runtime guardrails**:
  - sparse datasets use KS screening automatically,
  - larger datasets use adaptive bootstrap sample counts.
- **One-sided validation criterion** now correctly implemented: `Ē ≥ -U_val` (positive Ē always passes as conservative). Previously, the validation check always used the two-sided `|Ē| ≤ U_val` criterion regardless of the one-sided setting.
- **Per-location validation**: Validate at each measurement location independently instead of pooling. Reports "Validated at N/M locations" with spatial breakdown. See per-location table on Tab 1 and per-location findings on Tab 4.
- **Pooling default changed to OFF**: Per-location validation is now the default. Enable pooling explicitly in Tab 3 only when justified.
- **Source scope** field added to uncertainty sources: "Global" (same at all locations) or "Per-location" (varies by T/C).
- **Collapsible advanced sections** across all tools: Correlation groups, uncertainty classification, and other advanced settings are now behind expandable sections to reduce UI clutter.

### Quick Example (Multivariate Supplement)

1. Load comparison data with multiple locations/conditions in **Tab 1**.
2. In **Tab 3**, set **Validation metric mode** to `Multivariate supplement (covariance-aware)`.
3. Compute RSS.
4. In **Tab 4**, review both:
   - scalar check (one-sided: `Ē ≥ -U_val`, or two-sided: `|Ē| ≤ U_val`), and
   - multivariate p-value guidance.
5. Export HTML and use the **Decision Card** section as the one-page transfer summary.

### Quick Example (Per-Location Validation)

1. Load comparison data with **multiple T/C locations × multiple operating conditions** in **Tab 1**.
2. In **Tab 3**, ensure **Pooling** is unchecked (default) and **One-sided** is checked.
3. Set uncertainty **Scope** on each source in **Tab 2** (most are Global; sensor placement may be Per-location).
4. Compute RSS. The per-location breakdown on **Tab 1** now shows PASS/FAIL per location.
5. Review **Tab 4** findings: per-location summary ("Validated at 47/50 locations") and which locations failed.
6. Export HTML for the complete per-location validation report.

---

## Table of Contents

1. [What Is This Tool and Why Do I Need It?](#1-what-is-this-tool-and-why-do-i-need-it)
2. [The Big Picture: What Is V&V 20?](#2-the-big-picture-what-is-vv-20)
3. [The Core Question: "Is My CFD Model Any Good?"](#3-the-core-question-is-my-cfd-model-any-good)
4. [Getting Started — Application Layout](#4-getting-started--application-layout)
5. [Tab 1: Comparison Data](#5-tab-1-comparison-data)
6. [Tab 2: Uncertainty Sources](#6-tab-2-uncertainty-sources)
7. [Tab 3: Analysis Settings](#7-tab-3-analysis-settings)
8. [Tab 4: RSS Results](#8-tab-4-rss-results)
9. [Tab 5: Monte Carlo Results](#9-tab-5-monte-carlo-results)
10. [Tab 6: Comparison Roll-Up](#10-tab-6-comparison-roll-up)
11. [Tab 7: Reference Library](#11-tab-7-reference-library)
12. [Understanding the Math (Plain English)](#12-understanding-the-math-plain-english) *(includes per-location validation, one-sided vs. two-sided criteria)*
13. [Monte Carlo vs Latin Hypercube — What's the Difference?](#13-monte-carlo-vs-latin-hypercube--whats-the-difference)
14. [Distributions — Which One Should I Pick?](#14-distributions--which-one-should-i-pick)
15. [Reading the Guidance Panels](#15-reading-the-guidance-panels)
16. [Project Files and Reports](#16-project-files-and-reports)
17. [Certification and Regulatory Use](#17-certification-and-regulatory-use)
18. [Standards Reference Summary](#18-standards-reference-summary)
19. [Glossary](#19-glossary)
20. [Frequently Asked Questions](#20-frequently-asked-questions)

---

## 1. What Is This Tool and Why Do I Need It?

If you run CFD simulations and need to answer the question *"How much can I trust these results?"*, this tool is for you.

In industries like aerospace, power generation, and automotive, you can't just hand someone a CFD temperature or pressure prediction and call it a day. Regulators and certification authorities want to know:

- **How far off could your prediction be?**
- **What are all the things that could make it wrong?**
- **Can you put a number on that?**

That's what uncertainty quantification (UQ) does. This tool follows the ASME V&V 20 standard — the industry-accepted method for answering those questions for CFD simulations.

**In plain terms:** You feed in your CFD-vs-test-data comparisons and all the things that could affect your answer (grid size, boundary conditions, measurement accuracy, etc.), and the tool tells you:

> *"Your CFD prediction is accurate to within ±X degrees (or psi, or lb/s) with 95% confidence."*

That's the number you put in your report. That's what the certifying authority wants to see.

> **Common Mistakes to Avoid**
>
> Before you dive in, here are the most frequent errors new users make with the Aggregator. Catching these early will save you hours of rework:
>
> - **Entering a 2-sigma value as 1-sigma.** This is the single most common mistake. If a spec says "plus or minus 1.8 degrees F at 95% confidence," that is a 2-sigma number. Entering it as 1-sigma doubles your reported uncertainty. Always check the sigma basis before typing a value.
> - **Forgetting to set the correct sigma basis dropdown.** Every uncertainty source has a sigma basis selector (Confirmed 1-sigma, 2-sigma, 3-sigma, Bounding, etc.). Leaving it on the wrong setting silently corrupts your entire budget. Double-check each source after entry.
> - **Mixing up systematic and random uncertainties.** Systematic errors (bias) and random errors (scatter) are handled differently. A fixed calibration offset is not the same as measurement noise. Make sure each source is categorized correctly as Numerical, Input/BC, or Experimental.
> - **Not including all uncertainty sources.** A missing source means your total uncertainty is artificially low, which can flip a "not validated" result to "validated" for the wrong reasons. Walk through every boundary condition, every instrument, and every modeling assumption before running the analysis.
> - **Using RSS when your inputs are correlated.** RSS assumes all uncertainty sources are independent. If two sources share a common cause (e.g., two thermocouples calibrated with the same reference), RSS underestimates the total. Use the correlation group feature or run Monte Carlo instead.

---

## 2. The Big Picture: What Is V&V 20?

### ASME V&V 20-2009 (Reaffirmed 2016)

**Full title:** *Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer*

This standard provides a structured, defensible process for answering: **"How good is my CFD model?"**

It breaks the problem into two parts:

### Verification: "Did I solve the equations right?"

This is about the numerics — your mesh, your time step, your solver convergence. It's asking whether the computer is giving you the answer to the equations you asked it to solve, not whether those equations represent reality.

**Examples of verification uncertainty:**
- Grid convergence error (coarse mesh vs. fine mesh)
- Time step sensitivity
- Iterative convergence residuals
- Round-off error (rarely significant)

### Validation: "Did I solve the right equations?"

This is about physics — does your model actually represent what happens in the real world? You answer this by comparing your CFD results to experimental data and seeing how well they match up.

**The comparison error:** `E = S - D` (Simulation minus Data). If your CFD says 500°F and the thermocouple reads 495°F, then E = +5°F. Simple.

### The Catch

The comparison error E is NOT just model error. It's contaminated by:

- **Numerical uncertainty (u_num):** Your grid isn't infinitely fine
- **Input/BC uncertainty (u_input):** You don't know the exact inlet temperature
- **Experimental uncertainty (u_D):** The thermocouple has calibration error

V&V 20 says: before you can judge the model, you have to account for all these known error sources. Whatever is left over after that — that's the actual model deficiency.

### The Supporting Standards

| Standard | What It Covers | Think of It As... |
|---|---|---|
| **ASME V&V 20-2009** | The overall framework — how to set up the problem | The playbook |
| **JCGM 100:2008 (GUM)** | How to combine uncertainties mathematically (RSS method) | The math rules |
| **JCGM 101:2008 (GUM Supp. 1)** | Monte Carlo alternative when the math gets complicated | The backup plan |
| **ASME PTC 19.1-2018** | Test uncertainty analysis, sample size requirements | The test engineer's guide |
| **AIAA G-077-1998** | Guide for reporting V&V results | The report template |

---

## 3. The Core Question: "Is My CFD Model Any Good?"

Here's the entire V&V 20 process in a nutshell:

### Step 1: Measure the mismatch
Compare your CFD results to test data at multiple points:
```
E = S - D    (for each sensor, each test condition)
```
Compute the average mismatch: **E-bar (Ē)** — this is your mean bias.

### Step 2: Catalog every source of uncertainty
List everything that could make your answer uncertain:
- Grid convergence study results → gives you **u_num**
- Boundary condition tolerances → gives you **u_input**
- Thermocouple/instrumentation specs → gives you **u_D**

### Step 3: Combine them
Square each one, add them up, take the square root:
```
u_val = √(u_num² + u_input² + u_D²)
```
This is the **Root Sum of Squares (RSS)** method. Think of it like the Pythagorean theorem but for uncertainties.

### Step 4: Expand to your required confidence level
Multiply by a coverage factor **k** (typically k = 2 for 95% coverage):
```
U_val = k × u_val
```
This gives you the **expanded uncertainty** — the width of the error bar.

### Step 5: Make the call
Compare your mean bias to your expanded uncertainty:

| If... | Then... |
|---|---|
| \|Ē\| ≤ U_val | **VALIDATED** — The bias is within the noise. Your known uncertainties can explain the mismatch. |
| \|Ē\| > U_val | **NOT VALIDATED** — There's a systematic bias that can't be explained by known uncertainties. Something else is going on (model deficiency, missing physics, etc.). |

**That's it.** Everything else in this tool is about doing those five steps correctly, rigorously, and in a way that a certification authority will accept.

### What to Do When Validation Fails (|E| > U_val)

Seeing **"NOT VALIDATED"** can be alarming, but it does not mean your CFD is useless. It means the model bias is larger than your known uncertainties can explain. Here are your options, in order of preference:

1. **Check your inputs first.** Go back and verify:
   - Did you include all uncertainty sources? (Missing even one can cause failure.)
   - Are your sigma values on the right basis? (A common mistake is entering 2-sigma as 1-sigma.)
   - Is your grid study converged? (A divergent grid study inflates u_num.)

2. **Look for correctable systematic bias.** If E is consistently positive or negative across all comparison points, you may have a calibration offset. Document it and consider whether a bias correction is justified for your application.

3. **Add missing uncertainty sources.** The most common omissions are:
   - Turbulence model uncertainty (often the largest contributor for thermal predictions)
   - Geometry simplification effects
   - Material property uncertainty

4. **Document and move forward.** Many perfectly useful engineering models are "not validated" by the strict V&V 20 criterion. What matters for certification is that you:
   - Quantified every source you could
   - Reported the gap honestly
   - Explained the engineering significance (e.g., "The 3K bias is within the design margin")

**Do NOT artificially inflate uncertainty sources to force a VALIDATED result.** That defeats the purpose of the standard and will be caught in peer review.

---

## 4. Getting Started — Application Layout

### Main Window

The application has **seven tabs** across the top, plus a collapsible **Project Info** bar:

| Tab | Icon | Purpose |
|---|---|---|
| Comparison Data | 📊 | Enter your CFD-vs-test comparison errors |
| Uncertainty Sources | 📋 | Define all your uncertainty sources |
| Analysis Settings | ⚙️ | Choose coverage, confidence, k-method, etc. |
| Results — RSS | 📈 | See the RSS uncertainty budget and validation result |
| Results — Monte Carlo | 🎲 | Run the Monte Carlo simulation and see results |
| Comparison Roll-Up | 📑 | Side-by-side comparison table + certification statement |
| Reference | 📖 | Built-in standards reference and glossary |

### Project Info Bar
Click the **▶ Project Info** toggle at the top to expand fields for:
- **Program/Project name** — e.g., "Engine Thermal Model V2.3"
- **Analyst** — your name
- **Date** — defaults to today
- **Notes** — free-form text (assumptions, scope, etc.)

The panel starts **collapsed by default** to maximize working space. These fields are saved with the project and appear in the HTML report.

### Auto-Compute
The RSS analysis **automatically recomputes** whenever you change comparison data, uncertainty sources, or analysis settings. You don't need to click anything — just make your changes and the results update live. (Monte Carlo must be run manually because it takes a few seconds.)

### Menu Bar Shortcuts

| Shortcut | Action |
|---|---|
| Ctrl+N | New Project |
| Ctrl+O | Open Project |
| Ctrl+S | Save Project |
| Ctrl+Shift+S | Save As |
| Ctrl+H | Export HTML Report |
| Ctrl+R | Compute RSS |
| Ctrl+M | Run Monte Carlo |
| Ctrl+Shift+A | Compute All (RSS + MC) |

---

## 5. Tab 1: Comparison Data

### What Goes Here

Your **comparison errors** — the difference between your CFD prediction and the test data at each measurement point.

```
E = S - D = (CFD result) - (Test measurement)
```

**Example:** You have 8 thermocouples (TC-01 through TC-08) across 5 flight conditions (FC-001 through FC-005). Each cell in the table is the CFD temperature minus the measured temperature at that location and condition.

### How to Enter Data

**Option 1 — Type directly:** Click cells in the table and type values.

**Option 2 — Paste from Excel:** Copy a block of data from Excel (rows = sensor locations, columns = conditions), click the top-left cell, and paste (Ctrl+V).

**Option 3 — Import from file:** Use the import button to load data from a CSV or Excel file.

### What the Tool Computes Automatically

Once you enter data, the tool immediately calculates:

- **Ē (E-bar):** Mean of all comparison errors — this is your average bias
- **s_E:** Standard deviation of all comparison errors — this is the scatter
- **n:** Total number of data points
- **P5 / P95:** 5th and 95th percentile bounds (where 90% of your data falls)
- **Skewness:** Is the data lopsided? (0 = symmetric, like a normal bell curve)
- **Kurtosis:** Are there heavy tails / outliers? (0 = normal, >0 = more outliers than expected)
- **Shapiro-Wilk test:** A statistical normality test (p > 0.05 = data looks reasonably normal)

### Per-Location Statistics

A separate table below shows the mean, std, and count for **each sensor individually**. This helps you spot:
- A sensor that's consistently off (might indicate a local model issue)
- A sensor with way more scatter than the others (might be a bad TC)

### Distribution Fitting

The tool automatically fits 8 standard distributions to your data and ranks them by AICc (small-sample corrected AIC). For goodness-of-fit (GOF), it applies bootstrap AD when practical, and switches to KS screening automatically in sparse/fast-mode cases. This keeps the process responsive for junior users while preserving a clear pass/fail gate.

### Guidance Panels

- **Distribution Assessment:** Green if your data looks normal, yellow if skewed or heavy-tailed, red if severely non-normal. This matters because the RSS method assumes normality — if your data isn't normal, the Monte Carlo method is more reliable.
- **Sample Size Adequacy:** Green if you have enough data points (n ≥ 60), yellow for moderate samples (20–59), red for small samples (< 20). Small samples mean wider error bars because you're less certain about the true scatter.

### Plots

- **Histogram:** Shows the shape of your comparison error distribution with a normal curve overlay
- **QQ-Plot:** Points fall on the diagonal line if data is normal; curves away if not

---

## 6. Tab 2: Uncertainty Sources

### What Goes Here

Every source of uncertainty that could affect your CFD-vs-test comparison. Think of this as your **uncertainty budget** — an itemized list of everything that could make your answer wrong.

### The Three Categories (per V&V 20)

| Category | Symbol | What It Covers | Examples |
|---|---|---|---|
| **Numerical (u_num)** | u_num | Errors from solving the equations on a computer | Grid convergence, time step sensitivity, iterative residuals |
| **Input/BC (u_input)** | u_input | Uncertainty in what you told the CFD model | Inlet temperature tolerance, material property uncertainty, geometry tolerances |
| **Experimental (u_D)** | u_D | Uncertainty in the test measurements you're comparing to | Thermocouple accuracy, DAQ noise, probe positioning |

### Adding a Source

Click **"Add Source"** and fill in:
- **Name:** Something descriptive (e.g., "Grid Convergence — Fine to Medium")
- **Category:** Numerical, Input/BC, or Experimental
- **Distribution:** What shape the uncertainty has (see [Section 14](#14-distributions--which-one-should-i-pick))
- **Input Type:** How you're specifying the uncertainty magnitude (see below)
- **Enabled:** Check/uncheck to include or exclude from the analysis

### Five Ways to Specify Uncertainty (Input Types)

| Input Type | When to Use It | What You Enter |
|---|---|---|
| **Tabular Data** | You have actual measured data points | Paste or type the raw data values; tool computes σ and DOF automatically |
| **Sigma Value Only** | You have a single number from a study or spec | Enter the value and select its basis (see below) |
| **Tolerance / Expanded Value** | A manufacturer spec says "±X at 95% confidence" | Enter X and the k-factor it was computed with |
| **RSS of Sub-Components** | You already combined several sub-sources externally | Enter the pre-combined RSS value |
| **CFD Sensitivity Run** | You ran perturbed CFD cases to measure sensitivity | Enter the delta values from each perturbation |

### Sigma Basis — This Is Important!

When you enter a sigma value, you MUST tell the tool what basis it's on. This is the #1 source of mistakes in uncertainty analysis.

| Basis | What It Means | Example |
|---|---|---|
| **Confirmed 1σ** | You've verified this is a true 1-sigma (one standard deviation) value | You computed std dev from 30+ data points |
| **Assumed 1σ (unverified)** | You think it's 1σ but haven't proven it — this gets flagged in the audit | Engineering judgment or rough estimate |
| **2σ (95%)** | The value represents a 95% confidence interval half-width | Manufacturer spec: "±1.8°F at 95% confidence" |
| **3σ (99.7%)** | The value represents a 99.7% interval half-width | "Worst case" or "bounding" from specs |
| **Bounding (min/max)** | The value is an absolute maximum — the error can NEVER exceed this | Physical limits, calibration certificates |

**Why this matters:** If a thermocouple spec says "±1.8°F (2σ, 95%)" and you enter 1.8 as "Confirmed 1σ", you've just doubled your thermocouple uncertainty. The tool uses the basis to correctly convert to 1σ.

### Sigma Basis — Quick Reference with Real-World Examples

Not sure which sigma basis your data is in? Use this table:

| Where your number came from | What basis it probably is | What to select |
|---|---|---|
| Calibration certificate says "±0.5 K" | Usually 2σ (95% confidence) — check the certificate | Confirmed 2σ |
| Manufacturer datasheet says "accuracy ±1%" | Usually 2σ or 3σ — check the fine print | Confirmed 2σ (conservative) |
| You ran a grid study and got u_num from the GCI tool | Already 1σ (standard uncertainty) | Confirmed 1σ |
| You ran the Iterative Uncertainty tool and got sigma | Already 1σ | Confirmed 1σ |
| You ran the Statistical Analyzer and got σ | Already 1σ (population std deviation) | Confirmed 1σ |
| A colleague said "the uncertainty is about 2 degrees" | Unknown basis — ask them to clarify | Unverified 1σ (conservative) |
| You estimated it yourself from experience | Expert judgment — no statistical basis | Unverified 1σ |
| A textbook says "typical uncertainty is ±X" | Often 2σ, but varies — check the source | Unverified 2σ |
| Tolerance from a drawing (e.g., ±0.005 inches) | This is a range, not sigma | Uniform (half-range) |

**The #1 mistake:** Entering a 2σ or 95% CI value as if it were 1σ. This doubles your reported uncertainty. Always check the source document.

### The Mini Distribution Preview

Each source shows a small plot of the assumed PDF shape. This is a sanity check — does the shape look like what you expect? A uniform distribution is flat (equal probability everywhere), a normal distribution is the classic bell curve, etc.

### Degrees of Freedom (DOF)

DOF tells the tool how much data backs up each uncertainty estimate:

| Data Source | DOF | What It Means |
|---|---|---|
| Sample of n measurements | n - 1 | You computed σ from real data |
| Manufacturer spec / supplier data | ∞ (infinity) | You trust the number as-is; no sampling uncertainty |
| Expert judgment | Very high (∞) | Treat as fully known (but flag as assumption) |

**Why DOF matters:** Small DOF means you're less sure about your σ estimate, which means the coverage factor k needs to be larger to compensate. If you only have 5 data points, your k could be 3+ instead of 2. (More on this in the k-factor section.)

### Asymmetric Uncertainty (σ⁺/σ⁻)

Sometimes uncertainty is not symmetric — the effect of a +10% perturbation is different from a -10% perturbation. Common examples include:

- **Material property sensitivity:** A ±10% change in conductivity produces +3°F / -5°F in your prediction
- **Geometry tolerances:** Tighter gap → much hotter; wider gap → slightly cooler
- **Boundary condition one-sided tests:** You only ran the "hot" perturbation, not the "cold" one

The tool supports asymmetric uncertainty through dedicated σ⁺/σ⁻ fields:

**Enabling asymmetric mode:**
1. In the Sigma Value input area, check the **"Asymmetric"** checkbox
2. The single σ field is replaced by two fields: **σ⁺** (positive direction) and **σ⁻** (negative direction)
3. The tool computes an effective symmetric σ for RSS: `σ_eff = √((σ⁺² + σ⁻²) / 2)` per GUM §4.3.8

**One-sided sensitivity results:**

If you only tested one direction (e.g., you only ran the "hot" perturbation):
1. Check the **"One-sided"** checkbox
2. Select the direction you tested: **Upper** or **Lower**
3. If **"Mirror assumed"** is checked (default), the tool assumes the untested direction has the same magnitude — e.g., if σ⁺ = 3.0°F from your test, σ⁻ is assumed to also be 3.0°F
4. If unchecked, the untested direction is set to 0 — only use this if you have physical reasons to believe there is no effect in the other direction

**How asymmetric values propagate:**

| Method | How Asymmetry Is Handled |
|---|---|
| **RSS** | Uses effective σ = √((σ⁺² + σ⁻²) / 2), which gives a single combined value for the standard RSS formula |
| **Monte Carlo** | Samples from a **bifurcated Gaussian** (Barlow's split-normal): uses σ⁺ for the positive half and σ⁻ for the negative half. This preserves the full asymmetric shape in the output distribution. |

**Budget table display:** When any source is asymmetric, the budget table shows `σ⁺=X / σ⁻=Y` inline. Hover over the cell for a tooltip with the effective σ value.

**Evidence notes:** When one-sided with mirror assumption, the tool automatically adds a note to the source: "One-sided sensitivity — mirror symmetry assumed for untested direction."

**When to use asymmetric mode:**
- You have sensitivity results that differ significantly between +/- perturbations (ratio > 1.5)
- You only tested one direction and want to document the assumption
- Your certifying authority requires you to preserve directionality information

**When NOT to use asymmetric mode:**
- σ⁺ and σ⁻ are within ~20% of each other — just use the larger one as a symmetric σ
- You want the simplest defensible analysis (symmetric is always acceptable per GUM)

---

## 7. Tab 3: Analysis Settings

### Coverage and Confidence

These two numbers define how conservative your uncertainty statement is.

**Coverage (default: 95%):** "I want my error bars to contain the true answer X% of the time." A 95% coverage interval means that if you repeated the whole experiment 100 times, about 95 of those times the true answer would fall within your stated bounds.

**Confidence (default: 95%):** "I'm Y% confident that my error bars are actually wide enough to achieve that coverage." This accounts for the fact that your uncertainty estimates themselves are uncertain (because they're based on finite data).

**Together: "95/95"** means 95% coverage at 95% confidence — you're 95% sure that 95% of the distribution is captured. This is the standard requirement for aerospace thermal certification.

### One-Sided vs. Two-Sided

**One-sided (default):** You only care about underprediction — the model being too cold (or too low) is the safety risk. Overprediction (model too hot) is conservative and always acceptable. This affects **two things**:

1. **The k-factor:** Uses the one-sided t-distribution (e.g., k = 1.645 instead of 1.96 for infinite DOF at 95%)
2. **The validation criterion:** Changes from `|Ē| ≤ U_val` (two-sided) to `Ē ≥ -U_val` (one-sided). This means **positive Ē of any magnitude always passes** — the model is conservative. Only negative Ē (underprediction) exceeding -U_val triggers a failure.

**Two-sided:** You care about both directions equally — "How far off could it be in either direction?" Both over- and under-prediction count. The criterion is `|Ē| ≤ U_val`.

**For aerospace thermal certification:** One-sided is standard because you care about maximum temperature for material limits. A model that predicts too hot gives you conservative design margins. See [Section 12](#12-understanding-the-math-plain-english) for detailed examples and diagrams.

### Pooling

**Pooling (default: OFF):** When unchecked, the tool performs **per-location validation** — each thermocouple or measurement location is validated independently, and you get a spatial summary ("validated at 47/50 locations"). This is the recommended approach when you have multiple measurement locations with different physics (laminar vs. turbulent regions, gradient vs. uniform regions).

**When checked:** All comparison data points across all locations are pooled into a single dataset. One Ē, one s_E, one validation verdict. This is simpler but can hide spatial problems where some locations pass and others fail.

**When pooling is appropriate:** Only when all locations have similar mean errors and similar scatter (i.e., the physics is the same everywhere). If in doubt, leave pooling OFF and use per-location validation. See [Section 12](#12-understanding-the-math-plain-english) for a detailed explanation of when pooling can be misleading.

### K-Factor Method

The coverage factor **k** is the multiplier that converts your 1σ uncertainty into an expanded uncertainty at your chosen coverage level. There are four ways to get it:

| Method | When to Use | What It Does |
|---|---|---|
| **ASME V&V 20 Default (k=2)** | Standard practice, large datasets | Uses k=2 regardless of DOF. Simple, conservative for large datasets. |
| **GUM Welch-Satterthwaite** | You want a data-driven k | Computes effective DOF from your sources, then looks up k from the Student-t table. Gives you credit for having lots of data. |
| **One-Sided Tolerance Factor** | Certification applications | The most rigorous method — accounts for both coverage AND confidence using the non-central t-distribution. |
| **Manual k Entry** | Special requirements | You type in whatever k your certifying authority requires. |

**Rule of thumb:**
- If you have lots of data (effective DOF > 30): all methods give similar results (k ≈ 2)
- If you have limited data (DOF < 10): the tolerance factor method will give a larger k, which is more conservative and more defensible

### Monte Carlo Settings

**Sampling Method:**
- **Monte Carlo (Random):** Standard random sampling. The classic approach. Reliable but may need more trials.
- **Latin Hypercube (LHS):** A smarter sampling strategy that ensures better coverage of the probability space. Gets the same accuracy with fewer trials. (See [Section 13](#13-monte-carlo-vs-latin-hypercube--whats-the-difference) for the full explanation.)

**Number of Trials:** How many random draws to make. 100,000 is the default and is adequate for most applications. Increase to 1,000,000 if you need rock-solid tail probabilities.

**Random Seed:** Set to a specific number for reproducible results (same seed = same answer every time). Leave at "None (random)" for a fresh random run each time.

**Bootstrap Confidence Intervals:** When enabled (recommended), the tool estimates how uncertain the MC percentile bounds themselves are. This is a quality check on the MC simulation.

### Bound Type

How to construct the validation bound:

| Option | What It Uses | When to Choose |
|---|---|---|
| **Known uncertainties only (u_val)** | Only your catalogued uncertainty sources | Strict V&V 20 interpretation |
| **Total observed scatter (s_E)** | The standard deviation of comparison errors | More conservative; includes model form effects |
| **Both (for comparison)** *(recommended)* | Shows both side by side | Best for understanding — you can see if s_E > u_val (which means there are unmodelled effects) |

---

## 8. Tab 4: RSS Results

### The Uncertainty Budget Table

This is the heart of the analysis. It's a table listing every enabled uncertainty source with:

| Column | What It Shows |
|---|---|
| Source | Name of the uncertainty source |
| Category | u_num, u_input, or u_D |
| σ (1σ) | Standard uncertainty in your chosen units |
| σ² | Variance — this is what gets added in RSS |
| ν (DOF) | Degrees of freedom (∞ for supplier data) |
| % of u_val² | How much of the total variance comes from this source |
| Distribution | Assumed distribution shape |
| Data Basis | Where the number came from |
| Class | Uncertainty class: Aleatoric, Epistemic, or Mixed |
| Reducibility | How much this uncertainty can be reduced by additional work: Low, Medium, or High |

**Color coding in the % column:**
- **Red highlight (> 80%):** This source dominates the total uncertainty — focus your efforts on reducing this one
- **Yellow highlight (> 50%):** This source is a major contributor

**Subtotal rows** (gray italic) show the RSS within each category, and the **Grand Total row** (blue/white) shows the combined u_val.

### Source Classification and Correlation

#### Uncertainty Classification

Each uncertainty source can be classified using these fields (set in the source editor on Tab 2, under the collapsible "Advanced: Uncertainty Classification" section):

- **Uncertainty Class** — Aleatoric (inherent variability), Epistemic (knowledge gap), or Mixed. This classification affects the class-split summary (U_A, U_E) shown in results.
- **Basis Type** — How the uncertainty value was obtained: measured, assumed, spec_limit, expert_judgment, or model_ensemble.
- **Reducibility** — Whether additional testing or analysis could reduce the uncertainty: low, medium, or high.
- **Evidence Note** — Free-text field for documenting the basis or justification.
- **Scope** — "Global (same at all locations)" or "Per-location (varies by T/C)". Global sources (grid convergence, iteration sensitivity, inlet BC) contribute identically to every location's U_val during per-location validation. Per-location sources (sensor placement accuracy in gradient regions) are tracked separately. See [Section 12](#12-understanding-the-math-plain-english) for details on per-location validation.

The results text and HTML report include a class-split summary showing U_A (aleatoric), U_E (epistemic), and the epistemic fraction of total variance. When epistemic uncertainty exceeds 50% of total variance, a warning is displayed recommending knowledge-reduction actions.

#### Correlation Groups

Sources that share a common systematic influence (e.g., thermocouple calibration, shared boundary condition) can be placed in a **correlation group** with a pairwise correlation coefficient:

- **Correlation Group** — A text label (e.g., "TC_cal"). Correlation is applied when sources share both the same group name and the same V&V category (`u_num`, `u_input`, or `u_D`). Cross-category terms are intentionally not applied.
- **Correlation Coefficient (ρ)** — The correlation of this source with the group reference source. Range: -1.0 to +1.0.

**Reference source rule:** Within each group, the first source alphabetically is automatically designated the reference source (ρ = 1.0). Other sources specify their correlation with this reference.

**Transitivity formula:** For sources a and b in the same group, the pairwise correlation is ρ(a,b) = ρ_a × ρ_b. This single-reference factor model produces a valid positive-semi-definite correlation matrix.

**Monte Carlo correlation scope:** Correlated Monte Carlo sampling is applied when all sources in a same-category correlation group use `Normal` distributions. If a group includes non-Normal sources, Monte Carlo falls back to independent sampling for that group and reports a note in the MC results/report.

**Worked example (3 sources in group "TC_cal"):**

| Source | User-entered ρ | Role |
|--------|---------------|------|
| TC_inlet | 0.9 | Reference (1st alphabetically — ρ forced to 1.0) |
| TC_outlet | 0.8 | Correlated with reference |
| TC_wall | 0.6 | Correlated with reference |

Effective pairwise correlation matrix computed by the tool:

|  | TC_inlet | TC_outlet | TC_wall |
|--|----------|-----------|---------|
| TC_inlet | 1.00 | 0.80 | 0.60 |
| TC_outlet | 0.80 | 1.00 | 0.48 |
| TC_wall | 0.60 | 0.48 | 1.00 |

Note: ρ(outlet, wall) = 0.8 × 0.6 = 0.48 via transitivity. The user-entered ρ = 0.9 for TC_inlet is overridden to 1.0 because it is the reference source. This matrix appears in the results text and the HTML report.

The results text and HTML report display the effective pairwise correlation matrix for each group, showing the actual coefficients used in the computation.

If ρ = 0.0 for a source in a group, an audit warning is logged since it makes that source effectively independent of group members.

#### Chart Export Controls

Each chart toolbar includes three output actions:

1. **Copy to Clipboard** — Draft quality (150 DPI), for quick sharing.
2. **Copy Report-Quality** — 300 DPI with a light colour scheme for formal reports.
3. **Export Figure Package...** — Multi-format archive (PNG 300/600, SVG, PDF) plus JSON metadata sidecar with traceability fields (tool version, analysis ID, settings hash, timestamps, units, method context).

### The Results Summary

A monospace text panel showing the full computation trace:

```
Combined Standard Uncertainty (u_val):
  u_num   = 1.7088 [°F]    (45.4% of u_val²)
  u_input = 2.0000 [°F]    (62.2% of u_val²) ← note: percentages don't
  u_D     = 0.9849 [°F]    (15.1% of u_val²)    add to 100% because that
  u_val   = 2.8284 [°F]                           would be the variance %

Effective DOF (Welch-Satterthwaite):
  ν_eff = 47.3

Coverage Factor:
  Method: ASME V&V 20 Default
  k = 2.0000

Expanded Uncertainty:
  U_val = k × u_val = 5.6569 [°F]

Validation Assessment (one-sided):
  Ē ≥ -U_val:  +5.0000 vs -5.6569 → ✓ VALIDATED
  (One-sided underprediction criterion: E ≥ -U_val)
```

In two-sided mode, the same results would display as:
```
Validation Assessment (two-sided):
  |Ē| ≤ U_val:  5.0000 vs 5.6569 → ✓ VALIDATED
```

### The Four Guidance Panels

These traffic-light panels give you immediate, plain-language feedback:

1. **Dominant Source Check:** Tells you which source(s) drive the total uncertainty. If one source is > 80% of the total, that's where your effort should go.

2. **Degrees of Freedom Check:** Warns you if your effective DOF is low (meaning k=2 might not be conservative enough). Green ≥ 30, Yellow 5–30, Red < 5.

3. **Model Form Assessment:** Compares s_E (observed scatter) to u_val (known uncertainty). If s_E is much bigger than u_val, there are physics your model is missing.

4. **Validation Assessment:** The big one — VALIDATED or NOT VALIDATED. In one-sided mode, the criterion is `Ē ≥ -U_val` (positive Ē always passes). In two-sided mode, the criterion is `|Ē| ≤ U_val`.

### Plots

- **Variance Pie Chart:** Visual breakdown of which sources contribute most
- **Category Bar Chart:** u_num vs u_input vs u_D comparison
- **Normal PDF with Bounds:** Shows the assumed normal distribution with your prediction bounds overlaid

---

## 9. Tab 5: Monte Carlo Results

### Why Monte Carlo?

The RSS method assumes all your uncertainties combine into a nice, normal (bell curve) distribution. That's often true (thanks to the Central Limit Theorem), but not always. If you have:

- Highly skewed distributions (like lognormal)
- Very few uncertainty sources (CLT needs several to kick in)
- Uniform or triangular distributions (which have hard cutoffs)
- One dominant source with a non-normal distribution

...then the RSS assumption may be wrong, and the Monte Carlo method gives you a more honest answer because it does not force a single Normal combined shape.

### How It Works (The Dartboard Analogy)

Imagine each uncertainty source is a spinner wheel. The width of the wheel represents how uncertain that source is, and the shape of the markings represents the distribution.

The Monte Carlo method spins ALL the wheels simultaneously — say, 100,000 times. Each spin gives you one possible "total error." After 100,000 spins, you have a complete picture of what the combined error distribution actually looks like. You still rely on the source distributions you selected, but you do not force a Normal combined shape.

You then simply read off the 5th and 95th percentile (or whatever your coverage requires) from the actual distribution of results.

### Running the Simulation

1. Make sure your uncertainty sources and comparison data are entered
2. Click the blue **"Run Monte Carlo"** button (or press Ctrl+M)
3. A progress bar shows the computation progress
4. Results appear in about 1–10 seconds depending on settings

### Understanding the Results

The results text shows:

```
Latin Hypercube (LHS) Results (N = 100,000 trials):

Combined Error Distribution:
  Mean      = +5.0123 [°F]
  Std Dev   = 2.8345 [°F]
  P5        = +0.2456 [°F]
  P95       = +9.7891 [°F]

Prediction Bounds (95% one-sided):
  Lower bound (P5)  = +0.2456 [°F]
  Upper bound (P95) = +9.7891 [°F]

Bootstrap Confidence on Percentiles (1000 resamples):
  P5:  +0.2456 ± 0.0812  (95% CI: [+0.0876, +0.4067])
  P95: +9.7891 ± 0.0743  (95% CI: [+9.6448, +9.9312])
```

**Key things to look at:**
- **Mean** should be close to Ē from the comparison data
- **P5 and P95** are your Monte Carlo prediction bounds
- **Bootstrap CI** tells you how stable those bounds are — narrow = good

### Guidance Panels

1. **MC Convergence Check:** Did you run enough trials? Green if the percentile estimates are stable to within 1%. If you get yellow or red, increase the trial count.

2. **MC vs RSS Comparison:** Compares the MC bounds to the RSS bounds. If they agree (within ~5%), the normal distribution assumption was fine. If MC gives wider bounds, your data has heavier tails or skew — use the MC results.

### Plots

- **Histogram:** The actual shape of the combined uncertainty distribution, with the RSS normal curve overlaid for comparison
- **CDF (Cumulative Distribution Function):** Shows the probability of the error being below each value, with your coverage percentiles marked
- **Convergence Plot:** Running percentile values vs. number of trials — the curves should flatten out (converge) well before 100,000

---

## 10. Tab 6: Comparison Roll-Up

### The Roll-Up Table

This is the executive summary — a single table that puts all the results side by side for easy comparison:

| Row | RSS (u_val) | RSS (s_E) | Monte Carlo | Empirical |
|---|---|---|---|---|
| Combined σ | u_val | s_E | MC std dev | Data std dev |
| k-factor | k | k | N/A (dist-free) | N/A |
| Expanded U | k × u_val | k × s_E | ½ × (P95 - P5) | ½ × (P95 - P5) |
| Lower bound | Ē - k·u_val | Ē - k·s_E | MC P5 | Data P5 |
| Upper bound | Ē + k·u_val | Ē + k·s_E | MC P95 | Data P95 |
| Mean error | Ē | Ē | MC mean | Data mean |
| Validated? | Ē ≥ -U_val? (one-sided) or \|Ē\| ≤ U_val? (two-sided) | \|Ē\| ≤ k·s_E? | 0 in [P5, P95]? | — |
| Distribution | Normal | Normal | Actual (sampled) | Empirical |
| Reference | V&V 20-2009 | V&V 20-2009 | JCGM 101:2008 | — |

### Auto-Generated Certification Statement

Below the table, the tool generates a multi-section certification-ready finding that covers:

1. **Mean Bias Assessment** — Is the bias statistically significant?
2. **Underprediction/Overprediction Bounds** — Worst case in each direction
3. **Dominant Source** — What drives the uncertainty?
4. **Data Quality** — Any concerns about sample size or normality?
5. **Validation Verdicts** — RSS assessment, MC assessment, MC-vs-RSS comparison
6. **Recommended CFD Accuracy Statement** — Copy-paste-ready language for your report

### Compare Projects

You can load a second project file to add comparison columns to the roll-up table (e.g., comparing a previous model version to the current one).

### Export Options

- **Export to Clipboard:** Copies the table in tab-separated format for pasting into Excel
- **Export Full Report (HTML):** Generates a comprehensive HTML report (see [Section 16](#16-project-files-and-reports))
- **Save Project:** Saves everything to a JSON file

---

## 11. Tab 7: Reference Library

Eight built-in reference sub-tabs so you don't have to leave the application:

| Sub-Tab | What's In It |
|---|---|
| **V&V 20 Overview** | The complete framework diagram, E = S - D equation, RSS formula, validation assessment criterion |
| **k-Factor Tables** | Interactive calculator + precomputed tables for all combinations of sample size, coverage, and confidence |
| **Welch-Satterthwaite** | Full explanation of the effective DOF formula with a worked example |
| **Distribution Guide** | All 12 distributions with shape descriptions, k-factors, and when to use each |
| **Uncertainty Classification Guide** | Aleatory vs. epistemic uncertainty classification with practical CFD examples, combination diagram, one-sided uncertainty guidance, and budget dominance interpretation |
| **Distribution-Free Bounds** | Non-parametric tolerance intervals — when you can't assume any distribution at all |
| **Monte Carlo Method** | MC and LHS explanation, convergence criteria, bootstrap interpretation |
| **Glossary** | Definitions of every technical term used in the tool |

### Uncertainty Classification Guide (New)

This sub-tab helps you understand whether each uncertainty source is **aleatory** (inherent randomness — cannot be reduced) or **epistemic** (knowledge gap — can be reduced with more data or better models). This classification matters because:

- **Epistemic-dominant budgets** suggest the analysis can be improved by collecting more data, using better instrumentation, or employing higher-fidelity models
- **Aleatory-dominant budgets** are at the irreducible floor — further investment won't shrink the uncertainty

The guide includes:
- **Definitions** of aleatory and epistemic uncertainty with practical explanations
- **8 practical CFD examples** with classification, rationale, and reducibility assessment (e.g., iterative convergence scatter, discretization error, thermocouple measurement error, turbulence model form error)
- **Combination diagram** showing how u_num, u_input, and u_D combine into u_val and how they split into aleatory and epistemic components
- **One-sided uncertainty guidance** for cases where only one direction was tested
- **Budget dominance table** to help interpret whether your budget is dominated by reducible or irreducible sources

### Classification Guardrails for Common CFD Cases

- **Inlet mass flow from sensors is often mixed, not purely epistemic.**
  Random repeatability/noise is aleatory, while calibration/setup bias is epistemic.
  If you did not separate those components, classify as **Mixed** and explain in `Evidence`.
- **Thermocouple/temperature measurement uncertainty is usually mixed for the same reason.**
  Noise and drift/recalibration effects are different mechanisms.
- **Over-refined/asymptotic mesh behavior does not automatically mean “pure epistemic.”**
  The source is still numerical/modeling in origin, but if refinement no longer reduces it,
  treat it as **Mixed with low reducibility** and document the asymptotic evidence.
- **Classification does not exclude sources from the math.**
  RSS and Monte Carlo still combine all enabled sources; the class tags are used for split reporting and improvement prioritization.
- **Type A / Type B is not the same as aleatory / epistemic.**
  Type A/B describes how the number was estimated (from data vs other information), not its physical class.

### ASME V&V 20 Compliance Note — Epistemic and Aleatory Uncertainty Combination

This tool combines all uncertainty sources via Root-Sum-Square (RSS) regardless of their epistemic or aleatory classification, following ASME V&V 20-2009 (R2021) Section 9 and the GUM (JCGM 100:2008) framework. Under this approach:

- Each uncertainty source is characterized as a standard uncertainty (1-sigma equivalent)
- Epistemic intervals are converted to standard uncertainties by assuming a distribution (e.g., Uniform interval +/-a becomes sigma = a/sqrt(3))
- All standard uncertainties are combined via RSS, assuming independence

This is the industry-standard pragmatic approach used in most aerospace CFD validation programs. However, when epistemic sources dominate the uncertainty budget (>50% of u_val), analysts should be aware that RSS assumes random cancellation that may not occur for systematic knowledge-gap uncertainties. For applications requiring stricter separation of epistemic and aleatory uncertainties, more advanced frameworks such as Oberkampf & Roy (2010) recommend double-loop Monte Carlo methods producing probability boxes (p-boxes).

The tool tracks and reports the epistemic/aleatory split to support prioritization of uncertainty reduction efforts. When the Monte Carlo method is used, all sources are sampled from their declared distributions in a single loop (consistent with GUM Supplement 1, JCGM 101:2008).

---

## 12. Understanding the Math (Plain English)

### "Root Sum of Squares" — RSS

You have several independent uncertainty sources. You need to combine them into one total uncertainty. You can't just add them (that would be way too conservative — it assumes everything goes wrong in the same direction at the same time). Instead, you add the **squares**, then take the square root:

```
u_total = √(u₁² + u₂² + u₃² + ...)
```

**Think of it like this:** Uncertainties are like vectors pointing in random directions. If you add vectors that point in random directions, the total length is the square root of the sum of the squared lengths — that's the Pythagorean theorem. RSS is the Pythagorean theorem for uncertainties.

### Coverage Factor k — "The Multiplier"

Your u_val is a 1-sigma value — it only covers about 68% of the distribution. You need to widen it to cover 95% (or whatever your requirement is). That's what k does:

```
U_val = k × u_val
```

**For a normal distribution:**
- k = 1.0 → covers 68% (1σ)
- k = 1.645 → covers 95% one-sided
- k = 1.96 → covers 95% two-sided
- k = 2.0 → covers ~95.4% two-sided (the V&V 20 default)
- k = 3.0 → covers 99.7% two-sided

**When you have limited data,** k gets bigger because you're less certain about your σ estimate. With only 5 data points, k might be 3.4 instead of 2.0. That's not being pessimistic — that's being honest about how much you don't know.

### Welch-Satterthwaite — "Effective Degrees of Freedom"

When you combine multiple uncertainty sources, each with different amounts of data behind them, the combined result has an "effective" number of degrees of freedom. This is a weighted blend of all the individual DOFs.

```
ν_eff = u_val⁴ / Σ(uᵢ⁴ / νᵢ)
```

**The intuition:** If you have one source based on 5 data points and another based on 1000 data points, the combined DOF is somewhere in between — pulled toward the smaller DOF because the weakest link limits your overall confidence.

**Type B sources (supplier specs)** have infinite DOF, so they drop out of the formula entirely. They don't help or hurt your effective DOF — they're just taken at face value.

### The Validation Check — Two Modes

This is the final pass/fail criterion from ASME V&V 20. The exact rule depends on whether you selected **one-sided** or **two-sided** mode in Analysis Settings (Tab 3).

#### Two-Sided Criterion: |Ē| ≤ U_val

Use this when **both over- and under-prediction matter equally** (rare in thermal certification, common in structural or aerodynamic applications).

> **"Is the average mismatch small enough — in either direction — to be explained by the known uncertainties?"**

- **Yes (|Ē| ≤ U_val):** Validated. The bias could plausibly be zero.
- **No (|Ē| > U_val):** Not validated. The bias is too large in one direction or the other.

```
Two-sided criterion:

  NOT VALIDATED       VALIDATED       NOT VALIDATED
  ◄──────────────|─────────────────|──────────────────►
              -U_val       0       +U_val
                 ↑                    ↑
          Ē must be between these two lines
```

#### One-Sided Underprediction Criterion: Ē ≥ -U_val

Use this when **only underprediction is a concern** — the model being too cold (or too low) is the safety risk. **This is the default and the most common choice for aerospace thermal work.**

> **"Is the model's underprediction small enough to be explained by the known uncertainties?"**

The key insight: **overprediction (positive Ē) is always conservative.** If the model predicts hotter than reality, your design margins are on the safe side. So positive Ē — no matter how large — always passes.

- **Ē = +50°F, U_val = 10°F → VALIDATED.** The model is conservative by 50°F. That's a feature, not a problem.
- **Ē = +2°F, U_val = 10°F → VALIDATED.** Slight overprediction, well within uncertainty.
- **Ē = -8°F, U_val = 10°F → VALIDATED.** The underprediction (-8°F) is within what the uncertainties can explain (-10°F).
- **Ē = -15°F, U_val = 10°F → NOT VALIDATED.** The underprediction (-15°F) exceeds what the uncertainties can explain (-10°F). The model is too cold and we can't blame it on measurement noise.

```
One-sided underprediction criterion:

  NOT VALIDATED          VALIDATED (always)
  ◄──────────────|──────────────────────────────────────►
              -U_val         0        +anything
                ↑
         Ē must be to the right of this line
         (positive Ē always passes — it's conservative)
```

#### Which Mode Should I Use?

| Situation | Mode | Why |
|-----------|------|-----|
| Thermal certification (max temp matters) | **One-sided** | You only care if the model underpredicts temperature. Overprediction is conservative. |
| Structural certification (max stress matters) | **One-sided** | You only care if the model underpredicts stress. Overprediction is conservative. |
| Aerodynamic lift/drag — both directions matter | **Two-sided** | Overpredicting drag is as bad as underpredicting it for performance guarantees. |
| General scientific comparison | **Two-sided** | You want the model to be accurate, not just conservative. |

#### Common Misconception

> "My Ē is +50°F but my U_val is only 10°F — that means the model is terrible, right?"

**No.** For one-sided underprediction, this is a pass. The model is very conservative (it predicts much hotter than reality), which is safe. A large positive Ē means your design has extra margin. Whether you *want* that much conservatism is an engineering judgment call — but the V&V 20 validation criterion is satisfied.

### Model Form Uncertainty — "What's Left Over"

If the observed scatter (s_E) is bigger than your catalogued uncertainties (u_val), the excess is attributed to unmodelled physics:

```
u_model = √(s_E² - u_val²)
```

This isn't something you can fix with better grids or better BCs — it requires improved physics modeling (e.g., better turbulence model, adding radiation, including conjugate heat transfer).

### Per-Location Validation — When Pooling Doesn't Make Sense

#### The Problem with Pooling

Suppose you have 50 thermocouples (T/Cs), each measured at 12 operating conditions — that's 600 data points. The default approach in many V&V studies is to **pool** all 600 comparison errors together: compute one Ē, one s_E, one U_val, and get one pass/fail verdict.

But pooling hides spatial problems. Consider this scenario:

- TC-001 through TC-025: Ē ≈ +5°F (model overpredicts — conservative, great)
- TC-026 through TC-050: Ē ≈ -15°F (model underpredicts — potentially unsafe)
- Pooled Ē ≈ -5°F (looks OK — the good locations mask the bad ones)

The pooled analysis says "validated" because the average across all locations is within the uncertainty band. But half your T/Cs are showing significant underprediction that's hidden by the other half's overprediction. That's dangerous.

#### The Per-Location Approach

Instead of pooling, validate **at each T/C location independently**:

1. For each T/C, compute Ē_i (the mean comparison error across operating conditions at that location)
2. Apply the validation criterion at each location: Ē_i ≥ -U_val (one-sided) or |Ē_i| ≤ U_val (two-sided)
3. Report the result as: **"Validated at 47 of 50 locations (94%)"**

This gives you:
- **Spatial information:** Which T/Cs pass and which fail? Where is the model weak?
- **Actionable results:** "The model underpredicts at TC-033, TC-041, and TC-048 — all in the recirculation zone" is far more useful than "the pooled result is 0.3°F from the threshold"
- **No pooling justification needed:** Pooling requires proving that all locations have the same mean and variance (Levene's test, etc.). Per-location analysis avoids this entirely.

#### When to Pool vs. When Not To

| Scenario | Recommendation |
|----------|---------------|
| 50 T/Cs × 12 conditions, different physics at different locations | **Per-location** (do not pool) |
| Single T/C × 30 repeated tests at identical conditions | **Pooling is fine** (same location, same physics) |
| 5 nearly-identical T/Cs in a uniform region × 20 conditions | **Pooling is defensible** (verify with Levene's test) |
| Comparing two CFD codes at the same T/C locations | **Per-location** (you want spatial comparison) |

**Rule of thumb:** If the physics changes across locations (laminar vs. turbulent, attached vs. separated, hot-side vs. cold-side), don't pool. Use per-location validation.

#### Global vs. Per-Location Uncertainty Sources

When doing per-location validation, some uncertainty sources apply the same way at every location (they're **global**), while others can vary by location (they're **per-location**):

| Source | Scope | Why |
|--------|-------|-----|
| Grid convergence (GCI) | **Global** | You ran one grid study — same numerical error estimate everywhere |
| Iteration convergence | **Global** | Same residual targets apply to the whole model |
| Inlet BC uncertainty | **Global** | Same boundary condition affects all locations equally |
| Thermocouple accuracy | **Global** | All T/Cs are the same make/model with the same spec |
| Sensor placement accuracy | **Per-location** | A T/C in a steep gradient region has more placement uncertainty than one in a uniform region |
| Local mesh quality effects | **Per-location** | Some regions have better mesh quality than others |

In the source editor (Tab 2), you can set each source's **Scope** to "Global" or "Per-location" under the Advanced: Uncertainty Classification section. Global sources contribute identically to every location's U_val. Per-location sources are included separately, allowing future support for location-specific values.

The per-location U_val at each location is:

```
U_val,i = k × √(u_global² + u_per_loc²)
```

where `u_global²` is the RSS of all global-scope sources, and `u_per_loc²` is the RSS of all per-location-scope sources.

#### Reading the Per-Location Table

The per-location breakdown table (on Tab 1, in the collapsible "Per-Location Breakdown" section) shows:

| Column | What It Shows |
|--------|---------------|
| Location | The T/C or location name (from your comparison data) |
| Mean Ē | Average comparison error at this location across all conditions |
| Std Dev | Standard deviation of comparison errors at this location |
| n | Number of conditions (data points) at this location |
| U_val | Expanded uncertainty at this location |
| Verdict | PASS or FAIL based on the selected criterion |
| Flag | Outlier flags (e.g., if this location has unusually high scatter) |

A summary line below the table shows the overall result: **"Validated at 47/50 locations (94%)"**.

#### Budget Covariance Check — The "One Number" Across All Locations

Below the per-location table, the tool automatically computes a **budget covariance multivariate check** (V&V 20.1). This produces a single number that answers: *"Is the overall pattern of bias across all locations consistent with the uncertainty budget?"*

**Why you need this in addition to per-location checks:**

When you run 50 independent PASS/FAIL checks, you're implicitly treating shared uncertainties as 50 independent pieces of evidence. If DAQ calibration bias is ±0.5°F, it shifts all 50 T/Cs the same direction simultaneously — but 50 independent checks don't know this. The budget covariance check properly accounts for these shared uncertainties by building a covariance matrix where:

- **Global sources** (grid convergence, DAQ calibration, inlet BC) create off-diagonal terms — they shift all locations together
- **Per-location sources** (sensor placement) create diagonal-only terms — they're independent between locations

**The number: d²/m**

The metric produces a ratio called **d²/m** (Mahalanobis distance squared, normalized by the number of locations):

| d²/m | What it means |
|------|---------------|
| **< 0.5** | The model errors are well within the uncertainty budget. The budget comfortably explains the bias pattern. |
| **0.5 – 1.0** | The model errors are consistent with the budget. Normal result. |
| **1.0 – 2.0** | The model errors are somewhat larger than the budget predicts. Some locations may have bias beyond what the current sources explain. |
| **> 2.0** | The model errors significantly exceed the budget. The budget does not explain the observed bias pattern. Investigate. |

The associated **p-value** is the probability of seeing this extreme a bias pattern if the model were perfect (all bias explained by the budget). p ≥ 0.05 → PASS; p < 0.05 → FAIL.

**Concrete example:**

Suppose all 50 T/Cs show Ē ≈ -3°F and your U_val = 5°F at each location.

- **50 independent scalar checks:** All 50 pass (-3 ≥ -5). You report "50/50 validated." Looks great.
- **Budget covariance check:** "Wait — all 50 locations are biased in the *same direction* by the *same amount*. The probability of this happening by random chance is essentially zero. There's a systematic -3°F bias that the shared uncertainties can't explain." **FAILS.**

The per-location checks missed this because each one individually can explain -3°F, but the *pattern* of all 50 being -3°F is too coordinated to be random noise. The budget covariance check catches it because it looks at the joint probability, not 50 independent probabilities.

**No user action needed** — this auto-computes whenever per-location validation runs. The result appears below the per-location table on Tab 1, in the findings on Tab 4, and in the HTML report.

---

## 13. Monte Carlo vs Latin Hypercube — What's the Difference?

### Standard Monte Carlo: "Throwing Darts Randomly"

Imagine you need to figure out the shape of a dartboard by throwing darts at it in the dark. You throw randomly — some areas get hit a lot, other areas (especially the edges) get missed. You might need 100,000 darts to get a clear picture.

**How it works technically:**
1. For each uncertainty source, draw a random number from its distribution
2. Add them all up — that's one "trial"
3. Repeat 100,000 times
4. The collection of 100,000 totals IS your combined distribution

### Latin Hypercube Sampling (LHS): "Organized Dart Throwing"

Now imagine you divide the dartboard into 100,000 equal slices (like pizza slices of equal probability) and throw exactly ONE dart into each slice. You're guaranteed to hit every part of the board — no gaps, no clusters.

**How it works technically:**
1. Divide the probability range [0%, 100%] into N equal intervals
2. Place exactly one random sample in each interval
3. Shuffle them (so source A's 47th sample isn't always paired with source B's 47th sample)
4. Convert from probability back to physical values using the inverse CDF

### Why LHS Is Better (Usually)

| Property | Monte Carlo (Random) | Latin Hypercube (LHS) |
|---|---|---|
| **Coverage of tails** | Sparse — random gaps in extreme values | Guaranteed — every probability band gets a sample |
| **Convergence speed** | ~1/√N (slow) | Typically faster than random MC for the same N |
| **Samples needed for similar percentile stability** | N | Often lower than random MC (problem-dependent) |
| **Results with 10,000 trials** | Good for mean, noisy for percentiles | Excellent for both mean and percentiles |
| **Reproducibility** | Varies significantly between runs | Much more stable between runs |

### When to Use Which

| Situation | Recommendation |
|---|---|
| General uncertainty propagation | **LHS** — faster convergence, better tail coverage |
| Quick sanity check | Either works at 100,000 trials |
| Need to match legacy results exactly | **MC (Random)** — matches older analyses |
| Comparing to textbook examples | **MC (Random)** — what most textbooks describe |
| Certification application | **LHS** — efficient stratified sampling (McKay et al. 1979) within the JCGM 101 Monte Carlo framework |

### Standards Recognition

Both methods are defensible in this workflow:
- **JCGM 101:2008** provides the Monte Carlo propagation framework used by the tool
- **ASME V&V 20, Section 4.4** recognizes Monte Carlo propagation methods in general
- **McKay, Beckman & Conover (1979)** provides the original Latin Hypercube sampling method

---

## 14. Distributions — Which One Should I Pick?

### The Short Answer

**When in doubt, use Normal.** It's the most common assumption in uncertainty analysis, and the Central Limit Theorem means that even if individual sources aren't normal, their combination tends toward normal.

### The Longer Answer

| Distribution | Shape | When to Use It | Typical Source |
|---|---|---|---|
| **Normal** | Classic bell curve | Most general-purpose uncertainty sources | Test data statistics, calibration labs, repeated measurements |
| **Uniform** | Flat/rectangular | You know the limits but nothing else — equal probability everywhere | Manufacturer specs with "±X max", resolution limits |
| **Triangular** | Tent shape — peaks at center | You know the limits AND the most likely value is in the middle | Engineering judgment with a best estimate and bounds |
| **Lognormal** | Skewed right, always positive | Naturally positive quantities with right skew | Manufacturing tolerances, material properties, decay processes |
| **Lognormal (σ=0.5)** | Moderately skewed right | Positive quantities, less extreme skew | Moderate manufacturing variability |
| **Logistic** | Bell-shaped, heavier tails | Like normal but with more outliers | Growth/decay processes, contaminated measurements |
| **Laplace** | Peaked center, heavy tails | Sharp peak with frequent outliers | Noise processes, financial data, some sensor errors |
| **Student-t (df=5)** | Bell with very heavy tails | Small samples, expect occasional large errors | Limited calibration data, early-stage testing |
| **Student-t (df=10)** | Bell with moderately heavy tails | Moderate samples, some outlier concern | Moderate-sized test campaigns |
| **Exponential** | One-sided, decays to zero | Waiting times, gap sizes, always positive | Time to failure, positive-only error processes |
| **Weibull** | Flexible shape, always positive | Strength, lifetime, wind speed data | Material testing, reliability analysis |
| **Custom/Empirical** | Whatever your data looks like | You have actual data and don't want to assume any shape | Use the "Tabular Data" input type + Auto-Fit |

### Decision Flowchart

```
Do you have actual data for this source?
├── YES → Use "Tabular Data" input + Auto-Fit Distribution
│         (let the tool fit it for you)
│
└── NO → What do you know about it?
    ├── "I have a ±X spec with no other info"
    │   └── Use UNIFORM (conservative — equal probability everywhere)
    │
    ├── "I have a ±X spec and the center is most likely"
    │   └── Use TRIANGULAR (less conservative — peaks at center)
    │
    ├── "I have a standard deviation from a cal lab or test report"
    │   └── Use NORMAL (standard assumption)
    │
    ├── "The quantity is always positive and tends to be skewed"
    │   └── Use LOGNORMAL
    │
    └── "I honestly don't know"
        └── Use NORMAL (safest general assumption)
```

### Impact on Results

The distribution choice mainly affects the Monte Carlo results (since RSS assumes normal regardless). If all your sources are normal, the MC and RSS results will agree closely. If you have uniform or triangular sources, the MC bounds will typically be *tighter* than RSS (because those distributions don't have infinite tails like the normal distribution).

---

## 15. Reading the Guidance Panels

Throughout the tool, color-coded guidance panels give you real-time feedback. Here's how to read them:

### Color Coding

| Color | Icon | Meaning | Action Needed |
|---|---|---|---|
| **Green** | ✔ | Everything looks good | No action needed |
| **Yellow** | ⚠ | Something to be aware of | Review and document your rationale |
| **Red** | ✖ | A significant concern | Take action — add data, change approach, or justify in your report |

### Common Panel Messages and What to Do

| Panel | Color | Message (Summarized) | What to Do |
|---|---|---|---|
| Dominant Source | Red | "Source X contributes >80% of u_val²" | Focus effort on reducing this source; all others are noise |
| Dominant Source | Yellow | "Source X contributes >50% of u_val²" | Be aware — refining other sources won't help much |
| DOF Check | Red | "ν_eff < 5" | k=2 is dangerously non-conservative; use tolerance factor method |
| DOF Check | Yellow | "ν_eff = 5–30" | k=2 is slightly non-conservative; consider W-S or tolerance method |
| Model Form | Red | "s_E >> u_val" | You're missing significant physics in your model |
| Model Form | Yellow | "u_val > s_E" | You may be over-estimating uncertainties (could be okay) |
| Validation | Green | "VALIDATED" | The bias is within the noise — model is acceptable |
| Validation | Red | "NOT VALIDATED" | Systematic bias detected — investigate your model |
| MC Convergence | Red | "Relative SE > 2%" | Increase trial count (try 500K or 1M) |
| MC vs RSS | Yellow | "MC wider than RSS" | Non-normal effects detected; MC bounds are more reliable |

---

## 16. Project Files and Reports

### Saving a Project (Ctrl+S)

Creates a dated folder containing three files:

```
MyProject_2025-02-17/
├── MyProject.json          ← Complete project data (can be reloaded)
├── MyProject_AuditLog.txt  ← Plain-text audit trail
└── MyProject_Report.html   ← Full HTML report (if analysis was run)
```

The **JSON file** contains everything: comparison data, all uncertainty sources, all settings, computed results (except the raw MC sample arrays), and the full audit log. You can reload this file later to continue your analysis.

### Loading a Project (Ctrl+O)

Opens a previously saved JSON project file. All comparison data, uncertainty sources, settings, audit entries, and project metadata are restored. RSS results are automatically recomputed from the restored inputs. Monte Carlo simulation results must be rerun manually if needed (click "Run Monte Carlo" on the Results — Monte Carlo tab). If the project was saved with an older version of the tool that didn't have some newer settings (like the LHS sampling method), the defaults are applied automatically — backward compatibility is built in.

### The HTML Report (Ctrl+H)

A comprehensive, self-contained HTML document suitable for printing or attaching to a certification package. It includes:

1. **Header:** Company/project info, proprietary notice, export control notice
2. **Table of Contents:** Clickable links to each section
3. **Analysis Configuration:** All settings used (coverage, confidence, k-method, MC method, etc.)
4. **Comparison Data Summary:** Statistics and plots of the comparison errors
5. **Uncertainty Budget:** Full budget table with category subtotals and variance breakdown
6. **RSS Results:** Complete RSS analysis with validation verdict and plots
7. **Monte Carlo Results:** MC/LHS analysis with distribution plots (if MC was run)
8. **Comparison Roll-Up:** Side-by-side comparison table with certification statement
9. **Assumptions & Engineering Judgments:** Auto-populated from the audit log
10. **Audit Trail:** Complete timestamped record of every action and computation

The report uses a **light/print-friendly theme** (white background) even though the tool uses a dark theme. All charts are embedded as base64 images, so the HTML file is fully self-contained — no external files needed.

### The Audit Log

The tool automatically records every significant action:

| Action | What Gets Logged |
|---|---|
| Data entry | When data is imported or pasted |
| Settings changes | Every setting modification |
| Computations | Every step of the RSS and MC calculations |
| Assumptions | Any "Assumed 1σ (unverified)" selections |
| Warnings | Validation failures, convergence concerns |

This creates a defensible record for certification review. An auditor can trace exactly how every number was produced.

---

## 17. Certification and Regulatory Use

### For Aerospace Thermal Certification

The typical requirement is **95/95 one-sided** — 95% coverage at 95% confidence. This means:

> *"We are 95% confident that the true thermal prediction error will not exceed X degrees on the hot side."*

**Recommended settings:**
- Coverage: 95%
- Confidence: 95%
- One-sided: Yes
- k-method: One-Sided Tolerance Factor (most rigorous) or V&V 20 Default (simpler, requires DOF > 30)
- MC sampling: Latin Hypercube (LHS) — more efficient convergence
- Bootstrap: Enabled — provides uncertainty on the MC bounds

### What to Put in Your Report

The tool's auto-generated certification statement (Tab 6) provides ready-to-use language. A typical statement looks like:

> *"Validation uncertainty was computed per ASME V&V 20-2009 using 12 identified uncertainty sources aggregated by RSS (u_val = 2.83°F, k = 2.00, U_val = 5.66°F). Monte Carlo propagation (Latin Hypercube stratified sampling, 100,000 trials; JCGM 101 framework; McKay et al. 1979) yields a 95% one-sided prediction interval of [+0.25°F, +9.79°F] with bootstrap 95% CI envelope [+0.09°F, +9.93°F]. The mean comparison error Ē = +5.00°F gives a validation ratio |Ē|/U_val = 0.88 (< 1.0). Model is VALIDATED at the 95/95 one-sided level."*

### Documentation Package

For a certification submission, include:
1. The **HTML report** — comprehensive documentation
2. The **project JSON file** — for reproducibility
3. The **audit log** — for traceability
4. The **comparison raw data** — in your preferred format (Excel/CSV)
5. Any supporting documents (grid convergence studies, cal certs, test reports)

---

## 18. Standards Reference Summary

| Standard | Full Title | What This Tool Uses It For |
|---|---|---|
| **ASME V&V 20-2009 (R2021)** | Standard for Verification and Validation in Computational Fluid Dynamics and Heat Transfer | Overall framework: E = S - D, three uncertainty categories, RSS combination, |Ē| ≤ U_val validation criterion, k = 2 default |
| **JCGM 100:2008 (GUM)** | Guide to the Expression of Uncertainty in Measurement | RSS combination rules, Welch-Satterthwaite effective DOF, coverage factors from Student-t, Type A/B evaluation methods |
| **JCGM 101:2008 (GUM Supplement 1)** | Propagation of Distributions Using a Monte Carlo Method | Monte Carlo propagation framework, convergence criteria (§7.9), bootstrap confidence intervals |
| **ASME PTC 19.1-2018** | Test Uncertainty | Sample size requirements, distribution-free tolerance intervals, sigma-basis conversions |
| **AIAA G-077-1998** | Guide for the Verification and Validation of Computational Fluid Dynamics Simulations | V&V reporting best practices, model form uncertainty characterization |
| **Krishnamoorthy & Mathew (2009)** | Statistical Tolerance Regions | One-sided and two-sided tolerance factor formulas using non-central t-distribution |
| **McKay, Beckman & Conover (1979)** | A Comparison of Three Methods for Selecting Values of Input Variables... | Original Latin Hypercube Sampling paper |
---

## 19. Glossary

| Term | Plain English Definition |
|---|---|
| **CFD** | Computational Fluid Dynamics — computer simulation of fluid flow and heat transfer |
| **V&V** | Verification & Validation — the process of proving your simulation is correct (V) and accurate (V) |
| **Comparison Error (E)** | CFD result minus test measurement: E = S - D |
| **Ē (E-bar)** | Mean comparison error — the average bias across all data points |
| **s_E** | Standard deviation of comparison errors — the scatter around the mean |
| **u_val** | Combined standard uncertainty from all known sources (RSS combined) |
| **U_val** | Expanded uncertainty — u_val multiplied by coverage factor k |
| **u_num** | Numerical uncertainty — from grid, time step, solver convergence |
| **u_input** | Input/boundary condition uncertainty — from BCs, material properties, geometry |
| **u_D** | Experimental (data) uncertainty — from sensors, DAQ, test conditions |
| **u_model** | Model form uncertainty — the part of error not explained by known sources |
| **k (coverage factor)** | Multiplier that converts 1σ uncertainty to an expanded uncertainty at the desired coverage level |
| **ν_eff (nu-eff)** | Effective degrees of freedom — a measure of how much data supports your uncertainty estimate |
| **DOF** | Degrees of Freedom — roughly, the number of independent data points minus 1 |
| **RSS** | Root Sum of Squares — method for combining independent uncertainties |
| **Monte Carlo** | A method of computing results by running many random simulations |
| **LHS** | Latin Hypercube Sampling — a stratified version of Monte Carlo that's more efficient |
| **Coverage** | The percentage of the distribution captured by the uncertainty interval (e.g., 95%) |
| **Confidence** | How sure you are that the interval actually achieves the stated coverage |
| **One-sided** | An interval that only bounds one direction (e.g., maximum overprediction) |
| **Two-sided** | An interval that bounds both directions symmetrically |
| **Bootstrap** | A resampling technique to estimate how uncertain a statistic is |
| **PDF** | Probability Density Function — the shape of the distribution curve |
| **CDF** | Cumulative Distribution Function — the running total of probability from left to right |
| **Percentile (Pxx)** | The value below which xx% of the data falls (e.g., P95 = value below which 95% falls) |
| **Shapiro-Wilk test** | A statistical test for whether data follows a normal distribution (p > 0.05 = probably normal) |
| **Bootstrap GOF** | Parametric bootstrap goodness-of-fit p-value (primary when bootstrap mode is active) |
| **KS test** | Kolmogorov-Smirnov goodness-of-fit diagnostic (secondary context) |
| **Type A evaluation** | Uncertainty estimated from actual measured data (statistical analysis) |
| **Type B evaluation** | Uncertainty estimated from other information (specs, handbooks, engineering judgment) |
| **ppf** | Percent Point Function — the inverse of the CDF (given a probability, returns the value) |
| **GUM** | Guide to the Expression of Uncertainty in Measurement (JCGM 100:2008) |
| **Per-location validation** | Validating at each measurement location independently rather than pooling all data together |
| **Pooling** | Combining comparison data from multiple locations into a single dataset for analysis |
| **Source scope** | Whether an uncertainty source applies globally (same at all locations) or per-location (can vary by T/C) |
| **One-sided criterion** | Validation check where only underprediction matters: Ē ≥ -U_val. Positive Ē (overprediction) always passes. |
| **Two-sided criterion** | Validation check where both directions matter equally: \|Ē\| ≤ U_val |
| **Conservative (overprediction)** | When the model predicts higher than reality (e.g., hotter temperature). Safe for certification because design margins have extra room. |
| **Budget covariance check** | V&V 20.1 multivariate metric that tests the overall bias pattern against the uncertainty budget, accounting for shared (global) vs. independent (per-location) sources |
| **d²/m** | Normalized Mahalanobis distance — the budget covariance metric. ≈1.0 means normal, <1.0 means better than budget, >1.0 means worse than budget |
| **Mahalanobis distance** | A distance measure that accounts for correlations. Used in the budget covariance check to measure how far the bias pattern is from zero in uncertainty-normalized units |
| **Compound symmetry** | The covariance structure used in the budget covariance check: all location pairs have the same correlation (driven by shared global uncertainties) |

---

## 20. Frequently Asked Questions

### Q: My RSS says VALIDATED but Monte Carlo says NOT VALIDATED (or vice versa). Which one do I trust?

**A:** The Monte Carlo result is more general because it doesn't assume a normal combined distribution. If they disagree, it usually means one of your sources has a non-normal distribution (e.g., Uniform, heavy-tailed) that affects the tails differently than a normal assumption would predict. **Use the Monte Carlo result** and document why.

### Q: How many Monte Carlo trials do I need?

**A:** 100,000 is fine for most applications. The convergence check (green/yellow/red panel) will tell you if you need more. If you're using LHS, even 10,000 may be enough. If the convergence panel is green, you're good.

### Q: When should I use k=2 vs. the tolerance factor method?

**A:** Use k=2 when you have lots of data (effective DOF > 30) and the certifying authority accepts it. Use the tolerance factor method when you have limited data (DOF < 30), need a more rigorous basis, or the certification requires accounting for confidence as well as coverage. When in doubt, use the tolerance factor — it automatically gives you k ≈ 2 when you have plenty of data.

### Q: My model form assessment shows s_E >> u_val. What do I do?

**A:** This means your observed scatter is much larger than what your catalogued uncertainties predict. Options:
1. **Look for missing uncertainty sources** — did you forget to include something?
2. **Improve your model** — the excess scatter may indicate physics your model doesn't capture (e.g., radiation, conjugate heat transfer, transition)
3. **Use the s_E-based bound** — this is more conservative but includes the model form effect
4. **Document u_model** — the tool estimates the model form uncertainty for you

### Q: Can I use this for things other than CFD?

**A:** Yes! The mathematical framework (RSS, Monte Carlo, coverage factors) is completely general. Any situation where you need to combine independent uncertainties and compare predictions to measurements can use this tool. The terminology is CFD-oriented, but the math doesn't care about the application.

### Q: What does "Assumed 1σ (unverified)" mean and should I worry about it?

**A:** It means you entered a sigma value based on engineering judgment rather than actual data. The tool flags this in the audit log so that reviewers know it's an assumption, not a measurement. This is perfectly acceptable in uncertainty analysis (it's a "Type B" evaluation per the GUM), but you should document your rationale.

### Q: I loaded an old project and some settings are missing. Is that a problem?

**A:** No. The tool automatically applies sensible defaults for any settings that didn't exist in older versions. For example, old projects default to "Monte Carlo (Random)" for sampling method. Check the Analysis Settings tab to verify everything looks right.

### Q: Why is one source showing as 80%+ of the total variance?

**A:** This is common and not necessarily a problem — it just means that one source dominates. This is actually useful information because it tells you where to focus your effort. Reducing a source that contributes 80% of the variance has a much bigger impact than reducing one that contributes 2%.

### Q: The tool says "NOT VALIDATED" — does that mean my CFD is useless?

**A:** No. "Not validated" means the model bias is larger than the known uncertainties can explain. It doesn't mean the model is useless — it means:
1. There may be additional uncertainty sources you haven't accounted for
2. The model may have a correctable systematic bias (which you could calibrate out)
3. The model may need physics improvements for this particular quantity

Many perfectly useful engineering models are "not validated" by the strict V&V 20 criterion. What matters is that you understand the limitations and document them.

### Q: My Ē is +50°F but U_val is only 10°F — is that validated?

**A:** **Yes, if you're using one-sided underprediction mode** (the default). The one-sided criterion is `Ē ≥ -U_val`. Your Ē = +50°F is far to the right of -10°F, so it passes easily. The model is very conservative — it overpredicts temperature by 50°F. That's safe for certification (the hardware will never get as hot as you predicted).

If you're using **two-sided mode**, then no — `|Ē| = 50 > U_val = 10`, so it would fail. Two-sided mode penalizes overprediction just as much as underprediction.

Whether you *want* 50°F of conservatism is an engineering judgment call (it might mean your design is heavier than necessary), but the V&V 20 validation criterion is satisfied in one-sided mode.

### Q: Should I pool my thermocouple locations together?

**A:** **Probably not.** Pooling is only justified when all locations have similar mean errors AND similar scatter — that is, the physics is essentially the same everywhere. In most real CFD validations, different T/C locations see different flow regimes (laminar vs. turbulent, attached vs. separated, near walls vs. freestream), and the model performance varies across those regions.

Per-location validation gives you spatial information: "The model underpredicts at 3 T/Cs in the recirculation zone" is far more useful than "the pooled result is borderline." Start with per-location validation (pooling OFF) and only enable pooling if you have a specific justification.

### Q: What's the difference between global and per-location uncertainty sources?

**A:** **Global sources** have the same value at every measurement location. Grid convergence uncertainty, iteration convergence uncertainty, and inlet boundary condition uncertainty are all global — they come from one study or one setting that affects the entire model equally.

**Per-location sources** can vary by measurement location. The classic example is sensor placement accuracy: a thermocouple in a steep temperature gradient has much more placement uncertainty than one in a uniform region. If you move the T/C by 1mm in a gradient of 50°F/mm, that's ±50°F of uncertainty. The same 1mm shift in a flat region might be ±0.1°F.

In the source editor, set the **Scope** field (under Advanced: Uncertainty Classification) to "Global" or "Per-location". Both contribute to each location's U_val via RSS, but the tool tracks them separately for reporting.

### Q: I switched from two-sided to one-sided and now more locations pass. Did I just game the system?

**A:** **No — you chose the physically appropriate criterion.** If your application genuinely only cares about underprediction (e.g., maximum temperature for material limits), then one-sided is the correct choice. It's not "easier" — it's *different*. It says "I accept the risk of overprediction" (because overprediction is conservative for my application). You must document and justify the choice of one-sided vs. two-sided in your V&V report. The tool includes the criterion type in all findings and the HTML report.

### Q: What does d²/m mean and when should I worry?

**A:** d²/m is the budget covariance metric — a single number summarizing whether the overall pattern of bias across all your T/C locations is consistent with the uncertainty budget.

Think of it as a "suspicion score":
- **d²/m ≈ 0.3:** The model errors are well within the budget. Nothing to worry about.
- **d²/m ≈ 0.9:** Normal. The bias pattern is about what you'd expect given the uncertainty sources.
- **d²/m ≈ 1.5:** Getting elevated. The model has more bias than the budget predicts, but it might not be statistically significant yet. Check the p-value.
- **d²/m ≈ 3.0:** The bias pattern is much larger than the budget can explain. Either you're missing uncertainty sources, or there's a systematic model issue the budget doesn't capture.

The associated **p-value** gives the statistical significance: p ≥ 0.05 means PASS, p < 0.05 means FAIL. The d²/m ratio tells you *how far off* you are; the p-value tells you *whether it's statistically significant*.

**Key insight:** This metric can fail even when all 50 per-location checks pass individually. That happens when the bias is coordinated across locations (e.g., all locations underpredicting by the same amount), which is exactly the signature of a shared systematic error that the per-location checks can't detect.

---

*VVUQ Uncertainty Aggregator v1.2.0 — Built for engineers who need defensible uncertainty numbers, not statistics PhDs.*

*Standards: ASME V&V 20-2009 (R2021), JCGM 100:2008, JCGM 101:2008, ASME PTC 19.1-2018, AIAA G-077-1998*
