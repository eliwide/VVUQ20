# GCI Calculator v1.4.0 — User Guide & Technical Reference

**Grid Convergence Index Tool per Celik et al. (2008), Roache (1998), Xing & Stern (2010), and Eca & Hoekstra (2014)**

*Written for engineers — not statisticians.*

---

## Revision Update (v1.4.0)

- Added **Decision Consequence** in the Project Info bar (`Low`, `Medium`, `High`).
- HTML report now includes:
  - Decision card for direct carry-over to the Aggregator,
  - Credibility framing checklist,
  - VVUQ terminology panel,
  - Conformity assessment template wording.
- Fluent `.prof` import coverage is aligned to documented profile types:
  - `point`, `line`, `mesh`, `radial`, `axial`,
  - and multi-surface files are merged into one analysis set.

### Quick Example (Carry to the Aggregator)

1. Run a 3-grid (or higher) study for each required quantity.
2. Confirm convergence type is not divergent.
3. Copy **u_num (1σ)** from the carry-over box / report decision card.
4. In the Aggregator, enter that value as a `Numerical (u_num)` source with distribution `Normal` and method `RSS` unless the report explicitly states otherwise.

### 60-Second Operator Decision Flow

1. **If convergence is Monotonic or Grid-Independent:** carry the `u_num` shown in the carry-over box.
2. **If convergence is Oscillatory:** carry `u_num` only as a conservative value and keep the warning note in your report.
3. **If convergence is Divergent:** stop and fix the study first (do not carry `u_num`), unless your process explicitly allows a documented conservative bounding estimate.
4. **If production grid is not the finest:** use production-grid `u_num`, not fine-grid `u_num`.

---

## Table of Contents

1. [What Is This Tool and Why Do I Need It?](#1-what-is-this-tool-and-why-do-i-need-it)
2. [What Is the Grid Convergence Index?](#2-what-is-the-grid-convergence-index)
3. [How Many Grids Do I Need?](#3-how-many-grids-do-i-need)
4. [Is Richardson Extrapolation Required?](#4-is-richardson-extrapolation-required)
5. [Getting Started — Application Layout](#5-getting-started--application-layout)
6. [Step-by-Step: Running a GCI Study](#6-step-by-step-running-a-gci-study)
7. [Understanding the Results](#7-understanding-the-results)
8. [Production Grid Selection](#8-production-grid-selection)
9. [Saving and Loading Studies](#9-saving-and-loading-studies)
10. [Unit Labels for Quantities](#10-unit-labels-for-quantities)
11. [Reading the Guidance Panels](#11-reading-the-guidance-panels)
12. [Convergence Types Explained](#12-convergence-types-explained)
13. [Spatial / Field GCI Analysis](#13-spatial--field-gci-analysis)
14. [The Three-Grid Procedure (Celik et al. 2008)](#14-the-three-grid-procedure-celik-et-al-2008)
15. [Safety Factor Recommendations](#15-safety-factor-recommendations)
16. [Converting GCI to u_num for V&V 20](#16-converting-gci-to-u_num-for-vv-20)
17. [Multiple Quantities of Interest](#17-multiple-quantities-of-interest)
18. [Convergence Plot](#18-convergence-plot)
19. [Report Statements](#19-report-statements)
20. [Alternative Uncertainty Methods (FS and LSR)](#20-alternative-uncertainty-methods-fs-and-lsr)
21. [Method Comparison](#21-method-comparison)
22. [Built-In Example Datasets](#22-built-in-example-datasets)
23. [Exporting Results](#23-exporting-results)
24. [How to Use u_num in the Uncertainty Aggregator](#24-how-to-use-u_num-in-the-uncertainty-aggregator)
25. [Tips for Getting a Good Grid Study](#25-tips-for-getting-a-good-grid-study)
26. [Key Formulas Reference](#26-key-formulas-reference)
27. [Standards References](#27-standards-references)
28. [Glossary](#28-glossary)
29. [Frequently Asked Questions](#29-frequently-asked-questions)

---

## 1. What Is This Tool and Why Do I Need It?

If you run CFD simulations, you need to know how much of your answer is real physics and how much is just the grid talking. The GCI Calculator answers that question.

**The problem:** Every CFD simulation uses a finite mesh. If you made the mesh infinitely fine, the answer would change — but by how much? That gap between your current answer and the "true" numerical answer is your **numerical uncertainty (u_num)**.

**Why it matters:** In the ASME V&V 20 framework for CFD validation uncertainty, u_num is often the single largest contributor to the total uncertainty budget. If you don't quantify it properly, your entire uncertainty analysis is undermined.

**What this tool does:** You feed in your CFD solution values from 2 or more grids of different fineness, and the tool tells you:

> *"Your fine-grid solution has a numerical uncertainty of ±X (conservative error bound). The estimated true solution (infinite grid) is Y."*

That u_num number goes directly into your V&V 20 uncertainty budget.

> **Common Mistakes to Avoid**
>
> Grid convergence studies are straightforward in concept but easy to get wrong in practice. Watch out for these pitfalls:
>
> - **Only running 2 grids instead of 3.** With 2 grids, you cannot compute the observed order of accuracy and must assume it. This forces a safety factor of 3.0 (instead of 1.25) and you cannot verify you are in the asymptotic range. Always budget for at least 3 grids.
> - **Confusing mesh count with mesh spacing (h).** The GCI formulas use the representative cell size h, not the total number of cells. If you double the cell count, h does not halve -- it changes by the cube root (in 3D). The tool computes h for you, but understand what it means before interpreting refinement ratios.
> - **Using non-integer refinement ratios without understanding the implications.** The standard procedure works best with consistent refinement ratios (e.g., 2x or 1.5x). Arbitrary ratios like 1.13 or 2.7 can produce unreliable observed-order estimates, especially if you are not deep in the asymptotic range.
> - **Not checking if the solution is in the asymptotic range.** If your observed order of accuracy is wildly different from your scheme's theoretical order (e.g., you get p = 0.5 for a second-order scheme), your grids may be too coarse for Richardson extrapolation to work. The guidance panels flag this, but many users ignore the warning.
> - **Applying GCI to time-dependent problems without converging each time step.** GCI assumes each grid level gives a converged solution at that resolution. If your transient simulation has not reached iterative convergence at each time step, the grid study results are meaningless. Converge each time step first, then compare across grids.

---

## 2. What Is the Grid Convergence Index?

The **Grid Convergence Index (GCI)** is a standardized method for estimating how much your CFD solution would change if you kept making the mesh finer and finer until it was infinitely fine. It was developed by Roache (1998) and formalized into a recommended procedure by Celik et al. (2008).

### The Basic Idea

Run your CFD case on multiple grids of different fineness. If the answer changes less and less as you refine the grid, you can mathematically estimate:

1. **What the answer would be on an infinitely fine grid** (Richardson extrapolation)
2. **How far your current grid's answer is from that** (the GCI)

Think of it like surveying — if you take measurements from 3 different distances, you can triangulate the exact position. Similarly, 3 grids let you triangulate the exact numerical answer.

### Why Not Just Use the Finest Grid?

Because "finest" doesn't mean "fine enough." Your 5-million-cell mesh might still have a 3°F numerical error that you'd never know about without doing this study. GCI gives you a number for that error.

---

## 3. How Many Grids Do I Need?

| Grids | What You Get | Safety Factor | Recommendation |
|-------|-------------|---------------|----------------|
| **2** | GCI with *assumed* order of accuracy. Cannot compute the observed order — you must assume it equals the theoretical order of your numerical scheme. | Fs = 3.0 (conservative) | **Minimum viable.** Quick estimate. Not recommended for certification or publication because you can't verify your assumption. |
| **3** | GCI with *computed* observed order of accuracy. Can verify the grids are in the asymptotic range. Can perform the full Celik et al. procedure. Also enables the **Factor of Safety (FS)** alternative method (see [Section 20](#20-alternative-uncertainty-methods-fs-and-lsr)). | Fs = 1.25 (standard) | **Standard procedure.** This is what Celik et al. (2008) recommends and what most journals require. Use this for any formal work. |
| **4+** | Same as 3-grid for the primary GCI (uses finest 3 grids), plus one extra triplet for cross-checking. Also enables the **Least Squares Root (LSR)** method (see [Section 20](#20-alternative-uncertainty-methods-fs-and-lsr)), which fits power-law models to all grids simultaneously. | Fs = 1.25 | **Good practice / best practice.** More grids give you consistency checks AND allow more sophisticated uncertainty methods. 4 grids is the minimum for LSR. |
| **5-6** | Multiple cross-checking triplets. Very high confidence in the result. All three methods (GCI, FS, LSR) available for comparison. | Fs = 1.25 | **Best practice for safety-critical work.** Multiple independent GCI estimates and cross-method comparison should converge on the same answer. |

### What if I Only Have 2 Grids?

You can still get a GCI, but with important caveats:

- You must **assume** the order of accuracy (typically p = 2 for second-order schemes)
- The safety factor jumps from 1.25 to **3.0** to compensate for not knowing the actual order
- Richardson extrapolation still works, but its accuracy depends entirely on whether your assumed order is correct
- You **cannot** check the asymptotic range

**Bottom line:** A 2-grid study is much better than no grid study, but budget for 3 grids if you can.

### What if I Have More Than 3 Grids?

Great — use them all. The tool accepts up to 6+ grids. The primary GCI is always computed from the finest 3 grids (because that's where the answer is most accurate), but the additional grids provide cross-checking:

- Each consecutive triplet (grids 1-2-3, grids 2-3-4, grids 3-4-5, etc.) produces its own observed order and convergence ratio
- These are shown in the results under "Additional grid triplets"
- If they all agree, your result is very robust
- If they disagree, the coarser grids may not be in the asymptotic range (which is expected)

---

## 4. Is Richardson Extrapolation Required?

| Scenario | Richardson Extrapolation? | Why |
|----------|--------------------------|-----|
| **3+ grids, monotonic convergence** | **Yes** | RE is the mathematical foundation of GCI. It estimates the "exact" (zero-spacing) solution by extrapolating the observed convergence trend to h = 0. |
| **2 grids** | **Used, but assumed** | RE requires knowing the order of accuracy p. With only 2 grids, p is underdetermined — you assume p = theoretical order and accept a larger safety factor. |
| **Oscillatory convergence (-1 < R < 0)** | **No** | When solutions oscillate between grid levels (with damping oscillations), RE is unreliable. The GCI is computed from the oscillation range with a conservative Fs = 3.0. |
| **Divergent convergence (R >= 1 or R <= -1)** | **No** | If the solution diverges with refinement, or oscillations are growing (R <= -1), GCI is invalid entirely. Fix your simulation first. |
| **Grid-independent** | **Not needed** | If all grids give the same answer, numerical uncertainty is zero. |

---

## 5. Getting Started — Application Layout

### How to Run

```
python gci_calculator.py
```

### Main Window

At the top of the window is a collapsible **Project Info** bar (click "▶ Project Info" to expand). This lets you record:

| Field | Purpose |
|-------|---------|
| **Program/Project** | Project name for traceability |
| **Analyst** | Who performed the analysis |
| **Date** | Analysis date |
| **Notes** | Free-text notes (boundary conditions, mesh strategy, etc.) |

These fields are saved with the `.gci` project file and included in HTML reports.

Below the project bar, the application has **three tabs**:

| Tab | Icon | Purpose |
|-----|------|---------|
| **GCI Calculator** | 📊 | Single-point GCI calculation — input data, compute, see results |
| **Spatial GCI** | 🌍 | Field/spatial GCI — point-by-point analysis over surface maps |
| **Reference** | 📖 | Built-in documentation covering the entire GCI procedure |

### Calculator Tab Layout

The Calculator tab is split into two panels:

**Left panel — Input:**
- Grid Study Setup (number of grids, dimensions, theoretical order, safety factor, production grid)
- Grid Data Table (cell counts and solution values)
- Quantity management buttons (+ Add Quantity, - Remove Last, Paste from Clipboard)
- Quantity Properties (unit labels per quantity)
- **Compute GCI** button
- Three guidance panels (Convergence, Order, Asymptotic Range)

**Right panel — Results (sub-tabs):**
The right panel is organized into four sub-tabs:
- **Results** — Full results text (monospace, copy-ready), including Method Comparison (new in v1.3)
- **Summary Table** — GCI Results Summary table (color-coded)
- **Convergence Plot** — Convergence plot with Richardson extrapolation and GCI band
- **Report Statements** *(new in v1.3)* — Copy-pasteable regulatory paragraphs for V&V reports (see [Section 19](#19-report-statements))

### Resizable and Auto-Width Table Columns

All tables in the application (grid data, results summary, per-grid uncertainty, spatial statistics) now have **resizable columns**. Drag any column header border left or right to adjust the column width. This is useful when quantity names or values are long and the default column width truncates them.

The grid data table also features **auto-width columns** — column headers automatically resize to fit long quantity names (e.g., "Area-Weighted Average of Static Temperature") with word wrapping enabled. Column headers update dynamically when you change the unit selection for a quantity.

### Menu Bar

**File Menu:**

| Menu Item | Shortcut | Action |
|-----------|----------|--------|
| New Study | Ctrl+N | Clear all data and start fresh |
| Open Study... | Ctrl+O | Load a previously saved .gci project file |
| Save Study | Ctrl+S | Save the current study to a .gci project file (overwrites if previously saved) |
| Save Study As... | Ctrl+Shift+S | Save to a new .gci project file |
| Export Results to Clipboard | Ctrl+Shift+C | Copy the full results text |
| Export Results to File | Ctrl+E | Save results to a text file |
| Export HTML Report... | Ctrl+H | Generate and export a self-contained HTML report |
| Exit | Alt+F4 | Close the application |

> **Note:** Example datasets are loaded from the top-level **Examples** menu (not the File menu). See [Section 22](#22-built-in-example-datasets) for details.

The window title shows the current project file name (e.g., "GCI Calculator — my_study.gci"). An unsaved new study shows "GCI Calculator" with no file name. See [Section 9](#9-saving-and-loading-studies) for details.

---

## 6. Step-by-Step: Running a GCI Study

### Step 1: Set Up the Grid Study

| Setting | What to Enter | Guidance |
|---------|---------------|----------|
| **Number of grids** | 2–6 (default: 3) | Use 3 unless you have a good reason not to |
| **Dimensions** | 2D or 3D | Match your CFD problem — affects how cell count converts to representative spacing |
| **Theoretical order** | 1.0–4.0 (default: 2.0) | The formal order of your numerical scheme. Most CFD codes are 2nd-order. 1st-order upwind = 1.0. High-order DG = 3.0 or 4.0. |
| **Safety factor Fs** | Auto (recommended) or 1.0–5.0 | Auto picks Fs = 1.25 for 3-grid monotonic, Fs = 3.0 for 2-grid or oscillatory. Only override if you have a specific requirement. |
| **Production grid** | Grid 1 (default) to Grid N | Which grid is your production mesh — the one you use for actual analysis. Default is Grid 1 (finest). Change this if your production mesh is coarser. See [Section 8](#8-production-grid-selection). |
| **Reference scale** *(new in v1.2)* | Auto (default) or a positive number | A characteristic scale value used to compute relative errors. Default "Auto" uses the fine-grid solution magnitude |f1|. Set this manually when your solution is near zero (e.g., a temperature difference that is close to 0). Near-zero solutions cause relative errors to blow up (division by a tiny number), producing misleading GCI percentages. Entering a physically meaningful reference scale (e.g., the overall temperature range, a freestream velocity, or a characteristic dimension) stabilizes the relative error calculation. |

### Step 2: Enter Grid Data

Fill in the table with:
- **Column 1 (Cell Count):** Total number of cells in each grid
- **Column 2+ (Solution values):** The CFD solution value on each grid for your quantity of interest

**Important:** Order the grids **finest first, coarsest last**. The finest grid (most cells) goes in Row 1.

**Three ways to enter data:**
1. **Type directly** into the table cells
2. **Paste from Excel:** Copy your data (rows = grids, columns = cell count + quantities), click "Paste from Clipboard"
3. **Load an example:** Use the top-level **Examples** menu (see [Section 22](#22-built-in-example-datasets))

### Step 3: Click "Compute GCI"

The blue **Compute GCI** button runs the calculation.

**Grid ordering validation (new in v1.2):** The tool checks that the cell counts are in descending order (finest grid first, coarsest last). If the cell counts are not in descending order, a warning dialog appears explaining the issue and offering to **auto-sort** the rows so that the finest grid is in Row 1. You can accept the auto-sort or cancel and reorder manually. This prevents a common data-entry mistake that would produce incorrect GCI results.

### Step 4: Read the Results

- Check the **guidance panels** first — they give you the traffic-light assessment
- Read the **u_num value** — this is what goes into your V&V 20 uncertainty budget
- Look for the **carry-over box** — the highlighted box at the end of the numerical results tells you exactly what value to enter and how to enter it in the Aggregator
- Review the **Celik Table 1** — auto-generated standard reporting table, ready to copy into your report
- Check the **Method Comparison** *(new in v1.3)* — if you have 3+ grids, the tool auto-computes the Factor of Safety (FS) method; with 4+ grids, the Least Squares Root (LSR) method is also computed. A comparison table shows u_num from each method side by side. See [Section 21](#21-method-comparison).
- Check the **reviewer checklist** — quick pass/fail assessment of your study quality
- Check the **convergence plot** — visually confirm the grids are converging. For 3+ grids with monotonic convergence, the log-log subplot shows the order of accuracy visually
- Check the **Report Statements** tab *(new in v1.3)* — copy-pasteable regulatory paragraphs for your V&V report. See [Section 19](#19-report-statements).

### Step 5: Save Your Study

Before proceeding, save your study with **File → Save Study** (Ctrl+S) so you have a complete record of the inputs. See [Section 9](#9-saving-and-loading-studies).

### Step 6: Enter u_num into the Uncertainty Aggregator

Take the u_num value from the carry-over box in the results and enter it into the main VVUQ Uncertainty Aggregator:
1. Add a new source in the Uncertainty Sources tab
2. Set category to **"Numerical (u_num)"**
3. Set input type to **"Sigma Value Only"**
4. Enter the u_num value
5. Set sigma basis to **"Confirmed 1σ"**
6. Set DOF to **∞** (infinity) — the GCI is a systematic estimate, not a sample statistic

---

## 7. Understanding the Results

The results panel shows a comprehensive breakdown. Here's what each section means:

### Grid Details
Lists each grid with its cell count, representative spacing h, and solution value. Verify this matches what you entered.

### Refinement Ratios
The ratio r between each consecutive grid pair. Celik et al. recommend r > 1.3 for reliable results. If r is too close to 1.0, the grids are too similar and the GCI becomes sensitive to numerical noise.

### Discretization Errors
- **|e21| = |f2 - f1|:** Absolute difference between grid 2 and grid 1 (finest)
- **|e21|/|f1| (relative):** Same, as a percentage of the fine-grid solution

### Convergence Type
One of: monotonic (ideal), oscillatory (caution), divergent (invalid), or grid-independent (perfect). See [Section 12](#12-convergence-types-explained).

### Observed Order (p)
The rate at which the numerical error decreases with grid refinement. Should be close to your theoretical order (e.g., ~2 for 2nd-order schemes). If it's much higher or lower, the grids may not be in the asymptotic range.

### Richardson Extrapolation
The estimated solution on an infinitely fine grid (h → 0). Think of this as the "true" numerical answer, before model errors.

### GCI Results
- **GCI_fine:** The main result — the estimated relative error band on the fine grid. Expressed as a fraction (multiply by 100 for percentage).
- **GCI_coarse:** Same for the coarse grid pair (typically larger).
- **Asymptotic ratio:** Should be approximately 1.0. This confirms the grids are in the asymptotic range where the error decreases at the expected rate.

### Numerical Uncertainty (u_num) — Fine Grid
- **u_num (1-sigma):** The GCI converted to a standard uncertainty for V&V 20. This is the number you enter into the Uncertainty Aggregator (if your production mesh is the finest grid).
- **u_num / |f1|:** As a percentage of the fine-grid solution.
- **Expanded (k=2):** The expanded uncertainty (approximately plus/minus 2-sigma).

### Per-Grid Numerical Uncertainty
A table showing u_num for **every** grid in the study, from finest to coarsest. Columns:

| Column | Meaning |
|--------|---------|
| Grid | Grid number (1 = finest) |
| N cells | Cell count for that grid |
| f_i | Solution value on that grid |
| \|f_i - f_RE\| | Absolute difference from Richardson extrapolation |
| u_num | Standard uncertainty for that grid (= \|f_i - f_RE\|) |
| u_num% | u_num as percentage of the solution value |

The production grid is marked with a "PRODUCTION" arrow. See [Section 8](#8-production-grid-selection) for details.

### Production Grid Summary
If the production grid is not the finest, a highlighted section shows:
- u_num for the production grid specifically
- Expanded uncertainty (k=2)
- Ratio to the fine-grid u_num (e.g., "3.5x — 250% larger")

### Carry-Over Box

After the numerical results, a prominent highlighted box tells you exactly what to enter into the Uncertainty Aggregator:

```
══════════════════════════════════════════════════════════════
║  ➜  CARRY THIS VALUE TO THE UNCERTAINTY AGGREGATOR:       ║
══════════════════════════════════════════════════════════════
║                                                            ║
║    u_num = 0.0326148   (0.0046% of solution)               ║
║                                                            ║
║    Source: Grid 1 (finest)  |  Fs = 1.25                   ║
║    Enter as: Sigma Value = above  |  Basis = Confirmed 1σ  |  DOF = ∞║
══════════════════════════════════════════════════════════════
```

This box eliminates guesswork. The value shown is always the u_num for whichever grid is selected as the production grid. If your production grid is Grid 3, the box shows Grid 3's u_num — not the fine-grid value.

**What each line means:**
- **u_num = ...:** The exact number to type into the Aggregator's "Sigma Value" field
- **% of solution:** Context for how large this uncertainty is relative to the answer
- **Source:** Which grid this u_num came from
- **Enter as:** The three Aggregator settings — Sigma Value (the number), Basis (**Confirmed 1σ**), DOF (infinity)

### Celik Table 1 — Standard Reporting Format

The results include an automatically generated **Celik et al. (2008) Table 1**, which is the standard format for reporting GCI results in journal papers and certification reports. For a 3-grid study, the table shows:

```
  Celik et al. (2008) Table 1 — Blade Temperature (K)
  --------------------------------------------------
                                 Grid 1     Grid 2     Grid 3
  --------------------------------------------------
  N (cells)                   2,400,000    800,000    267,000
  r_21                            1.4422     1.4422
  phi (solution)               712.4800   713.2100   714.5500
  p (observed order)              2.025
  phi_ext^21                    711.960
  e_a^21 (approx rel error)     0.1024%
  e_ext^21 (extrap rel error)   0.0730%
  GCI_fine^21                    0.0913%
  --------------------------------------------------
```

This table follows the exact variable naming and layout from Celik et al. (2008) Section 5. You can copy it directly into your report or paper. The tool generates this table for each quantity of interest.

### Reviewer Checklist

At the end of the results, an auto-generated **Grid Convergence Review Checklist** provides a quick pass/fail assessment of your study. Each item is evaluated automatically from the computed results:

```
  Grid Convergence Review Checklist:
  --------------------------------------------------
  [✓ PASS]  Grids: 3 grids used (≥3 recommended)
  [✓ PASS]  Refinement ratio: r_min = 1.44 (≥1.3 recommended)
  [✓ PASS]  Convergence: monotonic (ideal for GCI)
  [✓ PASS]  Observed order: p = 2.025 vs theoretical 2.0 (within 30%)
  [✓ PASS]  Asymptotic ratio: 1.003 (within 0.95–1.05)
  [✓ PASS]  GCI magnitude: 0.09% (excellent, <2%)
  [i INFO]  Verify iterative convergence at each grid level
  [i INFO]  Confirm identical solver settings across all grids
  --------------------------------------------------
```

**Status indicators:**
| Symbol | Meaning | Action Needed |
|--------|---------|---------------|
| **[✓ PASS]** | Criterion met. No issues. | None |
| **[⚠ NOTE]** | Marginal or advisory. Study is valid but could be stronger. | Document in your report and consider improvement |
| **[✗ FAIL]** | Criterion not met. Results may be unreliable. | Fix the issue before using the GCI for certification |
| **[i INFO]** | Reminder — cannot be checked automatically. | Verify manually as part of your V&V process |

This checklist is designed for engineering reviewers. Print it alongside the Celik Table 1 to give reviewers immediate confidence (or concern) about the quality of your grid study.

### Method Comparison (New in v1.3)

When 3 or more grids are provided, the results include a **Method Comparison** section showing u_num from multiple methods side by side. This lets you cross-check the GCI result against independent approaches:

- **GCI** — the standard Celik et al. (2008) method (always computed)
- **FS** — the Xing & Stern (2010) Factor of Safety method (3+ grids)
- **LSR** — the Eca & Hoekstra (2014) Least Squares Root method (4+ grids)

A comparison table shows u_num and u_num% from each available method. Consistent results across methods increase confidence; large discrepancies suggest the grids may not be in the asymptotic range. See [Section 20](#20-alternative-uncertainty-methods-fs-and-lsr) and [Section 21](#21-method-comparison) for details.

### Log-Log Convergence Plot

When Richardson extrapolation is available (3+ grids with monotonic convergence), the convergence plot includes a second subplot below the main solution-vs-spacing plot. This **log-log error plot** shows:

- **X-axis:** log(h) — logarithm of representative grid spacing
- **Y-axis:** log(|f_i - f_RE|) — logarithm of the absolute error relative to Richardson extrapolation
- **Data points:** Each grid's error, labeled G1, G2, G3, etc.
- **Reference line:** A slope line at the observed order of accuracy p

**Why this matters:** On a log-log scale, the relationship between error and grid spacing should be a straight line with slope equal to the order of accuracy p. If the points fall along the reference line, the grids are in the asymptotic range and the GCI is reliable. If the points curve or scatter, the grids may not be fine enough.

This plot is the standard way to visualize convergence order in journal papers and is particularly useful for explaining your GCI results to reviewers who want visual confirmation that the order of accuracy is correct.

---

## 8. Production Grid Selection

In real-world CFD work, the grid you actually use for your production runs is often **not** the finest grid in your convergence study. You might build your daily analysis on a 500K-cell mesh, but then create two finer grids (1M and 3M cells) plus two coarser ones specifically for the GCI study. The question is: what's the numerical uncertainty on **your production mesh**, not just the finest one?

### The "Production Grid" Concept

The standard GCI procedure computes u_num for the finest grid. But if your production mesh is Grid 3 out of 5, you need u_num for Grid 3. The GCI Calculator handles this automatically.

### How It Works — The Math

Once Richardson extrapolation gives you the estimated exact solution (f_exact), the discretization error on **any** grid is simply:

```
error_i = |f_i - f_exact|
u_num_i = |f_i - f_exact|
```

This is the standard uncertainty (1-sigma) for grid i. It comes directly from the relationship between Richardson extrapolation and the GCI: for the fine grid, GCI = Fs * |f1 - f_exact| / |f1|, and u_num = GCI * |f1| / Fs = |f1 - f_exact|.

The same logic applies to every grid — not just the finest.

### How to Use It

1. **Set your production grid** using the "Production grid" dropdown in the Grid Study Setup panel. By default it's set to Grid 1 (finest), but change it to whichever grid you actually use.

2. **Click Compute GCI.** The results will show:
   - **Per-grid uncertainty table:** u_num for every grid, from finest to coarsest
   - **Production grid summary:** A highlighted section showing the u_num, percentage, and expanded uncertainty specifically for your production mesh
   - **Ratio to fine grid:** How much larger the production-grid u_num is compared to the fine-grid u_num

3. **Read the convergence plot.** Your production grid is highlighted with an orange diamond marker and a star label. If it's not the finest grid, you'll also see an orange uncertainty band showing the expanded (k=2) interval around the production-grid solution.

### Example: 5-Grid Study with Middle Production Grid

Load the built-in example **"Exhaust Duct — 5-Grid (Production = Grid 3)"** from the Examples menu. This shows a realistic scenario:

- Grid 1 (3.375M cells): finest — built only for the GCI study
- Grid 2 (1.0M cells): finer verification mesh
- **Grid 3 (500K cells): your production mesh** (the one you run every day)
- Grid 4 (148K cells): coarser check
- Grid 5 (44K cells): coarsest

The results show u_num for every grid. Grid 3's u_num is roughly 12x larger than Grid 1's — that's the price of using a coarser production mesh. This information lets you make an informed decision: is the production-grid uncertainty acceptable, or do you need a finer production mesh?

### What Value Goes into V&V 20?

**Use the u_num for your production grid** — that's the mesh your actual CFD predictions come from. The fine-grid u_num is irrelevant to your production analysis unless you're actually running production cases on the fine grid.

| Situation | Which u_num to use |
|-----------|-------------------|
| Production mesh = finest grid | u_num from standard GCI (the usual case) |
| Production mesh = coarser grid | u_num from the per-grid table for your production grid |
| No clear production mesh (research study) | u_num from the finest grid (standard GCI) |

---

## 9. Saving and Loading Studies

The GCI Calculator lets you save your entire study — all grid data, settings, and quantity names — to a project file. This means you can close the application, come back later, and pick up exactly where you left off. It also lets you share studies with colleagues or maintain archives for certification records.

### Project File Format

Studies are saved as **`.gci` files** — these are standard JSON text files with a `.gci` extension. They contain:

- Application version
- Number of grids and grid data (cell counts and solution values)
- Dimensions setting (2D/3D)
- Theoretical order of accuracy
- Safety factor
- Production grid selection
- Number and names of quantities of interest
- Timestamp of when the file was saved

Because the files are plain JSON, they are human-readable, version-control friendly (Git), and can be inspected or edited in any text editor if needed.

### Saving a Study

1. **First save:** Use **File → Save Study As** (Ctrl+Shift+S) to choose a location and file name
2. **Subsequent saves:** Use **File → Save Study** (Ctrl+S) to overwrite the same file
3. The window title updates to show the project name (e.g., "GCI Calculator — turbine_blade.gci")
4. The status bar confirms: "Saved: C:\path\to\turbine_blade.gci"

### Loading a Study

1. Use **File → Open Study** (Ctrl+O)
2. Browse to a `.gci` or `.json` file
3. The tool restores all settings and grid data exactly as saved
4. Click **Compute GCI** to regenerate results

The tool handles backward compatibility — project files from earlier versions load correctly in newer versions.

### Starting Fresh

Use **File → New Study** (Ctrl+N) to clear everything and start over. The tool will prompt you to confirm before discarding unsaved data.

### Tips for Project Files

- **Save before computing:** Save your input data before clicking Compute GCI, so you have a record of the raw inputs regardless of the results
- **One file per grid study:** Keep each grid convergence study in its own .gci file. Name it descriptively (e.g., `combustor_liner_3grid_temp.gci`)
- **Archive with your V&V report:** Include the .gci file alongside your V&V documentation. It provides a complete, reproducible record of the GCI calculation
- **Share with reviewers:** Send the .gci file to reviewers so they can load it, inspect the inputs, and verify the results independently

---

## 10. Unit Labels for Quantities

Starting in v1.1 (enhanced in v1.2), you can assign **unit labels** to each quantity of interest. Units appear in column headers, results text, carry-over boxes, Celik Table 1, and plot axes.

### How It Works

Below the "Grid Data" table on the left panel, you'll find a **"Quantity Properties (units)"** group box. For each quantity, you see:

1. **Category combo** — select a category (Temperature, Pressure, Velocity, Mass Flow, Force, Length, Other)
2. **Unit combo** — populated with preset units for that category. The combo is **editable**, so you can type any custom unit (e.g., "BTU/hr", "kg/m^3", "MW")

### Important: Units Are Purely Cosmetic

The tool does **not** perform any unit conversions. GCI is computed on the raw numbers you enter, regardless of what unit label is attached. The unit label simply ensures that all output displays carry the correct dimensional annotation for clarity and traceability.

### Preset Unit Categories

| Category | Preset Units |
|----------|-------------|
| Temperature | K, C, F, R |
| Pressure | Pa, kPa, psia, psig, bar, atm |
| Velocity | m/s, ft/s, ft/min |
| Mass Flow | kg/s, lb/s, lb/min |
| Force | N, kN, lbf |
| Length | m, mm, in, ft |
| Other | (empty — type your own) |

### Where Units Appear

- **Column headers:** "Solution: Blade Temp (K)"
- **Results text:** Grid details, errors, Richardson extrapolation, u_num lines
- **Carry-over box:** "u_num = 0.342 K"
- **Celik Table 1:** header includes the quantity label with unit
- **Convergence plot:** y-axis label shows "Solution (K)"
- **Saved project files:** units are stored in .gci files and restored on load

### Tips

- Set units **before** computing GCI so they appear correctly in all output
- The built-in examples come with pre-configured units
- Use consistent units across your V&V 20 uncertainty budget
- Custom units persist in saved project files

---

## 11. Reading the Guidance Panels

Three color-coded panels give you immediate, plain-language feedback:

### Convergence Assessment

| Color | Meaning | What to Do |
|-------|---------|------------|
| **Green** | Monotonic convergence. Solutions converge steadily toward one value. Ideal for GCI. | Nothing — proceed with confidence. |
| **Yellow** | Oscillatory convergence (-1 < R < 0). Solutions bounce between grid levels but oscillations are damping. Richardson extrapolation is unreliable. | GCI is computed from the oscillation range with Fs = 3.0. Consider checking solver settings, iterative convergence, and odd/even decoupling. |
| **Red** | Divergent (R >= 1 or R <= -1). Solutions get worse with refinement, or oscillations are growing. GCI is NOT valid. | Stop and fix your simulation. Check mesh quality, boundary layers, solver convergence, and modeling issues. |

### Order of Accuracy

| Color | Meaning | What to Do |
|-------|---------|------------|
| **Green** | Observed order p is within 30% of theoretical. Grids are in the asymptotic range. | Nothing — ideal situation. |
| **Yellow** | Observed p differs moderately from theoretical. Grids may not be fully in asymptotic range. | GCI is still valid but less precise. Consider finer grids for higher confidence. |
| **Red** | Observed p is much lower than theoretical (< 50%), or much higher (> 2×). Grids are likely too coarse. | Add a finer grid level, or verify solver convergence at each grid level. |

### Asymptotic Range Check

| Color | Meaning | What to Do |
|-------|---------|------------|
| **Green** | Ratio 0.95–1.05. Excellent — grids are well within the asymptotic range. GCI is reliable. | Nothing — gold standard. |
| **Yellow** | Ratio 0.8–1.2. Acceptable — approximately in asymptotic range. | GCI is reasonable. Document the ratio in your report. |
| **Red** | Ratio outside 0.8–1.2. Grids are NOT in asymptotic range. GCI may be unreliable. | Add finer grids, increase refinement ratios, or improve solver convergence. |

---

## 12. Convergence Types Explained

When you run a CFD case on 3 grids, the solutions fall into one of four convergence patterns:

### Convergence Ratio: R = (f2 - f1) / (f3 - f2)

*Note:* Some references (e.g., Celik et al. 2008) define the convergence ratio as epsilon_32/epsilon_21 (the reciprocal). The classification thresholds are equivalent — our `0 < R < 1` corresponds to Celik's `epsilon_32/epsilon_21 > 1` for monotonic convergence.

| R Value | Type | What's Happening | GCI Valid? |
|---------|------|------------------|------------|
| 0 < R < 1 | **Monotonic** | Each finer grid gives a better answer. The corrections get smaller with refinement. This is the ideal case. | **Yes** — standard Celik procedure |
| -1 < R < 0 | **Oscillatory** | The solution oscillates — grid 2 is on one side of the true answer, grid 1 is on the other side. Oscillations are damping with refinement. | **Yes, with caution** — uses oscillation range + Fs = 3.0 |
| R <= -1 | **Divergent (oscillatory)** | The solution oscillates AND the oscillations are growing or not damping with refinement. This is classified as divergent because the oscillatory GCI procedure is unreliable when oscillations amplify. *(Updated in v1.2)* | **No** — GCI cannot be computed |
| R >= 1 | **Divergent** | The solution gets *worse* with refinement. Something is wrong. | **No** — GCI cannot be computed |
| R ≈ 0 (both e21 ≈ 0 AND e32 ≈ 0) | **Grid-independent** | All grids give essentially the same answer. Numerical uncertainty is zero (or negligible). Both consecutive differences must be near zero. *(Tightened in v1.2 — see note below)* | **Yes** — u_num ≈ 0 |

**Grid-independent classification (v1.2 fix):** In previous versions, the tool could classify a triplet as "grid-independent" if only e32 (the coarse-pair difference) was near zero, even when e21 (the fine-pair difference) was large. This was a false positive — it indicated that the two coarser grids happened to agree, not that the solution was truly grid-independent. Starting in v1.2, the tool requires **both** |e21| ≈ 0 **and** |e32| ≈ 0 to classify a result as grid-independent. If only e32 ≈ 0 but e21 is significant, the result is classified as **divergent** because the fine grid departs from the coarser grids that have stalled.

### What Causes Each Type

**Monotonic convergence (what you want):**
- Properly resolved simulation
- Grids in the asymptotic range
- Well-posed problem with smooth solutions

**Oscillatory convergence (common, manageable):**
- Odd/even pressure-velocity decoupling (common in SIMPLE-based solvers)
- Near-singularities or discontinuities (shocks, contact surfaces)
- Marginal solver convergence at some grid level
- Sometimes just the nature of the discretization scheme

**Divergent convergence (needs fixing):**
- Grids too coarse to be in the asymptotic range
- Insufficient iterative convergence (residuals not low enough)
- Mesh quality issues (high skewness, aspect ratio)
- Boundary layer not resolved (first cell too large)
- Non-smooth problem features (singularities, re-entrant corners)

**Oscillatory divergence (R <= -1, new classification in v1.2):**
- Same causes as oscillatory convergence, but worse — the oscillations are growing with refinement instead of damping
- Often indicates severe odd/even decoupling or that the grids have not yet entered the asymptotic range
- Treated as divergent because the oscillatory GCI bounding approach assumes oscillations are damping

### What To Do When You Get Divergent Convergence

If the GCI Calculator classifies your result as divergent, **do not try to force a numerical uncertainty value**. Divergent convergence means the Richardson extrapolation assumptions are violated and GCI is mathematically invalid. Instead, follow this remediation checklist:

**Step 1 — Verify iterative convergence on every grid.**
Check that solver residuals have dropped at least 3-4 orders of magnitude on each grid level. A fine-grid solution that is only partially converged can appear worse than a fully-converged coarse-grid result, producing false divergence.

**Step 2 — Check mesh quality metrics.**
Inspect maximum skewness, aspect ratio, and orthogonal quality for each grid. Poor cell quality on one grid level (especially the finest) can degrade the solution enough to cause divergent behavior.

**Step 3 — Confirm consistent physics across grids.**
Ensure that wall functions, turbulence model switches, limiters, and boundary conditions behave the same way on all grids. For example, if a wall function switches from low-Re to high-Re mode between grid levels, the solution difference is dominated by model switching rather than discretization error.

**Step 4 — Check the measurement location.**
Avoid extracting quantities at geometric singularities (re-entrant corners, sharp leading edges) or at discontinuities (shocks, contact surfaces). Move the extraction point to a smooth region, or use area-averaged quantities.

**Step 5 — Add a finer grid level.**
Sometimes all three grids are too coarse to be in the asymptotic range. Adding a significantly finer grid (refinement ratio >= 1.3) often resolves apparent divergence.

**Step 6 — If divergence persists after all remediation:**
Document the steps you took and report numerical uncertainty as "not quantifiable via GCI" in your V&V report. Use engineering judgment (e.g., the spread across grid solutions) as a bounding estimate, clearly stated as a non-formal uncertainty. The tool's report statement generator will produce appropriate "INCONCLUSIVE" language for this case.

---

## 13. Spatial / Field GCI Analysis

The **Spatial GCI** tab (introduced in v1.1, enhanced in v1.2) lets you compute GCI at hundreds or thousands of spatial locations simultaneously, rather than at a single integral quantity. This is essential when numerical uncertainty varies across a surface (e.g., higher near boundary layers, lower in the free stream).

### When to Use Spatial GCI

- You have **surface maps** (temperature, pressure, etc.) from 2+ grids
- Numerical uncertainty varies **spatially** across the domain
- You need a **single representative u_num** for the V&V 20 budget that accounts for the worst-case locations
- You want to **visualize** where numerical error is concentrated

**Note:** The alternative uncertainty methods (Factor of Safety and LSR) are not applied in the spatial analysis. The standard Celik et al. (2008) GCI procedure is computed at each point independently. Use the single-point GCI Calculator tab for FS and LSR cross-checks on integral quantities.

### Data Input Modes

The Spatial GCI tab supports three input modes:

**1. Pre-interpolated (single CSV)**
All grids share the same point locations. CSV format:
```
x, y, [z], f_grid1, f_grid2, f_grid3
```
This is the simplest mode. You handle interpolation externally (e.g., in your post-processor).

**2. Separate CSV files per grid**
One CSV per grid with columns `x, y, [z], value`. Points are automatically interpolated onto the finest grid using inverse-distance-weighted (IDW) interpolation.

**3. Separate Fluent .prof files**
One Fluent profile export per grid. Interpolation is handled automatically. *(New in v1.3:)* Multi-surface .prof files are now supported — if Fluent exports a .prof file containing multiple surface definitions (e.g., `upper-lipskin` and `lower-lipskin`), the parser automatically combines all surfaces into a single merged dataset. This is transparent; no special user action is needed.

### Setup Steps

1. Select **Data mode** in the Spatial Study Setup group
2. Set **Dimensions** (2D or 3D), **Theoretical order**, and **Safety factor**
3. Enter the **Quantity name** and **Unit**
4. Load your data files
5. Enter **cell counts** for each grid (finest first)
6. Click **Compute Spatial GCI**

### Spatial GCI Sub-Tab Layout

The right panel of the Spatial GCI tab is organized into five sub-tabs:
- **Results** — Full spatial results text (monospace, copy-ready), including per-point statistics and the carry-over box
- **Statistics** — Summary statistics table (mean, median, 95th percentile, max, RMS, std dev) and convergence breakdown counts
- **Plots** — Diagnostic plots, with nested sub-tabs inside:
  - **Histogram** — Distribution of u_num values
  - **CDF** — Cumulative distribution function
  - **Conv Map** — Convergence type map (spatial scatter colored by convergence type)
  - **u_num Map** — Spatial u_num map (hot colormap)
- **Report Statements** *(new in v1.3)* — Copy-pasteable regulatory paragraphs for spatial V&V reports (see [Section 19](#19-report-statements))
- **3D Point Cloud** *(new)* — Interactive 3D viewer for visualizing spatial sample points (see below)

### 3D Point Cloud Viewer

The **3D Point Cloud** sub-tab provides an interactive 3D visualization of your spatial GCI results. You can orbit, zoom, and pan to inspect where numerical uncertainty is highest.

**Requirements:** This feature requires `pyvista` and `pyvistaqt`. Install with:
```
pip install pyvista pyvistaqt
```
If these packages are not installed, the tab will show installation instructions instead.

**Color-by modes:** Use the dropdown at the top to color points by:

| Mode | What It Shows | Colors |
|------|---------------|--------|
| **Convergence Type** | Classification of each point | Green = monotonic, Blue = oscillatory, Yellow = grid-independent, Red = divergent |
| **u_num Magnitude** | Numerical uncertainty value | Cool-to-warm colormap (blue = low, red = high) |
| **Observed Order (p)** | Computed convergence order | Viridis colormap |
| **GCI (%)** | GCI as percentage | Hot colormap (yellow = low, dark red = high) |

**How to use:**
1. Run the Spatial GCI computation first — the 3D viewer populates automatically
2. Select a color mode from the dropdown
3. Click and drag to orbit; scroll to zoom; middle-click to pan
4. Use the scalar bar legend to identify values at specific points

### Point Sampling (New in v1.2)

For large spatial datasets (thousands or tens of thousands of points), computing GCI at every point can be slow and produces redundant information in densely sampled regions. The **Point Sampling** group in the Spatial GCI tab controls how many points are analyzed.

| Option | Behavior |
|--------|----------|
| **Use all points** (default) | Computes GCI at every point in the dataset. Best for small to moderate datasets (up to a few thousand points). |
| **Subsample (equal spacing)** | Selects a user-specified number of approximately equally spaced points from the finest grid, then computes GCI only at those points. |

**How subsampling works — Farthest Point Sampling (FPS):**

When you select "Subsample (equal spacing)" and specify a target number of points (e.g., 500), the tool uses the **farthest-point sampling (FPS)** algorithm to select well-distributed points:

1. Start with an arbitrary seed point from the finest grid
2. Greedily select the next point that is farthest (in 3D Euclidean distance) from all previously selected points
3. Repeat until the target number of points is reached

This is a greedy maximin algorithm — it maximizes the minimum distance between any two selected points. The result is an approximately uniform spatial distribution of sample points. FPS works naturally on curved surfaces and irregular geometries because it operates in 3D Euclidean space, not in a parametric coordinate system.

**When to use subsampling:**
- Datasets with more than ~5,000 points where full computation is slow
- Datasets with highly non-uniform point density (e.g., refined near walls but sparse in the freestream) where full analysis would oversample dense regions
- Quick exploratory runs before committing to a full-dataset analysis

### Understanding the Results

The spatial analysis computes GCI at every point, then reports distribution statistics:

| Statistic | Meaning |
|-----------|---------|
| **Mean** | Average u_num across all valid points |
| **Median** | 50th percentile u_num (robust to outliers) |
| **95th Percentile** | Recommended value for V&V 20 budget |
| **Maximum** | Worst-case point (may be an outlier) |
| **RMS** | Root-mean-square u_num |
| **Std Dev** | Spread of the u_num distribution |

The **95th percentile** is highlighted and recommended as the representative u_num for the V&V 20 uncertainty budget. It is conservative enough to cover 95% of the domain but not dominated by extreme outliers (unlike the maximum).

### Convergence Breakdown

Each point is classified as:
- **Monotonic** (green): Solutions converge steadily. Ideal for GCI.
- **Oscillatory** (yellow): Solutions oscillate between grid levels. Included in statistics by default (toggle in Analysis Options).
- **Divergent** (red): Solutions diverge. Excluded from statistics.

The convergence map plot shows the spatial distribution of these types.

### Divergent Point Reporting (New in v1.2)

When more than **10% of spatial points** are classified as divergent, the results include additional diagnostic information to help you identify and investigate the problematic region:

- **Bounding box of divergent points:** The axis-aligned bounding box (min/max x, y, z coordinates) enclosing all divergent points. This tells you where in the domain the divergence is concentrated.
- **Mean |R| of divergent points:** The average absolute convergence ratio among divergent points. Values much larger than 1 indicate severe divergence; values near 1 suggest the points are marginally divergent.
- **Convex hull on convergence map:** The convergence map plot (Conv Map sub-tab under Plots) draws a **dashed convex hull** around clusters of divergent points, making it visually obvious where the problematic region lies. This overlay helps you correlate divergent locations with mesh features, boundary conditions, or flow phenomena.

Use this information to investigate whether the divergence is caused by a localized mesh issue (e.g., poor cell quality in one region), a physical feature (e.g., flow separation or a singularity), or a global problem that invalidates the entire spatial study.

### Four Diagnostic Plots

1. **Histogram** — Distribution of u_num with vertical lines at mean, median, 95th percentile, and maximum
2. **CDF** — Cumulative distribution function with the 95th percentile marker
3. **Convergence Map** — Scatter plot colored by convergence type (green/yellow/red)
4. **Spatial u_num Map** — Scatter plot colored by u_num magnitude (hot colormap)

### IDW Interpolation (Separate Files Mode)

When using separate files per grid, points from coarser grids are interpolated onto the finest grid's locations using inverse-distance-weighted interpolation:

- **k neighbors** (default 8): Number of nearest source points used per target point
- **Power** (default 2.0): IDW weighting exponent. Higher values weight closer points more heavily

### Built-In Example

The **"Flat Plate Heat Transfer — Spatial (3-Grid)"** example in the Examples menu generates ~200 synthetic points with a sinusoidal temperature field. Use it to explore the spatial GCI workflow without needing external data files.

### References

- Eca, L. & Hoekstra, M. (2014b) "A procedure for the estimation of the numerical uncertainty of CFD calculations based on grid refinement studies" *Int. J. Numer. Meth. Fluids* 75 — spatial/field GCI procedure
- Eca, L. & Hoekstra, M. (2014a) "The numerical friction line" *J. Comp. Physics* 262, 104-130 — Least Squares Root (LSR) method (see [Section 20](#20-alternative-uncertainty-methods-fs-and-lsr))
- Roache, P.J. (1998) Ch. 5 — Field GCI and Local Truncation Error
- Celik et al. (2008) applied point-by-point

---

## 14. The Three-Grid Procedure (Celik et al. 2008)

This is the standard procedure used by the tool for 3+ grids with monotonic convergence.

### Step 1: Define Grid Spacings

Label grids 1 (finest), 2 (medium), 3 (coarsest). Compute representative cell size:

```
h = (1/N)^(1/dim)
```

where N = number of cells and dim = spatial dimension (2 or 3).

### Step 2: Compute Refinement Ratios

```
r_21 = h_2 / h_1 = (N_1 / N_2)^(1/dim)
r_32 = h_3 / h_2 = (N_2 / N_3)^(1/dim)
```

Celik et al. recommend **r > 1.3** for reliable results. If r is too close to 1.0, the differences between grids are dominated by noise rather than systematic discretization error.

### Step 3: Determine Convergence Type

```
R = (f_2 - f_1) / (f_3 - f_2)
```

If 0 < R < 1: monotonic convergence → proceed with standard GCI.
If -1 < R < 0: oscillatory convergence → use oscillation range + Fs = 3.0.
If R >= 1 or R <= -1: divergent → GCI is invalid.

### Step 4: Solve for Observed Order of Accuracy

For constant refinement ratio (r_21 = r_32 = r):
```
p = ln(e_32 / e_21) / ln(r)
```

For non-constant ratio, solve iteratively (Celik Eq. 5 — the tool does this automatically).

### Step 5: Richardson Extrapolation

Estimate the zero-spacing solution:
```
f_exact = f_1 + (f_1 - f_2) / (r_21^p - 1)
```

### Step 6: Compute GCI

```
e_a_21 = |(f_2 - f_1) / f_1|     (relative error)

GCI_fine = Fs * e_a_21 / (r_21^p - 1)
```

### Step 7: Asymptotic Range Check

```
Asymptotic ratio = GCI_coarse / (r_21^p * GCI_fine)
```

Should be approximately 1.0. If it's far from 1.0, the grids aren't in the asymptotic range.

---

## 15. Safety Factor Recommendations

| Scenario | Fs | Why | Source |
|----------|----|-----|--------|
| 3-grid, monotonic convergence | **1.25** | Standard. The observed order provides enough confidence. | Roache (1998) |
| 2-grid study | **3.0** | Conservative. The order is assumed, not computed — you need extra margin for that unknown. | Roache (1998) |
| Oscillatory convergence | **3.0** | Conservative. Richardson extrapolation is unreliable, so the GCI comes from the oscillation range. | Celik et al. (2008) |
| 1st-order scheme | **3.0** | First-order schemes are very sensitive to grid, and the error decreases slowly. | Engineering practice |
| p > 2× theoretical order | **3.0** | Suspiciously high observed order may indicate error cancellation or non-asymptotic behavior. | Engineering judgment |

The tool automatically selects the appropriate Fs when set to "Auto." Override only if your certifying authority requires a specific value.

---

## 16. Converting GCI to u_num for V&V 20

GCI_fine is a **relative error band** — it represents a conservative estimate of the discretization error around the fine-grid solution. Roache (1998) characterized this as bounding the error in the majority of well-resolved studies, though it is not a formal statistical confidence interval. To use it in the V&V 20 RSS uncertainty budget, convert it to a **standard uncertainty (1σ)**:

```
u_num = GCI_fine × |f_1| / Fs
```

**Why divide by Fs?** The safety factor Fs acts like a coverage factor k — it inflates the error band to provide margin. Dividing it out recovers the underlying 1σ estimate, which is what RSS needs.

**What to enter in the Uncertainty Aggregator:**
- **Value:** u_num (the number, in the same units as your CFD output)
- **Sigma basis:** "Confirmed 1σ"
- **DOF:** ∞ (infinity) — the GCI is a deterministic estimate, not a statistical sample

> **UQ Mapping Assumption:** The quantity u_num = |f1 - f_RE| is a deterministic error estimate, not a statistically derived standard deviation. Treating it as a "1-sigma equivalent" for RSS aggregation is a **modeling assumption** — one that is standard practice in ASME V&V 20 (Section 5.1) and Roache (1998), but not a rigorously proven statistical result. The GCI error band is a conservative engineering estimate, and the 1-sigma interpretation is an approximation that enables integration with the broader V&V 20 uncertainty framework. DOF is set to ∞ because the estimate is deterministic (model-based), not sampled from a statistical population.

---

## 17. Multiple Quantities of Interest

Click **"+ Add Quantity"** to analyze multiple output variables simultaneously. Common use cases:

- Temperature at thermocouple location TC-01 AND pressure at pitot tube PT-05
- Drag coefficient AND lift coefficient
- Multiple sensor locations in a thermal survey

Each quantity gets its own:
- Independent GCI computation
- Convergence type assessment
- Observed order of accuracy
- u_num value

All quantities share the same grid cell counts (since they came from the same meshes).

The Results Summary table shows all quantities side by side, making it easy to compare GCI across your quantities of interest.

---

## 18. Convergence Plot

The plot shows solution values vs. representative grid spacing (h):

- **Blue circles connected by lines:** Your grid solutions. Finest grid (smallest h) on the left, coarsest on the right.
- **Green star at h = 0:** The Richardson extrapolation — estimated zero-spacing solution.
- **Green dashed line:** Horizontal reference line at the Richardson extrapolation value.
- **Blue shaded band:** The GCI error band around the fine-grid solution. This represents the estimated numerical uncertainty.
- **Orange diamond:** The production grid (if not the finest). Highlighted with a larger marker and a star in the label.
- **Orange shaded band:** If the production grid is not the finest, an orange band shows the expanded (k=2) uncertainty interval around the production-grid solution. This is wider than the fine-grid band, showing the cost of using a coarser production mesh.
- **Grid labels:** Each point is annotated with its grid number. The production grid label includes a star symbol.

**What a good plot looks like:** The points should approach the green star smoothly from the right, with each step getting smaller. The GCI band should be narrow. If using a non-finest production grid, the orange band should still be acceptably narrow for your application.

**What a bad plot looks like:** Points that zigzag (oscillatory), spread apart (divergent), or jump erratically (noise-dominated).

### Log-Log Error Subplot

When Richardson extrapolation is available (3+ grids with monotonic convergence), a second subplot appears below the main plot. This **log-log convergence plot** shows:

- **X-axis:** log(h) — logarithm of grid spacing
- **Y-axis:** log(|f_i - f_RE|) — logarithm of absolute error relative to Richardson extrapolation
- **Points:** Each grid's error, labeled G1, G2, G3, etc.
- **Slope line:** A reference line at the observed order of accuracy p

On a log-log scale, the error-vs-spacing relationship should be a straight line with slope equal to p. If the data points follow the reference line, the grids are in the asymptotic range and the GCI is well-founded. This plot is the standard convergence-order visualization used in journal papers and certification reports.

You can copy the plot to the clipboard by clicking the **"Copy to Clipboard"** button below the plot.

---

## 19. Report Statements

*(New in v1.3)*

The **Report Statements** sub-tab provides ready-to-use regulatory paragraphs that you can copy directly into your V&V report. This eliminates the need to manually compose uncertainty language — the tool generates it automatically based on your computed results.

### GCI Calculator Tab — Report Statements

After computing GCI, click the **Report Statements** sub-tab in the right panel. The text area contains up to 6 sections:

**1. NUMERICAL UNCERTAINTY STATEMENT** — One paragraph per quantity of interest, automatically adapted to the convergence type:
- **Monotonic:** Full statement citing grid counts, refinement ratios, observed order, GCI_fine percentage, u_num value, and asymptotic ratio.
- **Oscillatory:** Modified language noting oscillatory convergence, Fs = 3.0, and that Richardson extrapolation is unreliable.
- **Divergent:** "INCONCLUSIVE" language stating that no valid numerical uncertainty can be assigned.
- **2-grid:** Notes assumed order and recommends a 3-grid study for certification.
- **Grid-independent:** States that all grids produced the same result.

**2. PRODUCTION GRID STATEMENT** — Only appears when the production grid is not the finest. Reports the production grid's u_num and notes which grid it is.

**3. MULTI-QUANTITY SUMMARY** — Only appears when more than one quantity is analyzed. Lists each quantity with its u_num and convergence type, and identifies the dominant numerical uncertainty.

**4. LIMITATIONS & CAVEATS** — Auto-populated from any warnings detected during computation:
- Low refinement ratio (< 1.3)
- Observed order deviating significantly from theoretical
- Oscillatory or divergent behavior
- 2-grid limitation
- Near-zero solution scaling

**5. ALTERNATIVE METHOD COMPARISON** — Only appears when FS and/or LSR methods were computed. Notes the u_num from each method and whether the estimates are consistent.

**6. STANDARDS COMPLIANCE** — A fixed block citing the method, references, procedure description, basis, and coverage factor.

### Spatial GCI Tab — Report Statements

The Spatial GCI tab also has a **Report Statements** sub-tab with 3 sections:

**1. SPATIAL NUMERICAL UNCERTAINTY STATEMENT** — Paragraph citing the number of points analyzed, convergence breakdown percentages (monotonic/oscillatory/divergent), and the recommended 95th percentile u_num. If subsampling was used, notes the FPS method and sample count.

**2. LIMITATIONS & CAVEATS** — Auto-populated for spatial analysis:
- High divergent fraction warnings (>10%)
- Subsampling note if enabled
- IDW interpolation caveat (separate files mode)

**3. STANDARDS COMPLIANCE** — Same fixed reference block as the GCI calculator.

### Copying Report Statements

Click the **"Copy Statements to Clipboard"** button at the top of the Report Statements sub-tab to copy all text to the system clipboard. Paste into Word, LaTeX, or any document editor.

---

## 20. Alternative Uncertainty Methods (FS and LSR)

*(New in v1.3)*

In addition to the standard GCI (Celik et al. 2008), the tool automatically computes two alternative numerical uncertainty methods when enough grids are available. These methods use the same input data but apply different mathematical frameworks, providing independent cross-checks on the GCI result.

### Factor of Safety Method (FS) — variant after Xing & Stern (2010)

**Requires: 3+ grids**

The FS method uses a **variable safety factor** instead of the fixed Fs = 1.25 used in GCI. The safety factor adapts based on how close the observed order of accuracy is to the theoretical order:

```
CF = r21^p / (r21^p - 1)          (correction factor)
FS = FS1 * |1 - CF| + FS0        when |1 - CF| < 1/FS1
     FS2 * |1 - CF| + FS0        otherwise
```

where FS0 = 1.6, FS1 = 2.45, FS2 = 14.8 are empirically calibrated constants (Xing & Stern 2010, Table 1). The branching depends on `|1 - CF|`, which measures how close the correction factor is to unity (the ideal asymptotic case). The ratio `P = p_observed / p_theoretical` is also computed and displayed as a diagnostic — it indicates how close the grids are to the asymptotic range (P ≈ 1 is ideal) — but P does not directly drive the safety factor formula. The numerical uncertainty is then:

```
u_num_FS = FS * |delta_RE| / 2
```

where delta_RE is the Richardson extrapolation correction.

**Key difference from GCI:** The FS method's safety factor increases smoothly as the solution moves away from the asymptotic range (P deviates from 1.0), whereas GCI uses a fixed Fs = 1.25. This makes FS more conservative when grids are not fully in the asymptotic range and less conservative when they are.

**For oscillatory convergence:** The FS method uses Fs = 3.0 with the oscillation range, similar to the standard GCI approach.

**For divergent convergence:** No valid FS estimate can be computed.

### Least Squares Root Method (LSR) — variant with AICc, after Eca & Hoekstra (2014)

**Requires: 4+ grids**

The LSR method takes a fundamentally different approach: instead of computing the order of accuracy from a triplet of grids, it **fits power-law models** to all grid solutions simultaneously using least squares regression. Four models are fitted:

| Model | Formula | Parameters |
|-------|---------|------------|
| **M1** | phi = phi_0 + alpha * h^p | 3 (phi_0, alpha, p — free order) |
| **M2** | phi = phi_0 + alpha * h^p_th | 2 (phi_0, alpha — fixed theoretical order) |
| **M3** | phi = phi_0 + alpha * h + beta * h^2 | 3 (phi_0, alpha, beta — two-term) |
| **M4** | phi = phi_0 + alpha * h | 2 (phi_0, alpha — first order) |

The best model is selected using the **corrected Akaike Information Criterion (AICc)**, which balances goodness of fit against model complexity. This prevents overfitting with too many parameters.

The numerical uncertainty is then computed from the selected model's extrapolated solution (phi_0) with a variable safety factor based on the observed-to-theoretical order ratio:

```
u_base    = FS_LSR * |f1 - phi_0| / 2
u_num_LSR = sqrt( u_base^2 + std_model^2 )
```

where `std_model` is the residual standard error of the best-fit model. The RSS combination accounts for both the extrapolation error and the model fit uncertainty.

> **Implementation Note:** This tool's LSR implementation enhances the original Eca & Hoekstra (2014) procedure in two ways: (1) model selection uses AICc rather than minimum standard deviation, which penalizes model complexity and prevents overfitting; and (2) the final uncertainty combines the extrapolation error and model standard deviation via RSS (root-sum-square) rather than the paper's additive form. These are improvements that make the method more robust for small numbers of grids.

**Key advantages of LSR:**
- Uses ALL grids simultaneously (not just the finest triplet)
- Less sensitive to individual grid outliers
- Can detect when no single power-law describes the convergence (a sign of pre-asymptotic behavior)
- Statistically rigorous model selection via AICc

**Key limitation:** Requires at least 4 grids. With only 3 grids, there are too few data points for meaningful regression.

### How to Read FS and LSR Results

After the standard GCI results, the Results text shows:
1. **FS Method Details** — FS value, P ratio, correction factor CF, and u_num
2. **LSR Method Details** (if 4+ grids) — Best model name, extrapolated solution phi_0, observed order, and u_num
3. **Method Comparison Table** — Side-by-side u_num from all available methods

These methods are computed **automatically** — no additional user input is required. They use the same grid data and settings as the GCI computation.

### Which Method Should I Use?

For most applications, **use the standard GCI**. It is the most widely accepted and cited method in the CFD verification literature. The FS and LSR methods serve as **cross-checks**:

| Scenario | Recommendation |
|----------|---------------|
| All methods agree (within ~50%) | High confidence in the result. Use GCI. |
| FS gives a notably different u_num | The grids may not be in the asymptotic range. The FS method's variable safety factor is accounting for this. Consider using the more conservative estimate. |
| LSR gives a notably different u_num | The power-law fit may reveal non-monotonic convergence behavior across all grids. Investigate the LSR model choice. |
| Large disagreement (>2x) between methods | The grid study may be inadequate. Add more grids or refine further. |

### References

- Xing, T. and Stern, F. (2010) "Factors of Safety for Richardson Extrapolation" *ASME J. Fluids Eng.* 132(6), 061403
- Eca, L. and Hoekstra, M. (2014a) "The numerical friction line" *J. Comp. Physics* 262, 104-130
- ITTC 7.5-03-01-01 (2021/2024) "Uncertainty Analysis in CFD Verification and Validation"

---

## 21. Method Comparison

*(New in v1.3)*

The **Method Comparison** section appears in the results text whenever alternative methods are available (3+ grids). It provides a comparison table:

```
  Method Comparison: GCI vs FS [vs LSR]
  --------------------------------------------------
  Quantity          GCI u_num   FS u_num   [LSR u_num]
  --------------------------------------------------
  Blade Temp (K)    0.0326      0.0289      0.0341
  Pressure (psi)    0.0142      0.0118      0.0155
  --------------------------------------------------
```

### Interpreting the Comparison

- **Consistent results** (all within ~50% of each other): Strong evidence that the grid study is well-resolved and the u_num estimate is reliable. Use the standard GCI value.
- **Moderate spread** (50-100% variation): The methods are responding differently to the grid data. This often indicates the grids are near the boundary of the asymptotic range. Consider using the most conservative (largest) estimate.
- **Large spread** (>2x variation): The methods disagree significantly. This is a strong signal that additional grids or finer grids are needed. Do not rely on any single method's result without further investigation.

The Method Comparison section is also referenced in the Report Statements sub-tab (Section 19), which auto-generates a paragraph noting the alternative method results.

---

## 22. Built-In Example Datasets

The tool includes built-in example datasets accessible via the top-level **Examples** menu in the menu bar.

### Example 1: Turbine Blade 4-Grid Study (Monotonic)
A realistic thermal CFD example with 4 grids and 2 quantities (blade temperature and pressure). This demonstrates ideal monotonic convergence with a well-resolved grid study. With 4 grids, both the FS and LSR methods are automatically computed for comparison.

- 4 grids: 2.4M → 800K → 267K → 89K cells
- 2 quantities: Blade Temperature (K) and Pressure (psi)
- Expected result: Monotonic convergence, observed order near 2.0
- FS and LSR methods auto-computed for cross-checking

### Example 2: Pipe Flow 3-Grid Study (Oscillatory)
A standard 3-grid study where the solution oscillates between grid levels. Demonstrates how the tool handles oscillatory convergence with a conservative safety factor.

- 3 grids: 500K → 150K → 45K cells
- 1 quantity: Pressure Drop (Pa)
- Expected result: Oscillatory convergence, Fs = 3.0

### Example 3: Heat Exchanger 3-Grid Study (Clean 2nd-Order)
A textbook-clean second-order convergence example. The observed order should be very close to 2.0, and the asymptotic ratio should be near 1.0. The FS method is auto-computed for comparison.

- 3 grids: 1M → 125K → 15.6K cells (constant r = 2.0)
- 1 quantity: Outlet Temperature (K)
- Expected result: p ≈ 2.0, asymptotic ratio ≈ 1.0
- FS method auto-computed (3 grids; LSR requires 4+)

### Example 4: Nozzle Flow 3-Grid Study (Divergent)
An example where the solution diverges with grid refinement. Demonstrates what divergence looks like and the warning messages the tool produces.

- 3 grids: 800K → 200K → 50K cells
- 1 quantity: Thrust (N)
- Expected result: Divergent, GCI invalid

### Example 5: Combustor Liner 2-Grid Study (Assumed Order)
A common real-world scenario where you only have two grids — perhaps a production mesh and one refinement. The tool uses the assumed theoretical order (p = 2) and the conservative safety factor Fs = 3.0.

- 2 grids: 3.2M → 800K cells
- 1 quantity: Wall Temperature (K)
- Expected result: 2-grid assumed order, Fs = 3.0, valid result with wider uncertainty band

### Example 6: Exhaust Duct 5-Grid Study (Production = Grid 3)
The key example for the production grid feature. Your production CFD runs on a 500K-cell mesh (Grid 3), but you created two finer grids and two coarser grids for the GCI study. The production mesh is NOT the finest — it sits in the middle. With 5 grids, all three methods (GCI, FS, LSR) are available.

- 5 grids: 3.375M → 1M → **500K (production)** → 148K → 44K cells
- 1 quantity: Exit Temperature (K)
- Production grid: Grid 3 (pre-selected)
- Expected result: Monotonic convergence, production-grid u_num roughly 12x larger than fine-grid u_num
- All three methods (GCI, FS, LSR) auto-computed for cross-checking

This example demonstrates:
- Per-grid uncertainty table showing u_num for all 5 grids
- Production grid summary with highlighted u_num
- Convergence plot with orange diamond on Grid 3 and orange uncertainty band
- The ratio between production-grid and fine-grid u_num
- Method Comparison table with GCI, FS, and LSR u_num side by side

### Example 7: Flat Plate Heat Transfer — Spatial (3-Grid)
This example demonstrates the **Spatial GCI** tab. It generates ~200 synthetic points on a flat plate with a sinusoidal temperature field. Three grids (2.4M, 800K, 267K cells) are simulated with grid-dependent noise and systematic h^2 bias.

- ~196 points on a rectangular plate (2D coordinates)
- 3 grids: 2.4M (fine), 800K (medium), 267K (coarse)
- Pre-interpolated format (no external files needed)
- Expected result: mostly monotonic convergence, 95th percentile u_num as recommended value

This example demonstrates:
- Histogram of u_num distribution with statistical markers
- CDF with 95th percentile recommendation
- Convergence type map (green/yellow/red spatial scatter)
- Spatial u_num map (hot colormap showing where uncertainty concentrates)
- Carry-over box with the 95th percentile u_num

---

## 23. Exporting Results

### Copy to Clipboard (Ctrl+C)

Copies the full results text — everything in the results panel — to the system clipboard. Paste into Word, Notepad, email, or any other application. The copied text includes the Celik Table 1 and reviewer checklist, which are formatted for direct inclusion in reports.

### Export to File (Ctrl+E)

Saves the results to a plain text file (.txt). The file contains all the information needed for a V&V 20 report section on numerical uncertainty, including the Celik Table 1, carry-over box, and reviewer checklist.

### Export HTML Report (Ctrl+H) *(new in v1.3)*

Generates a self-contained HTML document that can be opened in any web browser, printed to PDF, or attached to a certification package. The report includes:

- **Configuration summary** (dimensions, safety factor, convergence threshold, grids)
- **Celik Table 1** per quantity (formatted HTML tables matching the journal standard)
- **Convergence plot** (embedded as a 300 DPI PNG with light theme for print)
- **Method comparison** (GCI vs FS variant vs LSR variant, side-by-side)
- **Report statements** (auto-generated regulatory paragraphs)
- **Reviewer checklist** (PASS/NOTE/FAIL items with color-coded verdicts)
- **UQ mapping assumption** (highlighted documentation of the u_num = 1σ policy)
- **References** (ASME V&V 20, Celik et al. 2008, Roache 1998, etc.)

The file is fully self-contained with no external dependencies — all plots and styles are embedded inline. Use **File → Export HTML Report...** or press **Ctrl+H**.

### Copy Plot to Clipboard

Each chart toolbar includes three output actions:

1. **Copy to Clipboard** — Copies the chart at 150 DPI (draft quality). Paste directly into Word, PowerPoint, or email.
2. **Copy Report-Quality** — Copies the chart at 300 DPI using a light theme suitable for formal reports and publications. The light colour scheme avoids dark backgrounds that print poorly.
3. **Export Figure Package...** — Exports a complete figure archive containing:
   - PNG at 300 DPI and 600 DPI (raster)
   - SVG (scalable vector)
   - PDF (vector)
   - JSON metadata sidecar with traceability fields (tool version, analysis ID, settings hash, generation timestamp, units, method context)

The JSON sidecar supports regulatory audit chains by recording the provenance of each figure.

### Copy Report Statements *(new in v1.3)*

Click the **"Copy Statements to Clipboard"** button in the **Report Statements** sub-tab to copy the auto-generated regulatory paragraphs. These are ready-to-paste V&V report language covering numerical uncertainty statements, limitations, and standards compliance. See [Section 19](#19-report-statements).

### Save Study (Ctrl+S)

Saves the input data and settings to a `.gci` project file for later retrieval. This does NOT save the results — results are regenerated by clicking Compute GCI after loading. See [Section 9](#9-saving-and-loading-studies).

---

## 24. How to Use u_num in the Uncertainty Aggregator

Once you have u_num from the GCI Calculator, here's exactly how to enter it into the main VVUQ Uncertainty Aggregator:

1. Open the Uncertainty Aggregator (`vv20_validation_tool.py`)
2. Go to the **Uncertainty Sources** tab (Tab 2)
3. Click **"Add Source"**
4. Set the fields:

| Field | Value |
|-------|-------|
| **Name** | "Grid Convergence (GCI)" or similar descriptive name |
| **Category** | Numerical (u_num) |
| **Distribution** | Normal (standard assumption for numerical error) |
| **Input Type** | Sigma Value Only |
| **Sigma Value** | The u_num number from the GCI Calculator |
| **Sigma Basis** | Confirmed 1σ |
| **DOF** | ∞ (infinity) |

5. Ensure the source is **enabled** (checked)
6. The RSS results will automatically update to include your grid convergence uncertainty

---

## 25. Tips for Getting a Good Grid Study

### Grid Generation

- **Use a consistent refinement approach.** Refine globally (all cells) rather than just in one region. If you only refine the boundary layer, the refinement ratio doesn't apply uniformly.
- **Target refinement ratio r > 1.3.** If r is too close to 1.0, the differences between grids are dominated by noise. r = 1.5 to 2.0 is ideal.
- **Keep mesh quality comparable.** Don't change mesh topology between grids — just change the density. Same blocking, same inflation layers (just more of them on finer grids).

### Running the Cases

- **Use the same solver settings on every grid.** Same turbulence model, same discretization scheme, same convergence criteria. The only difference should be the mesh.
- **Converge each case fully.** Iterative convergence residuals should be at least 3 orders of magnitude below the initial value. If grid 2 isn't converged, its solution will pollute the GCI.
- **Use the same boundary conditions.** Obvious, but worth stating — the grids must represent the same physical problem.
- **Watch for solver artifacts.** Some solvers behave differently on different grid densities (e.g., limiters activating on coarse grids). This can cause apparent oscillatory or divergent convergence.

### Choosing What to Measure

- **Use a point quantity, not an integral.** GCI works best for solution values at specific locations (temperature at a thermocouple, pressure at a port). Area-averaged or volume-averaged quantities tend to converge faster and may mask local errors.
- **Choose a quantity in a region of interest.** Don't measure temperature at the inlet (where it's prescribed) — measure it at the location you actually care about.
- **Avoid quantities at singularities.** Leading-edge stagnation points, re-entrant corners, and contact lines may not converge at the theoretical rate.

---

## 26. Key Formulas Reference

### Representative Cell Size
```
h = (1/N)^(1/dim)
```
where N = cell count, dim = spatial dimension (2 or 3).

### Refinement Ratio
```
r = h_coarse / h_fine = (N_fine / N_coarse)^(1/dim)
```
Should be > 1.0 (>1.3 recommended).

### Convergence Ratio
```
R = (f_2 - f_1) / (f_3 - f_2)
```
0 < R < 1: monotonic. -1 < R < 0: oscillatory. R <= -1 or R >= 1: divergent.

### Observed Order of Accuracy (constant r)
```
p = ln(e_32 / e_21) / ln(r)
```
where e_21 = f_2 - f_1, e_32 = f_3 - f_2.

### Richardson Extrapolation
```
f_exact = f_1 + (f_1 - f_2) / (r_21^p - 1)
```

### GCI (Fine Grid)
```
GCI_fine = Fs * |e_a_21| / (r_21^p - 1)
```
where e_a_21 = |(f_2 - f_1) / f_1| (relative error).

### Asymptotic Range Check
```
Asymptotic ratio = GCI_coarse / (r_21^p * GCI_fine)  ≈  1.0
```

### u_num Conversion (Fine Grid)
```
u_num = |f_1 - f_exact|
```
This gives the 1-sigma standard uncertainty for V&V 20. Note: the algebraic equivalence `GCI_fine * |f_1| / Fs = |f_1 - f_exact|` holds when the default reference scale is used (i.e., the relative error denominator is `|f_1|`). When a custom reference scale is set, the tool computes u_num directly as `|f_1 - f_exact|` to avoid the normalization mismatch.

### Per-Grid u_num (Any Grid)
```
u_num_i = |f_i - f_exact|
```
where f_exact is the Richardson extrapolation estimate. This gives the numerical uncertainty for any grid in the study, not just the finest. Use this when your production mesh is not the finest grid.

### Factor of Safety (FS) Method *(new in v1.3)*
```
CF = r21^p / (r21^p - 1)
FS = FS1 * |1 - CF| + FS0    when |1 - CF| < 1/FS1
     FS2 * |1 - CF| + FS0    otherwise
u_num_FS = FS * |delta_RE| / 2
P = p_observed / p_theoretical  (diagnostic only)
```
where FS0 = 1.6, FS1 = 2.45, FS2 = 14.8, and delta_RE = (f1 - f2) / (r21^p - 1).

### Least Squares Root (LSR) Method *(new in v1.3)*
```
Model 1: phi = phi_0 + alpha * h^p           (3 params, free order)
Model 2: phi = phi_0 + alpha * h^p_th        (2 params, fixed order)
Model 3: phi = phi_0 + alpha*h + beta*h^2    (3 params, two-term)
Model 4: phi = phi_0 + alpha * h             (2 params, first order)

AICc = n * ln(RSS/n) + 2k + 2k(k+1)/(n-k-1)
```
Best model selected by minimum AICc. Uncertainty:
u_base = FS_LSR * |f1 - phi_0| / 2; u_num_LSR = sqrt(u_base^2 + std_model^2).

---

## 27. Standards References

| Standard | Full Title | What This Tool Uses It For |
|----------|-----------|---------------------------|
| **Celik et al. (2008)** | "Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications" — *J. Fluids Eng.* 130(7), 078001 | The primary procedure: observed order, Richardson extrapolation, GCI, asymptotic check |
| **Roache (1998)** | *Verification and Validation in Computational Science and Engineering* — Hermosa Publishers | GCI concept, safety factor recommendations (Fs = 1.25 / 3.0) |
| **Xing & Stern (2010)** | "Factors of Safety for Richardson Extrapolation" — *ASME J. Fluids Eng.* 132(6), 061403 | Factor of Safety (FS) method with variable safety factor *(new in v1.3)* |
| **Eca & Hoekstra (2014a)** | "The numerical friction line" — *J. Comp. Physics* 262, 104-130 | Least Squares Root (LSR) method with 4-model power-law fit *(new in v1.3)* |
| **Eca & Hoekstra (2014b)** | "A procedure for the estimation of the numerical uncertainty of CFD calculations based on grid refinement studies" — *Int. J. Numer. Meth. Fluids* 75(12), 803-826 | Spatial/field GCI procedure |
| **ASME V&V 20-2009 (R2021)** | Standard for Verification and Validation in CFD and Heat Transfer — Section 5.1. *Note: A 2023 revision (V&V 20-2023) exists with updated guidance.* | How u_num fits into the overall validation uncertainty budget |
| **Richardson (1911)** | "The Approximate Arithmetical Solution by Finite Differences of Physical Problems" — *Phil. Trans. R. Soc. A* 210, 307-357 | The mathematical foundation of Richardson extrapolation |
| **ITTC (2024)** | "Uncertainty Analysis in CFD Verification and Validation" — Procedure 7.5-03-01-01, Rev. 05 | Additional guidance on GCI for marine/naval CFD applications |
| **Phillips & Roy (2014)** | "Richardson Extrapolation-Based Discretization Uncertainty Estimation for Computational Fluid Dynamics" — *ASME J. Fluids Eng.* 136(12), 121401 | Comprehensive review of RE-based uncertainty methods including GCI, CF, and FS |

---

## 28. Glossary

| Term | Plain English Definition |
|------|------------------------|
| **GCI** | Grid Convergence Index — a standardized error band estimating numerical uncertainty from grid refinement |
| **Richardson Extrapolation** | A method for estimating the infinitely-fine-grid solution by extrapolating the convergence trend |
| **Refinement Ratio (r)** | The ratio of cell sizes between two grids: r = h_coarse / h_fine. Always > 1.0 |
| **Observed Order (p)** | The rate at which the numerical error decreases with grid refinement. Should be close to the theoretical order of your scheme |
| **Theoretical Order** | The formal accuracy order of your numerical scheme (e.g., 2 for 2nd-order, 1 for 1st-order upwind) |
| **Asymptotic Range** | The grid spacing regime where error decreases at the theoretical rate. Grids must be fine enough to be in this regime for GCI to work |
| **Safety Factor (Fs)** | A multiplier applied to GCI to add margin. Fs = 1.25 (3-grid standard) or 3.0 (2-grid/oscillatory, conservative). In the FS method, Fs is variable. |
| **u_num** | Numerical uncertainty — a 1-sigma standard uncertainty representing the estimated numerical error. Goes into V&V 20 budgets |
| **Factor of Safety Method (FS)** | Variant after Xing & Stern (2010) that uses a variable safety factor based on the ratio of observed to theoretical order and a correction factor. Requires 3+ grids |
| **Least Squares Root Method (LSR)** | Variant with AICc after Eca & Hoekstra (2014) that fits 4 power-law models to grid solutions via least squares. Uses AICc model selection and RSS uncertainty combination. Requires 4+ grids |
| **Method Comparison** | A side-by-side table of u_num values from GCI, FS, and LSR methods. Auto-generated when 3+ grids are available |
| **Report Statements** | Auto-generated regulatory paragraphs for V&V reports, available in a dedicated sub-tab for both GCI and Spatial analyses |
| **AICc** | Corrected Akaike Information Criterion — a model selection metric that balances goodness of fit against number of parameters, with a small-sample correction |
| **Correction Factor (CF)** | CF = r^p / (r^p - 1). Used in the FS method to gauge how reliable Richardson extrapolation is |
| **Production Grid** | The mesh you actually use for your day-to-day CFD analysis runs. It may or may not be the finest grid in the GCI study |
| **Per-Grid u_num** | The numerical uncertainty for a specific grid level, computed as the absolute difference between that grid's solution and the Richardson extrapolation |
| **Monotonic Convergence** | Solutions converge steadily in one direction with grid refinement (0 < R < 1). Ideal for GCI |
| **Oscillatory Convergence** | Solutions oscillate between grid levels with damping oscillations (-1 < R < 0). GCI uses a conservative bounding approach |
| **Oscillatory Divergence** | Solutions oscillate between grid levels with growing oscillations (R <= -1). Classified as divergent — GCI is invalid |
| **Divergent Convergence** | Solutions get worse with refinement (R >= 1 or R <= -1). GCI is invalid — fix the simulation |
| **Grid-Independent** | All grids give the same answer (both e21 and e32 near zero). Numerical uncertainty is essentially zero |
| **Reference Scale** | A characteristic value used for relative error calculation. Set manually for near-zero solutions to avoid division-by-small-number issues |
| **Farthest Point Sampling (FPS)** | A greedy maximin algorithm that selects approximately equally spaced points from a point cloud by iteratively picking the point farthest from all previously selected points |
| **Cell Count (N)** | Total number of computational cells in a mesh |
| **Representative Spacing (h)** | An effective cell size computed from cell count: h = (1/N)^(1/dim) |
| **Convergence Ratio (R)** | R = (f2 - f1) / (f3 - f2). Determines convergence type |
| **Carry-Over Box** | The highlighted box in the results showing the exact u_num value and Aggregator entry instructions |
| **Celik Table 1** | The standard reporting table format from Celik et al. (2008) — auto-generated in the results |
| **Reviewer Checklist** | An automated pass/fail assessment of the grid study quality, generated from the computed results |
| **Log-Log Plot** | A convergence-order visualization: log(error) vs log(spacing). Slope equals observed order p |
| **Project File (.gci)** | A JSON file containing all input data and settings for a GCI study, for saving and reloading |
| **Unit Label** | A cosmetic string (e.g., K, Pa, m/s) attached to a quantity. Appears in headers, results, and plots. No conversion is performed |
| **Spatial GCI** | Point-by-point GCI computation over a surface map, producing a distribution of u_num values |
| **95th Percentile u_num** | The recommended representative numerical uncertainty from a spatial GCI analysis, covering 95% of the domain |
| **IDW Interpolation** | Inverse-distance-weighted interpolation used to map field data from one grid's point locations onto another's |
| **Pre-interpolated CSV** | A CSV file where all grids share the same point locations (x, y, [z], f1, f2, f3, ...) |

---

## 29. Frequently Asked Questions

### Q: My GCI calculation shows "divergent" — what's wrong?

**A:** Divergent means the solution gets worse as you refine the grid. GCI cannot be computed. Common causes:
1. **Grids too coarse** — none of them are in the asymptotic range yet. Try adding a much finer grid.
2. **Insufficient iterative convergence** — check that solver residuals are fully converged on each grid. A partially-converged fine-grid solution can look worse than a fully-converged coarse-grid solution.
3. **Mesh quality issues** — high skewness, high aspect ratio cells, or topology changes between grids.
4. **Singularities** — measuring at a re-entrant corner, leading edge, or other singular point.
5. **Modeling issues** — boundary conditions that interact with the grid (e.g., wall functions that switch behavior at certain y+ values).

### Q: The observed order is much higher than my theoretical order. Is that a problem?

**A:** Not necessarily, but be cautious. Observed p > 2× theoretical may indicate:
- **Error cancellation** — two sources of error happening to cancel at certain grid spacings
- **Superconvergence** — the scheme happens to converge faster at your particular measurement location (e.g., a symmetry point)
- **Not fully in the asymptotic range** — the error hasn't settled into its theoretical decay rate yet

The GCI is still computed, but consider the result with reduced confidence. Adding more grid levels helps clarify the situation.

### Q: The observed order is much lower than my theoretical order. What does that mean?

**A:** Observed p < 0.5× theoretical suggests:
- **Grids too coarse** — the dominant error term hasn't started decaying at the theoretical rate
- **Mixed-order effects** — your scheme may be 2nd-order in smooth regions but 1st-order near discontinuities, boundary layers, or limiters
- **Poor mesh quality** — high skewness or aspect ratio degrading the formal order
- **Solver contamination** — iterative errors competing with discretization errors

### Q: Can I use this for time-step convergence studies?

**A:** Yes. Set **Dimensions** to **"1D (temporal)"** *(new in v1.3)* and enter:
- "Cell Count" = number of time steps (or 1/dt, so that "finer" = more steps = larger number)
- Solution values at each time-step size

The math is identical — Richardson extrapolation works the same for temporal refinement. The 1D dimension setting ensures the refinement ratio is computed correctly as `r = N_fine / N_coarse` (i.e., `h = (1/N)^(1/1)`). Using 2D or 3D for temporal studies would give incorrect refinement ratios because the 1/dim exponent would compress the ratio.

### Q: How do I report GCI results in a paper or certification report?

**A:** Follow Celik et al. (2008) Table 1 format. Report, at minimum:

1. Number of grids and their cell counts
2. Refinement ratios (r_21, r_32)
3. Solutions on each grid (f_1, f_2, f_3)
4. Observed order of accuracy (p)
5. Richardson extrapolation (f_exact)
6. GCI_fine and GCI_coarse (as percentages)
7. Asymptotic ratio

The tool automatically generates a Celik Table 1 in the results text — copy it directly into your report. The "Export to File" function saves all of this in a copy-ready format.

### Q: I have unstructured meshes — does GCI still work?

**A:** Yes. The tool computes an effective refinement ratio from cell counts:

```
r = (N_fine / N_coarse)^(1/dim)
```

This is the standard approach for unstructured meshes per Celik et al. (2008) Eq. 3. It assumes the cells are roughly isotropic. If your mesh has highly anisotropic cells (e.g., boundary layer prisms), the effective refinement ratio may not perfectly represent the actual resolution change, but the GCI is still a reasonable estimate.

### Q: My production mesh is not the finest grid. Which u_num should I use?

**A:** Use the u_num from the **per-grid uncertainty table** for your production grid. Set the "Production grid" dropdown to the correct grid before computing. The tool will highlight the production grid's u_num in the results.

For example, if your production mesh is Grid 3 out of 5 grids, set "Production grid" to "Grid 3." The results will show a dedicated section with the production grid's u_num, its percentage of the solution value, and the ratio to the fine-grid u_num.

### Q: How much worse is u_num on a coarser production grid?

**A:** It depends on the observed order of accuracy and the refinement ratio. For a 2nd-order scheme with r = 1.5, going from the finest grid to one grid coarser roughly doubles u_num. Going two grids coarser can be 4-10x worse.

The per-grid uncertainty table shows this directly. Load the "Exhaust Duct — 5-Grid" example to see a realistic case where Grid 3's u_num is about 12x Grid 1's.

Use this information to decide whether your production mesh is "good enough" — if the production-grid u_num is too large for your validation requirements, you need a finer production mesh.

### Q: Can I save my study and come back to it later?

**A:** Yes. Use **File → Save Study** (Ctrl+S) to save all your input data and settings to a `.gci` file. Later, use **File → Open Study** (Ctrl+O) to reload it. The file saves everything — grid counts, cell counts, solution values, quantity names, dimensions, theoretical order, safety factor, and production grid selection. After loading, click **Compute GCI** to regenerate the results.

### Q: What is the Celik Table 1 in the results?

**A:** It's the standard reporting format from Celik et al. (2008) Section 5. Journals and certification bodies expect GCI results in this tabular format. The tool generates it automatically — just copy it from the results and paste into your report. It includes all the required fields: cell counts, refinement ratios, solutions, observed order, Richardson extrapolation, approximate and extrapolated errors, and GCI percentages.

### Q: What does the reviewer checklist check?

**A:** The checklist automatically evaluates seven aspects of your grid study:
1. **Grid count** — do you have 3+ grids (recommended)?
2. **Refinement ratio** — is r > 1.3 (Celik recommendation)?
3. **Convergence type** — is it monotonic (ideal)?
4. **Observed order** — is it close to theoretical?
5. **Asymptotic ratio** — is it near 1.0?
6. **GCI magnitude** — is it reasonably small (<5%)?
7. **Reminders** — iterative convergence and solver settings (manual checks)

Each item gets a PASS, NOTE, or FAIL based on the computed results. It's designed so an engineering reviewer can quickly assess whether the study meets basic quality criteria.

### Q: What's the difference between GCI and just comparing two grid solutions?

**A:** Comparing two solutions tells you the *change* between grids, not the *error* on either grid. The change might be 2°F, but the error on the finer grid might only be 0.3°F (because most of the 2°F was already in the coarse-grid error). GCI uses the mathematical structure of how solutions converge to estimate that 0.3°F — the actual error on *your* grid.

### Q: How do I set units for my quantities?

**A:** In the "Quantity Properties (units)" group below the data table, select a category (Temperature, Pressure, etc.) and pick a unit from the dropdown, or type a custom unit. Units are purely cosmetic — they appear in column headers, results text, carry-over boxes, Celik Table 1, and plot axes. No unit conversion is performed. Units are saved with project files.

### Q: When should I use Spatial GCI instead of single-point GCI?

**A:** Use spatial GCI when numerical uncertainty **varies across the domain** and you need a single representative value for the V&V 20 budget. Typical scenarios:
- Surface temperature distributions where near-wall regions have higher discretization error
- Pressure fields where gradients drive local error
- Any situation where you have exported surface maps from multiple grids

For integral quantities (average temperature, total force, mass flow rate), single-point GCI is sufficient and simpler.

### Q: Why does the spatial GCI recommend the 95th percentile instead of the maximum?

**A:** The maximum u_num over thousands of points is often dominated by a single outlier point (e.g., a near-singularity corner, a cell with bad aspect ratio). The 95th percentile captures the "worst 5% of the domain" while being robust to extreme outliers. This is standard practice in field GCI analysis per Eca & Hoekstra (2014b).

### Q: What format does the pre-interpolated CSV need?

**A:** A simple CSV with columns: `x, y, [z], f_grid1, f_grid2, f_grid3, ...` where the first 2-3 columns are coordinates and the remaining columns are solution values from each grid level (finest first). Header row is optional but recommended. All grids must share the same point locations. Example:
```
x, y, fine, medium, coarse
0.0, 0.0, 423.15, 423.49, 424.51
0.1, 0.0, 425.32, 425.88, 426.91
...
```

### Q: How does automatic point subsampling work?

**A:** When you select "Subsample (equal spacing)" in the Spatial GCI tab, the tool uses **farthest-point sampling (FPS)** to pick a subset of points that are approximately equally distributed in space. The algorithm starts with one arbitrary point and then greedily selects the next point that is farthest (in 3D Euclidean distance) from all previously selected points, repeating until the target count is reached. This maximin approach ensures good spatial coverage regardless of the original point density. It works naturally on curved surfaces because it uses straight-line distance in 3D space, not surface parameterization. Use it when your dataset has thousands of points and full computation would be unnecessarily slow or when point density is highly non-uniform.

### Q: What is the reference scale option?

**A:** The **Reference scale** input (in Grid Study Setup) sets a characteristic value used as the denominator when computing relative errors. By default ("Auto"), the tool uses the fine-grid solution magnitude |f1|. This works well when the solution value is far from zero. However, when your quantity of interest is near zero (e.g., a differential temperature that happens to be 0.01 K, or a force component that is nearly balanced), dividing by |f1| produces enormous relative errors that are numerically meaningless. Setting the reference scale to a physically meaningful characteristic value (such as the overall temperature range, a freestream velocity, or a characteristic pressure) stabilizes the relative error calculation and produces sensible GCI percentages. The absolute u_num value is unaffected — only relative error displays and percentage-based reporting change.

### Q: What are the FS and LSR methods?

**A:** They are two alternative numerical uncertainty methods that the tool computes automatically alongside the standard GCI:

- **Factor of Safety (FS)** — Xing & Stern (2010). Uses a variable safety factor that adapts based on how close your grids are to the asymptotic range. Requires 3+ grids. Same inputs as GCI, no extra work needed.
- **Least Squares Root (LSR)** — Eca & Hoekstra (2014). Fits power-law models to all your grid solutions simultaneously using least squares regression. Requires 4+ grids.

Both are computed automatically when enough grids are available. See [Section 20](#20-alternative-uncertainty-methods-fs-and-lsr).

### Q: Which u_num method should I report — GCI, FS, or LSR?

**A:** For most applications, report the **standard GCI**. It is the most widely cited and accepted method. Use the FS and LSR results as cross-checks:
- If all three methods agree (within ~50%), you have strong confidence in the result.
- If they disagree significantly, the grid study may be inadequate — investigate further.
- Some certifying authorities may prefer a specific method. Check their requirements.

See [Section 21](#21-method-comparison) for interpretation guidance.

### Q: What are Report Statements?

**A:** The **Report Statements** sub-tab (new in v1.3) generates ready-to-use paragraphs for V&V reports. Instead of manually writing uncertainty language, you can copy the auto-generated text directly into your report. The statements adapt to your specific results (convergence type, number of grids, whether oscillatory or divergent behavior was detected, etc.). Both the GCI Calculator tab and Spatial GCI tab have their own Report Statements sub-tab. See [Section 19](#19-report-statements).

### Q: How does the Fluent .prof multi-surface parsing work?

**A:** When Fluent exports a `.prof` file containing multiple profile sections, the parser automatically detects each section using balanced-parenthesis parsing, extracts coordinates and field values, and concatenates them into a single merged dataset. It supports the documented profile header types (`point`, `line`, `mesh`, `radial`, `axial`) and handles multi-surface exports (for example `upper-lipskin` + `lower-lipskin`) transparently.

### Q: What is oscillatory divergence?

**A:** Oscillatory divergence occurs when the convergence ratio R is less than or equal to -1 (R <= -1). This means the solution oscillates between grid levels (the sign of the change alternates), **and** the oscillations are growing or staying the same magnitude with refinement rather than damping out. In v1.2, this is classified as **divergent** rather than oscillatory because the standard oscillatory GCI procedure (which bounds the oscillation range) assumes the oscillations are damping. When oscillations grow, that bounding approach underestimates the true uncertainty. Common causes include severe odd/even pressure-velocity decoupling, grids that are all too coarse for the asymptotic range, and solver instabilities that interact with the mesh. If you see oscillatory divergence, the recommended action is the same as for standard divergence: investigate mesh quality, solver convergence, and boundary conditions before attempting a GCI study.

---

*GCI Calculator v1.4.0 — Compute defensible numerical uncertainty for your CFD grid convergence studies.*

*Standards: Celik et al. (2008) JFE 130(7), Roache (1998), Xing & Stern (2010) JFE 132(6), Eca & Hoekstra (2014a) JCP 262, Eca & Hoekstra (2014b) IJNMF 75, ASME V&V 20-2009 Section 5.1*
