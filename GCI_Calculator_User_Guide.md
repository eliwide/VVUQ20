# GCI Calculator v1.0 — User Guide & Technical Reference

**Grid Convergence Index Tool per Celik et al. (2008) & Roache (1998)**

*Written for engineers — not statisticians.*

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
10. [Reading the Guidance Panels](#10-reading-the-guidance-panels)
11. [Convergence Types Explained](#11-convergence-types-explained)
12. [The Three-Grid Procedure (Celik et al. 2008)](#12-the-three-grid-procedure-celik-et-al-2008)
13. [Safety Factor Recommendations](#13-safety-factor-recommendations)
14. [Converting GCI to u_num for V&V 20](#14-converting-gci-to-u_num-for-vv-20)
15. [Multiple Quantities of Interest](#15-multiple-quantities-of-interest)
16. [Convergence Plot](#16-convergence-plot)
17. [Built-In Example Datasets](#17-built-in-example-datasets)
18. [Exporting Results](#18-exporting-results)
19. [How to Use u_num in the Uncertainty Aggregator](#19-how-to-use-u_num-in-the-uncertainty-aggregator)
20. [Tips for Getting a Good Grid Study](#20-tips-for-getting-a-good-grid-study)
21. [Key Formulas Reference](#21-key-formulas-reference)
22. [Standards References](#22-standards-references)
23. [Glossary](#23-glossary)
24. [Frequently Asked Questions](#24-frequently-asked-questions)

---

## 1. What Is This Tool and Why Do I Need It?

If you run CFD simulations, you need to know how much of your answer is real physics and how much is just the grid talking. The GCI Calculator answers that question.

**The problem:** Every CFD simulation uses a finite mesh. If you made the mesh infinitely fine, the answer would change — but by how much? That gap between your current answer and the "true" numerical answer is your **numerical uncertainty (u_num)**.

**Why it matters:** In the ASME V&V 20 framework for CFD validation uncertainty, u_num is often the single largest contributor to the total uncertainty budget. If you don't quantify it properly, your entire uncertainty analysis is undermined.

**What this tool does:** You feed in your CFD solution values from 2 or more grids of different fineness, and the tool tells you:

> *"Your fine-grid solution has a numerical uncertainty of ±X (95% confidence). The estimated true solution (infinite grid) is Y."*

That u_num number goes directly into your V&V 20 uncertainty budget.

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
| **3** | GCI with *computed* observed order of accuracy. Can verify the grids are in the asymptotic range. Can perform the full Celik et al. procedure. | Fs = 1.25 (standard) | **Standard procedure.** This is what Celik et al. (2008) recommends and what most journals require. Use this for any formal work. |
| **4** | Same as 3-grid for the primary GCI (uses finest 3 grids), plus one extra triplet for cross-checking. | Fs = 1.25 | **Good practice.** The extra grid gives you a consistency check — if the GCI from grids 1-2-3 and grids 2-3-4 agree, your result is robust. |
| **5-6** | Multiple cross-checking triplets. Very high confidence in the result. | Fs = 1.25 | **Best practice for safety-critical work.** Multiple independent GCI estimates should converge on the same answer. |

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
| **Oscillatory convergence** | **No** | When solutions oscillate between grid levels, RE is unreliable. The GCI is computed from the oscillation range with a conservative Fs = 3.0. |
| **Divergent convergence** | **No** | If the solution diverges with refinement, GCI is invalid entirely. Fix your simulation first. |
| **Grid-independent** | **Not needed** | If all grids give the same answer, numerical uncertainty is zero. |

---

## 5. Getting Started — Application Layout

### How to Run

```
python gci_calculator.py
```

### Main Window

The application has **two tabs**:

| Tab | Icon | Purpose |
|-----|------|---------|
| **GCI Calculator** | 📊 | The main calculation interface — input data, compute, see results |
| **Reference** | 📖 | Built-in documentation covering the entire GCI procedure |

### Calculator Tab Layout

The Calculator tab is split into two panels:

**Left panel — Input:**
- Grid Study Setup (number of grids, dimensions, theoretical order, safety factor, production grid)
- Grid Data Table (cell counts and solution values)
- Quantity management buttons (+ Add Quantity, - Remove Last, Paste from Clipboard)
- **Compute GCI** button
- Three guidance panels (Convergence, Order, Asymptotic Range)

**Right panel — Results:**
- Full results text (monospace, copy-ready)
- GCI Results Summary table (color-coded)
- Convergence plot with Richardson extrapolation and GCI band

### Menu Bar

**File Menu:**

| Menu Item | Shortcut | Action |
|-----------|----------|--------|
| New Study | Ctrl+N | Clear all data and start fresh |
| Open Study... | Ctrl+O | Load a previously saved .gci project file |
| Save Study | Ctrl+S | Save the current study to a .gci project file (overwrites if previously saved) |
| Save Study As... | Ctrl+Shift+S | Save to a new .gci project file |
| Export Results to Clipboard | Ctrl+C | Copy the full results text |
| Export Results to File | Ctrl+E | Save results to a text file |
| Load Example Data | — | Load a built-in example dataset |
| Exit | Alt+F4 | Close the application |

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

### Step 2: Enter Grid Data

Fill in the table with:
- **Column 1 (Cell Count):** Total number of cells in each grid
- **Column 2+ (Solution values):** The CFD solution value on each grid for your quantity of interest

**Important:** Order the grids **finest first, coarsest last**. The finest grid (most cells) goes in Row 1.

**Three ways to enter data:**
1. **Type directly** into the table cells
2. **Paste from Excel:** Copy your data (rows = grids, columns = cell count + quantities), click "Paste from Clipboard"
3. **Load an example:** File → Load Example Data (see [Section 17](#17-built-in-example-datasets))

### Step 3: Click "Compute GCI"

The blue **Compute GCI** button runs the calculation. If your grids aren't in descending cell-count order, the tool will offer to auto-sort them.

### Step 4: Read the Results

- Check the **guidance panels** first — they give you the traffic-light assessment
- Read the **u_num value** — this is what goes into your V&V 20 uncertainty budget
- Look for the **carry-over box** — the highlighted box at the end of the numerical results tells you exactly what value to enter and how to enter it in the Aggregator
- Review the **Celik Table 1** — auto-generated standard reporting table, ready to copy into your report
- Check the **reviewer checklist** — quick pass/fail assessment of your study quality
- Check the **convergence plot** — visually confirm the grids are converging. For 3+ grids with monotonic convergence, the log-log subplot shows the order of accuracy visually

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
One of: monotonic (ideal), oscillatory (caution), divergent (invalid), or grid-independent (perfect). See [Section 11](#11-convergence-types-explained).

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
- **Expanded (k=2):** The 95% expanded uncertainty (approximately plus/minus 2-sigma).

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
║    Enter as: Sigma Value = above  |  Basis = 1σ  |  DOF = ∞║
══════════════════════════════════════════════════════════════
```

This box eliminates guesswork. The value shown is always the u_num for whichever grid is selected as the production grid. If your production grid is Grid 3, the box shows Grid 3's u_num — not the fine-grid value.

**What each line means:**
- **u_num = ...:** The exact number to type into the Aggregator's "Sigma Value" field
- **% of solution:** Context for how large this uncertainty is relative to the answer
- **Source:** Which grid this u_num came from
- **Enter as:** The three Aggregator settings — Sigma Value (the number), Basis (1-sigma), DOF (infinity)

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

## 10. Reading the Guidance Panels

Three color-coded panels give you immediate, plain-language feedback:

### Convergence Assessment

| Color | Meaning | What to Do |
|-------|---------|------------|
| **Green** | Monotonic convergence. Solutions converge steadily toward one value. Ideal for GCI. | Nothing — proceed with confidence. |
| **Yellow** | Oscillatory convergence. Solutions bounce between grid levels. Richardson extrapolation is unreliable. | GCI is computed from the oscillation range with Fs = 3.0. Consider checking solver settings, iterative convergence, and odd/even decoupling. |
| **Red** | Divergent. Solutions get worse with refinement. GCI is NOT valid. | Stop and fix your simulation. Check mesh quality, boundary layers, solver convergence, and modeling issues. |

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

## 11. Convergence Types Explained

When you run a CFD case on 3 grids, the solutions fall into one of four convergence patterns:

### Convergence Ratio: R = (f2 - f1) / (f3 - f2)

| R Value | Type | What's Happening | GCI Valid? |
|---------|------|------------------|------------|
| 0 < R < 1 | **Monotonic** | Each finer grid gives a better answer. The corrections get smaller with refinement. This is the ideal case. | **Yes** — standard Celik procedure |
| R < 0 | **Oscillatory** | The solution oscillates — grid 2 is on one side of the true answer, grid 1 is on the other side. | **Yes, with caution** — uses oscillation range + Fs = 3.0 |
| R ≥ 1 | **Divergent** | The solution gets *worse* with refinement. Something is wrong. | **No** — GCI cannot be computed |
| R ≈ 0 | **Grid-independent** | All grids give essentially the same answer. Numerical uncertainty is zero (or negligible). | **Yes** — u_num ≈ 0 |

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

---

## 12. The Three-Grid Procedure (Celik et al. 2008)

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

## 13. Safety Factor Recommendations

| Scenario | Fs | Why | Source |
|----------|----|-----|--------|
| 3-grid, monotonic convergence | **1.25** | Standard. The observed order provides enough confidence. | Roache (1998) |
| 2-grid study | **3.0** | Conservative. The order is assumed, not computed — you need extra margin for that unknown. | Roache (1998) |
| Oscillatory convergence | **3.0** | Conservative. Richardson extrapolation is unreliable, so the GCI comes from the oscillation range. | Celik et al. (2008) |
| 1st-order scheme | **3.0** | First-order schemes are very sensitive to grid, and the error decreases slowly. | Engineering practice |
| p > 2× theoretical order | **3.0** | Suspiciously high observed order may indicate error cancellation or non-asymptotic behavior. | Engineering judgment |

The tool automatically selects the appropriate Fs when set to "Auto." Override only if your certifying authority requires a specific value.

---

## 14. Converting GCI to u_num for V&V 20

GCI_fine is a **relative error band** — it represents an approximately 95% confidence interval around the fine-grid solution. To use it in the V&V 20 RSS uncertainty budget, convert it to a **standard uncertainty (1σ)**:

```
u_num = GCI_fine × |f_1| / Fs
```

**Why divide by Fs?** The safety factor Fs acts like a coverage factor k — it inflates the error band to provide margin. Dividing it out recovers the underlying 1σ estimate, which is what RSS needs.

**What to enter in the Uncertainty Aggregator:**
- **Value:** u_num (the number, in the same units as your CFD output)
- **Sigma basis:** "Confirmed 1σ"
- **DOF:** ∞ (infinity) — the GCI is a deterministic estimate, not a statistical sample

---

## 15. Multiple Quantities of Interest

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

## 16. Convergence Plot

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

## 17. Built-In Example Datasets

The tool includes built-in example datasets accessible via **File → Load Example Data** and the **Examples** submenu.

### Example 1: Turbine Blade 4-Grid Study (Monotonic)
A realistic thermal CFD example with 4 grids and 2 quantities (blade temperature and pressure). This demonstrates ideal monotonic convergence with a well-resolved grid study.

- 4 grids: 2.4M → 800K → 267K → 89K cells
- 2 quantities: Blade Temperature (K) and Pressure (psi)
- Expected result: Monotonic convergence, observed order near 2.0

### Example 2: Pipe Flow 3-Grid Study (Oscillatory)
A standard 3-grid study where the solution oscillates between grid levels. Demonstrates how the tool handles oscillatory convergence with a conservative safety factor.

- 3 grids: 500K → 150K → 45K cells
- 1 quantity: Pressure Drop (Pa)
- Expected result: Oscillatory convergence, Fs = 3.0

### Example 3: Heat Exchanger 3-Grid Study (Clean 2nd-Order)
A textbook-clean second-order convergence example. The observed order should be very close to 2.0, and the asymptotic ratio should be near 1.0.

- 3 grids: 1M → 125K → 15.6K cells (constant r = 2.0)
- 1 quantity: Outlet Temperature (K)
- Expected result: p ≈ 2.0, asymptotic ratio ≈ 1.0

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
The key example for the production grid feature. Your production CFD runs on a 500K-cell mesh (Grid 3), but you created two finer grids and two coarser grids for the GCI study. The production mesh is NOT the finest — it sits in the middle.

- 5 grids: 3.375M → 1M → **500K (production)** → 148K → 44K cells
- 1 quantity: Exit Temperature (K)
- Production grid: Grid 3 (pre-selected)
- Expected result: Monotonic convergence, production-grid u_num roughly 12x larger than fine-grid u_num

This example demonstrates:
- Per-grid uncertainty table showing u_num for all 5 grids
- Production grid summary with highlighted u_num
- Convergence plot with orange diamond on Grid 3 and orange uncertainty band
- The ratio between production-grid and fine-grid u_num

---

## 18. Exporting Results

### Copy to Clipboard (Ctrl+C)

Copies the full results text — everything in the results panel — to the system clipboard. Paste into Word, Notepad, email, or any other application. The copied text includes the Celik Table 1 and reviewer checklist, which are formatted for direct inclusion in reports.

### Export to File (Ctrl+E)

Saves the results to a plain text file (.txt). The file contains all the information needed for a V&V 20 report section on numerical uncertainty, including the Celik Table 1, carry-over box, and reviewer checklist.

### Copy Plot to Clipboard

Click the **"Copy to Clipboard"** button below the convergence plot to copy the plot image (including the log-log subplot if present). Paste directly into Word, PowerPoint, or any other application.

### Save Study (Ctrl+S)

Saves the input data and settings to a `.gci` project file for later retrieval. This does NOT save the results — results are regenerated by clicking Compute GCI after loading. See [Section 9](#9-saving-and-loading-studies).

---

## 19. How to Use u_num in the Uncertainty Aggregator

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

## 20. Tips for Getting a Good Grid Study

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

## 21. Key Formulas Reference

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
0 < R < 1: monotonic. R < 0: oscillatory. R ≥ 1: divergent.

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
u_num = GCI_fine * |f_1| / Fs = |f_1 - f_exact|
```
This gives the 1-sigma standard uncertainty for V&V 20.

### Per-Grid u_num (Any Grid)
```
u_num_i = |f_i - f_exact|
```
where f_exact is the Richardson extrapolation estimate. This gives the numerical uncertainty for any grid in the study, not just the finest. Use this when your production mesh is not the finest grid.

---

## 22. Standards References

| Standard | Full Title | What This Tool Uses It For |
|----------|-----------|---------------------------|
| **Celik et al. (2008)** | "Procedure for Estimation and Reporting of Uncertainty Due to Discretization in CFD Applications" — *J. Fluids Eng.* 130(7), 078001 | The primary procedure: observed order, Richardson extrapolation, GCI, asymptotic check |
| **Roache (1998)** | *Verification and Validation in Computational Science and Engineering* — Hermosa Publishers | GCI concept, safety factor recommendations (Fs = 1.25 / 3.0) |
| **ASME V&V 20-2009 (R2016)** | Standard for Verification and Validation in CFD and Heat Transfer — Section 5.1 | How u_num fits into the overall validation uncertainty budget |
| **Richardson (1911)** | "The Approximate Arithmetical Solution by Finite Differences of Physical Problems" — *Phil. Trans. R. Soc. A* 210, 307-357 | The mathematical foundation of Richardson extrapolation |
| **ITTC (2024)** | "Uncertainty Analysis in CFD Verification and Validation" — Procedure 7.5-03-01-01 | Additional guidance on GCI for marine/naval CFD applications |

---

## 23. Glossary

| Term | Plain English Definition |
|------|------------------------|
| **GCI** | Grid Convergence Index — a standardized error band estimating numerical uncertainty from grid refinement |
| **Richardson Extrapolation** | A method for estimating the infinitely-fine-grid solution by extrapolating the convergence trend |
| **Refinement Ratio (r)** | The ratio of cell sizes between two grids: r = h_coarse / h_fine. Always > 1.0 |
| **Observed Order (p)** | The rate at which the numerical error decreases with grid refinement. Should be close to the theoretical order of your scheme |
| **Theoretical Order** | The formal accuracy order of your numerical scheme (e.g., 2 for 2nd-order, 1 for 1st-order upwind) |
| **Asymptotic Range** | The grid spacing regime where error decreases at the theoretical rate. Grids must be fine enough to be in this regime for GCI to work |
| **Safety Factor (Fs)** | A multiplier applied to GCI to add margin. Fs = 1.25 (3-grid standard) or 3.0 (2-grid/oscillatory, conservative) |
| **u_num** | Numerical uncertainty — a 1-sigma standard uncertainty representing the estimated numerical error. Goes into V&V 20 budgets |
| **Production Grid** | The mesh you actually use for your day-to-day CFD analysis runs. It may or may not be the finest grid in the GCI study |
| **Per-Grid u_num** | The numerical uncertainty for a specific grid level, computed as the absolute difference between that grid's solution and the Richardson extrapolation |
| **Monotonic Convergence** | Solutions converge steadily in one direction with grid refinement (0 < R < 1). Ideal for GCI |
| **Oscillatory Convergence** | Solutions oscillate between grid levels (R < 0). GCI uses a conservative bounding approach |
| **Divergent Convergence** | Solutions get worse with refinement (R ≥ 1). GCI is invalid — fix the simulation |
| **Grid-Independent** | All grids give the same answer. Numerical uncertainty is essentially zero |
| **Cell Count (N)** | Total number of computational cells in a mesh |
| **Representative Spacing (h)** | An effective cell size computed from cell count: h = (1/N)^(1/dim) |
| **Convergence Ratio (R)** | R = (f2 - f1) / (f3 - f2). Determines convergence type |
| **Carry-Over Box** | The highlighted box in the results showing the exact u_num value and Aggregator entry instructions |
| **Celik Table 1** | The standard reporting table format from Celik et al. (2008) — auto-generated in the results |
| **Reviewer Checklist** | An automated pass/fail assessment of the grid study quality, generated from the computed results |
| **Log-Log Plot** | A convergence-order visualization: log(error) vs log(spacing). Slope equals observed order p |
| **Project File (.gci)** | A JSON file containing all input data and settings for a GCI study, for saving and reloading |

---

## 24. Frequently Asked Questions

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

**A:** Yes, with a small adaptation. Instead of cell counts and spatial spacing, enter:
- "Cell Count" = number of time steps (or 1/dt for consistency)
- Solution values at each time-step size

The math is identical — Richardson extrapolation works the same for temporal refinement. Set "Dimensions" to 2D (since time is 1D, but the formula uses dim; for temporal studies with uniform refinement, the dim doesn't affect the refinement ratio if you enter h directly).

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

---

*GCI Calculator v1.0 — Compute defensible numerical uncertainty for your CFD grid convergence studies.*

*Standards: Celik et al. (2008) JFE 130(7), Roache (1998), ASME V&V 20-2009 Section 5.1*
