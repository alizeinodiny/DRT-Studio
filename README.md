# DRT Studio 1.0

A desktop application for **distribution-of-relaxation-times (DRT) analysis of EIS data**,
built around aqueous zinc-ion cells (galvanized Zn anode / hard-carbon cathode / ZnSO₄
electrolyte) but usable for any electrochemical system.

It does the full job end to end: import raw impedance, validate it, invert it three ways
(Re / Im / Re+Im), detect and *physically assign* the peaks, compare cells, and export a
written report.

---

## Running it

```bash
cd /home/user/DRT_Studio
python3 run_drt_studio.py             # launch the GUI
python3 run_drt_studio.py --selftest  # 55 headless checks, no display needed
python3 run_drt_studio.py --demo OUT  # build the 4-cell demo project and export to OUT/
python3 run_drt_studio.py --diagnose  # verify the numerics against a known circuit
```

Requires **Python 3.8 or newer**, plus numpy, scipy, pandas, matplotlib, openpyxl,
reportlab, python-docx (optional, for .docx) and tkinter. On a bare Debian/Ubuntu
box: `sudo apt-get install python3-tk`.

Both **numpy 1.x and numpy 2.x** are supported. NumPy 2.0 renamed `np.trapz` to
`np.trapezoid`; `core/compat.py` picks whichever exists, so the app runs on the
older numpy that Python 3.8 is limited to. The launcher runs a pre-flight check
and prints a plain-language message if a dependency is missing or unusable,
instead of failing part-way through an analysis.

Click **Demo** on the Data tab to load four synthetic cells and see everything working
before you import anything of your own.

---

## The seven tabs

| Tab | What it is for |
|---|---|
| **1. Data** | Import, inspect and annotate samples. Nyquist / Bode / raw table views. |
| **2. Analysis** | Set up and run the inversion; DRT panels, overlay, window bars, findings. |
| **3. Peaks** | Peak table with τ, f, γ_max, Rᵢ, % of R_pol, FWHM and the assignment. Click a row for the physical reference note. |
| **4. Quality** | Kramers-Kronig test, fit residuals, L-curve. Validate *before* interpreting. |
| **5. Compare** | Multi-cell comparison: panels, overlay, window bars, process fingerprint heat-map, summary, findings. |
| **6. Export** | PDF / Word / Excel / PNG bundles for one sample, all samples, or the comparison. |
| **Log** | Everything the app did, timestamped. |

### If something looks wrong

**Export ▸ Verify ▸ Run self-check** (also **Help ▸ Run self-check**) inverts a
synthetic cell whose answer is known analytically and checks that the recovered
R_inf and R_pol match the circuit, that the three modes agree, that the window
resistances sum to R_pol, that Kramers-Kronig passes on clean data and fails on
drifting data, and that a project round-trips through disk. **Help ▸ Environment**
lists which optional packages are installed and exactly what is disabled without
them. Both also run headlessly via `--diagnose`.

---

## Importing data

**Import data…** takes CSV, TSV, TXT, DAT and Excel files. The dialog auto-detects the
delimiter (including European `;` files with decimal commas), the header row, and the
meaning of each column, then lets you correct any of it.

Each file gets a **role**:

- `full` — frequency + Re(Z) + Im(Z) in one file
- `re` / `im` — the two halves in separate files (they are merged on frequency)
- `drt` — an already-computed γ(τ) curve, for plotting and peak assignment only

Files sharing a **group** name become one sample. Frequency may be Hz, kHz or MHz, and
the Im(Z) sign convention is detected automatically (or forced with the `neg` control).

---

## The inversion

γ(τ) is obtained by Tikhonov-regularized deconvolution on a log-spaced τ grid,
fitting R_∞ and a series inductance L alongside γ.

Two engines are available:

- **`drttools`** (default) — the analytic radial-basis discretisation and roughness
  matrix used by DRTtools / pyDRTtools (Ciucci group). The kernel integrals are
  verified against adaptive quadrature to better than 1e-9, and the Toeplitz fast
  path for the roughness matrix is verified exact for all 9 RBFs × both derivative
  orders. Use this for results you intend to publish.
- **`legacy`** — the original lightweight kernel, kept as an independent cross-check.

Controls that matter:

- **Modes** — invert the real part, the imaginary part, or both together. Running all
  three is the point: agreement between them is the strongest evidence a peak is real.
- **RBF type** — Gaussian (default), C0/C2/C4/C6 Matérn, inverse quadratic, inverse
  quadric, Cauchy, or piecewise linear. See the warning below.
- **λ (regularisation)** — chosen automatically by **GCV** (default), **mGCV**,
  **rGCV**, **L-curve**, **re-im discrepancy** or **k-fold CV**, or set by hand.
  Too small → ringing; too large → merged peaks.
- **Derivative order** — 1st or 2nd-order smoothness penalty.
- **Non-negative γ** — physically required; leave it on.
- **Points/decade, FWHM coefficient, extend** — discretisation of the τ grid.
  `extend = 0` reproduces the DRTtools native grid τ = 1/f.

### Choosing λ automatically

DRTtools' GUI ships a fixed λ = 1e-3. That is a fair guess for noisy laboratory data
but it over-smooths clean spectra — on the synthetic test cell it gives a 3.5 %
residual and a single broad peak, where GCV gives 0.6 % and recovers all four true
processes. DRT Studio therefore defaults to **GCV** and only uses a fixed λ when you
select `manual`. λ tracks the noise level as it should (1e-7 at 0.3 % noise,
2e-6 at 3 %).

### Inverse quadric and Cauchy need non-negativity switched off

These two basis functions have very broad, slowly decaying tails. Reproducing a sharp
DRT peak with them requires **negative** coefficients, so combining them with the
`non-negative γ` constraint cannot work: the best achievable non-negative fit is 4–6 %
residual (versus 0.0002 % unconstrained), R_pol lands ~15 % low and the reported
residual exceeds 90 %. Either untick non-negativity for these two, or use Gaussian, a
Matérn, or the inverse quadratic. The GUI warns before running this combination.

### R_∞ absorbs sub-grid relaxations

Any relaxation faster than the first τ node cannot be represented by a basis function,
so its resistance is folded into R_∞. R_∞ on its own is therefore grid dependent; the
invariant worth quoting is the total DC resistance R_∞ + R_pol, which was measured
constant at 218.0 Ω across τ-grid extensions of 0 to 2 decades (true value 218.0 Ω).

Two processes are only separable if their relaxation times differ by roughly a factor
of 2–4; closer than that and they merge into one broadened peak regardless of λ.

### Why Re and Im can disagree

If the three inversions give very different R_pol, that is a **data** problem, not a
numerical one — the imaginary part is far more sensitive to slow drift. Run the
Kramers-Kronig test first: residuals under ~0.5 % mean Re and Im carry the same
information and any disagreement is numerical. Residuals above ~1 % at low frequency
mean the cell drifted during the sweep and the low-frequency resistances are upper
bounds only. The Re+Im combined result is the recommended one to quote.

---

## Peak assignment windows

Peaks are binned into seven τ-windows with a built-in physical interpretation for
aqueous Zn-ion cells:

| | τ range | Process |
|---|---|---|
| **P1** | < 3×10⁻⁵ s | Contact/lead resistance, Zn-coating–substrate interface, electrolyte + separator ionic transport |
| **P2** | 3×10⁻⁵ – 3×10⁻⁴ s | Surface films on galvanized Zn (ZnO, Zn(OH)₂, ZHS) |
| **P3** | 3×10⁻⁴ – 10⁻³ s | Pore-network / transmission-line ion transport in hard carbon |
| **P4** | 10⁻³ – 2×10⁻² s | Hard-carbon cathode charge transfer + double layer |
| **P5** | 2×10⁻² – 0.1 s | Zn²⁺ desolvation and Zn/Zn²⁺ anode charge transfer |
| **P6** | 0.1 – 2 s | Solid-state/pore diffusion, Zn nucleation & growth, ZHS formation |
| **P7** | > 2 s | Very slow relaxation / low-frequency truncation edge — treat with caution |

Resistances are quoted as **Rᵢ = ∫γ dlnτ over the window**, not as peak height: area is
what corresponds to polarization resistance. The Peaks tab shows both.

---

## Exports

- **PDF** — full written report: theory, method, every figure, peak tables, KK
  validation, automatic interpretation and a measurement checklist.
- **Word (.docx)** — the same report, editable.
- **Excel (.xlsx)** — raw EIS, DRT curves, peak tables, window resistances and the
  embedded figures, one sheet group per sample, so every number behind every figure
  can be re-checked.
- **PNG** — all figures at publication DPI.

Projects save to a single `.drtproj` JSON file (**File ▸ Save**), including settings,
notes and results.

---

## Layout

```
DRT_Studio/
├── run_drt_studio.py        launcher (GUI / --selftest / --demo)
├── selftest.py              55 headless checks
└── drt_studio/
    ├── core/                no GUI imports — fully scriptable
    │   ├── compat.py         numpy 1.x / 2.x shim (trapz vs trapezoid)
    │   ├── diagnostics.py    environment + numerical self-check
    │   ├── model.py         Sample / Project / settings, JSON persistence
    │   ├── io_data.py       delimiter sniffing, column guessing, Re+Im merge
    │   ├── drt.py           kernel, Tikhonov solver, L-curve / GCV
    │   ├── kk.py            linear Kramers-Kronig test
    │   ├── peaks.py         detection, τ-windows, assignment, interpretation
    │   ├── plots.py         every figure
    │   ├── synth.py         synthetic demo cells
    │   ├── pipeline.py      analyse / render / export orchestration
    │   ├── export_excel.py  workbook writer
    │   └── export_report.py PDF + Word writer
    └── gui/
        ├── app.py           main window, 7 tabs, threaded work queue
        ├── import_dialog.py multi-file import with per-file column mapping
        └── widgets.py       plot pane, sortable table, scroll frame, theming
```

`core` never imports tkinter, so it can be scripted or run on a headless machine:

```python
from drt_studio.core.pipeline import build_demo_project, analyse_project
p = build_demo_project()
analyse_project(p)
print(p.samples[0].results["combined"].r_pol)
```

---

## Good practice

1. Run the Kramers-Kronig test before interpreting anything.
2. Measure down to 1–10 mHz so the diffusion peak is fully inside the window.
3. Use the same λ for every spectrum you intend to compare.
4. Quote the Re+Im combined result as primary.
5. Confirm assignments with temperature and SOC series — a charge-transfer peak shifts
   strongly with both; an ohmic one barely moves.
6. Quantify by integrating γ over a window, never by peak height.

---

## Notes on window sizing

The app measures your screen at start-up and never opens a window larger than it,
so the controls cannot end up off the desktop on a 1366×768 or 1024×600 display.
The settings sidebar on the Analysis tab scrolls (mouse wheel works over it) while
the **Analyse** buttons stay pinned to the bottom of the sidebar, and the
**Import**/**Cancel** buttons of the import dialog are always visible.
