"""
DRT Studio - main application window.

Tabs
----
Data       : samples, raw EIS views, notes
Analysis   : settings, run inversions, per-mode panels
Peaks      : peak table, window table, assignment reference
Quality    : Kramers-Kronig test, fit residuals, L-curve
Compare    : multi-sample tables and figures, comparison report
Export     : report/workbook generation
Log        : everything the program did
"""
from __future__ import annotations

import os
import queue
import threading
import traceback
import webbrowser
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import numpy as np

from ..core import plots
from ..core.model import (MODES, MODE_LABELS, MODE_SHORT, DRTSettings, Project,
                          Sample, ensure_dir)
from ..core.diagnostics import check_environment, missing_for, run_diagnostics
from ..core.lambda_select import LAMBDA_METHODS
from ..core.rbf import POOR_WITH_NONNEG, RBF_TYPES, SHAPE_CONTROLS
from ..core.peaks import (ZINC_WINDOWS, analyse as peaks_analyse, compare_results,
                          interpret)
from ..core.pipeline import (analyse_sample, build_demo_project,
                             export_comparison_bundle, export_sample_bundle,
                             render_comparison_figures, render_sample_figures)
from .import_dialog import ImportDialog
from .widgets import (ACCENT, DARK, LabeledEntry, PlotPane, ScrollFrame, SortableTree,
                      Tooltip, fit_to_screen, style_app)

APP_NAME = "DRT Studio"
VERSION = "1.0"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("%s %s - Distribution of Relaxation Times for battery impedance" %
                   (APP_NAME, VERSION))
        # never larger than the screen: on a 1366x768 laptop a hard-coded
        # 1420x900 window pushes the controls off the desktop
        fit_to_screen(self, 1420, 900, min_w=900, min_h=600)
        style_app(self)

        self.project = Project()
        self.current: Optional[str] = None
        self._q: "queue.Queue[tuple]" = queue.Queue()
        self._busy = False

        self._build_menu()
        self._build_header()
        # status bar is packed BEFORE the expanding notebook so that the
        # notebook gives up space first and the status bar always stays visible
        self._build_status()
        self._build_tabs()
        self.after(80, self._pump)
        self.protocol("WM_DELETE_WINDOW", self._quit)
        self.log("%s %s ready. Use Data > Import, or Help > Load demo project." %
                 (APP_NAME, VERSION))

    # ================================================================== chrome
    def _build_menu(self):
        m = tk.Menu(self)
        f = tk.Menu(m, tearoff=0)
        f.add_command(label="New project", command=self.new_project, accelerator="Ctrl+N")
        f.add_command(label="Open project...", command=self.open_project, accelerator="Ctrl+O")
        f.add_command(label="Save project as...", command=self.save_project,
                      accelerator="Ctrl+S")
        f.add_separator()
        f.add_command(label="Import data...", command=self.import_data, accelerator="Ctrl+I")
        f.add_separator()
        f.add_command(label="Exit", command=self._quit)
        m.add_cascade(label="File", menu=f)

        a = tk.Menu(m, tearoff=0)
        a.add_command(label="Analyse current sample", command=self.run_current,
                      accelerator="F5")
        a.add_command(label="Analyse all samples", command=self.run_all, accelerator="F6")
        a.add_separator()
        a.add_command(label="Kramers-Kronig test (current)", command=self.run_kk_current)
        m.add_cascade(label="Analysis", menu=a)

        e = tk.Menu(m, tearoff=0)
        e.add_command(label="Export current sample...", command=lambda: self.nb.select(self.tab_export))
        e.add_command(label="Export comparison...", command=lambda: self.nb.select(self.tab_cmp))
        m.add_cascade(label="Export", menu=e)

        h = tk.Menu(m, tearoff=0)
        h.add_command(label="Load demo project", command=self.load_demo)
        h.add_separator()
        h.add_command(label="Run self-check (verify the maths)...",
                      command=self.run_self_check)
        h.add_command(label="Environment / installed packages...",
                      command=self.show_environment)
        h.add_command(label="Quick start", command=self.show_help)
        h.add_command(label="About", command=self.show_about)
        m.add_cascade(label="Help", menu=h)
        self.config(menu=m)

        self.bind_all("<Control-n>", lambda e: self.new_project())
        self.bind_all("<Control-o>", lambda e: self.open_project())
        self.bind_all("<Control-s>", lambda e: self.save_project())
        self.bind_all("<Control-i>", lambda e: self.import_data())
        self.bind_all("<F5>", lambda e: self.run_current())
        self.bind_all("<F6>", lambda e: self.run_all())

    def _build_header(self):
        h = ttk.Frame(self, padding=(12, 8))
        h.pack(fill="x")
        ttk.Label(h, text="DRT Studio", style="Header.TLabel").pack(side="left")
        ttk.Label(h, text="   Distribution of relaxation times - EIS deconvolution, "
                          "peak assignment and reporting", style="Sub.TLabel").pack(side="left")
        self.lbl_proj = ttk.Label(h, text="", foreground="#666666")
        self.lbl_proj.pack(side="right")
        ttk.Separator(self, orient="horizontal").pack(fill="x")

    def _build_status(self):
        s = ttk.Frame(self, padding=(10, 4))
        s.pack(fill="x", side="bottom")
        self.pb = ttk.Progressbar(s, mode="determinate", length=190)
        self.pb.pack(side="right")
        self.lbl_status = ttk.Label(s, text="Ready", foreground="#444444")
        self.lbl_status.pack(side="left")

    def _build_tabs(self):
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=6)
        self.tab_data = self._tab_data()
        self.tab_ana = self._tab_analysis()
        self.tab_peaks = self._tab_peaks()
        self.tab_qual = self._tab_quality()
        self.tab_cmp = self._tab_compare()
        self.tab_export = self._tab_export()
        self.tab_log = self._tab_log()
        for t, n in ((self.tab_data, "  1. Data  "), (self.tab_ana, "  2. Analysis  "),
                     (self.tab_peaks, "  3. Peaks  "), (self.tab_qual, "  4. Quality  "),
                     (self.tab_cmp, "  5. Compare  "), (self.tab_export, "  6. Export  "),
                     (self.tab_log, "  Log  ")):
            self.nb.add(t, text=n)

    # ================================================================== tab 1
    def _tab_data(self):
        f = ttk.Frame(self.nb)
        pan = ttk.Panedwindow(f, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=6, pady=6)

        left = ttk.Frame(pan)
        pan.add(left, weight=1)
        bar = ttk.Frame(left)
        bar.pack(fill="x", pady=(0, 4))
        ttk.Button(bar, text="Import data...", command=self.import_data,
                   style="Accent.TButton").pack(side="left")
        ttk.Button(bar, text="Demo", command=self.load_demo).pack(side="left", padx=4)
        ttk.Button(bar, text="Remove", command=self.remove_sample).pack(side="left")

        box = ttk.LabelFrame(left, text="Samples (one group per cell)", padding=6,
                             style="Section.TLabelframe")
        box.pack(fill="both", expand=True)
        self.lst = tk.Listbox(box, exportselection=False, activestyle="dotbox")
        self.lst.pack(fill="both", expand=True)
        self.lst.bind("<<ListboxSelect>>", lambda e: self.select_sample())
        self.lst.bind("<Double-Button-1>", lambda e: self.rename_sample())

        info = ttk.LabelFrame(left, text="Sample details", padding=6,
                              style="Section.TLabelframe")
        info.pack(fill="x", pady=(6, 0))
        self.txt_info = tk.Text(info, height=9, wrap="word", font=("TkDefaultFont", 8),
                                relief="flat", background="#f7f9fb")
        self.txt_info.pack(fill="both", expand=True)

        nb = ttk.LabelFrame(left, text="Notes (included in the report)", padding=6,
                            style="Section.TLabelframe")
        nb.pack(fill="x", pady=(6, 0))
        self.txt_notes = tk.Text(nb, height=4, wrap="word", font=("TkDefaultFont", 8))
        self.txt_notes.pack(fill="x")
        ttk.Button(nb, text="Save note", command=self.save_note).pack(anchor="e", pady=(4, 0))

        right = ttk.Frame(pan)
        pan.add(right, weight=3)
        sub = ttk.Notebook(right)
        sub.pack(fill="both", expand=True)
        self.p_nyq = PlotPane(sub, figsize=(7.2, 5.2))
        self.p_bode = PlotPane(sub, figsize=(7.2, 5.2))
        self.p_table = ttk.Frame(sub)
        sub.add(self.p_nyq, text="Nyquist")
        sub.add(self.p_bode, text="Bode")
        sub.add(self.p_table, text="Data table")
        self.tbl_data = SortableTree(
            self.p_table, ["f (Hz)", "Z' (ohm)", "Z'' (ohm)", "-Z'' (ohm)", "|Z| (ohm)",
                           "phase (deg)"], [110, 110, 110, 110, 110, 110], height=22)
        self.tbl_data.pack(fill="both", expand=True, padx=4, pady=4)
        return f

    # ================================================================== tab 2
    def _tab_analysis(self):
        f = ttk.Frame(self.nb)
        pan = ttk.Panedwindow(f, orient="horizontal")
        pan.pack(fill="both", expand=True, padx=6, pady=6)

        # sidebar = fixed action footer + scrollable settings above it, so the
        # Analyse buttons stay reachable no matter how short the screen is
        side = ttk.Frame(pan)
        pan.add(side, weight=1)
        gb = ttk.Frame(side)
        gb.pack(side="bottom", fill="x", pady=6, padx=4)
        left = ScrollFrame(side)
        left.pack(side="top", fill="both", expand=True)
        p = left.body

        g0 = ttk.LabelFrame(p, text="Inversion modes", padding=8,
                            style="Section.TLabelframe")
        g0.pack(fill="x", pady=4, padx=4)
        self.v_modes = {}
        for m in MODES:
            v = tk.BooleanVar(value=True)
            self.v_modes[m] = v
            cb = ttk.Checkbutton(g0, text=MODE_LABELS[m], variable=v)
            cb.pack(anchor="w")
        Tooltip(g0, "Re-only is ill-conditioned and must also estimate R_inf. Im-only is "
                    "well conditioned but blind to R_inf. Combined uses all the information "
                    "and is the recommended reference.")

        g1 = ttk.LabelFrame(p, text="Discretisation", padding=8,
                            style="Section.TLabelframe")
        g1.pack(fill="x", pady=4, padx=4)
        row = ttk.Frame(g1)
        row.pack(anchor="w", pady=2)
        ttk.Label(row, text="engine").pack(side="left")
        self.v_engine = tk.StringVar(value="drttools")
        cbe = ttk.Combobox(row, textvariable=self.v_engine, state="readonly", width=12,
                           values=["drttools", "legacy"])
        cbe.pack(side="left", padx=6)
        Tooltip(cbe, "'drttools' uses the analytic RBF discretisation and roughness "
                     "matrix of DRTtools (Ciucci group) - this is the accurate one. "
                     "'legacy' is the original lightweight kernel, kept as an "
                     "independent cross-check.")
        row = ttk.Frame(g1)
        row.pack(anchor="w", pady=2)
        ttk.Label(row, text="RBF type").pack(side="left")
        self.v_rbf = tk.StringVar(value="Gaussian")
        cbr = ttk.Combobox(row, textvariable=self.v_rbf, state="readonly", width=18,
                           values=list(RBF_TYPES))
        cbr.pack(side="left", padx=6)
        Tooltip(cbr, "Basis used to discretise gamma(ln tau). Gaussian is the "
                     "DRTtools default. Cauchy and Inverse Quadric have very broad "
                     "tails and only work with non-negativity switched OFF.")
        self.e_ppd = LabeledEntry(g1, "points / decade", 10, 8,
                                  "Number of basis functions per decade of tau. "
                                  "Leave the extend at 0 to use the DRTtools native "
                                  "grid tau = 1/f.")
        self.e_ppd.pack(anchor="w", pady=2)
        row = ttk.Frame(g1)
        row.pack(anchor="w", pady=2)
        ttk.Label(row, text="shape control").pack(side="left")
        self.v_shape = tk.StringVar(value="FWHM Coefficient")
        ttk.Combobox(row, textvariable=self.v_shape, state="readonly", width=18,
                     values=list(SHAPE_CONTROLS)).pack(side="left", padx=6)
        self.e_fwhm = LabeledEntry(g1, "FWHM coeff / shape factor", 1.0, 8,
                                   "With 'FWHM Coefficient' the RBF full width at half "
                                   "maximum is this many times the tau spacing "
                                   "(DRTtools default 1.0). With 'Shape Factor' the "
                                   "value is used directly as epsilon.")
        self.e_fwhm.pack(anchor="w", pady=2)
        self.e_ext = LabeledEntry(g1, "extend (decades)", 0.0, 8,
                                  "Extend the tau grid beyond 1/w_max .. 1/w_min. "
                                  "0 = DRTtools native grid.")
        self.e_ext.pack(anchor="w", pady=2)
        row = ttk.Frame(g1)
        row.pack(anchor="w", pady=2)
        ttk.Label(row, text="basis (legacy)").pack(side="left")
        self.v_basis = tk.StringVar(value="gaussian")
        ttk.Combobox(row, textvariable=self.v_basis, state="readonly", width=12,
                     values=["gaussian", "rectangular"]).pack(side="left", padx=6)

        g2 = ttk.LabelFrame(p, text="Regularisation", padding=8,
                            style="Section.TLabelframe")
        g2.pack(fill="x", pady=4, padx=4)
        row = ttk.Frame(g2)
        row.pack(anchor="w", pady=2)
        ttk.Label(row, text="lambda mode").pack(side="left")
        self.v_lmode = tk.StringVar(value="GCV")
        cbl = ttk.Combobox(row, textvariable=self.v_lmode, state="readonly", width=12,
                           values=list(LAMBDA_METHODS))
        cbl.pack(side="left", padx=6)
        Tooltip(cbl, "How lambda is chosen. GCV / mGCV / rGCV are cross-validation "
                     "criteria, LC is the L-curve corner, re-im uses the real/imaginary "
                     "discrepancy and kf is k-fold CV. All follow Maradesa et al., "
                     "J. Electrochem. Soc. 170 (2023) 030502.")
        self.e_lam = LabeledEntry(g2, "lambda", 1e-3, 10,
                                  "Too small -> spurious oscillations (ringing). "
                                  "Too large -> genuine peaks merge.")
        self.e_lam.pack(anchor="w", pady=2)
        row = ttk.Frame(g2)
        row.pack(anchor="w", pady=2)
        ttk.Label(row, text="derivative order").pack(side="left")
        self.v_dord = tk.IntVar(value=2)
        ttk.Combobox(row, textvariable=self.v_dord, state="readonly", width=6,
                     values=[1, 2]).pack(side="left", padx=6)
        self.v_nn = tk.BooleanVar(value=True)
        ttk.Checkbutton(g2, text="non-negative gamma (physical)",
                        variable=self.v_nn).pack(anchor="w")
        self.v_ind = tk.BooleanVar(value=True)
        ttk.Checkbutton(g2, text="fit series inductance L", variable=self.v_ind).pack(anchor="w")
        row = ttk.Frame(g2)
        row.pack(anchor="w", pady=2)
        ttk.Label(row, text="weighting").pack(side="left")
        self.v_wt = tk.StringVar(value="unit")
        cbw = ttk.Combobox(row, textvariable=self.v_wt, state="readonly", width=13,
                           values=["unit", "modulus", "proportional"])
        cbw.pack(side="left", padx=6)
        Tooltip(cbw, "DRTtools solves the unweighted problem and its lambda scale "
                     "assumes that, so 'unit' is the default. Other weightings are "
                     "renormalised to keep lambda on the same scale.")

        g3 = ttk.LabelFrame(p, text="Peak detection", padding=8,
                            style="Section.TLabelframe")
        g3.pack(fill="x", pady=4, padx=4)
        self.e_prom = LabeledEntry(g3, "prominence (% of max)", 2.0, 8,
                                   "Minimum prominence for a maximum to count as a peak.")
        self.e_prom.pack(anchor="w", pady=2)
        self.v_kk = tk.BooleanVar(value=True)
        ttk.Checkbutton(g3, text="run Kramers-Kronig test first",
                        variable=self.v_kk).pack(anchor="w")

        ttk.Button(gb, text="Analyse current sample  (F5)", command=self.run_current,
                   style="Accent.TButton").pack(fill="x", pady=2)
        ttk.Button(gb, text="Analyse ALL samples  (F6)", command=self.run_all).pack(
            fill="x", pady=2)
        ttk.Button(gb, text="Reset to defaults", command=self.reset_settings).pack(
            fill="x", pady=2)

        right = ttk.Frame(pan)
        pan.add(right, weight=3)
        top = ttk.Frame(right)
        top.pack(fill="x")
        self.v_zoom = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="magnified view (clip tall peaks)", variable=self.v_zoom,
                        command=self.refresh_analysis).pack(side="left")
        ttk.Button(top, text="Save figure...", command=self.save_panels).pack(side="right")
        sub = ttk.Notebook(right)
        sub.pack(fill="both", expand=True, pady=(4, 0))
        self.p_panels = PlotPane(sub, figsize=(8.4, 7.0))
        self.p_over = PlotPane(sub, figsize=(8.4, 5.4))
        self.p_bars = PlotPane(sub, figsize=(8.4, 4.6))
        self.f_find = ttk.Frame(sub)
        sub.add(self.p_panels, text="Separate panels")
        sub.add(self.p_over, text="Overlay")
        sub.add(self.p_bars, text="Window resistances")
        sub.add(self.f_find, text="Findings")
        self.txt_find = tk.Text(self.f_find, wrap="word", font=("TkDefaultFont", 9),
                                padx=10, pady=10)
        fs = ttk.Scrollbar(self.f_find, orient="vertical", command=self.txt_find.yview)
        self.txt_find.configure(yscrollcommand=fs.set)
        self.txt_find.pack(side="left", fill="both", expand=True)
        fs.pack(side="right", fill="y")
        return f

    # ================================================================== tab 3
    def _tab_peaks(self):
        f = ttk.Frame(self.nb)
        top = ttk.Frame(f, padding=6)
        top.pack(fill="x")
        ttk.Label(top, text="Detected peaks", style="Sub.TLabel").pack(side="left")
        ttk.Button(top, text="Copy table", command=lambda: self.tbl_peaks.copy()).pack(
            side="right")
        ttk.Button(top, text="Export CSV...", command=self.export_peaks_csv).pack(
            side="right", padx=6)
        self.tbl_peaks = SortableTree(
            f, ["Mode", "Peak", "tau (s)", "f (Hz)", "gamma max (ohm)", "R_i (ohm)",
                "% of R_pol", "FWHM (dec)", "Assignment"],
            [70, 60, 105, 105, 115, 100, 90, 85, 210], height=11)
        self.tbl_peaks.pack(fill="both", expand=True, padx=6)

        mid = ttk.Frame(f, padding=(6, 8))
        mid.pack(fill="x")
        ttk.Label(mid, text="Resistance per assignment window",
                  style="Sub.TLabel").pack(side="left")
        self.tbl_win = SortableTree(
            f, ["Window", "tau range (s)", "Meaning", "Re (ohm)", "Im (ohm)",
                "Combined (ohm)"], [80, 150, 250, 100, 100, 110], height=8)
        self.tbl_win.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        ref = ttk.LabelFrame(f, text="Assignment reference (click a peak row above)",
                             padding=8, style="Section.TLabelframe")
        ref.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.txt_assign = tk.Text(ref, height=8, wrap="word", font=("TkDefaultFont", 9),
                                  background="#f7f9fb")
        self.txt_assign.pack(fill="both", expand=True)
        self.tbl_peaks.tree.bind("<<TreeviewSelect>>", self._show_assignment)
        return f

    # ================================================================== tab 4
    def _tab_quality(self):
        f = ttk.Frame(self.nb)
        bar = ttk.Frame(f, padding=6)
        bar.pack(fill="x")
        ttk.Button(bar, text="Run Kramers-Kronig test", command=self.run_kk_current,
                   style="Accent.TButton").pack(side="left")
        ttk.Button(bar, text="Compute L-curve", command=self.run_lcurve).pack(
            side="left", padx=6)
        ttk.Label(bar, text="   Always validate the spectrum before trusting a DRT.",
                  foreground="#777777").pack(side="left")
        sub = ttk.Notebook(f)
        sub.pack(fill="both", expand=True, padx=6, pady=6)
        self.p_kk = PlotPane(sub, figsize=(7.6, 4.6))
        self.p_res = PlotPane(sub, figsize=(7.6, 4.6))
        self.p_lc = PlotPane(sub, figsize=(7.6, 4.6))
        sub.add(self.p_kk, text="Kramers-Kronig")
        sub.add(self.p_res, text="Fit residuals")
        sub.add(self.p_lc, text="L-curve")
        self.txt_kk = tk.Text(f, height=7, wrap="word", font=("TkDefaultFont", 9),
                              background="#f7f9fb", padx=8, pady=6)
        self.txt_kk.pack(fill="x", padx=6, pady=(0, 6))
        return f

    # ================================================================== tab 5
    def _tab_compare(self):
        f = ttk.Frame(self.nb)
        top = ttk.LabelFrame(f, text="Select samples to compare", padding=8,
                             style="Section.TLabelframe")
        top.pack(fill="x", padx=6, pady=6)
        inner = ttk.Frame(top)
        inner.pack(fill="x")
        self.lst_cmp = tk.Listbox(inner, selectmode="extended", height=5,
                                  exportselection=False)
        self.lst_cmp.pack(side="left", fill="both", expand=True)
        side = ttk.Frame(inner)
        side.pack(side="left", padx=10)
        ttk.Button(side, text="Select all", command=self.cmp_select_all).pack(fill="x", pady=2)
        row = ttk.Frame(side)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text="mode").pack(side="left")
        self.v_cmode = tk.StringVar(value="combined")
        ttk.Combobox(row, textvariable=self.v_cmode, state="readonly", width=11,
                     values=list(MODES)).pack(side="left", padx=4)
        self.v_norm = tk.BooleanVar(value=False)
        ttk.Checkbutton(side, text="normalise by R_pol",
                        variable=self.v_norm).pack(anchor="w", pady=2)
        ttk.Button(side, text="COMPARE", command=self.do_compare,
                   style="Accent.TButton").pack(fill="x", pady=(6, 2))
        ttk.Button(side, text="Export comparison report...",
                   command=self.export_comparison).pack(fill="x", pady=2)

        sub = ttk.Notebook(f)
        sub.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.f_ctab = ttk.Frame(sub)
        self.p_cstack = PlotPane(sub, figsize=(8.6, 6.6))
        self.p_cover = PlotPane(sub, figsize=(8.6, 5.0))
        self.p_cbars = PlotPane(sub, figsize=(8.6, 4.6))
        self.p_cheat = PlotPane(sub, figsize=(8.6, 4.2))
        self.p_csum = PlotPane(sub, figsize=(8.6, 4.2))
        self.f_cfind = ttk.Frame(sub)
        sub.add(self.f_ctab, text="Table")
        sub.add(self.p_cstack, text="Separate panels")
        sub.add(self.p_cover, text="Overlay")
        sub.add(self.p_cbars, text="Window bars")
        sub.add(self.p_cheat, text="Fingerprint")
        sub.add(self.p_csum, text="Summary")
        sub.add(self.f_cfind, text="Findings")

        cols = (["Sample", "R_inf", "R_pol", "R_total", "dom. tau (s)", "dom. R", "RMS %"] +
                [w.key for w in ZINC_WINDOWS])
        self.tbl_cmp = SortableTree(self.f_ctab, cols,
                                    [150, 70, 80, 80, 95, 75, 65] + [60] * len(ZINC_WINDOWS),
                                    height=14)
        self.tbl_cmp.pack(fill="both", expand=True, padx=4, pady=4)
        ttk.Button(self.f_ctab, text="Export table to CSV...",
                   command=self.export_cmp_csv).pack(anchor="e", padx=6, pady=(0, 6))

        self.txt_cfind = tk.Text(self.f_cfind, wrap="word", font=("TkDefaultFont", 9),
                                 padx=10, pady=10)
        cs = ttk.Scrollbar(self.f_cfind, orient="vertical", command=self.txt_cfind.yview)
        self.txt_cfind.configure(yscrollcommand=cs.set)
        self.txt_cfind.pack(side="left", fill="both", expand=True)
        cs.pack(side="right", fill="y")
        return f

    # ================================================================== tab 6
    def _tab_export(self):
        f = ttk.Frame(self.nb)
        wrap = ttk.Frame(f, padding=14)
        wrap.pack(fill="both", expand=True)
        ttk.Label(wrap, text="Export reports and data", style="Header.TLabel").pack(anchor="w")
        ttk.Label(wrap, wraplength=900, foreground="#555555",
                  text=("Every export bundles the figures, the numeric tables and the "
                        "automatic interpretation. The Excel workbook always contains the raw "
                        "EIS data, the DRT curves, the peak tables and the window "
                        "resistances, so the numbers behind every figure can be "
                        "re-checked.")).pack(anchor="w", pady=(4, 12))

        g = ttk.LabelFrame(wrap, text="What to produce", padding=10,
                           style="Section.TLabelframe")
        g.pack(fill="x")
        self.v_pdf = tk.BooleanVar(value=True)
        self.v_docx = tk.BooleanVar(value=False)
        self.v_xlsx = tk.BooleanVar(value=True)
        self.v_png = tk.BooleanVar(value=True)
        ttk.Checkbutton(g, text="PDF report", variable=self.v_pdf).grid(row=0, column=0,
                                                                        sticky="w", padx=6)
        ttk.Checkbutton(g, text="Word report (.docx)", variable=self.v_docx).grid(
            row=0, column=1, sticky="w", padx=6)
        ttk.Checkbutton(g, text="Excel workbook (.xlsx)", variable=self.v_xlsx).grid(
            row=0, column=2, sticky="w", padx=6)
        ttk.Checkbutton(g, text="PNG figures", variable=self.v_png).grid(
            row=0, column=3, sticky="w", padx=6)

        g2 = ttk.LabelFrame(wrap, text="Report metadata", padding=10,
                            style="Section.TLabelframe")
        g2.pack(fill="x", pady=10)
        self.e_pname = LabeledEntry(g2, "project name", "DRT project", 40)
        self.e_pname.pack(anchor="w", pady=3)
        self.e_oper = LabeledEntry(g2, "operator", "", 40)
        self.e_oper.pack(anchor="w", pady=3)
        self.e_cell = LabeledEntry(
            g2, "cell description",
            "Aqueous zinc-ion cell: galvanized zinc anode, hard-carbon cathode, "
            "ZnSO4 electrolyte", 80)
        self.e_cell.pack(anchor="w", pady=3)

        g3 = ttk.LabelFrame(wrap, text="Generate", padding=10,
                            style="Section.TLabelframe")
        g3.pack(fill="x")
        ttk.Button(g3, text="Export CURRENT sample...", command=self.export_current,
                   style="Accent.TButton").pack(side="left", padx=4)
        ttk.Button(g3, text="Export ALL samples (one bundle each)...",
                   command=self.export_all).pack(side="left", padx=4)
        ttk.Button(g3, text="Export COMPARISON...",
                   command=self.export_comparison).pack(side="left", padx=4)

        g4 = ttk.LabelFrame(wrap, text="Verify", padding=10,
                            style="Section.TLabelframe")
        g4.pack(fill="x", pady=(10, 0))
        ttk.Button(g4, text="Run self-check", command=self.run_self_check).pack(side="left")
        ttk.Button(g4, text="Environment...", command=self.show_environment).pack(
            side="left", padx=6)
        ttk.Label(g4, foreground="#777777", wraplength=760,
                  text=("  Inverts a synthetic cell with a known answer and checks the "
                        "recovered resistances, peak positions, area conservation and "
                        "Kramers-Kronig test.")).pack(side="left")

        self.txt_export = tk.Text(wrap, height=16, wrap="word", font=("TkDefaultFont", 8),
                                  background="#f7f9fb")
        self.txt_export.pack(fill="both", expand=True, pady=(12, 0))
        return f

    def _tab_log(self):
        f = ttk.Frame(self.nb)
        self.txt_log = tk.Text(f, wrap="word", font=("TkFixedFont", 8), background="#1e1e1e",
                               foreground="#d4d4d4", insertbackground="white")
        sb = ttk.Scrollbar(f, orient="vertical", command=self.txt_log.yview)
        self.txt_log.configure(yscrollcommand=sb.set)
        self.txt_log.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        return f

    # ================================================================== helpers
    def log(self, msg: str):
        import datetime
        line = "[%s] %s\n" % (datetime.datetime.now().strftime("%H:%M:%S"), msg)
        try:
            self.txt_log.insert("end", line)
            self.txt_log.see("end")
        except Exception:
            pass

    def status(self, msg: str, frac: Optional[float] = None):
        self.lbl_status.configure(text=msg)
        if frac is not None:
            self.pb["value"] = max(0, min(100, frac * 100))
        self.update_idletasks()

    def _pump(self):
        """Drain messages from worker threads."""
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "status":
                    msg, frac = payload
                    self.status(msg, frac)
                elif kind == "log":
                    self.log(payload)
                elif kind == "done":
                    fn = payload
                    self._busy = False
                    self.pb["value"] = 0
                    try:
                        fn()
                    except Exception:
                        self.log("post-task error: " + traceback.format_exc())
                elif kind == "error":
                    self._busy = False
                    self.pb["value"] = 0
                    self.status("Error")
                    self.log("ERROR: " + payload)
                    messagebox.showerror("Error", payload[:1500])
        except queue.Empty:
            pass
        self.after(80, self._pump)

    def _run_bg(self, fn, done=None, label="Working"):
        if self._busy:
            messagebox.showinfo(APP_NAME, "A task is already running, please wait.")
            return
        self._busy = True
        self.status(label, 0.02)

        def worker():
            try:
                res = fn(lambda m, f: self._q.put(("status", (m, f))))
                self._q.put(("done", (lambda: done(res)) if done else (lambda: None)))
            except Exception:
                self._q.put(("error", traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()

    def sample(self) -> Optional[Sample]:
        if self.current is None:
            return None
        return self.project.get(self.current)

    def settings_from_ui(self) -> DRTSettings:
        st = DRTSettings()
        st.engine = self.v_engine.get()
        st.rbf_type = self.v_rbf.get()
        st.shape_control = self.v_shape.get()
        st.points_per_decade = max(2, self.e_ppd.get(int, 10) or 10)
        st.rbf_fwhm_factor = max(0.05, self.e_fwhm.get(float, 1.0) or 1.0)
        st.tau_extend_decades = max(0.0, self.e_ext.get(float, 0.0) or 0.0)
        st.basis = self.v_basis.get()
        st.lambda_mode = self.v_lmode.get()
        st.lambda_value = max(1e-12, self.e_lam.get(float, 1e-3) or 1e-3)
        st.derivative_order = int(self.v_dord.get())
        st.non_negative = bool(self.v_nn.get())
        st.include_inductance = bool(self.v_ind.get())
        st.induct_used = 1 if st.include_inductance else 0
        st.weighting = self.v_wt.get()
        return st

    def reset_settings(self):
        d = DRTSettings()
        self.v_engine.set(d.engine)
        self.v_rbf.set(d.rbf_type)
        self.v_shape.set(d.shape_control)
        self.e_ppd.set(d.points_per_decade)
        self.e_fwhm.set(d.rbf_fwhm_factor)
        self.e_ext.set(d.tau_extend_decades)
        self.v_basis.set(d.basis)
        self.v_lmode.set(d.lambda_mode)
        self.e_lam.set(d.lambda_value)
        self.v_dord.set(d.derivative_order)
        self.v_nn.set(d.non_negative)
        self.v_ind.set(d.include_inductance)
        self.v_wt.set(d.weighting)
        self.e_prom.set(2.0)
        self.log("Settings reset to defaults.")

    # ================================================================== project
    def new_project(self):
        if self.project.samples and not messagebox.askyesno(
                APP_NAME, "Discard the current project?"):
            return
        self.project = Project()
        self.current = None
        self.refresh_all()
        self.log("New project.")

    def open_project(self):
        p = filedialog.askopenfilename(title="Open project",
                                       filetypes=[("DRT project", "*.drtproj *.json"),
                                                  ("All files", "*.*")])
        if not p:
            return
        try:
            self.project = Project.load(p)
            self.current = self.project.names[0] if self.project.samples else None
            self.refresh_all()
            self.log("Opened project %s (%d samples)." % (p, len(self.project.samples)))
        except Exception as ex:
            messagebox.showerror(APP_NAME, "Could not open project:\n%s" % ex)

    def save_project(self):
        if not self.project.samples:
            messagebox.showinfo(APP_NAME, "Nothing to save.")
            return
        p = filedialog.asksaveasfilename(title="Save project", defaultextension=".drtproj",
                                         filetypes=[("DRT project", "*.drtproj")])
        if not p:
            return
        self._sync_meta()
        try:
            self.project.save(p)
            self.log("Saved project to %s" % p)
            self.status("Project saved")
        except Exception as ex:
            messagebox.showerror(APP_NAME, "Could not save:\n%s" % ex)

    def _sync_meta(self):
        self.project.name = self.e_pname.get(str, "DRT project") or "DRT project"
        self.project.operator = self.e_oper.get(str, "") or ""
        self.project.cell_description = self.e_cell.get(str, "") or ""
        self.project.settings = self.settings_from_ui()

    def load_demo(self):
        self.project = build_demo_project()
        self.current = self.project.names[0]
        self.e_pname.set(self.project.name)
        self.refresh_all()
        self.log("Demo project loaded: 4 synthetic zinc-ion cells. "
                 "Press F6 to analyse them all.")
        messagebox.showinfo(APP_NAME,
                            "Demo project loaded with 4 synthetic cells.\n\n"
                            "Press F6 (or Analysis > Analyse all samples) to run the "
                            "inversions, then open the Compare tab and press COMPARE.")

    def import_data(self):
        dlg = ImportDialog(self, self.project.names)
        self.wait_window(dlg)
        if not dlg.samples:
            return
        for s in dlg.samples:
            self.project.add(s)
            self.log("Imported '%s': %s" % (s.name, s.describe()))
            if s.meta.get("merge"):
                self.log("   Re/Im merge: %s" % s.meta["merge"])
            if s.meta.get("imag_sign"):
                self.log("   %s" % s.meta["imag_sign"])
        self.current = dlg.samples[0].name
        self.refresh_all()

    def remove_sample(self):
        s = self.sample()
        if s is None:
            return
        if messagebox.askyesno(APP_NAME, "Remove sample '%s'?" % s.name):
            self.project.remove(s.name)
            self.current = self.project.names[0] if self.project.samples else None
            self.refresh_all()

    def rename_sample(self):
        s = self.sample()
        if s is None:
            return
        from tkinter.simpledialog import askstring
        new = askstring(APP_NAME, "New name:", initialvalue=s.name, parent=self)
        if new and new.strip():
            old, s.name = s.name, new.strip()
            self.current = s.name
            self.refresh_all()
            self.log("Renamed '%s' -> '%s'" % (old, s.name))

    def save_note(self):
        s = self.sample()
        if s is None:
            return
        s.notes = self.txt_notes.get("1.0", "end").strip()
        self.log("Note saved for '%s'." % s.name)

    # ================================================================== refresh
    def refresh_all(self):
        names = self.project.names
        self.lst.delete(0, "end")
        for n in names:
            self.lst.insert("end", n)
        self.lst_cmp.delete(0, "end")
        for n in names:
            self.lst_cmp.insert("end", n)
        if self.current in names:
            i = names.index(self.current)
            self.lst.selection_clear(0, "end")
            self.lst.selection_set(i)
        self.lbl_proj.configure(text="%d sample(s)" % len(names))
        self.select_sample(from_refresh=True)

    def select_sample(self, from_refresh=False):
        if not from_refresh:
            sel = self.lst.curselection()
            if sel:
                self.current = self.project.names[sel[0]]
        s = self.sample()
        self.txt_info.delete("1.0", "end")
        self.txt_notes.delete("1.0", "end")
        if s is None:
            for p in (self.p_nyq, self.p_bode, self.p_panels, self.p_over, self.p_bars,
                      self.p_kk, self.p_res):
                p.clear()
            self.tbl_data.set_rows([])
            self.tbl_peaks.set_rows([])
            self.tbl_win.set_rows([])
            return

        lines = ["Name:  %s" % s.name]
        if s.has_eis:
            lines += ["Points: %d" % s.n_points, "Range:  %s" % s.f_range,
                      "|Z|:    %.4g - %.4g ohm" % (s.z_mod.min(), s.z_mod.max())]
        if s.source_files:
            lines.append("Files:  " + "; ".join(os.path.basename(x) for x in s.source_files))
        for k, v in s.meta.items():
            lines.append("%s: %s" % (k, v))
        if s.results:
            lines.append("")
            for m in MODES:
                if m in s.results:
                    r = s.results[m]
                    lines.append("%-9s R_inf %7.2f  R_pol %8.2f  peaks %d  RMS %.2f%%"
                                 % (MODE_SHORT[m], r.r_inf, r.r_pol, len(r.peaks), r.rms_pct))
        if s.kk:
            lines.append("")
            lines.append("KK max |res| %.2f %%" % s.kk.max_abs_pct)
        self.txt_info.insert("1.0", "\n".join(lines))
        self.txt_notes.insert("1.0", s.notes or "")

        if s.has_eis:
            plots.fig_nyquist(self.p_nyq.figure, [s], show_fit=True,
                              mode="combined" if "combined" in s.results else "re")
            self.p_nyq.draw()
            plots.fig_bode(self.p_bode.figure, [s])
            self.p_bode.draw()
            ph = np.degrees(np.arctan2(s.z_im, s.z_re))
            self.tbl_data.set_rows([
                ["%.6g" % a, "%.6g" % b, "%.6g" % c, "%.6g" % (-c),
                 "%.6g" % np.hypot(b, c), "%.3f" % d]
                for a, b, c, d in zip(s.f, s.z_re, s.z_im, ph)])
        self.refresh_analysis()
        self.refresh_peaks()
        self.refresh_quality()

    def refresh_analysis(self):
        s = self.sample()
        if s is None:
            return
        plots.fig_panels(self.p_panels.figure, s, zoom=bool(self.v_zoom.get()))
        self.p_panels.draw()
        plots.fig_overlay(self.p_over.figure, s)
        self.p_over.draw()
        from ..core.pipeline import _bars_single
        _bars_single(self.p_bars.figure, s, ZINC_WINDOWS)
        self.p_bars.draw()

        self.txt_find.delete("1.0", "end")
        if not s.results:
            self.txt_find.insert("1.0", "Run the analysis to see automatic findings.")
            return
        buf = ["AUTOMATIC FINDINGS - %s\n%s\n" % (s.name, "=" * 60)]
        for m in MODES:
            if m in s.results:
                buf.append("\n%s\n%s" % (MODE_LABELS[m], "-" * 60))
                for line in interpret(s.results[m]):
                    buf.append("  * " + line)
        cl = compare_results(s.results)
        if cl:
            buf.append("\nCROSS-CHECK BETWEEN INVERSIONS\n" + "-" * 60)
            for line in cl:
                buf.append("  * " + line)
        self.txt_find.insert("1.0", "\n".join(buf))

    def refresh_peaks(self):
        s = self.sample()
        rows = []
        if s:
            for m in MODES:
                if m not in s.results:
                    continue
                r = s.results[m]
                for p in r.peaks:
                    rows.append([MODE_SHORT[m], p.label, "%.4g" % p.tau, "%.4g" % p.freq,
                                 "%.3f" % p.gamma_max, "%.3f" % p.resistance,
                                 "%.1f" % (100 * p.resistance / max(r.r_pol, 1e-12)),
                                 "%.2f" % p.fwhm_decades, p.assignment])
        self.tbl_peaks.set_rows(rows)

        wrows = []
        if s:
            for w in ZINC_WINDOWS:
                row = [w.key, "%.0e - %.0e" % (w.tau_lo, w.tau_hi), w.short]
                for m in MODES:
                    row.append("%.3f" % s.results[m].region_r.get(w.key, 0.0)
                               if m in s.results else "-")
                wrows.append(row)
        self.tbl_win.set_rows(wrows)

    def _show_assignment(self, _=None):
        sel = self.tbl_peaks.tree.selection()
        self.txt_assign.delete("1.0", "end")
        if not sel:
            return
        vals = self.tbl_peaks.tree.item(sel[0], "values")
        key = str(vals[1])[:2]
        w = next((x for x in ZINC_WINDOWS if x.key == key), None)
        if w is None:
            return
        txt = ["%s  -  %s" % (w.key, w.short),
               "tau window: %.0e to %.0e s   (f = %.3g to %.3g Hz)"
               % (w.tau_lo, w.tau_hi, 1 / (2 * np.pi * w.tau_hi), 1 / (2 * np.pi * w.tau_lo)),
               "", "PROCESS", w.process, "", "DETAIL", w.detail,
               "", "OBSERVED HERE",
               "  %s at tau = %s s, gamma_max = %s ohm, R = %s ohm (%s %% of R_pol), "
               "FWHM %s decades, mode %s"
               % (vals[1], vals[2], vals[4], vals[5], vals[6], vals[7], vals[0])]
        self.txt_assign.insert("1.0", "\n".join(txt))

    def refresh_quality(self):
        s = self.sample()
        if s is None:
            return
        plots.fig_kk(self.p_kk.figure, s)
        self.p_kk.draw()
        best = "combined" if (s.results and "combined" in s.results) else (
            list(s.results)[0] if s.results else "combined")
        plots.fig_residuals(self.p_res.figure, s, best)
        self.p_res.draw()
        self.txt_kk.delete("1.0", "end")
        if s.kk:
            self.txt_kk.insert("1.0",
                               "KRAMERS-KRONIG VALIDATION\n%s\n\n%s\n\nmax |residual| = %.3f %%"
                               ", mean = %.3f %%, %d RC elements."
                               % ("-" * 70, s.kk.verdict, s.kk.max_abs_pct,
                                  s.kk.mean_abs_pct, s.kk.n_rc))
        else:
            self.txt_kk.insert("1.0", "Run the Kramers-Kronig test to validate this spectrum "
                                      "before trusting the DRT.")

    # ================================================================== actions
    def run_current(self):
        s = self.sample()
        if s is None:
            messagebox.showinfo(APP_NAME, "Import or select a sample first.")
            return
        if not s.has_eis:
            # An imported gamma(tau) curve is already a DRT: there is nothing to
            # invert, but peaks/windows/assignment are still meaningful, so
            # refresh them instead of dead-ending the user.
            n = 0
            for m, r in s.results.items():
                if getattr(r, "imported", False):
                    peaks_analyse(r, prominence_frac=max(
                        0.001, (self.e_prom.get(float, 2.0) or 2.0) / 100.0))
                    n += 1
            self.select_sample(from_refresh=True)
            self.nb.select(self.tab_peaks)
            self.log("'%s' is an imported DRT curve; re-detected peaks on %d curve(s)."
                     % (s.name, n))
            self.status("Imported DRT curve - peaks updated", 1.0)
            messagebox.showinfo(
                APP_NAME,
                "'%s' holds an imported DRT curve, so there is no impedance to "
                "invert.\n\nIts peaks, window resistances and assignments have been "
                "recomputed and are shown on the Peaks tab. Kramers-Kronig and fit "
                "residuals need raw impedance and stay unavailable." % s.name)
            return
        modes = [m for m in MODES if self.v_modes[m].get()]
        if not modes:
            messagebox.showinfo(APP_NAME, "Select at least one inversion mode.")
            return
        st = self.settings_from_ui()
        if not self._check_rbf_combo(st):
            return
        self.project.settings = st
        prom = max(0.001, (self.e_prom.get(float, 2.0) or 2.0) / 100.0)
        kk = bool(self.v_kk.get())
        self.log("Analysing '%s' (%s), %s" % (s.name, "+".join(modes), st.describe()))

        def task(cb):
            analyse_sample(s, st, modes, kk, prom, cb)
            return s

        def done(_):
            self.status("Analysis complete", 1.0)
            self.log("Done '%s': %s" % (s.name, ", ".join(
                "%s R_pol=%.2f" % (MODE_SHORT[m], s.results[m].r_pol)
                for m in modes if m in s.results)))
            self.select_sample(from_refresh=True)
        self._run_bg(task, done, "Analysing %s" % s.name)

    def run_all(self):
        todo = [s for s in self.project.samples if s.has_eis]
        prom0 = max(0.001, (self.e_prom.get(float, 2.0) or 2.0) / 100.0)
        # imported gamma(tau) curves cannot be inverted, but their peaks can be
        # (re)detected so they still take part in the tables and comparison
        imported = [s for s in self.project.samples if not s.has_eis and s.results]
        for s in imported:
            for r in s.results.values():
                if getattr(r, "imported", False):
                    peaks_analyse(r, prominence_frac=prom0)
        if not todo:
            if imported:
                self.select_sample(from_refresh=True)
                self.log("No invertible spectra; refreshed peaks on %d imported "
                         "DRT curve(s)." % len(imported))
                messagebox.showinfo(
                    APP_NAME,
                    "This project only contains imported DRT curves, which are "
                    "already deconvoluted.\n\nTheir peaks and window resistances "
                    "have been refreshed. Import raw impedance (f, Re, Im) to run "
                    "an inversion.")
            else:
                messagebox.showinfo(APP_NAME, "No samples with impedance data.")
            return
        modes = [m for m in MODES if self.v_modes[m].get()]
        st = self.settings_from_ui()
        if not self._check_rbf_combo(st):
            return
        self.project.settings = st
        prom = max(0.001, (self.e_prom.get(float, 2.0) or 2.0) / 100.0)
        kk = bool(self.v_kk.get())
        self.log("Analysing %d samples (%s)" % (len(todo), "+".join(modes)))

        def task(cb):
            for i, s in enumerate(todo):
                analyse_sample(s, st, modes, kk, prom,
                               lambda m, fr, i=i: cb(m, (i + fr) / len(todo)))
            return todo

        def done(_):
            self.status("All samples analysed", 1.0)
            for s in todo:
                self.log("  %s: %s" % (s.name, ", ".join(
                    "%s R_pol=%.2f" % (MODE_SHORT[m], s.results[m].r_pol)
                    for m in MODES if m in s.results)))
            self.select_sample(from_refresh=True)
        self._run_bg(task, done, "Analysing all samples")

    def run_kk_current(self):
        s = self.sample()
        if s is None or not s.has_eis:
            messagebox.showinfo(APP_NAME, "Select a sample with impedance data.")
            return
        from ..core.kk import lin_kk

        def task(cb):
            cb("Kramers-Kronig", 0.4)
            s.kk = lin_kk(s.f, s.z_re, s.z_im)
            return s

        def done(_):
            self.refresh_quality()
            self.log("KK '%s': max %.3f %% - %s" % (s.name, s.kk.max_abs_pct,
                                                    s.kk.verdict.split(" - ")[0]))
            self.nb.select(self.tab_qual)
        self._run_bg(task, done, "Kramers-Kronig test")

    def run_lcurve(self):
        s = self.sample()
        if s is None or not s.has_eis:
            messagebox.showinfo(APP_NAME, "Select a sample with impedance data.")
            return
        st = self.settings_from_ui()
        from ..core.drt import _assemble, _solve_one, derivative_matrix, make_tau_grid

        def task(cb):
            tau = make_tau_grid(s.f, st)
            M, b, wv, m, n_extra, _, _ = _assemble(s.f, s.z_re, s.z_im, tau, st, "combined")
            D = derivative_matrix(m, st.derivative_order)
            n = m + n_extra
            Df = np.zeros((D.shape[0], n))
            Df[:, :m] = D
            lams = np.logspace(np.log10(st.lambda_min), np.log10(st.lambda_max),
                               st.lambda_steps)
            rn, sn = [], []
            Mw = M * wv[:, None]
            bw = b * wv
            for i, lam in enumerate(lams):
                cb("L-curve %d/%d" % (i + 1, len(lams)), i / len(lams))
                x = _solve_one(M, b, wv, m, n_extra, lam, st)
                rn.append(float(np.linalg.norm(Mw @ x - bw)))
                sn.append(float(np.linalg.norm(Df @ x)))
            lr, ls = np.log10(np.maximum(rn, 1e-30)), np.log10(np.maximum(sn, 1e-30))
            d1r, d1s = np.gradient(lr), np.gradient(ls)
            d2r, d2s = np.gradient(d1r), np.gradient(d1s)
            k = np.abs(d1r * d2s - d2r * d1s) / np.power(d1r ** 2 + d1s ** 2, 1.5) + 1e-30
            return lams, np.array(rn), np.array(sn), int(np.argmax(k))

        def done(res):
            lams, rn, sn, idx = res
            plots.fig_lambda_scan(self.p_lc.figure, lams, rn, sn, idx)
            self.p_lc.draw()
            self.e_lam.set("%.4g" % lams[idx])
            self.v_lmode.set("manual")
            self.log("L-curve corner at lambda = %.4g (applied to the settings)." % lams[idx])
            self.nb.select(self.tab_qual)
        self._run_bg(task, done, "Computing L-curve")

    # ================================================================== compare
    def cmp_select_all(self):
        self.lst_cmp.selection_set(0, "end")

    def _cmp_samples(self) -> List[Sample]:
        sel = self.lst_cmp.curselection()
        names = self.project.names
        chosen = [names[i] for i in sel] if sel else names
        return [self.project.get(n) for n in chosen if self.project.get(n)]

    def do_compare(self):
        samples = self._cmp_samples()
        mode = self.v_cmode.get()
        have = [s for s in samples if mode in s.results]
        if len(have) < 1:
            messagebox.showinfo(APP_NAME,
                                "None of the selected samples has a '%s' result yet.\n\n"
                                "Run the analysis first (F6 analyses everything)."
                                % MODE_SHORT.get(mode, mode))
            return
        self.log("Comparing %d samples using the %s inversion."
                 % (len(have), MODE_SHORT.get(mode, mode)))

        rows = []
        for s in have:
            r = s.results[mode]
            dom = max(r.peaks, key=lambda p: p.resistance) if r.peaks else None
            row = [s.name, "%.2f" % r.r_inf, "%.2f" % r.r_pol,
                   "%.2f" % (r.r_inf + r.r_pol),
                   "%.4g" % dom.tau if dom else "-",
                   "%.2f" % dom.resistance if dom else "-", "%.2f" % r.rms_pct]
            row += ["%.2f" % r.region_r.get(w.key, 0.0) for w in ZINC_WINDOWS]
            rows.append(row)
        self.tbl_cmp.set_rows(rows)

        norm = bool(self.v_norm.get())
        plots.fig_compare_stacked(self.p_cstack.figure, have, mode)
        self.p_cstack.draw()
        plots.fig_compare_overlay(self.p_cover.figure, have, mode, normalise=norm)
        self.p_cover.draw()
        plots.fig_compare_bars(self.p_cbars.figure, have, mode)
        self.p_cbars.draw()
        plots.fig_compare_heatmap(self.p_cheat.figure, have, mode)
        self.p_cheat.draw()
        plots.fig_compare_summary(self.p_csum.figure, have, mode)
        self.p_csum.draw()

        self.txt_cfind.delete("1.0", "end")
        self.txt_cfind.insert("1.0", self._comparison_text(have, mode))
        self.status("Comparison ready", 1.0)

    def _comparison_text(self, have: List[Sample], mode: str) -> str:
        buf = ["SAMPLE COMPARISON - %s inversion" % MODE_SHORT.get(mode, mode),
               "=" * 72, ""]
        by_tot = sorted(have, key=lambda s: s.results[mode].r_inf + s.results[mode].r_pol)
        best, worst = by_tot[0], by_tot[-1]
        rb = best.results[mode].r_inf + best.results[mode].r_pol
        rw = worst.results[mode].r_inf + worst.results[mode].r_pol
        buf.append("Ranking by total resistance (best first):")
        for i, s in enumerate(by_tot, 1):
            r = s.results[mode]
            buf.append("  %d. %-34s R_tot %8.2f ohm   (R_inf %6.2f + R_pol %8.2f)"
                       % (i, s.name, r.r_inf + r.r_pol, r.r_inf, r.r_pol))
        buf.append("")
        buf.append("Spread: %s is a factor %.2f better than %s."
                   % (best.name, rw / max(rb, 1e-12), worst.name))
        buf.append("")
        buf.append("WHICH PROCESS EXPLAINS THE DIFFERENCE")
        buf.append("-" * 72)
        diffs = []
        for w in ZINC_WINDOWS:
            a = best.results[mode].region_r.get(w.key, 0.0)
            b = worst.results[mode].region_r.get(w.key, 0.0)
            diffs.append((abs(b - a), w, a, b))
        diffs.sort(reverse=True, key=lambda x: x[0])
        for d, w, a, b in diffs[:3]:
            if d < 0.05:
                continue
            buf.append("  %s (%s): %.2f ohm in %s vs %.2f ohm in %s  ->  gap %.2f ohm"
                       % (w.key, w.short, a, best.name, b, worst.name, d))
            buf.append("      %s" % w.process)
        buf.append("")
        buf.append("PER-WINDOW SPREAD ACROSS THE SET")
        buf.append("-" * 72)
        for w in ZINC_WINDOWS:
            vals = [s.results[mode].region_r.get(w.key, 0.0) for s in have]
            if max(vals) < 0.05:
                continue
            buf.append("  %-4s %-26s min %7.2f  max %7.2f  mean %7.2f  ohm"
                       % (w.key, w.short, min(vals), max(vals), float(np.mean(vals))))
        buf.append("")
        buf.append("PER-SAMPLE DIGEST")
        buf.append("-" * 72)
        for s in have:
            buf.append("")
            buf.append("* %s" % s.name)
            if s.notes:
                buf.append("    note: %s" % s.notes)
            if s.kk:
                buf.append("    KK: max %.2f %% - %s" % (s.kk.max_abs_pct,
                                                         s.kk.verdict.split(" - ")[0]))
            for line in interpret(s.results[mode]):
                buf.append("    - " + line)
        return "\n".join(buf)

    # ================================================================== export
    def _ask_dir(self, title):
        return filedialog.askdirectory(title=title, mustexist=False)

    # ------------------------------------------------------------ diagnostics
    def _check_export_libs(self, opts) -> bool:
        """Refuse an export up-front if the library it needs is not installed."""
        bad = missing_for(want_pdf=opts.get("want_pdf"),
                          want_docx=opts.get("want_docx"),
                          want_excel=opts.get("want_excel"))
        if not bad:
            return True
        messagebox.showerror(
            APP_NAME,
            "This export cannot run:\n\n  - %s\n\nUntick that format, or install "
            "the package and restart." % "\n  - ".join(bad))
        for b in bad:
            self.log("export blocked: " + b)
        return False

    def _check_rbf_combo(self, st) -> bool:
        """Warn about RBF / non-negativity combinations that cannot work."""
        if (st.engine == "drttools" and st.non_negative
                and st.rbf_type in POOR_WITH_NONNEG):
            return messagebox.askokcancel(
                APP_NAME,
                "The %s basis has very broad tails. Reproducing a sharp DRT peak "
                "with it needs negative coefficients, so with 'non-negative gamma' "
                "switched on the fit will be poor (typically >90 %% residual and a "
                "badly underestimated R_pol).\n\n"
                "Either untick 'non-negative gamma (physical)', or use Gaussian, "
                "a Matern, or Inverse Quadratic.\n\nRun anyway?" % st.rbf_type)
        return True

    def show_environment(self):
        rep = check_environment()
        _TextWindow(self, "Environment", rep.text())
        self.log("Environment check: %d ok, %d missing."
                 % (rep.n_pass, rep.n_fail))

    def run_self_check(self):
        """Numerical end-to-end verification against a known circuit."""
        def task(cb):
            return run_diagnostics(progress=cb)

        def done(rep):
            _TextWindow(self, "Self-check - %d passed, %d failed"
                        % (rep.n_pass, rep.n_fail), rep.text())
            self.log("Self-check: %d passed, %d failed (%.1f s)"
                     % (rep.n_pass, rep.n_fail, rep.seconds))
            self.status("Self-check: %d passed, %d failed"
                        % (rep.n_pass, rep.n_fail), 1.0)
        self._run_bg(task, done, "Running self-check")

    def export_current(self):
        s = self.sample()
        if s is None:
            messagebox.showinfo(APP_NAME, "Select a sample first.")
            return
        if not s.results:
            messagebox.showinfo(APP_NAME, "Analyse the sample before exporting.")
            return
        opts = dict(want_pdf=self.v_pdf.get(), want_docx=self.v_docx.get(),
                    want_excel=self.v_xlsx.get(), want_figures=self.v_png.get())
        if not self._check_export_libs(opts):
            return
        d = self._ask_dir("Choose an output folder")
        if not d:
            return
        self._sync_meta()

        def task(cb):
            return export_sample_bundle(s, self.project, d, progress=cb, **opts)
        self._run_bg(task, lambda made: self._after_export(made, d),
                     "Exporting %s" % s.name)

    def export_all(self):
        done = [s for s in self.project.samples if s.results]
        if not done:
            messagebox.showinfo(APP_NAME, "Analyse at least one sample first.")
            return
        opts = dict(want_pdf=self.v_pdf.get(), want_docx=self.v_docx.get(),
                    want_excel=self.v_xlsx.get(), want_figures=self.v_png.get())
        if not self._check_export_libs(opts):
            return
        d = self._ask_dir("Choose an output folder")
        if not d:
            return
        self._sync_meta()

        def task(cb):
            made = []
            for i, s in enumerate(done):
                sd = ensure_dir(os.path.join(
                    d, "".join(c if c.isalnum() or c in "-_ " else "_" for c in s.name).strip()))
                made += export_sample_bundle(
                    s, self.project, sd,
                    progress=lambda m, f, i=i: cb(m, (i + f) / len(done)), **opts)
            return made
        self._run_bg(task, lambda made: self._after_export(made, d), "Exporting all samples")

    def export_comparison(self):
        samples = self._cmp_samples()
        mode = self.v_cmode.get()
        have = [s for s in samples if mode in s.results]
        if len(have) < 2:
            messagebox.showinfo(APP_NAME,
                                "Select at least two analysed samples in the Compare tab.")
            return
        opts = dict(want_pdf=self.v_pdf.get(), want_docx=self.v_docx.get(),
                    want_excel=self.v_xlsx.get(), want_figures=self.v_png.get())
        if not self._check_export_libs(opts):
            return
        d = self._ask_dir("Choose an output folder for the comparison")
        if not d:
            return
        self._sync_meta()

        def task(cb):
            return export_comparison_bundle(self.project, have, d, mode, progress=cb, **opts)
        self._run_bg(task, lambda made: self._after_export(made, d), "Exporting comparison")

    def _after_export(self, made: List[str], folder: str):
        self.status("Export complete", 1.0)
        self.txt_export.delete("1.0", "end")
        self.txt_export.insert("1.0", "Created %d file(s) in\n%s\n\n%s"
                               % (len(made), folder,
                                  "\n".join("  " + os.path.relpath(m, folder) for m in made)))
        for m in made:
            self.log("wrote %s" % m)
        self.nb.select(self.tab_export)
        if messagebox.askyesno(APP_NAME, "Export finished (%d files).\n\nOpen the folder?"
                                         % len(made)):
            try:
                webbrowser.open("file://" + os.path.abspath(folder))
            except Exception:
                pass

    def export_peaks_csv(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if p:
            self.tbl_peaks.to_csv(p)
            self.log("Peak table exported to %s" % p)

    def export_cmp_csv(self):
        p = filedialog.asksaveasfilename(defaultextension=".csv",
                                         filetypes=[("CSV", "*.csv")])
        if p:
            self.tbl_cmp.to_csv(p)
            self.log("Comparison table exported to %s" % p)

    def save_panels(self):
        p = filedialog.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"),
                                                    ("SVG", "*.svg")])
        if p:
            self.p_panels.save(p)
            self.log("Figure saved to %s" % p)

    # ================================================================== help
    def show_help(self):
        txt = (
            "QUICK START\n\n"
            "1. Data tab > Import data...\n"
            "   Add your files. Set each file's ROLE:\n"
            "     full - one file with f, Re(Z), Im(Z)\n"
            "     re   - a file with f and Re(Z) only\n"
            "     im   - a file with f and Im(Z) only\n"
            "     drt  - a precomputed tau/gamma curve\n"
            "   Files sharing a GROUP name become ONE sample, so a separate Re file and "
            "Im file are merged automatically.\n\n"
            "2. Analysis tab > choose the inversion modes and the regularisation, then "
            "press F5 (current sample) or F6 (all samples).\n\n"
            "3. Peaks tab shows every detected peak with its physical assignment; click a "
            "row for the full explanation.\n\n"
            "4. Quality tab runs the Kramers-Kronig test - always check this before "
            "trusting a DRT - and draws the L-curve to pick lambda objectively.\n\n"
            "5. Compare tab: select samples, press COMPARE, then export a separate "
            "comparison report.\n\n"
            "6. Export tab: PDF and/or Word reports plus an Excel workbook containing "
            "every number behind the figures.\n\n"
            "Tip: Help > Load demo project gives you four synthetic cells to explore.")
        _TextWindow(self, "Quick start", txt)

    def show_about(self):
        messagebox.showinfo(
            "About " + APP_NAME,
            "%s %s\n\nDistribution of relaxation times for electrochemical impedance "
            "spectroscopy.\n\nRegularised DRT inversion (Re / Im / combined), Kramers-Kronig "
            "validation, automatic peak detection and assignment for aqueous zinc-ion cells, "
            "multi-sample comparison and PDF / Word / Excel reporting."
            % (APP_NAME, VERSION))

    def _quit(self):
        if self._busy and not messagebox.askyesno(APP_NAME, "A task is running. Quit anyway?"):
            return
        self.destroy()


class _TextWindow(tk.Toplevel):
    def __init__(self, master, title, text):
        super().__init__(master)
        self.title(title)
        fit_to_screen(self, 760, 560, min_w=420, min_h=320)
        t = tk.Text(self, wrap="word", font=("TkDefaultFont", 9), padx=14, pady=12)
        sb = ttk.Scrollbar(self, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=sb.set)
        t.insert("1.0", text)
        t.configure(state="disabled")
        t.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")


def main():
    app = App()
    app.mainloop()
