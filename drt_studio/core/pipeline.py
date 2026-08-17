"""
High-level orchestration shared by the GUI and the command line.

Keeps the GUI thin: analysis, figure rendering and export bundles all live here
and are fully testable without a display.
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import numpy as np

from .model import MODES, MODE_SHORT, DRTSettings, Project, Sample, ensure_dir
from .drt import compute_drt
from .kk import lin_kk
from .peaks import ZINC_WINDOWS, analyse
from . import plots
from .export_excel import export_workbook
from .export_report import (comparison_report_docx, comparison_report_pdf,
                            sample_report_docx, sample_report_pdf)

Progress = Optional[Callable[[str, float], None]]


def _tick(cb: Progress, msg: str, frac: float):
    if cb:
        try:
            cb(msg, frac)
        except Exception:
            pass


# --------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------

def analyse_sample(sample: Sample, settings: DRTSettings, modes: List[str] = None,
                   run_kk: bool = True, prominence_frac: float = 0.02,
                   progress: Progress = None) -> Sample:
    modes = modes or list(MODES)
    if not sample.has_eis:
        raise ValueError("Sample '%s' has no impedance data to analyse." % sample.name)

    if run_kk:
        _tick(progress, "%s: Kramers-Kronig test" % sample.name, 0.05)
        sample.kk = lin_kk(sample.f, sample.z_re, sample.z_im)

    n = max(len(modes), 1)
    for i, m in enumerate(modes):
        _tick(progress, "%s: inverting (%s)" % (sample.name, MODE_SHORT.get(m, m)),
              0.1 + 0.85 * i / n)
        res = compute_drt(sample.f, sample.z_re, sample.z_im, settings, m)
        analyse(res, prominence_frac=prominence_frac)
        sample.results[m] = res
    _tick(progress, "%s: done" % sample.name, 1.0)
    return sample


def analyse_project(project: Project, modes: List[str] = None, run_kk: bool = True,
                    prominence_frac: float = 0.02, progress: Progress = None) -> None:
    todo = [s for s in project.samples if s.has_eis]
    for i, s in enumerate(todo):
        def sub(msg, frac, i=i, n=len(todo)):
            _tick(progress, msg, (i + frac) / max(n, 1))
        analyse_sample(s, project.settings, modes, run_kk, prominence_frac, sub)


# --------------------------------------------------------------------------------------
# figure rendering to files
# --------------------------------------------------------------------------------------

def render_sample_figures(sample: Sample, outdir: str, dpi: int = 165,
                          windows=None) -> Dict[str, str]:
    """Render every per-sample figure to PNG.  Returns {key: path}."""
    from matplotlib.figure import Figure

    ensure_dir(outdir)
    windows = windows or ZINC_WINDOWS
    out: Dict[str, str] = {}
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in sample.name).strip()

    def save(fig, key):
        p = os.path.join(outdir, "%s_%s.png" % (safe, key))
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        out[key] = p

    if sample.results:
        n = len([m for m in MODES if m in sample.results])
        fig = Figure(figsize=(9.6, 2.2 + 2.7 * n))
        plots.fig_panels(fig, sample, windows=windows)
        save(fig, "panels")

        fig = Figure(figsize=(9.6, 2.2 + 2.7 * n))
        plots.fig_panels(fig, sample, windows=windows, zoom=True)
        save(fig, "panels_zoom")

        fig = Figure(figsize=(8.4, 4.4))
        plots.fig_overlay(fig, sample, windows=windows)
        save(fig, "overlay")

        fig = Figure(figsize=(8.0, 3.6))
        _bars_single(fig, sample, windows)
        save(fig, "bars")

        best = "combined" if "combined" in sample.results else list(sample.results)[0]
        fig = Figure(figsize=(6.6, 4.0))
        plots.fig_residuals(fig, sample, best)
        save(fig, "residuals")

    if sample.has_eis:
        fig = Figure(figsize=(6.4, 4.6))
        plots.fig_nyquist(fig, [sample], show_fit=True,
                          mode="combined" if "combined" in sample.results else "re")
        save(fig, "nyquist")

        fig = Figure(figsize=(6.4, 4.6))
        plots.fig_bode(fig, [sample])
        save(fig, "bode")

    if sample.kk is not None and sample.kk.res_re_pct.size:
        fig = Figure(figsize=(6.6, 3.8))
        plots.fig_kk(fig, sample)
        save(fig, "kk")
    return out


def _bars_single(fig, sample: Sample, windows):
    """Per-window resistance for one sample, all modes side by side."""
    fig.clear()
    ax = fig.add_subplot(111)
    have = [m for m in MODES if m in sample.results]
    keys = [w.key for w in windows]
    if not have:
        ax.axis("off")
        return fig
    from .model import MODE_COLORS
    width = 0.8 / len(have)
    for i, m in enumerate(have):
        r = sample.results[m]
        vals = [r.region_r.get(k, 0.0) for k in keys]
        xs = np.arange(len(keys)) + (i - (len(have) - 1) / 2) * width
        ax.bar(xs, vals, width=width, color=MODE_COLORS[m], label=MODE_SHORT[m])
        for x, v in zip(xs, vals):
            if v > 0.5:
                ax.text(x, v, "%.0f" % v, ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["%s\n%s" % (w.key, w.short) for w in windows], fontsize=7)
    ax.set_ylabel(r"$R_i$ / $\Omega$")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Resistance per assignment window - %s" % sample.name,
                 fontsize=10, weight="bold")
    fig.tight_layout()
    return fig


def render_comparison_figures(samples: List[Sample], outdir: str, mode: str = "combined",
                              dpi: int = 165, windows=None) -> Dict[str, str]:
    from matplotlib.figure import Figure

    ensure_dir(outdir)
    windows = windows or ZINC_WINDOWS
    out: Dict[str, str] = {}

    def save(fig, key):
        p = os.path.join(outdir, "comparison_%s.png" % key)
        fig.savefig(p, dpi=dpi, bbox_inches="tight", facecolor="white")
        out[key] = p

    n = max(len([s for s in samples if mode in s.results]), 1)
    fig = Figure(figsize=(9.6, 2.0 + 2.2 * n))
    plots.fig_compare_stacked(fig, samples, mode, windows)
    save(fig, "stacked")

    fig = Figure(figsize=(8.6, 4.6))
    plots.fig_compare_overlay(fig, samples, mode, windows)
    save(fig, "overlay")

    fig = Figure(figsize=(8.6, 4.0))
    plots.fig_compare_bars(fig, samples, mode, windows)
    save(fig, "bars")

    fig = Figure(figsize=(8.0, 0.9 + 0.5 * n))
    plots.fig_compare_heatmap(fig, samples, mode, windows)
    save(fig, "heatmap")

    fig = Figure(figsize=(9.0, 3.8))
    plots.fig_compare_summary(fig, samples, mode)
    save(fig, "summary")

    fig = Figure(figsize=(6.6, 4.8))
    plots.fig_compare_nyquist(fig, samples)
    save(fig, "nyquist")
    return out


# --------------------------------------------------------------------------------------
# export bundles
# --------------------------------------------------------------------------------------

def export_sample_bundle(sample: Sample, project: Project, outdir: str,
                         want_pdf: bool = True, want_docx: bool = False,
                         want_excel: bool = True, want_figures: bool = True,
                         progress: Progress = None) -> List[str]:
    ensure_dir(outdir)
    figdir = ensure_dir(os.path.join(outdir, "figures"))
    _tick(progress, "rendering figures", 0.15)
    figs = render_sample_figures(sample, figdir)
    made: List[str] = []
    safe = "".join(c if c.isalnum() or c in "-_ " else "_" for c in sample.name).strip()

    # PDF and Word need optional third-party packages.  A missing optional
    # dependency must never abort a bundle that can still deliver its figures
    # and workbook, so each writer is attempted independently and what could
    # not be produced is recorded in `skipped`.
    skipped: List[str] = []

    if want_pdf:
        _tick(progress, "writing PDF", 0.45)
        p = os.path.join(outdir, "DRT_report_%s.pdf" % safe)
        try:
            made.append(sample_report_pdf(sample, project, p, figs))
        except ImportError as exc:
            skipped.append("PDF report (%s) - pip install reportlab" % exc)
    if want_docx:
        _tick(progress, "writing Word document", 0.65)
        p = os.path.join(outdir, "DRT_report_%s.docx" % safe)
        try:
            made.append(sample_report_docx(sample, project, p, figs))
        except ImportError as exc:
            skipped.append("Word report (%s) - pip install python-docx" % exc)
    if want_excel:
        _tick(progress, "writing Excel workbook", 0.85)
        p = os.path.join(outdir, "DRT_data_%s.xlsx" % safe)
        one = Project(name=project.name, settings=project.settings,
                      operator=project.operator,
                      cell_description=project.cell_description)
        one.samples = [sample]
        try:
            made.append(export_workbook(one, p, {sample.name: list(figs.values())},
                                        include_comparison=False))
        except ImportError as exc:
            skipped.append("Excel workbook (%s) - pip install openpyxl" % exc)
    if want_figures:
        made.extend(sorted(figs.values()))
    if skipped:
        note = os.path.join(outdir, "SKIPPED.txt")
        with open(note, "w") as fh:
            fh.write("These outputs could not be produced because an optional\n"
                     "package is not installed:\n\n")
            for line in skipped:
                fh.write("  - %s\n" % line)
        made.append(note)
    _tick(progress, "done", 1.0)
    return made


def export_comparison_bundle(project: Project, samples: List[Sample], outdir: str,
                             mode: str = "combined", want_pdf: bool = True,
                             want_docx: bool = False, want_excel: bool = True,
                             want_figures: bool = True,
                             progress: Progress = None) -> List[str]:
    ensure_dir(outdir)
    figdir = ensure_dir(os.path.join(outdir, "figures"))
    _tick(progress, "rendering comparison figures", 0.2)
    figs = render_comparison_figures(samples, figdir, mode)
    made: List[str] = []
    skipped: List[str] = []

    if want_pdf:
        _tick(progress, "writing comparison PDF", 0.45)
        p = os.path.join(outdir, "DRT_comparison_report.pdf")
        try:
            made.append(comparison_report_pdf(project, samples, p, figs, mode))
        except ImportError as exc:
            skipped.append("comparison PDF (%s) - pip install reportlab" % exc)
    if want_docx:
        _tick(progress, "writing comparison Word document", 0.65)
        p = os.path.join(outdir, "DRT_comparison_report.docx")
        try:
            made.append(comparison_report_docx(project, samples, p, figs, mode))
        except ImportError as exc:
            skipped.append("comparison Word report (%s) - pip install python-docx" % exc)
    if want_excel:
        _tick(progress, "writing comparison workbook", 0.85)
        p = os.path.join(outdir, "DRT_comparison_data.xlsx")
        sub = Project(name=project.name, settings=project.settings,
                      operator=project.operator,
                      cell_description=project.cell_description)
        sub.samples = list(samples)
        allfigs = {"__comparison__": list(figs.values())}
        try:
            made.append(export_workbook(sub, p, allfigs, include_comparison=True))
        except ImportError as exc:
            skipped.append("comparison workbook (%s) - pip install openpyxl" % exc)
    if want_figures:
        made.extend(sorted(figs.values()))
    if skipped:
        note = os.path.join(outdir, "SKIPPED.txt")
        with open(note, "w") as fh:
            fh.write("These outputs could not be produced because an optional\n"
                     "package is not installed:\n\n")
            for line in skipped:
                fh.write("  - %s\n" % line)
        made.append(note)
    _tick(progress, "done", 1.0)
    return made


# --------------------------------------------------------------------------------------
# demo project
# --------------------------------------------------------------------------------------

def build_demo_project() -> Project:
    from .synth import DEMO_SAMPLES, make_spectrum

    p = Project(name="Demo - zinc-ion cells")
    notes = {
        "Cell A - baseline galvanized Zn": "Reference build, as-received galvanized zinc foil.",
        "Cell B - passivated anode": "Anode stored 4 weeks before assembly; thicker surface film.",
        "Cell C - thin hard-carbon cathode": "Cathode loading halved to shorten the diffusion path.",
        "Cell D - drifting (KK fail)": "Deliberately drifting cell - demonstrates the KK warning.",
    }
    for name, kw in DEMO_SAMPLES.items():
        f, zr, zi = make_spectrum(**kw)
        s = Sample(name=name, f=f, z_re=zr, z_im=zi, notes=notes.get(name, ""))
        s.source_files = ["<synthetic demo data>"]
        p.add(s)
    return p
