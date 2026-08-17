"""
All figures used by the GUI and by the exported reports.

Every function takes a matplotlib Figure and draws into it, so the same code
serves the on-screen canvases and the PNG files embedded in PDF/Word/Excel.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .model import MODES, MODE_COLORS, MODE_LABELS, MODE_SHORT, DRTResult, Sample
from .peaks import ZINC_WINDOWS, Window

GRID = dict(alpha=0.35, lw=0.4)


def _log_axis(ax, xlabel=r"relaxation time  $\tau$  /  s"):
    ax.set_xscale("log")
    ax.grid(which="both", **GRID)
    ax.set_xlabel(xlabel, fontsize=9.5)


def _band_strip(ax, windows: List[Window], tau_lo, tau_hi, freq_axis=True, fs=9.0):
    ax.set_xscale("log")
    ax.set_xlim(tau_lo, tau_hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.tick_params(labelbottom=False, length=0)
    span = np.log10(tau_hi) - np.log10(tau_lo)
    for w in windows:
        a, b = max(w.tau_lo, tau_lo), min(w.tau_hi, tau_hi)
        if b <= a:
            continue
        ax.axvspan(a, b, color=w.color)
        xm = np.sqrt(a * b)
        frac = (np.log10(b) - np.log10(a)) / max(span, 1e-9)
        # only label what will actually fit, otherwise the strip turns to mush
        if frac > 0.035:
            ax.text(xm, 0.70, w.key, ha="center", va="center", fontsize=fs,
                    weight="bold", color="#333333")
        if frac > 0.11:
            short = w.short if frac > 0.16 else w.short.split()[0]
            ax.text(xm, 0.25, short, ha="center", va="center",
                    fontsize=min(fs * 0.68, fs * 0.68 * frac / 0.16 + 3.2),
                    color="#555555")
        ax.axvline(a, color="white", lw=1.1)
    if freq_axis:
        sec = ax.secondary_xaxis(
            "top", functions=(lambda x: 1.0 / (2 * np.pi * np.where(x == 0, 1e-30, x)),
                              lambda f: 1.0 / (2 * np.pi * np.where(f == 0, 1e-30, f))))
        sec.set_xlabel(r"characteristic frequency  $f = 1/2\pi\tau$  /  Hz", fontsize=8.5,
                       labelpad=6)
        sec.tick_params(labelsize=7.5)


def _shade(ax, windows, tau_lo, tau_hi, alpha=0.5):
    for w in windows:
        a, b = max(w.tau_lo, tau_lo), min(w.tau_hi, tau_hi)
        if b > a:
            ax.axvspan(a, b, color=w.color, alpha=alpha, zorder=0)


# --------------------------------------------------------------------------------------
# single sample: the three inversions in separate panels
# --------------------------------------------------------------------------------------

def fig_panels(fig, sample: Sample, modes: Optional[List[str]] = None,
               windows: Optional[List[Window]] = None, zoom: bool = False,
               zoom_max: Optional[float] = None, annotate: bool = True,
               show_strip: bool = True):
    """The signature figure: one panel per inversion mode, shared tau axis."""
    windows = windows or ZINC_WINDOWS
    modes = [m for m in (modes or MODES) if m in sample.results]
    fig.clear()
    if not modes:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No DRT results yet.\nUse the Analysis tab to compute.",
                ha="center", va="center", fontsize=11, color="#777777")
        ax.axis("off")
        return fig

    taus = np.concatenate([sample.results[m].tau for m in modes])
    tau_lo, tau_hi = float(taus.min()), float(taus.max())

    n = len(modes)
    heights = ([0.40] if show_strip else []) + [1.0] * n
    gs = fig.add_gridspec(len(heights), 1, height_ratios=heights, hspace=0.12)
    row = 0
    if show_strip:
        _band_strip(fig.add_subplot(gs[0]), windows, tau_lo, tau_hi)
        row = 1

    if zoom and zoom_max is None:
        vals = []
        for m in modes:
            g = sample.results[m].gamma
            if g.size:
                vals.append(np.percentile(g, 75) * 3.0)
        zoom_max = max(vals) if vals else 1.0

    axes = []
    for i, m in enumerate(modes):
        r = sample.results[m]
        ax = fig.add_subplot(gs[row + i])
        axes.append(ax)
        _shade(ax, windows, tau_lo, tau_hi)
        ax.plot(r.tau, r.gamma, color=MODE_COLORS[m], lw=1.9, zorder=3)
        ax.fill_between(r.tau, 0, r.gamma, color=MODE_COLORS[m], alpha=0.13, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(tau_lo, tau_hi)
        top = zoom_max if zoom else (float(np.nanmax(r.gamma)) * 1.22 + 1e-9)
        ax.set_ylim(0, top)
        ax.grid(which="both", zorder=1, **GRID)
        ax.set_ylabel(r"$\gamma(\tau)$ / $\Omega$", fontsize=9)
        tag = "(%s)  %s" % ("abc"[i], MODE_LABELS[m])
        if r.imported:
            tag += "   [imported]"
        ax.text(0.008, 0.95, tag, transform=ax.transAxes, fontsize=9.5, weight="bold",
                color=MODE_COLORS[m], va="top", zorder=6,
                bbox=dict(fc="white", ec=MODE_COLORS[m], lw=0.8, alpha=0.93, pad=3))
        ax.text(0.992, 0.95, r"$R_{pol}\approx$ %.1f $\Omega$" % r.r_pol,
                transform=ax.transAxes, fontsize=9, ha="right", va="top", zorder=6,
                bbox=dict(fc="white", ec="#999999", lw=0.6, alpha=0.93, pad=2.5))
        if annotate:
            for p in r.peaks:
                if zoom and p.gamma_max > top:
                    ax.annotate("%s \u2191 %.0f $\\Omega$" % (p.label, p.gamma_max),
                                (p.tau, top * 0.86), ha="center", va="center",
                                fontsize=7.4, weight="bold", color=MODE_COLORS[m], zorder=7,
                                bbox=dict(fc="white", ec=MODE_COLORS[m], lw=0.6,
                                          alpha=0.9, pad=1.6))
                else:
                    ax.annotate(p.label, (p.tau, p.gamma_max), xytext=(0, 6),
                                textcoords="offset points", ha="center", fontsize=7.8,
                                weight="bold", color="#222222", zorder=5)
        if i < n - 1:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel(r"relaxation time  $\tau$  /  s", fontsize=10)
    return fig


def fig_overlay(fig, sample: Sample, windows: Optional[List[Window]] = None):
    """All available modes of one sample on a single axes."""
    windows = windows or ZINC_WINDOWS
    fig.clear()
    ax = fig.add_subplot(111)
    modes = [m for m in MODES if m in sample.results]
    if not modes:
        ax.text(0.5, 0.5, "No DRT results yet.", ha="center", va="center", color="#777777")
        ax.axis("off")
        return fig
    taus = np.concatenate([sample.results[m].tau for m in modes])
    _shade(ax, windows, float(taus.min()), float(taus.max()), alpha=0.35)
    for m in modes:
        r = sample.results[m]
        ax.plot(r.tau, r.gamma, color=MODE_COLORS[m], lw=1.9,
                label="%s  (R_pol %.1f $\\Omega$)" % (MODE_LABELS[m], r.r_pol))
    _log_axis(ax)
    ax.set_xlim(float(taus.min()), float(taus.max()))
    ax.set_ylabel(r"$\gamma(\tau)$ / $\Omega$")
    ax.set_title("All inversions - %s" % sample.name, fontsize=10, weight="bold")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# EIS views
# --------------------------------------------------------------------------------------

def fig_nyquist(fig, samples: List[Sample], show_fit: bool = True, mode: str = "combined"):
    fig.clear()
    ax = fig.add_subplot(111)
    for s in samples:
        if not s.has_eis:
            continue
        ax.plot(s.z_re, -s.z_im, "o", ms=3.4, color=s.color, label=s.name, alpha=0.85)
        r = s.results.get(mode)
        if show_fit and r is not None and r.z_fit_re.size:
            ax.plot(r.z_fit_re, -r.z_fit_im, "-", lw=1.3, color=s.color, alpha=0.75)
    ax.set_xlabel(r"$Z'$ / $\Omega$")
    ax.set_ylabel(r"$-Z''$ / $\Omega$")
    ax.grid(alpha=0.35, lw=0.4)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Nyquist plot (points = data, line = DRT reconstruction)",
                 fontsize=9.5, weight="bold")
    if len(samples) > 1:
        ax.legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def fig_bode(fig, samples: List[Sample]):
    fig.clear()
    ax1, ax2 = fig.subplots(2, 1, sharex=True)
    for s in samples:
        if not s.has_eis:
            continue
        ax1.loglog(s.f, s.z_mod, "o-", ms=2.6, lw=1.0, color=s.color, label=s.name)
        ph = np.degrees(np.arctan2(s.z_im, s.z_re))
        ax2.semilogx(s.f, ph, "o-", ms=2.6, lw=1.0, color=s.color)
    ax1.set_ylabel(r"$|Z|$ / $\Omega$")
    ax2.set_ylabel("phase / deg")
    ax2.set_xlabel("frequency / Hz")
    for a in (ax1, ax2):
        a.grid(which="both", alpha=0.35, lw=0.4)
    if len(samples) > 1:
        ax1.legend(fontsize=7.5)
    ax1.set_title("Bode plot", fontsize=9.5, weight="bold")
    fig.tight_layout()
    return fig


def fig_residuals(fig, sample: Sample, mode: str = "combined"):
    fig.clear()
    ax = fig.add_subplot(111)
    r = sample.results.get(mode)
    if r is None or r.res_re_pct.size == 0:
        ax.text(0.5, 0.5, "No fit residuals available.", ha="center", va="center",
                color="#777777")
        ax.axis("off")
        return fig
    ax.semilogx(sample.f, r.res_re_pct, "o-", ms=3, lw=1.0, color="#1f5fa8",
                label=r"$\Delta$Re / |Z|")
    ax.semilogx(sample.f, r.res_im_pct, "s-", ms=3, lw=1.0, color="#c62828",
                label=r"$\Delta$Im / |Z|")
    ax.axhline(0, color="k", lw=0.7)
    for y in (-1, 1):
        ax.axhline(y, color="#999999", lw=0.7, ls=":")
    ax.set_xlabel("frequency / Hz")
    ax.set_ylabel("relative residual / %")
    ax.grid(which="both", alpha=0.35, lw=0.4)
    ax.legend(fontsize=8)
    ax.set_title("DRT fit residuals - %s (%s), RMS %.2f %%"
                 % (sample.name, MODE_SHORT[mode], r.rms_pct), fontsize=9.5, weight="bold")
    fig.tight_layout()
    return fig


def fig_kk(fig, sample: Sample):
    fig.clear()
    ax = fig.add_subplot(111)
    kk = sample.kk
    if kk is None or kk.res_re_pct.size == 0:
        ax.text(0.5, 0.5, "Kramers-Kronig test has not been run.", ha="center",
                va="center", color="#777777")
        ax.axis("off")
        return fig
    ax.semilogx(sample.f, kk.res_re_pct, "o-", ms=3, lw=1.0, color="#1f5fa8",
                label="real residual")
    ax.semilogx(sample.f, kk.res_im_pct, "s-", ms=3, lw=1.0, color="#c62828",
                label="imag residual")
    ax.axhline(0, color="k", lw=0.7)
    for y in (-1, 1):
        ax.axhline(y, color="#999999", lw=0.7, ls=":")
    ax.fill_between(sample.f, -1, 1, color="#c8e6c9", alpha=0.4, zorder=0)
    ax.set_xlabel("frequency / Hz")
    ax.set_ylabel("KK residual / %")
    ax.grid(which="both", alpha=0.35, lw=0.4)
    ax.legend(fontsize=8)
    ax.set_title("Linear Kramers-Kronig test - %s (max |res| %.2f %%)"
                 % (sample.name, kk.max_abs_pct), fontsize=9.5, weight="bold")
    fig.tight_layout()
    return fig


def fig_lambda_scan(fig, lams, res_norm, sol_norm, chosen=None):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.loglog(res_norm, sol_norm, "o-", ms=3, lw=1.0, color="#37474f")
    if chosen is not None and 0 <= chosen < len(lams):
        ax.plot(res_norm[chosen], sol_norm[chosen], "*", ms=15, color="#c62828",
                label=r"chosen $\lambda$ = %.3g" % lams[chosen])
        ax.legend(fontsize=8)
    for i in range(0, len(lams), max(1, len(lams) // 8)):
        ax.annotate("%.0e" % lams[i], (res_norm[i], sol_norm[i]), fontsize=6.5,
                    xytext=(3, 3), textcoords="offset points", color="#666666")
    ax.set_xlabel("residual norm  ||Ax - b||")
    ax.set_ylabel("solution norm  ||Dx||")
    ax.grid(which="both", alpha=0.35, lw=0.4)
    ax.set_title("L-curve", fontsize=9.5, weight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# comparison figures
# --------------------------------------------------------------------------------------

def fig_compare_overlay(fig, samples: List[Sample], mode: str = "combined",
                        windows: Optional[List[Window]] = None, normalise: bool = False):
    windows = windows or ZINC_WINDOWS
    fig.clear()
    ax = fig.add_subplot(111)
    have = [s for s in samples if mode in s.results]
    if not have:
        ax.text(0.5, 0.5, "No sample has a %s result." % MODE_SHORT.get(mode, mode),
                ha="center", va="center", color="#777777")
        ax.axis("off")
        return fig
    taus = np.concatenate([s.results[mode].tau for s in have])
    _shade(ax, windows, float(taus.min()), float(taus.max()), alpha=0.32)
    for s in have:
        r = s.results[mode]
        g = r.gamma / max(r.r_pol, 1e-12) if normalise else r.gamma
        ax.plot(r.tau, g, lw=1.9, color=s.color,
                label="%s  (R_pol %.1f $\\Omega$)" % (s.name, r.r_pol))
    _log_axis(ax)
    ax.set_xlim(float(taus.min()), float(taus.max()))
    ax.set_ylabel(r"$\gamma(\tau)/R_{pol}$" if normalise else r"$\gamma(\tau)$ / $\Omega$")
    ax.set_title("Sample comparison - %s inversion%s"
                 % (MODE_SHORT.get(mode, mode), " (normalised)" if normalise else ""),
                 fontsize=10, weight="bold")
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    return fig


def fig_compare_stacked(fig, samples: List[Sample], mode: str = "combined",
                        windows: Optional[List[Window]] = None):
    """One panel per sample - the 'separate areas' layout applied to comparison."""
    windows = windows or ZINC_WINDOWS
    fig.clear()
    have = [s for s in samples if mode in s.results]
    if not have:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "No sample has a %s result." % MODE_SHORT.get(mode, mode),
                ha="center", va="center", color="#777777")
        ax.axis("off")
        return fig
    taus = np.concatenate([s.results[mode].tau for s in have])
    tau_lo, tau_hi = float(taus.min()), float(taus.max())
    n = len(have)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=[0.40] + [1.0] * n, hspace=0.12)
    _band_strip(fig.add_subplot(gs[0]), windows, tau_lo, tau_hi)
    gmax = max(float(np.nanmax(s.results[mode].gamma)) for s in have) * 1.15
    axes = []
    for i, s in enumerate(have):
        r = s.results[mode]
        ax = fig.add_subplot(gs[i + 1])
        axes.append(ax)
        _shade(ax, windows, tau_lo, tau_hi)
        ax.plot(r.tau, r.gamma, lw=1.9, color=s.color, zorder=3)
        ax.fill_between(r.tau, 0, r.gamma, color=s.color, alpha=0.13, zorder=2)
        ax.set_xscale("log")
        ax.set_xlim(tau_lo, tau_hi)
        ax.set_ylim(0, gmax)
        ax.grid(which="both", zorder=1, **GRID)
        ax.set_ylabel(r"$\gamma$ / $\Omega$", fontsize=8.5)
        ax.text(0.008, 0.93, s.name, transform=ax.transAxes, fontsize=9.5, weight="bold",
                color=s.color, va="top", zorder=6,
                bbox=dict(fc="white", ec=s.color, lw=0.8, alpha=0.93, pad=3))
        ax.text(0.992, 0.93, r"$R_{pol}$ %.1f $\Omega$" % r.r_pol, transform=ax.transAxes,
                fontsize=8.5, ha="right", va="top", zorder=6,
                bbox=dict(fc="white", ec="#999999", lw=0.6, alpha=0.9, pad=2.2))
        for p in r.peaks:
            ax.annotate(p.label, (p.tau, p.gamma_max), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=7.4,
                        weight="bold", color="#222222", zorder=5)
        if i < n - 1:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel(r"relaxation time  $\tau$  /  s", fontsize=10)
    return fig


def fig_compare_bars(fig, samples: List[Sample], mode: str = "combined",
                     windows: Optional[List[Window]] = None, percent: bool = False):
    windows = windows or ZINC_WINDOWS
    fig.clear()
    ax = fig.add_subplot(111)
    have = [s for s in samples if mode in s.results]
    if not have:
        ax.text(0.5, 0.5, "Nothing to compare.", ha="center", va="center", color="#777777")
        ax.axis("off")
        return fig
    keys = [w.key for w in windows]
    n = len(have)
    width = 0.8 / n
    for i, s in enumerate(have):
        r = s.results[mode]
        vals = [r.region_r.get(k, 0.0) for k in keys]
        if percent:
            tot = max(sum(vals), 1e-12)
            vals = [100 * v / tot for v in vals]
        xs = np.arange(len(keys)) + (i - (n - 1) / 2) * width
        ax.bar(xs, vals, width=width, color=s.color, label=s.name)
        for x, v in zip(xs, vals):
            if v > (1.0 if not percent else 2.0):
                ax.text(x, v, "%.0f" % v, ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(["%s\n%s" % (w.key, w.short) for w in windows], fontsize=7)
    ax.set_ylabel("share of R_pol / %" if percent else r"$R_i$ / $\Omega$")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7.5)
    ax.set_title("Polarisation resistance per time-constant window (%s)"
                 % MODE_SHORT.get(mode, mode), fontsize=10, weight="bold")
    fig.tight_layout()
    return fig


def fig_compare_summary(fig, samples: List[Sample], mode: str = "combined"):
    """R_inf / R_pol / R_total and dominant tau across samples."""
    fig.clear()
    have = [s for s in samples if mode in s.results]
    if not have:
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Nothing to compare.", ha="center", va="center", color="#777777")
        ax.axis("off")
        return fig
    ax1, ax2 = fig.subplots(1, 2)
    names = [s.name for s in have]
    xs = np.arange(len(have))
    rinf = [s.results[mode].r_inf for s in have]
    rpol = [s.results[mode].r_pol for s in have]
    ax1.bar(xs, rinf, 0.6, label=r"$R_\infty$", color="#90a4ae")
    ax1.bar(xs, rpol, 0.6, bottom=rinf, label=r"$R_{pol}$", color="#ef6c00")
    for x, a, b in zip(xs, rinf, rpol):
        ax1.text(x, a + b, "%.0f" % (a + b), ha="center", va="bottom", fontsize=7)
    ax1.set_xticks(xs)
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=7.5)
    ax1.set_ylabel(r"resistance / $\Omega$")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("Resistance budget", fontsize=9.5, weight="bold")

    for s in have:
        r = s.results[mode]
        if r.peaks:
            p = max(r.peaks, key=lambda q: q.resistance)
            ax2.scatter(p.tau, p.resistance, s=70, color=s.color, label=s.name, zorder=3)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$\tau$ of dominant peak / s")
    ax2.set_ylabel(r"its resistance / $\Omega$")
    ax2.grid(which="both", alpha=0.35, lw=0.4)
    ax2.legend(fontsize=7)
    ax2.set_title("Dominant process", fontsize=9.5, weight="bold")
    fig.tight_layout()
    return fig


def fig_compare_heatmap(fig, samples: List[Sample], mode: str = "combined",
                        windows: Optional[List[Window]] = None, percent: bool = True):
    windows = windows or ZINC_WINDOWS
    fig.clear()
    ax = fig.add_subplot(111)
    have = [s for s in samples if mode in s.results]
    if not have:
        ax.text(0.5, 0.5, "Nothing to compare.", ha="center", va="center", color="#777777")
        ax.axis("off")
        return fig
    keys = [w.key for w in windows]
    M = []
    for s in have:
        r = s.results[mode]
        v = np.array([r.region_r.get(k, 0.0) for k in keys], float)
        if percent:
            v = 100 * v / max(v.sum(), 1e-12)
        M.append(v)
    M = np.array(M)
    vmax = float(M.max()) if M.size else 1.0
    im = ax.imshow(M, aspect="auto", cmap="YlOrRd", vmin=0.0, vmax=max(vmax, 1e-9))
    ax.set_xticks(range(len(keys)))
    # wrap the window description so neighbouring labels cannot collide
    xlab = []
    for w in windows:
        words, lines, cur = w.short.split(), [], ""
        for wd in words:
            trial = (cur + " " + wd).strip()
            if len(trial) > 12 and cur:
                lines.append(cur)
                cur = wd
            else:
                cur = trial
        if cur:
            lines.append(cur)
        xlab.append("%s\n%s" % (w.key, "\n".join(lines)))
    ax.set_xticklabels(xlab, fontsize=7)
    ax.set_yticks(range(len(have)))
    ax.set_yticklabels([s.name for s in have], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            # white text on dark cells keeps every number readable
            shade = M[i, j] / max(vmax, 1e-9)
            ax.text(j, i, "%.0f" % M[i, j], ha="center", va="center", fontsize=7.5,
                    weight="bold" if shade > 0.6 else "normal",
                    color="white" if shade > 0.6 else "#222222")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("share of R_pol / %" if percent else r"$R_i$ / $\Omega$", fontsize=8)
    ax.set_title("Process fingerprint (%s)" % MODE_SHORT.get(mode, mode),
                 fontsize=10, weight="bold")
    fig.tight_layout()
    return fig


def fig_compare_nyquist(fig, samples: List[Sample]):
    return fig_nyquist(fig, samples, show_fit=False)


def save_fig(fig, path: str, dpi: int = 170):
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    return path
