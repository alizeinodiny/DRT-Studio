"""
Peak detection, integration and physical assignment.

The assignment library encodes the time-constant windows discussed in the
report for an aqueous zinc-ion cell (galvanized Zn anode / hard-carbon cathode /
ZnSO4 electrolyte).  Windows are editable by the user at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from .compat import trapezoid
from scipy.signal import find_peaks

from .model import DRTResult, Peak


# --------------------------------------------------------------------------------------
# assignment library
# --------------------------------------------------------------------------------------

@dataclass
class Window:
    key: str
    tau_lo: float
    tau_hi: float
    short: str
    process: str
    detail: str
    color: str


ZINC_WINDOWS: List[Window] = [
    Window("P1", 1e-7, 3e-5, "Contact / separator",
           "Contact and current-collector resistance, electrolyte and separator ionic transport",
           "Includes the Zn-coating/steel-substrate interface of a galvanized anode and any "
           "residual cable inductance not absorbed into L. Weakly temperature activated "
           "(0.1-0.2 eV), insensitive to SOC. Growth on cycling indicates coating delamination "
           "or electrode dry-out.",
           "#ffdf99"),
    Window("P2", 3e-5, 3e-4, "Zn surface film",
           "Zn(2+) migration through surface films on the galvanized zinc",
           "Native ZnO / Zn(OH)2 plus basic zinc sulfate (ZHS, Zn4SO4(OH)6.nH2O) formed in the "
           "mildly acidic ZnSO4 electrolyte. The aqueous-Zn analogue of the Li-ion SEI peak. "
           "Strongly activated (0.4-0.7 eV) and grows with calendar time - the peak to watch "
           "for zinc passivation.",
           "#d3eed3"),
    Window("P3", 3e-4, 1e-3, "Carbon pore transport",
           "Ionic transport in the porous hard-carbon layer (transmission-line response)",
           "A porous electrode has no single time constant but a continuum, appearing in the DRT "
           "as a broad flat band rather than a spike. Scales with electrode thickness and "
           "porosity, not with temperature.",
           "#d3eed3"),
    Window("P4", 1e-3, 2e-2, "Cathode CT + dl",
           "Hard-carbon cathode: double-layer charging, ion adsorption and charge transfer",
           "Charge storage in hard carbon in aqueous Zn cells is dominated by surface processes: "
           "double-layer charging, adsorption of Zn(2+)/H(+)/SO4(2-) and faradaic reaction at "
           "oxygen surface groups. The most SOC-sensitive feature of the cathode; activated "
           "0.3-0.5 eV.",
           "#c9e6f5"),
    Window("P5", 2e-2, 1.5e-1, "Zn CT / desolvation",
           "Zn(2+) desolvation and Zn/Zn(2+) charge transfer at the anode",
           "[Zn(H2O)6](2+) must shed its hydration shell before the two-electron transfer. "
           "Desolvation is a documented slow step in mildly acidic ZnSO4 and pushes the anodic "
           "charge transfer to 10-100 ms, much slower than a Li-ion charge-transfer peak. "
           "Verify with a symmetric Zn//Zn cell.",
           "#ded8f7"),
    Window("P6", 1.5e-1, 2.0, "Diffusion / nucleation / ZHS",
           "Solid-state and pore diffusion, Zn nucleation and growth, ZHS precipitation",
           "Three overlapping mechanisms: (i) intra-particle diffusion of Zn(2+)/H(+) in hard "
           "carbon (tau = L^2/D with D ~ 1e-11..1e-13 cm2/s); (ii) Zn nucleation, adatom surface "
           "diffusion and the 2D->3D growth transition; (iii) precipitation/dissolution of zinc "
           "hydroxide sulfate as local pH swings. Usually the dominant polarisation of the cell.",
           "#ffcfcf"),
    Window("P7", 2.0, 1e6, "Slow relaxation / edge",
           "Concentration polarisation, slow ZHS ripening, parasitic reactions - and the "
           "truncation edge of the tau window",
           "Physically this region holds electrolyte concentration polarisation, ZHS ripening and "
           "parasitic hydrogen evolution / self-discharge. WARNING: features within about one "
           "decade of the edge of the tau window are extrapolated rather than measured and should "
           "not be quoted unless the EIS is extended to 1-10 mHz.",
           "#eed9f7"),
]


def window_for(tau: float, windows: List[Window] = None) -> Window:
    ws = windows or ZINC_WINDOWS
    for w in ws:
        if w.tau_lo <= tau < w.tau_hi:
            return w
    return ws[-1] if tau >= ws[-1].tau_lo else ws[0]


# --------------------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------------------

def detect_peaks(res: DRTResult, prominence_frac: float = 0.02,
                 min_resistance: float = 0.0,
                 windows: List[Window] = None) -> List[Peak]:
    """Find maxima of gamma, integrate each between neighbouring minima, assign."""
    tau, g = res.tau, res.gamma
    if tau.size < 5 or not np.any(g > 0):
        return []
    lt = np.log(tau)
    total = float(trapezoid(g, lt))
    prom = max(prominence_frac * float(np.nanmax(g)), 1e-12)
    idx, props = find_peaks(g, prominence=prom)

    # --- reject grid-locked ringing -------------------------------------
    # A regularised solution can oscillate at exactly the tau-grid period,
    # which slices one broad physical peak into many slivers (measured: 8
    # maxima spaced 0.1004 decade apart with std 0.0000, against a grid
    # spacing of 0.1037 decade).  Those are discretisation artefacts, not
    # processes, and counting them makes the peak areas sum to far less than
    # R_pol.  Peaks whose separation is locked to the grid and whose dividing
    # valley is shallow are therefore merged into the stronger neighbour.
    if idx.size > 1:
        keep = list(idx)
        merged = True
        while merged and len(keep) > 1:
            merged = False
            for i in range(len(keep) - 1):
                a, b = keep[i], keep[i + 1]
                valley = float(np.min(g[a:b + 1]))
                lo = min(float(g[a]), float(g[b]))
                # A genuine pair of processes is separated by a real valley.
                # Ringing leaves only a shallow dimple: measured 4-10 % deep on
                # an over-smoothed synthetic spectrum whose maxima were spaced
                # by exactly the basis-function spacing.  Require the valley to
                # drop at least 20 % below the weaker maximum, otherwise the two
                # maxima belong to one physical peak.
                if lo > 0 and valley > 0.80 * lo:
                    keep.pop(i if g[a] < g[b] else i + 1)
                    merged = True
                    break
        idx = np.array(sorted(keep), dtype=int)

    # --- integration bounds --------------------------------------------
    # Each peak owns the span between the deepest points separating it from
    # its neighbours, and the outermost peaks own everything out to the ends
    # of the grid.  Walking downhill from the maximum instead (the obvious
    # implementation) stops at the first tiny inflexion, so the areas then
    # cover only a sliver of the curve and sum to far less than R_pol.
    bounds = []
    for j, p in enumerate(idx):
        if j == 0:
            l = 0
        else:
            l = int(np.argmin(g[idx[j - 1]:p + 1])) + idx[j - 1]
        if j == len(idx) - 1:
            r = len(g) - 1
        else:
            r = int(np.argmin(g[p:idx[j + 1] + 1])) + p
        bounds.append((l, r))

    peaks: List[Peak] = []
    for p, (l, r) in zip(idx, bounds):
        if r - l < 2:
            continue
        area = float(trapezoid(g[l:r + 1], lt[l:r + 1]))
        if area < min_resistance:
            continue
        # An over-regularised solution is numerically flat (gamma ~ 1e-2 ohm);
        # its residual ripple must not be reported as structure.  Require each
        # peak to carry a meaningful share of the total area.
        if total > 1e-9 and area < 0.005 * total:
            continue
        # FWHM in decades
        half = g[p] / 2.0
        a = p
        while a > l and g[a] > half:
            a -= 1
        bb = p
        while bb < r and g[bb] > half:
            bb += 1
        fwhm = float(np.log10(tau[bb]) - np.log10(tau[a]))
        win = window_for(float(tau[p]), windows)
        peaks.append(Peak(
            label=win.key, tau=float(tau[p]), freq=float(1.0 / (2 * np.pi * tau[p])),
            gamma_max=float(g[p]), resistance=area,
            tau_lo=float(tau[l]), tau_hi=float(tau[r]), fwhm_decades=fwhm,
            assignment=win.short, process=win.process,
        ))
    peaks.sort(key=lambda q: q.tau)
    # disambiguate repeated labels: P6 -> P6a, P6b ...
    from collections import Counter
    cnt = Counter(p.label for p in peaks)
    seen: Dict[str, int] = {}
    for p in peaks:
        if cnt[p.label] > 1:
            i = seen.get(p.label, 0)
            p.label = "%s%s" % (p.label, "abcdefgh"[i] if i < 8 else str(i))
            seen[p.label[:2]] = i + 1
    return peaks


def _integrate_window(lt: np.ndarray, g: np.ndarray, lo: float, hi: float) -> float:
    """
    Integrate gamma d(ln tau) over [lo, hi] with INTERPOLATED end points.

    Selecting points with a boolean mask silently drops the trapezoid that
    straddles each window boundary, so the windows no longer add up to R_pol
    (about 4 % was disappearing across the seven windows).  Clipping the curve
    and interpolating gamma exactly at the two edges conserves the area.
    """
    a, b = np.log(lo), np.log(hi)
    a = max(a, lt[0])
    b = min(b, lt[-1])
    if not (b > a):
        return 0.0
    inner = lt[(lt > a) & (lt < b)]
    xs = np.concatenate(([a], inner, [b]))
    ys = np.interp(xs, lt, g)
    return float(trapezoid(ys, xs))


def region_resistances(res: DRTResult, windows: List[Window] = None) -> Dict[str, float]:
    """Integrate gamma over each assignment window (edge-conserving)."""
    ws = windows or ZINC_WINDOWS
    tau, g = res.tau, res.gamma
    out: Dict[str, float] = {}
    if tau.size < 2:
        return out
    o = np.argsort(tau)
    lt, g = np.log(np.asarray(tau, float)[o]), np.asarray(g, float)[o]
    for w in ws:
        out[w.key] = _integrate_window(lt, g, w.tau_lo, w.tau_hi)
    return out


def analyse(res: DRTResult, prominence_frac: float = 0.02,
            windows: List[Window] = None) -> DRTResult:
    res.peaks = detect_peaks(res, prominence_frac, windows=windows)
    res.region_r = region_resistances(res, windows)
    return res


# --------------------------------------------------------------------------------------
# automatic interpretation text
# --------------------------------------------------------------------------------------

def ringing_score(res: DRTResult) -> Tuple[float, str]:
    """
    Heuristic detector for Tikhonov ringing.

    Genuine processes are not periodic in log(tau); regularisation oscillation is.
    Returns (score 0..1, verdict text).
    """
    pk = [p for p in res.peaks if p.resistance > 0]
    if len(pk) < 4:
        return 0.0, "No sign of periodic oscillation (too few peaks to judge)."
    lt = np.log10([p.tau for p in pk])
    d = np.diff(lt)
    if d.size < 3:
        return 0.0, "No sign of periodic oscillation."
    cv = float(np.std(d) / max(np.mean(d), 1e-9))
    # near-constant spacing AND many peaks -> suspicious
    regular = max(0.0, 1.0 - cv / 0.35)
    many = min(1.0, (len(pk) - 3) / 4.0)
    score = float(np.clip(regular * many, 0, 1))
    if score > 0.6:
        v = ("STRONG: %d peaks spaced by an almost constant %.2f decade (CV %.2f). "
             "This is the classic signature of Tikhonov ringing rather than %d distinct "
             "processes. Increase lambda or use the L-curve/GCV setting and check whether "
             "the peaks collapse into a plateau." % (len(pk), float(np.mean(d)), cv, len(pk)))
    elif score > 0.3:
        v = ("MODERATE: peak spacing is fairly regular (CV %.2f). Some of these maxima may be "
             "numerical. Compare against the combined Re+Im inversion." % cv)
    else:
        v = "LOW: peak spacing is irregular, consistent with genuine physical processes."
    return score, v


def interpret(res: DRTResult) -> List[str]:
    """Produce plain-language findings for one result."""
    out: List[str] = []
    if res.r_pol <= 0:
        return ["The distribution is empty - check the data and the regularisation."]

    reg = res.region_r or {}
    slow = sum(v for k, v in reg.items() if k in ("P6", "P7"))
    fast = sum(v for k, v in reg.items() if k in ("P1", "P2", "P3", "P4", "P5"))
    tot = max(slow + fast, 1e-12)
    out.append("Total polarisation resistance R_pol = %.1f ohm; series resistance R_inf = %.2f ohm."
               % (res.r_pol, res.r_inf))
    if slow / tot > 0.55:
        out.append("The cell is MASS-TRANSPORT LIMITED: %.0f %% of R_pol lies at tau > 0.15 s. "
                   "Improving it is a diffusion / electrode-architecture problem, not primarily a "
                   "charge-transfer problem." % (100 * slow / tot))
    elif fast / tot > 0.65:
        out.append("The cell is KINETICS/INTERFACE LIMITED: %.0f %% of R_pol lies at tau < 0.15 s, "
                   "i.e. in the film and charge-transfer windows." % (100 * fast / tot))
    else:
        out.append("Polarisation is shared roughly evenly between the fast (interface/kinetic) and "
                   "slow (diffusion) windows.")

    if res.peaks:
        big = max(res.peaks, key=lambda p: p.resistance)
        out.append("Dominant process: %s at tau = %.3g s (f = %.3g Hz), carrying %.1f ohm "
                   "(%.0f %% of R_pol) - %s."
                   % (big.label, big.tau, big.freq, big.resistance,
                      100 * big.resistance / max(res.r_pol, 1e-12), big.process))

    # flat plateau detection
    tau, g = res.tau, res.gamma
    m = (tau > 1e-5) & (tau < 1e-2)
    if m.sum() > 5:
        seg = g[m]
        if np.max(seg) > 0 and (np.max(seg) - np.min(seg)) / max(np.max(seg), 1e-12) < 0.55:
            out.append("gamma is nearly FLAT between 1e-5 and 1e-2 s - the fingerprint of a "
                       "distributed, CPE-like interface (rough galvanized zinc plus a porous "
                       "hard-carbon layer), not of a single process.")

    score, verdict = ringing_score(res)
    out.append("Ringing check - " + verdict)

    if res.rms_pct:
        q = ("excellent" if res.rms_pct < 1 else "good" if res.rms_pct < 3
             else "moderate" if res.rms_pct < 8 else "poor")
        out.append("Fit quality: RMS residual %.2f %% (%s). lambda = %.3g." %
                   (res.rms_pct, q, res.lambda_used))

    # edge warning
    if res.peaks:
        edge = res.tau.max() / 10.0
        late = [p for p in res.peaks if p.tau > edge]
        if late:
            out.append("WARNING: %d peak(s) lie within one decade of the upper edge of the tau "
                       "window and are extrapolated rather than measured. Extend the EIS to lower "
                       "frequency before quoting them." % len(late))
    return out


def compare_results(results: Dict[str, DRTResult]) -> List[str]:
    """Cross-check the re / im / combined inversions of one sample."""
    out: List[str] = []
    have = [m for m in ("re", "im", "combined") if m in results]
    if len(have) < 2:
        return out
    rp = {m: results[m].r_pol for m in have}
    lo, hi = min(rp.values()), max(rp.values())
    ratio = hi / max(lo, 1e-12)
    txt = ", ".join("%s = %.1f ohm" % (m.capitalize(), rp[m]) for m in have)
    out.append("R_pol by inversion mode: " + txt + ".")
    if ratio > 1.8:
        out.append("INCONSISTENCY (factor %.1f): because Re(Z) and Im(Z) are linked by the "
                   "Kramers-Kronig relations, a clean stationary data set must give essentially "
                   "the same DRT whichever channel is inverted. A discrepancy this large points to "
                   "low-frequency drift (a non-stationary cell during the slow part of the sweep), "
                   "an inconsistent R_inf/inductance treatment, or different regularisation. Run "
                   "the Kramers-Kronig test and treat the low-frequency resistance as an upper "
                   "bound." % ratio)
    else:
        out.append("The three inversions agree within a factor %.2f, which is good evidence that "
                   "the spectrum is consistent and stationary." % ratio)

    if "re" in results:
        s, v = ringing_score(results["re"])
        if s > 0.3:
            out.append("Re-only inversion: " + v)
    out.append("Recommendation: quote the COMBINED Re+Im inversion for all numbers. The Im-only "
               "result is the best-localising (use it to argue whether a peak is really a "
               "doublet); the Re-only result should be used for peak POSITIONS only.")
    return out
