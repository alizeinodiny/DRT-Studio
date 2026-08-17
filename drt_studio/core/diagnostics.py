"""
Runtime diagnostics for DRT Studio.

Two jobs:

1.  `check_environment()` - what is installed, what is missing, and which
    features are therefore unavailable.  This is what turns a mid-export
    ``ModuleNotFoundError: reportlab`` traceback into a sentence the user can
    act on *before* they press the button.

2.  `run_diagnostics()` - an end-to-end numerical self-check.  It builds a
    synthetic cell whose answer is known analytically, inverts it, and verifies
    that the recovered resistances, the peak positions, the Kramers-Kronig test
    and the round-trip of a project file are all correct.  If the maths is
    wrong this reports it in numbers rather than failing silently.

Both are GUI-free and can be run from a script or the Help menu.
"""
from __future__ import annotations

import importlib
import platform
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

# feature name -> (module, pip name, what breaks without it)
OPTIONAL = {
    "PDF reports": ("reportlab", "reportlab", "PDF export is disabled."),
    "Word reports": ("docx", "python-docx", "Word (.docx) export is disabled."),
    "Excel workbooks": ("openpyxl", "openpyxl", "Excel (.xlsx) export is disabled."),
    "Excel/CSV import": ("pandas", "pandas", "File import is disabled."),
    "Plotting": ("matplotlib", "matplotlib", "All figures are disabled."),
    "Solver": ("scipy", "scipy", "Falls back to a slower/less robust solver."),
}

REQUIRED = ["numpy"]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""
    fatal: bool = False


@dataclass
class Report:
    checks: List[Check] = field(default_factory=list)
    seconds: float = 0.0

    def add(self, name: str, ok: bool, detail: str = "", fatal: bool = False):
        self.checks.append(Check(name, ok, detail, fatal))
        return ok

    @property
    def n_pass(self) -> int:
        return sum(1 for c in self.checks if c.ok)

    @property
    def n_fail(self) -> int:
        return sum(1 for c in self.checks if not c.ok)

    @property
    def ok(self) -> bool:
        return self.n_fail == 0

    def text(self) -> str:
        w = max((len(c.name) for c in self.checks), default=10)
        out = []
        for c in self.checks:
            mark = "PASS" if c.ok else ("FAIL" if c.fatal else "WARN")
            line = "  %-4s  %-*s  %s" % (mark, w, c.name, c.detail)
            out.append(line.rstrip())
        out.append("")
        out.append("%d passed, %d failed  (%.1f s)"
                   % (self.n_pass, self.n_fail, self.seconds))
        return "\n".join(out)


# --------------------------------------------------------------------- env
def check_environment() -> Report:
    r = Report()
    t0 = time.time()
    r.add("Python", sys.version_info >= (3, 8),
          "%s on %s" % (platform.python_version(), platform.system()),
          fatal=sys.version_info < (3, 8))
    for mod in REQUIRED:
        try:
            m = importlib.import_module(mod)
            r.add(mod, True, getattr(m, "__version__", ""))
        except Exception as e:
            r.add(mod, False, "MISSING - %s" % e, fatal=True)
    # numpy renamed trapz -> trapezoid in 2.0; we support both via core.compat
    try:
        import numpy as _np
        from .compat import trapezoid as _tz
        r.add("numpy integration API", True,
              "numpy %s -> using np.%s" % (_np.__version__, _tz.__name__))
    except Exception as e:
        r.add("numpy integration API", False, str(e), fatal=True)
    for feat, (mod, pipname, consequence) in OPTIONAL.items():
        try:
            m = importlib.import_module(mod)
            r.add(feat, True, "%s %s" % (mod, getattr(m, "__version__", "")))
        except Exception:
            r.add(feat, False, "%s not installed -> %s  Fix: pip install %s"
                  % (mod, consequence, pipname))
    try:
        import tkinter
        r.add("Tkinter (GUI)", True, "available")
    except Exception as e:
        r.add("Tkinter (GUI)", False,
              "missing - %s.  Fix: sudo apt-get install python3-tk" % e)
    r.seconds = time.time() - t0
    return r


def missing_for(want_pdf=False, want_docx=False, want_excel=False) -> List[str]:
    """Return human-readable reasons an export cannot run."""
    bad = []
    if want_pdf:
        try:
            importlib.import_module("reportlab")
        except Exception:
            bad.append("PDF export needs 'reportlab'  (pip install reportlab)")
    if want_docx:
        try:
            importlib.import_module("docx")
        except Exception:
            bad.append("Word export needs 'python-docx'  (pip install python-docx)")
    if want_excel:
        try:
            importlib.import_module("openpyxl")
        except Exception:
            bad.append("Excel export needs 'openpyxl'  (pip install openpyxl)")
    return bad


# ------------------------------------------------------------ numerical
def run_diagnostics(progress: Optional[Callable[[str, float], None]] = None,
                    quick: bool = False) -> Report:
    """
    End-to-end correctness check against a circuit whose answer we know.
    """
    r = Report()
    t0 = time.time()

    def tick(msg, frac):
        if progress:
            try:
                progress(msg, frac)
            except Exception:
                pass

    # --- environment first: everything below depends on it
    tick("checking environment", 0.05)
    env = check_environment()
    r.checks.extend(env.checks)

    try:
        from .synth import make_spectrum
        from .drt import compute_drt
        from .model import DRTSettings, Project, Sample
        from .kk import lin_kk
        from .peaks import analyse, ZINC_WINDOWS
    except Exception:
        r.add("core imports", False, traceback.format_exc(limit=2), fatal=True)
        r.seconds = time.time() - t0
        return r
    r.add("core imports", True, "model, drt, kk, peaks, synth")

    # --- known synthetic cell -------------------------------------------
    tick("building synthetic spectrum", 0.15)
    # the exact circuit we are about to reconstruct
    TRUE = dict(r_inf=6.0, film_r=12.0, cat_r=8.0, an_r=12.0, diff_r=180.0)
    TRUE["r_pol"] = TRUE["film_r"] + TRUE["cat_r"] + TRUE["an_r"] + TRUE["diff_r"]
    f, z_re, z_im = make_spectrum(noise_pct=0.0, drift_pct=0.0, seed=0)
    z = z_re + 1j * z_im
    r.add("synthetic spectrum", f.size > 10,
          "%d points, %.3g - %.3g Hz, true R_pol = %.0f ohm"
          % (f.size, f.min(), f.max(), TRUE["r_pol"]))

    # --- Kramers-Kronig on clean data must pass -------------------------
    tick("Kramers-Kronig", 0.3)
    try:
        kk = lin_kk(f, z_re, z_im)
        r.add("KK on clean data", kk.max_abs_pct < 1.0,
              "max |residual| = %.3f %% (expect < 1 %%)" % kk.max_abs_pct)
    except Exception:
        r.add("KK on clean data", False, traceback.format_exc(limit=2))

    # --- KK must FAIL on deliberately drifting data ---------------------
    tick("Kramers-Kronig (drift)", 0.4)
    try:
        fd, dre, dim = make_spectrum(noise_pct=0.0, drift_pct=6.0, seed=1)
        kkd = lin_kk(fd, dre, dim)
        r.add("KK detects drift", kkd.max_abs_pct > kk.max_abs_pct,
              "drifting %.2f %% vs clean %.2f %%" % (kkd.max_abs_pct, kk.max_abs_pct))
    except Exception:
        r.add("KK detects drift", False, traceback.format_exc(limit=2))

    # --- inversion in all three modes -----------------------------------
    st = DRTSettings(lambda_mode="GCV",
                     points_per_decade=8 if quick else 10)
    res = {}
    for i, mode in enumerate(("re", "im", "combined")):
        tick("inverting (%s)" % mode, 0.45 + 0.12 * i)
        try:
            res[mode] = compute_drt(f, z_re, z_im, st, mode)
            rr = res[mode]
            r.add("inversion '%s'" % mode, rr.rms_pct < 5.0,
                  "R_inf=%.2f  R_pol=%.2f  RMS=%.2f %%"
                  % (rr.r_inf, rr.r_pol, rr.rms_pct))
        except Exception:
            r.add("inversion '%s'" % mode, False, traceback.format_exc(limit=2))

    # --- gamma must be non-negative and finite --------------------------
    if "combined" in res:
        g = res["combined"].gamma
        r.add("gamma is physical", bool(np.all(np.isfinite(g)) and g.min() >= -1e-9),
              "min gamma = %.3g" % float(g.min()))

    # --- the three modes must agree on a clean, KK-consistent spectrum ---
    if len(res) == 3:
        rp = [res[m].r_pol for m in ("re", "im", "combined")]
        spread = (max(rp) - min(rp)) / max(max(rp), 1e-12) * 100
        r.add("Re/Im/Combined agree", spread < 25.0,
              "R_pol = %.1f / %.1f / %.1f ohm, spread %.1f %% (expect < 25 %% "
              "on clean data)" % (rp[0], rp[1], rp[2], spread))

    # --- recovered R_pol must match the circuit that generated the data --
    if "combined" in res:
        want, got = TRUE["r_pol"], res["combined"].r_pol
        err = abs(got - want) / want * 100
        r.add("R_pol matches the model", err < 15.0,
              "recovered %.1f vs true %.1f ohm (%.1f %% error)" % (got, want, err))
        wri = TRUE["r_inf"]
        gri = res["combined"].r_inf
        # R_inf alone is grid dependent: any relaxation faster than the first
        # tau node cannot be represented by a basis function, so its resistance
        # is absorbed into R_inf.  The physically meaningful invariant is the
        # total DC resistance R_inf + R_pol, which is grid independent - it was
        # measured as 218.0 ohm for extend = 0.0 ... 2.0 decades.
        r.add("R_inf matches the model", abs(gri - wri) < 0.35 * wri + 0.5,
              "recovered %.2f vs true %.2f ohm (high R_inf absorbs sub-grid "
              "relaxations; see total below)" % (gri, wri))
        wtot, gtot = wri + want, gri + got
        r.add("total DC resistance matches", abs(gtot - wtot) / wtot * 100 < 5.0,
              "R_inf + R_pol = %.1f vs true %.1f ohm (%.1f %%)"
              % (gtot, wtot, abs(gtot - wtot) / wtot * 100))

    # --- peak detection + area bookkeeping ------------------------------
    tick("peak detection", 0.85)
    if "combined" in res:
        try:
            rr = analyse(res["combined"], prominence_frac=0.02)
            r.add("peaks detected", len(rr.peaks) >= 1,
                  "%d peak(s): %s" % (len(rr.peaks),
                                      ", ".join("%s@%.3gs" % (p.label, p.tau)
                                                for p in rr.peaks[:5])))
            tot = sum(rr.region_r.values())
            err = abs(tot - rr.r_pol) / max(rr.r_pol, 1e-12) * 100
            r.add("window areas sum to R_pol", err < 2.0,
                  "sum(R_i) = %.2f vs R_pol = %.2f  (%.2f %% off)"
                  % (tot, rr.r_pol, err))
        except Exception:
            r.add("peak detection", False, traceback.format_exc(limit=2))

    # --- DRTtools engine: analytic matrices vs numerical quadrature ------
    tick("DRTtools discretisation", 0.9)
    try:
        from scipy.integrate import quad as _quad
        from .rbf import (assemble_A_im, assemble_A_re, compute_epsilon,
                          rbf_callable)
        ff = np.logspace(4, -2, 41)
        ff = np.sort(ff)
        tt = np.sort(1.0 / ff)
        eps = compute_epsilon(ff, 1.0, "Gaussian", "FWHM Coefficient")
        Are = assemble_A_re(ff, tt, eps, "Gaussian")
        Aim = assemble_A_im(ff, tt, eps, "Gaussian")
        phi = rbf_callable("Gaussian", eps)
        k = tt.size // 2
        xk = np.zeros(tt.size); xk[k] = 1.0
        worst = 0.0
        for pp in (3, 15, 30):
            al = 2 * np.pi * ff[pp] * tt[k]
            ref_r = _quad(lambda u: phi(u) / (1 + al ** 2 * np.exp(2 * u)),
                          -50, 50, limit=400)[0]
            ref_i = -_quad(lambda u: (al * np.exp(u)) / (1 + (al * np.exp(u)) ** 2)
                           * phi(u), -50, 50, limit=400)[0]
            worst = max(worst, abs((Are @ xk)[pp] - ref_r),
                        abs((Aim @ xk)[pp] - ref_i))
        r.add("RBF matrices match quadrature", worst < 1e-7,
              "max abs error %.2e (analytic vs adaptive quad)" % worst)
    except Exception:
        r.add("RBF matrices match quadrature", False, traceback.format_exc(limit=2))

    # --- roughness-matrix Toeplitz fast path -----------------------------
    try:
        import drt_studio.core.rbf as _R
        from .rbf import RBF_TYPES, assemble_M, compute_epsilon
        ff2 = np.sort(np.logspace(4, -2, 31))
        tt2 = np.sort(1.0 / ff2)
        worst_m, where = 0.0, ""
        for rt in RBF_TYPES:
            for order in (1, 2):
                ep = compute_epsilon(ff2, 1.0, rt, "FWHM Coefficient")
                fast = assemble_M(tt2, ep, rt, order)
                keep = _R._is_log_uniform
                _R._is_log_uniform = lambda *a, **k: False
                try:
                    slow = assemble_M(tt2, ep, rt, order)
                finally:
                    _R._is_log_uniform = keep
                e = (np.max(np.abs(fast - slow))
                     / max(float(np.max(np.abs(slow))), 1e-30))
                if e > worst_m:
                    worst_m, where = e, "%s d%d" % (rt, order)
        r.add("roughness matrix Toeplitz path exact", worst_m < 1e-10,
              "max rel. difference %.1e over 9 RBFs x 2 orders%s"
              % (worst_m, (" (%s)" % where) if where else ""))
    except Exception:
        r.add("roughness matrix Toeplitz path exact", False,
              traceback.format_exc(limit=2))

    # --- the DRTtools engine must recover the known circuit --------------
    try:
        st2 = DRTSettings(engine="drttools", lambda_mode="GCV")
        rr2 = compute_drt(f, z_re, z_im, st2, "combined")
        err = abs(rr2.r_pol - TRUE["r_pol"]) / TRUE["r_pol"] * 100
        r.add("DRTtools engine recovers R_pol", err < 15.0,
              "%.1f vs %.1f ohm (%.1f %%), lambda=%.2e, RMS=%.2f %%"
              % (rr2.r_pol, TRUE["r_pol"], err, rr2.lambda_used, rr2.rms_pct))
    except Exception:
        r.add("DRTtools engine recovers R_pol", False, traceback.format_exc(limit=2))

    # --- every lambda criterion must return a value in range -------------
    try:
        from .lambda_select import LAMBDA_METHODS
        bad = []
        for meth in LAMBDA_METHODS:
            if meth == "manual":
                continue
            stm = DRTSettings(engine="drttools", lambda_mode=meth)
            rm = compute_drt(f, z_re, z_im, stm, "combined")
            if not (1e-7 <= rm.lambda_used <= 1.0) or rm.rms_pct > 10:
                bad.append("%s(%.1e, %.1f%%)" % (meth, rm.lambda_used, rm.rms_pct))
        r.add("all lambda criteria usable", not bad,
              ", ".join(bad) if bad else "GCV, mGCV, rGCV, LC, re-im, kf")
    except Exception:
        r.add("all lambda criteria usable", False, traceback.format_exc(limit=2))

    # --- project save/load round trip -----------------------------------
    tick("project round trip", 0.95)
    try:
        import os
        p = Project(name="diagnostic")
        s = Sample(name="cell")
        s.f, s.z_re, s.z_im = f, z_re, z_im
        s.results.update(res)
        p.add(s)
        d = tempfile.mkdtemp()
        path = os.path.join(d, "diag.drtproj")
        p.save(path)
        q = Project.load(path)
        same = (q.names == p.names
                and abs(q.get("cell").results["combined"].r_pol
                        - res["combined"].r_pol) < 1e-6)
        r.add("project save/load", same, "round-tripped %d sample(s)" % len(q.names))
    except Exception:
        r.add("project save/load", False, traceback.format_exc(limit=2))

    tick("done", 1.0)
    r.seconds = time.time() - t0
    return r
