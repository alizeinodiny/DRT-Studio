"""
Linear Kramers-Kronig validation (Boukamp-style measurement model).

The spectrum is fitted with a series of N RC elements whose time constants are
fixed and log-spaced over the measured range, plus R_inf, an inductance and an
optional series capacitance.  Because such a circuit is KK-transformable by
construction, a good fit means the data themselves are KK-consistent; large
systematic residuals indicate drift / non-stationarity / non-linearity.

This is the test that should always precede a DRT inversion.
"""
from __future__ import annotations

import numpy as np

from .model import KKResult


def lin_kk(f: np.ndarray, z_re: np.ndarray, z_im: np.ndarray,
           n_rc: int = 0, add_cap: bool = True) -> KKResult:
    f = np.asarray(f, float)
    z_re = np.asarray(z_re, float)
    z_im = np.asarray(z_im, float)
    o = np.argsort(f)
    f, z_re, z_im = f[o], z_re[o], z_im[o]
    w = 2 * np.pi * f

    if n_rc <= 0:
        n_rc = int(np.clip(len(f), 8, 80))

    tau = np.logspace(np.log10(1.0 / w.max()), np.log10(1.0 / w.min()), n_rc)

    WT = np.outer(w, tau)
    a_re = 1.0 / (1.0 + WT ** 2)
    a_im = -WT / (1.0 + WT ** 2)

    cols_re = [a_re, np.ones((f.size, 1)), np.zeros((f.size, 1))]
    cols_im = [a_im, np.zeros((f.size, 1)), w.reshape(-1, 1)]
    if add_cap:
        cols_re.append(np.zeros((f.size, 1)))
        cols_im.append((-1.0 / w).reshape(-1, 1))

    A = np.vstack([np.hstack(cols_re), np.hstack(cols_im)])
    b = np.concatenate([z_re, z_im])

    mod = np.hypot(z_re, z_im)
    mod = np.where(mod <= 0, 1.0, mod)
    wv = np.concatenate([1.0 / mod, 1.0 / mod])

    x, *_ = np.linalg.lstsq(A * wv[:, None], b * wv, rcond=None)
    fit = A @ x
    fr, fi = fit[:f.size], fit[f.size:]

    res_re = 100.0 * (fr - z_re) / mod
    res_im = 100.0 * (fi - z_im) / mod
    both = np.concatenate([np.abs(res_re), np.abs(res_im)])
    mx, mean = float(both.max()), float(both.mean())

    bad = (np.abs(res_re) > 1.0) | (np.abs(res_im) > 1.0)
    if mx < 0.5:
        verdict = ("PASS - residuals stay below 0.5 %. The spectrum is Kramers-Kronig "
                   "consistent, so Re and Im carry the same information and any disagreement "
                   "between the three DRT inversions is numerical, not experimental.")
    elif mx < 1.0:
        verdict = ("PASS (marginal) - maximum residual %.2f %%. Acceptable, but check the "
                   "low-frequency end before quoting diffusion resistances." % mx)
    elif mx < 3.0:
        verdict = ("CAUTION - maximum residual %.2f %%. Part of the spectrum is not KK "
                   "consistent; the usual cause in an aqueous Zn cell is drift during the slow "
                   "low-frequency sweep (self-discharge, H2 evolution, ZHS growth). Treat "
                   "low-frequency resistances as upper bounds." % mx)
    else:
        verdict = ("FAIL - maximum residual %.2f %%. The data violate linearity/stationarity. "
                   "Re-measure with a smaller amplitude, longer equilibration, or a narrower "
                   "frequency window; DRT results from these data are not trustworthy." % mx)

    return KKResult(res_re_pct=res_re, res_im_pct=res_im, max_abs_pct=mx,
                    mean_abs_pct=mean, n_rc=n_rc, verdict=verdict,
                    suspect_freqs=f[bad])
