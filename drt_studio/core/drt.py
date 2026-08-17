"""
Regularised DRT inversion.

Model
-----
    Z(w) = R_inf + j*w*L + integral gamma(ln tau) / (1 + j*w*tau) d ln tau

gamma is expanded on radial basis functions (Gaussian) or piecewise-constant
elements placed on a log-spaced tau grid.  The coefficients are found by solving

    min_x  || W (A x - z) ||^2 + lambda * || D x ||^2      subject to  x >= 0

with a non-negative least squares solver.  `lambda` can be chosen manually or
automatically by the L-curve or GCV criterion.

Three inversion modes are supported and are the core comparison of this app:
  * 're'       - fit the real part only  (ill-conditioned, must also fit R_inf)
  * 'im'       - fit the imaginary part only (well-conditioned, blind to R_inf)
  * 'combined' - fit both simultaneously (recommended reference)
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from .compat import trapezoid
from scipy.optimize import nnls, lsq_linear

from .model import DRTResult, DRTSettings


# --------------------------------------------------------------------------------------
# tau grid and basis
# --------------------------------------------------------------------------------------

def make_tau_grid(f: np.ndarray, st: DRTSettings) -> np.ndarray:
    w = 2.0 * np.pi * np.asarray(f, float)
    t_lo = 1.0 / w.max()
    t_hi = 1.0 / w.min()
    lo = np.log10(t_lo) - st.tau_extend_decades
    hi = np.log10(t_hi) + st.tau_extend_decades
    n = max(10, int(round((hi - lo) * st.points_per_decade)) + 1)
    return np.logspace(lo, hi, n)


def _rbf_sigma(log_tau: np.ndarray, st: DRTSettings) -> float:
    """Gaussian sigma in ln(tau) units derived from the requested FWHM factor."""
    if log_tau.size < 2:
        return 1.0
    d = float(np.mean(np.diff(log_tau)))          # spacing in ln tau
    fwhm = st.rbf_fwhm_factor * d
    return fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _quad_nodes(n: int = 61, span: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes/weights on [-span, span] for the RBF convolution."""
    x, wq = np.polynomial.legendre.leggauss(n)
    return x * span, wq * span


def build_kernel(f: np.ndarray, tau: np.ndarray, st: DRTSettings
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (A_re, A_im): contribution of each basis coefficient to Re(Z) and Im(Z).

    For the Gaussian basis the integral

        A(w, tau_m) = int phi_m(ln t) / (1 + j w t) d ln t

    is evaluated by Gauss-Legendre quadrature in ln t.  For the rectangular
    basis the analytic piecewise-constant result is used.
    """
    w = 2.0 * np.pi * np.asarray(f, float)
    lt = np.log(tau)

    if st.basis == "rectangular":
        # piecewise-constant elements of width dlt centred on each node
        dlt = np.gradient(lt)
        WT = np.outer(w, tau)                       # (F, M)
        denom = 1.0 + WT ** 2
        a_re = (1.0 / denom) * dlt[None, :]
        a_im = (-WT / denom) * dlt[None, :]
        return a_re, a_im

    sig = _rbf_sigma(lt, st)
    xs, wq = _quad_nodes()
    a_re = np.zeros((w.size, tau.size))
    a_im = np.zeros((w.size, tau.size))
    norm = 1.0 / (sig * np.sqrt(2.0 * np.pi))
    for m, lt_m in enumerate(lt):
        u = lt_m + sig * xs                          # ln t nodes around the centre
        t = np.exp(u)
        phi = norm * np.exp(-0.5 * xs ** 2) * sig    # phi(u) du -> weight
        WT = np.outer(w, t)
        denom = 1.0 + WT ** 2
        a_re[:, m] = ((1.0 / denom) * (phi * wq)[None, :]).sum(axis=1)
        a_im[:, m] = ((-WT / denom) * (phi * wq)[None, :]).sum(axis=1)
    return a_re, a_im


def gamma_from_coeffs(tau_out: np.ndarray, tau: np.ndarray, x: np.ndarray,
                      st: DRTSettings) -> np.ndarray:
    """Evaluate gamma(ln tau) on an output grid from the basis coefficients."""
    if st.basis == "rectangular":
        return np.interp(np.log(tau_out), np.log(tau), x)
    lt = np.log(tau)
    sig = _rbf_sigma(lt, st)
    out = np.zeros_like(tau_out, dtype=float)
    lo = np.log(tau_out)
    norm = 1.0 / (sig * np.sqrt(2.0 * np.pi))
    for m, lt_m in enumerate(lt):
        if x[m] == 0.0:
            continue
        out += x[m] * norm * np.exp(-0.5 * ((lo - lt_m) / sig) ** 2)
    return out


def derivative_matrix(n: int, order: int) -> np.ndarray:
    if order <= 0:
        return np.eye(n)
    if order == 1:
        D = np.zeros((n - 1, n))
        for i in range(n - 1):
            D[i, i] = -1.0
            D[i, i + 1] = 1.0
        return D
    D = np.zeros((max(n - 2, 1), n))
    for i in range(n - 2):
        D[i, i] = 1.0
        D[i, i + 1] = -2.0
        D[i, i + 2] = 1.0
    return D


# --------------------------------------------------------------------------------------
# assembling the least squares problem
# --------------------------------------------------------------------------------------

def _assemble(f, z_re, z_im, tau, st: DRTSettings, mode: str):
    """
    Build the design matrix M and data vector b for the chosen mode.

    Unknown vector x = [gamma coefficients ... , R_inf?, L?]
    R_inf is a free parameter for 're' and 'combined'; for 'im' it is absent
    (the imaginary part carries no information about it).
    L is a free parameter whenever the imaginary part is fitted.
    """
    a_re, a_im = build_kernel(f, tau, st)
    w = 2.0 * np.pi * np.asarray(f, float)
    m = tau.size

    use_rinf = mode in ("re", "combined")
    use_l = st.include_inductance and mode in ("im", "combined")

    cols = [a_re] if mode == "re" else ([a_im] if mode == "im" else [np.vstack([a_re, a_im])])
    blocks = cols[0]
    extra = []
    if use_rinf:
        if mode == "re":
            extra.append(np.ones((f.size, 1)))
        else:
            extra.append(np.vstack([np.ones((f.size, 1)), np.zeros((f.size, 1))]))
    if use_l:
        if mode == "im":
            extra.append(w.reshape(-1, 1))
        else:
            extra.append(np.vstack([np.zeros((f.size, 1)), w.reshape(-1, 1)]))
    M = np.hstack([blocks] + extra) if extra else blocks

    if mode == "re":
        b = np.asarray(z_re, float)
    elif mode == "im":
        b = np.asarray(z_im, float)
    else:
        b = np.concatenate([np.asarray(z_re, float), np.asarray(z_im, float)])

    # weighting
    if st.weighting == "modulus":
        mod = np.hypot(z_re, z_im)
        mod = np.where(mod <= 0, np.nanmax(mod) if np.nanmax(mod) > 0 else 1.0, mod)
        wv = 1.0 / mod
        if mode == "combined":
            wv = np.concatenate([wv, wv])
    else:
        wv = np.ones_like(b)

    n_extra = len(extra)
    return M, b, wv, m, n_extra, use_rinf, use_l


def _col_scale(Mw, m, n_extra):
    """
    Column equilibration.

    The inductance column carries w (up to ~1e6) while the gamma columns are of
    order 1.  Without rescaling the problem is hopelessly ill-conditioned and the
    solver dumps polarisation into the edge of the tau grid.  Only the extra
    (R_inf / L) columns are rescaled, so the regularisation applied to the gamma
    block keeps its physical meaning.
    """
    n = m + n_extra
    sc = np.ones(n)
    if n_extra <= 0:
        return sc
    ref = np.linalg.norm(Mw[:, :m]) / max(np.sqrt(m), 1.0)
    ref = ref if ref > 0 else 1.0
    for k in range(m, n):
        nk = np.linalg.norm(Mw[:, k])
        sc[k] = (nk / ref) if nk > 0 else 1.0
    return sc


def _solve_one(M, b, wv, m, n_extra, lam, st: DRTSettings):
    """Solve the regularised problem for one lambda.  Returns x (unscaled)."""
    D = derivative_matrix(m, st.derivative_order)
    n = m + n_extra
    Dfull = np.zeros((D.shape[0], n))
    Dfull[:, :m] = D

    Mw = M * wv[:, None]
    bw = b * wv

    cs = _col_scale(Mw, m, n_extra)
    Ms = Mw / cs[None, :]

    scale = np.linalg.norm(Ms) / max(np.linalg.norm(Dfull), 1e-30)
    A = np.vstack([Ms, np.sqrt(lam) * scale * Dfull])
    y = np.concatenate([bw, np.zeros(Dfull.shape[0])])

    if st.non_negative:
        if n_extra > 0:
            lo = np.zeros(n)
            hi = np.full(n, np.inf)
            lo[m:] = -np.inf          # R_inf and L may take either sign
            res = lsq_linear(A, y, bounds=(lo, hi), max_iter=500, tol=1e-12)
            xs = res.x
        else:
            xs, _ = nnls(A, y)
    else:
        xs, *_ = np.linalg.lstsq(A, y, rcond=None)
    return xs / cs


def _fit_error(M, b, wv, x):
    r = (M @ x - b) * wv
    return float(np.linalg.norm(r))


def choose_lambda(M, b, wv, m, n_extra, st: DRTSettings) -> float:
    """Pick lambda by the L-curve corner (max curvature) or by GCV."""
    lams = np.logspace(np.log10(st.lambda_min), np.log10(st.lambda_max), st.lambda_steps)
    D = derivative_matrix(m, st.derivative_order)
    n = m + n_extra
    Dfull = np.zeros((D.shape[0], n))
    Dfull[:, :m] = D

    res_n, sol_n, gcvs = [], [], []
    Mw = M * wv[:, None]
    bw = b * wv
    cs = _col_scale(Mw, m, n_extra)
    Ms = Mw / cs[None, :]
    scale = np.linalg.norm(Ms) / max(np.linalg.norm(Dfull), 1e-30)
    for lam in lams:
        x = _solve_one(M, b, wv, m, n_extra, lam, st)
        r = Mw @ x - bw
        res_n.append(np.linalg.norm(r))
        sol_n.append(np.linalg.norm(Dfull @ x))
        # GCV with an approximate trace of the influence matrix
        try:
            H = Ms.T @ Ms + lam * (scale ** 2) * (Dfull.T @ Dfull)
            tr = np.trace(Ms @ np.linalg.solve(H, Ms.T))
        except Exception:
            tr = 0.0
        den = max(len(bw) - tr, 1e-6)
        gcvs.append(len(bw) * np.sum(r ** 2) / den ** 2)

    if st.lambda_mode == "gcv":
        return float(lams[int(np.argmin(gcvs))])

    # L-curve: maximum curvature in log-log space
    lr = np.log10(np.maximum(res_n, 1e-30))
    ls = np.log10(np.maximum(sol_n, 1e-30))
    if lr.size < 5:
        return float(lams[lr.size // 2])
    d1r, d1s = np.gradient(lr), np.gradient(ls)
    d2r, d2s = np.gradient(d1r), np.gradient(d1s)
    kappa = np.abs(d1r * d2s - d2r * d1s) / np.power(d1r ** 2 + d1s ** 2, 1.5) + 1e-30
    return float(lams[int(np.argmax(kappa))])


# --------------------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------------------

def compute_drt(f: np.ndarray, z_re: np.ndarray, z_im: np.ndarray,
                st: DRTSettings, mode: str,
                out_points_per_decade: int = 40,
                progress=None) -> DRTResult:
    """
    Run one DRT inversion and return a fully populated DRTResult.

    Two engines are available:
      * 'drttools' (default) - analytic RBF discretisation and roughness matrix
        with the DRTtools lambda-selection criteria.  This is the accurate one.
      * 'legacy'  - the original lightweight quadrature kernel, kept because it
        is fast and useful as an independent cross-check.
    """
    if str(getattr(st, "engine", "drttools")).lower() == "drttools":
        from .drt_engine import compute_drt_drttools
        return compute_drt_drttools(f, z_re, z_im, st, mode,
                                    out_points_per_decade, progress=progress)
    return _compute_drt_legacy(f, z_re, z_im, st, mode, out_points_per_decade)


def _compute_drt_legacy(f: np.ndarray, z_re: np.ndarray, z_im: np.ndarray,
                        st: DRTSettings, mode: str,
                        out_points_per_decade: int = 40) -> DRTResult:
    """Original engine (kept as an independent cross-check)."""
    f = np.asarray(f, float)
    z_re = np.asarray(z_re, float)
    z_im = np.asarray(z_im, float)

    order = np.argsort(f)
    f, z_re, z_im = f[order], z_re[order], z_im[order]

    tau = make_tau_grid(f, st)
    M, b, wv, m, n_extra, use_rinf, use_l = _assemble(f, z_re, z_im, tau, st, mode)

    lam = st.lambda_value
    if st.lambda_mode in ("lcurve", "gcv"):
        lam = choose_lambda(M, b, wv, m, n_extra, st)

    x = _solve_one(M, b, wv, m, n_extra, lam, st)

    coeffs = x[:m]
    idx = m
    r_inf = 0.0
    ind = 0.0
    if use_rinf:
        r_inf = float(x[idx]); idx += 1
    if use_l:
        ind = float(x[idx]); idx += 1

    # output grid, denser for smooth plotting
    lo, hi = np.log10(tau.min()), np.log10(tau.max())
    n_out = max(50, int((hi - lo) * out_points_per_decade) + 1)
    tau_out = np.logspace(lo, hi, n_out)
    gamma = gamma_from_coeffs(tau_out, tau, coeffs, st)
    gamma = np.maximum(gamma, 0.0) if st.non_negative else gamma

    r_pol = float(trapezoid(gamma, np.log(tau_out)))

    # reconstruct the impedance from the solution (all modes -> full complex Z)
    a_re, a_im = build_kernel(f, tau, st)
    w = 2.0 * np.pi * f
    z_fit_re = a_re @ coeffs + r_inf
    z_fit_im = a_im @ coeffs + (ind * w if use_l else 0.0)

    # For the 'im' mode R_inf is not identifiable from the data (it cancels out of
    # Im(Z)).  Estimate it for display only, clamped to a physical value.
    if not use_rinf:
        est = float(np.median(z_re - (a_re @ coeffs)))
        r_inf = max(est, 0.0)
        z_fit_re = z_fit_re + r_inf

    denom = np.hypot(z_re, z_im)
    denom = np.where(denom <= 0, 1.0, denom)
    res_re = 100.0 * (z_fit_re - z_re) / denom
    res_im = 100.0 * (z_fit_im - z_im) / denom
    # The RMS must only score the channel(s) that were actually fitted: the
    # Re-only inversion says nothing about Im(Z) and vice versa, so mixing them
    # would report a meaningless error for the single-channel modes.
    if mode == "re":
        scored = res_re
    elif mode == "im":
        scored = res_im
    else:
        scored = np.concatenate([res_re, res_im])
    rms = float(np.sqrt(np.mean(scored ** 2)))

    return DRTResult(
        mode=mode, tau=tau_out, gamma=gamma, r_inf=r_inf, inductance=ind,
        r_pol=r_pol, lambda_used=float(lam), z_fit_re=z_fit_re, z_fit_im=z_fit_im,
        res_re_pct=res_re, res_im_pct=res_im, rms_pct=rms, settings=st.copy(),
    )
