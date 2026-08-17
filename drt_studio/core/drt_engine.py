"""
DRTtools-compatible inversion engine.

Solves the ridge-regression (Tikhonov) DRT problem in exactly the form used by
DRTtools / pyDRTtools:

    min_x  ||A_re x - Z_re||^2 + ||A_im x - Z_im||^2 + lambda * x^T M x
    s.t.   x >= 0

where the unknown vector is

    x = [ R_inf , L , x_1 ... x_N ]

with the RBF coefficients x_m, and where A_re / A_im are the analytic RBF
design matrices and M is the analytic roughness matrix (see core/rbf.py).

The inductance handling follows DRTtools' ``induct_used``:
    0 / 2  -> no inductance term (N_RL = 1, only R_inf)
    1      -> fit R_inf and L  (N_RL = 2)
For the 'im' mode R_inf drops out of the model (it does not affect Im Z) and is
recovered afterwards for display only.

The quadratic program is solved with scipy; cvxopt is used when available
because that is what DRTtools uses, but it is optional.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import lsq_linear, nnls

from .compat import trapezoid
from .lambda_select import select_lambda
from .model import DRTResult, DRTSettings
from .rbf import (assemble_A_im, assemble_A_re, assemble_M, compute_epsilon,
                  x_to_gamma)

__all__ = ["compute_drt_drttools", "make_tau_grid_drttools"]


def make_tau_grid_drttools(f: np.ndarray, st: DRTSettings) -> np.ndarray:
    """
    Collocation timescales.

    DRTtools places one tau per measured frequency, tau_m = 1/f_m, which makes
    A square and enables the Toeplitz trick.  When the user asks for a finer
    grid (points_per_decade) or an extended range we fall back to a log grid.
    """
    f = np.asarray(f, float).ravel()
    f = np.sort(f)[::-1]                       # descending f -> ascending tau
    tau_native = 1.0 / f
    ppd = getattr(st, "points_per_decade", 10)
    ext = float(getattr(st, "tau_extend_decades", 0.0) or 0.0)

    lo = np.log10(tau_native.min()) - ext
    hi = np.log10(tau_native.max()) + ext
    n_native = tau_native.size
    span = hi - lo
    n_req = int(round(span * ppd)) + 1

    if ext <= 0 and abs(n_req - n_native) <= max(2, 0.1 * n_native):
        return np.sort(tau_native)             # use the native grid
    return np.logspace(lo, hi, max(n_req, 10))


def _weights(z_re, z_im, weighting: str):
    if weighting == "modulus":
        mod = np.hypot(z_re, z_im)
        mx = np.nanmax(mod)
        mod = np.where(mod <= 0, mx if mx > 0 else 1.0, mod)
        return 1.0 / mod
    if weighting == "proportional":
        a = np.abs(z_re) + np.abs(z_im)
        a = np.where(a <= 0, 1.0, a)
        return 1.0 / a
    return np.ones_like(np.asarray(z_re, float))


def _solve_qp(H: np.ndarray, c: np.ndarray, n: int, non_negative: bool,
              free: int = 0) -> np.ndarray:
    """
    Minimise 1/2 x^T H x + c^T x, optionally with x >= 0 on the gamma block.

    `free` leading entries (R_inf, L) are left unconstrained in sign only when
    they are physically allowed to be negative; DRTtools constrains the whole
    vector to be non-negative, and we match that by default.
    """
    H = 0.5 * (H + H.T)
    # try cvxopt first: this is what DRTtools uses
    if non_negative:
        try:
            from cvxopt import matrix, solvers
            solvers.options["show_progress"] = False
            solvers.options["abstol"] = 1e-10
            solvers.options["reltol"] = 1e-9
            G = matrix(-np.eye(n))
            h = matrix(np.zeros(n))
            sol = solvers.qp(matrix(H), matrix(c), G, h)
            if sol.get("status") in ("optimal", "unknown"):
                return np.asarray(sol["x"]).ravel()
        except Exception:
            pass

    # scipy fallback: express the QP as a bounded linear least-squares problem
    # H = 2 F^T F  and  c = -2 F^T d  =>  min ||F x - d||^2
    try:
        w, V = np.linalg.eigh(H)
        w = np.clip(w, 0.0, None)
        F = (V * np.sqrt(w / 2.0)) @ V.T
        # solve F^T F x = -c/2  in least-squares sense
        rhs = -c / 2.0
        d = np.linalg.lstsq(F.T, rhs, rcond=None)[0]
        if non_negative:
            lb = np.zeros(n)
            ub = np.full(n, np.inf)
            if free:
                lb[:free] = -np.inf
            res = lsq_linear(F, d, bounds=(lb, ub), method="bvls",
                             max_iter=max(300, 12 * n), tol=1e-12)
            return res.x
        return np.linalg.solve(H + 1e-12 * np.eye(n), -c)
    except Exception:
        pass

    if non_negative:
        w, V = np.linalg.eigh(H)
        w = np.clip(w, 1e-12, None)
        F = (V * np.sqrt(w / 2.0)) @ V.T
        d = np.linalg.lstsq(F.T, -c / 2.0, rcond=None)[0]
        return nnls(F, d)[0]
    return np.linalg.solve(H + 1e-9 * np.eye(n), -c)


def compute_drt_drttools(f: np.ndarray, z_re: np.ndarray, z_im: np.ndarray,
                         st: DRTSettings, mode: str,
                         out_points_per_decade: int = 40,
                         progress=None) -> DRTResult:
    """Full DRTtools-style inversion for one mode."""
    f = np.asarray(f, float).ravel()
    z_re = np.asarray(z_re, float).ravel()
    z_im = np.asarray(z_im, float).ravel()
    o = np.argsort(f)
    f, z_re, z_im = f[o], z_re[o], z_im[o]

    rbf_type = getattr(st, "rbf_type", "Gaussian")
    shape_control = getattr(st, "shape_control", "FWHM Coefficient")
    coeff = float(getattr(st, "rbf_fwhm_factor", 1.0) or 1.0)
    order = int(getattr(st, "derivative_order", 2))
    induct_used = int(getattr(st, "induct_used", 1 if st.include_inductance else 0))

    tau = make_tau_grid_drttools(f, st)
    eps = compute_epsilon(f, coeff, rbf_type, shape_control)

    if progress:
        progress("assembling design matrices", 0.15)
    A_re = assemble_A_re(f, tau, eps, rbf_type)
    A_im = assemble_A_im(f, tau, eps, rbf_type)
    M_gamma = assemble_M(tau, eps, rbf_type, order)

    n_tau = tau.size
    omega = 2.0 * np.pi * f

    # ---- prepend the R_inf / L columns, DRTtools layout x = [R_inf, L, x...]
    use_L = (induct_used == 1)
    use_Rinf = mode in ("re", "combined")
    n_rl = (1 if use_Rinf else 0) + (1 if use_L else 0)

    cols_re, cols_im = [], []
    if use_Rinf:
        cols_re.append(np.ones((f.size, 1)))
        cols_im.append(np.zeros((f.size, 1)))
    if use_L:
        cols_re.append(np.zeros((f.size, 1)))
        cols_im.append(omega.reshape(-1, 1))
    # The inductance column is omega (up to ~1e6) while the RBF columns are of
    # order 1.  Left as-is, A^T A is dominated by that one column (norm ~1e11),
    # the QP becomes hopelessly ill-conditioned and the solver returns gamma~0.
    # Scale the R_inf / L columns to unit norm and undo it after the solve.
    col_scale = np.ones(n_rl + n_tau)
    if n_rl:
        A_re_f = np.hstack(cols_re + [A_re])
        A_im_f = np.hstack(cols_im + [A_im])
        for j in range(n_rl):
            nj = np.sqrt(np.linalg.norm(A_re_f[:, j]) ** 2
                         + np.linalg.norm(A_im_f[:, j]) ** 2)
            if nj > 0:
                col_scale[j] = nj
                A_re_f[:, j] /= nj
                A_im_f[:, j] /= nj
    else:
        A_re_f, A_im_f = A_re, A_im

    n_x = n_rl + n_tau
    M_full = np.zeros((n_x, n_x))
    M_full[n_rl:, n_rl:] = M_gamma

    # ---- weighting
    #
    # DRTtools solves the UNWEIGHTED problem, and its lambda values (default
    # 1e-3, search range 1e-7..1e0) are calibrated against that.  Any weighting
    # rescales the data term and therefore silently changes what a given lambda
    # means -- modulus weighting on a 200 ohm cell makes lambda ~1600x stronger,
    # which drives gamma to zero.  We keep the weighting option, but renormalise
    # the weights so that the data term keeps the same magnitude as the
    # unweighted one; lambda then means exactly what it means in DRTtools.
    wv = _weights(z_re, z_im, getattr(st, "weighting", "modulus"))
    wv = np.asarray(wv, float)
    Wr = wv[:, None]
    A_re_w, A_im_w = A_re_f * Wr, A_im_f * Wr
    z_re_w, z_im_w = z_re * wv, z_im * wv

    # Renormalise so that ||A_w^T A_w|| matches the unweighted ||A^T A||.  This
    # keeps lambda on the DRTtools scale whatever weighting the user picks.
    n_unw = np.linalg.norm(A_re_f.T @ A_re_f + A_im_f.T @ A_im_f)
    n_w = np.linalg.norm(A_re_w.T @ A_re_w + A_im_w.T @ A_im_w)
    if n_w > 0 and n_unw > 0:
        k = np.sqrt(n_unw / n_w)
        A_re_w, A_im_w = A_re_w * k, A_im_w * k
        z_re_w, z_im_w = z_re_w * k, z_im_w * k

    # ---- pick lambda
    lam = float(getattr(st, "lambda_value", 1e-3))
    lam_mode = str(getattr(st, "lambda_mode", "manual"))
    lam_info = {"method": "manual"}
    # accept the legacy names too
    legacy = {"lcurve": "LC", "gcv": "GCV", "manual": "manual"}
    lam_mode = legacy.get(lam_mode, lam_mode)

    if lam_mode != "manual":
        if progress:
            progress("selecting lambda (%s)" % lam_mode, 0.35)
        if mode == "re":
            A_stack, Z_stack = A_re_w, z_re_w
        elif mode == "im":
            A_stack, Z_stack = A_im_w, z_im_w
        else:
            A_stack = np.vstack([A_re_w, A_im_w])
            Z_stack = np.concatenate([z_re_w, z_im_w])
        try:
            lam, lam_info = select_lambda(
                lam_mode, A_stack, Z_stack, M_full,
                A_re=A_re_w, A_im=A_im_w, Z_re=z_re_w, Z_im=z_im_w,
                log_lambda_0=np.log(max(lam, 1e-7)))
        except Exception as ex:
            lam_info = {"method": lam_mode, "ok": False, "error": str(ex)}

    # ---- assemble and solve the QP
    if progress:
        progress("solving", 0.6)
    if mode == "re":
        H = 2.0 * (A_re_w.T @ A_re_w + lam * M_full)
        c = -2.0 * (A_re_w.T @ z_re_w)
    elif mode == "im":
        H = 2.0 * (A_im_w.T @ A_im_w + lam * M_full)
        c = -2.0 * (A_im_w.T @ z_im_w)
    else:
        H = 2.0 * (A_re_w.T @ A_re_w + A_im_w.T @ A_im_w + lam * M_full)
        c = -2.0 * (A_re_w.T @ z_re_w + A_im_w.T @ z_im_w)

    x = _solve_qp(H, c, n_x, bool(getattr(st, "non_negative", True)))
    x = x / col_scale          # back to physical units

    # ---- unpack
    idx = 0
    r_inf = 0.0
    ind = 0.0
    if use_Rinf:
        r_inf = float(x[idx]); idx += 1
    if use_L:
        ind = float(x[idx]); idx += 1
    coef = x[idx:]

    # ---- gamma on a dense output grid
    lo, hi = np.log10(tau.min()), np.log10(tau.max())
    n_out = max(80, int((hi - lo) * out_points_per_decade) + 1)
    tau_out = np.logspace(lo, hi, n_out)
    gamma = x_to_gamma(coef, tau_out, tau, eps, rbf_type)
    if getattr(st, "non_negative", True):
        gamma = np.maximum(gamma, 0.0)
    r_pol = float(trapezoid(gamma, np.log(tau_out)))

    # ---- reconstruct Z
    z_fit_re = A_re @ coef + r_inf
    z_fit_im = A_im @ coef + (ind * omega if use_L else 0.0)
    if not use_Rinf:
        est = float(np.median(z_re - (A_re @ coef)))
        r_inf = max(est, 0.0)
        z_fit_re = z_fit_re + r_inf

    denom = np.hypot(z_re, z_im)
    denom = np.where(denom <= 0, 1.0, denom)
    res_re = 100.0 * (z_fit_re - z_re) / denom
    res_im = 100.0 * (z_fit_im - z_im) / denom
    if mode == "re":
        scored = res_re
    elif mode == "im":
        scored = res_im
    else:
        scored = np.concatenate([res_re, res_im])
    rms = float(np.sqrt(np.mean(scored ** 2)))

    res = DRTResult(
        mode=mode, tau=tau_out, gamma=gamma, r_inf=r_inf, inductance=ind,
        r_pol=r_pol, lambda_used=float(lam), z_fit_re=z_fit_re,
        z_fit_im=z_fit_im, res_re_pct=res_re, res_im_pct=res_im,
        rms_pct=rms, settings=st.copy(),
    )
    # extra provenance so the report can state exactly how the number was made
    try:
        res.meta = {
            "engine": "drttools",
            "rbf_type": rbf_type,
            "epsilon": float(eps),
            "shape_control": shape_control,
            "derivative_order": order,
            "n_tau": int(n_tau),
            "lambda_method": lam_info.get("method", lam_mode),
            "induct_used": induct_used,
            "tau_collocation": tau,
            "x": x,
        }
    except Exception:
        pass
    return res
