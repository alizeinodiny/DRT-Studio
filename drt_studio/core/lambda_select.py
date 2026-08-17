"""
Regularisation-parameter selection, following DRTtools.

Implements the criteria compared in

    A. Maradesa, B. Py, T.H. Wan, M.B. Effat, F. Ciucci,
    "Selecting the regularization parameter in the distribution of relaxation
    times", J. Electrochem. Soc. 170 (2023) 030502.

    GCV    generalised cross-validation                    eq. (13)
    mGCV   modified GCV, stabilisation rho                 eq. (14)-(15)
    rGCV   robust GCV, robustness xi                       eq. (16)
    LC     L-curve maximum curvature
    re-im  real/imaginary discrepancy
    kf     k-fold cross-validation

All scores are functions of log(lambda) and are minimised over
[log 1e-7, log 1e0], the interval used by DRTtools.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np

__all__ = ["LAMBDA_METHODS", "LAMBDA_LABELS", "select_lambda", "lambda_curve"]

LAMBDA_METHODS = ["manual", "GCV", "mGCV", "rGCV", "LC", "re-im", "kf"]

LAMBDA_LABELS = {
    "manual": "custom (use the value below)",
    "GCV": "GCV - generalised cross-validation",
    "mGCV": "mGCV - modified GCV",
    "rGCV": "rGCV - robust GCV",
    "LC": "L-curve (maximum curvature)",
    "re-im": "re-im discrepancy",
    "kf": "k-fold cross-validation",
}

LOG_LAMBDA_MIN = np.log(1e-7)
LOG_LAMBDA_MAX = np.log(1e0)


# --------------------------------------------------------------------------
def _nearest_pd(A: np.ndarray) -> np.ndarray:
    """Nearest symmetric positive-definite matrix (Higham 1988), as DRTtools."""
    B = 0.5 * (A + A.T)
    try:
        _, s, V = np.linalg.svd(B)
    except np.linalg.LinAlgError:
        return B + np.eye(B.shape[0]) * 1e-10
    H = V.T @ np.diag(s) @ V
    A2 = 0.5 * (B + H)
    A3 = 0.5 * (A2 + A2.T)
    if _is_pd(A3):
        return A3
    spacing = np.spacing(np.linalg.norm(A))
    I = np.eye(A.shape[0])
    k = 1
    while not _is_pd(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k ** 2 + spacing)
        k += 1
        if k > 40:
            break
    return A3


def _is_pd(A: np.ndarray) -> bool:
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def _hat_matrix(A: np.ndarray, M: np.ndarray, lam: float):
    """K = A (A^T A + lambda M)^-1 A^T, via Cholesky as in DRTtools."""
    A_in = A.T @ A + lam * M
    if not _is_pd(A_in):
        A_in = _nearest_pd(A_in)
    try:
        L = np.linalg.cholesky(A_in)
    except np.linalg.LinAlgError:
        A_in = A_in + np.eye(A_in.shape[0]) * 1e-10
        L = np.linalg.cholesky(A_in)
    inv_L = np.linalg.inv(L)
    inv_A_in = inv_L.T @ inv_L
    return A @ inv_A_in @ A.T, inv_A_in


# --------------------------------------------------------------------------
# the individual scores
# --------------------------------------------------------------------------
def _score_gcv(log_lam, A, Z, M, **kw):
    lam = float(np.exp(log_lam))
    n = Z.size
    K, _ = _hat_matrix(A, M, lam)
    I = np.eye(n)
    num = np.linalg.norm((I - K) @ Z) ** 2 / n
    den = (np.trace(I - K) / n) ** 2
    return num / max(den, 1e-300)


def _score_mgcv(log_lam, A, Z, M, **kw):
    lam = float(np.exp(log_lam))
    n = Z.size
    K, _ = _hat_matrix(A, M, lam)
    I = np.eye(n)
    rho = 1.3 if n < 50 else 2.0
    num = np.linalg.norm((I - K) @ Z) ** 2 / n
    den = (np.trace(I - rho * K) / n) ** 2
    return num / max(den, 1e-300)


def _score_rgcv(log_lam, A, Z, M, **kw):
    lam = float(np.exp(log_lam))
    n = Z.size
    K, _ = _hat_matrix(A, M, lam)
    I = np.eye(n)
    num = np.linalg.norm((I - K) @ Z) ** 2 / n
    den = (np.trace(I - K) / n) ** 2
    gcv = num / max(den, 1e-300)
    xi = 0.2 if n < 100 else 0.3
    mu2 = np.trace(K.T @ K) / n
    return (xi + (1 - xi) * mu2) * gcv


def _score_lc(log_lam, A, Z, M, **kw):
    """
    L-curve corner.

    The curve (log residual, log penalty) is convex and L-shaped: the penalty
    falls steeply while lambda is small, then the residual rises steeply once
    lambda is too large.  The corner is the point of maximum POSITIVE curvature
    of that convex bend.

    Taking -|kappa| (the obvious mistake) also matches the shallow bend at the
    over-smoothed end and can return lambda ~ 0.1, which destroys the DRT, so
    the sign is enforced here and non-corner points are given a score of 0.
    """
    h = 0.08

    def pt(ll):
        lam = float(np.exp(ll))
        A_in = A.T @ A + lam * M
        if not _is_pd(A_in):
            A_in = _nearest_pd(A_in)
        try:
            x = np.linalg.solve(A_in, A.T @ Z)
        except np.linalg.LinAlgError:
            return np.nan, np.nan
        r = np.log(max(np.linalg.norm(A @ x - Z) ** 2, 1e-300))
        e = np.log(max(float(x @ (M @ x)), 1e-300))
        return r, e

    r0, e0 = pt(log_lam - h)
    r1, e1 = pt(log_lam)
    r2, e2 = pt(log_lam + h)
    if not np.all(np.isfinite([r0, r1, r2, e0, e1, e2])):
        return 0.0
    dr = (r2 - r0) / (2 * h)
    de = (e2 - e0) / (2 * h)
    ddr = (r2 - 2 * r1 + r0) / (h * h)
    dde = (e2 - 2 * e1 + e0) / (h * h)
    denom = (dr ** 2 + de ** 2) ** 1.5
    if denom < 1e-300:
        return 0.0
    kappa = (dr * dde - de * ddr) / denom
    # we minimise, so return the negative of the (positive) corner curvature
    return -max(kappa, 0.0)


def _score_reim(log_lam, A, Z, M, A_re=None, A_im=None, Z_re=None, Z_im=None,
                **kw):
    """
    re-im discrepancy: fit on one part, measure the misfit on the other.
    """
    if A_re is None:
        return _score_gcv(log_lam, A, Z, M)
    lam = float(np.exp(log_lam))

    def solve(Ax, b):
        H = Ax.T @ Ax + lam * M
        if not _is_pd(H):
            H = _nearest_pd(H)
        return np.linalg.solve(H, Ax.T @ b)

    x_re = solve(A_re, Z_re)
    x_im = solve(A_im, Z_im)
    # cross-prediction error, normalised
    e1 = np.linalg.norm(A_im @ x_re - Z_im) ** 2 / max(np.linalg.norm(Z_im) ** 2, 1e-300)
    e2 = np.linalg.norm(A_re @ x_im - Z_re) ** 2 / max(np.linalg.norm(Z_re) ** 2, 1e-300)
    return e1 + e2


def _score_kf(log_lam, A, Z, M, n_folds: int = 5, **kw):
    """k-fold cross-validation on the stacked residual vector."""
    lam = float(np.exp(log_lam))
    n = Z.size
    idx = np.arange(n)
    folds = np.array_split(idx, min(n_folds, n))
    err = 0.0
    for te in folds:
        tr = np.setdiff1d(idx, te, assume_unique=False)
        At, Zt = A[tr], Z[tr]
        H = At.T @ At + lam * M
        if not _is_pd(H):
            H = _nearest_pd(H)
        x = np.linalg.solve(H, At.T @ Zt)
        err += np.linalg.norm(A[te] @ x - Z[te]) ** 2
    return err / n


_SCORES: Dict[str, Callable] = {
    "GCV": _score_gcv,
    "mGCV": _score_mgcv,
    "rGCV": _score_rgcv,
    "LC": _score_lc,
    "re-im": _score_reim,
    "kf": _score_kf,
}


# --------------------------------------------------------------------------
def select_lambda(method: str, A: np.ndarray, Z: np.ndarray, M: np.ndarray,
                  A_re: Optional[np.ndarray] = None,
                  A_im: Optional[np.ndarray] = None,
                  Z_re: Optional[np.ndarray] = None,
                  Z_im: Optional[np.ndarray] = None,
                  log_lambda_0: float = np.log(1e-3),
                  n_grid: int = 24) -> Tuple[float, dict]:
    """
    Return (lambda, info).  Coarse log-grid scan followed by a golden-section
    refinement; this is more robust than SLSQP from a single starting guess,
    which is what DRTtools uses and which can stop at a local minimum.
    """
    if method in (None, "", "manual"):
        raise ValueError("select_lambda called with manual mode")
    if method not in _SCORES:
        raise ValueError("unknown lambda method %r" % (method,))
    fn = _SCORES[method]
    kw = dict(A_re=A_re, A_im=A_im, Z_re=Z_re, Z_im=Z_im)

    grid = np.linspace(LOG_LAMBDA_MIN, LOG_LAMBDA_MAX, int(n_grid))
    vals = np.full(grid.size, np.inf)
    for i, ll in enumerate(grid):
        try:
            v = float(fn(ll, A, Z, M, **kw))
            vals[i] = v if np.isfinite(v) else np.inf
        except Exception:
            vals[i] = np.inf
    if not np.any(np.isfinite(vals)):
        return float(np.exp(log_lambda_0)), {"method": method, "ok": False}

    k = int(np.argmin(vals))
    lo = grid[max(k - 1, 0)]
    hi = grid[min(k + 1, grid.size - 1)]

    # golden-section refinement
    gr = (np.sqrt(5.0) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    fc = fd = None
    for _ in range(28):
        if fc is None:
            try:
                fc = float(fn(c, A, Z, M, **kw))
            except Exception:
                fc = np.inf
        if fd is None:
            try:
                fd = float(fn(d, A, Z, M, **kw))
            except Exception:
                fd = np.inf
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = None
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = None
        if abs(b - a) < 1e-4:
            break
    best_ll = 0.5 * (a + b)
    try:
        best_val = float(fn(best_ll, A, Z, M, **kw))
    except Exception:
        best_val = float(vals[k])
    if not np.isfinite(best_val) or best_val > vals[k]:
        best_ll, best_val = float(grid[k]), float(vals[k])

    lam = float(np.exp(best_ll))
    lam = float(np.clip(lam, 1e-7, 1e0))
    return lam, {"method": method, "ok": True, "score": best_val,
                 "log_lambda": best_ll,
                 "grid": np.exp(grid), "scores": vals}


def lambda_curve(method: str, A, Z, M, n_grid: int = 24, **kw):
    """Score vs lambda, for plotting the selection criterion."""
    fn = _SCORES[method]
    grid = np.linspace(LOG_LAMBDA_MIN, LOG_LAMBDA_MAX, int(n_grid))
    out = np.full(grid.size, np.nan)
    for i, ll in enumerate(grid):
        try:
            out[i] = float(fn(ll, A, Z, M, **kw))
        except Exception:
            pass
    return np.exp(grid), out
