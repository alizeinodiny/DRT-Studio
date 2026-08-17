"""
DRTtools-compatible discretisation of the DRT problem.

This implements the radial-basis-function (RBF) formulation of

        Z(f) = R_inf + i*omega*L + integral gamma(ln tau) / (1 + i*omega*tau) dln tau

exactly as in the reference DRTtools / pyDRTtools code of the Ciucci group, so
that DRT Studio reproduces their numbers.

References
----------
[1] T.H. Wan, M. Saccoccio, C. Chen, F. Ciucci, "Influence of the discretization
    methods on the distribution of relaxation times deconvolution: implementing
    radial basis functions with DRTtools", Electrochim. Acta 184 (2015) 483-499.
[2] M. Saccoccio, T.H. Wan, C. Chen, F. Ciucci, "Optimal regularization in
    distribution of relaxation times ... ridge and lasso regression",
    Electrochim. Acta 147 (2014) 470-482.
[3] A. Maradesa, B. Py, T.H. Wan, M.B. Effat, F. Ciucci, "Selecting the
    regularization parameter in the distribution of relaxation times",
    J. Electrochem. Soc. 170 (2023) 030502.

Two things matter for accuracy and are done the DRTtools way here:

* the design matrices A_re / A_im are the *analytic* integrals (32)-(33) of [1]
  of the RBF against the Debye kernel, not a crude midpoint sum;
* the roughness penalty is the *analytic* inner-product matrix M of the RBF
  derivatives, eq. (38) of [1], not a finite difference of the coefficients.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz

__all__ = [
    "RBF_TYPES", "RBF_LABELS", "SHAPE_CONTROLS", "POOR_WITH_NONNEG",
    "rbf_callable", "fwhm_coefficient", "compute_epsilon",
    "assemble_A_re", "assemble_A_im", "assemble_M", "x_to_gamma", "gamma_to_x",
]

# order matters: this is what the GUI shows
RBF_TYPES = [
    "Gaussian",
    "C0 Matern",
    "C2 Matern",
    "C4 Matern",
    "C6 Matern",
    "Inverse Quadratic",
    "Inverse Quadric",
    "Cauchy",
    "PWL",
]

# RBFs whose non-negative expansion cannot represent a sharp DRT: their tails
# are so broad that reproducing a narrow peak needs NEGATIVE coefficients, which
# the physical x >= 0 constraint forbids.  Verified numerically: the best
# unconstrained fit is 0.0002 % but the best non-negative fit is 4-6 %.
POOR_WITH_NONNEG = {"Cauchy", "Inverse Quadric"}

RBF_LABELS = {
    "Gaussian": "Gaussian (default)",
    "C0 Matern": "C0 Matern",
    "C2 Matern": "C2 Matern",
    "C4 Matern": "C4 Matern",
    "C6 Matern": "C6 Matern",
    "Inverse Quadratic": "Inverse quadratic",
    "Inverse Quadric": "Inverse quadric (needs non-negativity OFF)",
    "Cauchy": "Cauchy (needs non-negativity OFF)",
    "PWL": "Piecewise linear (no RBF)",
}

SHAPE_CONTROLS = ["FWHM Coefficient", "Shape Factor"]


# --------------------------------------------------------------------------
# the radial basis functions themselves, phi(x) with x = ln tau - ln tau_m
# --------------------------------------------------------------------------
def rbf_callable(rbf_type: str, epsilon: float):
    """Return phi(x) for the requested RBF, with shape factor epsilon."""
    e = float(epsilon)

    def gaussian(x):
        return np.exp(-((e * x) ** 2))

    def c0(x):
        return np.exp(-np.abs(e * x))

    def c2(x):
        a = np.abs(e * x)
        return np.exp(-a) * (1 + a)

    def c4(x):
        a = np.abs(e * x)
        return 1.0 / 3.0 * np.exp(-a) * (3 + 3 * a + a ** 2)

    def c6(x):
        a = np.abs(e * x)
        return 1.0 / 15.0 * np.exp(-a) * (15 + 15 * a + 6 * a ** 2 + a ** 3)

    def inv_quadratic(x):
        return 1.0 / (1 + (e * x) ** 2)

    def inv_quadric(x):
        return 1.0 / np.sqrt(1 + (e * x) ** 2)

    def cauchy(x):
        return 1.0 / (1 + np.abs(e * x))

    table = {
        "Gaussian": gaussian,
        "C0 Matern": c0,
        "C2 Matern": c2,
        "C4 Matern": c4,
        "C6 Matern": c6,
        "Inverse Quadratic": inv_quadratic,
        "Inverse Quadric": inv_quadric,
        "Cauchy": cauchy,
    }
    if rbf_type not in table:
        raise ValueError("unknown RBF type %r" % (rbf_type,))
    return table[rbf_type]


def fwhm_coefficient(rbf_type: str) -> float:
    """
    2 * x_half, where phi(x_half) = 1/2 for the unit-epsilon RBF.

    DRTtools solves this numerically with fsolve; the analytic roots are known
    for every supported RBF, so we use them (identical values, no solver).
    """
    ln2 = np.log(2.0)
    if rbf_type == "Gaussian":                  # exp(-x^2) = 1/2
        x = np.sqrt(ln2)
    elif rbf_type == "C0 Matern":               # exp(-|x|) = 1/2
        x = ln2
    elif rbf_type == "Inverse Quadratic":       # 1/(1+x^2) = 1/2
        x = 1.0
    elif rbf_type == "Inverse Quadric":         # 1/sqrt(1+x^2) = 1/2
        x = np.sqrt(3.0)
    elif rbf_type == "Cauchy":                  # 1/(1+|x|) = 1/2
        x = 1.0
    else:
        # C2/C4/C6 Matern: solve by bisection on a monotone decreasing function
        phi = rbf_callable(rbf_type, 1.0)
        lo, hi = 0.0, 50.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if phi(mid) > 0.5:
                lo = mid
            else:
                hi = mid
        x = 0.5 * (lo + hi)
    return 2.0 * float(x)


def compute_epsilon(freq: np.ndarray, coeff: float, rbf_type: str,
                    shape_control: str = "FWHM Coefficient") -> float:
    """Shape factor of the RBF; eq. (13) of [1]."""
    if rbf_type == "PWL":
        return 0.0
    if shape_control == "Shape Factor":
        return float(coeff)
    freq = np.asarray(freq, float).ravel()
    delta = float(np.mean(np.diff(np.log(1.0 / freq))))
    if delta == 0.0 or not np.isfinite(delta):
        delta = 1.0
    return float(coeff) * fwhm_coefficient(rbf_type) / abs(delta)


# --------------------------------------------------------------------------
# quadrature for the A matrices, eq. (32)-(33) of [1]
# --------------------------------------------------------------------------
def _quad_grid(epsilon: float, rbf_type: str):
    """
    Fixed Gauss-Legendre nodes for  integral_{-50}^{50} f(x) phi(x) dx.

    The truncation range is [-50, 50] to match DRTtools exactly.  That matters:
    Cauchy and the inverse quadric decay only algebraically, so ~30 % of their
    mass lies beyond |x| = 50 and integrating further gives a different (and
    incompatible) discretisation.  DRTtools calls scipy.integrate.quad once per
    matrix element; we use one fixed high-order rule instead, which is
    vectorisable and gives the same value to ~1e-12 for these smooth integrands.
    """
    e = max(float(epsilon), 1e-6)
    # Composite rule: the integrand has a narrow core of width ~1/e sitting in a
    # wide [-50, 50] window, so a single Gauss rule would need a huge order.
    # Use a dense rule on the core and a sparse one on each tail.
    core = min(50.0, 12.0 / e)
    # Several RBFs (all the Matern family and Cauchy) contain |x| and so have a
    # derivative kink at x = 0.  Gauss-Legendre assumes smoothness, so split the
    # core at 0 and integrate each half separately; otherwise the kink costs
    # several digits of accuracy.
    xc, wc = np.polynomial.legendre.leggauss(400)
    h = 0.5 * core
    xs = [xc * h + h, xc * h - h]
    ws = [wc * h, wc * h]
    if core < 50.0:
        xt, wt = np.polynomial.legendre.leggauss(400)
        mid, half = 0.5 * (core + 50.0), 0.5 * (50.0 - core)
        xs += [xt * half + mid, xt * half - mid]
        ws += [wt * half, wt * half]
    return np.concatenate(xs), np.concatenate(ws)


def _A_elements(freq: np.ndarray, tau: np.ndarray, epsilon: float,
                rbf_type: str, imag: bool) -> np.ndarray:
    """Vectorised evaluation of g_i / g_ii for all (freq, tau) pairs."""
    phi_fn = rbf_callable(rbf_type, epsilon)
    xs, ws = _quad_grid(epsilon, rbf_type)
    phi = phi_fn(xs) * ws                                    # (Q,)

    f = np.asarray(freq, float).ravel()
    t = np.asarray(tau, float).ravel()
    alpha = 2.0 * np.pi * np.outer(f, t)                     # (F, T)

    out = np.empty(alpha.shape, float)
    e2x = np.exp(2.0 * xs)
    ex = np.exp(xs)

    # chunk over frequencies to bound memory at ~T*Q doubles per step
    chunk = max(1, int(4e6 // max(t.size * xs.size, 1)))
    for i0 in range(0, f.size, chunk):
        a = alpha[i0:i0 + chunk, :, None]                    # (c, T, 1)
        if imag:
            # alpha / (e^-x + alpha^2 e^x)
            integ = a / (1.0 / ex + (a ** 2) * ex)
        else:
            # 1 / (1 + alpha^2 e^2x)
            integ = 1.0 / (1.0 + (a ** 2) * e2x)
        out[i0:i0 + chunk, :] = integ @ phi
    return out


def _is_log_uniform(vec: np.ndarray) -> bool:
    d = np.diff(np.log(np.asarray(vec, float).ravel()))
    if d.size < 2:
        return False
    m = np.mean(d)
    return bool(m != 0 and np.std(d) / abs(m) < 0.01)


def _toeplitz_ok(freq: np.ndarray, tau: np.ndarray) -> bool:
    """
    A[p, q] depends only on the product omega_p * tau_q, i.e. on
    (log f_p + log tau_q).  That is Toeplitz in (q - p) only when the two grids
    are log-uniform with EQUAL AND OPPOSITE spacing, which is exactly the
    DRTtools layout tau_m = 1/f_m.  With both grids ascending the matrix is
    Hankel instead, and using toeplitz() there silently produces constant rows.
    """
    if freq.size != tau.size or freq.size < 3:
        return False
    if not (_is_log_uniform(freq) and _is_log_uniform(tau)):
        return False
    df = float(np.mean(np.diff(np.log(freq))))
    dt = float(np.mean(np.diff(np.log(tau))))
    return abs(df + dt) < 1e-9 * max(abs(df), 1e-30)


def assemble_A_re(freq: np.ndarray, tau: np.ndarray, epsilon: float,
                  rbf_type: str) -> np.ndarray:
    """Design matrix for Re Z; eq. (A.3a)/(A.4) of [2] for PWL, (32) of [1]."""
    freq = np.asarray(freq, float).ravel()
    tau = np.asarray(tau, float).ravel()
    nf, nt = freq.size, tau.size
    omega = 2.0 * np.pi * freq

    if rbf_type == "PWL":
        A = np.zeros((nf, nt))
        lt = np.log(tau)
        for q in range(nt):
            if q == 0:
                dl = lt[1] - lt[0]
            elif q == nt - 1:
                dl = lt[-1] - lt[-2]
            else:
                dl = lt[q + 1] - lt[q - 1]
            A[:, q] = 0.5 / (1.0 + (omega * tau[q]) ** 2) * dl
        return A

    if _toeplitz_ok(freq, tau):
        # first row (p = 0, all q) and first column (all p, q = 0)
        R = _A_elements(freq[:1], tau, epsilon, rbf_type, imag=False).ravel()
        C = _A_elements(freq, tau[:1], epsilon, rbf_type, imag=False).ravel()
        return toeplitz(C, R)

    return _A_elements(freq, tau, epsilon, rbf_type, imag=False)


def assemble_A_im(freq: np.ndarray, tau: np.ndarray, epsilon: float,
                  rbf_type: str) -> np.ndarray:
    """Design matrix for Im Z (sign convention of DRTtools: A_im = -g_ii)."""
    freq = np.asarray(freq, float).ravel()
    tau = np.asarray(tau, float).ravel()
    nf, nt = freq.size, tau.size
    omega = 2.0 * np.pi * freq

    if rbf_type == "PWL":
        A = np.zeros((nf, nt))
        lt = np.log(tau)
        for q in range(nt):
            if q == 0:
                dl = lt[1] - lt[0]
            elif q == nt - 1:
                dl = lt[-1] - lt[-2]
            else:
                dl = lt[q + 1] - lt[q - 1]
            A[:, q] = -0.5 * (omega * tau[q]) / (1.0 + (omega * tau[q]) ** 2) * dl
        return A

    if _toeplitz_ok(freq, tau):
        R = -_A_elements(freq[:1], tau, epsilon, rbf_type, imag=True).ravel()
        C = -_A_elements(freq, tau[:1], epsilon, rbf_type, imag=True).ravel()
        return toeplitz(C, R)

    return -_A_elements(freq, tau, epsilon, rbf_type, imag=True)


# --------------------------------------------------------------------------
# roughness penalty: analytic inner products of the RBF derivatives, eq. (38)
# --------------------------------------------------------------------------
def _inner_prod_1(a: np.ndarray, epsilon: float, rbf_type: str) -> np.ndarray:
    """<phi'_n, phi'_m> as a function of a = epsilon * ln(f_n/f_m)."""
    e = epsilon
    aa = np.abs(a)
    s = np.sqrt(np.pi / 2.0)
    if rbf_type == "Gaussian":
        return -e * (-1 + a ** 2) * np.exp(-(a ** 2) / 2) * s
    if rbf_type == "C0 Matern":
        return e * (1 - aa) * np.exp(-aa)
    if rbf_type == "C2 Matern":
        return e / 6.0 * (3 + 3 * aa - aa ** 3) * np.exp(-aa)
    if rbf_type == "C4 Matern":
        return (e / 30.0 * (105 + 105 * aa + 30 * aa ** 2 - 5 * aa ** 3
                            - 5 * aa ** 4 - aa ** 5) * np.exp(-aa))
    if rbf_type == "C6 Matern":
        return (e / 140.0 * (10395 + 10395 * aa + 3780 * aa ** 2 + 315 * aa ** 3
                             - 210 * aa ** 4 - 84 * aa ** 5 - 14 * aa ** 6
                             - aa ** 7) * np.exp(-aa))
    if rbf_type == "Inverse Quadratic":
        return 4 * e * (4 - 3 * a ** 2) * np.pi / ((4 + a ** 2) ** 3)
    if rbf_type == "Cauchy":
        out = np.empty_like(aa, dtype=float)
        z = aa == 0
        out[z] = 2.0 / 3.0 * e
        b = aa[~z]
        num = (b * (2 + b) * (4 + 3 * b * (2 + b))
               - 2 * (1 + b) ** 2 * (4 + b * (2 + b)) * np.log1p(b))
        den = b ** 3 * (1 + b) * (2 + b) ** 3
        out[~z] = 4 * e * num / den
        return out
    raise ValueError("no analytic first-derivative inner product for %r" % rbf_type)


def _inner_prod_2(a: np.ndarray, epsilon: float, rbf_type: str) -> np.ndarray:
    """<phi''_n, phi''_m> as a function of a = epsilon * ln(f_n/f_m)."""
    e = epsilon
    aa = np.abs(a)
    s = np.sqrt(np.pi / 2.0)
    if rbf_type == "Gaussian":
        return e ** 3 * (3 - 6 * a ** 2 + a ** 4) * np.exp(-(a ** 2) / 2) * s
    if rbf_type == "C0 Matern":
        return e ** 3 * (1 + aa) * np.exp(-aa)
    if rbf_type == "C2 Matern":
        return e ** 3 / 6.0 * (3 + 3 * aa - 6 * aa ** 2 + aa ** 3) * np.exp(-aa)
    if rbf_type == "C4 Matern":
        return (e ** 3 / 30.0 * (45 + 45 * aa - 15 * aa ** 3 - 5 * aa ** 4
                                 + aa ** 5) * np.exp(-aa))
    if rbf_type == "C6 Matern":
        return (e ** 3 / 140.0 * (2835 + 2835 * aa + 630 * aa ** 2
                                  - 315 * aa ** 3 - 210 * aa ** 4
                                  - 42 * aa ** 5 + aa ** 7) * np.exp(-aa))
    if rbf_type == "Inverse Quadratic":
        return 48 * (16 + 5 * a ** 2 * (-8 + a ** 2)) * np.pi * e ** 3 / ((4 + a ** 2) ** 5)
    if rbf_type == "Cauchy":
        out = np.empty_like(aa, dtype=float)
        z = aa == 0
        out[z] = 8.0 / 5.0 * e ** 3
        b = aa[~z]
        num = (b * (2 + b) * (-96 + b * (2 + b) * (-30 + b * (2 + b))
                              * (4 + b * (2 + b)))
               + 12 * (1 + b) ** 2 * (16 + b * (2 + b) * (12 + b * (2 + b)))
               * np.log1p(b))
        den = b ** 5 * (1 + b) * (2 + b) ** 5
        out[~z] = 8 * e ** 3 * num / den
        return out
    raise ValueError("no analytic second-derivative inner product for %r" % rbf_type)


def _numeric_inner_prod(tau: np.ndarray, epsilon: float, rbf_type: str,
                        order: int) -> np.ndarray:
    """Fallback (Inverse Quadric): differentiate numerically and integrate."""
    phi = rbf_callable(rbf_type, epsilon)
    xs, ws = _quad_grid(epsilon, rbf_type)
    lt = np.log(np.asarray(tau, float).ravel())
    h = 1e-5 / max(epsilon, 1e-3)
    # d^order phi(x - c) evaluated on the quadrature grid for every centre
    n = lt.size
    D = np.empty((n, xs.size))
    for i in range(n):
        u = xs - (lt[i] - lt[0])
        if order == 1:
            D[i] = (phi(u + h) - phi(u - h)) / (2 * h)
        else:
            D[i] = (phi(u + h) - 2 * phi(u) + phi(u - h)) / (h * h)
    return (D * ws) @ D.T


def assemble_M(tau: np.ndarray, epsilon: float, rbf_type: str,
               order: int) -> np.ndarray:
    """
    Roughness matrix M with  x^T M x = integral (d^order gamma/dln tau^2)^2.

    eq. (38) of [1].  For PWL this reduces to L^T L with the usual finite
    difference stencils of DRTtools.
    """
    tau = np.asarray(tau, float).ravel()
    n = tau.size
    order = int(order)

    if order == 0:
        return np.eye(n)

    if rbf_type == "PWL":
        lt = np.log(tau)
        if order == 1:
            L = np.zeros((n - 1, n))
            for i in range(n - 1):
                d = lt[i + 1] - lt[i]
                L[i, i] = -1.0 / d
                L[i, i + 1] = 1.0 / d
        else:
            L = np.zeros((max(n - 2, 0), n))
            for p in range(n - 2):
                d = lt[p + 1] - lt[p]
                if p == 0 or p == n - 3:
                    L[p, p] = 2.0 / d ** 2
                    L[p, p + 1] = -4.0 / d ** 2
                    L[p, p + 2] = 2.0 / d ** 2
                else:
                    L[p, p] = 1.0 / d ** 2
                    L[p, p + 1] = -2.0 / d ** 2
                    L[p, p + 2] = 1.0 / d ** 2
        return L.T @ L

    if rbf_type == "Inverse Quadric":
        M = _numeric_inner_prod(tau, epsilon, rbf_type, order)
        return 0.5 * (M + M.T)

    # a = epsilon * ln(f_n/f_m) = epsilon * (ln tau_m - ln tau_n)
    lt = np.log(tau)
    A = epsilon * (lt[None, :] - lt[:, None])
    M = _inner_prod_1(A, epsilon, rbf_type) if order == 1 else \
        _inner_prod_2(A, epsilon, rbf_type)
    M = np.asarray(M, float)
    return 0.5 * (M + M.T)


# --------------------------------------------------------------------------
# mapping between RBF coefficients x and the DRT gamma(tau)
# --------------------------------------------------------------------------
def x_to_gamma(x: np.ndarray, tau_map: np.ndarray, tau: np.ndarray,
               epsilon: float, rbf_type: str) -> np.ndarray:
    """gamma(tau_map) = sum_m x_m phi(ln tau_map - ln tau_m)."""
    x = np.asarray(x, float).ravel()
    if rbf_type == "PWL":
        return np.interp(np.log(tau_map), np.log(tau), x)
    phi = rbf_callable(rbf_type, epsilon)
    B = phi(np.log(np.asarray(tau_map, float))[:, None]
            - np.log(np.asarray(tau, float))[None, :])
    return B @ x


def gamma_to_x(gamma: np.ndarray, tau: np.ndarray, epsilon: float,
               rbf_type: str) -> np.ndarray:
    """Inverse of x_to_gamma on the collocation grid."""
    gamma = np.asarray(gamma, float).ravel()
    if rbf_type == "PWL":
        return gamma
    phi = rbf_callable(rbf_type, epsilon)
    lt = np.log(np.asarray(tau, float))
    B = phi(lt[:, None] - lt[None, :])
    B = 0.5 * (B + B.T)
    return np.linalg.solve(B, gamma)
