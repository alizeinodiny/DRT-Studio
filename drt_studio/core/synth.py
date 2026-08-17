"""
Synthetic zinc-ion-cell impedance generator.

Used for the built-in demo project and for the self-test.  The circuit mimics the
cell discussed in the report: series R + L, a distributed (CPE-like) interface
band, a cathode charge-transfer arc, an anode charge-transfer arc and a
finite-length Warburg diffusion element.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def zarc(w: np.ndarray, r: float, tau: float, n: float = 0.85) -> np.ndarray:
    """Cole-Cole / ZARC element: R / (1 + (jw tau)^n)."""
    return r / (1.0 + (1j * w * tau) ** n)


def warburg_short(w: np.ndarray, r: float, tau: float) -> np.ndarray:
    """Finite-length (short-circuit terminated) Warburg."""
    x = np.sqrt(1j * w * tau)
    x = np.where(np.abs(x) < 1e-12, 1e-12, x)
    return r * np.tanh(x) / x


def make_spectrum(f: np.ndarray = None,
                  r_inf: float = 6.0,
                  ind: float = 1e-6,
                  film_r: float = 12.0, film_tau: float = 1.5e-4, film_n: float = 0.62,
                  cat_r: float = 8.0, cat_tau: float = 4.0e-3, cat_n: float = 0.80,
                  an_r: float = 12.0, an_tau: float = 8.0e-2, an_n: float = 0.85,
                  diff_r: float = 180.0, diff_tau: float = 0.55,
                  noise_pct: float = 0.0, drift_pct: float = 0.0,
                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (f, Z', Z'') with Z'' negative for capacitive behaviour."""
    if f is None:
        f = np.logspace(np.log10(5e4), np.log10(3e-2), 61)
    w = 2 * np.pi * f
    z = (r_inf + 1j * w * ind
         + zarc(w, film_r, film_tau, film_n)
         + zarc(w, cat_r, cat_tau, cat_n)
         + zarc(w, an_r, an_tau, an_n)
         + warburg_short(w, diff_r, diff_tau))

    rng = np.random.default_rng(seed)
    if noise_pct:
        mod = np.abs(z)
        z = z + (rng.normal(0, noise_pct / 100, z.size) * mod
                 + 1j * rng.normal(0, noise_pct / 100, z.size) * mod)
    if drift_pct:
        # low-frequency drift: violates stationarity, inflates the slow peak
        k = np.clip((np.log10(f.max()) - np.log10(f)) / (np.log10(f.max()) - np.log10(f.min())),
                    0, 1) ** 2
        z = z * (1.0 + drift_pct / 100.0 * k)
    return f, z.real, z.imag


DEMO_SAMPLES: Dict[str, dict] = {
    "Cell A - baseline galvanized Zn": dict(seed=1, noise_pct=0.3),
    "Cell B - passivated anode": dict(seed=2, noise_pct=0.3, film_r=32.0, film_tau=3.0e-4,
                                      an_r=22.0),
    "Cell C - thin hard-carbon cathode": dict(seed=3, noise_pct=0.3, diff_r=95.0,
                                              diff_tau=0.22, cat_r=5.0),
    "Cell D - drifting (KK fail)": dict(seed=4, noise_pct=0.5, drift_pct=18.0),
}
