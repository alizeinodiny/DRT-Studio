"""
Compatibility helpers so DRT Studio runs on old and new dependency versions.

The integration routine was renamed in NumPy 2.0:

    numpy 1.x :  np.trapz      (np.trapezoid does not exist)
    numpy 2.x :  np.trapezoid  (np.trapz is deprecated, removed in 2.3)

Python 3.8 cannot install numpy >= 2.0 at all, so calling np.trapezoid there
raises ``AttributeError: module 'numpy' has no attribute 'trapezoid'``.
Import ``trapezoid`` from here instead of calling either name directly.
"""
from __future__ import annotations

import numpy as np

__all__ = ["trapezoid", "NUMPY_MAJOR"]

NUMPY_MAJOR = int(np.__version__.split(".")[0])

if hasattr(np, "trapezoid"):          # numpy >= 2.0
    trapezoid = np.trapezoid
elif hasattr(np, "trapz"):            # numpy 1.x
    trapezoid = np.trapz
else:                                 # pragma: no cover - should not happen
    def trapezoid(y, x=None, dx=1.0, axis=-1):
        """Minimal fallback trapezoidal integration."""
        y = np.asanyarray(y)
        if x is None:
            d = dx
        else:
            x = np.asanyarray(x)
            d = np.diff(x, axis=-1 if x.ndim > 1 else 0)
        nd = y.ndim
        s1 = [slice(None)] * nd
        s2 = [slice(None)] * nd
        s1[axis] = slice(1, None)
        s2[axis] = slice(None, -1)
        return np.sum(d * (y[tuple(s1)] + y[tuple(s2)]) / 2.0, axis=axis)
