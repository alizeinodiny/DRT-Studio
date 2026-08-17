"""
Flexible data import.

Supports
--------
* .csv .txt .dat .tsv .asc  (auto delimiter sniffing, comment lines, decimal comma)
* .xlsx .xls                (any sheet)
* column auto-detection by header name, with manual override
* three import styles:
    - full   : one file containing f, Re(Z), Im(Z)
    - re     : file containing f and Re(Z) only
    - im     : file containing f and Im(Z) only
    - drt    : a precomputed DRT curve (tau, gamma) to be displayed/compared
  Separate Re and Im files are merged onto a common frequency axis.
"""
from __future__ import annotations

import io
import os
import re as _re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


TABLE_EXT = {".csv", ".txt", ".dat", ".tsv", ".asc", ".prn", ".z", ".mpt"}
EXCEL_EXT = {".xlsx", ".xlsm", ".xls"}

FREQ_KEYS = ["freq", "frequency", "f/hz", "f (hz)", "f[hz]", "hz", "f_hz", "fhz", "f"]
RE_KEYS = ["z'", "zre", "z_re", "re(z)", "rez", "real", "z real", "zr", "z1", "z_real",
           "z'/ohm", "re z", "z' (ohm)", "z'(ohm)", "resistance"]
IM_KEYS = ["z''", "z\"", "zim", "z_im", "im(z)", "imz", "imag", "z imag", "zi", "z2",
           "z_imag", "-z''", "-zim", "-z\"", "im z", "z''(ohm)", "reactance"]
TAU_KEYS = ["tau", "tau/s", "tau (s)", "tau_s", "t/s", "relaxation time"]
GAMMA_KEYS = ["gamma", "gamma(tau)", "gamma/ohm", "g(tau)", "gamma_ohm", "drt", "y"]


def _norm(s: str) -> str:
    return _re.sub(r"[\s\u00b5\u03bc]+", "", str(s).strip().lower())


def _match(cols: List[str], keys: List[str]) -> Optional[int]:
    nc = [_norm(c) for c in cols]
    for k in keys:                                  # exact
        nk = _norm(k)
        for i, c in enumerate(nc):
            if c == nk:
                return i
    for k in keys:                                  # contains
        nk = _norm(k)
        for i, c in enumerate(nc):
            if nk and nk in c:
                return i
    return None


# --------------------------------------------------------------------------------------
# raw reading
# --------------------------------------------------------------------------------------

def _sniff_sep(lines: List[str]) -> str:
    """
    Pick the delimiter that splits the probe lines into a CONSISTENT number of
    fields (>= 2).  Counting raw occurrences is not enough: a European file like
    "1,5;2,7;3,1" contains more commas than semicolons even though ';' is the
    real separator.
    """
    best, best_score = r"\s+", -1.0
    for cand in (";", "\t", ",", r"\s+"):
        counts = []
        for l in lines:
            if not l.strip():
                continue
            parts = _re.split(cand, l.strip()) if cand == r"\s+" else l.split(cand)
            counts.append(len([p for p in parts if p != ""]))
        if not counts:
            continue
        ncol = max(set(counts), key=counts.count)
        if ncol < 2:
            continue
        consistency = counts.count(ncol) / len(counts)
        # prefer consistent splits, then more columns
        score = consistency * 10 + min(ncol, 8) * 0.1
        if score > best_score:
            best, best_score = cand, score
    return best


def list_sheets(path: str) -> List[str]:
    if os.path.splitext(path)[1].lower() in EXCEL_EXT:
        try:
            return pd.ExcelFile(path).sheet_names
        except Exception:
            return []
    return []


def read_table(path: str, sheet: Optional[str] = None,
               header_row: Optional[int] = None) -> pd.DataFrame:
    """Read any supported file into a DataFrame, tolerating messy text formats."""
    ext = os.path.splitext(path)[1].lower()
    if ext in EXCEL_EXT:
        df = pd.read_excel(path, sheet_name=sheet or 0,
                           header=0 if header_row is None else header_row)
        return _clean(df)

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    lines = [l for l in raw.splitlines() if l.strip()]
    # drop leading comment/metadata lines
    body: List[str] = []
    for l in lines:
        s = l.strip()
        if s.startswith(("#", "%", "!", ";")):
            continue
        body.append(l)
    if not body:
        raise ValueError("File contains no data rows: %s" % os.path.basename(path))

    probe = body[:40]
    sep = _sniff_sep(probe)

    text = "\n".join(body)
    # Decimal comma: only meaningful when the separator is not the comma itself.
    dec = "."
    if sep in (";", "\t", r"\s+"):
        if len(_re.findall(r"\d,\d", text)) > 3:
            dec = ","

    try:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python",
                         header=0 if header_row is None else header_row, decimal=dec)
    except Exception:
        df = pd.read_csv(io.StringIO(text), sep=sep, engine="python", header=None, decimal=dec)

    # header row that is actually data -> reread headerless
    if df.shape[1] >= 2:
        first = df.columns.astype(str)
        numeric_header = all(_re.fullmatch(r"[-+]?[\d.eE+\-]+", c.strip()) is not None
                             for c in first[:2])
        if numeric_header:
            df = pd.read_csv(io.StringIO(text), sep=sep, engine="python",
                             header=None, decimal=dec)
            df.columns = ["col%d" % (i + 1) for i in range(df.shape[1])]
    return _clean(df)


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = pd.to_numeric(
                df[c].astype(str).str.replace(",", ".", regex=False).str.strip(),
                errors="coerce")
    df = df.dropna(axis=1, how="all")
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------
# column mapping
# --------------------------------------------------------------------------------------

@dataclass
class ColumnMap:
    freq: Optional[int] = None
    re: Optional[int] = None
    im: Optional[int] = None
    tau: Optional[int] = None
    gamma: Optional[int] = None
    im_is_negated: bool = False       # column already holds -Z''


def guess_columns(df: pd.DataFrame, kind: str = "full") -> ColumnMap:
    cols = list(df.columns)
    cm = ColumnMap()
    if kind == "drt":
        cm.tau = _match(cols, TAU_KEYS)
        cm.gamma = _match(cols, GAMMA_KEYS)
        if cm.tau is None and len(cols) >= 2:
            cm.tau = 0
        if cm.gamma is None and len(cols) >= 2:
            cm.gamma = 1
        return cm

    cm.freq = _match(cols, FREQ_KEYS)
    cm.re = _match(cols, RE_KEYS)
    cm.im = _match(cols, IM_KEYS)
    if cm.im is not None:
        cm.im_is_negated = _norm(cols[cm.im]).startswith("-")

    # positional fallbacks
    n = len(cols)
    if cm.freq is None and n >= 1:
        cm.freq = 0
    if kind == "full":
        if cm.re is None and n >= 2:
            cm.re = 1
        if cm.im is None and n >= 3:
            cm.im = 2
    elif kind == "re":
        if cm.re is None and n >= 2:
            cm.re = 1
        cm.im = None
    elif kind == "im":
        if cm.im is None and n >= 2:
            cm.im = 1
        cm.re = None
    return cm


def extract(df: pd.DataFrame, cm: ColumnMap) -> Dict[str, np.ndarray]:
    cols = list(df.columns)

    def col(i):
        if i is None or i < 0 or i >= len(cols):
            return None
        return pd.to_numeric(df[cols[i]], errors="coerce").to_numpy(float)

    out: Dict[str, np.ndarray] = {}
    for key, idx in (("f", cm.freq), ("z_re", cm.re), ("z_im", cm.im),
                     ("tau", cm.tau), ("gamma", cm.gamma)):
        v = col(idx)
        if v is not None:
            out[key] = v
    if "z_im" in out and cm.im_is_negated:
        out["z_im"] = -out["z_im"]

    keys = [k for k in out]
    if keys:
        mask = np.ones(len(out[keys[0]]), bool)
        for k in keys:
            mask &= np.isfinite(out[k])
        for k in keys:
            out[k] = out[k][mask]
    return out


def fix_imag_sign(z_im: np.ndarray, mode: str = "auto") -> Tuple[np.ndarray, bool]:
    """
    Internal convention: capacitive behaviour -> Z'' negative.
    Many instruments export -Z''.  'auto' flips when the data are mostly positive.
    """
    z = np.asarray(z_im, float)
    if mode == "flip":
        return -z, True
    if mode == "as_is":
        return z, False
    if z.size and np.sum(z > 0) > 0.7 * z.size:
        return -z, True
    return z, False


def merge_re_im(f_re, z_re, f_im, z_im, tol: float = 1e-6):
    """Merge separately supplied Re and Im files onto a common frequency axis."""
    f_re = np.asarray(f_re, float); z_re = np.asarray(z_re, float)
    f_im = np.asarray(f_im, float); z_im = np.asarray(z_im, float)

    if f_re.size == f_im.size and np.allclose(f_re, f_im, rtol=1e-4, atol=0):
        return f_re, z_re, z_im, "identical frequency axes"

    lo = max(f_re.min(), f_im.min())
    hi = min(f_re.max(), f_im.max())
    if hi <= lo:
        raise ValueError("The Re and Im files have no overlapping frequency range.")
    base = f_re[(f_re >= lo) & (f_re <= hi)]
    if base.size < 4:
        base = np.logspace(np.log10(lo), np.log10(hi), 40)
    o = np.argsort(f_im)
    zi = np.interp(np.log10(base), np.log10(f_im[o]), z_im[o])
    o2 = np.argsort(f_re)
    zr = np.interp(np.log10(base), np.log10(f_re[o2]), z_re[o2])
    return base, zr, zi, ("interpolated onto %d common frequencies (%.4g - %.4g Hz)"
                          % (base.size, lo, hi))
