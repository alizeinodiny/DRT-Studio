"""
Data model for DRT Studio.

Everything in `core` is GUI-free so that it can be scripted, unit-tested and run
headlessly.  The GUI layer only orchestrates these objects.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------------------
# DRT computation settings
# --------------------------------------------------------------------------------------

MODES = ("re", "im", "combined")
MODE_LABELS = {
    "re": "DRT from the REAL part of Z",
    "im": "DRT from the IMAGINARY part of Z",
    "combined": "DRT from RE + IM combined",
}
MODE_SHORT = {"re": "Re", "im": "Im", "combined": "Combined"}
MODE_COLORS = {"re": "#1f5fa8", "im": "#c62828", "combined": "#2e7d32"}


@dataclass
class DRTSettings:
    """All tunable parameters of the DRT inversion."""

    # engine
    engine: str = "drttools"           # 'drttools' (accurate) | 'legacy' (fast)

    # discretisation
    basis: str = "gaussian"            # legacy engine only
    rbf_type: str = "Gaussian"         # DRTtools RBF; see core.rbf.RBF_TYPES
    shape_control: str = "FWHM Coefficient"   # or 'Shape Factor'
    points_per_decade: int = 10
    rbf_fwhm_factor: float = 1.0       # FWHM coefficient (DRTtools default 1.0)
    tau_extend_decades: float = 0.0    # 0 = DRTtools native grid (tau = 1/f)

    # regularisation
    lambda_value: float = 1e-3         # used only when lambda_mode == 'manual'
    # DRTtools' GUI ships a fixed lambda = 1e-3.  That is a reasonable guess for
    # noisy laboratory data but it over-smooths clean spectra (measured on the
    # synthetic cell: 3.5 % residual and one broad peak, against 0.6 % and the
    # four true peaks with GCV).  Since the cross-validation criteria are
    # available and cheap, choose lambda from the data by default.
    lambda_mode: str = "GCV"           # manual | GCV | mGCV | rGCV | LC | re-im | kf
    lambda_min: float = 1e-7
    lambda_max: float = 1e1
    lambda_steps: int = 25
    derivative_order: int = 2          # 1 or 2

    # model / fit
    include_inductance: bool = True
    induct_used: int = 1               # DRTtools: 0/2 = R_inf only, 1 = R_inf + L
    non_negative: bool = True
    # DRTtools solves the unweighted problem and its lambda scale assumes that;
    # 'unit' is therefore the default for the drttools engine.
    weighting: str = "unit"            # 'unit' | 'modulus' | 'proportional'

    # data conventions
    imag_sign: str = "auto"            # 'auto' | 'as_is' | 'flip'

    def copy(self) -> "DRTSettings":
        return DRTSettings(**asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "DRTSettings":
        base = DRTSettings()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, v)
        return base

    def describe(self) -> str:
        lam = ("auto (%s)" % self.lambda_mode) if self.lambda_mode != "manual" else "%g" % self.lambda_value
        return (
            "basis=%s, %d pts/decade, RBF FWHM x%.2f, extend %.1f dec, "
            "lambda=%s, D-order=%d, weighting=%s, non-negative=%s, inductance=%s"
            % (self.basis, self.points_per_decade, self.rbf_fwhm_factor,
               self.tau_extend_decades, lam, self.derivative_order,
               self.weighting, self.non_negative, self.include_inductance)
        )


# --------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------

@dataclass
class Peak:
    label: str = ""
    tau: float = 0.0
    freq: float = 0.0
    gamma_max: float = 0.0
    resistance: float = 0.0
    tau_lo: float = 0.0
    tau_hi: float = 0.0
    fwhm_decades: float = 0.0
    assignment: str = ""
    process: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DRTResult:
    """Outcome of one inversion (one mode) for one sample."""

    mode: str = "combined"
    tau: np.ndarray = field(default_factory=lambda: np.array([]))
    gamma: np.ndarray = field(default_factory=lambda: np.array([]))
    r_inf: float = 0.0
    inductance: float = 0.0
    r_pol: float = 0.0
    lambda_used: float = 0.0
    peaks: List[Peak] = field(default_factory=list)
    region_r: Dict[str, float] = field(default_factory=dict)
    # fit quality
    z_fit_re: np.ndarray = field(default_factory=lambda: np.array([]))
    z_fit_im: np.ndarray = field(default_factory=lambda: np.array([]))
    res_re_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    res_im_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    rms_pct: float = 0.0
    settings: Optional[DRTSettings] = None
    imported: bool = False             # True when the curve was imported, not computed
    meta: Dict[str, object] = field(default_factory=dict)   # engine provenance

    def summary(self) -> dict:
        return {
            "mode": MODE_SHORT.get(self.mode, self.mode),
            "R_inf (ohm)": self.r_inf,
            "R_pol (ohm)": self.r_pol,
            "R_total (ohm)": self.r_inf + self.r_pol,
            "L (H)": self.inductance,
            "lambda": self.lambda_used,
            "n_peaks": len(self.peaks),
            "RMS residual (%)": self.rms_pct,
        }


@dataclass
class KKResult:
    """Linear Kramers-Kronig validation result."""

    res_re_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    res_im_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    max_abs_pct: float = 0.0
    mean_abs_pct: float = 0.0
    n_rc: int = 0
    verdict: str = ""
    suspect_freqs: np.ndarray = field(default_factory=lambda: np.array([]))


# --------------------------------------------------------------------------------------
# Sample
# --------------------------------------------------------------------------------------

@dataclass
class Sample:
    """One measured cell / one data group."""

    name: str = "sample"
    f: np.ndarray = field(default_factory=lambda: np.array([]))     # Hz
    z_re: np.ndarray = field(default_factory=lambda: np.array([]))  # ohm
    z_im: np.ndarray = field(default_factory=lambda: np.array([]))  # ohm, capacitive = negative
    source_files: List[str] = field(default_factory=list)
    notes: str = ""
    meta: Dict[str, str] = field(default_factory=dict)
    results: Dict[str, DRTResult] = field(default_factory=dict)
    kk: Optional[KKResult] = None
    color: str = "#1f5fa8"

    # ---- convenience -------------------------------------------------------
    @property
    def has_eis(self) -> bool:
        return self.f.size > 0 and self.z_re.size == self.f.size and self.z_im.size == self.f.size

    @property
    def n_points(self) -> int:
        return int(self.f.size)

    @property
    def f_range(self) -> str:
        if self.f.size == 0:
            return "-"
        return "%.4g Hz - %.4g Hz" % (self.f.min(), self.f.max())

    @property
    def omega(self) -> np.ndarray:
        return 2.0 * np.pi * self.f

    @property
    def z_mod(self) -> np.ndarray:
        return np.hypot(self.z_re, self.z_im)

    def best_result(self) -> Optional[DRTResult]:
        """Preferred result for reporting: combined > im > re."""
        for m in ("combined", "im", "re"):
            if m in self.results:
                return self.results[m]
        return None

    def describe(self) -> str:
        bits = [self.name]
        if self.has_eis:
            bits.append("%d points, %s" % (self.n_points, self.f_range))
        if self.results:
            bits.append("DRT: " + ", ".join(MODE_SHORT[m] for m in MODES if m in self.results))
        return " | ".join(bits)


# --------------------------------------------------------------------------------------
# Project (collection of samples) - JSON persistence
# --------------------------------------------------------------------------------------

PALETTE = ["#1f5fa8", "#c62828", "#2e7d32", "#ef6c00", "#6a1b9a", "#00838f",
           "#ad1457", "#4e342e", "#37474f", "#9e9d24"]


@dataclass
class Project:
    name: str = "DRT project"
    samples: List[Sample] = field(default_factory=list)
    settings: DRTSettings = field(default_factory=DRTSettings)
    operator: str = ""
    cell_description: str = ("Aqueous zinc-ion cell: galvanized zinc anode, hard-carbon "
                             "cathode, ZnSO4 electrolyte")

    # ---- sample management -------------------------------------------------
    def add(self, s: Sample) -> Sample:
        base, i = s.name, 2
        names = {x.name for x in self.samples}
        while s.name in names:
            s.name = "%s (%d)" % (base, i)
            i += 1
        s.color = PALETTE[len(self.samples) % len(PALETTE)]
        self.samples.append(s)
        return s

    def remove(self, name: str) -> None:
        self.samples = [s for s in self.samples if s.name != name]

    def get(self, name: str) -> Optional[Sample]:
        for s in self.samples:
            if s.name == name:
                return s
        return None

    @property
    def names(self) -> List[str]:
        return [s.name for s in self.samples]

    # ---- persistence -------------------------------------------------------
    def save(self, path: str) -> None:
        def arr(a):
            return np.asarray(a).tolist()

        data = {
            "name": self.name,
            "operator": self.operator,
            "cell_description": self.cell_description,
            "settings": self.settings.to_dict(),
            "samples": [],
        }
        for s in self.samples:
            sd = {
                "name": s.name, "notes": s.notes, "meta": s.meta, "color": s.color,
                "source_files": s.source_files,
                "f": arr(s.f), "z_re": arr(s.z_re), "z_im": arr(s.z_im),
                "results": {},
            }
            for m, r in s.results.items():
                sd["results"][m] = {
                    "mode": r.mode, "tau": arr(r.tau), "gamma": arr(r.gamma),
                    "r_inf": r.r_inf, "inductance": r.inductance, "r_pol": r.r_pol,
                    "lambda_used": r.lambda_used, "rms_pct": r.rms_pct,
                    "imported": r.imported,
                    "region_r": r.region_r,
                    "peaks": [p.to_dict() for p in r.peaks],
                    "z_fit_re": arr(r.z_fit_re), "z_fit_im": arr(r.z_fit_im),
                    "res_re_pct": arr(r.res_re_pct), "res_im_pct": arr(r.res_im_pct),
                    "settings": r.settings.to_dict() if r.settings else None,
                }
            data["samples"].append(sd)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1)

    @staticmethod
    def load(path: str) -> "Project":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        p = Project(name=d.get("name", "DRT project"),
                    operator=d.get("operator", ""),
                    cell_description=d.get("cell_description", ""))
        p.settings = DRTSettings.from_dict(d.get("settings", {}))
        for sd in d.get("samples", []):
            s = Sample(
                name=sd["name"], notes=sd.get("notes", ""), meta=sd.get("meta", {}),
                source_files=sd.get("source_files", []),
                f=np.asarray(sd.get("f", []), float),
                z_re=np.asarray(sd.get("z_re", []), float),
                z_im=np.asarray(sd.get("z_im", []), float),
            )
            s.color = sd.get("color", "#1f5fa8")
            for m, rd in (sd.get("results") or {}).items():
                r = DRTResult(
                    mode=rd["mode"],
                    tau=np.asarray(rd["tau"], float), gamma=np.asarray(rd["gamma"], float),
                    r_inf=rd["r_inf"], inductance=rd["inductance"], r_pol=rd["r_pol"],
                    lambda_used=rd["lambda_used"], rms_pct=rd.get("rms_pct", 0.0),
                    imported=rd.get("imported", False),
                    region_r=rd.get("region_r", {}),
                    peaks=[Peak(**pk) for pk in rd.get("peaks", [])],
                    z_fit_re=np.asarray(rd.get("z_fit_re", []), float),
                    z_fit_im=np.asarray(rd.get("z_fit_im", []), float),
                    res_re_pct=np.asarray(rd.get("res_re_pct", []), float),
                    res_im_pct=np.asarray(rd.get("res_im_pct", []), float),
                    settings=DRTSettings.from_dict(rd.get("settings") or {}),
                )
                s.results[m] = r
            p.samples.append(s)
        return p


def ensure_dir(path: str) -> str:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)
    return path
