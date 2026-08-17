"""
Import wizard.

Handles the three ways a user supplies data:
  1. one file per sample containing f, Re(Z), Im(Z)
  2. two files per sample: one with Re, one with Im  (merged onto a common axis)
  3. a precomputed DRT curve (tau, gamma) for display / comparison only

Every guess is shown and can be overridden before the sample is created.
"""
from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import numpy as np
from ..core.compat import trapezoid

from ..core import io_data
from ..core.model import DRTResult, Sample
from .widgets import PlotPane, Tooltip, fit_to_screen


class ImportDialog(tk.Toplevel):
    """Modal import wizard.  Result is in `self.samples` (list) after close."""

    def __init__(self, master, existing_names: List[str] = None):
        super().__init__(master)
        self.title("Import measurement data")
        # clamp to the real screen; a 1150x760 modal on a 768-tall display puts
        # the Import/Cancel buttons below the desktop and the dialog can never
        # be completed, which looks like "there is no import option"
        fit_to_screen(self, 1150, 760, min_w=760, min_h=480)
        self.transient(master)
        self.samples: List[Sample] = []
        self.existing = existing_names or []
        self._frames: Dict[str, "object"] = {}

        self._build()
        self.grab_set()

    # ------------------------------------------------------------------ layout
    def _build(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill="x")
        ttk.Label(top, text="Import measurement data", style="Header.TLabel").pack(anchor="w")
        ttk.Label(top, wraplength=1100, foreground="#555555",
                  text=("Add one row per file. Set the ROLE of each file: a complete spectrum "
                        "(f, Re, Im), a real-part-only file, an imaginary-part-only file, or a "
                        "precomputed DRT curve. Files that share a GROUP name are merged into a "
                        "single sample - this is how you pair a separate Re file with its Im "
                        "file.")).pack(anchor="w", pady=(4, 0))

        btns = ttk.Frame(self, padding=(10, 4))
        btns.pack(fill="x")
        ttk.Button(btns, text="Add files...", command=self.add_files,
                   style="Accent.TButton").pack(side="left")
        ttk.Button(btns, text="Remove selected", command=self.remove_selected).pack(
            side="left", padx=6)
        ttk.Button(btns, text="Auto-group by filename", command=self.auto_group).pack(
            side="left", padx=6)
        ttk.Label(btns, text="   Tip: 'cellA_re.csv' and 'cellA_im.csv' auto-group as 'cellA'.",
                  foreground="#777777").pack(side="left")

        # the action footer is created and packed to the BOTTOM before the
        # expanding body, so Import/Cancel stay visible at any window size
        foot = ttk.Frame(self, padding=10)
        foot.pack(side="bottom", fill="x")
        self.status = ttk.Label(foot, text="No files added.", foreground="#555555")
        self.status.pack(side="left")
        ttk.Button(foot, text="Cancel", command=self._cancel).pack(side="right")
        ttk.Button(foot, text="Import", command=self._ok,
                   style="Accent.TButton").pack(side="right", padx=6)

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=10, pady=6)

        left = ttk.Frame(body)
        body.add(left, weight=3)
        cols = ("file", "role", "group", "sheet", "freq", "re", "im", "rows")
        self.tree = ttk.Treeview(left, columns=cols, show="headings", height=11)
        heads = [("file", "File", 220), ("role", "Role", 95), ("group", "Group (sample)", 150),
                 ("sheet", "Sheet", 80), ("freq", "f column", 95), ("re", "Re column", 95),
                 ("im", "Im column", 95), ("rows", "Rows", 55)]
        for c, t, w in heads:
            self.tree.heading(c, text=t)
            self.tree.column(c, width=w, anchor="center")
        vs = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vs.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        right = ttk.Frame(body)
        body.add(right, weight=2)
        self.editor = ttk.LabelFrame(right, text="Selected file", padding=8,
                                     style="Section.TLabelframe")
        self.editor.pack(fill="x")
        self._build_editor(self.editor)

        self.preview = PlotPane(right, figsize=(5.2, 3.4), dpi=96)
        self.preview.pack(fill="both", expand=True, pady=(8, 0))

        self.rows: List[dict] = []

    def _build_editor(self, p):
        r = 0
        ttk.Label(p, text="Role").grid(row=r, column=0, sticky="w")
        self.v_role = tk.StringVar(value="full")
        cb = ttk.Combobox(p, textvariable=self.v_role, state="readonly", width=28,
                          values=["full  (f, Re, Im)", "re    (f, Re only)",
                                  "im    (f, Im only)", "drt   (tau, gamma)"])
        cb.grid(row=r, column=1, sticky="ew", pady=2)
        cb.bind("<<ComboboxSelected>>", lambda e: self._apply_role())
        r += 1

        ttk.Label(p, text="Group / sample name").grid(row=r, column=0, sticky="w")
        self.v_group = tk.StringVar()
        e = ttk.Entry(p, textvariable=self.v_group, width=30)
        e.grid(row=r, column=1, sticky="ew", pady=2)
        e.bind("<KeyRelease>", lambda ev: self._commit(refresh_preview=False))
        Tooltip(e, "Files sharing this name become one sample.")
        r += 1

        ttk.Label(p, text="Sheet (Excel)").grid(row=r, column=0, sticky="w")
        self.v_sheet = tk.StringVar()
        self.cb_sheet = ttk.Combobox(p, textvariable=self.v_sheet, state="readonly", width=28)
        self.cb_sheet.grid(row=r, column=1, sticky="ew", pady=2)
        self.cb_sheet.bind("<<ComboboxSelected>>", lambda e: self._reload_current())
        r += 1

        for key, label in (("freq", "Frequency column"), ("re", "Real column"),
                           ("im", "Imag column"), ("tau", "tau column"),
                           ("gamma", "gamma column")):
            ttk.Label(p, text=label).grid(row=r, column=0, sticky="w")
            var = tk.StringVar()
            cb = ttk.Combobox(p, textvariable=var, state="readonly", width=28)
            cb.grid(row=r, column=1, sticky="ew", pady=2)
            cb.bind("<<ComboboxSelected>>", lambda e: self._commit())
            setattr(self, "v_" + key, var)
            setattr(self, "cb_" + key, cb)
            r += 1

        self.v_neg = tk.BooleanVar(value=False)
        c = ttk.Checkbutton(p, text="This column holds -Z'' (negate on import)",
                            variable=self.v_neg, command=lambda: self._commit())
        c.grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 0))
        Tooltip(c, "Internal convention: capacitive behaviour has Z'' negative. Many "
                   "instruments export -Z''. Auto-detection handles the common cases.")
        r += 1
        ttk.Label(p, text="Frequency unit").grid(row=r, column=0, sticky="w")
        self.v_funit = tk.StringVar(value="Hz")
        cbu = ttk.Combobox(p, textvariable=self.v_funit, state="readonly", width=28,
                           values=["Hz", "kHz", "MHz", "rad/s"])
        cbu.grid(row=r, column=1, sticky="ew", pady=2)
        cbu.bind("<<ComboboxSelected>>", lambda e: self._commit())
        p.columnconfigure(1, weight=1)

    # ------------------------------------------------------------------ files
    def add_files(self):
        paths = filedialog.askopenfilenames(
            parent=self, title="Select data files",
            filetypes=[("All supported", "*.csv *.txt *.dat *.tsv *.asc *.prn *.xlsx *.xls *.mpt *.z"),
                       ("CSV / text", "*.csv *.txt *.dat *.tsv *.asc *.prn"),
                       ("Excel", "*.xlsx *.xls"), ("All files", "*.*")])
        if not paths:
            return
        for p in paths:
            self._add_one(p)
        self.auto_group()
        self._refresh()

    def _add_one(self, path):
        sheets = io_data.list_sheets(path)
        row = dict(path=path, role="full", group="", sheet=(sheets[0] if sheets else ""),
                   sheets=sheets, df=None, cm=None, neg=False, funit="Hz", error="")
        try:
            row["df"] = io_data.read_table(path, row["sheet"] or None)
            row["cm"] = io_data.guess_columns(row["df"], "full")
            row["neg"] = bool(row["cm"].im_is_negated)
        except Exception as ex:
            row["error"] = str(ex)
        base = os.path.splitext(os.path.basename(path))[0]
        row["group"] = _strip_role_suffix(base)
        # guess the role from the filename
        low = base.lower()
        if any(k in low for k in ("_im", "-im", " im", "imag", "zim")):
            row["role"] = "im"
        elif any(k in low for k in ("_re", "-re", " re", "real", "zre")):
            row["role"] = "re"
        elif any(k in low for k in ("drt", "gamma", "tau")):
            row["role"] = "drt"
        if row["df"] is not None:
            row["cm"] = io_data.guess_columns(row["df"], row["role"])
        self.rows.append(row)

    def remove_selected(self):
        for iid in self.tree.selection():
            i = int(iid)
            if 0 <= i < len(self.rows):
                self.rows[i] = None
        self.rows = [r for r in self.rows if r is not None]
        self._refresh()

    def auto_group(self):
        for r in self.rows:
            base = os.path.splitext(os.path.basename(r["path"]))[0]
            r["group"] = _strip_role_suffix(base)
        self._refresh()

    # ------------------------------------------------------------------ table
    def _refresh(self):
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(self.rows):
            cols = list(r["df"].columns) if r["df"] is not None else []
            cm = r["cm"]

            def nm(idx):
                return cols[idx] if (cm and idx is not None and 0 <= idx < len(cols)) else "-"
            if r["role"] == "drt":
                fcol, rcol, icol = nm(cm.tau if cm else None), nm(cm.gamma if cm else None), "-"
            else:
                fcol = nm(cm.freq if cm else None)
                rcol = nm(cm.re if cm else None)
                icol = nm(cm.im if cm else None)
            n = len(r["df"]) if r["df"] is not None else 0
            self.tree.insert("", "end", iid=str(i), values=(
                os.path.basename(r["path"]), r["role"], r["group"], r["sheet"] or "-",
                fcol, rcol, icol, n if not r["error"] else "ERR"))
        groups = {}
        for r in self.rows:
            groups.setdefault(r["group"], []).append(r["role"])
        self.status.configure(
            text="%d file(s) -> %d sample(s): %s" % (
                len(self.rows), len(groups),
                ", ".join("%s [%s]" % (g, "+".join(v)) for g, v in list(groups.items())[:6])))

    def _current(self) -> Optional[dict]:
        sel = self.tree.selection()
        if not sel:
            return None
        i = int(sel[0])
        return self.rows[i] if 0 <= i < len(self.rows) else None

    def _on_select(self, _=None):
        r = self._current()
        if r is None:
            return
        self.v_role.set({"full": "full  (f, Re, Im)", "re": "re    (f, Re only)",
                         "im": "im    (f, Im only)", "drt": "drt   (tau, gamma)"}[r["role"]])
        self.v_group.set(r["group"])
        self.cb_sheet.configure(values=r["sheets"] or [])
        self.v_sheet.set(r["sheet"] or "")
        self.v_neg.set(r["neg"])
        self.v_funit.set(r.get("funit", "Hz"))
        cols = list(r["df"].columns) if r["df"] is not None else []
        opts = ["(none)"] + cols
        cm = r["cm"]
        for key in ("freq", "re", "im", "tau", "gamma"):
            cb = getattr(self, "cb_" + key)
            cb.configure(values=opts)
            idx = getattr(cm, key, None) if cm else None
            getattr(self, "v_" + key).set(cols[idx] if (idx is not None and idx < len(cols))
                                          else "(none)")
        self._preview(r)

    def _apply_role(self):
        r = self._current()
        if r is None:
            return
        r["role"] = self.v_role.get().split()[0]
        if r["df"] is not None:
            r["cm"] = io_data.guess_columns(r["df"], r["role"])
            r["neg"] = bool(r["cm"].im_is_negated)
        self._on_select()
        self._refresh()
        self._reselect(r)

    def _reload_current(self):
        r = self._current()
        if r is None:
            return
        r["sheet"] = self.v_sheet.get()
        try:
            r["df"] = io_data.read_table(r["path"], r["sheet"] or None)
            r["cm"] = io_data.guess_columns(r["df"], r["role"])
            r["error"] = ""
        except Exception as ex:
            r["error"] = str(ex)
        self._on_select()
        self._refresh()
        self._reselect(r)

    def _commit(self, refresh_preview=True):
        r = self._current()
        if r is None or r["df"] is None:
            return
        cols = list(r["df"].columns)
        r["group"] = self.v_group.get().strip() or r["group"]
        r["neg"] = bool(self.v_neg.get())
        r["funit"] = self.v_funit.get()
        cm = r["cm"] or io_data.ColumnMap()
        for key in ("freq", "re", "im", "tau", "gamma"):
            name = getattr(self, "v_" + key).get()
            setattr(cm, key, cols.index(name) if name in cols else None)
        cm.im_is_negated = r["neg"]
        r["cm"] = cm
        self._refresh()
        self._reselect(r)
        if refresh_preview:
            self._preview(r)

    def _reselect(self, r):
        try:
            i = self.rows.index(r)
            self.tree.selection_set(str(i))
        except Exception:
            pass

    # ------------------------------------------------------------------ preview
    def _preview(self, r):
        fig = self.preview.figure
        fig.clear()
        ax = fig.add_subplot(111)
        if r["error"]:
            ax.text(0.5, 0.5, "Could not read file:\n%s" % r["error"], ha="center",
                    va="center", color="#c62828", fontsize=8, wrap=True)
            ax.axis("off")
            self.preview.draw()
            return
        try:
            d = io_data.extract(r["df"], r["cm"])
            if r["role"] == "drt" and "tau" in d and "gamma" in d:
                ax.semilogx(d["tau"], d["gamma"], "-", color="#2e7d32")
                ax.set_xlabel("tau / s")
                ax.set_ylabel("gamma / ohm")
            elif "f" in d:
                fv = _convert_freq(d["f"], r.get("funit", "Hz"))
                if "z_re" in d and "z_im" in d:
                    zi, _ = io_data.fix_imag_sign(d["z_im"], "auto")
                    ax.plot(d["z_re"], -zi, "o-", ms=3, color="#1f5fa8")
                    ax.set_xlabel("Z' / ohm")
                    ax.set_ylabel("-Z'' / ohm")
                elif "z_re" in d:
                    ax.semilogx(fv, d["z_re"], "o-", ms=3, color="#1f5fa8")
                    ax.set_xlabel("f / Hz")
                    ax.set_ylabel("Z' / ohm")
                elif "z_im" in d:
                    ax.semilogx(fv, d["z_im"], "o-", ms=3, color="#c62828")
                    ax.set_xlabel("f / Hz")
                    ax.set_ylabel("Z'' / ohm")
            ax.grid(alpha=0.35, lw=0.4)
            ax.set_title(os.path.basename(r["path"]), fontsize=8)
            fig.tight_layout()
        except Exception as ex:
            ax.text(0.5, 0.5, "Preview failed:\n%s" % ex, ha="center", va="center",
                    fontsize=8, color="#c62828")
            ax.axis("off")
        self.preview.draw()

    # ------------------------------------------------------------------ finish
    def _ok(self):
        groups: Dict[str, List[dict]] = {}
        for r in self.rows:
            if r["error"]:
                messagebox.showerror("Import", "File could not be read:\n%s\n\n%s"
                                     % (r["path"], r["error"]), parent=self)
                return
            groups.setdefault(r["group"] or "sample", []).append(r)

        out: List[Sample] = []
        for gname, items in groups.items():
            try:
                s = _build_sample(gname, items)
            except Exception as ex:
                messagebox.showerror("Import", "Group '%s' could not be built:\n%s"
                                     % (gname, ex), parent=self)
                return
            if s is not None:
                out.append(s)
        if not out:
            messagebox.showwarning("Import", "Nothing to import.", parent=self)
            return
        self.samples = out
        self.destroy()

    def _cancel(self):
        self.samples = []
        self.destroy()


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def _strip_role_suffix(base: str) -> str:
    low = base
    for suf in ("_re", "-re", " re", "_im", "-im", " im", "_real", "_imag",
                "_zre", "_zim", "_drt", "-drt", " drt"):
        if low.lower().endswith(suf):
            return low[: -len(suf)].strip(" _-") or low
    return low


def _convert_freq(f: np.ndarray, unit: str) -> np.ndarray:
    f = np.asarray(f, float)
    if unit == "kHz":
        return f * 1e3
    if unit == "MHz":
        return f * 1e6
    if unit == "rad/s":
        return f / (2 * np.pi)
    return f


def _build_sample(name: str, items: List[dict]) -> Optional[Sample]:
    """Merge all files of one group into a single Sample."""
    s = Sample(name=name)
    f_re = z_re = f_im = z_im = None
    src = []

    for r in items:
        src.append(r["path"])
        d = io_data.extract(r["df"], r["cm"])
        role = r["role"]
        if role == "drt":
            if "tau" not in d or "gamma" not in d:
                raise ValueError("DRT file needs a tau column and a gamma column.")
            o = np.argsort(d["tau"])
            res = DRTResult(mode="combined", tau=d["tau"][o], gamma=d["gamma"][o],
                            imported=True)
            res.r_pol = float(trapezoid(res.gamma, np.log(res.tau)))
            from ..core.peaks import analyse
            analyse(res)
            s.results["combined"] = res
            continue

        if "f" not in d:
            raise ValueError("No frequency column identified in %s"
                             % os.path.basename(r["path"]))
        fv = _convert_freq(d["f"], r.get("funit", "Hz"))
        if role == "full":
            if "z_re" not in d or "z_im" not in d:
                raise ValueError("A 'full' file needs both a real and an imaginary column.")
            f_re, z_re = fv, d["z_re"]
            f_im, z_im = fv, d["z_im"]
        elif role == "re":
            if "z_re" not in d:
                raise ValueError("No real column identified in %s"
                                 % os.path.basename(r["path"]))
            f_re, z_re = fv, d["z_re"]
        elif role == "im":
            if "z_im" not in d:
                raise ValueError("No imaginary column identified in %s"
                                 % os.path.basename(r["path"]))
            f_im, z_im = fv, d["z_im"]

    s.source_files = src

    if f_re is not None and f_im is not None:
        f, zr, zi, how = io_data.merge_re_im(f_re, z_re, f_im, z_im)
        zi, flipped = io_data.fix_imag_sign(zi, "auto")
        s.f, s.z_re, s.z_im = f, zr, zi
        s.meta["merge"] = how
        if flipped:
            s.meta["imag_sign"] = "input treated as -Z'' and negated"
    elif f_re is not None:
        raise ValueError("Group '%s' has a real-part file but no imaginary-part file. "
                         "DRT needs both; add the Im file to the same group." % name)
    elif f_im is not None:
        raise ValueError("Group '%s' has an imaginary-part file but no real-part file. "
                         "DRT needs both; add the Re file to the same group." % name)
    elif not s.results:
        return None
    return s
