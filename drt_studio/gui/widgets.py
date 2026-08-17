"""Small reusable Tkinter helpers."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, List, Optional

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

ACCENT = "#1f6390"
DARK = "#0d2f4c"
BG = "#f4f6f8"


def fit_to_screen(win, want_w: int, want_h: int, min_w: int = 720,
                  min_h: int = 520, margin: int = 90, center: bool = True):
    """
    Size a window so it ALWAYS fits on the actual screen.

    Hard-coding a geometry like 1420x900 puts most of the UI off-screen on a
    1366x768 laptop, and a minsize larger than the screen makes it impossible
    to shrink the window back into view.  This clamps both the requested size
    and the minimum size to what the display can really show, then centres it.
    """
    win.update_idletasks()
    sw = win.winfo_screenwidth()
    sh = win.winfo_screenheight()
    w = max(320, min(want_w, sw - margin // 2))
    h = max(240, min(want_h, sh - margin))
    # the minimum must never exceed what the screen can display
    win.minsize(min(min_w, w), min(min_h, h))
    if center:
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 3)
        win.geometry("%dx%d+%d+%d" % (w, h, x, y))
    else:
        win.geometry("%dx%d" % (w, h))
    return w, h


class PlotPane(ttk.Frame):
    """A matplotlib figure with the standard navigation toolbar and a save button."""

    def __init__(self, master, figsize=(7.5, 5.0), dpi=100, **kw):
        super().__init__(master, **kw)
        self.figure = Figure(figsize=figsize, dpi=dpi, facecolor="white")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        bar = ttk.Frame(self)
        bar.pack(side="bottom", fill="x")
        self.toolbar = NavigationToolbar2Tk(self.canvas, bar, pack_toolbar=False)
        self.toolbar.update()
        self.toolbar.pack(side="left", fill="x")
        self.canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

    def draw(self):
        try:
            self.canvas.draw_idle()
        except Exception:
            pass

    def clear(self):
        self.figure.clear()
        self.draw()

    def save(self, path, dpi=180):
        self.figure.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        return path


class ScrollFrame(ttk.Frame):
    """Vertically scrollable container."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self.canvas = tk.Canvas(self, highlightthickness=0, bg=BG)
        vs = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.body = ttk.Frame(self.canvas)
        self.body.bind("<Configure>",
                       lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self._win = self.canvas.create_window((0, 0), window=self.body, anchor="nw")
        self.canvas.bind("<Configure>",
                         lambda e: self.canvas.itemconfig(self._win, width=e.width))
        self.canvas.configure(yscrollcommand=vs.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        vs.pack(side="right", fill="y")
        # bind on the widgets themselves (not bind_all): with several scroll
        # panes in one app, bind_all makes every one of them scroll at once
        for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self.canvas.bind(seq, self._wheel, add="+")
        self.body.bind("<Enter>", self._hook_children)

    def _hook_children(self, _=None):
        """Make the wheel work while hovering any child widget."""
        def hook(w):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                try:
                    w.bind(seq, self._wheel, add="+")
                except Exception:
                    pass
            for c in w.winfo_children():
                hook(c)
        hook(self.body)

    def _wheel(self, ev):
        try:
            d = 0
            if getattr(ev, "num", None) == 4:
                d = -1
            elif getattr(ev, "num", None) == 5:
                d = 1
            elif getattr(ev, "delta", 0):
                d = -1 if ev.delta > 0 else 1
            self.canvas.yview_scroll(d, "units")
        except Exception:
            pass


class SortableTree(ttk.Frame):
    """Treeview with column sorting, copy-to-clipboard and CSV export."""

    def __init__(self, master, columns: List[str], widths: Optional[List[int]] = None,
                 height=12, **kw):
        super().__init__(master, **kw)
        self.columns = list(columns)
        self.tree = ttk.Treeview(self, columns=self.columns, show="headings", height=height)
        for i, c in enumerate(self.columns):
            self.tree.heading(c, text=c, command=lambda cc=c: self.sort_by(cc))
            self.tree.column(c, width=(widths[i] if widths and i < len(widths) else 110),
                             anchor="center", stretch=True)
        vs = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        hs = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vs.grid(row=0, column=1, sticky="ns")
        hs.grid(row=1, column=0, sticky="ew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self._asc = {}
        self.tree.bind("<Control-c>", lambda e: self.copy())

    def set_rows(self, rows):
        self.tree.delete(*self.tree.get_children())
        for i, r in enumerate(rows):
            self.tree.insert("", "end", values=[("" if v is None else v) for v in r],
                             tags=("odd",) if i % 2 else ("even",))
        self.tree.tag_configure("odd", background="#eef3f8")
        self.tree.tag_configure("even", background="#ffffff")

    def get_rows(self):
        return [self.tree.item(i, "values") for i in self.tree.get_children()]

    def sort_by(self, col):
        idx = self.columns.index(col)
        asc = not self._asc.get(col, False)
        self._asc[col] = asc
        rows = self.get_rows()

        def key(r):
            v = r[idx]
            try:
                return (0, float(str(v).replace("%", "")))
            except Exception:
                return (1, str(v))
        self.set_rows(sorted(rows, key=key, reverse=not asc))

    def copy(self):
        rows = self.get_rows()
        txt = "\t".join(self.columns) + "\n" + "\n".join("\t".join(map(str, r)) for r in rows)
        try:
            self.clipboard_clear()
            self.clipboard_append(txt)
        except Exception:
            pass
        return txt

    def to_csv(self, path):
        import csv
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(self.columns)
            w.writerows(self.get_rows())
        return path


class LabeledEntry(ttk.Frame):
    def __init__(self, master, label, value="", width=12, tooltip="", **kw):
        super().__init__(master, **kw)
        self.var = tk.StringVar(value=str(value))
        ttk.Label(self, text=label).pack(side="left")
        e = ttk.Entry(self, textvariable=self.var, width=width)
        e.pack(side="left", padx=(6, 0))
        if tooltip:
            Tooltip(e, tooltip)

    def get(self, cast=str, default=None):
        try:
            return cast(self.var.get())
        except Exception:
            return default

    def set(self, v):
        self.var.set(str(v))


class Tooltip:
    def __init__(self, widget, text, delay=450):
        self.widget, self.text, self.delay = widget, text, delay
        self._id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _=None):
        self._cancel()
        self._id = self.widget.after(self.delay, self._show)

    def _cancel(self):
        if self._id:
            try:
                self.widget.after_cancel(self._id)
            except Exception:
                pass
            self._id = None

    def _show(self):
        if self._tip:
            return
        try:
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            self._tip = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry("+%d+%d" % (x, y))
            tk.Label(tw, text=self.text, justify="left", background="#ffffe0",
                     relief="solid", borderwidth=1, font=("TkDefaultFont", 8),
                     wraplength=380).pack()
        except Exception:
            self._tip = None

    def _hide(self, _=None):
        self._cancel()
        if self._tip:
            try:
                self._tip.destroy()
            except Exception:
                pass
            self._tip = None


def style_app(root):
    style = ttk.Style(root)
    for theme in ("clam", "alt", "default"):
        if theme in style.theme_names():
            style.theme_use(theme)
            break
    style.configure(".", font=("TkDefaultFont", 9))
    style.configure("TNotebook.Tab", padding=(14, 7), font=("TkDefaultFont", 9, "bold"))
    style.configure("Header.TLabel", font=("TkDefaultFont", 15, "bold"), foreground=DARK)
    style.configure("Sub.TLabel", font=("TkDefaultFont", 10), foreground=ACCENT)
    style.configure("Section.TLabelframe.Label", font=("TkDefaultFont", 9, "bold"),
                    foreground=ACCENT)
    style.configure("Accent.TButton", font=("TkDefaultFont", 9, "bold"))
    style.configure("Treeview.Heading", font=("TkDefaultFont", 8, "bold"))
    style.configure("Treeview", rowheight=21)
    return style
