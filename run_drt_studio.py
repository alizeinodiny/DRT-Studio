#!/usr/bin/env python3
"""
DRT Studio - launcher.

Usage
-----
    python run_drt_studio.py              start the graphical application
    python run_drt_studio.py --selftest   run the headless self-test (no display needed)
    python run_drt_studio.py --demo DIR   build the demo project, analyse it and export
                                          a full set of reports into DIR
    python run_drt_studio.py --diagnose   verify the numerics against a known circuit
                                          and list the installed packages
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # package root


def _selftest():
    import selftest
    return selftest.main()


MIN_PY = (3, 8)


def _preflight():
    """Fail loudly and helpfully instead of crashing mid-analysis."""
    problems = []
    if sys.version_info < MIN_PY:
        problems.append("Python %d.%d+ is required; this is %d.%d."
                        % (MIN_PY[0], MIN_PY[1],
                           sys.version_info[0], sys.version_info[1]))
    try:
        import numpy as np
        if not (hasattr(np, "trapezoid") or hasattr(np, "trapz")):
            problems.append("numpy %s has neither trapezoid nor trapz."
                            % np.__version__)
    except Exception as ex:
        problems.append("numpy is required but could not be imported: %s" % ex)
    for mod, pipname in (("scipy", "scipy"), ("matplotlib", "matplotlib"),
                         ("pandas", "pandas")):
        try:
            __import__(mod)
        except Exception:
            problems.append("%s is missing.  Fix: pip install %s" % (mod, pipname))
    if problems:
        print("DRT Studio cannot start:\n")
        for p_ in problems:
            print("  - %s" % p_)
        print("\nRun 'python run_drt_studio.py --diagnose' for a full report.")
        return False
    return True


def _diagnose():
    import matplotlib
    matplotlib.use("Agg")
    from drt_studio.core.diagnostics import run_diagnostics
    rep = run_diagnostics(progress=lambda m, f: None)
    print(rep.text())
    return 0 if rep.ok else 1


def _demo(outdir):
    import matplotlib
    matplotlib.use("Agg")
    from drt_studio.core.pipeline import (analyse_project, build_demo_project,
                                          export_comparison_bundle, export_sample_bundle)
    p = build_demo_project()
    print("Analysing %d demo samples..." % len(p.samples))
    analyse_project(p, progress=lambda m, f: None)
    made = []
    for s in p.samples:
        d = os.path.join(outdir, s.name.replace(" ", "_").replace("/", "_"))
        made += export_sample_bundle(s, p, d, want_pdf=True, want_docx=True, want_excel=True)
    made += export_comparison_bundle(p, p.samples, os.path.join(outdir, "_comparison"),
                                     want_pdf=True, want_docx=True, want_excel=True)
    print("Wrote %d files into %s" % (len(made), outdir))
    return 0


def main():
    args = sys.argv[1:]
    if not _preflight():
        return 2
    if args and args[0] == "--selftest":
        return _selftest()
    if args and args[0] in ("--diagnose", "--diagnostics"):
        return _diagnose()
    if args and args[0] == "--demo":
        out = args[1] if len(args) > 1 else "drt_demo_output"
        return _demo(out)
    try:
        import tkinter  # noqa: F401
    except Exception as ex:
        print("Tkinter is not available: %s\n\nOn Debian/Ubuntu install it with:\n"
              "    sudo apt-get install python3-tk\n\n"
              "You can still use the headless modes:\n"
              "    python run_drt_studio.py --selftest\n"
              "    python run_drt_studio.py --demo output_folder" % ex)
        return 1
    from drt_studio.gui.app import main as gui_main
    gui_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
