#!/usr/bin/env python3
"""Check that FORM and Forcer are installed correctly for this repository (about 1 minute).

1. finds the FORM executable (environment variable FORM_BIN, default: `form` on the PATH) and prints its version;
2. finds Forcer (environment variable FORCER_PATH, default: ~/forcer) and checks for forcer.h;
3. recomputes three four-loop diagrams from scratch (<JJ> y^6, <JJ> lambda^3, <TT> y^4 lambda) at the
   expansion depth used for the paper and requires EXACT agreement with the cached results in data/.

Run from the repository root:   python3 scripts/test_form_setup.py
Exit code 0 means the installation reproduces the published integrals."""
import json, os, shutil, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fractions import Fraction

import sympy as sp

from ctcj import forcer as F

ROOT = os.path.join(os.path.dirname(__file__), "..")
TESTS = [("J", "J60_561"), ("J", "J03_006"), ("T", None)]


def fail(msg):
    print("FAIL:", msg)
    sys.exit(1)


exe = shutil.which(F.FORM_BIN) or (F.FORM_BIN if os.path.isfile(F.FORM_BIN) else None)
if not exe:
    fail(f"FORM executable '{F.FORM_BIN}' not found. Install FORM (see INSTALL_FORM.md), then either put `form` on "
         "your PATH or set FORM_BIN=/full/path/to/form.")
v = subprocess.run([exe, "-v"], capture_output=True, text=True)
ver = (v.stdout or v.stderr).strip().splitlines()[0] if (v.stdout or v.stderr) else "?"
print("FORM:  ", exe, "->", ver)
if not os.path.isfile(os.path.join(F.FORCER_PATH, "forcer.h")):
    fail(f"Forcer not found: {F.FORCER_PATH}/forcer.h is missing. Clone https://github.com/benruijl/forcer to ~/forcer "
         "or set FORCER_PATH=/path/to/forcer.")
print("Forcer:", F.FORCER_PATH)

L = {"ep": F.EP, "z3": sp.Symbol("z3"), "z4": sp.Symbol("z4"), "z5": sp.Symbol("z5"), "I": sp.I}
ok = True
for kind, name in TESTS:
    recs = [json.loads(l) for l in open(os.path.join(ROOT, "data", f"diagrams4_{kind}.jsonl"))]
    if name is None:   # a quick <TT> diagram with a quartic vertex
        r = next(r for r in recs if r["nQ"] == 1 and r["X"] != "0")
    else:
        r = next(r for r in recs if r["name"] == name)
    c = dict(ops=tuple(r["ops"]), ferm=tuple(tuple(f) for f in r["ferm"]),
             sig=tuple((tuple(a), tuple(b)) for a, b in r["sig"]), nY=r["nY"], nQ=r["nQ"], kind=kind,
             name=r["name"] + "_selftest", weight=Fraction(r["weight"]))
    try:
        X = F.run_class(c, os.path.join(ROOT, "build", "selftest"), depth=r["depth"])
    except Exception as e:
        fail(f"FORM/Forcer run failed for {r['name']}: {e}\nSee INSTALL_FORM.md, section 'Troubleshooting'.")
    same = sp.simplify(sp.expand(X - sp.sympify(r["X"], locals=L))) == 0
    ok &= same
    print(f"  {r['name']:8s} <{kind}{kind}> nY={r['nY']} nQ={r['nQ']}: {'identical to cached result' if same else 'DIFFERENT'}")
if not ok:
    fail("results differ from the cached integrals; please report the FORM version printed above.")
print("PASS: FORM and Forcer reproduce the published four-loop integrals exactly.")
