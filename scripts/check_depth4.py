#!/usr/bin/env python3
"""Re-run four-loop diagrams at Forcer depth 4 and compare every pole coefficient ep^-4 ... ep^-1 with the
stored depth-3 values (the ep^-1 coefficient is what enters C at O(eps^3)).  Run from the repository root:
    python3 scripts/check_depth4.py T T60_053 T60_255 ...
    python3 scripts/check_depth4.py J J60_123 J41_007 ..."""
import json, sys, time
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sympy as sp
from fractions import Fraction
from ctcj.forcer import run_class, EP
kind, names = sys.argv[1], sys.argv[2:]
recs = {json.loads(l)["name"]: json.loads(l) for l in open(f"data/diagrams4_{kind}.jsonl")}
L = {"ep": EP, "z3": sp.Symbol("z3"), "z4": sp.Symbol("z4"), "z5": sp.Symbol("z5"), "I": sp.I}
ok = True
for n in names:
    r = recs[n]
    c = dict(ops=tuple(r["ops"]), ferm=tuple(tuple(f) for f in r["ferm"]), sig=tuple((tuple(a), tuple(b)) for a, b in r["sig"]),
             nY=r["nY"], nQ=r["nQ"], kind=kind, name=n + "_d4", weight=Fraction(r["weight"]))
    t0 = time.time()
    X4 = run_class(c, f"build/form4d_{kind}", depth=4)
    X3 = sp.sympify(r["X"], locals=L)
    d = [sp.simplify(sp.expand(X4 - X3).coeff(EP, k)) for k in (-4, -3, -2, -1)]
    ok &= all(x == 0 for x in d)
    print(n, r["ops"], f"nY={r['nY']} nQ={r['nQ']}", f"{time.time()-t0:.0f}s", "pole differences depth4-depth3:", d, flush=True)
print("PASS" if ok else "FAIL")
