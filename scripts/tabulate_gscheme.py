#!/usr/bin/env python3
"""Tabulate the Forcer G-scheme value X summed over all diagram classes, per flavour invariant and loop order,
for <TT> and <JJ>.  These are the complete Feynman-integral input of the results: with them, the Fourier
transform, renormalisation and fixed-point reduction (paper sections 2 and 6) can be redone independently.
Writes results/gscheme_totals.md (~1 min).  Normalisation: paper appendix A."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sympy as sp
from ctcj.forcer import EP, Z3, Z4, Z5
ROOT = os.path.join(os.path.dirname(__file__), "..")
loc = {"ep": EP, "z3": Z3, "z4": Z4, "z5": Z5, "I": sp.I}


def loops(r):
    return {(0, 0): 1, (2, 0): 2, (4, 0): 3, (0, 2): 3, (6, 0): 4, (4, 1): 4, (2, 2): 4, (0, 3): 4}[(r["nY"], r["nQ"])]


lines = ["# G-scheme sums per invariant", "",
         "X summed over all diagram classes with the given flavour invariant (D = 4 - 2 ep, Q^2 = 1, per-loop normalisation ep*G(1,1)),",
         "<TT> = (z1.z2)^2 Q^4 (Q^2)^(-L ep) [ep G(1,1)]^L 2X/((D-1)(D+1)(D-2)),  <JJ> = (z1.z2) Q^2 (Q^2)^(-L ep) [ep G(1,1)]^L X/(D-1).",
         "Only the orders needed for O(eps^3) are listed: L loops through ep^(3-L) (one loop through ep^2, four loops through ep^-1).",
         "z3 = zeta_3.  Invariant names as in ctcj/formulas.py and ctcj/formulas4.py.", ""]
for kind, files in (("T", ["diagrams_T.json", "diagrams4_T.json"]), ("J", ["diagrams_J.json", "diagrams4_J.json"])):
    tot = {}
    for fn in files:
        for r in json.load(open(os.path.join(ROOT, "data", fn))):
            if r["X"] == "0":
                continue
            X = sp.sympify(r["X"], locals=loc)
            for k, c in r["decomp"].items():
                key = (loops(r), k)
                tot[key] = sp.expand(tot.get(key, 0) + sp.sympify(c, locals=loc) * X)
    lines += [f"## <{kind}{kind}>", "", "| L | invariant | ep^-4 | ep^-3 | ep^-2 | ep^-1 | ep^0 | ep^1 |", "|---|---|---|---|---|---|---|---|"]
    for (L, k) in sorted(tot, key=lambda t: (t[0], t[1])):
        v = tot[(L, k)]
        if v == 0:
            continue
        top = 3 - L                        # highest order needed (and computed)
        cells = []
        for j in range(-4, 2):
            cells.append(str(sp.nsimplify(v.coeff(EP, j))) if j <= top else "")
        if all(c in ("0", "") for c in cells):
            continue
        lines.append(f"| {L} | {k} | " + " | ".join(cells) + " |")
    lines.append("")
open(os.path.join(ROOT, "results", "gscheme_totals.md"), "w").write("\n".join(lines) + "\n")
print("wrote results/gscheme_totals.md")
