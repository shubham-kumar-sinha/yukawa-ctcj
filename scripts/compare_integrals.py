#!/usr/bin/env python3
"""Compare the Feynman integrals X of two copies of the repository, class by class.

    python3 scripts/compare_integrals.py PUBLISHED_DATA_DIR RECOMPUTED_DATA_DIR

Each directory is a data/ folder.  Classes are matched by name, and their diagrams (fermion and scalar lines)
must be identical, so a renumbering cannot hide a difference.  Only the orders present in both copies are compared
(the published four-loop integrals stop at ep^-1).  Exit code 0 if every common class agrees exactly."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sympy as sp
from ctcj.forcer import EP

L = {"ep": EP, "z3": sp.Symbol("z3"), "z4": sp.Symbol("z4"), "z5": sp.Symbol("z5"), "I": sp.I}
FILES = ["diagrams_T.json", "diagrams_J.json", "diagrams4_T.jsonl", "diagrams4_J.jsonl"]


def load(d, fn):
    p = os.path.join(d, fn)
    if not os.path.exists(p):
        return None
    if fn.endswith(".jsonl"):
        return {r["name"]: r for r in (json.loads(l) for l in open(p))}
    return {r["name"]: r for r in json.load(open(p))}


def orders(x):
    x = sp.expand(sp.sympify(x, locals=L))
    if x == 0:
        return {}
    return {j: x.coeff(EP, j) for j in range(-6, 8) if x.coeff(EP, j) != 0}


old_dir, new_dir = sys.argv[1], sys.argv[2]
bad = 0
for fn in FILES:
    a, b = load(old_dir, fn), load(new_dir, fn)
    if a is None or b is None:
        print(f"{fn}: not present in both directories, skipped")
        continue
    common = sorted(set(a) & set(b))
    nbad = 0
    for n in common:
        ra, rb = a[n], b[n]
        if (ra["ferm"], ra["sig"], ra["ops"]) != (rb["ferm"], rb["sig"], rb["ops"]):
            print(f"  {n}: different diagram under the same name"); nbad += 1; continue
        oa, ob = orders(ra["X"]), orders(rb["X"])
        top = min(max(oa, default=-99), max(ob, default=-99))
        for j in set(oa) | set(ob):
            if j <= top and sp.simplify(oa.get(j, 0) - ob.get(j, 0)) != 0:
                print(f"  {n}: ep^{j} differs: {oa.get(j, 0)} vs {ob.get(j, 0)}"); nbad += 1; break
    print(f"{fn}: {len(common)} common classes ({len(a)} published, {len(b)} recomputed), {nbad} differences")
    bad += nbad
print("PASS: all common classes agree" if bad == 0 else f"FAIL: {bad} differences")
sys.exit(0 if bad == 0 else 1)
