#!/usr/bin/env python3
"""Flavour decomposition of all four-loop <JJ> classes (data/diagrams4_J.jsonl) on the four-loop J basis,
plus the beta-function insertions needed for renormalisation and the fixed-point form.

Writes data/diagrams4_J.json (records with 'decomp') and data/basis4_J.json (basis names per sector and
the decomposed insertions).  Needs numpy, sympy, scipy, pynauty (~2 min)."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from ctcj.invariantsJ4 import default_ensemble, build_basis, SECTOR_OF, decompose_insertions
from ctcj.decompJ import eval_matrix, decompose
from ctcj.net import class_net

ROOT = os.path.join(os.path.dirname(__file__), "..")
t0 = time.time()
ths = default_ensemble()
basis = build_basis(ths)
print("basis sizes:", {s: len(b["names"]) for s, b in basis.items()}, f"{time.time()-t0:.0f}s", flush=True)

# rank stability on an independent ensemble (spurious relations would change the count)
from ctcj.invariantsJ4 import build_basis as _bb
from ctcj.ensembleJ import ensemble
alt = _bb(ensemble(90, families=("B",), seed=777))
assert all(alt[s]["names"] == basis[s]["names"] for s in basis), "basis not stable across ensembles"
print("basis identical on an independent ensemble", flush=True)

recs = [json.loads(l) for l in open(os.path.join(ROOT, "data", "diagrams4_J.jsonl"))]
dst = os.path.join(ROOT, "data", "diagrams4_J.json")
if os.path.exists(dst) and "--redo" not in sys.argv:          # class decompositions are cached
    recs = json.load(open(dst)); print("class decompositions loaded from cache", flush=True)
for i, r in enumerate(recs):
    if "decomp" in r:
        continue
    if r["X"] == "0":
        r["decomp"] = {}
        continue
    c = dict(ops=tuple(r["ops"]), ferm=tuple(tuple(f) for f in r["ferm"]),
             sig=tuple((tuple(a), tuple(b)) for a, b in r["sig"]))
    b = basis[SECTOR_OF[(r["nY"], r["nQ"])]]
    v = eval_matrix([class_net(c)], ths)[:, 0]
    r["decomp"] = {k: str(val) for k, val in decompose(v, b["names"], b["M"]).items()}
    if i % 100 == 0:
        print(i, f"{time.time()-t0:.0f}s", flush=True)
json.dump(recs, open(os.path.join(ROOT, "data", "diagrams4_J.json"), "w"))

ins = decompose_insertions(ths, basis)
json.dump({"basis": {s: b["names"] for s, b in basis.items()},
           "insertions": {f"{w}|{k}": {n: str(c) for n, c in d.items()} for (w, k), d in ins.items()}},
          open(os.path.join(ROOT, "data", "basis4_J.json"), "w"), indent=1)
print("done", f"{time.time()-t0:.0f}s")
