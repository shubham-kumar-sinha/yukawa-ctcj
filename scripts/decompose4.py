#!/usr/bin/env python3
"""Decompose the flavour factor of every four-loop class in data/diagrams4_T.jsonl on the four-loop basis.
Writes data/diagrams4_T.json (records with 'decomp').  Resumable."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from ctcj.invariants import class_invariant
from ctcj.invariants4 import T_basis4, theories, independent, decompose_vals
ROOT = os.path.join(os.path.dirname(__file__), "..")
src, dst = os.path.join(ROOT, "data", "diagrams4_T.jsonl"), os.path.join(ROOT, "data", "diagrams4_T.json")
done = {r["name"]: r for r in json.load(open(dst))} if os.path.exists(dst) else {}
ths = theories(60)
names, M = independent(T_basis4(), ths)
recs = []
for line in open(src):
    r = json.loads(line)
    if r["name"] in done:
        recs.append(done[r["name"]]); continue
    c = dict(ops=tuple(r["ops"]), ferm=tuple(tuple(f) for f in r["ferm"]),
             sig=tuple((tuple(a), tuple(b)) for a, b in r["sig"]), nY=r["nY"], nQ=r["nQ"], name=r["name"])
    vals = np.array([class_invariant(c, th).real for th in ths])
    r["decomp"] = {k: str(v) for k, v in decompose_vals(vals, names, M).items()} if r["X"] != "0" else {}
    recs.append(r)
    if len(recs) % 20 == 0:
        json.dump(recs, open(dst, "w")); print(len(recs), flush=True)
json.dump(recs, open(dst, "w"))
print("decomposed", len(recs))
