#!/usr/bin/env python3
"""Generate all <TT> and <JJ> diagram classes through three loops, reduce each with FORM+Forcer,
decompose its flavour factor on the invariant basis, and write data/diagrams_{T,J}.json.
Resumable: classes already present in the output file are skipped.

    FORM_BIN=/path/to/form FORCER_PATH=/path/to/forcer python3 scripts/compute_diagrams.py [T|J]
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sympy as sp
from ctcj.diagrams import generate, ORDERS
from ctcj.forcer import run_class
from ctcj.invariants import (T_BASIS, J_BASIS, random_generic, symmetric_ensemble, decompose, order_of)

ROOT = os.path.join(os.path.dirname(__file__), "..")
kinds = sys.argv[1:] or ["T", "J"]
for kind in kinds:
    out = os.path.join(ROOT, "data", f"diagrams_{kind}.json")
    done = {r["name"]: r for r in json.load(open(out))} if os.path.exists(out) else {}
    ths = [random_generic(2 + i % 2, 3 + (i // 2) % 2) for i in range(8)] if kind == "T" else symmetric_ensemble(7)
    basis = T_BASIS if kind == "T" else J_BASIS
    recs = []
    for nY, nQ in ORDERS:
        for c in generate(kind, nY, nQ):
            if c["name"] in done:
                recs.append(done[c["name"]]); continue
            t0 = time.time()
            X = run_class(c, os.path.join(ROOT, "build", "form"), depth=5)
            dec = decompose(c, ths, basis[order_of(c)]) if X != 0 else {}
            rec = dict(name=c["name"], kind=kind, ops=list(c["ops"]), ferm=[list(f) for f in c["ferm"]],
                       sig=[[list(a), list(b)] for a, b in c["sig"]], weight=str(c["weight"]), nY=nY, nQ=nQ,
                       X=str(X), decomp={k: str(v) for k, v in dec.items()})
            recs.append(rec); done[c["name"]] = rec
            json.dump(recs, open(out, "w"), indent=1)
            print(f"{c['name']:12s} {str(c['ops']):22s} {time.time()-t0:5.1f}s  inv={rec['decomp']}", flush=True)
    json.dump(recs, open(out, "w"), indent=1)
    print(f"wrote {out}: {len(recs)} classes")
