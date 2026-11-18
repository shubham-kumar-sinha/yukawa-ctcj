#!/usr/bin/env python3
"""Ward-identity relations among the four-loop <JJ> candidate invariants.

Every candidate of ctcj.basisJ4 that is not in the basis (data/basis4_J.json) is expressed exactly on the
basis, using U(1)-symmetric theories (ctcj.ensembleJ); candidates that vanish identically are listed with an
empty expansion.  Each relation is then re-verified on an independent ensemble with different field content.
Writes data/relations4_J.json.  (~2 min)"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from ctcj.basisJ4 import candidates
from ctcj.decompJ import eval_matrix, decompose
from ctcj.ensembleJ import ensemble

ROOT = os.path.join(os.path.dirname(__file__), "..")
basis = json.load(open(os.path.join(ROOT, "data", "basis4_J.json")))["basis"]
ths = ensemble(110, families=("A", "C"), seed=4242)
chk = ensemble(40, families=("B",), seed=99)
out, nrel = {}, 0
for sec, names in basis.items():
    cand = candidates(sec)
    Mb = eval_matrix([cand[k] for k in names], ths)
    Mc = eval_matrix([cand[k] for k in names], chk)
    scale = np.median(np.linalg.norm(Mb, axis=0))
    rel = {}
    for k, net in cand.items():
        if k in names:
            continue
        v = eval_matrix([net], ths)[:, 0]
        d = {} if np.linalg.norm(v) < 1e-8 * scale else decompose(v, names, Mb)
        # independent verification
        vc = eval_matrix([net], chk)[:, 0]
        pred = Mc @ np.array([complex(d.get(n, 0)) for n in names])
        assert np.linalg.norm(vc - pred) <= 1e-8 * max(np.linalg.norm(vc), 1e-8 * scale), (sec, k)
        rel[k] = {n: str(c) for n, c in d.items()}
        nrel += 1
    out[sec] = rel
    print(sec, f"{len(names)} basis, {len(rel)} relations "
          f"({sum(1 for r in rel.values() if not r)} identically zero)", flush=True)
json.dump(out, open(os.path.join(ROOT, "data", "relations4_J.json"), "w"), indent=1)
print("wrote data/relations4_J.json:", nrel, "relations, all re-verified on an independent ensemble")
