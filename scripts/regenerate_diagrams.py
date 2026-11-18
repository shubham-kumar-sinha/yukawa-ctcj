#!/usr/bin/env python3
"""Regenerate every diagram class from scratch and compare with the published data (no FORM needed).

* 1-3 loops: brute-force Wick contraction (ctcj.diagrams.generate), every labelled contraction enumerated.
* 4 loops: nauty-based generator (ctcj.diagrams_fast.generate_fast); at 1-3 loops it is cross-checked
  against the brute-force generator.
For every class the graph (operators, fermion arrows, scalar lines) and the weight = 1/symmetry factor must
equal the published record in data/.  Exit code 0 if everything agrees."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fractions import Fraction
from ctcj.diagrams import generate, _canonical
from ctcj.diagrams_fast import generate_fast

ROOT = os.path.join(os.path.dirname(__file__), "..")
PUB = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data")      # directory with the published data
key = lambda c: (tuple(c["ops"]), tuple(map(tuple, c["ferm"])), tuple((tuple(a), tuple(b)) for a, b in c["sig"]))
ok = True
for kind in ("T", "J"):
    pub = {}
    recs = json.load(open(os.path.join(PUB, f"diagrams_{kind}.json")))
    recs += [json.loads(l) for l in open(os.path.join(PUB, f"diagrams4_{kind}.jsonl"))]
    for r in recs:
        pub[key(r)] = (r["name"], Fraction(r["weight"]))
    orders = [(0, 0), (2, 0), (4, 0), (0, 2), (6, 0), (4, 1), (2, 2), (0, 3)]
    for o in orders:
        t0 = time.time()
        if o in [(6, 0), (4, 1), (2, 2), (0, 3)]:
            gen = generate_fast(kind, *o)
            how = "nauty"
        else:
            gen = generate(kind, *o)
            fast = generate_fast(kind, *o)
            canon = lambda c: _canonical(dict(zip(("x", "0"), c["ops"])), [f"Y{i}" for i in range(o[0])],
                                         [f"Q{i}" for i in range(o[1])], [tuple(e) for e in c["ferm"]],
                                         [(tuple(a), tuple(b)) for a, b in c["sig"]])
            same = {canon(c): Fraction(c["weight"]) for c in gen} == {canon(c): Fraction(c["weight"]) for c in fast}
            ok &= same
            how = "brute force; nauty generator agrees" if same else "brute force; nauty generator DIFFERS"
        bad = [c["name"] for c in gen if key(c) not in pub or pub[key(c)][1] != Fraction(c["weight"])]
        npub = sum(1 for k, (n, w) in pub.items() if n.startswith(f"{kind}{o[0]}{o[1]}_"))
        ok &= not bad and len(gen) == npub
        print(f"<{kind}{kind}> (nY,nQ)={o}: {len(gen):4d} classes ({how}, {time.time()-t0:.0f}s); published {npub}; "
              f"{'all graphs and weights identical' if not bad and len(gen) == npub else f'MISMATCH {bad[:5]}'}", flush=True)
print("PASS: every diagram class and weight reproduced" if ok else "FAIL")
sys.exit(0 if ok else 1)
