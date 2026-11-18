#!/usr/bin/env python3
"""Four-loop classes (nauty generator) -> FORM/Forcer, appended to data/diagrams4_{kind}.jsonl (resumable).
Flavour decomposition is done afterwards by scripts/decompose4.py.
    python3 scripts/compute_diagrams4.py T|J [depth]"""
import json, os, sys, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ctcj.diagrams_fast import generate_fast
from ctcj.forcer import run_class, EP, Z3, Z4, Z5
import sympy as sp
LOC = {"ep": EP, "z3": Z3, "z4": Z4, "z5": Z5, "I": sp.I}
ROOT = os.path.join(os.path.dirname(__file__), "..")
kind = sys.argv[1]; depth = int(sys.argv[2]) if len(sys.argv) > 2 else 3
jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 1      # parallel Forcer processes
cache = os.path.join(ROOT, "build", f"classes4_{kind}.pkl")
if os.path.exists(cache):
    cl = pickle.load(open(cache, "rb"))
else:
    cl = {o: generate_fast(kind, *o) for o in [(0, 3), (2, 2), (4, 1), (6, 0)]}
    pickle.dump(cl, open(cache, "wb"))
out = os.path.join(ROOT, "data", f"diagrams4_{kind}.jsonl")
done = {}
if os.path.exists(out):
    done = {json.loads(l)["name"]: json.loads(l) for l in open(out)}
from ctcj.orbits import orbit_key, orbit_key_J, selftest_J
reps, sgn = {}, {}
if kind == "J":                       # one Forcer run per orbit, X(member) = sign * X(rep)
    assert selftest_J(os.path.join(ROOT, "data", "diagrams_J.json")), "orbit sign rule failed"
    print("J orbit sign rule verified on three-loop data", flush=True)
    orb = {}
    for cs in cl.values():
        for c in cs:
            k, s = orbit_key_J(c)
            orb.setdefault(k, []).append((c, s))
    for members in orb.values():
        rep, srep = next(((m, s) for m, s in members if m["name"] in done), members[0])
        for m, s in members:
            reps[m["name"]] = rep["name"]; sgn[m["name"]] = s * srep
if kind == "T":                       # one Forcer run per symmetry orbit (see ctcj/orbits.py)
    orb = {}
    for cs in cl.values():
        for c in cs:
            orb.setdefault(orbit_key(c), []).append(c)
    for members in orb.values():
        rep = next((m for m in members if m["name"] in done), members[0])
        for m in members:
            reps[m["name"]] = rep["name"]
n = sum(len(v) for v in cl.values())
print(f"{kind}: {n} classes, {len(done)} done", flush=True)
with open(out, "a") as fh:
    order = [c for cs in cl.values() for c in cs]
    order.sort(key=lambda c: reps.get(c["name"], c["name"]) != c["name"])   # representatives first
    if jobs > 1:                     # compute all missing representatives in parallel first
        from concurrent.futures import ProcessPoolExecutor
        todo = [c for c in order if c["name"] not in done and reps.get(c["name"], c["name"]) == c["name"]]
        wd = lambda c: os.path.join(ROOT, "build", "form4", c["name"])
        with ProcessPoolExecutor(jobs) as ex:
            futs = {c["name"]: (c, ex.submit(run_class, c, wd(c), depth)) for c in todo}
            for name, (c, fu) in futs.items():
                X = fu.result()
                rec = dict(name=c["name"], kind=kind, ops=list(c["ops"]), ferm=[list(f) for f in c["ferm"]],
                           sig=[[list(a), list(b)] for a, b in c["sig"]], weight=str(c["weight"]),
                           nY=c["nY"], nQ=c["nQ"], X=str(X), depth=depth)
                fh.write(json.dumps(rec) + "\n"); fh.flush(); done[c["name"]] = rec
                print(c["name"], flush=True)
    for c in order:
        if True:
            if c["name"] in done:
                continue
            t0 = time.time()
            r = reps.get(c["name"], c["name"])
            if r != c["name"] and r in done:
                X = done[r]["X"] if kind == "T" else str(sp.expand(sgn[c["name"]] * sp.sympify(done[r]["X"], locals=LOC)))
            else:
                X = run_class(c, os.path.join(ROOT, "build", "form4"), depth=depth)
            rec = dict(name=c["name"], kind=kind, ops=list(c["ops"]), ferm=[list(f) for f in c["ferm"]],
                       sig=[[list(a), list(b)] for a, b in c["sig"]], weight=str(c["weight"]),
                       nY=c["nY"], nQ=c["nQ"], X=str(X), depth=depth)
            fh.write(json.dumps(rec) + "\n"); fh.flush(); done[c["name"]] = rec
            print(c["name"], f"{time.time()-t0:.1f}s", flush=True)
