#!/usr/bin/env python3
"""Compute G-scheme values of <S Sbar> diagram classes with FORM/Forcer (parallel), cached in
data/supercurrent_classes.json under "<class name>|<projection>" together with the class itself.

    python3 scripts/supercurrent_compute.py --w 4 --proj P              (D-D classes, biconnected: C_S)
    python3 scripts/supercurrent_compute.py --w 4 --proj W1 W2 W3 --ward  (all operator types, scaleless dropped)
    python3 scripts/supercurrent_compute.py --w 4 6 --fast --depth 3 --proj P   (nauty classes; four loops)

Weight w = nY + nW + 2 nQ: w = 2 is two loops (O(eps)), w = 4 three loops (O(eps^2)).
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ctcj.supercurrent import generate_S, generate_S_fast, run_diagram

ROOT = os.path.join(os.path.dirname(__file__), "..")
CACHE = os.path.join(ROOT, "data", "supercurrent_classes.json")
WORK = os.path.join(ROOT, "build", "supercurrent")
OPS = [("D", "D"), ("W", "D"), ("D", "W"), ("W", "W")]


def classes(w, ward, fast=False, jj=False):
    if jj:                                     # <j jbar>: not conserved, so cut-vertex graphs stay
        gen = (lambda *a: generate_S_fast(*a, reps=True)) if fast else generate_S
        return [c for nQ in range(w // 2 + 1) for c in gen(w - 2 * nQ, nQ, ("j", "j"), "not_scaleless")]
    if fast:                                   # nauty representatives (needed from four loops on)
        assert not ward
        return [c for nQ in range(w // 2 + 1) for c in generate_S_fast(w - 2 * nQ, nQ, ("D", "D"), reps=True)]
    out = []
    for ops in (OPS if ward else OPS[:1]):
        nW = ops.count("W")
        for nQ in range(0, (w - nW) // 2 + 1):
            nY = w - nW - 2 * nQ
            if nY >= 0:
                out += generate_S(nY, nQ, ops, "not_scaleless" if ward else "biconnected")
    return out


def load():
    return json.load(open(CACHE)) if os.path.exists(CACHE) else {}



def _job(arg):
    c, proj, depth = arg
    t = time.time()
    x = run_diagram(c, os.path.join(WORK, c["name"]), proj, depth=depth)
    return c, proj, str(x), time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, nargs="+", default=[4])
    ap.add_argument("--proj", nargs="+", default=["P"])
    ap.add_argument("--ward", action="store_true")
    ap.add_argument("-j", type=int, default=4)
    ap.add_argument("--fast", action="store_true", help="nauty generator (four loops)")
    ap.add_argument("--jj", action="store_true", help="<j jbar> classes (use with --proj Pj)")
    ap.add_argument("--depth", type=int, default=5, help="Forcer gschemeexpand (5 through three loops, 3 at four)")
    a = ap.parse_args()
    cache = load()
    jobs = [(c, p, a.depth) for w in a.w for c in classes(w, a.ward, a.fast, a.jj) for p in a.proj
            if f"{c['name']}|{p}" not in cache]
    jobs.sort(key=lambda j: -j[0]["nY"])      # slowest (most fermion lines) first
    print(f"{len(jobs)} integrals to compute", flush=True)
    with Pool(a.j) as pool:
        for c, proj, x, dt in pool.imap_unordered(_job, jobs):
            cache[f"{c['name']}|{proj}"] = dict(X=x, ops=list(c["ops"]), ferm=[list(e) for e in c["ferm"]],
                                                sig=[list(e) for e in c["sig"]], weight=str(c["weight"]),
                                                nY=c["nY"], nQ=c["nQ"], depth=a.depth)
            json.dump(cache, open(CACHE, "w"), indent=1, sort_keys=True)
            print(f"  {c['name']}|{proj}  {dt:.0f}s", flush=True)


if __name__ == "__main__":
    main()
