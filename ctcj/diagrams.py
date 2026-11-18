"""Wick-contraction generator for <O(x) O(0)>, O = T (stress tensor) or J (conserved current),
in the general scalar-fermion theory

    L = 1/2 (d phi_a)^2 + psibar dslash psi + phi_a psibar y^a psi + lambda_abcd/24 phi_a phi_b phi_c phi_d .

Operators (null polarisation z with z.Q = 0):
    T = Tpsi + Tsig,  Tpsi = 1/2[psibar zslash (z.d psi) - (z.d psibar) zslash psi],  Tsig = (z.d phi_a)^2
    J = Jpsi + Jphi,  Jpsi = psibar zslash T psi,  Jphi = -i (z.d phi_a) t_ab phi_b
(J is the Noether current of  delta psi = i T psi,  delta phi = t phi,  divided by i.)
Interaction and improvement terms of T, and all delta_{mu nu} pieces, drop out under the
(transverse-)traceless projection.

Every labelled Wick contraction is generated; classes are formed by canonical relabelling of
the internal vertices.  Zero rules: self-loops, disconnected graphs and graphs with a cut
vertex (scaleless, or killed by the projection) are dropped.
"""
import itertools
import math
from collections import defaultdict
from fractions import Fraction

FERMIONIC = {"Tpsi", "Jpsi"}
SCALAR2 = {"Tsig", "Jphi"}
OPTYPES = {"T": ("Tpsi", "Tsig"), "J": ("Jpsi", "Jphi")}
ORDERS = [(0, 0), (2, 0), (0, 1), (4, 0), (2, 1), (0, 2)]   # (nY, nQ) through three loops


def perfect_matchings(items):
    if not items:
        yield []
        return
    a = items[0]
    for i in range(1, len(items)):
        rest = items[1:i] + items[i + 1:]
        for m in perfect_matchings(rest):
            yield [(a, items[i])] + m


def _connected(vs, adj):
    vs = set(vs)
    if not vs:
        return True
    start = next(iter(vs))
    seen, stack = {start}, [start]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w in vs and w not in seen:
                seen.add(w)
                stack.append(w)
    return seen == vs


def biconnected(vertices, edges):
    """Connected and free of cut vertices."""
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    if not _connected(vertices, adj):
        return False
    if len(vertices) <= 2:
        return True
    return all(_connected([w for w in vertices if w != v], adj) for v in vertices)


def loops_of(ferm):
    """Fermion cycles, each listed along the arrows."""
    nxt = dict(ferm)
    seen, loops = set(), []
    for v in nxt:
        if v in seen:
            continue
        cyc = [v]
        seen.add(v)
        w = nxt[v]
        while w != v:
            cyc.append(w)
            seen.add(w)
            w = nxt[w]
        loops.append(cyc)
    return loops


def _canonical(opmap, Y, Qv, ferm, sig):
    best = None
    for py in itertools.permutations(range(len(Y))):
        for pq in itertools.permutations(range(len(Qv))):
            ren = {"x": "x", "0": "0"}
            ren.update({Y[i]: f"Y{py[i]}" for i in range(len(Y))})
            ren.update({Qv[i]: f"Q{pq[i]}" for i in range(len(Qv))})
            f = tuple(sorted((ren[a], ren[b]) for a, b in ferm))
            s = tuple(sorted(tuple(sorted(((ren[a[0]], a[1]), (ren[b[0]], b[1])))) for a, b in sig))
            k = (opmap["x"], opmap["0"], f, s)
            if best is None or k < best:
                best = k
    return best


def generate(kind, nY, nQ):
    """Classes of <O O> diagrams (O = 'T' or 'J') with nY Yukawa and nQ quartic vertices.

    Each class: dict(ops, ferm, sig, weight, nY, nQ, kind, name)
      ferm : fermion propagators as arrows (a, b), a -> b
      sig  : scalar propagators, endpoints (vertex, legtag); legtag 0/1 for the two legs of Jphi
             (0 = derivative leg), -1 otherwise
      weight: sum over labelled contractions of 1/(nY! nQ! 24^nQ)
    """
    classes = {}
    Y = [f"Y{i}" for i in range(nY)]
    Qv = [f"Q{i}" for i in range(nQ)]
    base_w = Fraction(1, math.factorial(nY) * math.factorial(nQ) * 24 ** nQ)
    for ops in itertools.product(OPTYPES[kind], repeat=2):
        opmap = {"x": ops[0], "0": ops[1]}
        fv = [v for v in ("x", "0") if opmap[v] in FERMIONIC] + Y
        legs = [(v, k) for v in ("x", "0") if opmap[v] in SCALAR2 for k in range(2)]
        legs += [(y, 0) for y in Y] + [(q, k) for q in Qv for k in range(4)]
        if len(legs) % 2:
            continue
        verts = ["x", "0"] + Y + Qv

        def tag(leg):
            return (leg[0], leg[1]) if opmap.get(leg[0]) == "Jphi" else (leg[0], -1)

        for perm in itertools.permutations(fv):
            ferm = list(zip(fv, perm))
            if any(a == b for a, b in ferm):
                continue
            for m in perfect_matchings(legs):
                if any(a[0] == b[0] for a, b in m):
                    continue
                if not biconnected(verts, ferm + [(a[0], b[0]) for a, b in m]):
                    continue
                sig = [(tag(a), tag(b)) for a, b in m]
                key = _canonical(opmap, Y, Qv, ferm, sig)
                c = classes.get(key)
                if c is None:
                    c = classes[key] = dict(ops=ops, ferm=key[2], sig=key[3], weight=Fraction(0),
                                            nY=nY, nQ=nQ, kind=kind)
                c["weight"] += base_w
    out = [classes[k] for k in sorted(classes)]
    for i, c in enumerate(out):
        c["name"] = f"{kind}{nY}{nQ}_{i:02d}"
    return out


if __name__ == "__main__":
    for kind in ("T", "J"):
        print(kind, {o: len(generate(kind, *o)) for o in ORDERS})
