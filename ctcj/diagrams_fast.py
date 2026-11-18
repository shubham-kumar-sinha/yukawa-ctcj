"""Fast Wick-contraction generator (same output as ctcj.diagrams.generate) using nauty canonical
labelling.  Needed from four loops on, where relabelling by brute force is too slow.

Each labelled contraction becomes a vertex-coloured simple graph: operator vertices x, 0 (colour =
position + operator type), Yukawa and quartic vertices; every fermion propagator u->v becomes a path
u - t - h - v (tail/head colours), every scalar propagator a node s joined to its ends, with an extra
marker node for the two distinguishable legs of Jphi.  Isomorphism classes = nauty certificates.
"""
import itertools
import math
from fractions import Fraction

from pynauty import Graph, certificate

from .diagrams import FERMIONIC, SCALAR2, OPTYPES, perfect_matchings


def _bicon_mask(nv, edges):
    adj = [0] * nv
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    full = (1 << nv) - 1

    def connected(mask):
        if mask == 0:
            return True
        start = mask & -mask
        seen = start
        frontier = start
        while frontier:
            nxt = 0
            m = frontier
            while m:
                b = m & -m
                nxt |= adj[b.bit_length() - 1]
                m ^= b
            nxt &= mask & ~seen
            seen |= nxt
            frontier = nxt
        return seen == mask
    if not connected(full):
        return False
    if nv <= 2:
        return True
    return all(connected(full & ~(1 << v)) for v in range(nv))


def generate_fast(kind, nY, nQ, progress=False):
    Y = [f"Y{i}" for i in range(nY)]
    Qv = [f"Q{i}" for i in range(nQ)]
    verts = ["x", "0"] + Y + Qv
    vid = {v: i for i, v in enumerate(verts)}
    nv = len(verts)
    base_w = Fraction(1, math.factorial(nY) * math.factorial(nQ) * 24 ** nQ)
    classes = {}
    for ops in itertools.product(OPTYPES[kind], repeat=2):
        opmap = {"x": ops[0], "0": ops[1]}
        fv = [v for v in ("x", "0") if opmap[v] in FERMIONIC] + Y
        legs = [(v, k) for v in ("x", "0") if opmap[v] in SCALAR2 for k in range(2)]
        legs += [(y, 0) for y in Y] + [(q, k) for q in Qv for k in range(4)]
        if len(legs) % 2:
            continue
        matchings = [m for m in perfect_matchings(legs) if all(a[0] != b[0] for a, b in m)]
        nF = len(fv)
        nS = len(legs) // 2
        jphi = [v for v in ("x", "0") if opmap[v] == "Jphi"]
        # node layout: verts | fermion tails | fermion heads | scalar nodes | jphi markers (leg0, leg1)
        nT0 = nv
        nH0 = nT0 + nF
        nS0 = nH0 + nF
        nM0 = nS0 + nS
        ntot = nM0 + 2 * len(jphi)
        coloring = [{vid["x"]}, {vid["0"]}, set(range(2, 2 + nY)), set(range(2 + nY, nv)),
                    set(range(nT0, nH0)), set(range(nH0, nS0)), set(range(nS0, nM0)),
                    set(range(nM0, nM0 + len(jphi))), set(range(nM0 + len(jphi), ntot))]
        coloring = [c for c in coloring if c]
        for perm in itertools.permutations(fv):
            ferm = list(zip(fv, perm))
            if any(a == b for a, b in ferm):
                continue
            fe = [(vid[a], vid[b]) for a, b in ferm]
            for m in matchings:
                se = [(vid[a[0]], vid[b[0]]) for a, b in m]
                if not _bicon_mask(nv, fe + se):
                    continue
                adj = {i: [] for i in range(ntot)}
                def link(i, j):
                    adj[i].append(j)
                for k, (u, v) in enumerate(fe):
                    t, h = nT0 + k, nH0 + k
                    link(u, t); link(t, h); link(h, v)
                mk = {}
                for k, v in enumerate(jphi):
                    mk[(v, 0)] = nM0 + k
                    mk[(v, 1)] = nM0 + len(jphi) + k
                    link(vid[v], mk[(v, 0)]); link(vid[v], mk[(v, 1)])
                for k, (a, b) in enumerate(m):
                    s = nS0 + k
                    for end in (a, b):
                        link(s, mk[end] if end in mk else vid[end[0]])
                cert = certificate(Graph(ntot, directed=False, adjacency_dict=adj, vertex_coloring=coloring))
                key = (ops, cert)
                c = classes.get(key)
                if c is None:
                    tag = lambda l: (l[0], l[1]) if opmap.get(l[0]) == "Jphi" else (l[0], -1)
                    c = classes[key] = dict(ops=ops, ferm=tuple(ferm),
                                            sig=tuple((tag(a), tag(b)) for a, b in m),
                                            weight=Fraction(0), nY=nY, nQ=nQ, kind=kind)
                c["weight"] += base_w
        if progress:
            print(kind, nY, nQ, ops, len(classes), flush=True)
    out = [classes[k] for k in sorted(classes, key=lambda k: (k[0], k[1]))]
    for i, c in enumerate(out):
        c["name"] = f"{kind}{nY}{nQ}_{i:03d}"
    return out
