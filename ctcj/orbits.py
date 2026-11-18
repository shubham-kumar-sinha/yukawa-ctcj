"""Orbits of <TT> diagram classes under exact symmetries of the TT form factor:
x <-> 0 exchange and reversal of any fermion loop (T insertions and Yukawa vertices are C-even, and
traces of symmetric y^a are reversal invariant).  Members of an orbit have identical X and flavour
invariants, so one Forcer run per orbit suffices (validated against all three-loop classes)."""
import itertools

from pynauty import Graph, certificate

from .diagrams import loops_of


def _cert(ops, ferm, sig, nY, nQ):
    verts = ["x", "0"] + [f"Y{i}" for i in range(nY)] + [f"Q{i}" for i in range(nQ)]
    vid = {v: i for i, v in enumerate(verts)}
    nv, nF, nS = len(verts), len(ferm), len(sig)
    nT0, nH0, nS0 = nv, nv + nF, nv + 2 * nF
    ntot = nS0 + nS
    adj = {i: [] for i in range(ntot)}
    for k, (u, v) in enumerate(ferm):
        adj[vid[u]].append(nT0 + k)
        adj[nT0 + k].append(nH0 + k)
        adj[nH0 + k].append(vid[v])
    for k, (a, b) in enumerate(sig):
        adj[nS0 + k] += [vid[a[0]], vid[b[0]]]
    col = [{vid["x"]}, {vid["0"]}, set(range(2, 2 + nY)), set(range(2 + nY, nv)),
           set(range(nT0, nH0)), set(range(nH0, nS0)), set(range(nS0, ntot))]
    col = [c for c in col if c]
    return (tuple(ops), certificate(Graph(ntot, directed=False, adjacency_dict=adj, vertex_coloring=col)))


def orbit_key(c):
    loops = loops_of(list(c["ferm"]))
    keys = []
    for swap in (False, True):
        ren = (lambda v: {"x": "0", "0": "x"}.get(v, v)) if swap else (lambda v: v)
        ops = tuple(reversed(c["ops"])) if swap else tuple(c["ops"])
        for flips in itertools.product((False, True), repeat=len(loops)):
            rev = set()
            for L, fl in zip(loops, flips):
                if fl:
                    rev |= set(L)
            ferm = [((b, a) if a in rev else (a, b)) for a, b in c["ferm"]]
            ferm = [(ren(a), ren(b)) for a, b in ferm]
            sig = [((ren(a[0]), a[1]), (ren(b[0]), b[1])) for a, b in c["sig"]]
            keys.append(_cert(ops, ferm, sig, c["nY"], c["nQ"]))
    return min(keys)


# ----------------------------------------------------------------------------- <JJ>
# For J (C-odd), reversing a fermion loop multiplies the integral by (-1)^(number of Jpsi on that loop);
# the flavour factor changes and is evaluated per class.  x <-> 0 exchange is a symmetry of the form factor.
def _cert_J(ops, ferm, sig, nY, nQ):
    verts = ["x", "0"] + [f"Y{i}" for i in range(nY)] + [f"Q{i}" for i in range(nQ)]
    vid = {v: i for i, v in enumerate(verts)}; nv, nF, nS = len(verts), len(ferm), len(sig)
    T0, H0, S0, M0 = nv, nv + nF, nv + 2 * nF, nv + 2 * nF + nS
    adj = {i: [] for i in range(M0 + 4)}
    for k, (u, v) in enumerate(ferm):
        adj[vid[u]].append(T0 + k); adj[T0 + k].append(H0 + k); adj[H0 + k].append(vid[v])
    mk = {("x", 0): M0, ("0", 0): M0 + 1, ("x", 1): M0 + 2, ("0", 1): M0 + 3}
    for v in ("x", "0"):
        adj[vid[v]] += [mk[(v, 0)], mk[(v, 1)]]
    for k, (a, b) in enumerate(sig):
        for e in (a, b):
            adj[S0 + k].append(mk[(e[0], e[1])] if e[1] >= 0 else vid[e[0]])
    col = [{0}, {1}, set(range(2, 2 + nY)), set(range(2 + nY, nv)), set(range(T0, H0)), set(range(H0, S0)),
           set(range(S0, M0)), {M0, M0 + 1}, {M0 + 2, M0 + 3}]
    return (tuple(ops), certificate(Graph(M0 + 4, directed=False, adjacency_dict=adj, vertex_coloring=[c for c in col if c])))

def _variants_J(c):
    loops = loops_of(list(c["ferm"])); out = []
    for swap in (False, True):
        ren = (lambda v: {"x": "0", "0": "x"}.get(v, v)) if swap else (lambda v: v)
        ops = tuple(reversed(c["ops"])) if swap else tuple(c["ops"])
        for flips in itertools.product((False, True), repeat=len(loops)):
            rev, sign = set(), 1
            for L, fl in zip(loops, flips):
                if fl:
                    rev |= set(L); sign *= (-1) ** sum(1 for v in L if v in ("x", "0"))
            ferm = [(ren(b), ren(a)) if a in rev else (ren(a), ren(b)) for a, b in c["ferm"]]
            sig = [((ren(a[0]), a[1]), (ren(b[0]), b[1])) for a, b in c["sig"]]
            out.append((_cert_J(ops, ferm, sig, c["nY"], c["nQ"]), sign))
    return out



def orbit_key_J(c):
    """(orbit key, sign) with X(c) = sign * X(canonical orientation)."""
    return min(_variants_J(c), key=lambda v: v[0])


def selftest_J(data_path):
    """Check the sign rule on the stored three-loop <JJ> data."""
    import json
    import sympy as sp
    from .diagrams import generate
    ep = sp.Symbol("ep")
    L = {"ep": ep, "z3": sp.Symbol("z3"), "z4": sp.Symbol("z4"), "z5": sp.Symbol("z5"), "I": sp.I}
    X = {r["name"]: sp.sympify(r["X"], locals=L) for r in json.load(open(data_path))}
    for o in [(2, 0), (4, 0), (0, 2)]:
        key = {}
        for c in generate("J", *o):
            k, s = orbit_key_J(c)
            key.setdefault(k, []).append((c["name"], s))
        for m in key.values():
            for n, s in m:
                assert sp.expand(X[n] * s - X[m[0][0]] * m[0][1]) == 0, n
    return True
