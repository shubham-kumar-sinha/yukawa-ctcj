"""Tensor-network representation of flavour invariants, evaluated with numpy.einsum.

A network is a tuple of factors (name, idx) with name in {'y','T','t','lam'} plus the number of fermion
traces.  Index ids are integers; 'y' factors are (scalar, row, col), 'T' (row, col), 't' (s0, s1),
'lam' (s0, s1, s2, s3).  Repeated ids are summed.  The value is mult**ntr * contraction, where
mult (theory key 'mult', default 1) emulates tr 1 / replicated fermions.

Conventions match ctcj.invariants.class_invariant: along a fermion loop the matrices are multiplied
against the arrows, i.e. the matrix at vertex v carries (outgoing edge, incoming edge) as (row, col).
"""
from collections import defaultdict

import numpy as np

from .diagrams import loops_of

_PATHS = {}


def evaluate(net, th, subs=None):
    """net = (factors, ntr).  subs: {factor position: tensor} replaces individual factors (insertions)."""
    factors, ntr = net
    ops = []
    for k, (name, idx) in enumerate(factors):
        A = subs[k] if subs and k in subs else th[name]
        ops.append(A)
        ops.append(list(idx))
    ops.append([])
    key = (net, tuple(np.shape(o) for o in ops[0:-1:2]))
    path = _PATHS.get(key)
    if path is None:
        path = np.einsum_path(*ops, optimize=("greedy", 2e6))[0]
        _PATHS[key] = path
    val = np.einsum(*ops, optimize=path)
    return complex(val) * th.get("mult", 1.0) ** ntr


def class_net(c):
    """Network of a diagram class (dict with ops, ferm, sig)."""
    ops = dict(zip(("x", "0"), c["ops"]))
    ferm, sig = list(c["ferm"]), list(c["sig"])
    nid = iter(range(10 ** 6))
    fidx = {(a, b): next(nid) for a, b in ferm}
    out_e = {a: fidx[(a, b)] for a, b in ferm}
    in_e = {b: fidx[(a, b)] for a, b in ferm}
    leg = defaultdict(dict)
    for (a, b) in sig:
        s = next(nid)
        for (v, tg) in (a, b):
            d = leg[v]
            d[tg if tg >= 0 else len(d)] = s
    facs = []
    fverts = set(out_e)
    for v in sorted(set(leg) | fverts):
        if v.startswith("Y"):
            facs.append(("y", (leg[v][0], out_e[v], in_e[v])))
        elif v.startswith("Q"):
            facs.append(("lam", tuple(leg[v][k] for k in range(4))))
        elif ops.get(v) == "Jpsi":
            facs.append(("T", (out_e[v], in_e[v])))
        elif ops.get(v) == "Jphi":
            facs.append(("t", (leg[v][0], leg[v][1])))
        elif ops.get(v) == "Tsig":
            s0, s1 = leg[v][0], leg[v][1]
            facs.append(("delta", (s0, s1)))
        elif ops.get(v) == "Tpsi":
            facs.append(("onef", (out_e[v], in_e[v])))
    return (tuple(facs), len(loops_of(ferm)) if ferm else 0)


def words_net(words, lam_legs=None, extra=()):
    """Product of traces.  words: list of strings over letters 'T' and scalar labels (any other char).
    Repeated labels are contracted.  extra: additional factors (name, labels) with labels given as chars,
    e.g. ('lam', 'abcd') or ('t', 'ae')."""
    ids = {}
    nid = iter(range(10 ** 6))

    def sid(ch):
        if ch not in ids:
            ids[ch] = next(nid)
        return ids[ch]
    facs = []
    for w in words:
        n = len(w)
        f = [next(nid) for _ in range(n)]           # f[k]: fermion index between letter k-1 and k
        for k, ch in enumerate(w):
            row, col = f[k], f[(k + 1) % n]
            if ch == "T":
                facs.append(("T", (row, col)))
            else:
                facs.append(("y", (sid(ch), row, col)))
    for name, labs in extra:
        facs.append((name, tuple(sid(ch) for ch in labs)))
    return (tuple(facs), len(words))


def theory_tensors(th):
    """Add helper tensors used by class nets (delta, onef)."""
    th = dict(th)
    ns, nf = th["y"].shape[0], th["y"].shape[1]
    th.setdefault("delta", np.eye(ns))
    th.setdefault("onef", np.eye(nf))
    return th
