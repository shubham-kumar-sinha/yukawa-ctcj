"""C_J at O(eps^3): four-loop <JJ> diagrams + two-loop renormalisation of the tensor couplings.

Same scheme as ctcj.assemble4 (C_T):
    y0 = y + beta1/eps + beta2/(2 eps) + D2/eps^2,  D2 = 1/2 Dbeta1[beta1];   lam0 = lam + betalam1/eps,
with eps = 4 - d = 2 ep.  Bare words expanded:
    W2(y0)   = W2 + DW2[b1]/(2ep) + DW2[b2]/(4ep) + (1/2 D^2W2[b1,b1] + DW2[D2])/(4ep^2)
    I4(y0)   = I4 + DI4[b1]/(2ep)
    ttll(l0) = ttll + D ttll[bl]/(2ep)
Fixed-point form: eps*I = DI[(beta_y, beta_lam)]/w(I), w = (y-degree)/2 + (lam-degree).
Two-loop words: W2a = Tr(T T y^a y^a), W2b = Tr(T y^a T y^a); three-loop words: ctcj.invariants.J_BASIS[4]
and t t lam lam; four-loop basis: ctcj.invariantsJ4 (data/basis4_J.json).
"""
import json
import os

import sympy as sp

from . import epsseries as S
from .assemble import position_ratio, totals
from .invariants import J_BASIS

e = S.EP
W2S = ["Tr(TTaa)", "Tr(TaTa)"]
TTLL = "t t lam lam"


def load_insertions(path):
    d = json.load(open(path))
    ins = {}
    for key, v in d["insertions"].items():
        w, kind = key.split("|")
        ins[(w, kind)] = {n: sp.sympify(c) for n, c in v.items()}
    basis = d["basis"]
    names4 = [n for s in ("y6", "y4lam", "y2lam2", "lam3") for n in basis[s]]
    return ins, names4, basis


def assemble_J4(rec123, rec4, ins_path):
    tot = totals(rec123 + rec4)
    conj = lambda v: sp.expand(v).subs(sp.I, -sp.I)          # all other symbols are real
    for k, v in tot.items():
        assert sp.expand(v - conj(v)) == 0, f"total of {k} is not real"
    tot = {k: sp.expand(v) for k, v in tot.items()}
    assert all(tot.get(k, 0) == 0 for k in ("Tr(TTaa)Tr(bb)",)), "disconnected three-loop word must vanish"
    norm = tot["Tr(TT)"]
    ins, names4, basis = load_insertions(ins_path)
    R = lambda L, k: position_ratio("J", L, tot.get(k, 0), norm)
    R2 = {w: R(2, w) for w in W2S}
    keys3 = list(J_BASIS[4]) + [TTLL]
    R3 = {k: R(3, k) for k in keys3}
    g = lambda key, n: ins.get(key, {}).get(n, 0)
    c = {}
    for w in W2S:
        c[w] = R2[w]
    for k in keys3:                                   # three-loop invariants (renormalised)
        x = R3[k]
        for w in W2S:
            x += g((w, "b1"), k) * sp.expand(R2[w] / (2 * e))
        c[k] = S.trunc(sp.expand(x))
    for m in names4:                                  # four-loop invariants
        x = R(4, m)
        for k in J_BASIS[4]:
            x += g((k, "b1"), m) * sp.expand(R3[k] / (2 * e))
        x += g((TTLL, "bl"), m) * sp.expand(R3[TTLL] / (2 * e))
        for w in W2S:
            x += g((w, "b2"), m) * sp.expand(R2[w] / (4 * e))
            x += g((w, "11"), m) * sp.expand(R2[w] / (4 * e ** 2))
        c[m] = S.trunc(sp.expand(x))
    poles = {k: [sp.simplify(v.coeff(e, j)) for j in (-4, -3, -2, -1)] for k, v in c.items()}
    # fixed-point reduction
    Lt = sp.Symbol("Lt")
    sub = lambda x: sp.expand(sp.expand(x).subs(S.ELL, Lt - S.GAMMA_E - S.LOGPI))
    co = lambda k, j: sub(c[k].coeff(e, j))
    f = {w: co(w, 0) for w in W2S}
    for k in keys3:
        f[k] = co(k, 0) + sum((co(w, 1) / 2) * g((w, "b1"), k) for w in W2S)
    for m in names4:
        x = co(m, 0)
        for w in W2S:
            x += (co(w, 1) / 2) * g((w, "b2"), m)                                  # eps W2 -> DW2[b2]
            for k in J_BASIS[4]:
                x += (co(w, 2) / 4) * g((w, "b1"), k) * g((k, "b1"), m) / 2       # eps^2 W2
        for k in J_BASIS[4]:
            x += (co(k, 1) / 2) * g((k, "b1"), m) / 2                              # eps I4 -> DI4[b1]/2
        x += (co(TTLL, 1) / 2) * g((TTLL, "bl"), m) / 2                            # eps ttll -> D[bl]/2
        f[m] = sp.nsimplify(sp.expand(x))
    f = {k: sp.nsimplify(sp.expand(v)) for k, v in f.items()}
    logs_ok = all(sp.diff(v, Lt) == 0 for v in f.values())
    d = sp.Symbol("d")
    free = {k: sp.factor(sp.simplify((tot.get(k, 0) / norm).subs(e, (4 - d) / 2))) for k in ("Tr(TT)", "t.t")}
    return dict(coef=f, poles=poles, logs_ok=logs_ok, totals=tot, insertions=ins, names4=names4,
                basis=basis, free=free, renorm=c)
