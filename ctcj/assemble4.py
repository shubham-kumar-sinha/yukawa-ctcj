"""C_T at O(eps^3): four-loop diagrams + two-loop renormalisation of the tensor couplings.

Bare -> renormalised (eps = 4 - d = 2 ep):
    y0 = y + beta1/eps + beta2/(2 eps) + D2/eps^2,   D2 = 1/2 Dbeta1[beta1]     ('t Hooft consistency)
    lam0 = lam + betalam1/eps
Fixed-point reduction: with (eps/2) y = beta_y and eps lam = beta_lam at a fixed point,
    eps * I = DI[(beta_y, beta_lam)] / w(I),   w = (y-degree)/2 + (lam-degree),
applied until all coefficients are eps-independent.  All insertions are decomposed numerically on
random generic tensors.
"""
import numpy as np
import sympy as sp

from . import epsseries as S
from .assemble import position_ratio, totals
from .invariants import T_BASIS
from .invariants4 import T_basis4, theories, independent, decompose_vals, tr_prod_deriv, WORDS4
from .theory import Theory

e = S.EP


def _betas(th):
    y, lam = th["y"], th["lam"]
    T = Theory(Y=[y], mult=1.0)
    b1 = T.beta1_y(y).real
    b2 = T.beta2_y(y, lam).real
    bl = T.beta1_lam(y, lam)
    f = lambda t: T.beta1_y(y + t * b1).real                      # cubic in t
    D2 = 0.5 * (8 * (f(1) - f(-1)) - (f(2) - f(-2))) / 12
    return b1, b2, bl, D2


def _dir(fun, th, dy=None, dl=None):
    """exact directional derivative of a polynomial invariant (degree <= 6) via a 6-point stencil"""
    def g(t):
        t2 = dict(th)
        if dy is not None: t2["y"] = th["y"] + t * dy
        if dl is not None: t2["lam"] = th["lam"] + t * dl
        return fun(t2)
    # derivative at 0 of a polynomial of degree <= 6: central differences, exact up to degree 6
    return (45 * (g(1) - g(-1)) - 9 * (g(2) - g(-2)) + (g(3) - g(-3))) / 60


def insertions():
    ths = theories(60)
    for th in ths:
        th["b1"], th["b2"], th["bl"], th["D2"] = _betas(th)
    B4 = {k: v for k, v in T_BASIS[4].items()}
    B6 = T_basis4()
    n4, M4 = independent(B4, ths)
    n6, M6 = independent(B6, ths)
    I2 = T_BASIS[2]["Tr(aa)"]
    ins = {}
    vals = lambda f: np.array([np.real(f(th)) for th in ths], dtype=float)
    E = np.einsum
    ins[("2", "b1")] = decompose_vals(vals(lambda th: 2 * E("aij,aji->", th["y"], th["b1"])), n4, M4)
    ins[("2", "b2")] = decompose_vals(vals(lambda th: 2 * E("aij,aji->", th["y"], th["b2"])), n6, M6)
    ins[("2", "11")] = decompose_vals(vals(lambda th: E("aij,aji->", th["b1"], th["b1"])
                                           + 2 * E("aij,aji->", th["y"], th["D2"])), n6, M6)
    for k in n4:
        ins[(k, "b1")] = decompose_vals(vals(lambda th: tr_prod_deriv(WORDS4[k], th, "b1")), n6, M6)
    ins[("lam.lam", "bl")] = decompose_vals(vals(lambda th: 2 * E("abcd,abcd->", th["lam"], th["bl"])), n6, M6)
    return ins, n4, n6, (ths, M6)


def assemble_T4(rec123, rec4):
    tot = totals(rec123 + rec4)
    norm = tot["ns"]
    ins, n4, n6, _ = insertions()
    R = lambda L, k: position_ratio("T", L, tot.get(k, 0), norm)
    R2 = R(2, "Tr(aa)")
    keys3 = n4 + ["lam.lam"]
    R3 = {k: R(3, k) for k in keys3}
    c = {"Tr(aa)": R2}
    for k in keys3:                                   # three-loop invariants (renormalised to O(e))
        x = R3[k]
        if k in n4:
            x += ins[("2", "b1")].get(k, 0) * sp.expand(R2 / (2 * e))
        c[k] = S.trunc(sp.expand(x))
    for m in n6:                                      # four-loop invariants
        x = R(4, m)
        for k in n4:
            x += ins[(k, "b1")].get(m, 0) * sp.expand(R3[k] / (2 * e))
        x += ins[("lam.lam", "bl")].get(m, 0) * sp.expand(R3["lam.lam"] / (2 * e))
        x += ins[("2", "b2")].get(m, 0) * sp.expand(R2 / (4 * e))
        x += ins[("2", "11")].get(m, 0) * sp.expand(R2 / (4 * e ** 2))
        c[m] = S.trunc(sp.expand(x))
    poles = {k: [sp.simplify(v.coeff(e, j)) for j in (-4, -3, -2, -1)] for k, v in c.items()}
    # fixed-point reduction (eps = 2e):  c(e) = c0 + c1 e + c2 e^2 = c0 + c1/2 eps + c2/4 eps^2
    Lt = sp.Symbol("Lt")
    sub = lambda x: sp.expand(sp.expand(x).subs(S.ELL, Lt - S.GAMMA_E - S.LOGPI))
    co = lambda k, j: sub(c[k].coeff(e, j))
    f = {"Tr(aa)": co("Tr(aa)", 0)}
    for k in keys3:
        f[k] = co(k, 0) + (co("Tr(aa)", 1) / 2) * ins[("2", "b1")].get(k, 0)
    for m in n6:
        x = co(m, 0)
        x += (co("Tr(aa)", 1) / 2) * ins[("2", "b2")].get(m, 0)                   # eps I2 -> DI2[beta2]
        for k in n4:
            x += (co(k, 1) / 2) * ins[(k, "b1")].get(m, 0) / 2                       # eps I4 -> DI4[b1]/2
            x += (co("Tr(aa)", 2) / 4) * ins[("2", "b1")].get(k, 0) * ins[(k, "b1")].get(m, 0) / 2
        x += (co("lam.lam", 1) / 2) * ins[("lam.lam", "bl")].get(m, 0) / 2           # eps ll -> D(ll)[bl]/2
        f[m] = sp.nsimplify(sp.expand(x))
    logs_ok = all(sp.diff(v, Lt) == 0 for v in f.values())
    return dict(coef=f, poles=poles, logs_ok=logs_ok, totals=tot, insertions=ins, n4=n4, n6=n6)
