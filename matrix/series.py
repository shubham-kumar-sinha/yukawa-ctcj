"""Evaluate C_T/C_T,free and C_J/C_J,free as eps-series at a fixed point, given explicit tensors."""
import sympy as sp
import numpy as np
from formulas import CT2, CT3, CJ2, CJ3, evaluate

eps, z3 = sp.symbols('epsilon zeta3')

def _assemble(formula, Yunit, ysq, lam_parts, lam_coef, T=None, t=None):
    """sum over invariants: value * (y^2)^(ny/2) * prod lam coefs; returns sympy expr in eps, zeta3."""
    res = evaluate(formula, Yunit, lam_parts, T, t)
    tot = 0
    for (ny, combo), (r, z) in res.items():
        assert ny % 2 == 0
        c = ysq ** (ny // 2)
        for nm in combo:
            c = c * lam_coef[nm]
        tot += (sp.Float(float(np.real(r)), 30) + sp.Float(float(np.real(z)), 30) * z3) * c
    return tot

def ct_ratio(Yunit, ysq, lam_parts, lam_coef, n_s, Tr1, order=3):
    free = n_s + sp.Rational(1, 2) * (3 - eps) * Tr1
    corr = _assemble(CT2, Yunit, ysq, lam_parts, lam_coef)
    if order >= 3:
        corr += _assemble(CT3, Yunit, ysq, lam_parts, lam_coef)
    r = sp.series(1 + corr / free, eps, 0, order + 1).removeO()
    return sp.expand(r)

def cj_ratio(Yunit, ysq, lam_parts, lam_coef, T, t, order=3):
    trTT = float(np.trace(T @ T).real)
    tt = float(np.einsum('ab,ab->', t, t)) if t is not None else 0.0
    free = trTT + tt / (2 - eps)
    corr = _assemble(CJ2, Yunit, ysq, lam_parts, lam_coef, T, t)
    if order >= 3:
        corr += _assemble(CJ3, Yunit, ysq, lam_parts, lam_coef, T, t)
    r = sp.series(1 + corr / free, eps, 0, order + 1).removeO()
    return sp.expand(r)

def coeffs(expr, order=3):
    expr = sp.expand(expr)
    return [sp.nsimplify(0) + expr.coeff(eps, k) for k in range(order + 1)]
