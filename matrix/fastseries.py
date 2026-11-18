"""Fast truncated eps-series (order 3) with zeta3 tracked linearly: a series is an array c[k, j] for eps^k zeta3^j, j=0,1."""
import numpy as np, sympy as sp
from formulas import CT2, CT3, CJ2, CJ3, evaluate

ORD = 4
eps, z3 = sp.symbols('epsilon zeta3')

def S(expr):
    """sympy expression in eps, zeta3 (and possibly sqrt) -> array [ORD,2]."""
    out = np.zeros((ORD, 2))
    e = sp.expand(sp.series(sp.sympify(expr), eps, 0, ORD).removeO()) if sp.sympify(expr).has(eps) else sp.sympify(expr)
    for k in range(ORD):
        ck = sp.expand(e.coeff(eps, k)) if k else sp.expand(e.subs(eps, 0))
        a1 = ck.coeff(z3, 1); a0 = sp.expand(ck - a1*z3)
        assert not a0.has(z3)
        out[k] = [float(a0), float(a1)]
    return out

def mul(a, b):
    c = np.zeros((ORD, 2))
    for i in range(ORD):
        for j in range(ORD - i):
            c[i + j, 0] += a[i, 0]*b[j, 0]
            c[i + j, 1] += a[i, 0]*b[j, 1] + a[i, 1]*b[j, 0]   # drop zeta3^2 (beyond eps^3 here)
    return c

def const(x):
    c = np.zeros((ORD, 2)); c[0, 0] = x; return c

def inv(a):
    """1/a for a[0] = const != 0 (rational part)."""
    a0 = a[0, 0]; x = a.copy(); x[0, 0] = 0
    r = const(1/a0); term = const(1/a0); xr = x/a0
    for k in range(1, ORD):
        term = mul(term, -xr); r = r + term
    return r

def power(a, k):
    r = const(1.0)
    for _ in range(k): r = mul(r, a)
    return r

def assemble(formula, Yunit, ysq, lam_parts, lam_coef, T=None, t=None):
    res = evaluate(formula, Yunit, lam_parts, T, t)
    tot = np.zeros((ORD, 2))
    for (ny, combo), (r, z) in res.items():
        c = power(ysq, ny // 2)
        for nm in combo: c = mul(c, lam_coef[nm])
        assert abs(np.imag(r)) < 1e-9 * max(1.0, abs(r)) and abs(np.imag(z)) < 1e-9 * max(1.0, abs(z)), (ny, combo)
        coefser = np.zeros((ORD, 2)); coefser[0] = [np.real(r), np.real(z)]
        tot = tot + mul(coefser, c)
    return tot

def ct_ratio(Yunit, ysq, lam_parts, lam_coef, n_s, Tr1):
    free = const(n_s + 1.5*Tr1); free[1, 0] = -0.5*Tr1
    corr = assemble(CT2, Yunit, ysq, lam_parts, lam_coef) + assemble(CT3, Yunit, ysq, lam_parts, lam_coef)
    return const(1) + mul(corr, inv(free))

def cj_ratio(Yunit, ysq, lam_parts, lam_coef, T, t):
    trTT = float(np.trace(T @ T).real); tt = float(np.einsum('ab,ab->', t, t))
    free = const(trTT) + tt*S(1/(2 - eps))
    corr = assemble(CJ2, Yunit, ysq, lam_parts, lam_coef, T, t) + assemble(CJ3, Yunit, ysq, lam_parts, lam_coef, T, t)
    return const(1) + mul(corr, inv(free))

def numeric(a):
    return a[:, 0] + a[:, 1]*1.2020569031595942
