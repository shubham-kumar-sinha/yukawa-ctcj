"""From per-diagram G-scheme values to C_T and C_J at a fixed point.

1. Sum X over classes per basis invariant.
2. Fourier transform each L-loop term (Q^2)^(-L ep) to position space and normalise to a free field
   (T: free real scalar; J: free fermion per unit Tr(TT)).
3. MS renormalisation of the Yukawa tensor: y0 = mu^ep (y + beta1(y)/eps), eps = 2 ep, with the
   one-loop beta function of Jack-Osborn-Steudtner (JOS) eq. (8.1):
       beta1^a = 2 y^b y^a y^b + 1/2 (y^b y^b y^a + y^a y^b y^b) + 1/2 y^b Tr(y^b y^a).
   lambda first enters at O(lambda^2) and needs no counterterm at this order.
4. Check that all poles cancel per invariant and that log(mu^2 x^2) multiplies only terms that
   vanish at a fixed point.
5. Fixed-point form: eliminate the O(eps) part of two-loop coefficients with
       eps * W2(y) = W2-insertion of beta1  (from beta_y = -eps/2 y + beta1 = 0),
   which gives eps-independent rational coefficients.
"""
import numpy as np
import sympy as sp

from . import epsseries as S
from .invariants import (T_BASIS, J_BASIS, random_generic, symmetric_ensemble, tr_words, rational)

e, ell = S.EP, S.ELL


def _K(kind, L):
    d = 4 - 2 * e
    beta = -L * e
    s = d / 2 + beta
    # coefficient of the structure with all free indices on x: x_mu x_nu x_rho x_si (T), x_mu x_nu (J),
    # xslash x_mu x_nu (S, the supercurrent; ctcj/supercurrent.py)
    # j: xslash for <j jbar>, j = t_ab phi_a psi_b (ctcj/supercurrent.py)
    poly = {"T": s * (s + 1) * (s + 2) * (s + 3), "J": s * (s + 1), "S": s * (s + 1) * (s + 2), "j": s}[kind]
    poly = sp.expand(poly * (1 - (L + 1) * e) * L)   # Gamma(d/2+beta) and 1/Gamma(-beta) prefactors
    lg = beta * S.LOG4 + e * S.LOGPI + S.lgam1(-(L + 1) * e) - S.lgam1(L * e)
    return poly, lg


def position_ratio(kind, L, X, Xnorm):
    """b_L / b_free for an L-loop term with G-scheme value X (couplings rescaled y -> 4 pi y)."""
    pL, gL = _K(kind, L)
    p1, g1 = _K(kind, 1)
    lgG = (L - 1) * (e * (S.LOG4 + S.LOGPI) + S.lgam1(e) + 2 * S.lgam1(-e) - S.lgam1(-2 * e))
    ex = S.exp_series(sp.expand(gL - g1 + lgG + (L - 1) * e * ell))
    pre = S.mul(S.mul(pL, S.inv_series(p1)), S.inv_series(sp.expand((1 - 2 * e) ** (L - 1))))
    return S.mul(S.mul(S.mul(pre, ex), S.trunc(sp.expand(X * e))), S.inv_series(sp.expand(Xnorm * e)))


def beta1(y):
    ns = y.shape[0]
    return [sum(2 * y[c] @ y[a] @ y[c] + 0.5 * (y[c] @ y[c] @ y[a] + y[a] @ y[c] @ y[c])
                + 0.5 * y[c] * np.trace(y[c] @ y[a]) for c in range(ns)) for a in range(ns)]


def _ct_T(th):
    y = th["y"]; b = beta1(y)
    return sum(np.trace(b[a] @ y[a]) + np.trace(y[a] @ b[a]) for a in range(len(b)))


def _ct_Ja(th):
    y, T = th["y"], th["T"]; b = beta1(y)
    return sum(np.trace(T @ T @ b[a] @ y[a]) + np.trace(T @ T @ y[a] @ b[a]) for a in range(len(b)))


def _ct_Jb(th):
    y, T = th["y"], th["T"]; b = beta1(y)
    return sum(np.trace(T @ b[a] @ T @ y[a]) + np.trace(T @ y[a] @ T @ b[a]) for a in range(len(b)))


def counterterm_insertions(kind):
    """beta1-insertions into the two-loop words, decomposed on the three-loop basis."""
    if kind == "T":
        ths, basis, cts = [random_generic(2 + i % 2, 3 + (i // 2) % 2) for i in range(8)], T_BASIS[4], {"Tr(aa)": _ct_T}
    else:
        ths, basis, cts = symmetric_ensemble(7), J_BASIS[4], {"Tr(TTaa)": _ct_Ja, "Tr(TaTa)": _ct_Jb}
    names = list(basis)
    B = np.array([[basis[n](th) for n in names] for th in ths])
    out = {}
    for w2, f in cts.items():
        v = np.array([f(th) for th in ths])
        sol = np.linalg.lstsq(B, v, rcond=None)[0]
        assert np.allclose(B @ sol, v)
        out[w2] = {n: rational(s) for n, s in zip(names, sol) if abs(s) > 1e-9}
    return out


def totals(records):
    tot = {}
    for r in records:
        for k, cf in r["decomp"].items():
            tot[k] = sp.expand(tot.get(k, 0) + cf * r["X"])
    return tot


def assemble(kind, records):
    basis = T_BASIS if kind == "T" else J_BASIS
    tot = totals(records)
    norm = tot["ns"] if kind == "T" else tot["Tr(TT)"]
    free = {k: sp.factor(sp.simplify((tot.get(k, 0) / norm).subs(e, (4 - sp.Symbol("d")) / 2)))
            for k in basis[0]}
    R2 = {k: position_ratio(kind, 2, tot.get(k, 0), norm) for k in basis[2]}
    ins = counterterm_insertions(kind)
    ren = {}
    for k in list(basis[4]) + list(basis[-2]):
        x = position_ratio(kind, 3, tot.get(k, 0), norm)
        if k in basis[4]:
            for w2 in basis[2]:
                x += ins.get(w2, {}).get(k, 0) * sp.expand(R2[w2] / (2 * e))
        ren[k] = S.trunc(sp.expand(x))
    for k in basis[2]:
        ren[k] = R2[k]
    # diagnostics
    poles = {k: [sp.simplify(v.coeff(e, j)) for j in (-3, -2, -1)] for k, v in ren.items()}
    Lt = sp.Symbol("Lt")
    sub = lambda x: sp.expand(sp.expand(x).subs(ell, Lt - S.GAMMA_E - S.LOGPI))
    coef = {}
    for k in basis[2]:
        coef[k] = sub(ren[k].coeff(e, 0))
    for k in list(basis[4]) + list(basis[-2]):
        x = sub(ren[k].coeff(e, 0))
        if k in basis[4]:
            for w2 in basis[2]:
                x += sub(ren[w2].coeff(e, 1)) * ins.get(w2, {}).get(k, 0) / 2
        coef[k] = sp.simplify(x)
    logs_ok = all(sp.diff(v, Lt) == 0 for v in coef.values())
    coef = {k: sp.nsimplify(v) for k, v in coef.items()}
    return dict(free=free, coef=coef, poles=poles, logs_ok=logs_ok, totals=tot, insertions=ins,
                two_loop_eps=[sub(ren[k].coeff(e, 1)) for k in basis[2]])
