"""C_T/C_T,free and C_J/C_J,free through O(eps^3) at the stable fixed points of the chiral Ising, XY and
Heisenberg GNY models.

Fixed points: three-loop RG functions of Zerf et al. (ctcj.gny3loop), mapped to our couplings by
    y^2 (ours) = y_Z / 2,   lambda_S (ours) = 4 lambda_Z,   N (ours, = Tr 1) = 4 N_Z,
a dictionary fixed at one loop and verified on every two-loop term (scripts/gny_eps3.py).
Coefficients: the fixed-point forms in results/coefficients4_{T,J}.json (all orders through four loops).
Invariants are evaluated as tensor networks (ctcj.net) with the fermion multiplicity included.
"""
import json
import os
import re

import numpy as np
import sympy as sp

from . import gny3loop as G
from .net import evaluate, words_net

ROOT = os.path.join(os.path.dirname(__file__), "..")
eps = G.eps
Z3 = sp.Symbol("z3")

s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s2 = np.array([[0, -1j], [1j, 0]])
s3 = np.array([[1, 0], [0, -1]], dtype=complex)

# model: (Yukawa matrices with unit coupling, fermion multiplicity per unit N, current (T, t) or None)
MODELS = {
    "chiral Ising": (np.ones((1, 1, 1), complex), 1.0, (np.eye(1, dtype=complex), np.zeros((1, 1)))),
    "chiral XY": (np.array([s3, s1]), 0.5, (s2, np.array([[0, 2.0], [-2.0, 0]]))),
    "chiral Heisenberg": (np.array([s1, s2, s3]), 0.5, (s3 / 2, np.array([[0, 1.0, 0], [-1.0, 0, 0], [0, 0, 0]]))),
}


def S_tensor(n):
    d = np.eye(n)
    return np.einsum("ab,cd->abcd", d, d) + np.einsum("ac,bd->abcd", d, d) + np.einsum("ad,bc->abcd", d, d)


def name_net(name):
    """Tensor network for a coefficient name of results/coefficients4_{T,J}.json."""
    if name == "lam.lam":
        return words_net([], extra=[("lam", "abcd"), ("lam", "abcd")])
    if name == "t t lam lam":
        return words_net([], extra=[("t", "ae"), ("t", "bf"), ("lam", "abcd"), ("lam", "efcd")])
    words = re.findall(r"Tr\(([^)]*)\)", name)
    extra = [(n, idx) for n, idx in re.findall(r"\b(lam|t)_([a-z]+)", name)]
    if re.search(r"(^|\s)lam\s", name):                     # bare 'lam': contracted with the labels appearing once
        labs = "".join(w.replace("T", "") for w in words)
        once = "".join(c for c in dict.fromkeys(labs) if labs.count(c) == 1)
        assert len(once) == 4, name
        extra.append(("lam", once))
    return words_net(words, extra=extra)


def degrees(name):
    """(y-degree, lambda-degree) of an invariant name."""
    if name == "lam.lam":
        return 0, 2
    if name == "t t lam lam":
        return 0, 2
    tr = "".join(re.findall(r"Tr\(([^)]*)\)", name)).replace("T", "")
    return len(tr), name.count("lam")


def load_coefficients(kind):
    if kind == "T":
        d = json.load(open(os.path.join(ROOT, "results", "coefficients4_T.json")))
    else:
        d = json.load(open(os.path.join(ROOT, "results", "coefficients4_J.json")))["coef"]
    out = {}
    for k, v in d.items():
        c = sp.sympify(v, locals={"z3": Z3})
        if c != 0:
            out[k] = c.subs(Z3, sp.zeta(3))
    return out


def fixed_point_ours(model, N, order=3):
    """y*^2 and lambda*_S in our normalisation as eps-series, for N = Tr 1."""
    ys, ls = G.fixed_point(model, sp.Rational(N) / 4, order)
    Y2 = sum(ys[i] / 2 * eps**i for i in range(1, order + 1))
    L = sum(4 * ls[i] * eps**i for i in range(1, order + 1))
    return Y2, L


def _value(x):
    from fractions import Fraction
    z = complex(x)
    assert abs(z.imag) < 1e-9 * max(1.0, abs(z)), x
    f = Fraction(z.real).limit_denominator(10**7)
    assert abs(float(f) - z.real) < 1e-9 * max(1.0, abs(z.real)), x
    return sp.Rational(f.numerator, f.denominator)


def series(model, N, kind, order=3, current=None, odd=False):
    """C/C_free through eps^order: kind 'T' or 'J'.  N = Tr 1 (number of 3d Majorana fermions).
    odd=True (T, order >= 3) adds the odd fermion-loop term (ctcj.formulas4.KAPPA_T_ODD).  The models use y = 1 (Ising) with
    Tr 1 = N, which reproduces every even invariant but is the actual flavour structure only of the theory of N two-component
    Majorana fermions with equal signs of y.  So use odd=True only where that is the theory meant: chiral Ising at N = 1, and at
    N = 2 for the U(1)-symmetric theory (one two-component Dirac fermion).  The chiral Ising models at N = 4, 8 (graphene) come
    from four dimensions and have Tr(yyy) = 0; for chiral XY and Heisenberg the odd invariants vanish."""
    Y, mper, cur = MODELS[model]
    T, t = current if current is not None else cur
    ns = Y.shape[0]
    th = dict(y=Y, lam=S_tensor(ns), T=np.asarray(T, complex), t=np.asarray(t, float), mult=mper * N)
    Y2, L = fixed_point_ours(model, N, order)
    d = 4 - eps
    coef = load_coefficients(kind)
    if kind == "T":
        free = ns + (d - 1) / 2 * N
    else:
        free = _value(evaluate(words_net(["TT"]), th)) + _value(evaluate(words_net([], extra=[("t", "ab"), ("t", "ab")]), th)) / (d - 2)
    tot = free
    for k, c in coef.items():
        ny, nl = degrees(k)
        if ny // 2 + nl > order:            # an invariant of weight w = ny/2 + nl starts at eps^w
            continue
        tot += c * _value(evaluate(name_net(k), th)) * Y2**(ny // 2) * L**nl
    if odd and order >= 3:
        # Not for C_J: these models write a fermion current as T = 1 with Tr 1 = N, which reproduces every even invariant
        # but not Tr(T y^a y^b y^c).  In the actual theories it vanishes (e.g. T = sigma_2 for the U(1) theory at N = 2).
        assert kind == "T", "odd-loop term of C_J needs the actual T; it vanishes for the chiral models (use ctcj.theory)"
        from .formulas4 import KAPPA_T_ODD, odd_invariant_T
        a, b = KAPPA_T_ODD
        tot += (sp.Rational(a.numerator, a.denominator) + sp.Rational(b.numerator, b.denominator) * sp.zeta(3)) \
            * _value(odd_invariant_T(Y, mper * N)) * Y2**3
    return sp.expand(sp.series(tot / free, eps, 0, order + 1).removeO())
