#!/usr/bin/env python3
"""N=2 SUSY test at O(eps^3): chiral XY with N = 2 (one 3d two-component Dirac = N=2 Wess-Zumino point).
C_J^R / C_J^R,free must equal C_T / C_T,free.  The O(eps^3) shift u3 of y*^2 and the O(eps^2) shift l2 of
lambda* are kept symbolic; l2 can be supplied (two-loop beta_lambda) via --l2."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import sympy as sp
from ctcj import models as M
from ctcj.invariants import T_BASIS, J_BASIS
from ctcj.invariants4 import T_basis4
from ctcj.basisJ4 import candidates
from ctcj.net import evaluate, words_net

ROOT = os.path.join(os.path.dirname(__file__), "..")
eps, N, y2, lam = M.eps, M.N, M.y2, M.lam
u3, l2 = sp.symbols("u3 l2")
d = 4 - eps
loc = {"z3": sp.Symbol("z3"), "N": N}
fT = {k: sp.sympify(v, locals=loc) for k, v in json.load(open(os.path.join(ROOT, "results", "coefficients4_T.json"))).items()}
fJ = {k: sp.sympify(v, locals=loc) for k, v in json.load(open(os.path.join(ROOT, "results", "coefficients4_J.json")))["coef"].items()}

ts, A, t, _ = M.MODELS["chiral XY (NJLY)"]
fp, (B1, B2, BL) = M.fixed_point(ts, 1)
fp = {k: sp.nsimplify(v.subs(N, 2)) for k, v in fp.items()}
u1, uu2 = fp[y2].coeff(eps, 1), fp[y2].coeff(eps, 2)
l1 = fp[lam].coeff(eps, 1)
print("u1, u2, l1 =", u1, uu2, l1)
Y = np.array([np.array(x, dtype=complex) for x in ts])
dd = np.eye(2)
S = np.einsum("ab,cd->abcd", dd, dd) + np.einsum("ac,bd->abcd", dd, dd) + np.einsum("ad,bc->abcd", dd, dd)
th = dict(y=Y, lam=S, T=np.array(A, dtype=complex) / 3, t=np.array(t, dtype=float) / 3, mult=1.0)

def ydeg(k):
    return sum(c in "abcdefgh" for c in "".join(p for p in k.split("Tr(")[1:]).replace("T", "")) if "Tr(" in k else 0

def rat(x):
    from fractions import Fraction
    z = complex(x)
    assert abs(z.imag) < 1e-9 * max(1, abs(z))
    f = Fraction(z.real).limit_denominator(10 ** 6)
    assert abs(float(f) - z.real) < 1e-9 * max(1, abs(z.real))
    return sp.Rational(f.numerator, f.denominator)

# lambda* to O(eps^2) from the two-loop beta_lambda of JOS (ctcj.betasJ.beta2_lam, validated against FGKT):
from ctcj.betasJ import beta2_lam
Ytens = np.array([np.array(x, dtype=complex) for x in ts])
S2 = np.einsum("ab,cd->abcd", np.eye(2), np.eye(2)); S2 = S2 + S2.transpose(0, 2, 1, 3) + S2.transpose(0, 3, 2, 1)
rows, vals = [], []
for uu in (0.0, 0.5, 1.0, 2.0):
    for ll in (0.0, 0.5, 1.0, 2.0):
        b = beta2_lam(np.sqrt(uu) * Ytens, ll * S2, 1.0)
        vals.append(np.einsum("abcd,abcd->", b, S2) / np.einsum("abcd,abcd->", S2, S2))
        rows.append([ll ** 3, ll ** 2 * uu, ll * uu ** 2, uu ** 3])
c3 = [rat(x) for x in np.linalg.lstsq(np.array(rows), np.array(vals), rcond=None)[0]]
BL2 = c3[0] * lam ** 3 + c3[1] * lam ** 2 * y2 + c3[2] * lam * y2 ** 2 + c3[3] * y2 ** 3
BL1 = sp.expand(BL.subs(N, 2))
L2s = sp.Symbol("L2s")
eq = sp.expand((-eps * lam + BL1 + BL2).subs({y2: u1 * eps + uu2 * eps ** 2, lam: l1 * eps + L2s * eps ** 2}))
l2_val = sp.solve(eq.coeff(eps, 3), L2s)[0]
print("two-loop beta_lambda on S (chiral XY, N=2):", BL2, "  ->  l2 =", l2_val, "  (u2 =", uu2, ")")
ystar = u1 * eps + uu2 * eps ** 2 + u3 * eps ** 3
lstar = l1 * eps + l2 * eps ** 2
# ---- C_T
IT = {}
for order in (2, 4, -2):
    for k, fn in T_BASIS[order].items():
        IT[k] = rat(fn(th))
for k, fn in T_basis4().items():
    IT[k] = rat(fn(th))
def wT(k):
    nl = k.count("lam")
    ny = {2: 1, 4: 2}.get(len(k), None)
    return nl
CT = 2 + (d - 1) / 2 * 2
for k, c in fT.items():
    nl = k.count("lam")
    ny = 0 if k.startswith("lam") and "Tr" not in k else None
    # y-degree: count scalar letters inside traces
    tr = "".join(p.split(")")[0] for p in k.split("Tr(")[1:])
    ny = len(tr)
    CT += c * IT[k] * ystar ** (ny // 2) * lstar ** nl
CTfree = 2 + (d - 1)
rT = sp.expand(sp.series(CT / CTfree, eps, 0, 4).removeO())
# ---- C_J (R-current)
nets = {}
for s in ("y6", "y4lam", "y2lam2", "lam3"):
    nets.update(candidates(s))
nets["t t lam lam"] = words_net([], extra=[("t", "ae"), ("t", "bf"), ("lam", "abcd"), ("lam", "efcd")])
for k in list(J_BASIS[2]) + list(J_BASIS[4]):
    nets[k] = words_net([p.rstrip(")") for p in k.split("Tr(") if p])
CJ = J_BASIS[0]["Tr(TT)"](th).real + (np.einsum("ab,ab->", th["t"], th["t"])) / (d - 2)
CJ = sp.nsimplify(round(float(J_BASIS[0]["Tr(TT)"](th).real), 12)) + sp.nsimplify(round(float((th["t"] ** 2).sum()), 12)) / (d - 2)
for k, c in fJ.items():
    if c == 0:
        continue
    nl = k.count("lam")
    tr = "".join(p.split(")")[0] for p in k.split("Tr(")[1:]).replace("T", "")
    CJ += c * rat(evaluate(nets[k], th)) * ystar ** (len(tr) // 2) * lstar ** nl
Rphi, Rpsi = (d - 2) / (d - 1), 1 / (d - 1)
CJfree = 2 * Rpsi ** 2 + 2 * Rphi ** 2 / (d - 2)
rJ = sp.expand(sp.series(CJ / CJfree, eps, 0, 4).removeO())
print("C_T/C_T,free   =", rT)
print("C_J^R/C_J^R,fr =", rJ)
diff = sp.expand(rT - rJ)
print("difference     =", sp.collect(diff, [eps, u3, l2]))
print("difference at the MS-bar fixed point (l2 from two-loop beta_lambda):", sp.simplify(diff.subs(l2, l2_val)))
ok = sp.simplify(diff.subs(l2, l2_val)) == 0 and sp.diff(diff, u3) == 0
print("PASS" if ok else "FAIL", "N=2 SUSY: C_J^R/C_J^R,free = C_T/C_T,free through eps^3")
sys.exit(0 if ok else 1)
