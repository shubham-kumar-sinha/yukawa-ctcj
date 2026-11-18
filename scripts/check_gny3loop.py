#!/usr/bin/env python3
"""Checks of the three-loop chiral GNY RG functions in ctcj/gny3loop.py (transcribed from Zerf et al.,
arXiv:1709.05057) and of their use in our scheme.  (~2 min; needs internet once for step 1)

1. Transcription: eta_phi, eta_psi, 1/nu through eps^3 for chiral Ising, XY and Heisenberg at N_Dirac = 1/4, 1/2, 1, 2
   must equal the exponent files published with arXiv:1709.05057 (downloaded to build/zerf_anc/).
2. Dictionary to our couplings (y^2 = y_Z/2, lambda_S = 4 lambda_Z, Tr 1 = 4 N_Z): derived from the one-loop terms,
   then every two-loop term of beta_y and beta_lambda must agree with our own two-loop beta functions
   (ctcj.models for beta_y, ctcj.betasJ.beta2_lam for beta_lambda, the latter validated against FGKT).
3. Supersymmetry: at N=2 (chiral XY) eta_phi = eta_psi = eps/3 exactly and lambda* = y*^2; at N=1 (chiral Ising)
   eta_phi = eta_psi and 1/nu = (d - eta)/2, all through eps^3."""
import os, sys, urllib.request
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fractions import Fraction
import numpy as np
import sympy as sp
from ctcj import gny3loop as G, models as Mo
from ctcj.betasJ import beta2_lam

ROOT = os.path.join(os.path.dirname(__file__), "..")
eps, N = G.eps, G.N
ok_all = True


def report(name, ok, detail=""):
    global ok_all
    ok_all &= bool(ok)
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else "   " + detail), flush=True)


# 1. transcription vs published exponent files
FILES = {"chiral Ising": "chiralIsing.m", "chiral XY": "chiralXY.m", "chiral Heisenberg": "chiralHeisenberg.m"}
cache = os.path.join(ROOT, "build", "zerf_anc")
os.makedirs(cache, exist_ok=True)
for model, fn in FILES.items():
    p = os.path.join(cache, fn)
    if not os.path.exists(p):
        urllib.request.urlretrieve(f"https://arxiv.org/src/1709.05057/anc/{fn}", p)
    anc = G.load_ancillary(p)
    worst = 0.0
    for Nv in (sp.Rational(1, 4), sp.Rational(1, 2), sp.Integer(1), sp.Integer(2)):
        ours = G.exponents(model, Nv, 3)
        for k in ("etaphi", "etapsi", "nuinv"):
            ref = sp.series(anc[k].subs(N, Nv), eps, 0, 4).removeO()
            worst = max(worst, max(float(abs(sp.N(sp.expand(ours[k] - ref).coeff(eps, j), 30))) for j in range(4)))
    report(f"{model}: eta_phi, eta_psi, 1/nu through eps^3 equal arXiv:1709.05057 (4 values of N)", worst < 1e-20, f"{worst:.1e}")

# 2. dictionary and two-loop agreement with our beta functions
names = {"chiral Ising": "chiral Ising (GNY)", "chiral XY": "chiral XY (NJLY)", "chiral Heisenberg": "chiral Heisenberg"}
a, b, c = sp.symbols("a b c", positive=True)


def S(n):
    d = np.eye(n)
    return np.einsum("ab,cd->abcd", d, d) + np.einsum("ac,bd->abcd", d, d) + np.einsum("ad,bc->abcd", d, d)


def bl2(ts, mval):
    Y = np.array([np.array(t.tolist(), dtype=complex) for t in ts]); Sn = S(len(ts)); rows, vals = [], []
    for u in (0., .5, 1., 2.):
        for l in (0., .5, 1., 2.):
            bb = beta2_lam(np.sqrt(u) * Y, l * Sn, mval)
            vals.append(np.einsum("abcd,abcd->", bb, Sn) / np.einsum("abcd,abcd->", Sn, Sn)); rows.append([l**3, l**2 * u, l * u**2, u**3])
    return [sp.Rational(Fraction(x).limit_denominator(10**6)) for x in np.linalg.lstsq(np.array(rows), np.array(vals), rcond=None)[0]]


for zm, om in names.items():
    ts = Mo.MODELS[om][0]
    B1, B2, BL = Mo.beta_functions(ts)
    m = [s for s in (B1 + BL).free_symbols if s.name == "m"][0]
    sub = {Mo.y2: a * G.y, Mo.lam: b * G.lam, m: c * N}
    eqs = []
    for ours, theirs in (((2 * B1 * Mo.y2).subs(sub) / a, G.RG[zm]["beta_y"][0]), (BL.subs(sub) / b, G.RG[zm]["beta_lam"][0])):
        eqs += sp.Poly(sp.expand(ours - theirs), G.y, G.lam, N).coeffs()
    s0 = sp.solve(eqs, [a, b, c], dict=True)[0]
    mper = 1 if zm == "chiral Ising" else sp.Rational(1, 2)       # our multiplicity per unit Tr 1
    ok1 = s0[a] == sp.Rational(1, 2) and s0[b] == 4 and s0[c] == 4 * mper
    co = [bl2(ts, mv) for mv in (1.0, 2.0, 3.0)]
    BL2 = sum(sp.interpolate([(1, co[0][k]), (2, co[1][k]), (3, co[2][k])], m) * mon
              for k, mon in enumerate([Mo.lam**3, Mo.lam**2 * Mo.y2, Mo.lam * Mo.y2**2, Mo.y2**3]))
    dy = sp.simplify((2 * B2 * Mo.y2).subs(sub).subs(s0) / s0[a] - G.RG[zm]["beta_y"][1])
    dl = sp.simplify(BL2.subs(sub).subs(s0) / s0[b] - G.RG[zm]["beta_lam"][1])
    report(f"{zm}: dictionary y^2 = y_Z/2, lambda = 4 lambda_Z, Tr1 = 4 N_Z; all two-loop terms agree", ok1 and dy == 0 and dl == 0,
           f"{s0}, {dy}, {dl}")

# 3. supersymmetry
d = 4 - eps
e2 = G.exponents("chiral XY", sp.Rational(1, 2), 3)
ys, ls = G.fixed_point("chiral XY", sp.Rational(1, 2), 3)
report("N=2 (chiral XY, N=2): eta_phi = eta_psi = eps/3 and lambda* = y*^2 through eps^3",
       sp.simplify(e2["etaphi"] - eps / 3) == 0 and sp.simplify(e2["etapsi"] - eps / 3) == 0
       and all(sp.simplify(4 * l - x / 2) == 0 for x, l in zip(ys[1:], ls[1:])))
e1 = G.exponents("chiral Ising", sp.Rational(1, 4), 3)
report("N=1 (chiral Ising, N=1): eta_phi = eta_psi and 1/nu = (d - eta)/2 through eps^3",
       sp.simplify(e1["etaphi"] - e1["etapsi"]) == 0 and sp.simplify(sp.expand(e1["nuinv"] - (d - e1["etaphi"]) / 2)) == 0)
print("ALL PASS" if ok_all else "SOME CHECKS FAILED")
sys.exit(0 if ok_all else 1)
