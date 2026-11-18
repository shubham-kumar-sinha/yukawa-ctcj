#!/usr/bin/env python3
"""Walk through every step of the method on the one-loop diagrams, printing each intermediate quantity,
and recover the textbook free-field values of C_T and C_J in general d.  See METHOD.md for the formulas.

  1. Wick contractions: enumerate labelled contractions, group into classes, weight = 1/symmetry factor.
  2. Integral: G-scheme value X of each class (recomputed with FORM/Forcer if available, else read from data/).
  3. Form factor: X -> A(Q^2) (Q^2)^beta in momentum space.
  4. Fourier transform: (Q^2)^beta -> c(beta)/(x^2)^(d/2+beta).
  5. Extraction: C_T, C_J from the coefficient of the (z1.x)^2 (z2.x)^2, resp. (z1.x)(z2.x), structure.
Run from the repository root:  python3 scripts/walkthrough_one_loop.py"""
import itertools, json, os, shutil, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fractions import Fraction
import sympy as sp
from ctcj.diagrams import generate

ROOT = os.path.join(os.path.dirname(__file__), "..")
d, ep = sp.symbols("d epsilon_p")          # ep = epsilon' with D = 4 - 2 ep, i.e. d = 4 - 2 ep
print(__doc__.split("\n\n")[0], "\n")

# ---------------------------------------------------------------- 1. diagrams and weights
print("1. Wick contractions.  A class's weight is the number of labelled contractions producing it, times")
print("   1/(nY! nQ! 24^nQ) from expanding exp(-S_int); this equals 1/(symmetry factor).")
classes = {k: generate(k, 0, 0) for k in ("T", "J")}
for k, cs in classes.items():
    for c in cs:
        print(f"   <{k}{k}> {c['name']}: operators {c['ops']}, fermion lines {list(c['ferm'])}, "
              f"scalar lines {[ (a[0], b[0]) for a, b in c['sig']]}, weight {c['weight']}")
print("   e.g. T_sigma(x) T_sigma(0): two scalar lines x-0; the two ways of pairing the legs give weight 2.\n")

# ---------------------------------------------------------------- 2. integrals
loc = {"ep": ep}
pub = {r["name"]: r for k in ("T", "J") for r in json.load(open(os.path.join(ROOT, "data", f"diagrams_{k}.json")))}
X = {}
have_form = (shutil.which(os.environ.get("FORM_BIN", "form")) is not None
             and os.path.exists(os.path.join(os.environ.get("FORCER_PATH", os.path.expanduser("~/forcer")), "forcer.h")))
print("2. Integrals.  X = G-scheme value of the projected correlator (Forcer normalisation, Q^2 = 1, D = 4 - 2 ep,")
print("   per loop eps'*G(1,1)); it already contains the weight, the fermion-loop sign and the Feynman-rule factors.")
for k, cs in classes.items():
    for c in cs:
        Xc = sp.sympify(pub[c["name"]]["X"], locals=loc)
        src = "data/"
        if have_form:
            from ctcj.forcer import run_class, EP
            cc = dict(c, kind=k, weight=Fraction(c["weight"]), name=c["name"] + "_walk")
            Xf = run_class(cc, os.path.join(ROOT, "build", "walkthrough"), depth=3).subs(EP, ep)
            assert sp.simplify(Xf - Xc) == 0, "FORM result differs from data/"
            src = "FORM/Forcer (identical to data/)"
        X[c["name"]] = Xc
        print(f"   {c['name']} {c['ops']}: X = {Xc}    [{src}]   flavour factor {pub[c['name']]['decomp']}")
print()

# ---------------------------------------------------------------- 3-5. form factor, Fourier transform, extraction
D = 4 - 2 * ep
G11 = sp.gamma(1 - ep) ** 2 * sp.gamma(ep) / ((4 * sp.pi) ** (2 - ep) * sp.gamma(2 - 2 * ep))
beta = -ep                                  # L-loop term ~ (Q^2)^(-L ep); here L = 1
s = D / 2 + beta
c_beta = 4 ** beta * sp.gamma(D / 2 + beta) / (sp.pi ** (D / 2) * sp.gamma(-beta))
Sd = 2 * sp.pi ** (D / 2) / sp.gamma(D / 2)
print("3. Form factors:  <T T> = (z1.z2)^2 Q^4 A_T (Q^2)^beta,  A_T = [ep G(1,1)] 2X/((D-1)(D+1)(D-2));")
print("                  <J J> = (z1.z2) Q^2 A_J (Q^2)^beta,      A_J = [ep G(1,1)] X/(D-1),   beta = -ep.")
print("4. Fourier transform: int d^dQ/(2pi)^d e^{iQx} (Q^2)^beta = c(beta)/(x^2)^s,")
print("   c(beta) = 4^beta Gamma(d/2+beta)/(pi^{d/2} Gamma(-beta)),  s = d/2 + beta  (= d - 2 at one loop).")
print("5. Extraction.  Restoring the full transverse-traceless structure, the coefficient of (z1.x)^2(z2.x)^2/(x^2)^{d+2}")
print("   equals 4 C_T for C_T (z1.I.z2)^2/x^{2d}, I = 1 - 2xx/x^2, which gives")
print("   (the factor (d-2)/(d-1) comes from the trace part of the transverse-traceless projector), and the coefficient")
print("   of (z1.x)(z2.x)/(x^2)^d equals -2 C_J for C_J (z1.I.z2)/x^{2d-2}, with Q_mu Q_nu -> -d_mu d_nu:")
print("       C_T = 4 (d-2)/(d-1) s(s+1)(s+2)(s+3) c(beta) A_T,     C_J = -2 s(s+1) c(beta) A_J.")
print("   Classes are summed per flavour invariant first (X_inv = sum over classes of coefficient x X).\n")
AT = lambda Xv: ep * G11 * 2 * Xv / ((D - 1) * (D + 1) * (D - 2))
AJ = lambda Xv: ep * G11 * Xv / (D - 1)
CT = lambda Xv: 4 * (D - 2) / (D - 1) * s * (s + 1) * (s + 2) * (s + 3) * c_beta * AT(Xv)
CJ = lambda Xv: -2 * s * (s + 1) * c_beta * AJ(Xv)
Xinv = {}
for nm, Xc in X.items():
    for inv, co in pub[nm]["decomp"].items():
        Xinv[inv] = sp.expand(Xinv.get(inv, 0) + sp.Rational(co) * Xc)
print("   G-scheme sums per invariant:", {k: str(v) for k, v in Xinv.items()})
textbook = {"Tr1": ("C_T of free fermions per unit Tr 1", D / (2 * Sd ** 2), CT),
            "ns": ("C_T of one free real scalar", D / ((D - 1) * Sd ** 2), CT),
            "Tr(TT)": ("C_J of free fermions per unit Tr(TT)", -1 / Sd ** 2, CJ),
            "t.t": ("C_J of free scalars per unit t.t", -1 / ((D - 2) * Sd ** 2), CJ)}
ok = True
for name, (label, ref, fn) in textbook.items():
    for dv in (sp.Rational(3), sp.Rational(5, 2), sp.Rational(37, 10)):
        e0 = (4 - dv) / 2
        ours, tb = sp.N(fn(Xinv[name]).subs(ep, e0), 25), sp.N(ref.subs(ep, e0), 25)
        ok &= abs(ours - tb) < 1e-20 * abs(tb)
    print(f"   {label}: pipeline / textbook at d = 3, 5/2, 37/10 -> "
          + ", ".join(f"{sp.N((fn(Xinv[name]) / ref).subs(ep, (4 - dv) / 2), 15)}" for dv in (3, sp.Rational(5, 2), sp.Rational(37, 10))))
print("\nAt higher loops the same steps apply to each term (Q^2)^(-L ep); the ratio to the one-loop free value is")
print("ctcj.assemble.position_ratio, followed by MS renormalisation of the couplings and the fixed-point form (METHOD.md).")
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
