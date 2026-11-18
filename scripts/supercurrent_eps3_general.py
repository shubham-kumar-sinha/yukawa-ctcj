#!/usr/bin/env python3
"""C_S/C_S,free through O(eps^3) at every N=1 fixed point (any number of fields),
and kappa_T from each point with Tr(y^a y^b y^c)Tr(y^a y^b y^c) != 0.

    python3 scripts/supercurrent_eps3_general.py      (cached integrals; ~2 min; writes results/supercurrent_eps3.md)

Renormalisation as ctcj/assemble4.py, but with the flavour factors F_c(y, lambda) of each class kept as functions:
    y0 = y + b1/eps + b2/(2 eps) + D2/eps^2  (D2 = 1/2 Db1[b1]),   lambda0 = lambda + bl1/eps,   eps = 2 ep
    two-loop classes:   F + DF[b1]/(2ep) + DF[b2]/(4ep) + DF[D2]/(4ep^2) + D^2F[b1,b1]/(8ep^2)
    three-loop classes: F + DF[(b1, bl1)]/(2ep)
    four-loop classes:  F
(directional derivatives exact by polynomial fitting).  Every coupling polynomial P of grade g (y ~ 1, lambda ~ 2)
is evaluated at the fixed point as eps^(g/2) P(Y(eps), L(eps)), Y = a + eps b + eps^2 d, L = c + eps e
(Theory.third_order), with the regulator ep = eps/2.  Logs must cancel; C_T from Theory.CT3 (four-dimensional traces).
Then kappa_T = [C_S - C_T^(4d)]_(eps^3) x (C_T,free/C_T,S) / I3,  I3 = Tr(y^a y^b y^c)Tr(y^a y^b y^c) at a,
C_T,free/C_T,S = n_s + 3/2 Tr 1 = 5N/2.
"""
import itertools, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import sympy as sp
from ctcj import epsseries as S
from ctcj.assemble import position_ratio
from ctcj.supercurrent import generate_S, generate_S_fast, flavour
from ctcj.theory import Theory
from ctcj import betasJ as BJ

ROOT = os.path.join(os.path.dirname(__file__), "..")
e = S.EP
cache = json.load(open(os.path.join(ROOT, "data", "supercurrent_classes.json")))
SYM = {"ep": e, **{f"z{k}": sp.Symbol(f"z{k}") for k in range(2, 8)}}
NUM = {S.GAMMA_E: 0.5772156649015329, S.LOGPI: float(np.log(np.pi)), S.LOG4: float(np.log(4.0)),
       **{sp.Symbol(f"z{k}"): float(sp.zeta(k)) for k in range(2, 8)}}
X = lambda c: sp.sympify(cache[f"{c['name']}|P"]["X"], locals=SYM)
XF = X(generate_S(0, 0)[0])
C2 = generate_S(2, 0)
C3 = generate_S(4, 0) + generate_S(2, 1) + generate_S(0, 2)
C4 = [c for nY, nQ in [(6, 0), (4, 1), (2, 2), (0, 3)] for c in generate_S_fast(nY, nQ, reps=True)]
R = {c["name"]: position_ratio("S", 2, X(c), XF) for c in C2}
R.update({c["name"]: position_ratio("S", 3, X(c), XF) for c in C3})
X4 = {c["name"]: X(c) for c in C4}


# ------------------------------------------------------------------ couplings and beta functions (tensors)
def betas(y, lam):
    yc = y.astype(complex)
    b1 = BJ.beta1_y(yc, 1.0).real
    b2 = BJ.beta2_y(yc, lam, 1.0).real
    bl = BJ.beta1_lam(yc, lam, 1.0).real
    f = lambda s: BJ.beta1_y(yc + s * b1, 1.0).real           # cubic in s
    D2 = 0.5 * (8 * (f(1) - f(-1)) - (f(2) - f(-2))) / 12
    return b1, b2, bl, D2


def taylor(g, deg):
    """coefficients of the polynomial g(s) of degree <= deg"""
    s = np.linspace(-1.5, 1.5, deg + 3)
    return np.polynomial.polynomial.polyfit(s, [g(x) for x in s], deg + 2)


def F(c, y, lam):
    return flavour(c, y, lam)


def terms(y, lam):
    """{term label: value} of all coupling polynomials at (y, lam) (one evaluation of the renormalised sum)"""
    b1, b2, bl, D2 = betas(y, lam)
    out = {}
    for c in C2:
        n = c["name"]
        out[(n, "F")] = F(c, y, lam)
        k1 = taylor(lambda s: F(c, y + s * b1, lam), 2)
        out[(n, "b1")] = k1[1]
        out[(n, "b1b1")] = 2 * k1[2]
        out[(n, "b2")] = taylor(lambda s: F(c, y + s * b2, lam), 2)[1]
        out[(n, "D2")] = taylor(lambda s: F(c, y + s * D2, lam), 2)[1]
    for c in C3:
        n = c["name"]
        out[(n, "F")] = F(c, y, lam)
        out[(n, "b1")] = taylor(lambda s: F(c, y + s * b1, lam + s * bl), 4)[1]
    return out


# series and grade of each term (grade: y ~ 1, lambda ~ 2)
def term_series(label):
    n, kind = label
    r = R[n]
    two = n in {c["name"] for c in C2}
    if two:
        return {"F": (r, 2), "b1": (r / (2 * e), 4), "b2": (r / (4 * e), 6), "D2": (r / (4 * e ** 2), 6),
                "b1b1": (r / (8 * e ** 2), 6)}[kind]
    return {"F": (r, 4), "b1": (r / (2 * e), 6)}[kind]


def susy_lambda(h):
    return np.einsum("abe,cde->abcd", h, h) + np.einsum("ace,bde->abcd", h, h) + np.einsum("ade,bce->abcd", h, h)


def CS_series(fp3, th):
    a, b, d = (th.y_of(fp3[k]).real for k in ("a", "b", "d"))
    c, ee = th.lam_of(fp3["c"]), th.lam_of(fp3["e"])
    N = a.shape[0]
    Yf = lambda x: a + x * b + x * x * d
    Lf = lambda x: c + x * ee
    eps = sp.Symbol("eps")
    # P(Y(x), L(x)) as polynomials in x: grade g needs order x^(3 - g/2)
    xs = np.linspace(-0.6, 0.6, 9)
    vals = [terms(Yf(x), Lf(x)) for x in xs]
    total = 0
    for label in vals[0]:
        ser, g = term_series(label)
        need = 3 - g // 2
        if need == 0:                                  # grade 6: value at x = 0 (degree up to 12 in x)
            coef = [vals[len(xs) // 2][label]]
        else:                                          # grades 2, 4: degree <= 2 g <= 8, exact fit
            coef = np.polynomial.polynomial.polyfit(xs, [v[label] for v in vals], 8)[:need + 1]
        P = sum(float(coef[k]) * eps ** k for k in range(need + 1))
        s = S.trunc(sp.expand(ser), need)
        total += sp.expand(s.subs(e, eps / 2) * P * eps ** (g // 2))
    # four loops: leading order only
    X4tot = sp.expand(sum(sp.Float(F(cl, a, c), 16) * X4[cl["name"]] for cl in C4))
    r4 = S.trunc(position_ratio("S", 4, X4tot, XF), 0)
    total += sp.expand(r4.subs(e, eps / 2) * eps ** 3)
    total = sp.expand(sp.series(sp.expand(total), eps, 0, 4).removeO()) / N
    out, logs = [], []
    for j in range(1, 4):
        cj = sp.expand(total.coeff(eps, j))
        logs.append(max(abs(float(sp.N(sp.diff(cj, s).subs(NUM).subs(S.ELL, 0.3)))) for s in (S.ELL, S.GAMMA_E, S.LOGPI)))
        out.append(float(sp.N(cj.subs({S.ELL: 0, S.GAMMA_E: 0, S.LOGPI: 0, S.LOG4: 0}).subs(NUM))))
    tr3 = np.einsum("aij,bjk,cki->abc", a, a, a)
    I3 = float((tr3 * tr3).sum())
    return out, logs, I3, N


exec(open(os.path.join(ROOT, "scripts", "liendo_rong.py")).read().split("MODELS = [")[0])   # potts, o3_traceless
from fractions import Fraction
from ctcj.supercurrent import has_odd_loop
MODELS = [("N=1 super-Ising", ([np.ones((1, 1, 1))], [np.ones((1, 1, 1, 1))], None), 1 / 14),
          ("super-Potts S3 (N=2)", potts(3), 1 / 6), ("super-Potts S4", potts(4), 1 / 10),
          ("super-Potts S5", potts(5), 3 / 34), ("O(3)/Z2", o3_traceless(), 7 / 30)]
z3 = float(sp.zeta(3))
OUT = ["# The N=1 supercurrent through O(eps^3) at five N=1 fixed points: C_S = C_T and the odd-loop coefficient", "",
       "Generated by `scripts/supercurrent_eps3_general.py` from the integrals cached in `data/supercurrent_classes.json`;",
       "see ODD_LOOPS.md.  Paper: table 4 (upper part), section 5.4 and appendix F.", ""]


def say(line=""):
    print(line, flush=True)
    OUT.append(line)


# classes of <S Sbar> (D-D, biconnected): counts and the vanishing (odd-loop) ones
say("## Diagram classes of <S Sbar> contributing to C_S")
say("")
say("| loops | classes | zero | with an odd fermion loop | zero = odd-loop |")
say("|---|---|---|---|---|")
ok_cls = True
for L, cl in [(1, [generate_S(0, 0)[0]]), (2, C2), (3, C3), (4, C4)]:
    zero = {c["name"] for c in cl if X(c) == 0}
    odd = {c["name"] for c in cl if has_odd_loop(c)}
    ok_cls &= zero == odd
    say(f"| {L} | {len(cl)} | {len(zero)} | {len(odd)} | {'yes' if zero == odd else '**no**'} |")
say("")
say("## C_S/C_S,free versus C_T/C_T,free at O(eps^3)")
say("")
say("c3(T, even): C_T/C_T,free without the odd fermion loops (`Theory.CT3(fp)`).  I3 = Tr(y^a y^b y^c)Tr(y^a y^b y^c)/eps^3 at")
say("leading order; kappa_T = (c3(S) - c3(T, even)) (5/2) N_Phi / I3.")
say("")
say("| fixed point | equal at eps, eps^2; no logs | I3 | c1/I3 | c3(S) | c3(T, even) | kappa_T |")
say("|---|---|---|---|---|---|---|")
res, ok_pts = [], True
for name, (Ys, Ls, _), gref in MODELS:
    th = Theory(Y=[np.array(y, complex) for y in Ys], L=Ls, mult=1.0, name=name)
    best = None
    for fp in th.fixed_points(starts=80):
        h = th.y_of(fp["a"]).real
        if np.abs(h).max() < 1e-6:
            continue
        gam = 0.5 * np.einsum("aij,bji->ab", h, h)
        if not np.allclose(gam, gam[0, 0] * np.eye(len(gam)), atol=1e-8) or abs(gam[0, 0] - gref) > 1e-6:
            continue
        if np.abs(th.lam_of(fp["c"]) - susy_lambda(h)).max() < 1e-7:
            best = fp
            break
    assert best is not None, name
    fp3 = th.third_order(best)
    cS, logs, I3, N = CS_series(fp3, th)
    cT = th.CT3(best)
    low = abs(cS[0] - cT[1]) < 1e-12 and abs(cS[1] - cT[2]) < 1e-12 and max(logs) < 1e-12
    ok_pts &= low
    I3s = str(Fraction(I3).limit_denominator(100000)) if I3 > 1e-12 else "0"
    if I3 > 1e-12:
        kappa = (cS[2] - cT[3]) * 2.5 * N / I3
        res.append(kappa)
        say(f"| {name} | {'yes' if low else '**no**'} | {I3s} | {cT[1] / I3:+.2f} | {cS[2]:+.10f} | {cT[3]:+.10f} | {kappa:+.8f} |")
    else:
        same = abs(cS[2] - cT[3]) < 1e-12
        ok_pts &= same
        say(f"| {name} | {'yes' if low else '**no**'} | 0 | --- | {cS[2]:+.10f} | {cT[3]:+.10f} | (equal: {'yes' if same else '**no**'}) |")
say("")
spread = max(res) - min(res)
ok_kappa = spread < 1e-9 and abs(np.mean(res) - (7 / 6 - z3)) < 1e-9
say(f"7/6 - zeta3 = {7 / 6 - z3:+.8f}.  The four values of kappa_T agree with each other and with 7/6 - zeta3 to 1e-9: "
    f"{'yes' if ok_kappa else '**no**'}.")
say("")
ok_all = ok_cls and ok_pts and ok_kappa
say(f"C_S = C_T through O(eps^3) with Delta C_T/C_T,S = (7/6 - zeta3) Tr(yyy)Tr(yyy): {'PASS' if ok_all else 'FAIL'}")
open(os.path.join(ROOT, "results", "supercurrent_eps3.md"), "w").write("\n".join(OUT) + "\n")
sys.exit(0 if ok_all else 1)
