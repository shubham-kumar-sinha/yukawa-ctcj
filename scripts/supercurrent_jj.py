#!/usr/bin/env python3
"""A supersymmetric test for the odd-loop term of C_J (ODD_LOOPS.md).

In 3d N=1 a flavour current J_mu (generator t_ab acting alike on phi and psi) sits in the spinor superfield
J_alpha = t_ab Phi_a D_alpha Phi_b, whose bottom component is j_alpha = t_ab phi_a psi_b (Delta = 3/2).  So at d = 3
    C_j/C_j,free = C_J/C_J,free,
and, like <S Sbar>, <j jbar> has no odd fermion loops through O(eps^3) (same legs as the S_D vertex;
scripts/count_supercurrent_odd.py).  j is not conserved, so graphs with a cut vertex do not drop out ("not_scaleless"
classes), and j may need its own renormalisation: poles and logs are checked, not assumed away.

    python3 scripts/supercurrent_compute.py --w 0 2 4 --jj --proj Pj                    (through three loops)
    python3 scripts/supercurrent_compute.py --w 6 --jj --fast --depth 3 --proj Pj        (four loops)
    python3 scripts/supercurrent_jj.py [--three] [--only O|SU]   (~2 min; writes results/supercurrent_jj.md)

Projection X = Tr[G Qslash]; position-space component xslash (polynomial s, ctcj.assemble._K kind "j").
Renormalisation and fixed-point evaluation as in scripts/supercurrent_eps3_general.py.
"""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import sympy as sp
from ctcj import epsseries as S
from ctcj.assemble import position_ratio
from ctcj.supercurrent import generate_S, generate_S_fast, flavour
from ctcj.theory import Theory
from ctcj import betasJ as BJ

ROOT = os.path.join(os.path.dirname(__file__), "..")
THREE = "--three" in sys.argv
e = S.EP
cache = json.load(open(os.path.join(ROOT, "data", "supercurrent_classes.json")))
SYM = {"ep": e, **{f"z{k}": sp.Symbol(f"z{k}") for k in range(2, 8)}}
NUM = {S.GAMMA_E: 0.5772156649015329, S.LOGPI: float(np.log(np.pi)), S.LOG4: float(np.log(4.0)),
       **{sp.Symbol(f"z{k}"): float(sp.zeta(k)) for k in range(2, 8)}}
X = lambda c: sp.sympify(cache[f"{c['name']}|Pj"]["X"], locals=SYM)
OPS, RULE = ("j", "j"), "not_scaleless"
FREE = generate_S(0, 0, OPS, RULE)[0]
XF = X(FREE)
C2 = generate_S(2, 0, OPS, RULE)
C3 = generate_S(4, 0, OPS, RULE) + generate_S(2, 1, OPS, RULE) + generate_S(0, 2, OPS, RULE)
C4 = [] if THREE else [c for nY, nQ in [(6, 0), (4, 1), (2, 2), (0, 3)]
                       for c in generate_S_fast(nY, nQ, OPS, RULE, reps=True)]
missing = [c["name"] for c in C4 if f"{c['name']}|Pj" not in cache]
assert not missing, f"{len(missing)} four-loop classes missing; run with --three or compute them"
R = {c["name"]: position_ratio("j", 2, X(c), XF) for c in C2}
R.update({c["name"]: position_ratio("j", 3, X(c), XF) for c in C3})
X4 = {c["name"]: X(c) for c in C4}


def betas(y, lam):
    yc = y.astype(complex)
    b1 = BJ.beta1_y(yc, 1.0).real
    b2 = BJ.beta2_y(yc, lam, 1.0).real
    bl = BJ.beta1_lam(yc, lam, 1.0).real
    f = lambda s: BJ.beta1_y(yc + s * b1, 1.0).real
    D2 = 0.5 * (8 * (f(1) - f(-1)) - (f(2) - f(-2))) / 12
    return b1, b2, bl, D2


def taylor(g, deg):
    s = np.linspace(-1.5, 1.5, deg + 3)
    return np.polynomial.polynomial.polyfit(s, [g(x) for x in s], deg + 2)


def terms(y, lam, t):
    b1, b2, bl, D2 = betas(y, lam)
    F = lambda c, yy, ll: flavour(c, yy, ll, t=t)
    out = {}
    for c in C2:
        n = c["name"]
        out[(n, "F")] = F(c, y, lam)
        k1 = taylor(lambda s: F(c, y + s * b1, lam), 2)
        out[(n, "b1")], out[(n, "b1b1")] = k1[1], 2 * k1[2]
        out[(n, "b2")] = taylor(lambda s: F(c, y + s * b2, lam), 2)[1]
        out[(n, "D2")] = taylor(lambda s: F(c, y + s * D2, lam), 2)[1]
    for c in C3:
        n = c["name"]
        out[(n, "F")] = F(c, y, lam)
        out[(n, "b1")] = taylor(lambda s: F(c, y + s * b1, lam + s * bl), 4)[1]
    return out


TWO = {c["name"] for c in C2}


def term_series(label):
    n, kind = label
    r = R[n]
    if n in TWO:
        return {"F": (r, 2), "b1": (r / (2 * e), 4), "b2": (r / (4 * e), 6), "D2": (r / (4 * e ** 2), 6),
                "b1b1": (r / (8 * e ** 2), 6)}[kind]
    return {"F": (r, 4), "b1": (r / (2 * e), 6)}[kind]


def renormalised_poles(y, lam, t):
    """pole parts of the renormalised ratio at generic couplings (y, lam) (not at a fixed point), weights 2, 4"""
    v = terms(y, lam, t)
    nfree = flavour(FREE, y, lam, t=t)
    out = {}
    for g in (2, 4) if THREE else (2, 4, 6):
        tot = sum(term_series(lb)[0] * val for lb, val in v.items() if term_series(lb)[1] == g)
        if g == 6 and C4:
            tot += position_ratio("j", 4, sp.expand(sum(sp.Float(flavour(c, y, lam, t=t), 16) * X4[c["name"]] for c in C4)), XF)
        tot = S.trunc(sp.expand(tot / nfree), 1)
        out[g] = [abs(complex(sp.N(sp.expand(tot.coeff(e, k)).subs(NUM).subs(S.ELL, 0.3)))) for k in (-3, -2, -1)]
    return out


def Cj_series(fp3, th, t):
    a, b, d = (th.y_of(fp3[k]).real for k in ("a", "b", "d"))
    c, ee = th.lam_of(fp3["c"]), th.lam_of(fp3["e"])
    nfree = flavour(FREE, a, c, t=t)
    Yf = lambda x: a + x * b + x * x * d
    Lf = lambda x: c + x * ee
    eps = sp.Symbol("eps")
    xs = np.linspace(-0.6, 0.6, 9)
    vals = [terms(Yf(x), Lf(x), t) for x in xs]
    total = 0
    order = 2 if THREE else 3
    for label in vals[0]:
        ser, g = term_series(label)
        need = order - g // 2
        if need < 0:
            continue
        coef = [vals[len(xs) // 2][label]] if need == 0 and g == 6 else \
            np.polynomial.polynomial.polyfit(xs, [v[label] for v in vals], 8)[:need + 1]
        P = sum(float(coef[k]) * eps ** k for k in range(need + 1))
        s = S.trunc(sp.expand(ser), need)
        total += sp.expand(s.subs(e, eps / 2) * P * eps ** (g // 2))
    if not THREE:
        X4tot = sp.expand(sum(sp.Float(flavour(cl, a, c, t=t), 16) * X4[cl["name"]] for cl in C4))
        total += sp.expand(S.trunc(position_ratio("j", 4, X4tot, XF), 0).subs(e, eps / 2) * eps ** 3)
    total = sp.expand(sp.series(sp.expand(total), eps, 0, order + 1).removeO()) / nfree
    out, logs = [], []
    for j in range(1, order + 1):
        cj = sp.expand(total.coeff(eps, j))
        logs.append(max(abs(float(sp.N(sp.diff(cj, s).subs(NUM).subs(S.ELL, 0.3)))) for s in (S.ELL, S.GAMMA_E, S.LOGPI)))
        out.append(float(sp.N(cj.subs({S.ELL: 0, S.GAMMA_E: 0, S.LOGPI: 0, S.LOG4: 0}).subs(NUM))))
    return out, logs


def susy_lambda(h):
    return np.einsum("abe,cde->abcd", h, h) + np.einsum("ace,bde->abcd", h, h) + np.einsum("ade,bce->abcd", h, h)


def t3(y, T):
    return np.einsum("ij,ajk,bkl,cli->abc", T, y, y, y)


exec(open(os.path.join(ROOT, "scripts", "liendo_rong.py")).read().split("MODELS = [")[0])   # o3_traceless
sys.path.insert(0, os.path.join(ROOT, "matrix"))
from run_models import setup as matrix_setup
_Y, _L1, _L2, _T, _t, _, _ = matrix_setup("SU", 3)            # SU(3) adjoint: Y real, totally symmetric; T = -i t
POINTS = [("O(3)/Z2 (N_Phi=5)", o3_traceless(), 7 / 30),
          ("SU(3) adjoint [susy]", ([np.real(_Y)], [_L1, _L2], [np.real(_t)]), None)]
if "--only" in sys.argv:
    POINTS = [p for p in POINTS if p[0].startswith(sys.argv[sys.argv.index("--only") + 1])]
from ctcj.supercurrent import has_odd_loop
OUT = ["# The fermionic partner of a flavor current, j = theta_ab phi_a psi_b, through O(eps^3): C_j = C_J", "",
       "Generated by `scripts/supercurrent_jj.py` from the integrals cached in `data/supercurrent_classes.json`; see",
       "ODD_LOOPS.md.  Paper: table 4 (lower part), section 5.4 and appendix F.", ""]


def say(line=""):
    print(line, flush=True)
    OUT.append(line)


say("## Diagram classes of <j jbar> (classes with a cut vertex kept)")
say("")
say("| loops | classes | zero | with an odd fermion loop | zero = odd-loop |")
say("|---|---|---|---|---|")
ok_all = True
for L, cl in [(1, [FREE]), (2, C2), (3, C3)] + ([] if THREE else [(4, C4)]):
    zero = {c["name"] for c in cl if X(c) == 0}
    odd = {c["name"] for c in cl if has_odd_loop(c)}
    ok_all &= zero == odd
    say(f"| {L} | {len(cl)} | {len(zero)} | {len(odd)} | {'yes' if zero == odd else '**no**'} |")
say("")
rows = []
for name, (Ys, Ls, gens), gref in POINTS:
    th = Theory(Y=[np.array(y, complex) for y in Ys], L=Ls, mult=1.0, name=name)
    best = None
    for fp in th.fixed_points(starts=80):
        h = th.y_of(fp["a"]).real
        if np.abs(h).max() < 1e-6:
            continue
        gam = 0.5 * np.einsum("aij,bji->ab", h, h)
        if np.allclose(gam, gam[0, 0] * np.eye(len(gam)), atol=1e-8) and (gref is None or abs(gam[0, 0] - gref) < 1e-6) \
                and np.abs(th.lam_of(fp["c"]) - susy_lambda(h)).max() < 1e-7:
            best = fp
            break
    assert best is not None, name
    A = gens[0].real
    cur = next((1j * A, s * A) for s in (1, -1) if th.check_current(1j * A, s * A, best))
    fp3 = th.third_order(best)
    a = th.y_of(fp3["a"]).real
    lam = th.lam_of(fp3["c"])
    # poles of the renormalised ratio along the SUSY line through the fixed point (y = s a, lambda = s^2 lam)
    pol = max(max(max(v) for v in renormalised_poles(s_ * a, s_ ** 2 * lam, A).values()) for s_ in (0.7, 1.3))
    ok_pol = pol < 1e-10
    cj, logs = Cj_series(fp3, th, A)
    cJ = th._series(fp3, "J", cur)
    ok_low = abs(cj[0] - cJ[1]) < 1e-10 and abs(cj[1] - cJ[2]) < 1e-10 and max(logs) < 1e-10
    ok_all &= ok_pol and ok_low
    row = dict(name=name, pol=ok_pol, low=ok_low, cj=cj, cJ=cJ)
    if not THREE:
        tt = t3(a, 1j * A)
        I0, I1 = np.einsum("abc,abc->", tt, tt).real, np.einsum("abc,bca->", tt, tt).real
        den = float(np.trace(A.T @ A)) + float((A ** 2).sum()) / 2          # C_J,free/C_J,psi = Tr(TT) + t.t/2
        z3 = float(sp.zeta(3))
        row.update(pred=(-3 * I0 + (2 - 6 * z3) * I1) / den, I0=I0 / den, I1=I1 / den)
        ok_all &= abs(cj[2] - cJ[3] - row["pred"]) < 1e-9
    rows.append(row)
say("## C_j/C_j,free versus C_J/C_J,free")
say("")
say("Poles: after coupling renormalization alone, along the supersymmetric line through the fixed point (y = s y*, lambda = s^2 lambda*,")
say("s = 0.7, 1.3), so that j needs no renormalization.  c3(J, even): C_J/C_J,free without the odd fermion loops.  The prediction is")
say("(-3 I0 + (2 - 6 zeta3) I1) / (C_J,free/C_J,psi), the same-line rule applied to the <JJ> diagrams (ODD_LOOPS.md).")
say("")
say("| fixed point (current) | poles cancel | equal at eps, eps^2; no logs | c1, c2 | " +
    ("c3(j) | c3(J, even) | difference | prediction |" if not THREE else ""))
say("|---|---|---|---|" + ("---|---|---|---|" if not THREE else ""))
for r in rows:
    line = (f"| {r['name']} | {'yes' if r['pol'] else '**no**'} | {'yes' if r['low'] else '**no**'} | "
            f"{r['cj'][0]:+.10f}, {r['cj'][1]:+.10f} |")
    if not THREE:
        line += f" {r['cj'][2]:+.10f} | {r['cJ'][3]:+.10f} | {r['cj'][2] - r['cJ'][3]:+.8f} | {r['pred']:+.8f} |"
    say(line)
say("")
say(f"C_j = C_J through O(eps^{2 if THREE else 3})" + ("" if THREE else " with Delta C_J/C_J,psi = -3 I0 + (2 - 6 zeta3) I1")
    + f": {'PASS' if ok_all else 'FAIL'}")
if not THREE and "--only" not in sys.argv:
    open(os.path.join(ROOT, "results", "supercurrent_jj.md"), "w").write("\n".join(OUT) + "\n")
sys.exit(0 if ok_all else 1)
