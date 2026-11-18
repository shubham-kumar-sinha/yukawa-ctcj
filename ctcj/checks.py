"""Cross-checks.  Each check returns (name, passed, detail)."""
import numpy as np
import sympy as sp

from . import models as M
from .epsseries import EP as e
from .invariants import symmetric_ensemble, tr_words, J_RELATIONS

eps, N = M.eps, M.N


def _eq(a, b):
    return sp.simplify(sp.expand(a - b)) == 0


def check_counts(recT, recJ):
    cT = [sum(1 for r in recT if (r["nY"], r["nQ"]) == o) for o in [(0, 0), (2, 0), (4, 0), (0, 2)]]
    cJ = [sum(1 for r in recJ if (r["nY"], r["nQ"]) == o) for o in [(0, 0), (2, 0), (4, 0), (0, 2)]]
    return ("diagram classes T (1,2,3-loop Yukawa, lambda^2) = 2,6,43,2;  J = 3,11,76,6",
            cT == [2, 6, 43, 2] and cJ == [3, 11, 76, 6], f"T {cT}, J {cJ}")


def check_free(resT, resJ):
    d = sp.Symbol("d")
    ok = (_eq(resT["free"]["Tr1"], (d - 1) / 2) and _eq(resT["free"]["ns"], 1)
          and _eq(resJ["free"]["Tr(TT)"], 1) and _eq(resJ["free"]["t.t"], 1 / (d - 2)))
    return ("free values: C_T fermion/scalar = (d-1)/2 per Tr1; C_J scalar/fermion = 1/(d-2)", ok,
            f"{resT['free']}, {resJ['free']}")


def check_two_loop_table(recT):
    ref = {("Tpsi", "Tpsi", "self"): (-sp.Rational(3, 32), -sp.Rational(17, 640), -sp.Rational(3239, 12800)),
           ("Tpsi", "Tpsi", "vertex"): (sp.Rational(1, 16), -sp.Rational(53, 960), sp.Rational(1259, 19200)),
           ("Tpsi", "Tsig"): (sp.Rational(1, 8), sp.Rational(9, 160), sp.Rational(393, 3200)),
           ("Tsig", "Tsig"): (-sp.Rational(1, 8), -sp.Rational(77, 480), -sp.Rational(1593, 3200))}
    got = []
    for r in recT:
        if (r["nY"], r["nQ"]) != (2, 0):
            continue
        s = sp.series(r["X"], e, 0, 1).removeO()
        got.append((r["ops"], tuple(s.coeff(e, k) for k in (-2, -1, 0))))
    ok = all(v in [g for _, g in got] for v in ref.values()) and len(got) == 6
    return ("two-loop <TT> diagrams reproduce the project table", ok, str([g for _, g in got]))


def check_largeN_family_sum(resT):
    s = sp.series(resT["totals"]["Tr(ab)Tr(ab)"], e, 0, 0).removeO()
    v = tuple(s.coeff(e, k) for k in (-3, -2, -1))
    return ("11 large-N diagrams sum to (0, 5/96, 1/2)", v == (0, sp.Rational(5, 96), sp.Rational(1, 2)), str(v))


def check_poles_logs(res, kind):
    ok = all(all(p == 0 for p in v) for v in res["poles"].values()) and res["logs_ok"]
    return (f"<{kind}{kind}>: every pole cancels per invariant; log(mu^2 x^2) drops at fixed points", ok, "")


def check_On(resT, resJ):
    n = sp.Symbol("n", positive=True)
    lstar = eps / (n + 8)
    # lambda_abcd = l S_abcd:  S.S = 3n(n+2);  t t S S = (n+2) l^2 t.t (t = O(n) generator)
    cT = sp.simplify(resT["coef"]["lam.lam"] * 3 * n * (n + 2) * lstar ** 2 / n)
    cJ = sp.simplify(resJ["coef"]["t t lam lam"] * (n + 2) * lstar ** 2 * 2)   # / (t.t/(d-2)) at d=4
    okT = _eq(cT, -5 * (n + 2) * eps ** 2 / (12 * (n + 8) ** 2))
    okJ = _eq(cJ, -3 * (n + 2) * eps ** 2 / (4 * (n + 8) ** 2))
    return ("O(n) Wilson-Fisher: C_T, C_J eps^2 terms (Osborn-Stergiou / Petkou)", okT and okJ, f"{cT}, {cJ}")


def check_ward_relations():
    ths = symmetric_ensemble(4)
    ok = True
    def w(name, th):
        if name.startswith("t"):
            if name == "t t lam lam":
                return np.einsum("ae,bf,abcd,efcd->", th["t"], th["t"], th["lam"], th["lam"])
            return np.einsum("ae,be,acdf,bcdf->", th["t"], th["t"], th["lam"], th["lam"])
        parts = [p.rstrip(")") for p in name.split("Tr(") if p]
        return tr_words(parts, th)
    for lhs, rhs in J_RELATIONS.items():
        for th in ths:
            ok &= np.isclose(w(lhs, th), sum(c * w(k, th) for k, c in rhs.items()))
    return ("Ward-identity relations among <JJ> invariants hold on U(1) and SU(2) ensembles", bool(ok), "")


def check_gny_beta():
    (_, (B1, B2, BL)) = M.fixed_point(*[M.MODELS["chiral Ising (GNY)"][i] for i in (0, 3)])
    y2, lam = M.y2, M.lam
    # FGKT (1607.05316) eq. (2.1) with g2 = 16 pi^2 * 3 lambda (S-normalised lambda), y = g1/(4 pi)
    ref = 3 * lam ** 2 / 4 - 6 * lam * y2 - (3 * N + sp.Rational(9, 4)) * y2 ** 2
    return ("GNY two-loop beta_y from JOS tensors = FGKT", _eq(B2, ref), str(sp.expand(B2)))


def model_results(resT, resJ):
    out = {}
    for name, (ts, A, t, mN) in M.MODELS.items():
        fp, _ = M.fixed_point(ts, mN)
        rT = M.evaluate("T", resT, M.T_invariants(ts, mN), fp)
        rJ = M.evaluate("J", resJ, M.J_invariants(ts, A, t, mN), fp)
        out[name] = (sp.simplify(rT.coeff(eps, 1)), sp.simplify(rT.coeff(eps, 2)),
                     sp.simplify(rJ.coeff(eps, 1)), sp.simplify(rJ.coeff(eps, 2)))
    return out


def check_gny_order_eps(models):
    a1, _, b1, _ = models["chiral Ising (GNY)"]
    ok = _eq(a1, -5 * N / (2 * (N + 6) * (3 * N + 2))) and _eq(b1, -sp.Rational(3, 2) / (N + 6))
    return ("GNY O(eps): C_T (FGKT eq. 2.9) and C_J", ok, f"{a1}, {b1}")


def check_largeN(models):
    a1, a2, b1, b2 = models["chiral Ising (GNY)"]
    d = 4 - eps
    # C_T/C_T,free = 1 + (C_T1 - 2/(d-1))/N + ... ;  C_J/C_J,free = 1 + C_J1/N
    cT = sp.limit(N * (a1 * eps + a2 * eps ** 2), N, sp.oo) + sp.series(2 / (d - 1), eps, 0, 3).removeO()
    cJ = sp.limit(N * (b1 * eps + b2 * eps ** 2), N, sp.oo)
    eta1 = sp.gamma(d - 1) * (d - 2) ** 2 * (eps / 2) / (4 * sp.gamma(1 + eps / 2) * sp.gamma(d / 2 + 1) * sp.gamma(d / 2) ** 2)
    CGN = (sp.polygamma(0, 1 + eps / 2) - 2 / eps) + sp.polygamma(0, d - 1) - sp.polygamma(0, 1) - sp.polygamma(0, d / 2)
    CT1 = sp.series(-4 * eta1 * (CGN / (d + 2) + (d - 2) / (d * (d + 2) * (d - 1))), eps, 0, 3).removeO()
    CJ1 = sp.series(-8 * (d - 1) / (d * (d - 2)) * eta1, eps, 0, 3).removeO()
    ok = _eq(cT, CT1) and _eq(cJ, CJ1)
    return ("large N: C_T1, C_J1 = exact Gross-Neveu results of Diab et al. (1601.07198)", ok,
            f"C_T1 = {sp.expand(cT)}, C_J1 = {sp.expand(cJ)}")


def check_largeN_scaling(models):
    out = []
    for name, ns in (("chiral Ising (GNY)", 1), ("chiral XY (NJLY)", 2), ("chiral Heisenberg", 3)):
        a1, a2, _, _ = models[name]
        out.append(sp.limit(N * a2, N, sp.oo) / ns)
    return ("large N: eps^2 coefficient of C_T scales with n_s", all(_eq(o, out[0]) for o in out), str(out))


def check_On_limit(models):
    ref = {"chiral Ising (GNY)": 1, "chiral XY (NJLY)": 2, "chiral Heisenberg": 3}
    ok, det = True, []
    for name, n in ref.items():
        a2 = models[name][1]
        v = sp.limit(a2, N, 0)
        ok &= _eq(v, -sp.Rational(5) * (n + 2) / (12 * (n + 8) ** 2))
        det.append(v)
    return ("N -> 0 of the three models = O(n) Wilson-Fisher C_T (n = 1, 2, 3)", ok, str(det))


def check_susy(resT, resJ):
    ts, A, t, _ = M.MODELS["chiral XY (NJLY)"]
    fp, _ = M.fixed_point(ts, 1)            # N = 2: one 3d two-component Dirac = N=2 Wess-Zumino point
    fp = {k: v.subs(N, 2) for k, v in fp.items()}
    d = 4 - eps
    rT = M.evaluate("T", resT, M.T_invariants(ts, 1), fp)
    # R-current: fermion charge 1/3, scalar charge 2/3 (A = s2/3, t = (2/3) eps_ab)
    W = M.J_invariants(ts, A / 3, t / 3, 1)
    dd = sp.Symbol("d")
    CJ = sum(resJ["free"][k].subs(dd, d) * W[k] for k in resJ["free"]) + sum(resJ["coef"][k] * W.get(k, 0) for k in resJ["coef"])
    Rphi, Rpsi = (d - 2) / (d - 1), 1 / (d - 1)
    CJfree = 2 * Rpsi ** 2 + 2 * Rphi ** 2 / (d - 2)        # free chiral multiplet, superconformal R
    rJ = sp.expand(sp.series(CJ.subs(fp) / CJfree, eps, 0, 3).removeO())
    return ("N=2 SUSY (chiral XY, N=2): C_J^R/C_J^R,free = C_T/C_T,free to O(eps^2)", _eq(rT, rJ),
            f"C_T: {rT};  C_J^R: {rJ}")
