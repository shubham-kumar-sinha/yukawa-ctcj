"""Specialisation of the general formulae to models with y^a = y t^a (x) 1_m and
lambda_abcd = lambda (d_ab d_cd + d_ac d_bd + d_ad d_bc).

Fixed points use the general beta functions of Jack, Osborn, Steudtner (arXiv:2301.10903):
one loop (8.1) and two loops (2.1)-(2.3), (3.2)-(3.3), (4.1)-(4.3).  Couplings rescaled
y -> 4 pi y, lambda -> 16 pi^2 lambda.  N = Tr 1 = m * dim(t).
"""
import itertools

import sympy as sp

m, y2, lam, eps, N = sp.symbols("m y2 lambda epsilon N")
R = sp.Rational
s1 = sp.Matrix([[0, 1], [1, 0]])
s2 = sp.Matrix([[0, -sp.I], [sp.I, 0]])
s3 = sp.Matrix([[1, 0], [0, -1]])

# name: (Yukawa matrices t^a, fermion generator A, scalar generator t, m in terms of N)
MODELS = {
    "chiral Ising (GNY)": ([sp.Matrix([[1]])], sp.Matrix([[1]]), sp.zeros(1), N),
    "chiral XY (NJLY)": ([s3, s1], s2, sp.Matrix([[0, 2], [-2, 0]]), N / 2),
    "chiral Heisenberg": ([s1, s2, s3], s3 / 2, sp.Matrix([[0, 1, 0], [-1, 0, 0], [0, 0, 0]]), N / 2),
}


def _S(a, b, c, d):
    return (a == b) * (c == d) + (a == c) * (b == d) + (a == d) * (b == c)


def beta_functions(ts):
    """beta_y^a = (B1 + B2) y t^a (y2 = y^2), beta_lambda on the O(n_s) structure = BL."""
    ns, n = len(ts), ts[0].shape[0]
    I = sp.eye(n)
    tr = lambda M: m * sp.trace(M)
    P = lambda *ix: sp.prod([ts[i] for i in ix], start=I)
    L = lambda a, b, c, d: lam * _S(a, b, c, d)
    rg = range(ns)
    Z = sp.zeros(n)
    gphi1 = sp.Matrix(ns, ns, lambda a, b: R(1, 2) * tr(P(a, b)) * y2)
    gpsi1 = R(1, 2) * sum((P(a, a) for a in rg), Z) * y2
    gphi2 = sp.Matrix(ns, ns, lambda a, b: R(1, 12) * sum(L(a, c, d, e) * L(b, c, d, e) for c in rg for d in rg for e in rg)
                      + (-R(3, 4) * sum(tr(P(a, b, c, c)) for c in rg) - R(1, 2) * sum(tr(P(a, c, b, c)) for c in rg)) * y2 ** 2)
    gpsi2 = (-R(3, 8) * sum((P(a, b) * tr(P(a, b)) for a in rg for b in rg), Z)
             - R(1, 8) * sum((P(a, b, b, a) for a in rg for b in rg), Z)) * y2 ** 2
    B = []
    for a in rg:
        b1 = 2 * sum((P(b, a, b) for b in rg), Z) * y2 + gpsi1 * ts[a] + ts[a] * gpsi1 + sum((gphi1[a, b] * ts[b] for b in rg), Z)
        bt2 = (-2 * sum((L(a, b, c, d) * P(b, c, d) for b in rg for c in rg for d in rg), Z) * y2
               + (-sum((P(b, a, c) * tr(P(b, c)) for b in rg for c in rg), Z)
                  - sum((P(b, a, c, c, b) + P(b, c, c, a, b) for b in rg for c in rg), Z)
                  - 2 * sum((P(b, c, a, c, b) for b in rg for c in rg), Z)
                  + 2 * sum((P(b, c, a, b, c) for b in rg for c in rg), Z)) * y2 ** 2)
        b2 = bt2 + gpsi2 * ts[a] + ts[a] * gpsi2 + sum((gphi2[a, b] * ts[b] for b in rg), Z)
        B.append((b1, b2))
    k = [(i, j) for i in range(n) for j in range(n) if ts[0][i, j] != 0][0]
    B1 = sp.simplify(B[0][0][k] / ts[0][k])
    B2 = sp.simplify(B[0][1][k] / ts[0][k])
    for a in rg:
        assert sp.simplify(B[a][0] - B1 * ts[a]) == Z and sp.simplify(B[a][1] - B2 * ts[a]) == Z

    def bl(a, b, c, d):
        pairs = [((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))]
        t1 = sum(L(p[0], p[1], e, f) * L(q[0], q[1], e, f) for p, q in pairs for e in rg for f in rg)
        t2 = R(1, 2) * sum(L(e, *[x for j, x in enumerate((a, b, c, d)) if j != i]) * tr(P((a, b, c, d)[i], e))
                           for i in range(4) for e in rg) * y2
        t3 = -R(1, 2) * sum(tr(P(*pm)) for pm in itertools.permutations((a, b, c, d))) * y2 ** 2
        return t1 + t2 + t3
    BL = sp.simplify(bl(0, 0, 0, 0) / _S(0, 0, 0, 0))
    for q in itertools.product(rg, repeat=4):
        assert sp.simplify(bl(*q) - BL * _S(*q)) == 0
    return B1, B2, BL


def fixed_point(ts, mN, branch="+"):
    """(y^2, lambda) at the fixed point: y^2 to O(eps^2), lambda to O(eps)."""
    B1, B2, BL = [sp.simplify(x.subs(m, mN)) for x in beta_functions(ts)]
    b = sp.simplify(B1 / y2)
    u1 = R(1, 2) / b
    roots = sp.solve(sp.Eq((-lam + BL).subs(y2, u1), 0), lam)
    l1 = max(roots, key=lambda r: float(r.subs(N, 7))) if branch == "+" else min(roots, key=lambda r: float(r.subs(N, 7)))
    u2 = -sp.simplify(B2.subs({y2: u1, lam: l1})) / b
    return {y2: u1 * eps + u2 * eps ** 2, lam: l1 * eps}, (B1, B2, BL)


def _tr(ts, A, mm):
    n = ts[0].shape[0]
    def tr(*w):
        M = sp.eye(n)
        for ch in w:
            M = M * (A if ch == "T" else ts[ch])
        return mm * sp.trace(M)
    return tr


def T_invariants(ts, mm):
    tr = _tr(ts, None, mm); rg = range(len(ts))
    S = _S
    return {"ns": len(ts), "Tr1": tr(),
            "Tr(aa)": sum(tr(a, a) for a in rg) * y2,
            "Tr(ab)Tr(ab)": sum(tr(a, b) ** 2 for a in rg for b in rg) * y2 ** 2,
            "Tr(aabb)": sum(tr(a, a, b, b) for a in rg for b in rg) * y2 ** 2,
            "Tr(abab)": sum(tr(a, b, a, b) for a in rg for b in rg) * y2 ** 2,
            "Tr(aa)Tr(bb)": sum(tr(a, a) for a in rg) ** 2 * y2 ** 2,
            "lam.lam": lam ** 2 * sum(S(*q) ** 2 for q in itertools.product(rg, repeat=4)),
            "lam_aacd lam_bbcd": lam ** 2 * sum(S(a, a, c, d) * S(b, b, c, d) for a, b, c, d in itertools.product(rg, repeat=4))}


def J_invariants(ts, A, t, mm):
    tr = _tr(ts, A, mm); rg = range(len(ts))
    W = {"Tr(TT)": tr("T", "T"), "t.t": sum(t[a, b] ** 2 for a in rg for b in rg),
         "Tr(TTaa)": sum(tr("T", "T", a, a) for a in rg) * y2,
         "Tr(TaTa)": sum(tr("T", a, "T", a) for a in rg) * y2}
    for pat in ["TTaabb", "TTabab", "TTabba", "TaTabb", "TaTbab", "TabTba"]:
        W[f"Tr({pat})"] = sum(tr(*[{"a": a, "b": b, "T": "T"}[ch] for ch in pat]) for a in rg for b in rg) * y2 ** 2
    W["Tr(TTab)Tr(ab)"] = sum(tr("T", "T", a, b) * tr(a, b) for a in rg for b in rg) * y2 ** 2
    W["Tr(TaTb)Tr(ab)"] = sum(tr("T", a, "T", b) * tr(a, b) for a in rg for b in rg) * y2 ** 2
    W["Tr(Tab)Tr(Tab)"] = sum(tr("T", a, b) ** 2 for a in rg for b in rg) * y2 ** 2
    W["Tr(TTaa)Tr(bb)"] = sum(tr("T", "T", a, a) for a in rg) * sum(tr(b, b) for b in rg) * y2 ** 2
    W["t t lam lam"] = lam ** 2 * sum(t[a, e] * t[b, f] * _S(a, b, c, d) * _S(e, f, c, d)
                                      for a, b, c, d, e, f in itertools.product(rg, repeat=6))
    return W


def evaluate(kind, res, inv, fp):
    """Ratio to the free theory, expanded to O(eps^2)."""
    d = sp.Symbol("d")
    free = sum(res["free"][k].subs(d, 4 - eps) * inv[k] for k in res["free"])
    inter = sum(res["coef"][k] * inv.get(k, 0) for k in res["coef"])
    return sp.expand(sp.series((free + inter.subs(fp)) / free, eps, 0, 3).removeO())
