#!/usr/bin/env python3
"""The N=1 supercurrent S_mu and its Feynman rules in the repo's conventions,
with numerical checks in D = 3 (Pauli matrices) and D = 4 (Euclidean Dirac matrices).

    python3 scripts/supercurrent_rules.py        (< 1 s)

Conventions (ctcj/forcer.py): Euclidean, d = D, fields ~ exp(i p.x), action

    S_E = int  1/2 (d phi_a)^2 + psibar_a dslash psi_a + h_abc phi_a psibar_b psi_c + 1/2 W_a W_a,
    W = h_abc phi_a phi_b phi_c / 6,  W_a = dW/dphi_a = 1/2 h_abc phi_b phi_c,  W_ab = h_abc phi_c,

written Dirac-like (Majorana: 1/2 in front of both fermion terms, same Feynman rules, closed loop -> Tr 1 = 1 per
two-component Majorana, as in the rest of the repo).  So y^a_bc = h_abc, lambda_abcd = h_abe h_cde + 2 pairings.
Feynman rules: fermion propagator -i pslash/p^2 (p along the arrow), Yukawa -h, derivative of a field with momentum q
flowing INTO the vertex -> i q; operators enter with +1.

Supercurrent (classical EOM: dslash psi_a = -W_ab psi_b, psibar_a dslash<- = W_ab psibar_b,
d^2 phi_a = W_ab W_b + 1/2 h_abc psibar_b psi_c):

    S_mu    = (dslash phi_a + W_a) gamma_mu psi_a      + xi (gamma_mu dslash - d_mu)(phi_a psi_a)
    Sbar_nu = psibar_a gamma_nu (dslash phi_a - W_a)   + xi psibar_a phi_a (dslash<- gamma_nu - d<-_nu)
    xi = (D-2)/(D-1)  (1/2 in D = 3)

    d.S      = 1/2 h_abc psi_a (psibar_b psi_c)   -- zero in D = 3 by Fierz (2-component spinors), not in D = 4
    gamma.S  = (4-D) W_a psi_a                     -- classical breaking, like T_mu^mu ~ (4-D) lambda phi^4

Momentum-space vertices (fermion line open from Sbar(0) to S(x); spinor chain written against the arrow):
    S_D at x:    (i qslash) gamma_mu            q = scalar momentum into x
    S_W at x:    + h_abc gamma_mu               (scalar legs b, c; fermion a)
    Sbar_D at 0: gamma_nu (i qslash)            q = scalar momentum into 0
    Sbar_W at 0: - h_abc gamma_nu
    improvement terms: gamma_mu X or P_mu X (P = total momentum) -- annihilated by the projector below.

Projection of G_{mu nu} = <S_mu Sbar_nu>(Q):  X_S = Tr[ Qslash P_{nu mu} G_{mu nu} ] with
    P_{nu mu} = pi_{nu mu} - gT_nu gT_mu/(D-1),  pi = delta - Q Q/Q^2,  gT_mu = pi_{mu rho} gamma_rho.
P kills Q_mu, gamma_mu on either side, so it is blind to the improvement terms and to gamma.S, and Qslash P has the
same property.  It is a projector (P P = P) with Tr P = (D-2) tr 1; both checked below.
"""
import itertools
import numpy as np

rng = np.random.default_rng(1)
s1 = np.array([[0, 1], [1, 0]], complex)
s2 = np.array([[0, -1j], [1j, 0]], complex)
s3 = np.array([[1, 0], [0, -1]], complex)
Z2 = np.zeros((2, 2))
I2 = np.eye(2)


def gammas(D):
    if D == 3:
        return [s1, s2, s3]
    if D == 4:
        return [np.block([[Z2, -1j * s], [1j * s, Z2]]) for s in (s1, s2, s3)] + [np.block([[Z2, I2], [I2, Z2]])]
    raise ValueError(D)


def sl(g, p):
    return sum(pi * gi for pi, gi in zip(p, g))


def null_vec(D):
    """complex momentum with p.p = 0 (on-shell massless, Euclidean)"""
    a, b = rng.normal(size=D), rng.normal(size=D)
    b -= a * (a @ b) / (a @ a)
    b *= np.linalg.norm(a) / np.linalg.norm(b)
    return a + 1j * b


def kernel(M):
    u, s, vh = np.linalg.svd(M)
    return vh.conj()[-1]


def charge_conj(g):
    """antisymmetric C with C gamma_mu C^-1 = -gamma_mu^T (so psibar chi = psi^T C chi is symmetric)"""
    n = g[0].shape[0]
    rows = []
    for gm in g:                      # C gm + gm^T C = 0, linear in vec(C)
        rows.append(np.kron(np.eye(n), gm.T) + np.kron(gm.T, np.eye(n)))
    A = np.vstack(rows)
    _, s, vh = np.linalg.svd(A)
    C = vh.conj()[-1].reshape(n, n)
    assert s[-1] < 1e-10 and np.allclose(C, -C.T), "no antisymmetric C"
    return C


results = []


def check(name, ok):
    results.append((name, bool(ok)))
    print(f"  [{'ok' if ok else 'FAIL'}] {name}")


for D in (3, 4):
    print(f"D = {D}")
    g = gammas(D)
    n = g[0].shape[0]
    one = np.eye(n)
    check("Clifford algebra", all(np.allclose(a @ b + b @ a, 2 * (i == j) * one)
                                  for i, a in enumerate(g) for j, b in enumerate(g)))
    xi = (D - 2) / (D - 1)

    # ---------------------------------------------------------------- tree-level Ward identities, <0| S |phi phi psi>
    # incoming scalars p1 (flavour b), p2 (flavour c), incoming fermion k (flavour a, spinor u, kslash u = 0);
    # flavour factor h_abc common to all three diagrams and dropped.  q = p1 + p2 + k flows out through S.
    p1, p2, k = null_vec(D), null_vec(D), null_vec(D)
    q = p1 + p2 + k
    u = kernel(sl(g, k))

    def amp_S(s, imp=True):
        M = []
        for mu in range(D):
            gm = g[mu]
            m = s * gm @ u                                            # S_W contact
            for pi_, pj in ((p1, p2), (p2, p1)):                     # S_D: scalar pi at x, fermion -> Yukawa(pj)
                K = k + pj
                prop = -1j * sl(g, K) / (K @ K)
                v = 1j * sl(g, pi_) @ gm
                if imp:                                               # xi (gamma_mu i Pslash - i P_mu), P = pi + K = q
                    v = v + xi * (gm @ (1j * sl(g, q)) - 1j * q[mu] * one)
                m = m + v @ prop @ (-1 * u)                           # Yukawa -h
            M.append(m)
        return M

    for s in (+1, -1):
        M = amp_S(s)
        div = sum(q[mu] * M[mu] for mu in range(D))
        ok = np.allclose(div, 0, atol=1e-10)
        check(f"S:    q.<S|phi phi psi> = 0 with W-sign s = {s:+d}: {ok}", ok == (s == +1))
    M = amp_S(+1)
    trace = sum(g[mu] @ M[mu] for mu in range(D))
    check("S:    gamma.<S|phi phi psi> = (4-D) h u   (gamma-trace = (4-D) W psi)", np.allclose(trace, (4 - D) * u))
    M0 = amp_S(+1, imp=False)
    tr0 = sum(g[mu] @ M0[mu] for mu in range(D))
    check("      without improvement the gamma-trace is not (4-D) h u", not np.allclose(tr0, (4 - D) * u))
    # <0|S|phi psi>: (i pslash) gamma_mu u + improvement; conserved and gamma-traceless on shell
    p = null_vec(D)
    k2 = null_vec(D)
    u2 = kernel(sl(g, k2))
    P = p + k2
    M2 = [(1j * sl(g, p) @ g[mu] + xi * (g[mu] @ (1j * sl(g, P)) - 1j * P[mu] * one)) @ u2 for mu in range(D)]
    check("S:    <S|phi psi> conserved and gamma-traceless", np.allclose(sum(P[m] * M2[m] for m in range(D)), 0)
          and np.allclose(sum(g[m] @ M2[m] for m in range(D)), 0))

    # ---------------------------------------------------------------- the same for Sbar: <phi phi psi| Sbar |0>
    # outgoing scalars p1, p2 and fermion k (ubar kslash = 0); scalar momentum INTO vertex 0 is -p_i.
    ub = kernel(sl(g, k).T)                                           # row spinor: ub @ kslash = 0

    def amp_Sbar(sb):
        M = []
        for nu in range(D):
            gn = g[nu]
            m = sb * ub @ gn
            for pi_, pj in ((p1, p2), (p2, p1)):
                K = k + pj
                prop = -1j * sl(g, K) / (K @ K)
                v = gn @ (1j * sl(g, -pi_))
                m = m + (-1 * ub) @ prop @ v
            M.append(m)
        return M

    for sb in (+1, -1):
        M = amp_Sbar(sb)
        div = sum(q[nu] * M[nu] for nu in range(D))
        ok = np.allclose(div, 0, atol=1e-10)
        check(f"Sbar: q.<phi phi psi|Sbar> = 0 with W-sign {sb:+d}: {ok}", ok == (sb == -1))

    # ---------------------------------------------------------------- the remainder of d.S: h_abc psi_a (psibar_b psi_c)
    # Grassmann algebra: elements {sorted tuple of generators: coefficient}.
    C = charge_conj(g)

    def gmul(A, B):
        out = {}
        for ka, va in A.items():
            for kb, vb in B.items():
                if set(ka) & set(kb):
                    continue
                arr = list(ka + kb)
                sign = 1
                for i in range(len(arr)):                             # bubble sort for the sign
                    for j in range(len(arr) - 1 - i):
                        if arr[j] > arr[j + 1]:
                            arr[j], arr[j + 1] = arr[j + 1], arr[j]
                            sign = -sign
                key = tuple(arr)
                out[key] = out.get(key, 0) + sign * va * vb
        return {kk: v for kk, v in out.items() if abs(v) > 1e-12}

    def gadd(A, B, c=1):
        out = dict(A)
        for kk, v in B.items():
            out[kk] = out.get(kk, 0) + c * v
        return {kk: v for kk, v in out.items() if abs(v) > 1e-12}

    nf = 3
    psi = [[{(a * n + al,): 1.0} for al in range(n)] for a in range(nf)]

    def bil(a, b):                                                    # psi_a^T C psi_b
        out = {}
        for al in range(n):
            for be in range(n):
                if abs(C[al, be]) > 1e-12:
                    out = gadd(out, gmul(psi[a][al], psi[b][be]), C[al, be])
        return out

    check("psibar_a psi_b = psi^T C psi symmetric in a, b", all(
        not gadd(bil(a, b), bil(b, a), -1) for a in range(nf) for b in range(nf)))

    def remainder(h):
        E = []
        for al in range(n):
            e = {}
            for a, b, c in itertools.product(range(nf), repeat=3):
                if abs(h[a, b, c]) > 1e-12:
                    e = gadd(e, gmul(psi[a][al], bil(b, c)), h[a, b, c])
            E.append(e)
        return max((abs(v) for e in E for v in e.values()), default=0.0)

    hs = rng.normal(size=(nf, nf, nf))
    hs = sum(np.transpose(hs, pp) for pp in itertools.permutations(range(3))) / 6
    hn = rng.normal(size=(nf, nf, nf))
    hn = hn + hn.transpose(0, 2, 1)                                  # symmetric in b, c only
    r_sym, r_gen = remainder(hs), remainder(hn)
    if D == 3:
        check(f"d.S remainder h_abc psi_a(psibar_b psi_c) = 0 for symmetric h (max coeff {r_sym:.1e})", r_sym < 1e-12)
        check(f"      ... but not for h symmetric in (b,c) only (max coeff {r_gen:.2f})", r_gen > 1e-3)
    else:
        check(f"d.S remainder is NOT zero in D = 4 for symmetric h (max coeff {r_sym:.2f}): S not conserved in D = 4",
              r_sym > 1e-3)

    # ---------------------------------------------------------------- projector
    Q = rng.normal(size=D)
    Q /= np.linalg.norm(Q)
    pi = np.eye(D) - np.outer(Q, Q)
    gT = [sum(pi[m, r] * g[r] for r in range(D)) for m in range(D)]
    Pm = [[pi[nu, mu] * one - gT[nu] @ gT[mu] / (D - 1) for mu in range(D)] for nu in range(D)]
    Qs = sl(g, Q)
    check("P_{nu mu} Q_mu = Q_nu P_{nu mu} = 0", np.allclose(sum(Pm[0][m] * Q[m] for m in range(D)), 0)
          and all(np.allclose(sum(Q[nu] * Pm[nu][m] for nu in range(D)), 0) for m in range(D)))
    check("P_{nu mu} gamma_mu = gamma_nu P_{nu mu} = 0 (also with Qslash in front)",
          all(np.allclose(sum(Pm[nu][m] @ g[m] for m in range(D)), 0) for nu in range(D))
          and all(np.allclose(sum(g[nu] @ Qs @ Pm[nu][m] for nu in range(D)), 0) for m in range(D)))
    PP = [[sum(Pm[nu][r] @ Pm[r][mu] for r in range(D)) for mu in range(D)] for nu in range(D)]
    check("P P = P", all(np.allclose(PP[a][b], Pm[a][b]) for a in range(D) for b in range(D)))
    trP = sum(np.trace(Pm[m][m]) for m in range(D)).real / n
    check(f"Tr P / tr 1 = (D-1) - 1 = D-2 = {trP:g}  (spin-3/2 states: (D-2) x spinor)", np.isclose(trP, D - 2))

# ---------------------------------------------------------------- flavour factors: einsum vs explicit index sums
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
from ctcj.supercurrent import generate_S, generate_S_fast, flavour, flavour_loops
print("flavour factors")
n = 3
hs = rng.normal(size=(n,) * 3)
hs = sum(np.transpose(hs, pp) for pp in itertools.permutations(range(3))) / 6
yg = rng.normal(size=(n, n, n))
yg = yg + yg.transpose(0, 2, 1)                                       # general Yukawa, not supersymmetric
lam = rng.normal(size=(n,) * 4)
lam = sum(np.transpose(lam, pp) for pp in itertools.permutations(range(4))) / 24
tt = rng.normal(size=(n, n))
tt = tt - tt.T
cls = []
for w in (0, 2, 4):
    for ops in [("D", "D"), ("W", "D"), ("D", "W"), ("W", "W"), ("j", "j")]:
        nW = ops.count("W")
        for nQ in range(0, (w - nW) // 2 + 1):
            if w - nW - 2 * nQ >= 0:
                cls += generate_S(w - nW - 2 * nQ, nQ, ops, "not_scaleless")
cls += [c for o in [(6, 0), (4, 1), (2, 2)] for c in generate_S_fast(*o, ("D", "D"), reps=True)[:15]]
cls += [c for o in [(6, 0), (4, 1), (2, 2), (0, 3)] for c in generate_S_fast(*o, ("j", "j"), "not_scaleless", reps=True)[:10]]
dev = max(abs(flavour(c, hh, lam, mult=0.7, t=tt) - flavour_loops(c, hh, lam, mult=0.7, t=tt))
          / max(1.0, abs(flavour_loops(c, hh, lam, mult=0.7, t=tt))) for c in cls for hh in (hs, yg))
check(f"tensor-contraction flavour factor = explicit index sums ({len(cls)} classes, D/W/j ends, quartics, loops)", dev < 1e-12)

print()
bad = [r for r, ok in results if not ok]
print("ALL CHECKS PASS" if not bad else f"FAILED: {bad}")
raise SystemExit(1 if bad else 0)
