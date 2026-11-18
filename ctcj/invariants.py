"""Flavour (coupling-tensor) invariants of diagram classes and their decomposition on a basis.

Each diagram's flavour factor is fixed by its graph: fermion loops give traces (matrices ordered
along the spinor chain, i.e. against the arrows), scalar lines contract indices, Tsig gives
delta_ab, Jphi gives t_ab (first index = derivative leg), Jpsi inserts T, quartic gives lambda.

The flavour factor of every class is decomposed numerically on a basis of invariants:
  * <TT>: random real symmetric y^a and symmetric lambda (no constraints needed).
  * <JJ>: random theories with an exact symmetry (U(1) and SU(2) examples), so that the Ward
          identity  i[y^a, T] = t_ab y^b  and  t.lambda = 0  hold; this eliminates t wherever it
          touches a Yukawa vertex.  The independent basis below is complete modulo the relations
          listed in J_RELATIONS (all consequences of the symmetry).
"""
import itertools
from collections import defaultdict

import numpy as np
import sympy as sp

from .diagrams import loops_of

RNG = np.random.default_rng(20261027)


# ------------------------------------------------------------------ class invariants
def class_invariant(c, th):
    y, lam = th["y"], th.get("lam")
    T, t = th.get("T"), th.get("t")
    ns, nf = y.shape[0], y.shape[1]
    sig = list(c["sig"])
    leg = defaultdict(dict)
    for e, (a, b) in enumerate(sig):
        for (v, tg) in (a, b):
            d = leg[v]
            d[tg if tg >= 0 else len(d)] = e
    ops = dict(zip(("x", "0"), c["ops"]))
    loops = loops_of(list(c["ferm"]))
    pred = {b: a for a, b in c["ferm"]}
    total = 0j
    for idx in itertools.product(range(ns), repeat=len(sig)):
        val = 1 + 0j
        for v, d in leg.items():
            if v.startswith("Q"):
                val *= lam[tuple(idx[e] for e in d.values())]
            elif ops.get(v) == "Jphi":
                val *= t[idx[d[0]], idx[d[1]]]
            elif ops.get(v) == "Tsig":
                val *= float(idx[d[0]] == idx[d[1]])
        if val == 0:
            continue
        for L in loops:
            M = np.eye(nf, dtype=complex)
            v = L[0]
            while True:
                if v.startswith("Y"):
                    M = M @ y[idx[leg[v][0]]]
                elif ops.get(v) == "Jpsi":
                    M = M @ T
                v = pred[v]
                if v == L[0]:
                    break
            val *= np.trace(M)
        total += val
    return total


# ------------------------------------------------------------------ generic theories (for <TT>)
def random_generic(ns=3, nf=4):
    y = RNG.normal(size=(ns, nf, nf))
    y = y + y.transpose(0, 2, 1)
    lam = RNG.normal(size=(ns,) * 4)
    lam = sum(np.transpose(lam, p) for p in itertools.permutations(range(4)))
    return dict(y=y, lam=lam)


# ------------------------------------------------------------------ symmetric theories (for <JJ>)
def _herm_basis(n):
    B = []
    for i in range(n):
        M = np.zeros((n, n), complex); M[i, i] = 1; B.append(M)
    for i in range(n):
        for j in range(i + 1, n):
            M = np.zeros((n, n), complex); M[i, j] = M[j, i] = 1; B.append(M)
            M = np.zeros((n, n), complex); M[i, j] = -1j; M[j, i] = 1j; B.append(M)
    return B


def _nullspace(A, tol=1e-10):
    _, s, vh = np.linalg.svd(A)
    r = int((s > tol * max(1.0, s.max())).sum())
    return vh[r:].conj().T


def _invariant_y(Ts, ts):
    nf, ns = Ts[0].shape[0], ts[0].shape[0]
    HB = _herm_basis(nf)
    blocks = []
    for T, t in zip(Ts, ts):
        cols = []
        for a in range(ns):
            for H in HB:
                ya = [np.zeros((nf, nf), complex) for _ in range(ns)]
                ya[a] = H
                out = [1j * (ya[b] @ T - T @ ya[b]) - sum(t[b, cc] * ya[cc] for cc in range(ns)) for b in range(ns)]
                cols.append(np.concatenate([np.concatenate([o.real.ravel(), o.imag.ravel()]) for o in out]))
        blocks.append(np.array(cols).T)
    return _nullspace(np.vstack(blocks)), HB


def _invariant_lambda(ts, ns):
    idx = list(itertools.product(range(ns), repeat=4))
    pos = {k: i for i, k in enumerate(idx)}
    rows = []
    for k in idx:
        for p in itertools.permutations(range(4)):
            kp = tuple(k[i] for i in p)
            if kp != k:
                r = np.zeros(len(idx)); r[pos[k]] += 1; r[pos[kp]] -= 1; rows.append(r)
    for t in ts:
        for k in idx:
            r = np.zeros(len(idx))
            for leg in range(4):
                for e in range(ns):
                    kk = list(k); kk[leg] = e
                    r[pos[tuple(kk)]] += t[k[leg], e]
            rows.append(r)
    return _nullspace(np.array(rows))


def _u1():
    q, Q = [0, 1, 1, -1, 2], [1, 2]
    ns = 2 * len(Q) + 1
    t = np.zeros((ns, ns))
    for j, Qj in enumerate(Q):
        t[2 * j:2 * j + 2, 2 * j:2 * j + 2] = Qj * np.array([[0, -1], [1, 0]])
    return [np.diag(q).astype(complex)], [t]


def _su2():
    s = [np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.array([[1, 0], [0, -1]])]
    eps = np.zeros((3, 3, 3))
    for a, b, c in itertools.permutations(range(3)):
        eps[a, b, c] = np.linalg.det(np.eye(3)[[a, b, c]])
    Ts = [np.block([[np.kron(s[k] / 2, np.eye(2)), np.zeros((4, 3))], [np.zeros((3, 4)), -1j * eps[k]]])
          for k in range(3)]
    ts = []
    for k in range(3):
        t = np.zeros((4, 4)); t[:3, :3] = eps[k]; ts.append(t)
    return Ts, ts


def symmetric_ensemble(nsamp=7):
    out = []
    for maker in (_u1, _su2):
        Ts, ts = maker()
        K, HB = _invariant_y(Ts, ts)
        KL = _invariant_lambda(ts, ts[0].shape[0])
        ns = ts[0].shape[0]
        for _ in range(nsamp):
            cvec = (K @ RNG.normal(size=K.shape[1])).real.reshape(ns, len(HB))
            y = np.array([sum(cvec[a, r] * HB[r] for r in range(len(HB))) for a in range(ns)])
            lam = (KL @ RNG.normal(size=KL.shape[1])).real.reshape((ns,) * 4)
            r = RNG.normal(size=len(Ts))
            T = sum(ri * Ti for ri, Ti in zip(r, Ts))
            t = sum(ri * ti for ri, ti in zip(r, ts))
            for a in range(ns):   # Ward identity
                assert np.allclose(1j * (y[a] @ T - T @ y[a]), sum(t[a, b] * y[b] for b in range(ns)))
            out.append(dict(y=y, lam=lam, T=T, t=t))
    return out


# ------------------------------------------------------------------ word traces
def tr_words(words, th):
    """Sum over repeated labels of prod_k Tr(word_k); letters 'T' or a label (a, b, ...)."""
    y, T = th["y"], th.get("T")
    labs = sorted(set("".join(words)) - {"T"})
    tot = 0j
    for vals in itertools.product(range(y.shape[0]), repeat=len(labs)):
        m = dict(zip(labs, vals))
        v = 1
        for w in words:
            M = np.eye(y.shape[1], dtype=complex)
            for ch in w:
                M = M @ (T if ch == "T" else y[m[ch]])
            v *= np.trace(M)
        tot += v
    return tot


# ------------------------------------------------------------------ bases
# keys of a basis are the names used throughout (and in the paper)
T_BASIS = {
    0: {"ns": lambda th: th["y"].shape[0], "Tr1": lambda th: th["y"].shape[1]},
    2: {"Tr(aa)": lambda th: tr_words(["aa"], th)},
    4: {"Tr(ab)Tr(ab)": lambda th: tr_words(["ab", "ab"], th),
        "Tr(aabb)": lambda th: tr_words(["aabb"], th),
        "Tr(abab)": lambda th: tr_words(["abab"], th),
        "Tr(aa)Tr(bb)": lambda th: tr_words(["aa", "bb"], th)},
    -2: {"lam.lam": lambda th: np.einsum("abcd,abcd->", th["lam"], th["lam"]),
         "lam_aacd lam_bbcd": lambda th: np.einsum("aacd,bbcd->", th["lam"], th["lam"])},
}
J_WORDS4 = ["TTaabb", "TTabab", "TTabba", "TaTabb", "TaTbab", "TabTba"]
J_BASIS = {
    0: {"Tr(TT)": lambda th: np.trace(th["T"] @ th["T"]), "t.t": lambda th: (th["t"] ** 2).sum()},
    2: {"Tr(TTaa)": lambda th: tr_words(["TTaa"], th), "Tr(TaTa)": lambda th: tr_words(["TaTa"], th)},
    4: {**{f"Tr({w})": (lambda w: lambda th: tr_words([w], th))(w) for w in J_WORDS4},
        "Tr(TTab)Tr(ab)": lambda th: tr_words(["TTab", "ab"], th),
        "Tr(TaTb)Tr(ab)": lambda th: tr_words(["TaTb", "ab"], th),
        "Tr(Tab)Tr(Tab)": lambda th: tr_words(["Tab", "Tab"], th),
        "Tr(TTaa)Tr(bb)": lambda th: tr_words(["TTaa", "bb"], th)},
    -2: {"t t lam lam": lambda th: np.einsum("ae,bf,abcd,efcd->", th["t"], th["t"], th["lam"], th["lam"])},
}
# relations used to reduce the J basis (verified numerically by checks.py)
J_RELATIONS = {
    "Tr(TaTbba)": {"Tr(TaTabb)": 1},
    "Tr(TaaTbb)": {"Tr(TTaabb)": 1},
    "Tr(TabTab)": {"Tr(TTabab)": -1, "Tr(TaTbab)": 2},
    "Tr(Tab)Tr(Tba)": {"Tr(TTab)Tr(ab)": 1, "Tr(TaTb)Tr(ab)": -1, "Tr(Tab)Tr(Tab)": 1},
    "t_ae t_be lam_acdf lam_bcdf": {"t t lam lam": 3},
}


def order_of(c):
    return c["nY"] if c["nQ"] == 0 else (-2 if c["nY"] == 0 else None)


def rational(z, tol=1e-7):
    z = complex(z)
    return sp.nsimplify(round(z.real, 9), tolerance=tol) + sp.I * sp.nsimplify(round(z.imag, 9), tolerance=tol)


def decompose(c, theories, basis):
    """Express the flavour factor of class c on the basis (exact rationals)."""
    names = list(basis)
    B = np.array([[basis[n](th) for n in names] for th in theories])
    v = np.array([class_invariant(c, th) for th in theories])
    sol = np.linalg.lstsq(B, v, rcond=None)[0]
    if not np.allclose(B @ sol, v, atol=1e-8 * max(1.0, abs(v).max())):
        raise ValueError(f"class {c['name']} not expressible in basis")
    if np.linalg.matrix_rank(B, tol=1e-8 * abs(B).max()) < len(names):
        raise ValueError("basis degenerate on ensemble")
    return {n: rational(s) for n, s in zip(names, sol) if abs(s) > 1e-9}
