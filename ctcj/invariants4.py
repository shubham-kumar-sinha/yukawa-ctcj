"""Invariant bases for <TT> through four loops (orders y^6, y^4 lam, y^2 lam^2, lam^3) and the
beta-function insertions needed for renormalisation and for the fixed-point reduction."""
import itertools
import numpy as np
from .invariants import random_generic, rational

LAB = "abc"


def _canon_word(w):
    """canonical form of a cyclic word up to rotation, reflection and relabelling of labels"""
    best = None
    labs = sorted(set(w))
    for perm in itertools.permutations(labs):
        m = dict(zip(labs, perm))
        s = "".join(m[c] for c in w)
        for t in (s, s[::-1]):
            for k in range(len(t)):
                r = t[k:] + t[:k]
                if best is None or r < best:
                    best = r
    return best


def _canon_product(ws):
    """canonical form of a product of traces (relabelling acts on all traces simultaneously)"""
    labs = sorted(set("".join(ws)))
    best = None
    for perm in itertools.permutations(labs):
        m = dict(zip(labs, perm))
        parts = []
        for w in ws:
            s = "".join(m[c] for c in w)
            cands = [t[k:] + t[:k] for t in (s, s[::-1]) for k in range(len(t))]
            parts.append(min(cands))
        key = tuple(sorted(parts))
        if best is None or key < best:
            best = key
    return best


def y6_words():
    letters = "aabbcc"
    out = set()
    for p in set(itertools.permutations(letters)):
        s = "".join(p)
        for cut in [(6,), (4, 2), (2, 4), (2, 2, 2)]:
            ws, i = [], 0
            for c in cut:
                ws.append(s[i:i + c]); i += c
            out.add(_canon_product(ws))
    return sorted(out, key=lambda k: (len(k), k))


def tr_prod(ws, th):
    y = th["y"]
    labs = sorted(set("".join(ws)))
    tot = 0.0
    for vals in itertools.product(range(y.shape[0]), repeat=len(labs)):
        m = dict(zip(labs, vals))
        v = 1.0
        for w in ws:
            M = np.eye(y.shape[1])
            for ch in w:
                M = M @ y[m[ch]]
            v *= np.trace(M)
        tot += v
    return tot


def _lam_terms():
    E = np.einsum
    def H(th):
        y = th["y"]; return E("aij,bji->ab", y, y)
    def Q4(th):
        y = th["y"]; return E("aij,bjk,ckl,dli->abcd", y, y, y, y)
    return {
        # y^4 lambda
        "lam Tr(abcd)": lambda th: E("abcd,abcd->", th["lam"], Q4(th)),
        "lam Tr(ab)Tr(cd)": lambda th: E("abcd,ab,cd->", th["lam"], H(th), H(th)),
        "lam_abcc Tr(abdd)": lambda th: E("abcc,abdd->", th["lam"], Q4(th)),
        "lam_abcc Tr(adbd)": lambda th: E("abcc,adbd->", th["lam"], Q4(th)),
        "lam_abcc Tr(ab)Tr(dd)": lambda th: E("abcc,ab,dd->", th["lam"], H(th), H(th)),
        # y^2 lambda^2
        "lam_acde lam_bcde Tr(ab)": lambda th: E("acde,bcde,ab->", th["lam"], th["lam"], H(th)),
        "lam_acdd lam_bcee Tr(ab)": lambda th: E("acdd,bcee,ab->", th["lam"], th["lam"], H(th)),
        "lam_abcd lam_cdee Tr(ab)": lambda th: E("abcd,cdee,ab->", th["lam"], th["lam"], H(th)),
        # lambda^3
        "lam_abcd lam_abef lam_cdef": lambda th: E("abcd,abef,cdef->", th["lam"], th["lam"], th["lam"]),
        "lam_abcd lam_abce lam_dfff": lambda th: E("abcd,abce,dfff->", th["lam"], th["lam"], th["lam"]),
    }


def T_basis4():
    B = {}
    for ws in y6_words():
        B["Tr(" + ")Tr(".join(ws) + ")"] = (lambda ws: lambda th: tr_prod(ws, th))(ws)
    B.update(_lam_terms())
    return B


def theories(n=60):
    return [random_generic(2 + i % 3, 3 + (i // 3) % 2) for i in range(n)]


def independent(basis, ths, tol=1e-8):
    names = list(basis)
    M = np.real(np.array([[basis[k](th) for k in names] for th in ths]))
    keep = []
    for j in range(len(names)):
        if np.linalg.matrix_rank(M[:, keep + [j]], tol=tol * abs(M).max()) > len(keep):
            keep.append(j)
    return [names[j] for j in keep], M[:, keep]


def decompose_vals(vals, names, M):
    sc = abs(M).max(axis=0)
    vals = np.real(vals)
    sol = np.linalg.lstsq(M / sc, vals, rcond=None)[0] / sc
    if not np.allclose(M @ sol, vals, rtol=1e-9, atol=1e-9 * max(1.0, abs(vals).max())):
        raise ValueError("not in span")
    out = {}
    for n, s in zip(names, sol):
        r = rational(round(s, 6), tol=1e-9)
        if r != 0:
            assert abs(float(r) - s) < 1e-6 * max(1, abs(s)), (n, s)
            out[n] = r
    return out


def tr_prod_deriv(ws, th, dkey):
    """Exact directional derivative of sum_labels prod Tr(words) along y -> th[dkey]."""
    y, dy = th["y"], th[dkey]
    labs = sorted(set("".join(ws)))
    flat = [(i, j) for i, w in enumerate(ws) for j in range(len(w))]
    tot = 0.0
    for vals in itertools.product(range(y.shape[0]), repeat=len(labs)):
        m = dict(zip(labs, vals))
        for (ii, jj) in flat:
            v = 1.0
            for i, w in enumerate(ws):
                M = np.eye(y.shape[1])
                for j, ch in enumerate(w):
                    M = M @ (dy[m[ch]] if (i, j) == (ii, jj) else y[m[ch]])
                v *= np.trace(M)
            tot += v
    return tot


WORDS4 = {"Tr(ab)Tr(ab)": ["ab", "ab"], "Tr(aabb)": ["aabb"], "Tr(abab)": ["abab"], "Tr(aa)Tr(bb)": ["aa", "bb"]}
