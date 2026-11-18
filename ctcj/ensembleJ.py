"""Random theories with an exact U(1) symmetry (delta psi = i T psi, delta phi = t phi), for <JJ>.

Any single current generates a one-parameter group, so integer-charge U(1) theories are fully general
for the Ward identity  i[y^a, T] = t_ab y^b,  t.lambda = 0.  Invariant couplings are obtained from random
ones by exact averaging over the cyclic group Z_K inside U(1) (K larger than every total charge), which
projects onto charge-zero components.  'mult' replicates the fermions (Tr -> mult * trace).
"""
import numpy as np
from scipy.linalg import expm


def _rot(t, th):
    return expm(th * t)


def u1_theory(qf, qs, n0, rng, mult=1.0):
    """qf: fermion charges; qs: charges of complex scalars (each = 2 real fields); n0 neutral reals."""
    nf = len(qf)
    ns = 2 * len(qs) + n0
    T = np.diag(np.array(qf, dtype=float)).astype(complex)
    t = np.zeros((ns, ns))
    for j, Q in enumerate(qs):
        t[2 * j:2 * j + 2, 2 * j:2 * j + 2] = Q * np.array([[0, -1], [1, 0]])
    mq = max([abs(q) for q in qf] + [0])
    mQ = max([abs(Q) for Q in qs] + [0])
    K = int(max(2 * mq + mQ, 4 * mQ)) + 1
    ths = 2 * np.pi * np.arange(K) / K
    R = [_rot(t, a) for a in ths]
    U = [np.diag(np.exp(1j * a * np.array(qf, float))) for a in ths]
    # Yukawa: y^a = sum_b R_ba U^dag y^b U  averaged
    y0 = rng.normal(size=(ns, nf, nf)) + 1j * rng.normal(size=(ns, nf, nf))
    y0 = y0 + y0.conj().transpose(0, 2, 1)
    y = np.zeros_like(y0)
    for Rk, Uk in zip(R, U):
        y += np.einsum("ba,ij,bjk,kl->ail", Rk, Uk.conj().T, y0, Uk)
    y /= K
    # quartic
    l0 = rng.normal(size=(ns,) * 4)
    import itertools
    l0 = sum(np.transpose(l0, p) for p in itertools.permutations(range(4))) / 24
    lam = np.zeros_like(l0)
    for Rk in R:
        lam += np.einsum("ae,bf,cg,dh,efgh->abcd", Rk, Rk, Rk, Rk, l0, optimize=True)
    lam /= K
    th = dict(y=y, lam=lam, T=T, t=t, mult=float(mult))
    ward(th)
    return th


def ward(th, tol=1e-9):
    y, T, t, lam = th["y"], th["T"], th["t"], th["lam"]
    lhs = 1j * (np.einsum("aij,jk->aik", y, T) - np.einsum("ij,ajk->aik", T, y))
    rhs = np.einsum("ab,bij->aij", t, y)
    assert np.allclose(lhs, rhs, atol=tol * max(1, abs(y).max())), "Yukawa Ward identity"
    w = (np.einsum("ae,ebcd->abcd", t, lam) + np.einsum("be,aecd->abcd", t, lam)
         + np.einsum("ce,abed->abcd", t, lam) + np.einsum("de,abce->abcd", t, lam))
    assert np.allclose(w, 0, atol=tol * max(1, abs(lam).max())), "quartic Ward identity"
    assert np.allclose(y, y.conj().transpose(0, 2, 1))


FAMILIES = {
    # name: (fermion charges, complex-scalar charges, neutral scalars)
    "A": ([0, 0, 1, 1, -1, 2, -1, 3, 1], [1, 2, 1], 2),          # nf 9, ns 8
    "B": ([1, 0, -1, 2, 2, -2, 0, 1, 3, -1], [1, 3, 2, 1], 1),     # nf 10, ns 9
    "C": ([0, 1, 1, 2, -1, -1, 0, 2, -2], [1, 1, 2, 2], 0),        # nf 9, ns 8
}


def ensemble(n, families=("A", "C"), seed=12345, mult=True):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        qf, qs, n0 = FAMILIES[families[i % len(families)]]
        perm = rng.permutation(len(qf))
        m = rng.uniform(0.5, 3.0) if mult else 1.0
        out.append(u1_theory([qf[k] for k in perm], qs, n0, rng, m))
    return out
