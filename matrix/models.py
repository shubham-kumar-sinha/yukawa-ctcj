"""Explicit tensors for the matrix GNY models of Prakash-Sinha (arXiv:2307.06841), in JOS conventions,
using eqs (A.1), (A.2) of that paper. Couplings: Y = y^2/(4pi)^2, G1 = g1/(4pi)^2, G2 = g2/(4pi)^2."""
import numpy as np, itertools

def su_basis(N):
    """Hermitian traceless, tr(L^a L^b) = 1/2 delta."""
    B = []
    for i in range(N):
        for j in range(i + 1, N):
            M = np.zeros((N, N), complex); M[i, j] = M[j, i] = 1; B.append(M / 2)
            M = np.zeros((N, N), complex); M[i, j] = -1j; M[j, i] = 1j; B.append(M / 2)
    for k in range(1, N):
        M = np.zeros((N, N), complex)
        M[:k, :k] = np.eye(k); M[k, k] = -k
        B.append(M / np.sqrt(2 * k * (k + 1)))
    return np.array(B)

def s2_basis(N):
    """real symmetric traceless, tr(T^a T^b) = delta."""
    B = []
    for i in range(N):
        for j in range(i + 1, N):
            M = np.zeros((N, N)); M[i, j] = M[j, i] = 1 / np.sqrt(2); B.append(M)
    for k in range(1, N):
        M = np.zeros((N, N)); M[:k, :k] = np.eye(k); M[k, k] = -k
        B.append(M / np.sqrt(k * (k + 1)))
    return np.array(B, dtype=complex)

def a2_basis(N, norm):
    """i * real antisymmetric, Hermitian; tr(H^a H^b) = norm * delta."""
    B = []
    for i in range(N):
        for j in range(i + 1, N):
            M = np.zeros((N, N), complex); M[i, j] = -1j; M[j, i] = 1j
            B.append(M * np.sqrt(norm / 2))
    return np.array(B)

def tr(*Ms):
    P = Ms[0]
    for M in Ms[1:]: P = P @ M
    return np.trace(P)

def yukawa(Tb, Tf):
    """(A.1) with y = 1:  y^a_ij = 1/2 (tr T^a T'^i T'^j + tr T^a T'^j T'^i)."""
    Y = 0.5 * (np.einsum('aij,bjk,cki->abc', Tb, Tf, Tf) + np.einsum('aij,cjk,bki->abc', Tb, Tf, Tf))
    assert np.allclose(Y.imag, 0, atol=1e-12)
    return Y.real

def quartics(Tb):
    """(A.2) with g1 = 1 (L1) and g2 = 1 (L2) separately."""
    four = np.einsum('aij,bjk,ckl,dli->abcd', Tb, Tb, Tb, Tb)
    L1 = 4 * (four + four.transpose(0, 1, 3, 2) + four.transpose(0, 2, 1, 3) + four.transpose(0, 2, 3, 1)
              + four.transpose(0, 3, 1, 2) + four.transpose(0, 3, 2, 1))
    two = np.einsum('aij,bji->ab', Tb, Tb)
    L2 = 8 * (np.einsum('ab,cd->abcd', two, two) + np.einsum('ad,bc->abcd', two, two) + np.einsum('ac,bd->abcd', two, two))
    assert np.allclose(L1.imag, 0, atol=1e-12) and np.allclose(L2.imag, 0, atol=1e-12)
    return L1.real, L2.real

def current(Tb, Tf, G):
    """symmetry phi -> phi + i[G,phi]: t_ab real antisymmetric on scalars, T Hermitian on fermions."""
    def gen(B):
        nb = np.einsum('aij,bji->ab', B, B).real  # Gram matrix tr(B^a B^b)
        X = np.einsum('aij,jk,bki->ab', B, G, B) - np.einsum('aij,bjk,ki->ab', B, B, G)  # tr(B^a [G,B^b])
        M = np.linalg.solve(nb, 1j * X)   # components: i[G,B^b] = sum_a M_ab B^a
        return M
    t = gen(Tb); tf = gen(Tf)
    assert np.allclose(t.imag, 0, atol=1e-12) and np.allclose(tf.imag, 0, atol=1e-12)
    return t.real, -1j * tf.real   # delta phi = t phi ; delta psi = i T psi

def ward_ok(Y, T, t):
    lhs = 1j * (np.einsum('aij,jk->aik', Y, T) - np.einsum('ij,ajk->aik', T, Y))
    rhs = np.einsum('ab,bij->aij', t, Y)
    return np.allclose(lhs, rhs, atol=1e-10), np.allclose(lhs, -rhs, atol=1e-10)

# ---------------- one-loop checks against Prakash-Sinha ----------------
def beta_y_coeff(Y):
    """one-loop JOS beta_y with unit couplings, returned as multiple of y^a (must be proportional)."""
    B = 2 * np.einsum('bij,ajk,bkl->ail', Y, Y, Y) + 0.5 * (np.einsum('bij,bjk,akl->ail', Y, Y, Y)
        + np.einsum('aij,bjk,bkl->ail', Y, Y, Y)) + 0.5 * np.einsum('bij,ba->aij', Y, np.einsum('bkl,alk->ba', Y, Y))
    c = np.vdot(Y, B) / np.vdot(Y, Y)
    assert np.allclose(B, c * Y, atol=1e-10)
    return c.real

def gamma_phi_coeff(Y):
    g = 0.5 * np.einsum('aij,bji->ab', Y, Y)
    c = g[0, 0]; assert np.allclose(g, c * np.eye(len(g)))
    return c
