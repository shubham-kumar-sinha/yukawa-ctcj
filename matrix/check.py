"""Checks of this evaluation path, which uses the O(eps^3) formulas exactly as printed in the paper (ct3.tex, cj3.tex)
and explicit tensors, against results obtained independently in the main pipeline:

  - chiral Ising (N = 2, 4, 8) and chiral XY (N = 2) through eps^2, from the two-loop fixed point of the two-coupling families;
  - the O(n) model (n = 1, 2, 3) at O(eps^3), against Osborn-Stergiou (C_T) and the analytic bootstrap (C_J);
  - the closed-form planar limit (largeN.py) against the paper, including its odd fermion-loop terms.

Exit code non-zero on failure."""
import sys
import numpy as np
import sympy as sp
from series import ct_ratio, cj_ratio, coeffs, eps, z3

fails = []


def expect(name, got, want, tol=5e-5):
    ok = all(abs(g - w) < tol for g, w in zip(got, want))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {np.round(got, 4).tolist()}")
    if not ok:
        fails.append(name)


def sym3(n):
    d = np.eye(n)
    return np.einsum('ab,cd->abcd', d, d) + np.einsum('ac,bd->abcd', d, d) + np.einsum('ad,bc->abcd', d, d)


def family_fp(ns, N, al, be, ga):
    """Two-loop fixed point of the two-coupling families (section 2.2 of the paper)."""
    K = ns*al + 2*be + sp.Rational(1, 2)*al*N
    c5 = 2*ga - 2*be**2 - 2*ns*al*be - sp.Rational(3, 2)*N*al*be - sp.Rational(3, 2)*N*ns*al**2 - sp.Rational(1, 4)*ns**2*al**2
    c3 = -2*(2*ns*al + be); c1 = sp.Rational(1, 4)*(ns + 2)
    x = 1 - N*al/K
    ell = (x + sp.sqrt(x**2 + 4*(ns + 8)*N*al*(2*ns*al + be)/((ns + 2)*K**2))) / (2*(ns + 8))
    y2 = eps/(2*K) - (c5/(4*K**2) + c3*ell/(2*K) + c1*ell**2)*eps**2/K
    return y2, ell*eps


def num(c):
    return [float(sp.N(x.subs(z3, sp.zeta(3)))) for x in c]


# chiral Ising, through eps^2 (values of the chiral-model table of the paper)
table = {2: ([-0.0781, -0.0906], [-0.1875, -0.0742]), 4: ([-0.0714, -0.0779], [-0.1500, -0.0479]),
         8: ([-0.0549, -0.0527], [-0.1071, -0.0193])}
for N, (wT, wJ) in table.items():
    Y = np.eye(N)[None, :, :]
    y2, lam = family_fp(1, N, 1, 1, 1)
    T = np.zeros((N, N), complex); T[0, 1], T[1, 0] = -1j, 1j
    ct = ct_ratio(Y, y2, {'L': sym3(1)}, {'L': lam}, 1, N, order=2)
    cj = cj_ratio(Y, y2, {'L': sym3(1)}, {'L': lam}, T, np.zeros((1, 1)), order=2)
    expect(f"chiral Ising N={N}, C_T", num(coeffs(ct, 2))[1:], wT, 1e-4)
    expect(f"chiral Ising N={N}, C_J", num(coeffs(cj, 2))[1:], wJ, 1e-4)

# chiral XY, N = 2, U(1) current with charges (1, 2), through eps^2
s1 = np.array([[0, 1], [1, 0]]); s3 = np.diag([1, -1]); s2 = np.array([[0, -1j], [1j, 0]])
Y = np.array([s3, s1], dtype=float)
y2, lam = family_fp(2, 2, 1, 0, 4)
t = np.array([[0, 2.], [-2., 0]])
for a in range(2):
    assert np.allclose(1j*(Y[a] @ s2 - s2 @ Y[a]), sum(t[a, b]*Y[b] for b in range(2))), 'Ward identity'
ct = ct_ratio(Y, y2, {'L': sym3(2)}, {'L': lam}, 2, 2, order=2)
cj = cj_ratio(Y, y2, {'L': sym3(2)}, {'L': lam}, s2, t, order=2)
expect("chiral XY N=2, C_T", num(coeffs(ct, 2))[1:], [-1/6, -1/8])
expect("chiral XY N=2, C_J", num(coeffs(cj, 2))[1:], [-1/6, -1/8 - 1/18])

# O(n) Wilson-Fisher at O(eps^3)
for n in [1, 2, 3]:
    lam = eps/(n + 8) + 3*(3*n + 14)*eps**2/(n + 8)**3
    Y = np.zeros((n, 1, 1))
    ct = ct_ratio(Y, 0, {'L': sym3(n)}, {'L': lam}, n, 0, order=3)
    exp_ct = -sp.Rational(5)*(n + 2)/(12*(n + 8)**2)*eps**2 - sp.Rational(n + 2)*(7*n**2 + 382*n + 1708)/(36*(n + 8)**4)*eps**3
    expect(f"O({n}) C_T - Osborn-Stergiou", num(coeffs(ct - 1 - exp_ct, 3)), [0, 0, 0, 0], 1e-12)
    if n >= 2:
        t = np.zeros((n, n)); t[0, 1], t[1, 0] = 1, -1
        cj = cj_ratio(Y, 0, {'L': sym3(n)}, {'L': lam}, np.zeros((1, 1)), t, order=3)
        exp_cj = -sp.Rational(3)*(n + 2)/(4*(n + 8)**2)*eps**2 - sp.Rational(n + 2)*(n**2 + 132*n + 632)/(8*(n + 8)**4)*eps**3
        expect(f"O({n}) C_J - analytic bootstrap", num(coeffs(cj - 1 - exp_cj, 3)), [0, 0, 0, 0], 1e-12)

# planar limit in closed form
import largeN
R = sp.Rational
e, z = largeN.eps, largeN.z3
okT = sp.simplify(largeN.CT - (1 - e/10 - R(101, 1000)*e**2 + (R(13, 250)*z - R(21881, 300000))*e**3)) == 0
okJ = sp.simplify(largeN.CJ - (1 - e/5 - R(19, 125)*e**2 + (z/20 - R(272, 3125))*e**3)) == 0
# the odd fermion loops survive the planar limit: (7/6 - zeta3)/1250 and 1/500 (paper, section 5.5)
okTo = sp.simplify(largeN.CT - largeN.CT_EVEN - (R(7, 6) - z)*e**3/1250) == 0
okJo = sp.simplify(largeN.CJ - largeN.CJ_EVEN - e**3/500) == 0
for nm, ok in [("planar C_T (paper)", okT), ("planar C_J (paper)", okJ),
               ("planar odd fermion loops in C_T: (7/6 - zeta3)/1250", okTo), ("planar odd fermion loops in C_J: 1/500", okJo)]:
    print(f"{'PASS' if ok else 'FAIL'}  {nm}")
    if not ok:
        fails.append(nm)

print("ALL CHECKS PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
