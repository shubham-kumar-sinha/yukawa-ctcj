"""C_T and C_J through eps^3 at fixed points of the matrix GNY models of Prakash and Sinha (arXiv:2307.06841).

The fixed-point couplings are the closed forms of that paper: for all N at the supersymmetric points of the SU(N) adjoint
and SO(N) S^2 models, and at N = 3 for the other fixed points (one quartic coupling, g1 = 0).  Y = y^2/(4 pi)^2, G = g/(4 pi)^2.

    python3 run_models.py SU3_susy SUN4 S2N4 SO3S2A2_ns+       # print the coefficients of eps, eps^2, eps^3
"""
import sys, time
import numpy as np, sympy as sp
from models import su_basis, s2_basis, a2_basis, yukawa, quartics, current, ward_ok
from fastseries import ct_ratio, cj_ratio, S, numeric, eps, z3

Z = z3
def fp_su_susy(N):
    A = 81*N**8 + 40*N**6*(-57 + 32*Z) - 32*N**4*(-738 + 1453*Z) - 768*(-257 + 2178*Z) + 128*N**2*(-847 + 3912*Z)
    D = 5*N**2 - 36; P = 3*N**4 - 48*N**2 + 208
    base = eps/D + 3*P*eps**2/D**3 + sp.Rational(3, 4)*eps**3*A/D**5
    return 16*N*base, N*base, -base                      # Y, G1, G2

def fp_s2_susy(N):
    B = (81*N**8 + N**7*(894 - 690*Z) - 8*N**6*(111 + 494*Z) - 512*N**2*(296 + 583*Z) - 12288*(-257 + 2178*Z)
         + 8*N**5*(-3827 + 5825*Z) + 16*N**4*(-157 + 8304*Z) + 1024*N*(-2389 + 16980*Z) - 64*N**3*(-6573 + 23386*Z))
    D = 5*N**2 + 14*N - 72; P = 832 - 304*N - 68*N**2 + 18*N**3 + 3*N**4
    base = eps/(2*D) + sp.Rational(3, 2)*P*eps**2/D**3 + sp.Rational(3, 8)*eps**3*B/D**5
    return 8*N*base, N*base, -base

R = sp.Rational
FP3 = {  # N = 3 points with a single quartic coupling g = g2 (g1 = 0): (Y, G2)
    'SU3_susy': (16*eps/3 + R(304, 81)*eps**2 + 4*(288*Z + 1283)*eps**3/2187, eps/18 + R(19, 486)*eps**2 + (288*Z + 1283)*eps**3/52488),
    'SU3_ns': (16*eps/3 - R(1175, 1296)*eps**2 + (589824*Z + 10969801)*eps**3/1119744,
               -R(17, 288)*eps + R(34183, 497664)*eps**2 + (34875432*Z - 34011811)*eps**3/859963392),
    'SO3S2_susy': (4*eps/5 + R(148, 375)*eps**2 + eps**3*(3097 - 810*Z)/28125, eps/60 + R(37, 4500)*eps**2 + eps**3*(3097 - 810*Z)/1350000),
    'SO3S2_ns': (4*eps/5 - R(758, 2535)*eps**2 - eps**3*(-1187710595 + 18507528*Z)/642622500,
                 -5*eps/312 + R(90403, 3954600)*eps**2 + eps**3*(-2025716417 + 970207524*Z)/80199288000),
}
s113 = sp.sqrt(113)
for sgn, nm in [(1, 'SO3S2A2_ns+'), (-1, 'SO3S2A2_ns-')]:
    FP3[nm] = (16*eps/5 + 4*(2693 + sgn*343*s113)*eps**2/21125 - 7*eps**3*(-8097806339 + sgn*684445563*s113 + 7447830000*Z)/41894043750,
               (3 + sgn*s113)*eps/1040 + (661841 - sgn*14981*s113)*eps**2/114582000
               + eps**3*(-885818563403 + sgn*430102347235*s113 + 2838154137900*Z - sgn*912675495900*s113*Z)/3938710417200000)

def setup(model, N):
    if model == 'SU':
        Tb = Tf = su_basis(N); G = np.zeros((N, N), complex); G[0, 0], G[1, 1] = .5, -.5
    elif model == 'S2':
        Tb = Tf = s2_basis(N); G = np.zeros((N, N), complex); G[0, 1], G[1, 0] = -1j, 1j
    else:
        Tb = s2_basis(N); Tf = a2_basis(N, 0.5); G = np.zeros((N, N), complex); G[0, 1], G[1, 0] = -1j, 1j
    Y = yukawa(Tb, Tf); L1, L2 = quartics(Tb)
    t, T = current(Tb, Tf, G)
    ok, okm = ward_ok(Y, T, t)
    if not ok:
        assert okm; T = -T
    return Y, L1, L2, T, t, len(Tb), len(Tf)

def run(model, N, Ys, G1, G2):
    Y, L1, L2, T, t, ns, nf = setup(model, N)
    parts = {'L1': L1, 'L2': L2}; coef = {'L1': S(G1), 'L2': S(G2)}
    ct = ct_ratio(Y, S(Ys), parts, coef, ns, nf)
    cj = cj_ratio(Y, S(Ys), parts, coef, T, t)
    return ct, cj

def show(name, c):
    nums = numeric(c)
    return '%-14s %s' % (name, '  '.join('%+.5f' % v for v in nums[1:]))

def fixed_point(w):
    """(C_T/C_T,free, C_J/C_J,free) as arrays [eps^k, zeta3^j] for a fixed point named as in FP3, or SUN<n> / S2N<n>."""
    if w.startswith('SUN') or w.startswith('S2N'):
        N = int(w[3:])
        Ys, G1, G2 = (fp_su_susy if w.startswith('SU') else fp_s2_susy)(sp.Integer(N))
        return run('SU' if w.startswith('SU') else 'S2', N, Ys, G1, G2)
    Ys, G = FP3[w]
    model = 'SU' if w.startswith('SU') else ('S2' if 'A2' not in w else 'A2')
    return run(model, 3, Ys, 0, G)

if __name__ == '__main__':
    which = sys.argv[1:] or ['SU3_susy']
    for w in which:
        t0 = time.time()
        ct, cj = fixed_point(w)
        print(show(w + ' CT', ct)); print(show(w + ' CJ', cj)); print('   (%.0fs)' % (time.time() - t0), flush=True)
