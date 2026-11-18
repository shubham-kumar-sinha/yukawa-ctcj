"""Several Yukawa couplings: one real scalar coupled with independent strengths g1, g2 to two fermion
species with Tr 1 = N1 and N2.  The solver finds all fixed points:
  * g1 = g2      : chiral Ising with N = N1 + N2 (stable)
  * g2 = 0       : chiral Ising with N1, plus N2 decoupled free fermions
  * g1 = 0       : chiral Ising with N2, plus N1 decoupled free fermions
  * g1 = g2 = 0  : Wilson-Fisher / Gaussian
The decoupled fixed points are a consistency check: C_T must equal C_T(GNY, N1) + free fermions exactly."""
from _common import *
import sympy as sp
from ctcj.theory import Theory, O_n_quartic

N1, N2 = 4, 2
P1 = np.diag([1.0]*N1 + [0.0]*N2)[None]; P2 = np.diag([0.0]*N1 + [1.0]*N2)[None]
th = Theory(Y=[P1, P2], L=[O_n_quartic(1)], name=f"two species N1={N1}, N2={N2}")
U1 = (np.diag([1.0]*N1 + [0.0]*N2), np.zeros((1, 1)))          # current of species 1
print(th.report(currents={"U(1)_1": U1}))

# check: decoupled fixed point vs chiral Ising(N1) + N2 free fermions
e = sp.Symbol("e")
F = lambda n: 1 + n*(3 - e)/2                     # C_T,free / C_T,S for 1 scalar + Tr1 = n
gny = Theory(Y=[np.ones((1, 1, 1))], L=[O_n_quartic(1)], mult=N1)
c_gny = gny.CT(stable(gny.fixed_points())[0])
pred = sp.series(((1 + c_gny[0]*e + c_gny[1]*e**2)*F(N1) + N2*(3 - e)/2)/F(N1 + N2), e, 0, 3).removeO()
for fp in th.fixed_points():
    if abs(fp["a"][1]) < 1e-9 and abs(fp["a"][0]) > 1e-6 and fp["c"][0] > 0:
        c = th.CT(fp)
        print(f"\ndecoupled fixed point: C_T = ({c[0]:+.6f}, {c[1]:+.6f});  GNY(N1) + free N2 predicts "
              f"({float(pred.coeff(e,1)):+.6f}, {float(pred.coeff(e,2)):+.6f})")
