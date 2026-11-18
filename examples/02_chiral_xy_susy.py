"""Chiral XY (Nambu-Jona-Lasinio-Yukawa): complex scalar = 2 real scalars, Yukawas sigma_3, sigma_1 in a
Majorana basis, N = Tr 1.  The U(1) acts with fermion charge 1, scalar charge 2.

At N = 2 the fixed point is the N=2 Wess-Zumino model; its U(1) is the superconformal R-symmetry
(R_phi = 2/3, R_psi = -1/3), whose current sits in the stress-tensor multiplet.  Superconformal symmetry
then requires C_J^R / C_J^R(free chiral multiplet) = C_T / C_T(free chiral multiplet)."""
from _common import *
import sympy as sp
from ctcj.theory import Theory, O_n_quartic

Y = [np.array([s3, s1])]
t_U1 = rotation(2, 1, 0, 2.0)          # t_01 = +2: fixed by the Ward identity i[y^a, T] = t_ab y^b with T = s2
for N in [2, 4, 8]:
    th = Theory(Y=Y, L=[O_n_quartic(2)], mult=N/2, name=f"chiral XY N={N}")
    print(th.report(currents={"U(1)": (s2, t_U1)}, show_unstable=False))

# --- supersymmetry test at N = 2
th = Theory(Y=Y, L=[O_n_quartic(2)], mult=1, name="N=2 WZ")
fp = stable(th.fixed_points())[0]
cT = th.CT(fp)
cJ = th.CJ(fp, s2/3, t_U1/3)                      # R-current normalisation
e = sp.Symbol("e"); d = 4 - e
free_R_here = 2*sp.Rational(1, 9) + 2*sp.Rational(4, 9)/(d - 2)          # free fields with R = (2/3, -1/3)
Rphi, Rpsi = (d-2)/(d-1), 1/(d-1)
free_R_sc = 2*Rpsi**2 + 2*Rphi**2/(d-2)                                   # free chiral multiplet, superconformal R
r = sp.series((1 + cJ[0]*e + cJ[1]*e**2) * free_R_here/free_R_sc, e, 0, 3).removeO()
print(f"\nSUSY test:  C_T/C_T,free     = 1 {cT[0]:+.6f} eps {cT[1]:+.6f} eps^2")
print(f"            C_J^R/C_J^R,free = 1 {float(r.coeff(e,1)):+.6f} eps {float(r.coeff(e,2)):+.6f} eps^2")
