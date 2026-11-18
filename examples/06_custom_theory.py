"""Template for your own theory, here with several quartic couplings: the chiral XY model with a
cubic-anisotropy term,   lambda_abcd = l1 (d_ab d_cd + d_ac d_bd + d_ad d_bc) + l2 delta_abcd .
The solver lists every fixed point; the U(1) current is evaluated only where the Ward identity holds
(the anisotropic fixed points break U(1) to Z_4, so C_J is not defined there).

To define a theory:
  Y    : list of Yukawa basis tensors, each an array (n_s, n_f, n_f) of hermitian matrices
  L    : list of totally symmetric quartic basis tensors (n_s,)*4
  mult : multiplies every fermion trace (Tr = mult * trace), e.g. N copies without big matrices;
         normalisation: Tr 1 = 4 for one four-component Dirac fermion
  currents: {name: (T, t)} with T hermitian on fermions, t real antisymmetric on scalars, satisfying
         i[y^a, T] = t_ab y^b  (use Theory.check_current to test)."""
from _common import *
from ctcj.theory import Theory, O_n_quartic

D = np.zeros((2,)*4); D[0, 0, 0, 0] = D[1, 1, 1, 1] = 1.0          # cubic anisotropy
for N in [2, 8]:
    th = Theory(Y=[np.array([s3, s1])], L=[O_n_quartic(2), D], mult=N/2, name=f"chiral XY + cubic anisotropy, N={N}")
    print(th.report(currents={"U(1)": (s2, rotation(2, 1, 0, 2.0))}))

# O(eps^3): the three-loop Yukawa beta function gives y* to O(eps^3), and with it C_T and C_J through eps^3.
print("\nThrough O(eps^3) at the stable fixed point of the isotropic model (N = 2 is the N=2 super-Ising point):")
for N in [2, 8]:
    th = Theory(Y=[np.array([s3, s1])], L=[O_n_quartic(2)], mult=N/2, name=f"chiral XY, N={N}")
    fp = [f for f in th.fixed_points() if f["stable"]][0]
    ct, cj = th.CT3(fp), th.CJ3(fp, s2, rotation(2, 1, 0, 2.0))
    print(f"  N={N}:  C_T/C_T,free = 1 {ct[1]:+.6f} eps {ct[2]:+.6f} eps^2 {ct[3]:+.6f} eps^3;"
          f"  C_J/C_J,free = 1 {cj[1]:+.6f} eps {cj[2]:+.6f} eps^2 {cj[3]:+.6f} eps^3")
