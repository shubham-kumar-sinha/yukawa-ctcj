"""Chiral Ising (Gross-Neveu-Yukawa): one real scalar, y = g * 1, N = Tr 1 (N = 4 N_f for 4-component Dirac).
Prints C_T and C_J (any flavour current acting on the fermions) at the stable fixed point and compares
with the closed forms of the paper.  N = 1 is the emergent N=1 supersymmetric point (a single Majorana fermion has no continuous symmetry,
so the C_J entry at N = 1 is formal)."""
from _common import *
from ctcj.theory import Theory, O_n_quartic

S = lambda N: np.sqrt(N**2 + 132*N + 36)
def closed(N):
    P = 2253*N**4 + 62153*N**3 + 217302*N**2 + 93672*N + 4320
    R = 15*N**3 + 2980*N**2 + 3060*N + 720
    cT = (-5*N/(2*(N+6)*(3*N+2)), -(P + S(N)*R)/(648*(N+6)**3*(3*N+2)**2))
    cJ = (-3/(2*(N+6)), (37*N**2 - 327*N - 1044 - S(N)*(N+66))/(72*(N+6)**3))
    return cT, cJ

print(f"{'N':>3} | {'C_T: eps':>10} {'eps^2':>10} | {'C_J: eps':>10} {'eps^2':>10} | closed form agrees")
for N in [1, 2, 4, 8, 16]:
    th = Theory(Y=[np.ones((1, 1, 1))], L=[O_n_quartic(1)], mult=N, name=f"GNY N={N}")
    fp = stable(th.fixed_points())[0]
    cT, cJ = th.CT(fp), th.CJ(fp, np.eye(1), np.zeros((1, 1)))
    ref = closed(N)
    ok = np.allclose(cT, ref[0]) and np.allclose(cJ, ref[1])
    print(f"{N:>3} | {cT[0]:>10.6f} {cT[1]:>10.6f} | {cJ[0]:>10.6f} {cJ[1]:>10.6f} | {ok}")
print("\nN = 1 (N=1 SUSY): exact C_T/C_T,free = 1 - eps/14 - 33 eps^2/392 =", -1/14, -33/392)
