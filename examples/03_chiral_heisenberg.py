"""Chiral Heisenberg: O(3) vector scalar, Yukawas = Pauli matrices, N = Tr 1; current = SU(2) generator."""
from _common import *
from ctcj.theory import Theory, O_n_quartic

T = s3/2
t = rotation(3, 1, 0, 1.0)        # Ward identity: i[sigma^a, sigma_3/2] = t_ab sigma^b  ->  t_01 = +1
for N in [2, 4, 8]:
    th = Theory(Y=[np.array([s1, s2, s3])], L=[O_n_quartic(3)], mult=N/2, name=f"chiral Heisenberg N={N}")
    print(th.report(currents={"SU(2)": (T, t)}, show_unstable=False))
