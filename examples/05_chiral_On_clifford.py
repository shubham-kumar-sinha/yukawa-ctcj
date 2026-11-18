"""Chiral O(n) Gross-Neveu-Yukawa models: an O(n) vector of scalars coupled through n anticommuting
4x4 Clifford matrices, y^a = g Gamma^a (x) 1, N = Tr 1.  n = 3 is chiral Heisenberg (doubled), n = 4, 5
are the O(4) / SO(5) models discussed for Dirac spin liquids and deconfined criticality.
Currents: an SO(n) rotation in the (1,2) plane, T = (i/2) Gamma^1 Gamma^2 with t_12 = -1, and fermion number."""
from _common import *
from ctcj.theory import Theory, O_n_quartic

for n in [3, 4, 5]:
    for N in [4, 8]:
        Y = [np.array(GAMMA[:n])]
        th = Theory(Y=Y, L=[O_n_quartic(n)], mult=N/4, name=f"chiral O({n}), N={N}")
        T = 0.5j * GAMMA[0] @ GAMMA[1]
        t = rotation(n, 0, 1, 1.0)
        assert th.check_current(T, t)
        print(th.report(currents={f"SO({n})": (T, t), "U(1)": (np.eye(4), np.zeros((n, n)))}, show_unstable=False))
