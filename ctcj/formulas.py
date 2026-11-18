"""Fixed-point coefficients of C_T and C_J (output of scripts/compute_results.py, frozen here;
compute_results.py checks that the assembled values still agree).

C_T / C_T,S = n_s + (d-1)/2 Tr1 + sum_k CT[k] * invariant_k
C_J / C_J,psi = Tr(TT) + t.t/(d-2) + sum_k CJ[k] * invariant_k
Invariant names: 'Tr(aabb)' = Tr(y^a y^a y^b y^b), 'Tr(TaTb)Tr(ab)' = Tr(T y^a T y^b) Tr(y^a y^b), etc.
"""
from fractions import Fraction as F

CT = {"Tr(aa)": F(-5, 4), "Tr(ab)Tr(ab)": F(-1, 6), "Tr(aabb)": F(-11, 48), "Tr(abab)": F(-19, 6),
      "lam.lam": F(-5, 36)}
CJ = {"Tr(TTaa)": F(-3, 2), "Tr(TaTa)": F(0), "Tr(TTaabb)": F(-1), "Tr(TTabab)": F(-17, 2),
      "Tr(TTabba)": F(-5, 8), "Tr(TaTabb)": F(3, 4), "Tr(TaTbab)": F(15, 2), "Tr(TabTba)": F(3, 2),
      "Tr(TTab)Tr(ab)": F(-1), "Tr(TaTb)Tr(ab)": F(3, 2), "t t lam lam": F(-3, 8)}
CT = {k: float(v) for k, v in CT.items()}
CJ = {k: float(v) for k, v in CJ.items()}
