"""O(eps^3) fixed-point coefficients of C_T and C_J (four loops), frozen from scripts/compute_results4.py and
scripts/compute_results4J.py (which check that the assembled values still agree).

Each entry is (a, b) meaning a + b*zeta_3.  At any fixed point, the O(eps^3) terms are
    Delta C_T / C_T,S   = sum_k CT4[k] * I_k(y, lam)            (I_k: ctcj.invariants4.T_basis4)
    Delta C_J / C_J,psi = sum_k CJ4[k] * J_k(y, lam, T, t)      (J_k: ctcj.basisJ4 candidates, basis data/basis4_J.json)
with couplings rescaled y -> 4 pi y, lam -> 16 pi^2 lam and Tr = tr 1 x flavour trace (JOS normalisation).
The O(eps^2) coefficients (ctcj.formulas) are unchanged; inside them y* is needed to O(eps^3) in the two-loop
invariants and to O(eps^2) in the three-loop ones, and lambda* to O(eps^2) in the lambda^2 terms.
Word notation: 'TaTbcabc' = Tr(T y^a T y^b y^c y^a y^b y^c); 't_ab t_ac lam_bdef lam_cdeg Tr(fg)' =
t_ab t_ac lam_bdef lam_cdeg Tr(y^f y^g).
"""
from fractions import Fraction as F

ZETA3 = 1.2020569031595942853997

CT4 = {
    "Tr(aabbcc)": (F(1, 72), F(9, 8)),
    "Tr(aabcbc)": (F(-49, 24), F(21, 2)),
    "Tr(aabccb)": (F(-289, 576), F(9, 8)),
    "Tr(abacbc)": (F(-119, 18), F(11, 1)),
    "Tr(abcabc)": (F(-11, 4), F(-15, 4)),
    "Tr(aabc)Tr(bc)": (F(-37, 32), F(21, 8)),
    "Tr(ab)Tr(acbc)": (F(-187, 288), F(23, 4)),
    "Tr(ab)Tr(ac)Tr(bc)": (F(-31, 576), F(3, 4)),
    "lam Tr(abcd)": (F(127, 36), F(0, 1)),
    "lam_acde lam_bcde Tr(ab)": (F(253, 1728), F(0, 1)),
    "lam_abcd lam_abef lam_cdef": (F(-7, 36), F(0, 1)),
}

CJ4 = {
    "Tr(TTaabbcc)": (F(-9, 32), F(3, 4)),
    "Tr(TTaabcbc)": (F(-35, 16), F(9, 2)),
    "Tr(TTaabccb)": (F(-5, 8), F(3, 4)),
    "Tr(TTabacbc)": (F(-5, 2), F(3, 1)),
    "Tr(TTabaccb)": (F(-29, 32), F(3, 1)),
    "Tr(TTabbcac)": (F(-1, 32), F(0, 1)),
    "Tr(TTabbcca)": (F(9, 32), F(0, 1)),
    "Tr(TTabcabc)": (F(-11, 2), F(-9, 2)),
    "Tr(TTabcacb)": (F(-15, 4), F(3, 2)),
    "Tr(TTabcbac)": (F(-19, 4), F(3, 2)),
    "Tr(TTabcbca)": (F(9, 16), F(0, 1)),
    "Tr(TTabccab)": (F(-27, 16), F(3, 2)),
    "Tr(TTabccba)": (F(-1, 32), F(0, 1)),
    "Tr(TaTabbcc)": (F(1, 8), F(0, 1)),
    "Tr(TaTabcbc)": (F(13, 4), F(0, 1)),
    "Tr(TaTabccb)": (F(-1, 8), F(0, 1)),
    "Tr(TaTbabcc)": (F(7, 8), F(-3, 1)),
    "Tr(TaTbacbc)": (F(-2, 1), F(0, 1)),
    "Tr(TaTbaccb)": (F(7, 8), F(-3, 1)),
    "Tr(TaTbbacc)": (F(13, 16), F(0, 1)),
    "Tr(TaTbcabc)": (F(13, 2), F(0, 1)),
    "Tr(TaTbcacb)": (F(11, 2), F(0, 1)),
    "Tr(TaabcTcb)": (F(-1, 8), F(0, 1)),
    "Tr(TabTbcac)": (F(-13, 4), F(0, 1)),
    "Tr(TabTbcca)": (F(-1, 8), F(0, 1)),
    "Tr(TabTcabc)": (F(-5, 2), F(0, 1)),
    "Tr(TabTcbac)": (F(-3, 2), F(0, 1)),
    "Tr(TabTcbca)": (F(-13, 4), F(0, 1)),
    "Tr(TabcTcba)": (F(-3, 2), F(0, 1)),
    "Tr(TTaabc)Tr(bc)": (F(15, 32), F(3, 4)),
    "Tr(TTab)Tr(abcc)": (F(-5, 8), F(3, 4)),
    "Tr(TTab)Tr(acbc)": (F(-25, 8), F(3, 1)),
    "Tr(TTab)Tr(accb)": (F(-5, 8), F(3, 4)),
    "Tr(TTabac)Tr(bc)": (F(51, 32), F(3, 4)),
    "Tr(TTabbc)Tr(ac)": (F(25, 32), F(0, 1)),
    "Tr(TTabca)Tr(bc)": (F(-3, 16), F(0, 1)),
    "Tr(TTabcb)Tr(ac)": (F(51, 32), F(3, 4)),
    "Tr(TaTabc)Tr(bc)": (F(-17, 8), F(0, 1)),
    "Tr(TaTb)Tr(abcc)": (F(-5, 8), F(-3, 2)),
    "Tr(TaTb)Tr(acbc)": (F(21, 8), F(-3, 1)),
    "Tr(TaTbac)Tr(bc)": (F(-7, 2), F(0, 1)),
    "Tr(Tab)Tr(Tcabc)": (F(-13, 8), F(0, 1)),
    "Tr(Tab)Tr(Tcbac)": (F(13, 8), F(0, 1)),
    "Tr(TTab)Tr(ac)Tr(bc)": (F(-5, 32), F(3, 4)),
    "Tr(TaTb)Tr(ac)Tr(bc)": (F(5, 8), F(-3, 4)),
    "lam_abcd Tr(TTabcd)": (F(9, 1), F(0, 1)),
    "lam_abcd Tr(TaTbcd)": (F(-19, 2), F(0, 1)),
    "lam_abcd lam_abce Tr(TTde)": (F(-5, 32), F(0, 1)),
    "lam_abcd lam_abce Tr(TdTe)": (F(19, 48), F(0, 1)),
    "t_ab t_ac lam_bdef lam_cdeg Tr(fg)": (F(1, 16), F(0, 1)),
    "t_ab t_ac lam_bdef lam_cdgh lam_efgh": (F(-3, 8), F(0, 1)),
    "t_ab t_cd lam_acef lam_bdgh lam_efgh": (F(1, 8), F(0, 1)),
}


# Odd fermion loops (theories without a four-dimensional counterpart, DREG_3; ODD_LOOPS.md):
#     Delta C_T / C_T,S = KAPPA_T_ODD * Tr(y^a y^b y^c) Tr(y^a y^b y^c)      (O(eps^3), leading-order couplings)
# with the five-gamma odd traces read from the two ends of one scalar line and eps~ eps~ = det[delta] (f = 1).
# This value is fixed by N=1 supersymmetry: C_S/C_S,free = C_T/C_T,free through O(eps^3) at four N=1 points
# (scripts/supercurrent_eps3_general.py).  Not included by default: callers opt in for 3d-only theories.
KAPPA_T_ODD = (F(7, 6), F(-1, 1))


def odd_invariant_T(y, mult=1.0):
    """Tr(y^a y^b y^c) Tr(y^a y^b y^c), Tr = mult x trace.  Only for real symmetric Yukawa matrices (Majorana
    fermions), the case for which the coefficient was derived."""
    import numpy as np
    y = np.asarray(y)
    assert np.allclose(np.imag(y), 0) and np.allclose(y, np.transpose(y, (0, 2, 1))), \
        "odd-loop term derived for real symmetric Yukawa matrices only"
    y = np.real(y)
    t = mult * np.einsum("aij,bjk,cki->abc", y, y, y, optimize=True)
    return float(np.einsum("abc,abc->", t, t))


# The same rule for C_J (confirmed by the N=1 flavour-current multiplet, scripts/supercurrent_jj.py):
#     Delta C_J / C_J,psi = KAPPA_J_ODD[0] * I0 + KAPPA_J_ODD[1] * I1,   coefficients (a, b) = a + b zeta3,
#     I0 = Tr(T y^a y^b y^c) Tr(T y^a y^b y^c),  I1 = Tr(T y^a y^b y^c) Tr(T y^b y^c y^a)   (T: fermion generator)
KAPPA_J_ODD = ((F(-3, 1), F(0, 1)), (F(2, 1), F(-6, 1)))


def odd_invariants_J(y, T, mult=1.0):
    """(I0, I1) for the odd-loop term of C_J; real symmetric Yukawa matrices only (see odd_invariant_T)."""
    import numpy as np
    y = np.asarray(y)
    assert np.allclose(np.imag(y), 0) and np.allclose(y, np.transpose(y, (0, 2, 1))), \
        "odd-loop term derived for real symmetric Yukawa matrices only"
    y = np.real(y)
    t = mult * np.einsum("ij,ajk,bkl,cli->abc", np.asarray(T, complex), y, y, y, optimize=True)
    return complex(np.einsum("abc,abc->", t, t)).real, complex(np.einsum("abc,bca->", t, t)).real


def value(c):
    a, b = c
    return float(a) + float(b) * ZETA3


def cj4_invariants(th):
    """Four-loop <JJ> invariants for a theory dict(y=(ns,nf,nf) hermitian, T=(nf,nf), t=(ns,ns), lam=(ns,)*4,
    mult=Tr 1 per flavour-trace unit)."""
    from .basisJ4 import candidates
    from .net import evaluate
    th = dict(th); th.setdefault("mult", 1.0)
    nets = {}
    for s in ("y6", "y4lam", "y2lam2", "lam3"):
        nets.update(candidates(s))
    return {k: evaluate(nets[k], th).real for k in CJ4}


def delta_CJ_eps3(th):
    """O(eps^3) four-loop term of C_J/C_J,psi for couplings at the one-loop fixed point."""
    inv = cj4_invariants(th)
    return sum(value(c) * inv[k] for k, c in CJ4.items())
