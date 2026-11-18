"""Three-loop anomalous dimensions and Yukawa beta function of a general scalar-fermion theory, transcribed from
I. Jack, H. Osborn, T. Steudtner, arXiv:2301.10903: section 2 for gamma_phi, section 3 for
gamma_psi, section 4 for tilde-beta_y, with the coefficients listed after each (MS-bar; originally computed in
Steudtner:2021fzs, Herren:2017uxn, Bednyakov:2021qxa, Davies:2021mnc).

Conventions are those of ctcj.betasJ: hermitian Yukawa matrices y[a] (Dirac language), Tr = mult x trace,
couplings rescaled by 4 pi, beta_y^a = tilde-beta_y^a + gamma_psi y^a + y^a gamma_psi + gamma_phi^{ab} y^b.
JOS products y^{abc..} = y^a y^b y^c ...; their tr maps to our Tr (checked at one and two loops).
S_2 (symmetrisation) is implemented as 'add the reversed product' (= hermitian conjugate), and for gamma_phi as
'add a <-> b'; the totals are then projected on their hermitian (gamma_psi, beta_y^a) or real symmetric
(gamma_phi) parts, which is what both orientations of Dirac fermion loops give.  The optional antisymmetric
pieces upsilon of JOS (relevant only for fixed points of the B-function) are not included, as in Zerf et al.

Each term is (coefficient, [lambda index strings], [trace index strings], chain or None, S2 flag); the free
scalar indices are 'a' (beta_y), 'ab' (gamma_phi), none (gamma_psi).
"""
import numpy as np

Z3 = 1.2020569031595942853997
F = lambda p, q: p / q

GAMMA_PHI3 = [
    (F(-1, 16), ["acde", "defg", "bcfg"], [], None, False),            # 3a
    (F(-5, 32), ["acde", "bcdf"], ["ef"], None, False),                # 3b
    (F(5, 8), ["acde"], ["bcde"], None, True),                         # 3c  S2
    (1.0, [], ["abcd", "cd"], None, False),                            # 3d
    (F(9, 16), [], ["acbd", "cd"], None, False),                       # 3e
    (F(-3, 16), [], ["abccdd"], None, False),                          # 3f
    (F(5, 16), [], ["abcddc"], None, False),                           # 3g
    (F(1, 32), [], ["accbdd"], None, False),                           # 3h
    (F(-3, 8), [], ["abcdcd"], None, False),                           # 3i
    (F(7, 16), [], ["acbcdd"], None, True),                            # 3j  S2
    (F(7, 4), [], ["acdbdc"], None, False),                            # 3k
    (F(-3, 4), [], ["acbdcd"], None, False),                           # 3l
    (1.5 * Z3 - 1, [], ["acdbcd"], None, False),                       # 3m
]

GAMMA_PSI3 = [
    (F(-11, 96), ["acde", "bcde"], [], "ab", False),                   # 3a
    (1.0, ["abcd"], [], "abcd", False),                                # 3b
    (F(-3, 32), [], ["ac", "bc"], "ab", False),                        # 3c
    (F(-1, 8), [], ["ab"], "accb", False),                             # 3d
    (F(9, 32), [], ["ab"], "cabc", False),                             # 3e
    (F(-3, 32), [], ["ab"], "acbc", True),                             # 3f  S2
    (1.0, [], ["abcc"], "ab", False),                                  # 3g
    (F(1, 2), [], ["acbc"], "ab", False),                              # 3h
    (F(-5, 32), [], [], "abbcca", False),                              # 3i
    (F(1, 16), [], [], "abccba", False),                               # 3j
    (F(1, 2), [], [], "abacbc", False),                                # 3k
    (F(-5, 16), [], [], "abcbca", False),                              # 3l
    (0.0, [], [], "abcbac", True),                                     # 3m  S2
    (F(3, 32), [], [], "abbcac", True),                                # 3n  S2
    (F(1, 4), [], [], "abccab", False),                                # 3o
    (1.5 * Z3 - 1, [], [], "abcabc", False),                           # 3p
]

BETA_Y3 = [
    (F(-3, 8), ["bdef", "cdef"], [], "bac", False),                    # 3a
    (F(1, 2), ["abef", "efcd"], [], "bcd", True),                      # 3b  S2
    (F(3, 2), ["abef", "efcd"], [], "cbd", False),                     # 3c
    (F(3, 2), ["acde"], ["eb"], "cbd", False),                         # 3d
    (F(1, 2), ["acde"], ["eb"], "bcd", True),                          # 3e  S2
    (2.0, ["bcde"], [], "bcade", False),                               # 3f
    (3.0, ["bcde"], [], "bacde", True),                                # 3g  S2
    (5.0, ["abcd"], [], "ebcde", False),                               # 3h
    (3.0, ["abcd"], [], "beced", False),                               # 3i
    (F(1, 2), ["abcd"], [], "beecd", True),                            # 3j  S2
    (-1.0, ["abcd"], [], "ebecd", True),                               # 3k  S2
    (2.0, ["abcd"], [], "ebced", True),                                # 3l  S2
    (F(-1, 2), [], ["bd", "cd"], "bac", False),                        # 3m
    (2.0, [], ["bc"], "dbacd", False),                                 # 3n
    (-1.0, [], ["bc"], "bdadc", False),                                # 3o
    (F(-1, 2), [], ["bc"], "baddc", True),                             # 3p  S2
    (F(3, 2), [], ["bc"], "dabdc", True),                              # 3q  S2
    (F(-3, 2), [], ["bc"], "badcd", True),                             # 3r  S2
    (F(-1, 2), [], ["bc"], "dbadc", True),                             # 3s  S2
    (F(25, 16), [], ["bc"], "dabcd", True),                            # 3t  S2
    (F(25, 8), [], ["bcdd"], "bac", False),                            # 3u
    (F(5, 4), [], ["bdcd"], "bac", False),                             # 3v
    (0.0, [], ["abcd"], "bcd", False),                                 # 3w
    (2 * (3 * Z3 - 2), [], ["abcd"], "bdc", True),                     # 3x  S2
    (F(-1, 2), [], [], "bccaddb", False),                              # 3y
    (F(-1, 2), [], [], "baccddb", True),                               # 3z  S2
    (F(7, 16), [], [], "bacddcb", True),                               # 3a~ S2
    (-1.0, [], [], "bcacddb", True),                                   # 3b~ S2
    (2.0, [], [], "bcaddcb", True),                                    # 3c~ S2
    (-1.0, [], [], "bcadcdb", True),                                   # 3d~ S2
    (-1.0, [], [], "bacdcdb", True),                                   # 3e~ S2
    (4.0, [], [], "bcdadcb", False),                                   # 3f~
    (6 * Z3 - 5, [], [], "bcdacdb", False),                            # 3g~
    (F(-3, 2), [], [], "bcbaddc", True),                               # 3h~ S2
    (F(-3, 2), [], [], "baccdbd", True),                               # 3i~ S2
    (F(1, 2), [], [], "bacddbc", True),                                # 3j~ S2
    (F(3, 2), [], [], "bacbddc", True),                                # 3k~ S2
    (2.0, [], [], "bacbdcd", True),                                    # 3l~ S2
    (1.0, [], [], "bacdbdc", True),                                    # 3m~ S2
    (-2.0, [], [], "badcdbc", True),                                   # 3n~ S2
    (F(-1, 2), [], [], "bcabddc", True),                               # 3o~ S2
    (F(-3, 2), [], [], "bcaddbc", True),                               # 3p~ S2
    (-3.0, [], [], "bcacdbd", True),                                   # 3q~ S2
    (3 * (2 * Z3 - 1), [], [], "bcadbdc", True),                       # 3r~ S2
    (1.0, [], [], "bcabdcd", True),                                    # 3s~ S2
    (-2.0, [], [], "bcbadcd", False),                                  # 3t~
    (-3.0, [], [], "bcdadbc", False),                                  # 3u~
    (3 * (2 * Z3 - 1), [], [], "bacdbcd", True),                       # 3v~ S2
    (2 * (3 * Z3 - 1), [], [], "bcdacbd", True),                       # 3w~ S2
    (2 * (3 * Z3 - 1), [], [], "bcadbcd", True),                       # 3x~ S2
    (2.0, [], [], "bcdabcd", False),                                   # 3y~
    (-4.0, [], [], "bcadcbd", True),                                   # 3z~ S2
]


def _contract(term, y, lam, m, free):
    c, lams, trs, chain, _ = term
    ops, sid, fid = [], {}, iter(range(10, 52))      # einsum integer labels must be < 52
    s = lambda ch: sid.setdefault(ch, len(sid))
    for L in lams:
        ops += [lam, [s(ch) for ch in L]]
    pref = 1.0
    for tr in trs:
        f = [next(fid) for _ in tr]
        for k, ch in enumerate(tr):
            ops += [y, [s(ch), f[k], f[(k + 1) % len(tr)]]]
        pref *= m
    out = [s(ch) for ch in free]
    if chain:
        f = [next(fid) for _ in range(len(chain) + 1)]
        for k, ch in enumerate(chain):
            ops += [y, [s(ch), f[k], f[k + 1]]]
        out += [f[0], f[-1]]
    ops.append(out)
    return c * pref * np.einsum(*ops, optimize=("greedy", 1e8))


def _sum(terms, y, lam, m, free):
    tot = 0
    for t in terms:
        X = _contract(t, y, lam, m, free)
        if t[4]:                                   # S2
            if t[3]:
                X = X + np.conj(np.swapaxes(X, -1, -2))
            else:
                X = X + X.T
        tot = tot + X
    return tot


def gamma_phi3(y, lam, m):
    G = _sum(GAMMA_PHI3, y, lam, m, "ab")
    return np.real(0.5 * (G + G.T))


def gamma_psi3(y, lam, m):
    G = _sum(GAMMA_PSI3, y, lam, m, "")
    return 0.5 * (G + G.conj().T)


def betat_y3(y, lam, m):
    B = _sum(BETA_Y3, y, lam, m, "a")
    return 0.5 * (B + np.conj(np.swapaxes(B, -1, -2)))


def beta3_y(y, lam, m):
    """Full three-loop Yukawa beta function beta_y^(3)a."""
    gp, gs = gamma_psi3(y, lam, m), gamma_phi3(y, lam, m)
    return (betat_y3(y, lam, m) + np.einsum("ij,ajk->aik", gp, y) + np.einsum("aij,jk->aik", y, gp)
            + np.einsum("ab,bij->aij", gs, y))
