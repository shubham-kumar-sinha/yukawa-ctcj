"""Vectorised JOS beta functions (same formulas as ctcj.theory.Theory, Tr = mult * trace)."""
import itertools

import numpy as np


def E(s, *ops):
    return np.einsum(s, *ops, optimize=("greedy", 1e8))


def beta1_y(y, m):
    gpsi = 0.5 * E("bij,bjk->ik", y, y)
    gphi = 0.5 * m * E("aij,bji->ab", y, y)
    return (2 * E("bij,ajk,bkl->ail", y, y, y) + E("ij,ajk->aik", gpsi, y) + E("aij,jk->aik", y, gpsi)
            + E("ab,bij->aij", gphi, y))


def beta2_y(y, lam, m):
    yy = E("aij,bjk->abik", y, y)                          # (y^a y^b)_ik
    gphi = (E("acde,bcde->ab", lam, lam) / 12
            - 0.75 * m * E("abik,ccki->ab", yy, yy)
            - 0.5 * m * E("acij,bcji->ab", yy, yy))
    gphi = 0.5 * (gphi + gphi.T)
    H = m * E("aij,bji->ab", y, y)
    gpsi = -0.375 * E("abik,ab->ik", yy, H) - 0.125 * E("abij,bajk->ik", yy, yy)
    yyy = E("abij,cjk->abcik", yy, y)
    bt = (-2 * E("abcd,bcdik->aik", lam, yyy)
          - E("bij,ajk,ckl,bc->ail", y, y, y, H)
          - E("bij,ajk,cckl,blm->aim", y, y, yy, y) - E("bij,ccjk,akl,blm->aim", y, yy, y, y)
          - 2 * E("bcij,ajk,cbkl->ail", yy, y, yy)
          + 2 * E("bcij,ajk,bckl->ail", yy, y, yy))
    return bt + E("ij,ajk->aik", gpsi, y) + E("aij,jk->aik", y, gpsi) + E("ab,bij->aij", gphi, y)


def beta1_lam(y, lam, m):
    H = (m * E("aij,bji->ab", y, y)).real
    t1 = E("abef,cdef->abcd", lam, lam) + E("acef,bdef->abcd", lam, lam) + E("adef,bcef->abcd", lam, lam)
    t2 = 0.5 * (E("ebcd,ae->abcd", lam, H) + E("aecd,be->abcd", lam, H)
                + E("abed,ce->abcd", lam, H) + E("abce,de->abcd", lam, H))
    W = m * E("aij,bjk,ckl,dli->abcd", y, y, y, y).real
    t3 = -0.5 * sum(np.transpose(W, p) for p in itertools.permutations(range(4)))
    return t1 + t2 + t3


# ---------------------------------------------------------------- two-loop beta_lambda (JOS 2301.10903 sec. 6)
_PERMS = list(itertools.permutations(range(4)))


def _sym(X, n):
    """S_n X of JOS: the n distinct terms needed to symmetrise X over a,b,c,d (= n/24 * sum over S_4)."""
    return n / 24 * sum(np.transpose(X, p) for p in _PERMS)


def gamma_phi2(y, lam, m):
    yy = E("aij,bjk->abik", y, y)
    g = (E("acde,bcde->ab", lam, lam) / 12
         - 0.75 * m * E("abik,ccki->ab", yy, yy)
         - 0.5 * m * E("acij,bcji->ab", yy, yy))
    return 0.5 * (g + g.T).real


def beta2_lam(y, lam, m):
    """Two-loop beta_lambda: tilde-beta^(2) (coefficients 2a..2g of JOS) + S_4 gamma_phi^(2) lambda."""
    H = (m * E("aij,bji->ab", y, y)).real
    tr = lambda s, *ops: (m * E(s, *ops)).real
    t = -1 * _sym(E("abef,cfgh,degh->abcd", lam, lam, lam), 6)                     # 2a
    t += -1 * _sym(E("abef,cdeg,fg->abcd", lam, lam, H), 3)                          # 2b
    t += 2 * _sym(E("abef,efcd->abcd", lam, tr("eij,cjk,fkl,dli->ecfd", y, y, y, y).transpose(0, 2, 1, 3)), 6)  # 2d: tr(y^e y^c y^f y^d)
    t += 2 * _sym(tr("aij,ejk,ekl,blm,cmn,dni->abcd", y, y, y, y, y, y), 12)       # 2e: tr(y^a y^e y^e y^b y^c y^d)
    t += 4 * _sym(tr("aij,ejk,bkl,clm,dmn,eni->abcd", y, y, y, y, y, y), 12)       # 2f: tr(y^a y^e y^b y^c y^d y^e)
    t += 4 * _sym(tr("eij,ajk,bkl,elm,cmn,dni->abcd", y, y, y, y, y, y), 6)        # 2g: tr(y^e y^a y^b y^e y^c y^d)
    g2 = gamma_phi2(y, lam, m)
    t += E("ae,ebcd->abcd", g2, lam) + E("be,aecd->abcd", g2, lam) + E("ce,abed->abcd", g2, lam) + E("de,abce->abcd", g2, lam)
    return t
