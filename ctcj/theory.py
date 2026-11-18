"""Evaluate C_T and C_J through O(eps^2) for any user-specified scalar-fermion theory.

A theory is given by coupling bases
    y^a      = sum_i g_i  Y[i][a]          (Y[i]: array (n_s, n_f, n_f), hermitian in flavour)
    lambda   = sum_j l_j  L[j]             (L[j]: totally symmetric array (n_s,)*4)
with couplings rescaled y -> 4 pi y, lambda -> 16 pi^2 lambda.  Fermion traces are Tr = mult * trace
(mult lets one emulate N copies without large matrices; Tr is normalised as in JOS, so that
gamma_phi^(1)ab = 1/2 Tr(y^a y^b); one four-component Dirac fermion with g sigma psibar psi has Tr 1 = 4).

Fixed points: g = sqrt(eps)(a + eps b), l = eps c.  One loop fixes (a, c) (nonlinear, solved by Newton
from many random starts); the O(eps^2) shift b of the Yukawa couplings follows linearly from the
two-loop beta function of JOS (arXiv:2301.10903).  lambda is needed only at O(eps).

C_T, C_J use the general fixed-point formulas (ctcj.formulas).
"""
import itertools

import numpy as np

from .formulas import CT, CJ

RNG = np.random.default_rng(1)


class Theory:
    def __init__(self, Y, L=(), mult=1.0, name="theory"):
        self.Y = [np.asarray(y, dtype=complex) for y in Y]
        self.L = [np.asarray(l, dtype=float) for l in L]
        self.ns, self.nf = self.Y[0].shape[0], self.Y[0].shape[1]
        self.mult = float(mult)
        self.name = name
        for y in self.Y:
            for a in range(self.ns):
                assert np.allclose(y[a], y[a].conj().T), "Yukawa matrices must be hermitian"
        for l in self.L:
            for p in itertools.permutations(range(4)):
                assert np.allclose(l, np.transpose(l, p)), "quartic must be totally symmetric"

    # ---------------------------------------------------------------- tensors
    def tr(self, M):
        return self.mult * np.trace(M)

    def y_of(self, g):
        return sum(gi * Yi for gi, Yi in zip(g, self.Y))

    def lam_of(self, l):
        return sum(li * Li for li, Li in zip(l, self.L)) if self.L else np.zeros((self.ns,) * 4)

    # ---------------------------------------------------------------- beta functions (JOS)
    def beta1_y(self, y):
        ns = self.ns
        gpsi = 0.5 * sum(y[b] @ y[b] for b in range(ns))
        gphi = np.array([[0.5 * self.tr(y[a] @ y[b]) for b in range(ns)] for a in range(ns)])
        return np.array([2 * sum(y[b] @ y[a] @ y[b] for b in range(ns)) + gpsi @ y[a] + y[a] @ gpsi
                         + sum(gphi[a, b] * y[b] for b in range(ns)) for a in range(ns)])

    def beta2_y(self, y, lam):
        ns, tr = self.ns, self.tr
        rg = range(ns)
        gphi = np.array([[np.einsum("cde,cde->", lam[a], lam[b]) / 12
                          - 0.75 * sum(tr(y[a] @ y[b] @ y[c] @ y[c]) for c in rg)
                          - 0.5 * sum(tr(y[a] @ y[c] @ y[b] @ y[c]) for c in rg) for b in rg] for a in rg])
        gphi = 0.5 * (gphi + gphi.T)
        gpsi = (-0.375 * sum(y[a] @ y[b] * tr(y[a] @ y[b]) for a in rg for b in rg)
                - 0.125 * sum(y[a] @ y[b] @ y[b] @ y[a] for a in rg for b in rg))
        out = []
        for a in rg:
            bt = (-2 * sum(lam[a, b, c, d] * y[b] @ y[c] @ y[d] for b in rg for c in rg for d in rg)
                  - sum(y[b] @ y[a] @ y[c] * tr(y[b] @ y[c]) for b in rg for c in rg)
                  - sum(y[b] @ y[a] @ y[c] @ y[c] @ y[b] + y[b] @ y[c] @ y[c] @ y[a] @ y[b] for b in rg for c in rg)
                  - 2 * sum(y[b] @ y[c] @ y[a] @ y[c] @ y[b] for b in rg for c in rg)
                  + 2 * sum(y[b] @ y[c] @ y[a] @ y[b] @ y[c] for b in rg for c in rg))
            out.append(bt + gpsi @ y[a] + y[a] @ gpsi + sum(gphi[a, b] * y[b] for b in rg))
        return np.array(out)

    def beta1_lam(self, y, lam):
        ns, tr = self.ns, self.tr
        rg = range(ns)
        H = np.array([[tr(y[a] @ y[b]).real for b in rg] for a in rg])
        t1 = np.einsum("abef,cdef->abcd", lam, lam)
        t1 = t1 + np.einsum("acef,bdef->abcd", lam, lam) + np.einsum("adef,bcef->abcd", lam, lam)
        t2 = 0.5 * (np.einsum("ebcd,ae->abcd", lam, H) + np.einsum("aecd,be->abcd", lam, H)
                    + np.einsum("abed,ce->abcd", lam, H) + np.einsum("abce,de->abcd", lam, H))
        W = self.mult * np.einsum("aij,bjk,ckl,dli->abcd", y, y, y, y).real
        t3 = -0.5 * sum(np.transpose(W, p) for p in itertools.permutations(range(4)))
        return t1 + t2 + t3

    # ---------------------------------------------------------------- projections onto coupling bases
    def _proj(self, T, basis):
        if not basis:
            return np.zeros(0), 0.0
        A = np.array([b.ravel() for b in basis]).T
        sol, *_ = np.linalg.lstsq(A, T.ravel(), rcond=None)
        res = np.linalg.norm(A @ sol - T.ravel()) / max(1.0, np.linalg.norm(T))
        return sol.real, res

    def one_loop(self, x):
        nY = len(self.Y)
        a, c = x[:nY], x[nY:]
        y, lam = self.y_of(a), self.lam_of(c)
        fy, ry = self._proj(-0.5 * y + self.beta1_y(y), self.Y)
        fl, rl = self._proj(-lam + self.beta1_lam(y, lam), self.L)
        return np.concatenate([fy, fl]), max(ry, rl)

    # ---------------------------------------------------------------- fixed points
    def fixed_points(self, starts=60, tol=1e-12):
        nY, nL = len(self.Y), len(self.L)
        n = nY + nL
        sols = []
        for _ in range(starts):
            x = RNG.normal(size=n) * RNG.choice([0.1, 0.3, 1.0])
            for _ in range(100):
                f, _ = self.one_loop(x)
                J = np.zeros((n, n))
                for k in range(n):
                    h = np.zeros(n); h[k] = 1e-7
                    J[:, k] = (self.one_loop(x + h)[0] - self.one_loop(x - h)[0]) / 2e-7
                dx = np.linalg.lstsq(J, -f, rcond=None)[0]
                x = x + dx
                if np.linalg.norm(dx) < tol:
                    break
            f, res = self.one_loop(x)
            if np.linalg.norm(f) < 1e-10 and res < 1e-8:
                sols.append(x)
        # deduplicate by physical invariants
        uniq = {}
        for x in sols:
            a, c = x[:nY], x[nY:]
            y, lam = self.y_of(a), self.lam_of(c)
            key = (round(float(sum(self.tr(y[i] @ y[i]).real for i in range(self.ns))), 8),
                   round(float((lam ** 2).sum()), 8), round(float(lam.sum()), 8))
            uniq.setdefault(key, x)
        out = []
        for x in uniq.values():
            fp = self._second_order(x)
            fp["stable"], fp["eigs"] = self._stability(x)
            out.append(fp)
        out.sort(key=lambda f: (not f["stable"], -f["yy"]))
        return out

    def _stability(self, x):
        n = len(x)
        J = np.zeros((n, n))
        for k in range(n):
            h = np.zeros(n); h[k] = 1e-6
            J[:, k] = (self.one_loop(x + h)[0] - self.one_loop(x - h)[0]) / 2e-6
        ev = np.linalg.eigvals(J)
        return bool(np.all(ev.real > 1e-7)), np.sort(ev.real)

    def _second_order(self, x):
        nY = len(self.Y)
        a, c = x[:nY], x[nY:]
        y, lam = self.y_of(a), self.lam_of(c)
        # linear map b -> P_Y[-b/2 Y + D beta1(a)[b]];  D exact for a cubic via 4-point stencil
        def Dbeta(bvec):
            yb = self.y_of(bvec)
            f = lambda t: self.beta1_y(y + t * yb)
            return (8 * (f(1) - f(-1)) - (f(2) - f(-2))) / 12
        M = np.zeros((nY, nY))
        for k in range(nY):
            e = np.zeros(nY); e[k] = 1
            M[:, k] = self._proj(-0.5 * self.y_of(e) + Dbeta(e), self.Y)[0]
        rhs = -self._proj(self.beta2_y(y, lam), self.Y)[0]
        b = np.linalg.lstsq(M, rhs, rcond=None)[0]
        yy = float(sum(self.tr(y[i] @ y[i]).real for i in range(self.ns)))
        return dict(a=a, b=b, c=c, yy=yy)

    # ---------------------------------------------------------------- O(eps^3)
    @staticmethod
    def _taylor(fun, deg):
        """Taylor coefficients c_0..c_deg of fun(t), a polynomial of degree <= deg in real t (tensor or scalar
        valued), by an exact least-squares fit on deg + 5 Chebyshev nodes.  (Real nodes, because the beta
        functions take real/hermitian parts, which is not analytic in complex t.)"""
        K = deg + 5
        x = np.cos(np.pi * (np.arange(K) + 0.5) / K)
        vals = [np.asarray(fun(t)) for t in x]
        V = np.vander(x, deg + 1, increasing=True)
        coef = np.linalg.lstsq(V, np.array([v.ravel() for v in vals]), rcond=None)[0]
        return [coef[j].reshape(vals[0].shape) for j in range(deg + 1)]

    def third_order(self, fp):
        """Add lambda* to O(eps^2) ('e') and y* to O(eps^3) ('d') to a fixed point from fixed_points():
            y* = sqrt(eps) (a + eps b + eps^2 d) . Y,   lambda* = eps (c + eps e) . L
        using the two-loop beta_lambda and the three-loop beta_y (ctcj.betasJ, ctcj.betas3)."""
        from . import betasJ as BJ, betas3 as B3
        nY, nL = len(self.Y), len(self.L)
        m = self.mult
        Y1, Y2, L1 = self.y_of(fp["a"]), self.y_of(fp["b"]), self.lam_of(fp["c"])
        # lambda at second order:  -L2 + D bl1[Y2, L2] + bl2(Y1, L1) = 0
        bl1 = lambda y, lam: BJ.beta1_lam(y, lam, m)
        def dbl1(yv, lv):
            c = self._taylor(lambda t: bl1(Y1 + t * yv, L1 + t * lv), 4)
            return c[1].real
        if nL:
            ML = np.zeros((nL, nL))
            for k in range(nL):
                ev = np.zeros(nL); ev[k] = 1
                ML[:, k] = self._proj(-self.lam_of(ev) + dbl1(np.zeros_like(Y1), self.lam_of(ev)), self.L)[0]
            rhsL = -self._proj(dbl1(Y2, np.zeros_like(L1)) + BJ.beta2_lam(Y1, L1, m), self.L)[0]
            e = np.linalg.lstsq(ML, rhsL, rcond=None)[0]
        else:
            e = np.zeros(0)
        L2 = self.lam_of(e) if nL else np.zeros_like(L1)
        # y at third order: -Y3/2 + Db1[Y3] + 1/2 D^2 b1[Y2,Y2] + Db2[Y2, L2] + b3(Y1, L1) = 0
        b1 = lambda y: BJ.beta1_y(y, m)
        def Db1(v):
            return self._taylor(lambda t: b1(Y1 + t * v), 3)[1]
        M = np.zeros((nY, nY))
        for k in range(nY):
            ev = np.zeros(nY); ev[k] = 1
            M[:, k] = self._proj(-0.5 * self.y_of(ev) + Db1(self.y_of(ev)), self.Y)[0]
        d2 = self._taylor(lambda t: b1(Y1 + t * Y2), 3)[2]
        db2 = self._taylor(lambda t: BJ.beta2_y(Y1 + t * Y2, L1 + t * L2, m), 5)[1]
        rhs = -self._proj(d2 + db2 + B3.beta3_y(Y1, L1, m), self.Y)[0]
        d = np.linalg.lstsq(M, rhs, rcond=None)[0]
        out = dict(fp); out.update(d=d, e=e)
        return out

    def _series(self, fp3, kind, current=None, odd=False):
        """C/C_free = 1 + c1 eps + c2 eps^2 + c3 eps^3 at a fixed point from third_order().
        odd=True adds the odd fermion-loop term of 3d-only theories (ctcj.formulas4.KAPPA_T_ODD, KAPPA_J_ODD)."""
        from .eps3models import load_coefficients, name_net, degrees
        from .net import evaluate, words_net
        Y = lambda t: self.y_of(fp3["a"]) + t * self.y_of(fp3["b"]) + t * t * self.y_of(fp3["d"])
        if self.L:
            Lm = lambda t: self.lam_of(fp3["c"]) + t * self.lam_of(fp3["e"])
        else:
            Lm = lambda t: np.zeros((self.ns,) * 4)
        T, tt = (np.zeros((self.nf, self.nf), complex), np.zeros((self.ns, self.ns))) if current is None else \
            (np.asarray(current[0], complex), np.asarray(current[1], float))
        th = lambda t: dict(y=Y(t), lam=Lm(t), T=T, t=tt, mult=self.mult)
        X = np.zeros(4)
        for k, cf in load_coefficients(kind).items():
            ny, nl = degrees(k)
            w = ny // 2 + nl
            if w > 3:
                continue
            net = name_net(k)
            co = self._taylor(lambda t: evaluate(net, th(t)), 2 * ny + nl)    # full degree: no aliasing
            for j in range(3 - w + 1):
                X[w + j] += float(cf) * co[j].real
        if odd:                                     # odd fermion loops of 3d-only theories (ctcj.formulas4)
            from .formulas4 import KAPPA_T_ODD, KAPPA_J_ODD, odd_invariant_T, odd_invariants_J, value
            if kind == "T":
                X[3] += value(KAPPA_T_ODD) * odd_invariant_T(Y(0), self.mult)
            else:
                I0, I1 = odd_invariants_J(Y(0), T, self.mult)
                X[3] += value(KAPPA_J_ODD[0]) * I0 + value(KAPPA_J_ODD[1]) * I1
        tr1 = self.tr(np.eye(self.nf)).real
        if kind == "T":
            F = np.array([self.ns + 1.5 * tr1, -0.5 * tr1, 0, 0])                       # n_s + (d-1)/2 Tr1
        else:
            trTT, t2 = self.tr(T @ T).real, float((tt ** 2).sum())
            F = np.array([trTT + t2 / 2, t2 / 4, t2 / 8, t2 / 16])                       # Tr(TT) + t.t/(2-eps)
        num = F + X                                                                      # series division
        r = np.zeros(4)
        for n in range(4):
            r[n] = (num[n] - sum(r[k] * F[n - k] for k in range(n))) / F[0]
        return r

    def CT3(self, fp, odd=False):
        """(1, c1, c2, c3): C_T/C_T,free through eps^3 at fixed point fp (from fixed_points()).  odd=True: include
        the odd fermion-loop term, for theories without a four-dimensional counterpart (real symmetric y only)."""
        return self._series(self.third_order(fp), "T", odd=odd)

    def CJ3(self, fp, T, t, odd=False):
        """(1, c1, c2, c3): C_J/C_J,free through eps^3 for the current (T, t).  odd=True: include the odd
        fermion-loop term, for theories without a four-dimensional counterpart (real symmetric y only)."""
        assert self.check_current(T, t, fp), "current violates the Ward identity at this fixed point"
        return self._series(self.third_order(fp), "J", (T, t), odd=odd)

    # ---------------------------------------------------------------- invariants
    def _words(self, y, T=None):
        ns, tr = self.ns, self.tr
        rg = range(ns)
        W = {"Tr(aa)": sum(tr(y[a] @ y[a]) for a in rg),
             "Tr(ab)Tr(ab)": sum(tr(y[a] @ y[b]) ** 2 for a in rg for b in rg),
             "Tr(aabb)": sum(tr(y[a] @ y[a] @ y[b] @ y[b]) for a in rg for b in rg),
             "Tr(abab)": sum(tr(y[a] @ y[b] @ y[a] @ y[b]) for a in rg for b in rg)}
        if T is not None:
            def w(pat):
                tot = 0
                for a in rg:
                    for b in rg:
                        M = np.eye(self.nf, dtype=complex)
                        for ch in pat:
                            M = M @ {"T": T, "a": y[a], "b": y[b]}[ch]
                        tot += tr(M)
                return tot
            for pat in ["TTaabb", "TTabab", "TTabba", "TaTabb", "TaTbab", "TabTba"]:
                W[f"Tr({pat})"] = w(pat)
            W["Tr(TTaa)"] = sum(tr(T @ T @ y[a] @ y[a]) for a in rg)
            W["Tr(TaTa)"] = sum(tr(T @ y[a] @ T @ y[a]) for a in rg)
            W["Tr(TTab)Tr(ab)"] = sum(tr(T @ T @ y[a] @ y[b]) * tr(y[a] @ y[b]) for a in rg for b in rg)
            W["Tr(TaTb)Tr(ab)"] = sum(tr(T @ y[a] @ T @ y[b]) * tr(y[a] @ y[b]) for a in rg for b in rg)
        return {k: complex(v).real for k, v in W.items()}

    def check_current(self, T, t, fp=None):
        """Ward identity i[y^a,T] = t_ab y^b and t.lambda = 0 -- for the couplings at the fixed point fp
        if given, otherwise for every basis element."""
        T, t = np.asarray(T, complex), np.asarray(t, float)
        ok = True
        Ys = [self.y_of(fp["a"]), self.y_of(fp["b"])] if fp is not None else self.Y
        Ls = [self.lam_of(fp["c"])] if fp is not None else self.L
        for Yi in Ys:
            for a in range(self.ns):
                ok &= np.allclose(1j * (Yi[a] @ T - T @ Yi[a]), sum(t[a, b] * Yi[b] for b in range(self.ns)))
        for Li in Ls:
            v = (np.einsum("ae,ebcd->abcd", t, Li) + np.einsum("be,aecd->abcd", t, Li)
                 + np.einsum("ce,abed->abcd", t, Li) + np.einsum("de,abce->abcd", t, Li))
            ok &= np.allclose(v, 0)
        return bool(ok)

    # ---------------------------------------------------------------- C_T and C_J
    def CT(self, fp):
        """Return (C_T/C_T,free) = 1 + c1 eps + c2 eps^2 as (c1, c2); also absolute coefficients."""
        a, b, c = fp["a"], fp["b"], fp["c"]
        ya, yb, lam = self.y_of(a), self.y_of(b), self.lam_of(c)
        Wa = self._words(ya)
        trab = sum(self.tr(ya[i] @ yb[i]).real for i in range(self.ns))
        tr1 = self.tr(np.eye(self.nf)).real
        X1 = CT["Tr(aa)"] * Wa["Tr(aa)"]
        X2 = (CT["Tr(aa)"] * 2 * trab + sum(CT[k] * Wa[k] for k in ("Tr(ab)Tr(ab)", "Tr(aabb)", "Tr(abab)"))
              + CT["lam.lam"] * float((lam ** 2).sum()))
        F0, F1 = self.ns + 1.5 * tr1, -0.5 * tr1          # free part n_s + (3 - eps)/2 Tr1
        return X1 / F0, X2 / F0 - X1 * F1 / F0 ** 2

    def CJ(self, fp, T, t):
        """C_J/C_J,free = 1 + c1 eps + c2 eps^2 for the current (T, t)."""
        assert self.check_current(T, t, fp), "current violates the Ward identity at this fixed point"
        T, t = np.asarray(T, complex), np.asarray(t, float)
        a, b, c = fp["a"], fp["b"], fp["c"]
        ya, yb, lam = self.y_of(a), self.y_of(b), self.lam_of(c)
        Wa = self._words(ya, T)
        ns = self.ns
        cross_a = sum(self.tr(T @ T @ (ya[i] @ yb[i] + yb[i] @ ya[i])).real for i in range(ns))
        X1 = CJ["Tr(TTaa)"] * Wa["Tr(TTaa)"] + CJ["Tr(TaTa)"] * Wa["Tr(TaTa)"]
        cross_b = sum(self.tr(T @ ya[i] @ T @ yb[i] + T @ yb[i] @ T @ ya[i]).real for i in range(ns))
        X2 = (CJ["Tr(TTaa)"] * cross_a + CJ["Tr(TaTa)"] * cross_b
              + sum(CJ[k] * Wa[k] for k in CJ if k in Wa and k not in ("Tr(TTaa)", "Tr(TaTa)"))
              + CJ["t t lam lam"] * float(np.einsum("ae,bf,abcd,efcd->", t, t, lam, lam)))
        trTT, tt = self.tr(T @ T).real, float((t ** 2).sum())
        F0, F1 = trTT + tt / 2, tt / 4                     # Tr(TT) + t.t/(d-2),  1/(2-eps) = 1/2 + eps/4 + ...
        return X1 / F0, X2 / F0 - X1 * F1 / F0 ** 2

    def report(self, currents=None, show_unstable=True):
        lines = [f"== {self.name}  (n_s={self.ns}, Tr1={self.tr(np.eye(self.nf)).real:g}; fixed points related by coupling sign flips are merged)"]
        for fp in self.fixed_points():
            st = fp["stable"]
            if not st and not show_unstable:
                continue
            c1, c2 = self.CT(fp)
            lines.append(f"  fixed point: y* ~ sqrt(eps)*{np.round(fp['a'], 5)}, lambda* ~ eps*{np.round(fp['c'], 5)}"
                         f"   [{'stable' if st else 'unstable'}]")
            lines.append(f"    C_T/C_T,free = 1 {c1:+.6f} eps {c2:+.6f} eps^2")
            for nm, (T, t) in (currents or {}).items():
                if not self.check_current(T, t, fp):
                    lines.append(f"    C_J [{nm}]: symmetry broken at this fixed point")
                    continue
                d1, d2 = self.CJ(fp, T, t)
                lines.append(f"    C_J/C_J,free [{nm}] = 1 {d1:+.6f} eps {d2:+.6f} eps^2")
        return "\n".join(lines)


def O_n_quartic(n):
    """lambda_abcd = d_ab d_cd + d_ac d_bd + d_ad d_bc."""
    d = np.eye(n)
    return (np.einsum("ab,cd->abcd", d, d) + np.einsum("ac,bd->abcd", d, d) + np.einsum("ad,bc->abcd", d, d))
