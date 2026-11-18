"""Planar (large-N) limit of C_T and C_J at the supersymmetric point of the SU(N) adjoint model, in closed form.

The invariants are evaluated exactly in N by double-line (Fierz) contraction (planar.py), and the limit N -> oo is taken
coefficient by coefficient.  By the large-N equivalence of the three matrix models, the result holds for all of them."""
import sympy as sp
import planar
from formulas import CT2, CT3, CJ2, CJ3, CT3_EVEN, CJ3_EVEN
eps, z3, N = sp.symbols('epsilon zeta3 N', positive=True)
b = eps/5 + sp.Rational(9,125)*eps**2 + sp.Rational(243,12500)*eps**3      # large-N SU(N) [susy]
Y, G = 16*b/N, {'L1': b/N, 'L2': -b/N**2}
def total(formula):
    s = 0
    for rat, z, ny, combo, poly in planar.evaluate_formula(formula):
        val = sum(sp.Rational(c.numerator, c.denominator)*N**p for p, c in poly.items())
        cpl = Y**(ny//2)
        for cname in combo: cpl *= G[cname]
        s += (sp.Rational(rat.numerator, rat.denominator) + sp.Rational(z.numerator, z.denominator)*z3)*val*cpl
    return sp.expand(s)
def lim(expr):
    expr = sp.series(expr, eps, 0, 4).removeO()
    return sp.expand(sum(sp.limit(sp.expand(expr).coeff(eps, k), N, sp.oo)*eps**k for k in range(4)))
CT = lim(1 + (total(CT2) + total(CT3))/(N**2*(5 - eps)/2))                  # as printed, with the odd fermion loops
CT_EVEN = lim(1 + (total(CT2) + total(CT3_EVEN))/(N**2*(5 - eps)/2))        # four-dimensional Dirac algebra only
from fractions import Fraction as Fr
def poly_of(st):
    p = planar.evaluate_structure(st, [])
    return sum(sp.Rational(c.numerator, c.denominator)*N**k for k, c in p.items())
trTT = poly_of((('tr','TT'),)); tt = poly_of((('t','ab'),('t','ab')))
CJ = lim(1 + (total(CJ2) + total(CJ3))/(trTT + tt/(2 - eps)))
CJ_EVEN = lim(1 + (total(CJ2) + total(CJ3_EVEN))/(trTT + tt/(2 - eps)))
if __name__ == '__main__':
    print('free: Tr(TT) =', sp.factor(trTT), '  t.t =', sp.factor(tt))
    print('large-N C_T/C_T,free =', CT, '  ~', [sp.N(CT.coeff(eps,k).subs(z3, sp.zeta(3)), 6) for k in range(4)])
    print('large-N C_J/C_J,free =', CJ, '  ~', [sp.N(CJ.coeff(eps,k).subs(z3, sp.zeta(3)), 6) for k in range(4)])
