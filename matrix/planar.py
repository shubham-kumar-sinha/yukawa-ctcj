"""Exact U(N) adjoint invariants via double-line (Fierz) contraction.
Generators X^a with tr(X^a X^b) = 1/2 delta, sum_a X^a_ij X^a_kl = 1/2 delta_il delta_jk (exact for U(N)).
A term is (coef, traces) with traces a tuple of tuples of tokens; tokens are int labels (each appears twice) or 'G'."""
from fractions import Fraction as Fr
from collections import defaultdict
import itertools
from formulas import CT2, CT3, CJ2, CJ3

class Gen:
    def __init__(self): self.n = 0
    def __call__(self): self.n += 1; return self.n

def y_terms(a, i, j):            # y^a_ij = 1/2 [tr(a i j) + tr(a j i)]
    return [(Fr(1, 2), ((a, i, j),)), (Fr(1, 2), ((a, j, i),))]

def T_terms(i, j):               # T_ij = 2 [tr(i G j) - tr(i j G)]   (overall sign irrelevant)
    return [(Fr(2), ((i, 'G', j),)), (Fr(-2), ((i, j, 'G'),))]

def t_terms(a, b):               # t_ab = 2 i [tr(a G b) - tr(a b G)]; the factor i is tracked separately
    return [(Fr(2), ((a, 'G', b),)), (Fr(-2), ((a, b, 'G'),))]

def L1_terms(a, b, c, d):        # 4 * sum of 6 orderings
    return [(Fr(4), ((a,) + p,)) for p in [(b, c, d), (b, d, c), (c, b, d), (c, d, b), (d, b, c), (d, c, b)]]

def L2_terms(a, b, c, d):        # 8 * [tr(ab)tr(cd) + tr(ad)tr(bc) + tr(ac)tr(bd)]
    return [(Fr(8), ((a, b), (c, d))), (Fr(8), ((a, d), (b, c))), (Fr(8), ((a, c), (b, d)))]

def product(factor_lists):
    out = []
    for combo in itertools.product(*factor_lists):
        c = Fr(1); trs = ()
        for cc, tt in combo: c *= cc; trs += tt
        out.append((c, trs))
    return out

def contract(term):
    """fully contract; returns dict power_of_N -> coefficient (with tr(GG)=1, tr(G)=0)."""
    c, trs = term
    trs = [list(t) for t in trs]
    while True:
        lab = None
        for t in trs:
            for x in t:
                if x != 'G': lab = x; break
            if lab is not None: break
        if lab is None: break
        pos = [(k, p) for k, t in enumerate(trs) for p, x in enumerate(t) if x == lab]
        assert len(pos) == 2, (lab, trs)
        (k1, p1), (k2, p2) = pos
        c *= Fr(1, 2)
        if k1 == k2:
            w = trs[k1]; w = w[p1:] + w[:p1]; q = (p2 - p1) % len(w)
            B, C = w[1:q], w[q + 1:]
            trs[k1] = B; trs.append(C)
        else:
            w1 = trs[k1]; w1 = w1[p1 + 1:] + w1[:p1]     # A with X removed (rotated so X was at end)
            w2 = trs[k2]; w2 = w2[p2 + 1:] + w2[:p2]
            new = w1 + w2
            trs = [t for k, t in enumerate(trs) if k not in (k1, k2)] + [new]
    npow = 0
    for t in trs:
        if len(t) == 0: npow += 1
        elif len(t) == 1: return {}
        elif len(t) == 2: pass         # tr(GG) = 1
        else: raise ValueError(t)
    return {npow: c}

def evaluate_structure(st, lamchoice):
    """st: structure tuple from formulas; lamchoice: list of 'L1'/'L2' per lam factor.
    Returns polynomial in N (dict), including the i^2 from t factors."""
    g = Gen(); names = {}
    def lab(ch):
        if ch not in names: names[ch] = g()
        return names[ch]
    factors = []; nt = 0; li = 0
    for f in st:
        if f[0] == 'lam':
            idx = [lab(ch) for ch in f[1]]
            factors.append((L1_terms if lamchoice[li] == 'L1' else L2_terms)(*idx)); li += 1
        elif f[0] == 't':
            factors.append(t_terms(*[lab(ch) for ch in f[1]])); nt += 1
        else:
            w = f[1]; fl = [g() for _ in w]
            for k, ch in enumerate(w):
                i, j = fl[k], fl[(k + 1) % len(w)]
                factors.append(T_terms(i, j) if ch == 'T' else y_terms(lab(ch), i, j))
    poly = defaultdict(Fr)
    for term in product(factors):
        for p, v in contract(term).items(): poly[p] += v
    sign = (-1) ** (nt // 2) if nt % 2 == 0 else None
    assert sign is not None
    return {p: v * sign for p, v in poly.items() if v != 0}

def evaluate_formula(formula):
    """returns list of (rat, z3, ny, lamcombo, poly)"""
    out = []
    for rat, z, st in formula:
        k = sum(1 for f in st if f[0] == 'lam')
        ny = sum(sum(1 for ch in f[1] if ch != 'T') for f in st if f[0] == 'tr')
        for combo in itertools.product(['L1', 'L2'], repeat=k):
            out.append((rat, z, ny, combo, evaluate_structure(st, list(combo))))
    return out
