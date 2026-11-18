"""Two-sided Pade approximants in eps = 4 - d.

pade(a, b, m, n): R = P(eps)/Q(eps), Q(0) = 1, matching the 4 - eps coefficients a[0..K] at eps = 0 and the d = 2
data b[0..L] (coefficients in d - 2; empty if none) at eps = 2.  Returns (P, Q, has_pole_in_2<=d<=4).
all_pades(a, b): every pole-free [m, n] with m + n + 1 = len(a) + len(b), as (label, value at d = 3)
or, with functions=True, (label, callable of eps).
"""
import numpy as np
import sympy as sp


def pade(a, b, m, n):
    """a: 4-eps coefficients (x = 4 - d); b: d = 2 coefficients in u = d - 2 = 2 - x (may be empty)."""
    K, L = len(a) - 1, len(b) - 1
    assert m + n + 1 == K + 1 + L + 1
    rows, rhs = [], []
    # unknowns: p_0..p_m, q_1..q_n
    for k in range(K + 1):                 # [x^k] (P - A Q) = 0
        r = np.zeros(m + 1 + n)
        if k <= m:
            r[k] = 1
        for j in range(1, n + 1):
            if k - j >= 0:
                r[m + j] = -a[k - j]
        rows.append(r); rhs.append(a[k])
    if L >= 0:                             # [u^l] (P(2-u) - B(u) Q(2-u)) = 0
        def poly_in_u(c):                  # coefficients in u of sum_i c_i (2-u)^i
            out = np.zeros(len(c))
            for i, ci in enumerate(c):
                for l in range(i + 1):
                    out[l] += ci * float(sp.binomial(i, l)) * 2.0 ** (i - l) * (-1) ** l
            return out
        for l in range(L + 1):
            r = np.zeros(m + 1 + n)
            for i in range(m + 1):
                e = np.zeros(m + 1); e[i] = 1
                r[i] = poly_in_u(e)[l] if l <= m else 0.0
            for j in range(1, n + 1):
                e = np.zeros(n + 1); e[j] = 1
                qu = poly_in_u(e)
                r[m + j] = -sum(b[l - s] * qu[s] for s in range(min(l, n) + 1) if l - s >= 0 and s < len(qu))
            q0u = 1.0
            rows.append(r); rhs.append(b[l] * q0u)
    sol = np.linalg.solve(np.array(rows), np.array(rhs))
    P = np.polynomial.Polynomial(sol[:m + 1]); Q = np.polynomial.Polynomial(np.concatenate([[1.0], sol[m + 1:]]))
    roots = Q.roots() if n else []
    bad = any(abs(r.imag) < 1e-9 and -1e-9 <= r.real <= 2 + 1e-9 for r in roots)
    return P, Q, bad


def all_pades(a, b, functions=False):
    tot = len(a) + len(b)
    res = []
    for n in range(0, tot):
        m = tot - 1 - n
        try:
            P, Q, bad = pade(a, b, m, n)
        except np.linalg.LinAlgError:
            continue
        if not bad:
            res.append((f"[{m},{n}]", (lambda x, P=P, Q=Q: P(x) / Q(x)) if functions else P(1.0) / Q(1.0)))
    return res


