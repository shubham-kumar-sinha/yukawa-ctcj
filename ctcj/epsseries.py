"""Truncated Laurent series in ep with explicit log-Gamma expansions (sympy's Gamma series is slow).

log Gamma(1+x) = -gammaE x + sum_{k>=2} (-1)^k zeta_k x^k / k ;  zeta_k kept as symbols z2, z3, ...
"""
import sympy as sp

EP = sp.Symbol("ep")
ELL = sp.Symbol("ell")            # log(mu^2 x^2)
GAMMA_E, LOG4, LOGPI = sp.symbols("gammaE log4 logpi")
ZETA = {k: sp.Symbol(f"z{k}") for k in range(2, 10)}
ORDER = 6


def trunc(x, n=ORDER):
    x = sp.expand(x)
    return sum(x.coeff(EP, k) * EP ** k for k in range(-4, n + 1))


def mul(a, b):
    return trunc(sp.expand(a * b))


def exp_series(p):
    """exp(p) for p = O(ep)."""
    res, term = 1, 1
    for k in range(1, ORDER + 2):
        term = mul(term, p) / k
        res = trunc(res + term)
    return res


def inv_series(a):
    a = sp.expand(a)
    a0 = a.coeff(EP, 0)
    r = a / a0 - 1
    res, t = 1, 1
    for _ in range(ORDER + 1):
        t = mul(t, -r)
        res = trunc(res + t)
    return trunc(res / a0)


def lgam1(x):
    return -GAMMA_E * x + sum((-1) ** k * ZETA[k] * x ** k / k for k in range(2, ORDER + 2))
