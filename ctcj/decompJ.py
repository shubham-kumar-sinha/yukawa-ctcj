"""Independent subsets and exact rational decomposition of (complex-valued) flavour invariants."""
import numpy as np
import sympy as sp

from .net import evaluate


def eval_matrix(nets, ths):
    return np.array([[evaluate(n, th) for n in nets] for th in ths], dtype=complex)


def _rank(A, tol):
    if A.shape[1] == 0:
        return 0
    s = np.linalg.svd(A, compute_uv=False)
    return int((s > tol * s[0]).sum()) if s[0] > 0 else 0


def independent(names, M, tol=1e-9, ztol=1e-8):
    """Greedy: keep a column if it raises the rank (columns normalised).  Preference = order of names."""
    nrm = np.linalg.norm(M, axis=0)
    zero = nrm < ztol * np.median(nrm[nrm > 0])        # identically vanishing candidates (round-off only)
    keep = []
    for j in range(M.shape[1]):
        if zero[j]:
            continue
        A = M[:, keep + [j]] / nrm[keep + [j]]
        if _rank(A, tol) > len(keep):
            keep.append(j)
    return [names[j] for j in keep], M[:, keep]


def rational(z, tol=1e-9, maxden=10 ** 6):
    from fractions import Fraction
    re = Fraction(z.real).limit_denominator(maxden)
    im = Fraction(z.imag).limit_denominator(maxden)
    out = sp.Rational(re.numerator, re.denominator) + sp.I * sp.Rational(im.numerator, im.denominator)
    assert abs(complex(out) - z) < 1e-6 * max(1, abs(z)), (z, out)
    return out


def decompose(vals, names, M, rtol=1e-9):
    """vals ~ M @ c with complex c; returns {name: exact complex rational}."""
    nrm = np.linalg.norm(M, axis=0)
    sol = np.linalg.lstsq(M / nrm, vals, rcond=None)[0] / nrm
    res = np.linalg.norm(M @ sol - vals) / max(1e-300, np.linalg.norm(vals)) if np.linalg.norm(vals) else 0.0
    if res > rtol:
        raise ValueError(f"not in span (relative residual {res:.2e})")
    out = {}
    for n, s in zip(names, sol):
        if abs(s) > 1e-8:
            out[n] = rational(complex(s))
    # exact re-check
    chk = M @ np.array([complex(out.get(n, 0)) for n in names]) - vals
    assert np.linalg.norm(chk) <= 1e-7 * max(1, np.linalg.norm(vals)), "rationalisation failed"
    return out
