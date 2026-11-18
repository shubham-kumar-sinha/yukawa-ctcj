#!/usr/bin/env python3
"""C_T at O(eps^3) from data/diagrams_T.json (1-3 loops) and data/diagrams4_T.json (4 loops).
Checks: pole/log cancellation; O(n) Wilson-Fisher eps^3 (Ising -233/8748); large-N C_T1 eps^3 term vs
the exact Gross-Neveu formula of Diab et al.  Writes results/coefficients4_T.json."""
import json, os, sys, itertools, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import sympy as sp
from ctcj.assemble4 import assemble_T4
from ctcj.forcer import EP, Z3, Z4, Z5

ROOT = os.path.join(os.path.dirname(__file__), "..")
loc = {"ep": EP, "z3": Z3, "z4": Z4, "z5": Z5, "I": sp.I}
def load(path):
    recs = json.load(open(path))
    for r in recs:
        r["X"] = sp.sympify(r["X"], locals=loc)
        r["decomp"] = {k: sp.sympify(v, locals=loc) for k, v in r["decomp"].items()}
    return recs

t0 = time.time()
res = assemble_T4(load(os.path.join(ROOT, "data", "diagrams_T.json")), load(os.path.join(ROOT, "data", "diagrams4_T.json")))
f = res["coef"]
print(f"assembled in {time.time()-t0:.0f}s")
checks = []
bad = {k: v for k, v in res["poles"].items() if any(sp.simplify(p) != 0 for p in v)}
checks.append(("all poles cancel per invariant (four-loop renormalisation)", not bad, str(list(bad)[:5])))
checks.append(("log(mu^2 x^2) drops at fixed points", res["logs_ok"], ""))

# --- O(n) Wilson-Fisher: lambda_abcd = l S_abcd,  l* = eps/(n+8) + 3(3n+14) eps^2/(n+8)^3
eps = sp.Symbol("epsilon")
ok_On, det = True, []
for n in (1, 2, 3, 4):
    d = np.eye(n)
    Sn = np.einsum("ab,cd->abcd", d, d) + np.einsum("ac,bd->abcd", d, d) + np.einsum("ad,bc->abcd", d, d)
    ll = int(round(np.einsum("abcd,abcd->", Sn, Sn)))
    lll = int(round(np.einsum("abcd,abef,cdef->", Sn, Sn, Sn)))
    l = eps / (n + 8) + 3 * (3 * n + 14) * eps ** 2 / (n + 8) ** 3
    CT = n + f["lam.lam"] * ll * l ** 2 + f["lam_abcd lam_abef lam_cdef"] * lll * l ** 3
    r = sp.expand(sp.series(CT / n, eps, 0, 4).removeO())
    ref = 1 - sp.Rational(5 * (n + 2), 12 * (n + 8) ** 2) * eps ** 2 - sp.Rational((n + 2) * (7 * n ** 2 + 382 * n + 1708), 36 * (n + 8) ** 4) * eps ** 3
    ok_On &= sp.simplify(r - ref) == 0
    det.append(r.coeff(eps, 3))
checks.append(("O(n) Wilson-Fisher C_T at eps^3 (n=1..4; Ising -233/8748)", ok_On, str(det)))

# --- large N (chiral Ising): C_T1 through eps^3 vs Diab et al.
N = sp.Symbol("N")
trip = [k for k in f if k.count("Tr(") == 3 and "lam" not in k]
dbl6 = [k for k in f if k.count("Tr(") == 2 and "lam" not in k and len(k) > 14]
F3 = sum(f[k] for k in trip)
d = 4 - eps
CTfree_lead = N * (d - 1) / 2
# leading in N: Tr(aa) = eps, Tr(ab)Tr(ab) = eps^2, triple traces = eps^3 (N y*^2 = eps), others subleading
num = 1 + f["Tr(aa)"] * eps + f["Tr(ab)Tr(ab)"] * eps ** 2 + F3 * eps ** 3
CT1 = sp.expand(sp.series(num / ((d - 1) / 2), eps, 0, 4).removeO())
eta1 = sp.gamma(d - 1) * (d - 2) ** 2 * (eps / 2) / (4 * sp.gamma(1 + eps / 2) * sp.gamma(d / 2 + 1) * sp.gamma(d / 2) ** 2)
CGN = (sp.polygamma(0, 1 + eps / 2) - 2 / eps) + sp.polygamma(0, d - 1) - sp.polygamma(0, 1) - sp.polygamma(0, d / 2)
ref = sp.expand(sp.series(-4 * eta1 * (CGN / (d + 2) + (d - 2) / (d * (d + 2) * (d - 1))), eps, 0, 4).removeO())
ok_N = sp.simplify(sp.expand((CT1 - ref).subs(sp.zeta(3), Z3))) == 0
checks.append(("large N: C_T1 through eps^3 = exact Gross-Neveu (Diab et al.)", ok_N, f"ours {CT1};  exact {ref}"))

for n_, ok, dd in checks:
    print(("PASS " if ok else "FAIL ") + n_, ("" if ok else dd))
print("four-loop fixed-point coefficients:")
for k, v in f.items():
    print(f"   {k:32s} {v}")
json.dump({k: str(v) for k, v in f.items()}, open(os.path.join(ROOT, "results", "coefficients4_T.json"), "w"), indent=1)

sys.exit(0 if all(ok for _, ok, _ in checks) else 1)
