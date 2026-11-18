#!/usr/bin/env python3
"""C_J at O(eps^3) from data/diagrams_J.json (1-3 loops) and data/diagrams4_J.json (4 loops, decomposed by
scripts/decompose4J.py, insertions in data/basis4_J.json).
Checks: pole/log cancellation; O(eps^2) coefficients unchanged; large-N Gross-Neveu C_J1 through eps^3 (Diab et al.);
O(n) Wilson-Fisher C_J at eps^3 and its large-n limit (Petkou's exact 1/n formula).
Writes results/coefficients4_J.json."""
import json, os, sys, time, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import sympy as sp
from ctcj.assembleJ4 import assemble_J4
from ctcj.forcer import EP, Z3, Z4, Z5
from ctcj.formulas import CJ as CJ_frozen
from ctcj.basisJ4 import candidates
from ctcj.net import evaluate

ROOT = os.path.join(os.path.dirname(__file__), "..")
loc = {"ep": EP, "z3": Z3, "z4": Z4, "z5": Z5, "I": sp.I}


def load(path):
    recs = json.load(open(path))
    for r in recs:
        r["X"] = sp.sympify(r["X"], locals=loc)
        r["decomp"] = {k: sp.sympify(v, locals=loc) for k, v in r["decomp"].items()}
    return recs


t0 = time.time()
res = assemble_J4(load(os.path.join(ROOT, "data", "diagrams_J.json")), load(os.path.join(ROOT, "data", "diagrams4_J.json")),
                  os.path.join(ROOT, "data", "basis4_J.json"))
f = res["coef"]
print(f"assembled in {time.time()-t0:.0f}s")
checks = []
bad = {k: v for k, v in res["poles"].items() if any(sp.simplify(p) != 0 for p in v)}
checks.append(("all poles cancel per invariant (four-loop <JJ> renormalisation)", not bad, str({k: bad[k] for k in list(bad)[:4]})))
checks.append(("log(mu^2 x^2) drops at fixed points", res["logs_ok"], ""))
ok3 = all(abs(float(f.get(k, 0)) - v) < 1e-12 for k, v in CJ_frozen.items())
checks.append(("O(eps^2) coefficients reproduced (new ensemble, new insertions)", ok3,
               str({k: f.get(k) for k in CJ_frozen})))

eps, N, n = sp.symbols("epsilon N n")
d = 4 - eps

# ---------------- large N Gross-Neveu: T = 1, single scalar, t = 0, N y*^2 = eps
ntr = lambda k: k.count("Tr(")
W2 = ["Tr(TTaa)", "Tr(TaTa)"]
from ctcj.invariants import J_BASIS
two_tr3 = [k for k in J_BASIS[4] if ntr(k) == 2]
three_tr4 = [k for k in res["basis"]["y6"] if ntr(k) == 3]
CJ1 = sum(f[k] for k in W2) * eps + sum(f.get(k, 0) for k in two_tr3) * eps ** 2 + sum(f[k] for k in three_tr4) * eps ** 3
CJ1 = sp.expand(CJ1)
eta1 = sp.gamma(d - 1) * (d - 2) ** 2 * (eps / 2) / (4 * sp.gamma(1 + eps / 2) * sp.gamma(d / 2 + 1) * sp.gamma(d / 2) ** 2)
ref = sp.expand(sp.series(-8 * (d - 1) / (d * (d - 2)) * eta1, eps, 0, 4).removeO())
checks.append(("large N: C_J1 through eps^3 = exact Gross-Neveu (Diab et al.)", sp.simplify(CJ1 - ref) == 0,
               f"ours {CJ1};  exact {ref}"))

# ---------------- O(n) Wilson-Fisher: lambda = l S, t = E12 - E21, l* to O(eps^2)
lam_names = ["t t lam lam"] + res["basis"]["lam3"]
cand = {**candidates("lam3")}
from ctcj.net import words_net
nets = {"t t lam lam": words_net([], extra=[("t", "ae"), ("t", "bf"), ("lam", "abcd"), ("lam", "efcd")]), **cand}
vals = {k: [] for k in lam_names}
for nn in range(2, 7):
    dd = np.eye(nn)
    Sn = np.einsum("ab,cd->abcd", dd, dd) + np.einsum("ac,bd->abcd", dd, dd) + np.einsum("ad,bc->abcd", dd, dd)
    t = np.zeros((nn, nn)); t[0, 1], t[1, 0] = 1, -1
    th = dict(y=np.zeros((1, 1, 1)), T=np.zeros((1, 1)), t=t, lam=Sn, mult=1.0)
    for k in lam_names:
        vals[k].append(evaluate(nets[k], th).real)
poly = {}
for k in lam_names:     # polynomial in n of degree <= 4, exact integers
    c = np.polyfit(range(2, 7), vals[k], 4)
    poly[k] = sp.expand(sum(sp.Integer(int(round(ci))) * n ** (4 - i) for i, ci in enumerate(c)))
    assert all(abs(float(poly[k].subs(n, nn)) - v) < 1e-6 for nn, v in zip(range(2, 7), vals[k]))
l = eps / (n + 8) + 3 * (3 * n + 14) * eps ** 2 / (n + 8) ** 3
tt = 2
CJ_On = 1 + (d - 2) / tt * (f["t t lam lam"] * poly["t t lam lam"] * l ** 2
                            + sum(f[k] * poly[k] * l ** 3 for k in res["basis"]["lam3"]))
CJ_On = sp.expand(sp.series(CJ_On, eps, 0, 4).removeO())
c2, c3 = sp.factor(CJ_On.coeff(eps, 2)), sp.factor(CJ_On.coeff(eps, 3))
ok2 = sp.simplify(c2 + sp.Rational(3, 4) * (n + 2) / (n + 8) ** 2) == 0
# exact large-n (Petkou): C_J1 = -8(d-1)/(d(d-2)) gamma_phi1, gamma_phi1 = -2 Gamma(d-2)/(Gamma(2-d/2)Gamma(d/2-1)Gamma(d/2-2)Gamma(d/2+1))
g1 = -2 * sp.gamma(d - 2) / (sp.gamma(2 - d / 2) * sp.gamma(d / 2 - 1) * sp.gamma(d / 2 - 2) * sp.gamma(d / 2 + 1))
refOn = sp.expand(sp.series(-8 * (d - 1) / (d * (d - 2)) * g1, eps, 0, 4).removeO())
oursOn = sp.expand(sp.limit(n * (c2 * eps ** 2 + c3 * eps ** 3), n, sp.oo))
checks.append(("O(n): eps^2 term = -3(n+2)/(4(n+8)^2) (Petkou)", ok2, str(c2)))
hvl = -(n + 2) * (n ** 2 + 132 * n + 632) / (8 * (n + 8) ** 4)        # Henriksson-van Loon 1801.03512 eq. (5.2); Dey et al.
checks.append(("O(n) Wilson-Fisher eps^3 = analytic bootstrap (Dey-Kaviraj-Sinha; Henriksson-van Loon)",
               sp.simplify(c3 - hvl) == 0, str(c3)))
checks.append(("O(n) large n: C_J1 through eps^3 = Petkou exact 1/n", sp.simplify(oursOn - refOn) == 0,
               f"ours {oursOn}; exact {refOn}"))

from ctcj.formulas4 import CJ4
froz = all(sp.simplify(f[k] - (sp.Rational(CJ4[k][0].numerator, CJ4[k][0].denominator)
                               + sp.Rational(CJ4[k][1].numerator, CJ4[k][1].denominator) * Z3)) == 0 if k in CJ4 else f[k] == 0
           for k in res["names4"])
checks.append(("frozen coefficients in ctcj/formulas4.py equal the assembled ones", froz, ""))
out = {k: str(v) for k, v in f.items()}
json.dump(dict(coef=out, On_eps3=str(c3), GN_CJ1=str(CJ1)), open(os.path.join(ROOT, "results", "coefficients4_J.json"), "w"), indent=1)
import subprocess
sus = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "susy_check4.py")], capture_output=True, text=True)
checks.append(("N=2 SUSY at eps^3: C_J^R/C_J^R,free = C_T/C_T,free (chiral XY, N=2; scripts/susy_check4.py)",
               sus.returncode == 0, sus.stdout[-400:]))

for nm, ok, dt in checks:
    print(("PASS " if ok else "FAIL ") + nm, "" if ok else dt)
print("O(n) Wilson-Fisher: C_J/C_J,free = 1 +", c2, "eps^2 +", c3, "eps^3")
print("large-N GN C_J1 =", CJ1)
print("four-loop fixed-point coefficients:")
for k in res["names4"]:
    if f[k] != 0:
        print(f"   {k:45s} {f[k]}")
sys.exit(0 if all(ok for _, ok, _ in checks) else 1)
