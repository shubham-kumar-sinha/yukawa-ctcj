#!/usr/bin/env python3
"""Assemble C_T and C_J from data/diagrams_{T,J}.json, evaluate the models, run all cross-checks.
Needs only numpy + sympy (no FORM).  Writes results/results.md and results/coefficients.json."""
import json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sympy as sp
from ctcj.assemble import assemble
from ctcj.forcer import EP, Z3, Z4, Z5
from ctcj import checks as C

ROOT = os.path.join(os.path.dirname(__file__), "..")
loc = {"ep": EP, "z3": Z3, "z4": Z4, "z5": Z5, "I": sp.I}


def load(kind):
    recs = json.load(open(os.path.join(ROOT, "data", f"diagrams_{kind}.json")))
    for r in recs:
        r["X"] = sp.sympify(r["X"], locals=loc)
        r["decomp"] = {k: sp.sympify(v, locals=loc) for k, v in r["decomp"].items()}
        r["ops"] = tuple(r["ops"])
    return recs


t0 = time.time()
recT, recJ = load("T"), load("J")
resT, resJ = assemble("T", recT), assemble("J", recJ)
print(f"assembled in {time.time()-t0:.0f}s")
models = C.model_results(resT, resJ)

checks = [C.check_counts(recT, recJ), C.check_free(resT, resJ), C.check_two_loop_table(recT),
          C.check_largeN_family_sum(resT), C.check_poles_logs(resT, "T"), C.check_poles_logs(resJ, "J"),
          C.check_ward_relations(), C.check_On(resT, resJ), C.check_gny_beta(),
          C.check_gny_order_eps(models), C.check_largeN(models), C.check_largeN_scaling(models),
          C.check_On_limit(models), C.check_susy(resT, resJ)]
from ctcj.formulas import CT as CTf, CJ as CJf
froz = all(abs(float(resT["coef"][k]) - v) < 1e-12 for k, v in CTf.items()) and \
       all(abs(float(resJ["coef"][k]) - v) < 1e-12 for k, v in CJf.items())
checks.append(("frozen coefficients in ctcj/formulas.py equal the assembled ones", froz, ""))

lines = ["# Results", "", "Couplings rescaled y -> 4 pi y, lambda -> 16 pi^2 lambda; Tr = tr(1) x flavour trace",
         "(JOS normalisation, gamma_phi^(1)ab = 1/2 Tr(y^a y^b)).  Valid at any fixed point, through O(eps^2).", "",
         "## C_T / C_T,scalar", "", "| invariant | coefficient |", "|---|---|"]
lines += [f"| {k} | {v} |" for k, v in resT["free"].items()]
lines += [f"| {k} | {v} |" for k, v in resT["coef"].items()]
lines += ["", "## C_J / C_J,fermion (per unit Tr(TT))", "", "| invariant | coefficient |", "|---|---|"]
lines += [f"| {k} | {v} |" for k, v in resJ["free"].items()]
lines += [f"| {k} | {v} |" for k, v in resJ["coef"].items()]
lines += ["", "## Models (lambda_+ fixed point), ratios to the free theory", ""]
for name, (a1, a2, b1, b2) in models.items():
    lines += [f"### {name}", "", f"- C_T: 1 + ({sp.factor(a1)}) eps + ({a2}) eps^2",
              f"- C_J: 1 + ({sp.factor(b1)}) eps + ({b2}) eps^2", ""]
lines += ["## Cross-checks", "", "| check | result | detail |", "|---|---|---|"]
lines += [f"| {n} | {'PASS' if ok else 'FAIL'} | {d[:150]} |" for n, ok, d in checks]
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
open(os.path.join(ROOT, "results", "results.md"), "w").write("\n".join(lines) + "\n")
json.dump({"C_T": {**{k: str(v) for k, v in resT["free"].items()}, **{k: str(v) for k, v in resT["coef"].items()}},
           "C_J": {**{k: str(v) for k, v in resJ["free"].items()}, **{k: str(v) for k, v in resJ["coef"].items()}},
           "models": {n: [str(x) for x in v] for n, v in models.items()}},
          open(os.path.join(ROOT, "results", "coefficients.json"), "w"), indent=1)
for n, ok, d in checks:
    print(("PASS " if ok else "FAIL ") + n)
print("C_T:", resT["coef"])
print("C_J:", resJ["coef"])
sys.exit(0 if all(ok for _, ok, _ in checks) else 1)
