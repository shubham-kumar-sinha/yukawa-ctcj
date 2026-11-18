#!/usr/bin/env python3
"""Generate the human-readable record of every diagram, and LaTeX tables of the diagram counts and G-scheme sums, from data/.

  results/diagram_catalogue.csv  every diagram class (1-4 loops, <TT> and <JJ>), one row per class: operators, fermion and
                                 scalar lines, weight (1/symmetry factor), G-scheme value X and flavour-invariant decomposition
  results/diagram_counts.tex      numbers of classes by correlator, loop order, couplings and operator pair
  results/gscheme4.tex           four-loop G-scheme sums per invariant (longtable)
Nothing in these files is typed by hand.  (~1 min)"""
import collections, csv, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import sympy as sp
from ctcj.forcer import EP, Z3, Z4, Z5

ROOT = os.path.join(os.path.dirname(__file__), "..")
loc = {"ep": EP, "z3": Z3, "z4": Z4, "z5": Z5, "I": sp.I}
LOOPS = {(0, 0): 1, (2, 0): 2, (4, 0): 3, (0, 2): 3, (6, 0): 4, (4, 1): 4, (2, 2): 4, (0, 3): 4}


def records():
    for kind in ("T", "J"):
        for fn in (f"diagrams_{kind}.json", f"diagrams4_{kind}.json"):
            for r in json.load(open(os.path.join(ROOT, "data", fn))):
                yield kind, LOOPS[(r["nY"], r["nQ"])], r


def edges(r):
    f = ", ".join(f"{a}→{b}" for a, b in r["ferm"])
    s = ", ".join(f"{a[0]}–{b[0]}" for a, b in r["sig"])
    return f, s


rows = []
for kind, L, r in records():
    f, s = edges(r)
    dec = "; ".join(f"{v} {k}" for k, v in r.get("decomp", {}).items())
    rows.append(dict(correlator=f"<{kind}{kind}>", loops=L, name=r["name"], ops=" ".join(r["ops"]), nY=r["nY"], nQ=r["nQ"],
                     weight=r["weight"], fermion_lines=f, scalar_lines=s, X=r["X"], decomposition=dec))

with open(os.path.join(ROOT, "results", "diagram_catalogue.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)



# ---------------- counts table
cnt = collections.Counter((r["correlator"], r["loops"], r["nY"], r["nQ"], r["ops"]) for r in rows)
nz = collections.Counter((r["correlator"], r["loops"], r["nY"], r["nQ"], r["ops"]) for r in rows if r["X"] != "0")
tex = [r"\begin{center}\small", r"\begin{longtable}{llllrr}", r"\toprule",
       r"correlator & loops & $(n_y,n_\lambda)$ & operators & classes & non-zero\\ \midrule", r"\endhead"]
for key in sorted(cnt, key=lambda k: (k[0] != "<TT>", k[1], -k[2], k[4])):
    c, L, ny, nq, ops = key
    o = ops.replace("psi", r"_\psi").replace("phi", r"_\phi").replace("sig", r"_\sigma").replace(" ", r",\ ")
    tex.append(f"$\\langle {c[1]}{c[2]}\\rangle$ & {L} & $({ny},{nq})$ & ${o}$ & {cnt[key]} & {nz[key]}\\\\")
tot_T = sum(v for k, v in cnt.items() if k[0] == "<TT>")
tot_J = sum(v for k, v in cnt.items() if k[0] == "<JJ>")
tex += [r"\midrule", f"\\multicolumn{{4}}{{l}}{{total $\\langle TT\\rangle$ / $\\langle JJ\\rangle$}} & {tot_T} / {tot_J} & \\\\",
        r"\bottomrule", r"\end{longtable}", r"\end{center}"]
open(os.path.join(ROOT, "results", "diagram_counts.tex"), "w").write("\n".join(tex) + "\n")

# ---------------- four-loop G-scheme sums per invariant
tot = collections.defaultdict(lambda: 0)
for kind, L, r in records():
    if L != 4 or r["X"] == "0":
        continue
    X = sp.sympify(r["X"], locals=loc)
    for k, c in r["decomp"].items():
        tot[(kind, k)] += sp.sympify(c, locals=loc) * X
lat = lambda e: sp.latex(sp.nsimplify(sp.expand(e)), symbol_names={Z3: r"\zeta_3", Z4: r"\zeta_4", Z5: r"\zeta_5"})
esc = lambda s: s.replace("_", r"\_")
tex = [r"\begin{center}\small", r"\begin{longtable}{lllll}", r"\toprule",
       r"invariant & $\epsilon'^{-4}$ & $\epsilon'^{-3}$ & $\epsilon'^{-2}$ & $\epsilon'^{-1}$\\ \midrule", r"\endhead"]
for kind in ("T", "J"):
    tex.append(f"\\multicolumn{{5}}{{l}}{{$\\langle {kind}{kind}\\rangle$, four loops}}\\\\")
    for (kk, k), v in sorted(tot.items(), key=lambda t: t[0][1]):
        if kk != kind:
            continue
        v = sp.expand(v)
        if v == 0:
            continue
        cells = [lat(v.coeff(EP, j)) for j in (-4, -3, -2, -1)]
        tex.append(f"\\texttt{{{esc(k)}}} & " + " & ".join(f"${c}$" for c in cells) + r"\\")
tex += [r"\bottomrule", r"\end{longtable}", r"\end{center}"]
open(os.path.join(ROOT, "results", "gscheme4.tex"), "w").write("\n".join(tex) + "\n")
print(f"wrote catalogue ({len(rows)} classes), results/diagram_counts.tex, results/gscheme4.tex")
