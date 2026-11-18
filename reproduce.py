#!/usr/bin/env python3
"""Reproduce every result of the paper from scratch.  See METHOD.md, section 10.

    python3 reproduce.py                  # from the cached Feynman integrals (~40 min on one core)
    python3 reproduce.py --integrals 8    # also recompute all integrals with FORM/Forcer (8 parallel jobs)
    python3 reproduce.py --fast           # skip the slowest steps (four-loop <TT> decomposition, d = 3 estimates)

Works in a fresh copy, build/reproduce/work, from which every derived file is deleted, then reruns:
diagram generation and weights -> (integrals) -> flavour decompositions and beta insertions -> C_T and C_J at all
orders with all checks -> three-loop beta-function checks -> d = 3, chiral-model, conductivity, large-N, Liendo-Rong
and matrix-model results -> odd fermion loops and the N=1 supercurrent relations that fix them -> generated tables and
figure.  Finally the regenerated results are compared with the published ones.  A summary is written
to build/reproduce/summary.md and each step's output to build/reproduce/logs/."""
import argparse, filecmp, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--integrals", type=int, metavar="JOBS", default=0, help="recompute all Feynman integrals with FORM/Forcer")
ap.add_argument("--fast", action="store_true", help="skip the slowest steps")
args = ap.parse_args()

base = os.path.join(ROOT, "build", "reproduce")
work, logs = os.path.join(base, "work"), os.path.join(base, "logs")
if os.path.exists(base):
    shutil.rmtree(base)
os.makedirs(logs)
shutil.copytree(ROOT, work, ignore=shutil.ignore_patterns("build", "__pycache__", "*.pyc", ".git"))
derived = ["data/diagrams4_T.json", "data/diagrams4_J.json", "data/basis4_J.json", "data/relations4_J.json"]
derived += [os.path.join("results", f) for f in os.listdir(os.path.join(ROOT, "results"))]
if args.integrals:
    derived += ["data/diagrams_T.json", "data/diagrams_J.json", "data/diagrams4_T.jsonl", "data/diagrams4_J.jsonl",
                "data/supercurrent_classes.json"]
for f in derived:
    p = os.path.join(work, f)
    if os.path.exists(p):
        os.remove(p)

steps = [("diagram classes and weights (1-4 loops)", ["scripts/regenerate_diagrams.py", os.path.join(ROOT, "data")])]
if args.integrals:
    j = str(args.integrals)
    steps += [("FORM/Forcer installation test", [os.path.join(ROOT, "scripts", "test_form_setup.py")], ROOT),
              ("integrals, 1-3 loops", ["scripts/compute_diagrams.py"]),
              ("integrals, four-loop <TT>", ["scripts/compute_diagrams4.py", "T", "3", j]),
              ("integrals, four-loop <JJ>", ["scripts/compute_diagrams4.py", "J", "3", j]),
              ("recomputed vs published integrals", ["scripts/compare_integrals.py", os.path.join(ROOT, "data"), os.path.join(work, "data")]),
              ("integrals, <S Sbar> through three loops",
               ["scripts/supercurrent_compute.py", "--w", "0", "2", "4", "--proj", "P", "-j", j]),
              ("integrals, <S Sbar> Ward contractions", ["scripts/supercurrent_compute.py", "--w", "2", "4", "--proj", "W1", "W2", "W3", "--ward", "-j", j]),
              ("integrals, <j jbar> through three loops", ["scripts/supercurrent_compute.py", "--w", "0", "2", "4", "--jj", "--proj", "Pj", "-j", j]),
              ("integrals, four-loop <S Sbar>", ["scripts/supercurrent_compute.py", "--w", "6", "--fast", "--depth", "3", "--proj", "P", "-j", j]),
              ("integrals, four-loop <j jbar>", ["scripts/supercurrent_compute.py", "--w", "6", "--jj", "--fast", "--depth", "3", "--proj", "Pj", "-j", j])]
if not args.fast:
    steps += [("four-loop <TT> flavour decomposition", ["scripts/decompose4.py"])]
else:                                     # fast mode: reuse the published four-loop <TT> decomposition and figure
    shutil.copy(os.path.join(ROOT, "data", "diagrams4_T.json"), os.path.join(work, "data"))
    for f in ("pade_3d.pdf", "pade_3d.md"):
        shutil.copy(os.path.join(ROOT, "results", f), os.path.join(work, "results"))
steps += [("four-loop <JJ> basis, decomposition, insertions", ["scripts/decompose4J.py", "--redo"]),
          ("Ward relations of the <JJ> basis", ["scripts/ward_relations4J.py"]),
          ("C_T, C_J through O(eps^2), models, checks", ["scripts/compute_results.py"]),
          ("C_T at O(eps^3), checks", ["scripts/compute_results4.py"]),
          ("C_J at O(eps^3), checks incl. N=2 SUSY", ["scripts/compute_results4J.py"]),
          ("three-loop beta functions (JOS) checks", ["scripts/check_betas3.py"]),
          ("three-loop chiral GNY functions (Zerf et al.) checks", ["scripts/check_gny3loop.py"]),
          ("one-loop walkthrough (free C_T, C_J)", ["scripts/walkthrough_one_loop.py"])]
if not args.fast:
    steps += [("d = 3 estimates", ["scripts/pade3d.py"]), ("figure", ["scripts/plot_pade.py"])]
steps += [("chiral-model tables, N=2 R-current, conductivity, 1/N^2", ["scripts/paper_tables.py"]),
          ("Liendo-Rong fixed points", ["scripts/liendo_rong.py"]),
          ("matrix GNY models: checks", ["check.py"], os.path.join(work, "matrix")),
          ("matrix GNY models: results", ["make_results.py"], os.path.join(work, "matrix")),
          ("odd fermion loops (3d-only theories), from cached integrals", ["scripts/odd_traces.py"]),
          ("N=1 supercurrent: no odd loops through O(eps^3)", ["scripts/count_supercurrent_odd.py"]),
          ("N=1 supercurrent: operator, Feynman rules, projector, flavour factors", ["scripts/supercurrent_rules.py"]),
          ("N=1 supercurrent: Ward identities, poles, C_S = C_T through O(eps^2)", ["scripts/supercurrent_oeps2.py"]),
          ("N=1 supercurrent: four loops at super-Ising, kappa_T", ["scripts/supercurrent_eps3.py"]),
          ("N=1 supercurrent: four loops at five N=1 points", ["scripts/supercurrent_eps3_general.py"]),
          ("N=1 flavour-current multiplet: C_j = C_J through O(eps^3)", ["scripts/supercurrent_jj.py"]),
          ("diagram catalogue and LaTeX tables", ["scripts/make_tables.py"]),
          ("G-scheme sums per invariant", ["scripts/tabulate_gscheme.py"])]

rows, ok_all = [], True
for step in steps:
    title, cmd, cwd = (step + (work,))[:3]
    t0 = time.time()
    log = os.path.join(logs, f"{len(rows) + 1:02d}_{os.path.basename(cmd[0]).replace('.py', '')}.log")
    print(f"[{len(rows) + 1:2d}/{len(steps)}] {title} ...", end=" ", flush=True)
    with open(log, "w") as fh:
        r = subprocess.run([sys.executable, "-u"] + cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    st = "PASS" if r.returncode == 0 else "FAIL"
    ok_all &= r.returncode == 0
    print(f"{st} ({time.time() - t0:.0f} s)", flush=True)
    rows.append((title, st, f"{time.time() - t0:.0f} s", os.path.relpath(log, base)))

# compare regenerated results with the published ones
def same(rel):
    a, b = os.path.join(ROOT, rel), os.path.join(work, rel)
    if not os.path.exists(b):
        return "not regenerated"
    if rel.endswith(".json"):
        return "identical" if json.load(open(a)) == json.load(open(b)) else "DIFFERENT"
    return "identical" if filecmp.cmp(a, b, shallow=False) else "DIFFERENT"

compare = ["results/coefficients.json", "results/coefficients4_T.json", "results/coefficients4_J.json",
           "data/basis4_J.json", "data/relations4_J.json", "data/diagrams4_J.json", "results/liendo_rong.md",
           "results/gscheme_totals.md", "results/diagram_catalogue.csv", "results/gscheme4.tex", "results/diagram_counts.tex",
           "results/paper_tables.md", "results/matrix_models.md",
           "results/odd_traces.md", "results/supercurrent_eps2.md", "results/supercurrent_ising.md",
           "results/supercurrent_eps3.md", "results/supercurrent_jj.md"]
if not args.fast:
    compare += ["data/diagrams4_T.json", "results/pade_3d.md"]
cmp_rows = [(rel, same(rel)) for rel in compare]
ok_all &= all(s == "identical" for _, s in cmp_rows)

out = ["# Reproduction summary", "", f"mode: {'integrals recomputed with FORM/Forcer' if args.integrals else 'cached integrals'}"
       + (", fast" if args.fast else ""), "", "| step | result | time | log |", "|---|---|---|---|"]
out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
out += ["", "| regenerated file | compared with published |", "|---|---|"] + [f"| {a} | {b} |" for a, b in cmp_rows]
out += ["", "ALL STEPS PASS AND ALL RESULTS IDENTICAL" if ok_all else "SOME STEPS FAILED OR RESULTS DIFFER (see logs)"]
open(os.path.join(base, "summary.md"), "w").write("\n".join(out) + "\n")
print("\n".join(out[-len(cmp_rows) - 4:]))
print(f"summary: {os.path.relpath(os.path.join(base, 'summary.md'), ROOT)}")
sys.exit(0 if ok_all else 1)
