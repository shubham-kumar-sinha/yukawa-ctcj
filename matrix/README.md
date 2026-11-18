# Matrix Gross–Neveu–Yukawa models

This directory evaluates C_T and C_J through O(ε³) at the fixed points of the matrix Gross–Neveu–Yukawa models of S. Prakash and S. Sinha, *Emergent supersymmetry at large N*, [arXiv:2307.06841](https://arxiv.org/abs/2307.06841):

- **SU(N) adjoint:** real pseudo-scalars and Majorana fermions in the adjoint of SU(N);
- **SO(N) S²:** both in the real symmetric traceless representation of SO(N);
- **SO(N) S²–A²:** scalars in S², fermions in the adjoint (A²).

It also evaluates the planar (large-N) limit in closed form. The results are in [`../results/matrix_models.md`](../results/matrix_models.md).

## Running

Requires `numpy`, `sympy` and `opt_einsum`. Run from this directory:

```sh
python3 check.py           # checks of this evaluation path (~10 s); non-zero exit code on failure
python3 make_results.py    # writes ../results/matrix_models.md (~20 s)
python3 run_models.py SU3_susy SUN4 S2N4 SO3S2A2_ns+    # print the coefficients for selected fixed points
python3 largeN.py          # the planar limit
```

Both scripts are run by `../reproduce.py`.

## How it works

This is a second evaluation path, separate from the main package `ctcj`. It evaluates the O(ε³) formulas exactly as printed in the paper. `ct3.tex` and `cj3.tex` are verbatim copies of equations (4.2) and (D.1), with the generators written T and t instead of Θ and θ. Their last terms are the odd fermion loops ([`../ODD_LOOPS.md`](../ODD_LOOPS.md)); none of these theories has a four-dimensional counterpart, so they contribute at every point here. `formulas.py` parses the equations and adds the O(ε²) terms. Agreement with the main pipeline (`check.py`, and the comparison of the odd-loop terms with `ctcj.formulas4` in `make_results.py`) therefore also checks that the printed equations are correct.

| File | Content |
|---|---|
| `models.py` | The coupling tensors y^a_ij and λ_abcd of the three models, from eqs. (A.1)–(A.2) of arXiv:2307.06841 in the conventions of Jack, Osborn and Steudtner; the current generators; one-loop checks of β_y and γ_φ |
| `run_models.py` | Fixed-point couplings (closed forms of arXiv:2307.06841: all N at the supersymmetric points of the SU(N) and S² models, N = 3 otherwise) and C_T, C_J at each point |
| `formulas.py` | Parser for `ct3.tex`, `cj3.tex`; numerical evaluation of every invariant for explicit tensors (`opt_einsum`) |
| `fastseries.py`, `series.py` | Assembly of C/C_free as truncated ε-series, with ζ₃ tracked exactly (fast numerical and sympy versions) |
| `planar.py`, `largeN.py` | Exact U(N) evaluation of the invariants by double-line (Fierz) contraction, and the planar limit |
| `check.py` | Chiral Ising and XY through ε², the O(n) model at ε³, and the planar limit with its odd-loop terms, against independent results |
| `make_results.py` | The table of the paper and d = 3 estimates from pole-free Padé approximants of the 4 − ε series (`../ctcj/pade.py`) |

## Conventions

As in the main package: N = Tr 1 counts two-component Majorana fermions, the fermion number of arXiv:2307.06841 is continued to N_f = ½, and couplings are rescaled as y → 4πy and λ → 16π²λ. The fixed points are named as in arXiv:2307.06841: [susy] for the supersymmetric points and [ns±] for the non-supersymmetric ones.
