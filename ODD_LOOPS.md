# Odd fermion loops in three-dimensional-only theories

For theories without a four-dimensional counterpart, fermion loops with an odd number of Yukawa vertices contribute to
C_T and C_J at O(ε³).  Their continuation away from d = 3 is fixed by 𝒩=1 supersymmetry:

    ΔC_T/C_T,S = (7/6 − ζ₃) Tr(yᵃyᵇyᶜ)Tr(yᵃyᵇyᶜ),
    ΔC_J/C_J,ψ = −3 I₀ + (2 − 6ζ₃) I₁,   I₀ = Tr(Tyᵃyᵇyᶜ)Tr(Tyᵃyᵇyᶜ),  I₁ = Tr(Tyᵃyᵇyᶜ)Tr(Tyᵇyᶜyᵃ),

at leading order in the couplings.  The physics is in the paper: section 3.4 (the prescription and why it needs fixing),
section 5.4 (the supersymmetric relations), appendix E (the other continuations, table 9) and appendix F (the
supercurrent and flavour-current computation); eqs. (5.9), (5.10) and the last terms of (4.2) and (D.1).

## Using it

The coefficients are `ctcj.formulas4.KAPPA_T_ODD` and `KAPPA_J_ODD`, with the invariants `odd_invariant_T` and
`odd_invariants_J`.  They are off by default and switched on with `odd=True` in `Theory.CT3`, `Theory.CJ3` and
`ctcj.eps3models.series`.

The odd invariants must be evaluated with the theory's actual three-dimensional flavour structure, not a
representation that only reproduces the even invariants.  A four-dimensional Majorana fermion gives two two-component
Majoranas with opposite signs of y, so Tr(yyy) = 0; the same two fermions with equal signs form a two-component Dirac
fermion with a U(1) and Tr(yyy) ≠ 0, a different 3d theory with the same even invariants.  The models in
`ctcj/eps3models.py` use y ∝ 1 (and T ∝ 1 for a fermion current), so there the odd term is added only for C_T, and only
where the equal-sign theory is meant.  It is included at chiral Ising N = 1 and N = 2 (U(1) theory), the Liendo–Rong
points other than S₃, and the matrix models including their planar limit.  Nothing changes through O(ε²), or for any
theory obtained from four dimensions.

The odd loops first appear in six ⟨TT⟩ and six ⟨JJ⟩ four-loop classes, T60_004, 013, 015, 059, 085, 091 and J60_544,
553, 555, 599, 625, 631 in `data/diagrams4_*.json`, where they are stored with X = 0 (four-dimensional traces).

## Scripts

All run from cached integrals and are run by `reproduce.py`.

| script | what | output |
|---|---|---|
| `scripts/odd_traces.py` | the odd ⟨TT⟩, ⟨JJ⟩ classes for every continuation (paper table 9); the size of the effect | `results/odd_traces.md` |
| `scripts/count_supercurrent_odd.py` | no odd loops in ⟨S S̄⟩, ⟨j j̄⟩ through O(ε³) | (checks) |
| `scripts/supercurrent_rules.py` | S, its Feynman rules and projector; numerical checks in D = 3, 4 | (checks) |
| `scripts/supercurrent_oeps2.py` | Ward identities, pole cancellation, C_S = C_T through O(ε²) | `results/supercurrent_eps2.md` |
| `scripts/supercurrent_eps3.py` | four loops, one superfield: exact poles (paper appendix F), κ_T at super-Ising | `results/supercurrent_ising.md` |
| `scripts/supercurrent_eps3_general.py` | four loops at the five 𝒩=1 points (paper table 4, upper part) | `results/supercurrent_eps3.md` |
| `scripts/supercurrent_jj.py` | ⟨j j̄⟩ through four loops (paper table 4, lower part) | `results/supercurrent_jj.md` |
| `scripts/supercurrent_compute.py` | FORM/Forcer integrals for the above | `data/supercurrent_classes.json` |

Code: `ctcj/odd3.py` (odd traces, opening points, sewing), `ctcj/supercurrent.py` (rules, generators, projections,
flavour factors).  Cached integrals: `data/odd_traces*.json`, `data/supercurrent_classes.json`.
