# How the results are computed

This document follows one result from the Lagrangian to the numbers in the paper. Each step names the code that does it, and `python3 scripts/walkthrough_one_loop.py` runs all steps on the one-loop diagrams, printing every intermediate quantity. `python3 reproduce.py` reruns the whole chain (section 10).

## 1. Theory, operators, Feynman rules

The general renormalizable scalar–fermion theory, in Euclidean signature with the conventions of Jack, Osborn and Steudtner, is

    L = ½ (∂φ_a)² + ψ̄ ∂̸ ψ + φ_a ψ̄ y^a ψ + (1/24) λ_abcd φ_a φ_b φ_c φ_d .

The y^a are hermitian flavor matrices, and chirality is carried by the flavor structure. A fermion trace Tr is the spinor trace times the flavor trace, with the Majorana factor ½ absorbed into it: Tr 1 counts three-dimensional Majorana fermions. One two-component Majorana fermion has Tr 1 = 2 × ½ = 1, and one four-component Dirac fermion has Tr 1 = 4, with no ½. Couplings are rescaled as y → 4πy and λ → 16π²λ.

The operators are contracted with null polarizations z (z² = 0):

    T(x, z) = ½[ψ̄ z̸ (z·∂ψ) − (z·∂ψ̄) z̸ ψ] + (z·∂φ)² − ξ (z·∂)²φ²,   ξ = (d−2)/(4(d−1)),
    J(x, z) = ψ̄ z̸ T ψ + J_φ,  with J_φ ∝ (z·∂φ_a) t_ab φ_b ,

where (T, t) generates a symmetry, i[y^a, T] = t_ab y^b. The paper writes these generators as (Θ, θ); the code and these notes use (T, t). The normalization is fixed by the free values this code reproduces (section 5): C_J = −Tr(TT)/S_d² for fermions and −t·t/((d−2)S_d²) for scalars, the overall sign being the Euclidean convention.

| Element | Momentum-space rule |
|---|---|
| fermion propagator | −i p̸/p², p along the arrow |
| scalar propagator | 1/p² |
| Yukawa vertex | −y^a |
| quartic vertex | −λ_abcd |
| T_ψ insertion | (i/2) z̸ z·(p_in + p_out) |
| T_σ insertion | (i z·q₁)(i z·q₂) |
| J_ψ insertion | z̸ T |
| J_φ insertion | t_ab z·q_d, with q_d the momentum of the differentiated leg into the vertex |
| closed fermion loop | −1 × Tr, the Dirac rule; for Majorana fermions the factor ½ is included in Tr (see above) |

Projecting with z_i·Q = 0 removes the contact, longitudinal and improvement terms (the ξ term) of T.

Code: `ctcj/forcer.py` (integrands); `scripts/walkthrough_one_loop.py` checks these rules against the textbook free-field values.

## 2. Diagrams and symmetry factors

Expanding exp(−S_int) gives n_Y Yukawa vertices with factor 1/n_Y!, and n_Q quartic vertices with factor 1/(n_Q! 24^{n_Q}). Every Wick contraction with *labeled* vertices is enumerated:

- fermion lines as permutations of the fermionic vertices, with no fermion self-loops;
- scalar lines as perfect matchings of the scalar legs, with no tadpoles;
- only diagrams whose graph is **biconnected** are kept. A cut vertex separating x from 0 gives a contribution killed by the transverse projection, or a massless tadpole.

Contractions that give isomorphic graphs (vertices relabeled, operators x and 0 kept fixed) form one **class**. The class weight is

    weight = (number of labeled contractions in the class) / (n_Y! n_Q! 24^{n_Q}) = 1/(symmetry factor).

Example: in T_σ(x) T_σ(0) there are two scalar legs at each operator and two ways of pairing them, so the weight is 2. The fermion sign (−1)^(number of loops) and the coupling factors are applied separately.

Code:
- `ctcj/diagrams.py` is the brute-force generator, used at one to three loops.
- `ctcj/diagrams_fast.py` gives the same classes using nauty canonical labels, used at four loops.
- `scripts/regenerate_diagrams.py` regenerates all 1414 classes, cross-checks the two generators, and compares every graph and weight with `data/`.

**Vanishing classes.** Some classes vanish identically: a fermion loop with an odd number of γ matrices, or Furry's theorem for an odd number of J_ψ on a loop. They are kept with X = 0. In three-dimensional-only theories the odd fermion loops contribute at O(ε³); see [`ODD_LOOPS.md`](ODD_LOOPS.md).

**Four loops.** ⟨TT⟩ has 444 classes (407 y⁶, 18 y⁴λ, 16 y²λ², 3 λ³; 158 Forcer runs after symmetry orbits) and ⟨JJ⟩ has 821 (729, 41, 41, 10; 269 runs). The nauty generator reproduces the brute-force one exactly at two and three loops.

**Symmetry orbits.** Classes related by x ↔ 0 or by reversing a fermion loop have equal integrals; for J the sign is (−1)^(number of J_ψ on the reversed loop). Only one class per orbit is integrated (`ctcj/orbits.py`). The rule is verified on all three-loop data.

## 3. Integrals

For each class, FORM builds the integrand with momentum routing from the graph. The open indices of the two operators are contracted with the transverse-traceless projector (for T) or the transverse projector (for J), so only scalar products remain. FORM takes the D-dimensional Dirac traces (D = 4 − 2ε′), and Forcer reduces the resulting massless propagator integrals, up to four loops, to master integrals.

The output is the **G-scheme value**

    X(ε′) = (projected correlator at Q² = 1) / [ε′ G(1,1)]^L ,
    G(1,1) = Γ(1−ε′)² Γ(ε′) / ((4π)^{2−ε′} Γ(2−2ε′)),

normalized so that

    ⟨TT⟩ = (z₁·z₂)² Q⁴ (Q²)^{−Lε′} [ε′G(1,1)]^L · 2X/((D−1)(D+1)(D−2)),
    ⟨JJ⟩ = (z₁·z₂) Q² (Q²)^{−Lε′} [ε′G(1,1)]^L · X/(D−1).

X includes the class weight, the fermion-loop sign and all factors of i. Forcer expands to a finite depth in ε′; four-loop integrals are needed only through ε′⁻¹ (see section 6). Depth 3 was used. Depth 4 gives identical coefficients ε′⁻⁴ … ε′⁻¹ on the 73 classes rechecked (7 ⟨TT⟩, 66 ⟨JJ⟩); `python3 scripts/check_depth4.py T|J <class names>` repeats this. Most four-loop classes take 1–5 s in Forcer, a few ⟨TT⟩ classes 100–150 s.

Code: `ctcj/forcer.py`; `scripts/compute_diagrams.py` (one to three loops) and `scripts/compute_diagrams4.py` (four loops). Installation and an exact-reproduction test of FORM and Forcer are in `INSTALL_FORM.md`.

## 4. Flavor factors

The flavor factor of a class is its graph read as a tensor network: each fermion loop is a trace of y's (and T's), each scalar line contracts two indices, and the operators insert T or t. The results are expressed in a basis of independent invariants.

- **⟨TT⟩.** The factors are traces of y's (and λ's), reduced to a basis by numerical evaluation on random theories (`ctcj/invariants.py`, `ctcj/invariants4.py`).
- **⟨JJ⟩.** Invariants containing T and t are related by the Ward identity i[y^a, T] = t_ab y^b. The basis is found by evaluating all candidate invariants on random theories with an exact U(1) symmetry, built by group averaging; any single current generates a U(1), so this is general. The independent subset is then extracted. The candidates (`ctcj/basisJ4.py`) are connected trace products of TT and the y's, λ contracted into such traces, and every connected network of t, λ and small trace blobs (the scalar-current structures the Yukawa Ward identity cannot remove). Evaluated with einsum (`ctcj/net.py`) on random U(1)-symmetric theories (`ctcj/ensembleJ.py`), the greedy independent subset is the basis:

  | sector | candidates | basis | Ward relations (identically zero) |
  |---|---|---|---|
  | y⁶ | 94 | 48 | 46 (0) |
  | y⁴λ | 6 | 6 | 0 |
  | y²λ² | 33 | 4 | 29 (6) |
  | λ³ | 15 | 3 | 12 (4) |

  The basis is the same on an independent ensemble with different field content, and the same machinery recovers the three-loop basis and its four Ward relations. The 87 relations, each re-verified on a second ensemble, are in `data/relations4_J.json` (`scripts/ward_relations4J.py`); every class and β-insertion lies in the span of the basis (`ctcj/invariantsJ4.py`, `scripts/decompose4J.py`).

Every class is decomposed exactly, with rational coefficients (`data/diagrams*_?.json`, `results/diagram_catalogue.csv`). Summing X over classes per invariant gives the G-scheme sums (`results/gscheme_totals.md`, paper appendices).

## 5. Fourier transform and extraction of C_T, C_J

An L-loop term is a pure power of Q², A_L (Q²)^β with β = −Lε′. In d dimensions,

    ∫ d^dQ/(2π)^d e^{iQ·x} (Q²)^β = c(β)/(x²)^s,    c(β) = 4^β Γ(d/2+β) / (π^{d/2} Γ(−β)),   s = d/2 + β.

The conformal structures are C_T (z₁·I·z₂)²/x^{2d} and C_J (z₁·I·z₂)/x^{2d−2}, with I = 1 − 2xx/x². To read off C_T and C_J, the full tensor structure is restored. The transverse-traceless projector contracted with null polarizations is (z₁·π·z₂)² − (z₁·Q)²(z₂·Q)²/((d−1)Q⁴), with π = 1 − QQ/Q². Under the transform Q_μ → −i∂_μ.

- The coefficient of (z₁·x)²(z₂·x)²/(x²)^{s+4} equals 4C_T.
- The coefficient of (z₁·x)(z₂·x)/(x²)^{s+2} equals −2C_J.

Together these give

    C_T = 4 (d−2)/(d−1) · s(s+1)(s+2)(s+3) · c(β) · A_T ,      C_J = −2 s(s+1) · c(β) · A_J .

At one loop (s = d − 2) these give the textbook values:
- C_T of a free scalar: d/((d−1)S_d²);
- C_T of free fermions: d·Tr1/(2S_d²);
- C_J: −Tr(TT)/S_d² for fermions and −t·t/((d−2)S_d²) for scalars, with S_d = 2π^{d/2}/Γ(d/2).

`scripts/walkthrough_one_loop.py` checks all four at three values of d.

In the assembly, each L-loop term is normalized to the one-loop free value:

    R_L = [C from (Q²)^{−Lε′} term] / [C of the one-loop fermion term],

with the ε-expansion of the Γ-function ratios, (4π) and G(1,1) factors, and the logarithm L = log μ²x² from (μ²x²)^{(L−1)ε′} kept explicitly (`ctcj.assemble.position_ratio`). Because 1/Γ(−β) = 1/Γ(Lε′) ~ Lε′, an L-loop integral is needed only through ε′^{3−L} for a result at O(ε³). This is why the four-loop integrals are needed only through their poles.

## 6. Renormalization

T and J are conserved, so with bare fields only the couplings renormalize. In MS-bar, with ε = 4 − d = 2ε′,

    y₀ = μ^{ε/2} [ y + β₁(y)/ε + β₂(y,λ)/(2ε) + ½Dβ₁[β₁]/ε² ],     λ₀ = μ^ε [ λ + β_λ⁽¹⁾/ε ],

with the one- and two-loop β-functions of the general theory (`ctcj/theory.py`, `ctcj/betasJ.py`). Substituting into the bare series, Σ_L R_L × I(y₀, λ₀), requires the derivatives of lower-loop invariants along the β-functions. These "insertions", DI[β] and ½D²I[β,β], are computed exactly and decomposed on the higher-loop basis (`ctcj/assemble4.py`, `ctcj/assembleJ4.py`, `ctcj/invariantsJ4.py`). Then two checks follow, separately for every invariant:

- every pole in ε cancels;
- log(μ²x²) multiplies only combinations that vanish at a fixed point.

Both hold at every order. Pole cancellation tests the ε′⁻⁴ … ε′⁻² terms of the four-loop integrals. The ε³ coefficients come from the ε′⁻¹ terms, which are tested by the depth check (section 3) and by known results (`scripts/compute_results4.py`, `scripts/compute_results4J.py`, `scripts/susy_check4.py`):

- O(n) Wilson–Fisher: C_T/C_T,free − 1 = −5(n+2)ε²/(12(n+8)²) − (n+2)(7n²+382n+1708)ε³/(36(n+8)⁴) (Osborn–Stergiou), and C_J/C_J,free − 1 = −3(n+2)ε²/(4(n+8)²) − (n+2)(n²+132n+632)ε³/(8(n+8)⁴) (analytic bootstrap; Henriksson–van Loon, arXiv:1801.03512, eq. 5.2);
- large-N Gross–Neveu: C_T1 and C_J1 = −3ε/2 + ε²/2 + 15ε³/32 equal the exact results of Diab et al.;
- 𝒩=2 supersymmetry at chiral XY with N = 2: C_J^R/C_J,free^R = C_T/C_T,free through ε³, with λ* from the two-loop β_λ;
- 𝒩=1 supersymmetry: C_S = C_T at five points and C_j = C_J at two ([`ODD_LOOPS.md`](ODD_LOOPS.md)).

The two-loop β_λ (`ctcj/betasJ.py`, JOS section 6) reproduces the O(n) λ³ term −3(3n+14) and the two-loop β_g2 of FGKT for the GNY and NJLY models.

**Away from fixed points (through O(ε²)).**  With L̃ = log(πμ²x²) + γ_E (minimal subtraction of the rescaled couplings), the
renormalized coefficients before the fixed-point reduction of section 7 are, in addition to the free terms,

    C_T/C_T,S:  Tr(yᵃyᵃ) [−5/4 − (ε/2)(103/16 + (5/4) L̃)] + Tr(yᵃyᵇ)Tr(yᵃyᵇ) [293/96 + (5/8) L̃]
              + Tr(yᵃyᵃyᵇyᵇ) [149/24 + (5/4) L̃] + Tr(yᵃyᵇyᵃyᵇ) [233/24 + (5/2) L̃] − (5/36) λ·λ

    C_J/C_J,ψ:  Tr(TTyᵃyᵃ) [−3/2 − (ε/2)(15/2 + (3/2) L̃)] + (9ε/4) Tr(TyᵃTyᵃ)
              + Tr(TTyᵃyᵃyᵇyᵇ) [11/4 + (3/4) L̃] + Tr(TTyᵃyᵇyᵃyᵇ) [13/2 + 3 L̃] + Tr(TTyᵃyᵇyᵇyᵃ) [25/8 + (3/4) L̃]
              − (15/4) Tr(TyᵃTyᵃyᵇyᵇ) − (3/2) Tr(TyᵃTyᵇyᵃyᵇ) + (3/2) Tr(TyᵃyᵇTyᵇyᵃ)
              + Tr(TTyᵃyᵇ)Tr(yᵃyᵇ) [11/4 + (3/4) L̃] − (3/4) Tr(TyᵃTyᵇ)Tr(yᵃyᵇ) − (3/8) t_ae t_bf λ_abcd λ_efcd

The L̃ dependence is proportional to Tr(yᵃβ̂ᵃ) and Tr(TT{yᵃ, β̂ᵃ}), with β̂ the full β_y, and vanishes at fixed points.  Away
from a fixed point the coefficients are ambiguous by such terms; the fixed-point form removes the O(ε) parts of the
two-loop coefficients with εyᵃ* = 2β₁ᵃ(y*).

## 7. Fixed-point form

At a fixed point β = 0. For an invariant I with n_y Yukawa and n_λ quartic couplings this gives εI = DI[β]/w(I), with w = n_y/2 + n_λ. Using it to eliminate every explicit ε leaves coefficients that are pure numbers (rationals and ζ₃) multiplying invariants:

    C_T/C_{T,S} = n_s + (d−1)/2 Tr1 + Σ_k f_k I_k(y*, λ*),    and similarly for C_J.

These are the formulas in the paper (`results/coefficients*.json`, frozen in `ctcj/formulas.py` and `ctcj/formulas4.py`).

## 8. Specific fixed points through O(ε³)

At a specific fixed point, y* is needed to O(ε³) and λ* to O(ε²):

    y* = √ε (Y₁ + εY₂ + ε²Y₃),   λ* = ε (L₁ + εL₂).

- **One loop:** (Y₁, L₁) solve the one-loop equations (multi-start Newton).
- **Higher orders:** L₂ and then Y₃ solve linear equations with the one-loop stability matrix. These use the two-loop β_λ and the three-loop β_y, γ_ψ, γ_φ of the general theory, as collected by Jack, Osborn and Steudtner (`ctcj/betas3.py`; checks in `scripts/check_betas3.py`).

L₃, and hence the three-loop β_λ, does not enter. `Theory.CT3` and `Theory.CJ3` then evaluate the series. For the chiral Ising, XY and Heisenberg models the model-specific β-functions of Zerf et al. give the same series to seven digits (`ctcj/gny3loop.py`, `ctcj/eps3models.py`, `scripts/check_gny3loop.py`).

## 9. Estimates in d = 3

Two-sided Padé approximants P(ε)/Q(ε) match the 4 − ε series and, where available, the exact d = 2 value or the 2 + ε series of Diab et al. Only approximants without poles for 2 ≤ d ≤ 4 are kept, and their spread is quoted (`ctcj/pade.py`, `scripts/pade3d.py`, `scripts/plot_pade.py`, `scripts/liendo_rong.py`). For the conductivities the d = 2 input is the free value, and for the Liendo–Rong and matrix-model fixed points, whose d = 2 limits are unknown, only the 4 − ε series is used (`scripts/paper_tables.py`, `matrix/make_results.py`).

## 10. Reproducing everything

    python3 reproduce.py                 # everything from the cached integrals (~40 min)
    python3 reproduce.py --integrals 8   # also recompute every Feynman integral with FORM/Forcer, 8 parallel jobs

Without `--integrals`, all cached integrals are kept and everything downstream is recomputed.

`reproduce.py` works in a fresh copy (`build/reproduce/work`) with all derived files deleted and reruns every step:

1. diagram generation with weights;
2. optionally, all integrals;
3. flavor decompositions, β-insertions and Ward relations;
4. the assembly of C_T and C_J at all orders with all checks;
5. the three-loop β-function checks;
6. the chiral-model tables, the N=2 R-current check, the conductivities and the 1/N² coefficients;
7. the d = 3, Liendo–Rong and matrix-model results;
8. the odd fermion loops of 3d-only theories and the 𝒩=1 supercurrent and flavour-current relations that fix them;
9. the generated tables and figure.

It then compares the regenerated results with the published ones and writes a summary to `build/reproduce/summary.md`.

| Paper item | Produced by | Output |
|---|---|---|
| C_T and C_J through O(ε²) at any fixed point; two-coupling families; checks | `scripts/compute_results.py` | `results/results.md`, `results/coefficients.json` |
| C_T at O(ε³) | `scripts/decompose4.py`, `scripts/compute_results4.py` | `results/coefficients4_T.json` |
| C_J at O(ε³); N=2 supersymmetry check | `scripts/decompose4J.py`, `scripts/compute_results4J.py`, `scripts/susy_check4.py` | `results/coefficients4_J.json` |
| chiral Ising, XY and Heisenberg models through ε³ (table) | `scripts/paper_tables.py` | `results/paper_tables.md` §1 |
| U(1) current vs. R-current at the N=2 point, factor 9(d−2)/((d−1)(d+2)) | `scripts/paper_tables.py` | `results/paper_tables.md` §2 |
| Gross–Neveu–Yukawa model at large N: C_T1, C_J1, C_T2, C_J2 | `scripts/paper_tables.py` | `results/paper_tables.md` §5 |
| Liendo–Rong fixed points | `scripts/liendo_rong.py` | `results/liendo_rong.md` |
| matrix GNY models (table) and planar limit | `matrix/make_results.py`, `matrix/check.py` | `results/matrix_models.md` |
| estimates in d = 3 (table and figure) | `scripts/pade3d.py`, `scripts/plot_pade.py`, `matrix/make_results.py` | `results/pade_3d.md`, `results/pade_3d.pdf` |
| Gross–Neveu estimates in d = 3 | `scripts/pade3d.py` | `results/pade_3d.md` |
| universal conductivity (graphene table; topological-insulator surface) | `scripts/paper_tables.py` | `results/paper_tables.md` §3–4 |
| three-loop input checks | `scripts/check_betas3.py`, `scripts/check_gny3loop.py` | logs |
| G-scheme sums per invariant, diagram counts | `scripts/tabulate_gscheme.py`, `scripts/make_tables.py` | `results/gscheme_totals.md`, `results/gscheme4.tex`, `results/diagram_counts.tex` |
| odd fermion loops in 3d-only theories: the ⟨TT⟩ and ⟨JJ⟩ diagrams and their dependence on the continuation (§3.4, appendix E, table 9) | `scripts/odd_traces.py`, `ctcj/odd3.py` | `results/odd_traces.md`; see `ODD_LOOPS.md` |
| 𝒩=1 supercurrent and flavour-current multiplet: the relations that fix the odd loops, eqs. (5.9), (5.10), table 4, appendix F | `scripts/supercurrent_*.py`, `ctcj/supercurrent.py` | `results/supercurrent_eps2.md`, `_ising.md`, `_eps3.md`, `_jj.md`; see `ODD_LOOPS.md` |
| every diagram with its weight and value | `scripts/regenerate_diagrams.py`, `scripts/make_tables.py` | `results/diagram_catalogue.csv` |
