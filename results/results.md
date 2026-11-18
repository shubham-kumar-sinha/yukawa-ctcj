# Results

Couplings rescaled y -> 4 pi y, lambda -> 16 pi^2 lambda; Tr = tr(1) x flavour trace
(JOS normalisation, gamma_phi^(1)ab = 1/2 Tr(y^a y^b)).  Valid at any fixed point, through O(eps^2).

## C_T / C_T,scalar

| invariant | coefficient |
|---|---|
| ns | 1 |
| Tr1 | (d - 1)/2 |
| Tr(aa) | -5/4 |
| Tr(ab)Tr(ab) | -1/6 |
| Tr(aabb) | -11/48 |
| Tr(abab) | -19/6 |
| Tr(aa)Tr(bb) | 0 |
| lam.lam | -5/36 |
| lam_aacd lam_bbcd | 0 |

## C_J / C_J,fermion (per unit Tr(TT))

| invariant | coefficient |
|---|---|
| Tr(TT) | 1 |
| t.t | 1/(d - 2) |
| Tr(TTaa) | -3/2 |
| Tr(TaTa) | 0 |
| Tr(TTaabb) | -1 |
| Tr(TTabab) | -17/2 |
| Tr(TTabba) | -5/8 |
| Tr(TaTabb) | 3/4 |
| Tr(TaTbab) | 15/2 |
| Tr(TabTba) | 3/2 |
| Tr(TTab)Tr(ab) | -1 |
| Tr(TaTb)Tr(ab) | 3/2 |
| Tr(Tab)Tr(Tab) | 0 |
| Tr(TTaa)Tr(bb) | 0 |
| t t lam lam | -3/8 |

## Models (lambda_+ fixed point), ratios to the free theory

### chiral Ising (GNY)

- C_T: 1 + (-5*N/(2*(N + 6)*(3*N + 2))) eps + ((-2253*N**4 - 15*N**3*sqrt(N**2 + 132*N + 36) - 62153*N**3 - 2980*N**2*sqrt(N**2 + 132*N + 36) - 217302*N**2 - 3060*N*sqrt(N**2 + 132*N + 36) - 93672*N - 720*sqrt(N**2 + 132*N + 36) - 4320)/(648*(9*N**5 + 174*N**4 + 1192*N**3 + 3312*N**2 + 3024*N + 864))) eps^2
- C_J: 1 + (-3/(2*(N + 6))) eps + ((37*N**2 - N*sqrt(N**2 + 132*N + 36) - 327*N - 66*sqrt(N**2 + 132*N + 36) - 1044)/(72*(N**3 + 18*N**2 + 108*N + 216))) eps^2

### chiral XY (NJLY)

- C_T: 1 + (-5*N/((N + 4)*(3*N + 4))) eps + ((-417*N**4 - 3*N**3*sqrt(N**2 + 152*N + 16) - 7854*N**3 - 688*N**2*sqrt(N**2 + 152*N + 16) - 5144*N**2 - 1008*N*sqrt(N**2 + 152*N + 16) + 8576*N - 128*sqrt(N**2 + 152*N + 16) - 512)/(60*(9*N**5 + 132*N**4 + 736*N**3 + 1920*N**2 + 2304*N + 1024))) eps^2
- C_J: 1 + (-3*N/(N + 4)**2) eps + ((-503*N**3 + 3*N**2*sqrt(N**2 + 152*N + 16) - 3550*N**2 - 228*N*sqrt(N**2 + 152*N + 16) + 2464*N - 96*sqrt(N**2 + 152*N + 16) - 384)/(100*(N**4 + 16*N**3 + 96*N**2 + 256*N + 256))) eps^2

### chiral Heisenberg

- C_T: 1 + (-5*N/(2*(N + 2)**2)) eps + ((-3363*N**3 - 25*N**2*sqrt(N**2 + 172*N + 4) - 27337*N**2 - 6450*N*sqrt(N**2 + 172*N + 4) + 177908*N - 200*sqrt(N**2 + 172*N + 4) - 400)/(2904*(N**4 + 8*N**3 + 24*N**2 + 32*N + 16))) eps^2
- C_J: 1 + (-9*N/(2*(N + 2)*(N + 4))) eps + (3*(-1457*N**4 + 5*N**3*sqrt(N**2 + 172*N + 4) - 7993*N**3 - 1270*N**2*sqrt(N**2 + 172*N + 4) + 30816*N**2 - 5240*N*sqrt(N**2 + 172*N + 4) + 169360*N - 320*sqrt(N**2 + 172*N + 4) - 640)/(968*(N**5 + 14*N**4 + 76*N**3 + 200*N**2 + 256*N + 128))) eps^2

## Cross-checks

| check | result | detail |
|---|---|---|
| diagram classes T (1,2,3-loop Yukawa, lambda^2) = 2,6,43,2;  J = 3,11,76,6 | PASS | T [2, 6, 43, 2], J [3, 11, 76, 6] |
| free values: C_T fermion/scalar = (d-1)/2 per Tr1; C_J scalar/fermion = 1/(d-2) | PASS | {'ns': 1, 'Tr1': (d - 1)/2}, {'Tr(TT)': 1, 't.t': 1/(d - 2)} |
| two-loop <TT> diagrams reproduce the project table | PASS | [(-3/32, -17/640, -3239/12800), (1/16, -53/960, 1259/19200), (-3/32, -17/640, -3239/12800), (1/8, 9/160, 393/3200), (1/8, 9/160, 393/3200), (-1/8, -77 |
| 11 large-N diagrams sum to (0, 5/96, 1/2) | PASS | (0, 5/96, 1/2) |
| <TT>: every pole cancels per invariant; log(mu^2 x^2) drops at fixed points | PASS |  |
| <JJ>: every pole cancels per invariant; log(mu^2 x^2) drops at fixed points | PASS |  |
| Ward-identity relations among <JJ> invariants hold on U(1) and SU(2) ensembles | PASS |  |
| O(n) Wilson-Fisher: C_T, C_J eps^2 terms (Osborn-Stergiou / Petkou) | PASS | 5*epsilon**2*(-n - 2)/(12*(n + 8)**2), 3*epsilon**2*(-n - 2)/(4*(n + 8)**2) |
| GNY two-loop beta_y from JOS tensors = FGKT | PASS | -3*N*y2**2 + 3*lambda**2/4 - 6*lambda*y2 - 9*y2**2/4 |
| GNY O(eps): C_T (FGKT eq. 2.9) and C_J | PASS | -5*N/(6*N**2 + 40*N + 24), -3/(2*N + 12) |
| large N: C_T1, C_J1 = exact Gross-Neveu results of Diab et al. (1601.07198) | PASS | C_T1 = -17*epsilon**2/54 - 11*epsilon/18 + 2/3, C_J1 = epsilon**2/2 - 3*epsilon/2 |
| large N: eps^2 coefficient of C_T scales with n_s | PASS | [-7/18, -7/18, -7/18] |
| N -> 0 of the three models = O(n) Wilson-Fisher C_T (n = 1, 2, 3) | PASS | [-5/324, -1/60, -25/1452] |
| N=2 SUSY (chiral XY, N=2): C_J^R/C_J^R,free = C_T/C_T,free to O(eps^2) | PASS | C_T: -epsilon**2/8 - epsilon/6 + 1;  C_J^R: -epsilon**2/8 - epsilon/6 + 1 |
| frozen coefficients in ctcj/formulas.py equal the assembled ones | PASS |  |
