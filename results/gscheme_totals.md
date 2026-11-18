# G-scheme sums per invariant

X summed over all diagram classes with the given flavour invariant (D = 4 - 2 ep, Q^2 = 1, per-loop normalisation ep*G(1,1)),
<TT> = (z1.z2)^2 Q^4 (Q^2)^(-L ep) [ep G(1,1)]^L 2X/((D-1)(D+1)(D-2)),  <JJ> = (z1.z2) Q^2 (Q^2)^(-L ep) [ep G(1,1)]^L X/(D-1).
Only the orders needed for O(eps^3) are listed: L loops through ep^(3-L) (one loop through ep^2, four loops through ep^-1).
z3 = zeta_3.  Invariant names as in ctcj/formulas.py and ctcj/formulas4.py.

## <TT>

| L | invariant | ep^-4 | ep^-3 | ep^-2 | ep^-1 | ep^0 | ep^1 |
|---|---|---|---|---|---|---|---|
| 1 | Tr1 | 0 | 0 | 0 | 3/8 | -5/8 | 1/4 |
| 1 | ns | 0 | 0 | 0 | 1/4 | -1/4 | 0 |
| 2 | Tr(aa) | 0 | 0 | 0 | -5/32 | -133/192 | 15*z3/8 - 253/128 |
| 3 | Tr(aabb) | 0 | 0 | 5/48 | 581/576 | 65/12 - 51*z3/16 |  |
| 3 | Tr(ab)Tr(ab) | 0 | 0 | 5/96 | 1/2 | 9143/3456 - 3*z3/2 |  |
| 3 | Tr(abab) | 0 | 0 | 5/24 | 43/24 | 3721/432 - 145*z3/24 |  |
| 3 | lam.lam | 0 | 0 | 0 | -5/432 | -77/648 |  |
| 4 | Tr(aabbcc) | 0 | -5/128 | -7/12 | 9*z3/4 - 5303/1152 |  |  |
| 4 | Tr(aabc)Tr(bc) | 0 | -5/64 | -39/32 | 69*z3/16 - 7705/768 |  |  |
| 4 | Tr(aabcbc) | 0 | -5/16 | -141/32 | 555*z3/32 - 4223/128 |  |  |
| 4 | Tr(aabccb) | 0 | -5/128 | -911/1536 | 9*z3/4 - 22049/4608 |  |  |
| 4 | Tr(ab)Tr(ac)Tr(bc) | 0 | -5/256 | -443/1536 | 33*z3/32 - 20581/9216 |  |  |
| 4 | Tr(ab)Tr(acbc) | 0 | -5/32 | -1697/768 | 265*z3/32 - 4747/288 |  |  |
| 4 | Tr(abacbc) | 0 | -5/16 | -781/192 | 133*z3/8 - 32387/1152 |  |  |
| 4 | Tr(abcabc) | 0 | 0 | 5/64 | 131/128 - 15*z3/64 |  |  |
| 4 | lam Tr(abcd) | 0 | 0 | -35/192 | -3031/1152 |  |  |
| 4 | lam_abcd lam_abef lam_cdef | 0 | 0 | 5/192 | 463/1152 |  |  |
| 4 | lam_acde lam_bcde Tr(ab) | 0 | 0 | 95/4608 | 4631/13824 |  |  |

## <JJ>

| L | invariant | ep^-4 | ep^-3 | ep^-2 | ep^-1 | ep^0 | ep^1 |
|---|---|---|---|---|---|---|---|
| 1 | Tr(TT) | 0 | 0 | 0 | 1 | -1 | 0 |
| 1 | t.t | 0 | 0 | 0 | 1/2 | 0 | 0 |
| 2 | Tr(TTaa) | 0 | 0 | 0 | -3/4 | -23/8 | 6*z3 - 137/16 |
| 2 | Tr(TaTa) | 0 | 0 | 0 | 0 | 9/4 | 53/8 - 3*z3 |
| 3 | Tr(TTaabb) | 0 | 0 | 1/4 | 11/6 | 461/48 - 9*z3/2 |  |
| 3 | Tr(TTab)Tr(ab) | 0 | 0 | 1/4 | 11/6 | 473/48 - 9*z3/2 |  |
| 3 | Tr(TTabab) | 0 | 0 | 1 | 35/6 | 31 - 19*z3 |  |
| 3 | Tr(TTabba) | 0 | 0 | 1/4 | 47/24 | 523/48 - 6*z3 |  |
| 3 | Tr(TaTabb) | 0 | 0 | 0 | -11/4 | 6*z3 - 191/12 |  |
| 3 | Tr(TaTb)Tr(ab) | 0 | 0 | 0 | -1 | 3*z3/2 - 19/3 |  |
| 3 | Tr(TaTbab) | 0 | 0 | 0 | -7/2 | 8*z3 - 70/3 |  |
| 3 | Tr(TabTba) | 0 | 0 | 0 | 1/2 | 5/12 |  |
| 3 | t t lam lam | 0 | 0 | 0 | -1/8 | -11/12 |  |
| 4 | Tr(TTaabbcc) | 0 | -3/32 | -65/64 | 3*z3 - 965/128 |  |  |
| 4 | Tr(TTaabc)Tr(bc) | 0 | -9/64 | -63/64 | 3*z3 - 1725/256 |  |  |
| 4 | Tr(TTaabcbc) | 0 | -9/16 | -153/32 | 261*z3/16 - 2261/64 |  |  |
| 4 | Tr(TTaabccb) | 0 | -9/64 | -51/32 | 21*z3/4 - 3181/256 |  |  |
| 4 | Tr(TTab)Tr(abcc) | 0 | -3/64 | -19/16 | 3*z3 - 2667/256 |  |  |
| 4 | Tr(TTab)Tr(ac)Tr(bc) | 0 | -3/32 | -65/64 | 3*z3 - 997/128 |  |  |
| 4 | Tr(TTab)Tr(acbc) | 0 | -3/16 | -61/16 | 39*z3/4 - 2185/64 |  |  |
| 4 | Tr(TTab)Tr(accb) | 0 | -3/64 | -19/16 | 3*z3 - 2667/256 |  |  |
| 4 | Tr(TTabac)Tr(bc) | 0 | -9/32 | -111/64 | 15*z3/2 - 1383/128 |  |  |
| 4 | Tr(TTabacbc) | 0 | -3/4 | -35/8 | 21*z3 - 435/16 |  |  |
| 4 | Tr(TTabaccb) | 0 | -9/32 | -267/64 | 255*z3/16 - 4595/128 |  |  |
| 4 | Tr(TTabbc)Tr(ac) | 0 | -3/32 | -29/64 | 9*z3/4 - 349/128 |  |  |
| 4 | Tr(TTabbcac) | 0 | -9/32 | -75/64 | 99*z3/16 - 739/128 |  |  |
| 4 | Tr(TTabbcca) | 0 | -3/32 | -71/64 | 9*z3/2 - 1139/128 |  |  |
| 4 | Tr(TTabca)Tr(bc) | 0 | -3/64 | -5/8 | 9*z3/4 - 1403/256 |  |  |
| 4 | Tr(TTabcabc) | 0 | 0 | 3/8 | 59/16 - 9*z3/8 |  |  |
| 4 | Tr(TTabcacb) | 0 | -3/8 | -37/8 | 69*z3/4 - 637/16 |  |  |
| 4 | Tr(TTabcb)Tr(ac) | 0 | -9/32 | -111/64 | 15*z3/2 - 1383/128 |  |  |
| 4 | Tr(TTabcbac) | 0 | -3/8 | -25/8 | 51*z3/4 - 395/16 |  |  |
| 4 | Tr(TTabcbca) | 0 | -3/16 | -71/32 | 9*z3 - 1139/64 |  |  |
| 4 | Tr(TTabccab) | 0 | -3/16 | -77/32 | 141*z3/16 - 1217/64 |  |  |
| 4 | Tr(TTabccba) | 0 | -3/64 | -37/64 | 9*z3/4 - 1231/256 |  |  |
| 4 | Tr(TaTabbcc) | 0 | 0 | 3/2 | 103/8 - 9*z3/2 |  |  |
| 4 | Tr(TaTabc)Tr(bc) | 0 | 0 | 3/8 | 13/2 - 9*z3/4 |  |  |
| 4 | Tr(TaTabcbc) | 0 | 0 | 3 | 53/2 - 9*z3 |  |  |
| 4 | Tr(TaTabccb) | 0 | 0 | 3/4 | 109/16 - 9*z3/4 |  |  |
| 4 | Tr(TaTb)Tr(abcc) | 0 | 0 | 27/16 | 263/16 - 15*z3/4 |  |  |
| 4 | Tr(TaTb)Tr(ac)Tr(bc) | 0 | 0 | 15/32 | 295/64 - 3*z3/4 |  |  |
| 4 | Tr(TaTb)Tr(acbc) | 0 | 0 | 21/8 | 27 - 21*z3/4 |  |  |
| 4 | Tr(TaTbabcc) | 0 | 0 | 3 | 241/8 - 39*z3/4 |  |  |
| 4 | Tr(TaTbac)Tr(bc) | 0 | 0 | 3/4 | 17/2 - 9*z3/4 |  |  |
| 4 | Tr(TaTbacbc) | 0 | 0 | 3 | 121/4 - 9*z3 |  |  |
| 4 | Tr(TaTbaccb) | 0 | 0 | 3 | 241/8 - 39*z3/4 |  |  |
| 4 | Tr(TaTbbacc) | 0 | 0 | 3/4 | 53/8 - 9*z3/4 |  |  |
| 4 | Tr(TaTbcabc) | 0 | 0 | 0 | -7/4 |  |  |
| 4 | Tr(TaTbcacb) | 0 | 0 | 3 | 71/2 - 9*z3 |  |  |
| 4 | Tr(TaabcTcb) | 0 | 0 | -3/8 | -11/16 |  |  |
| 4 | Tr(Tab)Tr(Tcabc) | 0 | 0 | -3/8 | -17/16 |  |  |
| 4 | Tr(Tab)Tr(Tcbac) | 0 | 0 | 3/8 | 17/16 |  |  |
| 4 | Tr(TabTbcac) | 0 | 0 | -3/4 | -17/8 |  |  |
| 4 | Tr(TabTbcca) | 0 | 0 | -3/8 | -11/16 |  |  |
| 4 | Tr(TabTcabc) | 0 | 0 | 0 | -5/8 |  |  |
| 4 | Tr(TabTcbac) | 0 | 0 | 0 | -3/8 |  |  |
| 4 | Tr(TabTcbca) | 0 | 0 | -3/4 | -17/8 |  |  |
| 4 | Tr(TabcTcba) | 0 | 0 | 0 | -3/8 |  |  |
| 4 | lam_abcd Tr(TTabcd) | 0 | 0 | -9/8 | -183/16 |  |  |
| 4 | lam_abcd Tr(TaTbcd) | 0 | 0 | 3/4 | 77/8 |  |  |
| 4 | lam_abcd lam_abce Tr(TTde) | 0 | 0 | 3/64 | 17/32 |  |  |
| 4 | lam_abcd lam_abce Tr(TdTe) | 0 | 0 | -1/32 | -77/192 |  |  |
| 4 | t_ab t_ac lam_bdef lam_cdeg Tr(fg) | 0 | 0 | 3/64 | 71/128 |  |  |
| 4 | t_ab t_ac lam_bdef lam_cdgh lam_efgh | 0 | 0 | 3/32 | 63/64 |  |  |
| 4 | t_ab t_cd lam_acef lam_bdgh lam_efgh | 0 | 0 | 0 | 1/32 |  |  |

