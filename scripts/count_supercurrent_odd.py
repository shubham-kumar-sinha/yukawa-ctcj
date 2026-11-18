#!/usr/bin/env python3
"""Which <S Sbar> graphs (3d N=1 supercurrent) can carry DREG_3 odd traces?  Existence count, not Feynman rules.
See ODD_LOOPS.md.  Result: none through O(eps^3); the first at O(eps^4) (five loops).

    python3 scripts/count_supercurrent_odd.py      (a few seconds)


Vertices: S(x), Sbar(0): one fermion leg + 1 scalar leg ('D': dslash phi gamma_mu psi, or the improvement term)
or 2 scalar legs ('W': W_a(phi) gamma_mu psi, carries one Yukawa coupling).  Yukawa: 2 fermion + 1 scalar (weight 1).
Quartic: 4 scalars (weight 2).  Weight w = #Yukawa + #W + 2 #quartic; the term contributes at O(eps^(w/2)).
The fermion line through S is open (S has one fermion field), so closed loops contain only Yukawa vertices.
A closed loop with an odd number n of Yukawa vertices gives one eps~ whose three slots are filled by loop momenta;
after integrating the loop it lies in the span of the loop's external momenta, so it survives only if the loop has
E_ext >= 4 external scalar lines (>= 3 independent momenta).  The open line, projected so that the tree level is a
non-vanishing even trace, gives an eps~ iff (#Yukawa on it + #W) is odd.  A parity-even contribution to C_S needs an
even number of eps~ in total.
"""
import itertools
from collections import Counter

def matchings(items):
    if not items:
        yield []
        return
    a = items[0]
    for i in range(1, len(items)):
        for m in matchings(items[1:i] + items[i + 1:]):
            yield [(a, items[i])] + m

def cycle_partitions(n, minsize=2):
    """partitions of n into cycle sizes >= minsize (sizes only; vertices are interchangeable)"""
    def rec(n, maxp):
        if n == 0:
            yield []
            return
        for p in range(min(n, maxp), minsize - 1, -1):
            for rest in rec(n - p, p):
                yield [p] + rest
    yield from rec(n, n)

def connected(nodes, edges):
    adj = {v: set() for v in nodes}
    for a, b in edges:
        adj[a].add(b); adj[b].add(a)
    seen, st = {nodes[0]}, [nodes[0]]
    while st:
        u = st.pop()
        for w in adj[u] - seen:
            seen.add(w); st.append(w)
    return len(seen) == len(nodes)

res = Counter()
examples = {}
for w in range(0, 9):
    for tx, t0 in (("D", "D"), ("D", "W"), ("W", "W")):
        nW = (tx == "W") + (t0 == "W")
        for nQ in range(0, (w - nW) // 2 + 1):
            nY = w - nW - 2 * nQ
            if nY < 0:
                continue
            for nopen in range(0, nY + 1):
                for parts in cycle_partitions(nY - nopen):
                    if not any(p % 2 for p in parts):
                        continue                      # no odd closed loop
                    Y = [f"Y{i}" for i in range(nY)]
                    openY, rest = Y[:nopen], Y[nopen:]
                    loops, k = [], 0
                    for p in parts:
                        loops.append(rest[k:k + p]); k += p
                    ferm = list(zip(["0"] + openY, openY + ["x"]))
                    for L in loops:
                        ferm += list(zip(L, L[1:] + L[:1]))
                    Q = [f"Q{i}" for i in range(nQ)]
                    legs = [("x", 0)] + ([("x", 1)] if tx == "W" else []) + [("0", 0)] + ([("0", 1)] if t0 == "W" else []) \
                        + [(y, 0) for y in Y] + [(q, j) for q in Q for j in range(4)]
                    if len(legs) % 2:
                        continue
                    nodes = ["x", "0"] + Y + Q
                    status = None
                    for m in matchings(legs):
                        sc = [(a[0], b[0]) for a, b in m]
                        if any(a == b for a, b in sc):
                            continue                  # scalar tadpole (massless: zero)
                        if not connected(nodes, ferm + sc):
                            continue
                        n_eps = (nopen + nW) % 2
                        ok = True
                        for L in loops:
                            if len(L) % 2:
                                n_eps += 1
                                Ls = set(L)
                                ext = sum(1 for a, b in sc if (a in Ls) != (b in Ls))
                                ok &= ext >= 4
                        st = "survives" if ok and n_eps % 2 == 0 else ("parity-odd" if ok else "vanishes (E_ext<4)")
                        if status != "survives":
                            status = st if status is None or st == "survives" else status
                        if st == "survives":
                            break
                    if status is None:
                        continue
                    L4 = len(ferm) + (len(legs) // 2) - len(nodes) + 1
                    res[(w, status)] += 1
                    if status == "survives" and (w, status) not in examples:
                        examples[(w, status)] = dict(ops=(tx, t0), nY=nY, nopen=nopen, loops=parts, nQ=nQ, loops_L=L4)
for w in range(9):
    row = {s: res[(w, s)] for s in ("survives", "parity-odd", "vanishes (E_ext<4)")}
    if any(row.values()):
        print(f"weight {w} (O(eps^{w / 2:g})): vertex configurations with an odd closed loop:", row)
for k, v in examples.items():
    print("first surviving example at weight", k[0], v)
assert all(res[(w, "survives")] == 0 for w in range(7)), "an odd-trace contribution at O(eps^3) or below"
print("OK: no surviving odd-trace contribution to <S Sbar> through O(eps^3)")
