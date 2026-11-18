"""<S_mu Sbar_nu> for the 3d N=1 supercurrent (ODD_LOOPS.md): diagram builder, FORM projection, Forcer.

Feynman rules (scripts/supercurrent_rules.py; conventions of ctcj/forcer.py):
    fermion propagator -i pslash/p^2 (p along the arrow), scalar 1/p^2, Yukawa -h (flavour stripped),
    closed fermion loop (-1) x Tr (FORM tr 1 = 4 -> 1)
    S_D at x:    (i qslash) gamma_mu    S_W at x:    + gamma_mu       (q = scalar momentum INTO the vertex)
    Sbar_D at 0: gamma_nu (i qslash)    Sbar_W at 0: - gamma_nu
The open fermion line runs from Sbar(0) to S(x); its spinor chain is written against the arrow, from x to 0:
G_{mu nu} = V_x,mu ... V_0,nu.

Projections (Q^2 = 1, Q flows in at x and out at 0, as in ctcj/forcer.py):
    "P":  X  = Tr[G_{mu nu} Qslash P_{nu mu}],  P_{nu mu} = pi_{nu mu} - gT_nu gT_mu/(D-1)  (spin 3/2, gamma-traceless)
    "W1": Tr[Q_mu G_{mu nu} gamma_nu]           (conservation at x)
    "W2": Tr[Q_mu G_{mu nu} Q_nu Qslash]        (conservation at x, longitudinal at 0)
    "W3": Tr[G_{mu nu} Q_nu gamma_mu]           (conservation at 0)

A diagram: dict(name, ops=(x-type, 0-type) in {"D","W"}, ferm=[(a, b), ...] fermion edges a -> b along the arrow
(open path from "0" to "x", plus closed loops), sig=[(a, b), ...] scalar edges (momentum a -> b), weight=symmetry
factor x flavour factor).
"""
import os
import subprocess
from fractions import Fraction

import sympy as sp

from .forcer import EP, Z3, Z4, Z5, FORM_BIN, _env

PROJ = {
    "P": "g_(1,Q)*(d_(nu,mu)-Q(nu)*Q(mu)) - 1/(D-1)*g_(1,Q)*(g_(1,nu)-Q(nu)*g_(1,Q))*(g_(1,mu)-Q(mu)*g_(1,Q))",
    "W1": "Q(mu)*g_(1,nu)",
    "W2": "Q(mu)*Q(nu)*g_(1,Q)",
    "W3": "Q(nu)*g_(1,mu)",
    "Pj": "g_(1,Q)",                 # <j jbar>, j = t_ab phi_a psi_b (flavour-current multiplet): Tr[G Qslash]
}


def build(c):
    ferm = [tuple(e) for e in c["ferm"]]
    sig = [tuple(e) for e in c["sig"]]
    edges = [("f", a, b) for a, b in ferm] + [("s", a, b) for a, b in sig]
    P = {i: f"p{i + 1}" for i in range(len(edges))}
    vin = {}
    for i, (_, a, b) in enumerate(edges):
        vin.setdefault(b, []).append(P[i])
        vin.setdefault(a, []).append(f"-{P[i]}")
    vin["x"].insert(0, "Q")
    vin["0"].insert(0, "-Q")
    vx = "*".join(f"vx({','.join(m)})" for m in vin.values())
    props = "".join(f"/{p}.{p}" for p in P.values())

    def scalar_into(v):
        out = [(1, P[i]) for i, (t, a, b) in enumerate(edges) if t == "s" and b == v] + \
              [(-1, P[i]) for i, (t, a, b) in enumerate(edges) if t == "s" and a == v]
        assert len(out) >= 1
        return out

    coef, ipow = Fraction(c.get("weight", 1)) * (-1) ** c.get("nQ", 0), 0         # quartic vertex -lambda
    pred = {b: a for a, b in ferm}
    fe = {(a, b): P[i] for i, (t, a, b) in enumerate(edges) if t == "f"}
    tx, t0 = c["ops"]
    # open line: x -> ... -> 0 against the arrow
    chain, v = [], "x"
    if tx == "D":
        sgn, q = scalar_into("x")[0]
        chain += [f"g_(1,{q})", "g_(1,mu)"]
        coef *= sgn
        ipow += 1
    elif tx == "W":
        chain.append("g_(1,mu)")                          # tx == "j": vertex t_ab, unit spinor matrix
    while v != "0":
        u = pred[v]
        chain.append(f"g_(1,{fe[(u, v)]})")
        coef *= -1
        ipow += 1
        if u != "0":
            coef *= -1                                   # Yukawa
        v = u
    if t0 == "D":
        sgn, q = scalar_into("0")[0]
        chain += ["g_(1,nu)", f"g_(1,{q})"]
        coef *= sgn
        ipow += 1
    elif t0 == "W":
        chain.append("g_(1,nu)")
        coef *= -1
    coef /= 4                                            # FORM tr 1 = 4 -> 1 on the open line too
    fac = ["*".join(chain)]
    # closed loops: vertices not on the open path
    on_path = set()
    v = "x"
    while v != "0":
        on_path.add((pred[v], v))
        v = pred[v]
    rest = [e for e in ferm if e not in on_path]
    li = 1
    while rest:
        li += 1
        a0, b0 = rest[0]
        loop, v = [], a0
        while True:                                       # follow pred backwards
            u = pred[v]
            loop.append(f"g_({li},{fe[(u, v)]})")
            rest.remove((u, v))
            coef *= -1                                   # Yukawa at v
            coef *= -1
            ipow += 1                                     # -i pslash
            v = u
            if v == a0:
                break
        coef *= Fraction(-1, 4)                           # closed loop, tr 1 = 4 -> 1
        fac.append("*".join(loop))
    return ",".join(P.values()), fac, coef, ipow, vx, props, li


def run_diagram(c, workdir, proj="P", depth=5):
    os.makedirs(workdir, exist_ok=True)
    name = f"{c['name']}_{proj}"
    vecs, fac, coef, ipow, vx, props, nl = build(c)
    tr = "".join(f"tracen,{l};" for l in range(nl, 0, -1))
    step1 = f"""#-
Off Statistics;
Dimension D;
Vectors {vecs},Q;
Indices mu,nu;
Local F = ({coef.numerator})/({coef.denominator})*{'*'.join(fac)}*({PROJ[proj]});
{tr}
contract;
id Q.Q = 1;
.sort
Format nospaces;
#write <{name}_num.h> "#define NUM \\"%E\\"", F
.end
"""
    open(os.path.join(workdir, f"{name}_1.frm"), "w").write(step1)
    r = subprocess.run([FORM_BIN, "-q", f"{name}_1.frm"], cwd=workdir, env=_env(), capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    numtxt = open(os.path.join(workdir, f"{name}_num.h")).read()
    if '"0"' in numtxt.replace(" ", ""):
        return sp.Integer(0)
    open(os.path.join(workdir, f"{name}_num.h"), "w").write(numtxt.replace("D", "DD"))
    res = os.path.join(workdir, f"{name}_res.txt")
    if os.path.exists(res):
        os.remove(res)
    step2 = f"""#-
Off Statistics;
#include- forcer.h
S DD;
#include- {name}_num.h
Local F = (`NUM'){props}*{vx};
id DD = 4-2*ep;
#call Forcer(gschemeexpand={depth})
.sort
Format nospaces;
Format Mathematica;
#write <{name}_res.txt> "%E", F
.end
"""
    open(os.path.join(workdir, f"{name}_2.frm"), "w").write(step2)
    r = subprocess.run([FORM_BIN, "-q", f"{name}_2.frm"], cwd=workdir, env=_env(), capture_output=True, text=True)
    if r.returncode or not os.path.exists(res):
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-2000:])
    txt = open(res).read().strip().rstrip(";").replace("^", "**")
    X = sp.sympify(txt, locals={"ep": EP, "z3": Z3, "z4": Z4, "z5": Z5, "DD": sp.Symbol("DD")})
    X = X.subs(sp.Symbol("DD"), 4 - 2 * EP)          # 1/(D-1) of the projector survives Forcer's id DD = 4-2*ep
    return sp.expand(sp.series(sp.I ** ipow * X, EP, 0, depth - 1).removeO())


# ------------------------------------------------------------------ Wick-contraction generator
# Open fermion line from Sbar(0) to S(x); D-type operators have one scalar leg, W-type two equivalent ones
# (S_W = 1/2 h_abc phi_b phi_c gamma_mu psi_a: weight 1/2 per W operator, the leg permutations are generated).
# Weight of a labelled contraction: 1/(nY! nQ! 24^nQ) x (1/2)^(#W), as in ctcj.diagrams.
import itertools as _it
import math as _math
from collections import defaultdict as _dd

import numpy as _np

from .diagrams import perfect_matchings, biconnected


def _components(vertices, edges):
    adj = _dd(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    left, comps = set(vertices), []
    while left:
        s = left.pop()
        comp, st = {s}, [s]
        while st:
            u = st.pop()
            for w in adj[u]:
                if w in left:
                    left.discard(w)
                    comp.add(w)
                    st.append(w)
        comps.append(comp)
    return comps


def not_scaleless(vertices, edges):
    """Connected, and every cut vertex separates x from 0 (otherwise a blob carries no momentum: zero)."""
    if len(_components(vertices, edges)) != 1:
        return False
    for v in vertices:
        rest = [w for w in vertices if w != v]
        for comp in _components(rest, [(a, b) for a, b in edges if v not in (a, b)]):
            if "x" not in comp and "0" not in comp:
                return False
    return True


def _legs(ops, Y, Qv):
    tx, t0 = ops
    legs = [("x", k) for k in range(1 if tx in "Dj" else 2)] + [("0", k) for k in range(1 if t0 in "Dj" else 2)]
    return legs + [(y, 0) for y in Y] + [(q, k) for q in Qv for k in range(4)]


def _labelled(nY, nQ, ops, rule):
    """Yield (ferm, matching) for every labelled contraction passing the zero rule."""
    Y = [f"Y{i}" for i in range(nY)]
    Qv = [f"Q{i}" for i in range(nQ)]
    verts = ["x", "0"] + Y + Qv
    legs = _legs(ops, Y, Qv)
    if len(legs) % 2:
        return
    keep = biconnected if rule == "biconnected" else not_scaleless
    sources, targets = ["0"] + Y, Y + ["x"]
    for perm in _it.permutations(targets):
        ferm = list(zip(sources, perm))
        if any(a == b for a, b in ferm):
            continue
        for m in perfect_matchings(legs):
            if any(a[0] == b[0] for a, b in m):
                continue
            if not keep(verts, ferm + [(a[0], b[0]) for a, b in m]):
                continue
            yield ferm, m


def _weight(nY, nQ, ops):
    return Fraction(1, _math.factorial(nY) * _math.factorial(nQ) * 24 ** nQ) * Fraction(1, 2) ** ops.count("W")


def generate_S(nY, nQ, ops=("D", "D"), rule="biconnected"):
    """Classes of <S Sbar> diagrams, canonical relabelling by brute force (ctcj.diagrams style)."""
    Y = [f"Y{i}" for i in range(nY)]
    Qv = [f"Q{i}" for i in range(nQ)]
    classes = {}
    for ferm, m in _labelled(nY, nQ, ops, rule):
        best = None
        for py in _it.permutations(range(nY)):
            for pq in _it.permutations(range(nQ)):
                ren = {"x": "x", "0": "0"}
                ren.update({Y[i]: f"Y{py[i]}" for i in range(nY)})
                ren.update({Qv[i]: f"Q{pq[i]}" for i in range(nQ)})
                f = tuple(sorted((ren[a], ren[b]) for a, b in ferm))
                s = tuple(sorted(tuple(sorted((ren[a[0]], ren[b[0]]))) for a, b in m))
                if best is None or (f, s) < best:
                    best = (f, s)
        c = classes.setdefault(best, dict(ops=tuple(ops), ferm=best[0], sig=best[1], weight=Fraction(0),
                                          nY=nY, nQ=nQ))
        c["weight"] += _weight(nY, nQ, ops)
    out = [classes[k] for k in sorted(classes)]
    for i, c in enumerate(out):
        c["name"] = f"S{nY}{nQ}{''.join(ops)}_{i:03d}"
    return out


def generate_S_fast(nY, nQ, ops=("D", "D"), rule="biconnected", reps=False):
    """Same classes via nauty certificates (ctcj.diagrams_fast style).  Returns {certificate: weight}, or with
    reps=True a list of classes (first labelled contraction of each certificate as representative), named
    S<nY><nQ><ops>n_<i> in certificate order."""
    from pynauty import Graph, certificate
    Y = [f"Y{i}" for i in range(nY)]
    Qv = [f"Q{i}" for i in range(nQ)]
    verts = ["x", "0"] + Y + Qv
    vid = {v: i for i, v in enumerate(verts)}
    nv = len(verts)
    nF = nY + 1
    nS = len(_legs(ops, Y, Qv)) // 2
    nT0, nH0 = nv, nv + nF
    nS0 = nH0 + nF
    ntot = nS0 + nS
    rep = {}
    coloring = [c for c in ({0}, {1}, set(range(2, 2 + nY)), set(range(2 + nY, nv)), set(range(nT0, nH0)),
                            set(range(nH0, nS0)), set(range(nS0, ntot))) if c]
    out = {}
    for ferm, m in _labelled(nY, nQ, ops, rule):
        adj = {i: [] for i in range(ntot)}
        for k, (u, v) in enumerate(ferm):
            t, h = nT0 + k, nH0 + k
            adj[vid[u]].append(t); adj[t].append(h); adj[h].append(vid[v])
        for k, (a, b) in enumerate(m):
            adj[nS0 + k] += [vid[a[0]], vid[b[0]]]
        cert = certificate(Graph(ntot, directed=False, adjacency_dict=adj, vertex_coloring=coloring))
        out[cert] = out.get(cert, Fraction(0)) + _weight(nY, nQ, ops)
        if reps and cert not in rep:
            rep[cert] = (tuple(ferm), tuple((a[0], b[0]) for a, b in m))
    if not reps:
        return out
    cl = []
    for i, cert in enumerate(sorted(out)):
        f, sg = rep[cert]
        cl.append(dict(ops=tuple(ops), ferm=f, sig=sg, weight=out[cert], nY=nY, nQ=nQ,
                       name=f"S{nY}{nQ}{''.join(ops)}n_{i:03d}"))
    return cl


# ------------------------------------------------------------------ flavour factors (N=1: y^a_bc = h_abc)
def flavour(c, h, lam=None, mult=1.0, t=None):
    """Flavour factor of class c as one tensor contraction (numpy einsum): every scalar line and every fermion propagator
    carries an index; Yukawa vertex h[s, f_out, f_in], quartic lambda[s1..s4], D end delta(s, f), W end h[f, s1, s2],
    j end t[s, f]; each closed fermion loop a factor mult.  Same value as flavour_loops (checked in
    scripts/supercurrent_rules.py), much faster for many fields."""
    sig = [tuple(e) for e in c["sig"]]
    ferm = [tuple(e) for e in c["ferm"]]
    n = h.shape[0]
    ns = len(sig)
    sidx = _dd(list)                                     # vertex -> scalar-line indices of its legs
    for k, (a, b) in enumerate(sig):
        sidx[a].append(k)
        sidx[b].append(k)
    fid = {e: ns + k for k, e in enumerate(ferm)}        # fermion propagator indices
    fin = {b: fid[(a, b)] for a, b in ferm}              # propagator into vertex
    fout = {a: fid[(a, b)] for a, b in ferm}             # propagator out of vertex
    ops = []
    tx, t0 = c["ops"]
    for v, typ, f in (("x", tx, fin.get("x")), ("0", t0, fout.get("0"))):
        if typ == "D":
            ops += [_np.eye(n), [sidx[v][0], f]]
        elif typ == "j":
            ops += [t, [sidx[v][0], f]]
        else:
            ops += [h, [f, sidx[v][0], sidx[v][1]]]
    for v in sidx:
        if v.startswith("Y"):
            ops += [h, [sidx[v][0], fout[v], fin[v]]]
        elif v.startswith("Q"):
            ops += [lam, list(sidx[v])]
    # closed fermion loops = cycles among the vertices off the open path 0 -> ... -> x
    pred, succ = {b: a for a, b in ferm}, dict(ferm)
    on_path, v = {"x"}, "x"
    while v != "0":
        v = pred[v]
        on_path.add(v)
    loops, seen = 0, set()
    for a, _ in ferm:
        if a in on_path or a in seen:
            continue
        loops += 1
        u = a
        while u not in seen:
            seen.add(u)
            u = succ[u]
    val = _np.einsum(*ops, [], optimize="greedy")
    return float(_np.real(val)) * mult ** loops


def flavour_loops(c, h, lam=None, mult=1.0, t=None):
    """Flavour factor of class c: open line u_x . y^{s1} ... y^{sn} . u_0 (D end: delta between the scalar leg and
    the fermion index; W end: h_{f, legs}), closed loops mult x trace, quartic lambda.  Explicit index sums (slow;
    reference implementation for flavour)."""
    sig = [tuple(e) for e in c["sig"]]
    legs = _dd(list)
    for k, (a, b) in enumerate(sig):
        legs[a].append(k)
        legs[b].append(k)
    ferm = [tuple(e) for e in c["ferm"]]
    pred = {b: a for a, b in ferm}
    path, v = ["x"], "x"
    while v != "0":
        v = pred[v]
        path.append(v)
    on = set(zip(path[1:], path[:-1]))
    rest = [e for e in ferm if e not in on]
    loops = []
    while rest:
        a0 = rest[0][0]
        L, v = [], a0
        while True:
            u = pred[v]
            rest.remove((u, v))
            L.append(v)
            v = u
            if v == a0:
                break
        loops.append(L)
    tx, t0 = c["ops"]
    n = h.shape[0]
    tot = 0.0
    for idx in _it.product(range(n), repeat=len(sig)):
        val = 1.0
        for v, ks in legs.items():
            if v.startswith("Q"):
                val *= lam[tuple(idx[k] for k in ks)]
        if val == 0:
            continue
        end = lambda v, typ: (_np.eye(n)[idx[legs[v][0]]] if typ == "D" else t[idx[legs[v][0]]] if typ == "j"
                              else h[:, idx[legs[v][0]], idx[legs[v][1]]])
        ux, u0 = end("x", tx), end("0", t0)
        vec = ux
        for v in path[1:-1]:
            vec = vec @ h[idx[legs[v][0]]]
        val *= vec @ u0
        for L in loops:
            M = _np.eye(n)
            for v in L:
                M = M @ h[idx[legs[v][0]]]
            val *= mult * _np.trace(M)
        tot += val
    return float(tot)


def has_odd_loop(c):
    """True if class c has a closed fermion loop with an odd number of Yukawa vertices (zero in our Dirac algebra)."""
    ferm = [tuple(e) for e in c["ferm"]]
    pred, succ = {b: a for a, b in ferm}, dict(ferm)
    on_path, v = {"x"}, "x"
    while v != "0":
        v = pred[v]
        on_path.add(v)
    seen = set()
    for a, _ in ferm:
        if a in on_path or a in seen:
            continue
        L, u = 0, a
        while u not in seen:
            seen.add(u)
            u = succ[u]
            L += 1
        if L % 2:
            return True
    return False
