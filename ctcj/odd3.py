"""Odd-trace (DREG_3) contributions: fermion loops with an odd number of gamma matrices, in the semi-naive
prescription of Zerf et al. (arXiv:1709.05057):
    Tr(g_a g_b g_c) = i eps~_abc tr1,   eps~_abc eps~_def = det[delta_(abc),(def)]  (D-dimensional deltas),
with the even-trace algebra unchanged.  Longer odd traces reduce to three-gamma traces through the Clifford product
expansion Tr(g_a1..g_a5) = sum_{i<j} (-1)^(j-i-1) delta_{ai aj} Tr(remaining three).
Handles classes in which every fermion loop is odd and there are exactly two loops (the four-loop classes)."""
import itertools, os, re, subprocess
import sympy as sp
from sympy.combinatorics import Permutation
from .forcer import build, _env, FORM_BIN, EP, Z3, Z4, Z5, PROJ_T, PROJ_J

INDICES = {"mu", "nu", "rho", "si"}


def _delta(a, b):
    ia, ib = a in INDICES, b in INDICES
    if ia and ib:
        return f"d_({a},{b})"
    if ia:
        return f"{b}({a})"
    if ib:
        return f"{a}({b})"
    return f"{a}.{b}"


def odd_trace(elems):
    """Tr(g_e1...g_en)/(i tr1) for odd n, as a list of (sign, [deltas], (a,b,c)) meaning sign*prod(deltas)*eps~(a,b,c)."""
    n = len(elems)
    if n == 3:
        return [(1, [], tuple(elems))]
    out = []
    for i, j in itertools.combinations(range(n), 2):
        s = -1 if (j - i - 1) % 2 else 1
        rest = [e for k, e in enumerate(elems) if k not in (i, j)]
        for s2, ds, abc in odd_trace(rest):
            out.append((s * s2, [_delta(elems[i], elems[j])] + ds, abc))
    return out


def pauli_trace(elems):
    """Tr(s_a s_b s_c s_d s_e)/(2i) from the explicit three-dimensional algebra s_a s_b = delta_ab + i eps_abc s_c,
    multiplied out from the left (with eps_abx eps_xcd = delta delta - delta delta inside the chain):
        delta_ab eps_cde - delta_ac eps_bde + delta_bc eps_ade + delta_de eps_abc.
    Equal to odd_trace for three-dimensional indices (Schouten identity), but a different function of D."""
    a, b, c, d, e = elems
    return [(1, [_delta(a, b)], (c, d, e)), (-1, [_delta(a, c)], (b, d, e)), (1, [_delta(b, c)], (a, d, e)),
            (1, [_delta(d, e)], (a, b, c))]


def eps_eps(abc, defg):
    terms = []
    for p in itertools.permutations(range(3)):
        sgn = Permutation(list(p)).signature()
        terms.append((sgn, [_delta(abc[k], defg[p[k]]) for k in range(3)]))
    return terms


def numerator(chains, trace=odd_trace):
    """FORM string for prod of two odd traces (each /(i tr1)), with eps~ eps~ -> det."""
    t1, t2 = trace(chains[0]), trace(chains[1])
    parts = []
    for s1, d1, e1 in t1:
        for s2, d2, e2 in t2:
            for s3, d3 in eps_eps(e1, e2):
                s = s1 * s2 * s3
                parts.append(("+" if s > 0 else "-") + "*".join(d1 + d2 + d3))
    return "(" + "".join(parts) + ")"


def numerator_transport(chains, p, trace=odd_trace):
    """As numerator(), with the two grade-3 parts x, y paired through the inversion tensor of the scalar line with
    momentum p that joins the two reading vertices.  In position space, with n the unit vector along that line,
    the conformally covariant pairing is x.(2 n^(n_|y) - y) (equal to x.y in D = 3); the Fourier transform of
    n_r n_s/z^(D-2) is (delta_rs/2 - p_r p_s/p^2)/(1 - ep)/p^2 in units of the scalar propagator, which gives
        [(2 + ep) x.y - 2 (p_|x).(p_|y)/p^2]/(1 - ep),   2 + ep = (8 - D)/2.
    Returns the FORM string without the factors 1/p^2 (second term only) and 1/(1 - ep): the first term carries an
    explicit p.p instead, so that the caller adds one factor 1/p.p and 1/(1 - ep) overall."""
    t1, t2 = trace(chains[0]), trace(chains[1])
    parts = []
    for s1, d1, e1 in t1:
        for s2, d2, e2 in t2:
            for s3, d3 in eps_eps(e1, e2):
                s = s1 * s2 * s3
                parts.append(("+" if s > 0 else "-") + "(8-D)/2*" + f"{p}.{p}*" + "*".join(d1 + d2 + d3))
            for i in range(3):
                for j in range(3):
                    ri = [e1[k] for k in range(3) if k != i]
                    rj = [e2[k] for k in range(3) if k != j]
                    for s3, d3 in ((1, [_delta(ri[0], rj[0]), _delta(ri[1], rj[1])]),
                                   (-1, [_delta(ri[0], rj[1]), _delta(ri[1], rj[0])])):
                        s = -s1 * s2 * s3 * (-1) ** (i + j)
                        parts.append(("+" if s > 0 else "-") + "2*" + "*".join(d1 + d2 + [_delta(p, e1[i]), _delta(p, e2[j])] + d3))
    return "(" + "".join(parts) + ")"


def line_reading(c, k):
    """(r1, r2) that start both traces at the two ends of scalar line k, and the momentum name of that line"""
    ferm = list(c["ferm"])
    pred = {b: a for a, b in ferm}
    pos = {}
    for op in ("x", "0"):
        v, j = pred[op], 1
        while v != op:
            pos[v] = (op, j + 1)
            v, j = pred[v], j + 1
    a, b = c["sig"][k]
    ends = {pos[a[0]][0]: pos[a[0]][1], pos[b[0]][0]: pos[b[0]][1]}
    order = [l[0] if l[0] in ("x", "0") else next(v for v in l if v in ("x", "0")) for l in __import__("ctcj.diagrams", fromlist=["loops_of"]).loops_of(ferm)]
    return tuple(ends[o] for o in order), f"p{len(ferm) + k + 1}"


def run_class_odd(c, workdir, depth=3, read=(0, 0), pauli=False, line=None, fierz=False):
    """X(ep) of class c with both fermion loops odd.  read = (r1, r2): reading point of each trace, i.e. the chain
    [operator, S, S, S, S] (against the arrows, as built by forcer.build) is rotated cyclically by r places before
    the Clifford expansion.  r = 0 starts at the operator.  The odd trace is not cyclic for D != 3, so r matters.
    pauli=True uses pauli_trace (explicit 3d algebra) instead of the Clifford expansion odd_trace.
    line=k: both traces are read at the ends of scalar line k and paired through its inversion tensor
    (numerator_transport); read is then taken from the line unless given explicitly.
    fierz=True: no eps~ at all.  The two odd traces (2x2 algebra) are sewn into one even trace by the
    three-dimensional Fierz identity Tr(A) Tr(B) = 1/2 [Tr(A B) + Tr(A g_m B g_m)], cut at the reading points
    (A, B = the two rotated chains), and the even trace is taken in D dimensions (FORM tracen).  With line=k the cut
    points are the two ends of scalar line k (no inversion tensor in this mode)."""
    os.makedirs(workdir, exist_ok=True)
    pline = None
    if line is not None:
        lread, pline = line_reading(c, line)
        read = lread if tuple(read) == (0, 0) else read      # an explicit read overrides (for tests)
        if fierz:
            pline = None
    name = c["name"] + "_odd" + ("" if tuple(read) == (0, 0) else f"_r{read[0]}{read[1]}") + ("_pauli" if pauli else "") \
        + ("" if line is None or fierz else f"_line{line}") + ("_fierz" if fierz else "")
    vecs, fac, coef, ipow, vx, props, nl = build(c)
    chains, other = [], []
    for f in fac:
        if f.startswith("g_("):
            chains.append(re.findall(r"g_\(\d+,([^)]*)\)", f))
        else:
            other.append(f)
    assert len(chains) == 2 and all(len(ch) % 2 for ch in chains), c["name"]
    chains = [ch[r % len(ch):] + ch[:r % len(ch)] for ch, r in zip(chains, read)]
    tr = pauli_trace if pauli else odd_trace
    if fierz:
        # coef carries 1/4 per loop, i.e. 16 x t(A) t(B) with t = Tr/tr1; Fierz with tr1 = 2 gives
        # t(A) t(B) = [t(AB) + t(A g B g)]/4, and FORM's tracen has tr1 = 4: 16 t(A) t(B) = TrF(AB) + TrF(A g B g)
        A, B = chains
        num = "*".join(other + [f"(g_(1,{','.join(A + B)})+g_(1,{','.join(A + ['fz'] + B + ['fz'])}))"])
    else:
        ipow += 2                               # Tr(ggg) = i eps~ tr1 for each loop
        num = "*".join(other + ["16*" + (numerator(chains, tr) if pline is None else numerator_transport(chains, pline, tr))])   # tr1 = 4 per loop (coef carries 1/4 per loop)
    T = c["kind"] == "T"
    step1 = f"""#-
Off Statistics;
Dimension D;
Vectors {vecs},Q;
Indices mu,nu,rho,si,fz;
CFunction pi;
Local F = ({coef.numerator})/({coef.denominator})*{num}*{PROJ_T if T else PROJ_J};
id pi(mu?,nu?) = d_(mu,nu) - Q(mu)*Q(nu);
{"tracen,1;" if fierz else ""}
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
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-2000:])
    numtxt = open(os.path.join(workdir, f"{name}_num.h")).read()
    if '"0"' in numtxt.replace(" ", ""):
        return sp.Integer(0)
    open(os.path.join(workdir, f"{name}_num.h"), "w").write(numtxt.replace("D", "DD"))
    step2 = f"""#-
Off Statistics;
#include- forcer.h
S DD;
#include- {name}_num.h
Local F = (`NUM'){props}{"" if pline is None else f"/{pline}.{pline}"}*{vx};
id DD = 4-2*ep;
#call Forcer(gschemeexpand={depth})
.sort
Format nospaces;
Format Mathematica;
#write <{name}_res.txt> "%E", F
.end
"""
    res = os.path.join(workdir, f"{name}_res.txt")
    if os.path.exists(res):
        os.remove(res)
    open(os.path.join(workdir, f"{name}_2.frm"), "w").write(step2)
    r = subprocess.run([FORM_BIN, "-q", f"{name}_2.frm"], cwd=workdir, env=_env(), capture_output=True, text=True)
    if r.returncode or not os.path.exists(res):
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-2000:])
    txt = open(res).read().strip().rstrip(";").replace("^", "**")
    X = sp.sympify(txt, locals={"ep": EP, "z3": Z3, "z4": Z4, "z5": Z5})
    if pline is not None:                       # overall 1/(1 - ep), to the order computed
        lo, hi = -4, depth - 4
        X = sum(sp.expand(X).coeff(EP, n) * EP ** n for n in range(lo, hi + 1))
        X = sp.expand(X * sum(EP ** n for n in range(0, hi - lo + 1)))
        X = sum(X.coeff(EP, n) * EP ** n for n in range(lo, hi + 1))
    return sp.expand(sp.I ** ipow * X)


def run_vertex3(perm, read=0, exit_at=2, depth=3, workdir="build/odd_vertex"):
    """Three-loop subgraph <J psi psibar> of the six <JJ> classes: the J loop (x -> A1 -> A2 -> A3 -> x), three scalar
    lines A_i - B_perm[i], and the open fermion line B1 -> B2 -> B3.  Q enters at x and leaves at B_{exit_at+1}; the
    other external fermion leg carries zero momentum (infrared safe).  The open chain gamma(l1) gamma(l2) is projected
    on its Gamma^(2) part with eps~(mu, l1, l2), so the result is proportional to the coefficient of
    eps~_{mu a b} gamma_a gamma_b in the vertex; its 1/ep pole is the overall (momentum-independent) divergence.
    read = reading point of the five-gamma trace, as in run_class_odd.  Returns the G-scheme series in ep."""
    os.makedirs(workdir, exist_ok=True)
    name = "V3_" + "".join(map(str, perm)) + f"_r{read}_e{exit_at}"
    A, B = ["A1", "A2", "A3"], ["B1", "B2", "B3"]
    edges = [("x", "A1"), ("A1", "A2"), ("A2", "A3"), ("A3", "x"), ("B1", "B2"), ("B2", "B3")]
    edges += [(A[i], B[perm[i]]) for i in range(3)]
    P = [f"p{i + 1}" for i in range(len(edges))]
    vin = {}
    for p, (a, b) in zip(P, edges):
        vin.setdefault(b, []).append(p)
        vin.setdefault(a, []).append("-" + p)
    vin["x"].insert(0, "Q")
    vin[B[exit_at]].insert(0, "-Q")
    props = "".join(f"/{p}.{p}" for p in P)
    (v2, (m1, m2)), = [(v, m) for v, m in vin.items() if len(m) == 2]     # zero-momentum leg: merge the two lines
    del vin[v2]
    a, b = m1.lstrip("-"), m2.lstrip("-")
    sub = lambda t: t.replace(b, a) if (m1[0] == "-") != (m2[0] == "-") else t.replace("-" + b, "+" + a).replace(b, "-" + a).replace("+", "")
    vx = "*".join(f"vx({','.join(sub(x) for x in m)})" for m in vin.values())
    props = props.replace(f"/{b}.{b}", f"/{a}.{a}")
    assert exit_at in (0, 2) and b in ("p7", "p8", "p9")
    chain = ["mu", "p4", "p3", "p2", "p1"]
    chain = chain[read:] + chain[:read]
    parts = []
    for s1, d1, e1 in odd_trace(chain):
        for s3, d3 in eps_eps(e1, ("mu", "p6", "p5")):
            parts.append(("+" if s1 * s3 > 0 else "-") + "*".join(d1 + d3))
    step1 = f"""#-
Off Statistics;
Dimension D;
Vectors {','.join(P)},Q;
Indices mu;
Local F = ({''.join(parts)});
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
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-2000:])
    numtxt = open(os.path.join(workdir, f"{name}_num.h")).read()
    open(os.path.join(workdir, f"{name}_num.h"), "w").write(numtxt.replace("D", "DD"))
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
    res = os.path.join(workdir, f"{name}_res.txt")
    if os.path.exists(res):
        os.remove(res)
    open(os.path.join(workdir, f"{name}_2.frm"), "w").write(step2)
    r = subprocess.run([FORM_BIN, "-q", f"{name}_2.frm"], cwd=workdir, env=_env(), capture_output=True, text=True)
    if r.returncode or not os.path.exists(res) or "WARNING" in r.stdout:
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-2000:])
    txt = open(res).read().strip().rstrip(";").replace("^", "**")
    return sp.expand(sp.sympify(txt, locals={"ep": EP, "z3": Z3, "z4": Z4, "z5": Z5}))
