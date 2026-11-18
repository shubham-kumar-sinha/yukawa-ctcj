"""FORM (traces + projection) and Forcer (IBP reduction) for one diagram class.

Euclidean Feynman rules (momenta of fermion lines along the arrow):
    fermion propagator  -i pslash / p^2          scalar propagator  1/p^2
    Yukawa vertex       -y^a (flavour stripped)  quartic vertex     -lambda_abcd (stripped)
    closed fermion loop (-1) x Tr                (Tr = tr 1 x flavour trace; tr 1 -> 1 here)
    Tpsi  (i/2) zslash z.(p_in + p_out)          Tsig  (i z.q1)(i z.q2), q into the vertex
    Jpsi  zslash (x T)                           Jphi  z.q_d t_ab, q_d = momentum of derivative leg into vertex
Projection (Q^2 = 1, z.Q = 0):
    T: replace z^mu z^nu (x) and z^rho z^si (0) by (D-1) P_{mu nu, rho si}(Q); X_T = coefficient
       so that <T T> = (z1.z2)^2 Q^4 (Q^2)^(-L ep) [ep G(1,1)]^L 2 X_T / ((D-1)(D+1)(D-2)).
    J: contract with pi^{mu nu}(Q); <J J> = (z1.z2) Q^2 (Q^2)^(-L ep) [ep G(1,1)]^L X_J / (D-1).
G-scheme: each loop normalised by ep*G(1,1) at Q^2 = 1, D = 4 - 2 ep (Forcer convention).
"""
import os
import subprocess
from fractions import Fraction

import sympy as sp

from .diagrams import loops_of

EP = sp.Symbol("ep")
Z3, Z4, Z5 = sp.symbols("z3 z4 z5")
FORM_BIN = os.environ.get("FORM_BIN", "form")
FORCER_PATH = os.environ.get("FORCER_PATH", os.path.expanduser("~/forcer"))


def _env():
    return dict(os.environ, FORMPATH=f"{FORCER_PATH}:{FORCER_PATH}/forcer")


def build(c):
    """Return (vecs, numerator factors, rational coefficient, power of i, vx string, props, ntraces)."""
    ferm, sig = list(c["ferm"]), list(c["sig"])
    edges = [("f", a, b, None, None) for a, b in ferm] + [("s", a[0], b[0], a[1], b[1]) for a, b in sig]
    P = {i: f"p{i + 1}" for i in range(len(edges))}
    vin = {}
    for i, e in enumerate(edges):              # momentum p_i flows e[1] -> e[2]
        vin.setdefault(e[2], []).append(P[i])
        vin.setdefault(e[1], []).append(f"-{P[i]}")
    vin["x"].insert(0, "Q")
    vin["0"].insert(0, "-Q")
    vx = "*".join(f"vx({','.join(m)})" for m in vin.values())
    props = "".join(f"/{p}.{p}" for p in P.values())
    T = c["kind"] == "T"
    idx = {"x": ("mu", "nu"), "0": ("rho", "si")} if T else {"x": ("mu",), "0": ("nu",)}
    ops = dict(zip(("x", "0"), c["ops"]))
    fac, coef, ipow = [], Fraction(1), 0
    fe = {(e[1], e[2]): i for i, e in enumerate(edges) if e[0] == "f"}
    pred = {b: a for a, b in ferm}
    succ = dict(ferm)
    loops = loops_of(ferm)
    for li, L in enumerate(loops, start=1):
        coef *= Fraction(-1, 4)                 # closed loop; FORM tr 1 = 4 -> 1
        chain, v = [], L[0]
        while True:                             # spinor chain against the arrows
            if v in ("x", "0"):
                chain.append(f"g_({li},{idx[v][0]})")
                if ops[v] == "Tpsi":
                    pin, pout = P[fe[(pred[v], v)]], P[fe[(v, succ[v])]]
                    fac.append(f"({pin}({idx[v][1]})+{pout}({idx[v][1]}))")
                    coef /= 2
                    ipow += 1
            else:
                coef *= -1                      # Yukawa
            chain.append(f"g_({li},{P[fe[(pred[v], v)]]})")
            coef *= -1
            ipow += 1                           # -i pslash
            v = pred[v]
            if v == L[0]:
                break
        fac.append("*".join(chain))
    for v in ("x", "0"):
        if ops[v] == "Tsig":
            qs = []
            for i, e in enumerate(edges):
                if e[0] == "s":
                    if e[2] == v:
                        qs.append((1, P[i]))
                    if e[1] == v:
                        qs.append((-1, P[i]))
            fac.append(f"{qs[0][1]}({idx[v][0]})*{qs[1][1]}({idx[v][1]})")
            coef *= qs[0][0] * qs[1][0]
            ipow += 2
        elif ops[v] == "Jphi":
            for i, e in enumerate(edges):
                if e[0] == "s":
                    if e[2] == v and e[4] == 0:
                        fac.append(f"{P[i]}({idx[v][0]})")
                    if e[1] == v and e[3] == 0:
                        fac.append(f"{P[i]}({idx[v][0]})")
                        coef *= -1
    coef *= (-1) ** c["nQ"] * c["weight"]
    return ",".join(P.values()), fac, coef, ipow, vx, props, len(loops)


PROJ_T = "( (D-1)/2*(pi(mu,rho)*pi(nu,si) + pi(mu,si)*pi(nu,rho)) - pi(mu,nu)*pi(rho,si) )"
PROJ_J = "pi(mu,nu)"


def run_class(c, workdir, depth=5):
    """Return X(ep) for class c (sympy expression in ep, z3, z4, z5)."""
    os.makedirs(workdir, exist_ok=True)
    name = c["name"]
    vecs, fac, coef, ipow, vx, props, nl = build(c)
    T = c["kind"] == "T"
    tr = "".join(f"tracen,{l};" for l in range(nl, 0, -1))
    num = "*".join(fac) if fac else "1"
    step1 = f"""#-
Off Statistics;
Dimension D;
Vectors {vecs},Q;
Indices mu,nu,rho,si;
CFunction pi;
Local F = ({coef.numerator})/({coef.denominator})*{num}*{PROJ_T if T else PROJ_J};
id pi(mu?,nu?) = d_(mu,nu) - Q(mu)*Q(nu);
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
    if r.returncode or not os.path.exists(res):
        raise RuntimeError(r.stdout[-3000:] + r.stderr[-2000:])
    txt = open(res).read().strip().rstrip(";").replace("^", "**")
    X = sp.sympify(txt, locals={"ep": EP, "z3": Z3, "z4": Z4, "z5": Z5})
    return sp.expand(sp.I ** ipow * X)
