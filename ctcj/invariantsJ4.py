"""Four-loop <JJ>: invariant basis, class decomposition and beta-function insertions.

Sectors by coupling order: y6, y4lam, y2lam2, lam3.  Each sector's basis is the greedy independent
subset (on U(1)-symmetric theories, ctcj.ensembleJ) of the candidates in ctcj.basisJ4, so it is unique
modulo the Ward identities.  Insertions D I[beta] are computed exactly by replacing one tensor at a time
in the einsum network; their lambda-sectors are separated exactly by scaling lambda -> s lambda.
"""
import numpy as np

from .basisJ4 import candidates
from .decompJ import eval_matrix, independent, decompose
from .ensembleJ import ensemble
from .invariants import J_BASIS
from .net import evaluate, words_net
from . import betasJ as B

SECTORS = ["y6", "y4lam", "y2lam2", "lam3"]
SECTOR_OF = {(6, 0): "y6", (4, 1): "y4lam", (2, 2): "y2lam2", (0, 3): "lam3"}
LAMDEG = {"y6": 0, "y4lam": 1, "y2lam2": 2, "lam3": 3}

# two- and three-loop words (names as in ctcj.invariants.J_BASIS)
W2 = {"Tr(TTaa)": ["TTaa"], "Tr(TaTa)": ["TaTa"]}
# Tr(TTaa)Tr(bb) is disconnected (a control): its three-loop total is identically zero (asserted in the
# assembly), and its beta-insertion would leave the connected four-loop basis, so it is not inserted.
DISCONNECTED3 = ["Tr(TTaa)Tr(bb)"]
W3 = {k: [p.rstrip(")") for p in k.split("Tr(") if p] for k in J_BASIS[4] if k not in DISCONNECTED3}
NET2 = {k: words_net(v) for k, v in W2.items()}
NET3 = {k: words_net(v) for k, v in W3.items()}
NET_TTLL = words_net([], extra=[("t", "ae"), ("t", "bf"), ("lam", "abcd"), ("lam", "efcd")])


def default_ensemble(n=120, seed=20261028):
    return ensemble(n, families=("A", "B", "C"), seed=seed)


def build_basis(ths):
    out = {}
    for s in SECTORS:
        cand = candidates(s)
        names = list(cand)
        M = eval_matrix([cand[k] for k in names], ths)
        ind, Mi = independent(names, M)
        out[s] = dict(names=ind, nets={k: cand[k] for k in ind}, M=Mi)
    return out


# ------------------------------------------------------------------ beta functions on a theory
def betas(th, s=1.0):
    """beta1, beta2 (y), beta_lambda^(1) and D2 = 1/2 Dbeta1[beta1], with lambda -> s*lambda (cached)."""
    cache = th.setdefault("_betas", {})
    if s not in cache:
        y, lam, m = th["y"], s * th["lam"], th["mult"]
        b1 = B.beta1_y(y, m)
        f = lambda u: B.beta1_y(y + u * b1, m)
        D2 = 0.5 * (8 * (f(1) - f(-1)) - (f(2) - f(-2))) / 12
        cache[s] = (b1, B.beta2_y(y, lam, m), B.beta1_lam(y, lam, m), D2)
    return cache[s]


def _positions(net, name):
    return [k for k, (n, _) in enumerate(net[0]) if n == name]


def D1(net, th, v, name="y"):
    """first-order insertion: sum over factors `name` replaced by v."""
    return sum(evaluate(net, th, {k: v}) for k in _positions(net, name))


def D2half(net, th, v, name="y"):
    """1/2 D^2 I[v, v]: sum over unordered pairs of factors replaced by v."""
    pos = _positions(net, name)
    return sum(evaluate(net, th, {pos[i]: v, pos[j]: v}) for i in range(len(pos)) for j in range(i + 1, len(pos)))


def _split3(fun):
    """fun(s) is a polynomial of degree <= 3 in s; return its coefficients [c0, c1, c2, c3]."""
    S = [0.0, 1.0, -1.0, 2.0]
    V = np.array([[s ** k for k in range(4)] for s in S])
    return np.linalg.solve(V, np.array([fun(s) for s in S]))


def insertion_values(th):
    """Values of every insertion on theory th.  Keys: (word, kind, sector)."""
    out = {}
    b1, _, _, D2 = betas(th)
    for w, net in NET2.items():
        out[(w, "b1", "y4")] = D1(net, th, b1)
        out[(w, "11", "y6")] = D2half(net, th, b1) + D1(net, th, D2)
        cs = _split3(lambda s: D1(net, th, betas(th, s)[1]))
        for sec in SECTORS:
            if sec != "lam3":
                out[(w, "b2", sec)] = cs[LAMDEG[sec]]
    for k, net in NET3.items():
        out[(k, "b1", "y6")] = D1(net, th, b1)
    cs = _split3(lambda s: s * D1(NET_TTLL, th, betas(th, s)[2], name="lam"))
    for sec in ("y4lam", "y2lam2", "lam3"):
        out[("t t lam lam", "bl", sec)] = cs[LAMDEG[sec]]
    return out


def decompose_insertions(ths, basis4):
    """{(word, kind): {invariant: coefficient}} on the four-loop basis (and three-loop basis for b1 into W2)."""
    vals = [insertion_values(th) for th in ths]
    keys = sorted(vals[0])
    names3 = list(J_BASIS[4])
    M3 = eval_matrix([words_net([q.rstrip(")") for q in k.split("Tr(") if q]) for k in names3], ths)
    res = {}
    for key in keys:
        v = np.array([x[key] for x in vals])
        w, kind, sec = key
        if sec == "y4":
            d = decompose(v, names3, M3)
        else:
            if np.linalg.norm(v) < 1e-9 * max(1.0, max(abs(np.array([x[kk] for x in vals])).max() for kk in keys)):
                d = {}
            else:
                try:
                    d = decompose(v, basis4[sec]["names"], basis4[sec]["M"])
                except ValueError as err:
                    raise ValueError(f"insertion {key}: {err}")
        res.setdefault((w, kind), {}).update(d)
    return res
