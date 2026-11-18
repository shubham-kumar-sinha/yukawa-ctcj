"""Candidate flavour invariants for <JJ> at four loops, grouped by sector (y^6, y^4 lam, y^2 lam^2, lam^3).

* Trace words: products of traces over letters T (current generator) and contracted scalar labels.
  Canonical up to rotation of each trace and relabelling, NOT reflection (y, T hermitian, not real).
  Every trace carries an even number of y's (odd ones vanish: odd number of gamma matrices) and at
  least one y; the word must be connected (labels link traces; lambda links all its labels).
* Scalar networks: t, lambda and small trace blobs H = Tr(y^a y^b), TH = Tr(T y^a y^b), TTH, TyTy,
  all connected contractions without self-contractions of a node, deduplicated with nauty.
  t touching a y can be removed by the Ward identity t_ab y^b = i[y^a, T]; such terms are still
  generated (they are harmless) but listed after the T-only structures, so the independent subset
  prefers T-traces.
"""
import itertools

from pynauty import Graph, certificate

from .net import words_net

LABELS = "abcdefghijklmnop"


# ------------------------------------------------------------------ trace words
def canon_words(words, sym_labels=True):
    labs = sorted(set("".join(words)) - {"T"})
    best = None
    for perm in itertools.permutations(labs):
        m = dict(zip(labs, perm)); m["T"] = "T"
        parts = []
        for w in words:
            s = "".join(m[c] for c in w)
            parts.append(min(s[k:] + s[:k] for k in range(len(s))))
        key = tuple(sorted(parts))
        if best is None or key < best:
            best = key
    return best


def _compositions(n):
    for k in range(n):
        for cuts in itertools.combinations(range(1, n), k):
            b = (0,) + cuts + (n,)
            yield [b[i + 1] - b[i] for i in range(len(b) - 1)]


def _connected(words, all_linked=False):
    if all_linked:
        return all(any(c != "T" for c in w) for w in words)
    n = len(words)
    adj = {i: {j for j in range(n) if j != i and set(words[i]) & set(words[j]) - {"T"}} for i in range(n)}
    seen, st = {0}, [0]
    while st:
        u = st.pop()
        for v in adj[u] - seen:
            seen.add(v); st.append(v)
    return len(seen) == n


def trace_words(letters, all_linked=False):
    """All canonical connected products of traces using exactly the multiset `letters`."""
    out = set()
    for p in set(itertools.permutations(letters)):
        s = "".join(p)
        for comp in _compositions(len(s)):
            ws, i = [], 0
            for c in comp:
                ws.append(s[i:i + c]); i += c
            if any(sum(ch != "T" for ch in w) % 2 or all(ch == "T" for ch in w) for w in ws):
                continue
            if not _connected(ws, all_linked):
                continue
            out.add(canon_words(ws))
    return sorted(out, key=lambda k: (len(k), k))


def wname(ws):
    return "Tr(" + ")Tr(".join(ws) + ")"


def y6_candidates():
    return {wname(ws): words_net(list(ws)) for ws in trace_words("TTaabbcc")}


def y4lam_candidates():
    out = {}
    for ws in trace_words("TTabcd", all_linked=True):
        out["lam_abcd " + wname(ws)] = words_net(list(ws), extra=[("lam", "abcd")])
    return out


# ------------------------------------------------------------------ scalar networks
# node type: (number of legs, symmetric legs?, einsum builder)
NODE = {
    "lam": (4, True), "t": (2, False), "H": (2, True), "TH": (2, False), "TTH": (2, False), "TyTy": (2, True),
}
BLOB_WORD = {"H": "ab", "TH": "Tab", "TTH": "TTab", "TyTy": "TaTb"}


def _matchings(legs, node_of):
    if not legs:
        yield []
        return
    a = legs[0]
    for i in range(1, len(legs)):
        b = legs[i]
        if node_of[a] == node_of[b]:
            continue
        rest = legs[1:i] + legs[i + 1:]
        for m in _matchings(rest, node_of):
            yield [(a, b)] + m


def networks(nodes):
    """All connected contractions of the given node types, up to isomorphism.
    Returns list of (name, net, n_t_on_y) where net is an einsum network."""
    legs, node_of, legpos = [], {}, {}
    for n, typ in enumerate(nodes):
        for k in range(NODE[typ][0]):
            l = len(legs); legs.append(l); node_of[l] = n; legpos[l] = k
    nn = len(nodes)
    types = sorted(set(nodes))
    seen = {}
    for m in _matchings(legs, node_of):
        # connectivity of nodes
        adj = {i: set() for i in range(nn)}
        for a, b in m:
            adj[node_of[a]].add(node_of[b]); adj[node_of[b]].add(node_of[a])
        st, vis = [0], {0}
        while st:
            u = st.pop()
            for v in adj[u] - vis:
                vis.add(v); st.append(v)
        if len(vis) != nn:
            continue
        # nauty graph: node vertices, leg vertices for ordered nodes, line vertices
        vid = {}
        nv = nn
        legv = {}
        for l in legs:
            if not NODE[nodes[node_of[l]]][1]:
                legv[l] = nv; nv += 1
        line0 = nv
        nv += len(m)
        adjg = {i: [] for i in range(nv)}
        for l, v in legv.items():
            adjg[node_of[l]].append(v)
        for k, (a, b) in enumerate(m):
            for l in (a, b):
                adjg[line0 + k].append(legv.get(l, node_of[l]))
        col = [set(i for i in range(nn) if nodes[i] == ty) for ty in types]
        for ty in types:
            if not NODE[ty][1]:
                for k in range(NODE[ty][0]):
                    col.append({v for l, v in legv.items() if nodes[node_of[l]] == ty and legpos[l] == k})
        col.append(set(range(line0, nv)))
        col = [c for c in col if c]
        cert = certificate(Graph(nv, directed=False, adjacency_dict=adjg, vertex_coloring=col))
        if cert in seen:
            continue
        seen[cert] = m
    out = []
    for m in seen.values():
        lab = {}
        for k, (a, b) in enumerate(m):
            lab[a] = lab[b] = LABELS[k]
        words, extra, parts = [], [], []
        t_on_y = 0
        for n, typ in enumerate(nodes):
            ls = "".join(lab[l] for l in legs if node_of[l] == n)
            if typ in ("lam", "t"):
                extra.append((typ, ls)); parts.append(f"{typ}_{ls}")
            else:
                w = BLOB_WORD[typ]
                it = iter(ls)
                w = "".join(next(it) if ch in "ab" else ch for ch in w)
                words.append(w); parts.append(f"Tr({w})")
        for a, b in m:
            ta, tb = nodes[node_of[a]], nodes[node_of[b]]
            if (ta == "t" and tb in BLOB_WORD) or (tb == "t" and ta in BLOB_WORD):
                t_on_y += 1
        out.append((" ".join(parts), words_net(words, extra=extra), t_on_y))
    out.sort(key=lambda x: (x[2], x[0]))
    return out


SECTORS = {
    "y6": None,
    "y4lam": None,
    "y2lam2": [["lam", "lam", "TTH"], ["lam", "lam", "TyTy"], ["t", "lam", "lam", "TH"], ["t", "t", "lam", "lam", "H"]],
    "lam3": [["t", "t", "lam", "lam", "lam"]],
}


def candidates(sector):
    if sector == "y6":
        return y6_candidates()
    if sector == "y4lam":
        return y4lam_candidates()
    out = {}
    nets = []
    for nodes in SECTORS[sector]:
        nets += networks(nodes)
    nets.sort(key=lambda x: x[2])           # t touching y last
    for name, net, _ in nets:
        out[name] = net
    return out
