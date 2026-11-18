"""Three-loop RG functions of the chiral Ising, XY and Heisenberg GNY models, transcribed from
N. Zerf, L.N. Mihaila, P. Marquard, I.F. Herbut, M.M. Scherer, Phys. Rev. D 96 (2017) 096010, arXiv:1709.05057,
section III (one to three loops; the chiral Ising expressions agree with Mihaila et al., arXiv:1703.08801).

Their conventions: N = number of four-component Dirac fermions, y = g^2/(8 pi^2) (squared Yukawa coupling),
lambda/(8 pi^2) -> lambda, beta = d/d ln mu, d = 4 - eps.  Exponents: eta_phi = gamma_phi*, eta_psi = gamma_psi*,
1/nu = 2 - eta_phi + eta_phi2.  Their MS-bar scheme agrees with ours (and with Jack-Osborn-Steudtner) through three
loops: see scripts/gny_eps3.py, which checks the dictionary to our couplings against all one- and two-loop terms
and the transcription against the ancillary exponent files of arXiv:1709.05057 through eps^3.
"""
import sympy as sp

y, lam, N, eps = sp.symbols("y lambda N epsilon")
z3 = sp.zeta(3)
R = sp.Rational

RG = {
    "chiral Ising": dict(
        M=0,
        beta_y=[(3 + 2 * N) * y**2,
                24 * y * lam * (lam - y) - (R(9, 8) + 6 * N) * y**3,
                y / 64 * (1152 * (7 + 5 * N) * y**2 * lam + 192 * (91 - 30 * N) * y * lam**2
                          + (912 * z3 - 697 + 2 * N * (67 + 112 * N + 432 * z3)) * y**3 - 13824 * lam**3)],
        beta_lam=[36 * lam**2 + 4 * N * y * lam - N * y**2,
                  4 * N * y**3 + 7 * N * y**2 * lam - 72 * N * y * lam**2 - 816 * lam**3,
                  R(1, 32) * (6912 * (145 + 96 * z3) * lam**4 + 49536 * N * y * lam**3
                              - 48 * N * (72 * N - 361 - 648 * z3) * y**2 * lam**2
                              + 2 * N * (1736 * N - 4395 - 1872 * z3) * y**3 * lam
                              + N * (5 - 628 * N - 384 * z3) * y**4)],
        gamma_psi=[y / 2, -y**2 / 16 * (12 * N + 1),
                   y**3 / 128 * (48 * z3 + 4 * N * (47 - 12 * N) - 15) + 6 * lam * y**2 - R(33, 2) * lam**2 * y],
        gamma_phi=[2 * N * y, 24 * lam**2 - R(5, 2) * N * y**2,
                   -216 * lam**3 + R(1, 32) * N * y**3 * (48 * z3 + 200 * N + 21) + 30 * lam * N * y**2 - 90 * lam**2 * N * y],
        gamma_phi2=[-12 * lam, 144 * lam**2 - 2 * N * y * (y - 12 * lam),
                    R(3, 2) * N * y**2 * lam * (24 * N - 120 * z3 - 11) - 6264 * lam**3
                    - 4 * N * y**3 * (4 * N - 9 + 3 * z3) - 288 * N * y * lam**2],
    ),
    "chiral XY": dict(
        M=1,
        beta_y=[(2 + 2 * N) * y**2,
                32 * lam**2 * y + (R(7, 2) - 6 * N) * y**3 - 32 * lam * y**2,
                R(1, 8) * (N * (52 * N + 15) - 227) * y**4 + 24 * lam * (5 * N + 6) * y**3 - 120 * lam**2 * (N - 3) * y**2
                - 320 * lam**3 * y + 6 * z3 * (N + 1) * y**4],
        beta_lam=[40 * lam**2 + 4 * N * lam * y - N * y**2,
                  2 * N * y * (-40 * lam**2 + 2 * y**2 + lam * y) - 960 * lam**3,
                  39488 * lam**4 + R(1, 8) * N * (53 - 154 * N) * y**4 + 1808 * lam**3 * N * y + 2 * lam**2 * N * (509 - 60 * N) * y**2
                  - 12 * z3 * (N * y**2 * (-68 * lam**2 + y**2 + 7 * lam * y) - 2048 * lam**4)
                  + R(1, 4) * lam * N * (512 * N - 1339) * y**3],
        gamma_psi=[y, -y**2 / 4 * (6 * N + 1),
                   -R(1, 16) * (2 * N * (6 * N - 37) + 35) * y**3 + 3 * z3 * y**3 + 16 * lam * y**2 - 44 * lam**2 * y],
        gamma_phi=[2 * N * y, 32 * lam**2 - 3 * N * y**2,
                   -320 * lam**3 + 6 * z3 * N * y**3 + R(1, 8) * N * (64 * N - 27) * y**3 + 40 * lam * N * y**2 - 120 * lam**2 * N * y],
        gamma_phi2=[-16 * lam, 32 * lam * (6 * lam + N * y),
                    -4 * (2336 * lam**3 + N * (6 * N - 11) * y**3 + 96 * lam**2 * N * y
                          + 6 * z3 * N * y**2 * (8 * lam + y) + lam * N * (25 - 12 * N) * y**2)],
    ),
    "chiral Heisenberg": dict(
        M=2,
        beta_y=[(1 + 2 * N) * y**2,
                y * (40 * lam**2 + (R(47, 8) - 6 * N) * y**2 - 40 * lam * y),
                R(1, 64) * (608 * N**2 + 278 * N - 2731) * y**4 + R(9, 4) * z3 * (7 - 2 * N) * y**4 + 150 * lam * (N + 1) * y**3
                + 5 * lam**2 * (89 - 30 * N) * y**2 - 440 * lam**3 * y],
        beta_lam=[44 * lam**2 + 4 * N * y * lam - N * y**2,
                  N * y * (-88 * lam**2 + 4 * y**2 - 3 * lam * y) - 1104 * lam**3,
                  48184 * lam**4 + R(1, 32) * N * (365 - 604 * N) * y**4 + R(1, 16) * lam * N * (2360 * N - 6339) * y**3
                  + 3 * z3 * (9472 * lam**4 + N * y**2 * (212 * lam**2 - 4 * y**2 - 19 * lam * y))
                  + R(1, 2) * lam**2 * N * (3067 - 264 * N) * y**2 + 2068 * lam**3 * N * y],
        gamma_psi=[3 * y / 2, -R(9, 16) * (4 * N + 1) * y**2,
                   R(3, 128) * y * ((-48 * N**2 + 404 * N - 183) * y**2 - 3520 * lam**2 + 240 * z3 * y**2 + 1280 * lam * y)],
        gamma_phi=[2 * N * y, 40 * lam**2 - R(7, 2) * N * y**2,
                   -440 * lam**3 + R(15, 2) * z3 * N * y**3 + R(1, 32) * N * (312 * N - 131) * y**3 + 50 * lam * N * y**2 - 150 * lam**2 * N * y],
        gamma_phi2=[-20 * lam, 240 * lam**2 + 2 * N * y * (20 * lam + y),
                    -12920 * lam**3 + 8 * N * (7 - 4 * N) * y**3 - 480 * lam**2 * N * y
                    - 36 * z3 * N * y**2 * (5 * lam + y) + R(5, 2) * lam * N * (24 * N - 89) * y**2],
    ),
}


def full(model, name, loops=3):
    return sum(RG[model][name][:loops])


def fixed_point(model, Nval, order=3):
    """Stable non-Gaussian fixed point (y*, lambda*) as series in eps through eps^order, for a numerical N
    (Zerf normalisation).  Solved order by order: one-loop roots, then linear equations."""
    by = -eps * y + full(model, "beta_y")
    bl = -eps * lam + full(model, "beta_lam")
    by, bl = by.subs(N, Nval), bl.subs(N, Nval)
    M = RG[model]["M"]
    s = sp.sqrt(9 + 4 * Nval * (33 + Nval) + 20 * Nval * M + M**2 - 6 * M)
    y1 = 1 / (3 + 2 * sp.Integer(Nval) - M) if not isinstance(Nval, sp.Rational) else 1 / (3 + 2 * Nval - M)
    l1 = R(1, 8) * (3 - 2 * Nval - M + s) / ((3 + 2 * Nval - M) * (M + 9))
    ys, ls = [0, sp.nsimplify(y1)], [0, sp.nsimplify(l1)]
    t = sp.Symbol("t")
    for k in range(2, order + 1):
        yk, lk = sp.symbols("yk lk")
        Y = sum(ys[i] * t**i for i in range(1, k)) + yk * t**k
        L = sum(ls[i] * t**i for i in range(1, k)) + lk * t**k
        e1 = sp.expand(by.subs({y: Y, lam: L, eps: t}))
        e2 = sp.expand(bl.subs({y: Y, lam: L, eps: t}))
        sol = sp.solve([e1.coeff(t, k + 1), e2.coeff(t, k + 1)], [yk, lk], dict=True)[0]
        ys.append(sp.radsimp(sp.simplify(sol[yk])))
        ls.append(sp.radsimp(sp.simplify(sol[lk])))
    return ys, ls


def exponents(model, Nval, order=3):
    """eta_phi, eta_psi, 1/nu through eps^order at the stable fixed point."""
    ys, ls = fixed_point(model, Nval, order)
    Y = sum(ys[i] * eps**i for i in range(1, order + 1))
    L = sum(ls[i] * eps**i for i in range(1, order + 1))
    out = {}
    for name in ("gamma_phi", "gamma_psi", "gamma_phi2"):
        g = full(model, name).subs(N, Nval).subs({y: Y, lam: L})
        out[name] = sp.expand(sp.series(g, eps, 0, order + 1).removeO())
    return dict(etaphi=out["gamma_phi"], etapsi=out["gamma_psi"],
                nuinv=sp.expand(2 - out["gamma_phi"] + out["gamma_phi2"]))


def load_ancillary(path):
    """Parse the Mathematica exponent files shipped with arXiv:1709.05057 (anc/*.m)."""
    import re
    txt = open(path).read()
    out = {}
    for m in re.finditer(r"(etaphi|etapsi|nuinv)\s*=\s*(.*?);", txt, re.S):
        e = m.group(2)
        e = (e.replace("\\[Epsilon]", "eps").replace("Sqrt[", "sqrt(").replace("Zeta[", "zeta(")
             .replace("Pi", "pi").replace("[", "(").replace("]", ")").replace("^", "**"))
        out[m.group(1)] = sp.sympify(e, locals={"eps": eps, "Nf": N, "zeta": sp.zeta, "sqrt": sp.sqrt, "pi": sp.pi})
    return out
