#!/usr/bin/env python3
"""Figure of the two-sided Pade approximants of scripts/pade3d.py for 2 <= d <= 4.
Writes results/pade_3d.pdf.  (~2 min)"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import sympy as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import gamma as G
from ctcj.eps3models import series, eps
from ctcj.pade import all_pades

ROOT = os.path.join(os.path.dirname(__file__), "..")
d = np.linspace(2, 4, 401)
x = 4 - d


def coeffs(s, K=3):
    return [float(sp.N(s.coeff(eps, k), 30)) for k in range(K + 1)]


def draw(ax, a, b, color, label, transform=lambda v: v):
    fs = all_pades(a, b, functions=True)
    for i, (k, f) in enumerate(fs):
        ax.plot(d, transform(f(x)), color=color, lw=1.0, alpha=0.75, label=label if i == 0 else None)
    return fs


fig, axs = plt.subplots(2, 2, figsize=(10, 7.6))
# (a) N=2
ax = axs[0, 0]
a2 = coeffs(series("chiral XY", 2, "T"))
draw(ax, a2, [1 / 3], "C0", "Pade, $\\epsilon^3$ + $d{=}2$")
draw(ax, a2[:3], [1 / 3], "0.6", "Pade, $\\epsilon^2$ + $d{=}2$")
ax.plot(d, sum(c * x**k for k, c in enumerate(a2)), "k:", lw=1, label="series to $\\epsilon^3$")
ax.plot(3, 256 / 243 - 16 / (9 * np.sqrt(3) * np.pi), "r*", ms=11, label="exact (localization)")
ax.plot(2, 1 / 3, "ks", ms=5, label="$d{=}2$: $c=1$")
ax.set_title("$\\mathcal{N}=2$ super-Ising: $C_T/C_{T,\\rm free}$")
# (b) N=1
ax = axs[0, 1]
a1 = coeffs(series("chiral Ising", 1, "T", odd=True))       # 3d-only: odd fermion-loop term included
draw(ax, a1, [7 / 15], "C0", "Pade, $\\epsilon^3$ + $d{=}2$")
draw(ax, a1[:3], [7 / 15], "0.6", "Pade, $\\epsilon^2$ + $d{=}2$")
ax.plot(d, sum(c * x**k for k, c in enumerate(a1)), "k:", lw=1, label="series to $\\epsilon^3$")
ax.plot(3, 1.68414 / 2, "rD", ms=6, label="bootstrap (Atanasov et al.)")
ax.plot(2, 7 / 15, "ks", ms=5, label="$d{=}2$: $c=7/10$")
ax.set_title("$\\mathcal{N}=1$ super-Ising: $C_T/C_{T,\\rm free}$")
# (c), (d) Gross-Neveu
for N, col in ((4, "C1"), (8, "C2")):
    aJ = coeffs(series("chiral Ising", N, "J"))
    bJ = [1.0, -1 / (N - 2), -1 / (2 * (N - 2) ** 2)]
    draw(axs[1, 0], aJ, bJ, col, f"$N={N}$")
    aF = coeffs(sp.expand(series("chiral Ising", N, "T") - 1))
    bF = [-2 / (N + 2), 2 * N / (N + 2) ** 2, -2 * N ** 2 / (N + 2) ** 3,
          (15 * N ** 5 - 69 * N ** 4 + 58 * N ** 3 + 4 * N ** 2 + 8 * N) / (8 * (N - 2) ** 2 * (N + 2) ** 4)]
    draw(axs[1, 1], aF, bF, col, f"$N={N}$", transform=lambda v: (1 + v) * (1 + 2 / (N * (d - 1))))
    e = 4 - d[1:-1]
    eta1 = G(d[1:-1] - 1) * (d[1:-1] - 2) ** 2 * (e / 2) / (4 * G(1 + e / 2) * G(d[1:-1] / 2 + 1) * G(d[1:-1] / 2) ** 2)
    axs[1, 0].plot(d[1:-1], 1 - 8 * (d[1:-1] - 1) / (d[1:-1] * (d[1:-1] - 2)) * eta1 / N, "--", color=col, lw=1)
    axs[1, 1].plot(3, 1 + 8 / (9 * np.pi ** 2 * N), "o", color=col, mfc="none")
axs[1, 0].set_title("Gross–Neveu: $C_J/C_{J,\\rm free}$ (dashed: exact $1/N$)")
axs[1, 1].set_title("Gross–Neveu: $C_T/C_{T,\\rm ferm}$ (circles: $1/N$ at $d{=}3$)")
for ax in axs.flat:
    ax.axvline(3, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("$d$"); ax.set_xlim(2, 4); ax.legend(fontsize=7, frameon=False)
axs[0, 0].set_ylim(0.3, 1.02); axs[0, 1].set_ylim(0.45, 1.02)
axs[1, 0].set_ylim(0.55, 1.02)
fig.tight_layout()
fig.savefig(os.path.join(ROOT, "results", "pade_3d.pdf"), dpi=150)
print("wrote results/pade_3d.pdf")
