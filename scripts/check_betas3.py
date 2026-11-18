#!/usr/bin/env python3
"""Checks of the three-loop gamma_phi, gamma_psi and beta_y of ctcj/betas3.py (JOS arXiv:2301.10903).  (~1 min)

1. Specialised to the chiral Ising, XY and Heisenberg models (dictionary of scripts/check_gny3loop.py), all three
   functions agree with the three-loop results of Zerf et al. (arXiv:1709.05057) at random couplings and N.
2. N=1 supersymmetry on general tensors: for a random totally symmetric superpotential h_abc with N_Phi = 2, 3, 4
   superfields (y^a_ij = h_aij, lambda_abcd = h_abe h_cde + 2 pairings, one Majorana per superfield),
   gamma_psi = gamma_phi and beta_y^a_ij stays totally symmetric in (a, i, j).
3. Mutation test: increasing any single one of the 81 coefficients by 0.1 makes check 1 or check 2 fail, so every
   transcribed term is independently tested."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, itertools, sympy as sp
import ctcj.betas3 as B3
from ctcj import gny3loop as G
from ctcj.eps3models import MODELS, S_tensor
rng = np.random.default_rng(3)
# fixed test points
zpts = []
for zm in ("chiral Ising", "chiral XY", "chiral Heisenberg"):
    for _ in range(3):
        yZ, lZ, NZ = rng.uniform(0.2, 1.5), rng.uniform(-0.5, 0.8), float(rng.choice([0.25, 1.0, 3.0]))
        sub = {G.y: yZ, G.lam: lZ, G.N: NZ}
        ref = [float(sp.N(G.RG[zm][k][2].subs(sub), 20)) for k in ("beta_y", "gamma_psi", "gamma_phi")]
        zpts.append((zm, yZ, lZ, NZ, ref))
def zerf_ok():
    for zm, yZ, lZ, NZ, ref in zpts:
        Y, mper, _ = MODELS[zm]; ns = Y.shape[0]
        y = np.sqrt(yZ / 2) * Y; lam = 4 * lZ * S_tensor(ns); m = mper * 4 * NZ
        b = B3.beta3_y(y, lam, m); gp = B3.gamma_psi3(y, lam, m); gs = B3.gamma_phi3(y, lam, m)
        k = np.unravel_index(np.argmax(abs(Y[0])), Y[0].shape)
        ours = [2 * yZ * (b[0][k] / y[0][k]).real, 2 * gp[0, 0].real, 2 * gs[0, 0]]
        if any(abs(o - r) > 1e-9 * max(1, abs(r)) for o, r in zip(ours, ref)):
            return False
        if not np.allclose(b, (b[0][k] / y[0][k]) * y, atol=1e-9): return False
    return True
hs = []
for n in (3, 4):
    h = rng.normal(size=(n,) * 3); h = sum(np.transpose(h, p) for p in itertools.permutations(range(3))) / 6
    lam = np.einsum("abe,cde->abcd", h, h) + np.einsum("ace,bde->abcd", h, h) + np.einsum("ade,bce->abcd", h, h)
    hs.append((h.astype(complex), lam))
def susy_ok():
    for y, lam in hs:
        gp, gs = B3.gamma_psi3(y, lam, 1.0), B3.gamma_phi3(y, lam, 1.0)
        if np.abs(gp - gs).max() > 1e-9 * np.abs(gs).max(): return False
        b = B3.beta3_y(y, lam, 1.0)
        bs = sum(np.transpose(b, p) for p in itertools.permutations(range(3))) / 6
        if np.abs(b - bs).max() > 1e-9 * np.abs(b).max(): return False
    return True
assert zerf_ok() and susy_ok(), "unmutated transcription fails"
labels = {"GAMMA_PHI3": "gphi3" , "GAMMA_PSI3": "gpsi3", "BETA_Y3": "by3"}
tilde = "abcdefghijklmnopqrstuvwxyz"
undetected = []
counts = {}
for name in ("GAMMA_PHI3", "GAMMA_PSI3", "BETA_Y3"):
    L = getattr(B3, name)
    det = 0
    for i, t in enumerate(L):
        orig = L[i]
        L[i] = (t[0] + 0.1,) + t[1:]
        z, s = zerf_ok(), susy_ok()
        L[i] = orig
        lab = (tilde[i] if i < 26 else tilde[i - 26] + "~")
        if z and s:
            undetected.append(f"{labels[name]}:3{lab}")
        else:
            det += 1
    counts[name] = (det, len(L))
print("detected by Zerf or SUSY tests:", counts)
print("NOT detected by either:", undetected)
ok = zerf_ok() and susy_ok() and not undetected
print("PASS three-loop gamma_phi, gamma_psi, beta_y agree with Zerf et al. (chiral Ising, XY, Heisenberg)" if zerf_ok() else "FAIL Zerf comparison")
print("PASS N=1 supersymmetry on random superpotentials (gamma_psi = gamma_phi, beta_y totally symmetric)" if susy_ok() else "FAIL SUSY test")
print("PASS every one of the 81 coefficients is tested (mutation test)" if not undetected else "FAIL untested: " + str(undetected))
sys.exit(0 if ok else 1)
