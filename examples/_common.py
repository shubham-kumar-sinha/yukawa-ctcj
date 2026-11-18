"""Shared helpers for the examples."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

s0 = np.eye(2)
s1 = np.array([[0, 1], [1, 0]], dtype=complex)
s2 = np.array([[0, -1j], [1j, 0]])
s3 = np.array([[1, 0], [0, -1]], dtype=complex)
# five mutually anticommuting hermitian 4x4 Clifford matrices
GAMMA = [np.kron(s1, s1), np.kron(s1, s2), np.kron(s1, s3), np.kron(s2, s0), np.kron(s3, s0)]


def rotation(n, i=0, j=1, q=1.0):
    t = np.zeros((n, n)); t[i, j] = -q; t[j, i] = q
    return t


def stable(fps):
    return [f for f in fps if f["stable"]]
