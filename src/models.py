"""Intermediate measurements indexed by original frame pair."""

import numpy as np


class PairData:
    """Indices always refer to the original sequence of N = T - 1 frame pairs."""

    def __init__(self, n):
        self.n = n
        self.good = np.zeros(n, bool)
        self.a = np.zeros((n, 2))
        self.center = np.zeros((n, 2))
        self.height = np.zeros(n)
        self.aspect = np.zeros(n)
        self.hist = np.zeros((12, 8))
        self.energy = np.zeros((n, 12))
        self.b_support = np.zeros((n, 12), bool)
        self.sides = np.zeros((n, 6, 2))
        self.coord = np.zeros((n, 6, 3))
        self.shape = np.zeros((n, 32))
        self.change = np.full((n, 16), np.nan)
