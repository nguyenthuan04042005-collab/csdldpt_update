"""Fit z-score statistics on the valid values of the candidate set."""

import numpy as np


def fit_normalizer(vectors, masks):
    n = masks.sum(axis=0)
    mean = np.where(masks, vectors, 0).sum(axis=0) / np.maximum(n, 1)
    variance = np.where(masks, (vectors - mean) ** 2, 0).sum(axis=0) / np.maximum(n, 1)
    std = np.sqrt(variance)
    usable = (n > 0) & (std > 1e-12)
    return mean, std, usable


def normalize(vectors, mean, std, usable):
    return np.where(usable, (vectors - mean) / np.where(usable, std, 1), 0)
