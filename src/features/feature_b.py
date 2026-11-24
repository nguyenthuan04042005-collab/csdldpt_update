"""Group B: relative motion direction and intensity."""

import numpy as np

from .feature_utils import GROUPS


def collect_histogram(relative, person, cells, mag, data, t):
    inside = person != 0
    active = cells[inside]
    weights = mag[inside].astype(float)
    flow = relative[inside]
    bins = np.mod(np.arctan2(flow[:, 1].astype(float), flow[:, 0]), 2 * np.pi) * (4 / np.pi)
    low = np.floor(bins).astype(int)
    frac = bins - low
    for direction, share in ((low, 1 - frac), (low + 1, frac)):
        idx = active * 8 + direction % 8
        data.hist += np.bincount(idx, weights=weights * share, minlength=96).reshape(12, 8)
    counts = np.bincount(active, minlength=12)
    data.b_support[t] = counts > 0
    data.energy[t] = np.bincount(active, weights=weights, minlength=12) / np.maximum(counts, 1)


def motion_histogram(data, eps):
    """B: 96 globally normalized direction bins, 12 means, then 12 standard deviations."""
    if not data.good.any():
        return np.full(GROUPS["B"], np.nan)
    energy = data.energy[data.good]
    return np.r_[
        data.hist.ravel() / max(data.hist.sum(), eps), energy.mean(axis=0), energy.std(axis=0)
    ]
