"""Group A: translation and shape statistics."""

import numpy as np


def translation(data, eps):
    """3.3.1. q and d use N, not |G|. Differencing never bridges invalid frames."""
    g = data.good
    if not g.any():
        return np.full(16, np.nan)
    a = data.a[g]
    speed = np.linalg.norm(a, axis=1)
    adjacent = g[1:] & g[:-1]
    d = np.full(2, np.nan)
    delta_stats = [np.nan, np.nan]
    if adjacent.any():
        d = (np.diff(data.center, axis=0)[adjacent] / data.height[1:][adjacent, None]).sum(
            axis=0
        ) / data.n
        delta = np.linalg.norm(np.diff(data.a, axis=0)[adjacent], axis=1)
        delta_stats = [delta.mean(), delta.std()]
    return np.r_[
        a.mean(axis=0),
        a.std(axis=0),
        np.quantile(speed, [0.25, 0.5, 0.75], method="linear"),
        speed.mean(),
        speed.sum() / data.n,
        d,
        np.linalg.norm(a.sum(axis=0)) / max(speed.sum(), eps),
        delta_stats,
        data.aspect[g].mean(),
        data.aspect[g].std(),
    ]
