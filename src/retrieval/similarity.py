"""Group distance functions supported by the dashboard."""

import numpy as np

METRICS = ("manhattan", "sqeuclidean", "huber", "cosine")


def measure(metric, query, candidates, common):
    delta = np.abs(candidates - query)
    size = np.maximum(common.sum(axis=1), 1)
    if metric == "cosine":
        left = np.where(common, query, 0)
        right = np.where(common, candidates, 0)
        norm_l = np.linalg.norm(left, axis=1)
        norm_r = np.linalg.norm(right, axis=1)
        norms = norm_l * norm_r
        similarity = np.divide((left * right).sum(axis=1), norms,
                               out=np.zeros(len(right)), where=norms > 1e-12)
        distance = 1 - np.clip(similarity, -1, 1)
        return np.where((norm_l <= 1e-12) & (norm_r <= 1e-12), 0, distance)
    if metric == "sqeuclidean":
        delta = delta ** 2
    elif metric == "huber":
        delta = np.where(delta <= 1, .5 * delta ** 2, delta - .5)
    return np.where(common, delta, 0).sum(axis=1) / size
