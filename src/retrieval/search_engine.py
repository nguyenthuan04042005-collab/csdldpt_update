"""Compare a catalog video with its candidates using the shared feature mask."""

from functools import lru_cache

import numpy as np

from ..config import DEFAULT_QUERY
from ..features.feature_utils import GROUP_SLICES
from .normalizer import fit_normalizer, normalize
from .ranker import similarity_score, top_five
from .similarity import measure


class SearchEngine:
    def __init__(self, items, vectors, valid):
        self.items = items
        self.vectors = vectors
        self.valid = valid
        self.index = {item["id"]: i for i, item in enumerate(items)}
        self.people = np.array([item["person"] for item in items])
        self.default = DEFAULT_QUERY if DEFAULT_QUERY in self.index else items[0]["id"]

    def candidates(self, key, other_people):
        q = self.index[key]
        keep = np.arange(len(self.items)) != q
        if other_people:
            keep &= self.people != self.people[q]
        return np.flatnonzero(keep)

    @lru_cache(maxsize=512)
    def rank(self, key, metric, with_d, other_people):
        q = self.index[key]
        pool = self.candidates(key, other_people)
        if not len(pool):
            return [], 0
        vectors = self.vectors[pool]
        masks = self.valid[pool]
        mean, std, usable = fit_normalizer(vectors, masks)
        query = normalize(self.vectors[q], mean, std, usable)
        vectors = normalize(vectors, mean, std, usable)
        distances = np.zeros(len(pool))
        weights = np.zeros(len(pool))
        details = {}
        for group, span in GROUP_SLICES.items():
            common = masks[:, span] & self.valid[q, span]
            if group == "D":
                common &= usable[span]
            counts = common.sum(axis=1)
            size = span.stop - span.start
            weight = counts / size if with_d or group != "D" else np.zeros(len(pool))
            cost = measure(metric, query[span], vectors[:, span], common)
            distances += cost * weight
            weights += weight
            details[group] = (counts, size, weight, cost)
        distance = np.divide(distances, weights, out=np.full(len(pool), np.inf), where=weights > 0)
        results = []
        for j in top_five(pool, distance):
            groups = [
                {"key": g, "shared": int(n_shared[j]), "size": size, "weight": float(weight[j]),
                 "distance": float(cost[j]) if n_shared[j] else None}
                for g, (n_shared, size, weight, cost) in details.items()
            ]
            results.append({"video": self.items[pool[j]], "distance": float(distance[j]),
                            "score": similarity_score(distance[j]), "groups": groups})
        return results, len(pool)
