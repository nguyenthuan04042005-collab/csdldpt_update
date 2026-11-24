"""Order candidates by ascending distance, excluding unavailable scores."""

import numpy as np


def top_five(candidate_indices, distances):
    order = np.lexsort((candidate_indices, distances))
    return order[np.isfinite(distances[order])][:5]


def similarity_score(distance):
    return float(1 / (1 + distance))
