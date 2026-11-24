"""Group C: coordination between both sides of the person."""

import cv2
import numpy as np

from .feature_utils import pearson


def collect_coordination(relative, person, cells, mag, data, t, eps):
    support = (
        cv2.dilate(person, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        != 0
    )
    area = np.maximum(np.bincount(cells.ravel(), minlength=12), 1)

    def density(values):
        return (np.bincount(cells[support], weights=values[support], minlength=12) / area).reshape(
            6, 2
        )

    sides = density(mag)
    pos = density(np.maximum(relative[..., 0], 0))
    neg = density(np.maximum(-relative[..., 0], 0))
    total = np.maximum((pos + neg).sum(axis=1), eps)
    data.sides[t] = sides
    data.coord[t, :, 0] = sides.min(axis=1) / np.maximum(sides.max(axis=1), eps)
    data.coord[t, :, 1] = 2 * np.minimum(pos[:, 0], neg[:, 1]) / total
    data.coord[t, :, 2] = 2 * np.minimum(neg[:, 0], pos[:, 1]) / total


def coordination(data, eps):
    """C: balance, Pearson, inward and outward activity for each of six rows."""
    result = np.full((6, 4), np.nan)
    g = data.good
    if g.any():
        result[:, [0, 2, 3]] = data.coord[g].mean(axis=0)
        sides = data.sides[g]
        for row in range(6):
            result[row, 1] = pearson(sides[:, row, 0], sides[:, row, 1], eps)
    return result.ravel()
