"""Group E: silhouette projection and grayscale summaries."""

import cv2
import numpy as np

from .feature_utils import GROUPS


def appearance(gray_person, person):
    """3.3.5. Eight widths, eight masked gray sums / whole cell area, sixteen frequencies."""
    gray = cv2.resize(gray_person, (32, 32), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(person, (32, 32), interpolation=cv2.INTER_NEAREST) != 0
    width = mask.reshape(8, 4, 32).any(axis=1).mean(axis=1)
    layout = ((gray / 255) * mask).reshape(4, 8, 2, 16).mean(axis=(1, 3)).ravel()
    hist = np.bincount(gray[mask] // 16, minlength=16) / max(1, int(mask.sum()))
    return np.r_[width, layout, hist], bool(mask.any())


def appearance_summary(data):
    """E: projection statistics, masked grayscale layout statistics and mean histogram."""
    if not data.good.any():
        return np.full(GROUPS["E"], np.nan)
    shape = data.shape[data.good]
    return np.r_[
        shape[:, :8].mean(axis=0),
        shape[:, :8].std(axis=0),
        shape[:, 8:16].mean(axis=0),
        shape[:, 8:16].std(axis=0),
        shape[:, 16:].mean(axis=0),
    ]
