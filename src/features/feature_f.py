"""Group F: pixel changes around the person."""

import cv2
import numpy as np


def changes(gray, following, box, cfg):
    """3.3.6. Round real boundaries only after partition; compare original pixels."""
    result = np.full(16, np.nan)
    if box is None or box[3] < cfg["min_height"]:
        return result
    x, y, w, h = box
    width = max(cfg["min_width_heights"] * h, w + cfg["extra_width_heights"] * h)
    cx = x + w / 2
    xe = np.clip(
        np.ceil(np.linspace(cx - width / 2, cx + width / 2, 5)).astype(int), 0, gray.shape[1]
    )
    ye = np.clip(
        np.ceil(np.linspace(y - cfg["top_pad"] * h, y + h + cfg["bottom_pad"] * h, 5)).astype(int),
        0,
        gray.shape[0],
    )
    changed = cv2.absdiff(gray, following) > cfg["threshold"]
    for row in range(4):
        for col in range(4):
            cell = changed[ye[row] : ye[row + 1], xe[col] : xe[col + 1]]
            if cell.size:
                result[4 * row + col] = cell.mean()
    return result


def change_summary(data):
    """F: each of 16 cells has its own set of available frame pairs."""
    result = np.full((2, 16), np.nan)
    for k, column in enumerate(data.change.T):
        observed = column[np.isfinite(column)]
        if observed.size:
            result[:, k] = [observed.mean(), observed.std()]
    return result.ravel()
