"""Dense Farneback flow and body/background references."""

import cv2
import numpy as np


def farneback(gray, following, cfg):
    return cv2.calcOpticalFlowFarneback(gray, following, None, **cfg["farneback"])


def reference_motion(flow, mask, box, cfg):
    """3.2.1–3.2.3. Median by component on the masked central patch and background."""
    x, y, w, h = box
    x0, x1 = x + np.ceil(w * np.array([cfg["x_min"], cfg["x_max"]])).astype(int)
    y0, y1 = y + np.ceil(h * np.array([cfg["y_min"], cfg["y_max"]])).astype(int)
    body = flow[y0:y1, x0:x1][mask[y0:y1, x0:x1] != 0]
    if len(body) < cfg["min_body_pixels"]:
        return None, "insufficient_body_reference"
    halo = cfg["background_halo"]
    bg = np.ones(mask.shape, bool)
    bg[max(0, y - halo) : y + h + halo, max(0, x - halo) : x + w + halo] = False
    if np.count_nonzero(bg) < cfg["min_background_pixels"]:
        return None, "insufficient_background"
    g = np.median(body, axis=0)
    c = np.median(flow[bg], axis=0)
    height = max(h, cfg["height_floor"])
    center = (x + w * (cfg["x_min"] + cfg["x_max"]) / 2, y + h * (cfg["y_min"] + cfg["y_max"]) / 2)
    relative = (flow[y : y + h, x : x + w] - g) / height
    return ((g - c) / height, relative, center, height), ""
