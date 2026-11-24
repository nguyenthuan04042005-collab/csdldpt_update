"""Person masks using the OpenCV Zoo PP-HumanSeg model, independently of the label."""

import math

import cv2
import numpy as np


def refine_silhouette(gray, temporal_background, mask, box, cfg):
    """Extend a semantic seed into connected, locally contrasting limbs on static backgrounds."""
    if box is None:
        return mask, None
    x, y, w, h = box
    rows, cols = gray.shape
    margin = max(8, int(np.ceil(cfg["side"] * h)))
    x0, x1 = max(0, x - margin), min(cols, x + w + margin)
    y0, y1 = (
        max(0, y - int(np.ceil(cfg["top"] * h))),
        min(rows, y + h + int(np.ceil(cfg["bottom"] * h))),
    )
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    img = smooth[y0:y1, x0:x1]
    seed = mask[y0:y1, x0:x1]
    lo = max(0, x0 - cfg["strip"])
    hi = min(cols, x1 + cfg["strip"])
    strips = []
    if x0 > lo:
        strips.append(np.median(smooth[y0:y1, lo:x0], axis=1))
    if hi > x1:
        strips.append(np.median(smooth[y0:y1, x1:hi], axis=1))
    if not strips:
        return mask, box
    alpha = np.linspace(0.0, 1.0, img.shape[1])
    spatial_bg = strips[0][:, None] * (1 - alpha) + strips[-1][:, None] * alpha
    deviation = np.abs(img - spatial_bg)
    temporal_delta = cv2.absdiff(img, temporal_background[y0:y1, x0:x1])
    # A bright wall can have a moving shadow connected to the person. Use the
    # seed's contrast to set a higher threshold when the body is clearly visible.
    seed_contrast = float(np.median(temporal_delta[seed != 0]))
    threshold = max(cfg["temporal_delta"], cfg["seed_contrast_fraction"] * seed_contrast)
    candidates = (deviation > cfg["spatial_delta"]) & (temporal_delta > threshold)
    # Stick to the neighborhood of a detected subject. Do not extend a floor shadow horizontally.
    yy, xx = np.ogrid[y0:y1, x0:x1]
    near_foot = yy >= y + cfg["foot_start"] * h
    candidates &= ~near_foot | (np.abs(xx - (x + w / 2)) <= max(w / 2, h * cfg["foot_halfwidth"]))
    distance = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 3)
    candidates &= distance < max(4, cfg["max_distance"] * h)
    possible = np.maximum(seed, cv2.medianBlur(candidates.astype(np.uint8), 3))
    # Connect a single-pixel break; only original candidate or semantic pixels may be added.
    closed = cv2.morphologyEx(
        possible,
        cv2.MORPH_CLOSE,
        np.ones((3, 3), np.uint8),
        borderType=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    _, labels = cv2.connectedComponents(np.maximum(possible, closed), connectivity=8)
    touching = np.unique(labels[seed != 0])
    enhanced = np.isin(labels, touching) & (possible != 0)
    result = mask.copy()
    result[y0:y1, x0:x1] = enhanced
    return result, cv2.boundingRect(result)


def clean_mask(probability, cfg):
    binary = (probability >= cfg["threshold"]).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(binary), None
    largest = 1 + int(stats[1:, cv2.CC_STAT_AREA].argmax())
    x, y, w, h, area = (int(i) for i in stats[largest])
    if area < cfg["min_pixels"]:
        return np.zeros_like(binary), None
    # Include nearby detached hands/feet, without using an action-specific rule.
    left, top, width, height, sizes = stats.T
    margin_x = max(3, h * cfg["component_margin_x"])
    margin_y = max(3, h * cfg["component_margin_y"])
    keep = (
        (sizes >= max(cfg["min_component_pixels"], area * cfg["min_component_fraction"]))
        & (left < x + w + margin_x)
        & (left + width > x - margin_x)
        & (top < y + h + margin_y)
        & (top + height > y - margin_y)
    )
    keep[0] = False
    mask = keep[labels].astype(np.uint8)
    box = cv2.boundingRect(mask)
    if box[2] < cfg["min_width"] or box[3] < cfg["min_height"]:
        return np.zeros_like(mask), None
    return mask, box


class PersonSegmenter:
    def __init__(self, model_path, cfg):
        self.net = cv2.dnn.readNetFromONNX(str(model_path))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.cfg = cfg

    def infer(self, gray):
        side = self.cfg["input_side"]
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        blob = cv2.dnn.blobFromImage(rgb, 1 / 127.5, (side, side), (127.5,) * 3)
        self.net.setInput(blob)
        prob = cv2.resize(
            self.net.forward()[0, 1], gray.shape[::-1], interpolation=cv2.INTER_LINEAR
        )
        return clean_mask(prob, self.cfg)

    def search_window(self, box, shape):
        x, y, w, h = box
        cfg = self.cfg
        halfwidth = max(
            cfg["crop_min_halfwidth"], w * cfg["crop_width_factor"], h * cfg["crop_height_factor"]
        )
        return (
            max(0, math.floor(x + w / 2 - halfwidth)),
            max(0, math.floor(y - max(cfg["crop_top_min"], cfg["crop_top_fraction"] * h))),
            min(shape[1], math.ceil(x + w / 2 + halfwidth)),
            min(
                shape[0],
                math.ceil(y + h + max(cfg["crop_bottom_min"], cfg["crop_bottom_fraction"] * h)),
            ),
        )

    def frame(self, gray, index, previous):
        # On periodic full-frame detections, refine once in the same coordinate system.
        coarse = None
        if previous is None or index % self.cfg["reacquire_interval"] == 0:
            coarse, previous = self.infer(gray)
            if previous is None:
                return coarse, None
        x0, y0, x1, y1 = self.search_window(previous, gray.shape)
        patch, rect = self.infer(gray[y0:y1, x0:x1])
        if rect is not None:
            x, y, w, h = rect
            cut = (
                (x <= 1 and x0 > 0)
                or (y <= 1 and y0 > 0)
                or (x + w >= x1 - x0 - 1 and x1 < gray.shape[1])
                or (y + h >= y1 - y0 - 1 and y1 < gray.shape[0])
            )
            if not cut:
                mask = np.zeros_like(gray)
                mask[y0:y1, x0:x1] = patch
                return mask, (x0 + x, y0 + y, w, h)
        if coarse is not None:
            return coarse, previous
        return self.infer(gray)
