"""Collect frame-pair measurements and assemble the A?F vector."""

import cv2
import numpy as np

from ..models import PairData
from ..preprocessing.foreground import refine_silhouette
from ..preprocessing.optical_flow import farneback, reference_motion
from ..preprocessing.preprocess_pipeline import prepare_video
from ..visualization.visualizer import debug_images
from ..evaluation.metrics import dimension_reasons, reason_intervals
from .feature_a import translation
from .feature_b import collect_histogram, motion_histogram
from .feature_c import collect_coordination, coordination
from .feature_d import temporal
from .feature_e import appearance, appearance_summary
from .feature_f import change_summary, changes
from .feature_utils import FEATURE_COUNT, GROUPS, cell_ids, feature_names, runs


def regional_motion(relative, person, data, t, eps):
    cells = cell_ids(*person.shape)
    magnitude = np.hypot(relative[..., 0], relative[..., 1])
    collect_histogram(relative, person, cells, magnitude, data, t)
    collect_coordination(relative, person, cells, magnitude, data, t, eps)


def summarize(data, cfg):
    """Concatenate A–F; encode missing measurements as zero plus a False flag."""
    eps = cfg["epsilon"]
    d, segment, peak_status = temporal(data, cfg["temporal"], eps)
    raw = np.concatenate(
        [
            translation(data, eps),
            motion_histogram(data, eps),
            coordination(data, eps),
            d,
            appearance_summary(data),
            change_summary(data),
        ]
    )
    if raw.shape != (FEATURE_COUNT,) or np.isinf(raw).any():
        raise ArithmeticError("invalid vector size or overflow")
    return np.nan_to_num(raw).astype(np.float32), np.isfinite(raw), segment, peak_status


def extract(path, relative_id, cfg, segmenter, debug_dir=None):
    """Extract one clip without using its action label or file name in the feature values."""
    frames, fps, background = prepare_video(path, cfg)
    refine = cfg["segmentation"]["refine"]
    data = PairData(len(frames) - 1)
    previous = None
    boxes, masks, reasons, areas, added = [], [], [], [], []
    empty_resized = []
    for t in range(data.n):
        gray, following = frames[t : t + 2]
        mask, box = segmenter.frame(gray, t, previous)
        previous = box
        seed_area = cv2.countNonZero(mask)
        mask, box = refine_silhouette(gray, background, mask, box, refine)
        boxes.append(box)
        if debug_dir:
            masks.append(mask)
        data.change[t] = changes(gray, following, box, cfg["difference"])
        if box is None:
            reasons.append("no_person_mask")
            continue
        areas.append(cv2.countNonZero(mask))
        added.append((areas[-1] - seed_area) / areas[-1])
        flow = farneback(gray, following, cfg)
        ref, reason = reference_motion(flow, mask, box, cfg["reference"])
        reasons.append(reason)
        if ref is None:
            continue
        x, y, w, h = box
        person = mask[y : y + h, x : x + w]
        data.a[t], relative, data.center[t], data.height[t] = ref
        data.good[t] = True
        data.aspect[t] = w / h
        regional_motion(relative, person, data, t, cfg["epsilon"])
        data.shape[t], has_resized_mask = appearance(gray[y : y + h, x : x + w], person)
        if not has_resized_mask:
            empty_resized.append(t)
    vector, valid, run, peak_status = summarize(data, cfg)
    ok_count = int(data.good.sum())
    finite_f = np.isfinite(data.change)
    indices_f = finite_f.any(axis=1)
    height, width = frames[0].shape
    at_border = sum(
        1
        for r in boxes
        if r and (r[0] == 0 or r[1] == 0 or r[0] + r[2] == width or r[1] + r[3] == height)
    )
    info = {
        "video_id": relative_id,
        "status": "complete" if valid.all() else ("partial" if valid.any() else "unavailable"),
        "frames": len(frames),
        "pairs": data.n,
        "fps": fps,
        "width": width,
        "height": height,
        "valid_pairs": ok_count,
        "valid_ratio": ok_count / data.n,
        "valid_group_pair_ratios": {
            g: (float(indices_f.mean()) if g == "F" else ok_count / data.n) for g in GROUPS
        },
        "longest_valid_run": list(run),
        "longest_valid_pairs": run[1] - run[0],
        "invalid_G_intervals": reason_intervals(reasons),
        "adjacent_valid_pairs": int((data.good[1:] & data.good[:-1]).sum()),
        "visible_pairs": len(areas),
        "border_pairs": at_border,
        "median_mask_pixels": float(np.median(areas)) if areas else None,
        "median_fraction_added_to_final_mask": float(np.median(added)) if added else None,
        "median_box_height": float(np.median([b[3] for b in boxes if b])) if areas else None,
        "peak_status": peak_status,
        "B_cell_support_counts": data.b_support.sum(axis=0).tolist(),
        "E_empty_resized_mask_frames": empty_resized,
        "F_cell_valid_counts": finite_f.sum(axis=0).tolist(),
        "F_invalid_intervals_by_cell": [runs(~v) for v in finite_f.T],
        "F_invalid_box_intervals": reason_intervals(
            "no_person_box"
            if box is None
            else ("box_below_min_height" if box[3] < cfg["difference"]["min_height"] else "")
            for box in boxes
        ),
        "valid_dimensions": int(valid.sum()),
    }
    names = feature_names(cfg["temporal"]["lags"])
    info["unavailable"] = dimension_reasons(valid, names, info, cfg)
    if debug_dir:
        info["preview_pair_indices"] = debug_images(debug_dir, frames, masks, boxes, data, cfg)
    return vector, valid, info
