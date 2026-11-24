"""Render diagnostic images, pilot sheets and dashboard posters."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..evaluation.metrics import require
from ..preprocessing.optical_flow import farneback


def save_png(path, array):
    encoded_ok, encoded = cv2.imencode(".png", array)
    if not encoded_ok:
        raise OSError(f"Cannot encode {path}")
    encoded.tofile(path)


def debug_images(directory, frames, masks, boxes, data, cfg):
    """Three valid pairs distributed over the visible action, with exact frame indexes."""
    directory.mkdir(parents=True, exist_ok=True)
    indices = np.flatnonzero(data.good)
    if not indices.size:
        indices = np.arange(data.n)
    selected = np.unique(
        indices[np.rint((len(indices) - 1) * np.array([0.2, 0.5, 0.8])).astype(int)]
    )
    rows = []
    for idx in selected:
        t = int(idx)
        gray, mask, box = frames[t], masks[t], boxes[t]
        flow = cv2.calcOpticalFlowFarneback(gray, frames[t + 1], None, **cfg["farneback"])
        mag = np.hypot(flow[..., 0], flow[..., 1])
        # Fixed 4 px/frame visualization scale makes brightness comparable across images.
        hue = np.mod(np.arctan2(flow[..., 1], flow[..., 0]), 2 * np.pi) * (180 / (2 * np.pi))
        hsv = np.dstack([hue, np.full(gray.shape, 255), np.minimum(255, mag * 255 / 4)]).astype(
            np.uint8
        )
        flow_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        input_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        overlay = input_bgr.copy()
        tint = overlay.copy()
        tint[mask != 0] = (40, 180, 35)
        overlay = cv2.addWeighted(tint, 0.48, overlay, 0.52, 0)
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(overlay, (x, y), (x + w - 1, y + h - 1), (0, 200, 255), 1)
        prefix = f"t{t:04d}"
        save_png(directory / f"{prefix}_mask.png", mask * 255)
        save_png(directory / f"{prefix}_flow.png", flow_bgr)
        row = np.hstack(
            [input_bgr, overlay, cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR), flow_bgr]
        )
        row = cv2.copyMakeBorder(row, 18, 0, 0, 0, cv2.BORDER_CONSTANT)
        for col, label in enumerate(
            (f"t={t}  input", "mask + box", "binary mask", "flow: color=direction")
        ):
            cv2.putText(
                row,
                label,
                (col * gray.shape[1] + 3, 13),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (235, 235, 235),
                1,
                cv2.LINE_AA,
            )
        rows.append(row)
    save_png(directory / "preview.png", np.vstack(rows))
    # Remove diagnostics from earlier trials if a different frame was selected.
    keep = {f"t{int(t):04d}_{kind}.png" for t in selected for kind in ("mask", "flow")}
    for kind in ("mask", "flow"):
        for old in directory.glob(f"t*_{kind}.png"):
            if old.name.partition("_")[0][1:].isdigit() and old.name not in keep:
                old.unlink()
    return selected.tolist()


def contact_sheets(folder, infos, output=None):
    """Keep the original 160x120 pixels in each tile, for visual inspection."""
    output = output or folder
    output.mkdir(parents=True, exist_ok=True)
    sheets = []
    people = sorted({Path(v["video_id"]).stem.split("_")[0] for v in infos})
    for person in people:
        samples = [v for v in infos if Path(v["video_id"]).stem.startswith(person + "_")]
        sets = {"masks": [], "flow": []}
        for info in samples:
            stem = Path(info["video_id"]).stem
            directory = folder / "images" / stem
            preview = cv2.imread(str(directory / "preview.png"))
            require(preview is not None, f"Missing preview for {stem}")
            for t in info["preview_pair_indices"]:
                mask = cv2.imread(str(directory / f"t{t:04d}_mask.png"), cv2.IMREAD_GRAYSCALE)
                flow = cv2.imread(str(directory / f"t{t:04d}_flow.png"))
                require(
                    mask is not None and mask.shape == (120, 160) and np.isin(mask, [0, 255]).all(),
                    f"Invalid mask {stem}/{t}",
                )
                require(
                    flow is not None and flow.shape == (120, 160, 3), f"Invalid flow {stem}/{t}"
                )
            for key, x in (("masks", 0), ("flow", 320)):
                tile = cv2.copyMakeBorder(preview[:, x : x + 320], 20, 0, 0, 0, cv2.BORDER_CONSTANT)
                cv2.putText(tile, stem, (3, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
                sets[key].append(tile)
        for key, tiles in sets.items():
            image = np.vstack([np.hstack(tiles[i : i + 3]) for i in range(0, len(tiles), 3)])
            name = person + "_" + key + ".png"
            save_png(output / name, image)
            sheets.append(name)
    return sheets


@lru_cache(maxsize=128)
def poster(path):
    capture = cv2.VideoCapture(str(path))
    try:
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(count * .49)))
        ok, frame = capture.read()
        if not ok:
            raise OSError(f"Cannot read {path.name}")
        ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise OSError(f"Cannot encode {path.name}")
        return data.tobytes()
    finally:
        capture.release()
