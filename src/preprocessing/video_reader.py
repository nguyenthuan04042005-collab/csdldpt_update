"""Decode grayscale video and enforce the existing input format."""

import cv2
import numpy as np

from .frame_processor import to_grayscale


def read_frames(path, cfg):
    """Decode all original grayscale frames; reject inputs outside the configured format."""
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 1])
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError("cannot_open_video")
        fps = capture.get(cv2.CAP_PROP_FPS)
        declared = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not np.isfinite(fps) or abs(fps - cfg["fps"]) > cfg["fps_tolerance"]:
            raise ValueError(f"unsupported_fps: {fps}")
        while True:
            decoded, color = capture.read()
            if not decoded:
                break
            if color.shape[:2] != (cfg["height"], cfg["width"]):
                raise ValueError(f"unexpected_frame_size: {color.shape[:2]}")
            frames.append(to_grayscale(color))
    finally:
        capture.release()
    if declared > 0 and len(frames) != declared:
        raise ValueError(f"incomplete_decode: {len(frames)} of {declared}")
    if len(frames) < 2:
        raise ValueError(f"too_few_frames: {len(frames)}")
    return frames, fps
