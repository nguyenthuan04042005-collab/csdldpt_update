"""Grayscale conversion and temporal background estimation."""

import cv2
import numpy as np


def to_grayscale(color):
    return cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)


def estimate_background(frames, sample_count):
    samples = np.rint(np.linspace(0, len(frames) - 1, min(len(frames), sample_count))).astype(int)
    median = np.median(np.stack([frames[i] for i in samples]), axis=0).astype(np.uint8)
    return cv2.GaussianBlur(median, (3, 3), 0)
