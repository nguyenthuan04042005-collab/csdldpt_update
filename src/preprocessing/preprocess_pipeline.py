"""Prepare the video for frame-pair feature extraction."""

from .frame_processor import estimate_background
from .video_reader import read_frames


def prepare_video(path, cfg):
    frames, fps = read_frames(path, cfg["input"])
    sample_count = cfg["segmentation"]["refine"]["background_samples"]
    return frames, fps, estimate_background(frames, sample_count)
