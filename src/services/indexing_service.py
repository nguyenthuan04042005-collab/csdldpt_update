"""Run extraction in batches, reusing checkpoints and recording input hashes."""

import hashlib
import json
import platform
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from ..config import PILOT, ROOT
from ..features.extractor import extract
from ..features.feature_utils import FEATURE_COUNT, feature_names
from ..preprocessing.foreground import PersonSegmenter
from ..visualization.result_exporter import export, json_text, save_npz, sha

SOURCE_FILES = tuple(
    path.relative_to(ROOT) for name in ("features", "preprocessing")
    for path in sorted((ROOT / "src" / name).glob("*.py"))
) + (
    "src/models.py",
    "src/services/indexing_service.py",
    "src/evaluation/metrics.py",
    "src/visualization/result_exporter.py",
    "src/visualization/visualizer.py",
)


def initialize(cfg, model):
    global SETTINGS, SEGMENTER
    SETTINGS = cfg
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    SEGMENTER = PersonSegmenter(model, cfg["segmentation"])


def process_one(task):
    path, relative_id, debug_dir = task
    try:
        vector, valid, info = extract(path, relative_id, SETTINGS, SEGMENTER, debug_dir)
    except (ValueError, cv2.error, OSError, ArithmeticError) as error:
        # Preserve a file-level failure; never mark the placeholder as measured.
        vector = np.zeros(FEATURE_COUNT, np.float32)
        valid = np.zeros(FEATURE_COUNT, bool)
        info = {
            "video_id": relative_id,
            "status": "error",
            "error": str(error),
            "valid_dimensions": 0,
            "unavailable": {"*": "file_error"},
        }
    return relative_id, vector, valid, info


def extract_dataset(dataset, config, out, mode, workers):
    cfg = json.loads(config.read_text(encoding="utf-8"))
    feature_names(cfg["temporal"]["lags"])
    model = (config.resolve().parent / cfg["segmentation"]["model"]).resolve()
    if not model.is_file():
        raise ValueError(f"Model not found: {model}")
    if sha(model) != cfg["segmentation"]["sha256"]:
        raise ValueError("Unexpected model checksum")
    dataset = dataset.resolve()
    if not dataset.is_dir():
        raise ValueError(f"Dataset not found: {dataset}")
    out = out.resolve()
    files = (
        list(PILOT)
        if mode == "pilot"
        else sorted(
            p.relative_to(dataset).as_posix()
            for p in dataset.rglob("*")
            if p.is_file() and p.suffix.lower() == ".mp4"
        )
    )
    if not files or workers < 1:
        raise ValueError("Need video files and at least one worker")
    missing = [key for key in files if not (dataset / key).is_file()]
    if missing:
        raise ValueError("Missing input videos: " + ", ".join(missing))
    record = {
        "extraction": cfg,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": {p: version(p) for p in ("numpy", "scipy", "opencv-python-headless")},
        "code_sha256": {Path(p).as_posix(): sha(ROOT / p) for p in SOURCE_FILES},
    }
    stamp = hashlib.sha256(json_text(record).encode()).hexdigest()[:16]
    cache = out / "checkpoint" / stamp
    cache.mkdir(parents=True, exist_ok=True)
    rows, todo, paths, hashes = {}, [], {}, {}
    for key in files:
        paths[key] = cache / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".npz")
        hashes[key] = sha(dataset / key)
        debug = out / "images" / Path(key).stem if mode == "pilot" else None
        if paths[key].exists() and (debug is None or (debug / "preview.png").exists()):
            with np.load(paths[key], allow_pickle=False) as saved:
                info = json.loads(str(saved["quality"]))
                if info["input_sha256"] == hashes[key] and info["status"] != "error":
                    rows[key] = saved["vector"], saved["valid"], info
                    continue
        todo.append((dataset / key, key, debug))
    print(
        f"{mode}: {len(files)} files, {len(todo)} to process, {workers} workers",
        flush=True,
    )
    if todo:
        with ProcessPoolExecutor(
            workers, initializer=initialize, initargs=(cfg, model)
        ) as pool:
            futures = [pool.submit(process_one, task) for task in todo]
            for future in as_completed(futures):
                key, vector, valid, info = future.result()
                info["input_sha256"] = hashes[key]
                rows[key] = vector, valid, info
                save_npz(paths[key], vector=vector, valid=valid, quality=np.array(json_text(info)))
                if mode == "pilot" or len(rows) % 25 == 0 or info["status"] == "error":
                    print(
                        f"{len(rows)}/{len(files)} {key}  {info['status']}  {int(valid.sum())}/{FEATURE_COUNT}",
                        flush=True,
                    )
    return export(out, rows, cfg, record)
