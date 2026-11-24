"""Write extraction artifacts and summaries in the existing format."""

import csv
import hashlib
import json
from collections import Counter

import numpy as np

from ..features.feature_utils import GROUPS, GROUP_SLICES, feature_names


def json_text(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2)


def atomic_write(path, content):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def save_npz(path, **arrays):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def export(out, rows, cfg, record):
    """Export rows in stable video-ID order, keeping values and availability together."""
    out.mkdir(parents=True, exist_ok=True)
    names = feature_names(cfg["temporal"]["lags"])
    ids = sorted(rows)
    infos = [rows[key][2] for key in ids]
    vectors = np.stack([rows[key][0] for key in ids])
    valid = np.stack([rows[key][1] for key in ids])
    save_npz(
        out / "features.npz",
        video_ids=np.array(ids),
        feature_names=np.array(names),
        vectors=vectors,
        valid=valid,
    )
    atomic_write(out / "config.json", json_text(record))
    atomic_write(
        out / "quality.jsonl",
        "".join(json.dumps(info, ensure_ascii=False, allow_nan=False) + "\n" for info in infos),
    )
    write_csv(
        out / "feature_schema.csv",
        ["index_0based", "group", "name"],
        ((index, name[0], name) for index, name in enumerate(names)),
    )
    fields = [
        "video_id",
        "status",
        "frames",
        "fps",
        "valid_pairs",
        "valid_ratio",
        "longest_valid_pairs",
        "valid_dimensions",
        "input_sha256",
    ]
    write_csv(
        out / "manifest.csv",
        ["row", *fields],
        ([index, *(info.get(field, "") for field in fields)] for index, info in enumerate(infos)),
    )
    write_csv(
        out / "exceptions.csv",
        ["video_id", "feature", "reason", "detail"],
        (
            (info["video_id"], name, reason, info.get("error", ""))
            for info in infos
            for name, reason in info["unavailable"].items()
        ),
    )
    summary = {
        "videos": len(rows),
        "shape": list(vectors.shape),
        "status": dict(Counter(info["status"] for info in infos)),
        "decoded_frames": sum(info.get("frames", 0) for info in infos),
        "valid_dimensions_total": int(valid.sum()),
        "invalid_dimensions_total": int(valid.size - valid.sum()),
        "invalid_by_group": {
            group: int((~valid[:, GROUP_SLICES[group]]).sum()) for group in GROUPS
        },
        "exception_reasons": dict(
            Counter(reason for info in infos for reason in info["unavailable"].values())
        ),
    }
    atomic_write(out / "summary.json", json_text(summary))
    return summary
