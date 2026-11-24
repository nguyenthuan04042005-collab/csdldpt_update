"""Validate exported features, input hashes and agreement with pilot results."""

import csv
import json
from collections import Counter

import numpy as np

from ..config import PILOT
from ..features.feature_utils import FEATURE_COUNT, GROUPS, GROUP_SLICES, feature_names
from ..visualization.result_exporter import atomic_write, json_text, sha
from ..visualization.visualizer import contact_sheets
from .metrics import require


def verify(folder, dataset, mode, pilot, *, report_path=None):
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))["extraction"]
    with np.load(folder / "features.npz", allow_pickle=False) as data:
        ids, names, v, ok = (data[k] for k in ("video_ids", "feature_names", "vectors", "valid"))
    expected = sorted(
        PILOT
        if mode == "pilot"
        else [
            p.relative_to(dataset).as_posix()
            for p in dataset.rglob("*")
            if p.is_file() and p.suffix.lower() == ".mp4"
        ]
    )
    require(
        ids.tolist() == expected and len(ids) == len(set(ids)), "Catalog differs from input files"
    )
    require(
        names.tolist() == feature_names(cfg["temporal"]["lags"]), "Feature names or order differ"
    )
    require(v.shape == ok.shape == (len(ids), FEATURE_COUNT), "Invalid matrix dimensions")
    require(v.dtype == np.float32 and ok.dtype == bool, "Wrong array type")
    require(
        np.isfinite(v).all() and np.all(v[~ok] == 0), "Nonfinite values or invalid placeholders"
    )
    with (folder / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
    require(
        [r["video_id"] for r in manifest] == expected
        and [int(r["row"]) for r in manifest] == list(range(len(ids))),
        "Row map mismatch",
    )
    with (folder / "feature_schema.csv").open(encoding="utf-8", newline="") as stream:
        schema = [(int(r["index_0based"]), r["group"], r["name"]) for r in csv.DictReader(stream)]
    require(
        schema == [(i, name[0], name) for i, name in enumerate(names)],
        "Feature catalog differs from matrix",
    )
    with (folder / "exceptions.csv").open(encoding="utf-8", newline="") as stream:
        exception_rows = [
            (r["video_id"], r["feature"], r["reason"]) for r in csv.DictReader(stream)
        ]
    exceptions = set(exception_rows)
    require(len(exceptions) == len(exception_rows), "Duplicate exception row")
    infos = [
        json.loads(line)
        for line in (folder / "quality.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    require([r["video_id"] for r in infos] == expected, "Quality row map mismatch")
    expected_exceptions = set()
    for i, info in enumerate(infos):
        video = ids[i]
        expected_exceptions.update((video, k, r) for k, r in info["unavailable"].items())
        require(
            sha(dataset / video) == manifest[i]["input_sha256"] == info["input_sha256"],
            f"Input changed: {video}",
        )
        require(info["valid_dimensions"] == int(ok[i].sum()), f"Quality count for {video}")
        if info["status"] == "error":
            require(not ok[i].any(), f"Error video has observations: {video}")
            continue
        require(set(info["unavailable"]) == set(names[~ok[i]]), f"Missing dimension map: {video}")
        require(info["frames"] == info["pairs"] + 1, f"Wrong number of pairs: {video}")
        g = info["valid_pairs"]
        n = info["pairs"]
        spans = info["invalid_G_intervals"]
        usable = np.ones(n, dtype=bool)
        last = 0
        for start, stop, reason in spans:
            require(last <= start < stop <= n and reason, f"Invalid observation interval: {video}")
            usable[start:stop] = False
            last = stop
        require(
            usable.sum() == g and np.isclose(g / n, info["valid_ratio"]), f"G coverage: {video}"
        )
        islands = np.flatnonzero(np.diff(np.r_[False, usable, False].astype(np.int8))).reshape(
            -1, 2
        )
        longest = (
            islands[np.argmax(islands[:, 1] - islands[:, 0])].tolist() if len(islands) else [0, 0]
        )
        require(
            longest == info["longest_valid_run"]
            and longest[1] - longest[0] == info["longest_valid_pairs"],
            f"Longest run: {video}",
        )
        f_counts = np.array(
            [
                n - sum(stop - start for start, stop in spans)
                for spans in info["F_invalid_intervals_by_cell"]
            ]
        )
        require(
            np.array_equal(f_counts, info["F_cell_valid_counts"])
            and np.array_equal(
                ok[i, GROUP_SLICES["F"]].reshape(2, 16), np.tile(f_counts > 0, (2, 1))
            ),
            f"F cell eligibility: {video}",
        )
        if g:
            hist_sum = v[i, 16:112].sum()
            require(
                np.isclose(hist_sum, 1, atol=2e-6) or np.isclose(hist_sum, 0),
                f"B normalization: {video}",
            )
            require(
                np.isclose(v[i, 8], v[i, 7] * g / info["pairs"], rtol=3e-6, atol=1e-9),
                f"A denominator: {video}",
            )
            expected_e = 1 - len(info["E_empty_resized_mask_frames"]) / g
            require(
                np.isclose(v[i, 276:292].sum(), expected_e, atol=2e-6), f"E normalization: {video}"
            )
        lag_flags = ok[i, 160:208].reshape(6, 8)
        for j, lag in enumerate(cfg["temporal"]["lags"]):
            if info["longest_valid_pairs"] < max(cfg["temporal"]["min_run"], 2 * lag):
                require(not lag_flags[:, j].any(), f"D unsupported lag: {video}")
        cn = v[i, GROUP_SLICES["C"]].reshape(6, 4)
        require(
            (np.abs(cn[:, 1]) <= 1.000001).all()
            and ((cn[:, [0, 2, 3]] >= 0) & (cn[:, [0, 2, 3]] <= 1.000001)).all(),
            f"C bounds: {video}",
        )
        require(
            ((v[i, GROUP_SLICES["F"]] >= 0) & (v[i, GROUP_SLICES["F"]] <= 1)).all(),
            f"F bounds: {video}",
        )
    require(exceptions == expected_exceptions, "Exception table differs from per-video indicators")
    report = {
        "mode": mode,
        "verified_rows": len(ids),
        "shape": list(v.shape),
        "finite": True,
        "dimensions_by_group": GROUPS,
        "row_map_and_exceptions_verified": True,
        "checks": [
            "input SHA-256",
            "column schema",
            "G observation intervals",
            "F per-cell availability",
            "B histogram",
            "A full-sequence denominator",
            "E histogram",
            "D lag length",
            "C and F bounds",
        ],
        "status_by_action": {},
    }
    for action in sorted({p.split("/")[0] for p in expected}):
        selected = [i for i in infos if i["video_id"].split("/")[0] == action]
        report["status_by_action"][action] = dict(Counter(i["status"] for i in selected))
    report_path = report_path or folder / "verification.json"
    if mode == "pilot":
        report["contact_sheets"] = contact_sheets(folder, infos, report_path.parent)
    if mode == "full" and pilot:
        p_cfg = json.loads((pilot / "config.json").read_text(encoding="utf-8"))
        run_cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        require(p_cfg == run_cfg, "Pilot and full configurations differ")
        with np.load(pilot / "features.npz", allow_pickle=False) as p:
            indices = [expected.index(key) for key in p["video_ids"]]
            require(np.array_equal(ok[indices], p["valid"]), "Pilot availability differs")
            np.testing.assert_allclose(v[indices], p["vectors"], rtol=1e-6, atol=1e-8)
            report["pilot_rows_reproduced"] = len(indices)
            report["max_abs_pilot_difference"] = float(np.max(np.abs(v[indices] - p["vectors"])))
    report_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(report_path, json_text(report))
    return report
