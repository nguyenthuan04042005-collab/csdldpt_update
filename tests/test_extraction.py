"""Pipeline and persistence checks at the boundaries between extraction modules."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.config import CONFIG_PATH
from src.features.extractor import extract
from src.visualization.result_exporter import export
from src.features.feature_utils import FEATURE_COUNT, GROUP_SLICES, feature_names

CFG = json.loads((CONFIG_PATH).read_text(encoding="utf-8"))


class FixedSegmenter:
    def __init__(self, mask, box):
        self.mask, self.box = mask, box

    def frame(self, gray, index, previous):
        return self.mask.copy(), self.box


class ExtractionPipeline(unittest.TestCase):
    def setUp(self):
        self.frames = [np.full((120, 160), value, np.uint8) for value in (80, 93, 105)]
        self.mask = np.zeros_like(self.frames[0])

    def extract_with(self, segmenter):
        with (
            patch("src.preprocessing.preprocess_pipeline.read_frames", return_value=(self.frames, 25.0)),
            patch(
                "src.features.extractor.refine_silhouette",
                side_effect=lambda gray, bg, mask, box, cfg: (mask, box),
            ),
        ):
            return extract(Path("sample.mp4"), "sample.mp4", CFG, segmenter)

    def test_no_person_retains_original_intervals_and_missing_flags(self):
        vector, valid, info = self.extract_with(FixedSegmenter(self.mask, None))
        self.assertEqual(vector.shape, (FEATURE_COUNT,))
        self.assertFalse(valid.any())
        np.testing.assert_array_equal(vector, 0)
        self.assertEqual(info["status"], "unavailable")
        self.assertEqual(info["invalid_G_intervals"], [[0, 2, "no_person_mask"]])
        self.assertEqual(info["F_cell_valid_counts"], [0] * 16)
        self.assertEqual(len(info["unavailable"]), FEATURE_COUNT)

    def test_F_remains_observed_when_body_reference_is_missing(self):
        # The box is visible, but no foreground pixel lies in its central reference patch.
        self.mask[20:40, 20] = 1
        vector, valid, info = self.extract_with(FixedSegmenter(self.mask, (20, 20, 10, 20)))
        self.assertFalse(valid[: GROUP_SLICES["F"].start].any())
        self.assertTrue(valid[GROUP_SLICES["F"]].all())
        np.testing.assert_allclose(vector[GROUP_SLICES["F"]], 0.5)
        self.assertEqual(info["invalid_G_intervals"], [[0, 2, "insufficient_body_reference"]])
        self.assertEqual(info["valid_group_pair_ratios"]["F"], 1.0)
        self.assertEqual(info["valid_dimensions"], 32)

        info["input_sha256"] = "synthetic-test-input"
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / "output"
            summary = export(
                folder, {"sample.mp4": (vector, valid, info)}, CFG, {"extraction": CFG}
            )
            with np.load(folder / "features.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["vectors"][0], vector)
                np.testing.assert_array_equal(saved["valid"][0], valid)
                self.assertEqual(
                    saved["feature_names"].tolist(), feature_names(CFG["temporal"]["lags"])
                )
            with (folder / "exceptions.csv").open(encoding="utf-8", newline="") as stream:
                exceptions = list(csv.DictReader(stream))
            self.assertEqual(len(exceptions), FEATURE_COUNT - 32)
            self.assertEqual({item["reason"] for item in exceptions}, {"no_valid_G_pairs"})
            self.assertEqual(summary["invalid_by_group"]["F"], 0)
            self.assertEqual(summary["valid_dimensions_total"], 32)


if __name__ == "__main__":
    unittest.main()
