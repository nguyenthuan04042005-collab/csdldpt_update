"""Analytic cases for formula mistakes that can still produce exactly 324 finite numbers."""
import json
import unittest

import numpy as np

from src.config import CONFIG_PATH
from src.features.extractor import regional_motion, summarize
from src.features.feature_a import translation
from src.features.feature_d import temporal
from src.features.feature_e import appearance
from src.features.feature_f import changes
from src.features.feature_utils import feature_names
from src.models import PairData
from src.preprocessing.foreground import refine_silhouette
from src.preprocessing.optical_flow import reference_motion

CFG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))



class ChapterThree(unittest.TestCase):
    def test_mask_refinement_recovers_limb_and_rejects_attached_shadow(self):
        background = np.full((70,80), 210, np.uint8)
        gray = background.copy()
        mask = np.zeros_like(gray)
        mask[12:54,36:46] = 1
        gray[mask != 0] = 50
        gray[18:25,28:36] = 60
        gray[30:43,27:36] = 180
        cfg = CFG["segmentation"]["refine"]
        refined, _ = refine_silhouette(gray, background, mask, (36,12,10,42), cfg)
        self.assertTrue(refined[21,31])
        self.assertFalse(refined[36,31])
        np.testing.assert_array_equal(refined[mask != 0], 1)

    def test_background_and_torso_reference_are_separate(self):
        mask = np.zeros((60, 60), np.uint8)
        mask[10:30, 20:30] = 1
        flow = np.zeros((*mask.shape, 2), np.float32)
        flow[:] = (1, 1)
        flow[mask != 0] = (5, 3)
        flow[10:13, 20:30] = (7, 1)
        values, reason = reference_motion(flow, mask, (20, 10, 10, 20), CFG["reference"])
        self.assertEqual(reason, "")
        a, relative, center, height = values
        np.testing.assert_allclose(a, [.2, .1])
        np.testing.assert_allclose(relative[0, 0], [.1, -.1])
        np.testing.assert_allclose(relative[10, 5], 0)
        self.assertEqual((center, height), ((25., 19.), 20))

    def test_A_denominators_adjacency_and_direction(self):
        data = PairData(4)
        data.good[:] = [True, True, False, True]
        data.height[:] = 20
        data.a[:,0] = [1, 3, 0, -2]
        data.center[:,0] = [1, 5, 50, 100]
        data.aspect[:] = .5
        a = translation(data, 1e-12)
        expected = [2/3, 0, np.std([1,3,-2]), 0, 1.5, 2, 2.5, 2, 1.5,
                    .05, 0, 2/6, 2, 0, .5, 0]
        np.testing.assert_allclose(a, expected)

    def test_B_circular_votes_and_C_simultaneous_directions(self):
        data = PairData(2)
        person = np.ones((12, 8), np.uint8)
        motion = np.ones((12,8,2))
        motion[..., 1] = 0
        motion[:,4:,0] = -1
        regional_motion(motion, person, data, 0, 1e-12)
        np.testing.assert_allclose(data.coord[0], np.tile([1,1,0], (6,1)))
        np.testing.assert_allclose(data.energy[0], 1)
        h = data.hist.reshape(6, 2, 8)
        np.testing.assert_equal(h[:,0,0], 8)
        np.testing.assert_equal(h[:,1,4], 8)
        self.assertEqual(data.hist.sum(), 96)
        # Halfway between 315 and 0 degrees, not a hard assignment to either bin.
        unit = np.zeros_like(motion)
        unit[:] = (np.cos(np.pi/8), -np.sin(np.pi/8))
        before = data.hist.copy()
        regional_motion(unit, person, data, 1, 1e-12)
        votes = data.hist-before
        np.testing.assert_allclose(votes[:, 0], 4)
        np.testing.assert_allclose(votes[:, 7], 4)
        np.testing.assert_allclose(votes[:, 1:7], 0)

    def test_C_uses_whole_half_cell_area(self):
        data = PairData(1)
        person = np.zeros((12, 20), np.uint8)
        person[:, 0] = 1
        motion = np.zeros((12,20,2))
        motion[...,0] = 1
        regional_motion(motion, person, data, 0, 1e-12)
        np.testing.assert_allclose(data.sides[0,:,0], .4)
        np.testing.assert_allclose(data.sides[0,:,1], 0)
        np.testing.assert_allclose(data.energy[0,::2], 1)

    def test_D_earliest_tied_run_and_original_quarters(self):
        data = PairData(201)
        data.good[:] = True
        data.good[100] = False
        t = np.arange(data.n)
        # A weak, noncommensurate second rhythm separates the peaks at 13, 26 and 39.
        z = np.where(t<100, np.sin(2*np.pi*t/13) + .14*np.sin(.71*t), np.cos(2*np.pi*t/7)) + 4 + .02*t
        data.energy[:] = z[:, None]
        d, run, _ = temporal(data, CFG["temporal"], 1e-12)
        self.assertEqual(run, (0,100))
        s = z[:100] - np.polyval(np.polyfit(t[:100], z[:100], 1), t[:100])
        curve = {l: np.corrcoef(s[:-l], s[l:])[0,1] for l in range(3,49)}
        expected = [curve[l] for l in CFG["temporal"]["lags"]]
        np.testing.assert_allclose(d[:48].reshape(6,8), np.tile(expected,(6,1)))
        peaks = [l for l in range(4,48) if 0 < curve[l] and curve[l-1]<curve[l]>curve[l+1]]
        k = max(peaks, key=curve.get)
        np.testing.assert_allclose(d[48:60].reshape(6,2), np.tile([k/48,curve[k]], (6,1)))
        means = [z[i[data.good[i]]].mean() for i in np.array_split(t,4)]
        np.testing.assert_allclose(d[60:].reshape(6,4), np.tile(means,(6,1)))

    def test_E_projection_masked_layout_and_gray_bins(self):
        mask = np.zeros((32,32), np.uint8)
        mask[:, :8] = 1
        gray = np.full_like(mask, 200)
        gray[:, :8] = 80
        e, nonempty = appearance(gray, mask)
        self.assertTrue(nonempty)
        np.testing.assert_allclose(e[:8], .25)
        np.testing.assert_allclose(e[8:16:2], 40/255)
        np.testing.assert_allclose(e[9:16:2], 0)
        np.testing.assert_array_equal(e[16:], np.eye(16)[5])

    def test_F_clipped_geometry_unsigned_difference_and_strict_threshold(self):
        gray = np.full((30, 29), 120, np.uint8)
        yy, xx = np.indices(gray.shape)
        following = np.where((xx+2*yy)%4==0, 107, 132).astype(np.uint8)
        measured = changes(gray, following, (0,0,8,20), CFG["difference"])
        # Unclipped boundaries: x=[-11,19), y=[-3,22). First F column is outside.
        col = np.floor((xx+11)/7.5)
        row = np.floor((yy+3)/6.25)
        expected = []
        for r in range(4):
            for c in range(4):
                selected = (col==c) & (row==r)
                expected.append(((xx+2*yy)[selected]%4==0).mean() if selected.any() else np.nan)
        np.testing.assert_allclose(measured, expected, equal_nan=True)
        self.assertTrue(np.isnan(changes(gray, following, (0,0,8,15), CFG["difference"])).all())

    def test_short_empty_and_constant_observations(self):
        names = feature_names(CFG["temporal"]["lags"])
        p = PairData(4)
        p.change[:,1] = 0  # Observed, true zero even when G is empty.
        v, ok, _, _ = summarize(p, CFG)
        self.assertEqual(int(ok.sum()), 2)
        for f in ("F_cell01_mean", "F_cell01_std"):
            self.assertTrue(ok[names.index(f)])
        np.testing.assert_equal(v, 0)
        p.good[[0, 2]] = True
        v, ok, _, statuses = summarize(p, CFG)
        self.assertFalse(ok[names.index("A_dx")])
        self.assertTrue(ok[names.index("A_speed_mean")])
        self.assertFalse(ok[names.index("C_row0_pearson")])
        self.assertFalse(ok[names.index("D_row0_quarter1")])
        self.assertTrue(ok[names.index("D_row0_quarter2")])
        self.assertEqual(set(statuses), {"run_too_short"})
        p = PairData(96)
        p.good[:] = True
        _, _, statuses = temporal(p, CFG["temporal"], 1e-12)
        self.assertEqual(set(statuses), {"low_detrended_variance"})


if __name__ == "__main__":
    unittest.main()
