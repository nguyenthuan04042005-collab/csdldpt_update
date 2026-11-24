"""Column definitions, spatial cells and shared statistics."""

import numpy as np


GROUPS = {"A": 16, "B": 120, "C": 24, "D": 84, "E": 48, "F": 32}
FEATURE_COUNT = sum(GROUPS.values())
GROUP_SLICES = {}
_offset = 0
for _group, _size in GROUPS.items():
    GROUP_SLICES[_group] = slice(_offset, _offset + _size)
    _offset += _size


def feature_names(lags):
    """Column order shared by extraction, exported schema and verification."""
    if len(lags) != 8:
        raise ValueError("Chapter 3 group D requires exactly eight lags")
    names = [
        "A_" + k
        for k in (
            "ax_mean",
            "ay_mean",
            "ax_std",
            "ay_std",
            "speed_q25",
            "speed_q50",
            "speed_q75",
            "speed_mean",
            "speed_all",
            "dx",
            "dy",
            "direction_consistency",
            "delta_mean",
            "delta_std",
            "aspect_mean",
            "aspect_std",
        )
    ]
    names += [f"B_cell{k:02d}_dir{45 * j:03d}" for k in range(12) for j in range(8)]
    names += [f"B_cell{k:02d}_{stat}" for stat in ("mean", "std") for k in range(12)]
    names += [f"C_row{r}_{s}" for r in range(6) for s in ("balance", "pearson", "in", "out")]
    names += [f"D_row{r}_lag{lag:02d}" for r in range(6) for lag in lags]
    names += [f"D_row{r}_{s}" for r in range(6) for s in ("peak_lag_norm", "peak_r")]
    names += [f"D_row{r}_quarter{q}" for r in range(6) for q in range(4)]
    names += [
        f"E_{kind}{k}_{s}" for kind in ("band", "cell") for s in ("mean", "std") for k in range(8)
    ]
    names += [f"E_graybin{j:02d}" for j in range(16)]
    names += [f"F_cell{k:02d}_{s}" for s in ("mean", "std") for k in range(16)]
    assert len(names) == len(set(names)) == FEATURE_COUNT
    return names


def cell_ids(h, w):
    return (np.arange(h)[:, None] * 6 // h) * 2 + np.arange(w)[None, :] * 2 // w


def runs(indices):
    """Return half-open runs of consecutive true entries."""
    edges = np.diff(np.r_[False, indices, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def pearson(x, y, eps):
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= eps:
        return np.nan
    return float(np.clip(np.dot(x, y) / denom, -1, 1))
