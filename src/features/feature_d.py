"""Group D: lag correlations, peaks and temporal quarters."""

import numpy as np
from scipy import signal

from .feature_utils import pearson, runs


def temporal(data, cfg, eps):
    """3.3.4. Original-time quarters, lagged Pearson on the earliest longest run."""
    lags = cfg["lags"]
    corr = np.full((6, len(lags)), np.nan)
    peak = np.full((6, 2), np.nan)
    quarters = np.full((6, 4), np.nan)
    z = data.energy.reshape(data.n, 6, 2).mean(axis=2)
    for q, indices in enumerate(np.array_split(np.arange(data.n), 4)):
        usable = indices[data.good[indices]]
        if usable.size:
            quarters[:, q] = z[usable].mean(axis=0)
    start, stop = max(runs(data.good), key=lambda run: run[1] - run[0], default=(0, 0))
    n = stop - start
    status = ["run_too_short"] * 6
    limit = min(max(lags), n // 2)
    if n >= cfg["min_run"]:
        series = signal.detrend(z[start:stop], axis=0, type="linear")
        for row, s in enumerate(series.T):
            if s.std() < cfg["min_detrended_std"]:
                status[row] = "low_detrended_variance"
                continue
            curve = np.full(limit + 1, np.nan)
            for lag in range(3, limit + 1):
                curve[lag] = pearson(s[:-lag], s[lag:], eps)
            for i, lag in enumerate(lags):
                if lag <= limit:
                    corr[row, i] = curve[lag]
            # On ties keep the shorter lag; never use an unavailable neighbor.
            candidates = []
            for a, b in runs(np.isfinite(curve)):
                peaks = signal.find_peaks(curve[a:b])[0] + a
                candidates.extend(int(k) for k in peaks if curve[k] > 0)
            status[row] = (
                "no_positive_local_peak" if np.isfinite(curve).any() else "undefined_pearson"
            )
            if candidates:
                k = max(candidates, key=curve.__getitem__)
                peak[row] = [k / max(lags), curve[k]]
                status[row] = "ok"
    return np.r_[corr.ravel(), peak.ravel(), quarters.ravel()], (start, stop), status
