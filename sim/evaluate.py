"""Scoring game predictions: log loss, calibration, and resample ranges. numpy only.

- Log loss: for each game, minus the log of the chance the model gave to what happened
  (home win or not). Lower is better.
- Calibration line: fit "home won" against the predicted chance on the log-odds scale
  (logistic regression). Slope 1 and intercept 0 means the chances mean what they say.
  A slope under 1 means the chances are too spread out (too sure), over 1 too timid.
- Ten groups: games sorted by predicted chance into ten equal-count groups, predicted vs.
  actual home-win share in each.
- Ranges: 95%, from resampling games 1,000 times.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def log_loss(p, y, w=None):
    p = np.clip(p, EPS, 1 - EPS)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return ll.mean() if w is None else (ll * w).sum() / w.sum()


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_line(p, y, w=None, iters=25):
    """(intercept, slope) of y ~ intercept + slope * logit(p), by Newton's method."""
    x = logit(p)
    w = np.ones_like(x) if w is None else w
    X = np.stack([np.ones_like(x), x], axis=1)
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(X @ beta)))
        g = X.T @ (w * (y - q))
        h = (X * (w * q * (1 - q))[:, None]).T @ X
        step = np.linalg.solve(h, g)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta[0], beta[1]


def ten_groups(p, y):
    order = np.argsort(p, kind="stable")
    groups = np.array_split(order, 10)
    return [(p[g].mean(), y[g].mean(), len(g)) for g in groups], groups


def resample(preds, y, n_boot=1000, seed=0):
    """Ranges for each model's log loss, the gaps between models, and calibration.

    preds: {name: p}. Returns {key: (lo, hi)} from n_boot resamples of the games.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    names = list(preds)
    out = {f"ll:{k}": [] for k in names}
    for a in names:
        for b in names:
            if a < b:
                out[f"gap:{a}-{b}"] = []
    for k in names:
        out[f"cal_int:{k}"], out[f"cal_slope:{k}"] = [], []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        lls = {k: log_loss(preds[k], y, w) for k in names}
        for k in names:
            out[f"ll:{k}"].append(lls[k])
            a_, s_ = calibration_line(preds[k], y, w)
            out[f"cal_int:{k}"].append(a_)
            out[f"cal_slope:{k}"].append(s_)
        for a in names:
            for b in names:
                if a < b:
                    out[f"gap:{a}-{b}"].append(lls[a] - lls[b])
    return {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in out.items()}


def group_ranges(p, y, n_boot=1000, seed=0):
    """Ranges on each of the ten groups' actual home-win share (groups fixed)."""
    rng = np.random.default_rng(seed)
    _, groups = ten_groups(p, y)
    res = []
    for g in groups:
        yy = y[g]
        bs = [yy[rng.integers(0, len(g), len(g))].mean() for _ in range(n_boot)]
        res.append(tuple(np.percentile(bs, [2.5, 97.5])))
    return res
