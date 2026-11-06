"""Bootstrap confidence intervals and the rank statistics used by the E2 and E3 analyses.

Every function takes an explicit ``seed`` so a reported interval is reproducible.
The default interval is the percentile bootstrap with 10,000 resamples.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np


def bootstrap_stat(stat: Callable, *arrays, n_boot: int = 10_000, ci: float = 0.95, seed: int = 0,
                   strata=None) -> dict:
    """Percentile bootstrap of ``stat(*arrays)``, resampling rows of all arrays jointly.

    ``strata`` (optional labels) resamples within each stratum, keeping group sizes fixed.
    Resamples where ``stat`` is undefined (NaN, e.g. AUC with one class) are dropped
    and counted in ``n_invalid``.
    """
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays must have the same length")
    rng = np.random.default_rng(seed)
    groups = [np.arange(n)] if strata is None else [np.flatnonzero(np.asarray(strata) == s) for s in np.unique(strata)]
    point = float(stat(*arrays))
    reps = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        reps[b] = stat(*(a[idx] for a in arrays))
    valid = reps[np.isfinite(reps)]
    alpha = (1 - ci) / 2
    lo, hi = (np.quantile(valid, [alpha, 1 - alpha]) if valid.size else (np.nan, np.nan))
    return {"value": point, "ci": [float(lo), float(hi)], "n": int(n), "ci_level": ci,
            "method": f"percentile bootstrap, {n_boot} resamples", "n_invalid": int(n_boot - valid.size)}


def bootstrap_ci(x, stat: Callable = np.mean, **kw) -> dict:
    """CI for a one-sample statistic (default: the mean)."""
    return bootstrap_stat(stat, np.asarray(x, dtype=float), **kw)


def paired_bootstrap_diff(a, b, stat: Callable = np.mean, **kw) -> dict:
    """CI for stat(a - b) over paired observations (same seeds)."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    return bootstrap_stat(stat, d, **kw)


def paired_permutation_test(a, b) -> float:
    """Exact two-sided sign-flip test of mean(a - b) = 0 (exact for n <= 20).

    With n paired seeds the smallest attainable p-value is 2 / 2^n, which is
    0.0625 for n = 5. Reported so readers can see how much the design can show.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if d.size > 20:
        raise ValueError("exact test enumerates 2^n sign patterns; use n <= 20")
    obs = abs(d.mean())
    signs = np.array(list(product([1.0, -1.0], repeat=d.size)))
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= obs - 1e-15))


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), ties share the mean rank."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc(scores, labels) -> float:
    """ROC AUC = P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney). NaN if one class is absent."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels).astype(bool)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = _rank(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc_curve(scores, labels) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds) with a point for every distinct score, predicting positive when score >= t."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels).astype(bool)
    thr = np.concatenate([[np.inf], np.unique(s)[::-1]])
    tpr = np.array([(s[y] >= t).mean() for t in thr])
    fpr = np.array([(s[~y] >= t).mean() for t in thr])
    return fpr, tpr, thr


def spearman(x, y) -> float:
    """Spearman rank correlation with average ranks for ties. NaN if either input is constant."""
    rx, ry = _rank(np.asarray(x, dtype=float)), _rank(np.asarray(y, dtype=float))
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def wilson_interval(k: int, n: int, z: float = 1.959964) -> list[float]:
    """Wilson score interval for a binomial proportion k / n (well behaved at 0 and 1)."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(max(0.0, centre - half)), float(min(1.0, centre + half))]


def median_iqr(x) -> dict:
    x = np.asarray(x, dtype=float)
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75])
    return {"value": float(med), "iqr": [float(q1), float(q3)], "n": int(x.size), "method": "median and IQR"}
