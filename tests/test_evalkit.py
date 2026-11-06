"""evalkit v0 (F11): bootstrap CIs, rank statistics, and the results.json schema (M0 exit criterion)."""
import json

import jsonschema
import numpy as np
import pytest
from scipy import stats as sps

from evalkit import auc, bootstrap_ci, metric, paired_permutation_test, spearman, validate, wilson_interval
from evalkit.results import _clean, env_info


def test_bootstrap_ci_covers_the_true_mean_most_of_the_time():
    rng = np.random.default_rng(0)
    hits = 0
    for i in range(200):
        x = rng.normal(1.0, 1.0, size=30)
        lo, hi = bootstrap_ci(x, n_boot=500, seed=i)["ci"]
        hits += lo <= 1.0 <= hi
    assert 0.88 <= hits / 200 <= 0.99


def test_auc_matches_mann_whitney():
    rng = np.random.default_rng(1)
    s, y = rng.standard_normal(60).round(1), rng.integers(0, 2, 60)
    u = sps.mannwhitneyu(s[y == 1], s[y == 0]).statistic
    assert auc(s, y) == pytest.approx(u / ((y == 1).sum() * (y == 0).sum()))


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(2)
    x, y = rng.integers(0, 5, 40), rng.standard_normal(40)
    assert spearman(x, y) == pytest.approx(sps.spearmanr(x, y).statistic)


def test_permutation_test_floor_and_wilson():
    assert paired_permutation_test([1, 2, 3, 4, 5], [0, 0, 0, 0, 0]) == pytest.approx(2 / 32)
    lo, hi = wilson_interval(0, 5)
    assert lo == 0.0 and 0.4 < hi < 0.5


def test_schema_validates_a_sample():
    doc = {
        "schema_version": "1.0", "project": "sample", "generated_at": "2026-01-01T00:00:00+00:00",
        "git": {"commit": None, "dirty": None}, "environment": env_info(),
        "acceptance": {"A1": {"criterion": "c", "threshold": "< 1e-7", "value": 1e-9, "pass": True, "checked_by": "t"}},
        "experiments": {"E0": {"id": "E0", "title": "sample", "generated_at": "x", "environment": env_info(),
                               "metrics": {"m": metric(bootstrap_ci([1.0, 2.0, 3.0], n_boot=100))}}},
    }
    validate(json.loads(json.dumps(_clean(doc))))
    doc["acceptance"]["B1"] = {}
    with pytest.raises(jsonschema.ValidationError):
        validate(doc)
