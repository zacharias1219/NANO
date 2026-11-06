"""evalkit v0: statistics with confidence intervals and the shared results.json schema.

Shared infrastructure for every project in the portfolio (PRD goal 6). NumPy only.
"""
from .results import collect, env_info, metric, validate, write_fragment
from .stats import (
    auc,
    bootstrap_ci,
    bootstrap_stat,
    paired_bootstrap_diff,
    paired_permutation_test,
    spearman,
    wilson_interval,
)

__version__ = "0.1.0"
__all__ = [
    "auc", "bootstrap_ci", "bootstrap_stat", "collect", "env_info", "metric", "paired_bootstrap_diff",
    "paired_permutation_test", "spearman", "validate", "wilson_interval", "write_fragment",
]
