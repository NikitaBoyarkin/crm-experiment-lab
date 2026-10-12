"""Shared statistics used by every module of the lab.

Kept deliberately small and dependency-light: scipy + numpy. These are the
primitives the analysers compose (proportion tests, cluster-robust SE, bootstrap
CIs, MDE). The test suite calibrates each of them under a known null.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ZTestResult:
    """Result of a two-proportion z-test."""

    diff: float
    z: float
    pvalue: float
    se: float
    ci_low: float
    ci_high: float


@dataclass
class SRMResult:
    """Chi-square sample-ratio-mismatch check against an expected split."""

    chi2: float
    pvalue: float
    observed: list[int]
    expected: list[float]
    is_mismatch: bool


def two_proportion_ztest(
    successes_a: int, n_a: int, successes_b: int, n_b: int, alpha: float = 0.05
) -> ZTestResult:
    """Two-proportion z-test with a normal-approximation confidence interval.

    ``diff`` is ``p_b - p_a`` (positive = B higher).
    """
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    diff = p_b - p_a
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool if se_pool > 0 else 0.0
    pvalue = 2 * (1 - stats.norm.cdf(abs(z)))
    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    return ZTestResult(
        diff=diff,
        z=z,
        pvalue=pvalue,
        se=se_unpooled,
        ci_low=diff - z_crit * se_unpooled,
        ci_high=diff + z_crit * se_unpooled,
    )


def srm_chi2(counts: list[int], expected_share: list[float] | None = None) -> SRMResult:
    """Sample-ratio-mismatch check via a chi-square goodness-of-fit test.

    ``expected_share`` defaults to an even split across the observed buckets.
    """
    observed = np.asarray(counts, dtype=float)
    n = observed.sum()
    if expected_share is None:
        expected_share = [1 / len(observed)] * len(observed)
    expected = n * np.asarray(expected_share, dtype=float)
    chi2 = float(((observed - expected) ** 2 / expected).sum())
    pvalue = float(1 - stats.chi2.cdf(chi2, df=len(observed) - 1))
    return SRMResult(
        chi2=chi2,
        pvalue=pvalue,
        observed=[int(x) for x in observed],
        expected=[float(x) for x in expected],
        is_mismatch=pvalue < 0.001,
    )


def cluster_robust_se(df: pd.DataFrame, value_col: str, cluster_col: str, treat_col: str) -> float:
    """Cluster-robust standard error of a difference in means.

    Standard cluster-robust (Liang-Zeger) sandwich with a finite-sample factor
    for the number of clusters. Used to show why a naive per-row t-test
    under-estimates the SE when observations are nested (switchback / campaign).
    """
    work = df[[value_col, cluster_col, treat_col]].dropna()
    x = work[treat_col].to_numpy(dtype=float)
    y = work[value_col].to_numpy(dtype=float)
    design = np.column_stack([np.ones(len(x)), x])
    xtx_inv = np.linalg.pinv(design.T @ design)
    beta = xtx_inv @ design.T @ y
    resid = y - design @ beta
    clusters = work[cluster_col].to_numpy()
    unique = np.unique(clusters)
    meat = np.zeros((2, 2))
    for c in unique:
        mask = clusters == c
        u = (design[mask] * resid[mask, None]).sum(axis=0)
        meat += np.outer(u, u)
    g = len(unique)
    n, k = design.shape
    correction = g / (g - 1) * (n - 1) / (n - k)
    cov = xtx_inv @ (meat * correction) @ xtx_inv
    return float(np.sqrt(cov[1, 1]))


def bootstrap_ci(
    values: np.ndarray,
    stat: str | callable = "mean",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a statistic of a 1-D array."""
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return (np.nan, np.nan)
    if callable(stat):
        fn = stat
    elif stat == "mean":
        fn = np.mean
    elif stat == "median":
        fn = np.median
    else:  # pragma: no cover - guarded by tests
        raise ValueError(f"unknown statistic '{stat}'")
    draws = np.array([fn(rng.choice(values, size=n, replace=True)) for _ in range(n_boot)])
    return (float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2)))


def mde_proportion(p: float, n_per_arm: int, alpha: float = 0.05, power: float = 0.80) -> float:
    """Minimum detectable absolute effect on a proportion at fixed per-arm n.

    Uses the normal approximation ``M = z_{1-a/2} + z_{power}`` and
    ``MDE = M * sqrt(2 * p * (1-p) / n)``.
    """
    m = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(m * np.sqrt(2 * p * (1 - p) / n_per_arm))


def required_n_proportion(p: float, mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Per-arm sample size needed to detect ``mde`` (absolute) on a proportion."""
    m = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    n = 2 * p * (1 - p) * (m / mde) ** 2
    return int(np.ceil(n))
