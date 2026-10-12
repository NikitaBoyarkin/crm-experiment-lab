"""Calibration of the shared statistics against statsmodels / scipy.

The lab's :mod:`crm_lab.stats` module reimplements the few primitives every
analyser composes. A reimplementation is only trustworthy if it agrees with a
reference on the closed-form cases and behaves correctly under a known null, so
each test below pins one primitive to an independent implementation:

* ``required_n_proportion`` / ``mde_proportion`` vs ``statsmodels`` power;
* ``srm_chi2`` vs ``scipy.stats.chisquare`` and the clean-vs-planted split;
* ``two_proportion_ztest`` p-value vs ``statsmodels.stats.proportion``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportions_ztest

from crm_lab import stats

# (baseline proportion, absolute MDE) cases spanning small and large rates.
_CASES = [(0.05, 0.02), (0.10, 0.03), (0.02, 0.01), (0.30, 0.05)]


def _statsmodels_n(p: float, mde: float, alpha: float = 0.05, power: float = 0.80) -> int:
    """Reference per-arm n from statsmodels' two-sample normal power model.

    ``NormalIndPower`` models a two-sample z-test on means with unit sd, so the
    proportion difference is turned into an effect size by dividing by the sd of
    a Bernoulli, ``sqrt(p(1-p))``.
    """
    effect_size = mde / np.sqrt(p * (1 - p))
    n = NormalIndPower().solve_power(
        effect_size=effect_size, alpha=alpha, power=power, ratio=1.0, alternative="two-sided"
    )
    return int(np.ceil(n))


@pytest.mark.parametrize(("p", "mde"), _CASES)
def test_required_n_matches_statsmodels(p: float, mde: float) -> None:
    assert stats.required_n_proportion(p, mde) == _statsmodels_n(p, mde)


@pytest.mark.parametrize(("p", "mde"), _CASES)
def test_required_n_achieves_target_mde(p: float, mde: float) -> None:
    """``mde_proportion`` at the required n returns the requested MDE."""
    n = stats.required_n_proportion(p, mde)
    assert stats.mde_proportion(p, n) == pytest.approx(mde, rel=0.02)


def test_mde_and_required_n_are_inverses() -> None:
    for p, n in [(0.05, 2000), (0.10, 5000), (0.25, 800)]:
        mde = stats.mde_proportion(p, n)
        assert stats.required_n_proportion(p, mde) == pytest.approx(n, rel=0.01)


def test_required_n_grows_as_mde_shrinks() -> None:
    wide = stats.required_n_proportion(0.05, 0.03)
    narrow = stats.required_n_proportion(0.05, 0.01)
    assert narrow > wide


def test_mde_shrinks_as_n_grows() -> None:
    assert stats.mde_proportion(0.05, 4000) < stats.mde_proportion(0.05, 1000)


# ---------------------------------------------------------------------------
# SRM
# ---------------------------------------------------------------------------


def test_srm_clean_split_does_not_fire() -> None:
    result = stats.srm_chi2([5000, 5000])
    assert result.chi2 == pytest.approx(0.0)
    assert result.pvalue == pytest.approx(1.0)
    assert result.is_mismatch is False


def test_srm_planted_60_40_split_fires() -> None:
    result = stats.srm_chi2([6000, 4000])
    assert result.is_mismatch is True
    assert result.pvalue < 0.001
    assert result.chi2 == pytest.approx(400.0)


def test_srm_chi2_matches_scipy() -> None:
    for counts in ([5000, 5000], [5200, 4800], [6000, 4000], [300, 320, 380]):
        ours = stats.srm_chi2(counts)
        reference = scipy_stats.chisquare(counts)
        assert ours.chi2 == pytest.approx(reference.statistic)
        assert ours.pvalue == pytest.approx(reference.pvalue)


def test_srm_respects_non_uniform_expected_share() -> None:
    # 10/90 observed against a planned 50/50 split must fire...
    assert stats.srm_chi2([100, 900], [0.5, 0.5]).is_mismatch is True
    # ...but against the matching 10/90 plan it must not.
    assert stats.srm_chi2([100, 900], [0.1, 0.9]).is_mismatch is False


# ---------------------------------------------------------------------------
# Two-proportion z-test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ca", "na", "cb", "nb"),
    [(100, 1000, 150, 1000), (50, 500, 50, 500), (200, 4000, 240, 4000)],
)
def test_ztest_pvalue_matches_statsmodels(ca: int, na: int, cb: int, nb: int) -> None:
    ours = stats.two_proportion_ztest(ca, na, cb, nb)
    reference_z, reference_p = proportions_ztest([ca, cb], [na, nb])
    # statsmodels orders z as arm_a - arm_b; ours is p_b - p_a, so the sign flips.
    assert ours.z == pytest.approx(-reference_z)
    assert ours.pvalue == pytest.approx(reference_p)


def test_ztest_identical_arms_is_null() -> None:
    result = stats.two_proportion_ztest(120, 2000, 120, 2000)
    assert result.diff == pytest.approx(0.0)
    assert result.pvalue == pytest.approx(1.0)


def test_ztest_ci_excludes_zero_for_large_effect() -> None:
    result = stats.two_proportion_ztest(100, 1000, 150, 1000)
    assert result.diff > 0
    assert result.ci_low > 0
    assert result.ci_low < result.diff < result.ci_high


def test_ztest_ci_contains_diff_under_null() -> None:
    result = stats.two_proportion_ztest(101, 2000, 99, 2000)
    assert result.ci_low < result.diff < result.ci_high
    assert result.ci_low < 0 < result.ci_high
