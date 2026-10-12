"""Tests for the exposure-window metric.

Two properties are pinned:

1. ``apply_window`` truncates **both** arms over the same horizon (symmetry).
   Truncating only the treated arm would bias the comparison.
2. The window's real benefit shows up on a *transient*-effect DGP as a higher
   signal-to-noise ratio (``effect / SE``), not as a smaller SE in isolation —
   ``window_sensitivity_demo`` provides that DGP.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from crm_lab.config import SimulationConfig
from crm_lab.contract import (
    ASSIGNMENTS_COLUMNS,
    CAMPAIGNS_COLUMNS,
    EXPOSURES_COLUMNS,
    OUTCOMES_COLUMNS,
    USERS_COLUMNS,
    SimData,
)
from crm_lab.exposure import apply_window, compare_any_vs_true, window_sensitivity_demo

WINDOW_DAYS = 6
N_DAYS = 21
N_TEST = 20
N_CONTROL = 20


def _fixture() -> SimData:
    test_ids = [f"t{i}" for i in range(N_TEST)]
    control_ids = [f"c{i}" for i in range(N_CONTROL)]

    outcome_rows = []
    user_rows = []
    assignment_rows = []
    for uid in test_ids + control_ids:
        is_test = uid.startswith("t")
        group = "test" if is_test else "control"
        for day in range(N_DAYS):
            in_window = day <= WINDOW_DAYS
            purchased = int(in_window if is_test else False)
            outcome_rows.append((uid, day, purchased, float(purchased), 0, purchased))
        user_rows.append((uid, 0, "buyer", "MOW", 9, 0.0))
        assignment_rows.append((uid, "c1", 0, group, "segment"))

    users = pd.DataFrame(user_rows, columns=list(USERS_COLUMNS))
    campaigns = pd.DataFrame(
        [("c1", "Campaign One", "buyer", 1.0, 0.02, 0.001)], columns=list(CAMPAIGNS_COLUMNS)
    )
    assignments = pd.DataFrame(assignment_rows, columns=list(ASSIGNMENTS_COLUMNS))
    exposures = pd.DataFrame(
        [(uid, "c1", 0, True, 9) for uid in test_ids], columns=list(EXPOSURES_COLUMNS)
    )
    outcomes = pd.DataFrame(outcome_rows, columns=list(OUTCOMES_COLUMNS))
    return SimData(
        users=users,
        campaigns=campaigns,
        assignments=assignments,
        exposures=exposures,
        outcomes=outcomes,
        config=SimulationConfig(
            n_users=N_TEST + N_CONTROL, n_days=N_DAYS, exposure_window_days=WINDOW_DAYS
        ),
    )


@pytest.fixture
def sim() -> SimData:
    return _fixture()


def test_apply_window_truncates_both_arms_symmetrically(sim: SimData) -> None:
    windowed = apply_window(sim.outcomes, sim.exposures, WINDOW_DAYS)
    exposed = windowed.loc[windowed["user_id"].str.startswith("t")]
    unexposed = windowed.loc[windowed["user_id"].str.startswith("c")]
    # Both arms are scored over the same horizon (first exposure .. +window).
    assert len(exposed) == N_TEST * (WINDOW_DAYS + 1)
    assert len(unexposed) == N_CONTROL * (WINDOW_DAYS + 1)
    assert (exposed["day"] <= WINDOW_DAYS).all()
    assert (unexposed["day"] <= WINDOW_DAYS).all()


def test_window_met_vs_any_uses_same_rows(sim: SimData) -> None:
    windowed = apply_window(sim.outcomes, sim.exposures, WINDOW_DAYS)
    assert len(windowed) < len(sim.outcomes)
    assert set(windowed["user_id"]) == set(sim.outcomes["user_id"])


def test_compare_fields_are_finite(sim: SimData) -> None:
    comparison = compare_any_vs_true(sim)
    for value in (
        comparison.any_effect,
        comparison.any_variance_signal,
        comparison.true_effect,
        comparison.true_variance_signal,
        comparison.any_std_error,
        comparison.true_std_error,
    ):
        assert isinstance(value, float)
        assert math.isfinite(value)


def test_symmetric_window_preserves_effect_on_transient_dgp() -> None:
    """On a transient-effect DGP the window keeps the effect (does not dilute it)."""
    demo = window_sensitivity_demo()
    # Effect must survive the window rather than shrink toward zero.
    assert demo["true_effect"] > 0
    assert demo["true_effect"] > demo["any_effect"]


def test_window_raises_sensitivity_on_transient_dgp() -> None:
    """`Is Exposed = True` wins on signal-to-noise, not on raw SE."""
    demo = window_sensitivity_demo()
    assert demo["true_sensitivity"] > demo["any_sensitivity"]
    # The mechanism: windowing drops rows, so its SE alone is not smaller.
    assert demo["true_std_error"] > demo["any_std_error"]
