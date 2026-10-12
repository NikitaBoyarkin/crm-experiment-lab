"""Tests for the global control group (GCG) and cumulative incrementality."""

from __future__ import annotations

import numpy as np
import pandas as pd

from crm_lab.config import SimulationConfig
from crm_lab.contract import SimData
from crm_lab.gcg import (
    compare_refresh_strategies,
    estimate_cumulative_incrementality,
    gcg_day_mask,
    refresh_gcg,
)

UPLIFT_COLUMNS = ["day", "uplift", "ci_low", "ci_high"]
MASK_COLUMNS = ["user_id", "day", "group", "is_gcg", "is_marketed"]


def _make_sim(
    n_test: int = 50,
    n_control: int = 50,
    n_gcg: int = 40,
    n_days: int = 8,
    seed: int = 0,
) -> SimData:
    test_ids = list(range(n_test))
    control_ids = list(range(n_test, n_test + n_control))
    gcg_ids = list(range(n_test + n_control, n_test + n_control + n_gcg))
    group_of = {u: "test" for u in test_ids}
    group_of.update({u: "control" for u in control_ids})
    group_of.update({u: "gcg" for u in gcg_ids})
    all_ids = test_ids + control_ids + gcg_ids

    users = pd.DataFrame(
        {
            "user_id": all_ids,
            "registered_day": 0,
            "segment": "buyer",
            "geo": "MOW",
            "prime_hour": 9,
            "baseline_activity": 0.0,
        }
    )
    campaigns = pd.DataFrame(
        {
            "campaign_id": [0],
            "name": ["c0"],
            "segment": ["buyer"],
            "priority": [1.0],
            "true_lift_pp": [1.0],
            "true_unsub_pp": [0.1],
        }
    )
    assignments = pd.DataFrame(
        {
            "user_id": all_ids,
            "campaign_id": 0,
            "day": 1,
            "group": [group_of[u] for u in all_ids],
            "stage": "segment",
        }
    )

    rows = []
    for user_id in all_ids:
        # GCG users are cleanly separated: they never purchase.
        purchased = 1 if group_of[user_id] == "test" else 0
        for day in range(1, n_days + 1):
            rows.append(
                {
                    "user_id": user_id,
                    "day": day,
                    "purchased": purchased,
                    "purchase_amount": 100.0 if purchased else 0.0,
                    "unsubscribed": 0,
                    "sessions": 5,
                }
            )
    outcomes = pd.DataFrame(rows)
    exposures = pd.DataFrame(
        {"user_id": [], "campaign_id": [], "day": [], "exposed": [], "send_hour": []}
    )
    config = SimulationConfig(
        n_users=len(all_ids),
        n_days=n_days,
        seed=seed,
        treated_share=0.5,
        gcg_share=n_gcg / len(all_ids),
        gcg_refresh_share=0.02,
    )
    return SimData(users, campaigns, assignments, exposures, outcomes, config)


def test_refresh_gcg_preserves_size_and_moves_share() -> None:
    universe = list(range(100))
    current = set(range(20))
    new = refresh_gcg(universe, current, 0.25, np.random.default_rng(7))
    assert len(new) == len(current)
    assert len(current - new) == 5
    assert len(new - current) == 5
    assert new <= set(universe)


def test_refresh_gcg_is_deterministic() -> None:
    universe = list(range(100))
    current = set(range(20))
    first = refresh_gcg(universe, current, 0.25, np.random.default_rng(7))
    second = refresh_gcg(universe, current, 0.25, np.random.default_rng(7))
    assert first == second
    other = refresh_gcg(universe, current, 0.25, np.random.default_rng(8))
    assert other != first


def test_refresh_gcg_zero_share_is_noop() -> None:
    universe = list(range(100))
    current = set(range(20))
    assert refresh_gcg(universe, current, 0.0, np.random.default_rng(1)) == current


def test_estimate_cumulative_incrementality_columns_and_finite() -> None:
    frame = estimate_cumulative_incrementality(_make_sim())
    assert list(frame.columns) == UPLIFT_COLUMNS
    assert len(frame) > 0
    assert np.isfinite(frame[["uplift", "ci_low", "ci_high"]].to_numpy()).all()
    assert (frame["uplift"] > 0).all()
    assert (frame["ci_low"] <= frame["uplift"]).all()
    assert (frame["uplift"] <= frame["ci_high"]).all()


def test_partial_reshuffle_is_smoother_than_full() -> None:
    result = compare_refresh_strategies(_make_sim())
    assert set(result) == {"full_reshuffle", "partial"}
    assert result["partial"] < result["full_reshuffle"]


def test_gcg_day_mask_flags() -> None:
    sim = _make_sim()
    mask = gcg_day_mask(sim)
    assert list(mask.columns) == MASK_COLUMNS
    assert mask["user_id"].nunique() == 90  # 50 test + 40 gcg
    assert int(mask["is_gcg"].sum()) == 40 * sim.config.n_days
    assert int(mask["is_marketed"].sum()) == 50 * sim.config.n_days
    assert not (mask["is_gcg"] & mask["is_marketed"]).any()
