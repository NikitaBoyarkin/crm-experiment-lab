"""Tests for the synthetic data-generating process.

Every test runs on synthetic data only. Small configs keep the suite fast; one
default-config smoke test guards the shipped dimensions.
"""

from __future__ import annotations

import numpy as np
from pandas.testing import assert_frame_equal

from crm_lab.balancer import daily_cap_violations
from crm_lab.config import SimulationConfig
from crm_lab.contract import OUTCOMES_COLUMNS
from crm_lab.simulator import default_config, simulate

SMALL = SimulationConfig(n_users=1500, n_days=10, seed=7)
TABLES = ("users", "campaigns", "assignments", "exposures", "outcomes")


def test_default_config_returns_simulation_config() -> None:
    assert isinstance(default_config(), SimulationConfig)


def test_simulate_validates_and_shapes() -> None:
    sim = simulate(SMALL)
    sim.validate()

    assert len(sim.users) == SMALL.n_users
    assert len(sim.campaigns) == SMALL.n_campaigns
    assert len(sim.outcomes) == SMALL.n_users * SMALL.n_days

    offered = SMALL.n_users * SMALL.n_campaigns_per_day * SMALL.n_days
    assert len(sim.assignments) == offered


def test_determinism_same_seed_identical_frames() -> None:
    a = simulate(SMALL)
    b = simulate(SMALL)
    for name in TABLES:
        assert_frame_equal(getattr(a, name), getattr(b, name))


def test_different_seed_changes_data() -> None:
    a = simulate(SMALL)
    b = simulate(SimulationConfig(n_users=1500, n_days=10, seed=8))
    assert not a.outcomes["purchased"].equals(b.outcomes["purchased"])


def test_users_fields_within_domain() -> None:
    sim = simulate(SMALL)
    users = sim.users

    assert set(users["segment"]).issubset(set(SMALL.segments))
    assert set(users["geo"]).issubset(set(SMALL.geos))
    assert set(users["prime_hour"]).issubset(set(SMALL.prime_hours))
    assert users["registered_day"].between(0, SMALL.n_days - 1).all()

    share = users["segment"].value_counts(normalize=True)
    assert (share - 0.5).abs().max() < 0.05


def test_campaigns_fields() -> None:
    campaigns = simulate(SMALL).campaigns
    assert set(campaigns["segment"]).issubset(set(SMALL.segments))
    assert campaigns["priority"].between(0.0, 1.0, inclusive="left").all()
    assert campaigns["true_lift_pp"].between(0.003, 0.02).all()
    assert campaigns["true_unsub_pp"].between(0.0005, 0.004).all()
    assert campaigns["name"].is_unique
    assert campaigns["name"].str.contains("·").all()


def test_group_is_fixed_per_user() -> None:
    sim = simulate(SMALL)
    per_user = sim.assignments.groupby("user_id")["group"].nunique()
    assert (per_user == 1).all()
    assert set(sim.assignments["group"].unique()).issubset({"test", "control", "gcg"})


def test_gcg_share_matches_config() -> None:
    sim = simulate(SMALL)
    first = sim.assignments.drop_duplicates("user_id")
    gcg_share = (first["group"] == "gcg").mean()
    assert abs(gcg_share - SMALL.gcg_share) < 0.05


def test_offers_target_user_segment() -> None:
    sim = simulate(SMALL)
    user_seg = sim.users.set_index("user_id")["segment"]
    camp_seg = sim.campaigns.set_index("campaign_id")["segment"]
    offered_seg = camp_seg.reindex(sim.assignments["campaign_id"]).to_numpy()
    expected = user_seg.reindex(sim.assignments["user_id"]).to_numpy()
    assert (offered_seg == expected).all()


def test_assignment_stage_is_segment() -> None:
    sim = simulate(SMALL)
    assert set(sim.assignments["stage"].unique()) == {"segment"}


def test_exposures_respect_cap_and_subset_of_offers() -> None:
    sim = simulate(SMALL)
    assert daily_cap_violations(sim.exposures, SMALL.balancer.daily_cap).empty

    offered = set(
        zip(
            sim.assignments["user_id"],
            sim.assignments["campaign_id"],
            sim.assignments["day"],
            strict=False,
        )
    )
    sent = set(
        zip(
            sim.exposures["user_id"],
            sim.exposures["campaign_id"],
            sim.exposures["day"],
            strict=False,
        )
    )
    assert sent.issubset(offered)


def test_exposures_hours_are_distinct_within_cell_and_prime_time() -> None:
    sim = simulate(SMALL)
    exp = sim.exposures
    assert set(exp["send_hour"]).issubset(set(SMALL.balancer.hours))
    per_cell = exp.groupby(["user_id", "day"])["send_hour"].nunique()
    counts = exp.groupby(["user_id", "day"]).size()
    assert (per_cell == counts).all()


def test_exposed_true_only_for_test_users() -> None:
    sim = simulate(SMALL)
    group = sim.users.assign(
        group=sim.assignments.drop_duplicates("user_id").set_index("user_id")["group"]
    )
    merged = sim.exposures.merge(group[["user_id", "group"]], on="user_id")
    assert merged.loc[merged["exposed"], "group"].eq("test").all()


def test_outcomes_columns_and_amount_consistency() -> None:
    outcomes = simulate(SMALL).outcomes
    assert list(outcomes.columns) == list(OUTCOMES_COLUMNS)
    assert (outcomes.loc[outcomes["purchased"], "purchase_amount"] > 0).all()
    assert (outcomes.loc[~outcomes["purchased"], "purchase_amount"] == 0).all()
    assert (outcomes["sessions"] >= 0).all()


def test_planted_lift_raises_test_purchase_rate() -> None:
    sim = simulate(SMALL)
    group = sim.assignments.drop_duplicates("user_id").set_index("user_id")["group"]
    merged = sim.outcomes.merge(group, on="user_id")
    rates = merged.groupby("group")["purchased"].mean()
    assert rates["test"] > rates["control"]
    assert rates["test"] > rates["gcg"]


def test_sessions_asymmetry_between_test_and_control() -> None:
    sim = simulate(SMALL)
    group = sim.assignments.drop_duplicates("user_id").set_index("user_id")["group"]
    merged = sim.outcomes.merge(group, on="user_id")
    sessions = merged.groupby("group")["sessions"].mean()
    assert sessions["test"] > sessions["control"] + 0.5


def test_baseline_activity_latent_effect_on_sessions() -> None:
    sim = simulate(SMALL)
    outcomes = sim.outcomes.merge(sim.users[["user_id", "baseline_activity"]], on="user_id")
    corr = np.corrcoef(outcomes["baseline_activity"], outcomes["sessions"])[0, 1]
    assert corr > 0.05
