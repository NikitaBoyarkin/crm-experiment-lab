"""Tests for the send balancer.

The balancer is pure: given the same seeded generator it returns the same
selection, and it must never let a user exceed the daily notification cap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from crm_lab.balancer import balance_day, daily_cap_violations
from crm_lab.config import BalancerConfig, SimulationConfig
from crm_lab.simulator import simulate

CFG = BalancerConfig(daily_cap=3, hours=(9, 13, 19, 21))


def _offers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2],
            "campaign_id": [10, 11, 12, 13, 14, 20, 21],
            "priority": [0.9, 0.5, 0.7, 0.2, 0.1, 0.3, 0.8],
            "day": [0, 0, 0, 0, 0, 0, 0],
        }
    )


def test_selects_top_priority_under_cap() -> None:
    out = balance_day(_offers(), CFG, np.random.default_rng(0))
    user1 = set(out.loc[out["user_id"] == 1, "campaign_id"])
    user2 = set(out.loc[out["user_id"] == 2, "campaign_id"])
    assert user1 == {10, 12, 11}  # 0.9, 0.7, 0.5
    assert user2 == {21, 20}  # only two offered, both fit


def test_cap_never_exceeded() -> None:
    out = balance_day(_offers(), CFG, np.random.default_rng(0))
    per_user_day = out.groupby(["user_id", "day"]).size()
    assert (per_user_day <= CFG.daily_cap).all()


def test_hours_distinct_and_from_config() -> None:
    out = balance_day(_offers(), CFG, np.random.default_rng(0))
    assert set(out["send_hour"]).issubset(set(CFG.hours))
    for _, cell in out.groupby(["user_id", "day"]):
        assert cell["send_hour"].is_unique


def test_hours_follow_slot_order() -> None:
    out = balance_day(_offers(), CFG, np.random.default_rng(0))
    user1 = out.loc[out["user_id"] == 1].sort_values("send_hour")
    # top-priority 10 -> first hour, then 12, then 11
    assert list(user1["campaign_id"]) == [10, 12, 11]
    assert list(user1["send_hour"]) == [9, 13, 19]


def test_cap_limited_by_available_hours() -> None:
    cfg = BalancerConfig(daily_cap=5, hours=(9, 13))
    out = balance_day(_offers(), cfg, np.random.default_rng(0))
    assert (out.groupby(["user_id", "day"]).size() <= 2).all()


def test_deterministic_for_same_seed() -> None:
    a = balance_day(_offers(), CFG, np.random.default_rng(1))
    b = balance_day(_offers(), CFG, np.random.default_rng(1))
    assert_frame_equal(a, b)


def test_empty_input_returns_empty_with_columns() -> None:
    empty = pd.DataFrame(columns=["user_id", "campaign_id", "priority", "day"])
    out = balance_day(empty, CFG, np.random.default_rng(0))
    assert out.empty
    assert list(out.columns) == ["user_id", "campaign_id", "day", "send_hour"]


def test_daily_cap_violations_flags_overflow() -> None:
    exposures = pd.DataFrame({"user_id": [1, 1, 1, 1, 2, 2], "day": [0, 0, 0, 0, 0, 1]})
    violations = daily_cap_violations(exposures, cap=3)
    assert len(violations) == 1
    row = violations.iloc[0]
    assert (row["user_id"], row["day"], row["n_sends"]) == (1, 0, 4)


def test_daily_cap_violations_none_on_simulated_exposures() -> None:
    sim = simulate(SimulationConfig(n_users=1500, n_days=10, seed=7))
    assert daily_cap_violations(sim.exposures, sim.config.balancer.daily_cap).empty


def test_daily_cap_violations_empty_input() -> None:
    out = daily_cap_violations(pd.DataFrame(columns=["user_id", "day"]), cap=3)
    assert out.empty
    assert list(out.columns) == ["user_id", "day", "n_sends"]
