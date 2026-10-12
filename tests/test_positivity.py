"""Hand-made fixtures for the net-of-unsubscribe analyses.

The frames are tiny and fully deterministic: no simulator import, so the module
is exercised (and its arithmetic pinned) even while ``simulator.py`` is being
written in parallel.
"""

from __future__ import annotations

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
from crm_lab.positivity import balance_table, net_effect, unsubscribe_financials

N_DAYS = 10


def _fixture() -> SimData:
    test_ids = [f"t{i}" for i in range(100)]
    control_ids = [f"c{i}" for i in range(100)]
    purchasers = set(test_ids[:60]) | set(control_ids[:40])
    unsubscribers = set(test_ids[:20]) | set(control_ids[:10])

    outcome_rows = []
    user_rows = []
    assignment_rows = []
    for uid in test_ids + control_ids:
        group = "test" if uid.startswith("t") else "control"
        for day in range(N_DAYS):
            outcome_rows.append(
                (
                    uid,
                    day,
                    int(uid in purchasers and day == 0),
                    float(uid in purchasers and day == 0),
                    int(uid in unsubscribers and day == 0),
                    0,
                )
            )
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
        config=SimulationConfig(n_users=200, n_days=N_DAYS),
    )


@pytest.fixture
def sim() -> SimData:
    return _fixture()


def test_fixture_is_contract_valid(sim: SimData) -> None:
    sim.validate()


def test_net_effect_planted_sign_and_arithmetic(sim: SimData) -> None:
    result = net_effect(sim)
    # test: 60/100 purchase, 20/100 unsub; control: 40/100, 10/100.
    assert result.uplift_purchasers == pytest.approx(200.0)
    assert result.unsubscribe_penalty == pytest.approx(100.0)
    assert result.net_effect == pytest.approx(100.0)
    assert result.balance_ratio == pytest.approx(2.0)
    assert result.net_effect > 0


def test_net_effect_penalty_dominates_is_negative() -> None:
    sim = _fixture()
    # Flip the unsubscribe sets so the treated arm churns far more than it lifts.
    outcomes = sim.outcomes.copy()
    treated_ids = set(sim.users.loc[sim.users["user_id"].str.startswith("t"), "user_id"])
    outcomes.loc[outcomes["user_id"].isin(treated_ids) & (outcomes["day"] == 0), "purchased"] = 0
    outcomes.loc[outcomes["user_id"].isin(treated_ids) & (outcomes["day"] == 0), "unsubscribed"] = 1
    sim.outcomes = outcomes

    result = net_effect(sim)
    assert result.unsubscribe_penalty == pytest.approx(900.0)
    assert result.net_effect < 0


def test_net_effect_guards_zero_penalty() -> None:
    sim = _fixture()
    outcomes = sim.outcomes.copy()
    outcomes["unsubscribed"] = 0
    sim.outcomes = outcomes

    result = net_effect(sim)
    assert result.unsubscribe_penalty == 0.0
    assert result.net_effect == pytest.approx(200.0)
    assert result.balance_ratio == float("inf")


def test_balance_table_shape_values_and_sign(sim: SimData) -> None:
    table = balance_table(sim)
    assert list(table.columns) == ["campaign_id", "positivity", "negativity", "net", "sign"]
    assert len(table) == 1
    row = table.iloc[0]
    assert row["campaign_id"] == "c1"
    assert row["positivity"] == pytest.approx(200.0)
    assert row["negativity"] == pytest.approx(100.0)
    assert row["net"] == pytest.approx(100.0)
    assert row["sign"] == "positive"


def test_balance_table_neutral_sign_when_arms_match() -> None:
    sim = _fixture()
    sim.outcomes["purchased"] = (sim.outcomes["day"] == 0).astype(int)
    sim.outcomes["unsubscribed"] = 0
    table = balance_table(sim)
    assert table.iloc[0]["net"] == 0.0
    assert table.iloc[0]["sign"] == "neutral"


def test_unsubscribe_financials_prices_monthly_churn(sim: SimData) -> None:
    frame = unsubscribe_financials(sim, arpu=500.0)
    assert set(frame["group"]) == {"control", "test"}
    by_group = frame.set_index("group")
    assert by_group.loc["test", "monthly_unsubscribes"] == 20
    assert by_group.loc["control", "monthly_unsubscribes"] == 10
    assert by_group.loc["test", "lost_revenue"] == pytest.approx(20 * 500.0)
    assert by_group.loc["control", "lost_revenue"] == pytest.approx(10 * 500.0)
    assert by_group.loc["test", "unsub_rate"] == pytest.approx(0.2)


def test_unsubscribe_financials_scales_with_arpu(sim: SimData) -> None:
    cheap = unsubscribe_financials(sim, arpu=100.0).set_index("group")["lost_revenue"]
    pricey = unsubscribe_financials(sim, arpu=1000.0).set_index("group")["lost_revenue"]
    assert pricey["test"] == pytest.approx(10 * cheap["test"])
