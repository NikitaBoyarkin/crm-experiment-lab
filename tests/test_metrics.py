"""Hand-made fixture for the metric taxonomy and guardrail checks."""

from __future__ import annotations

import pandas as pd
import pytest

from crm_lab.config import BalancerConfig, SimulationConfig
from crm_lab.contract import (
    ASSIGNMENTS_COLUMNS,
    CAMPAIGN_METRICS,
    CAMPAIGNS_COLUMNS,
    EXPOSURES_COLUMNS,
    GUARDRAIL_METRICS,
    OEC,
    OUTCOMES_COLUMNS,
    USERS_COLUMNS,
    SimData,
)
from crm_lab.metrics import (
    atomic_metric_defs,
    campbell_law_check,
    evaluate_guardrails,
    metric_hierarchy,
)

N_DAYS = 10


def _fixture(daily_cap: int = 3, sends_per_test_user: int = 1) -> SimData:
    test_ids = [f"t{i}" for i in range(10)]
    control_ids = [f"c{i}" for i in range(10)]

    user_rows = []
    assignment_rows = []
    outcome_rows = []
    for uid in test_ids + control_ids:
        group = "test" if uid.startswith("t") else "control"
        user_rows.append((uid, 0, "buyer", "MOW", 9, 0.0))
        assignment_rows.append((uid, "c1", 0, group, "segment"))
        for day in range(N_DAYS):
            unsubscribed = int(group == "test" and int(uid[1:]) < 8 and day == 0)
            unsubscribed = unsubscribed or int(group == "control" and int(uid[1:]) < 2 and day == 0)
            outcome_rows.append((uid, day, 0, 0.0, int(unsubscribed), 0))

    exposure_rows = []
    for uid in test_ids:
        for k in range(sends_per_test_user):
            exposure_rows.append((uid, "c1", k, True, 9))
    for uid in control_ids:
        exposure_rows.append((uid, "c1", 0, False, 9))

    return SimData(
        users=pd.DataFrame(user_rows, columns=list(USERS_COLUMNS)),
        campaigns=pd.DataFrame(
            [("c1", "Campaign One", "buyer", 1.0, 0.02, 0.001)], columns=list(CAMPAIGNS_COLUMNS)
        ),
        assignments=pd.DataFrame(assignment_rows, columns=list(ASSIGNMENTS_COLUMNS)),
        exposures=pd.DataFrame(exposure_rows, columns=list(EXPOSURES_COLUMNS)),
        outcomes=pd.DataFrame(outcome_rows, columns=list(OUTCOMES_COLUMNS)),
        config=SimulationConfig(
            n_users=20, n_days=N_DAYS, balancer=BalancerConfig(daily_cap=daily_cap)
        ),
    )


def test_metric_hierarchy_keys_and_non_empty() -> None:
    hierarchy = metric_hierarchy()
    assert set(hierarchy) == {"oec", "driver", "proxy", "debug", "guardrail"}
    for values in hierarchy.values():
        assert isinstance(values, list)
        assert len(values) > 0


def test_metric_hierarchy_uses_contract_constants() -> None:
    hierarchy = metric_hierarchy()
    assert hierarchy["oec"] == [OEC]
    assert list(GUARDRAIL_METRICS) == hierarchy["guardrail"]


def test_atomic_metric_defs_schema_and_coverage() -> None:
    defs = atomic_metric_defs()
    required = {"name", "unit", "randomization_unit", "lag_days", "segments", "aggregation"}
    names = [d["name"] for d in defs]
    assert len(names) == len(set(names))
    for definition in defs:
        assert set(definition) == required
        assert definition["unit"]
        assert definition["randomization_unit"] == "user"
        assert isinstance(definition["lag_days"], int)
        assert len(definition["segments"]) >= 1
        assert definition["aggregation"] in {"mean", "sum"}
    assert OEC in names
    assert "retention_d7" in names
    for metric in CAMPAIGN_METRICS:
        assert metric in names
    for metric in GUARDRAIL_METRICS:
        assert metric in names


def test_campbell_law_check_flags_degradation() -> None:
    gamed = campbell_law_check(state_rate=0.5, notified_rate=0.4)
    assert gamed["violates"] is True
    assert gamed["degradation_pp"] == pytest.approx(10.0)

    honest = campbell_law_check(state_rate=0.3, notified_rate=0.4)
    assert honest["violates"] is False
    assert honest["degradation_pp"] == pytest.approx(-10.0)


def test_evaluate_guardrails_columns_and_unsubscribe_breach() -> None:
    frame = evaluate_guardrails(_fixture())
    assert list(frame.columns) == [
        "metric",
        "test_value",
        "control_value",
        "delta",
        "threshold",
        "breached",
    ]
    assert set(frame["metric"]) == {"unsubscribe_rate", "notifications_per_user_per_day"}

    unsub = frame.set_index("metric").loc["unsubscribe_rate"]
    assert unsub["test_value"] == pytest.approx(0.8)
    assert unsub["control_value"] == pytest.approx(0.2)
    assert unsub["delta"] == pytest.approx(0.6)
    assert bool(unsub["breached"]) is True


def test_evaluate_guardrails_notifications_respect_cap() -> None:
    within_cap = evaluate_guardrails(_fixture(daily_cap=1, sends_per_test_user=1))
    notify = within_cap.set_index("metric").loc["notifications_per_user_per_day"]
    assert notify["test_value"] == pytest.approx(0.1)
    assert bool(notify["breached"]) is False

    over_cap = evaluate_guardrails(_fixture(daily_cap=0, sends_per_test_user=1))
    notify_over = over_cap.set_index("metric").loc["notifications_per_user_per_day"]
    assert bool(notify_over["breached"]) is True
