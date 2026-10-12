"""Tests for CRM-correct vs product-style randomization.

The frames are built by hand (not via ``crm_lab.simulator``) so the planted
signal is explicit: test users are more active, so an exposure-stage bucket
drifts while the true segment-stage assignment stays balanced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crm_lab import stats
from crm_lab.config import SimulationConfig
from crm_lab.contract import SimData
from crm_lab.randomization import (
    assign_exposure_stage,
    assign_segment_stage,
    compare_randomization,
)

SEGMENT_COLUMNS = ["user_id", "group", "stage"]


def _make_sim(
    n_test: int = 100,
    n_control: int = 100,
    n_gcg: int = 40,
    n_active_control: int = 20,
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
    active_control = set(control_ids[:n_active_control])

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
        group = group_of[user_id]
        for day in range(1, n_days + 1):
            if group == "control":
                sessions = 5 if user_id in active_control else 0
            else:
                sessions = 5
            purchased = 1 if (group == "test" and day >= 3) else 0
            rows.append(
                {
                    "user_id": user_id,
                    "day": day,
                    "purchased": purchased,
                    "purchase_amount": 100.0 if purchased else 0.0,
                    "unsubscribed": 0,
                    "sessions": sessions,
                }
            )
    outcomes = pd.DataFrame(rows)
    exposures = pd.DataFrame(
        {"user_id": [], "campaign_id": [], "day": [], "exposed": [], "send_hour": []}
    )
    config = SimulationConfig(
        n_users=len(all_ids),
        n_days=n_days,
        treated_share=0.5,
        gcg_share=n_gcg / len(all_ids),
    )
    return SimData(users, campaigns, assignments, exposures, outcomes, config)


def test_assign_segment_stage_returns_fixed_per_user_assignment() -> None:
    sim = _make_sim()
    segment = assign_segment_stage(sim)
    assert list(segment.columns) == SEGMENT_COLUMNS
    assert len(segment) == len(sim.users)
    assert (segment["stage"] == "segment").all()
    assert segment["user_id"].is_unique


def test_assign_exposure_stage_only_buckets_active_users() -> None:
    sim = _make_sim()
    exposure = assign_exposure_stage(sim)
    assert list(exposure.columns) == SEGMENT_COLUMNS
    assert (exposure["stage"] == "exposure").all()
    # 100 test + 20 active control + 40 gcg ever show activity.
    assert len(exposure) == 160


def test_segment_stage_is_balanced() -> None:
    report = compare_randomization(_make_sim())
    assert report.segment_srm_pvalue > 0.05
    assert abs(report.segment_test_share - 0.5) < 1e-9


def test_exposure_stage_drifts_and_fires_srm() -> None:
    report = compare_randomization(_make_sim())
    assert report.exposure_test_share > 0.7
    assert report.exposure_test_share != report.segment_test_share
    assert report.exposure_srm_pvalue < 0.001
    assert report.segment_srm_pvalue > report.exposure_srm_pvalue


def test_planted_imbalance_fires_detector() -> None:
    """A corrupted split makes the SRM detector (and ``stats.srm_chi2``) fire."""
    report = compare_randomization(_make_sim(n_test=180, n_control=20))
    assert report.segment_srm_pvalue < 0.001
    assert report.segment_test_share > 0.8
    assert stats.srm_chi2([180, 20], [0.5, 0.5]).is_mismatch is True


def test_exposure_srm_links_to_srm_chi2() -> None:
    report = compare_randomization(_make_sim())
    expected = stats.srm_chi2([100, 20], [0.5, 0.5]).pvalue
    assert np.isclose(report.exposure_srm_pvalue, expected)
