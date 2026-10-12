"""Calibration tests for the ab-test-analysis readout (``scripts/ab_test_readout.py``).

These pin the *decision logic* the readout adds on top of the shared statistics:

* the SRM check uses the **planned** split, not a hardcoded 50/50;
* Holm adjustment is a real step-down (never smaller than the raw p, larger for the
  losers in a family);
* the default experiment reads as a significant positive primary with a breached
  unsubscribe guardrail and a cap guardrail that holds;
* the ship scenario shows a strict guardrail is unreachable at **every** send cap,
  while an explicit churn budget makes a ship verdict reachable at the cap that
  maximises net.

Everything runs on the synthetic DGP only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import ab_test_readout as r  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sim():
    return r.default_sim()


@pytest.fixture(scope="module")
def per_user(sim):
    return r._per_user_table(sim)


@pytest.fixture(scope="module")
def report():
    return r.build()


# ---------------------------------------------------------------------------
# SRM against the planned split
# ---------------------------------------------------------------------------


def test_srm_passes_on_the_planned_split():
    result = r.srm_check(9030, 8881, 0.5)
    assert not result.is_mismatch
    assert result.pvalue > 0.01


def test_srm_fires_on_a_corrupted_split():
    result = r.srm_check(6000, 4000, 0.5)
    assert result.is_mismatch
    assert result.pvalue < 0.001


def test_srm_respects_a_planned_unequal_split():
    # A designed 90/10 rollout is balanced against 0.10 and broken against 0.50.
    balanced = r.srm_check(9000, 1000, 0.10)
    false_alarm = r.srm_check(9000, 1000, 0.50)
    assert not balanced.is_mismatch
    assert false_alarm.is_mismatch


def test_default_experiment_has_no_srm(sim, per_user):
    test = per_user.loc[per_user["group"] == "test"]
    control = per_user.loc[per_user["group"] == "control"]
    result = r.srm_check(len(control), len(test), float(sim.config.treated_share))
    assert not result.is_mismatch


# ---------------------------------------------------------------------------
# Holm correction
# ---------------------------------------------------------------------------


def test_holm_matches_the_worked_example():
    adjusted = r.holm([0.01, 0.04, 0.03, 0.005])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_is_never_below_the_raw_pvalue():
    raw = [0.001, 0.008, 0.02, 0.2]
    adjusted = r.holm(raw)
    assert all(a >= p for a, p in zip(adjusted, raw, strict=False))


def test_fmt_p_never_prints_a_bare_zero():
    assert r.fmt_p(0.0) == "<1e-300"
    assert r.fmt_p(0.042) == "0.042"
    assert r.fmt_p(float("nan")) == "n/a"


# ---------------------------------------------------------------------------
# Default-experiment readout
# ---------------------------------------------------------------------------


def test_primary_metric_is_significant_and_positive(per_user):
    test = per_user.loc[per_user["group"] == "test"]
    control = per_user.loc[per_user["group"] == "control"]
    result = r.stats.two_proportion_ztest(
        int(control["purchased"].sum()),
        len(control),
        int(test["purchased"].sum()),
        len(test),
    )
    assert result.pvalue < r.ALPHA
    assert result.diff > 0
    assert result.ci_low > 0


def test_unsubscribe_guardrail_breaches(sim, per_user):
    guards = r.guardrail_family(sim, per_user)
    unsub = guards.loc[guards["metric"] == "unsubscribe_rate"].iloc[0]
    assert unsub["delta"] > 0
    assert unsub["p_holm"] < r.ALPHA
    assert bool(unsub["breached"])


def test_notifications_cap_holds(sim, per_user):
    guards = r.guardrail_family(sim, per_user)
    notify = guards.loc[guards["metric"] == "notifications_per_user_per_day"].iloc[0]
    assert notify["treatment"] <= notify["threshold"]
    assert not bool(notify["breached"])


# ---------------------------------------------------------------------------
# "Can we ever ship?" scenario
# ---------------------------------------------------------------------------


def test_ship_scenario_strict_guardrail_never_ships():
    ship = r.ship_scenario(caps=(3, 2, 1))
    assert (ship["verdict_strict"] == "Investigate").all()


def test_ship_scenario_unsub_delta_positive_and_significant_everywhere():
    ship = r.ship_scenario(caps=(3, 2, 1))
    assert (ship["unsub_delta"] > 0).all()
    assert (ship["unsub_p"] < r.ALPHA).all()


def test_ship_scenario_lower_cap_reduces_the_churn_budget():
    ship = r.ship_scenario(caps=(3, 1)).set_index("daily_cap")
    assert ship.loc[1, "unsub_delta"] < ship.loc[3, "unsub_delta"]


def test_ship_scenario_budget_makes_ship_reachable_at_the_best_net_cap():
    ship = r.ship_scenario(caps=(3, 2, 1), tolerance=0.12)
    shippable = ship.loc[ship["verdict_tolerance"] == "Ship"]
    assert not shippable.empty
    best_net_cap = ship.loc[ship["net_per_1k"].idxmax(), "daily_cap"]
    assert best_net_cap in set(shippable["daily_cap"])


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


def test_report_contains_the_decision_sections(report):
    for section in (
        "## Hypothesis",
        "## Design",
        "### Sample Ratio Mismatch (SRM) check",
        "### Primary metric",
        "### Guardrail metrics",
        "### Ship scenario",
        "## Decision",
        "**Verdict:** Investigate",
    ):
        assert section in report
