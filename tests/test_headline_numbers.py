"""Bind the numbers published in ``README.md`` to the simulator.

``README.md`` is this repo's public face, and until now its headline block was
verified only by ``reports/eda/scripts/02_reproduce_readme.py`` — a script that
pytest never collects (``testpaths = ["tests"]``) and CI never runs. The README
could therefore drift from the simulator with the whole suite green.

The direction of the check matters: the *expected* value is the README's, and it
is compared against what the code recomputes. Formatting the recomputed value the
way the README formats it and requiring that exact substring keeps the assertion
from being a tautology.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crm_lab.contract import SimData
from crm_lab.exposure import compare_any_vs_true, window_sensitivity_demo
from crm_lab.gcg import compare_refresh_strategies, estimate_cumulative_incrementality
from crm_lab.metrics import evaluate_guardrails
from crm_lab.positivity import net_effect
from crm_lab.randomization import compare_randomization
from crm_lab.simulator import simulate

README_PATH = Path(__file__).resolve().parent.parent / "README.md"


@pytest.fixture(scope="module")
def sim() -> SimData:
    return simulate()


def test_readme_headline_numbers_reproduce_from_the_simulator(sim: SimData) -> None:
    readme = README_PATH.read_text(encoding="utf-8")

    rand = compare_randomization(sim)
    uplift = estimate_cumulative_incrementality(sim).iloc[-1]
    refresh = compare_refresh_strategies(sim)
    net = net_effect(sim)
    exposure = compare_any_vs_true(sim)
    guards = evaluate_guardrails(sim)
    unsub_delta = float(guards.loc[guards["metric"] == "unsubscribe_rate", "delta"].iloc[0])

    published = {
        "GCG cumulative uplift": f"{float(uplift['uplift']):+.4f}",
        "GCG uplift CI low": f"{float(uplift['ci_low']):+.4f}",
        "GCG uplift CI high": f"{float(uplift['ci_high']):+.4f}",
        "Net effect / 1k": f"{float(net.net_effect):.1f}",
        "Incremental purchasers / 1k": f"{float(net.uplift_purchasers):+.1f}",
        "Incremental unsubscribes / 1k": f"{float(net.unsubscribe_penalty):.1f}",
        "CV full refresh": f"{float(refresh['full_reshuffle']):.4f}",
        "CV partial refresh": f"{float(refresh['partial']):.4f}",
        "Segment-stage test share": f"{float(rand.segment_test_share):.4f}",
        "Exposure-stage test share": f"{float(rand.exposure_test_share):.4f}",
        "Exposure window: all-days effect": f"{exposure.any_effect:.5f}",
        "Exposure window: all-days SE": f"{exposure.any_std_error:.5f}",
        "Exposure window: windowed effect": f"{exposure.true_effect:.5f}",
        "Exposure window: windowed SE": f"{exposure.true_std_error:.5f}",
        "Unsubscribe guardrail delta": f"{unsub_delta:.4f}",
    }

    missing = {name: value for name, value in published.items() if value not in readme}
    assert not missing, (
        "README no longer matches the simulator. Published value not found in README.md:\n"
        + "\n".join(f"  {name}: expected {value!r} in README" for name, value in missing.items())
    )


def test_readme_transient_sensitivity_matches_the_demo() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    sens = window_sensitivity_demo()
    published = f"{sens['any_sensitivity']:.1f} → {sens['true_sensitivity']:.1f}"
    assert published in readme, (
        f"README's transient-DGP sensitivity line is stale: expected {published!r}"
    )
