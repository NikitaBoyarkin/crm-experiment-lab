#!/usr/bin/env python
"""Truth check: reproduce every README headline number from the simulator.

The committed fixture ``data/experiment_sample.csv`` carries only the outcomes
schema (no group / segment columns), so the README's 20,000-user headline numbers
cannot be verified from it. They *can* be reproduced in memory from the known DGP
(``crm_lab.simulator.simulate``, seed 42). This script does exactly that and
prints a REPRODUCED / MISMATCH / NOT-CHECKABLE label per claim.

Read-only: nothing is written outside ``reports/eda/``.

Run:
    <shared-venv>/bin/python reports/eda/scripts/02_reproduce_readme.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT / "src"))

from crm_lab.exposure import compare_any_vs_true, window_sensitivity_demo  # noqa: E402
from crm_lab.gcg import compare_refresh_strategies, estimate_cumulative_incrementality  # noqa: E402
from crm_lab.metrics import evaluate_guardrails  # noqa: E402
from crm_lab.positivity import net_effect  # noqa: E402
from crm_lab.randomization import compare_randomization  # noqa: E402
from crm_lab.simulator import simulate  # noqa: E402


def _label(claim: str, expected: float, got: float, tol: float) -> dict:
    ok = abs(got - expected) <= tol
    return {
        "claim": claim,
        "readme_value": expected,
        "recomputed": round(got, 6),
        "status": "REPRODUCED" if ok else "MISMATCH",
        "delta": round(got - expected, 6),
    }


def main() -> None:
    sim = simulate()
    cfg = sim.config

    rand = compare_randomization(sim)
    uplift = estimate_cumulative_incrementality(sim)
    last = uplift.iloc[-1]
    refresh = compare_refresh_strategies(sim)
    net = net_effect(sim)
    exposure = compare_any_vs_true(sim)
    sens = window_sensitivity_demo()
    guards = evaluate_guardrails(sim)
    n_breached = int(guards["breached"].sum())
    cap = float(cfg.balancer.daily_cap)
    notify = float(
        guards.loc[guards["metric"] == "notifications_per_user_per_day", "test_value"].iloc[0]
    )
    unsub_delta = float(guards.loc[guards["metric"] == "unsubscribe_rate", "delta"].iloc[0])

    checks = [
        _label("GCG cumulative uplift/user by day 27", 0.9968, float(last["uplift"]), 5e-5),
        _label("GCG uplift CI low", 0.9433, float(last["ci_low"]), 5e-5),
        _label("GCG uplift CI high", 1.0502, float(last["ci_high"]), 5e-5),
        _label("Net effect / 1k treated", 50.5, float(net.net_effect), 0.05),
        _label("Incremental purchasers / 1k", 212.8, float(net.uplift_purchasers), 0.05),
        _label("Incremental unsubscribes / 1k", 162.3, float(net.unsubscribe_penalty), 0.05),
        _label("CV full refresh", 0.3280, float(refresh["full_reshuffle"]), 5e-5),
        _label("CV partial refresh", 0.0090, float(refresh["partial"]), 5e-5),
        _label("Segment-stage test share", 0.4958, float(rand.segment_test_share), 5e-5),
        _label("Exposure-stage test share", 0.5629, float(rand.exposure_test_share), 5e-5),
        _label("Exposure window: all-days effect", 0.03621, float(exposure.any_effect), 5e-6),
        _label("Exposure window: windowed effect", 0.03598, float(exposure.true_effect), 5e-6),
        _label(
            "Window sensitivity all-days (effect/SE)", 7.6, float(sens["any_sensitivity"]), 0.05
        ),
        _label(
            "Window sensitivity windowed (effect/SE)", 12.3, float(sens["true_sensitivity"]), 0.05
        ),
        _label("Unsubscribe guardrail delta", 0.1623, unsub_delta, 5e-5),
    ]

    results = {
        "config": {
            "n_users": cfg.n_users,
            "n_days": cfg.n_days,
            "n_campaigns": cfg.n_campaigns,
            "seed": cfg.seed,
            "daily_cap": cap,
        },
        "pvalues": {
            "segment_srm": round(float(rand.segment_srm_pvalue), 4),
            "exposure_srm": float(rand.exposure_srm_pvalue),
        },
        "guardrails": {
            "n_breached": n_breached,
            "n_total": int(len(guards)),
            "notifications_per_user_per_day": notify,
            "daily_cap": cap,
            "cap_holds": bool(notify <= cap),
            "unsubscribe_delta": round(unsub_delta, 6),
        },
        "checks": checks,
        "all_reproduced": all(c["status"] == "REPRODUCED" for c in checks),
    }

    (PROJECT / "reports" / "eda" / "readme_truth_check.json").write_text(
        json.dumps(results, indent=2, default=str)
    )
    for c in checks:
        print(
            f"{c['status']:12s} {c['claim']:42s} readme={c['readme_value']} got={c['recomputed']}"
        )
    print(
        f"\nSRM p: segment={results['pvalues']['segment_srm']} exposure={results['pvalues']['exposure_srm']}"
    )
    print(
        f"Guardrails: {n_breached} of {len(guards)} breached; "
        f"notifications {notify:.1f} <= cap {cap:.1f} -> cap_holds={notify <= cap}"
    )
    print(f"All README claims reproduced: {results['all_reproduced']}")


if __name__ == "__main__":
    main()
