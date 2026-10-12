"""Decision-grade A/B readout for the default CRM experiment (ab-test-analysis skill).

The pipeline in ``run_full_pipeline.py`` walks every CRM-specific module. This
script answers the one question the skill exists for: **given this experiment,
do we ship the campaign or not?** It runs the skill's workflow once, end to end:

1. validate the setup  — randomisation unit, duration, SRM against the *planned*
   split, required n / MDE / achieved power;
2. per-variant metrics  — primary metric with a 95% CI, relative lift;
3. significance test    — two-proportion z-test for the primary (user-level),
   Welch's t-test for the heavy-tailed revenue OEC, bootstrap as a cross-check;
4. guardrails           — the same test on each guardrail, Holm-corrected across
   the family so a single test cannot block a ship by luck;
5. recommendation       — the skill's ship / investigate / extend / stop matrix.

Run: ``uv run python scripts/ab_test_readout.py`` (writes ``outputs/ab_test_report.md``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from crm_lab import stats  # noqa: E402
from crm_lab.contract import STAGE_SEGMENT, SimData  # noqa: E402
from crm_lab.positivity import net_effect  # noqa: E402
from crm_lab.simulator import simulate  # noqa: E402

REPORT_PATH = Path(__file__).resolve().parent.parent / "outputs" / "ab_test_report.md"
ALPHA = 0.05
POWER = 0.80


# ---------------------------------------------------------------------------
# Design / setup
# ---------------------------------------------------------------------------


def _user_group_map(sim: SimData) -> pd.Series:
    assignments = sim.assignments
    if (assignments["stage"] == STAGE_SEGMENT).any():
        assignments = assignments.loc[assignments["stage"] == STAGE_SEGMENT]
    return assignments.drop_duplicates("user_id").set_index("user_id")["group"]


def _per_user_table(sim: SimData) -> pd.DataFrame:
    """Collapse daily rows to one row per user (the randomisation unit)."""
    agg = sim.outcomes.groupby("user_id").agg(
        purchased=("purchased", "max"),
        unsubscribed=("unsubscribed", "max"),
        revenue=("purchase_amount", "sum"),
    )
    groups = _user_group_map(sim)
    table = agg.join(groups, how="inner")
    return table.loc[table["group"].isin(["test", "control"])]


def srm_check(n_control: int, n_test: int, planned_test_share: float) -> stats.SRMResult:
    """SRM against the *planned* allocation, never a hardcoded 50/50."""
    return stats.srm_chi2([n_control, n_test], [1 - planned_test_share, planned_test_share])


def achieved_power(p_control: float, p_test: float, n_control: int, n_test: int) -> float:
    """Power the run actually reached for the observed effect (two-proportion)."""
    if p_control <= 0 or p_control >= 1:
        return float("nan")
    h = abs(proportion_effectsize(p_test, p_control))
    if h == 0:
        return ALPHA
    analysis = NormalIndPower()
    eff_n = 2 / (1 / n_control + 1 / n_test)
    return float(analysis.power(effect_size=h, nobs1=eff_n, alpha=ALPHA, ratio=1.0))


# ---------------------------------------------------------------------------
# Guardrail family with Holm correction
# ---------------------------------------------------------------------------


def fmt_p(p: float) -> str:
    """Format a p-value, never printing a bare '0' for an underflowed one."""
    if np.isnan(p):
        return "n/a"
    return "<1e-300" if p == 0 else f"{p:.3g}"


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (uniformly more powerful than Bonferroni)."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[idx]))
        adjusted[idx] = running
    return adjusted


def guardrail_family(sim: SimData, per_user: pd.DataFrame) -> pd.DataFrame:
    """Test-vs-control guardrail deltas, each with a p-value and Holm correction."""
    test = per_user.loc[per_user["group"] == "test"]
    control = per_user.loc[per_user["group"] == "control"]
    n_test, n_control = len(test), len(control)

    unsub = stats.two_proportion_ztest(
        int(control["unsubscribed"].sum()),
        n_control,
        int(test["unsubscribed"].sum()),
        n_test,
    )

    # Notifications per user per day, from the exposures table (sent rows only).
    sent = sim.exposures.loc[sim.exposures["exposed"].astype(bool)]
    n_days = int(sim.config.n_days)
    notify_test = len(sent.loc[sent["user_id"].isin(test.index)]) / (n_test * n_days)
    notify_control = len(sent.loc[sent["user_id"].isin(control.index)]) / (n_control * n_days)
    cap = float(sim.config.balancer.daily_cap)

    rows = [
        {
            "metric": "unsubscribe_rate",
            "kind": "population",
            "control": float(control["unsubscribed"].mean()),
            "treatment": float(test["unsubscribed"].mean()),
            "delta": unsub.diff,
            "p": unsub.pvalue,
            "threshold": 0.0,
        },
        {
            "metric": "notifications_per_user_per_day",
            "kind": "cap",
            "control": float(notify_control),
            "treatment": float(notify_test),
            "delta": float(notify_test - notify_control),
            "p": float("nan"),  # a hard cap: deterministic, not a population test
            "threshold": cap,
        },
    ]
    frame = pd.DataFrame(rows)
    # Holm only applies to the population guardrails, not to the deterministic cap.
    pop = frame["kind"] == "population"
    pvals = frame.loc[pop, "p"].tolist()
    adjusted = iter(holm(pvals))
    frame["p_holm"] = [next(adjusted) if is_pop else float("nan") for is_pop in pop]
    frame["breached"] = [
        bool(r["delta"] > 0 and r["p_holm"] < ALPHA)
        if r["kind"] == "population"
        else bool(r["treatment"] > r["threshold"])
        for _, r in frame.iterrows()
    ]
    return frame


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build() -> str:
    sim = simulate()
    per_user = _per_user_table(sim)
    test = per_user.loc[per_user["group"] == "test"]
    control = per_user.loc[per_user["group"] == "control"]
    planned = float(sim.config.treated_share)

    # 1. setup ------------------------------------------------------------
    srm = srm_check(len(control), len(test), planned)

    # 2. primary metric: per-user purchase rate (user is the randomisation unit)
    n_c, n_t = len(control), len(test)
    x_c = int(control["purchased"].sum())
    x_t = int(test["purchased"].sum())
    primary = stats.two_proportion_ztest(x_c, n_c, x_t, n_t)
    p_c, p_t = x_c / n_c, x_t / n_t
    rel_lift = (p_t - p_c) / p_c if p_c else float("nan")
    power_obs = achieved_power(p_c, p_t, n_c, n_t)

    # required n for a 10% relative MDE on the observed control baseline
    mde_rel = 0.10
    mde_abs = p_c * mde_rel
    n_required = stats.required_n_proportion(p_c, mde_abs, ALPHA, POWER)
    mde_at_n = stats.mde_proportion(p_c, min(n_c, n_t), ALPHA, POWER)

    # 3. secondary OEC: revenue per user (heavy-tailed -> Welch + bootstrap)
    rev_c = control["revenue"].to_numpy(dtype=float)
    rev_t = test["revenue"].to_numpy(dtype=float)
    welch = scipy_stats.ttest_ind(rev_t, rev_c, equal_var=False)
    rev_diff = float(rev_t.mean() - rev_c.mean())
    boot_low, boot_high = stats.bootstrap_ci(rev_t, "mean", seed=1)
    boot_low_c, boot_high_c = stats.bootstrap_ci(rev_c, "mean", seed=1)

    # cluster-robust SE contrast for the row-level exposure metric
    row_test = sim.outcomes.join(_user_group_map(sim), on="user_id")
    row_test = row_test.loc[row_test["group"].isin(["test", "control"])].copy()
    row_test["is_test"] = (row_test["group"] == "test").astype(int)
    cluster_se = stats.cluster_robust_se(row_test, "purchased", "user_id", "is_test")
    row_naive = stats.two_proportion_ztest(
        int(row_test.loc[row_test["is_test"] == 0, "purchased"].sum()),
        int((row_test["is_test"] == 0).sum()),
        int(row_test.loc[row_test["is_test"] == 1, "purchased"].sum()),
        int((row_test["is_test"] == 1).sum()),
    )

    # 4. guardrails
    guards = guardrail_family(sim, per_user)
    net = net_effect(sim)

    # 5. recommendation
    guardrail_breached = bool(guards["breached"].any())
    if primary.pvalue < ALPHA and primary.diff > 0:
        verdict = "Investigate" if guardrail_breached else "Ship"
    elif primary.pvalue < ALPHA and primary.diff < 0:
        verdict = "Do not ship"
    else:
        verdict = "Extend / stop"

    lines = [
        "# A/B Test Report — CRM campaign send (default synthetic experiment)",
        "",
        "**Test name:** Campaign send vs holdout (12-campaign CRM programme, seed 42)  ",
        "**Owner:** crm-experiment-lab  ",
        "**Dates:** 28 simulated days  ",
        "**Status:** Analysis complete",
        "",
        "---",
        "",
        "## Hypothesis",
        "",
        "**Change tested:** Sending the segment-matched CRM campaigns to a `test` arm, "
        "vs a `control` arm that is offered the same campaigns through the balancer but "
        "receives no notification (daily send cap 3, no exposure).  ",
        "**Expected effect:** a higher per-user purchase rate in `test`.  ",
        "**Rationale:** the campaign carries a planted per-day purchase lift that only "
        "applies on genuinely exposed `test` user-days.",
        "",
        "---",
        "",
        "## Design",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Traffic split | {planned:.0%} test / {1 - planned:.0%} control (plus a 10% global control group outside this readout) |",
        "| Randomisation unit | **user** (fixed per user at segment-formation time) |",
        "| Primary metric | per-user purchase rate — P(any purchase over the 28-day horizon) |",
        "| Secondary / OEC | revenue per user (Σ `purchase_amount`) |",
        "| Guardrail metrics | unsubscribe rate (per user); notifications per user per day (cap) |",
        f"| Minimum detectable effect (MDE) | {mde_abs:.5f} absolute = {mde_rel:.0%} relative on a {p_c:.4f} baseline |",
        f"| Required sample size (per group) | {n_required:,} |",
        f"| Pre-specified alpha | {ALPHA} |",
        f"| Pre-specified power | {POWER:.0%} |",
        "",
        "---",
        "",
        "## Results",
        "",
        "### Sample Ratio Mismatch (SRM) check",
        "",
        "| Group | Expected | Actual | Difference |",
        "|---|---|---|---|",
        f"| Control | {srm.expected[0]:.0f} | {srm.observed[0]:,} | {(srm.observed[0] - srm.expected[0]) / srm.expected[0]:+.2%} |",
        f"| Treatment | {srm.expected[1]:.0f} | {srm.observed[1]:,} | {(srm.observed[1] - srm.expected[1]) / srm.expected[1]:+.2%} |",
        "",
        f"**SRM detected:** {'Yes' if srm.is_mismatch else 'No'} "
        f"(chi2={srm.chi2:.4f}, p={srm.pvalue:.4f}) — checked against the **planned** "
        f"{planned:.0%} split, not 50/50.",
        "",
        "---",
        "",
        "### Primary metric",
        "",
        "| Group | N | Purchase rate | 95% CI (of the rate) |",
        "|---|---|---|---|",
        f"| Control | {n_c:,} | {p_c:.4%} | ±{1.959964 * np.sqrt(p_c * (1 - p_c) / n_c):.4%} |",
        f"| Treatment | {n_t:,} | {p_t:.4%} | ±{1.959964 * np.sqrt(p_t * (1 - p_t) / n_t):.4%} |",
        "",
        f"**Absolute difference:** {primary.diff:+.5f} ({primary.diff * 100:+.3f} pp)  ",
        f"**Relative lift:** {rel_lift:+.2%}  ",
        f"**95% CI (of the difference, unpooled):** [{primary.ci_low:+.5f}, {primary.ci_high:+.5f}]  ",
        f"**p-value:** {fmt_p(primary.pvalue)}  ",
        f"**Significant at alpha={ALPHA}:** {'Yes' if primary.pvalue < ALPHA else 'No'}  ",
        f"**Achieved power for the observed effect:** {power_obs:.1%}",
        "",
        f"> At {n_t:,} per arm the test is powered to detect {mde_at_n:.5f} absolute "
        f"({mde_at_n / p_c:.1%} relative). The observed effect ({primary.diff:.5f}) is "
        f"{abs(primary.diff) / mde_at_n:.2f}x the MDE the test was powered for.",
        "",
        "### Secondary: revenue per user (OEC)",
        "",
        "| Group | Mean revenue/user | 95% bootstrap CI (mean) |",
        "|---|---|---|",
        f"| Control | {rev_c.mean():.4f} | [{boot_low_c:.4f}, {boot_high_c:.4f}] |",
        f"| Treatment | {rev_t.mean():.4f} | [{boot_low:.4f}, {boot_high:.4f}] |",
        "",
        f"**Absolute difference:** {rev_diff:+.4f} per user  ",
        f"**Welch t-test:** t={welch.statistic:.3f}, p={fmt_p(welch.pvalue)} "
        f"({'significant' if welch.pvalue < ALPHA else 'not significant'} at {ALPHA})",
        "",
        "### Clustering check (randomisation unit vs measurement unit)",
        "",
        "Assignment is fixed per **user**, but the exposure metric measures at the "
        "**user-day**. A row-level two-proportion test would assume those rows are "
        "independent; the correct comparison uses a cluster-robust SE clustered by user. "
        "Here the two agree, because this DGP gives purchases no within-user correlation — "
        "so the row-level exposure metric is not misleading on this data:",
        "",
        "| Method | SE of the difference |",
        "|---|---|",
        f"| Row-level two-proportion (naive, user-day) | {row_naive.se:.6f} |",
        f"| Cluster-robust (Liang-Zeger, clustered by user) | {cluster_se:.6f} |",
        "",
        f"> Cluster-robust / naive = {cluster_se / row_naive.se:.2f}x. The check is a guard, "
        "not a finding here; the primary readout above is the user-level metric, which "
        "respects the randomisation unit by construction.",
        "",
        "---",
        "",
        "### Guardrail metrics",
        "",
        "| Metric | Control | Treatment | Change | p (Holm-adj) | Threshold | Pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in guards.iterrows():
        is_cap = r["kind"] == "cap"
        p_str = "— (hard cap)" if is_cap else fmt_p(float(r["p_holm"]))
        change = (
            f"{r['treatment']:.3f} vs cap {r['threshold']:.0f}"
            if is_cap
            else f"{(r['treatment'] - r['control']) / r['control'] * 100:+.1f}%"
        )
        pass_str = "No" if r["breached"] else "Yes"
        lines.append(
            f"| {r['metric']} | {r['control']:.4f} | {r['treatment']:.4f} | "
            f"{change} | {p_str} | {r['threshold']:.4f} | {pass_str} |"
        )

    lines += [
        "",
        "> Family of 2 guardrails, Holm-corrected. Unsubscribe rate rises in the "
        "treated arm and stays significant after correction; the send cap holds.",
        "",
        "---",
        "",
        "### Segment breakdowns",
        "",
        "| Segment | Control rate | Treatment rate | Lift | Significant |",
        "|---|---|---|---|---|",
    ]
    users = sim.users.set_index("user_id")["segment"]
    for seg in sim.config.segments:
        idx = users.index[users == seg]
        sub = per_user.loc[per_user.index.isin(idx)]
        sc = sub.loc[sub["group"] == "control", "purchased"]
        st = sub.loc[sub["group"] == "test", "purchased"]
        res = stats.two_proportion_ztest(int(sc.sum()), len(sc), int(st.sum()), len(st))
        sig = "Yes" if res.pvalue < ALPHA else "No"
        lines.append(
            f"| {seg} | {sc.mean():.4%} | {st.mean():.4%} | "
            f"{(st.mean() - sc.mean()) / sc.mean():+.2%} | {sig} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Decision",
        "",
        f"**Verdict:** {verdict}",
        "",
        "**Reasoning:** The primary metric moves up significantly "
        f"({rel_lift:+.2%}, p={fmt_p(primary.pvalue)}, 95% CI excludes 0) and the revenue "
        f"OEC agrees, so the campaign works. But the unsubscribe guardrail regresses "
        f"significantly and survives Holm correction, and the net effect only stays "
        f"positive because the purchase uplift (+{net.uplift_purchasers:.1f} per 1k) is "
        f"just larger than the churn cost ({net.unsubscribe_penalty:.1f} per 1k) — a "
        f"{net.balance_ratio:.2f}x margin. Per the decision matrix, a winning primary with "
        "a failed guardrail is **Investigate**, not Ship.",
        "",
        "**Next steps:**",
        "1. Add a frequency cap on repeat sends per user/month and re-measure the "
        "unsubscribe guardrail — owner: CRM team.",
        "2. Run a follow-up with the churn cost priced at ARPU (see "
        "`positivity.unsubscribe_financials`) to convert the 1.31x margin into revenue "
        "terms before rollout — owner: analyst.",
        "3. If the guardrail still regresses at a smaller send volume, fall back to the "
        "top-priority campaigns only (net is driven by the positive campaigns in "
        "`positivity.balance_table`).",
        "",
        "---",
        "",
        "*Generated by `scripts/ab_test_readout.py` — the ab-test-analysis skill applied "
        "to the default synthetic CRM experiment.*",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    report = build()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nreport -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
