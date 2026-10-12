#!/usr/bin/env python
"""Render one chart per module of the CRM experiment lab into ``plots/``.

Every figure goes through :mod:`crm_lab.plotting` (Agg backend, Okabe-Ito
palette, ``save_fig``) so the gallery stays visually consistent. Nothing here is
committed: the figures are regenerated artifacts.

Usage:
    uv run python scripts/make_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from crm_lab.balancer import daily_cap_violations  # noqa: E402
from crm_lab.exposure import window_sensitivity_demo  # noqa: E402
from crm_lab.gcg import estimate_cumulative_incrementality  # noqa: E402
from crm_lab.metrics import evaluate_guardrails  # noqa: E402
from crm_lab.plotting import OKABE_ITO, apply_style, save_fig  # noqa: E402
from crm_lab.positivity import net_effect  # noqa: E402
from crm_lab.randomization import compare_randomization  # noqa: E402
from crm_lab.simulator import simulate  # noqa: E402


def figure_balancer_cap(sim, out_dir: Path) -> Path:
    """Distribution of sends per user-day against the daily cap."""
    counts = sim.exposures.groupby(["user_id", "day"]).size()
    cap = sim.config.balancer.daily_cap
    values = counts.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(values.index.to_numpy(), values.to_numpy(), color=OKABE_ITO["blue"])
    ax.axvline(cap + 0.5, color=OKABE_ITO["red"], linestyle="--", linewidth=1.4)
    ax.text(cap + 0.55, values.max() * 0.8, f"cap = {cap}", color=OKABE_ITO["red"], fontsize=9)
    violations = daily_cap_violations(sim.exposures, cap)
    ax.set_xlabel("sends per user-day")
    ax.set_ylabel("user-days")
    ax.set_title(f"Balancer: daily-cap usage ({len(violations)} violations)")
    for bar, value in zip(bars, values.to_numpy(), strict=True):
        ax.annotate(
            str(int(value)),
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=8,
        )
    return save_fig(fig, "balancer_cap_usage.png", out_dir)


def figure_randomization_srm(sim, out_dir: Path) -> Path:
    """Test/control counts for the two bucketing stages, with SRM p-values."""
    rand = compare_randomization(sim)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["segment-stage", "exposure-stage"]
    shares = [rand.segment_test_share, rand.exposure_test_share]
    pvalues = [rand.segment_srm_pvalue, rand.exposure_srm_pvalue]
    bars = ax.bar(labels, shares, color=[OKABE_ITO["green"], OKABE_ITO["red"]], width=0.55)
    ax.axhline(0.5, color=OKABE_ITO["grey"], linestyle="--", linewidth=1.2)
    for bar, share, pvalue in zip(bars, shares, pvalues, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            share + 0.01,
            f"{share:.3f}\np={pvalue:.2g}",
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(0, min(1.0, max(shares) + 0.15))
    ax.set_ylabel("test share")
    ax.set_title("Randomization: SRM fires when bucketing on first activity")
    return save_fig(fig, "randomization_srm.png", out_dir)


def figure_gcg_uplift(sim, out_dir: Path) -> Path:
    """Cumulative marketed-minus-GCG uplift over the experiment horizon."""
    frame = estimate_cumulative_incrementality(sim)
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    days = frame["day"].to_numpy()
    ax.plot(days, frame["uplift"], color=OKABE_ITO["blue"], linewidth=2, label="cumulative uplift")
    ax.fill_between(
        days, frame["ci_low"], frame["ci_high"], color=OKABE_ITO["sky"], alpha=0.3, label="95% CI"
    )
    ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.8)
    ax.set_xlabel("day")
    ax.set_ylabel("purchases per user")
    ax.set_title("GCG: cumulative incrementality with confidence band")
    ax.legend(frameon=False)
    return save_fig(fig, "gcg_uplift_trend.png", out_dir)


def figure_net_effect(sim, out_dir: Path) -> Path:
    """Incremental purchasers, unsubscribe penalty and the net."""
    net = net_effect(sim)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["incremental\npurchasers", "incremental\nunsubscribes", "net effect"]
    values = [net.uplift_purchasers, -net.unsubscribe_penalty, net.net_effect]
    colors = [
        OKABE_ITO["green"],
        OKABE_ITO["red"],
        OKABE_ITO["blue"] if net.net_effect >= 0 else OKABE_ITO["orange"],
    ]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.8)
    for bar, value in zip(bars, values, strict=True):
        offset = 4 if value >= 0 else -4
        ax.annotate(
            f"{value:+.1f}",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, offset),
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("events per 1,000 treated users")
    ax.set_title("Net effect: positivity minus negativity")
    return save_fig(fig, "positivity_net_effect.png", out_dir)


def figure_exposure_window(sim, out_dir: Path) -> Path:
    """Effect/SE for all-days vs windowed on a transient-effect DGP."""
    demo = window_sensitivity_demo()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["all days\n(Is Exposed = Any)", "windowed\n(Is Exposed = True)"]
    sensitivity = [demo["any_sensitivity"], demo["true_sensitivity"]]
    bars = ax.bar(labels, sensitivity, color=[OKABE_ITO["grey"], OKABE_ITO["blue"]], width=0.55)
    for bar, value in zip(bars, sensitivity, strict=True):
        ax.annotate(
            f"{value:.1f}",
            (bar.get_x() + bar.get_width() / 2, value),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            fontsize=9,
        )
    ax.set_ylabel("signal-to-noise ratio  (effect / SE)")
    ax.set_title("Exposure window on a transient effect: sensitivity, not raw SE")
    return save_fig(fig, "exposure_window_sensitivity.png", out_dir)


def figure_guardrails(sim, out_dir: Path) -> Path:
    """Test-minus-control guardrail deltas, one panel per metric (separate scales)."""
    guardrails = evaluate_guardrails(sim)
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2))
    for ax, (_, row) in zip(axes, guardrails.iterrows(), strict=True):
        breached = bool(row["breached"])
        color = OKABE_ITO["red"] if breached else OKABE_ITO["green"]
        bars = ax.bar(["test − control"], [row["delta"]], color=color, width=0.5)
        ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.8)
        ax.annotate(
            f"{row['delta']:+.4f}",
            (bars[0].get_x() + bars[0].get_width() / 2, row["delta"]),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
        )
        ax.set_title(str(row["metric"]).replace("_", " "), fontsize=11)
        status = "BREACH" if breached else "holds"
        ax.set_xlabel(f"{status} (threshold {row['threshold']:g})")
    fig.suptitle("Guardrails: test-minus-control deltas", fontweight="bold")
    fig.tight_layout()
    return save_fig(fig, "metrics_guardrails.png", out_dir)


def main() -> int:
    apply_style()
    out_dir = Path(__file__).resolve().parents[1] / "plots"
    sim = simulate()

    figures = [
        figure_balancer_cap(sim, out_dir),
        figure_randomization_srm(sim, out_dir),
        figure_gcg_uplift(sim, out_dir),
        figure_net_effect(sim, out_dir),
        figure_exposure_window(sim, out_dir),
        figure_guardrails(sim, out_dir),
    ]
    for figure in figures:
        print(f"figure -> {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
