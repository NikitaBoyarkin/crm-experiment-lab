#!/usr/bin/env python
"""End-to-end CRM experiment pipeline.

Runs the whole lab from one seeded simulation: SRM/randomization, cumulative GCG
incrementality, net effect, the exposure-window comparison and the guardrails.
Writes ``outputs/report.md`` and three ``plots/pipeline_*.png`` figures, and
prints a headline table to stdout.

Usage:
    uv run python scripts/run_full_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt  # noqa: E402

from crm_lab.exposure import compare_any_vs_true  # noqa: E402
from crm_lab.gcg import estimate_cumulative_incrementality  # noqa: E402
from crm_lab.plotting import OKABE_ITO, apply_style, save_fig  # noqa: E402
from crm_lab.positivity import net_effect  # noqa: E402
from crm_lab.randomization import compare_randomization  # noqa: E402
from crm_lab.report import build_report, headline_rows, write_report  # noqa: E402
from crm_lab.simulator import simulate  # noqa: E402


def _figure_srm(sim, rand, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["segment-stage", "exposure-stage"]
    shares = [rand.segment_test_share, rand.exposure_test_share]
    pvalues = [rand.segment_srm_pvalue, rand.exposure_srm_pvalue]
    bars = ax.bar(labels, shares, color=[OKABE_ITO["green"], OKABE_ITO["red"]], width=0.55)
    ax.axhline(sim.config.treated_share, color=OKABE_ITO["grey"], linestyle="--", linewidth=1.2)
    ax.text(
        1.02,
        sim.config.treated_share,
        f" planned {sim.config.treated_share:.2f}",
        va="center",
        fontsize=9,
        color=OKABE_ITO["grey"],
    )
    for bar, share, pvalue in zip(bars, shares, pvalues, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            share + 0.01,
            f"share {share:.3f}\nSRM p={pvalue:.2g}",
            ha="center",
            fontsize=9,
        )
    ax.set_ylim(0, min(1.0, max(shares) + 0.15))
    ax.set_ylabel("test share of the entering population")
    ax.set_title("Randomization: fixed segment split vs first-activity bucketing")
    return save_fig(fig, "pipeline_srm.png", out_dir)


def _figure_gcg(sim, uplift_frame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    days = uplift_frame["day"].to_numpy()
    uplift = uplift_frame["uplift"].to_numpy()
    ax.plot(days, uplift, color=OKABE_ITO["blue"], linewidth=2, label="cumulative uplift")
    ax.fill_between(
        days,
        uplift_frame["ci_low"],
        uplift_frame["ci_high"],
        color=OKABE_ITO["sky"],
        alpha=0.3,
        label="95% CI",
    )
    ax.axhline(0.0, color=OKABE_ITO["black"], linewidth=0.8)
    ax.set_xlabel("day")
    ax.set_ylabel("purchases per user (marketed − GCG)")
    ax.set_title("Global control group: cumulative incrementality")
    ax.legend(frameon=False)
    return save_fig(fig, "pipeline_gcg_uplift.png", out_dir)


def _figure_net(net, out_dir: Path) -> Path:
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
    ax.set_title("Net campaign effect (uplift minus unsubscribe penalty)")
    return save_fig(fig, "pipeline_net_effect.png", out_dir)


def _print_headline(sim) -> None:
    headers, rows = headline_rows(sim)
    widths = [max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))]
    line = "  ".join(h.upper().ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-" * len(line))
    for row in rows:
        print("  ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> int:
    apply_style()
    figures_dir = Path(__file__).resolve().parents[1] / "plots"

    print("Running full CRM experiment pipeline ...")
    sim = simulate()
    sim.validate()

    rand = compare_randomization(sim)
    uplift_frame = estimate_cumulative_incrementality(sim)
    net = net_effect(sim)
    exposure = compare_any_vs_true(sim)

    report_path = write_report(sim)
    figures = [
        _figure_srm(sim, rand, figures_dir),
        _figure_gcg(sim, uplift_frame, figures_dir),
        _figure_net(net, figures_dir),
    ]

    print()
    _print_headline(sim)
    print()
    print(
        f"exposure window  all-days effect {exposure.any_effect:+.5f} (SE {exposure.any_std_error:.5f}) "
        f"vs windowed {exposure.true_effect:+.5f} (SE {exposure.true_std_error:.5f})"
    )
    print(f"report  -> {report_path}")
    for figure in figures:
        print(f"figure  -> {figure}")
    print(f"report bytes: {len(build_report(sim))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
