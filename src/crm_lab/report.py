"""Assemble the end-to-end markdown report for the CRM experiment lab.

The report is a pure function of a :class:`~crm_lab.contract.SimData`: it runs
every analyser in the package and renders the results into one self-contained
markdown document with a headline table, one section per analysis, embedded
figures and an explicit limitations section. ``write_report`` is the only thing
that touches the filesystem.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crm_lab.balancer import daily_cap_violations
from crm_lab.contract import SimData
from crm_lab.exposure import compare_any_vs_true, window_sensitivity_demo
from crm_lab.gcg import compare_refresh_strategies, estimate_cumulative_incrementality
from crm_lab.metrics import evaluate_guardrails, metric_hierarchy
from crm_lab.plotting import PLOTS_DIR
from crm_lab.positivity import balance_table, net_effect, unsubscribe_financials
from crm_lab.randomization import compare_randomization

OUTPUTS_DIR = PLOTS_DIR.parent / "outputs"
DEFAULT_REPORT_PATH = OUTPUTS_DIR / "report.md"

PIPELINE_FIGURES: tuple[str, ...] = (
    "pipeline_srm.png",
    "pipeline_gcg_uplift.png",
    "pipeline_net_effect.png",
)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------


def _fmt(value: float, nd: int = 4) -> str:
    """Format a float for a markdown cell; special-cases nan / infinities."""
    value = float(value)
    if value != value:
        return "n/a"
    if value == float("inf"):
        return "inf"
    if value == float("-inf"):
        return "-inf"
    return f"{value:.{nd}f}"


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a list of rows as a GitHub-flavoured markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _df_table(frame: pd.DataFrame, float_nd: int = 4) -> str:
    """Render a DataFrame as markdown, rounding float columns."""
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_float_dtype(formatted[column]):
            formatted[column] = formatted[column].map(lambda v: _fmt(v, float_nd))
    return _md_table([str(c) for c in formatted.columns], formatted.astype(str).values.tolist())


# ---------------------------------------------------------------------------
# Headline numbers (shared by the report and the pipeline stdout)
# ---------------------------------------------------------------------------


def headline_rows(sim: SimData) -> tuple[list[str], list[list[str]]]:
    """Compute one compact headline row per analysis.

    Returns ``(headers, rows)`` so both the markdown report and the pipeline's
    stdout table render exactly the same numbers.
    """
    rand = compare_randomization(sim)

    uplift_frame = estimate_cumulative_incrementality(sim)
    last = uplift_frame.iloc[-1] if len(uplift_frame) else None
    if last is not None:
        uplift_cell = (
            f"{last['uplift']:+.4f} / user by day {int(last['day'])} "
            f"[{last['ci_low']:+.4f}, {last['ci_high']:+.4f}]"
        )
    else:  # pragma: no cover - default config always yields rows
        uplift_cell = "n/a"

    refresh = compare_refresh_strategies(sim)
    net = net_effect(sim)
    exp = compare_any_vs_true(sim)
    sens = window_sensitivity_demo()
    guardrails = evaluate_guardrails(sim)
    n_breached = int(guardrails["breached"].sum())

    headers = ["Analysis", "Result", "Reading"]
    rows = [
        [
            "Randomization (SRM)",
            f"segment p={rand.segment_srm_pvalue:.3g}, exposure p={rand.exposure_srm_pvalue:.3g}",
            f"fixed split share {rand.segment_test_share:.3f} is balanced; "
            f"first-activity bucketing share {rand.exposure_test_share:.3f} is not",
        ],
        [
            "GCG cumulative uplift",
            uplift_cell,
            "marketed minus global control, purchases per user; CI excludes 0",
        ],
        [
            "GCG refresh stability",
            f"CV full={refresh['full_reshuffle']:.4f} vs partial={refresh['partial']:.4f}",
            "partial rotation keeps the control baseline stable",
        ],
        [
            "Net effect / 1k treated",
            f"{net.net_effect:+.1f} (uplift {net.uplift_purchasers:+.1f} "
            f"minus unsub {net.unsubscribe_penalty:.1f})",
            "positive net once unsubscribes are priced in",
        ],
        [
            "Exposure window",
            f"SE any={exp.any_std_error:.5f} vs windowed={exp.true_std_error:.5f}",
            f"on the default DGP the window anchors the horizon, not the SE; on a "
            f"transient effect it lifts sensitivity {sens['any_sensitivity']:.1f} -> "
            f"{sens['true_sensitivity']:.1f} (effect/SE)",
        ],
        [
            "Guardrails",
            f"{n_breached} of {len(guardrails)} breached",
            "unsubscribe guardrail breaches, the daily send cap holds",
        ],
    ]
    return headers, rows


# ---------------------------------------------------------------------------
# Report body
# ---------------------------------------------------------------------------


def build_report(sim: SimData) -> str:
    """Build the full markdown report for ``sim``."""
    cfg = sim.config
    rand = compare_randomization(sim)
    uplift_frame = estimate_cumulative_incrementality(sim)
    refresh = compare_refresh_strategies(sim)
    net = net_effect(sim)
    balance = balance_table(sim)
    financials = unsubscribe_financials(sim)
    exp = compare_any_vs_true(sim)
    guardrails = evaluate_guardrails(sim)
    hierarchy = metric_hierarchy()
    cap = cfg.balancer.daily_cap
    violations = daily_cap_violations(sim.exposures, cap)

    headers, rows = headline_rows(sim)

    parts: list[str] = []
    parts.append("# CRM Experiment Lab — pipeline report\n")
    parts.append(
        f"Synthetic CRM experiment: **{cfg.n_users:,} users**, **{cfg.n_days} days**, "
        f"**{cfg.n_campaigns} campaigns** ({cfg.n_campaigns_per_day} offered per user/day, "
        f"daily cap {cap}), seed `{cfg.seed}`. All numbers below are produced by "
        "`scripts/run_full_pipeline.py` from a known data-generating process.\n"
    )

    parts.append("## Headline\n")
    parts.append(_md_table(headers, rows) + "\n")

    # --- Randomization -----------------------------------------------------
    parts.append("## 1. Randomization: segment-stage vs exposure-stage\n")
    parts.append(
        "CRM has to bucket a user **once**, when the campaign segment is built, and keep "
        "that group for every campaign. The product habit of bucketing a user when they "
        '"show up" on launch day instead drops the part of the control arm that arrives '
        "later, so the entering population no longer matches the assigned one and the "
        "sample-ratio-mismatch check fires even though the assignment itself was clean.\n"
    )
    parts.append(
        _md_table(
            ["Stage", "test share", "test", "control", "SRM p-value"],
            [
                [
                    "segment (CRM-correct)",
                    f"{rand.segment_test_share:.4f}",
                    "-",
                    "-",
                    f"{rand.segment_srm_pvalue:.3g}",
                ],
                [
                    "exposure (product-style)",
                    f"{rand.exposure_test_share:.4f}",
                    "-",
                    "-",
                    f"{rand.exposure_srm_pvalue:.3g}",
                ],
            ],
        )
        + "\n"
    )
    parts.append(f"> {rand.note}\n")

    # --- GCG ---------------------------------------------------------------
    parts.append("## 2. Global control group: cumulative incrementality\n")
    parts.append(
        "A share of users is held out of **all** marketing and used to estimate the "
        "cumulative platform-wide uplift. The control is refreshed periodically; a partial "
        "rotation keeps the baseline stable, a full reshuffle makes it jump.\n"
    )
    if len(uplift_frame):
        tail = uplift_frame.tail(3)
        parts.append(_df_table(tail) + "\n")
    parts.append(
        _md_table(
            ["Refresh strategy", "CV of uplift trend", "Reading"],
            [
                ["full reshuffle", _fmt(refresh["full_reshuffle"]), "control baseline jumps"],
                ["partial (config share)", _fmt(refresh["partial"]), "stable, lower variation"],
            ],
        )
        + "\n"
    )

    # --- Net effect --------------------------------------------------------
    parts.append("## 3. Net effect of campaigns (net of unsubscribes)\n")
    parts.append(
        "A campaign only pays off if the incremental purchasers it drives outnumber the "
        "incremental unsubscribes it causes. Both sides are priced per 1,000 treated "
        "users so the sign of the net is directly readable.\n"
    )
    parts.append(
        _md_table(
            ["Metric", "Per 1,000 treated users"],
            [
                ["incremental purchasers (uplift)", _fmt(net.uplift_purchasers, 1)],
                ["incremental unsubscribes (penalty)", _fmt(net.unsubscribe_penalty, 1)],
                ["net effect", _fmt(net.net_effect, 1)],
                ["positivity / negativity ratio", _fmt(net.balance_ratio, 3)],
            ],
        )
        + "\n"
    )
    parts.append("**Per-campaign balance** (campaigns sorted by net):\n")
    parts.append(_df_table(balance.sort_values("net", ascending=False).head(6)) + "\n")
    parts.append("**Churn priced in revenue** (one month of ARPU per unsubscribe):\n")
    parts.append(
        _df_table(financials[["group", "n_users", "monthly_unsubscribes", "lost_revenue"]]) + "\n"
    )

    # --- Exposure window ---------------------------------------------------
    parts.append("## 4. Exposure-window metric vs all-days metric\n")
    parts.append(
        "A CRM send influences a user for a bounded horizon. The `Is Exposed = True` "
        "metric anchors **both arms** at the first exposure day and keeps only the "
        "following days. The honest claim is narrow: dropping rows always *raises* a "
        "proportion's standard error, so the window is not a free variance reduction. "
        "Its value is (a) that it scores the two arms over the same short horizon, and "
        "(b) that it preserves the effect when the campaign's influence is transient. "
        "The default simulator has a *persistent* effect, so here the window anchors the "
        "horizon without shrinking the SE:\n"
    )
    parts.append(
        _md_table(
            ["Metric", "Effect", "Std error", "Var signal", "Effect / SE"],
            [
                [
                    "all days (Is Exposed = Any)",
                    _fmt(exp.any_effect, 5),
                    _fmt(exp.any_std_error, 5),
                    _fmt(exp.any_variance_signal, 12),
                    _fmt(exp.any_effect / exp.any_std_error if exp.any_std_error else 0.0, 2),
                ],
                [
                    "windowed (Is Exposed = True)",
                    _fmt(exp.true_effect, 5),
                    _fmt(exp.true_std_error, 5),
                    _fmt(exp.true_variance_signal, 12),
                    _fmt(exp.true_effect / exp.true_std_error if exp.true_std_error else 0.0, 2),
                ],
            ],
        )
        + "\n"
    )
    sens = window_sensitivity_demo()
    parts.append(
        "**When the window actually pays off** — a transient-effect DGP "
        "(`window_sensitivity_demo`), where the treated lift exists only inside the "
        "window. Windowing no longer just re-anchors the horizon: it stops the worn-off "
        "tail from diluting the effect, so both the effect and the signal-to-noise ratio "
        "rise.\n"
    )
    parts.append(
        _md_table(
            ["Metric (transient DGP)", "Effect", "Std error", "Effect / SE"],
            [
                [
                    "all days (Is Exposed = Any)",
                    _fmt(sens["any_effect"], 5),
                    _fmt(sens["any_std_error"], 5),
                    _fmt(sens["any_sensitivity"], 2),
                ],
                [
                    "windowed (Is Exposed = True)",
                    _fmt(sens["true_effect"], 5),
                    _fmt(sens["true_std_error"], 5),
                    _fmt(sens["true_sensitivity"], 2),
                ],
            ],
        )
        + "\n"
    )

    # --- Guardrails --------------------------------------------------------
    parts.append("## 5. Guardrails and metric hierarchy\n")
    parts.append(
        "A single headline metric is never optimized alone. The OEC floats on drivers, "
        "proxies and debug metrics, and guardrails can block a launch on their own.\n"
    )
    parts.append(_df_table(guardrails) + "\n")
    parts.append(
        _md_table(
            ["Tier", "Metrics"],
            [
                ["OEC", ", ".join(hierarchy["oec"])],
                ["driver", ", ".join(hierarchy["driver"])],
                ["proxy", ", ".join(hierarchy["proxy"])],
                ["debug", ", ".join(hierarchy["debug"])],
                ["guardrail", ", ".join(hierarchy["guardrail"])],
            ],
        )
        + "\n"
    )
    parts.append(
        f"Balancer daily-cap violations (cap={cap}): **{len(violations)}** "
        "user-days exceeded the cap.\n"
    )

    # --- Figures -----------------------------------------------------------
    parts.append("## Figures\n")
    captions = {
        "pipeline_srm.png": "Segment-stage stays at the planned test share; exposure-stage inflates it.",
        "pipeline_gcg_uplift.png": "Cumulative marketed-minus-GCG uplift with a 95% band.",
        "pipeline_net_effect.png": "Incremental purchasers, unsubscribe penalty and the net.",
    }
    for figure in PIPELINE_FIGURES:
        parts.append(f"![{figure}](../plots/{figure})\n\n*{captions[figure]}*\n")

    # --- Limitations -------------------------------------------------------
    parts.append("## Limitations & scope\n")
    parts.append(
        "- Everything runs on **synthetic data from a known DGP** (`crm_lab.simulator`): "
        "no interference between users, no bots, no delayed conversions, no logging loss, "
        "and the planted lift is constant in time."
    )
    parts.append(
        "- There is **no real assignment service, metric store or monitoring** here: the "
        "lab is a single-experiment playground, not an experimentation platform."
    )
    parts.append(
        "- The GCG and window comparisons are **estimators exercised on one seed**, not "
        "a power study; for calibration of the shared statistics see `tests/test_stats.py`."
    )
    parts.append(
        "- `outputs/report.md` and the three `plots/pipeline_*.png` figures are the only "
        "committed artifacts; everything else under `plots/` is regenerated by "
        "`scripts/make_figures.py`.\n"
    )

    return "\n".join(parts)


def write_report(sim: SimData, path: str | Path | None = None) -> Path:
    """Write the markdown report for ``sim``; defaults to ``outputs/report.md``."""
    out_path = Path(path) if path is not None else DEFAULT_REPORT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_report(sim), encoding="utf-8")
    return out_path
