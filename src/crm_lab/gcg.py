"""The global control group (GCG) and cumulative incrementality.

CRM cannot run a per-campaign *local* control forever — a user enrolled in the
local control of one campaign is still a legitimate target of every other
campaign. So platforms keep a **global** control group: a share of users who
are never marketed at all and are used to estimate the cumulative, platform-wide
uplift of all campaigns together (the Avito-style GCG).

This module implements:

* ``refresh_gcg`` — the periodic, *partial* reshuffle of the GCG. Full
  reshuffles throw away the whole control sample at once (the control baseline
  jumps); partial reshuffles move only ``refresh_share`` of it, keeping most of
  the sample stable.
* ``estimate_cumulative_incrementality`` — the cumulative test-minus-GCG
  purchase rate with a normal-approximation confidence interval, per day.
* ``compare_refresh_strategies`` — a coefficient-of-variation proxy of the
  uplift trend under a full vs a partial reshuffle (the partial one is smoother).
* ``gcg_day_mask`` — a per user-day flag frame marking GCG vs marketed users.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crm_lab.contract import STAGE_SEGMENT, SimData

MARKETED_GROUP = "test"
GCG_GROUP = "gcg"
_Z_95 = 1.959963984540054
_UPLIFT_COLUMNS = ("day", "uplift", "ci_low", "ci_high")


def _per_user_group(sim: SimData) -> pd.Series:
    """Map every ``user_id`` to its single fixed group (segment-stage truth)."""
    assignments = sim.assignments
    stage = assignments.get("stage")
    if stage is not None and (stage == STAGE_SEGMENT).any():
        assignments = assignments[stage == STAGE_SEGMENT]
    return assignments.groupby("user_id")["group"].first()


def refresh_gcg(
    user_ids: list[int] | np.ndarray,
    current_gcg: set[int],
    refresh_share: float,
    rng: np.random.Generator,
) -> set[int]:
    """Partially reshuffle the global control group.

    Move roughly ``refresh_share`` of the current GCG out and bring the same
    number of non-GCG users in, leaving the GCG size unchanged and most of its
    membership stable. ``rng`` makes the reshuffle reproducible.
    """
    universe = np.asarray(sorted({int(u) for u in user_ids}))
    universe_set = set(universe.tolist())
    current = {int(u) for u in current_gcg} & universe_set
    n_move = int(round(float(refresh_share) * len(current)))
    non_gcg = np.asarray(sorted(universe_set - current))
    n_move = min(n_move, len(current), len(non_gcg))
    if n_move <= 0:
        return current
    leaving = rng.choice(np.asarray(sorted(current)), size=n_move, replace=False)
    joining = rng.choice(non_gcg, size=n_move, replace=False)
    return (current - {int(u) for u in leaving}) | {int(u) for u in joining}


def _cumulative_means(
    totals: pd.Series, marketed: pd.Index, gcg: pd.Index
) -> tuple[np.ndarray, np.ndarray]:
    marketed_values = totals.reindex(marketed, fill_value=0).to_numpy(dtype=float)
    gcg_values = totals.reindex(gcg, fill_value=0).to_numpy(dtype=float)
    return marketed_values, gcg_values


def _diff_ci(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Normal-approx CI for ``mean(a) - mean(b)``; finite even for tiny arms."""
    diff = float(a.mean() - b.mean())
    var_a = float(a.var(ddof=1)) if len(a) > 1 else 0.0
    var_b = float(b.var(ddof=1)) if len(b) > 1 else 0.0
    se = float(np.sqrt(var_a / len(a) + var_b / len(b)))
    return diff, diff - _Z_95 * se, diff + _Z_95 * se


def estimate_cumulative_incrementality(sim: SimData) -> pd.DataFrame:
    """Cumulative marketed-minus-GCG purchase rate, per day.

    Returns a frame ``day, uplift, ci_low, ci_high`` where ``uplift`` is the
    difference in cumulative purchases *per user* between the marketed (test)
    population and the global control group, with a normal-approximation CI.
    """
    groups = _per_user_group(sim)
    marketed = groups.index[groups == MARKETED_GROUP]
    gcg = groups.index[groups == GCG_GROUP]

    outcomes = sim.outcomes
    relevant = outcomes[outcomes["user_id"].isin(marketed.union(gcg))]
    days = np.sort(relevant["day"].unique())

    rows: list[dict[str, float]] = []
    for day in days:
        totals = relevant[relevant["day"] <= day].groupby("user_id")["purchased"].sum()
        a, b = _cumulative_means(totals, marketed, gcg)
        if len(a) == 0 or len(b) == 0:
            continue
        uplift, ci_low, ci_high = _diff_ci(a, b)
        rows.append({"day": int(day), "uplift": uplift, "ci_low": ci_low, "ci_high": ci_high})
    return pd.DataFrame(rows, columns=list(_UPLIFT_COLUMNS))


def _uplift_series(
    universe: list[int],
    initial_gcg: set[int],
    totals: pd.Series,
    marketed_mean: float,
    days: np.ndarray,
    refresh_share: float,
    seed: int,
) -> np.ndarray:
    """Uplift trend under a given reshuffle strategy over a few refresh cycles.

    The GCG is refreshed at three evenly spaced cycle boundaries inside the
    window. The uplift on each day is the (fixed) marketed mean purchase count
    minus the mean purchase count of the *current* control composition — so a
    strategy that keeps the sample stable yields a flat trend.
    """
    rng = np.random.default_rng(seed)
    gcg = set(initial_gcg)
    n = len(days)
    boundaries = {int(days[i]) for i in sorted({n // 4, n // 2, (3 * n) // 4})}

    series: list[float] = []
    for day in days:
        if int(day) in boundaries:
            gcg = refresh_gcg(universe, gcg, refresh_share, rng)
        values = totals.reindex(sorted(gcg), fill_value=0).to_numpy(dtype=float)
        series.append(marketed_mean - float(values.mean()))
    return np.asarray(series, dtype=float)


def _cv(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    mean = abs(float(values.mean()))
    return float(values.std(ddof=1) / mean) if mean > 0 else 0.0


def compare_refresh_strategies(sim: SimData) -> dict:
    """CV-of-uplift proxy for full vs partial GCG reshuffles.

    A full reshuffle replaces the whole control group at each refresh cycle, so
    its composition — and therefore the control baseline the uplift is measured
    against — jumps; a partial reshuffle rotates only ``gcg_refresh_share`` of
    the sample and keeps most of it stable, giving a smoother trend and a
    *lower* coefficient of variation.
    """
    groups = _per_user_group(sim)
    totals = sim.outcomes.groupby("user_id")["purchased"].sum()

    universe = sorted(int(u) for u in groups.index)
    marketed = groups.index[groups == MARKETED_GROUP]
    initial_gcg = {int(u) for u in groups.index[groups == GCG_GROUP]}
    marketed_mean = float(totals.reindex(marketed, fill_value=0).mean())
    days = np.sort(sim.outcomes["day"].unique())

    seed = int(sim.config.seed)
    full = _uplift_series(universe, initial_gcg, totals, marketed_mean, days, 1.0, seed + 101)
    partial = _uplift_series(
        universe,
        initial_gcg,
        totals,
        marketed_mean,
        days,
        float(sim.config.gcg_refresh_share),
        seed + 202,
    )
    return {"full_reshuffle": _cv(full), "partial": _cv(partial)}


def gcg_day_mask(sim: SimData) -> pd.DataFrame:
    """Per user-day flag frame marking GCG vs marketed users.

    Columns: ``user_id, day, group, is_gcg, is_marketed``. Only marketed (test)
    and GCG users are included, so the frame lines up directly with
    ``estimate_cumulative_incrementality`` and the report figures.
    """
    groups = _per_user_group(sim)
    relevant = groups[groups.isin([MARKETED_GROUP, GCG_GROUP])]
    days = np.sort(sim.outcomes["day"].unique())
    users = relevant.index.to_numpy()

    frame = pd.DataFrame(
        {
            "user_id": np.repeat(users, len(days)),
            "day": np.tile(days, len(users)),
        }
    )
    frame = frame.merge(relevant.rename("group").reset_index(), on="user_id", how="left")
    frame["is_gcg"] = frame["group"] == GCG_GROUP
    frame["is_marketed"] = frame["group"] == MARKETED_GROUP
    return frame[["user_id", "day", "group", "is_gcg", "is_marketed"]]
