"""The ``Is Exposed = True`` exposure-window metric vs the all-days metric.

A CRM send influences a user for a limited horizon, not forever. Counting a
user's outcomes over *every* day after the campaign therefore mixes days that
carry the effect with a long tail of days that do not — the effect gets diluted
by unrelated activity. The ``Is Exposed = True`` metric anchors every user at
their first exposure day and keeps only the ``window_days`` that follow it, so
both arms are scored over the same short horizon.

What this does and does not buy (the honest version, demonstrated in
``window_sensitivity_demo``):

- **It preserves the measured effect** when the influence is transient: dropping
  worn-off days stops the effect from being diluted toward zero.
- **It does *not* shrink the standard error by itself.** Fewer rows always make a
  proportion's SE larger; the win is in the signal-to-noise ratio
  ``effect / SE`` (the z-statistic), not in the SE in isolation.
- **It must be applied symmetrically** to both arms. Windowing only the treated
  arm while the control keeps its full timeline biases the comparison.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crm_lab import stats
from crm_lab.config import SimulationConfig
from crm_lab.contract import (
    ASSIGNMENTS_COLUMNS,
    CAMPAIGNS_COLUMNS,
    EXPOSURES_COLUMNS,
    OUTCOMES_COLUMNS,
    STAGE_SEGMENT,
    USERS_COLUMNS,
    ExposureComparison,
    SimData,
)


def _user_group_map(assignments: pd.DataFrame) -> pd.Series:
    """Return a ``user_id -> group`` series from segment-stage assignments."""
    work = assignments
    if "stage" in work.columns and bool((work["stage"] == STAGE_SEGMENT).any()):
        work = work.loc[work["stage"] == STAGE_SEGMENT]
    return work.drop_duplicates("user_id").set_index("user_id")["group"]


def apply_window(outcomes: pd.DataFrame, exposures: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """Keep only the outcome rows that fall inside a user's exposure window.

    ``Is Exposed = True``, applied **symmetrically**:

    - a user with an exposure is anchored at their first exposure day;
    - a user without an exposure (the control/gcg holdout) is anchored at the
      global first-exposure day, so both arms are scored over the same horizon.

    Rows strictly after ``anchor + window_days`` are dropped as post-window
    noise. The symmetry matters: truncating only the treated arm would compare a
    short horizon against a whole timeline and bias the effect.
    """
    out = outcomes.copy()
    exposed = exposures
    if "exposed" in exposed.columns:
        exposed = exposed.loc[exposed["exposed"].astype(bool)]
    first_exposure = exposed.groupby("user_id")["day"].min()
    if first_exposure.empty:
        global_anchor = int(out["day"].min())
    else:
        global_anchor = int(first_exposure.min())
    anchor = out["user_id"].map(first_exposure).fillna(global_anchor)
    keep = out["day"] <= anchor + window_days
    return out.loc[keep].reset_index(drop=True)


def _effect_and_se(
    outcomes: pd.DataFrame, groups: pd.Series, metric: str = "purchased"
) -> tuple[float, float]:
    """Test-vs-control difference in ``metric`` rate and its standard error."""
    joined = outcomes.join(groups, on="user_id", how="inner")
    tested = joined.loc[joined["group"] == "test"]
    control = joined.loc[joined["group"] == "control"]
    n_test = int(len(tested))
    n_control = int(len(control))
    if n_test == 0 or n_control == 0:
        return 0.0, 0.0
    successes_test = int(tested[metric].sum())
    successes_control = int(control[metric].sum())
    # Pass control as arm "a" so the reported diff is test - control (B - A).
    result = stats.two_proportion_ztest(successes_control, n_control, successes_test, n_test)
    return float(result.diff), float(result.se)


def compare_any_vs_true(sim: SimData) -> ExposureComparison:
    """Estimate the effect under the all-days metric and the windowed metric.

    ``any_*`` uses every outcome row; ``true_*`` uses ``apply_window`` with the
    config's ``exposure_window_days``. ``variance_signal`` is the squared
    standard error of the effect. On a *persistent*-effect DGP such as the
    default simulator the windowed SE is not smaller — the value of the window is
    that it anchors the horizon and preserves the effect; see
    ``window_sensitivity_demo`` for the transient regime where it also raises the
    signal-to-noise ratio.
    """
    groups = _user_group_map(sim.assignments)
    windowed = apply_window(sim.outcomes, sim.exposures, sim.config.exposure_window_days)

    any_effect, any_se = _effect_and_se(sim.outcomes, groups)
    true_effect, true_se = _effect_and_se(windowed, groups)

    return ExposureComparison(
        any_effect=any_effect,
        any_variance_signal=any_se**2,
        true_effect=true_effect,
        true_variance_signal=true_se**2,
        any_std_error=any_se,
        true_std_error=true_se,
    )


def window_sensitivity_demo(
    n_per_arm: int = 2000, n_days: int = 28, window_days: int = 7, seed: int = 7
) -> dict[str, float]:
    """Show when the exposure window pays off: a *transient* campaign effect.

    The treated arm's purchase lift applies only inside the window and is gone
    afterwards; the control arm is flat. Under ``Is Exposed = Any`` the treated
    average is diluted by the worn-off tail, so the effect shrinks and the
    signal-to-noise ratio ``effect / SE`` drops. The symmetric window restores
    it. Returns effect, SE and sensitivity for both metrics.
    """
    rng = np.random.default_rng(seed)
    base = 0.04
    lift = 0.03

    def _arm(ids: list[str], treated: bool) -> pd.DataFrame:
        uid = np.repeat(np.asarray(ids, dtype=object), n_days)
        day = np.tile(np.arange(n_days, dtype=np.int64), len(ids))
        p = np.full(len(day), base, dtype=float)
        if treated:
            p = np.where(day <= window_days, base + lift, base)
        purchased = (rng.random(len(day)) < p).astype(np.int64)
        return pd.DataFrame(
            {
                "user_id": uid,
                "day": day,
                "purchased": purchased,
                "purchase_amount": purchased.astype(float),
                "unsubscribed": np.zeros(len(day), dtype=np.int64),
                "sessions": purchased,
            }
        )

    test_ids = [f"t{i}" for i in range(n_per_arm)]
    control_ids = [f"c{i}" for i in range(n_per_arm)]
    outcomes = pd.concat([_arm(test_ids, True), _arm(control_ids, False)], ignore_index=True)[
        list(OUTCOMES_COLUMNS)
    ]

    users = pd.DataFrame(
        [(uid, 0, "buyer", "MOW", 9, 0.0) for uid in test_ids + control_ids],
        columns=list(USERS_COLUMNS),
    )
    campaigns = pd.DataFrame(
        [("c1", "Campaign One", "buyer", 1.0, lift, 0.0)], columns=list(CAMPAIGNS_COLUMNS)
    )
    assignments = pd.DataFrame(
        [
            (uid, "c1", 0, "test" if uid.startswith("t") else "control", "segment")
            for uid in test_ids + control_ids
        ],
        columns=list(ASSIGNMENTS_COLUMNS),
    )
    exposures = pd.DataFrame(
        [(uid, "c1", 0, True, 9) for uid in test_ids], columns=list(EXPOSURES_COLUMNS)
    )
    sim = SimData(
        users=users,
        campaigns=campaigns,
        assignments=assignments,
        exposures=exposures,
        outcomes=outcomes,
        config=SimulationConfig(
            n_users=2 * n_per_arm, n_days=n_days, exposure_window_days=window_days
        ),
    )
    comparison = compare_any_vs_true(sim)

    def _sens(effect: float, se: float) -> float:
        return effect / se if se else 0.0

    return {
        "any_effect": comparison.any_effect,
        "any_std_error": comparison.any_std_error,
        "any_sensitivity": _sens(comparison.any_effect, comparison.any_std_error),
        "true_effect": comparison.true_effect,
        "true_std_error": comparison.true_std_error,
        "true_sensitivity": _sens(comparison.true_effect, comparison.true_std_error),
    }
