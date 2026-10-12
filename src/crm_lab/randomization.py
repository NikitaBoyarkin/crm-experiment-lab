"""CRM-correct vs product-style randomization.

An experiment can bucket a user in two places:

* at **segment-formation** time — once, fixed for the whole campaign. This is
  what CRM has to do, because the same user is offered many campaigns and must
  keep the same group across all of them (``assign_segment_stage``).
* at **exposure** time — the product-team instinct of bucketing a user only
  when they "show up" in the app. Modelled here as *the users who are active
  (``sessions > 0``) on the launch day*. Because treated users are more active
  (their session count is bumped on send days), nearly all of them enter on day
  one while part of the control arm enters later and is dropped, so the
  entering population is no longer the assigned one and the sample-ratio-mismatch
  check fires even though the true assignment was perfectly balanced
  (``assign_exposure_stage``).

``compare_randomization`` runs both and reports the two SRM p-values side by
side, which is the whole point of the module.
"""

from __future__ import annotations

import pandas as pd

from crm_lab import stats
from crm_lab.contract import (
    STAGE_EXPOSURE,
    STAGE_SEGMENT,
    RandomizationReport,
    SimData,
)

TEST_GROUP = "test"
CONTROL_GROUP = "control"
GCG_GROUP = "gcg"
_LOCAL_GROUPS = (TEST_GROUP, CONTROL_GROUP)


def _per_user_group(sim: SimData) -> pd.Series:
    """Map every ``user_id`` to its single fixed group.

    Reads the segment-stage rows when present (the CRM-correct source of truth);
    falls back to all assignment rows so the function still works on a frame
    that only carries one stage label.
    """
    assignments = sim.assignments
    stage = assignments.get("stage")
    if stage is not None and (stage == STAGE_SEGMENT).any():
        assignments = assignments[stage == STAGE_SEGMENT]
    return assignments.groupby("user_id")["group"].first()


def assign_segment_stage(sim: SimData) -> pd.DataFrame:
    """Per-user fixed assignment (CRM-correct): ``user_id, group, stage``."""
    groups = _per_user_group(sim)
    out = groups.rename("group").reset_index()
    out["stage"] = STAGE_SEGMENT
    return out[["user_id", "group", "stage"]]


def assign_exposure_stage(sim: SimData) -> pd.DataFrame:
    """Product-style mistake: bucket only the users who enter on launch day.

    The product opens the experiment and buckets whoever shows activity
    (``sessions > 0``) on that first day — the moment they "enter the
    experiment". Treated users are more active (their session count is bumped on
    send days), so essentially all of them are present on day one, while a slice
    of the control arm only shows up later and is silently excluded. The
    entering population is therefore no longer the assigned population.
    """
    outcomes = sim.outcomes
    launch_day = outcomes["day"].min()
    active = outcomes[(outcomes["day"] == launch_day) & (outcomes["sessions"] > 0)]
    groups = _per_user_group(sim)
    frame = (
        active[["user_id"]]
        .drop_duplicates()
        .merge(groups.rename("group").reset_index(), on="user_id", how="inner")
    )
    frame["stage"] = STAGE_EXPOSURE
    return frame[["user_id", "group", "stage"]]


def _local_counts(frame: pd.DataFrame) -> tuple[int, int]:
    counts = frame.loc[frame["group"].isin(_LOCAL_GROUPS), "group"].value_counts()
    return int(counts.get(TEST_GROUP, 0)), int(counts.get(CONTROL_GROUP, 0))


def _srm(counts: tuple[int, int], planned: float) -> stats.SRMResult:
    n_test, n_control = counts
    if n_test + n_control == 0:
        return stats.SRMResult(0.0, 1.0, [0, 0], [0.0, 0.0], False)
    return stats.srm_chi2([n_test, n_control], [planned, 1.0 - planned])


def compare_randomization(sim: SimData) -> RandomizationReport:
    """Compare segment-stage vs exposure-stage bucketing via ``stats.srm_chi2``."""
    planned = float(sim.config.treated_share)

    segment = assign_segment_stage(sim)
    exposure = assign_exposure_stage(sim)

    seg_counts = _local_counts(segment)
    exp_counts = _local_counts(exposure)
    seg_srm = _srm(seg_counts, planned)
    exp_srm = _srm(exp_counts, planned)

    seg_total = sum(seg_counts)
    exp_total = sum(exp_counts)
    segment_test_share = seg_counts[0] / seg_total if seg_total else float("nan")
    exposure_test_share = exp_counts[0] / exp_total if exp_total else float("nan")

    note = (
        f"segment-stage buckets test={seg_counts[0]} control={seg_counts[1]} "
        f"(test share {segment_test_share:.3f}, SRM p={seg_srm.pvalue:.3g}); "
        f"exposure-stage buckets test={exp_counts[0]} control={exp_counts[1]} "
        f"(test share {exposure_test_share:.3f}, SRM p={exp_srm.pvalue:.3g}). "
        "The fixed assignment is balanced; bucketing on first activity breaks it."
    )

    return RandomizationReport(
        segment_srm_pvalue=float(seg_srm.pvalue),
        exposure_srm_pvalue=float(exp_srm.pvalue),
        segment_test_share=float(segment_test_share),
        exposure_test_share=float(exposure_test_share),
        note=note,
    )
