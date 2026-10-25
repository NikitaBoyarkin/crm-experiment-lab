"""The metric taxonomy and the guardrail / Campbell's-law checks.

Every experiment needs a hierarchy (OEC -> driver -> proxy -> debug -> guardrail)
so that a single headline metric is never optimized in isolation. The hierarchy
here is built straight from the constants in :mod:`crm_lab.contract`, and
``atomic_metric_defs`` gives each metric an atomic definition in the sense of the
Trisigma handbook: unit, randomization unit, lag, segments and aggregation.
"""

from __future__ import annotations

import pandas as pd

from crm_lab.contract import (
    CAMPAIGN_METRICS,
    DEBUG_METRICS,
    DRIVER_METRICS,
    GUARDRAIL_METRICS,
    OEC,
    PROXY_METRICS,
    STAGE_SEGMENT,
    SimData,
)

_SEGMENTS = ["buyer", "seller"]

# name -> (unit, randomization_unit, lag_days, segments, aggregation)
_METRIC_META: dict[str, tuple[str, str, int, list[str], str]] = {
    OEC: ("currency", "user", 7, _SEGMENTS, "sum"),
    "retention_d7": ("ratio", "user", 7, _SEGMENTS, "mean"),
    "orders_per_user": ("count", "user", 0, _SEGMENTS, "mean"),
    "aov": ("currency", "user", 0, _SEGMENTS, "mean"),
    "session_rate": ("ratio", "user", 0, _SEGMENTS, "mean"),
    "send_hour_match": ("ratio", "user", 0, _SEGMENTS, "mean"),
    "exposure_rate": ("ratio", "user", 0, _SEGMENTS, "mean"),
    "open_rate": ("ratio", "user", 0, _SEGMENTS, "mean"),
    "ctr": ("ratio", "user", 0, _SEGMENTS, "mean"),
    "conversion_to_purchase": ("ratio", "user", 0, _SEGMENTS, "mean"),
    "unsubscribe_rate": ("ratio", "user", 0, _SEGMENTS, "mean"),
    "notifications_per_user_per_day": ("count", "user", 0, _SEGMENTS, "mean"),
}


def metric_hierarchy() -> dict[str, list[str]]:
    """Return the OEC / driver / proxy / debug / guardrail taxonomy.

    Built only from the contract constants, so the report and the tests share one
    source of truth. Every bucket is non-empty.
    """
    return {
        "oec": [OEC],
        "driver": list(DRIVER_METRICS),
        "proxy": list(PROXY_METRICS),
        "debug": list(DEBUG_METRICS),
        "guardrail": list(GUARDRAIL_METRICS),
    }


def atomic_metric_defs() -> list[dict]:
    """One atomic definition per metric, in hierarchy order.

    Each dict has exactly the keys ``name, unit, randomization_unit, lag_days,
    segments, aggregation``. The taxonomy buckets are flattened without
    duplication, so the OEC, drivers, proxies, debug metrics, campaign metrics
    and guardrails all appear once.
    """
    ordered: list[str] = [OEC]
    for bucket in (
        DRIVER_METRICS,
        PROXY_METRICS,
        DEBUG_METRICS,
        CAMPAIGN_METRICS,
        GUARDRAIL_METRICS,
    ):
        ordered.extend(bucket)

    seen: set[str] = set()
    definitions: list[dict] = []
    for name in ordered:
        if name in seen:
            continue
        seen.add(name)
        unit, randomization_unit, lag_days, segments, aggregation = _METRIC_META[name]
        definitions.append(
            {
                "name": name,
                "unit": unit,
                "randomization_unit": randomization_unit,
                "lag_days": lag_days,
                "segments": list(segments),
                "aggregation": aggregation,
            }
        )
    return definitions


def campbell_law_check(state_rate: float, notified_rate: float) -> dict:
    """Flag whether a metric degraded once it became a target.

    ``state_rate`` is the reported metric under the incentive and
    ``notified_rate`` is the independently notified (honest) benchmark.
    ``degradation_pp`` is the reported excess in percentage points and
    ``violates`` is True when the reported metric exceeds the honest one — the
    signature of Campbell's law (the measure was gamed once it became a target).
    """
    degradation = float(state_rate) - float(notified_rate)
    return {
        "state_rate": float(state_rate),
        "notified_rate": float(notified_rate),
        "degradation": degradation,
        "degradation_pp": degradation * 100.0,
        "violates": bool(state_rate > notified_rate),
    }


def _user_group_map(assignments: pd.DataFrame) -> pd.Series:
    """Return a ``user_id -> group`` series from segment-stage assignments."""
    work = assignments
    if "stage" in work.columns and bool((work["stage"] == STAGE_SEGMENT).any()):
        work = work.loc[work["stage"] == STAGE_SEGMENT]
    return work.drop_duplicates("user_id").set_index("user_id")["group"]


def _group_slices(sim: SimData) -> dict[str, pd.Index]:
    groups = _user_group_map(sim.assignments)
    return {str(key): idx for key, idx in groups.groupby(groups).groups.items()}


def _unsubscribe_rate(sim: SimData, users: pd.Index) -> float:
    outcomes = sim.outcomes.loc[sim.outcomes["user_id"].isin(users)]
    if outcomes.empty:
        return 0.0
    per_user = outcomes.groupby("user_id")["unsubscribed"].max()
    return float(per_user.mean()) if len(per_user) else 0.0


def _notifications_per_user_per_day(sim: SimData, users: pd.Index) -> float:
    if len(users) == 0:
        return 0.0
    exposures = sim.exposures.loc[sim.exposures["user_id"].isin(users)]
    if "exposed" in exposures.columns:
        exposures = exposures.loc[exposures["exposed"].astype(bool)]
    n_days = max(int(sim.config.n_days), 1)
    return float(len(exposures)) / (float(len(users)) * n_days)


def evaluate_guardrails(sim: SimData) -> pd.DataFrame:
    """Test-vs-control guardrail deltas with a ``breached`` flag.

    Guardrails: ``unsubscribe_rate`` must not rise in the treated arm (breach on
    any positive delta), and ``notifications_per_user_per_day`` must stay inside
    the balancer's ``daily_cap`` (breach when the treated value exceeds it).
    """
    slices = _group_slices(sim)
    test_users = slices.get("test", pd.Index([]))
    control_users = slices.get("control", pd.Index([]))
    cap = float(sim.config.balancer.daily_cap)

    unsub_test = _unsubscribe_rate(sim, test_users)
    unsub_control = _unsubscribe_rate(sim, control_users)
    notify_test = _notifications_per_user_per_day(sim, test_users)
    notify_control = _notifications_per_user_per_day(sim, control_users)

    rows = [
        {
            "metric": "unsubscribe_rate",
            "test_value": unsub_test,
            "control_value": unsub_control,
            "delta": unsub_test - unsub_control,
            "threshold": 0.0,
            "breached": bool(unsub_test - unsub_control > 0.0),
        },
        {
            "metric": "notifications_per_user_per_day",
            "test_value": notify_test,
            "control_value": notify_control,
            "delta": notify_test - notify_control,
            "threshold": cap,
            "breached": bool(notify_test > cap),
        },
    ]

    return pd.DataFrame(
        rows,
        columns=["metric", "test_value", "control_value", "delta", "threshold", "breached"],
    )
