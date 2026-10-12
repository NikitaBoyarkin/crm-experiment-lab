"""Net-of-unsubscribe campaign value and the positivity/negativity balance.

A CRM campaign only pays off if the incremental purchasers it drives outnumber
the incremental unsubscribes it causes. These functions price both sides of that
trade in the same unit — incremental events per 1,000 users in an arm — so the
sign of the net effect is directly interpretable.

The group assignment is read from ``SimData.assignments``. Per the data-generating
process the group is fixed per user at segment-formation time, so the mapping
``user_id -> group`` is well defined for the whole experiment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crm_lab.contract import STAGE_SEGMENT, NetEffectResult, SimData

_CONTROL_GROUP = "control"
_PER_THOUSAND = 1000.0


def _user_group_map(assignments: pd.DataFrame) -> pd.Series:
    """Return a ``user_id -> group`` series from segment-stage assignments."""
    work = assignments
    if "stage" in work.columns and bool((work["stage"] == STAGE_SEGMENT).any()):
        work = work.loc[work["stage"] == STAGE_SEGMENT]
    return work.drop_duplicates("user_id").set_index("user_id")["group"]


def _user_outcome_indicators(outcomes: pd.DataFrame) -> pd.DataFrame:
    """Collapse daily outcome rows to one row per user.

    A user counts as a *purchaser* / *unsubscriber* if the event happened on any
    of their days, so the resulting rates are per-user probabilities.
    """
    grouped = outcomes.groupby("user_id").agg(
        purchased=("purchased", "max"),
        unsubscribed=("unsubscribed", "max"),
    )
    return grouped.astype(float)


def _rate(frame: pd.DataFrame, column: str) -> float:
    """Mean of ``column`` over ``frame``; 0.0 for an empty frame."""
    if len(frame) == 0:
        return 0.0
    return float(frame[column].mean())


def net_effect(sim: SimData, group_tested: str = "test") -> NetEffectResult:
    """Net campaign effect per 1,000 treated users, net of unsubscribe cost.

    ``uplift_purchasers`` is ``(test purchase rate - local control purchase
    rate) * 1000``; ``unsubscribe_penalty`` is the same contrast on the
    unsubscribe rate. The penalty is a *cost* and is non-negative in healthy
    data (campaigns create unsubscribes, not remove them), so
    ``net_effect = uplift_purchasers - unsubscribe_penalty``.
    ``balance_ratio = uplift_purchasers / unsubscribe_penalty`` and is
    ``inf`` / ``nan`` when the penalty is zero (guarded division).
    """
    groups = _user_group_map(sim.assignments)
    per_user = _user_outcome_indicators(sim.outcomes).join(groups, how="inner")

    tested = per_user.loc[per_user["group"] == group_tested]
    control = per_user.loc[per_user["group"] == _CONTROL_GROUP]

    uplift = (_rate(tested, "purchased") - _rate(control, "purchased")) * _PER_THOUSAND
    penalty = (_rate(tested, "unsubscribed") - _rate(control, "unsubscribed")) * _PER_THOUSAND
    net = uplift - penalty

    if penalty == 0:
        balance_ratio = float("inf") if uplift > 0 else (np.nan if uplift == 0 else float("-inf"))
    else:
        balance_ratio = uplift / penalty

    return NetEffectResult(
        uplift_purchasers=float(uplift),
        unsubscribe_penalty=float(penalty),
        net_effect=float(net),
        balance_ratio=float(balance_ratio),
    )


def balance_table(sim: SimData) -> pd.DataFrame:
    """Per-campaign positivity / negativity / net and a directional ``sign``.

    For each campaign the treated arm is ``test`` and the reference is the local
    ``control`` arm. Positivity is incremental purchasers per 1,000 users,
    negativity is incremental unsubscribes per 1,000 users, and ``sign`` is
    ``"positive"`` / ``"negative"`` / ``"neutral"`` from the net.
    """
    per_user = _user_outcome_indicators(sim.outcomes)
    assignments = sim.assignments
    if "stage" in assignments.columns and bool((assignments["stage"] == STAGE_SEGMENT).any()):
        assignments = assignments.loc[assignments["stage"] == STAGE_SEGMENT]

    rows: list[dict] = []
    for campaign_id, frame in assignments.groupby("campaign_id"):
        user_groups = frame.drop_duplicates("user_id").set_index("user_id")["group"]
        joined = per_user.reindex(user_groups.index).assign(group=user_groups.to_numpy())
        tested = joined.loc[joined["group"] == "test"]
        control = joined.loc[joined["group"] == _CONTROL_GROUP]
        positivity = (_rate(tested, "purchased") - _rate(control, "purchased")) * _PER_THOUSAND
        negativity = (
            _rate(tested, "unsubscribed") - _rate(control, "unsubscribed")
        ) * _PER_THOUSAND
        net = positivity - negativity
        if net > 0:
            sign = "positive"
        elif net < 0:
            sign = "negative"
        else:
            sign = "neutral"
        rows.append(
            {
                "campaign_id": campaign_id,
                "positivity": float(positivity),
                "negativity": float(negativity),
                "net": float(net),
                "sign": sign,
            }
        )

    return pd.DataFrame(rows, columns=["campaign_id", "positivity", "negativity", "net", "sign"])


def unsubscribe_financials(sim: SimData, arpu: float = 500.0) -> pd.DataFrame:
    """Price the monthly unsubscribes of each group in lost revenue.

    ``monthly_unsubscribes`` is the number of users in the group who unsubscribed
    at least once over the simulated horizon (treated as one month). Each such
    churn is valued at one month of ``arpu``, so
    ``lost_revenue = monthly_unsubscribes * arpu``. ``arpu`` is the monthly
    revenue per user (default 500.0).
    """
    groups = _user_group_map(sim.assignments)
    per_user = _user_outcome_indicators(sim.outcomes).join(groups, how="inner")

    rows: list[dict] = []
    for group, frame in per_user.groupby("group"):
        n_users = int(len(frame))
        monthly_unsubscribes = int(frame["unsubscribed"].sum())
        unsub_rate = float(frame["unsubscribed"].mean()) if n_users else 0.0
        rows.append(
            {
                "group": group,
                "n_users": n_users,
                "monthly_unsubscribes": monthly_unsubscribes,
                "unsub_rate": unsub_rate,
                "arpu": float(arpu),
                "lost_revenue": float(monthly_unsubscribes * arpu),
            }
        )

    return pd.DataFrame(
        rows,
        columns=["group", "n_users", "monthly_unsubscribes", "unsub_rate", "arpu", "lost_revenue"],
    ).sort_values("group", ignore_index=True)
