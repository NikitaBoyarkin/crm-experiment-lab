"""Send balancer: decides which offered campaigns actually go out.

The balancer is the CRM-specific piece that generic product experiment tooling
ignores. A user may be *offered* many campaigns on a given day, but the platform
can only send a small number of marketing notifications without burning the user
out. The balancer enforces a per-user daily cap, ranks the candidates by
relevancy-weighted campaign priority, and spreads the winners across prime-time
send hours.

Both functions are pure and deterministic given the incoming ``rng``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crm_lab.config import BalancerConfig

__all__ = ["balance_day", "daily_cap_violations"]

_SENT_COLUMNS = ("user_id", "campaign_id", "day", "send_hour")
_VIOLATION_COLUMNS = ("user_id", "day", "n_sends")


def _empty(columns: tuple[str, ...]) -> pd.DataFrame:
    """Return an empty frame with integer dtypes for every column."""
    return pd.DataFrame({c: np.empty(0, dtype=np.int64) for c in columns})


def balance_day(offers: pd.DataFrame, config: BalancerConfig, rng) -> pd.DataFrame:
    """Select the sends for one (or many) days.

    Parameters
    ----------
    offers:
        Candidate rows with columns ``user_id, campaign_id, priority, day``.
        This is the pre-balancer offer stream: every campaign the platform
        *considered* sending to every user on every day.
    config:
        Cap, send hours and relevancy weight.
    rng:
        A ``numpy.random.Generator``. Used only to break priority ties so the
        result is reproducible for a given seed.

    Returns
    -------
    DataFrame with ``user_id, campaign_id, day, send_hour`` — one row per send
    that survived the cap, with a distinct prime-time hour per slot.
    """
    if len(offers) == 0:
        return _empty(_SENT_COLUMNS)

    user_id = offers["user_id"].to_numpy(dtype=np.int64)
    day = offers["day"].to_numpy(dtype=np.int64)
    campaign_id = offers["campaign_id"].to_numpy(dtype=np.int64)
    priority = offers["priority"].to_numpy(dtype=float)

    # Tiny tie-break jitter: keeps the ranking a pure priority sort while making
    # equal-priority offers reproducibly ordered by the seeded rng.
    score = config.relevancy_weight * priority + rng.random(len(offers)) * 1e-9

    # Composite group key (user, day) so ranking is a single vectorised lexsort
    # instead of a costly pandas groupby on millions of rows.
    span = int(day.max()) + 1
    key = user_id * span + day
    order = np.lexsort((-score, key))

    sorted_key = key[order]
    change = np.empty(len(order), dtype=bool)
    change[0] = True
    change[1:] = sorted_key[1:] != sorted_key[:-1]
    starts = np.maximum.accumulate(np.where(change, np.arange(len(order)), 0))
    rank = np.arange(len(order), dtype=np.int64) - starts

    # A user can only receive as many distinct hours as the prime-time grid has.
    n_slots = min(config.daily_cap, len(config.hours))
    keep = rank < n_slots
    sel = order[keep]
    sel_rank = rank[keep]

    hours = np.asarray(config.hours, dtype=np.int64)
    send_hour = hours[sel_rank] if len(hours) else np.empty(0, dtype=np.int64)

    return pd.DataFrame(
        {
            "user_id": user_id[sel],
            "campaign_id": campaign_id[sel],
            "day": day[sel],
            "send_hour": send_hour,
        }
    ).reset_index(drop=True)


def daily_cap_violations(exposures: pd.DataFrame, cap: int) -> pd.DataFrame:
    """Return the ``(user_id, day)`` cells where more than ``cap`` sends landed.

    An empty frame means the cap held everywhere.
    """
    if len(exposures) == 0:
        return _empty(_VIOLATION_COLUMNS)

    counts = (
        exposures.groupby(["user_id", "day"], as_index=False)
        .size()
        .rename(columns={"size": "n_sends"})
    )
    violations = counts[counts["n_sends"] > cap]
    return violations.reset_index(drop=True)
