"""Synthetic data-generating process for the CRM experiment lab.

The simulator plant a known CRM-correct experiment: randomisation happens at
*segment-formation* time (fixed per user), a global control group (GCG) never
receives marketing, the balancer caps daily sends, and outcomes respond only to
campaigns that were actually sent to ``test`` users.

Everything is driven by one seeded ``numpy`` generator, so a given
``SimulationConfig.seed`` reproduces byte-identical tables.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from crm_lab.balancer import balance_day
from crm_lab.config import SimulationConfig
from crm_lab.contract import (
    CAMPAIGNS_COLUMNS,
    OUTCOMES_COLUMNS,
    STAGE_SEGMENT,
    USERS_COLUMNS,
    SimData,
)

__all__ = ["default_config", "simulate"]

# Sessions: base Poisson rate plus a small per-user latent from activity.
_SESSION_BASE_RATE = 1.5
_SESSION_ACTIVITY_SLOPE = 0.3
_PURCHASE_AMOUNT_MEAN = 6.0
_PURCHASE_AMOUNT_SIGMA = 0.8


def default_config() -> SimulationConfig:
    """Return the default simulation configuration."""
    return SimulationConfig()


# ---------------------------------------------------------------------------
# Users and campaigns
# ---------------------------------------------------------------------------


def _make_users(config: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = config.n_users
    segments = np.asarray(config.segments)
    users = pd.DataFrame(
        {
            "user_id": np.arange(n, dtype=np.int64),
            "registered_day": rng.integers(0, config.n_days, size=n).astype(np.int64),
            "segment": rng.choice(segments, size=n, p=np.full(len(segments), 1 / len(segments))),
            "geo": rng.choice(np.asarray(config.geos), size=n),
            "prime_hour": rng.choice(np.asarray(config.prime_hours), size=n).astype(np.int64),
            "baseline_activity": rng.normal(0.0, 1.0, size=n),
        }
    )
    return users[list(USERS_COLUMNS)]


def _make_campaigns(config: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = config.n_campaigns
    segments = np.asarray(config.segments)
    campaign_id = np.arange(n, dtype=np.int64)
    segment = segments[campaign_id % len(segments)]
    campaigns = pd.DataFrame(
        {
            "campaign_id": campaign_id,
            "name": [f"Camp {i:02d} · {segment[i]}" for i in range(n)],
            "segment": segment,
            "priority": rng.uniform(0.05, 1.0, size=n),
            "true_lift_pp": rng.uniform(0.003, 0.02, size=n),
            "true_unsub_pp": rng.uniform(0.0005, 0.004, size=n),
        }
    )
    return campaigns[list(CAMPAIGNS_COLUMNS)]


# ---------------------------------------------------------------------------
# Assignment (segment stage) and offers
# ---------------------------------------------------------------------------


def _assign_groups(config: SimulationConfig, n_users: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed per-user group: gcg cohort first, then a test/control split."""
    is_gcg = rng.random(n_users) < config.gcg_share
    is_test = (~is_gcg) & (rng.random(n_users) < config.treated_share)
    return np.where(is_gcg, "gcg", np.where(is_test, "test", "control"))


def _make_assignments(
    config: SimulationConfig,
    users: pd.DataFrame,
    campaigns: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """Offer ``n_campaigns_per_day`` segment-matched campaigns per user-day.

    Returns the segment-stage ``assignments``, the pre-balancer ``offers`` and
    the per-user group vector.
    """
    n_users = config.n_users
    n_days = config.n_days
    user_seg = users["segment"].to_numpy()
    campaign_seg = campaigns["segment"].to_numpy()
    campaign_id = campaigns["campaign_id"].to_numpy(dtype=np.int64)
    priority = campaigns["priority"].to_numpy(dtype=float)

    pools = {s: campaign_id[campaign_seg == s] for s in config.segments}
    sizes = [len(p) for p in pools.values()]
    k = min([config.n_campaigns_per_day, *sizes]) if sizes else 0

    if k <= 0:  # pragma: no cover - defensive; default config never hits this
        group = _assign_groups(config, n_users, rng)
        empty_int = np.empty(0, dtype=np.int64)
        assignments = pd.DataFrame(
            {
                "user_id": empty_int,
                "campaign_id": empty_int,
                "day": empty_int,
                "group": pd.Categorical(np.empty(0, dtype=object)),
                "stage": pd.Categorical(np.empty(0, dtype=object)),
            }
        )
        offers = pd.DataFrame(
            {
                "user_id": empty_int,
                "campaign_id": empty_int,
                "priority": np.empty(0, dtype=float),
                "day": empty_int,
            }
        )
        return assignments, offers, group

    chosen = np.empty((n_days, n_users, k), dtype=np.int64)
    for d in range(n_days):
        for s, pool in pools.items():
            if len(pool) == 0:
                continue
            uidx = np.where(user_seg == s)[0]
            if len(uidx) == 0:
                continue
            if len(pool) == k:
                pick = np.broadcast_to(np.arange(len(pool)), (len(uidx), k))
            else:
                keys = rng.random((len(uidx), len(pool)))
                pick = np.argpartition(keys, k - 1, axis=1)[:, :k]
            chosen[d, uidx, :] = pool[pick]

    # Flatten (day, user, slot) in C order to a single offer stream.
    cid = chosen.reshape(-1)
    uid = np.tile(np.repeat(np.arange(n_users, dtype=np.int64), k), n_days)
    day = np.repeat(np.arange(n_days, dtype=np.int64), n_users * k)

    group = _assign_groups(config, n_users, rng)

    assignments = pd.DataFrame(
        {
            "user_id": uid,
            "campaign_id": cid,
            "day": day,
            "group": pd.Categorical(group[uid]),
            "stage": STAGE_SEGMENT,
        }
    )
    offers = pd.DataFrame(
        {
            "user_id": uid,
            "campaign_id": cid,
            "priority": priority[cid],
            "day": day,
        }
    )
    return assignments, offers, group


def _make_exposures(
    config: SimulationConfig,
    offers: pd.DataFrame,
    group: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Run the balancer and flag real notifications.

    Every group goes through the balancer (contract): the selected slots become
    ``exposures`` rows, and ``exposed`` is True only for ``test`` users, so the
    CRM-style holdout (control / gcg) never receives a notification.
    """
    sent = balance_day(offers, config.balancer, rng)
    group_map = pd.DataFrame({"user_id": np.arange(config.n_users, dtype=np.int64), "group": group})
    sent = sent.merge(group_map, on="user_id", how="left")
    sent["exposed"] = sent["group"].to_numpy() == "test"
    return sent[["user_id", "campaign_id", "day", "exposed", "send_hour"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------


def _make_outcomes(
    config: SimulationConfig,
    users: pd.DataFrame,
    campaigns: pd.DataFrame,
    exposures: pd.DataFrame,
    group: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One row per (user, day) for *all* users, including control and gcg."""
    n_users = config.n_users
    n_days = config.n_days

    # Per (user, day) planted lift / unsub only from genuinely exposed sends.
    campaign_effects = campaigns[["campaign_id", "true_lift_pp", "true_unsub_pp"]]
    exposed_sends = exposures.loc[exposures["exposed"]].merge(
        campaign_effects, on="campaign_id", how="left"
    )
    lift = (
        exposed_sends.groupby(["user_id", "day"], as_index=False)[["true_lift_pp"]]
        .sum()
        .rename(columns={"true_lift_pp": "lift"})
    )
    unsub = (
        exposed_sends.groupby(["user_id", "day"], as_index=False)[["true_unsub_pp"]]
        .sum()
        .rename(columns={"true_unsub_pp": "unsub"})
    )

    uid = np.repeat(np.arange(n_users, dtype=np.int64), n_days)
    day = np.tile(np.arange(n_days, dtype=np.int64), n_users)
    outcomes = pd.DataFrame({"user_id": uid, "day": day})
    outcomes = outcomes.merge(lift, on=["user_id", "day"], how="left")
    outcomes = outcomes.merge(unsub, on=["user_id", "day"], how="left")
    outcomes[["lift", "unsub"]] = outcomes[["lift", "unsub"]].fillna(0.0)

    lift_arr = outcomes["lift"].to_numpy()
    unsub_arr = outcomes["unsub"].to_numpy()

    p_purchase = np.clip(config.base_purchase_rate + lift_arr, 0.0, 1.0)
    purchased = rng.random(len(outcomes)) < p_purchase

    p_unsub = np.clip(config.base_unsub_rate + unsub_arr, 0.0, 1.0)
    unsubscribed = rng.random(len(outcomes)) < p_unsub

    purchase_amount = np.zeros(len(outcomes), dtype=float)
    n_purchased = int(purchased.sum())
    if n_purchased:
        purchase_amount[purchased] = rng.lognormal(
            mean=_PURCHASE_AMOUNT_MEAN, sigma=_PURCHASE_AMOUNT_SIGMA, size=n_purchased
        )

    baseline = users["baseline_activity"].to_numpy()[uid]
    lam = np.clip(_SESSION_BASE_RATE + _SESSION_ACTIVITY_SLOPE * baseline, 0.05, None)
    sessions = rng.poisson(lam)

    # Test users get one extra session on days they were actually exposed —
    # the asymmetry that later makes product-style exposure randomisation
    # inflate the test share.
    is_test = group[uid] == "test"
    exposed_day = lift_arr > 0
    sessions = sessions + (is_test & exposed_day).astype(np.int64)

    result = pd.DataFrame(
        {
            "user_id": uid,
            "day": day,
            "purchased": purchased,
            "purchase_amount": purchase_amount,
            "unsubscribed": unsubscribed,
            "sessions": sessions,
        }
    )
    return result[list(OUTCOMES_COLUMNS)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def simulate(config: SimulationConfig | None = None) -> SimData:
    """Generate one synthetic CRM experiment from ``config``.

    The same ``config`` (same ``seed``) always yields identical tables.
    """
    config = config or default_config()
    rng = np.random.default_rng(config.seed)

    users = _make_users(config, rng)
    campaigns = _make_campaigns(config, rng)
    assignments, offers, group = _make_assignments(config, users, campaigns, rng)
    exposures = _make_exposures(config, offers, group, rng)
    outcomes = _make_outcomes(config, users, campaigns, exposures, group, rng)

    sim = SimData(
        users=users,
        campaigns=campaigns,
        assignments=assignments,
        exposures=exposures,
        outcomes=outcomes,
        config=config,
    )
    sim.validate()
    return sim
