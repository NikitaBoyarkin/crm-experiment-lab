"""Frozen configuration objects for the CRM experiment simulator.

These are the only knobs the rest of the package reads. Keep them immutable
(frozen dataclasses) so that every experiment is reproducible from a single
config object and a seed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BalancerConfig:
    """Controls the CRM send balancer.

    The balancer decides, for each user and each day, which of the offered
    campaigns actually get sent. It enforces the daily notification cap and
    spreads sends across the prime-time hours.
    """

    daily_cap: int = 3
    """Max marketing notifications a single user may receive per day."""

    hours: tuple[int, ...] = (9, 13, 19, 21)
    """Prime-time send hours. A user gets one campaign per selected hour slot."""

    relevancy_weight: float = 1.0
    """Weight on campaign priority when ranking candidates for a user."""


@dataclass(frozen=True)
class SimulationConfig:
    """Full description of one synthetic CRM experiment."""

    n_users: int = 20_000
    n_days: int = 28
    seed: int = 42

    n_campaigns: int = 12
    """Number of distinct campaign templates in the platform."""

    n_campaigns_per_day: int = 4
    """Candidate campaigns offered to a user on a given day (pre-balancer)."""

    balancer: BalancerConfig = BalancerConfig()
    """Send-balancer settings."""

    gcg_share: float = 0.10
    """Share of users assigned to the global control group (never marketed)."""

    gcg_refresh_share: float = 0.02
    """Share of users reshuffled out of/into the GCG every 28 days."""

    treated_share: float = 0.50
    """Test share within a local campaign experiment (rest is local control)."""

    exposure_window_days: int = 7
    """Indirect-influence window for the `Is Exposed = True` metric computation.

    Days after a user's first exposure that still count toward their metric.
    Anything past this window is treated as noise and dropped.
    """

    base_purchase_rate: float = 0.04
    """Baseline daily probability a user purchases (before any campaign)."""

    base_unsub_rate: float = 0.002
    """Baseline daily probability a user unsubscribes (before any campaign)."""

    prime_hours: tuple[int, ...] = (9, 13, 19, 21)
    """Hours at which a user is considered active (used as send-time affinity)."""

    segments: tuple[str, ...] = ("buyer", "seller")
    """User segments. Campaigns target exactly one segment."""

    geos: tuple[str, ...] = ("MOW", "SPB", "NSK", "KZN", "REG")
    """Regions used for segment balance checks."""
