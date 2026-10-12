"""The data contract between the modules of the CRM Experiment Lab.

Every module agrees on these table shapes and metric names. If a module returns
a DataFrame, its columns are declared here. This is what lets the simulator, the
analysers, the tests and the report be written independently.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from crm_lab.config import SimulationConfig

# ---------------------------------------------------------------------------
# Table schemas
# ---------------------------------------------------------------------------

USERS_COLUMNS: tuple[str, ...] = (
    "user_id",
    "registered_day",
    "segment",
    "geo",
    "prime_hour",
    "baseline_activity",
)

CAMPAIGNS_COLUMNS: tuple[str, ...] = (
    "campaign_id",
    "name",
    "segment",
    "priority",
    "true_lift_pp",
    "true_unsub_pp",
)

# One row per (user, campaign, day) that the platform *offered* to the user.
# `group` is the assignment decided at segment-formation time (the CRM-correct
# way): "test" | "control" | "gcg".
ASSIGNMENTS_COLUMNS: tuple[str, ...] = (
    "user_id",
    "campaign_id",
    "day",
    "group",
    "stage",
)

# One row per (user, campaign, day) that survived the balancer and was sent.
EXPOSURES_COLUMNS: tuple[str, ...] = (
    "user_id",
    "campaign_id",
    "day",
    "exposed",
    "send_hour",
)

# One row per (user, day): the user's daily behaviour.
OUTCOMES_COLUMNS: tuple[str, ...] = (
    "user_id",
    "day",
    "purchased",
    "purchase_amount",
    "unsubscribed",
    "sessions",
)

# ---------------------------------------------------------------------------
# Metric taxonomy (Trisigma handbook: OEC -> driver -> proxy -> debug -> guardrail)
# ---------------------------------------------------------------------------

OEC = "revenue"
"""Overall Evaluation Criterion: the single headline success metric."""

DRIVER_METRICS: tuple[str, ...] = ("retention_d7", "orders_per_user", "aov")
PROXY_METRICS: tuple[str, ...] = ("session_rate",)
DEBUG_METRICS: tuple[str, ...] = ("send_hour_match", "exposure_rate")
CAMPAIGN_METRICS: tuple[str, ...] = (
    "open_rate",
    "ctr",
    "conversion_to_purchase",
    "unsubscribe_rate",
)
GUARDRAIL_METRICS: tuple[str, ...] = ("unsubscribe_rate", "notifications_per_user_per_day")

# Metric used for the positivity / negativity balance in `positivity.py`.
POSITIVITY_METRIC = "incremental_purchasers"
NEGATIVITY_METRIC = "unsubscribes"

# ---------------------------------------------------------------------------
# Assignment stages
# ---------------------------------------------------------------------------

STAGE_SEGMENT = "segment"
"""Users are bucketed when the campaign segment is built (CRM-correct)."""

STAGE_EXPOSURE = "exposure"
"""Users are bucketed when they hit the tested element (product-style; wrong for CRM)."""


# ---------------------------------------------------------------------------
# SimData container
# ---------------------------------------------------------------------------


@dataclass
class SimData:
    """Everything one synthetic CRM experiment produces.

    Attributes
    ----------
    users, campaigns, assignments, exposures, outcomes:
        The five tables described by the ``*_COLUMNS`` constants above.
    config:
        The config that generated this run, so downstream code stays parameterised.
    """

    users: pd.DataFrame
    campaigns: pd.DataFrame
    assignments: pd.DataFrame
    exposures: pd.DataFrame
    outcomes: pd.DataFrame
    config: SimulationConfig = field(default_factory=SimulationConfig)

    def validate(self) -> None:
        """Raise ``ValueError`` if any table is missing required columns."""
        for name, table, columns in (
            ("users", self.users, USERS_COLUMNS),
            ("campaigns", self.campaigns, CAMPAIGNS_COLUMNS),
            ("assignments", self.assignments, ASSIGNMENTS_COLUMNS),
            ("exposures", self.exposures, EXPOSURES_COLUMNS),
            ("outcomes", self.outcomes, OUTCOMES_COLUMNS),
        ):
            missing = [c for c in columns if c not in table.columns]
            if missing:
                raise ValueError(f"table '{name}' is missing columns: {missing}")


# ---------------------------------------------------------------------------
# Result containers returned by the analyser modules
# ---------------------------------------------------------------------------


@dataclass
class NetEffectResult:
    """Net effect of a campaign once unsubscribes are priced in."""

    uplift_purchasers: float
    """Incremental purchasers per 1,000 treated users (test minus control)."""

    unsubscribe_penalty: float
    """Incremental unsubscribes per 1,000 treated users, as a cost."""

    net_effect: float
    """``uplift_purchasers - unsubscribe_penalty`` in the same unit."""

    balance_ratio: float
    """Positivity gained per unit of negativity (``uplift / penalty``)."""


@dataclass
class ExposureComparison:
    """`Is Exposed = True` (windowed, symmetric) vs `Is Exposed = Any` (all days).

    The window is not a free variance reduction: dropping rows always raises a
    proportion's standard error. Its value is that it anchors both arms to the
    same short horizon and preserves the effect when the campaign's influence is
    transient (see ``exposure.window_sensitivity_demo``).
    """

    any_effect: float
    any_variance_signal: float
    true_effect: float
    true_variance_signal: float
    any_std_error: float
    true_std_error: float


@dataclass
class RandomizationReport:
    """Result of comparing segment-stage vs exposure-stage randomization."""

    segment_srm_pvalue: float
    exposure_srm_pvalue: float
    segment_test_share: float
    exposure_test_share: float
    note: str
