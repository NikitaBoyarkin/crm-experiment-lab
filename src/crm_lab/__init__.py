"""CRM Experiment Lab — a synthetic CRM-communication experimentation platform.

The lab reproduces the *CRM-specific* layer of A/B testing that generic product
experimentation tooling misses: randomization at segment-formation time, the
global control group, the send-time balancer with a per-user daily cap, and the
net effect of campaigns once unsubscribes are subtracted from the uplift.

Everything runs on synthetic data with a known data-generating process, so every
claim in the README and the test suite is reproducible.
"""

from crm_lab.config import BalancerConfig, SimulationConfig
from crm_lab.contract import SimData

__all__ = ["SimulationConfig", "BalancerConfig", "SimData"]
