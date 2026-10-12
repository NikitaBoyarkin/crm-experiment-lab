# CONTRACT.md — frozen interfaces for the CRM Experiment Lab

This file is the interface contract for the fan-out. Do **not** change
`src/crm_lab/config.py`, `contract.py`, `stats.py`, `plotting.py` — they are
frozen. Implement the module bodies against the signatures below.

All modules import the shared pieces like this:

```python
from crm_lab.config import SimulationConfig, BalancerConfig
from crm_lab.contract import SimData, NetEffectResult, ExposureComparison, RandomizationReport, STAGE_SEGMENT, STAGE_EXPOSURE
from crm_lab import stats
```

## Table schemas (from `contract.py`)

- `users`: `user_id, registered_day, segment, geo, prime_hour, baseline_activity`
- `campaigns`: `campaign_id, name, segment, priority, true_lift_pp, true_unsub_pp`
- `assignments`: `user_id, campaign_id, day, group, stage` — `group ∈ {test, control, gcg}`, `stage ∈ {segment, exposure}`
- `exposures`: `user_id, campaign_id, day, exposed, send_hour`
- `outcomes`: `user_id, day, purchased, purchase_amount, unsubscribed, sessions`

`SimData` bundles all five + `config`; `SimData.validate()` checks columns.

## Module interfaces (implement exactly these names)

### `src/crm_lab/simulator.py`
```python
def simulate(config: SimulationConfig | None = None) -> SimData: ...
def default_config() -> SimulationConfig: ...
```
DGP rules (must hold so tests pass):
- Users: `segment` split ~50/50 buyer/seller; `baseline_activity ~ Normal(0,1)` latent.
- Each day, `n_campaigns_per_day` campaigns targeting the user's segment are offered.
- Assignment at segment stage: every offered (user, campaign, day) gets `group` from a **fixed per-user** split controlled by `treated_share`; `gcg_share` of users are `gcg` for all campaigns. `stage == "segment"`.
- Balancer picks the ≤`daily_cap` highest-priority offers per user/day; only those become sent rows in `exposures` with `send_hour` from `balancer.hours`; `exposed` is True for `test` sends, and control/gcg get no send (exposed False when they do appear — see note).
- Outcomes: daily `purchased` ~ Bernoulli(base + lift·1[group==test & exposed]); `true_lift_pp` from the campaign scales the lift; `unsubscribed` ~ Bernoulli(base + unsub_risk·1[test & exposed]); `purchase_amount` positive when purchased; `sessions` Poisson-ish proxy.
- Determinism: same `config.seed` → identical frames.

### `src/crm_lab/balancer.py`
```python
def balance_day(offers: pd.DataFrame, config: BalancerConfig, rng) -> pd.DataFrame: ...
def daily_cap_violations(exposures: pd.DataFrame, cap: int) -> pd.DataFrame: ...
```
`balance_day` input columns: `user_id, campaign_id, priority, day`. Returns rows
for selected sends with `user_id, campaign_id, day, send_hour`; selects at most
`cap` per user/day ranked by `relevancy_weight * priority`, assigns distinct
hours from `config.hours`. `daily_cap_violations` returns users×days exceeding `cap`.

### `src/crm_lab/randomization.py`
```python
def assign_segment_stage(sim: SimData) -> pd.DataFrame: ...
def assign_exposure_stage(sim: SimData) -> pd.DataFrame: ...
def compare_randomization(sim: SimData) -> RandomizationReport: ...
```
- `assign_segment_stage`: group fixed per user (CRM-correct) → balanced test/control, SRM p large.
- `assign_exposure_stage`: group assigned only among **exposed** rows (product-style) → test share inflates because test users get treated more, SRM fires.
- `compare_randomization`: bucket counts → `stats.srm_chi2`; fill `RandomizationReport`.

### `src/crm_lab/gcg.py`
```python
def refresh_gcg(user_ids, current_gcg: set, refresh_share: float, rng) -> set: ...
def estimate_cumulative_incrementality(sim: SimData) -> pd.DataFrame: ...
def compare_refresh_strategies(sim: SimData) -> dict: ...
def gcg_day_mask(sim: SimData) -> pd.DataFrame: ...
```
- `estimate_cumulative_incrementality` → per-day frame `day, uplift, ci_low, ci_high` = test-vs-gcg difference in cumulative purchase rate.
- `compare_refresh_strategies` → `{"full_reshuffle": float, "partial": float}` CV-of-uplift proxy for the two strategies described in the Avito article.

### `src/crm_lab/exposure.py`
```python
def apply_window(outcomes: pd.DataFrame, exposures: pd.DataFrame, window_days: int) -> pd.DataFrame: ...
def compare_any_vs_true(sim: SimData) -> ExposureComparison: ...
```
- `apply_window`: keep only outcome rows within `window_days` of a user's first exposure day (`Is Exposed = True`).
- `compare_any_vs_true`: effect + SE under windowed vs all-days metric; windowed should have lower noise (smaller SE relative to effect). Fill `ExposureComparison`.

### `src/crm_lab/positivity.py`
```python
def net_effect(sim: SimData, group_tested: str = "test") -> NetEffectResult: ...
def balance_table(sim: SimData) -> pd.DataFrame: ...
def unsubscribe_financials(sim: SimData, arpu: float = 500.0) -> pd.DataFrame: ...
```
- `net_effect`: per 1,000 treated users — incremental purchasers (test−control) minus incremental unsubscribes (cost). Fill `NetEffectResult`.
- `balance_table`: per campaign, positivity/negativity/net/sign.
- `unsubscribe_financials`: convert unsubscribes to lost revenue at `arpu`.

### `src/crm_lab/metrics.py`
```python
def metric_hierarchy() -> dict[str, list[str]]: ...
def atomic_metric_defs() -> list[dict]: ...
def campbell_law_check(state_rate: float, notified_rate: float) -> dict: ...
def evaluate_guardrails(sim: SimData) -> pd.DataFrame: ...
```
- `metric_hierarchy`: `{oec: [...], driver: [...], proxy: [...], debug: [...], guardrail: [...]}` built from `contract` constants.
- `atomic_metric_defs`: one dict per metric with keys `name, unit, randomization_unit, lag_days, segments, aggregation`.
- `evaluate_guardrails`: test-vs-control guardrail deltas with a breached flag.

### `src/crm_lab/report.py`
```python
def build_report(sim: SimData) -> str: ...
def write_report(sim: SimData, path=None) -> Path: ...
```
Markdown report assembling all analyses; `write_report` defaults to `outputs/report.md`.

### `scripts/run_full_pipeline.py`
End-to-end: `simulate` → all analyses → write `outputs/report.md` + three figures `plots/pipeline_*.png`. Prints a headline table.

### `scripts/make_figures.py`
One function per module chart → `plots/`. Uses `crm_lab.plotting`.

## Tests
`tests/` calibrates against the DGP: SRM detector fires on a corrupted split,
balancer never exceeds the cap, GCG uplift CI covers the planted effect,
windowed SE ≤ all-days SE, net effect sign matches the planted lift, MDE/N
closed forms match `statsmodels`. Every test must run on synthetic data only.
