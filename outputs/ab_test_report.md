# A/B Test Report — CRM campaign send (default synthetic experiment)

**Test name:** Campaign send vs holdout (12-campaign CRM programme, seed 42)  
**Owner:** crm-experiment-lab  
**Dates:** 28 simulated days  
**Status:** Analysis complete

---

## Hypothesis

**Change tested:** Sending the segment-matched CRM campaigns to a `test` arm, vs a `control` arm that is offered the same campaigns through the balancer but receives no notification (daily send cap 3, no exposure).  
**Expected effect:** a higher per-user purchase rate in `test`.  
**Rationale:** the campaign carries a planted per-day purchase lift that only applies on genuinely exposed `test` user-days.

---

## Design

| Parameter | Value |
|---|---|
| Traffic split | 50% test / 50% control (plus a 10% global control group outside this readout) |
| Randomisation unit | **user** (fixed per user at segment-formation time) |
| Primary metric | per-user purchase rate — P(any purchase over the 28-day horizon) |
| Secondary / OEC | revenue per user (Σ `purchase_amount`) |
| Guardrail metrics | unsubscribe rate (per user); notifications per user per day (cap) |
| Minimum detectable effect (MDE) | 0.06765 absolute = 10% relative on a 0.6765 baseline |
| Required sample size (per group) | 751 |
| Pre-specified alpha | 0.05 |
| Pre-specified power | 80% |

---

## Results

### Sample Ratio Mismatch (SRM) check

| Group | Expected | Actual | Difference |
|---|---|---|---|
| Control | 8956 | 9,030 | +0.83% |
| Treatment | 8956 | 8,881 | -0.83% |

**SRM detected:** No (chi2=1.2395, p=0.2656) — checked against the **planned** 50% split, not 50/50.

---

### Primary metric

| Group | N | Purchase rate | 95% CI (of the rate) |
|---|---|---|---|
| Control | 9,030 | 67.6523% | ±0.9649% |
| Treatment | 8,881 | 88.9314% | ±0.6525% |

**Absolute difference:** +0.21279 (+21.279 pp)  
**Relative lift:** +31.45%  
**95% CI (of the difference, unpooled):** [+0.20114, +0.22444]  
**p-value:** <1e-300  
**Significant at alpha=0.05:** Yes  
**Achieved power for the observed effect:** 100.0%

> At 8,881 per arm the test is powered to detect 0.01967 absolute (2.9% relative). The observed effect (0.21279) is 10.82x the MDE the test was powered for.

### Secondary: revenue per user (OEC)

| Group | Mean revenue/user | 95% bootstrap CI (mean) |
|---|---|---|
| Control | 608.3699 | [592.9063, 624.7256] |
| Treatment | 1180.5793 | [1157.8328, 1202.6968] |

**Absolute difference:** +572.2094 per user  
**Welch t-test:** t=40.419, p=<1e-300 (significant at 0.05)

### Clustering check (randomisation unit vs measurement unit)

Assignment is fixed per **user**, but the exposure metric measures at the **user-day**. A row-level two-proportion test would assume those rows are independent; the correct comparison uses a cluster-robust SE clustered by user. Here the two agree, because this DGP gives purchases no within-user correlation — so the row-level exposure metric is not misleading on this data:

| Method | SE of the difference |
|---|---|
| Row-level two-proportion (naive, user-day) | 0.001947 |
| Cluster-robust (Liang-Zeger, clustered by user) | 0.001942 |

> Cluster-robust / naive = 1.00x. The check is a guard, not a finding here; the primary readout above is the user-level metric, which respects the randomisation unit by construction.

---

### Guardrail metrics

| Metric | Control | Treatment | Change | p (Holm-adj) | Threshold | Pass |
|---|---|---|---|---|---|---|
| unsubscribe_rate | 0.0537 | 0.2160 | +302.1% | <1e-300 | 0.0000 | No |
| notifications_per_user_per_day | 0.0000 | 3.0000 | 3.000 vs cap 3 | — (hard cap) | 3.0000 | Yes |

> Family of 2 guardrails, Holm-corrected. Unsubscribe rate rises in the treated arm and stays significant after correction; the send cap holds.

---

### Segment breakdowns

| Segment | Control rate | Treatment rate | Lift | Significant |
|---|---|---|---|---|
| buyer | 67.5688% | 89.8223% | +32.93% | Yes |
| seller | 67.7355% | 88.0827% | +30.04% | Yes |

---

### Ship scenario — can the guardrail ever pass?

The CRM lever is send volume: a lower daily cap means fewer notifications, so fewer unsubscribes but also a smaller lift. `unsub delta` is therefore the **churn budget a cap requires** before it may ship. Sweeping the cap and re-reading the guardrail under two definitions — **strict** (rate must not rise at all) and **budget** (may rise up to 12% absolute) — answers whether a *ship* verdict is reachable at all:

| daily cap | relative lift | unsub delta (= budget needed) | unsub p | net / 1k | verdict (strict) | verdict (12% budget) |
|---|---|---|---|---|---|---|
| 3 | +31.5% | +0.1623 | <1e-300 | +50.5 | Investigate | Investigate |
| 2 | +25.5% | +0.1155 | <1e-300 | +56.8 | Investigate | Ship |
| 1 | +15.5% | +0.0620 | <1e-300 | +42.7 | Investigate | Ship |

> Because the treated arm is **sent to** and the control arm is not, the unsubscribe delta is positive and significant at every cap — the strict guardrail (`delta > 0` blocks) can never be satisfied, at any volume, so a ship verdict is reachable **only** with an explicit churn budget. Under a 12% budget: caps 1 and 2 ship, cap 3 (needs 16.3%) does not — and **cap 2 maximises the net effect (+56.8 / 1k)**, cutting churn from 16.2% to 11.6% at a smaller lift cost than cap 1. The actionable read: set the churn budget from the ARPU value of a lost user, then **ship at the cap that maximises net within it** — not chase a strict zero that no sending campaign can meet.

---

## Decision

**Verdict:** Investigate

**Reasoning:** The primary metric moves up significantly (+31.45%, p=<1e-300, 95% CI excludes 0) and the revenue OEC agrees, so the campaign works. But the unsubscribe guardrail regresses significantly and survives Holm correction, and the net effect only stays positive because the purchase uplift (+212.8 per 1k) is just larger than the churn cost (162.3 per 1k) — a 1.31x margin. Per the decision matrix, a winning primary with a failed guardrail is **Investigate**, not Ship.

**Next steps:**
1. Add a frequency cap on repeat sends per user/month and re-measure the unsubscribe guardrail — owner: CRM team.
2. Run a follow-up with the churn cost priced at ARPU (see `positivity.unsubscribe_financials`) to convert the 1.31x margin into revenue terms before rollout — owner: analyst.
3. If the guardrail still regresses at a smaller send volume, fall back to the top-priority campaigns only (net is driven by the positive campaigns in `positivity.balance_table`).

---

*Generated by `scripts/ab_test_readout.py` — the ab-test-analysis skill applied to the default synthetic CRM experiment.*
