#!/usr/bin/env python
"""Phase 1-5 EDA on the committed fixture ``data/experiment_sample.csv``.

Read-only: loads the fixture and prints every number that lands in the report.
Grain = one row per (user_id, day). Metric = ``purchased`` (rate) /
``purchase_amount`` (skewed sum). Segments = ``day``. Time = ``day``.

Run:
    <shared-venv>/bin/python reports/eda/scripts/01_fixture_eda.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
FIXTURE = PROJECT / "data" / "experiment_sample.csv"


def _iqr_outliers(s: pd.Series) -> float:
    s = s.dropna()
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())


def main() -> None:
    df = pd.read_csv(FIXTURE)
    out: dict[str, object] = {}

    # --- Phase 1: structure ------------------------------------------------
    out["shape"] = list(df.shape)
    out["columns"] = list(df.columns)
    out["dtypes"] = {c: str(t) for c, t in df.dtypes.items()}
    out["n_users"] = int(df["user_id"].nunique())
    out["n_days"] = int(df["day"].nunique())
    out["day_range"] = [int(df["day"].min()), int(df["day"].max())]
    grid = out["n_users"] * out["n_days"]
    out["expected_grid"] = grid
    out["grid_complete"] = bool(len(df) == grid)
    out["null_share"] = {c: float(df[c].isna().mean()) for c in df.columns}
    out["full_dup_rows"] = int(df.duplicated().sum())
    out["dup_key_user_day"] = int(df.duplicated(["user_id", "day"]).sum())
    out["constant_cols"] = [c for c in df.columns if df[c].nunique(dropna=False) <= 1]

    # --- Phase 2: validity -------------------------------------------------
    out["purchased_true"] = int(df["purchased"].sum())
    out["purchased_rate"] = float(df["purchased"].mean())
    out["unsubscribed_true"] = int(df["unsubscribed"].sum())
    out["purchase_amount_min"] = float(df["purchase_amount"].min())
    out["purchase_amount_nonzero"] = int((df["purchase_amount"] > 0).sum())
    out["sessions_min"] = int(df["sessions"].min())
    out["sessions_max"] = int(df["sessions"].max())

    # --- Phase 3: univariate + bivariate -----------------------------------
    num = ["user_id", "day", "purchase_amount", "sessions"]
    out["numeric_stats"] = {
        c: {
            "mean": float(df[c].mean()),
            "median": float(df[c].median()),
            "std": float(df[c].std()),
            "p95": float(df[c].quantile(0.95)),
            "skew": float(df[c].skew()),
            "iqr_outlier_share": _iqr_outliers(df[c]),
        }
        for c in num
    }
    corr = df[num].corr()
    pairs = []
    for i, a in enumerate(num):
        for b in num[i + 1 :]:
            r = float(corr.loc[a, b])
            if abs(r) >= 0.5:
                pairs.append({"a": a, "b": b, "r": round(r, 3)})
    out["corr_pairs_ge_0_5"] = pairs

    # purchase amount conditional on a purchase (the skew driver)
    paid = df.loc[df["purchased"], "purchase_amount"]
    out["paid_amount_stats"] = {
        "n": int(len(paid)),
        "min": float(paid.min()),
        "median": float(paid.median()),
        "mean": float(paid.mean()),
        "max": float(paid.max()),
    }

    # --- Phase 4: metric by segment (day) ----------------------------------
    seg = df.groupby("day").agg(count=("purchased", "size"), rate=("purchased", "mean"))
    out["segment_by_day"] = {
        "overall_rate": float(df["purchased"].mean()),
        "min_rate": float(seg["rate"].min()),
        "max_rate": float(seg["rate"].max()),
        "days_with_purchase": int((seg["rate"] > 0).sum()),
        "users_per_day_constant": bool((seg["count"] == df["user_id"].nunique()).all()),
    }

    # composition check: every day holds the same 15 users -> no composition shift
    users_per_day = df.groupby("day")["user_id"].nunique()
    out["users_per_day_min"] = int(users_per_day.min())
    out["users_per_day_max"] = int(users_per_day.max())

    # --- lead-candidate: does session activity track purchases? ------------
    df["session_bucket"] = pd.cut(
        df["sessions"], bins=[-1, 0, 1, 2, 3, 10], labels=["0", "1", "2", "3", "4+"]
    )
    by_sess = df.groupby("session_bucket", observed=True).agg(
        n=("purchased", "size"), rate=("purchased", "mean")
    )
    out["purchase_rate_by_sessions"] = {
        str(k): {"n": int(v["n"]), "rate": round(float(v["rate"]), 4)}
        for k, v in by_sess.iterrows()
    }
    # point-biserial correlation sessions vs purchased
    out["corr_sessions_purchased"] = float(
        np.corrcoef(df["sessions"].astype(float), df["purchased"].astype(int))[0, 1]
    )

    # --- Phase 5: trend (20 days -> weekly) --------------------------------
    df["week"] = df["day"] // 7
    wk = df.groupby("week").agg(
        rows=("purchased", "size"),
        purchases=("purchased", "sum"),
        rate=("purchased", "mean"),
        sessions=("sessions", "mean"),
    )
    out["weekly"] = {
        int(w): {
            "rows": int(r["rows"]),
            "purchases": int(r["purchases"]),
            "rate": round(float(r["rate"]), 4),
            "sessions_per_user_day": round(float(r["sessions"]), 3),
        }
        for w, r in wk.iterrows()
    }
    wk["rate_pct_change"] = wk["rate"].pct_change()
    out["weekly_rate_delta"] = {
        int(w): (None if pd.isna(v) else round(float(v), 3))
        for w, v in wk["rate_pct_change"].items()
    }
    # note: week 2 = days 14..19 (6 days), week 0/1 = 7 days -> partial period
    out["week_day_counts"] = {
        int(w): int(n) for w, n in df.groupby("week")["day"].nunique().items()
    }

    # --- fixture regenerability (truth-check caveat) -----------------------
    out["fixture_regenerable_note"] = (
        "Searched simulator seeds 0..1999 for n_users=15, n_days=20; "
        "no full match on (purchased, sessions). Fixture is a schema example, "
        "not a committed simulator artifact."
    )

    (PROJECT / "reports" / "eda" / "fixture_results.json").write_text(
        json.dumps(out, indent=2, default=str)
    )
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
