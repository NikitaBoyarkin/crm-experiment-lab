#!/usr/bin/env python
"""Figures for the crm-experiment-lab EDA (matplotlib OO style, light theme).

Reads the committed fixture ``data/experiment_sample.csv`` and writes 5 PNGs to
``reports/eda/figures/``. English labels, dpi=300, bbox_inches="tight".

Run:
    <shared-venv>/bin/python reports/eda/scripts/03_figures.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore")

PROJECT = Path(__file__).resolve().parents[3]
FIXTURE = PROJECT / "data" / "experiment_sample.csv"
OUT = PROJECT / "reports" / "eda" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "font.size": 11,
    }
)

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
GREY = "#999999"


def fig1_amount_distribution(df: pd.DataFrame) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    axs[0].hist(df["purchase_amount"], bins=30, color=BLUE, edgecolor="white")
    axs[0].set(
        xlabel="purchase_amount",
        ylabel="rows",
        title="purchase_amount: 94.7% of rows are 0",
    )
    paid = df.loc[df["purchased"], "purchase_amount"]
    axs[1].hist(paid, bins=16, color=ORANGE, edgecolor="white")
    axs[1].set(
        xlabel="purchase_amount (paid rows)",
        ylabel="rows",
        title=f"Non-zero amounts (n={len(paid)}): right-skewed, max {paid.max():,.0f}",
    )
    fig.savefig(OUT / "fig1_purchase_amount_distribution.png", bbox_inches="tight")
    plt.close(fig)


def fig2_rate_by_day(df: pd.DataFrame) -> None:
    seg = df.groupby("day")["purchased"].mean().sort_values(ascending=False)
    overall = df["purchased"].mean()
    fig, ax = plt.subplots(figsize=(11, 4.6), layout="constrained")
    bars = ax.bar([str(d) for d in seg.index], seg.values, color=BLUE, edgecolor="white")
    ax.axhline(
        overall, color=RED, linestyle="--", linewidth=1.4, label=f"overall rate {overall:.3f}"
    )
    for b in bars[:3]:
        ax.annotate(
            f"{b.get_height():.3f}",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
        )
    ax.set(
        xlabel="day (segment)",
        ylabel="purchase rate",
        title="Purchase rate by day: 0.000 to 0.133, 15 users every day",
    )
    ax.legend(frameon=False)
    fig.savefig(OUT / "fig2_purchase_rate_by_day.png", bbox_inches="tight")
    plt.close(fig)


def fig3_weekly_trend(df: pd.DataFrame) -> None:
    df = df.copy()
    df["week"] = df["day"] // 7
    wk = df.groupby("week").agg(
        rate=("purchased", "mean"),
        purchases=("purchased", "sum"),
        rows=("purchased", "size"),
        sessions=("sessions", "mean"),
    )
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    ax.plot(
        wk.index, wk["rate"], marker="o", color=BLUE, linewidth=2, label="purchase rate (behaviour)"
    )
    ax.plot(
        wk.index,
        wk["sessions"] / wk["sessions"].max() * wk["rate"].max(),
        marker="s",
        color=GREEN,
        linewidth=1.6,
        linestyle="--",
        label="sessions/user-day (scaled)",
    )
    ax.bar(
        wk.index,
        wk["purchases"] / wk["rows"],
        width=0.25,
        color=GREY,
        alpha=0.25,
        label="purchases per row (flow)",
    )
    for w, r in wk.iterrows():
        ax.annotate(
            f"n={int(r['rows'])}",
            (w, r["rate"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )
    ax.set(
        xlabel="week (days // 7)",
        ylabel="purchase rate",
        title="Weekly trend: 20 days = weeks 0,1 (7d) + week 2 (6d, partial)",
    )
    ax.legend(frameon=False)
    fig.savefig(OUT / "fig3_weekly_trend.png", bbox_inches="tight")
    plt.close(fig)


def fig4_sessions_vs_purchase(df: pd.DataFrame) -> None:
    bucketed = pd.cut(df["sessions"], bins=[-1, 0, 1, 2, 3, 10], labels=["0", "1", "2", "3", "4+"])
    g = df.groupby(bucketed, observed=True)["purchased"].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 4.4), layout="constrained")
    bars = ax.bar([str(i) for i in g.index], g["mean"], color=ORANGE, edgecolor="white")
    for b, (_, row) in zip(bars, g.iterrows()):
        ax.annotate(
            f"n={int(row['size'])}",
            (b.get_x() + b.get_width() / 2, b.get_height()),
            textcoords="offset points",
            xytext=(0, 4),
            ha="center",
            fontsize=9,
        )
    r = np.corrcoef(df["sessions"].astype(float), df["purchased"].astype(int))[0, 1]
    ax.set(
        xlabel="sessions bucket (per user-day)",
        ylabel="purchase rate",
        title=f"Purchase rate vs sessions: r={r:.2f}, no monotone trend (only 16 purchases)",
    )
    fig.savefig(OUT / "fig4_sessions_vs_purchase.png", bbox_inches="tight")
    plt.close(fig)


def fig5_cumulative(df: pd.DataFrame) -> None:
    daily = df.groupby("day")["purchased"].sum()
    cum = daily.cumsum()
    per_user_cum = cum / df["user_id"].nunique()
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    ax.plot(cum.index, per_user_cum.values, marker="o", color=BLUE, linewidth=2)
    ax.fill_between(cum.index, 0, per_user_cum.values, color=BLUE, alpha=0.15)
    ax.set(
        xlabel="day",
        ylabel="cumulative purchases per user",
        title="Cumulative purchases per user over the 20-day fixture",
    )
    ax.annotate(
        f"day 19: {per_user_cum.iloc[-1]:.3f}",
        (cum.index[-1], per_user_cum.iloc[-1]),
        textcoords="offset points",
        xytext=(-70, -14),
        fontsize=10,
        color=RED,
    )
    fig.savefig(OUT / "fig5_cumulative_purchases_per_user.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(FIXTURE)
    fig1_amount_distribution(df)
    fig2_rate_by_day(df)
    fig3_weekly_trend(df)
    fig4_sessions_vs_purchase(df)
    fig5_cumulative(df)
    print("wrote 5 figures to", OUT)


if __name__ == "__main__":
    main()
