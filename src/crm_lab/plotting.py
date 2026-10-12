"""Shared matplotlib style for the lab.

Headless Agg backend, a colorblind-safe Okabe-Ito palette, constrained layout,
300 dpi, and one ``save_fig`` helper. Every figure in the project goes through
here so the gallery stays visually consistent.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colorblind-safe palette
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#999999",
}

PLOTS_DIR = Path(__file__).resolve().parents[2] / "plots"


def apply_style() -> None:
    """Apply the shared rcParams. Idempotent."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.grid": True,
            "grid.color": "#DDDDDD",
            "grid.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save_fig(fig: plt.Figure, name: str, out_dir: Path | None = None) -> Path:
    """Save ``fig`` as ``<out_dir or plots>/<name>`` and close it."""
    apply_style()
    out_dir = Path(out_dir or PLOTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path)
    plt.close(fig)
    return path
