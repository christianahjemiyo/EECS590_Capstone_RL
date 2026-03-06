from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt


V2_PALETTE = {
    "DP_PolicyIter": "#264653",
    "DP_ValueIter": "#2a9d8f",
    "Policy Iter": "#264653",
    "Value Iter": "#2a9d8f",
    "Q_Learning": "#e76f51",
    "Q-Learning": "#e76f51",
    "Double_Q": "#6a4c93",
    "Double Q-Learning": "#6a4c93",
    "Offline_CQL": "#1d3557",
    "Offline_IQL": "#457b9d",
    "Behavior_Action0": "#8d99ae",
    "mc": "#1d3557",
    "td_n": "#457b9d",
    "td_lambda": "#2a9d8f",
    "sarsa_n": "#e9c46a",
    "sarsa_lambda": "#f4a261",
    "q_learning": "#e76f51",
    "double_q_learning": "#6a4c93",
}


def apply_v2_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f8f9fb",
            "axes.facecolor": "#fdfdfd",
            "axes.edgecolor": "#d8dbe2",
            "axes.grid": True,
            "grid.color": "#e6e8ee",
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.titlesize": 12.5,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
        }
    )


def colors_for(labels: Iterable[str]) -> list[str]:
    return [V2_PALETTE.get(str(x), "#577590") for x in labels]


def annotate_bars(ax, values: list[float], fmt: str = "{:.2f}") -> None:
    for i, v in enumerate(values):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color="#2b2d42")

