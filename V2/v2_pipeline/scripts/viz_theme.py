from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt


V2_PALETTE = {
    "DP_PolicyIter": "#203a43",
    "DP_ValueIter": "#2c6e63",
    "Policy Iter": "#203a43",
    "Value Iter": "#2c6e63",
    "DP": "#203a43",
    "Tabular": "#2c6e63",
    "Approximation": "#7f5539",
    "Deep Value": "#c75b39",
    "Actor-Critic": "#8c1c13",
    "Advanced": "#5f0f40",
    "Q_Learning": "#c75b39",
    "Q-Learning": "#c75b39",
    "Double_Q": "#7f5539",
    "Double Q-Learning": "#7f5539",
    "Offline_CQL": "#8c1c13",
    "Offline_IQL": "#335c67",
    "Behavior_Action0": "#7b8c7a",
    "Intervention strength": "#2c6e63",
    "Intervention cost": "#c75b39",
    "mc": "#5f0f40",
    "td_n": "#335c67",
    "td_lambda": "#2c6e63",
    "sarsa_n": "#d4a373",
    "sarsa_lambda": "#b56576",
    "q_learning": "#c75b39",
    "double_q_learning": "#7f5539",
}


def apply_v2_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#f3efe6",
            "axes.facecolor": "#fffdf8",
            "axes.edgecolor": "#c9c1b7",
            "axes.grid": True,
            "grid.color": "#e2d9cc",
            "grid.linewidth": 0.9,
            "grid.alpha": 0.9,
            "grid.linestyle": ":",
            "font.family": "DejaVu Serif",
            "font.size": 10.0,
            "axes.titlesize": 13.0,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9.0,
            "axes.titleweight": "semibold",
            "axes.labelcolor": "#2b2a28",
            "xtick.color": "#3b3835",
            "ytick.color": "#3b3835",
            "text.color": "#2b2a28",
        }
    )


def colors_for(labels: Iterable[str]) -> list[str]:
    return [V2_PALETTE.get(str(x), "#6b705c") for x in labels]


def annotate_bars(ax, values: list[float], fmt: str = "{:.2f}") -> None:
    for i, v in enumerate(values):
        ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8, color="#2f2724")
