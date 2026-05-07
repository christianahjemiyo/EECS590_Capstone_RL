from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eecs590_capstone.utils.representation import build_tokenizer_states
from eecs590_capstone.utils.reward_search import candidate_name, materialize_reward_candidate


def test_materialize_reward_candidate_scales_action_costs() -> None:
    cfg = {
        "action_costs": [0.0, 0.1, 0.2],
        "reward_map": {"<30": -10.0, ">30": -2.0, "NO": 4.0},
    }
    updated, tag = materialize_reward_candidate(cfg, no_reward=5.0, late_penalty=-1.0, early_penalty=-9.0, cost_scale=1.5)
    assert updated["reward_map"]["NO"] == 5.0
    assert updated["reward_map"][">30"] == -1.0
    assert updated["reward_map"]["<30"] == -9.0
    assert all(abs(a - b) < 1e-9 for a, b in zip(updated["action_costs"], [0.0, 0.15, 0.3]))
    assert "cost_1.50" in candidate_name(tag)


def test_build_tokenizer_states_returns_discrete_states() -> None:
    df = pd.DataFrame(
        {
            "age": ["[40-50)", "[60-70)", "[70-80)", "[30-40)"],
            "num_medications": [8, 20, 25, 4],
            "number_inpatient": [0, 2, 3, 0],
            "race": ["Caucasian", "AfricanAmerican", "Caucasian", "Asian"],
            "readmitted": ["NO", ">30", "<30", "NO"],
        }
    )
    artifacts = build_tokenizer_states(df, label_col="readmitted", n_states=3, embedding_dim=4, epochs=8, lr=0.05, seed=3)
    assert len(artifacts.states) == len(df)
    assert artifacts.states.min() >= 0
    assert artifacts.states.max() < 3
    assert len(artifacts.loss_curve) == 8
