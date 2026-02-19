from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eecs590_capstone.envs.data_env import DataDrivenEnv
from eecs590_capstone.envs.mdp_sim_env import MDPSimEnv


def test_data_env_step() -> None:
    data_path = Path("data/processed/train.csv")
    if not data_path.exists():
        return

    env = DataDrivenEnv(data_path=str(data_path), seed=7)
    state = env.reset()
    assert state is not None

    step = env.step(0)
    assert step.state is not None
    assert isinstance(step.reward, float)
    assert isinstance(step.done, bool)
    assert "label" in step.info


def test_mdp_sim_env_step() -> None:
    mdp_path = Path("outputs/mdp/mdp.npz")
    if not mdp_path.exists():
        return

    env = MDPSimEnv(mdp_path=str(mdp_path), seed=7)
    state = env.reset()
    assert isinstance(state, int)

    step = env.step(0)
    assert isinstance(step.state, int)
    assert isinstance(step.reward, float)
    assert isinstance(step.done, bool)
