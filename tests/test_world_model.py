from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eecs590_capstone.world_model import (
    TabularWorldModel,
    WorldModelSimulator,
    compare_world_model_to_mdp,
    evaluate_policy_in_world_model,
)


def _tiny_synthetic_mdp() -> tuple[np.ndarray, np.ndarray]:
    """Small tabular MDP used only for lightweight V3 world-model tests."""

    P = np.array(
        [
            [[0.7, 0.3, 0.0], [0.2, 0.6, 0.2]],
            [[0.1, 0.8, 0.1], [0.0, 0.4, 0.6]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )
    R = np.array(
        [
            [[1.0, 0.2, -0.5], [0.5, 0.1, -1.0]],
            [[0.8, 0.0, -0.8], [0.0, -0.2, -1.2]],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    return P, R


def test_tabular_world_model_can_fit_a_tiny_synthetic_mdp() -> None:
    P, R = _tiny_synthetic_mdp()
    model = TabularWorldModel().fit_from_mdp(P=P, R=R, terminal_states=[2])

    assert model.n_states == 3
    assert model.n_actions == 2
    assert model.transition_probs is not None
    assert model.reward_table is not None


def test_predicted_next_state_distributions_sum_to_one() -> None:
    P, R = _tiny_synthetic_mdp()
    model = TabularWorldModel().fit_from_mdp(P=P, R=R, terminal_states=[2])

    for state in range(model.n_states):
        for action in range(model.n_actions):
            probs = model.predict_next_state_distribution(state, action)
            assert probs.shape == (3,)
            assert abs(probs.sum() - 1.0) < 1e-9


def test_rewards_can_be_predicted_for_valid_state_action_pairs() -> None:
    P, R = _tiny_synthetic_mdp()
    model = TabularWorldModel().fit_from_mdp(P=P, R=R, terminal_states=[2])

    predicted_reward = model.predict_reward(0, 1)
    expected_reward = float(np.sum(P[0, 1, :] * R[0, 1, :]))
    assert isinstance(predicted_reward, float)
    assert abs(predicted_reward - expected_reward) < 1e-9


def test_sample_next_state_returns_a_valid_state() -> None:
    P, R = _tiny_synthetic_mdp()
    model = TabularWorldModel().fit_from_mdp(P=P, R=R, terminal_states=[2])

    next_state = model.sample_next_state(1, 0, rng=np.random.default_rng(13))
    assert isinstance(next_state, int)
    assert 0 <= next_state < model.n_states


def test_world_model_simulator_reset_and_step_work() -> None:
    P, R = _tiny_synthetic_mdp()
    model = TabularWorldModel().fit_from_mdp(P=P, R=R, terminal_states=[2])
    simulator = WorldModelSimulator(model, seed=11, max_steps=5, start_state=0)

    start_state = simulator.reset()
    assert isinstance(start_state, int)
    assert 0 <= start_state < model.n_states

    step = simulator.step(1)
    assert isinstance(step.state, int)
    assert 0 <= step.state < model.n_states
    assert isinstance(step.reward, float)
    assert isinstance(step.done, bool)


def test_evaluation_metrics_return_expected_keys() -> None:
    P, R = _tiny_synthetic_mdp()
    model = TabularWorldModel().fit_from_mdp(P=P, R=R, terminal_states=[2])

    comparison_df = compare_world_model_to_mdp(model, P=P, R=R)
    assert {"transition_mae", "transition_mse", "reward_mae", "transition_kl"}.issubset(comparison_df.columns)

    policy_metrics = evaluate_policy_in_world_model(
        model,
        {"0": 0, "1": 0, "2": 0},
        episodes=12,
        seed=5,
        max_steps=5,
        start_state=0,
    )
    expected_keys = {"episodes", "avg_return", "std_return", "avg_length", "terminal_rate"}
    assert isinstance(policy_metrics, dict)
    assert expected_keys.issubset(policy_metrics.keys())
