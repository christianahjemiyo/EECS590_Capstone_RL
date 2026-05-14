from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from eecs590_capstone.utils.bayesian_tuning import SearchBounds, as_tag, propose_next


def test_bayesian_tuning_proposal_stays_within_bounds() -> None:
    bounds = SearchBounds(
        no_reward=(3.0, 6.0),
        late_penalty=(-3.5, -0.5),
        early_penalty=(-14.0, -6.0),
        cost_scale=(0.4, 1.8),
    )
    X = np.array(
        [
            [4.0, -2.0, -10.0, 1.0],
            [5.0, -1.0, -8.0, 1.2],
            [3.5, -3.0, -12.0, 0.8],
        ],
        dtype=float,
    )
    y = np.array([-10.0, -8.0, -9.5], dtype=float)
    candidate = propose_next(X, y, bounds, rng=np.random.default_rng(7), n_candidates=64)
    tag = as_tag(candidate)

    assert bounds.no_reward[0] <= tag["no_reward"] <= bounds.no_reward[1]
    assert bounds.late_penalty[0] <= tag["late_penalty"] <= bounds.late_penalty[1]
    assert bounds.early_penalty[0] <= tag["early_penalty"] <= bounds.early_penalty[1]
    assert bounds.cost_scale[0] <= tag["cost_scale"] <= bounds.cost_scale[1]
