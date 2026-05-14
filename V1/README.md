# V1

`V1/` contains the original capstone scaffold.

## Contents

- `configs/`: baseline environment and MDP configs.
- `scripts/`: baseline preprocessing, MDP, training, evaluation, and plotting scripts.
- `src/`: baseline package code.
- `tests/`: baseline tests.
- `outputs/`: baseline and smoke-test artifacts.
- `docs/`: V1-specific notes.

## Data

The canonical dataset is shared at the repository root under `../data/`.

## Typical Workflow

Run from `V1/` with:

```powershell
$env:PYTHONPATH="src"
python scripts/preprocess.py
python scripts/data_profile.py
python scripts/build_mdp.py
python scripts/run_all_rl.py --runs 5
python -m pytest tests/test_data_env.py
```
