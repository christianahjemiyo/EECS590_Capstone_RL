# Data Setup (Kaggle)

This project uses the Kaggle dataset **Diabetes 130-US Hospitals for Years 1999–2008**.

## 1) Install Kaggle CLI
```powershell
python -m pip install kaggle
```

## 2) Configure Kaggle credentials
Download your `kaggle.json` API token from Kaggle:
- Kaggle account settings → API → "Create New API Token"

Place it at:
```text
C:\Users\Christianah\.kaggle\kaggle.json
```

## 3) Download the dataset
From the repo root:
```powershell
kaggle datasets download -d ashikuzzamanshishir/diabetes-130-us-hospitals-for-years-1999-2008 -p data/raw
```

If the download is a zip file, extract it into `data/raw/`.

## 4) Expected raw files
At minimum, the dataset should include a file with the readmission label:
- `readmitted` column with values `<30`, `>30`, `NO`

## 5) Preprocess and split
From the repo root:
```powershell
python scripts/preprocess.py
```

This writes:
- `data/processed/diabetic_data_clean.csv`
- `data/processed/train.csv`
- `data/processed/val.csv`
- `data/processed/test.csv`

## 6) Data profile
```powershell
python scripts/data_profile.py
```

This writes:
- `data/processed/data_profile.md`

## 7) Data-driven env config
Default config lives in:
- `configs/data_env.json`

## 8) Build the MDP simulator
```powershell
python scripts/build_mdp.py
```

This writes:
- `outputs/mdp/mdp.npz`
- `outputs/mdp/mdp_meta.json`

## 9) Run DP algorithms
```powershell
python -m eecs590_capstone.cli.mdp_train --algo policy_iter
python -m eecs590_capstone.cli.mdp_train --algo value_iter
```

## 10) Visualize DP results
```powershell
python scripts/plot_mdp_results.py --algo policy_iter
python scripts/plot_mdp_results.py --algo value_iter
```

## 11) Render HTML playback
```powershell
python scripts/render_mdp_html.py --policy outputs/mdp/policy_iter_policy.json
```

## 12) Train tabular RL algorithms
```powershell
python -m eecs590_capstone.cli.rl_train --algo mc
python -m eecs590_capstone.cli.rl_train --algo td_n --n 3
python -m eecs590_capstone.cli.rl_train --algo td_lambda --lambda 0.8
python -m eecs590_capstone.cli.rl_train --algo sarsa_n --n 3
python -m eecs590_capstone.cli.rl_train --algo sarsa_lambda --lambda 0.8
python -m eecs590_capstone.cli.rl_train --algo q_learning
```

## 13) Compare learning curves
```powershell
python scripts/run_all_rl.py --runs 5
python scripts/plot_learning_curves.py
```
