# V2 Upgrade Pack (MIMIC-IV)

This folder contains the Version 2 upgrade artifacts for the capstone:
- MIMIC-IV preprocessing pipeline
- MIMIC-tuned configs
- Reproducible V2 training commands

## Files
- `V2/scripts/preprocess_mimic.py`
- `V2/scripts/plot_v2_results.py`
- `V2/scripts/write_v2_interpretation.py`
- `V2/configs/data_env_mimic.json`
- `V2/configs/mdp_sim_mimic.json`
- `V2/docs/external_validation_plan.md` (planning note only, implementation on hold)

## Run V2 End-to-End
From repo root:

```powershell
$env:PYTHONPATH="src"
python V2/scripts/preprocess_mimic.py --mimic-zip "C:\Users\christianah.jemiyo\OneDrive - North Dakota University System\Grad_applications\PhysioNet_Data\mimic-iv-3.1.zip"
python scripts/data_profile.py --data data/processed/mimic_data_clean.csv --out outputs/V2/data_profile_mimic.md
python scripts/build_mdp.py --config V2/configs/mdp_sim_mimic.json --outdir outputs/V2/mdp
python -m eecs590_capstone.cli.mdp_train --algo policy_iter --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/mdp
python -m eecs590_capstone.cli.mdp_train --algo value_iter --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/mdp
python -m eecs590_capstone.cli.rl_train --algo q_learning --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/rl
python -m eecs590_capstone.cli.rl_train --algo double_q_learning --mdp outputs/V2/mdp/mdp.npz --outdir outputs/V2/rl_double_q
python scripts/plot_mdp_results.py --outdir outputs/V2/mdp --algo policy_iter
python scripts/plot_mdp_results.py --outdir outputs/V2/mdp --algo value_iter
python V2/scripts/plot_v2_results.py
python V2/scripts/write_v2_interpretation.py
```

## Why commit outputs/V2
- You can show concrete evidence of results in the repo.
- Reviewers can inspect metrics without rerunning training.
- Keeping results in `outputs/V2` avoids mixing with V1 outputs.
