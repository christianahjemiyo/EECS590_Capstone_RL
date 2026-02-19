from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _split_by_patient(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "patient_nbr" in df.columns:
        bucket = df["patient_nbr"].astype("int64") % 100
        train = df[bucket < 70]
        val = df[(bucket >= 70) & (bucket < 85)]
        test = df[bucket >= 85]
        return train, val, test

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(len(df) * 0.70)
    n_val = int(len(df) * 0.15)
    train = df.iloc[idx[:n_train]]
    val = df.iloc[idx[n_train:n_train + n_val]]
    test = df.iloc[idx[n_train + n_val:]]
    return train, val, test


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess Kaggle diabetes readmission data.")
    parser.add_argument("--raw", default="data/raw/diabetic_data.csv")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    raw_path = Path(args.raw)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path, low_memory=False)
    df = df.replace("?", np.nan)

    # Basic string cleanup
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()

    # Drop duplicate encounters
    if "encounter_id" in df.columns:
        df = df.drop_duplicates(subset=["encounter_id"])
    else:
        df = df.drop_duplicates()

    # Ensure readmitted label exists
    if "readmitted" not in df.columns:
        raise ValueError("Expected a 'readmitted' column in the dataset.")

    clean_path = out_dir / "diabetic_data_clean.csv"
    df.to_csv(clean_path, index=False)

    train, val, test = _split_by_patient(df, seed=args.seed)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    print(f"Wrote: {clean_path}")
    print(f"Wrote: {out_dir / 'train.csv'}")
    print(f"Wrote: {out_dir / 'val.csv'}")
    print(f"Wrote: {out_dir / 'test.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
