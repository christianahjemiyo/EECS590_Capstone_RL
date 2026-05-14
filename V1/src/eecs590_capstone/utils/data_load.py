from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_dataframe(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(Path(path), low_memory=False)
    df = df.replace("?", np.nan)
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].astype(str).str.strip()
    return df


def encode_tabular(
    df: pd.DataFrame,
    label_col: str = "readmitted",
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    if label_col not in df.columns:
        raise ValueError(f"Expected label column '{label_col}'.")

    features = df.drop(columns=[label_col])
    feature_names = list(features.columns)

    encoded_cols = []
    for col in feature_names:
        series = features[col]
        if series.dtype.kind in {"i", "u", "f"}:
            filled = series.fillna(series.median())
            encoded_cols.append(filled.astype(float).to_numpy())
        else:
            filled = series.fillna("MISSING").astype(str)
            codes, _ = pd.factorize(filled, sort=True)
            encoded_cols.append(codes.astype(float))

    X = np.column_stack(encoded_cols).astype(float)
    y = df[label_col].astype(str).to_numpy()
    return X, feature_names, y
