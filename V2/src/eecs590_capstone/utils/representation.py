from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class TokenizerStateArtifacts:
    states: np.ndarray
    score: np.ndarray
    embedding_dim: int
    feature_columns: list[str]
    label_values: list[str]
    loss_curve: list[float]
    numeric_columns: list[str]
    categorical_columns: list[str]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _quantile_bins(series: pd.Series, bins: int) -> np.ndarray:
    clean = pd.to_numeric(series, errors="coerce")
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = clean.quantile(quantiles).to_numpy(dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    for i in range(1, len(edges)):
        if not np.isfinite(edges[i - 1]):
            continue
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6
    return edges


def _numeric_projection(values: np.ndarray, emb_num: np.ndarray) -> np.ndarray:
    return values @ emb_num


def build_tokenizer_states(
    df: pd.DataFrame,
    label_col: str,
    n_states: int = 4,
    embedding_dim: int = 8,
    numeric_bins: int = 5,
    epochs: int = 60,
    lr: float = 0.03,
    seed: int = 7,
) -> TokenizerStateArtifacts:
    if label_col not in df.columns:
        raise ValueError(f"Expected label column '{label_col}'.")

    features = df.drop(columns=[label_col]).copy()
    labels = df[label_col].astype(str).to_numpy()
    label_values = sorted(pd.unique(labels).tolist())
    label_to_idx = {label: idx for idx, label in enumerate(label_values)}
    y = np.array([label_to_idx[label] for label in labels], dtype=np.int64)

    numeric_columns: list[str] = []
    categorical_columns: list[str] = []
    token_sequences: list[list[str]] = []
    numeric_rows: list[list[float]] = []

    numeric_edges: dict[str, np.ndarray] = {}
    for col in features.columns:
        series = features[col]
        if series.dtype.kind in {"i", "u", "f"}:
            numeric_columns.append(col)
            numeric_edges[col] = _quantile_bins(series, numeric_bins)
        else:
            categorical_columns.append(col)

    numeric_stats: dict[str, tuple[float, float]] = {}
    for col in numeric_columns:
        clean = pd.to_numeric(features[col], errors="coerce")
        median = float(clean.median()) if len(clean.dropna()) else 0.0
        std = float(clean.std(ddof=0)) if len(clean.dropna()) else 1.0
        numeric_stats[col] = (median, std if std > 1e-6 else 1.0)

    for _, row in features.iterrows():
        row_tokens: list[str] = []
        numeric_vals: list[float] = []

        for col in categorical_columns:
            val = str(row[col]).strip()
            if val == "" or val.lower() == "nan":
                val = "MISSING"
            row_tokens.append(f"{col}={val}")

        for col in numeric_columns:
            raw = pd.to_numeric(pd.Series([row[col]]), errors="coerce").iloc[0]
            median, std = numeric_stats[col]
            if pd.isna(raw):
                raw = median
            numeric_vals.append((float(raw) - median) / std)
            edges = numeric_edges[col]
            bucket = int(np.digitize([float(raw)], edges[1:-1], right=True)[0])
            row_tokens.append(f"{col}_bin={bucket}")

        token_sequences.append(row_tokens)
        numeric_rows.append(numeric_vals)

    numeric_matrix = np.array(numeric_rows, dtype=float) if numeric_columns else np.zeros((len(df), 0), dtype=float)

    vocab = sorted({token for seq in token_sequences for token in seq})
    token_to_idx = {token: idx for idx, token in enumerate(vocab)}
    token_ids = [np.array([token_to_idx[token] for token in seq], dtype=np.int64) for seq in token_sequences]

    rng = np.random.default_rng(seed)
    emb_cat = rng.normal(0.0, 0.08, size=(len(vocab), embedding_dim))
    emb_num = (
        rng.normal(0.0, 0.08, size=(numeric_matrix.shape[1], embedding_dim))
        if numeric_matrix.shape[1]
        else np.zeros((0, embedding_dim), dtype=float)
    )
    head = rng.normal(0.0, 0.08, size=(embedding_dim, len(label_values)))
    bias = np.zeros(len(label_values), dtype=float)
    loss_curve: list[float] = []

    for _ in range(epochs):
        total_loss = 0.0
        order = rng.permutation(len(token_ids))
        for idx in order:
            ids = token_ids[idx]
            cat_emb = emb_cat[ids].mean(axis=0)
            num_emb = _numeric_projection(numeric_matrix[idx : idx + 1], emb_num)[0]
            hidden = cat_emb + num_emb

            logits = hidden @ head + bias
            probs = _softmax(logits.reshape(1, -1))[0]
            target = y[idx]
            total_loss += -float(np.log(probs[target] + 1e-12))

            grad_logits = probs
            grad_logits[target] -= 1.0

            grad_head = np.outer(hidden, grad_logits)
            grad_bias = grad_logits
            grad_hidden = head @ grad_logits

            head -= lr * grad_head
            bias -= lr * grad_bias

            if ids.size:
                grad_cat = grad_hidden / float(ids.size)
                emb_cat[ids] -= lr * grad_cat

            if numeric_matrix.shape[1]:
                grad_num = np.outer(numeric_matrix[idx], grad_hidden)
                emb_num -= lr * grad_num

        loss_curve.append(total_loss / max(1, len(token_ids)))

    hidden_rows = []
    for idx, ids in enumerate(token_ids):
        cat_emb = emb_cat[ids].mean(axis=0)
        num_emb = _numeric_projection(numeric_matrix[idx : idx + 1], emb_num)[0]
        hidden_rows.append(cat_emb + num_emb)
    hidden_matrix = np.array(hidden_rows, dtype=float)
    centered = hidden_matrix - hidden_matrix.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    score = centered @ vh[0]
    quantiles = np.quantile(score, np.linspace(0.0, 1.0, n_states + 1)[1:-1])
    states = np.digitize(score, quantiles, right=True).astype(int)

    return TokenizerStateArtifacts(
        states=states,
        score=score.astype(float),
        embedding_dim=embedding_dim,
        feature_columns=list(features.columns),
        label_values=label_values,
        loss_curve=loss_curve,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )


def representation_meta(artifacts: TokenizerStateArtifacts) -> dict[str, Any]:
    return {
        "mode": "tokenizer_embedding",
        "embedding_dim": int(artifacts.embedding_dim),
        "feature_columns": artifacts.feature_columns,
        "label_values": artifacts.label_values,
        "loss_curve": [float(x) for x in artifacts.loss_curve],
        "numeric_columns": artifacts.numeric_columns,
        "categorical_columns": artifacts.categorical_columns,
    }
