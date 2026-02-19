from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a data profile summary.")
    parser.add_argument("--data", default="data/processed/diabetic_data_clean.csv")
    parser.add_argument("--out", default="data/processed/data_profile.md")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path, low_memory=False)

    rows = []
    for col in df.columns:
        series = df[col]
        missing = series.isna().mean() * 100.0
        nunique = series.nunique(dropna=True)
        dtype = str(series.dtype)
        rows.append((col, dtype, f"{missing:.2f}%", int(nunique)))

    lines = []
    lines.append(f"# Data Profile")
    lines.append("")
    lines.append(f"- Source: `{data_path.as_posix()}`")
    lines.append(f"- Rows: {len(df)}")
    lines.append(f"- Columns: {len(df.columns)}")
    lines.append("")
    lines.append("| Column | Dtype | Missing | Unique |")
    lines.append("| --- | --- | --- | --- |")
    for col, dtype, missing, nunique in rows:
        lines.append(f"| {col} | {dtype} | {missing} | {nunique} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
