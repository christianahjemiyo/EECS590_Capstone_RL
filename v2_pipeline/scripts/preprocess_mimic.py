from __future__ import annotations

import argparse
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def _split_by_patient(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    bucket = df["patient_nbr"].astype("int64") % 100
    train = df[bucket < 70]
    val = df[(bucket >= 70) & (bucket < 85)]
    test = df[bucket >= 85]
    return train, val, test


def _age_bin(age: float) -> str:
    if pd.isna(age):
        return "[0-10)"
    lo = int(np.floor(max(0.0, min(90.0, float(age))) / 10.0) * 10)
    hi = lo + 10 if lo < 90 else 100
    return f"[{lo}-{hi})"


def _find_member(members: list[str], suffix: str) -> str:
    for m in members:
        if m.endswith(suffix):
            return m
    raise FileNotFoundError(f"Could not find '{suffix}' inside zip archive.")


def _read_csv_gz_from_zip(zip_path: Path, member_name: str, usecols: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name, "r") as fp:
            return pd.read_csv(fp, compression="gzip", usecols=usecols, low_memory=False)


def _count_by_hadm_id_from_zip(
    zip_path: Path,
    member_name: str,
    chunksize: int = 500_000,
) -> dict[int, int]:
    """Count rows per hadm_id from a large gzip CSV stored inside the zip."""
    counts: Counter[int] = Counter()
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(member_name, "r") as fp:
            reader = pd.read_csv(
                fp,
                compression="gzip",
                usecols=["hadm_id"],
                low_memory=False,
                chunksize=chunksize,
            )
            for chunk in reader:
                s = pd.to_numeric(chunk["hadm_id"], errors="coerce").dropna().astype("int64")
                if s.empty:
                    continue
                vc = s.value_counts()
                counts.update({int(k): int(v) for k, v in vc.items()})
    return dict(counts)


def _try_find_member(members: list[str], suffix: str) -> str | None:
    for m in members:
        if m.endswith(suffix):
            return m
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build RL-ready readmission dataset from MIMIC-IV admissions + patients."
    )
    parser.add_argument("--mimic-zip", required=True, help="Path to mimic-iv-3.1.zip")
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--sample-rows", type=int, default=0, help="Optional dev-time row cap.")
    parser.add_argument(
        "--skip-direct-counts",
        action="store_true",
        help="Skip direct feature counts from diagnoses/procedures/prescriptions/labevents (faster).",
    )
    args = parser.parse_args()

    zip_path = Path(args.mimic_zip)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
    admissions_member = _find_member(members, "hosp/admissions.csv.gz")
    patients_member = _find_member(members, "hosp/patients.csv.gz")
    diagnoses_member = _try_find_member(members, "hosp/diagnoses_icd.csv.gz")
    procedures_member = _try_find_member(members, "hosp/procedures_icd.csv.gz")
    prescriptions_member = _try_find_member(members, "hosp/prescriptions.csv.gz")
    labevents_member = _try_find_member(members, "hosp/labevents.csv.gz")

    admissions = _read_csv_gz_from_zip(
        zip_path,
        admissions_member,
        usecols=[
            "subject_id",
            "hadm_id",
            "admittime",
            "dischtime",
            "admission_type",
            "admission_location",
            "discharge_location",
            "insurance",
            "language",
            "marital_status",
            "race",
            "hospital_expire_flag",
        ],
    )
    patients = _read_csv_gz_from_zip(
        zip_path,
        patients_member,
        usecols=["subject_id", "gender", "anchor_age"],
    )

    if args.sample_rows and args.sample_rows > 0:
        admissions = admissions.head(args.sample_rows).copy()

    admissions["admittime"] = pd.to_datetime(admissions["admittime"], errors="coerce")
    admissions["dischtime"] = pd.to_datetime(admissions["dischtime"], errors="coerce")

    df = admissions.merge(patients, on="subject_id", how="left")
    df = df.dropna(subset=["subject_id", "hadm_id", "admittime", "dischtime"]).copy()
    df = df.sort_values(["subject_id", "admittime", "hadm_id"]).reset_index(drop=True)

    df["next_admittime"] = df.groupby("subject_id")["admittime"].shift(-1)
    days_to_next = (df["next_admittime"] - df["dischtime"]).dt.total_seconds() / 86400.0
    df["readmitted"] = np.select(
        [days_to_next.between(0, 30, inclusive="left"), days_to_next >= 30],
        ["<30", ">30"],
        default="NO",
    )

    los_days = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400.0
    df["time_in_hospital"] = np.ceil(los_days.clip(lower=0)).fillna(0).astype(int)
    df["number_inpatient"] = df.groupby("subject_id").cumcount()

    adm_type = df["admission_type"].fillna("").astype(str).str.upper()
    df["number_emergency"] = adm_type.str.contains("EMER", regex=False).astype(int)
    df["number_outpatient"] = 0
    df["num_lab_procedures"] = 0
    df["num_medications"] = 0
    df["num_procedures"] = 0
    df["number_diagnoses"] = 0

    if not args.skip_direct_counts:
        print("Computing direct clinical count features from MIMIC tables...")
        hadm = df["hadm_id"].astype("int64")

        if diagnoses_member is not None:
            dx_counts = _count_by_hadm_id_from_zip(zip_path, diagnoses_member)
            df["number_diagnoses"] = hadm.map(dx_counts).fillna(0).astype(int)
        else:
            print("Warning: diagnoses_icd.csv.gz not found; number_diagnoses stays 0.")

        if procedures_member is not None:
            proc_counts = _count_by_hadm_id_from_zip(zip_path, procedures_member)
            df["num_procedures"] = hadm.map(proc_counts).fillna(0).astype(int)
        else:
            print("Warning: procedures_icd.csv.gz not found; num_procedures stays 0.")

        if prescriptions_member is not None:
            med_counts = _count_by_hadm_id_from_zip(zip_path, prescriptions_member)
            df["num_medications"] = hadm.map(med_counts).fillna(0).astype(int)
        else:
            print("Warning: prescriptions.csv.gz not found; num_medications stays 0.")

        if labevents_member is not None:
            lab_counts = _count_by_hadm_id_from_zip(zip_path, labevents_member)
            df["num_lab_procedures"] = hadm.map(lab_counts).fillna(0).astype(int)
        else:
            print("Warning: labevents.csv.gz not found; num_lab_procedures stays 0.")
    df["age"] = df["anchor_age"].apply(_age_bin)

    out = pd.DataFrame(
        {
            "encounter_id": df["hadm_id"].astype("int64"),
            "patient_nbr": df["subject_id"].astype("int64"),
            "readmitted": df["readmitted"].astype(str),
            "time_in_hospital": df["time_in_hospital"].astype(int),
            "num_lab_procedures": df["num_lab_procedures"].astype(int),
            "num_medications": df["num_medications"].astype(int),
            "num_procedures": df["num_procedures"].astype(int),
            "number_diagnoses": df["number_diagnoses"].astype(int),
            "number_inpatient": df["number_inpatient"].astype(int),
            "number_emergency": df["number_emergency"].astype(int),
            "number_outpatient": df["number_outpatient"].astype(int),
            "age": df["age"].astype(str),
            "gender": df["gender"].fillna("UNK").astype(str),
            "admission_type": df["admission_type"].fillna("UNK").astype(str),
            "admission_location": df["admission_location"].fillna("UNK").astype(str),
            "discharge_location": df["discharge_location"].fillna("UNK").astype(str),
            "insurance": df["insurance"].fillna("UNK").astype(str),
            "language": df["language"].fillna("UNK").astype(str),
            "marital_status": df["marital_status"].fillna("UNK").astype(str),
            "race": df["race"].fillna("UNK").astype(str),
            "hospital_expire_flag": pd.to_numeric(df["hospital_expire_flag"], errors="coerce").fillna(0).astype(int),
        }
    ).drop_duplicates(subset=["encounter_id"])

    clean_path = out_dir / "mimic_data_clean.csv"
    out.to_csv(clean_path, index=False)

    train, val, test = _split_by_patient(out)
    train.to_csv(out_dir / "train.csv", index=False)
    val.to_csv(out_dir / "val.csv", index=False)
    test.to_csv(out_dir / "test.csv", index=False)

    print(f"Wrote: {clean_path}")
    print(f"Wrote: {out_dir / 'train.csv'}")
    print(f"Wrote: {out_dir / 'val.csv'}")
    print(f"Wrote: {out_dir / 'test.csv'}")
    print("Readmission distribution:")
    print(out["readmitted"].value_counts(dropna=False).to_string())
    print("Direct feature medians:")
    for c in ["number_diagnoses", "num_procedures", "num_medications", "num_lab_procedures"]:
        print(f"- {c}: {float(pd.to_numeric(out[c], errors='coerce').median()):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
