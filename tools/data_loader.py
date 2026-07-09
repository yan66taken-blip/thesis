import os
from typing import Optional
import pandas as pd

STORE_DIR = "logs"
EVAL_DIR = "eval_final"
ALL_VENDORS = ["Azure", "GCP", "AWS"]

# eval_final/ holds the final extraction evaluation (predictions + ground
# truth + match flags, see eval_final/README.md). Only the model's own
# predictions are relevant for analysis charts, so pred_* columns are
# renamed to match logs/<Vendor>.csv's unprefixed schema before merging.
EVAL_PATHS = {
    "Azure": os.path.join(EVAL_DIR, "Azure.csv"),
    "AWS": os.path.join(EVAL_DIR, "AWS.csv"),
    "GCP": os.path.join(EVAL_DIR, "GCP.csv"),
}

PRED_COLUMN_MAP = {
    # note: "vendor" is deliberately NOT remapped from pred_vendor — the eval
    # CSVs already carry an unprefixed ground-truth "vendor" column, and
    # renaming pred_vendor onto it too would create a duplicate column.
    "pred_service_name": "service_name",
    "pred_service_category": "service_category",
    "pred_start_time": "start_time",
    "pred_end_time": "end_time",
    "pred_user_symptom": "user_symptom",
    "pred_user_symptom_category": "user_symptom_category",
    "pred_root_cause": "root_cause",
    "pred_root_cause_category": "root_cause_category",
    "pred_time_sec": "time_sec",
    "pred_tokens_used": "tokens_used",
    "pred_cached_tokens": "cached_tokens",
}

KEEP_COLUMNS = [
    "vendor", "year", "service_name", "service_category", "start_time",
    "end_time", "user_symptom", "user_symptom_category", "root_cause",
    "root_cause_category", "time_sec", "tokens_used", "cached_tokens",
]

def _load_eval_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {k: v for k, v in PRED_COLUMN_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    return df[[c for c in KEEP_COLUMNS if c in df.columns]]


def load_filtered_df(
    vendors: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
) -> pd.DataFrame:
    targets = [v.strip() for v in vendors] if vendors else ALL_VENDORS

    frames = []
    for vendor in targets:
        live_path = os.path.join(STORE_DIR, f"{vendor}.csv")
        if os.path.exists(live_path):
            frames.append(pd.read_csv(live_path))
        eval_path = EVAL_PATHS.get(vendor)
        if eval_path and os.path.exists(eval_path):
            frames.append(_load_eval_csv(eval_path))

    if not frames:
        raise FileNotFoundError("No data found. Please ingest a report first.")

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError("Store is empty. Please ingest a report first.")

    if years:
        df = df[df["year"].isin(years)]
        if df.empty:
            raise ValueError(f"No records found for year(s): {years}")

    return df
