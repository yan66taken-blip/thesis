import os
from typing import Optional
import pandas as pd

STORE_DIR = "logs"
EVAL_DIR = "eval"
ALL_VENDORS = ["Azure", "GCP", "AWS"]

# Evaluation runs already extracted full per-vendor datasets with the same
# schema as logs/<Vendor>.csv — merge them in so analysis charts aren't
# limited to the small set of live-extracted records.
EVAL_PATHS = {
    "Azure": os.path.join(EVAL_DIR, "azure_label", "Azure.csv"),
    "AWS": os.path.join(EVAL_DIR, "aws_label", "AWS.csv"),
    "GCP": os.path.join(EVAL_DIR, "gcp_label", "GCP.csv"),
}

DEDUP_KEYS = [
    "vendor", "service_name", "service_category", "start_time", "end_time",
    "user_symptom", "root_cause", "root_cause_category",
]


def load_filtered_df(
    vendors: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
) -> pd.DataFrame:
    targets = [v.strip() for v in vendors] if vendors else ALL_VENDORS

    frames = []
    for vendor in targets:
        for path in (os.path.join(STORE_DIR, f"{vendor}.csv"), EVAL_PATHS.get(vendor)):
            if path and os.path.exists(path):
                frames.append(pd.read_csv(path))

    if not frames:
        raise FileNotFoundError("No data found. Please ingest a report first.")

    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        raise ValueError("Store is empty. Please ingest a report first.")

    df = df.drop_duplicates(subset=DEDUP_KEYS).reset_index(drop=True)

    if years:
        df = df[df["year"].isin(years)]
        if df.empty:
            raise ValueError(f"No records found for year(s): {years}")

    return df
