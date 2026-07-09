"""
Batch evaluation of the extractor against a ground-truth CSV.

Run directly:
    python evaluate.py

Extracted records and eval results are written to:
    eval/<input_stem>/
        <Vendor>.csv          — raw extracted records
        <input_stem>_eval.csv — per-row predictions + match flags

Re-running the same input overwrites the same folder.
Accuracy is computed only over rows where the ground-truth value is not
UNKNOWN / NaN / empty (fields with only unknown labels are skipped entirely).
"""

from __future__ import annotations

import os
import re
import shutil
import pandas as pd
from datetime import timezone, timedelta
from dateutil import parser as dateutil_parser
from tools.extractor import Metadata, extract_record

EVAL_ROOT = "eval"
EVAL_FIELDS = list(Metadata.model_fields.keys())


_SKIP_VALUES = {"UNKNOWN", "NAN", ""}
_TIME_FIELDS = {"start_time", "end_time"}


def normalize(value) -> str:
    return str(value).strip().upper()


def is_valid_gt(val: str) -> bool:
    return normalize(val) not in _SKIP_VALUES


def fields_match(pred: str, gt: str) -> bool:
    """Set comparison for comma-separated multi-label fields."""
    pred_set = {v.strip() for v in pred.upper().split(",")}
    gt_set = {v.strip() for v in gt.upper().split(",")}
    return pred_set == gt_set


_TZ_OFFSETS = {
    "UTC": timedelta(0),
    "PST": timedelta(hours=-8),
    "PDT": timedelta(hours=-7),
    "CST": timedelta(hours=-6),
    "CDT": timedelta(hours=-5),
    "EST": timedelta(hours=-5),
    "EDT": timedelta(hours=-4),
}


def _parse_dt(value: str):
    try:
        return dateutil_parser.parse(value.strip(), fuzzy=True)
    except (ValueError, OverflowError, TypeError):
        return None


def _to_utc(dt, tz_label: str):
    offset = _TZ_OFFSETS.get(tz_label.upper())
    if offset is None:
        return dt
    return dt - offset


def normalize_time(value: str) -> str | None:
    """Parse any datetime/time string and return 'HH:MM' (24-hour)."""
    dt = _parse_dt(value)
    if dt is None:
        return None
    return f"{dt.hour:02d}:{dt.minute:02d}"


def times_match(pred: str, gt: str, tz_label: str = "") -> bool:
    """Compare two datetime strings at minute granularity (HH:MM).

    When tz_label is provided (e.g. 'PDT'), the ground truth is assumed to
    be in that timezone.  The prediction is tried as-is first; if that fails
    it is also tried as if it were in the label timezone, so a match is found
    regardless of whether the model output UTC or the local timezone.
    """
    gt_dt = _parse_dt(gt)
    pred_dt = _parse_dt(pred)
    if gt_dt is None or pred_dt is None:
        return False

    gt_hm = f"{gt_dt.hour:02d}:{gt_dt.minute:02d}"
    pred_hm = f"{pred_dt.hour:02d}:{pred_dt.minute:02d}"

    if gt_hm == pred_hm:
        return True

    if tz_label and tz_label.upper() not in ("", "UNKNOWN"):
        gt_utc = _to_utc(gt_dt, tz_label)
        gt_utc_hm = f"{gt_utc.hour:02d}:{gt_utc.minute:02d}"

        if pred_hm == gt_utc_hm:
            return True

        pred_utc = _to_utc(pred_dt, tz_label)
        pred_utc_hm = f"{pred_utc.hour:02d}:{pred_utc.minute:02d}"
        if pred_utc_hm == gt_hm:
            return True

    return False


def run_evaluation(input_path: str, desc_col: str = "description", output_path: str = None, eval_root: str = None) -> dict:
    """
    Run the extractor against a ground-truth CSV and return a results dict.

    Returns:
        {
            "results_df": pd.DataFrame,
            "gt_fields": list[str],
            "output_path": str,
            "run_dir": str,
        }
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]
    run_dir = os.path.join(eval_root or EVAL_ROOT, stem)
    if os.path.isdir(run_dir):
        shutil.rmtree(run_dir)
    os.makedirs(run_dir)
    print(f"Output directory: {run_dir}/")

    if output_path is None:
        output_path = os.path.join(run_dir, f"{stem}_eval.csv")

    df = pd.read_csv(input_path)

    if desc_col not in df.columns:
        raise ValueError(f"Column '{desc_col}' not found. Available: {list(df.columns)}")

    # Auto-detect label prefix: "label_<field>" or "<field>"
    prefix = ""
    for f in EVAL_FIELDS:
        if f"label_{f}" in df.columns:
            prefix = "label_"
            break

    gt_fields = [f for f in EVAL_FIELDS if f"{prefix}{f}" in df.columns]
    if gt_fields:
        print(f"Ground truth fields found (prefix='{prefix}'): {gt_fields}")
    else:
        print("No ground truth fields found — predictions only, no accuracy computed.")

    rows = []
    total = len(df)
    for i, row in df.iterrows():
        report = str(row[desc_col])
        parts = []
        if "service" in row and str(row["service"]) not in ("", "nan"):
            parts.append(f"service: {row['service']}")
        if "status" in row and str(row["status"]) not in ("", "nan"):
            parts.append(f"status: {row['status']}")
        parts.append(f"description: {report}")
        if "external_description" in row and str(row["external_description"]) not in ("", "nan"):
            parts.append(f"external_description: {row['external_description']}")
        report = "\n".join(parts)
        print(f"[{i+1}/{total}] Extracting...", end=" ", flush=True)

        pred = extract_record(report, run_dir)

        result = row.drop(labels=[desc_col], errors="ignore").to_dict()

        for field, value in pred.items():
            result[f"pred_{field}"] = value

        tz_col = f"{prefix}timezone"
        tz_label = str(row[tz_col]).strip() if tz_col in df.columns else ""

        for field in gt_fields:
            gt_col = f"{prefix}{field}"
            gt_val = normalize(str(row[gt_col]))
            pred_val = normalize(str(pred.get(field, "")))
            if is_valid_gt(gt_val):
                if field in _TIME_FIELDS:
                    result[f"{field}_match"] = times_match(
                        str(pred.get(field, "")), str(row[gt_col]), tz_label
                    )
                else:
                    result[f"{field}_match"] = fields_match(pred_val, gt_val)
            else:
                result[f"{field}_match"] = None

        rows.append(result)
        print(f"done ({pred.get('time_sec', '?')}s, cached={pred.get('cached_tokens', 0)})")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")

    return {"results_df": results_df, "gt_fields": gt_fields, "output_path": output_path, "run_dir": run_dir}


LABEL_DATA_DIR = "label_data"

VENDOR_DESC_COL = {
    "aws_label.csv": "description",
    "azure_label.csv": "description",
    "gcp_label.csv": "description",
}


def print_accuracy(results_df: pd.DataFrame, gt_fields: list[str]) -> None:
    print("\n=== Accuracy Summary (UNKNOWN/NaN GT rows excluded per field) ===")
    for field in gt_fields:
        match_col = f"{field}_match"
        if match_col not in results_df.columns:
            continue
        valid = results_df[match_col].notna()
        valid_count = valid.sum()
        skipped = len(results_df) - valid_count
        if valid_count == 0:
            print(f"  {field:<30} all GT unknown — skipped")
            continue
        correct = results_df.loc[valid, match_col].sum()
        acc = correct / valid_count * 100
        skip_note = f"  [{skipped} skipped]" if skipped else ""
        print(f"  {field:<30} {correct}/{valid_count}  ({acc:.1f}%){skip_note}")


def main():
    for filename, desc_col in VENDOR_DESC_COL.items():
        input_path = os.path.join(LABEL_DATA_DIR, filename)
        if not os.path.exists(input_path):
            print(f"[SKIP] {input_path} not found")
            continue
        print(f"\n{'='*60}")
        print(f"Processing: {filename}")
        print(f"{'='*60}")
        result = run_evaluation(input_path, desc_col)
        if result["gt_fields"]:
            print_accuracy(result["results_df"], result["gt_fields"])


if __name__ == "__main__":
    main()
