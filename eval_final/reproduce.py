"""
Reproduces the results in eval_final/{AWS,Azure,GCP}.csv.

Differs from evaluate.py in one way: it sends the FULL report text with no
truncation, bypassing tools.extractor.extract_record's MAX_REPORT_CHARS
default. This matches the exact methodology used to produce eval_final/.

Usage:
    python eval_final/reproduce.py [AWS|Azure|GCP]   # single vendor
    python eval_final/reproduce.py                    # all three
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from tools.extractor import Metadata, generate_prompt, client
from evaluate import times_match

MODEL = os.environ.get("BENCH_MODEL", "gpt-4o-mini")

VENDOR_FILES = {
    "AWS": "label_data/aws_label.csv",
    "Azure": "label_data/azure_label.csv",
    "GCP": "label_data/gcp_label.csv",
}

CAT_FIELDS = ["service_category", "user_symptom_category", "root_cause_category"]
TIME_FIELDS = ["start_time", "end_time"]
SKIP = {"UNKNOWN", "NAN", ""}


def build_report(row: pd.Series) -> str:
    parts = []
    if "service" in row and str(row["service"]) not in ("", "nan"):
        parts.append(f"service: {row['service']}")
    if "status" in row and str(row["status"]) not in ("", "nan"):
        parts.append(f"status: {row['status']}")
    parts.append(f"description: {row['description']}")  # full text, no truncation
    if "external_description" in row and str(row["external_description"]) not in ("", "nan"):
        parts.append(f"external_description: {row['external_description']}")
    return "\n".join(parts)


def extract(system_prompt: str, report: str, max_retries: int = 5) -> dict:
    t0 = time.time()
    for attempt in range(max_retries):
        try:
            resp = client.beta.chat.completions.parse(
                model=MODEL,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Incident report:\n{report}"},
                ],
                response_format=Metadata,
            )
            break
        except Exception as e:
            retryable = "429" in str(e) or "Connection error" in str(e) or "ConnectError" in str(e)
            if retryable and attempt < max_retries - 1:
                time.sleep(2 ** (attempt + 1))
            else:
                raise
    rec = resp.choices[0].message.parsed.model_dump()
    if isinstance(rec.get("user_symptom_category"), list):
        rec["user_symptom_category"] = ",".join(rec["user_symptom_category"])
    rec["time_sec"] = round(time.time() - t0, 3)
    return rec


def norm(v) -> str:
    return str(v).strip().upper().replace("UNVAIL", "UNAVIL")


def cat_match(pred: str, gt: str) -> bool:
    return {v.strip() for v in pred.upper().split(",")} == {v.strip() for v in gt.upper().split(",")}


def run_vendor(vendor: str) -> None:
    system_prompt = generate_prompt()  # reads current prompt/*.txt files live
    df = pd.read_csv(VENDOR_FILES[vendor])
    prefix = "label_" if any(c.startswith("label_") for c in df.columns) else ""
    tz_col = f"{prefix}timezone"

    rows = []
    print(f"\n=== {vendor} (n={len(df)}, model={MODEL}) ===")
    for i, row in df.iterrows():
        pred = extract(system_prompt, build_report(row))
        result = {"idx": i, "time_sec": pred["time_sec"]}
        for f in CAT_FIELDS:
            lf = f"{prefix}{f}"
            if lf not in df.columns:
                continue
            gt = norm(str(row[lf]))
            pv = norm(str(pred.get(f, "")))
            result[f"match_{f}"] = cat_match(pv, gt) if gt not in SKIP else None
        tz = str(row[tz_col]).strip() if tz_col in df.columns else ""
        for f in TIME_FIELDS:
            lf = f"{prefix}{f}"
            if lf not in df.columns:
                continue
            gt = str(row[lf])
            result[f"match_{f}"] = times_match(str(pred.get(f, "")), gt, tz) if norm(gt) not in SKIP else None
        rows.append(result)
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(df)}]")

    out = pd.DataFrame(rows)
    lat = out["time_sec"]
    print(f"latency mean={lat.mean():.2f}s median={lat.median():.2f}s p90={lat.quantile(0.9):.2f}s")
    for f in CAT_FIELDS + TIME_FIELDS:
        col = f"match_{f}"
        if col not in out.columns:
            continue
        vals = out[col].dropna()
        acc = vals.mean() * 100 if len(vals) else float("nan")
        print(f"  {f:<22} {vals.sum():.0f}/{len(vals)}  ({acc:.1f}%)")


if __name__ == "__main__":
    vendors = [sys.argv[1]] if len(sys.argv) > 1 else list(VENDOR_FILES)
    for v in vendors:
        run_vendor(v)
