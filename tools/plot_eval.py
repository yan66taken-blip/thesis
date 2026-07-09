from langchain_core.tools import tool
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import glob
import platform
import time
from typing import Optional
from tools.plot_logger import log_plot


MATCH_FIELDS = [
    "service_name",
    "service_category",
    "start_time",
    "end_time",
    "user_symptom_category",
    "root_cause_category",
]

FIELD_LABELS = {
    "service_name":           "Service Name",
    "service_category":       "Service Category",
    "start_time":             "Start Time",
    "end_time":               "End Time",
    "user_symptom_category":  "User Symptom Category",
    "root_cause_category":    "Root Cause Category",
}

FIELD_TYPES = {
    "service_name":          "service",
    "service_category":      "categorical",
    "start_time":            "datetime",
    "end_time":              "datetime",
    "user_symptom_category": "categorical",
    "root_cause_category":   "categorical",
}

VENDOR_COLORS = {"Azure": "#0078D4", "AWS": "#FF9900", "GCP": "#34A853"}


def _open_image(path: str):
    system = platform.system()
    if system == "Darwin":
        os.system(f"open -g '{path}'")
    elif system == "Windows":
        os.startfile(path)
    else:
        os.system(f"xdg-open '{path}'")


def _filter(df: pd.DataFrame, vendors: Optional[list], years: Optional[list]) -> pd.DataFrame:
    if vendors:
        df = df[df["vendor"].str.upper().isin([v.upper() for v in vendors])]
    if years:
        year_col = "pred_year" if "pred_year" in df.columns else "year" if "year" in df.columns else None
        if year_col:
            df = df[df[year_col].isin([float(y) for y in years])]
    return df


def _load(input_path: str, vendors: Optional[list], years: Optional[list]) -> pd.DataFrame:
    if not os.path.exists(input_path):
        return pd.DataFrame()
    df = pd.read_csv(input_path)
    return _filter(df, vendors, years)


def _load_all(eval_dir: str, vendors: Optional[list], years: Optional[list]) -> pd.DataFrame:
    paths = glob.glob(os.path.join(eval_dir, "**", "*_eval.csv"), recursive=True)
    if not paths:
        return pd.DataFrame()
    frames = [_filter(pd.read_csv(p), vendors, years) for p in paths]
    return pd.concat(frames, ignore_index=True)


def _save(fig, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    _open_image(output_path)


@tool
def plot_eval_latency(
    eval_dir: str = "eval",
    output_path: str = "eval_latency.png",
    vendors: Optional[list] = None,
    years: Optional[list] = None,
    input_path: Optional[str] = None,
) -> str:
    """
    Box plot showing extraction latency (seconds per report) distribution,
    with one box per vendor side by side in a single chart.

    Args:
        eval_dir: Directory containing per-vendor eval subdirectories (loads all *_eval.csv).
        output_path: Where to save the chart.
        vendors: Optional list of vendors to include, e.g. ['Azure', 'AWS'].
        years: Optional list of years to include.
        input_path: Optional path to a single eval CSV instead of eval_dir (single-vendor chart).
    """
    t0 = time.time()
    df = _load(input_path, vendors, years) if input_path else _load_all(eval_dir, vendors, years)
    if df.empty or "pred_time_sec" not in df.columns:
        return "No latency data found. Ensure pred_time_sec column exists."

    vendor_list = sorted(df["vendor"].dropna().unique())
    data = [df[df["vendor"] == v]["pred_time_sec"].dropna().tolist() for v in vendor_list]
    colors = [VENDOR_COLORS.get(v, f"C{i}") for i, v in enumerate(vendor_list)]

    fig, ax = plt.subplots(figsize=(max(6, len(vendor_list) * 2.5), 5))

    bp = ax.boxplot(data, labels=vendor_list, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    all_vals = pd.Series([v for d in data for v in d])
    y_cap = float(all_vals.quantile(0.95)) * 1.4
    ax.set_ylim(bottom=0, top=y_cap)

    for i, (vdata, vendor) in enumerate(zip(data, vendor_list), start=1):
        s = pd.Series(vdata)
        label = f"med={s.median():.2f}s\nmean={s.mean():.2f}s"
        ax.text(i, y_cap * 0.97, label, ha="center", va="top", fontsize=9)

    ax.set_ylabel("Latency (seconds)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    title = f"Extraction Latency Distribution  (n={len(df)})"
    if years:
        title += f"  [{', '.join(str(y) for y in years)}]"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    _save(fig, output_path)
    log_plot("plot_eval_latency", vendors, years, output_path, len(df), time.time() - t0)

    stats = ", ".join(
        f"{v}: med={pd.Series(d).median():.2f}s" for v, d in zip(vendor_list, data)
    )
    return f"Plot saved to '{output_path}'. Latency — {stats}"


@tool
def plot_eval_tokens(
    eval_dir: str = "eval",
    output_path: str = "eval_tokens.png",
    vendors: Optional[list] = None,
    years: Optional[list] = None,
    input_path: Optional[str] = None,
) -> str:
    """
    Grouped bar chart of average token usage (total vs cached), with one
    group per vendor side by side in a single chart.

    Args:
        eval_dir: Directory containing per-vendor eval subdirectories (loads all *_eval.csv).
        output_path: Where to save the chart.
        vendors: Optional list of vendors to include, e.g. ['Azure', 'AWS'].
        years: Optional list of years to include.
        input_path: Optional path to a single eval CSV instead of eval_dir (single-vendor chart).
    """
    t0 = time.time()
    df = _load(input_path, vendors, years) if input_path else _load_all(eval_dir, vendors, years)
    if df.empty or "pred_tokens_used" not in df.columns:
        return "No token data found. Ensure pred_tokens_used column exists."

    vendor_list = sorted(df["vendor"].dropna().unique())
    avg_total = [df[df["vendor"] == v]["pred_tokens_used"].mean() for v in vendor_list]
    avg_cached = [df[df["vendor"] == v]["pred_cached_tokens"].mean() for v in vendor_list]

    x = range(len(vendor_list))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(vendor_list) * 2.5), 5))

    bars1 = ax.bar([xi - width / 2 for xi in x], avg_total, width=width,
                   label="Total Tokens", color="#4C72B0", edgecolor="black")
    bars2 = ax.bar([xi + width / 2 for xi in x], avg_cached, width=width,
                   label="Cached Tokens", color="#55A868", edgecolor="black")

    for bar, val in list(zip(bars1, avg_total)) + list(zip(bars2, avg_cached)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{val:.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(list(x))
    ax.set_xticklabels(vendor_list, fontsize=12)
    ax.set_ylabel("Avg Tokens per Extraction", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=11)

    title = f"Avg Token Usage per Extraction  (n={len(df)})"
    if years:
        title += f"  [{', '.join(str(y) for y in years)}]"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    _save(fig, output_path)
    log_plot("plot_eval_tokens", vendors, years, output_path, len(df), time.time() - t0)

    stats = ", ".join(
        f"{v}: {t:.0f} total / {c:.0f} cached"
        for v, t, c in zip(vendor_list, avg_total, avg_cached)
    )
    return f"Plot saved to '{output_path}'. Token usage — {stats}"


@tool
def report_eval_accuracy(
    eval_dir: str = "eval",
    vendors: Optional[list] = None,
    years: Optional[list] = None,
) -> str:
    """
    Return per-field extraction accuracy as a markdown table across all providers.
    Fields are grouped by type: categorical, datetime, and free-text.
    Rows with missing ground truth are excluded per field.

    Args:
        eval_dir: Directory containing per-vendor eval subdirectories.
        vendors:  Optional list of vendors to include, e.g. ['Azure', 'AWS'].
        years:    Optional list of years to include, e.g. [2022, 2023].
    """
    df = _load_all(eval_dir, vendors, years)
    if df.empty:
        return f"No eval CSVs found in '{eval_dir}' (or no data after filtering)."

    fields = [f for f in MATCH_FIELDS if f"{f}_match" in df.columns]
    if not fields:
        return "No *_match columns found. Run evaluate.py first."

    vendor_list = sorted(df["vendor"].dropna().unique())
    vendor_ns = {v: len(df[df["vendor"] == v]) for v in vendor_list}
    total_n = len(df)

    def _acc(subset: pd.DataFrame, field: str):
        col = f"{field}_match"
        valid = subset[col].notna()
        n = int(valid.sum())
        if n == 0:
            return None, 0, 0
        correct = int(subset.loc[valid, col].sum())
        return correct / n * 100, correct, n

    # Build markdown table
    v_headers = [f"{v} (n={vendor_ns[v]})" for v in vendor_list]
    header = "| Field | Type | " + " | ".join(v_headers) + f" | Overall (n={total_n}) |"
    sep = "| --- | --- | " + " | ".join(["---"] * len(vendor_list)) + " | --- |"

    title_parts = [f"## Extraction Accuracy by Field and Vendor (n={total_n})"]
    if years:
        title_parts[0] += f"  [{', '.join(str(y) for y in years)}]"

    rows = [title_parts[0], "", header, sep]

    # Group fields by type for the average computation
    vendor_cat_avgs = {v: [] for v in vendor_list}
    overall_cat_avgs = []

    for f in fields:
        label = FIELD_LABELS[f]
        ftype = FIELD_TYPES.get(f, "")
        cells = [label, ftype]

        for v in vendor_list:
            pct, correct, n = _acc(df[df["vendor"] == v], f)
            if pct is None:
                cells.append("—")
            else:
                cells.append(f"{pct:.1f}%")
                if ftype == "categorical":
                    vendor_cat_avgs[v].append(pct)

        pct_all, correct_all, n_all = _acc(df, f)
        if pct_all is None:
            cells.append("—")
        else:
            cells.append(f"{pct_all:.1f}%")
            if ftype == "categorical":
                overall_cat_avgs.append(pct_all)

        rows.append("| " + " | ".join(cells) + " |")

    avg_cells = ["**Avg (categorical)**", "categorical"]
    for v in vendor_list:
        avgs = vendor_cat_avgs[v]
        avg_cells.append(f"**{sum(avgs)/len(avgs):.1f}%**" if avgs else "—")
    avg_cells.append(
        f"**{sum(overall_cat_avgs)/len(overall_cat_avgs):.1f}%**"
        if overall_cat_avgs else "—"
    )
    rows.append("| " + " | ".join(avg_cells) + " |")

    rows += [
        "",
        "> **Note:** `service_name` uses exact string matching — scores are a lower bound"
        " since phrasing differences count as incorrect. Datetime fields compare at"
        " minute granularity (HH:MM). Categorical fields use a predefined label taxonomy.",
    ]

    return "\n".join(rows)
