from langchain_core.tools import tool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Optional
import time
from tools.data_loader import load_filtered_df
from tools.plot_logger import log_plot

warnings.filterwarnings('ignore')

SQUARE = (6, 6)


def _compute_mttr(df: pd.DataFrame) -> pd.DataFrame:
    df['end_time'] = pd.to_datetime(df['end_time'].replace('UNKNOWN', np.nan), errors='coerce', utc=True)
    df['start_time'] = pd.to_datetime(df['start_time'].replace('UNKNOWN', np.nan), errors='coerce', utc=True)
    df['mttr'] = (df['end_time'] - df['start_time']) / np.timedelta64(1, 'h')
    df['mttr'] = pd.to_numeric(df['mttr'], errors='coerce')
    df = df[df['mttr'] > 0].copy()
    df = df[df['mttr'] < df['mttr'].mean() + 3 * df['mttr'].std()]
    return df


@tool
def plot_duration_box(
    output_path: str = "duration_box.png",
    vendors: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
) -> str:
    """
    Plots a square log-scale MTTR boxplot showing the distribution of incident durations.
    Optionally filter by vendors (e.g. ["AWS"]) and/or years (e.g. [2022, 2023]).
    """
    t0 = time.time()
    try:
        df = load_filtered_df(vendors=vendors, years=years)
    except (FileNotFoundError, ValueError) as e:
        return str(e)

    df = _compute_mttr(df)
    if df.empty:
        return "No valid MTTR data after cleaning."

    label = "+".join(sorted(df['vendor'].unique())) if vendors else "ALL"
    df['dataset'] = label

    fig, ax = plt.subplots(figsize=SQUARE)
    sns.boxplot(data=df, x='mttr', y='dataset', orient='h', width=0.4,
                palette='Set1', linecolor='black', ax=ax)

    ax.set_xlabel('MTTR [hours]', fontsize=13, fontweight='bold')
    ax.set_ylabel('')
    ax.set_xscale('log')
    ax.set_xlim(0.1, 100)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.tick_params(labelsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')
    for x, lbl in [(0.5, '0.5h'), (1, '1h'), (3, '3h'), (10, '10h'), (24, '24h')]:
        ax.axvline(x=x, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    median = df['mttr'].median()
    ax.set_title(f'MTTR Distribution  (median={median:.2f}h, n={len(df)})',
                 fontsize=12, fontweight='bold', pad=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log_plot("plot_duration_box", vendors, years, output_path, len(df), time.time() - t0)
    return f"Plot saved to '{output_path}'. Median MTTR: {median:.2f}h | n={len(df)}"


@tool
def plot_duration_bar(
    output_path: str = "duration_bar.png",
    vendors: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
) -> str:
    """
    Plots a square bar chart of mean MTTR per year.
    Optionally filter by vendors (e.g. ["Azure"]) and/or years (e.g. [2022, 2023]).
    """
    t0 = time.time()
    try:
        df = load_filtered_df(vendors=vendors, years=years)
    except (FileNotFoundError, ValueError) as e:
        return str(e)

    df = _compute_mttr(df)
    if df.empty:
        return "No valid MTTR data after cleaning."

    df['year'] = df['start_time'].dt.year
    yearly = df.groupby('year')['mttr'].mean().sort_index()

    fig, ax = plt.subplots(figsize=SQUARE)
    bars = ax.bar(yearly.index.astype(str), yearly.values,
                  color='#4C72B0', edgecolor='black', width=0.5)
    for bar, val in zip(bars, yearly.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + yearly.max() * 0.02,
                f'{val:.2f}h', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Year', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean MTTR [hours]', fontsize=13, fontweight='bold')
    ax.set_title('Mean MTTR per Year', fontsize=12, fontweight='bold', pad=8)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=11)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log_plot("plot_duration_bar", vendors, years, output_path, len(df), time.time() - t0)
    return f"Plot saved to '{output_path}'. Mean MTTR by year: " + \
           ", ".join(f"{y}={v:.2f}h" for y, v in yearly.items())


@tool
def plot_duration_line(
    output_path: str = "duration_line.png",
    vendors: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
) -> str:
    """
    Plots a square line chart of monthly mean MTTR trend over time.
    Optionally filter by vendors (e.g. ["GCP"]) and/or years (e.g. [2023]).
    """
    t0 = time.time()
    try:
        df = load_filtered_df(vendors=vendors, years=years)
    except (FileNotFoundError, ValueError) as e:
        return str(e)

    df = _compute_mttr(df)
    if df.empty:
        return "No valid MTTR data after cleaning."

    df['month'] = df['start_time'].dt.to_period('M')
    monthly = df.groupby('month')['mttr'].mean().sort_index()
    x = np.arange(len(monthly))
    labels = [str(p) for p in monthly.index]

    fig, ax = plt.subplots(figsize=SQUARE)
    ax.plot(x, monthly.values, marker='o', color='#C44E52', linewidth=2, markersize=5)
    ax.fill_between(x, monthly.values, alpha=0.15, color='#C44E52')

    step = max(1, len(x) // 6)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=30, ha='right', fontsize=10)
    ax.set_xlabel('Month', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean MTTR [hours]', fontsize=13, fontweight='bold')
    ax.set_title('Monthly MTTR Trend', fontsize=12, fontweight='bold', pad=8)
    ax.grid(linestyle='--', alpha=0.4)
    ax.tick_params(axis='y', labelsize=11)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    log_plot("plot_duration_line", vendors, years, output_path, len(df), time.time() - t0)
    return f"Plot saved to '{output_path}'. Trend plotted over {len(monthly)} months."
