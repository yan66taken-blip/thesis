from langchain_core.tools import tool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
import platform
from typing import Optional
import time
from tools.data_loader import load_filtered_df
from tools.plot_logger import log_plot

warnings.filterwarnings('ignore')

SQUARE = (6, 6)
RECT = (12, 5)

CATEGORIES = ['COMPUTE', 'STORAGE', 'NETWORK', 'SECURITY', 'AI',
              'MANAGEMENT', 'ANALYTICS', 'DATABASE', 'OTHERS', 'UNKNOWN']
COLORS = plt.cm.tab10.colors


def _open_image(path: str):
    system = platform.system()
    if system == "Darwin":
        os.system(f"open -g '{path}'")
    elif system == "Windows":
        os.startfile(path)
    else:
        os.system(f"xdg-open '{path}'")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['service_category'] = df['service_category'].str.upper().apply(
        lambda x: x if x in CATEGORIES else 'OTHERS'
    )
    return df


@tool
def plot_servicecategory_pie(
    output_path: str = "servicecategory_pie.png",
    vendors: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
) -> str:
    """
    Plots a square pie chart of service category distribution.
    Optionally filter by vendors (e.g. ["Azure", "AWS"]) and/or years (e.g. [2023]).
    """
    t0 = time.time()
    try:
        df = load_filtered_df(vendors=vendors, years=years)
    except (FileNotFoundError, ValueError) as e:
        return str(e)

    df = _clean(df)
    counts = df['service_category'].value_counts()
    colors = [COLORS[CATEGORIES.index(c) % len(COLORS)] if c in CATEGORIES else '#cccccc'
              for c in counts.index]

    fig, ax = plt.subplots(figsize=SQUARE)
    wedges, texts, autotexts = ax.pie(
        counts.values,
        labels=counts.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor='white', linewidth=1.2),
    )
    for t in texts + autotexts:
        t.set_fontsize(10)
        t.set_fontweight('bold')

    ax.set_title('Service Category Distribution', fontsize=13, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    _open_image(output_path)
    log_plot("plot_servicecategory_pie", vendors, years, output_path, len(df), time.time() - t0)

    breakdown = ", ".join(f"{c}: {counts[c]/counts.sum()*100:.1f}%" for c in counts.index)
    return f"Plot saved to '{output_path}'. {breakdown}"


@tool
def plot_servicecategory_bar(
    output_path: str = "servicecategory_bar.png",
    vendors: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
) -> str:
    """
    Plots a square horizontal bar chart of service category percentage distribution.
    Optionally filter by vendors (e.g. ["GCP"]) and/or years (e.g. [2022, 2023]).
    """
    t0 = time.time()
    try:
        df = load_filtered_df(vendors=vendors, years=years)
    except (FileNotFoundError, ValueError) as e:
        return str(e)

    df = _clean(df)
    counts = df['service_category'].value_counts()
    pct = (counts / counts.sum() * 100).sort_values()
    colors = [COLORS[CATEGORIES.index(c) % len(COLORS)] if c in CATEGORIES else '#cccccc'
              for c in pct.index]

    fig, ax = plt.subplots(figsize=SQUARE)
    bars = ax.barh(pct.index, pct.values, color=colors, edgecolor='black', height=0.6)
    for bar, val in zip(bars, pct.values):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, pct.max() + 12)
    ax.set_title('Service Category Distribution', fontsize=13, fontweight='bold', pad=8)
    ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.tick_params(labelsize=11)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    _open_image(output_path)
    log_plot("plot_servicecategory_bar", vendors, years, output_path, len(df), time.time() - t0)

    breakdown = ", ".join(f"{c}: {v:.1f}%" for c, v in pct.sort_values(ascending=False).items())
    return f"Plot saved to '{output_path}'. {breakdown}"


@tool
def plot_servicecategory_stacked(
    output_path: str = "servicecategory_stacked.png",
    vendors: Optional[list[str]] = None,
    years: Optional[list[int]] = None,
) -> str:
    """
    Plots a rectangular 100% stacked bar chart comparing service category distribution across vendors.
    Optionally filter by vendors (e.g. ["Azure", "GCP", "AWS"]) and/or years (e.g. [2023]).
    """
    t0 = time.time()
    try:
        df = load_filtered_df(vendors=vendors, years=years)
    except (FileNotFoundError, ValueError) as e:
        return str(e)

    df = _clean(df)
    vendor_list = sorted(df['vendor'].unique())

    matrix = pd.DataFrame(index=vendor_list, columns=CATEGORIES, data=0.0)
    for vendor in vendor_list:
        vdf = df[df['vendor'] == vendor]
        counts = vdf['service_category'].value_counts()
        total = counts.sum()
        for cat, cnt in counts.items():
            if cat in matrix.columns:
                matrix.loc[vendor, cat] = cnt / total * 100

    fig, ax = plt.subplots(figsize=RECT)
    left = np.zeros(len(vendor_list))
    for i, cat in enumerate(CATEGORIES):
        values = matrix[cat].values.astype(float)
        has_data = values.sum() > 0
        if not has_data:
            continue
        bars = ax.barh(vendor_list, values, left=left,
                       color=COLORS[i % len(COLORS)], edgecolor='white',
                       linewidth=0.8, label=cat)
        for bar, val, l in zip(bars, values, left):
            if val >= 7:
                ax.text(l + val / 2, bar.get_y() + bar.get_height() / 2,
                        f'{val:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold')
        left += values

    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage (%)', fontsize=13, fontweight='bold')
    ax.set_title('Service Category Distribution by Vendor', fontsize=13, fontweight='bold', pad=8)
    ax.legend(loc='lower right', fontsize=9, ncol=2)
    ax.tick_params(labelsize=11)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight('bold')
    ax.grid(axis='x', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    _open_image(output_path)
    log_plot("plot_servicecategory_stacked", vendors, years, output_path, len(df), time.time() - t0)
    return f"Plot saved to '{output_path}'. Vendors compared: {', '.join(vendor_list)}"
