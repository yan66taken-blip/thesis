from langchain_core.tools import tool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import warnings
import os
import platform

warnings.filterwarnings('ignore')

STORE_PATH = "cli_evaluation_log.csv"

VALID_CATEGORIES = {'CONFIG', 'OVERLOAD', 'DEPLOY', 'EXTERNAL', 'MAINTAIN', 'OTHERS', 'UNKNOWN'}
SET3_COLORS = [
    '#8dd3c7', '#ffffb3', '#bebada', '#fb8072',
    '#80b1d3', '#fdb462', '#b3de69'
]


def _open_image(path: str):
    system = platform.system()
    if system == "Darwin":
        os.system(f"open '{path}'")
    elif system == "Windows":
        os.startfile(path)
    else:
        os.system(f"xdg-open '{path}'")


@tool
def plot_root_cause_stacked_bar(output_path: str = "root_cause_stacked_bar.png") -> str:
    """
    Loads all saved incident records from the CSV store, plots a 100% horizontal
    stacked bar chart of root cause category distribution, saves and opens the
    chart automatically.

    Labels for wide slices (>= 6% of total) are rendered inside the bar.
    Labels for narrow slices (< 6%) are placed alternately above and below
    the bar with a leader line, preventing text overlap.

    Args:
        output_path: File path where the chart will be saved.

    Returns:
        A message with the saved path and category percentages.
    """
    if not os.path.exists(STORE_PATH):
        return "No data found. Please ingest a report first."

    df = pd.read_csv(STORE_PATH)
    if df.empty:
        return "Store is empty. Please ingest a report first."

    df['root_cause_category'] = df['root_cause_category'].apply(
        lambda x: x if x in VALID_CATEGORIES else 'OTHERS'
    )

    counts = df['root_cause_category'].value_counts()
    data = pd.DataFrame({'Count': counts})
    data['Percent'] = data['Count'] / data['Count'].sum() * 100
    data['Cumulative'] = data['Percent'].cumsum()

    MIN_WIDTH_FOR_LABEL = 6.0   # slices narrower than this % get an offset label
    FONT_SIZE           = 11
    BAR_Y               = 0
    BAR_HALF            = 0.38
    OFFSET_LEVELS       = [1.05, -1.05]   # alternating above / below

    fig, ax = plt.subplots(figsize=(14, 4))

    outside_labels = []

    for i, (cat, row) in enumerate(data.iterrows()):
        left  = row['Cumulative'] - row['Percent']
        mid_x = left + row['Percent'] / 2

        ax.barh(
            y=[BAR_Y],
            width=row['Percent'],
            left=left,
            color=SET3_COLORS[i % len(SET3_COLORS)],
            edgecolor="white",
            linewidth=1.2,
            height=0.75,
        )

        if row['Percent'] >= MIN_WIDTH_FOR_LABEL:
            # Wide enough → label sits inside the slice
            ax.text(
                mid_x, BAR_Y,
                f"{cat}\n{row['Percent']:.1f}%",
                ha="center", va="center",
                fontsize=FONT_SIZE, fontweight="bold",
            )
        else:
            # Too narrow → offset with a leader line
            level_idx = len(outside_labels) % len(OFFSET_LEVELS)
            outside_labels.append((mid_x, row['Percent'], cat, OFFSET_LEVELS[level_idx]))

    for mid_x, pct, cat, y_off in outside_labels:
        y_anchor = BAR_HALF if y_off > 0 else -BAR_HALF
        ax.annotate(
            f"{cat}\n{pct:.1f}%",
            xy=(mid_x, y_anchor),
            xytext=(mid_x, y_off),
            ha="center", va="center",
            fontsize=FONT_SIZE - 1, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="#555555", lw=0.9),
        )

    ax.set_xlim(0, 100)
    ax.set_ylim(-1.8, 1.8)
    ax.set_xticks(range(0, 101, 20))
    ax.tick_params(axis="x", labelsize=12)
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    ax.set_xlabel("Percentage (%)", fontsize=12, fontweight="bold")
    ax.set_yticks([])
    ax.spines[['left', 'top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300, bbox_inches='tight')
    plt.close()

    _open_image(output_path)

    breakdown = ", ".join([f"{cat}: {row['Percent']:.2f}%" for cat, row in data.iterrows()])
    return f"Stacked bar saved to '{output_path}'. Percentages — {breakdown}"