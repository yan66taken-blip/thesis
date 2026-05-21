from langchain_core.tools import tool
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import os
import platform

warnings.filterwarnings('ignore')

STORE_PATH = "cli_evaluation_log.csv"
BAR_COLOR = plt.cm.Set1.colors[1]


def _open_image(path: str):
    system = platform.system()
    if system == "Darwin":
        os.system(f"open '{path}'")
    elif system == "Windows":
        os.startfile(path)
    else:
        os.system(f"xdg-open '{path}'")


def _load_service_counts() -> pd.Series:
    """Shared helper: loads CSV and returns sorted service category counts."""
    if not os.path.exists(STORE_PATH):
        raise FileNotFoundError("No data found. Please ingest a report first.")

    df = pd.read_csv(STORE_PATH)

    if df.empty:
        raise ValueError("Store is empty. Please ingest a report first.")

    df['service_category'] = df['service_category'].str.upper()
    return df['service_category'].value_counts()


def _style_ax(ax):
    """Shared helper: applies common axis styling."""
    ax.tick_params(axis="both", which="major", labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.grid(axis="x", linestyle="--", alpha=0.7)

@tool
def plot_service_category_percent(output_path: str = "service_category_percent.png") -> str:
    """
    Loads all saved incident records from the CSV store, plots a horizontal
    bar chart of service category percentage distribution, saves and opens
    the chart automatically.

    Args:
        output_path: File path where the chart will be saved.

    Returns:
        A message with the saved path and top category by percentage.
    """
    try:
        counts = _load_service_counts()
    except (FileNotFoundError, ValueError) as e:
        return str(e)

    percent = (counts / counts.sum() * 100).sort_values()

    fig, ax = plt.subplots(figsize=(6, 6))

    percent.plot(
        kind="barh",
        width=0.8,
        color=BAR_COLOR,
        alpha=0.8,
        ax=ax
    )

    ax.set_xlabel("Percentage (%)", fontweight="bold", fontsize=12)
    ax.set_ylabel("")

    _style_ax(ax)

    for p in ax.patches:
        ax.annotate(
            f"{p.get_width():.2f}%",
            (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2.),
            ha="left", va="center", fontsize=12, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    _open_image(output_path)

    top_cat = percent.idxmax()
    top_pct = percent.max()
    return f"Percent chart saved to '{output_path}'. Top category: {top_cat} ({top_pct:.2f}%)."