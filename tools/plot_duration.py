from langchain_core.tools import tool
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import platform
 
warnings.filterwarnings('ignore')
 
STORE_PATH = "cli_evaluation_log.csv"
 
 
def _open_image(path: str):
    system = platform.system()
    if system == "Darwin":
        os.system(f"open '{path}'")
    elif system == "Windows":
        os.startfile(path)
    else:
        os.system(f"xdg-open '{path}'")
 
 
@tool
def plot_time_duration(output_path: str = "duration.png") -> str:
    """
    Loads all saved incident records from the CSV store, plots a horizontal
    log-scale MTTR boxplot, saves and opens the chart automatically.
 
    Args:
        output_path: File path where the chart will be saved.
 
    Returns:
        A message with the saved path and median MTTR.
    """
 
    if not os.path.exists(STORE_PATH):
        return "No data found. Please ingest a report first."
 
    df = pd.read_csv(STORE_PATH)
 
    if df.empty:
        return "Store is empty. Please ingest a report first."
 
    # Handle UNKNOWN values
    df['start_time'] = df['start_time'].replace('UNKNOWN', np.nan)
    df['end_time'] = df['end_time'].replace('UNKNOWN', np.nan)
 
    # Parse datetime
    df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')
 
    # Compute MTTR in hours
    df['mttr'] = (df['end_time'] - df['start_time']) / np.timedelta64(1, 'h')
    df['mttr'] = pd.to_numeric(df['mttr'], errors='coerce')
 
    # Filter invalid and remove 3-sigma outliers
    df = df[df['mttr'] > 0]
    df = df[df['mttr'] < df['mttr'].mean() + 3 * df['mttr'].std()]
    df['dataset'] = 'AZURE'
 
    if df.empty:
        return "No valid MTTR data to plot after cleaning."
 
    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
 
    sns.boxplot(
        data=df, x='mttr', y='dataset',
        orient='h', width=0.6, palette='Set1', linecolor='black'
    )
 
    ax.set_xlabel('MTTR [hours]', fontsize=18, fontweight='bold')
    ax.set_ylabel('')
    ax.grid(axis='both', linestyle='--', alpha=0.6, which='both')
    ax.set_xscale('log')
    ax.set_xlim(0.1, 100)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.tick_params(axis='both', which='minor', labelsize=16)
 
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
 
    ref_lines = [(30/60, '0.5h'), (1, '1h'), (3, '3h'), (10, '10h'), (24, '24h')]
    for x, label in ref_lines:
        ax.axvline(x=x, color='black', linestyle='--', linewidth=1, alpha=0.7)
        ax.text(x, -0.5, label, color='black', fontsize=18,
                fontweight='bold', ha='center', va='top')
 
    median = df['mttr'].median()
    # Use numeric y-index (0) instead of string label for reliable positioning
    ax.text(median, 0, f'{median:.2f}h',
            color='black', fontsize=18, ha='center',
            va='center', fontweight='bold')
 
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
 
    _open_image(output_path)
 
    return f"Plot saved to '{output_path}'. Median MTTR: {median:.2f}h | Records plotted: {len(df)}"