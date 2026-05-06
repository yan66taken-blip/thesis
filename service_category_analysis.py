import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


df_azure = pd.read_csv("cli_evaluation_log.csv")

df_azure['service_category'] = df_azure['service_category'].str.upper()

# -----------------------------
# 1. COUNT PLOT (Azure only)
# -----------------------------
azure_counts = df_azure['service_category'].value_counts()

fig, ax = plt.subplots(figsize=(12, 4))

azure_counts.sort_values().plot(
    kind="barh",
    width=0.8,
    color=plt.cm.Set1.colors[1],
    alpha=0.8,
    ax=ax
)

ax.set_xlabel("Count", fontweight="bold", fontsize=12)
ax.set_ylabel("Service Category", fontweight="bold", fontsize=12)

ax.tick_params(axis="both", which="major", labelsize=12)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

# value labels
for p in ax.patches:
    ax.annotate(
        str(int(p.get_width())),
        (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2.),
        ha="left", va="center", fontsize=11, fontweight="bold"
    )

ax.grid(axis="x", linestyle="--", alpha=0.7)

plt.tight_layout()

plt.show()


# -----------------------------
# 2. PERCENTAGE PLOT (Azure only)
# -----------------------------
azure_percent = azure_counts / azure_counts.sum() * 100
azure_percent = azure_percent.sort_values()

fig, ax = plt.subplots(figsize=(6, 6))

azure_percent.plot(
    kind="barh",
    width=0.8,
    color=plt.cm.Set1.colors[1],
    alpha=0.8,
    ax=ax
)

ax.set_xlabel("Percentage", fontweight="bold", fontsize=12)
ax.set_ylabel("")

ax.tick_params(axis="both", which="major", labelsize=12)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight("bold")

for p in ax.patches:
    ax.annotate(
        f"{p.get_width():.2f}%",
        (p.get_x() + p.get_width(), p.get_y() + p.get_height() / 2.),
        ha="left", va="center",
        fontsize=12, fontweight="bold"
    )

ax.grid(axis="x", linestyle="--", alpha=0.7)

plt.tight_layout()

plt.show()
