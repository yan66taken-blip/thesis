import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import warnings
warnings.filterwarnings('ignore')

df_azure = pd.read_csv("cli_evaluation_log.csv")

# Group categories - replace unknown categories with 'OTHERS'
df_azure['root_cause_category'] = df_azure['root_cause_category'].apply(
    lambda x: x if x in ['CONFIG', 'OVERLOAD', 'DEPLOY', 'EXTERNAL', 'MAINTAIN', 'OTHERS', 'UNKNOWN'] else 'OTHERS'
)

root_cause = df_azure['root_cause_category'].value_counts()

# Colors (using Set3 colormap)
colors = plt.cm.Set3.colors[:len(root_cause)]

# Plot pie chart
fig, ax = plt.subplots(figsize=(6,6))
wedges, texts, autotexts = ax.pie(
    root_cause, 
    labels=root_cause.index,        # category names
    autopct="%1.2f%%",              # percentage inside slices
    # startangle=90,                  # rotate so largest slice is at top
    colors=colors,
    textprops=dict(color="black", fontsize=17, fontweight="bold")
)

# Optional: improve label visibility
for text in texts + autotexts:
    text.set_fontsize(17)
    text.set_fontweight("bold")

# ax.set_title("Root Cause Category Distribution (Pie Chart)")

plt.tight_layout()
plt.show()


# save figure pdf and png
os.makedirs(f'{project_root}/results/figures/figure_root_cause', exist_ok=True)
plt.savefig(f'{project_root}/results/figures/figure_root_cause/figure_root_cause_analysis.pdf', format='pdf', dpi=300)
plt.savefig(f'{project_root}/results/figures/figure_root_cause/figure_root_cause_analysis.png', format='png', dpi=300)
plt.show()

## stacked bar plot

df_azure = load_extraction_for_analysis_jsonl(round_time="anl", dataset="azure", model_abbr="gemini-2.0", model_name="gemini-2.0-flash", prompt_type="1")

# Group categories - replace unknown categories with 'OTHERS'
df_azure['root_cause_category'] = df_azure['root_cause_category'].apply(
    lambda x: x if x in ['CONFIG', 'OVERLOAD', 'DEPLOY', 'EXTERNAL', 'MAINTAIN', 'OTHERS', 'UNKNOWN'] else 'OTHERS'
)

counts = df_azure['root_cause_category'].value_counts()
df = pd.DataFrame({'root_cause_category': counts.index, 'Count': counts.values})
# set root cause category as index
df = df.set_index('root_cause_category')

df["Percent"] = df["Count"] / df["Count"].sum() * 100
df["Cumulative"] = df["Percent"].cumsum()  # cumulative sum for positioning

# Plot a 100% stacked bar chart
fig, ax = plt.subplots(figsize=(12, 3.5))

# Draw stacked bar
ax.barh(
    y=["Root Cause"], 
    width=df["Percent"], 
    left=df["Percent"].cumsum().shift(fill_value=0), 
    color=plt.cm.Set3.colors[:len(df)], 
    edgecolor="white"
)

# Add labels inside segments
for i, (cat, row) in enumerate(df.iterrows()):
    # Adjust vertical position to avoid overlap
    y_pos = 0
    if cat in ["MAINTAIN", "OTHERS", "UNKNOWN", "EXTERNAL"]:
        # Move these labels slightly up or down to avoid overlap
        if cat == "MAINTAIN":
            y_pos = 0.25
        elif cat == "OTHERS":
            y_pos = -0.25
        elif cat == "UNKNOWN":
            y_pos = 0.00
        elif cat == "EXTERNAL":
            y_pos = -0.1
    
    ax.text(
        row["Cumulative"] - row["Percent"]/2, y_pos,  # position in middle of each segment
        f"{cat}\n{row['Percent']:.2f}%", 
        ha="center", va="center", fontsize=22, fontweight="bold"
    )

# Clean up chart
ax.set_xlim(0, 100)
ax.set_xticks(range(0, 101, 20))
ax.tick_params(axis="x", labelsize=22)
# Make tick labels bold
for label in ax.get_xticklabels():
    label.set_fontweight("bold")
ax.set_xlabel("Percentage (%)", fontsize=22, fontweight="bold")
ax.set_yticks([])
# ax.set_title("Root Cause Category Distribution (100% Stacked Bar)")

plt.tight_layout()
plt.show()

# save figure pdf and png
plt.savefig(f'{project_root}/results/figures/figure_root_cause/figure_root_cause_analysis_stacked_bar.pdf', format='pdf', dpi=300)
plt.savefig(f'{project_root}/results/figures/figure_root_cause/figure_root_cause_analysis_stacked_bar.png', format='png', dpi=300)
plt.show()