import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

import warnings
warnings.filterwarnings('ignore')

def process_mttr_data(df, dataset_name):
    """Process MTTR data for a given dataframe"""
    # handle 'UNKNOWN' values in start_time and end_time
    df['start_time'] = df['start_time'].replace('UNKNOWN', np.nan)
    df['end_time'] = df['end_time'].replace('UNKNOWN', np.nan)
    
    # transform start_time and end_time to datetime
    df['start_time'] = pd.to_datetime(df['start_time'])
    df['end_time'] = pd.to_datetime(df['end_time'])
    
    # calculate mttr
    df['mttr'] = df['end_time'] - df['start_time']
    # transform mttr to hours
    df['mttr'] = df['mttr'] / np.timedelta64(1, 'h')

    # select mttr larger than 0
    df = df[df['mttr'] > 0]
    
    # exclude outlier values in mttr larger than 3 sigma
    df = df[df['mttr'] < df['mttr'].mean() + 3 * df['mttr'].std()]
    
    # add dataset label
    df['dataset'] = dataset_name
    
    return df

df_azure = pd.read_csv("cli_evaluation_log.csv")
df_azure = pd.read_csv("cli_evaluation_log.csv")

# handle missing values if needed
df_azure['start_time'] = pd.to_datetime(df_azure['start_time'], errors='coerce')
df_azure['end_time'] = pd.to_datetime(df_azure['end_time'], errors='coerce')

# compute MTTR in hours
df_azure['mttr'] = (df_azure['end_time'] - df_azure['start_time']) / pd.Timedelta(hours=1)

# clean invalid values
df_azure = df_azure[df_azure['mttr'] > 0]

df_azure['dataset'] = 'AZURE'

# make sure MTTR is numeric (if needed)
df_azure['mttr'] = pd.to_numeric(df_azure['mttr'], errors='coerce')
df_azure = df_azure[df_azure['mttr'] > 0]

# add dataset label (since boxplot needs grouping)
df_azure['dataset'] = 'AZURE'

# plot box plot
fig, ax = plt.subplots(1, 1, figsize=(8, 3))

sns.boxplot(
    data=df_azure,
    x='mttr',
    y='dataset',
    orient='h',
    width=0.6,
    palette='Set1',
    linecolor='black'
)

ax.set_xlabel(r'MTTR [hours]', fontsize=18, fontweight='bold')
ax.set_ylabel('')
ax.grid(axis='both', linestyle='--', alpha=0.6, which='both')

ax.set_xscale('log')

ax.tick_params(axis='both', which='major', labelsize=18)
ax.tick_params(axis='both', which='minor', labelsize=16)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontweight('bold')

# ❗ FIX: log scale cannot use 0
ax.set_xlim(0.1, 100)

# reference lines
ref_lines = [
    (30/60, '0.5h'),
    (1, '1h'),
    (3, '3h'),
    (10, '10h'),
    (24, '24h')
]

for x, label in ref_lines:
    ax.axvline(x=x, color='black', linestyle='--', linewidth=1, alpha=0.7)
    ax.text(x, -0.5, label,
            color='black',
            fontsize=18,
            fontweight='bold',
            ha='center',
            va='top')

# median (single dataset now)
median = df_azure['mttr'].median()
ax.text(median, 'AZURE', f'{median:.2f}h',
        color='black',
        fontsize=18,
        ha='center',
        va='center',
        fontweight='bold')

plt.tight_layout()


plt.show()