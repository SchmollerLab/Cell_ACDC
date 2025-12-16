import os

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

cwd_path = os.path.dirname(os.path.abspath(__file__))

# Load data
csv_path = os.path.join(cwd_path, 'citations_per_year.csv')
df = pd.read_csv(csv_path)

# Calculate maximum of yticks
yticks_step = 5
yticks_max = np.ceil(df['citations'].max() / yticks_step) * yticks_step
yticks_max = int(yticks_max)

# Plot
font = {'size': 13}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(df['year'], df['citations'])
ax.set_xlabel('Year')
ax.set_ylabel('Citations')
ax.set_title('Citations per Year')
ax.set_xticks(df['year'])
ax.set_yticks(range(0, yticks_max+1, yticks_step))
ax.grid(
    True, 
    'major', 
    axis='y', 
    zorder=0, 
    alpha=0.7, 
    color='gray',
    linestyle='--',
)
ax.set_axisbelow(True)
plt.tight_layout()

# plt.show()
# exit()

# Save image
png_out_path = os.path.join(cwd_path, 'citations_per_year.png')
plt.savefig(png_out_path, dpi=300)
plt.close()