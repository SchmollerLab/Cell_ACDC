import os
import re

import pandas as pd
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

while True:
    answer = input('Do you want to save the plot (y/n)? ')
    if answer == 'y':
        save_plot = True
        break
    elif answer == 'n':
        save_plot = False
        break
    elif answer == 'q':
        exit('Process cancelled.')
    else:
        print(
            f'{answer} is not a valid answer. '
            'Enter "y" for "yes", "n" for "no", or "q" to exit the process.'
        )

cwd_path = os.path.dirname(os.path.abspath(__file__))
docs_source_path = os.path.dirname(cwd_path)
publications_rst_filepath = os.path.join(docs_source_path, 'publications.rst')

# Load data
with open(publications_rst_filepath, 'r') as rst:
    text = rst.read()

publications = re.findall(r"<li>(.*?)</li>", text, flags=re.DOTALL)

years = []
for publication in publications:
    year = int(re.search(r"\((\d{4})\)", publication).group(1))
    journal = re.search(r"<b>([A-Za-z \.]+)</b>", publication).group(1)
    years.append(year)
    # if journal != 'bioRxiv':
    #     years.append(year)

df = (
    pd.DataFrame({'year': years})
    .groupby('year')
    .size()
    .to_frame(name='citations')
    .reset_index()
)

# Calculate maximum of yticks
yticks_step = 5
yticks_max = np.ceil(df['citations'].max() / yticks_step) * yticks_step
yticks_max = int(yticks_max)

# Plot
font = {'size': 13}
matplotlib.rc('font', **font)

fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(df['year'], df['citations'], color='forestgreen')
ax.set_xlabel('Year')
ax.set_ylabel('Citations')
ax.set_title('Publications using Cell-ACDC')
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

if save_plot:
    # Save image
    filename = os.path.basename(os.path.abspath(__file__))
    filename, ext = os.path.splitext(filename)
    png_out_path = os.path.join(cwd_path, f'{filename}.png')
    plt.savefig(png_out_path, dpi=300)
    plt.close()

plt.show()

