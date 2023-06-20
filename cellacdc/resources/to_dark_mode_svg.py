import os
import re
import subprocess
from tqdm import tqdm

LIGHT_TO_DARK_MAPPER = {
    '#666666': '#9a9a9a',
    '#4d4d4d': '#f0f0f0',
    # '#d9d9d9': '#4d4d4d',
}


# Read resources_light.qrc file and extract SVG relative paths
resources_folderpath = os.path.dirname(os.path.abspath(__file__))
cellacdc_path = os.path.dirname(resources_folderpath)
resources_filepath = os.path.join(cellacdc_path, 'resources_light.qrc')

qrc_resources_light_path = os.path.join(cellacdc_path, 'qrc_resources_light.py')
qrc_resources_dark_path = os.path.join(cellacdc_path, 'qrc_resources_dark.py')
qrc_resources_path = os.path.join(cellacdc_path, 'qrc_resources.py')
if os.path.exists(qrc_resources_light_path):
    os.rename(qrc_resources_path, qrc_resources_dark_path)
    os.rename(qrc_resources_light_path, qrc_resources_path)
    

with open(resources_filepath, 'r') as resources_file:
    resources_txt = resources_file.read()

resources_dark_txt = resources_txt
svg_relpaths = re.findall(r'<file alias=".+\.svg">(.+)</file>', resources_txt)

# Iterate SVGs and replace colors 
for svg_relpath in tqdm(svg_relpaths, ncols=100):
    svg_relpath_parts = svg_relpath.split('/')
    svg_abspath = os.path.join(cellacdc_path, *svg_relpath_parts)
    svg_folderpath = os.path.dirname(svg_abspath)
    if 'icons' not in svg_relpath_parts:
        # Skip SVGs outside of the icons folder
        continue
    
    # Read svg files and replace colors
    with open(svg_abspath, 'r', encoding="utf8") as svg_file:
        svg_text = svg_file.read()
    for light_hex, dark_hex in LIGHT_TO_DARK_MAPPER.items():
        svg_text_dark = svg_text.replace(light_hex, dark_hex)
    
    # Save additional _dark.svg and replace them in resources_txt
    svg_dark_abspath = svg_abspath.replace('.svg', '_dark.svg')
    with open(svg_dark_abspath, 'w', encoding="utf8") as svg_file:
        svg_file.write(svg_text_dark)
    svg_relpath_dark = svg_relpath.replace('.svg', '_dark.svg')
    resources_txt = resources_txt.replace(svg_relpath, svg_relpath_dark)

# Save a new resouces_dark.qrc file
with open(qrc_resources_dark_path, 'w') as resources_file:
    resources_file.write(resources_txt)

# Compule new qrc_resources.py dark
print('Compiling the Qt resource file...')
qrc_resources_dark_filepath = os.path.join(cellacdc_path, 'qrc_resources_dark.py')
commands = ['pyrcc5', f"{qrc_resources_dark_path}", '-o', f"{qrc_resources_dark_filepath}"]
subprocess.run(commands, check=True)