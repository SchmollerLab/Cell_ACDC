import os
import re

LIGHT_TO_DARK_MAPPER = {
    '#666666': '#9a9a9a',
    '#4d4d4d': '#b3b3b3',
}

resources_folderpath = os.path.dirname(os.path.abspath(__file__))
cellacdc_path = os.path.dirname(resources_folderpath)
resources_filepath = os.path.join(cellacdc_path, 'resources.qrc')

with open(resources_filepath, 'r') as resources_file:
    resources_txt = resources_file.read()

resources_dark_txt = resources_txt
svg_relpaths = re.findall(r'<file alias=".+\.svg">(.+)</file>', resources_txt)

for svg_relpath in svg_relpaths:
    svg_relpath.replace('\\', '/')
    svg_relpath_parts = svg_relpath.split('/')
    svg_abspath = os.path.join(cellacdc_path, *svg_relpath_parts)
    with open(svg_abspath, 'r') as svg_file:
        svg_text = svg_file.read()
    for light_hex, dark_hex in LIGHT_TO_DARK_MAPPER.items():
        pass

resources_dark_filepath = resources_filepath.replace(
    'resources.qrc', 'resources_dark_mode.qrc'
)
with open(resources_dark_filepath, 'w') as resources_dark_file:
    resources_dark_file.write(resources_txt)