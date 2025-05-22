import inspect
import os

from cellacdc import cellacdc_path
from cellacdc import myutils

# comment this for debugging cellpose models!

version = '2.0'
# Check who is importing. If it's cellpose_v3 we check the correct version
for stack_item in inspect.stack():
    caller_filepath = stack_item.filename
    try:
        caller_relpath = os.path.relpath(caller_filepath, cellacdc_path)
    except Exception as err:
        continue
    print(f'caller_filepath: {caller_filepath}')
    if 'core' in caller_relpath:
        continue
    try:
        version = caller_relpath.split('cellpose_v')[1][0]
        version = f'{version}.0'
    except:
        continue

if version is not None:
    myutils.check_install_cellpose(version)