import inspect
import os

from cellacdc import cellacdc_path
from cellacdc import myutils

version = '2.0'
# Check who is importing. If it's cellpose_v3 we check the correct version
for stack_item in inspect.stack():
    caller_filepath = stack_item.filename
    try:
        caller_relpath = os.path.relpath(caller_filepath, cellacdc_path)
    except Exception as err:
        continue
    if caller_relpath.find('cellpose_v3') != -1:
        version = '3.0'
        break

if version == '2.0':
    # We take care of 3.0 in cellacdc.models.cellpose_v3
    myutils.check_install_cellpose(version)
    from cellpose.models import MODEL_NAMES
    CELLPOSE_V2_MODELS = MODEL_NAMES