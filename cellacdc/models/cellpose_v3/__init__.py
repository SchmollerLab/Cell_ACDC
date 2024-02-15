from cellacdc import myutils

myutils.check_install_cellpose('3.0')

from cellpose.models import MODEL_NAMES
CELLPOSE_V3_MODELS = MODEL_NAMES