from cellacdc import myutils

myutils.check_install_baby()

from baby import modelsets
meta = modelsets.meta()

BABY_MODELS = list(meta.keys())