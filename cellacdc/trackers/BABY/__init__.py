from cellacdc import utils

utils.check_install_baby()

from baby import modelsets

meta = modelsets.meta()

BABY_MODELS = list(meta.keys())
