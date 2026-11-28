from cellacdc.myutils import model_installs

model_installs.check_install_baby()

from baby import modelsets
meta = modelsets.meta()

BABY_MODELS = list(meta.keys())