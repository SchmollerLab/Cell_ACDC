import os

from cellacdc import myutils

myutils.check_install_tapir()
_, model_path = myutils.get_model_path('TAPIR', create_temp_dir=False)
TAPIR_CHECKPOINT_PATH = os.path.join(model_path, 'tapir_checkpoint.npy')