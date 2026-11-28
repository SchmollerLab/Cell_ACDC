import os

from cellacdc.myutils import downloads, model_installs

model_installs.check_install_tapir()
_, model_path = downloads.get_model_path('TAPIR', create_temp_dir=False)
TAPIR_CHECKPOINT_PATH = os.path.join(model_path, 'tapir_checkpoint.npy')