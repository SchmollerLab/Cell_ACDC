from cellacdc.myutils import model_installs

# micro-sam 2.0.0 does not install `sam2` (see `MODEL_ENVS['micro-sam_v2']`
# in `cellacdc/models/_env_config.py`)
model_installs.check_install_microsam(min_version='2.0.1')
