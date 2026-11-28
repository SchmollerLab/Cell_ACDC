from cellacdc.myutils import model_installs
model_installs.check_install_cellpose(4)

class AvailableModelsv4:
    from cellpose.models import MODEL_NAMES
    values = MODEL_NAMES
    
    is_exclusive_with = ['model_path']
    default_exclusive = 'Using custom model'