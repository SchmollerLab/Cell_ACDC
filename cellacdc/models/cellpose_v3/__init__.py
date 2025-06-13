import cellacdc.myutils as myutils
myutils.check_install_cellpose(3)

class AvailableModelsv3:
    from cellpose.models import MODEL_NAMES
    values = MODEL_NAMES
    
    is_exclusive_with = ['model_path']
    default_exclusive = 'Using custom model'