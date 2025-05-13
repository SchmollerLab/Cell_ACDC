from cellacdc import printl

def init_directML():    
    success = True
    try:
        import torch_directml
    except ImportError:
        printl(
            """DirectML GPU support is not available.
            Please install torch-directml.
            Note that this is only available for Windows OS
            and python <=3.11.
            Defaulting to CPU or normal GPU (if GPU was enabled)."""
        )
        success = False
    return success

def setup_custom_device(acdc_cp_model, device):
    acdc_cp_model.model.gpu = True
    acdc_cp_model.device = device
    acdc_cp_model.mkldnn = False
    if hasattr(acdc_cp_model.model, 'cp'):
        acdc_cp_model.model.cp.gpu = True
        acdc_cp_model.model.cp.device = device
        acdc_cp_model.model.cp.mkldnn = False
        acdc_cp_model.model.cp.net.to(device)
        acdc_cp_model.model.cp.net.mkldnn = False
    if hasattr(acdc_cp_model.model, 'net'):
        acdc_cp_model.model.net.to(device)
        acdc_cp_model.model.net.mkldnn = False
    if hasattr(acdc_cp_model.model, 'sz'):
        acdc_cp_model.model.sz.device = device

def setup_directML(acdc_cp_model):
    printl(
        'Using DirectML GPU for Cellpose model inference'
    )
    import torch_directml
    directml_device = torch_directml.device()
    setup_custom_device(acdc_cp_model, directml_device)