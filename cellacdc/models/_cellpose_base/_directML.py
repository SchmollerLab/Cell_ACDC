from cellacdc import printl
from cellacdc.myutils import check_install_package
import sys

def init_directML():    
    success = True
    try:
        import torch_directml
    except ImportError:
        py_ver = sys.version_info
        #check windows
        from cellacdc import is_win
        if is_win and py_ver.major == 3 and py_ver.minor < 13:
            success = check_install_package(
                pkg_name = 'torch-directml',
                import_pkg_name = 'torch_directml',
                pypi_name = 'torch-directml',
                return_outcome=True,
            )
        else:
            print(
                """DirectML GPU support is not available.
                Please install torch-directml.
                Note that this is only available for Windows OS
                and python <3.13.
                Defaulting to CPU or normal GPU (if GPU was enabled)."""
            )
            success = False
    return success

def setup_custom_device(model, device):
    """
    Forces the model to use a custom device (e.g., DirectML) for inference.
    This is a workaround, and could be handled better in the future. 
    (Ideally when all parameters are set initially)

    Args:
        model (cellpose.CellposeModel|cellpse.Cellpose): Cellpose model. Should work for v2, v3 and custom.
        torch.device (torch.device): Custom device.

    Returns:
        model (cellpose.CellposeModel): Cellpose model with custom device set.
    """
    if hasattr(model, 'model'):
        model = model.model
        
    model.gpu = True
    model.device = device
    model.mkldnn = False
    if hasattr(model, 'net'):
        model.net.to(device)
        model.net.mkldnn = False
    if hasattr(model, 'cp'):
        model.cp.gpu = True
        model.cp.device = device
        model.cp.mkldnn = False
        if hasattr(model.cp, 'net'):
            model.cp.net.to(device)
            model.cp.net.mkldnn = False
    if hasattr(model, 'sz'):
        model.sz.device = device
    
    return model


def setup_directML(acdc_cp_model):
    """
    Sets up the Cellpose model to use DirectML for inference.

    Args:
        model (cellpose.CellposeModel|cellpse.Cellpos): Cellpose model. Should work for v2, v3 and custom.
    
    Returns:
        model (cellpose.CellposeModel|cellpse.Cellpos): Cellpose model with DirectML set as the device.
    """
    print(
        'Using DirectML GPU for Cellpose model inference'
    )
    import torch_directml
    directml_device = torch_directml.device()
    acdc_cp_model = setup_custom_device(acdc_cp_model, directml_device)
    return acdc_cp_model