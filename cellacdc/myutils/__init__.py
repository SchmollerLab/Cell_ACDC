"""Helper library of Cell-ACDC without windows of its own (it replaces the
single module myutils.py), split by theme: filesystem.py (experiments,
files, recent paths), app_info.py (version, git, logging), installs.py
(pip/conda, check_install_package), model_installs.py
(check_install_<model>, GPU checks), segm_models.py (model and tracker
registry, ArgSpec, init), downloads.py (URLs, model weights, manual,
ffmpeg), image_utils.py (dtype conversions, TIFF), table_utils.py
(acdc_df helpers), misc.py (small Python helpers), model_modules.py
(Qt-free model introspection, was myutils_models.py) and torch_utils.py
(PyTorch install and device helpers, was _torch_utils.py). Helpers with
a single program part as user live next to it: gui/helpers.py,
data_structure/helpers.py, launcher/main_window.py (self-update),
fiji_macros/__init__.py (Fiji), qutils.py (setRetainSizePolicy),
help/about.py, help/welcome.py and the utilities in utils/.

Every module imports without Qt: Qt classes and the shared message box
and push buttons are imported under if GUI_INSTALLED, dialogs only
inside the functions that open them.

This __init__ imports no submodule. Its __getattr__ forwards the plugin
API (the names that custom models, trackers and promptable models use as
myutils.<name>, listed in the PLUGIN_API table of _compat_moves.py)
without a warning, imports a submodule when its name is accessed, and
forwards every other name of the old myutils.py with a MovedWarning
(DeprecationWarning). Cell-ACDC code imports the submodule that defines
a name and never uses the forwarding.
"""

from cellacdc import _compat as _acdc_compat

__getattr__, __dir__ = _acdc_compat.package_compat(__name__)
