"""C12: test_headless_server_config_import runs a script with Qt blocked
that imported myutils.py (all of its code) and checked that the GUI modules
widgets and apps stay unloaded and that myutils binds config only in
server mode. The rewrite turns the script's imports into new paths but
keeps its old checks (it reports them). The same checks for the package
layout: every module of the myutils package is imported (what importing
myutils.py did), segm_models binds config only in server mode (the module
that uses it), and no module of gui/, shared/widgets/ or shared/dialogs/
is loaded."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

SERVER = 'tests/test_model_server.py'


def main(repo):
    replace(repo, SERVER, '''sys.meta_path.insert(0, NoQt())
import cellacdc
from cellacdc import myutils
from cellacdc import config
assert not cellacdc.GUI_INSTALLED
assert cellacdc.RUNNING_AS_SERVER == ('--server' in sys.argv)
assert hasattr(myutils, 'config') == cellacdc.RUNNING_AS_SERVER
assert 'cellacdc.widgets' not in sys.modules
assert 'cellacdc.apps' not in sys.modules
if cellacdc.RUNNING_AS_SERVER:
    assert config.ConfigParser().sections() == []
''', '''sys.meta_path.insert(0, NoQt())
import importlib
import pkgutil
import cellacdc
import cellacdc.myutils
# every module of the myutils package (the code of the old myutils.py)
for module_info in pkgutil.iter_modules(cellacdc.myutils.__path__):
    importlib.import_module('cellacdc.myutils.' + module_info.name)
from cellacdc.myutils import segm_models
assert not cellacdc.GUI_INSTALLED
assert cellacdc.RUNNING_AS_SERVER == ('--server' in sys.argv)
assert hasattr(segm_models, 'config') == cellacdc.RUNNING_AS_SERVER
gui_modules = ('cellacdc.gui', 'cellacdc.shared.widgets',
               'cellacdc.shared.dialogs')
assert not [name for name in sys.modules if name.startswith(gui_modules)]
if cellacdc.RUNNING_AS_SERVER:
    assert segm_models.config.ConfigParser().sections() == []
''', items=[SERVER + '::test_headless_server_config_import'])


run(main)
