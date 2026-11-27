"""C12: test_torch_utils_import_does_not_load_gui_or_torch blocks the GUI
modules while importing the torch helpers in a subprocess; the rewrite
moved the script to cellacdc.myutils.torch_utils but keeps the blocked old
names widgets and apps (shims since C5/C6, it reports them). Block the GUI
packages of the new layout instead (gui/, shared/widgets/,
shared/dialogs/)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

TORCH = 'tests/test_torch_version.py'


def main(repo):
    replace(repo, TORCH, '''        if fullname.split('.')[0] in {'torch', 'torchvision', 'qtpy'} or fullname in {
            'cellacdc.apps', 'cellacdc.widgets',
            'cellacdc.service.environment',
        } or (
''', '''        if fullname.split('.')[0] in {'torch', 'torchvision', 'qtpy'} or fullname in {
            'cellacdc.service.environment',
        } or fullname.startswith((
            'cellacdc.gui', 'cellacdc.shared.widgets', 'cellacdc.shared.dialogs',
        )) or (
''', items=[TORCH + '::test_torch_utils_import_does_not_load_gui_or_torch'])


run(main)
