"""C1 (path-independent tests, synthesis 6E): three slow tests scan the
source of the main GUI by the path of gui.py, which becomes a package in C3
(its window class moves to gui/main_window/window.py, its methods into
mixins in C10). They now find the code wherever it lives:

- test_channel_luts: `_gui_win_calls(*names)` parses the source of each
  method resolved on guiWin (inspect.getsource), not the class body of
  gui.py (a few ms instead of parsing gui.py: no longer marked slow);
- test_linked_right_image_layers: the `ParentImageItem(...)` calls are
  collected from every module of the cellacdc package and reported with
  their posix path relative to the package;
- test_list_selection_required: the scan walks the folder of
  cellacdc/__init__.py (not of widgets.py), reports posix paths, and the
  expected caller (the model selection of guiWin.autoSegm_cb) is located
  by the method's source file and lines.

The checks are unchanged on the old layout (the same calls are found).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

LUTS = 'tests/test_channel_luts.py'
LINKED = 'tests/test_linked_right_image_layers.py'
LISTS = 'tests/test_list_selection_required.py'


def channel_luts(repo):
    replace(repo, LUTS,
            '"""\n'
            'import os\n'
            'import ast\n'
            'import json\n',
            '"""\n'
            'import os\n'
            'import ast\n'
            'import inspect\n'
            'import json\n',
            items=[LUTS + '::import:import inspect'])
    replace(repo, LUTS,
            'def _gui_win_calls():\n'
            '    """{method name of `gui.guiWin`: (calls, connected)}: `calls` lists the\n'
            '    `(line, name)` of every `<...>.name(...)` call, `connected` the names of\n'
            '    the `self.<name>` passed (also through `partial`) to a `connect`."""\n'
            '    with open(gui.__file__, encoding=\'utf-8\') as file:\n'
            '        tree = ast.parse(file.read())\n'
            '    guiWin = next(\n'
            '        node for node in tree.body\n'
            '        if isinstance(node, ast.ClassDef) and node.name == \'guiWin\'\n'
            '    )\n'
            '    methods = {}\n'
            '    for method in guiWin.body:\n'
            '        if not isinstance(method, ast.FunctionDef):\n'
            '            continue\n'
            '        calls, connected = [], set()\n',
            'def _method_node(function):\n'
            '    """`ast.FunctionDef` of `function`, parsed from its own source\n'
            '    (wherever it is defined; decorators are unwrapped)."""\n'
            '    source = inspect.getsource(function)\n'
            '    if source[:1].isspace():\n'
            '        # A method: its source is indented, parse it as a class body\n'
            '        return ast.parse(\'class _Host:\\n\' + source).body[0].body[0]\n'
            '    return ast.parse(source).body[0]\n'
            '\n'
            '\n'
            'def _gui_win_calls(*names):\n'
            '    """{name: (calls, connected)} of the methods `names` as resolved on\n'
            '    `gui.guiWin` (defined in its class body or in a mixin): `calls` lists\n'
            '    the `(line, name)` of every `<...>.name(...)` call (lines relative to\n'
            '    the method), `connected` the names of the `self.<name>` passed (also\n'
            '    through `partial`) to a `connect`."""\n'
            '    methods = {}\n'
            '    for name in names:\n'
            '        method = _method_node(getattr(gui.guiWin, name))\n'
            '        calls, connected = [], set()\n',
            items=[LUTS + '::_gui_win_calls', LUTS + '::_method_node'])
    replace(repo, LUTS,
            '                    if isSelfAttr:\n'
            '                        connected.add(sub.attr)\n'
            '        methods[method.name] = (calls, connected)\n'
            '    return methods\n',
            '                    if isSelfAttr:\n'
            '                        connected.add(sub.attr)\n'
            '        methods[name] = (calls, connected)\n'
            '    return methods\n',
            items=[LUTS + '::_gui_win_calls'])
    # Parsing six methods instead of gui.py takes a few ms: push tier
    replace(repo, LUTS,
            '@pytest.mark.slow\n'
            'def test_main_gui_calls_the_channel_lut_and_overlay_mode_hooks():\n',
            'def test_main_gui_calls_the_channel_lut_and_overlay_mode_hooks():\n',
            items=[LUTS + '::test_main_gui_calls_the_channel_lut_and_overlay_mode_hooks'])
    replace(repo, LUTS,
            '    `main` into `gui.py` have dropped or moved lines before."""\n'
            '    methods = _gui_win_calls()\n',
            '    `main` into `gui.py` have dropped or moved lines before."""\n'
            '    methods = _gui_win_calls(\n'
            '        \'loadingDataCompleted\', \'reInitGui\', \'imgGradLUTfinished_cb\',\n'
            '        \'gui_connectEditActions\', \'gui_createControlsToolbar\',\n'
            '        \'exportToCheckAskOverlay\'\n'
            '    )\n',
            items=[LUTS + '::test_main_gui_calls_the_channel_lut_and_overlay_mode_hooks'])


def linked_right_image_layers(repo):
    replace(repo, LINKED,
            '"""\n'
            'import ast\n'
            'import inspect\n'
            'import os\n'
            'from types import MethodType, SimpleNamespace\n',
            '"""\n'
            'import ast\n'
            'import inspect\n'
            'import os\n'
            'import pathlib\n'
            'import warnings\n'
            'from types import MethodType, SimpleNamespace\n',
            items=[LINKED + '::import:import pathlib'])
    # (the old text must not be part of the new one: idempotency)
    replace(repo, LINKED,
            'from qtpy.QtWidgets import QAction, QApplication\n'
            '\n'
            'from cellacdc import gui, widgets\n',
            'from qtpy.QtWidgets import QAction, QApplication\n'
            '\n'
            'import cellacdc\n'
            'from cellacdc import gui, widgets\n',
            items=[LINKED + '::import:import cellacdc'])
    replace(repo, LINKED,
            'def _parentImageItemCalls():\n'
            '    tree = ast.parse(inspect.getsource(gui))\n'
            '    for node in ast.walk(tree):\n'
            '        if not isinstance(node, ast.Call):\n'
            '            continue\n'
            '        func = node.func\n'
            '        if isinstance(func, ast.Attribute) and func.attr == \'ParentImageItem\':\n'
            '            yield node\n',
            'def _parentImageItemCalls():\n'
            '    """`(path, call)` of the `ParentImageItem(...)` calls in the modules of\n'
            '    cellacdc (`path`: posix path relative to the package folder)."""\n'
            '    package = os.path.dirname(cellacdc.__file__)\n'
            '    for dirpath, dirnames, filenames in os.walk(package):\n'
            '        dirnames[:] = sorted(\n'
            '            name for name in dirnames if name != \'__pycache__\'\n'
            '        )\n'
            '        for filename in sorted(filenames):\n'
            '            if not filename.endswith(\'.py\') or filename.startswith(\'qrc_\'):\n'
            '                continue\n'
            '            path = os.path.join(dirpath, filename)\n'
            '            with open(path, encoding=\'utf-8\') as file:\n'
            '                source = file.read()\n'
            '            if \'ParentImageItem(\' not in source:\n'
            '                continue\n'
            '            with warnings.catch_warnings():\n'
            '                # Invalid escape sequences of old code\n'
            '                warnings.simplefilter(\'ignore\', SyntaxWarning)\n'
            '                tree = ast.parse(source)\n'
            '            relpath = pathlib.Path(os.path.relpath(path, package)).as_posix()\n'
            '            for node in ast.walk(tree):\n'
            '                if not isinstance(node, ast.Call):\n'
            '                    continue\n'
            '                func = node.func\n'
            '                name = getattr(func, \'attr\', getattr(func, \'id\', None))\n'
            '                if name == \'ParentImageItem\':\n'
            '                    yield relpath, node\n',
            items=[LINKED + '::_parentImageItemCalls'])
    replace(repo, LINKED,
            '    calls = list(_parentImageItemCalls())\n'
            '    assert len(calls) >= 3\n'
            '    for call in calls:\n'
            '        for keyword in call.keywords:\n'
            '            assert keyword.arg in named, (\n'
            '                f\'gui.py:{call.lineno}: unknown keyword `{keyword.arg}` \'\n'
            '                \'passed to ParentImageItem\'\n'
            '            )\n',
            '    calls = list(_parentImageItemCalls())\n'
            '    # The main GUI\'s layers (also the viewer\'s image)\n'
            '    assert len(calls) >= 3\n'
            '    for path, call in calls:\n'
            '        for keyword in call.keywords:\n'
            '            if keyword.arg is None:\n'
            '                # **kwargs passed on\n'
            '                continue\n'
            '            assert keyword.arg in named, (\n'
            '                f\'{path}:{call.lineno}: unknown keyword `{keyword.arg}` \'\n'
            '                \'passed to ParentImageItem\'\n'
            '            )\n',
            items=[LINKED + '::test_parent_image_item_keywords_are_not_swallowed'])


def list_selection_required(repo):
    replace(repo, LISTS,
            '"""\n'
            'import ast\n'
            'import os\n'
            'import warnings\n',
            '"""\n'
            'import ast\n'
            'import inspect\n'
            'import os\n'
            'import pathlib\n'
            'import warnings\n',
            items=[LISTS + '::import:import inspect'])
    replace(repo, LISTS,
            'from cellacdc import widgets\n'
            'from cellacdc.gui_overlay import (\n',
            'import cellacdc\n'
            'from cellacdc import widgets\n'
            'from cellacdc.gui import guiWin\n'
            'from cellacdc.gui_overlay import (\n',
            items=[LISTS + '::import:from cellacdc.gui import guiWin'])
    replace(repo, LISTS,
            '    without a `not <dialog>.selectedItemsText` check, as\n'
            '    `(path, line, dialog, allowEmptySelection keyword value or None)`."""\n'
            '    package = os.path.dirname(widgets.__file__)\n',
            '    without a `not <dialog>.selectedItemsText` check, as\n'
            '    `(path, line, dialog, allowEmptySelection keyword value or None)`\n'
            '    (`path`: posix path relative to the package folder)."""\n'
            '    package = os.path.dirname(cellacdc.__file__)\n',
            items=[LISTS + '::_single_choice_lists_read_without_a_selection_check'])
    replace(repo, LISTS,
            '            relpath = os.path.relpath(path, package)\n'
            '            for scope in scopes:\n',
            '            relpath = pathlib.Path(os.path.relpath(path, package)).as_posix()\n'
            '            for scope in scopes:\n',
            items=[LISTS + '::_single_choice_lists_read_without_a_selection_check'])
    replace(repo, LISTS,
            '    found = _single_choice_lists_read_without_a_selection_check()\n'
            '    # The scan finds the callers (e.g. the main GUI\'s model selection)\n'
            '    assert (\'gui.py\', \'win\') in {(path, dialog) for path, _, dialog, _ in found}\n',
            '    found = _single_choice_lists_read_without_a_selection_check()\n'
            '    # The scan finds the callers (e.g. the main GUI\'s model selection,\n'
            '    # wherever the method is defined)\n'
            '    method = inspect.unwrap(guiWin.autoSegm_cb)\n'
            '    path = pathlib.Path(os.path.relpath(\n'
            '        inspect.getsourcefile(method), os.path.dirname(cellacdc.__file__)\n'
            '    )).as_posix()\n'
            '    lines, start = inspect.getsourcelines(method)\n'
            '    assert [\n'
            '        row for row in found\n'
            '        if row[0] == path and start <= row[1] < start + len(lines)\n'
            '        and row[2] == \'win\'\n'
            '    ], found\n',
            items=[LISTS + '::test_single_choices_read_by_index_require_a_selection'])


def main(repo):
    channel_luts(repo)
    linked_right_image_layers(repo)
    list_selection_required(repo)


run(main)
