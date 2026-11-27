"""Helpers of the codemod unit tests: a synthetic `cellacdc` package in a
temporary git repository, with its mapping shards.

The package mimics the hard cases of the real one: a module that is split
into several modules (widgets.py), a module moved whole (gui_nav.py), a
module replaced by a same-name package (gui.py -> gui/), a window class
split into mixins, guarded imports, a pyqtgraph-like config call, relative
imports inside functions, CRLF line endings.
"""
import json
import os
import subprocess
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL_DIR = os.path.dirname(HERE)
if TOOL_DIR not in sys.path:
    sys.path.insert(0, TOOL_DIR)

FILES = {
    'fakepg/__init__.py': '''
        CONFIG = {}


        def setConfigOption(key, value):
            CONFIG[key] = value
    ''',
    'cellacdc/__init__.py': '''
        import os

        cellacdc_path = os.path.dirname(os.path.abspath(__file__))
        GUI_INSTALLED = True
        settings_folderpath = os.path.join(cellacdc_path, '.settings')


        def printl(*args):
            print(*args)
    ''',
    'cellacdc/config_mod.py': '''
        VALUE = 7
    ''',
    'cellacdc/widgets.py': '''
        """Old widgets module."""
        import os
        import json
        import fakepg as pg
        pg.setConfigOption('imageAxisOrder', 'row-major')

        from . import printl
        from . import config_mod
        try:
            import tkinter_missing_xyz as tk
        except Exception:
            pass

        STYLE = 'red'

        # A helper used by the buttons
        # (two comment lines)
        def helper(x):
            return os.path.join(x, STYLE)


        class PushButton:
            def text(self):
                return helper('b')


        class okPushButton(PushButton):
            def text(self):
                return 'ok:' + super().text()
                # return 'old'


        def make_buttons():
            return [PushButton(), okPushButton()]


        def uses_json(value=STYLE):
            from . import config_mod as cm
            return json.dumps({'a': config_mod.VALUE, 'b': cm.VALUE,
                               'c': value})
    ''',
    'cellacdc/gui_nav.py': '''
        """Navigation mixin."""
        from . import widgets
        from .widgets import helper
        from . import printl


        class NavGuiElements:
            def nav(self):
                from . import config_mod
                return len(widgets.make_buttons()), helper('n'), \\
                    config_mod.VALUE

            def show(self):
                return 'nav-show'
    ''',
    'cellacdc/gui.py': '''
        import os
        from . import gui_nav
        from . import printl
        from .config_mod import VALUE

        CONST_A = 3


        def module_helper():
            return CONST_A * 2


        class Win(gui_nav.NavGuiElements):
            sig = 'signal'

            def __init__(self):
                self.x = CONST_A

            def show(self):
                return 'win-show'

            # Layout helpers
            def layout_stuff(self):
                return CONST_A + VALUE

            def other(self):
                return self.layout_stuff() + module_helper()

            def keep(self):
                return os.sep
    ''',
}

SHARDS = {
    'moves.json': {
        'shard': 'moves',
        'default_commit': 'C3',
        'module_moves': [
            {'old': 'cellacdc/gui.py',
             'new': 'cellacdc/gui/main_window/window.py', 'shim': 'none',
             'commit': 'C3'},
            {'old': 'cellacdc/gui_nav.py',
             'new': 'cellacdc/gui/mixins/navigation.py', 'shim': 'alias',
             'commit': 'C3'},
        ],
        'shims': {
            'cellacdc/gui_nav.py': {'kind': 'alias', 'commit': 'C3'},
            'cellacdc/widgets.py': {'kind': 'split', 'commit': 'C5'},
        },
        'packages': {
            'cellacdc/gui': {'doc': 'The main GUI (test).', 'commit': 'C3'},
            'cellacdc/gui/mixins': {'doc': 'Shared mixins.', 'commit': 'C3'},
            'cellacdc/gui/main_window': {'doc': 'Window only.',
                                         'commit': 'C3'},
            'cellacdc/shared': {'doc': 'Shared code.', 'commit': 'C5'},
            'cellacdc/shared/widgets': {'doc': 'Shared widgets.',
                                        'commit': 'C5'},
        },
        'modules': {
            'cellacdc/gui/mixins/navigation.py': {'alias': 'gui_navigation'},
            'cellacdc/gui/main_window/window.py': {'alias': 'gui_window'},
        },
    },
    'widgets.json': {
        'shard': 'widgets',
        'default_commit': 'C5',
        'modules': {
            'cellacdc/shared/widgets/style.py': {
                'alias': 'widgets_style', 'doc': 'Style constants.'},
            'cellacdc/shared/widgets/basic.py': {
                'alias': 'widgets_basic', 'doc': 'Basic widgets.'},
            'cellacdc/shared/widgets/push_buttons.py': {
                'alias': 'widgets_push_buttons', 'doc': 'Push buttons.'},
        },
        'items': {
            'cellacdc/widgets.py::STYLE': 'cellacdc/shared/widgets/style.py',
            "cellacdc/widgets.py::stmt:pg.setConfigOption('imageAxisOrder', "
            "'row-major')": 'cellacdc/shared/widgets/style.py',
            'cellacdc/widgets.py::helper': 'cellacdc/shared/widgets/basic.py',
            'cellacdc/widgets.py::PushButton':
                'cellacdc/shared/widgets/push_buttons.py',
            'cellacdc/widgets.py::okPushButton':
                'cellacdc/shared/widgets/push_buttons.py',
            'cellacdc/widgets.py::make_buttons':
                'cellacdc/shared/widgets/basic.py',
            'cellacdc/widgets.py::uses_json':
                'cellacdc/shared/widgets/basic.py',
            'cellacdc/widgets.py::stmt:try:': 'DELETE',
        },
    },
    'gui.json': {
        'shard': 'gui',
        'default_commit': 'C10',
        'modules': {
            'cellacdc/gui/main_window/constants.py': {
                'alias': 'gui_constants', 'doc': 'Constants of the GUI.'},
            'cellacdc/gui/main_window/layout.py': {
                'alias': 'gui_main_layout', 'doc': 'Layout mixin.'},
        },
        'items': {
            'cellacdc/gui.py::CONST_A':
                'cellacdc/gui/main_window/constants.py',
            'cellacdc/gui.py::module_helper':
                'cellacdc/gui/main_window/constants.py',
        },
        'classes': {
            'cellacdc/gui.py::Win': {
                'host': 'cellacdc/gui/main_window/window.py',
                'mixins': {
                    'cellacdc/gui/main_window/layout.py::LayoutGuiElements':
                        {'doc': 'Layout methods of Win.', 'commit': 'C10'},
                },
                'methods': {
                    'layout_stuff':
                        'cellacdc/gui/main_window/layout.py::'
                        'LayoutGuiElements',
                    'other': 'cellacdc/gui/main_window/layout.py::'
                             'LayoutGuiElements',
                },
                'bases': [
                    'cellacdc/gui/main_window/layout.py::LayoutGuiElements',
                    'cellacdc/gui/mixins/navigation.py::NavGuiElements',
                ],
            },
        },
    },
}


def git(repo, *args):
    proc = subprocess.run(['git', '-C', repo, *args], capture_output=True,
                          text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'git {args}: {proc.stderr}')
    return proc.stdout.strip()


def make_repo(root, crlf=False, files=None, shards=None):
    """Create the synthetic repository in `root`; returns the base sha."""
    os.makedirs(root, exist_ok=True)
    git(root, 'init', '-q')
    git(root, 'config', 'user.email', 'test@example.com')
    git(root, 'config', 'user.name', 'test')
    git(root, 'config', 'core.autocrlf', 'false')
    eol = '\r\n' if crlf else '\n'
    for path, text in (files or FILES).items():
        full = os.path.join(root, path)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        body = textwrap.dedent(text).lstrip('\n')
        with open(full, 'wb') as file:
            file.write(body.replace('\n', eol).encode('utf-8'))
    mapping = os.path.join(root, 'mapping')
    os.makedirs(mapping, exist_ok=True)
    for name, data in (shards or SHARDS).items():
        with open(os.path.join(mapping, name), 'w',
                  encoding='utf-8') as file:
            json.dump(data, file, indent=2)
    with open(os.path.join(root, '.gitignore'), 'w') as file:
        file.write('__pycache__/\n')
    git(root, 'add', '-A')
    git(root, 'commit', '-q', '-m', 'base')
    return git(root, 'rev-parse', 'HEAD')


def run_python(repo, code, timeout=60):
    env = dict(os.environ)
    env['PYTHONPATH'] = repo
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    proc = subprocess.run(
        [sys.executable, '-W', 'always', '-c', textwrap.dedent(code)],
        cwd=repo, capture_output=True, text=True, timeout=timeout, env=env)
    return proc


def read(repo, path):
    with open(os.path.join(repo, path), 'rb') as file:
        return file.read().decode('utf-8')
