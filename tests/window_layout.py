"""Checks of how a window class is composed of mixins (used by
``test_gui_window_layout.py`` and ``test_viewer_window_layout.py``).

The main GUI (``guiWin``) and the measurements viewer
(``MeasurementsViewerWin``) are composed of mixin classes (placement rule
7 of CODEMAP.md): ``QMainWindow`` first, then the window-only mixins, then
the mixins shared by both windows (``cellacdc/gui/mixins/``). Qt-named
methods, ``__init__``, ``super()`` users, Signals and class attributes that
override another class stay in the class body.

Names defined in two or more cellacdc classes of the MRO (a window or a
window-only mixin overriding a shared mixin) are pinned by an expectation
in each test file, ``name -> class it resolves to``. When you add or move
such an override on purpose, regenerate it with
``python -m tests.test_gui_window_layout`` (or
``tests.test_viewer_window_layout``) and paste the printed dictionary.
"""
import inspect
import pprint

from qtpy import QtCore
from qtpy.QtWidgets import QMainWindow

SHARED_MIXINS_PREFIX = 'cellacdc.gui.mixins'


def is_dunder(name):
    return name.startswith('__') and name.endswith('__')


def cellacdc_mro(window):
    return [
        cls for cls in window.__mro__
        if cls.__module__.split('.')[0] == 'cellacdc'
    ]


def definers(window):
    """{name: [cellacdc classes of the MRO defining it, in MRO order]}."""
    found = {}
    for cls in cellacdc_mro(window):
        for name in vars(cls):
            if not is_dunder(name):
                found.setdefault(name, []).append(cls)
    return found


def overrides(window):
    """{name: name of the class it resolves to} for the names defined in
    two or more cellacdc classes of the MRO."""
    return {
        name: classes[0].__name__
        for name, classes in sorted(definers(window).items())
        if len(classes) > 1
    }


def is_shared(cls):
    module = cls.__module__
    return module == SHARED_MIXINS_PREFIX or module.startswith(
        SHARED_MIXINS_PREFIX + '.')


def window_only_mixins(window, own_prefix):
    return [
        cls for cls in cellacdc_mro(window)[1:]
        if cls.__module__.startswith(own_prefix)
    ]


def check_layout(window, own_prefix):
    """Problems of the composition of `window` (an empty list if none)."""
    problems = []
    mro = cellacdc_mro(window)
    mixins = mro[1:]
    shared = [cls for cls in mixins if is_shared(cls)]
    window_only = window_only_mixins(window, own_prefix)
    others = [cls for cls in mixins if cls not in shared + window_only]
    if others:
        problems.append(
            'mixins neither window-only nor shared: '
            f'{[f"{c.__module__}.{c.__name__}" for c in others]}'
        )
    qt_names = set(dir(QMainWindow))
    for cls in mixins:
        for name, value in vars(cls).items():
            if is_dunder(name):
                continue
            if name in qt_names:
                problems.append(
                    f'{cls.__name__}.{name} shadows a Qt name: keep it in '
                    f'the {window.__name__} class body'
                )
            if isinstance(value, QtCore.Signal):
                problems.append(
                    f'Signal {cls.__name__}.{name}: Signals belong in the '
                    f'{window.__name__} class body'
                )
    found = definers(window)
    for name, classes in sorted(found.items()):
        in_window_only = [cls for cls in classes if cls in window_only]
        if len(in_window_only) > 1:
            problems.append(
                f'{name} is defined in two window-only mixins: '
                f'{[cls.__name__ for cls in in_window_only]}'
            )
        if len(classes) > 1:
            value = vars(classes[0])[name]
            is_attribute = not (
                inspect.isfunction(value)
                or isinstance(value, (property, staticmethod, classmethod))
            )
            if is_attribute and classes[0] is not window:
                problems.append(
                    f'class attribute {name} overrides another class from '
                    f'{classes[0].__name__}: keep it in the '
                    f'{window.__name__} class body'
                )
            if is_shared(classes[0]) and any(
                    not is_shared(cls) for cls in classes):
                problems.append(
                    f'{name} resolves to the shared {classes[0].__name__} '
                    f'before the window-only override'
                )
    position = {cls: i for i, cls in enumerate(window.__mro__)}
    if shared and window_only and max(
            position[cls] for cls in window_only) > min(
            position[cls] for cls in shared):
        problems.append(
            'window-only mixins must come before the shared mixins in the '
            'bases'
        )
    if window.__mro__[1] is not QMainWindow:
        problems.append('QMainWindow must be the first base')
    return problems


def print_expectation(window):
    """Print the ``name -> resolving class`` expectation to paste into the
    test file."""
    print(pprint.pformat(overrides(window), width=79, sort_dicts=True))
