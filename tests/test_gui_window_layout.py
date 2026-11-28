"""How the main GUI window class ``guiWin`` is composed of mixins.

``guiWin`` (``gui/main_window/window.py``) = ``QMainWindow``, the mixins of
``gui/main_window/`` (only guiWin composes them), then the mixins of
``gui/mixins/`` that the measurements viewer composes too. Checked here
(``tests/window_layout.py``): no mixin shadows a Qt name or defines a
Signal, no name is defined in two window-only mixins, class attributes
that override another class stay in the class body, the window-only
mixins come before the shared ones, and the names defined in two or more
cellacdc classes resolve as pinned in ``OVERRIDES``.

``OVERRIDES``: guiWin's overrides of the shared mixins (``FrameData``
counts the objects of all frames, ``Keyboard`` passes the reserved
shortcuts to the custom annotation dialog). Regenerate it after an
intended change with ``python -m tests.test_gui_window_layout`` (prints
the dictionary).
"""
import pytest

from cellacdc.gui.main_window.window import guiWin
from tests import window_layout

pytestmark = pytest.mark.gui

OWN_MIXINS_PREFIX = 'cellacdc.gui.main_window'
OVERRIDES = {
    'countObjects': 'FrameDataGuiElements',
    'countObjectsSnapshots': 'FrameDataGuiElements',
    'countObjectsTimelapse': 'FrameDataGuiElements',
    'customAnnotationDialogKwargs': 'KeyboardGuiElements',
    'setAllIDs': 'FrameDataGuiElements',
    'updateObjectCounts': 'FrameDataGuiElements',
}


def test_window_is_composed_by_the_placement_rules():
    assert window_layout.check_layout(guiWin, OWN_MIXINS_PREFIX) == []


def test_names_defined_by_several_classes_resolve_as_expected():
    assert window_layout.overrides(guiWin) == OVERRIDES, (
        'Regenerate OVERRIDES with python -m tests.test_gui_window_layout '
        'if the change is intended'
    )


def test_window_only_overrides_come_before_the_shared_mixins():
    definers = window_layout.definers(guiWin)
    for name in OVERRIDES:
        classes = definers[name]
        assert not window_layout.is_shared(classes[0]), name
        assert any(window_layout.is_shared(cls) for cls in classes[1:]), name


if __name__ == '__main__':
    window_layout.print_expectation(guiWin)
