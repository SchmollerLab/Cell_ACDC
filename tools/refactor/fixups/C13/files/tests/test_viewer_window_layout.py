"""How the measurements viewer window class is composed of mixins.

``MeasurementsViewerWin`` (``measurements_viewer/main_window/window.py``)
= ``QMainWindow``, the ``Viewer<Topic>GuiElements`` mixins and
``MeasurementsViewerUtils`` (``viewer_utils.py``, with the widget-free
``PositionDataUtils`` and ``AnnotationDataUtils``), then the shared mixins
of ``gui/mixins/`` (also composed by the main GUI). Checked here
(``tests/window_layout.py``): no mixin shadows a Qt name or defines a
Signal, no name is defined in two viewer mixins, class attributes that
override another class (hook defaults such as
``showDeleteObjectShortcut``) stay in the class body, the viewer's own
classes come before the shared mixins, and the names defined in two or
more cellacdc classes resolve as pinned in ``OVERRIDES`` (the viewer's
overrides of shared mixin methods and hooks).

Regenerate ``OVERRIDES`` after an intended change with
``python -m tests.test_viewer_window_layout`` (prints the dictionary).
"""
import pytest

from cellacdc.measurements_viewer.annotation_data import AnnotationDataUtils
from cellacdc.measurements_viewer.main_window.window import (
    MeasurementsViewerWin
)
from cellacdc.measurements_viewer.position_data import PositionDataUtils
from cellacdc.measurements_viewer.viewer_utils import MeasurementsViewerUtils
from tests import window_layout

pytestmark = pytest.mark.viewer

OWN_MIXINS_PREFIX = 'cellacdc.measurements_viewer'
OVERRIDES = {
    'addAlphaScrollbar': 'MeasurementsViewerWin',
    'addCustomAnnnotScatterPlot': 'MeasurementsViewerWin',
    'addCustomAnnotation': 'MeasurementsViewerWin',
    'addScaleBar': 'MeasurementsViewerWin',
    'addTimestamp': 'MeasurementsViewerWin',
    'annotateAllObjectTracks': 'MeasurementsViewerWin',
    'annotatedAxesIndices': 'ViewerImagePanelGuiElements',
    'bottomLayoutHeightSettingKey': 'MeasurementsViewerWin',
    'canDeleteSavedCustomAnnot': 'AnnotationDataUtils',
    'customAnnotColumnConflict': 'AnnotationDataUtils',
    'customAnnotColumnRemovalNote': 'ViewerCustomAnnotationsGuiElements',
    'customAnnotIsVisibleMode': 'ViewerCustomAnnotationsGuiElements',
    'customAnnotToolTip': 'ViewerCustomAnnotationsGuiElements',
    'customAnnotationDialogKwargs': 'ViewerCustomAnnotationsGuiElements',
    'doCustomAnnotation': 'ViewerCustomAnnotationsGuiElements',
    'initCustomAnnotColumn': 'ViewerCustomAnnotationsGuiElements',
    'invertBw': 'MeasurementsViewerWin',
    'launchSlideshow': 'MeasurementsViewerWin',
    'loadCustomAnnotations': 'MeasurementsViewerWin',
    'load_fluo_data': 'MeasurementsViewerUtils',
    'onCustomAnnotationChanged': 'ViewerCustomAnnotationsGuiElements',
    'onCustomAnnotationRemoved': 'ViewerCustomAnnotationsGuiElements',
    'onMotherDaughterLine': 'MeasurementsViewerWin',
    'onSearchId': 'ViewerNavigationGuiElements',
    'onSigUpdateCcaTableWindow': 'ViewerImagePanelGuiElements',
    'onToggleColorScheme': 'MeasurementsViewerWin',
    'overlayChannelToggled': 'MeasurementsViewerWin',
    'overlay_cb': 'MeasurementsViewerWin',
    'propsWidgetIDvalueChanged': 'MeasurementsViewerWin',
    'reinitCustomAnnot': 'MeasurementsViewerWin',
    'removeCustomAnnotColumn': 'ViewerCustomAnnotationsGuiElements',
    'renameCustomAnnotColumn': 'ViewerCustomAnnotationsGuiElements',
    'resetDefaultAnnotationOptions': 'MeasurementsViewerWin',
    'saveCustomAnnot': 'AnnotationDataUtils',
    'saveCustomAnnotPositionParams': 'AnnotationDataUtils',
    'setCustomAnnotShortcut': 'ViewerCustomAnnotationsGuiElements',
    'setShortcuts': 'MeasurementsViewerWin',
    'showDeleteObjectShortcut': 'MeasurementsViewerWin',
    'updatePropsWidget': 'MeasurementsViewerWin',
}


def test_window_is_composed_by_the_placement_rules():
    assert window_layout.check_layout(
        MeasurementsViewerWin, OWN_MIXINS_PREFIX
    ) == []


def test_names_defined_by_several_classes_resolve_as_expected():
    assert window_layout.overrides(MeasurementsViewerWin) == OVERRIDES, (
        'Regenerate OVERRIDES with python -m tests.test_viewer_window_layout '
        'if the change is intended'
    )


def test_viewer_overrides_come_before_the_shared_mixins():
    definers = window_layout.definers(MeasurementsViewerWin)
    for name in OVERRIDES:
        assert not window_layout.is_shared(definers[name][0]), name


def test_viewer_utils_keep_every_data_method():
    """``MeasurementsViewerUtils(PositionDataUtils, AnnotationDataUtils)``
    comes before the shared mixins, so that ``MeasurementsViewerUtils``
    holds every data method of the viewer (tests call them on stubs)."""
    assert MeasurementsViewerUtils.__mro__[1:3] == (
        PositionDataUtils, AnnotationDataUtils
    )
    mro = MeasurementsViewerWin.__mro__
    shared = [cls for cls in mro if window_layout.is_shared(cls)]
    assert mro.index(MeasurementsViewerUtils) < mro.index(shared[0])


if __name__ == '__main__':
    window_layout.print_expectation(MeasurementsViewerWin)
