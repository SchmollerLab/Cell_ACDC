"""Tests of ``apps.EditIDDialog`` (Edit ID tool of the main GUI)."""
import pytest
from qtpy.QtWidgets import QApplication

from cellacdc import apps, widgets


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _dialog(isSegm3D):
    return apps.EditIDDialog(
        3, [1, 2, 3], entryID=4, nextUniqueID=5,
        isSegm3D=isSegm3D, SizeZ=6 if isSegm3D else None,
        currentZslice=2, zSliceStartValue=2,
    )


def _click_assign_new_id(dialog):
    dialog.findChild(widgets.AssignNewIDButton).click()


def test_assign_new_id_reports_the_z_slices_choice(app):
    # gui_mousePressEventImg2 reads howToPropagateZslices after the
    # "Assign new, unique ID" button too (it raised AttributeError)
    dialog = _dialog(isSegm3D=True)
    dialog.setHowToPropagateZslices('Custom range', (1, 4))
    _click_assign_new_id(dialog)

    assert not dialog.cancel
    assert dialog.assignNewID
    assert dialog.howToPropagateZslices == ('Custom range', (1, 4))
    dialog.deleteLater()


def test_assign_new_id_without_z_stack_has_no_z_slices_choice(app):
    dialog = _dialog(isSegm3D=False)
    _click_assign_new_id(dialog)

    assert dialog.assignNewID
    assert dialog.howToPropagateZslices is None
    assert dialog.doPropagateFutureFrames is False
    dialog.deleteLater()


def test_cancelled_dialog_has_the_result_attributes(app):
    dialog = _dialog(isSegm3D=True)
    dialog.cancel_cb(None)

    assert dialog.cancel
    assert dialog.howToPropagateZslices is None
    dialog.deleteLater()
