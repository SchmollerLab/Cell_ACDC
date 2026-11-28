"""Typed spaces in `widgets.alphaNumericLineEdit`."""
import pytest
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from cellacdc.shared.widgets import line_edits as widgets_line_edits

pytestmark = pytest.mark.gui


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('kwargs', [{}, {'onlyWarn': True}])
def test_typed_space_becomes_underscore_when_not_allowed(app, kwargs):
    lineEdit = widgets_line_edits.alphaNumericLineEdit(**kwargs)
    QTest.keyClicks(lineEdit, 'my col')
    assert lineEdit.text() == 'my_col'
    lineEdit.deleteLater()


def test_typed_space_is_kept_when_allowed(app):
    # e.g. channel names of InitFijiMacroDialog ("ch1, ch2")
    lineEdit = widgets_line_edits.alphaNumericLineEdit(additionalChars=' ,')
    QTest.keyClicks(lineEdit, 'ch1, Phase Contrast')
    assert lineEdit.text() == 'ch1, Phase Contrast'
    lineEdit.deleteLater()
