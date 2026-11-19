"""Typed spaces in `widgets.alphaNumericLineEdit`."""
import pytest
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication

from cellacdc import widgets


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


@pytest.mark.parametrize('kwargs', [{}, {'onlyWarn': True}])
def test_typed_space_becomes_underscore_when_not_allowed(app, kwargs):
    lineEdit = widgets.alphaNumericLineEdit(**kwargs)
    QTest.keyClicks(lineEdit, 'my col')
    assert lineEdit.text() == 'my_col'
    lineEdit.deleteLater()


def test_typed_space_is_kept_when_allowed(app):
    # e.g. channel names of InitFijiMacroDialog ("ch1, ch2")
    lineEdit = widgets.alphaNumericLineEdit(additionalChars=' ,')
    QTest.keyClicks(lineEdit, 'ch1, Phase Contrast')
    assert lineEdit.text() == 'ch1, Phase Contrast'
    lineEdit.deleteLater()
