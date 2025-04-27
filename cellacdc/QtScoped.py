from qtpy.QtCore import PYQT6, Qt

from qtpy.QtWidgets import QAbstractSlider, QStyle

def SliderNoAction():
    if PYQT6:
        return QAbstractSlider.SliderAction.SliderNoAction.value
    else:
        return QAbstractSlider.SliderAction.SliderNoAction

def SliderSingleStepAdd():
    if PYQT6:
        return QAbstractSlider.SliderAction.SliderSingleStepAdd.value
    else:
        return QAbstractSlider.SliderAction.SliderSingleStepAdd

def SliderSingleStepSub():
    if PYQT6:
        return QAbstractSlider.SliderAction.SliderSingleStepSub.value
    else:
        return QAbstractSlider.SliderAction.SliderSingleStepSub

def SliderPageStepAdd():
    if PYQT6:
        return QAbstractSlider.SliderAction.SliderPageStepAdd.value
    else:
        return QAbstractSlider.SliderAction.SliderPageStepAdd

def SliderPageStepSub():
    if PYQT6:
        return QAbstractSlider.SliderAction.SliderPageStepAdd.value
    else:
        return QAbstractSlider.SliderAction.SliderPageStepAdd

def SliderToMinimum():
    if PYQT6:
        return QAbstractSlider.SliderAction.SliderPageStepAdd.value
    else:
        return QAbstractSlider.SliderAction.SliderPageStepAdd

def SliderToMaximum():
    if PYQT6:
        return QAbstractSlider.SliderAction.SliderPageStepAdd.value
    else:
        return QAbstractSlider.SliderAction.SliderPageStepAdd

def SliderMove():
    if PYQT6:
        return QAbstractSlider.SliderAction.SliderMove.value
    else:
        return QAbstractSlider.SliderAction.SliderMove

def QStyleCC_ScrollBar():
    if PYQT6:
        return QStyle.ComplexControl.CC_ScrollBar
    else:
        return QStyle.CC_ScrollBar

def QStyleSC_ScrollBarSubLine():
    if PYQT6:
        return QStyle.SubControl.SC_ScrollBarSubLine
    else:
        return QStyle.SC_ScrollBarSubLine