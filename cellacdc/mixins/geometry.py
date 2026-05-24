"""Mouse and interaction geometry helpers."""

from __future__ import annotations

from qtpy.QtCore import Qt

from cellacdc import is_mac


class Geometry:
    """Extracted from guiWin."""

    def isDefaultMiddleClick(self, mouseEvent, modifiers):
        if is_mac:
            middle_click = (
                mouseEvent.button() == Qt.MouseButton.LeftButton
                and modifiers == Qt.ControlModifier
                and not self.brushButton.isChecked()
            )
        else:
            middle_click = mouseEvent.button() == Qt.MouseButton.MiddleButton
        return middle_click

    def isMiddleClick(self, mouseEvent, modifiers):
        if self.delObjAction is None:
            return self.isDefaultMiddleClick(mouseEvent, modifiers)

        delObjKeySequence, delObjQtButton = self.delObjAction
        if delObjKeySequence is None:
            # Setting only middle click on mac is allowed, however the
            # delObjKeySequence is None and the tool button is never checked
            isDelObjectActive = True
        else:
            isDelObjectActive = self.delObjToolAction.isChecked()

        mouseEventButton = self.changeRightClickToLeftOnMac(mouseEvent)

        middle_click = mouseEventButton == delObjQtButton and isDelObjectActive

        return middle_click

    def isPanImageClick(self, mouseEvent, modifiers):
        left_click = mouseEvent.button() == Qt.MouseButton.LeftButton
        return modifiers == Qt.AltModifier and left_click

    def middleClickText(self):
        if self.delObjAction is None and is_mac:
            return "Command + Left Click"

        if self.delObjAction is None:
            return "Middle Click"

        delObjKeySequence, delObjQtButton = self.delObjAction

        if delObjQtButton == Qt.MouseButton.LeftButton:
            buttonName = "Left click"
        elif delObjQtButton == Qt.MouseButton.RightButton:
            buttonName = "Right click"
        else:
            buttonName = "Middle click"

        if delObjKeySequence is None:
            return buttonName

        return f"{delObjKeySequence.toString()} + {buttonName}"
