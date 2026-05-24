"""View adapter for duplicated right-image interactions."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication

from cellacdc import exception_handler

from .canvas_drawing import CanvasDrawing
from .canvas_events import CanvasEvents


class CanvasRightImage(CanvasDrawing, CanvasEvents):
    """Extracted from guiWin."""

    def getMouseDataCoordsRightImage(self):
        text = self.wcLabel.text()
        if not text:
            return

        ax_idx = int(re.findall(r"\(ax(\d)\)", text)[0])
        if ax_idx == 0:
            return

        coords = re.findall(r"x=(\d+), y=(\d+) \|", text)[0]

        return tuple([int(val) for val in coords])

    def gui_mousePressRightImage(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        isMod = alt
        right_click = event.button() == Qt.MouseButton.RightButton and not isMod
        is_right_click_action_ON = any(
            [b.isChecked() for b in self.checkableQButtonsGroup.buttons()]
        )
        self.typingEditID = False
        showLabelsGradMenu = right_click and not is_right_click_action_ON
        if showLabelsGradMenu:
            self.gui_rightImageShowContextMenu(event)
            event.ignore()
        else:
            self.gui_mousePressEventImg1(event)

    def gui_mouseDragRightImage(self, event):
        self.gui_mouseDragEventImg1(event)

    def gui_mouseReleaseRightImage(self, event):
        self.gui_mouseReleaseEventImg1(event)
