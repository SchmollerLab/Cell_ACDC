"""View adapter for duplicated right-image interactions."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication

from cellacdc import exception_handler
from cellacdc.viewmodels.canvas_right_image_viewmodel import (
    CanvasRightImageViewModel,
)


class CanvasRightImageView:
    """Qt-facing adapter for duplicated right-image mouse events."""

    def __init__(self, host, view_model: CanvasRightImageViewModel):
        self.host = host
        self.view_model = view_model

    @exception_handler
    def mouse_press(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        alt = modifiers == Qt.AltModifier
        right_click = event.button() == Qt.MouseButton.RightButton and not alt
        is_right_click_action_on = any([
            b.isChecked() for b in self.host.checkableQButtonsGroup.buttons()
        ])
        self.host.typingEditID = False
        show_menu = self.view_model.should_show_context_menu(
            right_click=right_click,
            is_right_click_action_on=is_right_click_action_on,
        )
        if show_menu:
            self.host.canvas_context_menu_view.show_right_image_context_menu(
                event
            )
            event.ignore()
        else:
            self.host.gui_mousePressEventImg1(event)

    @exception_handler
    def mouse_drag(self, event):
        self.host.gui_mouseDragEventImg1(event)

    @exception_handler
    def mouse_release(self, event):
        self.host.gui_mouseReleaseEventImg1(event)
