"""View adapter for duplicated right-image interactions."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication

from cellacdc import exception_handler


class CanvasRightImageMixin:
    """Qt-facing adapter for duplicated right-image mouse events."""

    """Headless duplicated right-image event rules."""

    @exception_handler
    def mouse_drag(self, event):
        self.gui_mouseDragEventImg1(event)

    @exception_handler
    def mouse_press(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        alt = modifiers == Qt.AltModifier
        right_click = event.button() == Qt.MouseButton.RightButton and not alt
        is_right_click_action_on = any(
            [b.isChecked() for b in self.checkableQButtonsGroup.buttons()]
        )
        self.typingEditID = False
        show_menu = self.should_show_context_menu(
            right_click=right_click,
            is_right_click_action_on=is_right_click_action_on,
        )
        if show_menu:
            self.canvas_context_menu_view.show_right_image_context_menu(event)
            event.ignore()
        else:
            self.gui_mousePressEventImg1(event)

    @exception_handler
    def mouse_release(self, event):
        self.gui_mouseReleaseEventImg1(event)

    def should_show_context_menu(
        self,
        *,
        right_click: bool,
        is_right_click_action_on: bool,
    ) -> bool:
        return right_click and not is_right_click_action_on
