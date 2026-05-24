"""Scriptable model rules for GUI actions and shortcuts."""

from __future__ import annotations


class ActionsModel:
    """Headless decisions for action and shortcut workflows."""

    keyboard_shortcuts_section = 'keyboard.shortcuts'
    delete_object_section = 'delete_object.action'
    delete_key_option = 'Key sequence'
    delete_button_option = 'Mouse button'

    def default_delete_object_texts(self, *, is_mac: bool) -> tuple[str, str]:
        if is_mac:
            return 'Ctrl', 'Left click'
        return '', 'Middle click'

    def sanitize_key_sequence_text(self, text) -> str:
        if text is None:
            return ''
        return str(text).encode('ascii', 'ignore').decode('utf-8')

    def delete_object_button_text(self, *, is_left_click: bool) -> str:
        return 'Left click' if is_left_click else 'Middle click'

    def delete_object_button_is_left_click(self, text: str) -> bool:
        return text == 'Left click'

    def should_restore_default_delete_action(self, *, had_error: bool) -> bool:
        return had_error
