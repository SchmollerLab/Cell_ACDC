"""View-model contracts for GUI actions and shortcuts."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.actions_model import ActionsModel

from .app_shell_viewmodel import AppShellViewModel
from .model_registry import ModelRegistryViewModel


@dataclass(frozen=True)
class ActionsViewModel:
    """Application-facing actions and shortcut decisions."""

    model: ActionsModel = field(default_factory=ActionsModel)
    app_shell: AppShellViewModel = field(default_factory=AppShellViewModel)
    model_registry: ModelRegistryViewModel = field(
        default_factory=ModelRegistryViewModel
    )

    @property
    def keyboard_shortcuts_section(self) -> str:
        return self.model.keyboard_shortcuts_section

    @property
    def delete_object_section(self) -> str:
        return self.model.delete_object_section

    @property
    def delete_key_option(self) -> str:
        return self.model.delete_key_option

    @property
    def delete_button_option(self) -> str:
        return self.model.delete_button_option

    def default_delete_object_texts(self, *, is_mac: bool) -> tuple[str, str]:
        return self.model.default_delete_object_texts(is_mac=is_mac)

    def sanitize_key_sequence_text(self, text) -> str:
        return self.model.sanitize_key_sequence_text(text)

    def delete_object_button_text(self, *, is_left_click: bool) -> str:
        return self.model.delete_object_button_text(
            is_left_click=is_left_click
        )

    def delete_object_button_is_left_click(self, text: str) -> bool:
        return self.model.delete_object_button_is_left_click(text)

    def should_restore_default_delete_action(self, *, had_error: bool) -> bool:
        return self.model.should_restore_default_delete_action(
            had_error=had_error
        )
