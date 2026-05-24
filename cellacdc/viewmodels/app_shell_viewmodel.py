"""View-model contracts for application shell services."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.app_shell_model import AppShellModel


@dataclass(frozen=True)
class AppShellViewModel:
    """Application-facing commands for app metadata and help services."""

    model: AppShellModel = field(default_factory=AppShellModel)

    def read_version(self) -> str:
        return self.model.read_version()

    def tooltips_from_docs(self) -> dict:
        return self.model.tooltips_from_docs()

    def browse_docs(self):
        return self.model.browse_docs()

    def show_in_file_manager(self, path: str):
        return self.model.show_in_file_manager(path)

    def rename_qrc_resources_file(self, color_scheme: str):
        return self.model.rename_qrc_resources_file(color_scheme)
