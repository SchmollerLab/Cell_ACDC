"""Scriptable model services for the application shell."""

from __future__ import annotations

from cellacdc import myutils


def get_tooltips_from_docs():
    from cellacdc.load.selection_omexml import get_tooltips_from_docs as func

    return func()


def rename_qrc_resources_file(color_scheme: str):
    from cellacdc.load.selection_omexml import (
        rename_qrc_resources_file as func,
    )

    return func(color_scheme)


class AppShellModel:
    """Headless application shell service wrappers."""

    def read_version(self) -> str:
        return myutils.read_version()

    def tooltips_from_docs(self) -> dict:
        return get_tooltips_from_docs()

    def browse_docs(self):
        return myutils.browse_docs()

    def show_in_file_manager(self, path: str):
        return myutils.showInExplorer(path)

    def rename_qrc_resources_file(self, color_scheme: str):
        return rename_qrc_resources_file(color_scheme)
