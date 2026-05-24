"""View-model commands for workspace path helpers."""

from __future__ import annotations

from cellacdc import load
from cellacdc.myutils import (
    addToRecentPaths,
    determine_folder_type,
    getMostRecentPath,
    get_pos_foldernames,
    listdir,
)


class WorkspaceMixin:
    """Application-facing commands for filesystem workspace discovery."""

    def add_recent_path(self, path, *, logger=None):
        return addToRecentPaths(str(path), logger=logger)

    def determine_folder_type(self, folder_path):
        is_pos_folder, is_images_folder, folder_path = determine_folder_type(
            str(folder_path)
        )
        return is_pos_folder, bool(is_images_folder), folder_path

    def endnames(self, basename, files):
        return load.get_endnames(basename, files)

    def listdir(self, path):
        return listdir(str(path))

    def most_recent_path(self):
        return getMostRecentPath()

    def path_from_endname(self, end_name, images_path, *, ext=None):
        return load.get_path_from_endname(end_name, str(images_path), ext=ext)

    def position_folder_names(
        self,
        exp_path,
        *,
        check_if_is_sub_folder=False,
    ):
        return get_pos_foldernames(
            str(exp_path),
            check_if_is_sub_folder=check_if_is_sub_folder,
        )

    def segmentation_files(self, images_path):
        return load.get_segm_files(str(images_path))
