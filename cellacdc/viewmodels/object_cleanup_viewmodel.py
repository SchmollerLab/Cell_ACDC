"""View-model contracts for object cleanup workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.object_cleanup_model import ObjectCleanupModel

from .workspace_viewmodel import WorkspaceViewModel


@dataclass(frozen=True)
class ObjectCleanupViewModel:
    """Application-facing object-cleanup commands."""

    model: ObjectCleanupModel = field(default_factory=ObjectCleanupModel)
    workspace: WorkspaceViewModel = field(default_factory=WorkspaceViewModel)

    def segmentation_roi_endnames(self, *, basename, images_path):
        segm_files = self.workspace.segmentation_files(images_path)
        return self.workspace.endnames(basename, segm_files)

    def cleared_segmentation_frames(self, cleared_segm_data, *, size_t: int):
        return self.model.cleared_segmentation_frames(
            cleared_segm_data,
            size_t=size_t,
        )

    def frame_labels(self, cleared_segm_data):
        return self.model.frame_labels(cleared_segm_data)
