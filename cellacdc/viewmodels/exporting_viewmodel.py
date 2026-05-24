"""View-model contracts for image and video export workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.exporting_model import ExportingModel


@dataclass(frozen=True)
class ExportingViewModel:
    """Application-facing export commands."""

    model: ExportingModel = field(default_factory=ExportingModel)

    def timestamped_export_filename(self, kind: str, *, timestamp=None):
        return self.model.timestamped_export_filename(
            kind,
            timestamp=timestamp,
        )

    def export_frame_plan(
        self,
        *,
        current_index: int,
        num_digits: int,
        filename: str,
        pngs_folderpath: str,
    ):
        return self.model.export_frame_plan(
            current_index=current_index,
            num_digits=num_digits,
            filename=filename,
            pngs_folderpath=pngs_folderpath,
        )

    def build_export_mask_image(
        self,
        image_shape,
        view_range,
        *,
        invert_bw=False,
    ):
        return self.model.build_export_mask_image(
            image_shape,
            view_range,
            invert_bw=invert_bw,
        )

    def zoom_ids(self, labels_2d, view_range):
        return self.model.zoom_ids(labels_2d, view_range)

    def shifted_view_range(self, previous_range, current_range, window_range):
        return self.model.shifted_view_range(
            previous_range,
            current_range,
            window_range,
        )
