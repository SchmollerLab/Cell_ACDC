"""View-model behavior for data loading workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.data_loading_model import (
    ChannelNameSuggestion,
    DataLoadingModel,
    EmptyDataPlan,
    ImageDataPreparation,
    OpenImageFileContext,
    OpenImageFileTarget,
)

from .formatting import FormattingViewModel
from .workspace import WorkspaceViewModel


@dataclass(frozen=True)
class DataLoadingViewModel:
    """GUI-facing helpers for data loading workflows."""

    model: DataLoadingModel = field(default_factory=DataLoadingModel)
    formatting: FormattingViewModel = field(default_factory=FormattingViewModel)
    workspace: WorkspaceViewModel = field(default_factory=WorkspaceViewModel)

    def open_image_file_context(
            self, file_path: str, timestamp: str | None = None
    ) -> OpenImageFileContext:
        return self.model.open_image_file_context(file_path, timestamp)

    def channel_name_suggestion(
            self, filename_no_ext: str
    ) -> ChannelNameSuggestion:
        return self.model.channel_name_suggestion(filename_no_ext)

    def open_image_file_target(
            self,
            context: OpenImageFileContext,
            channel_name: str | None = None,
    ) -> OpenImageFileTarget:
        return self.model.open_image_file_target(context, channel_name)

    def empty_data_plan(self, exp_path: str) -> EmptyDataPlan:
        return self.model.empty_data_plan(exp_path)

    def copy_action_text(self, do_copy: bool) -> str:
        return self.model.copy_action_text(do_copy)

    def is_imagej_dtype(self, dtype) -> bool:
        return self.model.is_imagej_dtype(dtype)

    def prepare_tiff_image_data(self, image) -> ImageDataPreparation:
        return self.model.prepare_tiff_image_data(image)

    def merge_default_segm_info(self, existing_df, default_df):
        return self.model.merge_default_segm_info(existing_df, default_df)

    def copy_single_zslice_segm_info(
            self,
            existing_df,
            default_dst_df,
            *,
            src_filename: str,
            dst_filename: str,
    ):
        return self.model.copy_single_zslice_segm_info(
            existing_df,
            default_dst_df,
            src_filename=src_filename,
            dst_filename=dst_filename,
        )
