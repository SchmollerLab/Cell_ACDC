"""Qt-free model rules for data loading workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import skimage
import skimage.color


@dataclass(frozen=True)
class ChannelNameSuggestion:
    """Default basename/channel split for a user-selected image filename."""

    basename: str
    channel_name: str


@dataclass(frozen=True)
class OpenImageFileContext:
    """Path context for opening a single image/video file."""

    file_path: str
    filename_no_ext: str
    extension: str
    source_dirpath: str
    source_dirname: str
    exp_path: str
    acdc_folder: str | None
    requires_images_folder: bool


@dataclass(frozen=True)
class OpenImageFileTarget:
    """Destination paths and metadata names for an opened image/video file."""

    context: OpenImageFileContext
    filename_no_ext: str
    channel_name: str | None
    basename: str | None
    new_filename: str
    new_filepath: str
    metadata_csv_filename: str | None
    metadata_csv_filepath: str | None
    tif_filename: str
    tif_path: str
    direct_copy_supported: bool

    @property
    def has_metadata(self) -> bool:
        return self.basename is not None


@dataclass(frozen=True)
class EmptyDataPlan:
    """Path and filename plan for creating an empty dataset."""

    exp_path: str
    pos_path: str
    images_path: str
    basename: str
    tif_filename: str
    tif_filepath: str
    metadata_filename: str
    metadata_filepath: str


@dataclass(frozen=True)
class ImageDataPreparation:
    """Prepared image data and conversion facts for TIFF writing."""

    image: np.ndarray
    converted_rgb_to_gray: bool
    converted_dtype: bool


class DataLoadingModel:
    """Headless data-loading rules and path plans."""

    def open_image_file_context(
            self, file_path: str, timestamp: str | None = None
    ) -> OpenImageFileContext:
        filename_no_ext, ext = os.path.splitext(os.path.basename(file_path))
        filename_no_ext = filename_no_ext.rstrip('_')
        ext = ext.lower()
        dirpath = os.path.dirname(file_path)
        dirname = os.path.basename(dirpath)
        requires_images_folder = dirname != 'Images'
        acdc_folder = None

        if requires_images_folder:
            timestamp = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
            acdc_folder = f'{timestamp}_acdc'
            exp_path = os.path.join(dirpath, acdc_folder, 'Images')
        else:
            exp_path = dirpath

        return OpenImageFileContext(
            file_path=file_path,
            filename_no_ext=filename_no_ext,
            extension=ext,
            source_dirpath=dirpath,
            source_dirname=dirname,
            exp_path=exp_path,
            acdc_folder=acdc_folder,
            requires_images_folder=requires_images_folder,
        )

    def channel_name_suggestion(
            self, filename_no_ext: str
    ) -> ChannelNameSuggestion:
        underscore_splits = filename_no_ext.split('_')
        if len(underscore_splits) > 1:
            return ChannelNameSuggestion(
                basename='_'.join(underscore_splits[:-1]),
                channel_name=underscore_splits[-1],
            )

        return ChannelNameSuggestion(
            basename=filename_no_ext,
            channel_name='channel_1',
        )

    def open_image_file_target(
            self,
            context: OpenImageFileContext,
            channel_name: str | None = None,
    ) -> OpenImageFileTarget:
        filename_no_ext = context.filename_no_ext
        basename = None
        metadata_csv_filename = None
        metadata_csv_filepath = None

        if channel_name is not None:
            underscore_splits = filename_no_ext.split('_')
            if len(underscore_splits) > 1:
                default_ch_name = underscore_splits[-1]
                if channel_name == default_ch_name:
                    filename_no_ext = '_'.join(underscore_splits[:-1])

            basename = f'{filename_no_ext}_'
            metadata_csv_filename = f'{basename}metadata.csv'
            metadata_csv_filepath = os.path.join(
                context.exp_path, metadata_csv_filename
            )
            new_filename = (
                f'{filename_no_ext}_{channel_name}{context.extension}'
            )
        else:
            new_filename = f'{filename_no_ext}{context.extension}'

        new_filepath = os.path.join(context.exp_path, new_filename)
        tif_filename_no_ext = os.path.splitext(new_filename)[0]
        tif_filename = f'{tif_filename_no_ext}.tif'
        tif_path = os.path.join(context.exp_path, tif_filename)

        return OpenImageFileTarget(
            context=context,
            filename_no_ext=filename_no_ext,
            channel_name=channel_name,
            basename=basename,
            new_filename=new_filename,
            new_filepath=new_filepath,
            metadata_csv_filename=metadata_csv_filename,
            metadata_csv_filepath=metadata_csv_filepath,
            tif_filename=tif_filename,
            tif_path=tif_path,
            direct_copy_supported=context.extension in ('.tif', '.npz'),
        )

    def empty_data_plan(self, exp_path: str) -> EmptyDataPlan:
        pos_path = os.path.join(exp_path, 'Position_1')
        images_path = os.path.join(pos_path, 'Images')
        basename = 'test_empty_'
        tif_filename = f'{basename}channel_1.tif'
        metadata_filename = f'{basename}metadata.csv'

        return EmptyDataPlan(
            exp_path=exp_path,
            pos_path=pos_path,
            images_path=images_path,
            basename=basename,
            tif_filename=tif_filename,
            tif_filepath=os.path.join(images_path, tif_filename),
            metadata_filename=metadata_filename,
            metadata_filepath=os.path.join(images_path, metadata_filename),
        )

    def copy_action_text(self, do_copy: bool) -> str:
        return 'Copying' if do_copy else 'Moving'

    def is_imagej_dtype(self, dtype: np.dtype) -> bool:
        return dtype in (np.uint8, np.uint32, np.float32)

    def prepare_tiff_image_data(self, image: np.ndarray) -> ImageDataPreparation:
        converted_rgb_to_gray = False
        converted_dtype = False
        prepared_image = image

        if (
                prepared_image.ndim == 3
                and (prepared_image.shape[-1] == 3
                     or prepared_image.shape[-1] == 4)
        ):
            converted_rgb_to_gray = True
            if prepared_image.shape[-1] == 3:
                prepared_image = skimage.color.rgb2gray(prepared_image)
            else:
                prepared_image = cv2.cvtColor(
                    prepared_image, cv2.COLOR_RGBA2GRAY
                )
            prepared_image = skimage.img_as_ubyte(prepared_image)

        if not self.is_imagej_dtype(prepared_image.dtype):
            converted_dtype = True
            prepared_image = skimage.img_as_ubyte(prepared_image)

        return ImageDataPreparation(
            image=prepared_image,
            converted_rgb_to_gray=converted_rgb_to_gray,
            converted_dtype=converted_dtype,
        )

    def merge_default_segm_info(
            self,
            existing_df: pd.DataFrame,
            default_df: pd.DataFrame,
    ) -> pd.DataFrame:
        merged_df = pd.concat([default_df, existing_df])
        unique_idx = ~merged_df.index.duplicated()
        return merged_df[unique_idx]

    def copy_single_zslice_segm_info(
            self,
            existing_df: pd.DataFrame,
            default_dst_df: pd.DataFrame,
            *,
            src_filename: str,
            dst_filename: str,
    ) -> pd.DataFrame:
        dst_df = default_dst_df.copy()
        src_df = existing_df.loc[src_filename].copy()

        for z_info in src_df.itertuples():
            frame_i = z_info.Index
            if z_info.which_z_proj != 'single z-slice':
                continue

            src_idx = (src_filename, frame_i)
            if existing_df.at[src_idx, 'resegmented_in_gui']:
                col = 'z_slice_used_gui'
            else:
                col = 'z_slice_used_dataPrep'

            z_slice = existing_df.at[src_idx, col]
            dst_idx = (dst_filename, frame_i)
            dst_df.at[dst_idx, 'z_slice_used_dataPrep'] = z_slice
            dst_df.at[dst_idx, 'z_slice_used_gui'] = z_slice

        return self.merge_default_segm_info(existing_df, dst_df)
