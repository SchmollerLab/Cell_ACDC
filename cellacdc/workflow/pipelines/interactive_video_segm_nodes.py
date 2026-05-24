"""Interactive timelapse segmentation nodes for the main viewer."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from cellacdc import core

from ..runnable import RunnableConfig
from ..state import InteractiveVideoSegmContext, InteractiveVideoSegmState
from .postprocess_nodes import apply_postprocess


def extend_segm_data(
    state: InteractiveVideoSegmState,
    ctx: InteractiveVideoSegmContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    pos_data = state.pos_data
    segm_data = pos_data.segm_data
    stop_frame_num = ctx.stop_frame_num

    if stop_frame_num <= len(segm_data):
        return {"segm_data": segm_data}

    extended_shape = (stop_frame_num, *segm_data.shape[1:])
    extended_segm_data = np.zeros(extended_shape, dtype=segm_data.dtype)
    extended_segm_data[: len(segm_data)] = segm_data

    if len(extended_shape) == 4 or pos_data.SizeZ == 1:
        pos_data.segm_data = extended_segm_data
        return {"segm_data": extended_segm_data}

    num_added_frames = len(extended_segm_data) - len(segm_data)
    half_z = int(pos_data.SizeZ / 2)
    segm_info_extended = pd.DataFrame(
        {
            "filename": [pos_data.filename] * num_added_frames,
            "frame_i": list(range(len(segm_data), len(extended_segm_data))),
            "z_slice_used_gui": [half_z] * num_added_frames,
            "which_z_proj_gui": ["single z-slice"] * num_added_frames,
        }
    ).set_index(["filename", "frame_i"])
    pos_data.segmInfo_df = pd.concat([pos_data.segmInfo_df, segm_info_extended])
    pos_data.segmInfo_df.to_csv(pos_data.segmInfo_df_csv_path)
    pos_data.segm_data = extended_segm_data
    return {"segm_data": extended_segm_data}


def prepare_video_stack(
    state: InteractiveVideoSegmState,
    ctx: InteractiveVideoSegmContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    pos_data = state.pos_data
    img_data = pos_data.img_data[ctx.start_frame_num - 1 : ctx.stop_frame_num]
    is_4d = img_data.ndim == 4
    is_2d_segm = pos_data.segm_data.ndim == 3
    z_slices = None
    if is_4d and is_2d_segm:
        z_slices = pos_data.segmInfo_df.loc[pos_data.filename, "z_slice_used_gui"]
    return {"img_data": img_data, "z_slices": z_slices}


def segment_video_frames(
    state: InteractiveVideoSegmState,
    ctx: InteractiveVideoSegmContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    pos_data = state.pos_data
    progress = ctx.progress_callback

    for i, img in enumerate(state.img_data):
        frame_i = i + ctx.start_frame_num - 1
        if ctx.second_channel_data is not None:
            img = ctx.model.second_ch_img_to_stack(img, ctx.second_channel_data)
        if state.z_slices is not None:
            img = img[state.z_slices.loc[frame_i]]

        lab = core.segm_model_segment(
            ctx.model,
            img,
            ctx.model_kwargs,
            frame_i=frame_i,
            preproc_recipe=ctx.preproc_recipe,
            posData=pos_data,
        )
        pos_data.saveSamEmbeddings(logger_func=ctx.logger_func)

        if ctx.apply_postprocessing:
            lab = apply_postprocess(
                lab,
                img,
                pos_data,
                frame_i,
                apply_postprocessing=True,
                standard_postprocess_kwargs=ctx.standard_postprocess_kwargs,
                custom_postprocess_features=ctx.custom_postprocess_features,
                custom_postprocess_grouped_features=ctx.custom_postprocess_grouped_features,
            )

        pos_data.segm_data[frame_i] = lab
        if progress is not None:
            progress.emit(1)

    return {}


def finalize_video_run(
    state: InteractiveVideoSegmState,
    ctx: InteractiveVideoSegmContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    return {}
