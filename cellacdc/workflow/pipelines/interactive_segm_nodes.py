"""Interactive single-frame segmentation nodes for the main viewer."""

from __future__ import annotations

import time
from typing import Any

from cellacdc import core

from ..runnable import RunnableConfig
from ..state import InteractiveSegmContext, InteractiveSegmState
from .postprocess_nodes import apply_postprocess


def prepare_frame(
    state: InteractiveSegmState,
    ctx: InteractiveSegmContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    main_win = state.main_win
    pos_data = main_win.data[main_win.pos_i]

    if ctx.segment_3d:
        img = main_win.getDisplayedZstack()
        if ctx.z_range is not None:
            start_z, stop_z = ctx.z_range
            img = img[start_z : stop_z + 1]
    else:
        img = main_win.getDisplayedImg1()

    lab = __import__("numpy").zeros_like(pos_data.segm_data[0])
    start_z_slice = 0
    if ctx.z_range is not None:
        start_z_slice, _ = ctx.z_range
    elif not ctx.segment_3d and pos_data.isSegm3D:
        idx = (pos_data.filename, pos_data.frame_i)
        start_z_slice = pos_data.segmInfo_df.at[idx, "z_slice_used_gui"]

    return {
        "pos_data": pos_data,
        "img": img,
        "lab": lab,
        "start_z_slice": start_z_slice,
    }


def segment_frame(
    state: InteractiveSegmState,
    ctx: InteractiveSegmContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    img = state.img
    if ctx.second_channel_data is not None:
        img = ctx.model.second_ch_img_to_stack(img, ctx.second_channel_data)

    lab = core.segm_model_segment(
        ctx.model,
        img,
        ctx.model_kwargs,
        frame_i=state.pos_data.frame_i,
        posData=state.pos_data,
        start_z_slice=state.start_z_slice,
    )
    state.pos_data.saveSamEmbeddings(logger_func=config.logger_func)
    return {"img": img, "segmented_lab": lab}


def postprocess_frame(
    state: InteractiveSegmState,
    ctx: InteractiveSegmContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    if not ctx.apply_postprocessing:
        return {}

    lab = apply_postprocess(
        state.segmented_lab,
        state.img,
        state.pos_data,
        state.pos_data.frame_i,
        apply_postprocessing=True,
        standard_postprocess_kwargs=ctx.standard_postprocess_kwargs,
        custom_postprocess_features=ctx.custom_postprocess_features,
        custom_postprocess_grouped_features=ctx.custom_postprocess_grouped_features,
    )
    return {"segmented_lab": lab}


def merge_result(
    state: InteractiveSegmState,
    ctx: InteractiveSegmContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    pos_data = state.pos_data
    lab = state.lab
    segmented = state.segmented_lab

    if ctx.z_range is not None:
        start_z, stop_z = ctx.z_range
        lab[start_z : stop_z + 1] = segmented
    elif not ctx.segment_3d and pos_data.isSegm3D:
        idx = (pos_data.filename, pos_data.frame_i)
        z = pos_data.segmInfo_df.at[idx, "z_slice_used_gui"]
        lab[z] = segmented
    else:
        lab = segmented

    return {"lab": lab}


def _route_postprocess(_state: InteractiveSegmState, ctx: InteractiveSegmContext) -> str:
    return "postprocess_frame" if ctx.apply_postprocessing else "merge_result"
