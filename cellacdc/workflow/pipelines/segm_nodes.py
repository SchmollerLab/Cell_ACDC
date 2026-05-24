"""Segmentation pipeline node implementations."""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
from tqdm import tqdm

from cellacdc import core, features, io, load, utils

from ..constants import END
from ..runnable import RunnableConfig
from ..state import PositionState, WorkflowContext


def passthrough(
    state: PositionState,
    ctx: WorkflowContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    return {}


def load_position(
    state: PositionState,
    ctx: WorkflowContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    pos_data = load.loadData(state.img_path, ctx.user_ch_name)
    config.logger_func(f"Loading {pos_data.relPath}...")

    pos_data.getBasenameAndChNames()
    pos_data.buildPaths()
    pos_data.loadImgData()
    pos_data.loadOtherFiles(
        load_segm_data=False,
        load_acdc_df=False,
        load_shifts=True,
        loadSegmInfo=True,
        load_delROIsInfo=False,
        load_dataPrep_ROIcoords=True,
        load_bkgr_data=True,
        load_last_tracked_i=False,
        load_metadata=True,
        load_dataprep_free_roi=True,
        end_filename_segm=ctx.segm_endname,
    )

    end_name = (
        ctx.segm_endname.replace("segm", "", 1).replace("_", "", 1).split(".")[0]
    )
    if end_name:
        pos_data.setFilePaths(end_name)

    if ctx.do_save:
        segm_filename = os.path.basename(pos_data.segm_npz_path)
        config.logger_func(f"\nSegmentation file {segm_filename}...")

    pos_data.SizeT = ctx.size_t
    if ctx.size_z > 1:
        pos_data.SizeZ = pos_data.img_data.shape[-3]
    else:
        pos_data.SizeZ = 1

    pos_data.isSegm3D = ctx.is_segm_3d
    pos_data.saveMetadata()

    is_roi_active = False
    roi_bounds = None
    if pos_data.dataPrep_ROIcoords is not None and ctx.use_roi:
        df_roi = pos_data.dataPrep_ROIcoords.loc[0]
        is_roi_active = df_roi.at["cropped", "value"] == 0
        x0, x1, y0, y1 = df_roi["value"].astype(int)[:4]
        y_shape, x_shape = pos_data.img_data.shape[-2:]
        x0 = x0 if x0 > 0 else 0
        y0 = y0 if y0 > 0 else 0
        x1 = x1 if x1 < x_shape else x_shape
        y1 = y1 if y1 < y_shape else y_shape
        roi_bounds = (x0, x1, y0, y1)

    return {
        "pos_data": pos_data,
        "is_roi_active": is_roi_active,
        "roi_bounds": roi_bounds,
        "stop_i": state.stop_frame_n,
        "t0": 0,
        "aborted": False,
        "error": None,
    }


def prepare_stack(
    state: PositionState,
    ctx: WorkflowContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    pos_data = state.pos_data
    stop_i = state.stop_i
    is_roi_active = state.is_roi_active
    pad_info = None
    second_ch_data = None

    if ctx.second_channel_name is not None:
        config.logger_func(f'Loading second channel "{ctx.second_channel_name}"...')
        second_ch_filepath = load.get_filename_from_channel(
            pos_data.images_path, ctx.second_channel_name
        )
        second_ch_img_data = load.load_image_file(second_ch_filepath)
    else:
        second_ch_img_data = None

    x0 = x1 = y0 = y1 = 0
    if state.roi_bounds is not None:
        x0, x1, y0, y1 = state.roi_bounds

    if pos_data.SizeT > 1:
        t0 = state.t0
        if pos_data.SizeZ > 1 and not ctx.is_segm_3d and not ctx.use_3d_data_for_2d_segm:
            img_data = pos_data.img_data
            if ctx.second_channel_name is not None:
                second_ch_data_slice = second_ch_img_data[t0:stop_i]
            if is_roi_active:
                y_shape, x_shape = img_data.shape[-2:]
                img_data = img_data[:, :, y0:y1, x0:x1]
                if ctx.second_channel_name is not None:
                    second_ch_data_slice = second_ch_data_slice[:, :, y0:y1, x0:x1]
                pad_info = ((0, 0), (y0, y_shape - y1), (x0, x_shape - x1))

            img_data_slice = img_data[t0:stop_i]
            postprocess_img = img_data
            y_shape, x_shape = img_data.shape[-2:]
            new_shape = (stop_i, y_shape, x_shape)
            img_data = np.zeros(new_shape, img_data.dtype)
            if ctx.second_channel_name is not None:
                second_ch_data = np.zeros(new_shape, second_ch_img_data.dtype)

            df = pos_data.segmInfo_df.loc[pos_data.filename]
            for z_info in df[:stop_i].itertuples():
                i = z_info.Index
                z = z_info.z_slice_used_dataPrep
                z_proj_how = z_info.which_z_proj
                img = img_data_slice[i]
                if ctx.second_channel_name is not None:
                    second_ch_img = second_ch_data_slice[i]
                if z_proj_how == "single z-slice":
                    img_data[i] = img[z]
                    if ctx.second_channel_name is not None:
                        second_ch_data[i] = second_ch_img[z]
                elif z_proj_how == "max z-projection":
                    img_data[i] = img.max(axis=0)
                    if ctx.second_channel_name is not None:
                        second_ch_data[i] = second_ch_img.max(axis=0)
                elif z_proj_how == "mean z-projection":
                    img_data[i] = img.mean(axis=0)
                    if ctx.second_channel_name is not None:
                        second_ch_data[i] = second_ch_img.mean(axis=0)
                elif z_proj_how == "median z-proj.":
                    img_data[i] = np.median(img, axis=0)
                    if ctx.second_channel_name is not None:
                        second_ch_data[i] = np.median(second_ch_img, axis=0)
        elif pos_data.SizeZ > 1 and (ctx.is_segm_3d or ctx.use_3d_data_for_2d_segm):
            img_data = pos_data.img_data[t0:stop_i]
            postprocess_img = img_data
            if ctx.second_channel_name is not None:
                second_ch_data = second_ch_img_data[t0:stop_i]
            if is_roi_active:
                y_shape, x_shape = img_data.shape[-2:]
                img_data = img_data[:, :, y0:y1, x0:x1]
                if ctx.second_channel_name is not None:
                    second_ch_data = second_ch_data[:, :, y0:y1, x0:x1]
                pad_info = ((0, 0), (0, 0), (y0, y_shape - y1), (x0, x_shape - x1))
        else:
            img_data = pos_data.img_data[t0:stop_i]
            postprocess_img = img_data
            if ctx.second_channel_name is not None:
                second_ch_data = second_ch_img_data[t0:stop_i]
            if is_roi_active:
                y_shape, x_shape = img_data.shape[-2:]
                img_data = img_data[:, y0:y1, x0:x1]
                if ctx.second_channel_name is not None:
                    second_ch_data = second_ch_data[:, :, y0:y1, x0:x1]
                pad_info = ((0, 0), (y0, y_shape - y1), (x0, x_shape - x1))
    elif pos_data.SizeZ > 1 and not ctx.is_segm_3d and not ctx.use_3d_data_for_2d_segm:
        img_data = pos_data.img_data
        if ctx.second_channel_name is not None:
            second_ch_data = second_ch_img_data
        if is_roi_active:
            y_shape, x_shape = img_data.shape[-2:]
            pad_info = ((y0, y_shape - y1), (x0, x_shape - x1))
            img_data = img_data[:, y0:y1, x0:x1]
            if ctx.second_channel_name is not None:
                second_ch_data = second_ch_data[:, :, y0:y1, x0:x1]

        postprocess_img = img_data
        z_info = pos_data.segmInfo_df.loc[pos_data.filename].iloc[0]
        z = z_info.z_slice_used_dataPrep
        z_proj_how = z_info.which_z_proj
        if z_proj_how == "single z-slice":
            img_data = img_data[z]
            if ctx.second_channel_name is not None:
                second_ch_data = second_ch_data[z]
        elif z_proj_how == "max z-projection":
            img_data = img_data.max(axis=0)
            if ctx.second_channel_name is not None:
                second_ch_data = second_ch_data.max(axis=0)
        elif z_proj_how == "mean z-projection":
            img_data = img_data.mean(axis=0)
            if ctx.second_channel_name is not None:
                second_ch_data = second_ch_data.mean(axis=0)
        elif z_proj_how == "median z-proj.":
            img_data = np.median(img_data, axis=0)
            if ctx.second_channel_name is not None:
                second_ch_data = np.median(second_ch_data, axis=0)
    elif pos_data.SizeZ > 1 and (ctx.is_segm_3d or ctx.use_3d_data_for_2d_segm):
        img_data = pos_data.img_data
        if ctx.second_channel_name is not None:
            second_ch_data = second_ch_img_data
        if is_roi_active:
            y_shape, x_shape = img_data.shape[-2:]
            pad_info = ((0, 0), (y0, y_shape - y1), (x0, x_shape - x1))
            img_data = img_data[:, y0:y1, x0:x1]
            if ctx.second_channel_name is not None:
                second_ch_data = second_ch_data[:, y0:y1, x0:x1]
        postprocess_img = img_data
    else:
        img_data = pos_data.img_data
        if ctx.second_channel_name is not None:
            second_ch_data = second_ch_img_data
        if is_roi_active:
            y_shape, x_shape = img_data.shape[-2:]
            pad_info = ((y0, y_shape - y1), (x0, x_shape - x1))
            img_data = img_data[y0:y1, x0:x1]
            if ctx.second_channel_name is not None:
                second_ch_data = second_ch_data[y0:y1, x0:x1]
        postprocess_img = img_data

    config.logger_func(f"\nImage shape = {img_data.shape}")
    return {
        "img_data": img_data,
        "second_ch_data": second_ch_data,
        "postprocess_img": postprocess_img,
        "pad_info": pad_info,
    }


def ensure_model(
    state: PositionState,
    ctx: WorkflowContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    if ctx.model is not None:
        return {}

    if ctx.signals is not None:
        ctx.signals.progress.emit(
            f"\nInitializing {ctx.model_name} segmentation model..."
        )
    else:
        config.logger_func(f"\nInitializing {ctx.model_name} segmentation model...")

    acdc_segment = utils.import_segment_module(ctx.model_name)
    init_argspecs, segment_argspecs = utils.getModelArgSpec(acdc_segment)
    ctx.init_model_kwargs = utils.parse_model_params(
        init_argspecs, ctx.init_model_kwargs
    )
    ctx.model_kwargs = utils.parse_model_params(segment_argspecs, ctx.model_kwargs)
    if ctx.second_channel_name is not None:
        ctx.init_model_kwargs["is_rgb"] = True

    ctx.model = utils.init_segm_model(
        acdc_segment, state.pos_data, ctx.init_model_kwargs
    )
    if ctx.model is None:
        message = f"Segmentation model {ctx.model_name} was not initialized!"
        config.logger_func(f"\n{message}")
        return {"aborted": True, "error": message}

    ctx.is_segment3dt_available = any(
        name == "segment3DT" for name in dir(ctx.model)
    ) and not ctx.reduce_memory_usage
    return {"model": ctx.model}


def segment(
    state: PositionState,
    ctx: WorkflowContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    pos_data = state.pos_data
    img_data = state.img_data
    second_ch_data = state.second_ch_data

    config.logger_func(f"\nSegmenting with {ctx.model_name}...")
    time.perf_counter()

    if pos_data.SizeT > 1:
        if ctx.inner_pbar_available and ctx.signals is not None:
            ctx.signals.resetInnerPbar.emit(len(img_data))

        if ctx.is_segment3dt_available and img_data.ndim == 3:
            ctx.model_kwargs["signals"] = (ctx.signals, ctx.inner_pbar_available)
            if ctx.second_channel_name is not None:
                img_data = ctx.model.second_ch_img_to_stack(img_data, second_ch_data)
            lab_stack = core.segm_model_segment(
                ctx.model,
                img_data,
                ctx.model_kwargs,
                is_timelapse_model_and_data=True,
                preproc_recipe=ctx.preproc_recipe,
                posData=pos_data,
            )
            if ctx.inner_pbar_available and ctx.signals is not None:
                ctx.signals.progressBar.emit(1)
        else:
            lab_stack = []
            pbar = tqdm(total=len(img_data), ncols=100)
            for t, img in enumerate(img_data):
                if ctx.second_channel_name is not None:
                    img = ctx.model.second_ch_img_to_stack(img, second_ch_data[t])
                lab = core.segm_model_segment(
                    ctx.model,
                    img,
                    ctx.model_kwargs,
                    frame_i=t,
                    preproc_recipe=ctx.preproc_recipe,
                    posData=pos_data,
                )
                lab_stack.append(lab)
                if ctx.signals is not None:
                    if ctx.inner_pbar_available:
                        ctx.signals.innerProgressBar.emit(1)
                    else:
                        ctx.signals.progressBar.emit(1)
                pbar.update()
            pbar.close()
            lab_stack = np.array(lab_stack, dtype=np.uint32)
            if ctx.inner_pbar_available and ctx.signals is not None:
                ctx.signals.progressBar.emit(1)
    else:
        if ctx.second_channel_name is not None:
            img_data = ctx.model.second_ch_img_to_stack(img_data, second_ch_data)
        lab_stack = core.segm_model_segment(
            ctx.model,
            img_data,
            ctx.model_kwargs,
            frame_i=0,
            preproc_recipe=ctx.preproc_recipe,
            posData=pos_data,
        )
        if ctx.signals is not None:
            ctx.signals.progressBar.emit(1)

    pos_data.saveSamEmbeddings(logger_func=config.logger_func)
    return {"lab_stack": lab_stack, "img_data": img_data}


def filter_freehand_roi(
    state: PositionState,
    ctx: WorkflowContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    pos_data = state.pos_data
    lab_stack = state.lab_stack
    if len(pos_data.dataPrepFreeRoiPoints) > 0 and ctx.use_freehand_roi:
        config.logger_func("Removing objects outside the dataprep free-hand ROI...")
        lab_stack = pos_data.clearSegmObjsDataPrepFreeRoi(
            lab_stack, is_timelapse=pos_data.SizeT > 1
        )
    return {"lab_stack": lab_stack}


def postprocess(
    state: PositionState,
    ctx: WorkflowContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    pos_data = state.pos_data
    lab_stack = state.lab_stack
    postprocess_img = state.postprocess_img

    if pos_data.SizeT > 1:
        pbar = tqdm(total=len(lab_stack), ncols=100)
        for t, lab in enumerate(lab_stack):
            lab_cleaned = core.post_process_segm(
                lab, **ctx.standard_postprocess_kwargs
            )
            lab_stack[t] = lab_cleaned
            if ctx.custom_postprocess_features:
                lab_filtered = features.custom_post_process_segm(
                    pos_data,
                    ctx.custom_postprocess_grouped_features,
                    lab_cleaned,
                    postprocess_img,
                    t,
                    pos_data.filename,
                    pos_data.user_ch_name,
                    ctx.custom_postprocess_features,
                )
                lab_stack[t] = lab_filtered
            pbar.update()
        pbar.close()
    else:
        lab_stack = core.post_process_segm(
            lab_stack, **ctx.standard_postprocess_kwargs
        )
        if ctx.custom_postprocess_features:
            lab_stack = features.custom_post_process_segm(
                pos_data,
                ctx.custom_postprocess_grouped_features,
                lab_stack,
                postprocess_img,
                0,
                pos_data.filename,
                pos_data.user_ch_name,
                ctx.custom_postprocess_features,
            )
    return {"lab_stack": lab_stack}


def track(
    state: PositionState,
    ctx: WorkflowContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    pos_data = state.pos_data
    lab_stack = state.lab_stack

    config.logger_func(f"\nTracking with {ctx.tracker_name} tracker...")
    if ctx.do_save:
        config.logger_func(f"Saving NON-tracked masks of {pos_data.relPath}...")
        io.savez_compressed(pos_data.segm_npz_path, lab_stack)

    if ctx.signals is not None:
        ctx.signals.innerPbar_available = ctx.inner_pbar_available
    ctx.track_params["signals"] = ctx.signals

    tracker_input_img = None
    if ctx.image_channel_tracker is not None:
        if "image" in ctx.track_params:
            tracker_input_img = ctx.track_params.pop("image")
        else:
            config.logger_func(
                f'Loading image data of channel "{ctx.image_channel_tracker}"'
            )
            tracker_input_img = pos_data.loadChannelData(ctx.image_channel_tracker)

    tracked_stack = core.tracker_track(
        lab_stack,
        ctx.tracker,
        ctx.track_params,
        intensity_img=tracker_input_img,
        logger_func=config.logger_func,
    )
    pos_data.fromTrackerToAcdcDf(ctx.tracker, tracked_stack, save=True)
    return {"tracked_stack": tracked_stack, "lab_stack": lab_stack}


def skip_track_progress(
    state: PositionState,
    ctx: WorkflowContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    if ctx.signals is None:
        return {"tracked_stack": state.lab_stack}

    try:
        if ctx.inner_pbar_available:
            ctx.signals.innerProgressBar.emit(state.stop_i)
        else:
            ctx.signals.progressBar.emit(state.stop_i)
    except AttributeError:
        if ctx.inner_pbar_available:
            ctx.signals.innerProgressBar.emit(1)
        else:
            ctx.signals.progressBar.emit(1)
    return {"tracked_stack": state.lab_stack}


def pad_roi(
    state: PositionState,
    ctx: WorkflowContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    tracked_stack = state.tracked_stack
    if state.pad_info is not None:
        config.logger_func(f"Padding with zeros {state.pad_info}...")
        tracked_stack = np.pad(tracked_stack, state.pad_info, mode="constant")
    return {"tracked_stack": tracked_stack}


def save(
    state: PositionState,
    ctx: WorkflowContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    pos_data = state.pos_data
    config.logger_func(f"Saving {pos_data.relPath}...")
    io.savez_compressed(pos_data.segm_npz_path, state.tracked_stack)
    config.logger_func(f"\n{pos_data.relPath} done.")
    return {}


def _route_after_model(state: PositionState, ctx: WorkflowContext) -> str:
    if state.aborted or ctx.model is None:
        return END
    return "segment"


def _route_postprocess(_state: PositionState, ctx: WorkflowContext) -> str:
    return "postprocess" if ctx.do_postprocess else "before_track"


def _route_track(state: PositionState, ctx: WorkflowContext) -> str:
    if not ctx.do_tracking:
        return "skip_track"
    size_t = getattr(state.pos_data, "SizeT", ctx.size_t)
    return "track" if size_t > 1 else "skip_track"


def _route_pad_roi(state: PositionState, _ctx: WorkflowContext) -> str:
    return "pad_roi" if state.is_roi_active else "before_save"


def _route_save(_state: PositionState, ctx: WorkflowContext) -> str:
    return "save" if ctx.do_save else END
