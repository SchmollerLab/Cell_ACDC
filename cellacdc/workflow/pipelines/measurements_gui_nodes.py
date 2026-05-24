"""GUI measurements pipeline nodes."""

from __future__ import annotations

import os
import traceback

import numpy as np
import skimage.measure

from cellacdc import load, utils

from ..constants import END
from ..runnable import RunnableConfig
from ..state import MeasurementsGuiContext, MeasurementsGuiState


def load_position(
    state: MeasurementsGuiState,
    ctx: MeasurementsGuiContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    end_name = ctx.end_filename_segm or ctx.kernel.end_filename_segm
    pos_data = ctx.kernel._load_posData(state.img_path, end_name)
    return {"pos_data": pos_data, "skipped": False, "aborted": False}


def prepare_gui_run(
    state: MeasurementsGuiState,
    ctx: MeasurementsGuiContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    kernel = ctx.kernel
    pos_data = state.pos_data
    worker = ctx.compute_metrics_worker
    save_worker = ctx.save_data_worker
    exp_foldername = os.path.basename(pos_data.exp_path)

    kernel._set_metrics_func_from_posData(pos_data)

    if worker is not None and ctx.do_init_metrics:
        worker.emitSigInitMetricsDialog(pos_data)
        if worker.abort:
            worker.signals.finished.emit(worker)
            return {"aborted": True}
        if kernel.setup_done:
            worker.signals.finished.emit(worker)
            return {"aborted": True}
        worker.emitSigAskRunNow()
        if worker.abort or worker.savedToWorkflow:
            worker.signals.finished.emit(worker)
            return {"aborted": True}

    if not pos_data.segmFound:
        rel_path = f"...{os.sep}{exp_foldername}{os.sep}{pos_data.pos_foldername}"
        kernel.log(f'Skipping "{rel_path}" because segm. file was not found.')
        return {"skipped": True}

    kernel.init_signals(worker, save_worker)
    kernel.log(
        "Loading the following files:\n"
        f"Segmentation file name: {os.path.basename(pos_data.segm_npz_path)}\n"
        f"ACDC output file name: {os.path.basename(pos_data.acdc_output_csv_path)}"
    )
    pos_data.init_segmInfo_df()

    if worker is not None:
        worker.emitSigComputeVolume(pos_data, state.stop_frame_n)

    kernel._init_metrics_to_save(pos_data)

    if worker is not None:
        worker.signals.initProgressBar.emit(state.stop_frame_n)

    channels_to_load = [
        ch
        for ch in pos_data.chNames
        if ch not in kernel.chNamesToSkip and ch in kernel.chNamesToProcess
    ]
    kernel.log(f"Loading channels {channels_to_load}...")
    kernel._load_image_data(pos_data, channels_to_load)
    return {}


def compute_metrics_frames(
    state: MeasurementsGuiState,
    ctx: MeasurementsGuiContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    acdc_df_li, keys = ctx.kernel._compute_metrics_gui_frames(
        state.pos_data,
        state.stop_frame_n,
        save_metrics=ctx.save_metrics,
        compute_metrics_worker=ctx.compute_metrics_worker,
        save_data_worker=ctx.save_data_worker,
    )
    return {"acdc_df_li": acdc_df_li, "keys": keys}


def save_metrics_results(
    state: MeasurementsGuiState,
    ctx: MeasurementsGuiContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    if not state.acdc_df_li:
        exp_foldername = os.path.basename(state.pos_data.exp_path)
        print("-" * 30)
        ctx.kernel.log(
            "All selected positions in the experiment folder "
            f"{exp_foldername} have EMPTY segmentation mask. "
            "Metrics will not be saved."
        )
        print("-" * 30)
        return {}

    ctx.kernel._concat_and_save_acdc_df(
        state.acdc_df_li,
        state.keys,
        state.pos_data,
        ctx.save_metrics,
        computeMetricsWorker=ctx.compute_metrics_worker,
        saveDataWorker=ctx.save_data_worker,
        last_cca_frame_i=ctx.last_cca_frame_i,
    )
    return {}


def _route_entry(state: MeasurementsGuiState, _ctx: MeasurementsGuiContext) -> str:
    return "prepare_gui_run" if state.pos_data is not None else "load_position"


def _route_after_prepare(state: MeasurementsGuiState, _ctx: MeasurementsGuiContext) -> str:
    if state.aborted or state.skipped:
        return END
    return "compute_metrics_frames"
