"""Measurements position pipeline nodes."""

from __future__ import annotations

import os
from typing import Any

from ..constants import END
from ..runnable import RunnableConfig
from ..state import MeasurementsContext, MeasurementsState


def load_position(
    state: MeasurementsState,
    ctx: MeasurementsContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    kernel = ctx.kernel
    pos_data = kernel._load_posData(state.img_path, ctx.end_filename_segm)
    return {"pos_data": pos_data, "skipped": False, "aborted": False, "error": None}


def validate_segm(
    state: MeasurementsState,
    ctx: MeasurementsContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    pos_data = state.pos_data
    if pos_data.segmFound:
        return {}

    exp_foldername = os.path.basename(pos_data.exp_path)
    rel_path = f"...{os.sep}{exp_foldername}{os.sep}{pos_data.pos_foldername}"
    ctx.kernel.log(f'Skipping "{rel_path}" because segm. file was not found.')
    return {"skipped": True}


def compute_and_save(
    state: MeasurementsState,
    ctx: MeasurementsContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    ctx.kernel._run_metrics_cli(
        state.pos_data,
        state.stop_frame_n,
        save_metrics=ctx.save_metrics,
        last_cca_frame_i=ctx.last_cca_frame_i,
    )
    return {}


def _route_after_validate(state: MeasurementsState, _ctx: MeasurementsContext) -> str:
    if state.skipped:
        return END
    return "compute_and_save"


def _route_after_load(state: MeasurementsState, _ctx: MeasurementsContext) -> str:
    if state.aborted:
        return END
    return "validate_segm"
