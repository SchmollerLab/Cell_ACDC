"""Batch execution helpers for workflow graphs."""

from __future__ import annotations

from typing import Any

from tqdm import tqdm

from ..runnable import RunnableConfig
from ..state import BatchState, BatchWorkflowContext, PositionState, WorkflowContext
from .batch_graph import build_segm_batch_graph
from .measurements_batch_graph import build_measurements_batch_graph
from ..state import (
    MeasurementsBatchContext,
    MeasurementsContext,
    MeasurementsGuiBatchContext,
    MeasurementsGuiState,
)
from .measurements_gui_batch_graph import build_gui_measurements_batch_graph


def run_segm_batch(
    ctx: WorkflowContext,
    paths: list[str],
    stop_frame_numbers: list[int],
    config: RunnableConfig | None = None,
    progress: tqdm | None = None,
) -> list[PositionState]:
    """Run the position segmentation graph for each path."""
    config = config or RunnableConfig()
    if progress is not None:
        config.metadata["progress"] = progress

    batch_ctx = BatchWorkflowContext(position_ctx=ctx)
    graph = build_segm_batch_graph(batch_ctx).compile()
    batch_state = graph.invoke(
        BatchState(paths=paths, stop_frame_numbers=stop_frame_numbers),
        config,
    )
    return batch_state.results


def run_measurements_batch(
    kernel: Any,
    paths: list[str],
    stop_frame_numbers: list[int],
    end_filename_segm: str,
    config: RunnableConfig | None = None,
    progress: tqdm | None = None,
) -> list[Any]:
    config = config or RunnableConfig(logger_func=kernel.log)
    if progress is not None:
        config.metadata["progress"] = progress

    measurements_ctx = MeasurementsContext(
        end_filename_segm=end_filename_segm,
        kernel=kernel,
    )
    batch_ctx = MeasurementsBatchContext(measurements_ctx=measurements_ctx)
    graph = build_measurements_batch_graph(batch_ctx).compile()
    batch_state = graph.invoke(
        BatchState(paths=paths, stop_frame_numbers=stop_frame_numbers),
        config,
    )
    return batch_state.results


def run_gui_measurements_batch(
    kernel: Any,
    paths: list[str],
    stop_frame_numbers: list[int],
    end_filename_segm: str,
    *,
    compute_metrics_worker: Any | None = None,
    save_data_worker: Any | None = None,
    save_metrics: bool = True,
    config: RunnableConfig | None = None,
    progress: tqdm | None = None,
) -> list[MeasurementsGuiState]:
    config = config or RunnableConfig(logger_func=kernel.log)
    if progress is not None:
        config.metadata["progress"] = progress

    batch_ctx = MeasurementsGuiBatchContext(
        kernel=kernel,
        compute_metrics_worker=compute_metrics_worker,
        save_data_worker=save_data_worker,
        save_metrics=save_metrics,
        end_filename_segm=end_filename_segm,
    )
    graph = build_gui_measurements_batch_graph(batch_ctx).compile()
    batch_state = graph.invoke(
        BatchState(paths=paths, stop_frame_numbers=stop_frame_numbers),
        config,
    )
    return batch_state.results


def batch_state_from_workflow_params(workflow_params: dict[str, Any]) -> BatchState:
    paths = workflow_params["paths_info"]["paths"]
    stop_frames = [int(n) for n in workflow_params["paths_info"]["stop_frame_numbers"]]
    return BatchState(paths=paths, stop_frame_numbers=stop_frames)
