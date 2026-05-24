"""GUI measurements batch parent graph."""

from __future__ import annotations

from typing import Any

from ..constants import END
from ..graph import StateGraph
from ..runnable import RunnableConfig
from ..state import BatchState, MeasurementsGuiBatchContext, MeasurementsGuiContext, MeasurementsGuiState
from .measurements_gui import build_gui_measurements_graph


def process_position(
    state: BatchState,
    ctx: MeasurementsGuiBatchContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    path = state.paths[state.current_index]
    stop_frame_n = state.stop_frame_numbers[state.current_index]
    config.logger_func(f'\nProcessing "{path}"...')

    gui_ctx = MeasurementsGuiContext(
        kernel=ctx.kernel,
        compute_metrics_worker=ctx.compute_metrics_worker,
        save_data_worker=ctx.save_data_worker,
        save_metrics=ctx.save_metrics,
        do_init_metrics=state.current_index == 0,
        end_filename_segm=ctx.end_filename_segm,
    )
    graph = build_gui_measurements_graph(gui_ctx, pos_data_loaded=False).compile()
    result = graph.invoke(
        MeasurementsGuiState(img_path=path, stop_frame_n=stop_frame_n),
        config,
    )

    results = list(state.results)
    results.append(result)
    progress = config.metadata.get("progress")
    if progress is not None:
        progress.update(1)

    aborted = bool(getattr(result, "aborted", False))
    return {
        "results": results,
        "current_index": state.current_index + 1,
        "aborted": aborted or state.aborted,
    }


def _route_batch(state: BatchState, ctx: MeasurementsGuiBatchContext) -> str:
    if state.aborted or ctx.kernel.setup_done:
        return END
    if state.current_index >= len(state.paths):
        return END
    return "process_position"


def build_gui_measurements_batch_graph(
    ctx: MeasurementsGuiBatchContext,
) -> StateGraph[BatchState, MeasurementsGuiBatchContext]:
    graph = StateGraph(BatchState, ctx)
    graph.add_node("process_position", process_position)
    graph.set_entry_point("process_position")
    graph.add_conditional_edges(
        "process_position",
        _route_batch,
        {"process_position": "process_position", END: END},
    )
    return graph
