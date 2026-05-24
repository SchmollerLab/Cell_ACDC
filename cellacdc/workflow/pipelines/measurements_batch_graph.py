"""Measurements batch parent graph."""

from __future__ import annotations

from ..constants import END
from ..graph import StateGraph
from ..runnable import RunnableConfig
from ..state import BatchState, MeasurementsBatchContext, MeasurementsState
from .measurements import build_measurements_position_graph


def _position_graph(ctx: MeasurementsBatchContext):
    if ctx.position_graph is None:
        ctx.position_graph = build_measurements_position_graph(
            ctx.measurements_ctx
        ).compile()
    return ctx.position_graph


def process_position(
    state: BatchState,
    ctx: MeasurementsBatchContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    path = state.paths[state.current_index]
    stop_frame_n = state.stop_frame_numbers[state.current_index]
    config.logger_func(f'\nProcessing "{path}"...')
    result = _position_graph(ctx).invoke(
        MeasurementsState(img_path=path, stop_frame_n=stop_frame_n),
        config,
    )
    results = list(state.results)
    results.append(result)
    progress = config.metadata.get("progress")
    if progress is not None:
        progress.update(1)
    return {"results": results, "current_index": state.current_index + 1}


def _route_batch(state: BatchState, _ctx: MeasurementsBatchContext) -> str:
    if state.current_index >= len(state.paths):
        return END
    return "process_position"


def build_measurements_batch_graph(
    ctx: MeasurementsBatchContext,
) -> StateGraph[BatchState, MeasurementsBatchContext]:
    graph = StateGraph(BatchState, ctx)
    graph.add_node("process_position", process_position)
    graph.set_entry_point("process_position")
    graph.add_conditional_edges(
        "process_position",
        _route_batch,
        {"process_position": "process_position", END: END},
    )
    return graph
