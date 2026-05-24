"""Parent graph for batch segmentation over many positions."""

from __future__ import annotations

from typing import Any

from ..constants import END
from ..graph import StateGraph
from ..runnable import RunnableConfig
from ..state import BatchState, BatchWorkflowContext, PositionState
from .segm import build_position_segm_graph


def _position_graph(ctx: BatchWorkflowContext):
    if ctx.position_graph is None:
        ctx.position_graph = build_position_segm_graph(ctx.position_ctx).compile()
    return ctx.position_graph


def process_position(
    state: BatchState,
    ctx: BatchWorkflowContext,
    config: RunnableConfig,
) -> dict[str, Any]:
    path = state.paths[state.current_index]
    stop_frame_n = state.stop_frame_numbers[state.current_index]
    config.logger_func(f'\nProcessing "{path}"...')
    result = _position_graph(ctx).invoke(
        PositionState(img_path=path, stop_frame_n=stop_frame_n),
        config,
    )
    results = list(state.results)
    results.append(result)
    progress = config.metadata.get("progress")
    if progress is not None:
        progress.update(1)
    return {"results": results, "current_index": state.current_index + 1}


def _route_batch(state: BatchState, _ctx: BatchWorkflowContext) -> str:
    if state.current_index >= len(state.paths):
        return END
    return "process_position"


def build_segm_batch_graph(
    ctx: BatchWorkflowContext,
) -> StateGraph[BatchState, BatchWorkflowContext]:
    graph = StateGraph(BatchState, ctx)
    graph.add_node("process_position", process_position)
    graph.set_entry_point("process_position")
    graph.add_conditional_edges(
        "process_position",
        _route_batch,
        {"process_position": "process_position", END: END},
    )
    return graph
