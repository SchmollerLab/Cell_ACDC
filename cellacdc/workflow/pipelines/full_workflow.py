"""Top-level INI workflow orchestration graph."""

from __future__ import annotations

from typing import Any

from ..constants import END
from ..graph import StateGraph
from ..runnable import RunnableConfig
from ..state import FullWorkflowState


def run_segm_phase(
    state: FullWorkflowState,
    ctx: Any,
    config: RunnableConfig,
) -> dict[str, Any]:
    if not state.run_segm:
        return {"segm_done": True}

    from cellacdc._run import run_segm_workflow

    run_segm_workflow(state.segm_params, ctx.logger, ctx.log_path)
    return {"segm_done": True}


def run_measurements_phase(
    state: FullWorkflowState,
    ctx: Any,
    config: RunnableConfig,
) -> dict[str, Any]:
    if not state.run_measurements or state.measurements_params is None:
        return {"measurements_done": True}

    from cellacdc._run import run_measurements_workflow

    run_measurements_workflow(state.measurements_params, ctx.logger, ctx.log_path)
    return {"measurements_done": True}


def _route_after_segm(state: FullWorkflowState, _ctx: Any) -> str:
    if state.run_measurements:
        return "run_measurements_phase"
    return END


def build_full_workflow_graph(
    ctx: Any,
) -> StateGraph[FullWorkflowState, Any]:
    graph = StateGraph(FullWorkflowState, ctx)
    graph.add_node("run_segm_phase", run_segm_phase)
    graph.add_node("run_measurements_phase", run_measurements_phase)
    graph.set_entry_point("run_segm_phase")
    graph.add_conditional_edges(
        "run_segm_phase",
        _route_after_segm,
        {"run_measurements_phase": "run_measurements_phase", END: END},
    )
    graph.add_edge("run_measurements_phase", END)
    return graph
