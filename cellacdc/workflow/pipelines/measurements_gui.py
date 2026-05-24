"""GUI measurements position pipeline graph."""

from __future__ import annotations

from ..constants import END
from ..graph import StateGraph
from ..state import MeasurementsGuiContext, MeasurementsGuiState
from . import measurements_gui_nodes as nodes


def build_gui_measurements_graph(
    ctx: MeasurementsGuiContext,
    *,
    pos_data_loaded: bool = False,
) -> StateGraph[MeasurementsGuiState, MeasurementsGuiContext]:
    graph = StateGraph(MeasurementsGuiState, ctx)
    graph.add_node("load_position", nodes.load_position)
    graph.add_node("prepare_gui_run", nodes.prepare_gui_run)
    graph.add_node("compute_metrics_frames", nodes.compute_metrics_frames)
    graph.add_node("save_metrics_results", nodes.save_metrics_results)

    if pos_data_loaded:
        graph.set_entry_point("prepare_gui_run")
    else:
        graph.set_entry_point("load_position")
        graph.add_edge("load_position", "prepare_gui_run")

    graph.add_conditional_edges(
        "prepare_gui_run",
        nodes._route_after_prepare,
        {"compute_metrics_frames": "compute_metrics_frames", END: END},
    )
    graph.add_edge("compute_metrics_frames", "save_metrics_results")
    graph.add_edge("save_metrics_results", END)
    return graph
