"""Measurements position pipeline graph."""

from __future__ import annotations

from ..constants import END
from ..graph import StateGraph
from ..state import MeasurementsContext, MeasurementsState
from . import measurements_nodes as nodes


def build_measurements_position_graph(
    ctx: MeasurementsContext,
) -> StateGraph[MeasurementsState, MeasurementsContext]:
    graph = StateGraph(MeasurementsState, ctx)
    graph.add_node("load_position", nodes.load_position)
    graph.add_node("validate_segm", nodes.validate_segm)
    graph.add_node("compute_and_save", nodes.compute_and_save)
    graph.set_entry_point("load_position")
    graph.add_conditional_edges(
        "load_position",
        nodes._route_after_load,
        {"validate_segm": "validate_segm", END: END},
    )
    graph.add_conditional_edges(
        "validate_segm",
        nodes._route_after_validate,
        {"compute_and_save": "compute_and_save", END: END},
    )
    graph.add_edge("compute_and_save", END)
    return graph
