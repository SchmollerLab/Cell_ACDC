"""Segmentation position pipeline graph."""

from __future__ import annotations

from ..constants import END
from ..graph import StateGraph
from ..state import PositionState, WorkflowContext
from . import segm_nodes as nodes


def build_position_segm_graph(
    ctx: WorkflowContext,
) -> StateGraph[PositionState, WorkflowContext]:
    """Build the per-position segmentation graph."""
    graph = StateGraph(PositionState, ctx)
    for name, fn in (
        ("load_position", nodes.load_position),
        ("prepare_stack", nodes.prepare_stack),
        ("ensure_model", nodes.ensure_model),
        ("segment", nodes.segment),
        ("filter_freehand_roi", nodes.filter_freehand_roi),
        ("postprocess", nodes.postprocess),
        ("before_track", nodes.passthrough),
        ("track", nodes.track),
        ("skip_track", nodes.skip_track_progress),
        ("before_pad", nodes.passthrough),
        ("pad_roi", nodes.pad_roi),
        ("before_save", nodes.passthrough),
        ("save", nodes.save),
    ):
        graph.add_node(name, fn)

    graph.set_entry_point("load_position")
    graph.add_edge("load_position", "prepare_stack")
    graph.add_edge("prepare_stack", "ensure_model")
    graph.add_conditional_edges(
        "ensure_model",
        nodes._route_after_model,
        {"segment": "segment", END: END},
    )
    graph.add_edge("segment", "filter_freehand_roi")
    graph.add_conditional_edges(
        "filter_freehand_roi",
        nodes._route_postprocess,
        {"postprocess": "postprocess", "before_track": "before_track"},
    )
    graph.add_edge("postprocess", "before_track")
    graph.add_conditional_edges(
        "before_track",
        nodes._route_track,
        {"track": "track", "skip_track": "skip_track"},
    )
    graph.add_edge("track", "before_pad")
    graph.add_edge("skip_track", "before_pad")
    graph.add_conditional_edges(
        "before_pad",
        nodes._route_pad_roi,
        {"pad_roi": "pad_roi", "before_save": "before_save"},
    )
    graph.add_edge("pad_roi", "before_save")
    graph.add_conditional_edges(
        "before_save",
        nodes._route_save,
        {"save": "save", END: END},
    )
    graph.add_edge("save", END)
    return graph
