"""Interactive single-frame segmentation graph."""

from __future__ import annotations

from ..constants import END
from ..graph import StateGraph
from ..state import InteractiveSegmContext, InteractiveSegmState
from . import interactive_segm_nodes as nodes


def build_interactive_segm_graph(
    ctx: InteractiveSegmContext,
) -> StateGraph[InteractiveSegmState, InteractiveSegmContext]:
    graph = StateGraph(InteractiveSegmState, ctx)
    graph.add_node("prepare_frame", nodes.prepare_frame)
    graph.add_node("segment_frame", nodes.segment_frame)
    graph.add_node("postprocess_frame", nodes.postprocess_frame)
    graph.add_node("merge_result", nodes.merge_result)
    graph.set_entry_point("prepare_frame")
    graph.add_edge("prepare_frame", "segment_frame")
    graph.add_conditional_edges(
        "segment_frame",
        nodes._route_postprocess,
        {"postprocess_frame": "postprocess_frame", "merge_result": "merge_result"},
    )
    graph.add_edge("postprocess_frame", "merge_result")
    graph.add_edge("merge_result", END)
    return graph
