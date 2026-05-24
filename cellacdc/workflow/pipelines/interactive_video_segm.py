"""Interactive timelapse segmentation graph."""

from __future__ import annotations

from ..constants import END
from ..graph import StateGraph
from ..state import InteractiveVideoSegmContext, InteractiveVideoSegmState
from . import interactive_video_segm_nodes as nodes


def build_interactive_video_segm_graph(
    ctx: InteractiveVideoSegmContext,
) -> StateGraph[InteractiveVideoSegmState, InteractiveVideoSegmContext]:
    graph = StateGraph(InteractiveVideoSegmState, ctx)
    graph.add_node("extend_segm_data", nodes.extend_segm_data)
    graph.add_node("prepare_video_stack", nodes.prepare_video_stack)
    graph.add_node("segment_video_frames", nodes.segment_video_frames)
    graph.add_node("finalize_video_run", nodes.finalize_video_run)
    graph.set_entry_point("extend_segm_data")
    graph.add_edge("extend_segm_data", "prepare_video_stack")
    graph.add_edge("prepare_video_stack", "segment_video_frames")
    graph.add_edge("segment_video_frames", "finalize_video_run")
    graph.add_edge("finalize_video_run", END)
    return graph
