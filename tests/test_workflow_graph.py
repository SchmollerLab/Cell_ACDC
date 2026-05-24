"""Tests for workflow graph modeling."""

import importlib.util
import sys
import types
import unittest
from pathlib import Path


def _bootstrap_workflow_package():
    root = Path(__file__).resolve().parents[1]
    workflow_root = root / "cellacdc" / "workflow"

    cellacdc_pkg = sys.modules.get("cellacdc")
    if cellacdc_pkg is None:
        cellacdc_pkg = types.ModuleType("cellacdc")
        cellacdc_pkg.__path__ = [str(root / "cellacdc")]
        sys.modules["cellacdc"] = cellacdc_pkg

    workflow_pkg = types.ModuleType("cellacdc.workflow")
    workflow_pkg.__path__ = [str(workflow_root)]
    sys.modules["cellacdc.workflow"] = workflow_pkg

    for name in ("constants", "state", "runnable", "graph"):
        module_name = f"cellacdc.workflow.{name}"
        if module_name in sys.modules:
            continue
        spec = importlib.util.spec_from_file_location(
            module_name, workflow_root / f"{name}.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        setattr(workflow_pkg, name, module)

    return sys.modules["cellacdc.workflow"]


workflow = _bootstrap_workflow_package()
END = workflow.constants.END
PositionState = workflow.state.PositionState
WorkflowContext = workflow.state.WorkflowContext
RunnableConfig = workflow.runnable.RunnableConfig
StateGraph = workflow.graph.StateGraph


class TestSegmWorkflowGraph(unittest.TestCase):
    def test_graph_structure_without_heavy_imports(self):
        ctx = WorkflowContext(user_ch_name="phase", model_name="cellpose")
        graph = StateGraph(PositionState, ctx)
        graph.add_node("load_position", lambda s, c, cfg: {})
        graph.add_node("segment", lambda s, c, cfg: {})
        graph.set_entry_point("load_position")
        graph.add_edge("load_position", "segment")
        graph.add_edge("segment", END)
        structure = graph.get_graph()
        self.assertEqual(structure["entrypoint"], "load_position")
        self.assertEqual(structure["edges"]["segment"], END)

    def test_compiled_graph_routes_to_end(self):
        ctx = WorkflowContext(user_ch_name="phase", do_save=False)
        logs: list[str] = []

        def load_node(state, workflow_ctx, config):
            logs.append("load")
            return {}

        def save_node(state, workflow_ctx, config):
            logs.append("save")
            return {}

        graph = StateGraph(PositionState, ctx)
        graph.add_node("load_position", load_node)
        graph.add_node("save", save_node)
        graph.set_entry_point("load_position")
        graph.add_conditional_edges(
            "load_position",
            lambda _s, workflow_ctx: END if not workflow_ctx.do_save else "save",
            {"save": "save", END: END},
        )
        graph.add_edge("save", END)

        compiled = graph.compile()
        compiled.invoke(
            PositionState(img_path="/tmp/test.tif"),
            RunnableConfig(logger_func=logs.append),
        )
        self.assertEqual(logs, ["load"])

    def test_batch_graph_loops_over_paths(self):
        invoked: list[str] = []

        class _PositionGraph:
            def invoke(self, state, config):
                invoked.append(state.img_path)
                return state

        BatchWorkflowContext = workflow.state.BatchWorkflowContext
        BatchState = workflow.state.BatchState

        position_ctx = WorkflowContext(user_ch_name="phase")
        batch_ctx = BatchWorkflowContext(position_ctx=position_ctx)
        batch_ctx.position_graph = _PositionGraph()

        def process_position(state, ctx, config):
            path = state.paths[state.current_index]
            stop_frame_n = state.stop_frame_numbers[state.current_index]
            result = ctx.position_graph.invoke(
                PositionState(img_path=path, stop_frame_n=stop_frame_n),
                config,
            )
            return {
                "results": [*state.results, result],
                "current_index": state.current_index + 1,
            }

        def route_batch(state, _ctx):
            return END if state.current_index >= len(state.paths) else "process_position"

        graph = StateGraph(BatchState, batch_ctx)
        graph.add_node("process_position", process_position)
        graph.set_entry_point("process_position")
        graph.add_conditional_edges(
            "process_position",
            route_batch,
            {"process_position": "process_position", END: END},
        )
        graph.compile().invoke(
            BatchState(paths=["/a.tif", "/b.tif"], stop_frame_numbers=[1, 2]),
            RunnableConfig(),
        )
        self.assertEqual(invoked, ["/a.tif", "/b.tif"])

    def test_gui_measurements_batch_loops_and_stops_on_abort(self):
        invoked: list[str] = []

        class _GuiMeasurementsGraph:
            def __init__(self, abort_on: str | None = None):
                self.abort_on = abort_on

            def invoke(self, state, config):
                invoked.append(state.img_path)
                aborted = state.img_path == self.abort_on
                return workflow.state.MeasurementsGuiState(
                    img_path=state.img_path,
                    aborted=aborted,
                )

        MeasurementsGuiBatchContext = workflow.state.MeasurementsGuiBatchContext
        BatchState = workflow.state.BatchState
        MeasurementsGuiContext = workflow.state.MeasurementsGuiContext
        MeasurementsGuiState = workflow.state.MeasurementsGuiState

        class _Kernel:
            setup_done = False

            @staticmethod
            def log(msg):
                pass

        batch_ctx = MeasurementsGuiBatchContext(kernel=_Kernel())

        def build_graph(ctx, pos_data_loaded=False):
            del pos_data_loaded

            class _Builder:
                def compile(self):
                    return _GuiMeasurementsGraph(abort_on="/b.tif")

            return _Builder()

        def process_position(state, ctx, config):
            path = state.paths[state.current_index]
            stop_frame_n = state.stop_frame_numbers[state.current_index]
            gui_ctx = MeasurementsGuiContext(kernel=ctx.kernel)
            graph = build_graph(gui_ctx, pos_data_loaded=False).compile()
            result = graph.invoke(
                MeasurementsGuiState(img_path=path, stop_frame_n=stop_frame_n),
                config,
            )
            results = [*state.results, result]
            aborted = bool(getattr(result, "aborted", False))
            return {
                "results": results,
                "current_index": state.current_index + 1,
                "aborted": aborted or state.aborted,
            }

        def route_batch(state, ctx):
            if state.aborted or ctx.kernel.setup_done:
                return END
            if state.current_index >= len(state.paths):
                return END
            return "process_position"

        graph = StateGraph(BatchState, batch_ctx)
        graph.add_node("process_position", process_position)
        graph.set_entry_point("process_position")
        graph.add_conditional_edges(
            "process_position",
            route_batch,
            {"process_position": "process_position", END: END},
        )
        final = graph.compile().invoke(
            BatchState(paths=["/a.tif", "/b.tif", "/c.tif"], stop_frame_numbers=[1, 1, 1]),
            RunnableConfig(),
        )
        self.assertEqual(invoked, ["/a.tif", "/b.tif"])
        self.assertTrue(final.aborted)

    def test_video_graph_structure(self):
        graph = StateGraph(
            workflow.state.InteractiveVideoSegmState,
            None,
        )
        steps = [
            "extend_segm_data",
            "prepare_video_stack",
            "segment_video_frames",
            "finalize_video_run",
        ]
        for step in steps:
            graph.add_node(step, lambda s, c, cfg: {})
        graph.set_entry_point("extend_segm_data")
        for left, right in zip(steps, steps[1:]):
            graph.add_edge(left, right)
        graph.add_edge(steps[-1], END)
        structure = graph.get_graph()
        self.assertEqual(structure["entrypoint"], "extend_segm_data")
        self.assertEqual(structure["edges"]["finalize_video_run"], END)


if __name__ == "__main__":
    unittest.main()
