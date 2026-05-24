from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeVar

from .constants import END, START
from .runnable import RunnableConfig
from .state import merge_state

StateT = TypeVar("StateT")
ContextT = TypeVar("ContextT")

NodeFn = Callable[[StateT, ContextT, RunnableConfig], dict[str, Any]]
RouteFn = Callable[[StateT, ContextT], str]


@dataclass(slots=True)
class CompiledStateGraph(Generic[StateT, ContextT]):
    """Executable graph returned by StateGraph.compile()."""

    nodes: dict[str, NodeFn[StateT, ContextT]]
    edges: dict[str, str]
    conditional_edges: dict[str, tuple[RouteFn[StateT, ContextT], dict[str, str]]]
    entrypoint: str
    state_type: type[StateT]
    context: ContextT

    def invoke(
        self,
        state: StateT,
        config: RunnableConfig | None = None,
    ) -> StateT:
        config = config or RunnableConfig()
        node_name = self.entrypoint
        while node_name != END:
            node = self.nodes[node_name]
            update = node(state, self.context, config)
            state = merge_state(state, update)
            if node_name in self.conditional_edges:
                route_fn, mapping = self.conditional_edges[node_name]
                node_name = mapping[route_fn(state, self.context)]
                continue
            node_name = self.edges[node_name]
        return state


@dataclass
class StateGraph(Generic[StateT, ContextT]):
    """Declarative workflow graph (LangGraph StateGraph analogue)."""

    state_type: type[StateT]
    context: ContextT
    _nodes: dict[str, NodeFn[StateT, ContextT]] = field(default_factory=dict)
    _edges: dict[str, str] = field(default_factory=dict)
    _conditional_edges: dict[str, tuple[RouteFn[StateT, ContextT], dict[str, str]]] = (
        field(default_factory=dict)
    )
    _entrypoint: str | None = None

    def add_node(self, name: str, fn: NodeFn[StateT, ContextT]) -> StateGraph:
        self._nodes[name] = fn
        return self

    def set_entry_point(self, name: str) -> StateGraph:
        self._entrypoint = name
        return self

    def add_edge(self, start: str, end: str) -> StateGraph:
        self._edges[start] = end
        return self

    def add_conditional_edges(
        self,
        start: str,
        route: RouteFn[StateT, ContextT],
        mapping: dict[str, str],
    ) -> StateGraph:
        self._conditional_edges[start] = (route, mapping)
        return self

    def compile(self) -> CompiledStateGraph[StateT, ContextT]:
        if self._entrypoint is None:
            raise ValueError("Graph has no entry point. Call set_entry_point().")
        if self._entrypoint not in self._nodes:
            raise ValueError(f"Unknown entry point node: {self._entrypoint}")
        return CompiledStateGraph(
            nodes=dict(self._nodes),
            edges=dict(self._edges),
            conditional_edges=dict(self._conditional_edges),
            entrypoint=self._entrypoint,
            state_type=self.state_type,
            context=self.context,
        )

    def get_graph(self) -> dict[str, Any]:
        """Return a serializable graph description for tests and debugging."""
        return {
            "nodes": sorted(self._nodes),
            "edges": dict(self._edges),
            "conditional_edges": {
                name: sorted(mapping)
                for name, (_, mapping) in self._conditional_edges.items()
            },
            "entrypoint": self._entrypoint,
        }
