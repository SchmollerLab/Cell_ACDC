"""Mixins for gui.py."""

from __future__ import annotations

import importlib

_GRAPH = None


def _load_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = importlib.import_module("cellacdc.mixins._graph")
    return _GRAPH


def __getattr__(name: str):
    graph = _load_graph()
    if name not in graph.MODULE_TO_CLASS.values():
        raise AttributeError(name)
    module = next(k for k, v in graph.MODULE_TO_CLASS.items() if v == name)
    mod = importlib.import_module(f"cellacdc.mixins.{graph.file_module(module)}")
    return getattr(mod, name)
