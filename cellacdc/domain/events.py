"""Lightweight pub/sub for session and canvas state (no Qt)."""

from __future__ import annotations

from typing import Any, Callable


class EventEmitter:
    """Minimal event bus keyed by event name."""

    def __init__(self) -> None:
        self._listeners: dict[str, list[Callable[..., None]]] = {}

    def on(self, event: str, callback: Callable[..., None]) -> None:
        self._listeners.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Callable[..., None]) -> None:
        if event not in self._listeners:
            return
        self._listeners[event] = [
            cb for cb in self._listeners[event] if cb is not callback
        ]

    def emit(self, event: str, *args: Any, **kwargs: Any) -> None:
        for callback in list(self._listeners.get(event, [])):
            callback(*args, **kwargs)
