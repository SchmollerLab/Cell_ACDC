"""Headless replacement of the worker signals: plain callbacks
(``ProgressCallbacks``) and the reverse adapter (``SignalsAdapter``) for code
that keeps the duck-typed ``signals=`` contract (standard library only).

Callbacks are always invoked on the thread that called the core function.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


def _noop(*args, **kwargs) -> None:
    """Default callback: does nothing."""


def _not_cancelled() -> bool:
    return False


@dataclass
class ProgressCallbacks:
    """Progress and log callbacks.

    Attributes
    ----------
    log : callable
        ``log(text, level='INFO')``, level in DEBUG/INFO/WARNING/ERROR.
    progress_init : callable
        ``progress_init(total_steps)``; 0 or 1 means "busy" (GUI semantics).
    progress_step : callable
        ``progress_step(increment)``.
    is_cancelled : callable
        ``is_cancelled() -> bool``, polled between units/jobs and by the
        runner.
    """
    log: Callable[..., None] = _noop
    progress_init: Callable[[int], None] = _noop
    progress_step: Callable[[int], None] = _noop
    is_cancelled: Callable[[], bool] = _not_cancelled


class _Emitter:
    """Object with an ``emit`` method that calls ``func``."""
    def __init__(self, func: Callable[..., None]) -> None:
        self._func = func

    def emit(self, *args) -> None:
        self._func(*args)


class SignalsAdapter:
    """Duck-typed ``signals`` object driven by ``ProgressCallbacks``.

    Attributes with ``.emit(...)``: ``progress`` (text, level='INFO') ->
    ``callbacks.log``, ``initProgressBar`` (n) -> ``progress_init``,
    ``progressBar`` (k) -> ``progress_step``, ``critical`` and ``finished``
    (no-op).
    """
    def __init__(self, callbacks: ProgressCallbacks) -> None:
        self.callbacks = callbacks
        self.progress = _Emitter(self._log)
        self.initProgressBar = _Emitter(
            lambda total: self.callbacks.progress_init(total)
        )
        self.progressBar = _Emitter(
            lambda step: self.callbacks.progress_step(step)
        )
        self.critical = _Emitter(_noop)
        self.finished = _Emitter(_noop)

    def _log(self, text: str, level: str = 'INFO') -> None:
        self.callbacks.log(text, level)
