"""Debug-mode timings for model environment preparation and worker startup."""

from collections.abc import Callable, Generator
from contextlib import contextmanager
from time import perf_counter


from typing import Callable, Generator

@contextmanager
def startup_timer(
    label: str, logger_func: Callable = print
) -> Generator[None, None, None]:
    from cellacdc.config import parser_args

    if not parser_args["debug"]:
        yield
        return

    start = perf_counter()
    completed = False
    try:
        yield
        completed = True
    finally:
        status = "completed" if completed else "failed"
        logger_func(
            f"[model startup] {label}: {perf_counter() - start:.3f}s ({status})"
        )
