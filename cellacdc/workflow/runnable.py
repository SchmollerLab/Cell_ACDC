from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass(slots=True)
class RunnableConfig:
    """Per-run callbacks and metadata (LangChain RunnableConfig analogue)."""

    logger_func: Callable[[str], None] = print
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    signals: Any | None = None


@runtime_checkable
class Runnable(Protocol):
    """Minimal composable step interface."""

    def invoke(self, input: Any, config: RunnableConfig | None = None) -> Any: ...


@dataclass(slots=True)
class RunnableLambda(Runnable):
    """Wrap a plain callable as a Runnable."""

    func: Callable[..., Any]
    name: str | None = None

    def invoke(self, input: Any, config: RunnableConfig | None = None) -> Any:
        if config is None:
            return self.func(input)
        return self.func(input, config)


@dataclass(slots=True)
class RunnableSequence(Runnable):
    """Linear chain of runnables (LangChain RunnableSequence analogue)."""

    steps: tuple[Runnable, ...]

    def invoke(self, input: Any, config: RunnableConfig | None = None) -> Any:
        value = input
        for step in self.steps:
            value = step.invoke(value, config)
        return value

    def __or__(self, other: Runnable) -> RunnableSequence:
        if isinstance(other, RunnableSequence):
            return RunnableSequence(self.steps + other.steps)
        return RunnableSequence(self.steps + (other,))
