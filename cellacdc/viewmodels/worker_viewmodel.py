"""View-model contracts for GUI worker lifecycle handling."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.worker_model import WorkerModel


@dataclass(frozen=True)
class WorkerViewModel:
    """Application-facing commands for worker progress decisions."""

    model: WorkerModel = field(default_factory=WorkerModel)

    def progress_log_level(self, logger_level: str = 'INFO') -> str:
        return self.model.progress_log_level(logger_level)

    def progressbar_maximum(self, total_iterations: int) -> int:
        return self.model.progressbar_maximum(total_iterations)

    def lazy_loader_progress_description(self, chunk_range) -> str:
        return self.model.lazy_loader_progress_description(chunk_range)

    def should_enqueue_autosave(self, is_saving: bool) -> bool:
        return self.model.should_enqueue_autosave(is_saving)

    def should_disable_realtime_tracking(
        self,
        tracking_on_never_visited_frames: bool,
        realtime_tracking_enabled: bool,
    ) -> bool:
        return self.model.should_disable_realtime_tracking(
            tracking_on_never_visited_frames,
            realtime_tracking_enabled,
        )
