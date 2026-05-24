"""Scriptable model rules for GUI worker lifecycle handling."""

from __future__ import annotations


class WorkerModel:
    """Headless worker progress and lifecycle decisions."""

    def progress_log_level(self, logger_level: str = 'INFO') -> str:
        return logger_level or 'INFO'

    def progressbar_maximum(self, total_iterations: int) -> int:
        if total_iterations == 1:
            return 0
        return total_iterations

    def lazy_loader_progress_description(self, chunk_range) -> str:
        coord0_chunk, coord1_chunk = chunk_range
        return (
            f'Loading new window, range = ({coord0_chunk}, {coord1_chunk})...'
        )

    def should_enqueue_autosave(self, is_saving: bool) -> bool:
        return not is_saving

    def should_disable_realtime_tracking(
        self,
        tracking_on_never_visited_frames: bool,
        realtime_tracking_enabled: bool,
    ) -> bool:
        return (
            tracking_on_never_visited_frames
            and realtime_tracking_enabled
        )
