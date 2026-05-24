"""View-model contracts for segmenting lost IDs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from cellacdc.models.seg_for_lost_ids_model import (
    SegForLostIdsModel,
    SegForLostIdsSettings,
)
from cellacdc.myutils import ArgSpec


@dataclass(frozen=True)
class SegForLostIdsViewModel:
    """Application-facing commands for lost-ID segmentation settings."""

    model: SegForLostIdsModel = field(default_factory=SegForLostIdsModel)

    @property
    def settings_key(self) -> str:
        return self.model.settings_key

    @property
    def worker_model_name(self) -> str:
        return self.model.worker_model_name

    def previous_model_name(self, df_settings) -> str | None:
        return self.model.previous_model_name(df_settings)

    def should_persist_model_choice(self, base_model_name: str | None) -> bool:
        return self.model.should_persist_model_choice(base_model_name)

    def extra_arg_specs(self) -> list[ArgSpec]:
        return self.model.extra_arg_specs()

    def split_model_kwargs(
        self,
        init_kwargs: dict[str, Any],
        extra_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.model.split_model_kwargs(init_kwargs, extra_kwargs)

    def settings_from_dialog(
        self,
        win,
        base_model_name: str,
    ) -> SegForLostIdsSettings:
        return self.model.settings_from_dialog(win, base_model_name)

    def can_start_from_frame(self, frame_i: int) -> bool:
        return self.model.can_start_from_frame(frame_i)
