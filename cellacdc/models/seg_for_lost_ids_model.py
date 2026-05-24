"""Scriptable model rules for segmenting lost IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cellacdc.myutils import ArgSpec


@dataclass(frozen=True)
class SegForLostIdsSettings:
    """Settings selected for the segment-lost-IDs worker."""

    win: Any
    init_kwargs_new: dict[str, Any]
    args_new: dict[str, Any]
    base_model_name: str


class SegForLostIdsModel:
    """Headless settings and launch rules for lost-ID segmentation."""

    settings_key = 'SegForLostIDsModel'
    worker_model_name = 'local_seg'

    def previous_model_name(self, df_settings) -> str | None:
        try:
            return str(df_settings.at[self.settings_key, 'value'])
        except KeyError:
            return None

    def should_persist_model_choice(self, base_model_name: str | None) -> bool:
        return bool(base_model_name)

    def extra_arg_specs(self) -> list[ArgSpec]:
        extra_params = (
            'overlap_threshold',
            'padding',
            'size_perc_diff',
            'distance_filler_growth',
            'max_iterations',
            'allow_only_tracked_cells',
        )
        extra_types = (float, float, float, float, int, bool)
        extra_defaults = (0.5, 0.8, 0.3, 1.0, 2, False)
        extra_desc = (
            (
                'Overlap threshold with other already segemented cells over '
                'which newly segmented cells are discarded'
            ),
            (
                'Padding of the box used for new segmentation around the '
                'segmentation from the previous frame'
            ),
            (
                'Relative size difference acceptable compared to previous '
                'frames'
            ),
            (
                'Cells which are already segmented are filled with random '
                'noise sampled from background to ensure that they do not get '
                'segmented again. This parameter controls the additional '
                'padding around the already segmented cells.'
            ),
            (
                'The algorithm will try and segment the maximum amount of '
                'cells in the image by running the model several times and '
                'filling new found cells with background noise. How many of '
                'these iterations should be run?'
            ),
            (
                'If no new cell IDs should be permitted '
                '(based on real time tracking)'
            ),
        )

        return [
            ArgSpec(
                name=name,
                default=default,
                type=arg_type,
                desc=desc,
                docstring='',
            )
            for name, default, arg_type, desc in zip(
                extra_params, extra_defaults, extra_types, extra_desc
            )
        ]

    def split_model_kwargs(
        self,
        init_kwargs: dict[str, Any],
        extra_kwargs: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_param_names = {arg.name for arg in self.extra_arg_specs()}
        init_kwargs_new = {}
        args_new = {}

        for key, val in init_kwargs.items():
            if key in extra_param_names:
                args_new[key] = val
            else:
                init_kwargs_new[key] = val

        for key, val in extra_kwargs.items():
            if key in extra_param_names:
                args_new[key] = val

        return init_kwargs_new, args_new

    def settings_from_dialog(self, win, base_model_name: str):
        init_kwargs_new, args_new = self.split_model_kwargs(
            win.init_kwargs,
            win.extra_kwargs,
        )
        return SegForLostIdsSettings(
            win=win,
            init_kwargs_new=init_kwargs_new,
            args_new=args_new,
            base_model_name=base_model_name,
        )

    def can_start_from_frame(self, frame_i: int) -> bool:
        return frame_i > 0
