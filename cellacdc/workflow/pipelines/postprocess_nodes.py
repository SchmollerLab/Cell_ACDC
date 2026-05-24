"""Shared postprocess helpers for workflow nodes."""

from __future__ import annotations

from typing import Any

from cellacdc import core, features


def apply_postprocess(
    lab: Any,
    img: Any,
    pos_data: Any,
    frame_i: int,
    *,
    apply_postprocessing: bool,
    standard_postprocess_kwargs: dict[str, Any],
    custom_postprocess_features: dict[str, Any],
    custom_postprocess_grouped_features: dict[str, Any],
    user_ch_name: str | None = None,
) -> Any:
    if not apply_postprocessing:
        return lab

    lab = core.post_process_segm(lab, **standard_postprocess_kwargs)
    if not custom_postprocess_features:
        return lab

    ch_name = user_ch_name or pos_data.user_ch_name
    return features.custom_post_process_segm(
        pos_data,
        custom_postprocess_grouped_features,
        lab,
        img,
        frame_i,
        pos_data.filename,
        ch_name,
        custom_postprocess_features,
    )
