"""Scriptable model rules for segmentation workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmptySegmentationPrompt:
    """Prompt decision for enabling automatic segmentation."""

    should_ask: bool
    scope_text: str = ''


class SegmentationModel:
    """Headless decisions for segmentation orchestration."""

    thresholding_backend_name = 'thresholding'
    thresholding_action_name = 'Automatic thresholding'

    def action_model_name(self, model_name: str) -> str:
        if model_name == self.thresholding_backend_name:
            return self.thresholding_action_name
        return model_name

    def backend_model_name(self, model_name: str) -> str:
        if model_name == self.thresholding_action_name:
            return self.thresholding_backend_name
        return model_name

    def should_compute_segmentation(
        self,
        *,
        mode: str,
        has_labels: bool,
        force: bool,
        auto_enabled: bool,
    ) -> bool:
        if mode in {'Viewer', 'Cell cycle analysis'}:
            return False
        if has_labels and not force:
            return False
        return auto_enabled

    def post_process_params(
        self,
        *,
        apply_postprocessing,
        standard_postprocess_kwargs=None,
        custom_postprocess_features=None,
    ) -> dict:
        params = {'applied_postprocessing': apply_postprocessing}
        params.update(standard_postprocess_kwargs or {})
        params.update(custom_postprocess_features or {})
        return params

    def empty_segmentation_prompt(self, position_data) -> EmptySegmentationPrompt:
        for pos_data in position_data:
            if pos_data.SizeT > 1:
                for lab in pos_data.segm_data:
                    if not np.any(lab):
                        return EmptySegmentationPrompt(True, 'frames')
            elif not np.any(pos_data.segm_data):
                return EmptySegmentationPrompt(True, 'positions')
        return EmptySegmentationPrompt(False)
