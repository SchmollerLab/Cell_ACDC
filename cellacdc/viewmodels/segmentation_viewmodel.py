"""View-model contracts for segmentation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.segmentation_model import SegmentationModel

from .model_registry import ModelRegistryViewModel


@dataclass(frozen=True)
class SegmentationViewModel:
    """Application-facing segmentation commands and decisions."""

    model: SegmentationModel = field(default_factory=SegmentationModel)
    model_registry: ModelRegistryViewModel = field(
        default_factory=ModelRegistryViewModel
    )

    def action_model_name(self, model_name: str) -> str:
        return self.model.action_model_name(model_name)

    def backend_model_name(self, model_name: str) -> str:
        return self.model.backend_model_name(model_name)

    def should_compute_segmentation(
        self,
        *,
        mode: str,
        has_labels: bool,
        force: bool,
        auto_enabled: bool,
    ) -> bool:
        return self.model.should_compute_segmentation(
            mode=mode,
            has_labels=has_labels,
            force=force,
            auto_enabled=auto_enabled,
        )

    def post_process_params(
        self,
        *,
        apply_postprocessing,
        standard_postprocess_kwargs=None,
        custom_postprocess_features=None,
    ) -> dict:
        return self.model.post_process_params(
            apply_postprocessing=apply_postprocessing,
            standard_postprocess_kwargs=standard_postprocess_kwargs,
            custom_postprocess_features=custom_postprocess_features,
        )

    def empty_segmentation_prompt(self, position_data):
        return self.model.empty_segmentation_prompt(position_data)

    def segmentation_models(self, *, include_local_seg: bool = False):
        return self.model_registry.segmentation_models(
            include_local_seg=include_local_seg
        )

    def store_custom_model_path(self, model_file_path):
        return self.model_registry.store_custom_model_path(model_file_path)

    def import_segmentation_module(self, model_name):
        return self.model_registry.import_segmentation_module(model_name)

    def model_arg_specs(self, acdc_segment):
        return self.model_registry.model_arg_specs(acdc_segment)

    def insert_model_arg_spec(self, *args, **kwargs):
        return self.model_registry.insert_model_arg_spec(*args, **kwargs)

    def log_segmentation_params(self, *args, **kwargs):
        return self.model_registry.log_segmentation_params(*args, **kwargs)

    def check_gpu_available(self, *args, **kwargs):
        return self.model_registry.check_gpu_available(*args, **kwargs)

    def init_segmentation_model(self, *args, **kwargs):
        return self.model_registry.init_segmentation_model(*args, **kwargs)
