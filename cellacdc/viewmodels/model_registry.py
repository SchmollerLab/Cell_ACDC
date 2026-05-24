"""View-model commands for model registry discovery."""

from __future__ import annotations

from cellacdc.myutils import (
    aliases_real_time_trackers,
    check_gpu_available,
    check_install_package,
    getModelArgSpec,
    get_list_of_models,
    get_list_of_real_time_trackers,
    import_segment_module,
    init_prompt_segm_model,
    init_segm_model,
    init_tracker,
    insertModelArgSpec,
    log_segm_params,
    setDefaultValueArgSpecsFromKwargs,
    store_custom_model_path,
    store_custom_promptable_model_path,
    validate_tracker_input,
)


class ModelRegistryViewModel:
    """Application-facing commands for available model registries."""

    def segmentation_models(self, *, include_local_seg: bool = False):
        models = list(get_list_of_models())
        if include_local_seg and 'local_seg' not in models:
            models.append('local_seg')
        return models

    def real_time_trackers(self):
        return get_list_of_real_time_trackers()

    def real_time_tracker_aliases(self, *, reverse: bool = False):
        return aliases_real_time_trackers(reverse=reverse)

    def model_arg_specs(self, acdc_segment):
        return getModelArgSpec(acdc_segment)

    def import_segmentation_module(self, model_name):
        return import_segment_module(model_name)

    def check_install_package(self, model_name):
        return check_install_package(model_name)

    def check_gpu_available(
        self,
        model_name,
        use_gpu,
        *,
        qparent=None,
        do_not_warn=False,
    ):
        return check_gpu_available(
            model_name,
            use_gpu,
            qparent=qparent,
            do_not_warn=do_not_warn,
        )

    def init_segmentation_model(self, acdc_segment, position_data, init_kwargs):
        return init_segm_model(acdc_segment, position_data, init_kwargs)

    def init_prompt_segmentation_model(
        self,
        acdc_prompt_segment,
        position_data,
        init_kwargs,
    ):
        return init_prompt_segm_model(
            acdc_prompt_segment,
            position_data,
            init_kwargs,
        )

    def init_tracker(self, position_data, tracker_name, **kwargs):
        return init_tracker(position_data, tracker_name, **kwargs)

    def validate_tracker_input(self, tracker, segmentation_video):
        return validate_tracker_input(tracker, segmentation_video)

    def log_segmentation_params(
        self,
        model_name,
        init_params,
        segment_params,
        *,
        logger_func=print,
        preproc_recipe=None,
        apply_post_process=False,
        standard_postprocess_kwargs=None,
        custom_postprocess_features=None,
    ):
        return log_segm_params(
            model_name,
            init_params,
            segment_params,
            logger_func=logger_func,
            preproc_recipe=preproc_recipe,
            apply_post_process=apply_post_process,
            standard_postprocess_kwargs=standard_postprocess_kwargs,
            custom_postprocess_features=custom_postprocess_features,
        )

    def store_custom_model_path(self, model_file_path):
        return store_custom_model_path(model_file_path)

    def store_custom_promptable_model_path(self, model_file_path):
        return store_custom_promptable_model_path(model_file_path)

    def set_default_arg_specs_from_kwargs(self, params, kwargs):
        return setDefaultValueArgSpecsFromKwargs(params, kwargs)

    def insert_model_arg_spec(
        self,
        params,
        param_name,
        param_value,
        *,
        param_type=None,
        desc='',
        docstring='',
    ):
        return insertModelArgSpec(
            params,
            param_name,
            param_value,
            param_type=param_type,
            desc=desc,
            docstring=docstring,
        )
