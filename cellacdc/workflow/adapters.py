from __future__ import annotations

from typing import Any

from .runnable import RunnableConfig
from .state import WorkflowContext


def workflow_context_from_segm_kernel(kernel: Any) -> WorkflowContext:
    """Build a mutable workflow context from a SegmKernel instance."""
    return WorkflowContext(
        user_ch_name=kernel.user_ch_name,
        segm_endname=kernel.segm_endname,
        model_name=kernel.model_name,
        tracker_name=kernel.tracker_name,
        do_tracking=kernel.do_tracking,
        do_postprocess=kernel.do_postprocess,
        do_save=kernel.do_save,
        is_segm_3d=kernel.isSegm3D,
        use_roi=kernel.use_ROI,
        use_freehand_roi=kernel.use_freehand_ROI,
        use_3d_data_for_2d_segm=kernel.use3DdataFor2Dsegm,
        second_channel_name=kernel.second_channel_name,
        image_channel_tracker=kernel.image_channel_tracker,
        size_t=kernel.SizeT,
        size_z=kernel.SizeZ,
        model_kwargs=dict(kernel.model_kwargs or {}),
        init_model_kwargs=dict(kernel.init_model_kwargs or {}),
        track_params=dict(kernel.track_params or {}),
        init_tracker_kwargs=dict(kernel.init_tracker_kwargs or {}),
        standard_postprocess_kwargs=dict(kernel.standard_postrocess_kwargs or {}),
        custom_postprocess_features=dict(kernel.custom_postproc_features or {}),
        custom_postprocess_grouped_features=dict(
            kernel.custom_postproc_grouped_features or {}
        ),
        preproc_recipe=kernel.preproc_recipe,
        reduce_memory_usage=getattr(kernel, "reduce_memory_usage", False),
        model=kernel.model,
        tracker=kernel.tracker,
        is_segment3dt_available=kernel.is_segment3DT_available,
        inner_pbar_available=kernel.innerPbar_available,
        signals=kernel.signals,
    )


def sync_segm_kernel_from_context(kernel: Any, ctx: WorkflowContext) -> None:
    """Copy mutable pipeline resources back onto the kernel after a graph run."""
    kernel.model = ctx.model
    kernel.tracker = ctx.tracker
    kernel.is_segment3DT_available = ctx.is_segment3dt_available
    kernel.model_kwargs = ctx.model_kwargs
    kernel.track_params = ctx.track_params
    kernel.init_model_kwargs = ctx.init_model_kwargs


def runnable_config_from_segm_kernel(kernel: Any) -> RunnableConfig:
    return RunnableConfig(
        logger_func=kernel.logger_func,
        signals=kernel.signals,
        metadata={"model_name": kernel.model_name},
    )


def update_workflow_context_from_segm_kernel(
    ctx: WorkflowContext, kernel: Any
) -> WorkflowContext:
    """Refresh context fields that may change between batch positions."""
    ctx.model = kernel.model
    ctx.tracker = kernel.tracker
    ctx.is_segment3dt_available = kernel.is_segment3DT_available
    ctx.model_kwargs = dict(kernel.model_kwargs or {})
    ctx.track_params = dict(kernel.track_params or {})
    ctx.signals = kernel.signals
    return ctx


def _parse_custom_postproc_features_grouped(workflow_params: dict[str, Any]) -> dict:
    custom_postproc_grouped_features: dict[str, Any] = {}
    for section, options in workflow_params.items():
        if not section.startswith("postprocess_features."):
            continue
        category = section.split(".")[-1]
        for option, value in options.items():
            if option == "names":
                values = value.strip("\n").strip().split("\n")
                custom_postproc_grouped_features[category] = values
                continue
            channel = option
            if category not in custom_postproc_grouped_features:
                custom_postproc_grouped_features[category] = {channel: [value]}
            elif channel not in custom_postproc_grouped_features[category]:
                custom_postproc_grouped_features[category][channel] = [value]
            else:
                custom_postproc_grouped_features[category][channel].append(value)
    return custom_postproc_grouped_features


def workflow_context_from_ini(workflow_params: dict[str, Any]) -> WorkflowContext:
    """Build a workflow context directly from parsed INI workflow parameters."""
    from cellacdc import config

    initialization = workflow_params["initialization"]
    return WorkflowContext(
        user_ch_name=initialization["user_ch_name"],
        segm_endname=initialization.get("segm_endname", "segm.npz"),
        model_name=initialization.get("model_name", ""),
        tracker_name=initialization.get("tracker_name", ""),
        do_tracking=initialization.get("do_tracking", False),
        do_postprocess=initialization.get("do_postprocess", True),
        do_save=initialization.get("do_save", True),
        is_segm_3d=initialization.get("isSegm3D", False),
        use_roi=initialization.get("use_ROI", True),
        use_freehand_roi=initialization.get("use_freehand_ROI", True),
        use_3d_data_for_2d_segm=initialization.get("use3DdataFor2Dsegm", False),
        second_channel_name=initialization.get("second_channel_name"),
        image_channel_tracker=initialization.get("image_channel_tracker"),
        size_t=workflow_params["metadata"]["SizeT"],
        size_z=workflow_params["metadata"]["SizeZ"],
        model_kwargs=dict(workflow_params.get("segmentation_model_params", {})),
        init_model_kwargs=dict(
            workflow_params.get("init_segmentation_model_params", {})
        ),
        track_params=dict(workflow_params.get("tracker_params", {})),
        init_tracker_kwargs=dict(workflow_params.get("init_tracker_params", {})),
        standard_postprocess_kwargs=dict(
            workflow_params.get("standard_postprocess_features", {})
        ),
        custom_postprocess_features=dict(
            workflow_params.get("custom_postprocess_features", {})
        ),
        custom_postprocess_grouped_features=_parse_custom_postproc_features_grouped(
            workflow_params
        ),
        preproc_recipe=config.preprocess_ini_items_to_recipe(workflow_params),
        reduce_memory_usage=initialization.get("reduce_memory_usage", False),
    )


def interactive_segm_context_from_main_win(main_win, second_channel_data=None, z_range=None):
    from cellacdc.workflow.state import InteractiveSegmContext

    return InteractiveSegmContext(
        model=main_win.model,
        model_kwargs=main_win.model_kwargs,
        apply_postprocessing=main_win.applyPostProcessing,
        standard_postprocess_kwargs=main_win.standardPostProcessKwargs,
        custom_postprocess_features=main_win.customPostProcessFeatures,
        custom_postprocess_grouped_features=main_win.customPostProcessGroupedFeatures,
        segment_3d=main_win.segment3D,
        second_channel_data=second_channel_data,
        z_range=z_range,
    )


def runnable_config_from_main_win(main_win):
    return RunnableConfig(logger_func=main_win.logger.info)


def interactive_video_segm_context_from_worker(worker) -> InteractiveSegmContext:
    from cellacdc.workflow.state import InteractiveVideoSegmContext

    return InteractiveVideoSegmContext(
        model=worker.model,
        model_kwargs=worker.model_kwargs,
        apply_postprocessing=worker.applyPostProcessing,
        standard_postprocess_kwargs=worker.standardPostProcessKwargs,
        custom_postprocess_features=worker.customPostProcessFeatures,
        custom_postprocess_grouped_features=worker.customPostProcessGroupedFeatures,
        preproc_recipe=worker.preproc_recipe,
        second_channel_data=getattr(worker, "secondChannelData", None),
        start_frame_num=worker.startFrameNum,
        stop_frame_num=worker.stopFrameNum,
        progress_callback=worker.progressBar,
        logger_func=worker.logger.log,
    )
