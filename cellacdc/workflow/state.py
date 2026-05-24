from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any


def merge_state(state: Any, update: dict[str, Any] | None) -> Any:
    if not update:
        return state
    if isinstance(state, dict):
        return {**state, **update}
    return replace(state, **update)


@dataclass(slots=True)
class WorkflowContext:
    """Immutable workflow configuration (LangGraph context_schema analogue)."""

    user_ch_name: str
    segm_endname: str = "segm.npz"
    model_name: str = ""
    tracker_name: str = ""
    do_tracking: bool = False
    do_postprocess: bool = True
    do_save: bool = True
    is_segm_3d: bool = False
    use_roi: bool = True
    use_freehand_roi: bool = True
    use_3d_data_for_2d_segm: bool = False
    second_channel_name: str | None = None
    image_channel_tracker: str | None = None
    size_t: int = 1
    size_z: int = 1
    model_kwargs: dict[str, Any] = field(default_factory=dict)
    init_model_kwargs: dict[str, Any] = field(default_factory=dict)
    track_params: dict[str, Any] = field(default_factory=dict)
    init_tracker_kwargs: dict[str, Any] = field(default_factory=dict)
    standard_postprocess_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_postprocess_features: dict[str, Any] = field(default_factory=dict)
    custom_postprocess_grouped_features: dict[str, Any] = field(default_factory=dict)
    preproc_recipe: list[dict[str, Any]] | None = None
    reduce_memory_usage: bool = False
    model: Any | None = None
    tracker: Any | None = None
    is_segment3dt_available: bool = False
    inner_pbar_available: bool = False
    signals: Any | None = None


@dataclass(slots=True)
class PositionState:
    """Mutable per-position pipeline state (LangGraph state_schema analogue)."""

    img_path: str
    stop_frame_n: int = 1
    pos_data: Any | None = None
    img_data: Any | None = None
    second_ch_data: Any | None = None
    postprocess_img: Any | None = None
    lab_stack: Any | None = None
    tracked_stack: Any | None = None
    model: Any | None = None
    tracker: Any | None = None
    is_roi_active: bool = False
    pad_info: tuple | None = None
    roi_bounds: tuple[int, int, int, int] | None = None
    stop_i: int = 1
    t0: int = 0
    aborted: bool = False
    error: str | None = None

    def as_update(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BatchState:
    """Outer loop over many positions."""

    paths: list[str] = field(default_factory=list)
    stop_frame_numbers: list[int] = field(default_factory=list)
    current_index: int = 0
    results: list[Any] = field(default_factory=list)
    aborted: bool = False

    @property
    def done(self) -> bool:
        return self.current_index >= len(self.paths)

    @property
    def current_path(self) -> str | None:
        if self.done:
            return None
        return self.paths[self.current_index]

    @property
    def current_stop_frame(self) -> int:
        if not self.stop_frame_numbers:
            return 1
        index = min(self.current_index, len(self.stop_frame_numbers) - 1)
        return int(self.stop_frame_numbers[index])


@dataclass(slots=True)
class BatchWorkflowContext:
    """Context for batch parent graphs."""

    position_ctx: WorkflowContext
    position_graph: Any | None = None


@dataclass(slots=True)
class MeasurementsContext:
    """Context for measurements position pipeline."""

    end_filename_segm: str
    kernel: Any
    save_metrics: bool = True
    last_cca_frame_i: Any | None = None


@dataclass(slots=True)
class MeasurementsBatchContext:
    measurements_ctx: MeasurementsContext
    position_graph: Any | None = None


@dataclass(slots=True)
class MeasurementsState:
    img_path: str = ""
    stop_frame_n: int = 1
    pos_data: Any | None = None
    skipped: bool = False
    aborted: bool = False
    error: str | None = None


@dataclass(slots=True)
class InteractiveSegmContext:
    """Context for in-viewer single-frame segmentation."""

    model: Any
    model_kwargs: dict[str, Any]
    apply_postprocessing: bool = False
    standard_postprocess_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_postprocess_features: dict[str, Any] = field(default_factory=dict)
    custom_postprocess_grouped_features: dict[str, Any] = field(default_factory=dict)
    segment_3d: bool = False
    second_channel_data: Any | None = None
    z_range: tuple[int, int] | None = None


@dataclass(slots=True)
class InteractiveSegmState:
    main_win: Any
    pos_data: Any | None = None
    img: Any | None = None
    lab: Any | None = None
    segmented_lab: Any | None = None
    start_z_slice: int = 0
    exec_time: float = 0.0


@dataclass(slots=True)
class InteractiveVideoSegmContext:
    """Context for in-viewer timelapse segmentation."""

    model: Any
    model_kwargs: dict[str, Any]
    apply_postprocessing: bool = False
    standard_postprocess_kwargs: dict[str, Any] = field(default_factory=dict)
    custom_postprocess_features: dict[str, Any] = field(default_factory=dict)
    custom_postprocess_grouped_features: dict[str, Any] = field(default_factory=dict)
    preproc_recipe: list[dict[str, Any]] | None = None
    second_channel_data: Any | None = None
    start_frame_num: int = 1
    stop_frame_num: int = 1
    progress_callback: Any | None = None
    logger_func: Any = print


@dataclass(slots=True)
class InteractiveVideoSegmState:
    pos_data: Any
    segm_data: Any | None = None
    img_data: Any | None = None
    z_slices: Any | None = None
    exec_time: float = 0.0


@dataclass(slots=True)
class MeasurementsGuiContext:
    """Context for GUI-driven measurements runs."""

    kernel: Any
    compute_metrics_worker: Any | None = None
    save_data_worker: Any | None = None
    save_metrics: bool = True
    do_init_metrics: bool = True
    last_cca_frame_i: Any | None = None
    end_filename_segm: str = ""


@dataclass(slots=True)
class MeasurementsGuiState:
    img_path: str = ""
    stop_frame_n: int = 1
    pos_data: Any | None = None
    skipped: bool = False
    aborted: bool = False
    acdc_df_li: list[Any] = field(default_factory=list)
    keys: list[Any] = field(default_factory=list)


@dataclass(slots=True)
class MeasurementsGuiBatchContext:
    kernel: Any
    compute_metrics_worker: Any | None = None
    save_data_worker: Any | None = None
    save_metrics: bool = True
    end_filename_segm: str = ""


@dataclass(slots=True)
class FullWorkflowState:
    """Top-level INI workflow state."""

    segm_params: dict[str, Any] = field(default_factory=dict)
    measurements_params: dict[str, Any] | None = None
    run_segm: bool = True
    run_measurements: bool = False
    segm_done: bool = False
    measurements_done: bool = False


def state_field_names(state_type: type) -> set[str]:
    if hasattr(state_type, "__dataclass_fields__"):
        return set(state_type.__dataclass_fields__)
    return {field.name for field in fields(state_type)}
