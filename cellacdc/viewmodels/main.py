"""Main GUI view-model composition root."""

from __future__ import annotations

from dataclasses import dataclass, field

from .app_shell_viewmodel import AppShellViewModel
from .actions_viewmodel import ActionsViewModel
from .annotation_display_viewmodel import AnnotationDisplayViewModel
from .brush_tools_viewmodel import BrushToolsViewModel
from .canvas_context_menu_viewmodel import CanvasContextMenuViewModel
from .canvas_drawing_viewmodel import CanvasDrawingViewModel
from .canvas_events_viewmodel import CanvasEventsViewModel
from .canvas_hover_viewmodel import CanvasHoverViewModel
from .canvas_right_image_viewmodel import CanvasRightImageViewModel
from .canvas_selection_viewmodel import CanvasSelectionViewModel
from .canvas_tool_viewmodel import CanvasToolViewModel
from .combine_viewmodel import CombineViewModel
from .cell_cycle_viewmodel import CellCycleViewModel
from .cca_edits import CcaEditViewModel
from .cca_workflows import CcaWorkflowViewModel
from .curvature_viewmodel import CurvatureViewModel
from .custom_annotations_viewmodel import CustomAnnotationsViewModel
from .data_loading_viewmodel import DataLoadingViewModel
from .deleted_rois_viewmodel import DeletedRoisViewModel
from .display_decorations_viewmodel import DisplayDecorationsViewModel
from .draw_clear_region_viewmodel import DrawClearRegionViewModel
from .edit_id import EditIdViewModel
from .exporting_viewmodel import ExportingViewModel
from .frame_metadata import FrameMetadataViewModel
from .frame_navigation_viewmodel import FrameNavigationViewModel
from .formatting import FormattingViewModel
from .geometry import GeometryViewModel
from .graphics_viewmodel import GraphicsViewModel
from .image_controls_viewmodel import ImageControlsViewModel
from .image_display_viewmodel import ImageDisplayViewModel
from .label_editing_viewmodel import LabelEditingViewModel
from .label_edits import LabelEditViewModel
from .label_roi_viewmodel import LabelRoiViewModel
from .label_transform_tools_viewmodel import LabelTransformToolsViewModel
from .layout_controls_viewmodel import LayoutControlsViewModel
from .lineage import LineageViewModel
from .lineage_interactions_viewmodel import LineageInteractionsViewModel
from .magic_prompts_viewmodel import MagicPromptsViewModel
from .main_menu_viewmodel import MainMenuViewModel
from .main_toolbar_viewmodel import MainToolbarViewModel
from .measurements_viewmodel import MeasurementsViewModel
from .mode_controls_viewmodel import ModeControlsViewModel
from .model_registry import ModelRegistryViewModel
from .object_counts import ObjectCountViewModel
from .object_cleanup_viewmodel import ObjectCleanupViewModel
from .object_properties_viewmodel import ObjectPropertiesViewModel
from .object_search_viewmodel import ObjectSearchViewModel
from .points import PointsViewModel
from .points_layers_viewmodel import PointsLayersViewModel
from .preprocessing_viewmodel import PreprocessingViewModel
from .quick_settings_viewmodel import QuickSettingsViewModel
from .saving_viewmodel import SavingViewModel
from .seg_for_lost_ids_viewmodel import SegForLostIdsViewModel
from .segmentation_viewmodel import SegmentationViewModel
from .session_viewmodel import SessionViewModel
from .status_hover_viewmodel import StatusHoverViewModel
from .tables import TableViewModel
from .tool_activation_viewmodel import ToolActivationViewModel
from .tracking_viewmodel import TrackingViewModel
from .undo_redo_viewmodel import UndoRedoViewModel
from .whitelist_viewmodel import WhitelistViewModel
from .worker_viewmodel import WorkerViewModel
from .window_events_viewmodel import WindowEventsViewModel
from .workspace import WorkspaceViewModel


@dataclass(frozen=True)
class MainGuiViewModel:
    """Application-facing commands available to the Qt GUI."""

    actions: ActionsViewModel = field(default_factory=ActionsViewModel)
    annotation_display: AnnotationDisplayViewModel = field(
        default_factory=AnnotationDisplayViewModel
    )
    app_shell: AppShellViewModel = field(default_factory=AppShellViewModel)
    brush_tools: BrushToolsViewModel = field(default_factory=BrushToolsViewModel)
    canvas_context_menu: CanvasContextMenuViewModel = field(
        default_factory=CanvasContextMenuViewModel
    )
    canvas_drawing: CanvasDrawingViewModel = field(
        default_factory=CanvasDrawingViewModel
    )
    canvas_events: CanvasEventsViewModel = field(
        default_factory=CanvasEventsViewModel
    )
    canvas_hover: CanvasHoverViewModel = field(
        default_factory=CanvasHoverViewModel
    )
    canvas_right_image: CanvasRightImageViewModel = field(
        default_factory=CanvasRightImageViewModel
    )
    canvas_selection: CanvasSelectionViewModel = field(
        default_factory=CanvasSelectionViewModel
    )
    canvas_tools: CanvasToolViewModel = field(
        default_factory=CanvasToolViewModel
    )
    combine: CombineViewModel = field(
        default_factory=CombineViewModel
    )
    cell_cycle: CellCycleViewModel = field(
        default_factory=CellCycleViewModel
    )
    cca_edits: CcaEditViewModel = field(default_factory=CcaEditViewModel)
    cca_workflows: CcaWorkflowViewModel = field(
        default_factory=CcaWorkflowViewModel
    )
    curvature: CurvatureViewModel = field(default_factory=CurvatureViewModel)
    custom_annotations: CustomAnnotationsViewModel = field(
        default_factory=CustomAnnotationsViewModel
    )
    data_loading: DataLoadingViewModel = field(
        default_factory=DataLoadingViewModel
    )
    deleted_rois: DeletedRoisViewModel = field(
        default_factory=DeletedRoisViewModel
    )
    display_decorations: DisplayDecorationsViewModel = field(
        default_factory=DisplayDecorationsViewModel
    )
    draw_clear_region: DrawClearRegionViewModel = field(
        default_factory=DrawClearRegionViewModel
    )
    edit_id: EditIdViewModel = field(default_factory=EditIdViewModel)
    exporting: ExportingViewModel = field(default_factory=ExportingViewModel)
    frame_navigation: FrameNavigationViewModel = field(
        default_factory=FrameNavigationViewModel
    )
    frame_metadata: FrameMetadataViewModel = field(
        default_factory=FrameMetadataViewModel
    )
    formatting: FormattingViewModel = field(default_factory=FormattingViewModel)
    geometry: GeometryViewModel = field(default_factory=GeometryViewModel)
    graphics: GraphicsViewModel = field(default_factory=GraphicsViewModel)
    image_controls: ImageControlsViewModel = field(
        default_factory=ImageControlsViewModel
    )
    image_display: ImageDisplayViewModel = field(
        default_factory=ImageDisplayViewModel
    )
    label_editing: LabelEditingViewModel = field(
        default_factory=LabelEditingViewModel
    )
    label_edits: LabelEditViewModel = field(default_factory=LabelEditViewModel)
    label_roi: LabelRoiViewModel = field(default_factory=LabelRoiViewModel)
    label_transform_tools: LabelTransformToolsViewModel = field(
        default_factory=LabelTransformToolsViewModel
    )
    layout_controls: LayoutControlsViewModel = field(
        default_factory=LayoutControlsViewModel
    )
    lineage: LineageViewModel = field(default_factory=LineageViewModel)
    lineage_interactions: LineageInteractionsViewModel = field(
        default_factory=LineageInteractionsViewModel
    )
    magic_prompts: MagicPromptsViewModel = field(
        default_factory=MagicPromptsViewModel
    )
    main_menu: MainMenuViewModel = field(default_factory=MainMenuViewModel)
    main_toolbar: MainToolbarViewModel = field(
        default_factory=MainToolbarViewModel
    )
    measurements: MeasurementsViewModel = field(
        default_factory=MeasurementsViewModel
    )
    mode_controls: ModeControlsViewModel = field(
        default_factory=ModeControlsViewModel
    )
    model_registry: ModelRegistryViewModel = field(
        default_factory=ModelRegistryViewModel
    )
    object_search: ObjectSearchViewModel = field(
        default_factory=ObjectSearchViewModel
    )
    object_counts: ObjectCountViewModel = field(
        default_factory=ObjectCountViewModel
    )
    object_cleanup: ObjectCleanupViewModel = field(
        default_factory=ObjectCleanupViewModel
    )
    object_properties: ObjectPropertiesViewModel = field(
        default_factory=ObjectPropertiesViewModel
    )
    points: PointsViewModel = field(default_factory=PointsViewModel)
    points_layers: PointsLayersViewModel = field(
        default_factory=PointsLayersViewModel
    )
    preprocessing: PreprocessingViewModel = field(
        default_factory=PreprocessingViewModel
    )
    quick_settings: QuickSettingsViewModel = field(
        default_factory=QuickSettingsViewModel
    )
    saving: SavingViewModel = field(default_factory=SavingViewModel)
    seg_for_lost_ids: SegForLostIdsViewModel = field(
        default_factory=SegForLostIdsViewModel
    )
    segmentation: SegmentationViewModel = field(
        default_factory=SegmentationViewModel
    )
    session: SessionViewModel = field(default_factory=SessionViewModel)
    status_hover: StatusHoverViewModel = field(
        default_factory=StatusHoverViewModel
    )
    tables: TableViewModel = field(default_factory=TableViewModel)
    tool_activation: ToolActivationViewModel = field(
        default_factory=ToolActivationViewModel
    )
    tracking: TrackingViewModel = field(default_factory=TrackingViewModel)
    undo_redo: UndoRedoViewModel = field(default_factory=UndoRedoViewModel)
    whitelist: WhitelistViewModel = field(
        default_factory=WhitelistViewModel
    )
    worker: WorkerViewModel = field(default_factory=WorkerViewModel)
    window_events: WindowEventsViewModel = field(
        default_factory=WindowEventsViewModel
    )
    workspace: WorkspaceViewModel = field(default_factory=WorkspaceViewModel)
