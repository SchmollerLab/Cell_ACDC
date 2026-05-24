"""Main GUI view-model composition root."""

from __future__ import annotations

from dataclasses import dataclass, field

from .cca_edits_viewmodel import CcaEditViewModel
from .cca_workflows_viewmodel import CcaWorkflowViewModel
from .edit_id_viewmodel import EditIdViewModel
from .frame_metadata_viewmodel import FrameMetadataViewModel
from .formatting_viewmodel import FormattingViewModel
from .geometry_viewmodel import GeometryViewModel
from .label_edits_viewmodel import LabelEditViewModel
from .lineage_viewmodel import LineageViewModel
from .model_registry_viewmodel import ModelRegistryViewModel
from .object_counts_viewmodel import ObjectCountViewModel
from .points_viewmodel import PointsViewModel
from .tables_viewmodel import TableViewModel
from .workspace_viewmodel import WorkspaceViewModel


@dataclass(frozen=True)
class MainGuiViewModel:
    """Application-facing commands available to the Qt GUI."""    cca_edits: CcaEditViewModel = field(default_factory=CcaEditViewModel)
    cca_workflows: CcaWorkflowViewModel = field(
        default_factory=CcaWorkflowViewModel
    )    edit_id: EditIdViewModel = field(default_factory=EditIdViewModel)    frame_metadata: FrameMetadataViewModel = field(
        default_factory=FrameMetadataViewModel
    )
    formatting: FormattingViewModel = field(default_factory=FormattingViewModel)
    geometry: GeometryViewModel = field(default_factory=GeometryViewModel)    label_edits: LabelEditViewModel = field(default_factory=LabelEditViewModel)    lineage: LineageViewModel = field(default_factory=LineageViewModel)    model_registry: ModelRegistryViewModel = field(
        default_factory=ModelRegistryViewModel
    )    object_counts: ObjectCountViewModel = field(
        default_factory=ObjectCountViewModel
    )    points: PointsViewModel = field(default_factory=PointsViewModel)    tables: TableViewModel = field(default_factory=TableViewModel)    workspace: WorkspaceViewModel = field(default_factory=WorkspaceViewModel)
