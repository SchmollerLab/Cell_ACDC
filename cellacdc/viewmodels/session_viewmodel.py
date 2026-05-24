"""View-model commands for session frame state."""

from __future__ import annotations

import os

import pandas as pd

from cellacdc.models.session_model import SessionModel
from cellacdc.domain import PositionSession
from cellacdc.domain.visited_frames import (
    LastVisitedFrameUpdate,
    update_last_visited_frame_state,
)

from .cca_edits_viewmodel import CcaEditViewModel
from .frame_metadata_viewmodel import FrameMetadataViewModel
from .tables_viewmodel import TableViewModel
from .workspace_viewmodel import WorkspaceViewModel


DEFAULT_SESSION_SETTINGS = {
    'is_bw_inverted': 'No',
    'fontSize': 12,
    'overlayColor': '255-255-0',
    'how_normIntensities': 'raw',
    'isLabelsVisible': 'No',
    'isNextFrameVisible': 'No',
    'isRightImageVisible': 'Yes',
    'manual_separate_draw_mode': 'threepoints_arc',
}


class SessionViewModel:
    """Application-facing commands for session progress state."""

    def __init__(
        self,
        model: SessionModel | None = None,
        *,
        cca_edits: CcaEditViewModel | None = None,
        frame_metadata: FrameMetadataViewModel | None = None,
        tables: TableViewModel | None = None,
        workspace: WorkspaceViewModel | None = None,
    ):
        self.model = model or SessionModel()
        self.cca_edits = cca_edits or CcaEditViewModel()
        self.frame_metadata = frame_metadata or FrameMetadataViewModel()
        self.tables = tables or TableViewModel()
        self.workspace = workspace or WorkspaceViewModel()

    def recent_paths(self, recent_paths_path) -> list[str]:
        if not os.path.exists(recent_paths_path):
            return []

        recent_paths_df = pd.read_csv(recent_paths_path, index_col='index')
        recent_paths_df['path'] = recent_paths_df['path'].str.replace('\\', '/')
        recent_paths_df = recent_paths_df.drop_duplicates(subset=['path'])
        recent_paths_df.to_csv(recent_paths_path)
        if 'opened_last_on' in recent_paths_df.columns:
            recent_paths_df = recent_paths_df.sort_values(
                'opened_last_on',
                ascending=False,
            )
        return recent_paths_df['path'].to_list()

    def load_settings(self, settings_csv_path) -> pd.DataFrame:
        if os.path.exists(settings_csv_path):
            settings_df = pd.read_csv(settings_csv_path, index_col='setting')
            settings_df['value'] = settings_df['value'].astype(object)
            if 'is_bw_inverted' in settings_df.index:
                settings_df.loc['is_bw_inverted'] = (
                    settings_df.loc['is_bw_inverted'].astype(str)
                )
            else:
                settings_df.at['is_bw_inverted', 'value'] = 'No'
            if 'how_normIntensities' not in settings_df.index:
                raw = 'Do not normalize. Display raw image'
                settings_df.at['how_normIntensities', 'value'] = raw
        else:
            settings_df = pd.DataFrame(
                {
                    'setting': list(DEFAULT_SESSION_SETTINGS.keys())[:4],
                    'value': list(DEFAULT_SESSION_SETTINGS.values())[:4],
                }
            ).set_index('setting')

        for key, value in DEFAULT_SESSION_SETTINGS.items():
            if key not in settings_df.index:
                settings_df.at[key, 'value'] = value

        return settings_df

    def position_session_from_load_data(self, pos_data) -> PositionSession:
        return PositionSession.from_loadData(pos_data)

    def update_last_visited_frame(
        self,
        mode: str,
        last_visited_frame_i: int,
        *,
        last_tracked_i: int,
        last_cca_frame_i: int,
    ) -> LastVisitedFrameUpdate:
        return update_last_visited_frame_state(
            mode,
            last_visited_frame_i,
            last_tracked_i=last_tracked_i,
            last_cca_frame_i=last_cca_frame_i,
        )

    def should_store_frame_data(
        self,
        *,
        frame_i: int,
        mode: str,
        enforce: bool,
    ) -> bool:
        return self.model.should_store_frame_data(
            frame_i=frame_i,
            mode=mode,
            enforce=enforce,
        )

    def should_disable_load_position(self, position_count: int) -> bool:
        return self.model.should_disable_load_position(position_count)

    def empty_labels(
        self,
        *,
        is_3d: bool,
        size_z: int,
        size_y: int,
        size_x: int,
    ):
        return self.model.empty_labels(
            is_3d=is_3d,
            size_z=size_z,
            size_y=size_y,
            size_x=size_x,
        )

    def should_resume_last_session_prompt(self, last_tracked_num: int) -> bool:
        return self.model.should_resume_last_session_prompt(last_tracked_num)
