"""Scriptable model rules for image control widgets."""

from __future__ import annotations


class ImageControlsModel:
    """Headless defaults for image-control UI construction."""

    draw_ids_cont_combo_items = (
        'Draw IDs and contours',
        'Draw IDs and overlay segm. masks',
        'Draw only cell cycle info',
        'Draw cell cycle info and contours',
        'Draw cell cycle info and overlay segm. masks',
        'Draw only mother-bud lines',
        'Draw only IDs',
        'Draw only contours',
        'Draw only overlay segm. masks',
        'Draw nothing',
    )
    z_projection_options = (
        'single z-slice',
        'max z-projection',
        'mean z-projection',
        'median z-proj.',
    )
    overlay_z_projection_options = (
        'single z-slice',
        'max z-projection',
        'mean z-projection',
        'median z-proj.',
        'same as above',
    )
    bottom_layout_zoom_values = tuple(range(50, 151, 10))

    def bottom_layout_zoom_percent(self, df_settings) -> int:
        if 'bottom_sliders_zoom_perc' not in df_settings.index:
            return 100
        return int(df_settings.at['bottom_sliders_zoom_perc', 'value'])

    def retain_space_hidden_sliders(self, df_settings) -> bool:
        if 'retain_space_hidden_sliders' not in df_settings.index:
            return True
        return df_settings.at['retain_space_hidden_sliders', 'value'] == 'Yes'
