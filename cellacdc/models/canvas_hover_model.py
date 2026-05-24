"""Scriptable model rules for canvas hover interactions."""

from __future__ import annotations

from typing import Any


class CanvasHoverModel:
    """Headless decisions for hover and cursor state."""

    def point_in_bounds(
        self,
        image_shape: tuple[int, int],
        xdata: int,
        ydata: int,
    ) -> bool:
        y_size, x_size = image_shape
        return 0 <= xdata < x_size and 0 <= ydata < y_size

    def hover_position(self, is_exit: bool, position) -> tuple[Any, Any]:
        if is_exit:
            return None, None
        return position

    def should_set_mirrored_cursor(
        self,
        *,
        override_cursor_is_none: bool,
        is_exit: bool,
        mirrored_cursor_enabled: bool,
        is_hover_img1: bool = True,
    ) -> bool:
        return (
            override_cursor_is_none
            and not is_exit
            and is_hover_img1
            and mirrored_cursor_enabled
        )

    def should_draw_ruler_line(
        self,
        *,
        ruler_checked: bool,
        add_deleted_polyline_checked: bool,
        temp_segment_on: bool,
        is_exit: bool,
    ) -> bool:
        return (
            (ruler_checked or add_deleted_polyline_checked)
            and temp_segment_on
            and not is_exit
        )

    def cursor_flags(
        self,
        *,
        is_exit: bool,
        no_modifier: bool,
        shift: bool,
        ctrl: bool,
        alt: bool,
        brush_checked: bool,
        eraser_checked: bool,
        add_deleted_polyline_checked: bool,
        label_roi_checked: bool,
        label_roi_circular_checked: bool,
        wand_checked: bool,
        move_label_checked: bool,
        expand_label_checked: bool,
        curvature_checked: bool,
        keep_ids_checked: bool,
        custom_annotation_available: bool,
        manual_tracking_checked: bool,
        manual_background_checked: bool,
        zoom_rect_checked: bool,
        edit_id_checked: bool,
        magic_prompts_checked: bool,
        points_layer_checked: bool,
        add_points_by_clicking_active: bool,
    ) -> dict[str, bool]:
        return {
            'setBrushCursor': (
                brush_checked and not is_exit and (no_modifier or shift or ctrl)
            ),
            'setEraserCursor': eraser_checked and not is_exit and no_modifier,
            'setAddDelPolyLineCursor': (
                add_deleted_polyline_checked and not is_exit and no_modifier
            ),
            'setLabelRoiCircCursor': (
                label_roi_checked
                and not is_exit
                and (no_modifier or shift or ctrl)
                and label_roi_circular_checked
            ),
            'setWandCursor': wand_checked and not is_exit and no_modifier,
            'setLabelRoiCursor': label_roi_checked and not is_exit and no_modifier,
            'setMoveLabelCursor': move_label_checked and not is_exit and no_modifier,
            'setExpandLabelCursor': (
                expand_label_checked and not is_exit and no_modifier
            ),
            'setCurvCursor': curvature_checked and not is_exit and no_modifier,
            'setKeepObjCursor': keep_ids_checked and not is_exit and no_modifier,
            'setCustomAnnotCursor': (
                custom_annotation_available and not is_exit and no_modifier
            ),
            'setManualTrackingCursor': (
                manual_tracking_checked and not is_exit and no_modifier
            ),
            'setManualBackgroundCursor': (
                manual_background_checked and not is_exit and no_modifier
            ),
            'setAddPointCursor': (
                (points_layer_checked or magic_prompts_checked)
                and add_points_by_clicking_active
                and not is_exit
                and no_modifier
            ),
            'setZoomRectCursor': zoom_rect_checked and not is_exit and no_modifier,
            'setEditIDCursor': edit_id_checked and not is_exit,
            'setPanImageCursor': alt and not is_exit,
        }
