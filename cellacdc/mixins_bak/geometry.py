"""View-model commands for geometric interaction helpers."""

from __future__ import annotations

from cellacdc.core import get_line, get_obj_contours
from cellacdc.core._legacy import _compute_all_obj_to_obj_contour_dist_pairs
from cellacdc.myutils import get_slices_local_into_global_arr, is_in_bounds
from cellacdc.transformation import crop_2D, snap_xy_to_closest_angle


class GeometryMixin:
    """Application-facing commands for geometric interaction transforms."""

    def crop_2d(
        self,
        image,
        xy_range,
        *,
        tolerance=0,
        return_copy=True,
    ):
        return crop_2D(
            image,
            xy_range,
            tolerance=tolerance,
            return_copy=return_copy,
        )

    def is_configured_middle_click(
        self,
        *,
        mouse_button,
        configured_button,
        key_sequence_is_none: bool,
        tool_is_checked: bool,
    ) -> bool:
        if key_sequence_is_none:
            is_del_object_active = True
        else:
            is_del_object_active = tool_is_checked
        return mouse_button == configured_button and is_del_object_active

    def is_default_middle_click(
        self,
        *,
        mouse_button,
        modifiers,
        is_mac: bool,
        brush_is_checked: bool,
        left_button,
        middle_button,
        control_modifier,
    ) -> bool:
        if is_mac:
            return (
                mouse_button == left_button
                and modifiers == control_modifier
                and not brush_is_checked
            )
        return mouse_button == middle_button

    def is_in_bounds(self, x, y, width, height):
        return is_in_bounds(x, y, width, height)

    def is_pan_image_click(
        self,
        *,
        mouse_button,
        left_button,
        modifiers,
        alt_modifier,
    ) -> bool:
        return modifiers == alt_modifier and mouse_button == left_button

    def line_coords(self, y1, x1, y2, x2, *, dashed=True):
        return get_line(y1, x1, y2, x2, dashed=dashed)

    def local_to_global_slices(self, bbox_coords, global_shape):
        return get_slices_local_into_global_arr(bbox_coords, global_shape)

    def middle_click_text(
        self,
        *,
        has_del_object_action: bool,
        is_mac: bool,
        button_name: str | None = None,
        key_sequence_text: str | None = None,
    ) -> str:
        if not has_del_object_action and is_mac:
            return "Command + Left Click"
        if not has_del_object_action:
            return "Middle Click"
        if key_sequence_text is None:
            return button_name
        return f"{key_sequence_text} + {button_name}"

    def object_contours(
        self,
        *,
        obj=None,
        obj_image=None,
        obj_bbox=None,
        all_external=False,
        all=False,
        only_longest_contour=True,
        local=False,
    ):
        return get_obj_contours(
            obj=obj,
            obj_image=obj_image,
            obj_bbox=obj_bbox,
            all_external=all_external,
            all=all,
            only_longest_contour=only_longest_contour,
            local=local,
        )

    def object_to_object_contour_distance_matrix(
        self,
        all_contours,
        regionprops,
        *,
        previous_regionprops=None,
        restrict_search=True,
    ):
        return _compute_all_obj_to_obj_contour_dist_pairs(
            all_contours,
            regionprops,
            prev_rp=previous_regionprops,
            restrict_search=restrict_search,
        )

    def should_auto_activate_viewer(
        self,
        *,
        is_data_loaded: bool,
        windows_overlap: bool,
        disable_auto_activate: bool,
    ) -> bool:
        return is_data_loaded and not windows_overlap and not disable_auto_activate

    def snap_xy_to_closest_angle(
        self,
        x0,
        y0,
        x1,
        y1,
        angle_factor=15,
    ):
        return snap_xy_to_closest_angle(
            x0,
            y0,
            x1,
            y1,
            angle_factor=angle_factor,
        )

    def windows_overlap_from_bounds(
        self,
        *,
        main_left,
        main_top,
        main_width,
        main_height,
        other_left,
        other_top,
    ) -> bool:
        main_right = main_left + main_width
        main_bottom = main_top + main_height
        return (other_top < main_bottom) and (other_left < main_right)
