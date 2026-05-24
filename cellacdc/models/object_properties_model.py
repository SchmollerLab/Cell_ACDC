"""Scriptable model rules for object-property workflows."""

from __future__ import annotations

import numpy as np


class ObjectPropertiesModel:
    """Headless decisions for object-property and highlight workflows."""

    def timelapse_default_categories(self) -> set[str]:
        return {
            'In current frame',
            'In all visited frames',
            'In entire video',
            'Unique objects in all visited frames',
            'Unique objects in entire video',
        }

    def snapshot_default_categories(self, *, is_segm_3d: bool) -> set[str]:
        categories = {
            'In current position',
            'In all visited positions (current session)',
            'In all visited positions (previous sessions)',
            'In all loaded positions',
        }
        if is_segm_3d:
            categories.add('In current z-slice')
        return categories

    def should_update_object_counts(
        self,
        *,
        window_exists: bool,
        is_visible: bool,
        live_preview_checked: bool,
    ) -> bool:
        return window_exists and is_visible and live_preview_checked

    def should_show_3d_property_controls(self, is_segm_3d: bool) -> bool:
        return is_segm_3d

    def should_highlight_props_id(
        self,
        *,
        dock_visible: bool,
        highlight_checked: bool,
        searched_highlight_checked: bool,
    ) -> bool:
        return (
            dock_visible
            and (highlight_checked or searched_highlight_checked)
        )

    def should_update_props_widget(
        self,
        *,
        dock_visible: bool,
        object_id: int,
        current_props_id: int,
    ) -> bool:
        return dock_visible and object_id != 0 and object_id != current_props_id

    def calculate_area_pxl(
        self,
        *,
        is_segm_3d: bool,
        z_proj_text: str,
        z_lab: int,
        bbox_0: int,
        obj_image: np.ndarray,
        obj_area: int,
    ) -> int:
        if is_segm_3d:
            if z_proj_text == 'single z-slice':
                local_z = z_lab - bbox_0
                return int(np.count_nonzero(obj_image[local_z]))
            else:
                return int(np.count_nonzero(obj_image.max(axis=0)))
        else:
            return obj_area

    def calculate_area_um2(
        self,
        *,
        area_pxl: int,
        physical_size_x: float,
        physical_size_y: float,
    ) -> float:
        return area_pxl * physical_size_y * physical_size_x

    def calculate_vol_3d(
        self,
        *,
        obj_area: int,
        physical_size_x: float,
        physical_size_y: float,
        physical_size_z: float,
    ) -> tuple[float, float]:
        vol_vox_3D = obj_area
        vol_fl_3D = vol_vox_3D * physical_size_z * physical_size_y * physical_size_x
        return float(vol_vox_3D), float(vol_fl_3D)

    def calculate_elongation(
        self,
        *,
        major_axis_length: float,
        minor_axis_length: float,
    ) -> float:
        minor_axis = max(1.0, minor_axis_length)
        return major_axis_length / minor_axis

    def get_object_and_background_images(
        self,
        *,
        image: np.ndarray,
        is_segm_3d: bool,
        pos_data_size_z: int,
        z_slice: int,
        obj_slice: tuple,
        obj_image: np.ndarray,
        img1_image: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if pos_data_size_z > 1 and not is_segm_3d:
            obj_data = image[z_slice][obj_slice][obj_image]
            img = img1_image if img1_image is not None else image[z_slice]
        else:
            obj_data = image[obj_slice][obj_image]
            img = image
        return obj_data, img

    def calculate_intensity_statistics(
        self,
        obj_data: np.ndarray,
    ) -> dict[str, float]:
        if obj_data.size == 0:
            return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'median': 0.0}
        return {
            'min': float(np.min(obj_data)),
            'max': float(np.max(obj_data)),
            'mean': float(np.mean(obj_data)),
            'median': float(np.median(obj_data)),
        }

    def calculate_additional_measure(
        self,
        *,
        func_desc: str,
        func: callable,
        obj_data: np.ndarray,
        img: np.ndarray,
        lab: np.ndarray,
        obj_area: int,
        vol_vox: float,
    ) -> float:
        if func_desc in ('Concentration', 'Amount'):
            background_pixels = img[lab == 0]
            bkgr_val = (
                float(np.median(background_pixels))
                if background_pixels.size > 0
                else 0.0
            )
            amount = func(obj_data, bkgr_val, obj_area)
            if func_desc == 'Concentration':
                return amount / vol_vox
            else:
                return amount
        else:
            return float(func(obj_data))

