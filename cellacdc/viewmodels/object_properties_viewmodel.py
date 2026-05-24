"""View-model contracts for object-property workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from cellacdc.models.object_properties_model import ObjectPropertiesModel

from .measurements_viewmodel import MeasurementsViewModel
from .object_counts import ObjectCountViewModel



@dataclass(frozen=True)
class ObjectPropertiesViewModel:
    """Application-facing object-property decisions and commands."""

    model: ObjectPropertiesModel = field(default_factory=ObjectPropertiesModel)
    measurements: MeasurementsViewModel = field(
        default_factory=MeasurementsViewModel
    )
    object_counts: ObjectCountViewModel = field(
        default_factory=ObjectCountViewModel
    )

    def timelapse_default_categories(self) -> set[str]:
        return self.model.timelapse_default_categories()

    def snapshot_default_categories(self, *, is_segm_3d: bool) -> set[str]:
        return self.model.snapshot_default_categories(is_segm_3d=is_segm_3d)

    def should_update_object_counts(
        self,
        *,
        window_exists: bool,
        is_visible: bool,
        live_preview_checked: bool,
    ) -> bool:
        return self.model.should_update_object_counts(
            window_exists=window_exists,
            is_visible=is_visible,
            live_preview_checked=live_preview_checked,
        )

    def should_show_3d_property_controls(self, is_segm_3d: bool) -> bool:
        return self.model.should_show_3d_property_controls(is_segm_3d)

    def should_highlight_props_id(
        self,
        *,
        dock_visible: bool,
        highlight_checked: bool,
        searched_highlight_checked: bool,
    ) -> bool:
        return self.model.should_highlight_props_id(
            dock_visible=dock_visible,
            highlight_checked=highlight_checked,
            searched_highlight_checked=searched_highlight_checked,
        )

    def should_update_props_widget(
        self,
        *,
        dock_visible: bool,
        object_id: int,
        current_props_id: int,
    ) -> bool:
        return self.model.should_update_props_widget(
            dock_visible=dock_visible,
            object_id=object_id,
            current_props_id=current_props_id,
        )

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
        return self.model.calculate_area_pxl(
            is_segm_3d=is_segm_3d,
            z_proj_text=z_proj_text,
            z_lab=z_lab,
            bbox_0=bbox_0,
            obj_image=obj_image,
            obj_area=obj_area,
        )

    def calculate_area_um2(
        self,
        *,
        area_pxl: int,
        physical_size_x: float,
        physical_size_y: float,
    ) -> float:
        return self.model.calculate_area_um2(
            area_pxl=area_pxl,
            physical_size_x=physical_size_x,
            physical_size_y=physical_size_y,
        )

    def calculate_vol_3d(
        self,
        *,
        obj_area: int,
        physical_size_x: float,
        physical_size_y: float,
        physical_size_z: float,
    ) -> tuple[float, float]:
        return self.model.calculate_vol_3d(
            obj_area=obj_area,
            physical_size_x=physical_size_x,
            physical_size_y=physical_size_y,
            physical_size_z=physical_size_z,
        )

    def calculate_elongation(
        self,
        *,
        major_axis_length: float,
        minor_axis_length: float,
    ) -> float:
        return self.model.calculate_elongation(
            major_axis_length=major_axis_length,
            minor_axis_length=minor_axis_length,
        )

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
        return self.model.get_object_and_background_images(
            image=image,
            is_segm_3d=is_segm_3d,
            pos_data_size_z=pos_data_size_z,
            z_slice=z_slice,
            obj_slice=obj_slice,
            obj_image=obj_image,
            img1_image=img1_image,
        )

    def calculate_intensity_statistics(
        self,
        obj_data: np.ndarray,
    ) -> dict[str, float]:
        return self.model.calculate_intensity_statistics(obj_data)

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
        return self.model.calculate_additional_measure(
            func_desc=func_desc,
            func=func,
            obj_data=obj_data,
            img=img,
            lab=lab,
            obj_area=obj_area,
            vol_vox=vol_vox,
        )

