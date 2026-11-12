from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from math import ceil, prod

from typing import Callable, Literal

import numpy as np
import pandas as pd

import skimage

from qtpy.QtCore import (
    Signal, Qt, QCoreApplication, QEventLoop, QTimer
)
from qtpy.QtWidgets import (
    QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton,
    QGraphicsProxyWidget, QGroupBox, QCheckBox, QMenu, QRadioButton, 
    QButtonGroup, QGridLayout, QSpacerItem
)

import pyqtgraph as pg

from .. import _window_stacking
from .. import printl
from .._run import _setup_app
from .. import apps, widgets
from .. import colors
from .. import plot
from .. import qutils

from . import _widgets, utils, _dialogs

import vispy.scene
import vispy.color

rng = np.random.default_rng(42)

AcdcColorMap = colors.AcdcColorMap
AcdcPyQtGraphColorMapName = colors.AcdcPyQtGraphColorMapName
PltColorName = colors.PltColorName
ColorMapPerChannel = (
    list[AcdcColorMap]
    | dict[str, AcdcColorMap]
)
LabelsGradientName = colors.AcdcPyQtGraphColorMapName

LutItemState = colors.LutItemState
LutItemStates = (
    list[LutItemState]
    | dict[str, LutItemState]
)

MarkerSymbols = plot.VisPyMarkerSymbols

_SLIDER_NORMALIZE_FACTOR = 20
_DEFAULT_LABELS_CMAP_NAME = 'viridis'

# Default (home) camera orientation. It is the "2D view" orientation
# (azimuth=180, elevation=90, roll=180, see `set_2d_view`), where left-right
# and top-bottom are the same as in the Cell-ACDC GUI, tilted by 45 degrees
# and turned by 20 degrees to give a 3D perspective without mirroring the
# image compared to the GUI.
_HOME_AZIMUTH = 160.0
_HOME_ELEVATION = 45.0
_HOME_ROLL = 180.0
_HOME_FOV = 60.0

# "2D view" camera orientation (looking perpendicular to the xy plane with
# left-right and top-bottom as in the Cell-ACDC GUI)
_TWO_D_VIEW_AZIMUTH = 180.0
_TWO_D_VIEW_ELEVATION = 90.0
_TWO_D_VIEW_ROLL = 180.0

# Mirrored cursor of the mouse cursor on the parent GUI image (same
# appearance as the mirrored cursor in the GUI)
_PARENT_CURSOR_SYMBOL = 'diamond'
_PARENT_CURSOR_SIZE = 16
_PARENT_CURSOR_FACE_RGBA = (1.0, 1.0, 1.0, 70/255)
_PARENT_CURSOR_EDGE_RGBA = (1.0, 1.0, 1.0, 1.0)

# Max. size of each dimension of the displayed volumes when the
# "Full resolution" mode is off (faster, volumes are stride-downsampled)
_PREVIEW_MAX_SIZE = 512

# OpenGL enum to query the max. size of 3D textures supported by the GPU
_GL_MAX_3D_TEXTURE_SIZE = 0x8073

# Max. 3D texture size assumed in "Full resolution" mode when it cannot
# be queried from the GPU (min. value required by OpenGL 4)
_FALLBACK_MAX_TEXTURE_3D = 2048

# Max. memory used by each cache of frames computed on demand
_FRAMES_CACHE_MAX_NBYTES = 512 * 1024**2

ParentViewPlane = Literal['xy', 'zy', 'zx']

# Plane displayed in the parent 2D GUI --> indices of the vispy (x, y, z)
# coordinates displayed (horizontal, vertical, depth) in the parent GUI.
_PARENT_VIEW_PLANE_AXES = {
    'xy': (0, 1, 2),
    'zy': (1, 2, 0),
    'zx': (0, 2, 1),
}

@dataclass
class _ChannelData:
    node: vispy.scene.visuals.Volume

    # (T, Z, Y, X) normalised (and possibly downsampled) data, frames
    # computed on demand. Never modified in place.
    volume: '_NormalisedVolume'

    auto_button: QPushButton
    reset_button: QPushButton
    lut_item: widgets.baseHistogramLUTitem

    opacity_slider: widgets.sliderWithSpinBox
    toolbutton: widgets.OverlayChannelToolButton

    # (T, Z, Y, X) blurred and dimmed `volume` used for the unselected
    # objects in "Focus on selected" display mode, frames computed on
    # demand. Created by `VolumeRendererWindow._get_off_focus_volume`
    _off_focus_volume: '_LazyFrames | None' = None

    # (Z, Y, X) scratch buffer displayed in the display modes
    # "Hide unselected" and "Focus on selected"
    _display_volume: np.ndarray | None = None

@dataclass
class _PointsLayer:
    name: str
    
    points_t_xyz: dict[int, np.ndarray] # {frame_i: (N, [x, y, z]) voxel coords}
    labels: list[str] | None

    markers: vispy.scene.visuals.Markers

    toolbutton: widgets.PointsLayerToolButton
    properties_dialog: apps.EditPointsLayerAppearanceDialog
    context_menu: widgets.PointsLayerContextMenu

    df: pd.DataFrame | None=None
    text: vispy.scene.visuals.Text | None=None

    # {frame_i: (N,) ID in VolumeRendererWindow._orig_lab of each point},
    # filled lazily by VolumeRendererWindow._get_points_label_ids
    label_ids_t: dict[int, np.ndarray] = field(default_factory=dict)

def _strides_to_slice(strides: tuple[int, ...]) -> tuple[slice, ...]:
    return tuple(slice(None, None, s) for s in strides)

class _LazyFrames:
    """Read-only (T, Z, Y, X) array-like whose (Z, Y, X) frames are
    computed on demand by `compute_frame(frame_i)` and cached.

    Only the most recently used frames are cached (up to
    `_FRAMES_CACHE_MAX_NBYTES`, at least 2 frames). Only integer indexing
    of the frames is supported.
    """
    def __init__(
            self,
            num_frames: int,
            frame_shape: tuple[int, ...],
            dtype,
            compute_frame: Callable[[int], np.ndarray],
        ):
        self.shape = (num_frames, *frame_shape)
        self.dtype = np.dtype(dtype)
        self._compute_frame = compute_frame
        frame_nbytes = max(1, prod(frame_shape) * self.dtype.itemsize)
        self._max_cached = max(2, _FRAMES_CACHE_MAX_NBYTES // frame_nbytes)
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, frame_i: int) -> np.ndarray:
        if not isinstance(frame_i, (int, np.integer)):
            raise TypeError(
                f'{type(self).__name__} supports only integer indexing of '
                f'the frames, got {frame_i!r}'
            )

        frame_i = int(frame_i)
        num_frames = len(self)
        if frame_i < 0:
            frame_i += num_frames

        if not 0 <= frame_i < num_frames:
            raise IndexError(
                f'Frame index {frame_i} out of range for {num_frames} frames'
            )

        frame = self._cache.get(frame_i)
        if frame is not None:
            self._cache.move_to_end(frame_i)
            return frame

        frame = self._compute_frame(frame_i)
        self._cache[frame_i] = frame
        while len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return frame

class _NormalisedVolume(_LazyFrames):
    """(T, Z, Y, X) float32 volume normalised to [0, 1] with the
    (full-resolution) value range [`vmin`, `vmax`] of `volume` and
    stride-downsampled with `strides`.

    Frames are computed on demand from `volume` (which is not copied and
    must not be modified).
    """
    def __init__(
            self,
            volume: np.ndarray,
            vmin: float,
            vmax: float,
            strides: tuple[int, int, int]
        ):
        self.input_volume = volume
        self.vmin = vmin
        self.vmax = vmax
        self.strides = tuple(strides)
        self._frame_slice = _strides_to_slice(self.strides)
        frame_shape = volume[0][self._frame_slice].shape
        super().__init__(
            len(volume), frame_shape, np.float32, self._normalise_frame
        )

    def _normalise(self, values: np.ndarray, out: np.ndarray):
        # In float32 directly into `out` to avoid temporary arrays
        np.subtract(values, self.vmin, out=out, dtype=np.float32)
        np.divide(out, self.vmax - self.vmin, out=out)

    def _normalise_frame(self, frame_i: int) -> np.ndarray:
        out = np.zeros(self.shape[1:], dtype=np.float32)
        if self.vmax > self.vmin:
            self._normalise(self.input_volume[frame_i][self._frame_slice], out)
        return out

    def with_strides(self, strides: tuple[int, int, int]):
        return _NormalisedVolume(
            self.input_volume, self.vmin, self.vmax, strides
        )

    def sample_values(self, max_num_samples=1_000_000) -> np.ndarray:
        """Return at most `max_num_samples` normalised values evenly
        sampled from all frames, i.e., the same values as
        `np.asarray(self).ravel()[::step]` without computing all frames.
        """
        size = prod(self.shape)
        step = 1
        if size > max_num_samples:
            step = size // max_num_samples + 1

        flat_idx = np.arange(0, size, step)
        values = np.zeros(len(flat_idx), dtype=np.float32)
        if not self.vmax > self.vmin:
            return values

        tt, zz, yy, xx = np.unravel_index(flat_idx, self.shape)
        sz, sy, sx = self.strides
        # Flat indices are sorted --> samples of each frame are contiguous
        frames, starts = np.unique(tt, return_index=True)
        stops = np.append(starts[1:], len(tt))
        for frame_i, start, stop in zip(frames, starts, stops):
            idx = slice(start, stop)
            frame_values = self.input_volume[frame_i][
                zz[idx]*sz, yy[idx]*sy, xx[idx]*sx
            ]
            self._normalise(frame_values, values[idx])
        return values

class _LabelsFrames(_LazyFrames):
    """(T, Z, Y, X) labels displayed, i.e., the original labels
    stride-downsampled with `strides`.

    Frames are copies of `source` computed on demand because they are
    modified in place (e.g., hidden objects are set to 0). `source` is a
    (strided) view of the original labels (that must not be modified).
    """
    def __init__(self, orig_lab: np.ndarray, strides: tuple[int, int, int]):
        self.source = orig_lab[(slice(None), *_strides_to_slice(strides))]
        super().__init__(
            len(self.source), self.source.shape[1:], self.source.dtype,
            self._copy_frame
        )

    def _copy_frame(self, frame_i: int) -> np.ndarray:
        return self.source[frame_i].copy()

    def value_range(self) -> tuple[float, float]:
        vmin, vmax = utils.frames_min_max(self.source)
        return float(vmin), float(vmax)

class _LazyFramesRegionprops(_LazyFrames):
    """`skimage.measure.regionprops` of each frame of a (T, Z, Y, X) labels
    array, computed on demand and cached.

    The labels array must not be modified in place.
    """
    def __init__(self, lab: np.ndarray):
        self._lab = lab
        # Cache budget estimated as one int64 label image per frame
        super().__init__(
            len(lab), lab.shape[1:], np.int64, self._compute_regionprops
        )

    def _compute_regionprops(self, frame_i: int) -> list:
        return skimage.measure.regionprops(self._lab[frame_i])

def _compute_off_focus_volume(raw_volume: np.ndarray) -> np.ndarray:
    """Blurred and dimmed version of `raw_volume` used for the unselected
    objects in "Focus on selected" display mode."""
    off_focus_volume = skimage.filters.gaussian(raw_volume, sigma=1.5)
    p_raw = np.percentile(raw_volume, 99.9)
    p_blur = np.percentile(off_focus_volume, 99.9)
    if p_blur > 0:
        off_focus_volume *= (p_raw / p_blur)
    off_focus_volume *= 0.8
    return off_focus_volume

@dataclass(slots=True)
class LabelAnnotation:
    label: int
    text: str
    xyz: np.ndarray
    screen_xy: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float32)
    )
    visible: bool = True


class VolumeRendererWindow(QMainWindow):
    """
    A standalone Qt window that displays a 3D z-stack volume using vispy.

    Usage (minimal)::

        renderer = VolumeRendererWindow()
        renderer.set_volume(zstack_array)   # (Z, Y, X) numpy array
        renderer.run()

    If ``hide_on_close=True``, the window hides (rather than closes) when the user
    presses X so the GPU state is preserved for re-display.  Set it to ``False``
    to destroy on close instead.
    """
    sigClose = Signal(object)
    sigUpdate = Signal()
    
    def __init__(
            self, 
            app=None, 
            parent=None, 
            parent_gui=None,
            version=None,
            title='Cell-ACDC - Volume Renderer',
            hide_on_close=False,
            logger_func=print
        ):
        """Initializer."""
        self._version = version
        self._parent_gui = parent_gui
        self._logger_func = logger_func
        self._ui_initialised = False
        self._is_labels_set = False
        self._hide_on_close = hide_on_close
        self._max_texture_3d = None
        self._lut_items_width = 20 # Start with some padding
        self._SizeZ = None
        self._gradient_item_state = None
        self._lab_gradient_cmap_name = None
        self._lab_gradient_item_state = None
        self._lab_node = None
        self._labels_text_overlay = None
        self._orig_lab = None
        self._canvas = None
        self._voxel_size = None
        self._voxel_size_transform = None
        self._lab_ncolors = 256
        self._voxel_size_strides_transform = None
        # Strides used to downsample the displayed (Z, Y, X) frames,
        # see `_get_display_strides`
        self._downsample_strides = None
        self._is_full_resolution = False
        self._object_labels_list_buttongroup = None
        self._data_shape = None
        self._frames_scrollbar_connected = False
        self._is_camera_view_set = False
        # Orientation (azimuth, elevation, roll) restored when syncing the
        # camera with the parent GUI, i.e., the last selected view
        # ("Home" or "2D view")
        self._synced_camera_orientation = (
            _HOME_AZIMUTH, _HOME_ELEVATION, _HOME_ROLL
        )
        self._parent_view_region = None

        if app is None:
            app = QCoreApplication.instance()
        
        if app is None:
            app, splashScreen = _setup_app(splashscreen=True)  
            splashScreen.close()
        
        self.app = app
        
        super().__init__(parent)
        self.setWindowTitle(title)
        
        self._channels_data: dict[str, _ChannelData] = {}
        self._points_layers: dict[str, _PointsLayer] = {}
        
        self._init_default_rgbs()
        self._init_vispy()
        self._init_ui()
    
    def _init_vispy(self) -> None:
        # Configure the backend to match the host Qt binding.
        import vispy
        from qtpy import API_NAME
        vispy.use(API_NAME)

        from vispy import scene 

        self._canvas = scene.SceneCanvas(
            keys='interactive', 
            bgcolor='black'
        )
        self._view = self._canvas.central_widget.add_view()
        # TurntableCamera: left-drag to orbit, scroll to zoom, right-drag to pan.
        # Keeps the "up" axis fixed, which suits microscopy z-stacks better than
        # a free arcball.
        self._view.camera = scene.cameras.TurntableCamera(
            fov=45,
            elevation=_HOME_ELEVATION,
            azimuth=_HOME_AZIMUTH,
            roll=_HOME_ROLL
        )

        self._canvas.events.mouse_press.connect(
            self._on_camera_mouse_press
        )
        self._canvas.events.mouse_move.connect(
            self._on_camera_mouse_move
        )
        self._canvas.events.mouse_release.connect(
            self._on_camera_mouse_release
        )
        self._canvas.events.mouse_wheel.connect(
            self._on_camera_mouse_wheel
        )

        self._camera_mouse_interaction = False

        # XYZ axis indicator at the front-bottom-left corner of the volume.
        # Red=X (data axis 2), Green=Y (data axis 1), Blue=Z (data axis 0).
        # Scale and visibility are updated in update_volume on first load.
        self._axis_visual = scene.visuals.XYZAxis(parent=self._view.scene)
        self._axis_visual.visible = False

        # Mirrored cursor of the mouse cursor on the parent GUI image.
        # Created at first use in `set_parent_cursor_pos` because an empty
        # Markers visual breaks `camera.set_range()` (no bounds)
        self._parent_cursor_marker = None

    def _on_camera_mouse_press(self, event):
        self._camera_mouse_interaction = True
        self._update_camera_indicators()
    
    def _on_camera_mouse_move(self, event):
        if not self._camera_mouse_interaction:
            return

        self._update_camera_indicators()
    
    def _on_camera_mouse_release(self, event):
        self._camera_mouse_interaction = False
        self._update_camera_indicators()
    
    def _on_camera_mouse_wheel(self, event):
        self._update_camera_indicators()

    def _update_camera_indicators(self):
        return
        # if not self._toolbar.setupAnimationAction.isChecked():
        #     return

        # self._setupAnimationDialog.set_from_camera(
        #     self._view.camera, 
        #     update_start=False
        # )

    def _init_lab_ui_items(
            self,
            lab_volume: _LabelsFrames,
            cmap_name=None,
            gradient_item_state=None,
            font_size=None,
            text_color='white'
        ):
        self._init_frames_scrollbar(len(lab_volume))

        from vispy.scene import visuals
        from .gl_blend import volume_gl_state

        # Kept when changing resolution so that the colors do not change
        vmin, vmax = lab_volume.value_range()
        
        self._lab_node = visuals.Volume(
            lab_volume[0],
            method="translucent",
            interpolation="nearest",
            parent=self._view.scene,
        )
        self._lab_node.clim = (vmin, vmax)

        self._lab_opacity_slider = widgets.sliderWithSpinBox(
            title_loc='in_line', 
            isFloat=True, 
            parent=self,
            normalize_factor=_SLIDER_NORMALIZE_FACTOR
        )
        
        self._lab_opacity_slider.setRange(0.0, 1.0)
        self._lab_opacity_slider.setSingleStep(0.05)
        self._lab_opacity_slider.setValue(0.3)
        self._lab_opacity_slider.setDecimals(2)
        
        self._lab_opacity_slider.setToolTip(f'Opacity of segmentation masks')
        opacity_slider_form_widget = widgets.formWidget(
            self._lab_opacity_slider,
            labelTextLeft='Segmentation masks opacity:',
        )
        self._controls_layout.addFormWidget(
            opacity_slider_form_widget, row=0
        )
        
        lab_reset_btn = QPushButton('Reset')
        self._lab_gradient_item_layout.addWidget(lab_reset_btn)
        
        lab_gradient_item = widgets.BaseLabelsGradientWidget(parent=self)
        self._lab_gradient_item_layout.addWidget(lab_gradient_item)
        self._lab_gradient_item = lab_gradient_item
        
        self._lab_opacity_slider.valueChanged.connect(
            self._on_lab_opacity_changed
        )
        lab_gradient_item.sigGradientChangeFinished.connect(
            self._on_lab_gradient_changed
        )
        lab_gradient_item.sigShuffleCmap.connect(
            self._random_shuffle_lab_gradient_cmap
        )
        lab_gradient_item.sigGreeedyShuffleCmap.connect(
            self._greedy_shuffle_lab_gradient_cmap
        )
        lab_gradient_item.shuffleCmapAction.setShortcut('Shift+S')
        lab_gradient_item.greedyShuffleCmapAction.setShortcut('Alt+Shift+S')
        # lab_lut_item.sigGradientChanged.connect(self.ticksCmapMoved)
        lab_reset_btn.clicked.connect(self._on_reset_lab_gradient)
        
        lab_reset_btn.setMaximumWidth(
            ceil(lab_gradient_item.sizeHint().width()))
        
        if gradient_item_state is not None:
            lab_gradient_item.item.restoreState(gradient_item_state)
            self._lab_gradient_item_state = gradient_item_state
        elif cmap_name is not None:
            self._lab_gradient_cmap_name = cmap_name
            lab_gradient_item.item.loadPreset(cmap_name)
        else:
            lab_gradient_item.item.loadPreset(_DEFAULT_LABELS_CMAP_NAME)
            
        self._lab_node.opacity = self._lab_opacity_slider.value()

        frame_i = self._frames_scrollbar.value() - 1
        self._object_labels_list_buttongroup = QButtonGroup(self)
        self._object_labels_list_buttongroup.setExclusive(False)
        self._obj_checkboxes_mapper = {}
        for obj in self._rp[frame_i]:
            obj_checkbox = QCheckBox(f'{obj.label}')
            obj_checkbox.setChecked(True)
            obj_checkbox.obj = obj
            self._object_labels_list_layout.addWidget(obj_checkbox)
            self._object_labels_list_buttongroup.addButton(obj_checkbox)
            obj_checkbox.toggled.connect(
                partial(self._set_object_checked, obj=obj)
            )
            obj_checkbox.setDisabled(True)
            self._obj_checkboxes_mapper[obj.label] = obj_checkbox
        
        self._object_labels_list_layout.addStretch(1)
        
        select_all_button = widgets.selectAllPushButton()
        select_all_button.sigClicked.connect(
            self._set_all_object_labels_list_checked
        )
        self._right_vertical_layout.addWidget(select_all_button)
        self._select_all_button = select_all_button
        self._select_all_button.setDisabled(True)
        
        self._right_vertical_layout.addSpacing(10)
        
        display_mode_groupbox = QGroupBox('Display mode')
        display_mode_layout = QVBoxLayout()
        display_mode_groupbox.setLayout(display_mode_layout)
        
        display_mode_buttonsgroup = QButtonGroup(self)
        
        self._display_mode_show_all_rb = QRadioButton(
            'Show all objects', self
        )
        self._display_mode_hide_unselected_rb = QRadioButton(
            'Hide unselected', self
        )
        self._display_mode_focus_selected_rb = QRadioButton(
            'Focus on selected', self
        )
        self._display_mode_show_all_rb.setChecked(True)
        display_mode_buttonsgroup.addButton(self._display_mode_show_all_rb)
        display_mode_buttonsgroup.addButton(
            self._display_mode_hide_unselected_rb)
        display_mode_buttonsgroup.addButton(
            self._display_mode_focus_selected_rb)
        
        display_mode_buttonsgroup.buttonToggled.connect(
            self._display_mode_radio_button_toggled
        )
        
        display_mode_layout.addWidget(self._display_mode_show_all_rb)
        display_mode_layout.addWidget(self._display_mode_hide_unselected_rb)
        display_mode_layout.addWidget(self._display_mode_focus_selected_rb)
        
        self._right_vertical_layout.addWidget(display_mode_groupbox)

        if font_size is None:
            font_size = _widgets.LABELS_TEXT_FONTSIZE

        frame_i = self._frames_scrollbar.value() - 1
        self._set_labels_text(self._rp[frame_i], font_size)
        self._labels_text_overlay.setTextColor(pg.mkColor(text_color))

        text_settings_groupbox = QGroupBox('Text settings')
        text_settings_layout = widgets.FormLayout()
        text_settings_groupbox.setLayout(text_settings_layout)

        row = 0
        show_labels_text_toggle = widgets.Toggle()
        show_labels_text_toggle.setChecked(True)
        show_labels_text_form_widget = widgets.formWidget(
            show_labels_text_toggle, 
            labelTextLeft='Show IDs',
            stretchWidget=False
        )
        text_settings_layout.addFormWidget(
            show_labels_text_form_widget, row=row
        )
        show_labels_text_toggle.toggled.connect(
            self._set_labels_text_visible
        )

        row += 1
        text_font_size_spinbox = widgets.SpinBox()
        text_font_size_spinbox.setMinimum(1)
        text_font_size_spinbox.setValue(font_size)
        text_font_size_form_widget = widgets.formWidget(
            text_font_size_spinbox, 
            labelTextLeft='Font size',
        )
        text_font_size_spinbox.valueChanged.connect(
            self._set_labels_text_font_size
        )
        self._text_font_size_spinbox = text_font_size_spinbox
        text_settings_layout.addFormWidget(text_font_size_form_widget, row=row)

        row += 1
        text_color_colorbutton = widgets.myColorButton(
            parent=self, color=text_color
        )
        text_color_form_widget = widgets.formWidget(
            text_color_colorbutton, 
            labelTextLeft='Color',
            stretchWidget=False
        )
        text_color_colorbutton.sigColorChanging.connect(
            self._set_labels_text_color
        )
        _window_stacking.set_window_flags_on_top(
            text_color_colorbutton.colorDialog, Qt.Window, parent=self
        )
        text_settings_layout.addFormWidget(text_color_form_widget, row=row)


        self._right_vertical_layout.addSpacing(10)
        self._right_vertical_layout.addWidget(text_settings_groupbox)

    def _set_labels_text_color(self, color_button):
        self._labels_text_overlay.setTextColor(color_button.color())

    def _set_labels_text_visible(self, checked: bool):
        self._labels_text_overlay.setAnnotationsVisible(checked)

    def _set_labels_text_font_size(self, font_size: int):
        self._labels_text_overlay.setFontSize(font_size)

    def _set_labels_text(self, rp, font_size: int):
        self._label_annotations = {}

        for obj in rp:
            z, y, x = obj.centroid

            visible = True
            obj_checkbox = self._obj_checkboxes_mapper.get(obj.label)
            if obj_checkbox is not None:
                visible = obj_checkbox.isChecked()
            self._label_annotations[obj.label] = LabelAnnotation(
                label=obj.label,
                text=str(obj.label),
                xyz=np.array([x, y, z], np.float32),
                screen_xy=np.zeros(2, np.float32),
                visible=visible,
            )

        if self._labels_text_overlay is None:
            self._labels_text_overlay = _widgets.LabelsOverlay(
                self, font_size=font_size
            )
            self._canvas.events.draw.connect(self._update_labels_overlay)
        else:
            self._labels_text_overlay.setFontSize(font_size)

    def _update_labels_overlay(self, event=None):
        if not self._label_annotations:
            return

        tr = self._lab_node.transforms.get_transform(
            map_from="visual",
            map_to="canvas",
        )

        pts = np.array(
            [ann.xyz for ann in self._label_annotations.values()],
            np.float32,
        )

        # if self._voxel_size_strides_transform is not None:
        #     pts = self._voxel_size_strides_transform.map(pts)[:, :3]

        canvas_pts = tr.map(pts)
        canvas_pts /= canvas_pts[:, 3:4]

        for ann, xy in zip(self._label_annotations.values(), canvas_pts[:, :2]):
            ann.screen_xy[:] = xy

        self._labels_text_overlay.update()
    
    def _display_mode_radio_button_toggled(
            self, button: QRadioButton, toggled: bool
        ):
        if not toggled:
            return
        
        is_show_all = self._display_mode_show_all_rb.isChecked()
        self._set_object_labels_checkboxes_disabled(disabled=is_show_all)
            
        self._update_display()
    
    def _set_all_object_labels_list_checked(self, select_all_button, checked):
        for checkbox in self._object_labels_list_buttongroup.buttons():
            checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(False)
            self._label_annotations[checkbox.obj.label].visible = checked

        self._update_lab_node(update=False)
        self._update_display(exclude_lab=True)
        self._labels_text_overlay.update()
    
    def _update_lab_node(self, update=True):
        frame_i = self._frames_scrollbar.value() - 1
        for checkbox in self._object_labels_list_buttongroup.buttons(): 
            obj = checkbox.obj
            checked = checkbox.isChecked()
            _id = obj.label if checked else 0
            self._lab[frame_i][obj.slice][obj.image] = _id
            
        self._lab_node.set_data(self._lab[frame_i])
        
        if update:
            self._canvas.update()
    
    def _update_markers(self, update=True):
        for points_layer in self._points_layers.values():
            state = points_layer.properties_dialog.state()
            self._update_points_layer_properties(
                state, points_layer=points_layer, update=update
            )
        
        if update:
            self._canvas.update()
    
    def _update_volume_nodes(self, update=True):
        frame_i = self._frames_scrollbar.value() - 1
        for channel_data in self._channels_data.values():
            raw_vol = channel_data.volume[frame_i]
            if self._display_mode_show_all_rb.isChecked():
                channel_data.node.set_data(raw_vol)
                continue

            displayed_volume = channel_data._display_volume
            if displayed_volume is None or displayed_volume.shape != raw_vol.shape:
                displayed_volume = np.empty_like(raw_vol)
                channel_data._display_volume = displayed_volume

            if self._display_mode_focus_selected_rb.isChecked():
                unselected_vals = self._get_off_focus_volume(
                    channel_data, frame_i
                )
                np.copyto(displayed_volume, unselected_vals)
            elif self._display_mode_hide_unselected_rb.isChecked():
                displayed_volume[:] = 0
            else:
                np.copyto(displayed_volume, raw_vol)

            for checkbox in self._object_labels_list_buttongroup.buttons():
                obj = checkbox.obj
                checked = checkbox.isChecked()
                if not checked:
                    continue

                src_intensities = raw_vol[obj.slice][obj.image]
                displayed_volume[obj.slice][obj.image] = src_intensities
            
            channel_data.node.set_data(displayed_volume)

        if update:
            self._canvas.update()
    
    def _update_display(self, exclude_lab=False, exclude_markers=False):
        if not exclude_lab:
            self._update_lab_node(update=False)
            frame_i = self._frames_scrollbar.value() - 1
            self._set_labels_text(
                self._rp[frame_i], 
                self._text_font_size_spinbox.value()
            )
            self._labels_text_overlay.update()
        
        if not exclude_markers:
            self._update_markers(update=False)

        self._update_volume_nodes(update=False)
        
        self._canvas.update()
        
    def _init_points_layer_ui_items(
            self,
            color: vispy.color.Color='red',
            size: float=8.0,
            opacity: float=1.0,
            symbol: MarkerSymbols='disc',
        ):
        rgb_color = vispy.color.Color(color).rgb
        rgb_color = [round(val*255) for val in rgb_color]
        
        toolbutton = widgets.PointsLayerToolButton(
            symbol, rgb_color, parent=self
        )
        toolbutton.setCheckable(True)
        toolbutton.setChecked(True)
        
        properties_dialog = apps.EditPointsLayerAppearanceDialog(
            backend='vispy', 
            is_3d=True, 
            add_opacity_slider=True,
            parent=toolbutton,
            hide_on_close=True
        )

        properties_dialog.restoreState({
            'symbol': symbol,
            'color': rgb_color,
            'pointSize': size,
            'opacity': opacity
        })
        properties_dialog.hide()
        
        context_menu = widgets.PointsLayerContextMenu(toolbutton)
        toolbutton._contextMenu = context_menu
        context_menu.hide()
        
        toolbutton.action = self._points_toolbar.addWidget(toolbutton)
        self._points_toolbar.show()
        
        return toolbutton, properties_dialog, context_menu
    
    def _random_shuffle_lab_gradient_cmap(self):
        from vispy.color import Colormap as VisPyColormap
        
        perm = rng.permutation(self._lab_ncolors)
        lut = self._lab_gradient_item.colorMap().getLookupTable(
            0.0, 1.0, self._lab_ncolors
        )
        shuffle_lut = lut[perm]
        
        shuffle_lut = np.array(shuffle_lut) / 255.0
        shuffle_lut = colors.replace_background_rgba_lut(shuffle_lut)
        
        cmap = VisPyColormap(shuffle_lut)
        
        self._lab_node.cmap = cmap
        self._canvas.update()

    def _greedy_shuffle_lab_gradient_cmap(self):
        from vispy.color import Colormap as VisPyColormap

        frame_i = self._frames_scrollbar.value() - 1
        lut = self._lab_gradient_item.colorMap().getLookupTable(
            0.0, 1.0, self._lab_ncolors
        )
        labels = [obj.label for obj in self._rp[frame_i]]
        greedy_lut = colors.get_greedy_lut(
            self._lab[frame_i], lut, ids=labels
        )
        
        greedy_lut = np.array(greedy_lut) / 255.0
        greedy_lut = colors.replace_background_rgba_lut(greedy_lut)
        
        cmap = VisPyColormap(greedy_lut)
        
        self._lab_node.cmap = cmap
        self._canvas.update()
    
    def _set_object_checked(self, checked: bool, obj=None):
        if obj is None:
            return

        frame_i = self._frames_scrollbar.value() - 1
        _id = obj.label if checked else 0
        self._lab[frame_i][obj.slice][obj.image] = _id

        self._label_annotations[obj.label].visible = checked
        self._lab_node.set_data(self._lab[frame_i])
        self._labels_text_overlay.update()

        self._update_display(exclude_lab=True)
    
    def _on_reset_lab_gradient(self, *args, **kwargs):        
        if self._lab_gradient_item_state is not None:
            self._lab_gradient_item.item.restoreState(
                self._lab_gradient_item_state)
            return
        
        if self._lab_gradient_cmap_name is not None:
            self._lab_gradient_item.item.loadPreset(
                self._lab_gradient_cmap_name)
            return
        
        self._lab_gradient_item.item.loadPreset(_DEFAULT_LABELS_CMAP_NAME)
    
    def _on_lab_gradient_changed(self, lab_gradient_item, update: bool=True):
        self._random_shuffle_lab_gradient_cmap()
    
    def _on_lab_opacity_changed(
            self, 
            value, 
            update: bool=True
        ):
        lab_gradient_item = self._lab_gradient_item
        
        if lab_gradient_item is None:
            return
        
        node = self._lab_node
        node.opacity = value
        if update:
            self._canvas.update()
    
    def _init_ui(self):
        if self._ui_initialised:
            return
        
        self._toolbar = _widgets.VolumeRendererToolbar(parent=self)
        self.addToolBar(Qt.TopToolBarArea, self._toolbar)
        
        self.addToolBarBreak(Qt.TopToolBarArea)
        self._points_toolbar = _widgets.PointsLayersToolbar(parent=self)
        self.addToolBar(Qt.TopToolBarArea, self._points_toolbar)
        self._points_toolbar.hide()
        
        self._toolbar.sigUpdate.connect(self.sigUpdate.emit)
        self._toolbar.sigHomeView.connect(self.reset_view)
        self._toolbar.sigTwoDView.connect(self.set_2d_view)
        self._toolbar.sigSave.connect(self.save_screenshot)
        self._toolbar.sigSetSingleChannel.connect(self._set_single_channel)
        self._toolbar.sigSetFullResolution.connect(self.set_full_resolution)
        # self._toolbar.sigSetupAnimation.connect(self.setup_animation_parameters)
        # self._toolbar.sigPlayAnimation.connect(self.play_animation)

        self._is_animation_setup = False

        # self._setupAnimationDialog = _dialogs.AnimationVolumeViewerSetupDialog()
        # self._setupAnimationDialog.sigOk.connect(self._set_animation_setup)
        # self._setupAnimationDialog.sigCancel.connect(
        #     self._cancel_animation_setup
        # )
        
        lut_items_graphics_layout = pg.GraphicsLayoutWidget()
        lut_items_graphics_layout.setBackground('black')
        self._lut_items_layout = lut_items_graphics_layout.addLayout(
            row=0, col=0
        )
        self._lut_items_graphics_layout = lut_items_graphics_layout

        self._lab_gradient_item_layout = QVBoxLayout()
        self._lab_gradient_item_layout.setContentsMargins(5, 5, 5, 5)
        
        self._object_labels_list_groupbox = QGroupBox('Object labels')
        self._object_labels_list_scrollarea = widgets.ScrollArea()
        self._object_labels_list_scrollarea.setWidget(
            self._object_labels_list_groupbox
        )
        self._object_labels_list_layout = QVBoxLayout()
        self._object_labels_list_groupbox.setLayout(
            self._object_labels_list_layout
        )
        self._set_object_labels_checkboxes_disabled()
        
        self._right_vertical_layout = QVBoxLayout()
        self._right_vertical_layout.addWidget(
            self._object_labels_list_scrollarea
        )
        self._right_vertical_layout.addSpacing(10)

        self._frames_scrollbar = widgets.ScrollBarWithNumericControl(
            labelText='Frame n.', parent=self
        )
        self._frames_scrollbar.setMinimum(1)
        self._frames_scrollbar.setValue(1)
        self._frames_scrollbar.setVisible(False)
        
        self._scene_layout = QGridLayout()

        row = 0
        self._scene_layout.addWidget(lut_items_graphics_layout, row, 0)
        self._scene_layout.addWidget(self._canvas.native, row, 1)
        self._scene_layout.addLayout(
            self._lab_gradient_item_layout, row, 2
        )
        self._scene_layout.addLayout(
            self._right_vertical_layout, row, 3
        )

        row += 1
        self._scene_layout.addItem(QSpacerItem(1, 5), row, 0)

        row += 1
        self._scene_layout.addWidget(self._frames_scrollbar, row, 0, 1, 2)

        if self._parent_gui is not None:
            self._sync_frames_scrollbar_main_gui_checkbox = QCheckBox(
                'Sync frames with parent GUI'
            )
            self._scene_layout.addWidget(
                self._sync_frames_scrollbar_main_gui_checkbox,
                row, 2, 1, 2,
                alignment=Qt.AlignLeft
            )
            self._sync_frames_scrollbar_main_gui_checkbox.setChecked(True)
            self._sync_frames_scrollbar_main_gui_checkbox.setVisible(False)
            self._sync_frames_scrollbar_main_gui_checkbox.toggled.connect(
                self._sync_frames_toggled
            )

            row += 1
            self._init_parent_view_ui_items(row)

        self._scene_layout.setColumnStretch(0, 0)
        self._scene_layout.setColumnStretch(1, 10)
        self._scene_layout.setColumnStretch(2, 0)
        self._scene_layout.setColumnStretch(3, 1)
        
        self._controls_groupbox = QGroupBox('Controls')
        self._controls_layout = widgets.FormLayout()
        self._controls_groupbox.setLayout(self._controls_layout)
        
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addLayout(self._scene_layout)
        main_layout.addSpacing(10)
        main_layout.addWidget(self._controls_groupbox)
        main_layout.setStretch(0, 10)
        main_layout.setStretch(1, 0)
        self.setCentralWidget(central)
        
        self._ui_initialised = True

    def _init_parent_view_ui_items(self, row: int):
        self._sync_view_main_gui_checkbox = QCheckBox(
            'Sync pan/zoom with parent GUI'
        )
        self._sync_view_main_gui_checkbox.setToolTip(
            'When checked, panning and zooming in the parent GUI moves the '
            'centre of rotation of the 3D camera to the centre of the region '
            'visible in the parent GUI (at the current slice) and zooms '
            'the camera to fit that region.\n\n'
            'Any manual rotation of the 3D view is reset to the last '
            'selected view ("Home" or "2D view").\n'
            'Tip: use the "2D view" button to look at the volume from the '
            'top, matching the parent GUI.'
        )
        self._sync_view_main_gui_checkbox.setChecked(True)
        self._sync_view_main_gui_checkbox.toggled.connect(
            self._sync_view_toggled
        )

        parent_view_layout = QHBoxLayout()
        parent_view_layout.setContentsMargins(5, 0, 5, 0)
        parent_view_layout.addWidget(self._sync_view_main_gui_checkbox)
        parent_view_layout.addStretch(1)

        self._scene_layout.addLayout(parent_view_layout, row, 0, 1, 4)

    def _sync_frames_toggled(self, checked):
        if checked:
            self._parent_gui.update3DViewer()

    def _sync_view_toggled(self, checked):
        if checked:
            self._sync_camera_to_parent_view_region()

    def _set_object_labels_checkboxes_disabled(self, disabled=True):
        if disabled:
            tooltip = (
                'Object selection is disabled when display mode '
                'is "Show all objects"\n\n'
                'Choose a different display mode below to activate selection.'
            )
        else:
            tooltip = None
            
        self._object_labels_list_groupbox.setToolTip(tooltip)
        if self._object_labels_list_buttongroup is None:
            return
        
        for checkbox in self._object_labels_list_buttongroup.buttons():
            checkbox.blockSignals(True)
            if disabled:
                checkbox.setChecked(True)
            checkbox.setDisabled(disabled)
            checkbox.blockSignals(False)

        self._select_all_button.setDisabled(disabled)
        
    def _block_exec(self):
        if hasattr(self, 'loop'):
            self.loop.exit()
            
        self.loop = QEventLoop()
        self.loop.exec_()
    
    def _resolve_max_texture_3d(self) -> int | None:
        """Return the GPU's maximum 3-D texture size queried from the
        OpenGL context of the canvas (cached after the first successful
        query), or None if it cannot be queried.
        """
        if self._max_texture_3d is not None:
            return self._max_texture_3d

        if self._canvas is None:
            return None

        try:
            from vispy.gloo import gl
            self._canvas.set_current()
            max_3d = int(gl.glGetParameter(_GL_MAX_3D_TEXTURE_SIZE))
        except Exception:
            return None

        if max_3d <= 0:
            return None

        self._max_texture_3d = max_3d
        return self._max_texture_3d

    def _get_display_max_size(self) -> int:
        """Max. size of each dimension of the displayed volumes."""
        max_texture_3d = self._resolve_max_texture_3d()
        if not self._is_full_resolution:
            if max_texture_3d is None:
                return _PREVIEW_MAX_SIZE
            return min(_PREVIEW_MAX_SIZE, max_texture_3d)

        if max_texture_3d is None:
            self._logger_func(
                '[WARNING]: Could not query the max. 3D texture size of the '
                f'GPU. Assuming {_FALLBACK_MAX_TEXTURE_3D}.'
            )
            return _FALLBACK_MAX_TEXTURE_3D

        return max_texture_3d

    def _get_display_strides(self) -> tuple[int, int, int]:
        """
        Return the (z, y, x) integer strides used to subsample the displayed
        (Z, Y, X) frames so no dimension exceeds the max. display size
        (see `_get_display_max_size`).

        Uses integer strides (fast, no interpolation). The strides are
        computed once (for the current resolution mode) so that all
        volumes are downsampled the same way.
        """
        if self._downsample_strides is None:
            max_size = self._get_display_max_size()
            self._downsample_strides = tuple(
                max(1, ceil(s / max_size)) for s in self._data_shape[-3:]
            )

        return self._downsample_strides

    def _preprocess_volume(self, volume: np.ndarray) -> _NormalisedVolume:
        if volume.ndim != 4:
            raise ValueError(
                f'Expected 4D (T, Z, Y, X) or 3D (Z, Y, X) array; got shape {volume.shape}'
            )

        self._SizeZ = len(volume[0])

        # Compute the value range on the full-resolution data BEFORE downsampling
        # so that stride-based subsampling cannot accidentally exclude extreme voxels
        # and cause incorrect normalisation (e.g. a single bright fluorescence spot
        # being dropped by the stride may lower the apparent maximum).
        # Computed on the input dtype (frame by frame in parallel threads)
        # to avoid a full-resolution float32 copy; casting to float32 gives
        # the same values as computing them on the float32 data (rounding
        # is monotonic).
        vmin, vmax = utils.frames_min_max(volume)
        vmin = float(np.float32(vmin))
        vmax = float(np.float32(vmax))

        # Frames are normalised with the full-resolution range and
        # downsampled to fit the display size only when displayed.
        # The strides are used by `_set_voxel_size_strides_transform` to
        # correct for non-uniform compression (e.g. stride_x=4 while
        # stride_z=1).
        return _NormalisedVolume(
            volume, vmin, vmax, self._get_display_strides()
        )

    def _get_off_focus_volume(
            self, channel_data: _ChannelData, frame_i: int
        ) -> np.ndarray:
        """Return the blurred and dimmed version of `channel_data.volume`
        at frame `frame_i`, used for the unselected objects in
        "Focus on selected" display mode.

        Computed on demand and cached (computing it for every frame at
        startup is slow with large timelapses).
        """
        if channel_data._off_focus_volume is None:
            volume = channel_data.volume
            channel_data._off_focus_volume = _LazyFrames(
                len(volume), volume.shape[1:], np.float32,
                lambda frame_i: _compute_off_focus_volume(volume[frame_i])
            )

        return channel_data._off_focus_volume[frame_i]

    def is_full_resolution(self) -> bool:
        return self._is_full_resolution

    def set_full_resolution(self, full_resolution: bool):
        """Display the volumes at full resolution (limited only by the max.
        3D texture size of the GPU) if True. If False, the volumes are
        stride-downsampled to max. `_PREVIEW_MAX_SIZE` voxels per dimension
        (faster, less GPU memory).
        """
        full_resolution = bool(full_resolution)
        checkbox = self._toolbar.fullResolutionCheckbox
        if checkbox.isChecked() != full_resolution:
            checkbox.blockSignals(True)
            checkbox.setChecked(full_resolution)
            checkbox.blockSignals(False)

        if full_resolution == self._is_full_resolution:
            return

        self._is_full_resolution = full_resolution
        prev_strides = self._downsample_strides
        self._downsample_strides = None
        if self._data_shape is None:
            # Applied when data is set
            return

        if self._get_display_strides() == prev_strides:
            # Data small enough to be already displayed at full resolution
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self._apply_display_strides()
        finally:
            QApplication.restoreOverrideCursor()

    def _apply_display_strides(self):
        """Re-create the displayed data with the current downsampling
        strides and refresh the display (current frame only, the other
        frames are computed when displayed)."""
        strides = self._get_display_strides()
        for channel_data in self._channels_data.values():
            channel_data.volume = channel_data.volume.with_strides(strides)
            channel_data._off_focus_volume = None
            channel_data._display_volume = None

        if self._orig_lab is not None:
            self._lab = _LabelsFrames(self._orig_lab, strides)
            self._rp = _LazyFramesRegionprops(self._lab.source)

        # Scale the nodes so that the scene coordinates do not change
        self._set_voxel_size_strides_transform(self._voxel_size)

        self._frame_number_changed(self._frames_scrollbar.value())

        # vispy caches the bounds of the nodes and does not update them
        # when the shape of the data changes (e.g., used in `set_2d_view`)
        nodes = [ch_data.node for ch_data in self._channels_data.values()]
        if self._lab_node is not None:
            nodes.append(self._lab_node)
        for node in nodes:
            bounds_changed = getattr(node, '_bounds_changed', None)
            if bounds_changed is not None:
                bounds_changed()
    
    def _init_default_rgbs(self):
        self._default_rgbs = colors.overlay_rgbs.copy()
        self._default_rgbs.insert(0, (255, 255, 255))
    
    def _set_channel_default_cmap(self, channel_name: str, channel_idx: int):        
        foregr_rgb = colors.FLUO_CHANNELS_COLORS.get(channel_name)
        if foregr_rgb is None:
            foregr_rgb = self._default_rgbs.pop(channel_idx)
        
        pg_gradient = colors.get_pg_gradient([(0,0,0,0), foregr_rgb])
        lut_item = self._channels_data[channel_name].lut_item
        lut_item.setGradient(pg_gradient)
        pg_cmap = lut_item.gradient.colorMap()
        cmap = colors.pg_to_vispy_cmap(pg_cmap)
        return cmap
    
    def _add_opacity_slider(self, channel_name: str, row: int):
        opacity_slider = widgets.sliderWithSpinBox(
            title_loc='in_line', 
            isFloat=True, 
            parent=self,
            normalize_factor=_SLIDER_NORMALIZE_FACTOR
        )
        opacity_slider.setRange(0.0, 1.0)
        opacity_slider.setSingleStep(0.05)
        opacity_slider.setValue(1.0)
        opacity_slider.setDecimals(2)
        opacity_slider.setToolTip(f'Opacity of channel {channel_name}')
        opacity_slider_form_widget = widgets.formWidget(
            opacity_slider,
            labelTextLeft=f'{channel_name} opacity:',
        )
        self._controls_layout.addFormWidget(
            opacity_slider_form_widget, row=row
        )
        
        return opacity_slider
    
    def _set_channels_data(
            self, 
            volumes: list[np.ndarray],
            channel_names: list[str],
        ):
        from vispy.scene import visuals
        
        for channel, volume in zip(channel_names, volumes):
            c_idx = len(self._channels_data)
            
            auto_btn = QPushButton('Auto')
            auto_btn_proxy = QGraphicsProxyWidget()
            auto_btn_proxy.setWidget(auto_btn)
            self._lut_items_layout.addItem(auto_btn_proxy, row=0, col=c_idx)

            reset_btn = QPushButton('Reset')
            reset_btn_proxy = QGraphicsProxyWidget()
            reset_btn_proxy.setWidget(reset_btn)
            self._lut_items_layout.addItem(reset_btn_proxy, row=1, col=c_idx)
            
            node = visuals.Volume(
                volume[0],
                method="mip",
                parent=self._view.scene,
            )
            
            lut_item = widgets.baseHistogramLUTitem()

            opacity_slider = self._add_opacity_slider(channel, row=c_idx+1)
            
            toolbutton = widgets.OverlayChannelToolButton(
                channel, lut_item, shortcut=str(c_idx)
            )
            toolbutton.action = self._toolbar.addWidget(toolbutton)
            toolbutton.setChecked(True)

            channel_data = _ChannelData(
                node=node,
                lut_item=lut_item,
                volume=volume,
                auto_button=auto_btn,
                reset_button=reset_btn,
                opacity_slider=opacity_slider,
                toolbutton=toolbutton
            )
            
            auto_btn.clicked.connect(
                partial(self._on_auto_clim, channel_data=channel_data)
            )
            reset_btn.clicked.connect(
                partial(self._on_reset_clim, channel_data=channel_data)
            )
            
            self._lut_items_layout.addItem(lut_item, row=2, col=c_idx)
            
            lut_item.sigLookupTableChanged.connect(
                partial(self._on_lut_changed, channel_data=channel_data)
            )
            opacity_slider.valueChanged.connect(
                partial(self._on_opacity_changed, channel_data=channel_data)
            )
            toolbutton.clicked.connect(
                partial(
                    self._on_channel_toolbutton_clicked, 
                    channel_data=channel_data
                )
            )
            self._channels_data[channel] = channel_data

            lut_item_width = lut_item.sizeHint(Qt.PreferredSize).width()
            self._lut_items_width += lut_item_width
            
            auto_btn.setMaximumWidth(ceil(lut_item_width))
            reset_btn.setMaximumWidth(ceil(lut_item_width))
        
        self._lut_items_graphics_layout.setFixedWidth(int(self._lut_items_width))  
    
    def number_of_visible_channels(self) -> int:
        return sum(
            1 for ch_data in self._channels_data.values() 
            if ch_data.node.visible
        )
    
    def _on_channel_toolbutton_clicked(
            self, 
            checked, 
            channel_data: _ChannelData,
            update: bool=True
        ):
        node = channel_data.node
        node.visible = checked
        if update:
            self._canvas.update()
        
        if not self._toolbar.is_single_channel_mode():
            return
        
        if checked:
            for other_ch, other_ch_data in self._channels_data.items():
                if other_ch_data is channel_data:
                    continue
                
                if other_ch_data.node.visible:
                    other_ch_data.toolbutton.setChecked(False)
            
            self._set_visiblity()
    
    def _set_visiblity(self, update=False):
        for channel_data in self._channels_data.values():
            channel_data.node.visible = channel_data.toolbutton.isChecked()
        
        self._set_gl_blend_states()
        
        if update:
            self._canvas.update()
    
    def _on_opacity_changed(
            self,
            value, 
            channel_data: _ChannelData=None,
            channel_name: str=None,
            update: bool=True
        ) -> None:
        if channel_data is None:
            channel_data = self._channels_data[channel_name]
        
        node = channel_data.node
        node.opacity = value
        if update:
            self._canvas.update()
    
    def _on_lut_changed(
            self, 
            lut_item, 
            channel_data: _ChannelData=None,
            channel_name: str=None,
            update: bool=True
        ) -> None:
        if channel_data is None:
            channel_data = self._channels_data[channel_name]
            
        cmap = colors.pg_to_vispy_cmap(lut_item.gradient.colorMap())
        node = channel_data.node
        node.cmap = cmap
        if update:
            self._canvas.update()
    
    def _on_auto_clim(self, channel_data: _ChannelData) -> None:
        lut_item = channel_data.lut_item
        
        if len(lut_item.gradient.listTicks()) != 2:
            self._logger_func(
                '[WARNING]: Auto contrast is available only with LUTs '
                'that have two ticks.'
            )
            return
        
        # Same samples of all frames used by `get_auto_contrast_percentile`
        # but without computing all the frames. The ticks are positions in
        # the colormap, which vispy maps to the colour limits of the node
        values = channel_data.volume.sample_values()
        vmin_raw, vmax_raw = channel_data.node.clim
        lo, hi = colors.get_auto_contrast_percentile(
            values, vmin_raw=vmin_raw, vmax_raw=vmax_raw
        )
        (low_tick, _), (high_tick, _) = lut_item.gradient.listTicks()            
        lut_item.gradient.setTickValue(high_tick, hi)
        lut_item.gradient.setTickValue(low_tick, lo)
    
    def _on_reset_clim(self, channel_data: _ChannelData) -> None:
        lut_item = channel_data.lut_item
        lut_item.resetState()
        
    def _set_gl_blend_states(self):
        from .gl_blend import volume_gl_state

        first_visible = False
        first_visible_found = False
        for c, (channel, channel_data) in enumerate(self._channels_data.items()):
            blending = "translucent_no_depth" if c == 0 else "additive"
            node = channel_data.node
            node.order = c
            node.opacity = channel_data.opacity_slider.value()
            if not first_visible_found:
                first_visible = channel_data.toolbutton.isChecked()
                first_visible_found = True
            else:
                first_visible = False
            gl_stage = volume_gl_state(blending, first_visible=first_visible)
            node.set_gl_state(**gl_stage)
    
    def _set_animation_setup(self, params: dict[str, float]):
        self._is_animation_setup = True
        self._toolbar.setupAnimationAction.setChecked(False)
        self._toolbar.playAnimationAction.setDisabled(False)
        self._animation_params = params
        blinker = qutils.QControlBlink(
            self._toolbar.playAnimationAction, qparent=self
        )
        blinker.start()
    
    def _cancel_animation_setup(self):
        self._toolbar.setupAnimationAction.setChecked(False)

    def _set_single_channel(self, single: bool):
        if single:
            for ch_data in self._channels_data.values():
                ch_data.toolbutton.setChecked(False)
                    
            self._last_visible_channels_data = [
                ch_data for ch_data in self._channels_data.values() 
                if ch_data.node.visible
            ]
            if len(self._last_visible_channels_data) == 0:
                self._last_visible_channels_data = [
                    list(self._channels_data.values())[0]
                ]
            
            first_visible_channel_data = self._last_visible_channels_data[0]
            first_visible_channel_data.toolbutton.setChecked(True)
        else:
            for ch_data in self._last_visible_channels_data:
                if not ch_data.node.visible:
                    ch_data.toolbutton.setChecked(True)

        self._set_visiblity(update=True)
    
    def _update_points_layer_properties(
            self, points_state, points_layer=None, update=True
        ):
        if points_layer is None:
            return
        
        symbol = points_state['symbol']
        color = [val/255 for val in points_state['color']]
        size = points_state['pointSize']
        opacity = points_state['opacity']
    
        face_color = vispy.color.Color(color, alpha=opacity*0.5)
        edge_color = vispy.color.Color(color, alpha=opacity)

        markers = points_layer.markers
        frame_i = self._frames_scrollbar.value() - 1
        points_xyz = points_layer.points_t_xyz.get(frame_i)
        if points_xyz is None:
            markers.visible = False
            return

        markers.visible = points_layer.toolbutton.isChecked()
        label_ids = self._get_points_label_ids(
            points_layer, frame_i, points_xyz
        )
        if label_ids is not None:
            # Hide only the points belonging to hidden objects. Points
            # outside of any object (ID 0) are always visible
            hidden_label_ids = [
                int(cb.text())
                for cb in self._object_labels_list_buttongroup.buttons()
                if not cb.isChecked()
            ]
            visible = ~np.isin(label_ids, hidden_label_ids)
            points_xyz = points_xyz[visible]
        
        markers.set_data(
            points_xyz, 
            symbol=symbol,
            size=size,
            face_color=face_color,
            edge_color=edge_color
        )
        
        if update:
            self._canvas.update()
    
    def _get_points_label_ids(
            self,
            points_layer: _PointsLayer,
            frame_i: int,
            points_xyz: np.ndarray
        ) -> np.ndarray | None:
        """Return the ID of the object each point of `points_xyz` (points
        of `points_layer` at frame `frame_i`) falls into.

        Each point is mapped to the voxel it is rendered in (rounded
        coordinates, since voxel centers are at integer coordinates).
        Points outside of any object, outside of the volume, or with
        non-finite coordinates get ID 0.

        Returns None if no labels are available for `frame_i`. Results are
        cached per frame in `points_layer.label_ids_t`.
        """
        if self._orig_lab is None or frame_i >= len(self._orig_lab):
            return None

        label_ids = points_layer.label_ids_t.get(frame_i)
        if label_ids is not None:
            return label_ids

        lab = self._orig_lab[frame_i]
        zyx = np.asarray(points_xyz, dtype=float).reshape(-1, 3)[:, ::-1]
        valid = np.all(np.isfinite(zyx), axis=1)
        zyx = np.round(np.where(valid[:, np.newaxis], zyx, -1)).astype(int)
        valid &= np.all((zyx >= 0) & (zyx < lab.shape), axis=1)

        label_ids = np.zeros(len(zyx), dtype=lab.dtype)
        zz, yy, xx = zyx[valid].T
        label_ids[valid] = lab[zz, yy, xx]

        points_layer.label_ids_t[frame_i] = label_ids
        return label_ids

    def _toggle_points_layer(self, checked, points_layer=None):
        if points_layer is None:
            return
        
        markers = points_layer.markers
        markers.visible = checked
        
        self._canvas.update()
    
    def _edit_points_layer_properties(self, points_layer=None):
        if points_layer is None:
            return
        
        properties_dialog = points_layer.properties_dialog
        properties_dialog.exec_()
    
    def _remove_points_layer(self, points_layer=None):
        if points_layer is None:
            return
        
        name = points_layer.name
        points_layer.markers.parent = None
        if points_layer.text is not None:
            points_layer.text = None
        
        points_layer.properties_dialog.force_close()
        
        points_layer.toolbutton.setChecked(False)
        self._points_toolbar.removeAction(points_layer.toolbutton.action)
        
        del self._points_layers[name]
    
    def set_labels(
            self, 
            lab: np.ndarray, 
            voxel_size: tuple[float, float, float]=None,
            cmap_name: AcdcPyQtGraphColorMapName=None,
            gradient_item_state: dict=None,
            lut: np.ndarray=None,
            font_size: int=None,
            text_color='white',
            SizeZ: int=None,
            SizeT: int=None,
            copy: bool=True,
        ):
        """
        Parameters
        ----------
        lut:  np.ndarray of {``ColorMap.BYTE``}, default=None
            The lookup table for the labels volume
        copy: bool, default=True
            If True, the viewer keeps its own copy of `lab`. If False, the
            viewer keeps a reference to `lab` (faster and less memory with
            large timelapses), which must then NOT be modified after
            calling this method.

        Notes
        -----
        If the labels have more timepoints than the volumes already
        displayed (e.g., the segmentation file has more frames than the
        loaded channel), the extra timepoints at the end are cropped away.
        The z-slices (and y, x) must still match.
        """
        input_lab = lab
        if self._is_labels_set:
            self._logger_func(
                '[WARNING]: Labels already set. '
                'Only one labels volumes can be set'
            )
            return
        
        if SizeZ is None:
            SizeZ = self._SizeZ
        
        if SizeT is None and not hasattr(self, '_SizeT'):
            raise ValueError(
                'SizeT information not present. Either call `set_volumes` '
                'before or pass `SizeT` to `set_labels`.'
            )
        
        if SizeT is None:
            SizeT = self._SizeT
        
        if lab.ndim == 2 and SizeZ is None:
            raise ValueError(
                'Labels array is 2D but SizeZ not set.'
            )

        if lab.ndim == 2 and SizeT > 1:
            raise ValueError(
                '2D labels array is not supported with timelapse data.'
            )
        
        if lab.ndim == 2:
            # Stack the static mask along z and add a time axis.
            lab = np.asarray([lab]*SizeZ)[np.newaxis]
        elif lab.ndim == 3:
            if SizeT == 1 and (SizeZ is None or lab.shape[0] == SizeZ):
                lab = lab[np.newaxis]
            else:
                # Stack 2D+time labels along z.
                lab = np.asarray([np.asarray([lab_i]*SizeZ) for lab_i in lab])

        if self._data_shape is None:
            self._data_shape = lab.shape

        lab = self._crop_lab_extra_frames(lab)

        if lab.shape != self._data_shape:
            raise ValueError(
                f'Labels array shape {lab.shape} does not match the current '
                f'viewer shape {self._data_shape}. '
                'All displayed volumes must have the same shape.'
            )
        
        if lab.dtype == bool:
            lab = lab.astype(np.uint8)
        
        if lab.ndim != 4:
            raise ValueError(
                f'Expected 4D (T, Z, Y, X) or 3D (Z, Y, X) labels array; got shape {lab.shape}')
        
        # The original labels are only read (e.g., to get the ID of the
        # object under each point) and never modified by the viewer. Copy
        # them only if requested and if they are (a view of) the caller's
        # array (stacking 2D/3D labels and casting bool already copied)
        if copy and np.may_share_memory(lab, input_lab):
            lab = lab.copy()
        self._orig_lab = lab

        # Displayed (possibly downsampled) labels, frames computed on demand
        self._lab = _LabelsFrames(lab, self._get_display_strides())

        # Regionprops of the displayed labels are computed lazily per
        # frame. They are computed from a (strided) view of the original
        # labels because the frames of `self._lab` are modified in place
        # (e.g., hidden objects are set to 0)
        self._rp = _LazyFramesRegionprops(self._lab.source)

        self._init_lab_ui_items(
            self._lab, 
            cmap_name=cmap_name, 
            gradient_item_state=gradient_item_state,
            font_size=font_size,
            text_color=text_color
        )

        if lut is not None:
            self._set_labels_lut(lut)

        if self._voxel_size_strides_transform is not None:
            self._lab_node.transform = self._voxel_size_strides_transform
        else:
            self._set_voxel_size_strides_transform(voxel_size)

        self._is_labels_set = True

    def _crop_lab_extra_frames(self, lab: np.ndarray) -> np.ndarray:
        """Crop the timepoints of a 4D (T, Z, Y, X) labels array that exceed
        the number of frames of the displayed volumes.

        Only the time axis is cropped. If the other dimensions do not match
        the viewer shape, `lab` is returned unchanged and the mismatch is
        reported by the shape check in `set_labels`.
        """
        if self._data_shape is None or lab.ndim != 4:
            return lab

        num_frames = self._data_shape[0]
        lab_num_frames = lab.shape[0]
        if lab_num_frames <= num_frames:
            return lab

        if lab.shape[1:] != tuple(self._data_shape[1:]):
            return lab

        self._logger_func(
            f'[WARNING]: Labels array has {lab_num_frames} frames while the '
            f'displayed volumes have {num_frames} frames. The last '
            f'{lab_num_frames - num_frames} frames of the labels will not '
            'be displayed in the 3D viewer.'
        )
        return lab[:num_frames]

    def _init_frames_scrollbar(self, num_frames):
        self._frames_scrollbar.setVisible(num_frames > 1)
        self._frames_scrollbar.setMaximum(num_frames)
        if self._parent_gui is not None:
            self._sync_frames_scrollbar_main_gui_checkbox.setVisible(
                num_frames > 1
            )
        if self._frames_scrollbar_connected:
            return

        self._frames_scrollbar.sigValueChanged.connect(
            self._frame_number_changed
        )
        self._frames_scrollbar_connected = True

    def _frame_number_changed(self, frame_n: int):
        if self._object_labels_list_buttongroup is None:
            for channel_data in self._channels_data.values():
                channel_data.node.set_data(channel_data.volume[frame_n - 1])
            self._update_markers(update=False)
            self._canvas.update()
            return

        for checkbox in self._object_labels_list_buttongroup.buttons():
            self._object_labels_list_buttongroup.removeButton(checkbox)
            self._object_labels_list_layout.removeWidget(checkbox)
            checkbox.hide()

        frame_i = frame_n - 1
        is_show_all = self._display_mode_show_all_rb.isChecked()
        for obj in self._rp[frame_i]:
            obj_checkbox = self._obj_checkboxes_mapper.get(obj.label)
            if obj_checkbox is None:
                obj_checkbox = QCheckBox(f'{obj.label}')
                obj_checkbox.setChecked(True)
                self._obj_checkboxes_mapper[obj.label] = obj_checkbox
            else:
                obj_checkbox.toggled.disconnect()
            obj_checkbox.obj = obj
            obj_checkbox.setDisabled(is_show_all)
            if is_show_all:
                obj_checkbox.setChecked(True)
            self._object_labels_list_layout.addWidget(obj_checkbox)
            self._object_labels_list_buttongroup.addButton(obj_checkbox)
            obj_checkbox.show()
            obj_checkbox.toggled.connect(
                partial(self._set_object_checked, obj=obj)
            )

        self._update_display()

    @property
    def num_frames(self) -> int | None:
        """Number of frames of the displayed volumes (None if no volumes
        have been set yet)."""
        return getattr(self, '_SizeT', None)

    def set_frame_number(self, frame_n: int):
        self._frames_scrollbar.setValue(frame_n)

    def _set_labels_lut(self, lut: np.ndarray, update=True):
        from vispy.color import Colormap as VisPyColormap

        lut_float = np.array(lut) / 255.0
        lut_float = colors.replace_background_rgba_lut(lut_float)
        cmap = VisPyColormap(lut_float)

        self._lab_node.cmap = cmap
        if update:
            self._canvas.update()
    
    def set_volume(
            self,
            volume: np.ndarray, 
            channel_name: str='',
            voxel_size: tuple[float, float, float]=None,
            lut_item_state: LutItemState | None=None,
            cmap: list[colors.RgbaColor] | AcdcPyQtGraphColorMapName=None,
            copy: bool=True,
        ):
        """See `set_volumes` for the description of the parameters."""
        num_channels = len(self._channels_data)
        if not channel_name:
            channel_name = f'channel_{num_channels+1}'

        cmaps = None
        lut_items_states = None
        if lut_item_state is not None:
            lut_items_states = {channel_name: lut_item_state}
        elif cmap is not None:
            cmaps = {channel_name: cmap}
            
        volumes = {channel_name: volume}
        
        self.set_volumes(
            volumes,
            cmaps=cmaps,
            lut_items_states=lut_items_states,
            voxel_size=voxel_size,
            copy=copy,
        )
    
    def _set_voxel_size_strides_transform(
            self, 
            voxel_size: tuple[float, float, float] | None
        ):
        if voxel_size is None:
            voxel_size = self._voxel_size
        
        if voxel_size is None:
            voxel_size = (1.0, 1.0, 1.0)
        
        if self._downsample_strides is None:
            self._downsample_strides = (1, 1, 1)
        
        self._voxel_size = voxel_size
        
        from vispy.visuals.transforms import STTransform

        sx, sy, sz = utils.voxel_display_scale(*voxel_size)
        scale = (
            sx * self._downsample_strides[2],
            sy * self._downsample_strides[1],
            sz * self._downsample_strides[0],
        )
        transform = STTransform(scale=scale)
        self._voxel_size_strides_transform = transform
        self._voxel_size_transform = STTransform(scale=(sx, sy, sz))
        
        for channel_data in self._channels_data.values():
            channel_data.node.transform = transform
            
        if self._lab_node is not None:
            self._lab_node.transform = transform
        
        for points_layer in self._points_layers.values():
            points_layer.markers.transform = self._voxel_size_transform

        if self._canvas is not None:
            self._canvas.update()
    
    def set_volumes(
            self,
            volumes: list[np.ndarray] | dict[str, np.ndarray], 
            channel_names: list[str] | None=None,
            voxel_size: tuple[float, float, float]=None,
            lut_items_states: LutItemStates | None=None,
            cmaps: ColorMapPerChannel=None,
            copy: bool=True,
        ):
        """
        Parameters
        ----------
        copy : bool, default=True
            If True, the viewer keeps its own copy of the volumes. If False,
            the viewer keeps a reference to the volumes (faster and less
            memory with large timelapses), which must then NOT be modified
            while the viewer is open (the displayed frames are computed
            from them on demand).
        """
        if isinstance(volumes, dict):
            channel_names = list(volumes.keys())
            volumes = list(volumes.values())
        else:
            volumes = list(volumes)

        for v, volume in enumerate(volumes):
            if volume.ndim == 3:
                # Add time axis
                volume = volume[np.newaxis]
            volumes[v] = volume

        if self._data_shape is None:
            self._data_shape = volumes[0].shape

        for volume in volumes:
            if volume.shape != self._data_shape:
                raise ValueError(
                    f'Volume shape {volume.shape} does not match the current '
                    f'viewer shape {self._data_shape}. '
                    'All displayed volumes must have the same shape.'
                )

        if copy:
            volumes = [volume.copy() for volume in volumes]

        volumes = [self._preprocess_volume(vol) for vol in volumes]
        
        num_existing_channels = len(self._channels_data)
        tot_num_channels = num_existing_channels + len(volumes)
        
        if channel_names is None:
            channel_names = [
                f'channel_{c+1}' 
                for c in range(num_existing_channels, tot_num_channels)
            ]
        
        self._set_channels_data(
            volumes, 
            channel_names, 
        )
        
        if lut_items_states is not None:
            for c, (ch, channel_data) in enumerate(self._channels_data.items()):
                if isinstance(lut_items_states, list):
                    state = lut_items_states[c]
                else:
                    state = lut_items_states.get(ch)
                channel_data.lut_item.restoreState(state)
        elif cmaps is not None:
            for c, (ch, channel_data) in enumerate(self._channels_data.items()):
                if isinstance(cmaps, list):
                    cmap = cmaps[c]
                else:
                    cmap = cmaps.get(ch)
                
                channel_data.lut_item.setColormap(cmap)
                for tick in channel_data.lut_item.gradient.listTicks():
                    tick[0].hide()
        else:
            cmaps = {
                ch: self._set_channel_default_cmap(ch, c) 
                for c, ch in enumerate(channel_names)
            }
        
        self._set_gl_blend_states()
        
        if self._voxel_size_strides_transform is not None:
            for channel_data in self._channels_data.values():
                channel_data.node.transform = self._voxel_size_strides_transform
        else:
            self._set_voxel_size_strides_transform(voxel_size)

        num_frames = len(volumes[0])
        self._SizeT = num_frames
        self._init_frames_scrollbar(num_frames)
        
        self._canvas.update()
    
    def add_points_layer(
            self,
            name: str,
            points: np.ndarray | None=None, # (N, [z, y, x]) voxel coordinates or (N, [t, z, y, x])
            points_df: pd.DataFrame | None=None,
            tzyx_columns_names: list[str] | None=None,
            labels: list[str] | None=None,
            color: vispy.color.Color='red',
            size: float=8.0,
            opacity: float=1.0,
            symbol: MarkerSymbols='disc',
            visible: bool=True,
            scaling: Literal['fixed', 'scene']='scene'
        ):
        if name in self._points_layers.keys():
            raise NameError(
                f'Points layer with name "{name}" already existing. '
                'Choose a different name'
            )
            
        from vispy.scene import visuals
        
        if tzyx_columns_names is None:
            tzyx_columns_names = ['frame_i', 'z', 'y', 'x']

        t_col_name = tzyx_columns_names[0]
        if points_df is not None:
            points_t_xyz = {}
            if t_col_name not in points_df.columns:
                points_df[t_col_name] = 0
            for t, points_df_t in points_df.groupby(t_col_name):
                zyx_columns_names = tzyx_columns_names[1:]
                points_xyz = points_df_t[zyx_columns_names[::-1]].to_numpy()
                points_t_xyz[t] = points_xyz
        elif points is not None:
            points_t_xyz = {}
            points = np.asarray(points)
            if points.shape[1] == 3:
                points_t_xyz[0] = points[:, [2, 1, 0]]
            else:
                for t in np.unique(points[:, 0]):
                    points_t_xyz[t] = points[points[:, 0] == t][:, [3, 2, 1]]
        else:
            raise ValueError(
                "Either 'points' or 'points_df' must be provided."
            ) 

        self._points_t_xyz = points_t_xyz
        face_color = vispy.color.Color(color, alpha=opacity*0.5)
        edge_color = vispy.color.Color(color, alpha=opacity)
        markers = visuals.Markers(
            parent=self._view.scene,
            spherical=True,
            scaling=scaling
        )

        frame_i = self._frames_scrollbar.value() - 1
        points_xyz = self._points_t_xyz.get(frame_i)
        if points_xyz is not None:
            markers.set_data(
                points_xyz, 
                symbol=symbol,
                size=size,
                face_color=face_color,
                edge_color=edge_color
            )
        
        markers.set_gl_state(depth_test=False)
        if self._voxel_size_transform is not None:
            markers.transform = self._voxel_size_transform
        
        ui_items = self._init_points_layer_ui_items(
            color=edge_color,
            size=size,
            opacity=opacity,
            symbol=symbol,
        )
        
        toolbutton, properties_dialog, context_menu = ui_items
        
        points_layer = _PointsLayer(
            name=name,
            points_t_xyz=points_t_xyz,
            labels=labels,
            markers=markers,
            toolbutton=toolbutton,
            properties_dialog=properties_dialog,
            context_menu=context_menu,
            df=points_df
        )
        
        properties_dialog.sigValueChanged.connect(
            partial(
                self._update_points_layer_properties, points_layer=points_layer
            )
        )
        
        context_menu.sigEditProperties.connect(
            partial(
                self._edit_points_layer_properties, points_layer=points_layer
            )
        )
        
        context_menu.sigRemove.connect(
            partial(
                self._remove_points_layer, points_layer=points_layer
            )
        )
        
        toolbutton.toggled.connect(
            partial(
                self._toggle_points_layer, points_layer=points_layer
            )
        )
        if not visible:
            toolbutton.setChecked(False)
        
        self._points_layers[name] = points_layer
    
    def show(self, block=False):
        self.resize(960, 720)
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        super().show()

        self.set_camera_view()

        self._update_parent_view_region()
        # The camera zoom fitting the parent view region depends on the
        # canvas aspect ratio --> re-apply once the layout has settled
        QTimer.singleShot(0, self._update_parent_view_region)

        try:
            self.setEnabled(True)
        except Exception as err:
            pass

        if block:
            self._block_exec()
    
    def run(self, block=True):        
        self.show()
        if block:
            self._block_exec()
    
    def force_close(self):
        """Force the window to close (bypassing hide_on_close)."""
        self._hide_on_close = False
        self.close()
    
    def closeEvent(self, event):
        self.sigClose.emit(self)
        
        if self._hide_on_close:
            event.ignore()
            self.hide()
            return
        
        for points_layer in self._points_layers.values():
            points_layer.properties_dialog.force_close()
        
        if hasattr(self, 'loop'):
            self.loop.exit()
        
        # self._setupAnimationDialog.forceClose()
        
        return super().closeEvent(event)

    def reset_view(self):
        camera = self._view.camera
        
        camera.center = self._home_center
        camera.distance = self._home_distance
        camera.azimuth = self._home_azimuth
        camera.elevation = self._home_elevation
        camera.scale_factor = self._default_scale_factor
        camera.roll = self._home_roll

        self._synced_camera_orientation = (
            self._home_azimuth, self._home_elevation, self._home_roll
        )

    def set_2d_view(self):
        camera = self._view.camera

        camera.azimuth = _TWO_D_VIEW_AZIMUTH
        camera.elevation = _TWO_D_VIEW_ELEVATION
        camera.roll = _TWO_D_VIEW_ROLL

        self._synced_camera_orientation = (
            _TWO_D_VIEW_AZIMUTH, _TWO_D_VIEW_ELEVATION, _TWO_D_VIEW_ROLL
        )

        first_channel = list(self._channels_data.keys())[0]
        first_node = self._channels_data[first_channel].node
        bounds = [first_node.bounds(axis) for axis in range(3)]

        center_local = np.array([
            (axis_bounds[0] + axis_bounds[1]) / 2
            for axis_bounds in bounds
        ])

        transform = first_node.transform
        center_scene = transform.map(center_local)

        camera.center = center_scene[:3]

        camera.distance = self._home_distance

        if self.is_view_synced():
            # Centre and zoom on the region visible in the parent GUI
            self._sync_camera_to_parent_view_region()

    def set_camera_view(self):
        """Reset the camera to the default orientation and fit the volume."""
        if not self._channels_data:
            T, Z, Y, X = self._lab.shape
            dummy_vol = np.zeros((Z, Y, X), dtype=np.float32)
            self.set_volume(dummy_vol, channel_name='1')
        
        first_channel = list(self._channels_data.keys())[0]
        first_volume = self._channels_data[first_channel].volume
        first_node = self._channels_data[first_channel].node
        T, Z, Y, X = first_volume.shape

        xyz_center = (X/2, Y/2, Z/2)     
        corners = np.array([
            [0, 0, 0],
            [X, Y, Z],
        ])
        world = first_node.transform.map(corners)[:, :3]
        diag = np.linalg.norm(world[1] - world[0])
        
        self._home_center = first_node.transform.map(xyz_center)[:3]
        self._home_elevation = _HOME_ELEVATION
        self._home_azimuth = _HOME_AZIMUTH
        self._home_fov = _HOME_FOV
        self._home_roll = _HOME_ROLL
        self._home_distance = diag

        self._view.camera.set_range()
        self._view.camera.center = first_node.transform.map(xyz_center)[:3]
        self._view.camera.elevation = self._home_elevation
        self._view.camera.azimuth = self._home_azimuth
        self._view.camera.roll = self._home_roll
        self._view.camera.fov = self._home_fov
        self._view.camera.distance = diag
        
        self._default_scale_factor = self._view.camera.scale_factor
        self._is_camera_view_set = True
        self._synced_camera_orientation = (
            self._home_azimuth, self._home_elevation, self._home_roll
        )

        self._canvas.update()
    
    def save_screenshot(self):
        ...
    
    def keyPressEvent(self, event):
        return super().keyPressEvent(event)
    
    def setup_animation_parameters(self, checked):
        if checked:
            self._setupAnimationDialog.show()
            self._setupAnimationDialog.set_from_camera(
                self._view.camera, 
                update_start=not self._is_animation_setup
            )
        else:
            self._setupAnimationDialog.hide()

    def play_animation(self, checked):
        self._animation_params

    def resizeEvent(self, event):
        super().resizeEvent(event)

        if self._labels_text_overlay is not None:
            self._labels_text_overlay.resize(self._canvas.native.size())

    def are_frames_synced(self):
        if self._parent_gui is None:
            return False

        return self._sync_frames_scrollbar_main_gui_checkbox.isChecked()

    def set_parent_cursor_pos(
            self,
            x: float | None,
            y: float | None,
            plane: ParentViewPlane='xy',
            depth_slice: int | None=None,
        ):
        """Show a mirrored cursor (diamond) at the position of the mouse
        cursor on the parent 2D GUI image. Pass `x=None` to hide it.

        Parameters
        ----------
        x, y : float or None
            Position of the mouse cursor in full-resolution image
            coordinates of the parent GUI (where pixel ``i`` spans
            ``[i, i+1]``). If None, the cursor is hidden.
        plane : {'xy', 'zy', 'zx'}, optional
            Plane displayed in the parent GUI (see
            `set_parent_view_region`). Default is 'xy'.
        depth_slice : int or None, optional
            Index of the displayed slice along the axis perpendicular to
            `plane`. If None (the parent GUI displays a projection), the
            cursor is placed at the centre of the volume along that axis.
            Default is None.
        """
        is_hidden = (
            x is None or y is None
            or self._data_shape is None
            or self._voxel_size_transform is None
        )
        if not is_hidden:
            SizeZ, SizeY, SizeX = self._data_shape[-3:]
            shape_xyz = np.array((SizeX, SizeY, SizeZ), dtype=float)
            h_axis, v_axis, d_axis = _PARENT_VIEW_PLANE_AXES[plane]

            # In pyqtgraph pixel i spans [i, i+1] while in vispy voxel i
            # spans [i-0.5, i+0.5] --> subtract 0.5
            pos = np.zeros(3)
            pos[h_axis] = x - 0.5
            pos[v_axis] = y - 0.5
            if depth_slice is None:
                pos[d_axis] = (shape_xyz[d_axis] - 1) / 2
            else:
                pos[d_axis] = depth_slice

            is_hidden = np.any(pos < -0.5) or np.any(pos > shape_xyz - 0.5)

        marker = self._parent_cursor_marker
        if is_hidden:
            if marker is not None and marker.visible:
                marker.visible = False
                self._canvas.update()
            return

        if marker is None:
            # Fixed size in pixels and drawn last without depth test so
            # that it is always visible
            marker = vispy.scene.visuals.Markers(
                parent=self._view.scene, scaling='fixed'
            )
            marker.set_gl_state('translucent', depth_test=False)
            marker.order = 1000
            self._parent_cursor_marker = marker

        pos_scene = self._voxel_size_transform.map(pos)[:3]
        marker.set_data(
            pos_scene[np.newaxis].astype(np.float32),
            symbol=_PARENT_CURSOR_SYMBOL,
            size=_PARENT_CURSOR_SIZE,
            face_color=_PARENT_CURSOR_FACE_RGBA,
            edge_color=_PARENT_CURSOR_EDGE_RGBA,
            edge_width=1,
        )
        marker.visible = True
        self._canvas.update()

    def is_view_synced(self):
        if self._parent_gui is None:
            return False

        return self._sync_view_main_gui_checkbox.isChecked()

    def set_parent_view_region(
            self,
            x_range: tuple[float, float],
            y_range: tuple[float, float],
            plane: ParentViewPlane='xy',
            depth_slice: int | None=None,
        ):
        """Set the region currently visible in the parent 2D GUI.

        If the "Sync pan/zoom with parent GUI" checkbox is checked, the
        region is used to centre and zoom the camera (without changing its
        orientation).

        Parameters
        ----------
        x_range, y_range : tuple[float, float]
            Horizontal and vertical range of the parent 2D view in
            full-resolution image coordinates, i.e., the output of
            ``pyqtgraph.ViewBox.viewRange()`` where pixel ``i`` spans
            ``[i, i+1]``.
        plane : {'xy', 'zy', 'zx'}, optional
            Plane displayed in the parent GUI. The horizontal and vertical
            axes are X and Y for 'xy', Y and Z for 'zy', and X and Z for
            'zx'. Default is 'xy'.
        depth_slice : int or None, optional
            Index of the displayed slice along the axis perpendicular to
            `plane` (e.g., the z-slice for 'xy'). Pass None when the parent
            GUI displays a projection. Default is None.
        """
        if plane not in _PARENT_VIEW_PLANE_AXES:
            raise ValueError(
                f'Invalid plane "{plane}". '
                f'Valid planes are {list(_PARENT_VIEW_PLANE_AXES.keys())}'
            )

        prev_region = self._parent_view_region
        region = {
            'x_range': (float(min(x_range)), float(max(x_range))),
            'y_range': (float(min(y_range)), float(max(y_range))),
            'plane': plane,
            'depth_slice': depth_slice,
        }
        self._parent_view_region = region

        # Move the camera only when the parent GUI was panned/zoomed (or the
        # plane changed). Changing slice or frame only stores the new slice
        # (used at the next sync) so that the user's 3D zoom/pan is not
        # discarded.
        is_pan_zoom = prev_region is None or any(
            prev_region[key] != region[key]
            for key in ('x_range', 'y_range', 'plane')
        )
        if is_pan_zoom:
            self._update_parent_view_region()

    def _update_parent_view_region(self):
        if self._parent_view_region is None:
            return

        if self.is_view_synced():
            self._sync_camera_to_parent_view_region()

    def _get_parent_view_region_voxel_coords(self):
        """Return the parent view region in full-resolution voxel
        coordinates in vispy (x, y, z) order.

        Returns
        -------
        None or tuple of (lower, upper, depth, axes)
            `lower` and `upper` are the corners of the region, spanning
            the entire volume along the depth axis. `depth` is the
            coordinate of the displayed slice (centre of the volume for
            projections) and `axes` is the tuple of indices (horizontal,
            vertical, depth) of the displayed axes.
        """
        region = self._parent_view_region
        if region is None or self._data_shape is None:
            return

        SizeZ, SizeY, SizeX = self._data_shape[-3:]
        shape_xyz = np.array((SizeX, SizeY, SizeZ), dtype=float)
        axes = _PARENT_VIEW_PLANE_AXES[region['plane']]
        h_axis, v_axis, d_axis = axes

        # In pyqtgraph pixel i spans [i, i+1] while in vispy voxel i spans
        # [i-0.5, i+0.5] --> subtract 0.5
        lower = np.zeros(3)
        upper = np.zeros(3)
        lower[h_axis], upper[h_axis] = np.array(region['x_range']) - 0.5
        lower[v_axis], upper[v_axis] = np.array(region['y_range']) - 0.5
        lower[d_axis], upper[d_axis] = -0.5, shape_xyz[d_axis] - 0.5

        depth = region['depth_slice']
        if depth is None:
            depth = (shape_xyz[d_axis] - 1) / 2
        depth = float(np.clip(depth, 0, shape_xyz[d_axis] - 1))

        return lower, upper, depth, axes

    def _scene_vectors_to_camera_axes(self, vectors: np.ndarray):
        """Express scene vectors in the camera axes (right, up, towards the
        camera). Only the current camera orientation is used.
        """
        transform = self._view.camera.transform # camera --> scene
        origin = transform.imap(np.zeros(3))
        mapped = transform.imap(vectors)
        origin = origin[:3] / origin[3]
        mapped = mapped[:, :3] / mapped[:, 3:4]
        return mapped - origin

    def _sync_camera_to_parent_view_region(self, update=True):
        """Centre the camera on the region visible in the parent GUI (at
        the depth of the displayed slice) and zoom so that the region fits
        the canvas.

        Any manual rotation is discarded: the camera orientation is reset
        to the last selected view ("Home" or "2D view"). The fit accounts
        for the orientation and perspective, therefore when looking
        perpendicular to the parent GUI plane (e.g., "2D view" with the 'xy'
        plane), the 3D view matches the parent GUI view.
        """
        if not self._is_camera_view_set:
            return

        if self._voxel_size_transform is None:
            return

        coords = self._get_parent_view_region_voxel_coords()
        if coords is None:
            return

        # Restore the orientation before fitting (the fit depends on it)
        camera = self._view.camera
        azimuth, elevation, roll = self._synced_camera_orientation
        camera.azimuth = azimuth
        camera.elevation = elevation
        camera.roll = roll

        lower, upper, depth, axes = coords
        h_axis, v_axis, d_axis = axes

        # Corners of the region at the depth of the displayed slice
        corners = np.zeros((4, 3))
        corners[:, d_axis] = depth
        corners[:, h_axis] = (
            lower[h_axis], upper[h_axis], upper[h_axis], lower[h_axis]
        )
        corners[:, v_axis] = (
            lower[v_axis], lower[v_axis], upper[v_axis], upper[v_axis]
        )
        corners_scene = self._voxel_size_transform.map(corners)[:, :3]
        center = corners_scene.mean(axis=0)

        # Corners relative to the centre in camera axes
        offsets = self._scene_vectors_to_camera_axes(corners_scene - center)
        offsets_right = np.abs(offsets[:, 0])
        offsets_up = np.abs(offsets[:, 1])
        offsets_towards_camera = offsets[:, 2]

        canvas_width, canvas_height = self._view.size
        if canvas_width > 0 and canvas_height > 0:
            aspect_ratio = canvas_width / canvas_height
        else:
            aspect_ratio = 1.0

        if camera.fov > 0:
            # With the camera at `distance` from the centre, a point with
            # camera-axes coordinates (x, y, z) is at depth (distance - z)
            # and it is visible if |y| <= (distance - z)*tan(fov/2) and
            # |x| <= (distance - z)*tan(fov/2)*aspect_ratio (fov is the
            # vertical field of view).
            tan_half_fov = np.tan(np.radians(camera.fov) / 2)
            distance = np.max(
                offsets_towards_camera + np.maximum(
                    offsets_up / tan_half_fov,
                    offsets_right / (tan_half_fov * aspect_ratio)
                )
            )
            # Visible height at the centre
            visible_height = 2 * distance * tan_half_fov
        else:
            distance = None
            visible_height = 2 * max(
                offsets_up.max(), offsets_right.max() / aspect_ratio
            )

        if not visible_height > 0:
            return

        camera.center = tuple(center)

        # vispy applies `scale_factor` to the shorter side of the canvas,
        # while the field of view (and hence the `distance`) is vertical.
        # We set both to keep them consistent (as vispy does when zooming
        # with the mouse wheel).
        if aspect_ratio > 1:
            camera.scale_factor = visible_height
        else:
            camera.scale_factor = visible_height * aspect_ratio

        if distance is not None:
            camera.distance = distance

        if update:
            self._canvas.update()