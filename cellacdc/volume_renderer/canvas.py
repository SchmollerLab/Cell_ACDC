from dataclasses import dataclass
from functools import partial
from math import ceil

from typing import Literal

import numpy as np
import pandas as pd

import skimage

from qtpy.QtCore import (
    Signal, Qt, QCoreApplication, QEventLoop
)
from qtpy.QtWidgets import (
    QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, 
    QGraphicsProxyWidget, QGroupBox, QCheckBox, QMenu, QRadioButton, 
    QButtonGroup   
)

import pyqtgraph as pg

from .. import printl
from .._run import _setup_app
from .. import apps, widgets
from .. import colors
from .. import plot

from . import _widgets, utils

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

@dataclass
class _ChannelData:
    node: vispy.scene.visuals.Volume
    
    volume: np.ndarray
    _raw_volume: np.ndarray
    _off_focus_volume: np.ndarray
    
    auto_button: QPushButton
    reset_button: QPushButton
    lut_item: widgets.baseHistogramLUTitem
    
    opacity_slider: widgets.sliderWithSpinBox
    toolbutton: widgets.OverlayChannelToolButton

@dataclass
class _PointsLayer:
    name: str
    
    points_xyz: np.ndarray # (N, [x, y, z]) voxel coordinates
    _label_ids: np.ndarray | None # (N, id) id in self._orig_lab
    labels: list[str] | None
    
    markers: vispy.scene.visuals.Markers
    
    toolbutton: widgets.PointsLayerToolButton
    properties_dialog: apps.EditPointsLayerAppearanceDialog
    context_menu: widgets.PointsLayerContextMenu
    
    df: pd.DataFrame | None=None
    text: vispy.scene.visuals.Text | None=None
    
class VolumeRendererWindow(QMainWindow):
    """
    A standalone Qt window that displays a 3D z-stack volume using vispy.

    Usage (minimal)::

        renderer = VolumeRenderer3DWindow()
        renderer.set_volume(zstack_array)   # (Z, Y, X) numpy array
        renderer.run()

    The window hides (rather than closes) when the user presses X so the GPU
    state is preserved for re-display.  Pass ``hide_on_close=False`` to
    destroy on close instead.
    """
    sigClose = Signal(object)
    sigUpdate = Signal()
    
    def __init__(
            self, 
            app=None, 
            parent=None, 
            version=None,
            title='Cell-ACDC - Volume Renderer',
            hide_on_close=False,
            logger_func=print
        ):
        """Initializer."""
        self._version = version
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
        self._orig_lab = None
        self._canvas = None
        self._voxel_size = None
        self._voxel_size_transform = None
        self._lab_ncolors = 256
        self._voxel_size_strides_transform = None
        self._downsample_strides = None
        self._object_labels_list_buttongroup = None
        self._data_shape = None
        
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
            elevation=30.0, 
            azimuth=-60.0
        )

        # XYZ axis indicator at the front-bottom-left corner of the volume.
        # Red=X (data axis 2), Green=Y (data axis 1), Blue=Z (data axis 0).
        # Scale and visibility are updated in update_volume on first load.
        self._axis_visual = scene.visuals.XYZAxis(parent=self._view.scene)
        self._axis_visual.visible = False
    
    def _init_lab_ui_items(
            self, 
            lab_volume, 
            cmap_name=None,
            gradient_item_state=None,
        ):
        from vispy.scene import visuals
        from .gl_blend import volume_gl_state
        
        vmin, vmax = float(lab_volume.min()), float(lab_volume.max())
        
        self._lab_node = visuals.Volume(
            lab_volume,
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
        
        self._object_labels_list_buttongroup = QButtonGroup(self)
        self._object_labels_list_buttongroup.setExclusive(False)
        for obj in self._rp:
            obj_checkbox = QCheckBox(f'{obj.label}')
            obj_checkbox.setChecked(True)
            obj_checkbox.obj = obj
            self._object_labels_list_layout.addWidget(obj_checkbox)
            self._object_labels_list_buttongroup.addButton(obj_checkbox)
            obj_checkbox.toggled.connect(
                partial(self._set_object_checked, obj=obj)
            )
            obj_checkbox.setDisabled(True)
        
        self._object_labels_list_layout.addStretch(1)
        
        select_all_button = widgets.selectAllPushButton()
        select_all_button.sigClicked.connect(
            self._set_all_object_labels_list_checked
        )
        self._right_vertical_layout.addWidget(select_all_button)
        
        self._right_vertical_layout.addSpacing(10)
        
        display_mode_groupbox = QGroupBox('Display mode')
        display_mode_layout = QVBoxLayout()
        display_mode_groupbox.setLayout(display_mode_layout)
        
        display_mode_buttonsgroup = QButtonGroup(self)
        
        self._display_mode_show_all_rb = QRadioButton(
            'Show all objets', self
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

        self._update_display()
    
    def _update_lab_node(self, update=True):
        for checkbox in self._object_labels_list_buttongroup.buttons(): 
            obj = checkbox.obj
            checked = checkbox.isChecked()
            _id = obj.label if checked else 0
            self._lab[obj.slice][obj.image] = _id
            
        self._lab_node.set_data(self._lab)
        
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
        for channel_data in self._channels_data.values():
            if self._display_mode_show_all_rb.isChecked():
                displayed_volume = channel_data._raw_volume
                channel_data.node.set_data(displayed_volume)
                continue
            
            displayed_volume = channel_data.volume
            if self._display_mode_focus_selected_rb.isChecked():
                unselected_vals = channel_data._off_focus_volume
                np.copyto(displayed_volume, unselected_vals)
            elif self._display_mode_hide_unselected_rb.isChecked():
                displayed_volume[:] = 0
            
            for checkbox in self._object_labels_list_buttongroup.buttons(): 
                obj = checkbox.obj
                checked = checkbox.isChecked()
                if not checked:
                    continue
                
                src_intensities = (
                    channel_data._raw_volume[obj.slice][obj.image]
                )
                displayed_volume[obj.slice][obj.image] = src_intensities
            
            channel_data.node.set_data(displayed_volume)

        if update:
            self._canvas.update()
    
    def _update_display(self, exclude_lab=False, exclude_markers=False):
        if not exclude_lab:
            self._update_lab_node(update=False)
        
        if not exclude_markers:
            self._update_markers(update=False)

        self._update_volume_nodes(update=False)
        
        self._canvas.update()
        
    def _init_points_layer_ui_items(
            self,
            markers: vispy.scene.visuals.Markers,
            name: str,
            points_xyz: np.ndarray | None=None, # (N, [z, y, x]) voxel coordinates
            labels: list[str] | None=None,
            color: vispy.color.Color='red',
            size: float=8.0,
            opacity: float=1.0,
            symbol: MarkerSymbols='disc',
            points_df: pd.DataFrame | None=None,
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
        
        lut = self._lab_gradient_item.colorMap().getLookupTable(
            0.0, 1.0, self._lab_ncolors
        )
        labels = [obj.label for obj in self._rp]
        greedy_lut = colors.get_greedy_lut(self._lab, lut, ids=labels)
        
        greedy_lut = np.array(greedy_lut) / 255.0
        greedy_lut = colors.replace_background_rgba_lut(greedy_lut)
        
        cmap = VisPyColormap(greedy_lut)
        
        self._lab_node.cmap = cmap
        self._canvas.update()
    
    def _set_object_checked(self, checked: bool, obj=None):
        if obj is None:
            return
        
        _id = obj.label if checked else 0
        self._lab[obj.slice][obj.image] = _id
        
        self._lab_node.set_data(self._lab)
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
        cmap = colors.pg_to_vispy_cmap(
            lab_gradient_item.colorMap(), transparent_zero=True,
            n=self._lab_ncolors
        )
        self._lab_node.cmap = cmap
        if update:
            self._canvas.update()
    
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
        self._toolbar.sigSave.connect(self.save_screenshot)
        self._toolbar.sigSetSingleChannel.connect(self._set_single_channel)
        
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
        
        self._scene_layout = QHBoxLayout()
        self._scene_layout.addWidget(lut_items_graphics_layout, stretch=0)
        self._scene_layout.addWidget(self._canvas.native, stretch=10)
        self._scene_layout.addLayout(
            self._lab_gradient_item_layout, stretch=0
        )
        self._scene_layout.addLayout(
            self._right_vertical_layout, stretch=1
        )
        
        self._controls_groupbox = QGroupBox('Controls')
        self._controls_layout = widgets.FormLayout()
        self._controls_groupbox.setLayout(self._controls_layout)
        
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addLayout(self._scene_layout)
        main_layout.addWidget(self._controls_groupbox)
        main_layout.setStretch(0, 10)
        main_layout.setStretch(1, 0)
        self.setCentralWidget(central)
        
        self._ui_initialised = True
    
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
        
    
    def _block_exec(self):
        if hasattr(self, 'loop'):
            self.loop.exit()
            
        self.loop = QEventLoop()
        self.loop.exec_()
    
    def _resolve_max_texture_3d(self) -> int:
        """Return the GPU's maximum 3-D texture size (cached after first call)."""
        if self._max_texture_3d is not None:
            return self._max_texture_3d
        try:
            from vispy.gloo.util import get_max_texture_sizes  # noqa: PLC0415
            _max_2d, max_3d = get_max_texture_sizes()
            self._max_texture_3d = int(max_3d)
        except Exception:
            # Fallback: conservatively assume 512³ if GL query fails.
            self._max_texture_3d = 512
        return self._max_texture_3d
    
    def _downsample(self, vol: np.ndarray, max_size: int) -> np.ndarray:
        """
        Return a stride-subsampled view of *vol* so no dimension exceeds *max_size*.

        Uses integer strides (fast, no interpolation) — suitable for interactive
        previews.  Returns the original array unchanged if no downsampling is needed.
        """
        if self._downsample_strides is None:
            strides = tuple(
                max(1, int(np.ceil(s / max_size))) for s in vol.shape
            )
            self._downsample_strides = strides
        else:
            strides = self._downsample_strides
        
        if all(s == 1 for s in strides):
            return vol
        return np.ascontiguousarray(vol[::strides[0], ::strides[1], ::strides[2]])
    
    def _preprocess_volume(self, volume: np.ndarray):
        if volume.ndim != 3:
            raise ValueError(
                f'Expected 3-D (Z, Y, X) array; got shape {volume.shape}')
        
        self._SizeZ = len(volume)
        
        # copy=False avoids a redundant allocation when data is already float32
        vol = volume.astype(np.float32, copy=False)

        # Compute the value range on the full-resolution data BEFORE downsampling
        # so that stride-based subsampling cannot accidentally exclude extreme voxels
        # and cause incorrect normalisation (e.g. a single bright fluorescence spot
        # being dropped by the stride may lower the apparent maximum).
        vmin, vmax = float(vol.min()), float(vol.max())

        # Downsample to fit GPU texture limits (after range is already captured).
        # Store the per-axis strides so set_voxel_scale can correct for
        # non-uniform compression (e.g. stride_x=4 while stride_z=1).
        max_tex = self._resolve_max_texture_3d()
        if max(vol.shape) > max_tex:
            vol = self._downsample(vol, max_tex)
            self._last_max_tex = max_tex

        # Normalise the (possibly downsampled) array using the full-resolution range.
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        else:
            vol = np.zeros_like(vol)
        
        return vol
    
    def _preprocess_lab(self, lab: np.ndarray):
        if lab.ndim != 3:
            raise ValueError(
                f'Expected 3-D (Z, Y, X) labels array; got shape {lab.shape}')
        
        max_tex = self._resolve_max_texture_3d()
        lab = self._downsample(lab, max_tex)
        
        return lab
    
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
                volume,
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
            
            _raw_volume = volume.copy()
            _off_focus_volume = skimage.filters.gaussian(_raw_volume, sigma=1.2)
            _off_focus_volume *= (
                np.percentile(_raw_volume, 99.9) 
                / np.percentile(_off_focus_volume, 99.9)
            )
            _off_focus_volume *= 0.8

            channel_data = _ChannelData(
                node=node,
                lut_item=lut_item,
                volume=volume,
                _raw_volume=_raw_volume,
                _off_focus_volume=_off_focus_volume,
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
        
        volume = channel_data.volume
        lo, hi = colors.get_auto_contrast_percentile(volume)
        (low_tick, _), (high_tick, _) = lut_item.gradient.listTicks()            
        lut_item.gradient.setTickValue(high_tick, hi)
        lut_item.gradient.setTickValue(low_tick, lo)
    
    def _on_reset_clim(self, channel_data: _ChannelData) -> None:
        lut_item = channel_data.lut_item
        lut_item.resetState()
        
    def _set_gl_blend_states(self):
        from .gl_blend import volume_gl_state

        for c, (channel, channel_data) in enumerate(self._channels_data.items()):
            blending = "translucent_no_depth" if c == 0 else "additive"
            node = channel_data.node
            node.order = c
            node.opacity = channel_data.opacity_slider.value()
            node.set_gl_state(**volume_gl_state(blending, first_visible=c==0))    
    
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
        
        points_xyz = points_layer.points_xyz
        if self._orig_lab is not None:
            visible_label_ids = [
                int(cb.text()) 
                for cb in self._object_labels_list_buttongroup.buttons()
                if cb.isChecked()
            ]
            visible = np.isin(points_layer._label_ids, visible_label_ids)
            points_xyz = points_xyz[visible]

        markers = points_layer.markers
        markers.set_data(
            points_xyz, 
            symbol=symbol,
            size=size,
            face_color=face_color,
            edge_color=edge_color
        )
        
        if update:
            self._canvas.update()
    
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
            SizeZ: int=None
        ):
        if self._is_labels_set:
            self._logger_func(
                '[WARNING]: Labels already set. '
                'Only one labels volumes can be set'
            )
            return
        
        if self._data_shape is None:
            self._data_shape = lab.shape
        
        if lab.shape != self._data_shape:
            raise ValueError(
                f'Labels array shape {lab.shape} does not match the current '
                f'viewer shape {self._data_shape}. '
                'All displayed volumes must have the same shape.'
            )
        
        if lab.dtype == bool:
            lab = lab.astype(np.uint8)
        
        if lab.ndim == 2 and SizeZ is None and self._SizeZ is None:
            raise ValueError(
                f'Labels array is 2D but SizeZ not set.'
            )
        
        if lab.ndim == 2:
            if SizeZ is None:
                SizeZ = self._SizeZ
        
            lab = np.array([lab]*SizeZ)
        
        if lab.ndim != 3:
            raise ValueError(
                f'Expected 3-D (Z, Y, X) labels array; got shape {lab.shape}')
        
        self._orig_lab = lab.copy()
        self._lab = self._preprocess_lab(lab)
        self._rp = skimage.measure.regionprops(self._lab)
        
        self._init_lab_ui_items(
            self._lab, 
            cmap_name=cmap_name, 
            gradient_item_state=gradient_item_state
        )

        if self._voxel_size_strides_transform is not None:
            self._lab_node.transform = self._voxel_size_strides_transform
        else:
            self._set_voxel_size_strides_transform(voxel_size)
            
        self._is_labels_set = True
        
    set_segmentation_masks = set_labels
    
    def set_volume(
            self,
            volume: np.ndarray, 
            channel_name: str='',
            voxel_size: tuple[float, float, float]=None,
            lut_item_state: LutItemState | None=None, 
            cmap: list[colors.RgbaColor] | AcdcPyQtGraphColorMapName=None,
        ):        
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
            voxel_size=voxel_size
        )
    
    def _set_voxel_size_strides_transform(
            self, 
            voxel_size: tuple[float, float, float] | None
        ):
        if voxel_size is None:
            voxel_size = self._voxel_size
        
        if voxel_size is None:
            voxel_size = (1.0, 1.0, 1.0)
        
        if self._downsample_strides:
            self._downsample_strides = (1.0, 1.0, 1.0)
        
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
        ):
        if isinstance(volumes, dict):
            channel_names = list(volumes.keys())
            volumes = list(volumes.values())
        
        if self._data_shape is None:
            self._data_shape = volumes[0].shape
        
        for volume in volumes:
            if volume.shape != self._data_shape:
                raise ValueError(
                    f'Volume shape {volume.shape} does not match the current '
                    f'viewer shape {self._data_shape}. '
                    'All displayed volumes must have the same shape.'
                ) 
        
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
        
        self._canvas.update()
    
    def add_points_layer(
            self,
            name: str,
            points: np.ndarray | None=None, # (N, [z, y, x]) voxel coordinates
            points_df: pd.DataFrame | None=None,
            zyx_columns_names: list[str] | None=None,
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
        
        if zyx_columns_names is None:
            zyx_columns_names = ['z', 'y', 'x']
        
        if points_df is not None:
            points_xyz = points_df[zyx_columns_names[::-1]].to_numpy()
        elif points is not None:
            points = np.asarray(points)
            points_xyz = points[:, [2, 1, 0]]
        else:
            raise ValueError(
                "Either 'points' or 'points_df' must be provided."
            ) 
        
        face_color = vispy.color.Color(color, alpha=opacity*0.5)
        edge_color = vispy.color.Color(color, alpha=opacity)
        markers = visuals.Markers(
            parent=self._view.scene,
            spherical=True,
            scaling=scaling
        )
        
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
        
        _label_ids = None
        if self._orig_lab is not None:
            _label_ids = self._orig_lab[
                points_xyz[:, 2], 
                points_xyz[:, 1], 
                points_xyz[:, 0], 
            ]
        
        ui_items = self._init_points_layer_ui_items(
            markers,
            name,
            points_xyz, 
            labels=labels,
            color=edge_color,
            size=size,
            opacity=opacity,
            symbol=symbol,
            points_df=points_df
        )
        
        toolbutton, properties_dialog, context_menu = ui_items
        
        points_layer = _PointsLayer(
            name=name,
            points_xyz=points_xyz, 
            _label_ids=_label_ids,
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
        
        context_menu.sigEditPropertes.connect(
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
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        
        self.set_camera_view()

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
        
        return super().closeEvent(event)

    def reset_view(self):
        camera = self._view.camera
        
        camera.center = self._home_center
        camera.distance = self._home_distance
        camera.azimuth = self._home_azimuth
        camera.elevation = self._home_elevation
        camera.scale_factor = self._default_scale_factor
    
    def set_camera_view(self):
        """Reset the camera to the default orientation and fit the volume."""
        if not self._channels_data:
            Z, Y, X = self._lab.shape
            dummy_vol = np.zeros((Z, Y, X), dtype=np.float32)
            self.set_volume(dummy_vol, channel_name='1')
        
        first_channel = list(self._channels_data.keys())[0]
        first_volume = self._channels_data[first_channel].volume
        first_node = self._channels_data[first_channel].node
        Z, Y, X = first_volume.shape

        xyz_center = (X/2, Y/2, Z/2)     
        corners = np.array([
            [0, 0, 0],
            [X, Y, Z],
        ])
        world = first_node.transform.map(corners)[:, :3]
        diag = np.linalg.norm(world[1] - world[0])
        
        self._home_center = first_node.transform.map(xyz_center)[:3]
        self._home_elevation = 30.0
        self._home_azimuth = 45.0
        self._home_fov = 60.0
        self._home_distance = diag
        
        self._view.camera.set_range()
        self._view.camera.center = first_node.transform.map(xyz_center)[:3]
        self._view.camera.elevation = 30.0
        self._view.camera.azimuth = 45.0
        self._view.camera.fov = 60.0
        self._view.camera.distance = diag
        
        self._default_scale_factor = self._view.camera.scale_factor
        
        self._canvas.update()
    
    def save_screenshot(self):
        ...