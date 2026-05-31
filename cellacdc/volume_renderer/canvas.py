from dataclasses import dataclass
from functools import partial

import numpy as np

from qtpy.QtCore import (
    Signal, Qt, QCoreApplication, QEventLoop
)
from qtpy.QtWidgets import (
    QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, 
    QGraphicsProxyWidget    
)

import pyqtgraph as pg

from .. import printl
from .._run import _setup_app
from .. import widgets
from .. import colors

from . import _widgets

RgbaColor = colors.RgbaColor
AcdcPyQtGraphColorMapName = colors.AcdcPyQtGraphColorMapName
ColorMapPerChannel = (
    list[list[RgbaColor]]
    | dict[str, list[RgbaColor]]
    | list[AcdcPyQtGraphColorMapName]
    | dict[str, AcdcPyQtGraphColorMapName]
)

@dataclass
class _ChannelData:
    node: object
    lut_item: widgets.baseHistogramLUTitem
    volume: np.ndarray
    auto_button: QPushButton
    reset_button: QPushButton

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
    
    def __init__(
            self, 
            app=None, 
            parent=None, 
            version=None,
            title='Cell-ACDC - Volume Renderer',
            hide_on_close=True
        ):
        """Initializer."""

        super().__init__(parent)
        self.setWindowTitle(title)

        self._version = version
        self._ui_initialised = False
        self._hide_on_close = hide_on_close
        self._max_texture_3d = None
        
        if app is None:
            app = QCoreApplication.instance()
        
        self.app = app
        
        self._channels_data: dict[_ChannelData] = {}
        self._default_rgbs = self._init_default_rgbs()
        
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
    
    def _init_ui(self):
        if self._ui_initialised:
            return
        
        from vispy.scene import visuals
        
        self._lab_node = visuals.Volume(
            np.zeros((2, 2, 2), dtype=np.float32),
            method="translucent",
            interpolation="nearest",
            parent=self._view.scene,
        )
        self._lab_node.visible = False
        
        self._toolbar = _widgets.VolumeRendererToolbar(parent=self)
        self.addToolBar(Qt.TopToolBarArea, self._toolbar)
        
        self._toolbar.sigHomeView.connect(self.reset_view)
        self._toolbar.sigSave.connect(self.save_screenshot)
        
        lut_items_graphics_layout = pg.GraphicsLayoutWidget()
        lut_items_graphics_layout.setBackground('black')
        self._lut_items_layout = lut_items_graphics_layout.addLayout(
            row=0, col=0
        )
        self._lut_items_graphics_layout = lut_items_graphics_layout
        
        self._scene_layout = QHBoxLayout()
        self._scene_layout.addWidget(lut_items_graphics_layout, stretch=0)
        self._scene_layout.addWidget(self._canvas.native, stretch=1)
        
        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addLayout(self._scene_layout)
        self.setCentralWidget(central)
        
        self._ui_initialised = True
    
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
    
    @staticmethod
    def _downsample(vol: np.ndarray, max_size: int) -> np.ndarray:
        """
        Return a stride-subsampled view of *vol* so no dimension exceeds *max_size*.

        Uses integer strides (fast, no interpolation) — suitable for interactive
        previews.  Returns the original array unchanged if no downsampling is needed.
        """
        strides = tuple(max(1, int(np.ceil(s / max_size))) for s in vol.shape)
        if all(s == 1 for s in strides):
            return vol
        return np.ascontiguousarray(vol[::strides[0], ::strides[1], ::strides[2]])
    
    def _preprocess_volume(self, volume: np.ndarray):
        if volume.ndim != 3:
            raise ValueError(
                f'Expected 3-D (Z, Y, X) array; got shape {volume.shape}')
        
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
            strides = tuple(max(1, int(np.ceil(s / max_tex))) for s in vol.shape)
            self._last_strides = strides
            vol = self._downsample(vol, max_tex)
        else:
            self._last_strides = (1, 1, 1)

        # Normalise the (possibly downsampled) array using the full-resolution range.
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        else:
            vol = np.zeros_like(vol)
        
        return vol
    
    def _init_default_rgbs(self):
        self._default_rgbs = colors.overlay_rgbs.copy()
        self._default_rgbs.insert(0, (255, 255, 255))
    
    def _get_channel_default_cmap(self, channel_name: str):        
        foregr_rgb = colors.FLUO_CHANNELS_COLORS.get(channel_name)
        if foregr_rgb is None:
            foregr_rgb = self._default_rgbs.pop(0)
        
        pg_gradient = colors.get_pg_gradient([(0,0,0,0), foregr_rgb])
        lut_item = self._channels_data[channel_name].lut_item
        lut_item.setGradient(pg_gradient)
        cmap = colors.pg_to_vispy_cmap(lut_item)
        return cmap
    
    def _set_channels_data(
            self, 
            volumes: list[np.ndarray],
            channel_names: list[str],
            cmaps: ColorMapPerChannel=None,
        ):
        from vispy.scene import visuals
        
        for channel, volume in zip(channel_names, volumes):
            col = len(self._channels_data)
            
            auto_btn = QPushButton('Auto')
            auto_btn_proxy = QGraphicsProxyWidget()
            auto_btn_proxy.setWidget(auto_btn)
            self._lut_items_layout.addItem(auto_btn_proxy, row=0, col=col)
            
            reset_btn = QPushButton('Reset')
            reset_btn_proxy = QGraphicsProxyWidget()
            reset_btn_proxy.setWidget(reset_btn)
            self._lut_items_layout.addItem(reset_btn_proxy, row=1, col=col)
            
            node = visuals.Volume(
                volume,
                method="mip",
                parent=self._view.scene,
            )
            
            lut_item = widgets.baseHistogramLUTitem()
            
            channel_data = _ChannelData(
                node=node,
                lut_item=lut_item,
                volume=volume,
                auto_button=auto_btn,
                reset_button=reset_btn
            )
            
            auto_btn.clicked.connect(
                partial(self._on_auto_clim, channel_data=channel_data)
            )
            reset_btn.clicked.connect(
                partial(self._on_reset_clim, channel_data=channel_data)
            )
            
            self._lut_items_layout.addItem(lut_item, row=2, col=col)
        
        if cmaps is None:
            cmaps = {
                ch: self._get_channel_default_cmap(ch) for ch in channel_names
            }
    
    def _on_auto_clim(self, channel_data: _ChannelData) -> None:
        lut_item = channel_data.lut_item
        
        if len(lut_item.gradient.listTicks()) != 2:
            self.logger.info(
                '[WARNING]: Auto contrast is available only with LUTs '
                'that have two ticks.'
            )
            return
        
        volume = channel_data.volume
        lo, hi = colors.get_auto_contrast_percentile(volume)
        (low_tick, _), (high_tick, _) = lut_item.gradient.listTicks()            
        lut_item.gradient.setTickValue(high_tick, hi)
        lut_item.gradient.setTickValue(low_tick, lo)
    
    def _on_reset_clim(self, lut_item: widgets.baseHistogramLUTitem) -> None:
        lut_item = channel_data.lut_item
        lut_item.resetState()
    
    def set_volume(
            self,
            volume: np.ndarray, 
            channel_name: str='',
            cmap: list[RgbaColor] | AcdcPyQtGraphColorMapName=None,
        ):        
        num_channels = len(self._channels_data)
        if not channel_name:
            channel_name = f'channel_{num_channels+1}'

        cmaps = None
        if cmap is not None:
            cmaps = {channel_name: cmap}
        
        volumes = {channel_name: volume}
        
        self.set_volumes(volumes, cmaps)
    
    def set_volumes(
            self,
            volumes: list[np.ndarray] | dict[str, np.ndarray], 
            channel_names: list[str] | None=None,
            cmaps: ColorMapPerChannel=None,
        ):
        if isinstance(volumes, dict):
            channel_names = list(volumes.keys())
            volumes = list(volumes.values())
        
        volumes = [self._preprocess_volume(vol) for vol in volumes]
        
        num_existing_channels = len(self._channels_data)
        tot_num_channels = num_existing_channels + len(volumes)
        
        if channel_names is None:
            channel_names = [
                f'channel_{c+1}' 
                for c in range(num_existing_channels, tot_num_channels)
            ]
        
        self._set_channels_data(
            volumes, channel_names, 
            cmaps=cmaps
        )
        
        printl(cmaps)
        printl(channel_names)
        
    def show(self, block=False):
        self.resize(960, 720)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()

        try:
            self.setEnabled(True)
        except Exception as err:
            pass

        if block:
            self._block_exec()
    
    def run(self, block=True):
        if self.app is None:
            app, splashScreen = _setup_app(splashscreen=True)  
            splashScreen.close()
        
        self.show()
        self.reset_view()
        if block:
            self._block_exec()
    
    def closeEvent(self, event):
        if self._hide_on_close:
            event.ignore()
            self.hide()
            return
            
        if hasattr(self, 'loop'):
            self.loop.exit()
        
        return super().closeEvent(event)

    def reset_view(self):
        """Reset the camera to the default orientation and fit the volume."""
        self._view.camera.set_range()
        self._view.camera.elevation = 30.0
        self._view.camera.azimuth = -60.0
        self._canvas.update()
    
    def save_screenshot(self):
        ...