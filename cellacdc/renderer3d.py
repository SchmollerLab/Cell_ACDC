from __future__ import annotations
from functools import partial

import numpy as np

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
    QGraphicsProxyWidget,
    QGridLayout
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence

import pyqtgraph as pg

from cellacdc import printl
from cellacdc import widgets
from cellacdc._run import _setup_app
from cellacdc import colors

# ---------------------------------------------------------------------------
# Rendering-mode registry
# Matches exactly the set in vispy.scene.visuals.Volume._rendering_methods
# ---------------------------------------------------------------------------
RENDERING_MODES: list[tuple[str, str]] = [
    ('mip',            'Max Intensity Projection (MIP)'),
    ('minip',          'Min Intensity Projection'),
    ('attenuated_mip', 'Attenuated MIP'),
    ('iso',            'Isosurface'),
    ('translucent',    'Translucent'),
    ('additive',       'Additive'),
    ('average',        'Average Intensity'),
]

COLORMAPS: list[str] = [
    'grays', 'viridis', 'hot', 'coolwarm', 'blues', 'reds',
    'greens', 'plasma', 'inferno', 'magma',
]

# Practical subset of vispy's 17 interpolation filters for 3D volumes.
# 'Linear' is napari's default (interpolation3d); 'Nearest' is fastest.
INTERPOLATION_MODES: list[tuple[str, str]] = [
    ('linear',  'Linear'),
    ('nearest', 'Nearest'),
    ('catrom',  'Catmull-Rom (sharp)'),
]

# Gaussian sigma used for smooth-ISO pre-filter (approximates napari's
# SMOOTH_GRADIENT_DEFINITION Sobel-Feldman shader without GLSL injection).
_SMOOTH_ISO_SIGMA: float = 1.0

# Modes that use the ISO threshold parameter
_ISO_MODES: frozenset[str] = frozenset({'iso'})
# Modes that use the attenuation parameter
_ATTN_MODES: frozenset[str] = frozenset({'attenuated_mip'})
# Modes where mip_cutoff = lower clim makes background transparent (napari approach)
_MIP_CUTOFF_MODES: frozenset[str] = frozenset({'mip', 'attenuated_mip'})
# Modes where minip_cutoff = upper clim makes bright values transparent
_MINIP_CUTOFF_MODES: frozenset[str] = frozenset({'minip'})

# Default ray-marching step size — matches napari and vispy defaults.
# Smaller → more samples per ray → sharper but slower.  Range: (0, 1].
_DEFAULT_STEP_SIZE: float = 0.8

# Depiction modes (napari: layer.depiction)
# 'volume' = full 3D raycasting; 'plane' = single cross-section in 3D space.
# For plane modes we store the axis index (0=Z, 1=Y, 2=X) in the data key.
DEPICTION_MODES: list[tuple[str, str]] = [
    ('volume',   'Volume'),
    ('plane_z',  'XY-Plane (Z-axis)'),
    ('plane_y',  'XZ-Plane (Y-axis)'),
    ('plane_x',  'YZ-Plane (X-axis)'),
]

# Map depiction key → (plane_normal in scene XYZ, data-shape axis moved by slider)
_PLANE_CONFIGS: dict[str, tuple[list[float], int]] = {
    #             normal (scene x,y,z)  axis in data shape (z,y,x)
    'plane_z':  ([0.0, 0.0, 1.0],      0),   # normal along scene-Z = data-Z axis
    'plane_y':  ([0.0, 1.0, 0.0],      1),   # normal along scene-Y = data-Y axis
    'plane_x':  ([1.0, 0.0, 0.0],      2),   # normal along scene-X = data-X axis
}

# Overlay channels live in ``_volume_nodes`` with this prefix (e.g.
# ``overlay:0``).  Primary fluorescence channels use plain names from
# ``self.channels`` and have LUT sliders; overlays do not.
OVERLAY_CHANNEL_PREFIX = 'overlay:'


def is_overlay_channel(channel: str) -> bool:
    return channel.startswith(OVERLAY_CHANNEL_PREFIX)


def overlay_channel_name(index: int) -> str:
    return f'{OVERLAY_CHANNEL_PREFIX}{index}'


OVERLAY_KIND_FLUO = 'fluo'
OVERLAY_KIND_SEGM = 'segm'
OVERLAY_KIND_OL_LABELS = 'ol_labels'


def _volume_cmap_from_spec(cmap_spec):
    """Build a vispy colormap from a PG ColorMap, colour name, or passthrough."""
    if cmap_spec is None:
        return colors.vispy_cmap_from_spec('green')
    if hasattr(cmap_spec, 'getLookupTable'):
        return colors.pg_to_vispy_cmap(cmap_spec)
    if isinstance(cmap_spec, str):
        return colors.vispy_cmap_from_spec(cmap_spec)
    return cmap_spec


def _parse_overlay_entry(entry, index: int):
    """Return (data, opacity, cmap_spec, mode_override, meta) from an overlay tuple."""
    data, opacity, cmap_spec = entry[0], entry[1], entry[2]
    mode_override = None
    meta: dict = {'overlay_index': index}
    if len(entry) > 3:
        if isinstance(entry[3], str):
            mode_override = entry[3]
            if len(entry) > 4 and isinstance(entry[4], dict):
                meta.update(entry[4])
        elif isinstance(entry[3], dict):
            meta.update(entry[3])
    return data, opacity, cmap_spec, mode_override, meta


def _mask_labels_for_display(labels: np.ndarray, cell_id: int) -> np.ndarray:
    """Build a float32 overlay mask from integer label data and active cell ID."""
    lab = labels.astype(np.int32, copy=False)
    if int(cell_id) <= 0:
        return (lab > 0).astype(np.float32)
    return (lab == int(cell_id)).astype(np.float32)


def _scene_pos_to_voxel_indices(
        local_pos,
        shape: tuple[int, int, int],
    ) -> tuple[int, int, int] | None:
    """Convert vispy volume local coordinates to (z, y, x) voxel indices."""
    nz, ny, nx = shape
    x = int(np.floor(float(local_pos[0]) + 0.5))
    y = int(np.floor(float(local_pos[1]) + 0.5))
    z = int(np.floor(float(local_pos[2]) + 0.5))
    if 0 <= z < nz and 0 <= y < ny and 0 <= x < nx:
        return (z, y, x)
    return None


# ---------------------------------------------------------------------------
# Pure-stdlib PNG writer (fallback when skimage is not available)
# ---------------------------------------------------------------------------
def _write_png(path: str, rgba: np.ndarray) -> None:
    """Write an RGBA uint8 array to *path* as PNG using only stdlib."""
    import struct, zlib  # noqa: PLC0415

    h, w = rgba.shape[:2]

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack('>I', len(data)) + tag + data
        return c + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b''.join(b'\x00' + rgba[y].tobytes() for y in range(h))
    with open(path, 'wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n')
        f.write(_chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 6, 0, 0, 0)))
        f.write(_chunk(b'IDAT', zlib.compress(raw, 9)))
        f.write(_chunk(b'IEND', b''))


# ---------------------------------------------------------------------------
# Adapter protocol (for type-checking / documentation)
# ---------------------------------------------------------------------------
class VolumeRendererAdapter:
    """
    Minimal interface an application must implement to drive VolumeRenderer3DWindow.

    Concrete examples live outside this module (e.g. in Cell_ACDC's gui.py).
    The adapter is responsible for:
      - Fetching the current (Z, Y, X) numpy array from the application data.
      - Calling ``renderer.update_volume(data)`` and ``renderer.show()``.
    """

    def get_current_zstack(self) -> np.ndarray:
        """Return the current (Z, Y, X) uint/float numpy array."""
        raise NotImplementedError

    def get_voxel_sizes(self) -> tuple[float, float, float] | None:
        """Return (dz, dy, dx) physical voxel sizes in µm, or None if unknown."""
        return None

    def on_renderer_closed(self) -> None:
        """Called when the renderer window is closed (hidden)."""

    def on_main_overlay_changed(self) -> None:
        """Push overlay opacity/cmap changes from the main GUI to the renderer."""

    def apply_overlay_control_from_renderer(
            self,
            channel_name: str,
            opacity: float | None = None,
            gradient_state: dict | None = None,
            labels_alpha: float | None = None,
        ) -> None:
        """Update main GUI overlay widgets from the 3D renderer (bidirectional sync)."""

    def get_available_cell_ids(self) -> list[int]:
        """Return valid cell IDs for the current frame, or empty if unknown."""
        return []

    def apply_cell_id_from_renderer(self, cell_id: int) -> None:
        """Update main GUI cell selection from the 3D renderer."""

    def on_cell_id_changed_from_main(self, cell_id: int) -> None:
        """Push main GUI cell ID selection to the 3D renderer."""


# ---------------------------------------------------------------------------
# Controls widget
# ---------------------------------------------------------------------------
class VolumeRendererControls(QWidget):
    """Parameter panel that drives a VolumeRenderer3DWindow."""

    def __init__(
            self, 
            renderer: 'VolumeRenderer3DWindow', 
            parent: QWidget | None = None,
            channels: list[str] | None = None,
        ):
        super().__init__(parent)
        self._renderer = renderer
        if channels is None:
            channels = ['Channel 1']  # default single channel name
        self._channels = channels
        self._build()

    def _build(self) -> None:
        layout = widgets.FormLayout()
        self.setLayout(layout)
        
        row = 0
        self._gamma_spin = widgets.sliderWithSpinBox(
            title_loc='in_line', 
            isFloat=True, 
            parent=self,
            normalize_factor=10
        )
        
        self._gamma_spin.setRange(0.1, 5.0)
        self._gamma_spin.setSingleStep(0.1)
        self._gamma_spin.setValue(1.0)
        self._gamma_spin.setToolTip('Gamma correction')
        self._gamma_spin.valueChanged.connect(self._on_gamma_changed)
        _gamma_form_widget = widgets.formWidget(
            self._gamma_spin,
            labelTextLeft='Gamma:',
        )
        layout.addFormWidget(_gamma_form_widget, row=row)
        
        row += 1
        self._step_spin = widgets.sliderWithSpinBox(
            title_loc='in_line', 
            isFloat=True, 
            parent=self,
            normalize_factor=10
        )
        self._step_spin.setRange(0.1, 2.0)
        self._step_spin.setSingleStep(0.1)
        self._step_spin.setValue(_DEFAULT_STEP_SIZE)
        self._step_spin.setDecimals(2)
        self._step_spin.setToolTip(
            'Ray-marching step size relative to voxel size.\n'
            'Smaller = sharper rendering; larger = faster.'
        )
        self._step_spin.valueChanged.connect(self._on_step_changed)
        _step_form_widget = widgets.formWidget(
            self._step_spin,
            labelTextLeft='Step:',
        )
        layout.addFormWidget(_step_form_widget, row=row)

        row += 1
        cell_id_row = QHBoxLayout()
        self._cell_id_spin = widgets.SpinBox()
        self._cell_id_spin.setMinimum(0)
        self._cell_id_spin.setMaximum(999999)
        self._cell_id_spin.setToolTip(
            'Show only this cell ID in segmentation overlays (0 = all cells).\n'
            'Shift+left-click on the volume to pick a cell.'
        )
        self._cell_id_spin.valueChanged.connect(self._on_cell_id_changed)
        cell_id_row.addWidget(self._cell_id_spin)
        self._show_all_cells_btn = QPushButton('Show all')
        self._show_all_cells_btn.setToolTip(
            'Reset to show all labelled cells (Cell ID 0)'
        )
        self._show_all_cells_btn.clicked.connect(self._on_show_all_cells)
        cell_id_row.addWidget(self._show_all_cells_btn)
        cell_id_widget = QWidget()
        cell_id_widget.setLayout(cell_id_row)
        _cell_id_form = widgets.formWidget(
            cell_id_widget,
            labelTextLeft='Cell ID:',
        )
        layout.addFormWidget(_cell_id_form, row=row)

        row += 1
        # Primary-channel opacity is controlled via the right-side grayscale
        # colorbar in the 3D window (see VolumeRenderer3DWindow._add_opacity_lut_items).

        layout.addNewColumn(with_separator=True)

        row = 0
        self._mode_combo = QComboBox()
        for mode_id, label in RENDERING_MODES:
            self._mode_combo.addItem(label, mode_id)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        _mode_form_widget = widgets.formWidget(
            self._mode_combo,
            labelTextLeft='Rendering mode:',
        )
        layout.addFormWidget(_mode_form_widget, row=row)
        
        # Interpolation (napari: interpolation3d)
        row += 1
        self._interp_combo = QComboBox()
        for iid, ilabel in INTERPOLATION_MODES:
            self._interp_combo.addItem(ilabel, iid)
        self._interp_combo.setToolTip(
            'Volume texture interpolation (Linear = smooth, Nearest = pixelated)'
        )
        self._interp_combo.currentIndexChanged.connect(self._on_interp_changed)
        _interp_form_widget = widgets.formWidget(
            self._interp_combo,
            labelTextLeft='Interpolation:',
        )
        layout.addFormWidget(_interp_form_widget, row=row)

        row += 1
        self._iso_spin = QDoubleSpinBox()
        self._iso_spin.setRange(0.0, 1.0)
        self._iso_spin.setSingleStep(0.01)
        self._iso_spin.setValue(0.5)
        self._iso_spin.setDecimals(3)
        self._iso_spin.setToolTip('Isosurface threshold')
        self._iso_spin.valueChanged.connect(self._on_iso_changed)
        self._iso_label = _iso_form_widget = widgets.formWidget(
            self._iso_spin,
            labelTextLeft='ISO:',
        )
        layout.addFormWidget(_iso_form_widget, row=row)

        row += 1
        self._smooth_iso_cb = QCheckBox('Pre-smooth volume for ISO rendering')
        self._smooth_iso_cb.setToolTip(
            'Pre-smooth the volume with a Gaussian filter (σ=1) before ISO\n'
            'rendering. Approximates napari\'s SMOOTH_GRADIENT_DEFINITION\n'
            'Sobel-Feldman shader — produces softer surface normals without\n'
            'requiring custom GLSL injection.'
        )
        self._smooth_iso_cb.toggled.connect(self._on_smooth_iso_changed)
        layout.addFormWidget(self._smooth_iso_cb, row=row)

        row += 1
        self._attn_spin = QDoubleSpinBox()
        self._attn_spin.setRange(0.0, 2.0)
        self._attn_spin.setSingleStep(0.05)
        self._attn_spin.setValue(0.5)
        self._attn_spin.setDecimals(2)
        self._attn_spin.setToolTip('Attenuation factor for Attenuated MIP')
        self._attn_spin.valueChanged.connect(self._on_attn_changed)
        self._attn_label = _attn_form_widget = widgets.formWidget(
            self._attn_spin,
            labelTextLeft='Attenuation:',
        )
        layout.addFormWidget(_attn_form_widget, row=row)

        row += 1
        self._depict_combo = QComboBox()
        for did, dlabel in DEPICTION_MODES:
            self._depict_combo.addItem(dlabel, did)
        self._depict_combo.setToolTip(
            'Volume: full 3D raycasting.  '
            'Z-Plane: single cross-section embedded in 3D space.'
        )
        self._depict_combo.currentIndexChanged.connect(self._on_depiction_changed)
        layout.addFormWidget(
            widgets.formWidget(self._depict_combo, labelTextLeft='Depiction:'),
            row=row,
        )

        row += 1
        self._zplane_slider = QSlider(Qt.Horizontal)
        self._zplane_slider.setMinimum(0)
        self._zplane_slider.setMaximum(99)
        self._zplane_slider.setValue(50)
        self._zplane_slider.setToolTip(
            'Position of the cross-section plane (0=start, 100=end)'
        )
        self._zplane_slider.valueChanged.connect(self._on_zplane_changed)
        self._zplane_label = _zplane_form_widget = widgets.formWidget(
            self._zplane_slider,
            labelTextLeft='Plane pos:',
        )
        layout.addFormWidget(_zplane_form_widget, row=row)

        row += 1
        self._plane_thick_spin = QDoubleSpinBox()
        self._plane_thick_spin.setRange(1.0, 50.0)
        self._plane_thick_spin.setSingleStep(1.0)
        self._plane_thick_spin.setValue(1.0)
        self._plane_thick_spin.setDecimals(1)
        self._plane_thick_spin.setToolTip(
            'Thickness of the plane cross-section in voxels.\n'
            'Mirrors napari\'s plane.thickness parameter.'
        )
        self._plane_thick_spin.valueChanged.connect(self._on_plane_thickness_changed)
        self._plane_thick_label = _plane_thick_form_widget = widgets.formWidget(
            self._plane_thick_spin,
            labelTextLeft='Plane thick:',
        )
        layout.addFormWidget(_plane_thick_form_widget, row=row)

        # Initially hide plane controls
        self._zplane_label.setVisible(False)
        self._zplane_slider.setVisible(False)
        self._plane_thick_label.setVisible(False)
        self._plane_thick_spin.setVisible(False)

        # Initial visibility for mode-specific controls
        self._update_mode_controls('mip')

    # -- mode-aware control visibility ----------------------------------------

    def _update_mode_controls(self, mode: str) -> None:
        show_iso = mode in _ISO_MODES
        show_attn = mode in _ATTN_MODES
        self._iso_label.setVisible(show_iso)
        self._smooth_iso_cb.setVisible(show_iso)
        self._attn_label.setVisible(show_attn)

    # -- slots ----------------------------------------------------------------

    def _on_mode_changed(self, idx: int) -> None:
        mode = self._mode_combo.itemData(idx)
        self._update_mode_controls(mode)
        self._renderer.set_rendering_mode(mode)

    def _on_cell_id_changed(self, value: int) -> None:
        if getattr(self._renderer, '_syncing_cell_id_from_main', False):
            return
        if getattr(self._renderer, '_syncing_cell_id_from_renderer', False):
            return
        self._renderer.set_active_cell_id(int(value))

    def _on_show_all_cells(self) -> None:
        self._cell_id_spin.setValue(0)

    def _on_gamma_changed(self, value: float) -> None:
        self._renderer.set_gamma(value)

    def _on_iso_changed(self, value: float) -> None:
        self._renderer.set_iso_threshold(value)

    def _on_attn_changed(self, value: float) -> None:
        self._renderer.set_attenuation(value)

    def _on_interp_changed(self, idx: int) -> None:
        iid = self._interp_combo.itemData(idx)
        self._renderer.set_interpolation(iid)

    def _on_step_changed(self, value: float) -> None:
        self._renderer.set_step_size(value)

    def _on_smooth_iso_changed(self, checked: bool) -> None:
        self._renderer.set_smooth_iso(checked)

    def _on_depiction_changed(self, idx: int) -> None:
        mode = self._depict_combo.itemData(idx)
        is_plane = mode in _PLANE_CONFIGS
        self._zplane_label.setVisible(is_plane)
        self._zplane_slider.setVisible(is_plane)
        self._plane_thick_label.setVisible(is_plane)
        self._plane_thick_spin.setVisible(is_plane)
        if is_plane:
            # Label shows which data axis the slider moves along.
            axis_names = {'plane_z': 'Z:', 'plane_y': 'Y:', 'plane_x': 'X:'}
            self._zplane_label.labelLeft.setText(
                axis_names.get(mode, 'Plane pos:')
            )
            # Reset the slider to centre so slider position always matches the
            # rendered plane (set_depiction initialises the plane at 0.5).
            self._zplane_slider.blockSignals(True)
            self._zplane_slider.setValue(50)
            self._zplane_slider.blockSignals(False)
        self._renderer.set_depiction(mode)

    def _on_zplane_changed(self, value: int) -> None:
        self._renderer.set_zplane_position(value / 100.0)

    def _on_plane_thickness_changed(self, value: float) -> None:
        self._renderer.set_plane_thickness(value)

    def _on_opacity_changed(self, value: float, channel: str) -> None:
        self._renderer.set_opacity(value, channel=channel)


# ---------------------------------------------------------------------------
# Main renderer window
# ---------------------------------------------------------------------------
class VolumeRenderer3DWindow(QMainWindow):
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
            parent: QWidget | None = None,
            hide_on_close: bool = False,
            adapter: VolumeRendererAdapter | None = None,
        ) -> None:
        
        from vispy.scene import visuals
        
        self.app, self.splashScreen = _setup_app(splashscreen=True)  
        super().__init__(parent)
        self.setWindowTitle('3D Z-Stack Renderer')
        self.resize(960, 720)
        
        self._hide_on_close = hide_on_close
        self._adapter = adapter
        self._volume_nodes: dict[str, visuals.Volume] | None = None
        self._volumes_data: dict[str, np.ndarray] | None = None
        self._raw_volumes_data: dict[str, np.ndarray] | None = None
        self.lut_items: dict[str, widgets.baseHistogramLUTitem] | None = None
        self._opacity_lut_items: dict[str, widgets.baseHistogramLUTitem] = {}
        self._overlay_meta: dict[str, dict] = {}
        self._overlay_widgets: dict[str, dict] = {}
        self._overlay_controls_layout: QVBoxLayout | None = None
        self._overlay_controls_host: QWidget | None = None
        self._syncing_overlay_from_main = False
        self._syncing_overlay_from_renderer = False
        self._label_volumes: dict[str, np.ndarray] = {}
        self._active_cell_id: int = 0
        self._syncing_cell_id_from_main = False
        self._syncing_cell_id_from_renderer = False
        self._overlay_mode_overrides: dict[str, str | None] = {}
        self.channels: list[str] | None = None
        self._last_shape: tuple | None = None
        self._max_texture_3d: int | None = None  # resolved on first upload
        self._smooth_iso: bool = False
        self._gpu_data_is_smoothed: bool = False  # tracks whether GPU texture has Gaussian filter
        self._last_raw_data: np.ndarray | None = None  # float32, for re-render
        self._ui_initialised: bool = False
        self._is_set_volumes_first_call: bool = True
        
        self._init_vispy()

    # -- volume-node helpers (primary + overlay share ``_volume_nodes``) ------

    def _ensure_volume_nodes(self) -> dict:
        if self._volume_nodes is None:
            self._volume_nodes = {}
        return self._volume_nodes

    def _has_volume_nodes(self) -> bool:
        return bool(self._volume_nodes)

    def _each_volume_node(self):
        if self._volume_nodes:
            yield from self._volume_nodes.values()

    def _volume_shape(self, channel: str) -> tuple[int, int, int] | None:
        if self._volumes_data and channel in self._volumes_data:
            return self._volumes_data[channel].shape
        return self._last_shape

    def _remove_overlay_channels(self, clear_widgets: bool = True) -> None:
        if not self._volume_nodes:
            if clear_widgets:
                self._clear_overlay_widgets()
            return
        for name in list(self._volume_nodes):
            if not is_overlay_channel(name):
                continue
            self._volume_nodes[name].parent = None
            del self._volume_nodes[name]
            if self._volumes_data is not None:
                self._volumes_data.pop(name, None)
            self._overlay_meta.pop(name, None)
            self._label_volumes.pop(name, None)
        self._overlay_mode_overrides.clear()
        if clear_widgets:
            self._clear_overlay_widgets()

    def _clear_overlay_widgets(self) -> None:
        self._overlay_widgets.clear()
        if self._overlay_controls_layout is None:
            return
        while self._overlay_controls_layout.count():
            item = self._overlay_controls_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _normalize_overlay_volume(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 3:
            raise ValueError(
                f'Expected 3-D (Z, Y, X) overlay array; got shape {data.shape}'
            )
        vol = data.astype(np.float32, copy=False)
        vmin, vmax = float(vol.min()), float(vol.max())
        max_tex = self._resolve_max_texture_3d()
        if max(vol.shape) > max_tex:
            vol = self._downsample(vol, max_tex)
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        else:
            vol = np.zeros_like(vol)
        return vol

    def _store_label_volume(
            self,
            channel_name: str,
            label_data: np.ndarray,
        ) -> None:
        if label_data.ndim != 3:
            return
        lab = label_data.astype(np.int32, copy=False)
        max_tex = self._resolve_max_texture_3d()
        if max(lab.shape) > max_tex:
            lab = self._downsample(lab, max_tex)
        self._label_volumes[channel_name] = np.ascontiguousarray(lab)

    def _overlay_volume_from_labels(self, channel_name: str) -> np.ndarray | None:
        labels = self._label_volumes.get(channel_name)
        if labels is None:
            return None
        mask = _mask_labels_for_display(labels, self._active_cell_id)
        return self._normalize_overlay_volume(mask)

    def _connect_canvas_events(self) -> None:
        if not hasattr(self, '_canvas') or self._canvas is None:
            return
        self._canvas.events.mouse_press.connect(self._on_canvas_mouse_press)

    def _on_canvas_mouse_press(self, event) -> None:
        modifiers = getattr(event, 'modifiers', ()) or ()
        if event.button != 1 or 'Shift' not in modifiers:
            return
        picked_id = self._pick_cell_id_at_canvas_pos(event.pos)
        if picked_id is None:
            return
        self.set_active_cell_id(int(picked_id))

    def _pick_cell_id_at_canvas_pos(self, canvas_pos) -> int | None:
        key = self._find_overlay_by_kind(OVERLAY_KIND_SEGM)
        if key is None:
            for candidate, labels in self._label_volumes.items():
                if labels is not None:
                    key = candidate
                    break
        if key is None or self._volume_nodes is None:
            return None
        labels = self._label_volumes.get(key)
        volume_node = self._volume_nodes.get(key)
        if labels is None or volume_node is None:
            return None
        try:
            scene_pos = self._canvas.scene.node_transform.imap(canvas_pos)
            node_tr = volume_node.transform
            if node_tr is None:
                return None
            inv = node_tr.inverse
            if inv is None:
                return None
            local_pos = inv.map(scene_pos)
        except Exception:
            return None
        indices = _scene_pos_to_voxel_indices(local_pos, labels.shape)
        if indices is None:
            return None
        z, y, x = indices
        cell_id = int(labels[z, y, x])
        return cell_id if cell_id > 0 else None

    def update_cell_id_range(self) -> None:
        if self._controls is None:
            return
        adapter = self._adapter
        if adapter is None or not hasattr(adapter, 'get_available_cell_ids'):
            return
        ids = adapter.get_available_cell_ids()
        if not ids:
            return
        spin = self._controls._cell_id_spin
        spin.setMaximum(max(ids))

    def set_active_cell_id(
            self,
            cell_id: int,
            *,
            sync_main_gui: bool = True,
        ) -> None:
        cell_id = int(cell_id)
        self._active_cell_id = cell_id
        if self._controls is not None and not self._syncing_cell_id_from_main:
            self._syncing_cell_id_from_renderer = True
            try:
                spin = self._controls._cell_id_spin
                spin.blockSignals(True)
                spin.setValue(cell_id)
                spin.blockSignals(False)
            finally:
                self._syncing_cell_id_from_renderer = False

        if self._volume_nodes is not None:
            for key in list(self._label_volumes.keys()):
                vol = self._overlay_volume_from_labels(key)
                if vol is None:
                    continue
                self._volumes_data[key] = vol
                volume_node = self._volume_nodes.get(key)
                if volume_node is not None:
                    volume_node.set_data(vol, clim=(0.0, 1.0))

        if (
            sync_main_gui
            and not self._syncing_cell_id_from_main
            and self._adapter is not None
            and hasattr(self._adapter, 'apply_cell_id_from_renderer')
        ):
            self._adapter.apply_cell_id_from_renderer(cell_id)

        if hasattr(self, '_canvas') and self._canvas is not None:
            self._canvas.update()

    # -- vispy setup ----------------------------------------------------------

    def _init_vispy(self) -> None:
        # Configure the backend to match the host Qt binding.
        import vispy
        from qtpy import API_NAME
        vispy.use(API_NAME)

        from vispy import scene  # noqa: PLC0415 (late import required by vispy)

        self._canvas = scene.SceneCanvas(keys='interactive', bgcolor='black')
        self._view = self._canvas.central_widget.add_view()
        # TurntableCamera: left-drag to orbit, scroll to zoom, right-drag to pan.
        # Keeps the "up" axis fixed, which suits microscopy z-stacks better than
        # a free arcball.
        self._view.camera = scene.cameras.TurntableCamera(
            fov=45, elevation=30.0, azimuth=-60.0
        )

        # XYZ axis indicator at the front-bottom-left corner of the volume.
        # Red=X (data axis 2), Green=Y (data axis 1), Blue=Z (data axis 0).
        # Scale and visibility are updated in update_volume on first load.
        self._axis_visual = scene.visuals.XYZAxis(parent=self._view.scene)
        self._axis_visual.visible = False

    def _configure_renderer_lut_item(
            self,
            lut_item: widgets.myHistogramLUTitem,
        ) -> None:
        """Strip 2D-only gradient menu entries not applicable to the 3D renderer."""
        lut_item.removeAddScaleBarAction()
        lut_item.removeAddTimestampAction()
        menu = lut_item.gradient.menu
        for attr in (
            'invertBwAction',
            'fontSizeMenu',
            'contLineWightActionGroup',
            'mothBudLineWightActionGroup',
        ):
            item = getattr(lut_item, attr, None)
            if item is None:
                continue
            try:
                if hasattr(item, 'menuAction'):
                    menu.removeAction(item.menuAction())
                else:
                    menu.removeAction(item)
            except Exception:
                pass
        for attr in ('textColorButton', 'contoursColorButton', 'mothBudLineColorButton'):
            button = getattr(lut_item, attr, None)
            if button is None:
                continue
            for action in menu.actions():
                if action.defaultWidget() is not None:
                    try:
                        if button in action.defaultWidget().findChildren(type(button)):
                            menu.removeAction(action)
                    except Exception:
                        pass

    def _add_lut_items(self, scene_layout: QHBoxLayout) -> None:
        self.lut_items_graphics_layout = pg.GraphicsLayoutWidget()
        self.lut_items_graphics_layout.setBackground('black')
        self.lut_items_layout = self.lut_items_graphics_layout.addLayout(
            row=0, col=0
        )
        self.lut_items = {}
        total_width = 0
        for c, channel in enumerate(self.channels):
            auto_btn = QPushButton('Auto')
            auto_btn_proxy = QGraphicsProxyWidget()
            auto_btn_proxy.setWidget(auto_btn)
            self.lut_items_layout.addItem(auto_btn_proxy, row=0, col=c)
            
            full_btn = QPushButton('Full')
            full_btn.setToolTip(
                'Reset contrast limits to full normalized range [0, 1]'
            )
            full_btn_proxy = QGraphicsProxyWidget()
            full_btn_proxy.setWidget(full_btn)
            self.lut_items_layout.addItem(full_btn_proxy, row=1, col=c)
            
            lut_item = widgets.myHistogramLUTitem(
                parent=self,
                name=channel,
                axisLabel='Clim:',
                include_rescale_lut_options=False,
            )
            self._configure_renderer_lut_item(lut_item)
            self.lut_items[channel] = (lut_item, auto_btn, full_btn)
            self.lut_items_layout.addItem(lut_item, row=2, col=c)
            
            lut_item.channel = channel
            
            lut_item.sigLookupTableChanged.connect(self._on_lut_changed)
            auto_btn.clicked.connect(
                partial(self._on_auto_clim, lut_item=lut_item)
            )
            full_btn.clicked.connect(
                partial(self._on_full_clim, lut_item=lut_item)
            )
            
            total_width += lut_item.sizeHint(Qt.PreferredSize).width()
        
        scene_layout.addWidget(self.lut_items_graphics_layout, stretch=0)
        self.lut_items_graphics_layout.setFixedWidth(int(total_width + 20))

    def _add_opacity_lut_items(self, scene_layout: QHBoxLayout) -> None:
        self._opacity_lut_graphics_layout = pg.GraphicsLayoutWidget()
        self._opacity_lut_graphics_layout.setBackground('black')
        self._opacity_lut_layout = self._opacity_lut_graphics_layout.addLayout(
            row=0, col=0
        )
        total_width = 0
        for c, channel in enumerate(self.channels):
            opacity_lut = widgets.baseHistogramLUTitem(
                parent=self,
                name=f'opacity_{channel}',
                axisLabel='Opacity:',
                gradientPosition='left',
                include_rescale_lut_options=False,
            )
            opacity_lut.channel = channel
            opacity_lut.vb.hide()
            from pyqtgraph.graphicsItems.GradientEditorItem import Gradients
            opacity_lut.setGradient(Gradients['grey'])
            ticks = opacity_lut.gradient.listTicks()
            if len(ticks) >= 2:
                opacity_lut.gradient.setTickValue(ticks[0][0], 0.0)
                opacity_lut.gradient.setTickValue(ticks[-1][0], 1.0)
            opacity_lut.sigLookupTableChanged.connect(
                partial(self._on_opacity_lut_changed, lut_item=opacity_lut)
            )
            primary_lut = self.lut_items[channel][0]
            primary_lut.setChildLutItem(opacity_lut)
            self._opacity_lut_items[channel] = opacity_lut
            self._opacity_lut_layout.addItem(opacity_lut, row=0, col=c)
            total_width += opacity_lut.sizeHint(Qt.PreferredSize).width()

        scene_layout.addWidget(self._opacity_lut_graphics_layout, stretch=0)
        self._opacity_lut_graphics_layout.setFixedWidth(int(total_width + 20))

    def _on_opacity_lut_changed(
            self,
            lut_item: widgets.baseHistogramLUTitem,
        ) -> None:
        ticks_pos = [x for _t, x in lut_item.gradient.listTicks()]
        if not ticks_pos:
            return
        opacity = max(0.0, min(1.0, max(ticks_pos)))
        self.set_opacity(opacity, channel=lut_item.channel)

    def _sync_opacity_lut_value(
            self,
            channel: str,
            opacity: float,
        ) -> None:
        opacity_lut = self._opacity_lut_items.get(channel)
        if opacity_lut is None:
            return
        ticks = opacity_lut.gradient.listTicks()
        if not ticks:
            return
        high_tick = max(ticks, key=lambda item: item[1])[0]
        opacity_lut.gradient.setTickValue(high_tick, max(0.0, min(1.0, opacity)))

    def _pg_cmap_to_gradient(self, pg_cmap):
        table = pg_cmap.getLookupTable(0.0, 1.0, 2)
        rgba = [tuple(row) for row in table]
        return colors.get_pg_gradient(rgba)

    def _rebuild_overlay_controls(self, overlay_entries) -> None:
        if self._overlay_controls_layout is None:
            return
        self._clear_overlay_widgets()
        if not overlay_entries:
            if self._overlay_controls_host is not None:
                self._overlay_controls_host.hide()
            return
        if self._overlay_controls_host is not None:
            self._overlay_controls_host.show()

        for channel_name, opacity, cmap_spec, _mode_override, meta in overlay_entries:
            kind = meta.get('kind', '')
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)
            widgets_info: dict = {}

            if kind == OVERLAY_KIND_FLUO:
                label = meta.get('channel_name', channel_name)
                lut_item = widgets.baseHistogramLUTitem(
                    parent=self,
                    name=channel_name,
                    axisLabel=label,
                    include_rescale_lut_options=False,
                )
                lut_item.vb.hide()
                if hasattr(cmap_spec, 'getLookupTable'):
                    lut_item.setGradient(self._pg_cmap_to_gradient(cmap_spec))
                elif isinstance(cmap_spec, str):
                    bkgr = (0, 0, 0, 255)
                    fg = colors.FLUO_CHANNELS_COLORS.get(
                        cmap_spec, (0, 255, 0, 255)
                    )
                    if len(fg) == 3:
                        fg = (*fg, 255)
                    lut_item.setGradient(colors.get_pg_gradient((bkgr, fg)))
                lut_item.sigLookupTableChanged.connect(
                    partial(
                        self._on_overlay_lut_changed,
                        overlay_key=channel_name,
                        lut_item=lut_item,
                    )
                )
                row_layout.addWidget(lut_item)
                widgets_info['lut_item'] = lut_item

            opacity_spin = widgets.sliderWithSpinBox(
                title_loc='in_line',
                isFloat=True,
                parent=row_widget,
                normalize_factor=20,
            )
            opacity_spin.setRange(0.0, 1.0)
            opacity_spin.setSingleStep(0.05)
            opacity_spin.setDecimals(2)
            opacity_spin.setValue(float(opacity))
            if kind == OVERLAY_KIND_SEGM:
                opacity_spin.setToolTip('Segmentation mask overlay opacity')
                label_widget = QLabel('Segmentation opacity:')
                row_layout.addWidget(label_widget)
            elif kind == OVERLAY_KIND_OL_LABELS:
                opacity_spin.setToolTip(
                    f'Opacity for overlay labels ({meta.get("channel_name", "")})'
                )
            else:
                opacity_spin.setToolTip(
                    f'Opacity for overlay channel {meta.get("channel_name", "")}'
                )
            opacity_spin.valueChanged.connect(
                partial(
                    self._on_overlay_opacity_changed,
                    overlay_key=channel_name,
                )
            )
            row_layout.addWidget(opacity_spin)
            widgets_info['opacity_spin'] = opacity_spin
            self._overlay_widgets[channel_name] = widgets_info
            self._overlay_controls_layout.addWidget(row_widget)

    def _on_overlay_lut_changed(
            self,
            overlay_key: str,
            lut_item: widgets.baseHistogramLUTitem,
        ) -> None:
        if self._syncing_overlay_from_main:
            return
        self.set_overlay_cmap(
            overlay_key,
            lut_item.gradient.colorMap(),
            sync_main_gui=True,
        )

    def _on_overlay_opacity_changed(
            self,
            overlay_key: str,
            value: float,
        ) -> None:
        if self._syncing_overlay_from_main:
            return
        self.set_overlay_opacity(overlay_key, value, sync_main_gui=True)
    
    def _on_lut_changed(self, lut_item: widgets.baseHistogramLUTitem) -> None:
        ticks = lut_item.gradient.listTicks()
        ticks_pos = [x for t, x in ticks]
        min_val = min(ticks_pos) if ticks_pos else 0.0
        max_val = max(ticks_pos) if ticks_pos else 1.0
        self.set_clim(min_val, max_val, lut_item.channel)
        self.set_cmap(lut_item)
        
    def _on_auto_clim(self, lut_item: widgets.baseHistogramLUTitem) -> None:
        lo, hi = self.get_auto_contrast_percentile(channel=lut_item.channel)
        low_tick = high_tick = None
        max_tick_val = -np.inf
        min_tick_val = np.inf
        for tick, x in lut_item.gradient.listTicks():
            if x > max_tick_val:
                high_tick = tick
                max_tick_val = x
            
            if x < min_tick_val:
                low_tick = tick
                min_tick_val = x
        
        if low_tick is not None and high_tick is not None:
            lut_item.gradient.setTickValue(low_tick, lo)
            lut_item.gradient.setTickValue(high_tick, hi)
            self.set_clim(lo, hi, lut_item.channel)

    def _on_full_clim(self, lut_item: widgets.baseHistogramLUTitem) -> None:
        lo, hi = 0.0, 1.0
        low_tick = high_tick = None
        max_tick_val = -np.inf
        min_tick_val = np.inf
        for tick, x in lut_item.gradient.listTicks():
            if x > max_tick_val:
                high_tick = tick
                max_tick_val = x
            
            if x < min_tick_val:
                low_tick = tick
                min_tick_val = x
        
        if low_tick is not None and high_tick is not None:
            lut_item.gradient.setTickValue(low_tick, lo)
            lut_item.gradient.setTickValue(high_tick, hi)
            self.set_clim(lo, hi, lut_item.channel)

    # -- Qt UI ----------------------------------------------------------------
    
    def _init_ui(self) -> None:
        if self._ui_initialised:
            return
        
        self.topToolBar = widgets.VolumeRendererToolbar(parent=self)
        self.addToolBar(Qt.TopToolBarArea, self.topToolBar)
        
        self.topToolBar.sigHomeView.connect(self.reset_view)
        self.topToolBar.sigSave.connect(self.save_screenshot)
        self.topToolBar.homeViewAction.setShortcutContext(Qt.WindowShortcut)
        
        controls_box = QGroupBox('Rendering Controls')
        self._controls = VolumeRendererControls(
            self, 
            parent=controls_box,
            channels=self.channels
        )
        box_layout = QVBoxLayout(controls_box)
        box_layout.setContentsMargins(4, 4, 4, 4)
        box_layout.addWidget(self._controls)
        
        self.scene_layout = QHBoxLayout()

        self._overlay_controls_host = QGroupBox('Overlays')
        self._overlay_controls_layout = QVBoxLayout(self._overlay_controls_host)
        self._overlay_controls_layout.setContentsMargins(4, 4, 4, 4)
        self._overlay_controls_host.hide()

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addLayout(self.scene_layout)
        main_layout.addWidget(self._overlay_controls_host)
        main_layout.addWidget(controls_box)
        self.setCentralWidget(central)
        
        self._ui_initialised = True

        # Restore settings saved in a previous session.
        # self._load_settings()

    # -- Settings persistence -------------------------------------------------

    # Class-level sentinels — always findable without triggering Qt's __getattr__
    # on incompletely-initialised test instances created with __new__.
    _axis_visual = None
    # Per-axis downsampling strides used in the last upload (z, y, x).
    # Stored so set_voxel_scale can correct for non-uniform stride compression.
    _last_strides: tuple = (1, 1, 1)
    # Last physical voxel sizes (µm) passed to set_voxel_scale.
    # Auto-reapplied when a new volume node is created in update_volume so
    # callers need not re-call set_voxel_scale after a shape change.
    _voxel_dz: float = 1.0
    _voxel_dy: float = 1.0
    _voxel_dx: float = 1.0

    _SETTINGS_ORG = 'Cell-ACDC'
    _SETTINGS_APP = 'renderer3d'

    def _load_settings(self) -> None:
        """Restore rendering settings from a previous session via QSettings."""
        from qtpy.QtCore import QSettings  # noqa: PLC0415
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        c = self._controls

        # Restore combobox indices safely (bounds-check against current count).
        mode_idx = s.value('mode_idx', 0, type=int)
        c._mode_combo.setCurrentIndex(min(mode_idx, c._mode_combo.count() - 1))
        interp_idx = s.value('interp_idx', 0, type=int)
        c._interp_combo.setCurrentIndex(
            min(interp_idx, c._interp_combo.count() - 1)
        )

        # Restore numeric spinboxes — clamp to widget range so stale values
        # from older versions don't break the UI.
        c._clim_min.setValue(
            max(c._clim_min.minimum(),
                min(s.value('clim_min', 0.0, type=float), c._clim_min.maximum()))
        )
        c._clim_max.setValue(
            max(c._clim_max.minimum(),
                min(s.value('clim_max', 1.0, type=float), c._clim_max.maximum()))
        )
        c._gamma_spin.setValue(
            max(c._gamma_spin.minimum(),
                min(s.value('gamma', 1.0, type=float), c._gamma_spin.maximum()))
        )
        c._step_spin.setValue(
            max(c._step_spin.minimum(),
                min(s.value('step_size', _DEFAULT_STEP_SIZE, type=float),
                    c._step_spin.maximum()))
        )
        c._smooth_iso_cb.setChecked(s.value('smooth_iso', False, type=bool))

        # Depiction and plane parameters.
        depict_idx = s.value('depict_idx', 0, type=int)
        c._depict_combo.setCurrentIndex(
            min(depict_idx, c._depict_combo.count() - 1)
        )
        c._plane_thick_spin.setValue(
            max(c._plane_thick_spin.minimum(),
                min(s.value('plane_thickness', 1.0, type=float),
                    c._plane_thick_spin.maximum()))
        )
        c._opacity_spin.setValue(
            max(c._opacity_spin.minimum(),
                min(s.value('opacity', 1.0, type=float),
                    c._opacity_spin.maximum()))
        )

    def _save_settings(self) -> None:
        """Persist current rendering settings so they survive app restarts."""
        from qtpy.QtCore import QSettings  # noqa: PLC0415
        s = QSettings(self._SETTINGS_ORG, self._SETTINGS_APP)
        c = self._controls
        s.setValue('mode_idx',       c._mode_combo.currentIndex())
        s.setValue('interp_idx',     c._interp_combo.currentIndex())
        s.setValue('clim_min',       c._clim_min.value())
        s.setValue('clim_max',       c._clim_max.value())
        s.setValue('gamma',          c._gamma_spin.value())
        s.setValue('step_size',      c._step_spin.value())
        s.setValue('smooth_iso',     c._smooth_iso_cb.isChecked())
        s.setValue('depict_idx',     c._depict_combo.currentIndex())
        s.setValue('plane_thickness', c._plane_thick_spin.value())
        s.setValue('opacity',         c._opacity_spin.value())

    # -- GPU helpers ----------------------------------------------------------

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

    def _apply_mode_cutoffs_to(self, node, mode: str, lo: float, hi: float) -> None:
        """Set mip_cutoff / minip_cutoff on *node* to make background transparent."""
        if node is None:
            return
        if mode in _MIP_CUTOFF_MODES:
            node.mip_cutoff = lo
        if mode in _MINIP_CUTOFF_MODES:
            node.minip_cutoff = hi

    def _apply_mode_cutoffs(self, mode: str, lo: float, hi: float) -> None:
        """Apply cutoffs to primary volume nodes (not overlays)."""
        if self._volume_nodes is None or not self.channels:
            return
        for channel in self.channels:
            volume_node = self._volume_nodes.get(channel)
            if volume_node is not None:
                self._apply_mode_cutoffs_to(volume_node, mode, lo, hi)

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

    def _preprocess_volume(
            self,
            volume: np.ndarray,
            channel: str | None = None,
        ):
        if volume.ndim != 3:
            raise ValueError(
                f'Expected 3-D (Z, Y, X) array; got shape {volume.shape}')
        
        # copy=False avoids a redundant allocation when data is already float32
        # (e.g. when _rerender calls update_volume(self._last_raw_data)).
        vol = volume.astype(np.float32, copy=False)
        
        # Cache raw float32 data so smooth-ISO toggle can re-process without
        # requiring a frame navigation in the host application.
        self._last_raw_data = vol
        if channel is not None:
            if self._raw_volumes_data is None:
                self._raw_volumes_data = {}
            self._raw_volumes_data[channel] = vol
        original_shape = vol.shape

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
        
        current_mode = self._controls._mode_combo.currentData() or 'mip'

        # Smooth ISO pre-filter: approximates napari's SMOOTH_GRADIENT_DEFINITION
        # (Sobel-Feldman 27-sample kernel) without requiring custom GLSL injection.
        # Applied after normalisation so the threshold remains in [0, 1].
        want_smooth = self._smooth_iso and current_mode in _ISO_MODES
        smoothed = False
        if want_smooth:
            try:
                import scipy.ndimage  # noqa: PLC0415
                vol = scipy.ndimage.gaussian_filter(vol, sigma=_SMOOTH_ISO_SIGMA)
                smoothed = True
            except ImportError:
                pass  # scipy not available — skip smoothing silently
        
        self._gpu_data_is_smoothed = smoothed
        
        return vol
    
    def _get_lut_item(self, channel_name: str):
        return self.lut_items[channel_name][0]
    
    def _init_volume_node(
            self, 
            volume: np.ndarray, 
            channel_name: str,
            update_canvas=False
        ):
        from vispy.scene import visuals
        
        current_mode = self._controls._mode_combo.currentData() or 'mip'        
        current_interp = self._controls._interp_combo.currentData() or 'linear'
        current_step = self._controls._step_spin.value()
        
        lut_item = self._get_lut_item(channel_name)
        
        pg_cmap = lut_item.gradient.colorMap()
        current_cmap = colors.pg_to_vispy_cmap(pg_cmap)
        ticks = lut_item.gradient.listTicks()
        ticks_pos = [x for t, x in ticks]
        min_val = min(ticks_pos) if ticks_pos else 0.0
        max_val = max(ticks_pos) if ticks_pos else 1.0
        clim = (min_val, max_val)
        
        volume_node = visuals.Volume(
            volume,
            clim=clim,
            method=current_mode,
            cmap=current_cmap,
            interpolation=current_interp,
            relative_step_size=current_step,
            parent=self._view.scene,
        )
        
        volume_node.gamma = self._controls._gamma_spin.value()
        volume_node.opacity = (
            self._controls._opacity_spins[channel_name].value()
        )
        volume_node.attenuation = 0.05
        
        if current_mode in _ATTN_MODES:
            volume_node.attenuation = self._controls._attn_spin.value()
        if current_mode in _ISO_MODES:
            volume_node.threshold = self._controls._iso_spin.value()
        
        self._apply_mode_cutoffs_to(
            volume_node, current_mode, clim[0], clim[1]
        )
        
        depict_mode = self._controls._depict_combo.currentData() or 'volume'
        
        if depict_mode in _PLANE_CONFIGS:
            volume_node.raycasting_mode = 'plane'
            # Pass vol.shape explicitly: _last_shape not yet updated here.
            self._set_plane_uniforms(
                depict_mode,
                self._controls._zplane_slider.value() / 100.0,
                shape=volume.shape,
                node=volume_node
            )
        
        self._apply_voxel_scale(volume_node)
        
        self._view.camera.set_range()
        
        from vispy.visuals.transforms import STTransform  # noqa: PLC0415
        axis_scale = max(2.0, max(volume.shape) * 0.10)
        self._axis_visual.transform = STTransform(
            scale=(axis_scale, axis_scale, axis_scale)
        )
        self._axis_visual.visible = True
        
        if update_canvas:
            self._canvas.update()
        
        return volume_node

    def _init_overlay_volume_node(
            self,
            volume: np.ndarray,
            channel_name: str,
            opacity: float,
            cmap_spec: str,
            mode_override: str | None,
        ):
        from vispy.scene import visuals  # noqa: PLC0415

        primary_mode = self._controls._mode_combo.currentData() or 'mip'
        node_mode = mode_override or primary_mode
        current_interp = self._controls._interp_combo.currentData() or 'linear'
        current_step = self._controls._step_spin.value()
        depict_mode = self._controls._depict_combo.currentData() or 'volume'
        is_plane = depict_mode in _PLANE_CONFIGS
        plane_fraction = self._controls._zplane_slider.value() / 100.0

        volume_node = visuals.Volume(
            volume,
            clim=(0.0, 1.0),
            method=node_mode,
            cmap=_volume_cmap_from_spec(cmap_spec),
            interpolation=current_interp,
            relative_step_size=current_step,
            parent=self._view.scene,
        )
        volume_node.opacity = max(0.0, min(1.0, opacity))
        volume_node.gamma = self._controls._gamma_spin.value()
        self._apply_mode_cutoffs_to(volume_node, node_mode, 0.0, 1.0)
        if node_mode in _ATTN_MODES:
            volume_node.attenuation = self._controls._attn_spin.value()
        if node_mode in _ISO_MODES:
            volume_node.threshold = self._controls._iso_spin.value()
        if is_plane:
            volume_node.raycasting_mode = 'plane'
            self._set_plane_uniforms(
                depict_mode,
                plane_fraction,
                shape=volume.shape,
                node=volume_node,
            )

        self._apply_voxel_scale(volume_node)
        self._overlay_mode_overrides[channel_name] = mode_override
        return volume_node
    
    def _get_clim(self, lut_item):
        ticks = lut_item.gradient.listTicks()
        ticks_pos = [x for t, x in ticks]
        min_val = min(ticks_pos) if ticks_pos else 0.0
        max_val = max(ticks_pos) if ticks_pos else 1.0
        clim = (min_val, max_val)
        return clim
    
    def _set_cmap(
            self, 
            cmap: str | list[colors.RgbaColor], 
            channel_name: str
        ):
        lut_item = self._get_lut_item(channel_name)
        if isinstance(cmap, str):
            lut_item.gradient.loadPreset(cmap)
        else:
            gradient = colors.get_pg_gradient(cmap)
            lut_item.setGradient(gradient)
    
    # -- Public API -----------------------------------------------------------

    def set_volume(
            self, 
            volume: np.ndarray,
            channel_name: None | str=None,
            cmap: str | list[colors.RgbaColor] | None=None,
        ):
        if self.channels is not None and channel_name is None:
            raise ValueError(
                'When setting up multiple volumes `channel_name` needs to '
                'be specified.'
            )
        
        channel_names = None
        if channel_name is not None:
            channel_names = [channel_name]
        
        self.set_volumes([volume], channel_names)
        
        if cmap is None:
            return
        
        if channel_name is None:
            channel_name = self.channels[0]
        
        self._set_cmap(cmap, channel_name)
        
    def set_volumes(
            self, 
            volumes: dict[str, np.ndarray] | list[np.ndarray],
            channel_names: None | list[str]=None,
            cmaps: None | dict[str, list[colors.RgbaColor]] | list[list[colors.RgbaColor]]=None,
        ):
        if self._volumes_data is None:
            self._volumes_data = {}
        elif channel_names is not None:
            channel_names = [*self.channels, *channel_names]
        
        num_volumes = len(self._volumes_data) + len(volumes)
        
        if not isinstance(volumes, dict):
            if channel_names is None:
                keys = [
                    f'Channel {ch_idx+1}' for ch_idx in range(num_volumes)
                ]
                
            volumes = dict(zip(keys, volumes))
            channel_names = keys
        
        if cmaps is not None and not isinstance(cmaps, dict):
            cmaps = dict(zip(channel_names, volumes))
        
        self.channels = list(volumes.keys())
        self._init_ui()
        
        if self._volume_nodes is None:
            self._volume_nodes = {}
        
        if self.lut_items is None:
            self._add_lut_items(self.scene_layout)
            self.scene_layout.addWidget(self._canvas.native, stretch=1)
            self._add_opacity_lut_items(self.scene_layout)
            self._connect_canvas_events()
        
        if cmaps is not None:
            for channel_name, cmap in cmaps.items():
                self._set_cmap(cmap, channel_name)
        
        for channel, volume in volumes.items():
            vol = self._preprocess_volume(volume, channel=channel)
            self._volumes_data[channel] = vol
            
            vol_node = self._init_volume_node(
                vol, channel, update_canvas=False
            )
            self._volume_nodes[channel] = vol_node
            self._last_shape = vol.shape
        
        if self._is_set_volumes_first_call:
            # Enable blending
            from vispy import gloo
            gloo.set_state(blend=True, depth_test=False)
            gloo.set_blend_func('one', 'one')
            self._is_set_volumes_first_call = False
        
        self._canvas.update()
    
    def update_volume(
            self, 
            data: np.ndarray, 
            channel_name: None | str=None,
            channel_index: None | int=None
        ) -> None:
        """
        Replace the displayed volume with *data*.

        Parameters
        ----------
        data:
            A (Z, Y, X) array of any numeric dtype.  Values are normalised to
            float32 [0, 1] before upload; existing contrast limits are preserved.
            Data is automatically downsampled if any dimension exceeds the GPU's
            maximum 3-D texture size.
        """
        vol = self._preprocess_volume(data, channel=channel_name)

        if self._volumes_data is None or self.channels is None:
            name = channel_name or 'Channel 1'
            self.set_volume(data, channel_name=name)
            return

        if channel_name is None and channel_index is None:
            channel_name = self.channels[0]
        
        if channel_index is not None and channel_index >= len(self.channels):
            self.channels.append(f'Channel {channel_index+1}')
            channel_name = self.channels[-1]
        elif channel_index is not None and channel_index < len(self.channels):
            channel_name = self.channels[channel_index]
        
        if channel_name not in self._volumes_data.keys():
            self.set_volume(data, channel_name)
        
        current_mode = self._controls._mode_combo.currentData() or 'mip'
        volume_node = self._volume_nodes[channel_name]
        
        lut_item = self._get_lut_item(channel_name)
        
        clim = self._get_clim(lut_item)
        lo, hi = clim
        
        volume_node.set_data(vol, clim=clim)
        self._apply_mode_cutoffs_to(volume_node, current_mode, lo, hi)

        self._last_shape = vol.shape

        self._canvas.update()

    def update_overlay_volumes(
            self,
            overlays: list[tuple],
            preserve_widgets: bool = False,
        ) -> None:
        """Replace overlay volumes stored in ``_volume_nodes`` under ``overlay:N``."""
        if self._controls is None:
            return

        self._remove_overlay_channels(clear_widgets=not preserve_widgets)

        if not overlays:
            if hasattr(self, '_canvas') and self._canvas is not None:
                self._canvas.update()
            return

        nodes = self._ensure_volume_nodes()
        if self._volumes_data is None:
            self._volumes_data = {}

        overlay_entries = []
        for index, entry in enumerate(overlays):
            data, opacity, cmap_spec, mode_override, meta = _parse_overlay_entry(
                entry, index
            )
            if data.ndim != 3:
                continue
            channel_name = overlay_channel_name(index)
            if 'label_data' in meta:
                self._store_label_volume(channel_name, meta['label_data'])
                data = _mask_labels_for_display(
                    self._label_volumes[channel_name],
                    self._active_cell_id,
                )
            vol = self._normalize_overlay_volume(data)
            self._volumes_data[channel_name] = vol
            meta.setdefault('overlay_index', index)
            meta.setdefault('channel_name', channel_name)
            self._overlay_meta[channel_name] = meta
            nodes[channel_name] = self._init_overlay_volume_node(
                vol,
                channel_name,
                opacity,
                cmap_spec,
                mode_override,
            )
            overlay_entries.append(
                (channel_name, opacity, cmap_spec, mode_override, meta)
            )

        if not preserve_widgets:
            self._rebuild_overlay_controls(overlay_entries)

        self._canvas.update()

    def _overlay_key_from_index(self, index_or_key) -> str | None:
        if isinstance(index_or_key, str):
            if index_or_key in (self._volume_nodes or {}):
                return index_or_key
            if index_or_key.isdigit():
                index_or_key = int(index_or_key)
        if isinstance(index_or_key, int):
            key = overlay_channel_name(index_or_key)
            if self._volume_nodes and key in self._volume_nodes:
                return key
        return None

    def _find_overlay_by_kind(self, kind: str) -> str | None:
        for key, meta in self._overlay_meta.items():
            if meta.get('kind') == kind:
                return key
        return None

    def set_overlay_opacity(
            self,
            index_or_key,
            value: float,
            *,
            sync_main_gui: bool = True,
        ) -> None:
        key = self._overlay_key_from_index(index_or_key)
        if key is None or self._volume_nodes is None:
            return
        volume_node = self._volume_nodes.get(key)
        if volume_node is None:
            return
        opacity = max(0.0, min(1.0, float(value)))
        volume_node.opacity = opacity
        widgets_info = self._overlay_widgets.get(key)
        if widgets_info is not None:
            opacity_spin = widgets_info.get('opacity_spin')
            if opacity_spin is not None and not self._syncing_overlay_from_main:
                self._syncing_overlay_from_renderer = True
                try:
                    opacity_spin.blockSignals(True)
                    opacity_spin.setValue(opacity)
                    opacity_spin.blockSignals(False)
                finally:
                    self._syncing_overlay_from_renderer = False
        if sync_main_gui and not self._syncing_overlay_from_main:
            meta = self._overlay_meta.get(key, {})
            adapter = self._adapter
            if adapter is not None and hasattr(
                adapter, 'apply_overlay_control_from_renderer'
            ):
                kind = meta.get('kind')
                channel_name = meta.get('channel_name', '')
                if kind == OVERLAY_KIND_SEGM:
                    adapter.apply_overlay_control_from_renderer(
                        channel_name, labels_alpha=opacity
                    )
                else:
                    adapter.apply_overlay_control_from_renderer(
                        channel_name, opacity=opacity
                    )
        self._canvas.update()

    def set_overlay_cmap(
            self,
            index_or_key,
            cmap_spec,
            *,
            sync_main_gui: bool = True,
        ) -> None:
        key = self._overlay_key_from_index(index_or_key)
        if key is None or self._volume_nodes is None:
            return
        volume_node = self._volume_nodes.get(key)
        if volume_node is None:
            return
        volume_node.cmap = _volume_cmap_from_spec(cmap_spec)
        if sync_main_gui and not self._syncing_overlay_from_main:
            meta = self._overlay_meta.get(key, {})
            adapter = self._adapter
            if adapter is not None and hasattr(
                adapter, 'apply_overlay_control_from_renderer'
            ):
                kind = meta.get('kind')
                if kind == OVERLAY_KIND_FLUO:
                    gradient_state = None
                    widgets_info = self._overlay_widgets.get(key)
                    lut_item = widgets_info.get('lut_item') if widgets_info else None
                    if lut_item is not None:
                        gradient_state = lut_item.gradient.saveState()
                    adapter.apply_overlay_control_from_renderer(
                        meta.get('channel_name', ''),
                        gradient_state=gradient_state,
                    )
        self._canvas.update()

    def set_segm_overlay_opacity(
            self,
            value: float,
            *,
            sync_main_gui: bool = True,
        ) -> None:
        key = self._find_overlay_by_kind(OVERLAY_KIND_SEGM)
        if key is None:
            return
        self.set_overlay_opacity(
            key, value, sync_main_gui=sync_main_gui
        )

    def set_rendering_mode(self, mode: str) -> None:
        if not self._has_volume_nodes():
            return

        if mode in _ISO_MODES and (self._smooth_iso != self._gpu_data_is_smoothed):
            for channel in self.channels or []:
                volume_node = self._volume_nodes.get(channel)
                if volume_node is not None:
                    volume_node.method = mode
                    self._rerender()
                    return

        for channel, volume_node in self._volume_nodes.items():
            if is_overlay_channel(channel):
                if self._overlay_mode_overrides.get(channel) is not None:
                    continue
                volume_node.method = mode
                self._apply_mode_cutoffs_to(volume_node, mode, 0.0, 1.0)
                if mode in _ISO_MODES:
                    volume_node.threshold = self._controls._iso_spin.value()
                if mode in _ATTN_MODES:
                    volume_node.attenuation = self._controls._attn_spin.value()
                continue

            lut_item = self._get_lut_item(channel)
            volume_node.method = mode
            ticks_pos = [x for t, x in lut_item.gradient.listTicks()]
            lo = min(ticks_pos) if ticks_pos else 0.0
            hi = max(ticks_pos) if ticks_pos else 1.0
            self._apply_mode_cutoffs_to(volume_node, mode, lo, hi)
            if mode in _ISO_MODES:
                volume_node.threshold = self._controls._iso_spin.value()
            if mode in _ATTN_MODES:
                volume_node.attenuation = self._controls._attn_spin.value()

        self._canvas.update()

    def set_clim(self, lo: float, hi: float, channel: str) -> None:
        volume_node = self._volume_nodes.get(channel, None)
        if volume_node is None:
            return
        
        volume_node.clim = (lo, hi)
        current_mode = self._controls._mode_combo.currentData() or 'mip'
        self._apply_mode_cutoffs_to(volume_node, current_mode, lo, hi)
        self._canvas.update()
    
    def set_cmap(self, lut_item: widgets.baseHistogramLUTitem):
        cmap = colors.pg_to_vispy_cmap(lut_item.gradient.colorMap())
        channel = lut_item.channel
        volume_node = self._volume_nodes.get(channel, None)
        if volume_node is None:
            return
        
        volume_node.cmap = cmap
        self._canvas.update()

    def get_auto_contrast_percentile(
            self,
            lo_pct: float = 1.0,
            hi_pct: float = 99.5,
            channel: str | None = None,
        ) -> tuple[float, float]:
        """
        Set contrast limits to the *lo_pct*–*hi_pct* percentile of the raw
        volume data, clipped to [0, 1] in the normalised space.

        This is more useful for fluorescence microscopy than a full-range reset
        because bright artefact voxels are excluded without manual adjustment.
        Mirrors the spirit of napari's auto-contrast which maps the meaningful
        intensity range rather than the absolute min–max.

        Falls back to [0, 1] when no volume has been loaded yet.
        """
        raw = None
        if (
            channel is not None
            and self._raw_volumes_data is not None
            and channel in self._raw_volumes_data
        ):
            raw = self._raw_volumes_data[channel]
        elif self._last_raw_data is not None:
            raw = self._last_raw_data

        if raw is None:
            lo, hi = 0.0, 1.0
        else:
            vmin_raw = float(raw.min())
            vmax_raw = float(raw.max())
            if vmax_raw <= vmin_raw:
                lo, hi = 0.0, 1.0
            else:
                # Subsample large volumes for speed (< 1 M samples is fast).
                flat = raw.ravel()
                if flat.size > 1_000_000:
                    step = flat.size // 1_000_000 + 1
                    flat = flat[::step]
                p_lo = float(np.percentile(flat, lo_pct))
                p_hi = float(np.percentile(flat, hi_pct))
                span = vmax_raw - vmin_raw
                lo = max(0.0, min(1.0, (p_lo - vmin_raw) / span))
                hi = max(0.0, min(1.0, (p_hi - vmin_raw) / span))
                if hi <= lo:
                    lo, hi = 0.0, 1.0

        return lo, hi

    def auto_contrast_percentile(
            self,
            lo_pct: float = 1.0,
            hi_pct: float = 99.5,
            channel: str | None = None,
        ) -> tuple[float, float]:
        """Backwards-compatible alias for :meth:`get_auto_contrast_percentile`."""
        return self.get_auto_contrast_percentile(
            lo_pct=lo_pct, hi_pct=hi_pct, channel=channel
        )

    def set_gamma(self, value: float) -> None:
        if not self._has_volume_nodes():
            return
        for volume_node in self._each_volume_node():
            volume_node.gamma = value
        self._canvas.update()

    def set_opacity(self, value: float, channel: str | None = None) -> None:
        """
        Set the overall volume opacity (0 = fully transparent, 1 = opaque).

        Mirrors napari's ``layer.opacity → node.opacity`` pathway.  The effect
        depends on the rendering mode: most visible in translucent and additive
        modes; has no visual effect in MIP/MinIP (which project to a 2D plane).
        """
        if self._volume_nodes is None:
            return
        volume_node = self._volume_nodes.get(channel)
        if volume_node is None:
            return
        
        volume_node.opacity = max(0.0, min(1.0, value))
        if channel is not None:
            self._sync_opacity_lut_value(channel, volume_node.opacity)
        self._canvas.update()

    def set_iso_threshold(self, value: float) -> None:
        if self._volume_nodes is None:
            return
        for volume_node in self._volume_nodes.values():
            volume_node.threshold = value
            
        self._canvas.update()

    def set_attenuation(self, value: float) -> None:
        if self._volume_nodes is None:
            return
        for volume_node in self._volume_nodes.values():
            volume_node.attenuation = value
            
        self._canvas.update()

    def set_interpolation(self, method: str) -> None:
        """Set 3D volume interpolation method (e.g. 'linear', 'nearest', 'catrom')."""
        if self._volume_nodes is None:
            return
        for volume_node in self._volume_nodes.values():
            volume_node.interpolation = method
            
        self._canvas.update()

    def set_depiction(self, mode: str) -> None:
        """
        Switch between full volume raycasting and a planar cross-section.

        Mirrors napari's ``layer.depiction`` which calls
        ``node.raycasting_mode = str(layer.depiction)``.

        Parameters
        ----------
        mode : {'volume', 'plane_z', 'plane_y', 'plane_x'}
            'volume'  — full 3D raycasting.
            'plane_z' — XY cross-section (normal along Z).
            'plane_y' — XZ cross-section (normal along Y).
            'plane_x' — YZ cross-section (normal along X).
        """
        is_plane = mode in _PLANE_CONFIGS
        if self._volume_nodes is not None:
            for channel, volume_node in self._volume_nodes.items():
                volume_node.raycasting_mode = 'plane' if is_plane else 'volume'
                if is_plane:
                    shape = self._volume_shape(channel)
                    if shape is not None:
                        self._set_plane_uniforms(
                            mode, 0.5, shape=shape, node=volume_node
                        )
        if hasattr(self, '_canvas') and self._canvas is not None:
            self._canvas.update()

    def set_zplane_position(self, fraction: float) -> None:
        """
        Move the active cross-section plane to *fraction* ∈ [0, 1] of its axis.

        The axis is determined by the currently selected depiction mode.
        """
        if self._last_shape is None and not self._has_volume_nodes():
            return
        current_mode = self._controls._depict_combo.currentData() or 'volume'
        if current_mode not in _PLANE_CONFIGS:
            return
        if self._volume_nodes is not None:
            for channel, volume_node in self._volume_nodes.items():
                shape = self._volume_shape(channel)
                if shape is None:
                    continue
                self._set_plane_uniforms(
                    current_mode, fraction, shape=shape, node=volume_node
                )
        self._canvas.update()

    def _set_plane_uniforms(
        self,
        mode: str,
        fraction: float,
        shape: tuple[int, int, int] | None = None,
        node=None,
    ) -> None:
        """Set plane_position/normal/thickness for the given plane mode.

        Parameters
        ----------
        shape:
            (NZ, NY, NX) to use for positioning.  Defaults to
            ``self._last_shape``.  Must be supplied when called from
            ``update_volume`` before ``_last_shape`` is updated.
        node:
            Volume node to update.  When omitted, the caller must iterate nodes.
        """
        if node is None:
            return
        normal, axis = _PLANE_CONFIGS[mode]
        if shape is None:
            shape = self._last_shape
        if shape is None:
            return  # no data loaded yet — cannot compute plane position
        nz, ny, nx = shape

        # Scene-space centre for the two axes orthogonal to the slice axis.
        centers = [nx / 2 - 0.5, ny / 2 - 0.5, nz / 2 - 0.5]

        # Move the plane along its axis: scene coords match data-shape order
        # (scene-x=data-X, scene-y=data-Y, scene-z=data-Z).
        # Voxel centres are at integer positions 0, 1, …, n-1 in scene space.
        # vispy's shader maps: texture_coord = (scene_pos + 0.5) / shape
        # so scene_pos = fraction*(n-1) gives texture_coord = (n*fraction)/n
        # = fraction → fraction=0.5 lands exactly at the volume centre.
        axis_scene = {0: 2, 1: 1, 2: 0}[axis]  # data axis → scene axis index
        n_along = shape[axis]
        centers[axis_scene] = fraction * (n_along - 1)

        thickness = (
            self._controls._plane_thick_spin.value()
            if self._controls is not None else 1.0
        )
        node.plane_normal = normal
        node.plane_position = centers
        node.plane_thickness = thickness

    def set_plane_thickness(self, thickness: float) -> None:
        """
        Set the cross-section plane thickness in voxels.

        Mirrors napari's ``layer.plane.thickness`` /
        ``_on_plane_thickness_change``.  Thickness ≥ 1 (one voxel).
        Larger values produce a thicker slab that shows more context.
        """
        t = max(1.0, thickness)
        if not self._has_volume_nodes():
            return
        for volume_node in self._each_volume_node():
            volume_node.plane_thickness = t
        self._canvas.update()

    def _rerender(self) -> None:
        """Re-process and re-upload the last received volume (smooth ISO toggle)."""
        if self._last_raw_data is not None:
            self.update_volume(self._last_raw_data)

    def set_smooth_iso(self, enabled: bool) -> None:
        """
        Toggle Gaussian pre-smoothing for ISO surface rendering.

        When *enabled*, ``scipy.ndimage.gaussian_filter(vol, sigma=1)`` is applied
        to the normalised volume before GPU upload whenever the rendering mode is
        ``iso``.  This approximates napari's SMOOTH_GRADIENT_DEFINITION (Sobel-
        Feldman 27-sample kernel) without requiring custom GLSL shader injection.
        The threshold control remains meaningful — smoothing is applied after
        normalisation, so values are still in [0, 1].

        The change takes effect immediately by re-uploading the cached volume data.
        """
        self._smooth_iso = enabled
        current_mode = self._controls._mode_combo.currentData() or 'mip'
        if current_mode in _ISO_MODES:
            self._rerender()

    def set_step_size(self, value: float) -> None:
        """
        Set the ray-marching relative step size (napari: relative_step_size).

        Smaller values cast more rays per voxel → sharper but slower.
        vispy default and napari default: 0.8.  Range: (0, 2].
        """
        if not self._has_volume_nodes():
            return
        for volume_node in self._each_volume_node():
            volume_node.relative_step_size = value
        if hasattr(self, '_canvas') and self._canvas is not None:
            self._canvas.update()

    def set_voxel_scale(
        self,
        dz: float = 1.0,
        dy: float = 1.0,
        dx: float = 1.0,
    ) -> None:
        """
        Correct for anisotropic voxel sizes by scaling the volume transform.

        Parameters
        ----------
        dz, dy, dx:
            Physical voxel size in µm along Z, Y, X.  All three are normalised
            to *dx* so the rendered volume has correct physical proportions.

        The scale also accounts for per-axis downsampling strides (stored in
        ``_last_strides`` from the last ``update_volume`` call).  When the GPU
        texture limit forces non-uniform downsampling (e.g. stride_x=4 while
        stride_z=1), each downsampled voxel along X spans ``stride_x × dx``
        physical µm.  Ignoring this would compress the X axis by 4×.

        Example — 100×512×2048 confocal stack on a 512-texture GPU:
          Physical voxels: dz=1 µm, dy=0.2 µm, dx=0.2 µm
          Downsampling:    stride=(1, 1, 4) → shape (100, 512, 512)
          Effective sizes: dz_eff=1, dy_eff=0.2, dx_eff=0.8 µm
          Scale:           (1.0, 0.2/0.8, 1.0/0.8) = (1, 0.25, 1.25)
        """
        # Persist so the transform is re-applied automatically when the node is
        # rebuilt due to a shape change (update_volume calls _apply_voxel_scale).
        self._voxel_dz, self._voxel_dy, self._voxel_dx = dz, dy, dx
        if not self._has_volume_nodes():
            return
        self._apply_voxel_scale()

    def _apply_voxel_scale(self, node=None) -> None:
        """Apply the stored voxel scale and stride correction to volume nodes."""
        from vispy.visuals.transforms import STTransform  # noqa: PLC0415
        sz, sy, sx = self._last_strides
        dx_eff = self._voxel_dx * sx
        dy_eff = self._voxel_dy * sy
        dz_eff = self._voxel_dz * sz
        if dx_eff <= 0:
            dx_eff = 1.0
        transform = STTransform(scale=(1.0, dy_eff / dx_eff, dz_eff / dx_eff))
        if node is not None:
            node.transform = transform
        else:
            for volume_node in self._each_volume_node():
                volume_node.transform = transform
        if hasattr(self, '_canvas') and self._canvas is not None:
            self._canvas.update()

    def reset_view(self) -> None:
        """Reset the camera to the default orientation and fit the volume."""
        self._view.camera.set_range()
        self._view.camera.elevation = 30.0
        self._view.camera.azimuth = -60.0
        self._canvas.update()

    def save_screenshot(self) -> None:
        """Save the current 3D view as a PNG file via a file-save dialog."""
        from qtpy.QtWidgets import QFileDialog  # noqa: PLC0415

        path, _ = QFileDialog.getSaveFileName(
            self,
            'Save 3D View',
            'renderer3d_snapshot.png',
            'PNG image (*.png);;TIFF image (*.tif *.tiff)',
        )
        if not path:
            return

        # canvas.render() returns (H, W, 4) uint8 RGBA
        frame = self._canvas.render(alpha=True)

        try:
            import skimage.io as skio  # noqa: PLC0415
            skio.imsave(path, frame, check_contrast=False)
        except ImportError:
            # stdlib PNG fallback — TIFF requires skimage; normalise extension.
            if not path.lower().endswith('.png'):
                path = path.rsplit('.', 1)[0] + '.png'
            _write_png(path, frame)

        self.statusBar().showMessage(f'Saved → {path}', 4000)

    # -- Qt overrides ---------------------------------------------------------

    def closeEvent(self, event):
        # self._save_settings()
        if self._hide_on_close:
            event.ignore()
            self.hide()
            if self._adapter is not None:
                self._adapter.on_renderer_closed()
        else:
            super().closeEvent(event)
    
    def run(self) -> None:
        """Start the Qt event loop (if not already running)."""
        super().show()
        self.splashScreen.close()
        self.app.exec_()


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------
def create_renderer(
        parent: QWidget | None = None,
        hide_on_close: bool = True,
        adapter: VolumeRendererAdapter | None = None,
    ) -> VolumeRenderer3DWindow:
    """Return a new :class:`VolumeRenderer3DWindow` instance."""
    return VolumeRenderer3DWindow(
        parent=parent,
        hide_on_close=hide_on_close,
        adapter=adapter,
    )
