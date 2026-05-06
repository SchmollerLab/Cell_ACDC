"""
Standalone 3D volume renderer for z-stack microscopy data.

This module is self-contained and only depends on: numpy, qtpy, vispy.
It is designed to be used independently or integrated into other applications
via a thin adapter — see the Cell_ACDC adapter in gui.py for an example.

Volume rendering follows the napari/vispy approach:

Rendering
  - GPU-accelerated raycasting via ``vispy.scene.visuals.Volume``
  - 7 rendering modes: MIP, MinIP, Attenuated MIP, ISO, Translucent, Additive, Average
  - Smooth ISO pre-filter (Gaussian σ=1, approximates napari's Sobel-Feldman shader)
  - Plane cross-sections: XY / XZ / YZ (napari ``layer.depiction``)
  - Anisotropic voxel-scale correction via ``STTransform``
  - GPU-aware stride downsampling; vmin/vmax computed on full-resolution data

Display
  - TurntableCamera (elevation=30°, azimuth=−60°) — left-drag orbit, scroll zoom
  - XYZ axis indicator at the volume corner (red=X, green=Y, blue=Z)
  - Opacity, gamma, colormap, interpolation, ray-marching step size
  - Percentile auto-contrast (1st–99.5th) and full-range [0, 1] reset
  - Keyboard shortcuts: Ctrl+R reset view · Ctrl+A auto contrast · Ctrl+S screenshot
  - QSettings persistence across sessions

Standalone usage (no Cell_ACDC required)::

    from qtpy.QtWidgets import QApplication
    from cellacdc.renderer3d import create_renderer
    import numpy as np
    import sys

    app = QApplication(sys.argv)
    zstack = np.load('my_zstack.npy')   # shape (Z, Y, X)

    renderer = create_renderer()
    renderer.update_volume(zstack)
    renderer.set_voxel_scale(dz=0.5, dy=0.2, dx=0.2)  # µm/pixel
    renderer.auto_contrast_percentile()                  # brighten dim signals
    renderer.show()

    sys.exit(app.exec())

To integrate into a host application, subclass :class:`VolumeRendererAdapter`
and pass the instance to :func:`create_renderer`::

    class MyAdapter(VolumeRendererAdapter):
        def get_current_zstack(self):
            return my_app.current_volume()   # (Z, Y, X) array
        def get_voxel_sizes(self):
            return my_app.voxel_sizes_um()   # (dz, dy, dx)
        def on_renderer_closed(self):
            my_app.on_3d_closed()

    renderer = create_renderer(adapter=MyAdapter())
"""

from __future__ import annotations

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
    QShortcut,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence

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


# Plain colour names that _make_cmap converts to black→colour two-stop maps.
# Anything else is forwarded directly to vispy as a standard colormap name.
_PLAIN_COLOURS = frozenset({
    'red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'white', 'orange',
})


def _make_cmap(spec: str):
    """Return a vispy colormap for *spec*.

    Plain colour names ('red', 'green', …) produce a black→colour ramp so
    the channel appears in that hue against the black renderer background.
    Anything else is passed through as a standard vispy colormap name.
    """
    if spec in _PLAIN_COLOURS:
        from vispy.color import Colormap  # noqa: PLC0415
        return Colormap(['black', spec])
    return spec


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


# ---------------------------------------------------------------------------
# Controls widget
# ---------------------------------------------------------------------------
class VolumeRendererControls(QWidget):
    """Parameter panel that drives a VolumeRenderer3DWindow."""

    def __init__(self, renderer: 'VolumeRenderer3DWindow', parent: QWidget | None = None):
        super().__init__(parent)
        self._renderer = renderer
        self._build()

    def _build(self) -> None:
        # Two-row layout: top = rendering parameters, bottom = display parameters.
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 2, 4, 2)
        outer.setSpacing(2)

        row1 = QHBoxLayout()
        row1.setSpacing(6)
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        outer.addLayout(row1)
        outer.addLayout(row2)

        # ── Row 1: Rendering parameters ──────────────────────────────────────

        # Rendering mode
        row1.addWidget(QLabel('Render:'))
        self._mode_combo = QComboBox()
        for mode_id, label in RENDERING_MODES:
            self._mode_combo.addItem(label, mode_id)
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        row1.addWidget(self._mode_combo)

        # Contrast limits
        row1.addWidget(QLabel('Clim:'))
        self._clim_min = QDoubleSpinBox()
        self._clim_min.setRange(0.0, 1.0)
        self._clim_min.setSingleStep(0.01)
        self._clim_min.setDecimals(3)
        self._clim_min.setValue(0.0)
        self._clim_min.setFixedWidth(65)
        self._clim_min.setToolTip('Lower contrast limit (0–1 normalised)')
        self._clim_min.valueChanged.connect(self._on_clim_changed)
        row1.addWidget(self._clim_min)

        row1.addWidget(QLabel('–'))

        self._clim_max = QDoubleSpinBox()
        self._clim_max.setRange(0.0, 1.0)
        self._clim_max.setSingleStep(0.01)
        self._clim_max.setDecimals(3)
        self._clim_max.setValue(1.0)
        self._clim_max.setFixedWidth(65)
        self._clim_max.setToolTip('Upper contrast limit (0–1 normalised)')
        self._clim_max.valueChanged.connect(self._on_clim_changed)
        row1.addWidget(self._clim_max)

        auto_btn = QPushButton('Auto')
        auto_btn.setFixedWidth(42)
        auto_btn.setToolTip(
            'Auto contrast: set limits to the 1st–99.5th percentile\n'
            'of the raw volume data (excludes outlier voxels).\n'
            'Useful for fluorescence data with bright artefacts.\n'
            'Shortcut: Ctrl+A'
        )
        auto_btn.clicked.connect(self._on_auto_clim)
        row1.addWidget(auto_btn)

        reset_clim_btn = QPushButton('Full')
        reset_clim_btn.setFixedWidth(36)
        reset_clim_btn.setToolTip('Reset contrast limits to full range [0, 1]')
        reset_clim_btn.clicked.connect(self._on_reset_clim)
        row1.addWidget(reset_clim_btn)

        # Gamma
        row1.addWidget(QLabel('γ:'))
        self._gamma_spin = QDoubleSpinBox()
        self._gamma_spin.setRange(0.1, 5.0)
        self._gamma_spin.setSingleStep(0.1)
        self._gamma_spin.setValue(1.0)
        self._gamma_spin.setFixedWidth(55)
        self._gamma_spin.setToolTip('Gamma correction')
        self._gamma_spin.valueChanged.connect(self._on_gamma_changed)
        row1.addWidget(self._gamma_spin)

        # ISO threshold + smooth option (iso mode only)
        self._iso_label = QLabel('ISO:')
        row1.addWidget(self._iso_label)
        self._iso_spin = QDoubleSpinBox()
        self._iso_spin.setRange(0.0, 1.0)
        self._iso_spin.setSingleStep(0.01)
        self._iso_spin.setValue(0.5)
        self._iso_spin.setDecimals(3)
        self._iso_spin.setFixedWidth(65)
        self._iso_spin.setToolTip('Isosurface threshold')
        self._iso_spin.valueChanged.connect(self._on_iso_changed)
        row1.addWidget(self._iso_spin)

        self._smooth_iso_cb = QCheckBox('Smooth')
        self._smooth_iso_cb.setToolTip(
            'Pre-smooth the volume with a Gaussian filter (σ=1) before ISO\n'
            'rendering. Approximates napari\'s SMOOTH_GRADIENT_DEFINITION\n'
            'Sobel-Feldman shader — produces softer surface normals without\n'
            'requiring custom GLSL injection.'
        )
        self._smooth_iso_cb.toggled.connect(self._on_smooth_iso_changed)
        row1.addWidget(self._smooth_iso_cb)

        # Attenuation (attenuated_mip mode only)
        self._attn_label = QLabel('Attn:')
        row1.addWidget(self._attn_label)
        self._attn_spin = QDoubleSpinBox()
        self._attn_spin.setRange(0.0, 2.0)
        self._attn_spin.setSingleStep(0.05)
        self._attn_spin.setValue(0.5)
        self._attn_spin.setDecimals(2)
        self._attn_spin.setFixedWidth(60)
        self._attn_spin.setToolTip('Attenuation factor for Attenuated MIP')
        self._attn_spin.valueChanged.connect(self._on_attn_changed)
        row1.addWidget(self._attn_spin)

        row1.addStretch()

        # ── Row 2: Display / camera parameters ───────────────────────────────

        # Interpolation (napari: interpolation3d)
        row2.addWidget(QLabel('Interp:'))
        self._interp_combo = QComboBox()
        for iid, ilabel in INTERPOLATION_MODES:
            self._interp_combo.addItem(ilabel, iid)
        self._interp_combo.setToolTip(
            'Volume texture interpolation (Linear = smooth, Nearest = pixelated)'
        )
        self._interp_combo.currentIndexChanged.connect(self._on_interp_changed)
        row2.addWidget(self._interp_combo)

        # Ray-marching step size (napari: relative_step_size)
        row2.addWidget(QLabel('Step:'))
        self._step_spin = QDoubleSpinBox()
        self._step_spin.setRange(0.1, 2.0)
        self._step_spin.setSingleStep(0.1)
        self._step_spin.setValue(_DEFAULT_STEP_SIZE)
        self._step_spin.setDecimals(2)
        self._step_spin.setFixedWidth(55)
        self._step_spin.setToolTip(
            'Ray-marching step size relative to voxel size.\n'
            'Smaller = sharper rendering; larger = faster.'
        )
        self._step_spin.valueChanged.connect(self._on_step_changed)
        row2.addWidget(self._step_spin)

        # Depiction (napari: layer.depiction — volume vs plane)
        row2.addWidget(QLabel('Depict:'))
        self._depict_combo = QComboBox()
        for did, dlabel in DEPICTION_MODES:
            self._depict_combo.addItem(dlabel, did)
        self._depict_combo.setToolTip(
            'Volume: full 3D raycasting.  '
            'Z-Plane: single cross-section embedded in 3D space.'
        )
        self._depict_combo.currentIndexChanged.connect(self._on_depiction_changed)
        row2.addWidget(self._depict_combo)

        self._zplane_label = QLabel('Pos:')
        row2.addWidget(self._zplane_label)
        self._zplane_slider = QSlider(Qt.Horizontal)
        self._zplane_slider.setMinimum(0)
        self._zplane_slider.setMaximum(99)
        self._zplane_slider.setValue(50)
        self._zplane_slider.setFixedWidth(80)
        self._zplane_slider.setToolTip('Position of the cross-section plane (0=start, 100=end)')
        self._zplane_slider.valueChanged.connect(self._on_zplane_changed)
        row2.addWidget(self._zplane_slider)

        # Plane thickness (napari: layer.plane.thickness / _on_plane_thickness_change)
        self._plane_thick_label = QLabel('Thick:')
        row2.addWidget(self._plane_thick_label)
        self._plane_thick_spin = QDoubleSpinBox()
        self._plane_thick_spin.setRange(1.0, 50.0)
        self._plane_thick_spin.setSingleStep(1.0)
        self._plane_thick_spin.setValue(1.0)
        self._plane_thick_spin.setDecimals(1)
        self._plane_thick_spin.setFixedWidth(55)
        self._plane_thick_spin.setToolTip(
            'Thickness of the plane cross-section in voxels.\n'
            'Mirrors napari\'s plane.thickness parameter.'
        )
        self._plane_thick_spin.valueChanged.connect(self._on_plane_thickness_changed)
        row2.addWidget(self._plane_thick_spin)

        # Initially hide plane controls
        self._zplane_label.setVisible(False)
        self._zplane_slider.setVisible(False)
        self._plane_thick_label.setVisible(False)
        self._plane_thick_spin.setVisible(False)

        # Opacity — mirrors napari's layer.opacity → node.opacity
        row2.addWidget(QLabel('Opacity:'))
        self._opacity_spin = QDoubleSpinBox()
        self._opacity_spin.setRange(0.0, 1.0)
        self._opacity_spin.setSingleStep(0.05)
        self._opacity_spin.setValue(1.0)
        self._opacity_spin.setDecimals(2)
        self._opacity_spin.setFixedWidth(55)
        self._opacity_spin.setToolTip(
            'Overall volume opacity (0 = transparent, 1 = opaque).\n'
            'Mirrors napari\'s layer opacity control.'
        )
        self._opacity_spin.valueChanged.connect(self._on_opacity_changed)
        row2.addWidget(self._opacity_spin)

        # Reset View
        reset_btn = QPushButton('⟳ View')
        reset_btn.setFixedWidth(58)
        reset_btn.setToolTip('Reset camera to default orientation  (Ctrl+R)')
        reset_btn.clicked.connect(self._renderer.reset_view)
        row2.addWidget(reset_btn)

        # Save screenshot
        save_btn = QPushButton('💾 PNG')
        save_btn.setFixedWidth(58)
        save_btn.setToolTip('Save current 3D view as a PNG image  (Ctrl+S)')
        save_btn.clicked.connect(self._renderer.save_screenshot)
        row2.addWidget(save_btn)

        row2.addStretch()

        # Initial visibility for mode-specific controls
        self._update_mode_controls('mip')

    # -- mode-aware control visibility ----------------------------------------

    def _update_mode_controls(self, mode: str) -> None:
        show_iso = mode in _ISO_MODES
        show_attn = mode in _ATTN_MODES
        self._iso_label.setVisible(show_iso)
        self._iso_spin.setVisible(show_iso)
        self._smooth_iso_cb.setVisible(show_iso)
        self._attn_label.setVisible(show_attn)
        self._attn_spin.setVisible(show_attn)

    # -- slots ----------------------------------------------------------------

    def _on_mode_changed(self, idx: int) -> None:
        mode = self._mode_combo.itemData(idx)
        self._update_mode_controls(mode)
        self._renderer.set_rendering_mode(mode)

    def _on_clim_changed(self, _value: float) -> None:
        lo = self._clim_min.value()
        hi = self._clim_max.value()
        if lo >= hi:
            return
        self._renderer.set_clim(lo, hi)

    def _on_auto_clim(self) -> None:
        # Delegate to renderer so it can use the cached raw data for percentile.
        self._renderer.auto_contrast_percentile()

    def _on_reset_clim(self) -> None:
        self._clim_min.blockSignals(True)
        self._clim_max.blockSignals(True)
        self._clim_min.setValue(0.0)
        self._clim_max.setValue(1.0)
        self._clim_min.blockSignals(False)
        self._clim_max.blockSignals(False)
        self._renderer.set_clim(0.0, 1.0)

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
            self._zplane_label.setText(axis_names.get(mode, 'Pos:'))
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

    def _on_opacity_changed(self, value: float) -> None:
        self._renderer.set_opacity(value)


# ---------------------------------------------------------------------------
# Main renderer window
# ---------------------------------------------------------------------------
class VolumeRenderer3DWindow(QMainWindow):
    """
    A standalone Qt window that displays a 3D z-stack volume using vispy.

    Usage (minimal)::

        renderer = VolumeRenderer3DWindow()
        renderer.update_volume(zstack_array)   # (Z, Y, X) numpy array
        renderer.show()

    The window hides (rather than closes) when the user presses X so the GPU
    state is preserved for re-display.  Pass ``hide_on_close=False`` to
    destroy on close instead.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        hide_on_close: bool = True,
        adapter: VolumeRendererAdapter | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle('3D Z-Stack Renderer')
        self.resize(960, 720)

        self._hide_on_close = hide_on_close
        self._adapter = adapter
        self._volume_node = None
        self._overlay_nodes: list = []
        self._overlay_mode_overrides: list = []  # str or None per overlay node
        self._last_shape: tuple | None = None
        self._max_texture_3d: int | None = None  # resolved on first upload
        self._smooth_iso: bool = False
        self._gpu_data_is_smoothed: bool = False  # tracks whether GPU texture has Gaussian filter
        self._last_raw_data: np.ndarray | None = None  # float32, for re-render

        self._init_vispy()
        self._init_ui()

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

    # -- Qt UI ----------------------------------------------------------------

    def _init_ui(self) -> None:
        controls_box = QGroupBox('Rendering Controls')
        self._controls = VolumeRendererControls(self, parent=controls_box)
        box_layout = QVBoxLayout(controls_box)
        box_layout.setContentsMargins(4, 4, 4, 4)
        box_layout.addWidget(self._controls)

        central = QWidget()
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addWidget(self._canvas.native, stretch=1)
        main_layout.addWidget(controls_box)
        self.setCentralWidget(central)

        # Keyboard shortcuts
        QShortcut(QKeySequence('Ctrl+R'), self).activated.connect(self.reset_view)
        QShortcut(QKeySequence('Ctrl+A'), self).activated.connect(
            self.auto_contrast_percentile
        )
        QShortcut(QKeySequence('Ctrl+S'), self).activated.connect(self.save_screenshot)

        # Restore settings saved in a previous session.
        self._load_settings()

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
        """Apply cutoffs to the primary volume node (wrapper for backwards compat)."""
        self._apply_mode_cutoffs_to(self._volume_node, mode, lo, hi)

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

    # -- Public API -----------------------------------------------------------

    def update_volume(self, data: np.ndarray) -> None:
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
        if data.ndim != 3:
            raise ValueError(f'Expected 3-D (Z, Y, X) array; got shape {data.shape}')

        # copy=False avoids a redundant allocation when data is already float32
        # (e.g. when _rerender calls update_volume(self._last_raw_data)).
        vol = data.astype(np.float32, copy=False)
        # Cache raw float32 data so smooth-ISO toggle can re-process without
        # requiring a frame navigation in the host application.
        self._last_raw_data = vol
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

        from vispy.scene import visuals  # noqa: PLC0415

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
        current_cmap = 'grays'
        current_interp = self._controls._interp_combo.currentData() or 'linear'
        current_step = self._controls._step_spin.value()
        clim = (self._controls._clim_min.value(), self._controls._clim_max.value())

        shape_changed = vol.shape != self._last_shape

        if self._volume_node is None or shape_changed:
            # (Re-)create the volume visual when no node exists or shape changed.
            # Shape changes require a new texture, so we rebuild the node.
            if self._volume_node is not None:
                self._volume_node.parent = None  # detach old node
            self._volume_node = visuals.Volume(
                vol,
                clim=clim,
                method=current_mode,
                cmap=current_cmap,
                interpolation=current_interp,
                relative_step_size=current_step,
                parent=self._view.scene,
            )
            self._volume_node.gamma = self._controls._gamma_spin.value()
            self._volume_node.opacity = self._controls._opacity_spin.value()
            if current_mode in _ATTN_MODES:
                self._volume_node.attenuation = self._controls._attn_spin.value()
            if current_mode in _ISO_MODES:
                self._volume_node.threshold = self._controls._iso_spin.value()
            # Apply MIP/MinIP cutoffs so the background is transparent (napari approach).
            self._apply_mode_cutoffs(current_mode, clim[0], clim[1])
            # Apply the current depiction mode (restored from settings or changed
            # before data was loaded — the Volume constructor always defaults to
            # 'volume', so we must explicitly set it on new nodes).
            depict_mode = self._controls._depict_combo.currentData() or 'volume'
            if depict_mode in _PLANE_CONFIGS:
                self._volume_node.raycasting_mode = 'plane'
                # Pass vol.shape explicitly: _last_shape not yet updated here.
                self._set_plane_uniforms(
                    depict_mode,
                    self._controls._zplane_slider.value() / 100.0,
                    shape=vol.shape,
                )
            # Re-apply voxel scale so the transform survives node rebuilds
            # (shape changes, first load).  Also handles the case where
            # set_voxel_scale was called before any data was loaded.
            self._apply_voxel_scale()
            # Fit camera only on the very first load; preserve user's view on
            # subsequent shape changes (e.g. switching between positions with
            # different SizeZ).
            if self._last_shape is None:
                self._view.camera.set_range()
            # Rescale axis indicator to match the new volume dimensions.
            if self._axis_visual is not None:
                from vispy.visuals.transforms import STTransform  # noqa: PLC0415
                axis_scale = max(2.0, max(vol.shape) * 0.10)
                self._axis_visual.transform = STTransform(
                    scale=(axis_scale, axis_scale, axis_scale)
                )
                self._axis_visual.visible = True
        else:
            self._volume_node.set_data(vol, clim=clim)
            self._apply_mode_cutoffs(current_mode, clim[0], clim[1])

        self._last_shape = vol.shape

        # Status bar: show volume dimensions and total voxel count.
        nz, ny, nx = original_shape
        mvoxels = nz * ny * nx / 1e6
        if vol.shape != original_shape:
            self.statusBar().showMessage(
                f'{nz}×{ny}×{nx} ({mvoxels:.1f} M)  ▸  '
                f'downsampled → {vol.shape}  (GPU limit: {max_tex})',
                0,  # stays until next message
            )
        else:
            self.statusBar().showMessage(
                f'Volume: {nz}×{ny}×{nx}   ({mvoxels:.1f} M voxels)',
                0,
            )

        self._canvas.update()

    def update_overlay_volumes(
        self,
        overlays: list[tuple],
    ) -> None:
        """Replace the displayed overlay volumes.

        Parameters
        ----------
        overlays:
            List of tuples ``(data, opacity, colormap[, mode_override])``.
            *data* is a ``(Z, Y, X)`` array; *opacity* ∈ ``[0, 1]``;
            *colormap* is a vispy colormap name.  The optional fourth element
            *mode_override* (str or None) forces a specific rendering mode for
            this overlay independently of the primary volume's mode — pass
            ``'mip'`` for binary masks so filled interiors are always visible
            regardless of the primary's ISO/translucent mode.
            Pass an empty list to clear all overlays.
        """
        from vispy.scene import visuals  # noqa: PLC0415

        for node in self._overlay_nodes:
            node.parent = None
        self._overlay_nodes.clear()
        self._overlay_mode_overrides.clear()

        if not overlays:
            self._canvas.update()
            return

        max_tex = self._resolve_max_texture_3d()
        primary_mode = self._controls._mode_combo.currentData() or 'mip'
        current_interp = self._controls._interp_combo.currentData() or 'linear'
        current_step = self._controls._step_spin.value()
        depict_mode = self._controls._depict_combo.currentData() or 'volume'
        is_plane = depict_mode in _PLANE_CONFIGS
        plane_fraction = self._controls._zplane_slider.value() / 100.0

        for entry in overlays:
            data, opacity, cmap = entry[0], entry[1], entry[2]
            mode_override = entry[3] if len(entry) > 3 else None
            node_mode = mode_override or primary_mode

            if data.ndim != 3:
                continue
            vol = data.astype(np.float32, copy=False)
            vmin, vmax = float(vol.min()), float(vol.max())
            if max(vol.shape) > max_tex:
                vol = self._downsample(vol, max_tex)
            if vmax > vmin:
                vol = (vol - vmin) / (vmax - vmin)
            else:
                vol = np.zeros_like(vol)

            node = visuals.Volume(
                vol,
                clim=(0.0, 1.0),
                method=node_mode,
                cmap=_make_cmap(cmap),
                interpolation=current_interp,
                relative_step_size=current_step,
                parent=self._view.scene,
            )
            node.opacity = max(0.0, min(1.0, opacity))
            node.gamma = self._controls._gamma_spin.value()
            self._apply_mode_cutoffs_to(node, node_mode, 0.0, 1.0)
            if node_mode in _ATTN_MODES:
                node.attenuation = self._controls._attn_spin.value()
            if node_mode in _ISO_MODES:
                node.threshold = self._controls._iso_spin.value()
            if is_plane:
                node.raycasting_mode = 'plane'
                self._set_plane_uniforms(depict_mode, plane_fraction,
                                         shape=vol.shape, node=node)
            self._overlay_nodes.append(node)
            self._overlay_mode_overrides.append(mode_override)

        # Apply the stored voxel-scale transform so overlays align with the
        # primary volume even when dz≠dy≠dx.
        self._apply_voxel_scale()

    def set_rendering_mode(self, mode: str) -> None:
        if self._volume_node is not None:
            # When entering ISO mode, the GPU texture must match the smooth flag.
            # Re-upload if the smooth state changed since the last upload (e.g.
            # the user toggled Smooth while in MIP mode, then switches to ISO —
            # the GPU still has the old texture from the last update_volume call).
            if mode in _ISO_MODES and (self._smooth_iso != self._gpu_data_is_smoothed):
                self._volume_node.method = mode  # set before _rerender reads it
                self._rerender()
                return
            self._volume_node.method = mode
            lo = self._controls._clim_min.value()
            hi = self._controls._clim_max.value()
            self._apply_mode_cutoffs(mode, lo, hi)
            # Sync mode-specific parameters from the UI so the node matches the
            # spinbox values even when the node was created in a different mode.
            if mode in _ISO_MODES:
                self._volume_node.threshold = self._controls._iso_spin.value()
            if mode in _ATTN_MODES:
                self._volume_node.attenuation = self._controls._attn_spin.value()
        for node, override in zip(self._overlay_nodes, self._overlay_mode_overrides):
            if override is not None:
                continue  # this overlay has a fixed mode; don't change it
            node.method = mode
            self._apply_mode_cutoffs_to(node, mode, 0.0, 1.0)
            if mode in _ISO_MODES:
                node.threshold = self._controls._iso_spin.value()
            if mode in _ATTN_MODES:
                node.attenuation = self._controls._attn_spin.value()
        self._canvas.update()

    def set_clim(self, lo: float, hi: float) -> None:
        if self._volume_node is not None:
            self._volume_node.clim = (lo, hi)
            current_mode = self._controls._mode_combo.currentData() or 'mip'
            self._apply_mode_cutoffs(current_mode, lo, hi)
            self._canvas.update()

    def auto_contrast_percentile(
        self, lo_pct: float = 1.0, hi_pct: float = 99.5
    ) -> None:
        """
        Set contrast limits to the *lo_pct*–*hi_pct* percentile of the raw
        volume data, clipped to [0, 1] in the normalised space.

        This is more useful for fluorescence microscopy than a full-range reset
        because bright artefact voxels are excluded without manual adjustment.
        Mirrors the spirit of napari's auto-contrast which maps the meaningful
        intensity range rather than the absolute min–max.

        Falls back to [0, 1] when no volume has been loaded yet.
        """
        if self._last_raw_data is None:
            lo, hi = 0.0, 1.0
        else:
            raw = self._last_raw_data
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

        c = self._controls
        c._clim_min.blockSignals(True)
        c._clim_max.blockSignals(True)
        c._clim_min.setValue(lo)
        c._clim_max.setValue(hi)
        c._clim_min.blockSignals(False)
        c._clim_max.blockSignals(False)
        self.set_clim(lo, hi)

    def set_gamma(self, value: float) -> None:
        if self._volume_node is not None:
            self._volume_node.gamma = value
        for node in self._overlay_nodes:
            node.gamma = value
        if self._volume_node is not None or self._overlay_nodes:
            self._canvas.update()

    def set_opacity(self, value: float) -> None:
        """
        Set the overall volume opacity (0 = fully transparent, 1 = opaque).

        Mirrors napari's ``layer.opacity → node.opacity`` pathway.  The effect
        depends on the rendering mode: most visible in translucent and additive
        modes; has no visual effect in MIP/MinIP (which project to a 2D plane).
        """
        if self._volume_node is not None:
            self._volume_node.opacity = max(0.0, min(1.0, value))
            self._canvas.update()

    def set_iso_threshold(self, value: float) -> None:
        if self._volume_node is not None:
            self._volume_node.threshold = value
            self._canvas.update()

    def set_attenuation(self, value: float) -> None:
        if self._volume_node is not None:
            self._volume_node.attenuation = value
            self._canvas.update()

    def set_interpolation(self, method: str) -> None:
        """Set 3D volume interpolation method (e.g. 'linear', 'nearest', 'catrom')."""
        if self._volume_node is not None:
            self._volume_node.interpolation = method
        for node in self._overlay_nodes:
            node.interpolation = method
        if self._volume_node is not None or self._overlay_nodes:
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
        if self._volume_node is not None:
            self._volume_node.raycasting_mode = 'plane' if is_plane else 'volume'
            if is_plane and self._last_shape is not None:
                self._set_plane_uniforms(mode, 0.5)
        for node in self._overlay_nodes:
            node.raycasting_mode = 'plane' if is_plane else 'volume'
            if is_plane and self._last_shape is not None:
                self._set_plane_uniforms(mode, 0.5, node=node)
        self._canvas.update()

    def set_zplane_position(self, fraction: float) -> None:
        """
        Move the active cross-section plane to *fraction* ∈ [0, 1] of its axis.

        The axis is determined by the currently selected depiction mode.
        """
        if self._last_shape is None:
            return
        current_mode = self._controls._depict_combo.currentData() or 'volume'
        if current_mode not in _PLANE_CONFIGS:
            return
        if self._volume_node is not None:
            self._set_plane_uniforms(current_mode, fraction)
        for node in self._overlay_nodes:
            self._set_plane_uniforms(current_mode, fraction, node=node)
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
            Volume node to update.  Defaults to ``self._volume_node``.
        """
        if node is None:
            node = self._volume_node
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
        if self._volume_node is not None:
            self._volume_node.plane_thickness = t
        for node in self._overlay_nodes:
            node.plane_thickness = t
        if self._volume_node is not None or self._overlay_nodes:
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
        if self._volume_node is not None:
            self._volume_node.relative_step_size = value
        for node in self._overlay_nodes:
            node.relative_step_size = value
        if self._volume_node is not None or self._overlay_nodes:
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
        if self._volume_node is None:
            return
        self._apply_voxel_scale()

    def _apply_voxel_scale(self) -> None:
        """Apply the stored voxel scale and stride correction to all volume nodes."""
        from vispy.visuals.transforms import STTransform  # noqa: PLC0415
        sz, sy, sx = self._last_strides
        dx_eff = self._voxel_dx * sx
        dy_eff = self._voxel_dy * sy
        dz_eff = self._voxel_dz * sz
        if dx_eff <= 0:
            dx_eff = 1.0
        transform = STTransform(scale=(1.0, dy_eff / dx_eff, dz_eff / dx_eff))
        if self._volume_node is not None:
            self._volume_node.transform = transform
        for node in self._overlay_nodes:
            node.transform = transform
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
        self._save_settings()
        if self._hide_on_close:
            event.ignore()
            self.hide()
            if self._adapter is not None:
                self._adapter.on_renderer_closed()
        else:
            super().closeEvent(event)


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
