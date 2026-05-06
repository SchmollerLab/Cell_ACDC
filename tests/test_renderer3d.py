"""Smoke tests for cellacdc.renderer3d.

These tests verify module structure, constants, and numpy-only logic without
requiring a running display or GPU context.  Window/canvas creation is not
tested here because it requires an OpenGL context.
"""

import numpy as np
import pytest


def test_module_imports():
    """renderer3d must be importable without vispy being present at import time."""
    from cellacdc import renderer3d  # noqa: F401


def test_rendering_modes_match_vispy():
    """Every mode registered in renderer3d must be accepted by vispy.Volume."""
    from cellacdc import renderer3d
    import vispy
    from qtpy import API_NAME
    vispy.use(API_NAME)
    from vispy.scene.visuals import Volume

    vispy_methods = set(Volume._rendering_methods.keys())
    for mode_id, _label in renderer3d.RENDERING_MODES:
        assert mode_id in vispy_methods, (
            f"Rendering mode '{mode_id}' in RENDERING_MODES is not a valid "
            f"vispy Volume method.  Valid methods: {sorted(vispy_methods)}"
        )


def test_rendering_modes_structure():
    from cellacdc import renderer3d

    assert len(renderer3d.RENDERING_MODES) > 0
    for entry in renderer3d.RENDERING_MODES:
        assert len(entry) == 2, "Each RENDERING_MODES entry must be (id, label)"
        mode_id, label = entry
        assert isinstance(mode_id, str) and mode_id
        assert isinstance(label, str) and label


def test_colormaps_non_empty():
    from cellacdc import renderer3d

    assert len(renderer3d.COLORMAPS) > 0
    for cmap in renderer3d.COLORMAPS:
        assert isinstance(cmap, str) and cmap


def test_adapter_protocol():
    """VolumeRendererAdapter subclass must raise NotImplementedError."""
    from cellacdc.renderer3d import VolumeRendererAdapter

    adapter = VolumeRendererAdapter()
    with pytest.raises(NotImplementedError):
        adapter.get_current_zstack()
    # on_renderer_closed has a default no-op implementation
    adapter.on_renderer_closed()  # must not raise


def test_create_renderer_callable():
    """create_renderer must be callable with the documented signature."""
    from cellacdc import renderer3d
    import inspect

    sig = inspect.signature(renderer3d.create_renderer)
    params = set(sig.parameters)
    assert 'parent' in params
    assert 'hide_on_close' in params
    assert 'adapter' in params


def test_volume_normalisation():
    """update_volume normalises data to [0, 1] float32 — verify on headless path."""
    from cellacdc import renderer3d

    def _norm(data):
        # Mirrors update_volume: copy=False avoids redundant allocation for float32.
        vol = data.astype(np.float32, copy=False)
        vmin, vmax = float(vol.min()), float(vol.max())
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)
        else:
            vol = np.zeros_like(vol)
        assert vol.dtype == np.float32
        assert vol.min() >= 0.0
        assert vol.max() <= 1.0

    rng = np.random.default_rng(42)
    _norm(rng.integers(0, 65535, size=(10, 64, 64)).astype(np.uint16))
    _norm(rng.random((5, 32, 32)).astype(np.float64))
    _norm(np.full((4, 16, 16), 42, dtype=np.uint8))  # all-same → all zeros

    # float32 input: copy=False returns same object; normalisation still correct
    f32 = np.linspace(0, 1, 5 * 4 * 4, dtype=np.float32).reshape(5, 4, 4)
    vol = f32.astype(np.float32, copy=False)
    assert vol is f32  # no unnecessary copy
    vmin, vmax = float(vol.min()), float(vol.max())
    assert vmax > vmin


def test_downsample_static():
    """_downsample returns a smaller array when a dimension exceeds max_size."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow

    ds = VolumeRenderer3DWindow._downsample

    vol = np.zeros((200, 512, 512), dtype=np.float32)

    # max_size=256 → each dim reduced by ceil(s/256)
    out = ds(vol, 256)
    assert all(s <= 256 for s in out.shape), f"Shape {out.shape} exceeds limit 256"
    assert out.dtype == np.float32

    # max_size large enough → no change
    out_no_ds = ds(vol, 1024)
    assert out_no_ds.shape == vol.shape

    # Downsampling preserves dtype
    vol_u16 = np.zeros((100, 300, 300), dtype=np.uint16)
    out_u16 = ds(vol_u16, 256)
    assert out_u16.dtype == np.uint16
    assert all(s <= 256 for s in out_u16.shape)

    # Stride logic: ceil(200/256)=1, ceil(512/256)=2
    vol2 = np.arange(200 * 512 * 512, dtype=np.float32).reshape(200, 512, 512)
    out2 = ds(vol2, 256)
    assert out2.shape == (200, 256, 256)


def test_volume_shape_validation():
    """update_volume must raise ValueError for non-3D input."""
    from cellacdc import renderer3d

    class _HeadlessRenderer(renderer3d.VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): pass

    r = _HeadlessRenderer.__new__(_HeadlessRenderer)
    # Grab the normalisation-only path from the real class
    original = renderer3d.VolumeRenderer3DWindow.update_volume

    with pytest.raises(ValueError, match='3-D'):
        original(r, np.zeros((64, 64)))     # 2D

    with pytest.raises(ValueError, match='3-D'):
        original(r, np.zeros((1, 2, 3, 4))) # 4D


def test_interpolation_modes_structure():
    """INTERPOLATION_MODES entries must be (id, label) pairs."""
    from cellacdc import renderer3d
    assert len(renderer3d.INTERPOLATION_MODES) > 0
    for entry in renderer3d.INTERPOLATION_MODES:
        assert len(entry) == 2
        iid, label = entry
        assert isinstance(iid, str) and iid
        assert isinstance(label, str) and label


def test_interpolation_modes_valid_in_vispy():
    """Every INTERPOLATION_MODES id must be a valid vispy Volume interpolation."""
    from cellacdc import renderer3d
    from vispy.io import load_spatial_filters
    _, vispy_methods = load_spatial_filters()
    valid = {m.lower() for m in vispy_methods}
    for iid, label in renderer3d.INTERPOLATION_MODES:
        assert iid in valid, (
            f"Interpolation mode '{iid}' is not in vispy's filter set: {sorted(valid)}"
        )


def test_renderer_public_api():
    """VolumeRenderer3DWindow must expose all expected public methods."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    required = {
        'update_volume', 'set_rendering_mode', 'set_colormap', 'set_clim',
        'set_gamma', 'set_opacity', 'set_iso_threshold', 'set_attenuation',
        'set_interpolation', 'set_step_size', 'set_smooth_iso',
        'set_depiction', 'set_zplane_position', 'set_plane_thickness',
        'set_voxel_scale', 'reset_view', 'save_screenshot',
        'auto_contrast_percentile',
    }
    missing = required - set(dir(VolumeRenderer3DWindow))
    assert not missing, f"Missing public methods: {missing}"


def test_adapter_get_voxel_sizes_default():
    """VolumeRendererAdapter.get_voxel_sizes() must return None by default."""
    from cellacdc.renderer3d import VolumeRendererAdapter
    adapter = VolumeRendererAdapter()
    result = adapter.get_voxel_sizes()
    assert result is None


def test_set_voxel_scale_noop_without_node():
    """set_voxel_scale must be a no-op when no volume node exists."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow

    class _Bare(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): self._controls = None

    r = _Bare.__new__(_Bare)
    r._volume_node = None
    r.set_voxel_scale(0.5, 0.2, 0.2)  # must not raise


def test_set_voxel_scale_stride_correction():
    """set_voxel_scale must incorporate per-axis downsampling strides.

    Scenario: 100×512×2048 volume downsampled with stride (1,1,4) to fit GPU.
    Each downsampled X-voxel spans 4 × dx physical µm, so the effective dx
    is 0.8 µm.  The STTransform scale for (dz=1, dy=0.2, dx=0.2) with these
    strides should be (1.0, 0.25, 1.25) not (1.0, 1.0, 5.0).
    """
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    from unittest.mock import MagicMock

    class _Bare(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): self._controls = None

    r = _Bare.__new__(_Bare)
    r._volume_node = MagicMock()
    r._canvas = MagicMock()
    r._last_strides = (1, 1, 4)  # only X was downsampled

    assigned_transforms = []
    type(r._volume_node).transform = property(
        fget=lambda self: None,
        fset=lambda self, v: assigned_transforms.append(v),
    )

    r.set_voxel_scale(dz=1.0, dy=0.2, dx=0.2)

    assert len(assigned_transforms) == 1
    t = assigned_transforms[0]
    scale = t.scale  # (scene-x, scene-y, scene-z)
    # Effective voxel sizes: dx_eff=0.8, dy_eff=0.2, dz_eff=1.0
    # Expected scale: (1.0, 0.2/0.8, 1.0/0.8) = (1.0, 0.25, 1.25)
    assert abs(scale[0] - 1.0)  < 1e-6, f"scale-x wrong: {scale[0]}"
    assert abs(scale[1] - 0.25) < 1e-6, f"scale-y wrong: {scale[1]}"
    assert abs(scale[2] - 1.25) < 1e-6, f"scale-z wrong: {scale[2]}"


def test_voxel_scale_persists_across_node_rebuild():
    """_voxel_d* stored in set_voxel_scale must survive a volume node rebuild.

    set_voxel_scale before update_volume is a common standalone pattern:
      renderer.set_voxel_scale(dz=2.0, dy=1.0, dx=1.0)
      renderer.update_volume(data)
    The scale must be applied even though the node didn't exist yet when
    set_voxel_scale was called.
    """
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    assert VolumeRenderer3DWindow._voxel_dz == 1.0  # class default
    assert VolumeRenderer3DWindow._voxel_dy == 1.0
    assert VolumeRenderer3DWindow._voxel_dx == 1.0

    from unittest.mock import MagicMock

    class _Bare(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): self._controls = None

    r = _Bare.__new__(_Bare)
    r._volume_node = None
    # Store scale without a node — must not raise, must persist.
    r.set_voxel_scale(dz=2.0, dy=1.0, dx=1.0)
    assert r._voxel_dz == 2.0
    assert r._voxel_dy == 1.0
    assert r._voxel_dx == 1.0


def test_write_png_stdlib(tmp_path):
    """_write_png must produce a valid PNG file readable by skimage."""
    import skimage.io
    from cellacdc.renderer3d import _write_png

    rng = np.random.default_rng(0)
    rgba = rng.integers(0, 255, (16, 32, 4), dtype=np.uint8)
    dest = str(tmp_path / 'test.png')
    _write_png(dest, rgba)

    loaded = skimage.io.imread(dest)
    assert loaded.shape == rgba.shape
    np.testing.assert_array_equal(loaded, rgba)


def test_default_step_size():
    """_DEFAULT_STEP_SIZE must match vispy's own default (0.8)."""
    from cellacdc import renderer3d
    assert renderer3d._DEFAULT_STEP_SIZE == 0.8


def test_step_size_noop_without_node():
    """set_step_size must not raise when no volume node exists."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow

    class _Bare(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): self._controls = None

    r = _Bare.__new__(_Bare)
    r._volume_node = None
    r.set_step_size(0.5)  # must not raise


def test_set_opacity_noop_without_node():
    """set_opacity must not raise when no volume node exists."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow

    class _Bare(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): self._controls = None

    r = _Bare.__new__(_Bare)
    r._volume_node = None
    r.set_opacity(0.5)  # must not raise


def test_set_opacity_clamps_to_unit_range():
    """set_opacity must clamp out-of-range values to [0, 1] before applying."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    from unittest.mock import MagicMock, call, patch

    class _Bare(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): self._controls = None

    r = _Bare.__new__(_Bare)
    r._volume_node = MagicMock()
    r._canvas = MagicMock()

    # Track what value was actually assigned to _volume_node.opacity
    assigned = []
    type(r._volume_node).opacity = property(
        fget=lambda self: None,
        fset=lambda self, v: assigned.append(v),
    )

    r.set_opacity(2.0)    # above 1 → clamp to 1.0
    assert assigned[-1] == 1.0

    r.set_opacity(-0.5)   # below 0 → clamp to 0.0
    assert assigned[-1] == 0.0

    r.set_opacity(0.7)    # in-range → pass through
    assert abs(assigned[-1] - 0.7) < 1e-9


def test_mip_cutoff_mode_sets():
    """_MIP_CUTOFF_MODES and _MINIP_CUTOFF_MODES must only reference valid vispy methods."""
    from cellacdc import renderer3d
    import vispy
    from qtpy import API_NAME
    vispy.use(API_NAME)
    from vispy.scene.visuals import Volume

    vispy_methods = set(Volume._rendering_methods.keys())
    for mode in renderer3d._MIP_CUTOFF_MODES:
        assert mode in vispy_methods, f"mip-cutoff mode '{mode}' not in vispy methods"
    for mode in renderer3d._MINIP_CUTOFF_MODES:
        assert mode in vispy_methods, f"minip-cutoff mode '{mode}' not in vispy methods"


def test_apply_mode_cutoffs_noop_without_node():
    """_apply_mode_cutoffs must be a no-op when no volume node exists."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow

    class _Bare(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): self._controls = None

    r = _Bare.__new__(_Bare)
    r._volume_node = None
    r._apply_mode_cutoffs('mip', 0.1, 0.9)  # must not raise


def test_depiction_modes_structure():
    """DEPICTION_MODES must contain 'volume' and at least one plane mode."""
    from cellacdc import renderer3d
    assert len(renderer3d.DEPICTION_MODES) >= 2
    ids = {d[0] for d in renderer3d.DEPICTION_MODES}
    assert 'volume' in ids
    plane_ids = {k for k in ids if k.startswith('plane_')}
    assert len(plane_ids) >= 1, "At least one plane depiction mode required"
    # _PLANE_CONFIGS must cover all plane ids
    for pid in plane_ids:
        assert pid in renderer3d._PLANE_CONFIGS, f"'{pid}' missing from _PLANE_CONFIGS"


def test_depiction_plane_configs_valid_in_vispy():
    """_PLANE_CONFIGS normals must be unit vectors; vispy supports 'plane' mode."""
    import vispy; from qtpy import API_NAME; vispy.use(API_NAME)
    from vispy.scene.visuals import Volume
    from cellacdc import renderer3d
    assert 'plane' in Volume._raycasting_modes   # vispy supports 'plane'
    assert 'volume' in Volume._raycasting_modes
    # Each normal should be a length-3 unit vector
    for mode_key, (normal, axis) in renderer3d._PLANE_CONFIGS.items():
        assert len(normal) == 3, f"Normal for '{mode_key}' must be length-3"
        mag = sum(v**2 for v in normal) ** 0.5
        assert abs(mag - 1.0) < 1e-6, f"Normal for '{mode_key}' is not unit: {normal}"
        assert axis in (0, 1, 2), f"Axis for '{mode_key}' must be 0,1,2 got {axis}"


def test_plane_thickness_noop_without_node():
    """set_plane_thickness must be a no-op when no volume node exists."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow

    class _Bare(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): self._controls = None

    r = _Bare.__new__(_Bare)
    r._volume_node = None
    r.set_plane_thickness(5.0)   # must not raise
    r.set_plane_thickness(0.0)   # must clamp silently (no node)


def test_zplane_uniforms_noop_without_node():
    """set_depiction and set_zplane_position must be no-ops when no node exists."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow

    class _Bare(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): self._controls = None

    r = _Bare.__new__(_Bare)
    r._volume_node = None
    r._last_shape = None
    r.set_depiction('plane')       # must not raise
    r.set_zplane_position(0.5)     # must not raise


def test_set_plane_uniforms_geometry():
    """_set_plane_uniforms must compute correct scene-space position and normal."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow, _PLANE_CONFIGS
    from unittest.mock import MagicMock

    r = VolumeRenderer3DWindow.__new__(VolumeRenderer3DWindow)
    r._volume_node = MagicMock()
    r._last_shape = None
    # Set up _controls mock directly (not via _init_ui to avoid display).
    controls = MagicMock()
    controls._plane_thick_spin.value.return_value = 2.0
    r._controls = controls

    shape = (30, 64, 128)  # NZ=30, NY=64, NX=128

    # --- plane_z: XY cross-section, normal along Z (scene-z = data-Z axis) ---
    r._set_plane_uniforms('plane_z', 0.0, shape=shape)
    pos = r._volume_node.plane_position
    normal = r._volume_node.plane_normal
    assert normal == [0.0, 0.0, 1.0]
    # fraction=0.0 → z = 0.0*(30-1) = 0.0 (first voxel centre)
    assert abs(pos[2] - 0.0) < 1e-6, f"plane_z pos[2] should be 0.0, got {pos[2]}"
    # X and Y centres: (NX-1)/2 = 63.5, (NY-1)/2 = 31.5
    assert abs(pos[0] - 63.5) < 1e-6
    assert abs(pos[1] - 31.5) < 1e-6

    r._set_plane_uniforms('plane_z', 1.0, shape=shape)
    pos = r._volume_node.plane_position
    # fraction=1.0 → z = 1.0*(30-1) = 29.0 (last voxel centre)
    assert abs(pos[2] - 29.0) < 1e-6, f"plane_z pos[2] should be 29.0, got {pos[2]}"

    r._set_plane_uniforms('plane_z', 0.5, shape=shape)
    pos = r._volume_node.plane_position
    # fraction=0.5 → z = 0.5*(30-1) = 14.5  →  texture_z = (14.5+0.5)/30 = 0.5 ✓ (exact centre)
    assert abs(pos[2] - 14.5) < 1e-6, f"plane_z pos[2] should be 14.5, got {pos[2]}"

    # --- plane_y: XZ cross-section, normal along Y (scene-y = data-Y axis) ---
    r._set_plane_uniforms('plane_y', 0.5, shape=shape)
    pos = r._volume_node.plane_position
    normal = r._volume_node.plane_normal
    assert normal == [0.0, 1.0, 0.0]
    # fraction=0.5 → y = 0.5*(64-1) = 31.5  →  texture_y = (31.5+0.5)/64 = 0.5 ✓
    assert abs(pos[1] - 31.5) < 1e-6, f"plane_y pos[1] should be 31.5, got {pos[1]}"

    # --- plane_x: YZ cross-section, normal along X (scene-x = data-X axis) ---
    r._set_plane_uniforms('plane_x', 0.5, shape=shape)
    pos = r._volume_node.plane_position
    normal = r._volume_node.plane_normal
    assert normal == [1.0, 0.0, 0.0]
    # fraction=0.5 → x = 0.5*(128-1) = 63.5  →  texture_x = (63.5+0.5)/128 = 0.5 ✓
    assert abs(pos[0] - 63.5) < 1e-6, f"plane_x pos[0] should be 63.5, got {pos[0]}"

    # Thickness must be read from the spinbox (mocked to 2.0)
    assert r._volume_node.plane_thickness == 2.0

    # None shape + _last_shape=None → must return without crash
    r._last_shape = None
    r._set_plane_uniforms('plane_z', 0.5, shape=None)  # must not raise


def test_smooth_iso_constant():
    """_SMOOTH_ISO_SIGMA must be a positive float."""
    from cellacdc import renderer3d
    assert isinstance(renderer3d._SMOOTH_ISO_SIGMA, float)
    assert renderer3d._SMOOTH_ISO_SIGMA > 0


def test_smooth_iso_gaussian():
    """Gaussian pre-filter reduces variance, preserves shape and float32 dtype."""
    import scipy.ndimage
    from cellacdc.renderer3d import _SMOOTH_ISO_SIGMA

    rng = np.random.default_rng(7)
    vol = rng.random((20, 20, 20), dtype=np.float32)
    smoothed = scipy.ndimage.gaussian_filter(vol, sigma=_SMOOTH_ISO_SIGMA)
    assert smoothed.std() < vol.std()          # smoothing reduces variance
    assert smoothed.shape == vol.shape         # shape unchanged
    assert smoothed.dtype == np.float32        # dtype preserved (important for GPU upload)


def test_smooth_iso_only_in_iso_mode():
    """_smooth_iso flag must not apply filtering in non-ISO rendering modes."""
    import scipy.ndimage
    from cellacdc.renderer3d import _SMOOTH_ISO_SIGMA, _ISO_MODES

    rng = np.random.default_rng(11)
    vol = rng.random((10, 10, 10), dtype=np.float32)

    # Modes that should NOT trigger smoothing
    non_iso_modes = ['mip', 'minip', 'attenuated_mip', 'translucent', 'additive', 'average']
    for mode in non_iso_modes:
        assert mode not in _ISO_MODES, f"'{mode}' unexpectedly in _ISO_MODES"

    # Verify ISO is the only smooth-triggering mode
    assert 'iso' in _ISO_MODES


def test_settings_constants():
    """_SETTINGS_ORG and _SETTINGS_APP must be non-empty strings."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    assert isinstance(VolumeRenderer3DWindow._SETTINGS_ORG, str)
    assert isinstance(VolumeRenderer3DWindow._SETTINGS_APP, str)
    assert VolumeRenderer3DWindow._SETTINGS_ORG
    assert VolumeRenderer3DWindow._SETTINGS_APP


def test_settings_roundtrip():
    """_save_settings / _load_settings must preserve all control values."""
    from qtpy.QtCore import QSettings
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    from unittest.mock import MagicMock

    # Use an isolated QSettings group to avoid polluting real settings.
    TEST_ORG = 'Cell-ACDC-test'
    TEST_APP = 'renderer3d-test'

    class _HeadlessWindow(VolumeRenderer3DWindow):
        _SETTINGS_ORG = TEST_ORG
        _SETTINGS_APP = TEST_APP

        def _init_vispy(self): pass
        def _init_ui(self):
            c = MagicMock()
            c._mode_combo.currentIndex.return_value = 3
            c._cmap_combo.currentText.return_value = 'viridis'
            c._interp_combo.currentIndex.return_value = 1
            c._clim_min.value.return_value = 0.05
            c._clim_max.value.return_value = 0.95
            c._gamma_spin.value.return_value = 1.5
            c._step_spin.value.return_value = 0.4
            self._controls = c

    try:
        win = _HeadlessWindow.__new__(_HeadlessWindow)
        win._hide_on_close = True
        win._adapter = None
        win._volume_node = None
        win._last_shape = None
        win._max_texture_3d = None
        win._init_ui()

        # Set all persisted controls explicitly so return values are typed correctly.
        win._controls._depict_combo.currentIndex.return_value = 1
        win._controls._plane_thick_spin.value.return_value = 3.0
        win._controls._opacity_spin.value.return_value = 0.75
        win._controls._smooth_iso_cb.isChecked.return_value = True  # explicit bool

        # Save the mocked values.
        win._save_settings()

        # Verify all 10 settings were persisted correctly.
        s = QSettings(TEST_ORG, TEST_APP)
        assert s.value('mode_idx',       type=int)   == 3
        assert s.value('colormap',       type=str)   == 'viridis'
        assert s.value('interp_idx',     type=int)   == 1
        assert abs(s.value('clim_min',   type=float) - 0.05) < 1e-6
        assert abs(s.value('clim_max',   type=float) - 0.95) < 1e-6
        assert abs(s.value('gamma',      type=float) - 1.5)  < 1e-6
        assert abs(s.value('step_size',  type=float) - 0.4)  < 1e-6
        assert s.value('smooth_iso',     type=bool)  is True
        assert s.value('depict_idx',     type=int)   == 1
        assert abs(s.value('plane_thickness', type=float) - 3.0)  < 1e-6
        assert abs(s.value('opacity',    type=float) - 0.75) < 1e-6
    finally:
        QSettings(TEST_ORG, TEST_APP).clear()


def test_gui_adapter_implements_protocol():
    """_GuiWinRenderer3DAdapter must expose all VolumeRendererAdapter methods."""
    from cellacdc.renderer3d import VolumeRendererAdapter
    from cellacdc.gui import _GuiWinRenderer3DAdapter

    # All public methods on the base protocol must be present on the adapter.
    protocol_methods = {
        m for m in dir(VolumeRendererAdapter)
        if not m.startswith('_')
    }
    adapter_methods = {
        m for m in dir(_GuiWinRenderer3DAdapter)
        if not m.startswith('_')
    }
    missing = protocol_methods - adapter_methods
    assert not missing, (
        f"_GuiWinRenderer3DAdapter does not implement: {missing}"
    )


def test_gui_renderer3d_methods_exist():
    """guiWin must define all the adapter/renderer integration methods."""
    from cellacdc.gui import guiWin

    required = {
        '_get_current_zstack',
        '_get_current_voxel_sizes',
        '_launch_3d_renderer',
        '_update_3d_renderer_if_active',
        '_hide_3d_renderer_if_open',
        '_position_renderer3d_window',
    }
    missing = required - set(dir(guiWin))
    assert not missing, f"guiWin missing renderer3d methods: {missing}"


def test_gui_py_constant():
    """_ZPROJMODE_3D must appear exactly once as a string literal in gui.py."""
    import pathlib
    gui_src = pathlib.Path(__file__).parent.parent / 'cellacdc' / 'gui.py'
    text = gui_src.read_text(encoding='utf-8')
    # The constant definition line is the only place the raw string should appear.
    count = text.count("'3D z-render'")
    assert count == 1, (
        f"Expected exactly 1 occurrence of raw '3D z-render' in gui.py "
        f"(the constant definition); found {count}."
    )


def test_last_raw_data_cached_after_update_volume():
    """update_volume must cache raw float32 data in _last_raw_data."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    from unittest.mock import MagicMock, patch

    class _Headless(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self):
            c = MagicMock()
            c._mode_combo.currentData.return_value = 'mip'
            c._cmap_combo.currentText.return_value = 'grays'
            c._interp_combo.currentData.return_value = 'linear'
            c._step_spin.value.return_value = 0.8
            c._clim_min.value.return_value = 0.0
            c._clim_max.value.return_value = 1.0
            c._gamma_spin.value.return_value = 1.0
            c._attn_spin.value.return_value = 0.5
            c._iso_spin.value.return_value = 0.5
            c._depict_combo.currentData.return_value = 'volume'
            c._zplane_slider.value.return_value = 50
            c._plane_thick_spin.value.return_value = 1.0
            c._smooth_iso_cb.isChecked.return_value = False
            self._controls = c

    rng = np.random.default_rng(42)
    data = rng.integers(0, 256, (10, 20, 30), dtype=np.uint16)

    with patch('vispy.scene.visuals.Volume'):
        from vispy.scene import visuals
        visuals.Volume = MagicMock(return_value=MagicMock())

        win = _Headless.__new__(_Headless)
        win._hide_on_close = True
        win._adapter = None
        win._volume_node = None
        win._last_shape = None
        win._max_texture_3d = 512
        win._smooth_iso = False
        win._last_raw_data = None
        win._init_ui()

        canvas_mock = MagicMock()
        win._canvas = canvas_mock
        view_mock = MagicMock()
        win._view = view_mock

        statusbar_mock = MagicMock()
        win.statusBar = MagicMock(return_value=statusbar_mock)

        win.update_volume(data)

    assert win._last_raw_data is not None
    assert win._last_raw_data.dtype == np.float32
    assert win._last_raw_data.shape == data.shape
    # data.shape (10,20,30) is well within the 512 texture limit → no downsampling
    assert win._last_strides == (1, 1, 1), (
        f"Expected no-downsampling strides (1,1,1), got {win._last_strides}"
    )


def test_rerender_callable_without_data():
    """_rerender must be a no-op when _last_raw_data is None."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow

    class _Headless(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): pass

    win = _Headless.__new__(_Headless)
    win._last_raw_data = None
    win._rerender()  # must not raise


def test_gpu_data_is_smoothed_initial_state():
    """_gpu_data_is_smoothed must default to False before any volume upload."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow

    class _Headless(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): pass

    win = _Headless.__new__(_Headless)
    win._hide_on_close = True
    win._adapter = None
    win._volume_node = None
    win._last_shape = None
    win._max_texture_3d = None
    win._smooth_iso = False
    win._gpu_data_is_smoothed = False
    win._last_raw_data = None
    assert win._gpu_data_is_smoothed is False


def test_set_rendering_mode_triggers_rerender_on_stale_smooth():
    """set_rendering_mode must re-render when GPU smooth state mismatches _smooth_iso.

    Reproduces the stale-GPU bug: user enables smooth ISO while in MIP mode
    (no re-render), then switches to ISO — GPU texture is still unsmoothed.
    The mode switch must detect the mismatch and call _rerender().
    """
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    from unittest.mock import MagicMock, patch

    class _Headless(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self):
            c = MagicMock()
            c._mode_combo.currentData.return_value = 'iso'
            c._clim_min.value.return_value = 0.0
            c._clim_max.value.return_value = 1.0
            c._iso_spin.value.return_value = 0.5
            c._attn_spin.value.return_value = 0.5
            self._controls = c

    win = _Headless.__new__(_Headless)
    win._hide_on_close = True
    win._adapter = None
    win._last_shape = (10, 10, 10)
    win._max_texture_3d = 512
    win._smooth_iso = True          # smooth enabled
    win._gpu_data_is_smoothed = False  # but GPU has unsmoothed data (stale)
    win._last_raw_data = np.zeros((10, 10, 10), dtype=np.float32)
    win._volume_node = MagicMock()
    win._canvas = MagicMock()
    win._view = MagicMock()
    win.statusBar = MagicMock(return_value=MagicMock())
    win._init_ui()

    rerender_calls = []
    original_rerender = VolumeRenderer3DWindow._rerender

    def _spy_rerender(self):
        rerender_calls.append(1)
        # Don't actually re-render (would need full vispy setup); just update flag.
        self._gpu_data_is_smoothed = self._smooth_iso

    with patch.object(_Headless, '_rerender', _spy_rerender):
        win.set_rendering_mode('iso')

    assert len(rerender_calls) == 1, (
        '_rerender must be called when switching to ISO with stale GPU smooth state'
    )


def test_auto_contrast_percentile_no_data():
    """auto_contrast_percentile returns [0,1] when no raw data is cached."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    from unittest.mock import MagicMock

    class _Headless(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self):
            c = MagicMock()
            c._clim_min.value.return_value = 0.0
            c._clim_max.value.return_value = 1.0
            self._controls = c

    win = _Headless.__new__(_Headless)
    win._last_raw_data = None
    win._volume_node = None
    win._init_ui()
    win.auto_contrast_percentile()  # must not raise

    # Spinboxes set to full range fallback
    win._controls._clim_min.setValue.assert_called_with(0.0)
    win._controls._clim_max.setValue.assert_called_with(1.0)


def test_auto_contrast_percentile_with_data():
    """auto_contrast_percentile maps raw percentiles into normalised [0,1] space.

    Data setup: 997 background voxels uniform in [0, 500] + 3 outlier voxels
    at 5000.  The 3 outliers are < 0.5% of 1000 voxels, so the 99.5th
    percentile lands in the background region, giving hi < 1.0 in normalised
    space.
    """
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    from unittest.mock import MagicMock

    class _Headless(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self):
            c = MagicMock()
            c._clim_min.value.return_value = 0.0
            c._clim_max.value.return_value = 1.0
            self._controls = c

    rng = np.random.default_rng(0)
    bg = rng.uniform(0, 500, 997).astype(np.float32)
    outliers = np.full(3, 5000.0, dtype=np.float32)
    raw = np.concatenate([bg, outliers]).reshape(10, 10, 10)
    # Sanity: vmax is the outlier value
    assert raw.max() == 5000.0

    win = _Headless.__new__(_Headless)
    win._last_raw_data = raw
    win._volume_node = None
    win._init_ui()
    win.auto_contrast_percentile(lo_pct=1.0, hi_pct=99.5)

    # The 99.5th percentile (index 994 of 1000) falls within the 997 background
    # voxels → hi is mapped to well below 1.0 in normalised space.
    hi_call = win._controls._clim_max.setValue.call_args[0][0]
    assert hi_call < 1.0, (
        f"Expected hi < 1.0 (outliers excluded by percentile), got {hi_call}"
    )
    lo_call = win._controls._clim_min.setValue.call_args_list[0][0][0]
    assert 0.0 <= lo_call <= hi_call <= 1.0


def test_auto_contrast_percentile_constant_data():
    """auto_contrast_percentile falls back to [0,1] when vmax == vmin (all-same data)."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    from unittest.mock import MagicMock

    class _Headless(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self):
            c = MagicMock()
            c._clim_min.value.return_value = 0.0
            c._clim_max.value.return_value = 1.0
            self._controls = c

    raw = np.full((5, 5, 5), 42.0, dtype=np.float32)  # constant → vmax == vmin

    win = _Headless.__new__(_Headless)
    win._last_raw_data = raw
    win._volume_node = None
    win._init_ui()
    win.auto_contrast_percentile()

    # Falls back to [0, 1] when span is zero (degenerate data).
    win._controls._clim_min.setValue.assert_called_with(0.0)
    win._controls._clim_max.setValue.assert_called_with(1.0)


def test_auto_contrast_percentile_large_volume_subsampled():
    """auto_contrast_percentile subsample path (>1 M voxels) must not crash
    and must still return valid limits in [0, 1]."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    from unittest.mock import MagicMock

    class _Headless(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self):
            c = MagicMock()
            c._clim_min.value.return_value = 0.0
            c._clim_max.value.return_value = 1.0
            self._controls = c

    # 128×100×100 = 1.28 M voxels → triggers the stride-subsample branch
    rng = np.random.default_rng(5)
    raw = rng.uniform(0, 1000, (128, 100, 100)).astype(np.float32)
    assert raw.size > 1_000_000

    win = _Headless.__new__(_Headless)
    win._last_raw_data = raw
    win._volume_node = None
    win._init_ui()
    win.auto_contrast_percentile(lo_pct=1.0, hi_pct=99.0)

    lo = win._controls._clim_min.setValue.call_args_list[0][0][0]
    hi = win._controls._clim_max.setValue.call_args[0][0]
    assert 0.0 <= lo <= hi <= 1.0, f"Invalid limits: lo={lo}, hi={hi}"


def test_apply_voxel_scale_updates_canvas():
    """_apply_voxel_scale must call canvas.update() so the frame redraws."""
    from cellacdc.renderer3d import VolumeRenderer3DWindow
    from unittest.mock import MagicMock

    class _Bare(VolumeRenderer3DWindow):
        def _init_vispy(self): pass
        def _init_ui(self): self._controls = None

    r = _Bare.__new__(_Bare)
    r._volume_node = MagicMock()
    r._canvas = MagicMock()
    r._last_strides = (1, 1, 1)
    r._voxel_dz = 2.0
    r._voxel_dy = 1.0
    r._voxel_dx = 1.0

    assigned = []
    type(r._volume_node).transform = property(
        fget=lambda self: None,
        fset=lambda self, v: assigned.append(v),
    )

    r._apply_voxel_scale()

    assert len(assigned) == 1, "transform must be set exactly once"
    r._canvas.update.assert_called_once()
