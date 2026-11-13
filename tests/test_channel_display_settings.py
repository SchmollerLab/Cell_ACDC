"""Per-user display settings of the channels (`gui_overlay`): the LUT of
each channel (displayed image and overlays, matched by name) and the overlay
transparency modes, saved in the settings CSV and restored on loading."""
import json
import sys
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QApplication

from cellacdc import measurements_viewer
from cellacdc.gui_overlay import (
    ALPHA_ENCODED_INTENSITY_SETTING, TRANSPARENCY_SETTING,
    channel_lut_setting_name, drop_channel_lut_settings,
    gradient_setting_value, overlay_gradient_for_background,
    parse_gradient_setting,
)
from cellacdc.measurements_viewer import MeasurementsViewerWin

BLACK = (0, 0, 0, 255)
WHITE = (255, 255, 255, 255)
GREEN = (0, 255, 0, 255)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def ticks_of(gradientItem):
    return [
        (round(pos, 6), tuple(tick.color.getRgb()))
        for tick, pos in gradientItem.listTicks()
    ]


def preset_ticks(name):
    item = pg.GradientEditorItem()
    item.loadPreset(name)
    return ticks_of(item)


# --- Serialization helpers --------------------------------------------------
def test_gradient_setting_round_trip(app):
    item = pg.GradientEditorItem()
    item.loadPreset('viridis')
    value = gradient_setting_value(item, inverted=True)
    state, inverted = parse_gradient_setting(value)
    assert inverted is True
    assert state['mode'] == 'rgb' and state['ticksVisible']
    restored = pg.GradientEditorItem()
    restored.restoreState(state)
    assert ticks_of(restored) == ticks_of(item)
    # Without a mode no flag is stored
    assert parse_gradient_setting(gradient_setting_value(item))[1] is None


def test_hidden_ticks_are_stored_from_the_ticks(app):
    item = pg.GradientEditorItem()
    item.loadPreset('grey')
    assert json.loads(gradient_setting_value(item))['ticksVisible']
    item.showTicks(False)
    assert not json.loads(gradient_setting_value(item))['ticksVisible']


@pytest.mark.parametrize('value', [
    'nan', 'not json', '[]', '{"ticks": [[0, [0, 0, 0]]]}',
    '{"ticks": [[0, [0, 0, 0]], [2, [1, 1, 1]]]}',
    '{"ticks": [[0, [0, 0, 0]], [1, [1, 1, 300]]]}',
    '{"ticks": [[0, [0, 0]], [1, [1, 1, 1]]]}',
    '{"mode": "cmyk", "ticks": [[0, [0, 0, 0]], [1, [1, 1, 1]]]}',
    float('nan'),
])
def test_invalid_gradient_settings_are_ignored(value):
    assert parse_gradient_setting(value) is None


@pytest.mark.parametrize('ticks, saved_inverted, inverted, expected_start', [
    # Background -> channel color built in the other mode: current background
    ([(0, BLACK), (1, GREEN)], False, True, WHITE),
    ([(0, WHITE), (1, GREEN)], True, False, BLACK),
    # Same mode, or unknown mode: kept
    ([(0, BLACK), (1, GREEN)], False, False, BLACK),
    ([(0, WHITE), (1, GREEN)], True, True, WHITE),
    ([(0, BLACK), (1, GREEN)], None, True, BLACK),
    # A start color chosen by the user (not the background of the mode the
    # gradient was built for) is kept
    ([(0, WHITE), (1, GREEN)], False, False, WHITE),
    ([(0, WHITE), (1, GREEN)], False, True, WHITE),
    ([(0, BLACK), (1, GREEN)], True, False, BLACK),
    # A grey gradient (its color is the background) is kept as it is
    ([(0, BLACK), (1, WHITE)], False, True, BLACK),
    ([(0, WHITE), (1, BLACK)], True, False, WHITE),
    # Other gradients are kept
    ([(0, (10, 0, 0, 255)), (1, GREEN)], False, True, (10, 0, 0, 255)),
    ([(0, BLACK), (0.5, GREEN), (1, WHITE)], False, True, BLACK),
])
def test_overlay_background_follows_invert_mode(
        ticks, saved_inverted, inverted, expected_start
    ):
    state = {'mode': 'rgb', 'ticks': ticks, 'ticksVisible': True}
    result = overlay_gradient_for_background(state, saved_inverted, inverted)
    assert result['ticks'][0][1] == expected_start
    assert result['ticks'][1:] == ticks[1:]


def test_drop_channel_lut_settings():
    df = pd.DataFrame(
        {'value': ['a', 'b', 'c', 'd']},
        index=pd.Index(
            ['image_lut_GFP', 'overlay_lut_GFP', 'GFP_rgb', 'img_cmap'],
            name='setting'
        )
    )
    assert list(drop_channel_lut_settings(df).index) == ['GFP_rgb', 'img_cmap']
    # The keys stay out of the global image colormap rows (`img_cmap_*`)
    for overlay in (False, True):
        key = channel_lut_setting_name('GFP', overlay=overlay)
        assert 'img_cmap' not in key and 'lab_cmap' not in key


# --- Measurements viewer ----------------------------------------------------
@pytest.fixture
def configuration_path(tmp_path):
    import tifffile

    images = tmp_path / 'Position_1' / 'Images'
    images.mkdir(parents=True)
    image = np.arange(900, dtype=np.uint16).reshape(30, 30)
    tifffile.imwrite(images / 'test_phase.tif', image)
    tifffile.imwrite(images / 'test_GFP.tif', image * 2)
    tifffile.imwrite(images / 'test_mCherry.tif', image * 3)
    labels = np.zeros((30, 30), dtype=np.uint16)
    labels[2:5, 3:7] = 1
    np.savez_compressed(images / 'test_segm_custom.npz', labels)
    pd.DataFrame({
        'frame_i': [0], 'Cell_ID': [1], 'area': [12.0], 'intensity': [20.0]
    }).to_csv(images / 'test_acdc_output_custom.csv', index=False)
    configuration = {
        'version': 2, 'images_paths': [str(images)], 'channel': 'phase',
        'segm_endname': 'segm_custom', 'overlay_channels': ['GFP'],
        'visible_channels': ['GFP'], 'measurements_mode': 1,
        'time_course_columns': ['intensity'],
        'measurements_axes': {'X': 'intensity', 'Y': 'area'},
        'normalize_measurements': True
    }
    path = tmp_path / 'real.json'
    path.write_text(json.dumps(configuration))
    return str(path)


@pytest.fixture
def open_viewer(app, monkeypatch, configuration_path):
    """Open viewers (sharing the test's settings file) with the GFP overlay
    configuration loaded; fails on exceptions raised inside Qt slots."""
    errors = []
    monkeypatch.setattr(
        sys, 'excepthook', lambda *exc_info: errors.append(exc_info)
    )
    windows = []

    def _open(filepath=configuration_path):
        window = MeasurementsViewerWin()
        window._startupConfigurationPrompted = True
        windows.append(window)
        configErrors = []
        window.configurationError = lambda *args: configErrors.append(args)
        window.selectConfigurationInput = (
            lambda config: window.configurationInputOptions(config)[0]
        )
        assert window.loadConfiguration(filepath=filepath)
        assert not configErrors
        app.processEvents()
        return window

    yield _open
    for window in windows:
        window.deleteLater()
    app.processEvents()
    assert not errors


def saved_settings(window):
    return pd.read_csv(window.settings_csv_path, index_col='setting')


def overlay_lut(window, channel='GFP'):
    return window.overlayLayersItems[channel][1]


def test_channel_luts_are_restored_by_channel_name(app, open_viewer):
    first = open_viewer()
    defaultOverlayTicks = ticks_of(overlay_lut(first).gradient)
    first.imgGrad.gradient.loadPreset('viridis')
    overlay_lut(first).gradient.loadPreset('plasma')
    app.processEvents()
    saved = saved_settings(first)
    assert json.loads(saved.at['image_lut_phase', 'value'])['inverted'] is False
    assert 'overlay_lut_GFP' in saved.index
    # The displayed channel's LUT is not stored as an overlay LUT
    assert 'overlay_lut_phase' not in saved.index
    assert 'image_lut_GFP' not in saved.index

    second = open_viewer()
    assert ticks_of(second.imgGrad.gradient) == preset_ticks('viridis')
    assert ticks_of(overlay_lut(second).gradient) == preset_ticks('plasma')
    # The channel toolbutton shows the restored colors
    toolbutton = second.allOverlayToolbuttons['GFP']
    lut = overlay_lut(second).gradient.getLookupTable(256)
    assert toolbutton._qcolors[-1] == pg.mkColor(lut[-1])
    assert defaultOverlayTicks != preset_ticks('plasma')


def test_image_lut_is_inverted_with_black_and_white(app, open_viewer):
    first = open_viewer()
    first.imgGrad.gradient.loadPreset('viridis')
    app.processEvents()
    # Inverting inverts the colormap picked from the menu (`lastGradient`)
    first.invertBwAction.setChecked(True)
    app.processEvents()
    inverted = ticks_of(first.imgGrad.gradient)
    assert inverted != preset_ticks('viridis')
    saved = json.loads(saved_settings(first).at['image_lut_phase', 'value'])
    assert saved['inverted'] is True

    # Started inverted: the restored LUT is inverted once
    second = open_viewer()
    assert second.invertBwAction.isChecked()
    assert ticks_of(second.imgGrad.gradient) == inverted
    second.invertBwAction.setChecked(False)
    app.processEvents()
    assert ticks_of(second.imgGrad.gradient) == preset_ticks('viridis')

    third = open_viewer()
    assert ticks_of(third.imgGrad.gradient) == preset_ticks('viridis')


def test_overlay_background_follows_saved_invert_mode(app, open_viewer):
    first = open_viewer()
    # Edit the tick (the channel color, `GFP_rgb`, stays the default)
    gradient = overlay_lut(first).gradient
    lastTick = gradient.listTicks()[-1][0]
    gradient.setTickColor(lastTick, QColor(0, 200, 0))
    app.processEvents()
    assert ticks_of(gradient)[0][1] == BLACK
    saved = saved_settings(first)
    assert json.loads(saved.at['overlay_lut_GFP', 'value'])['inverted'] is False
    assert 'GFP_rgb' not in saved.index

    first.invertBwAction.setChecked(True)
    second = open_viewer()
    assert ticks_of(overlay_lut(second).gradient) == [
        (0, WHITE), (1, (0, 200, 0, 255))
    ]


def test_overlay_start_color_chosen_by_the_user_is_kept(app, open_viewer):
    first = open_viewer()
    gradient = overlay_lut(first).gradient
    firstTick = gradient.listTicks()[0][0]
    gradient.setTickColor(firstTick, QColor(*WHITE))
    app.processEvents()
    for invert in (False, True):
        first.invertBwAction.setChecked(invert)
        window = open_viewer()
        assert ticks_of(overlay_lut(window).gradient)[0][1] == WHITE


def test_inverting_keeps_hidden_ticks_of_menu_colormaps(app, open_viewer):
    first = open_viewer()
    gradient = first.imgGrad.gradient
    # Non-preset colormaps from the menu hide their (many) ticks
    gradient.colorMapMenuClicked(pg.colormap.get('CET-L1'))
    app.processEvents()

    def ticksVisible(window):
        gradient = window.imgGrad.gradient
        return any(tick.isVisibleTo(gradient) for tick in gradient.ticks)

    def savedTicksVisible(window):
        value = saved_settings(window).at['image_lut_phase', 'value']
        return json.loads(value)['ticksVisible']

    assert not ticksVisible(first) and not savedTicksVisible(first)
    first.invertBwAction.setChecked(True)
    app.processEvents()
    assert not ticksVisible(first) and not savedTicksVisible(first)
    # Started inverted (first load inverts the restored LUT)
    second = open_viewer()
    app.processEvents()
    assert not ticksVisible(second) and not savedTicksVisible(second)


def test_restore_default_settings_resets_channel_luts(app, open_viewer):
    first = open_viewer()
    first.imgGrad.gradient.loadPreset('viridis')
    overlay_lut(first).gradient.loadPreset('plasma')
    app.processEvents()
    first.restoreDefaultSettings()
    app.processEvents()
    saved = saved_settings(first)
    assert not any(
        str(name).startswith(('image_lut_', 'overlay_lut_'))
        for name in saved.index
    )
    assert ticks_of(first.imgGrad.gradient) == preset_ticks('grey')
    color = overlay_lut(first).overlayColorButton.color().getRgb()
    assert ticks_of(overlay_lut(first).gradient) == [(0, BLACK), (1, color)]
    # Changes after restoring the defaults are saved again
    first.imgGrad.gradient.loadPreset('plasma')
    app.processEvents()
    assert 'image_lut_phase' in saved_settings(first).index


def test_restore_default_settings_when_inverted(app, open_viewer):
    first = open_viewer()
    first.labelsGrad.colorButton.setColor((1, 2, 3))
    first.df_settings.at['labels_bkgrColor', 'value'] = (1, 2, 3)
    first.invertBwAction.setChecked(True)
    first.restoreDefaultSettings()
    app.processEvents()
    assert not first.invertBwAction.isChecked()
    assert not first.imgGrad.isInverted
    assert first.labelsGrad.colorButton.color().getRgb()[:3] == (25, 25, 25)
    saved = saved_settings(first)
    assert saved.at['is_bw_inverted', 'value'] == 'No'
    assert saved.at['contLineWeight', 'value'] == '1'


def test_lut_changes_while_loading_another_channel_are_not_saved(
        app, open_viewer
    ):
    first = open_viewer()
    first.imgGrad.gradient.loadPreset('viridis')
    app.processEvents()
    # A gradient change of imgGrad while another channel is set as displayed
    # (before its LUT was restored) does not overwrite that channel's LUT
    first.user_ch_name = 'GFP'
    first.imgGrad.gradient.loadPreset('plasma')
    app.processEvents()
    saved = saved_settings(first)
    assert 'image_lut_GFP' not in saved.index
    state, _ = parse_gradient_setting(saved.at['image_lut_phase', 'value'])
    assert [tuple(t[1]) for t in state['ticks']] == [
        t[1] for t in preset_ticks('viridis')
    ]


def test_transparency_modes_are_saved_on_click_and_restored(
        app, open_viewer, monkeypatch
    ):
    first = open_viewer()
    toolbar = first.overlayToolbar
    toolbar.alphaEncodedIntensityCheckbox.click()
    assert toolbar.isTransparent() and toolbar.isAlphaEncodedIntensity()
    saved = saved_settings(first)
    assert saved.at[TRANSPARENCY_SETTING, 'value'] == 'True'
    assert saved.at[ALPHA_ENCODED_INTENSITY_SETTING, 'value'] == 'True'
    # Programmatic changes (e.g., exports) are not saved
    toolbar.setTransparent(False)
    assert not toolbar.isAlphaEncodedIntensity()
    assert saved_settings(first).at[TRANSPARENCY_SETTING, 'value'] == 'True'

    second = open_viewer()
    toolbar = second.overlayToolbar
    assert toolbar.isTransparent() and toolbar.isAlphaEncodedIntensity()
    # The mode is applied as when checking it by hand
    assert second.img1.opacity() < 0.01
    assert second.rgbaImg1.isVisible()
    imageItem, lutItem, alphaSB = second.overlayLayersItems['GFP'][:3]
    assert imageItem.opacity() < 0.01

    # Turning the overlay on again does not stack the redraw connections
    second.overlayButton.setChecked(False)
    second.overlayButton.setChecked(True)
    redraws = []
    monkeypatch.setattr(
        second, 'setOverlayImages', lambda *args, **kwargs: redraws.append(1)
    )
    alphaSB.setValue(alphaSB.value() + 1)
    assert len(redraws) == 1
    redraws.clear()
    second.imgGrad.sigLevelsChanged.emit(second.imgGrad)
    assert len(redraws) == 1

    # Unchecking by hand saves both modes off and disconnects the redraws
    toolbar.transparencyCheckbox.click()
    saved = saved_settings(second)
    assert saved.at[TRANSPARENCY_SETTING, 'value'] == 'False'
    assert saved.at[ALPHA_ENCODED_INTENSITY_SETTING, 'value'] == 'False'
    redraws.clear()
    second.imgGrad.sigLevelsChanged.emit(second.imgGrad)
    assert not redraws
    # Back to layer opacities (only GFP is visible in the configuration)
    assert not second.rgbaImg1.isVisible()
    assert imageItem.opacity() > 0.99


def test_restored_transparency_with_hidden_base_and_two_overlays(
        app, open_viewer, configuration_path, tmp_path
    ):
    with open(configuration_path) as file:
        configuration = json.load(file)
    configuration['overlay_channels'] = ['GFP', 'mCherry']
    configuration['visible_channels'] = ['GFP', 'mCherry']
    twoOverlaysPath = tmp_path / 'two_overlays.json'
    twoOverlaysPath.write_text(json.dumps(configuration))
    first = open_viewer(str(twoOverlaysPath))
    first.overlayToolbar.transparencyCheckbox.click()
    assert first.overlayToolbar.isTransparent()

    second = open_viewer(str(twoOverlaysPath))
    assert second.overlayToolbar.isTransparent()
    assert not second.baseLayerToolbutton.isChecked()
    rgba = second.rgbaImg1.image
    assert rgba is not None and rgba.shape[:2] == (30, 30)
    second.updateImages()


def test_restored_transparency_needs_the_overlay(
        app, open_viewer, configuration_path
    ):
    first = open_viewer()
    first.overlayToolbar.transparencyCheckbox.click()
    first.overlayButton.setChecked(False)
    # Loading restores the checkbox without hiding the image while the
    # overlay is off
    with open(configuration_path) as file:
        paths = json.load(file)['images_paths']
    first.setData(measurements_viewer.load_positions(paths, 'phase'))
    assert first.overlayToolbar.isTransparent()
    assert not first.overlayButton.isChecked()
    assert first.img1.opacity() > 0.99
