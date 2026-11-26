"""LUTs saved per channel name and saved check states of the overlay modes
(`gui_navigation.LutGuiElements`, `gui_overlay.OverlayChannelsGuiElements`),
shared by the main GUI and the measurements viewer (tested through the
viewer, which loads real data cheaply; the main GUI's own call sites are
checked on its source)."""
import os
import ast
import json
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pytest
from qtpy.QtGui import QColor
from qtpy.QtWidgets import QAbstractSlider, QApplication

from cellacdc import colors, gui, gui_navigation, gui_overlay, widgets
from cellacdc import measurements_viewer
from cellacdc.measurements_viewer import MeasurementsViewerWin
from cellacdc.gui_navigation import (
    LutGuiElements, default_image_lut, image_lut_setting_name,
    overlay_lut_setting_name, overlay_alpha_setting_name
)
from tests.test_measurement_viewer import (
    make_overlay_position, make_timelapse_position,
    real_viewer  # noqa: F401 (fixture)
)

GREY_TICKS = default_image_lut()['ticks']
PRESETS = pg.graphicsItems.GradientEditorItem.Gradients
CHANNEL_SETTING_PREFIXES = ('image_lut_', 'overlay_lut_', 'overlay_alpha_slider_')


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def load(window, images, channel='phase'):
    window.setData(measurements_viewer.load_positions([images], channel))


def new_viewer():
    window = MeasurementsViewerWin()
    window._startupConfigurationPrompted = True
    return window


def ticks(lutItem):
    return colors._gradient_ticks_rgba(lutItem.gradient.saveState()['ticks'])


def visible_ticks(lutItem):
    gradient = lutItem.gradient
    return [tick.isVisibleTo(gradient) for tick in gradient.ticks]


def preset_ticks(name, invertedBy=None):
    """Ticks of the pyqtgraph preset `name`, inverted as `Invert black/white`
    does with the LUT item `invertedBy`."""
    gradient = PRESETS[name]
    if invertedBy is not None:
        gradient = invertedBy.invertGradient(gradient)
    return colors._gradient_ticks_rgba(gradient['ticks'])


def read_settings(window):
    """Rows of `settings.csv` (whole-table writes of each window)"""
    return pd.read_csv(window.settings_csv_path, index_col='setting')


def read_store(window):
    """Channel display settings (their own file)"""
    return gui_navigation.read_channel_display_settings(
        window.channelDisplaySettingsPath()
    )


def saved_lut(store, setting):
    return colors.gradient_state_from_setting(store[setting])


def assert_no_channel_settings_in_csv(window):
    for setting in read_settings(window).index:
        assert not str(setting).startswith(CHANNEL_SETTING_PREFIXES)
        assert setting not in (
            gui_overlay.OVERLAY_TRANSPARENCY_SETTING,
            gui_overlay.OVERLAY_ALPHA_ENCODED_SETTING
        )


# --- Serialization and store (no window) -----------------------------------
def test_gradient_setting_round_trip_and_invalid_values():
    state = {
        'mode': 'rgb', 'ticksVisible': False,
        'ticks': [(1.0, (255, 0, 0, 255)), (0.0, (0, 0, 0))],
    }
    setting = colors.gradient_state_to_setting(state, bw_inverted=True)
    # Saved as a JSON object
    setting = json.loads(json.dumps(setting))
    restored, bwInverted = colors.gradient_state_from_setting(setting)
    assert bwInverted is True
    assert restored['ticksVisible'] is False
    # Sorted by position, alpha added
    assert restored['ticks'] == [(0.0, (0, 0, 0, 255)), (1.0, (255, 0, 0, 255))]
    assert colors.are_gradients_equal(restored, state)
    for invalid in (
            'not a dict', json.dumps(setting), None, 1.5, [[0, [0, 0, 0]]],
            {'ticks': []}, {'ticks': [[1.5, [0, 0, 0]]]},
            {'ticks': [[0.5, [0, 0, 300]]]}, {'ticks': [[0.5, 'red']]},
            {'mode': 'cmyk', 'ticks': [[0, [0, 0, 0]]]},
        ):
        assert colors.gradient_state_from_setting(invalid) is None


def test_gradients_made_by_code_state_visible_ticks():
    # Else pyqtgraph hides the ticks of a LUT item that is not shown
    assert default_image_lut()['ticksVisible'] is True
    assert colors.get_pg_gradient([(0, 0, 0), (0, 255, 0)])['ticksVisible']


def test_drop_channel_lut_settings_keeps_other_settings():
    df = pd.DataFrame(
        {'value': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']},
        index=pd.Index([
            image_lut_setting_name('GFP'), overlay_lut_setting_name('GFP'),
            overlay_alpha_setting_name('GFP'),
            'img_cmap_tick0_pos', 'GFP_rgb', 'how_rescale_intensities_GFP',
            'is_bw_inverted', gui_overlay.OVERLAY_ALPHA_ENCODED_SETTING
        ], name='setting')
    )
    kept = gui_navigation.drop_channel_lut_settings(df)
    assert list(kept.index) == [
        'GFP_rgb', 'how_rescale_intensities_GFP', 'is_bw_inverted',
        gui_overlay.OVERLAY_ALPHA_ENCODED_SETTING
    ]


def test_channel_display_settings_store(tmp_path):
    filepath = gui_navigation.channel_display_settings_path(
        str(tmp_path / 'settings.csv')
    )
    assert filepath == str(tmp_path / 'channel_display_settings.json')
    read = gui_navigation.read_channel_display_settings
    update = gui_navigation.update_channel_display_settings
    # Missing file: empty, nothing written when nothing changes
    assert read(filepath) == {}
    assert not update(filepath, {'a': None})
    assert not os.path.exists(filepath)

    assert update(filepath, {'image_lut_GFP': {'x': 1}, 'b': 0.5})
    # Written by another window meanwhile: kept
    otherWindowSettings = {**read(filepath), 'overlay_lut_GFP': [1]}
    with open(filepath, 'w', encoding='utf-8') as file:
        json.dump(otherWindowSettings, file)
    assert update(filepath, {'b': None, 'c': True})
    assert read(filepath) == {
        'image_lut_GFP': {'x': 1}, 'overlay_lut_GFP': [1], 'c': True
    }
    assert update(filepath, remove_prefixes=['image_lut_', 'overlay_lut_'])
    assert read(filepath) == {'c': True}
    # Only the file itself in the folder (temporary file replaced it)
    assert os.listdir(tmp_path) == ['channel_display_settings.json']

    # Invalid files: empty (logged), never an error
    logger = MagicMock()
    for text in ('{not json', '[1, 2]'):
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(text)
        assert read(filepath, logger=logger) == {}
    assert logger.warning.call_count == 2
    assert update(filepath, {'c': False}, logger=logger)
    assert read(filepath) == {'c': False}
    # Cannot be written (a folder with the file's name): logged
    logger = MagicMock()
    folder = tmp_path / 'folder'
    (folder / 'channel_display_settings.json').mkdir(parents=True)
    assert not update(
        str(folder / 'channel_display_settings.json'), {'c': 1},
        logger=logger
    )
    assert logger.warning.called
    assert os.listdir(folder) == ['channel_display_settings.json']


def test_channel_display_settings_are_in_the_isolated_settings_folder(
        app, real_viewer, _isolate_user_settings
    ):
    # Next to the settings.csv of the test (tests/conftest.py)
    filepath = real_viewer.channelDisplaySettingsPath()
    assert os.path.dirname(filepath) == _isolate_user_settings


# --- Displayed (base) channel ----------------------------------------------
@pytest.mark.slow
def test_base_lut_is_saved_and_restored_per_channel(
        app, real_viewer, tmp_path
    ):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    assert ticks(real_viewer.imgGrad) == GREY_TICKS
    real_viewer.imgGrad.gradient.loadPreset('viridis')
    state, bwInverted = saved_lut(
        read_store(real_viewer), image_lut_setting_name('phase')
    )
    assert state['ticks'] == preset_ticks('viridis')
    assert bwInverted is False

    # Another channel without a saved LUT: grey, not the previous LUT
    load(real_viewer, images, 'GFP')
    assert ticks(real_viewer.imgGrad) == GREY_TICKS
    assert image_lut_setting_name('GFP') not in read_store(real_viewer)
    real_viewer.imgGrad.gradient.loadPreset('magma')

    load(real_viewer, images, 'phase')
    assert ticks(real_viewer.imgGrad) == preset_ticks('viridis')

    # Also in a new window (from the settings file)
    other = new_viewer()
    try:
        load(other, images, 'GFP')
        assert ticks(other.imgGrad) == preset_ticks('magma')
    finally:
        other.deleteLater()
    # Not in the whole-table writes of `settings.csv`
    assert_no_channel_settings_in_csv(real_viewer)
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_base_lut_follows_invert_black_white(app, real_viewer, tmp_path):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    real_viewer.imgGrad.gradient.loadPreset('viridis')
    load(real_viewer, images, 'GFP')
    real_viewer.imgGrad.gradient.loadPreset('magma')
    # Inverts the LUT chosen in the menu (not the stale `lastGradient`)
    real_viewer.invertBwAction.setChecked(True)
    assert ticks(real_viewer.imgGrad) == preset_ticks('magma', invertedBy=real_viewer.imgGrad)
    state, bwInverted = saved_lut(
        read_store(real_viewer), image_lut_setting_name('GFP')
    )
    assert bwInverted is True
    assert state['ticks'] == preset_ticks('magma', invertedBy=real_viewer.imgGrad)
    # Saved with black/white not inverted: shown inverted
    load(real_viewer, images, 'phase')
    assert ticks(real_viewer.imgGrad) == preset_ticks('viridis', invertedBy=real_viewer.imgGrad)

    # First load of a new window with inverted black/white: the first-load
    # `invertBw` neither inverts the restored LUT again nor overwrites it
    # (`restoreImageChannelLutAfterLoading`, also used by the main GUI)
    other = new_viewer()
    try:
        assert other.invertBwAction.isChecked()
        load(other, images, 'phase')
        assert ticks(other.imgGrad) == preset_ticks('viridis', invertedBy=real_viewer.imgGrad)
        state, bwInverted = saved_lut(
            read_store(other), image_lut_setting_name('phase')
        )
        assert bwInverted is False
        assert state['ticks'] == preset_ticks('viridis')
        other.invertBwAction.setChecked(False)
        assert ticks(other.imgGrad) == preset_ticks('viridis')
    finally:
        other.deleteLater()
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_colormaps_with_hidden_ticks_are_saved_with_hidden_ticks(
        app, real_viewer, tmp_path, monkeypatch
    ):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    imgGrad = real_viewer.imgGrad
    assert visible_ticks(imgGrad) == [True, True]
    # Picked in the 'cet (local)' submenu of the LUT menu (`ColorMapMenu`
    # emits the colormap): pyqtgraph hides its 256 ticks
    cmap = pg.colormap.get('CET-L9')
    imgGrad.gradient.menu.sigColorMapTriggered.emit(cmap)
    assert len(visible_ticks(imgGrad)) == len(cmap.pos)
    assert not any(visible_ticks(imgGrad))
    setting = image_lut_setting_name('phase')
    assert read_store(real_viewer)[setting]['ticksVisible'] is False
    # Inverting keeps them hidden
    for checked in (True, False):
        real_viewer.invertBwAction.setChecked(checked)
        assert not any(visible_ticks(imgGrad))
    load(real_viewer, images, 'phase')
    assert len(visible_ticks(imgGrad)) == len(cmap.pos)
    assert not any(visible_ticks(imgGrad))
    # A channel restored with visible ticks: ticks can be added again
    # (pyqtgraph's `showTicks(True)` does not allow it after hidden ticks)
    load(real_viewer, images, 'GFP')
    assert visible_ticks(imgGrad) == [True, True]
    assert imgGrad.gradient.allowAdd
    # Preset gradients of the menu show (and save) their ticks
    load(real_viewer, images, 'phase')
    presetCmap = pg.colormap.ColorMap(*zip(*PRESETS['magma']['ticks']))
    presetCmap.name = 'preset-gradient:magma'
    imgGrad.gradient.menu.sigColorMapTriggered.emit(presetCmap)
    assert all(visible_ticks(imgGrad)) and imgGrad.gradient.allowAdd
    assert read_store(real_viewer)[setting]['ticksVisible'] is True

    # Overlay LUT item (shown with the overlay on)
    monkeypatch.setattr(real_viewer, 'askSelectOverlayChannel', lambda: ['GFP'])
    real_viewer.overlayButton.setChecked(True)
    lutItem = real_viewer.overlayLayersItems['GFP'][1]
    assert lutItem.isVisible() and all(visible_ticks(lutItem))
    lutItem.gradient.menu.sigColorMapTriggered.emit(cmap)
    assert not any(visible_ticks(lutItem))
    setting = overlay_lut_setting_name('GFP')
    assert read_store(real_viewer)[setting]['ticksVisible'] is False
    load(real_viewer, images, 'phase')
    lutItem = real_viewer.overlayLayersItems['GFP'][1]
    assert len(visible_ticks(lutItem)) == len(cmap.pos)
    assert not any(visible_ticks(lutItem))
    assert not real_viewer.slotErrors


# --- Overlay channels ------------------------------------------------------
@pytest.mark.slow
def test_overlay_lut_is_saved_per_channel_and_follows_the_color(
        app, real_viewer, tmp_path
    ):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    lutItem = real_viewer.overlayLayersItems['GFP'][1]
    assert ticks(lutItem)[0][1] == (0, 0, 0, 255)
    lutItem.gradient.loadPreset('magma')
    setting = overlay_lut_setting_name('GFP')
    state, _ = saved_lut(read_store(real_viewer), setting)
    assert state['ticks'] == preset_ticks('magma')

    load(real_viewer, images, 'phase')
    lutItem = real_viewer.overlayLayersItems['GFP'][1]
    assert ticks(lutItem) == preset_ticks('magma')
    # The toolbar button shows the restored LUT
    toolbutton = real_viewer.overlayLayersItems['GFP'][3]
    assert toolbutton._qcolors[-1].getRgb() == preset_ticks('magma')[-1][1]

    # Separate from the LUT of the channel as displayed image
    load(real_viewer, images, 'GFP')
    assert ticks(real_viewer.imgGrad) == GREY_TICKS

    # Choosing a color: the LUT follows the color again
    load(real_viewer, images, 'phase')
    lutItem = real_viewer.overlayLayersItems['GFP'][1]
    lutItem.overlayColorButton.setColor((255, 0, 0))
    expected = [(0.0, (0, 0, 0, 255)), (1.0, (255, 0, 0, 255))]
    assert ticks(lutItem) == expected
    assert setting not in read_store(real_viewer)
    # A channel display setting, not a row of `settings.csv`
    assert read_store(real_viewer)['GFP_rgb'] == '255_0_0'
    load(real_viewer, images, 'phase')
    assert ticks(real_viewer.overlayLayersItems['GFP'][1]) == expected
    assert 'GFP_rgb' not in read_settings(real_viewer).index
    assert_no_channel_settings_in_csv(real_viewer)
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_overlay_color_dialog_preview_and_cancel_keep_the_saved_lut(
        app, real_viewer, tmp_path
    ):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    lutItem = real_viewer.overlayLayersItems['GFP'][1]
    lutItem.gradient.loadPreset('magma')
    setting = overlay_lut_setting_name('GFP')
    red = [(0.0, (0, 0, 0, 255)), (1.0, (255, 0, 0, 255))]
    # The LUT item's own button and the window's "Overlay color..." button
    # (`mousePressColorButton`), whose dialog starts from another color
    real_viewer.overlayColorButton.channel = 'GFP'
    buttons = (lutItem.overlayColorButton, real_viewer.overlayColorButton)
    colorBefore = lutItem.overlayColorButton.color().getRgb()[:3]
    for button in buttons:
        button.selectColor()
        button.colorDialog.setCurrentColor(QColor(255, 0, 0))
        # Preview, not saved
        assert ticks(lutItem) == red
        assert setting in read_store(real_viewer)
        assert 'GFP_rgb' not in read_store(real_viewer)
        button.colorDialog.reject()
        assert ticks(lutItem) == preset_ticks('magma')
        assert lutItem.overlayColorButton.color().getRgb()[:3] == colorBefore
        assert setting in read_store(real_viewer)

    # OK: the color is saved and the LUT follows it
    button = lutItem.overlayColorButton
    button.selectColor()
    button.colorDialog.setCurrentColor(QColor(255, 0, 0))
    button.colorDialog.accept()
    assert ticks(lutItem) == red
    assert setting not in read_store(real_viewer)
    assert read_store(real_viewer)['GFP_rgb'] == '255_0_0'
    # Cancel now shows the chosen color
    button.selectColor()
    button.colorDialog.setCurrentColor(QColor(0, 0, 255))
    button.colorDialog.reject()
    assert ticks(lutItem) == red
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_overlay_color_saved_in_settings_csv_by_older_versions_is_read(
        app, real_viewer, tmp_path
    ):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    defaultColor = real_viewer.overlayColors['GFP']
    blue, red = (0, 0, 255), (255, 0, 0)
    assert defaultColor not in (blue, red)

    def overlayColor():
        load(real_viewer, images, 'phase')
        lutItem = real_viewer.overlayLayersItems['GFP'][1]
        rgb = real_viewer.overlayColors['GFP']
        assert lutItem.overlayColorButton.color().getRgb()[:3] == rgb
        assert ticks(lutItem) == [(0.0, (0, 0, 0, 255)), (1.0, (*rgb, 255))]
        return rgb

    # Row of `settings.csv` saved by an older version
    real_viewer.df_settings.at['GFP_rgb', 'value'] = '0_0_255'
    real_viewer.df_settings.to_csv(real_viewer.settings_csv_path)
    assert overlayColor() == blue
    # The channel display setting comes first
    real_viewer.saveChannelDisplaySettings({'GFP_rgb': '255_0_0'})
    assert overlayColor() == red
    # Invalid saved values: the default color
    real_viewer.df_settings = real_viewer.df_settings.drop(index='GFP_rgb')
    for value in ('red', '255_0', '0_0_256', [255, 0, 0], True):
        real_viewer.saveChannelDisplaySettings({'GFP_rgb': value})
        assert overlayColor() == defaultColor
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_overlay_color_saved_by_another_open_window_is_restored_and_kept(
        app, real_viewer, tmp_path
    ):
    images = make_overlay_position(tmp_path / 'exp')
    # Both open before anything is saved
    windowA, windowB = real_viewer, new_viewer()
    windowC = None
    red = [(0.0, (0, 0, 0, 255)), (1.0, (255, 0, 0, 255))]
    try:
        load(windowB, images, 'phase')
        lutItem = windowB.overlayLayersItems['GFP'][1]
        lutItem.gradient.loadPreset('magma')
        # Removes the saved LUT, shared with A
        lutItem.overlayColorButton.setColor((255, 0, 0))
        # A loads and updates its images (whole `settings.csv` writes)
        load(windowA, images, 'phase')
        windowA.updateAllImages()
        assert ticks(windowA.overlayLayersItems['GFP'][1]) == red
        assert read_store(windowA) == {'GFP_rgb': '255_0_0'}
        windowC = new_viewer()
        load(windowC, images, 'phase')
        assert ticks(windowC.overlayLayersItems['GFP'][1]) == red
    finally:
        windowB.deleteLater()
        if windowC is not None:
            windowC.deleteLater()
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_restore_default_settings_resets_the_channel_luts(
        app, real_viewer, tmp_path
    ):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    lutItem = real_viewer.overlayLayersItems['GFP'][1]
    lutItem.overlayColorButton.setColor((255, 0, 0))
    colorTicks = ticks(lutItem)
    assert colorTicks == [(0.0, (0, 0, 0, 255)), (1.0, (255, 0, 0, 255))]
    real_viewer.imgGrad.gradient.loadPreset('viridis')
    lutItem.gradient.loadPreset('magma')
    real_viewer.saveOverlayModes()

    real_viewer.restoreDefaultSettings()

    assert ticks(real_viewer.imgGrad) == GREY_TICKS
    # Back to the overlay color (not to the LUT of the displayed image,
    # `restoreState` of the overlay)
    assert ticks(lutItem) == colorTicks
    # The overlay modes and the overlay color are kept
    assert set(read_store(real_viewer)) == {
        gui_overlay.OVERLAY_TRANSPARENCY_SETTING,
        gui_overlay.OVERLAY_ALPHA_ENCODED_SETTING, 'GFP_rgb'
    }
    # Changes after the reset are saved again
    real_viewer.imgGrad.gradient.loadPreset('magma')
    lutItem.gradient.loadPreset('viridis')
    store = read_store(real_viewer)
    assert image_lut_setting_name('phase') in store
    assert overlay_lut_setting_name('GFP') in store
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_restore_default_settings_keeps_the_ticks_of_hidden_overlay_luts(
        app, real_viewer, tmp_path, monkeypatch
    ):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    lutItem = real_viewer.overlayLayersItems['GFP'][1]
    # Overlay off: the overlay LUT item is hidden
    assert not lutItem.isVisible()
    real_viewer.restoreDefaultSettings()
    monkeypatch.setattr(real_viewer, 'askSelectOverlayChannel', lambda: ['GFP'])
    real_viewer.overlayButton.setChecked(True)
    assert lutItem.isVisible()
    assert visible_ticks(lutItem) == [True, True]
    assert lutItem.gradient.allowAdd
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_overlay_alpha_is_saved_and_restored_per_channel(
        app, real_viewer, tmp_path
    ):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    alphaSB = real_viewer.overlayLayersItems['GFP'][2]
    default = gui_overlay.ALPHA_SCROLLBAR_DEFAULT_VALUE
    assert alphaSB.value() == default
    setting = overlay_alpha_setting_name('GFP')

    # User step (arrow button, keyboard): saved
    alphaSB.triggerAction(QAbstractSlider.SliderAction.SliderSingleStepAdd)
    assert alphaSB.value() == default + 1
    expected = (default + 1)/alphaSB.maximum()
    assert read_store(real_viewer)[setting] == expected
    # `setValue` by code: not saved
    alphaSB.setValue(5)
    assert read_store(real_viewer)[setting] == expected
    # Dragging: saved once, on release
    alphaSB.setSliderDown(True)
    alphaSB.setSliderPosition(30)
    assert alphaSB.value() == 30
    assert read_store(real_viewer)[setting] == expected
    alphaSB.setSliderDown(False)
    assert read_store(real_viewer)[setting] == 0.75

    # Restored on the next load, also in another window
    load(real_viewer, images, 'phase')
    assert real_viewer.overlayLayersItems['GFP'][2].value() == 30
    other = new_viewer()
    try:
        load(other, images, 'phase')
        assert other.overlayLayersItems['GFP'][2].value() == 30
    finally:
        other.deleteLater()

    # Invalid or out-of-range saved values
    for value, restored in (
            ('not a number', default), (True, default),
            (float('nan'), default), (2, 40)
        ):
        real_viewer.saveChannelDisplaySettings({setting: value})
        load(real_viewer, images, 'phase')
        assert real_viewer.overlayLayersItems['GFP'][2].value() == restored

    # Restore default settings: removed and back to the default alpha
    real_viewer.saveChannelDisplaySettings({setting: 0.75})
    load(real_viewer, images, 'phase')
    alphaSB = real_viewer.overlayLayersItems['GFP'][2]
    real_viewer.restoreDefaultSettings()
    assert alphaSB.value() == default
    assert setting not in read_store(real_viewer)
    assert_no_channel_settings_in_csv(real_viewer)
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_overlay_opacities_follow_the_saved_alphas(
        app, real_viewer, tmp_path, monkeypatch
    ):
    import tifffile

    images = make_overlay_position(tmp_path / 'exp')
    tifffile.imwrite(
        os.path.join(images, 'test_RFP.tif'),
        np.ones((3, 30, 30), dtype=np.uint16)
    )
    real_viewer.saveChannelDisplaySettings({
        overlay_alpha_setting_name('GFP'): 0.9,
        overlay_alpha_setting_name('RFP'): 0.25,
    })
    # Shown: hidden toolbuttons are skipped by the opacity computations
    real_viewer.show()
    load(real_viewer, images, 'phase')
    monkeypatch.setattr(real_viewer, 'askSelectOverlayChannel', lambda: ['GFP'])

    def assert_opacities_follow_the_scrollbars():
        expected = real_viewer.getOpacitiesFromAlphaScrollbarValues()
        for channel, opacity in expected.items():
            if channel == real_viewer.user_ch_name:
                item = real_viewer.img1
            else:
                item = real_viewer.overlayLayersItems[channel][0]
            assert item.opacity() == pytest.approx(min(opacity, 0.999))
        return expected

    real_viewer.overlayButton.setChecked(True)
    # The restored alpha reaches the opacities
    assert assert_opacities_follow_the_scrollbars() == {
        'phase': pytest.approx(0.1), 'GFP': pytest.approx(0.9)
    }
    # Channel added from the overlay button's context menu
    action = next(
        action for action in real_viewer.overlayContextMenu.actions()
        if action.text() == 'RFP'
    )
    action.setChecked(True)
    assert assert_opacities_follow_the_scrollbars() == {
        'phase': pytest.approx(0.075), 'GFP': pytest.approx(0.675),
        'RFP': pytest.approx(0.25)
    }
    # Overlay turned on again (overlay data already loaded)
    real_viewer.overlayButton.setChecked(False)
    real_viewer.overlayButton.setChecked(True)
    assert_opacities_follow_the_scrollbars()
    action.setChecked(False)
    assert set(assert_opacities_follow_the_scrollbars()) == {'phase', 'GFP'}
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_single_channel_selection_survives_context_menu_changes(
        app, real_viewer, tmp_path, monkeypatch
    ):
    import tifffile

    images = make_overlay_position(tmp_path / 'exp')
    tifffile.imwrite(
        os.path.join(images, 'test_RFP.tif'),
        np.ones((3, 30, 30), dtype=np.uint16)
    )
    real_viewer.show()
    load(real_viewer, images, 'phase')
    monkeypatch.setattr(real_viewer, 'askSelectOverlayChannel', lambda: ['GFP'])
    real_viewer.overlayButton.setChecked(True)
    real_viewer.overlayToolbar.singleChannelCheckbox.setChecked(True)
    real_viewer.allOverlayToolbuttons['GFP'].click()
    actions = {
        action.text(): action
        for action in real_viewer.overlayContextMenu.actions()
    }

    def shownChannel():
        """The one checked toolbutton, whose layer is the one drawn"""
        checked = [
            channel
            for channel, toolbutton in real_viewer.allOverlayToolbuttons.items()
            if toolbutton.isChecked()
        ]
        assert len(checked) == 1
        channel = checked[0]
        if channel == real_viewer.user_ch_name:
            item = real_viewer.img1
        else:
            item = real_viewer.overlayLayersItems[channel][0]
        assert item.opacity() == pytest.approx(0.999)
        return channel

    assert shownChannel() == 'GFP'
    # Another channel added: the selected one stays selected
    actions['RFP'].setChecked(True)
    assert shownChannel() == 'GFP'
    # Overlay turned on again (overlay data already loaded)
    real_viewer.overlayButton.setChecked(False)
    real_viewer.overlayButton.setChecked(True)
    assert shownChannel() == 'GFP'
    # The selected channel removed: back to the base layer
    actions['GFP'].setChecked(False)
    assert shownChannel() == 'phase'
    actions['GFP'].setChecked(True)
    assert shownChannel() == 'phase'
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_removing_the_last_shown_channel_shows_the_base_layer(
        app, real_viewer, tmp_path, monkeypatch
    ):
    # Not only in Single channel mode: with the base layer unchecked, removing
    # the last checked channel from the overlay must not hide every layer
    real_viewer.show()
    load(real_viewer, make_overlay_position(tmp_path / 'exp'), 'phase')
    monkeypatch.setattr(real_viewer, 'askSelectOverlayChannel', lambda: ['GFP'])
    real_viewer.overlayButton.setChecked(True)
    toolbuttons = real_viewer.allOverlayToolbuttons
    toolbuttons['phase'].click()
    assert [ch for ch, b in toolbuttons.items() if b.isChecked()] == ['GFP']
    actions = {
        action.text(): action
        for action in real_viewer.overlayContextMenu.actions()
    }

    actions['GFP'].setChecked(False)
    assert [ch for ch, b in toolbuttons.items() if b.isChecked()] == ['phase']
    assert real_viewer.img1.opacity() == pytest.approx(0.999)
    # Overlay turned on again (overlay data already loaded)
    real_viewer.overlayButton.setChecked(False)
    real_viewer.overlayButton.setChecked(True)
    assert real_viewer.img1.opacity() == pytest.approx(0.999)
    # Single channel mode has a channel to show
    real_viewer.overlayToolbar.singleChannelCheckbox.setChecked(True)
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_cancelled_overlay_channel_selection_hides_the_overlay_toolbar(
        app, real_viewer, tmp_path, monkeypatch
    ):
    load(real_viewer, make_overlay_position(tmp_path / 'exp'), 'phase')
    # The toolbar (whose mode checkboxes act on the image) stays hidden while
    # the non-modal channel prompt and load warnings are open
    hiddenWhilePrompting = []

    def cancelChannelSelection():
        hiddenWhilePrompting.append(real_viewer.overlayToolbar.isHidden())
        return None

    monkeypatch.setattr(
        real_viewer, 'askSelectOverlayChannel', cancelChannelSelection
    )
    real_viewer.overlayButton.setChecked(True)
    assert hiddenWhilePrompting == [True]
    assert not real_viewer.overlayButton.isChecked()
    assert real_viewer.overlayToolbar.isHidden()

    # Single channel: nothing to overlay
    monkeypatch.delattr(real_viewer, 'askSelectOverlayChannel')
    warned = []
    monkeypatch.setattr(real_viewer, 'warnNoItemsToOverlay', warned.append)
    load(real_viewer, make_timelapse_position(tmp_path / 'single'), 'phase')
    real_viewer.overlayButton.setChecked(True)
    assert warned == ['channels']
    assert not real_viewer.overlayButton.isChecked()
    assert real_viewer.overlayToolbar.isHidden()

    # Channel missing in another Position: loading it fails
    exp = tmp_path / 'missing'
    paths = [
        make_overlay_position(exp, pos=1), make_timelapse_position(exp, pos=2)
    ]
    real_viewer.setData(measurements_viewer.load_positions(paths, 'phase'))
    monkeypatch.setattr(real_viewer, 'askSelectOverlayChannel', lambda: ['GFP'])
    critical = []

    def warnChannelNotFound(ch, posData):
        critical.append(ch)
        hiddenWhilePrompting.append(real_viewer.overlayToolbar.isHidden())

    monkeypatch.setattr(
        real_viewer, 'criticalFluoChannelNotFound', warnChannelNotFound
    )
    real_viewer.overlayButton.setChecked(True)
    assert critical == ['GFP']
    assert hiddenWhilePrompting == [True, True]
    assert not real_viewer.overlayButton.isChecked()
    assert real_viewer.overlayToolbar.isHidden()
    # Still connected: a later cancel keeps the overlay off too
    monkeypatch.setattr(real_viewer, 'askSelectOverlayChannel', lambda: None)
    real_viewer.overlayButton.setChecked(True)
    assert not real_viewer.overlayButton.isChecked()
    assert real_viewer.overlayToolbar.isHidden()
    assert not real_viewer.slotErrors


# --- Other open windows (each writes its whole `settings.csv` table) ------
@pytest.mark.slow
def test_settings_saved_by_another_open_window_are_restored_and_kept(
        app, real_viewer, tmp_path
    ):
    images = make_overlay_position(tmp_path / 'exp')
    # Both open before anything is saved
    windowA, windowB = real_viewer, new_viewer()
    try:
        load(windowB, images, 'phase')
        windowB.imgGrad.gradient.loadPreset('viridis')
        alphaSB = windowB.overlayLayersItems['GFP'][2]
        alphaSB.triggerAction(QAbstractSlider.SliderAction.SliderSingleStepAdd)
        alphaValue = alphaSB.value()
        # A loads, changes a LUT and updates its images: each writes the
        # whole `settings.csv` of A
        load(windowA, images, 'GFP')
        windowA.imgGrad.gradient.loadPreset('magma')
        windowA.updateAllImages()
        store = read_store(windowA)
        assert image_lut_setting_name('phase') in store
        assert image_lut_setting_name('GFP') in store
        assert overlay_alpha_setting_name('GFP') in store
        # A restores what B saved after A was opened
        load(windowA, images, 'phase')
        assert ticks(windowA.imgGrad) == preset_ticks('viridis')
        assert windowA.overlayLayersItems['GFP'][2].value() == alphaValue

        # Restore default settings in B: A does not write them back
        windowB.restoreDefaultSettings()
        windowA.imgGrad.gradient.loadPreset('magma')
        windowA.updateAllImages()
        assert set(read_store(windowA)) == {image_lut_setting_name('phase')}
    finally:
        windowB.deleteLater()
    assert not real_viewer.slotErrors


# --- Overlay modes ---------------------------------------------------------
TRANSPARENCY = gui_overlay.OVERLAY_TRANSPARENCY_SETTING
ALPHA_ENCODED = gui_overlay.OVERLAY_ALPHA_ENCODED_SETTING


@pytest.mark.slow
def test_overlay_modes_are_saved_and_applied_when_the_overlay_is_on(
        app, real_viewer, tmp_path, monkeypatch
    ):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    toolbar = real_viewer.overlayToolbar
    assert not toolbar.isTransparent()
    monkeypatch.setattr(
        real_viewer, 'askSelectOverlayChannel', lambda: ['GFP']
    )
    real_viewer.overlayButton.setChecked(True)
    toolbar.alphaEncodedIntensityCheckbox.click()
    assert toolbar.isTransparent() and toolbar.isAlphaEncodedIntensity()
    store = read_store(real_viewer)
    assert store[TRANSPARENCY] is True
    assert store[ALPHA_ENCODED] is True
    # Changes made by code (e.g., exports of the main GUI) are not saved
    toolbar.setTransparent(False)
    assert not toolbar.isAlphaEncodedIntensity()
    assert read_store(real_viewer)[ALPHA_ENCODED] is True

    other = new_viewer()
    try:
        otherToolbar = other.overlayToolbar
        # Restored without applying them: no data, overlay off
        assert otherToolbar.isTransparent()
        assert otherToolbar.isAlphaEncodedIntensity()
        load(other, images, 'phase')
        assert other.img1.opacity() == 1.0
        monkeypatch.setattr(other, 'askSelectOverlayChannel', lambda: ['GFP'])
        other.overlayButton.setChecked(True)
        imageItem, lutItem = other.overlayLayersItems['GFP'][:2]
        assert other.rgbaImg1.isVisible()
        assert other.rgbaImg1.image is not None
        assert other.img1.opacity() < 0.01
        assert imageItem.opacity() < 0.01
        connected = lutItem.receivers(lutItem.sigLookupTableChanged)
        imgGrad = other.imgGrad
        connectedImg = imgGrad.receivers(imgGrad.sigLevelsChanged)
        # Turning the overlay on again never connects twice
        other.overlayButton.setChecked(False)
        other.overlayButton.setChecked(True)
        assert other.img1.opacity() < 0.01
        assert lutItem.receivers(lutItem.sigLookupTableChanged) == connected
        assert imgGrad.receivers(imgGrad.sigLevelsChanged) == connectedImg
        # Reloading: the new items get the mode when the overlay is on
        load(other, images, 'phase')
        assert otherToolbar.isTransparent()
        other.overlayButton.setChecked(True)
        assert other.overlayLayersItems['GFP'][0].opacity() < 0.01
        assert other.img1.opacity() < 0.01
        otherToolbar.transparencyCheckbox.click()
        assert not otherToolbar.isAlphaEncodedIntensity()
        assert other.img1.opacity() > 0.01
        store = read_store(other)
        assert store[TRANSPARENCY] is False
        assert store[ALPHA_ENCODED] is False
    finally:
        other.deleteLater()
    assert_no_channel_settings_in_csv(real_viewer)
    assert not real_viewer.slotErrors


@pytest.mark.slow
def test_configuration_with_overlays_in_transparent_mode(
        app, real_viewer, tmp_path, monkeypatch
    ):
    import tifffile

    images = make_overlay_position(tmp_path / 'exp')
    # Not constant: the composite depends on the visible layers
    tifffile.imwrite(
        os.path.join(images, 'test_GFP.tif'),
        np.arange(3*30*30, dtype=np.uint16).reshape(3, 30, 30)[:, ::-1]
    )
    configuration = {
        'version': 2, 'images_paths': [images], 'channel': 'phase',
        'segm_endname': 'segm', 'overlay_channels': ['GFP'],
        'visible_channels': ['GFP'], 'measurements_mode': 1,
        'time_course_columns': ['area'],
        'measurements_axes': {'X': 'area', 'Y': ''},
        'normalize_measurements': False
    }
    path = tmp_path / 'transparent.json'
    path.write_text(json.dumps(configuration))
    errors = []
    monkeypatch.setattr(
        real_viewer, 'configurationError', lambda *args: errors.append(args)
    )
    monkeypatch.setattr(
        real_viewer, 'selectConfigurationInput',
        lambda config: real_viewer.configurationInputOptions(config)[0]
    )
    real_viewer.overlayToolbar.setModesChecked(transparent=True)
    # Not shown, no events processed: a shown window rebuilds the composite
    # when the overlay LUT item is resized
    assert real_viewer.loadConfiguration(filepath=str(path)), errors
    assert not errors
    assert real_viewer.img1.opacity() < 0.01
    assert real_viewer.overlayLayersItems['GFP'][0].opacity() < 0.01
    # The composite already follows the visible channels (base layer hidden)
    assert not real_viewer.allOverlayToolbuttons['phase'].isChecked()
    rgba = real_viewer.rgbaImg1.image.copy()
    real_viewer.setOverlayImages()
    np.testing.assert_allclose(real_viewer.rgbaImg1.image, rgba)
    assert not real_viewer.slotErrors


def test_set_modes_checked_emits_nothing(app):
    toolbar = widgets.OverlayToolbar()
    emitted = []
    toolbar.sigSetTranspacency.connect(emitted.append)
    toolbar.sigModesClicked.connect(lambda: emitted.append('clicked'))
    toolbar.setModesChecked(transparent=False, alphaEncoded=True)
    assert toolbar.isTransparent() and toolbar.isAlphaEncodedIntensity()
    toolbar.setModesChecked(transparent=True, alphaEncoded=False)
    assert toolbar.isTransparent() and not toolbar.isAlphaEncodedIntensity()
    assert emitted == []
    toolbar.transparencyCheckbox.click()
    assert emitted == [False, 'clicked']
    toolbar.deleteLater()


# --- Main GUI callback (host object, as in `test_gui_mixins.py`) -----------
def make_lut_host(tmp_path):
    host = type('Host', (LutGuiElements,), {})()
    host.df_settings = pd.DataFrame(
        {'value': ['No']}, index=pd.Index(['is_bw_inverted'], name='setting'),
        dtype=object
    )
    host.settings_csv_path = str(tmp_path / 'settings.csv')
    host.imgGrad = widgets.myHistogramLUTitem()
    host.logger = MagicMock()
    host.user_ch_name = 'phase'
    return host


def test_main_gui_callback_saves_the_channel_lut_only_after_restoring(
        app, tmp_path
    ):
    host = make_lut_host(tmp_path)
    setting = image_lut_setting_name('phase')
    # While loading (e.g., the first-load `invertBw`) nothing is saved for
    # the channel, the older 'img_cmap_*' rows are
    host.resetImageChannelLut()
    host.imgGrad.gradient.loadPreset('viridis')
    gui.guiWin.imgGradLUTfinished_cb(host)
    assert setting not in read_store(host)
    assert 'img_cmap_tick0_pos' in host.df_settings.index

    host.restoreImageChannelLut()
    assert ticks(host.imgGrad) == GREY_TICKS
    host.imgGrad.gradient.loadPreset('magma')
    gui.guiWin.imgGradLUTfinished_cb(host)
    state, _ = saved_lut(read_store(host), setting)
    assert state['ticks'] == preset_ticks('magma')
    saved = pd.read_csv(host.settings_csv_path, index_col='setting')
    assert 'img_cmap_tick0_pos' in saved.index
    assert setting not in saved.index

    # Changes made by code are not saved
    host.saveChannelDisplaySettings({setting: 'corrupted'})
    with host.settingChannelLutsFromCode():
        host.imgGrad.gradient.loadPreset('grey')
        gui.guiWin.imgGradLUTfinished_cb(host)
    assert read_store(host)[setting] == 'corrupted'
    # An invalid saved LUT is ignored (logged): grey
    host.imgGrad.gradient.loadPreset('magma')
    host.restoreImageChannelLut()
    host.logger.warning.assert_called_once()
    assert ticks(host.imgGrad) == GREY_TICKS


def _gui_win_calls():
    """{method name of `gui.guiWin`: (calls, connected)}: `calls` lists the
    `(line, name)` of every `<...>.name(...)` call, `connected` the names of
    the `self.<name>` passed (also through `partial`) to a `connect`."""
    with open(gui.__file__, encoding='utf-8') as file:
        tree = ast.parse(file.read())
    guiWin = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'guiWin'
    )
    methods = {}
    for method in guiWin.body:
        if not isinstance(method, ast.FunctionDef):
            continue
        calls, connected = [], set()
        for node in ast.walk(method):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            calls.append((node.lineno, node.func.attr))
            if node.func.attr != 'connect':
                continue
            for arg in node.args:
                for sub in ast.walk(arg):
                    isSelfAttr = (
                        isinstance(sub, ast.Attribute)
                        and isinstance(sub.value, ast.Name)
                        and sub.value.id == 'self'
                    )
                    if isSelfAttr:
                        connected.add(sub.attr)
        methods[method.name] = (calls, connected)
    return methods


@pytest.mark.slow
def test_main_gui_calls_the_channel_lut_and_overlay_mode_hooks():
    """The main GUI's own call sites (no test builds `guiWin`): merges of
    `main` into `gui.py` have dropped or moved lines before."""
    methods = _gui_win_calls()

    def names(method):
        return [name for _, name in methods[method][0]]

    def line(method, name):
        return next(ln for ln, _name in methods[method][0] if _name == name)

    # First-load `invertBw`, then the saved LUT (order tested through the
    # viewer, `test_base_lut_follows_invert_black_white`)
    assert 'restoreImageChannelLutAfterLoading' in names('loadingDataCompleted')
    assert 'restoreImageChannelLut' not in names('loadingDataCompleted')
    assert 'resetImageChannelLut' in names('reInitGui')
    assert 'saveImageChannelLut' in names('imgGradLUTfinished_cb')
    # Cancel in "Overlay color..." shows the saved LUT again
    assert 'overlayColorDialogRejected' in methods['gui_connectEditActions'][1]
    # Overlay modes: restored when the toolbar is created, saved on clicks
    # and on the Yes of the export prompt
    assert 'restoreOverlayModes' in names('gui_createControlsToolbar')
    assert 'saveOverlayModes' in methods['gui_createControlsToolbar'][1]
    export = 'exportToCheckAskOverlay'
    assert line(export, 'setTransparent') < line(export, 'saveOverlayModes')
