"""LUTs saved per channel name and saved check states of the overlay modes
(`gui_navigation.LutGuiElements`, `gui_overlay.OverlayChannelsGuiElements`),
shared by the main GUI and the measurements viewer (tested through the
viewer, which loads real data cheaply)."""
import json
from unittest.mock import MagicMock

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
    make_overlay_position, real_viewer  # noqa: F401 (fixture)
)

GREY_TICKS = default_image_lut()['ticks']
PRESETS = pg.graphicsItems.GradientEditorItem.Gradients


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


def preset_ticks(name, invertedBy=None):
    """Ticks of the pyqtgraph preset `name`, inverted as `Invert black/white`
    does with the LUT item `invertedBy`."""
    gradient = PRESETS[name]
    if invertedBy is not None:
        gradient = invertedBy.invertGradient(gradient)
    return colors._gradient_ticks_rgba(gradient['ticks'])


def read_settings(window):
    return pd.read_csv(window.settings_csv_path, index_col='setting')


def saved_lut(df, setting):
    return colors.gradient_state_from_setting(df.at[setting, 'value'])


def saved_flag(df, setting):
    return str(df.at[setting, 'value'])


# --- Serialization (no window) ---------------------------------------------
def test_gradient_setting_round_trip_and_invalid_text():
    state = {
        'mode': 'rgb', 'ticksVisible': False,
        'ticks': [(1.0, (255, 0, 0, 255)), (0.0, (0, 0, 0))],
    }
    text = colors.gradient_state_to_setting(state, bw_inverted=True)
    restored, bwInverted = colors.gradient_state_from_setting(text)
    assert bwInverted is True
    assert restored['ticksVisible'] is False
    # Sorted by position, alpha added
    assert restored['ticks'] == [(0.0, (0, 0, 0, 255)), (1.0, (255, 0, 0, 255))]
    assert colors.are_gradients_equal(restored, state)
    for invalid in (
            'not json', json.dumps({'ticks': []}),
            json.dumps({'ticks': [[1.5, [0, 0, 0]]]}),
            json.dumps({'ticks': [[0.5, [0, 0, 300]]]}),
            json.dumps({'ticks': [[0.5, 'red']]}),
            json.dumps({'mode': 'cmyk', 'ticks': [[0, [0, 0, 0]]]}),
        ):
        assert colors.gradient_state_from_setting(invalid) is None


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


# --- Displayed (base) channel ----------------------------------------------
def test_base_lut_is_saved_and_restored_per_channel(
        app, real_viewer, tmp_path
    ):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    assert ticks(real_viewer.imgGrad) == GREY_TICKS
    real_viewer.imgGrad.gradient.loadPreset('viridis')
    state, bwInverted = saved_lut(
        read_settings(real_viewer), image_lut_setting_name('phase')
    )
    assert state['ticks'] == preset_ticks('viridis')
    assert bwInverted is False

    # Another channel without a saved LUT: grey, not the previous LUT
    load(real_viewer, images, 'GFP')
    assert ticks(real_viewer.imgGrad) == GREY_TICKS
    assert image_lut_setting_name('GFP') not in real_viewer.df_settings.index
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
    assert not real_viewer.slotErrors


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
        read_settings(real_viewer), image_lut_setting_name('GFP')
    )
    assert bwInverted is True
    assert state['ticks'] == preset_ticks('magma', invertedBy=real_viewer.imgGrad)
    # Saved with black/white not inverted: shown inverted
    load(real_viewer, images, 'phase')
    assert ticks(real_viewer.imgGrad) == preset_ticks('viridis', invertedBy=real_viewer.imgGrad)

    # First load of a new window with inverted black/white: the first-load
    # `invertBw` neither inverts the restored LUT again nor overwrites it
    other = new_viewer()
    try:
        assert other.invertBwAction.isChecked()
        load(other, images, 'phase')
        assert ticks(other.imgGrad) == preset_ticks('viridis', invertedBy=real_viewer.imgGrad)
        state, bwInverted = saved_lut(
            read_settings(other), image_lut_setting_name('phase')
        )
        assert bwInverted is False
        assert state['ticks'] == preset_ticks('viridis')
        other.invertBwAction.setChecked(False)
        assert ticks(other.imgGrad) == preset_ticks('viridis')
    finally:
        other.deleteLater()
    assert not real_viewer.slotErrors


# --- Overlay channels ------------------------------------------------------
def test_overlay_lut_is_saved_per_channel_and_follows_the_color(
        app, real_viewer, tmp_path
    ):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    lutItem = real_viewer.overlayLayersItems['GFP'][1]
    assert ticks(lutItem)[0][1] == (0, 0, 0, 255)
    lutItem.gradient.loadPreset('magma')
    setting = overlay_lut_setting_name('GFP')
    state, _ = saved_lut(read_settings(real_viewer), setting)
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
    saved = read_settings(real_viewer)
    assert setting not in saved.index
    assert saved.at['GFP_rgb', 'value'] == '255_0_0'
    load(real_viewer, images, 'phase')
    assert ticks(real_viewer.overlayLayersItems['GFP'][1]) == expected
    assert not real_viewer.slotErrors


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
        saved = read_settings(real_viewer)
        assert setting in saved.index
        assert 'GFP_rgb' not in saved.index
        button.colorDialog.reject()
        assert ticks(lutItem) == preset_ticks('magma')
        assert lutItem.overlayColorButton.color().getRgb()[:3] == colorBefore
        assert setting in read_settings(real_viewer).index

    # OK: the color is saved and the LUT follows it
    button = lutItem.overlayColorButton
    button.selectColor()
    button.colorDialog.setCurrentColor(QColor(255, 0, 0))
    button.colorDialog.accept()
    assert ticks(lutItem) == red
    saved = read_settings(real_viewer)
    assert setting not in saved.index
    assert saved.at['GFP_rgb', 'value'] == '255_0_0'
    # Cancel now shows the chosen color
    button.selectColor()
    button.colorDialog.setCurrentColor(QColor(0, 0, 255))
    button.colorDialog.reject()
    assert ticks(lutItem) == red
    assert not real_viewer.slotErrors


def test_restore_default_settings_resets_the_channel_luts(
        app, real_viewer, tmp_path
    ):
    images = make_overlay_position(tmp_path / 'exp')
    load(real_viewer, images, 'phase')
    lutItem = real_viewer.overlayLayersItems['GFP'][1]
    colorTicks = ticks(lutItem)
    real_viewer.imgGrad.gradient.loadPreset('viridis')
    lutItem.gradient.loadPreset('magma')

    real_viewer.restoreDefaultSettings()

    assert ticks(real_viewer.imgGrad) == GREY_TICKS
    # Not the LUT of the displayed image (`restoreState` of the overlay)
    assert ticks(lutItem) == colorTicks
    for setting in read_settings(real_viewer).index:
        assert not setting.startswith(('image_lut_', 'overlay_lut_'))
    # Changes after the reset are saved again
    real_viewer.imgGrad.gradient.loadPreset('magma')
    lutItem.gradient.loadPreset('viridis')
    saved = read_settings(real_viewer)
    assert image_lut_setting_name('phase') in saved.index
    assert overlay_lut_setting_name('GFP') in saved.index
    assert not real_viewer.slotErrors


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
    assert float(read_settings(real_viewer).at[setting, 'value']) == expected
    # `setValue` by code: not saved
    alphaSB.setValue(5)
    assert float(read_settings(real_viewer).at[setting, 'value']) == expected
    # Dragging: saved once, on release
    alphaSB.setSliderDown(True)
    alphaSB.setSliderPosition(30)
    assert alphaSB.value() == 30
    assert float(read_settings(real_viewer).at[setting, 'value']) == expected
    alphaSB.setSliderDown(False)
    assert float(read_settings(real_viewer).at[setting, 'value']) == 0.75

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
    for text, value in (('not a number', default), ('nan', default), ('2', 40)):
        real_viewer._set_setting_value(setting, text)
        load(real_viewer, images, 'phase')
        assert real_viewer.overlayLayersItems['GFP'][2].value() == value

    # Restore default settings: dropped and back to the default alpha
    real_viewer._set_setting_value(setting, 0.75)
    load(real_viewer, images, 'phase')
    alphaSB = real_viewer.overlayLayersItems['GFP'][2]
    real_viewer.restoreDefaultSettings()
    assert alphaSB.value() == default
    assert setting not in read_settings(real_viewer).index
    assert not real_viewer.slotErrors


# --- Overlay modes ---------------------------------------------------------
TRANSPARENCY = gui_overlay.OVERLAY_TRANSPARENCY_SETTING
ALPHA_ENCODED = gui_overlay.OVERLAY_ALPHA_ENCODED_SETTING


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
    saved = read_settings(real_viewer)
    assert saved_flag(saved, TRANSPARENCY) == 'True'
    assert saved_flag(saved, ALPHA_ENCODED) == 'True'
    # Changes made by code (e.g., exports of the main GUI) are not saved
    toolbar.setTransparent(False)
    assert not toolbar.isAlphaEncodedIntensity()
    saved = read_settings(real_viewer)
    assert saved_flag(saved, ALPHA_ENCODED) == 'True'

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
        saved = read_settings(other)
        assert saved_flag(saved, TRANSPARENCY) == 'False'
        assert saved_flag(saved, ALPHA_ENCODED) == 'False'
    finally:
        other.deleteLater()
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
    assert setting not in host.df_settings.index
    assert 'img_cmap_tick0_pos' in host.df_settings.index

    host.restoreImageChannelLut()
    assert ticks(host.imgGrad) == GREY_TICKS
    host.imgGrad.gradient.loadPreset('magma')
    gui.guiWin.imgGradLUTfinished_cb(host)
    state, _ = saved_lut(host.df_settings, setting)
    assert state['ticks'] == preset_ticks('magma')
    saved = pd.read_csv(host.settings_csv_path, index_col='setting')
    assert setting in saved.index

    # Changes made by code are not saved
    host.df_settings.at[setting, 'value'] = 'corrupted'
    with host.settingChannelLutsFromCode():
        host.imgGrad.gradient.loadPreset('grey')
        gui.guiWin.imgGradLUTfinished_cb(host)
    assert host.df_settings.at[setting, 'value'] == 'corrupted'
    # An invalid saved LUT is ignored (logged): grey
    host.imgGrad.gradient.loadPreset('magma')
    host.restoreImageChannelLut()
    host.logger.warning.assert_called_once()
    assert ticks(host.imgGrad) == GREY_TICKS
