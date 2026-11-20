"""Navigation GUI elements (frames, z-slices, objects, recent paths, LUT) 
shared between the main GUI (`gui.guiWin`) and other windows (e.g., 
`measurements_viewer.MeasurementsViewerWin`).

The classes are mixins meant to be inherited together with `QMainWindow`. 
They expect the host window to define:
    - `self.data`: list of `load.loadData` objects
    - `self.pos_i`: index of the current position in `self.data`
    - `self.ax1`: the `pyqtgraph.PlotItem` displaying the image
    - `self.isSegm3D`: whether the segmentation masks are 3D
    - `self.openRecentMenu`: `QMenu` populated with the recent paths
    - `self.openRecentFile(path)`: method called to load a recent path
"""
import os
import json
import logging
import tempfile
import contextlib
from functools import partial

import numpy as np
import pyqtgraph as pg

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QLabel, QCheckBox, QHBoxLayout, QAction

from . import widgets, apps, myutils, colors
from . import path as acdc_path

_font = QFont()
_font.setPixelSize(11)

_logger = logging.getLogger(__name__)

# File (next to `settings.csv`) with the display settings saved per channel
# name and the overlay modes, as one JSON object {setting name: value}
# (see `LutGuiElements`, `read_channel_display_settings`)
CHANNEL_DISPLAY_SETTINGS_FILENAME = 'channel_display_settings.json'

# Channel display settings with the LUT of a channel, followed by the
# channel name (see `LutGuiElements`)
IMAGE_LUT_SETTING_PREFIX = 'image_lut_'
OVERLAY_LUT_SETTING_PREFIX = 'overlay_lut_'
# Alpha (0-1) of the alpha scrollbar of an overlay channel, followed by the
# channel name (see `gui_overlay.OverlayChannelsGuiElements`). Not
# 'overlay_alpha_<channel>': it would be the overlay mode setting
# 'overlay_alpha_encoded_intensity' for a channel 'encoded_intensity'.
OVERLAY_ALPHA_SETTING_PREFIX = 'overlay_alpha_slider_'
# Rows of the LUT of the displayed image saved by `myHistogramLUTitem` for
# every channel before the LUTs were saved per channel
_GLOBAL_IMAGE_LUT_SETTING_PREFIX = 'img_cmap_'

def image_lut_setting_name(channel):
    """Name of the channel display setting with the LUT of `channel` when it
    is the displayed (base) image (`imgGrad`)."""
    return f'{IMAGE_LUT_SETTING_PREFIX}{channel}'

def overlay_lut_setting_name(channel):
    """Name of the channel display setting with the LUT of `channel` when
    it is overlaid on the displayed image (overlay LUT item)."""
    return f'{OVERLAY_LUT_SETTING_PREFIX}{channel}'

def overlay_alpha_setting_name(channel):
    """Name of the channel display setting with the alpha of the alpha
    scrollbar of `channel` when it is overlaid on the displayed image."""
    return f'{OVERLAY_ALPHA_SETTING_PREFIX}{channel}'

def overlay_color_setting_name(channel):
    """Name of the channel display setting with the overlay color of
    `channel` ('<r>_<g>_<b>'), also the name of the `settings.csv` row where
    older versions saved it (see `LutGuiElements`)."""
    return f'{channel}_rgb'

def channel_display_settings_path(settings_csv_path):
    """Path of the channel display settings file, in the folder of the user
    settings file `settings_csv_path`."""
    folderpath = os.path.dirname(os.path.abspath(settings_csv_path))
    return os.path.join(folderpath, CHANNEL_DISPLAY_SETTINGS_FILENAME)

def read_channel_display_settings(filepath, logger=None):
    """Read the channel display settings saved in `filepath`.

    The file holds one JSON object {setting name: value}. Every window reads
    it again whenever it restores a setting and changes only the settings it
    saves (`update_channel_display_settings`), so that windows open at the
    same time (main GUI, measurements viewer) never overwrite each other's
    settings, as whole-table writes of `settings.csv` would.

    Returns
    -------
    dict
        The settings. Empty when the file is missing, or unreadable or
        invalid (a warning is logged to `logger`).
    """
    logger = _logger if logger is None else logger
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            settings = json.load(file)
    except FileNotFoundError:
        return {}
    except Exception as err:
        logger.warning(
            f'Ignoring the channel display settings file "{filepath}" that '
            f'cannot be read ({err})'
        )
        return {}
    if not isinstance(settings, dict):
        logger.warning(
            f'Ignoring the channel display settings file "{filepath}" that '
            'does not contain a JSON object'
        )
        return {}
    return settings

def update_channel_display_settings(
        filepath, values=None, remove_prefixes=(), logger=None
    ):
    """Save channel display settings in `filepath` and keep the others.

    The file is read again first (`read_channel_display_settings`, settings
    saved meanwhile by other windows are kept), then written to a temporary
    file that replaces it (`os.replace`).

    Parameters
    ----------
    filepath : str
        Path of the file (`channel_display_settings_path`).
    values : dict, optional
        {setting name: JSON-serializable value} to save. None as value
        removes the setting.
    remove_prefixes : tuple of str, optional
        Remove every setting whose name starts with one of them.
    logger : logging.Logger, optional
        Logger of the warning when the file cannot be written.

    Returns
    -------
    bool
        Whether the file was written (not written when nothing changed).
    """
    logger = _logger if logger is None else logger
    remove_prefixes = tuple(remove_prefixes)
    settings = read_channel_display_settings(filepath, logger=logger)
    updated = {
        name: value for name, value in settings.items()
        if not (remove_prefixes and str(name).startswith(remove_prefixes))
    }
    for name, value in (values or {}).items():
        if value is None:
            updated.pop(name, None)
        else:
            updated[name] = value
    if updated == settings:
        return False
    folderpath = os.path.dirname(os.path.abspath(filepath))
    tempFilepath = None
    try:
        os.makedirs(folderpath, exist_ok=True)
        fd, tempFilepath = tempfile.mkstemp(
            prefix=f'.{CHANNEL_DISPLAY_SETTINGS_FILENAME}.', suffix='.tmp',
            dir=folderpath
        )
        with os.fdopen(fd, 'w', encoding='utf-8') as file:
            json.dump(updated, file, sort_keys=True)
        os.replace(tempFilepath, filepath)
    except Exception as err:
        logger.warning(
            f'Could not save the channel display settings to "{filepath}" '
            f'({err})'
        )
        if tempFilepath is not None and os.path.exists(tempFilepath):
            try:
                os.remove(tempFilepath)
            except OSError:
                pass
        return False
    return True

def default_image_lut():
    """LUT of a displayed channel without a saved LUT (grey, as the
    pyqtgraph preset 'grey'), with black and white not inverted."""
    return {
        'mode': 'rgb',
        'ticks': [(0.0, (0, 0, 0, 255)), (1.0, (255, 255, 255, 255))],
        'ticksVisible': True,
    }

def drop_channel_lut_settings(df_settings):
    """Return the user settings `df_settings` (`settings.csv`) without the
    LUT of the displayed image saved for every channel ('img_cmap_*', before
    the LUTs were saved per channel) and without rows named as the channel
    display settings with LUTs and overlay alphas (earlier development
    versions saved them in `settings.csv`; they are now in the channel
    display settings file, see `LutGuiElements`)."""
    prefixes = (
        IMAGE_LUT_SETTING_PREFIX, OVERLAY_LUT_SETTING_PREFIX,
        OVERLAY_ALPHA_SETTING_PREFIX, _GLOBAL_IMAGE_LUT_SETTING_PREFIX
    )
    keep = [
        not str(setting).startswith(prefixes) for setting in df_settings.index
    ]
    return df_settings[keep].copy()

Z_PROJECTION_METHODS = (
    'single z-slice',
    'max z-projection',
    'mean z-projection',
    'median z-proj.'
)

class FramesNavigationGuiElements:
    def gui_createFramesNavigationWidgets(self):
        self.navigateScrollBar = widgets.navigateQScrollBar(Qt.Horizontal)
        self.navigateScrollBar.setDisabled(True)
        self.navigateScrollBar.setMinimum(1)
        self.navigateScrollBar.setMaximum(1)
        
        self.t_label = QLabel('frame n.  ')
        self.t_label.setFont(_font)
        
        self.navSpinBox = widgets.SpinBox(disableKeyPress=True)
        self.navSpinBox.setMinimum(1)
        self.navSpinBox.setMaximum(100)
        self.navSizeLabel = QLabel('/ND')
        
        hight_spin_box = int(self.navSpinBox.sizeHint().height() * 0.8)
        self.navigateScrollBar.setFixedHeight(hight_spin_box)
    
    def gui_createFramesNavigationLabelLayout(self):
        layout = QHBoxLayout()
        layout.addWidget(self.t_label)
        layout.addWidget(self.navSpinBox)
        layout.addWidget(self.navSizeLabel)
        return layout
    
    def resetNavigateFramesScrollbar(self, frame_i=None):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i
            
        self.navigateScrollBar.setValueNoSignal(frame_i+1)

class ZsliceNavigationGuiElements:
    def gui_createZsliceNavigationWidgets(self):
        self.zSliceScrollBar = widgets.linkedQScrollbar(Qt.Horizontal)

        self.zProjComboBox = widgets.ComboBox()
        self.zProjComboBox.setFont(_font)
        self.zProjComboBox.addItems(Z_PROJECTION_METHODS)
        
        self.zProjLockViewButton = widgets.LockPushButton()
        self.zProjLockViewButton.setCheckable(True)
        self.zProjLockViewButton.setToolTip(
            'If active, the selected z-slice view is applied to all frames'
        )
        self.zProjLockViewButton.hide()
        
        self.switchPlaneCombobox = widgets.SwitchPlaneCombobox()
        self.switchPlaneCombobox.setToolTip(
            'Switch viewed plane'
        )
        
        self.launch3dViewerButton = widgets.threeDPushButton()
        self.launch3dViewerButton.setToolTip(
            'Launch 3D viewer for the currently visualised volume'
        )
        self.launch3dViewerButton.setCheckable(True)
        
        self.zSliceCheckbox = QCheckBox('z-slice')
        self.zSliceCheckbox.setToolTip(
            'Activate/deactivate control of the z-slices with keyboard arrows.\n\n'
            'SHORTCUT to toggle ON/OFF: "Z" key'
        )
        self.zSliceSpinbox = widgets.SpinBox(disableKeyPress=True)
        self.zSliceSpinbox.setMinimum(1)
        self.SizeZlabel = QLabel('/ND')
        self.zSliceSpinbox.connectValueChanged(self.onZsliceSpinboxValueChange)
        
        hight_spin_box = int(self.zSliceSpinbox.sizeHint().height() * 0.8)
        self.zSliceScrollBar.setFixedHeight(hight_spin_box)
    
    def gui_createZsliceCheckboxLayout(self):
        layout = QHBoxLayout()
        layout.addWidget(self.zSliceCheckbox)
        layout.addWidget(self.zSliceSpinbox)
        layout.addWidget(self.SizeZlabel)
        return layout
    
    def maxProjToggleActionTriggered(self):
        if not getattr(self, 'data', None):
            # Nothing loaded: Qt5's QAction.trigger() emits `triggered`
            # even when the action is disabled
            return
        posData = self.data[self.pos_i]
        if posData.SizeZ <= 1:
            return
        
        zProjHow_curr = self.zProjComboBox.currentText()
        if not hasattr(self, 'setMaxProjToggleOld') and zProjHow_curr == 'single z-slice':
            self.setMaxProjToggleOld = 'max z-projection'


        if zProjHow_curr == 'single z-slice':
            self.zProjComboBox.setCurrentText(self.setMaxProjToggleOld)
        else:
            self.setMaxProjToggleOld = zProjHow_curr
            self.zProjComboBox.setCurrentText('single z-slice')

    def setZsliceNavigationRange(self, SizeZ):
        self.zSliceScrollBar.setMaximum(SizeZ-1)
        self.zSliceSpinbox.setMaximum(SizeZ)
        # Displayed z-slice numbers may start at 0 ("Show z-slice index")
        start_value = self.zSliceSpinbox.startValue()
        if start_value is None:
            start_value = 1
        correction = start_value - self.zSliceSpinbox.minimum()
        maximum = self.zSliceSpinbox.maximum()
        self.SizeZlabel.setText(f'/{maximum + correction}')
    
    def onZsliceSpinboxValueChange(self, value):
        self.zSliceScrollBar.setSliderPosition(value-1)
    
    def z_slice_index(self):
        posData = self.data[self.pos_i]
        if posData.SizeZ == 1:
            return None
        zProjHow = self.zProjComboBox.currentText()
        if zProjHow != 'single z-slice':
            return zProjHow
        
        axis_slice = self.zSliceScrollBar.sliderPosition()
        if self.switchPlaneCombobox.depthAxes() == 'x':
            z_slice = (
                slice(None, None, None), slice(None, None, None), axis_slice
            )
        elif self.switchPlaneCombobox.depthAxes() == 'y':
            z_slice = (
                slice(None, None, None), axis_slice
            )
        else:
            z_slice = axis_slice
            
        return z_slice
    
    def get_2Dimg_from_switched_plane(self, imgData, yx=None):
        """Return the 2D image sliced along the depth axis selected with
        `self.switchPlaneCombobox` or None if the viewed plane is 'xy'.

        If `yx` is not None, return only the value at (y, x) of that 2D
        image.
        """
        axis_slice = self.zSliceScrollBar.sliderPosition()
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if depthAxes == 'x':
            if yx is not None:
                return imgData[yx[0], yx[1], axis_slice]
            return imgData[:, :, axis_slice].copy()
        elif depthAxes == 'y':
            if yx is not None:
                return imgData[yx[0], axis_slice, yx[1]]
            return imgData[:, axis_slice].copy()
    
    @staticmethod
    def zProjectImage(imgData, zProjHow, z):
        if zProjHow == 'single z-slice':
            img = imgData[z]
        elif zProjHow == 'max z-projection':
            img = imgData.max(axis=0)
        elif zProjHow == 'mean z-projection':
            img = imgData.mean(axis=0)
        elif zProjHow == 'median z-proj.':
            img = np.median(imgData, axis=0)
        return img

class ZoomToObjGuiElements:
    def getObjBbox(self, obj_bbox):
        if self.isSegm3D and len(obj_bbox)==6:
            obj_bbox = (obj_bbox[1], obj_bbox[2], obj_bbox[4], obj_bbox[5])
            return obj_bbox
        else:
            return obj_bbox
    
    def setViewRangeToObjBbox(self, obj_bbox, padding=5):
        min_row, min_col, max_row, max_col = self.getObjBbox(obj_bbox)
        xRange = min_col-padding, max_col+padding
        yRange = max_row+padding, min_row-padding
        self.ax1.setRange(xRange=xRange, yRange=yRange)

class RecentPathsGuiElements:
    def readRecentPaths(self):
        # Step 0. Remove the old options from the menu
        self.openRecentMenu.clear()        
        recentPaths, dates = myutils.get_recent_paths()
        
        # Step 2. Dynamically create 20 actions
        actions = []
        for path in recentPaths[:20]:
            path_text = acdc_path.unexpanduser(path)
            action = QAction(path_text, self)
            action.triggered.connect(partial(self.openRecentFile, path))
            actions.append(action)

        # Step 3. Add the actions to the menu
        self.openRecentMenu.addActions(actions)

        # Setp 4. Add the More action
        self.openRecentMoreDialog = apps.SearchableListboxDialog(
            recentPaths, 
            title='Recent paths', 
            searchLineEditText='Search recent paths (double-click to load)...',
            parent=self,
            dates=dates,
        )
        self.openRecentMoreAction = QAction('More/Search...', self)
        self.openRecentMoreAction.triggered.connect(
            self.showOpenRecentMoreWidget
        )
        self.openRecentMoreDialog.sigOk.connect(self.openRecentFile)
        self.openRecentMenu.addSeparator()
        self.openRecentMenu.addAction(self.openRecentMoreAction)
    
    def addPathToOpenRecentMenu(self, path):
        path = os.path.normpath(path)
        for action in self.openRecentMenu.actions():
            action_path = os.path.expanduser(os.path.normpath(action.text()))
            if path == action_path:
                return
        else:
            path_text = acdc_path.unexpanduser(path)
            action = QAction(path_text, self)
            action.triggered.connect(partial(self.openRecentFile, path))
        
        try:
            firstAction = self.openRecentMenu.actions()[0]
            self.openRecentMenu.insertAction(firstAction, action)
        except Exception as e:
            pass
    
    def addToRecentPaths(self, path, logger=None):
        myutils.addToRecentPaths(path, logger=getattr(self, 'logger', None))
    
    def getMostRecentPath(self):
        return myutils.getMostRecentPath()
    
    def showOpenRecentMoreWidget(self):
        self.openRecentMoreDialog.show()

class LutGuiElements:
    """Rescale intensities (LUT) behaviour of the image `self.img1` 
    (`widgets.ParentImageItem`) with LUT item `self.imgGrad` 
    (`widgets.myHistogramLUTitem`).
    
    The host window must define `self.isDataLoaded`, `self.logger`,
    `self.df_settings`, `self.settings_csv_path`, `self.user_ch_name`,
    `self.rescaleIntensChannelHowMapper`, `self.updateAllImages()`,
    `self.invertBwAction`, `self.invertBwAlreadyCalledOnce`,
    `self.invertBw(checked)` and `posData.lutLevels` (dict of
    {channel: {}}) for each loaded `posData`.

    LUTs saved per channel name
    ---------------------------
    The LUT (gradient) of every channel is saved in the channel display
    settings and restored whenever a channel with the same name is loaded.
    These settings (with the overlay colors, the overlay alphas and the
    overlay modes of `gui_overlay`) are not in `df_settings`: each window
    writes its whole `df_settings` to `settings.csv`, so a window open at
    the same time would overwrite them. They are in their own file
    (`CHANNEL_DISPLAY_SETTINGS_FILENAME`, in the folder of
    `settings_csv_path`, shared by the main GUI and the measurements
    viewer), read again on every restore (`channelDisplaySetting`) and
    changed one setting at a time (`saveChannelDisplaySettings`), so that
    every open window sees and keeps what the others saved:

    - Displayed (base) channel: setting `image_lut_<channel>`, saved on
      every finished change of the `imgGrad` gradient (the host's
      `imgGradLUTfinished_cb` calls `saveImageChannelLut`) and restored by
      `restoreImageChannelLut`, which the host calls at the end of every
      load, after the first-load `invertBw` (it would invert the restored
      LUT again): both run in `restoreImageChannelLutAfterLoading`. Until
      then nothing is saved for the new channel (the first-load inversion
      would overwrite its saved LUT). A channel without a saved LUT gets
      the grey LUT (`default_image_lut`): it never inherits the LUT of the
      channel loaded before. The LUTs are saved with the black/white
      inversion of the window (`Invert black/white`) and inverted as that
      menu does (`invertGradient`) when it differs.
    - Overlay channels: setting `overlay_lut_<channel>`, saved on every
      finished change of the gradient of the overlay LUT item
      (`overlayLutChangeFinished`) and restored when the item is created
      (`gui_overlay.getOverlayItems`). A LUT made from the overlay color
      (`initColormapOverlayLayerItem`) is not saved: choosing a color
      (`saveOverlayColor`) removes the saved LUT, so the LUT follows the
      color again. The preview while a color is picked in the color dialog
      (`changeOverlayColor`) keeps the saved LUT, and Cancel shows it again
      (`overlayColorDialogRejected`). `Invert black/white` does not change
      the overlay LUTs, so they are restored as saved. The overlay color
      is a channel display setting too (setting `<channel>_rgb`,
      `overlay_color_setting_name`, read on every load by
      `gui_overlay.OverlayChannelsGuiElements.savedOverlayColor`); older
      versions saved it in `settings.csv`, whose row of the same name is
      still read when the channel display settings have none. The alpha
      scrollbar of each overlay channel is saved per channel name too
      (setting `overlay_alpha_slider_<channel>`, see
      `gui_overlay.OverlayChannelsGuiElements.saveOverlayAlpha`).

    The two settings are separate because base images and overlays use
    different LUTs (e.g., grey vs. black to the overlay color). Gradient
    changes made by code that must not be saved (restoring saved LUTs,
    Restore default settings, the overlay color preview) run inside
    `settingChannelLutsFromCode`, so that the save slots skip them. A
    chosen overlay color runs outside it on purpose: its gradient is
    flagged (`isOverlayColorGradient`) and `overlayLutChangeFinished`
    removes the saved overlay LUT. The settings hold JSON objects
    (`colors.gradient_state_to_setting`) that state whether the ticks are
    visible, as every gradient set by code does (`default_image_lut`,
    `colors.get_pg_gradient`): pyqtgraph would otherwise hide the ticks of
    a LUT item that is not shown. Colormaps picked from the LUT menu that
    are not preset gradients are saved with hidden ticks
    (`widgets.baseHistogramLUTitem.colorMapMenuClicked`). Restore default
    settings removes them all, with the saved overlay alphas
    (`removeChannelLutSettings`; `drop_channel_lut_settings` removes the
    older 'img_cmap_*' rows of `settings.csv`), and keeps the overlay
    colors (the overlay LUTs follow them again).
    """
    def customLevelsLutChanged(self, levels, imageItem: pg.ImageItem=None):
        imageItem.setLevels(levels)
        if self.overlayToolbar.isTransparent():
            self.updateTransparentOverlayRgba()
    
    def getPreComputedMinMaxZstack(self, channel: str):
        if channel != self.user_ch_name:
            return None
        
        posData = self.data[self.pos_i]
        zstack_min, zstack_max = np.inf, 0
        for z in range(posData.SizeZ):
            key = (self.pos_i, posData.frame_i, z)
            levels = self.img1.minMaxValuesMapper.get(key)
            if levels is None:
                return
            
            img_min, img_max = levels
            if img_min < zstack_min:
                zstack_min = img_min
            
            if img_max > zstack_max:
                zstack_max = img_max
        
        return (zstack_min, zstack_max)
    
    # @exec_time
    def rescaleIntensitiesLut(
            self, 
            action: QAction=None, 
            setImage: bool=True,
            imageItem=None
        ):
        if not self.isDataLoaded:
            self.logger.info(
                'WARNING: Data is not loaded. '
                'Intensities will be rescaled later.'
            )
            return 

        posData = self.data[self.pos_i]
        if imageItem is None:
            imageItem = self.img1
            channel = self.user_ch_name
            image_data = posData.img_data
        else:
            channel = imageItem.channelName
            _, filename = self.getPathAndFilenameNoExtFromChName(channel, posData)
            image_data = posData.fluo_data_dict[filename]
        
        triggeredByUser = True
        if action is None:
            triggeredByUser = False
            action = imageItem.lutItem.rescaleActionGroup.checkedAction()
        
        how = action.text()
        
        self.df_settings.at[f'how_rescale_intensities_{channel}', 'value'] = how
        self.df_settings.to_csv(self.settings_csv_path)
        
        if how == 'Rescale each 2D image':
            if how == self.rescaleIntensChannelHowMapper[channel]:
                # No need to update since we have autoscale
                return       
            
            imageItem.setEnableAutoLevels(True)
            if setImage:
                imageItem.setImage(imageItem.image)
            return
        
        lutLevelsCh = posData.lutLevels[channel]
        
        if how == 'Rescale across z-stack':            
            imageItem.setEnableAutoLevels(False)
            levels_key = (how, posData.frame_i)
            levels = lutLevelsCh.get(levels_key)
            if levels is None:
                levels = self.getPreComputedMinMaxZstack(channel)
                
            if levels is None:
                image_zstack = image_data[posData.frame_i]
                levels = (image_zstack.min(), image_zstack.max())
            lutLevelsCh[levels_key] = levels
            imageItem.setLevels(levels)
        elif how == 'Rescale across time frames':            
            imageItem.setEnableAutoLevels(False)
            levels_key = (how, None)
            levels = lutLevelsCh.get(levels_key)
            if levels is None:
                levels = (image_data.min(), image_data.max())
                
            lutLevelsCh[levels_key] = levels
            imageItem.setLevels(levels)
        elif how == 'Choose custom levels...':
            autoLevelsEnabledBefore = imageItem.autoLevelsEnabled
            imageItem.setEnableAutoLevels(False)
            if triggeredByUser:
                current_min, current_max = imageItem.getLevels() 
                dtype_max = np.iinfo(image_data.dtype).max
                # max_value = image_data.max()
                # min_value = image_data.min()
                win = apps.SetCustomLevelsLut(
                    init_min_value=round(current_min),
                    init_max_value=round(current_max),
                    maximum_max_value=dtype_max,
                    minimum_min_value=0,
                    parent=self
                )
                win.sigLevelsChanged.connect(
                    partial(self.customLevelsLutChanged, imageItem=imageItem)
                )
                win.exec_()
                if win.cancel:
                    imageItem.setEnableAutoLevels(autoLevelsEnabledBefore)
                    self.logger.info('Custom LUT levels setting cancelled.')
                    self.updateAllImages()
                    return
                selectedLevels = win.selectedLevels
            else:
                selectedLevels = imageItem.getLevels()
            imageItem.setLevels(selectedLevels)
        elif how == 'Do no rescale, display raw image':            
            imageItem.setEnableAutoLevels(False)
            levels_key = (how, None)
            levels = lutLevelsCh.get(levels_key)
            if levels is None:
                dtype_max = np.iinfo(image_data.dtype).max
                levels = (0, dtype_max)
            lutLevelsCh[levels_key] = levels
            imageItem.setLevels(levels)
        
        self.rescaleIntensChannelHowMapper[channel] = how
        
        if setImage:
            imageItem.setImage(imageItem.image)

    def initImgGradRescaleIntensitiesHowPreference(self):
        posData = self.data[self.pos_i]
        channelName = posData.user_ch_name
        if f'how_rescale_intensities_{channelName}' not in self.df_settings.index:
            return
        
        how = self.df_settings.at[
            f'how_rescale_intensities_{channelName}', 'value'
        ]
        self.imgGrad.setRescaleIntensitiesHow(how)

    @contextlib.contextmanager
    def settingChannelLutsFromCode(self):
        """Context for LUT gradient changes made by code that must not be
        saved (restoring saved LUTs, Restore default settings, the overlay
        color preview): the slots that save the LUTs
        (`imgGradLUTfinished_cb`, `overlayLutChangeFinished`) ignore them.
        Not for a chosen overlay color (see `overlayLutChangeFinished`)."""
        wasSetting = self.isSettingChannelLutsFromCode()
        self._isSettingChannelLutsFromCode = True
        try:
            yield
        finally:
            self._isSettingChannelLutsFromCode = wasSetting

    def isSettingChannelLutsFromCode(self):
        return getattr(self, '_isSettingChannelLutsFromCode', False)

    def channelDisplaySettingsPath(self):
        """Path of the channel display settings file (see the class
        docstring), or None when the host has no `settings_csv_path`."""
        settings_csv_path = getattr(self, 'settings_csv_path', None)
        if not settings_csv_path:
            return None
        return channel_display_settings_path(settings_csv_path)

    def channelDisplaySetting(self, name, default=None):
        """Value of the channel display setting `name`, read from the file
        now (another window may have saved it since), or `default`."""
        filepath = self.channelDisplaySettingsPath()
        if filepath is None:
            return default
        settings = read_channel_display_settings(
            filepath, logger=getattr(self, 'logger', None)
        )
        return settings.get(name, default)

    def saveChannelDisplaySettings(self, values=None, remove_prefixes=()):
        """Save the channel display settings `values` ({name: value}, None
        removes the setting) and remove those starting with one of
        `remove_prefixes`, keeping the others in the file (also those saved
        by other windows, see `update_channel_display_settings`)."""
        filepath = self.channelDisplaySettingsPath()
        if filepath is None:
            return False
        return update_channel_display_settings(
            filepath, values=values, remove_prefixes=remove_prefixes,
            logger=getattr(self, 'logger', None)
        )

    def removeChannelLutSettings(self):
        """Remove the saved LUTs of all channels and the saved alphas of the
        overlay channels from the channel display settings (Restore default
        settings). The overlay modes are kept."""
        self.saveChannelDisplaySettings(remove_prefixes=(
            IMAGE_LUT_SETTING_PREFIX, OVERLAY_LUT_SETTING_PREFIX,
            OVERLAY_ALPHA_SETTING_PREFIX
        ))

    def savedChannelLut(self, settingName):
        """Return `(gradient state, bw_inverted)` saved in the channel
        display setting `settingName` (`colors.gradient_state_from_setting`),
        or None when it is missing or invalid (logged)."""
        setting = self.channelDisplaySetting(settingName)
        if setting is None:
            return None
        saved = colors.gradient_state_from_setting(setting)
        if saved is None:
            self.logger.warning(
                f'Ignoring the invalid LUT saved in the setting "{settingName}"'
            )
        return saved

    def isImageLutBwInverted(self):
        """Whether black and white are inverted in the LUT of `imgGrad`
        (`invertBw` checks the LUT item's own action before inverting)."""
        return self.imgGrad.invertBwAction.isChecked()

    def resetImageChannelLut(self):
        """Stop saving the LUT of `imgGrad` for the displayed channel until
        `restoreImageChannelLut` restored the LUT of the channel being
        loaded (call it when loading starts)."""
        self._imageLutChannel = None

    def restoreImageChannelLut(self):
        """Show the LUT saved for the displayed channel (`user_ch_name`) in
        `imgGrad`, or the grey LUT if none was saved, and save the next
        changes of the LUT for this channel.

        Call it at the end of a load, after the first-load `invertBw`
        (`restoreImageChannelLutAfterLoading` runs both): a saved LUT keeps
        the black/white inversion it was saved with and is inverted when
        the current one differs.
        """
        channel = self.user_ch_name
        saved = self.savedChannelLut(image_lut_setting_name(channel))
        if saved is None:
            state, wasInverted = default_image_lut(), False
        else:
            state, wasInverted = saved
        if wasInverted != self.isImageLutBwInverted():
            state = self.imgGrad.invertGradient(state)
        with self.settingChannelLutsFromCode():
            # `setGradient` (not `gradient.restoreState`) also sets the
            # `lastGradient` that `Invert black/white` inverts
            self.imgGrad.setGradient(state)
        self._imageLutChannel = channel

    def restoreImageChannelLutAfterLoading(self):
        """At the end of every load (main GUI `loadingDataCompleted`,
        measurements viewer `initViewerToolsAfterLoading`): the first-load
        `invertBw` when black/white are inverted, then the LUT saved for the
        displayed channel (`restoreImageChannelLut`). The other order would
        invert the restored LUT again and overwrite the saved one."""
        if self.invertBwAction.isChecked() and not self.invertBwAlreadyCalledOnce:
            self.invertBw(True)
        self.restoreImageChannelLut()

    def saveImageChannelLut(self):
        """Save the LUT of `imgGrad` for the displayed channel in the
        channel display settings. Skipped while loading, before
        `restoreImageChannelLut` (see `resetImageChannelLut`).

        Returns
        -------
        bool
            Whether the LUT was saved.
        """
        channel = getattr(self, '_imageLutChannel', None)
        if channel is None or channel != self.user_ch_name:
            return False
        state = self.imgGrad.gradient.saveState()
        # `Invert black/white` inverts `lastGradient`, which presets,
        # custom colormaps and edited ticks do not update
        self.imgGrad.lastGradient = state
        self.saveChannelDisplaySettings({
            image_lut_setting_name(channel): colors.gradient_state_to_setting(
                state, bw_inverted=self.isImageLutBwInverted()
            )
        })
        return True

    def restoreOverlayChannelLut(self, channel, lutItem):
        """Show the LUT saved for the overlay `channel` in its LUT item
        `lutItem`. Without a saved LUT the item keeps the LUT made from the
        overlay color.

        Returns
        -------
        bool
            Whether a saved LUT was restored.
        """
        saved = self.savedChannelLut(overlay_lut_setting_name(channel))
        if saved is None:
            return False
        # `Invert black/white` does not change overlay LUTs: restore as saved
        state, _ = saved
        with self.settingChannelLutsFromCode():
            lutItem.setGradient(state)
        return True

    def overlayLutChangeFinished(self, channel, lutItem, *args):
        """Save the LUT of the overlay `channel` (`sigGradientChangeFinished`
        of the gradient of its LUT item `lutItem`). A LUT made from the
        overlay color (`initColormapOverlayLayerItem`) removes the saved
        LUT instead: the color (setting `<channel>_rgb`) defines it."""
        if self.isSettingChannelLutsFromCode():
            return
        settingName = overlay_lut_setting_name(channel)
        if getattr(lutItem, 'isOverlayColorGradient', False):
            # The overlay color defines it: remove the saved LUT
            setting = None
        else:
            setting = colors.gradient_state_to_setting(
                lutItem.gradient.saveState(),
                bw_inverted=self.invertBwAction.isChecked()
            )
        self.saveChannelDisplaySettings({settingName: setting})
