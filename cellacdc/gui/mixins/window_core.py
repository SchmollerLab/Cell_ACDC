import logging

from qtpy.QtCore import QSettings
import os
import re
from typing import Tuple
from qtpy.QtCore import Qt, QObject, PYQT6
from qtpy.QtGui import QKeySequence
from cellacdc import exception_handler
from cellacdc import apps
from cellacdc import widgets
from . import navigation as gui_navigation, overlay as gui_overlay
from cellacdc import QtScoped
from cellacdc import is_mac
from cellacdc import shortcut_filepath

_logger = logging.getLogger(__name__)

class GUICoreElements():
    # settings helper
    def _get_setting_value(self, index_name, default, cast=int):
        if self.df_settings is None or index_name not in self.df_settings.index:
            return default
        value = self.df_settings.at[index_name, 'value']
        if cast is bool:
            return value in ('Yes', 'True', '1') if isinstance(value, str) else bool(int(value))
        try:
            return cast(value)
        except Exception:
            _logger.warning(
                "Could not cast setting '%s' value '%s' to %s. "
                "Using default '%s'.", index_name, value, cast, default
            )
            return default

    def _set_setting_value(self, index_name, value):
        if self.df_settings is None:
            return
        self.df_settings.at[index_name, 'value'] = value
        self.df_settings.to_csv(self.settings_csv_path)
        
    def get2DRP(
            self, 
            pos_i=None, 
            frame_i=None, 
            slice_i=None, 
            zProjHow=None, 
            depthAxes=None,
        ):
        """Get 2D regionprops for the current position and frame.

        This function returns the 2D regionprops for the specified position, frame, and slice.
        If the segmentation is 3D and a projection is requested, it returns the projected 2D regionprops.

        Parameters
        ----------
        pos_i : int, optional
            The index of the position to get the 2D regionprops for, by default None
        frame_i : int, optional
            The index of the frame to get the 2D regionprops for, by default None
        slice_i : int, optional
            The index of the slice to get the 2D regionprops for, by default None
        zProjHow : str, optional
            The type of z-projection to use ('single z-slice' or other), by default None
        depthAxes : tuple, optional
            The axes to use for depth slicing, by default None

        Returns
        -------
        acdc.RegionProps2D
            The 2D regionprops object for the specified position, frame, and slice.
        """
        posData = self.data[self.pos_i if pos_i is None else pos_i]
        frame_i = posData.frame_i if frame_i is None else frame_i

        if frame_i == posData.frame_i:
            rp = posData.rp
        else:
            rp = posData.allData_li[frame_i]['regionprops']

        if rp is None:
            self.logger.warning(
                "[WARNING] rp for displaying contours was requested before being set properly"
            )
            if frame_i == posData.frame_i:
                lab = posData.lab
            else:
                lab = posData.allData_li[frame_i]['labels']
            rp = self._acdcRegionProps(lab, precache_centroids=False)
            if frame_i == posData.frame_i:
                posData.rp = rp
            posData.allData_li[frame_i]['regionprops'] = rp
        if not self.isSegm3D:
            return rp

        if zProjHow is None:
            zProjHow = self.zProjComboBox.currentText()
        if depthAxes is None:
            depthAxes = self.switchPlaneCombobox.depthAxes()
        
        if zProjHow != 'single z-slice':
            rp_2D = rp.get_proj_rp('most_common', slicing=depthAxes)
            return rp_2D

        if slice_i is None:
            slice_i = self.zSliceScrollBar.sliderPosition()

        rp_2D = rp.get_slice_rp(slice_i, depthAxes)
        return rp_2D
    
    def readSettings(self, setting_suffix=''):
        settings = QSettings('schmollerlab', 'acdc_gui')
        if settings.value(f'geometry{setting_suffix}') is not None:
            self.restoreGeometry(settings.value(f"geometry{setting_suffix}"))
        # self.restoreState(settings.value(f"windowState{setting_suffix}"))

    def saveWindowGeometry(self, setting_suffix=''):
        settings = QSettings('schmollerlab', 'acdc_gui')
        settings.setValue(f"geometry{setting_suffix}", self.saveGeometry())
        # settings.setValue(f"windowState{setting_suffix}", self.saveState())


class WorkersGuiElements:
    def workerProgress(self, text, loggerLevel='INFO'): # used in cca and lin tree
        if self.progressWin is not None:
            self.progressWin.logConsole.append(text)
        self.logger.log(getattr(logging, loggerLevel), text)

    def workerInitProgressbar(self, totalIter):
        self.progressWin.mainPbar.setValue(0)
        if totalIter == 1:
            totalIter = 0
        self.progressWin.mainPbar.setMaximum(totalIter)

    def workerUpdateProgressbar(self, step):
        self.progressWin.mainPbar.update(step)

    @exception_handler
    def workerCritical(self, out: Tuple[QObject, Exception]):
        self.setDisabled(False)
        try:
            worker, error = out
        except TypeError as err:
            error = out
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.logger.info(error)
        try:
            worker.thread().quit()
            worker.deleteLater()
            worker.thread().deleteLater()
        except Exception as err:
            # Worker already closed
            pass

        if isinstance(error, str):
            self.logger.critical(error)
            raise
        else:
            raise error


class ShortcutsGuiElements:
    # Host hook: windows without a delete-object tool (e.g. the
    # measurements viewer) set this to False to hide the "Delete object"
    # row of the shortcut editor and keep their `delObjAction` unchanged
    showDeleteObjectShortcut = True

    def initShortcuts(self):
        from cellacdc import config
        cp = config.ConfigParser()
        if os.path.exists(shortcut_filepath):
            cp.read(shortcut_filepath)
        
        if 'keyboard.shortcuts' not in cp:
            cp['keyboard.shortcuts'] = {}
        
        if cp.has_option('keyboard.shortcuts', 'Zoom out'):
            zoomOutKeyValueStr = cp['keyboard.shortcuts']['Zoom out']
            try:
                self.zoomOutKeyValue = int(zoomOutKeyValueStr)
            except Exception as err:
                self.logger.warning(
                    f'{zoomOutKeyValueStr} is not a valid key '
                    'zooming out action. Restoring default key "H".'
                )
        
        if 'delete_object.action' not in cp:
            self.delObjAction = None
        else:
            delObjKeySequenceText = cp['delete_object.action']['Key sequence']
            delObjButtonText = cp['delete_object.action']['Mouse button']
            delObjQtButton = (
                Qt.MouseButton.LeftButton if delObjButtonText == 'Left click'
                else Qt.MouseButton.MiddleButton
            )
            if not delObjKeySequenceText:
                delObjKeySequence = None
            else:
                delObjKeySequence = widgets.KeySequenceFromText(
                    delObjKeySequenceText
                )
            self.delObjToolAction.setChecked(True)
            self.delObjAction = delObjKeySequence, delObjQtButton
        shortcuts = {}
        for name, widget in self.widgetsWithShortcut.items():
            if name not in cp.options('keyboard.shortcuts'):
                if name in self.defaultMouseShortcuts:
                    button = self.defaultMouseShortcuts[name]
                    if PYQT6:
                        btn_name = button.name
                    else:
                        btn_name = QtScoped.mouse_button_name(button)
                    shortcut_text = f'Mouse {btn_name}' # for easier finding keep this pattern
                    self.mouseBindings[name] = button
                    shortcut = button
                else:
                    if hasattr(widget, 'keyPressShortcut'):
                        key = widget.keyPressShortcut
                        shortcut = widgets.KeySequenceFromText(key)
                    else:
                        shortcut = widget.shortcut()
                    shortcut_text = shortcut.toString()
                cp['keyboard.shortcuts'][name] = shortcut_text
            else:
                shortcut_text = cp['keyboard.shortcuts'][name]
                if shortcut_text.startswith('Mouse '):
                    button_name = shortcut_text.split('Mouse ')[-1]
                    if PYQT6:
                        button = Qt.MouseButton[button_name]
                    else:
                        button =  getattr(Qt, button_name)
                    self.mouseBindings[name] = button
                    shortcut = button
                else:
                    shortcut = widgets.KeySequenceFromText(shortcut_text)
            
            shortcuts[name] = (shortcut_text, shortcut)
        self.setShortcuts(shortcuts, save=False)
        with open(shortcut_filepath, 'w') as ini:
            cp.write(ini)

    def setShortcuts(self, shortcuts: dict, save=True):
        for name, (text, shortcut) in shortcuts.items():
            widget = self.widgetsWithShortcut[name]
            if shortcut is None or text.startswith('Mouse '):
                shortcut = QKeySequence()
            if hasattr(widget, 'keyPressShortcut'):
                widget.keyPressShortcut = shortcut
            else:
                widget.setShortcut(shortcut)
            s = widget.toolTip()
            toolTip = re.sub(r'Shortcut: "(.*)"', f'Shortcut: "{text}"', s)
            widget.setToolTip(toolTip)

        # Buttons hidden by an overflowing toolbar use a copy of their key
        widgets.updateToolbarsOverflowShortcuts(self)

        if not save:
            return
        
        from cellacdc import config
        cp = config.ConfigParser()
        if os.path.exists(shortcut_filepath):
            cp.read(shortcut_filepath)
        
        if 'keyboard.shortcuts' not in cp:
            cp['keyboard.shortcuts'] = {}

        for name, (text, shortcut) in shortcuts.items():
            cp['keyboard.shortcuts'][name] = text
        
        cp['keyboard.shortcuts']['Zoom out'] = str(self.zoomOutKeyValue)
        
        if self.delObjAction is None:
            with open(shortcut_filepath, 'w') as ini:
                cp.write(ini)
            return
    
        delObjKeySequence, delObjQtButton = self.delObjAction
        try:
            if delObjKeySequence is None:
                delObjKeySequenceText = ''
            else:
                delObjKeySequenceText = delObjKeySequence.toString()
                
            delObjKeySequenceText = (
                delObjKeySequenceText
                .encode('ascii', 'ignore')
                .decode('utf-8')
            )
            delObjButtonText = (
                'Left click' if delObjQtButton == Qt.MouseButton.LeftButton
                else 'Middle click'
            )
            cp['delete_object.action'] = {
                'Key sequence': delObjKeySequenceText, 
                'Mouse button': delObjButtonText
            }
        except Exception as err:
            self.logger.warning(
                f'{delObjKeySequence} is not a valid keys sequence for '
                'deleting objects. Setting default action'
            )
            self.delObjAction = None
            cp.remove_section('delete_object.action')
            
        with open(shortcut_filepath, 'w') as ini:
            cp.write(ini)

    def editShortcuts_cb(self, highlighted_shortcut=None):
        if hasattr(self, 'showingShortcutEditorDialog') and self.showingShortcutEditorDialog:
            return
        self.showingShortcutEditorDialog = True
            
        if is_mac:
            delObjKeySequenceText = 'Ctrl'
            delObjButtonText = 'Left click'
        else:
            delObjKeySequenceText = ''
            delObjButtonText = 'Middle click'
            
        if self.delObjAction is not None:
            delObjKeySequence, delObjQtButton = self.delObjAction
            if delObjKeySequence is None:
                delObjKeySequenceText = ''
            else:
                delObjKeySequenceText = delObjKeySequence.toString()
            delObjKeySequenceText = (
                delObjKeySequenceText.encode('ascii', 'ignore').decode('utf-8')
            )
            delObjButtonText = (
                'Left click' if delObjQtButton == Qt.MouseButton.LeftButton
                else 'Middle click'
            )

        hard_shortcuts = self.getHardcodedShortcuts()
            
        win = apps.ShortcutEditorDialog(
            self.widgetsWithShortcut, 
            delObjectKey=delObjKeySequenceText,
            delObjectButton=delObjButtonText,
            zoomOutKeyValue=self.zoomOutKeyValue,
            parent=self,
            mouseBindings=self.mouseBindings,
            hard_shortcuts=hard_shortcuts,
            highlighted_shortcut=highlighted_shortcut,
            show_delete_object_shortcut=self.showDeleteObjectShortcut
        )
        win.exec_()
        if win.cancel:
            self.showingShortcutEditorDialog = False
            return

        self.mouseBindings = win.mouseBindings
        if self.showDeleteObjectShortcut:
            self.delObjAction = win.delObjAction
        self.zoomOutKeyValue = win.zoomOutKeyValue
        self.setShortcuts(win.customShortcuts)
        self.showingShortcutEditorDialog = False


class DefaultSettingsGuiElements:
    def restoreDefaultSettings(self):
        # LUT changes made here are not saved as the channels' LUTs (the
        # saved ones are removed below)
        with self.settingChannelLutsFromCode():
            self._restoreDefaultSettings()

    def _restoreDefaultSettings(self):
        df = self.df_settings
        df.at['contLineWeight', 'value'] = 1
        df.at['mothBudLineSize', 'value'] = 1
        df.at['mothBudLineColor', 'value'] = (255, 165, 0, 255)
        df.at['contLineColor', 'value'] = (205, 0, 0, 220)

        self._updateContColour((205, 0, 0, 220))
        self._updateMothBudLineColour((255, 165, 0, 255))
        self._updateMothBudLineSize(1)
        self._updateContLineThickness()

        df.at['overlaySegmMasksAlpha', 'value'] = 0.3
        df.at['img_cmap', 'value'] = 'grey'
        self.imgCmap = self.imgGrad.cmaps['grey']
        self.imgCmapName = 'grey'
        self.labelsGrad.item.loadPreset('viridis')

        if self.df_settings.at['is_bw_inverted', 'value'] == 'Yes':
            self.invertBw(False, update=False)
            # invertBw does not touch the window's own menu action: sync it
            # without re-triggering invertBw (updateAllImages runs below)
            self.invertBwAction.blockSignals(True)
            self.invertBwAction.setChecked(False)
            self.invertBwAction.blockSignals(False)
            # Measurements viewer: plot colors depend on the background
            recolorSignals = getattr(self, 'recolorSignalsForBackground', None)
            if recolorSignals is not None:
                recolorSignals(False)

        # `loadPreset` (-> `updateLabelsCmap`) rebinds `self.df_settings`
        # and `invertBw` writes to it: continue on the current table, else
        # `restoreState(df)` below restores the old state (e.g. re-inverts)
        df = self.df_settings
        df.at['labels_bkgrColor', 'value'] = (25, 25, 25)
        # Grey LUT for every channel: remove the LUTs and overlay alphas
        # saved per channel (channel display settings) and the older
        # 'img_cmap_*' rows of the displayed image
        self.removeChannelLutSettings()
        self.df_settings = df = gui_navigation.drop_channel_lut_settings(df)

        df = df[~df.index.str.contains('lab_cmap')]
        df.to_csv(self.settings_csv_path)
        self.imgGrad.restoreState(df)
        self.imgGrad.setGradient(gui_navigation.default_image_lut())
        for items in self.overlayLayersItems.values():
            lutItem = items[1]
            lutItem.restoreState(df)
            # Back to the LUT of the overlay color (`restoreState` does not
            # set the overlay LUTs)
            rgb = lutItem.overlayColorButton.color().getRgb()[:3]
            self.initColormapOverlayLayerItem(rgb, lutItem)
            # Saved alphas were dropped with the LUTs. `setValue` by code
            # updates the opacities and saves nothing
            # (`overlayAlphaActionTriggered`)
            alphaScrollBar = items[2]
            alphaScrollBar.setValue(gui_overlay.ALPHA_SCROLLBAR_DEFAULT_VALUE)

        self.labelsGrad.saveState(df)
        self.labelsGrad.restoreState(df, loadCmap=False)

        self.df_settings.to_csv(self.settings_csv_path)
        self.updateAllImages()
