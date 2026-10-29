"""Navigation GUI elements (frames, z-slices, objects, recent paths, LUT) 
shared between the main GUI (`gui.guiWin`) and other windows (e.g., 
`measurements_viewer.SignalsViewerWin`).

The classes are mixins meant to be inherited together with `QMainWindow`. 
They expect the host window to define:
    - `self.data`: list of `load.loadData` objects
    - `self.pos_i`: index of the current position in `self.data`
    - `self.ax1`: the `pyqtgraph.PlotItem` displaying the image
    - `self.isSegm3D`: whether the segmentation masks are 3D
    - `self.openRecentMenu`: `QMenu` populated with the recent paths
    - `self.openRecentFile(path)`: method called to load a recent path
"""
from functools import partial

import numpy as np
import pyqtgraph as pg

from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QLabel, QCheckBox, QHBoxLayout, QAction

from . import widgets, apps, myutils

_font = QFont()
_font.setPixelSize(11)

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
        self.SizeZlabel.setText(f'/{SizeZ}')
    
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
    
    def get_2Dimg_from_switched_plane(self, imgData):
        """Return the 2D image sliced along the depth axis selected with 
        `self.switchPlaneCombobox` or None if the viewed plane is 'xy'
        """
        axis_slice = self.zSliceScrollBar.sliderPosition()
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if depthAxes == 'x':
            return imgData[:, :, axis_slice].copy()
        elif depthAxes == 'y':
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
        recentPaths, dates = myutils.get_recent_paths(self)
        
        # Step 2. Dynamically create 20 actions
        actions = []
        for path in recentPaths[:20]:
            action = QAction(path, self)
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
        for action in self.openRecentMenu.actions():
            if path == action.text():
                return
        else:
            action = QAction(path, self)
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
    `self.rescaleIntensChannelHowMapper`, `self.updateAllImages()` and 
    `posData.lutLevels` (dict of {channel: {}}) for each loaded `posData`.
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
