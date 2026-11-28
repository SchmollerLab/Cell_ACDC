"""Object inspection mixins shared by the main GUI and the measurements
viewer: cell cycle table, object counts, search ID, hover info in the
status bar and the region properties dock. Changes affect both windows;
host-specific behaviour goes through hooks (the main GUI overrides
setAllIDs and countObjects* in gui/main_window/frame_data.py).
"""
import os
import re
import numpy as np
import skimage.segmentation
from qtpy.QtCore import Qt, QEventLoop, QThread
from qtpy.QtWidgets import QDockWidget, QVBoxLayout
from cellacdc import disableWindow
from cellacdc import apps, workers, html_utils
from cellacdc import myutils, widgets
from cellacdc import settings_csv_path
from cellacdc import qutils
from cellacdc import core
from cellacdc.cca_functions import _calc_rot_vol


class CcaTableGuiElements:
    def viewCcaTable(self):
        posData = self.data[self.pos_i]
        zoomIDs = self.getZoomIDs()
        
        df = posData.allData_li[posData.frame_i]['acdc_df']
        current_cca_df = posData.cca_df
        if current_cca_df is None:
            # E.g., measurements viewer on a frame without annotations
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                'There are <b>no cell cycle annotations</b> on frame '
                f'number {posData.frame_i+1}.'
            )
            msg.warning(self, 'No cell cycle annotations', txt)
            return

        if current_cca_df.empty:
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                'Cell cycle annotations\' table is <b>empty</b>.<br>'
            )
            msg.warning(self, 'Table empty', txt)
            return

        if zoomIDs is not None:
            # The measurements viewer's tables keep only cells with complete
            # cell cycle annotations (as its `updateCcaTableWin`)
            zoomIDs = [
                ID for ID in zoomIDs
                if ID in current_cca_df.index and ID in df.index
            ]
            df = df.loc[zoomIDs]
            current_cca_df = current_cca_df.loc[zoomIDs]

        for column in current_cca_df.columns:
            header = (
                '================================================\n'
                f'CURRENT vs STORED `{column}` column '
                f'for frame number {posData.frame_i+1}:\n'
            )
            df_compare = current_cca_df[[column]].copy()
            df_compare[f'STORED_{column}'] = df[column]
            text = f'{header}{df_compare}'
            self.logger.info(text)
        
        if 'cell_cycle_stage' in df.columns:
            cca_df = df[self.cca_df_colnames]
            cca_df = cca_df.merge(
                current_cca_df, how='outer', left_index=True, right_index=True,
                suffixes=('_STORED', '_CURRENT')
            )
            cca_df = cca_df.reindex(sorted(cca_df.columns), axis=1)
            num_cols = len(cca_df.columns)
            for j in range(0,num_cols,2):
                df_j_x = cca_df.iloc[:,j]
                df_j_y = cca_df.iloc[:,j+1]
                if any(df_j_x!=df_j_y):
                    self.logger.info('------------------------')
                    self.logger.info('DIFFERENCES:')
                    diff_df = cca_df.iloc[:,j:j+2]
                    diff_mask = diff_df.iloc[:,0]!=diff_df.iloc[:,1]
                    self.logger.info(diff_df[diff_mask])
        else:
            cca_df = None
            self.logger.info(cca_df)
        self.logger.info('========================')

        df = posData.add_tree_cols_to_cca_df(
            current_cca_df, frame_i=posData.frame_i
        )
        if self.ccaTableWin is None:
            self.ccaTableWin = apps.ViewCcaTableWindow(df, parent=self)
            self.ccaTableWin.show()
            self.ccaTableWin.setGeometryWindow()
            self.ccaTableWin.sigUpdateCcaTable.connect(
                self.onSigUpdateCcaTableWindow
            )
        else:
            self.ccaTableWin.setFocus()
            self.ccaTableWin.activateWindow()
            self.ccaTableWin.updateTable(current_cca_df)

    def getZoomIDs(self, viewRange=None):
        if viewRange is None:
            viewRange = self.ax1.viewRange()
        
        lab = self.currentLab2D
        Y, X = lab.shape
        ((xmin, xmax), (ymin, ymax)) = viewRange
        if xmin <= 0 and ymin <= 0 and xmax >= X and ymax >= Y:
            posData = self.data[self.pos_i]
            return None
        
        xmin = xmin if xmin >= 0 else 0
        ymin = ymin if ymin >= 0 else 0
        xmax = xmax if xmax < X else X
        ymax = ymax if ymax < Y else Y
        
        zoomSlice = (
            slice(round(ymin), round(ymax)), 
            slice(round(xmin), round(xmax)), 
        )
        
        zoomLab = skimage.segmentation.clear_border(lab[zoomSlice])
        zoomRp = self._acdcRegionProps(
            zoomLab, precache_centroids=False
        )
        zoomIDs = [obj.label for obj in zoomRp]
        return zoomIDs

    def onSigUpdateCcaTableWindow(self, *args):
        if not self.isDataLoaded:
            return
        
        if self.ccaTableWin is None:
            return
        
        viewRange = self.ax1.viewRange()
        posData = self.data[self.pos_i]
        zoomIDs = self.getZoomIDs(viewRange=viewRange)
        
        self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

class CountObjectsGuiElements:
    def countObjectsTimelapse(self):
        if self.countObjsWindow is None:
            activeCategories = {
                'In current frame', 
                'In all visited frames', 
                'In entire video', 
                'Unique objects in all visited frames', 
                'Unique objects in entire video'
            }
        else:
            activeCategories = self.countObjsWindow.activeCategories()
            
        posData = self.data[self.pos_i]        
        allCategoryCountMapper, _ = posData.countObjectsInSegmTimelapse(
            activeCategories
        )
        if self.countObjsWindow is None:
            return allCategoryCountMapper 
        
        categoryCountMapper = {}
        for category in activeCategories:
            categoryCountMapper[category] = allCategoryCountMapper[category]
            
        return categoryCountMapper

    def countObjectsSnapshots(self):
        posData = self.data[self.pos_i]
        if self.countObjsWindow is None:
            activeCategories = {
                'In current position', 
                'In all visited positions (current session)',
                'In all visited positions (previous sessions)',
                'In all loaded positions', 
            }
            if self.isSegm3D:
                activeCategories.add('In current z-slice')
        else:
            activeCategories = self.countObjsWindow.activeCategories()

        numObjectsCurrentPos = len(posData.IDs)
        numObjectsAllPos = 0
        numObjectsVisitedPosPrevious = 0
        numObjectsVisitedPosCurrent = 0
        numObjectsCurrentZslice = None
        if 'In current z-slice' in activeCategories:
            numObjectsCurrentZslice = len(
                self._acdcRegionProps(
                    self.currentLab2D, precache_centroids=False
                )
            )
        
        for pos_i, _posData in enumerate(self.data):
            # The measurements viewer creates `allData_li` lazily when a
            # position is first displayed (`initPosAnnotData`)
            stored = getattr(_posData, 'allData_li', None)
            rp = stored[0]['regionprops'] if stored else None
            if rp is not None:
                IDs = rp.IDs
            else:
                # Computed once (live counts run on every image update):
                # the viewer never edits `segm_data` and every load
                # creates new posData objects
                IDs = getattr(_posData, '_countObjectsIDs', None)
                if IDs is None:
                    IDs = self._acdcRegionProps(
                        _posData.segm_data[0], precache_centroids=False
                    ).IDs
                    _posData._countObjectsIDs = IDs
            if os.path.exists(_posData.acdc_output_csv_path):
                numObjectsVisitedPosPrevious += len(IDs)
            if IDs or rp is None:
                # rp None: `IDs` already come from `segm_data[0]`
                numObjs = len(IDs)
                numObjectsAllPos += len(IDs)
            else:
                lab = _posData.segm_data[0]
                rp = self._acdcRegionProps(
                    lab, precache_centroids=False
                )
                numObjs = len(rp)
                numObjectsAllPos += numObjs
                
            if _posData.visited:
                numObjectsVisitedPosCurrent += numObjs
        
        allCategoryCountMapper = {
            'In current position': numObjectsCurrentPos, 
            'In all visited positions (current session)': 
                numObjectsVisitedPosCurrent,
            'In all visited positions (previous sessions)': 
                numObjectsVisitedPosPrevious,
            'In all loaded positions': numObjectsAllPos, 
        }
        if numObjectsCurrentZslice is not None:
            allCategoryCountMapper['In current z-slice'] = (
                numObjectsCurrentZslice
            )
            
        if self.countObjsWindow is None:
            return allCategoryCountMapper 
        
        categoryCountMapper = {}
        for category in activeCategories:
            categoryCountMapper[category] = allCategoryCountMapper[category]
            
        return categoryCountMapper

    def countObjects(self):
        self.logger.info('Counting objects...')
        
        posData = self.data[self.pos_i]
        if posData.SizeT > 1:
            return self.countObjectsTimelapse()
        
        return self.countObjectsSnapshots()

    def updateObjectCounts(self):
        if self.countObjsWindow is None:
            return
        
        if not self.countObjsWindow.isVisible():
            return
        
        if not self.countObjsWindow.livePreviewCheckbox.isChecked():
            return
        
        categoryCountMapper = self.countObjects()
        self.countObjsWindow.updateCounts(categoryCountMapper)

    def countObjectsCb(self, checked):
        if self.countObjsWindow is None:
            categoryCountMapper = self.countObjects()
            self.countObjsWindow = apps.ObjectCountDialog(
                categoryCountMapper=categoryCountMapper, 
                parent=self,
                data=self.data
            )
            self.countObjsWindow.sigShowEvent.connect(self.updateObjectCounts)
            self.countObjsWindow.sigUpdateCounts.connect(self.updateObjectCounts)
            self.countObjsWindow.sigClose.connect(self.countObjsWindowClosed)
        if checked:
            self.countObjsWindow.show()
        else:
            self.countObjsWindow.hide()

    def countObjsWindowClosed(self, event):
        event.ignore()
        self.countObjsButton.setChecked(False)


class SearchIDGuiElements:
    def findID(self, checked=False, ID=None):
        posData = self.data[self.pos_i]
        if ID is None:
            searchIDdialog = apps.FindIDDialog(
                title='Search object by ID',
                msg='Enter object ID to find and highlight',
                parent=self,
                isInteger=True
            )
            searchIDdialog.exec_()
            if searchIDdialog.cancel:
                return

            searchedID = searchIDdialog.EntryID
        else:
            searchedID = ID
        
        if searchedID in posData.IDs:
            self.goToObjectID(searchedID)
            return

        if posData.SizeT == 1:
            self.warnIDnotFound(searchedID)
            return
        
        if searchedID in posData.lost_IDs:
            self.goToLostObjectID(searchedID)
            return
        
        tracked_lost_IDs = self.getTrackedLostIDs()
        if searchedID in tracked_lost_IDs:
            self.goToAcceptedLostObjectID(searchedID)
            return
        
        self.logger.info(f'Searching ID {searchedID} in other frames...')
        
        frame_i_found = self.startSearchIDworker(searchedID)
        if frame_i_found is None:
            self.warnIDnotFound(searchedID)
            return
        
        self.logger.info(
            f'Object ID {searchedID} found at frame n. {frame_i_found+1}.'
        )
        proceed = self.askGoToFrameFoundID(searchedID, frame_i_found)
        if not proceed:
            return

        prev_frame_i = posData.frame_i
        posData.frame_i = frame_i_found
        # The read-only `get_data` of the measurements viewer always loads
        # the frame and returns None (`restorePrevFrameIfNotProceed` is a
        # main GUI method, returning True when `proceed_cca` is True)
        proceed_cca, _ = self.get_data() or (True, None)
        if not proceed_cca:
            self.restorePrevFrameIfNotProceed(proceed_cca, prev_frame_i)
            return

        self.updateAllImages()
        self.updateScrollbars()
        
        self.goToObjectID(searchedID)

    @disableWindow
    def startSearchIDworker(self, searchedID):
        posData = self.data[self.pos_i]
        
        desc = 'Searching ID in all frames...'
        
        parent = getattr(self, 'mainWin', None)
        if parent is None:
            parent = self
        self.progressWin = apps.QDialogWorkerProgress(
            title=desc, parent=parent, pbarDesc=desc
        )
        self.progressWin.mainPbar.setMaximum(posData.SizeT)
        self.progressWin.show(self.app)
        
        self.searchIDthread = QThread()
        self.searchIDworker = workers.SimpleWorker(
            posData, self.searchIDworkerCallback, 
            func_args=(searchedID, )
        )
        self.searchIDworker.frame_i_found = None
        self.searchIDworker.moveToThread(self.searchIDthread)
        
        self.searchIDworker.signals.finished.connect(
            self.searchIDthread.quit
        )
        self.searchIDworker.signals.finished.connect(
            self.searchIDworker.deleteLater
        )
        self.searchIDthread.finished.connect(self.searchIDthread.deleteLater)
        
        self.searchIDworker.signals.critical.connect(
            self.searchIDworkerCritical
        )
        self.searchIDworker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.searchIDworker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.searchIDworker.signals.progress.connect(
            self.workerProgress
        )
        self.searchIDworker.signals.finished.connect(
            self.searchIDworkerFinished
        )
        
        self.searchIDthread.started.connect(self.searchIDworker.run)
        self.searchIDthread.start()
        
        self.searchIDworkerLoop = QEventLoop()
        self.searchIDworkerLoop.exec_()
        
        return self.searchIDworker.frame_i_found

    def searchIDworkerCritical(self, error):
        self.searchIDworkerLoop.exit()
        self.workerCritical(error)

    def searchIDworkerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        
        self.searchIDworkerLoop.exit()

    def searchIDworkerCallback(self, posData, searchedID):
        self.searchIDworker.signals.initProgressBar.emit(0)
        self.setAllIDs()
        self.searchIDworker.signals.initProgressBar.emit(posData.SizeT)
        frame_i_found = None
        for frame_i in range(len(posData.segm_data)):
            if frame_i >= len(posData.allData_li):
                break
            
            rp = posData.allData_li[frame_i]['regionprops']
            if rp is None:
                lab = posData.segm_data[frame_i]
                rp = self._acdcRegionProps(lab, precache_centroids=False)
                posData.allData_li[frame_i]['regionprops'] = rp
            if searchedID in rp.IDs:
                frame_i_found = frame_i
                break
            
            self.searchIDworker.signals.progressBar.emit(1)
            
        self.searchIDworker.frame_i_found = frame_i_found

    def warnIDnotFound(self, searchedID):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            Object ID {searchedID} was not found.<br><br>
        """)
        msg.warning(self, f'ID {searchedID} not found', txt)

    def askGoToFrameFoundID(self, searchedID, frame_i_found):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            Object ID {searchedID} was found at frame n. {frame_i_found+1}.<br><br>
            Do you want to go to frame n. {frame_i_found+1}.
        """)
        noButton, yesButton = msg.information(
            self, f'ID {searchedID} found at frame n. {frame_i_found+1}', txt,
            buttonsTexts=(
                'No, stay on current frame', 
                f'Yes, go to frame n. {frame_i_found+1}'
            )
        )
        return msg.clickedButton == yesButton

    def setAllIDs(self, onlyVisited=False):
        """Collect the IDs of all frames into `posData.allIDs`.

        Positions without stored data (`allData_li` missing or None) are
        skipped and keep their current `allIDs`: in the measurements viewer
        `allData_li` is created lazily by `initPosAnnotData` when a
        position is first displayed (which also sets its `allIDs`).
        """
        for posData in self.data:
            if getattr(posData, 'allData_li', None) is None:
                continue
            posData.allIDs = set()
            for frame_i in range(len(posData.segm_data)):
                if frame_i >= len(posData.allData_li):
                    break

                lab = posData.allData_li[frame_i]['labels']
                if lab is None and onlyVisited:
                    break
                
                rp = posData.allData_li[frame_i]['regionprops']
                if rp is None:
                    lab = posData.segm_data[frame_i]
                    rp = self._acdcRegionProps(
                        lab, precache_centroids=False
                    )
                    posData.allData_li[frame_i]['regionprops'] = rp
                    
                posData.allIDs.update(rp.IDs)

    def onSearchTriggerBlink(self, target):
        button = getattr(self, target, None) if isinstance(target, str) else target
        if button is None:
            return
        
        targets = self.searchWidget.highlightTargets(button)
        blinker = qutils.QControlBlink(targets, qparent=self)
        blinker.start()

    def onSearchId(self, ID):
        if not self.isDataLoaded:
            self.logger.warning(
                'Data not loaded yet. Key pressing events are not connected.'
            )
            return
        self.findID(ID=ID)

    def focusOnSearch(self, dummy=None, blinking=False):
        self.searchWidget.search_input.setFocus()
        if blinking:
            blinker = qutils.QControlBlink(self.searchWidget, qparent=self)
            blinker.start()

class HoverInfoGuiElements:
    def _channelHoverValues(self, descr, channel, value, ff=None):
        if ff is None:
            n_digits = len(str(int(value)))
            ff = myutils.get_number_fstring_formatter(
                type(value), precision=abs(n_digits-5)
            )
        txt = (
            f'<b>{descr} {channel}</b>: value={value:{ff}}'
        )
        return txt

    def _addOverlayHoverValuesFormatted(self, txt, xdata, ydata):
        posData = self.data[self.pos_i]
        if posData.ol_data is None:
            return txt
        
        for filename in posData.ol_data:
            chName = myutils.get_chname_from_basename(
                filename, posData.basename, remove_ext=False
            )
            if chName not in self.checkedOverlayChannels:
                continue
            
            # Only the value at (y, x) is computed (called at every move)
            raw_overlay_value = self.getRawImage(
                filename=filename, yx=(ydata, xdata)
            )

            raw_txt = self._channelHoverValues('Raw', chName, raw_overlay_value)

            txt = f'{txt} | {raw_txt}'
        return txt

    def getActiveToolButton(self):
        for button in self.leftClickButtons:
            if button.isChecked():
                return button

    def getMouseDataCoordsRightImage(self):
        text = self.wcLabel.text()
        if not text:
            return
        
        ax_idx = int(re.findall(r'\(ax(\d)\)', text)[0])
        if ax_idx == 0:
            return
        
        coords = re.findall(r'x=(\d+), y=(\d+) \|', text)[0]
        
        return tuple([int(val) for val in coords])

    def updateValuesStatusBar(self):
        (xl, xr), (yt, yb) = self.ax1ViewRange(integers=True)
        W = round(xr - xl)
        H = round(yb - yt)
        txt = self.wcLabel.text()
        pattern = (
            r'W=.*?, H=.*? \| '
            r'x_left=.*?, y_top=.*? \| '
            r'x_right=.*?, y_bottom=.*? \| '
        )
        replacing = (
            f'W={W:d}, H={H:d} | '
            f'x_left={xl:d}, y_top={yt:d} | '
            f'x_right={xr:d}, y_bottom={yb:d} | '
        )
        txt = re.sub(pattern, replacing, txt)
        self.wcLabel.setText(txt)

    def hoverValuesFormatted(self, xdata, ydata, activeToolButton, is_ax0):  
        (xl, xr), (yt, yb) = self.ax1ViewRange(integers=True)
        W = round(xr - xl)
        H = round(yb - yt)
        ax_idx = 0 if is_ax0 else 1
        txt = (
            f'x={xdata:d}, y={ydata:d} | '
            f'W={W:d}, H={H:d} | '
            f'x_left={xl:d}, y_top={yt:d} | '
            f'x_right={xr:d}, y_bottom={yb:d} | '
            f'(ax{ax_idx})'
        )
        if activeToolButton == self.rulerButton:
            txt = self._addRulerMeasurementText(txt)
            return txt
        elif activeToolButton is not None:
            return txt

        posData = self.data[self.pos_i]

        # Only the value at (y, x) is computed (called at every mouse move)
        raw_value = self.getRawImage(yx=(ydata, xdata))

        ch = self.user_ch_name
        raw_txt = self._channelHoverValues('Raw', ch, raw_value)

        txt = f'{txt} | {raw_txt}'

        txt = self._addOverlayHoverValuesFormatted(txt, xdata, ydata)
        
        ID = self.currentLab2D[ydata, xdata]
        maxID = max(posData.IDs, default=0)

        num_obj = len(posData.IDs)
        lab_txt = (
            f'<b>Objects</b>: ID={ID}, <i>max ID={maxID}, '
            f'num. of objects={num_obj}</i>'
        )
        txt = f'{txt} | {lab_txt}'

        txt = self._addRulerMeasurementText(txt)
        return txt

    def showMirroredCursorToggled(self, checked):
        value = 'Yes' if checked else 'No'
        self.df_settings.at['showMirroredCursor', 'value'] = value
        self.df_settings.to_csv(settings_csv_path)
        
        if not checked:
            self.clearCursors()

    def getRawImage(self, frame_i=None, filename=None, yx=None):
        """Return the raw 2D image displayed (or only its value at (y, x)
        if `yx` is not None, much faster for z-projections)."""
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i
        if filename is None:
            rawImgData = posData.img_data[frame_i]
            isLayer0 = True
        else:
            rawImgData = posData.ol_data[filename][frame_i]
            isLayer0 = False
        if posData.SizeZ > 1:
            rawImg = self.get_2Dimg_from_3D(
                rawImgData, isLayer0=isLayer0, yx=yx
            )
        elif yx is not None:
            rawImg = rawImgData[yx[0], yx[1]]
        else:
            rawImg = rawImgData
        return rawImg


class RegionPropsGuiElements:
    def gui_createRegionPropsDockWidget(self, side=Qt.LeftDockWidgetArea):
        self.propsDockWidget = QDockWidget('Cell-ACDC objects', self)
        self.guiTabControl = widgets.guiTabControl(self.propsDockWidget)

        # self.guiTabControl.setFont(_font)

        self.propsDockWidget.setWidget(self.guiTabControl)
        self.propsDockWidget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable 
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.propsDockWidget.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        
        self.addDockWidget(side, self.propsDockWidget)
        self.propsDockWidget.hide()

    def gui_createShowPropsButton(self, side='left'):
        self.leftSideDocksLayout = QVBoxLayout()            
        self.leftSideDocksLayout.setSpacing(0)
        self.leftSideDocksLayout.setContentsMargins(0,0,0,0)
        self.rightSideDocksLayout = QVBoxLayout()            
        self.rightSideDocksLayout.setSpacing(0)
        self.rightSideDocksLayout.setContentsMargins(0,0,0,0)
        self.showPropsDockButton = widgets.expandCollapseButton()
        self.showPropsDockButton.setDisabled(True)
        self.showPropsDockButton.setFocusPolicy(Qt.NoFocus)
        self.showPropsDockButton.setToolTip('Show object properties')
        if side == 'left':
            self.leftSideDocksLayout.addWidget(self.showPropsDockButton)
        else:
            self.rightSideDocksLayout.addWidget(self.showPropsDockButton)

    def highlightIDonHoverCheckBoxToggled(self, checked):
        doHighlight = (
            self.guiTabControl.highlightCheckbox.isChecked()
            or self.guiTabControl.highlightSearchedCheckbox.isChecked()
        )
        if not doHighlight:
            self.highlightedID = 0
            self.initLookupTableLab()
        else:
            self.highlightedID = self.guiTabControl.propsQGBox.idSB.value()
            self.highlightSearchedID(self.highlightedID, force=True)
            self.updatePropsWidget(self.highlightedID)
        self.updateAllImages()

    def highlightSearchedIDcheckBoxToggled(self, checked):
        self.highlightIDonHoverCheckBoxToggled(checked)
        if checked:
            posData = self.data[self.pos_i]
            self.highlightedID = self.getHighlightedID()
            if self.highlightedID == 0:
                return
            obj = posData.rp.get_obj_from_ID(self.highlightedID)
            if obj is None:
                return
            self.goToZsliceSearchedID(obj) 

    def setHighlightID(self, doHighlight):
        if not doHighlight:
            self.highlightedID = 0
            self.initLookupTableLab()
        else:
            self.highlightedID = self.guiTabControl.propsQGBox.idSB.value()
            self.highlightSearchedID(self.highlightedID, force=True)
            self.updatePropsWidget(self.highlightedID)
        self.updateAllImages()

    def propsWidgetIDvalueChanged(self, ID):
        posData = self.data[self.pos_i]
        if ID == 0:
            self.updatePropsWidget(int(ID))
            return
        
        propsQGBox = self.guiTabControl.propsQGBox
        obj = posData.rp.get_obj_from_ID(ID)
        if obj is None:
            s = f'Object ID {int(ID):d} does not exist'
            propsQGBox.notExistingIDLabel.setText(s)
            return
        
        self.goToZsliceSearchedID(obj)
        self.updatePropsWidget(int(ID))

    def updatePropsWidget(self, ID, fromHover=False):
        if isinstance(ID, str):
            # Function called by currentTextChanged of channelCombobox or
            # additionalMeasCombobox. We set self.currentPropsID = 0 to force update
            ID = self.guiTabControl.propsQGBox.idSB.value()
            self.currentPropsID = -1

        ID = int(ID)
        
        update = (
            self.propsDockWidget.isVisible()
            and ID != 0 and ID!=self.currentPropsID
        )
        if not update:
            return

        posData = self.data[self.pos_i]
        if not hasattr(posData, 'rp'):
            return 
        
        if posData.rp is None:
            self.update_rp() # IDK when can this happen?

        if not posData.IDs:
            # empty segmentation mask
            return

        if fromHover and not self.guiTabControl.highlightCheckbox.isChecked():
            # Do not highlight on hover
            return

        propsQGBox = self.guiTabControl.propsQGBox

        obj = posData.rp.get_obj_from_ID(ID)
        if obj is None:
            s = f'Object ID {int(ID):d} does not exist'
            propsQGBox.notExistingIDLabel.setText(s)
            return

        propsQGBox.notExistingIDLabel.setText('')
        self.currentPropsID = ID
        propsQGBox.idSB.setValue(ID)
        
        doHighlight = (
            self.guiTabControl.highlightCheckbox.isChecked()
            or self.guiTabControl.highlightSearchedCheckbox.isChecked()
        )
        if doHighlight:
            self.highlightSearchedID(ID)
        
        if self.isSegm3D:
            if self.zProjComboBox.currentText() == 'single z-slice':
                local_z = self.z_lab() - obj.bbox[0]
                area_pxl = np.count_nonzero(obj.image[local_z])
            else:
                area_pxl = np.count_nonzero(obj.image.max(axis=0))
        else:
            area_pxl = obj.area

        propsQGBox.cellAreaPxlSB.setValue(area_pxl)

        pixelSizeQGBox = self.guiTabControl.pixelSizeQGBox
        PhysicalSizeX = pixelSizeQGBox.pixelWidthWidget.value()
        PhysicalSizeY = pixelSizeQGBox.pixelHeightWidget.value()
        PhysicalSizeZ = pixelSizeQGBox.voxelDepthWidget.value()
        
        yx_pxl_to_um2 = PhysicalSizeY*PhysicalSizeX

        area_um2 = area_pxl*yx_pxl_to_um2

        propsQGBox.cellAreaUm2DSB.setValue(area_um2)

        if self.isSegm3D:
            PhysicalSizeZ = posData.PhysicalSizeZ
            vol_vox_3D = obj.area
            vol_fl_3D = vol_vox_3D*PhysicalSizeZ*PhysicalSizeY*PhysicalSizeX
            propsQGBox.cellVolVox3D_SB.setValue(vol_vox_3D)
            propsQGBox.cellVolFl3D_DSB.setValue(vol_fl_3D)

        vol_vox, vol_fl = _calc_rot_vol(
            obj, PhysicalSizeY, PhysicalSizeX
        )
        propsQGBox.cellVolVoxSB.setValue(int(vol_vox))
        propsQGBox.cellVolFlDSB.setValue(vol_fl)


        minor_axis_length = max(1, obj.minor_axis_length)
        elongation = obj.major_axis_length/minor_axis_length
        propsQGBox.elongationDSB.setValue(elongation)

        solidity = obj.solidity
        propsQGBox.solidityDSB.setValue(solidity)

        additionalPropName = propsQGBox.additionalPropsCombobox.currentText()
        additionalPropValue = getattr(obj, additionalPropName)
        propsQGBox.additionalPropsCombobox.indicator.setValue(additionalPropValue)

        intensMeasurQGBox = self.guiTabControl.intensMeasurQGBox
        selectedChannel = intensMeasurQGBox.channelCombobox.currentText()
        
        try:
            _, filename = self.getPathAndFilenameNoExtFromChName(selectedChannel, posData)
            image = posData.ol_data_dict[filename][posData.frame_i]
        except Exception as e:
            image = posData.img_data[posData.frame_i]

        if posData.SizeZ > 1 and not self.isSegm3D:
            z = self.zSliceScrollBar.sliderPosition()
            objData = image[z][obj.slice][obj.image]
            img = self.img1.image
        else:
            objData = image[obj.slice][obj.image]
            img = image

        intensMeasurQGBox.minimumDSB.setValue(np.min(objData))
        intensMeasurQGBox.maximumDSB.setValue(np.max(objData))
        intensMeasurQGBox.meanDSB.setValue(np.mean(objData))
        intensMeasurQGBox.medianDSB.setValue(np.median(objData))

        funcDesc = intensMeasurQGBox.additionalMeasCombobox.currentText()
        func = intensMeasurQGBox.additionalMeasCombobox.functions[funcDesc]
        if funcDesc == 'Concentration':
            bkgrVal = np.median(img[posData.lab == 0])
            amount = func(objData, bkgrVal, obj.area)
            value = amount/vol_vox
        elif funcDesc == 'Amount':
            bkgrVal = np.median(img[posData.lab == 0])
            amount = func(objData, bkgrVal, obj.area)
            value = amount
        else:
            value = func(objData)

        intensMeasurQGBox.additionalMeasCombobox.indicator.setValue(value)

    def initPixelSizePropsDockWidget(self):
        posData = self.data[self.pos_i]
        PhysicalSizeX = posData.PhysicalSizeX
        PhysicalSizeY = posData.PhysicalSizeY
        PhysicalSizeZ = posData.PhysicalSizeZ
        self.guiTabControl.initPixelSize(
            PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ
        )

    def showPropsDockWidget(self, checked=False):
        if self.showPropsDockButton.isExpand:
            self.propsDockWidget.setVisible(False)
            self.setHighlightID(False)
        else:
            self.highlightedID = self.guiTabControl.propsQGBox.idSB.value()
            if self.isSegm3D:
                self.guiTabControl.propsQGBox.cellVolVox3D_SB.show()
                self.guiTabControl.propsQGBox.cellVolVox3D_SB.label.show()
                self.guiTabControl.propsQGBox.cellVolFl3D_DSB.show()
                self.guiTabControl.propsQGBox.cellVolFl3D_DSB.label.show()
            else:
                self.guiTabControl.propsQGBox.cellVolVox3D_SB.hide()
                self.guiTabControl.propsQGBox.cellVolVox3D_SB.label.hide()
                self.guiTabControl.propsQGBox.cellVolFl3D_DSB.hide()
                self.guiTabControl.propsQGBox.cellVolFl3D_DSB.label.hide()

            self.propsDockWidget.setVisible(True)
            self.propsDockWidget.setEnabled(True)
        self.updateAllImages()

    def getHighlightedID(self):
        if self.highlightedID > 0:
            return self.highlightedID
        
        doHighlight = (
            self.propsDockWidget.isVisible() 
            and (
                self.guiTabControl.highlightCheckbox.isChecked()
                or self.guiTabControl.highlightSearchedCheckbox.isChecked()
            )
        )
        if not doHighlight:
            return 0
        
        return self.guiTabControl.propsQGBox.idSB.value()

    def goToZsliceSearchedID(self, obj):
        if not self.isSegm3D:
            return
        
        current_z = self.z_lab()
        nearest_nonzero_z = core.nearest_nonzero_z_idx_from_z_centroid(
            obj, current_z=current_z
        )
        if nearest_nonzero_z == current_z:
            self.drawPointsLayers(computePointsLayers=True)
            return
        
        self.zSliceScrollBar.setSliderPosition(nearest_nonzero_z)
        self.update_z_slice(nearest_nonzero_z)
