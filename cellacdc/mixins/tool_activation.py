"""Qt view adapter for active-tool workflows."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QEventLoop, QThread, QTimer, Qt

from cellacdc import apps, qutils, widgets, workers
from cellacdc import disableWindow


class ToolActivation:
    """Extracted from guiWin."""

    def _copyAllLostObjects_navigateToFrame(self, frame_i):
        posData = self.data[self.pos_i]
        self.store_data(mainThread=False, autosave=False)

        posData.frame_i = frame_i
        self.get_data()
        self.tracking(wl_update=False)
        self.currentLab2D = self.get_2Dlab(posData.lab)
        self.update_rp()
        self.updateLostNewCurrentIDs()
        self.store_data(mainThread=False, autosave=False)

        self.lostObjContoursImage[:] = 0
        self.lostObjImage[:] = 0
        prev_rp = posData.allData_li[frame_i-1]['regionprops']
        prev_IDs_idxs = posData.allData_li[frame_i-1]['IDs_idxs'] # need to change this when merging with opt.
        for lostID in posData.lost_IDs:
            obj = prev_rp[prev_IDs_idxs[lostID]]
            self.addLostObjsToLostObjImage(obj, lostID, force=True)

    def _copyAllLostObjects_refreshRp(self):
        self.update_rp(draw=False, wl_update=False) # need to change this when merging with opt.

    def _copyAllLostObjects_returnToFrame(self, frame_i):
        posData = self.data[self.pos_i]
        self.store_data(autosave=False, mainThread=False)
        posData.frame_i = frame_i
        self.get_data()

    def addLostObjsToLostObjImage(self, lostObj, lostID, force=False):
        if not force:
            if not self.copyLostObjButton.isChecked():
                return
        
        obj_slice = self.getObjSlice(lostObj.slice)
        obj_image = self.getObjImage(lostObj.image, lostObj.bbox)
        self.lostObjImage[obj_slice][obj_image] = lostID

    def annotLostObjsToggled(self, checked):
        if not self.isDataLoaded:
            return
        self.updateAllImages()

    def clearTempBrushImage(self, forceClearLinked=True):
        if not hasattr(self, 'tempLayerImg1'):
            return
        
        self.tempLayerImg1.setImage(
            self.emptyLab, force_set_linked=forceClearLinked
        )
        
        try:
            self.brushContourImage[:] = 0
        except Exception as err:
            pass
        
        try:
            self.brushImage[:] = 0
        except Exception as err:
            pass

    def connectLeftClickButtons(self):
        self.brushButton.toggled.connect(self.Brush_cb)
        self.curvToolButton.toggled.connect(self.curvTool_cb)
        self.rulerButton.toggled.connect(self.ruler_cb)
        self.eraserButton.toggled.connect(self.Eraser_cb)
        self.wandToolButton.toggled.connect(self.wand_cb)
        self.labelRoiButton.toggled.connect(self.labelRoi_cb)
        self.magicPromptsToolButton.toggled.connect(self.magicPrompts_cb)
        self.drawClearRegionButton.toggled.connect(self.drawClearRegion_cb)
        self.expandLabelToolButton.toggled.connect(self.expandLabelCallback)
        self.addDelPolyLineRoiButton.toggled.connect(self.addDelPolyLineRoi_cb)
        self.manualBackgroundButton.toggled.connect(self.manualBackground_cb)
        self.whitelistIDsButton.toggled.connect(self.whitelistIDs_cb)
        self.zoomRectButton.toggled.connect(self.zoomRectActionToggled)
        self.connectLeftClickButtonsPointsLayersToolbar()

    def connectLeftClickButtonsPointsLayersToolbar(self):
        for toolbar in self.pointsLayersToolbars:
            for action in toolbar.actions()[1:]:
                if not hasattr(action, 'layerTypeIdx'):
                    continue
                if action.layerTypeIdx != 4:
                    continue
                action.button.toggled.connect(
                    self.addPointsByClickingButtonToggled
                )

    def copyAllLostObjects(self, for_future_frame_n, max_overlap_perc):
        if not self.copyLostObjButton.isChecked():
            return

        posData = self.data[self.pos_i]

        desc = 'Copying all lost objects...'

        self.progressWin = apps.QDialogWorkerProgress(
            title=desc, parent=self.mainWin, pbarDesc=desc
        )
        self.progressWin.mainPbar.setMaximum(for_future_frame_n+1)
        self.progressWin.show(self.app)

        self.copyAllLostObjectsThread = QThread()

        self.copyAllLostObjectsWorker = workers.CopyAllLostObjectsWorker(
            self, posData, for_future_frame_n, max_overlap_perc
        )
        self.copyAllLostObjectsWorker.moveToThread(self.copyAllLostObjectsThread)

        self.copyAllLostObjectsWorker.navigateToFrame.connect(
            self._copyAllLostObjects_navigateToFrame,
            Qt.BlockingQueuedConnection
        )
        self.copyAllLostObjectsWorker.returnToFrame.connect(
            self._copyAllLostObjects_returnToFrame,
            Qt.BlockingQueuedConnection
        )
        self.copyAllLostObjectsWorker.copyLostObjectMask.connect(
            self.copyLostObjectMask,
            Qt.BlockingQueuedConnection
        )
        self.copyAllLostObjectsWorker.refreshRp.connect(
            self._copyAllLostObjects_refreshRp,
            Qt.BlockingQueuedConnection
        )
        self.copyAllLostObjectsWorker.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.copyAllLostObjectsWorker.critical.connect(
            self.copyAllLostObjectsWorkerCritical
        )
        self.copyAllLostObjectsWorker.finished.connect(
            self.copyAllLostObjectsThread.quit
        )
        self.copyAllLostObjectsWorker.finished.connect(
            self.copyAllLostObjectsWorker.deleteLater
        )
        self.copyAllLostObjectsThread.finished.connect(
            self.copyAllLostObjectsThread.deleteLater
        )
        self.copyAllLostObjectsWorker.finished.connect(
            self.copyAllLostObjectsWorkerFinished
        )

        self.copyAllLostObjectsThread.started.connect(
            self.copyAllLostObjectsWorker.run
        )
        self.copyAllLostObjectsThread.start()

        self.copyAllLostObjectsWorkerLoop = QEventLoop()
        self.copyAllLostObjectsWorkerLoop.exec_()

    def copyAllLostObjectsWorkerCritical(self, error):
        self.copyAllLostObjectsWorkerLoop.exit()
        self.workerCritical(error)

    def copyAllLostObjectsWorkerFinished(self, output):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

        if output.get('doReinitLastSegmFrame', False):
            self.reInitLastSegmFrame(
                from_frame_i=output.get('last_visited_frame_i'),
                updateImages=False,
                force=True
            )

        if output.get('overlap_warning', False):
            self.blinker = qutils.QControlBlink(
                self.copyLostObjToolbar.maxOverlapNumberControl,
                qparent=self.mainWin
            )
            self.blinker.start()

        self.copyAllLostObjectsWorkerLoop.exit()
        self.update_rp()
        self.updateAllImages()
        self.store_data()

    def copyLostObjContour_cb(self, checked):
        self.copyLostObjToolbar.setVisible(checked)
        
        self.ax1_lostObjScatterItem.hoverLostID = 0
        if not checked:
            return
        
        self.lostObjImage = np.zeros_like(self.currentLab2D)
        self.updateLostContoursImage(0)

    def copyLostObjectMask(self, ID: int):
        posData = self.data[self.pos_i]
        mask = self.lostObjImage == ID
        lab2D = self.get_2Dlab(posData.lab)
        lab2D[mask] = ID
        self.lostObjImage[mask] = 0
        self.set_2Dlab(lab2D)

    def disableNonFunctionalButtons(self):
        if not self.isSegm3D:
            return 

        for item in self.functionsNotTested3D:
            if hasattr(item, 'action'):
                toolButton = item
                action = toolButton.action
                toolButton.setDisabled(True)
            elif hasattr(item, 'toolbar'):
                toolbar = item.toolbar
                action = item
                toolButton = toolbar.widgetForAction(action)
                toolButton.setDisabled(True)    
            else: 
                action = item
            action.setDisabled(True)

    def disconnectLeftClickButtons(self):
        for button in self.LeftClickButtons:
            try:
                button.toggled.disconnect()
            except Exception as e:
                # Not all the LeftClickButtons have toggled connected
                pass

    def getPrevFrameIDs(self, current_frame_i=None):
        posData = self.data[self.pos_i]
        if current_frame_i is None:
            current_frame_i = posData.frame_i
        
        if current_frame_i is None:
            return []
        
        prev_frame_i = current_frame_i - 1
        prevIDs = posData.allData_li[prev_frame_i]['IDs']
        
        if prevIDs:
            return prevIDs
        
        # IDs in previous frame were not stored --> load prev lab from HDD
        prev_lab = self.get_labels(
            from_store=False, 
            frame_i=prev_frame_i,
            return_copy=False
        )
        rp = skimage.measure.regionprops(prev_lab)
        prevIDs = [obj.label for obj in rp]
        return prevIDs

    def hideItemsHoverBrush(self, xy=None, ID=None, force=False):
        if xy is not None:
            x, y = xy
            if x is None:
                return

            xdata, ydata = int(x), int(y)
            Y, X = self.currentLab2D.shape

            if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
                return

        if not self.brushAutoHideCheckbox.isChecked() and not force:
            return
        
        posData = self.data[self.pos_i]
        size = self.brushSizeSpinbox.value()*2

        if xy is not None:
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]

        if self.ax1_lostObjScatterItem.isVisible():
            self.ax1_lostObjScatterItem.setVisible(False)

        if self.ax1_lostTrackedScatterItem.isVisible():
            self.ax1_lostTrackedScatterItem.setVisible(False)
        
        if self.ax2_lostObjScatterItem.isVisible():
            self.ax2_lostObjScatterItem.setVisible(False)

        if self.ax2_lostTrackedScatterItem.isVisible():
            self.ax2_lostTrackedScatterItem.setVisible(False)
            
        # Restore ID previously hovered
        if ID != self.ax1BrushHoverID and not self.isMouseDragImg1:
            try:
                self.restoreHoverObjBrush()
            except Exception as e:
                self.ax1BrushHoverID = 0
                return

        # Hide items hover ID
        if ID != 0:
            self.clearObjContour(ID=ID, ax=0)
            self.clearObjContour(ID=ID, ax=1)
            self.ax1BrushHoverID = ID
        else:
            self.ax1BrushHoverID = 0

    def highlightHoverLostObj(self, modifiers, event):
        noModifier = modifiers == Qt.NoModifier
        if not noModifier:
            return
        
        if not self.copyLostObjButton.isChecked():
            return
        
        if event.isExit():
            return
        
        posData = self.data[self.pos_i]
        x, y = event.pos()
        xdata, ydata = int(x), int(y)
        try:
            hoverLostID = self.lostObjImage[ydata, xdata]
        except IndexError:
            return
        
        self.ax1_lostObjScatterItem.hoverLostID = hoverLostID        
        if hoverLostID == 0:
            self.ax1_lostObjScatterItem.setSize(self.contLineWeight+1)
            self.ax1_lostObjScatterItem.setData([], [])
        else:
            prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
            prev_IDs_idxs = posData.allData_li[posData.frame_i-1]['IDs_idxs']
            lostObj = prev_rp[prev_IDs_idxs[hoverLostID]]
            obj_contours = self.getObjContours(lostObj, all_external=True)
            for cont in obj_contours:
                xx = cont[:,0]
                yy = cont[:,1]
                self.ax1_lostObjScatterItem.addPoints(xx, yy)
            self.ax1_lostObjScatterItem.setSize(self.contLineWeight+2)

    def highlightLostNew(self):
        if self.modeComboBox.currentText() == 'Viewer':
            return
        
        posData = self.data[self.pos_i]
        delROIsIDs = self.getDelRoisIDs()
        
        # self.setAllContoursImages(delROIsIDs=delROIsIDs)
        if posData.frame_i == 0:
            return 

        if not self.annotLostObjsToggle.isChecked():
            return
        
        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        
        if prev_rp is None:
            return

        self.setAllLostObjContoursImage(delROIsIDs=delROIsIDs)        
        self.setAllLostTrackedObjContoursImage(delROIsIDs=delROIsIDs)

    def highlightManualAnnotMode(self, viewBox, viewRange):
        self.ax1.setHighlighted(True)

    def magicPrompts_cb(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.magicPromptsToolButton)
            self.connectLeftClickButtons()
            self.magicPromptsToolbar.setVisible(True)
            self.promptSegmentPointsLayerToolbar.setVisible(True)
            if not self.promptSegmentPointsLayerToolbar.isPointsLayerInit:
                self.addPointsLayerTriggered(
                    toolbar=self.promptSegmentPointsLayerToolbar
                )
        else:
            self.resetCursors()
            self.promptSegmentPointsLayerToolbar.setVisible(False)
            self.magicPromptsToolbar.setVisible(False)

    def manualAnnotPast_cb(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            for _ in range(3):
                self.onEscape(
                    buttonsToNotUncheck=[self.manualAnnotPastButton],
                    doAutoRange=False
                )

            self.brushButton.setChecked(True)
            self.store_data()
            self.manualAnnotState = {
                'editID': self.editIDspinbox.value(),
                'isAutoID': self.autoIDcheckbox.isChecked(),
                'doWarnLostObj': self.warnLostCellsAction.isChecked(),
            }
            self.autoIDcheckbox.setChecked(False)
            self.warnLostCellsAction.setChecked(False)
            hoverID = self.getLastHoveredID()
            if hoverID == 0:
                win = apps.QLineEditDialog(
                    title='Not hovering any ID',
                    msg='You are not hovering on any ID.\n'
                        'Enter the ID that you want to lock.',
                    parent=self, 
                    isInteger=True,
                    defaultTxt=self.setBrushID(return_val=True)
                )
                win.exec_()
                if win.cancel:
                    self.manualAnnotPastButton.setChecked(False)
                    return
                hoverID = win.EntryID
            self.logger.info(
                'Setting manual annotation for ID = '
                f'{hoverID}, at frame n. {posData.frame_i+1}'
            )
            self.editIDspinbox.setValue(hoverID)
            try:
                obj_idx = posData.IDs_idxs[hoverID]
                obj = posData.rp[obj_idx]
                radius = 0.9 * obj.minor_axis_length / 2 # math.sqrt(obj.area/math.pi)*0.9
                self.brushSizeSpinbox.setValue(round(radius))
            except Exception as err:
                pass
            
            self.manualAnnotState['frame_i_to_restore'] = posData.frame_i
            self.manualAnnotState['last_tracked_i'] = (
                self.navigateScrollBar.maximum()-1
            )
            self.ax1.sigRangeChanged.connect(self.highlightManualAnnotMode)
            self.ax1.setHighlighted(True, color='green')
        else:
            self.setStatusBarLabel()  
            self.autoIDcheckbox.setChecked(self.manualAnnotState['isAutoID'])
            self.editIDspinbox.setValue(self.manualAnnotState['editID'])
            self.warnLostCellsAction.setChecked(
                self.manualAnnotState['doWarnLostObj']
            )
            frame_to_restore = self.manualAnnotState.get('frame_i_to_restore')
            if frame_to_restore is None:
                return            
            
            self.store_data()
            self.store_manual_annot_data()
            
            last_tracked_i_to_restore = self.manualAnnotState['last_tracked_i']
            self.manualAnnotRestoreLastTrackedFrame(last_tracked_i_to_restore)
            
            self.logger.info(
                f'Restoring view to frame n. {posData.frame_i+1}...'
            )
            posData.frame_i = frame_to_restore
            self.get_data()
            self.updateAllImages()
            self.updateScrollbars()
            self.ax1.sigRangeChanged.disconnect()
            self.ax1.setHighlighted(False)
            QTimer.singleShot(150, self.autoRange)
        
        self.setManualAnnotModeEnabledTools(checked)

    def onEscape(
            self, 
            isTypingIDFunctionChecked=False, 
            buttonsToNotUncheck=None,
            doAutoRange=True    
        ):
        if buttonsToNotUncheck is None:
            buttonsToNotUncheck = set()
            
        if self.keepIDsButton.isChecked() and self.keptObjectsIDs:
            self.keptObjectsIDs = widgets.KeptObjectIDsList(
                self.keptIDsLineEdit, self.keepIDsConfirmAction
            )
            self.highlightHoverIDsKeptObj(0, 0, hoverID=0)
            QTimer.singleShot(300, self.autoRange)
            return

        if self.brushButton.isChecked() and self.typingEditID:
            self.autoIDcheckbox.setChecked(True)
            self.typingEditID = False
            QTimer.singleShot(300, self.autoRange)
            return
        
        if isTypingIDFunctionChecked and self.typingEditID:
            self.typingEditID = False
            QTimer.singleShot(300, self.autoRange)
            return
        
        if self.labelRoiButton.isChecked() and self.isMouseDragImg1:
            self.isMouseDragImg1 = False
            self.labelRoiItem.setPos((0,0))
            self.labelRoiItem.setSize((0,0))
            self.freeRoiItem.clear()
            QTimer.singleShot(300, self.autoRange)
            return
        
        if self.zoomRectButton.isChecked():
            self.zoomRectCancelled()
            QTimer.singleShot(300, self.autoRange)
            return
        
        self.setUncheckedAllButtons(buttonsToNotUncheck=buttonsToNotUncheck)
        self.setUncheckedAllCustomAnnotButtons()
        self.setUncheckedPointsLayers()
        self.clearTempBrushImage()
        self.isMouseDragImg1 = False
        self.typingEditID = False
        self.clearHighlightedID()
        try:
            self.polyLineRoi.clearPoints()
        except Exception as e:
            pass
        
        if doAutoRange:
            QTimer.singleShot(11, self.autoRange)

    def restoreHoverObjBrush(self):
        posData = self.data[self.pos_i]
        if self.ax1BrushHoverID in posData.IDs:
            obj_idx = posData.IDs_idxs[self.ax1BrushHoverID]
            obj = posData.rp[obj_idx]
            if not self.isObjVisible(obj.bbox):
                return
            
            self.addObjContourToContoursImage(obj=obj, ax=0)
            self.addObjContourToContoursImage(obj=obj, ax=1)

    def setLostNewOldPrevIDs(self):
        posData = self.data[self.pos_i]
        if posData.frame_i == 0:
            posData.lost_IDs = []
            posData.new_IDs = []
            posData.old_IDs = []
            # posData.multiContIDs = set()
            self.titleLabel.setText('Looking good!', color=self.titleColor)
            return []
        
        # elif self.modeComboBox.currentText() == 'Viewer':
        #     pass
        
        out = self.updateLostNewCurrentIDs()
        lost_IDs, new_IDs, IDs_with_holes, tracked_lost_IDs, curr_delRoiIDs = (
            out
        )
        self.setTitleText(
            lost_IDs, new_IDs, IDs_with_holes, tracked_lost_IDs
        )
        return curr_delRoiIDs

    def setManualAnnotModeEnabledTools(self, enabled):
        for action in self.editToolBar.actions():
            toolButton = self.editToolBar.widgetForAction(action)
            if toolButton in self.manulAnnotToolButtons:
                continue
            
            toolButton.setDisabled(enabled)  
            action.setDisabled(enabled) 

    def setTitleFormatter(self, htmlTxt_li, htmlTxtFull_li, pretxt, color, IDs):
        if not IDs:
            return htmlTxt_li, htmlTxtFull_li
        
        if isinstance(IDs, set):
            IDs = list(IDs)

        trim_IDs = myutils.get_trimmed_list(IDs)
        txt = f'{pretxt}: {trim_IDs}'
        txt_full = f'{pretxt}:<br>{IDs}'

        txt = f'<font color="{color}">{txt}</font>'
        txt_full = f'<font color="{color}">{txt_full}</font>'

        htmlTxt_li.append(txt)
        htmlTxtFull_li.append(txt_full)

        return htmlTxt_li, htmlTxtFull_li

    def setTitleText(   
            self, lost_IDs=None, new_IDs=None, IDs_with_holes=None, 
            tracked_lost_IDs=None
        ):
        if self.manualAnnotPastButton.isChecked():
            lockedID = self.editIDspinbox.value()
            frame_to_restore = self.manualAnnotState.get('frame_i_to_restore')
            txt = (
                f'Locked ID {lockedID} '
                f'since frame n. {frame_to_restore+1}'
            )
            htmlTxt = f'<font color="orange">{txt}</font>'
            self.titleLabel.setText(htmlTxt)
            return
        
        mode = self.modeComboBox.currentText()
        try:
            posData = self.data[self.pos_i]
            posData.segm_data[posData.frame_i]
            prev_segmented = True
        except IndexError:
            prev_segmented = False
            
        if prev_segmented:
            htmlTxt_li = []
            htmlTxtFull_li = []
        else:
            htmlTxt = f'<font color="white">Never segmented frame. </font>'
            self.titleLabel.setText(htmlTxt)
            self.titleLabel.setToolTip(htmlTxt)
            return
        
        if mode != 'Normal division: Lineage tree':
            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'IDs lost', 'orange', lost_IDs
            )
            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'New IDs', 'red', new_IDs
            )
            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'Acc. IDs lost', 'green', 
                tracked_lost_IDs
            )

            for i, htmlTxtFull in enumerate(htmlTxtFull_li):
                htmlTxtFull_li[i] = htmlTxtFull.replace('Acc.', 'Accepted')

            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'IDs with holes', 'red', 
                IDs_with_holes
            )
        else:
            try:
                cells_with_parent, orphan_cells, lost_cells = self.lineage_tree.export_lin_tree_info(posData.frame_i)
            except IndexError or KeyError:
                title = 'Processing lineage tree...'
                htmlTxt = f'<font color="{self.titleColor}">{title}</font>'
                self.titleLabel.setText(htmlTxt)
                self.titleLabel.setToolTip(htmlTxt)
                return
            except AttributeError:
                title = 'Lineage tree still initializing...'
                htmlTxt = f'<font color="{self.titleColor}">{title}</font>'
                self.titleLabel.setText(htmlTxt)
                self.titleLabel.setToolTip(htmlTxt)
                return
            
            parent_cell_txt_raw = []
            if cells_with_parent:
                # aggregate same parents
                parent_cell_groups = dict()
                for cell, parent in cells_with_parent:
                    if parent not in parent_cell_groups:
                        parent_cell_groups[parent] = []
                    parent_cell_groups[parent].append(cell)
                for parent, daughters in parent_cell_groups.items():
                    cells_str = ','.join([str(daughter) for daughter in daughters])
                    parent_cell_txt_raw.append(f'({parent}>{cells_str})')

            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'New w/out mother', 'red', 
                orphan_cells
            )
            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'Lost', 'yellow', lost_cells
            )
            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'Parent > Cell', 'green', 
                parent_cell_txt_raw
            )

        if not htmlTxt_li:
            title = 'Looking good'
            htmlTxt = f'<font color="{self.titleColor}">{title}</font>'
            self.titleLabel.setText(htmlTxt)
            self.titleLabel.setToolTip(htmlTxt)
            return

        htmlTxt = ', '.join(htmlTxt_li)
        htmlTxtFull = '<br>'.join(htmlTxtFull_li)

        self.titleLabel.setText(htmlTxt)
        self.titleLabel.setToolTip(htmlTxtFull)

    def setUncheckedAllButtons(self, buttonsToNotUncheck=None):
        self.clickedOnBud = False
        if buttonsToNotUncheck is None:
            buttonsToNotUncheck = set()
            
        try:
            self.BudMothTempLine.setData([], [])
        except Exception as e:
            pass
        for button in self.checkableButtons:
            if button in buttonsToNotUncheck:
                continue
            button.setChecked(False)
        
        if self.countObjsButton not in buttonsToNotUncheck:
            self.countObjsButton.setChecked(False)
        self.splineHoverON = False
        self.tempSegmentON = False
        self.isRightClickDragImg1 = False
        self.clearCurvItems(removeItems=False)

    def setUncheckedAllCustomAnnotButtons(self):
        for button in self.customAnnotDict.keys():
            button.setChecked(False)

    def setUncheckedPointsLayers(self):
        self.togglePointsLayerAction.setChecked(False)
        self.magicPromptsToolButton.setChecked(False)

    def uncheckLeftClickButtons(self, sender):
        for button in self.LeftClickButtons:
            if button != sender:
                button.setChecked(False)
        
        if button != self.labelRoiButton:
            # self.labelRoiButton is disconnected so we manually call uncheck
            self.labelRoi_cb(False)
        self.secondLevelToolbar.setVisible(True)
        for toolbar in self.controlToolBars:
            try:
                toolbar.keepVisibleWhenActive
                if toolbar.isVisible():
                    self.secondLevelToolbar.setVisible(False)
                    continue
            except:
                pass
            toolbar.setVisible(False) 
        
        self.enableSizeSpinbox(False)
        if sender is not None:
            self.keepIDsButton.setChecked(False)

    def uncheckQButton(self, button):
        # Manual exclusive where we allow to uncheck all buttons
        for b in self.checkableQButtonsGroup.buttons():
            if b != button:
                b.setChecked(False)

    def updateBrushCursor(self, x, y, isHoverImg1=True):
        if x is None:
            return

        xdata, ydata = int(x), int(y)
        _img = self.currentLab2D
        Y, X = _img.shape

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        size = self.brushSizeSpinbox.value()*2
        self.setHoverToolSymbolData(
            [x], [y], self.activeBrushCircleCursors(isHoverImg1),
            size=size
        )
        self.setHoverToolSymbolColor(
            xdata, ydata, self.ax2_BrushCirclePen,
            self.activeBrushCircleCursors(isHoverImg1),
            self.brushButton, brush=self.ax2_BrushCircleBrush
        )

    def updateHighlightedAxis(self):
        if not self.manualAnnotPastButton.isChecked():
            return
        
        frame_to_restore = self.manualAnnotState.get('frame_i_to_restore')
        posData = self.data[self.pos_i]
        if posData.frame_i == frame_to_restore:
            color = 'green'
        elif posData.frame_i < frame_to_restore:
            color = 'gold'
        else:
            color = 'red'
        
        self.ax1.setHighlightingRectItemsColor(color)

    def updateLostNewCurrentIDs(self):
        posData = self.data[self.pos_i]
        
        prev_IDs = self.getPrevFrameIDs()  
        tracked_lost_IDs = self.getTrackedLostIDs()
        curr_IDs = posData.IDs
        curr_delRoiIDs = self.getStoredDelRoiIDs()
        prev_delRoiIDs = self.getStoredDelRoiIDs(frame_i=posData.frame_i-1)
        lost_IDs = [
            ID for ID in prev_IDs if ID not in curr_IDs
            and ID not in prev_delRoiIDs and ID not in tracked_lost_IDs
        ]
        new_IDs = [
            ID for ID in curr_IDs if ID not in prev_IDs 
            and ID not in curr_delRoiIDs
        ]
        IDs_with_holes = []
        posData.lost_IDs = lost_IDs
        posData.new_IDs = new_IDs
        posData.old_IDs = prev_IDs
        posData.IDs = curr_IDs
        
        out = (
            lost_IDs, new_IDs, IDs_with_holes, tracked_lost_IDs, curr_delRoiIDs
        )
        return out

    def wand_cb(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.wandToolButton)
            self.connectLeftClickButtons()
            self.wandControlsToolbar.setVisible(True)
            # self.secondLevelToolbar.setVisible(False)
        else:
            self.resetCursors()
            # self.secondLevelToolbar.setVisible(True)
            self.wandControlsToolbar.setVisible(False)
