"""Qt view adapter for annotation display workflows."""

from __future__ import annotations

import re

from cellacdc import _palettes, apps, html_utils, widgets

from typing import Mapping

GREEN_HEX = _palettes.green()


class AnnotationDisplay:
    """Extracted from guiWin."""

    def activateAnnotations(self):
        if self.annotContourCheckbox.isChecked():
            return
        if self.annotSegmMasksCheckbox.isChecked():
            return
        
        self.annotSegmMasksCheckbox.setChecked(True)
        self.setDrawAnnotComboboxText()

    def annotGenNumTreeToggled(self, checked):
        self.textAnnot[0].setGenNumTreeAnnotationsEnabled(checked)

    def annotLabelIDtreeToggled(self, checked):
        self.textAnnot[0].setLabelTreeAnnotationsEnabled(checked)

    def annotOptionClicked(self, clicked=True, sender=None, saveSettings=True):
        if sender is None:
            sender = self.sender()
        # First manually set exclusive with uncheckable
        clickedIDs = sender == self.annotIDsCheckbox
        clickedCca = sender == self.annotCcaInfoCheckbox
        clickedMBline = sender == self.drawMothBudLinesCheckbox
        if self.annotIDsCheckbox.isChecked() and clickedIDs:
            if self.annotCcaInfoCheckbox.isChecked():
                self.annotCcaInfoCheckbox.setChecked(False)
            if self.drawMothBudLinesCheckbox.isChecked():
                self.drawMothBudLinesCheckbox.setChecked(False)
        
        if self.annotCcaInfoCheckbox.isChecked() and clickedCca:
            if self.annotIDsCheckbox.isChecked():
                self.annotIDsCheckbox.setChecked(False)
            if self.drawMothBudLinesCheckbox.isChecked():
                self.drawMothBudLinesCheckbox.setChecked(False)
        
        if self.drawMothBudLinesCheckbox.isChecked() and clickedMBline:
            if self.annotIDsCheckbox.isChecked():
                self.annotIDsCheckbox.setChecked(False)
            if self.annotCcaInfoCheckbox.isChecked():
                self.annotCcaInfoCheckbox.setChecked(False)
        
        clickedCont = sender == self.annotContourCheckbox
        clickedSegm = sender == self.annotSegmMasksCheckbox
        if self.annotContourCheckbox.isChecked() and clickedCont:
            if self.annotSegmMasksCheckbox.isChecked():
                self.annotSegmMasksCheckbox.setChecked(False)
        
        if self.annotSegmMasksCheckbox.isChecked() and clickedSegm:
            if self.annotContourCheckbox.isChecked():
                self.annotContourCheckbox.setChecked(False)
        
        clickedDoNot = sender == self.drawNothingCheckbox
        if clickedDoNot:
            self.annotIDsCheckbox.setChecked(False)
            self.annotCcaInfoCheckbox.setChecked(False)
            self.annotContourCheckbox.setChecked(False)
            self.annotSegmMasksCheckbox.setChecked(False)
            self.drawMothBudLinesCheckbox.setChecked(False)
            self.annotNumZslicesCheckbox.setChecked(False)
        else:
            self.drawNothingCheckbox.setChecked(False)
        
        if sender == self.annotNumZslicesCheckbox:
            self.annotIDsCheckbox.setChecked(True)
            self.drawNothingCheckbox.setChecked(False)
        
        self.setDrawAnnotComboboxText(saveSettings=saveSettings)

    def annotOptionClickedRight(
            self, clicked=True, sender=None, saveSettings=True
        ):
        if sender is None:
            sender = self.sender()
        # First manually set exclusive with uncheckable
        clickedIDs = sender == self.annotIDsCheckboxRight
        clickedCca = sender == self.annotCcaInfoCheckboxRight
        clickedMBline = sender == self.drawMothBudLinesCheckboxRight
        if self.annotIDsCheckboxRight.isChecked() and clickedIDs:
            if self.annotCcaInfoCheckboxRight.isChecked():
                self.annotCcaInfoCheckboxRight.setChecked(False)
            if self.drawMothBudLinesCheckboxRight.isChecked():
                self.drawMothBudLinesCheckboxRight.setChecked(False)
        
        if self.annotCcaInfoCheckboxRight.isChecked() and clickedCca:
            if self.annotIDsCheckboxRight.isChecked():
                self.annotIDsCheckboxRight.setChecked(False)
            if self.drawMothBudLinesCheckboxRight.isChecked():
                self.drawMothBudLinesCheckboxRight.setChecked(False)
        
        if self.drawMothBudLinesCheckboxRight.isChecked() and clickedMBline:
            if self.annotIDsCheckboxRight.isChecked():
                self.annotIDsCheckboxRight.setChecked(False)
            if self.annotCcaInfoCheckboxRight.isChecked():
                self.annotCcaInfoCheckboxRight.setChecked(False)
        
        clickedCont = sender == self.annotContourCheckboxRight
        clickedSegm = sender == self.annotSegmMasksCheckboxRight
        if self.annotContourCheckboxRight.isChecked() and clickedCont:
            if self.annotSegmMasksCheckboxRight.isChecked():
                self.annotSegmMasksCheckboxRight.setChecked(False)
        
        if self.annotSegmMasksCheckboxRight.isChecked() and clickedSegm:
            if self.annotContourCheckboxRight.isChecked():
                self.annotContourCheckboxRight.setChecked(False)
        
        clickedDoNot = sender == self.drawNothingCheckboxRight
        if clickedDoNot:
            self.annotIDsCheckboxRight.setChecked(False)
            self.annotCcaInfoCheckboxRight.setChecked(False)
            self.annotContourCheckboxRight.setChecked(False)
            self.annotSegmMasksCheckboxRight.setChecked(False)
            self.drawMothBudLinesCheckboxRight.setChecked(False)
            self.annotNumZslicesCheckboxRight.setChecked(False)
        else:
            self.drawNothingCheckboxRight.setChecked(False)
        
        if sender == self.annotNumZslicesCheckboxRight:
            self.annotIDsCheckboxRight.setChecked(True)
            self.drawNothingCheckboxRight.setChecked(False)

        self.setDrawAnnotComboboxTextRight(saveSettings=saveSettings)

    def annotateRightHowCombobox_cb(self, idx):
        how = self.annotateRightHowCombobox.currentText()
        saveSettings = True
        if hasattr(self.annotateRightHowCombobox, 'saveSettings'):
            saveSettings = self.annotateRightHowCombobox.saveSettings

        if saveSettings:
            self.df_settings.at['how_draw_right_annotations', 'value'] = how
            self.df_settings.to_csv(self.settings_csv_path)

        mode = self.modeComboBox.currentText()
        isCcaAnnot = (
            self.annotCcaInfoCheckboxRight.isChecked() and 
            mode != 'Normal division: Lineage tree'
        )
        isIDAnnot = (self.annotIDsCheckboxRight.isChecked() or (
            self.annotCcaInfoCheckboxRight.isChecked() and
            mode == 'Normal division: Lineage tree'
        ))
        self.textAnnot[1].setCcaAnnot(
            isCcaAnnot
        )

        self.textAnnot[1].setLabelAnnot(
            isIDAnnot
        )
        if not self.isDataLoading:
            self.updateAllImages()

    def annotate_rip_and_bin_IDs(self, updateLabel=False):
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if self.switchPlaneCombobox.isEnabled() and depthAxes != 'z':
            return 
        
        posData = self.data[self.pos_i]
        binnedIDs_xx = []
        binnedIDs_yy = []
        ripIDs_xx = []
        ripIDs_yy = []
        for obj in posData.rp:
            obj.excluded = obj.label in posData.binnedIDs
            obj.dead = obj.label in posData.ripIDs
            if not self.isObjVisible(obj.bbox):
                continue
            
            if obj.excluded:
                y, x = self.getObjCentroid(obj.centroid)
                binnedIDs_xx.append(x)
                binnedIDs_yy.append(y)
                if updateLabel:
                    self.getObjOptsSegmLabels(obj)
                    how = self.drawIDsContComboBox.currentText()
            
            if obj.dead:
                y, x = self.getObjCentroid(obj.centroid)
                ripIDs_xx.append(x)
                ripIDs_yy.append(y)
                if updateLabel:
                    self.getObjOptsSegmLabels(obj)
                    how = self.drawIDsContComboBox.currentText()
        
        self.ax2_binnedIDs_ScatterPlot.setData(binnedIDs_xx, binnedIDs_yy)
        self.ax2_ripIDs_ScatterPlot.setData(ripIDs_xx, ripIDs_yy)
        self.ax1_binnedIDs_ScatterPlot.setData(binnedIDs_xx, binnedIDs_yy)
        self.ax1_ripIDs_ScatterPlot.setData(ripIDs_xx, ripIDs_yy)

    def applyToolNewFrameActionToggled(self, checked, toolName=None):
        if toolName is None:
            parentToolButton = self.sender().parent()
            toolName = re.findall(r'Name: (.*)', parentToolButton.toolTip())[0]
        toolName = toolName.strip()
        button = self.applyToolNewFrameButtons[toolName]
        toolName = toolName.replace(' ', '_')
        settingName = f'{toolName}_applyNewFrame'
        if checked:
            self.df_settings.at[settingName, 'value'] = 'applyNewFrame'
            button.setStyleSheet(f'background-color: {GREEN_HEX}')
        else:
            self.df_settings = self.df_settings.drop(
                index=settingName, errors='ignore'
            )
            button.setStyleSheet('background-color: none')
        self.df_settings.to_csv(self.settings_csv_path)

    def areContoursRequested(self, ax):
        if ax == 0 and self.annotContourCheckbox.isChecked():
            return True

        if ax == 1:
            if not self.labelsGrad.showRightImgAction.isChecked():
                return False

            isRightDifferentAnnot = self.rightBottomGroupbox.isChecked()
            areContRequestedRight = self.annotContourCheckboxRight.isChecked()
           
            if isRightDifferentAnnot and areContRequestedRight:
                return True
            
            areContRequestedLeft = self.annotContourCheckbox.isChecked()
            if not isRightDifferentAnnot and areContRequestedLeft:
                return True
        return False

    def areMothBudLinesRequested(self, ax):
        if ax == 0:
            if self.annotCcaInfoCheckbox.isChecked():
                return True
            if self.drawMothBudLinesCheckbox.isChecked():
                return True
        else:
            if not self.labelsGrad.showRightImgAction.isChecked():
                return False
            
            isRightDifferentAnnot = self.rightBottomGroupbox.isChecked()
            areLinesRequestedRight = (
                self.annotCcaInfoCheckboxRight.isChecked()
                or self.drawMothBudLinesCheckboxRight.isChecked()
            )
            
            if isRightDifferentAnnot and areLinesRequestedRight:
                return True
        
            areLinesRequestedLeft = (
                self.drawMothBudLinesCheckbox.isChecked()
                or self.annotCcaInfoCheckbox.isChecked()
            )
            if not isRightDifferentAnnot and areLinesRequestedLeft:
                return True
        return False

    def autoPilotToggled(self, checked):
        self.autoPilotZoomToObjToolbar.setVisible(checked)
        if checked:
            self.autoPilotZoomToObjToggle.setChecked(False)
            self.autoPilotZoomToObjToggle.toggle()

    def changeTextResolution(self):
        mode = 'high' if self.highLowResAction.isChecked() else 'low'
        self.logger.info(
            f'Switching to {mode} for the text annnotations...'
        )
        self.pxModeAction.setDisabled(not self.highLowResAction.isChecked())
        if not self.isDataLoaded:
            return
        
        self.setAllIDs()
        posData = self.data[self.pos_i]
        allIDs = posData.allIDs
        img_shape = self.img1.image.shape[:2]
        self.textAnnot[0].changeResolution(mode, allIDs, self.ax1, img_shape)
        self.textAnnot[1].changeResolution(mode, allIDs, self.ax2, img_shape)
        self.updateAllImages()

    def clearAllCellToCellLines(self):
        self.ax1_newMothBudLinesItem.setData([], [])
        self.ax1_oldMothBudLinesItem.setData([], [])
        self.ax2_newMothBudLinesItem.setData([], [])
        self.ax2_oldMothBudLinesItem.setData([], [])

    def clearAnnotItems(self):
        self.textAnnot[0].clear()
        self.textAnnot[1].clear()

    def drawAllLineageTreeLines(self):
        """
        Draw all lineage tree lines on the GUI.

        This method retrieves the lineage tree data and draws the lineage tree lines
        connecting cells and their respective mothers when the mother has split.
        """
        if self.lineage_tree is None:
            return

        if len(self.lineage_tree.frames_for_dfs) < 2:
            return

        self.clearAllCellToCellLines()
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        lin_tree_df = posData.allData_li[frame_i]['acdc_df']
        lin_tree_df_prev = posData.allData_li[frame_i-1]['acdc_df']
        rp = posData.rp
        prev_rp = posData.allData_li[frame_i-1]['regionprops']

        self.setTitleText()

        new_cells = lin_tree_df.index.difference(lin_tree_df_prev.index) # I could use this for the if already but this is probably faster for frames where nothing changes
        if new_cells.shape[0] == 0:
            return
        
        for ax in (0, 1):
            if not self.areMothBudLinesRequested(ax):
                continue

            for ID in new_cells:
                curr_obj = myutils.get_obj_by_label(rp, ID)
                lin_tree_df_ID = lin_tree_df.loc[ID]

                # lin_tree_df_mother_ID = lin_tree_df_prev.loc[lin_tree_df_ID["parent_ID_tree"]]
                if lin_tree_df_ID["parent_ID_tree"] == -1: # make sure that new obj where the parents are not known get skipped
                    continue
                
                mother_obj = myutils.get_obj_by_label(prev_rp, lin_tree_df_ID["parent_ID_tree"])

                emerg_frame_i = lin_tree_df_ID["emerg_frame_i"]
                isNew = emerg_frame_i == frame_i

                self.drawObjLin_TreeMothBudLines(ax, curr_obj, mother_obj, isNew, ID=ID)

    def drawAllMothBudLines(self):
        posData = self.data[self.pos_i]
        for obj in posData.rp:
            self.drawObjMothBudLines(obj, posData, ax=0)
            self.drawObjMothBudLines(obj, posData, ax=1)

    def drawAnnotCombobox_to_options(self):
        self.uncheckAnnotOptions()

        # Left
        how = self.drawIDsContComboBox.currentText()
        if how.find('IDs') != -1:
            self.annotIDsCheckbox.setChecked(True)
        if how.find('cell cycle info') != -1:
            self.annotCcaInfoCheckbox.setChecked(True) 
        if how.find('contours') != -1:
            self.annotContourCheckbox.setChecked(True) 
        if how.find('segm. masks') != -1:
            self.annotSegmMasksCheckbox.setChecked(True) 
        if how.find('mother-bud lines') != -1:
            self.drawMothBudLinesCheckbox.setChecked(True) 
        if how.find('nothing') != -1:
            self.drawNothingCheckbox.setChecked(True)
        
        # Right
        how = self.annotateRightHowCombobox.currentText()
        if how.find('IDs') != -1:
            self.annotIDsCheckboxRight.setChecked(True)
        if how.find('cell cycle info') != -1:
            self.annotCcaInfoCheckboxRight.setChecked(True) 
        if how.find('contours') != -1:
            self.annotContourCheckboxRight.setChecked(True) 
        if how.find('segm. masks') != -1:
            self.annotSegmMasksCheckboxRight.setChecked(True) 
        if how.find('mother-bud lines') != -1:
            self.drawMothBudLinesCheckboxRight.setChecked(True) 
        if how.find('nothing') != -1:
            self.drawNothingCheckboxRight.setChecked(True)

    def drawIDsContComboBox_cb(self, idx):
        how = self.drawIDsContComboBox.currentText()
        saveSettings = True
        if hasattr(self.drawIDsContComboBox, 'saveSettings'):
            saveSettings = self.drawIDsContComboBox.saveSettings
        
        if saveSettings:
            self.df_settings.at['how_draw_annotations', 'value'] = how
            self.df_settings.to_csv(self.settings_csv_path)

        mode = self.modeComboBox.currentText()
        isCcaAnnot = (
            self.annotCcaInfoCheckbox.isChecked() and 
            mode != 'Normal division: Lineage tree'
        )
        isIDAnnot = (self.annotIDsCheckbox.isChecked() or (
            self.annotCcaInfoCheckbox.isChecked() and
            mode == 'Normal division: Lineage tree'
        ))
        self.textAnnot[0].setCcaAnnot(
            isCcaAnnot
        )

        self.textAnnot[0].setLabelAnnot(
            isIDAnnot
        )

        if not self.isDataLoading:
            self.updateAllImages()

        if self.eraserButton.isChecked():
            self.setTempImg1Eraser(None, init=True)

    def drawObjLin_TreeMothBudLines(self, ax, obj, mother_obj, isNew, ID=None):
        """
        Draw moth-bud lines between an object and its mother object.

        Parameters
        ----------
        ax : cellacdc.widgets.MainPlotItem
            The Cell-ACDC GUI axes object to draw on.
        obj : Object
            The object for which to draw the moth-bud lines.
        mother_obj : Object
            The mother object to connect with.
        isNew : bool
            Indicates whether the object is new or not.
        ID : int, optional
            The ID of the object, by default None.
        """
        if not self.areMothBudLinesRequested(ax):
            return

        if not ID:
            ID = obj.label
        
        isObjVisible = self.isObjVisible(obj.bbox)
        
        if not isObjVisible:
            return

        scatterItem = self.getMothBudLineScatterItem(ax, isNew)

        y1, x1 = self.getObjCentroid(obj.centroid)
        y2, x2 = self.getObjCentroid(mother_obj.centroid)
        xx, yy = core.get_line(y1, x1, y2, x2, dashed=True)
        scatterItem.addPoints(xx, yy)

    def drawObjMothBudLines(self, obj, posData, ax=0):
        areMothBudLinesRequested = self.areMothBudLinesRequested(ax)
        if not areMothBudLinesRequested:
            return
        
        if posData.cca_df is None:
            return 

        mode = str(self.modeComboBox.currentText())
        if mode == 'Normal division: Lineage Tree':
            return

        ID = obj.label
        try:
            cca_df_ID = posData.cca_df.loc[ID]
        except KeyError:
            return        
        
        isObjVisible = self.isObjVisible(obj.bbox)
        if not isObjVisible:
            return
        
        ccs_ID = cca_df_ID['cell_cycle_stage']
        if ccs_ID == 'G1':
            return

        relationship = cca_df_ID['relationship']
        if relationship != 'bud':
            return

        emerg_frame_i = cca_df_ID['emerg_frame_i']
        isNew = emerg_frame_i == posData.frame_i
        scatterItem = self.getMothBudLineScatterItem(ax, isNew)
        relative_ID = cca_df_ID['relative_ID']

        try:
            relative_rp_idx = posData.IDs_idxs[relative_ID]
        except KeyError:
            return

        relative_ID_obj = posData.rp[relative_rp_idx]
        y1, x1 = self.getObjCentroid(obj.centroid)
        y2, x2 = self.getObjCentroid(relative_ID_obj.centroid)
        xx, yy = core.get_line(y1, x1, y2, x2, dashed=True)
        scatterItem.addPoints(xx, yy)

    def getAnnotateHowRightImage(self):
        if not self.labelsGrad.showRightImgAction.isChecked():
            return 'nothing'
        
        if self.rightBottomGroupbox.isChecked():
            how = self.annotateRightHowCombobox.currentText()
        else:
            how = self.drawIDsContComboBox.currentText()
        return how

    def getMothBudLineScatterItem(self, ax, new):
        if ax == 0:
            if new:
                return self.ax1_newMothBudLinesItem
            else:
                return self.ax1_oldMothBudLinesItem
        else:
            if new:
                return self.ax2_newMothBudLinesItem
            else:
                return self.ax2_oldMothBudLinesItem

    def getObjCentroid(self, obj_centroid):
        if self.isSegm3D:
            depthAxes = self.switchPlaneCombobox.depthAxes()
            zc, yc, xc = obj_centroid
            if depthAxes == 'z':
                return yc, xc 
            elif depthAxes == 'y':
                return zc, xc 
            else:
                return zc, yc 
        else:
            return obj_centroid

    def getObjOptsSegmLabels(self, obj):
        if not self.labelsGrad.showLabelsImgAction.isChecked():
            return

        objOpts = self.getObjTextAnnotOpts(obj, 'Draw only IDs', ax=1)
        return objOpts

    def gui_raiseBottomLayoutContextMenu(self, event):
        try:
            # Convert QPointF to QPoint
            self.bottomLayoutContextMenu.popup(event.globalPos().toPoint())
        except AttributeError:
            self.bottomLayoutContextMenu.popup(event.globalPos())

    def highLowResToggled(self, clicked=True):
        self.changeTextResolution()

    def highlightZneighLabels_cb(self, checked):
        if checked:
            pass
        else:
            pass

    def keepAllToolsActiveActionToggled(self, checked):
        for action in self.keepToolActiveActions.values():
            action.setChecked(checked)

        data_loaded = True
        if not hasattr(self, 'data'):
            data_loaded = False
            try:
                self.labelRoiTrangeCheckbox.disconnect()
            except TypeError:
                pass
        self.labelRoiTrangeCheckbox.setChecked(checked) # why this is not wrapped in a QAction?

        if data_loaded:
            self.labelRoiTrangeCheckbox.toggled.connect(
                self.labelRoiTrangeCheckboxToggled
            )

    def keepToolActiveActionToggled(self, checked, toolName=None):
        if toolName is None:
            parentToolButton = self.sender().parent()
            toolName = re.findall(r'Name: (.*)', parentToolButton.toolTip())[0]

        if checked:
            self.df_settings.at[toolName, 'value'] = 'keepActive'
        else:
            self.df_settings = self.df_settings.drop(
                index=toolName, errors='ignore'
            )
        self.df_settings.to_csv(self.settings_csv_path)

    def labelRoiIsCircularRadioButtonToggled(self, checked):
        if checked:
            self.labelRoiCircularRadiusSpinbox.setDisabled(False)
        else:
            self.labelRoiCircularRadiusSpinbox.setDisabled(True)

    def onDoubleSpaceBar(self):
        how = self.drawIDsContComboBox.currentText()
        if how.find('nothing') == -1:
            self.storeCurrentAnnotOptions_ax1()
            self.drawNothingCheckbox.setChecked(True)
            self.annotOptionClicked(
                sender=self.drawNothingCheckbox, saveSettings=False
            )
        else:
            self.restoreAnnotOptions_ax1()
        
        how = self.annotateRightHowCombobox.currentText()
        if how.find('nothing') == -1:
            self.storeCurrentAnnotOptions_ax2()
            self.drawNothingCheckboxRight.setChecked(True)
            self.annotOptionClickedRight(
                sender=self.drawNothingCheckboxRight, saveSettings=False
            )
        else:
            self.restoreAnnotOptions_ax2()

    def pxModeActionToggled(self, checked):
        self.df_settings.at['pxMode', 'value'] = int(checked)
        self.df_settings.to_csv(self.settings_csv_path)
        
        if not self.isDataLoaded:
            return
        
        if self.highLowResAction.isChecked():
            for ax in range(2):
                self.textAnnot[ax].setPxMode(checked)
        
        self.updateAllImages()

    def relabelSequentialCallback(self): 
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer' or mode == 'Cell cycle analysis':
            self.startBlinkingModeCB()
            return
        
        posData = self.data[self.pos_i]
        selectedPos = (posData.pos_foldername, )
        if len(self.data) > 1:
            selectedPos = self.askSelectPos(action='to process')
            if selectedPos is None:
                self.logger.info('Re-labelling process stopped.')
                return
        
        self.store_data()
        # acdc_df_concat = self.getConcatAcdcDf()
        # load.store_unsaved_acdc_df(
        #     posData, acdc_df_concat, 
        #     log_func=self.logger.info
        # )
        # if posData.SizeT > 1:
        self.progressWin = apps.QDialogWorkerProgress(
            title='Re-labelling sequential', parent=self,
            pbarDesc='Relabelling sequential...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)
        self.startRelabellingWorker(selectedPos)

    def restoreAnnotOptions_ax1(self, options=None):
        if options is None and not hasattr(self, 'annotOptionsToRestore'):
            return

        if options is None:
            options = self.annotOptionsToRestore
        
        if options is None:
            return
            
        for option, state in options.items():
            checkbox = getattr(self, option)
            checkbox.setChecked(state)
        
        self.setDrawAnnotComboboxText()
        self.annotOptionsToRestore = None

    def restoreAnnotOptions_ax2(self):
        if not hasattr(self, 'annotOptionsToRestoreRight'):
            return

        if self.annotOptionsToRestoreRight is None:
            return

        for option, state in self.annotOptionsToRestoreRight.items():
            checkbox = getattr(self, option)
            checkbox.setChecked(state)
        
        self.setDrawAnnotComboboxTextRight()
        self.annotOptionsToRestoreRight = None

    def restoreAnnotationsOptions(self):
        self.restoreAnnotOptions_ax1()
        self.restoreAnnotOptions_ax2()

    def restoreSavedSettings(self):
        if 'how_draw_annotations' in self.df_settings.index:
            how = self.df_settings.at['how_draw_annotations', 'value']
            self.drawIDsContComboBox.setCurrentText(how)
        else:
            self.drawIDsContComboBox.setCurrentText('Draw IDs and contours')
        
        if 'how_draw_right_annotations' in self.df_settings.index:
            how = self.df_settings.at['how_draw_right_annotations', 'value']
            self.annotateRightHowCombobox.setCurrentText(how)
        else:
            self.annotateRightHowCombobox.setCurrentText(
                'Draw IDs and overlay segm. masks'
            )
        
        if 'addNewIDsWhitelistToggle' in self.df_settings.index:
            self.addNewIDsWhitelistToggle = (
                self.df_settings.at['addNewIDsWhitelistToggle', 'value']
                ) == 'Yes'
        else:
            self.addNewIDsWhitelistToggle = True
        
        self.drawAnnotCombobox_to_options()
        self.drawIDsContComboBox_cb(0)
        self.annotateRightHowCombobox_cb(0)

    def rtTrackerActionToggled(self, checked):
        if not checked:
            return

        aliases = myutils.aliases_real_time_trackers(reverse=True)
        if self.sender().text() in aliases:
            trackingAlgo = aliases[self.sender().text()]
        else:
            trackingAlgo = self.sender().text()
        self.df_settings.at['tracking_algorithm', 'value'] = trackingAlgo
        self.df_settings.to_csv(self.settings_csv_path)

        if self.sender().text() == 'YeaZ':
            msg = widgets.myMessageBox(wrapText=False)
            info_txt = html_utils.paragraph(f"""
                Note that YeaZ tracking algorithm tends to be sliglhtly more accurate
                overall, but it is <b>less capable of detecting segmentation
                errors.</b><br><br>
                If you need to correct as many segmentation errors as possible
                we recommend using Cell-ACDC tracking algorithm.
            """)
            msg.information(self, 'Info about YeaZ', info_txt)
        
        self.isRealTimeTrackerInitialized = False
        self.initRealTimeTracker()

    def setAllTextAnnotations(self, labelsToSkip=None):
        delROIsIDs = self.setLostNewOldPrevIDs()
        posData = self.data[self.pos_i]
        self.textAnnot[0].setAnnotations(
            posData=posData, 
            labelsToSkip=labelsToSkip, 
            isVisibleCheckFunc=self.isObjVisible,
            highlightedID=self.highlightedID, 
            delROIsIDs=delROIsIDs,
            annotateLost=self.annotLostObjsToggle.isChecked(), 
            getCurrentZfunc=self.z_lab, 
            getObjCentroidFunc=self.getObjCentroid
        )
        self.textAnnot[1].setAnnotations(
            posData=posData, labelsToSkip=labelsToSkip, 
            isVisibleCheckFunc=self.isObjVisible,
            highlightedID=self.highlightedID, 
            delROIsIDs=delROIsIDs,
            annotateLost=self.annotLostObjsToggle.isChecked(), 
            getCurrentZfunc=self.z_lab, 
            getObjCentroidFunc=self.getObjCentroid
        )
        self.textAnnot[0].update()
        self.textAnnot[1].update()
        return delROIsIDs

    def setAnnotInfoMode(self, checked):
        if checked:
            for action in self.annotSettingsIDmenu.actions():
                if action.text().find('tree') != -1:
                    self.textAnnot[0].setLabelTreeAnnotationsEnabled(True)
                    action.setChecked(True)
                    break
            for action in self.annotSettingsGenNumMenu.actions():
                if action.text().find('tree') != -1:
                    self.textAnnot[0].setGenNumTreeAnnotationsEnabled(True)
                    action.setChecked(True)
                    break
        else:
            for action in self.annotSettingsIDmenu.actions():
                if action.text().find('tree') == -1:
                    action.setChecked(False)
                    self.textAnnot[0].setLabelTreeAnnotationsEnabled(False)
                    break
            for action in self.annotSettingsGenNumMenu.actions():
                if action.text().find('tree') == -1:
                    action.setChecked(False)
                    self.textAnnot[0].setGenNumTreeAnnotationsEnabled(False)
                    break
        self.setAllTextAnnotations()

    def setAnnotOptionsCcaMode(self):
        self.prevAnnotOptions = self.storeCurrentAnnotOptions_ax1(
            return_value=True
        )
        self.annotCcaInfoCheckbox.setChecked(True)
        self.annotIDsCheckbox.setChecked(False)
        self.drawMothBudLinesCheckbox.setChecked(False)
        self.setDrawAnnotComboboxText()

    def setAnnotOptionsLin_treeMode(self):
        # self.prevAnnotOptions = self.storeCurrentAnnotOptions_ax1(
        #     return_value=True
        # )
        self.annotCcaInfoCheckbox.setChecked(True)
        self.annotIDsCheckbox.setChecked(False)
        self.drawMothBudLinesCheckbox.setChecked(False)
        self.setDrawAnnotComboboxText()
        self.showTreeInfoCheckbox.setChecked(True)

    def setDisabledAnnotCheckBoxesLeft(self, disabled):
        self.annotIDsCheckbox.setDisabled(disabled)
        self.annotCcaInfoCheckbox.setDisabled(disabled)
        self.annotContourCheckbox.setDisabled(disabled)
        self.annotSegmMasksCheckbox.setDisabled(disabled)
        self.drawMothBudLinesCheckbox.setDisabled(disabled)
        self.annotNumZslicesCheckbox.setDisabled(disabled)
        self.drawNothingCheckbox.setDisabled(disabled)

    def setDisabledAnnotCheckBoxesRight(self, disabled):
        self.annotIDsCheckboxRight.setDisabled(disabled)
        self.annotCcaInfoCheckboxRight.setDisabled(disabled)
        self.annotContourCheckboxRight.setDisabled(disabled)
        self.annotSegmMasksCheckboxRight.setDisabled(disabled)
        self.drawMothBudLinesCheckboxRight.setDisabled(disabled)
        self.annotNumZslicesCheckboxRight.setDisabled(disabled)
        self.drawNothingCheckboxRight.setDisabled(disabled)

    def setDisabledAnnotOptions(self, disabled):
        # Left
        self.annotIDsCheckbox.setDisabled(disabled)
        self.annotCcaInfoCheckbox.setDisabled(disabled)
        self.annotContourCheckbox.setDisabled(disabled)
        # self.annotSegmMasksCheckbox.setDisabled(disabled)
        self.drawMothBudLinesCheckbox.setDisabled(disabled)
        # self.drawNothingCheckbox.setDisabled(disabled)

        # Right 
        self.annotIDsCheckboxRight.setDisabled(disabled)
        self.annotCcaInfoCheckboxRight.setDisabled(disabled)
        self.annotContourCheckboxRight.setDisabled(disabled)
        # self.annotSegmMasksCheckboxRight.setDisabled(disabled)
        self.drawMothBudLinesCheckboxRight.setDisabled(disabled)

    def setDrawAnnotComboboxText(self, saveSettings=True):
        if self.annotIDsCheckbox.isChecked():
            if self.annotContourCheckbox.isChecked():
                t = 'Draw IDs and contours'
            elif self.annotSegmMasksCheckbox.isChecked():
                t = 'Draw IDs and overlay segm. masks'
            else:
                t = 'Draw only IDs'
        
        elif self.annotCcaInfoCheckbox.isChecked():
            if self.annotContourCheckbox.isChecked():
                t = 'Draw cell cycle info and contours'
            elif self.annotSegmMasksCheckbox.isChecked():
                t = 'Draw cell cycle info and overlay segm. masks'
            else:
                t = 'Draw only cell cycle info'
        
        elif self.annotSegmMasksCheckbox.isChecked():
            t = 'Draw only overlay segm. masks'

        elif self.annotContourCheckbox.isChecked():
            t = 'Draw only contours'
        
        elif self.drawMothBudLinesCheckbox.isChecked():
            t = 'Draw only mother-bud lines'
        
        elif self.drawNothingCheckbox.isChecked():
            t = 'Draw nothing'
        else:
            t = 'Draw nothing'

        if t == self.drawIDsContComboBox.currentText():
            self.drawIDsContComboBox_cb(0)
        
        self.drawIDsContComboBox.saveSettings = saveSettings
        self.drawIDsContComboBox.setCurrentText(t)

    def setDrawAnnotComboboxTextRight(self, saveSettings=True):
        if self.annotIDsCheckboxRight.isChecked():
            if self.annotContourCheckboxRight.isChecked():
                t = 'Draw IDs and contours'
            elif self.annotSegmMasksCheckboxRight.isChecked():
                t = 'Draw IDs and overlay segm. masks'
            else:
                t = 'Draw only IDs'
        
        elif self.annotCcaInfoCheckboxRight.isChecked():
            if self.annotContourCheckboxRight.isChecked():
                t = 'Draw cell cycle info and contours'
            elif self.annotSegmMasksCheckboxRight.isChecked():
                t = 'Draw cell cycle info and overlay segm. masks'
            else:
                t = 'Draw only cell cycle info'
        
        elif self.annotSegmMasksCheckboxRight.isChecked():
            t = 'Draw only overlay segm. masks'

        elif self.annotContourCheckboxRight.isChecked():
            t = 'Draw only contours'
        
        elif self.drawMothBudLinesCheckboxRight.isChecked():
            t = 'Draw only mother-bud lines'
        
        elif self.drawNothingCheckboxRight.isChecked():
            t = 'Draw nothing'
        else:
            t = 'Draw nothing'

        if t == self.annotateRightHowCombobox.currentText():
            self.annotateRightHowCombobox_cb(0)
        
        self.annotateRightHowCombobox.saveSettings = saveSettings
        self.annotateRightHowCombobox.setCurrentText(t)

    def setDrawNothingAnnotations(self):
        self.storeCurrentAnnotOptions_ax1()
        self.storeCurrentAnnotOptions_ax2()
        self.drawNothingCheckbox.setChecked(True)
        self.annotOptionClicked(
            sender=self.drawNothingCheckbox, saveSettings=False)
        self.drawNothingCheckboxRight.setChecked(True)
        self.annotOptionClickedRight(
            sender=self.drawNothingCheckboxRight, saveSettings=False
        )

    def setEnabledAnnotCheckBoxesLeftZdepthAxes(self):
        if not self.isSegm3D:
            return
        
        self.annotIDsCheckbox.setDisabled(False)
        self.annotContourCheckbox.setDisabled(False)
        self.annotIDsCheckbox.setChecked(True)
        self.annotContourCheckbox.setChecked(True)
        
        self.annotOptionClicked(
            sender=self.annotIDsCheckbox, saveSettings=False)

    def setVisible3DsegmWidgets(self):
        self.annotNumZslicesCheckbox.setVisible(self.isSegm3D)
        self.annotNumZslicesCheckboxRight.setVisible(self.isSegm3D)
        if not self.isSegm3D:
            self.annotNumZslicesCheckbox.setChecked(False)
            self.annotNumZslicesCheckboxRight.setChecked(False)

    def showHighlightZneighCheckbox(self):
        if self.isSegm3D:
            layout = self.bottomLeftLayout
            # layout.addWidget(self.annotOptionsWidget, 0, 1, 1, 2)
            # # layout.removeWidget(self.drawIDsContComboBox)
            # # layout.addWidget(self.drawIDsContComboBox, 0, 1, 1, 1,
            # #     alignment=Qt.AlignCenter
            # # )
            # layout.addWidget(self.highlightZneighObjCheckbox, 0, 2, 1, 2,
            #     alignment=Qt.AlignRight
            # )
            self.highlightZneighObjCheckbox.show()
            self.highlightZneighObjCheckbox.setChecked(True)
            self.highlightZneighObjCheckbox.toggled.connect(
                self.highlightZneighLabels_cb
            )

    def storeCurrentAnnotOptions_ax1(self, return_value=False):
        if self.annotOptionsToRestore is not None:
            return
        
        checkboxes = [
            'annotIDsCheckbox',
            'annotCcaInfoCheckbox',
            'annotContourCheckbox',
            'annotSegmMasksCheckbox',
            'drawMothBudLinesCheckbox',
            'annotNumZslicesCheckbox',
            'drawNothingCheckbox',
        ]
        annotOptions = {}
        for checkboxName in checkboxes:
            checkbox = getattr(self, checkboxName)
            annotOptions[checkboxName] = checkbox.isChecked()
        if return_value:
            return annotOptions
        self.annotOptionsToRestore = annotOptions

    def storeCurrentAnnotOptions_ax2(self):
        if self.annotOptionsToRestoreRight is not None:
            return
        
        checkboxes = [
            'annotIDsCheckboxRight',
            'annotCcaInfoCheckboxRight',
            'annotContourCheckboxRight',
            'annotSegmMasksCheckboxRight',
            'drawMothBudLinesCheckboxRight',
            'annotNumZslicesCheckboxRight',
            'drawNothingCheckboxRight',
        ]
        self.annotOptionsToRestoreRight = {}
        for checkboxName in checkboxes:
            checkbox = getattr(self, checkboxName)
            self.annotOptionsToRestoreRight[checkboxName] = checkbox.isChecked()

    def uncheckAnnotOptions(self, left=True, right=True):
        # Left
        if left:
            self.annotIDsCheckbox.setChecked(False)
            self.annotCcaInfoCheckbox.setChecked(False)
            self.annotContourCheckbox.setChecked(False)
            self.annotSegmMasksCheckbox.setChecked(False)
            self.drawMothBudLinesCheckbox.setChecked(False)
            self.drawNothingCheckbox.setChecked(False)

        # Right 
        if right:
            self.annotIDsCheckboxRight.setChecked(False)
            self.annotCcaInfoCheckboxRight.setChecked(False)
            self.annotContourCheckboxRight.setChecked(False)
            self.annotSegmMasksCheckboxRight.setChecked(False)
            self.drawMothBudLinesCheckboxRight.setChecked(False)
            self.drawNothingCheckboxRight.setChecked(False)

    def updateAnnotatedIDs(self, oldIDs, newIDs, logger=print):
        logger('Updating annotated IDs...')
        posData = self.data[self.pos_i]

        mapper = dict(zip(oldIDs, newIDs))
        posData.ripIDs = set([mapper[ripID] for ripID in posData.ripIDs])
        posData.binnedIDs = set([mapper[binID] for binID in posData.binnedIDs])
        self.keptObjectsIDs = widgets.KeptObjectIDsList(
            self.keptIDsLineEdit, self.keepIDsConfirmAction
        )

        customAnnotButtons = list(self.customAnnotDict.keys())
        for button in customAnnotButtons:
            customAnnotValues = self.customAnnotDict[button]
            annotatedIDs = customAnnotValues['annotatedIDs'][self.pos_i]
            mappedAnnotIDs = {}
            for frame_i, annotIDs_i in annotatedIDs.items():
                mappedIDs = [mapper[ID] for ID in annotIDs_i]
                mappedAnnotIDs[frame_i] = mappedIDs
            customAnnotValues['annotatedIDs'][self.pos_i] = mappedAnnotIDs

    def update_rp_metadata(self, draw=True):
        posData = self.data[self.pos_i]
        # Add to rp dynamic metadata (e.g. cells annotated as dead)
        for i, obj in enumerate(posData.rp):
            ID = obj.label
            obj.excluded = ID in posData.binnedIDs
            obj.dead = ID in posData.ripIDs

    def zoomRectActionToggled(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            self.connectLeftClickButtons()
            self.ax1.addItem(self.zoomRectItem)
        else:
            self.zoomRectItem.setPos((0,0))
            self.zoomRectItem.setSize((0,0))
            self.ax1.removeItem(self.zoomRectItem)

    def zoomRectCancelled(self):
        self.isMouseDragImg1 = False
        self.zoomRectItem.setPos((0,0))
        self.zoomRectItem.setSize((0,0))

    def zoomRectDone(self):
        xRange, yRange = self.ax1.viewRange()
        self.zoomRectItem.storeLastRange(xRange, yRange)
        
        ymin, xmin, ymax, xmax = self.zoomRectItem.bbox()
        
        self.zoomRectItem.setPos((0,0))
        self.zoomRectItem.setSize((0,0))
        
        self.ax1.setRange(
            xRange=(xmin, xmax), 
            yRange=(ymin, ymax), 
            padding=0
        )

    def showAllContoursToggled(self):
        if not self.isDataLoaded:
            return
        
        self.computeAllContours()
        self.updateAllImages()
