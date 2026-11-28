"""Object edits of the main GUI (edit ID, merge, delete, separate,
propagate to other frames) and the Keep IDs tool. Mixins of guiWin
(gui/main_window/window.py) only: the measurements viewer never composes
them.
"""
from tqdm import tqdm
from typing import Iterable
import math
import numpy as np
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
import skimage.segmentation
from qtpy.QtCore import QTimer, QThread
from qtpy.QtWidgets import QAction, QCheckBox
from cellacdc import exception_handler, disableWindow
from cellacdc import load, apps, workers, html_utils
from cellacdc import myutils
from cellacdc import widgets
from cellacdc import settings_csv_path


class ObjectEditingGuiElements:
    """Object edits: edit ID, merge, delete, separate, delete border, new
    and outside-mask objects, relabel, and propagation of changes to
    past and future frames.
    """

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
        
        # elif posData:
        #     self.storeUndoRedoStates(False)
        #     posData.lab, oldIDs, newIDs = core.relabel_sequential(posData.lab)
        #     # Update annotations based on relabelling
        #     self.update_cca_df_relabelling(posData, oldIDs, newIDs)
        #     self.updateAnnotatedIDs(oldIDs, newIDs, logger=self.logger.info)
        #     self.store_data()
        #     self.update_rp()
        #     li = list(zip(oldIDs, newIDs))
        #     s = '\n'.join([str(pair).replace(',', ' -->') for pair in li])
        #     s = f'IDs relabelled as follows:\n{s}'
        #     self.logger.info(s)
        #     self.updateAllImages()

    def delObjsOutSegmMaskWorkerFinished(self, result):
        posData = self.data[self.pos_i]
        worker, cleared_segm_data, delIDs = result
        if posData.SizeT == 1:
            cleared_segm_data = cleared_segm_data[np.newaxis]
        
        self.update_cca_df_deletedIDs(posData, delIDs)
        
        current_frame_i = posData.frame_i
        for frame_i, cleared_lab in enumerate(cleared_segm_data):
            # Store change
            posData.allData_li[frame_i]['labels'] = cleared_lab
            # Get the rest of the stored metadata based on the new lab
            posData.frame_i = frame_i
            self.get_data()
            self.update_rp(deletionIDs=delIDs)
            self.store_data(autosave=False)
        
        # Back to current frame
        posData.frame_i = current_frame_i
        self.get_data()
        
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.logger.info('Deleting objects outside of ROIs finished.')
        self.titleLabel.setText(
            'Deleting objects outside of ROIs finished.', color='w'
        )
        self.updateAllImages()

    def startRelabellingWorker(self, posFoldernames):
        self.thread = QThread()
        self.worker = workers.relabelSequentialWorker(self, posFoldernames)
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.progress.connect(self.workerProgress)
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.workerFinished)
        self.worker.finished.connect(self.relabelWorkerFinished)

        self.worker.debug.connect(self.workerDebug)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def relabelWorkerFinished(self):
        self.updateAllImages()

    # @exec_time
    def applyEditID(
            self, clickedID, currentIDs, oldIDnewIDMapper, clicked_x, clicked_y, 
            doPropagateUnvisited=False,
            merging_IDs=False,
            howToPropagateZslices=None
        ):  
        posData = self.data[self.pos_i]
        is_2d_edit_on_3d = (
            howToPropagateZslices is not None 
            and howToPropagateZslices[0] == 'Current z-slice'
        )
        is_entire_obj = (
            howToPropagateZslices is None
            or howToPropagateZslices[0] == 'Entire object'
        )
        is_to_last_z = (
            howToPropagateZslices is not None 
            and howToPropagateZslices[0] == 'From current to last z-slice'
        )
        is_to_first_z = (
            howToPropagateZslices is not None 
            and howToPropagateZslices[0] == 'From current to first z-slice'
        )
        is_custom_z_range = (
            howToPropagateZslices is not None 
            and howToPropagateZslices[0] == 'Custom range'
        )
        z_start, z_end = None, None
        if is_to_last_z:
            z_start = self.zSliceScrollBar.sliderPosition()
        elif is_to_first_z:
            z_end = self.zSliceScrollBar.sliderPosition()
        elif is_custom_z_range:
            z_start, z_end = howToPropagateZslices[1]

        rp = self.rpCurr2D() if is_2d_edit_on_3d else posData.rp
        use_3D_obj_centroid = is_entire_obj and self.isSegm3D
        # Ask to propagate change to all future visited frames
        key = 'Edit ID'
        askAction = self.askHowFutureFramesActions[key]
        doNotShow = not askAction.isChecked()
        (UndoFutFrames, applyFutFrames, endFrame_i,
        doNotShowAgain) = self.propagateChange(
            clickedID, key, doNotShow,
            posData.UndoFutFrames_EditID, posData.applyFutFrames_EditID,
            applyTrackingB=True
        )

        if UndoFutFrames is None:
            self.annotateAllObjectTracks()
            self.annotateAssignedObjsAcdcTrackerSecondStep()
            return

        if is_2d_edit_on_3d:
            lab = self.get_2Dlab(posData.lab)
        else:
            lab = posData.lab
        
        # Store undo state before modifying stuff
        # no risk of merging IDs if we are working with rp and dont updaet in the middle...
        self.storeUndoRedoStates(UndoFutFrames)
        # could this be chained??? If yes we have to "simplify" to least swops to since we keep RP stale
        # oldIDnewIDMapper
        assignments = {}        
        for old_ID, new_ID in oldIDnewIDMapper: 
            if new_ID in currentIDs and not self.editIDmergeIDs:
                objo = rp.get_obj_from_ID(old_ID)
                objn = rp.get_obj_from_ID(new_ID)

                slc_n = objn.slice
                mask_n = objn.image

                slc_o = objo.slice
                mask_o = objo.image

                if z_start is not None:
                    local_z_o = z_start - objo.bbox[0]
                    local_z_n = z_start - objo.bbox[0]
                    mask_o[:local_z_o] = False
                    mask_n[:local_z_n] = False

                if z_end is not None:
                    local_z_o = z_end - objo.bbox[0]
                    local_z_n = z_end - objo.bbox[0]
                    mask_o[local_z_o+1:] = False
                    mask_n[local_z_n+1:] = False

                # Relabel old_ID to new ID, save since rp is "stale"
                lab[slc_o][mask_o] = new_ID

                # Relabel new_ID to old_ID
                
                lab[slc_n][mask_n] = old_ID

                # ¯\_(ツ)_/¯
                if use_3D_obj_centroid:
                    objn_centroid = rp.get_centroid(old_ID, exact=True) # 
                    yn, xn = self.getObjCentroid(objn_centroid)
                    if not math.isnan(yn):
                        yn, xn = int(yn), int(xn)
                        posData.editID_info.append((yn, xn, new_ID))
                        yo, xo = int(clicked_y), int(clicked_x)
                        posData.editID_info.append((yo, xo, old_ID))
                assignments[new_ID] = old_ID
                assignments[old_ID] = new_ID
            else:
                # Use regionprops for old_ID
                obj = rp.get_obj_from_ID(old_ID)
                slc = obj.slice
                mask = obj.image
                if z_start is not None:
                    local_z = z_start - obj.bbox[0]
                    mask[:local_z] = False

                if z_end is not None:
                    local_z = z_end - obj.bbox[0]
                    mask[local_z+1:] = False

                lab[slc][mask] = new_ID
                if use_3D_obj_centroid:
                    centroid = rp.get_centroid(old_ID, exact=True)
                    y, x = self.getObjCentroid(centroid)
                    if not math.isnan(y) and not math.isnan(x):
                        y, x = int(y), int(x)
                        posData.editID_info.append((y, x, new_ID))
                assignments[old_ID] = new_ID
            
            self.updateAssignedObjsAcdcTrackerSecondStep(new_ID)
        
        if is_2d_edit_on_3d:
            self.set_2Dlab(lab)
        
        # Update rps
        # When shift is active we edited the 2D slice of a 3D lab; the cached 
        # slice/image data in the old RP objects is stale so we must do a full 
        # recompute rather than the fast assignments-only path. 
        # The fast assignments can be used only if `is_entire_obj` is True, 
        # which means 2D segmentation or "Entire object" option is selected 
        
        # check for merger
        assignments = assignments if not merging_IDs else None
        assignments = assignments if is_entire_obj else None
        self.update_rp(assignments=assignments)

        # Since we manually changed an ID we don't want to repeat tracking
        self.setAllTextAnnotations()        
        self.highlightLostNew()
        # self.checkIDsMultiContour()

        # Update colors for the edited IDs
        self.updateLookuptable()

        if self.isSnapshot:
            self.fixCcaDfAfterEdit('Edit ID')
            self.updateAllImages()
        else:
            self.warnEditingWithCca_df('Edit ID', update_images=False)
        
        if not self.editIDbutton.findChild(QAction).isChecked():
            self.editIDbutton.setChecked(False)

        posData.disableAutoActivateViewerWindow = True

        # Perform desired action on future frames
        posData.doNotShowAgain_EditID = doNotShowAgain
        posData.UndoFutFrames_EditID = UndoFutFrames
        posData.applyFutFrames_EditID = applyFutFrames
        includeUnvisited = (
            posData.includeUnvisitedInfo['Edit ID']
            or doPropagateUnvisited
        )
        
        if not applyFutFrames and not doPropagateUnvisited:
            self.annotateAllObjectTracks()
            self.annotateAssignedObjsAcdcTrackerSecondStep()
            return

        self.changeIDfutureFrames(
            endFrame_i, oldIDnewIDMapper, includeUnvisited,
            merging_IDs=merging_IDs,
            is_2d_edit_on_3d=is_2d_edit_on_3d,
            is_entire_obj=is_entire_obj,
            is_to_last_z=is_to_last_z,
            is_to_first_z=is_to_first_z,
            is_custom_z_range=is_custom_z_range,
            z_start=z_start,
            z_end=z_end,
        )
        
        self.annotateAllObjectTracks()
        self.annotateAssignedObjsAcdcTrackerSecondStep()

    def delBorderObj(self, checked):
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        posData = self.data[self.pos_i]
        edge_ids = myutils.clear_border(posData.lab, return_edge_ids=True) # modifies inplace
        self.update_rp(deletionIDs=edge_ids)
        self.update_cca_df_deletedIDs(
            posData, edge_ids, dropInPast=False, dropInFuture=False
        )
        self.store_data()
        self.updateAllImages()

    def delNewObj(self, checked):
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        
        if frame_i == 0:
            return
        
        prev_IDs = posData.allData_li[frame_i-1]['regionprops'].IDs
        curr_IDs = posData.IDs
        
        protected_new_IDs = self.protected_new_IDs.get(frame_i, [])
        new_IDs = list(set(curr_IDs) - set(prev_IDs) - set(protected_new_IDs))

        lab = posData.lab
        lab[np.isin(lab, new_IDs)] = 0
        posData.lab = lab
        
        self.update_rp(deletionIDs=new_IDs)
                
        if posData.cca_df is not None:
            posData.cca_df = posData.cca_df.drop(index=new_IDs)
        self.store_data()
        self.updateAllImages()

    def mergeIDs_cb(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.mergeIDsButton)
            self.connectLeftClickButtons()
            self.mergeIDsToolbar.setOnlyCurrentZsliceEnabled(self.isSegm3D)
        else:
            self.freeRoiItem.clear()
        
        self.mergeIDsToolbar.setVisible(checked)
        QTimer.singleShot(20, self.gui_refreshToolCursor)

    def acceptMergeMultipleIDs(self, IDs):
        if not self.mergeIDsButton.isChecked():
            return

        posData = self.data[self.pos_i]
        valid_ids = [ID for ID in IDs if ID in posData.IDs]
        if len(valid_ids) < 2:
            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph(
                'Select at least two existing IDs before merging.'
            )
            msg.warning(self, 'Not enough IDs selected', txt)
            return

        targetID = valid_ids[0]
        self.mergeSelectedIDs(valid_ids, target_id=targetID)

    def mergeSelectedIDs(self, IDs_to_merge, target_id=None):
        posData = self.data[self.pos_i]
        zRange = None
        if self.isSegm3D:
            zRange = (0, posData.SizeZ)
            
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(
            False, 
            storeImage=False, 
            storeOnlyZoom=True, 
            zRange=zRange
        )
    
        unique_ids = []
        seen = set()
        for ID in IDs_to_merge:
            if ID <= 0 or ID in seen:
                continue
            if ID not in posData.IDs:
                continue
            seen.add(ID)
            unique_ids.append(int(ID))

        if len(unique_ids) < 2:
            return False

        if target_id is None:
            target_id = unique_ids[0]
        elif target_id not in unique_ids:
            unique_ids.insert(0, target_id)

        self.firstID = target_id

        zProjHow = self.zProjComboBox.currentText()
        isZslice = zProjHow == 'single z-slice'
        only_current_zslice = (
            self.isSegm3D 
            and self.mergeIDsToolbar.isOnlyCurrentZslice()
            and isZslice
        )

        if only_current_zslice:
            lab2D = self.currentLab2D
            source_ids = [ID for ID in unique_ids if ID != target_id]
            merge_mask = np.isin(self.currentLab2D, source_ids)
            if not np.any(merge_mask):
                return False
            self.currentLab2D[merge_mask] = target_id
            self.set_2Dlab(self.currentLab2D)
            self.update_rp()
        else:
            for ID in unique_ids:
                if ID == target_id:
                    continue
                obj = posData.rp.get_obj_from_ID(ID, warn=False)
                if obj is None:
                    continue
                posData.lab[obj.slice][obj.image] = target_id

            bbox = self.update_rp_get_bbox(
                specific_IDs=unique_ids + [target_id], use_bbox=True
            )
            self.update_rp(
                specific_IDs=unique_ids + [target_id],
                preloaded_bbox=bbox
            )

        ask_back_prop = True
        if only_current_zslice:
            ask_back_prop = False
        elif posData.frame_i == 0:
            ask_back_prop = False
            prev_IDs = []
        else:
            prev_IDs = posData.allData_li[posData.frame_i-1]['regionprops'].IDs

        if ask_back_prop and sum(ID in prev_IDs for ID in unique_ids) <= 1:
            ask_back_prop = False

        # Merging in annotated past frames would make their annotations
        # wrong: propagate only to the frames after the last annotated one
        if ask_back_prop and self.isFrameCcaAnnotated():
            ask_back_prop = False

        last_annot_i = -1
        if ask_back_prop:
            last_annot_i = max(self.getLastAnnotatedFrames())
            if posData.frame_i-1 <= last_annot_i:
                ask_back_prop = False

        if ask_back_prop:
            proceed = self.askPropagateChangePast(f'Merge IDs {unique_ids}')
            if proceed:
                self.propagateMergeObjsPast(
                    unique_ids, stop_frame_i=last_annot_i
                )
                self.whitelistPropagateIDs(
                    only_future_frames=False, update_lab=True
                )

        self.tracking(
            enforce=True, assign_unique_new_IDs=False,
            separateByLabel=False
        )

        if self.isSnapshot:
            self.fixCcaDfAfterEdit('Merge IDs')
            self.updateAllImages()
        else:
            self.warnEditingWithCca_df('Merge IDs')

        self.store_data()
        return True

    def delObjsOutSegmMaskActionTriggered(self):
        posData = self.data[self.pos_i]
        segm_files = load.get_segm_files(posData.images_path)
        existingSegmEndnames = load.get_endnames(
            posData.basename, segm_files
        )
        selectSegmWin = widgets.QDialogListbox(
            'Select segmentation file',
            'Select segmentation file to use as ROI:\n',
            existingSegmEndnames, multiSelection=False, parent=self,
            allowEmptySelection=False
        )
        selectSegmWin.exec_()
        if selectSegmWin.cancel:
            self.logger.info('Delete objects process cancelled.')
            return
        
        selectedSegmEndname = selectSegmWin.selectedItemsText[0]
        
        self.startDelObjsOutSegmMaskWorker(selectedSegmEndname)

    def startDelObjsOutSegmMaskWorker(self, selectedSegmEndname):
        self.store_data(autosave=False)
        posData = self.data[self.pos_i]
        segm_data = np.squeeze(self.getStoredSegmData())
        
        self.progressWin = apps.QDialogWorkerProgress(
            title='Deleting objects outside of ROIs', parent=self,
            pbarDesc='Deleting objects outside of ROIs...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)
        
        self.thread = QThread()
        self.worker = workers.DelObjectsOutsideSegmROIWorker(
            selectedSegmEndname, segm_data, posData.images_path
        )
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.progress.connect(self.workerProgress)
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.delObjsOutSegmMaskWorkerFinished)

        self.worker.debug.connect(self.workerDebug)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def editID_cb(self, checked):
        self.editIDToolbar.setVisible(checked and self.isSegm3D)

    def showEditIDwidgets(self, visible):
        self.editIDLabelAction.setVisible(visible)
        self.editIDspinboxAction.setVisible(visible)
        self.autoIDcheckboxAction.setVisible(visible)
        showToolbar = (
            visible
            or self.brushSizeAction.isVisible()
            or self.brushAutoFillAction.isVisible()
            or self.brushAutoHideAction.isVisible()
        )
        self.brushEraserToolBar.setVisible(showToolbar)

    def askPropagateChangePast(self, change_txt):
        txt = html_utils.paragraph(f"""
            Do you want to propagate the change "{change_txt}" to the past frames?
        """)
        msg = widgets.myMessageBox(wrapText=False)
        yesButton, _ = msg.question(
            self, 'Propagate change to past frames', txt, 
            buttonsTexts=('Yes', 'No')
        )
        return msg.clickedButton == yesButton

    def propagateMergeObjsPast(self, IDs_to_merge, stop_frame_i=-1):
        """Merge `IDs_to_merge` also in the past frames, from the previous
        frame backwards until a frame lacks one of them.

        Parameters
        ----------
        IDs_to_merge : list of int
        stop_frame_i : int, optional
            Frames up to this index are not modified (e.g. the last frame
            with annotations, see `getLastAnnotatedFrames`). Default is -1
            (down to frame 0).
        """
        self.store_data(autosave=False)
        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i
        for past_frame_i in range(posData.frame_i-1, stop_frame_i, -1):
            posData.frame_i = past_frame_i
            self.get_data()
            
            IDs = posData.allData_li[past_frame_i]['regionprops'].IDs
            stop_loop = False
            for ID in IDs_to_merge:
                if ID not in IDs:
                    stop_loop = True
                    break

                if ID == 0:
                    continue
                obj = posData.rp.get_obj_from_ID(ID)
                posData.lab[obj.slice][obj.image] = self.firstID
            
            specific_IDs = [ID for ID in IDs_to_merge if ID in IDs]
            preloaded_bbox = self.update_rp_get_bbox(
                specific_IDs=specific_IDs, use_bbox=True
                ) # use old RP to get the correct bbox
            specific_IDs = [*IDs_to_merge, self.firstID]
            self.update_rp(preloaded_bbox=preloaded_bbox, specific_IDs=specific_IDs)
            self.store_data(autosave=False)
            
            if stop_loop:
                break
        
        posData.frame_i = current_frame_i
        self.get_data()

    def propagateChange(
            self, 
            affectedIDs: list[int] | set[int] | int, 
            modTxt, 
            doNotShow, 
            UndoFutFrames,
            applyFutFrames, 
            applyTrackingB=False, 
            force=False
        ):
        """
        This function determines whether any already visited future frames contain
        one of the IDs in ``affectedIDs``. If so, it triggers a pop-up asking the
        user whether to propagate the change to future frames.
        """
        if isinstance(affectedIDs, (int, np.uint32, np.int32)):
            affectedIDs = {affectedIDs}
        
        if isinstance(affectedIDs, list) or isinstance(affectedIDs, tuple):
            affectedIDs = set(affectedIDs)
        
        posData = self.data[self.pos_i]
        # Do not check the future for the last frame
        if posData.frame_i+1 == posData.SizeT:
            # No future frames to propagate the change to
            return False, False, None, doNotShow

        includeUnvisited = posData.includeUnvisitedInfo.get(modTxt, False)
        areFutureIDs_affected = False
        # Get number of future frames already visited and check if future
        # frames has an ID affected by the change
        last_tracked_i_found = False
        segmSizeT = len(posData.segm_data)
        for i in range(posData.frame_i+1, segmSizeT):
            if posData.allData_li[i]['labels'] is None:
                if not last_tracked_i_found:
                    # We set last tracked frame at -1 first None found
                    last_tracked_i = i - 1
                    last_tracked_i_found = True
                if not includeUnvisited:
                    # Stop at last visited frame since includeUnvisited = False
                    break
                else:
                    IDs = posData.allData_li[i]['regionprops'].IDs_set
            else:
                IDs = posData.allData_li[i]['regionprops'].IDs_set            
            
            if areFutureIDs_affected:
                continue
            
            if affectedIDs.intersection(IDs):
                areFutureIDs_affected = True
        
        if not last_tracked_i_found:
            # All frames have been visited in segm&track mode
            last_tracked_i = posData.SizeT - 1
        
        if last_tracked_i == posData.frame_i and not includeUnvisited:
            # No future frames to propagate the change to
            return False, False, None, doNotShow

        if not areFutureIDs_affected and not force:
            # There are future frames but they are not affected by the change
            return UndoFutFrames, False, None, doNotShow

        # Ask what to do unless the user has previously checked doNotShowAgain
        if doNotShow:
            endFrame_i = last_tracked_i
            if applyFutFrames and not UndoFutFrames and modTxt == 'Edit ID':
                self.whitelistSyncIDsOG(frame_is=range(posData.frame_i, endFrame_i+1))
            return UndoFutFrames, applyFutFrames, endFrame_i, doNotShow
        else:
            addApplyAllButton = (
                modTxt == 'Delete ID' or modTxt == 'Edit ID' 
                or modTxt == 'Assign new ID'
            )
            ffa = apps.FutureFramesAction_QDialog(
                posData.frame_i+1, last_tracked_i, modTxt, 
                applyTrackingB=applyTrackingB, parent=self, 
                addApplyAllButton=addApplyAllButton
            )
            ffa.exec_()
            decision = ffa.decision

            if decision is None:
                return None, None, None, doNotShow

            endFrame_i = ffa.endFrame_i
            doNotShowAgain = ffa.doNotShowCheckbox.isChecked()
            askAction = self.askHowFutureFramesActions[modTxt]
            askAction.setChecked( not doNotShowAgain)
            askAction.setDisabled(False)

            self.onlyTracking = False
            if decision == 'apply_and_reinit':
                UndoFutFrames = True
                applyFutFrames = False
            elif decision == 'apply_and_NOTreinit':
                UndoFutFrames = False
                applyFutFrames = False
            elif decision == 'apply_to_all_visited':
                UndoFutFrames = False
                applyFutFrames = True
            elif decision == 'only_tracking':
                UndoFutFrames = False
                applyFutFrames = True
                self.onlyTracking = True
            elif decision == 'apply_to_all':
                UndoFutFrames = False
                applyFutFrames = True
                posData.includeUnvisitedInfo[modTxt] = True

            if applyFutFrames and not UndoFutFrames and modTxt == 'Edit ID':
                self.whitelistSyncIDsOG(
                    frame_is=range(posData.frame_i, endFrame_i+1)
                )
                
        return UndoFutFrames, applyFutFrames, endFrame_i, doNotShowAgain

    def _changeIDhelper(
            self, lab, oldID, newID, rp, assignments,
            z_start=None,
            z_end=None,
        ):
        did_find_newID = False
        if newID in rp.IDs:
            # Relabel old_ID to tempID, safe as RP is safe so no merging
            objo = rp.get_obj_from_ID(oldID, warn=False)
            objn = rp.get_obj_from_ID(newID) 

            if objo is not None:
                slc_o = objo.slice
                mask_o = objo.image

                if z_start is not None:
                    local_z_o = z_start - objo.bbox[0]
                    mask_o[:local_z_o] = False

                if z_end is not None:
                    local_z_o = z_end - objo.bbox[0]
                    mask_o[local_z_o+1:] = False

                lab[slc_o][mask_o] = newID
                assignments[oldID] = newID
            
            objn_slice = objn.slice
            mask_n = objn.image
            if z_start is not None:
                local_z_n = z_start - objo.bbox[0]
                mask_n[:local_z_n] = False

            if z_end is not None:
                local_z_n = z_end - objo.bbox[0]
                mask_n[local_z_n+1:] = False

            # Relabel new_ID to old_ID
            lab[objn_slice][mask_n] = oldID
            assignments[newID] = oldID
            did_find_newID = True
        else:
            obj = rp.get_obj_from_ID(oldID, warn=False)
            if obj is not None:
                slc = obj.slice
                mask = obj.image
                if z_start is not None:
                    local_z = z_start - obj.bbox[0]
                    mask[:local_z] = False

                if z_end is not None:
                    local_z = z_end - obj.bbox[0]
                    mask[local_z+1:] = False

                lab[slc][mask] = newID
                assignments[oldID] = newID
        return did_find_newID

    def changeIDfutureFrames(
            self, endFrame_i, oldIDnewIDMapper, includeUnvisited,
            merging_IDs=False,
            is_2d_edit_on_3d=False,
            is_entire_obj=True,
            is_to_last_z=False,
            is_to_first_z=False,
            is_custom_z_range=False,
            z_start=None,
            z_end=None,
        ):
        posData = self.data[self.pos_i]
        self.current_frame_i = posData.frame_i
        
        # Store data for current frame
        self.store_data()
        if endFrame_i is None:
            self.app.restoreOverrideCursor()
            return
        
        segmSizeT = len(posData.segm_data)
        for i in range(posData.frame_i+1, segmSizeT):
            assignments = {}
            lab = posData.allData_li[i]['labels']
            if lab is None and not includeUnvisited:
                self.enqAutosave()
                break

            if lab is not None:
                # Visited frame
                posData.frame_i = i
                self.get_data(lin_tree_init=False)
                if is_2d_edit_on_3d:
                    lab = self.get_2Dlab(posData.lab)
                    rp = self.rpCurr2D()
                else:
                    lab = posData.lab
                    rp = posData.rp

                if self.onlyTracking:
                    self.tracking(enforce=True)
                elif not posData.IDs:
                    continue
                else:
                    for old_ID, new_ID in oldIDnewIDMapper:
                        self._changeIDhelper(
                            lab, old_ID, new_ID, rp, assignments,
                            z_start=z_start, z_end=z_end,
                        )  

                    if is_2d_edit_on_3d:
                        self.set_2Dlab(lab)

                    assignments = assignments if not merging_IDs else None
                    assignments = assignments if is_entire_obj else None
                    self.update_rp(
                        draw=False,
                        assignments=assignments
                    )
                self.store_data(autosave=i==endFrame_i)
            elif includeUnvisited:
                # Unvisited frame (includeUnvisited = True)
                lab = posData.segm_data[i]
                if is_2d_edit_on_3d:
                    lab = self.get_2Dlab(lab)
                    rp = self.rpCurr2D(frame_i=i)
                else:
                    lab = lab
                    rp = posData.allData_li[i]['regionprops']

                assignments = {}
                for old_ID, new_ID in oldIDnewIDMapper:
                    self._changeIDhelper(
                        lab, old_ID, new_ID, rp, assignments,
                        z_start=z_start, z_end=z_end,
                    ) 

                use_assignments = (
                    is_entire_obj and not merging_IDs
                )

                if use_assignments:
                    rp.update_regionprops_via_assignments(assignments, lab)
                else:
                    posData.segm_data[i][self.z_lab()] = lab
                    rp.update_regionprops(lab)
        
        # Back to current frame
        posData.frame_i = self.current_frame_i
        self.get_data()
        self.app.restoreOverrideCursor()

    def instructHowDeleteID(self):
        if 'showInfoDeleteObject' not in self.df_settings.index:
            self.df_settings.at['showInfoDeleteObject', 'value'] = 'Yes'
        
        showInfoDeleteObject = (
            self.df_settings.at['showInfoDeleteObject', 'value'] == 'Yes'
        )
        if not showInfoDeleteObject:
            return
        
        actionText = self.middleClickText()
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'You have deleted an object using the eraser tool.<br><br>'
            'Did you know that you can use the "Delete object" action<br>'
            'to <b>delete an object with a single click</b>?<br><br>'
            f'To do so, use the following action: <code>{actionText}</code><br><br>'
            'Note: You can also set a custom shortcut by going to the menu<br>'
            '<code>Settings --&gt; Customise keyboard shortcuts...</code>.'
        )
        doNotShowAgainCheckbox = QCheckBox('Do not show again')
        msg.information(
            self, 'Delete objects with single click', txt,
            widgets=doNotShowAgainCheckbox
        )
        
        showInfoDeleteObjectValue = (
            'No' if doNotShowAgainCheckbox.isChecked() else 'Yes'
        )
        self.df_settings.at['showInfoDeleteObject', 'value'] = (
            showInfoDeleteObjectValue
        )
        self.df_settings.to_csv(settings_csv_path)

    def deleteIDFromLab(
            self, lab, delID, 
            frame_i=None, 
            delMask=None, 
            z_start=None, 
            z_end=None,
        ):
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i if frame_i is None else frame_i

        if frame_i==posData.frame_i:
            rp = posData.rp
        else:
            rp = posData.allData_li[frame_i]['regionprops']
        if rp is None:
            rp = self._acdcRegionProps(lab, precache_centroids=False)

        single_slice_del_in_3D = (
            z_start == z_end and z_start is not None
        )

        if single_slice_del_in_3D:
            lab2D = self.get_2Dlab(lab)
            rp = rp.get_slice_rp(
                self.zSliceScrollBar.sliderPosition(), 
                self.switchPlaneCombobox.depthAxes()
            )

        if isinstance(delID, int):
            delID = [delID]
            
        is_any_id_present = False
        for _delID in delID:
            if _delID in rp.IDs_set:
                is_any_id_present = True
                break

        if delMask is None:
            delMask = np.zeros(lab.shape, dtype=bool)
        else:
            delMask[:] = False
            
        if not is_any_id_present:
            return lab, delMask
        
        if single_slice_del_in_3D:
            delMask2D = self.get_2Dlab(delMask)

        for _delID in delID:
            if _delID not in rp.IDs_set:
                continue
            obj = rp.get_obj_from_ID(_delID)
            obj_mask = obj.image
            if single_slice_del_in_3D:
                delMask2D[obj.slice][obj_mask] = True
            else:
                if z_start is not None:
                    local_z = z_start - obj.bbox[0]
                    obj_mask[:local_z] = False

                if z_end is not None:
                    local_z = z_end - obj.bbox[0]
                    obj_mask[local_z+1:] = False
                
                delMask[obj.slice][obj_mask] = True

        if single_slice_del_in_3D:
            lab2D[delMask2D] = 0
        else:
            lab[delMask] = 0
        
        if single_slice_del_in_3D:
            self.set_2Dlab(lab2D, lab3D=lab)
            if delMask2D is not None:
                self.set_2Dlab(delMask2D, lab3D=delMask)
        
        return lab, delMask

    def splitIDwith3DActionModifiers(
            self, 
            modifiers,
            clickedID: int
        ):
        if not self.isSegm3D:
            return False
        
        z_start, z_end = self.separateBudToolbar.getHowToPropagateZslices(
            self.zSliceScrollBar.sliderPosition()
        )
        if z_start is None and z_end is None:
            return False

        posData = self.data[self.pos_i]
        newID = self.setBrushID(return_val=True)
        lab = posData.lab
        obj = posData.rp.get_obj_from_ID(clickedID)
        slc = obj.slice
        mask = obj.image
        if z_start is not None:
            local_z = z_start - obj.bbox[0]
            mask[:local_z] = False

        if z_end is not None:
            local_z = z_end - obj.bbox[0]
            mask[local_z+1:] = False

        lab[slc][mask] = newID
        return True


    @disableWindow
    def deleteIDmiddleClick(
            self, 
            delIDs: Iterable, 
            applyFutFrames, 
            includeUnvisited,
            modifiers=None
        ):
        self.clearHighlightedID()

        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i

        z_start, z_end = (
            self.get3DZrangeFromKeyModifiers(modifiers)
        )
        is_entire_obj = z_start is None and z_end is None
        
        # Apply Delete ID to future frames if requested
        if applyFutFrames:
            delMask = np.zeros(posData.lab.shape, dtype=bool)
            # Store current data before going to future frames
            self.store_data()
            segmSizeT = len(posData.segm_data)

            for i in range(posData.frame_i+1, segmSizeT):
                lab = posData.allData_li[i]['labels']
                if lab is None and not includeUnvisited:
                    self.enqAutosave()
                    break
                
                if lab is not None:
                    # Get the rest of the stored metadata based on the new lab
                    posData.frame_i = i
                    self.get_data()
                    lab, _ = self.deleteIDFromLab(
                        lab, delIDs, 
                        frame_i=i, 
                        delMask=delMask,
                        z_start=z_start,
                        z_end=z_end,
                    )
            
                    # Store change
                    posData.lab = lab
                    self.update_rp(
                        deletionIDs=delIDs if is_entire_obj else None
                    )
                    self.store_data(autosave=False)
                elif includeUnvisited:
                    # Unvisited frame (includeUnvisited = True)
                    posData.segm_data[i], _ = self.deleteIDFromLab(
                        posData.segm_data[i], delIDs, 
                        frame_i=i, 
                        delMask=delMask, 
                        z_start=z_start,
                        z_end=z_end,
                    )
                    self.update_rp(
                        frame_i=i, 
                        deletionIDs=delIDs if is_entire_obj else None
                    )

        # Back to current frame
        if applyFutFrames:
            posData.frame_i = current_frame_i
            self.get_data()   
            
        posData.lab, delID_mask = self.deleteIDFromLab(
            posData.lab, delIDs, z_start=z_start, z_end=z_end,
        )
        for _delID in delIDs:
            self.clearObjContour(ID=_delID, ax=0)     
            self.clearObjContour(ID=_delID, ax=1)  

        if is_entire_obj:
            self.removeObjectFromRp(delIDs,posData.lab)
        else:
            self.update_rp()

        self.annotateAllObjectTracks()
        self.annotateAssignedObjsAcdcTrackerSecondStep()
        self.store_data(autosave=False)
        self.whitelistPropagateIDs(
            IDs_to_remove=delIDs, curr_frame_only=(not applyFutFrames)
        )
        return delID_mask

    def separateByLabelling(self, lab, rp, maxID=None):
        """
        Label each single object in posData.lab and if the result is more than
        one object then we insert the separated object into posData.lab
        """
        setRp = False
        posData = self.data[self.pos_i]
        if maxID is None:
            maxID = max(posData.IDs, default=1)
        for obj in rp:
            lab_obj = skimage.measure.label(obj.image)
            rp_lab_obj = self._acdcRegionProps(
                lab_obj, precache_centroids=False
            )
            if len(rp_lab_obj)<=1:
                continue
            lab_obj += maxID
            _slice = obj.slice # self.getObjSlice(obj.slice)
            _objMask = obj.image # self.getObjImage(obj.image)
            lab[_slice][_objMask] = lab_obj[_objMask]
            setRp = True
            maxID += 1
        return setRp

    def getLastManualSeparateState(self):
        state = dict()
        state['is_overlay_active'] = self._get_setting_value(
            'sep_IDs_is_overlay_active', True, bool
        )
        state['is_three_points_active'] = self._get_setting_value(
            'sep_IDs_is_three_points_active', True, bool
        )
        state['is_free_hand_active'] = self._get_setting_value(
            'is_free_hand_active', False, bool
        )
        return state

    def setLastManualSeparateState(self, state):
        self.df_settings.loc[
            'sep_IDs_is_overlay_active', 'value'
            ] = int(state.get('is_overlay_active', True))
        self.df_settings.loc[
            'sep_IDs_is_three_points_active', 'value'
            ] = int(state.get('is_three_points_active', True))
        self.df_settings.loc[
            'is_free_hand_active', 'value'
            ] = int(state.get('is_free_hand_active', False))
        self.df_settings.to_csv(self.settings_csv_path)


class KeepIDsGuiElements:
    """Keep IDs tool: keep only the selected objects."""

    def keepIDs_cb(self, checked):
        if checked:
            self.highlightedLab = np.zeros_like(self.currentLab2D)
            if self.isCellCycleInfoChecked(0):
                self.annotOptionsCheckboxes[0]['IDs'].setChecked(True)
            self.uncheckLeftClickButtons(None)
            self.initKeepObjLabelsLayers()      
            self.setAllIDs()
        else:
            # restore items to non-grayed out
            self.clearTempBrushImage()
            alpha = self.imgGrad.labelsAlphaSlider.value()
            self.labelsLayerImg1.setOpacity(alpha)
            self.labelsLayerRightImg.setOpacity(alpha)
            self.ax1_contoursImageItem.setOpacity(1.0)
            self.ax2_contoursImageItem.setOpacity(1.0)
            self.ax1_lostObjImageItem.setOpacity(1.0)
            self.ax2_lostObjImageItem.setOpacity(1.0)
            self.ax1_lostTrackedObjImageItem.setOpacity(1.0)
            self.ax2_lostTrackedObjImageItem.setOpacity(1.0)
        
        self.keepIDsToolbar.setVisible(checked)
        self.highlightedIDopts = None
        self.keptObjectsIDs = widgets.KeptObjectIDsList(
            self.keptIDsLineEdit, self.keepIDsConfirmAction
        )
        # Reset a previous warning (`setText('')` of `KeptObjectIDsList`
        # does not emit `textChanged` if the line edit is already empty)
        self.keptIDsLineEdit.setInstructionsText()
        self.updateAllImages()

        # QTimer.singleShot(300, self.autoRange)

    def initKeepObjLabelsLayers(self, imageItem=None):
        """Set the lookup table (labels colors, transparent background) of
        the layer showing the selected objects (default "Keep IDs" layer)"""
        if imageItem is None:
            imageItem = self.keepIDsTempLayerLeft
        lut = np.zeros((len(self.lut), 4), dtype=np.uint8)
        lut[:,:-1] = self.lut
        lut[:,-1:] = 255
        lut[0] = [0,0,0,0]
        imageItem.setLevels([0, len(lut)])
        imageItem.setLookupTable(lut)
        if imageItem is self.keepIDsTempLayerLeft:
            # The linked right image layer does not inherit the lut
            self.keepIDsTempLayerRight.setLookupTable(lut)

    def updateTempLayerKeepIDs(self):
        if not self.keepIDsButton.isChecked():
            return

        keptLab = self.getObjectsLab2D(self.keptObjectsIDs)
        self.keepIDsTempLayerLeft.setImage(keptLab, autoLevels=False)

    def _keepObjects(self, keepIDs=None, lab=None, rp=None):
        posData = self.data[self.pos_i]
        if lab is None:
            lab = posData.lab
        
        if rp is None:
            rp = posData.rp
        
        if keepIDs is None:
            keepIDs = self.keptObjectsIDs

        for obj in rp:
            if obj.label in keepIDs:
                continue

            lab[obj.slice][obj.image] = 0
        
        return lab

    def updateKeepIDs(self, IDs):
        posData = self.data[self.pos_i]

        self.clearHighlightedText()

        # The labels can change while "Keep IDs" is active (undo/redo,
        # applying the selection, visiting new frames) --> refresh the IDs
        # existing in any frame of the current position (cheap, it reads
        # the cached regionprops of each frame)
        self.setAllIDs(pos_i=self.pos_i)
        existingIDs = posData.allIDs

        isAnyIDnotExisting = False
        # Add the existing IDs from the line edit to the kept IDs
        for ID in IDs:
            if ID not in existingIDs:
                isAnyIDnotExisting = True
                continue
            if ID not in self.keptObjectsIDs:
                self.keptObjectsIDs.append(ID, editText=False)
                self.highlightLabelID(ID)

        # Remove the kept IDs that are not in the line edit anymore or that
        # do not exist (reported as IGNORED by the line edit)
        # (iterate over a copy because IDs are removed from the list)
        typedIDs = set(IDs)
        for ID in list(self.keptObjectsIDs):
            if ID not in typedIDs or ID not in existingIDs:
                self.keptObjectsIDs.remove(ID, editText=False)
        
        self.updateTempLayerKeepIDs()
        if isAnyIDnotExisting:
            self.keptIDsLineEdit.warnNotExistingID()
        else:
            self.keptIDsLineEdit.setInstructionsText()

    def askKeepIDsDeleteAllObjectsCurrentFrame(self):
        """Ask confirmation before applying "Keep IDs" when none of the IDs
        to keep is in the current frame (e.g., IDs existing only in other
        frames), since all its objects would be deleted.

        Returns True if the user wants to proceed."""
        posData = self.data[self.pos_i]
        currentIDs = set(posData.IDs)
        if not currentIDs or currentIDs.intersection(self.keptObjectsIDs):
            return True

        keptIDsText = myutils.format_IDs(self.keptObjectsIDs)
        txt = html_utils.paragraph(f"""
            None of the IDs to keep (<code>{keptIDsText}</code>) is present
            in the current frame.<br><br>
            Applying "Keep IDs" will <b>delete ALL the objects of the current
            frame</b>.<br><br>
            Do you want to continue?
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Delete all objects of the current frame?', txt,
            buttonsTexts=('Cancel', 'Yes, delete all objects')
        )
        return not msg.cancel

    @exception_handler
    def applyKeepObjects(self):
        if not self.keptObjectsIDs:
            # E.g., Enter pressed in the IDs line edit without (existing)
            # IDs --> applying would delete all the objects
            self.keptIDsLineEdit.warnNoIDsSelected()
            self.logger.warning(
                '[WARNING]: "Keep IDs" not applied because no existing ID is '
                'selected (it would delete all the objects).'
            )
            return

        proceed = self.askKeepIDsDeleteAllObjectsCurrentFrame()
        if not proceed:
            return

        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        self._keepObjects()
        self.highlightHoverIDsKeptObj(0, 0, hoverID=0)
        
        posData = self.data[self.pos_i]

        self.update_rp()
        # Repeat tracking
        self.tracking(enforce=True, assign_unique_new_IDs=False)

        if self.isSnapshot:
            self.fixCcaDfAfterEdit('Deleted non-selected objects')
            self.updateAllImages()
            self.keptObjectsIDs = widgets.KeptObjectIDsList(
                self.keptIDsLineEdit, self.keepIDsConfirmAction
            )
            return
        else:
            removeAnnot = self.warnEditingWithCca_df(
                'Deleted non-selected objects', get_answer=True
            )
            if not removeAnnot:
                # We can propagate changes only if the user agrees on 
                # removing annotations
                return
        
        self.current_frame_i = posData.frame_i
        if posData.frame_i > 0:
            txt = html_utils.paragraph("""
                Do you want to <b>remove un-kept objects in the past</b> frames too?
            """)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            _, _, applyToPastButton = msg.question(
                self, 'Propagate to past frames?', txt,
                buttonsTexts=('Cancel', 'No', 'Yes, apply to past frames')
            )
            if msg.cancel:
                return
            if msg.clickedButton == applyToPastButton:
                self.store_data()
                self.logger.info('Applying keep objects to past frames...')
                if not removeAnnot and posData.cca_df is not None:
                    delIDs = [
                        ID for ID in posData.cca_df.index 
                        if ID not in posData.IDs
                    ]
                    self.update_cca_df_deletedIDs(posData, delIDs)
                
                for i in tqdm(range(posData.frame_i), ncols=100):
                    lab = posData.allData_li[i]['labels']
                    rp = posData.allData_li[i]['regionprops']
                    keepLab = self._keepObjects(lab=lab, rp=rp)
                    # Store change
                    posData.allData_li[i]['labels'] = keepLab.copy()
                    # Get the rest of the stored metadata based on the new lab
                    posData.frame_i = i
                    self.get_data()
                    self.update_rp()
                    self.store_data(autosave=False)
                
                # Back to current frame
                posData.frame_i = self.current_frame_i
                self.get_data()

        # Ask to propagate change to all future visited frames
        key = 'Keep ID'
        askAction = self.askHowFutureFramesActions[key]
        doNotShow = not askAction.isChecked()
        (UndoFutFrames, applyFutFrames, endFrame_i,
        doNotShowAgain) = self.propagateChange(
            self.keptObjectsIDs, key, doNotShow,
            posData.UndoFutFrames_keepID, posData.applyFutFrames_keepID,
            force=True, applyTrackingB=True
        )

        if UndoFutFrames is None:
            # Future frames dialog cancelled (see `propagateChange`)
            self.keptObjectsIDs = widgets.KeptObjectIDsList(
                self.keptIDsLineEdit, self.keepIDsConfirmAction
            )
            return

        posData.doNotShowAgain_keepID = doNotShowAgain
        posData.UndoFutFrames_keepID = UndoFutFrames
        posData.applyFutFrames_keepID = applyFutFrames
        includeUnvisited = posData.includeUnvisitedInfo['Keep ID']

        if applyFutFrames:
            self.store_data()

            self.logger.info('Applying to future frames...')
            pbar = tqdm(total=posData.SizeT-posData.frame_i-1, ncols=100)
            segmSizeT = len(posData.segm_data)
            if not removeAnnot and posData.cca_df is not None:
                delIDs = [
                    ID for ID in posData.cca_df.index 
                    if ID not in posData.IDs
                ]
                self.update_cca_df_deletedIDs(posData, delIDs)
                
            for i in range(posData.frame_i+1, segmSizeT):
                lab = posData.allData_li[i]['labels']
                if lab is None and not includeUnvisited:
                    self.enqAutosave()
                    pbar.update(posData.SizeT-i)
                    break
                
                rp = posData.allData_li[i]['regionprops']

                if lab is not None:
                    keepLab = self._keepObjects(lab=lab, rp=rp)
                    # Store change
                    posData.allData_li[i]['labels'] = keepLab.copy()
                    # Get the rest of the stored metadata based on the new lab
                    posData.frame_i = i
                    self.get_data()
                    self.update_rp()
                    self.store_data(autosave=False)
                elif includeUnvisited:
                    # Unvisited frame (includeUnvisited = True)
                    lab = posData.segm_data[i]
                    rp = self._acdcRegionProps(
                        lab, precache_centroids=False
                    )
                    keepLab = self._keepObjects(lab=lab, rp=rp)
                    posData.segm_data[i] = keepLab
                    self.update_rp(frame_i=i)
                
                pbar.update()
            pbar.close()
        
        # Back to current frame
        if applyFutFrames:
            posData.frame_i = self.current_frame_i
            self.get_data()

        self.keptObjectsIDs = widgets.KeptObjectIDsList(
            self.keptIDsLineEdit, self.keepIDsConfirmAction
        )

    def highlightHoverIDsKeptObj(self, x, y, hoverID=None):
        if hoverID is None:
            try:
                hoverID = self.currentLab2D[int(y), int(x)]
            except IndexError:
                return

        self.highlightSearchedID(hoverID, greyOthers=False)
        
        if hoverID == 0 and self.highlightedID == 0:
            return
        
        if hoverID == 0 and self.highlightedID != 0:
            self.clearHighlightedKeepIDs()
            for ID in self.keptObjectsIDs:
                self.highlightLabelID(ID)
            return
        
        posData = self.data[self.pos_i]
        obj = posData.rp.get_obj_from_ID(hoverID, warn=False)
        if obj is None:
            return
        
        self.goToZsliceSearchedID(obj)
        
        for ID in self.keptObjectsIDs:
            self.highlightLabelID(ID)

    def clearHighlightedKeepIDs(self):
        self.setAllTextAnnotations()
        self.highlightedID = 0
        self.searchedIDitemRight.setData([], [])
        self.searchedIDitemLeft.setData([], [])
        self.highLightIDLayerImg1.clear()
        self.highLightIDLayerRightImage.clear()
