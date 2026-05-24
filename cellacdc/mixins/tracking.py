"""Qt view adapter for tracking and manual tracking workflows."""

from __future__ import annotations

import cv2
from functools import partial
from typing import Iterable, List, Set

import numpy as np
import pyqtgraph as pg
import skimage.measure
from tqdm import tqdm
from qtpy.QtCore import QTimer
from qtpy.QtGui import QFont

from cellacdc import apps, exception_handler, html_utils, widgets
from cellacdc.trackers.CellACDC import CellACDC_tracker


font_13px = QFont()
font_13px.setPixelSize(13)


class Tracking:
    """Extracted from guiWin."""

    def _drawGhostContour(self, x, y):
        if self.ghostObject is None:
            return
        
        ID = self.ghostObject.label
        yc, xc = self.ghostObject.local_centroid
        Dx = x-xc
        Dy = y-yc
        xx = self.ghostObject.xx_contour + Dx
        yy = self.ghostObject.yy_contour + Dy
        self.ghostContourItemLeft.setData(
            xx, yy, fontSize=self.fontSize, ID=ID, y_cursor=y, x_cursor=x
        )
        self.ghostContourItemRight.setData(
            xx, yy, fontSize=self.fontSize, ID=ID, y_cursor=y, x_cursor=x
        )

    def _drawGhostMask(self, x, y):
        if self.ghostObject is None:
            return
        
        self.clearGhostMask()
        ID = self.ghostObject.label
        h, w = self.ghostObject.image.shape[-2:]
        yc, xc = self.ghostObject.local_centroid
        Dx = int(x-xc)
        Dy = int(y-yc)
        bbox = ((Dy, Dy+h), (Dx, Dx+w))

        Y, X = self.currentLab2D.shape
        slices = myutils.get_slices_local_into_global_arr(bbox, (Y, X))
        slice_global_to_local, slice_crop_local = slices

        obj_image = self.ghostObject.image[slice_crop_local]

        self.ghostMaskItemLeft.image[slice_global_to_local][obj_image] = ID
        self.ghostMaskItemLeft.updateGhostImage(
            fontSize=self.fontSize, ID=ID, y_cursor=y, x_cursor=x
        )

        self.ghostMaskItemRight.image[slice_global_to_local][obj_image] = ID
        self.ghostMaskItemRight.updateGhostImage(
            fontSize=self.fontSize, ID=ID, y_cursor=y, x_cursor=x
        )

    def _drawManualBackgroundObjContour(self, x, y):
        if self.manualBackgroundObj is None:
            return
        
        ID = self.manualBackgroundObj.label
        yc, xc = self.manualBackgroundObj.local_centroid
        Dx = x-xc
        Dy = y-yc
        xx = self.manualBackgroundObj.xx_contour + Dx
        yy = self.manualBackgroundObj.yy_contour + Dy
        self.manualBackgroundObjItem.setData(
            xx, yy, fontSize=self.fontSize, ID=ID, y_cursor=y, x_cursor=x
        )

    def addManualBackgroundItems(self):
        self.manualBackgroundObjItem.addToPlotItem()
        self.ax1.addItem(self.manualBackgroundImageItem)

    def addManualBackgroundObject(self, x, y):
        posData = self.data[self.pos_i]
        
        if not hasattr(self, 'manualBackgroundObj'):
            self.initManualBackgroundObject()
        
        Y, X = self.currentLab2D.shape
        ymin, xmin, ymax, xmax = self.manualBackgroundObj.bbox
        width, height = xmax-xmin, ymax-ymin
        yc, xc = self.manualBackgroundObj.local_centroid
        xstart, ystart = round(x-xc), round(y-yc)
        xstart = xstart if xstart >= 0 else 0
        ystart = ystart if ystart >= 0 else 0
        
        xend = xstart+width
        yend = ystart+height
        xend = xend if xend <= X else X
        yend = yend if yend <= Y else Y
        
        width = xend-xstart
        height = yend-ystart
        
        obj_image = self.manualBackgroundObj.image[:height, :width]
        obj_slice = (slice(ystart, yend), slice(xstart, xend))
        ID = self.manualBackgroundObj.label
        self.clearManualBackgroundObject(ID)
        posData.manualBackgroundLab[obj_slice][obj_image] = ID
        
        if ID in self.manualBackgroundTextItems:
            self.manualBackgroundTextItems[ID].setPos(x, y)
            return
        
        textItem = pg.TextItem(
            text=str(ID), color='r', anchor=(0.5, 0.5)
        )
        textItem.setFont(font_13px)
        textItem.setPos(x, y)
        self.manualBackgroundTextItems[ID] = textItem
        
        self.ax1.addItem(textItem)

    def addManualTrackingItems(self):
        self.ghostContourItemLeft.addToPlotItem()
        self.ghostContourItemRight.addToPlotItem()

        self.ghostMaskItemLeft.addToPlotItem()
        self.ghostMaskItemRight.addToPlotItem()

        Y, X = self.img1.image.shape[:2]
        self.ghostMaskItemLeft.initImage((Y, X))
        self.ghostMaskItemRight.initImage((Y, X))

        self.updateGhostMaskOpacity()

    def annotateAssignedObjsAcdcTrackerSecondStep(self):
        posData = self.data[self.pos_i]
        annotInfo = posData.acdcTracker2stepsAnnotInfo.get(posData.frame_i)
        if annotInfo is None:
            return
        
        new_objs_1st_step, lost_objs_1st_step = annotInfo
        for lostObj, newObj in zip(lost_objs_1st_step, new_objs_1st_step):
            allContours = self.getObjContours(lostObj, all_external=True) 
            for objContours in allContours:
                isObjVisible = self.isObjVisible(newObj.bbox)
                if not isObjVisible:
                    continue
                xx = objContours[:,0] + 0.5
                yy = objContours[:,1] + 0.5
                self.yellowContourScatterItem.addPoints(xx, yy)
                
            y1, x1 = self.getObjCentroid(lostObj.centroid)
            y2, x2 = self.getObjCentroid(newObj.centroid)
            xx, yy = core.get_line(y1, x1, y2, x2, dashed=False)
            self.ax1_oldMothBudLinesItem.addPoints(xx, yy)
        
        posData.acdcTracker2stepsAnnotInfo[posData.frame_i] = None

    def clearAssignedObjsSecondStep(self):
        posData = self.data[self.pos_i]
        posData.acdcTracker2stepsAnnotInfo[posData.frame_i] = None

    def clearGhost(self):
        self.clearGhostContour()
        self.clearGhostMask()

    def clearGhostContour(self):
        self.ghostContourItemLeft.clear()
        self.ghostContourItemRight.clear()
        self.manualBackgroundObjItem.clear()

    def clearGhostMask(self):
        self.ghostMaskItemLeft.clear()
        self.ghostMaskItemRight.clear()

    def clearManualBackgroundAnnotations(self):
        try:
            for textItem in self.manualBackgroundTextItems.values():
                textItem.setText('')
        except Exception as error:
            pass

    def clearManualBackgroundObject(self, ID):
        posData = self.data[self.pos_i]
        mask = posData.manualBackgroundLab==ID
        posData.manualBackgroundImage[mask, :] = 0
        posData.manualBackgroundLab[mask] = 0

    def doSkipTracking(self, against_next: bool, enforce: bool):
        if self.isSnapshot:
            return True
        
        mode = str(self.modeComboBox.currentText())
        if mode != 'Segmentation and Tracking':
            return True
        
        if self.UserEnforced_DisabledTracking:
            return True
        
        if not self.realTimeTrackingToggle.isChecked():
            return True
        
        posData = self.data[self.pos_i]
        if against_next:
            reference_lab = posData.allData_li[posData.frame_i+1]['labels']
            if reference_lab is None:
                # Next frame never visited --> cannot track against next
                return True

            if posData.frame_i == posData.SizeT - 1:
                # Last frame --> cannot track against next
                return True
    
        else:
            # check that we are not on the last frame
            if posData.frame_i == 0:
                return True
            
        if enforce or self.UserEnforced_Tracking:
            # Enforce even if not last visited frame
            return False
        
        is_first_time_on_next_frame = self.isFirstTimeOnNextFrame()
        skip_tracking = not is_first_time_on_next_frame
        
        return skip_tracking

    def drawManualBackgroundObj(self, x, y):
        if x is None or y is None:
            self.clearGhost()
            return
        
        self._drawManualBackgroundObjContour(x, y)

    def drawManualTrackingGhost(self, x, y):
        if not self.manualTrackingToolbar.showGhostCheckbox.isChecked():
            return
        
        if x is None or y is None:
            self.clearGhost()
            return
        
        if self.manualTrackingToolbar.ghostContourRadiobutton.isChecked():
            self._drawGhostContour(x, y)
        else:
            self._drawGhostMask(x, y)

    def enableSmartTrack(self, checked):
        posData = self.data[self.pos_i]
        # Disable tracking for already visited frames

        if posData.allData_li[posData.frame_i]['labels'] is not None:
            trackingEnabled = True
        else:
            trackingEnabled = False

        if checked:
            self.UserEnforced_DisabledTracking = False
            self.UserEnforced_Tracking = False
        else:
            if trackingEnabled:
                self.UserEnforced_DisabledTracking = True
                self.UserEnforced_Tracking = False
            else:
                self.UserEnforced_DisabledTracking = False
                self.UserEnforced_Tracking = True

    def getLastTrackedFrame(self, posData):
        last_tracked_i = 0
        for frame_i, data_dict in enumerate(posData.allData_li):
            lab = data_dict['labels']
            if lab is None:
                frame_i -= 1
                break
        if frame_i > 0:
            return frame_i
        else:
            return last_tracked_i

    def getTrackedLostIDs(self, prev_lab=None, IDs_in_frames=None, frame_i=None):
        trackedLostIDs = set()
        posData = self.data[self.pos_i]
        if self.isExportingVideo:
            posData.trackedLostIDs = trackedLostIDs
            return trackedLostIDs
        
        retrackedLostcent = set()
        if frame_i is None:
            frame_i = posData.frame_i
            
        if prev_lab is None:
            prev_lab = self.get_labels(
                from_store=True, 
                frame_i=posData.frame_i-1, 
                return_existing=False,
                return_copy=False
            )

        if IDs_in_frames is None:
            IDs_in_frames = posData.IDs

        try:
            tracked_lost_centroids = posData.tracked_lost_centroids[frame_i]
        except KeyError:
            tracked_lost_centroids = set()

        for centroid in tracked_lost_centroids:
            if len(centroid) < 3 and prev_lab.ndim == 3:
                # Ignore wrongly stored centroids
                continue

            ID = prev_lab[centroid]
            if ID == 0:
                continue

            if ID in IDs_in_frames:
                retrackedLostcent.add(centroid)
                continue

            trackedLostIDs.add(ID)

        posData.tracked_lost_centroids[frame_i] = (
            tracked_lost_centroids - retrackedLostcent
        )
        posData.trackedLostIDs = trackedLostIDs

        return trackedLostIDs

    def get_last_tracked_i(self):
        posData = self.data[self.pos_i]
        last_tracked_i = 0
        for frame_i, data_dict in enumerate(posData.allData_li):
            lab = data_dict['labels']
            if lab is None and frame_i == 0:
                last_tracked_i = 0
                break
            elif lab is None:
                last_tracked_i = frame_i-1
                break
            else:
                last_tracked_i = posData.segmSizeT-1
        return last_tracked_i

    def handleAdditionalInfoRealTimeTracker(self, prev_rp, *args):
        if self._rtTrackerName == 'CellACDC_normal_division':
            tracked_lost_IDs = args[0]
            self.setTrackedLostCentroids(prev_rp, tracked_lost_IDs)
        elif self._rtTrackerName == 'CellACDC_2steps':
            if args[0] is None:
                return
            posData = self.data[self.pos_i]
            posData.acdcTracker2stepsAnnotInfo[posData.frame_i] = args[0]

    def initGhostObject(self, ID=None):
        mode = self.modeComboBox.currentText()
        if mode != 'Segmentation and Tracking':
            self.ghostObject = None
            return
        
        if not self.manualTrackingButton.isChecked():
            self.ghostObject = None
            return
        
        if not self.manualTrackingToolbar.showGhostCheckbox.isChecked():
            self.ghostObject = None
            return
        
        if ID is None:
            ID = self.manualTrackingToolbar.spinboxID.value()
        
        posData = self.data[self.pos_i]
        if posData.frame_i == 0:
            self.ghostObject = None
            return
        
        prevFrameRp = posData.allData_li[posData.frame_i-1]['regionprops']
        if prevFrameRp is None:
            self.ghostObject = None
            return
        
        for obj in prevFrameRp:
            if obj.label != ID:
                continue
            self.ghostObject = obj
            break
        else:
            self.ghostObject = None
            self.manualTrackingToolbar.showWarning(
                f'The ID {ID} does not exist in previous frame '
                '--> starting a new track.'
            )
            return
        
        self.manualTrackingToolbar.clearInfoText()

        self.ghostObject.contour = self.getObjContours(
            self.ghostObject, local=True
        )
        self.ghostObject.xx_contour = self.ghostObject.contour[:,0]
        self.ghostObject.yy_contour = self.ghostObject.contour[:,1]

        self.ghostMaskItemLeft.initLookupTable(self.lut[ID])
        self.ghostMaskItemRight.initLookupTable(self.lut[ID])

    def initManualBackgroundObject(self, ID=None):
        if not self.manualBackgroundButton.isChecked():
            self.manualBackgroundObj = None
            return
        
        if ID is None:
            ID = self.manualBackgroundToolbar.spinboxID.value()
        
        posData = self.data[self.pos_i]
        if ID not in posData.IDs:
            self.manualBackgroundObj = None
            self.manualBackgroundToolbar.showWarning(
                f'The ID {ID} does not exist'
            )
            self.manualBackgroundObjItem.clear()
            return
        
        ID_idx = posData.IDs_idxs[ID]
        self.manualBackgroundObj = posData.rp[ID_idx]
        
        self.manualBackgroundToolbar.clearInfoText()
        self.manualBackgroundObj.contour = self.getObjContours(
            self.manualBackgroundObj, local=True
        )
        xx_contour = self.manualBackgroundObj.contour[:,0]
        yy_contour = self.manualBackgroundObj.contour[:,1]
        self.manualBackgroundObj.xx_contour = xx_contour
        self.manualBackgroundObj.yy_contour = yy_contour

    def initRealTimeTracker(self, force=False):
        for rtTrackerAction in self.trackingAlgosGroup.actions():
            if rtTrackerAction.isChecked():
                break
        
        aliases = myutils.aliases_real_time_trackers(reverse=True)
        
        rtTracker = rtTrackerAction.text()
        rtTracker_txt = rtTracker

        if rtTracker in aliases:
            rtTracker = aliases[rtTracker]
        
        if rtTracker == 'Cell-ACDC':
            return
        if rtTracker == 'YeaZ':
            return
        
        if self.isRealTimeTrackerInitialized and not force:
            return
        
        self.logger.info(f'Initializing {rtTracker_txt} tracker...')
        self._rtTrackerName = rtTracker
        posData = self.data[self.pos_i]
        realTimeTracker, track_frame_params = myutils.init_tracker(
            posData, rtTracker, qparent=self, realTime=True
        )
        if realTimeTracker is None:
            self.logger.info(f'{rtTracker} tracker initialization cancelled.')
            return
        
        self.realTimeTracker = realTimeTracker
        self.track_frame_params = track_frame_params
        self.logger.info(f'{rtTracker} tracker successfully initialized.')
        if 'image_channel_name' in self.track_frame_params:
            # Remove the channel name since it was already loaded in init_tracker
            del self.track_frame_params['image_channel_name']

    def initSegmTrackMode(self):
        posData = self.data[self.pos_i]
        last_tracked_i = self.get_last_tracked_i()
        
        if posData.frame_i > last_tracked_i:
            # Prompt user to go to last tracked frame
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                f'The last visited frame in "Segmentation and Tracking mode" '
                f'is frame {last_tracked_i+1}.\n\n'
                f'We recommend to resume from that frame.<br><br>'
                'How do you want to proceed?'
            )
            goToButton, stayButton = msg.warning(
                self, 'Go to last visited frame?', txt,
                buttonsTexts=(
                    f'Resume from frame {last_tracked_i+1} (RECOMMENDED)',
                    f'Stay on current frame {posData.frame_i+1}'
                )
            )
            if msg.clickedButton == goToButton:
                posData.frame_i = last_tracked_i
                self.lastFrameRanOnFirstVisitTools = posData.frame_i
                self.get_data()
                self.updateAllImages()
                self.updateScrollbars()
            else:
                last_tracked_i = posData.frame_i
                current_frame_i = posData.frame_i
                self.lastFrameRanOnFirstVisitTools = posData.frame_i
                self.logger.info(
                    f'Storing data up until frame n. {current_frame_i+1}...'
                )
                pbar = tqdm(total=current_frame_i+1, ncols=100)
                for i in range(current_frame_i):
                    posData.frame_i = i
                    self.get_data()
                    self.store_data(autosave=i==current_frame_i-1)
                    pbar.update()
                pbar.close()

                posData.frame_i = current_frame_i
                self.get_data()
        
        self.highlightLostNew()
        self.updateLastCheckedFrameWidgets(last_tracked_i)

        self.isFirstTimeOnNextFrame()
        self.initRealTimeTracker()

    def isFirstTimeOnNextFrame(self):
        posData = self.data[self.pos_i]
        posData.last_tracked_i = self.navigateScrollBar.maximum()-1
        return posData.frame_i > posData.last_tracked_i

    def keepOnlyNewIDAssignedObjsSecondStep(self, trackedID):
        posData = self.data[self.pos_i]
        annotInfo = posData.acdcTracker2stepsAnnotInfo.get(posData.frame_i)
        
        if annotInfo is None:
            return 
        
        new_objs_1st_step, lost_objs_1st_step = annotInfo
        correct_new_objs, correct_lost_objs = [], []
        for lostObj, newObj in zip(lost_objs_1st_step, new_objs_1st_step):
            newObj_ID = posData.lab[newObj.slice][newObj.image][0]
            if newObj_ID != trackedID:
                continue
            
            correct_new_objs.append(newObj)
            correct_lost_objs.append(lostObj)
        
        if not correct_new_objs:
            posData.acdcTracker2stepsAnnotInfo[posData.frame_i] = None
        else:
            posData.acdcTracker2stepsAnnotInfo[posData.frame_i] = (
                correct_new_objs, correct_lost_objs
            )

    def manualBackground_cb(self, checked):
        if checked:
            posData = self.data[self.pos_i]
            minID = min(posData.IDs, default=0)
            if minID == self.manualBackgroundToolbar.spinboxID.value():
                self.initManualBackgroundObject()
            else:
                self.manualBackgroundToolbar.spinboxID.setValue(minID)
            # self.initManualBackgroundObject()
            # self.initManualBackgroundImage()
            self.addManualBackgroundItems()
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.manualBackgroundButton)
            self.connectLeftClickButtons()
            self.updateAllImages()
        else:
            self.removeManualTrackingItems()
            self.clearGhost()
            self.clearManualBackgroundAnnotations()
        self.manualBackgroundToolbar.setVisible(checked)

    def manualTracking_cb(self, checked):
        self.manualTrackingToolbar.setVisible(checked)
        if checked:
            self.realTimeTrackingToggle.previousStatus = (
                self.realTimeTrackingToggle.isChecked()
            )
            self.realTimeTrackingToggle.setChecked(False)
            self.UserEnforced_DisabledTracking_previousStatus = (
                self.UserEnforced_DisabledTracking
            )
            self.UserEnforced_Tracking_previousStatus = (
                self.UserEnforced_Tracking
            )

            self.UserEnforced_DisabledTracking = True
            self.UserEnforced_Tracking = False
            self.initGhostObject()
            self.addManualTrackingItems()
        else:
            self.realTimeTrackingToggle.setChecked(
                self.realTimeTrackingToggle.previousStatus
            )
            self.UserEnforced_DisabledTracking = (
                self.UserEnforced_DisabledTracking_previousStatus
            )
            self.UserEnforced_Tracking = (
                self.UserEnforced_Tracking_previousStatus
            )
            self.removeManualTrackingItems()
            self.clearGhost()

    def manuallyEditTracking(self, tracked_lab, allIDs):
        posData = self.data[self.pos_i]
        infoToRemove = []
        # Correct tracking with manually changed IDs
        maxID = max(allIDs, default=1)
        for y, x, new_ID in posData.editID_info:
            old_ID = tracked_lab[y, x]
            if old_ID == 0 or old_ID == new_ID:
                infoToRemove.append((y, x, new_ID))
                continue
            if new_ID in allIDs:
                tempID = maxID+1
                tracked_lab[tracked_lab == old_ID] = tempID
                tracked_lab[tracked_lab == new_ID] = old_ID
                tracked_lab[tracked_lab == tempID] = new_ID
            else:
                tracked_lab[tracked_lab == old_ID] = new_ID
                if new_ID > maxID:
                    maxID = new_ID
        for info in infoToRemove:
            posData.editID_info.remove(info)

    def realTimeTrackingClicked(self, checked):
        # Event called ONLY if the user click on Disable tracking
        # NOT called if setChecked is called. This allows to keep track
        # of the user choice. This way user con enforce tracking
        # NOTE: I know two booleans doing the same thing is overkill
        # but the code is more readable when we actually need them

        posData = self.data[self.pos_i]
        isRealTimeTrackingDisabled = not checked

        # Turn off smart tracking
        self.enableSmartTrackAction.toggled.disconnect()
        self.enableSmartTrackAction.setChecked(False)
        if isRealTimeTrackingDisabled:
            self.UserEnforced_DisabledTracking = True
            self.UserEnforced_Tracking = False
        else:
            txt = html_utils.paragraph("""

            Do you want to keep <b>tracking always active</b> including on already 
            visited frames?<br><br>
            Note: To re-activate automatic handling of tracking go to<br> 
            <code>Edit --> Smart handling of enabling/disabling tracking</code>.

            """)
            msg = widgets.myMessageBox(showCentered=False, wrapText=False)
            yesButton, noButton = msg.question(
                self, 'Keep tracking always active?', txt, 
                buttonsTexts=('Yes', 'No')
            )
            if msg.clickedButton == yesButton:
                self.repeatTracking()
                self.UserEnforced_DisabledTracking = False
                self.UserEnforced_Tracking = True
            else:
                self.enableSmartTrackAction.setChecked(True)

    def removeManualBackgroundItems(self):
        self.manualBackgroundObjItem.removeFromPlotItem()
        self.ax1.removeItem(self.manualBackgroundImageItem)

    def removeManualTrackingItems(self):        
        self.ghostContourItemLeft.removeFromPlotItem()
        self.ghostContourItemRight.removeFromPlotItem()

        self.ghostMaskItemLeft.removeFromPlotItem()
        self.ghostMaskItemRight.removeFromPlotItem()

    def repeatTracking(self):
        posData = self.data[self.pos_i]
        prev_lab = self.get_2Dlab(posData.lab).copy()
        self.tracking(enforce=True, DoManualEdit=False)
        if posData.editID_info:
            editedIDsInfo = {
                posData.lab[y,x]:newID 
                for y, x, newID in posData.editID_info
                if posData.lab[y,x] != newID
            }
            editedIDsInfoItems = [
                f'ID {oldID} --> {newID}'
                for oldID, newID in editedIDsInfo.items()
            ]
            editIDul = html_utils.to_list(editedIDsInfoItems)
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(f"""
                You requested to repeat tracking but <b>there are manually 
                edited IDs</b> (see edited IDs in the details section below)
                <br><br>
                Do you want to keep these edits or ignore them?
            """)
            keepManualEditButton = widgets.okPushButton(
                'Keep manually edited IDs'
            )
            ignoreButton = widgets.noPushButton(
                'Ignore manually edited IDs'
            )
            msg.question(
                self, 'Repeat tracking mode', txt, 
                buttonsTexts=(keepManualEditButton, ignoreButton), 
                detailsText=editIDul
            )
            if msg.cancel:
                return
            if msg.clickedButton == keepManualEditButton:
                allIDs = [obj.label for obj in posData.rp]
                lab2D = self.get_2Dlab(posData.lab)
                self.manuallyEditTracking(lab2D, allIDs)
                self.update_rp()
                self.setAllTextAnnotations()
                self.highlightLostNew()
                # self.checkIDsMultiContour()
            else:
                posData.editID_info = []
        if np.any(posData.lab != prev_lab):
            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Repeat tracking')
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df('Repeat tracking')
        else:
            self.updateAllImages()

    def repeatTrackingVideo(self, checked=False):
        posData = self.data[self.pos_i]
        win = widgets.selectTrackerGUI(
            posData.SizeT, currentFrameNo=posData.frame_i+1
        )
        win.exec_()
        if win.cancel:
            self.logger.info('Tracking aborted.')
            return

        trackerName = win.selectedItemsText[0]
        start_n = win.startFrame
        stop_n = win.stopFrame
        video_to_track = posData.segm_data
        for frame_i in range(start_n-1, stop_n):
            data_dict = posData.allData_li[frame_i]
            lab = data_dict['labels']
            if lab is None:
                break

            video_to_track[frame_i] = lab
        video_to_track = video_to_track[start_n-1:stop_n]        
        
        self.logger.info(f'Importing {trackerName} tracker...')
        self.tracker, self.track_params, init_params = myutils.init_tracker(
            posData, trackerName, qparent=self, return_init_params=True
        )
        if self.track_params is None:
            self.logger.info('Tracking aborted.')
            return
        
        warningText = myutils.validate_tracker_input(
            self.tracker, video_to_track
        )
        if warningText is not None:
            self.logger.info(warningText)
            self.warnTrackerInputNotValid(trackerName, warningText)
            return        
        
        if 'image_channel_name' in self.track_params:
            # Remove the channel name since it was already loaded in init_tracker
            del self.track_params['image_channel_name']
        
        track_params_log = {
            key: value for key, value in self.track_params.items()
            if key != 'image'
        }
        self.logger.info(
            'Tracking parameters:\n\n'
            f'Initialization parameters: {init_params}\n'
            f'Track parameters: {track_params_log}'
        )

        last_cca_i = self.get_last_cca_frame_i()
        if start_n-2 <= last_cca_i and start_n>1:
            proceed = self.warnRepeatTrackingVideoWithAnnotations(
                last_cca_i, start_n
            )
            if not proceed:
                self.logger.info('Tracking aborted.')
                return
            
            self.logger.info(f'Removing annotations from frame n. {start_n}.')
            self.resetCcaFuture(start_n-1)

        self.start_n = start_n
        self.stop_n = stop_n
        
        info_txt = f'Tracking from frame n. {start_n} to {stop_n}...'
        self.logger.info(info_txt)

        self.progressWin = apps.QDialogWorkerProgress(
            title='Tracking', parent=self, pbarDesc=info_txt
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(stop_n-start_n)
        self.startTrackingWorker(posData, video_to_track)

    def resetManualBackgroundItems(self):
        self.initManualBackgroundImage()
        self.resetManualBackgroundSpinboxID()
        self.drawManualTrackingGhost(self.xHoverImg, self.yHoverImg)
        self.drawManualBackgroundObj(self.xHoverImg, self.yHoverImg)

    def resetManualBackgroundSpinboxID(self):
        if not self.manualBackgroundButton.isChecked():
            self.manualBackgroundObj = None
            return
        
        posData = self.data[self.pos_i]
        minID = min(posData.IDs, default=0)
        self.manualBackgroundToolbar.spinboxID.setValue(minID)

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
            rp_lab_obj = skimage.measure.regionprops(lab_obj)
            if len(rp_lab_obj)<=1:
                continue
            lab_obj += maxID
            _slice = obj.slice # self.getObjSlice(obj.slice)
            _objMask = obj.image # self.getObjImage(obj.image)
            lab[_slice][_objMask] = lab_obj[_objMask]
            setRp = True
            maxID += 1
        return setRp

    def setManualBackgrounNextID(self):
        posData = self.data[self.pos_i]
        currentID = self.manualBackgroundObj.label
        idx = posData.IDs_idxs[currentID]
        next_idx = idx + 1
        if next_idx >= len(posData.IDs):
            return
        next_ID = posData.IDs[next_idx]
        self.manualBackgroundToolbar.spinboxID.setValue(next_ID)

    def setManualBackgroundImage(self):
        if not self.manualBackgroundButton.isChecked():
            return
        
        posData = self.data[self.pos_i]
        if not hasattr(posData, 'manualBackgroundImage'):
            self.initManualBackgroundImage()
        
        contours = []
        for obj in skimage.measure.regionprops(posData.manualBackgroundLab):    
            obj_contours = self.getObjContours(obj, all_external=True)  
            contours.extend(obj_contours)
            textItem = self.manualBackgroundTextItems[obj.label]
            textItem.setText(f'{obj.label}')
            self.ax1.addItem(textItem)
            yc, xc = obj.centroid
            textItem.setPos(xc, yc)
        
        cv2.drawContours(
            posData.manualBackgroundImage, contours, -1, (255, 0, 0, 200), 1
        )
        self.manualBackgroundImageItem.setImage(posData.manualBackgroundImage)

    def setManualBackgroundLab(self, load_from_store=False, debug=True):
        posData = self.data[self.pos_i]
        if posData.manualBackgroundLab is None:
            self.initManualBackgroundImage()
        
        for obj in skimage.measure.regionprops(posData.manualBackgroundLab):
            textItem = pg.TextItem(text='', color='r', anchor=(0.5, 0.5))
            if obj.label in self.manualBackgroundTextItems:
                continue
            self.manualBackgroundTextItems[obj.label] = textItem

    def setTrackedLostCentroids(self, prev_rp, tracked_lost_IDs):
        """Store centroids of those IDs the tracker decided is fine to lose 
        (e.g., upon standard cell division the ID of the mother is fine)

        Parameters
        ----------
        prev_rp : skimage.measure.RegionProperties
            List of region properties of the object in previous frame
        tracked_lost_IDs : iterable
            List-like container of the IDs that is fine to lose from previous
            frame to current frame
        
        Note
        ----
        This function stores the centroids because the user could change IDs 
        in multiple ways. Storing centroids is more robust.
        """        
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        
        for obj in prev_rp:
            if obj.label not in tracked_lost_IDs:
                continue
            
            int_centroid = tuple([int(val) for val in obj.centroid])
            try:
                posData.tracked_lost_centroids[frame_i].add(int_centroid)
            except KeyError:
                posData.tracked_lost_centroids[frame_i] = {int_centroid}  

    def trackFrame(
            self, prev_lab, prev_rp, curr_lab, curr_rp, curr_IDs,
            assign_unique_new_IDs=True, IDs=None, unique_ID=None
        ):
        if self.trackWithAcdcAction.isChecked():
            tracked_result = CellACDC_tracker.track_frame(
                prev_lab, prev_rp, curr_lab, curr_rp,
                IDs_curr_untracked=curr_IDs,
                setBrushID_func=self.setBrushID,
                posData=self.data[self.pos_i],
                assign_unique_new_IDs=assign_unique_new_IDs, 
                IDs=IDs,
                unique_ID=unique_ID
            )
        elif self.trackWithYeazAction.isChecked():
            tracked_result = self.tracking_yeaz.correspondence(
                prev_lab, curr_lab, use_modified_yeaz=True,
                use_scipy=True
            )
        else:
            tracked_result = self.trackFrameCustomTracker(
                prev_lab, curr_lab, IDs=IDs, unique_ID=unique_ID
            )

        # Check if tracker also returns additional info
        if isinstance(tracked_result, tuple):
            tracked_lab, tracked_lost_IDs = tracked_result
            self.handleAdditionalInfoRealTimeTracker(prev_rp, tracked_lost_IDs)
        else:
            tracked_lab = tracked_result
        
        return tracked_lab

    def trackFrameCustomTracker(
            self, prev_lab, currentLab, IDs=None, unique_ID=None
        ):
        if unique_ID is None:
            unique_ID = self.setBrushID()
        try:
            tracked_result = self.realTimeTracker.track_frame(
                prev_lab, currentLab,
                unique_ID=unique_ID,
                IDs=IDs,
                **self.track_frame_params,
            )
        except TypeError as err:
            if str(err).find('an unexpected keyword argument \'unique_ID\'') != -1:
                try:
                    tracked_result = self.realTimeTracker.track_frame(
                        prev_lab, currentLab, IDs=IDs,
                        **self.track_frame_params
                    )
                except TypeError as err:
                    if str(err).find('an unexpected keyword argument \'IDs\'') != -1:
                        tracked_result = self.realTimeTracker.track_frame(
                            prev_lab, currentLab,
                            **self.track_frame_params)
                    else:
                        raise err
            elif str(err).find('an unexpected keyword argument \'IDs\'') != -1:
                try:
                    tracked_result = self.realTimeTracker.track_frame(
                        prev_lab, currentLab,
                        unique_ID=unique_ID,
                        **self.track_frame_params
                    )
                except TypeError as err:
                    if str(err).find('an unexpected keyword argument \'unique_ID\'') != -1:
                        tracked_result = self.realTimeTracker.track_frame(
                            prev_lab, currentLab,
                            **self.track_frame_params
                        )
                    else:
                        raise err
            else:
                raise err
        return tracked_result

    def trackManuallyAddedObject(
            self, added_IDs: List[int] | int | Set[int], isNewID: bool,
            wl_update:bool=True, wl_track_og_curr:bool=False
        ):
        """Track object added manually on frame that was already visited.

        Parameters
        ----------
        added_IDs : int | list of int | set
            ID or IDs of the object added manually
        isNewID : bool
            If True, the added object is new
        
        Notes
        -----
        This method tracks the new added object against the previous frame 
        labels. If the ID determined by tracking is different from `added_ID` 
        (meaning that tracking thinks the new ID should be changed to the 
        tracked ID) and the tracked ID is not already existing (which would 
        otherwise causing merging) we assign the tracked ID to the object with 
        `added_ID`. 
        
        If instead the tracked ID is the same as `added_ID` we are dealing 
        with a truly new object. In this case we want to try tracking it against 
        the next frame (since the next frame was already validated). 
        As before, we assign the tracked ID (against the next frame) only if 
        not already existing in current frame (to avoid merging).    
        """        
        if self.isSnapshot:
            return 
        
        if not isNewID:
            return

        if isinstance(added_IDs, int):
            added_IDs = [added_IDs]
        
        posData = self.data[self.pos_i]
        tracked_lab = self.tracking(
            enforce=True, assign_unique_new_IDs=False, return_lab=True,
            IDs=added_IDs
        )
        self.clearAssignedObjsSecondStep()
        if tracked_lab is None:
            return
        
        # Track only new object
        prevIDs = posData.allData_li[posData.frame_i-1]['IDs']

        # mask = np.zeros(posData.lab.shape, dtype=bool)
        update_rp = False

        for added_ID in added_IDs:
            # try:
            #     obj = posData.rp[added_ID] # ID not present
            #     mask[obj.slice][obj.image] = True

            # except IndexError as err:
            mask = posData.lab == added_ID
            try:
                trackedID = tracked_lab[mask][0]
            except IndexError as err:
                # added_ID is not present
                continue 
            
            isTrackedIDalreadyPresentAndNotNew = (
                posData.IDs_idxs.get(trackedID) is not None
                and added_ID != trackedID
            )
            if isTrackedIDalreadyPresentAndNotNew:
                continue
            
            isTrackedIDinPrevIDs = trackedID in prevIDs
            if isTrackedIDinPrevIDs:
                posData.lab[mask] = trackedID
            else:
                # New object where we can try to track against next frame
                trackedID = self.trackNewIDtoNewIDsFutureFrame(added_ID, mask)
                if trackedID is None:
                    self.clearAssignedObjsSecondStep()
                    continue
                posData.lab[mask] = trackedID
        
            self.keepOnlyNewIDAssignedObjsSecondStep(trackedID)
            update_rp = True
        
        if update_rp:
            self.update_rp(wl_update=wl_update)

    def trackNewIDtoNewIDsFutureFrame(self, newID, newIDmask):
        posData = self.data[self.pos_i]
        try:
            nextLab = posData.allData_li[posData.frame_i+1]['labels']
        except IndexError:
            # This is last frame --> there are no future frames
            return
        
        if nextLab is None:
            return
        
        newID_lab = np.zeros_like(posData.lab)
        newID_lab[newIDmask] = newID
        newLab_rp = [posData.rp[posData.IDs_idxs[newID]]]
        newLab_IDs = [newID]        
        nextRp = posData.allData_li[posData.frame_i+1]['regionprops']
        
        tracked_lab = self.trackFrame(
            nextLab, nextRp, newID_lab, newLab_rp, newLab_IDs,
            assign_unique_new_IDs=False
        )
        trackedID = tracked_lab[newID_lab>0][0]
        if trackedID == newID:
            # Object does not exist in future frame --> do not track
            return
        
        if posData.IDs_idxs.get(trackedID) is not None:
            # Tracked ID already exists --> do not track to avoid merging
            return
                
        return trackedID

    def trackSubsetIDs(self, subsetIDs: Iterable[int]):
        posData = self.data[self.pos_i]
        if posData.frame_i == 0:
            return

        subsetLab = np.zeros_like(posData.lab)
        for subsetID in subsetIDs:
            subsetLab[posData.lab == subsetID] = subsetID
        
        prev_lab = posData.allData_li[posData.frame_i-1]['labels']
        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        tracked_lab = self.trackFrame(
            prev_lab, prev_rp, posData.lab, posData.rp, posData.IDs,
            assign_unique_new_IDs=True
        )
        doUpdateRp = False
        for subsetID in subsetIDs:
            subsetIDmask = posData.lab == subsetID
            trackedID = tracked_lab[subsetIDmask][0]
            if trackedID == subsetID:
                continue
            
            is_manually_edited = False
            for y, x, new_ID in posData.editID_info:
                if new_ID == subsetID:
                    # Do not track because it was manually edited
                    break
                
            posData.lab[subsetIDmask] = tracked_lab[subsetIDmask]
            doUpdateRp = True
        
        if not doUpdateRp:
            return
        
        self.update_rp()

    def tracking(
            self, enforce=False, DoManualEdit=True,
            storeUndo=False, prev_lab=None, prev_rp=None,
            return_lab=False, assign_unique_new_IDs=True,
            separateByLabel=True, wl_update=True,
            IDs=None, against_next=False,
        ):
        posData = self.data[self.pos_i]
        
        if self.doSkipTracking(against_next, enforce):
            self.setLostNewOldPrevIDs()
            return
        
        """Tracking starts here"""
        staturBarLabelText = self.statusBarLabel.text()
        self.statusBarLabel.setText('Tracking...')

        if storeUndo:
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)

        # First separate by labelling
        if separateByLabel:
            maxID = max(posData.IDs, default=1)
            setRp = core.split_connected_components(
                posData.lab, rp=posData.rp, max_ID=maxID
            )
            if setRp:
                self.update_rp(wl_update=wl_update, )

        if prev_lab is None:
            if not against_next:
                prev_lab = posData.allData_li[posData.frame_i-1]['labels']
            else:
                prev_lab = posData.allData_li[posData.frame_i+1]['labels']
        if prev_rp is None:
            if not against_next:
                prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
            else:
                prev_rp = posData.allData_li[posData.frame_i+1]['regionprops']
        
        unique_ID = None
        if posData.frame_i < self.get_last_tracked_i():
            unique_ID = self.setBrushID(return_val=True)
        
        tracked_lab = self.trackFrame(
            prev_lab, prev_rp, posData.lab, posData.rp, posData.IDs,
            assign_unique_new_IDs=assign_unique_new_IDs, IDs=IDs,
            unique_ID=unique_ID
        )
        
        if DoManualEdit:
            # Correct tracking with manually changed IDs
            rp = skimage.measure.regionprops(tracked_lab)
            IDs = [obj.label for obj in rp]
            self.manuallyEditTracking(tracked_lab, IDs)

        if return_lab:
            QTimer.singleShot(50, partial(
                self.statusBarLabel.setText, staturBarLabelText
            ))
            return tracked_lab
        
        # Update labels, regionprops and determine new and lost IDs
        posData.lab = tracked_lab
        self.update_rp(wl_update=wl_update, )
        self.setAllTextAnnotations()
        QTimer.singleShot(50, partial(
            self.statusBarLabel.setText, staturBarLabelText
        ))

    def updateAssignedObjsAcdcTrackerSecondStep(self, newID):
        posData = self.data[self.pos_i]
        annotInfo = posData.acdcTracker2stepsAnnotInfo.get(posData.frame_i)
        if annotInfo is None:
            return
        
        new_objs_1st_step, lost_objs_1st_step = annotInfo
        correct_new_objs, correct_lost_objs = [], []
        for lostObj, newObj in zip(lost_objs_1st_step, new_objs_1st_step):
            newObj_ID = posData.lab[newObj.slice][newObj.image][0]
            if newObj_ID == newID:
                # The ID of the new object tracked with 2nd step was 
                # manually edit --> do not annotate its linking to lost obj anymore
                continue
            correct_new_objs.append(newObj)
            correct_lost_objs.append(lostObj)
        
        if not correct_new_objs:
            posData.acdcTracker2stepsAnnotInfo[posData.frame_i] = None
        else:
            posData.acdcTracker2stepsAnnotInfo[posData.frame_i] = (
                correct_new_objs, correct_lost_objs
            )
        self.annotateAssignedObjsAcdcTrackerSecondStep()

    def updateGhostMaskOpacity(self, alpha_percentage=None):
        if alpha_percentage is None:
            alpha_percentage = (
                self.manualTrackingToolbar.ghostMaskOpacitySpinbox.value()
            )
        alpha = alpha_percentage/100
        self.ghostMaskItemLeft.setOpacity(alpha)
        self.ghostMaskItemRight.setOpacity(alpha)

    def updateLastCheckedFrameWidgets(self, last_tracked_i):
        self.navigateScrollBar.setMaximum(last_tracked_i+1)
        self.navSpinBox.setMaximum(last_tracked_i+1)
        self.lastTrackedFrameLabel.setText(
            f'Last checked frame n. = {last_tracked_i+1}'
        )

    def warnRepeatTrackingVideoOnVisitedFrames(self, last_tracked_i, start_n):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            'You are repeating tracking on frames that <b>have already '
            'been visited/tracked before</b>.<br><br>'
            'This will very likely make the <b>annotations wrong</b>.<br><br>'
            'If you really want to repeat tracking on the frames before '
            f'{last_tracked_i+1} the <b>annotations from frame '
            f'{start_n} to frame {last_tracked_i+1} '
            'will be removed</b>.<br><br>'
            'Do you want to continue?'
        )
        noButton, yesButton = msg.warning(
            self, 'Repating tracking with annotations!', txt,
            buttonsTexts=(
                '  No, stop tracking and keep annotations.',
                '  Yes, repeat tracking and DELETE annotations.' 
            )
        )
        if msg.cancel:
            return False

        if msg.clickedButton == noButton:
            return False
        else:
            return True

    def warnRepeatTrackingVideoWithAnnotations(self, last_tracked_i, start_n):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            'You are repeating tracking on frames that <b>have cell cycle '
            'annotations</b>.<br><br>'
            'This will very likely make the <b>annotations wrong</b>.<br><br>'
            'If you really want to repeat tracking on the frames before '
            f'{last_tracked_i+1} the <b>annotations from frame '
            f'{start_n} to frame {last_tracked_i+1} '
            'will be removed</b>.<br><br>'
            'Do you want to continue?'
        )
        noButton, yesButton = msg.warning(
            self, 'Repating tracking with annotations!', txt,
            buttonsTexts=(
                '  No, stop tracking and keep annotations.',
                '  Yes, repeat tracking and DELETE annotations.' 
            )
        )
        if msg.cancel:
            return False

        if msg.clickedButton == noButton:
            return False
        else:
            return True

    def warnTrackerInputNotValid(self, trackerName, warningText):
        msg = widgets.myMessageBox(wrapText=False)
        txt = warningText.replace('\n', '<br>')
        txt = html_utils.paragraph(
            f'{txt}<br><br>'
            'Tracking process will be cancelled. Thank you for your patience!'
        )
        msg.warning(self, 'Invalid input for tracker', txt)
