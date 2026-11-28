"""Tracking in the main GUI: real-time and repeat tracking, lost and new
objects, manual tracking ghost tool. Mixins of guiWin
(gui/main_window/window.py) only: the measurements viewer never composes
them.
"""
import inspect
from collections import Counter
from functools import partial
from typing import Iterable, List, Set
import cv2
import numpy as np
from qtpy.QtCore import Qt, QEventLoop, QTimer, QThread
from qtpy.QtWidgets import QCheckBox
from cellacdc import exception_handler, disableWindow
from cellacdc import html_utils
from cellacdc.shared.dialogs import generic as dialogs_generic
from ..workers import segmentation_tracking as gui_workers_segmentation_tracking
from cellacdc import core
from cellacdc.myutils import segm_models
from .. import helpers as gui_helpers
from cellacdc.shared.widgets import (
    message_box as widgets_message_box,
    push_buttons as widgets_push_buttons,
)
from ..dialogs import segmentation as gui_dialogs_segmentation
from cellacdc import qutils
from cellacdc import core_split_IDs


class TrackingGuiElements:
    """Real-time and repeat tracking, the second step of the trackers and
    tracked lost centroids.
    """

    def rtTrackerActionToggled(self, checked):
        if not checked:
            return

        aliases = segm_models.aliases_real_time_trackers(reverse=True)
        if self.sender().text() in aliases:
            trackingAlgo = aliases[self.sender().text()]
        else:
            trackingAlgo = self.sender().text()
        self.df_settings.at['tracking_algorithm', 'value'] = trackingAlgo
        self.df_settings.to_csv(self.settings_csv_path)

        if self.sender().text() == 'YeaZ':
            msg = widgets_message_box.myMessageBox(wrapText=False)
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

    @exception_handler
    def trackingWorkerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.logger.info('Worker process ended.')
        askDisableRealTimeTracking = (
            self.trackingWorker.trackingOnNeverVisitedFrames
            and self.realTimeTrackingToggle.isChecked()
        )
        if askDisableRealTimeTracking:
            msg = widgets_message_box.myMessageBox()
            title = 'Disable real-time tracking?'
            txt = (
                'You perfomed tracking on frames that you have '
                '<b>never visited.</b><br><br>'
                'Cell-ACDC default behaviour is to <b>track them again</b> when you '
                'will visit them.<br><br>'
                'However, you can <b>overwrite this behaviour</b> and explicitly '
                'disable tracking for all of the frames you already tracked.<br><br>'
                'NOTE: you can reactivate real-time tracking by clicking on the '
                '"Reset last segmented frame" button on the top toolbar.<br><br>'
                'What do you want me to do?'
            )
            _, disableTrackingButton = msg.information(
                self, title, html_utils.paragraph(txt),
                buttonsTexts=(
                    'Keep real-time tracking active (recommended)',
                    'Disable real-time tracking'
                )
            )
            if msg.clickedButton == disableTrackingButton:
                self.logger.info('Disabling real time tracking...')
                self.realTimeTrackingToggle.setChecked(False)

        posData = self.data[self.pos_i]
        self.updateAllImages()
        self.titleLabel.setText('Done', color='w')

    def startTrackingWorker(self, posData, video_to_track):
        self.thread = QThread()
        self.trackingWorker = gui_workers_segmentation_tracking.trackingWorker(
            posData, self, video_to_track
        )
        self.trackingWorker.moveToThread(self.thread)
        self.trackingWorker.finished.connect(self.thread.quit)
        self.trackingWorker.finished.connect(self.trackingWorker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.trackingWorker.signals.progress = self.trackingWorker.progress
        self.trackingWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.trackingWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.trackingWorker.signals.sigInitInnerPbar.connect(
            self.workerInitInnerPbar
        )
        self.trackingWorker.progress.connect(self.workerProgress)
        self.trackingWorker.critical.connect(self.workerCritical)
        self.trackingWorker.finished.connect(self.trackingWorkerFinished)

        self.trackingWorker.debug.connect(self.workerDebug)

        self.thread.started.connect(self.trackingWorker.run)
        self.thread.start()

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
            msg = widgets_message_box.myMessageBox(showCentered=False, wrapText=False)
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

    @exception_handler
    def repeatTrackingVideo(self, checked=False):
        posData = self.data[self.pos_i]
        win = gui_dialogs_segmentation.selectTrackerGUI(
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
        self.tracker, self.track_params, init_params = segm_models.init_tracker(
            posData, trackerName, qparent=self, return_init_params=True
        )
        if self.track_params is None:
            self.logger.info('Tracking aborted.')
            return
        
        warningText = gui_helpers.validate_tracker_input(
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

        proceed = self.askRemoveAnnotationsRepeatTrackingVideo(start_n)
        if not proceed:
            self.logger.info('Tracking aborted.')
            return

        self.start_n = start_n
        self.stop_n = stop_n
        
        info_txt = f'Tracking from frame n. {start_n} to {stop_n}...'
        self.logger.info(info_txt)

        self.progressWin = dialogs_generic.QDialogWorkerProgress(
            title='Tracking', parent=self, pbarDesc=info_txt
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(stop_n-start_n)
        self.startTrackingWorker(posData, video_to_track)

    def askRemoveAnnotationsRepeatTrackingVideo(self, start_n):
        """Before repeating tracking on the video from frame n. `start_n`,
        warn if the re-tracked frames have cell cycle or lineage tree
        annotations and, if the user continues, remove them from frame n.
        `start_n` (index `start_n-1`) to the end.

        The first re-tracked frame counts as changed although it gets the
        IDs it had before tracking back
        (`trackingWorker._relabel_first_frame_labels`): trackers that
        output annotations (normal division lineage tree, Trackastra,
        DeepSea) write their own table into it, where every object starts
        a new lineage, and their tables without cell cycle columns cannot
        be stored into a frame that kept them (`store_cca_df`).

        Parameters
        ----------
        start_n : int
            First re-tracked frame (1-based).

        Returns
        -------
        bool
            False if the user stopped tracking.
        """
        last_cca_i, last_lin_tree_i = self.getLastAnnotatedFrames()
        first_changed_i = start_n - 1
        remove_cca = last_cca_i >= first_changed_i
        remove_lin_tree = last_lin_tree_i >= first_changed_i
        if not remove_cca and not remove_lin_tree:
            return True

        annot_names = []
        if remove_cca:
            annot_names.append('cell cycle')
        if remove_lin_tree:
            annot_names.append('lineage tree')
        # `warnRepeatTrackingVideoWithAnnotations` takes 1-based frames
        proceed = self.warnRepeatTrackingVideoWithAnnotations(
            max(last_cca_i, last_lin_tree_i), first_changed_i+1,
            annotations=f'{" and ".join(annot_names)} annotations'
        )
        if not proceed:
            return False

        self.logger.info(
            f'Removing annotations from frame n. {first_changed_i+1}.'
        )
        if remove_cca:
            self.resetCcaFuture(first_changed_i)
        if remove_lin_tree:
            self.resetLin_tree_future(from_frame_i=first_changed_i)
        return True

    def warnTrackerInputNotValid(self, trackerName, warningText):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = warningText.replace('\n', '<br>')
        txt = html_utils.paragraph(
            f'{txt}<br><br>'
            'Tracking process will be cancelled. Thank you for your patience!'
        )
        msg.warning(self, 'Invalid input for tracker', txt)

    def repeatTracking(self):
        self.logger.info('Tracking current frame...')
        posData = self.data[self.pos_i]
        tracked_lab, assignments = self.tracking(
            enforce=True, 
            DoManualEdit=False, 
            return_assignments=True, 
            return_lab=True
        )
        posData.lab = tracked_lab
        if posData.editID_info:
            lab2D = self.get_2Dlab(posData.lab)
            editedIDsInfo = {
                lab2D[y,x]:newID 
                for y, x, newID in posData.editID_info
                if lab2D[y,x] != newID
            }
            editedIDsInfoItems = [
                f'ID {oldID} --> {newID}'
                for oldID, newID in editedIDsInfo.items()
            ]
            editIDul = html_utils.to_list(editedIDsInfoItems)
            msg = widgets_message_box.myMessageBox()
            txt = html_utils.paragraph(f"""
                You requested to repeat tracking but <b>there are manually 
                edited IDs</b> (see edited IDs in the details section below)
                <br><br>
                Do you want to keep these edits or ignore them?
            """)
            keepManualEditButton = widgets_push_buttons.okPushButton(
                'Keep manually edited IDs'
            )
            ignoreButton = widgets_push_buttons.noPushButton(
                'Ignore manually edited IDs'
            )
            msg.question(
                self, 'Repeat tracking mode', txt, 
                buttonsTexts=(keepManualEditButton, ignoreButton), 
                detailsText=editIDul
            )
            if msg.cancel:
                self.update_rp(assignments=assignments) # rp now stale as we return img
                return
            if msg.clickedButton == keepManualEditButton:
                allIDs = posData.rp.IDs
                lab2D = self.get_2Dlab(posData.lab)
                tracked_lab, assignments = self.manuallyEditTracking(
                    lab2D, assignments) # here not use tracked lab?
                self.update_rp(assignments=assignments) # rp now stale as we return img
                self.setAllTextAnnotations()
                self.highlightLostNew()
                # self.checkIDsMultiContour()
            else:
                self.update_rp(assignments=assignments) # rp now stale as we return img
                posData.editID_info = []
        else:
            self.update_rp(assignments=assignments)
        
        # filter self assignments
        assignments = {k: v for k, v in assignments.items() if k != v}
        if assignments:
            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Repeat tracking')
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df('Repeat tracking')
        else:
            self.updateAllImages()

    def trackNewIDtoNewIDsFutureFrame(
        self, newID, obj, assignments, 
        clearAssignedObjsSecondStep=False
        ):
        # here RP is stale
        posData = self.data[self.pos_i]
        try:
            nextLab = posData.allData_li[posData.frame_i+1]['labels']
        except IndexError:
            # This is last frame --> there are no future frames
            return None, assignments
        
        if nextLab is None:
            return None, assignments
        
        if obj is None:
            return None, assignments
             
        
        nextRp = posData.allData_li[posData.frame_i+1]['regionprops']
        nextLab = posData.allData_li[posData.frame_i+1]['labels']
        reverse_assignments = {v:k for k, v in assignments.items()}
        
        rp = posData.rp
        lab = posData.lab
        
        # make rp remporarliy not stale anymore
        rp.update_regionprops_via_assignments(assignments, lab) 
        assignments_new = self.trackFrame(
            nextLab, nextRp, lab, rp, rp.IDs,
            assign_unique_new_IDs=False, return_assignments=True,
            specific_IDs=[newID], dont_return_tracked_lab=True,
            clearAssignedObjsSecondStep=clearAssignedObjsSecondStep
        )
        # restore rp
        posData.rp.update_regionprops_via_assignments(reverse_assignments, lab)
        
        # clear self assignments
        assignments_new = {
            k:v for k, v in assignments_new.items() if k != v
        }
        if not assignments_new:
            return None, assignments
        
        trackedIDs = list(assignments_new.values())

        trackedID = trackedIDs[0]
        if trackedID == newID:
            # Object does not exist in future frame --> do not track
            return None, assignments
        
        if posData.rp.get_obj_from_ID(trackedID, warn=False) is not None:
            # Tracked ID already exists --> do not track to avoid merging
            return None, assignments
        
        # update assignments
        assignments = {
            old_ID: _tracked_ID for old_ID, _tracked_ID in assignments.items()
            if old_ID != newID
        }
        assignments[newID] = trackedID
        
        return trackedID, assignments

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

    def warnRepeatTrackingVideoOnVisitedFrames(self, last_tracked_i, start_n):
        msg = widgets_message_box.myMessageBox()
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

    def warnRepeatTrackingVideoWithAnnotations(
            self, last_tracked_i, start_n,
            annotations='cell cycle annotations'
        ):
        msg = widgets_message_box.myMessageBox()
        txt = html_utils.paragraph(
            'You are repeating tracking on frames that <b>have '
            f'{annotations}</b>.<br><br>'
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

    def trackManuallyAddedObject(
            self, added_IDs: List[int] | int | Set[int], isNewID: bool,
            wl_update:bool=True, wl_track_og_curr:bool=False,
            clearAssignedObjsSecondStep=False,
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
            return dict()
        
        if not isNewID:
            return dict()

        posData = self.data[self.pos_i]
        if posData.frame_i == 0:
            return dict()

        if isinstance(added_IDs, int):
            added_IDs = [added_IDs]
        
        tracked_lab, assignments = self.tracking(
            enforce=True, assign_unique_new_IDs=False, return_lab=True,
            specific_IDs=added_IDs, return_assignments=True,
            against_next=posData.frame_i==0,
            clearAssignedObjsSecondStep=clearAssignedObjsSecondStep,
            
        )
        
        # RP not updated after tracking!!!
        if clearAssignedObjsSecondStep:
            self.clearAssignedObjsSecondStep()
            
        if tracked_lab is None:
            return dict()
        
        # Track only new object
        prevIDs = posData.allData_li[posData.frame_i-1]['regionprops'].IDs
        for added_ID in added_IDs:
            
            # check if added ID is already present
            # here PR is "stale" so ID maps are not tracked
            obj = posData.rp.get_obj_from_ID(added_ID, warn=False)
            if obj is None:
                continue
            try:
                trackedID = tracked_lab[obj.slice][obj.image][0]
            except IndexError as err:
                # added_ID is not present
                continue 
            
            isTrackedIDalreadyPresentAndNotNew = (
                posData.rp.ID_to_idx.get(trackedID) is not None
                and added_ID != trackedID
            )
            if isTrackedIDalreadyPresentAndNotNew:
                # self.updatePointsLayerClickEntryTableEndname(
                #     'added obj already present', added_ID, trackedID
                # )
                continue
            
            isTrackedIDinPrevIDs = trackedID in prevIDs
            if isTrackedIDinPrevIDs:
                posData.lab[obj.slice][obj.image] = trackedID
            else:
                # New object where we can try to track against next frame
                trackedID, assignments = self.trackNewIDtoNewIDsFutureFrame(
                    added_ID, obj, assignments,
                    clearAssignedObjsSecondStep=clearAssignedObjsSecondStep
                    )
                if trackedID is None:
                    if clearAssignedObjsSecondStep:
                        self.clearAssignedObjsSecondStep()
                    # clean assignment
                    assignments.pop(added_ID, None)
                    continue
                posData.lab[obj.slice][obj.image] = trackedID
        
            self.keepOnlyNewIDAssignedObjsSecondStep(trackedID)
        self.update_rp(wl_update=wl_update, assignments=assignments)
        return assignments

    def trackFrameCustomTracker(
            self, prev_lab, curr_lab, prev_rp, curr_rp, specific_IDs=None, unique_ID=None,
            return_assignments=True, dont_return_tracked_lab=False,
            
        ):
        if unique_ID is None:
            unique_ID = self.setBrushID()
        
        if self.track_frame_params is None:
            self.initRealTimeTracker()
        
        kwargs_total = {
            'unique_ID': unique_ID,
            'return_assignments': return_assignments,
            'dont_return_tracked_lab': dont_return_tracked_lab,
            'specific_IDs': specific_IDs
        }
        kwargs_total.update(self.track_frame_params)
        
        kwargs = {k: v for k, v in kwargs_total.items() if k in self.realTimeTracker_kwargs}
        tracked_result = self.realTimeTracker.track_frame(
            prev_lab, curr_lab,
            **kwargs,
        )
        return tracked_result

    def trackFrame(
            self, prev_lab, prev_rp, curr_lab, curr_rp, curr_IDs,
            assign_unique_new_IDs=True, specific_IDs=None, unique_ID=None,
            dont_return_tracked_lab=False, return_assignments=False,
            clearAssignedObjsSecondStep=False
        ):
        from cellacdc.trackers.CellACDC import CellACDC_tracker
        
        if self.trackWithAcdcAction.isChecked():
            tracked_result = CellACDC_tracker.track_frame(
                prev_lab, prev_rp, curr_lab, curr_rp,
                IDs_curr_untracked=curr_IDs,
                setBrushID_func=self.setBrushID,
                posData=self.data[self.pos_i],
                assign_unique_new_IDs=assign_unique_new_IDs, 
                specific_IDs=specific_IDs,
                unique_ID=unique_ID,
                return_assignments=return_assignments,
                dont_return_tracked_lab=dont_return_tracked_lab
            )
        elif self.trackWithYeazAction.isChecked():
            tracked_result = self.tracking_yeaz.correspondence(
                prev_lab, curr_lab, use_modified_yeaz=True,
                use_scipy=True
            )
        else:
            tracked_result = self.trackFrameCustomTracker(
                prev_lab, curr_lab, prev_rp, curr_rp, specific_IDs=specific_IDs, unique_ID=unique_ID,
                dont_return_tracked_lab=dont_return_tracked_lab, return_assignments=return_assignments
            )

        # Check if tracker also returns additional info
        assignments = None
        if isinstance(tracked_result, tuple):
            tracked_lab, add_info = tracked_result
            assignments = self.handleAdditionalInfoRealTimeTracker(
                prev_rp, add_info, 
                clearAssignedObjsSecondStep=clearAssignedObjsSecondStep
                )
        elif isinstance(tracked_result, dict) and dont_return_tracked_lab:
            add_info = tracked_result
            if 'assignments' in add_info: # if still entire add_info is returned
                assignments = self.handleAdditionalInfoRealTimeTracker(
                    prev_rp, add_info, 
                    clearAssignedObjsSecondStep=clearAssignedObjsSecondStep
                    )
            else:
                assignments = add_info # its just assignements
        else:
            tracked_lab = tracked_result
        
        if not return_assignments and not dont_return_tracked_lab:
            return tracked_lab

        # get assignments
        if assignments is None:
            assignments = dict()
            for obj in curr_rp:
                try:
                    old_lab = obj.label
                    new_lab = tracked_lab[obj.slice][obj.image][0]
                    assignments[old_lab] = new_lab
                except:
                    import pdb; pdb.set_trace()

        if dont_return_tracked_lab:
            return assignments
        return tracked_lab, assignments

    def clearAssignedObjsSecondStep(self, force=False):
        checked = self.autoClearAssignedObjectsSecondStepAction.isChecked()
        if checked or force:
            posData = self.data[self.pos_i]
            posData.acdcTracker2stepsAnnotInfo[posData.frame_i] = None

    def trackSubsetIDs(self, subsetIDs: Iterable[int]):
        posData = self.data[self.pos_i]
        if posData.frame_i == 0:
            return
        
        prev_lab = posData.allData_li[posData.frame_i-1]['labels']
        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        assignments = self.trackFrame(
            prev_lab, prev_rp, posData.lab, posData.rp, posData.IDs,
            assign_unique_new_IDs=True, specific_IDs=subsetIDs,
            dont_return_tracked_lab=True, return_assignments=True
        )
        # I think assignments already avoids merging
        assignments_new = dict()
        for old_ID, new_ID in assignments.items():
            # get "old" id based on assignments
            if old_ID == new_ID:
                continue # nothing to do

            for y, x, editID in posData.editID_info:
                if editID == old_ID or editID == new_ID:
                    # Do not track because it was manually edited
                    continue
                
            
            obj = posData.rp.get_obj_from_ID(old_ID) # pr is still old, so we need to get the old ID
            posData.lab[obj.slice][obj.image] = new_ID
            assignments_new[old_ID] = new_ID # old ID : new tracked ID
        
        self.update_rp(assignments=assignments_new)

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


    # @exec_time
    @exception_handler
    def tracking(
            self, enforce=False, DoManualEdit=True,
            storeUndo=False, prev_lab=None, prev_rp=None,
            return_lab=False, assign_unique_new_IDs=True,
            separateByLabel=True, wl_update=True,
            against_next=False, specific_IDs=None , return_assignments=False,
            clearAssignedObjsSecondStep=False,
        ):
        posData = self.data[self.pos_i]
        return_tuple = (
            (None, None) if return_assignments and return_lab else None
        )
        if self.doSkipTracking(against_next, enforce):
            self.setLostNewOldPrevIDs()
            return return_tuple
        
        """Tracking starts here"""
        staturBarLabelText = self.statusBarLabel.text()
        self.statusBarLabel.setText('Tracking...')

        if storeUndo:
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)

        # First separate by labelling
        if separateByLabel:
            maxID = max(posData.IDs, default=1)
            setRp = core_split_IDs.split_connected_components(
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
        
        tracked_lab, assignments = self.trackFrame(
            prev_lab, prev_rp, posData.lab, posData.rp, posData.IDs,
            assign_unique_new_IDs=assign_unique_new_IDs,
            unique_ID=unique_ID, specific_IDs=specific_IDs,
            return_assignments=True, 
            clearAssignedObjsSecondStep=clearAssignedObjsSecondStep,
        )
        
        # debugutils.print_call_stack()
        # print(f'args: '
        #       f'enforce={enforce}, DoManualEdit={DoManualEdit}, storeUndo={storeUndo}, '
        #       f'prev_lab={prev_lab is not None}, prev_rp={prev_rp is not None}, '
        #       f'return_lab={return_lab}, assign_unique_new_IDs={assign_unique_new_IDs},'
        #       f'against_next={against_next}, specific_IDs={specific_IDs}, return_assignments={return_assignments}')
        # print(f'Tracking done. Assignments: {assignments}')
        if DoManualEdit:
            # Correct tracking with manually changed IDs
            tracked_lab, assignments = self.manuallyEditTracking(tracked_lab, assignments)

        if return_lab:
            QTimer.singleShot(50, partial(
                self.statusBarLabel.setText, staturBarLabelText
            ))
            if return_assignments:
                return tracked_lab, assignments
            return tracked_lab
        
        # Update labels, regionprops and determine new and lost IDs
        posData.lab = tracked_lab
        self.update_rp(wl_update=wl_update, assignments=assignments)
        self.setAllTextAnnotations()
        QTimer.singleShot(50, partial(
            self.statusBarLabel.setText, staturBarLabelText
        ))
        if return_assignments and return_lab:
            return tracked_lab, assignments
        elif return_assignments:
            return assignments
        elif return_lab:
            return tracked_lab

    def handleAdditionalInfoRealTimeTracker(
        self, prev_rp, add_info,
        clearAssignedObjsSecondStep=False
        ):
        assignments = None
        posData = self.data[self.pos_i]
        acdcTracker2stepsAnnotInfo = posData.acdcTracker2stepsAnnotInfo[
            posData.frame_i
            ] if (posData.frame_i 
                  in posData.acdcTracker2stepsAnnotInfo
                  ) else None
        if (
            acdcTracker2stepsAnnotInfo is not None 
            ):
            new_objs_1st_step, lost_objs_1st_step = acdcTracker2stepsAnnotInfo
        else:
            new_objs_1st_step, lost_objs_1st_step = [], []
            
        if clearAssignedObjsSecondStep:
            self.clearAssignedObjsSecondStep()
            
        if self._rtTrackerName == 'CellACDC_normal_division':
            tracked_lost_IDs = add_info['mothers']
            protected_new_IDs = add_info['daughters']
            self.setTrackedLostCentroids(prev_rp, tracked_lost_IDs)
            protected_new_IDs_frame = self.protected_new_IDs.get(posData.frame_i, set())
            protected_new_IDs_frame.update(protected_new_IDs)
            self.protected_new_IDs[posData.frame_i] = protected_new_IDs_frame
            assignments = add_info['assignments']
            if add_info['to_track_tracked_objs_2nd_step'] is not None:
                new_objs_1st_step_new, lost_objs_1st_step_new = add_info[
                            'to_track_tracked_objs_2nd_step']
                new_objs_1st_step.extend(new_objs_1st_step_new)
                lost_objs_1st_step.extend(lost_objs_1st_step_new)
                
        elif self._rtTrackerName == 'CellACDC_2steps':
            assignments = add_info['assignments']
            if add_info['to_track_tracked_objs_2nd_step'] is not None:
                new_objs_1st_step_new, lost_objs_1st_step_new = add_info[
                        'to_track_tracked_objs_2nd_step']
                new_objs_1st_step.extend(new_objs_1st_step_new)
                lost_objs_1st_step.extend(lost_objs_1st_step_new)
                    
        elif self._rtTrackerName == 'Cell-ACDC':
            assignments = add_info['assignments']
            
        if new_objs_1st_step or lost_objs_1st_step:
            posData.acdcTracker2stepsAnnotInfo[posData.frame_i] = (
                new_objs_1st_step, lost_objs_1st_step
            )
            
        return assignments

    def keepOnlyNewIDAssignedObjsSecondStep(self, trackedID):
        return
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
        # self.annotateAssignedObjsAcdcTrackerSecondStep()

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


    def annotateAssignedObjsAcdcTrackerSecondStep(self):
        posData = self.data[self.pos_i]
        annotInfo = posData.acdcTracker2stepsAnnotInfo.get(posData.frame_i)
        if annotInfo is None:
            return
        
        self.ax1_trackerMovementAgainstPrevLinesItem.clear()
        self.yellowContourScatterItem.clear()
        IDs_set = posData.rp.IDs_set

        new_objs_1st_step, lost_objs_1st_step = annotInfo
        new_objs_1st_step_new, lost_objs_1st_step_new = [], []
        for lostObj, newObj in zip(lost_objs_1st_step, new_objs_1st_step):
            # guard against removed cells
            # ID is actually not present in current frame !DO not use newObj, it is stale!
            if lostObj.label not in IDs_set:
                continue
            
            ID_in_lab = posData.lab[newObj.slice][newObj.image][0] # get the current lab of the object
            if ID_in_lab != lostObj.label:
                continue
            
            new_objs_1st_step_new.append(newObj)
            lost_objs_1st_step_new.append(lostObj)

            allContours = self.getObjContours(
                lostObj,
                all_external=True,
                include_internal=self.showAllContoursToggle.isChecked()
            ) 
            for objContours in allContours:
                if not self.isObjInCurrSlice(newObj.bbox):
                    continue
                xx = objContours[:,0] + 0.5
                yy = objContours[:,1] + 0.5
                self.yellowContourScatterItem.addPoints(xx, yy)
                
            y1, x1 = self.getObjCentroid(lostObj.centroid)
            y2, x2 = self.getObjCentroid(newObj.centroid)
            xx, yy = core.get_line(y1, x1, y2, x2, dashed=False)
            self.ax1_trackerMovementAgainstPrevLinesItem.addPoints(xx, yy)
            
        # update annotInfo with filtered objects
        posData.acdcTracker2stepsAnnotInfo[posData.frame_i] = (
            new_objs_1st_step_new, lost_objs_1st_step_new
        )

    # def annotateAllObjectTracks(self, settings=None):
    #     self.annotateObjTrack(0, settings=settings)
    #     self.annotateObjTrack(1, settings=settings)

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
        prev_lab = posData.allData_li[frame_i-1]['labels']
        
        for obj in prev_rp:
            if obj.label not in tracked_lost_IDs:
                continue
            if isinstance(prev_rp, self._acdcRegionProps):
                ID = obj.label
                centroid = prev_rp.get_centroid(ID, exact=True)
            else:
                centroid = obj.centroid
            int_centroid = tuple([int(val) for val in centroid])
            # check if centroid has right ID
            if prev_lab[int_centroid] != ID:
                # get closest point with the right ID
                coords = obj.coords
                distances = np.sqrt(np.sum((coords - centroid) ** 2, axis=1))
                closest_idx = np.argmin(distances)
                int_centroid = tuple([int(val) for val in coords[closest_idx]])
            try:
                posData.tracked_lost_centroids[frame_i].add(int_centroid)
            except KeyError:
                posData.tracked_lost_centroids[frame_i] = {int_centroid}  

    def manuallyEditTracking(self, tracked_lab, assignments):
        posData = self.data[self.pos_i]
        infoToRemove = []

        if not assignments:
            return tracked_lab, assignments

        # !!! RP is stale so we need to reverse search for the ID
        reversed_assignments = (
            {tracked_id: stale_id for stale_id, tracked_id in assignments.items()}
            if assignments else {}
        )
        stale_ids = set(posData.rp.IDs)

        covered_edited_IDs = set()
        for y, x, edited_ID in posData.editID_info:
            new_ID = assignments.get(edited_ID, edited_ID) # ID in tracked lab
            if new_ID in covered_edited_IDs:
                # This ID has already been edited by sawpping for example
                continue
            
            if new_ID == 0 or new_ID == edited_ID: # edited ID is not tracked to a different ID
                infoToRemove.append((y, x, edited_ID))
                continue

            old_RP_ID = reversed_assignments.get(edited_ID, edited_ID) # ID pre tracking
            old_obj = posData.rp.get_obj_from_ID(old_RP_ID) # obj pre tracking

            if edited_ID in stale_ids:
                # a swap has been made by the user between an old ID (old_RP_ID) and a new ID (edited_ID)
                new_obj = posData.rp.get_obj_from_ID(edited_ID)
                tracked_lab[old_obj.slice][old_obj.image] = edited_ID
                tracked_lab[new_obj.slice][new_obj.image] = old_RP_ID
                # update assignemnets
                assignments[old_RP_ID] = edited_ID
                assignments[edited_ID] = old_RP_ID
                # add the two swapped IDs
                
                covered_edited_IDs.add(edited_ID)
                covered_edited_IDs.add(old_RP_ID)
                
            else:
                tracked_lab[old_obj.slice][old_obj.image] = edited_ID
                
                assignments[old_RP_ID] = edited_ID

                covered_edited_IDs.add(edited_ID)

        for info in infoToRemove:
            posData.editID_info.remove(info)
            
        return tracked_lab, assignments

    def initRealTimeTracker(self, force=False):
        for rtTrackerAction in self.trackingAlgosGroup.actions():
            if rtTrackerAction.isChecked():
                break
        
        aliases = segm_models.aliases_real_time_trackers(reverse=True)
        
        rtTracker = rtTrackerAction.text()
        rtTracker_txt = rtTracker

        if rtTracker in aliases:
            rtTracker = aliases[rtTracker]
        
        if rtTracker == 'Cell-ACDC':
            self._rtTrackerName = 'Cell-ACDC'
            self.realTimeTracker_kwargs = None # This is hard coded
            return
        if rtTracker == 'YeaZ':
            self._rtTrackerName = 'YeaZ'
            self.realTimeTracker_kwargs = None # This is hard coded
            return
        
        if self.isRealTimeTrackerInitialized and not force:
            return
        
        self.logger.info(f'Initializing {rtTracker_txt} tracker...')
        self._rtTrackerName = rtTracker
        posData = self.data[self.pos_i]
        realTimeTracker, track_frame_params = segm_models.init_tracker(
            posData, rtTracker, qparent=self, realTime=True
        )
        if realTimeTracker is None:
            self.logger.info(f'{rtTracker} tracker initialization cancelled.')
            return
        
        self.realTimeTracker = realTimeTracker
        self.track_frame_params = track_frame_params
        self.realTimeTracker_kwargs = inspect.signature(
            self.realTimeTracker.track_frame).parameters
        self.logger.info(f'{rtTracker} tracker successfully initialized.')
        if 'image_channel_name' in self.track_frame_params:
            # Remove the channel name since it was already loaded in init_tracker
            del self.track_frame_params['image_channel_name']

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


class LostObjectsGuiElements:
    """Lost and new objects: contours display, copy of lost objects, go to
    lost or new ID.
    """

    def goToLostObjectID(self, lostID, color=(255, 165, 0, 255)):
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        prev_rp = posData.allData_li[frame_i-1]['regionprops']
        obj = prev_rp.get_obj_from_ID(lostID)
        self.goToZsliceSearchedID(obj)
        
        imageItem = self.getLostObjImageItem(0)
        thickness = 1
        if not hasattr(self, 'lostObjContoursImage'):
            self.initLostObjContoursImage()
        else:
            self.lostObjContoursImage[:] = 0  

        display_rp = self.get2DRP(frame_i=posData.frame_i -1)
        obj = display_rp.get_obj_from_ID(lostID)
        contours = self.getObjContours(
            obj,
            all_external=True,
            include_internal=self.showAllContoursToggle.isChecked()
        )
        
        self.addLostObjsToLostObjImage(obj, lostID)
        self.drawLostObjContoursImage(
            imageItem, contours, thickness=2, color=color
        )
        
        is_obj_visible = self.isObjVisibleViewRange(obj.bbox)
        if not is_obj_visible:
            centroid = prev_rp.get_centroid(obj.label)
            yc, xc = self.getObjCentroid(centroid)
            pos = (int(xc), int(yc))
            # set center of view to the lost object's position
            self.ax1.setCenter(pos)        

    def goToAcceptedLostObjectID(self, acceptedLostID):
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        prev_rp = posData.allData_li[frame_i-1]['regionprops']
        obj = prev_rp.get_obj_from_ID(acceptedLostID)
        self.goToZsliceSearchedID(obj)
        
        for ax in (self.ax1, self.ax2):
            self.updateLostTrackedContoursImage(ax=ax, tracked_lost_IDs=[acceptedLostID])
            
        is_obj_visible = self.isObjVisibleViewRange(obj.bbox)
        if not is_obj_visible:
            centroid = prev_rp.get_centroid(obj.label)
            yc, xc = self.getObjCentroid(centroid)
            pos = (int(xc), int(yc))
            # set center of view to the lost object's position
            self.ax1.setCenter(pos)

    def skipForwardToNewID(self):
        self.progressWin = dialogs_generic.QDialogWorkerProgress(
            title='Searching the next frame with a new object', parent=self,
            pbarDesc=f'Searching the next frame with a new object...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)
        
        self.startFindNextNewIdWorker()

    def startFindNextNewIdWorker(self):
        posData = self.data[self.pos_i]
        self._thread = QThread()
        self.findNextNewIdWorker = gui_workers_segmentation_tracking.FindNextNewIdWorker(posData, self)
        self.findNextNewIdWorker.moveToThread(self._thread)
        
        self.findNextNewIdWorker.signals.finished.connect(self._thread.quit)
        self.findNextNewIdWorker.signals.finished.connect(
            self.findNextNewIdWorker.deleteLater
        )
        self._thread.finished.connect(self._thread.deleteLater)

        self.findNextNewIdWorker.signals.finished.connect(
            self.findNextNewIdWorkerFinished
        )
        self.findNextNewIdWorker.signals.progress.connect(self.workerProgress)
        self.findNextNewIdWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.findNextNewIdWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.findNextNewIdWorker.signals.critical.connect(
            self.workerCritical
        )

        self._thread.started.connect(self.findNextNewIdWorker.run)
        self._thread.start()

    def findNextNewIdWorkerFinished(self, next_frame_i):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
                    
        self.navSpinBox.setValue(next_frame_i+1)
        self.framesScrollBarReleased()

    def clearLostObjContoursItems(self):
        self.ax1_lostObjScatterItem.setData([], [])
        self.ax2_lostObjScatterItem.setData([], [])

        self.ax1_lostTrackedScatterItem.setData([], [])
        self.ax2_lostTrackedScatterItem.setData([], [])
        
        self.ax2_lostObjImageItem.clear()
        self.ax2_lostTrackedObjImageItem.clear()
        
        self.ax1_lostObjImageItem.clear()
        self.ax1_lostTrackedObjImageItem.clear()

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

    def _copyAllLostObjects_navigateToFrame(self, frame_i):
        posData = self.data[self.pos_i]
        self.store_data(mainThread=False, autosave=False)

        posData.frame_i = frame_i
        self.get_data()
        self.tracking(wl_update=False)
        self.currentLab2D = self.get_2Dlab(posData.lab)
        self.update_rp() # cannot be more granular as lost obj could be anywhere
        self.updateLostNewCurrentIDs()
        self.store_data(mainThread=False, autosave=False)

        self.lostObjContoursImage[:] = 0
        self.lostObjImage[:] = 0
        prev_rp = posData.allData_li[frame_i-1]['regionprops']
        for lostID in posData.lost_IDs:
            obj = prev_rp.get_obj_from_ID(lostID)
            self.addLostObjsToLostObjImage(obj, lostID, force=True)

    def _copyAllLostObjects_returnToFrame(self, frame_i):
        posData = self.data[self.pos_i]
        self.store_data(autosave=False, mainThread=False)
        posData.frame_i = frame_i
        self.get_data()

    def _copyAllLostObjects_refreshRp(self):
        self.update_rp(draw=False, wl_update=False) # need to change this when merging with opt.

    @disableWindow
    def copyAllLostObjects(self, for_future_frame_n, max_overlap_perc):
        if not self.copyLostObjButton.isChecked():
            return

        posData = self.data[self.pos_i]

        desc = 'Copying all lost objects...'

        self.progressWin = dialogs_generic.QDialogWorkerProgress(
            title=desc, parent=self.mainWin, pbarDesc=desc
        )
        self.progressWin.mainPbar.setMaximum(for_future_frame_n+1)
        self.progressWin.show(self.app)

        self.copyAllLostObjectsThread = QThread()

        self.copyAllLostObjectsWorker = gui_workers_segmentation_tracking.CopyAllLostObjectsWorker(
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
            blinker = qutils.QControlBlink(
                self.copyLostObjToolbar.maxOverlapNumberControl,
                qparent=self.mainWin
            )
            blinker.start()

        self.copyAllLostObjectsWorkerLoop.exit()
        self.update_rp() # global op and obj added, no opt imo unless difference pic
        self.updateAllImages()
        self.store_data()

    def warnLostObjects(self, do_warn=True):
        if not do_warn:
            return True
        
        if not self.warnLostCellsAction.isChecked():
            return True
        
        mode = str(self.modeComboBox.currentText())
        if not mode == 'Segmentation and Tracking':
            return True
        
        posData = self.data[self.pos_i]
        if not posData.lost_IDs:
            return True
        
        frame_i = posData.frame_i
        try:
            accepted_lost_IDs = posData.accepted_lost_IDs.get(frame_i, [])
            already_accepted_lost = (
                Counter(accepted_lost_IDs) == Counter(posData.lost_IDs)
            )
        except AttributeError as err:
            already_accepted_lost = False
        
        if already_accepted_lost:
            return True

        self.nextAction.setDisabled(True)
        self.prevAction.setDisabled(True)
        self.navigateScrollBar.setDisabled(True)
        
        msg = widgets_message_box.myMessageBox()
        warn_msg = html_utils.paragraph(
            'Current frame (compared to previous frame) '
            'has <b>lost the following cells</b>:<br><br>'
            f'{posData.lost_IDs}<br><br>'
            'Are you <b>sure</b> you want to continue?<br>'
        )
        checkBox = QCheckBox('Do not show again')
        noButton, yesButton = msg.warning(
            self, 'Lost cells!', warn_msg,
            buttonsTexts=('No', 'Yes'),
            widgets=checkBox
        )
        doNotWarnLostCells = not checkBox.isChecked()
        self.warnLostCellsAction.setChecked(doNotWarnLostCells)
        if msg.clickedButton == noButton:
            self.nextAction.setDisabled(False)
            self.prevAction.setDisabled(False)
            self.navigateScrollBar.setDisabled(False)
            return False
        
        self.nextAction.setDisabled(False)
        self.prevAction.setDisabled(False)
        self.navigateScrollBar.setDisabled(False)
        if not hasattr(posData, 'accepted_lost_IDs'):
            posData.accepted_lost_IDs = {}
        if frame_i not in posData.accepted_lost_IDs:
            posData.accepted_lost_IDs[frame_i] = []
        
        posData.accepted_lost_IDs[frame_i].extend(posData.lost_IDs)
        # This section is adding the lost cells to tracked_lost_centroids... TBH I dont know why this wasnt done in the first place
        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        accepted_lost_centroids = {
            tuple(int(val) for val in prev_rp.get_centroid(ID, exact=True)) 
            for ID in posData.lost_IDs
        }
        try:
            posData.tracked_lost_centroids[frame_i] = (
                posData.tracked_lost_centroids[frame_i] | (accepted_lost_centroids)
            )
        except KeyError:
            posData.tracked_lost_centroids[frame_i] = accepted_lost_centroids
        return True

    def initLostObjContoursImage(self):
        posData = self.data[self.pos_i]
        z_slice = self.z_lab()
        img = posData.img_data[posData.frame_i]
        Y, X = img[z_slice].shape[-2:]
            
        self.lostObjContoursImage = np.zeros((Y, X, 4), dtype=np.uint8)

    def initLostTrackedObjContoursImage(self):
        posData = self.data[self.pos_i]
        z_slice = self.z_lab()
        img = posData.img_data[posData.frame_i]
        Y, X = img[z_slice].shape[-2:]
            
        self.lostTrackedObjContoursImage = np.zeros((Y, X, 4), dtype=np.uint8)

    def getLostObjImageItem(self, ax):
        if ax == 0:
            return self.ax1_lostObjImageItem
        else:
            return self.ax1_lostTrackedObjImageItem

    def getLostTrackedObjImageItem(self, ax):
        if ax == 0:
            return self.ax1_lostTrackedObjImageItem
        else:
            return self.ax2_lostTrackedObjImageItem

    def updateLostContoursImage(self, ax, draw=True, delROIsIDs=None):
        if draw:
            imageItem = self.getLostObjImageItem(ax)
            if imageItem is None:
                return
        
        if not hasattr(self, 'lostObjContoursImage'):
            self.initLostObjContoursImage()
        else:
            self.lostObjContoursImage[:] = 0

        if delROIsIDs is None:
            delROIsIDs = set()
        
        posData = self.data[self.pos_i]
        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        display_rp_prev = self.get2DRP(frame_i=posData.frame_i -1)
        if posData.whitelist is not None and posData.whitelist.whitelistIDs is not None:
            try:
                whitelist = posData.whitelist.whitelistIDs[posData.frame_i-1]
            except KeyError as e:
                self.logger.warning(
                    f'Failed to get whitelist for frame {posData.frame_i-1}: {e}'
                )
                whitelist = list(posData.IDs)
        else:
            whitelist = None

        contours = []
        for lostID in posData.lost_IDs:
            if lostID in delROIsIDs or (whitelist is not None and lostID not in whitelist):
                continue
            
            obj = prev_rp.get_obj_from_ID(lostID)
            if not self.isObjInCurrSlice(obj.bbox):
                continue
            obj_display = display_rp_prev.get_obj_from_ID(lostID)

            if obj_display is None:
                self.logger.warning(f"Failed to annotate {lostID} as lost ID!")
                continue
            obj_contours = self.getObjContours(
                obj_display,
                all_external=True,
                include_internal=self.showAllContoursToggle.isChecked()
            )
            
            if ax == 0:
                self.addLostObjsToLostObjImage(obj, lostID)
                
            contours.extend(obj_contours)

        if not draw:
            return

        self.drawLostObjContoursImage(imageItem, contours)

    def drawLostObjContoursImage(
            self, imageItem, contours, 
            thickness=1, 
            color=(255, 165, 0, 255) # orange
        ):
        img = self.lostObjContoursImage
        cv2.drawContours(img, contours, -1, color, thickness)
        imageItem.setImage(img)

    def updateLostTrackedContoursImage(
            self, ax, delROIsIDs=None, tracked_lost_IDs=None
        ):
        imageItem = self.getLostTrackedObjImageItem(ax)
        if imageItem is None:
            return
        
        if not hasattr(self, 'lostTrackedObjContoursImage'):
            self.initLostTrackedObjContoursImage()
        else:
            self.lostTrackedObjContoursImage[:] = 0
        
        if delROIsIDs is None:
            delROIsIDs = set()
        
        posData = self.data[self.pos_i]
        if tracked_lost_IDs is None:
            tracked_lost_IDs = self.getTrackedLostIDs()
            
        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        display_rp_prev = self.get2DRP(frame_i=posData.frame_i - 1)
        contours = []
        for tracked_lost_ID in tracked_lost_IDs:
            if tracked_lost_ID in delROIsIDs:
                continue
            
            obj = prev_rp.get_obj_from_ID(tracked_lost_ID)
            if not self.isObjInCurrSlice(obj.bbox):
                continue
        
            obj = display_rp_prev.get_obj_from_ID(tracked_lost_ID)
            obj_contours = self.getObjContours(
                obj,
                all_external=True,
                include_internal=self.showAllContoursToggle.isChecked()
            )
            contours.extend(obj_contours)

        self.drawLostTrackedObjContoursImage(imageItem, contours)

    def drawLostTrackedObjContoursImage(self, imageItem, contours):
        thickness = 1
        color = (0, 255, 0, 255) # green
        img = self.lostTrackedObjContoursImage
        cv2.drawContours(img, contours, -1, color, thickness)
        imageItem.setImage(img)

    def getNearestLostObjID(self, y, x):
        if not self.annotLostObjsToggle.isChecked():
            return
        
        posData = self.data[self.pos_i]
        if not posData.lost_IDs:
            return
        
        prev_lab = posData.allData_li[posData.frame_i-1]['labels']
        if prev_lab is None:
            return
            
        yy, xx, _ = np.nonzero(self.lostObjContoursImage)
        lostObjsContourMask = np.zeros(self.currentLab2D.shape, dtype=bool)
        lostObjsContourMask[yy.astype(int), xx.astype(int)] = True
        
        # Add accepted lost IDs
        try:
            yy, xx, _ = np.nonzero(self.lostTrackedObjContoursImage)
            lostObjsContourMask[yy.astype(int), xx.astype(int)] = True
        except Exception as err:
            pass
            
        _, y_nearest, x_nearest = core.nearest_nonzero_2D(
            lostObjsContourMask, y, x, return_coords=True
        )
        nearest_ID = self.get_2Dlab(prev_lab)[y_nearest, x_nearest]
        
        if nearest_ID == 0:
            return
        
        return nearest_ID

    def setAllLostObjContoursImage(self, delROIsIDs=None):
        self.updateLostContoursImage(ax=0, delROIsIDs=None)
        self.updateLostContoursImage(ax=1, delROIsIDs=None)

    def setAllLostTrackedObjContoursImage(self, delROIsIDs=None):
        self.updateLostTrackedContoursImage(ax=0, delROIsIDs=None)
        self.updateLostTrackedContoursImage(ax=1, delROIsIDs=None)

    # @exec_time
    def highlightLostNew(self):
        if self.modeComboBox.currentText() == 'Viewer':
            return
        
        posData = self.data[self.pos_i]
        
        # self.setAllContoursImages(delROIsIDs=delROIsIDs)
        if posData.frame_i == 0:
            return 

        if not self.annotLostObjsToggle.isChecked():
            return
        
        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        
        if prev_rp is None:
            return

        self.setAllLostObjContoursImage()        
        self.setAllLostTrackedObjContoursImage()

    def addLostObjsToLostObjImage(self, lostObj, lostID, force=False):
        if not force:
            if not self.copyLostObjButton.isChecked():
                return
        
        obj_slice = self.getObjSlice(lostObj.slice)
        obj_image = self.getObjImage(lostObj.image, lostObj.bbox)
        self.lostObjImage[obj_slice][obj_image] = lostID

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
            display_rp = self.get2DRP(frame_i=posData.frame_i - 1)
            lostObj = display_rp.get_obj_from_ID(hoverLostID)
            obj_contours = self.getObjContours(
                lostObj,
                all_external=True,
                include_internal=self.showAllContoursToggle.isChecked()
            )
            for cont in obj_contours:
                xx = cont[:,0]
                yy = cont[:,1]
                self.ax1_lostObjScatterItem.addPoints(xx, yy)
            self.ax1_lostObjScatterItem.setSize(self.contLineWeight+2)

    def resetAcceptedLostIDs(self, from_frame_i=None):
        posData = self.data[self.pos_i]
        if from_frame_i is None:
            from_frame_i = posData.frame_i
        
        posData.tracked_lost_centroids[from_frame_i-1] = set()
        for i in range(from_frame_i, posData.SizeT):
            posData.tracked_lost_centroids[i] = set()


class ManualTrackingGuiElements:
    """Manual tracking ghost-object tool (shortcut T)."""

    def updateGhostMaskOpacity(self, alpha_percentage=None):
        if alpha_percentage is None:
            alpha_percentage = (
                self.manualTrackingToolbar.ghostMaskOpacitySpinbox.value()
            )
        alpha = alpha_percentage/100
        self.ghostMaskItemLeft.setOpacity(alpha)
        self.ghostMaskItemRight.setOpacity(alpha)

    def addManualTrackingItems(self):
        self.ghostContourItemLeft.addToPlotItem()
        self.ghostContourItemRight.addToPlotItem()

        self.ghostMaskItemLeft.addToPlotItem()
        self.ghostMaskItemRight.addToPlotItem()

        Y, X = self.img1.image.shape[:2]
        self.ghostMaskItemLeft.initImage((Y, X))
        self.ghostMaskItemRight.initImage((Y, X))

        self.updateGhostMaskOpacity()

    def removeManualTrackingItems(self):        
        self.ghostContourItemLeft.removeFromPlotItem()
        self.ghostContourItemRight.removeFromPlotItem()

        self.ghostMaskItemLeft.removeFromPlotItem()
        self.ghostMaskItemRight.removeFromPlotItem()

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
        
        display_rp = self.get2DRP(frame_i=posData.frame_i -1)
        obj = display_rp.get_obj_from_ID(ID, warn=False)
        if obj is None:
            self.ghostObject = None
            self.manualTrackingToolbar.showWarning(
                f'The ID {ID} does not exist in previous frame '
                '--> starting a new track.'
            )
            return

        self.ghostObject = obj
        self.manualTrackingToolbar.clearInfoText()

        self.ghostObject.contour_local = self.getObjContours(
            self.ghostObject, local=True
        )

        self.ghostMaskItemLeft.initLookupTable(self.lut[ID])
        self.ghostMaskItemRight.initLookupTable(self.lut[ID])

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

    def _drawGhostContour(self, x, y):
        if self.ghostObject is None:
            return
        
        ID = self.ghostObject.label
        yc, xc = self.ghostObject.local_centroid
        Dx = x-xc
        Dy = y-yc
        xx = self.ghostObject.contour_local[:,0] + Dx
        yy = self.ghostObject.contour_local[:,1] + Dy
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
        slices = gui_helpers.get_slices_local_into_global_arr(bbox, (Y, X))
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
