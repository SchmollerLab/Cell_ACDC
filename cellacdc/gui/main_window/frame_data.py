"""Per-frame data of the main GUI: store_data/get_data, labels,
regionprops, object lookups and counts, undo/redo states. Also the
get_data_exception_handler decorator. Mixins of guiWin
(gui/main_window/window.py) only: the measurements viewer never composes
them.
"""
import os
import traceback
import uuid
from collections import defaultdict
import numpy as np
import pandas as pd
from functools import wraps
from qtpy.QtCore import QThread, QMutex, QWaitCondition
from cellacdc import exception_handler
from cellacdc import html_utils
from ..workers import loading_saving as gui_workers_loading_saving
from cellacdc.myutils import table_utils
from .. import helpers as gui_helpers
from cellacdc.shared.widgets import message_box as widgets_message_box
from ..widgets import edit_toolbars as gui_widgets_edit_toolbars
from cellacdc import printl
from cellacdc import transformation
from cellacdc.trackers.CellACDC_normal_division.CellACDC_normal_division_tracker import (
    is_lineage_tree_annotated, init_lineage_tree_cols_of_added_IDs
)
from . import constants as gui_main_constants


def get_data_exception_handler(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        try:
            if func.__code__.co_argcount==1 and func.__defaults__ is None:
                result = func(self)
            elif func.__code__.co_argcount>1 and func.__defaults__ is None:
                result = func(self, *args)
            else:
                result = func(self, *args, **kwargs)
        except Exception as e:
            try:
                if self.progressWin is not None:
                    self.progressWin.workerFinished = True
                    self.progressWin.close()
                    self.progressWin = None
            except AttributeError:
                pass
            result = None
            posData = self.data[self.pos_i]
            acdc_df_filename = os.path.basename(posData.acdc_output_csv_path)
            segm_filename = os.path.basename(posData.segm_npz_path)
            traceback_str = traceback.format_exc()
            self.logger.exception(traceback_str)
            msg = widgets_message_box.myMessageBox(wrapText=False, showCentered=False)
            msg.addShowInFileManagerButton(self.logs_path, txt='Show log file...')
            msg.setDetailedText(traceback_str)
            err_msg = html_utils.paragraph(f"""
                Error in function <code>{func.__name__}</code>.<br><br>
                One possbile explanation is that either the
                <code>{acdc_df_filename}</code> file<br>
                or the segmentation file <code>{segm_filename}</code><br>
                <b>are being synchronized by a cloud service (e.g., Google Drive 
                or OneDrive) or they are corrupted/damaged</b>.<br><br>
                <b>Try moving these files</b> (one by one) outside of the
                <code>{os.path.dirname(posData.relPath)}</code> folder
                <br>and reloading the data.<br><br>
                More details below or in the terminal/console.<br><br>
                Note that the <b>error details</b> from this session are
                also <b>saved in the following file</b>:<br><br>
                {self.log_path}<br><br>
                Please <b>send the log file</b> when reporting a bug, thanks!
                <b>Please restart Cell-ACDC, we apologise for any inconvenience.</b><br><br>

            """)

            msg.critical(self, 'Critical error', err_msg)
            self.is_error_state = True
            raise e
        return result
    return inner_function


class UndoRedoGuiElements:
    """Undo/redo state stacks of segmentation and cell cycle edits,
    including the (disabled) store-state worker.
    """

    def gui_createStoreStateWorker(self):
        self.storeStateWorker = None
        return
        self.storeStateThread = QThread()
        self.autoSaveMutex = QMutex()
        self.autoSaveWaitCond = QWaitCondition()

        self.storeStateWorker = gui_workers_loading_saving.StoreGuiStateWorker(
            self.autoSaveMutex, self.autoSaveWaitCond
        )

        self.storeStateWorker.moveToThread(self.storeStateThread)
        self.storeStateWorker.finished.connect(self.storeStateThread.quit)
        self.storeStateWorker.finished.connect(self.storeStateWorker.deleteLater)
        self.storeStateThread.finished.connect(self.storeStateThread.deleteLater)

        self.storeStateWorker.sigDone.connect(self.storeStateWorkerDone)
        self.storeStateWorker.progress.connect(self.workerProgress)
        self.storeStateWorker.finished.connect(self.storeStateWorkerClosed)
        
        self.storeStateThread.started.connect(self.storeStateWorker.run)
        self.storeStateThread.start()

        self.logger.info('Store state worker started.')

    def storeStateWorkerDone(self):
        if self.storeStateWorker.callbackOnDone is not None:
            self.storeStateWorker.callbackOnDone()
        self.storeStateWorker.callbackOnDone = None

    def storeStateWorkerClosed(self):
        self.logger.info('Store state worker started.')

    def addCcaState(self, frame_i, cca_df, undoId):
        posData = self.data[self.pos_i]
        posData.UndoRedoCcaStates[frame_i].insert(
            0, {'id': undoId, 'cca_df': cca_df.copy()}
        )

    def addCurrentState(
            self, 
            storeImage=False, 
            storeOnlyZoom=False,
            zRange: tuple[int, int] | None=None
        ): 
        posData = self.data[self.pos_i]
        if posData.cca_df is not None:
            cca_df = posData.cca_df.copy()
        else:
            cca_df = None

        if storeImage:
            image = self.img1.image.copy()
        else:
            image = None

        if storeOnlyZoom:
            labels, crop_slice = transformation.crop_2D(
                self.currentLab2D, self.ax1.viewRange(), tolerance=10,
                return_copy=False
            )
            if self.isSegm3D:
                z = self.z_lab(checkIfProj=True)
                if zRange is not None:
                    z_slice = slice(zRange[0], zRange[1])
                    crop_slice = (z_slice, *crop_slice)
                    labels = posData.lab[crop_slice].copy()
                elif z is None:
                    z_slice = slice(0, len(posData.lab))
                    crop_slice = (z_slice, *crop_slice)
                    labels = posData.lab[crop_slice].copy()
                else:
                    z_slice = z
                    crop_slice = (z_slice, *crop_slice)
                    labels = labels.copy()
            else:
                labels = labels.copy()
        else:
            labels = posData.lab.copy()
            crop_slice = None
        
        state = {
            'image': image,
            'labels': labels,
            'editID_info': posData.editID_info.copy(),
            'binnedIDs': posData.binnedIDs.copy(),
            'keptObejctsIDs': self.keptObjectsIDs.copy(),
            'ripIDs': posData.ripIDs.copy(),
            'cca_df': cca_df,
            'crop_slice': crop_slice
        }
        posData.UndoRedoStates[posData.frame_i].insert(0, state)
        
        # posData.storedLab = np.array(posData.lab, order='K', copy=True)
        # self.storeStateWorker.callbackOnDone = callbackOnDone
        # self.storeStateWorker.enqueue(posData, self.img1.image)

    def getCurrentState(self):
        posData = self.data[self.pos_i]
        i = posData.frame_i
        c = self.UndoCount
        state = posData.UndoRedoStates[i][c]
        if state['image'] is None:
            image_left = None
        else:
            image_left = state['image'].copy()
        
        crop_slice = state['crop_slice']
        if crop_slice is None:
            posData.lab = state['labels'].copy()
        elif self.isSegm3D:
            z_slice, slice_y, slice_x = crop_slice
            posData.lab[..., z_slice, slice_y, slice_x] = state['labels'].copy()
        else:
            slice_y, slice_x = crop_slice
            posData.lab[..., slice_y, slice_x] = state['labels'].copy()
        
        posData.editID_info = state['editID_info'].copy()
        posData.binnedIDs = state['binnedIDs'].copy()
        posData.ripIDs = state['ripIDs'].copy()
        # The state stores a plain list --> rebuild the "Keep IDs" selection
        # bound to its line edit and confirm action
        self.keptIDsLineEdit.blockSignals(True)
        try:
            self.keptObjectsIDs = gui_widgets_edit_toolbars.KeptObjectIDsList(
                self.keptIDsLineEdit, self.keepIDsConfirmAction
            )
            for ID in state['keptObejctsIDs']:
                self.keptObjectsIDs.append(ID, editText=False)
            self.keptObjectsIDs.setText()
        finally:
            self.keptIDsLineEdit.blockSignals(False)
        self.keptIDsLineEdit.setInstructionsText()
        cca_df = state['cca_df']
        if cca_df is not None:
            posData.cca_df = state['cca_df'].copy()
        else:
            posData.cca_df = None
        return image_left

    # @exec_time
    def storeUndoRedoStates(
            self, 
            UndoFutFrames, 
            storeImage=False, 
            storeOnlyZoom=False,
            zRange=None
        ):
        posData = self.data[self.pos_i]
        if UndoFutFrames:
            # Since we modified current frame all future frames that were already
            # visited are not valid anymore. Undo changes there
            self.reInitLastSegmFrame(updateImages=False)
        
        # Keep only 5 Undo/Redo states
        if len(posData.UndoRedoStates[posData.frame_i]) > 5:
            posData.UndoRedoStates[posData.frame_i].pop(-1)

        # Restart count from the most recent state (index 0)
        # NOTE: index 0 is most recent state before doing last change
        self.UndoCount = 0
        self.undoAction.setEnabled(True)
        self.addCurrentState(
            storeImage=storeImage, 
            storeOnlyZoom=storeOnlyZoom,
            zRange=zRange
        )

    def storeUndoRedoCca(self, frame_i, cca_df, undoId):
        if self.isSnapshot:
            # For snapshot mode we don't store anything because we have only
            # segmentation undo action active
            return
        """
        Store current cca_df along with a unique id to know which cca_df needs
        to be restored
        """

        posData = self.data[self.pos_i]

        # Restart count from the most recent state (index 0)
        # NOTE: index 0 is most recent state before doing last change
        self.UndoCcaCount = 0
        self.undoAction.setEnabled(True)

        self.addCcaState(frame_i, cca_df, undoId)

        # Keep only 10 Undo/Redo states
        if len(posData.UndoRedoCcaStates[frame_i]) > 10:
            posData.UndoRedoCcaStates[frame_i].pop(-1)

    def UndoCca(self):
        posData = self.data[self.pos_i]
        # Undo current ccaState
        storeState = False
        if self.UndoCount == 0:
            undoId = uuid.uuid4()
            self.addCcaState(posData.frame_i, posData.cca_df, undoId)
            storeState = True


        # Get previously stored state
        self.UndoCount += 1
        currentCcaStates = posData.UndoRedoCcaStates[posData.frame_i]
        prevCcaState = currentCcaStates[self.UndoCount]
        posData.cca_df = prevCcaState['cca_df']
        self.store_cca_df()
        self.updateAllImages()

        # Check if we have undone all states
        if len(currentCcaStates) > self.UndoCount:
            # There are no states left to undo for current frame_i
            self.undoAction.setEnabled(False)

        # Undo all past and future frames that has a last status inserted
        # when modyfing current frame
        prevStateId = prevCcaState['id']
        for frame_i in range(0, posData.SizeT):
            if storeState:
                cca_df_i = self.get_cca_df(frame_i=frame_i, return_df=True)
                if cca_df_i is None:
                    break
                # Store current state to enable redoing it
                self.addCcaState(frame_i, cca_df_i, undoId)

            CcaStates_i = posData.UndoRedoCcaStates[frame_i]
            if len(CcaStates_i) <= self.UndoCount:
                # There are no states to undo for frame_i
                continue

            CcaState_i = CcaStates_i[self.UndoCount]
            id_i = CcaState_i['id']
            if id_i != prevStateId:
                # The id of the state in frame_i is different from current frame
                continue

            cca_df_i = CcaState_i['cca_df']
            self.store_cca_df(frame_i=frame_i, cca_df=cca_df_i, autosave=False)
        
        self.resetWillDivideInfo()
        self.enqAutosave()

    def undo(self):
        if self.isBrushStrokesPromptActive() and self.undoBrushStrokes():
            return

        isBoxPromptActive = self.isBoxPromptActive()
        if isBoxPromptActive and self.undoMagicPromptsBoxes():
            return

        addPointsByClickingButton = self.buttonAddPointsByClickingActive()
        # With box prompts the prompts points layer stays active, but clicks
        # do not add points (nothing to undo there)
        if addPointsByClickingButton is not None and not isBoxPromptActive:
            done = self.undoAddPoint(addPointsByClickingButton.action)
            if done:
                return
        
        if self.UndoCount == 0:
            # Store current state to enable redoing it
            self.addCurrentState()
    
        posData = self.data[self.pos_i]
        # Get previously stored state
        if self.UndoCount < len(posData.UndoRedoStates[posData.frame_i])-1:
            self.UndoCount += 1
            # Since we have undone then it is possible to redo
            self.redoAction.setEnabled(True)

            # Restore state
            image_left = self.getCurrentState()
            self.update_rp()
            self.updateAllImages(image=image_left)
            self.store_data()

        if not self.UndoCount < len(posData.UndoRedoStates[posData.frame_i])-1:
            # We have undone all available states
            self.undoAction.setEnabled(False)
        
        if self.whitelistIDsButton.isChecked():
            self.whitelistHighlightIDs()

    def redo(self):
        posData = self.data[self.pos_i]
        # Get previously stored state
        if self.UndoCount > 0:
            self.UndoCount -= 1
            # Since we have redone then it is possible to undo
            self.undoAction.setEnabled(True)

            # Restore state
            image_left = self.getCurrentState()
            self.update_rp()
            self.updateAllImages(image=image_left)
            self.store_data()

        if not self.UndoCount > 0:
            # We have redone all available states
            self.redoAction.setEnabled(False)

        if self.whitelistIDsButton.isChecked():
            self.whitelistHighlightIDs()

    def clearUndoQueue(self):
        posData = self.data[self.pos_i]
        self.UndoCount = 0
        self.redoAction.setEnabled(False)
        self.undoAction.setEnabled(False)
        posData.UndoRedoStates = [[] for _ in range(posData.SizeT)]
        posData.UndoRedoCcaStates = [[] for _ in range(posData.SizeT)]
        if hasattr(self, 'undoAddPointQueueMapper'):
            self.undoAddPointQueueMapper = defaultdict(list)
        self.brushStrokesUndo = {}
        self.magicPromptsBoxesUndo = {}


class FrameDataGuiElements:
    """Per-frame data of the loaded positions: store_data/get_data
    (allData_li), labels and regionprops updates (update_rp), object
    lookups and object counts. setAllIDs, countObjects* and
    updateObjectCounts override the shared versions of
    gui/mixins/object_info.py.
    """

    def rpCurr2D(self, frame_i=None, slice_i=None, depth_axis=None, pos_i=None):
        posData = self.data[self.pos_i if pos_i is None else pos_i]
        # zProjHow = self.zProjComboBox.currentText()
        # isZslice = zProjHow == 'single z-slice'
        if frame_i is None:
            rp = posData.rp
        else:
            rp = posData.allData_li[frame_i]['regionprops']
        if not self.isSegm3D: # or not isZslice:
            return rp

        if slice_i is None:
            slice_i = self.zSliceScrollBar.sliderPosition()
        if depth_axis is None:
            depth_axis = self.switchPlaneCombobox.depthAxes()

        rp2D = rp.get_slice_rp(slice_i, slicing=depth_axis)
        return rp2D

    def updateAnnotatedIDs(self, oldIDs, newIDs, logger=print):
        logger('Updating annotated IDs...')
        posData = self.data[self.pos_i]

        mapper = dict(zip(oldIDs, newIDs))
        posData.ripIDs = set([mapper[ripID] for ripID in posData.ripIDs])
        posData.binnedIDs = set([mapper[binID] for binID in posData.binnedIDs])
        self.keptObjectsIDs = gui_widgets_edit_toolbars.KeptObjectIDsList(
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

    def getConcatAcdcDf(self):
        acdc_dfs = []
        keys = []
        posData = self.data[self.pos_i]
        for frame_i, data_dict in enumerate(posData.allData_li):
            lab = data_dict['labels']
            if lab is None:
                break
            
            acdc_df = data_dict['acdc_df']
            if acdc_df is None:
                break
            
            acdc_dfs.append(acdc_df)
            keys.append(frame_i)
        
        if not acdc_dfs:
            return
        
        return pd.concat(acdc_dfs, keys=keys, names=['frame_i'])

    def reload_cb(self):
        posData = self.data[self.pos_i]
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)
        labData = np.load(posData.segm_npz_path)
        # Keep compatibility with .npy and .npz files
        try:
            lab = labData['arr_0'][posData.frame_i]
        except Exception as e:
            lab = labData[posData.frame_i]
        posData.segm_data[posData.frame_i] = lab.copy()
        self.get_data()
        self.tracking()
        self.updateAllImages()

    def setAllIDs(self, onlyVisited=False, pos_i=None):
        """Set `posData.allIDs` to the IDs existing in any frame (only up to
        the first unvisited frame if `onlyVisited`) of every position, or
        only of the position `pos_i` if not None.

        For the displayed frame the live `posData.rp` is used, because
        `posData.allData_li` is synced only by `store_data`.
        """
        if pos_i is None:
            posDataList = self.data
        else:
            posDataList = (self.data[pos_i],)

        currentPosData = self.data[self.pos_i]
        for posData in posDataList:
            # Build the set locally and assign it at the end so that readers
            # in other threads (e.g., `trackingWorker`) never get a partial set
            allIDs = set()
            # Note that the segmentation file can have less frames than the
            # visited ones (frames past its end are visited with empty masks)
            for frame_i in range(len(posData.allData_li)):
                rp = None
                isCurrentFrame = (
                    posData is currentPosData and frame_i == posData.frame_i
                )
                if isCurrentFrame:
                    rp = getattr(posData, 'rp', None)

                if rp is None:
                    lab = posData.allData_li[frame_i]['labels']
                    if lab is None and onlyVisited:
                        break

                    rp = posData.allData_li[frame_i]['regionprops']

                if rp is None:
                    if lab is None and frame_i < len(posData.segm_data):
                        lab = posData.segm_data[frame_i]

                    if lab is None:
                        # Not segmented and not visited yet
                        continue

                    rp = self._acdcRegionProps(
                        lab, precache_centroids=False
                    )
                    posData.allData_li[frame_i]['regionprops'] = rp

                allIDs.update(rp.IDs)

            posData.allIDs = allIDs

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
            IDs = _posData.allData_li[0]['regionprops'].IDs
            if os.path.exists(_posData.acdc_output_csv_path):
                numObjectsVisitedPosPrevious += len(IDs)
            if IDs:
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

    def get_curr_lab(self, curr_lab: np.ndarray|None = None, frame_i: int|None = None):
        """Get the current labels for the position data. Hirarchically checks:
        1. If `curr_lab` is provided, use it.
        2. If `posData.lab` is not None, use it.
        3. If `posData.allData_li[frame_i]['labels']` exists, use it.
        4. If `posData.segm_data[frame_i]` exists, use it.
        
        If frame_i is None, uses the current frame index from `posData`.

        Parameters
        ----------
        curr_lab : np.ndarray, optional
            Current labels for the position data if it should be checked 
            if its not None first, by default None
        frame_i : int, optional
            Frame index to use for retrieving labels, by default None

        Returns
        -------
        np.ndarray
            Current labels for the position data
        """
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i
        
        if curr_lab is None and frame_i == posData.frame_i:
            curr_lab = posData.lab
            
        if curr_lab is None:
            try:
                curr_lab = posData.allData_li[frame_i]['labels'].copy()
            except:
                pass
            
        if curr_lab is None:
            try:
                curr_lab = posData.segm_data[frame_i].copy()
            except:
                pass
        
        return curr_lab

    def unstore_data(self):
        posData = self.data[self.pos_i]
        posData.allData_li[posData.frame_i] = table_utils.get_empty_stored_data_dict()

    def getStoredSegmData(self):
        posData = self.data[self.pos_i]
        segm_data = []
        for data_frame_i in posData.allData_li:
            lab = data_frame_i['labels']
            if lab is None:
                break
            segm_data.append(lab)
        return np.array(segm_data)

    @exception_handler
    def store_data(
            self, pos_i=None, enforce=True, debug=False, mainThread=True,
            autosave=True, store_cca_df_copy=False
        ):
        pos_i = self.pos_i if pos_i is None else pos_i
        posData = self.data[pos_i]
        if posData.frame_i < 0:
            # In some cases we set frame_i = -1 and then call next_frame
            # to visualize frame 0. In that case we don't store data
            # for frame_i = -1
            return

        mode = str(self.modeComboBox.currentText())

        if mode == 'Viewer' and not enforce:
            return

        # if not mainThread:
        #     self.lin_tree_ask_changes()
        
        allData_li = posData.allData_li[posData.frame_i]
        
        if hasattr(posData, 'rp'):
            allData_li['regionprops'] = posData.rp
        else:
            rp = self._acdcRegionProps(posData.lab, precache_centroids=False)
            allData_li['regionprops'] = rp
        allData_li['labels'] = posData.lab.copy()
        allData_li['manualBackgroundLab'] = (
            posData.manualBackgroundLab
        )
        if self.manualAnnotPastButton.isChecked():
            self.store_manual_annot_data(
                posData=posData, data_frame_i=allData_li
            )

        # Store dynamic metadata
        is_cell_dead_li = [False]*len(posData.rp)
        is_cell_excluded_li = [False]*len(posData.rp)
        IDs = [0]*len(posData.rp)
        xx_centroid = [0]*len(posData.rp)
        yy_centroid = [0]*len(posData.rp)
        if self.isSegm3D:
            zz_centroid = [0]*len(posData.rp)
        areManuallyEdited = [0]*len(posData.rp)
        editedNewIDs = [vals[2] for vals in posData.editID_info]
        for i, obj in enumerate(posData.rp):
            is_cell_dead_li[i] = obj.dead
            is_cell_excluded_li[i] = obj.excluded
            IDs[i] = obj.label
            centroid = posData.rp.get_centroid(obj.label, exact=True)
            if centroid is None:
                continue

            if self.isSegm3D:
                zz_centroid[i] = int(centroid[0])
                xx_centroid[i] = int(centroid[2])
                yy_centroid[i] = int(centroid[1])
            else:
                xx_centroid[i] = int(centroid[1])
                yy_centroid[i] = int(centroid[0])
            if obj.label in editedNewIDs:
                areManuallyEdited[i] = 1

        posData.STOREDmaxID = max(IDs, default=0)

        acdc_df = allData_li['acdc_df']
        if acdc_df is None:
            allData_li['acdc_df'] = pd.DataFrame(
                {
                    'Cell_ID': IDs,
                    'is_cell_dead': is_cell_dead_li,
                    'is_cell_excluded': is_cell_excluded_li,
                    'x_centroid': xx_centroid,
                    'y_centroid': yy_centroid,
                    'was_manually_edited': areManuallyEdited
                }
            ).set_index('Cell_ID')
           
            if self.isSegm3D:
               allData_li['acdc_df']['z_centroid'] = (
                    zz_centroid
                )
        else:
            # Filter or add IDs that were not stored yet
            acdc_df = acdc_df.drop(columns=['time_seconds'], errors='ignore')
            added_IDs = [ID for ID in IDs if ID not in acdc_df.index]
            is_lin_tree_annotated = is_lineage_tree_annotated(acdc_df)
            acdc_df = acdc_df.reindex(IDs, fill_value=0)
            if added_IDs:
                # Zeros are not valid lineage tree values (no ID 0)
                frame_i = posData.frame_i
                prev_acdc_df = None
                if frame_i > 0:
                    prev_acdc_df = posData.allData_li[frame_i-1].get('acdc_df')
                acdc_df = init_lineage_tree_cols_of_added_IDs(
                    acdc_df, added_IDs, frame_i, is_lin_tree_annotated,
                    prev_acdc_df=prev_acdc_df
                )
            acdc_df['is_cell_dead'] = is_cell_dead_li
            acdc_df['is_cell_excluded'] = is_cell_excluded_li
            acdc_df['x_centroid'] = xx_centroid
            acdc_df['y_centroid'] = yy_centroid
            if self.isSegm3D:
                acdc_df['z_centroid'] = zz_centroid
            acdc_df['was_manually_edited'] = areManuallyEdited
            allData_li['acdc_df'] = acdc_df

        if mainThread:
            self.pointsLayerDataToDf(posData)
            
        self.store_cca_df(
            pos_i=pos_i, mainThread=mainThread, autosave=autosave, 
            store_cca_df_copy=store_cca_df_copy
        )

    def nearest_point_2Dyx(self, points, all_others):
        """
        Given 2D array of [y, x] coordinates points and all_others return the
        [y, x] coordinates of the two points (one from points and one from all_others)
        that have the absolute minimum distance
        """
        # Compute 3D array where each ith row of each kth page is the element-wise
        # difference between kth row of points and ith row in all_others array.
        # (i.e. diff[k,i] = points[k] - all_others[i])
        diff = points[:, np.newaxis] - all_others
        # Compute 2D array of distances where
        # dist[i, j] = euclidean dist (points[i],all_others[j])
        dist = np.linalg.norm(diff, axis=2)
        # Compute i, j indexes of the absolute minimum distance
        i, j = np.unravel_index(dist.argmin(), dist.shape)
        nearest_point = all_others[j]
        point = points[i]
        min_dist = np.min(dist)
        return min_dist, nearest_point

    def warnFrameNeverVisitedSegmMode(self):
        msg = widgets_message_box.myMessageBox()
        warn_cca = msg.critical(
            self, 'Next frame NEVER visited',
            'Next frame was never visited in "Segmentation and Tracking"'
            'mode.\n You cannot perform cell cycle analysis on frames'
            'where segmentation and/or tracking errors were not'
            'checked/corrected.\n\n'
            'Switch to "Segmentation and Tracking" mode '
            'and check/correct next frame,\n'
            'before attempting cell cycle analysis again',
        )
        return False

    def set_2Dlab(self, lab2D, lab3D=None):
        posData = self.data[self.pos_i]
        
        if lab3D is None:
            lab3D = posData.lab
            
        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if isZslice:
                lab3D[self.z_lab()] = lab2D
            else:
                lab3D[:] = lab2D
        else:
            if lab3D.shape == lab2D.shape:
                lab3D[...] = lab2D
            else:
                posData.lab = lab2D

    def get_labels_array(
            self, 
            from_store=False, 
            frame_i=None, 
            return_existing=False,
            return_copy=True
        ):
        """Get the labels array.
        
        Parameters
        ----------
        from_store : bool, optional
            If True load the labels array from the stored posData.allData_li, 
            i.e., from RAM. Default is False
        frame_i : int, optional
            If None, use the current frame index. Default is  None
        return_existing : bool, optional
            If True, the second return element will be a boolean that 
            is True if the labels array was found stored in `posData.allData_li`. 
            Default is  False
        return_copy : bool, optional
            If True returns a copy of the labels array

        Returns
        -------
        numpy.ndarray or tuple of (numpy.ndarray, bool)
            The first element is the labels array requested. If `return_existing` 
            is True then this method also returns a second boolean element that 
            is True if the labels array was found in in `posData.allData_li`.
        
        Note
        ----
        
        If `from_store` is True then this method will try to get the stored 
        labels array. If any error occurs then the returned labels are the 
        saved ones in the segmentation file (i.e., from hard drive).
        
        """   
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i
        
        existing = True
        if from_store:
            try:
                labels = posData.allData_li[frame_i]['labels']
                if labels is None:
                    from_store = False
            except Exception as err:
                from_store = False
        
        if not from_store:
            try:
                labels = posData.segm_data[frame_i]
            except IndexError:
                existing = False
                # Visting a frame that was not segmented --> empty masks
                if self.isSegm3D:
                    shape = (posData.SizeZ, posData.SizeY, posData.SizeX)
                else:
                    shape = (posData.SizeY, posData.SizeX)
                labels = np.zeros(shape, dtype=np.uint32)
                rp = self._acdcRegionProps(labels, precache_centroids=False)
                if frame_i == posData.frame_i:
                    posData.rp = rp
                    posData.IDs = []
                posData.allData_li[frame_i]['regionprops'] = rp
                return_copy = False
        
        if return_copy:
            labels = labels.copy()
            
        if return_existing:
            return labels, existing
        else:
            return labels

    def addYXcentroidToDf(self, df):
        posData = self.data[self.pos_i]
        for obj in posData.rp:
            ID = obj.label
            centroid = posData.rp.get_centroid(obj, exact=True)
            y_centroid = int(self.getObjCentroid(centroid)[0])
            x_centroid = int(self.getObjCentroid(centroid)[1])
            df.at[ID, 'y_centroid'] = y_centroid
            df.at[ID, 'x_centroid'] = x_centroid
        return df

    def _get_editID_info(self, df):
        if 'was_manually_edited' not in df.columns:
            return []
        
        if 'y_centroid' not in df.columns or 'x_centroid' not in df.columns:
            df = self.addYXcentroidToDf(df)
        
        manually_edited_df = df[df['was_manually_edited'] > 0]
        editID_info = [
            (row.y_centroid, row.x_centroid, row.Index)
            for row in manually_edited_df.itertuples()
        ]
        return editID_info

    def apply_manual_edits_to_lab_if_needed(self, lab):
        posData = self.data[self.pos_i]
        data_frame_i = posData.allData_li[posData.frame_i]
        edited_lab_dict = data_frame_i['manually_edited_lab']['lab']        
        if not edited_lab_dict:
            return lab
        
        # zoom_slice = data_frame_i['manually_edited_lab']['zoom_slice']
        for z, lab_edited in edited_lab_dict.items():
            if not self.isSegm3D:
                # lab[zoom_slice] = lab_edited
                lab = lab_edited
                break
            
            lab[z] = lab_edited
            
            # lab[z, zoom_slice[0], zoom_slice[1]] = zoom_lab
            
        return lab

    def _get_data_unvisited(self, posData, debug=False, lin_tree_init=True,):
        posData.editID_info = []
        proceed_cca = True
        never_visited = True
        if str(self.modeComboBox.currentText()) == 'Cell cycle analysis':
            # Warn that we are visiting a frame that was never segm-checked
            # on cell cycle analysis mode
            msg = widgets_message_box.myMessageBox()
            txt = html_utils.paragraph(
                'Segmentation and Tracking was <b>never checked from '
                f'frame {posData.frame_i+1} onwards</b>.<br><br>'
                'To ensure correct cell cell cycle analysis you have to '
                'first visit the frames after '
                f'{posData.frame_i+1} with "Segmentation and Tracking" mode.'
            )
            warn_cca = msg.critical(
                self, 'Never checked segmentation on requested frame', txt                    
            )
            proceed_cca = False
            return proceed_cca, never_visited

        elif str(self.modeComboBox.currentText()) == 'Normal division: Lineage tree':
            # Warn that we are visiting a frame that was never segm-checked
            # on Normal division: Lineage tree mode
            msg = widgets_message_box.myMessageBox()
            txt = html_utils.paragraph(
                'Segmentation and Tracking was <b>never checked from '
                f'frame {posData.frame_i+1} onwards</b>.<br><br>'
                'To ensure correct lineage tree analysis you have to '
                'first visit the frames after '
                f'{posData.frame_i+1} with "Segmentation and Tracking" mode.'
            )
            warn_cca = msg.critical(#???
                self, 'Never checked segmentation on requested frame', txt                    
            )
            proceed_cca = False
            return proceed_cca, never_visited
        
        # Requested frame was never visited before. Load from HDD
        labels = self.get_labels_array()
        posData.lab = self.apply_manual_edits_to_lab_if_needed(
            labels
        )
        posData.rp = posData.allData_li[posData.frame_i]['regionprops']
        if posData.rp is None:
            posData.rp = self._acdcRegionProps(labels, precache_centroids=False)
        # get stored IDs
        self.setManualBackgroundLab()
        
        if posData.acdc_df is not None:
            frames = posData.acdc_df.index.get_level_values(0)
            if posData.frame_i in frames:
                # Since there was already segmentation metadata from
                # previous closed session add it to current metadata
                df = posData.acdc_df.loc[posData.frame_i].copy()
                binnedIDs_df = df[df['is_cell_excluded']>0]
                binnedIDs = set(binnedIDs_df.index).union(posData.binnedIDs)
                posData.binnedIDs = binnedIDs
                ripIDs_df = df[df['is_cell_dead']>0]
                ripIDs = set(ripIDs_df.index).union(posData.ripIDs)
                posData.ripIDs = ripIDs
                posData.editID_info.extend(self._get_editID_info(df))
                # Load cca df into current metadata
                if 'cell_cycle_stage' in df.columns:
                    cca_cols = df.columns.intersection(self.cca_df_colnames)
                    cca_df = df[cca_cols].dropna()
                    if cca_df.empty:
                        df = df.drop(
                            columns=self.cca_df_colnames, errors='ignore'
                        )
                    else:
                        df = df.loc[cca_df.index]
                        cols = self.cca_df_int_cols
                        df[cols] = df[cols].astype('Int64')
                    
                i = posData.frame_i
                posData.allData_li[i]['acdc_df'] = df.copy()
        
        if self.lineage_tree is None and lin_tree_init:
            self.initLinTree()
    
        self.get_cca_df()
        
        return proceed_cca, never_visited

    def _get_data_visited(self, posData, debug=False, lin_tree_init=True,):        
        # Requested frame was already visited. Load from RAM.
        never_visited = False
        posData.lab = self.get_labels_array(from_store=True)
        # posData.rp = self._acdcRegionProps(posData.lab, precache_centroids=False)
        posData.rp = posData.allData_li[posData.frame_i]['regionprops']
        df = posData.allData_li[posData.frame_i]['acdc_df']
        if df is None:
            posData.binnedIDs = set()
            posData.ripIDs = set()
            posData.editID_info = []
        else:
            try:
                binnedIDs_df = df[df['is_cell_excluded']>0]
            except Exception as err:
                df = gui_helpers.fix_acdc_df_dtypes(df)
                binnedIDs_df = df[df['is_cell_excluded']>0]
            posData.binnedIDs = set(binnedIDs_df.index)
            ripIDs_df = df[df['is_cell_dead']>0]
            posData.ripIDs = set(ripIDs_df.index)
            posData.editID_info = self._get_editID_info(df)
        self.setManualBackgroundLab(load_from_store=True, debug=debug)
        if self.lineage_tree is None and lin_tree_init:
            self.initLinTree()
        
        self.get_cca_df(debug=debug)

        return True, never_visited

    @get_data_exception_handler
    def get_data(self, debug=False, lin_tree_init=True):
        posData = self.data[self.pos_i]
        proceed_cca = True
        never_visited = False
        if posData.frame_i > 2:
            # Remove undo states from 4 frames back to avoid memory issues
            posData.UndoRedoStates[posData.frame_i-4] = []
            # Check if current frame contains undo states (not empty list)
            if posData.UndoRedoStates[posData.frame_i]:
                self.undoAction.setDisabled(False)
            elif posData.UndoRedoCcaStates[posData.frame_i]:
                self.undoAction.setDisabled(False)
            else:
                self.undoAction.setDisabled(True)
        self.UndoCount = 0
        
        # If stored labels is None then it is the first time we visit this frame
        if posData.allData_li[posData.frame_i]['labels'] is None:
            proceed_cca, never_visited =  self._get_data_unvisited(
                posData, lin_tree_init=lin_tree_init,
            )
            if not proceed_cca:
                return proceed_cca, never_visited
        else:
            proceed_cca, never_visited = self._get_data_visited(
                posData, lin_tree_init=lin_tree_init, debug=debug
            )

        if posData.rp is None: #
            rp = self._acdcRegionProps(posData.lab, precache_centroids=False)
            posData.rp = rp
            posData.allData_li[posData.frame_i]['regionprops'] = rp
        self.update_rp_metadata(draw=False)
        posData.IDs = posData.rp.IDs
        self.pointsLayerDfsToData(posData)
        return proceed_cca, never_visited

    def updateLastVisitedFrame(self, last_visited_frame_i=None):
        if last_visited_frame_i is None:
            posData = self.data[self.pos_i]
            last_visited_frame_i = posData.frame_i
        
        mode = str(self.modeComboBox.currentText())
        if mode == 'Viewer':
            return
        elif mode == 'Segmentation and Tracking':
            posData = self.data[self.pos_i]
            if posData.last_tracked_i >= last_visited_frame_i:
                return
            posData.last_tracked_i = last_visited_frame_i
        elif mode == 'Cell cycle analysis':
            if self.last_cca_frame_i >= last_visited_frame_i:
                return
            self.last_cca_frame_i = last_visited_frame_i

    def getObjOptsSegmLabels(self, obj):
        if not self.labelsGrad.showLabelsImgAction.isChecked():
            return

        objOpts = self.getObjTextAnnotOpts(obj, 'Draw only IDs', ax=1)
        return objOpts

    def removeObjectFromRp(self, delID, lab):
        posData = self.data[self.pos_i]
        if not isinstance(delID, (list, set, tuple)):
            delIDs = set([delID])
        else:
            delIDs = set(delID)
            
        posData.rp.update_regionprops_via_deletions(delIDs, lab)
        posData.IDs = posData.rp.IDs

    def _get_entire_depth_axis_from_2D_cutout(self, cutout):
        # cutout = (xl, xr), (yt, yb), z is always on the y position if depth axis is changed
        # cutout is in the current view; return grouped ranges in the order
        # expected by update_rp_get_bbox before conversion to NumPy bbox order.
        posData = self.data[self.pos_i]
        if self.isSegm3D:
            depthAxes = self.switchPlaneCombobox.depthAxes()
            if depthAxes == 'z':
                # cutout is (x, y) and we prepend the full z range.
                z_max = posData.SizeZ
                return ((0, z_max), cutout[0], cutout[1])
            if depthAxes == 'y':
                # cutout is (x, z); convert to (z, x, y).
                y_max = posData.SizeY
                return (cutout[1], cutout[0], (0, y_max))
            elif depthAxes == 'x':
                # cutout is (y, z); convert to (z, x, y).
                x_max = posData.SizeX
                return (cutout[1], (0, x_max), cutout[0])
        else:
            return cutout

    def _cutout_to_bbox(self, cutout):
        """
        Convert grouped view ranges into a flat bbox in NumPy array order.
        2D input: ((x_min, x_max), (y_min, y_max)) → (y_min, x_min, y_max, x_max)
        3D input: ((z_min, z_max), (y_min, y_max), (x_min, x_max)) → (z_min, y_min, x_min, z_max, y_max, x_max)
        """
        cutout = tuple(
            (min(r[0], r[1]), max(r[0], r[1])) for r in cutout
        )
        if len(cutout) > 2:
            (z_min, z_max), (y_min, y_max), (x_min, x_max) = cutout
            return (z_min, y_min, x_min, z_max, y_max, x_max)
        else:
            (x_min, x_max), (y_min, y_max) = cutout
            return (y_min, x_min, y_max, x_max)

    def _get_perc_cutout_from_total_img(self, cutout):
        posData = self.data[self.pos_i]
        single_timepoint_segm_size = posData.getSingleTimepointSegmSize()
        if len(cutout) > 2:
            size = (
                (cutout[0][1] - cutout[0][0]) 
                * (cutout[1][1] - cutout[1][0]) 
                * (cutout[2][1] - cutout[2][0])
            )
        else:
            depth = posData.SizeZ if self.isSegm3D else 1
            size = (
                depth
                * (cutout[0][1] - cutout[0][0]) 
                * (cutout[1][1] - cutout[1][0])
            )
        return size / single_timepoint_segm_size

    def update_rp_get_bbox(
            self, custom_bbox=None, use_bbox=False, use_curr_view=False,
            specific_IDs=None, add_frac_custom_bbox=0.05):
        """
        Returns an expanded bounding box (bbox) for the given IDs or custom_bbox.
        Returns False if not enough cells or cutout is too large.
        """
        posData = self.data[self.pos_i]
        if len(posData.rp.IDs) < gui_main_constants.RP_OPT_NUM_CELLS_MIN:
            return False
        if not isinstance(specific_IDs, (list, set, np.ndarray)) and specific_IDs is not None:
            specific_IDs = [specific_IDs]
        elif  specific_IDs is not None and len(specific_IDs) == 0:
            specific_IDs = None
        
        # Helper to merge bboxes
        def merge_bbox(b1, b2):
            if len(b1) == 4:
                return (
                    min(b1[0], b2[0]), min(b1[1], b2[1]),
                    max(b1[2], b2[2]), max(b1[3], b2[3])
                )
            else:
                return (
                    min(b1[0], b2[0]), min(b1[1], b2[1]), min(b1[2], b2[2]),
                    max(b1[3], b2[3]), max(b1[4], b2[4]), max(b1[5], b2[5])
                )

        bbox = None
        if custom_bbox or use_bbox:
            if not custom_bbox and use_bbox and specific_IDs is not None:
                rp_old = posData.rp
                for ID in specific_IDs:
                    b = rp_old.get_obj_from_ID(ID).bbox
                    bbox = b if bbox is None else merge_bbox(bbox, b)
            else:
                bbox = custom_bbox

            if bbox is None:
                return False
        
        elif use_curr_view:
            cutout = self.ax1ViewRange(integers=True)
            cutout = self._get_entire_depth_axis_from_2D_cutout(cutout)
            if len(cutout)==2:
                (xl, xr), (yt, yb) = cutout
            else:
                (z1, z2), (xl, xr), (yt, yb) = cutout
                z_min = min(z1, z2)
                z_max = max(z1, z2)
            x_min = min(xl, xr)
            x_max = max(xl, xr)
            y_min = min(yt, yb)
            y_max = max(yt, yb)
            bbox = (y_min, x_min, y_max, x_max) if len(cutout)==2 else (z_min, y_min, x_min, z_max, y_max, x_max)
        # Expand bbox by a fraction
        else:
            raise ValueError('''Either custom_bbox or use_bbox or use_curr_view must be provided as True''')
        
        if len(bbox) == 4:
            y_min, x_min, y_max, x_max = bbox
            offset_y = int((y_max - y_min) * add_frac_custom_bbox)
            offset_x = int((x_max - x_min) * add_frac_custom_bbox)
            offset_y = 1 if offset_y == 0 else offset_y
            offset_x = 1 if offset_x == 0 else offset_x
            size_y, size_x = posData.SizeY, posData.SizeX
            cutout = (
                (max(0, x_min - offset_x), min(size_x, x_max + offset_x)),
                (max(0, y_min - offset_y), min(size_y, y_max + offset_y))
            )
        else:
            z_min, y_min, x_min, z_max, y_max, x_max = bbox
            offset_z = int((z_max - z_min) * add_frac_custom_bbox)
            offset_y = int((y_max - y_min) * add_frac_custom_bbox)
            offset_x = int((x_max - x_min) * add_frac_custom_bbox)
            offset_z = 1 if offset_z == 0 else offset_z
            offset_y = 1 if offset_y == 0 else offset_y
            offset_x = 1 if offset_x == 0 else offset_x
            size_z, size_y, size_x = posData.SizeZ, posData.SizeY, posData.SizeX
            cutout = (
                (max(0, z_min - offset_z), min(size_z, z_max + offset_z)),
                (max(0, y_min - offset_y), min(size_y, y_max + offset_y)),
                (max(0, x_min - offset_x), min(size_x, x_max + offset_x))
            )

        perc_from_global = self._get_perc_cutout_from_total_img(cutout)
        if perc_from_global > gui_main_constants.RP_OPT_PERC_CUTOUT_MAX:
            return False
        return self._cutout_to_bbox(cutout)

    @exception_handler
    def update_rp(
            self, draw=True, debug=False, # og stuff
            assignments=None, deletionIDs=None, # very quick upates, rp labels are changed but rest is same
            specific_IDs=None, use_curr_view=False, use_bbox=False, preloaded_bbox=None, # for local updates to PR
            frame_i=None,
            wl_update=True, wl_track_og_curr=False, wl_update_lab=False, # wl stuff
            is_worker_thread=False
        ):
        """Update regionprops and related GUI/tracking metadata.

        This method supports full updates and optimized updates for specific
        scenarios (ID assignments, deletions, or local cutout-based refresh).
        It can operate on the current frame or on a provided frame index.

        Parameters
        ----------
        draw : bool, optional
            Not used anymore. Kept only because we don't know where else 
            update_rp is called with this kwarg.
        debug : bool, optional
            Reserved debug flag (currently not used directly in this method).
        assignments : dict, optional
            Mapping ``{old_ID: new_ID}`` used for fast regionprops updates when
            relabeling objects.
        deletionIDs : sequence of int, optional
            IDs to remove from regionprops using the deletion fast path.
        specific_IDs : int or sequence of int, optional
            Restrict updates to these IDs where supported.
        use_curr_view : bool, optional
            If ``True``, perform a local update in the currently visible area.
        use_bbox : bool, optional
            If ``True``, perform a local update using an automatically computed
            bounding box around ``specific_IDs``, based on bbox of specific_IDs.
        preloaded_bbox : tuple or bool, optional
            Precomputed bbox for local updates. If ``False``, force full update.
            If ``None``, bbox is computed when local update mode is active.
            Can be calculated using update_rp_get_bbox
        frame_i : int, optional
            Frame index to update. If ``None``, use the current frame.
        wl_update : bool, optional
            If ``True``, update the tracking whitelist after regionprops update.
        wl_track_og_curr : bool, optional
            Forwarded to :meth:`whitelistPropagateIDs` as ``track_og_curr``.
            If not deleted lab should be tracked against curr
        wl_update_lab : bool, optional
            Forwarded to :meth:`whitelistPropagateIDs` as ``update_lab``.

        Raises
        ------
        ValueError
            If mutually exclusive update modes are requested together:
            ``use_curr_view`` and ``use_bbox`` both ``True``, or more than one
            of ``assignments``, ``deletionIDs``, and local-update mode provided.

        Notes
        -----
        Updates ``posData.IDs`` when operating on the current frame and, when
        enabled, propagates whitelist additions/removals for tracking.
        """
        if is_worker_thread and use_curr_view:
            raise ValueError(
                '`use_curr_view` and `is_worker_thread` cannot be True '
                'at the same time, '
                'because `use_curr_view` cannot be True when this function '
                'is called by another thread.'
            )
        #updating rp is very clostly, as it deletes all the cashed
        if use_curr_view and use_bbox:
            raise ValueError('''use_curr_view and use_bbox cannot be True at the 
                             same time, as they are mutually exclusive''')
        local_rp_update = bool(use_curr_view or use_bbox or preloaded_bbox)
        posData = self.data[self.pos_i]
        # Update rp for current posData.lab (e.g. after any change)
        
        # check if only one of assignments, deletionIDs or only_current_view is given
        if sum([assignments is not None, 
                deletionIDs is not None, 
                local_rp_update, 
                ]) > 1:
            self.logger.warning(assignments is not None, deletionIDs is not None, local_rp_update)
            raise ValueError('Only one of assignments, deletionIDs, '
                             'use_curr_view or use_bbox, preloaded_bbox can be used '
                             'at a time')
        
        if (not isinstance(specific_IDs, (list, set, np.ndarray)) 
            and specific_IDs is not None
            ):
            specific_IDs = [specific_IDs]
        elif specific_IDs is not None and len(specific_IDs) == 0:
            specific_IDs = None

        curr_frame = posData.frame_i
        if frame_i is None:
            frame_i = curr_frame

        lab = None
        rp = None
        on_curr_frame = False
        unvisited = False
        if curr_frame == frame_i:
            lab = posData.lab
            rp = posData.rp
            on_curr_frame = True

        if lab is None:
            lab = posData.allData_li[frame_i]['labels']

        if lab is None:
            unvisited = True
            lab = posData.segm_data[frame_i]

        if rp is None:
            rp = posData.allData_li[frame_i]['regionprops']

        if rp is None:
            printl(f'''Warning: rp is None for pos {self.pos_i}, 
                   frame {frame_i}, Recomputing rp from labels.''')
            
            rp = self._acdcRegionProps(
                lab, precache_centroids=False
            )
        
        if wl_update and not unvisited:
            if self.whitelistOriginalIDs is None:
                old_IDs = rp.IDs.copy()
            else:
                old_IDs = self.whitelistOriginalIDs.copy()
                self.whitelistOriginalIDs = None
        elif self.whitelistOriginalIDs is None:
            self.whitelist_old_IDs = rp.IDs.copy()
        
        if assignments is not None:
            # {old_ID: new_ID, ...}
            rp.update_regionprops_via_assignments(assignments, lab)
        elif deletionIDs is not None:
            # (delID1, delID2, ...)
            rp.update_regionprops_via_deletions(deletionIDs, lab)
        elif local_rp_update:
            # first get current view
            if preloaded_bbox is None:
                # returns false if requirement for this opt. branch is not met
                preloaded_bbox = self.update_rp_get_bbox(use_bbox=use_bbox, 
                                                         use_curr_view=use_curr_view,
                                                         specific_IDs=specific_IDs)
            if preloaded_bbox is not False:
                rp.update_regionprops_via_cutout(
                    lab, cutout_bbox=preloaded_bbox, specific_IDs=specific_IDs
                )
                # if ID touches border but is not in specific_IDs, it will not be updated,
                # so be careful!
            else:
                rp.update_regionprops(
                    lab, specific_IDs_update_centroids=specific_IDs
                )
        else:
            rp.update_regionprops(
                lab,
                specific_IDs_update_centroids=specific_IDs if preloaded_bbox is not False else None, # since sometimes I preload
            )

        if on_curr_frame:
            posData.rp = rp
        else:
            posData.allData_li[frame_i]['regionprops'] = rp
            
        if on_curr_frame:
            posData.IDs = rp.IDs
        
        self.update_rp_metadata(draw=draw, frame_i=frame_i)        

        if not is_worker_thread and frame_i==posData.frame_i:
            # Update the overlay masks given the new rp
            self.currentLab2D = self.get_2Dlab(posData.lab, force_z=False)
            self.setOverlaySegmMasks()

        if not (wl_update and not unvisited):
            return

        # Update tracking whitelist
        if self.isSnapshot:
            accepted_lost_centroids = set()
        else:
            accepted_lost_centroids = self.getTrackedLostIDs()
        new_IDs = rp.IDs
        added_IDs = set(new_IDs) - set(old_IDs)
        removed_IDs = (
            set(old_IDs) 
            - set(new_IDs) 
            - set(accepted_lost_centroids)
        )

        self.whitelistPropagateIDs(
            IDs_to_add=added_IDs, IDs_to_remove=removed_IDs,
            curr_frame_only=True, IDs_curr=new_IDs,
            track_og_curr=wl_track_og_curr,
            curr_lab=lab, curr_rp=rp,
            update_lab=wl_update_lab
        )

    def getObjectsLab2D(self, IDs):
        """Get the displayed 2D labels (current z-slice or projection) with
        only the objects whose ID is in `IDs` (e.g., the "Keep IDs"
        selection)"""
        IDs = set(IDs)
        objsLab2D = np.zeros_like(self.currentLab2D)
        if not IDs:
            return objsLab2D

        posData = self.data[self.pos_i]
        for obj in posData.rp:
            if obj.label not in IDs:
                continue

            if not self.isObjInCurrSlice(obj.bbox):
                continue

            _slice = self.getObjSlice(obj.slice)
            _objMask = self.getObjImage(obj.image, obj.bbox)

            objsLab2D[_slice][_objMask] = obj.label

        return objsLab2D

    # @exec_time
    def update_rp_metadata(self, draw=True, frame_i=None):
        posData = self.data[self.pos_i]
        # Add to rp dynamic metadata (e.g. cells annotated as dead)
        if frame_i is None:
            frame_i = posData.frame_i

        if frame_i == posData.frame_i:
            rp = posData.rp
        else:
            rp = posData.allData_li[frame_i]['regionprops']
        
        if rp is None:
            return

        for i, obj in enumerate(rp):
            ID = obj.label
            obj.excluded = ID in posData.binnedIDs
            obj.dead = ID in posData.ripIDs

    # def _get_obj_for_current_view_rp(self, obj, posData):
    #     # 2D segmentation already has the correct regionprops object.
    #     if not self.isSegm3D or not posData.rp.is3D:
    #         return obj

    #     slicing = self.switchPlaneCombobox.depthAxes()
    #     zProjHow = self.zProjComboBox.currentText()
    #     if zProjHow == 'single z-slice':
    #         slice_selector = self.z_lab()
    #         slice_number = slice_selector[-1] if isinstance(slice_selector, tuple) else slice_selector
    #         obj_current_view = posData.rp.get_obj_from_slice_rp(
    #             obj.label, slice_number, slicing=slicing, warn=False
    #         )
    #         return obj_current_view or obj

    #     obj_current_view = posData.rp.get_obj_from_proj_rp(
    #         obj.label, kind='most_common', slicing=slicing, warn=False
    #     )
    #     return obj_current_view or obj

    # This funciton actually returns if its visible, not only on z slice
    # def isObjVisible(self, obj_bbox, debug=False, z_slice=None):
    #     if obj_bbox is None:
    #         return False

    #     cutout = self.ax1ViewRange(integers=True)
    #     (x1, x2), (y1, y2) = cutout
    #     view_x_min, view_x_max = min(x1, x2), max(x1, x2)
    #     view_y_min, view_y_max = min(y1, y2), max(y1, y2)

    #     if self.isSegm3D and len(obj_bbox) == 6:
    #         depthAxes = self.switchPlaneCombobox.depthAxes()
    #         min_z, min_y, min_x, max_z, max_y, max_x = obj_bbox
    #         if depthAxes == 'z':
    #             obj_x_min, obj_x_max = min_x, max_x
    #             obj_y_min, obj_y_max = min_y, max_y
    #         elif depthAxes == 'y':
    #             obj_x_min, obj_x_max = min_x, max_x
    #             obj_y_min, obj_y_max = min_z, max_z
    #         else:
    #             obj_x_min, obj_x_max = min_y, max_y
    #             obj_y_min, obj_y_max = min_z, max_z
    #     else:
    #         min_y, min_x, max_y, max_x = obj_bbox
    #         obj_x_min, obj_x_max = min_x, max_x
    #         obj_y_min, obj_y_max = min_y, max_y

    #     intersects_view = (
    #         obj_x_max > view_x_min and view_x_max > obj_x_min
    #         and obj_y_max > view_y_min and view_y_max > obj_y_min
    #     )
    #     if not intersects_view:
    #         return False

    #     if z_slice is None:
    #         z_slice = self.z_lab()

    #     if self.isSegm3D and len(obj_bbox) == 6:
    #         zProjHow = self.zProjComboBox.currentText()
    #         isZslice = zProjHow == 'single z-slice'
    #         if not isZslice:
    #             return True

    #         depthAxes = self.switchPlaneCombobox.depthAxes()
    #         min_z, min_y, min_x, max_z, max_y, max_x = obj_bbox

    #         if depthAxes == 'z':
    #             min_val, max_val = min_z, max_z
    #             val = z_slice
    #         elif depthAxes == 'y':
    #             min_val, max_val = min_y, max_y
    #             val = z_slice[-1]
    #         else:
    #             min_val, max_val = min_x, max_x
    #             val = z_slice[-1]

    #         return min_val <= val < max_val

    #     return True


    def getObjImage(self, obj_image, obj_bbox, z_slice=None):
        if self.isSegm3D and len(obj_bbox)==6:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if not isZslice:
                # required a projection
                return obj_image.max(axis=0)

            if z_slice is None:
                z_slice = self.z_lab()
            if isinstance(z_slice, tuple):
                z_slice = z_slice[-1]
                
            local_z = z_slice - obj_bbox[0]
            try:
                obi_image_2d = obj_image[local_z]
            except Exception as err:
                obi_image_2d = None
            return obi_image_2d
        else:
            return obj_image

    def getObjSlice(self, obj_slice):
        if self.isSegm3D:
            return obj_slice[1:3]
        else:
            return obj_slice

    def getObjFromID(self, ID):
        posData = self.data[self.pos_i]
        obj = posData.rp.get_obj_from_ID(ID, warn=False)
        if obj is None:
            # Object already cleared
            return
        return obj

    def isObjVisibleViewRange(self, bbox, viewRange=None):
        if self.isSegm3D and len(bbox) == 6:
            isObjInCurrSlice = self.isObjInCurrSlice(bbox)
            if not isObjInCurrSlice:
                return False

            # bbox has, no matter what projection, always the entire object    
            depthAxes = self.switchPlaneCombobox.depthAxes()
            z0, y0, x0, z1, y1, x1 = bbox
            if depthAxes == 'z':
                a0, a1, b0, b1 = x0, x1, y0, y1
            elif depthAxes == 'y':
                a0, a1, b0, b1 = x0, x1, z0, z1
            elif depthAxes == 'x':
                a0, a1, b0, b1 = y0, y1, z0, z1

        else:
            y0, x0, y1, x1 = bbox
            a0, a1, b0, b1 = x0, x1, y0, y1
            
        if viewRange is None:
            viewRange = self.ax1ViewRange()
            
        # 
        view_a0, view_a1 = viewRange[0] # x
        view_b0, view_b1 = viewRange[1] # y
        
        return not (a1 < view_a0 or a0 > view_a1 or b1 < view_b0 or b0 > view_b1)
