"""Qt view adapter for the Whitelist feature."""

from __future__ import annotations

import os
import json
import numpy as np
import skimage.measure
from typing import Set, List, Tuple, Any
import time

from cellacdc import (
    printl, myutils, html_utils, apps, widgets, exception_handler, disableWindow, gui_utils, exec_time
)
from cellacdc.trackers.CellACDC import CellACDC_tracker
from cellacdc.whitelist import Whitelist
from cellacdc.viewmodels.whitelist_viewmodel import WhitelistViewModel


class WhitelistView:
    """Qt-facing adapter for the Whitelist feature."""

    LEGACY_METHODS = (
        'whitelistCheckOriginalLabels',
        'whitelistTrackOGagainstPreviousFrame_cb',
        'whitelistLoadOGLabs_cb',
        'whitelistLoadOGLabs',
        'whitelistViewOGIDs',
        'whitelistSetViewOGIDsToggle',
        'whitelistAddNewIDsToggled',
        'whitelistAddNewIDs',
        'whitelistIDsAccepted',
        'whitelistUpdateLab',
        'whitelistIDsUpdateText',
        'whitelistTrackOGCurr',
        'whitelistTrackCurrOG',
        'whitelistSyncIDsOG',
        'whitelistInitNewFrames',
        'whitelistPropagateIDs',
        'whitelistIDs_cb',
        'whitelistHighlightIDs',
        'whitelistIDsChanged',
        'whitelistUpdateTempLayer',
    )

    def __init__(self, host, view_model: WhitelistViewModel):
        object.__setattr__(self, 'host', host)
        object.__setattr__(self, 'view_model', view_model)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host', 'view_model'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def bind_legacy_methods(self):
        for name in self.LEGACY_METHODS:
            setattr(self.host, name, getattr(self, name))

    def whitelistCheckOriginalLabels(self, warning:bool=True, 
                                       frame_i:int=None):
        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return False
        
        if frame_i is None:
            frame_i = posData.frame_i

        if not self.view_model.check_original_labels(posData.whitelist, frame_i):
            txt = """
            No original labels are present for the current frame,
            this action cannot be performed."""
            self.logger.warning(txt)
            if not warning:
                return False
            msg = widgets.myMessageBox.warning(
                self, 'No original labels', txt,
            )
            
            return False
        else:
            return True

    @disableWindow
    def whitelistTrackOGagainstPreviousFrame_cb(self, signal_slot=None):
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        if not self.whitelistCheckOriginalLabels():
            return
        old_cell_IDs = posData.whitelist.originalLabsIDs[frame_i]
        prev_cell_IDs = posData.allData_li[frame_i-1]['IDs']
        self.whitelistTrackOGCurr(against_prev=True)
        new_cell_IDs = posData.whitelist.originalLabsIDs[frame_i]

        new_IDs = self.view_model.get_diff_ids(
            old_cell_IDs, set(prev_cell_IDs), new_cell_IDs
        )

        self.whitelistUpdateLab(
            track_og_curr=False, IDs_to_add=new_IDs,
        )

    def whitelistLoadOGLabs_cb(self):
        posData = self.data[self.pos_i]
        curr_seg_path = posData.segm_npz_path

        segmFilename = os.path.basename(curr_seg_path)
        custom_first = f"{segmFilename[:-4]}_not_whitelisted.npz"
        images_path = posData.images_path
        existingEndnames = [
            files for files in os.listdir(images_path) if files.endswith('.npz')
        ]
        if custom_first not in existingEndnames:
            custom_first = None

        infoText = html_utils.paragraph(
            'Select the segmentation file containing the original labels '
            'of the objects. Pleae note that the current saved "original" '
            'labels will be replaced with the new ones, but the filtered '
            'labels will be kept.'
        )

        win = apps.SelectSegmFileDialog(
            existingEndnames, images_path, parent=self, 
            basename=posData.basename, infoText=infoText,
            custom_first=custom_first
        )
        win.exec_()
        if win.cancel:
            self.logger.info('Loading original labels canceled.')
            return
        selected = win.selectedItemText
        self.logger.info(f'Loading original labels from {selected}...')
        self.whitelistLoadOGLabs(selected)

    @disableWindow
    def whitelistLoadOGLabs(self, selected:str):
        posData = self.data[self.pos_i]
        images_path = posData.images_path

        selected_path = os.path.join(images_path, selected)
        posData.whitelist.loadOGLabs(selected_path)
        
        self.whitelistIDsToolbar.viewOGToggle.setCheckable(True)

    @exception_handler
    @disableWindow
    def whitelistViewOGIDs(self, checked:bool):
        switch_to_og = checked and not self.viewOriginalLabels
        switch_to_seg = not checked and self.viewOriginalLabels
        
        if not switch_to_og and not switch_to_seg:
            return

        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return
        
        if posData.whitelist._debug:
            printl('whitelistViewOGIDs', checked)
     
        frame_i = posData.frame_i
        frames_range = self.view_model.get_frames_range(frame_i)

        self.store_data(autosave=False)
    
        if not self.whitelistCheckOriginalLabels():
            return
        if switch_to_og:
            self.setFrameNavigationDisabled(True, why='Viewing original labels')
            self.viewOriginalLabels = True

            for i in frames_range:
                posData.frame_i = i
                self.get_data()
                self.whitelistTrackOGCurr(frame_i=i)

                posData.lab = self.view_model.construct_og_frame(
                    pos_lab=posData.lab,
                    og_frame_base=posData.whitelist.originalLabs[i],
                    whitelist_ids=posData.whitelist.whitelistIDs[i],
                    og_ids=posData.whitelist.originalLabsIDs[i]
                )
                self.update_rp(wl_update=False)
                self.store_data(autosave=False)

            if frame_i > 0:
                missing_IDs = self.view_model.get_missing_ids(posData.IDs, posData.allData_li[frame_i-1]['IDs'])
                self.trackManuallyAddedObject(missing_IDs, isNewID=True, wl_update=False)

            self.setAllTextAnnotations()
            self.updateAllImages()

        elif switch_to_seg:
            self.viewOriginalLabels = False
            self.setFrameNavigationDisabled(False, why='Viewing original labels')

            for i in frames_range:
                posData.frame_i = i
                self.get_data()
                try:
                    posData.whitelist.originalLabs[i] = posData.lab.copy()
                    posData.whitelist.originalLabsIDs[i] = set(posData.IDs)
                except AttributeError:
                    lab = posData.segm_data[i].copy()
                    IDs = [obj.label for obj in skimage.measure.regionprops(lab)]
                    posData.whitelist.originalLabs[i] = lab
                    posData.whitelist.originalLabsIDs[i] = set(IDs)

                self.update_rp(wl_update=False)
                self.store_data(autosave=False)
                self.whitelistUpdateLab(frame_i=i)
                self.setAllTextAnnotations()
                self.updateAllImages()

    def whitelistSetViewOGIDsToggle(self, checked: bool):
        self.viewOriginalLabels = checked
        self.whitelistIDsToolbar.viewOGToggle.blockSignals(True)
        self.whitelistIDsToolbar.viewOGToggle.setChecked(checked)
        self.whitelistIDsToolbar.viewOGToggle.blockSignals(False)

    def whitelistAddNewIDsToggled(self, checked: bool):
        self.addNewIDsWhitelistToggle = checked
        if checked:
            self.df_settings.at['addNewIDsWhitelistToggle', 'value'] = 'Yes'
        else:
            self.df_settings.at['addNewIDsWhitelistToggle', 'value'] = 'No'
        self.df_settings.to_csv(self.settings_csv_path)
        if checked:
            self.whitelistAddNewIDs(ignore_not_first_time=True)
            self.whitelistPropagateIDs()
            self.updateAllImages()
            self.whitelistIDsUpdateText()

    def whitelistAddNewIDs(self, ignore_not_first_time:bool=False):
        mode = self.modeComboBox.currentText()        
        if mode != 'Segmentation and Tracking':
            return
    
        if not self.addNewIDsWhitelistToggle:
            return
        
        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return

        debug = posData.whitelist._debug

        if debug:
            printl('whitelistAddNewIDs')

        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        
        if self.get_last_tracked_i() > frame_i and not ignore_not_first_time:
            return
    
        if frame_i == 0:
            return

        if self.whitelistAddNewIDsFrame is not None and frame_i == self.whitelistAddNewIDsFrame:
            return
                
        self.whitelistAddNewIDsFrame = frame_i

        curr_lab = self.get_curr_lab()

        posData.whitelist.addNewIDs(frame_i=frame_i,
                                    allData_li=posData.allData_li,
                                    IDs_curr=posData.IDs,
                                    curr_lab=curr_lab)        

    def whitelistIDsAccepted(self, 
                             whitelistIDs: Set[int] | List[int]):
        self.storeUndoRedoStates(False)

        self.whitelistIDsToolbar.viewOGToggle.setCheckable(True)
        self.whitelistSetViewOGIDsToggle(False)
        self.setFrameNavigationDisabled(False, why='Viewing original labels')
        
        self.store_data(autosave=False)

        posData = self.data[self.pos_i]

        if not posData.whitelist:
            posData.whitelist = Whitelist(
                total_frames=posData.SizeT,
            )
        
        if posData.whitelist._debug:
            printl('whitelistIDsAccepted', whitelistIDs)

        whitelistIDs = set(whitelistIDs)
        IDs_curr = set(posData.IDs)
        
        posData.whitelist.IDsAccepted(
            whitelistIDs,
            segm_data=posData.segm_data,
            frame_i=posData.frame_i,
            allData_li=posData.allData_li,
            IDs_curr=IDs_curr,
            curr_lab=posData.lab,
        )
                
        self.whitelistUpdateLab(track_og_curr=True)
        self.whitelistIDsUpdateText()
        self.keepIDsTempLayerLeft.clear()

    def whitelistUpdateLab(self, frame_i: int=None,
        track_og_curr=False, new_frame:bool=False,
        IDs_to_add:List[int] | Set[int]=None,
        IDs_to_remove:List[int]|Set[int]=None,
        ): 
        got_data = False
        benchmark = False
        if benchmark:
            ts = [time.perf_counter()]
            titles = [
                '',
                'store_data',
                'whitelistSetViewOGIDsToggle',
                'get_data',
                'get what to add/remove',
                'track_og_curr',
                'get current lab',
                'add/remove IDs',
                'store data',
                'update images',
                ]
            
        mode = self.modeComboBox.currentText()
        if mode != 'Segmentation and Tracking':
            return
        
        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return

        if frame_i is None:
            frame_i = posData.frame_i
            og_frame_i = frame_i
        else:
            og_frame_i = posData.frame_i
            posData.frame_i = frame_i
            
        debug = posData.whitelist._debug
        if debug:
            printl('whitelistUpdateLab', frame_i, og_frame_i)
            from cellacdc import debugutils
            debugutils.print_call_stack()

        if benchmark:
            ts.append(time.perf_counter())

        self.whitelistSetViewOGIDsToggle(False)

        if benchmark:
            ts.append(time.perf_counter())
            
        if self.whitelistCheckOriginalLabels(warning=False, frame_i=frame_i):
            og_lab = posData.whitelist.originalLabs[frame_i]
        else:
            og_lab = None
        if benchmark:
            ts.append(time.perf_counter())

        whitelist = posData.whitelist.get(frame_i=frame_i)
        IDs_to_add_remove_provided = IDs_to_add is not None or IDs_to_remove is not None
        if not IDs_to_add_remove_provided:
            self.get_data()
            got_data = True
            missing_IDs, to_be_removed_IDs = self.view_model.get_whitelist_missing_and_removed_ids(
                whitelist, set(posData.IDs)
            )
        else:
            missing_IDs = list(IDs_to_add) if IDs_to_add is not None else []
            to_be_removed_IDs = list(IDs_to_remove) if IDs_to_remove is not None else []

        if benchmark:
            ts.append(time.perf_counter())
        
        if not missing_IDs and not to_be_removed_IDs:
            if og_frame_i != frame_i:
                posData.frame_i = og_frame_i
            if got_data and og_frame_i != frame_i:
                self.get_data()
            if benchmark:
                print('No IDs to add/remove')
                ts.append(time.perf_counter())
                indx = titles.index('track_og_curr')
                titles[indx + 1] = 'store_data'
                time_taken = time.perf_counter() - ts[0]
                print(f'\nTotal time for whitelistUpdateLab: {time_taken:.2f}s')
                for i in range(1, len(ts)):
                    time_taken = ts[i] - ts[i-1]
                    print(f'Time taken for {titles[i]}: {time_taken:.2f}s')
                print('')
            return
        
        if not got_data and og_frame_i != frame_i:
            self.get_data()
            got_data = True
        
        if benchmark:
            ts.append(time.perf_counter())

        if missing_IDs and track_og_curr and not new_frame:
            self.whitelistTrackOGCurr(frame_i=frame_i, 
                                      lab = posData.lab,
                                      rp = posData.rp)
        
        if debug:
            printl(missing_IDs, to_be_removed_IDs)

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
        if curr_lab is None:
            printl('No current lab?')
            curr_lab = np.zeros_like(posData.segm_data[0])

        if benchmark:
            ts.append(time.perf_counter())

        curr_lab = self.view_model.apply_id_mask(
            curr_lab, og_lab, missing_IDs, to_be_removed_IDs
        )

        if benchmark:
            ts.append(time.perf_counter())
        
        posData.lab = curr_lab

        self.update_rp(wl_update=False)
        self.store_data()

        if benchmark:
            ts.append(time.perf_counter())
        if og_frame_i != frame_i:
            posData.frame_i = og_frame_i
            self.get_data()
        
        self.updateAllImages()
        self.setAllTextAnnotations()

        if benchmark:
            ts.append(time.perf_counter())
            time_taken = time.perf_counter() - ts[0]
            print(f'\nTotal time for whitelistUpdateLab: {time_taken:.2f}s')
            for i in range(1, len(ts)):
                time_taken = ts[i] - ts[i-1]
                print(f'Time taken for {titles[i]}: {time_taken:.2f}s')
            print('')

    def whitelistIDsUpdateText(self):
        mode = self.modeComboBox.currentText()
        if mode != 'Segmentation and Tracking':
            return

        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return
        
        if posData.whitelist._debug:
            printl('whitelistIDsUpdateText')
        
        frame_i = posData.frame_i
        whitelist = posData.whitelist.get(frame_i=frame_i)

        self.whitelistIDsToolbar.whitelistLineEdit.setText(whitelist)

    def whitelistTrackOGCurr(self, frame_i:int=None, 
                             against_prev:bool=False,
                             lab:np.ndarray=None,
                             rp:list=None,
                             IDs: Set[int] | List[int] =None):
        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return

        debug = posData.whitelist._debug

        if debug:
            from cellacdc import debugutils
            debugutils.print_call_stack(depth=2)
            printl('whitelistTrackOGCurr', against_prev)

        if against_prev and (rp is not None or lab is not None):
            raise ValueError('Cannot provide both rp and lab when tracking'
                             ' against previous frame.'
            'Instead only provide rp and lab, and dont set against_prev.')

        if frame_i is None:
            frame_i = posData.frame_i

        if against_prev and frame_i == 0:
            return
    
        if not self.whitelistCheckOriginalLabels(warning=False, 
                        frame_i=frame_i):
            if debug:
                printl('No original labels, cannot track.')
            return

        og_frame_i = posData.frame_i

        if lab is not None and not rp:
            rp = skimage.measure.regionprops(lab)
        
        changed_frame = False
        if lab is None:
            if debug:
                printl('No lab and no rp provided.')
            if against_prev:
                rp = posData.allData_li[frame_i-1]['regionprops']
                lab = posData.allData_li[frame_i-1]['labels']
            else:
                if frame_i != og_frame_i:
                    self.store_data(autosave=False)
                    posData.frame_i = frame_i
                    self.get_data()
                    changed_frame = True
                rp = posData.rp
                lab = posData.lab
        og_lab = posData.whitelist.originalLabs[frame_i]
        og_rp = skimage.measure.regionprops(og_lab)

        denom_overlap_matrix = 'union' if not against_prev else 'area_prev'

        og_lab = CellACDC_tracker.track_frame(
                lab, rp, og_lab, og_rp,
                denom_overlap_matrix=denom_overlap_matrix,
                posData = posData,
                setBrushID_func=self.setBrushID,
                IDs=IDs,
        )

        posData.whitelist.originalLabs[frame_i] = og_lab
        posData.whitelist.originalLabsIDs[frame_i] = {obj.label for obj in skimage.measure.regionprops(og_lab)}

        if changed_frame:
            posData.frame_i = og_frame_i
            self.get_data()

    def whitelistTrackCurrOG(self, frame_i:int=None, against_prev:bool=False):
        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return

        if posData.whitelist._debug:
            printl('whitelistTrackCurrOG', frame_i, against_prev)

        if frame_i is None:
            frame_i = posData.frame_i

        if against_prev and frame_i == 0:
            return

        og_frame = posData.frame_i
        if frame_i != og_frame:
            self.store_data(autosave=False)
            posData.frame_i = frame_i
            self.get_data()
        
        lab = posData.lab
        rp = posData.rp
        
        if not self.whitelistCheckOriginalLabels(warning=False,
                        frame_i=frame_i if not against_prev else frame_i-1):
            if posData.whitelist._debug:
                printl('No original labels, cannot track.')
            return

        if against_prev:
            og_lab = posData.whitelist.originalLabs[frame_i-1]
        else:
            og_lab = posData.whitelist.originalLabs[frame_i]

        og_rp = skimage.measure.regionprops(og_lab)

        denom_overlap_matrix = 'union' if not against_prev else 'area_prev'

        lab = CellACDC_tracker.track_frame(
            og_lab, og_rp, lab, rp,
            denom_overlap_matrix=denom_overlap_matrix,
            posData = posData,
            setBrushID_func=self.setBrushID
        )

        posData.lab = lab

        self.update_rp(wl_update=False)
        self.store_data(autosave=False)

        if frame_i != og_frame:
            posData.frame_i = og_frame
            self.get_data()

    def whitelistSyncIDsOG(self, 
                           frame_is: List[int]=None,
                           against_prev: bool=False,):
        posData = self.data[self.pos_i]
        if frame_is is None:
            frame_is = range(posData.SizeT)

        for frame_i in frame_is:
            self.whitelistTrackOGCurr(frame_i=frame_i, against_prev=against_prev)

    def whitelistInitNewFrames(self, frame_i:int=None, force:bool=False):
        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return False, []

        if frame_i is None:
            frame_i = posData.frame_i
        
        if posData.whitelist._debug:
            printl('whitelistInitNewFrames', frame_i, force)

        if frame_i not in posData.whitelist.initialized_i:
            self.whitelistTrackOGCurr(frame_i=frame_i, against_prev=True)

        new_frame, update_frames = posData.whitelist.initNewFrames(
            frame_i=frame_i, force=force)

        self.whitelistAddNewIDs()
        return new_frame, update_frames                    

    def whitelistPropagateIDs(self, 
                              new_whitelist: Set[int] | List[int] = None, 
                              IDs_to_add: Set[int] = None,
                              IDs_to_remove: Set[int] = None,
                              frame_i: int = None,
                              try_create_new_whitelists: bool = False,
                              curr_frame_only: bool = False,
                              force_not_dynamic_update: bool = False,
                              only_future_frames: bool = True,
                              allow_only_current_IDs: bool = False,
                              track_og_curr: bool = True,
                              IDs_curr: Set[int] | List[int] = None,
                              index_lab_combo: Tuple[int, np.ndarray] = None,
                              curr_rp: list = None,
                              curr_lab: np.ndarray = None,
                              store_data: bool = True,
                              update_lab: bool = False,
                              ):
        try:
            IDs_curr = IDs_curr.copy()
        except AttributeError:
            pass
            
        IDs_curr = set(IDs_curr) if IDs_curr is not None else None

        posData = self.data[self.pos_i]
        debug = posData.whitelist._debug if posData.whitelist is not None else False

        if debug:
            printl('Propagating IDs...')
            from cellacdc import debugutils
            debugutils.print_call_stack()
            printl(new_whitelist, IDs_to_add, IDs_to_remove)

        if posData.whitelist is None:
            return

        if frame_i is None:
            frame_i = posData.frame_i

        new_frame, update_frames_init = self.whitelistInitNewFrames(frame_i=frame_i)

        if new_frame:
            self.update_rp(wl_update=False)

        update_frames = posData.whitelist.propagateIDs(
            frame_i,
            posData.allData_li,
            new_whitelist=new_whitelist,
            IDs_to_add=IDs_to_add,
            IDs_to_remove=IDs_to_remove,
            try_create_new_whitelists=try_create_new_whitelists,
            curr_frame_only=curr_frame_only,
            force_not_dynamic_update=force_not_dynamic_update,
            only_future_frames=only_future_frames,
            allow_only_current_IDs=allow_only_current_IDs,
            IDs_curr=IDs_curr,
            index_lab_combo=index_lab_combo,
            curr_rp=curr_rp,
            curr_lab=curr_lab,
        )
        if update_lab:
            update_frames = update_frames_init + update_frames
        else:
            update_frames = update_frames_init
            
        self.whitelistIDsUpdateText()
        if store_data:
            self.store_data(autosave=False)

        for frame_i, IDs_to_add, IDs_to_remove, new_frame in update_frames:
            self.whitelistUpdateLab(frame_i=frame_i, track_og_curr=track_og_curr, 
                                    new_frame=new_frame, IDs_to_add=IDs_to_add, 
                                    IDs_to_remove=IDs_to_remove, )

    def whitelistIDs_cb(self, checked:bool):
        if checked:
            self.initKeepObjLabelsLayers()
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.whitelistIDsButton)
            self.connectLeftClickButtons()
            
        self.whitelistIDsToolbar.setVisible(checked)
        self.whitelistHighlightIDs(checked)
        self.whitelistIDsUpdateText()
        self.whitelistUpdateTempLayer()

        if not checked:
            self.setLostNewOldPrevIDs()
            self.updateAllImages()

    def whitelistHighlightIDs(self, checked:bool=True):
        if not checked:
            self.removeHighlightLabelID()
            return
        
        posData = self.data[self.pos_i]

        if posData.whitelist is None:
            if not hasattr(self, 'tempWhitelistIDs'):
                self.tempWhitelistIDs = set()
                current_whitelist = self.tempWhitelistIDs
            else:
                current_whitelist = self.tempWhitelistIDs
        else:
            current_whitelist = posData.whitelist.get(
                frame_i=posData.frame_i)
        
        for ID in current_whitelist:
            self.highlightLabelID(ID)
        
    def whitelistIDsChanged(self, 
                             whitelistIDs: Set[int] | List[int], 
                             debug: bool=False):
        if not self.whitelistIDsButton.isChecked():
            return
        
        posData = self.data[self.pos_i]

        if posData.whitelist:
            debug = posData.whitelist._debug
        if debug:
            printl('whitelistIDsChanged', whitelistIDs)

        if posData.whitelist is None:
            wl_init = False
            if not hasattr(self, 'tempWhitelistIDs'):
                self.tempWhitelistIDs = set()
                current_whitelist = self.tempWhitelistIDs
            else:
                current_whitelist = self.tempWhitelistIDs
        else:
            wl_init = True
            current_whitelist = posData.whitelist.get(
                frame_i=posData.frame_i)

        current_whitelist_copy = current_whitelist.copy()
        if not hasattr(posData, 'originalLabsIDs') or posData.whitelist.originalLabsIDs is None:
            possible_IDs = posData.IDs.copy()
        else:
            if not self.whitelistCheckOriginalLabels(warning=False):
                possible_IDs = set(posData.IDs)
            else:
                possible_IDs = posData.whitelist.originalLabsIDs[posData.frame_i]
                possible_IDs.update(posData.IDs)

        # Delegate validation of existing IDs to viewmodel/model
        filtered_whitelist, isAnyIDnotExisting = self.view_model.filter_existing_ids(
            whitelistIDs, possible_IDs
        )

        # Apply changes based on filtered_whitelist
        for ID in filtered_whitelist:
            if ID not in current_whitelist_copy:
                current_whitelist.add(ID)
                self.highlightLabelID(ID)

        for ID in current_whitelist_copy:
            if ID not in possible_IDs:
                continue
            if ID not in whitelistIDs:
                current_whitelist.remove(ID)
                self.removeHighlightLabelID(IDs=[ID])

        if wl_init:
            posData.whitelist.whitelistIDs[posData.frame_i] = current_whitelist
        else:
            self.tempWhitelistIDs = current_whitelist

        self.whitelistUpdateTempLayer()
        if isAnyIDnotExisting:
            self.whitelistIDsToolbar.whitelistLineEdit.warnNotExistingID()
        else:
            self.whitelistIDsToolbar.whitelistLineEdit.setInstructionsText()

    def whitelistUpdateTempLayer(self):
        if not self.whitelistIDsButton.isChecked():
            self.keepIDsTempLayerLeft.clear()
            return

        if not hasattr(self, 'keptLab'):
            self.keptLab = np.zeros_like(self.currentLab2D)
            keptLab = self.keptLab
        else:
            keptLab = self.keptLab
            keptLab[:] = 0

        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            if not hasattr(self, 'tempWhitelistIDs'):
                self.tempWhitelistIDs = set()
                current_whitelist = self.tempWhitelistIDs
            else:
                current_whitelist = self.tempWhitelistIDs
        else:
            current_whitelist = posData.whitelist.get(posData.frame_i)

        for obj in posData.rp:
            if obj.label not in current_whitelist:
                continue

            if not self.isObjVisible(obj.bbox):
                continue

            _slice = self.getObjSlice(obj.slice)
            _objMask = self.getObjImage(obj.image, obj.bbox)

            keptLab[_slice][_objMask] = obj.label

        self.keepIDsTempLayerLeft.setImage(keptLab, autoLevels=False)
