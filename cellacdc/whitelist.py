import os
import numpy as np
import skimage.measure
from . import printl, myutils
import json


class Whitelist:
    """
    A class to manage a whitelist.
    """
    def __init__(self, images_path, total_frames, segm_data, debug=True):
        self.images_path = images_path
        try:
            self.total_frames = range(total_frames)
        except TypeError:
            self.total_frames = total_frames

        self.total_frames_set = set(self.total_frames)
        self.last_frame = max(self.total_frames_set)

        self.segm_data = segm_data
        self._debug = debug

        self.originalLabs = None
        self.originalLabsIDs = None
        self.whitelistIDs = None
        self.whitelistOriginalFrame_i = None
        self.initialized_i = set()

    def __getitem__(self, index):
        return self.get(index)

    def __setitem__(self, index, value):
        self.whitelistIDs[index] = value

    def loadOGLabs(self, selected_path):
        segm_data = np.load(selected_path)
        segm_data = segm_data[segm_data.files[0]] # is this right?

        self.originalLabs = segm_data
        self.originalLabsIDs = [{obj.label for obj in skimage.measure.regionprops(frame)} for frame in segm_data]
    
    def load(self, whitelist_path):
        if not os.path.exists(whitelist_path):
            self.whitelistIDs = None
            return False
    
        any_whitelist_added = False
        with open(whitelist_path, 'r') as json_file:
            whitelist = json.load(json_file)
            wl_processed = dict()
            for key, val in whitelist.items():
                if val is None:
                    wl_processed[int(key)] = None
                elif val == "None": # if the string "none" is present in the json file, it will be converted to None
                    wl_processed[int(key)] = None
                else:
                    wl_processed[int(key)] = set(val)
                    any_whitelist_added = True
                    self.initialized_i.add(int(key))
        if any_whitelist_added:
            self.whitelistIDs = wl_processed
        else:
            self.whitelistIDs = None
            return False
            
        self.originalLabs = self.segm_data.copy()
        self.originalLabsIDs = [{obj.label for obj in 
                                 skimage.measure.regionprops(
                                     self.originalLabs[frame_i])
                                     }
                                 for frame_i in self.total_frames]
        return True

    def save(self, whitelist_path):
        if not self.whitelistIDs:
            return
        wl_copy = self.whitelistIDs.copy()
        for key, val in wl_copy:
            if val is None:
                wl_copy[key] = "None"
            else:
                wl_copy[key] = list(val)
        json.dump(wl_copy, open(whitelist_path, 'w+'), indent=4)
        
    def addNewIDs(self, frame_i,
                  allData_li,
                  ):
                #   per_frame_IDs=None,
                #   labs=None):
        
        IDs_curr_og_lab = self.originalLabsIDs[frame_i]
        IDs_prev_og_lab = self.originalLabsIDs[frame_i-1]

        new_IDs = IDs_curr_og_lab - IDs_prev_og_lab
        self.propagateIDs(IDs_to_add=new_IDs, 
                          curr_frame_only=True,
                          frame_i=frame_i,
                          allData_li=allData_li,
                        #   per_frame_IDs=per_frame_IDs,
                        #   labs=labs
                            )

    def IDsAccepted(self, whitelistIDs, frame_i, 
                    allData_li,
                    index_lab_combo=None,
                    rp=None,
                    IDs_curr=None,
                    curr_lab=None,
                    # labs=None
                    ):
        
        # if allData_li is None and labs is None:
        #     raise ValueError('Either allData_li or curr_labs must be provided')
        # elif allData_li is not None and labs is not None:
        #     raise ValueError('Either allData_li or curr_labs must be provided, not both')

        if self.whitelistIDs is None:
            self.whitelistIDs = {
                i: None for i in self.total_frames
            }

        if self.originalLabs is None:
            self.originalLabs = np.zeros_like(self.segm_data)
            self.originalLabsIDs = []
            for i in self.total_frames:
                try:
                    # if allData_li:
                    lab = allData_li[i]['labels'].copy()
                    # else:
                    #     lab = labs[i].copy()

                    self.originalLabs[i] = lab
                    self.originalLabsIDs.append({obj.label for obj in skimage.measure.regionprops(lab)})
                except AttributeError:
                    lab = self.segm_data[i].copy()
                    self.originalLabs[i] = lab
                    self.originalLabsIDs.append({obj.label for obj in skimage.measure.regionprops(lab)})

        whitelistIDs = set(whitelistIDs)
        self.propagateIDs(frame_i=frame_i,
                          new_whitelist=whitelistIDs,
                          try_create_new_whitelists=True, 
                          only_future_frames=True, 
                          force_not_dynamic_update=True,
                          allData_li=allData_li,
                          index_lab_combo=index_lab_combo,
                          IDs_curr=IDs_curr,
                            rp=rp,
                            curr_lab=curr_lab,
                        #   labs=labs,
                          )
        
    def get(self,frame_i,try_create_new_whitelists=False):
        try:
            old_whitelistIDs =self.whitelistIDs[frame_i]
        except Exception as e:
            if not try_create_new_whitelists:
                raise e
            elif e == KeyError:
                old_whitelistIDs = set()
            elif e == TypeError:
                old_whitelistIDs = set()
            else:
                raise e
            
        if old_whitelistIDs is None:
            old_whitelistIDs = set()
        else:
            old_whitelistIDs = set(old_whitelistIDs)
        
        return old_whitelistIDs
    
    def innitNewFrames(self, 
                       frame_i, 
                       force=False, 
                       ):
        """
        Initialize the whitelist for a new frame.
        """
        
        missing_frames = set(range(frame_i+1)) - self.initialized_i

        if self._debug:
            printl(missing_frames, self.initialized_i, frame_i)

        if frame_i not in self.initialized_i:
            new_frame = True
        else:
            new_frame = False

        if not force and not new_frame:
            return new_frame

        for i in missing_frames:
            self.whitelistIDs[i] = set()
            if i == 0:
                prev_wl = set()
            else:
                prev_wl = self.whitelistIDs[i-1]
            if prev_wl is None:
                prev_wl = set()
            
            available_IDs = self.originalLabsIDs[i]
            new_wl = prev_wl.intersection(available_IDs)
            if new_wl:
                self.whitelistIDs[i] = new_wl
            else:
                self.whitelistIDs[i] = set()
            
            self.initialized_i.add(i)

        if self._debug:
            printl('Whitelist IDs new frame (without adding new IDs):', self.whitelistIDs[frame_i])
        return new_frame
    
    def propagateIDs(self,
                     frame_i: int,
                    #  labs: np.ndarray = None,
                    #  per_frame_IDs: list = None,
                     allData_li: list,
                     new_whitelist: set[int] = None, 
                     IDs_to_add: set[int] = None,
                     IDs_to_remove: set[int] = None,
                     try_create_new_whitelists: bool = False,
                     curr_frame_only: bool = False,
                     force_not_dynamic_update: bool = False,
                     only_future_frames: bool = True,
                     allow_only_current_IDs: bool = True,
                     IDs_curr: set[int] = None,
                     rp: list = None,
                     new_frame: bool = False,
                     index_lab_combo: tuple = None,
                     curr_lab: np.ndarray = None,
                     ):
        """
        Propagates whitelist IDs across frames in the dataset. (Doesn't update labs)
        Should also be called when viewing a new frame!

        This function updates whitelist. If curr_frame_only is True, it only updates the
        whitelist of the current frame. If the frame changes, this function should be called 
        again to update the whitelist for the new frame (without this argument).
        It should also handle cases were this is not done, but this is less safe.
        Then, all the additions and removals are propagated to the other frames.
        If force_not_dynamic_update is True, the function will propagate the entire whitelist to 
        frames, and not only the IDs which were added or removed.

        Parameters
        ----------
        frame_i : int, optional
            The frame index for the propagation. If None, uses the current frame index.
        # labs : np.ndarray, optional
        #     All labels for the video. Mutually exclusive with `allData_li`.
        # per_frame_IDs : list, optional
        #     A list of IDs for each frame. Mutually exclusive with `allData_li`. If not provided, calculated from `labs`.
        all_data_li : list, optional
            See rest of ACDC. posData.allData_li
        new_whitelistIDs : set[int], optional
            A new set of whitelist IDs to replace the current whitelist. Cannot be 
            used together with `IDs_to_add` or `IDs_to_remove`.
        IDs_to_add : set[int], optional
            A set of IDs to add to the current whitelist.
        IDs_to_remove : set[int], optional
            A set of IDs to remove from the current whitelist.
        try_create_new_whitelists : bool, optional
            If True, creates new whitelist entries for frames that do not already 
            have them. Should only be necessary when its initialized...
        curr_frame_only : bool, optional
            If True, only updates the whitelist for the current frame. (See description of function)
        force_not_dynamic_update : bool, optional
            If True, disables dynamic updates to the whitelist. (See description of function)
        only_future_frames : bool, optional
            If True, propagates changes only to future frames.
        allow_only_current_IDs : bool, optional
            If True, only allows IDs that are present in the current frame to be added to the whitelist.
        IDs_curr : set[int], optional
            A set of IDs for the current frame. If None, will be calculated from the current labels.
        rp: list, optional
            Region properties for the current frame. If None, will be calculated from the current labels.
        lab: np.ndarray, optional
            Labels for the current frame. If None, will be calculated from the current labels.
        new_frame: bool, optional
            If True, indicates that a new frame is being processed. This is used to determine if the whitelist should be updated on that given frame.
        index_lab_combo: tuple, optional
            A tuple containing the frame index and the labels for that frame. If provided, this will be used instead of the `labs` parameter in lab.

        Raises
        ------
        ValueError
            If both `new_whitelistIDs` and `IDs_to_add`/`IDs_to_remove` are provided.

        Example
        -------
        To add IDs 5 and 6 to the whitelist for the current frame:
        ```python
        self.propagateIDs(IDs_to_add={5, 6}, curr_frame_only=True)
        ```
        Then when the frame changes:
        ```python
        self.propagateIDs()
        ```

        To replace the whitelist for frame 10 with a new set of IDs:
        ```python
        self.propagateIDs(new_whitelistIDs={1, 2, 3}, frame_i=10)
        ```
        This would also propagate the changes to all other frames.

        """
        #doesn't update the frame displayed, only wl

        # if allData_li is not None and per_frame_IDs is not None:
        #     raise ValueError('Cannot provide both allData_li and per_frame_IDs')
        # elif allData_li is None and per_frame_IDs is None and labs is None:
        #     raise ValueError('Either allData_li or per_frame_IDs or labs must be provided')
        # elif not allData_li and not per_frame_IDs:
        #     per_frame_IDs = [set() for _ in labs]
            
        if self._debug:
            printl('Propagating IDs...')
            myutils.print_call_stack()
            printl(new_whitelist, IDs_to_add, IDs_to_remove)

        # if labs is None and not allData_li and not IDs_curr:
        #     raise ValueError('Either labs or allData_li or IDs_curr/must be provided')
        # elif labs is not None and allData_li:
        #     raise ValueError('Cannot provide both labs and allData_li')  
        # elif 
        if IDs_curr:
            if self._debug:
                printl('Using IDs_curr')
            pass
        elif index_lab_combo and index_lab_combo[0] == frame_i:
            lab = index_lab_combo[1]
            if self._debug:
                printl('Using index_lab_combo')
            IDs_curr = {obj.label for obj in skimage.measure.regionprops(lab)}
        elif rp is not None:
            IDs_curr = {obj.label for obj in rp}
            if self._debug:
                printl('Using rp')
        elif curr_lab is not None:
            lab = curr_lab
            if self._debug:
                printl('Using curr_lab')
            IDs_curr = {obj.label for obj in skimage.measure.regionprops(lab)}


        else:
            IDs_curr = allData_li[frame_i]['IDs']
            if self._debug:
                printl('Using allData_li')
        
        IDs_curr = set(IDs_curr)
            
        # else:
        #     lab = labs[frame_i]
        #     if self._debug:
        #         printl('Using labs')

        new_frame = self.innitNewFrames(frame_i)
        last_frame_i = max(self.initialized_i) if self.initialized_i else 0

        # see what the siltation is with propagation
        propagate_after_curr_frame_only_flag = False
        if curr_frame_only:
            if self.whitelistOriginalFrame_i is None:
                self.whitelistOriginalFrame_i = frame_i
                self.whitelistOriginalIDs = self.whitelistIDs[frame_i].copy()
            elif self.whitelistOriginalFrame_i != frame_i:
                if self._debug:
                    printl('Frame changed, whitelist was not propagated, propagating...')
                self.propagateIDs(allData_li=allData_li,
                                   frame_i=self.whitelistOriginalFrame_i,
                                   index_lab_combo=index_lab_combo)
        else:
            if self.whitelistOriginalFrame_i is not None:
                if self.whitelistOriginalFrame_i != frame_i:
                    if self._debug:
                        printl('Frame changed, whitelist was not propagated, propagating...')
                    self.propagateIDs(allData_li=allData_li,
                                    frame_i=self.whitelistOriginalFrame_i,
                                    index_lab_combo=index_lab_combo)
                else:
                    propagate_after_curr_frame_only_flag = True
                self.whitelistOriginalFrame_i = None
        
        # see what the situation is with adding/removing IDs
        if new_whitelist and (IDs_to_add is not None or IDs_to_remove is not None):
            raise ValueError('Cannot provide both new_whitelist and IDs_to_add or IDs_to_remove')

        # figure out what old wl supposed to be...
        if force_not_dynamic_update:
            old_whitelist = set()
        elif propagate_after_curr_frame_only_flag:
            old_whitelist = self.whitelistOriginalIDs
        else:
            old_whitelist = self.get(frame_i,try_create_new_whitelists)

        # construct new_whitelist
        if new_whitelist is not None:
            new_whitelist = set(new_whitelist)
        else: # updated later if IDs_to_add or IDs_to_remove are provided
            new_whitelist = self.get(frame_i,try_create_new_whitelists)
            
        if IDs_to_add is not None or IDs_to_remove is not None:
            if IDs_to_add is None:
                IDs_to_add = set()
            else:
                IDs_to_add = set(IDs_to_add)
            if IDs_to_remove is None:
                IDs_to_remove = set()
            else:
                IDs_to_remove = set(IDs_to_remove)

            new_whitelist.update(IDs_to_add)
            new_whitelist -= IDs_to_remove

        if allow_only_current_IDs:
            IDs_curr.update(self.originalLabsIDs[frame_i])
            new_whitelist = new_whitelist.intersection(IDs_curr)
            if self._debug:
                printl(IDs_curr)
                printl(new_whitelist, old_whitelist)

        # get IDs to add/remove
        IDs_to_add = new_whitelist - old_whitelist
        IDs_to_remove = old_whitelist - new_whitelist

        if  IDs_to_add == IDs_to_remove == set():
            return

        if self._debug:
            printl(IDs_to_add, IDs_to_remove)
            
        # get the range of frames to update
        if new_frame:
            prop_to_frame_i = last_frame_i - 1
        else:
            prop_to_frame_i = last_frame_i

        if curr_frame_only:
            frames_range = [frame_i]
        elif only_future_frames:
            frames_range = range(frame_i, prop_to_frame_i + 1)
        else:
            frames_range = range(prop_to_frame_i + 1)

        if self._debug:
            printl(IDs_to_add, IDs_to_remove, frames_range)

        for i in frames_range:
            if IDs_to_add:
                IDs_og = self.originalLabsIDs[i]
                if frame_i == i:
                    IDs_curr_loc = IDs_curr
                else:
                    IDs_curr_loc = set(allData_li[i]['IDs'])

            old_whitelist = self.get(i,try_create_new_whitelists)

            if IDs_to_add:
                #                         intersection with...   all possible IDs           ...plus all old_whitelistIDs
                self.whitelistIDs[i] = IDs_to_add.intersection(IDs_curr_loc.union(IDs_og)) | old_whitelist
                # IDs_curr.union(IDs_og) are all possible IDs, IDs_to_add.intersection(IDs_curr.union(IDs_og)) is for finding all possible IDs which want ot be propagated
            if IDs_to_remove:
                self.whitelistIDs[i] = self.whitelistIDs[i] - IDs_to_remove

        if self._debug:
            printl(self.whitelistIDs[frame_i])
    
