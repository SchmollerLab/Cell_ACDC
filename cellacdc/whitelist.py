import os
import numpy as np
import skimage.measure
from . import printl, myutils
import json
from typing import Set, List, Tuple
import time


class Whitelist:
    """A class to manage the whitelist of IDs for a video.
    """
    def __init__(self, total_frames: int | list | set, debug=False):
        """Initializes the whitelist with the total number of frames.
        The whitelist is a dictionary with the frame index 
        as the key and a set of IDs as the value.
        Also the original not whitelisted labs are stored in the originalLabs variable.

        Parameters
        ----------
        total_frames : int | list | set
            total frames. range(total_frames) if int, else total_frames
        debug : bool, optional
            debug with a lot of prints, also in gui.py, by default False
        """
        try:
            self.total_frames = range(total_frames)
        except TypeError:
            self.total_frames = total_frames

        self.total_frames_set = set(self.total_frames)
        self.last_frame = max(self.total_frames_set)

        self._debug = debug

        self.originalLabs = None
        self.originalLabsIDs = None
        self.whitelistIDs = None
        self.whitelistOriginalFrame_i = None
        self.initialized_i = set()

    def __getitem__(self, index:int):
        """Gets a whitelist for a given index.

        Parameters
        ----------
        index : int
            index of the requested whitelist

        Returns
        -------
        set
            set of whitelist IDs for the given index
        """
        return self.get(index)

    def __setitem__(self, index:int, value:set):
        """Sets a whitelist for a given index.

        Parameters
        ----------
        index : int
            index of the whitelist to set
        value : set
            set of whitelist IDs to set for the given index
        """
        self.whitelistIDs[index] = set(value)

    def loadOGLabs(self, selected_path:str=None, og_data:np.ndarray=None):
        """Loads the original labels from a .npz file, 
        or from the provided og_data.

        Parameters
        ----------
        selected_path : str, optional
            path to be loaded, by default None
        og_data : np.ndarray, optional
            alternative original data, by default None
        """
        if og_data is None:
            og_data = np.load(selected_path)
            og_data = og_data[og_data.files[0]] # is this right?

        self.originalLabs = og_data
        self.originalLabsIDs = [{obj.label for obj in skimage.measure.regionprops(frame)} for frame in og_data]
    
    def saveOGLabs(self, save_path:str):
        """Saves the original labels to a .npz file.

        Parameters
        ----------
        save_path : str
            desired save path for the original labels
        """
        # original_frames = np.array(list(self.originalLabs.values())) 
        # the above is not necessary anymore, 
        #since I changed the originalLabs to be a np.ndarray
        np.savez_compressed(save_path, self.originalLabs)
    
    def load(self, whitelist_path:str, 
             segm_data:np.ndarray,
             allData_li:list=None,
             ):
        """Loads the whitelist from a json file.
        If the file does not exist, it initializes the whitelist to None.
        If the file exists, it loads the whitelist and initializes 
        the originalLabs variable.

        Parameters
        ----------
        whitelist_path : str
            path to the whitelist json file (should be in accordance to the 
            one provided in save)
        segm_data : np.ndarray
            segmentation data for the video
        allData_li : list, optional
            See rest of ACDC. posData.allData_li, by default None
            Used to get og IDs

        Returns
        -------
        bool
            True if the whitelist was loaded successfully, False if not
        """
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
            
        self.originalLabs = segm_data.copy()

        self.originalLabsIDs = [None] * len(self.total_frames)
        for frame_i in self.total_frames:
            try:
                originalLabsIDs_frame_i = set(allData_li[frame_i]['IDs'])
            except AttributeError:
                originalLabsIDs_frame_i = None
            if originalLabsIDs_frame_i is None:
                originalLabsIDs_frame_i = {obj.label for obj in skimage.measure.regionprops(
                                        self.originalLabs[frame_i])}
            self.originalLabsIDs[frame_i] = originalLabsIDs_frame_i
        return True

    def save(self, whitelist_path:str):
        """Saves the whitelist to a json file.
        If the whitelist is None, it will not be saved.
        Make sure that the path is in accordance to the one provided in load.

        Parameters
        ----------
        whitelist_path : str
            path to the whitelist json file (should be in accordance to the 
            one provided in load)
        """
        if not self.whitelistIDs:
            return
        wl_copy = self.whitelistIDs.copy()
        for key, val in wl_copy.items():
            if val is None:
                wl_copy[key] = "None"
            else:
                wl_copy[key] = list(val)
        json.dump(wl_copy, open(whitelist_path, 'w+'), indent=4)
        
    def addNewIDs(self, frame_i:int,
                  allData_li: list,
                  IDs_curr: List[int] | Set[int]=None,
                  index_lab_combo: Tuple[int, np.ndarray]=None,
                  curr_rp: list=None,
                  curr_lab: np.ndarray=None,
                #   per_frame_IDs=None,
                #   labs=None
                  ):
        """Adds new IDs to the whitelist for a given frame based on the
        original labels. The IDs are added to the whitelist for the current frame.
        Also propagates.

        Parameters
        ----------
        frame_i : int
            for which frame to add the IDs
        allData_li : list
            passed to self.propagateIDs(), see rest of ACDC: posData.allData_li
        IDs_curr : list | set, optional
            Currently present IDs,  passed to self.propagateIDs(). by default None
        index_lab_combo: Tuple[int, np.ndarray]=None,
            Combination of frame_i and current frame, 
            passed to self.propagateIDs(), by default None
        curr_rp : list, optional
            Region properties for the current frame, passed to self.propagateIDs(). by default None
        curr_lab : np.ndarray, optional
            Labels for the current frame, passed to self.propagateIDs(). by default None
        """
        
        IDs_curr_og_lab = self.originalLabsIDs[frame_i]
        IDs_prev_og_lab = self.originalLabsIDs[frame_i-1]

        new_IDs = IDs_curr_og_lab - IDs_prev_og_lab
        self.propagateIDs(IDs_to_add=new_IDs, 
                          curr_frame_only=True,
                          frame_i=frame_i,
                          allData_li=allData_li,
                          IDs_curr=IDs_curr,
                          index_lab_combo=index_lab_combo,
                          allow_only_current_IDs=False,
                          curr_rp=curr_rp,
                          curr_lab=curr_lab,
                        #   per_frame_IDs=per_frame_IDs,
                        #   labs=labs
                            )

    def IDsAccepted(self, 
                    whitelistIDs: Set[int] | List[int], 
                    frame_i: int, 
                    allData_li: list,
                    segm_data: np.ndarray,
                    curr_lab: np.ndarray=None,
                    index_lab_combo: Tuple[int, np.ndarray]=None,
                    IDs_curr: Set[int] | List[int]=None,
                    curr_rp: list=None,
                    # labs=None
                    ):
        """Called if the user accepted IDs. 
        This can also be called if one wants forced propagation of IDs.

        Parameters
        ----------
        whitelistIDs : Set[int] | List[int]
            The IDs in the whitelist.
        frame_i : int
            The frame index for the whitelist.
        allData_li : list
            See rest of ACDC. posData.allData_li
        segm_data : np.ndarray
            The segmentation data for the video. Fallback to when allData_li is not provided.
        curr_lab : np.ndarray, optional
            Labels for the current frame. Use instead of allData_li/segm_data 
            for current frame_i
            Also passed to self.propagateIDs(), by default None
        index_lab_combo : Tuple[int, np.ndarray], optional
            Combination of frame_i and current frame, 
            passed to self.propagateIDs(), by default None
        IDs_curr : list | set, optional
            Currently present IDs,  passed to self.propagateIDs(), by default None
        curr_rp : list, optional
            Region properties for the current frame, passed to self.propagateIDs(), by default None
        """
        
        # if allData_li is None and labs is None:
        #     raise ValueError('Either allData_li or curr_labs must be provided')
        # elif allData_li is not None and labs is not None:
        #     raise ValueError('Either allData_li or curr_labs must be provided, not both')

        if self.whitelistIDs is None:
            self.whitelistIDs = {
                i: None for i in self.total_frames
            }

        if IDs_curr:
            if self._debug:
                printl('Using IDs_curr')
            try:
                IDs_curr = IDs_curr.copy()
            except AttributeError:
                pass
            IDs_curr = set(IDs_curr)
        elif index_lab_combo and index_lab_combo[0] == frame_i:
            lab = index_lab_combo[1]
            if self._debug:
                printl('Using index_lab_combo')
            IDs_curr = {obj.label for obj in skimage.measure.regionprops(lab)}
        elif curr_rp is not None:
            IDs_curr = {obj.label for obj in curr_rp}
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

        if self.originalLabs is None:
            self.originalLabs = np.zeros_like(segm_data)
            self.originalLabsIDs = []
            for i in self.total_frames:
                if i == frame_i and curr_lab is not None:
                    self.originalLabs[i] = curr_lab.copy()
                    self.originalLabsIDs.append(IDs_curr)
                    continue
                try:
                    # if allData_li:
                    lab = allData_li[i]['labels'].copy()
                    # else:
                    #     lab = labs[i].copy()

                    self.originalLabs[i] = lab
                    self.originalLabsIDs.append(allData_li[i]['IDs'])
                except AttributeError:
                    lab = segm_data[i].copy()
                    self.originalLabs[i] = lab
                    self.originalLabsIDs.append({obj.label for obj in skimage.measure.regionprops(lab)})

        whitelistIDs = set(whitelistIDs)
        self.propagateIDs(frame_i,
                          allData_li,
                          new_whitelist=whitelistIDs,
                          try_create_new_whitelists=True, 
                          force_not_dynamic_update=True,
                          index_lab_combo=index_lab_combo,
                          IDs_curr=IDs_curr,
                          curr_rp=curr_rp,
                          curr_lab=curr_lab,
                        #   labs=labs,
                          )
        
    def get(self,frame_i:int,try_create_new_whitelists:bool=False):
        """Gets the whitelist for a given frame index.
        If the whitelist is not initialized, and try_create_new_whitelists is True,
        it will create a new whitelist empty for that frame.

        Parameters
        ----------
        frame_i : int
            The frame index for the whitelist.
        try_create_new_whitelists : bool, optional
            If new empty whitelist should be tried and created, by default False

        Returns
        -------
        set
            The whitelist for the given frame index.

        """

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
                       frame_i:int, 
                       force:bool=False, 
                       ):
        """Initialize the whitelists for all new frame.
        All frames up to and including frame_i will be initialized.
        Unless forced, it will only initialize the whitelist if the frame is not 
        already initialized, (tracked with self.initialized_i).

        Parameters
        ----------
        frame_i : int
            The frame index for where the initialization should be done.
        force : bool, optional
            If the frame_i (only this frame_i in that case) 
            should be reinnit, by default False

        Returns
        -------
        bool
            True if a new frame was initialized, False if not.
        """
        
        missing_frames = set(range(frame_i+1)) - self.initialized_i

        if self._debug:
            printl(missing_frames, self.initialized_i, frame_i)

        if frame_i not in self.initialized_i:
            new_frame = True
        else:
            new_frame = False

        if not force and not missing_frames:
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
                     allData_li: list,
                     new_whitelist: Set[int] | List[int] = None, 
                     IDs_to_add: Set[int] = None,
                     IDs_to_remove: Set[int] = None,
                     try_create_new_whitelists: bool = False,
                     curr_frame_only: bool = False,
                     force_not_dynamic_update: bool = False,
                     only_future_frames: bool = True,
                     allow_only_current_IDs: bool = True,
                     IDs_curr: Set[int] | List[int] = None,
                     index_lab_combo: Tuple[int, np.ndarray] = None,
                     curr_rp: list = None,
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

        Hierarchy of arguments for current_IDs:
        1. IDs_curr (if provided)
        (2. index_lab_combo (if provided) (is also passed to not current frame only 
        propagation if that propagation is necessary, and used when the frame_i matches))
        3. curr_rp (if provided)
        4. curr_lab (if provided)
        5. allData_li

        Parameters
        ----------
        frame_i : int
            The frame index for the propagation.
        allData_li : list
            See rest of ACDC. posData.allData_li. 
            Used to get the IDs for the current frame.
            Especially for when propagating after curr_frame_only was changed.
            Strictly speaking could be substituted with the correct index_lab_combo
            if necessary in the future.
        new_whitelist : Set[int] | List[int], optional
            A new set of whitelist IDs to replace the current whitelist. Cannot be 
            used together with `IDs_to_add` or `IDs_to_remove`, by default None.
        IDs_to_add : Set[int], optional
            A set of IDs to add to the current whitelist, by default None.
        IDs_to_remove : Set[int], optional
            A set of IDs to remove from the current whitelist, by default None.
        try_create_new_whitelists : bool, optional
            If True, creates new whitelist entries for frames that do not already 
            have them. Should only be necessary when its initialized, by default False.
        curr_frame_only : bool, optional
            If True, only updates the whitelist for the current frame. 
            (See description of function), by default False.
        force_not_dynamic_update : bool, optional
            If True, disables dynamic updates to the whitelist. 
            (See description of function), by default False.
        only_future_frames : bool, optional
            If True, propagates changes only to future frames, by default True.
        allow_only_current_IDs : bool, optional
            If True, only allows IDs that are present in the current frame 
            to be added to the whitelist, by default True.
        IDs_curr : Set[int] | List[int], optional
            A set of IDs for the current frame, if None, 
            will be calculated from other stuff (see description), by default None.
        index_lab_combo : Tuple[int, np.ndarray], optional
            Combination of frame_i and current frame, 
            Used to get IDs_curr (see description), when the frame_i matches
            (is also passed to not current frame only 
            propagation if that propagation is necessary, 
            and used when the frame_i matches), by default None.
        curr_rp : list, optional
            Region properties for the current frame. For IDs_curr. (see description), 
            by default None.
        curr_lab : np.ndarray, optional
            Labels for the current frame for IDs_curr. (see description),
            by default None.

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
            try:
                IDs_curr = IDs_curr.copy()
            except AttributeError:
                pass
            IDs_curr = set(IDs_curr)
            
        elif index_lab_combo and index_lab_combo[0] == frame_i:
            lab = index_lab_combo[1]
            if self._debug:
                printl('Using index_lab_combo')
            IDs_curr = {obj.label for obj in skimage.measure.regionprops(lab)}
        elif curr_rp is not None:
            IDs_curr = {obj.label for obj in curr_rp}
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
                self.propagateIDs(self.whitelistOriginalFrame_i,
                                  allData_li,
                                  index_lab_combo=index_lab_combo)
        else:
            if self.whitelistOriginalFrame_i is not None:
                if self.whitelistOriginalFrame_i != frame_i:
                    if self._debug:
                        printl('Frame changed, whitelist was not propagated, propagating...')
                    self.propagateIDs(self.whitelistOriginalFrame_i,
                                      allData_li,
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