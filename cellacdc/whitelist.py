import os
import numpy as np
import skimage.measure
from . import printl, myutils
import json
from typing import Set, List, Tuple
import time

from . import (
    html_utils, 
    apps, 
    widgets, 
    exception_handler, 
    disableWindow, 
    gui_utils,
    exec_time
)
from .trackers.CellACDC import CellACDC_tracker

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
        self.new_centroids = None

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
            og_data = og_data[og_data.files[0]]

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
             new_centroids_path:str,
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
            
        self.makeOriginalLabsAndIDs(segm_data, allData_li
                                    )
            
        self.load_centroids(new_centroids_path=new_centroids_path)
        return True
    
    def load_centroids(self, new_centroids_path:str):
        if os.path.exists(new_centroids_path):
            with open(new_centroids_path, 'r') as json_file:
                self.new_centroids = json.load(json_file)
            
            self.new_centroids = list(self.new_centroids) if isinstance(self.new_centroids, list) else self.new_centroids
            for i, val in enumerate(self.new_centroids):
                if isinstance(val, str) and val.lower() == "none":
                    self.new_centroids[i] = {}
                elif val is None:
                    self.new_centroids[i] = {}
                else: # convert to integers
                    self.new_centroids[i] = {tuple(map(int, centroid)) for centroid in val}
        else:
            printl('No new centroids file found, initializing new centroids.')
            self.create_new_centroids()
    
    def create_new_centroids(self,
                            curr_rp=None,
                            frame_i:int=None, 
                            ):
        """
        Creates self.new_centroids based on the input data.
        

        Parameters
        ----------
        curr_rp : skimage.measure.regionprops, optional
            Region properties for the current frame, by default None
        frame_i : int, optional
            Frame index for curr_rp, by default None

        Raises
        ------
        ValueError
            If curr_rp is provided, frame_i must also be provided.
        """
        if self.new_centroids is not None:
            return
        
        if frame_i is None and curr_rp is not None:
            raise ValueError(
                'If curr_rp is provided, frame_i must also be provided.'
            )
        
        self.new_centroids = []
        for i in self.total_frames:
            if i == 0:
                self.new_centroids.append({})
                continue
            
            all_there = (self.originalLabsIDs[i] is not None and 
                         self.originalLabsIDs[i-1] is not None)
            if all_there is False:
                self.new_centroids.append({})
                continue
                
            new_IDs = self.originalLabsIDs[i] - self.originalLabsIDs[i-1]
            
            rp = None
            if frame_i==i and curr_rp is not None:
                rp = curr_rp
            else:
                rp = skimage.measure.regionprops(self.originalLabs[i])

            self.new_centroids.append({
                tuple(map(int, obj.centroid)) for obj in rp if obj.label in new_IDs
            })
                

    def save(self, whitelist_path:str, new_centroids_path:str):
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
        
        for i, val in enumerate(self.new_centroids):
            if val is None:
                self.new_centroids[i] = "None"
            else:
                self.new_centroids[i] = list(val)

        with open(new_centroids_path, 'w+') as json_file:
            json.dump(self.new_centroids, json_file, indent=4)

    def checkOriginalLabels(self, frame_i:int):
        """Checks if there are no original labels for the current frame.
        
        Parameters
        ----------
        frame_i : int
            The frame index to check.
        Returns
        -------
        bool
            True if there are original labels, False otherwise.
        """
        if len(self.originalLabsIDs) <= frame_i or self.originalLabsIDs is None or self.originalLabsIDs[frame_i] is None:
            return False

        return True

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
        original labels. The IDs are added to the whitelist for the 
        current frame.
        Also propagates.

        Parameters
        ----------
        frame_i : int
            for which frame to add the IDs
        allData_li : list
            passed to self.propagateIDs(), see rest of ACDC: posData.
            allData_li
        IDs_curr : list | set, optional
            Currently present IDs,  passed to self.propagateIDs(). by default 
            None
        index_lab_combo: Tuple[int, np.ndarray]=None,
            Combination of frame_i and current frame, 
            passed to self.propagateIDs(), by default None
        curr_rp : list, optional
            Region properties for the current frame, passed to 
            self.propagateIDs(). by default None
        curr_lab : np.ndarray, optional
            Labels for the current frame, passed to self.propagateIDs(). 
            by default None
        """
        
        for i in [frame_i, frame_i-1]:
            if not self.checkOriginalLabels(i):
                return
        
        if curr_lab is None:
            curr_lab = allData_li[frame_i]['labels']
        
        new_centroids = self.new_centroids[frame_i]
        if not new_centroids:
            return
        
        new_IDs = {gui_utils.nearest_ID_to_centroid(curr_lab, *new_centroid) for new_centroid in new_centroids}

        self.propagateIDs(IDs_to_add=new_IDs, 
                          curr_frame_only=False,
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


        self.makeOriginalLabsAndIDs(segm_data, allData_li=allData_li, 
                                    frame_i=frame_i, curr_lab=curr_lab,
                                    IDs_curr=IDs_curr,
                                    )
        self.create_new_centroids()

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

    def makeOriginalLabsAndIDs(self, segm_data: np.ndarray,  
                               allData_li: list=None, frame_i: int=None, 
                               curr_lab: np.ndarray=None, 
                               IDs_curr: set | list=None,):
        """ Initializes the originalLabs and originalLabsIDs variables.

        Parameters
        ----------
        segm_data : np.ndarray
            The segmentation data for the video. (Unwhitelisted!)
        allData_li : list, optional
            The allData list for the video, by default None
        frame_i : int, optional
            The frame index for the current frame, by default None
        curr_lab : np.ndarray, optional
            The current lab for the frame, by default None
        IDs_curr : set | list, optional
            The current IDs for the frame, by default None
        """
        if self.originalLabs is not None:
            return
        if IDs_curr is not None or curr_lab is not None:
            if IDs_curr is None or curr_lab is None or frame_i is None:
                raise ValueError(
                    'If IDs_curr, curr_lab or frame_i are provided, all must be provided.'
                )
                
        self.originalLabs = np.copy(segm_data)
        self.originalLabsIDs = [None] * len(self.total_frames)
        
        if IDs_curr is not None:
            self.originalLabsIDs[frame_i] = IDs_curr
        
        if allData_li is not None:
            for i in range(len(allData_li)):
                if i == frame_i and IDs_curr is not None: # already set
                    continue
                lab = None
                try:
                    lab = allData_li[i]['labels']
                except:
                    pass
                if lab is not None:
                    self.originalLabs[i] = lab.copy()
            
        for i in range(len(segm_data)):
            IDs = None
            if IDs_curr is not None and i == frame_i:
                IDs = set(IDs_curr)
            elif allData_li is not None:
                try:
                    IDs = set(allData_li[i]['IDs'])
                except KeyError:
                    pass
            if IDs is None:
                IDs = {obj.label for obj in skimage.measure.regionprops(self.originalLabs[i])}
            self.originalLabsIDs[i] = IDs
        
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

    def initNewFrames(self,
                      frame_i: int,
                      force: bool = False,
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
            should be reinit, by default False

        Returns
        -------
        bool
            True if a new frame was initialized, False if not.
        """
        
        missing_frames = set(range(frame_i+1)) - self.initialized_i
        update_frames = []
    
        if self._debug:
            printl(missing_frames, self.initialized_i, frame_i)

        if frame_i not in self.initialized_i:
            new_frame = True
        else:
            new_frame = False

        if not force and not missing_frames:
            return new_frame, update_frames

        for i in missing_frames:
            self.whitelistIDs[i] = set()
            if i == 0:
                prev_wl = set()
            else:
                prev_wl = self.whitelistIDs[i-1]
            if prev_wl is None:
                prev_wl = set()
                
            if not self.checkOriginalLabels(i):
                available_IDs = set()
            else:
                available_IDs = self.originalLabsIDs[i]
            if available_IDs is None:
                new_wl = set()
            else:
                new_wl = prev_wl.intersection(available_IDs)
            if new_wl:
                self.whitelistIDs[i] = new_wl
            else:
                self.whitelistIDs[i] = set()
                        
            self.initialized_i.add(i)
            update_frames.append((i, None, None, True))

        if self._debug:
            printl('Whitelist IDs new frame (without adding new IDs):', self.whitelistIDs[frame_i])
        return new_frame, update_frames
    
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
                     update_frames: list = None,
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
        update_frames : list, optional
            List of frames that were changed, by default None.
            Returned for updating the labs later

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
        
        if not update_frames:
            update_frames = []

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

        new_frame, update_frames_new = self.initNewFrames(frame_i)
        update_frames.extend(update_frames_new)

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
                new_update_frames = self.propagateIDs(self.whitelistOriginalFrame_i,
                                  allData_li,
                                  index_lab_combo=index_lab_combo,
                                  update_frames=update_frames)
                update_frames.extend(new_update_frames)
        else:
            if self.whitelistOriginalFrame_i is not None:
                if self.whitelistOriginalFrame_i != frame_i:
                    if self._debug:
                        printl('Frame changed, whitelist was not propagated, propagating...')
                    new_update_frames = self.propagateIDs(self.whitelistOriginalFrame_i,
                                      allData_li,
                                      index_lab_combo=index_lab_combo,
                                      update_frames=update_frames
                                      )
                    update_frames.extend(new_update_frames)
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
            if self.checkOriginalLabels(frame_i):
                IDs_curr.update(self.originalLabsIDs[frame_i])
            new_whitelist = new_whitelist.intersection(IDs_curr)
            if self._debug:
                printl(IDs_curr)
                printl(new_whitelist, old_whitelist)

        # get IDs to add/remove
        IDs_to_add = new_whitelist - old_whitelist
        IDs_to_remove = old_whitelist - new_whitelist

        if IDs_to_add == IDs_to_remove == set():
            return update_frames

        if self._debug:
            printl(IDs_to_add, IDs_to_remove)
            
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
                if self.checkOriginalLabels(i):
                    IDs_og = self.originalLabsIDs[i]
                    if IDs_og is None:
                        IDs_og = set()
                else:
                    IDs_og = set()
                if frame_i == i:
                    IDs_curr_loc = IDs_curr
                else:
                    IDs_curr_loc = set(allData_li[i]['IDs'])

            new_whitelist = self.get(i, try_create_new_whitelists).copy()
            old_whitelist = new_whitelist.copy()
            added_IDs = []
            removed_IDs = []
            if IDs_to_add:
                #                         intersection with...   all possible IDs           ...plus all old_whitelistIDs
                new_whitelist = IDs_to_add.intersection(IDs_curr_loc.union(IDs_og)) | old_whitelist
                # IDs_curr.union(IDs_og) are all possible IDs, IDs_to_add.intersection(IDs_curr.union(IDs_og)) is for finding all possible IDs which want ot be propagated
                added_IDs = new_whitelist - old_whitelist
            if IDs_to_remove:
                new_whitelist = new_whitelist - IDs_to_remove
                removed_IDs = old_whitelist - new_whitelist
                        
            self.whitelistIDs[i] = new_whitelist
            if added_IDs or removed_IDs:
                update_frames.append((i,added_IDs, removed_IDs, False))

        if self._debug:
            printl(self.whitelistIDs[frame_i])
        
        return update_frames
    
class WhitelistGUIElements:
    """A class to manage the whitelist GUI elements.
    """
    def whitelistCheckOriginalLabels(self, warning:bool=True, 
                                       frame_i:int=None):
        """Warns the user that there are no original labels labels are present 
        for the frame"""
        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return False
        
        if frame_i is None:
            frame_i = posData.frame_i
        
        if posData.whitelist.originalLabsIDs is None:
            return False

        if (frame_i >= len(posData.whitelist.originalLabsIDs) or
           posData.whitelist.originalLabsIDs[frame_i] is None):
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
        """Tracks the original labels against the previous frame.
        This is used as a callback for sigTrackOGagainstPreviousFrame signal
        """
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        if not self.whitelistCheckOriginalLabels():
            return
        old_cell_IDs = posData.whitelist.originalLabsIDs[frame_i]
        prev_cell_IDs = posData.allData_li[frame_i-1]['IDs']
        self.whitelistTrackOGCurr(against_prev=True)
        new_cell_IDs = posData.whitelist.originalLabsIDs[frame_i]

        new_IDs = new_cell_IDs - old_cell_IDs
        new_IDs = new_IDs & set(prev_cell_IDs)

        self.whitelistUpdateLab(
            track_og_curr=False, IDs_to_add=new_IDs,
        )

    def whitelistLoadOGLabs_cb(self):
        """Generates a dialog to load the original (not whitelisted) labels
        """
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
        """Loads the original labels from the selected files

        Parameters
        ----------
        selected : str
            Selected file name from the dialog.
        """
        posData = self.data[self.pos_i]
        images_path = posData.images_path

        selected_path = os.path.join(images_path, selected)
        posData.whitelist.loadOGLabs(selected_path)
        
        self.whitelistIDsToolbar.viewOGToggle.setCheckable(True)

    @exception_handler
    @disableWindow
    def whitelistViewOGIDs(self, checked:bool):
        """Switch between selected and original labels.
        Uses self.viewOriginalLabels to see what has to be done.

        Parameters
        ----------
        checked : bool
            True if the original labels have to be shown, False otherwise.
        """
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
        if frame_i > 0:
            frames_range = [frame_i-1, frame_i]
        else:
            frames_range = [frame_i]

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

                IDs = posData.IDs

                og_frame = posData.whitelist.originalLabs[i].copy()
                IDs_to_uppdate = posData.whitelist.whitelistIDs[i] & posData.whitelist.originalLabsIDs[i]
                if IDs_to_uppdate:
                    mask = np.isin(og_frame, list(IDs_to_uppdate))
                    og_frame[mask] = 0

                    mask = np.isin(posData.lab, list(IDs_to_uppdate))
                    og_frame[mask] = posData.lab[mask]
                
                IDs_to_add = posData.whitelist.whitelistIDs[i] - posData.whitelist.originalLabsIDs[i]
                if IDs_to_add:
                    mask = np.isin(posData.lab, list(IDs_to_add))
                    og_frame[mask] = posData.lab[mask]

                posData.lab = og_frame
                self.update_rp(wl_update=False)
                self.store_data(autosave=False)

            if frame_i > 0:
                missing_IDs = set(posData.IDs) - set(posData.allData_li[frame_i-1]['IDs'])
                self.trackManuallyAddedObject(missing_IDs,isNewID=True, wl_update=False)

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

                # self.whitelistTrackCurrOG()
                self.update_rp(wl_update=False)
                self.store_data(autosave=False)
                self.whitelistUpdateLab(frame_i=i) #has update_rp and store data
                self.setAllTextAnnotations()
                self.updateAllImages()

    def whitelistSetViewOGIDsToggle(self, checked: bool):
        """Set the view original labels toggle button to checked or unchecked.
        This also updates the self.viewOriginalLabels variable.
        !!! Doesn't change the actually displayed labels, use self.whitelistViewOGIDs
        to do that.!!!
        
        Parameters
        ----------
        checked : bool
            True if the original labels are shown, False otherwise.
        """
        self.viewOriginalLabels = checked
        self.whitelistIDsToolbar.viewOGToggle.blockSignals(True)
        self.whitelistIDsToolbar.viewOGToggle.setChecked(checked)
        self.whitelistIDsToolbar.viewOGToggle.blockSignals(False)

    def whitelistAddNewIDsToggled(self, checked: bool):
        """Will set self.addNewIDsWhitelistToggle to checked and call
        whitelistAddNewIDs if checked is True.

        Parameters
        ----------
        checked : bool
            True if the add new IDs toggle is checked, False otherwise.
        """
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
        """Function which adds new IDs to the whitelist, based on the original labels.
        It will check if the frame is visited the first time, unless 
        ignore_not_first_time is True.
        It does nothing if self.addNewIDsWhitelistToggle is False.
        !!!Careful, does not change the lab, just the whitelist!!!

        Parameters
        ----------
        ignore_not_first_time : bool, optional
            Weather it should be checked if the frame is visited 
            the first time, by default False
        """
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
        """Function which is called when the user accepts a whitelist.
        Also initializes the whitelist if it is not already initialized. (Aka not loaded)

        Parameters
        ----------
        whitelistIDs : set | list
            The accepted IDs from the whitelist dialog.
        """
        # Store undo state before modifying stuff
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
                
        # self.whitelistPropagateIDs(new_whitelist=whitelistIDs, 
        #                            try_create_new_whitelists=True, 
        #                            only_future_frames=True, 
        #                            force_not_dynamic_update=True,
        #                            update_lab=True
        #                            )
        self.whitelistUpdateLab(track_og_curr=True)

        self.whitelistIDsUpdateText()
        self.keepIDsTempLayerLeft.clear()

    def whitelistUpdateLab(self, frame_i: int=None,
        track_og_curr=False, new_frame:bool=False,
        IDs_to_add:List[int] | Set[int]=None,
        IDs_to_remove:List[int]|Set[int]=None,
        ): 
        # this should also work for 3D i think...
        """Updates the displayed lab based on the whitelist.

        Parameters
        ----------
        frame_i : int, optional
            frame which should be updated. If not provided, 
            uses posData.frame_i, by default None
        track_og_curr : bool, optional
            if True, will track the original current IDs, by default False
        new_frame : bool, optional
            if True, will set the frame to the new frame, by default False
        IDs_to_add : list, optional
            IDs to add to the whitelist, by default None
        IDs_to_remove : list, optional
            IDs to remove from the whitelist, by default None
        """
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
            # getting data is handles later in the code
            
        debug = posData.whitelist._debug
        if debug:
            printl('whitelistUpdateLab', frame_i, og_frame_i)
            from . import debugutils
            debugutils.print_call_stack()

        if benchmark:
            ts.append(time.perf_counter())

        self.whitelistSetViewOGIDsToggle(False) ###

        if benchmark:
            ts.append(time.perf_counter())
            
        if self.whitelistCheckOriginalLabels(warning=False, frame_i=frame_i):
            og_lab = posData.whitelist.originalLabs[frame_i] ###
        else:
            og_lab = None
        if benchmark:
            ts.append(time.perf_counter())

        ####
        whitelist = posData.whitelist.get(frame_i=frame_i)
        IDs_to_add_remove_provided = IDs_to_add is not None or IDs_to_remove is not None
        if not IDs_to_add_remove_provided:
            self.get_data()
            got_data = True
            current_IDs = set(posData.IDs)
            missing_IDs = list(whitelist - current_IDs)
            to_be_removed_IDs = list(current_IDs - whitelist)
        else:
            missing_IDs = list(IDs_to_add) if IDs_to_add is not None else []
            to_be_removed_IDs = list(IDs_to_remove) if IDs_to_remove is not None else []

        ###

        if benchmark:
            ts.append(time.perf_counter())
        
        ###
        if not missing_IDs and not to_be_removed_IDs: # nothing to do
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

        ###
        if missing_IDs and track_og_curr and not new_frame:
            self.whitelistTrackOGCurr(frame_i=frame_i, 
                                      lab = posData.lab,
                                      rp = posData.rp)
        
        missing_IDs = np.array(missing_IDs, dtype=np.int32)
        to_be_removed_IDs = np.array(to_be_removed_IDs, dtype=np.int32)

        if debug:
            printl(missing_IDs, to_be_removed_IDs)

        curr_lab = posData.lab # or curr_lab = posData.lab??? 
        # convert values to int if they are not already
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
        curr_lab = curr_lab.astype(np.int32)
        if benchmark:
            ts.append(time.perf_counter())

        if missing_IDs.size > 0 and og_lab is not None:
            mask = np.isin(og_lab, missing_IDs) # add missing_IDs
            curr_lab[mask] = og_lab[mask]

        if to_be_removed_IDs.size > 0:
            curr_lab[np.isin(curr_lab, to_be_removed_IDs)] = 0 # remove to_be_removed_IDs

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
        """Updates the text. Carefull, triggers whitelistLineEdit.textChanged!
        """
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
        """Track the original labels in relation to the current (whitelisted) 
        labels.
        Parameters

        Parameters
        ----------
        frame_i : int, optional
            frame_i to be tracked, posData.frame_i if not provided, 
            by default None
        against_prev : bool, optional
            if the original frame should be tracked against frame_i-1. 
            Cannot be used with rp or lab, by default False
        lab : np.ndarray, optional
            lab to be tracked against, by default None
        rp : list, optional
            regionprops for this lab, by default None
        IDs : Set[int] | List[int], optional
            IDs that should be tracked based on og

        Raises
        ------
        ValueError
            Cannot provide both rp and lab when tracking against previous frame.
            Instead only provide rp and lab, and dont set against_prev.
        """
        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return

        debug = posData.whitelist._debug

        if debug:
            from . import debugutils
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
        ### against what should I track?

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
        # lab = lab.copy()

        denom_overlap_matrix = 'union' if not against_prev else 'area_prev'

        og_lab = CellACDC_tracker.track_frame(
                lab, rp, og_lab, og_rp,
                denom_overlap_matrix=denom_overlap_matrix,
                posData = posData,
                setBrushID_func=self.setBrushID,
                IDs=IDs,
                # assign_unique_new_IDs=False,
        )

        posData.whitelist.originalLabs[frame_i] = og_lab
        posData.whitelist.originalLabsIDs[frame_i] = {obj.label for obj in skimage.measure.regionprops(og_lab)}

        if changed_frame:
            posData.frame_i = og_frame_i
            self.get_data()

    def whitelistTrackCurrOG(self, frame_i:int=None, against_prev:bool=False):
        """Track the current (whitelisted) labels in relation to the original labels.
        Parameters
        ----------
        frame_i : int, optional
            frame_i to be tracked, posData.frame_i if not provided, by default None
        against_prev : bool, optional
            if the original frame should be tracked against frame_i-1. 
        """
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
        """Interates over the frames and calls whitelistTrackOGCurr for each frame.

        Parameters
        ----------
        frame_is : List[int], optional
            list of frame_i, if None goes through all, by default None
        against_prev : bool, optional
            if the original frame should be tracked against frame_i-1. 
        """
        posData = self.data[self.pos_i]
        if frame_is is None:
            frame_is = range(posData.SizeT)

        for frame_i in frame_is:
            self.whitelistTrackOGCurr(frame_i=frame_i, against_prev=against_prev)

    def whitelistInitNewFrames(self, frame_i:int=None, force:bool=False):
        """Initialize the whitelist for a new frame. The class whitelist keeps track
        of the init frames and doesnt try to init them again, unless forced.
        Does not init the class!

        Parameters
        ----------
        frame_i : int, optional
            frame_i to be init, posData.frame_i if not provided, by default None
        force : bool, optional
            if the init should be forced, by default False

        Returns
        -------
        bool
            if the frame was new or not
        list
            list of frames that were updated, and info about added/removed IDs
        """

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

    # @exec_time
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
        """
        Propagates whitelist IDs across frames in the dataset. (Doesnt update labs)
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
        new_whitelist : Set[int] | List[int], optional
            A new set of whitelist IDs to replace the current whitelist. Cannot be 
            used together with `IDs_to_add` or `IDs_to_remove`, by default None.
        IDs_to_add : Set[int], optional
            A set of IDs to add to the current whitelist, by default None.
        IDs_to_remove : Set[int], optional
            A set of IDs to remove from the current whitelist, by default None.
        frame_i : int, optional
            The frame index for the propagation. 
            If None, uses posData.frame_i, by default None.
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
        track_og_curr : bool, optional
            If True, tracks the original labels in relation to the current
            (whitelisted) labels. This is done by calling whitelistTrackOGCurr.
            If its a new frame, this is done in whitelistInitNewFrames against the 
            previous frame,
            by default True.
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
        store_data : bool, optional
            If True, stores the data before propagating the IDs.
        update_lab : bool, optional
            If True, updates the labels after propagating the IDs.
            Will always update labels for newly init frames, by default False.

        Raises
        ------
        ValueError
            If both `new_whitelistIDs` and `IDs_to_add`/`IDs_to_remove` are provided.

        Example
        -------
        To add IDs 5 and 6 to the whitelist for the current frame:
        ```python
        self.whitelistPropagateIDs(IDs_to_add={5, 6}, curr_frame_only=True)
        ```
        Then when the frame changes:
        ```python
        self.whitelistPropagateIDs()
        ```

        To replace the whitelist for frame 10 with a new set of IDs:
        ```python
        self.whitelistPropagateIDs(new_whitelistIDs={1, 2, 3}, frame_i=10)
        ```
        This would also propagate the changes to all other frames.

        """
        #doesnt update the frame displayed, only wl
        try: # safety XD
            IDs_curr = IDs_curr.copy()
        except AttributeError:
            pass
            
        IDs_curr = set(IDs_curr) if IDs_curr is not None else None

        posData = self.data[self.pos_i]

        debug = posData.whitelist._debug if posData.whitelist is not None else False

        if debug:
            printl('Propagating IDs...')
            from . import debugutils
            debugutils.print_call_stack()
            printl(new_whitelist, IDs_to_add, IDs_to_remove)

        if posData.whitelist is None:
            return

        # og_frame_i = posData.frame_i
        if frame_i is None:
            frame_i = posData.frame_i

        new_frame, update_frames_init = self.whitelistInitNewFrames(frame_i=frame_i)

        if new_frame:
            self.update_rp(wl_update=False)
        # if track_og_curr and not new_frame:
        #     self.whitelistTrackOGCurr(frame_i=frame_i, rp=curr_rp, lab=curr_lab)

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
        # printl(posData.whitelistIDs[frame_i])
        # posData.frame_i = og_frame_i
        self.whitelistIDsUpdateText()
        if store_data:
            self.store_data(autosave=False)

        for frame_i, IDs_to_add, IDs_to_remove, new_frame in update_frames:
            self.whitelistUpdateLab(frame_i=frame_i, track_og_curr=track_og_curr, 
                                    new_frame=new_frame, IDs_to_add=IDs_to_add, 
                                    IDs_to_remove=IDs_to_remove, )

    def whitelistIDs_cb(self, checked:bool):
        """Callback for when the whitelist IDs button is checked or unchecked.
        Initialises the pointlayer and the whitelist IDs toolbar if checked.

        Parameters
        ----------
        checked : bool
            True if the whitelist IDs button is checked, False otherwise.
        """
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
        """Highlights the IDs in the current frame based on the whitelist.

        Parameters
        ----------
        checked : bool, optional
            If False, will delete all highlights, by default True
        """
        if not checked:
            self.removeHighlightLabelID()
            return
        
        posData = self.data[self.pos_i]

        if posData.whitelist is None:
            if not hasattr(self, 'tempWhitelistIDs'):
                self.tempWhitelistIDs = set() # not updated, only use in this context
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
        """Callback for when the whitelist IDs are changed. 
        This is called when the user changed the IDs in the whitelist IDs toolbar
        (or when its programmatically changed, but if its not 
        visible it should return instantly)
        Will update the temp layer and also complain when IDs 
        are not valid/present in the current lab

        Parameters
        ----------
        whitelistIDs : set | list
            The IDs that are currently in the whitelist.
        debug : bool, optional
            debug, by default False
        """
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
                self.tempWhitelistIDs = set() # not updated, only use in this context
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

        isAnyIDnotExisting = False
        for ID in whitelistIDs:
            if ID not in possible_IDs:
                isAnyIDnotExisting = True
                continue
            if ID not in current_whitelist_copy:
                current_whitelist.add(ID)
                self.highlightLabelID(ID)

        for ID in current_whitelist_copy:
            if ID not in possible_IDs:
                isAnyIDnotExisting = True
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

    # @exec_time
    def whitelistUpdateTempLayer(self):
        """Updates the temp layer with the current whitelist IDs.
        """
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
                self.tempWhitelistIDs = set() # not updated, only use in this context
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