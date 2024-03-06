import os
from cellacdc.trackers.CellACDC.CellACDC_tracker import calc_IoA_matrix
from cellacdc.trackers.CellACDC.CellACDC_tracker import track_frame as track_frame_base
from cellacdc.core import getBaseCca_df, printl
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
import pandas as pd

def reorg_daughter_cells(lineage_tree_frame, max_daughter):    
    """
    Reorganizes the daughter cells in the lineage tree frame.

    Args:
        lineage_tree_frame (pandas.DataFrame): The lineage tree frame containing the daughter cells.
        max_daughter (int): The maximum number of daughter cells.

    Returns:
        pandas.DataFrame: The lineage tree frame with reorganized daughter cells (e.g. 'daughter_ID_tree_1', 'daughter_ID_tree_2', ...)
    """
    new_columns = [f'sister_ID_tree_{i+1}' for i in range(max_daughter-1)]
    sister_columns = lineage_tree_frame['sister_ID_tree'].apply(pd.Series) 
    lineage_tree_frame[new_columns] = sister_columns
    lineage_tree_frame['sister_ID_tree'] = sister_columns[0]

    return lineage_tree_frame


def mother_daughter_assign(IoA_matrix, IoA_thresh_daughter, min_daughter, max_daughter):
    """
    Identifies cells that have not undergone division based on the input IoA matrix.

    Args:
        IoA_matrix (numpy.ndarray): Matrix representing the Intersection over Area (IoA) values between cells.
        IoA_thresh_daughter (float): Threshold value for considering a cell as a daughter cell.
        min_daughter (int): Minimum number of daughter cells required for considering a cell as a mother cell.
        max_daughter (int): Maximum number of daughter cells required for considering a cell as a mother cell.

    Returns:
        tuple: A tuple containing two elements:
            - aggr_track (list): A list of indices of cells that have not undergone division.
            - mother_daughters (list): A list of tuples, where each tuple contains the index of a mother cell and a list of indices of its daughter cells. (INDICIES ARE BASED ON THE INPUT IoA MATRIX)
    """
    mother_daughters = []
    aggr_track = []
    daughter_range = range(min_daughter, max_daughter+1, 1)

    IoA_thresholded = IoA_matrix >= IoA_thresh_daughter
    nrows, ncols = IoA_matrix.shape
    for j in range(ncols):
        high_IoA_indices = np.where(IoA_thresholded[:, j])[0]
        
        if not high_IoA_indices.size:
            continue
        elif not len(high_IoA_indices) in daughter_range:
            aggr_track.extend(high_IoA_indices)
        else:
            mother_daughters.append((j, high_IoA_indices))

    return aggr_track, mother_daughters

def added_lineage_tree_to_cca_df(added_lineage_tree):
    """
    Converts the added lineage tree into a DataFrame with specific columns (see getBaseCca_df from cellacdc.core for more details). 

    Args:
        added_lineage_tree (list): The input lineage tree with the following columns:
            - 'Division frame'
            - 'Daughter ID' ("ID")
            - 'Mother ID'
            - 'Generation'
            - 'Origin ID'
            - 'Sister IDs'

    Returns:
        pandas.DataFrame: The converted DataFrame with the following columns:
            - 'emerg_frame_i'
            - 'division_frame_i'
            - 'generation_num_tree'
            - 'parent_ID_tree'
            - 'root_ID_tree'
            - 'sister_ID_tree'
    """

    cca_df = getBaseCca_df([row[1] for row in added_lineage_tree], with_tree_cols=True)
    cca_df['emerg_frame_i'] = [row[0] for row in added_lineage_tree]
    cca_df['division_frame_i'] = [row[0] for row in added_lineage_tree]
    cca_df['generation_num_tree'] = [row[3] for row in added_lineage_tree]
    cca_df['parent_ID_tree'] = [row[2] for row in added_lineage_tree]
    cca_df['root_ID_tree'] = [row[4] for row in added_lineage_tree]
    cca_df['sister_ID_tree'] = [row[5] for row in added_lineage_tree]
    return cca_df

def create_lineage_tree_video(segm_video, IoA_thresh_daughter, min_daughter, max_daughter):
    """
    Creates a lineage tree for a video of segmented frames (Currently not in use).

    Args:
        segm_video (list): A list of segmented frames.
        IoA_thresh_daughter (float): IoA threshold for identifying daughter cells.
        min_daughter (int): Minimum number of daughter cells.
        max_daughter (int): Maximum number of daughter cells.

    Returns:
        list: A list representing the lineage tree.

    """
    tree = normal_division_lineage_tree(lab=segm_video[0])
    for i, frame in enumerate(segm_video[1:], start=1):
        rp = regionprops(frame)
        prev_rp = regionprops(segm_video[i-1])
        IoA_matrix, IDs_curr_untracked, IDs_prev = calc_IoA_matrix(frame, segm_video[i-1], rp, prev_rp)
        _, mother_daughters = mother_daughter_assign(IoA_matrix, IoA_thresh_daughter, min_daughter, max_daughter)
        assignments = IDs_curr_untracked #bc we dont track the frame
        tree.create_tracked_frame_tree(i, mother_daughters, IDs_prev, IDs_curr_untracked, assignments)
    return tree.lineage_list

def filter_current_IDs(df, current_IDs):  
    """
    Filters for current IDs.

    Args:
        df (pandas.DataFrame): The DataFrame containing the IDs.
        indices_to_remove (list): A list of indices NOT to be removed.

    Returns:
        pandas.DataFrame: The DataFrame with only the current IDs.
    """
    return df[df.index.isin(current_IDs)]

def IoA_index_daughter_to_ID(daughters, assignments, IDs_curr_untracked):
    """
    Converts a list of daughter indices (IoA Matrix) to their corresponding IDs.

    Parameters:
    daughters (list): List of daughter indices.
    assignments (dict): Dictionary mapping IDs to assigned values.
    IDs_curr_untracked (list): List of current untracked IDs.

    Returns:
    list: List of daughter IDs.
    
    """

    if daughters is None:
        return
    
    daughter_IDs = []
    for daughter in daughters:
        if assignments:
            daughter_IDs.append(assignments[IDs_curr_untracked[daughter]]) # this works because daughter is a single ID. New items wont be in assignments but not in daughter so its fine *pew*
        else:
            daughter_IDs.append(IDs_curr_untracked[daughter])

    return daughter_IDs

def add_member_fam_from_df(families, df):
    """
    Updates the families list based on the input DataFrame. If new families are found, they are added to the list. This is probably an overly complicated way of doing this, but it works...

    Args:
        families (list): List of families in the lineage tree.
        df (pandas.DataFrame): The DataFrame containing the lineage tree.

    Returns:
        list: The updated list of families. For details see attribute disc. of normal_division_lineage_tree.
    """

    additions = []

    for index, row in df.iterrows():
        for family in families:
            in_family = False
            parent_in_family = False

            if row['root_ID_tree'] == family[0][0]: # find the right family
                for member in family: 
                    if index == member[0]: # already in the family
                        in_family = True
                        break

                if not in_family:
                    family.append((index, row['generation_num_tree']))
                    additions.append((family[0][0], index))
                    break
        else:
            families.append([(index, 1)]) # add new family if for loop is exhaused and no family is found

    return families


def del_member_fam_from_df(families, df):
    """
    Corrects families list based on the input DataFrame. If the cell is in a family it should not belong (based on Root ID) it is delted from that family.
    If this is slow it can be optimized by putting it together with update_fam_from_df

    Args:
        families (list): List of families in the lineage tree.
        df (pandas.DataFrame): The DataFrame containing the lineage tree.

    Returns:
        list: The updated list of families. For details see attribute disc. of normal_division_lineage_tree.
    """

    deletions = []
    
    for index, row in df.iterrows():
        for family in families:
            if row['root_ID_tree'] != family[0][0]: # find families which should not have the cell
                for i, member in enumerate(family): 
                    if index == member[0]: # this cant be right!
                        family.pop(i) # remove the cell from the family
                        deletions.append((family[0][0], index)) # add to deletions list
                        break




    return families, deletions

def update_generation_from_df(families, df):
    for fam in families:
        for member in fam:
            member[1] = df.loc[member[0], 'generation_num_tree'] 

    return families

def propagate_generation_from_fams(families, df_li):
    for fam in families:
        for member in fam:
            daughters = set()
            parent_indx = -1 # this is default
            for df in df_li:
                if member[0] in df.index:
                    df.loc[member[0], 'generation_num_tree'] = member[1]
                    parent_indx = df.loc[member[0], 'parent_ID_tree']
                daughters = daughters + set(df.loc[df['parent_ID_tree'] == member[0]].index.tolist())


class normal_division_tracker:
    """
    A class that tracks cell divisions in a video sequence.

    Args:
        segm_video (ndarray): The segmented video sequence.
        IoA_thresh_daughter (float): The IoA threshold for identifying daughter cells.
        min_daughter (int): The minimum number of daughter cells.
        max_daughter (int): The maximum number of daughter cells.
        IoA_thresh (float): The IoA threshold for tracking cells.
        IoA_thresh_aggressive (float): The aggressive IoA threshold for tracking cells.

    Attributes:
        IoA_thresh_daughter (float): The IoA threshold for identifying daughter cells.
        min_daughter (int): The minimum number of daughter cells.
        max_daughter (int): The maximum number of daughter cells.
        IoA_thresh (float): The IoA threshold for tracking cells.
        IoA_thresh_aggressive (float): The aggressive IoA threshold for tracking cells.
        segm_video (ndarray): The segmented video sequence.
        tracked_video (ndarray): The tracked video sequence.
        assignments (dict): A dictionary mapping untracked cell IDs to tracked cell IDs. (Only NEW cells are in this dictionary, so things the tracker could not assign to a cell in the previous frame and was given a new unique ID.)
        IDs_prev (list): A list mapping index from IoA matrix to IDs. (Index in list = Index in IoA matrix)
        rp (list): The region properties of the current frame.
        mother_daughters (list): A list of tuples, where each tuple contains the index of a mother cell and a list of indices of its daughter cells. (INDICIES ARE BASED ON THE INPUT IoA MATRIX)
        IDs_curr_untracked (list): A list of current cell IDs that are untracked. (INDICIES ARE BASED ON THE INPUT IoA MATRIX)
        tracked_IDs (set): A set of all cell IDs in current frame after tracking.
        tracked_lab (ndarray): The tracked current frame.

    Methods:
        track_frame: Tracks a single frame in the video sequence.
    """

    def __init__(self, 
                 segm_video, 
                 IoA_thresh_daughter, 
                 min_daughter, 
                 max_daughter, 
                 IoA_thresh, 
                 IoA_thresh_aggressive):
        """
        Initializes the normal_division_tracker object.

        Args:
            segm_video (ndarray): The segmented video sequence.
            IoA_thresh_daughter (float): The IoA threshold for identifying daughter cells.
            min_daughter (int): The minimum number of daughter cells.
            max_daughter (int): The maximum number of daughter cells.
            IoA_thresh (float): The IoA threshold for tracking cells. Applied first before further accessing if cell has divided.
            IoA_thresh_aggressive (float): The aggressive IoA threshold for tracking cells. Applied when the tracker thinks that a cell has NOT divided.
        """

        self.IoA_thresh_daughter = IoA_thresh_daughter
        self.min_daughter = min_daughter
        self.max_daughter = max_daughter
        self.IoA_thresh = IoA_thresh
        self.IoA_thresh_aggressive = IoA_thresh_aggressive
        self.segm_video = segm_video

        self.tracked_video = np.zeros_like(segm_video)
        self.tracked_video[0] = segm_video[0]

    def track_frame(self, frame_i, lab=None, prev_lab=None, rp=None, prev_rp=None):
        """
        Tracks a single frame in the video sequence.

        Args:
            frame_i (int): The index of the frame to track.
            lab (ndarray, optional): The segmented labels of the current frame. Defaults to None.
            prev_lab (ndarray, optional): The segmented labels of the previous frame. Defaults to None.
            rp (list, optional): The region properties of the current frame. Defaults to None.
            prev_rp (list, optional): The region properties of the previous frame. Defaults to None.
        """

        if lab is None:
            lab = self.segm_video[frame_i]

        if prev_lab is None:
            prev_lab = self.tracked_video[frame_i-1]

        if rp is None:
            self.rp = regionprops(lab.copy())
        else:
            self.rp = rp

        if prev_rp is None:
            prev_rp = regionprops(prev_lab)

        IoA_matrix, self.IDs_curr_untracked, self.IDs_prev = calc_IoA_matrix(lab, 
                                                                             prev_lab, 
                                                                             self.rp, 
                                                                             prev_rp
                                                                             )
        aggr_track, self.mother_daughters = mother_daughter_assign(IoA_matrix, 
                                                             IoA_thresh_daughter=self.IoA_thresh_daughter, 
                                                             min_daughter=self.min_daughter, 
                                                             max_daughter=self.max_daughter
                                                             )
        self.tracked_lab, IoA_matrix, self.assignments, tracked_IDs = track_frame_base(prev_lab, 
                                                                                       prev_rp, 
                                                                                       lab, 
                                                                                       self.rp, 
                                                                                       IoA_thresh=self.IoA_thresh,
                                                                                       IoA_matrix=IoA_matrix, 
                                                                                       aggr_track=aggr_track, 
                                                                                       IoA_thresh_aggr=self.IoA_thresh_aggressive, 
                                                                                       IDs_curr_untracked=self.IDs_curr_untracked, 
                                                                                       IDs_prev=self.IDs_prev, 
                                                                                       return_all=True
                                                                                       )
        self.tracked_IDs = set(tracked_IDs).union(set(self.assignments.values()))
        self.tracked_video[frame_i] = self.tracked_lab

class normal_division_lineage_tree:
    """
    Represents a lineage tree for normal cell divisions. Initializes the lineage tree with the first labeled frame, then updates the lineage tree with each subsequent frame (create_tracked_frame_tree). 

    Attributes:
    - max_daughter: Maximum number of daughter cells per division.
    - families: List of families in the lineage tree. (List of families. Each family itself is a list of touples (ID, generation))    
    - lineage_list: List of lineage dataframes representing the lineage tree.

    Methods:
    - __init__(self, lab, max_daughter): Initializes the normal_division_lineage_tree object.
    - create_tracked_frame_tree(self, frame_i, mother_daughters, IDs_prev, IDs_curr_untracked, assignments, curr_IDs): Creates df for the specified frame and appends it to lineage_list.
    - real_time_tree(self, frame_i, lab, prev_lab, rp=None, prev_rp=None): Calculates the real-time tree of cell divisions based on the input labels and region properties.
    - dict_curr_frame(self): NOT DONE DEVELOPING Creates a dictionary mapping mother IDs to daughter IDs for the current frame.
    """

    def __init__(self, lab=None, max_daughter=2, min_daughter=2, IoA_thresh_daughter=0.25, first_df=None):
        """
        Initializes the CellACDC_normal_division_tracker object.

        Parameters:
        - lab (ndarray): The labeled image of cells.
        - max_daughter (int, optional): The maximum number of daughter cells expected.
        - min_daughter (int, optional): The minimum number of daughter cells expected.
        - IoA_thresh_daughter (float, optional): The threshold for intersection over area (IoA) for daughter cells.

        Returns:
        None
        """
        if lab is None and first_df is None:
            raise ValueError('Either lab or first_df must be provided.')
        
        if lab and first_df:
            raise ValueError('Only one of lab and first_df can be provided.')
        
        self.max_daughter = max_daughter
        self.min_daughter = min_daughter 
        self.IoA_thresh_daughter = IoA_thresh_daughter
        self.mother_daughters = [] # just for the dict_curr_frame stuff...

        self.families = []

        if lab:
            added_lineage_tree = []

            rp = regionprops(lab.copy())
            for obj in rp:
                label = obj.label
                self.families.append([(label, 1)])
                added_lineage_tree.append((-1, label, -1, 1, label, [-1] * (max_daughter-1)))

            cca_df = added_lineage_tree_to_cca_df(added_lineage_tree)
            cca_df = reorg_daughter_cells(cca_df, max_daughter)
            self.lineage_list = [cca_df]

        elif first_df:
            self.lineage_list = [first_df]
            self.families.append([(label, 1) for label in first_df.index])

    def create_tracked_frame_tree(self, frame_i, mother_daughters, IDs_prev, IDs_curr_untracked, assignments, curr_IDs):        
        """
        Creates a tracked frame tree based on the given parameters.

        Parameters:
        - frame_i (int): Index of the current frame.
        - mother_daughters (list): List of tuples representing mother-daughter cell relationships.
        - IDs_prev (list): List mapping previous cell IDs to current cell IDs.
        - IDs_curr_untracked (list): List of current cell IDs that are untracked.
        - assignments (dict): Dictionary mapping untracked cell IDs to tracked cell IDs.
        - curr_IDs (list): List of current cell IDs.


        Returns:
        - None
        """

        added_lineage_tree = []
        for mother, daughters in mother_daughters:

            mother_ID = IDs_prev[mother]
            daughter_IDs = IoA_index_daughter_to_ID(daughters, assignments, IDs_curr_untracked)

            found = False  # flag to track if a family was associated

            for family in self.families:
                for member in family:
                    if mother_ID == member[0]:
                        origin_id = family[0][0]
                        generation = member[1] + 1
                        family.extend([(daughter_ID, generation) for daughter_ID in daughter_IDs])
                        found = True  # family was associated
                        break
                if found:
                    break

            if not found:  # if no family was associated
                printl(f"Warning: No family could be associated. Creating a new family for cells {daughter_IDs}.")
                # create a new family
                generation = 1
                self.families.append([(daughter_ID, generation) for daughter_ID in daughter_IDs])

            for i, daughter_ID in enumerate(daughter_IDs):
                daughter_IDs_copy = daughter_IDs.copy()
                daughter_IDs_copy.pop(i)
                daughter_IDs_copy = daughter_IDs_copy + [-1] * (self.max_daughter - len(daughter_IDs_copy) -1)
                added_lineage_tree.append((frame_i, daughter_ID, mother_ID, generation, origin_id, daughter_IDs_copy))

        cca_df = added_lineage_tree_to_cca_df(added_lineage_tree)
        cca_df = pd.concat([self.lineage_list[-1], cca_df], axis=0)
        cca_df = reorg_daughter_cells(cca_df, self.max_daughter)
        cca_df = filter_current_IDs(cca_df, curr_IDs)

        self.lineage_list.append(cca_df)

    def real_time_tree(self, frame_i, lab, prev_lab, rp=None, prev_rp=None): #TODO: make this ingnore corrected stuff (for later)           
        """
        Calculates the real-time tree of cell divisions based on the input labels and region properties.

        Args:
            frame_i (int): The index of the current frame.
            lab (ndarray): The label image of the current frame.
            prev_lab (ndarray): The label image of the previous frame.
            rp (list, optional): The region properties of the current frame.
            prev_rp (list, optional): The region properties of the previous frame.

        Returns:
            None
        """
        if not np.any(rp):
            rp = regionprops(lab)

        if not np.any(prev_rp):
            prev_rp = regionprops(prev_lab)

        IoA_matrix, self.IDs_curr_untracked, self.IDs_prev = calc_IoA_matrix(lab,
                                                                   prev_lab, 
                                                                   rp, 
                                                                   prev_rp
                                                                   )
        aggr_track, self.mother_daughters = mother_daughter_assign(IoA_matrix, 
                                                              IoA_thresh_daughter=self.IoA_thresh_daughter, 
                                                              min_daughter=self.min_daughter, 
                                                              max_daughter=self.max_daughter
                                                              )

        self.create_tracked_frame_tree(frame_i, self.mother_daughters, self.IDs_prev, self.IDs_curr_untracked, None, set(self.IDs_curr_untracked))

    def dict_curr_frame(self): # not done
        """
        Creates a dictionary mapping mother IDs to daughter IDs for the current frame.

        Returns:
            dict: A dictionary where the keys are mother IDs and the values are daughter IDs.
        """
        if not self.mother_daughters:
            return {}
        
        mother_IDs = [self.IDs_prev[mother] for mother, _ in self.mother_daughters]
        daughters = [daughter for _, daughter in self.mother_daughters]
        daughters = [daughter for sublist in daughters for daughter in sublist]
        daughter_IDs = IoA_index_daughter_to_ID(daughters, None, self.IDs_curr_untracked)
        printl(zip(mother_IDs, daughter_IDs))
        printl(dict(zip(mother_IDs, daughter_IDs)))
        return dict(zip(mother_IDs, daughter_IDs))
    
    def insert_lineage_df(self, lineage_df, frame_i, propagate_back=False, propagate_fwd=False, update_fams=True):
        """
        Inserts a lineage DataFrame to the lineage list at given position. If the position is greater than the length of the lineage list, a warning is printed.
        If the position is less than the length of the lineage list, the lineage DataFrame at the given position is replaced. The change can be propagated to the end of the lineage list, in both ways, or not at all, and also if the internal variables are updated or not. Assumes taht the segmentaion has not changed siginficantly, so IDs must be the same.

        Args:
            lineage_df (pandas.DataFrame): The lineage DataFrame to insert.
            frame_i (int): The index of the frame.

        Returns:
            None
        """
        if frame_i == len(self.lineage_list):
            self.lineage_list.append(lineage_df)
            if update_fams == True:
                new_fam = update_fam_from_df(self.families, lineage_df)
                new_fam = correct_fam_from_df(new_fam, lineage_df)

            if propagate_back == True:


        elif frame_i < len(self.lineage_list):
            self.lineage_list[frame_i] = lineage_df
            if update_fams == True:
                self.families = update_fam_from_df(self.families, lineage_df)
                self.families = correct_fam_from_df(self.families, lineage_df)

            if propagate_back == True:


        elif frame_i > len(self.lineage_list):
            printl(f'WARNING: Frame {frame_i} was inserted. The lineage list was only {len(self.lineage_list)} frames long, so the last known lineage tree was copy pasted up to frame {frame_i}')
            if update_fams == True:
                self.families = update_fam_from_df(self.families, lineage_df)
                self.families = correct_fam_from_df(self.families, lineage_df)


    def load_lineage_df_list():

class tracker:
    """
    Class representing a tracker for cell division in a video sequence. (Adapted from CellACDC_tracker.py)

    Attributes:
        None

    Methods:
        __init__(): Initializes the tracker object.
        track(): Tracks cell division in the video sequence.
        track_frame(): Tracks cell division in a single frame.
        updateGuiProgressBar(): Updates the GUI progress bar. (Used for GUI communication)
        save_output(): Saves the output of the tracker.
    """
    def __init__(self):
        """
        Initializes the CellACDC_normal_division_tracker object.
        """
        pass

    def track(self, 
              segm_video,
              signals=None,
              IoA_thresh = 0.8,
              IoA_thresh_daughter = 0.25,
              IoA_thresh_aggressive = 0.5,
              min_daughter = 2,
              max_daughter = 2,
              record_lineage = True
        ):
        """
        Tracks the segmented video frames and returns the tracked video.

        Args:
            segm_video (list): List of segmented video frames.
            signals (list, optional): List of signals. Defaults to None. (Used for GUI communication)
            IoA_thresh (float, optional): IoA threshold. Defaults to 0.8. (Used for tracking cells before even looking if a cell has divided)
            IoA_thresh_daughter (float, optional): IoA threshold for daughter cells. Defaults to 0.25. (Used for identifying daughter cells)
            IoA_thresh_aggressive (float, optional): Aggressive IoA threshold. Defaults to 0.5. (Used when the tracker thinks that a cell has NOT divided)
            min_daughter (int, optional): Minimum number of daughter cells. Defaults to 2. 
            max_daughter (int, optional): Maximum number of daughter cells. Defaults to 2.
            record_lineage (bool, optional): Flag to record lineage. Defaults to True. 

        Returns:
            list: Tracked video frames.
        """
        pbar = tqdm(total=len(segm_video), desc='Tracking', ncols=100)
            
        for frame_i, lab in enumerate(segm_video):
            if frame_i == 0:
                tracker = normal_division_tracker(segm_video, IoA_thresh_daughter, min_daughter, max_daughter, IoA_thresh, IoA_thresh_aggressive)
                if record_lineage:
                    tree = normal_division_lineage_tree(lab=lab, max_daughter=max_daughter)
                pbar.update()
                continue

            tracker.track_frame(frame_i)

            if record_lineage:
                mother_daughters = tracker.mother_daughters
                IDs_prev = tracker.IDs_prev
                assignments = tracker.assignments
                IDs_curr_untracked = tracker.IDs_curr_untracked
                # curr_IDs_legacy = tracker.tracked_IDs
                rp = regionprops(tracker.tracked_lab)
                curr_IDs = {obj.label for obj in rp}
                # printl(f'''Frame {frame_i}:\n{len(curr_IDs)} cells, {curr_IDs}\n{len(curr_IDs_legacy)} cells, {curr_IDs_legacy}\n{set(curr_IDs)-set(curr_IDs_legacy)}''')
                if record_lineage:
                    tree.create_tracked_frame_tree(frame_i, mother_daughters, IDs_prev, IDs_curr_untracked, assignments, curr_IDs)

            self.updateGuiProgressBar(signals)
            pbar.update()

        if record_lineage:
            self.cca_dfs = tree.lineage_list

        tracked_video = tracker.tracked_video            
        pbar.close()
        return tracked_video
    
    def track_frame(self, 
                    previous_frame_labels, 
                    current_frame_labels, 
                    IoA_thresh = 0.8,
                    IoA_thresh_daughter = 0.25,
                    IoA_thresh_aggressive = 0.5,
                    min_daughter = 2,
                    max_daughter = 2,
                    ):
        """
        Tracks cell division in a single frame. (This is used for real time tracking in the GUI)

        Args:
            previous_frame_labels: Labels of the previous frame.
            current_frame_labels: Labels of the current frame.
            IoA_thresh (float, optional): IoA threshold. Defaults to 0.8. (Used for tracking cells before even looking if a cell has divided)
            IoA_thresh_daughter (float, optional): IoA threshold for daughter cells. Defaults to 0.25. (Used for identifying daughter cells)
            IoA_thresh_aggressive (float, optional): Aggressive IoA threshold. Defaults to 0.5. (Used when the tracker thinks that a cell has NOT divided)
            min_daughter: Minimum number of daughter cells. Default is 2.
            max_daughter: Maximum number of daughter cells. Default is 2.

        Returns:
            tracked_video: Tracked video sequence.
            mothers: Set of IDs of mother cells in the current frame. (Used in GUI so it doesn't complain if IDs are missing)
        """
        
        if not np.any(current_frame_labels):
            # Skip empty frames
            return current_frame_labels

        segm_video = [previous_frame_labels, current_frame_labels]
        tracker = normal_division_tracker(segm_video, IoA_thresh_daughter, min_daughter, max_daughter, IoA_thresh, IoA_thresh_aggressive)
        tracker.track_frame(1)
        tracked_video = tracker.tracked_video
        
        mother_daughters_pairs = tracker.mother_daughters
        IDs_prev = tracker.IDs_prev
        mothers = {IDs_prev[mother[0]] for mother in mother_daughters_pairs}

        return tracked_video[-1], mothers
    
    def updateGuiProgressBar(self, signals):
        """
        Updates the GUI progress bar.

        Args:
            signals: Signals object for GUI communication.

        Returns:
            None
        """
        if signals is None:
            return
        
        if hasattr(signals, 'innerPbar_available'):
            if signals.innerPbar_available:
                # Use inner pbar of the GUI widget (top pbar is for positions)
                signals.innerProgressBar.emit(1)
                return

        signals.progressBar.emit(1)

    def save_output(self):
        """
        Used to signal to the rest of the program that the lineage tree should be saved.

        Args:
            None

        Returns:
            None
        """
        pass