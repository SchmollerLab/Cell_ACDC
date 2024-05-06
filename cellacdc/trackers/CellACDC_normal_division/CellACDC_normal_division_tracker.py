import os
from cellacdc.trackers.CellACDC.CellACDC_tracker import calc_IoA_matrix
from cellacdc.trackers.CellACDC.CellACDC_tracker import track_frame as track_frame_base
from cellacdc.core import getBaseCca_df, printl
from cellacdc.myutils import checked_reset_index
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
import pandas as pd

def reorg_daughter_cells(lineage_tree_frame):    
    """
    Reorganizes the daughter cells in the lineage tree frame.

    Args:
        lineage_tree_frame (pandas.DataFrame): The lineage tree frame containing the daughter cells.

    Returns:
        pandas.DataFrame: The lineage tree frame with reorganized daughter cells (e.g. 'daughter_ID_tree_1', 'daughter_ID_tree_2', ...)
    """
    # if not isinstance(lineage_tree_frame["sister_ID_tree"], list):
    #     return lineage_tree_frame

    # lineage_tree_frame['sister_ID_tree'] = lineage_tree_frame['sister_ID_tree'].apply(lambda x: x + [-1]*(max_daughter-1-len(x)) if len(x) < max_daughter-1 else x) # fill in -1 for fewer daughters (I think this is handled in other parts of the code already)
    sister_columns = lineage_tree_frame['sister_ID_tree'].apply(pd.Series)

    max_daughter = sister_columns.shape[1]
    new_columns = [f'sister_ID_tree_{i}' for i in range(max_daughter)]

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
    df = checked_reset_index(df)
    df = df.set_index('Cell_ID')
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

def update_generation_from_df(families, df): # may need to differentiate for cells which are not in df
    for fam in families:
        for member in fam:
            member[1] = df.loc[member[0], 'generation_num_tree'] 

    return families

def generate_fam_dict_from_df_li(df_li, frames_for_df_li=None):

    # df_li = df_li.copy()
    family_dict = {}

    if not frames_for_df_li:
        frames_for_df_li = range(len(df_li))

    for i, df in zip(frames_for_df_li, df_li):
        df_fams = df.groupby('root_ID_tree')
        for root_id, df_fam in df_fams:
            if root_id == 0:
                printl(f'''Root ID is 0. This is not allowed. Frame: {i}, Cell_ID(s): {df_fam.index.to_string()} \n
                        This is probably because the savedata is corrupted and a frame is only partially saved.''')
                continue
            # root_id = df_fam.iloc[0]['root_ID_tree']
            df_fam["frame_i"] = i
            df_fam = checked_reset_index(df_fam)
            df_fam = df_fam.set_index(['frame_i', 'Cell_ID'])
            if root_id not in family_dict.keys(): 
                family_dict[root_id] = df_fam
            else:
                family_dict[root_id] = pd.concat([family_dict[root_id], df_fam])

    return family_dict

def family_dict_to_unique_Cell_ID_df(family_dict):
    general_df = pd.concat(family_dict.values())
    general_df = checked_reset_index(general_df)
    general_df = (general_df
                  .drop_duplicates(subset='Cell_ID')
                  .set_index('Cell_ID')
                  )
    return general_df

def general_df_to_family_dict(general_df): # general are all dfs concatenated
    family_dict = {}
    general_df_fams = general_df.groupby('root_ID_tree')

    for root_id, df_fam in general_df_fams:
        family_dict[root_id] = df_fam

    return family_dict

# def general_dict_to_df_li(general_dict):
#     frames = general_dict.groupby('frame_i')


def update_dict_from_df(family_dict, df, frame_i, max_daughter=2):
    """
    Update a dictionary of dataframes (families) with new rows from a given dataframe for a specific frame. The resulting df for the entire family and all frames will not be consistent, but the single frame inserted should be.
    Please note that the following things are used to update the dict: parent_ID_tree, Cell_ID (Derived from this the following is corrected: generation_num_tree, root_ID_tree, sister_ID_tree).
    All other columns are copy pasted from input df.
    In the future it might make sense to add optons to replace.

    Args:
        family_dict (dict): A dictionary containing dataframes as values.
        df (pd.DataFrame): The dataframe containing new rows to be added.
        frame_i (int): The frame index.

    Returns:
        dict: The updated dictionary with new rows added.

    """
    df = df.copy()
    df['frame_i'] = frame_i
    IDs_covered = set()

    df = checked_reset_index(df)

    # we first need to correct generation_num_tree, root_ID_tree, sister_ID_tree
    unique_Cell_ID_df = family_dict_to_unique_Cell_ID_df(family_dict)
    corrected_df = pd.DataFrame()

    for _, Cell_info in df.iterrows(): # similar code is used in update_dict_consistency in the part which handles daughters
        parent_cell = unique_Cell_ID_df.loc[Cell_info['parent_ID_tree']]
        Cell_info['generation_num_tree'] = parent_cell['generation_num_tree'] + 1
        Cell_info['root_ID_tree'] = parent_cell['root_ID_tree']
        sister_cell_2 = set(df.loc[df['parent_ID_tree'] == Cell_info['parent_ID_tree'], 'Cell_ID'])
        sister_cell_2.remove(Cell_info['Cell_ID'])
        Cell_info['sister_ID_tree'] = list(sister_cell_2)
        corrected_df = pd.concat([corrected_df, Cell_info.to_frame().T])

    corrected_df = reorg_daughter_cells(corrected_df) # this might cause some problems

    # if not corrected_df.index.equals(pd.RangeIndex(start=0, stop=len(df))):
    #     corrected_df = (corrected_df
    #         .reset_index()
    #         .set_index(['frame_i', 'Cell_ID'])
    #         )
    # else:
    #     corrected_df = corrected_df.set_index(['frame_i', 'Cell_ID'])

    for key, df_fam in family_dict.items():
        df_fam = checked_reset_index(df_fam)
        
        if 'index' in df_fam.columns:
            df_fam = df_fam.drop(columns='index')
        if 'level_0' in df_fam.columns:
            df_fam = df_fam.drop(columns='level_0')

        df_fam = df_fam[df_fam['frame_i'] != frame_i] # select all frames except the current one
        new_rows = corrected_df[corrected_df['root_ID_tree'] == key]
        df_fam = pd.concat([df_fam, new_rows])
        df_fam = df_fam.sort_values(by=['frame_i', 'Cell_ID'])
        family_dict[key] = df_fam

        new_rows = checked_reset_index(new_rows)
        new_rows = new_rows.set_index(['frame_i', 'Cell_ID'])

        IDs_covered.update(new_rows.index.get_level_values(1))

    corrected_df = checked_reset_index(corrected_df)
    corrected_df = corrected_df.set_index(['frame_i', 'Cell_ID'])

    IDs_not_covered = set(corrected_df.index.get_level_values(1)) - IDs_covered
    if IDs_not_covered:
        for ID in IDs_not_covered:
            series = corrected_df.loc[(frame_i, ID)]
            family_dict[series['root_ID_tree']] = series.to_frame().T
    
    return family_dict

def update_dict_consistency(family_dict=None, fixed_frame_i=None, fixed_df=None, Cell_IDs_fixed=None, consider_children=True, fwd=False, bck=False, general_df=None, columns_to_replace=None): # families_to_consider=set(), iter=0):

    if consider_children and not fwd: # enforce that fwd is true when considering children
        raise ValueError('consider_children can\'t be true while fwd is not.')
    
    if not general_df and family_dict: # if we have a family dict we can create a general df
        general_df = pd.concat(family_dict.values())
    elif not general_df and not family_dict: # if we have nothing we have a problem
        raise ValueError('Either general_df or family_dict must be provided.')
    
    if not columns_to_replace: # if we don't have a list of columns to replace we take all columns except frame_i (default)
        # columns_to_replace = general_df.columns.tolist()
        # keep_cols = ['frame_i', 'Cell_ID']
        # columns_to_replace = [col for col in columns_to_replace if col not in keep_cols]
        columns_to_replace = ['generation_num_tree', 'root_ID_tree', 'sister_ID_tree'] # also implement search for sisters and add extra required columns, i just made a quick list here

    general_df = checked_reset_index(general_df)
    general_df = general_df.set_index(['frame_i', 'Cell_ID'])


    if fixed_df is None and general_df is not None: # if we don't have a given fixed df we take the one from the general df

        fixed_df = (general_df
                    .loc[fixed_frame_i]
                    .reset_index() # this is fine
                    .set_index('Cell_ID'))
        
    elif fixed_df is not None: # if we have a fixed df we are all good
        fixed_df  = checked_reset_index(fixed_df)
        fixed_df = fixed_df.set_index('Cell_ID')
    else: # if we have neither we have a problem
        raise ValueError('Either fixed_frame_df or fixed_df must be provided.')

    if not Cell_IDs_fixed: # if we don't have a list of Cell_IDs_fixed we take all Cell_IDs_fixed from the fixed_df (default)
        Cell_IDs_fixed = fixed_df.index

    general_df = checked_reset_index(general_df)

    if not fixed_frame_i:
        if fwd and bck: # this splits the df into two parts, only one is edited. Since there is no fixed frame we split so it also edits frame_i
            general_df_keep = pd.DataFrame()
            general_df_change = general_df
        elif fwd:
            general_df_keep = general_df[general_df['frame_i'] < fixed_frame_i]
            general_df_change = general_df[general_df['frame_i'] >= fixed_frame_i]
        elif bck:
            general_df_keep = general_df[general_df['frame_i'] > fixed_frame_i]
            general_df_change = general_df[general_df['frame_i'] <= fixed_frame_i]
        else:
            raise ValueError('one or both of fwd or bck must be True if fixed_frame_i is provided.')
    else:
        if fwd and bck: # this splits the df into two parts, only one is edited. Since there is a fixed frame we split so it doesn't edit frame_i
            general_df_keep = general_df[general_df['frame_i'] == fixed_frame_i]
            general_df_change = general_df[general_df['frame_i'] != fixed_frame_i]
        elif fwd:
            general_df_keep = general_df[general_df['frame_i'] <= fixed_frame_i]
            general_df_change = general_df[general_df['frame_i'] > fixed_frame_i]
        elif bck:
            general_df_keep = general_df[general_df['frame_i'] >= fixed_frame_i]
            general_df_change = general_df[general_df['frame_i'] < fixed_frame_i]
        else:
            raise ValueError('one or both of fwd or bck must be True if fixed_frame_i is provided.')
        

    general_df_change = checked_reset_index(general_df_change)
    frames = general_df_change['frame_i'].unique().astype(int)
    general_df_change = general_df_change.set_index(['Cell_ID', 'frame_i'])
    

    for Cell_ID in Cell_IDs_fixed: # replace values for the cells in the general df # definely needs to be optimized
        for frame in frames:
            if Cell_ID not in general_df_change.loc[(slice(None), frame), :].index.get_level_values(0): # if the cell is not in the frame we skip it
                continue

            general_df_change.loc[(Cell_ID, frame), columns_to_replace] = fixed_df.loc[Cell_ID, columns_to_replace] # we replace necessary (Need to define still)

    general_df_keep = checked_reset_index(general_df_keep)
    general_df_change  = checked_reset_index(general_df_change)

    printl('general_df_keep, general_df_change')
    import pandasgui
    pandasgui.show(general_df_keep, general_df_change)


    general_df = pd.concat([general_df_keep, general_df_change]) # we put the df back together

    import pandasgui
    pandasgui.show(general_df)

    if consider_children: # this also enforces that fwd is true (See ValueError above)
        unique_df = checked_reset_index(general_df)
        unique_df = (unique_df
                     .drop_duplicates(subset='Cell_ID')
                     .reset_index() # this is fine
                     .set_index('Cell_ID')
                     ) # we drop all duplicates (different frames) to reduce overhead

        children_df = unique_df[unique_df['parent_ID_tree'].isin(Cell_IDs_fixed)] # we select all children of the cells we are considering
        if children_df.empty: # if there are no children we are done
            family_dict = general_df_to_family_dict(general_df)
            return family_dict
        else: # else we construct a new fixed_df and run the update recursively.
            new_children_df = pd.DataFrame()
            for Cell_id, row in children_df.iterrows(): # we need to edit the daughters to be consistent
                parent_cell = unique_df.loc[row['parent_ID_tree']]
                row['generation_num_tree'] = parent_cell['generation_num_tree'] + 1
                row['root_ID_tree'] = parent_cell['root_ID_tree']
                sister_cell = unique_df.loc[unique_df['parent_ID_tree'] == row['parent_ID_tree']]['Cell_ID']
                row['sister_ID_tree'] = list(sister_cell)
                new_children_df = new_children_df.append(row)
                
            family_dict = update_dict_consistency(fixed_df=new_children_df, columns_to_replace=columns_to_replace, general_df=general_df, fwd=fwd, bck=bck, consider_children=consider_children)
            return family_dict
    else:
        family_dict = general_df_to_family_dict(general_df)
        return family_dict

def check_dict_consistency(family_dict):
    printl('Checking consistency of family_dict not implemented yet.')
    consistant = True
    return consistant #placeholder for later implementation

def dict_to_fams(family_dict):
    families = []
    for key, df_fam in family_dict.items():
        df_fam = checked_reset_index(df_fam).set_index('Cell_ID')

        df_fam = df_fam[~df_fam.index.duplicated(keep='first')]
        # import pandasgui
        # pandasgui.show(df_fam)
        # printl(key)
        
        familiy = [(key, df_fam.loc[key, 'generation_num_tree'])] # first entry is the root ID (should be first cell)
        df_fam = (df_fam
                  .drop(key)
                  .reset_index()
                  ) # this is fine
        familiy = familiy + list(zip(df_fam['Cell_ID'].tolist(), df_fam['generation_num_tree'].tolist()))
        families.append(familiy)
    return families
    
def dict_to_df_li(family_dict): # may need to drop some columns
    df_list = []

    general_df = pd.concat(family_dict.values())
    general_df = checked_reset_index(general_df)

    general_df = general_df.sort_values(by=['frame_i', 'Cell_ID'])

    if 'index' in general_df.columns:
        general_df = general_df.drop(columns='index')

    if 'level_0' in general_df.columns:
        general_df = general_df.drop(columns='level_0')
    
    frames = general_df['frame_i'].unique().astype(int)

    frames = frames[~np.isnan(frames)]
    frames_for_dfs = [int(frame) for frame in frames]

    df_group = general_df.groupby('frame_i')
    for _, df in df_group:
        df = (df
              .drop('frame_i', axis=1)
              .reset_index() # i dorp later
              .set_index('Cell_ID')
              )
        if "level_0" in df.columns:
            df = df.drop(columns="level_0")
        if "index" in df.columns:
            df = df.drop(columns="index")

        df_list.append(df)

    return df_list, frames_for_dfs

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
                                                                                       IDs_curr_untracked=self.IDs_curr_untracked, IDs_prev=self.IDs_prev, return_all=True)
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
            raise ValueError('Either lab or first_df must be provided')
        
        self.max_daughter = max_daughter
        self.min_daughter = min_daughter 
        self.IoA_thresh_daughter = IoA_thresh_daughter
        self.mother_daughters = [] # just for the dict_curr_frame stuff...
        self.frames_for_dfs = []
        self.family_dict = {}

        self.families = []

        self.init_lineage_tree(lab, first_df)

        printl('Lineage tree initialized.')

    def init_lineage_tree(self, lab=None, first_df=None):
        if lab.any() and first_df:
            raise ValueError('Only one of lab and first_df can be provided.')
    
        if lab.any():
            added_lineage_tree = []

            rp = regionprops(lab.copy())
            for obj in rp:
                label = obj.label
                self.families.append([(label, 1)])
                added_lineage_tree.append((-1, label, -1, 1, label, [-1] * (self.max_daughter-1)))

            cca_df = added_lineage_tree_to_cca_df(added_lineage_tree)
            cca_df = reorg_daughter_cells(cca_df)
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

            for daughter_ID in daughter_IDs:
                daughter_IDs_copy = daughter_IDs.copy()
                daughter_IDs_copy.remove(daughter_ID)
                daughter_IDs_copy = daughter_IDs_copy + [-1] * (self.max_daughter - len(daughter_IDs_copy) -1)
                added_lineage_tree.append((frame_i, daughter_ID, mother_ID, generation, origin_id, daughter_IDs_copy))

        cca_df = added_lineage_tree_to_cca_df(added_lineage_tree)
        cca_df = pd.concat([self.lineage_list[-1], cca_df], axis=0)
        cca_df = reorg_daughter_cells(cca_df)
        cca_df = filter_current_IDs(cca_df, curr_IDs)
        self.lineage_list.append(cca_df)

    def real_time_tree(self, frame_i, lab, prev_lab, rp=None, prev_rp=None):        
        """
        Calculates the real-time tree of cell divisions based on the input labels and region properties.

        Args:
            frame_i (int): The index of the current frame.
            lab (ndarray): The labeled image of the current frame.
            prev_lab (ndarray): The labeled image of the previous frame.
            rp (list, optional): The region properties of the current frame.
            prev_rp (list, optional): The region properties of the previous frame.

        Returns:
            None
        """
        if not np.any(rp):
            rp = regionprops(lab)

        if not np.any(prev_rp):
            prev_rp = regionprops(prev_lab)

        IoA_matrix, self.IDs_curr_untracked, self.IDs_prev = calc_IoA_matrix(lab, prev_lab, rp, prev_rp)
        aggr_track, self.mother_daughters = mother_daughter_assign(IoA_matrix, IoA_thresh_daughter=self.IoA_thresh_daughter, min_daughter=self.min_daughter, max_daughter=self.max_daughter)

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
    
    def insert_lineage_df(self, lineage_df, frame_i, propagate_back=False, propagate_fwd=False, update_fams=True, consider_children=True):
        """
        Inserts a lineage DataFrame to the lineage list at given position. If the position is greater than the length of the lineage list, a warning is printed.
        If the position is less than the length of the lineage list, the lineage DataFrame at the given position is replaced. The change can be propagated to the end of the lineage list, in both ways, or not at all, and also if the internal variables are updated or not. Assumes that the segmentaion has not changed siginficantly, so IDs must be the same.

        Args:
            lineage_df (pandas.DataFrame): The lineage DataFrame to insert.
            frame_i (int): The index of the frame.

        Returns:
            None
        """
        if list(range(len(self.lineage_list))) != self.frames_for_dfs:
            raise ValueError('Frames for dfs is not correct. This is not properly supported and I think I might drop self.frames_for_dfs in the future...')

        if frame_i == len(self.lineage_list):
            if not self.family_dict: # generate the family_dict if it is not present
                self.family_dict = generate_fam_dict_from_df_li(self.lineage_list)

            self.lineage_list.append(lineage_df)
            self.frames_for_dfs.append(frame_i)

            self.family_dict = update_dict_from_df(self.family_dict, lineage_df, frame_i, self.max_daughter)
            

            if propagate_back == True:
                self.family_dict = update_dict_consistency(family_dict=self.family_dict, fixed_frame_i=frame_i, fixed_df=lineage_df, consider_children=consider_children, fwd=False, bck=True)
            if update_fams == True:
                printl('Here!')
                self.families = dict_to_fams(self.family_dict)

            self.lineage_list, self.frames_for_dfs = dict_to_df_li(self.family_dict)


        elif frame_i < len(self.lineage_list):
            if not self.family_dict: # generate the family_dict if it is not present
                self.family_dict = generate_fam_dict_from_df_li(self.lineage_list)

            self.lineage_list[frame_i] = lineage_df

            self.family_dict = update_dict_from_df(self.family_dict, lineage_df, frame_i, self.max_daughter)

            if propagate_back == True or propagate_fwd == True:
                import pandasgui
                printl('Here!')
                pandasgui.show(*self.family_dict.values())
                self.family_dict = update_dict_consistency(family_dict=self.family_dict, fixed_frame_i=frame_i, fixed_df=lineage_df, consider_children=consider_children, fwd=propagate_fwd, bck=propagate_back)
            if update_fams == True:
                printl('Here!')
                import pandasgui
                pandasgui.show(*self.family_dict.values())
                self.families = dict_to_fams(self.family_dict)


            # from pandasgui import show as pgshow
            # dfs = self.family_dict.values()
            # pgshow(*dfs, *self.lineage_list)
                
            self.lineage_list, self.frames_for_dfs  = dict_to_df_li(self.family_dict)

        elif frame_i > len(self.lineage_list):
            printl(f'WARNING: Frame_i {frame_i} was inserted. The lineage list was only {len(self.lineage_list)} frames long, so the last known lineage tree was copy pasted up to frame_i {frame_i}')

            original_length = len(self.lineage_list)
            self.lineage_list = self.lineage_list + [self.lineage_list[-1]] * (frame_i - len(self.lineage_list))

            self.family_dict = generate_fam_dict_from_df_li(self.lineage_list) # regenerate the family_dict

            self.lineage_list.append(lineage_df)

            frame_is = list(range(len(self.lineage_list)-original_length))
            self.frames_for_dfs = self.frames_for_dfs + frame_is

            self.family_dict = update_dict_from_df(self.family_dict, lineage_df, frame_i, self.max_daughter)

            if propagate_back == True or propagate_fwd == True:
                self.family_dict = update_dict_consistency(family_dict=self.family_dict, fixed_frame_i=frame_i, fixed_df=lineage_df, consider_children=consider_children, fwd=propagate_fwd, bck=propagate_back)
            if update_fams == True:
                printl('Here!')
                self.families = dict_to_fams(self.family_dict)
            self.lineage_list, self.frames_for_dfs  = dict_to_df_li(self.family_dict)


    def load_lineage_df_list(self, df_li):
        # SUpport for first_frame was removed since it is not necessary, just make the df_li correct...
        # Also the tree needs to be init before. Also if df_li does not contain any relevenat dfs, nothing happens
        printl('Loading!')
        df_li_new = []
        for i, df in enumerate(df_li):
            if 'frame_i' in df.columns:
                df = df.drop('frame_i', axis=1)
            if 'generation_num_tree' in df.columns and not (df['generation_num_tree'] == 0).all():
                if not df.index.name == 'Cell_ID':
                    df = (df
                            .reset_index()
                            .set_index('Cell_ID')
                            )
                
                if "level_0" in df.columns:
                    df = df.drop(columns="level_0")
                if "index" in df.columns:
                    df = df.drop(columns="index")

                self.frames_for_dfs.append(i)
                df_li_new.append(df)

        if df_li_new:
            self.lineage_list = df_li_new
            self.family_dict = generate_fam_dict_from_df_li(self.lineage_list, frames_for_df_li=self.frames_for_dfs)
            printl('Here!')
            self.families = dict_to_fams(self.family_dict)
            self.lineage_list, self.frames_for_dfs = dict_to_df_li(self.family_dict)

    def export_df(self, frame_i):
        df = self.lineage_list[frame_i].copy()
        
        df = (df
              .reset_index()
              .set_index('Cell_ID')
              )
        
        if 'frame_i' in df.columns:
            df = df.drop('frame_i', axis=1)

        if "level_0" in df.columns:
            df = df.drop(columns="level_0")
        if "index" in df.columns:
            df = df.drop(columns="index")
        
        return df

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