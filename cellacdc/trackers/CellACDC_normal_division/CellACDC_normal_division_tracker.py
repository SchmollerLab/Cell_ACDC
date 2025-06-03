import os
from cellacdc.trackers.CellACDC.CellACDC_tracker import calc_Io_matrix
from cellacdc.trackers.CellACDC.CellACDC_tracker import track_frame as track_frame_base
from cellacdc.core import getBaseCca_df, printl
from cellacdc.myutils import checked_reset_index, checked_reset_index_Cell_ID
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
import pandas as pd
from cellacdc.myutils import exec_time
from cellacdc._types import NotGUIParam
import copy

def filter_cols(df):
    """
    Filters the columns of a DataFrame based on a predefined set of column names.
    'generation_num_tree', 'root_ID_tree', 'sister_ID_tree', 'parent_ID_tree', 'parent_ID_tree', 'emerg_frame_i', 'division_frame_i'
    plus any column that starts with 'sister_ID_tree'

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.

    Returns:
    - pandas.DataFrame: The filtered DataFrame containing only the specified columns.
    """
    lin_tree_cols = {'generation_num_tree', 'root_ID_tree', 
                     'sister_ID_tree', 'parent_ID_tree', 
                     'parent_ID_tree', 'emerg_frame_i', 
                     'division_frame_i', 'is_history_known'}
    sis_cols = {col for col in df.columns if col.startswith('sister_ID_tree')}
    lin_tree_cols = lin_tree_cols | sis_cols
    return df[list(lin_tree_cols)]

def reorg_sister_cells_for_export(lineage_tree_frame):
    """
    Reorganizes the daughter cells in the lineage tree frame for export.
    Translates the lists from 'sister_ID_tree' to 'sister_ID_tree_1', 'sister_ID_tree_2',
    while leaving the first entry from the list in 'sister_ID_tree'

    Parameters:
    - lineage_tree_frame (pandas.DataFrame): The lineage tree frame containing the daughter cells.

    Returns:
    - pandas.DataFrame: The lineage tree frame with reorganized daughter cells (e.g. 'daughter_ID_tree_1', 'daughter_ID_tree_2', ...)
    """
    if lineage_tree_frame.empty:
        return lineage_tree_frame
    
    old_sister_columns = {col for col in lineage_tree_frame.columns if col.startswith('sister_ID_tree')}

    sister_columns = lineage_tree_frame['sister_ID_tree'].apply(pd.Series)

    max_daughter = sister_columns.shape[1]
    new_columns = [f'sister_ID_tree_{i}' for i in range(max_daughter)]

    lineage_tree_frame = lineage_tree_frame.drop(columns=old_sister_columns)
    lineage_tree_frame[new_columns] = sister_columns
    lineage_tree_frame['sister_ID_tree'] = sister_columns[0]

    return lineage_tree_frame

def reorg_sister_cells_inner_func(row):
    """
    Reorganizes the sister cells in a row of a DataFrame. Used as an inner function for apply.

    Parameters:
    - row (pandas.Series): The input row of the DataFrame (alredy filtered for the sister columns).
    Returns:
    - pandas.Series: The reorganized row with the sister cells.
    """

    values = [int(i) for i in row if i not in {0, -1} and not np.isnan(i)] or [-1]
    values = list(set(values))  
    return values


def reorg_sister_cells_for_import(df):
    """
    Reorganizes the sister cells for import.

    This function takes a DataFrame `df` as input and performs the following steps:
    1. Identifies the sister columns in the DataFrame.
    2. Removes any values that are equal to 0 or -1 from the sister columns. (Which both represent no sister cell)
    3. Converts the remaining values in the sister columns to a set.
    4. Converts the set of values to a list if it is not empty, otherwise assigns [-1] to the sister column. (It actually shouldn't be empty, but just in case...)
    5. Removes the sister columns from the DataFrame. And adds the list as the new 'sister_ID_tree' column.

    Parameters:
    - df (pandas.DataFrame): The input DataFrame.

    Returns:
    - df (pandas.DataFrame): The modified DataFrame with reorganized sister cells.
    """
    sister_cols = [col for col in df.columns if col.startswith('sister_ID_tree')] # handling sister columns
    df.loc[:, 'sister_ID_tree'] = df[sister_cols].apply(reorg_sister_cells_inner_func, axis=1)
    sister_cols.remove('sister_ID_tree')
    df = df.drop(columns=sister_cols)
    df = checked_reset_index_Cell_ID(df)
    return df

def mother_daughter_assign(IoA_matrix, IoA_thresh_daughter, min_daughter, max_daughter, IoA_thresh_instant=None):
    """
    Identifies cells that have not undergone division based on the input IoA matrix.

    Parameters:
    - IoA_matrix (numpy.ndarray): Matrix representing the Intersection over Area (IoA) values between cells.
    - IoA_thresh_daughter (float): Threshold value for considering a cell as a daughter cell.
    - min_daughter (int): Minimum number of daughter cells required for considering a cell as a mother cell.
    - max_daughter (int): Maximum number of daughter cells required for considering a cell as a mother cell.

    Returns:
    - tuple: A tuple containing two elements:
        - aggr_track (list): A list of indices of cells that have not undergone division.
        - mother_daughters (list): A list of tuples, where each tuple contains the index of a mother cell and a list of indices of its daughter cells. (INDICIES ARE BASED ON THE INPUT IoA MATRIX)
    """
    mother_daughters = []
    aggr_track = []
    daughter_range = range(min_daughter, max_daughter+1, 1)
    instant_accept = []

    IoA_thresholded = IoA_matrix >= IoA_thresh_daughter

    if IoA_thresh_instant is not None:
        IoA_instant_accept = IoA_matrix >= IoA_thresh_instant
    else:
        IoA_instant_accept = np.zeros_like(IoA_matrix)

    nrows, ncols = IoA_matrix.shape
    for j in range(ncols):
        if IoA_instant_accept[:, j].any():
            instant_accept.append(j)
            continue
        
        high_IoA_indices = np.where(IoA_thresholded[:, j])[0]

        if not high_IoA_indices.size:
            continue
        elif not len(high_IoA_indices) in daughter_range:
            aggr_track.extend(high_IoA_indices)
        else:
            mother_daughters.append((j, high_IoA_indices))

    should_remove_idx = []
    for mother, daughters in mother_daughters:
        for daughter in daughters:
            high_IoA_greater_1 = np.count_nonzero(IoA_thresholded[daughter]) > 1
            if high_IoA_greater_1:
                should_remove_idx.append(True) 
                break
        else:
            should_remove_idx.append(False)
    
    # printl(f'length of mother_daughters: {len(mother_daughters), len(should_remove_idx)}')
    mother_daughters = [mother_daughters[i] for i, remove in enumerate(should_remove_idx) if not remove]

    # daughters_li = []
    # for _, daughters in mother_daughters:
    #     daughters_li.extend(daughters)

    return aggr_track, mother_daughters

def added_lineage_tree_to_cca_df(added_lineage_tree):
    """
    Converts the added lineage tree into a DataFrame with specific columns.

    Parameters:
    - added_lineage_tree (list): List of lists, second level lists contain the following elements in this order:
        - 'Division frame'
        - 'Daughter ID' ("ID")
        - 'Mother ID'
        - 'Generation'
        - 'Origin ID'
        - 'Sister IDs'

    Returns:
    - pandas.DataFrame: The converted DataFrame with the following columns:
        - 'emerg_frame_i'
        - 'division_frame_i'
        - 'generation_num_tree'
        - 'parent_ID_tree'
        - 'root_ID_tree'
        - 'sister_ID_tree'
        - 'is_history_known'
    """
    if not added_lineage_tree:
        return pd.DataFrame()

    # Use zip to unpack columns efficiently
    emerg_frame_i, cell_id, parent_id, gen_num, root_id, sister_ids = zip(*added_lineage_tree)
    cca_df = pd.DataFrame({
        'Cell_ID': cell_id,
        'emerg_frame_i': emerg_frame_i,
        'division_frame_i': emerg_frame_i,
        'generation_num_tree': gen_num,
        'parent_ID_tree': parent_id,
        'root_ID_tree': root_id,
        'sister_ID_tree': sister_ids,
    })
    cca_df['is_history_known'] = (cca_df['parent_ID_tree'] != -1).astype(int)
    cca_df = cca_df.set_index('Cell_ID')
    return cca_df

def filter_current_IDs(df, current_IDs):
    """
    Filters for current IDs.

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the IDs.
    - indices_to_remove (list): A list of indices NOT to be removed.

    Returns:
    - pandas.DataFrame: The DataFrame with only the current IDs.
    """
    df = checked_reset_index_Cell_ID(df)
    return df[df.index.isin(current_IDs)]

def IoA_index_daughter_to_ID(daughters, assignments, IDs_curr_untracked):
    """
    Converts a list of daughter indices (IoA Matrix) to their corresponding IDs.

    Parameters:
    - daughters (list): List of daughter indices.
    - assignments (dict): Dictionary mapping IDs to assigned values.
    - IDs_curr_untracked (list): List of current untracked IDs.

    Returns:
    - list: List of daughter IDs.

    """

    if daughters is None:
        return
    
    daughter_IDs = []
    for daughter in daughters:
        if assignments:
            daughter_IDs.append(assignments[IDs_curr_untracked[daughter]])
        else:
            daughter_IDs.append(IDs_curr_untracked[daughter])

    return daughter_IDs

def update_fam_dynamically(families, fixed_df, Cell_IDs_fixed=None):
    if Cell_IDs_fixed is None:
        Cell_IDs_fixed = fixed_df.index
    for idx, family in enumerate(families):
        # Keep only cellinfos where cell_id is in Cell_IDs_fixed
        families[idx] = [cellinfo for cellinfo in family if cellinfo[0] not in Cell_IDs_fixed]
    
    families = [family for family in families if family]  # Remove empty families
    handled_cells = set()
    for family in families:
        root_ID = family[0][0]  # The first cell in the family is the root
        relevant_cells = fixed_df.loc[fixed_df['root_ID_tree'] == root_ID]
        for relevant_cell in relevant_cells.index:
            # Update the family with the generation number and root ID
            family.append((relevant_cell, relevant_cells.loc[relevant_cell, 'generation_num_tree']))
        handled_cells.update(relevant_cells.index)
    
    for cell_id in Cell_IDs_fixed:
        if cell_id not in handled_cells:
            # If the cell is not handled, create a new family for it
            families.append([(cell_id, fixed_df.loc[cell_id, 'generation_num_tree'])])
    
    return families

def update_consistency(df_li, fixed_frame_i=None, fixed_df=None, Cell_IDs_fixed=None, consider_children=True, families=None):
    """
    Updates the consistency of lineage information across a list of DataFrames representing cell tracking over time.

    This function propagates fixed lineage information (such as generation number, root ID, parent ID, and sister IDs)
    from a reference frame or DataFrame to all relevant cells in all frames, ensuring consistency in the lineage tree.
    It can also update the lineage information for the children of fixed cells. If families are provided, it will update them as well.

    There are several ways to call this function:
    1. If `fixed_frame_i` is provided (and `fixed_df` is not), the fixed DataFrame is taken from `df_li[fixed_frame_i]`.
    2. If `fixed_df` is provided (and `fixed_frame_i` is not), the fixed frame index is inferred from `fixed_df`.
    3. If `Cell_IDs_fixed` is provided, only those Cell IDs are considered for consistency updates; otherwise, all IDs in the fixed DataFrame are used.

    Parameters:
    - df_li (list of pd.DataFrame): List of DataFrames, one per frame, each indexed by Cell_ID.
    - fixed_frame_i (int, optional): Index of the frame to use as the reference for fixed lineage information.
    - fixed_df (pd.DataFrame, optional): DataFrame containing fixed lineage information for selected cells.
    - Cell_IDs_fixed (iterable, optional): Iterable of Cell IDs to consider for consistency updates. If None, all IDs in `fixed_df` are used.
    - consider_children (bool, default=True): If True, also updates the lineage information for children of fixed cells.
    - families (list, optional): List of families to update along the way. If None, it is not used.

    Returns:
    - list of pd.DataFrame: The updated list of DataFrames with consistent lineage information.

    Notes:
    - The columns updated for consistency are: 'generation_num_tree', 'root_ID_tree', 'sister_ID_tree', 'parent_ID_tree'.
    - The function maintains a lookup dictionary and a list of fixed DataFrames to efficiently propagate updates.
    - Sister IDs are stored as sets, excluding the cell's own ID; if a cell has no sisters, the value is set to {-1}.
    """
    columns_to_replace = ['generation_num_tree', 
                          'root_ID_tree', 
                          'sister_ID_tree', 
                          'parent_ID_tree']

    if fixed_df is not None:
        fixed_df = checked_reset_index_Cell_ID(fixed_df)

    elif fixed_frame_i is not None:
        fixed_df = checked_reset_index_Cell_ID(df_li[fixed_frame_i])
    else:
        raise ValueError('Either fixed_frame_i or fixed_df must be provided.')

    if Cell_IDs_fixed is not None: # if we have a list of Cell_IDs to consider
        fixed_df = fixed_df[fixed_df.index.isin(Cell_IDs_fixed)]
    else: 
        Cell_IDs_fixed = fixed_df.index

    fixed_dfs = [fixed_df]
    fixed_dfs_lookup = {fixed_df.index[i]: 0 for i in range(len(fixed_df))}
    Cell_IDs_fixed = set(Cell_IDs_fixed) # we convert to a set for faster lookups

    if families is not None:
        families = update_fam_dynamically(families, fixed_df, Cell_IDs_fixed)

    for frame_df in df_li:
        frame_df = checked_reset_index_Cell_ID(frame_df)

        if consider_children:
            children = frame_df[frame_df['parent_ID_tree'].isin(Cell_IDs_fixed)]
            if not children.empty:
                for parent_ID, children in children.groupby('parent_ID_tree'):
                    parent_cell_loc = fixed_dfs_lookup[parent_ID] # we get the parent cell from the lookup dictionary
                    parent_line = fixed_dfs[parent_cell_loc].loc[parent_ID]
                    children['root_ID_tree'] = parent_line['root_ID_tree']
                    children['generation_num_tree'] = parent_line['generation_num_tree'] + 1
                    sisters = list(children.index)
                    children['sister_ID_tree'] = [
                        [s for s in sisters if s != cell_id] if len(sisters) > 1 else [-1]
                        for cell_id in children.index
                    ]
                
                if families is not None:
                    families = update_fam_dynamically(families, children)

                Cell_IDs_fixed = Cell_IDs_fixed.union(children.index) # we add the children IDs to the set of Cell_IDs_fixed
                fixed_dfs.append(children) # we append the children to the fixed_dfs list
                indx = len(fixed_dfs) - 1 # we get the index of the children in the fixed_dfs list
                fixed_dfs_lookup.update({children.index[i]: indx for i in range(len(children))}) # we update the lookup dictionary with the children

        relevant_cells_mask = frame_df.index.isin(Cell_IDs_fixed)
        if not relevant_cells_mask.any():
            continue

        relevant_cell_ids = frame_df.index[relevant_cells_mask]
        relevant_fixed_dfs_indx = {
            fixed_dfs_lookup[cell_id] for cell_id in relevant_cell_ids
        }

        for indx in relevant_fixed_dfs_indx:
            fixed_df = fixed_dfs[indx]
            # Find the intersection of indices
            common_idx = frame_df.index.intersection(fixed_df.index)
            if not common_idx.empty:
                frame_df.loc[common_idx, columns_to_replace] = fixed_df.loc[common_idx, columns_to_replace]


    if families is not None:
        return df_li, families
    
    return df_li

class normal_division_tracker:
    """
    A class that tracks cell divisions in a video sequence. The tracker uses the Intersection over Area (IoA) metric to track cells and identify daughter cells.

    Attributes:
    - IoA_thresh_daughter (float): The IoA threshold for identifying daughter cells.
    - min_daughter (int): The minimum number of daughter cells.
    - max_daughter (int): The maximum number of daughter cells.
    - IoA_thresh (float): The IoA threshold for tracking cells, which is used to instantly track a cell before looking for daughters.
    - IoA_thresh_aggressive (float): The aggressive IoA threshold for tracking cells.
    - segm_video (ndarray): The segmented video sequence.
    - tracked_video (ndarray): The tracked video sequence.
    - assignments (dict): A dictionary mapping untracked cell IDs to tracked cell IDs. (Only NEW cells are in this dictionary, so things the tracker could not assign to a cell in the previous frame and was given a new unique ID.)
    - IDs_prev (list): A list mapping index from IoA matrix to IDs. (Index in list = Index in IoA matrix)
    - rp (list): The region properties of the current frame.
    - mother_daughters (list): A list of tuples, where each tuple contains the index of a mother cell and a list of indices of its daughter cells. (INDICIES ARE BASED ON THE INPUT IoA MATRIX and not the IDs in the video.)
    - IDs_curr_untracked (list): A list of current cell IDs that are untracked. (INDICIES ARE BASED ON THE INPUT IoA MATRIX and not the IDs in the video.)
    - tracked_IDs (set): A set of all cell IDs in current frame after tracking.
    - tracked_lab (ndarray): The tracked current frame.

    Methods:
    - __init__(self, segm_video, IoA_thresh_daughter, min_daughter, max_daughter, IoA_thresh, IoA_thresh_aggressive): Initializes the normal_division_tracker object. Mainly sets the parameters. And introduces the tracked_videos dummy and the first frame.
    - track_frame(self, frame_i, lab=None, prev_lab=None, rp=None, prev_rp=None): Tracks a single frame in the video sequence.
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

        Parameters:
        - segm_video (ndarray): The segmented video sequence.
        - IoA_thresh_daughter (float): The IoA threshold for identifying daughter cells.
        - min_daughter (int): The minimum number of daughter cells.
        - max_daughter (int): The maximum number of daughter cells.
        - IoA_thresh (float): The IoA threshold for tracking cells.
        - IoA_thresh_aggressive (float): The aggressive IoA threshold for tracking cells. This is applied when the tracker thinks that a cell has NOT divided.
        """

        self.IoA_thresh_daughter = IoA_thresh_daughter
        self.min_daughter = min_daughter
        self.max_daughter = max_daughter
        self.IoA_thresh = IoA_thresh
        self.IoA_thresh_aggressive = IoA_thresh_aggressive
        self.segm_video = segm_video

        self.tracked_video = np.zeros_like(segm_video)
        self.tracked_video[0] = segm_video[0]

    def track_frame(self, frame_i, lab=None, prev_lab=None, rp=None, prev_rp=None,
                    IDs=None):
        """
        Tracks a single frame in the video sequence.

        Parameters:
        - frame_i (int): The index of the frame to track.
        - lab (ndarray, optional): The segmented labels of the current frame. Defaults to None.
        - prev_lab (ndarray, optional): The segmented labels of the previous frame. Defaults to None.
        - rp (list, optional): The region properties of the current frame. Defaults to None.
        - prev_rp (list, optional): The region properties of the previous frame. Defaults to None.
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
            prev_rp = regionprops(prev_lab.copy())

        IoA_matrix, self.IDs_curr_untracked, self.IDs_prev = calc_Io_matrix(lab,
                                                                             prev_lab,
                                                                             self.rp,
                                                                             prev_rp,
                                                                             IDs=IDs,
                                                                             )
        self.aggr_track, self.mother_daughters = mother_daughter_assign(IoA_matrix,
                                                                        IoA_thresh_daughter=self.IoA_thresh_daughter,
                                                                        min_daughter=self.min_daughter,
                                                                        max_daughter=self.max_daughter,
                                                                        IoA_thresh_instant=self.IoA_thresh
                                                                        )
        self.tracked_lab, IoA_matrix, self.assignments, _ = track_frame_base(prev_lab,
                                                                             prev_rp,
                                                                             lab,
                                                                             self.rp,
                                                                             IoA_thresh=self.IoA_thresh,
                                                                             IoA_matrix=IoA_matrix,
                                                                             aggr_track=self.aggr_track,
                                                                             IoA_thresh_aggr=self.IoA_thresh_aggressive,
                                                                             IDs_curr_untracked=self.IDs_curr_untracked, 
                                                                             IDs_prev=self.IDs_prev, 
                                                                             return_all=True,
                                                                             mother_daughters=self.mother_daughters
                                                                             )
        
        # not_self_assignemtns = {k: v for k, v in self.assignments.items() if k != v}
        # mother_IDs = [self.IDs_prev[mother] for mother, _ in self.mother_daughters]
        # daughter_indx = [daughters for _, daughters in self.mother_daughters]
        # daughter_indx = np.array(daughter_indx)
        # try:
        #     daughter_indx = np.concatenate(daughter_indx)
        # except ValueError:
        #     pass
        # daughter_IDs = IoA_index_daughter_to_ID(daughter_indx.tolist(), self.assignments, self.IDs_curr_untracked)
        # printl(f'Frame {frame_i} tracked, not_self_assignemtns: {not_self_assignemtns}, mothers {mother_IDs}, daughters {daughter_IDs}')
        
        # self.tracked_IDs = set(tracked_IDs).union(set(self.assignments.values()))
        self.tracked_video[frame_i] = self.tracked_lab

class normal_division_lineage_tree:
    """
    Class for tracking and managing cell lineage trees during normal cell division across multiple frames.

    Attributes:
        max_daughter (int): Maximum number of daughter cells expected per division.
        min_daughter (int): Minimum number of daughter cells expected per division.
        IoA_thresh_daughter (float): Intersection over Area (IoA) threshold for identifying daughter cells.
        mother_daughters (list): List of tuples representing mother-daughter relationships for the current frame.
        frames_for_dfs (set): Set of frame indices for which lineage data is available.
        need_update_gen_df (bool): Flag indicating if the general DataFrame needs updating.
        families (list): List of families, where each family is a list of (Cell_ID, generation_num_tree) tuples.
        lineage_list (list): List of DataFrames, one per frame, representing the lineage tree.

    Methods:
        __init__(lab=None, first_df=None, frame_i=0, max_daughter=2, min_daughter=2, IoA_thresh_daughter=0.25)
            Initialize the lineage tree with either a labeled image or a DataFrame.

        init_lineage_tree(lab=None, first_df=None)
            Initialize the lineage tree structure from a labeled image or DataFrame.

        create_tracked_frame(frame_i, mother_daughters, IDs_prev, IDs_curr_untracked, assignments, curr_IDs, new_IDs)
            Add a new frame to the lineage tree, updating families and tracking new and divided cells.

        real_time(frame_i, lab, prev_lab, rp=None, prev_rp=None)
            Compute the lineage tree for a frame in real time, typically for GUI annotation.

        update_df_li_locally(df, frame_i)
            Update the lineage DataFrame for a specific frame, correcting lineage columns for consistency.

        insert_lineage_df(lineage_df, frame_i, update_fams=True, consider_children=True)
            Insert or replace a lineage DataFrame at a given frame index, optionally updating families and propagating changes.

        propagate(frame_i, Cell_IDs_fixed=None)
            Propagate changes from a specific frame to the entire lineage tree and families.

        load_lineage_df_list(df_li)
            Load a list of lineage DataFrames, reconstructing the lineage tree and families.

        export_df(frame_i)
            Export the lineage DataFrame for a specific frame, cleaning up auxiliary columns.

        export_lin_tree_info(frame_i)
            Return information about new, orphan, and lost cells between two consecutive frames.
    """ 

    def __init__(self, lab=None, first_df=None, frame_i=0, max_daughter=2, min_daughter=2, IoA_thresh_daughter=0.25):
        """
        Initialize the lineage tree for normal cell divisions.

        Args:
            lab (ndarray, optional): Labeled image of cells for the initial frame.
            first_df (pd.DataFrame, optional): Initial DataFrame representing the lineage tree.
            frame_i (int, optional): Index of the initial frame. Defaults to 0.
            max_daughter (int, optional): Maximum number of daughter cells per division. Defaults to 2.
            min_daughter (int, optional): Minimum number of daughter cells per division. Defaults to 2.
            IoA_thresh_daughter (float, optional): IoA threshold for identifying daughter cells. Defaults to 0.25.

        Raises:
            ValueError: If neither lab nor first_df is provided.
        """

        self.max_daughter = max_daughter
        self.min_daughter = min_daughter
        self.IoA_thresh_daughter = IoA_thresh_daughter
        self.mother_daughters = [] # just for the dict_curr_frame stuff...
        self.frames_for_dfs = set([frame_i])
        self.need_update_gen_df = False # this is only when using the quick option in update_gen_df_from_df

        self.families = []

        self.init_lineage_tree(lab, first_df)
        

    def init_lineage_tree(self, lab=None, first_df=None):
        """
        Initialize the lineage tree structure from a labeled image or DataFrame.

        Args:
            lab (ndarray, optional): Labeled image representing the cells.
            first_df (pd.DataFrame, optional): Initial DataFrame representing the cells.

        Raises:
            ValueError: If both lab and first_df are provided.
        """
        print('Initializing lineage tree...')
        if lab.any() and first_df:
            raise ValueError('Only one of lab and first_df can be provided.')

        if lab.any():
            added_lineage_tree = []

            rp = regionprops(lab.copy())
            for obj in rp:
                label = obj.label
                self.families.append([(label, 1)])
                added_lineage_tree.append((-1, label, -1, 1, label, [-1] * (self.max_daughter-1))) # in effect I could actually just write this directly to a df, this is  a relic from the old code but works just fine

            cca_df = added_lineage_tree_to_cca_df(added_lineage_tree)
            self.lineage_list = [cca_df]

        elif first_df:
            first_df = checked_reset_index_Cell_ID(first_df)
            self.lineage_list = [first_df]
            self.families.append([(label, 1) for label in first_df.index])

    def create_tracked_frame(self, frame_i, mother_daughters, IDs_prev, IDs_curr_untracked, assignments, curr_IDs, new_IDs):
        """
        Add a new frame to the lineage tree, updating families and tracking new and divided cells.

        Args:
            frame_i (int): Index of the current frame.
            mother_daughters (list): List of (mother_index, daughter_indices) tuples.
            IDs_prev (list): List mapping previous cell indices to IDs.
            IDs_curr_untracked (list): List of current untracked cell IDs.
            assignments (dict): Mapping from untracked cell IDs to tracked cell IDs.
            curr_IDs (set): Set of current cell IDs.
            new_IDs (set): Set of new cell IDs not resulting from division.

        Returns:
            None
        """

        added_lineage_tree = []
        
        daughter_dict = {}
        daughter_set = set()
        for mother, daughters in mother_daughters:
            daughter_IDs = IoA_index_daughter_to_ID(daughters, assignments, IDs_curr_untracked)
            daughter_dict[mother] = daughter_IDs
            daughter_set.update(set(daughter_IDs))

        new_unknown_IDs = new_IDs - daughter_set
        # print(f'\n\nFrame {frame_i}:\nNew unknown IDs: {new_unknown_IDs}\nDaughter set: {daughter_set}\nnew_IDs: {new_IDs}')
        # print(f'Assignments: {assignments}')

        for ID in new_unknown_IDs:
            # print(f'Frame {frame_i}: New cell ID {ID} suspected of being a cell from the outside.')
            self.families.append([(ID, 1)])
            added_lineage_tree.append((frame_i, ID, -1, 1, ID, [-1] * (self.max_daughter-1)))

        for mother, _ in mother_daughters:
            mother_ID = IDs_prev[mother]
            daughter_IDs = daughter_dict[mother]
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
                printl(frame_i, mother_daughters, IDs_prev, IDs_curr_untracked, assignments, curr_IDs)
                printl(f"Warning: No family could be associated. Creating a new family for cells {daughter_IDs} with suspected mother ID {mother_ID}.")
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
        cca_df = filter_current_IDs(cca_df, curr_IDs)
        cca_df = checked_reset_index_Cell_ID(cca_df)
        try:
            self.lineage_list[frame_i] = cca_df
        except IndexError:
            len_lineage_list = len(self.lineage_list)
            if frame_i >= len_lineage_list:
                self.lineage_list.extend([pd.DataFrame()] * (frame_i + 1 - len_lineage_list))
            self.lineage_list[frame_i] = cca_df
        self.frames_for_dfs.add(frame_i)

    def real_time(self, frame_i, lab, prev_lab, rp=None, prev_rp=None):
        """
        Compute the lineage tree for a frame in real time, typically for GUI annotation.

        Args:
            frame_i (int): Index of the current frame.
            lab (ndarray): Labeled image of the current frame.
            prev_lab (ndarray): Labeled image of the previous frame.
            rp (list, optional): Region properties for the current frame.
            prev_rp (list, optional): Region properties for the previous frame.

        Returns:
            None
        """
        if rp is None:
            rp = regionprops(lab)

        if prev_rp is None:
            prev_rp = regionprops(prev_lab)

        IoA_matrix, self.IDs_curr_untracked, self.IDs_prev = calc_Io_matrix(lab, prev_lab, rp, prev_rp)
        aggr_track, self.mother_daughters = mother_daughter_assign(IoA_matrix, 
                                                                   IoA_thresh_daughter=self.IoA_thresh_daughter, 
                                                                   min_daughter=self.min_daughter, 
                                                                   max_daughter=self.max_daughter
                                                                   )

        curr_IDs = set(self.IDs_curr_untracked)
        prev_IDs = {obj.label for obj in prev_rp}
        new_IDs = curr_IDs - prev_IDs
        self.frames_for_dfs.add(frame_i)
        self.create_tracked_frame(frame_i, self.mother_daughters, self.IDs_prev, self.IDs_curr_untracked, None, curr_IDs, new_IDs)

    def update_df_li_locally(self, df, frame_i):
        """
        Update the lineage DataFrame for a specific frame, correcting lineage columns for consistency.

        Args:
            df (pd.DataFrame): DataFrame to update for the specified frame.
            frame_i (int): Frame index to update.

        Returns:
            None
        """
        # import pandasgui as pg
        # pg.show(df, self.lineage_list[frame_i], self.lineage_list[frame_i-1])

        # we first need to correct generation_num_tree, root_ID_tree, sister_ID_tree
        df = checked_reset_index(df)
        corrected_rows = []

        parent_lines = dict()

        for _, Cell_info in df.iterrows():
            if Cell_info['parent_ID_tree'] == -1:
                Cell_info['generation_num_tree'] = 1
                Cell_info['root_ID_tree'] = Cell_info['Cell_ID']
                Cell_info['sister_ID_tree'] = [-1]
                Cell_info['is_history_known'] = False
                corrected_rows.append(Cell_info)
                continue

            Cell_info['is_history_known'] = True

            if Cell_info['parent_ID_tree'] not in parent_lines:
                for i in range(frame_i-1, -1, -1):
                    if Cell_info['parent_ID_tree'] in self.lineage_list[i].index:
                        parent_line = self.lineage_list[i].loc[Cell_info['parent_ID_tree']]
                        parent_lines[Cell_info['parent_ID_tree']] = parent_line
                        break
                else:
                    raise ValueError(f"Parent ID {Cell_info['parent_ID_tree']} not found in lineage list at frame {i}.")
            else:
                parent_line = parent_lines[Cell_info['parent_ID_tree']]

            Cell_info['generation_num_tree'] = parent_line['generation_num_tree'] + 1
            Cell_info['root_ID_tree'] = parent_line['root_ID_tree']

            sisters = set(df.loc[df['parent_ID_tree'] == Cell_info['parent_ID_tree'], 'Cell_ID'])
            sisters.discard(Cell_info['Cell_ID'])
            Cell_info['sister_ID_tree'] = list(sisters) if sisters else [-1]

            corrected_rows.append(Cell_info)

        corrected_df = pd.DataFrame(corrected_rows)
        self.lineage_list[frame_i] = corrected_df.set_index('Cell_ID')

    def insert_lineage_df(self, lineage_df, frame_i, update_fams=True, consider_children=True, raw_input=False, propagate=True, relevant_cells=None):
        """
        Insert or replace a lineage DataFrame at a given frame index, optionally updating families and propagating changes.

        Args:
            lineage_df (pd.DataFrame): The lineage DataFrame to insert.
            frame_i (int): The index of the frame.
            update_fams (bool, optional): If True, update families based on the changes. Defaults to True.
            consider_children (bool, optional): If True, update children of the inserted frame. Defaults to True.

        Returns:
            None
        """
        if not raw_input:
            lineage_df = reorg_sister_cells_for_import(lineage_df)
            lineage_df = filter_cols(lineage_df)

        lineage_df = checked_reset_index_Cell_ID(lineage_df)
        len_lineage_list = len(self.lineage_list)
        if frame_i == len_lineage_list:
            self.lineage_list.append(lineage_df)
            self.frames_for_dfs.add(frame_i)

            self.update_df_li_locally(lineage_df, frame_i)

            if propagate:
                out = update_consistency(df_li=self.lineage_list, fixed_frame_i=frame_i,
                                        consider_children=consider_children, Cell_IDs_fixed=relevant_cells,
                                        families=self.families if update_fams else None)
                if update_fams:
                    self.lineage_list, self.families = out
                else:
                    self.lineage_list = out

        elif frame_i < len_lineage_list:
            self.lineage_list[frame_i] = lineage_df
            self.update_df_li_locally(lineage_df, frame_i)
            if propagate:
                out = update_consistency(df_li=self.lineage_list, fixed_frame_i=frame_i,
                                        consider_children=consider_children, Cell_IDs_fixed=relevant_cells,
                                        families=self.families if update_fams else None)
                if update_fams:
                    self.lineage_list, self.families = out
                else:
                    self.lineage_list = out


        elif frame_i > len_lineage_list:
            printl(f'WARNING: Frame_i {frame_i} was inserted. The lineage list was only {len(self.lineage_list)} frames long, so the last known lineage tree was copy pasted up to frame_i {frame_i}')

            original_length = len(self.lineage_list)
            self.lineage_list = self.lineage_list + [self.lineage_list[-1]] * (frame_i - len(self.lineage_list))

            self.generate_gen_df_from_df_li(self.lineage_list, force=True)

            self.lineage_list.append(lineage_df)

            frame_is = set(range(len(self.lineage_list)-original_length))
            self.frames_for_dfs = self.frames_for_dfs | frame_is

            self.update_df_li_locally(lineage_df, frame_i)
            if propagate:
                out = update_consistency(df_li=self.lineage_list, fixed_frame_i=frame_i,
                                        consider_children=consider_children, Cell_IDs_fixed=relevant_cells,
                                        families=self.families if update_fams else None)
                if update_fams:
                    self.lineage_list, self.families = out
                else:
                    self.lineage_list = out

    def propagate(self, frame_i, relevant_cells=None):
        """
        Propagate changes from a specific frame to the entire lineage tree and families.

        Args:
            frame_i (int): The index of the frame to propagate.
            Cell_IDs_fixed (list, optional): List of fixed cell IDs to propagate. If None, all are propagated.

        Returns:
            None
        """
        lineage_df = self.lineage_list[frame_i]
        self.update_df_li_locally(lineage_df, frame_i)
        self.lineage_list, self.families = update_consistency(df_li=self.lineage_list, fixed_frame_i=frame_i,
                                                consider_children=True, Cell_IDs_fixed=relevant_cells,
                                                families=self.families)

    def load_lineage_df_list(self, df_li):
        """
        Load a list of lineage DataFrames, reconstructing the lineage tree and families.

        Args:
            df_li (list): List of acdc_df DataFrames.

        Returns:
            None
        """
        df_li = copy.deepcopy(df_li)  # Ensure we don't modify the original list
        # Support for first_frame was removed since it is not necessary, just make the df_li correct...
        # Also the tree needs to be init before. Also if df_li does not contain any relevenat dfs, nothing happens
        print('Loading lineage data...')
        df_li_new = []
        families = []
        families_root_IDs = []
        added_IDs = set()

        for i, df in enumerate(df_li):
            if df is None:
                continue

            if ('generation_num_tree' in df.columns 
                and not (df['generation_num_tree'] == 0).any()
                and not df['generation_num_tree'].isnull().any() 
                and not df["generation_num_tree"].isna().any() 
                and not df["generation_num_tree"].empty):

                df = checked_reset_index_Cell_ID(df)

                df = filter_cols(df)
                df = reorg_sister_cells_for_import(df)
                self.frames_for_dfs.add(i)
                df_li_new.append(df)

                df_filter = df.index.isin(added_IDs)  
                for root_ID, group in df[df_filter].groupby('root_ID_tree'):
                    if root_ID not in families_root_IDs:
                        family = list(zip(group.index, group['generation_num_tree']))
                        families.append(family)
                        families_root_IDs.append(root_ID)
                    else:
                        # If the root_ID is already in families, we just update the family with the new cells
                        family_index = families_root_IDs.index(root_ID)
                        families[family_index].extend(zip(group.index, group['generation_num_tree']))
                        
                    added_IDs.update(group.index)
                    
        if df_li_new:
            self.lineage_list = df_li_new

    def export_df(self, frame_i):
        """
        Export the lineage DataFrame for a specific frame, cleaning up auxiliary columns.

        Args:
            frame_i (int): The index of the frame.

        Returns:
            pd.DataFrame: The cleaned DataFrame for the specified frame.
        """
        df = self.lineage_list[frame_i].copy()

        if df.empty:
            print(f'Warning: No dataframe for frame {frame_i} found.')

        df = reorg_sister_cells_for_export(df)

        df = checked_reset_index_Cell_ID(df)

        columns = df.columns
        if "level_0" in columns:
            df = df.drop(columns="level_0")
        if "index" in columns:
            df = df.drop(columns="index")
        if "frame_i" in columns:
            df = df.drop(columns="frame_i")

        return df
    
    def export_lin_tree_info(self, frame_i):
        """
        Return information about new, orphan, and lost cells between two consecutive frames.

        Args:
            frame_i (int): The index of the frame.

        Returns:
            tuple: (cells_with_parent, orphan_cells, lost_cells)
                - cells_with_parent (list of tuples): (cell_id, mother_id) for new cells with a parent.
                - orphan_cells (list): List of new cell IDs without a parent.
                - lost_cells (list): List of cell IDs lost in this frame.
        """
        if frame_i == 0:
            return [], [], []
        
        df_curr = self.lineage_list[frame_i]
        df_curr = checked_reset_index_Cell_ID(df_curr)
        df_prev = self.lineage_list[frame_i-1]
        df_prev = checked_reset_index_Cell_ID(df_prev)

        new_cells = set(df_curr.index) - set(df_prev.index)
        lost_cells = set(df_prev.index) - set(df_curr.index)

        cells_with_parent = []
        orphan_cells = []
        mother_cells = set()

        for cell in new_cells:
            cell_row = df_curr.loc[cell]
            mother = cell_row['parent_ID_tree']
            if mother == -1:
                orphan_cells.append(cell)
            else:
                cells_with_parent.append((cell, mother))
                mother_cells.add(mother)

        lost_cells = lost_cells - mother_cells

        lost_cells = [int(cell) for cell in lost_cells]
        cells_with_parent.sort(key=lambda x: x[1])  # Sort by mother ID
        cells_with_parent = [(int(cell), int(mother)) for cell, mother in cells_with_parent]
        orphan_cells = [int(cell) for cell in orphan_cells]

        return cells_with_parent, orphan_cells, lost_cells
        

class tracker:
    """
    Class representing a tracker for cell division in a video sequence. (Adapted from trackers.CellACDC.CellACDC_tracker.py)

    Attributes:
    - cca_dfs_auto (list): List of lineage dataframes for export.

    Methods:
    - __init__(): Initializes the tracker object.
    - track(): Tracks cell division in the video sequence. (Used for module 2)
    - track_frame(): Tracks cell division in a single frame. (Used for GUI tracking)
    - updateGuiProgressBar(): Updates the GUI progress bar. (Used for GUI communication)
    - save_output(): Signals to the rest of the programme that the lineage tree should be saved. (Used for module 2)
    """
    def __init__(self):
        """
        Initializes the CellACDC_normal_division_tracker object.
        """
        pass

    def track(self,
              segm_video,
              IoA_thresh:float = 0.8,
              IoA_thresh_daughter:float = 0.25,
              IoA_thresh_aggressive:float = 0.5,
              min_daughter:int = 2,
              max_daughter:int = 2,
              record_lineage:bool = True,
              return_tracked_lost_centroids:bool = True,
              signals = None,
        ):
        """
        Tracks the segmented video frames and returns the tracked video. (Used for module 2)

        Parameters:
        - segm_video (list): List of segmented video frames.
        - signals (list, optional): List of signals. Used for GUI communication. Defaults to None.
        - IoA_thresh (float, optional): IoA threshold. Used for tracking cells before even looking if a cell has divided. Defaults to 0.8.
        - IoA_thresh_daughter (float, optional): IoA threshold for daughter cells. Used for identifying daughter cells. Defaults to 0.25.
        - IoA_thresh_aggressive (float, optional): Aggressive IoA threshold. Used when the tracker thinks that a cell has NOT divided. Defaults to 0.5.
        - min_daughter (int, optional): Minimum number of daughter cells. Used for determining if a cell has devided. Defaults to 2.
        - max_daughter (int, optional): Maximum number of daughter cells. Used for determining if a cell has devided. Defaults to 2.
        - record_lineage (bool, optional): Flag to record and save lineage. Defaults to True.
        
        Returns:
        - list: Tracked video frames.
        """
        if not record_lineage and return_tracked_lost_centroids:
            print('return_tracked_lost_centroids is set to True if record_lineage is True.')
            record_lineage = True
        
        pbar = tqdm(total=len(segm_video), desc='Tracking', ncols=100)

        if return_tracked_lost_centroids:
            self.tracked_lost_centroids = {
                frame: [] for frame in range(len(segm_video))
            }

        for frame_i, lab in enumerate(segm_video):
            if frame_i == 0:
                tracker = normal_division_tracker(
                    segm_video, IoA_thresh_daughter, min_daughter, 
                    max_daughter, IoA_thresh, IoA_thresh_aggressive
                )
                if record_lineage or return_tracked_lost_centroids:
                    tree = normal_division_lineage_tree(
                        lab=lab, max_daughter=max_daughter,
                        min_daughter=min_daughter, 
                        IoA_thresh_daughter=IoA_thresh_daughter
                    )
                pbar.update()
                rp = regionprops(segm_video[0])
                prev_IDs = {obj.label for obj in rp}
                prev_rp = rp
                continue

            tracker.track_frame(frame_i)

            if not record_lineage and not return_tracked_lost_centroids:
                continue

            mother_daughters = tracker.mother_daughters
            IDs_prev = tracker.IDs_prev
            assignments = tracker.assignments
            IDs_curr_untracked = tracker.IDs_curr_untracked
            rp = regionprops(tracker.tracked_lab)
            curr_IDs = {obj.label for obj in rp}
            new_IDs = curr_IDs - prev_IDs
            # print(f'Frame {frame_i}: {new_IDs}, {curr_IDs}, {prev_IDs}')
            tree.create_tracked_frame(
                frame_i, mother_daughters, IDs_prev, IDs_curr_untracked,
                assignments, curr_IDs, new_IDs
            )
            # printl(new_IDs, curr_IDs, prev_IDs)
            tracked_lost_centroids_loc = []
            for mother, _ in mother_daughters:
                mother_ID = IDs_prev[mother]
                
                found = False
                for obj in prev_rp:
                    if obj.label == mother_ID:
                        tracked_lost_centroids_loc.append(obj.centroid)
                        found = True
                        break
                if not found:
                    labels = [obj.label for obj in rp]
                    printl(mother, mother_ID, IDs_curr_untracked, labels)
                    raise ValueError('Something went wrong with the tracked lost centroids.')


            if len(mother_daughters) != len(tracked_lost_centroids_loc):
                raise ValueError('Something went wrong with the tracked lost centroids.')
            
            self.tracked_lost_centroids[frame_i] = tracked_lost_centroids_loc

            prev_IDs = curr_IDs.copy()
            prev_rp = rp.copy()

            self.updateGuiProgressBar(signals)
            pbar.update()

        if record_lineage:
            cca_li = []
            for i in range(len(tree.lineage_list)):
                cca_li.append(tree.export_df(i))

            self.cca_dfs_auto = cca_li
            # here we would also save make sure to save self.tracked_lost_centroids, but since we already assigned it correctly from the get go we dont need to do that


        tracked_video = tracker.tracked_video
        pbar.close()
        return tracked_video

    def track_frame(self,
                    previous_frame_labels,
                    current_frame_labels,
                    IDs : NotGUIParam =None,
                    IoA_thresh: float = 0.8,
                    IoA_thresh_daughter:float = 0.25,
                    IoA_thresh_aggressive:float  = 0.5,
                    min_daughter:int = 2,
                    max_daughter:int = 2,
                    ):
        """
        Tracks cell division in a single frame. (This is used for real time tracking in the GUI)

        Parameters:
        - previous_frame_labels: Labels of the previous frame.
        - current_frame_labels: Labels of the current frame.
        - IoA_thresh (float, optional): IoA threshold. Used for tracking cells before even looking if a cell has divided. Defaults to 0.8.
        - IoA_thresh_daughter (float, optional): IoA threshold for daughter cells. Used for identifying daughter cells. Defaults to 0.25.
        - IoA_thresh_aggressive (float, optional): Aggressive IoA threshold. Used when the tracker thinks that a cell has NOT divided. Defaults to 0.5.
        - min_daughter (int, optional): Minimum number of daughter cells. Used for determining if a cell has devided. Defaults to 2.
        - max_daughter (int, optional): Maximum number of daughter cells. Used for determining if a cell has devided. Defaults to 2.

        Returns:
        - tracked_video: Tracked video sequence.
        - mothers: Set of IDs of mother cells in the current frame. (Used in GUI so it doesn't complain if IDs are missing)
        """

        if not np.any(current_frame_labels):
            # Skip empty frames
            return current_frame_labels

        segm_video = [previous_frame_labels, current_frame_labels]
        tracker = normal_division_tracker(segm_video, IoA_thresh_daughter, min_daughter, max_daughter, IoA_thresh, IoA_thresh_aggressive)
        tracker.track_frame(1, IDs=IDs)
        tracked_video = tracker.tracked_video

        mother_daughters_pairs = tracker.mother_daughters
        IDs_prev = tracker.IDs_prev
        mothers = {IDs_prev[pair[0]] for pair in mother_daughters_pairs}

        return tracked_video[-1], mothers

    def updateGuiProgressBar(self, signals):
        """
        Updates the GUI progress bar.

        Parameters:
        - signals: Signals object for GUI communication.

        Returns:
        - None
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

        Parameters:
        - None

        Returns:
        - None
        """
        pass