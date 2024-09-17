import os
from cellacdc.trackers.CellACDC.CellACDC_tracker import calc_IoA_matrix
from cellacdc.trackers.CellACDC.CellACDC_tracker import track_frame as track_frame_base
from cellacdc.core import getBaseCca_df, printl
from cellacdc.myutils import checked_reset_index
import numpy as np
from skimage.measure import regionprops
from tqdm import tqdm
import pandas as pd
from cellacdc.myutils import exec_time

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
    lin_tree_cols = {'generation_num_tree', 'root_ID_tree', 'sister_ID_tree', 'parent_ID_tree', 'parent_ID_tree', 'emerg_frame_i', 'division_frame_i'}
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

    values = {i for i in row if i not in {0, -1} and not np.isnan(i)} or {-1}
    values = list(values) if values else [-1]
    return values


def reorg_sister_cells_for_inport(df):
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
    """

    cca_df = pd.DataFrame()
    cca_df['Cell_ID'] = [row[1] for row in added_lineage_tree]
    cca_df['emerg_frame_i'] = [row[0] for row in added_lineage_tree]
    cca_df['division_frame_i'] = [row[0] for row in added_lineage_tree]
    cca_df['generation_num_tree'] = [row[3] for row in added_lineage_tree]
    cca_df['parent_ID_tree'] = [row[2] for row in added_lineage_tree]
    cca_df['root_ID_tree'] = [row[4] for row in added_lineage_tree]
    cca_df['sister_ID_tree'] = [row[5] for row in added_lineage_tree]
    cca_df = cca_df.set_index('Cell_ID')
    return cca_df

def create_lineage_tree_video(segm_video, IoA_thresh_daughter, min_daughter, max_daughter):
    """
    Creates a lineage tree for a video of segmented frames (Currently not in use).

    Parameters:
    - segm_video (list): A list of segmented frames.
    - IoA_thresh_daughter (float): IoA threshold for identifying daughter cells.
    - min_daughter (int): Minimum number of daughter cells.
    - max_daughter (int): Maximum number of daughter cells.

    Returns:
    - list: A list representing the lineage tree.

    """
    raise NotImplementedError('This function is not implemented yet.')
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

    Parameters:
    - df (pandas.DataFrame): The DataFrame containing the IDs.
    - indices_to_remove (list): A list of indices NOT to be removed.

    Returns:
    - pandas.DataFrame: The DataFrame with only the current IDs.
    """
    df = checked_reset_index(df)
    df = df.set_index('Cell_ID')
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

def add_member_fam_from_df(families, df):
    """
    NOT IN USE!

    Updates the families list based on the input DataFrame. If new families are found, they are added to the list. This is probably an overly complicated way of doing this, but it works...

    Parameters:
    - families (list): List of families in the lineage tree.
    - df (pandas.DataFrame): The DataFrame containing the lineage tree.

    Returns:
    - list: The updated list of families. For details see attribute disc. of normal_division_lineage_tree.
    """
    raise NotImplementedError('This function is not implemented yet.')

    additions = []

    for index, row in df.iterrows():
        for family in families:
            in_family = False

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
    NOT IN USE!

    Corrects families list based on the input DataFrame. If the cell is in a family it should not belong (based on Root ID) it is delted from that family.
    If this is slow it can be optimized by putting it together with update_fam_from_df

    Parameters:
    - families (list): List of families in the lineage tree.
    - df (pandas.DataFrame): The DataFrame containing the lineage tree.

    Returns:
    - list: The updated list of families. For details see attribute disc. of normal_division_lineage_tree.
    """

    raise NotImplementedError('This function is not implemented yet.')

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
    """
    NOT IN USE!

    Updates the generation number of each member in the given families based on the corresponding values in the DataFrame.

    Parameters:
    - families (list): A list of families, where each family is a list of members.
    - df (pandas.DataFrame): The DataFrame containing the generation number information.

    Returns:
    - updated_families (list): The updated list of families with the generation numbers updated.
    """

    raise NotImplementedError('This function is not implemented yet.')
    for fam in families:
        for member in fam:
            member[1] = df.loc[member[0], 'generation_num_tree']

    return families

def update_consistency(fixed_frame_i=None, general_df=None, fixed_df=None, Cell_IDs_fixed=None, consider_children=True, fwd=True, bck=True, columns_to_replace=None, count=0):
    """
    - Update the consistency. Cell_IDs_fixed are the Cell_IDs which should be updated, if None all Cell_IDs are updated based on the fixed_df or fixed_frame_i in combination with general_df.
    - There are several ways to call this function:
    1. If fixed_frame_i is provided, general_df is provided but fixed_df is not provided: the fixed_df is taken from the general_df.
    2. Fixed_df is provided, general_df is provided but fixed_frame_i is not provided: the fixed_frame_i is taken from the fixed_df.

    Parameters:
    - fixed_frame_i (int): The fixed frame index.
    - fixed_df (pd.DataFrame): The fixed DataFrame.
    - Cell_IDs_fixed (list): The list of Cell IDs to consider.
    - consider_children (bool): Flag to indicate whether to consider children cells. Should be True!
    - fwd (bool): Flag to indicate forward propagation. Should be True!
    - bck (bool): Flag to indicate backward propagation. Should be True!
    - general_df (pd.DataFrame): The general DataFrame.
    - columns_to_replace (list): The list of columns to replace. (Internal, obsolete technically, but still used in the function.)
    - count (int): The count of the update consistency function. (Internal, not used for now.)

    Returns:
    - pd.DataFrame: The updated DataFrame with consistent lineage tree.
    """
    count += 1
    if consider_children and not fwd:
        raise ValueError('consider_children can\'t be true while fwd is not.')

    if not columns_to_replace:
        columns_to_replace = ['generation_num_tree', 'root_ID_tree', 'sister_ID_tree', 'parent_ID_tree']

    if not fwd or not bck:
        raise NotImplementedError('Not tested yet at all!')

    general_df = checked_reset_index(general_df)
    general_df = general_df.set_index(['frame_i', 'Cell_ID'])

    if Cell_IDs_fixed is not None and fixed_df is not None:
        general_df = general_df.reset_index()
        fixed_df = general_df.drop_duplicates(subset='Cell_ID')
        general_df = general_df.set_index(['frame_i', 'Cell_ID'])
        fixed_df = checked_reset_index(fixed_df)
        fixed_df = fixed_df[fixed_df['Cell_ID'].isin(Cell_IDs_fixed)]
        fixed_df = fixed_df[fixed_df['frame_i'] == fixed_frame_i]
        fixed_df = fixed_df.set_index('Cell_ID')

    elif fixed_df is None and general_df is not None and fixed_frame_i: # if we don't have a given fixed df we take the one from the general df

        fixed_df = (general_df
                    .loc[fixed_frame_i]
                    .reset_index() # this is fine
                    .set_index('Cell_ID'))

    elif fixed_df is not None: # if we have a fixed df we are all good
        fixed_df  = checked_reset_index(fixed_df)
        fixed_df = fixed_df.set_index('Cell_ID')

    else: # if we have neither we have a problem
        raise ValueError('Either fixed_frame_df or fixed_df must be provided.')

    if Cell_IDs_fixed is None: # if we don't have a list of Cell_IDs_fixed_df we take all Cell_IDs_fixed_df from the fixed_df (default)
        Cell_IDs_fixed = fixed_df.index

    general_df = checked_reset_index(general_df)

    if not fixed_frame_i: # this is for propagation mainly
        if fwd and bck: # this splits the df into two parts, only one is edited. Since there is no fixed frame we split so it also edits frame_i
            general_df_keep = pd.DataFrame()
            general_df_change = general_df
        else:
            raise ValueError('fwd or bck must be True if fixed_frame_i is provided.')
    else:
        if fwd and bck: # this splits the df into two parts, only one is edited. Since there is a fixed frame we split so it doesn't edit frame_i
            general_df_keep = pd.DataFrame()
            general_df_change = general_df
        elif fwd:
            general_df_keep = general_df[general_df['frame_i'] <= fixed_frame_i]
            general_df_change = general_df[general_df['frame_i'] > fixed_frame_i]
        elif bck:
            general_df_keep = general_df[general_df['frame_i'] >= fixed_frame_i]
            general_df_change = general_df[general_df['frame_i'] < fixed_frame_i]
        else:
            raise ValueError('one or both of fwd or bck must be True if fixed_frame_i is provided.')


    general_df_change = checked_reset_index(general_df_change)
    general_df_change = general_df_change.set_index('Cell_ID')
    occ_cells = general_df_change.index.value_counts() # for repeating the lines enough times
    
    for Cell_ID in Cell_IDs_fixed: # replace values for the cells in the general df # def needs to be optimized
        occ_cell = occ_cells[Cell_ID]
        Cell_df = pd.concat([fixed_df.loc[Cell_ID, columns_to_replace]]*occ_cell, axis=1).transpose()
        Cell_df.index.name = 'Cell_ID'
        general_df_change.loc[[Cell_ID], columns_to_replace] = Cell_df # we replace necessary columns

    general_df_keep = checked_reset_index(general_df_keep)
    general_df_change = checked_reset_index(general_df_change)

    general_df = pd.concat([general_df_keep, general_df_change]) # we put the df back together

    if consider_children: # this also enforces that fwd is true (See ValueError above)
        unique_df = checked_reset_index(general_df)
        unique_df = (unique_df
                    .drop_duplicates(subset='Cell_ID')
                    .set_index('Cell_ID')
                    )
                      # we drop all duplicates (different frames) to reduce overhead

        children_df = unique_df[unique_df['parent_ID_tree'].isin(Cell_IDs_fixed)] # we select all children of the cells we are considering
        children_df = checked_reset_index(children_df)
        if children_df.empty: # if there are no children we are done
            return general_df
        else: # else we construct a new fixed_df and run the update recursively.
            new_children_df = []
            for _, row in children_df.iterrows(): # we need to edit the daughters to be consistent
                parent_cell = unique_df.loc[row['parent_ID_tree']]
                row['generation_num_tree'] = parent_cell['generation_num_tree'] + 1
                row['root_ID_tree'] = parent_cell['root_ID_tree']
                sister_cell = unique_df[unique_df['parent_ID_tree'] == row['parent_ID_tree']].index # index here should be the Cell_ID
                try:
                    sister_cell = list(sister_cell)
                except TypeError:
                    sister_cell = [sister_cell]

                sister_cell.remove(row['Cell_ID'])
                row['sister_ID_tree'] = sister_cell
                new_children_df.append(row)

            new_children_df = pd.DataFrame(new_children_df)

            general_df = update_consistency(columns_to_replace=columns_to_replace,
                                            general_df=general_df,
                                            fwd=fwd, bck=bck,
                                            fixed_df=new_children_df,
                                            consider_children=consider_children,
                                            count=count)
            return general_df
    else:
        return general_df

def gen_df_to_fams(general_df):
    """
    Convert a general DataFrame to a list of families. Families is a list of lists of tuples. First level of list is the family, second one consists of the tuples for each memeber of the family. Each tuple contains the Cell_ID and generation_num_tree of a cell in the family. These families are mainly used internally.

    Parameters:
    - general_df (pandas.DataFrame): The general DataFrame containing cell information.

    Returns:
    - list: A list of families, where each family is represented as a list of tuples. Each tuple contains the Cell_ID and generation_num_tree of a cell in the family.
    """
    families = []
    general_df = checked_reset_index(general_df)
    general_df = general_df.set_index('Cell_ID')
    general_df = general_df[~general_df.index.duplicated(keep='first')]
    fam_groups = general_df.groupby('root_ID_tree')
    for root_id, fam in fam_groups:
        fam = fam.sort_values('frame_i')
        families.append(list(zip(fam.index, fam['generation_num_tree'])))

    return families

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

    def track_frame(self, frame_i, lab=None, prev_lab=None, rp=None, prev_rp=None):
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

        IoA_matrix, self.IDs_curr_untracked, self.IDs_prev = calc_IoA_matrix(lab,
                                                                             prev_lab,
                                                                             self.rp,
                                                                             prev_rp
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
    Represents a lineage tree for tracking normal cell divisions.

    Attributes:
    - max_daughter (int): The maximum number of daughter cells expected.
    - min_daughter (int): The minimum number of daughter cells expected.
    - IoA_thresh_daughter (float): The threshold for intersection over area (IoA) for daughter cells.
    - mother_daughters (list): List of tuples representing mother-daughter cell relationships.
    - frames_for_dfs (set): Set of frame indices for the lineage tree.
    - general_df (pandas.DataFrame): Dataframe which has all df from all frames concatinated together, with an additional column 'frame_i' to keep track of the frame.
    - need_update_gen_df (bool): Flag indicating whether the general_df needs to be updated (Internal).
    - families (list): List of families in the lineage tree.
    - lineage_list (list): List of dataframes representing the lineage tree.

    Methods:
    - __init__(self, lab=None, max_daughter=2, min_daughter=2, IoA_thresh_daughter=0.25, first_df=None, frame_i=0): Initializes the CellACDC_normal_division_tracker object.
    - init_lineage_tree(self, lab=None, first_df=None): Initializes the lineage tree.
    - create_tracked_frame_tree(self, frame_i, mother_daughters, IDs_prev, IDs_curr_untracked, assignments, curr_IDs): Creates a tracked frame tree based on the given parameters.
    - real_time_tree(self, frame_i, lab, prev_lab, rp=None, prev_rp=None): Calculates the real-time tree of cell divisions based on the input labels and region properties.
    - dict_curr_frame(self): Creates a dictionary mapping mother IDs to daughter IDs for the current frame.
    - generate_gen_df_from_df_li(self, df_li, force=False): Generates the general_df from a list of dataframes.
    - update_gen_df_from_df(self, df, frame_i): Updates the general_df with new rows from a given dataframe for a specific frame.
    """

    def __init__(self, lab=None, first_df=None, frame_i=0, max_daughter=2, min_daughter=2, IoA_thresh_daughter=0.25):
        """
        Initializes the CellACDC_normal_division_tracker object. Mainly sets the parameters and initializes the lineage tree if a labeled image or a dataframe is provided.

        Parameters:
        - lab (ndarray, optional): The labeled image of cells.
        - max_daughter (int, optional): The maximum number of daughter cells expected.
        - min_daughter (int, optional): The minimum number of daughter cells expected.
        - IoA_thresh_daughter (float, optional): The threshold for intersection over area (IoA) for daughter cells.
        - first_df (pd.DataFrame, optional): The first dataframe representing the lineage tree.
        - frame_i (int, optional): Index of the current frame. Defaults to 0. (Try not to change this, its not properly tested.)

        Returns:
        None
        """
        if lab is None and first_df is None:
            raise ValueError('Either lab or first_df must be provided')

        self.max_daughter = max_daughter
        self.min_daughter = min_daughter
        self.IoA_thresh_daughter = IoA_thresh_daughter
        self.mother_daughters = [] # just for the dict_curr_frame stuff...
        self.frames_for_dfs = set([frame_i])
        self.general_df = {}
        self.need_update_gen_df = False # this is only when using the quick option in update_gen_df_from_df

        self.families = []

        self.init_lineage_tree(lab, first_df)

    def init_lineage_tree(self, lab=None, first_df=None):
        """
        Initializes the lineage tree (self.lineage_list) based on the provided lab or first_df. If none are provided, an empty lineage_list is created.

        Args:
        - lab (ndarray, optional): The labeled image representing the cells. Defaults to None.
        - first_df (DataFrame, optional): The initial dataframe representing the cells. Defaults to None.

        Returns:
        - None
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
            self.lineage_list = [first_df]
            self.families.append([(label, 1) for label in first_df.index])

    def create_tracked_frame(self, frame_i, mother_daughters, IDs_prev, IDs_curr_untracked, assignments, curr_IDs, new_IDs):
        """
        Creates a tracked frame tree based on the given parameters. The tracked frame tree is added to self.lineage_list.

        Parameters:
        - frame_i (int): Index of the current frame.
        - mother_daughters (list): List of tuples representing mother-daughter cell relationships.
        - IDs_prev (list): List mapping previous cell IDs to current cell IDs.
        - IDs_curr_untracked (list): List of current cell IDs that are untracked (so that have the ID of the original lab)
        - assignments (dict): Dictionary mapping untracked cell IDs to tracked cell IDs.
        - curr_IDs (list): List of current cell IDs.
        - new_IDs (set): List of new cell IDs which did not come from cell division.

        Returns:
        - None
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
            added_lineage_tree.append((-1, ID, -1, 1, ID, [-1] * (self.max_daughter-1)))

            
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
        self.lineage_list.append(cca_df)

    def real_time(self, frame_i, lab, prev_lab, rp=None, prev_rp=None):
        """
        Calculates the real-time tree of cell divisions based on the input labels and region properties for frame_i. The real-time tree is added to self.lineage_list. This is used for real-time lineage annotation in the GUI.

        Parameters:
        - frame_i (int): The index of the current frame.
        - lab (ndarray): The labeled image of the current frame.
        - prev_lab (ndarray): The labeled image of the previous frame.
        - rp (list, optional): The region properties of the current frame. If not provided, it will be calculated. Defaults to None.
        - prev_rp (list, optional): The region properties of the previous frame. If not provided, it will be calculated. Defaults to None.

        Returns:
        - None
        """
        if not np.any(rp):
            rp = regionprops(lab)

        if not np.any(prev_rp):
            prev_rp = regionprops(prev_lab)

        IoA_matrix, self.IDs_curr_untracked, self.IDs_prev = calc_IoA_matrix(lab, prev_lab, rp, prev_rp)
        aggr_track, self.mother_daughters = mother_daughter_assign(IoA_matrix, 
                                                                   IoA_thresh_daughter=self.IoA_thresh_daughter, 
                                                                   min_daughter=self.min_daughter, 
                                                                   max_daughter=self.max_daughter
                                                                   )

        curr_IDs = set(self.IDs_curr_untracked)
        prev_IDs = {obj.label for obj in prev_rp}
        new_IDs = curr_IDs - prev_IDs
        self.create_tracked_frame(frame_i, self.mother_daughters, self.IDs_prev, self.IDs_curr_untracked, None, curr_IDs, new_IDs)

    def dict_curr_frame(self):
        """
        Creates a dictionary mapping mother IDs to daughter IDs for the current frame.

        Parameters:
        - None

        Returns:
        - dict: A dictionary where the keys are mother IDs and the values are daughter IDs.
        """
        if not self.mother_daughters:
            return {}

        mother_IDs = [self.IDs_prev[mother] for mother, _ in self.mother_daughters]
        daughters = [daughter for _, daughter in self.mother_daughters]
        daughters = [daughter for sublist in daughters for daughter in sublist]
        daughter_IDs = IoA_index_daughter_to_ID(daughters, None, self.IDs_curr_untracked)
        return dict(zip(mother_IDs, daughter_IDs))

    def generate_gen_df_from_df_li(self, df_li, force=False, return_df=False):
        """
        Generates a general DataFrame (self.general_df) from a list of DataFrames. If force is False, self.need_update_gen_df is checked to see if the general DataFrame needs to be updated. If not, the function returns without updating the general DataFrame.

        Parameters:
        - df_li (list): A list of DataFrames.
        - force (bool, optional): If True, forces the generation of the general DataFrame even if it's not needed. Defaults to False.
        - return (bool, optional): If True, returns the general DataFrame. Defaults to False. 

        Returns:
        - None
        """
        if not force and not self.need_update_gen_df:
            return

        self.need_update_gen_df = False
        df_li = df_li.copy()
        for i, df in enumerate(df_li):
            df = checked_reset_index(df)
            if 'frame_i' not in df.columns:
                df['frame_i'] = i

            df = df.set_index(['frame_i', 'Cell_ID'])
            df_li[i] = df

        general_df = pd.concat(df_li)
        general_df = general_df.sort_values(by=['frame_i', 'Cell_ID'])
        self.general_df = general_df

        if return_df:
            return general_df
        else:
            return

    def update_gen_df_from_df(self, df, frame_i):
        """
        Updates self.general_df based on df. The resulting df for the entire family and all frames will not be consistent, but the single frame inserted should be consistent after the function ran.
        Please note that the following things are used to update the dict: parent_ID_tree, Cell_ID (Derived from this the following is corrected: generation_num_tree, root_ID_tree, sister_ID_tree).
        Please note that self.families is NOT updated!!! Run self.gen_df_to_fams() to update the families lists!!!
        All other columns are copy pasted from input df (there should be no other ones though!)

        Parameters:
        - df (pd.DataFrame): The dataframe on which to update the specified frame. parent_ID_tree, Cell_ID need to be correct, the rest is changed so it fits! 
        - frame_i (int): The frame index of df.

        Returns:
        - None
        """

        self.generate_gen_df_from_df_li(self.lineage_list)

        df = df.copy()
        df['frame_i'] = frame_i
        df = checked_reset_index(df)

        # we first need to correct generation_num_tree, root_ID_tree, sister_ID_tree
        self.general_df = checked_reset_index(self.general_df)
        unique_Cell_ID_df = (self.general_df
                            .drop_duplicates(subset='Cell_ID')
                            .set_index('Cell_ID')
                            )
        corrected_df = pd.DataFrame()

        for _, Cell_info in df.iterrows(): # similar code is used in update_dict_consistency in the part which handles daughters
            if Cell_info['parent_ID_tree'] == -1: # handle case where a cell was changed to unknown parent
                Cell_info['generation_num_tree'] = 1
                Cell_info['root_ID_tree'] = Cell_info['Cell_ID']
                Cell_info['sister_ID_tree'] = [-1]
                corrected_df = pd.concat([corrected_df, Cell_info.to_frame().T])
                continue

            parent_cell = unique_Cell_ID_df.loc[Cell_info['parent_ID_tree']]
            Cell_info['generation_num_tree'] = parent_cell['generation_num_tree'] + 1
            Cell_info['root_ID_tree'] = parent_cell['root_ID_tree']
            sister_cell_2 = set(df.loc[df['parent_ID_tree'] == Cell_info['parent_ID_tree'], 'Cell_ID'])
            sister_cell_2.remove(Cell_info['Cell_ID'])
            if sister_cell_2 == set():
                sister_cell_2 = [-1]
            try:
                Cell_info['sister_ID_tree'] = list(sister_cell_2)
            except TypeError:
                Cell_info['sister_ID_tree'] = [sister_cell_2]

            corrected_df = pd.concat([corrected_df, Cell_info.to_frame().T])

        self.general_df = self.general_df[self.general_df['frame_i'] != frame_i] # select all frames except the current one
        self.general_df = pd.concat([self.general_df, corrected_df])
        self.general_df = self.general_df.sort_values(by=['frame_i', 'Cell_ID'])

    def gen_df_to_df_li(self):
        """
        Updates the lineage list (self.lineage_list) based on the general DataFrame (self.general_df). The resulting dataframes are sorted based on frame_i and Cell_ID. Some columns are dropped, like 'index' and 'level_0', which should not be in the columns anyways.

        Parameters:
        - None

        Returns:
        - None
        """
        df_list = []
        self.generate_gen_df_from_df_li(self.lineage_list)
        self.general_df = checked_reset_index(self.general_df)
        self.general_df = self.general_df.sort_values(by=['frame_i', 'Cell_ID'])

        if 'index' in self.general_df.columns:
            self.general_df = self.general_df.drop(columns='index')

        if 'level_0' in self.general_df.columns:
            self.general_df = self.general_df.drop(columns='level_0')

        frames = self.general_df['frame_i'].unique().astype(int)

        frames = frames[~np.isnan(frames)]
        frames_for_dfs = {int(frame) for frame in frames}

        df_group = self.general_df.groupby('frame_i')
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

        self.lineage_list, self.frames_for_dfs = df_list, frames_for_dfs

    def insert_lineage_df(self, lineage_df, frame_i, propagate_back=False, propagate_fwd=False, update_fams=True, consider_children=True, quick=False):
        """
        Inserts a lineage DataFrame to the lineage list (self.lineage_list) at the given position. 
        If the position is greater than the length of the lineage list, a warning is printed and the last lineage df is copied until the inserted one.
        If the position is less than the length of the lineage list, the lineage DataFrame at the given position is replaced. 
        For now, please do not use this to propegate changes, and instead use self.propegate(). 
        PLEASE USE "quick=True" FOR NOW.
        Assumes that the IDs did not change.

        Parameters:
        - lineage_df (pandas.DataFrame): The lineage DataFrame to insert.
        - frame_i (int): The index of the frame.
        - propagate_back (bool, optional): If True, the changes are propagated backward in the lineage list. Defaults to False.
        - propagate_fwd (bool, optional): If True, the changes are propagated forward in the lineage list. Defaults to False.
        - update_fams (bool, optional): If True, the families are updated based on the changes. Defaults to True.
        - consider_children (bool, optional): If True, the changes are considered for the children of the inserted frame. Defaults to True.
        - quick (bool, optional): If True, the function will not update the general_df, families, or lineage_list. Defaults to False.

        Returns:
        - None
        """

        if quick == False:
            raise ValueError('Quick is not supported anymore (for now).')

        if quick and not propagate_back and not propagate_fwd and not update_fams and not consider_children:
            self.need_update_gen_df = True
        elif quick and (propagate_back or propagate_fwd or update_fams or consider_children):
            raise ValueError('Quick is True, other options are not supported.')

        lineage_df = reorg_sister_cells_for_inport(lineage_df)
        lineage_df = filter_cols(lineage_df)
        if frame_i == len(self.lineage_list):
            if not quick:
                self.generate_gen_df_from_df_li(self.lineage_list, force=True)

            self.lineage_list.append(lineage_df)
            self.frames_for_dfs.add(frame_i)

            if not quick:
                self.update_gen_df_from_df(lineage_df, frame_i)

            if propagate_back == True:
                self.general_df = update_consistency(general_df=self.general_df, fixed_frame_i=frame_i, fixed_df=lineage_df, consider_children=consider_children, fwd=False, bck=True)
            if update_fams == True:
                self.families = gen_df_to_fams(self.general_df)

            if not quick:
                self.gen_df_to_df_li()


        elif frame_i < len(self.lineage_list):

            if not quick:
                self.generate_gen_df_from_df_li(self.lineage_list, force=True)

            self.lineage_list[frame_i] = lineage_df

            if not quick:
                self.update_gen_df_from_df(lineage_df, frame_i)

            if propagate_back == True or propagate_fwd == True:
                self.general_df = update_consistency(general_df=self.general_df, fixed_frame_i=frame_i, fixed_df=lineage_df, consider_children=consider_children, fwd=propagate_fwd, bck=propagate_back)

            if update_fams == True:
                self.families = gen_df_to_fams(self.general_df)

            if not quick:
                self.gen_df_to_df_li()


        elif frame_i > len(self.lineage_list):
            printl(f'WARNING: Frame_i {frame_i} was inserted. The lineage list was only {len(self.lineage_list)} frames long, so the last known lineage tree was copy pasted up to frame_i {frame_i}')

            original_length = len(self.lineage_list)
            self.lineage_list = self.lineage_list + [self.lineage_list[-1]] * (frame_i - len(self.lineage_list))

            if not quick:
                self.generate_gen_df_from_df_li(self.lineage_list, force=True)

            self.lineage_list.append(lineage_df)

            frame_is = set(range(len(self.lineage_list)-original_length))
            self.frames_for_dfs = self.frames_for_dfs | frame_is

            if not quick:
                self.update_gen_df_from_df(lineage_df, frame_i)

            if propagate_back == True or propagate_fwd == True:
                self.general_df = update_consistency(general_df=self.general_df, fixed_frame_i=frame_i, fixed_df=lineage_df, consider_children=consider_children, fwd=propagate_fwd, bck=propagate_back)
            if update_fams == True:
                self.families = gen_df_to_fams(self.general_df)

            if not quick:
                self.gen_df_to_df_li()

    def propagate(self, frame_i, Cell_IDs_fixed=None):
        """
        Propagates the changes made to self.lineage_list at frame frame_i to the general DataFrame (self.general_df), families (self.families), and lineage list (self.gen_df_to_df_li()). The propagation can be done in both directions, and fixed cell IDs can be provided (in this case only those are propegated and the other Cell_IDs in the specified frame are ignored.)

        Parameters:
        - frame_i (int): The index of the frame to be propagated.
        - Cell_IDs_fixed (list, optional): List of fixed cell IDs. Defaults to None.

        Returns:
        - None
        """
        self.generate_gen_df_from_df_li(self.lineage_list, force=True)
        lineage_df = self.lineage_list[frame_i]
        self.update_gen_df_from_df(lineage_df, frame_i)
        self.general_df = update_consistency(general_df=self.general_df, fixed_frame_i=frame_i, consider_children=True, fwd=True, bck=True, Cell_IDs_fixed=Cell_IDs_fixed)
        self.families = gen_df_to_fams(self.general_df)
        self.gen_df_to_df_li()


    def load_lineage_df_list(self, df_li):
        """
        Loads a list of lineage dataframes into the CellACDC normal division tracker (self.lineage_list) based on df_li, which should be a list of acdc_df dataframes. Also updates the general_df (self.generate_gen_df_from_df_li()), families (self.gen_df_to_fams()), and lineage_list (self.gen_df_to_df_li())based on the loaded dataframes.

        Parameters:
        - df_li (list): A list of acdc_df dataframes.

        Returns:
        - None
        """
        # Support for first_frame was removed since it is not necessary, just make the df_li correct...
        # Also the tree needs to be init before. Also if df_li does not contain any relevenat dfs, nothing happens
        print('Loading lineage data...')
        df_li_new = []

        for i, df in enumerate(df_li):
            if df is None:
                continue

            if ('generation_num_tree' in df.columns 
                and not (df['generation_num_tree'] == 0).any()
                and not df['generation_num_tree'].isnull().any() 
                and not df["generation_num_tree"].isna().any() 
                and not df["generation_num_tree"].empty):

                if not df.index.name == 'Cell_ID':
                    df = (df
                            .reset_index()
                            .set_index('Cell_ID')
                            )

                df = filter_cols(df)
                df = reorg_sister_cells_for_inport(df)
                self.frames_for_dfs.add(i)
                df_li_new.append(df)

        if df_li_new:
            self.lineage_list = df_li_new
            self.generate_gen_df_from_df_li(self.lineage_list, force=True)
            self.families = gen_df_to_fams(self.general_df)
            self.gen_df_to_df_li()

    def export_df(self, frame_i):
        """
        Export the dataframe (from  self.lineage_list) for a specific frame. Drops the columns 'frame_i', 'level_0', and 'index' (latter two should not be present anysways, but just to be sure...)

        Parameters:
        - frame_i (int): The index of the frame.

        Returns:
        - pandas.DataFrame: The dataframe for frame_i.
        """
        df = self.lineage_list[frame_i].copy()

        if df.empty:
            print(f'Warning: No dataframe for frame {frame_i} found.')

        df = reorg_sister_cells_for_export(df)

        df = (df
              .reset_index()
              .set_index('Cell_ID')
              )

        try:
            df = df.drop('frame_i', axis=1)
        except KeyError:
            pass

        columns = df.columns
        if "level_0" in columns:
            df = df.drop(columns="level_0")
        if "index" in columns:
            df = df.drop(columns="index")

        return df
    
    def export_lin_tree_info(self, frame_i):
        df_curr = self.lineage_list[frame_i].copy()
        df_curr = checked_reset_index(df_curr)
        df_curr = df_curr.set_index('Cell_ID')
        df_prev = self.lineage_list[frame_i-1].copy()
        df_prev = checked_reset_index(df_prev)
        df_prev = df_prev.set_index('Cell_ID')

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
    - save_output(): Signals to the rest of the programme that the lineage tree should be saved. (Used for fodule 2)
    """
    def __init__(self):
        """
        Initializes the CellACDC_normal_division_tracker object.
        """
        pass

    def track(self,
              segm_video,
              IoA_thresh = 0.8,
              IoA_thresh_daughter = 0.25,
              IoA_thresh_aggressive = 0.5,
              min_daughter = 2,
              max_daughter = 2,
              record_lineage = True,
              return_tracked_lost_centroids = True,
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
            raise ValueError('return_tracked_lost_centroids can only be True if record_lineage is True.')
        
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
                    IoA_thresh = 0.8,
                    IoA_thresh_daughter = 0.25,
                    IoA_thresh_aggressive = 0.5,
                    min_daughter = 2,
                    max_daughter = 2,
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
        tracker.track_frame(1)
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