"""Non-UI helpers used only by the main GUI window (gui/main_window/) and
its private ID widgets (gui/widgets/edit_toolbars.py): objects touching
the border (clear_border, get_edge_ids), slices of a local array into
the global one, acdc_df dtype fixes, checking the input of the trackers,
formatting lists of IDs and sorting IDs by distance. The modules shared
with the measurements viewer (gui/mixins/, measurements_viewer/) never
import this module.
"""
import traceback
import numpy as np
from cellacdc import printl


def _edge_ids_2d(lab):
    border_labels = np.r_[
        lab[0, :],
        lab[-1, :],
        lab[:, 0],
        lab[:, -1],
    ]
    return np.unique(border_labels[border_labels != 0])

def _edge_ids_3d(lab):
    face_labels = np.r_[
        lab[ 0, :, :].ravel(),    # z min
        lab[-1, :, :].ravel(),   # z max
        lab[:,  0, :].ravel(),    # y min
        lab[:, -1, :].ravel(),   # y max
        lab[:, :,  0].ravel(),    # x min
        lab[:, :, -1].ravel(),   # x max
    ]
    ids = np.unique(face_labels)
    return ids[ids != 0]

def get_edge_ids(lab):
    if lab.ndim == 2:
        return _edge_ids_2d(lab)
    elif lab.ndim == 3:
        return _edge_ids_3d(lab)
    else:
        raise ValueError('Label array must be either 2D or 3D.')

def clear_border(lab, return_edge_ids=False):
    # probably faster than skimage since it avoids relabeling...
    # assumes continous unique IDs, which we have. Modifies inplace!
    edge_ids = get_edge_ids(lab)
    lab[np.isin(lab, edge_ids)] = 0
    if return_edge_ids:
        return edge_ids


def get_slices_local_into_global_arr(bbox_coords, global_shape):
    slice_global_to_local = []
    slice_crop_local = []
    for (_min, _max), _D in zip(bbox_coords, global_shape):
        _min_crop, _max_crop = None, None
        if _min < 0:
            _min_crop = abs(_min)
            _min = 0
        if _max > _D:
            _max_crop = _D - _max
            _max = _D
        
        slice_global_to_local.append(slice(_min, _max))
        slice_crop_local.append(slice(_min_crop, _max_crop))
    
    return tuple(slice_global_to_local), tuple(slice_crop_local)


def fix_acdc_df_dtypes(acdc_df):
    acdc_df['is_cell_excluded'] = acdc_df['is_cell_excluded'].astype(bool)
    return acdc_df


def validate_tracker_input(tracker, segm_video_to_track):
    try:
        warning_text = tracker.validate_input(segm_video_to_track)
        return warning_text
    except Exception as err:
        printl(traceback.format_exc())
        pass
    return
def format_IDs(IDs):
    if isinstance(IDs, str):
        raise ValueError('IDs must not be a string')

    IDsRange = []
    text = ''
    sorted_vals = sorted(IDs)
    for i, e in enumerate(sorted_vals):
        e = int(e)
        # Get previous and next value (if possible)
        if i > 0:
            prevVal = sorted_vals[i-1]
        else:
            prevVal = -1
        if i < len(sorted_vals)-1:
            nextVal = sorted_vals[i+1]
        else:
            nextVal = -1

        if e-prevVal == 1 or nextVal-e == 1:
            if not IDsRange:
                if nextVal-e == 1 and e-prevVal != 1:
                    # Current value is the first value of a new range
                    IDsRange = [e]
                else:
                    # Current value is the second element of a new range
                    IDsRange = [prevVal, e]
            else:
                if e-prevVal == 1:
                    # Current value is part of an ongoing range
                    IDsRange.append(e)
                else:
                    # Current value is the first element of a new range 
                    # --> create range text and this element will 
                    # be added to the new range at the next iter
                    start, stop = IDsRange[0], IDsRange[-1]
                    if stop-start > 1:
                        sep = '-'
                    else:
                        sep = ','
                    text = f'{text},{start}{sep}{stop}'
                    IDsRange = []
        else:
            # Current value doesn't belong to a range
            if IDsRange:
                # There was a range not added to text --> add it now
                start, stop = IDsRange[0], IDsRange[-1]
                if stop-start > 1:
                    sep = '-'
                else:
                    sep = ','
                text = f'{text},{start}{sep}{stop}'
            
            text = f'{text},{e}'    
            IDsRange = []

    if IDsRange:
        # Last range was not added  --> add it now
        start, stop = IDsRange[0], IDsRange[-1]
        text = f'{text},{start}-{stop}'

    text = text[1:]

    return text


def find_distances_ID(rps, point=None, ID=None, relevant_IDs=None):
    """
    Calculate the distances between a given point and the centroids of a list of regionprops.

    Parameters
    ----------
    rps : acdcRegionprops
        Regionprops collection.
    point : tuple, optional
        Point coordinates in (x, y) or (x, y, z) order. Defaults to None.
    ID : int, optional
        The label ID of the regionprops object. Defaults to None.
    relevant_IDs : collection of int, optional
        Restrict the distance calculation to these labels.

    Returns
    -------
    numpy.ndarray
        A (1, N) matrix of distances in regionprops iteration order, filtered
        by relevant_IDs if provided.

    Raises
    ------
    ValueError
        If ID is not found in the list of regionprops (list of cells).
    ValueError
        If neither ID nor point is provided.
    ValueError
        If both ID and point are provided.
    """

    if ID is not None and point is None:
        if ID not in rps.IDs:
            raise ValueError(f'ID {ID} not found in regionprops (list of cells).')
        point = rps.get_centroid(ID)

    elif ID is None and point is None:
        raise ValueError('Either ID or point must be provided.')

    elif ID is not None and point is not None:
        raise ValueError('Only one of ID or point must be provided.')
    else:
        point = point[::-1]

    point = np.array([point])
    if relevant_IDs is not None:
        centroids = np.array([rps.get_centroid(obj.label) for obj in rps if obj.label in relevant_IDs])
    else:
        centroids = np.array([rps.get_centroid(obj.label) for obj in rps]) # here RPS is a list and not a regionprops object
    if len(centroids) == 0:
        return np.empty((1, 0))
    diff = point[:, np.newaxis] - centroids
    dist_matrix = np.linalg.norm(diff, axis=2)
    return dist_matrix

def sort_IDs_dist(rps, point=None, ID=None, relevant_IDs=None):
    """Sorts the IDs of regionprops based on their distances to a given point.

    Parameters
    ----------
    rps : acdcRegionprops
        Regionprops collection representing cells.
    point : tuple, optional
        Point coordinates in (x, y) or (x, y, z) order.
        If not provided, it will be calculated based on the given ID.
    ID : int, optional
        The ID of the regionprops object to calculate distances from. 
        If this and point are both provided, or neither, an error will be 
        raised.
    relevant_IDs : collection of int, optional
        Restrict the returned IDs to these labels.

    Returns
    -------
    list
        IDs sorted nearest first, with equal distances ordered by ID.

    Raises
    ------
    ValueError
        If ID is not found in the list of regionprops objects.
    ValueError
        If neither ID nor point is provided.
    ValueError
        If both ID and point are provided.

    """
    IDs = [
        obj.label for obj in rps
        if relevant_IDs is None or obj.label in relevant_IDs
    ]
    dist_matrix = find_distances_ID(
        rps, point=point, ID=ID, relevant_IDs=relevant_IDs
    ).ravel()

    sorted_ids = sorted(zip(dist_matrix, IDs))
    sorted_ids = [ID for _, ID in sorted_ids]
    return sorted_ids
