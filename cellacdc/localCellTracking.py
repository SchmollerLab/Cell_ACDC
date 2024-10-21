import skimage
import numpy as np
from .core import segm_model_segment, post_process_segm
from .features import custom_post_process_segm

def find_overlap(lab_1, lab_2):
    """
    Function to find the overlap between two labeled images.
    Args:
        lab_1: first labeled image
        lab_2: second labeled image
    Returns:
        ID_overlap: list of tuples with the ID of the object in lab_1 and the percentage of overlap with lab_2
    """
    overlap_mask = np.logical_and(lab_1 > 0, lab_2 > 0)

    unique_labels_lab_1 = np.unique(lab_1)
    unique_labels_lab_1 = unique_labels_lab_1[unique_labels_lab_1 > 0]

    ID_overlap = []

    for label in unique_labels_lab_1:
        region_lab_1 = lab_1 == label
        overlap_area = np.sum(np.logical_and(region_lab_1, overlap_mask))
        if overlap_area == 0:
            continue

        total_area = np.sum(region_lab_1)
        overlap_perc = overlap_area / total_area
        ID_overlap.append((label, overlap_perc))

    return ID_overlap

def get_obj_from_rps(rps, ID):
    for obj in rps:
        if obj.label == ID:
            return obj
    return None

def get_box_coords(rps, prev_lab_shape, ID, padding):
    """
    Calculate the coordinates of a bounding box around a given ID in a labeled image,
    with optional padding.
    Parameters:
    rps (list): A list of regionprops objects for the labeled image.
    prev_lab_shape (tuple): The shape of the labeled image.
    ID (int): The ID of the object for which to calculate the bounding box.
    padding (float): The fraction of the object's size to use as padding around the bounding box.
    Returns:
    tuple: A tuple containing the coordinates of the bounding box (box_x_min, box_x_max, box_y_min, box_y_max).
    """

    obj = get_obj_from_rps(rps, ID)
    box_x_min, box_y_min, box_x_max, box_y_max = obj.bbox

    size_x = box_x_max - box_x_min
    size_y = box_y_max - box_y_min

    padding_x = int(size_x * padding)
    padding_y = int(size_y * padding)

    box_x_min = max(0, box_x_min - padding_x)
    box_x_max = min(prev_lab_shape[0], box_x_max + padding_x)
    box_y_min = max(0, box_y_min - padding_y)
    box_y_max = min(prev_lab_shape[1], box_y_max + padding_y)

    return box_x_min, box_x_max, box_y_min, box_y_max

def single_cell_seg(model, prev_lab, curr_lab, curr_img, IDs, new_unique_ID, 
                    new_max_obj,
                    overlap_threshold=0.5, padding=0.4, size_perc_threshold=0.5, 
                    win=None, posData=None,):
    """
    Function to segment single cells in the current frame using the previous frame segmentation as a reference. 
    IDs is from the previous frame segmentation, and the current frame should have alredy been tracked so the IDs match!
    Args:
        model: eval funciton used to segment the cells
        prev_lab: previous frame segmentation
        curr_lab: current frame segmentation
        curr_img: current frame image
        IDs: list of IDs of the cells to segment
        new_new_unique_ID: ID to start labeling new cells
        max_obj: maximum number of objects expected
        overlap_threshold: minimum overlap percentage to consider a cell already segmented
        padding: padding around the cell to segment
        size_perc_threshold: minimum percentage of the largest cell size to consider a cell
        win: from the gui window which sets model params
        posData: position data for the
    Returns:
        curr_lab: current frame segmentation with the segmented cells
        assigned_IDs: list of IDs assigned to the segmented cells

    """
    assigned_IDs = []

    model_kwargs = win.model_kwargs
    preproc_recipe = win.preproc_recipe
    applyPostProcessing = win.applyPostProcessing
    standardPostProcessKwargs = win.standardPostProcessKwargs
    customPostProcessFeatures = win.customPostProcessFeatures
    customPostProcessGroupedFeatures = win.customPostProcessGroupedFeatures

    prev_rps = skimage.measure.regionprops(prev_lab)
    prev_lab_shape = prev_lab.shape
    
    for ID in IDs:
        box_x_min, box_x_max, box_y_min, box_y_max = get_box_coords(prev_rps, prev_lab_shape, ID, padding)

        box_curr_img = curr_img[box_x_min:box_x_max, box_y_min:box_y_max].copy()
        box_curr_lab = curr_lab[box_x_min:box_x_max, box_y_min:box_y_max].copy()

        box_curr_lab_other_IDs = box_curr_lab.copy()
        box_curr_lab_other_IDs[box_curr_lab_other_IDs == ID] = 0

        # Fill other IDs with random samples from the background
        indices_to_fill = np.where(box_curr_lab_other_IDs != 0)
        box_background = box_curr_img[box_curr_lab==0]
        random_samples = np.random.choice(box_background, size=indices_to_fill[0].shape, replace=True)
        box_curr_img[indices_to_fill] = random_samples

        # Run model, give it the diameter of cell if possible
        obj = get_obj_from_rps(prev_rps, ID)
        diameter = obj.axis_major_length
        # try:
        model_kwargs['diameter'] = diameter
        box_model_lab = segm_model_segment(
            model, box_curr_img, model_kwargs,
            preproc_recipe=preproc_recipe,
            posData=posData
        )

        # Post-processing        
        if applyPostProcessing:
            box_model_lab = post_process_segm(
                box_model_lab, **standardPostProcessKwargs
            )
            if customPostProcessFeatures:
                box_model_lab = custom_post_process_segm(
                    posData, 
                    customPostProcessGroupedFeatures, 
                    box_model_lab, box_curr_img, posData.frame_i, 
                    posData.filename, 
                    posData.user_ch_name, 
                    customPostProcessFeatures
                )

        # Find the overlap between the model segmentation and the other IDs
        overlap = find_overlap(box_model_lab, box_curr_lab_other_IDs)

        # Set overlapping regions to 0, so already segmented cells are not overwritten
        for ID, overlap_perc in overlap:
            if overlap_perc > overlap_threshold:
                box_model_lab[box_model_lab == ID] = 0

        areas = np.unique(box_model_lab.ravel(), return_counts=True)

        filtered_areas = [label for label, area in zip(*areas) if label != 0 and area >= size_perc_threshold * area]
        
        if len(filtered_areas) not in range(1, new_max_obj + 1):
            # too many cells, could not successfully segment, for budding this is one mother and one duaghter, for normal cell division this is two daughters
            continue

        for label in filtered_areas:
            box_curr_lab_other_IDs[box_model_lab == label] = new_unique_ID
            assigned_IDs.append(new_unique_ID)
            new_unique_ID += 1
            
        curr_lab[box_x_min:box_x_max, box_y_min:box_y_max] = box_curr_lab_other_IDs

    return curr_lab, assigned_IDs