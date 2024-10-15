import skimage
import numpy as np

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

def single_cell_seg(model, prev_lab, curr_lab, curr_img, IDs, new_unique_ID, budding, max_daughters, min_daughters, overlap_threshold=0.5, padding=0.4, size_perc_threshold=0.5,*model_args, **model_kwargs):
    """
    Function to segment single cells in the current frame using the previous frame segmentation as a reference. 
    IDs is from the previous frame segmentation, and the current frame should have alredy been tracked so the IDs match! After this, run tracking agian!!!
    Args:
        model: eval funciton used to segment the cells
        prev_lab: previous frame segmentation
        curr_lab: current frame segmentation
        curr_img: current frame image
        IDs: list of IDs of the cells to segment
        new_unique_ID: ID to start labeling new cells
        budding: boolean, if the cells are budding
        max_daughters: maximum number of daughters expected
        min_daughters: minimum number of daughters expected
        overlap_threshold: minimum overlap percentage to consider a cell already segmented
        padding: padding around the cell to segment
        size_perc_threshold: minimum percentage of the largest cell size to consider a cell
        model_args: arguments for the model
        model_kwargs: keyword arguments for the model
    Returns:
        curr_lab: current frame segmentation with the segmented cells
    """
    if budding:
        # if budding, we expect one more resulting cell than there are daughters, as the original cell is also present
        max_daughters += 1
        min_daughters += 1

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
        try:
            if 'diameter' in model_kwargs:
                del model_kwargs['diameter']

            box_model_lab = model.segment(box_curr_img, diameter=diameter, *model_args, **model_kwargs)
        except TypeError as e:
            # Check if the TypeError is due to an unexpected keyword argument 'diameter'
            if "got an unexpected keyword argument 'diameter'" in str(e):
                # Retry without the 'diameter' keyword argument
                box_model_lab = model.segment(box_curr_img, *model_args, **model_kwargs)
            else:
                # Re-raise if it's a different TypeError
                raise
        
        # Find the overlap between the model segmentation and the other IDs
        overlap = find_overlap(box_model_lab, box_curr_lab_other_IDs)

        # Set overlapping regions to 0, so already segmented cells are not overwritten
        for ID, overlap_perc in overlap:
            if overlap_perc > overlap_threshold:
                box_model_lab[box_model_lab == ID] = 0

        areas = np.unique(box_model_lab.ravel(), return_counts=True)

        # Filter out the background (label 0) and sort by area
        areas = [(label, area) for label, area in zip(*areas) if label != 0]
        areas.sort(key=lambda x: x[1], reverse=True)

        # Get the label with the largest area
        if len(areas) == 0:
            continue

        labels_in_size_range = []
        _, largest_area = areas[0]
        for label, area in areas:
            if area >= size_perc_threshold * largest_area:
                labels_in_size_range.append(label)

        if len(labels_in_size_range) == 1:
            # just one cell, probably the one we are looking for
            box_curr_lab_other_IDs[box_model_lab == labels_in_size_range[0]] = ID
            continue
        
        elif len(labels_in_size_range) not in range(min_daughters, max_daughters + 1):
            # too many or too few cells, could not successfully segment, for budding this is one mother and one duaghter, for normal cell division this is two daughters
            continue
        
        # multiple cells in range for daugher amound, probably cell seperation
        if not budding:
            for label in labels_in_size_range:
                box_curr_lab_other_IDs[box_model_lab == label] = new_unique_ID
                new_unique_ID += 1
        else:
            # budding, one mother and daughters
            box_curr_lab_other_IDs[box_model_lab == labels_in_size_range[0]] = ID
            labels_in_size_range.pop(0)

            for label in labels_in_size_range:
                box_curr_lab_other_IDs[box_model_lab == label] = new_unique_ID
                new_unique_ID += 1
            
        curr_lab[box_x_min:box_x_max, box_y_min:box_y_max] = box_curr_lab_other_IDs

    return curr_lab