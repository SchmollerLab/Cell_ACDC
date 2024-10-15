import skimage
import numpy as np

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

def single_cell_seg(model, prev_lab, curr_lab, curr_img, IDs, padding=0.4, *model_args, **model_kwargs):
    """
    Function to segment single cells in the current frame using the previous frame segmentation as a reference. 
    IDs is from the previous frame segmentation, and the current frame should have alredy been tracked so the IDs match! After this, run tracking agian!!!
    Args:
        model: eval funciton used to segment the cells
        prev_lab: previous frame segmentation
        curr_lab: current frame segmentation
        curr_img: current frame image
        IDs: list of IDs of the cells to segment
        padding: padding to add to the bounding box of the cell
    Returns:
        curr_lab: current frame segmentation with the segmented cells
    """
    # curr_lab_copy = curr_lab.copy()
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

        box_model_lab = model.segment(box_curr_img, diameter=diameter, *model_args, **model_kwargs)

        areas = np.unique(box_model_lab.ravel(), return_counts=True)

        # Filter out the background (label 0) and sort by area
        areas = [(label, area) for label, area in zip(*areas) if label != 0]
        areas.sort(key=lambda x: x[1], reverse=True)

        # Get the label with the largest area
        if not len(areas) == 0:
            largest_id = areas[0][0]
            box_curr_lab_other_IDs[box_model_lab == largest_id] = ID

            curr_lab[box_x_min:box_x_max, box_y_min:box_y_max] = box_curr_lab_other_IDs

    return curr_lab