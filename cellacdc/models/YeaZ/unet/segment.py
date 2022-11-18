from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import dilation
from skimage.segmentation import watershed
from skimage.measure import label
from tqdm import tqdm

import numpy as np


def segment(th, pred, min_distance=10, topology=None, merge=True, q=0.75):
    """
    Performs watershed segmentation on thresholded image. Seeds have to
    have minimal distance of min_distance. topology defines the watershed
    topology to be used, default is the negative distance transform.
    Can either be an array with the same size af th, or a function that will
    be applied to the distance transform.

    After watershed, the borders found by watershed will be evaluated in terms
    of their predicted value. If the borders are highly predicted to be cells,
    the two cells are merged.
    """
    dtr = ndi.morphology.distance_transform_edt(th)
    if topology is None:
        topology = -dtr
    elif callable(topology):
        topology = topology(dtr)

    peak_idx = peak_local_max(-topology, min_distance)
    m = np.zeros_like(th)
    m[tuple(peak_idx.T)] = True

    # Uncomment to start with cross for every pixel instead of single pixel
    m_lab = label(m) #comment this
    #m_dil = dilation(m)
    #m_lab = label(m_dil)
    wsh = watershed(topology, m_lab, mask=th, connectivity=2)
    if merge:
        merged = cell_merge(wsh, pred, q)
    else:
        merged = wsh
    return correct_artefacts(merged)

def segment_stack(th, pred, min_distance=10, topology=None, signals=None):
    """
    source: YeaZ
    apply method segment on a stack of images, given a stack of thresholded
    images th and a stack of raw predictions pred
    """
    seg_stack = np.zeros(pred.shape, np.uint32)
    for idx, (th_single, pred_single) in enumerate(zip(th, pred)):
        lab = segment(th_single, pred_single, min_distance, topology)
        seg_stack[idx] = lab
        if signals is not None:
            signals.progressBar.emit(1)
    return seg_stack


def correct_artefacts(wsh):
    """
    Sometimes artefacts arise with 3 or less pixels which are surrounded entirely
    by another cell. Those are removed here.
    """
    unique, count = np.unique(wsh, return_counts=True)
    to_remove = unique[count<=3]
    for rem in to_remove:
        rem_im = wsh==rem
        rem_cont = dilation(rem_im) & ~rem_im
        vals, val_counts = np.unique(wsh[rem_cont], return_counts=True)
        replace_val = vals[np.argmax(val_counts)]
        if replace_val != 0:
            wsh[rem_im] = int(replace_val)
    return wsh


def cell_merge(wsh, pred, q=0.75):
    """
    Procedure that merges cells if the border between them is predicted to be
    cell pixels.
    """
    wshshape=wsh.shape

    # masks for the original cells
    objs = np.zeros((wsh.max()+1,wshshape[0],wshshape[1]), dtype=bool)

    # masks for dilated cells
    dil_objs = np.zeros((wsh.max()+1,wshshape[0],wshshape[1]), dtype=bool)

    # bounding box coordinates
    obj_coords = np.zeros((wsh.max()+1,4))

    # cleaned watershed, output of function
    wshclean = np.zeros((wshshape[0],wshshape[1]))

    # kernel to dilate objects
    kernel = np.ones((3,3), dtype=bool)

    for obj1 in range(wsh.max()):
        # create masks and dilated masks for obj
        objs[obj1,:,:] = wsh==(obj1+1)
        dil_objs[obj1,:,:] = dilation(objs[obj1,:,:], kernel)

        # bounding box
        obj_coords[obj1,:] = get_bounding_box(dil_objs[obj1,:,:])

    objcounter = 0	# counter for new watershed objects

    for obj1 in range(wsh.max()):
        dil1 = dil_objs[obj1,:,:]

        # check if mask has been deleted
        if np.sum(dil1) == 0:
            continue

        objcounter = objcounter + 1
        orig1 = objs[obj1,:,:]

        for obj2 in range(obj1+1,wsh.max()):
            dil2 = dil_objs[obj2,:,:]

            # only check border if bounding box overlaps, and second mask
            # is not yet deleted
            if (do_box_overlap(obj_coords[obj1,:], obj_coords[obj2,:])
                and np.sum(dil2) > 0):

                border = dil1 * dil2
                border_pred = pred[border]

                # Border is too small to be considered
                if len(border_pred) < 32:
                    continue

                # Sum of top 25% of predicted border values
                q75 = np.quantile(border_pred, q)
                top_border_pred = border_pred[border_pred >= q75]
                top_border_height = top_border_pred.sum()
                top_border_area = len(top_border_pred)

                # merge cells
                if top_border_height / top_border_area > .99:
                    orig1 = np.logical_or(orig1, objs[obj2,:,:])
                    dil_objs[obj1,:,:] = np.logical_or(dil1, dil2)
                    dil_objs[obj2,:,:] = np.zeros((wshshape[0], wshshape[1]))
                    obj_coords[obj1,:] = get_bounding_box(dil_objs[obj1,:,:])

        wshclean = wshclean + orig1*objcounter

    return wshclean


def do_box_overlap(coord1, coord2):
    """Checks if boxes, determined by their coordinates, overlap. Safety
    margin of 2 pixels"""
    return (
    (coord1[0] - 2 < coord2[0] and coord1[1] + 2 > coord2[0]
        or coord2[0] - 2 < coord1[0] and coord2[1] + 2 > coord1[0])
    and (coord1[2] - 2 < coord2[2] and coord1[3] + 2 > coord2[2]
        or coord2[2] - 2 < coord1[2] and coord2[3] + 2 > coord1[2]))


def get_bounding_box(im):
    """Returns bounding box of object in boolean image"""
    coords = np.where(im)

    return np.array([np.min(coords[0]), np.max(coords[0]),
                     np.min(coords[1]), np.max(coords[1])])
