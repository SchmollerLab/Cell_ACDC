import numpy as np

from skimage.transform import resize
from skimage.measure import regionprops

from . import printl

def resize_lab(lab, output_shape, rp=None):
    if rp is None:
        rp = regionprops(lab)
    _lab_obj_to_resize = np.zeros(lab.shape, dtype=np.float16)
    lab_resized = np.zeros(output_shape, dtype=np.uint32)
    for obj in rp:
        _lab_obj_to_resize[obj.slice][obj.image] = 1.0
        _lab_obj_resized = resize(
            _lab_obj_to_resize, output_shape, anti_aliasing=True,
            preserve_range=True
        ).round()
        lab_resized[_lab_obj_resized == 1.0] = obj.label
        _lab_obj_to_resize[:] = 0.0
    return lab_resized

def crop_2D(img, xy_range, tolerance=0, return_copy=True):
    (xmin, xmax), (ymin, ymax) = xy_range
    Y, X = img.shape
    xmin -= tolerance
    xmax += tolerance
    ymin -= tolerance
    ymax += tolerance
    xmin = 0 if xmin < 0 else round(xmin)
    xmax = X if xmax > X else round(xmax)
    ymin = 0 if ymin < 0 else round(ymin)
    ymax = Y if ymax > Y else round(ymax)
    crop_shape = (ymax-ymin, xmax-xmin)
    crop_slice = (slice(ymin, ymax, None), slice(xmin, xmax, None))
    if return_copy:
        cropped = np.zeros(crop_shape, dtype=img.dtype)
        cropped[:] = img[crop_slice]
    else:
        cropped = img[crop_slice]
    return cropped, crop_slice

def del_objs_outside_segm_roi(segm_roi, segm):
    del_IDs = np.unique(segm[segm_roi==0])
    cleared_segm = segm.copy()
    clearedIDs = []
    for del_ID in del_IDs:
        if del_ID == 0:
            continue
        cleared_segm[segm==del_ID] = 0
        clearedIDs.append(del_ID)
    return cleared_segm, clearedIDs