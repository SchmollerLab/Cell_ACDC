import numpy as np

import skimage.segmentation
import skimage.measure

import cv2

def _find_contours_2D(
        image, bbox_lower_coords=(0, 0), all=False, closed=True
    ):
    mode = cv2.RETR_CCOMP if all else cv2.RETR_EXTERNAL
    contours, _ = cv2.findContours(image, mode, cv2.CHAIN_APPROX_NONE)
    
    if all:
        all_contours = [
            np.squeeze(contour, axis=1)+bbox_lower_coords 
            for contour in contours
        ]
        if closed:
            all_contours = [
                np.vstack((contour, contour[0])) for contour in contours
            ]
        return all_contours
    else:
        contour = np.squeeze(contours[0], axis=1)
        if closed:
            contour = np.vstack((contour, contour[0]))
        contour = contour + bbox_lower_coords
        return contour

def find_obj_contour(
        obj: skimage.measure._regionprops.RegionProperties, all=False, 
        local=False, do_z_max_proj=False, closed=True
    ):
    is3D = obj.image.ndim == 3
    bbox_y_idx = 1 if is3D else 0

    if local:
        bbox_lower_coords=(0, 0)
    else:
        min_y, min_x = obj.bbox[bbox_y_idx:bbox_y_idx+2]
        bbox_lower_coords = (min_x, min_y)

    if is3D and do_z_max_proj:
        is3D = False
        obj_image = obj.max(axis=0).astype(np.uint8)
    else:
        obj_image = obj.image.astype(np.uint8)

    kwargs = {
        'bbox_lower_coords': bbox_lower_coords, 
        'all':all, 'closed': closed
    }
    if is3D:
        contours = [
            _find_contours_2D(image_z, **kwargs) for image_z in obj_image
        ]
    else:
        contours = _find_contours_2D(obj_image, **kwargs)
    return contours

def find_contours(
        label_img, connectivity=1, mode='thick', background=0, 
        return_coords=False, **kwargs
    ):
    """Return bool array where boundaries between labeled regions are True. 
    If `return_coords` is True then return also a list of objects' contours
    coordinates.

    Parameters
    ----------
    label_img : (M, N[, P]) ndarray
        An array in which different regions are labeled with either different
        integers or boolean values.
    connectivity : int, optional
        int in {1, ..., `label_img.ndim`}, optional
        A pixel is considered a boundary pixel if any of its neighbors
        has a different label. `connectivity` controls which pixels are
        considered neighbors. A connectivity of 1 (default) means
        pixels sharing an edge (in 2D) or a face (in 3D) will be
        considered neighbors. A connectivity of `label_img.ndim` means
        pixels sharing a corner will be considered neighbors. Default is 1.
    mode : str, optional
        How to mark the boundaries:
        - thick: any pixel not completely surrounded by pixels of the
          same label (defined by `connectivity`) is marked as a boundary.
          This results in boundaries that are 2 pixels thick.
        - inner: outline the pixels *just inside* of objects, leaving
          background pixels untouched.
        - outer: outline pixels in the background around object
          boundaries. When two objects touch, their boundary is also
          marked.
        - subpixel: return a doubled image, with pixels *between* the
          original pixels marked as boundary where appropriate.,
        
        By default 'thick'
    background : int, optional
        For modes 'inner' and 'outer', a definition of a background
        label is required. See `mode` for descriptions of these two,
        by default 0
    return_coords : bool, optional
        If ``True``, also return a list of objects' contours coordinates,
        by default False
    kwargs : dict, optional
        Additional arguments passed `acdctools.segmentation.find_obj_contour` 
        function. This function uses the opencv find contours function 
        `cv2.findContours`. Used only if `mode='inner'`.

    Returns
    -------
    boundaries : ndarray of bool, same shape as `label_img`
        A bool image where ``True`` represents a boundary pixel. For
        `mode` equal to 'subpixel', ``boundaries.shape[i]`` is equal
        to ``2 * label_img.shape[i] - 1`` for all ``i`` (a pixel is
        inserted in between all other pairs of pixels).
    contours_coords: list of ndarray
        A list of ndarrays with shape (N, n) where `n` is the number of
        dimensions of `label_img` and `N` is the number of points in each 
        object's contour. The list contains one ndarray per object in 
        `label_img`. 
        The ordering of columns follows the numpy's order of dimensions 
        convention, e.g., for 2-D, the first and second column are the 
        y and x coordinates, respectively. 
        Only provided if `return_coords` is True.
    """    
    boundaries = skimage.segmentation.find_boundaries(
        label_img, connectivity=connectivity, mode=mode, background=background
    )
    if not return_coords:
        return boundaries
    
    is2D = label_img.ndim == 2
    rp = skimage.measure.regionprops(label_img)
    contours_coords = []
    for obj in rp:
        if mode == 'inner' and is2D:
            pass
        else:
            pass
