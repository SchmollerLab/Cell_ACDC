import numpy as np

import skimage.transform
import skimage.measure

def rotational_volume(
        obj: skimage.measure._regionprops.RegionProperties, 
        PhysicalSizeY=1.0, PhysicalSizeX=1.0, vox_to_fl=None
    ):
    """Given the region properties of a 2D or 3D object (from skimage.measure.regionprops).
    calculate the rotation volume as described in the Supplementary information of
    https://www.nature.com/articles/s41467-020-16764-x

    Parameters
    ----------
    obj : skimage.measure.RegionProperties
        Single item of the list returned by from skimage.measure.regionprops.
    PhysicalSizeY : float, optional
        Physical size of the pixel in the Y-diretion in micrometer/pixel.
        By default 1.0
    PhysicalSizeX : float, optional
        Physical size of the pixel in the X-diretion in micrometer/pixel.
        By default 1.0
    
    Returns
    -------
    tuple
        Volume in voxels, volume in femtoliters.

    Notes
    -------
    For 3D objects we take the max projection. 

    We convert PhysicalSizeY and PhysicalSizeX to float because when they are
    read from csv they might be a string value.
    """    
    if obj.image.ndim == 3:
        obj_image = obj.image.max(axis=0)
        obj_rp = skimage.measure.regionprops(obj_image.astype(np.uint8))[0]
        obj_orientation = obj_rp.orientation
    else:
        obj_image = obj.image
        obj_orientation = obj.orientation
    
    if vox_to_fl is None:
        vox_to_fl = float(PhysicalSizeY)*(float(PhysicalSizeX)**2)
    
    rotate_ID_img = skimage.transform.rotate(
        obj_image.astype(np.uint8), -(obj_orientation*180/np.pi),
        resize=True, order=3, preserve_range=True
    )
    radii = np.sum(rotate_ID_img, axis=1)/2
    vol_vox = np.sum(np.pi*(radii**2))
    return vol_vox, float(vol_vox*vox_to_fl)

def separate_with_label(lab, rp, IDs_to_separate, maxID, click_coords_list=None):
    separate_lab = lab.copy()
    for obj_idx, obj in enumerate(rp):
        if obj.label not in IDs_to_separate:
            continue
        label_obj = skimage.measure.label(obj.image)
        label_obj_rp = skimage.measure.regionprops(label_obj)
        if click_coords_list is None:
            areas = [sub_obj.area for sub_obj in label_obj_rp]
            max_area = max(areas)
            max_area_idx = areas.index(max_area)
            id_to_keep = label_obj_rp[max_area_idx].label
        else:
            click_coords = click_coords_list[IDs_to_separate.index(obj.label)]
            if len(obj.bbox) == 6:
                zmin, ymin, xmin, _, _, _ = obj.bbox
                zclick, yclick, xclick = click_coords
                click_z_local = zclick - zmin
                click_y_local = yclick - ymin
                click_x_local = xclick - xmin
                id_to_keep = label_obj[click_z_local, click_y_local, click_x_local]
            else:
                ymin, xmin, _, _ = obj.bbox
                yclick, xclick = click_coords
                click_y_local = yclick - ymin
                click_x_local = xclick - xmin
                id_to_keep = label_obj[click_y_local, click_x_local]
        
        separate_lab[obj.slice][obj.image] = 0
        for sub_obj_idx, sub_obj in enumerate(label_obj_rp):
            if sub_obj.label == id_to_keep:
                new_ID = obj.label
            else:
                new_ID = maxID + 1 + sub_obj_idx
            separate_lab[obj.slice][sub_obj.slice][sub_obj.image] = new_ID
    return separate_lab
            