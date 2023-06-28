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
