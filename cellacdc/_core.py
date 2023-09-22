from typing import Union

import traceback

import numpy as np

from math import log10, floor
from skimage.transform import rotate as skimage_rotate
from skimage.measure import regionprops

from . import printl

time_units_formats = {
    'min': 'minutes', 
    'hour': 'hours', 
    'second': 'seconds', 
    'minutes': 'minutes',
    'seconds': 'seconds', 
    'hours': 'hours', 
    'H': 'hours',
    'd': 'days',
    'M': 'minutes',
    'S': 'seconds',
}

time_units_converters = {
    'seconds -> minutes': lambda x: x/60,
    'seconds -> hours': lambda x: x/3600,
    'seconds -> days': lambda x: x/3600/24,
    'minutes -> hours': lambda x: x/60,
    'minutes -> seconds': lambda x: x*60,
    'minutes -> days': lambda x: x/60/24,
    'hours -> minutes': lambda x: x*60,
    'hours -> seconds': lambda x: x*3600,
    'hours -> days': lambda x: x/24,
    'days -> minutes': lambda x: x*24*60,
    'days -> seconds': lambda x: x*24*3600,
    'days -> hours': lambda x: x*24*3600,
}

length_unit_converters = {
    'nm -> μm': lambda x: x/1000,
    'mm -> μm': lambda x: x*1e3,
    'cm -> μm': lambda x: x*1e4,
    'μm -> μm': lambda x: x,
}

def convert_length(value, from_unit, to_unit):
    key = f'{from_unit} -> {to_unit}'
    return length_unit_converters[key](value)

def round_to_significant(n, n_significant=1):
    return round(n, n_significant-int(floor(log10(abs(n))))-1)

def convert_time_units(x, from_unit, to_unit):
    try:
        from_unit = time_units_formats[from_unit.strip()]
        to_unit = time_units_formats[to_unit.strip()]
        key = f"{from_unit} -> {to_unit}"
        func = time_units_converters[key]
        return func(x)
    except Exception as e:
        return

def _calc_rotational_vol(obj, PhysicalSizeY=1, PhysicalSizeX=1, logger=None):
    """Given the region properties of a 2D object (from skimage.measure.regionprops).
    calculate the rotation volume as described in the Supplementary information of
    https://www.nature.com/articles/s41467-020-16764-x

    Parameters
    ----------
    obj : class skimage.measure.RegionProperties
        Single item of the list returned by from skimage.measure.regionprops.
    PhysicalSizeY : type
        Physical size of the pixel in the Y-diretion in micrometer/pixel.
    PhysicalSizeX : type
        Physical size of the pixel in the X-diretion in micrometer/pixel.

    Returns
    -------
    tuple
        Tuple of the calculate volume in voxels and femtoliters.

    Notes
    -------
    For 3D objects we take max projection

    We convert PhysicalSizeY and PhysicalSizeX to float because when they are
    read from csv they might be a string value.

    """
    is3Dobj = False
    try:
        orientation = obj.orientation
    except NotImplementedError as e:
        # if obj.image.ndim != 3:
        #     printl(e, obj.image.ndim, obj.bbox, obj.centroid)
        is3Dobj = True

    try:
        if is3Dobj:
            # For 3D objects we use a max projection for the rotation
            obj_lab = obj.image.max(axis=0).astype(np.uint32)*obj.label
            obj = regionprops(obj_lab)[0]

        vox_to_fl = float(PhysicalSizeY)*pow(float(PhysicalSizeX), 2)
        rotate_ID_img = skimage_rotate(
            obj.image.astype(np.single), -(obj.orientation*180/np.pi),
            resize=True, order=3
        )
        radii = np.sum(rotate_ID_img, axis=1)/2
        vol_vox = np.sum(np.pi*(radii**2))
        if vox_to_fl is not None:
            return vol_vox, float(vol_vox*vox_to_fl)
        else:
            return vol_vox, vol_vox
    except Exception as e:
        if logger is not None:
            logger.exception(traceback.format_exc())
        else:
            printl(traceback.format_exc())
        return np.nan, np.nan
