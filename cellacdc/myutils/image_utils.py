"""Image and array helpers: dtype conversions (to_uint8, to_uint16 and
img_to_float are plugin API), writing TIFF files with metadata, bounds
checks, iterating over axes, finding an object by its label, synthetic
test images and the input/output index mapper used by core.py.
"""
from typing import List, Tuple
import numpy as np
import skimage
import itertools
import tifffile
import skimage.io
import skimage.measure
from cellacdc import _warnings


def is_in_bounds(x,y,X,Y):
    in_bounds = x >= 0 and x < X and y >= 0 and y < Y
    return in_bounds


# def get_tiff_metadata(
#         image_arr,
#         SizeT=None, 
#         SizeZ=None, 
#         PhysicalSizeZ=None,
#         PhysicalSizeX=None, 
#         PhysicalSizeY=None,
#         TimeIncrement=None
#     ):
#     SizeY, SizeX = image_arr.shape[-2:]
#     Type = str(image_arr.dtype)
    
#     metadata = {
#         'SizeX': SizeX,
#         'SizeY': SizeY,
#         'Type': Type
#     }
    
#     axes = 'YX'
#     if SizeZ is not None and SizeZ > 1:
#         axes = f'Z{axes}'
#         metadata['SizeZ'] = SizeZ
        
#     if SizeT is not None and SizeT > 1:
#         axes = f'T{axes}'
#         metadata['SizeT'] = SizeT
        
#     metadata['axes'] = axes
    
#     if PhysicalSizeX is not None:
#         metadata['PhysicalSizeX'] = PhysicalSizeX
    
#     if PhysicalSizeY is not None:
#         metadata['PhysicalSizeY'] = PhysicalSizeY
    
#     if PhysicalSizeZ is not None:
#         metadata['PhysicalSizeZ'] = PhysicalSizeZ
    
#     if TimeIncrement is not None:
#         metadata['TimeIncrement'] = TimeIncrement
    
#     return metadata

def get_tiff_metadata(
        image_arr,
        SizeT=None, 
        SizeZ=None, 
        PhysicalSizeZ=None,
        PhysicalSizeX=None, 
        PhysicalSizeY=None,
        TimeIncrement=None
    ):
    SizeY, SizeX = image_arr.shape[-2:]
    Type = str(image_arr.dtype)
    
    metadata = {
        'Pixels': {
            'SizeX': SizeX,
            'SizeY': SizeY,
            'Type': Type
        }
    }
    
    axes = 'YX'
    if SizeZ is not None and SizeZ > 1:
        axes = f'Z{axes}'
        metadata['Pixels']['SizeZ'] = SizeZ
        
    if SizeT is not None and SizeT > 1:
        axes = f'T{axes}'
        metadata['Pixels']['SizeT'] = SizeT
        
    metadata['axes'] = axes
    
    if PhysicalSizeX is not None:
        metadata['Pixels']['PhysicalSizeX'] = PhysicalSizeX
    
    if PhysicalSizeY is not None:
        metadata['Pixels']['PhysicalSizeY'] = PhysicalSizeY
    
    if PhysicalSizeZ is not None:
        metadata['Pixels']['PhysicalSizeZ'] = PhysicalSizeZ
    
    if TimeIncrement is not None:
        metadata['Pixels']['TimeIncrement'] = TimeIncrement
    
    return metadata

def to_tiff(
        new_path, data, 
        SizeT=None, 
        SizeZ=None, 
        PhysicalSizeZ=None,
        PhysicalSizeX=None, 
        PhysicalSizeY=None,
        TimeIncrement=None
    ):
    valid_dtypes = (
        np.uint8, np.uint16, np.float32
    )
    is_valid_dtype = False
    for valid_dtype in valid_dtypes:
        if np.issubdtype(data.dtype, valid_dtype):
            is_valid_dtype = True
            break
    
    if not is_valid_dtype:
        data = data.astype(np.float32)
    
    metadata = get_tiff_metadata(
        data,
        SizeT=SizeT, 
        SizeZ=SizeZ, 
        PhysicalSizeZ=PhysicalSizeZ,
        PhysicalSizeX=PhysicalSizeX, 
        PhysicalSizeY=PhysicalSizeY,
        TimeIncrement=TimeIncrement
    )
    
    # # Potential alternative 
    # hyperstack = tifffile.memmap(
    #     new_path,
    #     shape=img.shape,
    #     dtype=img.dtype,
    #     imagej=True,
    #     metadata={'axes': 'TZYX'},
    # )
    # hyperstack[:] = img
    # hyperstack.flush()
    
    try:
        tifffile.imwrite(
            new_path, data, metadata=metadata, imagej=True
        )
    except Exception as err:
        tifffile.imwrite(new_path, data)


def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = np.round(img_to_float(img)*255).astype(np.uint8)
    return img

def to_uint16(img):
    if img.dtype == np.uint16:
        return img
    img = np.round(img_to_float(img)*65535).astype(np.uint16)
    return img


def img_to_float(img, force_dtype=None, force_missing_dtype=None, warn=True):
    input_img_dtype = img.dtype
    value = img[(0,) * img.ndim]
    img_max = np.max(img)
    # Check if float outside of -1, 1
    if img_max <= 1.0 and isinstance(value, (np.floating, float)):
        return img

    uint8_max = np.iinfo(np.uint8).max
    uint16_max = np.iinfo(np.uint16).max
    uint32_max = np.iinfo(np.uint32).max

    img = img.astype(float)
    if force_dtype is not None:
        dtype_max = np.iinfo(force_dtype).max
        img = img/dtype_max
    elif input_img_dtype == np.uint8:
        # Input image is 8-bit
        img = img/uint8_max
    elif input_img_dtype == np.uint16:
        # Input image is 16-bit
        img = img/uint16_max    
    elif input_img_dtype == np.uint32:
        # Input image is 32-bit
        img = img/uint32_max
    elif force_missing_dtype is not None:
        img = img.astype(force_dtype)
    elif img_max <= uint8_max:
        # Input image is probably 8-bit
        if warn:
            _warnings.warn_image_overflow_dtype(input_img_dtype, img_max, '8-bit')
        img = img/uint8_max
    elif img_max <= uint16_max:
        # Input image is probably 16-bit
        if warn:
            _warnings.warn_image_overflow_dtype(input_img_dtype, img_max, '16-bit')
        img = img/uint16_max
    elif img_max <= uint32_max:
        # Input image is probably 32-bit
        if warn:
            _warnings.warn_image_overflow_dtype(input_img_dtype, img_max, '32-bit')
        img = img/uint32_max
    else:
        # Input image is a non-supported data type
        raise TypeError(
            f'The maximum value in the image is {img_max} which is greater than the '
            f'maximum value supported of {uint32_max} (32-bit). '
            'Please consider converting your images to 32-bit or 16-bit first.'
        )
    return img

def float_img_to_dtype(img, dtype):
    if img.dtype == dtype:
        return img
    
    img_max = img.max()
    if img_max > 1.0:
        raise TypeError(
            'Images of float data type with values greater than 1.0 cannot '
            f'be safely casted to {dtype}. '
            f'The max value of the input image is {img_max:.3f}'
        )
    
    img_min = img.min()
    if img_min < -1.0:
        raise TypeError(
            'Images of float data type with values smaller than -1.0 cannot '
            f'be safely casted to {dtype}.'
            f'The minumum value of the input image is {img_min:.3f}'
        )

    if dtype == np.uint8:
        return skimage.img_as_ubyte(img)
    
    if dtype == np.uint16:
        return skimage.img_as_uint(img)
    
    if dtype == np.float32:
        return img.astype(np.float32)
    
    if dtype == np.float64:
        return img.astype(np.float64)
    
    raise TypeError(
        f'Invalid output data type `{dtype}`. '
        'Valid output data types are `np.uint8` and `np.uint16`'
    )

def convert_to_dtype(data: np.ndarray, dtype):
    dtype = np.dtype(dtype)

    if data.dtype == dtype:
        return data
    
    val = data[tuple([0]*data.ndim)]
    if isinstance(val, (np.floating, float)):
        data = float_img_to_dtype(data, dtype)
    elif dtype == np.uint8:
        data = np.round(img_to_float(data)*255).astype(np.uint8)
    elif dtype == np.uint16:
        data = np.round(img_to_float(data)*65535).astype(np.uint16)
    else:
        raise TypeError(
            f'Invalid output data type `{dtype}`. '
            'Valid data types are floating-point format, `np.uint8` '
            'and `np.uint16`'
        )
    return data


def synthetic_image_geneator(size=(512,512), f_x=1, f_y=1):
    Y, X = size
    x = np.linspace(0, 10, Y)
    y = np.linspace(0, 10, X)
    xx, yy = np.meshgrid(x, y)
    img = np.sin(f_x*xx)*np.cos(f_y*yy)
    return img


def iterate_along_axes(arr, axes, arr_ndim=None):
    if arr_ndim is None:
        arr_ndim = arr.ndim
    axes = list(axes)
    front_axes = axes + [i for i in range(arr_ndim) if i not in axes]
    arr_moved = np.moveaxis(arr, front_axes, range(arr_ndim))
    iter_shape = arr_moved.shape[:len(axes)]
    for idx in np.ndindex(iter_shape):
        # Build the index for the original array
        full_idx = [slice(None)] * arr_ndim
        for axis, i in zip(axes, idx):
            full_idx[axis] = i
        yield tuple(full_idx)

def get_input_output_mapper(
        input_shape: Tuple[int],
        iterate_axes: Tuple[int],
        output_shape: Tuple[int],
        output_axes: Tuple[int],
) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
    """Creates list of tuples with the input and output indices

    Parameters
    ----------
    input_shape : Tuple[int]
        Shape of the input array
    iterate_axes : Tuple[int]
        Axes to iterate over
    output_shape : Tuple[int]
        Shape of the output array
    output_axes : Tuple[int]
        Axes of the output array
    """
    assert len(iterate_axes) == len(output_axes)

    iterate_shape = tuple(input_shape[axis] for axis in iterate_axes)
    mapper = []

    for idx_vals in itertools.product(*[range(s) for s in iterate_shape]):
        # Build full input index
        input_index = [slice(None)] * len(input_shape)
        for axis in iterate_axes:
            i = iterate_axes.index(axis)
            input_index[axis] = idx_vals[i]

        # Build full output index
        output_index = [slice(None)] * len(output_shape)
        for axis in output_axes:
            i = output_axes.index(axis)
            output_index[axis] = idx_vals[i]

        input_index = tuple(input_index)
        output_index = tuple(output_index)

        mapper.append((input_index, output_index))

    return mapper


def get_obj_by_label(rp, target_label):
    """
    Returns the object with the specified label from the given list of objects.

    Parameters
    ----------
    rp : list
        The list of objects to search through.
    target_label : str
        The label of the object to find.

    Returns
    -------
    object
        The object with the specified label, or None if not found.
    """
    for obj in rp:
        if obj.label == target_label:
            return obj
    return None
