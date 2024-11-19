from tqdm import tqdm

import numpy as np
import pandas as pd
import math

try:
    from cupyx.scipy.ndimage import gaussian_filter as gpu_gaussian_filter
    import cupy as cp
    CUPY_INSTALLED = True
except Exception as e:
    CUPY_INSTALLED = False

import skimage.morphology
import skimage.filters
import skimage.exposure
from skimage.util import img_as_ubyte

from . import error_up_str
from . import types

SQRT_2 = math.sqrt(2)

def remove_hot_pixels(image, logger_func=print, progress=True):
    is_3D = image.ndim == 3
    if is_3D:
        if progress:
            pbar = tqdm(total=len(image), ncols=100)
        filtered = image.copy()
        for z, img in enumerate(image):
            filtered[z] = skimage.morphology.opening(img)
            if progress:
                pbar.update()
        if progress:
            pbar.close()
    else:
        filtered[z] = skimage.morphology.opening(img)
    return filtered

def gaussian_filter(
        image, sigma: types.Vector, 
        use_gpu=False, logger_func=print
    ):
    """Multi-dimensional Gaussian filter

    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale or color) to filter.
    sigma : types.Vector
        Standard deviation for Gaussian kernel. The standard deviations of the 
        Gaussian filter are given for each axis as a sequence, or as a single 
        number, in which case it is equal for all axes.
    use_gpu : bool, optional
        If True, uses `cupy` instead of `skimage.filters.gaussian`. 
        Default is False
    logger_func : callable, optional
        Function used to log information. Default is print

    Returns
    -------
    filtered_image : numpy.ndarray
        The filtered image

    See also
    --------
    Wikipedia link: `Gaussian blur <https://en.wikipedia.org/wiki/Gaussian_blur>`_
    """    
    try:
        if len(sigma) > 1 and sigma[0] == 0:
            return image
    except Exception as err:
        pass
    
    try:
        if sigma == 0:
            return image
    except Exception as err:
        pass
    
    try:
        if len(sigma) == 0:
            sigma = sigma[0]
    except Exception as err:
        pass
    
    if CUPY_INSTALLED and use_gpu:
        try:
            image = cp.array(image, dtype=float)
            filtered = gpu_gaussian_filter(image, sigma)
            filtered = cp.asnumpy(filtered)
        except Exception as err:
            logger_func('*'*100)
            logger_func(err)
            logger_func(
                '[WARNING]: GPU acceleration of the gaussian filter failed. '
                f'Using CPU...{error_up_str}'
            )
            filtered = skimage.filters.gaussian(image, sigma=sigma)
    else:
        filtered = skimage.filters.gaussian(image, sigma=sigma)
    return filtered

def ridge_filter(image, sigmas: types.Vector):
    input_shape = image.shape
    filtered = skimage.filters.sato(
        np.squeeze(image), sigmas=sigmas, black_ridges=False
    ).reshape(input_shape)
    return filtered

def spot_detector_filter(
        image, spots_zyx_radii_pxl: types.Vector, use_gpu=False, 
        logger_func=print, lab=None
    ):
    spots_zyx_radii_pxl = np.array(spots_zyx_radii_pxl)
    if image.ndim == 2 and len(spots_zyx_radii_pxl) == 3:
        spots_zyx_radii_pxl = spots_zyx_radii_pxl[1:]
    
    sigma1 = spots_zyx_radii_pxl/(1+SQRT_2)
    
    if 0 in sigma1:
        raise TypeError(
            f'Sharpening filter input sigmas cannot be 0. `zyx_sigma1 = {sigma1}`'
        )
        
    blurred1 = gaussian_filter(
        image, sigma1, use_gpu=use_gpu, logger_func=logger_func
    )
    
    sigma2 = SQRT_2*sigma1
    blurred2 = gaussian_filter(
        image, sigma2, use_gpu=use_gpu, logger_func=logger_func
    )
    
    sharpened = blurred1 - blurred2
    
    if lab is None:
        out_range = (image.min(), image.max())
        in_range = 'image'
    else:
        lab_mask = lab > 0
        img_masked = image[lab_mask]
        out_range = (img_masked.min(), img_masked.max())
        sharp_img_masked = sharpened[lab_mask]
        in_range = (sharp_img_masked.min(), sharp_img_masked.max())
    sharp_rescaled = skimage.exposure.rescale_intensity(
        sharpened, in_range=in_range, out_range=out_range
    )
    
    return sharp_rescaled

def correct_illumination(
        image, 
        block_size=60, 
        rescale_illumination=True,
        approximate_object_diameter=10,
        background_threshold=2.0,
        apply_gaussian_filter=True
    ):
    """Correct illumination of an image. Based on CorrectIlluminationCalculate 
    from Cell profiler

    Parameters
    ----------
    image : np.ndarray
        2D image to correct
    block_size : int, optional
        block size for which to calculate the background illumination. Default is 60
    rescale_illumination : bool, optional
        if illumination should be rescaled with cv2.normalize (alpha=0, beta=1). Default is True
    approximate_object_diameter : int, optional
        approximate object diameter for gaussian_filter. Default is 10
    background_threshold : float, optional
        threshold to be used to determine the background. Default is 2.0
    apply_gaussian_filter : bool, optional
        if gaussian_filter should be applied to the illumination_function. Default is True

    Returns
    -------
    np.ndarray
        corrected 2D image
    """
    
    # Step 1: Compute the Illumination function for each image individually
    illumination_function = np.zeros_like(image)

    # Divide image into blocks and compute average intensity per block
    footprint = np.ones((block_size, block_size))
    illumination_function = skimage.filters.rank.mean(
        image, footprint=footprint
    )

    # Step 2: Apply Gaussian smoothing based on the approximate object diameter
    if apply_gaussian_filter == True:
        illumination_function = skimage.filters.gaussian(
            illumination_function, sigma=approximate_object_diameter / 2
        )

    # Step 3: Thresholding background using background_threshold
    illum_mask = illumination_function < background_threshold
    illumination_function[illum_mask] = background_threshold

    # Step 4: Optionally rescale the illumination function
    if rescale_illumination:
        illumination_function = skimage.exposure.rescale_intensity(
            illumination_function, out_range=(0.0, 1.0)
        )

    # Step 5: Downsample and smooth illumination function if necessary
    # illumination_function_resampled = zoom(illumination_function, resampling_factor)

    # Apply the illumination correction to the input image
    # Avoid division by zero
    corrected_image = image / (illumination_function + 1e-6)  
    
    # Clip values to ensure they are in [0, 1] range
    corrected_image = np.clip(corrected_image, 0, 1)  

    # Convert back to 8-bit for visualization or saving
    corrected_image_8bit = img_as_ubyte(corrected_image)

    return corrected_image_8bit

def enhance_speckles(img, radius=15):
    """Enhance speckles in an image using white_tophat. Based on 
    EnhanceOrSuppressFeatures from Cell profiler with 'Feature type: Speckles'

    Parameters
    ----------
    image : np.ndarray
        2D image to enhance
    radius : int, optional
        Radius to use for the enhancer. Will suppress objects smaller than this 
        radius. Default is 15

    Returns
    -------
    np.ndarray
        corrected 2D image
    """
    footprint = skimage.morphology.disk(radius)
    output_image = skimage.morphology.white_tophat(img, footprint=footprint)
    return output_image

def fucci_filter(
        image, 
        block_size=60, 
        rescale_illumination=True,
        approximate_object_diameter=10,
        background_threshold=2.0,
        apply_gaussian_filter=True,
        speckle_radius=15
    ):
    """Applies 

    Parameters
    ----------
    image : (Y, X) numpy.ndarray
        2D image to correct
    block_size : int, optional
        Block size for which to calculate the background illumination.
        Default is 60
    rescale_illumination : bool, optional
        If illumination should be rescaled with cv2.normalize (alpha=0, beta=1). 
        Default is True
    approximate_object_diameter : int, optional
        Approximate object diameter for gaussian_filter.Default is 10
    background_threshold : float, optional
        Threshold to be used to determine the background. Default is 2.0
    apply_gaussian_filter : bool, optional
        If gaussian_filter should be applied to the illumination_function. 
        Default is True
    speckle_radius : int, optional
        Radius to use for the enhancer. Will suppress objects smaller than this 
        radius. Default is 15

    Returns
    -------
    (Y, X) numpy.ndarray
        Filtered image
    """    
    corrected_img = correct_illumination(
        image, 
        block_size=block_size, 
        rescale_illumination=rescale_illumination,
        approximate_object_diameter=approximate_object_diameter,
        background_threshold=background_threshold,
        apply_gaussian_filter=apply_gaussian_filter,
    )
    corrected_img = enhance_speckles(corrected_img, radius=speckle_radius)
    return corrected_img