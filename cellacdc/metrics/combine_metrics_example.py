import numpy as np

def combine_metrics_example(
        signal, autoBkgr, dataPrepBkgr, objectRp, metrics_values, image, lab,
        other_channel_foregr_img, correct_with_bkgr=False, which_bkgr='auto', 
        isSegm3D=False
    ):
    """Shows how to combine multiple metrics in a custom function.

    Parameters
    ----------
    signal : numpy 1D array
        This is a numpy array with all the intensities of the signal
        from each single segmented object.
    autoBkgr : single numeric value
        Median of all the background pixels (i.e. pixels with value 0 in the
        segmentation mask). Pass None if background correction with
        this value is not needed.
    dataPrepBkgr : single numeric value
        Median of all the pixels inside the background ROIs added during the
        data prep step (Cell-ACDC module 1).
        Pass None if background correction with this vaue is not needed.
    objectRp : skimage.measure.RegionProperties class
        Refer to `skimage.measure.regionprops` for more information
        on the available region properties.
    metrics_values : dict
        Dictionary of metrics values of the specific segmented object 
        (i.e., cell). You can access these values with the name of the 
        specific metric (i.e., column name in the acdc_output.csv file)
        Examples: 
            - mCitrine_mean = metrics_values['mCitrine_mean']
            - _mean_key = [key for key in metrics_values if key.endswith('_mean')][0] 
              _mean = metrics_values[_mean_key]
    image : numpy array
        Image signal being analysed (if time-lapse this is the current frame)
    lab : numpy array
        Segmentation mask of `image`
    other_channel_foregr_img : dict
        Dictionary with a single key another loaded channel name and values 
        the corresponding channel signal. Cell-ACDC will run this function 
        for as many other channles were loaded. Do not include in your custom 
        function if you don't need it.
    correct_with_bkgr : boolean
        Pass True if you need background correction.
    which_bkgr : string
        which_bkgr='auto' for correction with autoBkgr or
        which_bkgr='dataPrep' for correction with dataPrepBkgr

    Returns
    -------
    float
        Numerical value of the computed metric
    
    Notes
    -----
    
    1. The function must have the same name as the Python file containing it 
       (e.g., this file is called CV.py and the function is called CV)

    2. The function must return a single number. You will need
       one .py for each additional custom metric.
    
    This implementation shows how to compute the concentration for all the 
    available channels. Concentration is calculated as the ratio between 
    columns ending with `_amount_autoBkgr` and `cell_vol_fl`. 
    """
    _amount_key = [key for key in metrics_values if key.endswith('_amount_autoBkgr')][0]
    _amount = metrics_values[_amount_key]
    cell_vol_fl = metrics_values['cell_vol_fl']
    _concentration = _amount/cell_vol_fl

    return _concentration
