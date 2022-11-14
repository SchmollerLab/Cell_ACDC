import numpy as np

def combine_metrics_example(
        signal, autoBkgr, dataPrepBkgr, objectRp, metrics_values, image, lab,
        correct_with_bkgr=False, which_bkgr='auto', isSegm3D=False
    ):
    """Function used to show how to combine multiple metrics in a custom function.

    NOTE 1: Make sure to call the function with the same name as the Python file
    containing this function (e.g., this file is called CV.py and the function
    is called CV)

    NOTE 2: Make sure that this function returns a single number. You will need
    one .py for each additional custom metric.

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
    objectRp: skimage.measure.RegionProperties class
        Refer to `skimage.measure.regionprops` for more information
        on the available region properties.
    metrics_values: dict
        Dictionary of metrics values of the specific segmented object 
        (i.e., cell). You can access these values with the name of the 
        specific metric (i.e., column name in the acdc_output.csv file)
        Examples: 
            - mCitrine_mean = metrics_values['mCitrine_mean']
            - _mean_key = [key for key in metrics_values if key.endswith('_mean')][0] 
              _mean = metrics_values[_mean_key]
    image: numpy array
        Image signal being analysed (if time-lapse this is the current frame)
    lab: numpy array
        Segmentation mask of `image`
    correct_with_bkgr : boolean
        Pass True if you need background correction.
    which_bkgr : string
        which_bkgr='auto' for correction with autoBkgr or
        which_bkgr='dataPrep' for correction with dataPrepBkgr

    Returns
    -------
    float
        Numerical value of the computed metric

    """
    # Here we show the code required to compute the concentration for ALL channels
    # by taking the ratio between the two metrics 'amount' and 'cell_vol_fl'.
    # Since there are different columns for amount, (i.e. 'mCitrine_amount_autoBkgr')
    # we can take the key that ends with '_amount_autoBkgr'.
    _amount_key = [key for key in metrics_values if key.endswith('_amount_autoBkgr')][0]
    _amount = metrics_values[_amount_key]
    cell_vol_fl = metrics_values['cell_vol_fl']
    _concentration = _amount/cell_vol_fl

    return _concentration
