import numpy as np

from natsort import natsorted

from cellacdc import printl

# If you want to calculate the metric for each channel, set this to True. 
# If you want to calculate the metric only once after metrics for all channels 
# have been computed, set this to False.
CALCULATE_FOR_EACH_CHANNEL = False

def channel_indipendent_metric(
        all_channels_signals, 
        all_channels_autoBkgr, 
        all_channels_dataPrepBkgr, 
        objectRp, 
        metrics_values, 
        images, 
        lab, 
        isSegm3D=False
    ):
    """Shows how to combine multiple metrics in a channel-indipendent manner 
    using a custom function.

    Parameters
    ----------
    all_channels_signals : dictionary of numpy 1D arrays
        Dictionary with channel names as keys and the numpy array as value 
        with all the intensities of the signal from each single segmented object.
    all_channels_autoBkgr : dictionary of single numeric value
        Dictionary with channel names as keys and as value the median of all 
        the background pixels (i.e. pixels with value 0 in the
        segmentation mask). Pass None if background correction with
        this value is not needed.
    all_channels_dataPrepBkgr : dictionary of single numeric value
        Dictionary with channel names as keys and as value the median of all 
        the pixels inside the background ROIs added during the
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
    images : dictionary of numpy array
        Dictionary with channel names as keys and the corresponding image 
        signal as value
    lab : numpy array
        Segmentation mask of `image`

    Returns
    -------
    float
        Numerical value of the computed metric
    
    Notes
    -----
    
    1. The function must have the same name as the Python file containing it 
       (e.g., if this file is called CV.py the function must be called CV)

    2. The function must return a single number. You will need
       one .py for each additional custom metric.
    
    This implementation shows how to compute the ratio of the amount between 
    the first two channels (alphabetically) divided by the cell_vol_fl. 
    """
    
    channels = list(all_channels_signals.keys())
    channels = natsorted(channels)
    channel_1 = channels[0]
    
    try:
        channel_2 = channels[1]
    except IndexError:
        # Only one channel loaded. Returning 0.
        return 0.0
    
    ch1_amount_key = [
        key for key in metrics_values 
        if key.startswith(f'{channel_1}_amount_autoBkgr')][0]
    
    ch1_amount = metrics_values[ch1_amount_key]
    
    ch2_amount_key = [
        key for key in metrics_values 
        if key.startswith(f'{channel_2}_amount_autoBkgr')][0]
    ch2_amount = metrics_values[ch2_amount_key]
    
    cell_vol_fl = metrics_values['cell_vol_fl']
    
    amount_ratio = ch1_amount / ch2_amount
    
    totally_useless_metric = amount_ratio/cell_vol_fl

    return totally_useless_metric
