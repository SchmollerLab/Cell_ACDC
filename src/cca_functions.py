import numpy as np
from skimage.measure import label, regionprops, regionprops_table
import pandas as pd
from tifffile import imread
import os


def _auto_rescale_intensity(img, perc=0.01, clip_min=False):
    """
    function to clip outliers to the given percentiles and scale afterwards.
    
    use skimage.exposure.rescale_intensity for clipping all images in the
    given percentiles to the percentile border.
    
    Parameters
    ----------
    img: np.array
        image to eliminate outliers from
    perc: float
        percentage of pixels which are considered as outliers. 
        Example: given 0.01, the lowest and highest 1% of pixels are considered
        as outliers and clipped to the 1- and 99-percentile respectively.
    
    Returns
    -------
    scaled_img: np.array
        image, where outlier pixels are set to 1- and 99-percentiles and which is 
        scaled to [0,1] afterwards
    """
    if perc > 0:
        vmin, vmax = np.percentile(img, q=(perc, 100-perc))
        clip_min_indices = img < vmin
        clip_max_indices = img > vmax
        if clip_min:
            img[clip_min_indices] = vmin
        img[clip_max_indices] = vmax
    else:
        vmin, vmax = img.min(), img.max()
    scaled_img = (img-vmin)/(vmax-vmin)
    return scaled_img


def load_files(file_dir, channels):
    """
    Function to load files of all given channels and the corresponding segmentation masks.
    Check first if aligned files are available and use them if so.
    """
    file_list = os.listdir(file_dir)
    no_of_aligned_files = sum(['aligned' in f for f in file_list])
    if no_of_aligned_files == len(channels):
        channel_files = []
        for channel in channels:
            for f in file_list:
                if f'{channel}_aligned' in f:
                    # assume aligned files to be .npy
                    channel_files.append(np.load(os.path.join(file_dir,f)))
                    break
    elif no_of_aligned_files == 0:
        channel_files = []
        for channel in channels:
            for f in file_list:
                if channel in f:
                    # assume non-aligned files to be .tif
                    channel_files.append(imread(os.path.join(file_dir,f)))
                    break
    else:
        print('Make sure that you have aligned files either for all channels or for none of them')
        raise FileNotFoundError
    # append segmentation file
    for f in file_list:
        if '_segm' in f:
            # assume aligned files to be .npy
            channel_files.append(np.load(os.path.join(file_dir,f)))
            break
    # append cc-data at the end
    for f in file_list:
        if 'cc_stage' in f:
            # assume aligned files to be .npy
            channel_files.append(pd.read_csv(os.path.join(file_dir,f)))
            break
    return tuple(channel_files)




def calculate_rp_df(input_sequence, label_input=False):
    if label_input:
        #generate labeled video only when input is not labeled yet
        labeled_video = label(input_sequence)
    else:
        labeled_video = input_sequence.copy()
    # calculate rp's for rings
    t_df = pd.DataFrame()
    props = ('label', 'area', 'convex_area', 'filled_area','major_axis_length',
             'minor_axis_length', 'orientation', 'perimeter', 'centroid', 'solidity')
    rename_dict = {'label':'Cell_ID', 'centroid-0':'centroid_y', 'centroid-1':'centroid_x'}
    for t, img in enumerate(labeled_video):
        # build time-dependent dataframes for further use (later for cca)
        if img.max() > 0:
            t_rp = pd.DataFrame(regionprops_table(img.astype(int), properties=props)).rename(columns=rename_dict)
            t_rp['frame_i'] = t
            # determine id's which are falsely merged by 3D-labeling
            for r_id in t_rp.Cell_ID.unique():
                bin_label = label((img==r_id).astype(int))
                t_rp.loc[t_rp['Cell_ID']==r_id, '2d_label_count'] = bin_label.max()
            t_df = t_df.append(t_rp, ignore_index=True)
    # calculate global features by grouping
    grouped_df = t_df.groupby('Cell_ID').agg(
        min_t=('frame_i', min), 
        max_t=('frame_i', max),
        lifespan=('frame_i', lambda x: max(x)-min(x)+1)
    ).reset_index()
    merged_df = t_df.merge(grouped_df, how='left', on='Cell_ID')
    # calculate further indicators based on merged data
    merged_df['age'] = merged_df['frame_i'] - merged_df['min_t'] + 1
    merged_df['frames_till_gone'] = merged_df['max_t'] - merged_df['frame_i']
    merged_df['elongation'] = merged_df['major_axis_length']/merged_df['minor_axis_length']
    return merged_df


def calculate_flu_signal(seg_mask, channel_data, channels, cc_data):
    """
    function to calculate sum and scaled sum of fluorescence signal per frame and cell.
    channel_data is a tuple of (t,y,x) arrays, one for each channel.
    channels are the name of the channels in the tuple.
    cc_data the output of acdc.
    """
    max_frame = cc_data.frame_i.max()
    df = pd.DataFrame(columns=['frame_i', 'Cell_ID'])
    for cell_id in cc_data.Cell_ID.unique():
        temp_df = pd.DataFrame(columns=['frame_i', 'Cell_ID'])
        times = range(max_frame+1)
        temp_df['frame_i'] = times; temp_df['Cell_ID'] = cell_id
        index_array = (seg_mask[:max_frame+1] == cell_id).astype(int)
        for c_idx, c_array in enumerate(channel_data):
            signal = np.sum(index_array*c_array[:max_frame+1], axis=(1,2))
            temp_df[f'{channels[c_idx]}_signal'] = signal
            c_array_scaled = _auto_rescale_intensity(c_array)
            signal_scaled = np.sum(index_array*c_array_scaled[:max_frame+1], axis=(1,2))
            temp_df[f'{channels[c_idx]}_signal_scaled'] = signal_scaled
        df = df.append(temp_df, ignore_index=True)
    signal_indices = np.array(['signal' in col for col in df.columns])
    keep_rows = df.loc[:,signal_indices].sum(axis=1)>0
    df = df[keep_rows]
    df = df.sort_values(['frame_i', 'Cell_ID']).reset_index(drop=True)
    return df


def rename_columns(cc_data):
    rename_dict = {
        'Cell cycle stage': 'cell_cycle_stage',
        '# of cycles': 'generation_num',
        "Relative's ID": 'relative_ID',
        'Relationship': 'relationship',
        'Emerg_frame_i': 'emerg_frame_i',
        'Division_frame_i': 'division_frame_i',
        'Discard': 'is_cell_excluded'
    }
    cc_data.columns = [rename_dict.get(col, col) for col in cc_data.columns] 
    return cc_data