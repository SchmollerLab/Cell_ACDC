import numpy as np
import os
from PIL import Image
import scipy.io as sio
from nd2reader import ND2Reader

try:
    from . import myutils
except Exception as e:
    import myutils

### Helper Functions

def get_metadata_from_nd2(path, drop_z_coords=True):
    """
    function to extract the meta data of a .nd2 file.
    Reads nd2 file, extracts meta data and returns it in form of a dictionary
    Parameters
    ----------
    path: String
        path of the .nd2 file
    drop_z_coords: bool
        Standard True, set to False if convocal data is used
    Returns
    -------
    meta_data: dict
        dictionary containing the meta data of the .nd2 file
    """
    with ND2Reader(path) as images:
        meta_data = images.metadata
        if drop_z_coords:
            del meta_data['z_coordinates']
        return meta_data


def get_index_from_pos_ch_t(pos, ch, t, meta_data, parse_from_single_pos_file=False):
    """
    function to calculate the ND2Reader index of a particular frame.
    The function get_frame_2D does (afaik) not
    work for non-convocal data. A translation of position, channel,
    and time frame into the one-dimensional index of the flattened
    data is therefore needed and performed in this function.
    Parameters
    ----------
    pos: int
        position (/field of view) of the frame which should be extracted
    ch: int
        Channel of the frame which should be extracted
    t: int
        frame id of the frame which should be extracted
    meta_data: dict
        dictionary containing the .nd2 file's meta data as extracted
        by the function get_metadata_from_nd2()
    parse_from_single_pos_file: bool
        if True, the number of positions from the meta data is ignored.
    Returns
    -------
    idx: int
        index of the frame which can be used in parser.get_image()
        of the module ND2Reader. This index identifies the frame
        with the configured position, channel, and point of time.
    """
    n_pos = len(meta_data['fields_of_view'])
    n_channels = len(meta_data['channels'])
    n_frames = meta_data['num_frames']
    if parse_from_single_pos_file:
        idx = t * n_channels + ch
    else:
        idx = t * (n_pos * n_channels) + pos * n_channels + ch
    return idx


### Data handlers


def get_npy_from_nd2(path, positions='all', channels='all', frames='all',\
                     cut_images=0, convocal=False, parse_from_single_pos_file=False):
    """
    function to load the specified frames into a numpy array.
    The function takes a .nd2 file's path and ranges of positions,
    channels, and frames and returns a numpy array containing the
    image data of the specified input. In the course of loading the
    data, an manual cut can also be performed to save disk space
    and crop the data to relevant regions.
    Parameters
    ----------
    path: str
        path of the .nd2 file
    positions: list or str 'all'
        default 'all'. List of positions which should be extracted
        from the .nd2 file. If set to 'all', all available positions
        are extracted.
    channels: list or str 'all'
        default 'all'. List of channels which should be extracted
        from the .nd2 file. If set to 'all', all available channels
        are extracted.
    frames: list or str 'all'
        default 'all'. List of time frames which should be extracted
        from the .nd2 file. If set to 'all', all available frames
        are extracted.
    cut_images: int
        default 0. Number of pixels which are cropped away at both sides
        of the x and y dimension.
    convocal: bool
        default False. Specifying if the data is convocal. This is
        needed to extract the correct index in the flattened data
        using the function get_index_from_pos_ch_t().
    Returns
    -------
    img_array: np.array
        numpy array of dimension(pos, ch, t, x, y) containing the image
        data of the specified positions, channels and frames.
    """
    # only keep z_coords in case of convocal data
    drop_z_coords = bool(1-convocal)
    meta_data = get_metadata_from_nd2(path, drop_z_coords=drop_z_coords)
    # translate position, channel and frames arguments into lists
    if positions == 'all':
        pos_list = list(meta_data['fields_of_view'])
    else:
        pos_list = positions
    if channels == 'all':
        ch_list = list(range(len(meta_data['channels'])))
    else:
        ch_list = channels
    if frames == 'all':
        t_list = list(range(meta_data['num_frames']))
    else:
        t_list = frames
    # build image_array
    ## determine dimensions of final array
    if cut_images==0:
        x_dim, y_dim = meta_data['width'], meta_data['height']
    else:
        x_dim = meta_data['width'] - 2*cut_images
        y_dim = meta_data['height'] - 2*cut_images
    img_array_shape = (len(pos_list), len(ch_list), len(t_list), y_dim, x_dim)
    img_array = np.zeros(img_array_shape)
    ## load each position, channel, frame and save to array
    with ND2Reader(path) as images:
        parser = images.parser
        for pos_idx in range(len(pos_list)):
            for ch_idx in range(len(ch_list)):
                for t_idx in range(len(t_list)):
                    img_idx = get_index_from_pos_ch_t(pos_list[pos_idx], ch_list[ch_idx],\
                                                      t_list[t_idx],meta_data,\
                                                          parse_from_single_pos_file)
                    if cut_images == 0:
                        img_array[pos_idx, ch_idx, t_idx] =\
                        parser.get_image(img_idx)
                    else:
                        img_array[pos_idx, ch_idx, t_idx] =\
                        parser.get_image(img_idx)[cut_images:-cut_images,\
                                                  cut_images:-cut_images]
    return img_array



def load_series_of_tifs(folder, channel=None):
    """
    function to load images/masks into a numpy array from a folder containing
    .tif images.

    Parameters
    ----------
    folder: str
        path to folder containing the .tif images relative to home dir
    positions: str
        if given, filter for images containing this string to filter
        for a certain channel.

    Returns
    -------
    all_im: np.array
        numpy array of dim (n_images, shape_images) containing all images
        of the folder

    """
    if channel:
        files = [f for f in sorted(myutils.listdir(folder)) if channel in f]
    else:
        files = sorted(myutils.listdir(folder))
    all_im = []
    for f in files:
        im = Image.open(f'{folder}{f}')
        all_im.append(np.array(im))
    all_im = np.array(all_im)
    return all_im


def load_masks_from_matlab(mask_filename, gt_folder):
    """
    function to load masks into a numpy array from a matlab file (.mat)

    Parameters
    ----------
    mask_filename: str
        filename of the matlab file which should be loaded
    gt_folder: str
        folder containing the matlab file

    Returns
    -------
    all_masks: np.array
        numpy array of dim (n_frames, shape_masks) containing all masks
        of the mat file

    """
    mask_fn_full = f'{gt_folder}{mask_filename}'
    mat = sio.loadmat(mask_fn_full)
    masks = mat['all_obj']['cells'][0][0]
    all_masks = np.moveaxis(masks, -1, 0)
    return all_masks
