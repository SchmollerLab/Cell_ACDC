
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 18:54:10 2019

"""
import os
import sys
import numpy as np
import skimage
from skimage import io
import skimage.transform as trans

from .model import unet

def determine_path_weights():
    script_dirname = os.path.dirname(os.path.realpath(__file__))
    main_path = os.path.dirname(os.path.dirname(os.path.dirname(script_dirname)))
    model_path = os.path.join(main_path, 'models', 'YeaZ_model')

    if getattr(sys, 'frozen', False):
        path_weights  = os.path.join(sys._MEIPASS, 'unet/')
    else:
        path_weights = model_path
    return path_weights

def create_directory_if_not_exists(path):
    """
    Create in the file system a new directory if it doesn't exist yet.
    Param:
        path: the path of the new directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


def threshold(im, thresh_val=None):
    """
    Binarize an image with a threshold given by the user, or if the threshold is None, calculate the better threshold with isodata
    Param:
        im: a numpy array image (numpy array)
        thresh_val: the value of the threshold (feature to select threshold was asked by the lab)
    Return:
        bi: threshold given by the user (numpy array)
    """
    im2 = im.copy()
    if thresh_val == None:
        thresh_val = skimage.filters.threshold_isodata(im2)
    bi = im2
    bi[bi > thresh_val] = 255
    bi[bi <= thresh_val] = 0
    return bi


def prediction(im, is_pc, path_weights):
    """
    Calculate the prediction of the label corresponding to image im
    Param:
        im: a numpy array image (numpy array), with max size 2048x2048
    Return:
        res: the predicted distribution of probability of the labels (numpy array)
    """
    # pad with zeros such that is divisible by 16
    (nrow, ncol) = im.shape
    row_add = 16-nrow%16
    col_add = 16-ncol%16
    padded = np.pad(im, ((0, row_add), (0, col_add)), 'constant')

    # WHOLE CELL PREDICTION
    model = unet(pretrained_weights = None,
                 input_size = (None,None,1))

    if is_pc:
        path = os.path.join(
            path_weights,
            'unet_weights_batchsize_25_Nepochs_100_SJR0_10.hdf5'
        )
    else:
        path = os.path.join(
            path_weights,
            'weights_budding_BF_multilab_0_1.hdf5'
        )

    if not os.path.exists(path):
        raise ValueError(f'Weights file not found in {path}')

    model.load_weights(path)

    results = model.predict(padded[np.newaxis,:,:,np.newaxis], batch_size=1)

    res = results[0,:,:,0]
    return res[:nrow, :ncol]


def batch_prediction(im_stack, is_pc, path_weights, batch_size=1):
    """
    calculate the prediction for a stack of images.
    Param:
        im: a numpy array image (numpy array), with max size 2048x2048
    Return:
        res: the predicted distribution of probability of the labels (numpy array)
    """
    # pad with zeros such that is divisible by 16
    (nrow, ncol) = im_stack[0].shape
    row_add = 16 - nrow % 16
    col_add = 16 - ncol % 16
    im_stack_padded = []
    for im in im_stack:
        padded = np.pad(im, ((0, row_add), (0, col_add)), mode='constant')
        im_stack_padded.append(padded)
    im_stack_padded = np.array(im_stack_padded)
    # WHOLE CELL PREDICTION
    model = unet(pretrained_weights=None,
                 input_size=(None, None, 1))

    if is_pc:
        path = os.path.join(path_weights, 'unet_weights_batchsize_25_Nepochs_100_SJR0_10.hdf5')
    else:
        path = os.path.join(path_weights, 'unet_weights_BF_batchsize_25_Nepochs_100_SJR_0_1.hdf5')

    if not os.path.exists(path):
        raise ValueError(
            'Weights file not found! Download them from the link '
            f'below and place them into {path_weights}.\n'
            'Link: https://drive.google.com/file/d/1CO7uF-werl9y8s3Fel0cVjRHCdXRf2Ly/view?usp=sharing')

    model.load_weights(path)

    results = model.predict(
        im_stack_padded[:, :, :, np.newaxis], batch_size=1, verbose=1
    )

    res = results[:, :, :, 0]
    return res[:, :nrow, :ncol]
