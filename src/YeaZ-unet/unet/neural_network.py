
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 18:54:10 2019

"""
import os
import sys
from model import unet
import numpy as np
import skimage
from skimage import io
import skimage.transform as trans

script_dirname = os.path.dirname(os.path.realpath(__file__))
main_path = os.path.dirname(os.path.dirname(os.path.dirname(script_dirname)))
model_path = os.path.join(main_path, 'model')

if getattr(sys, 'frozen', False):
    path_weights  = os.path.join(sys._MEIPASS, 'unet/')
else:
    path_weights = model_path

def create_directory_if_not_exists(path):
    """
    Create in the file system a new directory if it doesn't exist yet.
    Param:
        path: the path of the new directory
    """
    if not os.path.exists(path):
        os.makedirs(path)


def threshold(im,th = None):
    """
    Binarize an image with a threshold given by the user, or if the threshold is None, calculate the better threshold with isodata
    Param:
        im: a numpy array image (numpy array)
        th: the value of the threshold (feature to select threshold was asked by the lab)
    Return:
        bi: threshold given by the user (numpy array)
    """
    im2 = im.copy()
    if th == None:
        th = skimage.filters.threshold_isodata(im2)
    bi = im2
    bi[bi > th] = 255
    bi[bi <= th] = 0
    return bi


def prediction(im, is_pc):
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
    padded = np.pad(im, ((0, row_add), (0, col_add)))

    # WHOLE CELL PREDICTION
    model = unet(pretrained_weights = None,
                 input_size = (None,None,1))

    if is_pc:
        path = path_weights + '/unet_weights_batchsize_25_Nepochs_100_SJR0_10.hdf5'
    else:
        path = path_weights + '/unet_weights_BF_batchsize_25_Nepochs_100_SJR_0_1.hdf5'

    if not os.path.exists(path):
        raise ValueError(f'Weights file not found in {path}')

    model.load_weights(path)

    results = model.predict(padded[np.newaxis,:,:,np.newaxis], batch_size=1)

    res = results[0,:,:,0]
    return res[:nrow, :ncol]
