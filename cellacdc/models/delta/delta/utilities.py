"""
Module for utility functions and class definitions that are used in pipeline.py

A few functions are redundant from data.py, but we keep the files separate
to minimize the risk of unforeseen bugs.

@author: jblugagne
"""

import re
import time
import sys
import importlib
from threading import Thread
from typing import cast, Tuple, List, Optional, Union, Any, Dict
from pathlib import Path

import cv2
# import ffmpeg
import numpy as np
import numpy.typing as npt
from skimage.morphology import skeletonize

from . import config as cfg

Image = npt.NDArray[np.float32]

Contour = np.ndarray

Labels = npt.NDArray[np.uint16]

SegmentationMask = npt.NDArray[np.uint8]

Pole = npt.NDArray[np.int16]

# TODO: properly type this
class CroppingBox(Dict):
    xtl: Any
    ytl: Any
    xbr: Any
    ybr: Any


class Cell(Dict):
    id: int
    mother: Optional[int]
    frames: List[int]
    daughters: List[Optional[int]]
    new_pole: List[Optional[Pole]]
    old_pole: List[Optional[Pole]]


#%% Image correction


def rangescale(frame: Image, rescale: Tuple[float, float]) -> Image:
    """
    Rescale image values to be within range

    Parameters
    ----------
    frame : ND numpy array of uint8/uint16/float/bool
        Input image(s).
    rescale : Tuple of 2 values
        Values range for the rescaled image.

    Returns
    -------
    2D numpy array of floats
        Rescaled image

    """
    frame = frame.astype(np.float32)
    if np.ptp(frame) > 0:
        frame = ((frame - np.min(frame)) / np.ptp(frame)) * np.ptp(rescale) + rescale[0]
    else:
        frame = np.ones_like(frame) * (rescale[0] + rescale[1]) / 2
    return frame


def deskew(image: Image) -> float:
    """
    Compute the rotation angle to apply to the image to remove its rotation.
    You can skip rotation correction if your chambers are about +/- 1 degrees of horizontal.

    Parameters
    ----------
    image : 2D numpy array
        Input image.

    Returns
    -------
    angle : float
        Rotation angle of the chambers for correction, in degrees.

    """
    from skimage.transform import hough_line, hough_line_peaks

    image8 = (256 * (image.astype(float) - image.min()) / (image.max() - image.min())).astype(np.uint8)

    # enhance edges
    low_threshold = np.quantile(image8, 0.1)
    high_threshold = np.quantile(image8, 0.2)
    edges = cv2.Canny(image8, low_threshold, high_threshold, L2gradient=True)

    # Hough transform
    N = 360  # precision in degree = 90 / N (here 0.25 degree)
    theta = np.linspace(-np.pi / 4, 3 * np.pi / 4, 2 * N + 1)
    hspace, angles, distances = hough_line(edges, theta=theta)

    _, corrections, _ = hough_line_peaks(hspace, angles, distances, num_peaks=1)

    if corrections[0] > np.pi / 4:
        return np.degrees(corrections[0] - np.pi / 2)
    return np.degrees(corrections[0])


def imrotate(frame: npt.NDArray[Any], rotate: float) -> npt.NDArray[Any]:
    """
    Rotate image

    Parameters
    ----------
    frame : ND numpy array of uint8/uint16/float/bool
        Input image(s).
    rotate : float
        Rotation angle, in degrees.

    Returns
    -------
    2D numpy array of floats
        Rotated image

    """
    M = cv2.getRotationMatrix2D((frame.shape[1] / 2, frame.shape[0] / 2), rotate, 1)
    frame = cv2.warpAffine(
        frame, M, (frame.shape[1], frame.shape[0]), borderMode=cv2.BORDER_REPLICATE
    )

    return frame


def driftcorr(img, template=None, box=None, drift=None):
    """
    Compute drift between current frame and the reference, and return corrected
    image

    Parameters
    ----------
    img : 2D or 3D numpy array of uint8/uint16/floats
        The frames to correct drfit for.
    template : None or 2D numpy array of uint8/uint16/floats, optional
        The template for drift correction (see getDriftTemplate()).
        default is None.
    box : None or dictionary, optional
        A cropping box to extract the part of the frame to compute drift
        correction over (see cropbox()).
        default is None.
    drift : None or tuple of 2 numpy arrays, optional
        Pre-computed drift to apply to the img stack. If this is None, you must
        provide a template and box.
        default it None.

    Returns
    -------
    2D/3D numpy array, tuple of len 2
        Drift-corrected image and drift.

    """

    if len(img.shape) == 2:
        twoDflag = True
        img = np.expand_dims(img, axis=0)
    else:
        twoDflag = False

    if drift is None:
        if template is None:
            # If we have a position with 0 chambers (see getDriftTemplate)
            return img, (0, 0)
        template = rangescale(template, (0, 255)).astype(np.uint8)
        xcorr = np.empty([img.shape[0]])
        ycorr = np.empty([img.shape[0]])
    elif twoDflag:
        (xcorr, ycorr) = ([drift[0]], [drift[1]])
    else:
        (xcorr, ycorr) = drift

    for i in range(img.shape[0]):
        if drift is None:
            frame = rangescale(img[i], (0, 255)).astype(np.uint8)
            driftcorrimg = cropbox(frame, box)
            res = cv2.matchTemplate(driftcorrimg, template, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(res)
            ycorr[i] = max_loc[0] - res.shape[1] / 2
            xcorr[i] = max_loc[1] - res.shape[0] / 2
        T = np.array([[1, 0, -ycorr[i]], [0, 1, -xcorr[i]]], dtype=np.float32)
        img[i] = cv2.warpAffine(img[i], T, img.shape[3:0:-1])

    if twoDflag:
        return np.squeeze(img), (xcorr[0], ycorr[0])
    else:
        return img, (xcorr, ycorr)


def getDriftTemplate(
    chamberboxes: List[CroppingBox], img: Image, whole_frame: bool = False
) -> Optional[Image]:
    """
    This function retrieves a region above the chambers to use as drift template

    Parameters
    ----------
    chamberboxes : list of dictionaries
        See getROIBoxes().
    img : 2D numpy array
        The first frame of a movie to use as reference for drift correction.
    whole_frame : bool, optional
        Whether to use the whole frame as reference instead of the area above
        the chambers.

    Returns
    -------
    2D numpy array or None
        A cropped region of the image to use as template for drift correction.
        If an empty list of chamber boxes is passed, None is returned.
        (see driftcorr()).

    """

    if len(chamberboxes) == 0 and not whole_frame:
        return None
    (y_cut, x_cut) = [
        round(i * 0.025) for i in img.shape
    ]  # Cutting out 2.5% of the image on eahc side as drift margin

    box = CroppingBox(
        xtl=x_cut,
        xbr=-x_cut,
        ytl=y_cut,
        ybr=-y_cut
        if whole_frame
        else max(chamberboxes, key=lambda elem: elem["ytl"])["ytl"] - y_cut,
    )

    return cropbox(img, box)


def opencv_areafilt(
    I: SegmentationMask, min_area: Optional[float] = 20, max_area: float = None
) -> SegmentationMask:
    """
    Area filtering using openCV instead of skimage

    Parameters
    ----------
    I : 2D array
        Segmentation mask.
    min_area : float or None, optional
        Minimum object area.
        The default is 20
    max_area : float or None, optional
        Maximum object area.
        The default is None.

    Returns
    -------
    I : 2D array
        Filtered mask.

    """

    # Get contours:
    contours = find_contours(I)

    # Loop through contours, flag them for deletion:
    to_remove = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if (min_area is not None and area < min_area) or (
            max_area is not None and area > max_area
        ):
            to_remove += [cnt]

    # Delete all at once:
    if len(to_remove) > 0:
        I = cv2.drawContours(I, to_remove, -1, 0, thickness=-1)

    return I


#%% Image cropping & stitching


def cropbox(img: npt.NDArray[Any], box: Optional[CroppingBox]) -> npt.NDArray[Any]:
    """
    Crop image

    Parameters
    ----------
    img : 2D numpy array
        Image to crop.
    box : Dictionary or None
        Dictionary describing the box to cut out, containing the following
        elements:
            - 'xtl': Top-left corner X coordinate.
            - 'ytl': Top-left corner Y coordinate.
            - 'xbr': Bottom-right corner X coordinate.
            - 'ybr': Bottom-right corner Y coordinate.

    Returns
    -------
    2D numpy array
        Cropped-out region.

    """
    if box is None or all([v is None for v in box.values()]):
        return img
    else:
        return img[box["ytl"] : box["ybr"], box["xtl"] : box["xbr"]]


def create_windows(
    img: Image, target_size: Tuple[int, int] = (512, 512), min_overlap: int = 24
) -> Tuple[npt.NDArray[np.float32], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Crop input image into windows of set size.

    Parameters
    ----------
    img : 2D array
        Input image.
    target_size : tuple, optional
        Dimensions of the windows to crop out.
        The default is (512,512).
    min_overlap : int, optional
        Minimum overlap between windows in pixels.
        Defaul is 24.


    Returns
    -------

    windows: 3D array
        Cropped out images to feed into U-Net. Dimensions are
        (nb_of_windows, target_size[0], target_size[1])
    loc_y : list
        List of lower and upper bounds for windows over the y axis
    loc_x : list
        List of lower and upper bounds for windows over the x axis

    """
    # Make sure image is minimum shape (bigger than the target_size)
    if img.shape[0] < target_size[0]:
        img = np.concatenate(
            (img, np.zeros((target_size[0] - img.shape[0], img.shape[1]))), axis=0
        )
    if img.shape[1] < target_size[1]:
        img = np.concatenate(
            (img, np.zeros((img.shape[0], target_size[1] - img.shape[1]))), axis=1
        )

    # Decide how many images vertically the image is split into
    ny = int(
        1 + float(img.shape[0] - min_overlap) / float(target_size[0] - min_overlap)
    )
    nx = int(
        1 + float(img.shape[1] - min_overlap) / float(target_size[1] - min_overlap)
    )
    # If image is 512 pixels or smaller, there is no need for anyoverlap
    if img.shape[0] == target_size[0]:
        ny = 1
    if img.shape[1] == target_size[1]:
        nx = 1

    # Compute y-axis indices:
    ovlp_y = -(img.shape[0] - ny * target_size[0]) / (ny - 1) if ny > 1 else 0
    loc_y = []
    for n in range(ny - 1):
        loc_y += [
            (
                int(target_size[0] * n - ovlp_y * n),
                int(target_size[0] * (n + 1) - ovlp_y * n),
            )
        ]
    loc_y += [(img.shape[0] - target_size[0], img.shape[0])]

    # Compute x-axis indices:
    ovlp_x = -(img.shape[1] - nx * target_size[1]) / (nx - 1) if nx > 1 else 0
    loc_x = []
    for n in range(nx - 1):
        loc_x += [
            (
                int(target_size[1] * n - ovlp_x * n),
                int(target_size[1] * (n + 1) - ovlp_x * n),
            )
        ]

    loc_x += [(img.shape[1] - target_size[1], img.shape[1])]

    # Store all cropped images into one numpy array called windows
    windows = np.zeros(((nx * ny,) + target_size), dtype=img.dtype)
    for i in range(len(loc_y)):
        for j in range(len(loc_x)):
            windows[i * len(loc_x) + j, :, :] = img[
                loc_y[i][0] : loc_y[i][1], loc_x[j][0] : loc_x[j][1]
            ]

    return windows, loc_y, loc_x


def stitch_pic(
    results: npt.NDArray[Any],
    loc_y: List[Tuple[int, int]],
    loc_x: List[Tuple[int, int]],
) -> npt.NDArray[Any]:
    """
    Stitch segmentation back together from the windows of create_windows()

    Parameters
    ----------
    results : 3D array
        Segmentation outputs from the seg model with dimensions
        (nb_of_windows, target_size[0], target_size[1])
    loc_y : list
        List of lower and upper bounds for windows over the y axis
    loc_x : list
        List of lower and upper bounds for windows over the x axis

    Returns
    -------
    stitch_norm : 2D array
        Stitched image.

    """

    # Create an array to store segmentations into a format similar to how the image was cropped
    stitch = np.zeros((loc_y[-1][1], loc_x[-1][1]), dtype=results.dtype)
    index = 0
    y_end = 0
    for i in range(len(loc_y)):

        # Compute y location of window:
        y_start = y_end
        if i + 1 == len(loc_y):
            y_end = loc_y[i][1]
        else:
            y_end = int((loc_y[i][1] + loc_y[i + 1][0]) / 2)

        x_end = 0
        for j in range(len(loc_x)):

            # Compute x location of window:
            x_start = x_end
            if j + 1 == len(loc_x):
                x_end = loc_x[j][1]
            else:
                x_end = int((loc_x[j][1] + loc_x[j + 1][0]) / 2)

            # Add to array:
            res_crop_y = -(loc_y[i][1] - y_end) if loc_y[i][1] - y_end > 0 else None
            res_crop_x = -(loc_x[j][1] - x_end) if loc_x[j][1] - x_end > 0 else None
            stitch[y_start:y_end, x_start:x_end] = results[
                index,
                y_start - loc_y[i][0] : res_crop_y,
                x_start - loc_x[j][0] : res_crop_x,
            ]

            index += 1

    return stitch


def gettrackingboxes(
    cell: npt.NDArray[np.uint8],
    frame_shape: Tuple[int, int] = None,
    target_size: Tuple[int, int] = None,
) -> Tuple[CroppingBox, CroppingBox]:
    """
    Get a crop box and a fill box around a cell that fits the tracking target
    size

    Parameters
    ----------
    cell : 2D array of uint8
        Mask of the cell to track.
    frame_shape : tuple of 2 ints or None, optional
        Original dimensions of the  image. If None, cell.shape is used.
        The default is None.
    target_size : tuple of 2 ints or None, optional
        Target dimensions of the cropped image. If None, cfg.target_size_track
        will be used.
        The default is None.

    Returns
    -------
    cropbox : dict
        Crop box in the cropbox() input format.
        The crop box determines which part of the full-size frame to crop out.
    fillbox : dict
        Fill box in the cropbox() input format.
        The fill box determines which part of the target-size input to fill with
        the cropped out image.

    """

    if frame_shape is None:
        frame_shape = cast(Tuple[int, int], cell.shape)  # for mypy

    if target_size is None:
        target_size = cfg.target_size_track

    cx, cy = getcentroid(cell)

    xtl = int(max(cx - target_size[1] / 2, 0))
    xbr = int(min(cx + target_size[1] / 2, frame_shape[1]))

    ytl = int(max(cy - target_size[0] / 2, 0))
    ybr = int(min(cy + target_size[0] / 2, frame_shape[0]))

    cropbox = CroppingBox({"xtl": xtl, "ytl": ytl, "xbr": xbr, "ybr": ybr})

    xtl = int(max(target_size[1] / 2 - cx, 0))
    xbr = int(min(target_size[1] / 2 + frame_shape[1] - cx, target_size[1]))

    ytl = int(max(target_size[0] / 2 - cy, 0))
    ybr = int(min(target_size[0] / 2 + frame_shape[0] - cy, target_size[0]))

    fillbox = CroppingBox({"xtl": xtl, "ytl": ytl, "xbr": xbr, "ybr": ybr})

    return cropbox, fillbox


def getshiftvalues(
    shift: int, img_shape: Tuple[int, int], cb: CroppingBox
) -> Tuple[int, int]:
    """

    Parameters
    ----------
    shift : int
        Max amount of pixels cropbox to be shifted in the y / x direction.
    img_shape : tuple
        Shape of the image / input that will be cropped.
    cropbox : dict
        Crop box in the cropbox() input format.
        The crop box determines which part of the full-size frame to crop out.

    Returns
    -------
    shiftY : int
        Number of pixels to shift cropbox in the y irection
    shiftX : int
        Number of pixels to shift cropbox in the x direction.

    """
    upperY = np.min((shift, img_shape[0] - cb["ybr"]))
    lowerY = np.min((shift, cb["ytl"]))
    shiftY = int(np.random.uniform(-lowerY, upperY))

    upperX = np.min((shift, img_shape[1] - cb["xbr"]))
    lowerX = np.min((shift, cb["xtl"]))
    shiftX = int(np.random.uniform(-lowerX, upperX))

    return shiftY, shiftX


#%% Masks, labels, objects identification


def find_contours(mask: npt.NDArray[np.uint8]) -> List[Contour]:
    """
    wrapper for CV2's findContours() because it
    keeps changing signatures

    Parameters
    ----------
    mask : 2D array of uint8.
        Segmentation mask to extract contours from.

    Returns
    -------
    contours : list
        List of cv2 type contour arrays.

    """

    # Default use:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Some versions of cv2.findcontours return it as a tuple, not a list:
    contours = list(contours)

    return contours


def getROIBoxes(chambersmask: np.ndarray) -> List[CroppingBox]:
    """
    Extract the bounding boxes of the chambers in the binary mask
    produced by the chambers identification unet

    Parameters
    ----------
    chambersmask : 2D array of uint8/uint16/floats
        The mask of the chambers as returned by the chambers id unet.

    Returns
    -------
    chamberboxes : list of dictionaries
        List of cropping box dictionaries (see cropbox()).

    """
    chamberboxes = []
    chambersmask = (chambersmask > 0.5).astype(np.uint8)

    contours = find_contours(chambersmask)
    for chamber in contours:
        xtl, ytl, boxwidth, boxheight = cv2.boundingRect(chamber)
        chamberboxes.append(
            # tl = top left, br = bottom right
            CroppingBox(
                xtl=xtl,
                # -10% of height to make sure the top is not cropped
                ytl=ytl - int(0.1 * boxheight),
                xbr=xtl + boxwidth,
                ybr=ytl + boxheight,
            )
        )

    # Sorting by top-left X (normally sorted by Y top left)
    chamberboxes.sort(key=lambda box: box["xtl"])

    return chamberboxes

    #### These prototypes help mypy understand the return type of the function
    #    but can only be used from Python 3.8 because of typing.Literal
    #
    #
    # @overload
    # def label_seg(
    #     seg: SegmentationMask,
    #     cellnumbers: List[int],
    #     return_contours: Literal[True],
    #     background: int = 0,
    # ) -> Tuple[Labels, List[Contour]]:
    #     ...
    #
    #
    # @overload
    # def label_seg(
    #     seg: SegmentationMask,
    #     cellnumbers: List[int] = None,
    #     *,
    #     return_contours: Literal[True],
    #     background: int = 0
    # ) -> Tuple[Labels, List[Contour]]:
    #     ...
    #
    #
    # @overload
    # def label_seg(
    #     seg: SegmentationMask,
    #     cellnumbers: List[int] = None,
    #     *,
    #     return_contours: Literal[False] = False,
    #     background: int = 0
    # ) -> Labels:
    ...


def label_seg(
    seg: SegmentationMask,
    cellnumbers: List[int] = None,
    return_contours: bool = False,
    background: int = 0,
) -> Union[Labels, Tuple[Labels, List[Contour]]]:
    """
    Label cells in segmentation mask

    Parameters
    ----------
    seg : numpy 2D array of float/uint8/uint16/bool
        Cells segmentation mask. Values >0.5 will be considered cell pixels
    cellnumbers : list of ints, optional
        Numbers to attribute to each cell mask, from top to bottom of image.
        Because we are using uint16s, maximum cell number is 65535. If None is
        provided, the cells will be labeled 1,2,3,... Background is 0
        The default is None.

    Returns
    -------
    label : 2D numpy array of uint16
        Labelled image. Each cell in the image is marked by adjacent pixels
        with values given by cellnumbers
    contours : list
        List of cv2 contours for each cell. Returned if return_contours==True.

    """
    seg = (seg > 0.5).astype(np.uint8)
    contours = find_contours(seg)
    contours.sort(key=lambda elem: np.max(elem[:, 0, 1]))  # Sorting along Y
    label = np.full(seg.shape, background, dtype=np.uint16)
    for c, contour in enumerate(contours):
        label = cv2.fillPoly(
            label, [contour], c + 1 if cellnumbers is None else cellnumbers[c]
        )
    if return_contours:
        return label, contours
    else:
        return label


#### These prototypes help mypy understand the return type of the function
#    but can only be used from Python 3.8 because of typing.Literal
#
#
# @overload
# def getcellsinframe(
#     labels: Labels, return_contours: Literal[True], background: int = 0
# ) -> Tuple[List[int], List[Contour]]:
#     ...
#
#
# @overload
# def getcellsinframe(
#     labels: Labels, return_contours: Literal[False] = False, background: int = 0
# ) -> List[int]:
#     ...


def getcellsinframe(
    labels: Labels, return_contours: bool = False, background: int = 0
) -> Union[List[int], Tuple[List[int], List[Contour]]]:
    """
    Get numbers of cells present in frame, sorted along Y axis

    Parameters
    ----------
    labels : 2D numpy array of ints
        Single frame from labels stack.
    return_contours : bool, optional
        Flag to get cv2 contours.

    Returns
    -------
    cells : list
        Cell numbers (0-based indexing).
    contours : list
        List of cv2 contours for each cell. Returned if return_contours==True.

    """

    cells, ind = np.unique(labels, return_index=True)
    cells = [
        cell - 1 for _, cell in sorted(zip(ind, cells))
    ]  # Sorting along Y axis, 1-based to 0-based, & removing 1st value (background)
    cells = [cell for cell in cells if cell != background - 1]  # Remove background

    # Get opencv contours:
    if return_contours:
        contours = []
        for c, cell in enumerate(cells):
            # Have to do it this way to avoid cases where border is shrunk out
            cnt = find_contours((labels == cell + 1).astype(np.uint8))
            contours += cnt
        return cells, contours
    else:
        return cells


def getcentroid(contour: np.ndarray) -> Tuple[int, int]:
    """
    Get centroid of cv2 contour

    Parameters
    ----------
    contour : 3D numpy array
        Blob contour generated by cv2.findContours().

    Returns
    -------
    cx : int
        X-axis coordinate of centroid.
    cy : int
        Y-axis coordinate of centroid.

    """

    if contour.shape[0] > 2:  # Looks like cv2.moments treats it as an image
        # Calculate moments for each contour
        M = cv2.moments(contour)
        # Calculate x,y coordinate of center
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0
    else:
        cx = int(np.mean(contour[:, :, 0]))
        cy = int(np.mean(contour[:, :, 1]))

    return cx, cy


#%% Poles


def getpoles(
    seg: SegmentationMask, labels: Labels = None, scaling: Tuple[float, float] = None
) -> List[Tuple[Pole, Pole]]:
    """
    Get cell poles

    Parameters
    ----------
    seg : 2D array of uint8
        Cell segmentation mask.
    labels : 2D array of int, optional
        Cell labels map. If None, label_seg() will be applied to seg
        The default is None.
    scaling : tuple of floats, optional
        Scaling to apply in both directions to rescale poles' positions to
        original image size (if resizing was applied)

    Returns
    -------
    poles : list
        List of poles per cell. Each cell in the list has exactly 2 poles.

    """

    # No label provided:
    if labels is None:
        lbls = label_seg(seg)
        assert isinstance(lbls, np.ndarray)  # needed for mypy on Python < 3.8
        labels = lbls

    # Get poles using skeleton method:
    skel = skeletonize(seg, method="lee")
    ends_map = skeleton_poles(skel)
    poles = extract_poles(ends_map, labels)

    clean_poles = []

    # Make sure cells have 2 poles each:
    for p in range(len(poles)):
        if len(poles[p]) < 2:  # Sometimes skeletonize fails
            clean_poles.append(extrema_poles(labels == p + 1, scaling=scaling))
        elif len(poles[p]) == 2:
            clean_poles.append((poles[p][0], poles[p][1]))
        else:
            clean_poles.append(two_poles(poles[p]))

    # Apply scaling:
    if scaling:
        clean_poles = [
            (
                np.round(p1 * scaling).astype(np.int16),
                np.round(p2 * scaling).astype(np.int16),
            )
            for p1, p2 in clean_poles
        ]

    return clean_poles


def skeleton_poles(skel: SegmentationMask) -> npt.NDArray[np.bool_]:
    """
    This function was adapted from stackoverflow
    #https://stackoverflow.com/questions/26537313/how-can-i-find-endpoints-of-binary-skeleton-image-in-opencv

    It uses a kernel to filter out the poles

    Parameters
    ----------
    skel : 2D numpy array of bool
        Contains skeletons of single cells from the segmentation.

    Returns
    -------
    out : 2D numpy array of bool
        Contains True at the poles of the skeletons from the input.

    """
    # create an integer array from the boolean one
    skel_uint8 = np.array(skel, dtype=np.uint8)

    # apply the convolution
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    ddepth = -1  # same depth as the source
    filtered = cv2.filter2D(skel_uint8, ddepth, kernel, borderType=cv2.BORDER_ISOLATED)

    # now look through to find the value of 11
    # this returns a mask of the poles
    return filtered == 11


def extrema_poles(
    cellmask: SegmentationMask, scaling: Tuple[float, float] = None
) -> Tuple[Pole, Pole]:
    """
    A slower and more rudimentary version of poles finding but that is
    guaranteed to find exactly 2 poles

    Parameters
    ----------
    cellmask : 2D array of int
        Segmentation mask of a single cell.

    Returns
    -------
    poles : tuple of 2 1D arrays
        Two poles, each being a 2-element array of y-x coordinates.

    """

    scaling = scaling or (1, 1)

    y, x = cellmask.nonzero()
    if np.ptp(y) * scaling[0] > np.ptp(x) * scaling[1]:  # Vertical cell
        i = np.lexsort((x, y))
    else:  # Horizontal cell
        i = np.lexsort((y, x))
    poles = (
        np.array((y[i[0]], x[i[0]]), dtype=np.int16),
        np.array((y[i[-1]], x[i[-1]]), dtype=np.int16),
    )

    return poles


def two_poles(poles: List[Pole]) -> Tuple[Pole, Pole]:
    """
    Sometimes the skeleton produces more than 2 poles.
    Thus function selects only 2 poles in the skeleton.

    Parameters
    ----------
    poles : list
        Coordinates for the mother/daughter poles for a single cell
        Dimension is (#_of_poles, [y_coordiante, x_coordinate])


    Returns
    -------
    poles : tuple of 2 1D arrays
        Two poles, each being a 2-element array of y-x coordinates.

    """

    dist = 0.0
    # Measure the distance from the first endpoint to all the other poles
    for i in range(1, len(poles)):
        d = eucl(poles[0], poles[i])
        if d > dist:
            dist = d
            i_max = i

    # Redo the same thing starting from the furthest endpoint:
    dist = 0.0
    for j in range(len(poles)):
        if j != i_max:  # skip same point
            d = eucl(poles[i_max], poles[j])
            if d > dist:
                dist = d
                j_max = j

    return (poles[i_max], poles[j_max])


def extract_poles(end_img: npt.NDArray[np.bool_], labels: Labels) -> List[List[Pole]]:
    """
    Extract poles per cell from ends image

    Parameters
    ----------
    end_img : 2D array of bool
        'Mask' of the poles in the image.
    labels : 2D array of int
        Cell labels map.

    Returns
    -------
    poles : list of tuples of 1D numpy arrays
        For each cell, list of numpy arrays with the y-x coordinates
        of the poles.  In principle, each list has two poles but sometimes not.

    """
    locations = end_img.nonzero()
    poles: List[List[Pole]] = [[] for _ in range(np.max(labels))]
    for p in range(len(locations[0])):

        cell = labels[locations[0][p], locations[1][p]] - 1
        poles[cell] += [np.array([locations[0][p], locations[1][p]], dtype=np.int16)]

    return poles


def eucl(p1: npt.NDArray[Any], p2: npt.NDArray[Any]) -> float:
    """
    Euclidean point to point distance

    Parameters
    ----------
    p1 : 1D array
        Coordinates of first point.
    p2 : 1D array
        Coordinates of second point.

    Returns
    -------
    float
        Euclidean distance between p1 and p2.

    """
    return np.linalg.norm(p1 - p2)


def track_poles(
    poles: Tuple[Pole, Pole], prev_old: Pole, prev_new: Pole
) -> Tuple[Pole, Pole]:
    """
    Track poles of a cell to the previous old and new poles

    Parameters
    ----------
    poles : Tuple[Pole, Pole]
        Coordinates for the 2 poles of a single cell
    prev_old : Pole
        Previous old pole of the cell.
    prev_new : Pole
        Previous new pole of the cell.

    Returns
    -------
    old_pole : Pole
        Which of the two poles tracked to the previous old pole.
    new_pole : Pole
        Which of the two poles tracked to the previous new pole.

    """

    if (
        eucl(poles[0], prev_old) ** 2 + eucl(poles[1], prev_new) ** 2
        < eucl(poles[0], prev_new) ** 2 + eucl(poles[1], prev_old) ** 2
    ):
        old_pole = poles[0]
        new_pole = poles[1]
    else:
        old_pole = poles[1]
        new_pole = poles[0]

    return old_pole, new_pole


def division_poles(
    poles_cell1: Tuple[Pole, Pole],
    poles_cell2: Tuple[Pole, Pole],
    prev_old: Pole,
    prev_new: Pole,
) -> Tuple[Tuple[Pole, Pole], Tuple[Pole, Pole], bool]:
    """
    Identify which poles belong to the mother and which to the daughter

    Parameters
    ----------
    poles_cell1 : Tuple[Pole, Pole]
        Poles of one of the 2 cells after division.
    poles_cell2 : Tuple[Pole, Pole]
        Poles of the other of the 2 cells after division.
    prev_old : Pole
        Previous old pole of the cell.
    prev_new : Pole
        Previous new pole of the cell.

    Returns
    -------
    Tuple[Tuple[Pole, Pole], Tuple[Pole, Pole], bool]
        Mother's old pole and new pole, daughter's old pole and new pole, and
        flag that is True if cell1 is the mother, False if cell2 is the mother

    """

    # Find new new poles (2 closest of the poles of the new cells):
    min_dist = np.inf
    for c1 in range(2):
        for c2 in range(2):
            dist = eucl(poles_cell1[c1], poles_cell2[c2])
            if dist < min_dist:
                min_dist = dist
                c1_new = c1
                c2_new = c2

    # Track poles closest to old and new pole from previous cell:
    if (
        eucl(poles_cell1[1 - c1_new], prev_old) ** 2
        + eucl(poles_cell2[1 - c2_new], prev_new) ** 2
        < eucl(poles_cell1[1 - c1_new], prev_new) ** 2
        + eucl(poles_cell2[1 - c2_new], prev_old) ** 2
    ):
        # cell 1 is mother, cell 2 is daughter
        first_cell_is_mother = True
        # Attribute mother:
        mother_old_new = (poles_cell1[1 - c1_new], poles_cell1[c1_new])

        # Attribute daughter:
        daughter_old_new = (poles_cell2[1 - c2_new], poles_cell2[c2_new])

    else:
        # cell 2 is mother, cell 1 is daughter
        first_cell_is_mother = False
        # Attribute mother:
        mother_old_new = (poles_cell2[1 - c2_new], poles_cell2[c2_new])

        # Attribute daughter:
        daughter_old_new = (poles_cell1[1 - c1_new], poles_cell1[c1_new])

    return (mother_old_new, daughter_old_new, first_cell_is_mother)


#%% Lineage


def getTrackingScores(
    labels: Labels,
    outputs: npt.NDArray[np.float32],
    boxes: List[Tuple[CroppingBox, CroppingBox]] = None,
) -> Optional[npt.NDArray[np.float32]]:
    """
    Get overlap scores between input/target cells and tracking outputs

    Parameters
    ----------
    inputs : 2D array of floats
        Segmentation mask of input/target cells that the tracking U-Net is
        tracking against. (ie segmentation mask of the 'current'/'new' frame)
    outputs : 4D array of floats
        Tracking U-Net output.
    boxes : list of dict, optional
        Cropping box and fill box to re-place output prediction masks in the
        original coordinates to index the labels frame.

    Returns
    -------
    scores : 2D array of floats, or None
        Overlap scores matrix between tracking predictions and current
        segmentation mask for each new-old cell.

    """

    total_cell = np.max(labels)
    if total_cell == 0:
        return None

    # Get areas for each cell:
    _, areas = np.unique(labels, return_counts=True)
    areas = areas[1:]

    # Compile scores:
    scores = np.zeros([outputs.shape[0], total_cell], dtype=np.float32)
    for o, output in enumerate(outputs):

        # Find pixels with a score > .05:
        nz = list((output > 0.05).nonzero())
        if len(nz[0]) == 0:
            continue

        # Find cells that are "hit" by those pixels in the labels image:
        if (
            boxes is None
            or boxes[o] is None
            or all([v is None for v in boxes[o][0].values()])
        ):
            cells, counts = np.unique(labels[tuple(nz)], return_counts=True)

        else:
            # Clean nz hits outside of fillbox:
            cb, fb = boxes[o][:]
            to_remove = np.logical_or.reduce(
                (
                    nz[0] <= fb["ytl"],
                    nz[0] >= fb["ybr"],
                    nz[1] <= fb["xtl"],
                    nz[1] >= fb["xbr"],
                )
            )
            nz[0] = np.delete(nz[0], to_remove.nonzero())
            nz[1] = np.delete(nz[1], to_remove.nonzero())

            # Offset hits by cropbox-fillbox:
            nz[0] = nz[0] + cb["ytl"] - fb["ytl"]
            nz[1] = nz[1] + cb["xtl"] - fb["xtl"]

            # Compute number of hits per cell:
            cells, counts = np.unique(labels[nz[0], nz[1]], return_counts=True)

        # Compile score for these cells:
        for c, cell in enumerate(cells):
            if cell > 0:
                scores[o, cell - 1] = counts[c] / areas[cell - 1]

    return scores


def getAttributions(scores: npt.NDArray[np.float32]) -> npt.NDArray[np.bool_]:
    """
    Get attribution matrix from tracking scores

    Parameters
    ----------
    scores : 2D array of floats
        Tracking scores matrix as produced by the getTrackingScores function.

    Returns
    -------
    attrib : 2D array of bools
        Attribution matrix. Cells from the old frame (axis 0) are attributed to
        cells in the new frame (axis 1). Each old cell can be attributed to 1
        or 2 new cells.

    """

    attrib = np.zeros(scores.shape, dtype=int)

    for n in range(scores.shape[1]):
        # Worst-to-best score of old cells for n-th new cell
        worst_old = np.argsort(scores[:, n])
        # Run through old cells from best to worst:
        for o in range(-1, -scores.shape[0] - 1, -1):
            o_best = worst_old[o]  # o-th best old cell
            s = scores[o_best, n]
            # If score gets too low, stop:
            if s < 0.2:
                break
            # Check if new cell is at least 2nd best for this old cell:
            # Worst-to-best score of new cells for o_best old cell
            worst_new = np.argsort(scores[o_best, :])
            if n in worst_new[-2:]:
                attrib[o_best, n] = 1
                break

    return attrib


class Lineage:
    """
    Class for cell lineages contained in each ROI
    """

    def __init__(self):
        """
        Initialize object.

        Returns
        -------
        None.

        """
        self.cells = []
        """List of single-cell dictionaries
        Each dictionary will contain single-cell lineage and extracted feature
        properties"""
        self.stack = []  # Orphan attribute?
        self.cellnumbers = []
        "List of cells present in FOV at each timepoint"

    def __str__(self):
        return "\n".join(
            f"cell #{icell:04}: "
            + "".join(
                " +"[iframe in cell["frames"]]
                for iframe in range(max(cell["frames"]) + 1)
            )
            for icell, cell in enumerate(self.cells)
        )

    def update(
        self,
        cell: Optional[int],
        frame: int,
        attrib: List[int] = [],
        poles: List[Tuple[Pole, Pole]] = [],
    ) -> None:
        """
        Attribute cell from previous frame to cell(s) in new frame.

        Parameters
        ----------
        cell : int or None
            Cell index in previous frame to update. If None, the cells in the
            new frame are treated as orphans.
        frame : int
            Frame / timepoint number.
        attrib : list of int, optional
            Cell index(es) in current frame to attribute the previous cell to.
            The default is [].
        poles : list of tuples of 2 1D numpy arrays, optional
            List of poles for the cells in the new frame.
            The default is [].

        Returns
        -------
        None.

        """

        # Check cell numbers list:
        if len(self.cellnumbers) <= frame:
            self.cellnumbers += [
                [] for _ in range(frame + 1 - len(self.cellnumbers))
            ]  # Initialize
        if len(attrib) > 0 and len(self.cellnumbers[frame]) <= max(attrib):
            self.cellnumbers[frame] += [
                -1 for _ in range(max(attrib) + 1 - len(self.cellnumbers[frame]))
            ]  # Initialize

        # If no attribution: (new/orphan cell)
        if cell is None and len(attrib) > 0:
            new_pole, old_pole = sorted(
                list(poles[0]), key=lambda p: p[0], reverse=True
            )
            cellnum = self.createcell(frame, new_pole=new_pole, old_pole=old_pole)
            self.cellnumbers[frame][attrib[0]] = cellnum

        # Simple tracking event:
        elif len(attrib) == 1:
            cellnum = self.cellnumbers[frame - 1][cell]  # Get old cell number
            # Update cellnumbers list:
            self.cellnumbers[frame][attrib[0]] = cellnum
            # Poles of mother/previous/old cell:
            prev_old = self.getvalue(cellnum, frame - 1, "old_pole")
            prev_new = self.getvalue(cellnum, frame - 1, "new_pole")
            assert isinstance(prev_old, np.ndarray)  # for mypy
            assert isinstance(prev_new, np.ndarray)  # for mypy
            # Find which pole is new vs old:
            old_pole, new_pole = track_poles(poles[0], prev_old, prev_new)
            # Update:
            self.updatecell(cellnum, frame, old_pole=old_pole, new_pole=new_pole)

        # Division event:
        elif len(attrib) == 2:
            mothernum = self.cellnumbers[frame - 1][cell]  # Get old cell number
            # Poles of mother/previous/old cell:
            prev_old = self.getvalue(mothernum, frame - 1, "old_pole")
            prev_new = self.getvalue(mothernum, frame - 1, "new_pole")
            assert isinstance(prev_old, np.ndarray)  # for mypy
            assert isinstance(prev_new, np.ndarray)  # for mypy
            # Identify mother and daughter poles:
            mother_poles, daughter_poles, first_cell_is_mother = division_poles(
                poles[0], poles[1], prev_old, prev_new
            )
            # Create daughter:
            daughternum = self.createcell(
                frame,
                old_pole=daughter_poles[0],
                new_pole=daughter_poles[1],
                mother=mothernum,
            )
            # Update mother:
            self.updatecell(
                mothernum,
                frame,
                old_pole=mother_poles[0],
                new_pole=mother_poles[1],
                daughter=daughternum,
            )
            # Update cellnumbers list:
            if first_cell_is_mother:
                self.cellnumbers[frame][attrib[0]] = mothernum
                self.cellnumbers[frame][attrib[1]] = daughternum
            else:
                self.cellnumbers[frame][attrib[0]] = daughternum
                self.cellnumbers[frame][attrib[1]] = mothernum

    def createcell(
        self,
        frame: int,
        new_pole: Pole = None,
        old_pole: Pole = None,
        mother: int = None,
    ) -> int:
        """
        Create cell to append to lineage list

        Parameters
        ----------
        frame : int
            Frame that the cell first appears (1-based indexing).
        new_pole : tuple of int, optional
            Coordinates of the new pole
        old_pole : tuple of int, optional
            Coordinates of the old pole
        mother : int or None, optional
            Number of the mother cell in the lineage (1-based indexing).
            The default is None. (ie unknown mother)

        Returns
        -------
        int
            new cell number

        """

        new_cell = Cell(
            {
                "id": len(self.cells),
                "mother": mother,
                "frames": [frame],
                "daughters": [None],
                "new_pole": [new_pole],
                "old_pole": [old_pole],
            }
        )
        self.cells.append(new_cell)

        return new_cell["id"]

    def updatecell(
        self,
        cell: int,
        frame: int,
        daughter: int = None,
        new_pole: Pole = None,
        old_pole: Pole = None,
    ) -> None:
        """
        Update cell lineage values

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Frame / timepoint number.
        daughter : int or None, optional
            Daughter cell number if division just happened.
            The default is None.
        new_pole : list, optional
            New pole location. The default is None.
        old_pole : list, optional
            Old pole location. The default is None.

        Returns
        -------
        None.

        """

        self.cells[cell]["frames"] += [frame]  # Update frame number list
        self.cells[cell]["daughters"] += [daughter]
        self.cells[cell]["old_pole"] += [old_pole]
        self.cells[cell]["new_pole"] += [new_pole]

    def setvalue(
        self, cell: int, frame: int, feature: str, value: Union[int, float, str]
    ) -> None:
        """
        Set feature value for specific cell and frame/timepoint

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Frame / timepoint number.
        feature : str
            Feature to set value for.
        value : int, float, str
            Value to assign.

        Raises
        ------
        ValueError
            Raised if the cell has not been detected in the frame.

        Returns
        -------
        None.

        """

        try:
            i = self.cells[cell]["frames"].index(frame)
        except ValueError:
            raise ValueError(f"Cell {cell} is not present in frame {frame}")

        # Get cell dict
        cell_dict = self.cells[cell]

        # If feature doesn't exist yet, create list:
        if feature not in cell_dict:
            cell_dict[feature] = [None for _ in range(len(cell_dict["frames"]))]

        # Add value:
        cell_dict[feature][i] = value

    def getpastvalue(self, cell: int, frame: int, feature: str, df: int = 1) -> float:
        """
        Get feature value for a frame in the past, ignoring cell divisions.

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Current frame number.
        feature : str
            Feature to get value for. Only works for numerical scalar features.
        df : int
            Number of frames in the past where we want to get the value.

        Returns
        -------
        float
            The value in the past, normalized by ratio of the value at current
            timepoint for initial cell by sum of value all progeny of the past
            cell (if any sister/cousin cells)

        """
        # This function works in two steps. First, it determines the ancestor
        # cell of cell `cell` at frame `frame - df`. Then, it determines what
        # percentage of the current progeny of the ancestor, the cell `cell`
        # represents.

        # This loop computes the ancestor cell of cell `cell` and puts its
        # number into `ancestor`.
        value_ancestor = None
        ancestor = cell
        while value_ancestor is None:
            try:
                # Did the current cell exist at the specified frame? If so, the search is over.
                value_ancestor = self.getvalue(ancestor, frame=frame - df, feature=feature)
            except ValueError:
                # The cell did not exist at the specified frame, so we try with its mother.
                ancestor = self.cells[ancestor]["mother"]
                # If the cell has no mother, we cannot continue.
                if ancestor is None:
                    return None

        # At this point, `ancestor` and `value_ancestor` are correctly filled.
        # Now we will collect the value of all the progeny of this cell, at the
        # current frame.
        value_now = self.getfuturevalue(ancestor, frame - df, feature, df=df)
        # If we cannot do that, it means that we lost track of a children. We exit.
        if value_now is None or value_now == 0.0:
            return None

        # At this point, we have everything needed. We can return the value of
        # the ancestor, weighted by the ratio of the current cell compared to
        # its sisters/cousins/etc.
        return self.getvalue(cell, frame, feature) * value_ancestor / value_now

    def getfuturevalue(self, cell: int, frame: int, feature: str, df: int = 1) -> float:
        """
        Get feature value for a frame in the future, ignoring cell divisions.

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Current frame number.
        feature : str
            Feature to get value for. Only works for numerical scalar features.
        df : int
            Number of frames in the future where we want to get the value.

        Returns
        -------
        value: float
            Sum of value for all progeny of initial cell in the future.
        """
        # This function makes progress recursively.
        # Here we handle the base case where the frame requested is the current one.
        if df == 0:
            return self.getvalue(cell, frame, feature)

        # Here we have df > 0, so we make progress to the next frame,
        # considering the current branch, and the potential branch of a
        # daughter cell if one exists at next frame.

        try:
            # Given our cell numbering, the mother cell has to exist at next
            # frame, unless it gets out of frame or there is a tracking error.
            # In any of these cases, we have to exit.
            value = self.getfuturevalue(
                cell, frame=frame + 1, feature=feature, df=df - 1
            )
        except ValueError:
            return None

        # If the cell has a daughter at next frame, we consider this branch as well.
        idaughter = self.getvalue(cell, frame + 1, "daughters")
        if idaughter is not None:
            value += self.getfuturevalue(
                idaughter, frame=frame + 1, feature=feature, df=df - 1
            )
        return value

    def getvalue(self, cell: int, frame: int, feature: str) -> Union[int, float, str]:
        """
        Get feature value for specific timepoint/frame

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Frame / timepoint number.
        feature : str
            Feature to get value for.

        Raises
        ------
        ValueError
            Raised if the cell has not been detected in the frame.

        Returns
        -------
        int, float, str
            Value for feature at frame.

        """

        try:
            i = self.cells[cell]["frames"].index(frame)
        except ValueError:
            raise ValueError(f"Cell {cell} is not present in frame {frame}")

        return self.cells[cell][feature][i]

    def growthrate(self, cell: int, frame: int, feature: str) -> float:
        """
        Compute exponential growth rate of a given feature for the given cell at the given frame.

        Compute the central derivative with the previous and the next frame if
        possible, otherwise a one-sided derivative.

        Parameters
        ----------
        cell : int
            Cell number in lineage.
        frame : int
            Frame / timepoint number.
        feature : str
            Feature to get value for.

        Returns
        -------
        float
            Growth rate of the feature at frame.
        """
        value_fm1 = self.getpastvalue(cell, frame, feature)
        value_f = self.getvalue(cell, frame, feature)
        value_fp1 = self.getfuturevalue(cell, frame, feature)
        if value_fm1 is not None and value_fp1 is not None and value_fm1 > 0.0 and value_fp1 > 0.0:
            return np.log(value_fp1 / value_fm1) / 2.0
        if value_fm1 is not None and value_fm1 > 0.0 and value_f > 0.0:
            return np.log(value_f / value_fm1)
        if value_fp1 is not None and value_f > 0.0 and value_fp1 > 0.0:
            return np.log(value_fp1 / value_f)
        return None


#%% Image files


def getxppathdialog(ask_folder: bool = False) -> str:
    """
    Pop up window to select experiment file or folder.

    Parameters
    ----------
    ask_folder : bool, optional
        Folder selection window will pop up instead of file selection.
        The default is False.

    Returns
    -------
    file_path : str
        Path to experiment file or folder.

    """
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    # root.withdraw() # For some reason this doesn't work with askdirectory?
    if ask_folder:
        file_path = filedialog.askdirectory(
            title="Please select experiment folder(s)", mustexist=True
        )

    else:
        file_path = filedialog.askopenfilename(
            title="Please select experiment files(s)"
        )

    root.destroy()

    return file_path


class xpreader:
    """
    Class to read experiment files from bioformats-compatible files or from
    file sequences in folders
    """

    def __init__(
        self,
        filename=None,
        channelnames: Tuple[str] = None,
        use_bioformats: bool = False,
        prototype: str = None,
        fileorder: str = "pct",
        filenamesindexing: int = 1,
        watchfiles: bool = False,
    ):
        """
        Initialize experiment reader

        Parameters
        ----------
        filename : String or None
            Path to experiment file or directory. If the path leads to a
            directory, the experiment folder is expected to contain exclusively
            single-page tif images. If None, an interactive selection dialog
            will be used. If no prototype is provided, the filenames are
            expected to be of the following C-style format:
            %s%d%s%d%s%d.tif, with the 3 %d digit strings being zero-padded
            decimal representation of the position, channel and frame/timepoint
            number of each image file.
            Valid examples:
                Pos01_Cha3_Fra0005.tif
                p3c1t034.tif
                xy 145 - fluo 029 - timepoint 005935 .TIFF
        channelnames : List/tuple of strings or None, optional
            Names of the acquisition channels ('trans', 'gfp', ...).
            The default is None.
        use_bioformats : bool, optional
            Flag to use the bioformats reader.
            The default is False.
        prototype: string, optional
            Filename prototype to use when reading single-page tif images from
            a sequence folder, in C-style formatting. Folder separators can be
            used. If None, the prototype will be estimated from the first tif
            file in the folder. For example, an experiment from micromanager
            can be processed with prototype =
            'Pos%01d/img_channel%03d_position%03d_time%09d_z000.tif'
            and fileorder = 'pcpt' and filenamesindexing = 0
            The default is None.
        fileorder: string, optional
            Order of the numbers in the prototype, with 'p' for positions/
            series, 'c' for imaging channels, and 't' for timepoints/frames.
            For example 'pct' indicates that the first number is going to be
            positions, then channels, then timepoints. You can use the same
            letter multiple times, like 'pcpt'.
            The default is 'pct'
        filenamesindexing = int
            Selects between 0-based or 1-based indexing in the filename. If
            1, position 0 will be referenced as position 1 in the filename.
            The default is 1


        Raises
        ------
        ValueError
            If the filenames in the experimental directory do not follow the
            correct format, a ValueError will be raised.

        Returns
        -------
        None.

        """

        # Set default parameters
        self.filename = filename
        "File or folder name for the experiment"
        self.use_bioformats = use_bioformats
        "Flag to use the Bio-Formats library to read file"
        self.fileorder = fileorder
        "File ordering (if image files sequence)"
        self.filenamesindexing = filenamesindexing
        "File indexing start (if image files sequence)"
        self.prototype = prototype
        "File name prototype (if image files sequence)"

        # Init base parameters:
        self.filetype = None
        "Type / extension of `filehandle`"
        self.filehandle: Any
        "Handle to file reader or base directory"
        self.positions: int
        "Number of positions in experiment"
        self.channels: int
        "Number of imaging channels"
        self.x: int
        "Size of images along X axis"
        self.y: int
        "Size of images along Y axis"
        self.dtype: str
        "Datatype of images"

        # Retrieve command line arguments (if any)
        cmdln_arguments = sys.argv

        if filename is None:
            if len(cmdln_arguments) >= 2:
                # If command line arguments were passed
                self._command_line_init(cmdln_arguments)
            else:
                # Interactive selection and pop up window:
                self._interactive_init()

        self.filename = Path(self.filename)

        self.channelnames = channelnames

        "Names of the imaging channels (optional)"
        self.watchfiles = watchfiles

        file_extension = self.filename.suffix

        if self.use_bioformats:
            import bioformats
            import javabridge

            javabridge.start_vm(class_path=bioformats.JARS)
            self.filetype = file_extension.lower()[1:]
            self.filehandle = bioformats.ImageReader(str(self.filename))
            md = bioformats.OMEXML(
                bioformats.get_omexml_metadata(path=str(self.filename))
            )
            self.positions = md.get_image_count()
            self.timepoints = md.image(
                0
            ).Pixels.SizeT  # Here I'm going to assume all series have the same number of timepoints
            self.channels = md.image(0).Pixels.channel_count
            self.x = md.image(0).Pixels.SizeX
            self.y = md.image(0).Pixels.SizeY
            # Get first image to get datatype (there's probably a better way to do this...)
            self.dtype = self.filehandle.read(rescale=False, c=0).dtype

        elif self.filename.is_dir():
            # Experiment is stored as individual image TIFF files in a folder
            self.filetype = "dir"
            self.filehandle = self.filename
            # If filename prototype is not provided, guess it from the first file:
            if self.prototype is None:
                imgfiles = [
                    x
                    for x in self.filename.iterdir()
                    if x.suffix.lower() in (".tif", ".tiff")
                ]
                # Here we assume all images in the folder follow the same naming convention:
                # Get digits sequences in first filename
                numstrs = re.findall(r"\d+", imgfiles[0].name)
                # Get character sequences in first filename
                charstrs = re.findall(r"\D+", imgfiles[0].name)
                if len(numstrs) != 3 or len(charstrs) != 4:
                    raise ValueError(
                        "Filename formatting error. See documentation for image sequence formatting"
                    )
                # Create the string prototype to be used to generate filenames on the fly:
                # Order is position, channel, frame/timepoint
                self.prototype = (
                    f"{charstrs[0]}%0{len(numstrs[0])}d"
                    f"{charstrs[1]}%0{len(numstrs[1])}d"
                    f"{charstrs[2]}%0{len(numstrs[2])}d"
                    f"{charstrs[3]}"
                )
            # Get experiment settings by testing if relevant files exist:
            # Get number of positions:
            if "p" in self.fileorder:
                self.positions = 0
                while self.getfilenamefromprototype(self.positions, 0, 0).exists():
                    self.positions += 1
            else:
                self.positions = 1
            # Get number of channels:
            if "c" in self.fileorder:
                self.channels = 0
                while self.getfilenamefromprototype(0, self.channels, 0).exists():
                    self.channels += 1
            else:
                self.channels = 1
            # Get number of frames/timepoints:
            if "t" in self.fileorder:
                self.timepoints = 0
                while self.getfilenamefromprototype(0, 0, self.timepoints).exists():
                    self.timepoints += 1
            else:
                self.timepoints = 1  # I guess this shouldn't really happen
            # Get image specs:
            if self.watchfiles:
                # Start file watcher thread:
                self.watcher = files_watcher(self)
                self.watcher.start()
            else:
                # Load first image, get image data from it
                I = cv2.imread(
                    str(self.getfilenamefromprototype(0, 0, 0)), cv2.IMREAD_ANYDEPTH
                )
                self.x = I.shape[1]
                self.y = I.shape[0]
                self.dtype = I.dtype

        elif file_extension.lower() == ".tif" or file_extension.lower() == ".tiff":
            # Works with single-series tif & mutli-series ome.tif
            from skimage.external.tifffile import TiffFile

            self.filetype = "tif"

            self.filehandle = TiffFile(str(self.filename))
            self.positions = len(self.filehandle.series)
            s = self.filehandle.series[
                0
            ]  # Here I'm going to assume all series have the same format
            self.timepoints = s.shape[s.axes.find("T")]
            self.channels = s.shape[s.axes.find("C")]
            self.x = s.shape[s.axes.find("X")]
            self.y = s.shape[s.axes.find("Y")]
            self.dtype = s.pages[0].asarray().dtype

    def _command_line_init(self, cmdln_arguments: List[str]) -> None:
        """
        Initialization routine if command line arguments were passed

        Parameters
        ----------
        cmdln_arguments : list of str
            List of command line arguments that were passed.

        Returns
        -------
        None.

        """

        self.filename = Path(cmdln_arguments[1])

        if importlib.util.find_spec("bioformats"):
            import bioformats

            self.use_bioformats = (
                True
                if self.filename.suffix[1:] in bioformats.READABLE_FORMATS
                else False
            )
        i = 2
        while i < len(cmdln_arguments):
            if cmdln_arguments[i] == "--bio-formats":
                self.use_bioformats = bool(int(cmdln_arguments[i + 1]))
                i += 2
            elif cmdln_arguments[i] == "--order":
                self.fileorder = cmdln_arguments[i + 1]
                i += 2
            elif cmdln_arguments[i] == "--index":
                self.filenamesindexing = int(cmdln_arguments[i + 1])
                i += 2
            elif cmdln_arguments[i] == "--proto":
                self.prototype = cmdln_arguments[i + 1]
                i += 2
            elif cmdln_arguments[i] == "--resfolder":
                # Resfolder will be read in pipeline
                i += 2
            else:
                print(f"Not valid argument {cmdln_arguments[i]}")
                i += 2

    def _interactive_init(self) -> None:
        """
        Interactive initialization routine.

        Raises
        ------
        ValueError
            If a non-valid experiment type was passed.

        Returns
        -------
        None.

        """

        # Get xp settings:
        print(
            (
                "Experiment type?\n"
                "1 - Bio-Formats compatible (.nd2, .oib, .czi, .ome.tif...)\n"
                "2 - bioformats2sequence (folder)\n"
                "3 - micromanager (folder)\n"
                "4 - high-throughput (folder)\n"
                "0 - other (folder)\n"
                "Enter a number: "
            ),
            end="",
        )
        answer = int(input())
        print()

        # If bioformats file(s):
        if answer is None or answer == 1:
            print("Please select experiment file(s)...")
            self.filename = getxppathdialog(ask_folder=False)
            self.use_bioformats = True
            self.prototype = None
            self.filenamesindexing = 1
            self.fileorder = "pct"

        # If folder:
        else:
            print("Please select experiment folder...")
            self.filename = getxppathdialog(ask_folder=True)
            self.use_bioformats = False
            if answer is None or answer == 2:
                self.prototype = None
                self.fileorder = "pct"
                self.filenamesindexing = 1
            elif answer == 3:
                self.prototype = (
                    "Pos%01d/img_channel%03d_position%03d_time%09d_z000.tif"
                )
                self.fileorder = "pcpt"
                self.filenamesindexing = 0
            elif answer == 4:
                self.prototype = "chan%02d_img/Position%06d_Frame%06d.tif"
                self.fileorder = "cpt"
                self.filenamesindexing = 1
            elif answer == 0:
                print("Enter files prototype: ", end="")
                self.prototype = input()
                print()
                print("Enter files order: ", end="")
                self.fileorder = input()
                print()
                print("Enter files indexing: ", end="")
                self.filenamesindexing = int(input())
                print()
            else:
                raise ValueError("Invalid experiment type")
            print()

    def close(self) -> None:
        # Close bioformats or tif reader
        if self.use_bioformats:
            self.filehandle.close()
            import javabridge

            javabridge.kill_vm()
        elif self.filetype == "tif":  # Nothing to do if sequence directory
            self.filehandle.close()

    def getfilenamefromprototype(self, position: int, channel: int, frame: int) -> Path:
        """
        Generate full filename for specific frame based on file path,
        prototype, fileorder, and filenamesindexing

        Parameters
        ----------
        position : int
            Position/series index (0-based indexing).
        channel : int
            Imaging channel index (0-based indexing).
        frame : int
            Frame/timepoint index (0-based indexing).

        Returns
        -------
        string
            Filename.

        """
        filenumbers = tuple(
            {"p": position, "c": channel, "t": frame}[c] + self.filenamesindexing
            for c in self.fileorder
        )
        assert self.prototype is not None  # FIXME: why is it not?
        return self.filehandle / (self.prototype % filenumbers)

    # TODO: typecheck this function
    def getframes(
        self,
        positions=None,
        channels=None,
        frames=None,
        squeeze_dimensions: bool = True,
        resize: Tuple[int, int] = None,
        rescale: Tuple[float, float] = None,
        globalrescale: Tuple[float, float] = None,
        rotate: float = None,
    ) -> np.ndarray:
        """
        Get frames from experiment.

        Parameters
        ----------
        positions : None, int, tuple/list of ints, optional
            The frames from the position index or indexes passed as an integer
            or a tuple/list will be returned. If None is passed, all positions
            are returned.
            The default is None.
        channels : None, int, tuple/list of ints, str, tuple/list of str, optional
            The frames from the channel index or indexes passed as an integer
            or a tuple/list will be returned. If the channel names have been
            defined, the channel(s) can be passed as a string or tuple/list of
            strings. If an empty list is passed, None is returned. If None is
            passed, all channels are returned.
            The default is None.
        frames : None, int, tuple/list of ints, optional
            The frame index or indexes passed as an integer or a tuple/list
            will be returned. If None is passed, all frames are returned. If -1
            is passed and the file watcher is activated, only new frames are
            read. Works only for one position and channel at a time.
            The default is None.
        squeeze_dimensions : bool, optional
            If True, the numpy squeeze function is applied to the output array,
            removing all singleton dimensions.
            The default is True.
        resize : None or tuple/list of 2 ints, optional
            Dimensions to resize the frames. If None, no resizing is performed.
            The default is None.
        rescale : None or tuple/list of 2 int/floats, optional
            Rescale all values in each frame to be within the given range.
            The default is None.
        globalrescale : None or tuple/list of 2 int/floats, optional
            Rescale all values in all frames to be within the given range.
            The default is None.
        rotate : None or float, optional
            Rotation to apply to the image (in degrees).
            The default is None.

        Raises
        ------
        ValueError
            If channel names are not correct.

        Returns
        -------
        Numpy Array
            Concatenated frames as requested by the different input options.
            If squeeze_dimensions=False, the array is 5-dimensional, with the
            dimensions order being: Position, Time, Channel, Y, X

        """
        # Handle options:
        if positions is None:  # No positions specified: load all
            positions = list(range(self.positions))
        elif not isinstance(positions, list) and not isinstance(positions, tuple):
            positions = [positions]

        if channels is None:  # No channel specified: load all
            channels = list(range(self.channels))
        elif isinstance(channels, str):  # refer to 1 channel by name (eg 'gfp')
            if self.channelnames is None:
                raise ValueError("Set channel names first")
            if channels in self.channelnames:
                channels = [self.channelnames.index(channels)]
            else:
                raise ValueError(channels + " is not a valid channel name.")
        elif isinstance(channels, list) or isinstance(channels, tuple):
            # If list of ints, nothing to do
            if all(isinstance(c, str) for c in channels):
                # refer to channels by list/tuple of names:
                if self.channelnames is None:
                    raise ValueError("Set channel names first")
                channels_list = []
                for channel in channels:
                    if channel in self.channelnames:
                        channels_list.append(self.channelnames.index(channel))
                    else:
                        raise ValueError(channel + " is not a valid channel name.")
                channels = channels_list
        elif not isinstance(channels, list) and not isinstance(channels, tuple):
            channels = [channels]

        if frames is None:  # No frames specfied: load all
            frames = list(range(self.timepoints))
        elif type(frames) is not list and type(frames) is not tuple:
            if frames == -1:  # Read new frames from files watcher:
                if self.watchfiles:
                    if len(positions) > 1 or len(channels) > 1:
                        raise ValueError(
                            "Can not load latest frames for more than one position/channel"
                        )
                    new = self.watcher.new[positions[0]][channels[0]]
                    frames = list(
                        range(self.watcher.old[positions[0]][channels[0]] + 1, new + 1)
                    )
                else:
                    frames = list(range(self.timepoints))
            else:
                frames = [frames]

        # If files watcher, update the old frames array:
        if self.watchfiles:
            for p in positions:
                for c in channels:
                    self.watcher.old[p][c] = frames[-1]

        # Allocate memory:
        dt: Union[str, type] = self.dtype if rescale is None else np.float32
        if resize is None:
            output = np.empty(
                [len(positions), len(frames), len(channels), self.y, self.x], dtype=dt
            )
        else:
            output = np.empty(
                [len(positions), len(frames), len(channels), resize[0], resize[1]],
                dtype=dt,
            )

        # Load images:
        for p, pos in enumerate(positions):
            for c, cha in enumerate(channels):
                for f, fra in enumerate(frames):
                    # Read frame:
                    if self.use_bioformats:
                        frame = self.filehandle.read(
                            series=pos, c=cha, t=fra, rescale=False
                        )
                    elif self.filetype == "dir":
                        frame = cv2.imread(
                            str(self.getfilenamefromprototype(pos, cha, fra)),
                            cv2.IMREAD_ANYDEPTH,
                        )
                    elif self.filetype == "tif":
                        frame = (
                            self.filehandle.series[pos]
                            .pages[fra * self.channels + cha]
                            .asarray()
                        )

                    frame = frame.astype(np.uint16)

                    # Optionally resize and rescale:
                    if rotate is not None:
                        frame = imrotate(frame, rotate)
                    if resize is not None:
                        frame = cv2.resize(frame, resize[::-1])  # cv2 inverts shape
                    if rescale is not None:
                        frame = rangescale(frame, rescale)
                    # Add to output array:
                    output[p, f, c, :, :] = frame

        # Rescale all images:
        if globalrescale is not None:
            output = rangescale(output, globalrescale)

        # Return:
        return np.squeeze(output) if squeeze_dimensions else output


class files_watcher(Thread):
    """
    Daemon to watch experiment files and signal new ones (not functional yet)
    """

    def __init__(self, reader):
        super().__init__()
        self.daemon = True
        self.reader = reader
        self.old = []
        self.new = []

    def run(self):
        while True:

            # Run through monitored positions:
            for p in range(self.reader.positions):

                if len(self.old) <= p:  # Position added / not done yet
                    # Append empty lists to the end:
                    self.old += [[] for _ in range(p + 1 - len(self.old))]
                    self.new += [[] for _ in range(p + 1 - len(self.new))]

                # Run through monitored channels:
                for channel in range(self.reader.channels):

                    if len(self.old[p]) <= channel:
                        # Append empty lists to the end:
                        self.old[p] += [
                            -1 for _ in range(channel + 1 - len(self.old[p]))
                        ]
                        self.new[p] += [-1] * (channel + 1 - len(self.new[p]))

                    # Check if new files have been written:
                    i = self.new[p][channel]
                    while self.reader.getfilenamefromprototype(
                        p, channel, i + 1
                    ).exists():
                        i += 1

                    # Store new timepoint number:
                    self.new[p][channel] = i
            time.sleep(0.01)

    def newfiles(self):
        """
        Get list of new files position and channel

        Returns
        -------
        newfiles : list of tuples of 2 ints
            List containing all new files position and channel.

        """

        newfiles = []
        for p, pos_old in enumerate(self.old):
            for c, latest_read in enumerate(pos_old):
                if latest_read < self.new[p][c]:
                    newfiles += [(p, c)]
        return newfiles


#%% Saving & Loading results


def loadmodels(toload: Tuple[str, ...] = None) -> Dict[str, Any]:
    """
    Load models (as specified in config.py)

    Parameters
    ----------
    toload : tuple of str, or None, optional
        Which of the 3 models to load. If None, cfg.models will be used.
        The default is None.

    Returns
    -------
    models : dict
        Dictionary containing the models specified.

    """

    import tensorflow as tf

    if toload is None:
        toload = cfg.models

    models = dict()

    if "rois" in toload:
        models["rois"] = tf.keras.models.load_model(cfg.model_file_rois, compile=False)

    if "segmentation" in toload:
        models["segmentation"] = tf.keras.models.load_model(
            cfg.model_file_seg, compile=False
        )

    if "tracking" in toload:
        models["tracking"] = tf.keras.models.load_model(
            cfg.model_file_track, compile=False
        )

    return models


def getrandomcolors(num_colors: int, seed: int = 0) -> List[Tuple[float, float, float]]:
    """
    Pseudo-randomly generate list of random hsv colors.

    Parameters
    ----------
    num_colors : int
        Number of desired colors.
    seed : None or int, optional
        Seed to use with numpy.random.seed().
        The default is 0.

    Returns
    -------
    colors : list
        List of RGB values (0-1 interval).

    """

    if num_colors == 0:
        return []

    # Get colors:
    colors = (
        cv2.applyColorMap(
            np.linspace(0, 256, num_colors, endpoint=False, dtype=np.uint8),
            cv2.COLORMAP_HSV,
        ).astype(np.float64)
        / 255
    )
    colors = [tuple(x) for x in colors[:, 0]]

    # Pseudo randomly shuffle colors:
    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(colors)

    return colors


def vidwrite(
    images: np.ndarray, filename: Union[str, Path], crf: int = 20, verbose: int = 1
) -> None:
    """
    Write images stack to video file with h264 compression.

    Parameters
    ----------
    images : 4D numpy array
        Stack of RGB images to write to video file.
    filename : str or Path
        File name to write video to. (Overwritten if exists)
    crf : int, optional
        Compression rate. 'Sane' values are 17-28. See
        https://trac.ffmpeg.org/wiki/Encode/H.264
        The default is 20.
    verbose : int, optional
        Verbosity of console output.
        The default is 1.

    Returns
    -------
    None.

    """

    # Initialize ffmpeg parameters:
    height, width, _ = images[0].shape
    if height % 2 == 1:
        height -= 1
    if width % 2 == 1:
        width -= 1
    quiet = [] if verbose else ["-loglevel", "error", "-hide_banner"]
    process = (
        ffmpeg.input(
            "pipe:",
            format="rawvideo",
            pix_fmt="rgb24",
            s="{}x{}".format(width, height),
            r=7,
        )
        .output(
            str(filename),
            pix_fmt="yuv420p",
            vcodec="libx264",
            crf=crf,
            preset="veryslow",
        )
        .global_args(*quiet)
        .overwrite_output()
        .run_async(pipe_stdin=True)
    )

    # Write frames:
    for frame in images:
        process.stdin.write(frame[:height, :width].astype(np.uint8).tobytes())

    # Close file stream:
    process.stdin.close()

    # Wait for processing + close to complete:
    process.wait()


#%% Feature extraction


def roi_features(
    labels_frame: npt.NDArray[np.uint16],
    features: Tuple[str, ...] = ("length", "width", "area", "perimeter", "edges"),
    fluo_frames: np.ndarray = None,
    roi_box: CroppingBox = None,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Extract single-cell morphological and fluorescence features

    Parameters
    ----------
    labels_frame : 2D array of uint16
        Labels image of numbered cell regions.
    features : list of str, optional
        Features to extract. Valid features are 'length','width','area',
        'perimeter','edges','fluo1','fluo2',...,'fluoN'.
        The default is ('length','width','area','perimeter','edges').
    fluo_frames : 3D array, optional
        Array of fluo frames to extract fluorescence from. Dimensions are
        (size_x, size_y, fluo_channels). The number of fluo_channels must match
        the number of fluo features to extract.
        The default is None.
    roi_box : dict, optional
        Cropping box dictionary (see cropbox())to apply shift when looking at
        full-frame fluo images.
        The default is None.

    Returns
    -------
    cell_nbs : list of ints
        IDs of cells present in the labels frame
    cell_features : list of dict
        List of dictionary with values for requested features.

    """

    # Cell numbers and contours:
    cell_nbs, contours = getcellsinframe(labels_frame, return_contours=True)
    assert isinstance(cell_nbs, list)  # needed for mypy on Python < 3.8
    assert isinstance(contours, list)  # needed for mypy on Python < 3.8

    # Loop through cells in image, extract single-cell features:
    cell_features = []
    for cell, contour in zip(cell_nbs, contours):

        # Append to features list:
        cell_features += [
            singlecell_features(
                contour,
                labels_frame == cell + 1,
                features=features,
                roi_box=roi_box,
                fluo_frames=fluo_frames,
            )
        ]

    return cell_nbs, cell_features


def singlecell_features(
    contour: Contour,
    mask: SegmentationMask,
    features: Tuple[str, ...] = ("length", "width", "area", "perimeter", "edges"),
    fluo_frames: np.ndarray = None,
    roi_box: CroppingBox = None,
) -> Dict[str, Any]:
    """
    Extract features for a single cell

    Parameters
    ----------
    contour : list
        Single cell contour from cv2 findcontours.
    mask : 2D numpy array of bool
        Mask of the region to extract (typically a single cell).
    features : list of str, optional
        Features to extract. Valid features are 'length','width','area',
        'perimeter','edges','fluo1','fluo2',...,'fluoN'.
        The default is ('length','width','area','perimeter','edges').
    fluo_frames : 3D array, optional
        Fluorescent images to extract fluo from. Dimensions are
        (channels, size_y, size_x).
        The default is None.
    roi_box : dict, optional
        Cropping box dictionary (see cropbox())to apply shift when looking at
        full-frame fluo images.
        The default is None.

    Raises
    ------
    ValueError
        If the number of fluo frames provided is different from the number of
        fluo features to extract.

    Returns
    -------
    features_dict : dict
        Dictionary of requested features.

    """

    features_dict: Dict[str, Any] = dict()

    # Morphological features:
    if "edges" in features:
        features_dict["edges"] = image_edges(contour, mask)

    if "length" in features or "width" in features:
        width, length = cell_width_length(contour)
        if "length" in features:
            features_dict["length"] = length
        if "width" in features:
            features_dict["width"] = width

    if "area" in features:
        features_dict["area"] = cell_area(contour)

    if "perimeter" in features:
        features_dict["perimeter"] = cell_perimeter(contour)

    # Fluo features:
    fluo_features = [x for x in features if x[0:4] == "fluo"]
    if len(fluo_features) > 0:
        if fluo_frames is None:
            raise ValueError("You did not provide fluo frames")
        if len(fluo_features) != fluo_frames.shape[0]:
            raise ValueError(
                "You should provide exactly as many fluo frames as you want "
                f"fluo features to extract. Here fluo_frames.shape[0]={fluo_frames.shape[0]} "
                f"but you ask for {len(fluo_features)} fluo features: {fluo_features}"
            )
        fluo_values = cell_fluo(fluo_frames, mask, roi_box)
        for fluo_feat, fluo_val in zip(fluo_features, fluo_values):
            features_dict[fluo_feat] = fluo_val

    return features_dict


def image_edges(contour: Contour, image: npt.NDArray[Any]) -> str:
    """
    Identify if cell touches image borders

    Parameters
    ----------
    contour : list
        Single cell contour from cv2 findcontours.
    image : 2D numpy array
        Image where the cell is present.

    Returns
    -------
    edge_str : str
        String describing edges touched by the cell. Can be a combination of
        the following strs: '-x', '+x', '-y', '+y'. Empty otherwise.

    """

    edge_str = ""
    if any(contour[:, 0, 0] == 0):
        edge_str += "-x"
    if any(contour[:, 0, 0] == image.shape[1] - 1):
        edge_str += "+x"
    if any(contour[:, 0, 1] == 0):
        edge_str += "-y"
    if any(contour[:, 0, 1] == image.shape[0] - 1):
        edge_str += "+y"

    return edge_str


def cell_width_length(contour: Contour) -> Tuple[float, float]:
    """
    Mesure width and length of single cell

    Parameters
    ----------
    contour : list
        Single cell contour from cv2 findcontours.

    Returns
    -------
    width : float
        Cell width.
    length : float
        cell length.

    """

    # Get rotated bounding box:
    _, size, _ = cv2.minAreaRect(contour)

    width = min(size)
    length = max(size)

    return width, length


def cell_area(contour: Contour) -> float:
    """
    Area of a single cell

    Parameters
    ----------
    contour : list
        Single cell contour from cv2 findcontours.

    Returns
    -------
    area : float
        Cell area

    """

    area = cv2.contourArea(contour)

    return area


def cell_perimeter(contour: Contour) -> float:
    """
    Get single cell perimeter

    Parameters
    ----------
    contour : list
        Single cell contour from cv2 findcontours.

    Returns
    -------
    perimeter : int
        Cell perimeter

    """

    perimeter = cv2.arcLength(contour, closed=True)

    return perimeter


def cell_fluo(
    fluo_frames: np.ndarray, mask: SegmentationMask, roi_box: CroppingBox = None
) -> List[float]:
    """
    Extract mean fluorescence level from mask

    Parameters
    ----------
    fluo_frames : 3D array
        Fluorescent images to extract fluo from. Dimensions are
        (channels, size_y, size_x).
    mask : 2D numpy array of bool
        Mask of the region to extract (typically a single cell).
    roi_box : dictionary, optional
        Cropping box dictionary (see cropbox())to apply shift when looking at
        full-frame fluo images.
        The default is None.

    Returns
    -------
    fluo_values :  list of floats
        Mean value per cell for each fluo frame.

    """

    # Pixels where the cell is:
    pixels = np.where(mask)
    if roi_box is not None:
        _pixels = list(pixels)
        _pixels[0] += roi_box["ytl"]
        _pixels[1] += roi_box["xtl"]
        pixels = tuple(_pixels)

    # Loop through fluo frames:
    fluo_values = []
    for frame in range(fluo_frames.shape[0]):
        fluo_values += [np.mean(fluo_frames[frame, pixels[0], pixels[1]])]

    return fluo_values


#%% Misc


def findfirst(mylist: List) -> Optional[int]:
    """
    Find first non-zero element of list

    Parameters
    ----------
    mylist : list
        List of elements to scan through.

    Returns
    -------
    int or None
        Index of first non-zero element, if it exists.

    """
    return next((i for i, x in enumerate(mylist) if x > 0), None)
