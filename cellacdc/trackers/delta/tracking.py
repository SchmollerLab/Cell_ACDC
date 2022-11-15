"""
Tracking processing pipeline.

@author: jroberts / jamesr787
"""

# Modules:
from typing import cast, List, Optional, Tuple

import cv2
import numpy as np

from delta import utilities as utils
from delta.utilities import cfg


class ROI:

    def __init__(self,
                 segm_video,
                 original_video,
                 roi_nb: int,
                 box: utils.CroppingBox,
                 crop_windows: bool = False):
        """
        Initialize ROI

        Parameters
        ----------
        roi_nb : int
            ROI index.
        box : dict
            Crop box for ROI, formatted as in the utilities.py cropbox
            function input dict.
        crop_windows : bool, optional
            Flag to crop and stitch back windows for segmentation and tracking.
            The default is False.

        Returns
        -------
        None.

        """
        self.roi_nb = roi_nb
        "The ROI index number"
        self.box = box
        "ROI crop box"

        self.img_stack: List[utils.Image] = []
        "Input images stack"
        # Preprocess Images
        for frame in range(len(original_video)):
            # Crop and scale:
            i = utils.rangescale(utils.cropbox(original_video[frame], self.box), rescale=(0, 1))
            # Append i as is to input images stack:
            self.img_stack.append(i)

        self.seg_stack: List[utils.SegmentationMask] = segm_video
        "Segmentation images stack"
        self.lineage = utils.Lineage()
        "Lineage object for ROI"
        self.label_stack: List[Optional[utils.Labels]] = []
        "Labelled images stack"
        self.crop_windows: bool = crop_windows
        "Flag to crop overlapping windows for segmentation"

        if crop_windows:
            self.scaling = None
        else:
            self.scaling = (
                (box["ybr"] - box["ytl"]) / cfg.target_size_seg[0],
                (box["xbr"] - box["xtl"]) / cfg.target_size_seg[1],
            )

    def get_tracking_inputs(self, frame: int = None):
        """
        Compile tracking inputs for ROI

        Parameters
        ----------
        frame : int, optional
            The frame to compile for. If None, the earliest frame not yet
            tracked is run.
            The default is None.

        Raises
        ------
        RuntimeError
            Segmentation has not been completed up to frame yet.

        Returns
        -------
        x : 4D array or None
            Tracking input array. Dimensions are (previous_cells,
            cfg.target_size_track[1], cfg.target_size_track[0], 4). If no
            previous cells to track from (e.g. first frame or glitch), None is
            returned.
        boxes : List of tuples of 2 dicts or None
            Crop and fill boxes to re-place outputs in the ROI

        """

        # If no frame number passed, run latest:
        if frame is None:
            frame = len(self.lineage.cellnumbers)

        # Check if segmentation data is ready:
        if len(self.seg_stack) <= frame:
            raise RuntimeError(f"Segmentation incomplete - frame {frame}")

        # If no previous cells to track from, directly update lineage:
        if (
            frame == 0
            or len(self.lineage.cellnumbers) < frame
            or len(self.lineage.cellnumbers[frame - 1]) == 0
        ):

            # Cell poles
            poles = utils.getpoles(self.seg_stack[frame], scaling=cfg.target_size_track)

            # Create new orphan cells
            for c in range(len(poles)):
                self.lineage.update(None, frame, attrib=[c], poles=[poles[c]])

            return None, None

        # Otherwise, get cell contours:
        cells = utils.find_contours(self.seg_stack[frame - 1])
        cells.sort(key=lambda elem: np.max(elem[:, 0, 1]))  # Sorting along Y

        # Allocate empty tracking inputs array:
        x = np.empty(
            shape=(len(cells),) + cfg.target_size_track + (4,), dtype=np.float32
        )

        # Run through contours and compile inputs:
        boxes = []
        for c, cell in enumerate(cells):

            draw_offset: Optional[Tuple[float, float]]
            if self.crop_windows:
                curr_img = self.img_stack[frame]
                prev_img = self.img_stack[frame - 1]
                # Cell-centered crop boxes:
                shape = cast(Tuple[int, int], curr_img.shape)  # for mypy
                cb, fb = utils.gettrackingboxes(cell, shape)
                draw_offset = (-cb["xtl"] + fb["xtl"], -cb["ytl"] + fb["ytl"])
            else:
                curr_img = cv2.resize(
                    self.img_stack[frame], dsize=cfg.target_size_seg[::-1]
                )
                prev_img = cv2.resize(
                    self.img_stack[frame - 1], dsize=cfg.target_size_seg[::-1]
                )
                cb = fb = utils.CroppingBox(ytl=None, xtl=None, ybr=None, xbr=None)
                draw_offset = None
            boxes += [(cb, fb)]

            # Current image
            x[c, fb["ytl"] : fb["ybr"], fb["xtl"] : fb["xbr"], 0] = utils.cropbox(
                curr_img, cb
            )

            # Segmentation mask of one previous cell (seed)
            x[c, :, :, 1] = cv2.drawContours(
                np.zeros(cfg.target_size_track, dtype=np.float32),
                [cell],
                0,
                offset=draw_offset,
                color=1.0,
                thickness=-1,
            )

            # Previous image
            x[c, fb["ytl"] : fb["ybr"], fb["xtl"] : fb["xbr"], 2] = utils.cropbox(
                prev_img, cb
            )

            # Segmentation of all current cells
            x[c, fb["ytl"] : fb["ybr"], fb["xtl"] : fb["xbr"], 3] = utils.cropbox(
                self.seg_stack[frame], cb
            )

        # Return tracking inputs and crop and fill boxes:
        return x, boxes

    def process_tracking_outputs(
        self,
        y: np.ndarray,
        frame: int = None,
        boxes: List[Tuple[utils.CroppingBox, utils.CroppingBox]] = None,
    ):
        """
        Process output from tracking U-Net

        Parameters
        ----------
        y : 4D array
            Tracking output array. Dimensions are (previous_cells,
            cfg.target_size_track[1], cfg.target_size_track[0], 1).
        frame : int, optional
            The frame to process for. If None, the earliest frame not yet
            tracked is run.
            The default is None.
        boxes : List of tuples of 2 dicts or None
            Crop and fill boxes to re-place outputs in the ROI

        Returns
        -------
        None.

        """

        if frame is None:
            frame = len(self.lineage.cellnumbers)

        # Get scores and attributions:
        labels = utils.label_seg(self.seg_stack[frame])
        assert isinstance(labels, np.ndarray)  # needed for mypy on Python < 3.8
        scores = utils.getTrackingScores(
            labels, y[:, :, :, 0], boxes=boxes if self.crop_windows else None
        )
        if scores is None:
            self.lineage.update(None, frame)
            return
        attributions = utils.getAttributions(scores)

        # Get poles:
        poles = utils.getpoles(self.seg_stack[frame], labels, scaling=self.scaling)

        # Update lineage:
        # Go through old cells:
        for o in range(attributions.shape[0]):
            attrib = attributions[o, :].nonzero()[0]
            new_cells_poles = []
            for n in attrib:
                new_cells_poles += [poles[n]]
            self.lineage.update(o, frame, attrib=attrib, poles=new_cells_poles)
        # Go through "orphan" cells:
        for n in range(attributions.shape[1]):
            attrib = attributions[:, n].nonzero()[0]
            new_cells_poles = [poles[n]]
            if len(attrib) == 0:
                self.lineage.update(None, frame, attrib=[n], poles=new_cells_poles)

    def extract_features(
        self,
        frame: int = None,
        fluo_frames: np.ndarray = None,
        features: Tuple[str, ...] = (
            "length",
            "width",
            "area",
            "perimeter",
            "edges",
        ),
    ):
        """
        Extract single cell features

        Parameters
        ----------
        frame : int, optional
            The frame to extract for. If None, the earliest frame not yet
            extracted is run.
            The default is None.
        fluo_frames : 3D array, optional
            Fluorescent images to extract fluo from. Dimensions are
            (channels, size_y, size_x).
            The default is None.
        features : list or tuple of str, optional
            Features to extract. Options are ("length", "width", "area", "fluo1",
            "fluo2", "fluo3"...)
            The default is ("length", "width", "area", "perimeter", "edges").

        Returns
        -------
        None.

        """

        # Default frame:
        if frame is None:
            frame = len(self.label_stack)

        # Add Nones to label stack list if not long enough:
        if len(self.label_stack) <= frame:
            self.label_stack += [None] * (frame + 1 - len(self.label_stack))

        # Compile labels frame:
        if self.label_stack[frame] is None:
            if len(self.lineage.cellnumbers) <= frame:
                cell_nbs = []
            else:
                cell_nbs = [c + 1 for c in self.lineage.cellnumbers[frame]]
            labels = utils.label_seg(self.seg_stack[frame], cell_nbs)
            assert isinstance(labels, np.ndarray)  # needed for mypy on Python < 3.8

            if self.crop_windows:
                self.label_stack[frame] = labels
            else:
                resize = (
                    self.box["xbr"] - self.box["xtl"],
                    self.box["ybr"] - self.box["ytl"],
                )
                self.label_stack[frame] = cv2.resize(
                    labels, resize, interpolation=cv2.INTER_NEAREST
                )

        if features is not None:
            fr = self.label_stack[frame]
            assert fr is not None  # for mypy
            # Extract features for all cells in the ROI:
            cells, extracted_features = utils.roi_features(
                fr,
                features=features,
                fluo_frames=fluo_frames,
                roi_box=self.box,
            )

            # Add feature values to lineage for each cell:
            for cell, cell_features in zip(cells, extracted_features):
                for feature_name, feature_val in cell_features.items():
                    self.lineage.setvalue(cell, frame, feature_name, feature_val)
