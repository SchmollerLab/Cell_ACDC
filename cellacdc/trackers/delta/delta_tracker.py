"""
This script runs the tracking U-Net
for 2D images.

@author: jroberts / jamesr787
"""

# Modules:
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from scipy.io import savemat
from typing import Dict, Any, List

import delta.utilities as utils
import delta.config as cfg
from delta.assets import download_assets
from delta.model import unet_track

from . import tracking


class tracker:

    def __init__(self, **params):
        """
        Initializes Tracker

        Parameters
        ----------
        params : dict
            model_type (2D or mothermachine),
            original_images_path,
            and verbose (0 or 1).

        Returns
        -------
        None.

        """
        self.rois = None
        self.params = params
        self.drift_correction = False
        self.drift_values = None
        self.rotate = 0

    def __read_tiff(self,
                    path):
        """
        Reads multipage tiff to numpy array.

        Parameters
        ----------
        path : string
            Path to tiff file.

        Returns
        -------
        images : np.array
            Images as 3D numpy array.

        """
        img = Image.open(path)
        images = []
        for i in range(img.n_frames):
            img.seek(i)
            images.append(np.array(img))
        return np.array(images)

    def __load_model_and_presets(self,
                                 model_type):
        """
        Loads Presets for 2D or mothermachine, initializes model for tracking
        and loads model weights.

        Parameters
        ----------
        model_type : string
            Either '2D' or 'mothermachine'.

        Raises
        ------
        ValueError
            Configuration files or models have not yet been downloaded.
            Downloads the configuration files and model weights.

        Returns
        -------
        None.

        """
        # Load Presets
        while True:
            try:
                # Loads Presets images
                cfg.load_config(presets=model_type)

                # Initialize Model and Load Weights with presets
                self.model = unet_track(input_size=cfg.target_size_track + (4,))
                self.model.load_weights(cfg.model_file_track)

                break

            except ValueError:
                # Downloads model weights and configuration files for 2D and mothermachine
                download_assets(load_models=True,
                                load_sets=False,
                                load_evals=False,
                                config_level='local')

    def track(self,
              segm_video,
              signals=None,
              export_to: os.PathLike=None):
        """
        Tracks Cells

        Parameters
        ----------
        segm_video : np.array
            3D numpy array of segmentation mask.

        Returns
        -------
        tracked_video : np.array
            3D numpy array of tracked and labeled video.
        """

        # Loads Presets and Initializes Model
        self.__load_model_and_presets(model_type=self.params['model_type'])

        # Original Shape
        original_shape = segm_video[0].shape

        # Get original video and original image size
        original_video = self.__read_tiff(self.params['original_images_path'])
        reference = utils.rangescale(original_video[0], (0, 1))

        # Preprocess Segmentation Video
        if not cfg.crop_windows:
            img_stack_sm = np.zeros((len(segm_video),
                                    cfg.target_size_seg[0],
                                    cfg.target_size_seg[1]))
        else:
            img_stack_sm = np.zeros((len(segm_video),
                                     original_shape[0],
                                     original_shape[1]))
        for idx in range(len(segm_video)):
            img = segm_video[idx, :, :]
            if not cfg.crop_windows:
                img = cv2.resize(img, cfg.target_size_seg[::-1])
            img_sm = (img > 0.5).astype(np.uint8)
            if cfg.crop_windows:
                img_sm = img_sm[: original_shape[0], : original_shape[1]]
            img_stack_sm[idx, :, :] = img_sm
        segm_video = img_stack_sm.astype(np.uint8)

        # Instantiate ROI
        self.rois = [
            tracking.ROI(
                roi_nb=0,
                box=utils.CroppingBox(
                    xtl=0, ytl=0,
                    xbr=reference.shape[1], ybr=reference.shape[0],
                ),
                crop_windows=cfg.crop_windows,
                segm_video=segm_video,
                original_video=original_video
            )
        ]

        for frame in tqdm(range(len(segm_video))):

            # Compile inputs and references:
            x = []
            references = []
            for r, roi in enumerate(self.rois):
                inputs, boxes = roi.get_tracking_inputs(frame=frame)
                if inputs is not None:
                    x += [inputs]
                    references += [(r, len(x[-1]), frame, boxes)]

            # Predict:
            if len(x) > 0:
                x = np.concatenate(x)
                y = np.empty_like(x)
                for i in range(0, len(x), cfg.pipeline_track_batch):
                    j = min((len(x), i + cfg.pipeline_track_batch))
                    y[i:j] = self.model.predict(
                        x[i:j],
                        batch_size=1,
                        workers=1,
                        use_multiprocessing=False,
                        verbose=self.params['verbose'],
                    )

            # Dispatch tracking outputs to rois:
            i = 0
            for ref in references:
                self.rois[ref[0]].process_tracking_outputs(
                    y=y[i: i + ref[1]],
                    frame=ref[2],
                    boxes=ref[3]
                )
                i += ref[1]

        # Run through frames and update cells:
        for f in range(len(segm_video)):
            # Run through ROIs and extract features:
            for roi in self.rois:
                roi.extract_features(
                    frame=f,
                    fluo_frames=None,
                    features=tuple(["length", "width", "area", "perimeter", "edges"]),
                )

        # Get labels
        for r in self.rois:
            res: Dict[str, Any] = dict()

            # Resized labels stack: (ie original ROI size)
            res["labelsstack_resized"] = np.array(r.label_stack, dtype=np.uint16)
        tracked_video = res['labelsstack_resized']

        # Save Results
        if self.params['legacy'] or self.params['pickle'] or self.params['movie']:
            self.save_output(original_video=original_video)

        # Return tracked and labeled video
        return tracked_video

    def save_output(self,
                    original_video):
        """
        Save Results to disk

        Parameters
        ----------
        original_video : np.array
            3D numpy array.

        Returns
        -------
        None.

        """

        # Get Save Path (File Name is same as Original Images + .format)
        savepath = self.params['original_images_path']
        savepath = savepath.replace('.tif', '')

        if self.params['legacy']:
            self.legacysave(original_video=original_video,
                            res_file=savepath+".mat")

        if self.params['pickle']:
            import pickle

            with open(savepath+".pkl", "wb") as file:
                pickle.dump(self, file)

        if self.params['movie']:
            movie = self.results_movie(original_video=original_video,
                                       frames=list(range(len(original_video))))
            utils.vidwrite(movie, savepath+".mp4", verbose=False)

    def legacysave(self,
                   original_video,
                   res_file):
        """
        Save pipeline data in the legacy Matlab format

        Parameters
        ----------
        original_video : np.array
            3D numpy array.
        res_file : str or Path
            Path to save file.

        Returns
        -------
        None.

        """

        # File reader info
        moviedimensions = [
            original_video.shape[1],
            original_video.shape[2],
            1,
            original_video.shape[0],
        ]
        xpfile = res_file.split('/')[-1]

        # If No ROIs detected for position:
        if len(self.rois) == 0:
            savemat(
                res_file,
                {
                    "res": [],
                    "tiffile": str(xpfile),
                    "moviedimensions": moviedimensions,
                    "proc": {"rotation": self.rotate, "chambers": [], "XYdrift": []},
                },
            )
            return

        # Initialize data structure/dict:
        data = dict(moviedimensions=moviedimensions, tifffile=str(xpfile))

        # Proc dict/structure:
        data["proc"] = dict(
            rotation=self.rotate,
            XYdrift=np.array(self.drift_values, dtype=np.float64),
            chambers=np.array(
                [
                    [
                        r.box["xtl"],
                        r.box["ytl"],
                        r.box["xbr"] - r.box["xtl"],
                        r.box["ybr"] - r.box["ytl"],
                    ]
                    for r in self.rois
                ],
                dtype=np.float64,
            ),
        )

        # Lineages:
        data["res"] = []
        for r in self.rois:
            res: Dict[str, Any] = dict()

            # Resized labels stack: (ie original ROI size)
            res["labelsstack_resized"] = np.array(r.label_stack, dtype=np.uint16)

            # Not resized stack: (ie U-Net seg target size)
            label_stack = []
            for f, cellnbs in enumerate(r.lineage.cellnumbers):
                label_stack += [
                    utils.label_seg(r.seg_stack[f], [c + 1 for c in cellnbs])
                ]
            res["labelsstack"] = np.array(label_stack, dtype=np.uint16)

            # Run through cells, update to 1-based indexing
            cells = r.lineage.cells
            lin: List[Dict[str, Any]] = []
            for c in cells:
                lin += [dict()]
                # Base lineage:
                lin[-1]["mother"] = c["mother"] + 1 if c["mother"] is not None else 0
                lin[-1]["frames"] = np.array(c["frames"], dtype=np.float32) + 1
                lin[-1]["daughters"] = np.array(c["daughters"], dtype=np.float32) + 1
                lin[-1]["daughters"][np.isnan(lin[-1]["daughters"])] = 0
                if "edges" in c:
                    lin[-1]["edges"] = c["edges"]

                # Morphological features:
                morpho_features = (
                    "area", "width", "length", "perimeter", "old_pole",
                    "new_pole", "growthrate_length", "growthrate_area",
                )
                for feat in morpho_features:
                    if feat in c:
                        lin[-1][feat] = np.array(c[feat], dtype=np.float32)

                # Loop through potential fluo channels:
                fluo = 0
                while True:
                    fluo += 1
                    fluostr = f"fluo{fluo}"
                    if fluostr in c:
                        lin[-1][fluostr] = np.array(c[fluostr], dtype=np.float32)
                    else:
                        break
            # Store into res dict:
            res["lineage"] = lin

            # Store into data structure:
            data["res"] += [res]

        # Finally, save to disk:
        savemat(res_file, data)

    def results_movie(self,
                      original_video,
                      frames: List[int] = None) -> Any:
        """
        Generate movie illustrating segmentation and tracking

        Parameters
        ----------
        original_video : np.array
            3D numpy array.
        frames : list of int or None, optional
            Frames to generate the movie for. If None, all frames are run.
            The default is None.

        Returns
        -------
        movie : list of 3D numpy arrays
            List of compiled movie frames

        """

        trans_frames = np.zeros((
            len(original_video),
            original_video.shape[1],
            original_video.shape[2]
        ))

        for frame in range(len(original_video)):
            # Crop and scale:
            img = utils.rangescale(original_video[frame], rescale=(0, 1))
            trans_frames[frame] = img

        if self.drift_correction:
            trans_frames, _ = utils.driftcorr(trans_frames, drift=self.drift_values)

        movie = []

        # Run through frames, compile movie:
        for f, fnb in enumerate(frames):

            frame = trans_frames[f]

            # RGB-ify:
            frame = np.repeat(frame[:, :, np.newaxis], 3, axis=-1)

            # Add frame number text:
            frame = cv2.putText(
                frame,
                text=f"frame {fnb:06d}",
                org=(int(frame.shape[0] * 0.05), int(frame.shape[0] * 0.97)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(1, 1, 1, 1),
                thickness=2,
            )

            for r, roi in enumerate(self.rois):

                # Get chamber-specific variables:
                colors = utils.getrandomcolors(len(roi.lineage.cells), seed=r)
                fr = roi.label_stack[fnb]

                cells, contours = utils.getcellsinframe(fr, return_contours=True)

                if roi.box is None:
                    xtl, ytl = (0, 0)
                else:
                    xtl, ytl = (roi.box["xtl"], roi.box["ytl"])

                # Run through cells in labelled frame:
                for c, cell in enumerate(cells):

                    # Draw contours:
                    frame = cv2.drawContours(
                        frame,
                        contours,
                        c,
                        color=colors[cell],
                        thickness=1,
                        offset=(xtl, ytl),
                    )

                    # Draw poles:
                    oldpole = roi.lineage.getvalue(cell, fnb, "old_pole")
                    assert isinstance(oldpole, np.ndarray)  # for mypy
                    frame = cv2.drawMarker(
                        frame,
                        (oldpole[1] + xtl, oldpole[0] + ytl),
                        color=colors[cell],
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=3,
                        thickness=1,
                    )

                    daughter = roi.lineage.getvalue(cell, fnb, "daughters")
                    bornago = roi.lineage.cells[cell]["frames"].index(fnb)
                    mother = roi.lineage.cells[cell]["mother"]

                    if daughter is None and (bornago > 0 or mother is None):
                        newpole = roi.lineage.getvalue(cell, fnb, "new_pole")
                        frame = cv2.drawMarker(
                            frame,
                            (newpole[1] + xtl, newpole[0] + ytl),
                            color=[1, 1, 1],
                            markerType=cv2.MARKER_TILTED_CROSS,
                            markerSize=3,
                            thickness=1,
                        )

                    # Plot division arrow:
                    if daughter is not None:

                        newpole = roi.lineage.getvalue(cell, fnb, "new_pole")
                        daupole = roi.lineage.getvalue(daughter, fnb, "new_pole")
                        # Plot arrow:
                        frame = cv2.arrowedLine(
                            frame,
                            (newpole[1] + xtl, newpole[0] + ytl),
                            (daupole[1] + xtl, daupole[0] + ytl),
                            color=(1, 1, 1),
                            thickness=1,
                        )

            # Add to movie array:
            movie += [(frame * 255).astype(np.uint8)]

        return movie
