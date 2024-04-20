from enum import unique
import os

import numpy as np
import numpy.ma as ma
import pandas as pd

import scipy.optimize

import skimage.measure

import cellacdc.workers
import cellacdc.core

from ..CellACDC import CellACDC_tracker

class SearchRangeUnits:
    values = ['pixels', 'micrometre']

class Integer:
    not_a_param = True

class tracker:
    def __init__(
            self, 
            annotate_objects_tracked_second_step=True,
            PhysicalSizeX=1.0,
            PhysicalSizeY=1.0,
            PhysicalSizeZ=1.0,
        ):
        """Initialize Cell-ACDC two steps tracker

        Parameters
        ----------
        annotate_objects_tracked_second_step : bool, optional
            If True, Cell-ACDC will draw a line on the GUI between the objects 
            in previous frame that were lost in current frame according to the 
            first step (based on overlap) and the objects in current frame that 
            were matched according to the second step (based on search range). 
            Default is True
        PhysicalSizeX : float, optional
            Pixel size in the x-direction in 'micrometre/pixel'. This will be 
            ignored if `search_range_unit` is `pixels`. Default is 1.0
        PhysicalSizeY : float, optional
            Pixel size in the y-direction in 'micrometre/pixel'. This will be 
            ignored if `search_range_unit` is `pixels`. Default is 1.0.
        PhysicalSizeZ : float, optional
            Pixel size in the z-direction in 'micrometre/pixel'. This will be 
            ignored if `search_range_unit` is `pixels`. Default is 1.0. 
        """        
        self._annot_obj_2nd_step = annotate_objects_tracked_second_step
        self._pixel_yx_size = (PhysicalSizeY, PhysicalSizeX)
        self._voxel_zyx_size = (PhysicalSizeZ, PhysicalSizeY, PhysicalSizeX)
        
    def track(
            self, segm_video,
            overlap_threshold=0.4,
            search_range_unit: SearchRangeUnits='pixels',
            lost_IDs_search_range=10,
            signals: cellacdc.workers.signals=None,
            export_to_extension='.csv', 
            export_to: os.PathLike=None,
        ):
        """Track the objects in `segm_video`.

        Parameters
        ----------
        segm_video : (T, Y, X) or (T, Z, Y, X) array of ints
            Input segmentation masks to track.
        overlap_threshold : float, optional
            Minimum overlap between objects of two consecutive frames to 
            consider the object as not new. The overlap is calculated as the 
            ratio between the intersection between current object and objects 
            in previous frame and are of the objects in previous frame. 
            All new objects will undergo a second step of matching based on 
            the `lost_IDs_search_range`. Default is 0.4
        search_range_unit : {'pixels', 'micrometre'}, optional
            Physical unit of the parameter `lost_IDs_search_range`. If 
            `micrometre`, distances will be converted using the pixel sizes. 
            See the parameters `PixelSizeX`, `PixelSizeY`, and `PixelSizeZ`. 
            Default is 'pixels'
        lost_IDs_search_range : int, optional
            Maximum distance that a new object (according to `overlap_threshold`) 
            can travel between two consecutive frames to be considered as 
            potential candidate to match to a lost object. The unit is 
            either `pixels` or `micrometre` (see `search_range_unit` parameter).
            Default is 10
        signals : cellacdc.workers.signals, optional
            Class with `qtpy.Signal` attributes used to display progress on the 
            GUI (text and progressbars). Default is None
        export_to_extension : str, optional
            Extension of the optional table that will be saved in the tracking 
            process. Default is '.csv'
        export_to : os.PathLike, optional
            Path of the table to export. Default is None
        """        
        tracked_video = np.copy(segm_video)
        for frame_i, lab in enumerate(segm_video):
            if frame_i == 0:
                continue
            prev_frame_lab = tracked_video[frame_i-1]
            tracked_lab, _ = self.track_frame(
                prev_frame_lab, lab, 
                search_range_unit=search_range_unit,
                overlap_threshold=overlap_threshold, 
                lost_IDs_search_range=lost_IDs_search_range
            )
            tracked_video[frame_i] = tracked_lab
            self.updateGuiProgressBar(signals)
        return tracked_video
        
    def track_frame(
            self, prev_frame_lab, current_frame_lab,
            overlap_threshold=0.4,
            search_range_unit: SearchRangeUnits='pixels',
            lost_IDs_search_range=10,
            unique_ID: Integer=None
        ):
        """Track two consecutive frames in two steps. First step based on 
        `overlap_threshold` and second step tracks only lost objects to new 
        objects detemined at first step.

        Parameters
        ----------
        prev_frame_lab : (Y, X) or (Z, Y, X) array of ints
            Segmentation masks of the previous frame.
        current_frame_lab : (Y, X) or (Z, Y, X) array of ints
            Segmentation masks of the current frame.
        overlap_threshold : float, optional
            Minimum overlap between objects of two consecutive frames to 
            consider the object as not new. The overlap is calculated as the 
            ratio between the intersection between current object and objects 
            in previous frame and are of the objects in previous frame. 
            All new objects will undergo a second step of matching based on 
            the `lost_IDs_search_range`. Default is 0.4
        search_range_unit : {'pixels', 'micrometre'}, optional
            Physical unit of the parameter `lost_IDs_search_range`. If 
            `micrometre`, distances will be converted using the pixel sizes. 
            See the parameters `PixelSizeX`, `PixelSizeY`, and `PixelSizeZ`. 
            Default is 'pixels'
        lost_IDs_search_range : int, optional
            Maximum distance that a new object (according to `overlap_threshold`) 
            can travel between two consecutive frames to be considered as 
            potential candidate to match to a lost object. The unit is 
            either `pixels` or `micrometre`and it is set in the 
            `search_range_unit` parameter. Default is 10
        unique_ID : int, optional
            If not None, uses this as starting ID for all the untracked objects.
            If None, this will be calculated based on the two input frames.
        """        
        to_track_tracked_objs_2nd_step = None
        
        prev_rp = skimage.measure.regionprops(prev_frame_lab)
        curr_rp = skimage.measure.regionprops(current_frame_lab)
        
        tracked_lab_1st_step = CellACDC_tracker.track_frame(
            prev_frame_lab, 
            prev_rp, 
            current_frame_lab, 
            curr_rp, 
            IoA_thresh=overlap_threshold, 
            return_prev_IDs=False, 
            uniqueID=unique_ID
        )
        
        prev_rp_mapper = {obj.label: obj for obj in prev_rp}
        
        tracked_rp_1st_step = skimage.measure.regionprops(tracked_lab_1st_step)
        tracked_rp_1st_step_mapper = {
            obj.label: obj for obj in tracked_rp_1st_step    
        }
        
        lost_rp_mapper = {
            obj.label: obj for obj in prev_rp 
            if tracked_rp_1st_step_mapper.get(obj.label) is None
        }
        
        if not lost_rp_mapper:
            return tracked_lab_1st_step, to_track_tracked_objs_2nd_step
        
        new_rp_mapper = {
            obj.label: obj for obj in tracked_rp_1st_step 
            if prev_rp_mapper.get(obj.label) is None
        }
        
        if not new_rp_mapper:
            return tracked_lab_1st_step, to_track_tracked_objs_2nd_step
        
        ndim = current_frame_lab.ndim
        lost_IDs_coords = np.zeros((len(lost_rp_mapper), ndim))
        lost_IDs_idx_to_obj_mapper = {}
        for lost_idx, lost_obj in enumerate(lost_rp_mapper.values()):
            lost_IDs_coords[lost_idx] = lost_obj.centroid
            lost_IDs_idx_to_obj_mapper[lost_idx] = lost_obj
        
        new_IDs_coords = np.zeros((len(new_rp_mapper), ndim))
        new_IDs_idx_to_obj_mapper = {}
        for new_idx, new_obj in enumerate(new_rp_mapper.values()):
            new_IDs_coords[new_idx] = new_obj.centroid
            new_IDs_idx_to_obj_mapper[new_idx] = new_obj
        
        if search_range_unit == 'micrometre':
            if ndim == 3:
                scaling = self._voxel_zyx_size
            else:
                scaling = self._pixel_yx_size
            lost_IDs_coords /= scaling
            new_IDs_coords /= scaling
        
        diff = lost_IDs_coords[:, np.newaxis] - new_IDs_coords
        # dist_matrix[i, j] = euclidean_dist(lost_IDs_coords[i], new_IDs_coords[j])
        dist_matrix = np.linalg.norm(diff, axis=2)
        
        assignments = scipy.optimize.linear_sum_assignment(dist_matrix)
        IDs_to_track = []
        tracked_IDs_2nd_step = []
        if self._annot_obj_2nd_step:
            objs_to_track = []
            tracked_objs_2nd_step = []
        for i, j in zip(*assignments):
            dist = dist_matrix[i, j]
            if dist > lost_IDs_search_range:
                continue
            
            IDs_to_track.append(new_IDs_idx_to_obj_mapper[j].label)
            tracked_IDs_2nd_step.append(lost_IDs_idx_to_obj_mapper[i].label)
            if self._annot_obj_2nd_step:
                objs_to_track.append(new_IDs_idx_to_obj_mapper[j])
                tracked_objs_2nd_step.append(lost_IDs_idx_to_obj_mapper[i])
        
        if not IDs_to_track:
            return tracked_lab_1st_step, to_track_tracked_objs_2nd_step
        
        tracked_lab_2nd_step = cellacdc.core.lab_replace_values(
            tracked_lab_1st_step, 
            tracked_rp_1st_step,
            IDs_to_track, 
            tracked_IDs_2nd_step
        )
        
        if self._annot_obj_2nd_step:
            to_track_tracked_objs_2nd_step = (
                objs_to_track, tracked_objs_2nd_step
            )
        
        return tracked_lab_2nd_step, to_track_tracked_objs_2nd_step
    
    def updateGuiProgressBar(self, signals):
        if signals is None:
            return
        
        if hasattr(signals, 'innerPbar_available'):
            if signals.innerPbar_available:
                # Use inner pbar of the GUI widget (top pbar is for positions)
                signals.innerProgressBar.emit(1)
                return

        signals.progressBar.emit(1)
        
        
        
        