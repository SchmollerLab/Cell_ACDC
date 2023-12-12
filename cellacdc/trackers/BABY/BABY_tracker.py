import numpy as np

import scipy.ndimage
import skimage.measure

from baby import modelsets
from baby import BabyCrawler

from cellacdc import myutils
from cellacdc.trackers import BABY

from ..CellACDC import CellACDC_tracker

class AvailableModels:
    values = BABY.BABY_MODELS

class tracker:
    def __init__(
            self, 
            model_type: AvailableModels='yeast-alcatras-brightfield-sCMOS-60x-5z',
        ):
        brain = modelsets.get(model_type)
        self.crawler = BabyCrawler(brain)
    
    def _preprocess(self, image, swap_YX_axes_to_XY):
        if image.ndim == 2:
            image = image[np.newaxis]
        
        image = myutils.to_uint16(image)
        
        # BABY requires z-slices as last dimension while Cell-ACDC takes 
        # Z, Y, X input
        if swap_YX_axes_to_XY:
            source_axes = (0, 1, 2)
            dst_axes = (2, 1, 0)
        else:
            source_axes = (0, 1, 2)
            dst_axes = (1, 2, 0)        
        
        image = np.moveaxis(image, source_axes, dst_axes)
        
        return image
    
    def iterate_result_series(self, result_series, swap_YX_axes_to_XY):
        for frame_i, result in enumerate(result_series):
            contour_masks = result[0]['edgemasks']
            IDs = result[0]['cell_label']
            for ID, contour_mask in zip(IDs, contour_masks):
                mask = scipy.ndimage.binary_fill_holes(contour_mask)  
                if swap_YX_axes_to_XY:
                    mask = np.swapaxes(mask, 0, 1)              
                yield frame_i, mask, ID
    
    def track_baby_segm_data(
            self, segm_data, result_series, swap_YX_axes_to_XY
        ):
        tracked_data = np.zeros_like(segm_data)
        result_generator = self.iterate_result_series(
            result_series, swap_YX_axes_to_XY
        )
        for frame_i, mask, ID in result_generator:              
            tracked_data[frame_i][mask] = ID
        return tracked_data
    
    def track_external_segm_data(
            self, segm_data, result_series, swap_YX_axes_to_XY
        ):
        result_generator = self.iterate_result_series(
            result_series, swap_YX_axes_to_XY
        )
        old_IDs_tracks = {}
        tracked_IDs_tracks = {}
        for frame_i, mask, track_ID in result_generator:   
            oldID = segm_data[frame_i][mask][0]
            if oldID == 0:
                continue
            
            if frame_i not in old_IDs_tracks:
                old_IDs_tracks[frame_i] = [oldID]
                tracked_IDs_tracks[frame_i] = [track_ID]
            else:
                old_IDs_tracks[frame_i].append(oldID)
                tracked_IDs_tracks[frame_i].append(track_ID)
        
        tracked_data = segm_data.copy()
        for frame_i in old_IDs_tracks.keys():
            tracked_IDs = tracked_IDs_tracks[frame_i]
            old_IDs = old_IDs_tracks[frame_i]
            
            lab = self.segm_video[frame_i]
            rp = skimage.measure.regionprops(lab)
            IDs_curr_untracked = [obj.label for obj in rp]
            
            uniqueID = max((max(tracked_IDs), max(IDs_curr_untracked)))+1
            tracked_lab = CellACDC_tracker.indexAssignment(
                old_IDs, tracked_IDs, IDs_curr_untracked,
                lab.copy(), rp, uniqueID
            )
            tracked_data[frame_i] = tracked_lab
        
        return tracked_data
    
    def track(
            self, segm_data, intensity_data, 
            resegment_data=True,
            swap_YX_axes_to_XY=True,
            refine_outlines=True,
            assign_mothers=True,
            with_edgemasks=True,
            with_volumes=True,
            parallel=False,
            signals=None
        ):
        """_summary_

        Parameters
        ----------
        segm_data : (T, Y, X) numpy.ndarray of ints
            Input segmentation data
        intensity_data : (T, Y, X) or (T, Z, Y, X)
            Input intensity data
        resegment_data : bool, optional
            If True, BABY will ignore the input `segm_data` and perform 
            segmentation de novo. 
            If False, BABY will only track the input `segm_data`. 
            Default is True

        Returns
        -------
        np.ndarray with the same shape as `segm_data`
            Tracked data
        """        
        image_series = [
            self._preprocess(image, swap_YX_axes_to_XY) for image in intensity_data
        ]
        
        result_series = []
        for image in image_series:
            result = self.crawler.step(
                image[None, ...], 
                refine_outlines=refine_outlines,
                assign_mothers=assign_mothers,
                with_edgemasks=with_edgemasks,
                with_volumes=with_volumes,
                parallel=parallel
            )
            result_series.append(result)
            self.updateGuiProgressBar(signals)
        
        if resegment_data:
            tracked_data = self.track_baby_segm_data(
                segm_data, result_series, swap_YX_axes_to_XY
            )
        else:
            tracked_data = self.track_external_segm_data(
                segm_data, result_series, swap_YX_axes_to_XY
            )
        
        return tracked_data

    def updateGuiProgressBar(self, signals):
        if signals is None:
            return
        
        if hasattr(signals, 'innerPbar_available'):
            if signals.innerPbar_available:
                # Use inner pbar of the GUI widget (top pbar is for positions)
                signals.innerProgressBar.emit(1)
                return

        if hasattr(signals, 'progressBar'):
            signals.progressBar.emit(1)
                
        