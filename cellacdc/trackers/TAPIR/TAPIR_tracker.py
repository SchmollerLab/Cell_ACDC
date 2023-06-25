import os
from tqdm import tqdm

import haiku as hk
import jax

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

import skimage.transform
import skimage.measure
import skimage.color

from cellacdc import printl
from cellacdc.transformation import resize_lab
from cellacdc.core import nearest_nonzero_2D

from ..CellACDC import CellACDC_tracker

from . import TAPIR_CHECKPOINT_PATH
from .tracking import build_model, inference

class SizesToResize:
    values = np.arange(256, 1025, 128)

class tracker:
    def __init__(
            self, model_checkpoint_path: os.PathLike=TAPIR_CHECKPOINT_PATH
        ):
        ckpt_state = np.load(model_checkpoint_path, allow_pickle=True).item()
        params, state = ckpt_state['params'], ckpt_state['state']
        model = hk.transform_with_state(build_model)
        model_apply = jax.jit(model.apply)
        self.params = params
        self.state = state
        self.model_apply = model_apply
    
    def track(
            self, segm_video, video_grayscale, 
            resize_to_square_with_size: SizesToResize=256,
            max_distance=5, save_napari_tracks=False, 
            use_visibile_information=True, export_to=None,
            signals=None, export_to_extension='.csv', 
            track_segmented_objects=False
        ):
        
        if video_grayscale.ndim == 4:
            ndim = video_grayscale.ndim
            msg = f'TAPIR can only track 2D frames over time. Input image is {ndim}D'
            raise TypeError(msg)
        
        self._use_visibile_information = use_visibile_information
        self.segm_video = segm_video
        self.max_dist = max_distance
        num_frames = len(video_grayscale)
        height, width = segm_video.shape[-2:]
        new_size = resize_to_square_with_size
        frames = skimage.transform.resize(
            video_grayscale, (num_frames, new_size, new_size)
        )
        frames = frames/frames.max()
        self.resize_ratio_height = height/new_size
        self.resize_ratio_width = width/new_size
        
        resized_segm_video = np.array(
            [resize_lab(lab, (new_size, new_size)) for lab in segm_video]
        )
        
        # We track from last frame backwards
        reversed_resized_frames = frames[::-1]
        reversed_resized_segm = resized_segm_video[::-1]
        
        self.reversed_resized_segm = reversed_resized_segm
        
        frames_rgb = self._get_frames_to_track(
            reversed_resized_frames, reversed_resized_segm, 
            track_segmented_objects
        )
        query_points = self._initialize_query_points(
            reversed_resized_segm, track_segmented_objects
        )
        
        import matplotlib.pyplot as plt
        plt.imshow(frames_rgb[0])
        plt.plot(query_points[:,2], query_points[:,1], 'r.')
        plt.show()
        
        self.reversed_tracks, self.reversed_visibles = inference(
            frames_rgb, query_points, self.model_apply, self.params, 
            self.state
        )
        
        tracked_video = self._apply_tracks(
            self.reversed_tracks, self.reversed_visibles
        )
        
        if save_napari_tracks:
            self._save_napari_tracks(export_to)
        
        if export_to is not None:
            self._save_tracks(export_to)
        return tracked_video

    def _get_frames_to_track(
            self, reversed_resized_frames, reversed_resized_segm, 
            track_segmented_objects
        ):
        if track_segmented_objects:
            frames = np.zeros(reversed_resized_segm.shape, dtype=np.float32)
            for frame_i, lab in enumerate(reversed_resized_segm):
                rp = skimage.measure.regionprops(lab)
                for obj in rp:
                    obj_edt = distance_transform_edt(obj.image).astype(np.float32)
                    obj_edt /= obj_edt.max()
                    frames[frame_i][obj.slice][obj.image] = obj_edt[obj.image]
        else:
            frames = reversed_resized_frames
        frames_rgb = (skimage.color.gray2rgb(frames)*255).astype(np.uint8)
        return frames_rgb
    
    def _save_napari_tracks(self, export_to):
        print('Saving napari tracks...')
        napari_tracks = self.to_napari_tracks()
        if export_to is None:
            napari_tracks_path = 'tapir_napari_tracks.csv'
        else:
            napari_tracks_path = export_to.replace('.csv', '_napari.csv')
        df = pd.DataFrame(data=napari_tracks, columns=['ID', 'T', 'Y', 'X'])
        df.to_csv(napari_tracks_path, index=False)
    
    def _save_tracks(self, export_to):
        print('Saving tracks...')
        tracks = self.reversed_tracks[:, ::-1]
        visibles = self.reversed_visibles[:, ::-1]
        resized_segm = self.reversed_resized_segm[::-1]
        track_IDs = []
        frames = []
        xx = []
        yy = []
        visibles_li = []
        Y, X = resized_segm.shape[-2:]
        for tr, track in enumerate(tqdm(tracks, ncols=100)):
            x, y = track[-1]
            y_int, x_int = round(y), round(x)
            y_int = max(0, min(y_int, Y-1))
            x_int = max(0, min(x_int, X-1))
            track_ID = resized_segm[-1, y_int, x_int]
            for frame_i, (x, y) in enumerate(track):                  
                yc = y*self.resize_ratio_height
                xc = x*self.resize_ratio_width
                visible = visibles[tr, frame_i]
                track_IDs.append(track_ID)
                frames.append(frame_i)
                xx.append(xc)
                yy.append(yc)
                visibles_li.append(visible)
        df = pd.DataFrame({
            'frame_i': frames, 
            'track_ID': track_ID,
            'y_point': yy, 
            'x_point': xx,
            'visible': visibles_li
        }).set_index(['frame_i', 'track_ID']).sort_index()
        df.to_csv(export_to)

    def to_napari_tracks(self, use_centroids=False):
        print('Building napari tracks data...')
        napari_tracks = []
        num_frames = len(self.reversed_resized_segm)
        Y, X = self.reversed_resized_segm.shape[-2:]
        for tr, track in enumerate(tqdm(self.reversed_tracks, ncols=100)):
            x, y = track[0]
            y_int, x_int = round(y), round(x)
            y_int = max(0, min(y_int, Y-1))
            x_int = max(0, min(x_int, X-1))
            track_ID = self.reversed_resized_segm[0, y_int, x_int]
            for reversed_frame_i, (x, y) in enumerate(track):
                visible = self.reversed_visibles[tr, reversed_frame_i]
                if not visible and self._use_visibile_information:
                    continue
                y_int, x_int = round(y), round(x)
                y_int = max(0, min(y_int, Y-1))
                x_int = max(0, min(x_int, X-1))
                self._append_napari_point(
                    napari_tracks, y, x, num_frames, reversed_frame_i, 
                    track_ID, use_centroids=use_centroids
                )
        napari_tracks = np.array(napari_tracks)
        return napari_tracks

    def _append_napari_point(
            self, napari_tracks, y, x, num_frames, 
            reversed_frame_i, track_ID, use_centroids=False
        ):
        frame_i = num_frames - reversed_frame_i - 1
        if use_centroids:
            lab = self.segm_video[frame_i]
            rp = skimage.measure.regionprops(lab)
            for obj in rp:
                if obj.label == track_ID:
                    yc, xc = obj.centroid
                    napari_tracks.append((track_ID, frame_i, yc, xc))
                    break
        else:
            yc = y*self.resize_ratio_height
            xc = x*self.resize_ratio_width
            napari_tracks.append((track_ID, frame_i, yc, xc))
    
    def _apply_tracks(self, reversed_tracks, reversed_visibles):
        print('Applying tracks data...')
        
        # Restore correct order (we tracked backwards)
        tracks = reversed_tracks[:, ::-1]
        resized_segm = self.reversed_resized_segm[::-1]
        visibles = reversed_visibles[:, ::-1]
        
        # Iterate tracks and determine tracked IDs
        old_IDs_tracks = {}
        tracked_IDs_tracks = {}
        Y, X = resized_segm.shape[-2:]
        for tr, track in enumerate(tqdm(tracks, ncols=100)):
            # Get the track ID from last frame (we track in reverse)
            x, y = track[-1]
            y0, x0 = round(y), round(x)
            y0 = max(0, min(y0, Y-1))
            x0 = max(0, min(x0, X-1))
            tracked_ID = resized_segm[-1, y0, x0]
            for frame_i, (x, y) in enumerate(track):
                if frame_i == 0:
                    continue
                
                visible = visibles[tr, frame_i]
                if not visible  and self._use_visibile_information:
                    continue
                y_int, x_int = round(y), round(x)
                y_int = max(0, min(y_int, Y-1))
                x_int = max(0, min(x_int, X-1))
                idxs = (frame_i, y_int, x_int)
                oldID = resized_segm[idxs]
                if oldID == 0:
                    oldID = nearest_nonzero_2D(
                        resized_segm[frame_i], y, x, max_dist=self.max_dist
                    )
                
                if oldID == 0:
                    continue
                
                if frame_i not in old_IDs_tracks:
                    old_IDs_tracks[frame_i] = [oldID]
                    tracked_IDs_tracks[frame_i] = [tracked_ID]
                else:
                    old_IDs_tracks[frame_i].append(oldID)
                    tracked_IDs_tracks[frame_i].append(tracked_ID)

        tracked_video = self.segm_video.copy()
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
            tracked_video[frame_i] = tracked_lab
        return tracked_video
    
    def _initialize_query_points(
            self, reversed_resized_segm, track_segmented_objects
        ):
        first_lab = reversed_resized_segm[0]
        first_lab_rp = skimage.measure.regionprops(first_lab)
        num_objs = len(first_lab_rp)
        query_points = np.zeros((num_objs, 3), dtype=int)        
        for o, obj in enumerate(first_lab_rp):
            if track_segmented_objects:
                obj_edt = distance_transform_edt(obj.image)
                argmax = np.argmax(obj_edt)
                yc_loc, xc_loc = np.unravel_index(argmax, obj_edt.shape)
                ymin, xmin, _, _ = obj.bbox
                yc, xc = yc_loc+ymin, xc_loc+xmin
            else:
                yc, xc = obj.centroid
            query_points[o, 1:] = int(yc), int(xc)
        return query_points

def url_help():
    return 'https://deepmind-tapir.github.io/'