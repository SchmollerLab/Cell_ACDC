import cv2
import random

import numpy as np

import torch

from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from skimage.transform import resize

from deepsea.model import DeepSeaTracker
from deepsea.utils import track_cells

from cellacdc import myutils, printl
from cellacdc.models.DeepSea import _init_model, _resize_img
from cellacdc.models.DeepSea import image_size as segm_image_size
from cellacdc.models.DeepSea import _get_segm_transforms

from . import _get_tracker_transforms

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

class tracker:
    def __init__(self, gpu=False):
        torch_device, checkpoint, model = _init_model(
            'tracker.pth', DeepSeaTracker, gpu=gpu
        )
        self.torch_device = torch_device
        self._transforms = _get_tracker_transforms()
        self._segm_transforms = _get_segm_transforms()
        self._checkpoint = checkpoint
        self.model = model
    
    def _resize_lab(self, lab, output_shape, rp):
        _lab_obj_to_resize = np.zeros(lab.shape, dtype=np.float16)
        lab_resized = np.zeros(output_shape, dtype=np.uint32)
        for obj in rp:
            _lab_obj_to_resize[obj.slice][obj.image] = 1.0
            _lab_obj_resized = resize(
                _lab_obj_to_resize, output_shape, anti_aliasing=True,
                preserve_range=True
            ).round()
            lab_resized[_lab_obj_resized == 1.0] = obj.label
            _lab_obj_to_resize[:] = 0.0
        return lab_resized

    def _relabel_sequential(self, segm_video):
        relabelled_video = np.zeros_like(segm_video)
        for frame_i, lab in enumerate(segm_video):
            relabelled_lab, _, _ = relabel_sequential(lab)
            relabelled_video[frame_i] = relabelled_lab
        return relabelled_video

    def track(
            self, segm_video, image, min_size=10, annotate_lineage_tree=True, 
            signals=None
        ):
        self.signals = signals
        segm_video = self._relabel_sequential(segm_video)
        labels_list = []
        resize_img_list = []
        for img, lab in zip(image, segm_video):
            img = (255 * ((img - img.min()) / img.ptp())).astype(np.uint8)
            rp = regionprops(lab)
            resized_img = _resize_img(
                img, self.torch_device, self._segm_transforms
            )
            resized_lab = self._resize_lab(
                lab, output_shape=tuple(segm_image_size), rp=rp
            )
            resize_img_list.append(resized_img)
            labels_list.append(resized_lab)
        
        result = track_cells(
            labels_list, resize_img_list, self.model, self.torch_device, 
            transforms=self._transforms, min_size=min_size
        )
        tracked_labels, tracked_centroids, tracked_imgs = result
        
        labels_to_IDs_mapper = self._from_labels_to_IDs(tracked_labels)
        if annotate_lineage_tree:
            self.cca_dfs = self._annotate_lineage_tree(
                tracked_labels, labels_to_IDs_mapper
            )
        tracked_video = self._replace_tracked_IDs(
            labels_list, tracked_labels, tracked_centroids, 
            labels_to_IDs_mapper, segm_video
        )

        return tracked_video

    def _annotate_lineage_tree(self, tracked_labels, labels_to_IDs_mapper):
        if self.signals is not None:
            self.signals.progress.emit('Annotating lineage trees...')
        from cellacdc.core import getBaseCca_df
        import pandas as pd
        IDs_to_labels_mapper = {
            ID:label for label, ID in labels_to_IDs_mapper.items()
        }
        cca_dfs = []
        keys = []
        for frame_i, tracked_frame_labels in enumerate(tracked_labels):
            keys.append(frame_i)
            IDs = [
                labels_to_IDs_mapper[label] for label in tracked_frame_labels
            ]
            if frame_i == 0:
                cca_df = getBaseCca_df(IDs)
                cca_dfs.append(cca_df)
                continue

            # Get cca_df from previous frame for existing cells
            cca_df = cca_dfs[frame_i-1]
            is_in_index = cca_df.index.isin(IDs)
            cca_df = cca_df[is_in_index]
            new_cells_cca_dfs = []

            for ID in IDs:
                if ID in cca_df.index:
                    continue
                
                newID = ID
                # New cell --> store cca info
                label = IDs_to_labels_mapper[newID]
                parent_label, _, sister_label = label.rpartition('_')
                if not parent_label:
                    # New single-cell --> check if it existed in past frames
                    for i in range(frame_i-2, -1, -1):
                        past_cca_df = cca_dfs[frame_i-1]
                        if newID in past_cca_df.index:
                            cca_df_single_ID = past_cca_df.loc[[newID]]
                            break
                    else:
                        cca_df_single_ID = getBaseCca_df([newID])
                        cca_df_single_ID.loc[newID, 'emerg_frame_i'] = frame_i
                else:
                    # New cell resulting from division --> store division
                    mothID = labels_to_IDs_mapper[parent_label]
                    cca_df_single_ID = getBaseCca_df([newID])
                    cca_df.at[mothID, 'generation_num'] += 1
                    cca_df.at[mothID, 'division_frame_i'] = frame_i
                    cca_df.at[mothID, 'relative_ID'] = newID
                    cca_df_single_ID.at[newID, 'emerg_frame_i'] = frame_i   
                    cca_df_single_ID.at[newID, 'division_frame_i'] = frame_i
                    cca_df_single_ID.at[newID, 'generation_num'] = 1  
                    cca_df_single_ID.at[newID, 'relative_ID'] = mothID

                new_cells_cca_dfs.append(cca_df_single_ID)
            
            cca_df = pd.concat([cca_df, *new_cells_cca_dfs]).sort_index()
            cca_dfs.append(cca_df)
        
        return cca_dfs

    def _from_labels_to_IDs(self, tracked_labels):
        labels_to_IDs_mapper = {}
        uniqueID = 1
        for tracked_frame_labels in tracked_labels:
            for tracked_label in tracked_frame_labels:
                if tracked_label in labels_to_IDs_mapper:
                    # Cell existed in the past, ID already stored
                    continue
                
                parent_label, _, sister_label = tracked_label.rpartition('_')
                if not parent_label:
                    # Single-cell that was not mapped yet
                    labels_to_IDs_mapper[tracked_label] = uniqueID
                    uniqueID += 1
                    continue

                if sister_label == '0':
                    # Sister label == 0 --> keep mother ID
                    ID = labels_to_IDs_mapper[parent_label]
                else:
                    # Sister label == 1 --> assign new ID
                    ID = uniqueID
                    uniqueID += 1
                labels_to_IDs_mapper[tracked_label] = ID

        return labels_to_IDs_mapper

    def _replace_tracked_IDs(
            self, resized_labels_list, tracked_labels, tracked_centroids,
            labels_to_IDs_mapper, segm_video
        ):
        if self.signals is not None:
            self.signals.progress.emit('Applying tracking information...')
        
        _zip = zip(tracked_labels, tracked_centroids)
        IDs_prev = []
        tracked_video = np.zeros_like(segm_video)
        for frame_i, track_info_frame in enumerate(_zip):
            tracked_frame_labels, tracked_frame_centroids = track_info_frame
            tracked_frame_IDs = [
                labels_to_IDs_mapper[label] for label in tracked_frame_labels
            ]
            lab = resized_labels_list[frame_i]
            tracked_lab = tracked_video[frame_i]
            untracked_lab = segm_video[frame_i]
            rp = regionprops(lab)
            IDs_curr_untracked = [obj.label for obj in rp]
            uniqueID = max(
                max(IDs_prev, default=0), 
                max(IDs_curr_untracked, default=0),
                max(tracked_frame_IDs, default=0)
            ) + 1
            IDs_to_replace = {
                lab[tuple(centr)]:idx
                for idx, centr in enumerate(tracked_frame_centroids)
            }
            IDs_prev = []            
            for obj in rp:
                idx_ID_to_replace = IDs_to_replace.get(obj.label)
                if idx_ID_to_replace is None:
                    newID = uniqueID
                    uniqueID += 1
                else:
                    newID = tracked_frame_IDs[idx_ID_to_replace]
                tracked_lab[untracked_lab == obj.label] = newID
                IDs_prev.append(newID)
            
            tracked_video[frame_i] = tracked_lab
            self.updateGuiProgressBar(self.signals)

        return tracked_video
    
    def updateGuiProgressBar(self, signals):
        if signals is None:
            return
        
        if hasattr(signals, 'innerPbar_available'):
            if signals.innerPbar_available:
                # Use inner pbar of the GUI widget (top pbar is for positions)
                signals.innerProgressBar.emit(1)
                return

        signals.progressBar.emit(1)
            
            