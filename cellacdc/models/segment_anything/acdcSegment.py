import os

from collections import defaultdict

import numpy as np
import pandas as pd
import cv2

import skimage.measure

from . import model_types, sam_models_path

from segment_anything import (
    sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
)
from cellacdc import myutils, widgets, printl

class AvailableModels:
    values = list(model_types.keys())

class DataFrame:
    not_a_param = True

class NotParam:
    not_a_param = True

class Boolean:
    not_a_param = True

class Integer:
    not_a_param = True

class Model:
    def __init__(
            self, 
            model_type: AvailableModels='Large', 
            input_points_path: widgets.SamInputPointsWidget='',
            input_points_df: DataFrame='None',
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1,
            gpu=False
        ):
        """Initialization of Segment Anything Model within Cell-ACDC

        Parameters
        ----------
        points_per_side : int or None, optional
            The number of points to be sampled along one side of the image. 
            The total number of points is points_per_side**2. 
            If None, 'point_grids' must provide explicit point sampling. 
            Ignored if `input_points_path` is not empty or `input_points_df` is not 
            'None'. Default is 32
        pred_iou_thresh : float, optional 
            A filtering threshold in [0,1], using the model's predicted mask 
            quality. 
            Ignored if `input_points_path` is not empty or `input_points_df` is not 
            'None'. Default is pred_iou_thresh  
        stability_score_thresh : float, optional
            A filtering threshold in [0,1], using the stability of the mask 
            under changes to the cutoff used to binarize the model's mask 
            predictions. 
            Ignored if `input_points_path` is not empty or `input_points_df` is not 
            'None'. Default is 0.95
        crop_n_layers : int 
            If >0, mask prediction will be run again on crops of the image. 
            Sets the number of layers to run, where each layer has 2**i_layer 
            number of image crops.
            Ignored if `input_points_path` is not empty or `input_points_df` is not 
            'None'. Default is 0
        crop_n_points_downscale_factor : int, optional
            The number of points-per-side sampled in layer n is scaled down by 
            crop_n_points_downscale_factor**n.
            Ignored if `input_points_path` is not empty or `input_points_df` is not 
            'None'. Default is 2
        min_mask_region_area: int, optional
            If >0, postprocessing will be applied mto remove disconnected 
            regions and holes in masks with area smaller than 
            min_mask_region_area. 
            Ignored if `input_points_path` is not empty or `input_points_df` is not 
            'None'. Default is 1
        input_points_path : str, optional
            If not empty, this is the path to the CSV file with the coordinates 
            of the input points for SAM. It must contain the columns 
            ('x', 'y', 'id') with an optional 'z' column for segmentation of 3D 
            z-stack data (slice-by-slice) and a 'frame_i' columns for 
            time-lapse data. 
            
            Note that `id = 0` will be used for the negative points, i.e. those 
            objects (like the background) that should not be segmented.
            
            In the Cell-ACDC GUI (module 3) you can click to add points and 
            save them to a file whose path or endname can be provided for the 
            `input_points_path`. To do so, click on the "Add points layer" 
            button on the top toolbar and choose "Add points with mouse clicks". 
            
            To add a new point for a new object click with the mouse left 
            button. To add points to the same object click with the right 
            button. The 'id' of the point will be visible next to the point 
            symbol. To delete a point click on the point. 
            
            To add negative points click with the middle button (Cmd+click on 
            macOS) or enter 0 in the "Point id" numeric control (top toolbar) 
            and then right-click to add points with the current id.   
            
            To load the coordinates from a CSV file click on the browse button.
            
            If empty string and `inputs_points_df` is 'None', SAM will run 
            in automatic mode on the entire image. Default is None
        
        input_points_df : pd.DataFrame or 'None', optional
            If not 'None', this is a pandas DataFrame (a table) with the 
            coordinates of the input points for SAM. 
            
            It must contain the columns ('x', 'y', 'id') with an optional 
            'z' column for segmentation of 3D z-stack data (slice-by-slice) and 
            a 'frame_i' columns for time-lapse data. Note that `id = 0` will 
            be used for the negative points, i.e. those objects (like the 
            background) that should not be segmented.
            
            If not 'None', `input_points_path` will be ignored and this will be used 
            instead. 
            
            If 'None' and `input_points_path` is empty, SAM will run 
            in automatic mode on the entire image. Default is 'None' 
        """        
        if gpu:
            from cellacdc import is_mac_arm64
            if is_mac_arm64:
                device = 'cpu'
            else:
                device = 'cuda'
        else:
            device = 'cpu'
        
        if isinstance(input_points_df, str) and input_points_df=='None':
            input_points_df = None
        
        load_points_df = (
            input_points_path
            and input_points_df is None
        )
        if load_points_df:
            input_points_df = pd.read_csv(input_points_path)
        
        if input_points_df is not None:
            if 'z' in input_points_df.columns:
                input_points_df = input_points_df.sort_values(['z', 'id'])
            else:
                input_points_df = input_points_df.sort_values('id')
        
        self._input_points_df = input_points_df
        
        model_type, sam_checkpoint = model_types[model_type]
        sam_checkpoint = os.path.join(sam_models_path, sam_checkpoint)
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint) 
        sam.to(device=device)
        if input_points_df is None:
            self.model = SamAutomaticMaskGenerator(
                sam, 
                points_per_side=points_per_side,
                pred_iou_thresh=pred_iou_thresh,
                stability_score_thresh=stability_score_thresh,
                crop_n_layers=crop_n_layers,
                crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                min_mask_region_area=min_mask_region_area,
            )
        else:
            self.model = SamPredictor(sam)
        
        self._embedded_img = None
    
    def segment(
            self, 
            image: np.ndarray, 
            frame_i: int,
            automatic_removal_of_background: bool=True,
            input_points_df: DataFrame='None',
            posData: NotParam=None, 
            save_embeddings: Boolean=False, 
            only_embeddings: Boolean=False,
            use_loaded_embeddings: Boolean=False, 
            start_z_slice: Integer=0
        ) -> np.ndarray:
        
        """_summary_

        image : ([Z], Y, X, [C]) numpy.ndarray
            Input image. It can be grayscale 2D (Y, X), or 3D (Z, Y, X) for 
            z-stack data, or it can have additional dimension C for the RGB 
            channels.
        
        frame_i : int
            Frame index (starting from 0). Used to get the input points from 
            `input_points_df` with timelapse data. Ignored if the 
            `input_points_df` does not have the 'frame_i' column.
        
        automatic_removal_of_background : bool, optional
            If True, the background object will be removed. The background 
            object is defined as the largest object touching the borders of the 
            image. Used only with automatic generator without input prompts, 
            i.e., `input_points_path` is empty and `input_points_df` is equal 
            to 'None'.
        
        input_points_df : pd.DataFrame or 'None', optional
            If not 'None', this is a pandas DataFrame (a table) with the 
            coordinates of the input points for SAM. 
            
            It must contain the columns ('x', 'y', 'id') with an optional 
            'z' column for segmentation of 3D z-stack data (slice-by-slice) and 
            a 'frame_i' columns for time-lapse data. Note that `id = 0` will 
            be used for the negative points, i.e. those objects (like the 
            background) that should not be segmented.
            
            If not 'None', and there is already an `input_points_df` from the 
            `__init__` (initialization of the model) method it will be 
            overwritten with the new table.
        
        posData : load.loadData or None, optional
            This is not a parameter configurable through the GUI. Cell-ACDC 
            will pass the class of the loaded data from the specific Position. 
            This is the used internally to add image embeddings if 
            `save_embeddings` is True.
        
        save_embeddings : bool, optional
            This is not a parameter configurable through the GUI. If `posData` 
            is not None, the image embeddings will be stored in the dictionary 
            `posData.sam_embeddings`. This dictionary can be later used to 
            save the embeddings to disk.
        
        only_embeddings : bool, optional
            This is not a parameter configurable through the GUI. If `True`, 
            The labels masks will not be generated and the model will only 
            be used to generate the image embeddings stored in 
            `posData.sam_embeddings`.
        
        use_loaded_embeddings : bool, optional      
            This is not a parameter configurable through the GUI. If `posData` 
            is not None, the image embeddings will be loaded from the dictionary 
            `posData.sam_embeddings`.
        
        start_z_slice : int, optional
            This is not a parameter configurable through the GUI. Cell-ACDC 
            will pass the correct start z-slice to store embeddings at the 
            right z-slice.
        
        Returns
        -------
        ([Z], Y, X) numpy.ndarray of ints
            Output labelled masks with the same shape as input image but without 
            the channel dimension. Every pixel belonging to the same object 
            will have the same integer ID. ID = 0 is for the background.
        """        
        
        if isinstance(input_points_df, pd.DataFrame):
            self._input_points_df = input_points_df
        
        is_rgb_image = image.shape[-1] == 3 or image.shape[-1] == 4
        is_z_stack = (image.ndim==3 and not is_rgb_image) or (image.ndim==4)
        if is_rgb_image:
            labels = np.zeros(image.shape[:-1], dtype=np.uint32)
        else:
            labels = np.zeros(image.shape, dtype=np.uint32)
        
        if self._input_points_df is None:
            df_points = None
        elif 'frame_i' in self._input_points_df.columns:
            mask = self._input_points_df['frame_i'] == frame_i
            df_points = self._input_points_df[mask]
        else:
            df_points = self._input_points_df
            
        input_points, input_labels = self._get_input_points(
            is_z_stack, df_points
        )
        if is_z_stack:                
            for z, img in enumerate(image):
                input_points_z = None
                if input_points is not None:
                    input_points_z = input_points.get(z, [])
                    input_labels_z = input_labels.get(z, [])
                
                embeddings_init = False
                if use_loaded_embeddings:
                    embeddings_init = self._get_img_embeddings(
                        posData, frame_i=frame_i, z=z+start_z_slice
                    )
                
                if only_embeddings:
                    self._init_embeddings(img)
                else:
                    labels[z] = self._segment_2D_image(
                        img, input_points_z, input_labels_z, 
                        embeddings_already_init=embeddings_init
                    )
                if save_embeddings or only_embeddings:
                    posData.storeSamEmbeddings(
                        self, frame_i=frame_i, z=z+start_z_slice
                    )
                    
            labels = skimage.measure.label(labels>0)
        else:
            embeddings_init = False
            if use_loaded_embeddings:
                embeddings_init = self._get_img_embeddings(
                    posData, frame_i=frame_i
                )
            if only_embeddings:
                self._init_embeddings(img)
            else:
                labels = self._segment_2D_image(
                    image, input_points, input_labels, 
                    embeddings_already_init=embeddings_init
                )
            
            if save_embeddings or only_embeddings:
                posData.storeSamEmbeddings(self, frame_i=frame_i)
        
        if automatic_removal_of_background and input_points is None:
            labels = self._remove_background(labels)
        
        return labels

    def _get_img_embeddings(self, posData, frame_i=0, z=0):
        img_embeddings = posData.getSamEmbeddings(frame_i=frame_i, z=z)
        if img_embeddings is None:
            return False
        
        for key, value in img_embeddings.items():
            setattr(self, key, value)
        
        return True
    
    def _get_input_points(self, is_z_stack, df_points):
        if df_points is None:
            return None, None
        
        if is_z_stack:
            input_points = defaultdict(dict)
            input_labels = defaultdict(dict)
            neg_input_points_df = (
                df_points[df_points['id'] == 0]
                .set_index('z')
            )
            for (z, id), sub_df in df_points.groupby(['z', 'id']):
                if id == 0:
                    continue
                
                # Concatenate negative points
                points_data_z = sub_df[['x', 'y']].to_numpy()
                points_labels_z = sub_df['id'].to_numpy()
                try:
                    neg_points_data_z = (
                        neg_input_points_df.loc[z][['x', 'y']].to_numpy())
                    points_data_z = np.row_stack((
                        neg_points_data_z, points_data_z
                    ))
                    points_labels_z = np.concatenate(
                        ([0]*len(neg_points_data_z), points_labels_z)
                    )
                except IndexError:
                    pass
                    
                input_points[z][id] = points_data_z
                input_labels[z][id] = points_labels_z
        else:
            input_points = {}
            input_labels = {}
            neg_input_points_df = (
                df_points[df_points['id'] == 0]
            )
            neg_input_points_data = neg_input_points_df[['x', 'y']].to_numpy()
            for id, df_id in df_points.groupby('id'):
                if id == 0:
                    continue
                
                points_data_id = df_id[['x', 'y']].to_numpy()
                points_data_id = np.row_stack((
                    neg_input_points_data, points_data_id
                ))
                
                points_labels_id = df_id['id'].to_numpy()
                points_labels_id = np.concatenate(
                    ([0]*len(neg_input_points_data), points_labels_id)
                )
                input_points[id] = points_data_id
                input_labels[id] = points_labels_id
        
        return input_points, input_labels
    
    def _init_embeddings(self, img_rgb): 
        if img_rgb.ndim == 2:
            img_rgb = myutils.to_uint8(img_rgb)
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            
        # Create embeddings only if new image     
        try:
            init_embeddings = not np.allclose(img_rgb, self._embedded_img)
        except Exception as err:
            init_embeddings = True
        
        if hasattr(self.model, 'predictor'):
            predictor = self.model.predictor
        else:
            predictor = self.model
        
        if init_embeddings: 
            predictor.set_image(img_rgb)
            self._embedded_img = img_rgb
    
    def _segment_2D_image(
            self, image: np.ndarray, 
            input_points: np.ndarray, 
            input_labels: np.ndarray, 
            embeddings_already_init: bool=False
        ) -> np.ndarray:
        
        img = myutils.to_uint8(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = np.zeros(image.shape[:3], dtype=np.uint32)
        
        # SAM Automatic mask generator (no prompts)
        if input_points is None:
            masks = self.model.generate(img)
            for id, mask in enumerate(masks):
                obj_image = mask['segmentation']
                labels[obj_image] = id+1
            
            return labels
        
        # No input points --> return empty labels
        if len(input_points) == 0:
            return labels
        
        # SAM with input points
        if not embeddings_already_init:
            self._init_embeddings(img)
        
        for id, point_coords in input_points.items():
            point_labels = input_labels[id]
            multimask_output = len(point_coords)==1
            masks, scores, logits = self.model.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=multimask_output,
            )
            if multimask_output:
                mask_idx = np.argmax(scores)
            else:
                mask_idx = 0
            mask = masks[mask_idx]
            labels[mask] = id
        return labels

    def _remove_background(self, labels: np.ndarray) -> np.ndarray:
        border_mask = np.ones(labels.shape, dtype=bool)
        border_slice = tuple([slice(2,-2) for _ in range(labels.ndim)])
        border_mask[border_slice] = False
        border_ids, counts = np.unique(labels[border_mask], return_counts=True)
        max_count_idx = list(counts).index(counts.max())
        largest_border_id = border_ids[max_count_idx]
        labels[labels == largest_border_id] = 0
        return labels


def url_help():
    return 'https://github.com/facebookresearch/segment-anything'
