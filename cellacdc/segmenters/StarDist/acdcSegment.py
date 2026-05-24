from stardist.models import StarDist2D, StarDist3D
from csbdeep.utils import normalize
import numpy as np
import os
import skimage.measure

from cellacdc import models

class AvailableModels:
    values = models.STARDIST_MODELS

class Model:
    def __init__(
            self, 
            model_name: AvailableModels='2D_versatile_fluo',
            load_stardist_3D=False
        ):
        """_summary_

        Parameters
        ----------
        model_name : str, optional
            Name of the pre-trained model to load. 
            
            Available models are '2D_versatile_fluo', '2D_versatile_he', and 
            '2D_paper_dsb2018'.
            
            Default is '2D_versatile_fluo'
        """        
        
        stardist_default_models = models.STARDIST_MODELS
        stardist_path = os.path.dirname(os.path.abspath(__file__))
        T_cell_path = os.path.join(stardist_path, 'model', 'T_cell')
        model_class = StarDist3D if load_stardist_3D else StarDist2D
        if not os.path.exists(T_cell_path):
            model_name = stardist_default_models[0]
        # Initialize model
        if model_name in stardist_default_models:
            self.model = model_class.from_pretrained(model_name)
        else:
            script_path = os.path.abspath(__file__)
            stardist_path = os.path.dirname(script_path)
            model_path = os.path.join(stardist_path, 'model')
            self.model = model_class(
                None, name=model_name, basedir=model_path
            )
        self.load_stardist_3D = load_stardist_3D

    def segment(
            self, image, prob_thresh=0.0, nms_thresh=0.0,
            segment_3D_volume=False
        ):
        # Check on image shape
        is2D = image.ndim == 2
        is3D = image.ndim == 3
        calling_stardist3D_on_2D_data = (
            (is3D and self.load_stardist_3D and not segment_3D_volume)
            or is2D and self.load_stardist_3D
        )
        calling_stardist2D_on_3D_data = (
            is3D and not self.load_stardist_3D and segment_3D_volume
        )
        if calling_stardist3D_on_2D_data:
            print('')
            print('='*30)
            raise ValueError(
                'StarDist3D cannot segment 2D image data. If you are trying to '
                'segment z-slices one by one you need to click "True" at the '
                '"Segment 3D Volume" entry.'
            )
        elif calling_stardist2D_on_3D_data:
            print('')
            print('='*30)
            raise ValueError(
                'StarDist2D cannot segment 3D image data. If you are trying to '
                'segment z-slices one by one you need to click "False" at the '
                '"Segment 3D Volume" entry.'
            )

        # Preprocess image
        prob_thresh = prob_thresh if prob_thresh > 0 else None
        nms_thresh = nms_thresh if nms_thresh > 0 else None
        if not segment_3D_volume and image.ndim == 3:
            labels = np.zeros(image.shape, dtype=np.uint32)
            for i, _img in enumerate(image):
                lab, _ = self.model.predict_instances(
                    normalize(_img),
                    prob_thresh=prob_thresh,
                    nms_thresh=nms_thresh
                )
                labels[i] = lab
            labels = skimage.measure.label(labels>0)
        else:
            labels, _ = self.model.predict_instances(
                normalize(image),
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh
            )
        return labels.astype(np.uint32)
