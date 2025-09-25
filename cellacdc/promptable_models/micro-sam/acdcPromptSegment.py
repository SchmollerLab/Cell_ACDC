# precompute_image_embeddings see micro_sam.util.precompute_image_embeddings
# segment_from_points see micro_sam.prompt_based_segmentation.segment_from_points
# get_sam_model see micro_sam.util.get_sam_model
# Also see micro_sam.promptable_segmentation in the main function
import os
import traceback

import numpy as np

from micro_sam.util import get_sam_model, precompute_image_embeddings
from micro_sam.prompt_based_segmentation import segment_from_points

from cellacdc import load
from cellacdc.promptable_models.base import BaseModel

class AvailableModels:
    from micro_sam.util import get_model_names
    values = get_model_names()

class GPUorCPU:
    values = ['gpu', 'cpu']

class Model(BaseModel):
    def __init__(
        self,
        model_type: AvailableModels = 'vit_b_lm',
        save_embeddings: bool = True,
        image_embeddings_zarr_path: os.PathLike = '',
        **kwargs
    ):
        super().__init__()
        
        self.predictor = get_sam_model(model_type)
        
        if not image_embeddings_zarr_path:
            image_embeddings_zarr_path = None
        
        if save_embeddings and os.path.exists(image_embeddings_zarr_path):
            try:
                os.remove(image_embeddings_zarr_path)
            except Exception as err:
                print(traceback.format_exc())
                raise PermissionError(
                    'Saving embeddings when the ZARR file already exists '
                    'requires removing the file firs. '
                    'Please, manually remove the following file: '
                    f'{image_embeddings_zarr_path} '
                )
        
        do_not_save_but_path_exists = (
            not save_embeddings
            and image_embeddings_zarr_path is not None
            and os.path.exists(image_embeddings_zarr_path)
        )
        
        if do_not_save_but_path_exists:
            raise ValueError(
                'The path to save the image embeddings already exists: '
                f'{image_embeddings_zarr_path}. '
                'In order to not save the embeddings, please, '
                'provide an empty path.'
            )
        
        if save_embeddings and image_embeddings_zarr_path is None:
            posData: load.loadData = kwargs.get('posData', None)
            if posData is None:
                raise ValueError(
                    'If `save_embeddings` is True, '
                    '`image_embeddings_zarr_path` must be provided.'
                )
            
            image_embeddings_zarr_path = posData.microSamEmbeddingsZarrPath()
        
        self.image_embeddings_zarr_path = image_embeddings_zarr_path
    
    def _get_points_and_labels(self, prompt_id, z_slice=1):
        points = []
        labels = []
        prompts = self.prompts[prompt_id]
        for prompt, prompt_type in prompts:
            if prompt_type == 'point':
                z, y, x = prompt
                if z != z_slice:
                    continue
                
                points.append((y, x))
                labels.append(1)  # Positive point
            else:
                raise ValueError(f"Unsupported prompt type: {prompt_type}")

        obj_negative_prompts = self.negative_prompts[prompt_id]
        for prompt, prompt_type in obj_negative_prompts:
            if prompt_type == 'point':
                z, y, x = prompt
                if z != z_slice:
                    continue
                
                points.append((y, x))
                labels.append(0)  # Negative point
            else:
                raise ValueError(f"Unsupported prompt type: {prompt_type}")
        
        global_negative_prompts = self.negative_prompts[0]
        for prompt, prompt_type in global_negative_prompts:
            if prompt_type == 'point':
                z, y, x = prompt
                if z != z_slice:
                    continue
                
                points.append((y, x))
                labels.append(0)  # Negative point
            else:
                raise ValueError(f"Unsupported prompt type: {prompt_type}")      
        
        return points, labels
        
    def segment(
            self, 
            image, 
            *args, 
            treat_other_objects_as_background=True,
            verbose=False,
            **kwargs
        ):
        """Run the segmentation model on the image using the prompts added

        Parameters
        ----------
        image : (Z, Y, X) np.ndarray
            3D z-stack image to segment. For 2D images, pass a 3D array 
            with a single slice (i.e., shape (1, Y, X)).
        treat_other_objects_as_background : bool, optional
            If True, when segmenting an object, the prompts added 
            for all the other objects are treated as negative prompts
            for the current object. Default is True
        verbose : bool, optional
            If True, print progress messages. Default is False.
            
        Returns
        -------
        (Z, Y, X) np.ndarray
            Labelled array with the segmentation masks of the objects. 
            Smaller objects are added on top to prevent larger 
            objects from removing smaller ones.
        """
        
        result_obj_masks = []
        lab = np.zeros(image.shape, dtype=np.uint32)        
        for prompt_id, value in self.prompt_ids_image_mapper.items():
            prompt_image, image_origin = value
            
            if prompt_image is None:
                prompt_image = image
            
            ndim = 3
            SizeZ = prompt_image.shape[0]
            input_image = prompt_image
            if SizeZ == 1:
                input_image = prompt_image[0]
                ndim = 2

            # If save_path is not None, and path exists, embeddings are loaded
            image_embeddings = precompute_image_embeddings(
                predictor=self.predictor,
                input_=input_image,
                ndim=ndim, 
                verbose=verbose,
                save_path=self.image_embeddings_zarr_path # If not None, and path exists, embeddings are loaded
            )
            
            import pdb; pdb.set_trace()
            
            for z, img in enumerate(prompt_image):
                ...

def url_help():
    return 'https://computational-cell-analytics.github.io/micro-sam/micro_sam.html'