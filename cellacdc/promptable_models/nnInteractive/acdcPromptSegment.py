import os
from collections import defaultdict
import heapq

import numpy as np

import torch

from cellacdc import user_profile_path
from cellacdc import myutils
from cellacdc import printl

from huggingface_hub import snapshot_download

class AvailableModels:
    values = ['nnInteractive_v1.0']

class GPUorCPU:
    values = ['gpu', 'cpu']

class Model:
    def __init__(
            self, 
            model_name: AvailableModels = 'nnInteractive_v1.0',
            run_on: GPUorCPU = 'cpu',
            device: torch.device | int ='None',
            verbose: bool = False,
            torch_number_of_threads: int = os.cpu_count(),
            **kwargs
        ):
        """_summary_

        Parameters
        ----------
        model_name : AvailableModels, optional
            nnInterative model to use. Default is 'nnInteractive_v1.0'
        run_on : {'cpu', 'gpu'}, optional
            Whether to run on CPU or first GPU available. Default is 'cpu'
        device : torch.device or int or None
            If not None, this is the device used for running the model
            (torch.device('cuda') or torch.device('cpu')). 
            It overrides `run_on`, recommended if you want to use a specific GPU 
            (e.g. torch.device('cuda:1'). Default is None
        verbose : bool, optional
            If True, more information will be displayed in the terminal. 
            Default is False
        torch_number_of_threads : int, optional
            Number of CPU threads to use for the computation. 
            Default is `os.cpu_count()`, i.e., the maximum available CPU cores.
        """      
        from nnInteractive.inference.inference_session import (
            nnInteractiveInferenceSession
        )
        
        if device == 'None':
            device = None
            
        if device is None:
            device = myutils.get_torch_device(gpu=run_on == 'gpu')
                      
        self.model = nnInteractiveInferenceSession(
            device=device,  # Set inference device
            use_torch_compile=False,  # Experimental: Not tested yet
            verbose=verbose,
            torch_n_threads=torch_number_of_threads,  # Use available CPU cores
            do_autozoom=True,  # Enables AutoZoom for better patching
            use_pinned_memory=True,  # Optimizes GPU memory transfers
        )
        
        download_dir = os.path.join(user_profile_path, 'acdc-nnInteractive')
        os.makedirs(download_dir, exist_ok=True)
        
        download_path = snapshot_download(
            repo_id='nnInteractive/nnInteractive',
            allow_patterns=[f"{model_name}/*"],
            local_dir=download_dir
        )
        
        model_path = os.path.join(download_dir, model_name)
        
        self.model.initialize_from_trained_model_folder(model_path)
        
        self.prompt_ids_image_mapper = {}
        self.prompts = defaultdict(list)
        self.negative_prompts = defaultdict(list)
    
    def _validate_prompt(self, prompt, prompt_type='point'):
        if prompt_type == 'point':
            prompt = tuple(prompt)
            if len(prompt) != 3:
                raise ValueError(
                    "Point prompt must be a sequence of 3 coordinates (z, y, x)."
                )
    
    def _validate_image(self, image):
        if image is None:
            return
        
        if image.ndim == 3:
            return
        
        raise ValueError(
            "Only 3D images are supported by nnInteractive. "
            "Please provide a 3D image with (Z, Y, X) dimensions."
        )
    
    def add_prompt(
            self, 
            prompt, 
            prompt_id: int, 
            *args, 
            image=None, 
            image_origin=(0, 0, 0),
            parent_obj_id=0,
            prompt_type='point', 
            **kwargs
        ):
        """Add prompt to model

        Parameters
        ----------
        prompt : np.ndarray
            Prompt to add. If 'point', this should be a sequence of 3 coordinates
            (z, y, x).
        prompt_id : int
            Unique identifier for the prompt. If 0, then it will be treated as a 
            negative prompt (i.e., the background).
        image : np.ndarray, optional
            Image to which the prompt is associated. If None, the prompt will 
            be associated to the entire image passed to the `segment` method.
        image_origin : tuple of (z0, y0, x0) coordinates, optional
            Origin of the image in the global image coordinate system. This 
            is useful when you want to pass a crop of the image to the model, 
            but still have the result inserted into the global image by 
            the `segment` method. Default is (0, 0, 0).
        parent_obj_id : int, optional
            The ID of the parent object. If not 0, this will be used to assign 
            negative prompts only to the parent object. 
        prompt_type : {'point'}, optional
            The type of prompt to add. Default is 'point'.
        """             
        self._validate_prompt(prompt, prompt_type=prompt_type)
        self._validate_image(image)
        
        if prompt_id not in self.prompt_ids_image_mapper and prompt_id != 0:
            self.prompt_ids_image_mapper[prompt_id] = (image, image_origin)
        
        if prompt_id != 0:
            self.prompts[prompt_id].append(
                (prompt, prompt_type) 
            )
        elif parent_obj_id != 0:
            # Negative prompt for a specific parent object
            self.negative_prompts[parent_obj_id].append(
                (prompt, prompt_type)
            )
        else:
            # Negative prompt for the background
            self.negative_prompts[0].append((prompt, prompt_type))
        
    def _add_object_prompts(self, prompt_id, is_negative=False):
        prompts = self.prompts[prompt_id]
        for prompt, prompt_type in prompts:
            if prompt_type == 'point':
                # nnInteractive requires (x, y, z) order
                point_prompt = tuple(prompt[::-1])
                self.model.add_point_interaction(
                    point_prompt,
                    include_interaction=not is_negative,
                    run_prediction=True
                )
            else:
                raise ValueError(f"Unsupported prompt type: {prompt_type}")
    
    def _add_object_specific_negative_prompts(self, prompt_id):
        obj_negative_prompts = self.negative_prompts[prompt_id]
        for prompt, prompt_type in obj_negative_prompts:
            if prompt_type == 'point':
                # nnInteractive requires (x, y, z) order
                point_prompt = tuple(prompt[::-1])
                self.model.add_point_interaction(
                    point_prompt,
                    include_interaction=False,
                    run_prediction=True
                )
            else:
                raise ValueError(f"Unsupported prompt type: {prompt_type}")
    
    def _add_global_negative_prompts(self):
        global_negative_prompts = self.negative_prompts[0]
        for prompt, prompt_type in global_negative_prompts:
            if prompt_type == 'point':
                # nnInteractive requires (x, y, z) order
                point_prompt = tuple(prompt[::-1])
                self.model.add_point_interaction(
                    point_prompt,
                    include_interaction=False,
                    run_prediction=True
                )
            else:
                raise ValueError(f"Unsupported prompt type: {prompt_type}")
    
    def _add_other_objects_prompts_as_negative(self, current_prompt_id):
        for prompt_id, prompts in self.prompts.items():
            if prompt_id == current_prompt_id:
                continue
            
            self._add_object_prompts(prompt_id, is_negative=True)
    
    def segment(
            self, 
            image, 
            treat_other_objects_as_background=True,
            *args, 
            **kwargs
        ):
        """Run the segmentation model on the image using the prompts added

        Parameters
        ----------
        image : (Z, Y, X) np.ndarray
            3D z-stack image to segment.
        treat_other_objects_as_background : bool, optional
            If True, when segmenting an object, the prompts added 
            for all the other objects are treated as negative prompts
            for the current object. Default is True

        Returns
        -------
        (Z, Y, X) np.ndarray
            Labelled array with the segmentation masks of the objects. 
            Smaller objects are added on top to prevent larger 
            objects from removing smaller ones.

        Raises
        ------
        ValueError
            Error raised if the image is not 3D.
        """
        self._validate_image(image)

        result_obj_masks = []
        lab = np.zeros(image.shape, dtype=np.uint32)        
        for prompt_id, value in self.prompt_ids_image_mapper.items():
            prompt_image, image_origin = value
            
            if prompt_image is None:
                prompt_image = image
                
            # Re-order axis from (z, y, x) to (x, y, z) for the model
            prompt_image = np.moveaxis(prompt_image, (0, 1, 2), (2, 1, 0))
            
            prompt_image = prompt_image[np.newaxis]
            self.model.set_image(prompt_image)
            
            target_tensor = torch.zeros(
                prompt_image.shape[1:], dtype=torch.uint8
            ) 
            self.model.set_target_buffer(target_tensor)
            
            self._add_object_prompts(prompt_id, is_negative=False)
            self._add_object_specific_negative_prompts(prompt_id)
            self._add_global_negative_prompts()
            
            if treat_other_objects_as_background:
                # Add the other objects prompts as negative
                self._add_other_objects_prompts_as_negative(prompt_id)
            
            # self.model._predict()
            
            result_tensor = target_tensor.clone()
            
            # Convert to numpy array and re-order axis back to (z, y, x)
            result_mask = np.moveaxis(
                result_tensor.numpy(), (2, 1, 0), (0, 1, 2)
            ).astype(bool)
            
            # Insert the result into the global label array
            z0, y0, x0 = image_origin
            d, h, w = result_mask.shape
            z1, y1, x1 = z0 + d, y0 + h, x0 + w
            
            obj_vol = np.count_nonzero(result_mask)
            obj_slice = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
            
            item = (-obj_vol, (prompt_id, obj_slice, result_mask))
            heapq.heappush(result_obj_masks, item)
            
            self.model.reset_interactions()
        
        # Now we need to insert the objects in the order of their volume
        for i in range(len(result_obj_masks)):
            item = heapq.heappop(result_obj_masks)
            obj_vol, (prompt_id, obj_slice, result_mask) = item
            lab[obj_slice][result_mask] = prompt_id
            
        self.prompt_ids_image_mapper = {}
        self.prompts = defaultdict(list)
        self.negative_prompts = defaultdict(list)
        
        return lab

def url_help():
    return 'https://github.com/MIC-DKFZ/nnInteractive'