from collections import defaultdict

class BaseModel:
    def __init__(self, *args, **kwargs):
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
        # To be implemented in each model class
        return image
    
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
        