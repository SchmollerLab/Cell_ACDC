import os

import torch

class Model:
    def __init__(
            self, 
            gpu: bool = False,
            device: torch.device | int ='None',
            verbose: bool = False,
            torch_n_threads: int = os.cpu_count(),
            *args, 
            **kwargs
        ):
        ...
    
    def add_point_prompt(self, point, *args, **kwargs):
        """Add point prompt to model

        Parameters
        ----------
        point : (z, x, y) coordinates sequence
            The point to add as a prompt.
        """
        ...
    
    def add_prompt(self, prompt, *args, prompt_type='point', **kwargs):
        """Add prompt to model

        Parameters
        ----------
        prompt : np.ndarray
            _description_
        prompt_type : {'point'}, optional
            The type of prompt to add. Default is 'point'.
        """        
        ...
    
    def segment(self, lab, *args, **kwargs):
        ...