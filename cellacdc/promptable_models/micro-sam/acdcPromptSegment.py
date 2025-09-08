# precompute_image_embeddings see micro_sam.util.precompute_image_embeddings
# segment_from_points see micro_sam.prompt_based_segmentation.segment_from_points
# get_sam_model see micro_sam.util.get_sam_model
# Also see micro_sam.promptable_segmentation in the main function

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
        image_embeddings_zarr_path: os.PathLike = '',
    ):
        super().__init__()
    
    def segment(
            self, 
            image, 
            treat_other_objects_as_background=True,
            *args, 
            **kwargs
        ):
        ...

def url_help():
    return 'https://computational-cell-analytics.github.io/micro-sam/micro_sam.html'