import numpy as np

from baby import modelsets
from baby import BabyCrawler

from cellacdc import myutils
from cellacdc.trackers import BABY
from cellacdc.trackers.BABY import BABY_tracker

class AvailableModels:
    values = BABY.BABY_MODELS

class Model:
    def __init__(
            self, 
            model_name: AvailableModels='yeast-alcatras-brightfield-sCMOS-60x-5z',
        ):
        self.tracker = BABY_tracker.tracker(model_name)
    
    def segment(
            self, image, 
            refine_outlines=True,
            swap_YX_axes_to_XY=True,
        ):
        Y, X = image.shape[-2:]
        lab = np.zeros((Y, X), dtype=np.uint32)
        
        image = self.tracker._preprocess(image, swap_YX_axes_to_XY)
        
        result_generator = self.tracker.crawler.baby_brain.segment(
            image[None, ...], 
            pixel_size=None,
            overlap_size=48,
            yield_edgemasks=False,
            yield_masks=True,
            yield_preds=False,
            yield_volumes=False,
            refine_outlines=refine_outlines,
            yield_rescaling=False,
            keep_bb_pixel_size=False
        )
        
        for result in result_generator:
            masks = result['masks']
            for i, mask in enumerate(masks):
                if swap_YX_axes_to_XY:
                    mask = np.swapaxes(mask, 0, 1)
                lab[mask] = i+1
        
        return lab