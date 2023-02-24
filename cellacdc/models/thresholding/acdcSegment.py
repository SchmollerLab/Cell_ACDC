import numpy as np

import skimage.filters

from cellacdc import printl

class Model:
    def __init__(self):
        pass
    
    def _preprocess(self, img, gauss_sigma):
        if gauss_sigma > 0:
            filtered = skimage.filters.gaussian(img, sigma=gauss_sigma)
            return filtered
        else:
            return img

    def _apply_threshold(self, img, threshold_method):
        thresh_val = getattr(skimage.filters, threshold_method)(img)
        return img > thresh_val
    
    def segment(
            self, image, gauss_sigma=1.0, 
            threshold_method='threshold_otsu',
            segment_3D_volume=False 
        ):
        is3D = image.ndim > 2
        if is3D and not segment_3D_volume:
            # Segment slice-by-slice
            thresh = np.zeros(image.shape, dtype=bool)
            for z, img in enumerate(image):
                filtered = self._preprocess(img, gauss_sigma)
                _thresh = self._apply_threshold(filtered, threshold_method)
                thresh[z] = _thresh
        else:
            filtered = self._preprocess(image, gauss_sigma)
            thresh = self._apply_threshold(filtered, threshold_method)
        
        labels = skimage.measure.label(thresh)

        return labels