import skimage.filters

class Model:
    def __init__(self):
        pass
    
    def segment(self, image, gauss_sigma=1.0, threshold_method='threshold_otsu'):
        if gauss_sigma > 0:
            image = skimage.filters.gaussian(image, sigma=gauss_sigma)
        
        thresh_val = getattr(skimage.filters, threshold_method)(image)
        labels = skimage.measure.label(image > thresh_val)
        return labels