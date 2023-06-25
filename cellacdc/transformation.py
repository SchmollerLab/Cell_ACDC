import numpy as np

from skimage.transform import resize
from skimage.measure import regionprops

def resize_lab(lab, output_shape, rp=None):
    if rp is None:
        rp = regionprops(lab)
    _lab_obj_to_resize = np.zeros(lab.shape, dtype=np.float16)
    lab_resized = np.zeros(output_shape, dtype=np.uint32)
    for obj in rp:
        _lab_obj_to_resize[obj.slice][obj.image] = 1.0
        _lab_obj_resized = resize(
            _lab_obj_to_resize, output_shape, anti_aliasing=True,
            preserve_range=True
        ).round()
        lab_resized[_lab_obj_resized == 1.0] = obj.label
        _lab_obj_to_resize[:] = 0.0
    return lab_resized