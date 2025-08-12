from typing import List

import numpy as np

import skimage.measure

def insert_model_output_into_labels(
        lab, 
        model_out, 
        edited_IDs: int | List[int] = 0,
    ):
    rp_lab = skimage.measure.regionprops(lab)
    rp_model_out = skimage.measure.regionprops(model_out)
    lab_union = lab.copy()
    lab_interesection = lab.copy()
    lab_new = lab.copy()
    
    if isinstance(edited_IDs, int) and edited_IDs == 0:
        edited_IDs = []
    
    for edited_ID in edited_IDs:
        if edited_ID == 0:
            continue
        
        rp_mapper = {obj.label: obj for obj in rp_lab}
        obj = rp_mapper.get(edited_ID, None)
        if obj is not None:
            lab_new[obj.slice][obj.image] = 0
    
    for obj_out in rp_model_out:
        lab_new[obj_out.slice][obj_out.image] = obj_out.label
        lab_union[obj_out.slice][obj_out.image] = obj_out.label
        
        intersect_mask = np.logical_and(lab[obj_out.slice] > 0, obj_out.image)
        lab_interesection[obj_out.slice][intersect_mask] = obj_out.label
    
    return lab_new, lab_union, lab_interesection
    