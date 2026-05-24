import os

import numpy as np
import pandas as pd

import pathlib

import skimage.exposure
import skimage.measure

# Add to path submodule to be imported

from yeastmatedetector.inference import YeastMatePredictor

from cellacdc.core import getBaseCca_df
from cellacdc import user_profile_path

class Model:
    def __init__(self):
        model_path = os.path.join(str(user_profile_path), f'acdc-YeastMate')
        yaml_path = os.path.join(model_path, 'yeastmate.yaml')
        weights_path = os.path.join(model_path, 'yeastmate_weights.pth')

        self.model = YeastMatePredictor(
            yaml_path,
            weights_path
        )

    def segment(
            self, image,
            score_threshold_0=0.9,
            score_thresholds_1=0.75,
            score_thresholds_2=0.75,
            pixel_size=110,
            reference_pixel_size=110,
            lower_quantile=1.5,
            upper_quantile=98.5
        ):

        score_thresholds = {
            0: score_threshold_0, 
            1: score_thresholds_1,
            2: score_thresholds_2
        }

        detections, lab = self.model.inference(
            image,
            score_thresholds=score_thresholds,
            pixel_size=pixel_size,
            reference_pixel_size=reference_pixel_size,
            lower_quantile=lower_quantile,
            upper_quantile=upper_quantile
        )
        return lab

    def predictCcaState(self, image, precomputed_lab):
        detections, lab = self.model.inference(image)
        rp = skimage.measure.regionprops(lab)
        IDs = [obj.label for obj in rp]
        areas = [obj.area for obj in rp]
        centroids = [obj.centroid for obj in rp]
        precomputedIDs = [
            obj.label for obj in skimage.measure.regionprops(precomputed_lab)
        ]
        cca_df = getBaseCca_df(precomputedIDs)
        for obj in rp:
            info = detections.get(str(obj.label))
            if info is None:
                continue

            obj_class = info.get('class')
            if len(obj_class) < 2:
                continue

            is_budding = float(obj_class[1])>2
            if not is_budding:
                continue

            links = info.get('links')
            if not links:
                continue

            link = links[0]
            mother_bud_info = detections.get(link)
            if mother_bud_info is None:
                continue

            mother_bud_ids = mother_bud_info.get('links')
            if mother_bud_ids is None:
                continue

            ID1, ID2 = int(mother_bud_ids[0]), int(mother_bud_ids[1])
            idx1 = IDs.index(ID1)
            idx2 = IDs.index(ID2)
            area1 = areas[IDs.index(ID1)]
            area2 = areas[IDs.index(ID2)]
            if area1 > area2:
                moth_y, moth_x = centroids[idx1]
                bud_y, bud_x = centroids[idx2]
            else:
                moth_y, moth_x = centroids[idx2]
                bud_y, bud_x = centroids[idx1]

            mothID = precomputed_lab[int(round(moth_y)), int(round(moth_x))]
            budID = precomputed_lab[int(round(bud_y)), int(round(bud_x))]

            if mothID not in cca_df.index:
                continue

            if budID not in cca_df.index:
                continue

            cca_df.at[mothID, 'relative_ID'] = budID
            cca_df.at[mothID, 'cell_cycle_stage'] = 'S'

            cca_df.at[budID, 'relative_ID'] = mothID
            cca_df.at[budID, 'cell_cycle_stage'] = 'S'
            cca_df.at[budID, 'relationship'] = 'bud'
            cca_df.at[budID, 'generation_num'] = 0
        return cca_df

def url_help():
    return 'https://github.com/hoerlteam/YeastMate/blob/main/examples/python_detection.ipynb'
