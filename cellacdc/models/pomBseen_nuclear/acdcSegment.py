import numpy as np
from skimage.measure import regionprops
from pombseen.main import pomBsegNuc, norm_img

class Model:
    def __init__(self, segm_data):
        self.segm_data = segm_data

    def _thresholdable_segm(self, image, frame_i=None):
        """Drop objects that pomBsegNuc cannot threshold.

        pomBsegNuc builds a per-object histogram (including the 0 background 
        of the mask) and drops its first bin, so objects need at least 3 
        distinct values, otherwise Otsu fails on an empty histogram.
        """
        segm = np.asarray(self.segm_data)
        if segm.ndim == image.ndim + 1:
            # Time-lapse auxiliary segmentation
            segm = segm[0 if frame_i is None else frame_i]
        segm = segm.copy()
        if segm.shape != image.shape:
            raise ValueError(
                f'pomBseen_nuclear: auxiliary segmentation shape {segm.shape} '
                f'does not match the image shape {image.shape}.'
            )
        norm = norm_img(image)
        skipped = []
        for obj in regionprops(segm):
            values = np.union1d(norm[obj.slice][obj.image], [0])
            if values.size < 3:
                segm[obj.slice][obj.image] = 0
                skipped.append(obj.label)
        if skipped:
            print(
                'pomBseen_nuclear: skipped objects with too few intensity '
                f'values to threshold: {skipped}'
            )
        return segm

    def segment(
            self,
            image,
            frame_i,
            connectivity = 1,
            offset = 0,
            min_size=5,
            max_size=100000,
            max_nuclei = 2,
            rel_size_max = 0.3         
        ):
        """Segment the input `image` and returns a labelled array with the same 
        shape as input image (i.e., instance segmentation).

        Parameters
        ----------
        image : (Y, X) np.ndarray
            Input image
        offset : float
            Percentage of maximum pixel value. Default is 0
        min_size : int
            Min size of nucleus. Default is 0
        max_size : int
            Max size of nucleus. Default is 100000
        max_nuclei : int
            Maximum number of nuclei per cell. Default is 2
        rel_size_max : float
            Maximum size of nucleus in respect to cell. Default is 0.3

        Returns
        -------
        _type_
            Segmented image
        """       

        segmented_img = pomBsegNuc(
            image, self._thresholdable_segm(image, frame_i), connectivity, offset, min_size, max_size, 
            max_nuclei, rel_size_max
        )

        return segmented_img