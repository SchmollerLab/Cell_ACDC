from pombseen.main import pomBsegNuc

class Model:
    def __init__(self, segm_data):
        self.segm_data = segm_data

    def segment(
            self,
            image,
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
            image, self.segm_data, connectivity, offset, min_size, max_size, 
            max_nuclei, rel_size_max
        )

        return segmented_img