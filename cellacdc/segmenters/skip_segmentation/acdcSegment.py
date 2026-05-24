class Model:
    def __init__(self, segm_data):
        self.segm_data = segm_data
        

    def segment(
            self,
            image,
            frame_i,
            skip_segmentation = True,
        ):
        """Skips the segmentation step and instead uses the provided segmentation data.

        Parameters
        ----------
        image : (Y, X) np.ndarray
            Input image for compatibility with other models
        skip_segmentation : int
            skip_segmentation flag to communicate that the segmentation step should be skipped. If it is there, the segmentation step is skipped. Default is True, but it doesnt matter...
        Returns
        -------
        _type_
            Segmented image (same as segm_data)
        """      
        
        return self.segm_data[frame_i]