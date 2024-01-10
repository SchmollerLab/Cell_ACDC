from pombseen.main import pomBseg

class Model:
    def __init__(self):
        pass

    def segment(
            self,
            image,
            connectivity = 1,
            sharpen_image = False,
            radius=1.0,
            amount=1.0,
            block_size = 15,
            offset = -2.5,
            inverse_bw_max_pix = 600,
            footprint = 'default',
            clear_border_conn = 8,
            clear_border_max_pix = 1200,
            ConvexFilterSlope = 12.8571,
            ConvexFilterIntercept = 12.5,
            min_size = 500,
            max_size = 100000                
        ):
        """Segment the input `image` and returns a labelled array with the same 
        shape as input image (i.e., instance segmentation).

        Parameters
        ----------
        image : (Y, X) np.ndarray
            Input image
        connectivity : int, optional
            _description_, by default 1
        sharpen_image : bool, optional
            _description_, by default False
        radius : float, optional
            _description_, by default 1.0
        amount : float, optional
            _description_, by default 1.0
        block_size : int, optional
            _description_, by default 15
        offset : float, optional
            Percentage of maximum pixel value, by default -2.5
        inverse_bw_max_pix : int, optional
            _description_, by default 600
        footprint : str, optional
            _description_, by default 'default'
        clear_border_conn : int, optional
            _description_, by default 8
        clear_border_max_pix : int, optional
            _description_, by default 1200
        ConvexFilterSlope : float, optional
            _description_, by default 12.8571
        ConvexFilterIntercept : float, optional
            _description_, by default 12.5
        min_size : int, optional
            _description_, by default 500
        max_size : int, optional
            _description_, by default 100000

        Returns
        -------
        _type_
            _description_
        """        
        if footprint == 'default':
            footprint = None
        
        # Make sure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        segmented_img = pomBseg(
            image, sharpen_image, radius, amount, block_size, offset, 
            footprint, inverse_bw_max_pix, connectivity, clear_border_conn, 
            clear_border_max_pix, ConvexFilterSlope, ConvexFilterIntercept, 
            min_size, max_size
        )

        return segmented_img