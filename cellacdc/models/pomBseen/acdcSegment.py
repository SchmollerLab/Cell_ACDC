from pombseen.main import pomBseg

class Model:
    def __init__(self):
        pass
    def segment(
            self,
            image,
            offset = -2.5,
            connectivity_remove_small_objects_inverse_bw = 1,
            connectivity_label = 1,
            connectivity_remove_small_objects_binarize = 1,
            sharpen_image = False,
            radius=1.0,
            amount=1.0,
            block_size = 15,
            min_pix_inverse_bw = 600,
            min_pix_inverted_inverse_bw = 600,
            min_pix_thresh_binarize = 600,
            footprint = 'default',
            clear_border_buffer = 2,
            clear_border_max_pix = 1200,
            convex_filter_slope = 12.8571,
            convex_filter_intercept = 12.5,
            min_size = 500,
            max_size = 100000,
            apply_convex_hull = False,
        ):
        """Segment the input `image` and returns a labelled array with the same 
        shape as input image (i.e., instance segmentation).

        Parameters
        ----------
        image : (Y, X) np.ndarray
            Input image
        offset : float, optional
            Percentage of maximum pixel value. Used in threshold_local. See skimage.filters for more info. By default -2.5
        connectivity_remove_small_objects_inverse_bw : (1, 2, ..., dim of data), optional
            Connectivity used in inverse_bw for remove_small_objects. See skimage.morphology for more info. By default 1
        connectivity_label : (1, 2, ..., dim of data), optional
            Connectivity used in convex_filter for measure.label. See skimage.morphology for more info. By default 1
        connectivity_remove_small_objects_binarize : (1, 2, ..., dim of data), optional
            Connectivity used in thresh_binarize for remove_small_objects. See skimage.morphology for more info. By default 1
        sharpen_image : bool, optional
            Should unsharp_mask be applied to sharpen the image? See skimage.filters for more info. By default False
        radius : float, optional
            radius for unsharp_mask. See skimage.filters for more info. By default 1.0
        amount : float, optional
            amount for unsharp_mask. See skimage.filters for more info. By default 1.0
        block_size : odd int, optional
            Block size for threshold_local. See skimage.filters for more info. By default 15
        min_pix_inverse_bw : int, optional
            min_size used in inverse_bw for remove_small_objects(min_size). See skimage.morphology for more info. By default 600
        min_pix_inverted_inverse_bw : int, optional
            Applied on the inverted img. min_size used in inverse_bw for remove_small_objects(min_size). See skimage.morphology for more info. By default 600
        min_pix_thresh_binarize : int, optional
            Used in thresh_binarize for remove_small_objects(min_size). See skimage.morphology for more info. By default 600
        footprint : str, optional
            Footprint used in binary_dilation and binary_closing in thresh_binarize. See skimage.morphology for more info. By default 'default'
        clear_border_buffer : int, optional
            Buffer (area from edge to consider) used in clear_border. See skimage.segmentation.clear_border for more info. By default 2
        clear_border_max_pix : int, optional
            Max pixels used for remove_small_objects in clear_border. See skimage.morphology for more info. By default 1200
        convex_filter_slope : float, optional
            Slope of the convex filter applied in convex_filter. By default 12.8571
        convex_filter_intercept : float, optional
            Intercept of the convex filter applied in convex_filter. By default 12.5
        min_size : int, optional
            Min pixel size used to filter in convex_filter. By default 500
        max_size : int, optional
            Max pixel size used to filter in convex_filter. By default 100000
        apply_convex_hull : int, optional
            If a convex_hull on each object should be applied. Checks for overlaps and discards if present. Highly personal preference if enabled or not. For more info see skimage.morphology.convex_hull_image. By default False

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
        
        segmented_img = pomBseg(image, 
                                sharpen_image, 
                                radius, 
                                amount, 
                                block_size, 
                                offset, 
                                footprint,
                                min_pix_inverse_bw,
                                min_pix_inverted_inverse_bw, 
                                min_pix_thresh_binarize, 
                                connectivity_remove_small_objects_inverse_bw, 
                                connectivity_label, 
                                connectivity_remove_small_objects_binarize, 
                                clear_border_buffer, 
                                clear_border_max_pix, 
                                convex_filter_slope, 
                                convex_filter_intercept, 
                                min_size, 
                                max_size,
                                apply_convex_hull,
        )

        return segmented_img