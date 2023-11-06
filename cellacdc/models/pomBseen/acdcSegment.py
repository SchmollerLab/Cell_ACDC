from pombseen.main import pomBseg

class Model:
    def __init__(self):
        pass

    def segment(
            self,
            ### maybe we consider using skimage.filters.try_all _threshold to allow user to decide on threshold used in future

            ### General
            image,
            connectivity = 1,

            ### image sharpening params
            sharpen_image = False,
            radius=1.0,
            amount=1.0,

            ### thresh_binarize params
            block_size = 15,
            offset = -5.5, # This will be subtracted from the threshold. The basic result is similar to 'sensitivity' from Matlab's 'adaptthresh'
            inverse_bw_max_pix = 600,
            footprint = 'default',

            ### Clearborder params
            clear_border_conn = 8,
            clear_border_max_pix = 1200,

            ### convex_filter params
            ConvexFilterSlope = 12.8571,
            ConvexFilterIntercept = 12.5,
            min_size = 500,
            max_size = 100000                
        ):
        ### Here starts the actual calculation. All functions are defined above (except loading image). The functions have the same name as the matplotlib modules, leading to sometimes misleading names

        if footprint == 'default':
            footprint = None
        
        segmented_img = pomBseg(image, sharpen_image, radius, amount, block_size, offset, footprint, inverse_bw_max_pix, connectivity, clear_border_conn, clear_border_max_pix, ConvexFilterSlope, ConvexFilterIntercept, min_size, max_size)

        return segmented_img