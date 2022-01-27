import numpy as np
# from cellpose.dynamics import SKIMAGE_ENABLED #circular import error 
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from scipy.ndimage import label # used as alternative to skimage.measure.label, need to test speed and output...
import fastremap



try:
    from skimage import measure
    SKIMAGE_ENABLED = True 
except:
    SKIMAGE_ENABLED = False

def normalize_field(mu):
    mag = np.sqrt(np.nansum(mu**2,axis=0))
    m = mag>0
    mu = np.divide(mu, mag, out=np.zeros_like(mu), where=np.logical_and(mag!=0,~np.isnan(mag)))        
    return mu

def normalize99(Y,lower=0.01,upper=99.99):
    """ normalize image so 0.0 is 0.01st percentile and 1.0 is 99.99th percentile """
    X = Y.copy()
    return np.interp(X, (np.percentile(X, lower), np.percentile(X, upper)), (0, 1))

def normalize_image(im,mask,bg=0.5):
    """ Normalize image by rescaling fro 0 to 1 and then adjusting gamma to bring average background to bg."""
    im = rescale(im)
    return im**(np.log(bg)/np.log(np.mean(im[binary_erosion(mask==0)])))

# Generate a color dictionary for use in visualizing N-colored labels.  
def sinebow(N,bg_color=[0,0,0,0]):
    colordict = {0:bg_color}
    for j in range(N): 
        angle = j*2*np.pi / (N)
        r = ((np.cos(angle)+1)/2)
        g = ((np.cos(angle+2*np.pi/3)+1)/2)
        b = ((np.cos(angle+4*np.pi/3)+1)/2)
        colordict.update({j+1:[r,g,b,1]})
    return colordict

def rescale(T):
    """Rescale array between 0 and 1"""
    T = np.interp(T, (T[:].min(), T[:].max()), (0, 1))
    return T

# Kevin's version of remove_edge_masks, need to merge (this one is more flexible)
def clean_boundary(labels,boundary_thickness=3,area_thresh=30):
    """Delete boundary masks below a given size threshold. Default boundary thickness is 3px,
    meaning masks that are 3 or fewer pixels from the boudnary will be candidates for removal. 
    """
    border_mask = np.zeros(labels.shape, dtype=bool)
    border_mask = binary_dilation(border_mask, border_value=1, iterations=boundary_thickness)
    clean_labels = np.copy(labels)
    for cell_ID in np.unique(labels):
        mask = labels==cell_ID 
        area = np.count_nonzero(mask)
        overlap = np.count_nonzero(np.logical_and(mask, border_mask))
        if overlap > 0 and area<area_thresh and overlap/area >= 0.5: #only premove cells that are 50% or more edge px
            clean_labels[mask] = 0
    return clean_labels


# Should work for 3D too. Could put into usigned integer form at the end... 
# Also could use some parallelization 

def format_labels(labels, clean=False, min_area=9, verbose=True):
    """
    Puts labels into 'standard form', i.e. background=0 and cells 1,2,3,...,N-1,N.
    Optional clean flag: disconnect and disjoint masks and discard small masks beflow min_area. 
    min_area default is 9px. 
    """
    
    # Labels are stored as a part of a float array in Cellpose, so it must be cast back here.
    # some people also use -1 as background, so we must cast to the signed integar class. We
    # can safely assume no 2D or 3D image will have more than 2^31 cells. Finally, cv2 does not
    # play well with unsigned integers (saves to default uint8), so we cast to uint32. 
    labels = labels.astype('int32') 
    labels -= np.min(labels) 
    labels = labels.astype('uint32') 
    
    # optional cleanup 
    if clean:
        inds = np.unique(labels)
        for j in inds[inds>0]:
            mask = labels==j
            if SKIMAGE_ENABLED:
                lbl = measure.label(mask)                       
                regions = measure.regionprops(lbl)
                regions.sort(key=lambda x: x.area, reverse=True)
                if len(regions) > 1:
                    if verbose:
                        print('Warning - found mask with disjoint label.')
                    for rg in regions[1:]:
                        if rg.area <= min_area:
                            labels[rg.coords[:,0], rg.coords[:,1]] = 0
                            if verbose:
                                print('secondary disjoint part smaller than min_area. Removing it.')
                        else:
                            if verbose:
                                print('secondary disjoint part bigger than min_area, relabeling. Area:',rg.area, 
                                        'Label value:',np.unique(labels[tuple(rg.coords.T)]))
                            labels[tuple(rg.coords.T)] = np.max(labels)+1
                            
                rg0 = regions[0]
                if rg0.area <= min_area:
                    labels[tuple(rg0.coords.T)] = 0
                    if verbose:
                        print('Warning - found mask area less than', min_area)
                        print('Removing it.')
            else:
                connectivity_shape = np.array([3 for i in range(mask.ndim)])
                lbl = label(mask, connectivity=np.ones(connectivity_shape))[0]
                labels = lbl
        
    fastremap.renumber(labels,in_place=True) # convenient to have unit increments from 1 to N cells
    labels = fastremap.refit(labels) # put into smaller data type if possible 
    return labels

# get the number of m-dimensional hypercubes connected to the n-cube
def cubestats(n):
    faces = []
    for m in range(n+1):
          faces.append((2**(n-m))*math.comb(n,m))
    return faces


