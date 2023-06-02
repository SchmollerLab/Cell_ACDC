import re
import traceback

import skimage.segmentation
import skimage.measure

from collections.abc import Callable, Sequence
from pyqtgraph.colormap import ColorMap

import numpy as np
import matplotlib
import colorsys
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

try:
    import networkx as nx
    NETWORKX_INSTALLED = True
except:
    NETWORKX_INSTALLED = False

__all__ = ['ColorMap']
_mapCache = {}
def getFromMatplotlib(name):
    """
    Added to pyqtgraph 0.12 copied/pasted here to allow pyqtgraph <0.12. Link:
    https://pyqtgraph.readthedocs.io/en/latest/_modules/pyqtgraph/colormap.html#get
    Generates a ColorMap object from a Matplotlib definition.
    Same as ``colormap.get(name, source='matplotlib')``.
    """
    # inspired and informed by "mpl_cmaps_in_ImageItem.py", published by Sebastian Hoefer at
    # https://github.com/honkomonk/pyqtgraph_sandbox/blob/master/mpl_cmaps_in_ImageItem.py
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    cm = None
    col_map = plt.get_cmap(name)
    if hasattr(col_map, '_segmentdata'): # handle LinearSegmentedColormap
        data = col_map._segmentdata
        if ('red' in data) and isinstance(data['red'], (Sequence, np.ndarray)):
            positions = set() # super-set of handle positions in individual channels
            for key in ['red','green','blue']:
                for tup in data[key]:
                    positions.add(tup[0])
            col_data = np.zeros((len(positions),4 ))
            col_data[:,-1] = sorted(positions)
            for idx, key in enumerate(['red','green','blue']):
                positions = np.zeros( len(data[key] ) )
                comp_vals = np.zeros( len(data[key] ) )
                for idx2, tup in enumerate( data[key] ):
                    positions[idx2] = tup[0]
                    comp_vals[idx2] = tup[1] # these are sorted in the raw data
                col_data[:,idx] = np.interp(col_data[:,3], positions, comp_vals)
            cm = ColorMap(pos=col_data[:,-1], color=255*col_data[:,:3]+0.5)
        # some color maps (gnuplot in particular) are defined by RGB component functions:
        elif ('red' in data) and isinstance(data['red'], Callable):
            col_data = np.zeros((64, 4))
            col_data[:,-1] = np.linspace(0., 1., 64)
            for idx, key in enumerate(['red','green','blue']):
                col_data[:,idx] = np.clip( data[key](col_data[:,-1]), 0, 1)
            cm = ColorMap(pos=col_data[:,-1], color=255*col_data[:,:3]+0.5)
    elif hasattr(col_map, 'colors'): # handle ListedColormap
        col_data = np.array(col_map.colors)
        cm = ColorMap(pos=np.linspace(0.0, 1.0, col_data.shape[0]),
                      color=255*col_data[:,:3]+0.5 )
    if cm is not None:
        cm.name = name
        _mapCache[name] = cm
    return cm

def get_pg_gradient(colors):
    ticks_pos = np.linspace(0,1,len(colors))
    ticks = [(tick_pos, color) for tick_pos, color in zip(ticks_pos, colors)]
    gradient = {'ticks': ticks, 'mode': 'rgb'}
    return gradient

def lighten_color(color, amount=0.3, hex=True):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = matplotlib.colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))
    lightened_c = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    if hex:
        lightened_c = tuple([int(round(v*255)) for v in lightened_c])
        lightened_c = '#%02x%02x%02x' % lightened_c
    return lightened_c

def rgb_str_to_values(rgbString, errorRgb=(255,255,255)):
    try:
        r, g, b = re.findall(r'(\d+), (\d+), (\d+)', rgbString)[0]
        r, g, b = int(r), int(g), int(b)
    except TypeError:
        try:
            r, g, b = rgbString
        except Exception as e:
            print('======================')
            traceback.print_exc()
            print('======================')
            r, g, b = errorRgb
    return r, g, b

def rgba_str_to_values(rgbaString, errorRgb=(255,255,255,255)):
    try:
        m = re.findall(r'(\d+), *(\d+), *(\d+),* *(\d+)*', rgbaString)
        r, g, b, a = m[0]
        if a:
            r, g, b, a = int(r), int(g), int(b), int(a)
        else:
            a = 255
            r, g, b = int(r), int(g), int(b)
    except TypeError:
        try:
            r, g, b, a = rgbaString
        except Exception as e:
            r, g, b, a = errorRgb
    return r, g, b, a

def get_lut_from_colors(colors, name='mycmap', N=256, to_uint8=False):
    cmap = LinearSegmentedColormap.from_list(name, colors, N=256)
    lut = np.array([cmap(i)[:3] for i in np.linspace(0,1,256)])
    if to_uint8:
        lut = (lut*255).astype(np.uint8)
    return lut

def invertRGB(self, rgb_img):
    if self.imgCmapName != 'grey':
        return
    # see https://forum.image.sc/t/invert-rgb-image-without-changing-colors/33571
    R = rgb_img[:, :, 0]
    G = rgb_img[:, :, 1]
    B = rgb_img[:, :, 2]
    GB_mean = np.mean([G, B], axis=0)
    RB_mean = np.mean([R, B], axis=0)
    RG_mean = np.mean([R, G], axis=0)
    rgb_img[:, :, 0] = 1-GB_mean
    rgb_img[:, :, 1] = 1-RB_mean
    rgb_img[:, :, 2] = 1-RG_mean
    return rgb_img

def get_greedy_lut(lab, lut, ids=None):    
    expanded = skimage.segmentation.expand_labels(lab, distance=7)
    adj_M = np.zeros([expanded.max() + 1]*2, dtype=bool)
    if ids is None:
        ids = [obj.label for obj in skimage.measure.regionprops(lab)]
    
    # Taken from https://stackoverflow.com/questions/26486898/matrix-of-labels-to-adjacency-matrix
    adj_M[expanded[:, :-1], expanded[:, 1:]] = 1
    adj_M[expanded[:, 1:], expanded[:, :-1]] = 1
    adj_M[expanded[:-1, :], expanded[1:, :]] = 1
    adj_M[expanded[1:, :], expanded[:-1, :]] = 1
    adj_M[expanded[:, 1:], expanded[:, :-1]] = 1
    adj_M[expanded[1:, :], expanded[:-1, :]] = 1
    adj_M[expanded[:-1, :], expanded[1:, :]] = 1
    # adj_M = adj_M[1:, 1:]

    G = nx.from_numpy_array(adj_M)
    color_ids = nx.coloring.greedy_color(
        G, strategy='independent_set', interchange=False
    )
    
    n_foregr_colors = len(lut)-1
    n_colors_greedy = max([color_id for color_id in color_ids.values()])
    color_idxs = {
        id:abs(int(n_foregr_colors * c/n_colors_greedy)-n_foregr_colors)
        for id, c in color_ids.items() if id!=0
    }

    greedy_lut = np.copy(lut)
    greedy_lut[list(color_idxs.keys())] = lut[list(color_idxs.values())]

    return greedy_lut
    