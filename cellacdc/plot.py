import traceback
import sys
from typing import Union, Iterable, List, Literal

import pandas as pd
import numpy as np
import scipy.stats

import skimage.measure

import matplotlib
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

from tqdm import tqdm

from qtpy import QtGui

from . import printl
from . import widgets, _core, error_below, error_close
from . import _run, core

def matplotlib_cmap_to_lut(
        cmap: Union[Iterable, matplotlib.colors.Colormap, str], 
        n_colors: int=256
    ):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    
    rgbs = [cmap(i) for i in np.linspace(0,1,n_colors)]
    lut = (np.array(rgbs)*255).astype(np.uint8)
    return lut

def imshow(
        *images: Union[np.ndarray, dict], 
        points_coords: np.ndarray=None, 
        points_coords_df: pd.DataFrame=None,
        points_groups: List[str]=None,
        points_data: Union[np.ndarray, pd.DataFrame, pd.Series]=None,
        hide_axes: bool=True, 
        lut: Union[Iterable, matplotlib.colors.Colormap, str]=None, 
        autoLevels: bool=True,
        autoLevelsOnScroll: bool=False,
        block: bool=True,
        showMaximised=False,
        max_ncols=4,
        axis_titles: Union[Iterable, None]=None, 
        parent=None, 
        window_title='Cell-ACDC image viewer',
        color_scheme=None, 
        link_scrollbars=True,
        annotate_labels_idxs: List[int]=None,
        show_duplicated_cursor=True, 
        infer_rgb=True
    ):
    if isinstance(images[0], dict):
        images_dict = images[0]
        images = []
        axis_titles = []
        for title, image in images_dict.items():
            images.append(image)
            axis_titles.append(title)
    if color_scheme is None:
        from ._palettes import get_color_scheme
        color_scheme = get_color_scheme()
    
    if lut is None:
        lut = matplotlib_cmap_to_lut('viridis')

    if isinstance(lut, str):
        lut = matplotlib_cmap_to_lut(lut)

    if isinstance(lut, np.ndarray):
        luts = [lut]*len(images)
    else:
        luts = lut
    
    if luts is not None:
        for l in range(len(luts)):
            if not isinstance(luts[l], str):
                continue
            
            luts[l] = matplotlib_cmap_to_lut(luts[l])
    
    casted_images = []
    for image in images:
        if image.dtype == bool:
            image = image.astype(np.uint8)
        casted_images.append(image)

    app = _run._setup_app()
    win = widgets.ImShow(
        parent=parent, 
        link_scrollbars=link_scrollbars,
        infer_rgb=infer_rgb
    )
    win.setWindowTitle(window_title)
    if app is not None:
        win.app = app
    win.setupMainLayout()
    win.setupStatusBar()
    win.setupGraphicLayout(
        *casted_images, 
        hide_axes=hide_axes, 
        max_ncols=max_ncols,
        color_scheme=color_scheme
    )
    if axis_titles is not None:
        win.setupTitles(*axis_titles)
    win.showImages(
        *casted_images, 
        luts=luts, 
        autoLevels=autoLevels, 
        autoLevelsOnScroll=autoLevelsOnScroll
    )
    if points_coords_df is not None:
        win.drawPointsFromDf(points_coords_df, points_groups=points_groups) 
    if points_coords is not None:
        points_coords = np.round(points_coords).astype(int)
        win.drawPoints(points_coords)
    if points_data is not None:
        win.setPointsData(points_data)
    if show_duplicated_cursor:
        win.setupDuplicatedCursors()
    win.annotateObjectIDs(annotate_labels_idxs=annotate_labels_idxs, init=True)
    win.run(block=block, showMaximised=showMaximised, screenToWindowRatio=0.8)
    return win

def _add_colorbar_axes(
        ax: plt.Axes, im: matplotlib.image.AxesImage, size='5%', pad=0.07,
        label=''
    ):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax)
    if label:
        cbar.set_label(label)

def _raise_non_unique_groups(grouping, dfs, groups_xx):
    groups_with_duplicates = {}
    for d, df in enumerate(dfs):
        if df.index.is_unique:
            continue
        group_xx = groups_xx[d]
        group_with_duplicates = df.columns[0].split(';;')[1].replace('-', ', ')
        duplicated_xx = group_xx[df.index.duplicated(keep='first')]
        groups_with_duplicates[group_with_duplicates] = duplicated_xx
    
    duplicates = []
    for group_values, duplicated_xx in groups_with_duplicates.items():
        xx_name = duplicated_xx.name
        xx_value = duplicated_xx.iloc[0]
        duplicates_str = (
            f'    * Duplicates of "{group_values}" --> {xx_name} = {xx_value}'
        )
        duplicates.append(duplicates_str)

    duplicates = '\n'.join(duplicates)

    traceback.print_exc()
    print(error_below)
    grouping_str = f'{grouping}'.strip('()').strip(',')
    print(f'The groups determined by "{grouping_str}" are not unique:\n')
    print(f'{duplicates}')
    print(error_close)
    exit()

def raise_missing_arg(argument_name):
    traceback.print_exc()
    print(error_below)
    print(f'The argument `{argument_name}` is required.')
    print(error_close)
    exit()

def _get_groups_data(
        df: pd.DataFrame, x: str, z: str, grouping: str, bin_size: int=None, 
        normalize_x: bool=False, zeroize_x: bool=False
    ):
    grouped = df.groupby(list(grouping))
    dfs = []
    groups_xx = []
    yticks_labels = []
    max_n_decimals = None
    min_norm_bin_size = None
    if normalize_x:
        min_dx = min([
            group_df[x].diff().abs().min() for _, group_df in grouped
        ])
        max_n_decimals = 0
        min_norm_bin_size = np.inf

    for name, group_df in grouped:
        groups_xx.append(group_df[x])
        if zeroize_x:
            group_xx = group_df[x]-group_df[x].min()
        else:
            group_xx = group_df[x]
        if len(grouping) == 1:
            name_str = str(name)
        else:
            name_str = '-'.join([str(n) for n in name])
        group_cols = {col:f'{col};;{name_str}' for col in group_df.columns}
        group_df = group_df.rename(columns=group_cols)
        if normalize_x: 
            max_xx = group_xx.max()
            norm_dx = min_dx/max_xx
            min_dx_rounded = _core.round_to_significant(norm_dx, 2)
            n_decimals = len(str(min_dx_rounded).split('.')[1])
            if n_decimals > max_n_decimals:
                max_n_decimals = n_decimals
            norm_xx = (group_xx/max_xx).round(n_decimals)
            norm_xx_perc = norm_xx*100
            if bin_size is not None:
                norm_bin_size = (bin_size/max_xx).round(n_decimals)*100
                if norm_bin_size < min_norm_bin_size:
                    min_norm_bin_size = norm_bin_size
            group_df['x'] = norm_xx_perc
        else:
            group_df['x'] = group_xx
        col_name = f'{z};;{name_str}'
        dfs.append(group_df[[col_name, 'x']].dropna().set_index('x'))
        
        yticks_labels.append(f'{name}'.strip('()'))
    
    try:
        df_data = pd.concat(dfs, names=[x], axis=1).sort_index()
    except pd.errors.InvalidIndexError as err:
        _raise_non_unique_groups(grouping, dfs, groups_xx)

    if min_norm_bin_size is not None:
        bin_size = min_norm_bin_size

    if bin_size is not None:
        order_of_magnitude = 1
        if max_n_decimals is not None:
            # Remove 2 because we work with percentage
            n_decimals = max_n_decimals - 2
            order_of_magnitude = 10**n_decimals
            df_data = df_data.reset_index()
            df_data['x_int'] = (df_data['x']*order_of_magnitude).astype(int)
            df_data = df_data.set_index('x_int').drop(columns='x')
            bin_size = int(bin_size*order_of_magnitude)

        df_data.index = pd.to_datetime(df_data.index.astype(int))
        rs = f'{bin_size}ns'
        df_data = df_data.resample(rs, label='right').mean()
        df_data.index = df_data.index.astype(np.int64)/order_of_magnitude

    data = df_data.fillna(0).values.T
    xx = df_data.index

    return data, xx, yticks_labels

def _check_df_data_args(**kwargs):
    for arg_name, arg_value in kwargs.items():
        if arg_value: 
            continue
        if arg_value is not None:
            continue
        raise_missing_arg(arg_name)

def _raise_group_label_depth_too_deep(group_label_depth, n_levels):
    traceback.print_exc()
    print(error_below)
    print(
        f'The `group_label_depth = {group_label_depth}` is too high, '
        f'there are only {n_levels} levels.'
    )
    print(error_close)
    exit()

def _get_heatmap_yticks(
        nrows, group_height, yticks_labels, group_label_depth
    ):
    yticks = np.arange(0,nrows*group_height, group_height) - 0.5
    # yticks = yticks + group_height/2 - 0.5

    if group_label_depth is not None:
        df_ticks = pd.DataFrame({
            'yticks': yticks,
            'yticks_labels': yticks_labels
        }).set_index('yticks').astype(str)
        df_ticks = df_ticks['yticks_labels'].str.split(',', expand=True)
        if group_label_depth > len(df_ticks.columns):
            n_levels = len(df_ticks.columns)
            _raise_group_label_depth_too_deep(group_label_depth, n_levels)
        df_ticks = df_ticks[list(range(group_label_depth))]
        df_ticks['yticks_labels'] = df_ticks.agg(','.join, axis=1)
        df_ticks = df_ticks.reset_index().set_index('yticks_labels')
        yticks_first = df_ticks[~df_ticks.index.duplicated(keep='first')]
        yticks_last = df_ticks[~df_ticks.index.duplicated(keep='last')]
        yticks_start = yticks_first['yticks']
        yticks_end = yticks_last['yticks']
        yticks_center = yticks_start + (yticks_end-yticks_start)/2
        yticks_center = yticks_center
    return yticks_start, yticks_end, yticks_center

def _raise_convert_time_how(convert_time_how):
    print(error_below)
    conversion_methods = [
        f'    * {how}' for how in _core.time_units_converters.keys()
    ]
    conversion_methods = '\n'.join(conversion_methods)
    print(
        f'"{convert_time_how}" is not a valid `convert_time_how` value.\n'
    )
    print(
        f'Valid methods are:\n\n{conversion_methods}'
    )
    print(error_close)
    exit()

def _get_heatmap_xticks(
        xx, x_unit_width, num_xticks, convert_time_how, 
        num_decimals_xticks_labels, x_label_loc='right',
        add_x_0_label=True, x_labels=None
    ):
    series_xindex = pd.Series(xx).repeat(x_unit_width)
    if x_label_loc == 'right':
        series_xindex.index = series_xindex.index + 1
    elif x_label_loc == 'center':
        series_xindex.index = series_xindex.index + 0.5
    elif x_label_loc == 'left':
        pass
    
    if x_labels is not None:
        series_xticks = (
            series_xindex[series_xindex.isin(x_labels)]
            .drop_duplicates(keep='first')
        )
    else:
        resampling_step = round(len(series_xindex)/(num_xticks))
        series_xticks = series_xindex.iloc[::resampling_step]
    
    xticks = series_xticks.index.to_list()
    xticks_labels = series_xticks.values.astype(int)
    
    if add_x_0_label and xticks[0] != 0:
        xticks = [0, *xticks]
        xticks_labels = np.zeros(len(xticks), dtype=int)
        xticks_labels[1:] = series_xticks

    if convert_time_how is None:
        return xticks, xticks_labels
    
    from_unit, to_unit = convert_time_how.split('->')
    xticks_labels = _core.convert_time_units(xticks_labels, from_unit, to_unit)
    if xticks_labels is None:
        _raise_convert_time_how(convert_time_how)
    
    if num_decimals_xticks_labels is None:
        return xticks, xticks_labels
    
    xticks_labels = xticks_labels.round(num_decimals_xticks_labels)
    
    return xticks, xticks_labels

def _check_x_dtype(df, x, force_x_to_int):
    if force_x_to_int:
        return

    if pd.api.types.is_integer_dtype(df[x]):
        return
    print(error_below)
    print(
        f'The `x` column must be of data type integer. '
        'Pass `force_x_to_int=True` if you want to force conversion to '
        'integers.'
    )
    print(error_close)
    exit()

def heatmap(
        data: Union[pd.DataFrame, np.ndarray], 
        x: str='',  
        z: str='',
        y_grouping: Union[str, List[str]]='',
        sort_groups: bool=True,
        normalize_x: bool=False,
        zeroize_x: bool=False,
        x_bin_size: int=None,
        x_label_loc: str='right',
        x_labels: np.ndarray=None,
        add_x_0_label: bool=False,
        convert_time_how: str=None,
        xlabel: str=None,
        num_decimals_xticks_labels: int=None,
        force_x_to_int: bool=False,
        z_min: Union[int, float]=None,
        z_max: Union[int, float]=None,
        stretch_height_factor: float=None,
        stretch_width_factor: float=None,
        group_label_depth: int=1,
        num_xticks: int=6,
        colormap: Union[str, matplotlib.colors.Colormap]='viridis',
        missing_values_color=None,
        colorbar_pad: float= 0.07,
        colorbar_size: float=0.05,
        colorbar_label: str='',
        ax: plt.Axes=None,
        fig: plt.Figure=None,
        backend: str='matplotlib',
        block: bool=False,
        imshow_kwargs: dict=None
    ):
    """Generate heatmap plot from data

    Parameters
    ----------
    data : Union[pd.DataFrame, np.ndarray]
        Table containing the data in long-format
    x : str, optional
        Name of the column used for the x-axis. Default is ''
    z : str, optional
        Name of the column used for the z-axis, i.e., the values that 
        determine the color of each pixel. Default is ''
    y_grouping : Union[str, List[str]], optional
        Column or list of columns that identifies a single row in the 
        heatmap. Default is ''
    sort_groups : bool, optional
        _description_. Default is True
    normalize_x : bool, optional
        _description_. Default is False
    zeroize_x : bool, optional
        _description_. Default is False
    x_bin_size : int, optional
        _description_. Default is None
    x_label_loc : str, optional
        _description_. Default is 'right'
    x_labels : np.ndarray, optional
        _description_. Default is None
    add_x_0_label : bool, optional
        _description_. Default is False
    convert_time_how : str, optional
        _description_. Default is None
    xlabel : str, optional
        _description_. Default is None
    num_decimals_xticks_labels : int, optional
        _description_. Default is None
    force_x_to_int : bool, optional
        _description_. Default is False
    z_min : Union[int, float], optional
        _description_. Default is None
    z_max : Union[int, float], optional
        _description_. Default is None
    stretch_height_factor : float, optional
        _description_. Default is None
    stretch_width_factor : float, optional
        _description_. Default is None
    group_label_depth : int, optional
        _description_. Default is 1
    num_xticks : int, optional
        _description_. Default is 6
    colormap : Union[str, matplotlib.colors.Colormap], optional
        _description_. Default is 'viridis'
    missing_values_color : _type_, optional
        _description_. Default is None
    colorbar_pad : float, optional
        _description_. Default is 0.07
    colorbar_size : float, optional
        _description_. Default is 0.05
    colorbar_label : str, optional
        _description_. Default is ''
    ax : plt.Axes, optional
        _description_. Default is None
    fig : plt.Figure, optional
        _description_. Default is None
    backend : str, optional
        _description_. Default is 'matplotlib'
    block : bool, optional
        _description_. Default is False
    imshow_kwargs : dict, optional
        _description_. Default is None

    Returns
    -------
    _type_
        _description_
    """    
    
    if ax is None:
        fig, ax = plt.subplots()
    
    if imshow_kwargs is None:
        imshow_kwargs = {}

    yticks_labels = None
    if isinstance(data, pd.DataFrame):
        _check_df_data_args(y_grouping=y_grouping, x=x, z=z)
        _check_x_dtype(data, x, force_x_to_int)
        if isinstance(y_grouping, str):
            y_cols = (y_grouping,)
        else:
            y_cols = y_grouping
        data = data[[*y_cols, x, z]]
        if sort_groups:
            data = data.sort_values(list(y_cols))
        data, xx, yticks_labels = _get_groups_data(
            data, x, z, grouping=y_cols, normalize_x=normalize_x,
            bin_size=x_bin_size, zeroize_x=zeroize_x
        )
    else:
        x = 'x' if not x else x
        y_grouping = 'groups' if not y_grouping else y_grouping
        z = 'x' if not z else z
        xx = np.arange(data.shape[-1])

    if z_min is None:
        z_min = np.nanmin(data)
    
    if z_max is None:
        z_max = np.nanmax(data)

    Y, X = data.shape
    group_height = round(X/Y)
    if stretch_height_factor is not None:
        group_height = round(group_height*stretch_height_factor)
    
    Y, X = data.shape
    x_unit_width = round(Y/X)
    if stretch_width_factor is not None:
        x_unit_width = round(stretch_width_factor)
    
    group_height = group_height if group_height>1 else 1
    x_unit_width = x_unit_width if x_unit_width>1 else 1

    yticks_start, yticks_end, yticks_center = _get_heatmap_yticks(
        len(data), group_height, yticks_labels, group_label_depth
    )
    yticks_labels = yticks_start.index.to_list()
    yticks = yticks_start.values

    xticks, xticks_labels = _get_heatmap_xticks(
        xx, x_unit_width, num_xticks, convert_time_how, 
        num_decimals_xticks_labels, x_label_loc=x_label_loc,
        add_x_0_label=add_x_0_label, x_labels=x_labels
    )

    if group_height > 1:
        data = np.repeat(data, [group_height]*len(data), axis=0)
        
    if x_unit_width > 1:
        ncols = data.shape[-1]
        data = np.repeat(data, [x_unit_width]*ncols, axis=1)
        xticks = [xtick*x_unit_width for xtick in xticks]
    
    if missing_values_color is not None:
        if isinstance(colormap, str):
            colormap = plt.get_cmap(colormap)

        bkgr_color = matplotlib.colors.to_rgba(missing_values_color)
        colors = colormap(np.linspace(0,1,256))
        colors[0] = bkgr_color
        colormap = matplotlib.colors.ListedColormap(colors)

    if xlabel is None:
        xlabel = x

    # Make sure to label the side of the pixel
    xticks = [x-0.5 for x in xticks]

    im = ax.imshow(data, cmap=colormap, vmin=z_min, vmax=z_max, **imshow_kwargs)
    ax.set_xlabel(xlabel)
    ax.set_xticks(xticks, labels=xticks_labels)
    ax.set_ylabel(y_grouping)
    ax.set_yticks(yticks, labels=yticks_labels)
    
    _size_perc = f'{int(colorbar_size*100)}%'
    _add_colorbar_axes(
        ax, im, size=_size_perc, pad=colorbar_pad, label=colorbar_label
    )
    
    if block:
        plt.show()
    else:
        return fig, ax, im

def _binned_mean_stats(x, y, bins, bins_min_count):
    bin_counts, _, _ = scipy.stats.binned_statistic(x, y, statistic='count', bins=bins)
    bin_means, bin_edges, _ = scipy.stats.binned_statistic(x, y, bins=bins)
    bin_std, _, _ = scipy.stats.binned_statistic(x, y, statistic='std', bins=bins)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    if bins_min_count > 1:
        bin_centers = bin_centers[bin_counts > bins_min_count]
        bin_means = bin_means[bin_counts > bins_min_count]
        bin_std = bin_std[bin_counts > bins_min_count]
        bin_counts = bin_counts[bin_counts > bins_min_count]
    std_err = bin_std/np.sqrt(bin_counts)
    return bin_centers, bin_means, bin_std, std_err

def binned_means_plot(
        x: Union[str, Iterable] = None, 
        y: Union[str, Iterable] = None, 
        bins: Union[int, Iterable] = 10, 
        bins_min_count: int = 1,
        data: pd.DataFrame = None,
        ci_plot: Literal['errorbar', 'fill_between']='errorbar',
        scatter: bool = True,
        line_plot = True,
        use_std_err: bool = True,
        color = None,
        label = None,
        scatter_kws = None,
        errorbar_kws = None,
        fill_between_kws = None,
        ax: plt.Axes = None,
        scatter_colors = None
    ):
    if ax is None:
        fig, ax = plt.subplots(1)

    if isinstance(x, str):
        if data is None:
            raise TypeError(
                "Passing strings to `x` and `y` also requires the `data` "
                "variable as a pandas DataFrame"
            )
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        x = data[x]
        y = data[y]

    if color is None:
        color = sns.color_palette(n_colors=1)[0]
    
    if scatter_kws is None:
        scatter_kws = {'alpha': 0.3}
    
    if 'alpha' not in scatter_kws:
        scatter_kws['alpha'] = 0.3
    
    if label is None:
        label = ''
    
    if scatter_colors is None:
        scatter_colors = color
    
    xe, ye, std, std_err = _binned_mean_stats(x, y, bins, bins_min_count)
    if scatter:
        ax.scatter(x, y, color=scatter_colors, **scatter_kws)
    yerr = std_err if use_std_err else std
    
    if ci_plot == 'errorbar':
        if errorbar_kws is None:
            errorbar_kws = {'capsize': 3, 'lw': 2}
        
        if not line_plot:
            fmt = '.'
        else:
            fmt = ''
            
        ax.errorbar(
            xe, ye, yerr=yerr, fmt=fmt, color=color, label=label, **errorbar_kws
        )
    elif ci_plot == 'fill_between':
        if fill_between_kws is None:
            fill_between_kws = {'alpha': 0.3}
        
        if line_plot:
            ax.plot(xe, ye, color=color, label=label)
            label = ''
            
        ax.fill_between(
            xe, ye-yerr, ye+yerr, color=color, label=label, **fill_between_kws
        )
        

    return ax

def text_to_pg_scatter_symbol(text: str, font=None, return_scale=False):
    if font is None:
        font = QtGui.QFont()
        font.setPixelSize(11)

    symbol = QtGui.QPainterPath()
    symbol.addText(0, 0, font, text)
    br = symbol.boundingRect()
    scale = min(1. / br.width(), 1. / br.height())
    tr = QtGui.QTransform()
    tr.scale(scale, scale)
    tr.translate(-br.x() - br.width()/2., -br.y() - br.height()/2.)
    symbol = tr.map(symbol)
    if return_scale:
        return symbol, scale
    else:
        return symbol

def get_symbol_sizes(scales: dict, symbols: dict, size: int):
    scales_arr = np.array([scales[text] for text in symbols.keys()])
    normalized_scales = scales_arr/scales_arr.max()
    sizes = np.round(size/normalized_scales).astype(int)
    sizes = {text:scale for text, scale in zip(symbols.keys(), sizes)}
    return sizes

def texts_to_pg_scatter_symbols(
        texts: Union[str, list], font=None, progress=True,
        return_scales=False
    ):
    if font is None:
        font = QtGui.QFont()
        font.setPixelSize(11)
    if isinstance(texts, str):
        texts = [texts]

    if progress:
        pbar = tqdm(total=len(texts)*2, ncols=100)

    symbols = {}
    scales = {}
    for text in texts:
        symbol = QtGui.QPainterPath()
        symbol.addText(0, 0, font, text)
        br = symbol.boundingRect()
        scale = min(1. / br.width(), 1. / br.height())
        if progress:
            pbar.update()
        tr = QtGui.QTransform()
        tr.scale(scale, scale)
        tr.translate(-br.x() - br.width()*0.5, -br.y() - br.height()*0.5)
        symbols[text] = tr.map(symbol)
        scales[text] = scale
        if progress:
            pbar.update()
    
    if progress:
        pbar.close()
    
    if return_scales:
        return symbols, scales
    else:
        return symbols

def plt_contours(
        ax, lab=None, rp=None, plot_kwargs=None, only_IDs=None, 
        clear_borders=True, obj_contours_kwargs=None
    ):
    if rp is None:
        rp = skimage.measure.regionprops(lab)

    if plot_kwargs is None:
        plot_kwargs = {}
    
    if obj_contours_kwargs is None:
        obj_contours_kwargs = {}
    
    for obj in rp:
        if only_IDs is not None and obj.label not in only_IDs:
            continue
        
        contours = core.get_obj_contours(obj, **obj_contours_kwargs)
        if not isinstance(contours, list):
            contours = [contours]        
        
        for contour in contours:
            xx = contour[:, 0]
            yy = contour[:, 1]
            if clear_borders:
                valid_mask = np.logical_and(xx>0.5, yy>0.5)
                xx = xx[valid_mask]
                yy = yy[valid_mask]
                
            ax.plot(xx, yy, **plot_kwargs)

def plt_moth_bud_lines(
        ax, cca_df, lab=None, rp=None, plot_kwargs=None, 
        only_moth_IDs=None
    ):
    if rp is None:
        rp = skimage.measure.regionprops(lab)

    if plot_kwargs is None:
        plot_kwargs = {}
    
    rp_mapper = {obj.label:obj for obj in rp}
    
    for obj in rp:
        ccs = cca_df.at[obj.label, 'cell_cycle_stage']
        if ccs == 'G1':
            continue
        
        status = cca_df.at[obj.label, 'relationship']
        if status == 'mother':
            continue
        
        mothID = cca_df.at[obj.label, 'relative_ID']
        if only_moth_IDs is not None and mothID not in only_moth_IDs:
            continue
        
        moth_obj = rp_mapper[mothID]
        
        y1, x1 = obj.centroid
        y2, x2 = moth_obj.centroid
        
        ax.plot([x1, x2], [y1, y2], **plot_kwargs)

if __name__ == '__main__':
    x = np.arange(0, 1000).astype(float)
    y = 2*x+10
    noise = np.random.normal(0, 100, size=1000)
    y += noise

    data = pd.DataFrame({'x': x, 'y': y})

    nbins = 10
    bins_min_count = 10

    binned_means_plot(
        x='x', y='y', data=data, nbins=nbins, bins_min_count=bins_min_count
    )
    
    plt.show()
    