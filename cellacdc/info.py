from . import urls, html_utils

forum_href = html_utils.href_tag('forum page', urls.forum_url)

utilsInfo = {
    'Convert _segm.npz file(s) to ImageJ ROIs...': (f"""
        Not documented yet. You can ask help about utilities  on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Track sub-cellular objects (assign same ID as the cell they belong to)...': (f"""
        Not documented yet. You can ask help about utilities  on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Apply tracking info from tabular data...': (f"""
        This utility is used to <b>load the information of an external tracker</b> into Cell-ACDC.<br><br>
        It creates a new (or overwrites existing) segmentation file where the <b>IDs of the segmented objects are taken from a table</b> in CSV format.<br><br>
        This table is a typical output of trackers. It <b>must contain the tracked IDs of the segmented objects</b>, plus either the<br> 
        corresponding IDs in the segmentation mask or the <code>(x, y)</code> coordinates of the objects' centroids.<br><br>
        The <b>name of the columnns is not relevant</b>, you will be asked to choose which column is what.<br><br>
        Note that to use this utility you <b>need to have a Cell-ACDC compatible segmentation file</b>.
    """),

    'Create required data structure from image files...': (f"""
        Not documented yet. You can ask help about utilities  on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Re-apply data prep steps to selected channels...': (f"""
        Not documented yet. You can ask help about utilities  on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Concatenate acdc output tables from multiple Positions...': (f"""
        Not documented yet. You can ask help about utilities  on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Compute measurements for one or more experiments...': (f"""
        Not documented yet. You can ask help about utilities  on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Combine measurements from multiple segmentation files...': (f"""
        Not documented yet. You can ask help about utilities  on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """),

    'Add lineage tree table to one or more experiments...': (f"""
        Not documented yet. You can ask help about utilities  on our {forum_href}.<br><br>
        Thank you <b>for your patience</b>! 
    """)
}